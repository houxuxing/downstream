"""
AnatCL 四分类下游任务主入口

重构后的模块结构：
- config.py: 配置类
- model/: 模型定义包（backbone、分类器、冻结工具等）
- dataset.py: 数据集 (使用 MONAI 进行医学影像处理)
- trainer.py: 训练器
- utils.py: 工具函数
"""

import os
import random
from typing import cast

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Subset

# 使用 MONAI 的 DataLoader（针对医学影像优化）
from monai.data.dataloader import DataLoader

from loss.AnatCL_loss import AnatCLGlobalLoss

from model.classifier import AnatCLClassifier
from model.freezing import setup_parameter_freezing

# DINOv2 相关导入
from dinov2_models.vision_transformer import vit_large_3d
from dinov2_utils.utils import load_pretrained_weights

from config import Config
from dataset import MRIDataset
from trainer import (
    EarlyStopping,
    train_epoch,
    validate_epoch,
    create_lr_scheduler,
    check_optimizer_config,
)
from utils import (
    save_plots,
    plot_confusion_matrix,
    split_dataset_by_patient,
    make_balanced_indices,
    print_dataset_info,
    print_test_results,
)


def seed_worker(worker_id: int):
    # 让 numpy/random 与 torch worker seed 对齐
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def DINOv2Backbone(dinov2_model: nn.Module) -> nn.Module:
    """DINOv2 3D Vision Transformer 包装器

    将 DINOv2 的 3D ViT 适配为分类任务的 backbone。
    通过 forward_features 提取 CLS token 作为特征输出。
    """
    
    class _Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            # DINOv2 ViT-Large 的 embed_dim 为 1024
            self.dim_in = 1024

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: 输入图像张量，形状 (B, C, D, H, W)

            Returns:
                特征向量，形状 (B, dim_in)
            """
            B = x.shape[0]
            
            # forward_features 返回字典，包含:
            # - "x_norm_clstoken": (B, embed_dim) - CLS token
            # - "x_norm_patchtokens": (B, N, embed_dim) - patch tokens
            features = self.model.forward_features(x)
            
            # 使用 CLS token 作为全局特征表示
            cls_token = features["x_norm_clstoken"]
            
            # 【关键断言】确保输出形状正确，防止维度错误
            assert cls_token.dim() == 2, (
                f"CLS token 应为 2D [B, C], 实际为 {cls_token.dim()}D, shape={cls_token.shape}"
            )
            assert cls_token.shape[0] == B, (
                f"CLS token batch 维度错误: 期望 {B}, 实际 {cls_token.shape[0]}"
            )
            
            return cls_token

    return _Wrapper(dinov2_model)


def create_dinov2_backbone(cfg: Config, device: torch.device) -> nn.Module:
    """创建 DINOv2 3D Vision Transformer backbone

    使用 vit_base_3d 创建 3D ViT 模型，并加载预训练权重。
    DINOv2 是一个强大的自监督视觉模型，其 3D 变体适用于医学影像。
    """
    # 获取输入尺寸（假设是立方体）
    img_size = cfg.target_size[0]

    # 创建 3D ViT 模型 (Large 版本，匹配预训练权重 embed_dim=1024)
    model = vit_large_3d(
        img_size=img_size,
        patch_size=16,
        in_chans=1,  # 医学影像通常为单通道
        block_chunks=4,
        init_values=1e-5,
    )

    # 加载预训练权重（如果指定）
    if cfg.dinov2_ckpt is not None:
        print(f"[INFO] 加载 DINOv2 预训练权重: {cfg.dinov2_ckpt}")
        load_pretrained_weights(model, cfg.dinov2_ckpt, checkpoint_key="teacher")

    # 创建包装器
    backbone = DINOv2Backbone(model).to(device)

    print(
        f"\n[INFO] 使用 DINOv2 Backbone (ViT-Large-3D)，输出特征维度: {backbone.dim_in}"
    )
    return backbone


def _apply_kaiming_init(module: nn.Module) -> None:
    """Apply Kaiming init to a module.
    
    WARNING: 仅用于初始化新增模块（如 classifier, projector）。
    严禁对 backbone、patch_embed、blocks、norm、pos_embed、cls_token 调用此函数，
    否则会覆盖预训练权重，导致特征坍缩。
    """

    def init_kaiming_all(m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    module.apply(init_kaiming_all)


def create_model(cfg: Config, device: torch.device) -> AnatCLClassifier:
    """创建并初始化模型（主入口函数）

    目前仅支持 DINOv2 作为 backbone。
    """
    # 创建 DINOv2 backbone
    backbone = create_dinov2_backbone(cfg, device)

    # 创建分类器（线性层在 AnatCLClassifier 中定义）
    model = AnatCLClassifier(
        backbone, 
        num_classes=cfg.num_classes, 
        embed_dim=cfg.embed_dim,
        use_projector=cfg.use_projector,
        projector_dim=cfg.projector_dim,
    ).to(device)

    # Kaiming 初始化
    if cfg.init_mode == "kaiming":
        print("[INFO] 使用 Kaiming 初始化")
        _apply_kaiming_init(model.classifier)

    return model


def create_dataloaders(
    cfg: Config,
    is_main: bool = True,
) -> tuple:
    """创建数据加载器"""
    full_dataset = MRIDataset(
        cfg.data_dir,
        cfg.csv_file_processed,
        target_size=cfg.target_size,
        is_train=True,
        age_csv=cfg.age_csv if cfg.use_global_loss else None,
        measure_root=cfg.measure_root if cfg.use_global_loss else None,
        region_order_json=cfg.region_order_json if cfg.use_global_loss else None,
        enable_memory_efficient=cfg.enable_memory_efficient_mode,
        two_view=cfg.two_view,
    )

    # 患者级拆分
    train_idx, val_idx, test_idx, train_ptids, val_ptids, test_ptids = (
        split_dataset_by_patient(full_dataset)
    )

    # 均衡采样
    balanced_train_idx = make_balanced_indices(train_idx, full_dataset, seed=42)
    balanced_val_idx = make_balanced_indices(val_idx, full_dataset, seed=2025)
    balanced_test_idx = make_balanced_indices(test_idx, full_dataset, seed=3407)

    # 设置训练索引（启用数据增强）
    full_dataset.set_train_idx(balanced_train_idx)

    # 创建子集
    train_set = Subset(full_dataset, balanced_train_idx)
    val_set = Subset(full_dataset, balanced_val_idx)
    test_set = Subset(full_dataset, balanced_test_idx)

    # 创建 MONAI DataLoader
    # MONAI DataLoader 继承自 PyTorch DataLoader，针对医学影像进行了优化
    g = torch.Generator()
    g.manual_seed(42)  # 固定随机种子

    common_kwargs: dict = dict(
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
        generator=g,
    )

    if cfg.num_workers > 0:
        common_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        **common_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        **common_kwargs,
    )

    if is_main:
        print_dataset_info(
            full_dataset,
            train_ptids,
            val_ptids,
            test_ptids,
            train_idx,
            val_idx,
            test_idx,
            balanced_train_idx,
            balanced_val_idx,
            balanced_test_idx,
        )
        print(
            f"\n[INFO] 使用平衡后数据集：训练影像数: {len(train_set)}，验证影像数: {len(val_set)}，测试影像数: {len(test_set)}"
        )

    return train_loader, val_loader, test_loader


def create_optimizer(model, cfg: Config) -> torch.optim.Optimizer:
    """创建优化器
    
    参数组分为三部分：
    1. 分类头参数：使用较高的学习率 (head_lr)
    2. 投影层参数：使用中等学习率 (head_lr * 0.5)
    3. backbone 参数：使用较低的学习率 (backbone_lr)
    """
    # 分类头参数
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    
    # 投影层参数（如果存在）
    projector_params = []
    if hasattr(model, 'feature_projector') and model.feature_projector is not None:
        projector_params = [p for p in model.feature_projector.parameters() if p.requires_grad]
    
    # backbone 参数
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    param_groups = [{"params": head_params, "lr": cfg.head_lr, "name": "head"}]
    
    if projector_params:
        # 投影层使用稍低的学习率
        param_groups.append({"params": projector_params, "lr": cfg.head_lr * 0.5, "name": "projector"})
    
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.backbone_lr, "name": "backbone"})

    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


@torch.no_grad()
def diagnose_loaded_collapse(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_main: bool = True,
) -> None:
    """Run a single-batch diagnostic to detect collapse right after loading weights."""
    if not is_main:
        return

    model.eval()
    try:
        batch_data = next(iter(loader))
    except StopIteration:
        print("[WARN] Diagnostic skipped: loader is empty.")
        return

    data, target, sample_id, age, measures = batch_data

    # If a two-view batch slips in, keep the first view only.
    if data.ndim == 6 and data.size(1) == 2:
        data = data[:, 0]

    data = data.to(device, non_blocking=True)

    backbone_features = model.backbone(data)
    logits = model(data, return_features=False)

    print("\n" + "=" * 60)
    print("[DIAG] Loaded-weight collapse check (single batch)")
    print("=" * 60)
    print(f"batch: {data.shape}, dtype={data.dtype}")

    # Cosine similarity across samples in the backbone feature space.
    if backbone_features.shape[0] > 1:
        feat_norm = backbone_features / (
            backbone_features.norm(dim=1, keepdim=True) + 1e-8
        )
        cos_sim = feat_norm @ feat_norm.t()
        mask = ~torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        off_diag = cos_sim[mask]
        cos_mean = off_diag.mean().item()
        cos_min = off_diag.min().item()
        cos_max = off_diag.max().item()
        print(
            f"backbone cosine(sim) off-diag: mean={cos_mean:.6f}, min={cos_min:.6f}, max={cos_max:.6f}"
        )
    else:
        print("backbone cosine(sim) off-diag: N/A (batch_size < 2)")

    # Logits variance (per-sample and across samples).
    if logits.shape[0] > 0:
        logits_std_per_sample = logits.std(dim=1).mean().item()
        if logits.shape[0] > 1:
            logits_std_across = logits.std(dim=0).tolist()
        else:
            logits_std_across = [float("nan")] * logits.shape[1]
        print(f"logits std per-sample (mean): {logits_std_per_sample:.6f}")
        print(f"logits std across samples: {logits_std_across}")
    else:
        print("logits std: N/A (empty batch)")

    print("=" * 60 + "\n")


@torch.no_grad()
def quick_inference_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_main: bool = True,
    max_samples: int = 10,
) -> None:
    """Run a small inference pass before training starts."""
    if not is_main:
        return

    model.eval()
    try:
        batch_data = next(iter(loader))
    except StopIteration:
        print("[WARN] Quick inference skipped: loader is empty.")
        return

    data, target, sample_id, age, measures = batch_data

    # If a two-view batch slips in, keep the first view only.
    if data.ndim == 6 and data.size(1) == 2:
        data = data[:, 0]

    data = data.to(device, non_blocking=True)

    logits = model(data, return_features=False)
    preds = logits.argmax(dim=1).detach().cpu().tolist()
    targets = target.detach().cpu().tolist()

    print("\n" + "=" * 60)
    print("[DIAG] Quick inference preview (single batch)")
    print("=" * 60)
    for i in range(min(max_samples, logits.shape[0])):
        logit_str = ", ".join([f"{v:.4f}" for v in logits[i].detach().cpu().tolist()])
        sid = sample_id[i] if isinstance(sample_id, list) else str(sample_id)
        print(f"  {sid} | pred={preds[i]} true={targets[i]} | logits=[{logit_str}]")
    print("=" * 60 + "\n")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    is_main: bool = True,
):
    """训练循环"""
    # 使用 Label Smoothing 减少过拟合，帮助模型在相似特征上更好地学习
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = create_optimizer(model, cfg)
    scheduler = create_lr_scheduler(optimizer, cfg.num_epochs, cfg.warmup_epochs)

    # 【诊断】检查优化器配置
    if is_main:
        check_optimizer_config(optimizer, model)

    # AMP
    amp_enabled = device.type == "cuda"
    GradScaler = getattr(torch.amp, "GradScaler", None)
    scaler = (
        GradScaler("cuda", enabled=amp_enabled)
        if GradScaler is not None
        else torch.cuda.amp.GradScaler(enabled=amp_enabled)
    )

    # Global Loss
    global_loss_fn = None
    if cfg.use_global_loss:
        global_loss_fn = AnatCLGlobalLoss(cfg)
        print(f"[INFO] 启用 Global Loss，权重 lambda_global = {cfg.lambda_global}")
    else:
        print("[INFO] 未启用 Global Loss")

    # 早停
    os.makedirs(cfg.model_save_dir, exist_ok=True)
    early_stopping = EarlyStopping(
        patience=cfg.patience,
        save_path=os.path.join(cfg.model_save_dir, cfg.model_save_name),
        is_main=is_main,
    )

    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, cfg.num_epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            max_grad_norm=cfg.max_grad_norm,
            global_loss_fn=global_loss_fn,
            lambda_global=cfg.lambda_global,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            enable_memory_efficient=cfg.enable_memory_efficient_mode,
        )
        
        # 修复 UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`
        # 应该在 train_epoch (包含 optimizer.step) 之后调用 scheduler.step()
        scheduler.step()
        
        va_loss, va_ce_loss, va_acc, val_preds, val_targets, _, _ = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            is_main,
            enable_memory_efficient=cfg.enable_memory_efficient_mode,
            global_loss_fn=global_loss_fn,
            lambda_global=cfg.lambda_global,
            debug_first_batch=(epoch == 1),  # 第一个 epoch 启用调试
        )

        # 学习率调度：在 optimizer.step() 之后调用，不传 epoch 参数
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(va_loss)
        val_accs.append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc

        if is_main:
            plot_confusion_matrix(
                val_targets, val_preds, cfg.classes, epoch, prefix="val"
            )
            print(
                f"Epoch {epoch:03d} | "
                f"Train loss: {tr_loss:.4f} | Train acc: {tr_acc:.2f}% | "
                f"Val loss: {va_loss:.4f} (CE: {va_ce_loss:.4f}) | Val acc: {va_acc:.2f}%"
            )

        # 使用 CE Loss 进行早停监测（关注分类性能）
        early_stopping(va_ce_loss, model)
        if early_stopping.early_stop:
            print("触发早停机制，停止训练")
            break

    if is_main:
        print(f"[INFO] 训练结束，最佳验证准确率: {best_val_acc:.2f}%")
        save_plots(train_losses, val_losses, train_accs, val_accs, cfg.png_save_dir, cfg)

    return early_stopping.save_path


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    is_main: bool = True,
):
    """评估测试集"""
    criterion = nn.CrossEntropyLoss()

    print("\n===== 测试集评估 =====")
    test_loss, test_ce_loss, test_acc, test_preds, test_targets, test_ids, test_logits = (
        validate_epoch(
            model,
            test_loader,
            criterion,
            device,
            is_main,
            enable_memory_efficient=cfg.enable_memory_efficient_mode,
            # 测试时不使用 global_loss，只用 CE 损失
        )
    )

    print(f"测试集准确率: {test_acc:.2f}%，测试集损失: {test_loss:.4f} (CE: {test_ce_loss:.4f})")
    plot_confusion_matrix(test_targets, test_preds, cfg.classes, "最终", prefix="test")
    print_test_results(test_targets, test_preds, test_ids, test_logits, cfg.classes)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_main = True

    # 配置（使用默认路径，或按需覆盖）
    cfg = Config()

    # 创建模型
    model = create_model(cfg, device)
    setup_parameter_freezing(
        model, 
        cfg.train_mode, 
        is_main,
        unfreeze_last_n_blocks=cfg.unfreeze_last_n_blocks,
    )

    # 参数统计
    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"\n[INFO] 可训练参数: {trainable:,} / 总参数: {total:,}")
        print(f"[INFO] 可训练比例: {100.0 * trainable / total:.2f}%")

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(cfg, is_main)

    # Diagnostic: check collapse right after loading weights (before training).
    diagnose_loaded_collapse(model, val_loader, device, is_main=is_main)

    # Quick inference test before training.
    quick_inference_test(model, val_loader, device, is_main=is_main)

    # 训练
    best_model_path = train(model, train_loader, val_loader, cfg, device, is_main)

    # 加载最佳模型并评估
    if is_main:
        model.load_state_dict(torch.load(best_model_path))
        evaluate(model, test_loader, cfg, device, is_main)


if __name__ == "__main__":
    main()
