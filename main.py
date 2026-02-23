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

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Subset

# 使用 MONAI 的 DataLoader（针对医学影像优化）
from monai.data.dataloader import DataLoader

from loss.AnatCL_loss import AnatCLGlobalLoss

from model.classifier import AnatCLClassifier, DualBranchClassifier
from model.freezing import setup_parameter_freezing, setup_dual_branch_freezing

from model_builder import create_model, create_dual_branch_model

from config import Config
from dataset import MRIDataset
from trainer import (
    EarlyStopping,
    train_epoch,
    validate_epoch,
    create_lr_scheduler,
    schedule_lambda_global,
)
from utils import (
    save_plots,
    plot_confusion_matrix,
    split_dataset_by_patient,
    make_balanced_indices,
    print_dataset_info,
    print_test_results,
    compute_metrics,
    build_weighted_sampler,
)


def seed_worker(worker_id: int):
    # 让 numpy/random 与 torch worker seed 对齐
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_global_seed(cfg: Config, is_main: bool = True) -> None:
    """设置全局随机种子，确保实验可复现"""
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(cfg.cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)

    if cfg.deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            if is_main:
                print(f"[WARNING] 无法使用确定性算法: {type(e).__name__}: {e}")

    if is_main:
        print(
            f"[SEED] seed={seed} deterministic={cfg.deterministic} "
            f"cudnn_deterministic={cfg.cudnn_deterministic} cudnn_benchmark={cfg.cudnn_benchmark}"
        )


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
        min_int=cfg.min_int,
        resize_scale=cfg.resize_scale,
        use_nnunet_zscore=cfg.use_nnunet_zscore,
    )

    # 患者级拆分
    train_idx, val_idx, test_idx, train_ptids, val_ptids, test_ptids = (
        split_dataset_by_patient(full_dataset)
    )

    # ========================
    # 实际使用的索引（默认训练可均衡；验证/测试默认全量）
    # ========================
    if cfg.balance_strategy == "downsample_to_pmci":
        train_idx_used = make_balanced_indices(
            train_idx,
            full_dataset,
            seed=int(cfg.seed),
            expected_classes=list(cfg.classes),
            base_class="pMCI",
        )
        val_idx_used = make_balanced_indices(
            val_idx,
            full_dataset,
            seed=2025,
            expected_classes=list(cfg.classes),
            base_class="pMCI",
        )
        test_idx_used = make_balanced_indices(
            test_idx,
            full_dataset,
            seed=3407,
            expected_classes=list(cfg.classes),
            base_class="pMCI",
        )
    else:
        if cfg.balance_strategy == "downsample":
            train_idx_used = make_balanced_indices(
                train_idx,
                full_dataset,
                seed=int(cfg.seed),
                expected_classes=list(cfg.classes),
            )
        else:
            train_idx_used = train_idx  # 全量训练集，不丢数据

        if cfg.balance_eval_downsample:
            val_idx_used = make_balanced_indices(
                val_idx, full_dataset, seed=2025, expected_classes=list(cfg.classes)
            )
            test_idx_used = make_balanced_indices(
                test_idx, full_dataset, seed=3407, expected_classes=list(cfg.classes)
            )
        else:
            val_idx_used = val_idx
            test_idx_used = test_idx

    # 设置训练索引（仅训练集启用数据增强）
    full_dataset.set_train_idx(train_idx_used)

    # 创建子集
    train_set = Subset(full_dataset, train_idx_used)
    val_set = Subset(full_dataset, val_idx_used)
    test_set = Subset(full_dataset, test_idx_used)

    # 创建 MONAI DataLoader
    # MONAI DataLoader 继承自 PyTorch DataLoader，针对医学影像进行了优化
    g = torch.Generator()
    g.manual_seed(int(cfg.seed))  # 固定随机种子

    train_kwargs: dict = dict(
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
        generator=g,
    )

    if cfg.num_workers > 0:
        train_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))

    val_num_workers = int(getattr(cfg, "num_workers_val", 0))
    val_kwargs: dict = dict(
        num_workers=val_num_workers,
        pin_memory=cfg.pin_memory,
        worker_init_fn=seed_worker if val_num_workers > 0 else None,
    )

    train_sampler_info = None
    if cfg.balance_strategy == "weighted_sampler":
        sampler, train_sampler_info = build_weighted_sampler(
            train_set, full_dataset, cfg, return_info=True
        )
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=sampler,
            **train_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            **train_kwargs,
        )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        **val_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        **val_kwargs,
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
            train_idx_used,
            val_idx_used,
            test_idx_used,
            cfg=cfg,
            sampler_info=train_sampler_info,
        )
        print(
            f"\n[INFO] 实际使用数据集：训练影像数: {len(train_set)}，验证影像数: {len(val_set)}，测试影像数: {len(test_set)}"
        )

    return train_loader, val_loader, test_loader


def compute_scaled_lr(
    base_lr: float, batch_size: int, base_batch_size: int = 1024
) -> float:
    """基于 Batch Size 的平方根缩放规则计算实际学习率

    公式: lr = base_lr * sqrt(batch_size / base_batch_size)

    Args:
        base_lr: 基准学习率（对应 base_batch_size）
        batch_size: 实际使用的总 batch size（包含梯度累积）
        base_batch_size: 基准 batch size（默认 1024）

    Returns:
        缩放后的学习率
    """
    import math

    scale_factor = math.sqrt(batch_size / base_batch_size)
    return base_lr * scale_factor


def create_optimizer(model, cfg: Config) -> torch.optim.Optimizer:
    """创建优化器，支持单分支和双分支模型"""
    total_batch_size = cfg.batch_size * cfg.gradient_accumulation_steps

    head_lr = compute_scaled_lr(cfg.base_head_lr, total_batch_size, cfg.base_batch_size)
    backbone_lr = compute_scaled_lr(
        cfg.base_backbone_lr, total_batch_size, cfg.base_batch_size
    )

    print()
    print(
        f"[INFO] LR 缩放: Total_BS={total_batch_size}, Head_LR={head_lr:.2e}, Backbone_LR={backbone_lr:.2e}"
    )

    # 判断是否为双分支模型
    is_dual_branch = isinstance(model, DualBranchClassifier)

    norm_types = (
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.GroupNorm,
    )

    def collect_norm_param_ids(module: nn.Module):
        norm_ids = set()
        for m in module.modules():
            if isinstance(m, norm_types):
                for p in m.parameters(recurse=False):
                    norm_ids.add(id(p))
        return norm_ids

    def is_no_decay(name: str, param: torch.nn.Parameter, norm_ids) -> bool:
        if id(param) in norm_ids:
            return True
        if name.endswith(".bias"):
            return True
        if ("pos_embed" in name) or ("cls_token" in name) or ("mask_token" in name):
            return True
        return False

    param_groups = []

    # Head: split decay/no_decay (classifier)
    # 注意：双分支模型的 aggregator 在融合层中处理，单分支模型的 aggregator 在这里处理
    head_modules = [("classifier", model.classifier)]
    if not is_dual_branch and hasattr(model, "aggregator") and model.aggregator is not None:
        head_modules.append(("aggregator", model.aggregator))

    head_norm_ids = set()
    for _, m in head_modules:
        head_norm_ids |= collect_norm_param_ids(m)

    head_decay, head_no_decay = [], []
    for prefix, m in head_modules:
        for name, param in m.named_parameters():
            if not param.requires_grad:
                continue
            full_name = f"{prefix}.{name}"
            if is_no_decay(full_name, param, head_norm_ids):
                head_no_decay.append(param)
            else:
                head_decay.append(param)

    if head_decay:
        param_groups.append(
            {
                "params": head_decay,
                "lr": head_lr,
                "weight_decay": cfg.weight_decay_init,
                "wd_scale": 1.0,
                "name": "head_decay",
            }
        )
    if head_no_decay:
        param_groups.append(
            {
                "params": head_no_decay,
                "lr": head_lr,
                "weight_decay": 0.0,
                "wd_scale": 0.0,
                "name": "head_no_decay",
            }
        )

    if head_decay or head_no_decay:
        print(
            f"  - Head(+Agg): decay={sum(p.numel() for p in head_decay):,}, "
            f"no_decay={sum(p.numel() for p in head_no_decay):,} params, lr={head_lr:.6e}"
        )

    # Backbone: split decay/no_decay
    # 支持双分支模型：使用 backbone1 处理第一个分支
    backbone_to_process = model.backbone1 if is_dual_branch else model.backbone
    
    backbone_norm_ids = collect_norm_param_ids(backbone_to_process)

    backbone_decay, backbone_no_decay = [], []
    for name, param in backbone_to_process.named_parameters():
        if not param.requires_grad:
            continue
        if is_no_decay(name, param, backbone_norm_ids):
            backbone_no_decay.append(param)
        else:
            backbone_decay.append(param)

    if backbone_decay:
        param_groups.append(
            {
                "params": backbone_decay,
                "lr": backbone_lr,
                "weight_decay": cfg.weight_decay_init * cfg.backbone_wd_scale,
                "wd_scale": cfg.backbone_wd_scale,
                "name": "backbone_decay",
            }
        )
    if backbone_no_decay:
        param_groups.append(
            {
                "params": backbone_no_decay,
                "lr": backbone_lr,
                "weight_decay": 0.0,
                "wd_scale": 0.0,
                "name": "backbone_no_decay",
            }
        )

    if backbone_decay or backbone_no_decay:
        print(
            f"  - Backbone: decay={sum(p.numel() for p in backbone_decay):,}, no_decay={sum(p.numel() for p in backbone_no_decay):,} params, lr={backbone_lr:.6e}"
        )

    # 双分支模型：处理第二个 backbone
    if is_dual_branch:
        backbone2_norm_ids = collect_norm_param_ids(model.backbone2)

        backbone2_decay, backbone2_no_decay = [], []
        for name, param in model.backbone2.named_parameters():
            if not param.requires_grad:
                continue
            if is_no_decay(name, param, backbone2_norm_ids):
                backbone2_no_decay.append(param)
            else:
                backbone2_decay.append(param)

        if backbone2_decay:
            param_groups.append(
                {
                    "params": backbone2_decay,
                    "lr": backbone_lr,
                    "weight_decay": cfg.weight_decay_init * cfg.backbone_wd_scale,
                    "wd_scale": cfg.backbone_wd_scale,
                    "name": "backbone2_decay",
                }
            )
        if backbone2_no_decay:
            param_groups.append(
                {
                    "params": backbone2_no_decay,
                    "lr": backbone_lr,
                    "weight_decay": 0.0,
                    "wd_scale": 0.0,
                    "name": "backbone2_no_decay",
                }
            )

        if backbone2_decay or backbone2_no_decay:
            print(
                f"  - Backbone2: decay={sum(p.numel() for p in backbone2_decay):,}, no_decay={sum(p.numel() for p in backbone2_no_decay):,} params, lr={backbone_lr:.6e}"
            )

        # 双分支模型：处理融合层参数 (proj1, proj2, ca1, ca2, aggregator)
        fusion_modules = []
        for name in ["proj1", "proj2", "ca1", "ca2", "aggregator"]:
            if hasattr(model, name):
                module = getattr(model, name)
                if module is not None:
                    fusion_modules.append((name, module))

        if fusion_modules:
            fusion_norm_ids = set()
            for _, m in fusion_modules:
                fusion_norm_ids |= collect_norm_param_ids(m)

            fusion_decay, fusion_no_decay = [], []
            for prefix, m in fusion_modules:
                for name, param in m.named_parameters():
                    if not param.requires_grad:
                        continue
                    full_name = f"{prefix}.{name}"
                    if is_no_decay(full_name, param, fusion_norm_ids):
                        fusion_no_decay.append(param)
                    else:
                        fusion_decay.append(param)

            if fusion_decay:
                param_groups.append(
                    {
                        "params": fusion_decay,
                        "lr": head_lr,
                        "weight_decay": cfg.weight_decay_init,
                        "wd_scale": 1.0,
                        "name": "fusion_decay",
                    }
                )
            if fusion_no_decay:
                param_groups.append(
                    {
                        "params": fusion_no_decay,
                        "lr": head_lr,
                        "weight_decay": 0.0,
                        "wd_scale": 0.0,
                        "name": "fusion_no_decay",
                    }
                )

            if fusion_decay or fusion_no_decay:
                print(
                    f"  - Fusion Layers: decay={sum(p.numel() for p in fusion_decay):,}, "
                    f"no_decay={sum(p.numel() for p in fusion_no_decay):,} params, lr={head_lr:.6e}"
                )

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(cfg.adamw_beta1, cfg.adamw_beta2),
        weight_decay=cfg.weight_decay_init,
    )

    # Sanity check: optimizer covers all trainable params
    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    missing = [
        n
        for n, p in model.named_parameters()
        if p.requires_grad and id(p) not in opt_param_ids
    ]
    if missing:
        raise RuntimeError(
            "Optimizer does NOT include all trainable parameters! Missing:\n"
            + "\n".join(missing[:50])
            + (f"\n... ({len(missing)} total)" if len(missing) > 50 else "")
        )

    print()
    print("[INFO] AdamW 优化器:")
    print(f"  - Betas: ({cfg.adamw_beta1}, {cfg.adamw_beta2})")
    print(
        f"  - Weight Decay (init -> end): {cfg.weight_decay_init} -> {cfg.weight_decay_end}"
    )
    print(f"  - Backbone WD Scale: {cfg.backbone_wd_scale}")
    print(f"  - Grad Clip: {cfg.max_grad_norm}")

    return optimizer


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

    data = data.to(device, non_blocking=True)

    # 判断是否为双分支模型
    is_dual_branch = isinstance(model, DualBranchClassifier)

    with torch.no_grad():
        if is_dual_branch:
            # 双分支模型：分别获取两个分支的特征
            extracted_features1 = model.backbone1(data)
            extracted_features2 = model.backbone2(data)
            print("\n" + "=" * 60)
            print("[DIAG] Loaded-weight collapse check (dual branch, single batch)")
            print("=" * 60)
            print(f"batch: {data.shape}, dtype={data.dtype}")
            print(f"backbone1 output: {extracted_features1.shape}")
            print(f"backbone2 output: {extracted_features2.shape}")
        else:
            # 单分支模型
            assert isinstance(model, AnatCLClassifier), (
                f"Model should be AnatCLClassifier, got {type(model)}"
            )
            extracted_features = model.backbone(data)
            print("\n" + "=" * 60)
            print("[DIAG] Loaded-weight collapse check (single batch)")
            print("=" * 60)
            print(f"batch: {data.shape}, dtype={data.dtype}")
            print(f"backbone output: {extracted_features.shape}")

            if hasattr(model, "aggregator") and model.aggregator is not None:
                aggregated = model.aggregator(extracted_features)
                print(f"aggregator output: {aggregated.shape}")

                # Check for collapse: all-same across batch dimension
                std_per_feature = aggregated.std(dim=0)
                n_dead = (std_per_feature < 1e-6).sum().item()
                print(
                    f"aggregator output std per feature: min={std_per_feature.min():.6f}, "
                    f"max={std_per_feature.max():.6f}, mean={std_per_feature.mean():.6f}"
                )
                print(f"Dead features (std < 1e-6): {n_dead} / {aggregated.shape[1]}")

                if n_dead == aggregated.shape[1]:
                    print("[WARNING] 检测到特征坍缩！所有特征的标准差都接近于0。")
                elif n_dead > aggregated.shape[1] * 0.5:
                    print(f"[WARNING] 检测到部分特征坍缩！{n_dead}/{aggregated.shape[1]} 特征死亡。")
                else:
                    print("[INFO] 未发现明显的特征坍缩。")

    print("=" * 60 + "\n")


def main():
    cfg = Config()

    # 设置随机种子
    set_global_seed(cfg, is_main=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(cfg, is_main=True)

    # 创建模型
    if cfg.use_dual_branch:
        model = create_dual_branch_model(cfg, device)
        setup_dual_branch_freezing(
            model,
            train_mode1=cfg.train_mode1,
            train_mode2=cfg.train_mode2,
            unfreeze_last_n_blocks1=cfg.unfreeze_last_n_blocks1,
            unfreeze_last_n_blocks2=cfg.unfreeze_last_n_blocks2,
            use_dora_branch2=cfg.use_dora_branch2,
            dora_branch2_r=cfg.dora_branch2_r,
            dora_branch2_alpha=cfg.dora_branch2_alpha,
            dora_branch2_target_modules=cfg.dora_branch2_target_modules,
            is_main=True,
        )
    else:
        model = create_model(cfg, device)
        setup_parameter_freezing(
            model,
            train_mode=cfg.train_mode,
            is_main=True,
            unfreeze_last_n_blocks=cfg.unfreeze_last_n_blocks,
        )

    # 诊断加载后的模型
    diagnose_loaded_collapse(model, train_loader, device, is_main=True)

    # 创建优化器
    optimizer = create_optimizer(model, cfg)

    # 创建学习率调度器
    scheduler = create_lr_scheduler(
        optimizer,
        cfg.num_epochs,
        cfg.warmup_epochs,
        min_lr=cfg.min_lr,
        wd_init=cfg.weight_decay_init,
        wd_end=cfg.weight_decay_end,
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    if cfg.balance_strategy == "class_weight":
        # 基于训练集（实际使用的索引）计算 class weights，避免丢数据但缓解类别不均衡
        train_subset = train_loader.dataset  # Subset(full_dataset, train_idx_used)
        base_ds = getattr(train_subset, "dataset", None)
        indices = getattr(train_subset, "indices", None)
        if base_ds is None or indices is None:
            raise RuntimeError("class_weight 需要 train_loader.dataset 为 Subset，并包含 dataset/indices")

        counts = np.zeros(int(cfg.num_classes), dtype=np.int64)
        for idx in indices:
            dx = base_ds.valid_samples[idx]["DX_group"]
            cls_id = int(base_ds.LABEL_MAP[dx])
            counts[cls_id] += 1

        total = int(counts.sum())
        weights = np.zeros(int(cfg.num_classes), dtype=np.float32)
        for c in range(int(cfg.num_classes)):
            if counts[c] > 0:
                # 使平均权重约为 1，便于训练稳定：w_c = total / (C * count_c)
                weights[c] = float(total) / float(int(cfg.num_classes) * int(counts[c]))

        weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print(
            f"[BALANCE] class_weight enabled: counts={counts.tolist()}, weights={weights.tolist()}"
        )
    global_loss_fn = None
    if cfg.use_global_loss:
        global_loss_fn = AnatCLGlobalLoss(cfg)
        print(
            f"[INFO] 启用 Global Loss: lambda_global={cfg.lambda_global}, "
            f"warmup_epochs={getattr(cfg, 'lambda_global_warmup_epochs', 0)}"
        )

    print(f"[INFO] 模型选择指标: {cfg.model_selection_metric}")

    # 早停
    os.makedirs(cfg.model_save_dir, exist_ok=True)
    save_path = os.path.join(cfg.model_save_dir, cfg.model_save_name or "model.pth")
    early_stopping = EarlyStopping(
        patience=cfg.patience,
        save_path=save_path,
        is_main=True,
        min_delta=1e-4,
    )

    # 训练循环
    train_losses, val_losses = [], []
    train_ce_losses, val_ce_losses = [], []
    train_global_losses, val_global_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(cfg.num_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.num_epochs}")
        print("-" * 40)

        # 修复 warmup/scheduler off-by-one：在每个 epoch 开始 step，使 epoch0 即生效
        if scheduler is not None:
            scheduler.step(epoch)

        # Global Loss 权重 warmup（用于 total loss，但选模/早停默认用 Val CE 对齐分类目标）
        lambda_global_epoch = schedule_lambda_global(epoch, cfg)
        if cfg.use_global_loss:
            print(
                f"[INFO] lambda_global(epoch={epoch})={lambda_global_epoch:.6f} "
                f"(base={cfg.lambda_global}, warmup_epochs={getattr(cfg, 'lambda_global_warmup_epochs', 0)})"
            )

        # 训练
        train_loss, train_ce_loss, train_global_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=None,  # 不使用混合精度
            max_grad_norm=cfg.max_grad_norm,
            global_loss_fn=global_loss_fn,
            lambda_global=lambda_global_epoch,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            enable_memory_efficient=cfg.enable_memory_efficient_mode,
        )

        # 验证
        val_loss, val_ce_loss, val_global_loss, val_acc, val_preds, val_targets, _, _ = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            is_main=True,
            enable_memory_efficient=cfg.enable_memory_efficient_mode,
            global_loss_fn=global_loss_fn,
            lambda_global=lambda_global_epoch,
        )

        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ce_losses.append(train_ce_loss)
        val_ce_losses.append(val_ce_loss)
        train_global_losses.append(train_global_loss)
        val_global_losses.append(val_global_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 打印分开的loss
        print(
            f"Train - Total: {train_loss:.4f}, CE: {train_ce_loss:.4f}, Global: {train_global_loss:.4f}, Acc: {train_acc:.2f}%"
        )
        print(
            f"Val   - Total: {val_loss:.4f}, CE: {val_ce_loss:.4f}, Global: {val_global_loss:.4f}, Acc: {val_acc:.2f}%"
        )

        # 宏平均指标（macro recall/F1）+ 每类 recall（定位不均衡瓶颈）
        val_metrics = compute_metrics(val_targets, val_preds, int(cfg.num_classes))
        per_class_recall = val_metrics["per_class_recall"]
        per_class_recall_str = ", ".join(
            f"{cls}={per_class_recall[i] * 100:.2f}%"
            for i, cls in enumerate(cfg.classes)
        )
        print(
            f"Val   - MacroRecall: {val_metrics['macro_recall'] * 100:.2f}%, "
            f"MacroF1: {val_metrics['macro_f1'] * 100:.2f}%, "
            f"PerClassRecall: [{per_class_recall_str}]"
        )

        # 打印混淆矩阵
        plot_confusion_matrix(val_targets, val_preds, cfg.classes, epoch + 1, prefix="Val")

        # 早停检查
        if cfg.model_selection_metric == "val_ce":
            monitor = float(val_ce_loss)
            monitor_str = f"val_ce={val_ce_loss:.6f}"
        elif cfg.model_selection_metric == "val_total":
            monitor = float(val_loss)
            monitor_str = f"val_total={val_loss:.6f}"
        elif cfg.model_selection_metric == "val_acc":
            # EarlyStopping 默认越小越好，使用 -acc 实现“越大越好”
            monitor = -float(val_acc)
            monitor_str = f"val_acc={val_acc:.2f}% (monitor={monitor:.6f})"
        else:
            raise ValueError(f"Unknown model_selection_metric: {cfg.model_selection_metric}")

        print(f"[EARLYSTOP] monitor({cfg.model_selection_metric}): {monitor_str}")
        early_stopping(monitor, model)
        if early_stopping.early_stop:
            print("早停触发，停止训练")
            break

    # 加载最佳模型
    print(f"\n[INFO] 加载最佳模型: {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))

    # 测试
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)
    test_loss, test_ce_loss, test_global_loss, test_acc, test_preds, test_targets, test_ids, test_logits = validate_epoch(
        model,
        test_loader,
        criterion,
        device,
        is_main=True,
        enable_memory_efficient=cfg.enable_memory_efficient_mode,
        global_loss_fn=None,  # 测试阶段仅使用 CE
    )

    print(f"Test - Total: {test_loss:.4f}, CE: {test_ce_loss:.4f}, Acc: {test_acc:.2f}%")
    test_metrics = compute_metrics(test_targets, test_preds, int(cfg.num_classes))
    test_per_class_recall = test_metrics["per_class_recall"]
    test_per_class_recall_str = ", ".join(
        f"{cls}={test_per_class_recall[i] * 100:.2f}%"
        for i, cls in enumerate(cfg.classes)
    )
    print(
        f"Test - MacroRecall: {test_metrics['macro_recall'] * 100:.2f}%, "
        f"MacroF1: {test_metrics['macro_f1'] * 100:.2f}%, "
        f"PerClassRecall: [{test_per_class_recall_str}]"
    )
    plot_confusion_matrix(test_targets, test_preds, cfg.classes, epoch="Final", prefix="Test")

    # 打印详细结果
    print_test_results(test_targets, test_preds, test_ids, test_logits, cfg.classes)

    # 保存训练曲线
    os.makedirs(cfg.png_save_dir, exist_ok=True)
    save_plots(train_losses, val_losses, train_accs, val_accs, cfg.png_save_dir, cfg)
    print(f"\n[INFO] 训练曲线已保存至: {cfg.png_save_dir}")


if __name__ == "__main__":
    main()
