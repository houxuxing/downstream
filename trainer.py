"""训练器模块"""

import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from model import set_frozen_batchnorm_eval


# =============================================================================
# 【诊断工具函数】用于检测数据/特征坍缩问题
# =============================================================================

def check_batch_diversity(
    data: torch.Tensor,
    sample_ids: List[str],
    batch_idx: int,
    max_print: int = 16,
) -> dict:
    """
    【Batch 数据多样性检查】
    
    检查 batch 内样本是否真正不同（排除 Dataset/DataLoader 重复样本 bug）
    
    Args:
        data: 输入数据 [B, C, D, H, W]
        sample_ids: 样本 ID 列表
        batch_idx: 当前 batch 索引
        max_print: 最多打印的样本数
    
    Returns:
        dict: 诊断结果
    """
    B = data.shape[0]
    device = data.device
    
    result = {
        "batch_size": B,
        "unique_ids": len(set(sample_ids)),
        "is_duplicate_suspected": False,
        "data_stats": {},
    }
    
    # 仅在第一个 batch 打印详细信息
    if batch_idx == 0:
        print("\n" + "=" * 70)
        print("[DIAG] Batch 数据多样性检查 (batch_idx=0)")
        print("=" * 70)
        
        # 1. 打印样本 ID
        print(f"\n[1] 样本 ID (前 {min(max_print, B)} 个):")
        for i, sid in enumerate(sample_ids[:max_print]):
            print(f"  [{i:2d}] {sid}")
        
        unique_count = len(set(sample_ids))
        print(f"\n  Unique IDs: {unique_count}/{B}")
        if unique_count < B:
            print(f"  ⚠️ 警告: 存在重复样本 ID！")
            result["is_duplicate_suspected"] = True
        
        # 2. 计算每个样本的统计量
        print(f"\n[2] 每个样本的输入统计 (shape={data.shape}):")
        
        # Per-sample mean/std [B]
        data_flat = data.view(B, -1)  # [B, C*D*H*W]
        per_sample_mean = data_flat.mean(dim=1)  # [B]
        per_sample_std = data_flat.std(dim=1)    # [B]
        
        print(f"  Per-sample mean: min={per_sample_mean.min().item():.6f}, "
              f"max={per_sample_mean.max().item():.6f}, "
              f"std={per_sample_mean.std().item():.6f}")
        print(f"  Per-sample std:  min={per_sample_std.min().item():.6f}, "
              f"max={per_sample_std.max().item():.6f}, "
              f"std={per_sample_std.std().item():.6f}")
        
        # 3. 计算与样本 0 的 L2 距离
        print(f"\n[3] 与样本 0 的 L2 距离:")
        sample0 = data_flat[0:1]  # [1, C*D*H*W]
        l2_to_0 = torch.norm(data_flat - sample0, dim=1)  # [B]
        
        print(f"  L2 distances to sample[0]: {l2_to_0[:min(10, B)].tolist()}")
        print(f"  L2 min (excl. self): {l2_to_0[1:].min().item():.6f}" if B > 1 else "  N/A")
        print(f"  L2 max: {l2_to_0.max().item():.6f}")
        print(f"  L2 mean (excl. self): {l2_to_0[1:].mean().item():.6f}" if B > 1 else "  N/A")
        
        # 4. 判断是否为重复数据
        if B > 1 and l2_to_0[1:].max().item() < 1e-5:
            print(f"\n  🚨 严重警告: 所有样本与样本 0 几乎相同 (L2 < 1e-5)!")
            print(f"     这表明 Dataset/DataLoader 返回了重复样本！")
            result["is_duplicate_suspected"] = True
        
        result["data_stats"] = {
            "per_sample_mean_std": per_sample_mean.std().item(),
            "l2_to_0_min": l2_to_0[1:].min().item() if B > 1 else 0,
            "l2_to_0_max": l2_to_0.max().item(),
        }
        
        print("=" * 70 + "\n")
    
    return result


def check_feature_diversity(
    features: torch.Tensor,
    batch_idx: int,
    source: str = "backbone",
    warn_threshold: float = 0.95,
) -> dict:
    """
    【特征多样性检查】
    
    检查 backbone 输出特征是否发生坍缩（所有样本特征几乎相同）
    
    Args:
        features: 特征张量 [B, C]
        batch_idx: 当前 batch 索引
        source: 特征来源名称
        warn_threshold: 余弦相似度警告阈值
    
    Returns:
        dict: 诊断结果
    """
    B, C = features.shape
    device = features.device
    
    result = {
        "shape": (B, C),
        "is_collapsed": False,
        "cosine_offdiag_mean": None,
        "feat_var_across_batch": None,
    }
    
    if B < 2:
        return result
    
    # 1. 计算余弦相似度矩阵
    feat_norm = F.normalize(features, p=2, dim=1)  # [B, C]
    cos_sim = torch.mm(feat_norm, feat_norm.t())   # [B, B]
    
    # 排除对角线
    mask = ~torch.eye(B, dtype=torch.bool, device=device)
    off_diag = cos_sim[mask]
    
    cos_mean = off_diag.mean().item()
    cos_min = off_diag.min().item()
    cos_max = off_diag.max().item()
    
    result["cosine_offdiag_mean"] = cos_mean
    result["cosine_offdiag_min"] = cos_min
    result["cosine_offdiag_max"] = cos_max
    
    # 2. 计算特征方差
    feat_var = features.var(dim=0).mean().item()  # 跨 batch 的方差
    result["feat_var_across_batch"] = feat_var
    
    # 3. 与样本 0 的 L2 距离
    l2_to_0 = torch.norm(features - features[0:1], dim=1)  # [B]
    result["l2_to_0"] = l2_to_0.tolist()
    
    # 4. 判断是否坍缩
    if cos_mean > warn_threshold and feat_var < 0.01:
        result["is_collapsed"] = True
    
    # 仅在第一个 batch 打印详细信息
    if batch_idx == 0:
        print(f"\n" + "-" * 60)
        print(f"[DIAG] 特征多样性检查 ({source})")
        print(f"-" * 60)
        print(f"  Shape: {features.shape}")
        print(f"  Cosine similarity (off-diag):")
        print(f"    mean={cos_mean:.6f}, min={cos_min:.6f}, max={cos_max:.6f}")
        print(f"  Feature variance across batch: {feat_var:.6f}")
        print(f"  L2 to sample[0] (first 5): {l2_to_0[:5].tolist()}")
        
        if result["is_collapsed"]:
            print(f"\n  🚨 严重警告: 特征发生坍缩！")
            print(f"     cos_mean={cos_mean:.4f} > {warn_threshold}, feat_var={feat_var:.6f} < 0.01")
            print(f"     可能原因:")
            print(f"       1. Batch 维度被错误聚合（如 mean(dim=0) 而非 mean(dim=(2,3,4))")
            print(f"       2. CLS token 取错（如 x[0] 而非 x[:, 0])")
            print(f"       3. 特征被 expand/repeat 复制")
            print(f"       4. Dataset 返回重复样本")
        
        print(f"-" * 60 + "\n")
    
    return result


def diagnose_first_batch(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    sample_ids: List[str],
    device: torch.device,
    batch_idx: int = 0,
) -> dict:
    """
    【完整的第一个 batch 诊断】
    
    在训练开始时对第一个 batch 进行全面诊断
    """
    if batch_idx != 0:
        return {}
    
    print("\n" + "=" * 70)
    print("[DIAG] 训练第一个 Batch 完整诊断")
    print("=" * 70)
    
    result = {}
    
    # 1. 检查输入数据多样性
    result["batch_check"] = check_batch_diversity(data, sample_ids, batch_idx)
    
    # 2. 检查标签分布
    print(f"\n[DIAG] 标签分布检查:")
    print(f"  Labels dtype: {target.dtype}")
    print(f"  Labels unique values: {target.unique().tolist()}")
    print(f"  Labels distribution: {torch.bincount(target, minlength=4).tolist()}")
    
    # 断言标签正确性
    assert target.dtype == torch.long, f"Labels dtype 应为 torch.long, 实际为 {target.dtype}"
    assert target.min() >= 0 and target.max() <= 3, f"Labels 应在 [0,3], 实际范围 [{target.min()}, {target.max()}]"
    
    # 3. 检查 backbone 特征
    with torch.no_grad():
        backbone_features = model.backbone(data)
    
    result["backbone_check"] = check_feature_diversity(
        backbone_features, batch_idx, source="backbone", warn_threshold=0.95
    )
    
    # 4. 检查 logits
    with torch.no_grad():
        logits = model(data, return_features=False)
    
    print(f"\n[DIAG] Logits 检查:")
    print(f"  Shape: {logits.shape}")
    print(f"  Per-sample std (mean): {logits.std(dim=1).mean().item():.6f}")
    print(f"  Across-sample std: {logits.std(dim=0).tolist()}")
    
    # 打印前几个样本的 logits
    print(f"\n  前 5 个样本的 logits:")
    for i in range(min(5, logits.shape[0])):
        logit_str = ", ".join([f"{v:.4f}" for v in logits[i].tolist()])
        pred = logits[i].argmax().item()
        true = target[i].item()
        print(f"    [{i}] [{logit_str}] -> pred={pred}, true={true}")
    
    # 5. 汇总诊断
    print("\n" + "=" * 70)
    print("[DIAG] 诊断汇总")
    print("=" * 70)
    
    issues = []
    
    if result["batch_check"].get("is_duplicate_suspected"):
        issues.append("Dataset/DataLoader 可能返回重复样本")
    
    if result["backbone_check"].get("is_collapsed"):
        issues.append("Backbone 特征发生坍缩")
    
    if logits.std(dim=0).mean().item() < 0.05:
        issues.append("Logits 跨样本方差极小，模型可能只输出固定值")
    
    if issues:
        print("\n🚨 检测到以下问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✅ 未检测到明显的数据/特征坍缩问题")
    
    print("=" * 70 + "\n")
    
    return result


def check_optimizer_config(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
) -> None:
    """
    【优化器配置检查】
    
    确保 optimizer 包含正确的参数组
    """
    print("\n" + "-" * 60)
    print("[DIAG] 优化器配置检查")
    print("-" * 60)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"  模型总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  可训练比例: {100.0 * trainable_params / total_params:.2f}%")
    
    print(f"\n  优化器参数组:")
    opt_total = 0
    for i, group in enumerate(optimizer.param_groups):
        group_params = sum(p.numel() for p in group['params'] if p.requires_grad)
        opt_total += group_params
        name = group.get('name', f'group_{i}')
        lr = group.get('lr', 'N/A')
        print(f"    [{i}] {name}: {group_params:,} params, lr={lr}")
    
    # 断言检查
    if opt_total == 0:
        raise RuntimeError(
            "🚨 优化器参数组为空！没有可训练参数被添加到优化器中。\n"
            "请检查 setup_parameter_freezing 和 create_optimizer 函数。"
        )
    
    if opt_total != trainable_params:
        print(f"\n  ⚠️ 警告: 优化器参数 ({opt_total:,}) != 可训练参数 ({trainable_params:,})")
    
    # 检查 head 是否在优化器中
    head_in_opt = False
    for group in optimizer.param_groups:
        if group.get('name') == 'head':
            head_in_opt = True
            break
    
    if not head_in_opt:
        print(f"\n  ⚠️ 警告: 未找到名为 'head' 的参数组")
    
    print("-" * 60 + "\n")


class EarlyStopping:
    """早停机制"""

    def __init__(
        self, patience: int = 10, save_path: str = "model.pth", is_main: bool = True
    ):
        self.patience = patience
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.save_path = save_path
        self.is_main = is_main

    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss >= self.best_score:
            self.counter += 1
            if self.is_main:
                print(f"早停计数器: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model: nn.Module):
        if self.is_main:
            torch.save(model.state_dict(), self.save_path)
            print(f"最佳模型保存至 {self.save_path}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    max_grad_norm: float = 1.0,
    global_loss_fn: Optional[nn.Module] = None,
    lambda_global: float = 1.0,
    gradient_accumulation_steps: int = 1,
    enable_memory_efficient: bool = False,
) -> Tuple[float, float]:
    """单个 epoch 的训练"""
    model.train()
    set_frozen_batchnorm_eval(model)

    running_loss = 0.0
    running_ce_loss = 0.0
    running_global_loss = 0.0
    running_corrects = 0
    total = 0
    valid_used = 0
    valid_total = 0
    any_global_loss_computed = False

    pbar = tqdm(train_loader, desc="训练")
    amp_enabled = (scaler is not None) and (device.type == "cuda")
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"

    first_batch_diagnosed = False
    
    for batch_idx, batch_data in enumerate(pbar):
        # Unpack data
        data, target, sample_ids, age, measures = batch_data

        # Handle two-view batch logic
        two_view_batch = data.ndim == 6 and data.size(1) == 2
        
        # 【诊断】第一个 batch 进行完整诊断（在 two_view 展开之前）
        if batch_idx == 0 and not first_batch_diagnosed:
            first_batch_diagnosed = True
            # 如果是 two_view，先取第一个视图进行诊断
            diag_data = data[:, 0] if two_view_batch else data
            diag_data = diag_data.to(device, non_blocking=True)
            diag_target = target.to(device, non_blocking=True)
            
            diagnose_first_batch(
                model=model,
                data=diag_data,
                target=diag_target,
                sample_ids=list(sample_ids) if not isinstance(sample_ids, list) else sample_ids,
                device=device,
                batch_idx=0,
            )
        if two_view_batch:
            B0 = data.size(0)
            target_base = target
            sample_ids_base = sample_ids

            # Flatten views: [B,2,C,D,H,W] -> [2B,C,D,H,W]
            data = data.view(B0 * 2, *data.shape[2:])
            target = target.repeat_interleave(2)
            sample_ids = [sid for sid in sample_ids for _ in range(2)]

            if global_loss_fn is not None:
                age = age.repeat_interleave(2)
                # Correctly handle measures dimensions:
                # If measures is [B, 2, K, 3], flatten to [2B, K, 3]
                if measures.ndim == 4 and measures.shape[1] == 2:
                    measures = measures.view(B0 * 2, *measures.shape[2:])
                else:
                    measures = measures.repeat_interleave(2, dim=0)
        else:
            target_base = None

        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        use_global = global_loss_fn is not None
        if use_global:
            age = age.to(device, non_blocking=True)
            measures = measures.to(device, non_blocking=True)

        # Gradient accumulation reset
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        is_accum_end = ((batch_idx + 1) % gradient_accumulation_steps == 0) or (
            (batch_idx + 1) == len(train_loader)
        )

        features = None
        with torch.autocast(device_type=autocast_device_type, enabled=amp_enabled):
            if use_global:
                logits, features = model(data, return_features=True)
            else:
                logits = model(data, return_features=False)

            ce_loss = criterion(logits, target)
        
        # 【运行时检查】周期性检测特征坍缩（每 100 个 batch 检查一次）
        if batch_idx > 0 and batch_idx % 100 == 0:
            with torch.no_grad():
                backbone_feat = model.backbone(data)
                if backbone_feat.shape[0] > 1:
                    feat_norm = F.normalize(backbone_feat, p=2, dim=1)
                    cos_sim = torch.mm(feat_norm, feat_norm.t())
                    mask = ~torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
                    cos_mean = cos_sim[mask].mean().item()
                    
                    if cos_mean > 0.98:
                        print(f"\n⚠️ [batch={batch_idx}] 特征坍缩警告: cos_sim_offdiag_mean={cos_mean:.4f} > 0.98")

        # Compute Global Loss if enabled
        if use_global:
            assert features is not None
            global_loss, was_computed = _compute_global_loss(
                global_loss_fn,
                features.float(),
                age.float(),
                measures.float(),
                sample_ids,
                batch_idx,
                device,
                autocast_device_type=autocast_device_type,
                amp_enabled=False,  # Force FP32 for stability
            )
            any_global_loss_computed = any_global_loss_computed or was_computed
            loss = ce_loss + lambda_global * global_loss
        else:
            global_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            loss = ce_loss

        loss = loss / gradient_accumulation_steps

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            if is_accum_end:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if is_accum_end:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        # Memory cleanup
        if enable_memory_efficient:
            del data
            if features is not None:
                del features
            if device.type == "cuda" and (batch_idx % 50 == 0):
                torch.cuda.empty_cache()
        
        # Calculate metrics
        if two_view_batch:
            # Average logits across views for accuracy
            logits_acc = logits.view(B0, 2, -1).mean(dim=1)  # [B0, num_classes]
            preds = logits_acc.argmax(dim=1)
            batch_size = target_base.size(0)
            correct = preds.eq(target_base.to(preds.device)).sum().item()
        else:
            preds = logits.argmax(dim=1)
            batch_size = target.size(0)
            correct = preds.eq(target).sum().item()

        total += batch_size
        running_loss += loss.item() * batch_size * gradient_accumulation_steps
        running_ce_loss += ce_loss.item() * batch_size
        running_global_loss += global_loss.item() * batch_size
        running_corrects += correct

        if len(optimizer.param_groups) > 1:
            postfix = {
                "lr_head": f"{optimizer.param_groups[0]['lr']:.2e}",
                "lr_back": f"{optimizer.param_groups[1]['lr']:.2e}",
                "loss": f"{running_loss / total:.4f}",
                "ce": f"{running_ce_loss / total:.4f}",
                "global": f"{running_global_loss / total:.4f}",
                "acc": f"{100.0 * running_corrects / total:.2f}%",
            }
        else:
            postfix = {
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "loss": f"{running_loss / total:.4f}",
                "ce": f"{running_ce_loss / total:.4f}",
                "global": f"{running_global_loss / total:.4f}",
                "acc": f"{100.0 * running_corrects / total:.2f}%",
            }
        if global_loss_fn is not None and valid_total > 0:
            postfix["valid"] = f"{100.0 * valid_used / valid_total:.1f}%"
        pbar.set_postfix(postfix)

    if global_loss_fn is not None and not any_global_loss_computed:
        raise RuntimeError(
            "Global Loss 已启用，但整个 epoch 内从未真正计算。"
            "请检查 batch_size 是否 >=2，数据集是否已过滤缺失 age/measures。"
        )

    return running_loss / total, running_corrects / total * 100.0


def _compute_global_loss(
    global_loss_fn: nn.Module,
    features: torch.Tensor,
    age: torch.Tensor,
    measures: torch.Tensor,
    sample_ids: List[str],
    batch_idx: int,
    device: torch.device,
    autocast_device_type: str,
    amp_enabled: bool,
) -> Tuple[torch.Tensor, bool]:
    """计算 Global Loss

    Returns:
        Tuple[torch.Tensor, bool]: (loss, was_computed)
            - loss: Global Loss 值
            - was_computed: 是否真正计算了 Global Loss
    """
    measures = torch.nan_to_num(measures, nan=0.0, posinf=0.0, neginf=0.0)

    # Handle measures dimensions
    # Single view: [B, K, 3] -> [B, K, 3]
    # Two view (flattened): [2B, K, 3] -> [2B, K, 3]
    # Incorrectly flattened: [B, 2, K, 3] -> flatten to [2B, K, 3]
    
    if measures.ndim == 4 and measures.shape[1] == 2:
        measures = measures.view(-1, *measures.shape[2:])
    
    if measures.ndim != 3:
         raise RuntimeError(f"Global Loss enabled but measures dim invalid (expected 3, got {measures.ndim}).")
    
    if measures.shape[1] <= 0:
        raise RuntimeError(
            f"Global Loss enabled but ROI count is 0 (shape={measures.shape}).\n"
            "Check config.measure_root and config.region_order_json."
        )

    if torch.isnan(age).any():
        bad_ids = [
            sid for sid, ok in zip(sample_ids, (~torch.isnan(age)).tolist()) if not ok
        ]
        raise RuntimeError(f"Global Loss enabled but NaN age found: {bad_ids[:10]}")

    zero_mask = measures.abs().sum(dim=(1, 2)) == 0
    if bool(zero_mask.any()):
        bad_ids = [sid for sid, z in zip(sample_ids, zero_mask.tolist()) if z]
        raise RuntimeError(f"Global Loss enabled but all-zero measures found: {bad_ids[:10]}")

    batch_size = age.size(0)
    if batch_idx == 0 and batch_size < 2:
        raise RuntimeError("Global Loss enabled but first batch_size < 2.")

    valid_mask = ~torch.isnan(age)
    cur_valid = int(valid_mask.sum().item())

    if cur_valid > 1:
        # Fix mismatch between flattened measures and non-flattened age in validation
        # If validation measures are [2B, ...] (due to dataset logic) but age is [B],
        # we only take the first view to align with age.
        if measures.shape[0] == age.shape[0] * 2:
             measures = measures.view(age.shape[0], 2, *measures.shape[1:])[:, 0, ...]
        
        valid_features = features[valid_mask].float()
        valid_ages = age[valid_mask].float()
        valid_measures = measures[valid_mask].float()
        with torch.autocast(device_type=autocast_device_type, enabled=amp_enabled):
            loss = global_loss_fn(valid_features, valid_ages, valid_measures)
            return loss, True  # 成功计算

    return torch.tensor(0.0, device=device, dtype=torch.float32), False  # 未计算


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_main: bool,
    enable_memory_efficient: bool = False,
    global_loss_fn: Optional[nn.Module] = None,
    lambda_global: float = 1.0,
    debug_first_batch: bool = False,
) -> Tuple[float, float, float, List[int], List[int], List[str], List[List[float]]]:
    """验证/测试阶段"""
    model.eval()
    running_loss = 0.0
    running_ce_loss = 0.0
    running_global_loss = 0.0
    running_corrects = 0
    total = 0

    all_preds = []
    all_targets = []
    all_logits = []
    all_sample_ids = []

    for batch_idx, batch_data in enumerate(
        tqdm(loader, desc="验证" if is_main else "val", leave=False)
    ):
        data, target, sample_id, age, measures = batch_data
        data = data.to(device, non_blocking=True)

        if batch_idx == 0:
            print("data.shape =", data.shape, "dtype =", data.dtype)
            
            # 调试：检查输入数据分布
            if debug_first_batch and is_main:
                print("\n" + "="*60)
                print("[DEBUG] 第一个 batch 诊断信息")
                print("="*60)
                print(f"输入数据统计:")
                print(f"  - min: {data.min().item():.6f}")
                print(f"  - max: {data.max().item():.6f}")
                print(f"  - mean: {data.mean().item():.6f}")
                print(f"  - std: {data.std().item():.6f}")

        target = target.to(device, non_blocking=True)

        use_global = global_loss_fn is not None
        if use_global:
            age = age.to(device, non_blocking=True)
            measures = measures.to(device, non_blocking=True)

        # 根据是否需要 global loss 决定是否返回 features
        features = None
        if use_global:
            logits, features = model(data, return_features=True)
        else:
            logits = model(data, return_features=False)
        
        # 调试：检查特征和 logits 分布
        if batch_idx == 0 and debug_first_batch and is_main:
            # 获取 backbone 特征
            with torch.no_grad():
                backbone_features = model.backbone(data)
                
            print(f"\nBackbone 特征统计:")
            print(f"  - shape: {backbone_features.shape}")
            print(f"  - min: {backbone_features.min().item():.6f}")
            print(f"  - max: {backbone_features.max().item():.6f}")
            print(f"  - mean: {backbone_features.mean().item():.6f}")
            print(f"  - std: {backbone_features.std().item():.6f}")
            
            # 检查特征是否坍缩（所有样本特征几乎相同）
            if backbone_features.shape[0] > 1:
                # 计算样本间特征的余弦相似度
                feat_norm = backbone_features / (backbone_features.norm(dim=1, keepdim=True) + 1e-8)
                cos_sim = torch.mm(feat_norm, feat_norm.t())
                # 排除对角线
                mask = ~torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
                off_diag_sim = cos_sim[mask]
                print(f"  - 样本间余弦相似度 (mean): {off_diag_sim.mean().item():.6f}")
                print(f"  - 样本间余弦相似度 (min): {off_diag_sim.min().item():.6f}")
                print(f"  - 样本间余弦相似度 (max): {off_diag_sim.max().item():.6f}")
                
                if off_diag_sim.mean().item() > 0.99:
                    print(f"  \u26a0\ufe0f 警告: 特征高度相似，可能发生特征坍塞！")
            
            print(f"\nLogits 统计:")
            print(f"  - shape: {logits.shape}")
            print(f"  - min: {logits.min().item():.6f}")
            print(f"  - max: {logits.max().item():.6f}")
            print(f"  - mean: {logits.mean().item():.6f}")
            print(f"  - std: {logits.std().item():.6f}")
            
            # 打印每个样本的 logits
            print(f"\n前 5 个样本的 logits:")
            for i in range(min(10, logits.shape[0])):
                logit_str = ", ".join([f"{v:.4f}" for v in logits[i].tolist()])
                pred = logits[i].argmax().item()
                true = target[i].item()
                print(f"  样本 {i}: [{logit_str}] -> pred={pred}, true={true}")
            
            # 检查 logits 是否几乎相同
            if logits.shape[0] > 1:
                logits_std_per_sample = logits.std(dim=1)  # 每个样本内部 logits 的标准差
                logits_std_across_samples = logits.std(dim=0)  # 跨样本的标准差
                print(f"\nLogits 方差分析:")
                print(f"  - 每个样本内部 logits 标准差 (mean): {logits_std_per_sample.mean().item():.6f}")
                print(f"  - 跨样本的 logits 标准差: {logits_std_across_samples.tolist()}")
                
                if logits_std_per_sample.mean().item() < 0.01:
                    print(f"  \u26a0\ufe0f 警告: 每个样本的 logits 方差极小，模型输出几乎均匀分布！")
            
            print("="*60 + "\n")

        ce_loss = criterion(logits, target)

        # 计算 global loss（与训练时对齐）
        if use_global:
            assert features is not None  # 由于 use_global=True 时必有 features
            global_loss, _ = _compute_global_loss(
                global_loss_fn,
                features.float(),
                age.float(),
                measures.float(),
                sample_id,
                batch_idx,
                device,
                autocast_device_type="cuda" if device.type == "cuda" else "cpu",
                amp_enabled=False,
            )
            loss = ce_loss + lambda_global * global_loss
        else:
            global_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            loss = ce_loss

        preds = logits.argmax(dim=1)
        bs = data.size(0)
        total += bs
        running_loss += loss.item() * bs
        running_ce_loss += ce_loss.item() * bs
        running_global_loss += global_loss.item() * bs
        running_corrects += preds.eq(target).sum().item()

        # 立即移到CPU以释放GPU内存
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(target.detach().cpu().numpy().tolist())
        all_logits.extend(logits.detach().cpu().numpy().tolist())
        all_sample_ids.extend(sample_id)

        # 内存优化：及时释放GPU内存
        if enable_memory_efficient:
            del data, logits, preds, target
            if features is not None:
                del features
            if batch_idx % 10 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()

    return (
        running_loss / total,
        running_ce_loss / total,
        running_corrects / total * 100.0,
        all_preds,
        all_targets,
        all_sample_ids,
        all_logits,
    )


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """创建学习率调度器（warmup + 余弦）"""

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))

        progress = float(epoch - warmup_epochs) / float(
            max(1, num_epochs - warmup_epochs)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = 1e-2  # 降低最小学习率到 1%
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
