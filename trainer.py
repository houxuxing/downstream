"""训练器模块"""

import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.cuda.amp import GradScaler  # Fallback for older PyTorch
from tqdm import tqdm

from config import Config
from model import set_frozen_batchnorm_eval


def schedule_lambda_global(epoch: int, cfg: Config) -> float:
    """Global Loss 权重调度（线性 warmup）。

    约定：
    - epoch0 -> 0
    - epoch == cfg.lambda_global_warmup_epochs -> cfg.lambda_global
    - 之后保持 cfg.lambda_global
    """
    if not getattr(cfg, "use_global_loss", False):
        return 0.0

    lambda_global = float(getattr(cfg, "lambda_global", 0.0))
    if lambda_global <= 0.0:
        return 0.0

    warmup_epochs = int(getattr(cfg, "lambda_global_warmup_epochs", 0))
    if warmup_epochs <= 0:
        return lambda_global

    scale = float(epoch) / float(warmup_epochs)
    scale = max(0.0, min(1.0, scale))
    return lambda_global * scale


class EarlyStopping:
    """早停机制

    Args:
        patience: 连续多少个 epoch 无改进后停止
        save_path: 最佳模型保存路径
        is_main: 是否为主进程
        min_delta: 最小改进阈值，loss 必须下降超过此值才算有效改进
    """

    def __init__(
        self,
        patience: int = 10,
        save_path: str = "model.pth",
        is_main: bool = True,
        min_delta: float = 1e-4,
    ):
        self.patience = patience
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.save_path = save_path
        self.is_main = is_main
        self.min_delta = min_delta

    def __call__(self, val_loss: float, model: nn.Module):
        if not math.isfinite(val_loss):
            if self.is_main:
                print(f"[WARNING] val_loss ????: {val_loss}????????")
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss >= self.best_score - self.min_delta:
            # Loss did not improve by at least min_delta
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
) -> Tuple[float, float, float, float]:
    """单个 epoch 的训练
    
    Returns:
        Tuple[float, float, float, float]: (total_loss, ce_loss, global_loss, accuracy)
    """
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

    for batch_idx, batch_data in enumerate(pbar):
        # Unpack data
        data, target, sample_ids, age, measures = batch_data


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
        preds = logits.argmax(dim=1)
        batch_size = target.size(0)
        correct = preds.eq(target).sum().item()

        total += batch_size
        running_loss += loss.item() * batch_size * gradient_accumulation_steps
        running_ce_loss += ce_loss.item() * batch_size
        running_global_loss += global_loss.item() * batch_size
        running_corrects += correct

        if len(optimizer.param_groups) > 1:
            # 动态寻找 Head 和 Backbone 的代表性学习率
            head_lrs = [
                g["lr"] for g in optimizer.param_groups if "head" in g.get("name", "")
            ]
            back_lrs = [
                g["lr"] for g in optimizer.param_groups if "back" in g.get("name", "")
            ]

            curr_lr_head = head_lrs[0] if head_lrs else optimizer.param_groups[0]["lr"]
            curr_lr_back = (
                max(back_lrs)
                if back_lrs
                else (
                    optimizer.param_groups[1]["lr"]
                    if len(optimizer.param_groups) > 1
                    else curr_lr_head
                )
            )

            postfix = {
                "lr_head": f"{curr_lr_head:.2e}",
                "lr_back": f"{curr_lr_back:.2e}",
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

    return running_loss / total, running_ce_loss / total, running_global_loss / total, running_corrects / total * 100.0


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
        raise RuntimeError(
            f"Global Loss enabled but measures dim invalid (expected 3, got {measures.ndim})."
        )

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
        raise RuntimeError(
            f"Global Loss enabled but all-zero measures found: {bad_ids[:10]}"
        )

    batch_size = age.size(0)
    if batch_idx == 0 and batch_size < 2:
        raise RuntimeError("Global Loss enabled but first batch_size < 2.")

    valid_mask = ~torch.isnan(age)
    cur_valid = int(valid_mask.sum().item())

    if cur_valid > 1:
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
) -> Tuple[float, float, float, float, List[int], List[int], List[str], List[List[float]]]:
    """验证/测试阶段
    
    Returns:
        Tuple containing:
        - total_loss: 总损失
        - ce_loss: CE损失
        - global_loss: Global损失
        - accuracy: 准确率
        - all_preds: 所有预测
        - all_targets: 所有目标
        - all_sample_ids: 所有样本ID
        - all_logits: 所有logits
    """
    model.eval()
    running_loss = 0.0
    running_ce_loss = 0.0
    running_global_loss = 0.0
    running_corrects = 0
    total = 0

    all_preds = []
    all_targets = []
    all_logits_tensors = []
    all_sample_ids = []

    for batch_idx, batch_data in enumerate(
        tqdm(loader, desc="验证" if is_main else "val", leave=False)
    ):
        data, target, sample_id, age, measures = batch_data
        data = data.to(device, non_blocking=True)

        target = target.to(device, non_blocking=True)
        use_global = global_loss_fn is not None
        if use_global:
            age = age.to(device, non_blocking=True)
            measures = measures.to(device, non_blocking=True)

        features = None
        if use_global:
            logits, features = model(data, return_features=True)
        else:
            logits = model(data, return_features=False)

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
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(target.detach().cpu().tolist())
        all_logits_tensors.append(logits.detach().cpu())
        all_sample_ids.extend(sample_id)

        # 内存优化：及时释放GPU内存
        if enable_memory_efficient:
            del data, logits, preds, target
            if features is not None:
                del features
            if batch_idx % 10 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()

    if all_logits_tensors:
        all_logits = torch.cat(all_logits_tensors, dim=0).tolist()
    else:
        all_logits = []

    return (
        running_loss / total,
        running_ce_loss / total,
        running_global_loss / total,
        running_corrects / total * 100.0,
        all_preds,
        all_targets,
        all_sample_ids,
        all_logits,
    )


class CosineSchedulerWithWarmup:
    """
    Cosine Scheduler with Warmup

    支持:
    - 学习率: Warmup 阶段线性增加，之后余弦衰减到 min_lr
    - Weight Decay: 余弦调度，从 wd_init 增加到 wd_end

    Args:
        optimizer: 优化器
        num_epochs: 总训练 epoch 数
        warmup_epochs: warmup 阶段的 epoch 数
        min_lr: 最小学习率
        wd_init: 初始 weight decay
        wd_end: 最终 weight decay
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        warmup_epochs: int,
        min_lr: float = 1e-6,
        wd_init: float = 0.04,
        wd_end: float = 0.4,
    ):
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.wd_init = wd_init
        self.wd_end = wd_end

        # 保存每个参数组的初始学习率
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        # 保存每个参数组的初始 weight decay 缩放因子
        self.wd_scales = [
            group.get("wd_scale", 1.0) for group in optimizer.param_groups
        ]

        # ?? -1?????? step() ?? epoch=0??? warmup ???????
        self.current_epoch = -1

    def step(self, epoch: Optional[int] = None):
        """更新学习率和 weight decay"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        epoch = self.current_epoch

        for i, (param_group, base_lr, wd_scale) in enumerate(
            zip(self.optimizer.param_groups, self.base_lrs, self.wd_scales)
        ):
            # 计算学习率
            lr = self._get_lr(epoch, base_lr)
            param_group["lr"] = lr

            # 计算 weight decay（余弦调度，从 wd_init 增加到 wd_end）
            wd = self._get_weight_decay(epoch) * wd_scale
            param_group["weight_decay"] = wd

    def _get_lr(self, epoch: int, base_lr: float) -> float:
        """计算当前 epoch 的学习率"""
        if epoch < self.warmup_epochs:
            # Warmup 阶段：线性增加
            return base_lr * (epoch + 1) / max(1, self.warmup_epochs)
        else:
            # Cosine decay 阶段
            progress = (epoch - self.warmup_epochs) / max(
                1, self.num_epochs - self.warmup_epochs
            )
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (base_lr - self.min_lr) * cosine_factor

    def _get_weight_decay(self, epoch: int) -> float:
        """计算当前 epoch 的 weight decay（余弦调度，递增）"""
        if epoch < self.warmup_epochs:
            # Warmup 阶段：保持初始值
            return self.wd_init
        else:
            # Cosine 调度：从 wd_init 增加到 wd_end
            progress = (epoch - self.warmup_epochs) / max(
                1, self.num_epochs - self.warmup_epochs
            )
            cosine_factor = 0.5 * (1.0 - math.cos(math.pi * progress))  # 从 0 到 1
            return self.wd_init + (self.wd_end - self.wd_init) * cosine_factor

    def state_dict(self):
        """返回调度器状态"""
        return {
            "current_epoch": self.current_epoch,
            "base_lrs": self.base_lrs,
            "wd_scales": self.wd_scales,
        }

    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.current_epoch = state_dict["current_epoch"]
        self.base_lrs = state_dict["base_lrs"]
        self.wd_scales = state_dict["wd_scales"]


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    warmup_epochs: int,
    min_lr: float = 1e-6,
    wd_init: float = 0.04,
    wd_end: float = 0.4,
) -> CosineSchedulerWithWarmup:
    """创建学习率调度器（Warmup + Cosine + Weight Decay 调度）

    Args:
        optimizer: 优化器
        num_epochs: 总训练 epoch 数
        warmup_epochs: warmup 阶段的 epoch 数
        min_lr: 最小学习率（余弦退火终点）
        wd_init: 初始 weight decay
        wd_end: 最终 weight decay

    Returns:
        CosineSchedulerWithWarmup 调度器
    """
    return CosineSchedulerWithWarmup(
        optimizer=optimizer,
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
        min_lr=min_lr,
        wd_init=wd_init,
        wd_end=wd_end,
    )
