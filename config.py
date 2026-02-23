"""下游任务配置模块"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Config:
    """四分类下游任务配置"""

    # ========================
    # 双分支模式配置
    # ========================
    use_dual_branch: bool = True  # 是否启用双分支模式

    # 主分支配置 (Branch 1) - 仅在 use_dual_branch=True 时生效
    backbone_name1: str = "voxhrnet"  # 主分支骨干网络
    train_mode1: str = "none"  # 主分支训练模式 (冻结)
    unfreeze_last_n_blocks1: int = 0  # 主分支解冻层数

    # 副分支配置 (Branch 2) - 仅在 use_dual_branch=True 时生效
    backbone_name2: str = "brainsegfounder"  # 副分支骨干网络
    train_mode2: str = "partial"  # 副分支训练模式 (冻结)
    unfreeze_last_n_blocks2: int = 0  # 副分支解冻层数
    # DoRA (Branch 2: BrainSegFounder, train_mode2='partial')
    use_dora_branch2: bool = True
    dora_branch2_r: int = 8
    dora_branch2_alpha: int = 16
    dora_branch2_target_modules: Tuple[str, ...] = ("qkv", "proj", "fc1", "fc2")

    # 自适应特征融合配置
    fusion_dim: int = 256  # 1x1x1 Conv 投影维度
    fusion_spatial_size: int = 4  # 空间对齐尺寸 (默认 4x4x4)
    fusion_ca_reduction: int = 4  # Channel Attention 压缩比例

    # ========================
    # 单分支模式配置 (原有配置)
    # ========================
    # 训练模式: 'none' | 'head' | 'partial'
    # - 'none': 全部可训练（默认，从头训练）
    # - 'head': 只训练分类头，冻结整个 backbone（容易导致特征坍缩）
    # - 'partial': 冻结 backbone 前面的层，解冻最后几层 + 分类头（推荐）
    train_mode: str = "partial"  # 改用 partial 模式，避免特征坍缩

    # partial 模式下解冻的 Transformer 块数量
    unfreeze_last_n_blocks: int = 6

    # 模型配置
    # 支持 'dinov2', 'brainsegfounder', 'sam_brain3d', 'voxhrnet', 'nnunet'，'brainfm'
    backbone_name: str = "brainsegfounder"
    num_classes: int = 4
    embed_dim: int = 512  # 增强版分类头中间层维度

    # 3DINO
    dinov2_ckpt: Optional[str] = (
        "/data/aim_nuist/aim_temp/aim_houxuxing/papercode/3DINO-main/output12/best_model.pth"
    )

    # BrainSegFounder
    brainsegfounder_ckpt: Optional[str] = (
        "/data/aim_nuist/aim_temp/aim_houxuxing/papercode/3DINO-main/brainseg_output12/best_model.pth"
    )

    # SAM-Brain 3D
    sam_brain3d_ckpt: Optional[str] = (
        "/data/aim_nuist/aim_temp/aim_houxuxing/papercode/3DINO-main/sam-brain_output12/best_model.pth"
    )

    # VoxHR-Net
    voxhrnet_ckpt: Optional[str] = (
        "/data/aim_nuist/aim_temp/aim_houxuxing/papercode/VoxHRNet-main/output/best_loss.pth"
    )

    # nnUNet
    nnunet_ckpt: Optional[str] = (
        "/data/aim_nuist/aim_temp/aim_houxuxing/papercode/nnUNet-master/nnUNet_results/Dataset500_ADNI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"  # 预训练权重路径（相对于 downstream 目录）
    )
    nnunet_plans_json: Optional[str] = (
        "/data/aim_nuist/aim_temp/aim_houxuxing/papercode/nnUNet-master/nnUNet_preprocessed/Dataset500_ADNI/nnUNetPlans.json"  # 模型配置文件
    )
    use_nnunet_zscore: Optional[bool] = (
        None  # 是否使用 ZScore 归一化（自动根据 backbone_name 设置）
    )

    # BrainFM
    brainfm_ckpt: Optional[str] = (
        "/data/aim_nuist/aim_temp/aim_houxuxing/papercode/3DINO-main/brainfm_output12/best_model.pth"
    )

    # 数据配置
    target_size: Tuple[int, int, int] = (128, 128, 128)  # 减小尺寸以节省内存
    min_int: float = -1.0
    resize_scale: float = 1.0
    num_workers_val: int = 0  # 验证/测试建议用 0，避免 DataLoader 多进程 IPC/SHM 报错
    batch_size: int = 8
    num_workers: int = 2  # 降低 worker 数量以减少共享内存占用
    pin_memory: bool = False  # 禁用 pin_memory 以解决 RuntimeError: Pin memory thread exited unexpectedly

    # ========================
    # 数据均衡/采样策略
    # ========================
    # - weighted_sampler:   训练集全量 + WeightedRandomSampler 近似均衡（推荐）
    # - class_weight:       训练集全量 + CrossEntropyLoss(class_weight)
    # - downsample:         训练集按最小类下采样到完全均衡（会丢数据）
    # - downsample_to_pmci: train/val/test 都按各自 pMCI 数量对其他类下采样（会丢数据）
    balance_strategy: str = "downsample"
    # 验证/测试是否也下采样（仅对 balance_strategy="downsample" 生效）
    # 默认 True：val/test 也按最小类下采样（与 train 的 downsample 行为一致）
    balance_eval_downsample: bool = True
    # WeightedRandomSampler 配置
    sampler_replacement: bool = True
    sampler_num_samples: Optional[int] = None  # None 表示 num_samples=len(train_set)

    # 内存优化配置
    gradient_accumulation_steps: int = (
        4  # 梯度累积步数，实际 effective_batch_size = 8 × 4 = 32
    )
    enable_memory_efficient_mode: bool = False  # 启用内存优化模式

    seed: int = 42
    deterministic: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False

    # 训练配置
    num_epochs: int = 100
    warmup_epochs: int = 2  # Warmup epochs，学习率从0线性增加到目标值

    # ========================
    # 学习率配置（基准值，实际会根据 batch size 自动缩放）
    # ========================
    # 基准 batch size（用于学习率缩放的参考值）
    # 注意: 设置为实际使用的 batch size 可禁用缩放
    base_batch_size: int = 8  # 改为实际 batch size，避免过度缩放

    # 基准学习率（对应 base_batch_size）
    base_head_lr: float = 1e-3  # 分类头基准学习率
    # Backbone 微调通常需要更小的 LR，避免破坏预训练表征
    base_backbone_lr: float = 1e-4  # Backbone 基准学习率
    min_lr: float = 1e-6  # 最小学习率（余弦退火终点）

    # ========================
    # 优化器配置 (AdamW)
    # ========================
    # AdamW betas
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999

    # Weight decay 配置（使用余弦调度）
    # 针对特征坍塌问题，降低 weight decay 以允许更大的参数更新
    weight_decay_init: float = (
        0.005  # 初始 weight decay（降低以防止分类头权重过度衰减）
    )
    weight_decay_end: float = 0.05  # 最终 weight decay
    backbone_wd_scale: float = 0.1  # Backbone 的 weight decay 缩放因子（降低）

    # 梯度裁剪
    max_grad_norm: float = 3.0  # 梯度裁剪阈值

    # 早停配置
    patience: int = 20
    # 最佳模型保存/早停监控指标
    # - val_ce:    验证集 CE loss（推荐，与分类目标对齐）
    # - val_acc:   验证集 Accuracy（更直观，但对不均衡敏感）
    # - val_total: CE + lambda_global * Global（可能与分类目标不一致）
    model_selection_metric: str = "val_ce"

    # Global Loss 配置
    use_global_loss: bool = True
    # Global Loss 的权重（建议比 CE 小，且配合 warmup）
    lambda_global: float = 0.1
    # 线性 warmup：epoch0=0，epoch==warmup_epochs 达到 lambda_global
    lambda_global_warmup_epochs: int = 5

    # AnatCL 损失函数配置
    # y-Aware: RBF bandwidth
    sigma: float = 5.0
    # InfoNCE temperature
    temperature: float = 0.1
    # final loss weights
    lambda_anat: float = 1.0
    lambda_age: float = 1.0

    # 路径配置
    data_dir: str = "/data/aim_nuist/aim_temp/aim_houxuxing/dataset/ADNI_nii"
    csv_file_processed: str = "/data/aim_nuist/aim_temp/aim_houxuxing/dataset_label/CN_MCI_AD_with_PTID_processed.csv"

    model_save_dir: str = "/data/aim_nuist/aim_temp/aim_houxuxing/output/anatcl/model"
    model_save_name: Optional[str] = None  # 自动基于 backbone_name 生成

    png_save_dir: str = "/data/aim_nuist/aim_temp/aim_houxuxing/output/anatcl/png"
    png_save_name: Optional[str] = None  # 自动基于 backbone_name 生成

    age_csv: Optional[str] = (
        "/data/aim_nuist/aim_temp/aim_houxuxing/dataset_label/CN_MCI_AD_with_PTID_processed.csv"
    )
    measure_root: Optional[str] = (
        "/data/aim_nuist/aim_temp/aim_houxuxing/dataset_label/csv_file"
    )

    region_order_json: Optional[str] = (
        "/data/aim_nuist/aim_temp/aim_houxuxing/papercode/AnatCL-ADNI/anatcl/ROI.json"
    )

    # 类别名称
    classes: Tuple[str, ...] = ("CN", "sMCI", "pMCI", "Dementia")

    def __post_init__(self):
        # 训练模式验证
        assert self.train_mode in {"none", "head", "partial"}, (
            f"train_mode must be 'none', 'head', or 'partial', got '{self.train_mode}'"
        )

        # 数值参数验证
        assert self.num_classes > 0, (
            f"num_classes must be positive, got {self.num_classes}"
        )
        assert self.batch_size > 0, (
            f"batch_size must be positive, got {self.batch_size}"
        )
        assert self.num_epochs > 0, (
            f"num_epochs must be positive, got {self.num_epochs}"
        )
        assert self.unfreeze_last_n_blocks >= 0, (
            f"unfreeze_last_n_blocks must be non-negative, got {self.unfreeze_last_n_blocks}"
        )

        assert self.resize_scale > 0, (
            f"resize_scale must be positive, got {self.resize_scale}"
        )
        assert self.seed >= 0, f"seed must be non-negative, got {self.seed}"

        # 学习率验证
        assert 0 < self.base_head_lr <= 1, (
            f"base_head_lr must be in (0, 1], got {self.base_head_lr}"
        )
        assert 0 < self.base_backbone_lr <= 1, (
            f"base_backbone_lr must be in (0, 1], got {self.base_backbone_lr}"
        )

        # 采样/评估策略验证
        valid_balance = {
            "weighted_sampler",
            "class_weight",
            "downsample",
            "downsample_to_pmci",
        }
        if self.balance_strategy not in valid_balance:
            raise ValueError(
                f"balance_strategy must be one of {sorted(valid_balance)}, got '{self.balance_strategy}'"
            )
        if self.sampler_num_samples is not None and self.sampler_num_samples <= 0:
            raise ValueError(
                f"sampler_num_samples must be positive or None, got {self.sampler_num_samples}"
            )

        # Global Loss 参数验证
        if self.lambda_global < 0:
            raise ValueError(f"lambda_global must be >= 0, got {self.lambda_global}")
        if self.lambda_global_warmup_epochs < 0:
            raise ValueError(
                f"lambda_global_warmup_epochs must be >= 0, got {self.lambda_global_warmup_epochs}"
            )

        valid_sel = {"val_ce", "val_acc", "val_total"}
        if self.model_selection_metric not in valid_sel:
            raise ValueError(
                f"model_selection_metric must be one of {sorted(valid_sel)}, got '{self.model_selection_metric}'"
            )

        # Backbone 名称验证
        valid_backbones = {
            "dinov2",
            "brainsegfounder",
            "sam_brain3d",
            "voxhrnet",
            "nnunet",
            "brainfm",
        }
        if self.backbone_name not in valid_backbones:
            raise ValueError(
                f"Unknown backbone_name: '{self.backbone_name}'. "
                f"Supported: {sorted(valid_backbones)}"
            )

        # 双分支模式验证
        if self.use_dual_branch:
            if self.backbone_name1 not in valid_backbones:
                raise ValueError(
                    f"Unknown backbone_name1: '{self.backbone_name1}'. "
                    f"Supported: {sorted(valid_backbones)}"
                )
            if self.backbone_name2 not in valid_backbones:
                raise ValueError(
                    f"Unknown backbone_name2: '{self.backbone_name2}'. "
                    f"Supported: {sorted(valid_backbones)}"
                )
            # 验证训练模式
            valid_train_modes = {"none", "head", "partial"}
            if self.train_mode1 not in valid_train_modes:
                raise ValueError(
                    f"train_mode1 must be 'none', 'head', or 'partial', got '{self.train_mode1}'"
                )
            if self.train_mode2 not in valid_train_modes:
                raise ValueError(
                    f"train_mode2 must be 'none', 'head', or 'partial', got '{self.train_mode2}'"
                )
            # DoRA params sanity check (Branch 2)
            if self.use_dora_branch2:
                if self.dora_branch2_r < 0:
                    raise ValueError(
                        f"dora_branch2_r must be >= 0, got {self.dora_branch2_r}"
                    )
                if self.dora_branch2_alpha <= 0:
                    raise ValueError(
                        f"dora_branch2_alpha must be > 0, got {self.dora_branch2_alpha}"
                    )
                if not self.dora_branch2_target_modules:
                    raise ValueError("dora_branch2_target_modules must not be empty")
            # 验证融合参数
            if self.fusion_dim <= 0:
                raise ValueError(f"fusion_dim must be positive, got {self.fusion_dim}")
            if self.fusion_spatial_size <= 0:
                raise ValueError(f"fusion_spatial_size must be positive, got {self.fusion_spatial_size}")
            if self.fusion_ca_reduction <= 0:
                raise ValueError(f"fusion_ca_reduction must be positive, got {self.fusion_ca_reduction}")

        # 根据 backbone_name 自动设置 use_nnunet_zscore
        if self.use_nnunet_zscore is None:
            self.use_nnunet_zscore = self.backbone_name == "nnunet"

        if self.deterministic:
            self.cudnn_deterministic = True
            if self.cudnn_benchmark:
                self.cudnn_benchmark = False

        # 自动生成 model_save_name 和 png_save_name
        if self.model_save_name is None:
            if self.use_dual_branch:
                # 双分支模式：使用 模型1_模型2 的形式命名
                self.model_save_name = f"{self.backbone_name1}_{self.backbone_name2}_pretrain_cls.pth"
            else:
                # 单分支模式：基于 backbone_name 命名
                self.model_save_name = f"{self.backbone_name}_pretrain_cls.pth"
        if self.png_save_name is None:
            if self.use_dual_branch:
                # 双分支模式：使用 模型1_模型2 的形式命名
                self.png_save_name = f"{self.backbone_name1}_{self.backbone_name2}_pretrain_cls.png"
            else:
                # 单分支模式：基于 backbone_name 命名
                self.png_save_name = f"{self.backbone_name}_pretrain_cls.png"
