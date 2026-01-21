"""下游任务配置模块"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Config:
    """四分类下游任务配置"""

    # 训练模式: 'none' | 'head' | 'partial'
    # - 'none': 全部可训练（默认，从头训练）
    # - 'head': 只训练分类头，冻结整个 backbone（容易导致特征坍缩）
    # - 'partial': 冻结 backbone 前面的层，解冻最后几层 + 分类头（推荐）
    train_mode: str = "partial"  # 改用 partial 模式，避免特征坍缩
    
    # partial 模式下解冻的 Transformer 块数量
    unfreeze_last_n_blocks: int = 6

    # 初始化模式: 'ckpt' | 'ckpt_adni' | 'kaiming'
    init_mode: str = "kaiming"

    # 模型配置
    # 仅支持 dinov2
    backbone_name: str = "dinov2"
    num_classes: int = 4
    embed_dim: int = 512  # 增强版分类头中间层维度

    # 3DINO
    dinov2_ckpt: Optional[str] = (
        # "/data/aim_nuist/aim_temp/aim_houxuxing/model/dinov2/3dino_vit_weights.pth"
        "/data/aim_nuist/aim_temp/aim_houxuxing/papercode/3DINO-main/output12/best_model.pth"
    )

    # 数据配置
    target_size: Tuple[int, int, int] = (112, 112, 112)  # 减小尺寸以节省内存
    batch_size: int = 32  # 减小批次大小
    num_workers: int = 2  # 减少worker数量

    # Two-view (contrastive-style) training
    two_view: bool = True

    # 内存优化配置
    gradient_accumulation_steps: int = (
        2  # 梯度累积步数，实际batch_size = batch_size * gradient_accumulation_steps
    )
    enable_memory_efficient_mode: bool = False  # 启用内存优化模式

    # 训练配置
    num_epochs: int = 100
    warmup_epochs: int = 5  # Warmup 提高训练初期稳定性
    head_lr: float = 1e-3  # 提高分类头学习率，让它更有效地学习
    backbone_lr: float = 1e-4
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0

    # 早停配置
    patience: int = 20

    # Global Loss 配置
    use_global_loss: bool = False
    lambda_global: float = 1

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
    model_save_name: str = "little_3DINO_pretrain_cls.pth"

    png_save_dir: str = "/data/aim_nuist/aim_temp/aim_houxuxing/output/anatcl/png"
    png_save_name: str = "little_3DINO_pretrain_cls.png"

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
        assert self.train_mode in {"none", "head", "partial"}
        # assert self.init_mode in {"ckpt", "ckpt_adni", "kaiming"}
