"""
模型构建模块

负责构建 Backbone 和分类头，支持动态选择 Backbone。
旨在将模型定义与训练逻辑解耦，便于未来扩展新的 Backbone。
"""

import torch
import torch.nn as nn

from dinov2_models.vision_transformer import vit_large_3d
from dinov2_utils.utils import load_pretrained_weights
from model.classifier import AnatCLClassifier, DualBranchClassifier
from config import Config
from brainsegfounder import build_brainsegfounder_model
from sam_brain3d import build_sam_brain3d_model
from voxhr_net import build_voxhrnet_model
from nnunet_backbone import build_nnunet_backbone
from brainfm import build_brainfm_model


def _create_backbone_by_name(
    backbone_name: str, cfg: Config, device: torch.device
) -> nn.Module:
    """根据名称创建对应的 backbone

    统一的 backbone 创建入口，支持所有已注册的 backbone 类型。
    """
    if backbone_name == "dinov2":
        return _create_dinov2_backbone(cfg, device)
    elif backbone_name == "brainsegfounder":
        return _create_brainsegfounder_backbone(cfg, device)
    elif backbone_name == "sam_brain3d":
        return _create_sam_brain3d_backbone(cfg, device)
    elif backbone_name == "voxhrnet":
        return _create_voxhrnet_backbone(cfg, device)
    elif backbone_name == "nnunet":
        return _create_nnunet_backbone(cfg, device)
    elif backbone_name == "brainfm":
        return _create_brainfm_backbone(cfg, device)
    else:
        raise ValueError(
            f"Unknown backbone_name: {backbone_name}. "
            f"Supported: ['dinov2', 'brainsegfounder', 'sam_brain3d', 'voxhrnet', 'nnunet', 'brainfm']"
        )


def _create_voxhrnet_backbone(cfg: Config, device: torch.device) -> nn.Module:
    """创建 VoxHR-Net Backbone 并加载预训练权重

    VoxHR-Net 是一个 3D 高分辨率网络 (HRNet)，适合用作医学影像分类任务的特征提取器。
    """
    backbone = build_voxhrnet_model(
        pretrained_weights=cfg.voxhrnet_ckpt, mode="backbone"
    ).to(device)

    print(
        f"\n[INFO] 使用 VoxHR-Net Backbone (HRNet-3D)，输出特征维度: {backbone.dim_in}"
    )
    return backbone


def _create_sam_brain3d_backbone(cfg: Config, device: torch.device) -> nn.Module:
    """创建 SAM-Brain 3D Backbone 并加载预训练权重

    SAM-Brain 是基于 SAM-Med3D 的医学影像模型，其 image encoder
    是一个 3D ViT，适合用作分类任务的特征提取器。
    """
    backbone = build_sam_brain3d_model(
        pretrained_weights=cfg.sam_brain3d_ckpt, mode="backbone"
    ).to(device)

    print(
        f"\n[INFO] 使用 SAM-Brain 3D Backbone (ViT-3D)，输出特征维度: {backbone.dim_in}"
    )
    return backbone


def _create_brainsegfounder_backbone(cfg: Config, device: torch.device) -> nn.Module:
    """创建 BrainSegFounder Backbone 并加载预训练权重"""

    # 直接调用 brainsegfounder.py 中的构建函数，指定 mode="backbone"
    # 该函数会返回一个包装好的、只输出特征向量的 backbone
    backbone = build_brainsegfounder_model(
        pretrained_weights=cfg.brainsegfounder_ckpt, mode="backbone"
    ).to(device)

    print(
        f"\n[INFO] 使用 BrainSegFounder Backbone (SwinViT)，输出特征维度: {backbone.dim_in}"
    )
    return backbone


def DINOv2Backbone(dinov2_model: nn.Module) -> nn.Module:
    """DINOv2 3D Vision Transformer 包装器

    将 DINOv2 的 3D ViT 适配为分类任务的 backbone。
    返回 3D 特征图（从 patch tokens 重塑而来），由 classifier 进行聚合。
    """

    class _Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            # DINOv2 ViT-Large 的 embed_dim 为 1024
            self.dim_in = 1024

            # 标记此 backbone 返回 3D 特征图（而非 GAP 后的向量）
            self.returns_3d = True

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: 输入图像张量，形状 (B, C, D, H, W)

            Returns:
                3D 特征图，形状 (B, embed_dim, d, h, w)
                其中 d, h, w = D/patch_size, H/patch_size, W/patch_size
            """
            B = x.shape[0]
            _, _, D, H, W = x.shape
            patch_size_attr = getattr(self.model, "patch_size", 16)
            if isinstance(patch_size_attr, (tuple, list)):
                patch_size = int(patch_size_attr[0])
            else:
                patch_size = int(patch_size_attr)

            # forward_features 返回字典，包含:
            # - "x_norm_clstoken": (B, embed_dim) - CLS token
            # - "x_norm_patchtokens": (B, N, embed_dim) - patch tokens
            features = self.model.forward_features(x)

            patch_tokens = features["x_norm_patchtokens"]  # (B, N, embed_dim)

            # 使用 patch tokens 并重塑为 3D 空间特征图

            # 计算空间维度
            d = D // patch_size
            h = H // patch_size
            w = W // patch_size
            expected_tokens = d * h * w
            actual_tokens = patch_tokens.shape[1]
            if actual_tokens != expected_tokens:
                raise RuntimeError(
                    f"Patch token 数量不匹配: 期望 {expected_tokens}, 实际 {actual_tokens}。"
                    f"请检查输入尺寸是否能被 patch_size 整除"
                )

            # 重塑为 3D 空间格式: (B, N, C) -> (B, d, h, w, C) -> (B, C, d, h, w)
            spatial_features = patch_tokens.view(B, d, h, w, -1).permute(0, 4, 1, 2, 3)

            # 【关键断言】确保输出形状正确
            assert spatial_features.dim() == 5, (
                f"输出应为 5D [B, C, D, H, W], 实际为 {spatial_features.dim()}D, shape={spatial_features.shape}"
            )
            assert spatial_features.shape[0] == B, (
                f"Batch 维度错误: 期望 {B}, 实际 {spatial_features.shape[0]}"
            )

            return spatial_features

    return _Wrapper(dinov2_model)


def _create_dinov2_backbone(cfg: Config, device: torch.device) -> nn.Module:
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
        # strict=False 允许加载包含 decoder 等多余参数的权重文件
        load_pretrained_weights(
            model, cfg.dinov2_ckpt, checkpoint_key="teacher", strict=False
        )

    # 创建包装器
    backbone = DINOv2Backbone(model).to(device)

    print(
        f"\n[INFO] 使用 DINOv2 Backbone (ViT-Large-3D)，输出特征维度: {backbone.dim_in}"
    )
    return backbone


def _create_nnunet_backbone(cfg: Config, device: torch.device) -> nn.Module:
    """创建 nnUNet Backbone 并加载预训练权重

    nnUNet 是一个自适应的医学影像分割框架，其 encoder 适合用作分类任务的特征提取器。
    根据 nnUNetPlans.json 配置构建 PlainConvUNet encoder。
    """
    import os

    # 获取配置文件路径
    plans_json = cfg.nnunet_plans_json
    if not os.path.isabs(plans_json):
        # 如果是相对路径，相对于项目目录
        plans_json = os.path.join(os.path.dirname(__file__), plans_json)

    backbone = build_nnunet_backbone(
        plans_json=plans_json, pretrained_weights=cfg.nnunet_ckpt, mode="backbone"
    ).to(device)

    print(
        f"\n[INFO] 使用 nnUNet Backbone (PlainConvUNet)，输出特征维度: {backbone.dim_in}"
    )
    return backbone


def _create_brainfm_backbone(cfg: Config, device: torch.device) -> nn.Module:
    """创建 BrainFM Backbone 并加载预训练权重"""

    # 直接调用 brainfm.py 中的构建函数，指定 mode="backbone"
    # 该函数会返回一个包装好的、只输出 encoder 特征的 backbone
    backbone = build_brainfm_model(
        pretrained_weights=cfg.brainfm_ckpt, mode="backbone"
    ).to(device)

    print(
        f"\n[INFO] 使用 BrainFM Backbone (UNet3D Encoder)，输出特征维度: {backbone.dim_in}"
    )
    return backbone


@torch.no_grad()
def _calibrate_backbone_channels(
    backbone: nn.Module, cfg: Config, device: torch.device
) -> nn.Module:
    if not getattr(backbone, "returns_3d", False):
        return backbone
    if getattr(backbone, "_channels_calibrated", False):
        return backbone

    original_mode = backbone.training
    backbone.eval()
    try:
        patch_size_attr = getattr(
            getattr(backbone, "model", backbone), "patch_size", 16
        )
        if isinstance(patch_size_attr, (tuple, list)):
            patch_size = int(patch_size_attr[0])
        else:
            patch_size = int(patch_size_attr)

        min_side = int(min(cfg.target_size))
        aligned = max(patch_size, (min_side // patch_size) * patch_size)
        
        # 针对 3D ViT 模型（如 SAM-Brain, DINOv2），强制 64x64x64 可能会导致 pos_embed 冲突
        # 使用 aligned 尺寸，但上限设为 128 以兼顾内存和兼容性
        test_side = int(min(128, aligned))

        dummy = torch.zeros(1, 1, test_side, test_side, test_side, device=device)
        out = backbone(dummy)
        if out.dim() != 5:
            raise RuntimeError(f"Backbone ??? 5D ???, ?? shape={tuple(out.shape)}")

        actual_dim = int(out.shape[1])
        if getattr(backbone, "dim_in", actual_dim) != actual_dim:
            print(
                f"[CALIB] backbone.dim_in: {getattr(backbone, 'dim_in', '<?>')} -> {actual_dim}"
            )
            backbone.dim_in = actual_dim

        if hasattr(backbone, "_calibrated"):
            backbone._calibrated = True
        backbone._channels_calibrated = True
        return backbone
    finally:
        if original_mode:
            backbone.train()


def create_model(cfg: Config, device: torch.device) -> AnatCLClassifier:
    """创建并初始化模型（统一入口）

    根据配置中的 backbone_name 动态选择 Backbone，并构建分类器。
    """
    # 1. 构建 Backbone
    backbone = _create_backbone_by_name(cfg.backbone_name, cfg, device)

    # 2. 校准 Backbone 通道数（特别是返回 3D 特征图的 backbone）
    backbone = _calibrate_backbone_channels(backbone, cfg, device)

    # 3. 构建分类器（线性层在 AnatCLClassifier 中定义）
    model = AnatCLClassifier(
        backbone,
        num_classes=cfg.num_classes,
        embed_dim=cfg.embed_dim,
    ).to(device)

    # 3. 返回模型
    # 注意：AnatCLClassifier.__init__ 已内置完成分类头的 Kaiming 初始化
    return model


def create_dual_branch_model(
    cfg: Config, device: torch.device
) -> DualBranchClassifier:
    """创建双分支模型

    根据配置中的 backbone_name1 和 backbone_name2 创建两个 backbone，
    并使用 DualBranchClassifier 进行自适应特征融合和分类。

    Args:
        cfg: 配置对象，需要包含 backbone_name1, backbone_name2, fusion_dim 等参数
        device: 设备

    Returns:
        DualBranchClassifier 实例
    """
    print("\n" + "=" * 60)
    print("[INFO] 创建双分支模型 (自适应融合)")
    print("=" * 60)

    # 1. 创建主分支 backbone
    print(f"\n[Branch 1] 创建主分支: {cfg.backbone_name1}")
    backbone1 = _create_backbone_by_name(cfg.backbone_name1, cfg, device)
    backbone1 = _calibrate_backbone_channels(backbone1, cfg, device)

    # 2. 创建副分支 backbone
    print(f"\n[Branch 2] 创建副分支: {cfg.backbone_name2}")
    backbone2 = _create_backbone_by_name(cfg.backbone_name2, cfg, device)
    backbone2 = _calibrate_backbone_channels(backbone2, cfg, device)

    # 3. 构建双分支分类器
    print(f"\n[INFO] 自适应融合配置: fusion_dim={cfg.fusion_dim}, "
          f"spatial_size={cfg.fusion_spatial_size}, ca_reduction={cfg.fusion_ca_reduction}")
    model = DualBranchClassifier(
        backbone1,
        backbone2,
        num_classes=cfg.num_classes,
        embed_dim=cfg.embed_dim,
        fusion_dim=cfg.fusion_dim,
        spatial_size=cfg.fusion_spatial_size,
        reduction=cfg.fusion_ca_reduction,
    ).to(device)

    print("=" * 60 + "\n")
    return model
