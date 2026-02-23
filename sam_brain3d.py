"""SAM-Brain 3D Backbone 模块

提供 SAM-Brain 3D 模型的 encoder 作为 backbone 用于下游分类任务。
只保留 image encoder 部分，移除 prompt encoder 和 mask decoder。
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Optional


class SAMBrain3DBackboneWrapper(nn.Module):
    """
    SAM-Brain 3D Backbone 包装器

    将 SAM-Brain 的 image encoder 适配为分类任务的 backbone。
    通过全局平均池化提取特征向量。

    Attributes:
        encoder: SAM-Brain 的 image encoder
        dim_in: backbone 输出的特征维度，默认为 768 (ViT-B) 或 384 (ViT-S)
    """

    def __init__(self, sam_model, feature_dim: int = 768):
        """
        Args:
            sam_model: 完整的 SAM-Brain 模型（包含 encoder, prompt_encoder, mask_decoder）
            feature_dim: 特征维度，将通过 forward 进行自动校准
        """
        super().__init__()

        # 只保留 image encoder
        self.encoder = sam_model.image_encoder

        # 特征维度（将在第一次 forward 时自动校准）
        self.dim_in = feature_dim
        self._calibrated = False

        # 标记此 backbone 返回 3D 特征图（而非 GAP 后的向量）
        self.returns_3d = True

    def _calibrate_dim(self, x: torch.Tensor):
        """通过一次 forward 自动校准特征维度"""
        with torch.no_grad():
            # 使用较小的输入进行校准（节省显存）
            B = 1
            D, H, W = x.shape[2:]
            dummy = torch.randn(B, 1, D, H, W, device=x.device, dtype=x.dtype)
            out = self.encoder(dummy)  # (B, C, d, h, w)
            actual_dim = out.shape[1]
            if actual_dim != self.dim_in:
                print(f"[SAMBrain3D] 校准特征维度: {self.dim_in} -> {actual_dim}")
                self.dim_in = actual_dim
        self._calibrated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像张量，形状 (B, C, D, H, W)

        Returns:
            3D 特征图，形状 (B, C, d, h, w)
            注意：不再进行 GAP，由 classifier 中的 FeatureAggregator 处理
        """
        # 第一次 forward 时校准特征维度
        if not self._calibrated:
            self._calibrate_dim(x)

        B = x.shape[0]

        # SAM-Brain encoder 输出: (B, C, d, h, w)
        # 其中 d, h, w 是空间维度（通常是 patch_size 的结果）
        encoder_output = self.encoder(x)

        # 返回 3D 特征图，不做 GAP
        # 全局平均池化由 classifier 中的 FeatureAggregator 处理

        # 【关键断言】确保输出形状正确（5D 特征图）
        assert encoder_output.dim() == 5, (
            f"Backbone 输出应为 5D [B, C, D, H, W], 实际为 {encoder_output.dim()}D, shape={encoder_output.shape}"
        )
        assert encoder_output.shape[0] == B, (
            f"Backbone 输出 batch 维度错误: 期望 {B}, 实际 {encoder_output.shape[0]}"
        )

        return encoder_output

    @property
    def blocks(self):
        """获取 encoder 中的 transformer blocks，用于 partial 冻结策略"""
        if hasattr(self.encoder, "blocks"):
            return self.encoder.blocks
        elif hasattr(self.encoder, "layers"):
            return self.encoder.layers
        return []


def build_sam_brain3d_model(
    pretrained_weights: Optional[str] = None, mode: str = "backbone"
):
    """构建 SAM-Brain 3D 模型

    Args:
        pretrained_weights: 预训练权重路径
        mode: 模式
            - 'backbone': 返回包装后的 backbone（只有 encoder，用于分类）
            - 'full': 返回完整模型（包含 decoder，用于分割）

    Returns:
        nn.Module: SAM-Brain 模型或其 backbone 包装器
    """
    # 添加 sam-brain3d 目录到 sys.path
    sam_brain_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sam-brain3d"
    )
    if sam_brain_dir not in sys.path:
        sys.path.insert(0, sam_brain_dir)

    # 使用 medim 库创建 SAM-Med3D 模型
    try:
        import medim
    except ImportError:
        raise ImportError(
            "medim 库未安装。请运行: pip install medim 或者按照 SAM-Brain 文档安装"
        )

    print("[INFO] 构建 SAM-Brain 3D 模型...")

    # 创建完整的 SAM-Med3D 模型
    full_model = medim.create_model("SAM-Med3D", pretrained=False)

    # 加载预训练权重
    if pretrained_weights:
        print(f"[INFO] 加载 SAM-Brain 3D 预训练权重: {pretrained_weights}")

        checkpoint = torch.load(
            pretrained_weights, map_location="cpu", weights_only=False
        )

        # 提取 state_dict
        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint.get("model")
                or checkpoint
            )
        else:
            state_dict = checkpoint

        # 处理前缀并根据模式过滤键
        new_state_dict = {}
        for k, v in state_dict.items():
            # 1. 移除 module. 前缀（DataParallel）
            if k.startswith("module."):
                k = k[7:]

            # 2. 移除 backbone.model. 或 backbone. 前缀
            if k.startswith("backbone.model."):
                k = k[15:]
            elif k.startswith("backbone."):
                k = k[9:]

            # 3. 如果是 backbone 模式，只保留 image_encoder 相关的键（实现“去除 decoder”的加载逻辑）
            if mode == "backbone":
                if k.startswith("image_encoder."):
                    new_state_dict[k] = v
            else:
                new_state_dict[k] = v

        state_dict = new_state_dict

        # 如果在 backbone 模式下需要彻底去除 decoder 架构
        if mode == "backbone":
            print("[INFO] 正在从模型中移除 prompt_encoder 和 mask_decoder...")
            if hasattr(full_model, "prompt_encoder"):
                # 将其设为 None 或者从模块中删除
                full_model.prompt_encoder = nn.Identity()  # 使用 Identity 占位或直接删除
            if hasattr(full_model, "mask_decoder"):
                full_model.mask_decoder = nn.Identity()

        # 加载权重
        msg = full_model.load_state_dict(state_dict, strict=False)
        print(f"\n{'=' * 60}")
        print("[SAM-Brain 3D 权重加载诊断]")
        print(f"{'=' * 60}")
        print(
            f"[INFO] 权重加载状态: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}"
        )

        if msg.missing_keys:
            print(f"\n[缺失的键详情] (共 {len(msg.missing_keys)} 个):")
            for k in msg.missing_keys:
                print(f"  - {k}")
        else:
            print("\n[缺失的键]: 无")

        if msg.unexpected_keys:
            print(f"\n[多余的键详情] (共 {len(msg.unexpected_keys)} 个):")
            for k in msg.unexpected_keys:
                print(f"  - {k}")
        else:
            print("\n[多余的键]: 无")
        print(f"{'=' * 60}\n")
    else:
        print("[WARNING] 未提供预训练权重，使用随机初始化")

    if mode == "backbone":
        # 包装为 backbone（只保留 encoder）
        # SAM-Med3D 的 image_encoder 通常是 ViT-B (768 dim) 或 ViT-S (384 dim)
        # 具体维度将在第一次 forward 时自动校准
        backbone = SAMBrain3DBackboneWrapper(full_model, feature_dim=768)

        # 确保所有参数可训练
        for param in backbone.parameters():
            param.requires_grad = True

        print(f"[INFO] SAM-Brain 3D Backbone 构建完成，预设特征维度: {backbone.dim_in}")
        return backbone

    elif mode == "full":
        # 返回完整模型（用于分割任务）
        print("[INFO] 返回完整 SAM-Brain 3D 模型")
        return full_model

    else:
        raise ValueError(f"未知的 mode: {mode}。支持: 'backbone', 'full'")
