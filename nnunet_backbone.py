"""nnUNet Backbone 模块

提供 nnUNet 预训练模型的 encoder 作为 backbone 用于下游分类任务。
根据 nnUNetPlans.json 配置构建 PlainConvUNet encoder，移除 decoder 部分。

特点:
- 从 nnUNetPlans.json 动态解析模型配置
- 使用官方 dynamic_network_architectures 库构建模型（确保权重兼容）
- 只保留 encoder 部分（用于特征提取）
- 支持加载预训练分割模型权重
"""

import json
import os
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

# 使用官方 nnUNet 依赖库
from dynamic_network_architectures.architectures.unet import PlainConvUNet


class nnUNetEncoderBackbone(nn.Module):
    """nnUNet Encoder Backbone 包装器

    将 nnUNet 的 PlainConvUNet encoder 适配为分类任务的 backbone。
    返回 3D 特征图，由 classifier 中的 FeatureAggregator 进行聚合。

    Attributes:
        encoder: nnUNet encoder 模块
        dim_in: backbone 输出的特征维度（最后阶段的通道数，如 320）
        returns_3d: 标记此 backbone 返回 3D 特征图
    """

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int = 320,
    ):
        """
        Args:
            encoder: PlainConvUNet 的 encoder 部分
            feature_dim: 特征维度（encoder 最后阶段的输出通道数）
        """
        super().__init__()
        self.encoder = encoder
        self.dim_in = feature_dim

        # 标记此 backbone 返回 3D 特征图（而非 GAP 后的向量）
        self.returns_3d = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像张量，形状 (B, C, D, H, W)

        Returns:
            3D 特征图，形状 (B, C, d, h, w)
            注意：不再进行 GAP，由 classifier 中的 FeatureAggregator 处理
        """
        B = x.shape[0]

        # encoder 输出可能是列表（多尺度特征）或单个张量
        encoder_output = self.encoder(x)

        # 如果输出是列表，取最后（最深层）的特征图
        if isinstance(encoder_output, (list, tuple)):
            features = encoder_output[-1]  # 最深层: (B, C, d, h, w)
        else:
            features = encoder_output

        # 返回 3D 特征图，不做 GAP
        # 【关键断言】确保输出形状正确
        assert features.dim() == 5, (
            f"Backbone 输出应为 5D [B, C, D, H, W], 实际为 {features.dim()}D, shape={features.shape}"
        )
        assert features.shape[0] == B, (
            f"Backbone 输出 batch 维度错误: 期望 {B}, 实际 {features.shape[0]}"
        )

        return features


def parse_nnunet_plans(
    plans_json_path: str, config_name: str = "3d_fullres"
) -> Dict[str, Any]:
    """解析 nnUNetPlans.json 配置文件

    Args:
        plans_json_path: nnUNetPlans.json 文件路径
        config_name: 配置名称，如 "3d_fullres" 或 "2d"

    Returns:
        解析后的配置字典
    """
    with open(plans_json_path, "r", encoding="utf-8") as f:
        plans = json.load(f)

    if config_name not in plans["configurations"]:
        available = list(plans["configurations"].keys())
        raise ValueError(f"配置 '{config_name}' 不存在。可用配置: {available}")

    config = plans["configurations"][config_name]
    arch = config["architecture"]
    arch_kwargs = arch["arch_kwargs"]

    return {
        "n_stages": arch_kwargs["n_stages"],
        "features_per_stage": arch_kwargs["features_per_stage"],
        "kernel_sizes": arch_kwargs["kernel_sizes"],
        "strides": arch_kwargs["strides"],
        "n_conv_per_stage": arch_kwargs["n_conv_per_stage"],
        "n_conv_per_stage_decoder": arch_kwargs["n_conv_per_stage_decoder"],
        "conv_bias": arch_kwargs["conv_bias"],
        "norm_op_kwargs": arch_kwargs["norm_op_kwargs"],
        "nonlin_kwargs": arch_kwargs["nonlin_kwargs"],
        "patch_size": config["patch_size"],
        "normalization_schemes": config["normalization_schemes"],
        # 解析 conv_op 类型
        "is_3d": "Conv3d" in arch_kwargs["conv_op"],
    }


def build_nnunet_backbone(
    plans_json: str = "nnUNetPlans.json",
    pretrained_weights: Optional[str] = None,
    config_name: str = "3d_fullres",
    mode: str = "backbone",
) -> nn.Module:
    """构建 nnUNet Backbone

    使用官方 dynamic_network_architectures 库构建 PlainConvUNet，
    确保与预训练权重完全兼容。

    Args:
        plans_json: nnUNetPlans.json 配置文件路径
        pretrained_weights: 预训练权重路径（可选）
        config_name: 配置名称，默认 "3d_fullres"
        mode: 模式
            - 'backbone': 返回包装后的 backbone（用于分类）
            - 'encoder': 返回原始 encoder（用于自定义）

    Returns:
        nn.Module: nnUNet encoder 或其 backbone 包装器
    """
    print(f"[INFO] 解析 nnUNet 配置: {plans_json}")

    # 解析配置
    config = parse_nnunet_plans(plans_json, config_name)

    print("[INFO] nnUNet 配置:")
    print(f"  - stages: {config['n_stages']}")
    print(f"  - features: {config['features_per_stage']}")
    print(f"  - patch_size: {config['patch_size']}")
    print(f"  - normalization: {config['normalization_schemes']}")
    print(f"  - is_3d: {config['is_3d']}")

    # 根据配置选择 conv_op 和 norm_op
    if config["is_3d"]:
        conv_op = nn.Conv3d
        norm_op = nn.InstanceNorm3d
    else:
        conv_op = nn.Conv2d
        norm_op = nn.InstanceNorm2d

    # 使用官方 PlainConvUNet 构建完整的 U-Net 模型
    # 然后只提取 encoder 部分
    full_unet = PlainConvUNet(
        input_channels=1,  # 医学影像通常为单通道
        n_stages=config["n_stages"],
        features_per_stage=config["features_per_stage"],
        conv_op=conv_op,
        kernel_sizes=config["kernel_sizes"],
        strides=config["strides"],
        n_conv_per_stage=config["n_conv_per_stage"],
        num_classes=2,  # 临时值，仅用于构建（不会使用 decoder）
        n_conv_per_stage_decoder=config["n_conv_per_stage_decoder"],
        conv_bias=config["conv_bias"],
        norm_op=norm_op,
        norm_op_kwargs=config["norm_op_kwargs"],
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=config["nonlin_kwargs"],
        deep_supervision=False,
    )

    # 提取 encoder 部分
    encoder = full_unet.encoder

    # 特征维度是最后阶段的通道数
    feature_dim = config["features_per_stage"][-1]

    # 加载预训练权重
    if pretrained_weights:
        print(f"[INFO] 加载 nnUNet 预训练权重: {pretrained_weights}")

        checkpoint = torch.load(
            pretrained_weights, map_location="cpu", weights_only=False
        )

        # 提取 state_dict
        if isinstance(checkpoint, dict):
            if "network_weights" in checkpoint:
                state_dict = checkpoint["network_weights"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # 处理 'module.' 前缀（来自 DataParallel）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        state_dict = new_state_dict

        # 过滤只保留 encoder 的权重
        # nnUNet 的 encoder 权重以 "encoder." 开头
        encoder_state_dict = {}
        decoder_keys = []

        for k, v in state_dict.items():
            if k.startswith("encoder."):
                # 移除 "encoder." 前缀以匹配 encoder 模块
                new_key = k[8:]
                encoder_state_dict[new_key] = v
            else:
                decoder_keys.append(k)

        # 加载权重
        if encoder_state_dict:
            msg = encoder.load_state_dict(encoder_state_dict, strict=False)
            print(f"\n{'=' * 60}")
            print("[nnUNet Encoder 权重加载诊断]")
            print(f"{'=' * 60}")
            print(
                f"[INFO] Encoder 权重: loaded={len(encoder_state_dict)}, "
                f"missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}"
            )

            if msg.missing_keys:
                print(f"\n[缺失的键] (共 {len(msg.missing_keys)} 个):")
                for k in msg.missing_keys[:10]:
                    print(f"  - {k}")
                if len(msg.missing_keys) > 10:
                    print(f"  ... 还有 {len(msg.missing_keys) - 10} 个")

            if msg.unexpected_keys:
                print(f"\n[未预期的键] (共 {len(msg.unexpected_keys)} 个):")
                for k in msg.unexpected_keys[:5]:
                    print(f"  - {k}")
                if len(msg.unexpected_keys) > 5:
                    print(f"  ... 还有 {len(msg.unexpected_keys) - 5} 个")

            if decoder_keys:
                print(f"\n[跳过的 Decoder 键] (共 {len(decoder_keys)} 个):")
                for k in decoder_keys[:5]:
                    print(f"  - {k}")
                if len(decoder_keys) > 5:
                    print(f"  ... 还有 {len(decoder_keys) - 5} 个")

            # 验证加载成功
            if len(msg.missing_keys) == 0 and len(msg.unexpected_keys) == 0:
                print("\n[SUCCESS] 所有 encoder 权重已成功加载!")
            elif len(msg.missing_keys) == 0:
                print("\n[SUCCESS] 所有 encoder 权重已加载（有额外的键被忽略）")
            else:
                print(f"\n[WARNING] 有 {len(msg.missing_keys)} 个参数未能加载")

            print(f"{'=' * 60}\n")
        else:
            print("[WARNING] 未找到匹配的 encoder 权重，请检查 checkpoint 格式")
    else:
        print("[WARNING] 未提供预训练权重，使用随机初始化")

    if mode == "backbone":
        # 包装为 backbone
        backbone = nnUNetEncoderBackbone(encoder, feature_dim=feature_dim)

        # 确保所有参数可训练
        for param in backbone.parameters():
            param.requires_grad = True

        print(f"[INFO] nnUNet Backbone 构建完成，特征维度: {backbone.dim_in}")
        return backbone

    elif mode == "encoder":
        print(f"[INFO] 返回原始 nnUNet Encoder，特征维度: {feature_dim}")
        return encoder

    else:
        raise ValueError(f"未知的 mode: {mode}。支持: 'backbone', 'encoder'")


# 测试代码
if __name__ == "__main__":
    import os

    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plans_json = os.path.join(script_dir, "nnUNetPlans.json")
    checkpoint_path = os.path.join(script_dir, "checkpoint_best.pth")

    if os.path.exists(plans_json):
        print("=" * 60)
        print("nnUNet Backbone 构建测试")
        print("=" * 60)

        # 测试加载预训练权重
        if os.path.exists(checkpoint_path):
            backbone = build_nnunet_backbone(
                plans_json=plans_json,
                pretrained_weights=checkpoint_path,
                mode="backbone",
            )
        else:
            print(f"[WARNING] 未找到 checkpoint: {checkpoint_path}")
            backbone = build_nnunet_backbone(
                plans_json=plans_json, pretrained_weights=None, mode="backbone"
            )

        # 测试前向传播
        x = torch.randn(2, 1, 128, 128, 128)
        print(f"\n输入形状: {x.shape}")

        with torch.no_grad():
            out = backbone(x)

        print(f"输出形状: {out.shape}")
        print(f"期望特征维度: {backbone.dim_in}")

        # backbone 现在返回 3D 特征图，不是向量
        assert out.dim() == 5, "输出应为 5D 特征图!"
        assert out.shape[1] == backbone.dim_in, "通道数不匹配!"
        print("\n[OK] 测试通过!")
    else:
        print(f"[ERROR] 配置文件不存在: {plans_json}")
