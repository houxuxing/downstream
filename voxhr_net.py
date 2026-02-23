"""VoxHR-Net Backbone 模块

提供 VoxHR-Net 模型作为 backbone 用于下游分类任务。
移除分割头（last_layer），只保留特征提取部分。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from VoxHR_Net.voxhrnet import HighResolutionNet, blocks_dict


class VoxHRNetConfig:
    """
    VoxHR-Net 模型配置

    与预训练项目 (config_lpba.yaml) 保持一致的模型架构配置。
    """

    class DatasetConfig:
        def __init__(self, num_classes: int):
            self.NUM_CLASSES = num_classes

    class ModelConfig:
        def __init__(self):
            self.EXTRA: Dict[str, Any] = {
                "STAGE2": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 2,
                    "NUM_BLOCKS": [3, 3],
                    "NUM_CHANNELS": [16, 32],
                    "BLOCK": "BASIC",
                },
                "STAGE3": {
                    "NUM_MODULES": 2,
                    "NUM_BRANCHES": 3,
                    "NUM_BLOCKS": [3, 3, 3],
                    "NUM_CHANNELS": [16, 32, 64],
                    "BLOCK": "BASIC",
                },
                # STAGE4 未启用，与预训练配置保持一致
            }

    def __init__(self, num_classes: int = 4):
        self.DATASET = self.DatasetConfig(num_classes)
        self.MODEL = self.ModelConfig()


class VoxHRNetBackboneWrapper(nn.Module):
    """
    VoxHR-Net Backbone 包装器

    将 VoxHR-Net 适配为分类任务的 backbone。
    移除分割头（last_layer），通过全局平均池化提取特征向量。

    Attributes:
        model: 原始 VoxHR-Net 模型（已移除 last_layer）
        dim_in: backbone 输出的特征维度
    """

    def __init__(
        self, hrnet_model: HighResolutionNet, feature_dim: Optional[int] = None
    ):
        """
        Args:
            hrnet_model: 原始 HighResolutionNet 模型
            feature_dim: 特征维度，如果为 None 则自动计算
        """
        super().__init__()

        self.model: HighResolutionNet = hrnet_model

        # 计算特征维度
        # VoxHR-Net concat 所有分支的特征，维度 = sum(all branch channels)
        if feature_dim is not None:
            self.dim_in = feature_dim
        else:
            # 从配置中计算
            self.dim_in = self._compute_feature_dim()

        self._calibrated = False

        # 标记此 backbone 返回 3D 特征图（而非 GAP 后的向量）
        self.returns_3d = True

    def _compute_feature_dim(self) -> int:
        """计算特征维度

        根据最后一个 stage 的配置计算特征维度。
        VoxHR-Net 将所有分支的特征 concat，维度 = sum(all branch channels * expansion)

        预训练配置 (STAGE3): [16, 32, 64] * 1 (BASIC expansion) = 112
        """
        # 尝试从模型配置中获取
        extra = getattr(self.model, "extra", None)
        if extra is None:
            # 默认值：与预训练配置一致 (STAGE3: [16, 32, 64])
            # channels: [16, 32, 64] * expansion(1 for BASIC) = 112
            return 112

        # 获取最后一个 stage 的配置
        if "STAGE4" in extra:
            stage_cfg = extra["STAGE4"]
        elif "STAGE3" in extra:
            stage_cfg = extra["STAGE3"]
        else:
            stage_cfg = extra["STAGE2"]

        # 计算总通道数
        block = blocks_dict[stage_cfg["BLOCK"]]
        num_channels = stage_cfg["NUM_CHANNELS"]
        total_channels = sum([ch * block.expansion for ch in num_channels])

        return total_channels

    def _calibrate_dim(self, x: torch.Tensor):
        """通过一次 forward 自动校准特征维度"""
        with torch.no_grad():
            B = 1
            D, H, W = x.shape[2:]
            # 使用较小的输入进行校准
            min_size = 32
            test_size = max(min_size, min(D, H, W) // 2)
            dummy = torch.randn(
                B, 1, test_size, test_size, test_size, device=x.device, dtype=x.dtype
            )

            # 执行前向传播到 concat 层
            features = self._forward_features(dummy)
            actual_dim = features.shape[1]

            if actual_dim != self.dim_in:
                print(f"[VoxHRNet] 校准特征维度: {self.dim_in} -> {actual_dim}")
                self.dim_in = actual_dim

        self._calibrated = True

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播到特征 concat 层（不包括 last_layer）"""
        # stem_net
        x = self.model.stem_net(x)

        # Stage 2
        x_list = [
            x if self.model.transition1[i] is None else self.model.transition1[i](x)
            for i in range(self.model.stage2_cfg["NUM_BRANCHES"])
        ]
        y_list = self.model.stage2(x_list)

        # Stage 3
        x_list = [
            y_list[i]
            if self.model.transition2[i] is None
            else self.model.transition2[i](
                y_list[i if i < self.model.stage2_cfg["NUM_BRANCHES"] else -1]
            )
            for i in range(self.model.stage3_cfg["NUM_BRANCHES"])
        ]
        y_list = self.model.stage3(x_list)

        # Stage 4 (如果存在)
        if hasattr(self.model, "stage4") and "STAGE4" in self.model.extra:
            x_list = [
                y_list[i]
                if self.model.transition3[i] is None
                else self.model.transition3[i](
                    y_list[i if i < self.model.stage3_cfg["NUM_BRANCHES"] else -1]
                )
                for i in range(self.model.stage4_cfg["NUM_BRANCHES"])
            ]
            y_list = self.model.stage4(x_list)

        # 将所有分辨率的特征上采样到最高分辨率并 concat
        x0_shape = y_list[0].shape[-3:]
        for i in range(1, len(y_list)):
            y_list[i] = F.interpolate(
                y_list[i], size=x0_shape, mode="trilinear", align_corners=False
            )

        # concat 所有分支特征
        features = torch.cat(y_list, 1)  # (B, C, D, H, W)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像张量，形状 (B, C, D, H, W)

        Returns:
            3D 特征图，形状 (B, C, D, H, W)
            注意：不再进行 GAP，由 classifier 中的 FeatureAggregator 处理
        """
        # 第一次 forward 时校准特征维度
        if not self._calibrated:
            self._calibrate_dim(x)

        B = x.shape[0]

        # 提取特征（返回 3D 特征图，不做 GAP）
        features = self._forward_features(x)  # (B, C, D, H, W)

        # 【关键断言】确保输出形状正确（5D 特征图）
        assert features.dim() == 5, (
            f"Backbone 输出应为 5D [B, C, D, H, W], 实际为 {features.dim()}D, shape={features.shape}"
        )
        assert features.shape[0] == B, (
            f"Backbone 输出 batch 维度错误: 期望 {B}, 实际 {features.shape[0]}"
        )

        return features

    @property
    def stages(self):
        """获取模型的 stages 列表，用于 partial 冻结策略

        预训练配置 (STAGE3): [stem_net, stage2, stage3] = 3 个 stages
        如果启用 STAGE4: [stem_net, stage2, stage3, stage4] = 4 个 stages
        """
        stages = [self.model.stem_net, self.model.stage2, self.model.stage3]
        if hasattr(self.model, "stage4"):
            stages.append(self.model.stage4)
        return stages

    @property
    def transitions(self):
        """获取模型的 transition layers

        预训练配置 (STAGE3): [transition1, transition2] = 2 个 transitions
        如果启用 STAGE4: [transition1, transition2, transition3] = 3 个 transitions
        """
        transitions = [self.model.transition1, self.model.transition2]
        if hasattr(self.model, "transition3"):
            transitions.append(self.model.transition3)
        return transitions


def build_voxhrnet_model(
    pretrained_weights: Optional[str] = None,
    mode: str = "backbone",
    config: Optional[VoxHRNetConfig] = None,
):
    """构建 VoxHR-Net 模型

    Args:
        pretrained_weights: 预训练权重路径
        mode: 模式
            - 'backbone': 返回包装后的 backbone（移除分割头，用于分类）
            - 'full': 返回完整模型（包含分割头）
        config: 模型配置，如果为 None 则使用默认配置

    Returns:
        Union[HighResolutionNet, VoxHRNetBackboneWrapper]: VoxHR-Net 模型或其 backbone 包装器
    """

    # 使用默认配置
    if config is None:
        config = VoxHRNetConfig(num_classes=4)

    print("[INFO] 构建 VoxHR-Net 模型...")

    # 创建 HighResolutionNet 模型
    full_model = HighResolutionNet(config, in_channels=1)

    # 加载预训练权重
    if pretrained_weights:
        print(f"[INFO] 加载 VoxHR-Net 预训练权重: {pretrained_weights}")

        checkpoint = torch.load(
            pretrained_weights, map_location="cpu", weights_only=False
        )

        # 提取 state_dict
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
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
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

        # 加载权重
        # strict=False 仍会在 shape 不匹配时报错，因此先过滤不匹配的权重
        model_state = full_model.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        for k, v in state_dict.items():
            # backbone 模式下不加载 last_layer（通常是分割头）
            if mode == "backbone" and k.startswith("last_layer."):
                skipped_keys.append((k, "skip last_layer in backbone mode"))
                continue
            if k not in model_state:
                skipped_keys.append((k, "unexpected key"))
                continue
            if v.shape != model_state[k].shape:
                skipped_keys.append(
                    (
                        k,
                        f"shape mismatch: ckpt {tuple(v.shape)} vs model {tuple(model_state[k].shape)}",
                    )
                )
                continue
            filtered_state_dict[k] = v

        msg = full_model.load_state_dict(filtered_state_dict, strict=False)
        print(f"\n{'=' * 60}")
        print("[VoxHR-Net 权重加载诊断]")
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
        if skipped_keys:
            print(f"\n[Skipped keys] (total {len(skipped_keys)}):")
            for k, reason in skipped_keys:
                print(f"  - {k} ({reason})")
        print(f"{'=' * 60}\n")
    else:
        print("[WARNING] 未提供预训练权重，使用随机初始化")

    if mode == "backbone":
        # 包装为 backbone（移除分割头）
        backbone = VoxHRNetBackboneWrapper(full_model)

        # 确保所有参数可训练
        for param in backbone.parameters():
            param.requires_grad = True

        print(f"[INFO] VoxHR-Net Backbone 构建完成，预设特征维度: {backbone.dim_in}")
        return backbone

    elif mode == "full":
        # 返回完整模型（用于分割任务）
        print("[INFO] 返回完整 VoxHR-Net 模型")
        return full_model

    else:
        raise ValueError(f"未知的 mode: {mode}。支持: 'backbone', 'full'")
