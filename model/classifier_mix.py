"""分类器模块

包含用于下游分类任务的分类器模型。
"""

import torch
import torch.nn as nn


class FeatureAggregator(nn.Module):
    """Spatially-aware feature aggregation for 3D backbones.

    This replaces pure global statistics with a lightweight projection
    plus small-grid pooling to retain some spatial structure.
    """

    def __init__(
        self,
        in_channels: int,
        grid_size: int = 2,
        proj_channels_cap: int = 256,
    ):
        super().__init__()
        if grid_size < 1:
            raise ValueError(f"grid_size must be >= 1, got {grid_size}")

        self.in_channels = int(in_channels)
        self.grid_size = int(grid_size)
        self.proj_channels = int(min(in_channels, proj_channels_cap))

        num_groups = self._choose_num_groups(self.proj_channels)
        self.proj = nn.Sequential(
            nn.Conv3d(self.in_channels, self.proj_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=self.proj_channels),
            nn.GELU(),
        )

        grid_shape = (self.grid_size, self.grid_size, self.grid_size)
        self.pool_avg = nn.AdaptiveAvgPool3d(grid_shape)
        self.pool_max = nn.AdaptiveMaxPool3d(grid_shape)

        grid_volume = self.grid_size**3
        # avg grid + max grid + global std
        self.out_dim = self.proj_channels * grid_volume * 2 + self.proj_channels

    @staticmethod
    def _choose_num_groups(channels: int, max_groups: int = 32) -> int:
        upper = min(max_groups, channels)
        for g in range(upper, 0, -1):
            if channels % g == 0:
                return g
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5, f"Expected 5D [B, C, D, H, W], got {x.dim()}D"

        x_proj = self.proj(x)

        avg_grid = self.pool_avg(x_proj).flatten(1)
        max_grid = self.pool_max(x_proj).flatten(1)
        std_global = x_proj.std(dim=(2, 3, 4), unbiased=False) + 1e-6

        aggregated = torch.cat([avg_grid, max_grid, std_global], dim=1)
        return aggregated


class AnatCLClassifier(nn.Module):
    """
    四分类的基础分类器（仅使用 CrossEntropyLoss）。

    - backbone: 特征提取器（返回 3D 特征图）
    - aggregator: 多尺度特征聚合（替代简单 GAP）
    - classifier: 输出 logits

    增强版分类头：多层 MLP + LayerNorm + Dropout
    用于处理高度相似的预训练特征
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 4,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim

        # 获取 backbone 输出通道数
        in_channels = getattr(backbone, "dim_in", 2048)

        # 检查 backbone 是否返回 3D 特征图
        self.backbone_returns_3d = getattr(backbone, "returns_3d", False)

        if self.backbone_returns_3d:
            # 使用多尺度特征聚合
            self.aggregator = FeatureAggregator(in_channels)
            in_features = self.aggregator.out_dim  # spatially-aware aggregated dim
            print(f"[INFO] 使用 FeatureAggregator: {in_channels} -> {in_features}")
        else:
            # 兼容旧版 backbone（直接返回向量）
            self.aggregator = None
            in_features = in_channels
            print(f"[WARNING] Backbone 不返回 3D 特征图，使用原始向量: {in_features}")

        # 增强版分类头：更深更宽的网络
        # 设计原理：当特征高度相似时，需要更复杂的分类头来提取细微差异
        # 分离 feature_layers 和 output_layer 以支持 return_features 而无需硬编码索引
        self.feature_layers = nn.Sequential(
            # 第一层：降维 + 非线性
            nn.Linear(in_features, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            # 第二层：进一步处理
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.output_layer = nn.Sequential(
            # 第三层：瓶颈层
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.05),
            # 输出层
            nn.Linear(embed_dim // 2, num_classes),
        )

        # 向后兼容：保留 classifier 属性（组合 feature_layers 和 output_layer）
        self.classifier = nn.Sequential(
            self.feature_layers,
            self.output_layer,
        )

        # 初始化
        self._init_classifier()

    def _init_classifier(self):
        """初始化分类头参数

        NOTE: 此方法仅初始化 self.classifier，不会触碰 backbone。
        预训练的 backbone 权重通过 load_pretrained_weights() 加载，
        此初始化在 backbone 权重加载之后，因此不会覆盖预训练权重。
        """
        # 初始化 feature_layers 和 output_layer（classifier 是它们的组合）
        for m in self.feature_layers.modules():
            if isinstance(m, nn.Linear):
                # 改用 Kaiming 初始化，适合配合 GELU/ReLU 激活函数
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.output_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        B = x.shape[0]

        # 获取 backbone 输出
        backbone_output = self.backbone(x)

        # 根据 backbone 输出类型处理
        if self.backbone_returns_3d:
            # 3D 特征图 -> 聚合为向量
            assert backbone_output.dim() == 5, (
                f"Backbone 输出应为 5D [B, C, D, H, W], 实际为 {backbone_output.dim()}D"
            )
            features = self.aggregator(backbone_output)
        else:
            # 兼容旧版：直接使用向量
            assert backbone_output.dim() == 2, (
                f"Backbone 输出应为 2D [B, C], 实际为 {backbone_output.dim()}D"
            )
            features = backbone_output

        assert features.shape[0] == B, (
            f"特征 batch 维度错误: 期望 {B}, 实际 {features.shape[0]}"
        )

        if return_features:
            # 使用分离的层结构，避免硬编码索引
            intermediate_features = self.feature_layers(features)
            logits = self.output_layer(intermediate_features)
            return logits, intermediate_features
        else:
            return self.classifier(features)
