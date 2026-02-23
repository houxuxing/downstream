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


class DualBranchClassifier(nn.Module):
    """双分支分类器 - 自适应特征融合

    融合策略:
    1. VoxHRNet: Fv64=(B,112,64,64,64) -> AdaptiveAvgPool3d((4,4,4)) -> Fv4=(B,112,4,4,4)
    2. BrainSegFounder: Fb4=(B,768,4,4,4)
    3. 两者用 1x1x1 Conv 投影到同一维度 d (默认 256)
    4. 对两路特征分别过 channel attention: squeeze FC -> GELU -> restore FC
    5. M = sigmoid(Λd + Λs) 得到 element-wise mask
    6. Foutput = Fd ⊙ M + Fs ⊙ (1 - M) 进行自适应融合
    7. 最后经过 FeatureAggregator，再经过 feature_layers 和 output_layer 分类

    Args:
        backbone1: 主分支 backbone (VoxHR-Net, dim_in=112, 输出 64x64x64)
        backbone2: 副分支 backbone (BrainSegFounder, dim_in=768, 输出 4x4x4)
        num_classes: 分类类别数
        embed_dim: 分类头中间层维度
        fusion_dim: 特征融合维度 (1x1x1 Conv 投影的目标维度)
        spatial_size: 空间尺寸 (默认 4，即 4x4x4)
        reduction: channel attention 的压缩比例
    """

    def __init__(
        self,
        backbone1: nn.Module,
        backbone2: nn.Module,
        num_classes: int = 4,
        embed_dim: int = 512,
        fusion_dim: int = 256,
        spatial_size: int = 4,
        reduction: int = 4,
    ):
        super().__init__()
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.embed_dim = embed_dim
        self.fusion_dim = fusion_dim
        self.spatial_size = spatial_size

        # 获取两个 backbone 的输出通道数
        in_channels1 = getattr(backbone1, "dim_in", 112)   # VoxHRNet: 112
        in_channels2 = getattr(backbone2, "dim_in", 768)   # BrainSegFounder: 768

        print(f"[INFO] Branch1 (VoxHRNet) 输入通道数: {in_channels1}")
        print(f"[INFO] Branch2 (BrainSegFounder) 输入通道数: {in_channels2}")

        # ========== 空间对齐: 将两个分支的空间尺寸统一到 spatial_size^3 ==========
        # VoxHRNet: (B, 112, 64, 64, 64) -> (B, 112, 4, 4, 4)
        self.pool1 = nn.AdaptiveAvgPool3d((spatial_size, spatial_size, spatial_size))
        # BrainSegFounder: (B, 768, 4, 4, 4) -> (B, 768, 4, 4, 4) (已经是 4x4x4，保留以防其他尺寸)
        self.pool2 = nn.AdaptiveAvgPool3d((spatial_size, spatial_size, spatial_size))

        # ========== 1x1x1 Conv 投影到同一维度 d ==========
        self.proj1 = nn.Sequential(
            nn.Conv3d(in_channels1, fusion_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(fusion_dim),
            nn.GELU(),
        )
        self.proj2 = nn.Sequential(
            nn.Conv3d(in_channels2, fusion_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(fusion_dim),
            nn.GELU(),
        )

        # ========== Channel Attention 层 ==========
        # squeeze FC -> GELU -> restore FC
        # 输入: (B, fusion_dim, 4, 4, 4)
        # 输出: (B, fusion_dim, 1, 1, 1) 的 logits
        reduced_dim = max(fusion_dim // reduction, 16)
        
        self.ca1 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # (B, C, 1, 1, 1)
            nn.Flatten(1),            # (B, C)
            nn.Linear(fusion_dim, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, fusion_dim),
        )
        
        self.ca2 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.Linear(fusion_dim, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, fusion_dim),
        )

        print(f"[INFO] 融合维度: {fusion_dim}, 空间尺寸: {spatial_size}x{spatial_size}x{spatial_size}")
        print(f"[INFO] Channel Attention 压缩维度: {reduced_dim}")

        # ========== 融合后的 FeatureAggregator ==========
        # 融合后特征: (B, fusion_dim, spatial_size, spatial_size, spatial_size)
        self.aggregator = FeatureAggregator(fusion_dim, grid_size=2)
        aggregated_dim = self.aggregator.out_dim
        print(f"[INFO] 融合后使用 FeatureAggregator: {fusion_dim} -> {aggregated_dim}")

        # ========== 分类头 ==========
        self.feature_layers = nn.Sequential(
            nn.Linear(aggregated_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(embed_dim // 2, num_classes),
        )

        # 向后兼容：保留 classifier 属性
        self.classifier = nn.Sequential(
            self.feature_layers,
            self.output_layer,
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化所有可训练层的参数"""
        # 初始化投影层
        for proj in [self.proj1, self.proj2]:
            for m in proj.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        # 初始化 Channel Attention 层
        for ca in [self.ca1, self.ca2]:
            for m in ca.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # 初始化分类头
        for module in [self.feature_layers, self.output_layer]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """前向传播

        Args:
            x: 输入图像张量，形状 (B, C, D, H, W)
            return_features: 是否返回中间特征

        Returns:
            如果 return_features=False: logits (B, num_classes)
            如果 return_features=True: (logits, intermediate_features)
        """
        B = x.shape[0]

        # ========== Step 1: 提取两个分支的原始特征 ==========
        # VoxHRNet: (B, 112, 64, 64, 64)
        feat1_raw = self.backbone1(x)
        # BrainSegFounder: (B, 768, 4, 4, 4)
        feat2_raw = self.backbone2(x)

        # ========== Step 2: 空间对齐 ==========
        # VoxHRNet: (B, 112, 64, 64, 64) -> (B, 112, 4, 4, 4)
        feat1_pooled = self.pool1(feat1_raw)
        # BrainSegFounder: (B, 768, 4, 4, 4) -> (B, 768, 4, 4, 4)
        feat2_pooled = self.pool2(feat2_raw)

        # ========== Step 3: 1x1x1 Conv 投影到同一维度 d ==========
        # (B, 112, 4, 4, 4) -> (B, fusion_dim, 4, 4, 4)
        Fd = self.proj1(feat1_pooled)
        # (B, 768, 4, 4, 4) -> (B, fusion_dim, 4, 4, 4)
        Fs = self.proj2(feat2_pooled)

        # ========== Step 4: Channel Attention ==========
        # 得到 logits Λd 和 Λs, shape: (B, fusion_dim)
        lambda_d = self.ca1(Fd)  # (B, fusion_dim)
        lambda_s = self.ca2(Fs)  # (B, fusion_dim)

        # ========== Step 5: 计算融合 mask ==========
        # M = sigmoid(Λd + Λs), shape: (B, fusion_dim)
        M = torch.sigmoid(lambda_d + lambda_s)
        # 扩展为 5D 以便与特征图相乘: (B, fusion_dim, 1, 1, 1)
        M = M.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # ========== Step 6: 自适应融合 ==========
        # Foutput = Fd ⊙ M + Fs ⊙ (1 - M)
        F_fused = Fd * M + Fs * (1 - M)  # (B, fusion_dim, 4, 4, 4)

        # ========== Step 7: FeatureAggregator ==========
        aggregated = self.aggregator(F_fused)  # (B, aggregated_dim)

        # ========== Step 8: 分类头 ==========
        assert aggregated.shape[0] == B, (
            f"特征 batch 维度错误: 期望 {B}, 实际 {aggregated.shape[0]}"
        )

        if return_features:
            intermediate_features = self.feature_layers(aggregated)
            logits = self.output_layer(intermediate_features)
            return logits, intermediate_features
        else:
            return self.classifier(aggregated)
