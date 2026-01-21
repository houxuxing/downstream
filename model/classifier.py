"""分类器模块

包含用于下游分类任务的分类器模型。
"""

import torch
import torch.nn as nn


class FeatureProjector(nn.Module):
    """
    特征投影层：用于打破特征坍缩
    
    当预训练特征高度相似时，这个投影层可以学习一个变换，
    将特征投影到更具判别性的空间。
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
        )
        self._init_weights()
    
    def _init_weights(self):
        """初始化投影层参数
        
        NOTE: 此方法仅初始化 self.projector，不会触碰 backbone。
        """
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class AnatCLClassifier(nn.Module):
    """
    四分类的基础分类器（仅使用 CrossEntropyLoss）。

    - backbone: 特征提取器
    - feature_projector: 特征投影层（可选，用于打破特征坍缩）
    - classifier: 输出四分类 logits
    
    增强版分类头：多层 MLP + LayerNorm + Dropout
    用于处理高度相似的预训练特征
    """

    def __init__(
        self, 
        backbone: nn.Module, 
        num_classes: int = 4, 
        embed_dim: int = 512,
        use_projector: bool = True,
        projector_dim: int = 512,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.use_projector = use_projector

        # 获取 backbone 输出维度
        in_features = getattr(backbone, "dim_in", 2048)
        
        # 特征投影层：用于打破特征坍缩
        if use_projector:
            self.feature_projector = FeatureProjector(
                in_features=in_features,
                out_features=projector_dim,
                dropout=0.1,
            )
            classifier_in_features = projector_dim
        else:
            self.feature_projector = None
            classifier_in_features = in_features

        # 增强版分类头：更深更宽的网络
        # 设计原理：当特征高度相似时，需要更复杂的分类头来提取细微差异
        self.classifier = nn.Sequential(
            # 第一层：降维 + 非线性
            nn.Linear(classifier_in_features, embed_dim * 2),  # projector_dim -> 1024
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            
            # 第二层：进一步处理
            nn.Linear(embed_dim * 2, embed_dim),  # 1024 -> 512
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            
            # 第三层：瓶颈层
            nn.Linear(embed_dim, embed_dim // 2),  # 512 -> 256
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            
            # 输出层
            nn.Linear(embed_dim // 2, num_classes),  # 256 -> 4
        )

        # 初始化
        self._init_classifier()

    def _init_classifier(self):
        """初始化分类头参数
        
        NOTE: 此方法仅初始化 self.classifier，不会触碰 backbone。
        预训练的 backbone 权重通过 load_pretrained_weights() 加载，
        此初始化在 backbone 权重加载之后，因此不会覆盖预训练权重。
        """
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                # 使用较小的初始化来避免初始输出偏差过大
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        B = x.shape[0]
        
        backbone_features = self.backbone(x)
        
        # 【关键断言】确保 backbone 输出形状正确
        assert backbone_features.dim() == 2, (
            f"Backbone 输出应为 2D [B, C], 实际为 {backbone_features.dim()}D, shape={backbone_features.shape}"
        )
        assert backbone_features.shape[0] == B, (
            f"Backbone 输出 batch 维度错误: 期望 {B}, 实际 {backbone_features.shape[0]}"
        )
        
        # 应用特征投影层
        if self.use_projector and self.feature_projector is not None:
            projected_features = self.feature_projector(backbone_features)
        else:
            projected_features = backbone_features

        if return_features:
            # 通过前几层获取中间特征
            features = projected_features
            for i, layer in enumerate(self.classifier):
                features = layer(features)
                # 在第二个 Dropout 后返回特征（经过两层处理）
                if i == 7:  # 第二个 Dropout 的索引
                    break
            
            logits = self.classifier(projected_features)
            return logits, features
        else:
            return self.classifier(projected_features)
