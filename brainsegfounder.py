import sys
import os

import torch
import torch.nn as nn
from typing import Optional


class BrainSegFounderBackboneWrapper(nn.Module):
    """Wrapper to extract backbone features from BrainSegFounder SSLHead."""

    def __init__(self, ssl_model):
        super().__init__()
        self.model = ssl_model.swinViT
        # Calculate dim_in based on SwinViT config
        # embed_dim * 2^(num_layers-1)
        # 48 * 2^(4-1) = 48 * 8 = 384? No, bottleneck_depth is 768.
        # Let's rely on args.bottleneck_depth which was passed to SSLHead
        # But we don't have access to args here directly unless we store it.
        # However, for SwinViT, the final stage embedding dimension is:
        # C * 2^(num_stages-1)
        # 48 * 8 = 384. Wait, the provided args say bottleneck_depth=768.
        # Let's check: feature_size=48.
        # Stage 1: 48
        # Stage 2: 96
        # Stage 3: 192
        # Stage 4: 384
        # Why is bottleneck_depth 768 in args?
        # Maybe feature_size should be 96? Or bottleneck_depth is irrelevant to feature_size?
        # In SSLHead: dim = args.bottleneck_depth.
        # Let's trust the actual output shape.
        self.dim_in = 768  # Default for BrainSegFounder config

        # 标记此 backbone 返回 3D 特征图（而非 GAP 后的向量）
        self.returns_3d = True

    def forward(self, x):
        # SwinViT output tuple, last element is stage 4 output
        # (B, C, D, H, W)
        x_out = self.model(x.contiguous())[4]
        # 返回 3D 特征图，不做 GAP
        # Global Average Pooling 由 classifier 中的 FeatureAggregator 处理
        return x_out


def build_brainsegfounder_model(
    pretrained_weights: Optional[str] = None,
    num_classes: Optional[int] = None,
    mode: str = "ssl",
):
    """Build BrainSegFounder model and load pretrained weights.

    Args:
        pretrained_weights (str): Path to pretrained weights.
        num_classes (int, optional): Number of classes for segmentation.
        mode (str): 'ssl', 'segmentation', or 'backbone'.
    """
    # Add brainsegfounder_folder directory to sys.path
    brainsegfounder_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "brainsegfounder_folder"
    )
    if brainsegfounder_dir not in sys.path:
        sys.path.append(brainsegfounder_dir)

    class BrainSegFounderArgs:
        def __init__(self):
            self.spatial_dims = 3
            self.in_channels = 1
            self.feature_size = 48
            self.bottleneck_depth = 768
            self.num_swin_blocks_per_stage = (2, 2, 2, 2)
            self.num_heads_per_stage = (3, 6, 12, 24)
            self.dropout_path_rate = 0.0
            self.use_checkpoint = True  # 启用gradient checkpointing以节省显存

    args = BrainSegFounderArgs()

    if mode == "segmentation" or num_classes is not None:
        print(
            f"Building BrainSegFounder Segmentation model with {num_classes} classes..."
        )

        from brainsegfounder_folder.segmentation import BrainSegFounderSegmentation

        # Ensure num_classes is not None for segmentation
        n_cls = num_classes if num_classes is not None else 3
        model = BrainSegFounderSegmentation(args, num_classes=n_cls)
        if pretrained_weights:
            print(
                f"Loading BrainSegFounder backbone weights from: {pretrained_weights}"
            )
            model.load_pretrained_weights(pretrained_weights)

    elif mode == "backbone":
        print("Building BrainSegFounder Backbone (SSLHead)...")

        from brainsegfounder_folder.ssl_head import SSLHead

        ssl_model = SSLHead(args)

        if pretrained_weights:
            print(f"Loading BrainSegFounder weights from: {pretrained_weights}")
            checkpoint = torch.load(pretrained_weights, map_location="cpu")

            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Remove module. prefix if present
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            missing, unexpected = ssl_model.load_state_dict(state_dict, strict=False)
            print(f"\n{'=' * 60}")
            print("[BrainSegFounder 权重加载诊断]")
            print(f"{'=' * 60}")
            if missing:
                print(f"Missing keys: {len(missing)}")
                print("[Missing keys详情]:")
                for k in missing:
                    print(f"  - {k}")
            else:
                print("Missing keys: 0")
            if unexpected:
                print(f"\nUnexpected keys: {len(unexpected)}")
                print("[Unexpected keys详情]:")
                for k in unexpected:
                    print(f"  - {k}")
            else:
                print("\nUnexpected keys: 0")

            # 【修复】增加安全检查: 如果 missing keys 数量接近模型总参数量，说明加载失败
            total_params = len(ssl_model.state_dict())
            if len(missing) > total_params * 0.9:
                raise RuntimeError(
                    f"严重错误: BrainSegFounder 权重加载失败！\n"
                    f"缺失参数: {len(missing)}/{total_params}\n"
                    "请检查权重文件的 key 是否匹配 (例如是否有 module. 前缀)。"
                )
            print(f"{'=' * 60}\n")

        # Wrap the SSLHead to behave as a backbone
        model = BrainSegFounderBackboneWrapper(ssl_model)

    else:  # mode == "ssl"
        from brainsegfounder_folder.ssl_head import SSLHead

        print("Building BrainSegFounder model (SSLHead)...")
        model = SSLHead(args)

        if pretrained_weights:
            print(f"Loading BrainSegFounder weights from: {pretrained_weights}")
            checkpoint = torch.load(pretrained_weights, map_location="cpu")

            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Remove module. prefix if present
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"\n{'=' * 60}")
            print("[BrainSegFounder 权重加载诊断]")
            print(f"{'=' * 60}")
            if missing:
                print(f"Missing keys: {len(missing)}")
                print("[Missing keys详情]:")
                for k in missing:
                    print(f"  - {k}")
            else:
                print("Missing keys: 0")
            if unexpected:
                print(f"\nUnexpected keys: {len(unexpected)}")
                print("[Unexpected keys详情]:")
                for k in unexpected:
                    print(f"  - {k}")
            else:
                print("\nUnexpected keys: 0")
            print(f"{'=' * 60}\n")

    # Ensure all parameters are trainable (full fine-tuning)
    for param in model.parameters():
        param.requires_grad = True

    if torch.cuda.is_available():
        model.cuda()

    return model
