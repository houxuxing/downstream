import os
import sys
import torch


import torch.nn as nn
from typing import Optional


class BrainFMBackboneWrapper(nn.Module):
    """BrainFM Encoder Wrapper (encoders only, discard decoders)

    Extracts the encoder part of UNet3D as a feature extractor.
    Outputs bottleneck feature maps (B, C, D', H', W').
    """

    def __init__(self, unet3d):
        super().__init__()
        self.encoders = unet3d.encoders  # Keep only encoders
        self.f_maps = unet3d.f_maps
        self.dim_in = self.f_maps[-1]  # Bottleneck channel count
        self.returns_3d = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for encoder in self.encoders:
            x = encoder(x)
        return x  # (B, dim_in, D', H', W')


def build_brainfm_model(
    pretrained_weights: Optional[str] = None,
    num_classes: Optional[int] = None,
    mode: str = "segmentation",
):
    """Build BrainFM model and load pretrained weights."""
    # Add BrainFM directory to sys.path to allow imports if needed
    brainfm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BrainFM")
    if brainfm_dir not in sys.path:
        sys.path.append(brainfm_dir)

    from BrainFM.load_brainfm_model import (
        BrainFMSegmentation,
        transform_state_dict_keys,
    )
    from BrainFM.unet3d import UNet3D

    if mode == "segmentation" or num_classes is not None:
        print(f"Building BrainFM model for {num_classes} classes...")
        # Initialize with pretrained=False because we will load weights manually
        model = BrainFMSegmentation(
            num_classes=num_classes if num_classes is not None else 3,
            pretrained=False,
            freeze_backbone=False,
        )

        if pretrained_weights:
            print(f"Loading BrainFM weights from: {pretrained_weights}")
            checkpoint = torch.load(
                pretrained_weights, map_location="cpu", weights_only=False
            )
            state_dict = checkpoint.get("model", checkpoint)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            # Load backbone weights
            backbone_sd = transform_state_dict_keys(state_dict, target="backbone")
            model.backbone.load_state_dict(backbone_sd, strict=False)

            # Load head weights
            head_sd = transform_state_dict_keys(state_dict, target="head")
            model_head_dict = model.head.state_dict()
            filtered_head_dict = {
                k: v
                for k, v in head_sd.items()
                if k in model_head_dict and v.shape == model_head_dict[k].shape
            }
            model.head.load_state_dict(filtered_head_dict, strict=False)

    elif mode == "backbone":
        print("Building BrainFM Backbone (UNet3D Encoder)...")
        unet3d = UNet3D(
            in_channels=1,
            f_maps=64,
            layer_order="gcl",
            num_groups=8,
            num_levels=6,
            is_unit_vector=False,
            conv_padding=1,
        )

        if pretrained_weights:
            print(f"Loading BrainFM weights from: {pretrained_weights}")
            checkpoint = torch.load(
                pretrained_weights, map_location="cpu", weights_only=False
            )
            state_dict = checkpoint.get("model", checkpoint)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            backbone_sd = transform_state_dict_keys(state_dict, target="backbone")
            missing, unexpected = unet3d.load_state_dict(backbone_sd, strict=False)
            print(f"Backbone weights loaded: missing={len(missing)}, unexpected={len(unexpected)}")

        model = BrainFMBackboneWrapper(unet3d)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    if torch.cuda.is_available():
        model.cuda()
    return model
