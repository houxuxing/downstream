"""
BrainFM 3D U-Net Model Loader

This module provides utilities to load the pretrained BrainFM 3D U-Net model
from the brainfm_pretrained.pth checkpoint file.

The pretrained model is a multi-task model with:
- A shared UNet3D backbone
- Multiple task-specific heads (T1, T2, CT, distance, registration, segmentation)

Usage:
    from load_brainfm_model import load_brainfm_unet3d, BrainFMSegmentation

    # Load only the backbone (for feature extraction)
    backbone = load_brainfm_unet3d()

    # Load full segmentation model (backbone + segmentation head)
    model = BrainFMSegmentation(num_classes=3, pretrained=True)
"""

import os
import torch
import torch.nn as nn
from pathlib import Path

# Import UNet3D from the same directory
try:
    from .unet3d import UNet3D
except ImportError:  # 兼容直接运行该脚本的情况
    from unet3d import UNet3D


def get_checkpoint_path() -> Path:
    """Get the path to the pretrained checkpoint file."""
    current_dir = Path(__file__).parent
    checkpoint_path = current_dir / "brainfm_pretrained.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at: {checkpoint_path}\n"
            "Please ensure 'brainfm_pretrained.pth' is in the same directory."
        )

    return checkpoint_path


def transform_state_dict_keys(state_dict: dict, target: str = "backbone") -> dict:
    """
    Transform state_dict keys to match the current model structure.

    The pretrained checkpoint uses:
        - backbone.encoders.0.basic_SingleConv1...

    But the current UNet3D model expects:
        - encoders.0.basic_module.SingleConv1...

    Args:
        state_dict: Original state dict from checkpoint
        target: Which part to extract ("backbone" or "head")

    Returns:
        Transformed state dict with corrected key names
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        if target == "backbone":
            # Only process backbone keys
            if not key.startswith("backbone."):
                continue

            # Remove 'backbone.' prefix
            new_key = key.replace("backbone.", "", 1)

            # Fix module naming: basic_SingleConv -> basic_module.SingleConv
            new_key = new_key.replace("basic_SingleConv", "basic_module.SingleConv")

            new_state_dict[new_key] = value

        elif target == "head":
            # Only process head keys
            if not key.startswith("head."):
                continue

            # Keep head keys as-is (without 'head.' prefix for the head module)
            new_key = key.replace("head.", "", 1)
            new_state_dict[new_key] = value

    return new_state_dict


class BrainFMHead(nn.Module):
    """
    Multi-task segmentation head for BrainFM model.

    This head contains multiple output branches for different tasks:
    - T1, T2, CT: Modality-specific outputs
    - distance: Distance transform output
    - registration: Registration field output
    - segmentation: Semantic segmentation output
    """

    def __init__(self, in_channels: int = 64, num_classes: int = 3):
        super().__init__()

        # All final convolutions: in_channels -> num_classes (1x1x1 conv)
        self.final_conv_T1 = nn.Conv3d(in_channels, num_classes, kernel_size=1)
        self.final_conv_T2 = nn.Conv3d(in_channels, num_classes, kernel_size=1)
        self.final_conv_CT = nn.Conv3d(in_channels, num_classes, kernel_size=1)
        self.final_conv_distance = nn.Conv3d(in_channels, num_classes, kernel_size=1)
        self.final_conv_registration = nn.Conv3d(
            in_channels, num_classes, kernel_size=1
        )
        self.final_conv_segmentation = nn.Conv3d(
            in_channels, num_classes, kernel_size=1
        )

    def forward(self, x, task: str = "segmentation"):
        """
        Forward pass for a specific task.

        Args:
            x: Input features from backbone (B, C, D, H, W)
            task: Which task head to use

        Returns:
            Task-specific output
        """
        if task == "T1":
            return self.final_conv_T1(x)
        elif task == "T2":
            return self.final_conv_T2(x)
        elif task == "CT":
            return self.final_conv_CT(x)
        elif task == "distance":
            return self.final_conv_distance(x)
        elif task == "registration":
            return self.final_conv_registration(x)
        elif task == "segmentation":
            return self.final_conv_segmentation(x)
        else:
            raise ValueError(f"Unknown task: {task}")


class BrainFMSegmentation(nn.Module):
    """
    BrainFM Segmentation Model.

    Complete model with UNet3D backbone and segmentation head.
    Can load pretrained weights from brainfm_pretrained.pth.

    Args:
        in_channels: Number of input channels (default: 1 for MRI)
        num_classes: Number of output segmentation classes
        f_maps: Base number of feature maps in U-Net
        num_levels: Number of levels in U-Net (default: 5)
        pretrained: Whether to load pretrained weights
        freeze_backbone: Whether to freeze backbone weights
        task: Which task head to use for forward pass
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        f_maps: int = 64,
        num_levels: int = 6,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        task: str = "segmentation",
        verbose: bool = True,
    ):
        super().__init__()

        self.task = task

        # Create backbone
        self.backbone = UNet3D(
            in_channels=in_channels,
            f_maps=f_maps,
            layer_order="gcl",
            num_groups=8,
            num_levels=num_levels,
            is_unit_vector=False,
            conv_padding=1,
        )

        # Create head
        self.head = BrainFMHead(in_channels=f_maps, num_classes=num_classes)

        if pretrained:
            self._load_pretrained_weights(verbose=verbose)

        if freeze_backbone:
            self._freeze_backbone()

    def _load_pretrained_weights(self, verbose: bool = True):
        """Load pretrained weights for backbone and head."""
        checkpoint_path = get_checkpoint_path()

        if verbose:
            print(f"Loading pretrained weights from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract model state dict
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if exists (from DataParallel)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Transform and load backbone weights
        backbone_state_dict = transform_state_dict_keys(state_dict, target="backbone")
        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            backbone_state_dict, strict=False
        )

        if verbose:
            if missing_keys:
                print(f"  Backbone - Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"  Backbone - Unexpected keys: {len(unexpected_keys)}")
            else:
                print(f"  ✓ Backbone weights loaded successfully!")

        # Transform and load head weights
        head_state_dict = transform_state_dict_keys(state_dict, target="head")

        # Filter out keys with shape mismatch (e.g. if num_classes differs)
        model_state_dict = self.head.state_dict()
        filtered_head_state_dict = {}
        for k, v in head_state_dict.items():
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    filtered_head_state_dict[k] = v
                else:
                    if verbose:
                        print(f"  Skipping head weight '{k}' due to shape mismatch: {v.shape} vs {model_state_dict[k].shape}")
        
        # Only load matching head weights
        head_missing, head_unexpected = self.head.load_state_dict(
            filtered_head_state_dict, strict=False
        )

        if verbose:
            if head_missing:
                print(
                    f"  Head - Missing keys (expected if num_classes differs): {len(head_missing)}"
                )
            if head_unexpected:
                print(f"  Head - Unexpected keys: {len(head_unexpected)}")
            else:
                print(f"  ✓ Head weights loaded successfully!")

            total_params = sum(p.numel() for p in self.parameters())
            print(f"  Total model parameters: {total_params:,}")

    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone parameters frozen.")

    def forward(self, x, task: str = None):
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, D, H, W)
            task: Override default task

        Returns:
            Segmentation output (B, num_classes, D, H, W)
        """
        task = task or self.task
        features = self.backbone(x)
        output = self.head(features, task=task)
        return output

    def get_features(self, x):
        """Get backbone features without applying head."""
        return self.backbone(x)


def load_brainfm_unet3d(
    device: str = "cuda" if torch.cuda.is_available() else "cpu", verbose: bool = True
) -> UNet3D:
    """
    Load only the UNet3D backbone with pretrained weights.

    This is useful for feature extraction or when you want to
    add your own custom head.

    Args:
        device: Device to load the model on
        verbose: Whether to print loading information

    Returns:
        UNet3D backbone with loaded pretrained weights
    """
    model = UNet3D(
        in_channels=1,
        f_maps=64,
        layer_order="gcl",
        num_groups=8,
        num_levels=6,
        is_unit_vector=False,
        conv_padding=1,
    )

    checkpoint_path = get_checkpoint_path()

    if verbose:
        print(f"Loading pretrained backbone weights from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model state dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if exists
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Transform keys for backbone
    backbone_state_dict = transform_state_dict_keys(state_dict, target="backbone")

    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(
        backbone_state_dict, strict=False
    )

    if verbose:
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")
        if not missing_keys and not unexpected_keys:
            print("✓ Successfully loaded pretrained backbone weights!")
        print(f"Backbone parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = model.to(device)
    model.eval()

    return model


def inspect_checkpoint(verbose: bool = True) -> dict:
    """
    Inspect the checkpoint file to understand its structure.

    Args:
        verbose: Whether to print detailed information

    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint_path = get_checkpoint_path()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    info = {
        "checkpoint_type": type(checkpoint).__name__,
        "keys": None,
        "state_dict_keys": None,
        "num_parameters": 0,
        "backbone_keys": [],
        "head_keys": [],
    }

    if isinstance(checkpoint, dict):
        info["keys"] = list(checkpoint.keys())

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        info["state_dict_keys"] = list(state_dict.keys())
        info["num_parameters"] = sum(
            v.numel() for v in state_dict.values() if hasattr(v, "numel")
        )

        # Separate backbone and head keys
        for key in state_dict.keys():
            if key.startswith("backbone."):
                info["backbone_keys"].append(key)
            elif key.startswith("head."):
                info["head_keys"].append(key)

    if verbose:
        print("=" * 60)
        print("Checkpoint Inspection Report")
        print("=" * 60)
        print(f"Checkpoint type: {info['checkpoint_type']}")
        if info["keys"]:
            print(f"Top-level keys: {info['keys']}")
        print(f"Total parameters: {info['num_parameters']:,}")
        print("-" * 60)
        print(f"Backbone keys: {len(info['backbone_keys'])}")
        print(f"Head keys: {len(info['head_keys'])}")
        print("-" * 60)
        print("Sample backbone keys (first 5):")
        for key in info["backbone_keys"][:5]:
            print(f"  - {key}")
        print("Sample head keys:")
        for key in info["head_keys"]:
            print(f"  - {key}")
        print("=" * 60)

    return info


# Convenience function for quick testing
if __name__ == "__main__":
    print("Testing BrainFM Model Loader...")
    print()

    # Inspect checkpoint
    print("1. Inspecting checkpoint file...")
    info = inspect_checkpoint()
    print()

    # Load backbone only
    print("2. Loading pretrained backbone...")
    try:
        backbone = load_brainfm_unet3d(device="cpu", verbose=True)
        print()

        # Test backbone forward pass
        print("3. Testing backbone forward pass...")
        dummy_input = torch.randn(1, 1, 96, 96, 96)

        with torch.no_grad():
            output = backbone(dummy_input)

        print(f"   Input shape:  {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print()

    except Exception as e:
        print(f"Error loading backbone: {e}")
        import traceback

        traceback.print_exc()
        print()

    # Load full segmentation model
    print("4. Loading full segmentation model...")
    try:
        model = BrainFMSegmentation(num_classes=3, pretrained=True, verbose=True)
        model.eval()
        print()

        # Test full model forward pass
        print("5. Testing segmentation model forward pass...")
        dummy_input = torch.randn(1, 1, 64, 64, 64)

        with torch.no_grad():
            output = model(dummy_input)

        print(f"   Input shape:  {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print()
        print("=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error loading segmentation model: {e}")
        import traceback

        traceback.print_exc()
