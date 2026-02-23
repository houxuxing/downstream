import sys
import os
import torch
import json
import traceback

# Add current directory to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import model builder functions
try:
    from model_builder import (
        _create_dinov2_backbone,
        _create_brainsegfounder_backbone,
        _create_sam_brain3d_backbone,
        _create_voxhrnet_backbone,
        _create_nnunet_backbone,
        _create_brainfm_backbone,
    )
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Create a dummy nnUNetPlans.json for testing if the real one isn't found
DUMMY_PLANS_FILE = "dummy_nnUNetPlans.json"


def create_dummy_plans():
    dummy_plans = {
        "configurations": {
            "3d_fullres": {
                "patch_size": [128, 128, 128],
                "normalization_schemes": ["ZScoreNormalization"],
                "architecture": {
                    "arch_kwargs": {
                        "n_stages": 6,
                        "features_per_stage": [32, 64, 128, 256, 320, 320],
                        "kernel_sizes": [[3, 3, 3]] * 6,
                        "strides": [[1, 1, 1]] + [[2, 2, 2]] * 5,
                        "n_conv_per_stage": [2] * 6,
                        "n_conv_per_stage_decoder": [2] * 5,
                        "conv_bias": True,
                        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                        "nonlin_kwargs": {"inplace": True},
                        "conv_op": "torch.nn.modules.conv.Conv3d",
                    }
                },
            }
        }
    }
    with open(DUMMY_PLANS_FILE, "w") as f:
        json.dump(dummy_plans, f)
    return os.path.abspath(DUMMY_PLANS_FILE)


# Mock Configuration
class MockConfig:
    def __init__(self):
        self.target_size = (128, 128, 128)  # Will be updated per model
        self.backbone_name = ""
        self.num_classes = 4
        self.embed_dim = 512
        self.dinov2_ckpt = None
        self.brainsegfounder_ckpt = None
        self.sam_brain3d_ckpt = None
        self.voxhrnet_ckpt = None
        self.nnunet_ckpt = None
        self.brainfm_ckpt = None
        self.nnunet_plans_json = create_dummy_plans()


def cleanup():
    if os.path.exists(DUMMY_PLANS_FILE):
        try:
            os.remove(DUMMY_PLANS_FILE)
        except:
            pass


def test_models():
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running tests on: {device}")

    cfg = MockConfig()

    models_to_test = [
        ("dinov2", (112, 112, 112)),
        ("brainsegfounder", (128, 128, 128)),
        ("sam_brain3d", (128, 128, 128)),
        ("voxhrnet", (128, 128, 128)),
        ("nnunet", (128, 128, 128)),
        ("brainfm", (128, 128, 128)),
    ]

    # Write report to file to avoid encoding issues
    with open("backbone_report.txt", "w", encoding="utf-8") as f:
        msg = f"\n{'=' * 80}\n"
        msg += f"{'Model Inspection Report':^80}\n"
        msg += f"{'=' * 80}\n"
        msg += f"{'Model':<20} | {'Input Size':<20} | {'Output Shape':<30} | {'Dim In':<10}\n"
        msg += f"{'-' * 80}\n"
        print(msg)  # Print to console too
        f.write(msg)

        for name, input_size in models_to_test:
            cfg.target_size = input_size
            cfg.backbone_name = name

            output_str = "Error"
            dim_in_str = "N/A"

            try:
                # Build Backbone
                backbone = None
                if name == "dinov2":
                    backbone = _create_dinov2_backbone(cfg, device)
                elif name == "brainsegfounder":
                    backbone = _create_brainsegfounder_backbone(cfg, device)
                elif name == "sam_brain3d":
                    backbone = _create_sam_brain3d_backbone(cfg, device)
                elif name == "voxhrnet":
                    backbone = _create_voxhrnet_backbone(cfg, device)
                elif name == "nnunet":
                    backbone = _create_nnunet_backbone(cfg, device)
                elif name == "brainfm":
                    backbone = _create_brainfm_backbone(cfg, device)

                if backbone is None:
                    output_str = "Backbone creation returned None"
                    continue

                # Check reported dim_in
                if hasattr(backbone, "dim_in"):
                    dim_in_str = str(backbone.dim_in)

                # Create Dummy Input
                # Shape: (Batch, Channel, D, H, W)
                dummy_input = torch.randn(1, 1, *input_size).to(device)

                # Forward Pass
                backbone.eval()
                with torch.no_grad():
                    output = backbone(dummy_input)

                output_str = str(tuple(output.shape))

                # Additional validation
                if output.dim() != 5:
                    output_str += " [WARNING: Not 5D]"

            except Exception as e:
                output_str = f"FAILED: {str(e).split('(')[0]}"

            row = f"{name:<20} | {str(input_size):<20} | {output_str:<30} | {dim_in_str:<10}\n"
            print(row.strip())
            f.write(row)

        f.write(f"{'=' * 80}\n")

    print("\nReport written to backbone_report.txt")
    cleanup()


if __name__ == "__main__":
    try:
        test_models()
    except KeyboardInterrupt:
        cleanup()
    except Exception:
        cleanup()
        traceback.print_exc()
