import torch
import torch.nn as nn
import medim
try:
    from .model import SAMMed3DAdapter
except ImportError:
    from model import SAMMed3DAdapter

# Import UNETRHead from dinov2.eval.segmentation_3d.segmentation_heads
try:
    from dinov2.eval.segmentation_3d.segmentation_heads import UNETRHead
except ImportError:
    # Fallback for relative imports if needed
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming dinov2 is at ../../../dinov2
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from dinov2.eval.segmentation_3d.segmentation_heads import UNETRHead


class SamBrainUNETR(nn.Module):
    """
    Combined model with SAM-Brain Encoder and UNETR Decoder Head.
    Supports full fine-tuning.
    """
    def __init__(self, pretrained_weights=None, num_classes=3, image_size=128):
        super().__init__()
        
        # 1. Build SAM-Brain Encoder (Backbone)
        print("Building SAM-Brain backbone...")
        full_model = medim.create_model("SAM-Med3D", pretrained=False)
        
        # Load pretrained weights for backbone if provided
        if pretrained_weights:
            print(f"Loading SAM-Brain backbone weights from: {pretrained_weights}")
            checkpoint = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
            
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            # Handle prefix and filter for encoder only (as we use UNETRHead as decoder)
            new_state_dict = {}
            for k, v in state_dict.items():
                # 1. Handle 'module.' prefix (from DataParallel)
                if k.startswith("module."):
                    k = k[7:]
                
                # 2. Strip potential backbone prefixes
                if k.startswith("backbone.model."):
                    k = k[15:]
                elif k.startswith("backbone."):
                    k = k[9:]
                
                # 3. Only keep image_encoder weights
                if k.startswith("image_encoder."):
                    new_state_dict[k] = v
            
            state_dict = new_state_dict
            
            # Remove unused components from medim model to save memory
            print("Removing unused components (prompt_encoder, mask_decoder) from medim model...")
            if hasattr(full_model, "prompt_encoder"):
                full_model.prompt_encoder = nn.Identity()
            if hasattr(full_model, "mask_decoder"):
                full_model.mask_decoder = nn.Identity()

            # Load weights into the MedIM model
            msg = full_model.load_state_dict(state_dict, strict=False)
            print(f"Backbone load status: {msg}")
        else:
            print("No pretrained weights provided for backbone, using random initialization.")

        # Wrap with Adapter to provide get_intermediate_layers interface
        self.backbone = SAMMed3DAdapter(full_model)
        
        # 1.5 Calibrate num_features using a dummy forward pass
        # This is necessary because SAM-Med3D model config might report 384 (ViT-S) 
        # but actually output 768 (ViT-B) or vice versa depending on checkpoints/config.
        print("Calibrating backbone output dimensions...")
        with torch.no_grad():
            # Create a small dummy input. Standard SAM-Brain input is 128x128x128
            # We use CPU to avoid OOM during init if possible, or same device as model
            dummy = torch.randn(1, 1, 128, 128, 128)
            try:
                # Get intermediate layers as UNETRHead would
                feats = self.backbone.get_intermediate_layers(dummy, n=[1], reshape=False)
                # feats[0] shape should be (B, N, C)
                actual_dim = feats[0].shape[-1]
                print(f"Detected actual feature dimension: {actual_dim}")
                
                if self.backbone.num_features != actual_dim:
                    print(f"Updating backbone.num_features from {self.backbone.num_features} to {actual_dim}")
                    self.backbone.num_features = actual_dim
                    self.backbone.embed_dim = actual_dim
            except Exception as e:
                print(f"Calibration failed: {e}. Using default num_features: {self.backbone.num_features}")

        # 2. Build UNETR Head
        # Note: SAM-Brain (ViT-Small) typically uses 384 embed dim. 
        # UNETRHead expects feature_model to have num_features attribute.
        # SAMMed3DAdapter sets num_features=384.
        
        # Dummy autocast context (can be replaced with actual if needed)
        autocast_ctx = torch.cuda.amp.autocast
        
        print(f"Building UNETR Head for {num_classes} classes...")
        self.head = UNETRHead(
            feature_model=self.backbone,
            input_channels=1,     # SAM-Brain takes 1 channel input
            image_size=image_size, # Typically 128 for SAM-Brain
            num_classes=num_classes,
            autocast_ctx=autocast_ctx
        )
        
    def forward(self, x):
        # UNETRHead.forward calls backbone.get_intermediate_layers internally
        return self.head(x)


if __name__ == "__main__":
    print("Testing SamBrainUNETR...")
    print("-" * 50)
    
    # Configuration
    NUM_CLASSES = 4
    IMAGE_SIZE = 128
    INPUT_CHANNELS = 1
    
    # 1. Initialize model
    # Note: Using random initialization (pretrained_weights=None) for testing
    try:
        model = SamBrainUNETR(
            pretrained_weights=None, 
            num_classes=NUM_CLASSES, 
            image_size=IMAGE_SIZE
        )
        print("\n[OK] Model initialized successfully.")
        
        # Move to CUDA if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model = model.to(device)
        model.eval()
        
        # 2. Create dummy input
        # Shape: (Batch, Channels, D, H, W)
        batch_size = 1
        dummy_input = torch.randn(
            batch_size, 
            INPUT_CHANNELS, 
            IMAGE_SIZE, 
            IMAGE_SIZE, 
            IMAGE_SIZE
        ).to(device)
        print(f"\nDummy input shape: {dummy_input.shape}")
        
        # 3. Forward pass
        print("Running forward pass...")
        with torch.no_grad():
            output = model(dummy_input)
            
        # 4. Check output
        print(f"Output shape: {output.shape}")
        
        expected_shape = (batch_size, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)
        if output.shape == expected_shape:
            print(f"\n[PASS] Output shape matches expected: {expected_shape}")
        else:
            print(f"\n[FAIL] Output shape mismatch! Expected {expected_shape}, got {output.shape}")
            
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()


