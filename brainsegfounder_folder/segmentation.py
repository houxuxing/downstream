import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BrainSegFounderSegmentation(nn.Module):
    def __init__(self, args, num_classes=3, upsample="vae"):
        super(BrainSegFounderSegmentation, self).__init__()
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        dim = args.bottleneck_depth

        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=args.num_swin_blocks_per_stage,
            num_heads=args.num_heads_per_stage,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
        )

        if upsample == "vae":
            self.upsample = nn.Upsample(
                scale_factor=2, mode="trilinear", align_corners=False
            )
            self.dec1 = _ConvBlock(dim, dim // 2)
            self.dec2 = _ConvBlock(dim // 2, dim // 4)
            self.dec3 = _ConvBlock(dim // 4, dim // 8)
            self.dec4 = _ConvBlock(dim // 8, dim // 16)
            self.dec5 = _ConvBlock(dim // 16, dim // 16)
            self.seg_head = nn.Conv3d(dim // 16, num_classes, kernel_size=1, stride=1)
        else:
            raise NotImplementedError(
                f"Upsample method {upsample} not implemented for segmentation"
            )

    def _add_skip(self, x, skip):
        if skip is None:
            return x
        if x.shape[1] != skip.shape[1] or x.shape[2:] != skip.shape[2:]:
            return x
        return x + skip

    def forward(self, x):
        x_out = self.swinViT(x.contiguous())
        if isinstance(x_out, (list, tuple)) and len(x_out) >= 5:
            x0, x1, x2, x3, x4 = x_out[:5]
        else:
            x0 = x1 = x2 = x3 = None
            x4 = x_out

        x = self.dec1(x4)
        x = self.upsample(x)
        x = self._add_skip(x, x3)

        x = self.dec2(x)
        x = self.upsample(x)
        x = self._add_skip(x, x2)

        x = self.dec3(x)
        x = self.upsample(x)
        x = self._add_skip(x, x1)

        x = self.dec4(x)
        x = self.upsample(x)
        x = self._add_skip(x, x0)

        x = self.dec5(x)
        x = self.upsample(x)
        logits = self.seg_head(x)
        return logits

    def load_pretrained_weights(self, weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Clean keys
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Load SwinViT weights
        # Keys in checkpoint are like "swinViT.layers..."
        # Keys in self.swinViT are like "layers..."
        # So we need to strip "swinViT." prefix if present, OR if we load into self, we keep it.
        # The checkpoint has "swinViT..." keys because SSLHead has self.swinViT.
        # Our class also has self.swinViT.
        # So loading state_dict into self should work for swinViT keys.
        # But we should ignore "rotation_head", "contrastive_head", and "conv" keys (or "decoder" keys).

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(
            f"Loading pretrained weights: {len(unexpected)} unexpected keys (ignored), {len(missing)} missing keys."
        )
        # Expected missing: decoder.*
        # Expected unexpected: rotation_head.*, contrastive_head.*, conv.*
