import torch

class SAMMed3DAdapter(torch.nn.Module):
    """Adapter to make SAM-Brain encoder compatible with 3DINO segmentation heads."""

    def __init__(self, sam_model):
        super().__init__()
        # Always use the full model structure
        self.model = sam_model
        self.encoder = self.model.image_encoder
        self.prompt_encoder = self.model.prompt_encoder
        self.mask_decoder = self.model.mask_decoder

        # Attributes expected by UNETRHead
        self.num_features = 384  # SAM-Brain (ViT-Small) usually has 384, check config if possible
        self.embed_dim = 384

        # Mock patch_embed with patch_size attribute
        class PatchEmbedMock:
            patch_size = (16, 16, 16)  # 128 / 8 = 16

        self.patch_embed = PatchEmbedMock()

        # Store intermediate outputs
        self._intermediate_outputs = []
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture intermediate block outputs."""

        def make_hook(idx):
            def hook(module, input, output):
                self._intermediate_outputs.append(output)

            return hook

        for idx, block in enumerate(self.encoder.blocks):
            h = block.register_forward_hook(make_hook(idx))
            self._hooks.append(h)

    def get_intermediate_layers(self, x, n, return_class_token=False, reshape=False):
        """
        Get intermediate layer outputs, compatible with 3DINO interface.

        Args:
            x: Input tensor (B, 1, H, W, D)
            n: List of block indices to return (e.g., [2, 5, 8, 11])
            return_class_token: Ignored (SAM-Brain has no class token)
            reshape: If True, reshape to (B, C, H, W, D)

        Returns:
            Tuple of intermediate features
        """
        # 清空之前的中间输出以释放显存
        self._intermediate_outputs = []
        import gc
        gc.collect()
        if x.is_cuda:
            torch.cuda.empty_cache()

        # Forward pass to collect intermediate outputs
        with torch.no_grad() if not self.training else torch.enable_grad():
            _ = self.encoder(x)

        # Select requested layers (n is list of indices like [2, 5, 8, 11])
        # Map to 0-indexed: layer 5 -> index 4, etc.
        results = []
        for idx in n:
            block_idx = idx - 1 if idx > 0 else 0  # Adjust for 1-based indexing
            block_idx = min(block_idx, len(self._intermediate_outputs) - 1)

            out = self._intermediate_outputs[block_idx]  # (B, 8, 8, 8, 768)

            if reshape:
                # Reshape to (B, C, H, W, D)
                out = out.permute(0, 4, 1, 2, 3).contiguous()
            else:
                # Flatten spatial dims to (B, N, C) format
                B = out.shape[0]
                # Use the actual channel dimension from output, do not force reshape with self.num_features
                # This handles cases where num_features might have been inferred incorrectly but output is correct
                C = out.shape[-1]
                out = out.reshape(B, -1, C)  # (B, 512, 768)

            results.append(out)

        # 清理不需要的中间输出以节省显存
        self._intermediate_outputs = []
        return tuple(results)

    def forward(self, x):
        """Forward pass returning final encoder output."""
        return self.encoder(x)
