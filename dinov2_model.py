import torch
from dinov2_models.vision_transformer import vit_base_3d
from dinov2_utils.utils import load_pretrained_weights
from config import Config

# 1. 构建模型
model = vit_base_3d(
    img_size=128,  # 输入尺寸
    patch_size=16,  # patch 大小
    in_chans=1,  # 输入通道（医学图像通常为1）
)

# 2. 加载预训练权重
load_pretrained_weights(model, Config.dinov2_ckpt, checkpoint_key="teacher")

# 3. 推理
model.eval()
model.cuda()

x = torch.randn(1, 1, 128, 128, 128).cuda()  # (B, C, H, W, D)
with torch.no_grad():
    features = model.forward_features(x)
    # features["x_norm_clstoken"]: (B, embed_dim) - CLS token
    # features["x_norm_patchtokens"]: (B, N, embed_dim) - patch tokens
