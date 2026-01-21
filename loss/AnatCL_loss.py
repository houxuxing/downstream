# loss.py
# -*- coding: utf-8 -*-
"""
AnatCL losses (Local / Global) extended from y-Aware (Duplums).

依据你的约束实现：
- y-Aware：RBF核，sigma=5（不对年龄差做额外尺度化）；cosine + L2 norm；temperature=0.1
- self 处理：按常规排除“自身”（对角线）参与分母与加权项
- AnatCL-Local：measures 为 ROI×(CT/GMV/SA)；gamma 将每个样本的每个通道在 ROI 维(K)做 min-max 映射到 [0,1]（强调个体内相对分布；如需全局归一化请改实现）
- AnatCL-Global：不做上述 gamma 的跨量纲 [0,1] 归一化；相似度采用 cosine（实现中会对向量做 L2 normalize/L2 归一化）。本实现为便于作为 soft target，将 beta 从 [-1,1] 线性映射到 [0,1]，并对负值截断后再做行归一化
- 省略项：严格加回（对每个 anchor 的权重做归一化）
- lambda：开放为超参（lambda_anat, lambda_age）

期望输入：
- z:      [N, D]   （N 可为 B 或 2B；若采用双视图对比学习，通常 N=2B）
- ages:   [N]      （必须与 z 对齐；双视图时通常把同一 subject 的 age 复制到两条 view 上）
- meas:   [N, K, 3]（必须与 z 对齐；双视图时通常把同一 subject 的 measures 复制到两条 view 上；最后一维依次为 [CT, GMV, SA]）
输出：
- 标量 loss
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from config import Config

# -------------------------
# helpers
# -------------------------


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))


def cosine_sim_matrix(z: torch.Tensor) -> torch.Tensor:
    """
    z: [N, D]
    return: [N, N] cosine similarity matrix
    """
    z = l2_normalize(z, dim=1)
    return z @ z.t()


def pairwise_rbf_age_weights(ages: torch.Tensor, sigma: float = 5.0) -> torch.Tensor:
    """
    ages: [N]
    w_ij = exp( - (age_i - age_j)^2 / (2*sigma^2) )
    return: [N, N]
    """
    ages = ages.view(-1, 1)
    diff = ages - ages.t()
    return torch.exp(-(diff * diff) / (2.0 * (sigma**2)))


def compute_soft_target_weights(
    weights: torch.Tensor,
    map_to_01: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    A.5-A.7: Process raw weights into normalized soft target distribution.

    Args:
        weights: [N, N] raw weights matrix (e.g. cosine sim or RBF weights)
        map_to_01: if True, map [-1, 1] to [0, 1] via (x+1)/2
        eps: epsilon for numerical stability

    Returns:
        [N, N] Row-normalized weights matrix (sum=1 per row, excluding diagonal)
    """
    N = weights.shape[0]
    device = weights.device

    # A.5 (Optional) Map to [0, 1] if needed
    if map_to_01:
        weights = (weights + 1.0) * 0.5
    
    # Ensure non-negative (clamp)
    weights = weights.clamp_min(0.0)

    # A.6 Mask: remove invalid pairs (diagonal)
    eye = torch.eye(N, dtype=torch.bool, device=device)
    weights.masked_fill_(eye, 0.0)

    # A.7 Row Normalization
    denom = weights.sum(dim=1, keepdim=True).clamp_min(eps)
    weights_norm = weights / denom

    return weights_norm


def soft_target_cross_entropy(
    z: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    B.1-B.3: Soft-target NCE (Cross Entropy) calculation.

    Args:
        z: [N, D] feature embeddings
        targets: [N, N] row-normalized soft target weights (beta_tilde)
        temperature: scaling factor for logits

    Returns:
        Scalar loss
    """
    N = z.shape[0]
    device = z.device

    # B.1 Logits: L_{a,i} = sim(z_a, z_i) / tau
    sim = cosine_sim_matrix(z)
    logits = sim / temperature

    # Mask invalid items (diagonal) for softmax
    # Set to very small number so softmax result is 0
    eye = torch.eye(N, dtype=torch.bool, device=device)
    neg_large = torch.finfo(logits.dtype).min
    logits.masked_fill_(eye, neg_large)

    # B.2 Predicted Distribution: p_{a,:} = softmax(L_{a,:})
    # log_softmax is more numerically stable than log(softmax)
    log_prob = torch.log_softmax(logits, dim=1)

    # B.3 Loss: - sum(beta_tilde * log_p)
    # Average over anchors (M)
    loss = -(targets * log_prob).sum(dim=1).mean()

    return loss


def masked_soft_target_nce(
    sim: torch.Tensor,
    weights: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    [Deprecated] Legacy implementation kept for compatibility.
    Use soft_target_cross_entropy instead.
    """
    # ... logic identical to previous implementation ...
    # Re-using new helper functions to maintain behavior but cleaner code
    
    # Normalize weights (similar to compute_soft_target_weights but without mapping)
    weights_norm = compute_soft_target_weights(weights, map_to_01=False, eps=eps)
    
    # Calculate loss (sim is passed directly here instead of z)
    N = sim.shape[0]
    device = sim.device
    logits = sim / temperature
    eye = torch.eye(N, dtype=torch.bool, device=device)
    neg_large = torch.finfo(logits.dtype).min
    logits.masked_fill_(eye, neg_large)
    log_prob = torch.log_softmax(logits, dim=1)
    
    return -(weights_norm * log_prob).sum(dim=1).mean()


# -------------------------
# y-Aware loss
# -------------------------


class YAwareLoss(nn.Module):
    """
    y-Aware Contrastive loss using continuous proxy meta-data (Age).

    z:    [N, D]
    age:  [N]
    """

    def __init__(self, cfg: Optional[Config] = None):
        super().__init__()
        self.cfg = cfg or Config()

    def forward(self, z: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"z must be [N,D], got {tuple(z.shape)}")
        if age.ndim != 1 or age.shape[0] != z.shape[0]:
            raise ValueError(
                f"age must be [N], got {tuple(age.shape)} with z {tuple(z.shape)}"
            )

        sim = cosine_sim_matrix(z)  # [N,N]
        w = pairwise_rbf_age_weights(age, sigma=self.cfg.sigma)  # [N,N]
        return masked_soft_target_nce(sim, w, temperature=self.cfg.temperature)


# -------------------------
# AnatCL weights
# -------------------------


def anatcl_local_weights(
    measures: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Local AnatCL positiveness degree alpha_{n,m}:
      measures: [N, K, 3]  (CT/GMV/SA)

    gamma: normalize each sample's each channel to [0,1] across ROIs (K维)
      gamma(x) = (x - min)/(max-min)
    then per ROI k: cosine between 3-d vectors, average over K.

    return: [N, N]
    """
    if measures.ndim != 3 or measures.shape[-1] != 3:
        raise ValueError(f"measures must be [N,K,3], got {tuple(measures.shape)}")

    # gamma to [0,1] per sample per channel across ROIs
    mn = measures.amin(dim=1, keepdim=True)
    mx = measures.amax(dim=1, keepdim=True)
    gamma = (measures - mn) / (mx - mn).clamp_min(eps)  # [N,K,3] in [0,1] (approx)

    # per ROI 3-vector cosine => need L2-normalize across last dim
    v = l2_normalize(gamma, dim=2, eps=eps)  # [N,K,3]

    # dot per ROI between sample pairs: einsum over 3-d
    # out: [N,N,K], then mean over K => [N,N]
    dot_per_roi = torch.einsum("nkd,mkd->nmk", v, v)  # [N,N,K]
    alpha = dot_per_roi.mean(dim=2)
    return alpha


def anatcl_global_weights(
    measures: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Global AnatCL positiveness degree beta_{n,m}:
      measures: [N, K, 3] (CT/GMV/SA), 不做 gamma（保持原尺度）

    For each modality j in {0,1,2}, build vector omega_j in R^K, compute cosine,
    then average over j.

    return: [N, N]
    """
    if measures.ndim != 3 or measures.shape[-1] != 3:
        raise ValueError(f"measures must be [N,K,3], got {tuple(measures.shape)}")

    sims = []
    for j in range(3):
        x = measures[:, :, j]  # [N,K]
        x = l2_normalize(x, dim=1, eps=eps)  # for cosine
        sims.append(x @ x.t())  # [N,N]
    beta = sum(sims) / 3.0
    return beta


# -------------------------
# AnatCL final losses (Local / Global)
# -------------------------


class AnatCLLocalLoss(nn.Module):
    """
    L = lambda_anat * L_anat(local) + lambda_age * L_age(y-aware)

    Inputs:
      z:        [N,D]
      ages:     [N]
      measures: [N,K,3]
    """

    def __init__(self, cfg: Optional[Config] = None):
        super().__init__()
        self.cfg = cfg or Config()
        self.yaware = YAwareLoss(self.cfg)

    def forward(
        self,
        z: torch.Tensor,
        ages: torch.Tensor,
        measures: torch.Tensor,
        return_dict: bool = False,
    ):
        sim = cosine_sim_matrix(z)

        # local anatomical weights alpha
        alpha = anatcl_local_weights(measures)

        # L_anat using soft-target NCE with alpha as weights
        L_anat = masked_soft_target_nce(sim, alpha, temperature=self.cfg.temperature)

        # L_age using y-aware RBF weights
        L_age = self.yaware(z, ages)

        L = self.cfg.lambda_anat * L_anat + self.cfg.lambda_age * L_age

        if return_dict:
            return {
                "loss": L,
                "loss_anat": L_anat.detach(),
                "loss_age": L_age.detach(),
                "lambda_anat": float(self.cfg.lambda_anat),
                "lambda_age": float(self.cfg.lambda_age),
            }
        return L


class AnatCLGlobalLoss(nn.Module):
    """
    L = lambda_anat * L_anat(global) + lambda_age * L_age(y-aware)

    Inputs:
      z:        [N,D] (Usually N=2B for two-view)
      ages:     [N]
      measures: [N,K,3]
    """

    def __init__(self, cfg: Optional[Config] = None):
        super().__init__()
        self.cfg = cfg or Config()
        # YAwareLoss now also needs to be updated to use the new helpers if we want full consistency,
        # but the request specifically targeted Global Loss structure. 
        # We can reuse the YAwareLoss class but might need to ensure it uses the new helpers or compatible logic.
        # For now, we'll keep YAwareLoss as is or update it if needed. 
        # Actually, let's update YAwareLoss logic inline here or in its own class to use the new helpers.
        self.yaware = YAwareLoss(self.cfg)

    def forward(
        self,
        z: torch.Tensor,
        ages: torch.Tensor,
        measures: torch.Tensor,
        return_dict: bool = False,
    ):
        # --- A. 生成解剖 soft 权重矩阵 beta ---
        
        # A.1-A.4: 计算原始 cosine 相似度均值
        # beta_raw range: [-1, 1]
        beta_raw = anatcl_global_weights(measures)

        # A.5-A.7: 映射、Mask、行归一化
        # beta_norm range: [0, 1], sum(row) = 1
        beta_norm = compute_soft_target_weights(
            beta_raw, 
            map_to_01=True, # A.5: (beta+1)/2
            eps=1e-12
        )

        # --- B. Soft-target NCE ---
        
        # B.1-B.3: 计算 Cross Entropy Loss
        L_anat = soft_target_cross_entropy(
            z, 
            beta_norm, 
            temperature=self.cfg.temperature
        )

        # --- L_age (Y-Aware) ---
        # 同样使用严谨的步骤
        
        # 1. 计算 RBF 权重
        age_weights_raw = pairwise_rbf_age_weights(ages, sigma=self.cfg.sigma)
        
        # 2. 归一化 (不需要 map_to_01，因为 RBF 输出已经是 [0,1] 且非负)
        age_weights_norm = compute_soft_target_weights(
            age_weights_raw, 
            map_to_01=False, 
            eps=1e-12
        )
        
        # 3. 计算 Loss
        L_age = soft_target_cross_entropy(
            z, 
            age_weights_norm, 
            temperature=self.cfg.temperature
        )

        # --- Total Loss ---
        L = self.cfg.lambda_anat * L_anat + self.cfg.lambda_age * L_age

        if return_dict:
            return {
                "loss": L,
                "loss_anat": L_anat.detach(),
                "loss_age": L_age.detach(),
                "lambda_anat": float(self.cfg.lambda_anat),
                "lambda_age": float(self.cfg.lambda_age),
            }
        return L
