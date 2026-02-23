"""参数冻结工具模块

包含模型参数冻结/解冻的相关工具函数。
"""

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoRALinear(nn.Module):
    """DoRA: weight re-parameterized low-rank adaptation for Linear layers."""

    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        if r <= 0:
            raise ValueError(f"DoRA rank r must be > 0, got {r}")
        device = base.weight.device
        dtype = base.weight.dtype
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.eps = 1e-6

        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # LoRA params
        self.lora_A = nn.Parameter(
            torch.zeros(r, self.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, r, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Magnitude (per-output) initialized from base weight norm
        with torch.no_grad():
            w_norm = self.weight.norm(dim=1)
        self.magnitude = nn.Parameter(w_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DoRA weight: W_dora = g * (W0 + BA) / ||W0 + BA||
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        weight_hat = self.weight + delta_w
        weight_norm = weight_hat.norm(dim=1, keepdim=True).clamp(min=self.eps)
        weight_dora = (self.magnitude.unsqueeze(1) * weight_hat) / weight_norm
        return F.linear(x, weight_dora, self.bias)


def _match_target(name: str, targets: Sequence[str]) -> bool:
    if not targets:
        return True
    name_l = name.lower()
    for t in targets:
        if t.lower() in name_l:
            return True
    return False


def _replace_linear_with_dora(
    module: nn.Module,
    targets: Sequence[str],
    r: int,
    alpha: int,
    replaced: List[str],
    prefix: str = "",
):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and _match_target(full_name, targets):
            if isinstance(child, DoRALinear):
                continue
            setattr(module, name, DoRALinear(child, r=r, alpha=alpha))
            replaced.append(full_name)
        else:
            _replace_linear_with_dora(child, targets, r, alpha, replaced, full_name)


def _apply_dora_to_brainsegfounder(
    backbone: nn.Module,
    r: int,
    alpha: int,
    target_modules: Sequence[str],
    is_main: bool,
    prefix: str,
) -> int:
    if r <= 0:
        return 0

    swinvit = None
    if hasattr(backbone, "model"):
        swinvit = backbone.model
    elif hasattr(backbone, "swinViT"):
        swinvit = backbone.swinViT

    if swinvit is None:
        return 0
    # BrainSegFounder SwinViT should have layers1-4
    if not (hasattr(swinvit, "layers1") and hasattr(swinvit, "layers4")):
        return 0

    replaced: List[str] = []
    _replace_linear_with_dora(
        swinvit, target_modules, r=r, alpha=alpha, replaced=replaced, prefix="swinViT"
    )

    if is_main:
        if replaced:
            print(
                f"{prefix}[INFO] DoRA applied to BrainSegFounder (SwinViT): "
                f"{len(replaced)} Linear layers, r={r}, alpha={alpha}"
            )
        else:
            print(
                f"{prefix}[WARNING] DoRA enabled but no Linear layers matched targets "
                f"{list(target_modules)}"
            )
    return len(replaced)


def set_frozen_batchnorm_eval(model: nn.Module):
    """
    防止冻结部分的 BatchNorm 在 train() 下继续更新 running_mean/running_var。
    """
    net: nn.Module = model.module if hasattr(model, "module") else model  # type: ignore[union-attr]
    for m in net.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            params = list(m.parameters(recurse=False))
            if params and all(not p.requires_grad for p in params):
                m.eval()


def setup_parameter_freezing(
    model: nn.Module,
    train_mode: str,
    is_main: bool = True,
    unfreeze_last_n_blocks: int = 6,
):
    """
    设置参数冻结策略。

    Args:
        model: 模型
        train_mode: 训练模式
            - 'none': 全部可训练（默认，从头训练）
            - 'head': 只训练分类头，冻结整个 backbone
            - 'partial': 冻结 backbone 前面的层，解冻最后几层 + 分类头
        is_main: 是否为主进程（用于打印信息）
        unfreeze_last_n_blocks: 当 train_mode='partial' 时，解冻最后 N 个 Transformer 块
    """
    net = model.module if hasattr(model, "module") else model

    if train_mode == "none":
        # 默认 requires_grad=True，无需额外操作
        if is_main:
            print("[INFO] 训练模式: none - 全部参数可训练")

    elif train_mode == "head":
        for p in net.backbone.parameters():  # type: ignore[attr-defined]
            p.requires_grad = False
        if is_main:
            print("[INFO] 训练模式: head - 仅训练分类头")

    elif train_mode == "partial":
        # 先冻结整个 backbone
        for p in net.backbone.parameters():  # type: ignore[attr-defined]
            p.requires_grad = False

        # 根据 backbone 类型选择不同的解冻策略
        backbone = net.backbone
        backbone_identified = False

        # 1. 检查是否是 nnUNet backbone
        #    nnUNetEncoderBackbone 有 encoder 属性，encoder 有 stages 属性 (PlainConvEncoder)
        if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "stages"):
            # nnUNet Backbone - 使用专用的解冻函数
            _unfreeze_nnunet_layers(backbone.encoder, unfreeze_last_n_blocks, is_main)
            backbone_identified = True

        # 2. 检查是否是 SAM-Brain 3D backbone
        #    SAM-Brain3D 的 encoder 是 ViT，有 blocks 或 layers 属性
        elif hasattr(backbone, "encoder") and (
            hasattr(backbone.encoder, "blocks") or hasattr(backbone.encoder, "layers")
        ):
            encoder = backbone.encoder
            _unfreeze_sam_brain3d_layers(encoder, unfreeze_last_n_blocks, is_main)
            backbone_identified = True

        # 3. 检查是否是 BrainFM backbone (UNet3D encoder)
        #    BrainFMEncoderWrapper 有 encoders 属性 (ModuleList)
        elif hasattr(backbone, "encoders"):
            _unfreeze_brainfm_layers(backbone, unfreeze_last_n_blocks, is_main)
            backbone_identified = True

        # 4. 检查是否是 VoxHR-Net Backbone
        #    VoxHRNetBackboneWrapper 有 stages 属性（不是 encoder.stages）
        elif hasattr(backbone, "stages"):
            _unfreeze_voxhrnet_layers(backbone, unfreeze_last_n_blocks, is_main)
            backbone_identified = True

        # 4. 检查是否是 BrainSegFounder 或 DINOv2 backbone
        elif hasattr(backbone, "model"):
            vit_model = backbone.model
            # BrainSegFounder: model 是 SwinViT，有 layers1-4 属性
            if hasattr(vit_model, "layers1") and hasattr(vit_model, "layers4"):
                _unfreeze_swinvit_layers(vit_model, unfreeze_last_n_blocks, is_main)
                backbone_identified = True
            # 备用：model 还包含 swinViT
            elif hasattr(vit_model, "swinViT"):
                _unfreeze_swinvit_layers(
                    vit_model.swinViT, unfreeze_last_n_blocks, is_main
                )
                backbone_identified = True
            # DINOv2: model 有 blocks 属性
            elif hasattr(vit_model, "blocks"):
                _unfreeze_vit_layers(vit_model, unfreeze_last_n_blocks, is_main)
                backbone_identified = True

        if not backbone_identified:
            if is_main:
                print("[WARNING] 未知的 backbone 类型，无法应用 partial 冻结策略")

        if is_main:
            print(
                f"[INFO] 训练模式: partial - 冻结 backbone 前面的层，解冻最后 {unfreeze_last_n_blocks} 个 Transformer 块"
            )

    else:
        raise ValueError(
            f"Unknown train_mode: {train_mode}. Expected 'none', 'head', or 'partial'."
        )

    set_frozen_batchnorm_eval(model)


def _unfreeze_vit_layers(
    vit_model: nn.Module, unfreeze_last_n_blocks: int, is_main: bool
):
    """解冻 ViT 模型的最后 N 个块（用于 DINOv2）"""
    # 解冻最后的 LayerNorm
    if hasattr(vit_model, "norm"):
        for p in vit_model.norm.parameters():
            p.requires_grad = True

    # 解冻最后 N 个块
    if hasattr(vit_model, "blocks"):
        blocks = vit_model.blocks
        all_blocks = []

        # 处理 chunked blocks 的情况
        if hasattr(vit_model, "chunked_blocks") and vit_model.chunked_blocks:
            for chunk in blocks:
                for block in chunk:
                    if not isinstance(block, nn.Identity):
                        all_blocks.append(block)
        else:
            # 非 chunked 情况
            all_blocks = list(blocks)

        total_layers = len(all_blocks)
        # 动态限制：解冻层数不超过总层数的一半
        limit = total_layers // 2
        if unfreeze_last_n_blocks > 0 and total_layers > 0:
            limit = max(1, limit)
        actual_n = min(unfreeze_last_n_blocks, limit)

        if is_main:
            print(
                f"[INFO] ViT (DINOv2): 总层数 {total_layers}，限制解冻最多 {limit} 层。"
                f"请求 {unfreeze_last_n_blocks} -> 实际解冻最后 {actual_n} 个 blocks"
            )

        for block in all_blocks[-actual_n:]:
            for p in block.parameters():
                p.requires_grad = True


def _unfreeze_sam_brain3d_layers(
    encoder: nn.Module, unfreeze_last_n_blocks: int, is_main: bool
):
    """解冻 SAM-Brain 3D encoder 的最后 N 个块

    SAM-Brain 3D 的 encoder 是基于 ViT 的，具有 blocks 属性。
    """
    # 解冻最后的 LayerNorm（如果有）
    if hasattr(encoder, "norm"):
        for p in encoder.norm.parameters():
            p.requires_grad = True

    all_blocks = []
    if hasattr(encoder, "blocks"):
        all_blocks = list(encoder.blocks)
    elif hasattr(encoder, "layers"):
        all_blocks = list(encoder.layers)

    total_layers = len(all_blocks)
    if total_layers > 0:
        # 动态限制：解冻层数不超过总层数的一半
        limit = total_layers // 2
        if unfreeze_last_n_blocks > 0 and total_layers > 0:
            limit = max(1, limit)
        actual_n = min(unfreeze_last_n_blocks, limit)

        for block in all_blocks[-actual_n:]:
            for p in block.parameters():
                p.requires_grad = True

        if is_main:
            print(
                f"[INFO] SAM-Brain 3D: 总层数 {total_layers}，限制解冻最多 {limit} 层。"
                f"请求 {unfreeze_last_n_blocks} -> 实际解冻最后 {actual_n} 个 blocks/layers"
            )


def _unfreeze_swinvit_layers(
    swinvit: nn.Module, unfreeze_last_n_blocks: int, is_main: bool
):
    """解冻 SwinViT 模型的最后 N 个 Block（用于 BrainSegFounder）

    SwinViT 结构: layers1, layers2... 是 ModuleList，其中包含 BasicLayer。
    BasicLayer 包含 blocks (ModuleList)。
    """
    # SwinViT 通常有 layers1, layers2, layers3, layers4 属性
    stage_names = ["layers1", "layers2", "layers3", "layers4"]
    existing_stages = [name for name in stage_names if hasattr(swinvit, name)]

    # 解冻最后的 norm
    if hasattr(swinvit, "norm"):
        for p in swinvit.norm.parameters():
            p.requires_grad = True

    # 收集所有的 Swin Transformer Blocks
    all_blocks = []
    for stage_name in existing_stages:
        stage_module_list = getattr(swinvit, stage_name)
        # stage_module_list 是 ModuleList，通常只有一个 BasicLayer 元素
        for basic_layer in stage_module_list:
            if hasattr(basic_layer, "blocks"):
                for block in basic_layer.blocks:
                    all_blocks.append(block)

    total_blocks = len(all_blocks)
    if total_blocks == 0:
        if is_main:
            print(
                "[WARNING] BrainSegFounder (SwinViT): 未找到任何 blocks，无法执行 block 级解冻。"
            )
        return

    # 动态限制：解冻层数不超过总层数的一半
    limit = total_blocks // 2
    if unfreeze_last_n_blocks > 0 and total_blocks > 0:
        limit = max(1, limit)
    actual_n = min(unfreeze_last_n_blocks, limit)

    # 解冻最后 N 个 blocks
    for block in all_blocks[-actual_n:]:
        for p in block.parameters():
            p.requires_grad = True

    if is_main:
        print(
            f"[INFO] BrainSegFounder (SwinViT): 总 Block 数 {total_blocks} (来自 {len(existing_stages)} 个 stages)。"
            f"限制解冻最多 {limit} 个。请求 {unfreeze_last_n_blocks} -> 实际解冻最后 {actual_n} 个 blocks"
        )


def _unfreeze_nnunet_layers(
    encoder: nn.Module, unfreeze_last_n_blocks: int, is_main: bool
):
    """解冻 nnUNet encoder 的最后 N 个 stage"""
    if not hasattr(encoder, "stages"):
        if is_main:
            print("[WARNING] nnUNet encoder 没有 stages 属性，跳过解冻")
        return

    stages = encoder.stages
    total_stages = len(stages)
    # 动态限制：解冻层数不超过总层数的一半
    limit = total_stages // 2
    if unfreeze_last_n_blocks > 0 and total_stages > 0:
        limit = max(1, limit)
    actual_n = min(unfreeze_last_n_blocks, limit)

    # 解冻最后 N 个 stages
    for stage in stages[-actual_n:]:
        for p in stage.parameters():
            p.requires_grad = True

    if is_main:
        print(
            f"[INFO] nnUNet (PlainConvEncoder): 总 Stage 数 {total_stages}，限制解冻最多 {limit} 个。"
            f"请求 {unfreeze_last_n_blocks} -> 实际解冻最后 {actual_n} 个 stages"
        )


def _unfreeze_brainfm_layers(
    backbone: nn.Module, unfreeze_last_n_blocks: int, is_main: bool
):
    """解冻 BrainFM (UNet3D) encoder 的最后 N 层"""
    if not hasattr(backbone, "encoders"):
        if is_main:
            print("[WARNING] BrainFM backbone 没有 encoders 属性，跳过解冻")
        return

    encoders = list(backbone.encoders)
    total = len(encoders)
    # 动态限制：解冻层数不超过总层数的一半
    limit = max(1, total // 2)
    actual_n = min(unfreeze_last_n_blocks, limit)

    # 解冻最后 N 个 encoders
    for enc in encoders[-actual_n:]:
        for p in enc.parameters():
            p.requires_grad = True

    if is_main:
        print(
            f"[INFO] BrainFM (UNet3D Encoder): 总 Encoder 数 {total}，限制解冻最多 {limit} 个。"
            f"请求 {unfreeze_last_n_blocks} -> 实际解冻最后 {actual_n} 个 encoders"
        )


def _unfreeze_voxhrnet_layers(
    backbone: nn.Module, unfreeze_last_n_blocks: int, is_main: bool
):
    """解冻 VoxHR-Net 模型的最后 N 个 stage"""
    # 获取所有 stages
    stages = backbone.stages  # [stem_net, stage2, stage3, (stage4)]
    transitions = backbone.transitions  # [transition1, transition2, (transition3)]

    total_stages = len(stages)
    # 动态限制：解冻层数不超过总层数的一半
    limit = total_stages // 2
    if unfreeze_last_n_blocks > 0 and total_stages > 0:
        limit = max(1, limit)
    actual_n = min(unfreeze_last_n_blocks, limit)

    # 解冻最后 N 个 stages
    for stage in stages[-actual_n:]:
        for p in stage.parameters():
            p.requires_grad = True

    # 解冻对应的 transition layers
    # transition[i] 对应 stage[i+1]
    # stages 下标: 0(stem), 1(stage2), 2(stage3)...
    # 假设我们解冻最后 k=actual_n 个 stages.
    # indices: [total_stages-k, ..., total_stages-1]
    # 对应的 transitions indices:
    # 如果 stage[i] 被解冻 (i >= 1)，则 transition[i-1] 也应该被解冻

    start_stage_idx = total_stages - actual_n
    # 只有当 stage 索引 >= 1 时才涉及 transition
    # transition idx = stage idx - 1
    start_trans_idx = max(0, start_stage_idx - 1)

    # 解冻 transition (从 start_trans_idx 到最后)
    # 注意：transition 列表长度通常比 stages 少 1
    for transition in transitions[start_trans_idx:]:
        if transition is not None:
            for p in transition.parameters():
                p.requires_grad = True

    if is_main:
        print(
            f"[INFO] VoxHR-Net: 总 Stage 数 {total_stages}，限制解冻最多 {limit} 个。"
            f"请求 {unfreeze_last_n_blocks} -> 实际解冻最后 {actual_n} 个 stages (+ transitions)"
        )


def _apply_freezing_to_backbone(
    backbone: nn.Module,
    train_mode: str,
    unfreeze_last_n_blocks: int,
    is_main: bool,
    branch_name: str = "",
    use_dora: bool = False,
    dora_r: int = 0,
    dora_alpha: int = 1,
    dora_target_modules: Sequence[str] = (),
):
    """对单个 backbone 应用冻结策略

    Args:
        backbone: 要冻结的 backbone 模块
        train_mode: 训练模式 ('none' | 'head' | 'partial')
        unfreeze_last_n_blocks: 解冻的层数
        is_main: 是否为主进程
        branch_name: 分支名称（用于打印信息）
    """
    prefix = f"[{branch_name}] " if branch_name else ""

    if train_mode == "none":
        if is_main:
            print(f"{prefix}[INFO] 训练模式: none - 全部参数可训练")
        return

    elif train_mode == "head":
        for p in backbone.parameters():
            p.requires_grad = False
        if is_main:
            print(f"{prefix}[INFO] 训练模式: head - 冻结整个 backbone")
        return

    elif train_mode == "partial":
        # 先冻结整个 backbone
        for p in backbone.parameters():
            p.requires_grad = False

        # DoRA for BrainSegFounder (branch-specific, partial mode)
        if use_dora:
            applied = _apply_dora_to_brainsegfounder(
                backbone=backbone,
                r=dora_r,
                alpha=dora_alpha,
                target_modules=dora_target_modules,
                is_main=is_main,
                prefix=prefix,
            )
            if applied > 0:
                return

        backbone_identified = False

        # 1. 检查是否是 nnUNet backbone
        if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "stages"):
            _unfreeze_nnunet_layers(backbone.encoder, unfreeze_last_n_blocks, is_main)
            backbone_identified = True

        # 2. 检查是否是 SAM-Brain 3D backbone
        elif hasattr(backbone, "encoder") and (
            hasattr(backbone.encoder, "blocks") or hasattr(backbone.encoder, "layers")
        ):
            encoder = backbone.encoder
            _unfreeze_sam_brain3d_layers(encoder, unfreeze_last_n_blocks, is_main)
            backbone_identified = True

        # 3. 检查是否是 BrainFM backbone
        elif hasattr(backbone, "encoders"):
            _unfreeze_brainfm_layers(backbone, unfreeze_last_n_blocks, is_main)
            backbone_identified = True

        # 4. 检查是否是 VoxHR-Net Backbone
        elif hasattr(backbone, "stages"):
            _unfreeze_voxhrnet_layers(backbone, unfreeze_last_n_blocks, is_main)
            backbone_identified = True

        # 5. 检查是否是 BrainSegFounder 或 DINOv2 backbone
        elif hasattr(backbone, "model"):
            vit_model = backbone.model
            if hasattr(vit_model, "layers1") and hasattr(vit_model, "layers4"):
                _unfreeze_swinvit_layers(vit_model, unfreeze_last_n_blocks, is_main)
                backbone_identified = True
            elif hasattr(vit_model, "swinViT"):
                _unfreeze_swinvit_layers(
                    vit_model.swinViT, unfreeze_last_n_blocks, is_main
                )
                backbone_identified = True
            elif hasattr(vit_model, "blocks"):
                _unfreeze_vit_layers(vit_model, unfreeze_last_n_blocks, is_main)
                backbone_identified = True

        if not backbone_identified:
            if is_main:
                print(f"{prefix}[WARNING] 未知的 backbone 类型，无法应用 partial 冻结策略")

        if is_main:
            print(
                f"{prefix}[INFO] 训练模式: partial - 冻结 backbone 前面的层，"
                f"解冻最后 {unfreeze_last_n_blocks} 个块"
            )

    else:
        raise ValueError(
            f"Unknown train_mode: {train_mode}. Expected 'none', 'head', or 'partial'."
        )


def setup_dual_branch_freezing(
    model: nn.Module,
    train_mode1: str,
    train_mode2: str,
    unfreeze_last_n_blocks1: int,
    unfreeze_last_n_blocks2: int,
    use_dora_branch2: bool = False,
    dora_branch2_r: int = 0,
    dora_branch2_alpha: int = 1,
    dora_branch2_target_modules: Sequence[str] = (),
    is_main: bool = True,
):
    """设置双分支模型的参数冻结策略

    分别对两个分支的 backbone 应用各自的冻结策略。

    Args:
        model: 双分支模型 (DualBranchClassifier)
        train_mode1: 主分支训练模式 ('none' | 'head' | 'partial')
        train_mode2: 副分支训练模式 ('none' | 'head' | 'partial')
        unfreeze_last_n_blocks1: 主分支解冻的层数
        unfreeze_last_n_blocks2: 副分支解冻的层数
        is_main: 是否为主进程
    """
    net = model.module if hasattr(model, "module") else model

    if is_main:
        print("\n" + "=" * 60)
        print("[INFO] 设置双分支参数冻结策略")
        print("=" * 60)

    # 检查模型是否具有双分支结构
    if not hasattr(net, "backbone1") or not hasattr(net, "backbone2"):
        raise ValueError(
            "模型不是双分支结构，缺少 backbone1 或 backbone2 属性"
        )

    # 应用主分支冻结策略
    if is_main:
        print("\n[Branch 1] 主分支冻结策略:")
    _apply_freezing_to_backbone(
        net.backbone1,
        train_mode1,
        unfreeze_last_n_blocks1,
        is_main,
        branch_name="Branch1",
    )

    # 应用副分支冻结策略
    if is_main:
        print("\n[Branch 2] 副分支冻结策略:")
    _apply_freezing_to_backbone(
        net.backbone2,
        train_mode2,
        unfreeze_last_n_blocks2,
        is_main,
        branch_name="Branch2",
        use_dora=use_dora_branch2,
        dora_r=dora_branch2_r,
        dora_alpha=dora_branch2_alpha,
        dora_target_modules=dora_branch2_target_modules,
    )

    # 设置冻结的 BatchNorm 为 eval 模式
    set_frozen_batchnorm_eval(model)

    if is_main:
        print("=" * 60 + "\n")
