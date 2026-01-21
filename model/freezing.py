"""参数冻结工具模块

包含模型参数冻结/解冻的相关工具函数。
"""

import torch.nn as nn


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
        
        # 解冻最后 N 个 Transformer 块
        vit_model = net.backbone.model  # DINOv2 ViT 模型
        
        # 解冻最后的 LayerNorm
        if hasattr(vit_model, 'norm'):
            for p in vit_model.norm.parameters():
                p.requires_grad = True
        
        # 解冻最后 N 个块
        if hasattr(vit_model, 'blocks'):
            blocks = vit_model.blocks
            total_blocks = len(blocks)
            
            # 处理 chunked blocks 的情况
            if hasattr(vit_model, 'chunked_blocks') and vit_model.chunked_blocks:
                # 统计总块数
                all_blocks = []
                for chunk in blocks:
                    for block in chunk:
                        if not isinstance(block, nn.Identity):
                            all_blocks.append(block)
                total_blocks = len(all_blocks)
                
                # 解冻最后 N 个块
                for block in all_blocks[-unfreeze_last_n_blocks:]:
                    for p in block.parameters():
                        p.requires_grad = True
            else:
                # 非 chunked 情况
                for block in blocks[-unfreeze_last_n_blocks:]:
                    for p in block.parameters():
                        p.requires_grad = True
        
        if is_main:
            print(f"[INFO] 训练模式: partial - 冻结 backbone 前面的层，解冻最后 {unfreeze_last_n_blocks} 个 Transformer 块")

    else:
        raise ValueError(f"Unknown train_mode: {train_mode}. Expected 'none', 'head', or 'partial'.")

    set_frozen_batchnorm_eval(model)
