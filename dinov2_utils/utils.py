# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random
import subprocess
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn


logger = logging.getLogger("dinov2")


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, strict=True):
    """
    加载预训练权重到模型。
    
    Args:
        model: 目标模型
        pretrained_weights: 预训练权重路径或 URL
        checkpoint_key: checkpoint 中的 key（如 "teacher"）
        strict: 是否严格匹配。如果为 True，任何不匹配都会抛出错误
    """
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(
            pretrained_weights, map_location="cpu"
        )
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    
    if checkpoint_key is not None and checkpoint_key in state_dict:
        print(f"[INFO] 使用 checkpoint key: '{checkpoint_key}'")
        state_dict = state_dict[checkpoint_key]
    
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    # If checkpoint wraps the ViT under feature_model.vit_model, strip that prefix.
    vit_prefixes = ("feature_model.vit_model.", "vit_model.")
    if any(k.startswith(vit_prefixes) for k in state_dict):
        filtered = {}
        for k, v in state_dict.items():
            for prefix in vit_prefixes:
                if k.startswith(prefix):
                    filtered[k[len(prefix):]] = v
                    break
        state_dict = filtered

    model_state = model.state_dict()
    
    # 检查形状不匹配的参数
    shape_mismatch = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape != v.shape:
            shape_mismatch.append({
                "name": k,
                "ckpt_shape": tuple(v.shape),
                "model_shape": tuple(model_state[k].shape),
            })
    
    # 检查缺失的参数（在 checkpoint 中有，但模型中没有）
    unexpected_keys = [k for k in state_dict.keys() if k not in model_state]
    
    # 检查模型中有但 checkpoint 中没有的参数
    missing_keys = [k for k in model_state.keys() if k not in state_dict]
    
    # 打印诊断信息
    print(f"\n{'='*60}")
    print(f"[预训练权重加载诊断]")
    print(f"{'='*60}")
    print(f"预训练权重路径: {pretrained_weights}")
    print(f"Checkpoint 参数数量: {len(state_dict)}")
    print(f"模型参数数量: {len(model_state)}")
    print(f"形状匹配的参数: {len(state_dict) - len(shape_mismatch) - len(unexpected_keys)}")
    print(f"形状不匹配的参数: {len(shape_mismatch)}")
    print(f"Checkpoint 中多余的参数: {len(unexpected_keys)}")
    print(f"模型中缺失的参数: {len(missing_keys)}")
    
    if shape_mismatch:
        print(f"\n[形状不匹配的参数详情]:")
        for item in shape_mismatch[:10]:  # 只显示前 10 个
            print(f"  - {item['name']}: checkpoint={item['ckpt_shape']} vs model={item['model_shape']}")
        if len(shape_mismatch) > 10:
            print(f"  ... 以及其他 {len(shape_mismatch) - 10} 个参数")
    
    if unexpected_keys:
        print(f"\n[Checkpoint 中多余的参数（前10个）]:")
        for k in unexpected_keys[:10]:
            print(f"  - {k}")
        if len(unexpected_keys) > 10:
            print(f"  ... 以及其他 {len(unexpected_keys) - 10} 个参数")
    
    if missing_keys:
        print(f"\n[模型中缺失的参数（前10个）]:")
        for k in missing_keys[:10]:
            print(f"  - {k}")
        if len(missing_keys) > 10:
            print(f"  ... 以及其他 {len(missing_keys) - 10} 个参数")
    
    print(f"{'='*60}\n")
    
    # 严格模式：如果有任何不匹配，抛出错误
    if strict:
        error_msgs = []
        
        if shape_mismatch:
            error_msgs.append(f"发现 {len(shape_mismatch)} 个形状不匹配的参数！")
        
        if unexpected_keys:
            error_msgs.append(f"发现 {len(unexpected_keys)} 个多余的参数（checkpoint 中有但模型中没有）！")
            
        if missing_keys:
            error_msgs.append(f"发现 {len(missing_keys)} 个缺失的参数（模型中有但 checkpoint 中没有）！")
            
        if error_msgs:
            raise RuntimeError(
                "预训练权重加载失败（严格模式）：\n" + 
                "\n".join(error_msgs) + 
                "\n请检查模型架构与权重文件是否完全一致。"
            )
    
    # 只加载形状匹配的参数
    matched_state = {k: v for k, v in state_dict.items() 
                     if k in model_state and model_state[k].shape == v.shape}
    
    # 如果 strict=True，使用 PyTorch 原生严格加载；否则允许不匹配
    msg = model.load_state_dict(matched_state, strict=strict)
    
    print(f"[INFO] 成功加载 {len(matched_state)} 个预训练参数")
    
    return msg


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


class CosineScheduler(object):
    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        warmup_iters=0,
        start_warmup_value=0,
        freeze_iters=0,
    ):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
