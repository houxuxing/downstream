"""工具函数模块"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler, Subset
from config import Config

# 设置中文字体
matplotlib.rcParams["font.family"] = [
    "SimHei",
    "WenQuanYi Micro Hei",
    "Heiti TC",
    "Microsoft YaHei",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def save_plots(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_dir: str,
    cfg: Config,
):
    """保存训练/验证曲线"""
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("Loss")

    plt.subplot(2, 1, 2)
    plt.plot(train_accs, label="train_acc")
    plt.plot(val_accs, label="val_acc")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("Accuracy")

    plt.tight_layout()

    # 使用 png_save_name（而不是 model_save_name）避免生成 ".pth_时间戳.png" 这种混淆命名
    stem_source = cfg.png_save_name or cfg.model_save_name or "training_curve.png"
    stem = Path(stem_source).stem
    plt.savefig(
        os.path.join(save_dir, f"{stem}_{ts}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    classes: Tuple[str, ...],
    epoch,
    prefix: str = "",
):
    """打印混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    print(f"\n{prefix} 混淆矩阵 (第{epoch}轮):")
    header = "真实类别 \\ 预测类别 | " + " | ".join(classes)
    print(header)
    print("-" * len(header))

    for i, row in enumerate(cm):
        class_name = classes[i]
        row_data = " | ".join([f"{x:>4}" for x in row])
        print(f"{class_name:14} | {row_data}")

    print("\n各类别统计:")
    for i, row in enumerate(cm):
        total = sum(row)
        correct = row[i]
        accuracy = correct / total if total > 0 else 0
        print(f"{classes[i]}: {correct}/{total} = {accuracy:.2%}")


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
) -> Dict[str, Any]:
    """计算分类指标（不依赖 sklearn）

    返回：
      - acc
      - macro_recall（各类 recall 平均，等价 balanced accuracy）
      - macro_f1
      - per_class_recall / per_class_precision / per_class_f1
      - confusion_matrix
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0, got {num_classes}")

    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true/y_pred shape mismatch: {yt.shape} vs {yp.shape}")

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(yt.tolist(), yp.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    total = int(cm.sum())
    acc = float(np.trace(cm) / total) if total > 0 else 0.0

    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)  # 真值计数（每类样本数）
    pred_count = cm.sum(axis=0).astype(np.float64)  # 预测计数

    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    macro_recall = float(recall.mean())
    macro_f1 = float(f1.mean())

    return {
        "acc": acc,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_recall": recall.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": cm,
    }


def build_weighted_sampler(
    train_subset: Subset,
    dataset,
    cfg: Config,
    return_info: bool = False,
):
    """为训练集构建 WeightedRandomSampler（按类频率的倒数加权，期望采样近似均衡）

    Args:
        train_subset: torch.utils.data.Subset(full_dataset, train_idx_used)
        dataset: 原始 full_dataset（需包含 valid_samples 与 DX_group）
        cfg: Config
        return_info: 是否同时返回用于日志打印的统计信息
    """
    if not hasattr(train_subset, "indices"):
        raise ValueError("train_subset must be a torch.utils.data.Subset with .indices")

    # 统计 train_subset 内各类样本数
    class_counts: Dict[str, int] = {c: 0 for c in cfg.classes}
    unknown: Dict[str, int] = defaultdict(int)
    for idx in train_subset.indices:
        dx = dataset.valid_samples[idx]["DX_group"]
        if dx in class_counts:
            class_counts[dx] += 1
        else:
            unknown[dx] += 1

    # 计算每类权重：1/count（缺失类权重设为 0）
    class_weights: Dict[str, float] = {
        k: (1.0 / v if v > 0 else 0.0) for k, v in class_counts.items()
    }

    # 生成每个样本的权重
    weights_per_sample: List[float] = []
    for idx in train_subset.indices:
        dx = dataset.valid_samples[idx]["DX_group"]
        weights_per_sample.append(float(class_weights.get(dx, 0.0)))

    num_samples = int(cfg.sampler_num_samples or len(train_subset))
    if num_samples <= 0:
        raise ValueError(f"sampler num_samples must be > 0, got {num_samples}")

    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))

    sampler = WeightedRandomSampler(
        weights=weights_per_sample,
        num_samples=num_samples,
        replacement=bool(cfg.sampler_replacement),
        generator=generator,
    )

    if not return_info:
        return sampler

    # 期望采样占比（归一化权重总和）
    weight_sum_per_class: Dict[str, float] = {
        k: float(v) * float(class_weights[k]) for k, v in class_counts.items()
    }
    total_w = float(sum(weight_sum_per_class.values()))
    expected_prob = {
        k: (weight_sum_per_class[k] / total_w if total_w > 0 else 0.0)
        for k in cfg.classes
    }

    info = {
        "class_counts": class_counts,
        "class_weights": class_weights,
        "expected_prob": expected_prob,
        "unknown_counts": dict(unknown),
        "num_samples": num_samples,
        "replacement": bool(cfg.sampler_replacement),
    }
    return sampler, info


def split_dataset_by_patient(
    dataset,
    test_size: float = 0.3,
    val_ratio: float = 0.5,
    random_state: int = 42,
) -> Tuple[List[int], List[int], List[int], Any, Any, Any]:
    """
    按患者级别拆分数据集

    Returns:
        train_idx, val_idx, test_idx, train_ptids, val_ptids, test_ptids
    """
    df_all = pd.DataFrame(dataset.valid_samples)

    # 每个患者只取一条代表记录用于 stratify
    if "Month" in df_all.columns:
        df_all["Month"] = pd.to_numeric(df_all["Month"], errors="coerce")
        df_patient = (
            df_all.sort_values(["PTID", "Month"], na_position="last")
            .groupby("PTID")
            .first()
            .reset_index()
        )
    else:
        df_patient = df_all.groupby("PTID").first().reset_index()

    # 拆分训练集和临时集
    train_ptids, temp_ptids = train_test_split(
        df_patient["PTID"],
        test_size=test_size,
        random_state=random_state,
        stratify=df_patient["DX_group"],
    )

    # 从临时集中拆分验证集和测试集（修复：stratify 顺序对齐 temp_ptids）
    ptid_to_dx = dict(zip(df_patient["PTID"].tolist(), df_patient["DX_group"].tolist()))
    temp_y = [ptid_to_dx[p] for p in temp_ptids]  # 与 temp_ptids 同顺序一一对应

    val_ptids, test_ptids = train_test_split(
        temp_ptids,
        test_size=val_ratio,
        random_state=random_state,
        stratify=temp_y,
    )

    train_idx = df_all.index[df_all["PTID"].isin(train_ptids)].tolist()
    val_idx = df_all.index[df_all["PTID"].isin(val_ptids)].tolist()
    test_idx = df_all.index[df_all["PTID"].isin(test_ptids)].tolist()

    return train_idx, val_idx, test_idx, train_ptids, val_ptids, test_ptids


def make_balanced_indices(
    idx_list: List[int],
    dataset,
    seed: int = 42,
    expected_classes: Optional[List[str]] = None,
    base_class: Optional[str] = None,
) -> List[int]:
    """
    对各类做均衡下采样。
    - base_class is None: 以最少样本类为基准（原有行为）
    - base_class is not None: 以指定类别样本数为基准（例如 pMCI）

    Args:
        idx_list: 样本索引列表
        dataset: 数据集对象
        seed: 随机种子
        expected_classes: 期望的类别列表（默认 None 表示自动检测）
        base_class: 作为下采样基准的类别名；若缺失则回退为不下采样（返回原 idx）
    """
    rng = np.random.RandomState(seed)

    class_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx in idx_list:
        dx = dataset.valid_samples[idx]["DX_group"]
        class_to_indices[dx].append(idx)

    class_counts = {cls: len(indices) for cls, indices in class_to_indices.items()}
    if not class_counts:
        print("[WARNING] make_balanced_indices 收到空 idx_list，直接返回空列表")
        return []

    # 自动检测类别或使用指定的类别列表
    if expected_classes is None:
        # 尝试使用数据集的类别定义，否则从数据中检测
        if hasattr(dataset, "LABEL_MAP"):
            expected_classes = list(dataset.LABEL_MAP.keys())
        else:
            expected_classes = sorted(class_counts.keys())

    missing = [c for c in expected_classes if c not in class_counts]
    if missing:
        print(f"[WARNING] 该集合缺少类别: {missing}，将仅对存在类别做下采样")

    print(f"[BALANCE] 该集合各类别样本数: {class_counts}")
    if base_class is not None:
        if base_class not in class_counts:
            print(
                f"[WARNING] 指定基准类别 '{base_class}' 在该集合中不存在，"
                "跳过下采样并返回原始索引。"
            )
            return list(idx_list)
        base_n = int(class_counts[base_class])
        if base_n <= 0:
            print(
                f"[WARNING] 指定基准类别 '{base_class}' 样本数 <= 0，"
                "跳过下采样并返回原始索引。"
            )
            return list(idx_list)
        print(f"[BALANCE] 以指定类别 '{base_class}' 为基准，样本数 = {base_n}")
    else:
        min_class = min(class_counts.items(), key=lambda x: x[1])
        base_n = int(min_class[1])
        print(f"[BALANCE] 以最少样本类别 '{min_class[0]}' 为基准，样本数 = {base_n}")

    balanced_idx = []
    for cls in expected_classes:
        cls_indices = class_to_indices.get(cls, [])
        if len(cls_indices) == 0:
            continue
        n_select = min(len(cls_indices), base_n)
        chosen = rng.choice(cls_indices, size=n_select, replace=False).tolist()
        balanced_idx.extend(chosen)

    rng.shuffle(balanced_idx)
    return balanced_idx


def print_dataset_info(
    dataset,
    train_ptids,
    val_ptids,
    test_ptids,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    balanced_train_idx: List[int],
    balanced_val_idx: Optional[List[int]] = None,
    balanced_test_idx: Optional[List[int]] = None,
    cfg: Optional[Config] = None,
    sampler_info: Optional[Dict[str, Any]] = None,
):
    """打印数据集信息（原始划分 + 实际使用）"""
    df_all = pd.DataFrame(dataset.valid_samples)

    print("【全体样本】每类患者数：")
    print(df_all.groupby("DX_group")["PTID"].nunique())
    print("【全体样本】每类影像数：")
    print(df_all["DX_group"].value_counts())

    def _print_split(title: str, idx_list: Optional[List[int]]):
        if idx_list is None:
            print(f"\n【{title}】未提供 idx_list，跳过")
            return
        if len(idx_list) == 0:
            print(f"\n【{title}】idx_list 为空")
            return

        df = df_all.iloc[idx_list]
        print(f"\n【{title}】每类患者数：")
        print(df.groupby("DX_group")["PTID"].nunique())
        print(f"【{title}】每类影像数：")
        print(df["DX_group"].value_counts())

    # 原始划分（患者级拆分后得到的影像索引集合）
    _print_split("训练集（原始划分）", train_idx)
    _print_split("验证集（原始划分）", val_idx)
    _print_split("测试集（原始划分）", test_idx)

    # 实际使用的划分（可能为全量、下采样或结合 sampler）
    _print_split("训练集（实际使用）", balanced_train_idx)
    _print_split("验证集（实际使用）", balanced_val_idx)
    _print_split("测试集（实际使用）", balanced_test_idx)

    if cfg is not None and cfg.balance_strategy == "weighted_sampler":
        print("\n[BALANCE] 训练集均衡策略: WeightedRandomSampler")
        if sampler_info is not None:
            print(
                f"[BALANCE] sampler: replacement={sampler_info.get('replacement')}, "
                f"num_samples={sampler_info.get('num_samples')}"
            )
            print(f"[BALANCE] class_counts: {sampler_info.get('class_counts')}")
            print(f"[BALANCE] class_weights(1/count): {sampler_info.get('class_weights')}")
            print(f"[BALANCE] expected_prob: {sampler_info.get('expected_prob')}")
            if sampler_info.get("unknown_counts"):
                print(f"[BALANCE] unknown DX_group: {sampler_info.get('unknown_counts')}")


def print_test_results(
    test_targets: List[int],
    test_preds: List[int],
    test_ids: List[str],
    test_logits: List[List[float]],
    classes: Tuple[str, ...],
):
    """打印测试结果详情"""
    case_dict: Dict[Tuple[int, int], List] = defaultdict(list)
    for t, p, sid, logit in zip(test_targets, test_preds, test_ids, test_logits):
        case_dict[(t, p)].append((sid, logit))

    print("\n[TEST] 各真实/预测组合对应的样本 (PTID_ImageID + logits)：")
    for i, true_name in enumerate(classes):
        for j, pred_name in enumerate(classes):
            key = (i, j)
            samp_list = case_dict.get(key, [])
            print(f"\n真实为{true_name}，预测为{pred_name}：")
            if not samp_list:
                print("  （无样本）")
            else:
                for sid, logit in samp_list:
                    logit_str = ", ".join(f"{v:.4f}" for v in logit)
                    print(f"  {sid} | logits = [{logit_str}]")
