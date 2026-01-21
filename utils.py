"""工具函数模块"""

import os
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
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
    plt.savefig(
        os.path.join(save_dir, f"{cfg.model_save_name}_{ts}.png"),
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
) -> List[int]:
    """
    按最少样本数量的类别为基准，对四类做均衡下采样
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

    expected = ["CN", "sMCI", "pMCI", "Dementia"]
    missing = [c for c in expected if c not in class_counts]
    if missing:
        print(f"[WARNING] 该集合缺少类别: {missing}，将仅对存在类别做下采样")

    min_class = min(class_counts.items(), key=lambda x: x[1])
    base_n = min_class[1]

    print(f"[BALANCE] 该集合各类别样本数: {class_counts}")
    print(f"[BALANCE] 以最少样本类别 '{min_class[0]}' 为基准，样本数 = {base_n}")

    balanced_idx = []
    for cls in ["CN", "sMCI", "pMCI", "Dementia"]:
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
):
    """打印数据集信息（支持 train/val/test 均衡下采样）"""
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

    # 均衡下采样后的划分（你当前 main.py 实际使用的集合）
    _print_split("训练集（均衡下采样）", balanced_train_idx)
    _print_split("验证集（均衡下采样）", balanced_val_idx)
    _print_split("测试集（均衡下采样）", balanced_test_idx)


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
