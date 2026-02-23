"""数据集模块

使用 MONAI 进行医学影像数据处理。
MONAI (Medical Open Network for AI) 是专为医学影像设计的深度学习框架。
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# MONAI transforms for medical image processing
# 使用具体模块路径导入以避免类型检查警告
from monai.transforms.io.array import LoadImage
from monai.transforms.intensity.array import (
    ScaleIntensityRangePercentiles,
    RandScaleIntensity,
    RandShiftIntensity,
    NormalizeIntensity,
)
from monai.transforms.spatial.array import (
    Orientation,
    Spacing,
)
from monai.transforms.croppad.array import (
    CropForeground,
    SpatialPad,
    RandSpatialCrop,
    ResizeWithPadOrCrop,
)
from monai.transforms import (
    Compose,
    RandFlip,
    Crop,
    Randomizable,
    RandRotate90,
    OneOf,
    RandGaussianSharpen,
    RandGaussianSmooth,
    RandGibbsNoise,
)
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import get_random_patch, get_valid_patch_size
import math
import json


# -------------------------
# 从 anatcl.dataset.dataset_split 提取的辅助函数
# -------------------------


def normalize_image_id(x: str) -> str:
    """标准化 ImageID 格式"""
    s = str(x).strip()
    # if already I12345, keep
    if re.match(r"^I\d+$", s, flags=re.IGNORECASE):
        return s.upper()
    # if pure digits, prefix I
    if re.match(r"^\d+$", s):
        return ("I" + s).upper()
    # otherwise best-effort: extract I\d+
    m = re.search(r"I\d+", s, flags=re.IGNORECASE)
    if m:
        return m.group(0).upper()
    return s.upper()


def normalize_hemi(x: str) -> str:
    """标准化半球标识"""
    s = str(x).strip().lower()
    # common values: lh/rh, L/R, left/right
    if s.startswith("l"):
        return "lh"
    if s.startswith("r"):
        return "rh"
    # unknown -> keep
    return s


def build_age_map(age_csv_path: Path) -> Dict[Tuple[str, str], float]:
    """从 CSV 构建年龄映射表"""
    df = pd.read_csv(age_csv_path)
    for col in ["PTID", "ImageID", "Age"]:
        if col not in df.columns:
            raise ValueError(f"age_csv missing required column: {col}")

    df["PTID"] = df["PTID"].astype(str).str.strip()
    df["ImageID"] = df["ImageID"].apply(normalize_image_id)

    # drop NaN Age rows
    df = df.dropna(subset=["Age"])

    age_map: Dict[Tuple[str, str], float] = {}
    for _, row in df.iterrows():
        key = (row["PTID"], row["ImageID"])
        # if duplicates exist, last one wins; you can also assert uniqueness if needed
        age_map[key] = float(row["Age"])
    return age_map


def load_region_order(region_order_json: Path) -> List[Tuple[str, str]]:
    """从 JSON 加载 region_order"""
    obj = json.loads(region_order_json.read_text(encoding="utf-8"))
    return [(a, b) for a, b in obj]


def _measure_key_set(
    measure_csv_path: Path,
    hemi_col: str = "hemi",
    region_col: str = "StructName",
) -> set:
    """读取 measure csv 并返回 (hemi, StructName) 的集合"""
    df = pd.read_csv(measure_csv_path, usecols=[hemi_col, region_col])
    if hemi_col not in df.columns or region_col not in df.columns:
        raise ValueError(
            f"Measure file missing columns '{hemi_col}/{region_col}': {measure_csv_path}"
        )

    df[hemi_col] = df[hemi_col].apply(normalize_hemi)
    df[region_col] = df[region_col].astype(str)
    return set(zip(df[hemi_col].tolist(), df[region_col].tolist()))


def measure_missing_rois(
    measure_csv_path: Path,
    region_order: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], float]:
    """
    返回：
      - missing_keys: 缺失 ROI 列表 (hemi, StructName)
      - missing_ratio: 缺失比例
    """
    present = _measure_key_set(measure_csv_path)
    missing = [k for k in region_order if k not in present]
    ratio = float(len(missing)) / float(max(len(region_order), 1))
    return missing, ratio


def vectorize_measures(
    measure_csv_path: Path,
    region_order: List[Tuple[str, str]],
    hemi_col: str = "hemi",
    region_col: str = "StructName",
    keep_cols: Tuple[str, str, str] = ("ThickAvg", "GrayVol", "SurfArea"),
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    将 measure csv 向量化为 shape: [len(region_order), 3] 的数组
    列顺序为 (ThickAvg, GrayVol, SurfArea)
    """
    df = pd.read_csv(measure_csv_path)
    for c in [hemi_col, region_col, *keep_cols]:
        if c not in df.columns:
            raise ValueError(f"Measure file missing column '{c}': {measure_csv_path}")

    df[hemi_col] = df[hemi_col].apply(normalize_hemi)
    df[region_col] = df[region_col].astype(str)

    # index by (hemi, StructName) -> values
    table: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    for _, row in df.iterrows():
        key = (row[hemi_col], row[region_col])
        vals = tuple(float(row[c]) for c in keep_cols)
        table[key] = vals

    out = np.full((len(region_order), len(keep_cols)), fill_value, dtype=np.float32)
    for i, key in enumerate(region_order):
        if key in table:
            out[i, :] = np.asarray(table[key], dtype=np.float32)
    return out


# 3D插值
def resize_3d_volume_trilinear(
    img: torch.Tensor, target_size: Tuple[int, int, int]
) -> torch.Tensor:
    """Resize a 3D volume tensor to target_size using trilinear interpolation.

    This is generally reasonable for medical *intensity* images (continuous values).
    NOTE: This resizes by voxel grid size only (does not account for physical spacing / affine).
    Expected input shapes:
      - (1, D, H, W)  or  (C, D, H, W)

    Returns the same tensor rank with spatial size == target_size.
    """
    if img.dim() != 4:
        raise ValueError(
            f"Expected a 4D tensor (C,D,H,W), got shape={tuple(img.shape)}"
        )

    td, th, tw = map(int, target_size)
    c, d, h, w = img.shape
    if (d, h, w) == (td, th, tw):
        return img

    # F.interpolate expects (N, C, D, H, W)
    x = img.unsqueeze(0)  # (1, C, D, H, W)
    x = F.interpolate(x, size=(td, th, tw), mode="trilinear", align_corners=False)
    return x.squeeze(0)


class RandomResizedCrop3d(Crop, Randomizable):
    def __init__(
        self,
        size,
        in_slice_scale,
        cross_slice_scale,
        interpolation="trilinear",
        aspect_ratio=(0.9, 1 / 0.9),
    ):
        """
        Adapting torch RandomResizedCrop to 3D data by separating in-slice/in-plane and cross-slice dimensions.

        Args:
            size: Size of output image.
            in_slice_scale: Range of the random size of the cropped in-slice/in-plane dimensions.
            cross_slice_scale: Range of the random size of the cropped cross-slice dimensions.
            interpolation: 3D interpolation method, defaults to 'trilinear'.
            aspect_ratio: Range of aspect ratios of the cropped in-slice/in-plane dimensions.
        """
        super().__init__()
        self.size = size
        self.in_slice_scale = in_slice_scale
        self.cross_slice_scale = cross_slice_scale
        self.interpolation = interpolation
        self.aspect_ratio = aspect_ratio
        self._slices: tuple[slice, ...] = ()

    def get_in_slice_crop(self, height, width):
        """
        Adapted from torchvision RandomResizedCrop, applied to the in-slice/in-plane dimensions
        """
        area = height * width

        log_ratio = math.log(self.aspect_ratio[0]), math.log(self.aspect_ratio[1])
        for _ in range(10):
            target_area = area * self.R.uniform(*self.in_slice_scale)
            aspect_ratio = math.exp(self.R.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                return h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.aspect_ratio):
            w = width
            h = int(round(w / min(self.aspect_ratio)))
        elif in_ratio > max(self.aspect_ratio):
            h = height
            w = int(round(h * max(self.aspect_ratio)))
        else:  # whole image
            w = width
            h = height
        return h, w

    def randomize(self, img_size):
        # first two dimensions are dicom slice dims/in-plane dims, third is number of slices
        height, width, depth = img_size

        # get in-slice crop size
        crop_h, crop_w = self.get_in_slice_crop(height, width)

        # get cross-slice crop size
        crop_d = int(round(depth * self.R.uniform(*self.cross_slice_scale)))

        crop_size = (crop_h, crop_w, crop_d)
        valid_size = get_valid_patch_size(img_size, crop_size)
        self._slices = get_random_patch(img_size, valid_size, self.R)

    def __call__(self, img, lazy=False):
        self.randomize(img.shape[1:])
        cropped = super().__call__(img=img, slices=self._slices)
        resized = F.interpolate(
            cropped.unsqueeze(0), size=self.size, mode=self.interpolation
        ).squeeze(0)
        return resized


def normalize_dx(x: str) -> Optional[str]:
    """标准化诊断标签"""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    sl = s.lower().replace(" ", "").replace("-", "").replace("_", "")

    if sl == "cn":
        return "CN"
    if sl in {"smci", "stmci", "stablemci"}:
        return "sMCI"
    if sl in {"pmci", "progressivemci"}:
        return "pMCI"
    if sl in {
        "dementia",
        "ad",
        "alzheimers",
        "alzheimer",
        "alzheimersdisease",
        "alzheimerdisease",
    }:
        return "Dementia"
    if s in {"CN", "sMCI", "pMCI", "Dementia"}:
        return s
    return None


def extract_folder_key(folder_name: str) -> str:
    """从文件夹名提取标准化的 key"""
    m = re.search(r"\d{3}_S_\d{4}_I\d+", folder_name)
    if m:
        return m.group(0)
    return folder_name.replace("_real", "").strip()


class MRIDataset(Dataset):
    """四分类 MRI 数据集"""

    LABEL_MAP = {"CN": 0, "sMCI": 1, "pMCI": 2, "Dementia": 3}
    VBM_NAME = "brain.nii.gz"

    # 【诊断】用于记录 __getitem__ 调用情况
    _debug_getitem_calls = 0

    def __init__(
        self,
        data_dir: str,
        csv_file_processed: str,
        target_size: Optional[Tuple[int, int, int]] = None,
        is_train: bool = True,
        age_csv: Optional[str] = None,
        measure_root: Optional[str] = None,
        region_order_json: Optional[str] = None,
        enable_memory_efficient: bool = False,
        min_int: float = -1.0,
        resize_scale: float = 1.0,
        use_nnunet_zscore: bool = False,
    ):
        """初始化 MRI 数据集。

        Args:
            data_dir: MRI 数据根目录路径，包含子目录 (CN_fs, P1_fs, ..., P6_fs)，
                每个子目录下是患者文件夹，结构为 {PTID}_{ImageID}/mri/brain.nii.gz。
            csv_file_processed: 处理后的 CSV 文件路径，必须包含以下列：
                - PTID: 患者 ID
                - ImageID: 影像 ID
                - VISCODE: 访视代码
                - NEW_DX: 诊断标签 (CN/sMCI/pMCI/Dementia)
                可选列：Month (用于排序)。
            target_size: 目标图像尺寸 (D, H, W)，若设置则会对图像进行三线性插值重采样。
                默认 None 表示保持原始尺寸。
            is_train: 是否为训练模式。目前数据增强已禁用，此参数暂未生效。
            age_csv: 年龄 CSV 文件路径，必须包含 PTID, ImageID, Age 列。
                若提供，则用于 Global Loss 计算。默认 None 表示不加载年龄。
            measure_root: 测量数据根目录路径，包含各样本的脑区测量 CSV 文件。
                文件命名格式应为 {PTID}_{ImageID}.csv，包含 ThickAvg, GrayVol, SurfArea 等测量值。
                若提供，则用于 Global Loss 计算。默认 None 表示不加载测量数据。
            region_order_json: 脑区顺序 JSON 文件路径，定义测量向量化时的 ROI 顺序。
                格式为 [["hemi", "StructName"], ...] 的二维列表。
                若 measure_root 已设置，则此参数必须提供。
            enable_memory_efficient: [已弃用] 此参数保留用于向后兼容，但已不再使用。
                MONAI 的 LoadImage 会自动处理内存优化。
            min_int: 背景裁剪阈值，默认 -1.0。
            resize_scale: 重采样缩放比例，默认 1.0。
        """
        self.data_dir = data_dir
        self.csv_file_processed = csv_file_processed
        self.region_order_json = region_order_json
        self.target_size = target_size
        self.is_train = is_train
        self.age_csv = age_csv
        self.measure_root = Path(measure_root) if measure_root else None
        self.region_order: Optional[List[Tuple[str, str]]] = None
        self.train_idx_set: set = set()
        self.enable_memory_efficient = enable_memory_efficient
        self.min_int = min_int
        self.resize_scale = resize_scale
        self.use_nnunet_zscore = use_nnunet_zscore
        self._folder_map: Dict[str, str] = {}
        # LRU-style cache with size limit to prevent unbounded memory growth
        self._measures_cache_max_size = 10000
        self._measures_cache: Dict[str, torch.Tensor] = {}
        self._measure_meta_cache: Dict[
            str, Tuple[List[Tuple[str, str]], float, bool]
        ] = {}

        # 构建年龄映射
        self.age_map = build_age_map(Path(age_csv)) if age_csv else {}

        # 加载 region_order
        if region_order_json and os.path.isfile(region_order_json):
            self.region_order = load_region_order(Path(region_order_json))

        # 验证配置
        if self.measure_root is not None and self.region_order is None:
            raise ValueError("measure_root 已设置但 region_order_json 缺失")

        # 控制初始化阶段 DROP 日志量（避免打印过多拖慢）
        self.drop_log_limit = 50
        self._drop_log_count = 0

        self._measure_index = self._build_measure_index()
        self._folder_map = self._build_folder_map()
        self.valid_samples = self._load_samples(
            self.csv_file_processed, self._folder_map
        )

        # ---------------------------------------------------------
        # 预训练流程对齐配置
        # ---------------------------------------------------------
        # 使用 MONAI 定义数据加载和预处理 (Array 模式)
        self.loader = LoadImage(
            image_only=False, ensure_channel_first=True, dtype=np.float32
        )

        def _to_meta(x):
            # LoadImage(image_only=False) -> (img, meta_dict)
            if isinstance(x, (tuple, list)) and len(x) == 2:
                img, meta = x
                img = torch.as_tensor(img, dtype=torch.float32)
                return MetaTensor(img, meta=meta)

            # 如果仍然是“单输出”（说明上面的 image_only=False 没生效或被别处覆盖），做降级兜底
            if isinstance(x, MetaTensor):
                return x
            if torch.is_tensor(x):
                return MetaTensor(x.to(torch.float32), meta={})
            if isinstance(x, np.ndarray):
                return MetaTensor(torch.as_tensor(x, dtype=torch.float32), meta={})

            return x  # 兜底，不再强行解包

        # 基础预处理流水线 (应用于 Train/Val/Test)
        # 1. 加载 -> 标准化 -> 方向 -> 重采样
        base_pre_ops = [
            self.loader,  # 首先加载图像
            _to_meta,
            Orientation(axcodes="RAS", labels=None),
            Spacing(
                pixdim=(
                    1.0 / self.resize_scale,
                    1.0 / self.resize_scale,
                    1.0 / self.resize_scale,
                ),
                mode="bilinear",
            ),
            # 2. 强度归一化 (使用 0.05% 到 99.95% 百分位)
            ScaleIntensityRangePercentiles(
                lower=0.05,
                upper=99.95,
                b_min=self.min_int,
                b_max=1.0,
                clip=True,
                channel_wise=True,
            ),
            # 3. 前景裁剪
            CropForeground(select_fn=lambda x: x > self.min_int),
            # 4. 填充到目标尺寸 (如果小于目标尺寸)
            SpatialPad(spatial_size=target_size, value=self.min_int),
            # 5. 移除 Resize，改为后续的 Crop 操作以保持分辨率一致
            # Resize(spatial_size=target_size, mode="trilinear"),
        ]
        self.base_pre = Compose(base_pre_ops, map_items=False)

        # 使用 ResizeWithPadOrCrop 确保输出尺寸严格等于 target_size
        self.val_transform = ResizeWithPadOrCrop(
            spatial_size=target_size, mode="minimum"
        )

        # 训练数据基础增强 (几何 + 强度) - 对应预训练项目的 RandCropByPosNegLabeld 等
        base_augment_ops = [
            # 1. 随机裁剪 (对标 RandCropByPosNegLabeld，由于分类任务无 label 影像，使用 RandSpatialCrop)
            # SpatialPad 保证了图像至少有 target_size 大小，RandSpatialCrop 将其裁剪为固定尺寸
            RandSpatialCrop(
                roi_size=target_size, random_center=True, random_size=False
            ),
            # 2. 几何增强 (对标 RandFlipd, RandRotate90d)
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandFlip(prob=0.5, spatial_axis=2),
            RandRotate90(prob=0.1, max_k=3),
            # 3. 强度增强 (对标 RandScaleIntensityd, RandShiftIntensityd)
            RandScaleIntensity(factors=0.1, prob=1.0),
            RandShiftIntensity(offsets=0.1, prob=1.0),
        ]
        base_augment = Compose(base_augment_ops)

        self.train_transform = base_augment if is_train else None

        # nnUNet ZScore 归一化 (在已有预处理后、增强前应用)
        if self.use_nnunet_zscore:
            self.zscore_normalize = NormalizeIntensity(
                subtrahend=None, divisor=None, nonzero=False, channel_wise=True
            )
            print("[INFO] 启用 nnUNet ZScore 归一化 (归一化后再增强)")
        else:
            self.zscore_normalize = None

        # 加载样本
        # self.valid_samples = self._load_samples(csv_file_processed)

    def _build_measure_index(self) -> Dict[str, str]:
        """构建 measure csv 索引"""
        index: Dict[str, str] = {}
        duplicates: Dict[str, set] = defaultdict(set)

        if self.measure_root is not None and self.measure_root.exists():
            # 重要：排序以保证可复现（不同文件系统/并发扫描下 rglob 顺序可能变化）
            for p in sorted(self.measure_root.rglob("*.csv"), key=lambda x: str(x)):
                key = p.stem
                sp = str(p)

                if key in index and index[key] != sp:
                    duplicates[key].add(index[key])
                    duplicates[key].add(sp)

                # 仍保留“更短路径优先”的启发式，但会对重复 key 做告警
                if key not in index or len(sp) < len(index[key]):
                    index[key] = sp

        if duplicates:
            keys = sorted(duplicates.keys())[:10]
            print(f"[WARNING] measure csv 存在重复 stem（仅展示前 {len(keys)} 个）：")
            for k in keys:
                cands = sorted(duplicates[k])
                print(f"  - {k}: {cands[:3]}{' ...' if len(cands) > 3 else ''}")

        return index

    def _load_samples(self, csv_file: str, folder_map: Dict[str, str]) -> List[Dict]:
        """加载并验证样本"""
        df = self._read_and_clean_csv(csv_file)

        valid_samples = []
        dropped_samples = []
        drop_reason_counter: Dict[str, int] = defaultdict(int)

        for _, row in df.iterrows():
            sample = self._process_row(
                row, folder_map, dropped_samples, drop_reason_counter
            )
            if sample:
                valid_samples.append(sample)

        self._print_drop_summary(dropped_samples, drop_reason_counter)
        self._print_sample_distribution(valid_samples)

        return valid_samples

    def _read_and_clean_csv(self, csv_file: str) -> pd.DataFrame:
        """读取并清洗 CSV"""
        df = pd.read_csv(csv_file)
        required_cols = ["PTID", "ImageID", "VISCODE", "NEW_DX"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"csv_file_processed 缺少必要列: {missing}")

        for col in ["PTID", "ImageID", "VISCODE", "NEW_DX"]:
            df[col] = df[col].astype(str).str.strip()

        for col in ["PTID", "ImageID", "NEW_DX"]:
            df[col] = df[col].replace(
                {"": np.nan, "nan": np.nan, "None": np.nan, "none": np.nan}
            )

        if "Month" in df.columns:
            df["Month"] = pd.to_numeric(df["Month"], errors="coerce")

        df = df.dropna(subset=["PTID", "ImageID", "NEW_DX"])

        if "Month" in df.columns:
            df = df.sort_values(["PTID", "Month"], na_position="last")
        else:
            df = df.sort_values(["PTID"])

        return df.drop_duplicates(subset=["PTID", "ImageID"], keep="first")

    def _build_folder_map(self) -> Dict[str, str]:
        """构建文件夹映射"""
        folder_map: Dict[str, str] = {}
        duplicates: Dict[str, set] = defaultdict(set)

        # 排序以保证可复现（不同文件系统/并发扫描下 listdir 顺序可能变化）
        sub_dirs = (
            "CN_change",
            "CN_nochange",
            "P1_fs",
            "P2_fs",
            "P3_fs",
            "P4_fs",
            "P5_fs",
            "P6_fs",
        )
        for sub_dir in sub_dirs:
            sub_path = os.path.join(self.data_dir, sub_dir)
            if not os.path.isdir(sub_path):
                continue

            for folder_name in sorted(os.listdir(sub_path)):
                rel_folder = os.path.join(sub_dir, folder_name)
                key = extract_folder_key(folder_name)

                if key in folder_map and folder_map[key] != rel_folder:
                    duplicates[key].add(folder_map[key])
                    duplicates[key].add(rel_folder)

                if key not in folder_map or len(folder_name) < len(
                    os.path.basename(folder_map[key])
                ):
                    folder_map[key] = rel_folder

        if duplicates:
            keys = sorted(duplicates.keys())[:10]
            print(f"[WARNING] MRI 文件夹存在重复 key（仅展示前 {len(keys)} 个）：")
            for k in keys:
                cands = sorted(duplicates[k])
                print(f"  - {k}: {cands[:3]}{' ...' if len(cands) > 3 else ''}")

        return folder_map

    def _process_row(
        self,
        row: pd.Series,
        folder_map: Dict[str, str],
        dropped_samples: List,
        drop_reason_counter: Dict[str, int],
    ) -> Optional[Dict]:
        """处理单行数据"""
        ptid = str(row["PTID"]).strip()
        image_id_raw = str(row["ImageID"]).strip()
        image_id_normalized = normalize_image_id(image_id_raw) or image_id_raw

        # folder key 兼容：同时尝试 raw / normalized，避免 CSV ImageID 类型或格式变化导致 miss
        rel_folder = None
        for fk in (f"{ptid}_{image_id_normalized}", f"{ptid}_{image_id_raw}"):
            rel_folder = folder_map.get(fk)
            if rel_folder is not None:
                break

        sample_id = f"{ptid}_{image_id_normalized}"

        if rel_folder is None:
            dropped_samples.append((sample_id, "missing_folder"))
            drop_reason_counter["missing_folder"] += 1
            return None

        dx_group = normalize_dx(str(row["NEW_DX"]))
        if dx_group is None:
            dropped_samples.append((sample_id, "unknown_dx"))
            drop_reason_counter["unknown_dx"] += 1
            return None

        vbm_path = os.path.join(self.data_dir, rel_folder, "mri", self.VBM_NAME)
        if not os.path.isfile(vbm_path):
            dropped_samples.append((sample_id, "missing_vbm"))
            drop_reason_counter["missing_vbm"] += 1
            return None

        # 获取年龄
        age_value = float("nan")
        if self.age_csv is not None:
            age_key = (ptid, image_id_normalized)
            age_value = self.age_map.get(age_key, float("nan"))

        # 获取测量数据路径
        measure_path = self._find_measure_path(rel_folder, row, image_id_normalized)

        # 完整性检查
        drop_reasons = self._check_completeness(age_value, measure_path, sample_id)

        if drop_reasons:
            reason = " ; ".join(drop_reasons)
            if self._drop_log_count < self.drop_log_limit:
                print(f"[DROP] {sample_id} | {reason}")
            self._drop_log_count += 1
            dropped_samples.append((sample_id, reason))
            for r in drop_reasons:
                code = r.split()[0].split("=", 1)[0]
                drop_reason_counter[code] += 1
            return None

        sample = {
            "FolderName": rel_folder,
            "DX_group": dx_group,
            "ImageID": row["ImageID"],
            "PTID": row["PTID"],
            "VISCODE": row["VISCODE"],
            "Age": age_value,
            "MeasurePath": measure_path,
        }
        if "Month" in row.index:
            sample["Month"] = row.get("Month", np.nan)

        return sample

    def _find_measure_path(
        self, rel_folder: str, row: pd.Series, image_id_normalized: str
    ) -> Optional[str]:
        """查找测量数据路径"""
        if self.measure_root is None:
            return None

        folder_name = os.path.basename(rel_folder)
        candidates = [folder_name]

        if "_real" in folder_name:
            candidates.append(folder_name.replace("_real", "").strip())

        folder_key_std = extract_folder_key(folder_name)
        if folder_key_std not in candidates:
            candidates.append(folder_key_std)

        folder_key_ptid = f"{row['PTID']}_{image_id_normalized}"
        if folder_key_ptid not in candidates:
            candidates.append(folder_key_ptid)

        for cand in candidates:
            possible_csv = self.measure_root / f"{cand}.csv"
            if possible_csv.exists():
                return str(possible_csv)
            hit = self._measure_index.get(cand)
            if hit is not None:
                return hit

        return None

    def _check_completeness(
        self, age_value: float, measure_path: Optional[str], sample_id: str
    ) -> List[str]:
        """检查数据完整性"""
        drop_reasons = []

        if self.age_csv is not None and not np.isfinite(age_value):
            drop_reasons.append("missing_age")

        if self.measure_root is not None:
            if measure_path is None:
                drop_reasons.append("missing_measure_file")
            else:
                try:
                    assert self.region_order is not None
                    cached_meta = self._measure_meta_cache.get(measure_path)
                    if cached_meta is None:
                        df = pd.read_csv(measure_path)
                        hemi_col = "hemi"
                        region_col = "StructName"
                        keep_cols = ("ThickAvg", "GrayVol", "SurfArea")
                        for c in [hemi_col, region_col, *keep_cols]:
                            if c not in df.columns:
                                raise ValueError(
                                    f"Measure file missing column '{c}': {measure_path}"
                                )

                        df[hemi_col] = df[hemi_col].apply(normalize_hemi)
                        df[region_col] = df[region_col].astype(str)

                        present = set(
                            zip(df[hemi_col].tolist(), df[region_col].tolist())
                        )
                        missing_rois = [
                            k for k in self.region_order if k not in present
                        ]
                        missing_ratio = float(len(missing_rois)) / float(
                            max(len(self.region_order), 1)
                        )

                        table: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
                        for _, row in df.iterrows():
                            key = (row[hemi_col], row[region_col])
                            vals = tuple(float(row[c]) for c in keep_cols)
                            table[key] = vals

                        out = np.full(
                            (len(self.region_order), len(keep_cols)),
                            0.0,
                            dtype=np.float32,
                        )
                        for i, key in enumerate(self.region_order):
                            if key in table:
                                out[i, :] = np.asarray(table[key], dtype=np.float32)

                        has_nonfinite = not np.isfinite(out).all()
                        measures = torch.from_numpy(out).to(dtype=torch.float32)
                        measures = torch.nan_to_num(
                            measures, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        self._measures_cache[measure_path] = measures
                        cached_meta = (missing_rois, missing_ratio, has_nonfinite)
                        self._measure_meta_cache[measure_path] = cached_meta

                    missing_rois, missing_ratio, has_nonfinite = cached_meta
                    if len(missing_rois) > 0:
                        preview = ", ".join([f"{h}:{r}" for h, r in missing_rois[:8]])
                        more = (
                            ""
                            if len(missing_rois) <= 8
                            else f" ...(+{len(missing_rois) - 8})"
                        )
                        drop_reasons.append(
                            f"missing_rois={len(missing_rois)} ({missing_ratio:.1%}) [{preview}{more}]"
                        )
                    elif has_nonfinite:
                        drop_reasons.append("measure_contains_NaN_or_inf")
                except Exception as e:
                    drop_reasons.append(
                        f"measure_check_error={type(e).__name__}: {str(e)}"
                    )

        return drop_reasons

    def _print_drop_summary(
        self, dropped_samples: List, drop_reason_counter: Dict[str, int]
    ):
        """打印丢弃样本汇总"""
        if dropped_samples:
            print(f"[INFO] 丢弃样本数（采集/完整性综合）: {len(dropped_samples)}")
            if drop_reason_counter:
                print("[INFO] 丢弃原因计数:")
                for k, v in sorted(
                    drop_reason_counter.items(), key=lambda x: (-x[1], x[0])
                ):
                    print(f"  - {k}: {v}")
        else:
            print("[INFO] 采集/完整性检查：未丢弃任何样本。")

    def _print_sample_distribution(self, valid_samples: List[Dict]):
        """打印样本分布"""
        dxs = [s["DX_group"] for s in valid_samples]
        if dxs:
            print(
                f"[INFO] 有效样本 {len(valid_samples)}，分布：{pd.Series(dxs).value_counts().to_dict()}"
            )
        else:
            print("[WARNING] 没有找到有效样本！")

    def set_train_idx(self, train_idx):
        """设置训练索引（用于启用数据增强）"""
        self.train_idx_set = set(train_idx)

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        MRIDataset._debug_getitem_calls += 1

        row = self.valid_samples[idx]

        # Define path early to ensure it's available for error messages
        path = os.path.join(self.data_dir, row["FolderName"], "mri", self.VBM_NAME)

        # 应用基础预处理流水线 (包含加载、重采样、归一化、裁剪、填充、调整尺寸)
        try:
            img = self.base_pre(path)  # 返回 numpy array, shape: (C, D, H, W)
        except Exception as e:
            sample_id = (
                f"{row.get('PTID', '?')}_{normalize_image_id(str(row.get('ImageID', '?')))}"
            )
            raise RuntimeError(
                f"Failed to load/preprocess NIfTI: sample={sample_id}, path={path}"
            ) from e

        # 检查数值有效性（兼容 ndarray / Tensor / MetaTensor）
        if torch.is_tensor(img):
            ok = torch.isfinite(img).all().item()
        else:
            ok = np.isfinite(img).all()

        if not ok:
            sample_id = f"{row.get('PTID', '?')}_{normalize_image_id(str(row.get('ImageID', '?')))}"
            raise ValueError(
                f"Non-finite values in image: sample={sample_id}, path={path}"
            )

        # 转换为 Tensor（避免重复拷贝）
        if not torch.is_tensor(img):
            img = torch.as_tensor(img, dtype=torch.float32)
        else:
            img = img.to(torch.float32)

        # nnUNet ZScore 归一化：归一化后再增强
        # 在数据增强之前应用，确保输入分布与预训练一致
        if self.zscore_normalize is not None:
            img = self.zscore_normalize(img)

        if (idx in self.train_idx_set) and (self.train_transform is not None):
            img = self.train_transform(img)  # [C, D, H, W]
        else:
            img = self.val_transform(img)

        # 标签
        label = self.LABEL_MAP[row["DX_group"]]
        sample_id = f"{row['PTID']}_{normalize_image_id(str(row['ImageID']))}"

        # 年龄
        age = torch.tensor(row.get("Age", float("nan")), dtype=torch.float32)

        # 测量数据
        measures = self._get_measures(row)

        return img, torch.tensor(label, dtype=torch.long), sample_id, age, measures

    def _get_measures(self, row: Dict) -> torch.Tensor:
        measure_path = row.get("MeasurePath")
        if not measure_path or self.region_order is None:
            k = len(self.region_order) if self.region_order else 0
            return torch.zeros((k, 3), dtype=torch.float32)

        # cache hit
        cached = self._measures_cache.get(measure_path)
        if cached is not None:
            return cached

        # Enforce cache size limit (simple LRU-like eviction)
        if len(self._measures_cache) >= self._measures_cache_max_size:
            # Remove oldest entries (first 10% of cache)
            keys_to_remove = list(self._measures_cache.keys())[
                : self._measures_cache_max_size // 10
            ]
            for k in keys_to_remove:
                del self._measures_cache[k]

        measures = vectorize_measures(Path(measure_path), self.region_order)
        measures = torch.from_numpy(measures).to(dtype=torch.float32)
        measures = torch.nan_to_num(measures, nan=0.0, posinf=0.0, neginf=0.0)

        self._measures_cache[measure_path] = measures
        return measures
