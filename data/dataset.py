"""
视频分类数据集模块

支持 DICOM 序列的视频分类数据集，适用于乳腺和甲状腺超声视频分类。
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dcm import DcmImage, read_series


# 属性标签映射（根据实际数据集修改）
ATTR_LIB = {
    "病理": ["良性", "恶性"],  # 甲状腺
}

ATTR_LIB_BREAST = {
    "Pathology": ["正常", "良性", "可疑", "恶性", "其他"],  # 乳腺 BI-RADS 分类
}


class VideoTransform:
    """视频数据增强变换"""

    def __init__(self, cfg, is_train: bool = True):
        """
        初始化变换

        Args:
            cfg: 配置对象
            is_train: 是否为训练模式
        """
        self.is_train = is_train
        self.input_size = cfg.DATA.INPUT_SIZE
        self.mean = cfg.DATA.NORMALIZE_MEAN
        self.std = cfg.DATA.NORMALIZE_STD

        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(self.input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        return self.transform(image)


class DcmVideoDataset(Dataset):
    """
    DICOM 视频数据集

    从目录加载 DICOM 序列，支持多帧采样和数据增强。
    """

    def __init__(
        self,
        data_dir: str,
        cfg,
        is_train: bool = True,
        num_frames: int = 16,
        sample_mode: str = "uniform",
    ):
        """
        初始化数据集

        Args:
            data_dir: 数据目录路径
            cfg: 配置对象
            is_train: 是否为训练模式
            num_frames: 每个视频采样的帧数
            sample_mode: 采样模式 ("uniform" | "random" | "continuous")
        """
        self.data_dir = Path(data_dir)
        self.cfg = cfg
        self.is_train = is_train
        self.num_frames = num_frames
        self.sample_mode = sample_mode

        # 获取器官类型
        self.organ = cfg.ORGAN
        if self.organ == "thyroid":
            self.attr_list = cfg.ATTR_LIST
            self.attr_lib = ATTR_LIB
        elif self.organ == "breast":
            self.attr_list = cfg.ATTR_LIST
            self.attr_lib = ATTR_LIB_BREAST
        else:
            raise ValueError(f"Unsupported organ: {self.organ}")

        # 数据变换
        self.transform = VideoTransform(cfg, is_train)

        # 扫描数据目录
        self.samples = self._scan_directory()

    def _scan_directory(self) -> List[Dict]:
        """
        扫描数据目录，构建样本列表

        Returns:
            样本列表，每个样本包含文件路径和标签信息
        """
        samples = []

        # 遍历子目录（每个子目录为一个视频序列）
        for subdir in self.data_dir.iterdir():
            if not subdir.is_dir():
                continue

            # 查找 DICOM 文件
            dcm_files = list(subdir.glob("*.dcm"))
            if not dcm_files:
                continue

            # 尝试从文件名或配置文件读取标签
            # 这里假设标签信息存储在子目录名或单独的配置文件中
            sample = {
                "video_path": str(subdir),
                "dcm_files": [str(f) for f in dcm_files],
                "file_name": subdir.name,
                "attr_label": self._load_labels(subdir),
                "dataset_type": self._detect_dataset_type(subdir),
            }
            samples.append(sample)

        return samples

    def _load_labels(self, subdir: Path) -> Dict[str, int]:
        """
        加载样本标签

        Args:
            subdir: 样本子目录

        Returns:
            标签字典 {attr_name: label_id}
        """
        # 默认实现：尝试从目录名解析标签
        # 实际使用时应根据数据格式修改

        # 检查是否有标签文件
        label_file = subdir / "labels.json"
        if label_file.exists():
            import json
            with open(label_file, "r") as f:
                return json.load(f)

        # 检查是否有 txt 标签文件
        label_file = subdir / "label.txt"
        if label_file.exists():
            with open(label_file, "r") as f:
                content = f.read().strip()
                # 假设格式为 "Pathology: 2"
                if ":" in content:
                    key, value = content.split(":")
                    return {key.strip(): int(value.strip())}

        # 默认返回 0（需要实际标签时修改此逻辑）
        if self.organ == "breast":
            return {"Pathology": 0}
        else:
            return {"病理": 0}

    def _detect_dataset_type(self, subdir: Path) -> str:
        """检测数据集类型（image 或 video）"""
        subdir_name = subdir.name.lower()
        if "video" in subdir_name:
            return "dataset_video"
        elif "image" in subdir_name:
            return "dataset_image"
        else:
            # 根据帧数判断
            dcm_files = list(subdir.glob("*.dcm"))
            if len(dcm_files) > 10:
                return "dataset_video"
            return "dataset_image"

    def _sample_frames(self, num_total_frames: int) -> List[int]:
        """
        采样帧索引

        Args:
            num_total_frames: 总帧数

        Returns:
            采样的帧索引列表
        """
        if num_total_frames <= self.num_frames:
            # 帧数不足，重复采样
            indices = list(range(num_total_frames))
            indices += random.choices(indices, k=self.num_frames - num_total_frames)
            return indices

        if self.sample_mode == "uniform":
            # 均匀采样
            step = num_total_frames / self.num_frames
            indices = [int(i * step) for i in range(self.num_frames)]
        elif self.sample_mode == "random":
            # 随机采样
            indices = random.sample(range(num_total_frames), self.num_frames)
        elif self.sample_mode == "continuous":
            # 连续采样
            start = random.randint(0, num_total_frames - self.num_frames)
            indices = list(range(start, start + self.num_frames))
        else:
            indices = list(range(self.num_frames))

        return indices

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> List[Dict]:
        """
        获取一个视频样本

        Args:
            idx: 样本索引

        Returns:
            帧列表，每帧为 {"image": Tensor, "attr_label": Dict, "file_name": str}
        """
        sample = self.samples[idx]

        # 加载 DICOM 序列
        try:
            series = read_series(sample["video_path"])
            num_frames = len(series)

            # 采样帧索引
            frame_indices = self._sample_frames(num_frames)

            # 构建帧列表
            frames = []
            for frame_idx in frame_indices:
                frame_img = series.get_slice(frame_idx)
                hu_array = frame_img.to_ndarray(apply_rescale=True)

                # 归一化 HU 值到 0-255 范围用于显示和变换
                hu_min, hu_max = -1000, 1000
                normalized = np.clip((hu_array - hu_min) / (hu_max - hu_min) * 255, 0, 255).astype(np.uint8)
                normalized = np.stack([normalized] * 3, axis=-1)  # 转为 3 通道

                # 应用变换
                image_tensor = self.transform(normalized)

                frames.append({
                    "image": image_tensor,
                    "attr_label": sample["attr_label"],
                    "file_name": f"{sample['file_name']}_frame{frame_idx}",
                })

            return frames

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # 返回空样本或跳过
            return self.__getitem__((idx + 1) % len(self))


class ImageDataset(Dataset):
    """
    DICOM 单图像数据集

    用于处理单帧 DICOM 图像分类任务。
    """

    def __init__(
        self,
        data_dir: str,
        cfg,
        is_train: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.cfg = cfg
        self.is_train = is_train
        self.transform = VideoTransform(cfg, is_train)

        self.organ = cfg.ORGAN
        if self.organ == "breast":
            self.attr_lib = ATTR_LIB_BREAST
        else:
            self.attr_lib = ATTR_LIB

        self.samples = self._scan_directory()

    def _scan_directory(self) -> List[Dict]:
        """扫描数据目录"""
        samples = []

        # 遍历所有 DICOM 文件
        for dcm_file in self.data_dir.rglob("*.dcm"):
            sample = {
                "file_path": str(dcm_file),
                "file_name": dcm_file.stem,
                "attr_label": self._load_labels(dcm_file),
            }
            samples.append(sample)

        return samples

    def _load_labels(self, dcm_file: Path) -> Dict[str, int]:
        """加载标签（实现同 VideoDataset）"""
        # 简单实现：从目录名获取
        parent = dcm_file.parent.name
        try:
            label = int(parent)
            return {"Pathology": label}
        except ValueError:
            return {"Pathology": 0}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        sample = self.samples[idx]

        try:
            # 加载 DICOM 图像
            img = DcmImage(sample["file_path"])
            hu_array = img.to_ndarray(apply_rescale=True)

            # 归一化
            hu_min, hu_max = -1000, 1000
            normalized = np.clip((hu_array - hu_min) / (hu_max - hu_min) * 255, 0, 255).astype(np.uint8)
            normalized = np.stack([normalized] * 3, axis=-1)

            # 应用变换
            image_tensor = self.transform(normalized)

            return {
                "image": image_tensor,
                "attr_label": sample["attr_label"],
                "file_name": sample["file_name"],
            }

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))


def build_dataloader(
    cfg,
    is_train: bool = True,
    distributed: bool = False,
) -> DataLoader:
    """
    构建数据加载器

    Args:
        cfg: 配置对象
        is_train: 是否为训练模式
        distributed: 是否分布式训练

    Returns:
        DataLoader 对象
    """
    if is_train:
        data_dir = cfg.DATA.TRAIN_DIR
        batch_size = cfg.TRAIN.BATCH_SIZE
        num_frames = cfg.TRAIN.NUM_SAMPLES_PER_VIDEO
    else:
        data_dir = cfg.DATA.VAL_DIR
        batch_size = cfg.TEST.BATCH_SIZE
        num_frames = cfg.TRAIN.NUM_SAMPLES_PER_VIDEO

    # 判断是否为视频数据集
    is_video = cfg.MODEL.MODE == "HYBRID"

    if is_video:
        dataset = DcmVideoDataset(data_dir, cfg, is_train, num_frames)
    else:
        dataset = ImageDataset(data_dir, cfg, is_train)

    # 构建 DataLoader
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            collate_fn=video_collate if is_video else None,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            collate_fn=video_collate if is_video else None,
        )

    return loader


def video_collate(batch: List[List[Dict]]) -> List[List[Dict]]:
    """
    视频数据的 collate 函数

    Args:
        batch: 批次数据

    Returns:
        打包后的批次
    """
    return batch
