"""
DICOM 医学图像读取和处理模块

支持冠脉 CT 影像的自动读取、处理、可视化和格式转换。
"""

import os
import glob
from pathlib import Path
from typing import Union, List, Optional, Tuple

import numpy as np
import pydicom
from pydicom.dataset import Dataset

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class DcmImage:
    """DICOM 图像类

    封装单个 DICOM 文件的读取和处理功能。
    """

    def __init__(self, filepath: Union[str, Path]):
        """初始化 DICOM 图像

        Args:
            filepath: DICOM 文件路径
        """
        self.filepath = Path(filepath)
        self._dataset: Optional[Dataset] = None
        self._pixel_array: Optional[np.ndarray] = None

    @property
    def dataset(self) -> Dataset:
        """获取 DICOM Dataset 对象"""
        if self._dataset is None:
            self._dataset = pydicom.dcmread(self.filepath)
        return self._dataset

    @property
    def pixel_array(self) -> np.ndarray:
        """获取像素数组"""
        if self._pixel_array is None:
            self._pixel_array = self.dataset.pixel_array
        return self._pixel_array

    @property
    def image(self) -> np.ndarray:
        """获取图像数据（同 pixel_array）"""
        return self.pixel_array

    def read(self) -> Dataset:
        """读取 DICOM 文件并返回 Dataset

        Returns:
            pydicom Dataset 对象
        """
        return self.dataset

    def to_ndarray(self, apply_rescale: bool = True) -> np.ndarray:
        """转换为 numpy ndarray

        Args:
            apply_rescale: 是否应用 RescaleSlope 和 RescaleIntercept

        Returns:
            numpy ndarray
        """
        arr = self.pixel_array.astype(np.float32)

        if apply_rescale:
            # 应用 HU 值转换
            slope = getattr(self.dataset, 'RescaleSlope', 1.0)
            intercept = getattr(self.dataset, 'RescaleIntercept', 0.0)
            arr = arr * slope + intercept

        return arr

    def to_tensor(self, apply_rescale: bool = True) -> 'torch.Tensor':
        """转换为 torch Tensor

        Args:
            apply_rescale: 是否应用 RescaleSlope 和 RescaleIntercept

        Returns:
            torch Tensor
        """
        if not HAS_TORCH:
            raise ImportError("torch not installed. Please install with: pip install torch")

        arr = self.to_ndarray(apply_rescale=apply_rescale)
        return torch.from_numpy(arr)

    def show(self, cmap: str = 'gray', title: Optional[str] = None,
             figsize: Tuple[int, int] = (10, 10), **kwargs):
        """显示图像

        Args:
            cmap: 颜色映射，默认 'gray'
            title: 图像标题
            figsize: 图像大小
            **kwargs: 传递给 plt.imshow 的其他参数
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib not installed. Please install with: pip install matplotlib")

        arr = self.pixel_array

        plt.figure(figsize=figsize)
        plt.imshow(arr, cmap=cmap, **kwargs)
        plt.axis('off')

        if title is None:
            title = f"{self.filepath.name}"
        plt.title(title)

        plt.tight_layout()
        plt.show()

    def get_metadata(self) -> dict:
        """获取 DICOM 元数据

        Returns:
            包含关键元数据的字典
        """
        ds = self.dataset
        return {
            'PatientID': getattr(ds, 'PatientID', None),
            'PatientName': str(getattr(ds, 'PatientName', '')),
            'StudyDate': getattr(ds, 'StudyDate', None),
            'Modality': getattr(ds, 'Modality', None),
            'Manufacturer': getattr(ds, 'Manufacturer', None),
            'ImagePositionPatient': getattr(ds, 'ImagePositionPatient', None),
            'ImageOrientationPatient': getattr(ds, 'ImageOrientationPatient', None),
            'PixelSpacing': getattr(ds, 'PixelSpacing', None),
            'SliceThickness': getattr(ds, 'SliceThickness', None),
            'KVP': getattr(ds, 'KVP', None),
            'Rows': getattr(ds, 'Rows', None),
            'Columns': getattr(ds, 'Columns', None),
            'BitsAllocated': getattr(ds, 'BitsAllocated', None),
            'BitsStored': getattr(ds, 'BitsStored', None),
            'WindowCenter': getattr(ds, 'WindowCenter', None),
            'WindowWidth': getattr(ds, 'WindowWidth', None),
        }

    def __repr__(self) -> str:
        return f"DcmImage('{self.filepath}')"

    def __getitem__(self, idx):
        """支持索引操作"""
        return self.pixel_array[idx]

    @property
    def shape(self) -> Tuple:
        """获取图像形状"""
        return self.pixel_array.shape

    @property
    def dtype(self) -> np.dtype:
        """获取数据类型"""
        return self.pixel_array.dtype


class DcmSeries:
    """DICOM 序列类

    用于加载和处理同一序列的多个 DICOM 文件（如 CT 的多帧图像）。
    """

    def __init__(self, directory: Union[str, Path], pattern: str = "*.dcm"):
        """初始化 DICOM 序列

        Args:
            directory: 包含 DICOM 文件的目录
            pattern: 文件匹配模式
        """
        self.directory = Path(directory)
        self.pattern = pattern
        self._images: List[DcmImage] = []
        self._datasets: List[Dataset] = []

    def load(self, sort_by_position: bool = True) -> 'DcmSeries':
        """加载 DICOM 文件

        Args:
            sort_by_position: 是否按 ImagePositionPatient 排序

        Returns:
            self
        """
        files = list(self.directory.glob(self.pattern))

        self._images = [DcmImage(f) for f in files]
        self._datasets = [img.dataset for img in self._images]

        if sort_by_position and self._datasets:
            # 尝试按 Z 轴位置排序
            try:
                positions = []
                for i, ds in enumerate(self._datasets):
                    pos = getattr(ds, 'ImagePositionPatient', None)
                    if pos:
                        positions.append((float(pos[2]), i))
                    else:
                        # 如果没有位置信息，使用 InstanceNumber
                        num = getattr(ds, 'InstanceNumber', i)
                        positions.append((float(num), i))

                positions.sort(reverse=True)  # 降序：从脚到头（解剖学方向）
                sorted_indices = [idx for _, idx in positions]
                self._images = [self._images[i] for i in sorted_indices]
                self._datasets = [self._datasets[i] for i in sorted_indices]
            except (AttributeError, ValueError, TypeError):
                pass  # 如果排序失败，保持原有顺序

        return self

    @property
    def images(self) -> List[DcmImage]:
        """获取所有 DcmImage 对象"""
        return self._images

    def to_volume(self, apply_rescale: bool = True) -> np.ndarray:
        """将序列转换为 3D numpy 数组

        Args:
            apply_rescale: 是否应用 HU 值转换

        Returns:
            3D numpy ndarray (z, y, x)
        """
        arrays = [img.to_ndarray(apply_rescale) for img in self._images]
        return np.stack(arrays, axis=0)

    def to_volume_tensor(self, apply_rescale: bool = True) -> 'torch.Tensor':
        """将序列转换为 3D torch Tensor

        Args:
            apply_rescale: 是否应用 HU 值转换

        Returns:
            3D torch Tensor (z, y, x)
        """
        if not HAS_TORCH:
            raise ImportError("torch not installed.")

        volume = self.to_volume(apply_rescale)
        return torch.from_numpy(volume)

    def get_slice(self, index: int) -> DcmImage:
        """获取指定索引的切片

        Args:
            index: 切片索引

        Returns:
            DcmImage 对象
        """
        return self._images[index]

    def show_slice(self, index: int, **kwargs):
        """显示指定索引的切片

        Args:
            index: 切片索引
            **kwargs: 传递给 DcmImage.show 的参数
        """
        self._images[index].show(**kwargs)

    def show_mpr(self, axis: str = 'sagittal', **kwargs):
        """显示多平面重建视图

        Args:
            axis: 视图方向 ('sagittal', 'coronal', 'axial')
            **kwargs: 传递给 plt.imshow 的参数
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib not installed.")

        volume = self.to_volume()

        plt.figure(figsize=(15, 5))

        if axis == 'sagittal':
            # 矢状面 (x-z 平面)
            slice_idx = volume.shape[2] // 2
            plt.subplot(1, 3, 1)
            plt.imshow(volume[:, :, slice_idx].T, cmap='gray', **kwargs)
            plt.title(f'Sagittal (x={slice_idx})')
            plt.axis('off')

            # 冠状面 (y-z 平面)
            slice_idx = volume.shape[1] // 2
            plt.subplot(1, 3, 2)
            plt.imshow(volume[:, slice_idx, :].T, cmap='gray', **kwargs)
            plt.title(f'Coronal (y={slice_idx})')
            plt.axis('off')

            # 轴状面 (x-y 平面)
            slice_idx = volume.shape[0] // 2
            plt.subplot(1, 3, 3)
            plt.imshow(volume[slice_idx, :, :], cmap='gray', **kwargs)
            plt.title(f'Axial (z={slice_idx})')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx) -> DcmImage:
        return self._images[idx]

    def __iter__(self):
        return iter(self._images)

    def __repr__(self) -> str:
        return f"DcmSeries('{self.directory}', {len(self._images)} images)"


def read_dcm(filepath: Union[str, Path]) -> DcmImage:
    """便捷函数：读取单个 DICOM 文件

    Args:
        filepath: DICOM 文件路径

    Returns:
        DcmImage 对象
    """
    return DcmImage(filepath)


def read_series(directory: Union[str, Path], pattern: str = "*.dcm",
                sort: bool = True) -> DcmSeries:
    """便捷函数：读取 DICOM 序列

    Args:
        directory: 包含 DICOM 文件的目录
        pattern: 文件匹配模式
        sort: 是否自动排序

    Returns:
        DcmSeries 对象
    """
    return DcmSeries(directory, pattern).load(sort_by_position=sort)


def load_dicom_files(directory: Union[str, Path]) -> Tuple[List[np.ndarray], List[Dataset]]:
    """加载目录下所有 DICOM 文件

    Args:
        directory: 包含 DICOM 文件的目录

    Returns:
        (images, datasets) 元组，分别为图像数组列表和 Dataset 列表
    """
    series = read_series(directory)
    images = [img.to_ndarray() for img in series.images]
    datasets = [img.dataset for img in series.images]
    return images, datasets


def dicom_to_ndarray(directory: Union[str, Path]) -> np.ndarray:
    """将 DICOM 序列转换为单个 3D numpy 数组

    Args:
        directory: 包含 DICOM 文件的目录

    Returns:
        3D numpy ndarray
    """
    return read_series(directory).to_volume()


def dicom_to_tensor(directory: Union[str, Path]) -> 'torch.Tensor':
    """将 DICOM 序列转换为单个 3D torch Tensor

    Args:
        directory: 包含 DICOM 文件的目录

    Returns:
        3D torch Tensor
    """
    return read_series(directory).to_volume_tensor()


def visualize_dcm(filepath: Union[str, Path], **kwargs):
    """便捷函数：显示单个 DICOM 图像

    Args:
        filepath: DICOM 文件路径
        **kwargs: 传递给 DcmImage.show 的参数
    """
    DcmImage(filepath).show(**kwargs)


# 窗宽窗位预设（用于 CT 显示）
WINDOW_PRESETS = {
    'soft_tissue': {'width': 350, 'center': 50},
    'lung': {'width': 1500, 'center': -600},
    'bone': {'width': 2500, 'center': 500},
    'brain': {'width': 80, 'center': 40},
    'cardiac': {'width': 400, 'center': 50},  # 冠脉 CT 适用
}


def apply_window_level(image: np.ndarray, width: int, center: int) -> np.ndarray:
    """应用窗宽窗位

    Args:
        image: 输入图像（HU 值）
        width: 窗宽
        center: 窗位

    Returns:
        处理后的图像（0-1 范围）
    """
    lower = center - width / 2
    upper = center + width / 2

    windowed = np.clip(image, lower, upper)
    windowed = (windowed - lower) / (upper - lower)

    return windowed


def apply_window_preset(image: np.ndarray, preset: str = 'cardiac') -> np.ndarray:
    """应用预设窗宽窗位

    Args:
        image: 输入图像（HU 值）
        preset: 预设名称 ('soft_tissue', 'lung', 'bone', 'brain', 'cardiac')

    Returns:
        处理后的图像
    """
    if preset not in WINDOW_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(WINDOW_PRESETS.keys())}")

    params = WINDOW_PRESETS[preset]
    return apply_window_level(image, params['width'], params['center'])
