"""
ImageList 工具类

用于处理批次图像的打包和填充。
"""

from typing import List, Tuple
import torch
from torch import nn


class ImageList:
    """
    批次图像容器，支持不同尺寸图像的打包。

    类似 detectron2 的 ImageList，用于将多个可能不同尺寸的图像
    打包成一个 batch，并记录每个图像的原始尺寸。
    """

    def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        """
        初始化 ImageList

        Args:
            tensor: 打包后的张量 (B, C, H, W) 或 (B*T, C, H, W)
            image_sizes: 每个图像的原始尺寸 [(H, W), ...]
        """
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self):
        return len(self.image_sizes)

    @property
    def device(self):
        return self.tensor.device


def imagelist_from_tensors(
    tensors: List[torch.Tensor],
    batch_size: int,
    size_divisibility: int = 32,
    pad_value: float = 0.0
) -> ImageList:
    """
    从张量列表创建 ImageList

    Args:
        tensors: 张量列表，每个张量为 (C, H, W)
        batch_size: 批次大小
        size_divisibility: 尺寸对齐因子（通常为 32，便于骨干网络下采样）
        pad_value: 填充值

    Returns:
        ImageList 对象
    """
    if not tensors:
        return ImageList(
            torch.zeros((0, 0, 0, 0), dtype=torch.float32),
            []
        )

    # 获取设备
    device = tensors[0].device

    # 获取最大尺寸
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)

    # 尺寸对齐
    if size_divisibility > 1:
        max_h = ((max_h + size_divisibility - 1) // size_divisibility) * size_divisibility
        max_w = ((max_w + size_divisibility - 1) // size_divisibility) * size_divisibility

    # 创建打包张量
    batched = torch.zeros(
        (len(tensors), tensors[0].shape[0], max_h, max_w),
        dtype=tensors[0].dtype,
        device=device
    )
    batched = batched.fill_(pad_value)

    # 复制图像数据
    image_sizes = []
    for i, t in enumerate(tensors):
        h, w = t.shape[1], t.shape[2]
        batched[i, :, :h, :w] = t
        image_sizes.append((h, w))

    return ImageList(batched, image_sizes)
