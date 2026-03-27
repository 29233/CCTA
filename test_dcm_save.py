"""
DICOM 图像处理功能测试与可视化示例（保存版）

演示 dcm.py 模块的各项功能，并将可视化结果保存为 PNG 文件。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非 GUI 后端
import matplotlib.pyplot as plt
from pathlib import Path

from dcm import (
    DcmImage,
    DcmSeries,
    read_dcm,
    read_series,
    apply_window_level,
    apply_window_preset,
    WINDOW_PRESETS,
)

# 输出目录
OUTPUT_DIR = Path(r"E:\pycharm23\Projs\DcmDataset\output")
OUTPUT_DIR.mkdir(exist_ok=True)


def visualize_single_image(img: DcmImage, save_path: Path):
    """可视化单个图像并保存"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Single DICOM Image Visualization", fontsize=14)

    # 1. 原始像素值
    axes[0, 0].imshow(img.pixel_array, cmap='gray')
    axes[0, 0].set_title("Raw Pixel Array")
    axes[0, 0].axis('off')

    # 2. HU 值（应用 Rescale）
    hu_arr = img.to_ndarray(apply_rescale=True)
    im = axes[0, 1].imshow(hu_arr, cmap='gray')
    axes[0, 1].set_title("HU Values (with Rescale)")
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3. 直方图
    axes[0, 2].hist(hu_arr.flatten(), bins=200, color='steelblue', alpha=0.7)
    axes[0, 2].set_title("HU Value Histogram")
    axes[0, 2].set_xlabel("HU Value")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].axvline(hu_arr.mean(), color='red', linestyle='--', label=f'Mean: {hu_arr.mean():.1f}')
    axes[0, 2].legend()

    # 4. 软组织窗
    soft_tissue = apply_window_preset(hu_arr, 'soft_tissue')
    axes[1, 0].imshow(soft_tissue, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title("Soft Tissue Window (W:350, C:50)")
    axes[1, 0].axis('off')

    # 5. 肺窗
    lung = apply_window_preset(hu_arr, 'lung')
    axes[1, 1].imshow(lung, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title("Lung Window (W:1500, C:-600)")
    axes[1, 1].axis('off')

    # 6. 骨窗
    bone = apply_window_preset(hu_arr, 'bone')
    axes[1, 2].imshow(bone, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title("Bone Window (W:2500, C:500)")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存：{save_path}")


def visualize_series(series: DcmSeries, save_path: Path):
    """可视化 DICOM 序列并保存"""
    n_slices = len(series)
    indices = [0, n_slices // 4, n_slices // 2, 3 * n_slices // 4, n_slices - 1]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f"DICOM Series Overview (Total: {n_slices} slices)", fontsize=14)

    for i, idx in enumerate(indices):
        img = series.get_slice(idx)
        hu_arr = img.to_ndarray()
        cardiac = apply_window_preset(hu_arr, 'cardiac')

        axes[i].imshow(cardiac, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f"Slice #{idx}\n{img.filepath.name}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存：{save_path}")


def visualize_mpr(series: DcmSeries, save_path: Path):
    """多平面重建可视化并保存"""
    volume = series.to_volume()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Multi-Planar Reconstruction (MPR)", fontsize=14)

    sag_idx = volume.shape[2] // 2
    sag = volume[:, :, sag_idx].T
    sag_wl = apply_window_level(sag, 400, 50)
    axes[0].imshow(sag_wl, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Sagittal (x={sag_idx})")
    axes[0].axis('off')

    cor_idx = volume.shape[1] // 2
    cor = volume[:, cor_idx, :].T
    cor_wl = apply_window_level(cor, 400, 50)
    axes[1].imshow(cor_wl, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Coronal (y={cor_idx})")
    axes[1].axis('off')

    ax_idx = volume.shape[0] // 2
    ax = volume[ax_idx, :, :]
    ax_wl = apply_window_level(ax, 400, 50)
    axes[2].imshow(ax_wl, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"Axial (z={ax_idx})")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存：{save_path}")


def visualize_volume_histogram(series: DcmSeries, save_path: Path):
    """可视化 3D 体积的 HU 值分布并保存"""
    volume = series.to_volume()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("3D Volume HU Distribution", fontsize=14)

    axes[0].hist(volume.flatten(), bins=500, color='steelblue', alpha=0.7)
    axes[0].set_title("Full HU Range Histogram")
    axes[0].set_xlabel("HU Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(-1024, 3000)

    mask = (volume > -500) & (volume < 1000)
    axes[1].hist(volume[mask], bins=100, color='coral', alpha=0.7)
    axes[1].set_title("Soft Tissue Range (-500 to 1000 HU)")
    axes[1].set_xlabel("HU Value")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存：{save_path}")


def compare_window_presets(img: DcmImage, save_path: Path):
    """比较不同窗宽窗位预设的效果并保存"""
    hu_arr = img.to_ndarray()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Window Preset Comparison", fontsize=14)

    presets = list(WINDOW_PRESETS.keys())

    for i, preset in enumerate(presets):
        row = i // 3
        col = i % 3
        windowed = apply_window_preset(hu_arr, preset)
        params = WINDOW_PRESETS[preset]
        axes[row, col].imshow(windowed, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f"{preset.title()}\nW:{params['width']}, C:{params['center']}")
        axes[row, col].axis('off')

    for i in range(len(presets), 6):
        row = i // 3
        col = i % 3
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存：{save_path}")


def run_all_tests():
    """运行所有测试和可视化"""
    print("\n" + "#" * 60)
    print("#" + " " * 18 + "DICOM 功能测试" + " " * 19 + "#")
    print("#" * 60 + "\n")

    # 1. 测试单个图像
    print("=" * 50)
    print("测试 1: 单个 DICOM 图像读取")
    print("=" * 50)
    img_path = Path(r"E:\pycharm23\Projs\DcmDataset\224699\100_99.dcm")
    img = DcmImage(img_path)
    print(f"文件：{img.filepath}")
    print(f"形状：{img.shape}, 数据类型：{img.dtype}")
    arr = img.to_ndarray()
    print(f"HU 范围：[{arr.min():.1f}, {arr.max():.1f}]")

    # 2. 测试序列加载
    print("\n" + "=" * 50)
    print("测试 2: DICOM 序列加载")
    print("=" * 50)
    directory = Path(r"E:\pycharm23\Projs\DcmDataset\224699")
    series = read_series(directory)
    print(f"序列包含 {len(series)} 帧图像")
    volume = series.to_volume()
    print(f"3D 体积：{volume.shape}")

    # 3. 测试 Tensor 转换
    print("\n" + "=" * 50)
    print("测试 3: torch Tensor 转换")
    print("=" * 50)
    tensor = img.to_tensor()
    print(f"单帧 Tensor: {tensor.shape}, {tensor.dtype}")
    volume_tensor = series.to_volume_tensor()
    print(f"体积 Tensor: {volume_tensor.shape}, {volume_tensor.dtype}")

    # 4. 生成可视化
    print("\n" + "=" * 50)
    print("生成可视化图表...")
    print("=" * 50)

    mid_idx = len(series) // 2
    mid_img = series.get_slice(mid_idx)

    visualize_single_image(mid_img, OUTPUT_DIR / "01_single_image_viz.png")
    compare_window_presets(mid_img, OUTPUT_DIR / "02_window_presets.png")
    visualize_series(series, OUTPUT_DIR / "03_series_overview.png")
    visualize_mpr(series, OUTPUT_DIR / "04_mpr_view.png")
    visualize_volume_histogram(series, OUTPUT_DIR / "05_volume_histogram.png")

    print("\n" + "#" * 60)
    print("#" + " " * 20 + "测试完成!" + " " * 20 + "#")
    print("#" * 60)
    print(f"\n可视化结果已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    run_all_tests()
