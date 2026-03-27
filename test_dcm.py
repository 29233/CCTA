"""
DICOM 图像处理功能测试与可视化示例

演示 dcm.py 模块的各项功能，包括图像加载、转换、可视化和窗宽窗位调整。
"""

import numpy as np
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


def test_single_image():
    """测试单个 DICOM 图像读取"""
    print("=" * 50)
    print("测试 1: 单个 DICOM 图像读取")
    print("=" * 50)

    # 选择中间帧进行测试
    img_path = Path(r"E:\pycharm23\Projs\DcmDataset\224699\100_99.dcm")
    img = DcmImage(img_path)

    print(f"文件路径：{img.filepath}")
    print(f"图像形状：{img.shape}")
    print(f"数据类型：{img.dtype}")
    print(f"\n元数据:")
    meta = img.get_metadata()
    for k, v in meta.items():
        if v is not None:
            print(f"  {k}: {v}")

    # 测试 ndarray 转换
    arr = img.to_ndarray()
    print(f"\nndarray 信息:")
    print(f"  形状：{arr.shape}")
    print(f"  数据类型：{arr.dtype}")
    print(f"  值范围：[{arr.min():.1f}, {arr.max():.1f}]")
    print(f"  均值：{arr.mean():.1f}")

    # 测试 tensor 转换
    try:
        tensor = img.to_tensor()
        print(f"\ntorch Tensor 信息:")
        print(f"  形状：{tensor.shape}")
        print(f"  数据类型：{tensor.dtype}")
        print(f"  设备：{tensor.device}")
    except ImportError:
        print("\ntorch 未安装，跳过 Tensor 测试")

    return img


def test_series_loading():
    """测试 DICOM 序列加载"""
    print("\n" + "=" * 50)
    print("测试 2: DICOM 序列加载")
    print("=" * 50)

    directory = Path(r"E:\pycharm23\Projs\DcmDataset\240118")
    series = read_series(directory)

    print(f"序列目录：{series.directory}")
    print(f"图像帧数：{len(series)}")

    # 测试 3D 体积转换
    volume = series.to_volume()
    print(f"\n3D 体积信息:")
    print(f"  形状：{volume.shape} (Z, Y, X)")
    print(f"  数据类型：{volume.dtype}")
    print(f"  值范围：[{volume.min():.1f}, {volume.max():.1f}]")

    # 测试体积 Tensor 转换
    try:
        volume_tensor = series.to_volume_tensor()
        print(f"\n体积 Tensor 信息:")
        print(f"  形状：{volume_tensor.shape}")
        print(f"  数据类型：{volume_tensor.dtype}")
    except ImportError:
        print("\ntorch 未安装，跳过体积 Tensor 测试")

    # 显示前 5 帧和后 5 帧的文件名（验证排序）
    print(f"\n前 5 帧文件:")
    for i in range(min(5, len(series))):
        print(f"  [{i}] {series[i].filepath.name}")
    print(f"...")
    print(f"后 5 帧文件:")
    for i in range(max(0, len(series) - 5), len(series)):
        print(f"  [{i}] {series[i].filepath.name}")

    return series


def visualize_single_image(img: DcmImage):
    """可视化单个图像"""
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
    plt.show()


def visualize_series(series: DcmSeries):
    """可视化 DICOM 序列"""
    # 选择几个代表性切片进行显示
    n_slices = len(series)
    indices = [0, n_slices // 4, n_slices // 2, 3 * n_slices // 4, n_slices - 1]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f"DICOM Series Overview (Total: {n_slices} slices)", fontsize=14)

    for i, idx in enumerate(indices):
        img = series.get_slice(idx)
        hu_arr = img.to_ndarray()
        # 应用心脏窗（适合冠脉 CT）
        cardiac = apply_window_preset(hu_arr, 'cardiac')

        axes[i].imshow(cardiac, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f"Slice #{idx}\n{img.filepath.name}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    # 显示 MPR 视图
    visualize_mpr(series)


def visualize_mpr(series: DcmSeries):
    """多平面重建可视化"""
    volume = series.to_volume()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Multi-Planar Reconstruction (MPR)", fontsize=14)

    # 矢状面 (Sagittal) - X-Z 平面
    sag_idx = volume.shape[2] // 2
    sag = volume[:, :, sag_idx].T
    sag_wl = apply_window_level(sag, 400, 50)
    axes[0].imshow(sag_wl, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Sagittal View (x={sag_idx})")
    axes[0].axis('off')

    # 冠状面 (Coronal) - Y-Z 平面
    cor_idx = volume.shape[1] // 2
    cor = volume[:, cor_idx, :].T
    cor_wl = apply_window_level(cor, 400, 50)
    axes[1].imshow(cor_wl, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Coronal View (y={cor_idx})")
    axes[1].axis('off')

    # 轴状面 (Axial) - X-Y 平面
    ax_idx = volume.shape[0] // 2
    ax = volume[ax_idx, :, :]
    ax_wl = apply_window_level(ax, 400, 50)
    axes[2].imshow(ax_wl, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"Axial View (z={ax_idx})")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_volume_histogram(series: DcmSeries):
    """可视化 3D 体积的 HU 值分布"""
    volume = series.to_volume()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("3D Volume HU Distribution", fontsize=14)

    # 整体直方图
    axes[0].hist(volume.flatten(), bins=500, color='steelblue', alpha=0.7)
    axes[0].set_title("Full HU Range Histogram")
    axes[0].set_xlabel("HU Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(-1024, 3000)

    # 裁剪直方图（突出软组织和心脏）
    mask = (volume > -500) & (volume < 1000)
    axes[1].hist(volume[mask], bins=100, color='coral', alpha=0.7)
    axes[1].set_title("Soft Tissue Range (-500 to 1000 HU)")
    axes[1].set_xlabel("HU Value")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def compare_window_presets(img: DcmImage):
    """比较不同窗宽窗位预设的效果"""
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
        axes[row, col].set_title(f"{preset.title()}\nWidth: {params['width']}, Center: {params['center']}")
        axes[row, col].axis('off')

    # 隐藏多余的子图
    if len(presets) < 6:
        for i in range(len(presets), 6):
            row = i // 3
            col = i % 3
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def test_iteration_and_indexing(series: DcmSeries):
    """测试迭代和索引功能"""
    print("\n" + "=" * 50)
    print("测试 3: 迭代和索引功能")
    print("=" * 50)

    # 测试 __len__
    print(f"序列长度：{len(series)}")

    # 测试 __getitem__
    print(f"第一帧：{series[0].filepath.name}")
    print(f"中间帧：{series[len(series) // 2].filepath.name}")
    print(f"最后一帧：{series[-1].filepath.name}")

    # 测试 __iter__
    print("\n遍历前 5 帧的 HU 值统计:")
    for i, img in enumerate(series):
        if i >= 5:
            break
        arr = img.to_ndarray()
        print(f"  [{i}] {img.filepath.name}: min={arr.min():.1f}, max={arr.max():.1f}, mean={arr.mean():.1f}")


def run_all_tests():
    """运行所有测试和可视化"""
    print("\n" + "#" * 60)
    print("#" + " " * 18 + "DICOM 功能测试" + " " * 19 + "#")
    print("#" * 60 + "\n")

    # 1. 测试单个图像
    img = test_single_image()

    # 2. 测试序列加载
    series = test_series_loading()

    # 3. 测试迭代功能
    test_iteration_and_indexing(series)

    # 4. 可视化
    print("\n" + "=" * 50)
    print("生成可视化图表...")
    print("=" * 50)

    # 选择中间帧进行详细可视化
    mid_idx = len(series) // 2
    mid_img = series.get_slice(mid_idx)

    print("1. 单图像多视图可视化...")
    visualize_single_image(mid_img)

    print("2. 窗宽窗位预设比较...")
    compare_window_presets(mid_img)

    print("3. 序列概览...")
    visualize_series(series)

    print("4. 体积 HU 分布直方图...")
    visualize_volume_histogram(series)

    print("\n" + "#" * 60)
    print("#" + " " * 20 + "测试完成!" + " " * 20 + "#")
    print("#" * 60)


if __name__ == "__main__":
    run_all_tests()
