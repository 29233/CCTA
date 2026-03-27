"""
视频分类模型测试脚本

支持乳腺和甲状腺 DICOM 视频序列的多属性分类测试和推理。

使用方法:
    python scripts/test.py --checkpoint output/best_model.pth --data_dir /path/to/test
    python scripts/test.py --config configs/config.yaml --checkpoint output/best_model.pth
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import get_cfg_defaults
from data.dataset import build_dataloader, DcmVideoDataset, VideoTransform
from modeling.kganet import NetTemporalFormer
from dcm import DcmImage, read_series


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DICOM Video Classification Testing")

    # 数据配置
    parser.add_argument("--data_dir", type=str, default="",
                        help="测试数据目录")
    parser.add_argument("--input", type=str, default="",
                        help="单个 DICOM 文件或序列目录")

    # 模型配置
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--config", type=str, default="",
                        help="配置文件路径")
    parser.add_argument("--organ", type=str, default="breast",
                        choices=["breast", "thyroid"],
                        help="器官类型")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        help="骨干网络")
    parser.add_argument("--mode", type=str, default="HYBRID",
                        choices=["HYBRID", "2D"],
                        help="模型模式")
    parser.add_argument("--feature_dim", type=int, default=2048,
                        help="特征维度")

    # 测试配置
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="每个视频的采样帧数")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")

    # 输出配置
    parser.add_argument("--output_dir", type=str, default="output_test",
                        help="输出目录")
    parser.add_argument("--save_results", action="store_true",
                        help="保存测试结果")
    parser.add_argument("--verbose", action="store_true",
                        help="显示详细信息")

    # 其他
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU 设备")

    args = parser.parse_args()
    return args


def setup_config(args) -> "CfgNode":
    """设置配置"""
    cfg = get_cfg_defaults()

    # 从文件加载配置
    if args.config:
        cfg.merge_from_file(args.config)

    # 从命令行参数覆盖配置
    cfg.ORGAN = args.organ
    cfg.DATA.ORGAN = args.organ
    cfg.DATA.TEST_DIR = args.data_dir

    cfg.MODEL.BACKBONE = args.backbone
    cfg.MODEL.MODE = args.mode
    cfg.MODEL.FEATURE_DIM = args.feature_dim

    cfg.TEST.BATCH_SIZE = args.batch_size
    cfg.TRAIN.NUM_SAMPLES_PER_VIDEO = args.num_frames
    cfg.DATA.NUM_WORKERS = args.num_workers

    cfg.TEST.CHECKPOINT = args.checkpoint
    cfg.TEST.OUTPUT_DIR = args.output_dir
    cfg.TEST.SAVE_RESULTS = args.save_results

    # 创建输出目录
    os.makedirs(cfg.TEST.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()
    return cfg


def load_model(cfg: "CfgNode") -> nn.Module:
    """加载模型"""
    model = NetTemporalFormer(cfg)

    # 加载检查点
    checkpoint_path = cfg.TEST.CHECKPOINT
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(torch.device(cfg.MODEL.DEVICE))
    model.eval()

    print(f"Model loaded: {cfg.MODEL.BACKBONE}, Mode: {cfg.MODEL.MODE}")
    return model


@torch.no_grad()
def predict_single_video(
    model: nn.Module,
    video_path: str,
    cfg: "CfgNode",
    num_frames: int = 16,
) -> Dict:
    """
    预测单个视频

    Args:
        model: 模型
        video_path: 视频目录路径
        cfg: 配置对象
        num_frames: 采样帧数

    Returns:
        预测结果字典
    """
    device = torch.device(cfg.MODEL.DEVICE)

    # 加载 DICOM 序列
    try:
        series = read_series(video_path)
        num_total_frames = len(series)

        # 均匀采样帧
        if num_total_frames <= num_frames:
            frame_indices = list(range(num_total_frames))
        else:
            step = num_total_frames / num_frames
            frame_indices = [int(i * step) for i in range(num_frames)]

        # 构建输入
        transform = VideoTransform(cfg, is_train=False)
        frames = []

        for frame_idx in frame_indices:
            frame_img = series.get_slice(frame_idx)
            hu_array = frame_img.to_ndarray(apply_rescale=True)

            # 归一化 HU 值到 0-255
            hu_min, hu_max = -1000, 1000
            normalized = np.clip((hu_array - hu_min) / (hu_max - hu_min) * 255, 0, 255).astype(np.uint8)
            normalized = np.stack([normalized] * 3, axis=-1)

            # 应用变换
            image_tensor = transform(normalized)
            frames.append({
                "image": image_tensor,
                "attr_label": {"Pathology": 0},  # 占位符
                "file_name": f"{Path(video_path).name}_frame{frame_idx}",
            })

        # 包装为批次
        batched_inputs = [[{
            "image": f["image"].to(device),
            "attr_label": f["attr_label"],
            "file_name": f["file_name"],
        } for f in frames]]

        # 推理
        pred_dict = model(batched_inputs)

        # 解析结果
        results = {
            "video_path": video_path,
            "file_name": Path(video_path).name,
            "num_frames": len(frames),
            "predictions": {},
            "probabilities": {},
        }

        # 获取属性列表
        if cfg.ORGAN == "breast":
            from data.dataset import ATTR_LIB_BREAST as ATTR_LIB
        else:
            from data.dataset import ATTR_LIB

        for attr in pred_dict.keys():
            if attr in ["attn", "features", "distmat"]:
                continue

            probs = pred_dict[attr][0].cpu().numpy()  # (num_classes,)
            pred_class = int(np.argmax(probs))

            # 获取类别名称
            if attr in ATTR_LIB:
                class_names = ATTR_LIB[attr]
                pred_name = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
            else:
                pred_name = str(pred_class)
                class_names = [str(i) for i in range(len(probs))]

            results["predictions"][attr] = {
                "class_id": pred_class,
                "class_name": pred_name,
                "confidence": float(np.max(probs)),
            }
            results["probabilities"][attr] = {
                class_names[i]: float(probs[i]) for i in range(len(probs))
            }

        return results

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return {
            "video_path": video_path,
            "file_name": Path(video_path).name,
            "error": str(e),
        }


@torch.no_grad()
def predict_single_image(
    model: nn.Module,
    image_path: str,
    cfg: "CfgNode",
) -> Dict:
    """
    预测单个 DICOM 图像

    Args:
        model: 模型
        image_path: DICOM 文件路径
        cfg: 配置对象

    Returns:
        预测结果字典
    """
    device = torch.device(cfg.MODEL.DEVICE)

    # 加载 DICOM 图像
    img = DcmImage(image_path)
    hu_array = img.to_ndarray(apply_rescale=True)

    # 归一化
    hu_min, hu_max = -1000, 1000
    normalized = np.clip((hu_array - hu_min) / (hu_max - hu_min) * 255, 0, 255).astype(np.uint8)
    normalized = np.stack([normalized] * 3, axis=-1)

    # 应用变换
    transform = VideoTransform(cfg, is_train=False)
    image_tensor = transform(normalized)

    # 包装为批次（单帧视频）
    frames = [{
        "image": image_tensor.to(device),
        "attr_label": {"Pathology": 0},
        "file_name": Path(image_path).name,
    }]
    batched_inputs = [frames]

    # 推理
    pred_dict = model(batched_inputs)

    # 解析结果
    results = {
        "image_path": image_path,
        "file_name": Path(image_path).name,
        "predictions": {},
        "probabilities": {},
    }

    # 获取属性列表
    if cfg.ORGAN == "breast":
        from data.dataset import ATTR_LIB_BREAST as ATTR_LIB
    else:
        from data.dataset import ATTR_LIB

    for attr in pred_dict.keys():
        if attr in ["attn", "features", "distmat"]:
            continue

        probs = pred_dict[attr][0].cpu().numpy()
        pred_class = int(np.argmax(probs))

        if attr in ATTR_LIB:
            class_names = ATTR_LIB[attr]
            pred_name = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
        else:
            pred_name = str(pred_class)
            class_names = [str(i) for i in range(len(probs))]

        results["predictions"][attr] = {
            "class_id": pred_class,
            "class_name": pred_name,
            "confidence": float(np.max(probs)),
        }
        results["probabilities"][attr] = {
            class_names[i]: float(probs[i]) for i in range(len(probs))
        }

    return results


@torch.no_grad()
def test_model(
    model: nn.Module,
    dataloader: DataLoader,
    cfg: "CfgNode",
) -> Tuple[Dict, List[Dict]]:
    """
    测试模型

    Args:
        model: 模型
        dataloader: 数据加载器
        cfg: 配置对象

    Returns:
        (metrics, results) 评估指标和详细结果
    """
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)

    all_predictions = []
    all_labels = []
    all_probs = []
    results_list = []

    for batched_inputs in dataloader:
        # 推理
        pred_dict = model(batched_inputs)

        # 解析结果
        for i, video_inputs in enumerate(batched_inputs):
            video_name = video_inputs[0]["file_name"]
            results = {
                "video_name": video_name,
                "predictions": {},
                "probabilities": {},
            }

            # 获取标签
            labels = {}
            for attr in video_inputs[0]["attr_label"]:
                labels[attr] = int(video_inputs[0]["attr_label"][attr])

            # 获取预测
            if cfg.ORGAN == "breast":
                from data.dataset import ATTR_LIB_BREAST as ATTR_LIB
            else:
                from data.dataset import ATTR_LIB

            for attr in pred_dict.keys():
                if attr in ["attn", "features", "distmat"]:
                    continue

                probs = pred_dict[attr][i].cpu().numpy()
                pred_class = int(np.argmax(probs))

                if attr in ATTR_LIB:
                    class_names = ATTR_LIB[attr]
                else:
                    class_names = [str(i) for i in range(len(probs))]

                results["predictions"][attr] = pred_class
                results["probabilities"][attr] = probs

                all_predictions.append(pred_class)
                if attr in labels:
                    all_labels.append(labels[attr])
                all_probs.append(probs)

            results_list.append(results)

    # 计算指标
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = (all_predictions == all_labels).mean() if len(all_labels) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy),
        "num_samples": len(all_labels),
    }

    return metrics, results_list


def save_results(results: List[Dict], output_dir: str):
    """保存测试结果"""
    output_path = os.path.join(output_dir, "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_path}")


def print_results(results: Dict, verbose: bool = False):
    """打印结果"""
    print("\n" + "=" * 50)

    if "video_path" in results:
        # 单个样本预测
        print(f"File: {results.get('file_name', 'Unknown')}")
        print(f"Path: {results.get('video_path', results.get('image_path', 'Unknown'))}")

        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print("\nPredictions:")
            for attr, pred in results["predictions"].items():
                print(f"  {attr}: {pred['class_name']} (confidence: {pred['confidence']:.4f})")

                if verbose and attr in results["probabilities"]:
                    print("    Probabilities:")
                    for cls, prob in results["probabilities"][attr].items():
                        print(f"      {cls}: {prob:.4f}")
    else:
        # 批量测试结果
        print(f"Total samples: {results.get('num_samples', 0)}")
        print(f"Accuracy: {results.get('accuracy', 0):.4f}")

    print("=" * 50 + "\n")


def main():
    """主函数"""
    args = parse_args()

    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 设置配置
    cfg = setup_config(args)

    # 加载模型
    model = load_model(cfg)

    # 创建输出目录
    os.makedirs(cfg.TEST.OUTPUT_DIR, exist_ok=True)

    # 单个文件/目录预测
    if args.input:
        input_path = Path(args.input)

        if input_path.is_file() and input_path.suffix.lower() in [".dcm", ".dicom"]:
            # 单个 DICOM 文件
            print(f"Processing image: {args.input}")
            results = predict_single_image(model, args.input, cfg)
        elif input_path.is_dir():
            # 目录（视频序列）
            print(f"Processing video: {args.input}")
            results = predict_single_video(model, args.input, cfg, args.num_frames)
        else:
            print(f"Invalid input: {args.input}")
            return

        # 打印和保存结果
        print_results(results, args.verbose)

        if args.save_results:
            save_results([results], cfg.TEST.OUTPUT_DIR)

        return

    # 批量测试
    if args.data_dir:
        print(f"Testing on dataset: {args.data_dir}")

        # 构建数据加载器
        test_loader = build_dataloader(cfg, is_train=False, distributed=False)
        print(f"Test samples: {len(test_loader)}")

        # 测试
        metrics, results = test_model(model, test_loader, cfg)

        # 打印结果
        print_results(metrics, args.verbose)

        # 保存结果
        if args.save_results:
            save_results(results, cfg.TEST.OUTPUT_DIR)
            print(f"Metrics: {json.dumps(metrics, indent=2)}")

        return

    print("Please provide --input or --data_dir")


if __name__ == "__main__":
    main()
