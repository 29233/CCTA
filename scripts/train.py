"""
视频分类模型训练脚本

支持乳腺和甲状腺 DICOM 视频序列的多属性分类训练。

使用方法:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --data_dir /path/to/data --organ breast
"""

import os
import sys
import time
import json
import argparse
import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import get_cfg_defaults, update_config
from data.dataset import build_dataloader
from modeling.kganet import NetTemporalFormer
from torch.utils.data import DataLoader


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DICOM Video Classification Training")

    # 数据配置
    parser.add_argument("--data_dir", type=str, default="",
                        help="数据目录路径")
    parser.add_argument("--train_dir", type=str, default="",
                        help="训练数据目录")
    parser.add_argument("--val_dir", type=str, default="",
                        help="验证数据目录")
    parser.add_argument("--organ", type=str, default="breast",
                        choices=["breast", "thyroid"],
                        help="器官类型")

    # 模型配置
    parser.add_argument("--backbone", type=str, default="resnet50",
                        help="骨干网络")
    parser.add_argument("--mode", type=str, default="HYBRID",
                        choices=["HYBRID", "2D"],
                        help="模型模式")
    parser.add_argument("--feature_dim", type=int, default=2048,
                        help="特征维度")
    parser.add_argument("--weights", type=str, default="",
                        help="预训练权重路径")

    # 训练配置
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="每个视频的采样帧数")

    # 检查点和日志
    parser.add_argument("--output_dir", type=str, default="output",
                        help="输出目录")
    parser.add_argument("--resume", type=str, default="",
                        help="恢复训练的检查点路径")
    parser.add_argument("--save_freq", type=int, default=10,
                        help="保存检查点的频率")
    parser.add_argument("--eval_freq", type=int, default=1,
                        help="评估频率")
    parser.add_argument("--print_freq", type=int, default=10,
                        help="打印日志频率")
    parser.add_argument("--use_tensorboard", action="store_true",
                        help="使用 TensorBoard")

    # 其他
    parser.add_argument("--config", type=str, default="",
                        help="配置文件路径")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
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
    if args.data_dir:
        cfg.DATA.TRAIN_DIR = os.path.join(args.data_dir, "train")
        cfg.DATA.VAL_DIR = os.path.join(args.data_dir, "val")
    if args.train_dir:
        cfg.DATA.TRAIN_DIR = args.train_dir
    if args.val_dir:
        cfg.DATA.VAL_DIR = args.val_dir

    cfg.ORGAN = args.organ
    cfg.DATA.ORGAN = args.organ

    cfg.MODEL.BACKBONE = args.backbone
    cfg.MODEL.MODE = args.mode
    cfg.MODEL.FEATURE_DIM = args.feature_dim
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights

    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.NUM_EPOCHS = args.num_epochs
    cfg.OPTIMIZER.BASE_LR = args.lr
    cfg.OPTIMIZER.WEIGHT_DECAY = args.weight_decay
    cfg.DATA.NUM_WORKERS = args.num_workers
    cfg.TRAIN.NUM_SAMPLES_PER_VIDEO = args.num_frames

    cfg.TRAIN.OUTPUT_DIR = args.output_dir
    cfg.TRAIN.RESUME = args.resume
    cfg.TRAIN.SAVE_FREQ = args.save_freq
    cfg.TRAIN.EVAL_FREQ = args.eval_freq
    cfg.TRAIN.PRINT_FREQ = args.print_freq

    cfg.DISTRIBUTED.NUM_GPUS = torch.cuda.device_count()

    # 创建输出目录
    os.makedirs(cfg.TRAIN.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()
    return cfg


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def build_model(cfg: "CfgNode") -> nn.Module:
    """构建模型"""
    model = NetTemporalFormer(cfg)

    # 加载预训练权重
    if cfg.MODEL.WEIGHTS:
        print(f"Loading weights from {cfg.MODEL.WEIGHTS}")
        checkpoint = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    return model


def build_optimizer(model: nn.Module, cfg: "CfgNode") -> optim.Optimizer:
    """构建优化器"""
    optimizer_type = cfg.OPTIMIZER.TYPE
    base_lr = cfg.OPTIMIZER.BASE_LR
    weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY

    # 收集需要优化和不需要权重衰减的参数
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_lr
        decay = weight_decay
        if "bias" in key or "bn" in key or "norm" in key:
            decay = cfg.OPTIMIZER.WEIGHT_DECAY_NORM
        params += [{"params": [value], "lr": lr, "weight_decay": decay}]

    if optimizer_type == "SGD":
        optimizer = optim.SGD(params, momentum=cfg.OPTIMIZER.MOMENTUM)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(params)
    elif optimizer_type == "AdamW":
        optimizer = optim.AdamW(params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    return optimizer


def build_scheduler(optimizer: optim.Optimizer, cfg: "CfgNode") -> optim.lr_scheduler._LRScheduler:
    """构建学习率调度器"""
    scheduler_type = cfg.LR_SCHEDULER.TYPE

    if scheduler_type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.LR_SCHEDULER.STEP_SIZE,
            gamma=cfg.LR_SCHEDULER.GAMMA
        )
    elif scheduler_type == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.LR_SCHEDULER.MILESTONES,
            gamma=cfg.LR_SCHEDULER.GAMMA
        )
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.TRAIN.NUM_EPOCHS,
            eta_min=1e-6
        )
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    return scheduler


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    output_dir: str,
    filename: str = "checkpoint.pth",
):
    """保存检查点"""
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": loss,
    }
    path = os.path.join(output_dir, filename)
    torch.save(state, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    resume_path: str,
) -> int:
    """加载检查点"""
    print(f"Resuming from {resume_path}")
    checkpoint = torch.load(resume_path, map_location="cpu")

    model.load_state_dict(checkpoint["model"])

    start_epoch = 0
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "epoch" in checkpoint:
        start_epoch = checkpoint["epoch"] + 1

    print(f"Resumed from epoch {start_epoch}")
    return start_epoch


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    cfg: "CfgNode",
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    """训练一个 epoch"""
    model.train()
    device = torch.device(cfg.MODEL.DEVICE)

    total_loss = 0.0
    loss_dict = {}
    num_batches = 0

    start_time = time.time()

    for batch_idx, batched_inputs in enumerate(dataloader):
        # 前向传播
        losses = model(batched_inputs)

        # 计算总损失
        total_batch_loss = sum(losses.values())

        # 反向传播
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()

        # 统计损失
        total_loss += total_batch_loss.item()
        for k, v in losses.items():
            if k not in loss_dict:
                loss_dict[k] = 0.0
            loss_dict[k] += v.item()

        num_batches += 1

        # 打印日志
        if (batch_idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
            avg_loss = total_loss / num_batches
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1}/{cfg.TRAIN.NUM_EPOCHS}] "
                  f"Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {avg_loss:.4f} "
                  f"({elapsed:.1f}s)")

    # 计算平均损失
    avg_loss_dict = {k: v / num_batches for k, v in loss_dict.items()}

    # 写入 TensorBoard
    if writer:
        writer.add_scalar("train/total_loss", total_loss / num_batches, epoch)
        for k, v in avg_loss_dict.items():
            writer.add_scalar(f"train/{k}", v, epoch)

    return avg_loss_dict


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    cfg: "CfgNode",
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)

    total_loss = 0.0
    loss_dict = {}
    num_batches = 0

    correct = 0
    total = 0

    for batched_inputs in dataloader:
        # 前向传播
        losses = model(batched_inputs)

        # 统计损失
        total_batch_loss = sum(losses.values())
        total_loss += total_batch_loss.item()
        for k, v in losses.items():
            if k not in loss_dict:
                loss_dict[k] = 0.0
            loss_dict[k] += v.item()

        num_batches += 1

        # 计算准确率（针对主要属性）
        if cfg.ORGAN == "breast":
            attr_name = "Pathology"
        else:
            attr_name = "病理"

        if attr_name in losses:
            # 获取预测和标签
            pred_dict = model(batched_inputs)
            # 这里需要根据实际输出格式调整
            # 简化实现，实际使用需要更完整的评估逻辑

    # 计算平均损失
    avg_loss_dict = {k: v / num_batches for k, v in loss_dict.items()}

    # 写入 TensorBoard
    if writer:
        writer.add_scalar("val/total_loss", total_loss / num_batches, epoch)
        for k, v in avg_loss_dict.items():
            writer.add_scalar(f"val/{k}", v, epoch)

    return avg_loss_dict


def main():
    """主函数"""
    args = parse_args()

    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 设置配置
    cfg = setup_config(args)

    # 设置随机种子
    set_seed(args.seed)

    # 设置输出目录
    output_dir = cfg.TRAIN.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # 设置 TensorBoard
    writer = None
    if args.use_tensorboard:
        log_dir = os.path.join(output_dir, "tensorboard")
        writer = SummaryWriter(log_dir)

    # 构建数据加载器
    print("Building dataloaders...")
    train_loader = build_dataloader(cfg, is_train=True, distributed=False)
    val_loader = build_dataloader(cfg, is_train=False, distributed=False)
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # 构建模型
    print("Building model...")
    model = build_model(cfg)
    model = model.to(torch.device(cfg.MODEL.DEVICE))
    print(f"Model: {cfg.MODEL.BACKBONE}, Mode: {cfg.MODEL.MODE}")

    # 构建优化器和调度器
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # 恢复训练
    start_epoch = 0
    if cfg.TRAIN.RESUME:
        start_epoch = load_checkpoint(model, optimizer, scheduler, cfg.TRAIN.RESUME)

    # 打印参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # 开始训练
    print(f"\nStarting training from epoch {start_epoch}...")
    print(f"Device: {cfg.MODEL.DEVICE}, Batch size: {cfg.TRAIN.BATCH_SIZE}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, cfg.TRAIN.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{cfg.TRAIN.NUM_EPOCHS}")
        print("-" * 40)

        # 训练
        train_losses = train_epoch(
            model, train_loader, optimizer, epoch, cfg, writer
        )

        # 打印训练损失
        print(f"Train Loss: {train_losses.get('loss_Pathology', train_losses.get('loss_病理', 0)):.4f}")

        # 评估
        if (epoch + 1) % cfg.TRAIN.EVAL_FREQ == 0:
            val_losses = evaluate(model, val_loader, epoch, cfg, writer)
            print(f"Val Loss: {val_losses.get('loss_Pathology', val_losses.get('loss_病理', 0)):.4f}")

            # 保存最佳模型
            current_val_loss = sum(val_losses.values())
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    current_val_loss, output_dir, "best_model.pth"
                )
                print(f"New best model saved! Val Loss: {best_val_loss:.4f}")

        # 更新学习率
        scheduler.step()

        # 保存检查点
        if (epoch + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                sum(train_losses.values()), output_dir,
                f"checkpoint_epoch{epoch + 1}.pth"
            )

    # 保存最终模型
    save_checkpoint(
        model, optimizer, scheduler, cfg.TRAIN.NUM_EPOCHS - 1,
        sum(train_losses.values()), output_dir, "final_model.pth"
    )

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")
    print("=" * 60)

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
