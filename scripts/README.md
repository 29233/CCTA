# 视频分类模型训练和测试指南

## 目录结构

```
DcmDataset/
├── config/                 # 配置模块
│   ├── __init__.py
│   └── config.py          # 配置定义
├── data/                   # 数据加载模块
│   ├── __init__.py
│   └── dataset.py         # 数据集和数据加载器
├── modeling/               # 模型定义
│   ├── __init__.py
│   ├── kganet.py          # 网络模型
│   └── losses.py          # 损失函数
├── utils/                  # 工具函数
│   ├── __init__.py
│   └── image_list.py      # ImageList 工具类
├── scripts/                # 训练和测试脚本
│   ├── train.py           # 训练脚本
│   └── test.py            # 测试脚本
├── configs/                # 配置文件
│   └── config.yaml        # 默认配置
├── dcm.py                  # DICOM 读取模块
└── output/                 # 模型输出目录（训练生成）
```

---

## 快速开始

### 1. 训练模型

#### 基本用法
```bash
# 使用默认配置训练（乳腺数据集）
python scripts/train.py --data_dir /path/to/data --organ breast

# 甲状腺数据集训练
python scripts/train.py --data_dir /path/to/data --organ thyroid

# 指定训练和验证目录
python scripts/train.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --organ breast
```

#### 使用配置文件
```bash
# 编辑 configs/config.yaml 后使用
python scripts/train.py --config configs/config.yaml
```

#### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | "" | 数据目录（包含 train/val 子目录） |
| `--train_dir` | "" | 训练数据目录 |
| `--val_dir` | "" | 验证数据目录 |
| `--organ` | breast | 器官类型：breast/thyroid |
| `--backbone` | resnet50 | 骨干网络 |
| `--mode` | HYBRID | 模型模式：HYBRID/2D |
| `--batch_size` | 8 | 批次大小 |
| `--num_epochs` | 100 | 训练轮数 |
| `--lr` | 0.001 | 学习率 |
| `--num_frames` | 16 | 每个视频采样帧数 |
| `--output_dir` | output | 输出目录 |
| `--resume` | "" | 恢复训练的检查点 |
| `--gpu` | 0 | GPU 设备 |

#### 训练输出
```
output/
├── config.json           # 保存的配置
├── best_model.pth        # 最佳验证集模型
├── final_model.pth       # 最终模型
├── checkpoint_epoch10.pth  # 定期保存的检查点
└── tensorboard/          # TensorBoard 日志
```

---

### 2. 测试模型

#### 单个样本预测
```bash
# 预测单个 DICOM 文件
python scripts/test.py \
    --checkpoint output/best_model.pth \
    --input /path/to/image.dcm

# 预测视频序列（目录）
python scripts/test.py \
    --checkpoint output/best_model.pth \
    --input /path/to/video_sequence/
```

#### 批量测试
```bash
python scripts/test.py \
    --checkpoint output/best_model.pth \
    --data_dir /path/to/test_data \
    --save_results
```

#### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | (必需) | 模型检查点路径 |
| `--input` | "" | 单个 DICOM 文件或序列目录 |
| `--data_dir` | "" | 测试数据集目录 |
| `--organ` | breast | 器官类型 |
| `--num_frames` | 16 | 采样帧数 |
| `--output_dir` | output_test | 输出目录 |
| `--save_results` | False | 保存测试结果 |
| `--verbose` | False | 显示详细信息 |

---

## 数据格式

### 训练数据目录结构
```
data/
├── train/
│   ├── sample_001/      # 每个样本一个子目录
│   │   ├── 1.dcm
│   │   ├── 2.dcm
│   │   ├── ...
│   │   └── labels.json  # 标签文件（可选）
│   ├── sample_002/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### 标签文件格式

#### labels.json
```json
{
    "Pathology": 2,
    "BI-RADS": 1
}
```

#### label.txt（简化格式）
```
Pathology: 2
```

如果没有标签文件，模型会尝试从目录名解析标签。

---

## 配置说明

### 主要配置项

```yaml
# 器官类型
ORGAN: "breast"  # breast 或 thyroid

# 数据配置
DATA:
  INPUT_SIZE: 224        # 输入图像尺寸
  NUM_WORKERS: 4         # 数据加载线程数

# 模型配置
MODEL:
  BACKBONE: "resnet50"   # 骨干网络
  FEATURE_DIM: 2048      # 特征维度
  MODE: "HYBRID"         # HYBRID(时序) 或 2D(单帧)

# 属性标签（乳腺）
ATTR_LIST: ["Pathology"]
ATTR_WEIGHT: [[1.0, 1.0, 1.0, 1.0, 1.0]]  # 5 类权重

# 训练配置
TRAIN:
  BATCH_SIZE: 8
  NUM_EPOCHS: 100
  NUM_SAMPLES_PER_VIDEO: 16  # 每视频采样帧数
```

---

## 模型架构

### NetTemporalFormer

```
输入：视频序列 [B, T, C, H, W]
       ↓
Net2DCore (骨干网络)
       ↓
每帧特征：[B*T, feature_dim]
       ↓
时序注意力机制
       ↓
加权平均特征：[B, feature_dim]
       ↓
多属性分类头
       ↓
输出：多个属性的预测概率
```

### 支持的模式

- **HYBRID**: 使用时序注意力机制融合多帧特征
- **2D**: 仅使用单帧特征（忽略时序信息）

---

## 使用示例

### 完整训练流程

```bash
# 1. 准备数据
# 将 DICOM 序列组织到 train/ 和 val/ 目录

# 2. 开始训练（使用 TensorBoard）
python scripts/train.py \
    --data_dir /data/breast_ultrasound \
    --organ breast \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.001 \
    --use_tensorboard

# 3. 查看训练日志
tensorboard --logdir output/tensorboard

# 4. 测试最佳模型
python scripts/test.py \
    --checkpoint output/best_model.pth \
    --data_dir /data/breast_ultrasound/test \
    --save_results \
    --verbose
```

### 单张图像预测

```python
import sys
sys.path.insert(0, '/path/to/DcmDataset')

import torch
from config.config import get_cfg_defaults
from modeling.kganet import NetTemporalFormer
from scripts.test import predict_single_image

# 加载配置
cfg = get_cfg_defaults()
cfg.ORGAN = "breast"
cfg.MODEL.DEVICE = "cuda"

# 加载模型
model = NetTemporalFormer(cfg)
checkpoint = torch.load("output/best_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval().cuda()

# 预测
results = predict_single_image(model, "sample.dcm", cfg)
print(f"预测结果：{results['predictions']['Pathology']}")
```

---

## 常见问题

### 1. 显存不足
- 减小 `--batch_size`
- 减少 `--num_frames`
- 使用更小的骨干网络（如 resnet18）

### 2. 数据加载慢
- 增加 `--num_workers`
- 将数据放在 SSD 上
- 设置 `PIN_MEMORY: True`

### 3. 标签解析错误
- 确保标签文件格式正确
- 检查目录命名与标签的对应关系

### 4. detectron2 依赖
代码已移除对 detectron2 的硬依赖，如无 detectron2 会自动使用替代实现。
