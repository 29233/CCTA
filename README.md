# CCTA —— 冠脉 CT 多任务学习项目

基于视频理解框架（SlowFast / MViTv2）与 DETR 式集合预测（匈牙利匹配）的冠状动脉 CT 影像多任务学习项目。

**任务设定**：对冠脉 CT 序列（DICOM）同时完成
1. **多类别分类**：识别 4 条冠脉分支（LAD 左前降支 / RCA 右冠状动脉 / LCX 左回旋支 / LM 左主干）+ 1 个背景类（共 5 类）
2. **回归**：预测斑块狭窄程度（归一化 0-1）

核心思路是把多分支预测建模为 **DETR 式集合预测问题**：模型输出 N 个 proposal token，每个 token 独立输出「分支类别 + 狭窄程度」，训练时用**匈牙利算法**求解预测与真值的一对一最优匹配。

---

## 目录结构

```
CCTA/
├── README.md                 # 本文件
├── dcm.py                    # DICOM 读取/处理模块（顶层本地框架）
├── test_dcm.py               # dcm.py 测试
├── upload_slowfast_files.py  # 将定制 SlowFast 文件批量上传到远程训练服务器
├── remote_config.json        # 远程服务器 SSH/conda 配置（不在 git 中，需自建）
│
├── config/                   # 顶层本地框架配置（yacs）
├── data/                     # 顶层本地数据集（乳腺/甲状腺超声视频分类）
├── modeling/                 # 顶层本地模型（NetTemporalFormer + Center Loss）
├── scripts/                  # 顶层本地训练/测试脚本
├── utils/                    # 冠脉 CT 数据预处理 + SSH 远程执行工具
│
├── SlowFast/                 # [git submodule] 定制的 SlowFast，含冠脉多任务/匈牙利损失实现 ★核心
├── detectron2/               # [git submodule] detectron2 fork（未使用硬依赖，可选）
└── pytorchvideo/             # [git submodule] pytorchvideo fork（可选）
```

> 项目由两部分组成：
> - **顶层本地框架**（`dcm.py`、`modeling/` 等）：早期版本，DICOM 读取、乳腺/甲状腺视频多属性分类，模型为 ResNet50 + 时序注意力。
> - **SlowFast 子模块内的冠脉实现**（★当前主线）：MViTv2 骨干 + DETR 式多 proposal 头 + 匈牙利损失，在远程 GPU 服务器上训练。

---

## 环境配置

### 1. 克隆与初始化子模块

```bash
git clone --recurse-submodules <repo_url>
# 或对已有仓库：
git submodule update --init SlowFast    # 必需，核心代码在此
git submodule update --init detectron2 pytorchvideo   # 可选
```

子模块指向自己的 fork（`github.com/29233/...`）。SlowFast 固定在提交 `53c0d1c`，其中的冠脉定制文件在远程服务器路径 `/CTA/slowfast` 下部署。

### 2. 本地环境（数据处理 / 轻量开发）

```bash
conda create -n ccta python=3.9 -y
conda activate ccta
pip install torch torchvision          # CPU 版即可满足数据预处理
pip install pydicom numpy matplotlib scipy yacs paramiko opencv-python
```

| 依赖 | 用途 |
|------|------|
| pydicom | DICOM 文件读取（`dcm.py`） |
| numpy / matplotlib | 体数据处理与可视化（HU 窗宽窗位、MPR） |
| torch / torchvision | 顶层模型 `modeling/kganet.py` |
| yacs | 顶层配置 `config/config.py` |
| paramiko | SSH 远程执行与文件上传（`utils/ssh_exec.py`） |
| scipy | 匈牙利算法 `linear_sum_assignment` |

### 3. 远程训练环境（GPU 服务器）

实际训练在远程 GPU 服务器上进行（代码通过 `upload_slowfast_files.py` 上传到 `/CTA/slowfast`，数据位于 `/data/Central`）。

**远程配置文件 `remote_config.json`**（仓库根目录，含密码，不入 git，需手动创建）：

```json
{
  "ssh": {"host": "<服务器IP>", "port": 22, "user": "<用户名>", "password": "<密码>"},
  "conda": {"env_name": "<远程conda环境名>", "extra_args": ""}
}
```

远程环境需要满足 SlowFast 官方依赖（PyTorch ≥ 1.10 + CUDA、fvcore、simplejson、psutil、tensorboard、opencv、scipy、decord/torchvision 解码等），安装方式见 `SlowFast/INSTALL.md`。

---

## 核心文件说明

### ★ DETR 式冠脉多任务实现（SlowFast 子模块内）

| 文件 | 功能 |
|------|------|
| `SlowFast/slowfast/models/coronary_head.py` | **`CoronaryMultiTaskHead` 多任务头（DETR 式）**。从 MViT 的 class token 经 `Linear(dim*2 → dim×N)` 生成 N 个 proposal token（`NUM_PROPOSALS`，默认 10，对应 DETR 的 object queries）；每个 token 接独立分类头（`CLS_HIDDEN_DIM→5 类`，无 softmax）和回归头（`REG_HIDDEN_DIM→1` + Sigmoid，输出 0-1 狭窄程度） |
| `SlowFast/slowfast/models/hungarian_loss.py` | **匈牙利匹配与损失（DETR 核心）**。`HungarianMatcher`：代价 = `-cls_prob(target_class) + λ·\|reg_pred - reg_target\|`，用 `scipy.optimize.linear_sum_assignment` 求一对一最优匹配；`HungarianLoss/V2`：匹配对算 CE+MSE（V2 可选 FocalLoss/SmoothL1），未匹配槽监督为背景类 |
| `SlowFast/slowfast/models/coronary_loss.py` | 非匹配版多任务损失（对照组），`build_coronary_loss(cfg)` 根据配置切换 |
| `SlowFast/slowfast/datasets/coronary.py` | `CoronaryMultiTask` 数据集：加载冠脉 CT 序列，产出 `cls_target`（0-3 分支 / 4 背景 / -1 padding）、`reg_target`（0-1 狭窄率）、`valid_mask` |
| `SlowFast/slowfast/tools/train_coronary_multitask.py` | 训练/验证入口脚本（AMP 混合精度、TensorBoard、断点续训） |
| `SlowFast/slowfast/config/custom_config.py` | `CORONARY.*`（NUM_PROPOSALS、损失权重、LOSS_TYPE='hungarian'）与 `HUNGARIAN.*`（匹配代价权重、COST_REG_TYPE='mse'/'l1'）配置节点定义 |
| `SlowFast/configs/Coronary/MVITv2_B_32x3_hungarian.yaml` | ★ 主配置：MViTv2-B 骨干（16 帧、224×224）、AdamW + cosine LR、hungarian v2 损失 |
| `SlowFast/configs/Coronary/MVITv2_B_32x3_multitask.yaml` | 非匈牙利（普通多任务）对照配置 |
| `SlowFast/slowfast/models/video_model_builder.py` | 模型组装：MViT 骨干 + 冠脉头 |
| `SlowFast/run_coronary_training.py` | 训练启动入口 |
| `SlowFast/HUNGARIAN_LOSS.md` / `CORONARY_MULTITASK.md` / `slowfast/models/hungarian.md` | 匈牙利损失的算法推导（公式）、数据格式与使用文档 |

### 顶层本地框架

| 文件 | 功能 |
|------|------|
| `dcm.py` | DICOM 读取模块：`DcmImage`（单帧，HU 值 Rescale 转换、窗宽窗位含冠脉预设 cardiac W400/C50）、`DcmSeries`（序列按 ImagePositionPatient 排序、3D 体数据、MPR 三平面重建） |
| `modeling/kganet.py` | `NetTemporalFormer`：ResNet50 逐帧特征 + 时序注意力加权融合 + 多属性分类头（乳腺 5 类 / 甲状腺 2 类），HYBRID/2D 两种模式 |
| `modeling/losses.py` | 多属性交叉熵 + CenterGramLoss（中心损失） |
| `data/dataset.py` | DICOM 序列视频数据集（每序列采样 16 帧） |
| `config/config.py` | yacs 配置：器官、骨干、损失权重、优化器、学习率调度 |
| `scripts/train.py` / `scripts/test.py` | 顶层框架训练/测试入口 |
| `utils/ssh_exec.py` | paramiko SSH 客户端：远程命令执行（可激活 conda）、文件上传 |
| `utils/compress_metadata.py`、`split_dataset.py`、`dcm_filename_pipeline.py`、`V1Filter.py` | 冠脉 CT 元数据清洗/压缩、数据集划分、DICOM 文件名标准化 pipeline |

---

## 使用方法

### 远程训练（主流程）

```bash
# 1. 创建 remote_config.json（见上文格式）

# 2. 上传定制的 SlowFast 文件到远程服务器
python upload_slowfast_files.py

# 3. 验证远程环境 import 是否正常
python test_remote_import.py

# 4. 远程启动训练（匈牙利损失版配置）
#    在远程服务器上执行：
python tools/run_net.py --cfg configs/Coronary/MVITv2_B_32x3_hungarian.yaml
```

### 常用配置覆盖

```bash
# 调整 proposal 槽数量
python tools/run_net.py --cfg configs/Coronary/MVITv2_B_32x3_hungarian.yaml \
    CORONARY.NUM_PROPOSALS 15

# 调整匹配中回归代价权重 / 类型
python tools/run_net.py --cfg configs/Coronary/MVITv2_B_32x3_hungarian.yaml \
    HUNGARIAN.COST_REG_WEIGHT 2.0 HUNGARIAN.COST_REG_TYPE l1

# 切换损失版本 v1/v2，或改用普通多任务损失
... CORONARY.HUNGARIAN_LOSS_VERSION v1
... CORONARY.LOSS_TYPE multi_task
```

### 本地数据预处理

```bash
# 冠脉 CT 元数据 pipeline，见 utils/readme.md
python utils/compress_metadata.py      # metadata_cleaned.xlsx -> metadataV0.csv
python utils/split_dataset.py          # 数据集划分
python utils/dcm_filename_pipeline.py  # DICOM 文件名标准化为 xxx.dcm
```

### 数据格式（匈牙利训练）

每个样本目录为一段 DICOM 序列；每个样本有 0~4 个真值目标：

- `cls_target: [B, N]`：0-3 = LAD/RCA/LCX/LM，4 = 背景，-1 = padding
- `reg_target: [B, N]`：0-1 斑块狭窄程度
- `valid_mask: [B, N]`：1 = 有效，0 = padding

训练数据远程路径：`DATA.PATH_TO_DATA_DIR=/data/Central`，`PATH_PREFIX=/data/Central/CTA`（见 yaml）。

---

## 模型架构（DETR 式，主流程）

```
冠脉 CT 序列 (DICOM, 16 帧, 224×224)
        ↓
MViTv2-B 骨干 (conv 模式, 24 层, pool attention)
        ↓
class token [B, D]
        ↓  Linear(D*2 → D×N)        ← proposal 生成
N 个 proposal token [B, N, D]        ← 对应 DETR object queries
        ↓
├── 分类头 ×N: Linear→256→GELU→Dropout→Linear(5)   (LAD/RCA/LCX/LM/背景)
└── 回归头 ×N: Linear→256→GELU→Dropout→Linear(1)→Sigmoid  (狭窄率 0-1)
        ↓
匈牙利匹配 (scipy linear_sum_assignment)
  cost = -p(c_target) + λ_reg · |v_pred - v_tgt|
        ↓
损失 = CE(匹配对) + λ·MSE/SmoothL1(匹配对回归) + CE(未匹配槽→背景)
```

---

## 备注

- detectron2 并非硬依赖：`modeling/kganet.py` 在 import 失败时自动降级为内置注册表。
- `SlowFast/` 内的 `README.md`、`INSTALL.md`、`MODEL_ZOO.md` 等为原版 SlowFast 文档；冠脉定制文档为 `README_CORONARY.md`、`CORONARY_MULTITASK.md`、`HUNGARIAN_LOSS.md`。
- 远程服务器上的部署根目录为 `/CTA`（远程绝对路径前缀 `/18018998051/...` 为服务器上的挂载路径写法）。
