# 冠脉 CT 数据预处理工具

本目录包含用于冠脉 CT 数据集元数据处理和 DICOM 文件名标准化的工具脚本。

---

## 文件概览

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `clean_metadata.py` | 数据清洗 | `metadata.xlsx` | `metadata_cleaned.xlsx` |
| `compress_metadata.py` | 数据压缩 | `metadata_cleaned.xlsx` | `metadataV0.csv` |
| `split_dataset.py` | 数据集划分 | `metadataV0.csv` | `metadata_split.csv` |
| `dcm_filename_pipeline.py` | DICOM 文件名标准化 | `metadata_cleaned.xlsx` + DICOM 文件 | 重命名后的 DICOM 文件 |

---

## DICOM 文件名标准化 Pipeline

### 功能描述

`dcm_filename_pipeline.py` 是一个完整的 DICOM 文件名处理 pipeline，用于将冠脉 CT 数据集中的 DICOM 文件统一标准化为 `xxx.dcm` 格式。

### 处理流程

```
原始 DICOM 文件
    ↓
第一阶段：处理 xxx_xxx.dcm 格式
    ├── 符合格式 → 重命名为 xxx.dcm
    └── 不符合格式 → 收集待处理
    ↓
第二阶段：处理不符合格式的文件
    ├── 规则 1: 末尾有≥4 位数字 → 截取最后 3 位
    └── 规则 2: 有小数点 → 保留最后一段
    ↓
验证：检查所有文件是否符合 xxx.dcm 格式
    ↓
标准化完成的 DICOM 文件
```

### 详细处理逻辑

#### 第一阶段：xxx_xxx.dcm -> xxx.dcm

```python
# 匹配模式：2-3 位数字_2-3 位数字.dcm
pattern = re.compile(r'^(\d{2,3})_(\d{2,3})\.dcm$', re.IGNORECASE)

for file in files:
    if pattern.match(file):
        # 提取下划线前的部分
        new_name = f"{match.group(1)}.dcm"
        os.rename(old_path, new_path)
```

**示例：**
- `001_001.dcm` -> `001.dcm`
- `12_34.dcm` -> `12.dcm`
- `123_456.dcm` -> `123.dcm`

#### 第二阶段：备用命名规则

对于不符合第一阶段格式的文件，按以下规则处理：

**规则 1：末尾连续数字截取**
```python
# 匹配末尾≥4 位连续数字
digit_pattern = re.compile(r'(\d{4,})\.dcm$', re.IGNORECASE)

if digit_match:
    # 截取最后 3 位
    new_name = f"{last_digits[-3:]}.dcm"
```

**示例：**
- `1.3.12.2.1107.5.1.4.73926.30000022030605305267100155326.dcm` -> `326.dcm`
- `1.2.840.113619.2.476.2414295668941465463751165007482057506.100.dcm` -> `100.dcm`

**规则 2：小数点后最后一段**
```python
elif '.' in file_stem:
    # 保留最后一个小数点后的部分
    last_part = file_stem.split('.')[-1]
    new_name = f"{last_part}.dcm"
```

**示例：**
- `IM-0001.0001.dcm` -> `0001.dcm`
- `SE.12.IM.01.dcm` -> `01.dcm`

**规则 3：文件茎末尾数字部分（后备规则）**
```python
else:
    # 提取文件茎末尾的连续数字
    stem_digit_match = re.search(r'(\d+)(?!.*\d)', file_stem)
    if stem_digit_match:
        new_name = f"{last_num}.dcm"
```

**示例：**
- `SE-12-IM-01.dcm` -> `01.dcm`
- `Image_001.dcm` -> `001.dcm`
- `Slice-123.dcm` -> `123.dcm`

### 容灾逻辑

1. **文件夹不存在处理**
   ```python
   if not os.path.exists(folder_path):
       result.missing_folders.append(id_str)
       logger.warning(f"ID {id_str}: 文件夹不存在，跳过")
       continue
   ```

2. **文件名冲突处理**
   ```python
   if os.path.exists(new_path):
       # 添加后缀避免冲突
       counter = 1
       while os.path.exists(new_path):
           new_name = f"{base}_{counter}{ext}"
           counter += 1
   ```

3. **异常捕获**
   ```python
   try:
       os.rename(old_path, new_path)
       result.phase_renamed += 1
   except Exception as e:
       result.add_error(id_str, f, str(e))
       logger.error(f"重命名失败 {f}: {e}")
   ```

4. **完整日志记录**
   - 控制台输出：简要处理进度
   - 日志文件：详细操作记录（包含时间戳）

### 使用方法

**命令行运行：**

```bash
# 使用默认路径
python utils/dcm_filename_pipeline.py

# 自定义路径
python utils/dcm_filename_pipeline.py \
  --metadata /path/to/metadata_cleaned.xlsx \
  --data-root /path/to/data \
  --subfolder CTA
```

**Python 调用：**

```python
from utils.dcm_filename_pipeline import run_pipeline

result = run_pipeline(
    metadata_path="/path/to/metadata_cleaned.xlsx",
    data_root="/path/to/data",
    subfolder="CTA",
)

# 查看处理结果
print(result.summary())
```

### 输出日志

处理完成后会生成：
1. **控制台输出**：处理进度和统计摘要
2. **日志文件**：`logs/dcm_rename_YYYYMMDD_HHMMSS.log`

### 类与函数说明

| 类/函数 | 功能 |
|--------|------|
| `RenameConfig` | 配置类，存储路径、正则模式等参数 |
| `ProcessingResult` | 结果类，统计各阶段处理数据 |
| `process_phase1()` | 第一阶段处理函数 |
| `process_phase2()` | 第二阶段处理函数 |
| `generate_new_name()` | 根据备用规则生成新文件名 |
| `verify_results()` | 验证处理结果 |
| `run_pipeline()` | 完整 pipeline 入口函数 |

---

## clean_metadata.py - 数据清洗脚本

### 功能描述

对原始冠脉 CT 数据集的 `metadata.xlsx` 文件进行清洗，去除非法数据，填充缺失值，生成标准化的中间文件。

### 处理流程

```
原始数据 (471 行)
    ↓
步骤 1: 删除 Start Frame > End Frame 的行 (-141 行)
    ↓
步骤 2: 向下填充 ID 列空白值
    ↓
步骤 3: 删除 Plaque 和 Branch 都为空的 ID 的所有行 (-42 行)
    ↓
步骤 4: 删除全空列
    ↓
清洗后数据 (288 行，97 个唯一 ID)
```

### 详细处理逻辑

#### 步骤 1: 检查帧范围合法性

```python
# 将 Start Frame 和 End Frame 转换为数值类型
df['Start Frame'] = pd.to_numeric(df['Start Frame'], errors='coerce')
df['End Frame'] = pd.to_numeric(df['End Frame'], errors='coerce')

# 删除 Start Frame > End Frame 的行
df = df[df['Start Frame'] <= df['End Frame']]
```

**说明：** 仅检查 `start > end` 的情况，不处理空白值。

#### 步骤 2: 填充 ID 列空白

```python
# 使用前向填充（forward fill）
df['ID'] = df['ID'].ffill()

# 删除仍为空的行（如第一行）
df = df.dropna(subset=['ID'])

# 转换为整数类型
df['ID'] = df['ID'].astype(int)
```

**说明：** 按照空白上方的 ID 向下填充至非空。

#### 步骤 3: 删除无效 ID

```python
def has_valid_plaque_or_branch(group):
    """检查组内是否有 Plaque 或 Branch 非空的行"""
    return group['Plaque'].notna().any() or group['Branch'].notna().any()

# 过滤掉所有行 Plaque 和 Branch 都为空的 ID
df_valid = df.groupby('ID').filter(has_valid_plaque_or_branch)
```

**说明：** 如果同一个 ID 的所有行中 Plaque 和 Branch 列数据都为空，删除这个 ID 的所有行。

#### 步骤 4: 删除空白列

```python
# 删除全为空的列
df = df.dropna(axis=1, how='all')
```

### 输入输出

| 项目 | 格式 | 列 |
|------|------|-----|
| **输入** | Excel (.xlsx) | ID, Type, Start Frame, End frame, Plaque, Branch |
| **输出** | Excel (.xlsx) | ID, Type, Start Frame, End Frame, Plaque, Branch |

### 使用方法

```python
from utils.clean_metadata import clean_metadata

clean_metadata(
    input_path="path/to/metadata.xlsx",
    output_path="path/to/metadata_cleaned.xlsx"
)
```

或命令行运行：

```bash
python utils/clean_metadata.py
```

---

## compress_metadata.py - 数据压缩脚本

### 功能描述

将清洗后的数据按 ID 进行压缩，每个 ID 的多行血管数据合并为 1 行，生成用于视频回归模型的最终数据集描述文件。

### 处理流程

```
清洗后数据 (288 行，97 个 ID)
    ↓
对每个 ID:
  1. 找到最大 Plaque 值
  2. 确定对应的 Branch
  3. 根据 Branch 类型确定起止帧
    ↓
压缩后数据 (97 行，每 ID 1 行)
```

### 详细处理逻辑

#### 核心算法

```python
VALID_TYPES = {'RCA', 'LAD', 'LCX'}

for id_val, group in df.groupby('ID'):
    # 1. 找到 Plaque 最大值
    plaque_max = group['Plaque'].max()

    # 2. 找到对应的 Branch
    target_branch = group[group['Plaque'] == plaque_max].iloc[0]['Branch']

    # 3. 根据 Branch 确定起止帧
    if target_branch in VALID_TYPES:
        # Branch 是 RCA/LAD/LCX 之一
        type_rows = group[group['Type'] == target_branch]
        if len(type_rows) > 0:
            # 使用该 Type 行的起止帧
            start_frame = type_rows.iloc[0]['Start Frame']
            end_frame = type_rows.iloc[0]['End Frame']
        else:
            # 无对应 Type 行，使用最大范围
            start_frame = group['Start Frame'].min()
            end_frame = group['End Frame'].max()
    else:
        # Branch 不是 RCA/LAD/LCX（如 CX、L Main、Vessel 等）
        # 使用最大范围
        start_frame = group['Start Frame'].min()
        end_frame = group['End Frame'].max()

    # 4. 创建记录（删除 Type 列）
    record = {
        'ID': int(id_val),
        'Start Frame': int(start_frame),
        'End Frame': int(end_frame),
        'Plaque': plaque_max,
        'Branch': target_branch,
    }
```

#### 决策逻辑图

```
最大 Plaque 对应的 Branch
         │
         ├── 是 RCA/LAD/LCX ──→ 使用该 Type 行的 Start/End Frame
         │
         └── 否 (CX/L Main/Vessel 等) ──→ 使用所有行的最小 Start 和最大 End
```

### 输入输出

| 项目 | 格式 | 列 |
|------|------|-----|
| **输入** | Excel (.xlsx) | ID, Type, Start Frame, End Frame, Plaque, Branch |
| **输出** | CSV (.csv) | ID, Start Frame, End Frame, Plaque, Branch |

**注意：** 输出文件删除了 Type 列。

### 输出字段说明

| 字段 | 说明 |
|------|------|
| ID | 患者唯一标识符 |
| Start Frame | 血管起始帧（根据 Branch 类型确定） |
| End Frame | 血管结束帧（根据 Branch 类型确定） |
| Plaque | 该 ID 的最大堵塞程度 |
| Branch | 最大 Plaque 对应的血管分支 |

### 使用方法

```python
from utils.compress_metadata import compress_metadata

compress_metadata(
    input_path="path/to/metadata_cleaned.xlsx",
    output_path="path/to/metadataV0.csv"
)
```

或命令行运行：

```bash
python utils/compress_metadata.py
```

---

## split_dataset.py - 数据集划分脚本

### 功能描述

将数据集按 8:1:1 的比例划分为训练集、验证集和测试集，使用分层抽样确保 Branch 分布的均匀性。

### 处理流程

```
metadataV0.csv (97 行)
    ↓
对每个 Branch 进行分层抽样:
  ├── 多样本分支 (≥3 个样本) → 8:1:1 划分
  ├── 双样本分支 (=2 个样本) → 1 训练 +1 验证
  └── 单样本分支 (=1 个样本) → 全部放入训练集
    ↓
metadata_split.csv (97 行，新增 Split 列)
```

### 详细处理逻辑

#### 分层抽样策略

```python
# 根据样本数量分类处理
branch_counts = df['Branch'].value_counts()

single_sample_branches = branch_counts[branch_counts == 1].index.tolist()
double_sample_branches = branch_counts[branch_counts == 2].index.tolist()
multi_sample_branches = branch_counts[branch_counts >= 3].index.tolist()

# 多样本分支：8:1:1 划分
for branch in multi_sample_branches:
    n_samples = len(branch_indices)
    n_val = max(1, round(n_samples * 0.1))
    n_test = max(1, round(n_samples * 0.1))
    n_train = n_samples - n_val - n_test
    np.random.shuffle(branch_indices)
    train_idx.extend(branch_indices[:n_train])
    val_idx.extend(branch_indices[n_train:n_train + n_val])
    test_idx.extend(branch_indices[n_train + n_val:])

# 双样本分支：1 训练 +1 验证
for branch in double_sample_branches:
    train_idx.extend(branch_indices[:1])
    val_idx.extend(branch_indices[1:2])

# 单样本分支：全部放入训练集
for branch in single_sample_branches:
    train_idx.extend(branch_indices)
```

#### 划分规则

| Branch 样本数 | 训练集 | 验证集 | 测试集 | 说明 |
|--------------|--------|--------|--------|------|
| ≥3 | ~80% | ~10% | ~10% | 分层抽样 |
| 2 | 1 | 1 | 0 | 平均分配 |
| 1 | 1 | 0 | 0 | 全部训练 |

### 输入输出

| 项目 | 格式 | 列 |
|------|------|-----|
| **输入** | CSV (.csv) | ID, Start Frame, End Frame, Plaque, Branch |
| **输出** | CSV (.csv) | ID, Start Frame, End Frame, Plaque, Branch, Split |

**注意：** 输出文件新增 `Split` 列，取值为 `train`/`val`/`test`。

### 输出字段说明

| 字段 | 说明 |
|------|------|
| ID | 患者唯一标识符 |
| Start Frame | 血管起始帧 |
| End Frame | 血管结束帧 |
| Plaque | 堵塞程度 |
| Branch | 血管分支 |
| Split | 数据集划分 (`train`/`val`/`test`) |

### 使用方法

```python
from utils.split_dataset import split_dataset

split_dataset(
    input_path="path/to/metadataV0.csv",
    output_path="path/to/metadata_split.csv",
    seed=42
)
```

或命令行运行：

```bash
python utils/split_dataset.py
```

### 示例输出

```
总样本数：96
训练集：77 样本 (80.2%)
验证集：9 样本 (9.4%)
测试集：10 样本 (10.4%)
划分比例：训练：验证：测试 = 7.7 : 0.9 : 1
```

---

## 完整处理流程

```bash
# 1. 数据清洗
python utils/clean_metadata.py
# 生成：metadata_cleaned.xlsx

# 2. 数据压缩
python utils/compress_metadata.py
# 生成：metadataV0.csv

# 3. 数据集划分
python utils/split_dataset.py
# 生成：metadata_split.csv
```

### 数据变化

| 阶段 | 文件 | 行数 | 唯一 ID | 列 |
|------|------|------|--------|-----|
| 原始 | metadata.xlsx | 471 | 111 | 6 |
| 清洗后 | metadata_cleaned.xlsx | 288 | 97 | 6 |
| 压缩后 | metadataV0.csv | 96 | 96 | 5 |
| 划分后 | metadata_split.csv | 96 | 96 | 6 |

### 划分结果示例

| Split | 样本数 | 比例 |
|-------|--------|------|
| train | 77 | 80.2% |
| val | 9 | 9.4% |
| test | 10 | 10.4% |

---

## 依赖

```python
pandas >= 2.0
numpy >= 1.20
```

---

## 注意事项

1. **列名标准化**：`clean_metadata.py` 会将 `End frame` 统一改为 `End Frame`
2. **ID 填充**：空白 ID 使用上方最近的非空 ID 填充
3. **Branch 类型**：有效血管类型为 RCA（右冠状动脉）、LAD（左前降支）、LCX（左回旋支）
4. **输出格式**：最终输出为 CSV 格式，便于深度学习框架读取
