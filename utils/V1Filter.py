"""
Data Cleaning Script: Process metadata_cleaned.xlsx

Features:
1. Remove Type, Start Frame, End Frame columns
2. Keep rows with Branch column in RCA, LAD, LCX
3. Unify LM/LM-LAD/L Main to LM
4. Remove rows with other Branch values
5. Split data into train/val/test (8:1:1) with stratified sampling
6. Output metadataV1.csv with statistics

Supports both local and remote server execution
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

# Load configuration
CONFIG_FILE = Path(__file__).parent.parent / "remote_config.json"

def load_config():
    """Load configuration file"""
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_paths(remote=False):
    """
    Get file paths from configuration

    Args:
        remote: Whether to use remote paths

    Returns:
        Tuple of (input_file, output_file)
    """
    config = load_config()

    if remote:
        base_dir = config['paths']['remote_base']
    else:
        base_dir = config['paths']['local_base']

    input_file = os.path.join(base_dir, "metadata_cleaned.xlsx")
    output_file = os.path.join(base_dir, "metadataV1.csv")

    return input_file, output_file


def stratified_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Stratified split by Branch and Plaque distribution

    Ensures:
    1. All entries of the same ID appear in only one split
    2. Branch distribution is as uniform as possible
    3. Plaque value distribution is as uniform as possible

    Args:
        df: Input DataFrame
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        DataFrame with added 'Split' column
    """
    np.random.seed(seed)

    # 获取所有唯一 ID
    unique_ids = df['ID'].unique().tolist()

    # 为每个 ID 计算特征用于分层抽样
    id_features = {}
    for id_val in unique_ids:
        id_data = df[df['ID'] == id_val]
        branches = id_data['Branch'].tolist()
        plaques = id_data['Plaque'].dropna().tolist()

        # 主要分支特征
        branch_mode = max(set(branches), key=branches.count) if branches else 'Unknown'
        # 斑块中位数作为代表值
        plaque_median = np.median(plaques) if plaques else 0

        # 使用更细粒度的分箱
        if plaque_median < 40:
            plaque_bin = 'Low'
        elif plaque_median < 55:
            plaque_bin = 'Medium'
        elif plaque_median < 70:
            plaque_bin = 'High'
        else:
            plaque_bin = 'VeryHigh'

        id_features[id_val] = {
            'branch': branch_mode,
            'plaque_bin': plaque_bin
        }

    # 按 branch 和 plaque_bin 分组
    groups = defaultdict(list)
    for id_val in unique_ids:
        key = (id_features[id_val]['branch'], id_features[id_val]['plaque_bin'])
        groups[key].append(id_val)

    # 计算目标 ID 数量
    total_ids = len(unique_ids)
    target_train = int(total_ids * train_ratio)
    target_val = int(total_ids * val_ratio)
    target_test = total_ids - target_train - target_val

    # 对每个组进行分配
    train_ids = []
    val_ids = []
    test_ids = []

    for key, id_list in groups.items():
        np.random.shuffle(id_list)
        n = len(id_list)

        # 计算该组应该分配的数量
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))

        if n == 1:
            train_ids.extend(id_list)
        elif n == 2:
            train_ids.extend(id_list[:1])
            val_ids.extend(id_list[1:])
        else:
            train_ids.extend(id_list[:n_train])
            remaining = id_list[n_train:]
            n_val_actual = max(1, len(remaining) // 2)
            val_ids.extend(remaining[:n_val_actual])
            test_ids.extend(remaining[n_val_actual:])

    # 调整到目标数量
    # 如果训练集太多，移动一些到验证集
    while len(train_ids) > target_train and len(val_ids) < target_val:
        # 从最大的组移动
        train_groups = defaultdict(list)
        for id_val in train_ids:
            key = (id_features[id_val]['branch'], id_features[id_val]['plaque_bin'])
            train_groups[key].append(id_val)

        # 找到最大的组并移动一个
        max_group = max(train_groups.items(), key=lambda x: len(x[1]))
        id_to_move = max_group[1][-1]
        train_ids.remove(id_to_move)
        val_ids.append(id_to_move)

    # 如果验证集太多，移动一些到测试集
    while len(val_ids) > target_val and len(test_ids) < target_test:
        id_to_move = val_ids[-1]
        val_ids.remove(id_to_move)
        test_ids.append(id_to_move)

    # 如果训练集还不够，从验证集移动
    while len(train_ids) < target_train and len(val_ids) > 0:
        id_to_move = val_ids[0]
        val_ids.remove(id_to_move)
        train_ids.append(id_to_move)

    # 创建 ID 到 split 的映射
    id_to_split = {}
    for id_val in train_ids:
        id_to_split[id_val] = 'Train'
    for id_val in val_ids:
        id_to_split[id_val] = 'Val'
    for id_val in test_ids:
        id_to_split[id_val] = 'Test'

    # 为原始数据添加 Split 列
    df_result = df.copy()
    df_result['Split'] = df_result['ID'].map(id_to_split)

    return df_result


def clean_and_split(input_file, output_file):
    """
    Execute data cleaning and splitting

    Args:
        input_file: Input Excel file path
        output_file: Output CSV file path
    """
    print("=" * 60)
    print("Data Cleaning Script")
    print("=" * 60)
    print(f"\nReading file: {input_file}")
    df = pd.read_excel(input_file, engine='openpyxl')

    print(f"\nOriginal Data Statistics:")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Total columns: {len(df.columns)}")
    print(f"  - Columns: {df.columns.tolist()}")
    print(f"\nBranch column unique values:")
    print(df['Branch'].value_counts(dropna=False))

    # Step 1: Remove Type, Start Frame, End Frame columns
    print("\n" + "-" * 60)
    print("Step 1: Remove Type, Start Frame, End Frame columns")
    df_cleaned = df.drop(columns=['Type', 'Start Frame', 'End Frame'])
    print(f"  Remaining columns: {df_cleaned.columns.tolist()}")

    # Step 2: Filter Branch column
    print("\n" + "-" * 60)
    print("Step 2: Filter Branch column")

    # Values to keep
    keep_values = ['RCA', 'LAD', 'LCX']

    # Values to unify to LM
    lm_values = ['LM', 'LM-LAD', 'L Main']

    # Unify LM/LM-LAD/L Main to LM
    print(f"  Unifying {lm_values} to 'LM'")
    df_cleaned['Branch'] = df_cleaned['Branch'].apply(
        lambda x: 'LM' if x in lm_values else x
    )

    # Keep rows with Branch in RCA, LAD, LCX, LM
    print(f"  Keeping rows with Branch in {keep_values + ['LM']}")
    df_filtered = df_cleaned[df_cleaned['Branch'].isin(keep_values + ['LM'])]

    print(f"\n  Rows after filtering: {len(df_filtered)} (Original: {len(df)})")
    print(f"  Rows removed: {len(df) - len(df_filtered)}")

    # Save to CSV
    print("\n" + "-" * 60)
    print(f"Saving file: {output_file}")
    df_filtered.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("  File saved successfully!")

    # ============================================================
    # Step 3: Split into train/val/test (8:1:1)
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 3: Split into train/val/test (8:1:1)")
    print("=" * 60)

    # Execute stratified split
    print("Executing stratified split...")
    df_with_split = stratified_split(df_filtered)

    # Split statistics
    print("\nSplit Statistics:")
    split_counts = df_with_split['Split'].value_counts()
    for split, count in split_counts.items():
        percentage = count / len(df_with_split) * 100
        print(f"  - {split}: {count} rows ({percentage:.2f}%)")

    print("\nBranch x Split Crosstab:")
    print(pd.crosstab(df_with_split['Branch'], df_with_split['Split']))

    print("\nPlaque Distribution by Split:")
    for split in ['Train', 'Val', 'Test']:
        split_data = df_with_split[df_with_split['Split'] == split]['Plaque'].dropna()
        print(f"  {split}: median={split_data.median():.1f}, mean={split_data.mean():.1f}, std={split_data.std():.1f}")

    print("\nVerifying each ID appears in only one split:")
    id_splits = df_with_split.groupby('ID')['Split'].nunique()
    violations = id_splits[id_splits > 1]
    if len(violations) > 0:
        print(f"  WARNING: {len(violations)} IDs appear in multiple splits!")
    else:
        print("  VERIFIED: All IDs appear in only one split")

    # Update output file
    print("\n" + "-" * 60)
    print(f"Updating file: {output_file}")
    df_with_split.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("  File saved successfully!")

    # Output statistics
    print("\n" + "=" * 60)
    print("Data Statistics After Cleaning")
    print("=" * 60)
    print(f"\nBasic Info:")
    print(f"  - Total rows: {len(df_with_split)}")
    print(f"  - Total columns: {len(df_with_split.columns)}")
    print(f"  - Columns: {df_with_split.columns.tolist()}")

    print(f"\nBranch Distribution:")
    branch_counts = df_with_split['Branch'].value_counts()
    for branch, count in branch_counts.items():
        percentage = count / len(df_with_split) * 100
        print(f"  - {branch}: {count} rows ({percentage:.2f}%)")

    print(f"\nMissing Values:")
    missing_counts = df_with_split.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  - {col}: {count} missing")
        else:
            print(f"  - {col}: No missing values")

    print(f"\nID Statistics:")
    print(f"  - Unique IDs: {df_with_split['ID'].nunique()}")
    print(f"  - ID Range: {df_with_split['ID'].min()} - {df_with_split['ID'].max()}")

    print(f"\nDataset Split Statistics:")
    print(f"  - Train: {len(df_with_split[df_with_split['Split']=='Train'])} rows")
    print(f"  - Val: {len(df_with_split[df_with_split['Split']=='Val'])} rows")
    print(f"  - Test: {len(df_with_split[df_with_split['Split']=='Test'])} rows")

    # Show first 10 rows
    print(f"\nFirst 10 rows preview:")
    print(df_with_split.head(10).to_string())

    print("\n" + "=" * 60)
    print("Data Cleaning Complete!")
    print("=" * 60)

    return df_with_split


def main():
    """Main function"""
    # Check if running in remote mode
    remote_mode = '--remote' in sys.argv or '-r' in sys.argv

    # Auto-detect if not specified
    if not remote_mode:
        config = load_config()
        remote_base = config['paths']['remote_base']
        script_path = os.path.abspath(__file__)
        if remote_base in script_path:
            remote_mode = True

    input_file, output_file = get_paths(remote=remote_mode)

    print(f"Mode: {'Remote' if remote_mode else 'Local'}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    clean_and_split(input_file, output_file)


if __name__ == "__main__":
    main()
