"""
冠脉 CT 数据集划分脚本

将数据集按 8:1:1 的比例划分为训练集、验证集和测试集，确保 Branch 分布的均匀性。
使用分层抽样保证各血管分支在三个集合中的分布与原始数据一致。
"""

import pandas as pd
import numpy as np


def split_dataset(input_path: str, output_path: str, seed: int = 42):
    """
    将数据集按 8:1:1 比例划分（训练：验证：测试），确保 Branch 分层采样

    Args:
        input_path: 输入 CSV 文件路径
        output_path: 输出 CSV 文件路径
        seed: 随机种子
    """
    # 读取数据
    print(f"读取数据：{input_path}")
    df = pd.read_csv(input_path)
    print(f"原始数据形状：{df.shape}")
    print(f"唯一样本数：{len(df)}")

    # 原始 Branch 分布
    print("\n原始 Branch 分布:")
    original_dist = df['Branch'].value_counts()
    print(original_dist)

    # 检查需要特殊处理的分支（样本数少于 3 的类别）
    branch_counts = df['Branch'].value_counts()

    # 单样本和双样本分支（需要特殊处理）
    single_sample_branches = branch_counts[branch_counts == 1].index.tolist()
    double_sample_branches = branch_counts[branch_counts == 2].index.tolist()
    multi_sample_branches = branch_counts[branch_counts >= 3].index.tolist()

    print(f"\n单样本 Branch (将全部放入训练集): {single_sample_branches}")
    print(f"双样本 Branch (2 个放入训练，1 个放入验证): {double_sample_branches}")
    print(f"多样本 Branch (进行 8:1:1 分层抽样): {multi_sample_branches}")

    # ============================================================
    # 执行 8:1:1 划分
    # ============================================================
    print("\n" + "="*60)
    print("生成数据集划分 (8:1:1)")
    print("="*60)

    np.random.seed(seed)

    train_idx = []
    val_idx = []
    test_idx = []

    # 处理多样本分支（进行 8:1:1 分层抽样）
    for branch in multi_sample_branches:
        branch_indices = df[df['Branch'] == branch].index.tolist()
        n_samples = len(branch_indices)

        # 计算验证集和测试集大小（各约 10%）
        n_val = max(1, round(n_samples * 0.1))
        n_test = max(1, round(n_samples * 0.1))
        n_train = n_samples - n_val - n_test

        # 确保至少有 80% 在训练集
        if n_train < n_samples * 0.8:
            n_train = int(n_samples * 0.8)
            remaining = n_samples - n_train
            n_val = remaining // 2
            n_test = remaining - n_val

        np.random.shuffle(branch_indices)
        train_idx.extend(branch_indices[:n_train])
        val_idx.extend(branch_indices[n_train:n_train + n_val])
        test_idx.extend(branch_indices[n_train + n_val:])

    # 处理双样本分支（2 个训练，0 个验证，0 个测试；或 1 训练 1 验证）
    for branch in double_sample_branches:
        branch_indices = df[df['Branch'] == branch].index.tolist()
        np.random.shuffle(branch_indices)
        train_idx.extend(branch_indices[:1])
        val_idx.extend(branch_indices[1:2])  # 另一个放入验证集

    # 处理单样本分支（全部放入训练集）
    for branch in single_sample_branches:
        branch_indices = df[df['Branch'] == branch].index.tolist()
        train_idx.extend(branch_indices)

    # 创建 Split 列
    df['Split'] = 'train'
    df.loc[val_idx, 'Split'] = 'val'
    df.loc[test_idx, 'Split'] = 'test'

    # ============================================================
    # 统计划分结果
    # ============================================================
    print(f"\n训练集大小：{len(train_idx)} ({len(train_idx)/len(df)*100:.1f}%)")
    print(f"验证集大小：{len(val_idx)} ({len(val_idx)/len(df)*100:.1f}%)")
    print(f"测试集大小：{len(test_idx)} ({len(test_idx)/len(df)*100:.1f}%)")
    print(f"划分比例：训练：验证：测试 = {len(train_idx)/len(test_idx):.1f} : {len(val_idx)/len(test_idx):.1f} : 1")

    print("\n训练集 Branch 分布:")
    train_dist = df[df['Split'] == 'train']['Branch'].value_counts().sort_index()
    print(train_dist)

    print("\n验证集 Branch 分布:")
    val_dist = df[df['Split'] == 'val']['Branch'].value_counts().sort_index()
    print(val_dist)

    print("\n测试集 Branch 分布:")
    test_dist = df[df['Split'] == 'test']['Branch'].value_counts().sort_index()
    print(test_dist)

    # ============================================================
    # 保存结果
    # ============================================================
    print("\n" + "="*60)
    print("保存结果")
    print("="*60)

    df.to_csv(output_path, index=False)
    print(f"\n处理完成！结果已保存到：{output_path}")

    # ============================================================
    # 输出最终统计
    # ============================================================
    print("\n" + "="*60)
    print("最终统计")
    print("="*60)

    print(f"\n总样本数：{len(df)}")
    print(f"训练集：{len(train_idx)} 样本")
    print(f"验证集：{len(val_idx)} 样本")
    print(f"测试集：{len(test_idx)} 样本")

    # 每个 Branch 的划分统计
    print("\n各 Branch 划分详情:")
    for branch in df['Branch'].unique():
        branch_data = df[df['Branch'] == branch]
        train_count = (branch_data['Split'] == 'train').sum()
        val_count = (branch_data['Split'] == 'val').sum()
        test_count = (branch_data['Split'] == 'test').sum()
        print(f"  {branch}: 训练={train_count}, 验证={val_count}, 测试={test_count}")

    print("\n数据预览 (前 20 行):")
    print(df[['ID', 'Branch', 'Plaque', 'Split']].head(20).to_string())

    return df


if __name__ == "__main__":
    # 文件路径
    input_file = r"/18018998051/CTA/metadataV0.csv"
    output_file = r"/18018998051/data/Central/metadata_split.csv"

    # 执行划分 (使用固定种子保证可复现性)
    result_df = split_dataset(input_file, output_file, seed=42)

    print("\n" + "="*60)
    print("处理完成")
    print("="*60)
