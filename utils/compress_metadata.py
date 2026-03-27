"""
冠脉 CT 视频回归模型数据集描述文件生成脚本

将每个 ID 的多行数据压缩为 1 行，规则如下：
1. 对于同一个 ID，找到 Plaque 列的最大值
2. 找到该最大值对应的 Branch
3. 如果 Branch 在 Type 中（RCA/LAD/LCX），使用该 Type 行的 Start Frame 和 End Frame
4. 如果 Branch 不在 Type 中（如 CX、L Main、Vessel 等），使用所有行中 Start Frame 的最小值和 End Frame 的最大值
5. 每个 ID 压缩为 1 条记录，删除 Type 列
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compress_metadata(input_path: str, output_path: str):
    """
    将 metadata 数据按 ID 压缩，每个 ID 保留 1 行

    Args:
        input_path: 输入 Excel 文件路径
        output_path: 输出 CSV 文件路径
    """
    # 读取数据
    print(f"读取数据：{input_path}")
    df = pd.read_excel(input_path)
    print(f"原始数据形状：{df.shape}")
    print(f"唯一 ID 数量：{df['ID'].nunique()}")

    # 目标血管类型
    VALID_TYPES = {'RCA', 'LAD', 'LCX'}

    # 存储处理后的结果
    results = []

    # 按 ID 分组处理
    grouped = df.groupby('ID')
    print(f"\n开始处理 {len(grouped)} 个 ID...")

    for id_val, group in grouped:
        # 找到 Plaque 的最大值（忽略 NaN）
        plaque_max = group['Plaque'].max()

        if pd.isna(plaque_max):
            # 如果所有 Plaque 都是 NaN，跳过该 ID
            print(f"  跳过 ID {id_val}: 所有 Plaque 值为空")
            continue

        # 找到 Plaque 最大值对应的行（可能有多个，取第一个）
        max_plaque_rows = group[group['Plaque'] == plaque_max]
        target_branch = max_plaque_rows.iloc[0]['Branch']

        # 检查 Branch 是否在有效的 Type 中
        if target_branch in VALID_TYPES:
            # Branch 在 Type 中，查找该 Type 对应的行
            type_rows = group[group['Type'] == target_branch]
            if len(type_rows) > 0:
                # 找到该行（第一个匹配的）
                target_row = type_rows.iloc[0]
                start_frame = target_row['Start Frame']
                end_frame = target_row['End Frame']
            else:
                # 没有找到对应的 Type 行，使用最大范围
                start_frame = group['Start Frame'].min()
                end_frame = group['End Frame'].max()
        else:
            # Branch 不在 Type 中，使用最大范围
            start_frame = group['Start Frame'].min()
            end_frame = group['End Frame'].max()

        # 创建压缩后的记录（不包含 Type 列）
        record = {
            'ID': int(id_val),
            'Start Frame': int(start_frame) if pd.notna(start_frame) else None,
            'End Frame': int(end_frame) if pd.notna(end_frame) else None,
            'Plaque': plaque_max,
            'Branch': target_branch,
        }
        results.append(record)

    # 创建结果 DataFrame
    result_df = pd.DataFrame(results)

    # 保存结果
    result_df.to_csv(output_path, index=False)
    print(f"\n处理完成！结果已保存到：{output_path}")

    # 输出统计信息
    print("\n" + "="*60)
    print("处理结果统计")
    print("="*60)
    print(f"原始数据行数：{len(df)}")
    print(f"压缩后数据行数：{len(result_df)}")
    print(f"原始唯一 ID 数：{df['ID'].nunique()}")
    print(f"有效样本数（有 Plaque 值）：{len(result_df)}")
    print()
    print("Branch 分布:")
    print(result_df['Branch'].value_counts())
    print()
    print("Plaque 统计:")
    print(result_df['Plaque'].describe())
    print()
    print("Start Frame - End Frame 范围统计:")
    result_df['Range'] = result_df['End Frame'] - result_df['Start Frame']
    print(result_df['Range'].describe())
    print()
    print("数据预览:")
    print(result_df.head(20).to_string())

    return result_df


if __name__ == "__main__":
    # 文件路径
    input_file = r"/18018998051/CTA/metadata_cleaned.xlsx"
    output_file = r"/18018998051/CTA/metadataV0.csv"

    # 执行压缩
    result_df = compress_metadata(input_file, output_file)

    print("\n" + "="*60)
    print("最终结果")
    print("="*60)
    print(f"最终获得的合法样本数量：{len(result_df)}")
