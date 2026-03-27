"""
DICOM 文件名标准化处理 Pipeline

用于冠脉 CT 数据集的 DICOM 文件名标准化处理，包含两个阶段：
1. 第一阶段：处理符合 xxx_xxx.dcm 格式的文件，重命名为 xxx.dcm
2. 第二阶段：处理不符合格式的文件，应用备用命名规则

作者：Automated Pipeline
日期：2026-03-19
"""

import pandas as pd
import os
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime

# ============================================================================
# 日志配置
# ============================================================================

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"dcm_rename_{timestamp}.log")

    logger = logging.getLogger("dcm_rename")
    logger.setLevel(logging.DEBUG)

    # 文件处理器（详细日志）
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # 控制台处理器（简要日志）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


# ============================================================================
# 配置类
# ============================================================================

class RenameConfig:
    """重命名配置"""

    def __init__(
        self,
        metadata_path: str,
        data_root: str,
        subfolder: str = "CTA",
        phase1_pattern: str = r'^(\d{2,3})_(\d{2,3})\.dcm$',
        min_digits_for_rule1: int = 4,
        digits_to_keep: int = 3,
    ):
        """
        初始化配置

        Args:
            metadata_path: metadata_cleaned.xlsx 路径
            data_root: 数据根目录
            subfolder: 数据子文件夹名称（默认 CTA）
            phase1_pattern: 第一阶段文件名匹配模式
            min_digits_for_rule1: 规则 1 所需的最小连续数字位数
            digits_to_keep: 规则 1 截取的数字位数
        """
        self.metadata_path = metadata_path
        self.data_root = data_root
        self.subfolder = subfolder
        self.phase1_pattern = re.compile(phase1_pattern, re.IGNORECASE)
        self.min_digits_for_rule1 = min_digits_for_rule1
        self.digits_to_keep = digits_to_keep

        # 规则 1：匹配末尾连续数字
        self.digit_pattern = re.compile(
            r'(\d{' + str(min_digits_for_rule1) + r',})\.dcm$',
            re.IGNORECASE
        )


# ============================================================================
# 处理结果类
# ============================================================================

class ProcessingResult:
    """处理结果统计"""

    def __init__(self):
        self.phase1_renamed = 0
        self.phase1_skipped = 0
        self.phase2_renamed = 0
        self.phase2_skipped = 0
        self.errors: List[Tuple[str, str, str]] = []  # (id, filename, error)
        self.missing_folders: List[str] = []
        self.non_conforming_ids_phase1: Dict[str, List[str]] = {}

    def add_error(self, id_str: str, filename: str, error: str):
        """添加错误记录"""
        self.errors.append((id_str, filename, error))

    def summary(self) -> str:
        """生成摘要报告"""
        lines = [
            "=" * 60,
            "DICOM 文件名标准化处理报告",
            "=" * 60,
            "",
            "第一阶段 (xxx_xxx.dcm -> xxx.dcm):",
            f"  - 重命名文件数：{self.phase1_renamed}",
            f"  - 跳过文件数：{self.phase1_skipped}",
            f"  - 不符合格式 ID 数：{len(self.non_conforming_ids_phase1)}",
            "",
            "第二阶段 (备用规则处理):",
            f"  - 重命名文件数：{self.phase2_renamed}",
            f"  - 跳过文件数：{self.phase2_skipped}",
            "",
            f"错误文件总数：{len(self.errors)}",
            f"缺失文件夹数：{len(self.missing_folders)}",
            "",
        ]

        if self.non_conforming_ids_phase1:
            lines.append("不符合第一阶段格式的 ID:")
            for id_str in list(self.non_conforming_ids_phase1.keys())[:10]:
                files = self.non_conforming_ids_phase1[id_str]
                lines.append(f"  - {id_str}: {len(files)} 个文件")
            if len(self.non_conforming_ids_phase1) > 10:
                lines.append(f"  ... 还有 {len(self.non_conforming_ids_phase1) - 10} 个 ID")
            lines.append("")

        if self.errors:
            lines.append("错误文件示例:")
            for id_str, filename, error in self.errors[:5]:
                lines.append(f"  - [{id_str}] {filename}: {error}")
            if len(self.errors) > 5:
                lines.append(f"  ... 还有 {len(self.errors) - 5} 个错误")
            lines.append("")

        if self.missing_folders:
            lines.append(f"缺失文件夹的 ID: {self.missing_folders}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# 第一阶段处理
# ============================================================================

def get_unique_ids(metadata_path: str) -> List[str]:
    """从 metadata 文件获取唯一 ID 列表"""
    df = pd.read_excel(metadata_path)
    unique_ids = df['ID'].unique()
    return [str(int(id_val)) for id_val in unique_ids]


def process_phase1(config: RenameConfig, result: ProcessingResult, logger: logging.Logger) -> List[str]:
    """
    第一阶段：处理符合 xxx_xxx.dcm 格式的文件

    规则：将 xxx_yyy.dcm 重命名为 xxx.dcm
    """
    logger.info("=" * 60)
    logger.info("第一阶段：处理 xxx_xxx.dcm 格式文件")
    logger.info("=" * 60)

    unique_ids = get_unique_ids(config.metadata_path)
    processed_ids = []

    for id_str in unique_ids:
        folder_path = os.path.join(config.data_root, config.subfolder, id_str)

        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            result.missing_folders.append(id_str)
            logger.debug(f"ID {id_str}: 文件夹不存在，跳过")
            continue

        files = [f for f in os.listdir(folder_path) if f.lower().endswith('.dcm')]

        if len(files) == 0:
            result.missing_folders.append(id_str)
            logger.debug(f"ID {id_str}: 文件夹为空，跳过")
            continue

        # 分类文件
        non_conforming_files = []
        conforming_files = []

        for f in files:
            if config.phase1_pattern.match(f):
                conforming_files.append(f)
            else:
                non_conforming_files.append(f)

        # 记录不符合格式的文件
        if non_conforming_files:
            result.non_conforming_ids_phase1[id_str] = non_conforming_files

        # 重命名符合格式的文件
        for f in conforming_files:
            match = config.phase1_pattern.match(f)
            if match:
                prefix = match.group(1)
                new_name = f"{prefix}.dcm"
                old_path = os.path.join(folder_path, f)
                new_path = os.path.join(folder_path, new_name)

                # 检查文件名冲突
                if os.path.exists(new_path):
                    result.phase1_skipped += 1
                    logger.debug(f"跳过：{f} -> {new_name} (目标已存在)")
                    continue

                try:
                    os.rename(old_path, new_path)
                    result.phase1_renamed += 1
                    logger.debug(f"重命名：{f} -> {new_name}")
                except Exception as e:
                    result.add_error(id_str, f, str(e))
                    logger.error(f"重命名失败 {f}: {e}")

        processed_ids.append(id_str)
        logger.info(f"ID {id_str}: 处理完成 ({len(conforming_files)} 符合，{len(non_conforming_files)} 不符合)")

    return processed_ids


# ============================================================================
# 第二阶段处理
# ============================================================================

def generate_new_name(
    filename: str,
    config: RenameConfig
) -> Optional[str]:
    """
    根据备用规则生成新文件名

    规则 1: 如果扩展名末尾有不少于 N 位的连续数字，截取最后 K 位
    规则 2: 否则，如果有小数点，只保留最后一个小数点后的部分
    规则 3: 否则，提取文件茎末尾的数字部分

    Args:
        filename: 原文件名
        config: 配置对象

    Returns:
        新文件名，如果无法生成则返回 None
    """
    file_stem = filename[:-4]  # 去掉 .dcm

    # 规则 1：检查末尾是否有不少于 N 位连续数字
    digit_match = config.digit_pattern.search(filename)
    if digit_match:
        last_digits = digit_match.group(1)
        new_name = f"{last_digits[-config.digits_to_keep:]}.dcm"
        return new_name

    # 规则 2：有小数点，保留最后一个小数点后的部分
    elif '.' in file_stem:
        last_part = file_stem.split('.')[-1]
        new_name = f"{last_part}.dcm"
        return new_name

    # 规则 3：提取文件茎末尾的连续数字部分
    else:
        stem_digit_match = re.search(r'(\d+)(?!.*\d)', file_stem)
        if stem_digit_match:
            last_num = stem_digit_match.group(1)
            new_name = f"{last_num}.dcm"
            return new_name

    return None


def process_phase2(config: RenameConfig, result: ProcessingResult, logger: logging.Logger):
    """
    第二阶段：处理不符合格式的文件

    使用备用命名规则重命名剩余的非标准文件名
    """
    logger.info("=" * 60)
    logger.info("第二阶段：处理不符合格式的文件")
    logger.info("=" * 60)

    non_conforming_ids = list(result.non_conforming_ids_phase1.keys())

    if not non_conforming_ids:
        logger.info("没有需要处理的文件，跳过第二阶段")
        return

    for id_str in non_conforming_ids:
        folder_path = os.path.join(config.data_root, config.subfolder, id_str)

        if not os.path.exists(folder_path):
            logger.warning(f"ID {id_str}: 文件夹已不存在，跳过")
            continue

        files = [f for f in os.listdir(folder_path) if f.lower().endswith('.dcm')]

        # 跳过已经是 xxx.dcm 格式的文件
        simple_pattern = re.compile(r'^\d+\.dcm$', re.IGNORECASE)
        files_to_process = [f for f in files if not simple_pattern.match(f)]

        if not files_to_process:
            logger.info(f"ID {id_str}: 无需处理")
            continue

        logger.info(f"处理 ID: {id_str} (共 {len(files_to_process)} 个文件)")

        for f in files_to_process:
            new_name = generate_new_name(f, config)

            if not new_name:
                result.add_error(id_str, f, "无法生成新文件名")
                logger.error(f"无法生成新文件名：{f}")
                continue

            old_path = os.path.join(folder_path, f)
            new_path = os.path.join(folder_path, new_name)

            # 处理文件名冲突
            if os.path.exists(new_path):
                base, ext = os.path.splitext(new_name)
                counter = 1
                while os.path.exists(new_path):
                    new_name = f"{base}_{counter}{ext}"
                    new_path = os.path.join(folder_path, new_name)
                    counter += 1
                logger.debug(f"文件名冲突，使用：{new_name}")

            try:
                os.rename(old_path, new_path)
                result.phase2_renamed += 1
                logger.debug(f"重命名：{f} -> {new_name}")
            except Exception as e:
                result.add_error(id_str, f, str(e))
                logger.error(f"重命名失败 {f}: {e}")

        logger.info(f"ID {id_str}: 处理完成")


# ============================================================================
# 验证函数
# ============================================================================

def verify_results(config: RenameConfig, result: ProcessingResult, logger: logging.Logger):
    """验证处理结果"""
    logger.info("=" * 60)
    logger.info("验证处理结果")
    logger.info("=" * 60)

    unique_ids = get_unique_ids(config.metadata_path)
    target_pattern = re.compile(r'^\d+\.dcm$', re.IGNORECASE)

    all_valid = True

    for id_str in unique_ids:
        folder_path = os.path.join(config.data_root, config.subfolder, id_str)

        if not os.path.exists(folder_path):
            continue

        files = [f for f in os.listdir(folder_path) if f.lower().endswith('.dcm')]
        non_target = [f for f in files if not target_pattern.match(f)]

        if non_target:
            logger.warning(f"ID {id_str}: 有 {len(non_target)} 个文件不符合目标格式")
            all_valid = False

    if all_valid:
        logger.info("所有文件都符合目标格式 xxx.dcm")
    else:
        logger.warning("部分文件仍不符合目标格式")

    return all_valid


# ============================================================================
# 主函数
# ============================================================================

def run_pipeline(
    metadata_path: str,
    data_root: str,
    subfolder: str = "CTA",
    log_to_console: bool = True
) -> ProcessingResult:
    """
    运行完整的 DICOM 文件名标准化 pipeline

    Args:
        metadata_path: metadata_cleaned.xlsx 路径
        data_root: 数据根目录
        subfolder: 数据子文件夹名称
        log_to_console: 是否输出到控制台

    Returns:
        ProcessingResult: 处理结果统计
    """
    # 创建配置
    config = RenameConfig(
        metadata_path=metadata_path,
        data_root=data_root,
        subfolder=subfolder,
    )

    # 设置日志
    logger, log_file = setup_logging()

    if not log_to_console:
        logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]

    # 创建结果对象
    result = ProcessingResult()

    # 打印开始信息
    print("=" * 60)
    print("DICOM 文件名标准化处理 Pipeline")
    print("=" * 60)
    print(f"元数据文件：{metadata_path}")
    print(f"数据目录：{os.path.join(data_root, subfolder)}")
    print(f"日志文件：{log_file}")
    print("=" * 60)
    print()

    # 第一阶段
    process_phase1(config, result, logger)

    # 第二阶段
    process_phase2(config, result, logger)

    # 验证结果
    verify_results(config, result, logger)

    # 打印摘要
    print()
    print(result.summary())

    return result


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DICOM 文件名标准化处理 Pipeline"
    )
    parser.add_argument(
        "--metadata", "-m",
        default="E:\pycharm23\Projs\DcmDataset\metadata_cleaned.xlsx",
        help="metadata_cleaned.xlsx 文件路径"
    )
    parser.add_argument(
        "--data-root", "-d",
        default="E:\dataset\Central",
        help="数据根目录"
    )
    parser.add_argument(
        "--subfolder", "-s",
        default="CTA",
        help="数据子文件夹名称"
    )

    args = parser.parse_args()

    run_pipeline(
        metadata_path=args.metadata,
        data_root=args.data_root,
        subfolder=args.subfolder,
    )
