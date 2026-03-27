"""
默认配置定义

包含模型训练和测试的所有超参数设置。
"""

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config 定义
# -----------------------------------------------------------------------------

CFG = CN()

# -----------------------------------------------------------------------------
# 数据相关配置
# -----------------------------------------------------------------------------
CFG.DATA = CN()
CFG.DATA.TRAIN_DIR = ""  # 训练数据目录
CFG.DATA.VAL_DIR = ""    # 验证数据目录
CFG.DATA.TEST_DIR = ""   # 测试数据目录
CFG.DATA.ORGAN = "breast"  # 器官类型：'breast' 或 'thyroid'
CFG.DATA.NUM_WORKERS = 4   # 数据加载线程数
CFG.DATA.PIN_MEMORY = True # 是否 pin memory

# 数据增强
CFG.DATA.INPUT_SIZE = 224   # 输入图像尺寸
CFG.DATA.NORMALIZE_MEAN = [0.485, 0.456, 0.406]
CFG.DATA.NORMALIZE_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# 模型相关配置
# -----------------------------------------------------------------------------
CFG.MODEL = CN()
CFG.MODEL.DEVICE = "cuda"       # 运行设备
CFG.MODEL.BACKBONE = "resnet50" # 骨干网络
CFG.MODEL.FEATURE_DIM = 2048    # 特征维度（ResNet50 为 2048）
CFG.MODEL.MODE = "HYBRID"       # 模式：'HYBRID' 或 '2D'
CFG.MODEL.WEIGHTS = ""          # 预训练权重路径

# -----------------------------------------------------------------------------
# 属性标签配置（甲状腺）
# -----------------------------------------------------------------------------
# 一级属性列表
CFG.ATTR_LIST = ["病理"]
# 一级属性类别数
CFG.ATTR_WEIGHT = [[1.0, 1.0]]  # 类别权重

# 多分类属性列表
CFG.ATTR_MULTI_LIST = []

# 二级属性列表
CFG.SECOND_ATTR_LIST = []
CFG.SECOND_ATTR_MULTI_LIST = []

# -----------------------------------------------------------------------------
# 属性标签配置（乳腺）
# -----------------------------------------------------------------------------
# 乳腺数据集属性定义
CFG.BREAST_ATTR_LIST = ["Pathology"]
CFG.BREAST_ATTR_WEIGHT = [[1.0, 1.0, 1.0, 1.0, 1.0]]  # 5 类
CFG.BREAST_SECOND_ATTR_MULTI_LIST = []

# -----------------------------------------------------------------------------
# 损失函数配置
# -----------------------------------------------------------------------------
CFG.LOSS = CN()
CFG.LOSS.CENTER_WEIGHT = 0.5   # 中心损失权重
CFG.LOSS.AUX_WEIGHT = 0.3      # 辅助损失权重
CFG.LOSS.LABEL_SMOOTHING = 0.0

# -----------------------------------------------------------------------------
# 优化器配置
# -----------------------------------------------------------------------------
CFG.OPTIMIZER = CN()
CFG.OPTIMIZER.TYPE = "SGD"
CFG.OPTIMIZER.BASE_LR = 0.001
CFG.OPTIMIZER.MOMENTUM = 0.9
CFG.OPTIMIZER.WEIGHT_DECAY = 1e-4
CFG.OPTIMIZER.WEIGHT_DECAY_NORM = 1e-4

# -----------------------------------------------------------------------------
# 学习率调度配置
# -----------------------------------------------------------------------------
CFG.LR_SCHEDULER = CN()
CFG.LR_SCHEDULER.TYPE = "StepLR"
CFG.LR_SCHEDULER.GAMMA = 0.1
CFG.LR_SCHEDULER.STEP_SIZE = 30
CFG.LR_SCHEDULER.MILESTONES = [30, 60, 90]

# -----------------------------------------------------------------------------
# 训练配置
# -----------------------------------------------------------------------------
CFG.TRAIN = CN()
CFG.TRAIN.BATCH_SIZE = 8
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.START_EPOCH = 0
CFG.TRAIN.RESUME = ""         # 恢复训练的检查点路径
CFG.TRAIN.OUTPUT_DIR = "output"
CFG.TRAIN.SAVE_FREQ = 10      # 保存检查点的频率
CFG.TRAIN.EVAL_FREQ = 1       # 评估频率
CFG.TRAIN.PRINT_FREQ = 10     # 打印日志频率
CFG.TRAIN.NUM_SAMPLES_PER_VIDEO = 16  # 每个视频的采样帧数

# -----------------------------------------------------------------------------
# 测试配置
# -----------------------------------------------------------------------------
CFG.TEST = CN()
CFG.TEST.BATCH_SIZE = 1
CFG.TEST.CHECKPOINT = ""      # 测试用的检查点路径
CFG.TEST.OUTPUT_DIR = "output_test"
CFG.TEST.SAVE_RESULTS = True  # 是否保存测试结果

# -----------------------------------------------------------------------------
# 分布式训练配置
# -----------------------------------------------------------------------------
CFG.DISTRIBUTED = CN()
CFG.DISTRIBUTED.NUM_GPUS = 1
CFG.DISTRIBUTED.DIST_URL = "env://"

# -----------------------------------------------------------------------------
# 其他配置
# -----------------------------------------------------------------------------
CFG.ORGAN = "breast"  # 兼容旧代码


def get_cfg_defaults():
    """获取默认配置的副本"""
    return CFG.clone()


def update_config(cfg, config_file):
    """从文件更新配置"""
    if config_file:
        cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg
