# 数据预处理
IMG_PATH = r"G:\灯检机数据\图片保存\工位1\2023_12_19\相机1\OK\原图"
IMG_HEIGHT = 512
IMG_WIDTH = 512

# 随机性与数据集划分
SEED = 42
TRAIN_RATIO = 0.8 # 增加训练集比例， 从0.75至0.8
TEST_RATIO = 0.2  # 相应调整测试集比例
NOISE_RATIO = 0.3 # 降低噪声因子，从0.5改为0.3 【高分辨率图像的噪声更明显】


# 训练超参数配置

LEARNING_RATE = 0.0001 # 降低学习率，从0.001改为0.0001【因为大图像需要更细致的学习】
EPOCHS = 50 # 增加训练轮次，从30提高至50
TRAIN_BATCH_SIZE = 4 # 大幅度减小批次大小，从32改为4 【目的是为了防止内存溢出】
TEST_BATCH_SIZE = 8  # 测试批次可以稍大一些

# 模型配置
MODEL_CHANNEL = [64,128,256,512] # U-Net各层通道数
USE_BATCH_NORM = True   # 是否使用批归一化
DROPOUT_RATE = 0.1      # Dropout率

# 优化器配置

WEIGHT_DECAY = 1e-5 # L2正则化
SCHEDULER_PAIENCE = 5 # 学习率调度器耐心值
SCHEDULER_FACTOR = 0.5 # 学习率衰减因子
# 损失函数配置
LOSS_FACTOR = 0.5   # L1损失权重
EDGE_LOSS_WEIGHT = 0.1 # 边缘损失权重

# 模块名称和保存模型参数的文件名
PROJECT_PACKAGE_NAME = "image_denoise_hd" # 修改项目名称
DENOISER_MODEL_NAME = "denoising_model_512.pt" # 修改模型文件名称

# 内存优化配置

GRADIENT_ACCUMULATION_STEPS = 4 # 梯度累积步数[模拟更大的批次]
MIXED_PRECISION = True  # 是否使用混合精度训练
NUM_WORKERS = 4    # DataLoader的工作进程数
PIN_MEMORY = True   # 是否固定内存【GPU训练时建议开启】

# 数据增强配置
USE_DATA_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    'rotation_degrees':10,  # 随机旋转角度
    'horizontal_flip_prob': 0.5, # 水平翻转概率
    'brightness_factor': 0.1, # 亮度调整范围
    'contrast_factor': 0.1, # 对比度调整范围
}

# 早停配置
EARLY_STOPPING_PATIENCE = 10    # 早停耐心值
EARLY_STOPPING_MIN_DELTA = 0.0001   # 最小改善阈值

# 日志和检查点配置
CHECKPOINT_DIR = "./checkpoints"   # 检查的保存目录
LOG_DIR = "./logs"  # 日志保存目录
SAVE_CHECKPOINT_EVERY = 5   # 每N个epoch保存一次检查点
