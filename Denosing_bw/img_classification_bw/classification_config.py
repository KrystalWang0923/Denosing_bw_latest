# classification_config.py
# 优化后的图像分类训练配置文件 - 支持数据分批处理
# 针对内存优化和稳定性改进

import platform
import multiprocessing as mp
import torch
import os


# 新增默认模型路径配置
DEFAULT_MODEL_PATH = "F:\conda3-test\PythonProject\Denosing_bw\img_classification_bw\models\run_20250721_124804_filter_0_split5\final_model_multi_task.pth"  # 替换为实际默认模型路径
# ===================== 基础训练参数（内存优化）=====================
IMAGE_SIZE = 128  # 降低图片尺寸以减少内存使用
BATCH_SIZE = 64  # 降低批次大小
NUM_EPOCHS = 35  # 训练轮数
LEARNING_RATE = 0.001  # 学习率
MAX_PATIENCE = 10  # 早停耐心值
WARMUP_EPOCHS = 2  # warmup训练

# ===================== 数据分批配置（新增）=====================
# 数据分批训练配置 - 解决内存不足问题
DATA_SPLIT_CONFIG = {
    'enabled': True,  # 是否启用数据分批
    'splits_per_epoch': 5,  # 每轮分成几批（可调整）
    'shuffle_splits': True,  # 是否在每轮开始时打乱分批
    'save_checkpoint_per_split': False,  # 是否每批都保存检查点
    'gc_collect_per_split': True,  # 每批后进行垃圾回收
}

# ===================== 内存优化配置（新增）=====================
MEMORY_CONFIG = {
    'max_memory_gb': 8,  # 最大使用内存（GB）
    'clear_cache_frequency': 50,  # 每N个批次清理一次缓存
    'use_memory_efficient_loader': True,  # 使用内存高效的数据加载器
    'preload_percentage': 0.1,  # 预加载数据的百分比
}

# ===================== 数据路径配置 =====================
CSV_PATH = r'F:\conda3-test\PythonProject\cuda126\test\image_map_generator_version_iter\map_result\CSV\images_工位6个_OK_NG_日期40个_相机3个_原图_20250704_104805.csv'
OUTPUT_DIR = r'F:\conda3-test\PythonProject\Denosing_bw\img_classification_bw\models'
CACHE_DIR = r'F:\conda3-test\PythonProject\Denosing_bw\img_classification_bw\cache'

# 标签过滤配置
LABEL_FILTER_CONFIG = {
    'enabled': True,
    'filter_values': [0],  # 只使用label=0
    'filter_column': 'label'
}

# 设置随机种子
SEED = 42
USE_MULTIPROCESSING = False  # Windows下禁用多进程

# ===================== 多任务学习标签映射 =====================
TASK_CONFIGS = {
    'quality': {
        'num_classes': 2,
        'labels': ['OK', 'NG'],
        'weight': 1.0
    },
    'workstation': {
        'num_classes': 6,
        'labels': ['工位1', '工位2', '工位3', '工位4', '工位5', '工位6'],
        'weight': 0.5
    },
    'camera': {
        'num_classes': 3,
        'labels': ['相机1', '相机2', '相机3'],
        'weight': 0.5
    }
}

# 标签映射
LABEL_MAPPINGS = {
    'quality': {label: i for i, label in enumerate(TASK_CONFIGS['quality']['labels'])},
    'workstation': {label: i for i, label in enumerate(TASK_CONFIGS['workstation']['labels'])},
    'camera': {label: i for i, label in enumerate(TASK_CONFIGS['camera']['labels'])}
}

# ===================== 多进程配置（Windows优化）=====================
if platform.system() == 'Windows':
    NUM_WORKERS = 0  # Windows必须为0
    PREPROCESSING_WORKERS = 2  # 预处理使用较少进程
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA操作
else:
    NUM_WORKERS = min(4, mp.cpu_count() - 1)
    PREPROCESSING_WORKERS = mp.cpu_count()

# 数据加载优化
PREFETCH_FACTOR = 2  # 降低预取因子
DATALOADER_TIMEOUT = 600  # 增加超时时间
PIN_MEMORY = False  # Windows下禁用pin_memory

# ===================== 数据增强配置 =====================
USE_AUGMENTATION = True
AUGMENTATION_STRENGTH = 'light'  # 使用轻度增强

# ===================== 训练优化配置 =====================
USE_MIXED_PRECISION = False  # 暂时禁用混合精度（提高稳定性）
GRADIENT_ACCUMULATION_STEPS = 2  # 梯度累积
USE_GRADIENT_CHECKPOINTING = True  # 启用梯度检查点节省内存

# 优化器配置
USE_ADAMW = True
WEIGHT_DECAY = 1e-4
USE_COSINE_ANNEALING = True
USE_WARMUP = True

# 模型配置
MODEL_TYPE = 'mobilenet'  # 使用轻量级模型
USE_PRETRAINED = True
FREEZE_BACKBONE_EPOCHS = 2

# ===================== 图片完整性检查配置 =====================
CHECK_IMAGE_INTEGRITY = True
SKIP_CORRUPTED_IMAGES = True
MAX_CORRUPTED_RATIO = 0.2
PARALLEL_CHECK = False  # Windows下禁用并行检查

# ===================== 数据加载优化 =====================
USE_MEMORY_MAPPING = False  # Windows下禁用
PERSISTENT_WORKERS = False  # Windows下必须禁用
USE_RAM_CACHE = False  # 禁用RAM缓存以节省内存
CACHE_IMAGES_IN_MEMORY = False

# 数据采样策略
USE_DATA_SAMPLING = False
BALANCE_SAMPLING = True

# ===================== 实时监控配置 =====================
ENABLE_REALTIME_PLOT = False  # 禁用实时绘图以节省资源
PLOT_UPDATE_FREQUENCY = 5
SAVE_CHECKPOINT_FREQUENCY = 5

# ===================== 错误处理配置（新增）=====================
ERROR_HANDLING = {
    'max_retries': 3,  # 最大重试次数
    'retry_delay': 5,  # 重试延迟（秒）
    'skip_on_error': True,  # 错误时跳过该批次
    'save_error_logs': True,  # 保存错误日志
}


# ===================== 工具函数 =====================
def get_device():
    """获取最佳计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        # 设置CUDA内存分配策略
        torch.cuda.set_per_process_memory_fraction(0.8)  # 只使用80%显存
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    return device


def estimate_memory_usage(dataset_size, batch_size, image_size):
    """估算内存使用"""
    # 估算单张图片内存（灰度图）
    single_image_mb = (image_size * image_size * 4) / (1024 * 1024)  # float32

    # 估算批次内存
    batch_memory_mb = single_image_mb * batch_size * 4  # 考虑梯度等

    # 估算总内存需求
    total_memory_gb = (batch_memory_mb * (dataset_size / batch_size)) / 1024

    return {
        'single_image_mb': single_image_mb,
        'batch_memory_mb': batch_memory_mb,
        'estimated_total_gb': total_memory_gb
    }


def get_num_classes(mode):
    """获取指定模式的类别数量"""
    if mode == 'multi_task':
        return {task: config['num_classes'] for task, config in TASK_CONFIGS.items()}
    elif mode in LABEL_MAPPINGS:
        return len(LABEL_MAPPINGS[mode])
    else:
        raise ValueError(f"未知的分类模式: {mode}")


def get_label_names(mode):
    """获取指定模式的标签名称列表"""
    if mode == 'multi_task':
        return {task: config['labels'] for task, config in TASK_CONFIGS.items()}
    elif mode in LABEL_MAPPINGS:
        label_map = LABEL_MAPPINGS[mode]
        return [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    else:
        raise ValueError(f"未知的分类模式: {mode}")


def validate_config():
    """验证配置的有效性"""
    import os

    errors = []

    # 检查路径
    if not os.path.exists(CSV_PATH):
        errors.append(f"CSV文件不存在: {CSV_PATH}")

    # 检查输出目录
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
    except Exception as e:
        errors.append(f"无法创建目录: {e}")

    # 检查内存配置
    if DATA_SPLIT_CONFIG['enabled'] and DATA_SPLIT_CONFIG['splits_per_epoch'] < 1:
        errors.append("splits_per_epoch 必须大于等于 1")

    # Windows特殊检查
    if platform.system() == 'Windows':
        if NUM_WORKERS != 0:
            errors.append("Windows系统下NUM_WORKERS必须为0")
        if PERSISTENT_WORKERS:
            errors.append("Windows系统下PERSISTENT_WORKERS必须为False")

    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("配置验证通过")
        return True


# ===================== 动态配置调整 =====================
def optimize_config_for_stability():
    """针对稳定性优化配置"""
    print("应用稳定性优化配置...")

    # 检测可用内存
    try:
        import psutil
        available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
        total_memory = psutil.virtual_memory().total / (1024 ** 3)  # GB

        print(f"检测到内存: 总量 {total_memory:.1f}GB, 可用 {available_memory:.1f}GB")

        # 根据可用内存调整配置
        if available_memory < 8:
            globals()['BATCH_SIZE'] = 32
            globals()['IMAGE_SIZE'] = 128
            globals()['DATA_SPLIT_CONFIG']['splits_per_epoch'] = 5
            print("低内存模式：降低批次大小和图像尺寸，增加数据分批")
        elif available_memory < 16:
            globals()['BATCH_SIZE'] = 64
            globals()['IMAGE_SIZE'] = 192
            globals()['DATA_SPLIT_CONFIG']['splits_per_epoch'] = 3
            print("中等内存模式")

    except ImportError:
        print("psutil未安装，使用默认配置")

    # Windows系统特殊处理
    if platform.system() == 'Windows':
        print("Windows系统优化:")
        print("  - 禁用多进程数据加载")
        print("  - 禁用内存映射")
        print("  - 启用CUDA同步模式")
        print("  - 启用数据分批处理")


# 自动执行优化
optimize_config_for_stability()

# ===================== 配置摘要 =====================
if __name__ == "__main__":
    print("=" * 70)
    print("配置文件摘要（稳定性优化版）")
    print("=" * 70)
    print(f"图片尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"学习率: {LEARNING_RATE}")

    print(f"\n数据分批配置:")
    if DATA_SPLIT_CONFIG['enabled']:
        print(f"  ✅ 已启用")
        print(f"  每轮分批数: {DATA_SPLIT_CONFIG['splits_per_epoch']}")
        print(f"  打乱分批: {'是' if DATA_SPLIT_CONFIG['shuffle_splits'] else '否'}")
    else:
        print(f"  ❌ 未启用")

    print(f"\n内存优化:")
    print(f"  最大内存限制: {MEMORY_CONFIG['max_memory_gb']}GB")
    print(f"  缓存清理频率: 每{MEMORY_CONFIG['clear_cache_frequency']}批次")

    print(f"\n系统配置:")
    print(f"  操作系统: {platform.system()}")
    print(f"  工作进程: {NUM_WORKERS}")
    print(f"  设备: {get_device()}")

    # 估算内存使用
    print(f"\n内存估算:")
    mem_est = estimate_memory_usage(100000, BATCH_SIZE, IMAGE_SIZE)
    print(f"  单张图片: {mem_est['single_image_mb']:.2f}MB")
    print(f"  单批次: {mem_est['batch_memory_mb']:.2f}MB")

    print("=" * 70)

    # 验证配置
    validate_config()