# classification_data.py
# 优化后的数据加载模块 - 支持多任务学习、数据分批和高速数据加载
# 针对i9-13900K + 64GB内存 + RTX A4000优化（修复多进程兼容性）
# 添加标签过滤功能和数据分批管理

__all__ = ['MultiTaskDataset', 'FastImageLoader', 'create_data_loaders', 'DataPrefetcher',
           'DataSplitManager', 'MemoryMonitor', 'create_split_data_loaders']

import os
import sys
import gc
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image, ImageFile
import numpy as np
from tqdm import tqdm
import cv2  # 使用OpenCV加速图片读取
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from collections import defaultdict, OrderedDict
import warnings
import psutil
from functools import lru_cache
import multiprocessing as mp

# 忽略PIL的警告
warnings.filterwarnings("ignore", category=UserWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 导入配置
try:
    from classification_config import *
except ImportError:
    print("错误：找不到 classification_config.py 配置文件！")
    exit(1)

# 多进程初始化标志
WORKER_INITIALIZED = False


# ===================== 数据分批管理器（新增）=====================
class DataSplitManager:
    """数据分批管理器 - 将大数据集分成多个小批次处理"""

    def __init__(self, dataset, splits_per_epoch=3, shuffle=True, seed=42):
        self.dataset = dataset
        self.splits_per_epoch = splits_per_epoch
        self.shuffle = shuffle
        self.seed = seed
        self.current_epoch = 0
        self.indices = list(range(len(dataset)))

        # 计算每个分批的大小
        self.total_size = len(dataset)
        self.split_size = self.total_size // splits_per_epoch
        self.remaining = self.total_size % splits_per_epoch

        print(f"\n数据分批管理器初始化:")
        print(f"  总样本数: {self.total_size}")
        print(f"  每轮分批数: {self.splits_per_epoch}")
        print(f"  每批大小: ~{self.split_size}")

    def get_split_indices(self, epoch, split_idx):
        """获取指定epoch和split的索引"""
        # 每个epoch开始时打乱索引
        if split_idx == 0 and self.shuffle:
            np.random.seed(self.seed + epoch)
            np.random.shuffle(self.indices)

        # 计算当前分批的起止索引
        start_idx = split_idx * self.split_size
        if split_idx < self.splits_per_epoch - 1:
            end_idx = start_idx + self.split_size
        else:
            # 最后一批包含剩余的所有样本
            end_idx = self.total_size

        return self.indices[start_idx:end_idx]

    def create_split_loader(self, epoch, split_idx, batch_size, num_workers=0):
        """创建指定分批的数据加载器"""
        # 获取当前分批的索引
        split_indices = self.get_split_indices(epoch, split_idx)

        # 创建子集
        subset = Subset(self.dataset, split_indices)

        # 创建数据加载器
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,  # 在分批内部打乱
            num_workers=num_workers,
            pin_memory=False,  # Windows下禁用
            drop_last=True if len(split_indices) > batch_size else False,
            persistent_workers=False
        )

        return loader, len(split_indices)


# ===================== 内存监控器（新增）=====================
class MemoryMonitor:
    """内存使用监控器"""

    def __init__(self, max_memory_gb=8, verbose=True):
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024 ** 3
        self.verbose = verbose
        self.memory_history = []

    def check_memory(self):
        """检查当前内存使用情况"""
        try:
            memory = psutil.virtual_memory()
            used_gb = (memory.total - memory.available) / (1024 ** 3)
            percent = memory.percent

            self.memory_history.append({
                'time': time.time(),
                'used_gb': used_gb,
                'percent': percent
            })

            # 只保留最近100条记录
            if len(self.memory_history) > 100:
                self.memory_history = self.memory_history[-100:]

            return used_gb, percent
        except:
            return 0, 0

    def should_clear_cache(self):
        """判断是否需要清理缓存"""
        used_gb, percent = self.check_memory()

        # 如果使用超过80%或接近限制，清理缓存
        if percent > 80 or used_gb > self.max_memory_gb * 0.9:
            if self.verbose:
                print(f"\n⚠️ 内存使用较高: {used_gb:.1f}GB ({percent:.1f}%), 清理缓存...")
            return True
        return False

    def clear_memory(self):
        """清理内存"""
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 给系统一点时间
        time.sleep(0.1)

        if self.verbose:
            used_gb, percent = self.check_memory()
            print(f"✅ 清理后内存: {used_gb:.1f}GB ({percent:.1f}%)")


# ===================== 内存缓存管理器（多进程安全） =====================
class MemoryCacheManager:
    """智能内存缓存管理器 - 多进程安全版本"""

    def __init__(self, max_size_gb=32):
        self.max_size_bytes = max_size_gb * 1024 ** 3
        self.cache = {}
        self.access_count = defaultdict(int)
        self.last_access = {}
        self.current_size = 0
        self.hits = 0
        self.misses = 0
        self.lock = mp.Lock() if USE_MULTIPROCESSING else None

    def get(self, key):
        """获取缓存项（线程/进程安全）"""
        if self.lock:
            with self.lock:
                return self._get_unsafe(key)
        else:
            return self._get_unsafe(key)

    def _get_unsafe(self, key):
        """非安全版本的get（内部使用）"""
        if key in self.cache:
            self.hits += 1
            self.access_count[key] += 1
            self.last_access[key] = time.time()
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value, size_bytes):
        """添加缓存项（线程/进程安全）"""
        if self.lock:
            with self.lock:
                self._put_unsafe(key, value, size_bytes)
        else:
            self._put_unsafe(key, value, size_bytes)

    def _put_unsafe(self, key, value, size_bytes):
        """非安全版本的put（内部使用）"""
        # 如果超出容量，使用LFU+LRU策略清理
        while self.current_size + size_bytes > self.max_size_bytes and self.cache:
            self._evict()

        if self.current_size + size_bytes <= self.max_size_bytes:
            self.cache[key] = value
            self.current_size += size_bytes
            self.access_count[key] = 1
            self.last_access[key] = time.time()

    def _evict(self):
        """驱逐最少使用的项"""
        if not self.cache:
            return

        # 结合访问频率和最后访问时间
        scores = {}
        current_time = time.time()

        for key in self.cache:
            frequency = self.access_count[key]
            recency = current_time - self.last_access.get(key, 0)
            # 频率越低、时间越久，分数越高（越容易被驱逐）
            scores[key] = recency / (frequency + 1)

        # 驱逐分数最高的项
        evict_key = max(scores.keys(), key=lambda k: scores[k])
        item_size = sys.getsizeof(self.cache[evict_key])
        del self.cache[evict_key]
        del self.access_count[evict_key]
        del self.last_access[evict_key]
        self.current_size -= item_size

    def get_stats(self):
        """获取缓存统计"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size_gb': self.current_size / 1024 ** 3,
            'items': len(self.cache)
        }


# ===================== 快速图片加载器（多进程安全） =====================
class FastImageLoader:
    """使用OpenCV和多线程的超快速图片加载器（多进程安全）"""

    def __init__(self, cache_manager=None, num_threads=4):
        self.cache_manager = cache_manager
        self.num_threads = num_threads
        # 延迟初始化线程池，以支持多进程
        self.thread_pool = None

        # 预编译的图像处理设置
        self.cv2_inter = cv2.INTER_LINEAR
        self.cv2_border = cv2.BORDER_REFLECT_101

    def _init_thread_pool(self):
        """延迟初始化线程池"""
        if self.thread_pool is None:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.num_threads)

    def load_image(self, path, size=(256, 256), use_cache=True):
        """快速加载并预处理图片（修复中文路径和空文件问题）"""
        # 在使用前确保线程池已初始化
        self._init_thread_pool()

        # 检查缓存
        cache_key = f"{path}_{size[0]}x{size[1]}"

        if use_cache and self.cache_manager:
            cached = self.cache_manager.get(cache_key)
            if cached is not None:
                return cached

        try:
            # 从文件读取到内存缓冲区（支持中文路径）
            img_buffer = np.fromfile(path, dtype=np.uint8)

            # 检查文件是否为空
            if img_buffer.size == 0:
                raise ValueError(f"文件为空 (0 bytes): {path}")

            # 从内存缓冲区解码图片
            img = cv2.imdecode(img_buffer, cv2.IMREAD_GRAYSCALE)

            # 检查解码是否成功
            if img is None:
                raise ValueError(f"无法解码图片: {path}")

            # 缩放图片
            if img.shape[:2] != size:
                img = cv2.resize(img, size, interpolation=self.cv2_inter)

            # 归一化
            img = img.astype(np.float32) * (1.0 / 255.0)

            # 缓存结果
            if use_cache and self.cache_manager:
                img_size = img.nbytes
                self.cache_manager.put(cache_key, img, img_size)

            return img

        except Exception as e:
            # 返回一个占位符，避免刷屏
            return np.zeros(size, dtype=np.float32)

    def preload_batch(self, paths, size=(256, 256)):
        """批量预加载图片（使用多线程）"""
        self._init_thread_pool()

        def load_single(path):
            return self.load_image(path, size)

        # 使用线程池并行加载
        images = list(self.thread_pool.map(load_single, paths))
        return images

    def close(self):
        """关闭线程池"""
        if self.thread_pool is not None:
            self.thread_pool.shutdown()
            self.thread_pool = None


# ===================== 数据预取器 =====================
class DataPrefetcher:
    """GPU数据预取器 - 在GPU上提前准备下一批数据"""

    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(loader)
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.next_data = None
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.iter)
        except StopIteration:
            self.next_data = None
            return

        if self.stream is not None and self.next_data is not None:
            with torch.cuda.stream(self.stream):
                self.next_images, self.next_labels = self.next_data
                # 异步传输到GPU
                self.next_images = self.next_images.cuda(non_blocking=True)
                if isinstance(self.next_labels, dict):
                    self.next_labels = {k: v.cuda(non_blocking=True) for k, v in self.next_labels.items()}
                else:
                    self.next_labels = self.next_labels.cuda(non_blocking=True)

    def next(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)

        if self.next_data is None:
            raise StopIteration

        if self.stream is not None:
            data = self.next_images, self.next_labels
        else:
            data = self.next_data

        self.preload()
        return data

    def __iter__(self):
        # 重置迭代器
        self.iter = iter(self.loader)
        self.preload()
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return len(self.loader)


# ===================== 多任务数据集（多进程安全） =====================
class MultiTaskDataset(Dataset):
    """
    支持多任务学习的高效数据集。
    通过延迟初始化，完美兼容Windows/Linux下的多进程数据加载。
    新增标签过滤功能。
    """

    def __init__(self, csv_path, transform=None, use_cache=True, validate_data=True,
                 label_filter=None, filter_column='label'):
        """
        初始化数据集

        Args:
            csv_path: CSV文件路径
            transform: 数据变换
            use_cache: 是否使用缓存
            validate_data: 是否验证数据
            label_filter: 标签过滤列表
            filter_column: 用于过滤的列名
        """
        # 1. 只保存可序列化的信息
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.use_cache = use_cache
        self.label_filter = label_filter
        self.filter_column = filter_column

        # 2. 将不可序列化的对象设置为 None
        self.image_loader = None
        self.cache_manager = None

        # 3. 应用标签过滤（如果指定）
        if label_filter is not None:
            self._apply_label_filter()

        # 4. 执行数据验证和索引创建
        if validate_data:
            self._validate_and_filter()
        self._create_indices()

        # 5. 打印统计信息
        self._print_statistics()

    def _apply_label_filter(self):
        """应用标签过滤"""
        print(f"\n应用标签过滤 - 列名: {self.filter_column}, 过滤值: {self.label_filter}")

        # 检查过滤列是否存在
        if self.filter_column not in self.data.columns:
            print(f"警告: 过滤列 '{self.filter_column}' 不存在于CSV中，跳过过滤")
            return

        # 打印过滤前的分布
        print("过滤前标签分布:")
        label_counts = self.data[self.filter_column].value_counts().sort_index()
        for label, count in label_counts.items():
            print(f"  Label {label}: {count} 张 ({count / len(self.data) * 100:.1f}%)")

        # 执行过滤
        original_size = len(self.data)
        self.data = self.data[self.data[self.filter_column].isin(self.label_filter)]

        # 重置索引
        self.data = self.data.reset_index(drop=True)

        # 打印过滤后的分布
        filtered_size = len(self.data)
        print(f"\n过滤后数据量: {filtered_size} (过滤掉 {original_size - filtered_size} 条)")

        if filtered_size > 0:
            label_counts = self.data[self.filter_column].value_counts().sort_index()
            print("过滤后标签分布:")
            for label, count in label_counts.items():
                print(f"  Label {label}: {count} 张")
        else:
            raise ValueError(f"过滤后没有数据！请检查标签过滤条件: {self.label_filter}")

    def _init_worker(self):
        """在每个工作进程中独立初始化非共享资源"""
        if self.image_loader is None:
            # 为当前进程创建独立的缓存管理器和图片加载器
            if self.use_cache and USE_RAM_CACHE:
                self.cache_manager = MemoryCacheManager(max_size_gb=MAX_CACHE_SIZE_GB)
            else:
                self.cache_manager = None

            self.image_loader = FastImageLoader(
                cache_manager=self.cache_manager,
                num_threads=NUM_THREADS_PER_WORKER if 'NUM_THREADS_PER_WORKER' in globals() else 2
            )

    def __getitem__(self, idx):
        """获取一个样本（多任务）"""
        # 确保工作进程已初始化
        self._init_worker()

        # 正常执行加载逻辑
        row = self.data.iloc[idx]
        image_path = row['full_path']

        # 使用图片加载器
        img = self.image_loader.load_image(
            image_path,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            use_cache=self.use_cache
        )

        # 转换为tensor
        img_tensor = torch.from_numpy(img.copy()).unsqueeze(0).float()

        # 应用变换
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # 获取多任务标签
        labels = {
            task: self.label_indices[task][idx] for task in self.tasks
        }

        return img_tensor, labels

    def __len__(self):
        return len(self.data)

    def _validate_and_filter(self):
        """验证并过滤无效数据"""
        print("验证数据有效性...")
        initial_size = len(self.data)

        # 过滤无效标签
        valid_mask = self.data['status'].isin(LABEL_MAPPINGS['quality'].keys()) & \
                     self.data['workstation'].isin(LABEL_MAPPINGS['workstation'].keys()) & \
                     self.data['camera'].isin(LABEL_MAPPINGS['camera'].keys())
        self.data = self.data[valid_mask].reset_index(drop=True)

        # 检查文件存在性和大小
        print("检查文件有效性（存在且非空）...")
        valid_files = []
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data), desc="检查文件"):
            path = row['full_path']
            if os.path.exists(path) and os.path.getsize(path) > 0:
                valid_files.append(True)
            else:
                valid_files.append(False)

        self.data = self.data[valid_files].reset_index(drop=True)

        removed = initial_size - len(self.data)
        if removed > 0:
            print(f"总共过滤了 {removed} 条无效数据。")
        print(f"最终有效数据: {len(self.data)} 条")

    def _create_indices(self):
        """创建标签索引"""
        self.tasks = list(LABEL_MAPPINGS.keys())
        self.label_indices = {
            task: self.data[col].map(LABEL_MAPPINGS[task]).values.astype(np.int64)
            for task, col in [('quality', 'status'), ('workstation', 'workstation'), ('camera', 'camera')]
        }

    def _print_statistics(self):
        """打印数据集统计信息"""
        print(f"\n{'=' * 60}")
        print("数据集统计信息")
        print(f"{'=' * 60}")
        print(f"总样本数: {len(self.data):,}")

        # 如果应用了标签过滤，显示过滤信息
        if self.label_filter is not None:
            print(f"\n标签过滤: {self.filter_column} = {self.label_filter}")

        # 如果数据集很小，给出更多诊断信息
        if len(self.data) < 100:
            print("\n⚠️ 数据集非常小！可能的原因：")
            print("1. CSV文件路径错误")
            print("2. 图片文件路径不正确")
            print("3. 标签格式不匹配")
            if self.label_filter is not None:
                print("4. 标签过滤条件过于严格")

        # 内存使用情况
        try:
            memory_info = psutil.virtual_memory()
            print(f"\n系统内存: {memory_info.total / 1024 ** 3:.1f}GB, 已用: {memory_info.percent}%")
        except:
            pass

        # 各任务的类别分布
        if len(self.data) > 0:
            for task in ['quality', 'workstation', 'camera']:
                print(f"\n{task}任务分布:")
                col_name = 'status' if task == 'quality' else task

                if col_name in self.data.columns:
                    value_counts = self.data[col_name].value_counts()

                    # 计算类别不平衡度
                    if len(value_counts) > 1:
                        max_count = value_counts.max()
                        min_count = value_counts.min()
                        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                    else:
                        imbalance_ratio = 1.0

                    for label, count in value_counts.items():
                        percentage = count / len(self.data) * 100
                        print(f"  {label}: {count:,} ({percentage:.1f}%)")

                    if imbalance_ratio > 2:
                        print(f"  ⚠️ 类别不平衡度: {imbalance_ratio:.1f}x")

        print(f"{'=' * 60}")

    def get_sample_weights(self):
        """获取用于平衡采样的权重"""
        if len(self.data) == 0:
            return np.array([])

        weights = np.ones(len(self.data), dtype=np.float32)
        for task, labels in self.label_indices.items():
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) == 0:
                continue
            class_weights = len(labels) / (len(unique) * counts)
            weight_map = {u: w for u, w in zip(unique, class_weights)}
            task_weights = np.array([weight_map[l] for l in labels])
            weights *= task_weights * TASK_CONFIGS.get(task, {}).get('weight', 1.0)
        return weights


# ===================== 创建数据加载器（支持分批）=====================
def create_split_data_loaders(csv_path=None, val_split=0.2, batch_size=None,
                              num_workers=None, use_weighted_sampler=False,
                              use_prefetcher=True, label_filter=None,
                              filter_column='label', use_splits=True):
    """
    创建支持分批的数据加载器

    Args:
        csv_path: CSV文件路径
        val_split: 验证集比例
        batch_size: 批次大小
        num_workers: 工作进程数
        use_weighted_sampler: 是否使用加权采样
        use_prefetcher: 是否使用预取器
        label_filter: 标签过滤列表
        filter_column: 用于过滤的列名
        use_splits: 是否使用数据分批
    """
    if csv_path is None:
        csv_path = CSV_PATH
    if batch_size is None:
        batch_size = BATCH_SIZE
    if num_workers is None:
        num_workers = NUM_WORKERS

    # 检查CSV文件
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

    print(f"\n创建数据加载器...")
    print(f"  批次大小: {batch_size}")
    print(f"  工作进程: {num_workers}")
    print(f"  数据分批: {'启用' if use_splits and DATA_SPLIT_CONFIG['enabled'] else '禁用'}")

    # 数据变换
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.Normalize([0.5], [0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Normalize([0.5], [0.5])
    ])

    # 创建完整数据集
    start_time = time.time()

    # 处理标签过滤
    if label_filter is None and LABEL_FILTER_CONFIG['enabled']:
        label_filter = LABEL_FILTER_CONFIG['filter_values']
        filter_column = LABEL_FILTER_CONFIG['filter_column']

    full_dataset = MultiTaskDataset(
        csv_path=csv_path,
        transform=None,  # transform 在子集上应用
        use_cache=USE_RAM_CACHE,
        validate_data=True,
        label_filter=label_filter,
        filter_column=filter_column
    )

    if len(full_dataset) == 0:
        raise ValueError("数据集为空！请检查CSV文件和数据过滤条件")

    print(f"数据集创建用时: {time.time() - start_time:.1f}秒")

    # 数据划分
    total_size = len(full_dataset)
    val_size = max(1, int(val_split * total_size))
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 创建子集并应用变换
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    train_dataset.dataset.transform = train_transform
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    val_dataset.dataset.transform = val_transform

    # 创建验证集加载器（不分批）
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=False
    )

    # 如果启用分批
    if use_splits and DATA_SPLIT_CONFIG['enabled']:
        # 创建分批管理器
        train_split_manager = DataSplitManager(
            train_dataset,
            splits_per_epoch=DATA_SPLIT_CONFIG['splits_per_epoch'],
            shuffle=DATA_SPLIT_CONFIG['shuffle_splits'],
            seed=SEED
        )

        print(f"\n数据加载器创建成功:")
        print(f"  训练集: {len(train_dataset)} 样本 (分{DATA_SPLIT_CONFIG['splits_per_epoch']}批)")
        print(f"  验证集: {len(val_dataset)} 样本")

        return train_split_manager, val_loader, full_dataset

    else:
        # 不使用分批，创建普通加载器
        train_sampler = None
        if use_weighted_sampler and USE_WEIGHTED_SAMPLER:
            try:
                train_weights = full_dataset.get_sample_weights()[train_indices]
                train_sampler = torch.utils.data.WeightedRandomSampler(
                    weights=train_weights,
                    num_samples=len(train_dataset),
                    replacement=True,
                    generator=torch.Generator().manual_seed(SEED)
                )
            except:
                train_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY and torch.cuda.is_available(),
            drop_last=True if len(train_dataset) > batch_size else False,
            persistent_workers=False
        )

        # 包装预取器
        if use_prefetcher and torch.cuda.is_available():
            train_loader = DataPrefetcher(train_loader)
            val_loader = DataPrefetcher(val_loader)

        print(f"\n数据加载器创建成功:")
        print(f"  训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
        print(f"  验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")

        return train_loader, val_loader, full_dataset


# ===================== 兼容旧接口 =====================
def create_data_loaders(csv_path=None, val_split=0.2, batch_size=None,
                        num_workers=None, use_weighted_sampler=False,
                        use_prefetcher=True, label_filter=None,
                        filter_column='label'):
    """保留原有接口，内部调用新的分批接口"""
    return create_split_data_loaders(
        csv_path=csv_path,
        val_split=val_split,
        batch_size=batch_size,
        num_workers=num_workers,
        use_weighted_sampler=use_weighted_sampler,
        use_prefetcher=use_prefetcher,
        label_filter=label_filter,
        filter_column=filter_column,
        use_splits=False  # 默认不使用分批
    )


# ===================== 测试函数 =====================
if __name__ == "__main__":
    print("测试数据加载模块...")

    # 创建内存监控器
    memory_monitor = MemoryMonitor(max_memory_gb=MEMORY_CONFIG['max_memory_gb'])

    # 测试分批数据加载
    print("\n1. 测试分批数据加载:")
    try:
        train_split_manager, val_loader, dataset = create_split_data_loaders(
            batch_size=4,
            use_splits=True
        )

        # 测试第一个分批
        split_loader, split_size = train_split_manager.create_split_loader(
            epoch=0,
            split_idx=0,
            batch_size=4,
            num_workers=0
        )

        print(f"第一批大小: {split_size} 样本")

        for i, (images, labels) in enumerate(split_loader):
            print(f"批次 {i}: 图像形状 {images.shape}")
            if i >= 2:
                break

        print("✅ 分批数据加载测试成功!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()