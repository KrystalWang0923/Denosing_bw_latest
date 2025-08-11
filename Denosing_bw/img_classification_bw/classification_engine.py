# classification_engine.py
# 优化后的训练引擎 - 支持多任务学习、数据分批和高速训练
# 针对CUDA 12.6 + PyTorch 2.7.1环境优化

__all__ = ['MultiTaskTrainer', 'train_model', 'evaluate_model', 'WarmupCosineScheduler',
           'EnhancedMultiTaskTrainer', 'train_model_enhanced']

import os
import gc
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
import json
from datetime import datetime
import warnings
import psutil
from typing import Dict, List, Optional, Tuple
import pandas as pd
import traceback

warnings.filterwarnings('ignore')
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 导入其他模块
from classification_config import *
from classification_model import MultiTaskLoss, save_checkpoint, load_checkpoint

# 尝试导入数据分批管理器
try:
    from classification_data import DataSplitManager, MemoryMonitor

    HAS_SPLIT_SUPPORT = True
except ImportError:
    print("警告：无法导入数据分批模块，使用基础功能")
    DataSplitManager = None
    MemoryMonitor = None
    HAS_SPLIT_SUPPORT = False

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import logging

logging.getLogger('matplotlib').setLevel(logging.ERROR)

# CUDA 12.6 优化设置
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.set_per_process_memory_fraction(0.95)
    torch.cuda.empty_cache()


def get_device():
    """获取设备，并进行CUDA优化设置"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用设备: {device}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")

        if hasattr(torch.cuda, 'set_sync_debug_mode'):
            torch.cuda.set_sync_debug_mode(0)
    else:
        device = torch.device("cpu")
        print(f"使用设备: {device}")

    return device


# ===================== 学习率调度器 =====================
class WarmupCosineScheduler:
    """带预热的余弦退火学习率调度器"""

    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 warmup_factor=0.1, min_lr=1e-6, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_factor = warmup_factor
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch
        self.step(self.last_epoch)

    def step(self, epoch=None):
        """更新学习率"""
        if epoch is None:
            self.last_epoch += 1
            epoch = self.last_epoch
        else:
            self.last_epoch = epoch

        if epoch < self.warmup_epochs:
            if self.warmup_epochs > 0:
                scale = self.warmup_factor + (1 - self.warmup_factor) * (epoch / self.warmup_epochs)
            else:
                scale = 1.0
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1 + np.cos(np.pi * progress))

        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            new_lr = max(self.min_lr, base_lr * scale)
            param_group['lr'] = new_lr

    def get_last_lr(self):
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        """返回调度器的状态字典"""
        return {
            'last_epoch': self.last_epoch,
            'base_lrs': self.base_lrs,
        }

    def load_state_dict(self, state_dict):
        """从状态字典中加载状态"""
        self.last_epoch = state_dict['last_epoch']
        self.base_lrs = state_dict.get('base_lrs', self.base_lrs)
        self.step(self.last_epoch)


# ===================== 训练监控器 =====================
class TrainingMonitor:
    """增强的训练监控器"""

    def __init__(self, max_history=1000):
        self.metrics = defaultdict(lambda: [])
        self.start_time = time.time()
        self.max_history = max_history

    def update(self, **kwargs):
        """更新指标"""
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            self.metrics[key].append(value)

            if len(self.metrics[key]) > self.max_history:
                self.metrics[key] = self.metrics[key][-self.max_history:]

    def update_epoch_metrics(self, epoch, metrics):
        """更新epoch级别的训练指标"""
        self.update(
            train_loss=metrics.get('loss', 0),
            train_acc=metrics.get('acc', 0)
        )

        if 'task_accs' in metrics:
            for task, acc in metrics['task_accs'].items():
                self.update(**{f'train_acc_{task}': acc})

        if 'task_losses' in metrics:
            for task, loss in metrics['task_losses'].items():
                self.update(**{f'train_loss_{task}': loss})

        if 'epoch_time' in metrics:
            self.update(epoch_time=metrics['epoch_time'])

    def update_validation_metrics(self, epoch, metrics):
        """更新验证指标"""
        self.update(
            val_loss=metrics['loss'],
            val_acc=metrics['acc']
        )

        if 'task_accs' in metrics:
            for task, acc in metrics['task_accs'].items():
                self.update(**{f'val_acc_{task}': acc})

        if 'task_losses' in metrics:
            for task, loss in metrics['task_losses'].items():
                self.update(**{f'val_loss_{task}': loss})

    def get_last(self, key, default=0):
        """获取最后一个值"""
        return self.metrics[key][-1] if self.metrics[key] else default

    def get_average(self, key, last_n=None):
        """获取平均值"""
        values = self.metrics[key]
        if not values:
            return 0
        if last_n:
            values = values[-last_n:]
        return np.mean(values)

    def get_best(self, key, mode='max'):
        """获取最佳值"""
        if not self.metrics[key]:
            return 0
        if mode == 'max':
            return max(self.metrics[key])
        else:
            return min(self.metrics[key])

    def get_elapsed_time(self):
        """获取已用时间"""
        return time.time() - self.start_time

    def get_history(self):
        """获取训练历史"""
        return dict(self.metrics)

    def save(self, path):
        """保存监控数据"""
        data = {}
        for key, values in self.metrics.items():
            if values and isinstance(values[0], (int, float)):
                data[key] = values
            else:
                data[key] = [float(v) for v in values]

        data['total_time'] = self.get_elapsed_time()

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# ===================== 工具类 =====================
class AverageMeter:
    """计算和存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ===================== 共享工具函数 =====================
def plot_training_curves_shared(history, save_dir):
    """共享的绘制训练曲线函数"""
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('训练过程监控', fontsize=16)

    # 1. 损失曲线
    ax = axes[0, 0]
    epochs = range(1, len(history.get('train_loss', [])) + 1)

    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], 'b-', label='训练损失')
    if 'val_loss' in history:
        ax.plot(range(1, len(history['val_loss']) + 1),
                history['val_loss'], 'r-', label='验证损失')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('损失曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 准确率曲线
    ax = axes[0, 1]
    if 'train_acc' in history:
        ax.plot(epochs, history['train_acc'], 'b-', label='训练准确率')
    if 'val_acc' in history:
        ax.plot(range(1, len(history['val_acc']) + 1),
                history['val_acc'], 'r-', label='验证准确率')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('准确率曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 各任务准确率
    ax = axes[1, 0]
    task_names = [key.replace('val_acc_', '') for key in history.keys()
                  if key.startswith('val_acc_') and key != 'val_acc']

    for task in task_names:
        key = f'val_acc_{task}'
        if key in history:
            ax.plot(range(1, len(history[key]) + 1),
                    history[key], label=task)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('各任务验证准确率')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 训练时间
    ax = axes[1, 1]
    if 'epoch_time' in history:
        epoch_times = history['epoch_time']
        ax.bar(range(1, len(epoch_times) + 1), epoch_times)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (s)')
        ax.set_title('每轮训练时间')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图形
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"训练曲线已保存到: {plot_path}")


# ===================== 增强的多任务训练器（支持数据分批）=====================
class EnhancedMultiTaskTrainer:
    """支持数据分批的多任务训练器"""

    def __init__(self, model, train_split_manager, val_loader, device=None):
        self.model = model
        self.train_split_manager = train_split_manager  # 分批管理器
        self.val_loader = val_loader
        self.device = device or get_device()

        # 将模型移到设备
        self.model = self.model.to(self.device)

        # 损失函数
        self.criterion = MultiTaskLoss(use_uncertainty=False)

        # 优化器
        if hasattr(model, 'get_params_groups'):
            param_groups = model.get_params_groups()
        else:
            param_groups = model.parameters()

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        # 学习率调度器
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=WARMUP_EPOCHS,
            total_epochs=NUM_EPOCHS
        )

        # 混合精度训练
        self.use_amp = USE_MIXED_PRECISION and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()

        # 训练监控
        self.monitor = TrainingMonitor()
        if MemoryMonitor is not None:
            self.memory_monitor = MemoryMonitor(max_memory_gb=MEMORY_CONFIG['max_memory_gb'])
        else:
            self.memory_monitor = None

        # 最佳指标
        # 最佳指标（初始化时包含 'epoch' 键，避免KeyError）
        self.best_metrics = {'val_loss': float('inf'), 'val_acc': 0, 'epoch': 0}
        self.start_epoch = 0

        # 错误处理
        self.error_count = 0
        self.max_errors = ERROR_HANDLING['max_retries']

    def train_epoch_with_splits(self, epoch):
        """使用数据分批训练一个epoch"""
        self.model.train()

        epoch_metrics = defaultdict(list)
        epoch_start_time = time.time()

        # 获取分批数
        num_splits = self.train_split_manager.splits_per_epoch

        print(f"\nEpoch {epoch}/{NUM_EPOCHS} - 分{num_splits}批训练")

        # 对每个分批进行训练
        for split_idx in range(num_splits):
            print(f"\n处理第 {split_idx + 1}/{num_splits} 批...")

            try:
                # 创建当前分批的数据加载器
                split_loader, split_size = self.train_split_manager.create_split_loader(
                    epoch=epoch - 1,  # DataSplitManager使用0索引
                    split_idx=split_idx,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS
                )

                # 训练当前分批
                split_metrics = self._train_split(split_loader, epoch, split_idx, split_size)

                # 记录指标
                for key, value in split_metrics.items():
                    epoch_metrics[key].append(value)

                # 内存管理
                if DATA_SPLIT_CONFIG['gc_collect_per_split']:
                    self._cleanup_memory()

                # 可选：每个分批后保存检查点
                if DATA_SPLIT_CONFIG['save_checkpoint_per_split']:
                    self._save_split_checkpoint(epoch, split_idx)

            except Exception as e:
                print(f"\n❌ 分批 {split_idx + 1} 训练失败: {e}")
                traceback.print_exc()

                if ERROR_HANDLING['skip_on_error']:
                    print("跳过此分批，继续训练...")
                    continue
                else:
                    raise

        # 计算epoch平均指标
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            if not values:
                continue

            # 处理嵌套字典（如 task_accs, task_losses）
            if isinstance(values[0], dict):
                # 合并嵌套字典：按任务取平均值
                merged = defaultdict(list)
                for sub_dict in values:
                    for task, val in sub_dict.items():
                        merged[task].append(val)
                avg_dict = {task: np.mean(vals) for task, vals in merged.items()}
                avg_metrics[key] = avg_dict
            else:
                # 处理标量列表（如 loss, acc）
                if key not in ['samples', 'errors']:
                    avg_metrics[key] = np.mean(values)
                elif key in ['samples', 'errors']:
                    avg_metrics[key] = sum(values)

        # 记录epoch时间
        epoch_time = time.time() - epoch_start_time
        avg_metrics['epoch_time'] = epoch_time

        # 更新监控器
        self.monitor.update_epoch_metrics(epoch, avg_metrics)

        return avg_metrics

    def _train_split(self, split_loader, epoch, split_idx, split_size):
        """训练单个数据分批"""
        # 统计变量
        split_loss = 0
        split_correct = defaultdict(int)
        split_total = defaultdict(int)
        task_losses = defaultdict(float)

        # 进度条
        pbar = tqdm(split_loader,
                    desc=f'Epoch {epoch} Split {split_idx + 1}',
                    ncols=100)

        batch_count = 0
        error_count = 0

        for batch_idx, (images, labels) in enumerate(pbar):
            try:
                # 数据移到设备
                images = images.to(self.device, non_blocking=True)
                for task in labels:
                    labels[task] = labels[task].to(self.device, non_blocking=True)

                # 检查数据有效性
                if torch.isnan(images).any() or images.size(0) == 0:
                    error_count += 1
                    continue

                # 前向传播
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss, batch_task_losses = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss, batch_task_losses = self.criterion(outputs, labels)

                # 反向传播
                self.optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                # 统计
                split_loss += loss.item()
                batch_count += 1

                # 更新任务损失
                for task, task_loss in batch_task_losses.items():
                    task_losses[task] += task_loss

                # 计算准确率
                with torch.no_grad():
                    for task in outputs:
                        _, predicted = outputs[task].max(1)
                        correct = predicted.eq(labels[task]).sum().item()
                        split_correct[task] += correct
                        split_total[task] += labels[task].size(0)

                # 更新进度条
                if batch_count > 0:
                    current_loss = split_loss / batch_count
                    overall_acc = sum(split_correct.values()) / max(sum(split_total.values()), 1) * 100

                    pbar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'acc': f'{overall_acc:.1f}%',
                        'err': error_count
                    })

                # 定期清理缓存
                if batch_idx % MEMORY_CONFIG['clear_cache_frequency'] == 0:
                    if self.memory_monitor and self.memory_monitor.should_clear_cache():
                        self.memory_monitor.clear_memory()

            except Exception as e:
                error_count += 1
                print(f"\n批次处理错误: {e}")

                if error_count > 10:
                    print("错误过多，跳过此分批")
                    break

                continue

        # 计算分批指标
        if batch_count > 0:
            avg_loss = split_loss / batch_count

            # 各任务损失
            for task in task_losses:
                task_losses[task] /= batch_count

            # 各任务准确率
            task_accs = {}
            for task in split_correct:
                if split_total[task] > 0:
                    task_accs[task] = 100. * split_correct[task] / split_total[task]
                else:
                    task_accs[task] = 0

            # 总体准确率
            overall_acc = np.mean(list(task_accs.values())) if task_accs else 0

            metrics = {
                'loss': avg_loss,
                'acc': overall_acc,
                'task_accs': task_accs,
                'task_losses': task_losses,
                'samples': sum(split_total.values()),
                'errors': error_count
            }

            print(f"\n分批统计 - Loss: {avg_loss:.4f}, Acc: {overall_acc:.2f}%, 样本数: {metrics['samples']}")

            return metrics
        else:
            return {'loss': 0, 'acc': 0, 'task_accs': {}, 'task_losses': {}, 'samples': 0, 'errors': error_count}

    def validate(self, epoch=None):
        """验证模型"""
        self.model.eval()

        val_loss = 0
        val_correct = defaultdict(int)
        val_total = defaultdict(int)
        task_losses = defaultdict(float)

        all_predictions = defaultdict(list)
        all_labels = defaultdict(list)

        error_count = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader,
                        desc=f'Validation {epoch}' if epoch else 'Validation',
                        ncols=100)

            batch_count = 0

            for images, labels in pbar:
                try:
                    # 数据传输
                    images = images.to(self.device, non_blocking=True)
                    for task in labels:
                        labels[task] = labels[task].to(self.device, non_blocking=True)

                    # 检查数据
                    if torch.isnan(images).any() or images.size(0) == 0:
                        error_count += 1
                        continue

                    # 前向传播
                    outputs = self.model(images)
                    loss, batch_task_losses = self.criterion(outputs, labels)

                    # 统计
                    val_loss += loss.item()
                    batch_count += 1

                    for task, task_loss in batch_task_losses.items():
                        task_losses[task] += task_loss

                    # 计算准确率
                    for task in outputs:
                        _, predicted = outputs[task].max(1)
                        correct = predicted.eq(labels[task]).sum().item()
                        val_correct[task] += correct
                        val_total[task] += labels[task].size(0)

                        # 收集预测结果
                        all_predictions[task].extend(predicted.cpu().numpy())
                        all_labels[task].extend(labels[task].cpu().numpy())

                    # 更新进度条
                    if batch_count > 0:
                        current_loss = val_loss / batch_count
                        overall_acc = sum(val_correct.values()) / max(sum(val_total.values()), 1) * 100

                        pbar.set_postfix({
                            'loss': f'{current_loss:.4f}',
                            'acc': f'{overall_acc:.1f}%'
                        })

                except Exception as e:
                    error_count += 1
                    print(f"\n验证批次错误: {e}")
                    continue

        # 计算最终指标
        if batch_count > 0:
            avg_loss = val_loss / batch_count

            # 各任务指标
            for task in task_losses:
                task_losses[task] /= batch_count

            task_accs = {}
            for task in val_correct:
                if val_total[task] > 0:
                    task_accs[task] = 100. * val_correct[task] / val_total[task]
                else:
                    task_accs[task] = 0

            overall_acc = np.mean(list(task_accs.values())) if task_accs else 0

            return {
                'loss': avg_loss,
                'acc': overall_acc,
                'task_accs': task_accs,
                'task_losses': task_losses,
                'predictions': all_predictions,
                'labels': all_labels,
                'errors': error_count
            }
        else:
            return {
                'loss': float('inf'),
                'acc': 0,
                'task_accs': {},
                'task_losses': {},
                'predictions': {},
                'labels': {},
                'errors': error_count
            }

    def train(self, num_epochs=None, save_dir=None):
        """完整训练流程"""
        if num_epochs is None:
            num_epochs = NUM_EPOCHS
        if save_dir is None:
            save_dir = OUTPUT_DIR

        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"开始多任务训练（数据分批模式）")
        print(f"{'=' * 80}")
        print(f"每轮分批数: {DATA_SPLIT_CONFIG['splits_per_epoch']}")
        print(f"设备: {self.device}")

        # 初始验证
        print("\n初始验证...")
        initial_metrics = self.validate(0)
        print(f"初始验证准确率: {initial_metrics['acc']:.2f}%")

        # 记录初始指标
        self.monitor.update_validation_metrics(0, initial_metrics)

        # 训练循环
        start_time = time.time()
        patience_counter = 0

        for epoch in range(self.start_epoch + 1, self.start_epoch + num_epochs + 1):
            try:
                # 更新学习率
                self.scheduler.step(epoch - 1)

                # 训练一个epoch（分批）
                train_metrics = self.train_epoch_with_splits(epoch)

                # 验证
                print(f"\n验证 Epoch {epoch}...")
                val_metrics = self.validate(epoch)

                # 记录验证指标
                self.monitor.update_validation_metrics(epoch, val_metrics)

                # 打印结果
                self._print_epoch_results(epoch, train_metrics, val_metrics)

                # 保存最佳模型
                if val_metrics['acc'] > self.best_metrics['val_acc']:
                    self.best_metrics.update({
                        'val_acc': val_metrics['acc'],
                        'val_loss': val_metrics['loss'],
                        'epoch': epoch,
                        'task_accs': val_metrics['task_accs']
                    })

                    checkpoint_path = os.path.join(save_dir, 'checkpoint_multi_task_best.pth')
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        epoch, val_metrics,
                        checkpoint_path, is_best=True
                    )

                    patience_counter = 0
                    print(f"✨ 新的最佳模型! 验证准确率: {val_metrics['acc']:.2f}%")
                else:
                    patience_counter += 1

                # 定期保存
                if epoch % SAVE_CHECKPOINT_FREQUENCY == 0:
                    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        epoch, val_metrics, checkpoint_path
                    )

                # 早停
                if patience_counter >= MAX_PATIENCE:
                    print(f"\n早停触发：{MAX_PATIENCE}轮未改善")
                    break

                # 保存训练历史
                self.save_training_history(save_dir)

                # 清理内存
                self._cleanup_memory()

            except KeyboardInterrupt:
                print("\n用户中断训练")
                break

            except Exception as e:
                print(f"\nEpoch {epoch} 训练错误: {e}")
                traceback.print_exc()

                self.error_count += 1
                if self.error_count >= self.max_errors:
                    print("错误次数过多，停止训练")
                    break

                print(f"尝试恢复训练... (错误 {self.error_count}/{self.max_errors})")
                time.sleep(ERROR_HANDLING['retry_delay'])

        # 训练完成
        total_time = time.time() - start_time
        self._print_training_summary(total_time)

        # 最终保存
        self.save_training_history(save_dir)
        self.plot_training_curves(save_dir)

        return self.monitor.get_history()

    def _cleanup_memory(self):
        """清理内存"""
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _save_split_checkpoint(self, epoch, split_idx):
        """保存分批检查点（可选）"""
        checkpoint_path = os.path.join(
            OUTPUT_DIR,
            f'checkpoint_epoch_{epoch}_split_{split_idx}.pth'
        )

        torch.save({
            'epoch': epoch,
            'split_idx': split_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_path)

    def _print_epoch_results(self, epoch, train_metrics, val_metrics):
        """打印epoch结果"""
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch} 结果:")
        print(f"训练 - Loss: {train_metrics.get('loss', 0):.4f}, "
              f"Acc: {train_metrics.get('acc', 0):.2f}%")
        print(f"验证 - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['acc']:.2f}%")

        # 各任务准确率
        print("各任务验证准确率:")
        for task, acc in val_metrics['task_accs'].items():
            print(f"  {task}: {acc:.2f}%", end='  ')
        print()

        # 训练时间
        if 'epoch_time' in train_metrics:
            print(f"用时: {train_metrics['epoch_time']:.1f}秒")

    def _print_training_summary(self, total_time):
        """打印训练总结"""
        print(f"\n{'=' * 80}")
        print("训练完成!")
        print(f"{'=' * 80}")
        print(f"总用时: {total_time / 60:.1f} 分钟")
        # 检查 'epoch' 键是否存在，避免KeyError
        best_epoch = self.best_metrics.get('epoch', '未记录')
        print(f"最佳验证准确率: {self.best_metrics['val_acc']:.2f}% "
              f"(Epoch {best_epoch})")

        if 'task_accs' in self.best_metrics:
            print("最佳模型各任务准确率:")
            for task, acc in self.best_metrics['task_accs'].items():
                print(f"  {task}: {acc:.2f}%")

    def save_training_history(self, save_dir):
        """保存训练历史"""
        history_path = os.path.join(save_dir, 'training_history.json')
        self.monitor.save(history_path)
        print(f"训练历史已保存到: {history_path}")

    def plot_training_curves(self, save_dir):
        """绘制训练曲线"""
        history = self.monitor.get_history()
        plot_training_curves_shared(history, save_dir)

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint.get('epoch', 0)
        self.best_metrics = checkpoint.get('best_metrics', self.best_metrics)

        print(f"已加载检查点: {checkpoint_path}")
        print(f"从 Epoch {self.start_epoch + 1} 继续训练")


# ===================== 基础多任务训练器 =====================
class MultiTaskTrainer:
    """基础多任务训练器（不使用数据分批）"""

    def __init__(self, model, train_loader, val_loader, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or get_device()

        # 将模型移到设备
        self.model = self.model.to(self.device)

        # 损失函数
        self.criterion = MultiTaskLoss(use_uncertainty=False)

        # 优化器
        if hasattr(model, 'get_params_groups'):
            param_groups = model.get_params_groups()
        else:
            param_groups = model.parameters()

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        # 学习率调度器
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=WARMUP_EPOCHS,
            total_epochs=NUM_EPOCHS
        )

        # 混合精度训练
        self.use_amp = USE_MIXED_PRECISION and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()

        # 训练监控
        self.monitor = TrainingMonitor()

        # 最佳指标
        self.best_metrics = {'val_loss': float('inf'), 'val_acc': 0}
        self.start_epoch = 0

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()

        train_loss = AverageMeter()
        train_acc = AverageMeter()
        task_losses = defaultdict(AverageMeter)
        task_accs = defaultdict(AverageMeter)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', ncols=100)

        for batch_idx, (images, labels) in enumerate(pbar):
            # 数据移到设备
            images = images.to(self.device, non_blocking=True)
            for task in labels:
                labels[task] = labels[task].to(self.device, non_blocking=True)

            # 前向传播
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss, batch_task_losses = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss, batch_task_losses = self.criterion(outputs, labels)

            # 反向传播
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # 统计
            train_loss.update(loss.item(), images.size(0))

            # 更新任务损失
            for task, task_loss in batch_task_losses.items():
                task_losses[task].update(task_loss, images.size(0))

            # 计算准确率
            with torch.no_grad():
                batch_total_correct = 0
                batch_total_samples = 0

                for task in outputs:
                    _, predicted = outputs[task].max(1)
                    correct = predicted.eq(labels[task]).sum().item()
                    task_accs[task].update(100. * correct / labels[task].size(0),
                                           labels[task].size(0))

                    batch_total_correct += correct
                    batch_total_samples += labels[task].size(0)

                batch_acc = 100. * batch_total_correct / batch_total_samples
                train_acc.update(batch_acc, batch_total_samples)

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{train_loss.avg:.4f}',
                'acc': f'{train_acc.avg:.1f}%'
            })

        # 返回epoch指标
        metrics = {
            'loss': train_loss.avg,
            'acc': train_acc.avg,
            'task_losses': {task: meter.avg for task, meter in task_losses.items()},
            'task_accs': {task: meter.avg for task, meter in task_accs.items()}
        }

        return metrics

    def validate(self, epoch=None):
        """验证模型"""
        self.model.eval()

        val_loss = AverageMeter()
        val_acc = AverageMeter()
        task_losses = defaultdict(AverageMeter)
        task_accs = defaultdict(AverageMeter)

        all_predictions = defaultdict(list)
        all_labels = defaultdict(list)

        with torch.no_grad():
            pbar = tqdm(self.val_loader,
                        desc=f'Validation {epoch}' if epoch else 'Validation',
                        ncols=100)

            for images, labels in pbar:
                # 数据移到设备
                images = images.to(self.device, non_blocking=True)
                for task in labels:
                    labels[task] = labels[task].to(self.device, non_blocking=True)

                # 前向传播
                outputs = self.model(images)
                loss, batch_task_losses = self.criterion(outputs, labels)

                # 统计
                val_loss.update(loss.item(), images.size(0))

                for task, task_loss in batch_task_losses.items():
                    task_losses[task].update(task_loss, images.size(0))

                # 计算准确率
                batch_total_correct = 0
                batch_total_samples = 0

                for task in outputs:
                    _, predicted = outputs[task].max(1)
                    correct = predicted.eq(labels[task]).sum().item()
                    task_accs[task].update(100. * correct / labels[task].size(0),
                                           labels[task].size(0))

                    batch_total_correct += correct
                    batch_total_samples += labels[task].size(0)

                    # 收集预测结果
                    all_predictions[task].extend(predicted.cpu().numpy())
                    all_labels[task].extend(labels[task].cpu().numpy())

                batch_acc = 100. * batch_total_correct / batch_total_samples
                val_acc.update(batch_acc, batch_total_samples)

                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{val_loss.avg:.4f}',
                    'acc': f'{val_acc.avg:.1f}%'
                })

        return {
            'loss': val_loss.avg,
            'acc': val_acc.avg,
            'task_losses': {task: meter.avg for task, meter in task_losses.items()},
            'task_accs': {task: meter.avg for task, meter in task_accs.items()},
            'predictions': all_predictions,
            'labels': all_labels
        }

    def train(self, num_epochs=None, save_dir=None):
        """完整训练流程"""
        if num_epochs is None:
            num_epochs = NUM_EPOCHS
        if save_dir is None:
            save_dir = OUTPUT_DIR

        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"开始多任务训练")
        print(f"{'=' * 80}")
        print(f"设备: {self.device}")

        # 初始验证
        print("\n初始验证...")
        initial_metrics = self.validate(0)
        print(f"初始验证准确率: {initial_metrics['acc']:.2f}%")

        # 记录初始指标
        self.monitor.update_validation_metrics(0, initial_metrics)

        # 训练循环
        start_time = time.time()
        patience_counter = 0

        for epoch in range(self.start_epoch + 1, self.start_epoch + num_epochs + 1):
            # 更新学习率
            self.scheduler.step(epoch - 1)

            # 训练一个epoch
            epoch_start_time = time.time()
            train_metrics = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start_time
            train_metrics['epoch_time'] = epoch_time

            # 验证
            val_metrics = self.validate(epoch)

            # 记录指标
            self.monitor.update_epoch_metrics(epoch, train_metrics)
            self.monitor.update_validation_metrics(epoch, val_metrics)

            # 打印结果
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{num_epochs} 结果:")
            print(f"训练 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.2f}%")
            print(f"验证 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.2f}%")
            print(f"学习率: {self.scheduler.get_last_lr()[0]:.6f}")
            print(f"用时: {epoch_time:.1f}秒")

            # 保存最佳模型
            if val_metrics['acc'] > self.best_metrics['val_acc']:
                self.best_metrics.update({
                    'val_acc': val_metrics['acc'],
                    'val_loss': val_metrics['loss'],
                    'epoch': epoch,
                    'task_accs': val_metrics['task_accs']
                })

                checkpoint_path = os.path.join(save_dir, 'checkpoint_best.pth')
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, val_metrics,
                    checkpoint_path, is_best=True
                )

                patience_counter = 0
                print(f"✨ 新的最佳模型! 验证准确率: {val_metrics['acc']:.2f}%")
            else:
                patience_counter += 1

            # 定期保存
            if epoch % SAVE_CHECKPOINT_FREQUENCY == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, val_metrics, checkpoint_path
                )

            # 早停
            if patience_counter >= MAX_PATIENCE:
                print(f"\n早停触发：{MAX_PATIENCE}轮未改善")
                break

            # 清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 训练完成
        total_time = time.time() - start_time
        print(f"\n{'=' * 80}")
        print("训练完成!")
        print(f"{'=' * 80}")
        print(f"总用时: {total_time / 60:.1f} 分钟")
        print(f"最佳验证准确率: {self.best_metrics['val_acc']:.2f}% "
              f"(Epoch {self.best_metrics['epoch']})")

        # 保存训练历史和曲线
        self.save_training_history(save_dir)
        self.plot_training_curves(save_dir)

        return self.monitor.get_history()

    def save_training_history(self, save_dir):
        """保存训练历史"""
        history_path = os.path.join(save_dir, 'training_history.json')
        self.monitor.save(history_path)
        print(f"训练历史已保存到: {history_path}")

    def plot_training_curves(self, save_dir):
        """绘制训练曲线"""
        history = self.monitor.get_history()

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练过程监控', fontsize=16)

        # 1. 损失曲线
        ax = axes[0, 0]
        epochs = range(1, len(history.get('train_loss', [])) + 1)

        if 'train_loss' in history:
            ax.plot(epochs, history['train_loss'], 'b-', label='训练损失')
        if 'val_loss' in history:
            ax.plot(range(1, len(history['val_loss']) + 1),
                    history['val_loss'], 'r-', label='验证损失')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('损失曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 准确率曲线
        ax = axes[0, 1]
        if 'train_acc' in history:
            ax.plot(epochs, history['train_acc'], 'b-', label='训练准确率')
        if 'val_acc' in history:
            ax.plot(range(1, len(history['val_acc']) + 1),
                    history['val_acc'], 'r-', label='验证准确率')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('准确率曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 各任务准确率
        ax = axes[1, 0]
        task_names = [key.replace('val_acc_', '') for key in history.keys()
                      if key.startswith('val_acc_') and key != 'val_acc']

        for task in task_names:
            key = f'val_acc_{task}'
            if key in history:
                ax.plot(range(1, len(history[key]) + 1),
                        history[key], label=task)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('各任务验证准确率')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 训练时间
        ax = axes[1, 1]
        if 'epoch_time' in history:
            epoch_times = history['epoch_time']
            ax.bar(range(1, len(epoch_times) + 1), epoch_times)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time (s)')
            ax.set_title('每轮训练时间')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # 保存图形
        plot_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"训练曲线已保存到: {plot_path}")


# ===================== 训练和评估函数 =====================
def train_model(model, train_loader, val_loader, device=None):
    """基础训练函数"""
    trainer = MultiTaskTrainer(model, train_loader, val_loader, device)
    history = trainer.train()
    return trainer, history


def train_model_enhanced(model, train_split_manager, val_loader, device=None):
    """增强训练函数（支持数据分批）"""
    trainer = EnhancedMultiTaskTrainer(model, train_split_manager, val_loader, device)
    history = trainer.train()
    return trainer, history


def evaluate_model(model, test_loader, device=None):
    """评估模型"""
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    criterion = MultiTaskLoss(use_uncertainty=False)

    test_loss = AverageMeter()
    test_acc = AverageMeter()
    task_accs = defaultdict(AverageMeter)

    all_predictions = defaultdict(list)
    all_labels = defaultdict(list)

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing', ncols=100)

        for images, labels in pbar:
            # 数据移到设备
            images = images.to(device, non_blocking=True)
            for task in labels:
                labels[task] = labels[task].to(device, non_blocking=True)

            # 前向传播
            outputs = model(images)
            loss, _ = criterion(outputs, labels)

            # 统计
            test_loss.update(loss.item(), images.size(0))

            # 计算准确率
            batch_total_correct = 0
            batch_total_samples = 0

            for task in outputs:
                _, predicted = outputs[task].max(1)
                correct = predicted.eq(labels[task]).sum().item()
                task_accs[task].update(100. * correct / labels[task].size(0),
                                       labels[task].size(0))

                batch_total_correct += correct
                batch_total_samples += labels[task].size(0)

                # 收集预测结果
                all_predictions[task].extend(predicted.cpu().numpy())
                all_labels[task].extend(labels[task].cpu().numpy())

            batch_acc = 100. * batch_total_correct / batch_total_samples
            test_acc.update(batch_acc, batch_total_samples)

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{test_loss.avg:.4f}',
                'acc': f'{test_acc.avg:.1f}%'
            })

    # 打印结果
    print(f"\n{'=' * 60}")
    print("测试结果:")
    print(f"Loss: {test_loss.avg:.4f}")
    print(f"Overall Accuracy: {test_acc.avg:.2f}%")
    print("\n各任务准确率:")
    for task, acc_meter in task_accs.items():
        print(f"  {task}: {acc_meter.avg:.2f}%")

    return {
        'loss': test_loss.avg,
        'acc': test_acc.avg,
        'task_accs': {task: meter.avg for task, meter in task_accs.items()},
        'predictions': all_predictions,
        'labels': all_labels
    }