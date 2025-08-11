# classification_visualize.py
# 优化后的可视化模块 - 支持多任务模型和样本展示
# 针对内存优化和稳定性改进

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import json
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from collections import defaultdict
import warnings
from typing import Dict, List, Optional, Tuple, Union
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

# 导入项目模块
try:
    from classification_config import *
    from classification_model import create_model, load_checkpoint
    from classification_data import MultiTaskDataset
except ImportError as e:
    print(f"错误：无法导入必要的模块 - {e}")
    print("请确保以下文件存在：")
    print("  - classification_config.py")
    print("  - classification_model.py")
    print("  - classification_data.py")
    sys.exit(1)


# ===================== 工具函数 =====================
def get_device():
    """获取计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device


def load_training_history(history_path):
    """加载训练历史"""
    if not os.path.exists(history_path):
        print(f"训练历史文件不存在: {history_path}")
        return None

    with open(history_path, 'r') as f:
        history = json.load(f)

    return history


def ensure_output_dir(base_dir, sub_dir):
    """确保输出目录存在"""
    output_dir = os.path.join(base_dir, sub_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ===================== 可视化类 =====================
class MultiTaskVisualizer:
    """多任务模型可视化器"""

    def __init__(self, model_path, device=None):
        """
        初始化可视化器

        Args:
            model_path: 模型文件路径
            device: 计算设备
        """
        self.model_path = model_path
        self.device = device or get_device()

        # 加载模型
        self.model, self.checkpoint = self._load_model()
        self.model.eval()

        # 获取配置
        self.config = self.checkpoint.get('config', {})
        self.task_configs = self.config.get('task_configs', TASK_CONFIGS)

        # 创建输出目录
        self.output_dir = os.path.dirname(model_path)
        self.viz_dir = ensure_output_dir(self.output_dir, 'visualizations')

        print(f"\n可视化器初始化完成")
        print(f"输出目录: {self.viz_dir}")

    def _load_model(self):
        """加载模型"""
        print(f"加载模型: {self.model_path}")

        # 加载检查点
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # 创建模型
        config = checkpoint.get('config', {})
        model_type = config.get('model_type', MODEL_TYPE)

        model = create_model(
            model_type=model_type,
            pretrained=False,
            device=self.device
        )

        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("警告：检查点中未找到model_state_dict")

        return model, checkpoint

    def visualize_training_curves(self, history_path=None):
        """可视化训练曲线"""
        # 如果没有指定历史文件，尝试自动查找
        if history_path is None:
            history_path = os.path.join(self.output_dir, 'training_history.json')

        history = load_training_history(history_path)
        if history is None:
            print("无法加载训练历史")
            return

        # 创建图形
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. 总体损失曲线
        ax1 = fig.add_subplot(gs[0, :2])
        if 'train_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
        if 'val_loss' in history:
            epochs = range(1, len(history['val_loss']) + 1)
            ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('损失曲线', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        # 2. 总体准确率曲线
        ax2 = fig.add_subplot(gs[0, 2])
        if 'train_acc' in history:
            epochs = range(1, len(history['train_acc']) + 1)
            ax2.plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
        if 'val_acc' in history:
            epochs = range(1, len(history['val_acc']) + 1)
            ax2.plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2)

        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('准确率曲线', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)

        # 3. 各任务损失
        ax3 = fig.add_subplot(gs[1, :2])
        task_colors = {'quality': 'blue', 'workstation': 'green', 'camera': 'orange'}

        for task, color in task_colors.items():
            train_key = f'train_loss_{task}'
            val_key = f'val_loss_{task}'

            if train_key in history:
                epochs = range(1, len(history[train_key]) + 1)
                ax3.plot(epochs, history[train_key], f'{color[0]}-',
                         label=f'{task}(训练)', linewidth=1.5, alpha=0.7)

            if val_key in history:
                epochs = range(1, len(history[val_key]) + 1)
                ax3.plot(epochs, history[val_key], f'{color[0]}--',
                         label=f'{task}(验证)', linewidth=2)

        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Loss', fontsize=12)
        ax3.set_title('各任务损失曲线', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10, ncol=2)
        ax3.grid(True, alpha=0.3)

        # 4. 各任务准确率
        ax4 = fig.add_subplot(gs[1, 2])

        for task, color in task_colors.items():
            val_key = f'val_acc_{task}'

            if val_key in history:
                epochs = range(1, len(history[val_key]) + 1)
                ax4.plot(epochs, history[val_key], f'{color[0]}-',
                         label=task, linewidth=2)

        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Accuracy (%)', fontsize=12)
        ax4.set_title('各任务验证准确率', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)

        # 5. 训练时间分析
        ax5 = fig.add_subplot(gs[2, 0])
        if 'epoch_time' in history:
            epoch_times = history['epoch_time']
            epochs = range(1, len(epoch_times) + 1)

            bars = ax5.bar(epochs, epoch_times, color='skyblue', alpha=0.7)

            # 添加平均线
            avg_time = np.mean(epoch_times)
            ax5.axhline(y=avg_time, color='red', linestyle='--',
                        label=f'平均: {avg_time:.1f}s')

            ax5.set_xlabel('Epoch', fontsize=12)
            ax5.set_ylabel('Time (s)', fontsize=12)
            ax5.set_title('每轮训练时间', fontsize=14, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')

        # 6. 学习率变化
        ax6 = fig.add_subplot(gs[2, 1])
        if 'learning_rate' in history:
            epochs = range(1, len(history['learning_rate']) + 1)
            ax6.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
            ax6.set_xlabel('Epoch', fontsize=12)
            ax6.set_ylabel('Learning Rate', fontsize=12)
            ax6.set_title('学习率变化', fontsize=14, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            ax6.set_yscale('log')

        # 7. 最佳结果总结
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')

        # 计算最佳结果
        best_val_acc = max(history.get('val_acc', [0]))
        best_epoch = history.get('val_acc', []).index(best_val_acc) + 1 if history.get('val_acc') else 0

        summary_text = f"训练总结\n\n"
        summary_text += f"总轮数: {len(history.get('val_acc', []))}\n"
        summary_text += f"最佳验证准确率: {best_val_acc:.2f}%\n"
        summary_text += f"最佳轮次: {best_epoch}\n\n"

        # 各任务最佳准确率
        summary_text += "各任务最佳准确率:\n"
        for task in ['quality', 'workstation', 'camera']:
            key = f'val_acc_{task}'
            if key in history and history[key]:
                best_task_acc = max(history[key])
                summary_text += f"  {task}: {best_task_acc:.2f}%\n"

        # 训练时间
        if 'epoch_time' in history:
            total_time = sum(history['epoch_time'])
            summary_text += f"\n总训练时间: {total_time / 60:.1f} 分钟"

        ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 保存图形
        plt.suptitle('多任务训练过程可视化', fontsize=16, fontweight='bold')
        save_path = os.path.join(self.viz_dir, 'training_curves_detailed.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"训练曲线已保存: {save_path}")

    def visualize_predictions(self, dataset, num_samples=16, random_seed=42):
        """可视化预测结果"""
        np.random.seed(random_seed)

        # 随机选择样本
        total_samples = len(dataset)
        sample_indices = np.random.choice(total_samples,
                                          min(num_samples, total_samples),
                                          replace=False)

        # 创建图形
        cols = 4
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten() if num_samples > 1 else [axes]

        # 预测并显示
        self.model.eval()

        with torch.no_grad():
            for idx, sample_idx in enumerate(sample_indices):
                if idx >= len(axes):
                    break

                ax = axes[idx]

                # 获取图像和标签
                image, labels = dataset[sample_idx]

                # 添加batch维度并移到设备
                image_batch = image.unsqueeze(0).to(self.device)

                # 预测
                outputs = self.model(image_batch)

                # 获取预测结果
                predictions = {}
                for task in outputs:
                    _, pred = outputs[task].max(1)
                    predictions[task] = pred.item()

                # 显示图像
                img_display = image.squeeze().numpy()
                ax.imshow(img_display, cmap='gray')

                # 构建标题
                title_lines = []

                # 真实标签
                title_lines.append("真实标签:")
                for task in self.task_configs:
                    true_label = labels[task].item() if torch.is_tensor(labels[task]) else labels[task]
                    task_labels = self.task_configs[task]['labels']
                    true_name = task_labels[true_label] if true_label < len(task_labels) else f"未知{true_label}"
                    title_lines.append(f"  {task}: {true_name}")

                # 预测标签
                title_lines.append("\n预测标签:")
                all_correct = True
                for task in predictions:
                    pred_label = predictions[task]
                    true_label = labels[task].item() if torch.is_tensor(labels[task]) else labels[task]
                    task_labels = self.task_configs[task]['labels']
                    pred_name = task_labels[pred_label] if pred_label < len(task_labels) else f"未知{pred_label}"

                    is_correct = pred_label == true_label
                    all_correct &= is_correct

                    color = '✓' if is_correct else '✗'
                    title_lines.append(f"  {task}: {pred_name} {color}")

                # 设置标题颜色
                title = '\n'.join(title_lines)
                ax.set_title(title, fontsize=10,
                             color='green' if all_correct else 'red')
                ax.axis('off')

        # 移除多余的子图
        for idx in range(len(sample_indices), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('预测结果可视化', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图形
        save_path = os.path.join(self.viz_dir, 'predictions_visualization.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"预测可视化已保存: {save_path}")

    def visualize_confusion_matrices(self, dataset, batch_size=32):
        """生成并可视化混淆矩阵"""
        print("\n生成混淆矩阵...")

        # 创建数据加载器
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0)

        # 收集所有预测和真实标签
        all_predictions = defaultdict(list)
        all_labels = defaultdict(list)

        self.model.eval()

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="预测中"):
                images = images.to(self.device)
                outputs = self.model(images)

                for task in outputs:
                    _, preds = outputs[task].max(1)
                    all_predictions[task].extend(preds.cpu().numpy())

                    if isinstance(labels[task], torch.Tensor):
                        all_labels[task].extend(labels[task].cpu().numpy())
                    else:
                        all_labels[task].extend(labels[task])

        # 为每个任务创建混淆矩阵
        num_tasks = len(self.task_configs)
        fig, axes = plt.subplots(1, num_tasks, figsize=(6 * num_tasks, 5))

        if num_tasks == 1:
            axes = [axes]

        for idx, (task, config) in enumerate(self.task_configs.items()):
            ax = axes[idx]

            # 计算混淆矩阵
            cm = confusion_matrix(all_labels[task], all_predictions[task])

            # 归一化
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # 绘制混淆矩阵
            im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax)

            # 设置标签
            class_names = config['labels']
            tick_marks = np.arange(len(class_names))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_yticklabels(class_names)

            # 添加数值标签
            thresh = cm_normalized.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                            ha="center", va="center",
                            color="white" if cm_normalized[i, j] > thresh else "black")

            ax.set_ylabel('真实标签', fontsize=12)
            ax.set_xlabel('预测标签', fontsize=12)
            ax.set_title(f'{task}任务混淆矩阵', fontsize=14, fontweight='bold')

        plt.suptitle('各任务混淆矩阵', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图形
        save_path = os.path.join(self.viz_dir, 'confusion_matrices.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"混淆矩阵已保存: {save_path}")

        # 生成分类报告
        self._generate_classification_reports(all_labels, all_predictions)

    def _generate_classification_reports(self, all_labels, all_predictions):
        """生成分类报告"""
        report_path = os.path.join(self.viz_dir, 'classification_reports.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("多任务分类报告\n")
            f.write("=" * 80 + "\n\n")

            for task, config in self.task_configs.items():
                f.write(f"\n{task}任务分类报告:\n")
                f.write("-" * 60 + "\n")

                # 生成分类报告
                report = classification_report(
                    all_labels[task],
                    all_predictions[task],
                    target_names=config['labels'],
                    digits=3
                )

                f.write(report)
                f.write("\n")

                # 计算总体准确率
                accuracy = np.mean(np.array(all_labels[task]) == np.array(all_predictions[task]))
                f.write(f"总体准确率: {accuracy:.3f}\n")
                f.write("\n")

        print(f"分类报告已保存: {report_path}")

    def visualize_feature_maps(self, image_path, layer_names=None):
        """可视化特征图"""
        print("\n生成特征图可视化...")

        # 加载并预处理图像
        if isinstance(image_path, str):
            # 从文件加载
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"无法加载图像: {image_path}")
                return
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        else:
            # 使用提供的numpy数组
            img = image_path

        # 转换为tensor
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        # 获取中间层输出
        activations = {}

        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()

            return hook

        # 注册钩子
        hooks = []

        # 如果没有指定层，选择一些默认层
        if layer_names is None:
            if hasattr(self.model, 'backbone'):
                # 预训练模型
                if hasattr(self.model.backbone, 'features'):
                    # MobileNet等
                    for i, layer in enumerate(self.model.backbone.features):
                        if i % 3 == 0:  # 每3层选一个
                            hook = layer.register_forward_hook(hook_fn(f'feature_{i}'))
                            hooks.append(hook)
                elif hasattr(self.model.backbone, 'layer1'):
                    # ResNet等
                    for name in ['layer1', 'layer2', 'layer3', 'layer4']:
                        if hasattr(self.model.backbone, name):
                            layer = getattr(self.model.backbone, name)
                            hook = layer.register_forward_hook(hook_fn(name))
                            hooks.append(hook)
            else:
                # 自定义模型
                if hasattr(self.model, 'features'):
                    for i, layer in enumerate(self.model.features):
                        if i % 2 == 0:  # 每2层选一个
                            hook = layer.register_forward_hook(hook_fn(f'feature_{i}'))
                            hooks.append(hook)

        # 前向传播
        with torch.no_grad():
            output = self.model(img_tensor)

        # 移除钩子
        for hook in hooks:
            hook.remove()

        # 可视化特征图
        num_activations = len(activations)
        if num_activations == 0:
            print("未能获取特征图")
            return

        # 创建图形
        cols = 4
        rows = num_activations + 1  # +1 for original image
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))

        # 显示原始图像
        for i in range(cols):
            if i == 0:
                axes[0, i].imshow(img, cmap='gray')
                axes[0, i].set_title('原始图像', fontsize=12)
            else:
                axes[0, i].axis('off')
            axes[0, i].axis('off')

        # 显示特征图
        for idx, (name, activation) in enumerate(activations.items()):
            row = idx + 1

            # 获取前4个通道
            feat = activation[0].cpu().numpy()
            num_channels = min(4, feat.shape[0])

            for i in range(cols):
                if i < num_channels:
                    axes[row, i].imshow(feat[i], cmap='hot')
                    axes[row, i].set_title(f'{name} - Ch{i}', fontsize=10)
                axes[row, i].axis('off')

        plt.suptitle('特征图可视化', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图形
        save_path = os.path.join(self.viz_dir, 'feature_maps.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"特征图已保存: {save_path}")

    def visualize_error_analysis(self, dataset, num_errors=20):
        """错误分析可视化"""
        print("\n分析错误样本...")

        # 收集错误样本
        error_samples = []

        self.model.eval()

        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc="查找错误样本"):
                image, labels = dataset[idx]

                # 预测
                image_batch = image.unsqueeze(0).to(self.device)
                outputs = self.model(image_batch)

                # 检查是否有错误
                has_error = False
                errors = {}

                for task in outputs:
                    _, pred = outputs[task].max(1)
                    true_label = labels[task].item() if torch.is_tensor(labels[task]) else labels[task]

                    if pred.item() != true_label:
                        has_error = True
                        errors[task] = {
                            'predicted': pred.item(),
                            'true': true_label
                        }

                if has_error:
                    error_samples.append({
                        'idx': idx,
                        'image': image,
                        'labels': labels,
                        'errors': errors
                    })

                if len(error_samples) >= num_errors:
                    break

        if not error_samples:
            print("未找到错误样本")
            return

        # 可视化错误样本
        cols = 4
        rows = (len(error_samples) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten() if len(error_samples) > 1 else [axes]

        for idx, error_data in enumerate(error_samples):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # 显示图像
            img_display = error_data['image'].squeeze().numpy()
            ax.imshow(img_display, cmap='gray')

            # 构建标题
            title_lines = [f"样本 #{error_data['idx']}"]

            for task, error_info in error_data['errors'].items():
                task_labels = self.task_configs[task]['labels']
                pred_name = task_labels[error_info['predicted']]
                true_name = task_labels[error_info['true']]
                title_lines.append(f"{task}: {true_name}→{pred_name}")

            ax.set_title('\n'.join(title_lines), fontsize=10, color='red')
            ax.axis('off')

        # 移除多余的子图
        for idx in range(len(error_samples), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('错误样本分析', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图形
        save_path = os.path.join(self.viz_dir, 'error_analysis.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"错误分析已保存: {save_path}")

        # 生成错误统计报告
        self._generate_error_statistics(error_samples)

    def _generate_error_statistics(self, error_samples):
        """生成错误统计报告"""
        error_stats = defaultdict(lambda: defaultdict(int))

        # 统计错误类型
        for sample in error_samples:
            for task, error_info in sample['errors'].items():
                pred = error_info['predicted']
                true = error_info['true']
                error_stats[task][(true, pred)] += 1

        # 生成报告
        report_path = os.path.join(self.viz_dir, 'error_statistics.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("错误统计分析\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"总错误样本数: {len(error_samples)}\n\n")

            for task, errors in error_stats.items():
                f.write(f"\n{task}任务错误分析:\n")
                f.write("-" * 40 + "\n")

                task_labels = self.task_configs[task]['labels']

                # 按错误次数排序
                sorted_errors = sorted(errors.items(),
                                       key=lambda x: x[1],
                                       reverse=True)

                for (true, pred), count in sorted_errors:
                    true_name = task_labels[true] if true < len(task_labels) else f"未知{true}"
                    pred_name = task_labels[pred] if pred < len(task_labels) else f"未知{pred}"
                    f.write(f"  {true_name} → {pred_name}: {count}次\n")

        print(f"错误统计已保存: {report_path}")


# ===================== 数据分析函数 =====================
def analyze_dataset(dataset, output_dir):
    """分析数据集分布"""
    print("\n分析数据集...")

    # 收集标签分布
    label_counts = defaultdict(lambda: defaultdict(int))

    for idx in tqdm(range(len(dataset)), desc="统计标签"):
        _, labels = dataset[idx]

        for task, label in labels.items():
            if isinstance(label, torch.Tensor):
                label = label.item()
            label_counts[task][label] += 1

    # 创建可视化
    num_tasks = len(label_counts)
    fig, axes = plt.subplots(1, num_tasks, figsize=(6 * num_tasks, 5))

    if num_tasks == 1:
        axes = [axes]

    colors = plt.cm.Set3(np.linspace(0, 1, 10))

    for idx, (task, counts) in enumerate(label_counts.items()):
        ax = axes[idx]

        # 获取标签名称
        if task in TASK_CONFIGS:
            label_names = TASK_CONFIGS[task]['labels']
        else:
            label_names = [f"类别{i}" for i in range(max(counts.keys()) + 1)]

        # 准备数据
        labels = []
        values = []
        for label_id in sorted(counts.keys()):
            if label_id < len(label_names):
                labels.append(label_names[label_id])
            else:
                labels.append(f"未知{label_id}")
            values.append(counts[label_id])

        # 绘制柱状图
        bars = ax.bar(labels, values, color=colors[:len(labels)])

        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value}\n({value / sum(values) * 100:.1f}%)',
                    ha='center', va='bottom')

        ax.set_xlabel('类别', fontsize=12)
        ax.set_ylabel('样本数', fontsize=12)
        ax.set_title(f'{task}任务标签分布', fontsize=14, fontweight='bold')
        ax.set_xticklabels(labels, rotation=45 if len(labels) > 3 else 0, ha='right')

        # 添加网格
        ax.yaxis.grid(True, alpha=0.3)

    plt.suptitle('数据集标签分布分析', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存图形
    save_path = os.path.join(output_dir, 'dataset_distribution.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"数据集分布图已保存: {save_path}")

    # 生成统计报告
    report_path = os.path.join(output_dir, 'dataset_statistics.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("数据集统计信息\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"总样本数: {len(dataset)}\n\n")

        for task, counts in label_counts.items():
            f.write(f"\n{task}任务:\n")
            f.write("-" * 40 + "\n")

            total = sum(counts.values())

            if task in TASK_CONFIGS:
                label_names = TASK_CONFIGS[task]['labels']

                for label_id in sorted(counts.keys()):
                    count = counts[label_id]
                    percentage = count / total * 100

                    if label_id < len(label_names):
                        label_name = label_names[label_id]
                    else:
                        label_name = f"未知{label_id}"

                    f.write(f"  {label_name}: {count} ({percentage:.2f}%)\n")

            # 计算类别不平衡度
            if len(counts) > 1:
                max_count = max(counts.values())
                min_count = min(counts.values())
                imbalance_ratio = max_count / min_count
                f.write(f"\n  类别不平衡度: {imbalance_ratio:.2f}x\n")

    print(f"数据集统计已保存: {report_path}")


def visualize_sample_grid(dataset, output_dir, num_samples=25, random_seed=42):
    """创建样本网格展示"""
    np.random.seed(random_seed)

    # 随机选择样本
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    # 创建网格
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for idx, sample_idx in enumerate(indices):
        ax = axes[idx]

        # 获取图像和标签
        image, labels = dataset[sample_idx]

        # 显示图像
        img_display = image.squeeze().numpy()
        ax.imshow(img_display, cmap='gray')

        # 构建标签文本
        label_text = []
        for task in ['quality', 'workstation', 'camera']:
            if task in labels:
                label_idx = labels[task].item() if torch.is_tensor(labels[task]) else labels[task]
                if task in TASK_CONFIGS and label_idx < len(TASK_CONFIGS[task]['labels']):
                    label_name = TASK_CONFIGS[task]['labels'][label_idx]
                else:
                    label_name = f"未知{label_idx}"

                # 简化标签名称以适应显示
                if task == 'quality':
                    label_text.append(label_name)
                elif task == 'workstation':
                    label_text.append(label_name.replace('工位', 'W'))
                elif task == 'camera':
                    label_text.append(label_name.replace('相机', 'C'))

        ax.set_title(' | '.join(label_text), fontsize=8)
        ax.axis('off')

    # 隐藏多余的子图
    for idx in range(len(indices), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('数据集样本展示', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存图形
    save_path = os.path.join(output_dir, 'sample_grid.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"样本网格已保存: {save_path}")


# ===================== 主函数 =====================
def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='多任务图像分类可视化工具')

    # 基础参数
    parser.add_argument('--model', type=str,
                        help='模型文件路径 (.pth文件)')
    parser.add_argument('--data', type=str, default=None,
                        help='CSV数据文件路径（可选）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录（默认为模型所在目录）')

    # 可视化选项
    parser.add_argument('--all', action='store_true',
                        help='执行所有可视化')
    parser.add_argument('--curves', action='store_true',
                        help='绘制训练曲线')
    parser.add_argument('--predictions', action='store_true',
                        help='可视化预测结果')
    parser.add_argument('--confusion', action='store_true',
                        help='生成混淆矩阵')
    parser.add_argument('--features', action='store_true',
                        help='可视化特征图')
    parser.add_argument('--errors', action='store_true',
                        help='错误分析')
    parser.add_argument('--dataset', action='store_true',
                        help='数据集分析')

    # 其他参数
    parser.add_argument('--num-samples', type=int, default=16,
                        help='可视化的样本数量')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')

    # 标签过滤参数（与训练脚本一致）
    parser.add_argument('--label-filter', type=int, nargs='+', default=None,
                        help='只使用指定标签的数据')
    parser.add_argument('--filter-column', type=str, default='label',
                        help='用于过滤的列名')

    args = parser.parse_args()

    # 尝试从配置文件加载默认模型路径
    try:
        from classification_config import DEFAULT_MODEL_PATH
        default_model = DEFAULT_MODEL_PATH
    except ImportError:
        default_model = None
    except AttributeError:
        default_model = None

    # 确定最终模型路径：命令行参数 > 配置文件默认值
    if args.model:
        model_path = args.model
    elif default_model:
        model_path = default_model
        print(f"未传入 --model 参数，使用配置文件默认模型路径: {model_path}")
    else:
        print("错误：未传入 --model 参数，且配置文件无 DEFAULT_MODEL_PATH 配置！")
        return

    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 - {model_path}")
        return

    # 设置输出目录
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.dirname(args.model)

    # 创建可视化器
    print("\n" + "=" * 60)
    print("多任务图像分类可视化工具")
    print("=" * 60)

    visualizer = MultiTaskVisualizer(model_path)

    # 如果没有指定任何选项，默认执行所有
    if not any([args.curves, args.predictions, args.confusion,
                args.features, args.errors, args.dataset]):
        args.all = True

    # 训练曲线
    if args.all or args.curves:
        print("\n[1/6] 绘制训练曲线...")
        try:
            visualizer.visualize_training_curves()
        except Exception as e:
            print(f"训练曲线绘制失败: {e}")

    # 需要数据集的可视化
    if any([args.all, args.predictions, args.confusion,
            args.errors, args.dataset]):

        # 确定CSV路径
        if args.data:
            csv_path = args.data
        else:
            csv_path = CSV_PATH

        if not os.path.exists(csv_path):
            print(f"\n警告：找不到数据文件 - {csv_path}")
            print("跳过需要数据集的可视化")
        else:
            # 创建数据集
            print(f"\n加载数据集: {csv_path}")

            # 应用标签过滤（如果指定）
            label_filter = args.label_filter
            filter_column = args.filter_column

            # 检查模型配置中的过滤设置
            model_config = visualizer.config
            if label_filter is None and 'label_filter' in model_config:
                label_filter = model_config['label_filter']
                filter_column = model_config.get('filter_column', 'label')
                print(f"使用模型训练时的过滤设置: {filter_column} = {label_filter}")

            try:
                dataset = MultiTaskDataset(
                    csv_path=csv_path,
                    transform=None,
                    use_cache=False,
                    validate_data=True,
                    label_filter=label_filter,
                    filter_column=filter_column
                )

                # 预测可视化
                if args.all or args.predictions:
                    print("\n[2/6] 可视化预测结果...")
                    visualizer.visualize_predictions(dataset,
                                                     num_samples=args.num_samples,
                                                     random_seed=args.seed)

                # 混淆矩阵
                if args.all or args.confusion:
                    print("\n[3/6] 生成混淆矩阵...")
                    visualizer.visualize_confusion_matrices(dataset,
                                                            batch_size=args.batch_size)

                # 特征图
                if args.all or args.features:
                    print("\n[4/6] 可视化特征图...")
                    # 使用第一个样本
                    if len(dataset) > 0:
                        sample_image, _ = dataset[0]
                        sample_np = sample_image.squeeze().numpy()
                        visualizer.visualize_feature_maps(sample_np)

                # 错误分析
                if args.all or args.errors:
                    print("\n[5/6] 错误分析...")
                    visualizer.visualize_error_analysis(dataset,
                                                        num_errors=20)

                # 数据集分析
                if args.all or args.dataset:
                    print("\n[6/6] 数据集分析...")
                    analyze_dataset(dataset, visualizer.viz_dir)
                    visualize_sample_grid(dataset, visualizer.viz_dir,
                                          num_samples=25,
                                          random_seed=args.seed)

            except Exception as e:
                print(f"数据集相关可视化失败: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 60)
    print("可视化完成!")
    print("所有文件已保存到: {}".format(visualizer.viz_dir))
    print("=" * 60)


if __name__ == "__main__":
    main()