# classification_model.py
# 优化后的多任务模型模块 - 支持多任务学习和高效训练
# 针对i9-13900K + 64GB内存 + RTX A4000优化

__all__ = ['create_model', 'MultiTaskCNN', 'MultiTaskLoss',
           'save_checkpoint', 'load_checkpoint', 'get_device']

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from collections import OrderedDict
import warnings
from typing import Dict, Optional, Tuple, Union

warnings.filterwarnings('ignore')

# 导入配置
from classification_config import *


# ===================== 设备配置 =====================
def get_device():
    """获取最佳计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        # 设置CUDA优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device


# ===================== 基础模块 =====================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力模块"""

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att

        return x


# ===================== 多任务头部 =====================
class MultiTaskHead(nn.Module):
    """多任务分类头部"""

    def __init__(self, in_features, task_configs, dropout_rate=0.5):
        super(MultiTaskHead, self).__init__()
        self.task_configs = task_configs
        self.heads = nn.ModuleDict()

        for task, config in task_configs.items():
            num_classes = config['num_classes']
            # 每个任务独立的分类头
            self.heads[task] = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(256, num_classes)
            )

    def forward(self, x):
        outputs = {}
        for task in self.task_configs:
            outputs[task] = self.heads[task](x)
        return outputs


# ===================== 自定义CNN模型 =====================
class MultiTaskCNN(nn.Module):
    """自定义的多任务CNN模型"""

    def __init__(self, task_configs=None, use_attention=True):
        super(MultiTaskCNN, self).__init__()

        if task_configs is None:
            task_configs = TASK_CONFIGS

        self.task_configs = task_configs
        self.use_attention = use_attention

        # 特征提取器
        self.features = nn.Sequential(
            # 第一层
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 残差块1
            self._make_block(64, 128, 2),

            # 残差块2
            self._make_block(128, 256, 2),

            # 残差块3
            self._make_block(256, 512, 2),
        )

        # 注意力模块
        if self.use_attention:
            self.attention = CBAM(512)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 多任务头部
        self.classifier = MultiTaskHead(512, task_configs, dropout_rate=0.5)

        # 初始化权重
        self._initialize_weights()

    def _make_block(self, in_channels, out_channels, blocks):
        """创建残差块"""
        layers = []

        # 第一个块需要调整通道数
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))

        # 其余块
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 特征提取
        x = self.features(x)

        # 注意力
        if self.use_attention:
            x = self.attention(x)

        # 全局池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 多任务分类
        outputs = self.classifier(x)

        return outputs


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# ===================== 预训练模型适配器 =====================
class PretrainedMultiTaskModel(nn.Module):
    """基于预训练模型的多任务模型"""

    def __init__(self, model_type='mobilenet', task_configs=None,
                 pretrained=True, freeze_epochs=0):
        super(PretrainedMultiTaskModel, self).__init__()

        if task_configs is None:
            task_configs = TASK_CONFIGS

        self.task_configs = task_configs
        self.model_type = model_type
        self.freeze_epochs = freeze_epochs
        self.current_epoch = 0

        # 创建backbone
        self._create_backbone(model_type, pretrained)

        # 获取特征维度
        self.feature_dim = self._get_feature_dim()

        # 创建多任务头部
        self.classifier = MultiTaskHead(
            self.feature_dim,
            task_configs,
            dropout_rate=0.5
        )

        # 如果需要冻结backbone
        if self.freeze_epochs > 0:
            self._freeze_backbone()

    def _create_backbone(self, model_type, pretrained):
        """创建预训练backbone"""
        if model_type == 'mobilenet':
            # MobileNetV3
            backbone = models.mobilenet_v3_small(pretrained=pretrained)
            # 修改第一层以接受灰度图像
            backbone.features[0][0] = nn.Conv2d(
                1, backbone.features[0][0].out_channels,
                kernel_size=3, stride=2, padding=1, bias=False
            )
            # 移除分类层
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        elif model_type == 'resnet':
            # ResNet18
            backbone = models.resnet18(pretrained=pretrained)
            # 修改第一层
            backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # 移除最后的全连接层
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        elif model_type == 'efficientnet':
            # EfficientNet-B0
            self.backbone = timm.create_model(
                'efficientnet_b0',
                pretrained=pretrained,
                in_chans=1,
                num_classes=0,  # 移除分类头
                global_pool='avg'
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def _get_feature_dim(self):
        """获取backbone输出特征维度"""
        # 创建一个dummy输入
        dummy_input = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)

        # 前向传播获取输出维度
        with torch.no_grad():
            features = self.backbone(dummy_input)
            if len(features.shape) == 4:
                features = F.adaptive_avg_pool2d(features, 1)
            features = features.view(features.size(0), -1)

        return features.shape[1]

    def _freeze_backbone(self):
        """冻结backbone参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"Backbone已冻结，将在{self.freeze_epochs}轮后解冻")

    def unfreeze_backbone(self):
        """解冻backbone参数"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone已解冻")

    def forward(self, x):
        # 特征提取
        features = self.backbone(x)

        # 确保是2D张量
        if len(features.shape) == 4:
            features = F.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)

        # 多任务分类
        outputs = self.classifier(features)

        return outputs

    def update_epoch(self, epoch):
        """更新当前epoch，用于控制解冻"""
        self.current_epoch = epoch
        if epoch >= self.freeze_epochs and self.freeze_epochs > 0:
            self.unfreeze_backbone()
            self.freeze_epochs = 0  # 防止重复解冻

    def get_params_groups(self):
        """获取参数组，用于不同的学习率"""
        if self.freeze_epochs > 0 and self.current_epoch < self.freeze_epochs:
            # 只返回分类器参数
            return self.classifier.parameters()
        else:
            # 返回不同学习率的参数组
            return [
                {'params': self.backbone.parameters(), 'lr': LEARNING_RATE * 0.1},
                {'params': self.classifier.parameters(), 'lr': LEARNING_RATE}
            ]


# ===================== 损失函数 =====================
class MultiTaskLoss(nn.Module):
    """多任务损失函数"""

    def __init__(self, task_configs=None, use_uncertainty=True):
        super(MultiTaskLoss, self).__init__()

        if task_configs is None:
            task_configs = TASK_CONFIGS

        self.task_configs = task_configs
        self.use_uncertainty = use_uncertainty

        # 任务损失函数
        self.criterions = nn.ModuleDict()
        for task, config in task_configs.items():
            self.criterions[task] = nn.CrossEntropyLoss()

        # 不确定性权重（可学习）
        if self.use_uncertainty:
            self.log_vars = nn.ParameterDict()
            for task in task_configs:
                self.log_vars[task] = nn.Parameter(torch.zeros(1))

    def forward(self, outputs, targets):
        """
        计算多任务损失

        Args:
            outputs: 模型输出字典 {task: predictions}
            targets: 目标字典 {task: labels}

        Returns:
            total_loss: 总损失
            task_losses: 各任务损失字典
        """
        task_losses = {}
        total_loss = 0

        for task in self.task_configs:
            if task in outputs and task in targets:
                # 计算任务损失
                loss = self.criterions[task](outputs[task], targets[task])
                task_losses[task] = loss.item()

                # 应用权重
                if self.use_uncertainty:
                    # 使用不确定性加权
                    precision = torch.exp(-self.log_vars[task])
                    weighted_loss = precision * loss + self.log_vars[task]
                    total_loss += weighted_loss
                else:
                    # 使用固定权重
                    weight = self.task_configs[task].get('weight', 1.0)
                    total_loss += weight * loss

        return total_loss, task_losses


# ===================== 模型工厂函数 =====================
def create_model(model_type='mobilenet', pretrained=True,
                 device=None, compile_model=False):
    """
    创建多任务模型

    Args:
        model_type: 模型类型 ('custom', 'mobilenet', 'resnet', 'efficientnet')
        pretrained: 是否使用预训练权重
        device: 计算设备
        compile_model: 是否编译模型（PyTorch 2.0+）

    Returns:
        model: 多任务模型
    """
    if device is None:
        device = get_device()

    print(f"\n创建模型: {model_type}")

    # 创建模型
    if model_type == 'custom':
        model = MultiTaskCNN(task_configs=TASK_CONFIGS, use_attention=True)
    else:
        model = PretrainedMultiTaskModel(
            model_type=model_type,
            task_configs=TASK_CONFIGS,
            pretrained=pretrained,
            freeze_epochs=FREEZE_BACKBONE_EPOCHS
        )

    # 移动到设备
    model = model.to(device)

    # 编译模型（PyTorch 2.0+）
    if compile_model and hasattr(torch, 'compile'):
        print("编译模型以提高性能...")
        model = torch.compile(model)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型参数统计:")
    print(f"  总参数数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")

    return model


# ===================== 检查点管理 =====================
def save_checkpoint(model, optimizer, scheduler, epoch, metrics,
                    checkpoint_path, is_best=False):
    """保存训练检查点"""
    # 准备保存内容
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': {
            'model_type': MODEL_TYPE,
            'image_size': IMAGE_SIZE,
            'task_configs': TASK_CONFIGS,
        }
    }

    # 保存检查点
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        print(f"✅ 最佳模型已保存: {checkpoint_path}")
    else:
        print(f"检查点已保存: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model=None, optimizer=None,
                    scheduler=None, device=None):
    """加载训练检查点"""
    if device is None:
        device = get_device()

    print(f"加载检查点: {checkpoint_path}")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 如果没有提供模型，则创建新模型
    if model is None:
        config = checkpoint.get('config', {})
        model_type = config.get('model_type', MODEL_TYPE)
        model = create_model(model_type=model_type, device=device)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])

    # 加载优化器状态
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 加载调度器状态
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 返回模型和检查点信息
    return model, checkpoint


# ===================== 测试代码 =====================
if __name__ == "__main__":
    print("测试多任务模型模块...")

    # 设备
    device = get_device()

    # 创建模型
    print("\n1. 测试自定义CNN模型:")
    model = create_model(model_type='custom', device=device)

    # 测试前向传播
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)

    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"\n输出结构:")
    for task, output in outputs.items():
        print(f"  {task}: {output.shape}")

    # 测试损失函数
    print("\n2. 测试多任务损失:")
    criterion = MultiTaskLoss(use_uncertainty=True)

    # 创建假标签
    targets = {
        'quality': torch.randint(0, 2, (batch_size,)).to(device),
        'workstation': torch.randint(0, 6, (batch_size,)).to(device),
        'camera': torch.randint(0, 3, (batch_size,)).to(device)
    }

    loss, task_losses = criterion(outputs, targets)
    print(f"总损失: {loss.item():.4f}")
    print(f"任务损失: {task_losses}")

    # 测试预训练模型
    print("\n3. 测试预训练模型:")
    for model_type in ['mobilenet', 'resnet']:
        print(f"\n测试 {model_type}:")
        model = create_model(model_type=model_type, device=device)

        with torch.no_grad():
            outputs = model(dummy_input)

        print(f"✅ {model_type} 测试通过")

    print("\n所有测试完成!")