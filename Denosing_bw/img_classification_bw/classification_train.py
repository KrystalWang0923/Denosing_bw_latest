# classification_train.py
# 优化后的多任务训练主脚本 - 支持标签过滤和数据分批功能

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import warnings
import traceback
import platform
import json
import gc
from typing import Optional, Dict, Any, Tuple

# 证书设置（如果需要）
cert_path = r"E:\Anaconda\Anaconda-install\envs\cuda126\Lib\site-packages\certifi\cacert.pem"
if os.path.exists(cert_path):
    os.environ['REQUESTS_CA_BUNDLE'] = cert_path

warnings.filterwarnings('ignore')

# 导入配置
from classification_config import *

# Windows 多进程修复
if platform.system() == 'Windows':
    import multiprocessing

    multiprocessing.set_start_method('spawn', force=True)

# 检查其他模块是否存在
try:
    from classification_data import create_data_loaders

    print("✓ classification_data 模块加载成功")
    # 尝试导入增强版函数
    try:
        from classification_data import create_split_data_loaders

        HAS_SPLIT_SUPPORT = True
    except ImportError:
        HAS_SPLIT_SUPPORT = False
        print("  注意：未找到create_split_data_loaders，使用基础数据加载")
except ImportError as e:
    print(f"✗ 无法加载 classification_data: {e}")
    sys.exit(1)

try:
    from classification_model import create_model

    print("✓ classification_model 模块加载成功")
except ImportError as e:
    print(f"✗ 无法加载 classification_model: {e}")
    sys.exit(1)

try:
    from classification_engine import train_model, evaluate_model, get_device

    print("✓ classification_engine 模块加载成功")
except ImportError as e:
    print(f"✗ 无法加载 classification_engine: {e}")
    sys.exit(1)

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# 检查并设置默认配置值
# 这些值应该在 classification_config.py 中定义，这里提供默认值以防未定义
if 'USE_WEIGHTED_SAMPLER' not in globals():
    USE_WEIGHTED_SAMPLER = False
    print("⚠️ USE_WEIGHTED_SAMPLER 未在配置中定义，使用默认值: False")

if 'USE_MIXED_PRECISION' not in globals():
    USE_MIXED_PRECISION = False
    print("⚠️ USE_MIXED_PRECISION 未在配置中定义，使用默认值: False")

if 'MAX_PATIENCE' not in globals():
    MAX_PATIENCE = 10
    print("⚠️ MAX_PATIENCE 未在配置中定义，使用默认值: 10")

if 'USE_PRETRAINED' not in globals():
    USE_PRETRAINED = True
    print("⚠️ USE_PRETRAINED 未在配置中定义，使用默认值: True")

if 'TASK_CONFIGS' not in globals():
    TASK_CONFIGS = {}
    print("⚠️ TASK_CONFIGS 未在配置中定义，使用默认值: {}")

if 'DATA_SPLIT_CONFIG' not in globals():
    DATA_SPLIT_CONFIG = {
        'enabled': False,
        'splits_per_epoch': 1,
        'shuffle_splits': True
    }
    print("⚠️ DATA_SPLIT_CONFIG 未在配置中定义，使用默认值")


# ===================== 系统检查 =====================
def check_system_resources():
    """检查系统资源"""
    print("\n" + "=" * 60)
    print("系统资源检查")
    print("=" * 60)

    # CPU信息
    try:
        import psutil
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"CPU: {cpu_count} 核心, 当前使用率: {cpu_percent}%")

        # 内存信息
        memory = psutil.virtual_memory()
        print(f"内存: 总量 {memory.total / (1024 ** 3):.1f}GB, "
              f"可用 {memory.available / (1024 ** 3):.1f}GB ({memory.percent}% 使用)")

    except ImportError:
        print("psutil未安装，跳过详细系统信息")

    # GPU信息
    if torch.cuda.is_available():
        print(f"\nGPU信息:")
        print(f"  设备: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.1f}GB")
    else:
        print("\n未检测到GPU，将使用CPU训练")


# ===================== 主训练函数 =====================
def train_multi_task_model(args) -> Tuple[Optional[torch.nn.Module], Optional[Dict[str, Any]]]:
    """训练多任务模型的主函数"""

    print("\n" + "=" * 70)
    print("多任务图像分类训练系统 v4.0 - 支持数据分批")
    print("针对高性能硬件优化版本")
    print("=" * 70)

    # 获取设备
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')

    # 系统信息
    print("\n系统信息:")
    print(f"  操作系统: {platform.system()}")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  使用设备: {device}")

    if torch.cuda.is_available() and device.type == 'cuda':
        print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"  GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")

    # 检查系统资源
    check_system_resources()

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 在输出目录名中包含关键信息
    run_name = f"run_{timestamp}"
    if args.label_filter is not None:
        run_name += f"_filter_{'_'.join(map(str, args.label_filter))}"
    if DATA_SPLIT_CONFIG['enabled'] and not args.no_splits:
        run_name += f"_split{DATA_SPLIT_CONFIG['splits_per_epoch']}"

    run_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n输出目录: {run_dir}")

    # 保存配置
    save_config(run_dir, args)

    # ========== 1. 数据准备 ==========
    print("\n" + "=" * 60)
    print("步骤1: 准备数据")
    print("=" * 60)

    # 打印过滤设置
    if args.label_filter is not None:
        print(f"\n标签过滤设置:")
        print(f"  过滤列: {args.filter_column}")
        print(f"  保留标签: {args.label_filter}")
    else:
        print("\n未设置标签过滤，使用所有数据")

    # 数据分批设置
    use_splits = DATA_SPLIT_CONFIG['enabled'] and not args.no_splits and HAS_SPLIT_SUPPORT
    if use_splits:
        print(f"\n数据分批设置:")
        print(f"  每轮分批数: {DATA_SPLIT_CONFIG['splits_per_epoch']}")
        print(f"  打乱分批: {'是' if DATA_SPLIT_CONFIG['shuffle_splits'] else '否'}")
    else:
        if not HAS_SPLIT_SUPPORT:
            print("\n数据分批功能不可用，使用传统加载方式")

    # Windows 系统特殊处理
    if platform.system() == 'Windows':
        print("\n检测到Windows系统，调整数据加载配置...")
        num_workers = 0
        use_prefetcher = False
    else:
        num_workers = NUM_WORKERS
        use_prefetcher = True

    # 初始化变量
    train_loader = None
    val_loader = None
    dataset = None
    model = None
    history = None
    final_results = None

    try:
        start_time = time.time()

        # 检查并设置 USE_WEIGHTED_SAMPLER 默认值
        use_weighted_sampler = globals().get('USE_WEIGHTED_SAMPLER', False)

        # 创建数据加载器
        if use_splits:
            train_split_manager, val_loader, dataset = create_split_data_loaders(
                csv_path=CSV_PATH,
                val_split=0.2,
                batch_size=BATCH_SIZE,
                num_workers=num_workers,
                use_weighted_sampler=use_weighted_sampler,
                use_prefetcher=use_prefetcher,
                label_filter=args.label_filter,
                filter_column=args.filter_column,
                use_splits=True
            )
            train_loader = train_split_manager  # 用于传递给train_model
        else:
            train_loader, val_loader, dataset = create_data_loaders(
                csv_path=CSV_PATH,
                val_split=0.2,
                batch_size=BATCH_SIZE,
                num_workers=num_workers,
                use_weighted_sampler=use_weighted_sampler,
                use_prefetcher=use_prefetcher,
                label_filter=args.label_filter,
                filter_column=args.filter_column
            )

        data_time = time.time() - start_time
        print(f"\n数据准备完成，用时: {data_time:.1f}秒")

        # 测试数据加载
        print("\n测试数据加载...")
        test_data_loading(train_loader, val_loader, use_splits)

    except Exception as e:
        print(f"数据加载失败: {e}")
        traceback.print_exc()
        save_error_log(run_dir, "data_loading_error", e)
        return None, None

    # ========== 2. 创建模型 ==========
    print("\n" + "=" * 60)
    print("步骤2: 创建模型")
    print("=" * 60)

    try:
        # 创建模型
        model = create_model(
            model_type=MODEL_TYPE,
            pretrained=USE_PRETRAINED,
            device=device,
            compile_model=False  # Windows下禁用编译
        )

        if model is None:
            raise ValueError("create_model 返回了 None")

        print(f"模型创建成功!")
        print(f"模型类型: {type(model).__name__}")

        # 确保模型在正确的设备上
        model = model.to(device)

        # 测试模型前向传播
        print("\n测试模型前向传播...")
        test_input = torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"测试输出类型: {type(test_output)}")
        if isinstance(test_output, dict):
            for task, output in test_output.items():
                print(f"  {task}: {output.shape}")

    except Exception as e:
        print(f"模型创建失败: {e}")
        traceback.print_exc()
        save_error_log(run_dir, "model_creation_error", e)
        return None, None

    # 修复 classification_train.py 中的训练部分

    # ========== 3. 开始训练 ==========
    print("\n" + "=" * 60)
    print("步骤3: 开始训练")
    print("=" * 60)

    # 打印训练配置
    print("\n训练配置:")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  训练轮数: {NUM_EPOCHS}")
    print(f"  早停耐心值: {globals().get('MAX_PATIENCE', 10)}")
    print(f"  混合精度: {'启用' if globals().get('USE_MIXED_PRECISION', False) else '禁用'}")

    try:
        # 检查是否使用分批训练
        if use_splits and hasattr(train_loader, 'create_split_loader'):
            # 使用增强版训练器（支持数据分批）
            from classification_engine import EnhancedMultiTaskTrainer

            # train_loader 实际上是 DataSplitManager
            trainer = EnhancedMultiTaskTrainer(
                model,
                train_loader,  # train_split_manager
                val_loader,
                device
            )

            # 执行训练
            history = trainer.train(
                num_epochs=NUM_EPOCHS,
                save_dir=run_dir
            )

            # 获取训练后的模型
            model = trainer.model

        else:
            # 使用基础训练器
            from classification_engine import MultiTaskTrainer

            trainer = MultiTaskTrainer(
                model,
                train_loader,
                val_loader,
                device
            )

            # 如果有恢复检查点
            if args.resume:
                trainer.load_checkpoint(args.resume)

            # 执行训练
            history = trainer.train(
                num_epochs=NUM_EPOCHS,
                save_dir=run_dir
            )

            # 获取训练后的模型
            model = trainer.model

    except KeyboardInterrupt:
        print("\n用户中断训练")
        save_interrupted_state(run_dir, model if 'model' in locals() else None, "user_interrupted")
        return model if 'model' in locals() else None, history if 'history' in locals() else None

    except Exception as e:
        print(f"训练过程出错: {e}")
        traceback.print_exc()
        save_error_log(run_dir, "training_error", e)
        save_interrupted_state(run_dir, model if 'model' in locals() else None, "error_interrupted")
        return model if 'model' in locals() else None, history if 'history' in locals() else None
    # # 定位到训练部分的 except 块（约第 360 行）
    # except KeyError as e:
    #     print(f"训练过程中出现键缺失错误: {e}")
    #     print("可能是训练引擎中的指标字典未正确初始化")
    #     traceback.print_exc()
    #     save_error_log(run_dir, "key_error", e)
    #     save_interrupted_state(run_dir, model if 'model' in locals() else None, "key_error")
    #     return model, history
    # except Exception as e:
    #     # 保留原有的其他异常处理逻辑
    #     print(f"训练过程出错: {e}")
    #     traceback.print_exc()
    #     save_error_log(run_dir, "training_error", e)
    #     save_interrupted_state(run_dir, model if 'model' in locals() else None, "error_interrupted")
    #     return model, history
    # ========== 4. 评估最终模型 ==========
    if model is not None and history is not None:
        print("\n" + "=" * 60)
        print("步骤4: 最终评估")
        print("=" * 60)

        try:
            # 使用evaluate_model函数进行最终评估
            from classification_engine import evaluate_model
            final_results = evaluate_model(model, val_loader, device=device)

            # 从返回的字典中提取准确率
            # evaluate_model 返回的是 'acc' 而不是 'overall_accuracy'
            overall_accuracy = final_results.get('acc', 0)
            task_accuracies = final_results.get('task_accs', {})

            print(f"\n最终验证结果:")
            print(f"  总体准确率: {overall_accuracy:.2f}%")
            print(f"  各任务准确率:")
            for task, acc in task_accuracies.items():
                print(f"    {task}: {acc:.2f}%")

            # 为了后续保存，将结果格式化为期望的格式
            final_results['overall_accuracy'] = overall_accuracy
            final_results['task_accuracies'] = task_accuracies

        except Exception as e:
            print(f"评估失败: {e}")
            traceback.print_exc()
            final_results = None

    # ========== 5. 保存结果 ==========
    print("\n" + "=" * 60)
    print("步骤5: 保存结果")
    print("=" * 60)

    if model is not None:
        # 保存最终模型
        final_model_path = os.path.join(run_dir, 'final_model_multi_task.pth')
        try:
            # 将模型移到CPU再保存
            model_cpu = model.cpu()

            # 保存模型和配置信息
            save_dict = {
                'model_state_dict': model_cpu.state_dict(),
                'config': {
                    'model_type': MODEL_TYPE,
                    'image_size': IMAGE_SIZE,
                    'task_configs': TASK_CONFIGS,
                    'label_filter': args.label_filter,
                    'filter_column': args.filter_column,
                    'data_splits': DATA_SPLIT_CONFIG['splits_per_epoch'] if use_splits else 1
                },
                'results': final_results if final_results is not None else None
            }

            torch.save(save_dict, final_model_path)
            print(f"最终模型已保存: {final_model_path}")

            # 如果需要继续使用，将模型移回原设备
            model = model.to(device)

        except Exception as e:
            print(f"保存模型失败: {e}")

    # 保存训练报告
    save_training_report(run_dir, history, args, use_splits)

    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"所有文件已保存到: {run_dir}")
    print("=" * 70)

    return model, history


# ===================== 辅助函数 =====================
def test_data_loading(train_loader, val_loader, use_splits):
    """测试数据加载"""
    test_successful = False

    try:
        if use_splits and hasattr(train_loader, 'create_split_loader'):
            # 测试分批数据加载
            print("测试分批数据加载...")
            split_loader, split_size = train_loader.create_split_loader(
                epoch=0,
                split_idx=0,
                batch_size=BATCH_SIZE,
                num_workers=0
            )
            print(f"  第一批大小: {split_size} 样本")

            for i, batch in enumerate(split_loader):
                if len(batch) == 2:
                    images, labels = batch
                    print(f"  批次 {i + 1}: 图像形状 {images.shape}")
                    test_successful = True
                    break
        else:
            # 测试普通数据加载
            print("测试普通数据加载...")
            for i, batch in enumerate(train_loader):
                if len(batch) == 2:
                    images, labels = batch
                    print(f"  批次 {i + 1}: 图像形状 {images.shape}")
                    if isinstance(labels, dict):
                        for task, label in labels.items():
                            print(f"    {task}: {label.shape}")
                    test_successful = True
                    break

    except Exception as e:
        print(f"  数据加载测试失败: {e}")

    if not test_successful:
        print("\n警告：数据加载测试失败，请检查数据集")


def save_config(save_dir, args):
    """保存训练配置"""
    config_path = os.path.join(save_dir, 'training_config.json')

    config = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'command_args': vars(args),
        'data_config': {
            'csv_path': CSV_PATH,
            'image_size': IMAGE_SIZE,
            'batch_size': BATCH_SIZE,
            'label_filter': args.label_filter,
            'filter_column': args.filter_column
        },
        'split_config': DATA_SPLIT_CONFIG,
        'training_config': {
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'model_type': MODEL_TYPE,
            'use_pretrained': globals().get('USE_PRETRAINED', True),
            'use_mixed_precision': globals().get('USE_MIXED_PRECISION', False),
            'max_patience': globals().get('MAX_PATIENCE', 10)
        },
        'system_info': {
            'platform': platform.system(),
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def save_error_log(save_dir, error_type, exception):
    """保存错误日志"""
    error_dir = os.path.join(save_dir, 'errors')
    os.makedirs(error_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    error_path = os.path.join(error_dir, f'{error_type}_{timestamp}.txt')

    with open(error_path, 'w', encoding='utf-8') as f:
        f.write(f"Error Type: {error_type}\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Exception: {str(exception)}\n\n")
        f.write("Traceback:\n")
        f.write(traceback.format_exc())


def save_interrupted_state(save_dir, model, reason):
    """保存中断状态"""
    if model is not None:
        checkpoint_path = os.path.join(save_dir, f'interrupted_{reason}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'reason': reason,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, checkpoint_path)
        print(f"\n中断状态已保存: {checkpoint_path}")


def save_training_report(save_dir, history, args, use_splits):
    """保存训练报告"""
    report_path = os.path.join(save_dir, 'training_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("多任务图像分类训练报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输出目录: {save_dir}\n\n")

        # 训练配置
        f.write("训练配置:\n")
        f.write(f"  模型类型: {MODEL_TYPE}\n")
        f.write(f"  批次大小: {BATCH_SIZE}\n")
        f.write(f"  图像尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}\n")
        f.write(f"  学习率: {LEARNING_RATE}\n")
        f.write(f"  训练轮数: {NUM_EPOCHS}\n")

        if use_splits:
            f.write(f"\n数据分批:\n")
            f.write(f"  每轮分批数: {DATA_SPLIT_CONFIG['splits_per_epoch']}\n")

        if args.label_filter:
            f.write(f"\n标签过滤:\n")
            f.write(f"  过滤列: {args.filter_column}\n")
            f.write(f"  保留标签: {args.label_filter}\n")

        # 训练结果
        if history and 'val_acc' in history and history['val_acc']:
            f.write(f"\n训练结果:\n")
            f.write(f"  最佳验证准确率: {max(history['val_acc']):.2f}%\n")
            f.write(f"  最终验证准确率: {history['val_acc'][-1]:.2f}%\n")

            # 各任务最终准确率
            if any(f'val_acc_{task}' in history for task in ['quality', 'workstation', 'camera']):
                f.write(f"\n各任务最终准确率:\n")
                for task in ['quality', 'workstation', 'camera']:
                    key = f'val_acc_{task}'
                    if key in history and history[key]:
                        f.write(f"  {task}: {history[key][-1]:.2f}%\n")

            # 训练时间
            if 'epoch_time' in history:
                total_time = sum(history['epoch_time'])
                f.write(f"\n训练时间:\n")
                f.write(f"  总时间: {total_time / 60:.1f} 分钟\n")
                f.write(f"  平均每轮: {np.mean(history['epoch_time']):.1f} 秒\n")


# ===================== 命令行接口 =====================
def main():
    """主程序入口"""
    # Windows下的特殊处理
    if platform.system() == 'Windows' and __name__ == '__main__':
        import multiprocessing
        multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description='多任务图像分类训练 - 支持标签过滤和数据分批')

    # 基础参数
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--workers', type=int, help='数据加载进程数')
    parser.add_argument('--no-cuda', action='store_true', help='禁用CUDA')

    # 数据分批参数
    parser.add_argument('--splits-per-epoch', type=int, help='每轮分成几批')
    parser.add_argument('--no-splits', action='store_true', help='禁用数据分批')

    # 标签过滤参数
    parser.add_argument('--label-filter', type=int, nargs='+', default=None,
                        help='只使用指定标签的数据，例如: --label-filter 0 或 --label-filter 0 1')
    parser.add_argument('--filter-column', type=str, default='label',
                        help='用于过滤的列名，默认为"label"')

    # 快速设置
    parser.add_argument('--only-normal', action='store_true',
                        help='只使用正常图片(label=0)进行训练')
    parser.add_argument('--test-run', action='store_true',
                        help='测试运行（2轮，小批次）')

    # 恢复训练
    parser.add_argument('--resume', type=str, help='从检查点恢复训练')

    args = parser.parse_args()

    # 在 main() 函数中处理配置文件中的标签过滤设置
    if hasattr(sys.modules.get('classification_config', None), 'LABEL_FILTER_CONFIG'):
        config = sys.modules['classification_config'].LABEL_FILTER_CONFIG
        if config.get('enabled', False) and args.label_filter is None:
            args.label_filter = config.get('filter_values', None)
            args.filter_column = config.get('filter_column', 'label')

    # 处理快速设置选项
    if args.only_normal:
        args.label_filter = [0]
        print("已启用 --only-normal 选项，只使用正常图片(label=0)进行训练")

    # 调试/测试模式
    if args.debug or args.test_run:
        print("调试/测试模式已启用")
        globals()['NUM_EPOCHS'] = 2
        globals()['BATCH_SIZE'] = 4
        if args.test_run:
            globals()['DATA_SPLIT_CONFIG']['splits_per_epoch'] = 2

    # 应用命令行参数
    if args.epochs:
        globals()['NUM_EPOCHS'] = args.epochs
    if args.batch_size:
        globals()['BATCH_SIZE'] = args.batch_size
    if args.lr:
        globals()['LEARNING_RATE'] = args.lr
    if args.workers is not None:
        globals()['NUM_WORKERS'] = args.workers
    if args.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # 数据分批设置
    if args.no_splits:
        globals()['DATA_SPLIT_CONFIG']['enabled'] = False
    elif args.splits_per_epoch:
        globals()['DATA_SPLIT_CONFIG']['enabled'] = True
        globals()['DATA_SPLIT_CONFIG']['splits_per_epoch'] = args.splits_per_epoch

    print("=" * 70)
    print("配置信息:")
    print(f"  NUM_EPOCHS: {NUM_EPOCHS}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  LEARNING_RATE: {LEARNING_RATE}")
    print(f"  NUM_WORKERS: {NUM_WORKERS if platform.system() != 'Windows' else 0}")
    print(f"  MODEL_TYPE: {MODEL_TYPE}")

    # 打印数据分批配置
    if DATA_SPLIT_CONFIG['enabled']:
        print(f"\n  数据分批: 启用")
        print(f"  每轮分批数: {DATA_SPLIT_CONFIG['splits_per_epoch']}")
    else:
        print(f"\n  数据分批: 禁用")

    # 打印过滤配置
    if args.label_filter is not None:
        print(f"\n  标签过滤: {args.filter_column} = {args.label_filter}")
    else:
        print(f"\n  标签过滤: 未启用（使用所有数据）")

    print("=" * 70)

    # 确认是否继续（除非是测试模式）
    if not args.test_run and not args.debug:
        response = input("\n是否继续训练? (y/n): ")
        if response.lower() != 'y':
            print("训练已取消")
            return

    try:
        # 运行训练
        model, history = train_multi_task_model(args)

        if model is not None:
            print("\n✅ 训练任务完成!")
        else:
            print("\n⚠️ 训练未能成功完成")

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断训练")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()