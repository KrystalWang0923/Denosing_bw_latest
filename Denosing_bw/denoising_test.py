import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T # 图像转换
from torch.utils.data import Dataset,DataLoader,random_split # 数据集和数据加载器

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm # 进度条工具


# 导入自定义组件

from common import untils
from bw_config import *
from denoising_data import *
from denoising_engine import *
from denoising_model import *

# 定义函数：用一批数据进行测试，并画图

# 可视化函数
def visualize_results(model, test_loader, device, num_samples=5):
    model.eval()
    noise_imgs, clean_imgs = next(iter(test_loader))
    noise_imgs = noise_imgs.to(device)

    with torch.no_grad():
        denoised_imgs = model(noise_imgs)

    # 转换为numpy数组 (B, C, H, W) → (B, H, W)
    noise_imgs = noise_imgs.cpu().squeeze(1).numpy()
    denoised_imgs = denoised_imgs.cpu().squeeze(1).numpy()
    clean_imgs = clean_imgs.squeeze(1).numpy()

    # 创建图像网格
    num_samples = min(num_samples, noise_imgs.shape[0])
    fig, axes = plt.subplots(3, num_samples, figsize=(2.5*num_samples, 7.5))
    if num_samples == 1: axes = axes.reshape(3, 1)  # 处理单样本情况

    titles = ['Noisy', 'Denoised', 'Original']
    for i in range(num_samples):
        # 噪声图像
        axes[0, i].imshow(noise_imgs[i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title(titles[0])

        # 去噪图像
        axes[1, i].imshow(denoised_imgs[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title(titles[1])

        # 原始图像
        axes[2, i].imshow(clean_imgs[i], cmap='gray')
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title(titles[2])


if __name__ == '__main__':
    # 0. 准备工作
    # 检测GPU是否可用并定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 指定随机数种子，去除训练的不确定性
    untils.random_seed(SEED)
    # 定义图像预处理操作
    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor()
    ])


    # 1. 创建数据集
    print("--- 1. 创建数据集 ---")
    dataset = BalancedGrayscaleDataset(IMG_PATH,transform=transform)
    # 划分训练集和测试集
    train_dataset,test_dataset = random_split(dataset,[TRAIN_RATIO,TEST_RATIO])
    print("--- 创建数据集完成 ---")

    # 2.创建数据加载器
    print("--- 2. 创建数据加载器 ---")
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )


    test_loader = DataLoader(test_dataset,batch_size=TEST_BATCH_SIZE)
    # 3. 加载模型
    loaded_denoiser = ImprovedGrayscaleDenoiser()
    print("--- 3. 从文件加载模型 ---")
    model_state_dict = torch.load(DENOISER_MODEL_NAME, map_location=device)
    loaded_denoiser.load_state_dict(model_state_dict)
    print("--- 模型加载完成 ---")

    loaded_denoiser.to(device)

    # 4. 测试
    print("--- 4. 测试结果如下 ---")
    visualize_results(loaded_denoiser,test_loader,device)
    plt.tight_layout()
    plt.show()  # 关键：添加显示图像的语句

