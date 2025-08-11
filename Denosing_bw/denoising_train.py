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
from img_denoise_bw.denoising_engine import train_step,test_step
from denoising_test import *

if __name__ == "__main__":
    #0. 准备工作
    # 检测GPU是否可用并定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


    # 3. 训练模型

    # 定义模型、损失函数和优化器
    denoiser = ImprovedGrayscaleDenoiser()
    loss = nn.MSELoss()
    optimizer = optim.Adam(denoiser.parameters(), lr=LEARNING_RATE)

    denoiser.to(device)
    min_test_loss = 9999 # 初始化最小测试误差为极大值
    print("--- 3. 开始训练模型 ---")

    for epoch in tqdm(range(EPOCHS)):
        # 训练一轮
        train_loss = train_step(denoiser, train_loader, loss, optimizer, device)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}: Train loss: {train_loss:.6f}")

        # 进行测试，获取测试误差
        test_loss = test_step(denoiser, test_loader, loss, device)
        print(f"\nEpoch {epoch + 1}: Test loss: {test_loss:.6f}")

        # 判断当前测试误差是否小于历史最小值，如果小于则保存模型参数
        if test_loss < min_test_loss:
            print(f"测试误差减小，保存模型...\n")
            min_test_loss = test_loss
            torch.save(denoiser.state_dict(), DENOISER_MODEL_NAME)
        else:
            print(f"测试误差没有减小，不做保存！")

    print("--- 训练完成 ---")


    # 4. 测试效果
    print("--- 4. 测试模型效果 ---")
    visualize_results(denoiser,test_loader,device)

    # 5. 对比：从文件加载模型测试效果展示
    loaded_denoiser = ImprovedGrayscaleDenoiser()
    print("--- 5. 从文件加载模型 ---")
    model_state_dict = torch.load(DENOISER_MODEL_NAME, map_location=device)
    loaded_denoiser.load_state_dict(model_state_dict)
    print("--- 模型加载完成 ---")

    loaded_denoiser.to(device)
    print("--- 测试结果如下 ---")
    visualize_results(denoiser,test_loader,device)

