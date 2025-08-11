__all__ = ["BalancedGrayscaleDataset"]
import os
from torch.utils.data import Dataset,random_split,DataLoader
import torch
from PIL import Image
import torchvision.transforms as T
from img_denoise_bw.bw_config import IMG_PATH, IMG_HEIGHT, IMG_WIDTH


# 数据集类定义
class BalancedGrayscaleDataset(Dataset):
    def __init__(self, image_dir, transform, noise_factor=0.3):
        self.image_dir = image_dir
        self.transform = transform
        self.noise_factor = noise_factor
        self.image_names = [f for f in os.listdir(image_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        print(f"找到 {len(self.image_names)} 张图片")

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(image_path).convert('L')

        if self.transform:
            clean_img = self.transform(image)

        # 自适应噪声：在白色区域添加更少的噪声
        noise = torch.randn_like(clean_img) * self.noise_factor

        # 根据像素强度调整噪声
        noise_mask = clean_img.clamp(0, 1)  # 白色区域会有更大的值
        adaptive_noise = noise * (1 - noise_mask * 0.5)  # 白色区域噪声减半

        noisy_img = clean_img + adaptive_noise
        noisy_img = torch.clamp(noisy_img, 0., 1.)

        return noisy_img, clean_img

    def __len__(self):
        return len(self.image_names)

# if __name__ == "__main__":
#     image_names = os.listdir(IMG_PATH)
#
#
#     transform = T.Compose([
#         T.Resize((IMG_HEIGHT, IMG_WIDTH)),
#         T.ToTensor()
#     ])
#
# dataset = BalancedGrayscaleDataset(IMG_PATH, transform = transform)
# print(len(dataset))

