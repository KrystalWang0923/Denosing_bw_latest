import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedGrayscaleDenoiser(nn.Module):
    """
    针对512x512黑白工业检测图像优化的U-Net降噪器
    """

    def __init__(self, in_channels=1, init_features=32):
        super(ImprovedGrayscaleDenoiser, self).__init__()

        features = init_features

        # 编码器路径 (512 -> 256 -> 128 -> 64 -> 32 -> 16)
        self.encoder1 = self._block(in_channels, features, name="enc1")  # 512x512
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 256x256

        self.encoder2 = self._block(features, features * 2, name="enc2")  # 256x256
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 128x128

        self.encoder3 = self._block(features * 2, features * 4, name="enc3")  # 128x128
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 64x64

        self.encoder4 = self._block(features * 4, features * 8, name="enc4")  # 64x64
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 32x32

        self.encoder5 = self._block(features * 8, features * 16, name="enc5")  # 32x32
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 16x16

        # 瓶颈层
        self.bottleneck = self._block(features * 16, features * 32, name="bottleneck")  # 16x16

        # 解码器路径
        self.upconv5 = nn.ConvTranspose2d(features * 32, features * 16, kernel_size=2, stride=2)
        self.decoder5 = self._block((features * 16) * 2, features * 16, name="dec5")  # 32x32

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")  # 64x64

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")  # 128x128

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")  # 256x256

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")  # 512x512

        # 输出层
        self.conv_out = nn.Conv2d(features, in_channels, kernel_size=1)

        # Dropout层用于正则化
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        # 保存输入用于残差连接
        identity = x

        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))

        # 瓶颈
        bottleneck = self.bottleneck(self.pool5(enc5))
        bottleneck = self.dropout(bottleneck)

        # 解码器 with skip connections
        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)

        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # 输出
        out = self.conv_out(dec1)

        # 残差连接 + sigmoid激活
        return torch.sigmoid(out + identity)

    def _block(self, in_channels, features, name):
        """创建一个基本的卷积块"""
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )


# 测试代码
# if __name__ == "__main__":
#     # 测试标准模型
#     model = ImprovedGrayscaleDenoiser(in_channels=1, init_features=32)
#     x = torch.randn(1, 1, 512, 512)
#     y = model(x)
#     print(f"标准模型 - 输入: {x.shape}, 输出: {y.shape}")
#
#     # 计算参数量
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"标准模型参数量: {total_params:,}")
#
