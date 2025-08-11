import torch
import torch.nn as nn
import torch.nn.functional as F


# 改进的损失函数
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, num_channels=3):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

        # 预定义边缘检测卷积核，支持多通道
        kernel = torch.tensor([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]], dtype=torch.float32)
        # 扩展为多通道卷积核
        self.register_buffer('kernel', kernel.view(1, 1, 3, 3).repeat(num_channels, 1, 1, 1))

    def forward(self, output, target):
        # 结合MSE和L1损失
        mse_loss = self.mse(output, target)
        l1_loss = self.l1(output, target)

        # 边缘保持损失
        output_edges = self.get_edges(output)
        target_edges = self.get_edges(target)
        edge_loss = self.mse(output_edges, target_edges)

        return mse_loss + self.alpha * l1_loss + 0.1 * edge_loss

    def get_edges(self, x):
        # 多通道边缘检测
        edges = F.conv2d(x, self.kernel, padding=1, groups=x.shape[1])
        return edges


# 确保device变量被正确定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化自定义损失函数，指定通道数（RGB为3，灰度图为1）
loss = CombinedLoss(alpha=0.5, num_channels=3).to(device)  # 根据实际数据调整通道数


# 训练和测试函数
def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        # 假设batch包含输入和目标
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def test_step(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item()

    return total_loss / len(dataloader)