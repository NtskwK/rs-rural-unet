import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-6):
        """
        alpha: Focal Loss 的权重因子，平衡正负样本
        gamma: Focal Loss 的难易样本调节因子
        smooth: 防止 Dice 分母为 0 的平滑项
        """
        super(DiceFocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.smooth: float = smooth

    def forward(self, logits, targets):
        # 1. 准备工作：确保概率分布
        # 如果 targets 是类索引 (B, H, W)，转换为 one-hot (B, C, H, W)
        if logits.dim() != targets.dim():
            targets = (
                F.one_hot(targets, num_classes=logits.shape[1])
                .permute(0, 3, 1, 2)
                .float()
            )

        probs = torch.sigmoid(logits)

        # 将 targets 拉平处理，确保计算不受 batch/空间维度限制
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)
        probs_flat = probs.view(-1)

        # --- Focal Loss 计算 ---
        # 使用 binary_cross_entropy_with_logits 保证数值稳定性
        BCE = F.binary_cross_entropy_with_logits(
            logits_flat, targets_flat, reduction="none"
        )
        pt = torch.exp(-BCE)  # pt 是模型预测正确的概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE
        focal_loss = focal_loss.mean()

        # --- Dice Loss 计算 ---
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1 - dice_score

        # --- 合并 ---
        # 常见的经验配比是 1:1，也可以根据需求调整权重
        return focal_loss + dice_loss


def main():
    # --- 使用示例 ---
    # 假设有 3 类，给第 2、3 类（小物体）更高权重
    weights = torch.tensor([1.0, 5.0, 10.0]).cuda()
    criterion = DiceFocalLoss().cuda()

    # 模拟输入: [Batch=2, Classes=3, H=64, W=64]
    outputs = torch.randn(2, 3, 64, 64).cuda()
    labels = torch.randint(0, 3, (2, 64, 64)).cuda()

    loss = criterion(outputs, labels)
    print(f"Total Loss: {loss.item()}")


if __name__ == "__main__":
    main()
