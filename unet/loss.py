import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    带类别权重的 CrossEntropy + 多类别 Dice Loss
    """

    def __init__(self, weight=None, num_classes=3, dice_weight=1.0, ce_weight=1.0):
        """
        Args:
            weight: 类别权重张量, 如 torch.tensor([0.1, 1.0, 2.0])
            num_classes: 类别数量
            dice_weight: Dice Loss 的系数
            ce_weight: CrossEntropy Loss 的系数
        """
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        # 1. 初始化带权重的交叉熵
        # ignore_index 可以根据需要设置，比如忽略背景或填充
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, pred, target):
        """
        pred: 模型输出 [Batch, C, H, W] 的 Raw Logits (不经过 Softmax)
        target: 标签 [Batch, H, W], 元素值为类别索引 (0, 1, ..., C-1)
        """

        # --- [1] 计算 Weighted Cross Entropy ---
        ce_loss = self.ce_loss(pred, target)

        # --- [2] 计算 Multi-class Dice Loss ---
        # 首先将 pred 转换为概率分布
        probs = F.softmax(pred, dim=1)

        # 将 target 转换为 One-hot 编码 [Batch, C, H, W]
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # 计算各个通道的 Dice
        dims = (0, 2, 3)  # 在 Batch, H, W 维度上求和，保留通道 C
        intersection = torch.sum(probs * target_one_hot, dims)
        cardinality = torch.sum(probs + target_one_hot, dims)

        # smooth=1e-6 防止分母为 0
        dice_score = (2. * intersection + 1e-6) / (cardinality + 1e-6)

        # 对所有类别取平均 (Macro Dice)
        dice_loss = 1 - dice_score.mean()

        # --- [3] 加权合并 ---
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        return total_loss


def main():
    # --- 使用示例 ---
    # 假设有 3 类，给第 2、3 类（小物体）更高权重
    weights = torch.tensor([1.0, 5.0, 10.0]).cuda()
    criterion = CombinedLoss(weight=weights, num_classes=3).cuda()

    # 模拟输入: [Batch=2, Classes=3, H=64, W=64]
    outputs = torch.randn(2, 3, 64, 64).cuda()
    labels = torch.randint(0, 3, (2, 64, 64)).cuda()

    loss = criterion(outputs, labels)
    print(f"Total Loss: {loss.item()}")


if __name__ == "__main__":
    main()
