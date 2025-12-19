import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceFocalLoss(nn.Module):
    def __init__(self, num_classes=1, alpha=0.25, gamma=2.0, smooth=1e-6, ignore_index=-100):
        """
        alpha: Focal Loss 的权重因子，平衡正负样本
        gamma: Focal Loss 的难易样本调节因子
        smooth: 防止 Dice 分母为 0 的平滑项
        ignore_index: 忽略的类别索引，默认-100表示不忽略任何类别
        """
        super(DiceFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.smooth: float = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # 处理忽略索引
        if self.ignore_index != -100:
            # 创建非忽略位置的mask
            valid_mask = targets != self.ignore_index

            # 如果所有像素都被忽略，则返回0
            if not valid_mask.any():
                return torch.tensor(0.0, device=logits.device)

            # 只保留有效的像素
            valid_targets = targets[valid_mask]
            valid_logits = logits[valid_mask.unsqueeze(0).expand_as(logits)].view(-1, *logits.shape[1:])

            # 如果logits的第一维是类别数，则需要特殊处理
            if valid_logits.shape[0] == valid_mask.sum() * logits.shape[1]:
                valid_logits = valid_logits.view(logits.shape[1], -1).t().contiguous().view(-1, logits.shape[1])
        else:
            valid_targets = targets
            valid_logits = logits

        # 对于多类别分割，使用 softmax 而不是 sigmoid
        if self.num_classes > 1:
            # 使用 CrossEntropyLoss 兼容的格式
            if logits.shape[1] == 1:  # 单通道输出
                # 二分类情况
                probs = torch.sigmoid(valid_logits)
                BCE = F.binary_cross_entropy_with_logits(
                    valid_logits, valid_targets.float(), reduction="none"
                )

                pt = torch.exp(-BCE)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE
                focal_loss = focal_loss.mean()

                # Dice loss
                intersection = (probs * valid_targets.float()).sum()
                dice_score = (2.0 * intersection + self.smooth) / (
                        probs.sum() + valid_targets.float().sum() + self.smooth
                )
                dice_loss = 1 - dice_score

                return focal_loss + dice_loss
            else:
                # 多分类情况
                # valid_targets 应该是 (B, H, W) 格式，包含类别索引
                ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                focal_loss = focal_loss.mean()

                # 对每个类别计算 Dice loss
                dice_loss_total = 0
                valid_logits_softmax = F.softmax(valid_logits, dim=1)
                classes_to_consider = self.num_classes
                if self.ignore_index != -100 and self.ignore_index < self.num_classes:
                    classes_to_consider -= 1  # 排除被忽略的类别

                for class_idx in range(self.num_classes):
                    if class_idx == self.ignore_index:
                        continue

                    # 创建当前类别的 mask
                    class_mask = (valid_targets == class_idx).float()
                    class_probs = valid_logits_softmax[:, class_idx,
                                  ...] if valid_logits_softmax.dim() > 3 else valid_logits_softmax[:, class_idx]

                    intersection = (class_probs * class_mask).sum()
                    union = class_probs.sum() + class_mask.sum()
                    dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
                    dice_loss_total += 1 - dice_score

                dice_loss = dice_loss_total / classes_to_consider if classes_to_consider > 0 else 0

                return focal_loss + dice_loss
        else:
            # 原始逻辑保持不变
            if valid_logits.dim() != valid_targets.dim():
                valid_targets = (
                    F.one_hot(valid_targets, num_classes=valid_logits.shape[1])
                    .permute(0, 3, 1, 2)
                    .float()
                )

            probs = torch.sigmoid(valid_logits)

            # 将 targets 拉平处理，确保计算不受 batch/空间维度限制
            logits_flat = valid_logits.contiguous().view(-1)
            targets_flat = valid_targets.contiguous().view(-1)
            probs_flat = probs.contiguous().view(-1)

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