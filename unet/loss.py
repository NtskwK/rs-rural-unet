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
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)

        # 对于多类别分割，使用 softmax 而不是 sigmoid
        if self.num_classes > 1:
            # 使用 CrossEntropyLoss 兼容的格式
            if logits.shape[1] == 1:  # 单通道输出
                # 二分类情况
                probs = torch.sigmoid(logits)
                valid_probs = probs[:, 0][valid_mask]  # 取第一个通道
                valid_targets_float = targets[valid_mask].float()
                
                BCE = F.binary_cross_entropy_with_logits(
                    logits[:, 0][valid_mask],
                    valid_targets_float,
                    reduction="none"
                )

                pt = torch.exp(-BCE)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE
                focal_loss = focal_loss.mean()

                # Dice loss
                intersection = (valid_probs * valid_targets_float).sum()
                dice_score = (2.0 * intersection + self.smooth) / (
                        valid_probs.sum() + valid_targets_float.sum() + self.smooth
                )
                dice_loss = 1 - dice_score

                return focal_loss + dice_loss
            else:
                # 多分类情况
                # targets 应该是 (B, H, W) 格式，包含类别索引
                valid_targets = targets[valid_mask]
                # 对logits进行相应筛选
                valid_logits = logits.permute(0, 2, 3, 1)[valid_mask]  # (B, C, H, W) -> (B, H, W, C) -> (N, C)

                if valid_logits.numel() == 0:
                    return torch.tensor(0.0, device=logits.device, requires_grad=True)
                
                ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                focal_loss = focal_loss.mean()

                # 对每个类别计算 Dice loss
                dice_loss_total = 0
                valid_logits_softmax = F.softmax(valid_logits, dim=1)

                effective_classes = 0
                for class_idx in range(self.num_classes):
                    if class_idx == self.ignore_index:
                        continue

                    # 创建当前类别的 mask
                    class_mask = (valid_targets == class_idx).float()
                    if class_mask.sum() == 0:  # 如果这个类别没有像素，跳过
                        continue

                    class_probs = valid_logits_softmax[:, class_idx]
                    
                    intersection = (class_probs * class_mask).sum()
                    union = class_probs.sum() + class_mask.sum()
                    dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
                    dice_loss_total += 1 - dice_score
                    effective_classes += 1

                dice_loss = dice_loss_total / effective_classes if effective_classes > 0 else torch.tensor(0.0,
                                                                                                           device=logits.device,
                                                                                                           requires_grad=True)
                
                return focal_loss + dice_loss
        else:
            # 原始逻辑保持不变（适用于二分类）
            valid_logits = logits[valid_mask]
            valid_targets = targets[valid_mask]

            if valid_logits.numel() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

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


def main():
    # --- 使用示例 ---
    # 假设有 3 类，给第 2、3 类（小物体）更高权重
    weights = torch.tensor([1.0, 5.0, 10.0]).cuda()
    criterion = DiceFocalLoss(num_classes=8, ignore_index=0).cuda()

    # 模拟输入: [Batch=2, Classes=8, H=64, W=64]
    outputs = torch.randn(2, 8, 64, 64).cuda()
    labels = torch.randint(0, 8, (2, 64, 64)).cuda()  # 包含0-7的标签

    loss = criterion(outputs, labels)
    print(f"Total Loss: {loss.item()}")


if __name__ == "__main__":
    main()