from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from unet.dataloader import LoveDA
from unet.loss import CombinedLoss  # 导入自定义损失函数
from unet.net import UNet
from unet.train import train


def main():
    # 创建数据集对象
    train_rural_dataset = LoveDA(Path("dataset/Train/Rural/images_png_resized"),
                                 Path("dataset/Train/Rural/masks_png_resized"))
    train_urban_dataset = LoveDA(Path("dataset/Train/Urban/images_png_resized"),
                                 Path("dataset/Train/Urban/masks_png_resized"))
    train_dataset = torch.utils.data.ConcatDataset([train_rural_dataset, train_urban_dataset])
    val_rural_dataset = LoveDA(Path("dataset/Val/Rural/images_png_resized"),
                               Path("dataset/Val/Rural/masks_png_resized"))
    val_urban_dataset = LoveDA(Path("dataset/Val/Urban/images_png_resized"),
                               Path("dataset/Val/Urban/masks_png_resized"))
    val_dataset = torch.utils.data.ConcatDataset([val_rural_dataset, val_urban_dataset])

    # 使用DataLoader包装数据集，并设置batch_size和num_workers
    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(3, 3).to(device)  # 修改输出通道数为3（对应3个类别）
    # 使用自定义的CombinedLoss替代binary_cross_entropy_with_logits
    loss = CombinedLoss(num_classes=3).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)

    net.train()

    train(net, train_dataloader, device, val_dataloader, loss, optimizer, num_epochs=20)


if __name__ == "__main__":
    main()
