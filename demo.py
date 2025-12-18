from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from unet.dataloader import LoveDA
from unet.loss import CombinedLoss  # 导入自定义损失函数
from unet.net import UNet
from unet.train import train


def main():
    print("Initializing datasets...")
    # 创建数据集对象
    train_rural_dataset = LoveDA(
        Path("dataset/Train/Rural/images_png_resized"),
        Path("dataset/Train/Rural/masks_png_resized"),
    )
    train_urban_dataset = LoveDA(
        Path("dataset/Train/Urban/images_png_resized"),
        Path("dataset/Train/Urban/masks_png_resized"),
    )
    train_dataset = torch.utils.data.ConcatDataset(
        [train_rural_dataset, train_urban_dataset]
    )
    val_rural_dataset = LoveDA(
        Path("dataset/Val/Rural/images_png_resized"),
        Path("dataset/Val/Rural/masks_png_resized"),
    )
    val_urban_dataset = LoveDA(
        Path("dataset/Val/Urban/images_png_resized"),
        Path("dataset/Val/Urban/masks_png_resized"),
    )
    val_dataset = torch.utils.data.ConcatDataset([val_rural_dataset, val_urban_dataset])

    # 使用DataLoader包装数据集，并设置batch_size和num_workers
    print("Initializing dataloaders...")
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, num_workers=4, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=4)

    print("Initializing model...")
    if torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device("cuda")
    else:
        print("Using CPU...")
        device = torch.device("cpu")

    net = UNet(3, 8).to(device)

    # 根据实际情况调整类别数量和预设权重
    # 无效      0
    # 背景	    1
    # 建筑物	2
    # 道路	    3
    # 水体	    4
    # 荒地	    5
    # 森林	    6
    # 农业用地	7
    print("Initializing loss...")
    loss = CombinedLoss(
        weight=torch.tensor([0.0, 1.0, 5.0, 8.0, 10.0, 1.0, 1.0, 1.0]), num_classes=8
    ).to(device)

    print("Initializing optimizer...")
    optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)

    print("Training...")
    net.train()
    train(net, train_dataloader, device, val_dataloader, loss, optimizer, num_epochs=20)


if __name__ == "__main__":
    main()
