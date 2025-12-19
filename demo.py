from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from unet.dataloader import LoveDA
from unet.loss import DiceFocalLoss
from unet.net import UNet
from unet.train import train

lr = 1e-5
batch_size = 32


def main():
    print("Initializing datasets...")
    # 创建数据集对象
    train_transform = v2.Compose(
        [
            v2.RandomResizedCrop((256, 256), scale=(0.8, 1.2), ratio=(0.75, 1.33)),
            v2.RandomRotation(degrees=(-180.0, 180.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = v2.Compose(
        [
            v2.Resize((256, 256), antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_rural_dataset = LoveDA(
        Path("dataset/Train/Rural/images_png_resized"),
        Path("dataset/Train/Rural/masks_png_resized"),
        transform=train_transform,
    )
    train_urban_dataset = LoveDA(
        Path("dataset/Train/Urban/images_png_resized"),
        Path("dataset/Train/Urban/masks_png_resized"),
        transform=train_transform,
    )
    train_dataset = torch.utils.data.ConcatDataset(
        [train_rural_dataset, train_urban_dataset]
    )
    val_rural_dataset = LoveDA(
        Path("dataset/Val/Rural/images_png_resized"),
        Path("dataset/Val/Rural/masks_png_resized"),
        transform=val_transform,
    )
    val_urban_dataset = LoveDA(
        Path("dataset/Val/Urban/images_png_resized"),
        Path("dataset/Val/Urban/masks_png_resized"),
        transform=val_transform,
    )
    val_dataset = torch.utils.data.ConcatDataset([val_rural_dataset, val_urban_dataset])

    # 使用DataLoader包装数据集，并设置batch_size和num_workers
    print("Initializing dataloaders...")
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    print("Initializing model...")
    if torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device("cuda")
        if torch.backends.cudnn.is_available():
            print("Using cuDNN...")
            torch.backends.cudnn.benchmark = True
    else:
        print("Using CPU...")
        device = torch.device("cpu")

    net = UNet(3, 8).to(device)

    print("Initializing loss...")
    loss = DiceFocalLoss().to(device)

    print("Initializing optimizer...")
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)

    print("Training...")
    net.train()
    train(net, train_dataloader, device, val_dataloader, loss, optimizer, num_epochs=20)

    print("Saving model...")
    torch.save(net.state_dict(), "unet_model.pth")
    print("Model saved to unet_model.pth")


if __name__ == "__main__":
    main()
