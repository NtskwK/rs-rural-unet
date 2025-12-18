from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2
from torchvision import tv_tensors


class LoveDA(Dataset):
    def __init__(
        self,
        data_dir: Path,
        label_dir: Path,
        transform: transforms.Compose | v2.Compose | None = None,
    ):
        self.data_paths = sorted(list(data_dir.glob("*.png")))
        self.label_paths = sorted(list(label_dir.glob("*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        label_path = self.label_paths[index]
        img = cv2.imread(str(data_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {data_path}")
        data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            # 将 numpy 数组转换为 tv_tensors，以便 v2 transforms 能够识别图像和掩码
            # Image 需要是 (C, H, W) 且通常为 float [0, 1] 以便后续处理
            data_tensor = torch.from_numpy(data).permute(2, 0, 1).float() / 255.0
            data_t = tv_tensors.Image(data_tensor)
            # Mask 可以是 (H, W) 或 (1, H, W)
            label_t = tv_tensors.Mask(torch.from_numpy(label))

            data, label = self.transform(data_t, label_t)
        else:
            data = torch.from_numpy(data).permute(2, 0, 1).float() / 255.0
            label = torch.from_numpy(label).long()

        # 确保 label 是 LongTensor 且没有多余的维度 (如果有的话)
        if isinstance(label, torch.Tensor):
            label = label.long().squeeze()

        return data, label
