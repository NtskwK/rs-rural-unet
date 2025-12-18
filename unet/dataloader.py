from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LoveDA(Dataset):
    def __init__(
        self,
        data_dir: Path,
        label_dir: Path,
        transform: transforms.Compose | None = None,
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

        # 转换为tensor
        data = torch.from_numpy(data).permute(2, 0, 1).float()
        label = torch.from_numpy(label).long()

        if self.transform:
            data = self.transform(data)
        else:
            data = torch.tensor(data, dtype=torch.float32)

        return data, label
