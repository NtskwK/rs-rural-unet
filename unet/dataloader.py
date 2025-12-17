from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class LoveDA(Dataset):
    def __init__(self, data_dir: Path, label_dir: Path):
        self.data_paths = list(data_dir.glob('*.png'))
        self.label_paths = list(label_dir.glob('*.png'))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        label_path = self.label_paths[idx]
        img = cv2.imread(str(data_path), cv2.IMREAD_COLOR)
        data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

        # 转换为tensor
        data = torch.from_numpy(data).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label).long()

        return data, label
