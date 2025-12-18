# Copyright (C) 2025 ntskwk
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from pathlib import Path

import itertools

import cv2
import numpy as np
from tqdm import tqdm

ds_type = ["Train", "Val"]
data_type = ["Rural", "Urban"]
img_type = ["images_png", "masks_png"]


def resize(img: np.ndarray, scale: float = 0.25) -> np.ndarray:
    """
    Resize the input image by a scale factor.

    Args:
        img (np.ndarray): Input image.
        scale (float): Scale factor. Defaults to 0.25 (quarter size).

    Returns:
        np.ndarray: Resized image.
    """
    height, width = img.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Use INTER_AREA for shrinking (better quality), INTER_LINEAR for enlarging
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

    return cv2.resize(img, (new_width, new_height), interpolation=interpolation)


def main():
    for ds, dt, it in tqdm(list(itertools.product(ds_type, data_type, img_type))):
        target_dir = Path(f"dataset/{ds}/{dt}/{it}")
        dist_dir = Path(str(target_dir) + "_resized")
        dist_dir.mkdir(exist_ok=True)

        files = list(Path(target_dir).glob("*.png"))
        for file in tqdm(files):
            img = cv2.imread(str(file))
            resized_img = resize(img)  # ty:ignore[invalid-argument-type]
            cv2.imwrite(str(dist_dir / file.name), resized_img)


if __name__ == "__main__":
    main()
