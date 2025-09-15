import os
import glob
from pathlib import Path
from typing import Tuple
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def find_images_recursive(root: Path):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in exts:
            files.append(str(p))
    return sorted(files)

def to_rgb(img):
    # ensure 3-channel RGB np.uint8
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def downscale(hr_rgb: np.ndarray, scale: int, kernel_name: str = "bicubic") -> np.ndarray:
    h, w = hr_rgb.shape[:2]
    lr_w = w // scale
    lr_h = h // scale
    interp = cv2.INTER_CUBIC
    if kernel_name.lower() == "bilinear":
        interp = cv2.INTER_LINEAR
    elif kernel_name.lower() == "nearest":
        interp = cv2.INTER_NEAREST
    return cv2.resize(hr_rgb, (lr_w, lr_h), interpolation=interp)

def upscale(lr_rgb: np.ndarray, scale: int, kernel_name: str = "bicubic") -> np.ndarray:
    h, w = lr_rgb.shape[:2]
    hr_w = w * scale
    hr_h = h * scale
    interp = cv2.INTER_CUBIC
    if kernel_name.lower() == "bilinear":
        interp = cv2.INTER_LINEAR
    elif kernel_name.lower() == "nearest":
        interp = cv2.INTER_NEAREST
    return cv2.resize(lr_rgb, (hr_w, hr_h), interpolation=interp)

class SRDataset(Dataset):
    """
    Patch-based SR dataset.
    - Takes HR images, generates LR on-the-fly via chosen kernel.
    - Returns (lr_tensor, hr_tensor, path) as CHW float32 [0,1].
    """
    def __init__(
        self,
        hr_dir: Path,
        patch_size: int = 128,
        scale: int = 2,
        kernel_name: str = "bicubic",
        use_patches: bool = True,
        recursive: bool = True
    ):
        self.hr_dir = Path(hr_dir)
        self.patch_size = patch_size
        self.scale = scale
        self.kernel_name = kernel_name
        self.use_patches = use_patches

        self.files = find_images_recursive(self.hr_dir) if recursive else glob.glob(str(self.hr_dir/"*"))
        if len(self.files) == 0:
            raise RuntimeError(f"No images found under {self.hr_dir}")

    def __len__(self):
        return len(self.files)

    def _random_patch_pair(self, hr_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H, W = hr_rgb.shape[:2]
        ps = self.patch_size
        y = np.random.randint(0, max(1, H - ps))
        x = np.random.randint(0, max(1, W - ps))
        hr_patch = hr_rgb[y:y+ps, x:x+ps]
        lr_patch = downscale(hr_patch, self.scale, self.kernel_name)
        return lr_patch, hr_patch

    def __getitem__(self, idx):
        img_path = self.files[idx]
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read {img_path}")
        hr_rgb = to_rgb(bgr)

        if self.use_patches:
            lr_rgb, hr_patch = self._random_patch_pair(hr_rgb)
        else:
            lr_rgb = downscale(hr_rgb, self.scale, self.kernel_name)
            hr_patch = hr_rgb

        lr = torch.from_numpy(lr_rgb.transpose(2,0,1)).float() / 255.0
        hr = torch.from_numpy(hr_patch.transpose(2,0,1)).float() / 255.0
        return lr, hr, img_path
