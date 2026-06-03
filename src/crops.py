"""H&E crop utilities shared by the feature extractors."""
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from config import HE_DIR

CROP = 224
HALF = CROP // 2
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def read_he(sample):
    """Load an aligned H&E as an HWC uint8 array, taking the first full-resolution page."""
    img = tifffile.imread(HE_DIR / f"{sample}.tif")
    if img.ndim == 4:                                   # (page, H, W, C) -> largest page
        img = max(img, key=lambda p: p.shape[0] * p.shape[1])
    if img.ndim == 2:                                   # grayscale -> RGB
        img = np.repeat(img[..., None], 3, axis=2)
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[0] < img.shape[-1]:
        img = np.moveaxis(img, 0, -1)                   # CHW -> HWC
    return img[..., :3]


def crop_window(image, x, y):
    """Return a CROP x CROP x 3 uint8 patch centered at (x, y), zero-padded at borders."""
    H, W = image.shape[:2]
    x, y = int(x), int(y)
    x0, y0 = max(x - HALF, 0), max(y - HALF, 0)
    x1, y1 = min(x + HALF, W), min(y + HALF, H)
    patch = np.zeros((CROP, CROP, 3), dtype=np.uint8)
    crop = image[y0:y1, x0:x1]
    patch[: crop.shape[0], : crop.shape[1]] = crop
    return patch


def normalize_crop(patch):
    t = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
    return (t - MEAN) / STD


class CellCrops(Dataset):
    """Normalized crops for the foundation model; yields a tensor, or (tensor, label)."""
    def __init__(self, image, xs, ys, labels=None):
        self.image = image
        self.xs = np.asarray(xs).astype(int)
        self.ys = np.asarray(ys).astype(int)
        self.labels = labels

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        t = normalize_crop(crop_window(self.image, self.xs[i], self.ys[i]))
        if self.labels is None:
            return t
        return t, int(self.labels[i])
