import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def psnr_rgb(hr: np.ndarray, sr: np.ndarray) -> float:
    return float(psnr(hr, sr, data_range=255))

def ssim_rgb(hr: np.ndarray, sr: np.ndarray) -> float:
    return float(ssim(hr, sr, channel_axis=2, data_range=255))
