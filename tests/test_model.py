"""Sanity checks for the SR model, ONNX export, and metrics.

Run with: .venv/bin/python -m pytest tests/ -v
All tests are CPU-only and need no checkpoint or GPU.
"""
import numpy as np
import pytest
import torch

from models.sr_residual import SRResCNN


@pytest.fixture(scope="module")
def model():
    m = SRResCNN(channels=32, num_blocks=2, scale=2, use_bn=False)
    m.eval()
    return m


def test_forward_shape(model):
    x = torch.randn(1, 3, 64, 48)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 128, 96)


def test_scale_4():
    m = SRResCNN(channels=16, num_blocks=1, scale=4, use_bn=False).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 3, 128, 128)


def test_onnx_export_parity(model, tmp_path):
    from ml.export_onnx import export, verify

    onnx_path = export(model, tmp_path / "sr.onnx", sample_hw=64)
    assert onnx_path.exists()
    diff = verify(onnx_path, model, sample_hw=80)  # different size than export -> dynamic axes work
    assert diff < 1e-3, f"PyTorch/ONNX mismatch: {diff}"


def test_tiled_inference_matches_direct(model):
    """Tiled inference should closely match a single full-frame pass."""
    from utils.tiling import infer_tiled

    rng = np.random.default_rng(0)
    lr = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)

    sr_tiled = infer_tiled(model, lr, scale=2, tile=64, overlap=16, amp=False, device="cpu")
    with torch.no_grad():
        x = torch.from_numpy(lr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        direct = model(x).squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)
    sr_direct = np.clip(direct * 255.0 + 0.5, 0, 255).astype(np.uint8)

    assert sr_tiled.shape == sr_direct.shape == (192, 192, 3)
    mean_diff = np.abs(sr_tiled.astype(float) - sr_direct.astype(float)).mean()
    assert mean_diff < 2.0, f"tiled vs direct mean abs diff too high: {mean_diff}"


def test_metrics_identity():
    from utils.metrics import psnr_rgb, ssim_rgb

    img = np.random.default_rng(1).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    assert psnr_rgb(img, img) == float("inf")
    assert ssim_rgb(img, img) == pytest.approx(1.0)


def test_metrics_degrade():
    from utils.data import downscale, upscale
    from utils.metrics import psnr_rgb

    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    bic = upscale(downscale(img, 2), 2)
    p = psnr_rgb(img, bic)
    assert 0 < p < 60  # degraded but finite
