"""ONNX Runtime inference for the exported SR model.

Mirrors infer.py but runs through ONNX Runtime instead of PyTorch.
Uses the same seam-free tiled strategy for large inputs.

Usage:
    python -m ml.infer_onnx --ONNX_PATH artifacts/sr_model.onnx \
        --INPUT_DIR data/lr --OUTPUT_DIR outputs/onnx --UPSCALE 2
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from utils.tiling import feather_mask

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def make_session(onnx_path: str, prefer_gpu: bool = True) -> ort.InferenceSession:
    providers = []
    available = ort.get_available_providers()
    if prefer_gpu:
        for gpu_ep in ("TensorrtExecutionProvider", "CUDAExecutionProvider"):
            if gpu_ep in available:
                providers.append(gpu_ep)
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(onnx_path, providers=providers)
    return sess


def infer_tiled_onnx(sess: ort.InferenceSession, lr_rgb: np.ndarray,
                     scale: int = 2, tile: int = 512, overlap: int = 32) -> np.ndarray:
    """Same feathered tiling as utils.tiling.infer_tiled, but through ORT."""
    H, W, _ = lr_rgb.shape
    out_h, out_w = H * scale, W * scale
    sr_acc = np.zeros((out_h, out_w, 3), dtype=np.float32)
    w_acc = np.zeros((out_h, out_w), dtype=np.float32)

    step = tile - overlap * 2
    if step <= 0:
        raise ValueError("tile must be > 2*overlap")

    for y in range(0, H, step):
        for x in range(0, W, step):
            y1 = min(y + tile, H)
            x1 = min(x + tile, W)
            lr_tile = lr_rgb[y:y1, x:x1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            sr_tile = sess.run(["sr"], {"lr": lr_tile})[0]
            sr_tile = np.clip(sr_tile[0], 0, 1).transpose(1, 2, 0)

            th, tw = sr_tile.shape[:2]
            Y0, X0 = y * scale, x * scale
            mask = feather_mask(th, tw)[..., None]
            sr_acc[Y0:Y0 + th, X0:X0 + tw] += sr_tile * mask
            w_acc[Y0:Y0 + th, X0:X0 + tw] += mask[..., 0]

    w_acc = np.maximum(w_acc, 1e-6)
    sr = sr_acc / w_acc[..., None]
    return np.clip(sr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ONNX_PATH", type=str, required=True)
    ap.add_argument("--INPUT_DIR", type=str, required=True)
    ap.add_argument("--OUTPUT_DIR", type=str, required=True)
    ap.add_argument("--UPSCALE", type=int, default=2)
    ap.add_argument("--TILE", type=int, default=512)
    ap.add_argument("--OVERLAP", type=int, default=32)
    ap.add_argument("--CPU_ONLY", action="store_true")
    args = ap.parse_args()

    sess = make_session(args.ONNX_PATH, prefer_gpu=not args.CPU_ONLY)
    print("Execution providers:", sess.get_providers())

    in_dir, out_dir = Path(args.INPUT_DIR), Path(args.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(in_dir.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        lr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sr = infer_tiled_onnx(sess, lr, scale=args.UPSCALE, tile=args.TILE, overlap=args.OVERLAP)
        cv2.imwrite(str(out_dir / p.name), cv2.cvtColor(sr, cv2.COLOR_RGB2BGR))
        print("Wrote", out_dir / p.name)


if __name__ == "__main__":
    main()
