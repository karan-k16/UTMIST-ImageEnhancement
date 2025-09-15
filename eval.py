import argparse, csv
from pathlib import Path
import numpy as np
import cv2
import torch

from models.sr_residual import SRResCNN
from utils.data import find_images_recursive, to_rgb, downscale, upscale
from utils.metrics import psnr_rgb, ssim_rgb
from utils.tiling import infer_tiled

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--WEIGHTS_PATH", type=str, required=True)
    ap.add_argument("--VAL_HR_DIR", type=str, required=True)
    ap.add_argument("--UPSCALE", type=int, default=2)
    ap.add_argument("--TILE", type=int, default=512)
    ap.add_argument("--OVERLAP", type=int, default=32)
    ap.add_argument("--AMP", action="store_true")
    ap.add_argument("--DEVICE", type=str, default="cuda")
    ap.add_argument("--DOWNSCALE_KERNEL", type=str, default="bicubic")
    ap.add_argument("--OUT_CSV", type=str, default="outputs/bench/val_metrics.csv")
    ap.add_argument("--CHANNELS", type=int, default=64)
    ap.add_argument("--NUM_BLOCKS", type=int, default=8)
    ap.add_argument("--USE_BN", action="store_true")
    args = ap.parse_args()

    DEVICE = args.DEVICE if (args.DEVICE == "cpu" or torch.cuda.is_available()) else "cpu"
    model = SRResCNN(channels=args.CHANNELS, num_blocks=args.NUM_BLOCKS, scale=args.UPSCALE, use_bn=args.USE_BN).to(DEVICE)
    state = torch.load(args.WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state)

    files = find_images_recursive(Path(args.VAL_HR_DIR))

    rows = []
    psnr_m_all, psnr_b_all = [], []
    ssim_m_all, ssim_b_all = [], []

    for idx, fp in enumerate(files):
        hr_bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
        hr = to_rgb(hr_bgr)
        lr = downscale(hr, args.UPSCALE, kernel_name=args.DOWNSCALE_KERNEL)
        bic = upscale(lr, args.UPSCALE, kernel_name=args.DOWNSCALE_KERNEL)
        sr  = infer_tiled(model, lr, scale=args.UPSCALE, tile=args.TILE, overlap=args.OVERLAP, amp=args.AMP, device=DEVICE)

        p_m = psnr_rgb(hr, sr);   s_m = ssim_rgb(hr, sr)
        p_b = psnr_rgb(hr, bic);  s_b = ssim_rgb(hr, bic)

        rows.append({
            "idx": idx,
            "file": fp,
            "psnr_model": round(p_m, 4),
            "psnr_bicubic": round(p_b, 4),
            "delta_psnr": round(p_m - p_b, 4),
            "ssim_model": round(s_m, 4),
            "ssim_bicubic": round(s_b, 4),
            "delta_ssim": round(s_m - s_b, 5),
        })

        psnr_m_all.append(p_m); psnr_b_all.append(p_b)
        ssim_m_all.append(s_m); ssim_b_all.append(s_b)

    out_csv = Path(args.OUT_CSV); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        import csv as _csv
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    p_m = float(np.mean(psnr_m_all)); p_b = float(np.mean(psnr_b_all))
    s_m = float(np.mean(ssim_m_all)); s_b = float(np.mean(ssim_b_all))
    print(f"Avg PSNR model = {p_m:.3f} dB | bicubic = {p_b:.3f} dB | Δ = {p_m - p_b:.3f} dB")
    print(f"Avg SSIM model = {s_m:.4f}   | bicubic = {s_b:.4f}   | Δ = {s_m - s_b:.4f}")
    print("Wrote:", out_csv)

if __name__ == "__main__":
    main()
