"""Quick demo training: fits the SR model on a small image folder so the
demo site has real (non-random) weights. For a serious model, use train.py
with a proper dataset (e.g. DIV2K) on a GPU.

Usage:
    python -m ml.train_demo --TRAIN_HR_DIR data/train --VAL_HR_DIR data/val \
        --STEPS 1500 --OUT weights/demo.pth
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.sr_residual import SRResCNN
from utils.data import SRDataset
from train import validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--TRAIN_HR_DIR", type=str, required=True)
    ap.add_argument("--VAL_HR_DIR", type=str, required=True)
    ap.add_argument("--OUT", type=str, default="weights/demo.pth")
    ap.add_argument("--STEPS", type=int, default=1500)
    ap.add_argument("--BATCH_SIZE", type=int, default=8)
    ap.add_argument("--PATCH_SIZE", type=int, default=96)
    ap.add_argument("--LR", type=float, default=2e-4)
    ap.add_argument("--UPSCALE", type=int, default=2)
    ap.add_argument("--CHANNELS", type=int, default=64)
    ap.add_argument("--NUM_BLOCKS", type=int, default=8)
    ap.add_argument("--VAL_EVERY", type=int, default=300)
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    model = SRResCNN(channels=args.CHANNELS, num_blocks=args.NUM_BLOCKS,
                     scale=args.UPSCALE, use_bn=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.STEPS)
    criterion = nn.L1Loss()

    ds = SRDataset(Path(args.TRAIN_HR_DIR), patch_size=args.PATCH_SIZE,
                   scale=args.UPSCALE, use_patches=True)
    print(f"Training on {len(ds.files)} images, {args.STEPS} steps, device={device}")

    def batch():
        idxs = np.random.randint(0, len(ds), size=args.BATCH_SIZE)
        pairs = [ds[i] for i in idxs]
        lr = torch.stack([p[0] for p in pairs]).to(device)
        hr = torch.stack([p[1] for p in pairs]).to(device)
        return lr, hr

    out_path = Path(args.OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_psnr = -1.0
    t0 = time.time()

    model.train()
    for step in range(1, args.STEPS + 1):
        lr, hr = batch()
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(lr), hr)
        loss.backward()
        opt.step()
        sched.step()

        if step % 100 == 0:
            print(f"step {step:5d}/{args.STEPS} | loss={loss.item():.4f} "
                  f"| {(time.time() - t0):.0f}s elapsed", flush=True)

        if step % args.VAL_EVERY == 0 or step == args.STEPS:
            psnr, ssim = validate(model, args.VAL_HR_DIR, args.UPSCALE,
                                  DEVICE=device, AMP=False)
            print(f"  val PSNR={psnr:.2f} dB SSIM={ssim:.4f}", flush=True)
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), out_path)
                print(f"  saved {out_path} (best PSNR {best_psnr:.2f} dB)", flush=True)
            model.train()

    print(f"Done in {(time.time() - t0):.0f}s. Best val PSNR: {best_psnr:.2f} dB -> {out_path}")


if __name__ == "__main__":
    main()
