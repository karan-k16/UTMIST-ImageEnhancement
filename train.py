import argparse, os, random, time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.data import SRDataset, find_images_recursive, to_rgb, downscale, upscale
from utils.metrics import psnr_rgb, ssim_rgb
from utils.tiling import infer_tiled
from models.sr_residual import SRResCNN

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def validate(model, VAL_HR_DIR, UPSCALE, DEVICE="cuda", TILE=512, OVERLAP=32, AMP=True, DOWNSCALE_KERNEL="bicubic"):
    model.eval()
    files = find_images_recursive(Path(VAL_HR_DIR))
    psnrs, ssims = [], []
    for fp in files:
        hr_bgr = cv2.imread(fp, cv2.IMREAD_COLOR); hr = to_rgb(hr_bgr)
        lr = downscale(hr, UPSCALE, kernel_name=DOWNSCALE_KERNEL)
        bic = upscale(lr, UPSCALE, kernel_name=DOWNSCALE_KERNEL)
        sr  = infer_tiled(model, lr, scale=UPSCALE, tile=TILE, overlap=OVERLAP, amp=AMP, device=DEVICE)
        psnrs.append(psnr_rgb(hr, sr)); ssims.append(ssim_rgb(hr, sr))
    return float(np.mean(psnrs)), float(np.mean(ssims))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--TRAIN_HR_DIR", type=str, required=True)
    ap.add_argument("--VAL_HR_DIR",   type=str, required=True)
    ap.add_argument("--UPSCALE", type=int, default=2)
    ap.add_argument("--PATCH_SIZE", type=int, default=128)
    ap.add_argument("--BATCH_SIZE", type=int, default=16)
    ap.add_argument("--EPOCHS", type=int, default=50)
    ap.add_argument("--LR", type=float, default=1e-4)
    ap.add_argument("--NUM_WORKERS", type=int, default=2)
    ap.add_argument("--DOWNSCALE_KERNEL", type=str, default="bicubic")
    ap.add_argument("--AMP", action="store_true")
    ap.add_argument("--SAVE_DIR", type=str, default="runs")
    ap.add_argument("--DEVICE", type=str, default="cuda")
    ap.add_argument("--CHANNELS", type=int, default=64)
    ap.add_argument("--NUM_BLOCKS", type=int, default=8)
    ap.add_argument("--USE_BN", action="store_true")
    ap.add_argument("--TILE", type=int, default=512)
    ap.add_argument("--OVERLAP", type=int, default=32)
    args = ap.parse_args()

    set_seed(42)
    DEVICE = args.DEVICE if (args.DEVICE == "cpu" or torch.cuda.is_available()) else "cpu"

    model = SRResCNN(channels=args.CHANNELS, num_blocks=args.NUM_BLOCKS, scale=args.UPSCALE, use_bn=args.USE_BN).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=args.LR)
    scaler = torch.cuda.amp.GradScaler(enabled=args.AMP and DEVICE=="cuda")
    criterion = nn.L1Loss()

    train_ds = SRDataset(Path(args.TRAIN_HR_DIR), patch_size=args.PATCH_SIZE, scale=args.UPSCALE, kernel_name=args.DOWNSCALE_KERNEL, use_patches=True,  recursive=True)
    val_ds   = SRDataset(Path(args.VAL_HR_DIR),   patch_size=args.PATCH_SIZE, scale=args.UPSCALE, kernel_name=args.DOWNSCALE_KERNEL, use_patches=False, recursive=True)

    train_loader = DataLoader(train_ds, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.NUM_WORKERS, pin_memory=True)
    # val_loader (full frames) is iterated inside validate()

    SAVE_DIR = Path(args.SAVE_DIR); SAVE_DIR.mkdir(parents=True, exist_ok=True)
    BEST_PATH = SAVE_DIR/"best.pth"
    best_psnr = -1.0

    for epoch in range(1, args.EPOCHS+1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for (lr, hr, _) in train_loader:
            lr = lr.to(DEVICE, non_blocking=True)
            hr = hr.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    sr = model(lr)
                    loss = criterion(sr, hr)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                sr = model(lr)
                loss = criterion(sr, hr)
                loss.backward()
                opt.step()

            epoch_loss += float(loss.detach().cpu().item())

        val_psnr, val_ssim = validate(
            model,
            args.VAL_HR_DIR,
            args.UPSCALE,
            DEVICE=DEVICE,
            TILE=args.TILE,
            OVERLAP=args.OVERLAP,
            AMP=args.AMP,
            DOWNSCALE_KERNEL=args.DOWNSCALE_KERNEL
        )

        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.EPOCHS} | loss={epoch_loss/len(train_loader):.4f} | val_PSNR={val_psnr:.2f} dB | val_SSIM={val_ssim:.4f} | {dt:.1f}s")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), BEST_PATH)
            print(f"  ↳ saved best: {BEST_PATH} (val_PSNR={best_psnr:.2f} dB)")

    LAST_PATH = SAVE_DIR/"last.pth"
    torch.save(model.state_dict(), LAST_PATH)
    print("Saved final checkpoint:", LAST_PATH)

if __name__ == "__main__":
    main()
