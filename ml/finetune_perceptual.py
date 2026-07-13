"""Perceptual fine-tune: continue training from a PSNR-trained checkpoint with
an added VGG feature loss, which sharpens perceived detail (textures, edges)
at a small cost in PSNR. This is the standard SRResNet -> perceptual step,
without the full GAN machinery.

Usage:
    python -m ml.finetune_perceptual --INIT weights/demo_c96b16.pth \
        --TRAIN_HR_DIR data/div2k/train_full --VAL_HR_DIR data/div2k/val \
        --CHANNELS 96 --NUM_BLOCKS 16 --STEPS 6000 --OUT weights/demo_perc.pth
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

from models.sr_residual import SRResCNN
from utils.data import SRDataset
from train import validate


class VGGFeatureLoss(nn.Module):
    """L1 distance in VGG19 relu3_3 feature space (features[:17])."""

    def __init__(self, device):
        super().__init__()
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)
        self.features = vgg.features[:17].to(device).eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, sr, hr):
        sr = (sr.clamp(0, 1) - self.mean) / self.std
        hr = (hr - self.mean) / self.std
        return nn.functional.l1_loss(self.features(sr), self.features(hr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--INIT", type=str, required=True)
    ap.add_argument("--TRAIN_HR_DIR", type=str, required=True)
    ap.add_argument("--VAL_HR_DIR", type=str, required=True)
    ap.add_argument("--OUT", type=str, default="weights/demo_perc.pth")
    ap.add_argument("--STEPS", type=int, default=6000)
    ap.add_argument("--BATCH_SIZE", type=int, default=8)
    ap.add_argument("--PATCH_SIZE", type=int, default=128)
    ap.add_argument("--LR", type=float, default=2e-5)
    ap.add_argument("--PERC_WEIGHT", type=float, default=0.08)
    ap.add_argument("--UPSCALE", type=int, default=2)
    ap.add_argument("--CHANNELS", type=int, default=96)
    ap.add_argument("--NUM_BLOCKS", type=int, default=16)
    ap.add_argument("--VAL_EVERY", type=int, default=1000)
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
    model.load_state_dict(torch.load(args.INIT, map_location=device))

    perc = VGGFeatureLoss(device)
    l1 = nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=args.LR)

    ds = SRDataset(Path(args.TRAIN_HR_DIR), patch_size=args.PATCH_SIZE,
                   scale=args.UPSCALE, use_patches=True)
    print(f"Fine-tuning from {args.INIT} on {len(ds.files)} images, "
          f"{args.STEPS} steps, device={device}", flush=True)

    def batch():
        idxs = np.random.randint(0, len(ds), size=args.BATCH_SIZE)
        pairs = [ds[i] for i in idxs]
        return (torch.stack([p[0] for p in pairs]).to(device),
                torch.stack([p[1] for p in pairs]).to(device))

    out_path = Path(args.OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    model.train()
    for step in range(1, args.STEPS + 1):
        lr_b, hr_b = batch()
        opt.zero_grad(set_to_none=True)
        sr = model(lr_b)
        loss_l1 = l1(sr, hr_b)
        loss_p = perc(sr, hr_b)
        loss = loss_l1 + args.PERC_WEIGHT * loss_p
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"step {step:5d}/{args.STEPS} | l1={loss_l1.item():.4f} "
                  f"perc={loss_p.item():.4f} | {(time.time() - t0):.0f}s", flush=True)

        if step % args.VAL_EVERY == 0 or step == args.STEPS:
            psnr, ssim = validate(model, args.VAL_HR_DIR, args.UPSCALE,
                                  DEVICE=device, AMP=False)
            print(f"  val PSNR={psnr:.2f} dB SSIM={ssim:.4f}", flush=True)
            torch.save(model.state_dict(), out_path)
            print(f"  saved {out_path}", flush=True)
            model.train()

    print(f"Done in {(time.time() - t0):.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
