import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from models.sr_residual import SRResCNN
from utils.tiling import infer_tiled

def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def to_bgr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--WEIGHTS_PATH", type=str, required=True)
    ap.add_argument("--INPUT_DIR", type=str, required=True)
    ap.add_argument("--OUTPUT_DIR", type=str, required=True)
    ap.add_argument("--UPSCALE", type=int, default=2)
    ap.add_argument("--TILE", type=int, default=512)
    ap.add_argument("--OVERLAP", type=int, default=32)
    ap.add_argument("--AMP", action="store_true")
    ap.add_argument("--DEVICE", type=str, default="cuda")
    ap.add_argument("--CHANNELS", type=int, default=64)
    ap.add_argument("--NUM_BLOCKS", type=int, default=8)
    ap.add_argument("--USE_BN", action="store_true")
    args = ap.parse_args()

    DEVICE = args.DEVICE if (args.DEVICE == "cpu" or torch.cuda.is_available()) else "cpu"
    model = SRResCNN(channels=args.CHANNELS, num_blocks=args.NUM_BLOCKS, scale=args.UPSCALE, use_bn=args.USE_BN).to(DEVICE)
    state = torch.load(args.WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state)

    in_dir = Path(args.INPUT_DIR); out_dir = Path(args.OUTPUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")

    for p in sorted(in_dir.rglob("*")):
        if p.suffix.lower() not in exts:
            continue
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        lr = to_rgb(bgr)
        sr = infer_tiled(model, lr, scale=args.UPSCALE, tile=args.TILE, overlap=args.OVERLAP, amp=args.AMP, device=DEVICE)
        cv2.imwrite(str(out_dir/p.name), to_bgr(sr))
        print("Wrote", out_dir/p.name)

if __name__ == "__main__":
    main()
