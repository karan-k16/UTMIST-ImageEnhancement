"""Crop large HR images into fixed-size tiles for faster patch training.

Reading a 2K DIV2K png just to take one 96px crop wastes most of the decode
time; pre-cropping into ~400px tiles makes each training step cheap.

Usage:
    python -m ml.prep_div2k --SRC /tmp/DIV2K_train_HR --OUT data/div2k/train --TILE 400
"""
import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--SRC", type=str, required=True)
    ap.add_argument("--OUT", type=str, required=True)
    ap.add_argument("--TILE", type=int, default=400)
    ap.add_argument("--MIN_KEEP", type=int, default=200,
                    help="Drop edge remainders smaller than this")
    args = ap.parse_args()

    src, out = Path(args.SRC), Path(args.OUT)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    n_tiles = 0
    for i, fp in enumerate(files):
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        t = args.TILE
        for y in range(0, H, t):
            for x in range(0, W, t):
                tile = img[y:y + t, x:x + t]
                if tile.shape[0] < args.MIN_KEEP or tile.shape[1] < args.MIN_KEEP:
                    continue
                cv2.imwrite(str(out / f"{fp.stem}_{y}_{x}.png"), tile)
                n_tiles += 1
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(files)} images -> {n_tiles} tiles", flush=True)
    print(f"Done: {len(files)} images -> {n_tiles} tiles in {out}")


if __name__ == "__main__":
    main()
