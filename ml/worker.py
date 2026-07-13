"""Job worker: polls the Rust backend for queued SR jobs, runs real inference,
and reports results (output path, latency, PSNR/SSIM when ground truth exists).

The worker claims jobs atomically via POST /api/jobs/claim, so multiple
workers can run in parallel safely.

Usage:
    python -m ml.worker --WEIGHTS_PATH runs/best.pth              # PyTorch backend
    python -m ml.worker --ONNX_PATH artifacts/sr_model.onnx      # ORT jobs
    python -m ml.worker --RANDOM_INIT --ONCE                      # smoke test
"""
import argparse
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import requests
import torch

from models.sr_residual import SRResCNN
from utils.metrics import psnr_rgb, ssim_rgb
from utils.tiling import infer_tiled


def load_pytorch_model(args, device):
    model = SRResCNN(channels=args.CHANNELS, num_blocks=args.NUM_BLOCKS,
                     scale=args.UPSCALE, use_bn=args.USE_BN).to(device)
    if not args.RANDOM_INIT:
        model.load_state_dict(torch.load(args.WEIGHTS_PATH, map_location=device))
    model.eval()
    return model


def process_job(job: dict, args, device: str, model, ort_sess) -> dict:
    """Run SR on the job's input image. Returns the PATCH payload."""
    input_path = Path(job["input_path"])
    if not input_path.exists():
        return {"status": "failed", "error": f"input not found: {input_path}"}

    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"status": "failed", "error": f"could not decode image: {input_path}"}
    lr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    scale = job.get("upscale", args.UPSCALE)
    backend = job.get("backend", "pytorch")

    # The model upscales low-res inputs. If someone uploads an already-large
    # image, first shrink it to ~480p so the 2x upscale is meaningful and the
    # before/after comparison is honest (before = the actual model input).
    max_h = args.MAX_INPUT_HEIGHT
    result_extra = {}
    if lr.shape[0] > max_h or lr.shape[1] > max_h * 16 // 9:
        f = min(max_h / lr.shape[0], (max_h * 16 // 9) / lr.shape[1])
        new_w, new_h = int(lr.shape[1] * f), int(lr.shape[0] * f)
        lr = cv2.resize(lr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        lr_path = input_path.with_name(f"{input_path.stem}_lr.png")
        cv2.imwrite(str(lr_path), cv2.cvtColor(lr, cv2.COLOR_RGB2BGR))
        result_extra["input_path"] = str(lr_path)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    if backend.startswith("onnxruntime") and ort_sess is not None:
        from ml.infer_onnx import infer_tiled_onnx
        sr = infer_tiled_onnx(ort_sess, lr, scale=scale, tile=args.TILE, overlap=args.OVERLAP)
    else:
        backend = "pytorch"
        sr = infer_tiled(model, lr, scale=scale, tile=args.TILE,
                         overlap=args.OVERLAP, amp=args.AMP, device=device)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    out_dir = Path(args.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{job['id']}_{input_path.stem}_x{scale}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(sr, cv2.COLOR_RGB2BGR))

    result = {
        "status": "done",
        "output_path": str(out_path),
        "latency_ms": round(latency_ms, 2),
        **result_extra,
    }
    if device == "cuda":
        result["gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    # If the input is actually an HR image (e.g. evaluation jobs), compute
    # quality vs ground truth by downscaling first. Controlled by --EVAL_MODE.
    if args.EVAL_MODE:
        from utils.data import downscale
        hr = lr
        lr_small = downscale(hr, scale)
        if backend.startswith("onnxruntime") and ort_sess is not None:
            from ml.infer_onnx import infer_tiled_onnx
            sr_eval = infer_tiled_onnx(ort_sess, lr_small, scale=scale,
                                       tile=args.TILE, overlap=args.OVERLAP)
        else:
            sr_eval = infer_tiled(model, lr_small, scale=scale, tile=args.TILE,
                                  overlap=args.OVERLAP, amp=args.AMP, device=device)
        h = min(hr.shape[0], sr_eval.shape[0])
        w = min(hr.shape[1], sr_eval.shape[1])
        result["psnr"] = round(psnr_rgb(hr[:h, :w], sr_eval[:h, :w]), 3)
        result["ssim"] = round(ssim_rgb(hr[:h, :w], sr_eval[:h, :w]), 4)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--BACKEND_URL", type=str, default="http://localhost:8080")
    ap.add_argument("--WEIGHTS_PATH", type=str, default=None)
    ap.add_argument("--RANDOM_INIT", action="store_true")
    ap.add_argument("--ONNX_PATH", type=str, default=None)
    ap.add_argument("--OUTPUT_DIR", type=str, default="outputs/jobs")
    ap.add_argument("--UPSCALE", type=int, default=2)
    ap.add_argument("--CHANNELS", type=int, default=64)
    ap.add_argument("--NUM_BLOCKS", type=int, default=8)
    ap.add_argument("--USE_BN", action="store_true")
    ap.add_argument("--TILE", type=int, default=512)
    ap.add_argument("--OVERLAP", type=int, default=32)
    ap.add_argument("--MAX_INPUT_HEIGHT", type=int, default=480,
                    help="Uploads taller/wider than this are downscaled before SR")
    ap.add_argument("--AMP", action="store_true")
    ap.add_argument("--POLL_SECONDS", type=float, default=2.0)
    ap.add_argument("--ONCE", action="store_true", help="Process at most one job, then exit")
    ap.add_argument("--EVAL_MODE", action="store_true",
                    help="Treat inputs as HR ground truth: downscale, super-resolve, report PSNR/SSIM")
    args = ap.parse_args()

    if not args.RANDOM_INIT and not args.WEIGHTS_PATH:
        ap.error("Provide --WEIGHTS_PATH or pass --RANDOM_INIT")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = load_pytorch_model(args, device)

    ort_sess = None
    if args.ONNX_PATH and Path(args.ONNX_PATH).exists():
        from ml.infer_onnx import make_session
        ort_sess = make_session(args.ONNX_PATH)

    print(f"Worker started (device={device}, backend_url={args.BACKEND_URL})")
    while True:
        try:
            resp = requests.post(f"{args.BACKEND_URL}/api/jobs/claim", timeout=10)
        except requests.ConnectionError:
            print("Backend unreachable, retrying...")
            time.sleep(args.POLL_SECONDS)
            continue

        if resp.status_code == 204:
            if args.ONCE:
                print("Queue empty, exiting (--ONCE)")
                return
            time.sleep(args.POLL_SECONDS)
            continue
        resp.raise_for_status()
        job = resp.json()
        print(f"Claimed job {job['id']} ({job['input_path']}, backend={job['backend']})")

        try:
            payload = process_job(job, args, device, model, ort_sess)
        except Exception as e:  # report failure to the API instead of crashing
            traceback.print_exc()
            payload = {"status": "failed", "error": str(e)[:500]}

        patch = requests.patch(f"{args.BACKEND_URL}/api/jobs/{job['id']}",
                               json=payload, timeout=10)
        patch.raise_for_status()
        print(f"Job {job['id']} -> {payload['status']}"
              + (f" ({payload.get('latency_ms')} ms)" if payload.get("latency_ms") else ""))

        if args.ONCE:
            return


if __name__ == "__main__":
    main()
