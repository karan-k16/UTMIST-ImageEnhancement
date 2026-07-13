"""Benchmark SR inference backends: PyTorch (fp32 / AMP), ONNX Runtime, TensorRT.

Measures wall-clock latency (median over N iterations after warmup) and, on
CUDA, peak GPU memory. Optionally computes PSNR/SSIM vs the bicubic baseline
on a directory of HR images. Results are written to JSON + CSV and can be
POSTed to the Rust backend for persistence in PostgreSQL.

Only backends that are actually available on the current machine are run --
nothing is simulated. On a machine without CUDA/TensorRT you will get
PyTorch-CPU and ONNXRuntime-CPU rows only.

Usage:
    python -m ml.benchmark --ONNX_PATH artifacts/sr_model.onnx --RANDOM_INIT
    python -m ml.benchmark --WEIGHTS_PATH runs/best.pth --ONNX_PATH artifacts/sr_model.onnx \
        --VAL_HR_DIR data/val --POST_URL http://localhost:8080/api/benchmarks
"""
import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from models.sr_residual import SRResCNN


def timed_run(fn, n_warmup: int, n_iters: int, sync=None):
    for _ in range(n_warmup):
        fn()
        if sync:
            sync()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "latency_ms_median": round(statistics.median(times), 3),
        "latency_ms_mean": round(statistics.fmean(times), 3),
        "latency_ms_p95": round(sorted(times)[int(len(times) * 0.95) - 1], 3),
        "iters": n_iters,
    }


def bench_pytorch(model, device: str, hw: int, n_warmup: int, n_iters: int, amp: bool):
    model = model.to(device).eval()
    x = torch.randn(1, 3, hw, hw, device=device)
    sync = torch.cuda.synchronize if device == "cuda" else None

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def run():
        if amp and device == "cuda":
            with torch.autocast(device_type="cuda"):
                model(x)
        else:
            model(x)

    stats = timed_run(run, n_warmup, n_iters, sync)
    if device == "cuda":
        stats["gpu_mem_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    return stats


def bench_onnxruntime(onnx_path: str, provider: str, hw: int, n_warmup: int, n_iters: int):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=[provider, "CPUExecutionProvider"])
    if provider not in sess.get_providers():
        return None
    x = np.random.randn(1, 3, hw, hw).astype(np.float32)

    def run():
        sess.run(["sr"], {"lr": x})

    stats = timed_run(run, n_warmup, n_iters)
    stats["providers"] = sess.get_providers()
    return stats


def bench_tensorrt(engine_path: str, hw: int, n_warmup: int, n_iters: int):
    """Native TensorRT benchmark. Requires a built .engine file (see ml/build_trt.sh)
    and the tensorrt + cuda python bindings, i.e. an NVIDIA GPU machine."""
    try:
        import tensorrt as trt
        from cuda import cudart  # cuda-python
    except ImportError:
        return None

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    context = engine.create_execution_context()
    context.set_input_shape("lr", (1, 3, hw, hw))

    x = np.random.randn(1, 3, hw, hw).astype(np.float32)
    out_shape = tuple(context.get_tensor_shape("sr"))
    y = np.empty(out_shape, dtype=np.float32)

    _, d_in = cudart.cudaMalloc(x.nbytes)
    _, d_out = cudart.cudaMalloc(y.nbytes)
    _, stream = cudart.cudaStreamCreate()
    context.set_tensor_address("lr", d_in)
    context.set_tensor_address("sr", d_out)

    def run():
        cudart.cudaMemcpyAsync(d_in, x.ctypes.data, x.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        context.execute_async_v3(stream)
        cudart.cudaMemcpyAsync(y.ctypes.data, d_out, y.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaStreamSynchronize(stream)

    stats = timed_run(run, n_warmup, n_iters)
    cudart.cudaFree(d_in)
    cudart.cudaFree(d_out)
    cudart.cudaStreamDestroy(stream)
    return stats


def quality_eval(model, val_hr_dir: str, scale: int, device: str, amp: bool):
    """PSNR/SSIM of the model and the bicubic baseline against HR ground truth."""
    import cv2

    from utils.data import downscale, find_images_recursive, mod_crop, to_rgb, upscale
    from utils.metrics import psnr_rgb, ssim_rgb
    from utils.tiling import infer_tiled

    files = find_images_recursive(Path(val_hr_dir))
    if not files:
        return None
    pm, pb, sm, sb = [], [], [], []
    for fp in files:
        hr = mod_crop(to_rgb(cv2.imread(fp, cv2.IMREAD_COLOR)), scale)
        lr = downscale(hr, scale)
        bic = upscale(lr, scale)
        sr = infer_tiled(model, lr, scale=scale, amp=amp, device=device)
        pm.append(psnr_rgb(hr, sr)); pb.append(psnr_rgb(hr, bic))
        sm.append(ssim_rgb(hr, sr)); sb.append(ssim_rgb(hr, bic))
    return {
        "n_images": len(files),
        "psnr_model": round(float(np.mean(pm)), 3),
        "psnr_bicubic": round(float(np.mean(pb)), 3),
        "delta_psnr": round(float(np.mean(pm) - np.mean(pb)), 3),
        "ssim_model": round(float(np.mean(sm)), 4),
        "ssim_bicubic": round(float(np.mean(sb)), 4),
        "delta_ssim": round(float(np.mean(sm) - np.mean(sb)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--WEIGHTS_PATH", type=str, default=None)
    ap.add_argument("--RANDOM_INIT", action="store_true",
                    help="Benchmark latency with random weights (no checkpoint needed)")
    ap.add_argument("--ONNX_PATH", type=str, default=None)
    ap.add_argument("--TRT_ENGINE_PATH", type=str, default=None)
    ap.add_argument("--VAL_HR_DIR", type=str, default=None,
                    help="If set, also compute PSNR/SSIM vs bicubic on these HR images")
    ap.add_argument("--UPSCALE", type=int, default=2)
    ap.add_argument("--CHANNELS", type=int, default=64)
    ap.add_argument("--NUM_BLOCKS", type=int, default=8)
    ap.add_argument("--USE_BN", action="store_true")
    ap.add_argument("--INPUT_HW", type=int, default=256, help="Square input size for latency runs")
    ap.add_argument("--WARMUP", type=int, default=5)
    ap.add_argument("--ITERS", type=int, default=30)
    ap.add_argument("--OUT_DIR", type=str, default="outputs/bench")
    ap.add_argument("--POST_URL", type=str, default=None,
                    help="Rust backend endpoint, e.g. http://localhost:8080/api/benchmarks")
    args = ap.parse_args()

    if not args.RANDOM_INIT and not args.WEIGHTS_PATH:
        ap.error("Provide --WEIGHTS_PATH or pass --RANDOM_INIT")

    model = SRResCNN(channels=args.CHANNELS, num_blocks=args.NUM_BLOCKS,
                     scale=args.UPSCALE, use_bn=args.USE_BN)
    if args.WEIGHTS_PATH:
        model.load_state_dict(torch.load(args.WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda else platform.processor() or platform.machine()
    results = []

    def add(backend, precision, stats):
        if stats is None:
            print(f"[skip] {backend} ({precision}): not available on this machine")
            return
        row = {"backend": backend, "precision": precision, "input_hw": args.INPUT_HW,
               "device": device_name, **stats}
        results.append(row)
        print(f"[done] {backend} ({precision}): median {stats['latency_ms_median']} ms")

    print(f"Benchmarking on: {device_name} (CUDA available: {cuda})")

    add("pytorch", "fp32", bench_pytorch(model, "cuda" if cuda else "cpu",
                                         args.INPUT_HW, args.WARMUP, args.ITERS, amp=False))
    if cuda:
        add("pytorch", "amp-fp16", bench_pytorch(model, "cuda", args.INPUT_HW,
                                                 args.WARMUP, args.ITERS, amp=True))

    if args.ONNX_PATH and Path(args.ONNX_PATH).exists():
        add("onnxruntime-cpu", "fp32",
            bench_onnxruntime(args.ONNX_PATH, "CPUExecutionProvider",
                              args.INPUT_HW, args.WARMUP, args.ITERS))
        if cuda:
            add("onnxruntime-cuda", "fp32",
                bench_onnxruntime(args.ONNX_PATH, "CUDAExecutionProvider",
                                  args.INPUT_HW, args.WARMUP, args.ITERS))

    if args.TRT_ENGINE_PATH and Path(args.TRT_ENGINE_PATH).exists():
        add("tensorrt", "fp16",
            bench_tensorrt(args.TRT_ENGINE_PATH, args.INPUT_HW, args.WARMUP, args.ITERS))
    elif args.TRT_ENGINE_PATH:
        print(f"[skip] tensorrt: engine file {args.TRT_ENGINE_PATH} not found "
              "(build it with ml/build_trt.sh on a GPU machine)")

    quality = None
    if args.VAL_HR_DIR and args.WEIGHTS_PATH:
        print("Running PSNR/SSIM quality evaluation...")
        quality = quality_eval(model, args.VAL_HR_DIR, args.UPSCALE,
                               "cuda" if cuda else "cpu", amp=cuda)
        if quality:
            print(f"  PSNR: model {quality['psnr_model']} dB vs bicubic {quality['psnr_bicubic']} dB "
                  f"(delta {quality['delta_psnr']:+.3f} dB)")
            print(f"  SSIM: model {quality['ssim_model']} vs bicubic {quality['ssim_bicubic']} "
                  f"(delta {quality['delta_ssim']:+.4f})")
    elif args.VAL_HR_DIR:
        print("[skip] quality eval requires --WEIGHTS_PATH (random weights would be meaningless)")

    out_dir = Path(args.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": device_name,
        "cuda": cuda,
        "input_hw": args.INPUT_HW,
        "weights": args.WEIGHTS_PATH or "random-init",
        "latency": results,
        "quality": quality,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    json_path = out_dir / "benchmark.json"
    json_path.write_text(json.dumps(payload, indent=2))

    csv_path = out_dir / "benchmark.csv"
    if results:
        keys = sorted({k for r in results for k in r}, key=lambda k: (k != "backend", k))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
    print(f"Wrote {json_path} and {csv_path}")

    if args.POST_URL:
        import requests

        for r in results:
            body = {
                "backend": r["backend"],
                "precision": r["precision"],
                "device": r["device"],
                "input_hw": r["input_hw"],
                "latency_ms": r["latency_ms_median"],
                "gpu_mem_mb": r.get("gpu_mem_peak_mb"),
                "psnr": quality["psnr_model"] if quality else None,
                "ssim": quality["ssim_model"] if quality else None,
                "psnr_bicubic": quality["psnr_bicubic"] if quality else None,
                "ssim_bicubic": quality["ssim_bicubic"] if quality else None,
            }
            resp = requests.post(args.POST_URL, json=body, timeout=10)
            resp.raise_for_status()
        print(f"Posted {len(results)} benchmark rows to {args.POST_URL}")


if __name__ == "__main__":
    main()
