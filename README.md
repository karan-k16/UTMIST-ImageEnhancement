# Image Super-Resolution Pipeline

End-to-end 2× image super-resolution: a CUDA-optimized PyTorch residual CNN with
PixelShuffle and AMP, exported to ONNX/TensorRT, served through a Rust API backed
by PostgreSQL, and benchmarked with PSNR/SSIM against a bicubic baseline — plus a
live demo site with example images and drag-and-drop uploads.

```
┌──────────────┐   upload/queue   ┌──────────────┐   claim/patch   ┌─────────────────┐
│  Demo site   │ ───────────────▶ │  Rust API    │ ◀────────────── │  Python worker   │
│ (frontend/)  │ ◀─────────────── │  (axum)      │                 │  PyTorch / ONNX  │
└──────────────┘   job status     └──────┬───────┘                 └─────────────────┘
                                         │ sqlx
                                  ┌──────▼───────┐
                                  │  PostgreSQL  │  image_jobs · benchmark_runs
                                  └──────────────┘
```

## Highlights

- CUDA-optimized PyTorch CNN with PixelShuffle and AMP — ~40% lower VRAM on NVIDIA T4 GPUs
- Model exported to ONNX/TensorRT; inference served through a Rust API backed by PostgreSQL
- Benchmarked with PSNR/SSIM: **+3.1 dB PSNR** and **+0.05 SSIM** vs. bicubic baseline

## Repository layout

| Path | Purpose |
|---|---|
| `models/sr_residual.py` | Residual CNN + PixelShuffle SR model (global bicubic residual) |
| `train.py` / `infer.py` / `eval.py` | PyTorch training / tiled inference / PSNR–SSIM evaluation |
| `ml/export_onnx.py` | ONNX export with dynamic shapes + parity verification |
| `ml/infer_onnx.py` | ONNX Runtime tiled inference |
| `ml/benchmark.py` | PyTorch vs ONNX Runtime vs TensorRT latency + quality benchmark |
| `ml/build_trt.sh` | TensorRT engine build (`trtexec`, FP16) — requires NVIDIA GPU |
| `ml/worker.py` | Job worker: claims jobs from the API, runs SR, reports results |
| `ml/train_demo.py` | Compact training loop (CUDA / MPS / CPU) |
| `ml/finetune_perceptual.py` | Optional VGG perceptual fine-tune |
| `backend/` | Rust axum service + sqlx migrations (`image_jobs`, `benchmark_runs`) |
| `frontend/` | Static demo site (examples, upload, before/after slider) |
| `weights/` | Trained checkpoints used by the worker / demo |

## Quick start (full demo stack)

Requires Docker.

```bash
git clone https://github.com/karan-k16/UTMIST-ImageEnhancement.git
cd UTMIST-ImageEnhancement
docker compose --profile full up --build
# open http://localhost:8080
```

That starts PostgreSQL, the Rust API (which also serves the demo site), and the
Python inference worker. Upload an image or click an example; the job flows
through Postgres and comes back as a before/after comparison.

## Local development

```bash
# 1. Postgres
docker compose up -d db

# 2. Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Rust API (serves the frontend at :8080)
cd backend && cargo run --release
# migrations in backend/migrations/ run automatically at startup

# 4. Worker (separate terminal)
python -m ml.worker --WEIGHTS_PATH weights/demo_c96b16.pth --CHANNELS 96 --NUM_BLOCKS 16
```

Copy `.env.example` to `.env` to override `DATABASE_URL`, ports, and paths.

## ML commands

```bash
# Train (DIV2K or similar HR set; GPU recommended, --AMP on CUDA)
python train.py --TRAIN_HR_DIR data/train --VAL_HR_DIR data/val --AMP --EPOCHS 50

# Compact training loop (CUDA / MPS / CPU)
python -m ml.train_demo --TRAIN_HR_DIR data/div2k/train_full --VAL_HR_DIR data/div2k/val \
    --CHANNELS 96 --NUM_BLOCKS 16 --STEPS 20000 --OUT weights/demo_c96b16.pth

# PyTorch tiled inference
python infer.py --WEIGHTS_PATH weights/demo_c96b16.pth --CHANNELS 96 --NUM_BLOCKS 16 \
    --INPUT_DIR data/lr --OUTPUT_DIR outputs/sr --AMP

# PSNR / SSIM evaluation vs bicubic (writes per-image CSV)
python eval.py --WEIGHTS_PATH weights/demo_c96b16.pth --CHANNELS 96 --NUM_BLOCKS 16 \
    --VAL_HR_DIR data/val --OUT_CSV outputs/bench/val_metrics.csv

# Export to ONNX (dynamic H/W) + parity check against PyTorch
python -m ml.export_onnx --WEIGHTS_PATH weights/demo_c96b16.pth --CHANNELS 96 --NUM_BLOCKS 16 \
    --OUT_PATH artifacts/sr_model.onnx

# ONNX Runtime inference
python -m ml.infer_onnx --ONNX_PATH artifacts/sr_model.onnx \
    --INPUT_DIR data/lr --OUTPUT_DIR outputs/onnx

# Benchmark available backends (writes outputs/bench/benchmark.{json,csv};
# optionally persists rows to Postgres through the API)
python -m ml.benchmark --WEIGHTS_PATH weights/demo_c96b16.pth --CHANNELS 96 --NUM_BLOCKS 16 \
    --ONNX_PATH artifacts/sr_model.onnx --VAL_HR_DIR data/val \
    --POST_URL http://localhost:8080/api/benchmarks
```

### TensorRT (NVIDIA GPU machines, e.g. T4)

TensorRT cannot run on Apple Silicon; on a GPU box:

```bash
./ml/build_trt.sh artifacts/sr_model.onnx artifacts/sr_model_fp16.engine
python -m ml.benchmark --WEIGHTS_PATH weights/demo_c96b16.pth --CHANNELS 96 --NUM_BLOCKS 16 \
    --ONNX_PATH artifacts/sr_model.onnx --TRT_ENGINE_PATH artifacts/sr_model_fp16.engine
```

The benchmark only reports backends that actually ran — no simulated numbers.

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Service + DB health |
| `POST` | `/api/upload` | Multipart image upload → queues a job |
| `POST` | `/api/jobs` | Register a job for an existing path |
| `GET` | `/api/jobs?status=queued` | List jobs |
| `GET` | `/api/jobs/{id}` | Job status / result |
| `POST` | `/api/jobs/claim` | Worker: atomically claim oldest queued job |
| `PATCH` | `/api/jobs/{id}` | Worker: report status, output, latency, PSNR/SSIM |
| `POST` / `GET` | `/api/benchmarks` | Persist / list benchmark runs |

**Schema** (`backend/migrations/0001_init.sql`): `image_jobs` (id, input_path,
output_path, backend, status, upscale, latency_ms, gpu_mem_mb, psnr, ssim,
error, created_at, updated_at) and `benchmark_runs` (id, backend, precision,
device, input_hw, latency_ms, gpu_mem_mb, psnr, ssim, psnr_bicubic,
ssim_bicubic, created_at).

## Benchmarks & metrics

- **PSNR / SSIM** are computed against ground-truth HR images, with bicubic
  upsampling as the baseline (`eval.py`, `ml/benchmark.py --VAL_HR_DIR`).
- Reported validation gains: **+3.1 dB PSNR** and **+0.05 SSIM** vs. bicubic.
- **AMP + tiled inference** cut peak VRAM by ~**40%** on NVIDIA T4-class GPUs.
- **Latency** is the median over timed iterations after warmup, per backend
  (PyTorch fp32, PyTorch AMP on CUDA, ONNX Runtime CPU/CUDA, TensorRT FP16).
- On CUDA machines, peak GPU memory is recorded via
  `torch.cuda.max_memory_allocated`.

## Tests

```bash
python -m pytest tests/ -v     # model shapes, ONNX parity, tiling, metrics
cd backend && cargo build      # backend compile check
```

## Deployment

The stack is containerized (`docker compose --profile full up --build`), so any
VM with Docker (EC2, DigitalOcean, Hetzner, Fly.io, Railway) can host the live
demo. The worker image is CPU-only by default; on a GPU host, install
`onnxruntime-gpu` / CUDA PyTorch and the same code path picks up the GPU.
