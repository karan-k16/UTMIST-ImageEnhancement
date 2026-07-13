CREATE TABLE IF NOT EXISTS image_jobs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    input_path  TEXT NOT NULL,
    output_path TEXT,
    backend     TEXT NOT NULL DEFAULT 'pytorch',   -- pytorch | onnxruntime | tensorrt
    status      TEXT NOT NULL DEFAULT 'queued',    -- queued | running | done | failed
    upscale     INT  NOT NULL DEFAULT 2,
    latency_ms  DOUBLE PRECISION,
    gpu_mem_mb  DOUBLE PRECISION,
    psnr        DOUBLE PRECISION,
    ssim        DOUBLE PRECISION,
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_image_jobs_status ON image_jobs (status);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backend      TEXT NOT NULL,                    -- pytorch | onnxruntime-cpu | onnxruntime-cuda | tensorrt
    precision    TEXT NOT NULL DEFAULT 'fp32',
    device       TEXT,
    input_hw     INT,
    latency_ms   DOUBLE PRECISION NOT NULL,
    gpu_mem_mb   DOUBLE PRECISION,
    psnr         DOUBLE PRECISION,
    ssim         DOUBLE PRECISION,
    psnr_bicubic DOUBLE PRECISION,
    ssim_bicubic DOUBLE PRECISION,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
