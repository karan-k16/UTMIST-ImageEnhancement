use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct ImageJob {
    pub id: Uuid,
    pub input_path: String,
    pub output_path: Option<String>,
    pub backend: String,
    pub status: String,
    pub upscale: i32,
    pub latency_ms: Option<f64>,
    pub gpu_mem_mb: Option<f64>,
    pub psnr: Option<f64>,
    pub ssim: Option<f64>,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
pub struct CreateJob {
    pub input_path: String,
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default = "default_upscale")]
    pub upscale: i32,
}

fn default_backend() -> String {
    "pytorch".into()
}

fn default_upscale() -> i32 {
    2
}

#[derive(Debug, Deserialize)]
pub struct UpdateJob {
    pub status: Option<String>,
    pub output_path: Option<String>,
    pub input_path: Option<String>,
    pub latency_ms: Option<f64>,
    pub gpu_mem_mb: Option<f64>,
    pub psnr: Option<f64>,
    pub ssim: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct BenchmarkRun {
    pub id: Uuid,
    pub backend: String,
    pub precision: String,
    pub device: Option<String>,
    pub input_hw: Option<i32>,
    pub latency_ms: f64,
    pub gpu_mem_mb: Option<f64>,
    pub psnr: Option<f64>,
    pub ssim: Option<f64>,
    pub psnr_bicubic: Option<f64>,
    pub ssim_bicubic: Option<f64>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
pub struct CreateBenchmark {
    pub backend: String,
    #[serde(default = "default_precision")]
    pub precision: String,
    pub device: Option<String>,
    pub input_hw: Option<i32>,
    pub latency_ms: f64,
    pub gpu_mem_mb: Option<f64>,
    pub psnr: Option<f64>,
    pub ssim: Option<f64>,
    pub psnr_bicubic: Option<f64>,
    pub ssim_bicubic: Option<f64>,
}

fn default_precision() -> String {
    "fp32".into()
}
