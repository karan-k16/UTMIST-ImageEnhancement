use axum::{
    extract::{Multipart, Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::models::{BenchmarkRun, CreateBenchmark, CreateJob, ImageJob, UpdateJob};
use crate::AppState;

type ApiError = (StatusCode, Json<Value>);

fn internal(e: impl std::fmt::Display) -> ApiError {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({ "error": e.to_string() })),
    )
}

fn bad_request(msg: &str) -> ApiError {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": msg })))
}

fn not_found() -> ApiError {
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "error": "not found" })),
    )
}

pub async fn health(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    sqlx::query_scalar::<_, i32>("SELECT 1")
        .fetch_one(&state.pool)
        .await
        .map_err(internal)?;
    Ok(Json(json!({ "status": "ok", "db": "up" })))
}

pub async fn create_job(
    State(state): State<AppState>,
    Json(body): Json<CreateJob>,
) -> Result<(StatusCode, Json<ImageJob>), ApiError> {
    let job = sqlx::query_as::<_, ImageJob>(
        "INSERT INTO image_jobs (input_path, backend, upscale)
         VALUES ($1, $2, $3) RETURNING *",
    )
    .bind(&body.input_path)
    .bind(&body.backend)
    .bind(body.upscale)
    .fetch_one(&state.pool)
    .await
    .map_err(internal)?;
    Ok((StatusCode::CREATED, Json(job)))
}

/// Multipart image upload from the demo site; stores the file and queues a job.
pub async fn upload_image(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<ImageJob>), ApiError> {
    let mut saved: Option<String> = None;

    while let Some(field) = multipart.next_field().await.map_err(internal)? {
        if field.name() != Some("image") {
            continue;
        }
        let filename = field.file_name().unwrap_or("upload.png").to_string();
        let ext = std::path::Path::new(&filename)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("png")
            .to_lowercase();
        if !["png", "jpg", "jpeg", "bmp", "webp"].contains(&ext.as_str()) {
            return Err(bad_request("unsupported file type"));
        }
        let data = field.bytes().await.map_err(internal)?;
        if data.is_empty() {
            return Err(bad_request("empty file"));
        }
        let path = format!("{}/{}.{}", state.upload_dir, Uuid::new_v4(), ext);
        tokio::fs::write(&path, &data).await.map_err(internal)?;
        saved = Some(path);
        break;
    }

    let input_path = saved.ok_or_else(|| bad_request("missing 'image' field"))?;
    let job = sqlx::query_as::<_, ImageJob>(
        "INSERT INTO image_jobs (input_path, backend, upscale)
         VALUES ($1, 'pytorch', 2) RETURNING *",
    )
    .bind(&input_path)
    .fetch_one(&state.pool)
    .await
    .map_err(internal)?;
    Ok((StatusCode::CREATED, Json(job)))
}

#[derive(Deserialize)]
pub struct ListJobsParams {
    pub status: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: i64,
}

fn default_limit() -> i64 {
    50
}

pub async fn list_jobs(
    State(state): State<AppState>,
    Query(params): Query<ListJobsParams>,
) -> Result<Json<Vec<ImageJob>>, ApiError> {
    let jobs = match params.status {
        Some(status) => {
            sqlx::query_as::<_, ImageJob>(
                "SELECT * FROM image_jobs WHERE status = $1
                 ORDER BY created_at DESC LIMIT $2",
            )
            .bind(status)
            .bind(params.limit)
            .fetch_all(&state.pool)
            .await
        }
        None => {
            sqlx::query_as::<_, ImageJob>(
                "SELECT * FROM image_jobs ORDER BY created_at DESC LIMIT $1",
            )
            .bind(params.limit)
            .fetch_all(&state.pool)
            .await
        }
    }
    .map_err(internal)?;
    Ok(Json(jobs))
}

pub async fn get_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<ImageJob>, ApiError> {
    sqlx::query_as::<_, ImageJob>("SELECT * FROM image_jobs WHERE id = $1")
        .bind(id)
        .fetch_optional(&state.pool)
        .await
        .map_err(internal)?
        .map(Json)
        .ok_or_else(not_found)
}

/// Atomically claim the oldest queued job (used by the Python worker).
/// Returns 204 if the queue is empty.
pub async fn claim_job(
    State(state): State<AppState>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let job = sqlx::query_as::<_, ImageJob>(
        "UPDATE image_jobs SET status = 'running', updated_at = now()
         WHERE id = (
             SELECT id FROM image_jobs WHERE status = 'queued'
             ORDER BY created_at
             FOR UPDATE SKIP LOCKED
             LIMIT 1
         )
         RETURNING *",
    )
    .fetch_optional(&state.pool)
    .await
    .map_err(internal)?;

    match job {
        Some(j) => Ok((
            StatusCode::OK,
            Json(serde_json::to_value(j).map_err(internal)?),
        )),
        None => Ok((StatusCode::NO_CONTENT, Json(Value::Null))),
    }
}

pub async fn update_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(body): Json<UpdateJob>,
) -> Result<Json<ImageJob>, ApiError> {
    sqlx::query_as::<_, ImageJob>(
        "UPDATE image_jobs SET
            status      = COALESCE($2, status),
            output_path = COALESCE($3, output_path),
            latency_ms  = COALESCE($4, latency_ms),
            gpu_mem_mb  = COALESCE($5, gpu_mem_mb),
            psnr        = COALESCE($6, psnr),
            ssim        = COALESCE($7, ssim),
            error       = COALESCE($8, error),
            input_path  = COALESCE($9, input_path),
            updated_at  = now()
         WHERE id = $1 RETURNING *",
    )
    .bind(id)
    .bind(&body.status)
    .bind(&body.output_path)
    .bind(body.latency_ms)
    .bind(body.gpu_mem_mb)
    .bind(body.psnr)
    .bind(body.ssim)
    .bind(&body.error)
    .bind(&body.input_path)
    .fetch_optional(&state.pool)
    .await
    .map_err(internal)?
    .map(Json)
    .ok_or_else(not_found)
}

pub async fn create_benchmark(
    State(state): State<AppState>,
    Json(body): Json<CreateBenchmark>,
) -> Result<(StatusCode, Json<BenchmarkRun>), ApiError> {
    let row = sqlx::query_as::<_, BenchmarkRun>(
        "INSERT INTO benchmark_runs
            (backend, precision, device, input_hw, latency_ms, gpu_mem_mb,
             psnr, ssim, psnr_bicubic, ssim_bicubic)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) RETURNING *",
    )
    .bind(&body.backend)
    .bind(&body.precision)
    .bind(&body.device)
    .bind(body.input_hw)
    .bind(body.latency_ms)
    .bind(body.gpu_mem_mb)
    .bind(body.psnr)
    .bind(body.ssim)
    .bind(body.psnr_bicubic)
    .bind(body.ssim_bicubic)
    .fetch_one(&state.pool)
    .await
    .map_err(internal)?;
    Ok((StatusCode::CREATED, Json(row)))
}

pub async fn list_benchmarks(
    State(state): State<AppState>,
) -> Result<Json<Vec<BenchmarkRun>>, ApiError> {
    let rows = sqlx::query_as::<_, BenchmarkRun>(
        "SELECT * FROM benchmark_runs ORDER BY created_at DESC LIMIT 100",
    )
    .fetch_all(&state.pool)
    .await
    .map_err(internal)?;
    Ok(Json(rows))
}
