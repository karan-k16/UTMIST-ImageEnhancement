mod models;
mod routes;

use std::net::SocketAddr;

use axum::{
    extract::DefaultBodyLimit,
    routing::{get, patch, post},
    Router,
};
use sqlx::postgres::PgPoolOptions;
use tower_http::{cors::CorsLayer, services::ServeDir};

#[derive(Clone)]
pub struct AppState {
    pub pool: sqlx::PgPool,
    pub upload_dir: String,
    pub output_dir: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::init();

    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://sr_user:sr_password@localhost:5432/sr_pipeline".into());
    let bind_addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into());
    let upload_dir = std::env::var("UPLOAD_DIR").unwrap_or_else(|_| "uploads".into());
    let output_dir = std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "outputs/jobs".into());
    let frontend_dir = std::env::var("FRONTEND_DIR").unwrap_or_else(|_| "frontend".into());

    std::fs::create_dir_all(&upload_dir)?;
    std::fs::create_dir_all(&output_dir)?;

    let pool = PgPoolOptions::new()
        .max_connections(8)
        .connect(&database_url)
        .await?;

    sqlx::migrate!("./migrations").run(&pool).await?;
    tracing::info!("migrations applied");

    let state = AppState {
        pool,
        upload_dir: upload_dir.clone(),
        output_dir: output_dir.clone(),
    };

    let app = Router::new()
        .route("/health", get(routes::health))
        .route("/api/jobs", post(routes::create_job).get(routes::list_jobs))
        .route("/api/jobs/claim", post(routes::claim_job))
        .route("/api/jobs/{id}", get(routes::get_job))
        .route("/api/jobs/{id}", patch(routes::update_job))
        .route("/api/upload", post(routes::upload_image))
        .route(
            "/api/benchmarks",
            post(routes::create_benchmark).get(routes::list_benchmarks),
        )
        .nest_service("/files/uploads", ServeDir::new(&upload_dir))
        .nest_service("/files/outputs", ServeDir::new(&output_dir))
        .fallback_service(ServeDir::new(&frontend_dir))
        .layer(DefaultBodyLimit::max(25 * 1024 * 1024)) // 25 MB uploads
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr: SocketAddr = bind_addr.parse()?;
    tracing::info!("sr-backend listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
