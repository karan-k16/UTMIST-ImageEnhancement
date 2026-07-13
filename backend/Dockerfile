# Build from the repo root: docker build -f backend/Dockerfile .
# Works on amd64 and arm64 (Oracle Ampere).
FROM rust:1-bookworm AS builder
WORKDIR /app

# System deps sometimes needed for crates on slim hosts
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY backend/Cargo.toml backend/Cargo.lock ./
COPY backend/src ./src
COPY backend/migrations ./migrations
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/sr-backend /usr/local/bin/sr-backend
COPY --from=builder /app/migrations ./migrations
COPY frontend ./frontend
ENV BIND_ADDR=0.0.0.0:8080
ENV FRONTEND_DIR=/app/frontend
EXPOSE 8080
CMD ["sr-backend"]
