# Build from the repo root: docker build -f backend/Dockerfile .
FROM rust:slim AS builder
WORKDIR /app
COPY backend/Cargo.toml backend/Cargo.lock* ./
COPY backend/src ./src
COPY backend/migrations ./migrations
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/sr-backend /usr/local/bin/sr-backend
COPY --from=builder /app/migrations ./migrations
COPY frontend ./frontend
EXPOSE 8080
CMD ["sr-backend"]
