# Multi-Stage Dockerfile for ControlPlane.ai Gateway (Rust + C++ ONNX/llama.cpp)
#
# STAGE 1: C++ Native Dependencies & ONNX Runtime Libraries
FROM debian:bookworm-slim AS native-builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    clang \
    git \
    wget \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime C++ Shared Libraries
WORKDIR /tmp
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz \
    && tar -xzf onnxruntime-linux-x64-1.17.1.tgz \
    && cp -r onnxruntime-linux-x64-1.17.1/include/* /usr/local/include/ \
    && cp -r onnxruntime-linux-x64-1.17.1/lib/* /usr/local/lib/ \
    && ldconfig

# STAGE 2: Rust Gateway Compilation
FROM rust:1.76-slim-bookworm AS rust-builder
COPY --from=native-builder /usr/local /usr/local

RUN apt-get update && apt-get install -y build-essential cmake clang && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY CMakeLists.txt ./
COPY src/ ./src/
COPY cpp/ ./cpp/

RUN cargo build --release

# STAGE 3: Final Slim Production Runtime
FROM debian:bookworm-slim AS runner

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=native-builder /usr/local/lib/libonnxruntime* /usr/local/lib/
RUN ldconfig

WORKDIR /app
COPY --from=rust-builder /app/target/release/controlplane-ai /app/controlplane-ai
COPY models/ /app/models/

EXPOSE 8080 9090

ENV RUST_LOG=info
ENV CP_SERVER__PORT=8080

ENTRYPOINT ["/app/controlplane-ai"]
