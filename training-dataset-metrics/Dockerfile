ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=12
ARG OS_VARIANT=slim-bookworm
ARG PY_BASE=public.ecr.aws/docker/library/python:${PYTHON_MAJOR}.${PYTHON_MINOR}-${OS_VARIANT}


# =====================================================================
# Stage 1: builder — installs Python deps with the full toolchain.
# =====================================================================
FROM ${PY_BASE} AS builder

# Build-time only: compilers + libav headers for any sdist that needs them
# (mainly the vendored roboto-to-lerobot decoders' transitive deps).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        pkg-config \
        libavformat-dev \
        libavcodec-dev \
        libavdevice-dev \
        libavutil-dev \
        libswscale-dev \
        libswresample-dev \
        libavfilter-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
COPY pyproject.toml ./

# Pull torch from the CPU-only index. Default PyPI torch wheels bundle the
# full CUDA + cuDNN + Triton stack (~5 GB); this action runs inference-free
# metrics so GPU support is pure dead weight. `unsafe-best-match` lets uv
# pick `torch==X.Y.Z+cpu` (PEP 440 local segment sorts after the PyPI
# `X.Y.Z` CUDA build, so the CPU wheel wins).
RUN uv pip install --system \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        -r pyproject.toml


# =====================================================================
# Stage 2: runtime — slim image with only what the action actually needs.
# =====================================================================
FROM ${PY_BASE}

# Runtime libs only: ffmpeg for video decoding (the vendored decoders use
# it transitively); the libav* runtime comes bundled with it. No
# compilers, no -dev headers, no uv.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy installed site-packages and console scripts from the builder stage.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/training_dataset_metrics/ ./training_dataset_metrics

# Force the stdout and stderr streams to be unbuffered to decrease latency
# when viewing real-time action logs and to maximize context captured in
# logs if this action fails.
ENV PYTHONUNBUFFERED=0

# torch >= 2.9 calls getpass.getuser() at import time to build an inductor
# cache dir. When the container runs as a non-root host UID with no matching
# /etc/passwd entry (e.g. ``roboto actions invoke-local``), that raises
# KeyError. Setting USER short-circuits getpass.getuser() to the env var
# before it falls back to pwd.getpwuid().
ENV USER=app LOGNAME=app

ENTRYPOINT [ "python", "-m", "training_dataset_metrics.bin.entrypoint" ]
