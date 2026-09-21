ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=12
ARG OS_VARIANT=slim-bookworm
ARG PY_BASE=public.ecr.aws/docker/library/python:${PYTHON_MAJOR}.${PYTHON_MINOR}-${OS_VARIANT}


# =====================================================================
# Stage 1: builder — install Python deps with the full toolchain.
# build-essential is what makes `evdev` (pulled in transitively by
# lerobot via pynput) compilable: it brings in linux-libc-dev which
# provides linux/input.h. The libav-*-dev headers are needed for any
# PyAV-adjacent sdists that compile against ffmpeg.
# =====================================================================
FROM ${PY_BASE} AS builder

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

# CPU-only torch wheel. Installed before lerobot so lerobot doesn't pull
# the default CUDA-enabled torch (~2 GB). aggregate_datasets does not
# need a GPU; CPU torch is enough to satisfy the lerobot import surface.
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch

RUN uv pip install --system -r pyproject.toml

# Strip lerobot transitive deps that aggregate_datasets does not import
# at merge time. Verified against the live import graph and against
# LeRobotDatasetMetadata.create() in the built image. Removing torchvision
# breaks lerobot.datasets.io_utils, so it stays. Each name here was
# checked individually; do not extend without re-running the smoke test.
RUN uv pip uninstall --system \
        pynput \
        pyserial \
        wandb \
        diffusers \
        rerun-sdk \
        imageio \
        imageio-ffmpeg \
        accelerate \
        gymnasium \
        deepdiff \
        draccus \
        termcolor \
        einops \
        triton \
        torchcodec


# =====================================================================
# Stage 2: runtime — slim image with only what the action actually needs.
# ffmpeg bundles the libav* runtime; no compilers, no -dev headers, no uv.
# =====================================================================
FROM ${PY_BASE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/lerobot_merge/ ./lerobot_merge

ENV PYTHONUNBUFFERED=0

# torch >= 2.9 calls getpass.getuser() at import to build an inductor
# cache dir. When the container runs as a non-root host UID with no
# matching /etc/passwd entry (e.g. `roboto actions invoke-local`), that
# raises KeyError. Setting USER short-circuits getpass.getuser() before
# it falls back to pwd.getpwuid().
ENV USER=app LOGNAME=app

ENTRYPOINT [ "python", "-m", "lerobot_merge.bin.entrypoint" ]
