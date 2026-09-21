ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=12
ARG OS_VARIANT=slim-bookworm
ARG PY_BASE=public.ecr.aws/docker/library/python:${PYTHON_MAJOR}.${PYTHON_MINOR}-${OS_VARIANT}


# =====================================================================
# Stage 1: builder — install Python deps with the full toolchain.
# build-essential brings linux-libc-dev (linux/input.h), which lets evdev
# (pulled transitively by lerobot via pynput) compile. The libav*-dev headers
# satisfy any PyAV-adjacent sdists that compile against ffmpeg.
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

# CPU-only torch wheel, installed before lerobot so lerobot doesn't pull the
# default CUDA-enabled torch (~2 GB). The downgrade is pure parquet/video
# surgery — no GPU needed; CPU torch only has to satisfy lerobot's import surface.
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch

RUN uv pip install --system -r pyproject.toml


# =====================================================================
# Stage 2: runtime — slim image with only what the action needs at run time.
# ffmpeg bundles the libav* runtime used to split (and, when a stream-copy cut
# is not frame-exact, re-encode) the per-episode videos. No compilers, no -dev
# headers, no uv.
# =====================================================================
FROM ${PY_BASE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/lerobot_v3_to_v2_1/ ./lerobot_v3_to_v2_1

ENV PYTHONUNBUFFERED=0

# torch >= 2.9 calls getpass.getuser() at import to build an inductor cache dir.
# When the container runs as a non-root host UID with no /etc/passwd entry
# (e.g. `roboto actions invoke-local`), that raises KeyError. Setting USER
# short-circuits getpass.getuser() before it falls back to pwd.getpwuid().
ENV USER=app LOGNAME=app

ENTRYPOINT [ "python", "-m", "lerobot_v3_to_v2_1.bin.entrypoint" ]
