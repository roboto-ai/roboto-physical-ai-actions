ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=12
ARG OS_VARIANT=slim-bookworm
# Which variant this image becomes. `scripts/build.sh` always passes this
# explicitly (it refuses to run without --lerobot-version), so deployed
# images never rely on the default. The default exists for the two paths
# that cannot pass a build arg at all: a bare `docker build`, and
# `roboto actions invoke-local` against this directory. Both therefore
# build the v3_0 variant. Keep in step with deploy.sh's LEROBOT_V3_0_VERSION.
ARG LEROBOT_VERSION=0.5.1
ARG PY_BASE=public.ecr.aws/docker/library/python:${PYTHON_MAJOR}.${PYTHON_MINOR}-${OS_VARIANT}


# =====================================================================
# Stage 1: builder — installs Python deps with the full toolchain.
# =====================================================================
FROM ${PY_BASE} AS builder

# Re-declare so the ARG is visible inside this stage's RUN steps.
ARG LEROBOT_VERSION

# Build-time only: compilers + libav headers for any sdist that needs them.
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

# CPU-only torch wheel from the PyTorch CPU index. Installed before lerobot
# so lerobot doesn't pull the default CUDA-enabled torch (~2 GB).
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch

# Pin lerobot to the variant we're building. pyproject.toml declares the
# loose ``lerobot >=0.3.3,<0.6`` so the install below satisfies that pin
# without a version conflict, and only this layer rebuilds when
# LEROBOT_VERSION changes.
RUN if [ -z "${LEROBOT_VERSION}" ]; then \
        echo "ERROR: --build-arg LEROBOT_VERSION is required" >&2; \
        exit 1; \
    fi && \
    uv pip install --system "lerobot==${LEROBOT_VERSION}"

# Install everything else. lerobot is already satisfied so it stays at the
# pinned version above.
RUN uv pip install --system -r pyproject.toml


# =====================================================================
# Stage 2: runtime — slim image with only what the action actually needs.
# =====================================================================
FROM ${PY_BASE}

# Runtime libs only: ffmpeg for video decoding; the libav* runtime comes
# bundled with it. No compilers, no -dev headers, no uv.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy installed site-packages and console scripts from the builder stage.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/roboto_to_lerobot/ ./roboto_to_lerobot

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

ENTRYPOINT [ "python", "-m", "roboto_to_lerobot.bin.entrypoint" ]
