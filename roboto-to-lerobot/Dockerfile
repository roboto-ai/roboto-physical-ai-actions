ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=12
ARG OS_VARIANT=slim-bookworm
# Which variant this image becomes. `scripts/build.sh` always passes this
# explicitly (it refuses to run without --lerobot-version), so deployed
# images never rely on the default. The default exists for the two paths
# that cannot pass a build arg at all: a bare `docker build`, and
# `roboto actions invoke-local` against this directory. Both therefore
# build the v3_0 variant. Keep in step with LEROBOT_V3_0_VERSION in
# scripts/variants.sh.
ARG LEROBOT_VERSION=0.7.0
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
COPY pyproject.toml scripts/lerobot_pins.sh ./

# CPU-only torch wheels from the PyTorch CPU index. Installed before lerobot
# so lerobot doesn't pull the default CUDA-enabled torch (~2 GB) from PyPI.
# That works only while the versions that lerobot_pins.sh picks satisfy
# lerobot's torch caps.
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu \
        $(sh lerobot_pins.sh torch "${LEROBOT_VERSION}")

# Pin lerobot to the variant being built; pyproject.toml accepts any
# ``lerobot >=0.3.3,<0.8``.
RUN if [ -z "${LEROBOT_VERSION}" ]; then \
        echo "ERROR: --build-arg LEROBOT_VERSION is required" >&2; \
        exit 1; \
    fi && \
    uv pip install --system "lerobot==${LEROBOT_VERSION}"

# Install pyproject.toml's dependencies with the lerobot extras and uv
# overrides from lerobot_pins.sh. lerobot is requested again at the pinned
# version because, on a conflict, uv would otherwise install another lerobot
# version instead of failing.
RUN sh lerobot_pins.sh overrides "${LEROBOT_VERSION}" > /tmp/overrides.txt && \
    uv pip install --system --overrides /tmp/overrides.txt -r pyproject.toml \
        "lerobot[$(sh lerobot_pins.sh extras "${LEROBOT_VERSION}")]==${LEROBOT_VERSION}"


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
