ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=12
ARG OS_VARIANT=slim-bookworm
FROM public.ecr.aws/docker/library/python:${PYTHON_MAJOR}.${PYTHON_MINOR}-${OS_VARIANT}

# -- INSTALL SYSTEM DEPENDENCIES --
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        pkg-config \
        linux-headers-generic \
        ffmpeg \
        libavformat-dev \
        libavcodec-dev \
        libavdevice-dev \
        libavutil-dev \
        libswscale-dev \
        libswresample-dev \
        libavfilter-dev && \
    rm -rf /var/lib/apt/lists/*

# -- INSTALL THIRD-PARTY PYTHON LIBRARIES --
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
COPY pyproject.toml ./
RUN uv pip install --system -r pyproject.toml && rm -rf /bin/uv

COPY src/enrich_lerobot_dataset/ ./enrich_lerobot_dataset

# Force the stdout and stderr streams to be unbuffered to decrease latency when viewing real-time action logs
# and to maximize context captured in logs if this action fails.
ENV PYTHONUNBUFFERED=0

ENTRYPOINT [ "python", "-m", "enrich_lerobot_dataset.bin.entrypoint" ]
