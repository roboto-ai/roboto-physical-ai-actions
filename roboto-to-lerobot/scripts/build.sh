#!/usr/bin/env bash

set -euo pipefail

SCRIPTS_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PACKAGE_ROOT=$(dirname "${SCRIPTS_ROOT}")

QUIET=false
LEROBOT_VERSION=""
LOCAL_TAG="roboto_to_lerobot:latest"

usage() {
    cat <<EOF
Usage: $0 --lerobot-version <ver> [--tag <local_tag>] [--quiet]

Required:
  --lerobot-version VER  lerobot version to install in the image (e.g. 0.3.3, 0.5.1).

Optional:
  --tag TAG              local docker tag for the built image (default: ${LOCAL_TAG}).
  --quiet                suppress docker build output.
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --quiet)
            QUIET=true
            shift
            ;;
        --lerobot-version)
            LEROBOT_VERSION="$2"
            shift 2
            ;;
        --tag)
            LOCAL_TAG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$LEROBOT_VERSION" ]]; then
    echo "Error: --lerobot-version is required (e.g. 0.3.3 or 0.5.1)" >&2
    usage >&2
    exit 1
fi

if ! docker buildx version &> /dev/null; then
    echo "Error: docker buildx is not available." >&2
    echo "Please install Docker Engine >= 19.03 to build this image." >&2
    exit 1
fi

build_subcommand=(
    buildx build
    --platform linux/amd64
    --output type=image
    --build-arg "LEROBOT_VERSION=${LEROBOT_VERSION}"
)

if [ "$QUIET" = true ]; then
    # Suppress build progress on stdout but keep stderr visible so an
    # actual build failure surfaces to the user.
    docker "${build_subcommand[@]}" --quiet \
        -f "$PACKAGE_ROOT/Dockerfile" \
        -t "$LOCAL_TAG" \
        "$PACKAGE_ROOT" > /dev/null
else
    docker "${build_subcommand[@]}" \
        -f "$PACKAGE_ROOT/Dockerfile" \
        -t "$LOCAL_TAG" \
        "$PACKAGE_ROOT"
fi
