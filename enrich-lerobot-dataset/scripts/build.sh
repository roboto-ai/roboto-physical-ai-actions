#!/usr/bin/env bash

set -euo pipefail

SCRIPTS_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PACKAGE_ROOT=$(dirname "${SCRIPTS_ROOT}")

# Parse command line arguments
QUIET=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --quiet)
            QUIET=true
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--quiet]" >&2
            exit 1
            ;;
    esac
done

if ! docker buildx version &> /dev/null; then
    echo "Error: docker buildx is not available." >&2
    echo "Please install Docker Engine >= 19.03 to build this image." >&2
    exit 1
fi

build_subcommand=(buildx build --platform linux/amd64 --output type=image)

if [ "$QUIET" = true ]; then
    docker "${build_subcommand[@]}" --quiet -f $PACKAGE_ROOT/Dockerfile -t enrich_lerobot_dataset:latest $PACKAGE_ROOT &> /dev/null
else
    docker "${build_subcommand[@]}" -f $PACKAGE_ROOT/Dockerfile -t enrich_lerobot_dataset:latest $PACKAGE_ROOT
fi
