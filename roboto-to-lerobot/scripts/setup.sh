#!/usr/bin/env bash

# Provision the local ``.venv`` for development and ``invoke-local`` runs.
#
# By default the venv gets the v3_0 variant's lerobot (``scripts/variants.sh``).
# Pass ``--lerobot-version <ver>`` to pin another release, for example 0.3.3
# to run the v2_1 code path without Docker.
#
# Usage:
#   ./scripts/setup.sh                            # default: v3_0 variant
#   ./scripts/setup.sh --lerobot-version 0.3.3    # pin v2_1 variant

set -euo pipefail

SCRIPTS_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PACKAGE_ROOT=$(dirname "${SCRIPTS_ROOT}")
# shellcheck source=./variants.sh
source "$SCRIPTS_ROOT/variants.sh"

LEROBOT_VERSION=""

usage() {
    cat <<EOF
Usage: $0 [--lerobot-version <ver>]

Optional:
  --lerobot-version VER  pin lerobot to this version in the venv
                         (e.g. 0.3.3 for v2_1). If omitted, uses the v3_0
                         variant's version from scripts/variants.sh.
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --lerobot-version)
            LEROBOT_VERSION="$2"
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

LEROBOT_VERSION="${LEROBOT_VERSION:-$LEROBOT_V3_0_VERSION}"

venv_dir="$PACKAGE_ROOT/.venv"

# Create a virtual environment
python3 -m venv --clear --upgrade-deps "$venv_dir"

# uv installs everything below.
"$venv_dir/bin/pip" install uv

# Install in the Dockerfile's order, with lerobot_pins.sh's pins: CPU-only
# torch, then lerobot, then pyproject.toml's runtime and dev dependencies
# with the lerobot extras and uv overrides.
echo "==> Pinning lerobot==${LEROBOT_VERSION}"
# shellcheck disable=SC2046  # the torch specs are separate words
"$venv_dir/bin/uv" pip install --index-url https://download.pytorch.org/whl/cpu \
    $(sh "$SCRIPTS_ROOT/lerobot_pins.sh" torch "$LEROBOT_VERSION")
"$venv_dir/bin/uv" pip install "lerobot==${LEROBOT_VERSION}"
overrides="$venv_dir/lerobot-overrides.txt"
sh "$SCRIPTS_ROOT/lerobot_pins.sh" overrides "$LEROBOT_VERSION" > "$overrides"

"$venv_dir/bin/uv" pip install --all-extras --overrides "$overrides" --requirements pyproject.toml \
    "lerobot[$(sh "$SCRIPTS_ROOT/lerobot_pins.sh" extras "$LEROBOT_VERSION")]==${LEROBOT_VERSION}"
# --no-deps: resolving the dependencies again without the overrides would
# replace lerobot with another version.
"$venv_dir/bin/uv" pip install --no-deps --editable .
