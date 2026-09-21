#!/usr/bin/env bash

# Provision the local ``.venv`` for development and ``invoke-local`` runs.
#
# By default the venv resolves whatever lerobot version satisfies
# ``pyproject.toml``'s ``lerobot >=0.3.3,<0.6`` constraint (typically the
# latest 0.5.x). Pass ``--lerobot-version <ver>`` to pin a specific
# release — useful when invoke-local'ing against the v2_1 code path
# without Docker.
#
# Usage:
#   ./scripts/setup.sh                            # default: latest in range
#   ./scripts/setup.sh --lerobot-version 0.3.3    # pin v2_1 variant
#   ./scripts/setup.sh --lerobot-version 0.5.1    # pin v3_0 variant

set -euo pipefail

SCRIPTS_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PACKAGE_ROOT=$(dirname "${SCRIPTS_ROOT}")

LEROBOT_VERSION=""

usage() {
    cat <<EOF
Usage: $0 [--lerobot-version <ver>]

Optional:
  --lerobot-version VER  pin lerobot to this version in the venv
                         (e.g. 0.3.3 for v2_1, 0.5.1 for v3_0).
                         If omitted, pip resolves the latest version
                         allowed by pyproject.toml.
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

venv_dir="$PACKAGE_ROOT/.venv"

# Create a virtual environment
python3 -m venv --clear --upgrade-deps "$venv_dir"

# Install runtime and dev deps from pyproject.toml
"$venv_dir/bin/pip" install uv

# When a specific lerobot is requested, install it BEFORE pyproject.toml so
# the loose ``lerobot >=0.3.3,<0.6`` pin is satisfied without uv upgrading
# us off the requested version. Mirrors the Dockerfile's install order.
if [[ -n "$LEROBOT_VERSION" ]]; then
    echo "==> Pinning lerobot==${LEROBOT_VERSION}"
    "$venv_dir/bin/uv" pip install --index-url https://download.pytorch.org/whl/cpu torch
    "$venv_dir/bin/uv" pip install "lerobot==${LEROBOT_VERSION}"
fi

"$venv_dir/bin/uv" pip install --all-extras --requirements pyproject.toml
"$venv_dir/bin/uv" pip install --editable .
