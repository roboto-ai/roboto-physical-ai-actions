#!/usr/bin/env bash

set -euo pipefail

SCRIPTS_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PACKAGE_ROOT=$(dirname "${SCRIPTS_ROOT}")

venv_dir="$PACKAGE_ROOT/.venv"

# Create a virtual environment
python3 -m venv --clear --upgrade-deps $venv_dir

# Install runtime and dev deps from pyproject.toml
$venv_dir/bin/pip install uv
$venv_dir/bin/uv pip install --all-extras --requirements pyproject.toml
$venv_dir/bin/uv pip install --editable .
