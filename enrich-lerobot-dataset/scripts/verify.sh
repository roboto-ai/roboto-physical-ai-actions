#!/usr/bin/env bash

set -euo pipefail

SCRIPTS_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PACKAGE_ROOT=$(dirname "${SCRIPTS_ROOT}")

venv_dir="$PACKAGE_ROOT/.venv"

# Early exit if virtual environment does not exist
if [ ! -d "$venv_dir" ]; then
    echo "Virtual environment does not exist at $venv_dir. Please run ./scripts/setup.sh first."
    exit 1
fi

# Check that required executables exist
if [ ! -f "$venv_dir/bin/ruff" ]; then
    echo "ruff is not installed in the virtual environment. Please run ./scripts/setup.sh first."
    exit 1
fi

if [ ! -f "$venv_dir/bin/pytest" ]; then
    echo "pytest is not installed in the virtual environment. Please run ./scripts/setup.sh first."
    exit 1
fi

echo "########## Lint ##########"
if ! $venv_dir/bin/ruff check .; then
    exit 1
fi

echo "########## Test ##########"
if ! $venv_dir/bin/pytest; then
    exit 1
fi

