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

echo "########## Vendored sync ##########"
# `_vendored/roboto_to_lerobot/` is a hand-refreshed copy of the sibling
# package. Source the refresh script for its VENDORED_FILES list (and the two
# directory paths) without letting it copy anything, so this check and the
# command that fixes it read one definition of what is mirrored.
VENDORED_LIST_ONLY=1 source "$SCRIPTS_ROOT/sync_vendored.sh"

if [ ! -d "$UPSTREAM_DIR" ]; then
    echo "Skipped: no upstream at $UPSTREAM_DIR (sibling package not checked out)."
else
    drifted=()

    for f in "${VENDORED_FILES[@]}"; do
        if [ ! -f "$VENDORED_DIR/$f" ]; then
            drifted+=("$f (mirrored file missing from the vendored copy)")
        elif [ ! -f "$UPSTREAM_DIR/$f" ]; then
            drifted+=("$f (no longer exists upstream)")
        elif ! cmp -s "$UPSTREAM_DIR/$f" "$VENDORED_DIR/$f"; then
            drifted+=("$f (differs from upstream)")
        fi
    done

    # Anything sitting in the vendored tree that upstream does not have,
    # including files dropped from VENDORED_FILES but left behind on disk.
    # `__init__.py` is exempt: it is a local override that deliberately does
    # not mirror upstream's, so upstream's copy of it is irrelevant here.
    while IFS= read -r path; do
        rel=${path#"$VENDORED_DIR/"}
        if [ "$rel" != "__init__.py" ] && [ ! -f "$UPSTREAM_DIR/$rel" ]; then
            drifted+=("$rel (present here, absent upstream)")
        fi
    done < <(find "$VENDORED_DIR" -name '*.py' -not -path '*/__pycache__/*' | sort)

    if [ ${#drifted[@]} -gt 0 ]; then
        echo "Vendored copy has drifted from $UPSTREAM_DIR:"
        for d in "${drifted[@]}"; do
            echo "  $d"
        done
        echo
        echo "Refresh it with:"
        echo "  scripts/sync_vendored.sh"
        exit 1
    fi

    echo "${#VENDORED_FILES[@]} vendored files match upstream."
fi

echo "########## Lint ##########"
if ! $venv_dir/bin/ruff check .; then
    exit 1
fi

echo "########## Test ##########"
if ! $venv_dir/bin/pytest; then
    exit 1
fi

