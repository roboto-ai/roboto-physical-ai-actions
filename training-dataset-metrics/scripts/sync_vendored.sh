#!/usr/bin/env bash
#
# Refresh the in-tree `_vendored/roboto_to_lerobot/` subset from the sibling
# `roboto-to-lerobot` package. Run this after making changes in upstream that
# this action should pick up.
#
# Only files in $VENDORED_FILES below are mirrored. If upstream adds or
# removes a file we depend on, update that list in this script *by hand* —
# silent schema drift is worse than a loud "file list out of date" error.
#
# `scripts/verify.sh` reads $VENDORED_FILES from here and fails if the mirror
# has fallen out of step with upstream, so forgetting to run this is loud too.

set -euo pipefail

SCRIPTS_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PACKAGE_ROOT=$(dirname "${SCRIPTS_ROOT}")
REPO_ROOT=$(dirname "${PACKAGE_ROOT}")

UPSTREAM_DIR="$REPO_ROOT/roboto-to-lerobot/src/roboto_to_lerobot"
VENDORED_DIR="$PACKAGE_ROOT/src/training_dataset_metrics/_vendored/roboto_to_lerobot"

# Paths are relative to $UPSTREAM_DIR and mirror into the same layout under
# $VENDORED_DIR. `runtime/` is a partial mirror on purpose: `decoders.py` and
# `converters.py` upstream are now thin re-export shims over `runtime/`, so the
# four modules the decoder registry actually pulls in have to come along, while
# the live-inference modules behind `runtime/__init__.py`'s lazy exports
# (contract_io, live_adapter, replay_adapter, stream_buffer, encoders,
# causal_lowpass) do not, because this action never runs a live node. Touching one
# of those names here raises rather than silently working on stale code.
VENDORED_FILES=(
    contract_utils.py
    decoders.py
    extract.py
    converters.py
    logger.py
    lerobot.py
    alignment.py
    transforms.py
    video.py
    runtime/__init__.py
    runtime/converters.py
    runtime/decoders.py
    runtime/image.py
    runtime/LICENSE-rosetta
)

# Sourcing this script with $VENDORED_LIST_ONLY set defines UPSTREAM_DIR,
# VENDORED_DIR and VENDORED_FILES and then returns, copying nothing. That is how
# `scripts/verify.sh` checks the mirror against the same list the refresh copies
# from, so the check and the fix can never disagree about which files are
# vendored. Everything below this point is the refresh itself.
if [ -n "${VENDORED_LIST_ONLY:-}" ]; then
    return 0
fi

if [ ! -d "$UPSTREAM_DIR" ]; then
    echo "Upstream not found at $UPSTREAM_DIR" >&2
    exit 1
fi

for f in "${VENDORED_FILES[@]}"; do
    src="$UPSTREAM_DIR/$f"
    if [ ! -f "$src" ]; then
        echo "Missing upstream file: $src" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$VENDORED_DIR/$f")"
    cp "$src" "$VENDORED_DIR/$f"
    echo "  refreshed $f"
done

echo
echo "Done. Review changes with:"
echo "  git diff -- $VENDORED_DIR"
