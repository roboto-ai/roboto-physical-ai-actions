#!/usr/bin/env bash

# Host-side driver for the docker bag-replay smoke.
#
# What this does, in order:
#   1. Fetches an MCAP fixture from the roboto SDK into test/fixtures/
#      if not already present (one-time per developer machine).
#   2. Builds the docker image from Dockerfile.ros2_humble.
#   3. Runs the container, bind-mounting the fixture and the smoke
#      harness, executing ``run_in_container.sh``.
#
# Exit codes:
#   0 — the generated node published a message on the action topic
#       within the timeout
#   1 — any step failed (fetch, build, or smoke)
#
# Env vars:
#   ROS_SMOKE_IMAGE_TAG       image tag to build (default: roboto-ros-smoke)
#   ROS_SMOKE_TIMEOUT         seconds the in-container wait will tolerate
#                             before declaring failure (default: 30)
#   ROBOTO_PROFILE            roboto SDK profile for the fixture fetch,
#                             read by the roboto SDK (default: your
#                             configured profile)
#   ROBOTO_ORG_ID             org that owns the dataset, read by the
#                             roboto SDK (only needed if your user
#                             belongs to more than one org)
#   ROBOTO_ROS_SMOKE_DATASET  dataset id to fetch the MCAP from, read by
#                             fetch_fixture.py (required, no default; the
#                             dataset must hold an MCAP carrying the topics
#                             contract.yaml declares)

set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
PACKAGE_ROOT="$( cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd)"
FIXTURES_DIR="$PACKAGE_ROOT/test/fixtures"

IMAGE_TAG="${ROS_SMOKE_IMAGE_TAG:-roboto-ros-smoke}"
TIMEOUT_SECONDS="${ROS_SMOKE_TIMEOUT:-30}"

# Step 1: fetch the MCAP fixture if missing. The fetcher itself is
# idempotent (skips if the file already exists), but a top-level guard
# lets us skip the python subprocess on warm runs.
#
# Glob into an array (nullglob => empty when nothing matches) rather than
# parsing `ls` output — robust against filenames with spaces/newlines.
mkdir -p "$FIXTURES_DIR"
shopt -s nullglob
fixtures=("$FIXTURES_DIR"/*.mcap)
shopt -u nullglob
if (( ${#fixtures[@]} == 0 )); then
    echo "run.sh: no mcap in $FIXTURES_DIR — running fetch_fixture.py"
    venv_python="$PACKAGE_ROOT/.venv/bin/python"
    if [[ -x "$venv_python" ]]; then
        python="$venv_python"
    else
        python="python3"
    fi
    "$python" "$SCRIPT_DIR/fetch_fixture.py"
    shopt -s nullglob
    fixtures=("$FIXTURES_DIR"/*.mcap)
    shopt -u nullglob
fi

if (( ${#fixtures[@]} == 0 )); then
    echo "run.sh: fetch_fixture.py ran but no .mcap appeared in $FIXTURES_DIR" >&2
    exit 1
fi

# Pick the first .mcap deterministically — fetch_fixture writes only
# one, but a developer who pre-staged multiple shouldn't get a non-
# deterministic pick. Sort the glob results (handles spaces).
mapfile -t sorted_fixtures < <(printf '%s\n' "${fixtures[@]}" | sort)
fixture_path="${sorted_fixtures[0]}"
echo "run.sh: fixture=$fixture_path"

# Step 2: build the docker image. Build context is the package root so
# COPY pyproject.toml / src/ / test/ros_smoke/ all resolve.
echo "run.sh: docker build -t $IMAGE_TAG"
docker build \
    -t "$IMAGE_TAG" \
    -f "$SCRIPT_DIR/Dockerfile.ros2_humble" \
    "$PACKAGE_ROOT"

# Step 3: run the smoke inside the container.
# Bind mounts:
#   $fixture_path → /fixture.mcap (read-only)
#   $SCRIPT_DIR    → /smoke (read-write so the generated node.py and
#                    the colcon ws/ can land there; on success the
#                    artefacts are useful for diagnosis)
echo "run.sh: docker run (timeout=${TIMEOUT_SECONDS}s)"
docker run --rm \
    -v "$fixture_path:/fixture.mcap:ro" \
    -v "$SCRIPT_DIR:/smoke" \
    -e "ROS_SMOKE_TIMEOUT=$TIMEOUT_SECONDS" \
    "$IMAGE_TAG" \
    bash /smoke/run_in_container.sh
