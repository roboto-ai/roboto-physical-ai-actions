#!/usr/bin/env bash
#
# Run `roboto actions invoke-local` against a chosen variant (v2_1 or v3_0).
#
# Why this wrapper exists
# -----------------------
# The two variants differ only by the lerobot version baked into the image at
# build time, via the Dockerfile's LEROBOT_VERSION build arg. But
# `roboto actions invoke-local <dir>` always rebuilds that directory with a
# plain `docker build` and exposes no way to pass --build-arg, so run directly
# it can only ever produce the Dockerfile's default pin (v3_0).
#
# So: build the variant here, where the build arg *can* be set, tag it
# roboto-to-lerobot-<variant>:latest, and hand the CLI a throwaway action
# directory whose Dockerfile is a bare `FROM` of that tag. The CLI's mandatory
# rebuild becomes a sub-second passthrough and the container it runs is the
# variant image. Everything else -- input staging, parameters, --dry-run --
# is ordinary invoke-local behaviour.
#
# Usage:
#   ./scripts/invoke_local.sh --variant v2_1 --dataset=ds_xxx --dry-run
#   ./scripts/invoke_local.sh --variant v2_1 --skip-build --dataset=ds_xxx
#
# --variant defaults to v3_0. Every other argument is forwarded verbatim to
# `roboto actions invoke-local`. --skip-build reuses an already-built variant
# image, which turns a re-run into seconds rather than a full image rebuild.

set -euo pipefail

SCRIPTS_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PACKAGE_ROOT=$(dirname "${SCRIPTS_ROOT}")
# shellcheck source=./variants.sh
source "$SCRIPTS_ROOT/variants.sh"

VARIANT="v3_0"
SKIP_BUILD=false
passthrough=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        -h|--help)
            sed -n '3,27p' "$0"
            exit 0
            ;;
        *)
            passthrough+=("$1")
            shift
            ;;
    esac
done

lerobot_version=$(lerobot_version_for_variant "$VARIANT")
image_tag="roboto-to-lerobot-${VARIANT}:latest"

roboto_exe="$PACKAGE_ROOT/.venv/bin/roboto"
if [[ ! -x "$roboto_exe" ]]; then
    echo "ERROR: $roboto_exe not found. Run ./scripts/setup.sh first." >&2
    exit 1
fi

if [[ "$SKIP_BUILD" == true ]]; then
    if ! docker image inspect "$image_tag" &> /dev/null; then
        echo "ERROR: --skip-build given but $image_tag does not exist locally." >&2
        echo "Drop --skip-build to build it." >&2
        exit 1
    fi
    echo "==> Reusing existing $image_tag (--skip-build)"
else
    echo "==> Building $VARIANT (lerobot==${lerobot_version}) as $image_tag"
    "$SCRIPTS_ROOT/build.sh" --lerobot-version "$lerobot_version" --tag "$image_tag"
fi

# Throwaway action directory: action.json gives the CLI the action name it
# needs, and the one-line Dockerfile makes its forced rebuild a no-op layer
# over the variant image we just built.
shim_dir=$(mktemp -d "${TMPDIR:-/tmp}/roboto-to-lerobot-${VARIANT}.XXXXXX")
trap 'rm -rf "$shim_dir"' EXIT
cp "$PACKAGE_ROOT/action.json" "$shim_dir/action.json"
printf 'FROM %s\n' "$image_tag" > "$shim_dir/Dockerfile"

echo "==> Invoking $VARIANT locally"
# Deliberately not `exec`: that would replace this shell and the EXIT trap
# above would never run, leaking the shim directory into $TMPDIR. Run it as a
# child, then exit with its status so callers still see the CLI's exit code.
"$roboto_exe" actions invoke-local "$shim_dir" "${passthrough[@]+"${passthrough[@]}"}"
exit $?
