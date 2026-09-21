#!/usr/bin/env bash

# Build and register LeRobot variants from the same source tree:
#   - roboto-to-lerobot-v2_1  (lerobot==0.3.3 → LeRobot 2.1 format)
#   - roboto-to-lerobot-v3_0  (lerobot==0.5.1 → LeRobot 3.0 format)
#
# ``roboto actions create`` upserts under the hood (falls back to
# ``Action.update`` on conflict), so re-running this after a code change
# advances each action name to the freshly-built image.
#
# Usage:
#   ./scripts/deploy.sh [--variant v2_1|v3_0|all] [<org_id>]
#
# Default variant is ``all`` (deploys both). Org id is read from
# ``$ROBOTO_ORG_ID`` if set, then from the positional argument.
#
# Override the lerobot pin per variant via env vars when smoke-testing a
# new patch release without editing this script:
#   LEROBOT_V2_1_VERSION=0.3.4 ./scripts/deploy.sh --variant v2_1
#   LEROBOT_V3_0_VERSION=0.5.2 ./scripts/deploy.sh --variant v3_0

set -euo pipefail

SCRIPTS_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PACKAGE_ROOT=$(dirname "${SCRIPTS_ROOT}")

# shellcheck source=./variants.sh
source "$SCRIPTS_ROOT/variants.sh"

usage() {
    cat <<EOF
Usage: $0 [--variant v2_1|v3_0|all] [<org_id>]

Options:
  --variant VARIANT   which variant to deploy (default: all).
  -h, --help          show this help and exit.

Positional:
  org_id              Roboto org id; falls back to \$ROBOTO_ORG_ID.

Env overrides:
  LEROBOT_V2_1_VERSION   lerobot pin for the v2_1 variant (default: 0.3.3).
  LEROBOT_V3_0_VERSION   lerobot pin for the v3_0 variant (default: 0.5.1).
EOF
}

VARIANT="all"
positional=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            positional+=("$1")
            shift
            ;;
    esac
done

case "$VARIANT" in
    v2_1|v3_0|all) ;;
    *)
        echo "ERROR: --variant must be one of: v2_1, v3_0, all (got: ${VARIANT})" >&2
        exit 1
        ;;
esac

if [ ! -f "$PACKAGE_ROOT/.venv/bin/roboto" ]; then
    echo "Virtual environment with roboto SDK does not exist. Please run ./scripts/setup.sh first."
    exit 1
fi

# Pick the org_id from $ROBOTO_ORG_ID, then the first positional arg as fallback.
org_id=${ROBOTO_ORG_ID:-}
if [ "${#positional[@]}" -gt 0 ]; then
    org_id=${positional[0]}
fi

roboto_exe="$PACKAGE_ROOT/.venv/bin/roboto"

# ``--org`` is treated as a subcommand-level flag throughout the Roboto
# CLI; mirror that convention everywhere we expand ``${org_args[@]}``.
org_args=()
if [[ -n "$org_id" ]]; then
    org_args+=(--org "$org_id")
fi

deploy_variant() {
    local lerobot_version=$1
    local action_name=$2
    local description=$3
    local local_tag="${action_name}:latest"

    echo "==> Building ${action_name} (lerobot==${lerobot_version})"
    if ! "$SCRIPTS_ROOT/build.sh" \
            --lerobot-version "$lerobot_version" \
            --tag "$local_tag" \
            --quiet; then
        echo "ERROR: build failed for ${action_name} (lerobot==${lerobot_version})" >&2
        exit 1
    fi

    echo "==> Pushing ${local_tag} to Roboto's private registry"
    local image_uri
    if ! image_uri=$(
        "$roboto_exe" \
            --suppress-upgrade-check \
            images push --quiet \
            "${org_args[@]}" \
            "$local_tag"
    ); then
        echo "ERROR: failed to push ${local_tag} to Roboto's private registry" >&2
        exit 1
    fi

    # Both variants register from the same action.json. ``shard_count`` and
    # ``encoder_threads`` are lerobot-0.5-only knobs; the v2_1 image accepts
    # both and ignores them at runtime (see ``_shard_path_supported`` in
    # main.py), so no per-variant patching is needed.
    local action_file="$PACKAGE_ROOT/action.json"

    echo "==> Registering action ${action_name}"
    "$roboto_exe" actions create \
        --from-file "$action_file" \
        --name "$action_name" \
        --description "$description" \
        --image "$image_uri" \
        --yes \
        "${org_args[@]}"
}

if [[ "$VARIANT" == "v2_1" || "$VARIANT" == "all" ]]; then
    deploy_variant "$LEROBOT_V2_1_VERSION" roboto-to-lerobot-v2_1 \
        "Converts a Roboto Collection of events into a LeRobot 2.1 dataset (lerobot ${LEROBOT_V2_1_VERSION}), one episode per event, driven by a contract YAML in the invocation dataset."
fi
if [[ "$VARIANT" == "v3_0" || "$VARIANT" == "all" ]]; then
    deploy_variant "$LEROBOT_V3_0_VERSION" roboto-to-lerobot-v3_0 \
        "Converts a Roboto Collection of events into a LeRobot 3.0 dataset (lerobot ${LEROBOT_V3_0_VERSION}), one episode per event, driven by a contract YAML in the invocation dataset; supports sharded parallel writing via shard_count."
fi
