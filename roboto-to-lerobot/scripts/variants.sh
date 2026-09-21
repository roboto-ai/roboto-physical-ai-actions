#!/usr/bin/env bash
#
# Single source of truth for the variant -> lerobot version mapping.
#
# Sourced by deploy.sh and invoke_local.sh. Override either pin in the
# environment to smoke-test a new patch release without editing this file:
#
#   LEROBOT_V2_1_VERSION=0.3.4 ./scripts/deploy.sh --variant v2_1
#
# The Dockerfile carries its own default for LEROBOT_VERSION, used only by
# builds that cannot pass --build-arg (a bare `docker build`, and
# `roboto actions invoke-local` run directly against this directory). Keep
# that default in step with LEROBOT_V3_0_VERSION below.

LEROBOT_V2_1_VERSION="${LEROBOT_V2_1_VERSION:-0.3.3}"
LEROBOT_V3_0_VERSION="${LEROBOT_V3_0_VERSION:-0.5.1}"

# Echo the lerobot version for a variant name; exit 1 on an unknown variant.
lerobot_version_for_variant() {
    case "$1" in
        v2_1) echo "$LEROBOT_V2_1_VERSION" ;;
        v3_0) echo "$LEROBOT_V3_0_VERSION" ;;
        *)
            echo "ERROR: unknown variant '$1' (expected v2_1 or v3_0)" >&2
            return 1
            ;;
    esac
}
