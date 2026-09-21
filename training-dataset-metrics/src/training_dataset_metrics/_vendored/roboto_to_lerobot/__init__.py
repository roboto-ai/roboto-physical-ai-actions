# Vendored subset of `roboto-to-lerobot` (https://github.com/roboto-ai/roboto-physical-ai-actions)
# kept in-tree so this action has no cross-package Docker build dependency.
#
# DO NOT EDIT THESE FILES DIRECTLY. Edit upstream at `roboto-to-lerobot/src/roboto_to_lerobot/`
# and re-run `scripts/sync_vendored.sh` to propagate changes.
#
# Upstream's own `__init__.py` imports `.main`, which pulls in a Roboto Action
# entrypoint we don't need here. We override it with an empty init so the
# vendored subset exposes only the modules that `contract_utils` pulls in.
