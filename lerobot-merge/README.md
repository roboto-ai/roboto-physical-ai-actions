# lerobot-merge

Merge N sharded LeRobot datasets (produced by parallel `roboto-to-lerobot` invocations) into one merged LeRobot dataset, server-side on Roboto compute.

The merge calls `lerobot.datasets.aggregate.aggregate_datasets`:
- Videos are stream-copy concatenated via ffmpeg's concat demuxer (or pure `shutil.copy` when the destination would overflow `video_files_size_in_mb`) — **no re-encode**.
- Parquet files are read once to rewrite the `episode_index`, `index`, and `task_index` columns into the merged global numbering.
- Tasks are unioned by string.
- Per-feature stats are aggregated across shards.

Running this action server-side instead of in a notebook keeps the bytes inside AWS — no WAN round-trip for video.

## Inputs

The shards dataset is consumed via the invocation's **`data_source`** binding (not as an opaque parameter). The runtime pre-downloads every file matched by `input_data` into `context.input_dir`, and the action reads each shard from there.

Pass on `Action.invoke(...)`:

- `data_source_id` — the dataset that holds every shard's LeRobot output.
- `input_data` — a list of glob patterns, typically one `"<iv>/**"` per shard, that covers each shard's `combined/`, `manifest.json`, and `contract.yaml`.

## Parameters

| Name | Required | Description |
|---|---|---|
| `shard_invocation_ids` | yes | JSON array of conversion invocation IDs, in the order they should be merged. Each ID names a per-invocation prefix in the `data_source` dataset, written by the conversion action as `<iv>/combined/`, `<iv>/manifest.json`, and `<iv>/contract.yaml`. Merge order is load-bearing: shards are stacked in input order and the merged manifest's `episode_to_event` indices are offset accordingly. Example: `'["iv_abc", "iv_def"]'`. |
| `parent_collection_id` | yes | ID of the parent event collection that was partitioned into sub-collections before fan-out. Stamped onto the merged manifest as `collection_id`. |
| `parent_collection_version` | no | Version of the parent collection at partition time. Recorded on the merged manifest as `collection_version`. |
| `data_files_size_in_mb` | no | Passthrough to `aggregate_datasets`. Default: lerobot's default (100). |
| `video_files_size_in_mb` | no | Passthrough to `aggregate_datasets`. Default: lerobot's default (200). Set to `0` to force every source video file to be a pure `shutil.copy` (no MP4 container rewrites). |
| `chunk_size` | no | Passthrough to `aggregate_datasets` (max files per `chunk-XXX/`). Default: lerobot's default (1000). |

## Output

Written to the action's output directory, auto-uploaded to the dataset specified via `upload_destination` at invoke time:

- `<merge_iv>/combined/` — the merged LeRobot dataset (load with `LeRobotDataset(repo_id, root="…/combined")`).
- `<merge_iv>/manifest.json` — self-contained provenance: parent collection id/version, contract identity, per-shard episode_to_event renumbered to merged global indices, dedup union, skipped events union, aggregate timing.
- `<merge_iv>/contract.yaml` — archived contract bytes (copied from the first shard that has one; all shards must share the same `contract.sha256`).

## Invocation example (Python SDK)

```python
import json
from roboto.domain import actions

merge = actions.Action.from_name("lerobot-merge")
iv = merge.invoke(
    invocation_source=actions.InvocationSource.Manual,
    data_source_id=shards_dataset_id,
    input_data=[f"{iv.id}/**" for iv in conversion_invocations],
    upload_destination=actions.InvocationUploadDestination.dataset(merged_dataset_id),
    parameter_values={
        "shard_invocation_ids": json.dumps([iv.id for iv in conversion_invocations]),
        "parent_collection_id": parent_collection_id,
        "parent_collection_version": str(parent_collection_version),
    },
)
iv.wait_for_terminal_status(timeout=3600)
```

## Local development

```bash
./scripts/setup.sh
./scripts/verify.sh   # ruff + pytest
./scripts/build.sh    # docker build
./scripts/deploy.sh   # push image + register/update action
```
