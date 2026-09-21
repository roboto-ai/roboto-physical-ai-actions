# lerobot-v3-to-v2_1

A Roboto Action that converts an uploaded **LeRobot v3.0** dataset into a
**LeRobot v2.1** dataset.

LeRobot v3.0 consolidates many episodes into a few shared parquet/MP4 files; v2.1
keeps one file per episode. This action reverses the consolidation:

- **De-aggregates** the shared data parquet back into one `episode_XXXXXX.parquet`
  per episode (slicing by the v3 `dataset_from_index`/`dataset_to_index`).
- **Splits** each concatenated per-camera video back into one `episode_XXXXXX.mp4`
  per episode, **losslessly and frame-exact** — a plain stream-copy where the
  episode boundary is keyframe-aligned, an exact re-encode of just that segment
  otherwise.
- **Rewrites** the metadata into the v2.1 schema: `info.json` (legacy path
  templates, `total_chunks`/`total_videos`, v3 size hints removed),
  `tasks.jsonl`, `episodes.jsonl`, and `episodes_stats.jsonl` (v3 quantile stats
  dropped to the legacy `min/max/mean/std/count`).

The conversion is **non-destructive**: the input dataset is read-only and the
v2.1 tree is written to the action's output directory, then uploaded back to
Roboto as a new dataset.

The core conversion is vendored from the MIT-licensed
[any4lerobot](https://github.com/Tavish9/any4lerobot) project; see
[`src/lerobot_v3_to_v2_1/convert/`](src/lerobot_v3_to_v2_1/convert/) and
[DEVELOPING.md](DEVELOPING.md#vendored-conversion-code).

## Behaviour

- **Already v2.1** input is copied through unchanged, so the action is safe to run
  unconditionally in a pipeline.
- **Image-in-parquet** datasets (camera frames stored as `dtype: image` rows
  rather than video) are **not yet supported** and are rejected with a clear
  error. Only video-backed v3.0 datasets convert.
- Any non-`v2.1`/`v3.0` `codebase_version` is rejected with a clear error.

## Parameters

| Name | Required | Description |
|---|---|---|
| `output_dataset_id` | no | Upload the v2.1 result into this existing Roboto dataset instead of creating a new one. By default the action creates a new dataset. |

The action locates the LeRobot dataset under its input directory (via
`meta/info.json`) and converts it.

## Inputs & Outputs

- **Input**: the files of a LeRobot v3.0 dataset, supplied via a file query or a
  dataset id at invocation. `requires_downloaded_inputs` is `true`, so the files
  are downloaded into the working directory before the action runs.
- **Output**: by default the action **creates a new Roboto dataset** named
  `"<source name> (LeRobot v2.1)"` (or `"LeRobot v2.1 conversion of <id>"` if the
  source is unnamed), uploads the v2.1 tree to its root, and stamps **provenance
  metadata** linking back to the source — top-level `source_dataset_id` /
  `codebase_version` plus an `invocations.<id>` block with the lerobot version,
  vendored commit, and episode/video counts. The **source dataset is never
  modified**. Pass `output_dataset_id` to upload into an existing dataset instead.
  Load the result with `LeRobotDataset(repo_id, root="<dataset-root>")` using a
  v2.1-capable lerobot.

## Quick Start

### Prerequisites

- **Docker** (Engine 19.03+): local invocation runs in Docker for production parity.
- **Python 3.12** (see [.python-version](.python-version)).

### Installation

```bash
$ ./scripts/setup.sh
$ .venv/bin/roboto users whoami   # verify programmatic access
```

### Local invocation

```bash
$ .venv/bin/roboto --log-level=info actions invoke-local \
    --file-query="dataset_id='ds_abc123'" \
    --dry-run
```

### Hosted invocation

```bash
$ .venv/bin/roboto actions invoke \
    --file-query="dataset_id='ds_abc123'" \
    lerobot-v3-to-v2_1
```

## Development & Deployment

See [DEVELOPING.md](DEVELOPING.md). In short:

```bash
$ ./scripts/verify.sh                 # ruff + pytest
$ ./scripts/build.sh                  # docker image
$ ./scripts/deploy.sh # push image + register action
```
