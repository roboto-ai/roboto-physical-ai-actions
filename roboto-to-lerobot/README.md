# roboto-to-lerobot

Convert data ingested into the Roboto Data Platform into a LeRobot dataset.


The action is driven by a **Roboto Collection of events or of sessions**:
each event or session in the input collection becomes one episode in a single
combined LeRobot dataset, and the events on a session become that episode's
sub-tasks (see [Session collections and sub-tasks](#session-collections-and-sub-tasks)).
The signal/topic schema, alignment strategy, and any pre/post transforms are
described in a `contract.yaml` read from the invocation dataset — see
[CONTRACT.md](CONTRACT.md) for the full schema reference, or
[`contract_demo_reassemble.yaml`](example_notebook/contract_demo_reassemble.yaml) for a worked
example.

The Roboto docs also cover this action:
[Convert Robot Data to a LeRobot Dataset](https://docs.roboto.ai/user-guides/convert-to-lerobot.html)
walks through a conversion end to end from the web UI, and
[Roboto to LeRobot Contract](https://docs.roboto.ai/reference/roboto-to-lerobot-contract.html)
is the published contract schema reference.

The contract format and the ROS message decoders are adapted from Isaac
Blankenau's [Rosetta](https://github.com/iblnkn/rosetta) project (Apache-2.0;
see [`LICENSE-rosetta`](src/roboto_to_lerobot/runtime/LICENSE-rosetta)).

Two action variants are deployed from the same source tree, one per LeRobot
dataset format:

| Action name              | LeRobot package | LeRobot dataset format |
|--------------------------|-----------------|------------------------|
| `roboto-to-lerobot-v2_1`   | `lerobot==0.3.x` | 2.1                    |
| `roboto-to-lerobot-v3_0`   | `lerobot==0.7.x` | 3.0                    |

There is one `src/` tree. The variants differ only by the lerobot version
baked into the image at build time, via the `LEROBOT_VERSION` build arg — so
**the image decides which variant runs, not the parameters**. Pick one by
action name when invoking on hosted compute, or with
`scripts/invoke_local.sh --variant` when running locally; the
version-to-variant mapping lives in [`scripts/variants.sh`](scripts/variants.sh).

## Table of Contents

- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Parameters](#parameters)
  - [Running](#running)
    - [Local Invocation](#local-invocation)
    - [Hosted Invocation](#hosted-invocation)
- [Session collections and sub-tasks](#session-collections-and-sub-tasks)
- [Lineage](#lineage)
- [Lance output](#lance-output)
- [LeRobot 0.7 in roboto-to-lerobot-v3_0](#lerobot-07-in-roboto-to-lerobot-v3_0)
- [Authoring a contract](#authoring-a-contract)
- [Development](#development)
- [Deployment](#deployment)

## Quick Start

### Prerequisites

- **Docker** (Engine 19.03+): Local invocation always runs in Docker for production parity
- **Python 3**: A supported version (see [.python-version](.python-version))

```bash
$ docker --version
$ python3 --version
```

### Installation

Set up a virtual environment and install dependencies with the following command:

```bash
$ ./scripts/setup.sh
```

You must be setup to [access Roboto programmatically](https://docs.roboto.ai/getting-started/programmatic-access.html). Verify with the following command:
```bash
$ .venv/bin/roboto users whoami
```

### Parameters

| Name            | Required | Default | Description                                                                                                                                                            |
|-----------------|----------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `collection_id` | yes      | —       | ID of a Roboto Collection whose `resource_type` is `event` or `session`. Each event or session in the collection becomes one episode in the output dataset. A session's window is its aggregate time bounds; a session without bounds falls back to the windows it holds of its files, or to the span of the contract topics on them. Sessions that re-anchor a file with a `unix_epoch_offset_ns` are rejected. The collection's version observed at invocation time is recorded in the output manifest for reproducibility. For session collections, each `episode_to_event` / `skipped_events` entry in the manifest carries `session_id` and `source_dataset_ids`, with `event_id` set to `null`. |
| `output_tags`   | no       | — | Comma-separated tags for the output dataset, applied once the converted files are uploaded, so a `dataset.tag_added` trigger sees the whole dataset. A tag the dataset already has is removed and added again so its trigger fires. Check which triggers in your org fire on a tag before using it. |
| `subtask_events` | no      | every event | Which events on a session become its episode's sub-tasks, by event name or tag: comma-separated glob patterns on the name, or on the tags when written `tag:<glob>`, where a leading `!` excludes. `grasp*,lift` uses only those events; `!qa/*` uses every event except those under `qa/`; `tag:subtask` uses the events tagged `subtask`; `grasp*,!tag:qa` uses the `grasp*` events not tagged `qa`. See [Session collections and sub-tasks](#session-collections-and-sub-tasks). |
| `contract`      | no       | `contract.yaml` (auto-discovered) | Path to the contract YAML inside the invocation dataset. Describes the topics, alignment, and transforms used to build each episode. See [CONTRACT.md](CONTRACT.md) for the full schema, or [`contract_demo_reassemble.yaml`](example_notebook/contract_demo_reassemble.yaml) for a worked example. |
| `pool_size`     | no       | auto-sized | Worker processes that read and process events in parallel, per shard. Left unset it auto-sizes to `min(16, cpu_count / shard_count)` — on the default 16-vCPU compute with `shard_count=3` that is `5`. Any explicit value is used verbatim and is not capped. |
| `encoder_threads` | no     | `2` (built in) | Level of parallelism given to the SVT-AV1 video encoder, per camera stream. Higher values encode each camera faster. The export waits for the encoder rather than dropping frames, so a low value costs time, not frames. Left unset it uses the built-in `2`. Only `roboto-to-lerobot-v3_0` acts on it; `roboto-to-lerobot-v2_1` (lerobot 0.3.3) accepts it and silently ignores it, having no equivalent knob. |
| `video_crf` | no | `30` (lerobot default) | Video quality for every camera, as the SVT-AV1 constant rate factor: an integer from 1 to 63. Lower values keep more detail and make larger files. Only `roboto-to-lerobot-v3_0` accepts it; `roboto-to-lerobot-v2_1` fails the run if it is set. |
| `video_preset` | no | `12` (lerobot default) | SVT-AV1 speed preset for every camera: an integer from 0 to 13. Lower values compress better but encode much more slowly, which lengthens the run. Only `roboto-to-lerobot-v3_0` accepts it. |
| `video_gop` | no | `2` (lerobot default) | Keyframe interval for every camera, in frames: an integer of 1 or more. Larger values make much smaller files, and reading one frame at random takes longer because decoding starts at the keyframe before it; every frame still decodes exactly. Only `roboto-to-lerobot-v3_0` accepts it. |
| `shard_count`   | no       | `3`     | Parallel writer shards. Each shard writes a partial dataset independently, and the shards are then merged by stream-copy with no re-encoding. Sharded writing needs `lerobot.datasets.aggregate.aggregate_datasets`, which lerobot 0.3.x lacks, so without the episode cache only `roboto-to-lerobot-v3_0` acts on it: `roboto-to-lerobot-v2_1` accepts it and ignores it, falling back to a single writer and logging a warning. With `episode_cache=true`, both variants use it as the maximum number of processes that convert the episodes missing from the cache. Without the episode cache, sharding also groups the merged dataset's episodes by shard rather than strictly chronologically; each episode's original `start_time_ns` is preserved in the output manifest. |
| `episode_cache` | no | `false` | Set to `true` to cache each converted episode in the org's assets, so a re-export converts only new or changed events or sessions. Caching requires org asset storage to be enabled for your org; where it is not enabled, this parameter has no effect: the action logs a warning and converts every event. Each episode is stored once, as a one-episode LeRobot dataset at `.lerobot-cache/<recipe_hash>/<episode_key>/`, a hidden folder in the org's assets. The recipe hash covers the contract, the converter source, the lerobot version, the output features and the video encoder settings, so changing `video_crf`, `video_preset` or `video_gop` converts every episode again. The episode key covers the event or session (id, bounds, task, and the source dataset of an event) and the source topics it reads, so an edited event, a session whose bounds or files changed, or a re-uploaded file is converted again. At most `shard_count` processes convert the episodes missing from the cache, and each new entry is uploaded as soon as it is converted. The export then merges cached and new episodes in chronological order without re-encoding: `roboto-to-lerobot-v3_0` stream-copies them into its packed files, and `roboto-to-lerobot-v2_1` moves each episode's files into place and renumbers the episode. The dataset's [lineage](#lineage) records which episodes came from the cache. Both variants act on it, each caching episodes in its own LeRobot format. |
| `output_format` | no | `lerobot` | `lerobot` writes the LeRobot dataset to `combined/`; `lance` writes its [Lance layout](#lance-output) to `lance/` instead; `both` writes both. Only `roboto-to-lerobot-v3_0` supports `lance` and `both`; `roboto-to-lerobot-v2_1` fails the run before converting. |

### Running

The invocation dataset must contain a contract YAML (default name `contract.yaml`)
that the action will load to drive conversion.

#### Local Invocation

> **Note:** For complete local invocation documentation and examples, see [DEVELOPING.md](DEVELOPING.md#invoking-locally).

Use `scripts/invoke_local.sh` and name the variant you want. It builds that
variant and runs it through `roboto actions invoke-local`:

```bash
$ ./scripts/invoke_local.sh --variant v2_1 \
    --dataset=<INVOCATION_DATASET_ID> \
    --parameter collection_id="<COLLECTION_ID>" \
    --parameter contract="contract.yaml" \
    --dry-run
```

`--variant` takes `v2_1` or `v3_0` (default `v3_0`); every other argument is
forwarded to the CLI unchanged. Add `--skip-build` to re-run against an
already-built variant image and skip the image rebuild.

The wrapper exists because the variant is decided by the lerobot version baked
into the image at build time, and `roboto actions invoke-local <dir>` always
rebuilds the directory with a plain `docker build`, with no way to pass
`--build-arg`. Run directly it can therefore only ever produce the Dockerfile's
default pin — the `v3_0` variant:

```bash
# Equivalent to --variant v3_0. Cannot be made to run v2_1.
$ .venv/bin/roboto --log-level=info actions invoke-local \
    --dataset=<INVOCATION_DATASET_ID> \
    --parameter collection_id="<COLLECTION_ID>" \
    --parameter contract="contract.yaml" \
    --dry-run
```

_Running without `--dry-run` may have side-effects, depending on how this action is implemented! See relevant section in [DEVELOPING.md](DEVELOPING.md#code-organization-best-practices) for more._

Full usage:
```bash
$ .venv/bin/roboto actions invoke-local --help
```

#### Hosted Invocation

> **Note:** To run this action on Roboto's hosted compute, you must first build and deploy it. See relevant section in [DEVELOPING.md](DEVELOPING.md#build-and-deployment) for more.

The action name selects the variant — that is the only thing that does:
```bash
$ .venv/bin/roboto actions invoke \
    --dataset=<INVOCATION_DATASET_ID> \
    --parameter collection_id="<COLLECTION_ID>" \
    roboto-to-lerobot-v3_0
```


Full usage:
```bash
$ .venv/bin/roboto actions invoke --help
```

## Session collections and sub-tasks

An event episode covers the event's time window, read from the event's first
associated dataset. A session episode covers the session's time window, read
from the session's own files only, which may span several datasets.

Events on a session become that episode's **sub-tasks**, named after the
event, in the `language_persistent` column that lerobot 0.6 and later read.
The session's own events are used, or, in an org without session events, the
events on its files. Every event is used unless `subtask_events` narrows
them by name or tag. Frames no event covers get the sub-task `no_subtask`,
and `manifest.json` records every segment under `subtasks`. An episode leaves
out the frames for which a stream has no data and numbers the rest without
them. Its `episode_to_event` entry in the manifest lists the left-out frames
under `dropped_frames`, each as its position before the drop: position `i`
is the frame `i / fps` seconds after `start_time_ns`. Sub-tasks stay on the
frames their events cover. Datasets whose sessions have no events that
`subtask_events` selects are unchanged. Only `roboto-to-lerobot-v3_0` writes
sub-tasks: lerobot 0.3.3 has no column for them, so `roboto-to-lerobot-v2_1`
logs a warning and writes none.

Session episodes go through the rest of the export as event episodes do:

- **Episode cache.** A session's episode is cached under the session's id,
  bounds, task and the topics of its files. Sub-tasks are not part of a
  cache entry: the export writes them after it has merged cached and new
  episodes. Adding, renaming or moving an event on a session therefore
  changes the sub-tasks of the next export without converting the session
  again.
- **Lineage.** The [lineage](#lineage) names each episode's session in
  `session_id` and leaves `event_id` empty.
- **Lance output.** The [Lance layout](#lance-output) keeps the sub-task
  columns.

## Lineage

Every export records where its episodes came from inside the LeRobot
dataset's `meta/` folder, which travels with the dataset when you copy it or
convert it to Lance:

- `meta/roboto_lineage.json` (both variants) names the invocation, the
  collection and its version, the contract (name, version, sha256), the
  lerobot version and the recipe hash, and lists each episode's event id or
  session id (the other is null), source dataset id, start and end time in
  nanoseconds, and whether it came from the episode cache (with its
  `episode_key` when the cache was on).
- In a `roboto-to-lerobot-v3_0` dataset, the episode metadata also has the
  columns `roboto/event_id`, `roboto/session_id`, `roboto/source_dataset_id`,
  `roboto/start_time_ns`, `roboto/end_time_ns`, `roboto/cache_hit` and
  `roboto/recipe_hash`, so LeRobot can select episodes by their source:

  ```python
  from lerobot.datasets.lerobot_dataset import LeRobotDataset

  ds = LeRobotDataset(
      "my-export", root="combined",
      episode_filter=lambda ep: ep["roboto/source_dataset_id"] == "<DATASET_ID>",
  )
  ```

  The columns survive LeRobot 0.7's `aggregate_datasets` and the Lance
  converter; `meta/roboto_lineage.json` survives only the Lance converter.
  LeRobot's `delete_episodes` and `split_dataset` keep neither.

The invocation folder also has a `manifest.json`, which maps each episode
to its event or session.

## Lance output

With `output_format=lance` or `both`, `roboto-to-lerobot-v3_0` writes the
dataset in LeRobot's Lance storage format to `lance/`, instead of or next to
the LeRobot dataset in `combined/`.
After the LeRobot dataset is built (with or without the episode cache), the
action converts it with the converter from the
[`lerobot-lancedb`](https://github.com/lancedb/lerobot-lancedb) package,
which LeRobot's `lancedb-convert` extra installs (its command-line form is
`lerobot-lance-convert`). The converter copies `meta/` (with the
[lineage](#lineage)), stores the frames in one Lance table and the bytes of
every video file, unchanged, in another, so the Lance layout is about the
size of the LeRobot dataset. It converts the whole dataset on every export,
even when the episode cache supplied every episode.

- **Reading.** Reading the Lance layout needs LeRobot >= 0.7.0, with
  torchcodec for the videos:
  `LeRobotDataset("<name>", root="<local path or fsspec URI such as s3://...>")`
  picks the Lance reader from `meta/info.json` (`"storage_format": "lance"`).
- **Training from object storage.** A training job can read the Lance layout in
  place from object storage with random access (a global shuffle) instead of
  downloading it first. The Lance reader lists the dataset's files, which the
  read-only credentials Roboto issues for a dataset do not allow, so it
  cannot read the copy in Roboto in place: download `lance/`, or copy it to a
  bucket you control, and read it from there.
- **Read-only.** LeRobot's `push_to_hub`, `aggregate_datasets` and dataset
  editing tools work only on the LeRobot dataset, so keep it (`both`) if you
  need them.

## LeRobot 0.7 in roboto-to-lerobot-v3_0

`roboto-to-lerobot-v3_0` writes its datasets with `lerobot==0.7.0`; it wrote
them with 0.6.1 before. The dataset format is still LeRobot 3.0, so datasets
from either version load in both.

- **The episode cache fills again once.** The recipe hash covers the lerobot
  version, so the first export after the upgrade finds none of the entries
  0.6.1 wrote and converts every event; later exports hit the new entries.
  The old entries stay in `.lerobot-cache/` until you delete them.
- **Merged quantiles are wider.** LeRobot computes quantiles (`q01`, `q10`,
  `q50`, `q90`, `q99`) per episode. To combine them into `meta/stats.json`,
  0.6 averaged each quantile across episodes, weighted by frame count; 0.7
  takes the envelope instead: each lower quantile (`q01`, `q10`, `q50`) is
  the smallest across episodes and each upper one (`q90`, `q99`) the largest.
  Over many episodes `q01` and `q99` come close to `min` and `max`, and
  `q50` is the smallest episode median, not a median of the dataset. This
  changes training only for policies that normalize with `QUANTILES`
  (`q01`/`q99`, for example pi0.5) or `QUANTILE10` (`q10`/`q90`);
  `MEAN_STD` and `MIN_MAX` use `mean`, `std`, `min` and `max`, which keep
  their meaning. Every export path (one writer, shards, the episode cache)
  combines episodes the same way, so the same events give the same stats
  either way. To normalize with dataset-wide quantiles instead, compute
  them from the data before training.
- **Per-episode statistics are computed in float64.** 0.6 computed them in
  float32, which could put the `std` of a nearly constant dimension off by
  an order of magnitude.

## Authoring a contract

Every invocation needs a contract YAML in the invocation dataset that
describes the streams, alignment, and transforms used to build each episode.
The full schema — top-level fields, observation / action / video specs,
selectors, alignment methods, transforms, role-based binding, and the list
of supported message types — is documented in [CONTRACT.md](CONTRACT.md),
and in the Roboto docs as
[Roboto to LeRobot Contract](https://docs.roboto.ai/reference/roboto-to-lerobot-contract.html).

A short worked example you can copy from is
[`contract_demo_reassemble.yaml`](example_notebook/contract_demo_reassemble.yaml).

## Development

See [DEVELOPING.md](DEVELOPING.md) for detailed information about developing this action, including:
- Project structure and key files
- Local invocation
- Adding dependencies (runtime, system, and development)
- Working with action parameters (including secrets)
- Handling input and output data
- Building and deploying to Roboto

## Deployment

Build and deploy both action variants (v2_1 and v3_0) to the Roboto Platform with:

```bash
$ ./scripts/deploy.sh
```

To deploy only one variant:

```bash
$ ./scripts/deploy.sh --variant v2_1
$ ./scripts/deploy.sh --variant v3_0
```

`deploy.sh` calls `build.sh` once per variant with the matching `--lerobot-version`
pin, pushes the image, and registers the action. To override the lerobot pin
per variant when smoke-testing a new patch release, set
`LEROBOT_V2_1_VERSION` / `LEROBOT_V3_0_VERSION` in the environment.
