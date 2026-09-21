# roboto-to-lerobot

Convert data ingested into the Roboto Data Platform into a LeRobot dataset.


The action is driven by a **Roboto Collection of events**: each event in the
input collection becomes one episode in a single combined LeRobot dataset.
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
| `roboto-to-lerobot-v3_0`   | `lerobot==0.5.x` | 3.0                    |

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
| `collection_id` | yes      | —       | ID of a Roboto Collection whose `resource_type` is `event`. Each event in the collection becomes one episode in the output dataset. The collection's version observed at invocation time is recorded in the output manifest for reproducibility. |
| `contract`      | no       | `contract.yaml` (auto-discovered) | Path to the contract YAML inside the invocation dataset. Describes the topics, alignment, and transforms used to build each episode. See [CONTRACT.md](CONTRACT.md) for the full schema, or [`contract_demo_reassemble.yaml`](example_notebook/contract_demo_reassemble.yaml) for a worked example. |
| `pool_size`     | no       | auto-sized | Worker processes that read and process events in parallel, per shard. Left unset it auto-sizes to `min(16, cpu_count / shard_count)` — on the default 16-vCPU compute with `shard_count=3` that is `5`. Any explicit value is used verbatim and is not capped. |
| `encoder_threads` | no     | `2` (built in) | Level of parallelism given to the SVT-AV1 video encoder, per camera stream. Must be `>= 2`: lower values risk the encoder's internal frame queue overflowing under bursty writes, silently dropping frames from the output video. Left unset it uses the built-in `2`. Only `roboto-to-lerobot-v3_0` acts on it; `roboto-to-lerobot-v2_1` (lerobot 0.3.3) accepts it and silently ignores it, having no equivalent knob. |
| `shard_count`   | no       | `3`     | Parallel writer shards. Each shard writes a partial dataset independently, and the shards are then merged by stream-copy with no re-encoding. Sharded writing needs `lerobot.datasets.aggregate.aggregate_datasets`, which is lerobot 0.5.x only, so only `roboto-to-lerobot-v3_0` acts on it — `roboto-to-lerobot-v2_1` accepts it and ignores it, falling back to a single writer and logging a warning. Sharding also groups the merged dataset's episodes by shard rather than strictly chronologically; each episode's original `start_time_ns` is preserved in the output manifest. |

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
