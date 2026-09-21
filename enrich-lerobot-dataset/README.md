# Enrich LeRobot Dataset

An example Roboto Action that demonstrates one approach to adding features to LeRobot datasets stored in Roboto. This example shows how to:
- Load LeRobot datasets from Roboto storage
- Process and compute derived features from existing dataset data
- Use LeRobot's dataset tools to create enriched datasets
- Write the enriched results to the action's output directory so Roboto uploads them

In this specific example, the action adds an `action_observation_difference` feature to LeRobot datasets by computing the element-wise difference between `action` and `observation.state` vectors for each frame. This pattern can be adapted to add any custom features or transformations to your LeRobot datasets.

The output is an enriched copy of the LeRobot dataset, written to the action's output directory under `<repo_id>_enriched/`. On hosted invocation Roboto uploads it to the invocation's output dataset, or, if none was specified, to the dataset the input files came from; it does not create a dataset of its own.

## Table of Contents

- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running](#running)
    - [Local Invocation](#local-invocation)
    - [Hosted Invocation](#hosted-invocation)
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

### Running

#### Local Invocation

> **Note:** For complete local invocation documentation and examples, see [DEVELOPING.md](DEVELOPING.md#invoking-locally).

Example invocation:
```bash
$ .venv/bin/roboto --log-level=info actions invoke-local \
    --file-query="dataset_id='ds_abc123'" \
    --dry-run
```


_Running without `--dry-run` may have side-effects, depending on how this action is implemented! See relevant section in [DEVELOPING.md](DEVELOPING.md#code-organization-best-practices) for more._

Full usage:
```bash
$ .venv/bin/roboto actions invoke-local --help
```

#### Hosted Invocation

> **Note:** To run this action on Roboto's hosted compute, you must first build and deploy it. See relevant section in [DEVELOPING.md](DEVELOPING.md#build-and-deployment) for more.

Example invocation:
```bash
$ .venv/bin/roboto actions invoke \
    --file-query="dataset_id='ds_abc123'" \
    enrich-lerobot-dataset  # Note required action name parameter for hosted invocation
```


Full usage:
```bash
$ .venv/bin/roboto actions invoke --help
```

## Development

See [DEVELOPING.md](DEVELOPING.md) for detailed information about developing this action, including:
- Project structure and key files
- Local invocation
- Adding dependencies (runtime, system, and development)
- Working with action parameters (including secrets)
- Handling input and output data
- Building and deploying to Roboto

## Deployment

Build and deploy to the Roboto Platform with the following commands:

```bash
$ ./scripts/build.sh
$ ./scripts/deploy.sh
```
