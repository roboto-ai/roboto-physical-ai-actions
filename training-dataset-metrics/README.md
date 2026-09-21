# training-dataset-metrics

A Roboto action that audits robot training data and emits a quality report
(HTML + JSON + plots) so you can see — before training a policy — which
episodes are likely to hurt rather than help.

The action runs in one of two modes, picked by whether you pass a
`collection_id` parameter:

| Mode                       | Trigger                | Input                                                                                          | Use when                                                                                              |
|----------------------------|------------------------|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| **A — pre-conversion**     | `collection_id` set    | A Roboto dataset with a `contract.yaml` plus a Roboto event Collection (same one you'd hand to [`roboto-to-lerobot`](../roboto-to-lerobot/README.md)).  | You want to audit raw scalar streams *before* running [`roboto-to-lerobot`](../roboto-to-lerobot/README.md), using the same contract.    |
| **B — post-conversion**    | `collection_id` unset  | A Roboto dataset that contains an already-written LeRobot dataset (v2.1 or v3.0). | You want to audit the LeRobot dataset that [`roboto-to-lerobot`](../roboto-to-lerobot/README.md) produced. |

The audit signal is identical across modes: in Mode A, the action drives
the same conversion pipeline that [`roboto-to-lerobot`](../roboto-to-lerobot/README.md) uses, so the metric
inputs match the converter's frame-by-frame outputs. Video streams are
skipped in Mode A; run visual-fidelity audits in Mode B against the
already-converted dataset.

## Table of Contents

- [What the audit produces](#what-the-audit-produces)
- [Audit tags](#audit-tags-mode-a-only)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Parameters](#parameters)
  - [Running](#running)
    - [Local Invocation](#local-invocation)
    - [Hosted Invocation](#hosted-invocation)
- [Development](#development)
- [Deployment](#deployment)

## What the audit produces

Each invocation produces an audit folder named
`audit_reports/<invocation_id>/`. On hosted compute the Roboto runtime
uploads the action's `output_dir` to the **invocation dataset** when
the action finishes, so the folder lands under that dataset's Files
view. Locally, the folder stays on disk under whichever `output_dir`
you passed to `actions invoke-local`. Contents:

- `audit_report.html` — single-file HTML report with embedded plots, the
  full metric table, and per-episode flag highlights.
- `report.json` — machine-readable counterpart of the HTML report,
  including a hash of the contract used.
- `plots/*.png` — one PNG per metric (referenced by the HTML).
- `contract.yaml` (Mode A only) — sidecar copy of the contract used.

The metric set covers: autocorrelation, state↔action alignment, speed
distribution, cross-episode variance, action velocity, filtering flags,
effective sample size, effective dimensionality, and state coverage. The
HTML report explains what each metric checks and how to read its plot.

## Audit tags (Mode A only)

Pass `--parameter write_audit_tags=true` to have the action stamp two
levels of Roboto state after the audit:

- **Per-event verdicts.** Each event in the collection receives one
  `audit:<flag_name>` tag for every metric it failed, or a single
  `audit:clean` tag if it passed every metric. Designed so that
  queries like `tags CONTAINS 'audit:stuck_sensor'` find every
  event that tripped a given flag, across collections and datasets.
- **Collection-level back-pointer.** The source collection receives
  one `audit:<invocation_id>` tag pointing at the audit run that
  produced the report. From that invocation ID, you can navigate to
  the invocation dataset and open
  `audit_reports/<invocation_id>/audit_report.html`. The collection
  always points at its most recent audit — prior pointers are
  replaced.

In both cases, stale tags within the `audit:` namespace from prior runs
are replaced as part of the same update; tags outside that namespace
are never touched.

**Flag-name stability.** Flag names that surface as `audit:<name>` tags
on events are part of this action's public surface, so queries like
`tags CONTAINS 'audit:stuck_sensor'` keep working across action
releases. Existing flag names will not be renamed; new flags may be
added.

**Idempotency.** Re-running the audit replaces the prior tags in
place. A partial failure during the tag write always leaves the new
state in place (any leftover stale tag is cleaned up by the next run).
A failed collection-tag write is logged and never aborts the run — the
on-disk report remains the source of truth.

## Quick Start

### Prerequisites

- **Docker** (Engine 19.03+): Local invocation always runs in Docker for production parity
- **Python 3**: A supported version (see [.python-version](.python-version))

```bash
$ docker --version
$ python3 --version
```

### Installation

Set up a virtual environment and install dependencies:

```bash
$ ./scripts/setup.sh
```

You must be set up to [access Roboto programmatically](https://docs.roboto.ai/getting-started/programmatic-access.html).
Verify with:
```bash
$ .venv/bin/roboto users whoami
```

### Parameters

| Name                | Required | Default          | Mode    | Description                                                                                                                                                                                                                          |
|---------------------|----------|------------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `collection_id`     | no       | —                | A       | ID of a Roboto Collection whose `resource_type` is `event`. Every event in the collection becomes one episode for audit; topics are prepared per source dataset and each event is routed to its own, so collections that span multiple datasets are supported. Events with no `dataset_ids` are skipped. Setting this switches the action into Mode A (pre-conversion). Leave unset to run Mode B (post-conversion). Mirrors [`roboto-to-lerobot`](../roboto-to-lerobot/README.md)'s input contract. |
| `contract`          | no       | `contract.yaml`  | A       | Path within the dataset to the [`roboto-to-lerobot`](../roboto-to-lerobot/README.md) contract YAML describing observation and action topic roles.                                                                                                                        |
| `write_audit_tags`  | no       | `false`          | A       | When true, stamps audit results back into Roboto: per-event `audit:<flag>` tags on every event (or `audit:clean` if all metrics passed), plus an `audit:<invocation_id>` tag on the source collection so it back-points at this audit run's report. Stale `audit:*` tags from prior runs are replaced. Read-only when false. |

### Running

#### Local Invocation

> **Note:** For complete local invocation documentation and examples, see [DEVELOPING.md](DEVELOPING.md#invoking-locally).

Mode A (pre-conversion) — audit raw events that you'd hand to
[`roboto-to-lerobot`](../roboto-to-lerobot/README.md):
```bash
$ .venv/bin/roboto --log-level=info actions invoke-local \
    --dataset=<INVOCATION_DATASET_ID> \
    --parameter collection_id="<COLLECTION_ID>" \
    --parameter contract="contract.yaml" \
    --dry-run
```

Mode B (post-conversion) — audit an already-converted LeRobot dataset
on the invocation dataset:
```bash
$ .venv/bin/roboto --log-level=info actions invoke-local \
    --dataset=<INVOCATION_DATASET_ID> \
    --dry-run
```

To also write `audit:*` tags back to the events and the collection, add
`--parameter write_audit_tags=true` and drop `--dry-run`.

_Running without `--dry-run` may have side-effects, depending on the
parameters (notably `write_audit_tags=true` mutates Roboto tags). See
the relevant section in
[DEVELOPING.md](DEVELOPING.md#code-organization-best-practices) for
more._

Full usage:
```bash
$ .venv/bin/roboto actions invoke-local --help
```

#### Hosted Invocation

> **Note:** To run this action on Roboto's hosted compute, you must first build and deploy it. See [DEVELOPING.md](DEVELOPING.md#build-and-deployment).

Mode A:
```bash
$ .venv/bin/roboto actions invoke \
    --dataset=<INVOCATION_DATASET_ID> \
    --parameter collection_id="<COLLECTION_ID>" \
    --parameter write_audit_tags=true \
    training-dataset-metrics
```

Mode B (omit `collection_id`):
```bash
$ .venv/bin/roboto actions invoke \
    --dataset=<INVOCATION_DATASET_ID> \
    training-dataset-metrics
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

The action carries a vendored subset of [`roboto-to-lerobot`](../roboto-to-lerobot/README.md) under
`src/training_dataset_metrics/_vendored/roboto_to_lerobot/` so the Docker
image can be built from this directory alone (which is what
`roboto actions invoke-local` requires). Refresh it after upstream
changes with:

```bash
$ ./scripts/sync_vendored.sh
```

## Deployment

Build and deploy to the Roboto Platform with:

```bash
$ ./scripts/build.sh
$ ./scripts/deploy.sh
```
