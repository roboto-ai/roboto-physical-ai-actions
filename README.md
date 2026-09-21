# Roboto Physical AI Actions

This repository contains example Roboto Actions demonstrating the interoperability between [LeRobot datasets](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) and the [Roboto](https://www.roboto.ai/) platform.

LeRobot is an open-source framework for robotics machine learning that defines its own dataset format. Roboto is a platform for ingesting, searching and processing robotics log data. The actions here bridge the two: they turn data already ingested into Roboto into LeRobot datasets, and then merge, audit or enrich those datasets. They are reference implementations, written to be read and adapted at least as much as to be run as-is.

## What is a Roboto Action?

A Roboto Action is a containerized program that the Roboto platform runs against your data. The platform stages the inputs (files selected by a dataset id, a file query or glob patterns), runs the container with the parameters you supply, and uploads whatever the action writes to its output directory back into Roboto. Actions can also read from and write to the platform directly through the Roboto SDK, for example to walk a collection of events or to tag results. See the [Roboto Actions guide](https://docs.roboto.ai/user-guides/process-data-actions.html) for the full model.

Every action directory in this repository has the same shape:

| Path | What it is |
|---|---|
| `action.json` | Action name, description, parameters and compute requirements |
| `Dockerfile` | The image the platform runs |
| `src/` | The action's Python package, with a container entrypoint |
| `test/` | Pytest suite |
| `scripts/` | `setup.sh`, `verify.sh`, `build.sh`, `deploy.sh` |

The usual development loop, run from inside an action directory:

```bash
$ ./scripts/setup.sh                    # virtualenv + dependencies
$ ./scripts/verify.sh                   # ruff + pytest
$ .venv/bin/roboto actions invoke-local \
      --dataset=ds_xxxxxxxxxxxx --dry-run
$ ./scripts/deploy.sh   # build, push, register the action
$ .venv/bin/roboto actions invoke --dataset=ds_xxxxxxxxxxxx <action-name>
```

Local invocation runs the same Docker image the hosted platform runs, so behaviour matches. Only the *compute* is local, despite the name: inputs are still pulled from Roboto and the action still talks to the platform over the network, so you need connectivity and credentials either way. `--dry-run` is a convention, not a sandbox — it sets a flag the action itself honours to gate uploads and metadata writes. The exact flags and parameters differ per action; each action's own README gives a working invocation. You will need [programmatic access to Roboto](https://docs.roboto.ai/getting-started/programmatic-access.html) configured first (`roboto users whoami` should succeed).

## The actions

| Directory | What it does |
|---|---|
| [`roboto-to-lerobot`](./roboto-to-lerobot/README.md) | Converts data ingested into Roboto into a LeRobot dataset, one episode per event in a Roboto event collection. Which topics become which state, action and video features is declared in a contract YAML. Deployed as two variants: `roboto-to-lerobot-v2_1` (LeRobot 2.1) and `roboto-to-lerobot-v3_0` (LeRobot 3.0). |
| [`lerobot-merge`](./lerobot-merge/README.md) | Merges the shard outputs of several parallel `roboto-to-lerobot` invocations into one LeRobot dataset, concatenating video by stream copy rather than re-encoding, and writing a single self-contained provenance manifest. |
| [`training-dataset-metrics`](./training-dataset-metrics/README.md) | Audits the statistical quality of robot training data and emits an HTML plus JSON report with plots, flagging episodes likely to hurt training. Runs either on raw Roboto topics before conversion or on an already-converted LeRobot dataset. |
| [`enrich-lerobot-dataset`](./enrich-lerobot-dataset/README.md) | A small worked example of adding a derived feature to an existing LeRobot dataset using LeRobot's `add_features`. The shipped example computes the element-wise difference between `action` and `observation.state`. |
| [`lerobot-v3-to-v2_1`](./lerobot-v3-to-v2_1/README.md) | Converts an uploaded LeRobot v3.0 dataset back to the v2.1 layout: consolidated parquet shards are split into per-episode files, concatenated per-camera video into frame-exact per-episode MP4s, and metadata is rewritten into the v2.1 schema. |

## How they fit together

`roboto-to-lerobot` is the centre of the repository. Almost everything else sits upstream or downstream of it.

**Before conversion.** `training-dataset-metrics` can audit the raw Roboto topics using the same contract YAML the converter would use, so you can spot stuck sensors, misaligned timing or dead channels before spending compute on a conversion. Video streams are skipped in this mode.

**Conversion.** `roboto-to-lerobot` reads a collection of Roboto events and writes one LeRobot episode per event, driven by the contract. The v3.0 variant can also write in parallel shards within a single invocation.

**Fan-out and merge.** For collections too large for one invocation, the pattern is to partition the collection, run several conversions in parallel, and then use `lerobot-merge` to stitch the shard outputs into one dataset whose manifest matches what a single-shot conversion would have produced.

**After conversion.** `training-dataset-metrics` runs again in its post-conversion mode against the produced LeRobot dataset, this time including anything that needed the encoded video. `enrich-lerobot-dataset` is the template for bolting derived features onto a dataset you already have.

**Using the result.** The actions write LeRobot datasets into Roboto; to pull
one back out and open it as a `LeRobotDataset`:

```python
import pathlib

import roboto
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# `roboto-to-lerobot` and `lerobot-merge` write to "<invocation_id>/combined/",
# so narrow the download to that prefix rather than fetching the whole dataset.
prefix = "iv_xxxxxxxxxxxx/combined"
output_dir = pathlib.Path("./.cache")

dataset = roboto.Dataset.from_id("ds_xxxxxxxxxxxx")
dataset.download_files(output_dir, include_patterns=[f"{prefix}/*"])

lerobot_dataset = LeRobotDataset(repo_id=prefix, root=output_dir / prefix)
print(lerobot_dataset.num_episodes)
```

`include_patterns` is gitignore-style, so the trailing `/*` pulls the whole
`meta/`, `data/` and `videos/` tree beneath the prefix. Drop it, and pass the
download directory directly as `root`, if the LeRobot data sits at the dataset
root instead.

**Off to the side.** `lerobot-v3-to-v2_1` is not part of that pipeline. It exists for datasets that arrive as LeRobot v3.0 but must be consumed by tooling that only reads v2.1.

## Where to start

Start with [`roboto-to-lerobot`](./roboto-to-lerobot/README.md). It is the action the others are built around, and it is the one whose behaviour you will need to understand to adapt any of this to your own data.

The substantial reference document in this repository is [`roboto-to-lerobot/CONTRACT.md`](./roboto-to-lerobot/CONTRACT.md). It documents the contract YAML in full: top-level fields, observation, action, video and task specs, selectors, alignment methods, transforms, role-based binding, the supported message types, and the validation errors you are likely to hit. Almost every question about how conversion behaves is answered there.

If you would rather see the shape of a run before reading the schema, [`roboto-to-lerobot/example_notebook/`](./roboto-to-lerobot/example_notebook/) holds worked contract YAMLs and end-to-end pipeline notebooks, including the parallel fan-out and merge pipeline and a run with the metrics audit wired in.

## Where to go next

- Each action's own `README.md` for its parameters, inputs, outputs and invocation examples.
- Each action's `DEVELOPING.md` for project structure, dependency management, local invocation and deployment details.
- [`roboto-to-lerobot/CONTRACT.md`](./roboto-to-lerobot/CONTRACT.md), the contract schema reference.
- [`roboto-to-lerobot/docs/quickstart.md`](./roboto-to-lerobot/docs/quickstart.md), an experimental path that generates a ROS 2 inference node from the same contract used for conversion, so a trained policy reads the same topics it was trained on. Treated as best-effort parity, not a guaranteed one.

## Resources

- [Roboto Platform Documentation](https://docs.roboto.ai/)
- [Roboto Actions Guide](https://docs.roboto.ai/user-guides/process-data-actions.html)
- [LeRobot Dataset Documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

## License

Mozilla Public License 2.0. See [LICENSE](./LICENSE).
