# Quickstart — `roboto-to-lerobot` runtime parity (v0.1.0)

This guide walks through generating a deployable ROS 2 inference node
from a Roboto `contract.yaml`, then validating it against a real MCAP
recording inside a Docker container. The end-to-end goal:

> You can train a policy on a LeRobot dataset converted by
> `roboto-to-lerobot`, then run `gen-node` against the same contract and
> ship a `ros2 run`-able inference node that reads the same topics and
> publishes the same action message shape.

Current scope: **best-effort parity**. The converter and runtime share the
same alignment / encode / decode primitives, but the two paths are not
yet pinned together by an adversarial CI gate (see "Parity disclaimer"
below).

## What you need

* A Roboto org with a converted LeRobot dataset and the corresponding
  `contract.yaml` + `manifest.json` (the converter writes both).
* A trained policy exposing `load_policy(path) -> Callable[[obs], action]`
  in a Python module.
* ROS 2 Humble + `rosbag2_storage_mcap` for the live runtime.
* Python ≥ 3.10 for the runtime kernel; ≥ 3.12 to install the package
  from the published wheel (use `pip install --ignore-requires-python`
  to install on the Humble interpreter — see the docker harness for a
  working recipe).

**Required for `gen-node` (step 3):** install the `[codegen]` extra. It
provides Jinja2, which the generator needs to render, plus `black`,
which formats the output:

    pip install 'roboto_to_lerobot[codegen]'

A base `pip install roboto_to_lerobot` (no extra) **cannot** run
`gen-node` — Jinja2 lives only in the `[codegen]`/`[dev]` extras, not in
the core dependencies. `black`, by contrast, is optional: without it
`gen-node` still produces valid (just unformatted) Python — formatting
is a readability nicety, not a correctness gate.

## The four-step flow

```bash
# 1. Convert (Roboto action; produces dataset + contract.yaml + manifest.json)
roboto actions invoke roboto-to-lerobot \
    --dataset $SOURCE_DATASET --collection $COLLECTION \
    --parameter contract=contract.yaml

# 2. Train (your policy on the converted LeRobot dataset)
python train.py --dataset-root ./lerobot_out

# 3. Generate the inference node from the same contract
#    (needs the [codegen] extra: pip install 'roboto_to_lerobot[codegen]')
roboto-to-lerobot gen-node ./contract.yaml \
    --policy-module my_pkg.policies.act \
    --manifest ./manifest.json \
    --out src/my_pkg/inference_node.py

# 4. Run on the robot (or inside the docker bag-replay below)
ros2 run my_pkg inference_node \
    --ros-args -p policy_path:=./checkpoints/last.ckpt
```

The generated file is a vanilla `rclpy.Node` subclass. Subscribe-only
topics map to the contract's `observations:`; the publish-only topics
map to `actions:`. The node owns one `LiveAdapter` (the same code path
the converter uses, sans pandas) that buffers messages, samples them at
the configured `fps`, and emits the policy output back over the action
publishers.

## Regenerating: the safe path

`gen-node` does **not** ship `--upgrade` (three-way merge for re-runs).
Regen workflow:

```bash
# Generate to a new path; review the diff against the previous output.
roboto-to-lerobot gen-node ./contract.yaml \
    --policy-module my_pkg.policies.act \
    --manifest ./manifest.json \
    --out src/my_pkg/inference_node.v2.py
diff src/my_pkg/inference_node.py src/my_pkg/inference_node.v2.py
# When you're happy:
mv src/my_pkg/inference_node.v2.py src/my_pkg/inference_node.py
```

`--force` (overwrite-in-place) exists for quick first-iteration regen,
not as the workflow to settle on; the refuse-overwrite default is what
keeps an accidental `gen-node` from wiping a hand-edited file.

## Parity disclaimer

Version 0.1.0 ships with **best-effort parity**: the converter and the
generated runtime use the same `runtime/` primitives (`StreamBuffer`,
encoders, decoders), but the converter still drives them through
`merge_asof`-based offline alignment while the runtime uses the live
`StreamBuffer.sample()` path. Boundary cases can drift. There is no
adversarial CI gate pinning the two paths together yet, so validate
your contract end-to-end with the docker bag-replay below before
shipping.

Also unsupported in this release:

* Alignment `linear` and `none` on observations/videos (only `hold` and
  `nearest` are supported there). Actions are exempt — the runtime never
  aligns actions, so any `align.method` on an action stream is accepted.
* Transforms `butterworth_lowpass` and `resample_uniform` (non-causal;
  cannot run faithfully in a live node).
* `safety_behavior` other than `publish_nothing` — when an observation
  is stale, the generated node simply does not publish, and the
  downstream controller times out and applies its own stop.
* Action encoders other than `JointState`, `Float64MultiArray`, `Float64`.

`gen-node` refuses contracts that ask for any of the above, with a
clear pointer to the offending stream.

## Validating a custom contract — docker bag-replay

Before deploying to real hardware, replay an MCAP through the generated
node inside a ROS 2 Humble container. The harness ships with no dataset
of its own: point it at a dataset that holds an MCAP carrying the
topics the contract declares by exporting `ROBOTO_ROS_SMOKE_DATASET`.
That one is required; the fetcher exits with a message naming it. If
your user belongs to more than one Roboto org, also export
`ROBOTO_ORG_ID` so the SDK knows which to read from.

```bash
# One-time: install docker + log in to the Roboto SDK profile that has
# read access to your dataset.

# Run the smoke (≈3-5 min on a warm docker cache, longer for the first
# build that has to pull ros:humble-ros-base).
ROBOTO_RUN_ROS_SMOKE=1 \
    ROBOTO_ROS_SMOKE_DATASET=ds_yourdataset \
    .venv/bin/pytest test/test_ros_smoke.py -v

# Or invoke the shell harness directly:
ROBOTO_ROS_SMOKE_DATASET=ds_yourdataset \
    bash test/ros_smoke/run.sh
```

What it does, end to end:

1. `test/ros_smoke/fetch_fixture.py` downloads the first `.mcap` in the
   dataset into `test/fixtures/` (gitignored; one-time per machine).
2. `docker build` produces a ros:humble-ros-base image with the package
   + its import-time deps (no torch, no lerobot — those would balloon
   the image and the smoke does not need them).
3. `docker run` mounts the MCAP + the harness into the container, then:
   * `gen-node` renders `test/ros_smoke/ros_pkg/inference_node/node.py`
     from the bundled `contract.yaml`,
   * `colcon build` produces an installable `inference_node` ROS 2
     package,
   * `ros2 bag play` and `ros2 run inference_node node` run in
     parallel,
   * `ros2 topic echo` captures one message from `/teleop/action`
     and a continuous stream from `/robot/joint_states`,
   * `test/ros_smoke/verify_action.py` then asserts the captured action
     matches the input-dependent stub invariant: its 8 values equal
     `obs["observation.state"][:8] + 100.0` for some observation in
     the recorded stream, within a 0.2 rad tolerance that absorbs DDS
     subscriber-discovery jitter. The whole sequence is bounded by
     `ROS_SMOKE_TIMEOUT` (default 30 s).

The invariant is what makes the gate informative: a frozen-default or
zero-output policy, or a broken observation path that never feeds the
policy, both fail it. A bare "did any message land?" check would not.

If you point the harness at a contract with different action / topic
names, edit `test/ros_smoke/contract.yaml` and
`test/ros_smoke/ros_pkg/inference_node/policy.py` (the stub returns
`obs["observation.state"][:_ACTION_WIDTH] + _STUB_OFFSET`; keep
`_ACTION_WIDTH` aligned with the contract's action selector width).

The smoke is **opt-in by design**: every CI run would otherwise pay the
docker-build + image-pull cost, and CI does not have a ROS 2 install
anyway. Run it locally before tagging a release.

## What the docker bag-replay does **not** prove

A real robot's driver differs from `ros2 bag play` in two narrow but
non-zero ways:

* **QoS handshake.** A driver may negotiate `BEST_EFFORT/KEEP_LAST/1`
  on a sensor topic where the bag publishes `RELIABLE`. Codegen
  defaults sensor types (`Image`, `CompressedImage`, `PointCloud`,
  `PointCloud2`) to `BEST_EFFORT/KEEP_LAST/1`; everything else to
  `RELIABLE/KEEP_LAST/10`. Override per-spec with an explicit `qos:`
  block in the contract if a driver mismatches.
* **Clock source.** `ros2 bag play` uses the bag's recorded
  timestamps; a real driver uses the system clock (or a hardware
  clock). The runtime uses `node.get_clock().now()` for the tick
  reference, which is the same surface either way — but a driver
  whose timestamps lag the clock by tens of milliseconds will see a
  different `tolerance_ms` behavior than the bag.

Both gaps are narrow and visible at the first hardware test, which is
where you should expect to find them. The docker smoke catches the
data-path + ROS-environment bugs that would otherwise surface only at
that test.

