# Development Guide

How this action is put together and how to develop, test, and deploy it. For the
general Roboto Action model (parameters, input/output, local invocation), see the
[Roboto docs](https://docs.roboto.ai) and the sibling `enrich-lerobot-dataset`
action.

## Project Structure

```
src/lerobot_v3_to_v2_1/
├── main.py            # orchestrator: find root → detect version → branch → downgrade
├── lerobot_dataset.py # find_root, detect_codebase_version, image_in_parquet_features (pure JSON)
├── downgrade.py       # non-destructive orchestration of the conversion steps
├── video.py           # frame-exact per-episode video splitting
├── roboto_io.py       # create output dataset, upload v2.1 tree, stamp provenance
├── logger.py
├── bin/entrypoint.py  # Docker entrypoint: builds InvocationContext, calls main
└── convert/
    ├── convert_dataset_v30_to_v21.py  # VENDORED, byte-exact (see below)
    ├── LICENSE-any4lerobot            # MIT license for the vendored file
    └── __init__.py                    # provenance + pin caveat
test/
├── test_main.py          # fast unit tests for the classification helpers (no lerobot)
└── test_downgrade_e2e.py # end-to-end: synthesise a real v3 dataset, downgrade, assert v2.1
```

The package `__init__` is intentionally import-light so the pure-JSON helpers in
`lerobot_dataset.py` can be imported and unit-tested without pulling in
lerobot/torch. `main` is imported directly from `main.py` by the entrypoint.

## Conversion Architecture

`main` locates the dataset (`meta/info.json`), reads `codebase_version`, and:

- `v2.1` → copies the dataset through to the output unchanged (idempotent).
- not `v3.0` → raises with a clear message.
- camera frames stored as `dtype: image` → raises (image-in-parquet unsupported).
- `v3.0` → calls `downgrade.downgrade_v30_to_v21(source_root, dest_root)`.

`downgrade.py` deliberately does **not** call the vendored top-level
`convert_dataset()` — that function may `snapshot_download` from the Hub and swaps
its result *in place over the source tree*. Instead it drives the vendored
building blocks (`convert_info`, `convert_tasks`, `convert_data`,
`convert_episodes_metadata`, `copy_ancillary_directories`, `load_episode_records`)
with explicit `source_root` (read-only) → `dest_root` (output) paths, and
substitutes `video.convert_videos_frame_exact` for the vendored `convert_videos`.

### Output (publishing)

By default the action **creates a new Roboto dataset** (name derived from the
source via `roboto_io.derive_output_name`), uploads the v2.1 tree to its root, and
stamps provenance (`roboto_io.build_provenance`) linking back to the source. The
`output_dataset_id` parameter overrides this to target an existing dataset.

The v2.1 tree is staged under `context.output_dir`, uploaded via the SDK, then
deleted — so the platform's automatic upload of `output_dir` is a no-op and the
result never lands back in the source dataset. The whole publish step is gated on
`context.is_dry_run`, so a local dry-run converts and leaves the tree for
inspection without touching Roboto.

### Frame-exact video (`video.py`)

The vendored splitter uses a plain `ffmpeg -ss/-t -c copy`, which is lossless but
not frame-exact: a timestamp-duration cut over-includes the trailing boundary
frame, and `-ss` on a non-keyframe boundary snaps to the previous keyframe. The
splitter here, per episode:

- if the boundary timestamp matches a source keyframe, **stream-copies exactly
  `length` frames** (lossless, no re-encode);
- otherwise **re-encodes the exact frame range** with ffmpeg's
  `trim=start_frame:end_frame` filter (quality-preserving fallback).

A decoded-frame-count assertion after each split fails loudly if either path is
wrong, so an off-by-one can never reach the output. The downgrade returns a report
with `{"copied": N, "reencoded": N}`.

## Vendored conversion code

`convert/convert_dataset_v30_to_v21.py` is vendored **byte-for-byte** from
[any4lerobot](https://github.com/Tavish9/any4lerobot) at commit `2ef2370d66`
(MIT, © 2025 Qizhi Chen; see `convert/LICENSE-any4lerobot`).

- Keep it byte-identical to upstream so it can be re-synced with a plain diff. It
  is excluded from ruff via `[tool.ruff] extend-exclude` in `pyproject.toml`.
- **The pin matters.** Upstream `main` is regressed: a later PR (#110) reverted
  this file to the v2.1 → v3.0 *upgrade* logic. Do not bump the pin without
  diffing against `2ef2370d66` and re-running the tests. The sha256 and these
  notes are recorded in `convert/__init__.py`.
- The imports it needs (`lerobot.datasets.io_utils`, `lerobot.datasets.utils`)
  resolve against **lerobot 0.5.1**, which is the pin in `pyproject.toml`
  (`lerobot >= 0.5.0, < 0.6`). A v2.1-writing lerobot (0.3.x) is **not** needed —
  the downgrade writes the v2.1 layout by hand.

## Testing

```bash
$ ./scripts/verify.sh         # ruff check . + pytest
```

The unit tests (`test_main.py`) are dependency-light. The end-to-end tests
(`test_downgrade_e2e.py`) synthesise a real v3.0 dataset with lerobot 0.5.x and
require `lerobot`, `av`, and an `ffmpeg` binary; they `importorskip` cleanly where
those are absent.

**ffmpeg:** the action shells out to `ffmpeg` for video splitting, and the host
may not have one. The Docker image installs `ffmpeg`. For local tests, the e2e
fixture falls back to the binary bundled with `imageio-ffmpeg` (a lerobot
dependency) by symlinking it onto `PATH`, so no system ffmpeg is required.

You can run the suite against any venv that has the deps (e.g. a sibling action's
`.venv` with lerobot 0.5.1), or set up this action's own venv:

```bash
$ PYTHONPATH=src <some-venv>/bin/pytest test/ -v
# or, standalone:
$ ./scripts/setup.sh && ./scripts/verify.sh
```

## Build & Deployment

```bash
$ ./scripts/build.sh                  # build the Docker image (CPU-only torch, ffmpeg)
$ ./scripts/deploy.sh # push image + create/update the action
```

`compute_requirements` in `action.json` (4 vCPU / 8 GB / 200 GB) is sized for
ffmpeg-bound video splitting plus holding the input v3 tree and the output v2.1
tree on disk simultaneously (~2× dataset size).
