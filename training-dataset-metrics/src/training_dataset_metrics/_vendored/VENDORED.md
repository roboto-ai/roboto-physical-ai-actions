# Vendored third-party code

This directory contains source copied verbatim from sibling packages in the
`roboto-physical-ai-actions` repo, so that this action's Docker image can be
built from the action directory alone (which is what
`roboto actions invoke-local` requires).

## `roboto_to_lerobot/`

Subset of `roboto-to-lerobot/src/roboto_to_lerobot/` needed by Mode A
(`sources/pre_conversion.py`), which drives the same `generate_frames`
pipeline the converter uses so the auditor sees the converter's
post-alignment frame stream.

Mirrored files (kept in sync by `scripts/sync_vendored.sh`):

- `alignment.py`
- `contract_utils.py`
- `converters.py`
- `decoders.py`
- `extract.py`
- `lerobot.py`
- `logger.py`
- `transforms.py`
- `video.py`
- `runtime/__init__.py`
- `runtime/converters.py`
- `runtime/decoders.py`
- `runtime/image.py`
- `runtime/LICENSE-rosetta`

`converters.py` and `decoders.py` upstream are now thin re-export shims
over `runtime/`, which is why the decoder registry's real home
(`runtime/decoders.py`, plus the `runtime/converters.py` registry and the
`runtime/image.py` helpers it uses) is mirrored too. `video.py` comes with
them: `runtime/decoders.py` imports its `COMPRESSED_VIDEO_SCHEMAS` to
register the `foxglove_msgs/CompressedVideo` decoders.

Modules deliberately **not** vendored:

- `main.py`, `bin/`, `writers/` — converter orchestration and the
  version-dispatched LeRobot writer (this action never writes a LeRobot
  dataset, so the writer dispatch is dead weight here).
- `runtime/contract_io.py`, `runtime/live_adapter.py`,
  `runtime/replay_adapter.py`, `runtime/stream_buffer.py`,
  `runtime/encoders.py`, `runtime/causal_lowpass.py`: the live-inference
  half of the runtime package. `runtime/__init__.py` resolves those names
  lazily, so they are absent rather than stale: reaching for one here
  raises instead of quietly running against an old copy.

### Refreshing from upstream

After editing `roboto-to-lerobot/src/roboto_to_lerobot/`, run:

```
scripts/sync_vendored.sh
```

from the action root. The script `cp`s only the files in its
`VENDORED_FILES` array, one by one — there is no `--delete` step, so a
file added or removed upstream will NOT appear/disappear here
automatically. Update the array in `sync_vendored.sh` if the mirrored
set needs to change.

Forgetting to run it is caught: `scripts/verify.sh` has a **Vendored
sync** step that compares every file in `VENDORED_FILES` byte-for-byte
against upstream and fails, naming the offending files, if any differs,
is missing here, or no longer exists upstream. It reads the file list by
sourcing `sync_vendored.sh` (with `VENDORED_LIST_ONLY=1`, which returns
before the copy loop), so there is only ever one list. When the sibling
`roboto-to-lerobot/` directory is absent — a partial checkout, or this
action packaged on its own — the step skips with a message rather than
failing, since a missing sibling is not drift.

### License

The vendored copy carries the same licensing as upstream. Most files are
covered by the MPL-2.0 `LICENSE` at the repo root. `contract_utils.py`,
`runtime/converters.py` and `runtime/decoders.py` are derived from
[Rosetta](https://github.com/iblnkn/rosetta) (© 2025 Isaac Blankenau) and
remain under Apache-2.0; see their in-file headers and
`roboto_to_lerobot/runtime/LICENSE-rosetta`.
