"""Download one MCAP file from a roboto dataset into ``test/fixtures/``.

The docker bag-replay needs a real MCAP recording with the topics the
docker contract declares (``/robot/joint_states`` + ``/teleop/action``),
so point this script at a dataset of your own that holds one.

This script is idempotent: re-running it after the MCAP is already
in ``test/fixtures/`` is a no-op. ``test/fixtures/`` is gitignored;
each developer fetches once and then ``./run.sh`` can use the
cached file across docker iterations.

Run as ``python3 -m test.ros_smoke.fetch_fixture`` from the package
root, or just ``python3 fetch_fixture.py`` from inside this dir.
The dataset id (``--dataset`` or ``ROBOTO_ROS_SMOKE_DATASET``) is
required and has no default; the SDK picks up your configured profile,
which ``ROBOTO_PROFILE`` overrides. If your user belongs to more than
one org, also export ``ROBOTO_ORG_ID`` to disambiguate.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _fixtures_dir() -> Path:
    """Resolve the gitignored ``test/fixtures/`` directory on disk.

    Resolves relative to this file so callers can run the script from
    any working directory (the docker harness execs it through a bind
    mount whose cwd is opaque).
    """
    return Path(__file__).resolve().parent.parent / "fixtures"


def fetch_one_mcap(
    dataset_id: str,
    target_dir: Path | None = None,
) -> Path:
    """Download the first MCAP file from ``dataset_id`` into ``target_dir``.

    Picks the first .mcap result deterministically (smallest by
    relative_path lexsort) so re-runs always select the same file —
    keeps developer machines fetching the same bytes the docker
    smoke ran against last time.
    """
    target_dir = target_dir or _fixtures_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    # Late-import so a stock checkout without the dev extras can still
    # import this module (pytest collection time) without erroring.
    import roboto

    dataset = roboto.Dataset.from_id(dataset_id)
    mcaps = sorted(
        dataset.list_files(include_patterns=["**/*.mcap"]),
        key=lambda f: f.relative_path,
    )
    if not mcaps:
        raise RuntimeError(
            f"Dataset {dataset_id} has no .mcap files. Set "
            "ROBOTO_ROS_SMOKE_DATASET to a dataset that contains one."
        )

    src = mcaps[0]
    dst = target_dir / Path(src.relative_path).name
    if dst.exists() and dst.stat().st_size > 0:
        print(f"fetch_fixture: {dst} already present ({dst.stat().st_size} bytes); skipping.")
        return dst

    # Download to a sibling temp path, then atomically rename into place.
    # An interrupted download (network drop, or this machine's OOM kill)
    # otherwise leaves a truncated-but-nonzero file at dst that both
    # idempotency guards (this function's st_size>0 check and run.sh's
    # glob) would accept as complete — feeding a corrupt MCAP into the
    # smoke, which then fails opaquely deep in ros2 instead of re-fetching.
    # The finally also covers KeyboardInterrupt and a (rare) failure in
    # os.replace itself — anything that leaves tmp on disk gets reaped.
    print(f"fetch_fixture: downloading {src.relative_path} → {dst}")
    tmp = dst.with_name(f".{dst.name}.partial")
    try:
        src.download(tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"fetch_fixture: wrote {dst} ({dst.stat().st_size} bytes).")
    return dst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download one MCAP file from a roboto dataset into test/fixtures/.",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("ROBOTO_ROS_SMOKE_DATASET"),
        help=(
            "Roboto dataset id to fetch from — required, no default "
            "(may also be given as $ROBOTO_ROS_SMOKE_DATASET). The dataset "
            "must hold an MCAP carrying the topics contract.yaml declares."
        ),
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Destination directory (default: test/fixtures/).",
    )
    args = parser.parse_args(argv)

    # The dataset id is caller-supplied: the harness ships with no dataset
    # of its own, so fail loudly here rather than letting the SDK fault deep
    # in Dataset.from_id with a less actionable message. The org is left to
    # the SDK, which resolves it from the profile for single-org users.
    if not args.dataset:
        print(
            "fetch_fixture: no dataset id. Pass --dataset ds_... or export "
            "ROBOTO_ROS_SMOKE_DATASET.",
            file=sys.stderr,
        )
        return 2

    try:
        path = fetch_one_mcap(args.dataset, args.target)
    except Exception as exc:
        print(f"fetch_fixture: FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"fetch_fixture: OK ({path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
