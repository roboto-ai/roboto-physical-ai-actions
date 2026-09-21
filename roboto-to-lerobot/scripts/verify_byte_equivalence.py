#!/usr/bin/env python3
"""Byte-equivalence verifier for two roboto-to-lerobot conversion runs.

Gates every PR in the conversion-perf stack and every step of the
runtime-parity refactor against a hosted-compute baseline.

Each input directory mirrors one invocation's output tree, downloaded
from the invocation's output dataset::

    <dir>/manifest.json
    <dir>/combined/meta/**/*.json
    <dir>/combined/data/**/*.parquet
    <dir>/combined/videos/**/*.mp4

Rules:

* All files under ``combined/`` are compared except mp4s under
  ``combined/videos/``:

  * ``*.json`` files are compared as parsed JSON. Byte-identical is the
    common case, but ``meta/stats.json`` is dumped with non-deterministic
    dict-key ordering by the lerobot writer; structural equality dodges
    that without weakening the gate.
  * Everything else (parquet, jsonl, txt, …) is SHA256-compared.

* ``combined/videos/**/*.mp4`` are decoded with PyAV (libdav1d under the
  hood, since OpenCV ships without AV1 support); PSNR is sampled across
  the first, middle, and last :data:`_VIDEO_FRAMES_PSNR_SAMPLE` frames
  of each stream. Sampling the spread rather than only the head catches
  regressions that show up after the opening GOP (rate-control drift,
  lookahead changes, motion-estimation differences). Any sampled frame
  below :data:`MP4_PSNR_FLOOR_DB` is a failure, as is a frame-count
  mismatch.
* ``manifest.json`` (one level above ``combined/``) is compared as
  parsed JSON after stripping fields that are nonces of the invocation
  (see :data:`_MANIFEST_VOLATILE_KEYS`).

The PSNR floor is **set empirically** from two same-SHA captures of the
unmodified action (originally via the now-retired ``capture_baseline.sh``
local-capture flow). The calibration is *assumed* to carry over to
hosted-compute captures — SVT-AV1 thread-partition noise is expected to
dominate and to be largely independent of host vs. lambda execution — but
this has not been measured across environments, so treat it as a
conservative assumption, not an established fact. SVT-AV1 multi-threading is
not bit-deterministic across CPU counts, so two runs of the unmodified
action can produce visually identical but byte-different mp4s; the floor
must sit comfortably below that observed delta.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path

import av
import numpy as np

MP4_PSNR_FLOOR_DB: float = 50.0
"""Minimum acceptable PSNR (dB) for any sampled mp4 frame pair.

Calibrated from two same-SHA captures on commit ``66c17e6dcccb`` in an
earlier byte-equivalence calibration run. Both captures produced
byte-identical SVT-AV1 streams on a single host (PSNR = inf on every
sampled frame), so on-host non-determinism is effectively zero. The
50 dB floor is deliberately conservative — cross-host SVT-AV1 runs
(different CPU counts ⇒ different thread partition ⇒ different bits) are
expected to sit comfortably above 50 dB while a genuine perceptual
regression drops well below 30 dB. Adjust only after taking two fresh same-SHA hosted
captures (invoke the action twice at the same commit and download both
output trees) and observing the noise-floor delta has moved.
"""

_VIDEO_FRAMES_PSNR_SAMPLE: int = 10
"""How many frames to sample from each segment (head/middle/tail) of an mp4.

The verifier decodes ``3 × _VIDEO_FRAMES_PSNR_SAMPLE`` frames per video at
most — leading, middle, and trailing — and PSNR-compares each pair. This
catches regressions that surface after the opening GOP (SVT-AV1 rate-
control, lookahead, motion-estimation) which a head-only sample would miss.
"""

_MANIFEST_VOLATILE_KEYS: frozenset[str] = frozenset({
    "generated_at",
    "invocation_id",
})
"""Top-level manifest fields stripped before equality comparison.

These are per-invocation nonces (UTC timestamp, ULID) that always differ
between two runs of the same workload and carry no signal.
"""


@dataclasses.dataclass(slots=True)
class _Failure:
    path: str
    detail: str

    def __str__(self) -> str:
        return f"  - {self.path}: {self.detail}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _relative_glob(root: Path, pattern: str) -> set[Path]:
    return {p.relative_to(root) for p in root.glob(pattern) if p.is_file()}


def _decode_sampled_frames(
    video_path: Path, sample: int
) -> tuple[list[np.ndarray], list[int]]:
    """Decode a spread of frames as BGR24 uint8 arrays.

    Returns ``(frames, indices)`` where ``frames[i]`` is the decoded frame
    at ``indices[i]``. Samples up to ``sample`` frames from each of the
    head, middle, and tail of the stream, deduplicated and sorted —
    so the total returned is ≤ ``3 × sample``. For streams shorter than
    ``3 × sample`` frames, every frame is returned.

    Uses PyAV (libdav1d) so AV1 streams from ``vcodec="libsvtav1"`` decode
    on hosts whose OpenCV build lacks AV1 support. One sequential decode
    pass keeps only the target indices; we don't ``seek`` because PyAV's
    keyframe-only seek over short SVT-AV1 streams is fiddly and the cost
    of a full decode is bounded (a few seconds per ~500-frame video).
    """
    container = av.open(str(video_path))
    try:
        all_frames: list[np.ndarray] = []
        for frame in container.decode(video=0):
            all_frames.append(frame.to_ndarray(format="bgr24"))
    finally:
        container.close()

    total = len(all_frames)
    if total == 0:
        return [], []

    head = range(0, min(sample, total))
    tail = range(max(0, total - sample), total)
    mid_start = max(0, total // 2 - sample // 2)
    mid = range(mid_start, min(mid_start + sample, total))
    indices = sorted(set(head) | set(mid) | set(tail))
    return [all_frames[i] for i in indices], indices


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(255.0 * 255.0 / mse)


def _compare_sha(
    baseline_root: Path,
    candidate_root: Path,
    rel_paths: Iterable[Path],
    label: str,
) -> list[_Failure]:
    failures: list[_Failure] = []
    for rel in sorted(rel_paths):
        bsum = _sha256(baseline_root / rel)
        csum = _sha256(candidate_root / rel)
        if bsum != csum:
            failures.append(_Failure(
                path=f"{label}:{rel}",
                detail=f"sha256 differs (baseline={bsum[:12]}..., candidate={csum[:12]}...)",
            ))
    return failures


def _compare_json(
    baseline_root: Path,
    candidate_root: Path,
    rel_paths: Iterable[Path],
    label: str,
) -> list[_Failure]:
    """Compare JSON files by parsed structural equality, not raw bytes.

    The lerobot writer dumps ``meta/stats.json`` with non-deterministic
    dict-key order (Python dict iteration ≠ stable across runs for stats
    keyed by feature name), so byte SHA can drift run-to-run despite the
    semantic content being identical.
    """
    failures: list[_Failure] = []
    for rel in sorted(rel_paths):
        try:
            a = json.loads((baseline_root / rel).read_text())
            b = json.loads((candidate_root / rel).read_text())
        except json.JSONDecodeError as e:
            failures.append(_Failure(
                path=f"{label}:{rel}",
                detail=f"json parse error: {e}",
            ))
            continue
        if a != b:
            failures.append(_Failure(
                path=f"{label}:{rel}",
                detail="parsed JSON content differs",
            ))
    return failures


def _compare_mp4s(
    baseline_root: Path,
    candidate_root: Path,
    rel_paths: Iterable[Path],
    psnr_floor_db: float,
    sample: int,
) -> list[_Failure]:
    failures: list[_Failure] = []
    for rel in sorted(rel_paths):
        bframes, bidx = _decode_sampled_frames(baseline_root / rel, sample)
        cframes, cidx = _decode_sampled_frames(candidate_root / rel, sample)
        if bidx != cidx:
            failures.append(_Failure(
                path=f"videos:{rel}",
                detail=(
                    f"sampled frame indices differ "
                    f"(baseline={bidx}, candidate={cidx}) — total frame "
                    f"counts likely don't match"
                ),
            ))
            continue
        if not bframes:
            failures.append(_Failure(
                path=f"videos:{rel}",
                detail="no frames decoded from either input",
            ))
            continue
        for offset, (bf, cf) in enumerate(zip(bframes, cframes, strict=True)):
            stream_idx = bidx[offset]
            psnr = _psnr(bf, cf)
            if math.isnan(psnr):
                failures.append(_Failure(
                    path=f"videos:{rel}#frame{stream_idx}",
                    detail=(
                        f"shape mismatch (baseline={bf.shape}, candidate={cf.shape})"
                    ),
                ))
                break
            if psnr < psnr_floor_db:
                failures.append(_Failure(
                    path=f"videos:{rel}#frame{stream_idx}",
                    detail=f"PSNR {psnr:.2f} dB < floor {psnr_floor_db:.2f} dB",
                ))
                break
    return failures


def _compare_manifest(baseline_root: Path, candidate_root: Path) -> list[_Failure]:
    baseline = baseline_root / "manifest.json"
    candidate = candidate_root / "manifest.json"
    # A normal action run always writes manifest.json; absence on either
    # side is a regression signal, and absence on both sides is silent
    # corruption (two partial captures verifying as equivalent).
    if not baseline.exists() and not candidate.exists():
        return [_Failure(
            path="manifest.json",
            detail="missing on both — captures look incomplete",
        )]
    if baseline.exists() ^ candidate.exists():
        side = "baseline" if not baseline.exists() else "candidate"
        return [_Failure(
            path="manifest.json",
            detail=f"missing on {side}",
        )]
    bm = json.loads(baseline.read_text())
    cm = json.loads(candidate.read_text())
    for key in _MANIFEST_VOLATILE_KEYS:
        bm.pop(key, None)
        cm.pop(key, None)
    if bm == cm:
        return []
    diffs: list[str] = []
    all_keys = sorted(set(bm) | set(cm))
    for key in all_keys:
        if bm.get(key) != cm.get(key):
            diffs.append(key)
    return [_Failure(
        path="manifest.json",
        detail=f"fields differ: {', '.join(diffs)}",
    )]


def _collect_combined_files(combined: Path) -> tuple[set[Path], set[Path]]:
    """Return (mp4_videos, non_video_files) as paths relative to combined/."""
    mp4_videos: set[Path] = set()
    other: set[Path] = set()
    for p in combined.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(combined)
        if rel.parts and rel.parts[0] == "videos" and p.suffix == ".mp4":
            mp4_videos.add(rel)
        else:
            other.add(rel)
    return mp4_videos, other


def verify(
    baseline: Path,
    candidate: Path,
    *,
    psnr_floor_db: float = MP4_PSNR_FLOOR_DB,
    psnr_sample: int = _VIDEO_FRAMES_PSNR_SAMPLE,
) -> list[_Failure]:
    """Compare two invocation output trees. Returns the list of failures.

    Rule: every file under ``combined/`` is compared by SHA256 *except*
    mp4s under ``combined/videos/``, which are PSNR-compared. ``manifest.json``
    at the invocation root is JSON-compared after stripping volatile keys.
    """
    if not (baseline / "combined").is_dir():
        raise SystemExit(f"baseline missing combined/: {baseline}")
    if not (candidate / "combined").is_dir():
        raise SystemExit(f"candidate missing combined/: {candidate}")

    base_videos, base_other = _collect_combined_files(baseline / "combined")
    cand_videos, cand_other = _collect_combined_files(candidate / "combined")

    # Guard against two empty captures verifying equivalent. A run that
    # crashed after ``mkdir combined/`` but before writing any output would
    # silently pass against another empty capture (or against a partial one
    # if the symmetric difference happens to be one-sided in only one
    # direction). Both sides must hold at least one parquet AND one mp4 to
    # count as a real capture.
    for side, parquets, videos in (
        ("baseline", {p for p in base_other if p.suffix == ".parquet"}, base_videos),
        ("candidate", {p for p in cand_other if p.suffix == ".parquet"}, cand_videos),
    ):
        if not parquets:
            raise SystemExit(
                f"{side} has no parquet files under combined/ — looks empty or partial"
            )
        if not videos:
            raise SystemExit(
                f"{side} has no mp4 files under combined/videos/ — looks empty or partial"
            )

    failures: list[_Failure] = []

    for rel in sorted(base_other - cand_other):
        failures.append(_Failure(
            path=f"combined/{rel}",
            detail="present in baseline, missing in candidate",
        ))
    for rel in sorted(cand_other - base_other):
        failures.append(_Failure(
            path=f"combined/{rel}",
            detail="present in candidate, missing in baseline",
        ))
    for rel in sorted(base_videos - cand_videos):
        failures.append(_Failure(
            path=f"combined/{rel}",
            detail="present in baseline, missing in candidate",
        ))
    for rel in sorted(cand_videos - base_videos):
        failures.append(_Failure(
            path=f"combined/{rel}",
            detail="present in candidate, missing in baseline",
        ))

    shared = base_other & cand_other
    json_rels = {rel for rel in shared if rel.suffix == ".json"}
    other_rels = shared - json_rels

    failures.extend(_compare_json(
        baseline / "combined", candidate / "combined", json_rels, "combined",
    ))
    failures.extend(_compare_sha(
        baseline / "combined", candidate / "combined", other_rels, "combined",
    ))
    failures.extend(_compare_mp4s(
        baseline / "combined", candidate / "combined",
        base_videos & cand_videos,
        psnr_floor_db, psnr_sample,
    ))

    failures.extend(_compare_manifest(baseline, candidate))
    return failures


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, type=Path, help="baseline invocation subfolder (contains combined/ and manifest.json)")
    parser.add_argument("--candidate", required=True, type=Path, help="candidate invocation subfolder (same shape)")
    parser.add_argument(
        "--psnr-floor-db", type=float, default=MP4_PSNR_FLOOR_DB,
        help=f"override mp4 PSNR floor in dB (default: {MP4_PSNR_FLOOR_DB})",
    )
    parser.add_argument(
        "--psnr-sample", type=int, default=_VIDEO_FRAMES_PSNR_SAMPLE,
        help=f"frames to sample per mp4 (default: {_VIDEO_FRAMES_PSNR_SAMPLE})",
    )
    parser.add_argument(
        "--report-psnr", action="store_true",
        help="report per-video min PSNR even on pass (useful for noise-floor capture)",
    )
    args = parser.parse_args(argv)

    if args.report_psnr:
        _report_psnr_summary(
            args.baseline, args.candidate,
            sample=args.psnr_sample,
        )

    failures = verify(
        args.baseline.resolve(),
        args.candidate.resolve(),
        psnr_floor_db=args.psnr_floor_db,
        psnr_sample=args.psnr_sample,
    )
    if failures:
        print(f"FAIL: {len(failures)} difference(s) between {args.baseline} and {args.candidate}")
        for fail in failures:
            print(fail)
        return 1
    print(
        f"PASS: {args.baseline} ≡ {args.candidate} "
        f"(parquet/JSON bit-identical, mp4 PSNR ≥ {args.psnr_floor_db:.2f} dB)"
    )
    return 0


def _report_psnr_summary(baseline: Path, candidate: Path, *, sample: int) -> None:
    """Print per-video min PSNR — used to calibrate the floor constant."""
    base_videos = baseline / "combined/videos"
    cand_videos = candidate / "combined/videos"
    if not base_videos.is_dir() or not cand_videos.is_dir():
        print("(no combined/videos/ on one side; skipping PSNR summary)")
        return
    shared = sorted(_relative_glob(base_videos, "**/*.mp4") & _relative_glob(cand_videos, "**/*.mp4"))
    if not shared:
        print("(no shared mp4s; skipping PSNR summary)")
        return
    print(
        f"PSNR summary across {len(shared)} mp4(s), "
        f"≤{3 * sample} frames sampled (head/middle/tail × {sample}):"
    )
    overall_min = float("inf")
    for rel in shared:
        bframes, _bidx = _decode_sampled_frames(base_videos / rel, sample)
        cframes, _cidx = _decode_sampled_frames(cand_videos / rel, sample)
        n = min(len(bframes), len(cframes))
        if n == 0:
            print(f"  {rel}: (no decoded frames)")
            continue
        per_frame = [_psnr(bframes[i], cframes[i]) for i in range(n)]
        finite = [v for v in per_frame if math.isfinite(v)]
        worst = min(per_frame) if per_frame else float("nan")
        if math.isfinite(worst):
            overall_min = min(overall_min, worst)
        avg = (sum(finite) / len(finite)) if finite else float("inf")
        print(
            f"  {rel}: n={n} min={worst:.2f} dB avg={avg:.2f} dB"
            + (" (some frames identical)" if len(finite) < len(per_frame) else "")
        )
    if math.isfinite(overall_min):
        print(f"Overall worst-frame PSNR across all videos: {overall_min:.2f} dB")
    else:
        print("Overall: all sampled frames byte-identical (PSNR = inf)")


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
