from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .alignment import merge_onto_timeline
from .contract_utils import AlignSpec, fps_to_time_step_ns, image_message_type
from .converters import DECODERS
from .logger import logger
from .runtime.image import resize_image
from .transforms import apply_transforms
from .video import VideoFrameReader, VideoFrameRef

if TYPE_CHECKING:
    from .contract_utils import Contract, DataCollection, ObservationSpec


@dataclass(frozen=True, slots=True)
class _DeferredFrame:
    """Undecoded video frame that crosses a process boundary cheaply.

    Encoded bytes pickle far more compactly than decoded HWC arrays, so when
    frame generation runs in a worker process the video entries travel as
    these sentinels and the consumer decodes (and optionally resizes) on the
    receiving side just before ``writer.add_frame``. A compressed-video
    entry's payload is ``{"frame": VideoFrameRef}``, whose session carries
    the packets to decode it from.
    """

    video_key: str
    decoder_type: str
    payload: dict[str, Any]
    resize: tuple[int, int] | None


def episode_video_reader(frames: Iterable[dict[str, Any] | None]) -> VideoFrameReader:
    """A ``VideoFrameReader`` for every compressed-video frame the deferred ``frames`` refer to.

    Build it from the whole episode before materializing any of its frames:
    the reader keeps a decoded frame only while a later frame still refers
    to it, so it must know every reference up front.
    """
    return VideoFrameReader(
        value.payload["frame"]
        for frame in frames
        if frame is not None
        for value in frame.values()
        if isinstance(value, _DeferredFrame) and isinstance(value.payload.get("frame"), VideoFrameRef)
    )


def materialize_deferred(
    frame: dict[str, Any],
    video_specs_by_key: dict[str, "ObservationSpec"],
    video_reader: VideoFrameReader | None = None,
) -> dict[str, Any]:
    """Decode (and optionally resize) every ``_DeferredFrame`` in ``frame``.

    Mutates ``frame`` in place. Entries that are not deferred pass through
    untouched, so it is safe to call on frames produced without
    ``defer_image_decode``.

    A compressed-video entry is read from ``video_reader``, built by
    :func:`episode_video_reader` over the episode's frames; every other entry
    goes through the decoder registered for its message type.

    Raises:
        ValueError: If no decoder is registered for an entry's message type,
            or an entry refers to a compressed-video frame and no
            ``video_reader`` is given.
    """
    for key, val in list(frame.items()):
        if not isinstance(val, _DeferredFrame):
            continue
        ref = val.payload.get("frame")
        if isinstance(ref, VideoFrameRef):
            if video_reader is None:
                raise ValueError(
                    f"Video key '{val.video_key}' refers to a compressed-video frame, which needs "
                    "the episode's video reader to decode."
                )
            image = video_reader.frame(ref)
        else:
            decoder = DECODERS.get(val.decoder_type)
            if decoder is None:
                raise ValueError(
                    f"No video decoder registered for message type "
                    f"'{val.decoder_type}' (video key '{val.video_key}')."
                )
            image = decoder(pd.Series(val.payload), video_specs_by_key[val.video_key])
        if val.resize is not None:
            expected_h, expected_w = val.resize
            image = resize_image(image, expected_h, expected_w)
        frame[key] = image
    return frame


def _suffix_columns(df: pd.DataFrame, key: str, keep_cols: list[str]) -> pd.DataFrame:
    """
    Rename non-timestamp columns by appending ``|key`` so that multiple
    DataFrames can be merged without column-name collisions.

    Only columns listed in *keep_cols* (plus ``"timestamp"``) are retained.
    """
    out = df.copy()
    rename_map = {}
    cols_to_keep = ["timestamp"]
    for col in keep_cols:
        new_name = f"{col}|{key}"
        rename_map[col] = new_name
        cols_to_keep.append(new_name)
    out = out.rename(columns=rename_map)
    return out[cols_to_keep]


def generate_frames(
    contract: "Contract",
    data_collection: "DataCollection",
    reference_timestamps: pd.Series,
    task: str = "default",
    defer_image_decode: bool = False,
    dropped_frames: list[int] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Generate observation-action frame dicts aligned to *reference_timestamps*.

    Each data stream is merged onto the reference timeline using the
    alignment method specified in its contract spec (``hold``, ``nearest``,
    ``linear``, or ``none``).  The yielded dict uses base keys
    (e.g. ``"observation.state"``, ``"action"``) with concatenated values
    from all topics sharing that base key.

    Args:
        contract: Contract specification describing observations, actions, videos.
        data_collection: DataCollection with already-loaded DataFrames whose
            columns correspond to the selector paths from the contract.
        reference_timestamps: DataFrame / Series with a ``"timestamp"`` column
            that defines the output cadence.
        dropped_frames: If given, receives the positions in
            ``reference_timestamps`` of the frames left out because a stream's
            value for them is missing (NaN). It is filled before the first
            frame is yielded.

    Yields:
        dict mapping base keys to numpy arrays (observations / actions)
        or decoded images (videos). Multiple topics with the same base key
        are concatenated into a single array.
    """
    logger.info("Generating frames from data collection")

    resolved = data_collection.resolved_features
    merged_df = reference_timestamps.copy()

    # Per-stream alignment lookup. Obs/actions use unique_key (multiple topics
    # can share a base key); videos use their key directly (1:1 with topic).
    align_specs: dict[str, AlignSpec] = {}
    for obs in contract.observations:
        align_specs[obs.unique_key] = obs.align
    for video in contract.videos:
        align_specs[video.key] = video.align
    for act in contract.actions:
        align_specs[act.unique_key] = act.align

    def _get_align(stream_key: str) -> AlignSpec:
        return align_specs.get(stream_key, AlignSpec())

    # ── helpers to classify keys ──────────────────────────────────────────────

    def _is_array_key(unique_key: str) -> bool:
        """True when the DataFrame for *unique_key* stores its values in a
        single ``values`` column (the shape produced by
        :class:`DataCollection`'s decoder loop for vector streams)."""
        return unique_key in resolved

    def _array_col_name(df: pd.DataFrame) -> str:
        """Return the name of the single non-timestamp column."""
        cols = [c for c in df.columns if c != "timestamp"]
        assert len(cols) == 1, f"Expected 1 non-timestamp column, got {cols}"
        return cols[0]

    # ── Group observations by base key ────────────────────────────────────────
    # Track which unique_keys belong to each base key for concatenation
    obs_by_base_key: dict[str, list[str]] = {}  # base_key -> [unique_keys]
    obs_scalar: dict[str, list[str]] = {}  # unique_key -> prefixed selector names

    for obs in contract.observations:
        if obs.image:
            continue
        unique_key = obs.unique_key
        base_key = obs.key

        if base_key not in obs_by_base_key:
            obs_by_base_key[base_key] = []
        obs_by_base_key[base_key].append(unique_key)

        if not _is_array_key(unique_key):
            if obs.selector and "names" in obs.selector:
                obs_scalar[unique_key] = obs.get_lerobot_selector_names()

    # ── Group actions by base key ─────────────────────────────────────────────
    action_by_base_key: dict[str, list[str]] = {}  # base_key -> [unique_keys]
    action_scalar: dict[str, list[str]] = {}  # unique_key -> prefixed selector names

    for action in contract.actions:
        unique_key = action.unique_key
        base_key = action.key

        if base_key not in action_by_base_key:
            action_by_base_key[base_key] = []
        action_by_base_key[base_key].append(unique_key)

        if not _is_array_key(unique_key):
            if action.selector and "names" in action.selector:
                action_scalar[unique_key] = action.get_lerobot_selector_names()

    # ── validate timestamps on every loaded stream ────────────────────────
    # Degenerate timestamps (non-monotonic or all-equal) would silently
    # produce garbage downstream: pre-transforms divide by median_dt to derive
    # raw_fs, and merge_onto_timeline's as-of joins resolve every reference
    # sample within tolerance to the same duplicated source row. Fail loudly
    # here so the operator knows exactly which stream's source topic has
    # malformed header stamps.
    for store_name, store in (
        ("observation", data_collection.observations),
        ("action", data_collection.actions),
    ):
        for unique_key, df in store.items():
            if df is None or df.empty:
                continue
            ts = df["timestamp"].values
            if len(ts) < 2:
                continue
            median_dt = float(np.median(np.diff(ts)))
            if median_dt <= 0:
                raise ValueError(
                    f"{store_name} stream {unique_key!r} has degenerate "
                    f"timestamps (len={len(ts)}, first={ts[0]}, last={ts[-1]}, "
                    f"median_dt={median_dt}). Non-monotonic or constant "
                    f"timestamps would produce undefined results in alignment "
                    f"and transforms. The source topic likely has malformed "
                    f"header stamps."
                )

    # ── apply pre-alignment transforms ────────────────────────────────────
    for spec in contract.observations + contract.actions:
        pre_specs = [t for t in (spec.transforms or []) if t.stage == "pre"]
        if not pre_specs:
            continue
        store = (
            data_collection.observations
            if spec in contract.observations
            else data_collection.actions
        )
        df = store.get(spec.unique_key)
        if df is None or df.empty:
            continue
        if not _is_array_key(spec.unique_key):
            continue
        col = _array_col_name(df)
        stacked = np.stack(df[col].values)  # (T, N)
        ts = df["timestamp"].values
        # Use the raw signal's actual sampling frequency, not the output fps.
        # Timestamps are in nanoseconds. Validation above guarantees median_dt > 0.
        raw_fs = float(1e9 / np.median(np.diff(ts))) if len(ts) > 1 else contract.fps
        logger.info("Applying %d pre-alignment transform(s) to %s", len(pre_specs), spec.unique_key)
        result = apply_transforms(stacked, pre_specs, raw_fs, timestamps=ts)
        stacked, ts = result  # pre-alignment always receives timestamps
        # Rebuild the DataFrame with (possibly resampled) data & timestamps
        store[spec.unique_key] = pd.DataFrame({
            "timestamp": ts,
            col: list(stacked),
        })

    # ── action lead: offset action timestamps so merge picks future values ──
    action_lead_ns = (
        contract.action_lead_steps * fps_to_time_step_ns(contract.fps)
        if contract.action_lead_steps else 0
    )
    if action_lead_ns:
        logger.info("action_lead_steps=%d: offsetting action timestamps by -%d ns",
                     contract.action_lead_steps, action_lead_ns)
        data_collection.actions = {
            k: v.assign(timestamp=v["timestamp"] - action_lead_ns)
            for k, v in data_collection.actions.items()
        }

    # ── merge array-column observations / actions ────────────────────────────
    array_keys: dict[str, str] = {}  # unique_key -> merged column name
    for unique_key, df in list(data_collection.observations.items()) + list(data_collection.actions.items()):
        if not _is_array_key(unique_key):
            continue
        src_col = _array_col_name(df)
        merged_col = f"{src_col}|{unique_key}"
        array_keys[unique_key] = merged_col
        logger.info("Merging array column %s for %s onto base timeline (method=%s)",
                   src_col, unique_key, _get_align(unique_key).method)
        to_merge = df[["timestamp", src_col]].copy().rename(columns={src_col: merged_col})
        merged_df = merge_onto_timeline(
            merged_df, to_merge, _get_align(unique_key), [merged_col],
        )

    # ── merge scalar observations ────────────────────────────────────────────
    for obs_key, obs_df in data_collection.observations.items():
        selector_names = obs_scalar.get(obs_key)
        if not selector_names:
            continue
        logger.info("Merging observation %s onto base timeline (method=%s)",
                   obs_key, _get_align(obs_key).method)
        suffixed = _suffix_columns(obs_df, obs_key, selector_names)
        value_cols = [f"{n}|{obs_key}" for n in selector_names]
        merged_df = merge_onto_timeline(
            merged_df, suffixed, _get_align(obs_key), value_cols,
        )

    # ── merge scalar actions ─────────────────────────────────────────────────
    for action_key, action_df in data_collection.actions.items():
        selector_names = action_scalar.get(action_key)
        if not selector_names:
            continue
        logger.info("Merging action %s onto base timeline (method=%s)",
                   action_key, _get_align(action_key).method)
        suffixed = _suffix_columns(action_df, action_key, selector_names)
        value_cols = [f"{n}|{action_key}" for n in selector_names]
        merged_df = merge_onto_timeline(
            merged_df, suffixed, _get_align(action_key), value_cols,
        )

    # Each video's DataFrame carries decoder-specific columns (CompressedImage
    # → format/data; Image → height/width/encoding/data). Suffix them with
    # |{video.key} to avoid collisions during merge; the original names are
    # rebuilt at decode time.
    video_payload_cols: dict[str, list[str]] = {}  # video.key -> original column names
    for video_key, video_df in data_collection.videos.items():
        logger.info("Merging video %s onto base timeline (method=%s)",
                   video_key, _get_align(video_key).method)
        payload_cols = [c for c in video_df.columns if c != "timestamp"]
        video_payload_cols[video_key] = payload_cols

        rename_map = {c: f"{c}|{video_key}" for c in payload_cols}
        suffixed_cols = list(rename_map.values())
        video_renamed = video_df[["timestamp", *payload_cols]].rename(columns=rename_map)
        merged_df = merge_onto_timeline(
            merged_df, video_renamed, _get_align(video_key), suffixed_cols,
        )
    # ── apply post-alignment transforms ────────────────────────────────────
    for spec in contract.observations + contract.actions:
        post_specs = [t for t in (spec.transforms or []) if t.stage == "post"]
        if not post_specs:
            continue
        col = array_keys.get(spec.unique_key)
        if col is None:
            continue
        # Frames the stream has no sample for hold NaN and are dropped
        # below. Transform each unbroken run of the other frames on its own,
        # so a transform that reads neighbouring frames (a difference, a
        # filter) never treats the two sides of a gap as neighbours.
        for run in _runs_of_matched_frames(merged_df[col].notna().to_numpy()):
            rows = merged_df.index[run]
            stacked = np.stack(merged_df.loc[rows, col].values)  # (T, N)
            stacked = apply_transforms(stacked, post_specs, contract.fps)
            merged_df.loc[rows, col] = pd.Series(list(stacked), index=rows)

    rows_before = len(merged_df)
    if dropped_frames is not None:
        # Each merge keeps one row per reference timestamp, in order, so a
        # row's position is its frame's position on the reference timeline.
        # (An exact-timestamp merge, align method ``none``, keeps that only
        # while its stream has no repeated timestamps.)
        dropped_frames.extend(np.flatnonzero(merged_df.isna().any(axis=1).to_numpy()).tolist())
    merged_df = merged_df.dropna()
    rows_after = len(merged_df)
    rows_dropped = rows_before - rows_after

    if rows_dropped > 0:
        logger.info(
            "Dropped %d rows with NaN values, retained %d rows",
            rows_dropped,
            rows_after,
        )

    # Without ``defer_image_decode`` each frame is decoded as it is yielded,
    # so the compressed-video reader is built now, from every reference the
    # remaining rows hold.
    video_reader = None
    if not defer_image_decode:
        video_reader = VideoFrameReader(
            value
            for video_key, payload_cols in video_payload_cols.items()
            for col in payload_cols
            for value in merged_df[f"{col}|{video_key}"]
            if isinstance(value, VideoFrameRef)
        )
    video_specs_by_key = {video.key: video for video in contract.videos}

    # ── yield one frame dict per row ─────────────────────────────────────────
    frame_count = 0
    for _, row in merged_df.iterrows():
        frame_dict: dict[str, Any] = {}

        # Concatenate observations by base key
        for base_key, unique_keys in obs_by_base_key.items():
            all_values: list[float] = []
            for unique_key in unique_keys:
                if unique_key in array_keys:
                    # Array column - read numpy array directly
                    merged_col = array_keys[unique_key]
                    arr = np.asarray(row[merged_col], dtype=np.float32)
                    all_values.extend(arr.tolist())
                elif unique_key in obs_scalar:
                    # Scalar columns - read individual values
                    selector_names = obs_scalar[unique_key]
                    for name in selector_names:
                        all_values.append(float(row[f"{name}|{unique_key}"]))

            if all_values:
                frame_dict[base_key] = np.array(all_values, dtype=np.float32)

        # Concatenate actions by base key
        for base_key, unique_keys in action_by_base_key.items():
            all_values: list[float] = []
            for unique_key in unique_keys:
                if unique_key in array_keys:
                    # Array column - read numpy array directly
                    merged_col = array_keys[unique_key]
                    arr = np.asarray(row[merged_col], dtype=np.float32)
                    all_values.extend(arr.tolist())
                elif unique_key in action_scalar:
                    # Scalar columns - read individual values
                    selector_names = action_scalar[unique_key]
                    for name in selector_names:
                        all_values.append(float(row[f"{name}|{unique_key}"]))

            if all_values:
                frame_dict[base_key] = np.array(all_values, dtype=np.float32)

        # Each video becomes one ``_DeferredFrame`` under its contract key,
        # so the caller can decode it after the frame crosses a process
        # boundary (``defer_image_decode``); otherwise it is decoded here,
        # through the same ``materialize_deferred``.
        for video in contract.videos:
            decoder_type = image_message_type(video)
            payload_cols = video_payload_cols[video.key]
            payload = {orig: row[f"{orig}|{video.key}"] for orig in payload_cols}
            # Payloads must be picklable to survive IPC. Arrow-backed
            # pandas reads can hand back ``memoryview`` (not picklable),
            # so coerce buffer-like cells to ``bytes`` up front rather
            # than failing opaquely at ``pool.submit`` time.
            payload = {
                k: bytes(v) if isinstance(v, (memoryview, bytearray)) else v
                for k, v in payload.items()
            }
            resize = (
                tuple(video.image["resize"])
                if video.image and "resize" in video.image
                else None
            )
            frame_dict[video.key] = _DeferredFrame(
                video_key=video.key,
                decoder_type=decoder_type,
                payload=payload,
                resize=resize,
            )
        if not defer_image_decode:
            materialize_deferred(frame_dict, video_specs_by_key, video_reader)

        frame_dict["task"] = task

        frame_count += 1
        yield frame_dict

    logger.info("Generated %d frames", frame_count)


def _runs_of_matched_frames(matched: np.ndarray) -> list[slice]:
    """Return the position ranges of the unbroken runs of ``True`` in ``matched``, in order."""
    edges = np.diff(np.concatenate([[False], matched, [False]]).astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [slice(start, stop) for start, stop in zip(starts, stops, strict=True)]
