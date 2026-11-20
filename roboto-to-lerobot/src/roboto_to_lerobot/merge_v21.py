"""Merge one-episode LeRobot v2.1 datasets, the episode cache's entries, into one dataset.

lerobot 0.3.x, which writes v2.1, has no merge API. v2.1 already keeps one
data parquet and one mp4 per camera for each episode, so a merge places
those files under their new episode indices and rewrites only what depends
on an episode's position in the dataset:

* the data parquet's ``index`` (dataset-wide frame number),
  ``episode_index`` and ``task_index`` columns;
* those three columns' stats in ``meta/episodes_stats.jsonl``;
* ``meta/episodes.jsonl``, ``meta/tasks.jsonl`` and the totals in
  ``meta/info.json``.

Every other byte comes from the entries unchanged: videos are moved, never
re-muxed, and each entry's image and feature stats are its own episode's,
as in a single-writer conversion. Data parquet and metadata are written
with the calls lerobot 0.3.x makes when it saves an episode, so a merge of
entries equals a single-writer conversion of the same events in the same
order, file by file.

Task indices are assigned in output order, each episode's new tasks in the
order its frames first use them, and each episode's ``tasks`` list follows
that order too. lerobot 0.3.x numbers an episode's new tasks in the
iteration order of a ``set``, which varies between processes for episodes
with several tasks; for those, the merge is deterministic where a
single-writer conversion is not.
"""

from __future__ import annotations

from pathlib import Path


def merge_v21_entries(parts: list[dict], dataset_root: Path) -> None:
    """Merge the one-episode datasets at ``part["root"]``, in order, into ``dataset_root``.

    ``dataset_root`` must not exist yet. Each entry's video files are moved
    into ``dataset_root``, not copied, so an entry cannot be merged again:
    each root may appear in ``parts`` only once.
    """
    import datasets
    import numpy as np
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import get_feature_stats
    from lerobot.datasets.utils import (
        EPISODES_PATH,
        EPISODES_STATS_PATH,
        TASKS_PATH,
        append_jsonlines,
        embed_images,
        get_hf_features_from_features,
        load_info,
        load_jsonlines,
        serialize_dict,
        write_info,
    )

    from .logger import logger

    logger.info("Merging %d episodes into %s...", len(parts), dataset_root)
    dataset_root.mkdir(parents=True)
    task_index: dict[str, int] = {}
    total_frames = 0
    info: dict = {}
    for new_episode, part in enumerate(parts):
        root = Path(part["root"])
        entry_info = load_info(root)
        if new_episode == 0:
            # The first episode's video info, as a single-writer conversion records it.
            info = entry_info
        hf_features = get_hf_features_from_features(entry_info["features"])
        video_keys = [k for k, ft in entry_info["features"].items() if ft["dtype"] == "video"]
        chunk = new_episode // info["chunks_size"]
        entry_tasks = {t["task_index"]: t["task"] for t in load_jsonlines(root / TASKS_PATH)}

        columns = pq.read_table(root / entry_info["data_path"].format(episode_chunk=0, episode_index=0)).to_pydict()
        length = len(columns["index"])
        frame_tasks = [entry_tasks[i] for i in columns["task_index"]]
        episode_tasks = list(dict.fromkeys(frame_tasks))
        for task in episode_tasks:
            if task not in task_index:
                task_index[task] = len(task_index)
                append_jsonlines({"task_index": task_index[task], "task": task}, dataset_root / TASKS_PATH)
        columns["index"] = list(range(total_frames, total_frames + length))
        columns["episode_index"] = [new_episode] * length
        columns["task_index"] = [task_index[t] for t in frame_tasks]
        data_path = dataset_root / info["data_path"].format(episode_chunk=chunk, episode_index=new_episode)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        embed_images(datasets.Dataset.from_dict(columns, features=hf_features, split="train")).to_parquet(data_path)

        for key in video_keys:
            video = dataset_root / info["video_path"].format(
                episode_chunk=chunk, video_key=key, episode_index=new_episode,
            )
            video.parent.mkdir(parents=True, exist_ok=True)
            (root / entry_info["video_path"].format(episode_chunk=0, video_key=key, episode_index=0)).rename(video)

        (episode,) = load_jsonlines(root / EPISODES_PATH)
        append_jsonlines(
            {**episode, "episode_index": new_episode, "tasks": episode_tasks}, dataset_root / EPISODES_PATH,
        )
        (episode_stats,) = load_jsonlines(root / EPISODES_STATS_PATH)
        stats = episode_stats["stats"]
        for column in ("index", "episode_index", "task_index"):
            values = np.asarray(columns[column], dtype=np.int64)
            stats[column] = serialize_dict(get_feature_stats(values, axis=0, keepdims=True))
        append_jsonlines({"episode_index": new_episode, "stats": stats}, dataset_root / EPISODES_STATS_PATH)
        total_frames += length

    n = len(parts)
    video_count = sum(ft["dtype"] == "video" for ft in info["features"].values())
    info.update(
        total_episodes=n,
        total_frames=total_frames,
        total_tasks=len(task_index),
        total_videos=n * video_count,
        total_chunks=(n - 1) // info["chunks_size"] + 1,
        splits={"train": f"0:{n}"},
    )
    write_info(info, dataset_root)
