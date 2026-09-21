"""Vendored third-party conversion code.

``convert_dataset_v30_to_v21`` is vendored verbatim from any4lerobot:

    https://github.com/Tavish9/any4lerobot
    path:    ds_version_convert/v30_to_v21/convert_dataset_v30_to_v21.py
    commit:  2ef2370d66
    sha256:  674c0ada14da9d6e862868e5cf1b8e1fa88585a6cc45ead6361bc33766edc733
    license: MIT (Copyright (c) 2025 Qizhi Chen) -- see LICENSE-any4lerobot

The file is kept byte-for-byte identical to upstream so it can be re-synced with
a plain diff. The action does NOT call its top-level ``convert_dataset()`` (which
may ``snapshot_download`` from the Hub and swaps its result in place over the
source tree); instead :mod:`lerobot_v3_to_v2_1.downgrade` calls the building-block
functions with explicit, non-destructive source/destination roots.

WARNING: upstream ``main`` is regressed -- its default-branch HEAD reverted this
file to the v2.1 -> v3.0 *upgrade* logic (any4lerobot PR #110). Do not bump the
pin without diffing against commit ``2ef2370d66`` and re-running the tests.
"""
