"""Source loaders are imported lazily so the sibling `roboto-to-lerobot` package
(required by Mode A for `Contract`/`DataCollection`) is only resolved at
invocation-time, not import-time. This lets unit tests that exercise only the
metrics modules run in environments without that sibling installed."""

from __future__ import annotations

from typing import Any


def load_post_conversion_episodes(*args: Any, **kwargs: Any):
    from .post_conversion import load_post_conversion_episodes as _impl
    return _impl(*args, **kwargs)


def load_pre_conversion_episodes(*args: Any, **kwargs: Any):
    from .pre_conversion import load_pre_conversion_episodes as _impl
    return _impl(*args, **kwargs)


__all__ = ["load_post_conversion_episodes", "load_pre_conversion_episodes"]
