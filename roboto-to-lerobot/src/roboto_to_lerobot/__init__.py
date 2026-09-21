"""roboto-to-lerobot package.

The package exposes two independent entry surfaces — ``roboto_to_lerobot.main``
(the converter) and ``roboto_to_lerobot.codegen.cli`` (live-inference codegen) —
and neither is eagerly imported here. ``main`` is exposed lazily (PEP 562):
importing it eagerly would pull the whole converter stack (pandas, roboto, mcap,
cv2) into every ``import roboto_to_lerobot``, including ``import
roboto_to_lerobot.runtime.*`` and ``python -m roboto_to_lerobot.codegen.cli``,
which must run on a stripped-down install without ``lerobot``/``torch`` (the
docker bag-replay smoke relies on this). Accessing ``roboto_to_lerobot.main``
(e.g. ``from roboto_to_lerobot import main`` in bin/entrypoint.py) still works
and triggers the import on demand.
"""

from typing import TYPE_CHECKING

__all__ = ("main",)

if TYPE_CHECKING:
    from .main import main


def __getattr__(name: str):
    if name == "main":
        # Importing the .main submodule binds `roboto_to_lerobot.main` (the
        # module) as a package attribute; rebind it to the function so
        # `from roboto_to_lerobot import main` yields the callable, as before,
        # and so this lazy path runs only once.
        from .main import main as _main

        globals()["main"] = _main
        return _main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
