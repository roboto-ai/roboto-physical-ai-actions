"""Live-runtime primitives shared between the converter and generated nodes.

The generated-node template imports the public surface from this
package: ``Contract`` / ``load_contract`` / ``verify_manifest`` for
contract handling, ``LiveAdapter`` for the message→tensor seam, and
``StreamBuffer`` for the per-stream sample policy. ``ReplayAdapter`` is
a smoke-test driver, not a public API; it's re-exported for the
codegen tests that drive a recorded bag through the same kernel.

Re-exports are resolved lazily via :pep:`562` ``__getattr__`` so this
package can be imported during ``contract_utils`` initialization
(``contract_utils.py:14`` triggers ``decoders.py`` → ``runtime.decoders``
→ this package's init). Eagerly resolving ``contract_io`` /
``live_adapter`` / ``replay_adapter`` here would create a cycle through
``..contract_utils``; deferring the imports avoids it without forcing
``contract_utils`` to restructure its decoder-registration side effect.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = (
    "Contract",
    "LiveAdapter",
    "ReplayAdapter",
    "StreamBuffer",
    "auto_bound_tolerance_ns",
    "load_contract",
    "verify_manifest",
)


_LAZY_EXPORTS = {
    "Contract": ".contract_io",
    "load_contract": ".contract_io",
    "verify_manifest": ".contract_io",
    "LiveAdapter": ".live_adapter",
    "ReplayAdapter": ".replay_adapter",
    "StreamBuffer": ".stream_buffer",
    "auto_bound_tolerance_ns": ".stream_buffer",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_LAZY_EXPORTS[name], package=__name__)
    value = getattr(module, name)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
