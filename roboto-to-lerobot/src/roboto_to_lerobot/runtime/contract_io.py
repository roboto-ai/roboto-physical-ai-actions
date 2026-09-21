"""Contract loading + manifest verification surface for the live runtime.

Generated nodes and the live adapter import three names
from here: ``load_contract``, the ``Contract`` wrapper (whose ``sha256()``
method is the boot-time drift check), and ``verify_manifest``. The
implementation delegates to ``contract_utils.load_contract`` so the
converter and the live runtime share one parsing path — keep the surface
here stable across releases, even when the underlying contract schema
widens.

``Contract.sha256()`` is the file-bytes SHA the converter records in
``manifest["contract"]["sha256"]``. Generated nodes bake the same value
into the source as ``CONTRACT_SHA256`` so a mismatch at boot — contract
YAML rewritten, node not regenerated — fails loudly instead of silently
misaligning observations.

The wrapper is a thin facade over the canonical ``contract_utils.Contract``
dataclass rather than a subclass because that dataclass is ``frozen=True,
slots=True`` and not designed for inheritance. Every attribute and
method on the underlying dataclass is delegated through ``__getattr__``,
so anything the converter pipeline can do with a ``Contract`` works on
this wrapper too — except identity-typed isinstance checks against
``contract_utils.Contract``, which the live-runtime call sites do not
perform.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..contract_utils import Contract as _Contract
from ..contract_utils import load_contract as _load_contract

__all__ = ("Contract", "load_contract", "verify_manifest")


PathLike = Path | str


class Contract:
    """Live-runtime contract: ``contract_utils.Contract`` plus a SHA helper.

    Constructed by :func:`load_contract`; not intended to be instantiated
    directly by application code. Carries the originating file path and the
    file-bytes SHA256 captured at load time, matching what the converter
    records in ``manifest["contract"]["sha256"]``.
    """

    __slots__ = ("_inner", "_sha256", "_source_path")

    def __init__(
        self, inner: _Contract, source_path: Path, sha256: str | None = None
    ) -> None:
        self._inner = inner
        self._source_path = source_path
        # ``load_contract`` snapshots the digest from the bytes it reads and
        # passes it in. Direct construction (e.g. building the facade around an
        # in-memory contract, with no YAML on disk) may omit it; it is then
        # computed lazily and cached on first ``sha256()`` call.
        self._sha256 = sha256

    @property
    def source_path(self) -> Path:
        return self._source_path

    def sha256(self) -> str:
        """File-bytes SHA256 of the source contract YAML.

        Snapshotted once, then cached. Via :func:`load_contract` the digest is
        captured from the bytes read at load time, so it cannot drift if the
        file changes afterwards. A directly-constructed facade that omitted the
        digest instead reads ``source_path`` lazily on the first call — which
        captures the file's state at that point (possibly later than
        construction), and raises ``FileNotFoundError`` if ``source_path`` is a
        placeholder with no file behind it. Matches the converter, which hashes
        the contract bytes once and writes the frozen value into
        ``manifest["contract"]["sha256"]``.
        """
        if self._sha256 is None:
            self._sha256 = hashlib.sha256(self._source_path.read_bytes()).hexdigest()
        return self._sha256

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only consulted when normal attribute lookup misses.
        # Guard the real slots so an instance built without __init__ (copy /
        # pickle bypass __init__ on a __slots__ class) raises AttributeError
        # instead of recursing forever through ``self._inner``.
        if name in ("_inner", "_source_path", "_sha256"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return f"Contract(source={self._source_path}, inner={self._inner!r})"


def load_contract(path: PathLike) -> Contract:
    """Load a contract YAML and return a live-runtime :class:`Contract`.

    Thin wrapper around ``contract_utils.load_contract`` — the parsing code
    stays in one place; this entry point adds source-path tracking and
    snapshots the file-bytes SHA256 once for :meth:`Contract.sha256`.
    """
    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    inner = _load_contract(source)
    return Contract(inner=inner, source_path=source, sha256=digest)


def verify_manifest(manifest_path: PathLike, contract: Contract) -> None:
    """Refuse to boot if the manifest was produced by a different contract.

    Reads ``manifest["contract"]["sha256"]`` from ``manifest_path`` and
    compares against ``contract.sha256()``. Raises ``ValueError`` on any
    mismatch — including a manifest that does not record a SHA at all, since
    that means the manifest predates the runtime-parity contract baking work
    and cannot give a safety guarantee.

    The check is per-boot in generated nodes. It is not a substitute for
    the codegen-time ``CONTRACT_SHA256`` constant: that check guards
    against the contract drifting between codegen and runtime; this check
    guards against the manifest having been produced by a third revision.
    """
    raw = Path(manifest_path).read_text(encoding="utf-8")
    data = json.loads(raw)
    contract_section = data.get("contract")
    recorded = contract_section.get("sha256") if isinstance(contract_section, dict) else None
    if not recorded:
        raise ValueError(
            f"Manifest {manifest_path} does not record a contract sha256; "
            "regenerate the manifest with the current converter."
        )
    actual = contract.sha256()
    if recorded != actual:
        raise ValueError(
            f"Manifest {manifest_path} was produced by contract sha {recorded}, "
            f"but the loaded contract at {contract.source_path} has sha {actual}. "
            "Regenerate either the manifest or the contract."
        )
