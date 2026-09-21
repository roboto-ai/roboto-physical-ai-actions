# Back-compat shim. Every symbol in runtime.decoders is private (the decoders
# are all `_dec_*`); the module's only public effect is registering them into
# the DECODERS registry on import, so this is a side-effect import, not `*`.
from .runtime import decoders  # noqa: F401
