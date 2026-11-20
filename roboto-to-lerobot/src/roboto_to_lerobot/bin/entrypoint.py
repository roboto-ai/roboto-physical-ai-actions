"""Docker entrypoint for the roboto-to-lerobot action.

Requires Roboto-specific environment variables, which are set automatically on
hosted compute and by ``roboto actions invoke-local``.

The ``__main__`` guard is load-bearing: the process pool the action uses for
per-event parallelism runs with ``mp_context="spawn"``, which re-imports this
module in each worker child to set up its ``__main__`` for unpickling. Without
the guard, the child would re-run ``main(context)`` against the already-created
LeRobot output directory.
"""

import os
import ssl


def _verify_tls_with_python_ssl() -> None:
    """Make the roboto SDK check API certificates with Python's ``ssl`` module.

    Unless ``SSL_CERT_FILE`` or ``SSL_CERT_DIR`` is set, the SDK checks them
    with the truststore package, whose context holds a lock during every TLS
    handshake. The SDK opens a new connection for each API call, so
    concurrent calls then make their handshakes one at a time.

    On Linux, truststore trusts OpenSSL's default CA file or directory, the
    same paths set here, so the SDK trusts the same certificates. httpx also
    reads these variables, so its clients trust this bundle instead of
    certifi's. Child processes inherit the variable.
    """
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return
    paths = ssl.get_default_verify_paths()
    if paths.cafile:
        os.environ["SSL_CERT_FILE"] = paths.cafile
    elif paths.capath and os.listdir(paths.capath):
        os.environ["SSL_CERT_DIR"] = paths.capath


# Runs before the imports below so that the variable is set before any of them
# can build a TLS context. The SDK reads it once, on its first HTTPS request.
_verify_tls_with_python_ssl()

import roboto  # noqa: E402

from .. import main  # noqa: E402

if __name__ == "__main__":
    context = roboto.InvocationContext.from_env()
    main(context)
