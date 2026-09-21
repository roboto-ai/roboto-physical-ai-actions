"""Docker entrypoint for the roboto-to-lerobot action.

Requires Roboto-specific environment variables, which are set automatically on
hosted compute and by ``roboto actions invoke-local``.

The ``__main__`` guard is load-bearing: the process pool the action uses for
per-event parallelism runs with ``mp_context="spawn"``, which re-imports this
module in each worker child to set up its ``__main__`` for unpickling. Without
the guard, the child would re-run ``main(context)`` against the already-created
LeRobot output directory.
"""

import roboto

from .. import main

if __name__ == "__main__":
    context = roboto.InvocationContext.from_env()
    main(context)
