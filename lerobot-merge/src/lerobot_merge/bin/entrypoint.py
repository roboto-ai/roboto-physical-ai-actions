"""
Used as the entrypoint script when running in a Docker container.

Requires Roboto-specific environment variables to be set,
which is done automatically when run on hosted compute or by `roboto actions invoke-local` local Docker execution.
"""

import roboto

from .. import main

context = roboto.InvocationContext.from_env()
main(context)
