"""Pytest wrapper for the docker bag-replay smoke.

This is layer-2 of the runtime-parity gate: it shells out to
``test/ros_smoke/run.sh``, which builds a ros:humble-ros-base image,
runs the generated node against a real MCAP fixture, and asserts a
message lands on the action topic. Unlike the layer-1 pytest smoke
(:mod:`test_generated_node_imports`), this test needs docker, an
internet connection (to fetch the fixture on first run), and several
minutes of wall-clock.

The default pytest run **skips** this test. Set
``ROBOTO_RUN_ROS_SMOKE=1`` to opt in — every release-cut should run it
locally before tagging v0.1.0. Keeping it skip-by-default means CI
stays ROS-infra-free and PR runs stay fast.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_SMOKE_DIR = Path(__file__).resolve().parent / "ros_smoke"


pytestmark = pytest.mark.skipif(
    os.environ.get("ROBOTO_RUN_ROS_SMOKE") != "1",
    reason=(
        "Docker bag-replay smoke is opt-in. Set ROBOTO_RUN_ROS_SMOKE=1 "
        "to run; requires docker, internet for first-run fixture "
        "download, and several minutes of wall-clock."
    ),
)


def test_docker_bag_replay_publishes_action_message():
    """Build the ROS 2 Humble image; assert run.sh exits 0.

    ``run.sh`` already handles fixture fetch, image build, container
    run, and asserts that a message arrives on ``/teleop/action`` via
    ``ros2 topic echo --once`` inside the container. This wrapper
    exists so a developer can run the gate as ``pytest -k smoke``
    instead of remembering the shell-script path.
    """
    if shutil.which("docker") is None:
        pytest.skip("docker not on PATH")

    run_sh = _SMOKE_DIR / "run.sh"
    if not run_sh.is_file():
        pytest.fail(f"missing harness script: {run_sh}")

    # Inherit the parent env so ROBOTO_PROFILE, ROBOTO_ORG_ID, and
    # ROS_SMOKE_TIMEOUT overrides flow through. Capture combined
    # stdout/stderr so the pytest -v report shows the failure trail
    # without the developer having to re-run by hand.
    #
    # ROS_SMOKE_TIMEOUT only bounds the in-container `ros2 topic echo`.
    # The host-side steps run.sh does first — fixture fetch, image pull,
    # apt, colcon build — are unbounded, so a stalled first-run base-image
    # pull would hang pytest forever. Bound the whole thing host-side;
    # the default is generous to cover a cold image pull.
    host_timeout = float(os.environ.get("ROBOTO_ROS_SMOKE_HOST_TIMEOUT", "1800"))
    try:
        result = subprocess.run(
            [str(run_sh)],
            cwd=str(_SMOKE_DIR.parent.parent),
            capture_output=True,
            text=True,
            check=False,
            timeout=host_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"docker bag-replay smoke timed out after {host_timeout:.0f}s "
            "(host-side; override via ROBOTO_ROS_SMOKE_HOST_TIMEOUT).\n"
            f"--- run.sh stdout (partial) ---\n{exc.stdout or ''}\n"
            f"--- run.sh stderr (partial) ---\n{exc.stderr or ''}"
        )

    if result.returncode != 0:
        pytest.fail(
            "docker bag-replay smoke failed.\n"
            f"--- run.sh stdout ---\n{result.stdout}\n"
            f"--- run.sh stderr ---\n{result.stderr}"
        )
