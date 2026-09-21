"""Minimal ament_python package for the docker bag-replay smoke.

``run.sh`` writes the gen-node output into ``inference_node/node.py``
before invoking ``colcon build``; the entry_point below registers it
as the executable ``ros2 run inference_node node``.

The ``policy`` submodule ships an input-dependent stub ``load_policy``:
each action is the first ``_ACTION_WIDTH`` entries of
``obs["observation.state"]`` plus a constant offset. That invariant is
what ``test/ros_smoke/verify_action.py`` searches for in the recorded
action — strong enough to fail a frozen-default policy or a broken obs
path, without bringing torch/lerobot/etc. into the container.
"""

from setuptools import setup

package_name = "inference_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "node = inference_node.node:main",
        ],
    },
)
