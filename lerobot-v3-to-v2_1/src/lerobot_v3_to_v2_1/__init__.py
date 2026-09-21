"""LeRobot v3.0 -> v2.1 downgrade Roboto action.

The package ``__init__`` is intentionally import-light: ``main`` lives in
:mod:`lerobot_v3_to_v2_1.main` and is imported there directly by the Docker
entrypoint. Keeping it out of here lets the pure metadata helpers in
:mod:`lerobot_v3_to_v2_1.lerobot_dataset` be imported (and unit-tested) without
pulling in lerobot/torch.
"""
