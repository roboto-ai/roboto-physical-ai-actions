import numpy as np
import roboto
from lerobot.datasets.dataset_tools import add_features

from .lerobot_dataset import (
    convert_to_v30_if_necessary,
    find_lerobot_dataset_root,
    load_from_directory,
)
from .logger import logger


def calculate_action_observation_difference(dataset):
    """Calculate action-observation difference for all frames in the dataset.

    Returns a 1D object array where each element is a separate numpy array,
    compatible with LeRobot's add_features function and pandas storage.
    """
    actions = dataset.hf_dataset["action"]
    observations = dataset.hf_dataset["observation.state"]

    if not isinstance(actions, np.ndarray):
        actions = np.array(actions)
    if not isinstance(observations, np.ndarray):
        observations = np.array(observations)

    differences = actions - observations

    if len(differences.shape) > 1:
        result = np.empty(len(differences), dtype=object)
        for i in range(len(differences)):
            result[i] = differences[i]
        return result

    return differences


def main(context: roboto.InvocationContext) -> None:
    logger.setLevel(context.log_level)

    dataset_root = find_lerobot_dataset_root(context.input_dir)
    logger.info("Found LeRobot dataset root: %s", dataset_root)
    convert_to_v30_if_necessary(dataset_root)

    source_lerobot_ds = load_from_directory(dataset_root)
    logger.info("Successfully loaded LeRobot dataset: %s", source_lerobot_ds.repo_id)

    logger.info("Calculating action-observation differences...")
    action_obs_diff_values = calculate_action_observation_difference(source_lerobot_ds)

    logger.info("Created derived LeRobot dataset with additional feature")
    action_feature = source_lerobot_ds.meta.features["action"]
    feature_info: dict = {
        "dtype": action_feature["dtype"],
        "shape": action_feature["shape"],
        "names": action_feature["names"],
    }

    features: dict = {
        "action_observation_difference": (action_obs_diff_values, feature_info)
    }
    repo_id = f"{source_lerobot_ds.repo_id}_enriched"
    add_features(
        dataset=source_lerobot_ds,
        features=features,
        output_dir=str(context.output_dir / repo_id),
        repo_id=repo_id,
    )
    logger.info("Finished creating enriched LeRobot dataset.")
