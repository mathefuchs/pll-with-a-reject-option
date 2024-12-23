""" Configurations. """

from glob import glob
from typing import Dict, Tuple

# Dataset kind
DATASET_KIND = {
    "uci": 0,
    "rl": 1,
    "mnistlike": 2,
}

# Augmentation type
AUG_TYPE = {
    "rl": 0,
    "uniform": 1,
    "class-dependent": 2,
    "instance-dependent": 3,
}

# Data splits
SPLIT_IDX = {
    "train": 0,
    "test": 1,
    "holdout": 2,
}

# Data
SELECTED_DATASETS: Dict[str, Tuple[int, str]] = {
    # UCI datasets
    "ecoli": (0, "uci"),
    "first-order-theorem": (1, "uci"),
    "mfeat-fourier": (2, "uci"),
    "pendigits": (3, "uci"),
    "semeion": (4, "uci"),
    "statlog-landsat-satellite": (5, "uci"),
    "flare": (6, "uci"),
    # Real-world datasets
    "bird-song": (7, "rl"),
    "mir-flickr": (8, "rl"),
    "msrc-v2": (9, "rl"),
    "yahoo-news": (10, "rl"),
    # MNIST datasets
    "mnist": (11, "mnistlike"),
    "fmnist": (12, "mnistlike"),
    "kmnist": (13, "mnistlike"),
}

# All UCI datasets
UCI_DATA = list(sorted(
    glob("data/ucipp/uci/*.arff")
))
UCI_DATA_LABELS = [
    path.split("/")[-1].split(".")[0] for path in UCI_DATA
]
UCI_LABEL_TO_PATH = dict(zip(UCI_DATA_LABELS, UCI_DATA))

# All real-world datasets
REAL_WORLD_DATA = list(sorted(
    glob("data/realworld-datasets/*.mat")
))
REAL_WORLD_DATA_LABELS = [
    path.split("/")[-1].split(".")[0] for path in REAL_WORLD_DATA
]
REAL_WORLD_LABEL_TO_PATH = dict(zip(REAL_WORLD_DATA_LABELS, REAL_WORLD_DATA))
