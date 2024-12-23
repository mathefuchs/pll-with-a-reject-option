""" Create data for PLL experiments. """

import random
import string

import numpy as np
import torch
from joblib import Parallel, delayed

from partial_label_learning.config import SELECTED_DATASETS
from partial_label_learning.data import (Experiment, get_mnist_dataset,
                                         get_rl_dataset, get_uci_dataset)

DEBUG = False


def create_experiment_data(
    dataset_name: str, dataset_kind: str, augment_type: str, seed: int,
):
    """ Create experiment data. """

    # Init random generator
    torch.manual_seed(seed)
    rng = np.random.Generator(np.random.PCG64(seed))

    # Load dataset
    if dataset_kind == "uci":
        dataset = get_uci_dataset(dataset_name)
        datasplit = dataset.create_data_split(rng)
    elif dataset_kind == "rl":
        dataset = get_rl_dataset(dataset_name)
        datasplit = dataset.create_data_split(rng)
    elif dataset_kind == "mnistlike":
        datasplit = get_mnist_dataset(dataset_name)
    else:
        raise ValueError()

    # Augment dataset
    if augment_type == "uniform":
        datasplit = datasplit.augment_targets(
            rng=rng, r_candidates=2, percent_partially_labeled=0.5,
            eps_cooccurrence=0.0,
        )
    elif augment_type == "class-dependent":
        datasplit = datasplit.augment_targets(rng, 1, 0.7, 0.7)
    elif augment_type == "instance-dependent":
        datasplit = datasplit.augment_targets_instance_dependent(rng)

    # Save experiment
    exp = Experiment(dataset_name, dataset_kind, augment_type, seed, datasplit)
    if not DEBUG:
        fname = "".join([
            random.choice(string.ascii_lowercase) for _ in range(10)])
        torch.save(exp, f"./experiments/{fname}.pt")
    else:
        torch.save(exp, f"./experiments/exp{seed}.pt")


if __name__ == "__main__":
    if not DEBUG:
        # Create experiment data
        Parallel(n_jobs=6)(
            delayed(create_experiment_data)(
                dataset_name, dataset_kind, augment_type, seed,
            )
            for dataset_name, (_, dataset_kind) in SELECTED_DATASETS.items()
            for augment_type in [
                "rl", "uniform", "class-dependent", "instance-dependent",
            ]
            if (
                (dataset_kind == "rl" and augment_type == "rl") or
                (dataset_kind != "rl" and augment_type != "rl")
            )
            for seed in range(5)
        )
    else:
        for s in range(5):
            # Create single data for debugging
            # create_experiment_data("first-order-theorem", "uci", "instance-dependent", s)
            create_experiment_data("kmnist", "mnistlike",
                                   "instance-dependent", s)
            # create_experiment_data("bird-song", "rl", "rl", s)
