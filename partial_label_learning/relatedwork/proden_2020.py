""" Module for PRODEN. """

import random
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult
from reference_models.lenet import LeNet
from reference_models.mlp import MLP


class Proden(PllBaseClassifier):
    """
    PRODEN by Lv et al.,
    "Progressive Identification of True Labels for Partial-Label Learning"
    """

    def __init__(
        self, rng: np.random.Generator, debug: bool = False, **kwargs,
    ) -> None:
        self.rng = rng
        self.device = torch.device("cpu")
        self.debug = debug
        self.loop_wrapper = tqdm if debug else (lambda x: x)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(int(self.rng.integers(1000)))
        torch.manual_seed(int(self.rng.integers(1000)))
        self.model: Optional[nn.Module] = None

    def fit(
        self, inputs: np.ndarray, partial_targets: np.ndarray,
    ) -> SplitResult:
        """ Fits the model to the given inputs.

        Args:
            inputs (np.ndarray): The inputs.
            partial_targets (np.ndarray): The partial targets.

        Returns:
            SplitResult: The disambiguated targets.
        """

        num_epoch = 200
        if len(inputs.shape) == 2:
            # Tabular data (Batch, Features)
            self.model = MLP(inputs.shape[1], partial_targets.shape[1])
            self.device = torch.device("cpu")  # Always fastest on CPU
            batch_size = 1024
        elif len(inputs.shape) == 4:
            # Image data (Batch, Channels, Height, Width)
            self.model = LeNet(partial_targets.shape[1])
            batch_size = 64
            if torch.cuda.is_available():
                cuda_idx = random.randrange(torch.cuda.device_count())
                self.device = torch.device(f"cuda:{cuda_idx}")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            raise ValueError(f"Malformed data of shape {inputs.shape}.")
        self.model.to(self.device)

        # Data preparation
        x_train = torch.tensor(inputs, dtype=torch.float32)
        y_train = torch.tensor(partial_targets, dtype=torch.float32)
        train_indices = torch.arange(x_train.shape[0], dtype=torch.int32)
        loss_weights = torch.tensor(partial_targets, dtype=torch.float32)
        loss_weights /= loss_weights.sum(dim=1, keepdim=True)
        data_loader = DataLoader(
            TensorDataset(train_indices, x_train, y_train, loss_weights),
            batch_size=batch_size, shuffle=True, generator=self.torch_rng,
        )

        # Optimizer
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())

        # Training loop
        for _ in self.loop_wrapper(range(num_epoch)):
            for idx, inputs_i, partial_targets_i, w_ij in data_loader:
                # Move to device
                inputs_i = inputs_i.to(self.device)
                partial_targets_i = partial_targets_i.to(self.device)
                w_ij = w_ij.to(self.device)

                # Forward-backward pass
                probs = self.model(inputs_i)
                loss = torch.mean(torch.sum(
                    w_ij * -torch.log(probs + 1e-10), dim=1,
                ))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update weights
                with torch.no_grad():
                    updated_w = partial_targets_i * probs
                    updated_w /= torch.sum(updated_w, dim=1, keepdim=True)
                    loss_weights[idx] = updated_w.to("cpu")

        # Return results
        return SplitResult.from_scores(self.rng, loss_weights.numpy())

    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        if self.model is None:
            raise ValueError()

        if len(inputs.shape) == 2:
            batch_size = 1024
        elif len(inputs.shape) == 4:
            batch_size = 64
        else:
            raise ValueError(f"Malformed data of shape {inputs.shape}.")
        inference_loader = DataLoader(
            TensorDataset(torch.tensor(
                inputs, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False,
        )

        # Switch to eval mode
        self.model.eval()
        all_results = []
        with torch.no_grad():
            for x_batch in inference_loader:
                x_batch = x_batch[0].to(self.device)
                all_results.append(
                    self.model(x_batch).to("cpu").numpy())
            train_probs = np.vstack(all_results)
        return SplitResult.from_scores(self.rng, train_probs)
