""" Module for PL-KNN. """

import random

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from partial_label_learning.data import flatten_if_image
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult
from reference_models.vae import VariationalAutoEncoder


class PlKnn(PllBaseClassifier):
    """
    PL-KNN by Hüllermeier and Beringer,
    "Learning from Ambiguously Labeled Examples."
    """

    def __init__(
        self, rng: np.random.Generator,
        debug: bool = False, dataset_kind: str = "uci",
        dataset_name: str = "", **kwargs,
    ) -> None:
        self.rng = rng
        self.debug = debug
        self.knn = None
        self.y_train = None

        # Init variational auto-encoder
        if dataset_kind == "mnistlike":
            self.n_neighbors = 20
            if torch.cuda.is_available():
                cuda_idx = random.randrange(torch.cuda.device_count())
                self.device = torch.device(f"cuda:{cuda_idx}")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            self.vae = VariationalAutoEncoder()
            self.vae.load_state_dict(torch.load(
                f"./saved_models/vae-{dataset_name}.pt",
                map_location=self.device,
            ))
            self.vae.eval()
            self.vae.to(self.device)
        else:
            self.n_neighbors = 10
            self.device = None
            self.vae = None

    def _encode_data(self, inputs: np.ndarray) -> np.ndarray:
        """ Encode the input data. """

        inputs = flatten_if_image(inputs)
        if self.vae is None or self.device is None:
            return inputs

        with torch.no_grad():
            x_tensor = torch.tensor(
                inputs, dtype=torch.float32, device=self.device)
            _, x_enc, _ = self.vae(x_tensor, compute_loss=False)
        return x_enc.cpu().numpy()

    def _get_knn_y_pred(
        self, candidates: np.ndarray, is_transductive: bool,
        nn_dists: np.ndarray, nn_indices: np.ndarray,
    ) -> SplitResult:
        y_voting = np.zeros(
            (nn_indices.shape[0], self.y_train.shape[1]))
        for i, (nn_dist, nn_idx) in enumerate(zip(nn_dists, nn_indices)):
            dist_sum = nn_dist.sum()
            if dist_sum < 1e-6:
                sims = np.ones_like(nn_dist)
            else:
                sims = 1 - nn_dist / dist_sum

            for sim, idx in zip(sims, nn_idx):
                y_voting[i, :] += self.y_train[idx, :] * sim

        if is_transductive:
            for i in range(y_voting.shape[0]):
                y_voting[i, candidates[i] == 0] = 0

        # Return predictions
        return SplitResult.from_scores(self.rng, y_voting)

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

        inputs = self._encode_data(inputs)
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1)
        self.knn.fit(inputs)
        self.y_train = partial_targets
        return self._get_knn_y_pred(partial_targets, True, *self.knn.kneighbors())

    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        inputs = self._encode_data(inputs)
        return self._get_knn_y_pred(
            None, False, *self.knn.kneighbors(inputs))
