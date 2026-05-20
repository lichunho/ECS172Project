"""Individual preference model (LightFM hybrid CF + content)."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PreferenceModel:
    """LightFM wrapper producing per-user, per-game predicted scores."""

    def __init__(
        self,
        loss: str = "warp",
        no_components: int = 64,
        learning_rate: float = 0.05,
        epochs: int = 30,
    ) -> None:
        self.loss = loss
        self.no_components = no_components
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = None

    def fit(self, ratings: pd.DataFrame, item_features: pd.DataFrame | None = None) -> "PreferenceModel":
        raise NotImplementedError

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Return predicted scores shape (len(user_ids), len(item_ids))."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "PreferenceModel":
        raise NotImplementedError
