"""Individual preference model (LightFM hybrid CF + content)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from lightfm import LightFM
from lightfm.data import Dataset


_TEXT_COLUMNS = {"name", "description"}
_ID_COLUMNS = {"game_id", "user_id"}


def _tokenize_value(column: str, value: object) -> list[str]:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        tokens: list[str] = []
        for element in value:
            if pd.isna(element):
                continue
            tokens.append(f"{column}={str(element)}")
        return tokens
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if pd.isna(value):
        return []
    if column in _TEXT_COLUMNS:
        return []
    if isinstance(value, str):
        return [f"{column}={value}"]
    if isinstance(value, (np.integer, int)):
        return [f"{column}={int(value)}"]
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return []
        if column in {
            "weight",
            "complexity",
            "language_dependence",
            "min_age_fit",
            "player_elimination",
            "social_conflict",
            "party_friendliness",
            "interaction_level",
            "competitiveness",
            "downtime",
            "teach_time",
            "mixed_skill_robustness",
            "min_players",
            "max_players",
            "min_playtime",
            "max_playtime",
        }:
            return [f"{column}={int(round(float(value)))}"]
        return [f"{column}={round(float(value), 3)}"]
    return [f"{column}={str(value)}"]


def _build_item_feature_lists(item_features: pd.DataFrame) -> tuple[list[object], dict[object, list[str]], list[str]]:
    frame = item_features.copy()
    if "game_id" in frame.columns:
        frame = frame.set_index("game_id")
    item_ids = list(frame.index.tolist())
    feature_map: dict[object, list[str]] = {}
    feature_set: set[str] = set()
    for item_id, row in frame.iterrows():
        tokens: list[str] = []
        for column, value in row.items():
            if column in _ID_COLUMNS or column in _TEXT_COLUMNS:
                continue
            tokens.extend(_tokenize_value(column, value))
        feature_map[item_id] = tokens
        feature_set.update(tokens)
    return item_ids, feature_map, sorted(feature_set)


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
        self._user_id_map: dict[object, int] | None = None
        self._item_id_map: dict[object, int] | None = None
        self._item_features_matrix: sp.csr_matrix | None = None

    def fit(self, ratings: pd.DataFrame, item_features: pd.DataFrame | None = None) -> "PreferenceModel":
        required = {"user_id", "game_id", "rating"}
        missing = required.difference(ratings.columns)
        if missing:
            raise ValueError(f"ratings is missing required columns: {sorted(missing)}")

        ratings = ratings.dropna(subset=["user_id", "game_id", "rating"]).copy()
        ratings["user_id"] = ratings["user_id"].astype(int)
        ratings["game_id"] = ratings["game_id"].astype(int)

        users = np.sort(ratings["user_id"].unique())
        items = np.sort(ratings["game_id"].unique())

        dataset = Dataset()
        item_features_matrix = None
        if item_features is not None and not item_features.empty:
            item_ids, feature_map, feature_names = _build_item_feature_lists(item_features)
            items = np.sort(np.unique(np.concatenate([items, np.asarray(item_ids, dtype=int)])))
            dataset.fit(users=users.tolist(), items=items.tolist(), item_features=feature_names)
            item_features_matrix = dataset.build_item_features(
                (item_id, feature_map.get(item_id, [])) for item_id in items.tolist()
            )
        else:
            dataset.fit(users=users.tolist(), items=items.tolist())

        user_mapping, _, item_mapping, _ = dataset.mapping()
        self._user_id_map = user_mapping
        self._item_id_map = item_mapping
        self._item_features_matrix = item_features_matrix

        user_codes = ratings["user_id"].map(user_mapping).to_numpy(dtype=np.int32, copy=True)
        item_codes = ratings["game_id"].map(item_mapping).to_numpy(dtype=np.int32, copy=True)
        values = ratings["rating"].to_numpy(dtype=np.float32, copy=True)
        interactions = sp.coo_matrix(
            (values.copy(), (user_codes.copy(), item_codes.copy())),
            shape=(len(user_mapping), len(item_mapping)),
        ).tocsr().copy()

        self._model = LightFM(
            loss=self.loss,
            no_components=self.no_components,
            learning_rate=self.learning_rate,
            random_state=42,
        )
        self._model.fit(
            interactions,
            item_features=self._item_features_matrix,
            epochs=self.epochs,
            num_threads=1,
            verbose=False,
        )
        return self

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Return predicted scores shape (len(user_ids), len(item_ids))."""
        if self._model is None or self._user_id_map is None or self._item_id_map is None:
            raise RuntimeError("model has not been fit or loaded")

        users = np.asarray(user_ids)
        items = np.asarray(item_ids)
        user_codes = np.array([self._user_id_map[int(user_id)] for user_id in users], dtype=np.int32)
        item_codes = np.array([self._item_id_map[int(item_id)] for item_id in items], dtype=np.int32)

        rows = []
        for user_code in user_codes:
            repeated_users = np.full(len(item_codes), user_code, dtype=np.int32)
            if self._item_features_matrix is not None:
                row = self._model.predict(
                    repeated_users,
                    item_codes,
                    item_features=self._item_features_matrix,
                    num_threads=1,
                )
            else:
                row = self._model.predict(repeated_users, item_codes, num_threads=1)
            rows.append(row)
        return np.vstack(rows).astype(np.float32)

    def save(self, path: str) -> None:
        if self._model is None or self._user_id_map is None or self._item_id_map is None:
            raise RuntimeError("model has not been fit")
        payload = {
            "loss": self.loss,
            "no_components": self.no_components,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "model": self._model,
            "user_id_map": self._user_id_map,
            "item_id_map": self._item_id_map,
            "item_features_matrix": self._item_features_matrix,
        }
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str) -> "PreferenceModel":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        model = cls(
            loss=payload.get("loss", "warp"),
            no_components=payload.get("no_components", 64),
            learning_rate=payload.get("learning_rate", 0.05),
            epochs=payload.get("epochs", 30),
        )
        model._model = payload["model"]
        model._user_id_map = payload["user_id_map"]
        model._item_id_map = payload["item_id_map"]
        model._item_features_matrix = payload.get("item_features_matrix")
        return model
