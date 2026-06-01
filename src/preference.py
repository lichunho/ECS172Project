"""Individual preference model.

Explicit-rating matrix factorization over the BGG 1-10 ratings. The public
surface (fit / predict / save / load) matches what the rest of the pipeline
depends on; the internals are a biased SVD rather than LightFM, because LightFM
has no build for this environment (Python 3.13 / numpy 2.x). Because we model the
explicit rating, predicted scores live on the 1-10 scale and RMSE is meaningful.

Model:  r_hat(u, i) = mu + b_u + b_i + P[u] . Q[i]
where mu is the global mean, b_u / b_i are regularized bias terms, and P / Q are
latent factors from a truncated SVD of the bias-corrected residual matrix.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class PreferenceModel:
    """Biased matrix-factorization model producing per-user, per-game scores.

    Constructor args are kept for interface/config compatibility. `no_components`
    is the latent dimension. `loss`, `learning_rate`, and `epochs` are inert for
    the SVD engine (no per-example SGD) and retained only so configs/callers keep
    working; see model.yaml.
    """

    def __init__(
        self,
        loss: str = "warp",
        no_components: int = 64,
        learning_rate: float = 0.05,
        epochs: int = 30,
        bias_reg: float = 5.0,
    ) -> None:
        self.loss = loss
        self.no_components = no_components
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias_reg = bias_reg

        self.mu_: float = 0.0
        self.user_bias_: np.ndarray | None = None
        self.item_bias_: np.ndarray | None = None
        self.user_factors_: np.ndarray | None = None
        self.item_factors_: np.ndarray | None = None
        self.n_users_: int = 0
        self.n_items_: int = 0

    def fit(self, ratings: pd.DataFrame, item_features: pd.DataFrame | None = None) -> "PreferenceModel":
        """Fit on a ratings frame with columns user_id, game_id, rating.

        `item_features` is accepted for interface compatibility but unused: this
        explicit-MF engine is collaborative-filtering only (content features were
        the LightFM hybrid path). user_id / game_id are assumed dense (0..N-1),
        which is what src/data.preprocess emits.
        """
        u = ratings["user_id"].to_numpy()
        i = ratings["game_id"].to_numpy()
        r = ratings["rating"].to_numpy(dtype=np.float64)

        self.n_users_ = int(u.max()) + 1
        self.n_items_ = int(i.max()) + 1
        lam = self.bias_reg

        self.mu_ = float(r.mean())
        dev = r - self.mu_

        # Regularized item bias: shrink toward 0 by lam pseudo-counts.
        item_sum = np.bincount(i, weights=dev, minlength=self.n_items_)
        item_cnt = np.bincount(i, minlength=self.n_items_)
        self.item_bias_ = item_sum / (item_cnt + lam)

        # Regularized user bias on the item-bias-corrected residual.
        dev_u = dev - self.item_bias_[i]
        user_sum = np.bincount(u, weights=dev_u, minlength=self.n_users_)
        user_cnt = np.bincount(u, minlength=self.n_users_)
        self.user_bias_ = user_sum / (user_cnt + lam)

        # Latent factors from the fully bias-corrected residual matrix.
        resid = dev_u - self.user_bias_[u]
        R = csr_matrix((resid, (u, i)), shape=(self.n_users_, self.n_items_))

        k = min(self.no_components, min(R.shape) - 1)
        svd = TruncatedSVD(n_components=k, random_state=0)
        self.user_factors_ = svd.fit_transform(R).astype(np.float32)   # U * S
        self.item_factors_ = svd.components_.T.astype(np.float32)       # V
        return self

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Return predicted scores, shape (len(user_ids), len(item_ids))."""
        if self.user_factors_ is None:
            raise RuntimeError("model is not fitted")
        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)

        latent = self.user_factors_[user_ids] @ self.item_factors_[item_ids].T
        scores = (
            self.mu_
            + self.user_bias_[user_ids][:, None]
            + self.item_bias_[item_ids][None, :]
            + latent
        )
        return scores

    def predict_pairs(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Predict the score for each aligned (user_id, item_id) pair.

        Returns a 1-D array of length len(user_ids). Used for RMSE on held-out
        (user, item, rating) triples without materializing a full matrix.
        """
        if self.user_factors_ is None:
            raise RuntimeError("model is not fitted")
        u = np.asarray(user_ids)
        i = np.asarray(item_ids)
        latent = np.einsum("ij,ij->i", self.user_factors_[u], self.item_factors_[i])
        return self.mu_ + self.user_bias_[u] + self.item_bias_[i] + latent

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "loss": self.loss,
            "no_components": self.no_components,
            "mu": self.mu_,
            "user_bias": self.user_bias_,
            "item_bias": self.item_bias_,
            "user_factors": self.user_factors_,
            "item_factors": self.item_factors_,
            "n_users": self.n_users_,
            "n_items": self.n_items_,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "PreferenceModel":
        with open(path, "rb") as f:
            state = pickle.load(f)
        model = cls(loss=state["loss"], no_components=state["no_components"])
        model.mu_ = state["mu"]
        model.user_bias_ = state["user_bias"]
        model.item_bias_ = state["item_bias"]
        model.user_factors_ = state["user_factors"]
        model.item_factors_ = state["item_factors"]
        model.n_users_ = state["n_users"]
        model.n_items_ = state["n_items"]
        return model
