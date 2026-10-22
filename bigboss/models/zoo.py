"""Model A family: net-excess-return forecasters.

All models share one contract:
    fit(X, y)                -> self
    predict(X) -> DataFrame with columns [mean, q10, q50, q90] (horizon excess return, decimal)
Models without native quantile outputs use empirical residual quantiles from training data;
all models are then conformally recalibrated by the caller on a held-out calibration slice.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

QUANTILES = (0.1, 0.5, 0.9)


class Preprocessor:
    """Median-impute, winsorize at 1/99 pct, standardize. Fitted on training data only."""

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.cols = list(X.columns)
        self.med = X.median()
        self.lo = X.quantile(0.01)
        self.hi = X.quantile(0.99)
        Z = X.clip(self.lo, self.hi, axis=1).fillna(self.med)
        self.mu = Z.mean()
        self.sd = Z.std().replace(0, 1).fillna(1)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = X[self.cols].clip(self.lo, self.hi, axis=1).fillna(self.med)
        return ((Z - self.mu) / self.sd).fillna(0).to_numpy(np.float32)


def _residual_quantiles(mean_pred: np.ndarray, resid: np.ndarray) -> pd.DataFrame:
    qs = np.quantile(resid, QUANTILES)
    return pd.DataFrame({"mean": mean_pred, "q10": mean_pred + qs[0], "q50": mean_pred + qs[1], "q90": mean_pred + qs[2]})


@dataclass
class BaseModel:
    name: str = "base"
    params: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series):  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError

    def feature_importance(self) -> dict[str, float]:
        return {}


class ZeroExcessModel(BaseModel):
    """Efficient-market baseline: expected excess return is zero; uncertainty from history."""

    def __init__(self):
        super().__init__("zero_excess")

    def fit(self, X, y):
        self.resid = np.asarray(y)
        return self

    def predict(self, X):
        return _residual_quantiles(np.zeros(len(X)), self.resid)


class HistoricalMeanModel(BaseModel):
    """Per-asset shrunk historical mean excess return (James-Stein-style shrinkage to the pooled mean)."""

    def __init__(self, shrink_obs: int = 500):
        super().__init__("hist_mean", {"shrink_obs": shrink_obs})

    def fit(self, X, y):
        syms = X.index.get_level_values("symbol")
        df = pd.DataFrame({"y": np.asarray(y), "s": syms})
        pooled = df["y"].mean()
        g = df.groupby("s")["y"].agg(["mean", "size"])
        w = g["size"] / (g["size"] + self.params["shrink_obs"])
        self.means = (w * g["mean"] + (1 - w) * pooled).to_dict()
        self.pooled = pooled
        fitted = df["s"].map(self.means).to_numpy()
        self.resid = df["y"].to_numpy() - fitted
        return self

    def predict(self, X):
        m = X.index.get_level_values("symbol").map(lambda s: self.means.get(s, self.pooled)).to_numpy(float)
        return _residual_quantiles(m, self.resid)


class MomentumRuleModel(BaseModel):
    """Transparent momentum rule: excess = b * (cross-sectional rank of 12-1 momentum - 0.5), b by OLS."""

    def __init__(self, feature: str = "mom_12_1_xrank"):
        super().__init__("momentum_rule", {"feature": feature})

    def fit(self, X, y):
        x = X[self.params["feature"]].fillna(0.5).to_numpy() - 0.5
        yy = np.asarray(y)
        self.b = float((x @ yy) / max(x @ x, 1e-12))
        self.a = float(np.mean(yy))
        self.resid = yy - (self.a + self.b * x)
        return self

    def predict(self, X):
        x = X[self.params["feature"]].fillna(0.5).to_numpy() - 0.5
        return _residual_quantiles(self.a + self.b * x, self.resid)

    def feature_importance(self):
        return {self.params["feature"]: abs(self.b)}


class LinearFactorModel(BaseModel):
    """Ridge regression on standardized features - the conventional linear factor baseline."""

    def __init__(self, alpha: float = 50.0):
        super().__init__("linear_ridge", {"alpha": alpha})

    def fit(self, X, y):
        self.pre = Preprocessor().fit(X)
        Z = self.pre.transform(X)
        self.m = Ridge(alpha=self.params["alpha"]).fit(Z, np.asarray(y))
        self.resid = np.asarray(y) - self.m.predict(Z)
        return self

    def predict(self, X):
        return _residual_quantiles(self.m.predict(self.pre.transform(X)), self.resid)

    def feature_importance(self):
        return dict(zip(self.pre.cols, np.abs(self.m.coef_).tolist()))


class LightGBMModel(BaseModel):
    """LightGBM: a Huber-loss mean model plus three quantile models."""

    def __init__(self, **params):
        base = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.7,
                    subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=max(1, (os.cpu_count() or 2) - 1))
        base.update(params)
        super().__init__("lightgbm", base)

    def fit(self, X, y):
        import lightgbm as lgb

        yy = np.asarray(y)
        delta = float(np.quantile(np.abs(yy - np.median(yy)), 0.9))
        self.mean_m = lgb.LGBMRegressor(objective="huber", alpha=max(delta, 1e-4), **self.params).fit(X, yy)
        self.q_m = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **self.params).fit(X, yy) for q in QUANTILES}
        self.cols = list(X.columns)
        return self

    def predict(self, X):
        X = X[self.cols]
        q = {t: m.predict(X) for t, m in self.q_m.items()}
        qs = np.sort(np.vstack([q[0.1], q[0.5], q[0.9]]), axis=0)  # repair any quantile crossing
        return pd.DataFrame({"mean": self.mean_m.predict(X), "q10": qs[0], "q50": qs[1], "q90": qs[2]})

    def feature_importance(self):
        gain = self.mean_m.booster_.feature_importance("gain")
        return dict(zip(self.cols, (gain / max(gain.sum(), 1e-12)).tolist()))


class XGBoostModel(BaseModel):
    """XGBoost: pseudo-Huber mean model + native multi-quantile model (xgboost >= 2.0)."""

    def __init__(self, **params):
        base = dict(n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=50, subsample=0.7,
                    colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", n_jobs=max(1, (os.cpu_count() or 2) - 1))
        base.update(params)
        super().__init__("xgboost", base)

    def fit(self, X, y):
        import xgboost as xgb

        dev = "cuda" if _cuda_available() else "cpu"
        yy = np.asarray(y)
        self.mean_m = xgb.XGBRegressor(objective="reg:pseudohubererror", huber_slope=float(np.std(yy)), device=dev, **self.params).fit(X, yy)
        self.q_m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(QUANTILES), device=dev, **self.params).fit(X, yy)
        self.cols = list(X.columns)
        return self

    def predict(self, X):
        X = X[self.cols]
        qs = np.sort(np.asarray(self.q_m.predict(X)).reshape(len(X), -1).T, axis=0)
        return pd.DataFrame({"mean": self.mean_m.predict(X), "q10": qs[0], "q50": qs[1], "q90": qs[2]})

    def feature_importance(self):
        imp = self.mean_m.feature_importances_
        return dict(zip(self.cols, (imp / max(imp.sum(), 1e-12)).tolist()))


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class MLPModel(BaseModel):
    """Small multi-head MLP trained with pinball loss (quantiles) + Huber (mean). Uses CUDA when available.

    Deliberately small: on daily cross-sectional data, larger networks mostly memorize noise.
    """

    def __init__(self, hidden: int = 64, epochs: int = 15, lr: float = 1e-3, dropout: float = 0.2, seed: int = 0, batch: int = 2048):
        super().__init__("mlp", dict(hidden=hidden, epochs=epochs, lr=lr, dropout=dropout, seed=seed, batch=batch))

    def fit(self, X, y):
        import torch
        from torch import nn

        torch.manual_seed(self.params["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pre = Preprocessor().fit(X)
        Z = torch.tensor(self.pre.transform(X))
        yy = np.asarray(y, dtype=np.float32)
        self.y_scale = float(np.std(yy)) or 1.0
        Y = torch.tensor(yy / self.y_scale)
        h, p = self.params["hidden"], self.params["dropout"]
        self.net = nn.Sequential(nn.Linear(Z.shape[1], h), nn.GELU(), nn.Dropout(p), nn.Linear(h, h // 2), nn.GELU(),
                                 nn.Dropout(p), nn.Linear(h // 2, 4)).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.params["lr"], weight_decay=1e-3)
        taus = torch.tensor(QUANTILES, device=self.device)
        n = len(Z)
        for _ in range(self.params["epochs"]):
            perm = torch.randperm(n)
            self.net.train()
            for i in range(0, n, self.params["batch"]):
                idx = perm[i:i + self.params["batch"]]
                xb, yb = Z[idx].to(self.device), Y[idx].to(self.device)
                out = self.net(xb)
                mean_loss = nn.functional.huber_loss(out[:, 0], yb, delta=1.0)
                d = yb[:, None] - out[:, 1:]
                q_loss = torch.maximum(taus * d, (taus - 1) * d).mean()
                loss = mean_loss + q_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        import torch

        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(self.pre.transform(X)).to(self.device)).cpu().numpy() * self.y_scale
        qs = np.sort(out[:, 1:].T, axis=0)
        return pd.DataFrame({"mean": out[:, 0], "q10": qs[0], "q50": qs[1], "q90": qs[2]})


MODEL_REGISTRY = {
    "zero_excess": ZeroExcessModel,
    "hist_mean": HistoricalMeanModel,
    "momentum_rule": MomentumRuleModel,
    "linear_ridge": LinearFactorModel,
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
    "mlp": MLPModel,
}

BASELINES = ["zero_excess", "hist_mean", "momentum_rule", "linear_ridge"]


def make_model(name: str, **params) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name}; options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**params)


def available_models() -> list[str]:
    out = list(BASELINES) + ["lightgbm"]
    try:
        import xgboost  # noqa: F401

        out.append("xgboost")
    except ImportError:
        pass
    try:
        import torch  # noqa: F401

        out.append("mlp")
    except ImportError:
        pass
    return out
