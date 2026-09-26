"""Per-series z-score scaler interface for future Rel-ESE experiments."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

NAME = "zscore"

@dataclass(frozen=True)
class ZScoreScaler:
    mean: float
    scale: float
    def transform(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=float) - self.mean) / self.scale
    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) * self.scale + self.mean


def fit(y: np.ndarray, *, min_scale: float = 1e-12) -> ZScoreScaler:
    arr = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Scaler input must contain only finite values.")
    mean = float(np.mean(arr))
    scale = float(np.std(arr))
    if not np.isfinite(scale) or scale < min_scale:
        scale = 1.0
    return ZScoreScaler(mean=mean, scale=scale)