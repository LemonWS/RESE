"""Median/IQR robust scaler interface for future anomaly-aware Rel-ESE."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

NAME = "robust"

@dataclass(frozen=True)
class RobustScaler:
    median: float
    scale: float
    def transform(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=float) - self.median) / self.scale
    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) * self.scale + self.median


def fit(y: np.ndarray, *, min_scale: float = 1e-12) -> RobustScaler:
    arr = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Scaler input must contain only finite values.")
    median = float(np.median(arr))
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < min_scale:
        mad = float(np.median(np.abs(arr - median)))
        scale = 1.4826 * mad if mad >= min_scale else 1.0
    return RobustScaler(median=median, scale=scale)