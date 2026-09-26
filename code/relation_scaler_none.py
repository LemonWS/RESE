"""No-op scaler interface reserved for Rel-ESE preprocessing."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

NAME = "none"

@dataclass(frozen=True)
class NoneScaler:
    def transform(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float).copy()
    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float).copy()


def fit(y: np.ndarray) -> NoneScaler:
    arr = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Scaler input must contain only finite values.")
    return NoneScaler()