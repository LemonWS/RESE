"""Identity/additive relation geometry for signed systems."""
from __future__ import annotations
import numpy as np
from relation_geometry_utils import as_1d_finite

NAME = "additive"
GEOMETRY = "additive"
ALLOWS_NEGATIVE = True
SIMPLEX_STATE = False


def validate(y: np.ndarray, *, epsilon: float = 1e-8, name: str = "y") -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def transform(y: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    return validate(y, epsilon=epsilon).copy()


def inverse(x: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    return np.asarray(x, dtype=float).copy()


def reconstruct(u: np.ndarray, aggregate: float, *, epsilon: float = 1e-8) -> np.ndarray:
    u = as_1d_finite(u, "u")
    aggregate = float(aggregate)
    if not np.isfinite(aggregate):
        raise ValueError("aggregate must be finite.")
    offset = (aggregate - float(np.sum(u))) / float(len(u))
    y_hat = u + offset
    correction = (aggregate - float(np.sum(y_hat))) / float(len(y_hat))
    return np.asarray(y_hat + correction, dtype=float)
