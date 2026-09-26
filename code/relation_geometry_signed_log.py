"""Signed logarithmic relation geometry.

T(y)=sign(y)*log(1+|y|) preserves sign and compresses large magnitudes while
remaining defined at zero and for negative observations.
"""
from __future__ import annotations
import numpy as np
from relation_geometry_utils import as_1d_finite, solve_common_offset

NAME = "signed_log"
GEOMETRY = "signed_logarithmic"
ALLOWS_NEGATIVE = True
SIMPLEX_STATE = False


def validate(y: np.ndarray, *, epsilon: float = 1e-8, name: str = "y") -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def transform(y: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    arr = validate(y, epsilon=epsilon)
    return np.sign(arr) * np.log1p(np.abs(arr))


def inverse(x: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    a = np.clip(np.abs(x), 0.0, 700.0)
    return np.sign(x) * np.expm1(a)


def reconstruct(u: np.ndarray, aggregate: float, *, epsilon: float = 1e-8) -> np.ndarray:
    u = as_1d_finite(u, "u")
    _, y_hat = solve_common_offset(
        u,
        aggregate,
        lambda z: inverse(z, epsilon=epsilon),
    )
    return y_hat