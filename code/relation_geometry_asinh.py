"""Signed-compressed asinh relation geometry.

T(y)=asinh(y) is approximately linear near zero and logarithmic for large
absolute values.  It is smooth, invertible, and defined on the full real line.
"""
from __future__ import annotations
import numpy as np
from relation_geometry_utils import as_1d_finite, solve_common_offset

NAME = "asinh"
GEOMETRY = "signed_compressed_asinh"
ALLOWS_NEGATIVE = True
SIMPLEX_STATE = False


def validate(y: np.ndarray, *, epsilon: float = 1e-8, name: str = "y") -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def transform(y: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    return np.arcsinh(validate(y, epsilon=epsilon))


def inverse(x: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    # np.sinh overflows only at very large arguments; clip far outside any
    # realistic transformed time-series range to keep root bracketing stable.
    return np.sinh(np.clip(x, -700.0, 700.0))


def reconstruct(u: np.ndarray, aggregate: float, *, epsilon: float = 1e-8) -> np.ndarray:
    u = as_1d_finite(u, "u")
    _, y_hat = solve_common_offset(
        u,
        aggregate,
        lambda z: inverse(z, epsilon=epsilon),
    )
    return y_hat
