"""Multiplicative/log-ratio relation geometry for non-negative systems."""
from __future__ import annotations
import numpy as np
from relation_geometry_utils import as_1d_finite, stable_softmax

NAME = "log_ratio"
GEOMETRY = "multiplicative"
ALLOWS_NEGATIVE = False
SIMPLEX_STATE = True


def validate(y: np.ndarray, *, epsilon: float = 1e-8, name: str = "y") -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if np.any(arr < 0):
        minimum = float(np.min(arr))
        raise ValueError(
            "log_ratio representation requires non-negative observations; "
            f"{name} has minimum {minimum:.6g}. Use 'asinh', 'signed_log', "
            "or 'additive' for signed data."
        )
    return arr


def transform(y: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    arr = validate(y, epsilon=epsilon)
    return np.log(arr + float(epsilon))


def inverse(x: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    # Used mainly for diagnostics.  Main ESE reconstruction below deliberately
    # preserves the historical M*softmax(u) interpretation.
    x = np.asarray(x, dtype=float)
    return np.maximum(np.exp(x) - float(epsilon), 0.0)


def reconstruct(u: np.ndarray, aggregate: float, *, epsilon: float = 1e-8) -> np.ndarray:
    u = as_1d_finite(u, "u")
    aggregate = float(aggregate)
    if not np.isfinite(aggregate):
        raise ValueError("aggregate must be finite.")
    if aggregate < 0:
        raise ValueError("log_ratio reconstruction requires a non-negative aggregate.")
    return aggregate * stable_softmax(u)
