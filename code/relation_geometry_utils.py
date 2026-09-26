"""Shared numerical utilities for Rel-ESE relation geometries."""
from __future__ import annotations

from typing import Callable
import numpy as np


def as_1d_finite(x: np.ndarray, name: str = "x") -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def stable_softmax(u: np.ndarray) -> np.ndarray:
    u = as_1d_finite(u, "u")
    shifted = u - float(np.max(u))
    exp_u = np.exp(shifted)
    total = float(np.sum(exp_u))
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Softmax normalization failed.")
    return exp_u / total


def solve_common_offset(
    u: np.ndarray,
    aggregate: float,
    inverse_transform: Callable[[np.ndarray], np.ndarray],
    *,
    initial_half_width: float = 1.0,
    max_expand: int = 80,
    max_iter: int = 160,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> tuple[float, np.ndarray]:
    """Find c such that sum_i T^{-1}(u_i+c) == aggregate.

    This supports any strictly increasing invertible geometry.  Bracketing is
    expanded exponentially and then a monotone bisection is used.
    """
    u = as_1d_finite(u, "u")
    aggregate = float(aggregate)
    if not np.isfinite(aggregate):
        raise ValueError("aggregate must be finite.")

    def f(c: float) -> float:
        y = np.asarray(inverse_transform(u + float(c)), dtype=float)
        if y.shape != u.shape or not np.all(np.isfinite(y)):
            # Preserve monotonic direction if an extreme inverse overflows.
            finite = y[np.isfinite(y)]
            if finite.size and np.mean(finite) > 0:
                return np.inf
            if finite.size and np.mean(finite) < 0:
                return -np.inf
            return np.inf if c > 0 else -np.inf
        return float(np.sum(y) - aggregate)

    half = float(initial_half_width)
    lo, hi = -half, half
    flo, fhi = f(lo), f(hi)

    for _ in range(max_expand):
        if flo <= 0 <= fhi:
            break
        if flo > 0:
            hi, fhi = lo, flo
            half *= 2.0
            lo = -half
            flo = f(lo)
        elif fhi < 0:
            lo, flo = hi, fhi
            half *= 2.0
            hi = half
            fhi = f(hi)
    else:
        raise RuntimeError(
            "Could not bracket the common reconstruction offset. "
            "Check the aggregate forecast and representation inverse."
        )

    target_tol = atol + rtol * max(1.0, abs(aggregate))
    mid = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) <= target_tol or (hi - lo) <= atol:
            break
        if fm < 0:
            lo = mid
        else:
            hi = mid

    y_hat = np.asarray(inverse_transform(u + mid), dtype=float)
    if y_hat.shape != u.shape or not np.all(np.isfinite(y_hat)):
        raise RuntimeError("Representation reconstruction produced invalid values.")

    # Tiny numerical correction in raw space; this keeps the aggregate exact
    # without materially changing the recovered relation geometry.
    correction = (aggregate - float(np.sum(y_hat))) / float(len(y_hat))
    y_hat = y_hat + correction
    return float(mid), np.asarray(y_hat, dtype=float)
