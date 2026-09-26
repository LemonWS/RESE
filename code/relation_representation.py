
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

import relation_geometry_log_ratio as _log
import relation_geometry_asinh as _asinh
import relation_geometry_signed_log as _signed_log
import relation_geometry_additive as _additive
from relation_geometry_utils import stable_softmax

AUTO = "auto"
LOG_RATIO = _log.NAME
ASINH = _asinh.NAME
SIGNED_LOG = _signed_log.NAME
ADDITIVE = _additive.NAME

_GEOMETRIES = {
    LOG_RATIO: _log,
    ASINH: _asinh,
    SIGNED_LOG: _signed_log,
    ADDITIVE: _additive,
}

_ALIASES = {
    AUTO: AUTO,
    "automatic": AUTO,
    "infer": AUTO,
    "log": LOG_RATIO,
    "logratio": LOG_RATIO,
    "log-ratio": LOG_RATIO,
    "multiplicative": LOG_RATIO,
    "positive": LOG_RATIO,
    LOG_RATIO: LOG_RATIO,
    "arcsinh": ASINH,
    "inverse_hyperbolic_sine": ASINH,
    ASINH: ASINH,
    "signed-log": SIGNED_LOG,
    "signedlog": SIGNED_LOG,
    SIGNED_LOG: SIGNED_LOG,
    "identity": ADDITIVE,
    "difference": ADDITIVE,
    "signed": ADDITIVE,
    ADDITIVE: ADDITIVE,
}


@dataclass(frozen=True)
class RepresentationInfo:
    name: str
    geometry: str
    allows_negative: bool
    simplex_state: bool


REPRESENTATION_INFO: Dict[str, RepresentationInfo] = {
    name: RepresentationInfo(
        name=name,
        geometry=module.GEOMETRY,
        allows_negative=bool(module.ALLOWS_NEGATIVE),
        simplex_state=bool(module.SIMPLEX_STATE),
    )
    for name, module in _GEOMETRIES.items()
}


def normalize_representation_name(name: str, *, allow_auto: bool = False) -> str:
    """Normalize aliases to a canonical geometry name.

    ``auto`` requires access to actual data, so callers that accept it should
    pass ``allow_auto=True`` and later call :func:`resolve_representation`.
    """
    key = str(name).strip().lower()
    try:
        normalized = _ALIASES[key]
    except KeyError as exc:
        allowed = [AUTO, *sorted(REPRESENTATION_INFO)]
        raise ValueError(
            f"Unknown relation representation {name!r}. Expected one of: "
            + ", ".join(allowed)
            + "."
        ) from exc
    if normalized == AUTO and not allow_auto:
        raise ValueError(
            "representation='auto' must be resolved from the dataset before "
            "pairwise estimation. Call resolve_representation(Y, 'auto')."
        )
    return normalized


def infer_representation(y: np.ndarray) -> str:
    """Safe default geometry inferred from actual model observations.

    Any negative observation rules out log-ratio.  For signed data the default
    is ``asinh`` rather than plain additive because it remains smooth near zero
    while compressing large absolute values.  For non-negative data the legacy
    multiplicative ESE geometry is retained.
    """
    arr = np.asarray(y, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot infer representation from an empty array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Cannot infer representation from NaN or Inf values.")
    return ASINH if np.any(arr < 0) else LOG_RATIO


def resolve_representation(y: np.ndarray, requested: str = AUTO) -> str:
    requested = normalize_representation_name(requested, allow_auto=True)
    if requested == AUTO:
        return infer_representation(y)
    return requested


def get_representation_info(name: str) -> RepresentationInfo:
    name = normalize_representation_name(name)
    return REPRESENTATION_INFO[name]


def available_representations(*, include_auto: bool = True) -> tuple[str, ...]:
    names = tuple(sorted(REPRESENTATION_INFO))
    return (AUTO, *names) if include_auto else names


def _module(name: str):
    return _GEOMETRIES[normalize_representation_name(name)]


def validate_observations(
    y: np.ndarray,
    representation: str = LOG_RATIO,
    *,
    epsilon: float = 1e-8,
    name: str = "y",
) -> np.ndarray:
    return _module(representation).validate(y, epsilon=epsilon, name=name)


def transform_series(
    y: np.ndarray,
    representation: str = LOG_RATIO,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    return np.asarray(
        _module(representation).transform(y, epsilon=epsilon), dtype=float
    )


def inverse_transform_series(
    x: np.ndarray,
    representation: str = LOG_RATIO,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    return np.asarray(
        _module(representation).inverse(x, epsilon=epsilon), dtype=float
    )


def relation_series(
    y_i: np.ndarray,
    y_j: np.ndarray,
    representation: str = LOG_RATIO,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    x_i = transform_series(y_i, representation, epsilon=epsilon)
    x_j = transform_series(y_j, representation, epsilon=epsilon)
    if x_i.shape != x_j.shape:
        raise ValueError(
            f"y_i and y_j must have the same shape; got {x_i.shape} and {x_j.shape}."
        )
    return x_i - x_j


def centered_latent_state(
    y: np.ndarray,
    representation: str = LOG_RATIO,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    x = transform_series(y, representation, epsilon=epsilon)
    if x.ndim != 1:
        raise ValueError("y must be one-dimensional for latent-state recovery.")
    return np.asarray(x - np.mean(x), dtype=float)


def simplex_state_from_latent(u: np.ndarray, representation: str) -> np.ndarray:
    representation = normalize_representation_name(representation)
    u = np.asarray(u, dtype=float)
    if representation == LOG_RATIO:
        return stable_softmax(u)
    return np.full(u.shape, np.nan, dtype=float)


def reconstruct_from_latent(
    u: np.ndarray,
    aggregate: float,
    representation: str = LOG_RATIO,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    return np.asarray(
        _module(representation).reconstruct(
            u, aggregate, epsilon=epsilon
        ),
        dtype=float,
    )


def realized_relation_matrix(
    y: np.ndarray,
    representation: str = LOG_RATIO,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    x = transform_series(y, representation, epsilon=epsilon)
    if x.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    R = x[:, None] - x[None, :]
    np.fill_diagonal(R, 0.0)
    return np.asarray(R, dtype=float)

