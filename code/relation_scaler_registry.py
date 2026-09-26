"""Registry for optional Rel-ESE preprocessing scalers.

The current forecasting demo keeps ``scaler='none'`` to preserve the existing
ESE semantics.  This registry exists so future anomaly-detection experiments
can add z-score or robust scaling without changing the relation-geometry API.
"""
from __future__ import annotations

import relation_scaler_none as _none
import relation_scaler_zscore as _zscore
import relation_scaler_robust as _robust

SCALERS = {
    _none.NAME: _none,
    _zscore.NAME: _zscore,
    _robust.NAME: _robust,
}

ALIASES = {
    "none": "none",
    "raw": "none",
    "identity": "none",
    "z": "zscore",
    "standard": "zscore",
    "standardize": "zscore",
    "zscore": "zscore",
    "robust": "robust",
    "iqr": "robust",
    "median_iqr": "robust",
}


def normalize_scaler_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in ALIASES:
        raise ValueError(
            f"Unknown scaler {name!r}. Expected one of: "
            + ", ".join(sorted(SCALERS))
        )
    return ALIASES[key]


def fit_scaler(y, name: str = "none", **kwargs):
    canonical = normalize_scaler_name(name)
    return SCALERS[canonical].fit(y, **kwargs)


def available_scalers() -> tuple[str, ...]:
    return tuple(sorted(SCALERS))