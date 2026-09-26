
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, TypeVar

import numpy as np


AUTO = "auto"
EXPANDING = "expanding"
SLIDING = "sliding"

_HISTORY_MODES = {AUTO, EXPANDING, SLIDING}

T = TypeVar("T")


def normalize_history_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    aliases = {
        "rolling": SLIDING,
        "fixed": SLIDING,
        "fixed_window": SLIDING,
        "full": EXPANDING,
        "full_history": EXPANDING,
        "automatic": AUTO,
        "infer": AUTO,
    }
    value = aliases.get(value, value)
    if value not in _HISTORY_MODES:
        raise ValueError(
            f"Unknown history_mode={mode!r}. "
            f"Expected one of {sorted(_HISTORY_MODES)}."
        )
    return value


def resolve_history_mode(
    input_length: Optional[int],
    history_mode: str = AUTO,
) -> str:
    mode = normalize_history_mode(history_mode)
    if input_length is not None:
        input_length = int(input_length)
        if input_length < 1:
            raise ValueError("input_length must be None or a positive integer.")

    if mode == AUTO:
        return EXPANDING if input_length is None else SLIDING

    if mode == SLIDING and input_length is None:
        raise ValueError(
            "history_mode='sliding' requires input_length to be specified."
        )

    return mode


def validate_output_length(output_length: int) -> int:
    output_length = int(output_length)
    if output_length < 1:
        raise ValueError("output_length must be at least 1.")
    return output_length


def fit_bounds(
    origin: int,
    *,
    input_length: Optional[int] = None,
    history_mode: str = AUTO,
) -> tuple[int, int]:
    """Return [start, end) history bounds visible at a forecast origin.

    ``origin`` is the number of observations available before forecasting.
    """
    origin = int(origin)
    if origin < 1:
        raise ValueError("origin must be at least 1.")

    mode = resolve_history_mode(input_length, history_mode)

    if mode == EXPANDING:
        return 0, origin

    assert input_length is not None
    L = int(input_length)
    if origin < L:
        raise ValueError(
            f"origin={origin} provides only {origin} observations, "
            f"but sliding input_length={L} is required."
        )
    return origin - L, origin


def slice_fit_history(
    values: Sequence[T] | np.ndarray,
    *,
    origin: Optional[int] = None,
    input_length: Optional[int] = None,
    history_mode: str = AUTO,
) -> np.ndarray:
    """Extract the observations that a model is allowed to fit on."""
    arr = np.asarray(values)
    if origin is None:
        origin = len(arr)
    start, end = fit_bounds(
        int(origin),
        input_length=input_length,
        history_mode=history_mode,
    )
    return arr[start:end]


def effective_input_length(
    n_available: int,
    *,
    input_length: Optional[int] = None,
    history_mode: str = AUTO,
) -> int:
    """Number of observations used by the final fit."""
    start, end = fit_bounds(
        int(n_available),
        input_length=input_length,
        history_mode=history_mode,
    )
    return int(end - start)


def filter_validation_origins(
    origins: Sequence[int],
    *,
    input_length: Optional[int] = None,
    history_mode: str = AUTO,
) -> list[int]:
    """Keep origins that can satisfy the configured history window."""
    mode = resolve_history_mode(input_length, history_mode)
    values = [int(x) for x in origins]

    if mode == EXPANDING:
        return values

    assert input_length is not None
    L = int(input_length)
    return [o for o in values if o >= L]


@dataclass(frozen=True)
class ForecastWindowConfig:
    """Explicit Rel-ESE input/output configuration."""

    input_length: Optional[int] = None
    output_length: int = 1
    history_mode: str = AUTO

    def __post_init__(self) -> None:
        if self.input_length is not None and int(self.input_length) < 1:
            raise ValueError("input_length must be None or a positive integer.")
        validate_output_length(self.output_length)
        resolve_history_mode(self.input_length, self.history_mode)

    @property
    def horizon(self) -> int:
        """Backward-compatible name used by the existing ESE code."""
        return int(self.output_length)

    @property
    def resolved_history_mode(self) -> str:
        return resolve_history_mode(self.input_length, self.history_mode)

    def bounds(self, origin: int) -> tuple[int, int]:
        return fit_bounds(
            origin,
            input_length=self.input_length,
            history_mode=self.history_mode,
        )

    def slice(self, values: Sequence[T] | np.ndarray, origin: Optional[int] = None) -> np.ndarray:
        return slice_fit_history(
            values,
            origin=origin,
            input_length=self.input_length,
            history_mode=self.history_mode,
        )

    def summary(self, n_available: Optional[int] = None) -> dict:
        result = {
            "input_length": self.input_length,
            "output_length": int(self.output_length),
            "history_mode": self.resolved_history_mode,
        }
        if n_available is not None:
            result["effective_input_length"] = effective_input_length(
                int(n_available),
                input_length=self.input_length,
                history_mode=self.history_mode,
            )
        return result
