"""
pairwise_relation_v8.py
=======================

v7 benchmark extension
----------------------
RelationResult now also exposes ``forecast_path`` for steps 1..h.  Model
selection remains horizon-matched at h; the selected model is then refitted on
the final input window and its full path is retained for sequence benchmarks.

Pairwise relation estimation for history-driven equilibrium state estimation.

Main mapping
------------
    (y_i, y_j)
        -> candidate statistical relation families
        -> horizon-matched rolling-origin validation
        -> shared candidate evaluations {L_k, r_k, w_k}
        -> hard statistical selection OR soft neural-teacher supervision
        -> r_{ij,t+h|t}

The canonical output is a transformed pairwise difference

    r_{ij,t+h|t} ~= T(y_i) - T(y_j).

The relation geometry is delegated to ``relation_representation_v2.py``.
Supported statistical modes include ``log_ratio``, ``asinh``, ``signed_log``,
and ``additive``; all reduce to the same potential-difference relation.  All default
relation models forecast the same selected relation coordinate.

Selection schemes
-----------------
Two relation-selection schemes are supported.

``selection_scheme="history"`` (default)
    Candidate relations are selected from historical one-step rolling-origin
    performance.  Selection and reliability therefore depend on the observed
    relation history, not on the requested forecast horizon.  The selected
    relation model is then refitted once and can forecast any requested path
    length h.

``selection_scheme="horizon"``
    Reproduces the horizon-matched alternative: for a requested horizon h,
    candidates are ranked by historical h-step-ahead error.

For either scheme, fixed-length sliding input windows are supported so every
rolling origin can use the same lookback length L.  The legacy single-path
holdout remains available through `validation_method="path"` for
reproducibility/comparison.

Default statistical relation families
-------------------------------------
1. StableRelation
   Robust constant relation using the median of the historical log-ratio.

2. TrendRelation
   Local-linear-trend state-space model for a smoothly evolving relation.

3. DynamicRelation
   Automatically selected ARIMA(p,d,q) model for pairwise relation dynamics.

4. EquilibriumRelation
   Bivariate long-run/short-run model.  VECM is used when cointegration is
   detected; otherwise VAR is used.

5. SmoothNonlinearRelation
   Spline + ridge regression for a smooth nonlinear relation trajectory.

6. RegimeRelation
   Two-regime SETAR(1) model for threshold/regime-dependent dynamics.

Null / rejection mechanism
--------------------------
A naive persistence forecast is used as a null baseline.  If no candidate
relation improves on the null baseline by the requested amount, the pair can
be marked invalid and assigned weight zero.

Extension interface
-------------------
Additional user-defined models can be supplied as factories via
`extra_relations`, but they are ignored unless `enable_extra=True`.
Each custom model must subclass `BaseRelationModel` and implement:

    fit(y_i, y_j, epsilon)
    predict_path(horizon)

This allows statistical, machine-learning, foundation-model, or LLM-based
relation estimators to participate in the same validation and selection
procedure without modifying the core ESE pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np

from relation_representation import (
    LOG_RATIO,
    normalize_representation_name,
    relation_series as _representation_relation_series,
    transform_series as _representation_transform_series,
    validate_observations as _validate_representation_observations,
)

from forecast_window import (
    AUTO as AUTO_HISTORY,
    filter_validation_origins,
    resolve_history_mode,
    slice_fit_history,
)


HISTORY_SELECTION = "history"
HORIZON_SELECTION = "horizon"
_SELECTION_SCHEMES = {HISTORY_SELECTION, HORIZON_SELECTION}


def _normalize_selection_scheme(selection_scheme: str) -> str:
    value = str(selection_scheme).strip().lower()
    aliases = {
        "history_based": HISTORY_SELECTION,
        "history-based": HISTORY_SELECTION,
        "fixed": HISTORY_SELECTION,
        "horizon_matched": HORIZON_SELECTION,
        "horizon-matched": HORIZON_SELECTION,
        "matched": HORIZON_SELECTION,
    }
    value = aliases.get(value, value)
    if value not in _SELECTION_SCHEMES:
        raise ValueError(
            f"Unknown selection_scheme={selection_scheme!r}. "
            f"Expected one of {sorted(_SELECTION_SCHEMES)}."
        )
    return value


def _selection_horizon(requested_horizon: int, selection_scheme: str) -> int:
    requested_horizon = int(requested_horizon)
    if requested_horizon < 1:
        raise ValueError("horizon must be at least 1.")
    scheme = _normalize_selection_scheme(selection_scheme)
    return 1 if scheme == HISTORY_SELECTION else requested_horizon

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import VECM, select_order as vecm_select_order


# ============================================================
# Types and result containers
# ============================================================

RelationFactory = Callable[[], "BaseRelationModel"]


@dataclass
class CandidateScore:
    """Backward-compatible validation information for one candidate model."""

    family: str
    model: str
    validation_loss: float
    is_custom: bool = False
    error: Optional[str] = None
    key: Optional[str] = None
    n_validation_origins: int = 0
    n_successful_origins: int = 0
    validation_success_rate: float = 0.0


@dataclass
class CandidateEvaluation:
    """Complete reusable evaluation of one relation candidate.

    A candidate is validated using the configured temporal protocol.  The v3
    default is horizon-matched rolling-origin validation. Depending on
    `refit_mode`, it may then be refitted once on the full history to obtain the
    horizon-specific relation target `r`. This object is the shared statistical
    output consumed by both hard model selection and neural distillation.
    """

    key: str
    family: str
    model: str
    validation_loss: float
    normalized_rmse: float
    reliability: float

    r: float = np.nan
    parameters: Dict[str, Any] = field(default_factory=dict)

    is_custom: bool = False
    validation_valid: bool = False
    validation_error: Optional[str] = None

    full_refit_attempted: bool = False
    full_refit_valid: bool = False
    refit_error: Optional[str] = None

    n_validation_origins: int = 0
    n_successful_origins: int = 0
    validation_success_rate: float = 0.0

    @property
    def is_valid(self) -> bool:
        """Validity at the deepest stage that has actually been attempted."""
        if not self.validation_valid:
            return False
        if self.full_refit_attempted:
            return bool(self.full_refit_valid)
        return True

    @property
    def error(self) -> Optional[str]:
        return self.refit_error or self.validation_error

    def to_score(self) -> CandidateScore:
        return CandidateScore(
            family=self.family,
            model=self.model,
            validation_loss=float(self.validation_loss),
            is_custom=bool(self.is_custom),
            error=self.error,
            key=self.key,
            n_validation_origins=int(self.n_validation_origins),
            n_successful_origins=int(self.n_successful_origins),
            validation_success_rate=float(self.validation_success_rate),
        )


@dataclass
class PairRelationEvaluations:
    """Reusable statistical evidence for all relation candidates of one pair."""

    candidates: List[CandidateEvaluation]
    null_loss: float
    relation_scale: float
    horizon: int
    validation_ratio: float
    epsilon: float
    weight_eta: float
    selection_scheme: str = HISTORY_SELECTION
    selection_horizon: int = 1
    validation_method: str = "rolling_origin"
    validation_stride: int = 1
    max_validation_origins: Optional[int] = 10
    min_validation_success_rate: float = 0.8
    validation_origins: List[int] = field(default_factory=list)
    refit_mode: str = "none"
    input_length: Optional[int] = None
    history_mode: str = "expanding"

    @property
    def candidate_scores(self) -> List[CandidateScore]:
        return [candidate.to_score() for candidate in self.candidates]

    @property
    def candidate_keys(self) -> List[str]:
        return [candidate.key for candidate in self.candidates]

    @property
    def best_validation_candidate(self) -> Optional[CandidateEvaluation]:
        valid = [
            candidate
            for candidate in self.candidates
            if candidate.validation_valid and np.isfinite(candidate.validation_loss)
        ]
        if not valid:
            return None
        return min(valid, key=lambda candidate: candidate.validation_loss)

    @property
    def best_refit_candidate(self) -> Optional[CandidateEvaluation]:
        valid = [
            candidate
            for candidate in self.candidates
            if candidate.full_refit_valid
            and np.isfinite(candidate.validation_loss)
            and np.isfinite(candidate.r)
        ]
        if not valid:
            return None
        return min(valid, key=lambda candidate: candidate.validation_loss)

    def get(self, key: str) -> CandidateEvaluation:
        for candidate in self.candidates:
            if candidate.key == key:
                return candidate
        raise KeyError(key)


@dataclass
class RelationResult:
    """Final hard-selected result for one pair of systems.

    The public interface is intentionally preserved from v1 so downstream code
    (`relation_matrix.py`, demos, experiments) does not need to change.
    """

    r: float
    selected_family: str
    selected_model: str
    validation_loss: float
    normalized_rmse: float
    weight: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    is_custom: bool = False
    null_loss: float = np.nan
    improvement_over_null: float = np.nan
    candidate_scores: List[CandidateScore] = field(default_factory=list)
    # Full 1..h relation forecast path from the finally selected/refitted model.
    # v6 exposed only the endpoint r=forecast_path[-1].  Keeping the full path
    # enables standard multivariate benchmark evaluation without refitting each
    # pair separately for every forecast step.  Invalid/null relations store NaNs.
    forecast_path: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))


# ============================================================
# Shared utilities
# ============================================================


def _as_1d_float(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    return x


def _validate_pair(
    y_i: np.ndarray,
    y_j: np.ndarray,
    *,
    representation: str = LOG_RATIO,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    y_i = _as_1d_float(y_i, "y_i")
    y_j = _as_1d_float(y_j, "y_j")

    if len(y_i) != len(y_j):
        raise ValueError("y_i and y_j must contain the same number of observations.")

    if len(y_i) < 12:
        raise ValueError("At least 12 observations are required for relation selection.")

    representation = normalize_representation_name(representation)
    _validate_representation_observations(
        y_i, representation, epsilon=epsilon, name="y_i"
    )
    _validate_representation_observations(
        y_j, representation, epsilon=epsilon, name="y_j"
    )

    return y_i, y_j


def _transformed_series(
    y: np.ndarray,
    epsilon: float,
    representation: str = LOG_RATIO,
) -> np.ndarray:
    return _representation_transform_series(
        y, representation, epsilon=epsilon
    )


def _relation_series(
    y_i: np.ndarray,
    y_j: np.ndarray,
    epsilon: float,
    representation: str = LOG_RATIO,
) -> np.ndarray:
    return _representation_relation_series(
        y_i, y_j, representation, epsilon=epsilon
    )


# Backward-compatible internal aliases used by external custom code.  Built-in
# v4 models no longer call these directly.
def _log_series(y: np.ndarray, epsilon: float) -> np.ndarray:
    return _transformed_series(y, epsilon, LOG_RATIO)


def _log_ratio(y_i: np.ndarray, y_j: np.ndarray, epsilon: float) -> np.ndarray:
    return _relation_series(y_i, y_j, epsilon, LOG_RATIO)


def _mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(
            f"actual and predicted must have the same shape; "
            f"got {actual.shape} and {predicted.shape}."
        )
    return float(np.mean((actual - predicted) ** 2))


def _relation_scale(z: np.ndarray, epsilon: float) -> float:
    """Robust-ish scale used only to report a normalized validation RMSE."""
    z = np.asarray(z, dtype=float)
    std = float(np.std(z))
    mad = float(np.median(np.abs(z - np.median(z)))) * 1.4826
    return max(std, mad, 1e-4, epsilon)


# ============================================================
# Base interface
# ============================================================


class BaseRelationModel(ABC):
    """
    Common interface for every pairwise relation estimator.

    A model may use the raw pair (y_i, y_j), their logarithms, their log-ratio,
    or any internal representation.  Regardless of its internal mechanism,
    `predict_path(horizon)` MUST return predictions in the canonical relation
    space

        r = log(gamma_i^* / gamma_j^*)

    for horizons 1, ..., h.
    """

    family: str = "base"
    name: str = "base"
    min_observations: int = 4

    def __init__(self) -> None:
        self._is_fitted = False
        self.representation = LOG_RATIO

    def set_representation(self, representation: str) -> "BaseRelationModel":
        """Select the shared relation coordinate used by this candidate."""
        self.representation = normalize_representation_name(representation)
        return self

    def _transform_series(self, y: np.ndarray, epsilon: float) -> np.ndarray:
        return _transformed_series(y, epsilon, self.representation)

    def _relation_series(
        self, y_i: np.ndarray, y_j: np.ndarray, epsilon: float
    ) -> np.ndarray:
        return _relation_series(y_i, y_j, epsilon, self.representation)

    @abstractmethod
    def fit(self, y_i: np.ndarray, y_j: np.ndarray, epsilon: float) -> "BaseRelationModel":
        raise NotImplementedError

    @abstractmethod
    def predict_path(self, horizon: int) -> np.ndarray:
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        return {}

    @property
    def fitted_model_name(self) -> str:
        return self.name

    def _check_horizon(self, horizon: int) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before prediction.")
        if horizon < 1:
            raise ValueError("horizon must be at least 1.")


# ============================================================
# 1. Stable relation
# ============================================================


class StableRelation(BaseRelationModel):
    """Robust constant equilibrium relation based on median log-ratio."""

    family = "stable"
    name = "robust_median"
    min_observations = 4

    def fit(self, y_i: np.ndarray, y_j: np.ndarray, epsilon: float) -> "StableRelation":
        z = self._relation_series(y_i, y_j, epsilon)
        self.location_ = float(np.median(z))
        self.dispersion_ = float(np.median(np.abs(z - self.location_)))
        self._is_fitted = True
        return self

    def predict_path(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        return np.full(horizon, self.location_, dtype=float)

    def get_params(self) -> Dict[str, Any]:
        return {
            "location": self.location_,
            "median_absolute_deviation": self.dispersion_,
        }


# ============================================================
# 2. Smoothly evolving trend relation
# ============================================================


class TrendRelation(BaseRelationModel):
    """Local-linear-trend structural time-series model for z_ij,t."""

    family = "trend"
    name = "local_linear_state_space"
    min_observations = 8

    def fit(self, y_i: np.ndarray, y_j: np.ndarray, epsilon: float) -> "TrendRelation":
        z = self._relation_series(y_i, y_j, epsilon)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = UnobservedComponents(z, level="local linear trend")
            self.result_ = model.fit(disp=False)

        self._is_fitted = True
        return self

    def predict_path(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        pred = self.result_.get_forecast(steps=horizon).predicted_mean
        return np.asarray(pred, dtype=float)

    def get_params(self) -> Dict[str, Any]:
        return {
            "aic": float(self.result_.aic),
            "bic": float(self.result_.bic),
        }


# ============================================================
# 3. Dynamic relation: auto-ARIMA
# ============================================================


class DynamicRelation(BaseRelationModel):
    """
    Pairwise dynamic relation using a small automatically selected ARIMA grid.

    The family is intentionally broad: AR, ARMA, differenced AR, and general
    low-order ARIMA models are all represented by the same family.
    """

    family = "dynamic"
    name = "auto_arima"
    min_observations = 12

    def __init__(
        self,
        max_p: int = 2,
        max_d: int = 1,
        max_q: int = 2,
    ) -> None:
        super().__init__()
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q

    def fit(self, y_i: np.ndarray, y_j: np.ndarray, epsilon: float) -> "DynamicRelation":
        z = self._relation_series(y_i, y_j, epsilon)

        best_result = None
        best_order = None
        best_aic = np.inf

        for d in range(self.max_d + 1):
            for p in range(self.max_p + 1):
                for q in range(self.max_q + 1):
                    order = (p, d, q)

                    # Keep a pure white-noise/level model as a valid member of
                    # the family, but ARIMA selection is AIC-based.
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            result = ARIMA(
                                z,
                                order=order,
                                trend=None,
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            ).fit()

                        if np.isfinite(result.aic) and result.aic < best_aic:
                            best_result = result
                            best_order = order
                            best_aic = float(result.aic)
                    except Exception:
                        continue

        if best_result is None:
            raise RuntimeError("No ARIMA specification could be fitted.")

        self.result_ = best_result
        self.order_ = best_order
        self._is_fitted = True
        return self

    def predict_path(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        pred = self.result_.get_forecast(steps=horizon).predicted_mean
        return np.asarray(pred, dtype=float)

    @property
    def fitted_model_name(self) -> str:
        return f"ARIMA{self.order_}"

    def get_params(self) -> Dict[str, Any]:
        return {
            "order": self.order_,
            "aic": float(self.result_.aic),
            "bic": float(self.result_.bic),
        }


# ============================================================
# 4. Long-run equilibrium relation: VECM or VAR
# ============================================================


class EquilibriumRelation(BaseRelationModel):
    """
    Bivariate equilibrium relation on log-level series.

    If Engle-Granger cointegration is detected, a VECM with cointegration rank
    one is fitted.  Otherwise the family falls back to a bivariate VAR.  Both
    models forecast log(y_i) and log(y_j), and the canonical relation forecast
    is their difference.
    """

    family = "equilibrium"
    name = "vecm_or_var"
    min_observations = 20

    def __init__(
        self,
        coint_alpha: float = 0.05,
        max_lags: int = 3,
    ) -> None:
        super().__init__()
        self.coint_alpha = coint_alpha
        self.max_lags = max_lags

    def fit(self, y_i: np.ndarray, y_j: np.ndarray, epsilon: float) -> "EquilibriumRelation":
        x_i = self._transform_series(y_i, epsilon)
        x_j = self._transform_series(y_j, epsilon)
        X = np.column_stack([x_i, x_j])
        self.X_ = X

        # Cointegration test.  Failure of the test itself is treated as no
        # reliable evidence of cointegration, after which VAR is attempted.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, pvalue, _ = coint(x_i, x_j, trend="c", autolag="aic")
            self.coint_pvalue_ = float(pvalue)
        except Exception:
            self.coint_pvalue_ = np.nan

        use_vecm = np.isfinite(self.coint_pvalue_) and self.coint_pvalue_ < self.coint_alpha

        if use_vecm:
            self._fit_vecm(X)
        else:
            self._fit_var(X)

        self._is_fitted = True
        return self

    def _fit_vecm(self, X: np.ndarray) -> None:
        maxlags = min(self.max_lags, max(0, len(X) // 8))

        k_ar_diff = 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                selected = vecm_select_order(
                    X,
                    maxlags=maxlags,
                    deterministic="co",
                ).selected_orders.get("aic")
            if selected is not None:
                k_ar_diff = int(selected)
        except Exception:
            pass

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = VECM(
                X,
                k_ar_diff=max(0, k_ar_diff),
                coint_rank=1,
                deterministic="co",
            ).fit()

        self.kind_ = "VECM"
        self.result_ = result
        self.k_ar_diff_ = max(0, k_ar_diff)

    def _fit_var(self, X: np.ndarray) -> None:
        maxlags = min(self.max_lags, max(1, len(X) // 8))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VAR(X)
            result = model.fit(maxlags=maxlags, ic="aic", trend="c")

        # Some samples select VAR(0), which has no dynamic forecast recursion.
        # Use VAR(1) as the minimal dynamic representation in that case.
        if result.k_ar == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = VAR(X).fit(1, trend="c")

        self.kind_ = "VAR"
        self.result_ = result
        self.k_ar_ = int(result.k_ar)

    def predict_path(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)

        if self.kind_ == "VECM":
            forecast = np.asarray(self.result_.predict(steps=horizon), dtype=float)
        else:
            forecast = np.asarray(
                self.result_.forecast(
                    self.X_[-self.result_.k_ar :],
                    steps=horizon,
                ),
                dtype=float,
            )

        # Canonical pairwise relation in log-ratio space.
        return forecast[:, 0] - forecast[:, 1]

    @property
    def fitted_model_name(self) -> str:
        return self.kind_

    def get_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "cointegration_pvalue": self.coint_pvalue_,
            "cointegration_alpha": self.coint_alpha,
            "kind": self.kind_,
        }

        if self.kind_ == "VECM":
            params["k_ar_diff"] = self.k_ar_diff_
            params["coint_rank"] = 1
        else:
            params["k_ar"] = self.k_ar_
            params["aic"] = float(self.result_.aic)
            params["bic"] = float(self.result_.bic)

        return params


# ============================================================
# 5. Smooth nonlinear relation: spline trend
# ============================================================


class SmoothNonlinearRelation(BaseRelationModel):
    """
    Smooth nonlinear pairwise relation trajectory.

    A cubic B-spline basis is fitted to time and regularized with ridge
    regression.  Linear basis extrapolation is used outside the historical
    window to avoid unrestricted high-order polynomial explosion.
    """

    family = "smooth_nonlinear"
    name = "spline_ridge"
    min_observations = 10

    def __init__(
        self,
        n_knots: int = 5,
        degree: int = 3,
        ridge_alpha: float = 1e-3,
    ) -> None:
        super().__init__()
        self.n_knots = n_knots
        self.degree = degree
        self.ridge_alpha = ridge_alpha

    def fit(self, y_i: np.ndarray, y_j: np.ndarray, epsilon: float) -> "SmoothNonlinearRelation":
        z = self._relation_series(y_i, y_j, epsilon)
        n = len(z)
        t = np.arange(n, dtype=float).reshape(-1, 1)

        n_knots = max(3, min(self.n_knots, max(3, n // 3)))

        self.model_ = Pipeline(
            [
                (
                    "spline",
                    SplineTransformer(
                        n_knots=n_knots,
                        degree=self.degree,
                        extrapolation="linear",
                        include_bias=False,
                    ),
                ),
                ("ridge", Ridge(alpha=self.ridge_alpha)),
            ]
        )

        self.model_.fit(t, z)
        self.n_obs_ = n
        self.used_n_knots_ = n_knots
        self._is_fitted = True
        return self

    def predict_path(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        future_t = np.arange(
            self.n_obs_,
            self.n_obs_ + horizon,
            dtype=float,
        ).reshape(-1, 1)
        return np.asarray(self.model_.predict(future_t), dtype=float)

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_knots": self.used_n_knots_,
            "degree": self.degree,
            "ridge_alpha": self.ridge_alpha,
        }


# ============================================================
# 6. Regime relation: SETAR(1)
# ============================================================


class RegimeRelation(BaseRelationModel):
    """
    Two-regime self-exciting threshold autoregression SETAR(1).

        z_t = a_1 + b_1 z_{t-1} + e_t,  z_{t-1} <= c
        z_t = a_2 + b_2 z_{t-1} + e_t,  z_{t-1} >  c

    The threshold is selected from empirical lagged-value quantiles by
    minimizing in-sample SSE on the fitting window.
    """

    family = "regime"
    name = "SETAR(1)"
    min_observations = 20

    def __init__(
        self,
        threshold_quantiles: Sequence[float] = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
        min_regime_size: int = 4,
    ) -> None:
        super().__init__()
        self.threshold_quantiles = tuple(threshold_quantiles)
        self.min_regime_size = min_regime_size

    @staticmethod
    def _fit_ar_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        X = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(beta[0]), float(beta[1])

    def fit(self, y_i: np.ndarray, y_j: np.ndarray, epsilon: float) -> "RegimeRelation":
        z = self._relation_series(y_i, y_j, epsilon)
        x = z[:-1]
        y = z[1:]

        thresholds = np.unique(np.quantile(x, self.threshold_quantiles))

        best = None
        best_sse = np.inf

        for threshold in thresholds:
            low = x <= threshold
            high = ~low

            if low.sum() < self.min_regime_size or high.sum() < self.min_regime_size:
                continue

            a1, b1 = self._fit_ar_line(x[low], y[low])
            a2, b2 = self._fit_ar_line(x[high], y[high])

            pred = np.empty_like(y)
            pred[low] = a1 + b1 * x[low]
            pred[high] = a2 + b2 * x[high]
            sse = float(np.sum((y - pred) ** 2))

            if sse < best_sse:
                best_sse = sse
                best = (float(threshold), a1, b1, a2, b2)

        if best is None:
            raise RuntimeError("SETAR could not form two sufficiently populated regimes.")

        self.threshold_, self.a_low_, self.b_low_, self.a_high_, self.b_high_ = best
        self.last_z_ = float(z[-1])
        self.training_sse_ = best_sse
        self._is_fitted = True
        return self

    def predict_path(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)

        out = np.empty(horizon, dtype=float)
        value = self.last_z_

        for h in range(horizon):
            if value <= self.threshold_:
                value = self.a_low_ + self.b_low_ * value
            else:
                value = self.a_high_ + self.b_high_ * value
            out[h] = value

        return out

    def get_params(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold_,
            "low_intercept": self.a_low_,
            "low_ar": self.b_low_,
            "high_intercept": self.a_high_,
            "high_ar": self.b_high_,
            "training_sse": self.training_sse_,
        }


# ============================================================
# Default relation registry
# ============================================================


DEFAULT_RELATION_FACTORIES: Tuple[RelationFactory, ...] = (
    StableRelation,
    TrendRelation,
    DynamicRelation,
    EquilibriumRelation,
    SmoothNonlinearRelation,
    RegimeRelation,
)


def get_default_relation_names() -> List[str]:
    """Return the default relation-family names without fitting any models."""
    names = []
    for factory in DEFAULT_RELATION_FACTORIES:
        model = factory()
        names.append(model.family)
    return names


# ============================================================
# Shared statistical engine
# ============================================================


@dataclass
class _CandidateRecord:
    key: str
    family: str
    base_name: str
    factory: RelationFactory
    is_custom: bool


def _candidate_factory_records(
    enable_extra: bool,
    extra_relations: Optional[Sequence[RelationFactory]],
) -> List[_CandidateRecord]:
    """Build deterministic candidate records without fitting any model."""
    raw: List[Tuple[RelationFactory, bool, BaseRelationModel]] = []

    for factory in DEFAULT_RELATION_FACTORIES:
        model = factory()
        if not isinstance(model, BaseRelationModel):
            raise TypeError("Every default relation factory must create BaseRelationModel.")
        raw.append((factory, False, model))

    if enable_extra:
        if not extra_relations:
            raise ValueError(
                "enable_extra=True but no extra_relations were supplied. "
                "Pass model factories/classes through extra_relations."
            )
        for factory in extra_relations:
            model = factory()
            if not isinstance(model, BaseRelationModel):
                raise TypeError(
                    "Every custom relation factory must return an instance of "
                    "BaseRelationModel."
                )
            raw.append((factory, True, model))

    family_counts: Dict[str, int] = {}
    for _, _, model in raw:
        family_counts[model.family] = family_counts.get(model.family, 0) + 1

    used_keys: Dict[str, int] = {}
    records: List[_CandidateRecord] = []

    for factory, is_custom, model in raw:
        if family_counts[model.family] == 1:
            base_key = str(model.family)
        else:
            base_key = f"{model.family}:{model.name}"

        occurrence = used_keys.get(base_key, 0)
        used_keys[base_key] = occurrence + 1
        key = base_key if occurrence == 0 else f"{base_key}#{occurrence + 1}"

        records.append(
            _CandidateRecord(
                key=key,
                family=str(model.family),
                base_name=str(model.name),
                factory=factory,
                is_custom=bool(is_custom),
            )
        )

    return records


def get_relation_candidate_keys(
    *,
    enable_extra: bool = False,
    extra_relations: Optional[Sequence[RelationFactory]] = None,
) -> List[str]:
    """Return stable neural/teacher output keys without fitting any models."""
    return [
        record.key
        for record in _candidate_factory_records(enable_extra, extra_relations)
    ]


def _build_validation_origins(
    n_observations: int,
    horizon: int,
    validation_ratio: float,
    validation_stride: int,
    max_validation_origins: Optional[int],
    *,
    input_length: Optional[int] = None,
    history_mode: str = AUTO_HISTORY,
) -> Tuple[List[int], int]:
    """Build recent horizon-matched validation origins.

    With ``history_mode='sliding'``, only origins that have a complete
    ``input_length`` history window are retained.  With expanding history the
    behavior is identical to v5.
    """
    validation_size = max(horizon, max(2, int(round(n_observations * validation_ratio))))
    validation_start = n_observations - validation_size
    last_origin = n_observations - horizon

    if validation_start < 1 or last_origin < validation_start:
        raise ValueError(
            "Not enough observations to create horizon-matched validation origins."
        )

    origins = list(range(validation_start, last_origin + 1, validation_stride))
    origins = filter_validation_origins(
        origins,
        input_length=input_length,
        history_mode=history_mode,
    )

    # Prefer the most recent origins when a computational cap is requested.
    if max_validation_origins is not None and len(origins) > max_validation_origins:
        origins = origins[-max_validation_origins:]

    if not origins:
        raise ValueError(
            "No legal validation origins were produced for the configured "
            f"input_length={input_length}, history_mode={history_mode!r}."
        )

    return origins, validation_start


def _validate_candidate_path(
    model: BaseRelationModel,
    y_i_train: np.ndarray,
    y_j_train: np.ndarray,
    z_val: np.ndarray,
    epsilon: float,
    representation: str,
    *,
    input_length: Optional[int] = None,
    history_mode: str = AUTO_HISTORY,
) -> Tuple[float, Optional[str], str, int, int, float]:
    """Legacy v2 single-fit validation path, kept for reproducibility."""
    y_i_train = slice_fit_history(
        y_i_train,
        input_length=input_length,
        history_mode=history_mode,
    )
    y_j_train = slice_fit_history(
        y_j_train,
        input_length=input_length,
        history_mode=history_mode,
    )

    if len(y_i_train) < model.min_observations:
        return (
            np.inf,
            f"requires at least {model.min_observations} training observations",
            model.name,
            1,
            0,
            0.0,
        )

    try:
        model.set_representation(representation)
        model.fit(y_i_train, y_j_train, epsilon)
        pred = np.asarray(model.predict_path(len(z_val)), dtype=float)

        if pred.shape != z_val.shape or not np.all(np.isfinite(pred)):
            raise RuntimeError("candidate produced an invalid validation forecast")

        loss = _mse(z_val, pred)
        return float(loss), None, model.fitted_model_name, 1, 1, 1.0

    except Exception as exc:
        return np.inf, str(exc), model.name, 1, 0, 0.0


def _validate_candidate_rolling(
    record: _CandidateRecord,
    y_i: np.ndarray,
    y_j: np.ndarray,
    z: np.ndarray,
    *,
    origins: Sequence[int],
    horizon: int,
    epsilon: float,
    representation: str,
    min_validation_success_rate: float,
    input_length: Optional[int] = None,
    history_mode: str = AUTO_HISTORY,
) -> Tuple[float, Optional[str], str, int, int, float]:
    """Evaluate exactly the requested h-step forecast at rolling origins."""
    squared_errors: List[float] = []
    fitted_names: List[str] = []
    failures: List[str] = []

    for origin in origins:
        model = record.factory()

        try:
            y_i_fit = slice_fit_history(
                y_i,
                origin=origin,
                input_length=input_length,
                history_mode=history_mode,
            )
            y_j_fit = slice_fit_history(
                y_j,
                origin=origin,
                input_length=input_length,
                history_mode=history_mode,
            )
        except ValueError as exc:
            failures.append(f"origin {origin}: {exc}")
            continue

        if len(y_i_fit) < model.min_observations:
            failures.append(
                f"origin {origin}: fit window has {len(y_i_fit)} observations; "
                f"requires at least {model.min_observations}"
            )
            continue

        target_index = int(origin + horizon - 1)

        try:
            model.set_representation(representation)
            model.fit(y_i_fit, y_j_fit, epsilon)
            pred_path = np.asarray(model.predict_path(horizon), dtype=float)

            if pred_path.shape != (horizon,) or not np.all(np.isfinite(pred_path)):
                raise RuntimeError("candidate produced an invalid h-step forecast")

            prediction = float(pred_path[-1])
            target = float(z[target_index])
            squared_errors.append((prediction - target) ** 2)
            fitted_names.append(str(model.fitted_model_name))

        except Exception as exc:
            failures.append(f"origin {origin}: {exc}")

    n_total = int(len(origins))
    n_success = int(len(squared_errors))
    success_rate = float(n_success / n_total) if n_total else 0.0

    if n_success == 0 or success_rate < min_validation_success_rate:
        message = (
            f"rolling validation success rate {n_success}/{n_total}="
            f"{success_rate:.3f} is below required "
            f"{min_validation_success_rate:.3f}"
        )
        if failures:
            message += f"; last failure: {failures[-1]}"
        return np.inf, message, record.base_name, n_total, n_success, success_rate

    loss = float(np.mean(np.asarray(squared_errors, dtype=float)))
    fitted_name = fitted_names[-1] if fitted_names else record.base_name

    # Partial failures are represented by the success-count diagnostics.  A
    # candidate that passes the configured threshold is still validation-valid.
    return loss, None, fitted_name, n_total, n_success, success_rate


def _validate_engine_arguments(
    horizon: int,
    validation_ratio: float,
    epsilon: float,
    weight_eta: float,
    validation_method: str,
    validation_stride: int,
    max_validation_origins: Optional[int],
    min_validation_success_rate: float,
    input_length: Optional[int] = None,
    history_mode: str = AUTO_HISTORY,
) -> None:
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if not 0.05 <= validation_ratio < 0.5:
        raise ValueError("validation_ratio must lie in [0.05, 0.5).")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if weight_eta < 0:
        raise ValueError("weight_eta must be non-negative.")
    if validation_method not in {"rolling_origin", "path"}:
        raise ValueError("validation_method must be 'rolling_origin' or 'path'.")
    if validation_stride < 1:
        raise ValueError("validation_stride must be at least 1.")
    if max_validation_origins is not None and max_validation_origins < 1:
        raise ValueError("max_validation_origins must be None or at least 1.")
    if not 0 < min_validation_success_rate <= 1:
        raise ValueError("min_validation_success_rate must lie in (0, 1].")
    resolve_history_mode(input_length, history_mode)


def _score_pair_relations_with_records(
    y_i: np.ndarray,
    y_j: np.ndarray,
    *,
    horizon: int,
    validation_ratio: float,
    epsilon: float,
    enable_extra: bool,
    extra_relations: Optional[Sequence[RelationFactory]],
    weight_eta: float,
    selection_scheme: str,
    validation_method: str,
    validation_stride: int,
    max_validation_origins: Optional[int],
    min_validation_success_rate: float,
    representation: str,
    input_length: Optional[int],
    history_mode: str,
) -> Tuple[
    PairRelationEvaluations,
    List[_CandidateRecord],
    np.ndarray,
    np.ndarray,
]:
    """Validation-only pass shared by hard selection and the neural teacher."""
    representation = normalize_representation_name(representation)
    y_i, y_j = _validate_pair(
        y_i, y_j, representation=representation, epsilon=epsilon
    )
    validation_method = str(validation_method).lower()
    selection_scheme = _normalize_selection_scheme(selection_scheme)
    selection_horizon = _selection_horizon(horizon, selection_scheme)
    history_mode = resolve_history_mode(input_length, history_mode)
    if validation_method == "path" and selection_scheme == HORIZON_SELECTION:
        raise ValueError(
            "selection_scheme='horizon' requires validation_method='rolling_origin'."
        )
    _validate_engine_arguments(
        selection_horizon,
        validation_ratio,
        epsilon,
        weight_eta,
        validation_method,
        validation_stride,
        max_validation_origins,
        min_validation_success_rate,
        input_length,
        history_mode,
    )

    n = len(y_i)
    z = _relation_series(y_i, y_j, epsilon, representation)
    records = _candidate_factory_records(enable_extra, extra_relations)
    evaluations: List[CandidateEvaluation] = []

    if validation_method == "rolling_origin":
        origins, validation_start = _build_validation_origins(
            n,
            selection_horizon,
            validation_ratio,
            validation_stride,
            max_validation_origins,
            input_length=input_length,
            history_mode=history_mode,
        )

        null_errors = []
        for origin in origins:
            target_index = origin + selection_horizon - 1
            null_prediction = float(z[origin - 1])
            null_target = float(z[target_index])
            null_errors.append((null_prediction - null_target) ** 2)

        null_loss = float(np.mean(null_errors))
        scale_history = slice_fit_history(
            z,
            origin=validation_start,
            input_length=input_length,
            history_mode=history_mode,
        )
        if len(scale_history) < 2:
            scale_history = slice_fit_history(
                z,
                origin=origins[0],
                input_length=input_length,
                history_mode=history_mode,
            )
        scale = _relation_scale(scale_history, epsilon)

        for record in records:
            loss, error, fitted_name, n_total, n_success, success_rate = (
                _validate_candidate_rolling(
                    record,
                    y_i,
                    y_j,
                    z,
                    origins=origins,
                    horizon=selection_horizon,
                    epsilon=epsilon,
                    representation=representation,
                    min_validation_success_rate=min_validation_success_rate,
                    input_length=input_length,
                    history_mode=history_mode,
                )
            )

            valid = bool(np.isfinite(loss))
            if valid:
                normalized_rmse = float(np.sqrt(max(float(loss), 0.0)) / scale)
                reliability = float(np.exp(-weight_eta * normalized_rmse))
            else:
                normalized_rmse = np.inf
                reliability = 0.0

            evaluations.append(
                CandidateEvaluation(
                    key=record.key,
                    family=record.family,
                    model=str(fitted_name),
                    validation_loss=float(loss),
                    normalized_rmse=float(normalized_rmse),
                    reliability=float(reliability),
                    r=np.nan,
                    parameters={},
                    is_custom=record.is_custom,
                    validation_valid=valid,
                    validation_error=error,
                    full_refit_attempted=False,
                    full_refit_valid=False,
                    refit_error=None,
                    n_validation_origins=n_total,
                    n_successful_origins=n_success,
                    validation_success_rate=success_rate,
                )
            )

    else:  # legacy path validation
        validation_size = max(2, int(round(n * validation_ratio)))
        split = n - validation_size
        y_i_train = y_i[:split]
        y_j_train = y_j[:split]
        z_train = z[:split]
        z_val = z[split:]
        origins = [split]

        null_pred = np.full(len(z_val), z_train[-1], dtype=float)
        null_loss = _mse(z_val, null_pred)
        scale_train = slice_fit_history(
            z_train,
            input_length=input_length,
            history_mode=history_mode,
        )
        scale = _relation_scale(scale_train, epsilon)

        for record in records:
            model = record.factory()
            loss, error, fitted_name, n_total, n_success, success_rate = (
                _validate_candidate_path(
                    model,
                    y_i_train,
                    y_j_train,
                    z_val,
                    epsilon,
                    representation,
                    input_length=input_length,
                    history_mode=history_mode,
                )
            )

            valid = bool(np.isfinite(loss))
            if valid:
                normalized_rmse = float(np.sqrt(max(float(loss), 0.0)) / scale)
                reliability = float(np.exp(-weight_eta * normalized_rmse))
            else:
                normalized_rmse = np.inf
                reliability = 0.0

            evaluations.append(
                CandidateEvaluation(
                    key=record.key,
                    family=record.family,
                    model=str(fitted_name),
                    validation_loss=float(loss),
                    normalized_rmse=float(normalized_rmse),
                    reliability=float(reliability),
                    r=np.nan,
                    parameters={},
                    is_custom=record.is_custom,
                    validation_valid=valid,
                    validation_error=error,
                    full_refit_attempted=False,
                    full_refit_valid=False,
                    refit_error=None,
                    n_validation_origins=n_total,
                    n_successful_origins=n_success,
                    validation_success_rate=success_rate,
                )
            )

    return (
        PairRelationEvaluations(
            candidates=evaluations,
            null_loss=float(null_loss),
            relation_scale=float(scale),
            horizon=int(horizon),
            validation_ratio=float(validation_ratio),
            epsilon=float(epsilon),
            weight_eta=float(weight_eta),
            selection_scheme=selection_scheme,
            selection_horizon=int(selection_horizon),
            validation_method=validation_method,
            validation_stride=int(validation_stride),
            max_validation_origins=max_validation_origins,
            min_validation_success_rate=float(min_validation_success_rate),
            validation_origins=[int(x) for x in origins],
            refit_mode="none",
            input_length=input_length,
            history_mode=history_mode,
        ),
        records,
        y_i,
        y_j,
    )

def score_pair_relations(
    y_i: np.ndarray,
    y_j: np.ndarray,
    horizon: int = 1,
    validation_ratio: float = 0.20,
    epsilon: float = 1e-8,
    *,
    enable_extra: bool = False,
    extra_relations: Optional[Sequence[RelationFactory]] = None,
    weight_eta: float = 1.0,
    selection_scheme: str = HISTORY_SELECTION,
    validation_method: str = "rolling_origin",
    validation_stride: int = 1,
    max_validation_origins: Optional[int] = 10,
    min_validation_success_rate: float = 0.8,
    representation: str = LOG_RATIO,
    input_length: Optional[int] = None,
    history_mode: str = AUTO_HISTORY,
) -> PairRelationEvaluations:
    """Validate all candidates without full-history refitting.

    By default, relation selection is history-based: candidates are ranked by
    one-step rolling-origin error, independently of the requested forecast
    horizon.  Set ``selection_scheme="horizon"`` to use horizon-matched
    candidate selection.
    """
    evaluations, _, _, _ = _score_pair_relations_with_records(
        y_i,
        y_j,
        horizon=horizon,
        validation_ratio=validation_ratio,
        epsilon=epsilon,
        enable_extra=enable_extra,
        extra_relations=extra_relations,
        weight_eta=weight_eta,
        selection_scheme=selection_scheme,
        validation_method=validation_method,
        validation_stride=validation_stride,
        max_validation_origins=max_validation_origins,
        min_validation_success_rate=min_validation_success_rate,
        representation=representation,
        input_length=input_length,
        history_mode=history_mode,
    )
    return evaluations

def _full_refit_candidate(
    record: _CandidateRecord,
    y_i: np.ndarray,
    y_j: np.ndarray,
    *,
    horizon: int,
    epsilon: float,
    representation: str,
    input_length: Optional[int] = None,
    history_mode: str = AUTO_HISTORY,
) -> Tuple[float, str, Dict[str, Any], Optional[str]]:
    """Fit one candidate once on full history and obtain the requested h-step r."""
    try:
        model = record.factory()
        model.set_representation(representation)
        y_i_fit = slice_fit_history(
            y_i,
            input_length=input_length,
            history_mode=history_mode,
        )
        y_j_fit = slice_fit_history(
            y_j,
            input_length=input_length,
            history_mode=history_mode,
        )
        if len(y_i_fit) < model.min_observations:
            raise RuntimeError(
                f"requires at least {model.min_observations} observations in the final fit window"
            )

        model.fit(y_i_fit, y_j_fit, epsilon)
        path = np.asarray(model.predict_path(horizon), dtype=float)

        if path.shape != (horizon,) or not np.all(np.isfinite(path)):
            raise RuntimeError("full-sample model produced an invalid forecast")

        return (
            float(path[-1]),
            str(model.fitted_model_name),
            dict(model.get_params()),
            None,
        )
    except Exception as exc:
        return np.nan, record.base_name, {}, str(exc)


def _refit_evaluation(
    candidate: CandidateEvaluation,
    record: _CandidateRecord,
    y_i: np.ndarray,
    y_j: np.ndarray,
    *,
    horizon: int,
    epsilon: float,
    representation: str,
    input_length: Optional[int] = None,
    history_mode: str = AUTO_HISTORY,
) -> bool:
    candidate.full_refit_attempted = True

    r, fitted_name, parameters, error = _full_refit_candidate(
        record,
        y_i,
        y_j,
        horizon=horizon,
        epsilon=epsilon,
        representation=representation,
        input_length=input_length,
        history_mode=history_mode,
    )

    candidate.model = str(fitted_name)
    candidate.r = float(r)
    candidate.parameters = parameters
    candidate.refit_error = error
    candidate.full_refit_valid = bool(error is None and np.isfinite(r))
    return candidate.full_refit_valid


def evaluate_pair_relations(
    y_i: np.ndarray,
    y_j: np.ndarray,
    horizon: int = 1,
    validation_ratio: float = 0.20,
    epsilon: float = 1e-8,
    *,
    enable_extra: bool = False,
    extra_relations: Optional[Sequence[RelationFactory]] = None,
    weight_eta: float = 1.0,
    refit_mode: str = "all",
    selection_scheme: str = HISTORY_SELECTION,
    validation_method: str = "rolling_origin",
    validation_stride: int = 1,
    max_validation_origins: Optional[int] = 10,
    min_validation_success_rate: float = 0.8,
    representation: str = LOG_RATIO,
    input_length: Optional[int] = None,
    history_mode: str = AUTO_HISTORY,
) -> PairRelationEvaluations:
    """Return reusable evidence for every candidate relation family.

    Parameters
    ----------
    refit_mode : {"none", "best", "all"}
        ``none`` validates all candidates only.

        ``best`` validates all candidates, then refits candidates in validation
        rank order until the best successfully refitted candidate is found.

        ``all`` validates all candidates and refits every validation-valid
        candidate exactly once on the full history.  This mode is intended for
        Statistical Teacher generation because neural expert supervision needs
        every candidate's horizon-specific r_k.

    Notes
    -----
    ``selection_scheme="history"`` (default) ranks relation models by
    one-step historical rolling-origin error, so the selected relation model and
    its reliability do not change merely because a different forecast horizon
    is requested.  ``selection_scheme="horizon"`` retains the previous
    horizon-matched selection behavior.

    All ARIMA/VECM/VAR/SETAR/spline/state-space fitting lives in this module.
    `neural_relation_v3.py` consumes this result and performs no statistical
    model fitting of its own.
    """
    mode = str(refit_mode).lower()
    if mode not in {"none", "best", "all"}:
        raise ValueError("refit_mode must be one of {'none', 'best', 'all'}.")

    evaluations, records, y_i, y_j = _score_pair_relations_with_records(
        y_i,
        y_j,
        horizon=horizon,
        validation_ratio=validation_ratio,
        epsilon=epsilon,
        enable_extra=enable_extra,
        extra_relations=extra_relations,
        weight_eta=weight_eta,
        selection_scheme=selection_scheme,
        validation_method=validation_method,
        validation_stride=validation_stride,
        max_validation_origins=max_validation_origins,
        min_validation_success_rate=min_validation_success_rate,
        representation=representation,
        input_length=input_length,
        history_mode=history_mode,
    )

    evaluations.refit_mode = mode
    if mode == "none":
        return evaluations

    record_by_key = {record.key: record for record in records}
    ranked = sorted(
        (
            candidate
            for candidate in evaluations.candidates
            if candidate.validation_valid and np.isfinite(candidate.validation_loss)
        ),
        key=lambda candidate: candidate.validation_loss,
    )

    if mode == "best":
        for candidate in ranked:
            if _refit_evaluation(
                candidate,
                record_by_key[candidate.key],
                y_i,
                y_j,
                horizon=horizon,
                epsilon=epsilon,
                representation=representation,
                input_length=input_length,
                history_mode=history_mode,
            ):
                break
    else:  # all
        for candidate in ranked:
            _refit_evaluation(
                candidate,
                record_by_key[candidate.key],
                y_i,
                y_j,
                horizon=horizon,
                epsilon=epsilon,
                representation=representation,
                input_length=input_length,
                history_mode=history_mode,
            )

    return evaluations


def _refit_selected_forecast_path(
    record: _CandidateRecord,
    y_i: np.ndarray,
    y_j: np.ndarray,
    *,
    horizon: int,
    epsilon: float,
    representation: str,
    input_length: Optional[int] = None,
    history_mode: str = AUTO_HISTORY,
) -> np.ndarray:
    """Refit one selected relation model and return forecasts for steps 1..h.

    This is intentionally separate from the legacy endpoint-only refit helper so
    downstream code that depends on the v6 endpoint behavior remains unchanged.
    The benchmark runner uses this path to reconstruct all future time steps from
    one horizon-conditioned relation selection.
    """
    model = record.factory()
    model.set_representation(representation)
    y_i_fit = slice_fit_history(
        y_i, input_length=input_length, history_mode=history_mode
    )
    y_j_fit = slice_fit_history(
        y_j, input_length=input_length, history_mode=history_mode
    )
    if len(y_i_fit) < model.min_observations:
        raise RuntimeError(
            f"requires at least {model.min_observations} observations in the final fit window"
        )
    model.fit(y_i_fit, y_j_fit, epsilon)
    path = np.asarray(model.predict_path(horizon), dtype=float)
    if path.shape != (int(horizon),) or not np.all(np.isfinite(path)):
        raise RuntimeError("selected relation model produced an invalid forecast path")
    return path



def _null_gate_passes(
    *,
    candidate_loss: float,
    null_loss: float,
    min_improvement_over_null: float,
    epsilon: float,
) -> Tuple[bool, float]:
    """Return whether one validation candidate survives persistence screening.

    With the default zero margin, the candidate must be strictly better than
    persistence in validation MSE.  For a positive requested margin, equality
    to the requested relative-improvement threshold is accepted.
    """
    candidate_loss = float(candidate_loss)
    null_loss = float(null_loss)
    margin = float(min_improvement_over_null)

    improvement = (null_loss - candidate_loss) / (null_loss + float(epsilon))

    if not (np.isfinite(candidate_loss) and np.isfinite(null_loss)):
        return False, float(improvement)

    if margin <= 0.0:
        passed = bool(candidate_loss < null_loss)
    else:
        passed = bool(improvement >= margin)

    return passed, float(improvement)



def estimate_pair_relation(
    y_i: np.ndarray,
    y_j: np.ndarray,
    horizon: int = 1,
    validation_ratio: float = 0.20,
    epsilon: float = 1e-8,
    *,
    enable_extra: bool = False,
    extra_relations: Optional[Sequence[RelationFactory]] = None,
    allow_null: bool = True,
    min_improvement_over_null: float = 0.0,
    weight_eta: float = 1.0,
    selection_scheme: str = HISTORY_SELECTION,
    validation_method: str = "rolling_origin",
    validation_stride: int = 1,
    max_validation_origins: Optional[int] = 10,
    min_validation_success_rate: float = 0.8,
    representation: str = LOG_RATIO,
    input_length: Optional[int] = None,
    history_mode: str = AUTO_HISTORY,
) -> RelationResult:
    """Hard-select one pairwise equilibrium relation.

    The default history-based scheme selects one relation model from historical
    one-step predictive performance and then uses that selected model to
    forecast the requested path.  Set ``selection_scheme="horizon"`` to retain
    the horizon-matched alternative.

    This public API is backward-compatible with v1.  Internally it uses the
    same shared statistical scoring engine as the neural teacher, but preserves
    hard-path efficiency: all candidates are validated once, persistence screening
    is applied to every eligible fallback candidate, and only surviving
    validation-ranked candidates are refitted.
    ``input_length`` + ``history_mode='sliding'`` enforces the same fixed
    lookback at every rolling origin and at the final fit.
    """
    if min_improvement_over_null < 0:
        raise ValueError("min_improvement_over_null must be non-negative.")

    evaluations, records, y_i, y_j = _score_pair_relations_with_records(
        y_i,
        y_j,
        horizon=horizon,
        validation_ratio=validation_ratio,
        epsilon=epsilon,
        enable_extra=enable_extra,
        extra_relations=extra_relations,
        weight_eta=weight_eta,
        selection_scheme=selection_scheme,
        validation_method=validation_method,
        validation_stride=validation_stride,
        max_validation_origins=max_validation_origins,
        min_validation_success_rate=min_validation_success_rate,
        representation=representation,
        input_length=input_length,
        history_mode=history_mode,
    )

    best = evaluations.best_validation_candidate
    scores = evaluations.candidate_scores

    if best is None:
        return RelationResult(
            r=np.nan,
            selected_family="null",
            selected_model="none",
            validation_loss=np.inf,
            normalized_rmse=np.inf,
            weight=0.0,
            parameters={},
            is_valid=False,
            is_custom=False,
            null_loss=float(evaluations.null_loss),
            improvement_over_null=-np.inf,
            candidate_scores=scores,
            forecast_path=np.full(int(horizon), np.nan, dtype=float),
        )

    best_passes_null, improvement = _null_gate_passes(
        candidate_loss=best.validation_loss,
        null_loss=evaluations.null_loss,
        min_improvement_over_null=min_improvement_over_null,
        epsilon=epsilon,
    )

    # Do not full-refit a relation that is rejected by the persistence gate.
    # With zero requested margin this requires strict validation improvement.
    if allow_null and not best_passes_null:
        return RelationResult(
            r=np.nan,
            selected_family="null",
            selected_model="persistence_baseline",
            validation_loss=float(best.validation_loss),
            normalized_rmse=float(best.normalized_rmse),
            weight=0.0,
            parameters={
                "best_rejected_key": best.key,
                "best_rejected_family": best.family,
                "best_rejected_model": best.model,
            },
            is_valid=False,
            is_custom=False,
            null_loss=float(evaluations.null_loss),
            improvement_over_null=float(improvement),
            candidate_scores=scores,
            forecast_path=np.full(int(horizon), np.nan, dtype=float),
        )

    record_by_key = {record.key: record for record in records}

    ranked_all = sorted(
        (
            candidate
            for candidate in evaluations.candidates
            if candidate.validation_valid and np.isfinite(candidate.validation_loss)
        ),
        key=lambda candidate: candidate.validation_loss,
    )

    if allow_null:
        ranked = []
        for candidate in ranked_all:
            passes_null, _ = _null_gate_passes(
                candidate_loss=candidate.validation_loss,
                null_loss=evaluations.null_loss,
                min_improvement_over_null=min_improvement_over_null,
                epsilon=epsilon,
            )
            if passes_null:
                ranked.append(candidate)
    else:
        ranked = ranked_all

    selected: Optional[CandidateEvaluation] = None
    for candidate in ranked:
        if _refit_evaluation(
            candidate,
            record_by_key[candidate.key],
            y_i,
            y_j,
            horizon=horizon,
            epsilon=epsilon,
            representation=representation,
            input_length=input_length,
            history_mode=history_mode,
        ):
            selected = candidate
            break

    # Refresh score diagnostics so a rare full-refit failure is visible.
    scores = evaluations.candidate_scores

    if selected is None:
        return RelationResult(
            r=np.nan,
            selected_family="null",
            selected_model="refit_failure",
            validation_loss=np.inf,
            normalized_rmse=np.inf,
            weight=0.0,
            parameters={},
            is_valid=False,
            is_custom=False,
            null_loss=float(evaluations.null_loss),
            improvement_over_null=float(improvement),
            candidate_scores=scores,
            forecast_path=np.full(int(horizon), np.nan, dtype=float),
        )

    _, selected_improvement = _null_gate_passes(
        candidate_loss=selected.validation_loss,
        null_loss=evaluations.null_loss,
        min_improvement_over_null=min_improvement_over_null,
        epsilon=epsilon,
    )

    # Expose the selected model's full 1..h path. Under history-based selection
    # the model choice is independent of h; under horizon selection it is
    # matched to the requested endpoint. The endpoint r remains path[-1].
    selected_path = _refit_selected_forecast_path(
        record_by_key[selected.key],
        y_i,
        y_j,
        horizon=horizon,
        epsilon=epsilon,
        representation=representation,
        input_length=input_length,
        history_mode=history_mode,
    )

    return RelationResult(
        r=float(selected.r),
        selected_family=selected.family,
        selected_model=selected.model,
        validation_loss=float(selected.validation_loss),
        normalized_rmse=float(selected.normalized_rmse),
        weight=float(selected.reliability),
        parameters=dict(selected.parameters),
        is_valid=True,
        is_custom=bool(selected.is_custom),
        null_loss=float(evaluations.null_loss),
        improvement_over_null=float(selected_improvement),
        candidate_scores=scores,
        forecast_path=np.asarray(selected_path, dtype=float),
    )


# ============================================================
# Example custom extension skeleton
# ============================================================


class ExampleCustomRelation(BaseRelationModel):
    """
    Minimal example showing the extension protocol.

    It is intentionally NOT part of DEFAULT_RELATION_FACTORIES.
    Users can replace its internal logic with another statistical method,
    machine-learning model, foundation model, or LLM-based estimator.
    """

    family = "custom"
    name = "example_custom"
    min_observations = 4

    def fit(self, y_i: np.ndarray, y_j: np.ndarray, epsilon: float) -> "ExampleCustomRelation":
        z = self._relation_series(y_i, y_j, epsilon)
        self.r_ = float(np.mean(z[-min(5, len(z)) :]))
        self._is_fitted = True
        return self

    def predict_path(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        return np.full(horizon, self.r_, dtype=float)

    def get_params(self) -> Dict[str, Any]:
        return {"example": True}


# ============================================================
# Standalone smoke test
# ============================================================


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    T = 120

    # Synthetic pair with a slowly evolving relative relation.
    base = np.exp(2.0 + np.cumsum(rng.normal(0.0, 0.02, T)))
    relative = np.exp(0.3 + 0.002 * np.arange(T))

    y_j = base
    y_i = base * relative * np.exp(rng.normal(0.0, 0.01, T))

    result = estimate_pair_relation(
        y_i,
        y_j,
        horizon=5,
        selection_scheme="history",
    )

    print("Default relation families:", get_default_relation_names())
    print("Candidate keys:", get_relation_candidate_keys())
    print("Selected family:", result.selected_family)
    print("Selected model:", result.selected_model)
    print("r:", result.r)
    print("validation loss:", result.validation_loss)
    print("weight:", result.weight)
    print("valid:", result.is_valid)

    horizon_result = estimate_pair_relation(
        y_i,
        y_j,
        horizon=5,
        selection_scheme="horizon",
    )
    print("\nHorizon-matched alternative:")
    print("Selected family:", horizon_result.selected_family)
    print("Selected model:", horizon_result.selected_model)

    # Optional extension interface (normally disabled).
    custom_result = estimate_pair_relation(
        y_i,
        y_j,
        horizon=5,
        enable_extra=True,
        extra_relations=[ExampleCustomRelation],
    )

    print("\nWith custom extension enabled:")
    print("Selected family:", custom_result.selected_family)
    print("Selected model:", custom_result.selected_model)