
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np

from relation_representation import LOG_RATIO, normalize_representation_name
from forecast_window import (
    AUTO as AUTO_HISTORY,
    filter_validation_origins,
    resolve_history_mode,
    slice_fit_history,
)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ============================================================
# Result containers
# ============================================================


MagnitudePredictorFactory = Callable[[], "BaseMagnitudePredictor"]


@dataclass
class PredictorCandidateScore:
    """Validation information for one candidate magnitude predictor."""

    family: str
    model: str
    validation_loss: float
    normalized_rmse: float
    is_custom: bool = False
    error: Optional[str] = None
    n_validation_origins: int = 0
    n_successful_origins: int = 0
    validation_success_rate: float = 0.0


@dataclass
class PredictorResult:
    """Final overall-magnitude forecast.

    Parameters
    ----------
    forecast_path : ndarray, shape (horizon,)
        Forecasts for M_{t+1}, ..., M_{t+h}.

    magnitude : float
        Horizon-specific final forecast, equal to forecast_path[-1].

    selected_family, selected_model : str
        Selected predictor identity.

    validation_loss : float
        Temporal validation MSE. Under the v2 default this is the requested
        horizon's rolling-origin MSE.

    normalized_rmse : float
        Validation RMSE divided by the scale of the fitting magnitude series.

    parameters : dict
        Predictor-specific fitted parameters / diagnostics.

    candidate_scores : list[PredictorCandidateScore]
        Validation results for every attempted default/custom predictor.

    history_magnitude : ndarray
        Magnitude series used for final full-history fitting.

    horizon : int
        Requested forecast horizon.
    """

    forecast_path: np.ndarray
    magnitude: float

    selected_family: str
    selected_model: str

    validation_loss: float
    normalized_rmse: float

    parameters: Dict[str, Any] = field(default_factory=dict)
    candidate_scores: List[PredictorCandidateScore] = field(default_factory=list)

    history_magnitude: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    horizon: int = 1

    is_custom: bool = False
    clip_nonnegative: bool = True

    validation_method: str = "rolling_origin"
    validation_stride: int = 1
    max_validation_origins: Optional[int] = 10
    min_validation_success_rate: float = 0.8
    validation_origins: List[int] = field(default_factory=list)
    input_length: Optional[int] = None
    history_mode: str = "expanding"

    def summary(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "magnitude": self.magnitude,
            "selected_family": self.selected_family,
            "selected_model": self.selected_model,
            "validation_loss": self.validation_loss,
            "normalized_rmse": self.normalized_rmse,
            "is_custom": self.is_custom,
            "validation_method": self.validation_method,
            "n_validation_origins": len(self.validation_origins),
            "input_length": self.input_length,
            "history_mode": self.history_mode,
        }


# ============================================================
# Input utilities
# ============================================================


def _as_1d_float(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    return x


def _validate_magnitude(magnitude: np.ndarray) -> np.ndarray:
    magnitude = _as_1d_float(magnitude, "magnitude")

    if len(magnitude) < 8:
        raise ValueError("At least 8 magnitude observations are required.")

    if not np.all(np.isfinite(magnitude)):
        raise ValueError("Magnitude must contain only finite values.")

    # Signed aggregate histories are valid for additive Rel-ESE.  Whether
    # forecasts are clipped to non-negative values is controlled separately by
    # ``clip_nonnegative`` in the predictor selection API.
    return magnitude


def aggregate_magnitude(Y: np.ndarray) -> np.ndarray:
    """Aggregate a T x N system matrix into the total magnitude series M_t.

    Parameters
    ----------
    Y : ndarray, shape (T, N)
        Rows are time observations; columns are systems.

    Returns
    -------
    ndarray, shape (T,)
        M_t = sum_i y_{i,t}.
    """
    Y = np.asarray(Y, dtype=float)

    if Y.ndim != 2:
        raise ValueError("Y must have shape (T, N).")

    if Y.shape[0] < 1 or Y.shape[1] < 1:
        raise ValueError("Y must contain at least one time point and one system.")

    if not np.all(np.isfinite(Y)):
        raise ValueError("Y must contain only finite values.")

    return np.sum(Y, axis=1, dtype=float)


def _mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError(
            f"actual and predicted must have the same shape; "
            f"got {actual.shape} and {predicted.shape}."
        )

    return float(np.mean((actual - predicted) ** 2))


def _series_scale(x: np.ndarray, epsilon: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    std = float(np.std(x))
    mad = float(np.median(np.abs(x - np.median(x)))) * 1.4826
    mean_abs = float(np.mean(np.abs(x)))

    # Keep normalization meaningful for nearly constant positive series.
    return max(std, mad, mean_abs * 1e-4, 1e-8, epsilon)


# ============================================================
# Base predictor interface
# ============================================================


class BaseMagnitudePredictor(ABC):
    """Common interface for all overall-magnitude predictors."""

    family: str = "base"
    name: str = "base"
    min_observations: int = 2

    def __init__(self) -> None:
        self._is_fitted = False

    @abstractmethod
    def fit(self, magnitude: np.ndarray) -> "BaseMagnitudePredictor":
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
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before prediction."
            )
        if horizon < 1:
            raise ValueError("horizon must be at least 1.")


# ============================================================
# 1. Naive persistence
# ============================================================


class NaiveMagnitudePredictor(BaseMagnitudePredictor):
    """Persistence predictor M_{t+h|t} = M_t."""

    family = "naive"
    name = "persistence"
    min_observations = 2

    def fit(self, magnitude: np.ndarray) -> "NaiveMagnitudePredictor":
        magnitude = _validate_magnitude(magnitude)
        self.last_value_ = float(magnitude[-1])
        self._is_fitted = True
        return self

    def predict_path(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        return np.full(horizon, self.last_value_, dtype=float)

    def get_params(self) -> Dict[str, Any]:
        return {"last_value": self.last_value_}


# ============================================================
# 2. Drift predictor
# ============================================================


class DriftMagnitudePredictor(BaseMagnitudePredictor):
    """Linear drift based on the average historical change."""

    family = "drift"
    name = "random_walk_with_drift"
    min_observations = 3

    def fit(self, magnitude: np.ndarray) -> "DriftMagnitudePredictor":
        magnitude = _validate_magnitude(magnitude)

        self.last_value_ = float(magnitude[-1])
        self.drift_ = float((magnitude[-1] - magnitude[0]) / (len(magnitude) - 1))
        self._is_fitted = True
        return self

    def predict_path(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        steps = np.arange(1, horizon + 1, dtype=float)
        return self.last_value_ + self.drift_ * steps

    def get_params(self) -> Dict[str, Any]:
        return {
            "last_value": self.last_value_,
            "drift_per_step": self.drift_,
        }


# ============================================================
# 3. ETS / Holt damped trend
# ============================================================


class ETSTrendMagnitudePredictor(BaseMagnitudePredictor):
    """Additive Holt trend with damping."""

    family = "ets"
    name = "holt_damped_trend"
    min_observations = 8

    def fit(self, magnitude: np.ndarray) -> "ETSTrendMagnitudePredictor":
        magnitude = _validate_magnitude(magnitude)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                magnitude,
                trend="add",
                damped_trend=True,
                seasonal=None,
                initialization_method="estimated",
            )
            self.result_ = model.fit(optimized=True, remove_bias=False)

        self._is_fitted = True
        return self

    def predict_path(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        pred = self.result_.forecast(horizon)
        return np.asarray(pred, dtype=float)

    def get_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}

        fitted_params = getattr(self.result_, "params", None)
        if fitted_params is not None:
            try:
                for key, value in dict(fitted_params).items():
                    if np.isscalar(value) and np.isfinite(value):
                        params[str(key)] = float(value)
            except Exception:
                pass

        sse = getattr(self.result_, "sse", None)
        if sse is not None and np.isfinite(sse):
            params["sse"] = float(sse)

        return params


# ============================================================
# 4. Auto-ARIMA
# ============================================================


class AutoARIMAMagnitudePredictor(BaseMagnitudePredictor):
    """Low-order ARIMA grid selected by AIC."""

    family = "arima"
    name = "auto_arima"
    min_observations = 12

    def __init__(
        self,
        max_p: int = 2,
        max_d: int = 1,
        max_q: int = 2,
    ) -> None:
        super().__init__()
        self.max_p = int(max_p)
        self.max_d = int(max_d)
        self.max_q = int(max_q)

    def fit(self, magnitude: np.ndarray) -> "AutoARIMAMagnitudePredictor":
        magnitude = _validate_magnitude(magnitude)

        best_result = None
        best_order = None
        best_aic = np.inf

        for d in range(self.max_d + 1):
            for p in range(self.max_p + 1):
                for q in range(self.max_q + 1):
                    order = (p, d, q)

                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            result = ARIMA(
                                magnitude,
                                order=order,
                                trend=None,
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            ).fit()

                        if np.isfinite(result.aic) and float(result.aic) < best_aic:
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
# Default registry + extension interface
# ============================================================


DEFAULT_PREDICTOR_FACTORIES: Tuple[MagnitudePredictorFactory, ...] = (
    NaiveMagnitudePredictor,
    DriftMagnitudePredictor,
    ETSTrendMagnitudePredictor,
    AutoARIMAMagnitudePredictor,
)


def get_default_predictor_names() -> List[str]:
    return [factory().family for factory in DEFAULT_PREDICTOR_FACTORIES]


def _instantiate_predictors(
    enable_extra: bool,
    extra_predictors: Optional[Sequence[MagnitudePredictorFactory]],
) -> List[Tuple[MagnitudePredictorFactory, bool]]:
    factories: List[Tuple[MagnitudePredictorFactory, bool]] = [
        (factory, False) for factory in DEFAULT_PREDICTOR_FACTORIES
    ]

    if enable_extra:
        if not extra_predictors:
            raise ValueError(
                "enable_extra=True but no extra_predictors were supplied."
            )

        for factory in extra_predictors:
            model = factory()
            if not isinstance(model, BaseMagnitudePredictor):
                raise TypeError(
                    "Every custom predictor factory must return an instance of "
                    "BaseMagnitudePredictor."
                )
            factories.append((factory, True))

    return factories


# ============================================================
# Candidate validation
# ============================================================


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
    """Build recent fixed-horizon validation origins."""
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
    if max_validation_origins is not None and len(origins) > max_validation_origins:
        origins = origins[-max_validation_origins:]

    if not origins:
        raise ValueError(
            "No legal validation origins were produced for the configured "
            f"input_length={input_length}, history_mode={history_mode!r}."
        )

    return origins, validation_start


def _validate_candidate_path(
    factory: MagnitudePredictorFactory,
    magnitude_train: np.ndarray,
    magnitude_val: np.ndarray,
    *,
    clip_nonnegative: bool,
    epsilon: float,
    input_length: Optional[int] = None,
    history_mode: str = AUTO_HISTORY,
) -> Tuple[float, float, str, Optional[str], int, int, float]:
    """Legacy v1 single-fit path validation."""
    model = factory()
    magnitude_train = slice_fit_history(
        magnitude_train,
        input_length=input_length,
        history_mode=history_mode,
    )

    if len(magnitude_train) < model.min_observations:
        return (
            np.inf,
            np.inf,
            model.name,
            f"requires at least {model.min_observations} training observations",
            1,
            0,
            0.0,
        )

    try:
        model.fit(magnitude_train)
        pred = np.asarray(model.predict_path(len(magnitude_val)), dtype=float)

        if pred.shape != magnitude_val.shape:
            raise RuntimeError("candidate produced an incorrect validation shape")
        if not np.all(np.isfinite(pred)):
            raise RuntimeError("candidate produced non-finite validation forecasts")
        if clip_nonnegative:
            pred = np.maximum(pred, 0.0)

        loss = _mse(magnitude_val, pred)
        nrmse = float(np.sqrt(loss) / _series_scale(magnitude_train, epsilon))
        return float(loss), nrmse, model.fitted_model_name, None, 1, 1, 1.0

    except Exception as exc:
        return np.inf, np.inf, model.name, str(exc), 1, 0, 0.0


def _validate_candidate_rolling(
    factory: MagnitudePredictorFactory,
    magnitude: np.ndarray,
    *,
    origins: Sequence[int],
    horizon: int,
    scale_history: np.ndarray,
    clip_nonnegative: bool,
    epsilon: float,
    min_validation_success_rate: float,
    input_length: Optional[int] = None,
    history_mode: str = AUTO_HISTORY,
) -> Tuple[float, float, str, Optional[str], int, int, float]:
    """Evaluate the requested h-step magnitude forecast at rolling origins."""
    squared_errors: List[float] = []
    fitted_names: List[str] = []
    failures: List[str] = []

    for origin in origins:
        model = factory()

        try:
            magnitude_fit = slice_fit_history(
                magnitude,
                origin=origin,
                input_length=input_length,
                history_mode=history_mode,
            )
        except ValueError as exc:
            failures.append(f"origin {origin}: {exc}")
            continue

        if len(magnitude_fit) < model.min_observations:
            failures.append(
                f"origin {origin}: fit window has {len(magnitude_fit)} observations; "
                f"requires at least {model.min_observations}"
            )
            continue

        target_index = int(origin + horizon - 1)

        try:
            model.fit(magnitude_fit)
            path = np.asarray(model.predict_path(horizon), dtype=float)

            if path.shape != (horizon,):
                raise RuntimeError("candidate produced an incorrect h-step shape")
            if not np.all(np.isfinite(path)):
                raise RuntimeError("candidate produced non-finite h-step forecasts")

            prediction = float(path[-1])
            if clip_nonnegative:
                prediction = max(prediction, 0.0)

            target = float(magnitude[target_index])
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
        return np.inf, np.inf, factory().name, message, n_total, n_success, success_rate

    loss = float(np.mean(np.asarray(squared_errors, dtype=float)))
    nrmse = float(np.sqrt(loss) / _series_scale(scale_history, epsilon))
    fitted_name = fitted_names[-1] if fitted_names else factory().name

    # Partial failures are represented by the success-count diagnostics.  A
    # candidate that passes the configured threshold is still validation-valid.
    return loss, nrmse, fitted_name, None, n_total, n_success, success_rate


# ============================================================
# Main predictor
# ============================================================


def predict_magnitude(
    magnitude: np.ndarray,
    *,
    horizon: int = 1,
    validation_ratio: float = 0.20,
    epsilon: float = 1e-8,
    clip_nonnegative: bool = True,
    enable_extra: bool = False,
    extra_predictors: Optional[Sequence[MagnitudePredictorFactory]] = None,
    validation_method: str = "rolling_origin",
    validation_stride: int = 1,
    max_validation_origins: Optional[int] = 10,
    min_validation_success_rate: float = 0.8,
    input_length: Optional[int] = None,
    history_mode: str = AUTO_HISTORY,
) -> PredictorResult:
    """Select and fit an overall-magnitude predictor.

    Default selection is horizon-matched rolling-origin validation:

        L_{M,h}^{(k)} = mean_o (M_hat_{o+h|o}^{(k)} - M_{o+h})^2.

    Therefore a request for h=5 selects the predictor by historical 5-step
    errors, not by a mixed 1..V forecast path.  Set `validation_method="path"`
    only to reproduce the v1 protocol.
    """
    magnitude = _validate_magnitude(magnitude)

    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if not 0.05 <= validation_ratio < 0.5:
        raise ValueError("validation_ratio must lie in [0.05, 0.5).")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    validation_method = str(validation_method).lower()
    if validation_method not in {"rolling_origin", "path"}:
        raise ValueError("validation_method must be 'rolling_origin' or 'path'.")
    if validation_stride < 1:
        raise ValueError("validation_stride must be at least 1.")
    if max_validation_origins is not None and max_validation_origins < 1:
        raise ValueError("max_validation_origins must be None or at least 1.")
    if not 0 < min_validation_success_rate <= 1:
        raise ValueError("min_validation_success_rate must lie in (0, 1].")

    history_mode = resolve_history_mode(input_length, history_mode)

    n = len(magnitude)
    factories = _instantiate_predictors(enable_extra, extra_predictors)

    if validation_method == "rolling_origin":
        origins, validation_start = _build_validation_origins(
            n,
            horizon,
            validation_ratio,
            validation_stride,
            max_validation_origins,
            input_length=input_length,
            history_mode=history_mode,
        )
        scale_history = slice_fit_history(
            magnitude,
            origin=validation_start,
            input_length=input_length,
            history_mode=history_mode,
        )
        if len(scale_history) < 2:
            scale_history = slice_fit_history(
                magnitude,
                origin=origins[0],
                input_length=input_length,
                history_mode=history_mode,
            )
    else:
        validation_size = max(2, int(round(n * validation_ratio)))
        split = n - validation_size
        magnitude_train = magnitude[:split]
        magnitude_val = magnitude[split:]
        origins = [split]
        scale_history = slice_fit_history(
            magnitude_train,
            input_length=input_length,
            history_mode=history_mode,
        )

    candidate_scores: List[PredictorCandidateScore] = []
    ranked: List[
        Tuple[
            float,
            float,
            MagnitudePredictorFactory,
            bool,
            str,
            str,
        ]
    ] = []

    for factory, is_custom in factories:
        probe = factory()

        if validation_method == "rolling_origin":
            (
                loss,
                nrmse,
                fitted_name,
                error,
                n_total,
                n_success,
                success_rate,
            ) = _validate_candidate_rolling(
                factory,
                magnitude,
                origins=origins,
                horizon=horizon,
                scale_history=scale_history,
                clip_nonnegative=clip_nonnegative,
                epsilon=epsilon,
                min_validation_success_rate=min_validation_success_rate,
                input_length=input_length,
                history_mode=history_mode,
            )
        else:
            (
                loss,
                nrmse,
                fitted_name,
                error,
                n_total,
                n_success,
                success_rate,
            ) = _validate_candidate_path(
                factory,
                magnitude_train,
                magnitude_val,
                clip_nonnegative=clip_nonnegative,
                epsilon=epsilon,
                input_length=input_length,
                history_mode=history_mode,
            )

        candidate_scores.append(
            PredictorCandidateScore(
                family=probe.family,
                model=fitted_name,
                validation_loss=float(loss),
                normalized_rmse=float(nrmse),
                is_custom=is_custom,
                error=error,
                n_validation_origins=n_total,
                n_successful_origins=n_success,
                validation_success_rate=success_rate,
            )
        )

        if np.isfinite(loss):
            ranked.append(
                (
                    float(loss),
                    float(nrmse),
                    factory,
                    is_custom,
                    probe.family,
                    fitted_name,
                )
            )

    if not ranked:
        raise RuntimeError("No magnitude predictor could be fitted successfully.")

    ranked.sort(key=lambda item: item[0])

    selected_model: Optional[BaseMagnitudePredictor] = None
    selected_loss = np.inf
    selected_nrmse = np.inf
    selected_is_custom = False

    for loss, nrmse, factory, is_custom, _, _ in ranked:
        try:
            model = factory()
            final_fit_history = slice_fit_history(
                magnitude,
                input_length=input_length,
                history_mode=history_mode,
            )
            if len(final_fit_history) < model.min_observations:
                raise RuntimeError(
                    f"final fit window has {len(final_fit_history)} observations; "
                    f"requires at least {model.min_observations}"
                )
            model.fit(final_fit_history)
            forecast_path = np.asarray(model.predict_path(horizon), dtype=float)

            if forecast_path.shape != (horizon,):
                raise RuntimeError("full-history predictor returned wrong shape")
            if not np.all(np.isfinite(forecast_path)):
                raise RuntimeError("full-history predictor returned non-finite values")
            if clip_nonnegative:
                forecast_path = np.maximum(forecast_path, 0.0)

            selected_model = model
            selected_loss = float(loss)
            selected_nrmse = float(nrmse)
            selected_is_custom = bool(is_custom)
            break

        except Exception:
            continue

    if selected_model is None:
        raise RuntimeError(
            "Validated magnitude predictors all failed during full-history refitting."
        )

    return PredictorResult(
        forecast_path=np.asarray(forecast_path, dtype=float),
        magnitude=float(forecast_path[-1]),

        selected_family=selected_model.family,
        selected_model=selected_model.fitted_model_name,

        validation_loss=selected_loss,
        normalized_rmse=selected_nrmse,

        parameters=selected_model.get_params(),
        candidate_scores=candidate_scores,

        history_magnitude=np.asarray(final_fit_history, dtype=float).copy(),
        horizon=int(horizon),

        is_custom=selected_is_custom,
        clip_nonnegative=bool(clip_nonnegative),

        validation_method=validation_method,
        validation_stride=int(validation_stride),
        max_validation_origins=max_validation_origins,
        min_validation_success_rate=float(min_validation_success_rate),
        validation_origins=[int(x) for x in origins],
        input_length=input_length,
        history_mode=history_mode,
    )


def predict_magnitude_from_systems(
    Y: np.ndarray,
    *,
    horizon: int = 1,
    validation_ratio: float = 0.20,
    epsilon: float = 1e-8,
    clip_nonnegative: Optional[bool] = None,
    representation: str = LOG_RATIO,
    enable_extra: bool = False,
    extra_predictors: Optional[Sequence[MagnitudePredictorFactory]] = None,
    validation_method: str = "rolling_origin",
    validation_stride: int = 1,
    max_validation_origins: Optional[int] = 10,
    min_validation_success_rate: float = 0.8,
    input_length: Optional[int] = None,
    history_mode: str = AUTO_HISTORY,
) -> PredictorResult:
    """Convenience wrapper: Y -> M -> selected horizon-matched magnitude forecast.

    If ``clip_nonnegative`` is omitted, multiplicative/log-ratio mode clips
    aggregate forecasts at zero while signed geometries leave aggregate forecasts unchanged.
    """
    representation = normalize_representation_name(representation)
    if clip_nonnegative is None:
        clip_nonnegative = bool(representation == LOG_RATIO)

    magnitude = aggregate_magnitude(Y)

    return predict_magnitude(
        magnitude,
        horizon=horizon,
        validation_ratio=validation_ratio,
        epsilon=epsilon,
        clip_nonnegative=clip_nonnegative,
        enable_extra=enable_extra,
        extra_predictors=extra_predictors,
        validation_method=validation_method,
        validation_stride=validation_stride,
        max_validation_origins=max_validation_origins,
        min_validation_success_rate=min_validation_success_rate,
        input_length=input_length,
        history_mode=history_mode,
    )

# ============================================================
# Optional reconstruction helpers
# ============================================================


def reconstruct_from_equilibrium(
    equilibrium: Any,
    magnitude_forecast: float,
) -> np.ndarray:
    """Representation-aware reconstruction from an EquilibriumResult-like object.

    The object must expose ``u``, ``representation``, and a ``reconstruct``
    method (as provided by ``equilibrium_solver_v3.EquilibriumResult``).
    """
    if not hasattr(equilibrium, "reconstruct"):
        raise TypeError(
            "equilibrium must expose reconstruct(aggregate); use "
            "equilibrium_solver_v3.EquilibriumResult."
        )
    y_hat = np.asarray(
        equilibrium.reconstruct(float(magnitude_forecast)),
        dtype=float,
    )
    if y_hat.ndim != 1 or not np.all(np.isfinite(y_hat)):
        raise RuntimeError("Equilibrium reconstruction produced invalid values.")
    return y_hat


def reconstruct_system_forecast(
    gamma: np.ndarray,
    magnitude_forecast: float,
) -> np.ndarray:
    """Combine an ES vector and overall magnitude into system-level forecasts.

    Parameters
    ----------
    gamma : ndarray, shape (N,)
        Equilibrium proportions, normally from `equilibrium_solver_v1.py`.

    magnitude_forecast : float
        Horizon-specific forecast M_hat_{t+h|t}.

    Returns
    -------
    ndarray, shape (N,)
        y_hat_i = magnitude_forecast * gamma_i.
    """
    gamma = _as_1d_float(gamma, "gamma")

    if not np.all(np.isfinite(gamma)):
        raise ValueError("gamma must contain only finite values.")

    if np.any(gamma < 0):
        raise ValueError("gamma must be non-negative.")

    total = float(np.sum(gamma))
    if total <= 0:
        raise ValueError("gamma must have a positive sum.")

    if not np.isfinite(magnitude_forecast) or magnitude_forecast < 0:
        raise ValueError("magnitude_forecast must be a finite non-negative scalar.")

    # Numerical normalization keeps the helper safe even if gamma differs from
    # exact sum-one by floating-point roundoff.
    gamma_normalized = gamma / total
    return float(magnitude_forecast) * gamma_normalized


# ============================================================
# Example custom extension
# ============================================================


class ExampleCustomMagnitudePredictor(BaseMagnitudePredictor):
    """Minimal extension example; not included in the default library."""

    family = "custom"
    name = "recent_mean"
    min_observations = 4

    def __init__(self, recent_window: int = 5) -> None:
        super().__init__()
        self.recent_window = int(recent_window)

    def fit(self, magnitude: np.ndarray) -> "ExampleCustomMagnitudePredictor":
        magnitude = _validate_magnitude(magnitude)
        window = min(self.recent_window, len(magnitude))
        self.level_ = float(np.mean(magnitude[-window:]))
        self.used_window_ = int(window)
        self._is_fitted = True
        return self

    def predict_path(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        return np.full(horizon, self.level_, dtype=float)

    def get_params(self) -> Dict[str, Any]:
        return {
            "recent_window": self.used_window_,
            "level": self.level_,
        }


# ============================================================
# Standalone smoke test
# ============================================================


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    T, N = 120, 5

    trend = 100.0 + 0.8 * np.arange(T)
    M_true = np.maximum(
        trend + rng.normal(0.0, 4.0, T),
        1.0,
    )

    # Split magnitude into synthetic positive system shares only to test the
    # convenience wrapper Y -> M.
    shares = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    Y = M_true[:, None] * shares[None, :]

    result = predict_magnitude_from_systems(
        Y,
        horizon=5,
    )

    print("default predictors:", get_default_predictor_names())
    print("selected family:", result.selected_family)
    print("selected model:", result.selected_model)
    print("forecast path:", np.round(result.forecast_path, 4))
    print("M(t+5):", result.magnitude)
    print("validation MSE:", result.validation_loss)
    print("normalized RMSE:", result.normalized_rmse)

    gamma = shares / shares.sum()
    y_hat = reconstruct_system_forecast(gamma, result.magnitude)
    print("reconstructed y(t+5):", np.round(y_hat, 4))
    print("reconstructed total:", float(y_hat.sum()))
