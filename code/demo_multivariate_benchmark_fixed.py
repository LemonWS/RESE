"""demo_multivariate_benchmark_fixed_v2.py
=================================================

Selection-locked multivariate benchmark runner for statistical RESE.

Purpose
-------
This demo is dedicated to direct comparison with fixed-model long-horizon
benchmarks such as PatchTST/iTransformer/Crossformer.

The key difference from ``demo_multivariate_benchmark_v5.py`` is that relation
model selection, Null screening, relation reliability, and magnitude-family
selection are performed ONCE using train+validation history only, before the
test split begins.  During test evaluation, the selected statistical
family/specification may be refitted on the past-only input window, but no
candidate family comparison, persistence screening, or family substitution is
performed using test-period history.

Protocol
--------
1. Train-only StandardScaler, identical to the ordinary benchmark runner.
2. Relation geometry is resolved once from train data.
3. At ``selection_origin = end_of_validation``:
   - select one relation family/specification for every pair;
   - decide Null/rejected pairs;
   - freeze validation reliability weights;
   - select one magnitude predictor family/specification.
4. At every test origin:
   - use exactly the latest ``input_length`` observations;
   - refit ONLY the previously selected family/specification;
   - never re-run candidate selection or Null screening;
   - a locked relation that cannot be refitted becomes unavailable for that
     window; it is NOT replaced by another family.
5. Evaluate all channels in standardized M-setting:
       mean((prediction - truth)^2) over windows x horizons x channels.

This is intentionally called a *selection-locked rolling-refit* protocol rather
than a frozen-parameter neural protocol.  It removes the principal source of
test-time model-selection advantage while retaining the natural past-only
re-estimation behavior of statistical models.

For the strictest paper comparison, use all eligible test windows:
    --max-test-windows 0 --test-stride 1
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM

from pairwise_systems import create_system_pairs
from pairwise_relation import (
    estimate_pair_relation,
    BaseRelationModel,
    StableRelation,
    TrendRelation,
    SmoothNonlinearRelation,
    DEFAULT_RELATION_FACTORIES,
)
from relation_matrix import PairwiseRelationEdge, build_relation_matrix
from relation_representation import (
    AUTO,
    available_representations,
    resolve_representation,
)
from equilibrium_solver import EquilibriumResult, solve_equilibrium_state
from equilibrium_refinement import refine_equilibrium_state
from equilibrium_adjustment import adjust_state_toward_equilibrium
from experiment_report import (
    print_locked_selection_report,
    print_locked_window_process_report,
)
from predictor import (
    predict_magnitude_from_systems,
    reconstruct_from_equilibrium,
    BaseMagnitudePredictor,
    NaiveMagnitudePredictor,
    DriftMagnitudePredictor,
    ETSTrendMagnitudePredictor,
)


# ---------------------------------------------------------------------------
# Dataset / scaling utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChronologicalSplit:
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int
    mode: str

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def val_size(self) -> int:
        return self.val_end - self.val_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


@dataclass(frozen=True)
class TrainStandardScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, train: np.ndarray) -> "TrainStandardScaler":
        train = np.asarray(train, dtype=float)
        if train.ndim != 2:
            raise ValueError("train must have shape (T, N).")
        mean = np.mean(train, axis=0)
        scale = np.std(train, axis=0, ddof=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        return cls(mean=np.asarray(mean, float), scale=np.asarray(scale, float))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.mean) / self.scale

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) * self.scale + self.mean


def _normalize_dataset_name(name: str, path: Path) -> str:
    name = str(name).strip()
    if name.lower() != "auto":
        return name
    stem = path.stem.lower()
    for canonical in ("ETTh1", "ETTh2", "ETTm1", "ETTm2"):
        if stem.startswith(canonical.lower()):
            return canonical
    return "generic"


def make_split(
    n_rows: int,
    dataset: str,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
) -> ChronologicalSplit:
    dataset_key = str(dataset).lower()

    if dataset_key in {"etth1", "etth2"}:
        train = 12 * 30 * 24
        val = 4 * 30 * 24
        test = 4 * 30 * 24
        required = train + val + test
        if n_rows < required:
            raise ValueError(
                f"{dataset} standard split requires at least {required} rows; got {n_rows}."
            )
        return ChronologicalSplit(0, train, train, train + val, train + val, required, "ETTh")

    if dataset_key in {"ettm1", "ettm2"}:
        multiplier = 4
        train = 12 * 30 * 24 * multiplier
        val = 4 * 30 * 24 * multiplier
        test = 4 * 30 * 24 * multiplier
        required = train + val + test
        if n_rows < required:
            raise ValueError(
                f"{dataset} standard split requires at least {required} rows; got {n_rows}."
            )
        return ChronologicalSplit(0, train, train, train + val, train + val, required, "ETTm")

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must lie in (0,1).")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must lie in [0,1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    # Match the common TSLib custom-dataset convention: integer train and test
    # sizes are taken first, while validation receives the rounding remainder.
    test_ratio = 1.0 - train_ratio - val_ratio
    train_size = int(n_rows * train_ratio)
    test_size = int(n_rows * test_ratio)
    val_size = int(n_rows - train_size - test_size)
    train_end = train_size
    val_end = train_size + val_size
    if train_end < 20 or val_size < 1 or test_size < 1:
        raise ValueError("Dataset is too short for the requested generic split.")
    return ChronologicalSplit(
        0, train_end,
        train_end, val_end,
        val_end, n_rows,
        "generic",
    )


def load_multivariate_csv(
    file_path: str | Path,
    *,
    date_col: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    path = Path(file_path)
    df = pd.read_csv(path)
    if date_col is not None:
        if date_col not in df.columns:
            raise ValueError(f"date column {date_col!r} not found in {path}.")
        dates = df[date_col].to_numpy()
        numeric = df.drop(columns=[date_col])
    else:
        dates = np.arange(len(df))
        numeric = df

    bad = [c for c in numeric.columns if not pd.api.types.is_numeric_dtype(numeric[c])]
    if bad:
        raise ValueError(
            "All non-date benchmark columns must be numeric. Non-numeric columns: "
            + ", ".join(map(str, bad[:10]))
        )

    Y = numeric.to_numpy(dtype=float)
    if Y.ndim != 2 or Y.shape[1] < 2:
        raise ValueError("Multivariate benchmark requires at least two numeric channels.")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Benchmark data contains NaN/inf. Preprocess missing values first.")
    return np.asarray(dates), Y, [str(c) for c in numeric.columns]


def choose_test_origins(
    split: ChronologicalSplit,
    *,
    pred_len: int,
    stride: int,
    max_windows: int,
) -> List[int]:
    if pred_len < 1:
        raise ValueError("pred_len must be at least 1.")
    if stride < 1:
        raise ValueError("test stride must be at least 1.")

    last_origin = split.test_end - pred_len
    if last_origin < split.test_start:
        raise ValueError("Test split is shorter than pred_len.")

    origins = list(range(split.test_start, last_origin + 1, stride))
    if max_windows < 0:
        raise ValueError("max_test_windows must be >=0; 0 means all windows.")
    if max_windows == 0 or max_windows >= len(origins):
        return origins

    # Evenly spaced subsample so smoke/ablation runs cover the whole test period.
    idx = np.linspace(0, len(origins) - 1, num=max_windows)
    idx = np.unique(np.round(idx).astype(int))
    return [origins[int(i)] for i in idx]



# ---------------------------------------------------------------------------
# Train/validation-only selection lock
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LockedRelationSpec:
    i: int
    j: int
    is_valid: bool
    selected_family: str
    selected_model: str
    parameters: Dict[str, Any]
    weight: float
    validation_loss: float
    normalized_rmse: float


@dataclass(frozen=True)
class LockedMagnitudeSpec:
    selected_family: str
    selected_model: str
    parameters: Dict[str, Any]
    validation_loss: float
    normalized_rmse: float


@dataclass
class LockedRelationForecast:
    is_valid: bool
    selected_family: str
    selected_model: str
    weight: float
    validation_loss: float
    normalized_rmse: float
    forecast_path: np.ndarray


class _FixedARIMARelation(BaseRelationModel):
    family = "dynamic"
    name = "fixed_arima"
    min_observations = 12

    def __init__(self, order: Sequence[int]):
        super().__init__()
        self.order = tuple(int(x) for x in order)

    def fit(self, y_i: np.ndarray, y_j: np.ndarray, epsilon: float):
        z = self._relation_series(y_i, y_j, epsilon)
        self.result_ = ARIMA(
            z,
            order=self.order,
            trend=None,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        self._is_fitted = True
        return self

    def predict_path(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        return np.asarray(
            self.result_.get_forecast(steps=horizon).predicted_mean,
            dtype=float,
        )

    @property
    def fitted_model_name(self) -> str:
        return f"ARIMA{self.order}"


class _FixedEquilibriumRelation(BaseRelationModel):
    family = "equilibrium"
    name = "fixed_vecm_or_var"
    min_observations = 20

    def __init__(self, *, kind: str, lag: int):
        super().__init__()
        self.kind = str(kind).upper()
        self.lag = int(lag)
        if self.kind not in {"VECM", "VAR"}:
            raise ValueError(f"Unknown fixed equilibrium kind: {kind!r}")

    def fit(self, y_i: np.ndarray, y_j: np.ndarray, epsilon: float):
        x_i = self._transform_series(y_i, epsilon)
        x_j = self._transform_series(y_j, epsilon)
        X = np.column_stack([x_i, x_j])
        self.X_ = X

        if self.kind == "VECM":
            self.result_ = VECM(
                X,
                k_ar_diff=max(0, self.lag),
                coint_rank=1,
                deterministic="co",
            ).fit()
        else:
            self.result_ = VAR(X).fit(maxlags=max(1, self.lag), ic=None, trend="c")

        self._is_fitted = True
        return self

    def predict_path(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        if self.kind == "VECM":
            forecast = np.asarray(self.result_.predict(steps=horizon), dtype=float)
        else:
            forecast = np.asarray(
                self.result_.forecast(
                    self.X_[-self.result_.k_ar:],
                    steps=horizon,
                ),
                dtype=float,
            )
        return forecast[:, 0] - forecast[:, 1]

    @property
    def fitted_model_name(self) -> str:
        return f"{self.kind}[locked]"


class _FixedThresholdRegimeRelation(BaseRelationModel):
    family = "regime"
    name = "SETAR(1)_fixed_threshold"
    min_observations = 20

    def __init__(self, threshold: float, min_regime_size: int = 4):
        super().__init__()
        self.threshold = float(threshold)
        self.min_regime_size = int(min_regime_size)

    @staticmethod
    def _fit_ar_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        X = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(beta[0]), float(beta[1])

    def fit(self, y_i: np.ndarray, y_j: np.ndarray, epsilon: float):
        z = self._relation_series(y_i, y_j, epsilon)
        x = z[:-1]
        y = z[1:]
        low = x <= self.threshold
        high = ~low
        if low.sum() < self.min_regime_size or high.sum() < self.min_regime_size:
            raise RuntimeError(
                "Locked SETAR threshold does not form two sufficiently populated regimes."
            )
        self.a_low_, self.b_low_ = self._fit_ar_line(x[low], y[low])
        self.a_high_, self.b_high_ = self._fit_ar_line(x[high], y[high])
        self.last_z_ = float(z[-1])
        self._is_fitted = True
        return self

    def predict_path(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        out = np.empty(horizon, dtype=float)
        value = self.last_z_
        for h in range(horizon):
            if value <= self.threshold:
                value = self.a_low_ + self.b_low_ * value
            else:
                value = self.a_high_ + self.b_high_ * value
            out[h] = value
        return out

    @property
    def fitted_model_name(self) -> str:
        return "SETAR(1)[threshold locked]"


class _FixedARIMAMagnitude(BaseMagnitudePredictor):
    family = "arima"
    name = "fixed_arima"
    min_observations = 12

    def __init__(self, order: Sequence[int]):
        super().__init__()
        self.order = tuple(int(x) for x in order)

    def fit(self, magnitude: np.ndarray):
        magnitude = np.asarray(magnitude, dtype=float)
        self.result_ = ARIMA(
            magnitude,
            order=self.order,
            trend=None,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        self._is_fitted = True
        return self

    def predict_path(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        return np.asarray(
            self.result_.get_forecast(steps=horizon).predicted_mean,
            dtype=float,
        )

    @property
    def fitted_model_name(self) -> str:
        return f"ARIMA{self.order}"


def _relation_factory_by_family() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for factory in DEFAULT_RELATION_FACTORIES:
        model = factory()
        out[str(model.family)] = factory
    return out


def _build_locked_relation_model(
    spec: LockedRelationSpec,
    representation: str,
) -> BaseRelationModel:
    family = str(spec.selected_family)
    params = dict(spec.parameters or {})

    if family == "stable":
        model: BaseRelationModel = StableRelation()
    elif family == "trend":
        model = TrendRelation()
    elif family == "dynamic":
        order = params.get("order")
        if order is None:
            raise RuntimeError("Locked dynamic relation is missing selected ARIMA order.")
        model = _FixedARIMARelation(order)
    elif family == "equilibrium":
        kind = params.get("kind")
        if kind == "VECM":
            lag = int(params.get("k_ar_diff", 1))
        elif kind == "VAR":
            lag = int(params.get("k_ar", 1))
        else:
            raise RuntimeError("Locked equilibrium relation is missing VECM/VAR kind.")
        model = _FixedEquilibriumRelation(kind=kind, lag=lag)
    elif family == "smooth_nonlinear":
        model = SmoothNonlinearRelation(
            n_knots=int(params.get("n_knots", 5)),
            degree=int(params.get("degree", 3)),
            ridge_alpha=float(params.get("ridge_alpha", 1e-3)),
        )
    elif family == "regime":
        if "threshold" not in params:
            raise RuntimeError("Locked regime relation is missing selected threshold.")
        model = _FixedThresholdRegimeRelation(float(params["threshold"]))
    else:
        factory_map = _relation_factory_by_family()
        if family not in factory_map:
            raise RuntimeError(f"No locked relation factory available for {family!r}.")
        model = factory_map[family]()

    model.set_representation(representation)
    return model


def _build_locked_magnitude_model(spec: LockedMagnitudeSpec) -> BaseMagnitudePredictor:
    family = str(spec.selected_family)
    params = dict(spec.parameters or {})
    if family == "naive":
        return NaiveMagnitudePredictor()
    if family == "drift":
        return DriftMagnitudePredictor()
    if family == "ets":
        return ETSTrendMagnitudePredictor()
    if family == "arima":
        order = params.get("order")
        if order is None:
            raise RuntimeError("Locked magnitude ARIMA is missing selected order.")
        return _FixedARIMAMagnitude(order)
    raise RuntimeError(f"Unsupported locked magnitude family: {family!r}")


def select_protocol_once(
    Y_model: np.ndarray,
    system_names: Sequence[str],
    *,
    selection_origin: int,
    input_length: int,
    pred_len: int,
    representation: str,
    allow_null: bool,
    relation_weight_eta: float,
    validation_stride: int,
    max_validation_origins: Optional[int],
    min_validation_success_rate: float,
) -> Tuple[List[LockedRelationSpec], LockedMagnitudeSpec]:
    """Select relation/magnitude specifications once before the test split."""
    history = np.asarray(Y_model[:selection_origin], dtype=float)
    pairs = create_system_pairs(history, list(system_names))
    relation_specs: List[LockedRelationSpec] = []

    for pair in pairs:
        result = estimate_pair_relation(
            pair.y_i,
            pair.y_j,
            horizon=pred_len,
            allow_null=allow_null,
            weight_eta=relation_weight_eta,
            validation_method="rolling_origin",
            validation_stride=validation_stride,
            max_validation_origins=max_validation_origins,
            min_validation_success_rate=min_validation_success_rate,
            representation=representation,
            input_length=input_length,
            history_mode="sliding",
        )
        relation_specs.append(
            LockedRelationSpec(
                i=int(pair.i),
                j=int(pair.j),
                is_valid=bool(result.is_valid),
                selected_family=str(result.selected_family),
                selected_model=str(result.selected_model),
                parameters=dict(getattr(result, "parameters", {}) or {}),
                weight=float(result.weight),
                validation_loss=float(result.validation_loss),
                normalized_rmse=float(result.normalized_rmse),
            )
        )

    magnitude = predict_magnitude_from_systems(
        history,
        horizon=pred_len,
        validation_method="rolling_origin",
        validation_stride=validation_stride,
        max_validation_origins=max_validation_origins,
        min_validation_success_rate=min_validation_success_rate,
        representation=representation,
        input_length=input_length,
        history_mode="sliding",
    )
    magnitude_spec = LockedMagnitudeSpec(
        selected_family=str(magnitude.selected_family),
        selected_model=str(magnitude.selected_model),
        parameters=dict(getattr(magnitude, "parameters", {}) or {}),
        validation_loss=float(magnitude.validation_loss),
        normalized_rmse=float(magnitude.normalized_rmse),
    )
    return relation_specs, magnitude_spec


def forecast_one_window_locked(
    Y_model: np.ndarray,
    system_names: Sequence[str],
    *,
    relation_specs: Sequence[LockedRelationSpec],
    magnitude_spec: LockedMagnitudeSpec,
    origin: int,
    input_length: int,
    pred_len: int,
    representation: str,
    es_weight_threshold: float,
    disconnected_policy: str,
    refine_equilibrium: bool,
    refinement_method: str,
    refinement_max_iterations: int,
    adjust_state: bool,
    state_adjustment_alpha: Optional[float],
    state_adjustment_rho: Optional[float],
) -> WindowForecast:
    """Forecast one test window without test-time family/Null reselection."""
    started = perf_counter()
    Y_model = np.asarray(Y_model, dtype=float)

    if origin < input_length:
        raise ValueError("Not enough history for locked test refit.")
    if origin + pred_len > len(Y_model):
        raise ValueError("Forecast target exceeds available data.")

    history = Y_model[:origin]
    fit_history = np.asarray(history[-input_length:], dtype=float)
    truth = np.asarray(Y_model[origin: origin + pred_len], dtype=float)
    pairs = create_system_pairs(fit_history, list(system_names))

    if len(pairs) != len(relation_specs):
        raise RuntimeError("Locked relation specification count does not match system pairs.")

    relation_results: List[LockedRelationForecast] = []
    for pair, spec in zip(pairs, relation_specs):
        if int(pair.i) != spec.i or int(pair.j) != spec.j:
            raise RuntimeError("Locked pair ordering changed unexpectedly.")

        if not spec.is_valid:
            relation_results.append(
                LockedRelationForecast(
                    is_valid=False,
                    selected_family=spec.selected_family,
                    selected_model=spec.selected_model,
                    weight=0.0,
                    validation_loss=spec.validation_loss,
                    normalized_rmse=spec.normalized_rmse,
                    forecast_path=np.full(pred_len, np.nan, dtype=float),
                )
            )
            continue

        try:
            model = _build_locked_relation_model(spec, representation)
            model.fit(pair.y_i, pair.y_j, 1e-8)
            path = np.asarray(model.predict_path(pred_len), dtype=float)
            if path.shape != (pred_len,) or not np.all(np.isfinite(path)):
                raise RuntimeError("Locked relation produced invalid forecast path.")
            relation_results.append(
                LockedRelationForecast(
                    is_valid=True,
                    selected_family=spec.selected_family,
                    selected_model=spec.selected_model,
                    weight=spec.weight,
                    validation_loss=spec.validation_loss,
                    normalized_rmse=spec.normalized_rmse,
                    forecast_path=path,
                )
            )
        except Exception:
            # Strict behavior: no family fallback after test begins.
            relation_results.append(
                LockedRelationForecast(
                    is_valid=False,
                    selected_family=spec.selected_family,
                    selected_model=spec.selected_model,
                    weight=0.0,
                    validation_loss=spec.validation_loss,
                    normalized_rmse=spec.normalized_rmse,
                    forecast_path=np.full(pred_len, np.nan, dtype=float),
                )
            )

    magnitude_series = np.sum(fit_history, axis=1)
    mag_model = _build_locked_magnitude_model(magnitude_spec)
    mag_model.fit(magnitude_series)
    mag_path = np.asarray(mag_model.predict_path(pred_len), dtype=float)
    if mag_path.shape != (pred_len,) or not np.all(np.isfinite(mag_path)):
        raise RuntimeError("Locked magnitude predictor produced invalid path.")

    prediction = np.empty((pred_len, Y_model.shape[1]), dtype=float)
    current_values = np.asarray(history[-1], dtype=float)
    st_component_anchor_steps = 0
    max_components_seen = 1
    latent_max_abs_by_horizon = np.full(pred_len, np.nan, dtype=float)

    for step in range(pred_len):
        state = _relation_state_at_step(
            pairs,
            relation_results,
            step_index=step,
            n_systems=Y_model.shape[1],
            system_names=system_names,
        )

        if refine_equilibrium:
            refined = refine_equilibrium_state(
                state,
                representation=representation,
                method=refinement_method,
                max_iterations=refinement_max_iterations,
                weight_threshold=es_weight_threshold,
                disconnected_policy=disconnected_policy,
                anchor_values=current_values,
                copy_state=True,
                verbose=False,
            )
            equilibrium: EquilibriumResult = refined.equilibrium
        else:
            equilibrium = solve_equilibrium_state(
                state,
                weight_threshold=es_weight_threshold,
                disconnected_policy=disconnected_policy,
                representation=representation,
                anchor_values=current_values,
            )

        latent_max_abs_by_horizon[step] = float(
            np.max(np.abs(np.asarray(equilibrium.u, dtype=float)))
        )
        max_components_seen = max(
            max_components_seen,
            int(equilibrium.diagnostics.get("n_components", len(equilibrium.components))),
        )
        if bool(equilibrium.diagnostics.get("st_component_anchor_used", False)):
            st_component_anchor_steps += 1

        if adjust_state:
            if (state_adjustment_alpha is None) == (state_adjustment_rho is None):
                raise ValueError("With --adjust-state provide exactly one of alpha or rho.")
            adjusted = adjust_state_toward_equilibrium(
                current_values,
                equilibrium,
                horizon=step + 1,
                alpha_h=state_adjustment_alpha,
                rho=state_adjustment_rho,
            )
            prediction[step] = adjusted.reconstruct(float(mag_path[step]))
        else:
            prediction[step] = reconstruct_from_equilibrium(
                equilibrium, float(mag_path[step])
            )

    family_counts = Counter(spec.selected_family for spec in relation_specs)
    model_counts = Counter(spec.selected_model for spec in relation_specs)
    relation_validation_losses = np.asarray(
        [spec.validation_loss for spec in relation_specs], dtype=float
    )
    relation_validation_nrmse = np.asarray(
        [spec.normalized_rmse for spec in relation_specs], dtype=float
    )

    valid_paths = [
        np.asarray(r.forecast_path, dtype=float)
        for r in relation_results
        if bool(r.is_valid)
        and np.asarray(r.forecast_path).shape == (pred_len,)
        and np.any(np.isfinite(np.asarray(r.forecast_path, dtype=float)))
    ]
    if valid_paths:
        relation_path_matrix = np.vstack(valid_paths)
        relation_path_max_abs_by_horizon = np.nanmax(
            np.abs(relation_path_matrix), axis=0
        )
    else:
        relation_path_max_abs_by_horizon = np.full(pred_len, np.nan, dtype=float)

    return WindowForecast(
        origin=int(origin),
        prediction_model_space=prediction,
        truth_model_space=truth,
        relation_family_counts=dict(family_counts),
        relation_model_counts=dict(model_counts),
        relation_validation_losses=relation_validation_losses,
        relation_validation_nrmse=relation_validation_nrmse,
        magnitude_validation_loss=float(magnitude_spec.validation_loss),
        magnitude_validation_nrmse=float(magnitude_spec.normalized_rmse),
        st_component_anchor_steps=int(st_component_anchor_steps),
        max_components_seen=int(max_components_seen),
        elapsed_seconds=float(perf_counter() - started),
        relation_path_max_abs_by_horizon=np.asarray(
            relation_path_max_abs_by_horizon, dtype=float
        ),
        magnitude_path=np.asarray(mag_path, dtype=float),
        latent_max_abs_by_horizon=np.asarray(
            latent_max_abs_by_horizon, dtype=float
        ),
    )


# ---------------------------------------------------------------------------
# Rel-ESE sequence forecast for one benchmark window
# ---------------------------------------------------------------------------


@dataclass
class WindowForecast:
    origin: int
    prediction_model_space: np.ndarray   # H x N
    truth_model_space: np.ndarray        # H x N
    relation_family_counts: Dict[str, int]
    relation_model_counts: Dict[str, int]

    # Internal model-selection diagnostics.  These are NOT on the same scale as
    # the final multivariate benchmark MSE/MAE: relation losses live in the
    # selected relation geometry, while magnitude losses live in aggregate-M
    # space.  They are retained to diagnose validation-vs-test behaviour.
    relation_validation_losses: np.ndarray
    relation_validation_nrmse: np.ndarray
    magnitude_validation_loss: float
    magnitude_validation_nrmse: float

    # Number of forecast horizons in this window for which a disconnected
    # relation graph was identified using the current-ST component anchor.
    st_component_anchor_steps: int
    max_components_seen: int

    elapsed_seconds: float

    # Process/stability traces used only for reporting and debugging.
    relation_path_max_abs_by_horizon: Optional[np.ndarray] = None
    magnitude_path: Optional[np.ndarray] = None
    latent_max_abs_by_horizon: Optional[np.ndarray] = None


def _relation_state_at_step(
    pairs: Sequence[Any],
    relation_results: Sequence[Any],
    *,
    step_index: int,
    n_systems: int,
    system_names: Sequence[str],
):
    edges: List[PairwiseRelationEdge] = []
    for pair, result in zip(pairs, relation_results):
        path = np.asarray(getattr(result, "forecast_path", []), dtype=float)
        valid = bool(result.is_valid)
        r = np.nan
        if valid and path.shape[0] > step_index and np.isfinite(path[step_index]):
            r = float(path[step_index])
        else:
            valid = False
        edges.append(
            PairwiseRelationEdge(
                i=int(pair.i),
                j=int(pair.j),
                r=r,
                weight=float(result.weight) if valid else 0.0,
                is_valid=valid,
                metadata={
                    "selected_family": getattr(result, "selected_family", None),
                    "selected_model": getattr(result, "selected_model", None),
                    "sequence_step": int(step_index + 1),
                },
            )
        )
    return build_relation_matrix(
        n_systems=n_systems,
        edges=edges,
        system_names=list(system_names),
        require_complete_input=True,
    )


def forecast_one_window(
    Y_model: np.ndarray,
    system_names: Sequence[str],
    *,
    origin: int,
    input_length: int,
    pred_len: int,
    representation: str,
    allow_null: bool,
    relation_weight_eta: float,
    validation_stride: int,
    max_validation_origins: Optional[int],
    min_validation_success_rate: float,
    es_weight_threshold: float,
    disconnected_policy: str,
    refine_equilibrium: bool,
    refinement_method: str,
    refinement_max_iterations: int,
    adjust_state: bool,
    state_adjustment_alpha: Optional[float],
    state_adjustment_rho: Optional[float],
) -> WindowForecast:
    started = perf_counter()
    Y_model = np.asarray(Y_model, dtype=float)
    if origin < input_length:
        raise ValueError(
            f"origin={origin} has fewer than input_length={input_length} previous observations."
        )
    if origin + pred_len > len(Y_model):
        raise ValueError("Forecast target exceeds available data.")

    history = Y_model[:origin]
    truth = Y_model[origin: origin + pred_len]

    pairs = create_system_pairs(history, list(system_names))
    relation_results = []
    for pair in pairs:
        result = estimate_pair_relation(
            pair.y_i,
            pair.y_j,
            horizon=pred_len,
            allow_null=allow_null,
            weight_eta=relation_weight_eta,
            validation_method="rolling_origin",
            validation_stride=validation_stride,
            max_validation_origins=max_validation_origins,
            min_validation_success_rate=min_validation_success_rate,
            representation=representation,
            input_length=input_length,
            history_mode="sliding",
        )
        relation_results.append(result)

    magnitude = predict_magnitude_from_systems(
        history,
        horizon=pred_len,
        validation_method="rolling_origin",
        validation_stride=validation_stride,
        max_validation_origins=max_validation_origins,
        min_validation_success_rate=min_validation_success_rate,
        representation=representation,
        input_length=input_length,
        history_mode="sliding",
    )
    mag_path = np.asarray(magnitude.forecast_path, dtype=float)
    if mag_path.shape != (pred_len,):
        raise RuntimeError("Magnitude predictor did not return a full pred_len path.")

    prediction = np.empty((pred_len, Y_model.shape[1]), dtype=float)
    current_values = np.asarray(history[-1], dtype=float)
    st_component_anchor_steps = 0
    max_components_seen = 1

    for step in range(pred_len):
        state = _relation_state_at_step(
            pairs,
            relation_results,
            step_index=step,
            n_systems=Y_model.shape[1],
            system_names=system_names,
        )

        if refine_equilibrium:
            refined = refine_equilibrium_state(
                state,
                representation=representation,
                method=refinement_method,
                max_iterations=refinement_max_iterations,
                weight_threshold=es_weight_threshold,
                disconnected_policy=disconnected_policy,
                anchor_values=current_values,
                copy_state=True,
                verbose=False,
            )
            equilibrium: EquilibriumResult = refined.equilibrium
        else:
            equilibrium = solve_equilibrium_state(
                state,
                weight_threshold=es_weight_threshold,
                disconnected_policy=disconnected_policy,
                representation=representation,
                anchor_values=current_values,
            )

        max_components_seen = max(
            max_components_seen,
            int(equilibrium.diagnostics.get("n_components", len(equilibrium.components))),
        )
        if bool(equilibrium.diagnostics.get("st_component_anchor_used", False)):
            st_component_anchor_steps += 1

        if adjust_state:
            if (state_adjustment_alpha is None) == (state_adjustment_rho is None):
                raise ValueError(
                    "With --adjust-state provide exactly one of alpha or rho."
                )
            adjusted = adjust_state_toward_equilibrium(
                current_values,
                equilibrium,
                horizon=step + 1,
                alpha_h=state_adjustment_alpha,
                rho=state_adjustment_rho,
            )
            prediction[step] = adjusted.reconstruct(float(mag_path[step]))
        else:
            prediction[step] = reconstruct_from_equilibrium(
                equilibrium, float(mag_path[step])
            )

    family_counts = Counter(str(r.selected_family) for r in relation_results)
    model_counts = Counter(str(r.selected_model) for r in relation_results)
    relation_validation_losses = np.asarray(
        [float(getattr(r, "validation_loss", np.nan)) for r in relation_results],
        dtype=float,
    )
    relation_validation_nrmse = np.asarray(
        [float(getattr(r, "normalized_rmse", np.nan)) for r in relation_results],
        dtype=float,
    )
    return WindowForecast(
        origin=int(origin),
        prediction_model_space=prediction,
        truth_model_space=np.asarray(truth, dtype=float),
        relation_family_counts=dict(family_counts),
        relation_model_counts=dict(model_counts),
        relation_validation_losses=relation_validation_losses,
        relation_validation_nrmse=relation_validation_nrmse,
        magnitude_validation_loss=float(getattr(magnitude, "validation_loss", np.nan)),
        magnitude_validation_nrmse=float(getattr(magnitude, "normalized_rmse", np.nan)),
        st_component_anchor_steps=int(st_component_anchor_steps),
        max_components_seen=int(max_components_seen),
        elapsed_seconds=float(perf_counter() - started),
    )


# ---------------------------------------------------------------------------
# Benchmark aggregation
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    dataset: str
    system_names: List[str]
    split: ChronologicalSplit
    scaler: TrainStandardScaler
    input_length: int
    pred_len: int
    model_space: str
    representation: str
    origins: List[int]
    predictions_standardized: np.ndarray  # W x H x N
    truths_standardized: np.ndarray       # W x H x N
    predictions_raw: np.ndarray
    truths_raw: np.ndarray
    window_seconds: np.ndarray

    # Internal rolling-origin model-selection diagnostics collected from each
    # test-origin refit.  Shapes: relation arrays W x E; magnitude arrays W.
    relation_validation_losses: np.ndarray
    relation_validation_nrmse: np.ndarray
    magnitude_validation_losses: np.ndarray
    magnitude_validation_nrmse: np.ndarray
    st_component_anchor_steps: np.ndarray
    max_components_seen: np.ndarray

    @property
    def mse(self) -> float:
        e = self.predictions_standardized - self.truths_standardized
        return float(np.mean(e ** 2))

    @property
    def mae(self) -> float:
        e = self.predictions_standardized - self.truths_standardized
        return float(np.mean(np.abs(e)))

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.mse))

    @property
    def raw_mse(self) -> float:
        e = self.predictions_raw - self.truths_raw
        return float(np.mean(e ** 2))

    @property
    def raw_mae(self) -> float:
        e = self.predictions_raw - self.truths_raw
        return float(np.mean(np.abs(e)))


def run_benchmark(
    file_path: str | Path,
    *,
    date_col: Optional[str] = "date",
    dataset: str = "auto",
    input_length: int = 96,
    pred_len: int = 96,
    model_space: str = "standardized",
    representation: str = AUTO,
    generic_train_ratio: float = 0.70,
    generic_val_ratio: float = 0.10,
    test_stride: int = 1,
    max_test_windows: int = 10,
    allow_null: bool = True,
    relation_weight_eta: float = 1.0,
    validation_stride: int = 1,
    max_validation_origins: Optional[int] = 10,
    min_validation_success_rate: float = 0.8,
    es_weight_threshold: float = 0.0,
    disconnected_policy: str = "st_component_anchor",
    refine_equilibrium: bool = False,
    refinement_method: str = "huber",
    refinement_max_iterations: int = 20,
    adjust_state: bool = False,
    state_adjustment_alpha: Optional[float] = None,
    state_adjustment_rho: Optional[float] = None,
    process_report: bool = False,
    verbose: bool = True,
) -> BenchmarkResult:
    path = Path(file_path).expanduser()
    dates, Y_raw, system_names = load_multivariate_csv(path, date_col=date_col)
    dataset_name = _normalize_dataset_name(dataset, path)
    split = make_split(
        len(Y_raw), dataset_name,
        train_ratio=generic_train_ratio,
        val_ratio=generic_val_ratio,
    )

    scaler = TrainStandardScaler.fit(Y_raw[split.train_start:split.train_end])
    Y_std = scaler.transform(Y_raw)

    model_space = str(model_space).strip().lower()
    if model_space not in {"standardized", "raw"}:
        raise ValueError("model_space must be 'standardized' or 'raw'.")
    Y_model = Y_std if model_space == "standardized" else Y_raw

    # Resolve AUTO only once from the train data so geometry cannot change from
    # one test window to another.
    resolved_representation = resolve_representation(
        Y_model[split.train_start:split.train_end],
        representation,
    )

    origins = choose_test_origins(
        split,
        pred_len=pred_len,
        stride=test_stride,
        max_windows=max_test_windows,
    )
    if input_length < 12:
        raise ValueError("input_length must be at least 12 for relation selection.")
    if split.test_start < input_length:
        raise ValueError("input_length exceeds history available at test start.")

    # Freeze all model-selection decisions before the test split begins.
    selection_origin = int(split.val_end)
    relation_specs, magnitude_spec = select_protocol_once(
        Y_model,
        system_names,
        selection_origin=selection_origin,
        input_length=input_length,
        pred_len=pred_len,
        representation=resolved_representation,
        allow_null=allow_null,
        relation_weight_eta=relation_weight_eta,
        validation_stride=validation_stride,
        max_validation_origins=max_validation_origins,
        min_validation_success_rate=min_validation_success_rate,
    )

    if verbose:
        print("=" * 84)
        print("RESE selection-locked multivariate benchmark")
        print("=" * 84)
        print(f"dataset                    : {dataset_name}")
        print(f"data file                  : {path}")
        print(f"shape                      : {Y_raw.shape}")
        print(f"channels                   : {len(system_names)}")
        print(f"split mode                 : {split.mode}")
        print(f"train/val/test             : {split.train_size}/{split.val_size}/{split.test_size}")
        print(f"input length               : {input_length}")
        print(f"prediction length          : {pred_len}")
        print(f"model space                : {model_space}")
        print("metric space               : train-standardized (primary) + raw (diagnostic)")
        print(f"representation             : {resolved_representation}")
        print(f"eligible test windows      : {len(choose_test_origins(split, pred_len=pred_len, stride=test_stride, max_windows=0))}")
        print(f"evaluated test windows     : {len(origins)}")
        if max_test_windows > 0:
            print("NOTE                       : sampled-window run, not the complete test table")
        else:
            print("NOTE                       : all eligible standard test windows")
        print(f"test stride                : {test_stride}")
        print(f"selection origin           : {selection_origin} (end of validation)")
        print("test-time family selection : False")
        print("test-time Null screening   : False")
        print("test-time refit            : locked family/spec only")
        print(f"validation origins / pair  : {max_validation_origins}")
        valid_locked = sum(int(s.is_valid) for s in relation_specs)
        print(f"locked valid relations     : {valid_locked}/{len(relation_specs)}")
        print(f"locked magnitude model     : {magnitude_spec.selected_family} / {magnitude_spec.selected_model}")
        print(f"equilibrium refinement     : {refine_equilibrium}")
        print(f"state adjustment           : {adjust_state}")
        print(f"disconnected policy        : {disconnected_policy}")
        if len(system_names) > 50:
            n_pairs = len(system_names) * (len(system_names) - 1) // 2
            print(f"WARNING                    : full graph has {n_pairs:,} pair relations")
        print("=" * 84)
        print_locked_selection_report(
            relation_specs,
            magnitude_spec,
            list(system_names),
        )

    pred_std_all: List[np.ndarray] = []
    truth_std_all: List[np.ndarray] = []
    pred_raw_all: List[np.ndarray] = []
    truth_raw_all: List[np.ndarray] = []
    seconds: List[float] = []
    anchor_steps_all: List[int] = []
    max_components_all: List[int] = []
    relation_val_losses_all: List[np.ndarray] = []
    relation_val_nrmse_all: List[np.ndarray] = []
    magnitude_val_losses: List[float] = []
    magnitude_val_nrmse: List[float] = []

    for wi, origin in enumerate(origins, start=1):
        if verbose:
            print(
                f"[window {wi:>4}/{len(origins)}] origin={origin} "
                f"date={dates[origin] if origin < len(dates) else origin} ...",
                flush=True,
            )
        window = forecast_one_window_locked(
            Y_model,
            system_names,
            relation_specs=relation_specs,
            magnitude_spec=magnitude_spec,
            origin=origin,
            input_length=input_length,
            pred_len=pred_len,
            representation=resolved_representation,
            es_weight_threshold=es_weight_threshold,
            disconnected_policy=disconnected_policy,
            refine_equilibrium=refine_equilibrium,
            refinement_method=refinement_method,
            refinement_max_iterations=refinement_max_iterations,
            adjust_state=adjust_state,
            state_adjustment_alpha=state_adjustment_alpha,
            state_adjustment_rho=state_adjustment_rho,
        )

        if model_space == "standardized":
            pred_std = window.prediction_model_space
            truth_std = window.truth_model_space
            pred_raw = scaler.inverse_transform(pred_std)
            truth_raw = scaler.inverse_transform(truth_std)
        else:
            pred_raw = window.prediction_model_space
            truth_raw = window.truth_model_space
            pred_std = scaler.transform(pred_raw)
            truth_std = scaler.transform(truth_raw)

        pred_std_all.append(pred_std)
        truth_std_all.append(truth_std)
        pred_raw_all.append(pred_raw)
        truth_raw_all.append(truth_raw)
        seconds.append(window.elapsed_seconds)
        anchor_steps_all.append(int(window.st_component_anchor_steps))
        max_components_all.append(int(window.max_components_seen))
        relation_val_losses_all.append(window.relation_validation_losses)
        relation_val_nrmse_all.append(window.relation_validation_nrmse)
        magnitude_val_losses.append(window.magnitude_validation_loss)
        magnitude_val_nrmse.append(window.magnitude_validation_nrmse)

        if verbose:
            e = pred_std - truth_std
            finite_rel = window.relation_validation_losses[
                np.isfinite(window.relation_validation_losses)
            ]
            rel_text = (
                f"rel-val-MSE(mean/med)={np.mean(finite_rel):.4g}/{np.median(finite_rel):.4g}"
                if finite_rel.size
                else "rel-val-MSE=n/a"
            )
            mag_text = (
                f"mag-val-MSE={window.magnitude_validation_loss:.4g}"
                if np.isfinite(window.magnitude_validation_loss)
                else "mag-val-MSE=n/a"
            )
            print(
                f"           TEST MSE={np.mean(e**2):.6f} "
                f"MAE={np.mean(np.abs(e)):.6f} | "
                f"{rel_text} | {mag_text} | "
                f"time={window.elapsed_seconds:.2f}s",
                flush=True,
            )
            if process_report:
                print_locked_window_process_report(
                    window,
                    list(system_names),
                    horizons=[h for h in (1, 24, 48, 96, pred_len) if h <= pred_len],
                )

    result = BenchmarkResult(
        dataset=dataset_name,
        system_names=list(system_names),
        split=split,
        scaler=scaler,
        input_length=int(input_length),
        pred_len=int(pred_len),
        model_space=model_space,
        representation=resolved_representation,
        origins=[int(o) for o in origins],
        predictions_standardized=np.stack(pred_std_all, axis=0),
        truths_standardized=np.stack(truth_std_all, axis=0),
        predictions_raw=np.stack(pred_raw_all, axis=0),
        truths_raw=np.stack(truth_raw_all, axis=0),
        window_seconds=np.asarray(seconds, dtype=float),
        relation_validation_losses=np.stack(relation_val_losses_all, axis=0),
        relation_validation_nrmse=np.stack(relation_val_nrmse_all, axis=0),
        magnitude_validation_losses=np.asarray(magnitude_val_losses, dtype=float),
        magnitude_validation_nrmse=np.asarray(magnitude_val_nrmse, dtype=float),
        st_component_anchor_steps=np.asarray(anchor_steps_all, dtype=int),
        max_components_seen=np.asarray(max_components_all, dtype=int),
    )
    return result


def _finite_summary(values: np.ndarray) -> Dict[str, float]:
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "count": 0.0,
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "count": float(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def print_benchmark_report(
    result: BenchmarkResult,
    *,
    target_col: Optional[str] = None,
) -> None:
    print()
    print("=" * 84)
    print("MULTIVARIATE BENCHMARK RESULT")
    print("=" * 84)
    print(f"dataset                    : {result.dataset}")
    print(f"channels                   : {len(result.system_names)}")
    print(f"windows x H x N            : {result.predictions_standardized.shape}")
    print(f"input length               : {result.input_length}")
    print(f"prediction length          : {result.pred_len}")
    print(f"model space                : {result.model_space}")
    print(f"representation             : {result.representation}")
    print("-" * 84)
    print("Primary benchmark metrics (TRAIN-standardized scale; all windows/H/channels)")
    print(f"MSE                        : {result.mse:.8f}")
    print(f"MAE                        : {result.mae:.8f}")
    print(f"RMSE                       : {result.rmse:.8f}")

    # Internal validation losses are useful for overfit/model-selection diagnostics,
    # but they are deliberately kept separate from the benchmark test MSE because
    # they live in different spaces.
    rel_loss = _finite_summary(result.relation_validation_losses)
    rel_nrmse = _finite_summary(result.relation_validation_nrmse)
    mag_loss = _finite_summary(result.magnitude_validation_losses)
    mag_nrmse = _finite_summary(result.magnitude_validation_nrmse)
    print("-" * 84)
    print("Train/validation-only LOCKED selection diagnostics (not test metrics)")
    print("relation validation MSE    : relation-space loss from the one locked selection stage")
    print(f"  finite count             : {int(rel_loss['count'])}")
    print(f"  mean                     : {rel_loss['mean']:.8f}")
    print(f"  median                   : {rel_loss['median']:.8f}")
    print(f"  min                      : {rel_loss['min']:.8f}")
    print(f"  max                      : {rel_loss['max']:.8f}")
    print("relation validation nRMSE  : scale-normalized relation loss")
    print(f"  mean                     : {rel_nrmse['mean']:.8f}")
    print(f"  median                   : {rel_nrmse['median']:.8f}")
    print("magnitude validation MSE   : aggregate-M loss from the one locked selection stage")
    print(f"  finite count             : {int(mag_loss['count'])}")
    print(f"  mean                     : {mag_loss['mean']:.8f}")
    print(f"  median                   : {mag_loss['median']:.8f}")
    print(f"  min                      : {mag_loss['min']:.8f}")
    print(f"  max                      : {mag_loss['max']:.8f}")
    print("magnitude validation nRMSE : scale-normalized aggregate-M loss")
    print(f"  mean                     : {mag_nrmse['mean']:.8f}")
    print(f"  median                   : {mag_nrmse['median']:.8f}")
    print("NOTE                       : locked validation values are repeated across windows")
    print("                             for result-array compatibility and are NOT directly comparable")
    print("                             to the primary standardized test MSE above")

    print("-" * 84)
    print("Raw-unit diagnostics (not directly comparable to standard benchmark tables)")
    print(f"raw MSE                    : {result.raw_mse:.8f}")
    print(f"raw MAE                    : {result.raw_mae:.8f}")

    if target_col is not None:
        if target_col not in result.system_names:
            print(f"target column              : {target_col!r} not found")
        else:
            j = result.system_names.index(target_col)
            e = (
                result.predictions_standardized[:, :, j]
                - result.truths_standardized[:, :, j]
            )
            print("-" * 84)
            print(f"Target-only diagnostic ({target_col}; not M-setting aggregate)")
            print(f"target MSE                 : {np.mean(e**2):.8f}")
            print(f"target MAE                 : {np.mean(np.abs(e)):.8f}")

    # Per-horizon aggregation is useful for diagnosing where the sequence error grows.
    e = result.predictions_standardized - result.truths_standardized
    horizon_mse = np.mean(e ** 2, axis=(0, 2))
    horizon_mae = np.mean(np.abs(e), axis=(0, 2))
    checkpoints = sorted(set([0, min(23, result.pred_len - 1), min(47, result.pred_len - 1), result.pred_len - 1]))
    print("-" * 84)
    print("Selected horizon diagnostics")
    for h0 in checkpoints:
        print(
            f"h={h0+1:<4d}                   : "
            f"MSE={horizon_mse[h0]:.8f}  MAE={horizon_mae[h0]:.8f}"
        )

    print("-" * 84)
    print(
        "ST component anchor       : "
        f"{int(np.sum(result.st_component_anchor_steps))} horizon-step(s) "
        f"across {int(np.sum(result.st_component_anchor_steps > 0))} window(s)"
    )
    print(
        "max components observed   : "
        f"{int(np.max(result.max_components_seen)) if result.max_components_seen.size else 1}"
    )
    print(f"mean runtime / window      : {np.mean(result.window_seconds):.4f}s")
    print(f"total benchmark runtime    : {np.sum(result.window_seconds):.4f}s")
    print("=" * 84)


def save_result_npz(result: BenchmarkResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        predictions_standardized=result.predictions_standardized,
        truths_standardized=result.truths_standardized,
        predictions_raw=result.predictions_raw,
        truths_raw=result.truths_raw,
        origins=np.asarray(result.origins, dtype=int),
        system_names=np.asarray(result.system_names, dtype=object),
        scaler_mean=result.scaler.mean,
        scaler_scale=result.scaler.scale,
        input_length=np.asarray(result.input_length),
        pred_len=np.asarray(result.pred_len),
        dataset=np.asarray(result.dataset),
        model_space=np.asarray(result.model_space),
        representation=np.asarray(result.representation),
        relation_validation_losses=result.relation_validation_losses,
        relation_validation_nrmse=result.relation_validation_nrmse,
        magnitude_validation_losses=result.magnitude_validation_losses,
        magnitude_validation_nrmse=result.magnitude_validation_nrmse,
    )
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Selection-locked RESE benchmark for direct fixed-model baseline comparison."
        )
    )
    p.add_argument("--data", required=True)
    p.add_argument("--date-col", default="date")
    p.add_argument(
        "--dataset",
        default="auto",
        choices=["auto", "ETTh1", "ETTh2", "ETTm1", "ETTm2", "generic"],
    )
    p.add_argument("--input-length", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=96)
    p.add_argument(
        "--model-space",
        choices=["standardized", "raw"],
        default="standardized",
        help=(
            "standardized matches common benchmark preprocessing; raw preserves "
            "original Rel-ESE system geometry while metrics are still standardized."
        ),
    )
    p.add_argument(
        "--representation",
        choices=list(available_representations(include_auto=True)),
        default=AUTO,
    )
    p.add_argument("--generic-train-ratio", type=float, default=0.70)
    p.add_argument("--generic-val-ratio", type=float, default=0.10)
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument(
        "--max-test-windows",
        type=int,
        default=10,
        help="0 evaluates every eligible test window; positive values evenly subsample.",
    )
    p.add_argument("--validation-stride", type=int, default=1)
    p.add_argument("--max-validation-origins", type=int, default=10)
    p.add_argument("--min-validation-success-rate", type=float, default=0.8)
    p.add_argument("--disable-null", action="store_true")
    p.add_argument("--relation-weight-eta", type=float, default=1.0)
    p.add_argument("--weight-threshold", type=float, default=0.0)
    p.add_argument(
        "--disconnected-policy",
        choices=["error", "minimum_norm", "st_component_anchor"],
        default="st_component_anchor",
    )

    p.add_argument("--refine-equilibrium", action="store_true")
    p.add_argument(
        "--refinement-method",
        choices=["huber", "cauchy", "exponential"],
        default="huber",
    )
    p.add_argument("--refinement-max-iterations", type=int, default=20)

    p.add_argument("--adjust-state", action="store_true")
    p.add_argument("--state-adjustment-alpha", type=float, default=None)
    p.add_argument("--state-adjustment-rho", type=float, default=None)

    p.add_argument(
        "--target-col",
        default="OT",
        help="Optional target-only diagnostic. PatchTST M-setting tables use ALL channels; this is diagnostic only.",
    )
    p.add_argument("--save-npz", default=None)
    p.add_argument(
        "--process-report",
        action="store_true",
        help=(
            "Print detailed per-window relation/magnitude/latent stability "
            "diagnostics through experiment_report_v6."
        ),
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_benchmark(
        args.data,
        date_col=args.date_col,
        dataset=args.dataset,
        input_length=args.input_length,
        pred_len=args.pred_len,
        model_space=args.model_space,
        representation=args.representation,
        generic_train_ratio=args.generic_train_ratio,
        generic_val_ratio=args.generic_val_ratio,
        test_stride=args.test_stride,
        max_test_windows=args.max_test_windows,
        allow_null=not args.disable_null,
        relation_weight_eta=args.relation_weight_eta,
        validation_stride=args.validation_stride,
        max_validation_origins=args.max_validation_origins,
        min_validation_success_rate=args.min_validation_success_rate,
        es_weight_threshold=args.weight_threshold,
        disconnected_policy=args.disconnected_policy,
        refine_equilibrium=args.refine_equilibrium,
        refinement_method=args.refinement_method,
        refinement_max_iterations=args.refinement_max_iterations,
        adjust_state=args.adjust_state,
        state_adjustment_alpha=args.state_adjustment_alpha,
        state_adjustment_rho=args.state_adjustment_rho,
        process_report=args.process_report,
        verbose=not args.quiet,
    )
    print_benchmark_report(result, target_col=args.target_col)
    if args.save_npz:
        saved = save_result_npz(result, args.save_npz)
        print(f"Saved predictions/targets : {saved}")


if __name__ == "__main__":
    main()