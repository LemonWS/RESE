
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from relation_matrix import RelationMatrixState
from relation_representation import (
    ADDITIVE,
    LOG_RATIO,
    centered_latent_state,
    normalize_representation_name,
    realized_relation_matrix as _representation_realized_relation_matrix,
    validate_observations as _validate_representation_observations,
)
from equilibrium_solver import EquilibriumResult
from predictor import PredictorResult


# ============================================================
# Result containers
# ============================================================


@dataclass
class ForecastEvaluation:
    """System-level final forecast metrics."""

    n_values: int
    mse: float
    rmse: float
    mae: float
    bias: float

    mape_percent: float
    smape_percent: float
    wape_percent: float

    max_absolute_error: float

    actual_total: float
    forecast_total: float
    total_error: float

    per_system_absolute_error: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )


@dataclass
class MagnitudeEvaluation:
    """Evaluation of the overall magnitude predictor."""

    actual_magnitude: float
    predicted_magnitude: float

    error: float
    absolute_error: float
    squared_error: float
    percentage_error_percent: float
    absolute_percentage_error_percent: float


@dataclass
class EquilibriumEvaluation:
    """Evaluation of the predicted equilibrium allocation."""

    gamma_true: np.ndarray
    gamma_pred: np.ndarray

    mse: float
    rmse: float
    mae: float
    l1_distance: float
    max_absolute_error: float

    cosine_similarity: float
    jensen_shannon_divergence: float

    sum_true: float
    sum_pred: float

    # In additive mode the legacy gamma_* arrays store zero-sum latent state
    # coordinates for backward compatibility with existing reporting code.
    representation: str = LOG_RATIO


@dataclass
class RelationEvaluation:
    """Evaluation of pairwise relation forecasts against realized relations."""

    n_possible_edges: int
    n_valid_edges: int
    n_evaluated_edges: int
    coverage: float

    mse: float
    rmse: float
    mae: float

    weighted_mse: float
    weighted_rmse: float
    weighted_mae: float

    bias: float
    weighted_bias: float

    sign_accuracy: float
    weighted_sign_accuracy: float

    mean_weight: float
    total_weight: float

    true_edge_relations: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    predicted_edge_relations: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    edge_weights: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    edge_index: np.ndarray = field(
        default_factory=lambda: np.empty((2, 0), dtype=int)
    )


@dataclass
class ConsistencyEvaluation:
    """Internal local-to-global consistency diagnostics.

    The raw residual fields are always retained because they describe the
    numerical projection performed by the ES solver.

    The cycle-consistency fields are the interpretation-safe metrics:

        cycle_rank = E - N + C

    where E is the number of used edges, N is the number of systems, and C is
    the number of connected components.

    `consistency_testable` is True only when the used relation graph is
    connected and contains at least one independent cycle (cycle_rank > 0).
    For a connected tree, raw residuals may be exactly zero structurally, so
    `cycle_consistency_weighted_rmse` and
    `cycle_consistency_normalized_error` are reported as NaN rather than being
    misinterpreted as evidence of perfect relation consistency.
    """

    connected: bool
    globally_identified: bool

    n_systems: int
    n_used_edges: int
    n_components: int

    cycle_rank: int
    has_cycle_redundancy: bool
    consistency_testable: bool

    # Raw solver projection diagnostics.  These remain available even when
    # there is no cycle redundancy.
    weighted_sse: float
    weighted_mse: float
    weighted_rmse: float
    normalized_consistency_error: float

    mean_absolute_residual: float
    max_absolute_residual: float

    # Interpretation-safe consistency scores.  These are NaN when
    # consistency_testable=False.
    cycle_consistency_weighted_rmse: float
    cycle_consistency_normalized_error: float

    edge_density_used: float
    algebraic_connectivity: float
    spectral_condition_number: float


@dataclass
class PipelineEvaluation:
    """Complete evaluation report for one ESE forecast origin/horizon."""

    horizon: int

    forecast: ForecastEvaluation
    magnitude: MagnitudeEvaluation
    equilibrium: EquilibriumEvaluation
    relation: RelationEvaluation
    consistency: ConsistencyEvaluation

    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, float]:
        """Compact scalar summary useful for logging and experiment tables."""
        return {
            "horizon": float(self.horizon),

            "forecast_rmse": self.forecast.rmse,
            "forecast_mae": self.forecast.mae,
            "forecast_mape_percent": self.forecast.mape_percent,
            "forecast_smape_percent": self.forecast.smape_percent,
            "forecast_wape_percent": self.forecast.wape_percent,

            "magnitude_absolute_error": self.magnitude.absolute_error,
            "magnitude_ape_percent": self.magnitude.absolute_percentage_error_percent,

            "state_rmse": self.equilibrium.rmse,
            "state_mae": self.equilibrium.mae,
            "state_l1": self.equilibrium.l1_distance,
            # Legacy keys retained for positive/log-ratio experiments.
            "gamma_rmse": self.equilibrium.rmse,
            "gamma_mae": self.equilibrium.mae,
            "gamma_l1": self.equilibrium.l1_distance,
            "gamma_js_divergence": self.equilibrium.jensen_shannon_divergence,

            "relation_rmse": self.relation.rmse,
            "relation_mae": self.relation.mae,
            "relation_weighted_rmse": self.relation.weighted_rmse,
            "relation_weighted_mae": self.relation.weighted_mae,
            "relation_coverage": self.relation.coverage,

            # v3: the primary consistency keys are interpretation-safe.
            # A connected tree has no over-identifying cycle restriction, so
            # these values become NaN rather than a misleading structural 0.
            "consistency_weighted_rmse": (
                self.consistency.cycle_consistency_weighted_rmse
            ),
            "consistency_normalized_error": (
                self.consistency.cycle_consistency_normalized_error
            ),

            # Raw projection residuals are retained separately for diagnostics.
            "consistency_raw_weighted_rmse": self.consistency.weighted_rmse,
            "consistency_raw_normalized_error": (
                self.consistency.normalized_consistency_error
            ),
            "consistency_cycle_rank": float(self.consistency.cycle_rank),
            "consistency_testable": float(
                self.consistency.consistency_testable
            ),
        }


@dataclass
class BackendComparison:
    """Direct comparison of two relation/pipeline backends."""

    n_common_edges: int

    relation_rmse_between_backends: float
    relation_mae_between_backends: float
    weight_rmse_between_backends: float
    weight_mae_between_backends: float

    gamma_l2_difference: float
    gamma_mae_difference: float

    forecast_l2_difference: float
    forecast_mae_difference: float

    magnitude_absolute_difference: float


# ============================================================
# Shared validation / numerical helpers
# ============================================================


def _as_float_array(
    x: np.ndarray,
    name: str,
    *,
    ndim: Optional[int] = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if ndim is not None and x.ndim != ndim:
        raise ValueError(
            f"{name} must have ndim={ndim}; got shape {x.shape}."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must contain only finite values.")

    return x


def _validate_system_vector(
    x: np.ndarray,
    name: str,
) -> np.ndarray:
    x = _as_float_array(x, name, ndim=1)

    if len(x) < 2:
        raise ValueError(f"{name} must contain at least two systems.")

    return x


def _safe_divide(
    numerator: np.ndarray | float,
    denominator: np.ndarray | float,
    epsilon: float,
) -> np.ndarray:
    denominator_array = np.asarray(denominator, dtype=float)
    sign = np.where(denominator_array < 0, -1.0, 1.0)
    safe = np.where(
        np.abs(denominator_array) > epsilon,
        denominator_array,
        sign * epsilon,
    )
    return np.asarray(numerator, dtype=float) / safe


def _weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    total = float(np.sum(weights))
    if total <= 0:
        return np.nan

    return float(np.sum(weights * values) / total)


def _jensen_shannon_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float,
) -> float:
    """Natural-log Jensen-Shannon divergence for probability vectors."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)

    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))

    return 0.5 * (kl_pm + kl_qm)


# ============================================================
# 1. Final system forecast evaluation
# ============================================================


def evaluate_system_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> ForecastEvaluation:
    """Evaluate final reconstructed system forecasts at one horizon."""

    y_true = _validate_system_vector(y_true, "y_true")
    y_pred = _validate_system_vector(y_pred, "y_pred")

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; "
            f"got {y_true.shape} and {y_pred.shape}."
        )

    error = y_pred - y_true
    abs_error = np.abs(error)

    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_error))
    bias = float(np.mean(error))

    # MAPE ignores actual values too close to zero rather than allowing the
    # metric to explode for structural zeros.
    nonzero = np.abs(y_true) > epsilon
    if np.any(nonzero):
        mape = float(
            100.0 * np.mean(abs_error[nonzero] / np.abs(y_true[nonzero]))
        )
    else:
        mape = np.nan

    smape_den = np.abs(y_true) + np.abs(y_pred)
    valid_smape = smape_den > epsilon
    if np.any(valid_smape):
        smape = float(
            200.0
            * np.mean(abs_error[valid_smape] / smape_den[valid_smape])
        )
    else:
        smape = 0.0

    actual_abs_sum = float(np.sum(np.abs(y_true)))
    if actual_abs_sum > epsilon:
        wape = float(100.0 * np.sum(abs_error) / actual_abs_sum)
    else:
        wape = np.nan

    return ForecastEvaluation(
        n_values=int(y_true.size),
        mse=mse,
        rmse=rmse,
        mae=mae,
        bias=bias,

        mape_percent=mape,
        smape_percent=smape,
        wape_percent=wape,

        max_absolute_error=float(np.max(abs_error)),

        actual_total=float(np.sum(y_true)),
        forecast_total=float(np.sum(y_pred)),
        total_error=float(np.sum(y_pred) - np.sum(y_true)),

        per_system_absolute_error=np.asarray(abs_error, dtype=float),
    )


# ============================================================
# 2. Magnitude predictor evaluation
# ============================================================


def evaluate_magnitude(
    y_true: np.ndarray,
    predicted_magnitude: float,
    *,
    epsilon: float = 1e-8,
) -> MagnitudeEvaluation:
    """Evaluate M_hat against the realized total sum_i y_i."""

    y_true = _validate_system_vector(y_true, "y_true")

    predicted = float(predicted_magnitude)
    if not np.isfinite(predicted):
        raise ValueError("predicted_magnitude must be a finite scalar.")

    actual = float(np.sum(y_true))
    error = predicted - actual
    absolute_error = abs(error)

    if abs(actual) > epsilon:
        percentage_error = float(100.0 * error / actual)
        ape = float(100.0 * absolute_error / abs(actual))
    else:
        percentage_error = np.nan
        ape = np.nan

    return MagnitudeEvaluation(
        actual_magnitude=actual,
        predicted_magnitude=predicted,

        error=float(error),
        absolute_error=float(absolute_error),
        squared_error=float(error**2),

        percentage_error_percent=percentage_error,
        absolute_percentage_error_percent=ape,
    )


# ============================================================
# 3. Equilibrium-state evaluation
# ============================================================


def realized_equilibrium_state(
    y_true: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Convert positive realized system values into simplex shares gamma_true."""

    y_true = _validate_system_vector(y_true, "y_true")
    _validate_representation_observations(
        y_true, LOG_RATIO, epsilon=epsilon, name="y_true"
    )
    total = float(np.sum(y_true))

    if total <= epsilon:
        raise ValueError(
            "Realized total magnitude is zero, so gamma_true is undefined."
        )

    return y_true / total


def evaluate_equilibrium_state(
    y_true: np.ndarray,
    state_pred: np.ndarray,
    *,
    representation: str = LOG_RATIO,
    epsilon: float = 1e-12,
) -> EquilibriumEvaluation:
    """Evaluate the representation-specific global equilibrium state.

    ``log_ratio`` compares simplex shares gamma.
    Non-log geometries compare zero-sum transformed latent coordinates u.  The
    result container retains the historical ``gamma_*`` field names for API
    compatibility; in those modes the arrays contain latent coordinates and Jensen-
    Shannon divergence is not applicable (NaN).
    """
    representation = normalize_representation_name(representation)
    y_true = _validate_system_vector(y_true, "y_true")

    state_pred = _as_float_array(
        state_pred,
        "state_pred",
        ndim=1,
    )
    if state_pred.shape != y_true.shape:
        raise ValueError(
            f"state_pred must have shape {y_true.shape}; got {state_pred.shape}."
        )

    if representation == LOG_RATIO:
        state_true = realized_equilibrium_state(y_true, epsilon=epsilon)
        if np.any(state_pred < 0):
            raise ValueError("log_ratio equilibrium shares must be non-negative.")
        pred_sum = float(np.sum(state_pred))
        if pred_sum <= epsilon:
            raise ValueError("Predicted equilibrium shares must have a positive sum.")
        state_pred_norm = state_pred / pred_sum
        js = _jensen_shannon_divergence(
            state_true, state_pred_norm, epsilon
        )
    else:
        state_true = centered_latent_state(
            y_true, representation, epsilon=epsilon
        )
        # The graph solver uses a zero-sum gauge.  Re-centering makes this
        # evaluator robust to any numerically equivalent additive gauge.
        state_pred_norm = state_pred - float(np.mean(state_pred))
        js = np.nan

    error = state_pred_norm - state_true
    abs_error = np.abs(error)

    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_error))

    denominator = float(
        np.linalg.norm(state_true) * np.linalg.norm(state_pred_norm)
    )
    cosine = (
        float(np.dot(state_true, state_pred_norm) / denominator)
        if denominator > epsilon
        else np.nan
    )

    return EquilibriumEvaluation(
        gamma_true=np.asarray(state_true, dtype=float),
        gamma_pred=np.asarray(state_pred_norm, dtype=float),
        mse=mse,
        rmse=rmse,
        mae=mae,
        l1_distance=float(np.sum(abs_error)),
        max_absolute_error=float(np.max(abs_error)),
        cosine_similarity=cosine,
        jensen_shannon_divergence=float(js),
        sum_true=float(np.sum(state_true)),
        sum_pred=float(np.sum(state_pred_norm)),
        representation=representation,
    )


# ============================================================
# 4. Pairwise relation evaluation
# ============================================================


def realized_relation_matrix(
    y_true: np.ndarray,
    *,
    representation: str = LOG_RATIO,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Realized pairwise relation matrix in the selected relation coordinate."""
    y_true = _validate_system_vector(y_true, "y_true")
    return _representation_realized_relation_matrix(
        y_true, representation, epsilon=epsilon
    )


def evaluate_relation_state(
    relation_state: RelationMatrixState,
    y_true: np.ndarray,
    *,
    use_base: bool = False,
    weight_threshold: float = 0.0,
    sign_tolerance: float = 1e-8,
    epsilon: float = 1e-8,
    representation: str = LOG_RATIO,
) -> RelationEvaluation:
    """Evaluate predicted pairwise relations against realized future relations."""

    if not isinstance(relation_state, RelationMatrixState):
        raise TypeError("relation_state must be a RelationMatrixState.")

    representation = normalize_representation_name(representation)
    y_true = _validate_system_vector(y_true, "y_true")

    if len(y_true) != relation_state.n_systems:
        raise ValueError(
            "y_true length must match relation_state.n_systems."
        )

    if weight_threshold < 0:
        raise ValueError("weight_threshold must be non-negative.")

    true_R = realized_relation_matrix(
        y_true,
        representation=representation,
        epsilon=epsilon,
    )

    edge_index, edge_r, edge_weight, edge_valid = relation_state.edge_arrays(
        use_base=use_base,
        valid_only=False,
    )

    used = (
        edge_valid
        & np.isfinite(edge_r)
        & np.isfinite(edge_weight)
        & (edge_weight > float(weight_threshold))
    )

    n_evaluated = int(np.sum(used))

    if n_evaluated == 0:
        return RelationEvaluation(
            n_possible_edges=relation_state.n_possible_edges,
            n_valid_edges=relation_state.n_valid_edges,
            n_evaluated_edges=0,
            coverage=0.0,

            mse=np.nan,
            rmse=np.nan,
            mae=np.nan,

            weighted_mse=np.nan,
            weighted_rmse=np.nan,
            weighted_mae=np.nan,

            bias=np.nan,
            weighted_bias=np.nan,

            sign_accuracy=np.nan,
            weighted_sign_accuracy=np.nan,

            mean_weight=0.0,
            total_weight=0.0,
        )

    used_index = edge_index[:, used]
    predicted = np.asarray(edge_r[used], dtype=float)
    weights = np.asarray(edge_weight[used], dtype=float)

    actual = np.asarray(
        [
            true_R[int(i), int(j)]
            for i, j in used_index.T
        ],
        dtype=float,
    )

    error = predicted - actual
    abs_error = np.abs(error)

    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_error))
    bias = float(np.mean(error))

    total_weight = float(np.sum(weights))

    if total_weight > 0:
        weighted_mse = _weighted_mean(error**2, weights)
        weighted_rmse = float(np.sqrt(weighted_mse))
        weighted_mae = _weighted_mean(abs_error, weights)
        weighted_bias = _weighted_mean(error, weights)
    else:
        weighted_mse = np.nan
        weighted_rmse = np.nan
        weighted_mae = np.nan
        weighted_bias = np.nan

    # Sign accuracy is meaningful only when the realized relation is not
    # numerically indistinguishable from equality.
    sign_mask = np.abs(actual) > sign_tolerance

    if np.any(sign_mask):
        correct_sign = (
            np.sign(predicted[sign_mask]) == np.sign(actual[sign_mask])
        ).astype(float)

        sign_accuracy = float(np.mean(correct_sign))
        sign_weights = weights[sign_mask]
        weighted_sign_accuracy = (
            _weighted_mean(correct_sign, sign_weights)
            if float(np.sum(sign_weights)) > 0
            else np.nan
        )
    else:
        sign_accuracy = np.nan
        weighted_sign_accuracy = np.nan

    return RelationEvaluation(
        n_possible_edges=relation_state.n_possible_edges,
        n_valid_edges=relation_state.n_valid_edges,
        n_evaluated_edges=n_evaluated,
        coverage=float(
            n_evaluated / relation_state.n_possible_edges
        ),

        mse=mse,
        rmse=rmse,
        mae=mae,

        weighted_mse=float(weighted_mse),
        weighted_rmse=float(weighted_rmse),
        weighted_mae=float(weighted_mae),

        bias=bias,
        weighted_bias=float(weighted_bias),

        sign_accuracy=float(sign_accuracy),
        weighted_sign_accuracy=float(weighted_sign_accuracy),

        mean_weight=float(np.mean(weights)),
        total_weight=total_weight,

        true_edge_relations=np.asarray(actual, dtype=float),
        predicted_edge_relations=np.asarray(predicted, dtype=float),
        edge_weights=np.asarray(weights, dtype=float),
        edge_index=np.asarray(used_index, dtype=int),
    )


# ============================================================
# 5. Internal global-consistency evaluation
# ============================================================


def evaluate_global_consistency(
    equilibrium: EquilibriumResult,
) -> ConsistencyEvaluation:
    """Evaluate internal relation consistency with cycle-rank awareness.

    The ES solver always returns projection residuals between the observed edge
    relations and the closest globally representable relation matrix.  Those
    residuals are numerically valid for any solved graph.

    Their *consistency-test* interpretation, however, requires redundant cycle
    constraints.  Let

        cycle_rank = E - N + C,

    with E used edges, N systems, and C connected components.

    For a connected tree, E=N-1 and cycle_rank=0.  In that case arbitrary edge
    values can be represented exactly by node potentials, so a zero residual is
    structural rather than evidence of cross-edge agreement.

    v3 therefore:
      1. preserves all raw solver residual diagnostics;
      2. reports cycle_rank and whether cycle redundancy exists;
      3. sets the interpretation-safe cycle consistency scores to NaN unless
         the graph is connected and cycle_rank > 0.
    """

    if not isinstance(equilibrium, EquilibriumResult):
        raise TypeError("equilibrium must be an EquilibriumResult.")

    diagnostics = equilibrium.diagnostics

    n_systems = int(equilibrium.n_systems)
    n_used_edges = int(equilibrium.n_used_edges)
    n_components = int(len(equilibrium.components))

    cycle_rank = int(
        n_used_edges
        - n_systems
        + n_components
    )

    # For any undirected graph, E - N + C cannot be negative.  A negative
    # value would indicate inconsistent solver/component bookkeeping.
    if cycle_rank < 0:
        raise RuntimeError(
            "Invalid relation-graph cycle rank: "
            f"E={n_used_edges}, N={n_systems}, C={n_components}, "
            f"E-N+C={cycle_rank}."
        )

    has_cycle_redundancy = bool(cycle_rank > 0)

    # The global inconsistency residual is used as an over-identifying
    # relation-conflict diagnostic only when one global ES is identified and
    # at least one independent cycle exists.
    consistency_testable = bool(
        equilibrium.connected
        and equilibrium.globally_identified
        and has_cycle_redundancy
    )

    raw_weighted_rmse = float(equilibrium.weighted_rmse)
    raw_normalized_error = float(
        equilibrium.normalized_consistency_error
    )

    if consistency_testable:
        cycle_weighted_rmse = raw_weighted_rmse
        cycle_normalized_error = raw_normalized_error
    else:
        cycle_weighted_rmse = np.nan
        cycle_normalized_error = np.nan

    return ConsistencyEvaluation(
        connected=bool(equilibrium.connected),
        globally_identified=bool(equilibrium.globally_identified),

        n_systems=n_systems,
        n_used_edges=n_used_edges,
        n_components=n_components,

        cycle_rank=cycle_rank,
        has_cycle_redundancy=has_cycle_redundancy,
        consistency_testable=consistency_testable,

        weighted_sse=float(equilibrium.weighted_sse),
        weighted_mse=float(equilibrium.weighted_mse),
        weighted_rmse=raw_weighted_rmse,
        normalized_consistency_error=raw_normalized_error,

        mean_absolute_residual=float(
            equilibrium.mean_absolute_residual
        ),
        max_absolute_residual=float(
            equilibrium.max_absolute_residual
        ),

        cycle_consistency_weighted_rmse=float(
            cycle_weighted_rmse
        ),
        cycle_consistency_normalized_error=float(
            cycle_normalized_error
        ),

        edge_density_used=float(
            diagnostics.get("edge_density_used", np.nan)
        ),
        algebraic_connectivity=float(
            diagnostics.get("algebraic_connectivity", np.nan)
        ),
        spectral_condition_number=float(
            diagnostics.get("spectral_condition_number", np.nan)
        ),
    )


# ============================================================
# 6. Complete pipeline evaluation
# ============================================================


def evaluate_pipeline(
    pipeline_result: Any,
    y_true: np.ndarray,
    *,
    relation_use_base: bool = False,
    relation_weight_threshold: float = 0.0,
    epsilon: float = 1e-8,
    metadata: Optional[Mapping[str, Any]] = None,
) -> PipelineEvaluation:
    """Evaluate one complete ESE pipeline result.

    `pipeline_result` is intentionally duck-typed so this module does not import
    `demo_v1.py` and create a circular dependency.  It must expose

        relation_state
        equilibrium
        predictor
        system_forecast

    which is exactly the interface of DemoPipelineResult.
    """

    required = (
        "relation_state",
        "equilibrium",
        "predictor",
        "system_forecast",
    )
    missing = [
        name for name in required
        if not hasattr(pipeline_result, name)
    ]
    if missing:
        raise TypeError(
            "pipeline_result is missing required attributes: "
            + ", ".join(missing)
        )

    y_true = _validate_system_vector(y_true, "y_true")

    relation_state = pipeline_result.relation_state
    equilibrium = pipeline_result.equilibrium
    predictor = pipeline_result.predictor
    y_pred = np.asarray(
        pipeline_result.system_forecast,
        dtype=float,
    )

    if not isinstance(relation_state, RelationMatrixState):
        raise TypeError(
            "pipeline_result.relation_state must be RelationMatrixState."
        )

    if not isinstance(equilibrium, EquilibriumResult):
        raise TypeError(
            "pipeline_result.equilibrium must be EquilibriumResult."
        )

    if not isinstance(predictor, PredictorResult):
        raise TypeError(
            "pipeline_result.predictor must be PredictorResult."
        )

    if len(y_true) != relation_state.n_systems:
        raise ValueError(
            "y_true length does not match the number of systems."
        )

    representation = normalize_representation_name(
        getattr(equilibrium, "representation", LOG_RATIO)
    )

    report_metadata = dict(metadata or {})
    report_metadata.setdefault("representation", representation)
    if hasattr(pipeline_result, "mode"):
        report_metadata.setdefault(
            "relation_mode",
            str(pipeline_result.mode),
        )

    report_metadata.setdefault(
        "system_names",
        list(relation_state.system_names),
    )
    report_metadata.setdefault(
        "relation_iteration",
        int(relation_state.iteration),
    )
    report_metadata.setdefault(
        "predictor_family",
        str(predictor.selected_family),
    )
    report_metadata.setdefault(
        "predictor_model",
        str(predictor.selected_model),
    )

    return PipelineEvaluation(
        horizon=int(predictor.horizon),

        forecast=evaluate_system_forecast(
            y_true,
            y_pred,
            epsilon=epsilon,
        ),

        magnitude=evaluate_magnitude(
            y_true,
            predictor.magnitude,
            epsilon=epsilon,
        ),

        equilibrium=evaluate_equilibrium_state(
            y_true,
            equilibrium.gamma if representation == LOG_RATIO else equilibrium.u,
            representation=representation,
            epsilon=epsilon,
        ),

        relation=evaluate_relation_state(
            relation_state,
            y_true,
            use_base=relation_use_base,
            weight_threshold=relation_weight_threshold,
            epsilon=epsilon,
            representation=representation,
        ),

        consistency=evaluate_global_consistency(
            equilibrium
        ),

        metadata=report_metadata,
    )


# ============================================================
# 7. Statistical-vs-neural / backend comparison
# ============================================================


def compare_backends(
    result_a: Any,
    result_b: Any,
) -> BackendComparison:
    """Compare two complete pipeline results without requiring future truth.

    Typical use:
        statistical result  vs  neural result.

    Only pairwise edges valid in BOTH relation states are compared.
    """

    for label, result in (("result_a", result_a), ("result_b", result_b)):
        for attr in (
            "relation_state",
            "equilibrium",
            "predictor",
            "system_forecast",
        ):
            if not hasattr(result, attr):
                raise TypeError(
                    f"{label} is missing required attribute '{attr}'."
                )

    state_a = result_a.relation_state
    state_b = result_b.relation_state

    if state_a.n_systems != state_b.n_systems:
        raise ValueError(
            "Both backends must contain the same number of systems."
        )

    idx_a, r_a, w_a, valid_a = state_a.edge_arrays(
        valid_only=False
    )
    idx_b, r_b, w_b, valid_b = state_b.edge_arrays(
        valid_only=False
    )

    if not np.array_equal(idx_a, idx_b):
        raise ValueError(
            "Relation states use different pair ordering."
        )

    common = (
        valid_a
        & valid_b
        & np.isfinite(r_a)
        & np.isfinite(r_b)
    )

    n_common = int(np.sum(common))

    if n_common:
        relation_diff = r_a[common] - r_b[common]
        weight_diff = w_a[common] - w_b[common]

        relation_rmse = float(
            np.sqrt(np.mean(relation_diff**2))
        )
        relation_mae = float(
            np.mean(np.abs(relation_diff))
        )
        weight_rmse = float(
            np.sqrt(np.mean(weight_diff**2))
        )
        weight_mae = float(
            np.mean(np.abs(weight_diff))
        )
    else:
        relation_rmse = np.nan
        relation_mae = np.nan
        weight_rmse = np.nan
        weight_mae = np.nan

    rep_a = normalize_representation_name(
        getattr(result_a.equilibrium, "representation", LOG_RATIO)
    )
    rep_b = normalize_representation_name(
        getattr(result_b.equilibrium, "representation", LOG_RATIO)
    )
    state_a = (
        result_a.equilibrium.gamma if rep_a == LOG_RATIO else result_a.equilibrium.u
    )
    state_b = (
        result_b.equilibrium.gamma if rep_b == LOG_RATIO else result_b.equilibrium.u
    )
    gamma_a = np.asarray(state_a, dtype=float)
    gamma_b = np.asarray(state_b, dtype=float)
    forecast_a = np.asarray(result_a.system_forecast, dtype=float)
    forecast_b = np.asarray(result_b.system_forecast, dtype=float)

    return BackendComparison(
        n_common_edges=n_common,

        relation_rmse_between_backends=relation_rmse,
        relation_mae_between_backends=relation_mae,
        weight_rmse_between_backends=weight_rmse,
        weight_mae_between_backends=weight_mae,

        gamma_l2_difference=float(
            np.linalg.norm(gamma_a - gamma_b)
        ),
        gamma_mae_difference=float(
            np.mean(np.abs(gamma_a - gamma_b))
        ),

        forecast_l2_difference=float(
            np.linalg.norm(forecast_a - forecast_b)
        ),
        forecast_mae_difference=float(
            np.mean(np.abs(forecast_a - forecast_b))
        ),

        magnitude_absolute_difference=float(
            abs(
                float(result_a.predictor.magnitude)
                - float(result_b.predictor.magnitude)
            )
        ),
    )


# ============================================================
# 8. Rolling-origin aggregation
# ============================================================


def aggregate_pipeline_evaluations(
    evaluations: Sequence[PipelineEvaluation],
) -> Dict[str, Dict[str, float]]:
    """Aggregate repeated one-origin reports.

    Returns
    -------
    dict
        For every scalar metric in `PipelineEvaluation.summary()`, returns

            mean
            std
            median
            min
            max
            count

        NaN values are ignored.
    """

    evaluations = list(evaluations)
    if not evaluations:
        raise ValueError(
            "At least one PipelineEvaluation is required."
        )

    keys = list(evaluations[0].summary().keys())

    rows: Dict[str, List[float]] = {
        key: [] for key in keys
    }

    for report in evaluations:
        summary = report.summary()

        if set(summary.keys()) != set(keys):
            raise ValueError(
                "All evaluation summaries must expose the same metric keys."
            )

        for key in keys:
            rows[key].append(float(summary[key]))

    out: Dict[str, Dict[str, float]] = {}

    for key, values in rows.items():
        x = np.asarray(values, dtype=float)
        finite = x[np.isfinite(x)]

        if finite.size == 0:
            out[key] = {
                "mean": np.nan,
                "std": np.nan,
                "median": np.nan,
                "min": np.nan,
                "max": np.nan,
                "count": 0.0,
            }
            continue

        out[key] = {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite, ddof=0)),
            "median": float(np.median(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "count": float(finite.size),
        }

    return out


# ============================================================
# 9. Human-readable reporting
# ============================================================


def _fmt(value: float, digits: int = 6) -> str:
    value = float(value)
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "+Inf"
    if np.isneginf(value):
        return "-Inf"
    return f"{value:.{digits}f}"


def print_evaluation(
    report: PipelineEvaluation,
    *,
    title: str = "ESE Evaluation",
) -> None:
    """Print a structured one-origin evaluation report."""

    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"horizon: {report.horizon}")

    if report.metadata.get("relation_mode") is not None:
        print(
            "relation backend: "
            f"{report.metadata['relation_mode']}"
        )

    print()
    print("[1] Final system forecast")
    print(
        f"  RMSE                   : "
        f"{_fmt(report.forecast.rmse)}"
    )
    print(
        f"  MAE                    : "
        f"{_fmt(report.forecast.mae)}"
    )
    print(
        f"  Bias                   : "
        f"{_fmt(report.forecast.bias)}"
    )
    print(
        f"  MAPE (%)               : "
        f"{_fmt(report.forecast.mape_percent)}"
    )
    print(
        f"  sMAPE (%)              : "
        f"{_fmt(report.forecast.smape_percent)}"
    )
    print(
        f"  WAPE (%)               : "
        f"{_fmt(report.forecast.wape_percent)}"
    )

    print()
    print("[2] Overall magnitude")
    print(
        f"  Actual M               : "
        f"{_fmt(report.magnitude.actual_magnitude)}"
    )
    print(
        f"  Predicted M            : "
        f"{_fmt(report.magnitude.predicted_magnitude)}"
    )
    print(
        f"  Absolute error         : "
        f"{_fmt(report.magnitude.absolute_error)}"
    )
    print(
        f"  APE (%)                : "
        f"{_fmt(report.magnitude.absolute_percentage_error_percent)}"
    )

    print()
    if report.equilibrium.representation == LOG_RATIO:
        print("[3] Equilibrium allocation (gamma)")
        state_label = "gamma"
    else:
        print("[3] Equilibrium latent state (additive u)")
        state_label = "state"
    print(
        f"  {state_label + ' RMSE':<23}: "
        f"{_fmt(report.equilibrium.rmse)}"
    )
    print(
        f"  {state_label + ' MAE':<23}: "
        f"{_fmt(report.equilibrium.mae)}"
    )
    print(
        f"  {state_label + ' L1':<23}: "
        f"{_fmt(report.equilibrium.l1_distance)}"
    )
    if report.equilibrium.representation == LOG_RATIO:
        print(
            f"  Jensen-Shannon div.    : "
            f"{_fmt(report.equilibrium.jensen_shannon_divergence)}"
        )
    print(
        f"  Cosine similarity      : "
        f"{_fmt(report.equilibrium.cosine_similarity)}"
    )

    print()
    print("[4] Pairwise relation accuracy")
    print(
        f"  evaluated edges        : "
        f"{report.relation.n_evaluated_edges}/"
        f"{report.relation.n_possible_edges}"
    )
    print(
        f"  coverage               : "
        f"{_fmt(report.relation.coverage)}"
    )
    print(
        f"  relation RMSE          : "
        f"{_fmt(report.relation.rmse)}"
    )
    print(
        f"  relation MAE           : "
        f"{_fmt(report.relation.mae)}"
    )
    print(
        f"  weighted relation RMSE : "
        f"{_fmt(report.relation.weighted_rmse)}"
    )
    print(
        f"  weighted relation MAE  : "
        f"{_fmt(report.relation.weighted_mae)}"
    )
    print(
        f"  sign accuracy          : "
        f"{_fmt(report.relation.sign_accuracy)}"
    )

    print()
    print("[5] Internal global consistency")
    print(
        f"  connected              : "
        f"{report.consistency.connected}"
    )
    print(
        f"  globally identified    : "
        f"{report.consistency.globally_identified}"
    )
    print(
        f"  systems                : "
        f"{report.consistency.n_systems}"
    )
    print(
        f"  used edges             : "
        f"{report.consistency.n_used_edges}"
    )
    print(
        f"  components             : "
        f"{report.consistency.n_components}"
    )
    print(
        f"  cycle rank (E-N+C)     : "
        f"{report.consistency.cycle_rank}"
    )
    print(
        f"  cycle redundancy       : "
        f"{report.consistency.has_cycle_redundancy}"
    )
    print(
        f"  consistency testable   : "
        f"{report.consistency.consistency_testable}"
    )

    if report.consistency.consistency_testable:
        print(
            f"  weighted residual RMSE : "
            f"{_fmt(report.consistency.cycle_consistency_weighted_rmse)}"
        )
        print(
            f"  normalized inconsistency: "
            f"{_fmt(report.consistency.cycle_consistency_normalized_error)}"
        )
    else:
        print(
            f"  raw residual RMSE      : "
            f"{_fmt(report.consistency.weighted_rmse)} "
            f"(diagnostic only)"
        )
        print(
            f"  raw normalized residual: "
            f"{_fmt(report.consistency.normalized_consistency_error)} "
            f"(diagnostic only)"
        )
        print(
            "  consistency score      : N/A"
        )

        if (
            report.consistency.connected
            and report.consistency.cycle_rank == 0
        ):
            print(
                "  interpretation         : no independent cycle redundancy; "
                "a zero residual is structural and is not evidence of "
                "cross-relation consistency."
            )
        elif not report.consistency.connected:
            print(
                "  interpretation         : the used relation graph is "
                "disconnected, so one global consistency test is not reported."
            )

    print(
        f"  max |residual|         : "
        f"{_fmt(report.consistency.max_absolute_residual)}"
    )


def print_backend_comparison(
    comparison: BackendComparison,
    *,
    name_a: str = "A",
    name_b: str = "B",
) -> None:
    """Print statistical-vs-neural or any two-backend agreement report."""

    print("=" * 72)
    print(f"Backend comparison: {name_a} vs {name_b}")
    print("=" * 72)

    print(
        f"common relation edges        : "
        f"{comparison.n_common_edges}"
    )
    print(
        f"relation RMSE                : "
        f"{_fmt(comparison.relation_rmse_between_backends)}"
    )
    print(
        f"relation MAE                 : "
        f"{_fmt(comparison.relation_mae_between_backends)}"
    )
    print(
        f"weight RMSE                  : "
        f"{_fmt(comparison.weight_rmse_between_backends)}"
    )
    print(
        f"gamma L2 difference          : "
        f"{_fmt(comparison.gamma_l2_difference)}"
    )
    print(
        f"forecast L2 difference       : "
        f"{_fmt(comparison.forecast_l2_difference)}"
    )
    print(
        f"magnitude absolute difference: "
        f"{_fmt(comparison.magnitude_absolute_difference)}"
    )


