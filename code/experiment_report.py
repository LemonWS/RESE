"""
experiment_report_v6.py
=======================

Experiment timing and reporting utilities for the history-driven ESE project.

Separation of responsibilities
------------------------------
evaluation_v6.py
    Defines accuracy / state / relation / consistency metrics.

experiment_report_v2.py
    Defines wall-clock timing containers and human-readable experiment reports.

demo / experiment runner
    Executes the algorithms and delegates timing/report presentation here.

This keeps methodology modules free from printing/profiling logic and allows the
same reporting code to be reused by demos, rolling-origin runners, statistical
vs neural comparisons, and future benchmark scripts.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from evaluation import PipelineEvaluation, print_evaluation


# ============================================================
# Generic wall-clock timer
# ============================================================


class WallClock:
    """Small context-manager stopwatch based on time.perf_counter()."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: Optional[float] = None

    def __enter__(self) -> "WallClock":
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._start is not None:
            self.elapsed = float(perf_counter() - self._start)


# ============================================================
# Timing result
# ============================================================


@dataclass
class TimingReport:
    """Wall-clock timings for one ESE forecast.

    Stage timings
    -------------
    data_loading
        File -> dates/Y/system names.

    preparation
        Chronological holdout / forecast-origin preparation.

    relation_estimation
        Statistical pairwise relation fitting or neural relation inference.

    relation_matrix
        Construction of RelationMatrixState when this is a separate operation.

    equilibrium_solver
        (R, W) -> gamma*.

    magnitude_predictor
        M history -> M_hat.

    reconstruction
        representation-aware reconstruction -> system forecast.

    evaluation
        Formal metrics from evaluation_v6.py.

    workflow_total
        Full computational workflow from the beginning of data loading through
        completion of evaluation. This may include small orchestration overheads.

    end_to_end_total
        Optional outer runtime assigned by a demo/runner after normal reporting.
        It can therefore include console reporting as well.

    pair_relation_times
        Optional individual statistical pair times.

    Notes
    -----
    `algorithm_total` is a derived quantity and deliberately excludes:
        data loading,
        holdout preparation,
        evaluation,
        printing/reporting.

    This is normally the most appropriate timing quantity for a paper's model
    runtime comparison.
    """

    data_loading: float = 0.0
    preparation: float = 0.0

    relation_estimation: float = 0.0
    relation_matrix: float = 0.0
    equilibrium_solver: float = 0.0
    magnitude_predictor: float = 0.0
    reconstruction: float = 0.0

    evaluation: float = 0.0

    workflow_total: float = 0.0
    end_to_end_total: Optional[float] = None

    pair_relation_times: List[float] = field(default_factory=list)

    @property
    def algorithm_total(self) -> float:
        """Forecasting/model runtime excluding I/O, evaluation, and reporting."""
        return float(
            self.relation_estimation
            + self.relation_matrix
            + self.equilibrium_solver
            + self.magnitude_predictor
            + self.reconstruction
        )

    @contextmanager
    def measure(self, stage: str) -> Iterator[WallClock]:
        """Measure and store one named TimingReport field.

        Example
        -------
        with timing.measure("equilibrium_solver"):
            equilibrium = solve_equilibrium_state(...)
        """
        if not hasattr(self, stage):
            raise AttributeError(f"Unknown timing stage: {stage!r}")

        timer = WallClock()
        with timer:
            yield timer

        current = getattr(self, stage)

        if current is None:
            current = 0.0

        if not isinstance(current, (int, float, np.floating)):
            raise TypeError(
                f"Timing field {stage!r} is not a scalar timing field."
            )

        setattr(self, stage, float(current) + timer.elapsed)

    @contextmanager
    def measure_pair(self) -> Iterator[WallClock]:
        """Measure one pairwise relation computation and append its duration."""
        timer = WallClock()
        with timer:
            yield timer
        self.pair_relation_times.append(float(timer.elapsed))

    def as_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "data_loading": float(self.data_loading),
            "preparation": float(self.preparation),
            "relation_estimation": float(self.relation_estimation),
            "relation_matrix": float(self.relation_matrix),
            "equilibrium_solver": float(self.equilibrium_solver),
            "magnitude_predictor": float(self.magnitude_predictor),
            "reconstruction": float(self.reconstruction),
            "evaluation": float(self.evaluation),
            "algorithm_total": float(self.algorithm_total),
            "workflow_total": float(self.workflow_total),
        }

        if self.end_to_end_total is not None:
            out["end_to_end_total"] = float(self.end_to_end_total)

        if self.pair_relation_times:
            x = np.asarray(self.pair_relation_times, dtype=float)
            out.update(
                {
                    "pair_count": float(len(x)),
                    "pair_total": float(np.sum(x)),
                    "pair_mean": float(np.mean(x)),
                    "pair_median": float(np.median(x)),
                    "pair_min": float(np.min(x)),
                    "pair_max": float(np.max(x)),
                }
            )

        return out


# ============================================================
# Forecast reporting
# ============================================================


def print_forecast_report(
    result: Any,
    *,
    include_actual: bool = True,
    title: str = "Forecast result",
) -> None:
    """Print the model-side forecast result.

    The object is intentionally duck-typed. Expected attributes:
        relation_mode
        system_names
        horizon
        target_date
        equilibrium.gamma
        predictor.selected_family
        predictor.selected_model
        predictor.magnitude
        system_forecast

    Optional:
        y_true
        relation_results
    """

    print()
    print("=" * 76)
    print(title)
    print("=" * 76)

    input_length = getattr(result, "input_length", None)
    output_length = int(getattr(result, "output_length", getattr(result, "horizon", 1)))
    history_mode = getattr(result, "history_mode", None)
    if hasattr(result, "Y_input"):
        effective_L = len(result.Y_input)
    elif input_length is not None:
        effective_L = int(input_length)
    else:
        effective_L = None

    if effective_L is not None:
        print(f"input length (L)          : {effective_L}")
    if history_mode is not None:
        print(f"history mode              : {history_mode}")
    print(f"output length / horizon   : {output_length}")

    relation_results = getattr(result, "relation_results", None)

    if relation_results:
        from collections import Counter

        family_counts = Counter(
            str(x.selected_family)
            for x in relation_results
        )

        print("selected relation families:")
        for family, count in sorted(family_counts.items()):
            print(f"  {family:<22} {count:>4}")

    print()
    representation = str(
        getattr(result.equilibrium, "representation", "log_ratio")
    )
    if representation == "additive":
        print("equilibrium latent state u* (zero-sum gauge):")
        state = np.asarray(result.equilibrium.u, dtype=float)
    else:
        print("equilibrium state gamma*:")
        state = np.asarray(result.equilibrium.gamma, dtype=float)

    for name, value in zip(result.system_names, state):
        print(f"  {name:<22}: {float(value): .8f}")

    print(f"  {'sum':<22}: {float(state.sum()): .8f}")

    print()
    print("overall magnitude:")
    print(
        f"  predictor              : "
        f"{result.predictor.selected_family}/"
        f"{result.predictor.selected_model}"
    )
    print(
        f"  M_hat(t+{result.horizon})"
        f"{' ' * max(1, 11-len(str(result.horizon)))}: "
        f"{float(result.predictor.magnitude):.8f}"
    )

    y_true = getattr(result, "y_true", None)

    if include_actual and y_true is not None:
        print(
            f"  actual M              : "
            f"{float(np.sum(y_true)):.8f}"
        )

    print()
    print(f"system forecasts at h={result.horizon}:")

    if include_actual and y_true is not None:
        for name, forecast, actual in zip(
            result.system_names,
            result.system_forecast,
            y_true,
        ):
            print(
                f"  {name:<18} "
                f"forecast={float(forecast): .8f}  "
                f"actual={float(actual): .8f}"
            )
    else:
        for name, forecast in zip(
            result.system_names,
            result.system_forecast,
        ):
            print(
                f"  {name:<22}: {float(forecast): .8f}"
            )


def print_final_prediction_values(
    result: Any,
    *,
    title: str = "FINAL PREDICTED VALUES",
) -> None:
    """Concise final h-step values intended to remain at the end of a demo."""

    forecast = np.asarray(result.system_forecast, dtype=float)

    print()
    print("=" * 76)
    print(title)
    print("=" * 76)
    print(
        f"Forecast target : t+{result.horizon} "
        f"({result.target_date})"
    )

    for name, value in zip(
        result.system_names,
        forecast,
    ):
        print(f"  {name:<22}: {float(value): .8f}")

    print("-" * 76)
    print(
        "Prediction vector : "
        + np.array2string(
            forecast,
            precision=8,
            separator=", ",
        )
    )
    print(
        f"Predicted total  : "
        f"{float(np.sum(forecast)):.8f}"
    )


# ============================================================
# Runtime reporting
# ============================================================


def print_timing_report(
    timing: TimingReport,
    *,
    include_end_to_end: bool = True,
) -> None:
    """Print detailed stage runtimes and the paper-oriented algorithm runtime."""

    print()
    print("=" * 76)
    print("Timing report (wall-clock seconds)")
    print("=" * 76)

    print(f"data loading              : {timing.data_loading:10.4f}")
    print(f"split / preparation       : {timing.preparation:10.4f}")
    print("-" * 76)
    print(f"relation estimation       : {timing.relation_estimation:10.4f}")
    print(f"relation-matrix build     : {timing.relation_matrix:10.4f}")
    print(f"equilibrium solver        : {timing.equilibrium_solver:10.4f}")
    print(f"magnitude predictor       : {timing.magnitude_predictor:10.4f}")
    print(f"reconstruction            : {timing.reconstruction:10.6f}")
    print("-" * 76)

    print(f"ALGORITHM TOTAL           : {timing.algorithm_total:10.4f}")

    print("-" * 76)
    print(f"evaluation                : {timing.evaluation:10.6f}")
    print(f"workflow total            : {timing.workflow_total:10.4f}")

    if timing.pair_relation_times:
        x = np.asarray(timing.pair_relation_times, dtype=float)

        print("-" * 76)
        print(f"number of pairs           : {len(x):10d}")
        print(f"pair total                : {np.sum(x):10.4f}")
        print(f"mean time / pair          : {np.mean(x):10.4f}")
        print(f"median time / pair        : {np.median(x):10.4f}")
        print(f"min time / pair           : {np.min(x):10.4f}")
        print(f"max time / pair           : {np.max(x):10.4f}")

    if include_end_to_end and timing.end_to_end_total is not None:
        print("-" * 76)
        print(
            f"END-TO-END TOTAL          : "
            f"{timing.end_to_end_total:10.4f}"
        )


def print_end_to_end_runtime(timing: TimingReport) -> None:
    """Print the final outer runtime after normal reporting has completed."""
    if timing.end_to_end_total is None:
        return

    print()
    print("=" * 76)
    print("TOTAL EXECUTION TIME")
    print("=" * 76)
    print(
        f"Algorithm runtime (model only)     : "
        f"{timing.algorithm_total:.4f} seconds"
    )
    print(
        f"Core workflow (load -> evaluation) : "
        f"{timing.workflow_total:.4f} seconds"
    )
    print(
        f"End-to-end incl. normal reporting  : "
        f"{timing.end_to_end_total:.4f} seconds"
    )
    print("=" * 76)


# ============================================================
# Unified experiment report
# ============================================================


def print_experiment_report(
    result: Any,
    *,
    evaluation: Optional[PipelineEvaluation] = None,
    timing: Optional[TimingReport] = None,
    include_forecast: bool = True,
    include_evaluation: bool = True,
    include_timing: bool = True,
    include_final_values: bool = True,
) -> None:
    """Print one complete experiment report without recomputing any metric."""

    if evaluation is None:
        evaluation = getattr(result, "evaluation", None)

    if timing is None:
        timing = getattr(result, "timing", None)

    if include_forecast:
        print_forecast_report(result)

    if include_evaluation and evaluation is not None:
        print()
        print_evaluation(
            evaluation,
            title=(
                f"ESE Evaluation "
                f"({getattr(result, 'relation_mode', 'unknown')}, file-driven)"
            ),
        )

    if include_timing and timing is not None:
        print_timing_report(
            timing,
            include_end_to_end=False,
        )

    if include_final_values:
        print_final_prediction_values(result)

# ============================================================
# Before/after comparison reporting
# ============================================================


@dataclass(frozen=True)
class BeforeAfterMetric:
    """Presentation-only before/after comparison.

    This does NOT define a new evaluation metric.  It only turns metrics already
    computed by evaluation_v6.py into an easier-to-read delta / improvement
    display.
    """

    name: str
    before: float
    after: float
    delta: float
    improvement_percent: float
    status: str


def compare_lower_is_better(
    name: str,
    before: float,
    after: float,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-9,
) -> BeforeAfterMetric:
    """Compare a metric such as RMSE where a smaller value is better."""
    before = float(before)
    after = float(after)
    delta = float(after - before)

    if np.isclose(before, after, atol=atol, rtol=rtol):
        status = "UNCHANGED"
    elif after < before:
        status = "IMPROVED"
    else:
        status = "WORSE"

    if np.isfinite(before) and abs(before) > 1e-15:
        improvement = float((before - after) / abs(before) * 100.0)
    else:
        improvement = np.nan

    return BeforeAfterMetric(
        name=str(name),
        before=before,
        after=after,
        delta=delta,
        improvement_percent=improvement,
        status=status,
    )


def _fmt_percent(x: float) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{x:+.3f}%"


def _feedback_trial_verdict(
    forecast_cmp: BeforeAfterMetric,
    relation_cmp: BeforeAfterMetric,
) -> str:
    if forecast_cmp.status == "IMPROVED":
        return "FORECAST IMPROVED"
    if forecast_cmp.status == "WORSE":
        return "FORECAST WORSE"
    if relation_cmp.status == "IMPROVED":
        return "NO FORECAST GAIN"
    return "NO FORECAST GAIN"


def print_feedback_experiment_report(
    *,
    base_evaluation: PipelineEvaluation,
    base_forecast: np.ndarray,
    system_names: List[str],
    experiments: List[Any],
    title: str = "ESE FEEDBACK RESULTS",
) -> None:
    """Print feedback results with explicit BEFORE / AFTER semantics.

    Expected experiment attributes
    ------------------------------
    alpha
    iterations
    feedback
    evaluation
    system_forecast
    relation_rmse_before / relation_rmse_after
    forecast_rmse_before / forecast_rmse_after
    forecast_change_max

    This function intentionally uses duck typing so the report module does not
    import a demo module and create a reverse dependency.
    """

    base_forecast = np.asarray(base_forecast, dtype=float)

    print()
    print("=" * 108)
    print(title)
    print("=" * 108)

    print("BASELINE (NO FEEDBACK)")
    print("-" * 108)
    print(
        f"Final forecast RMSE      : "
        f"{float(base_evaluation.forecast.rmse):.8f}"
    )
    print(
        f"Relation RMSE            : "
        f"{float(base_evaluation.relation.rmse):.8f}"
    )

    c = base_evaluation.consistency
    if c.consistency_testable:
        print(
            f"Consistency residual RMSE: "
            f"{float(c.cycle_consistency_weighted_rmse):.8f}"
        )
    else:
        print("Consistency residual RMSE: N/A (no independent cycle test)")

    print()
    print("Baseline forecast values:")
    for name, value in zip(system_names, base_forecast):
        print(f"  {name:<22}: {float(value): .8f}")

    if not experiments:
        print()
        print("No feedback experiments were supplied.")
        print("=" * 108)
        return

    print()
    print("FEEDBACK — DIRECT BEFORE / AFTER COMPARISON")
    print("-" * 108)
    print(
        f"{'alpha':>6} "
        f"{'iter':>4} | "
        f"{'Forecast RMSE before':>20} "
        f"{'after':>12} "
        f"{'gain':>10} | "
        f"{'Relation gain':>13} | "
        f"{'Consistency reduction':>21} | "
        f"{'Result':<18}"
    )
    print("-" * 108)

    rows = []

    for x in experiments:
        forecast_cmp = compare_lower_is_better(
            "Final forecast RMSE",
            x.forecast_rmse_before,
            x.forecast_rmse_after,
        )
        relation_cmp = compare_lower_is_better(
            "Relation RMSE",
            x.relation_rmse_before,
            x.relation_rmse_after,
        )

        raw_before = float(
            x.feedback.initial_equilibrium.weighted_rmse
        )
        raw_after = float(
            x.feedback.final_equilibrium.weighted_rmse
        )
        consistency_cmp = compare_lower_is_better(
            "Internal consistency residual",
            raw_before,
            raw_after,
        )

        verdict = _feedback_trial_verdict(
            forecast_cmp,
            relation_cmp,
        )

        rows.append(
            (
                x,
                forecast_cmp,
                relation_cmp,
                consistency_cmp,
                verdict,
            )
        )

        print(
            f"{float(x.alpha):6.2f} "
            f"{int(x.iterations):4d} | "
            f"{forecast_cmp.before:20.8f} "
            f"{forecast_cmp.after:12.8f} "
            f"{_fmt_percent(forecast_cmp.improvement_percent):>10} | "
            f"{_fmt_percent(relation_cmp.improvement_percent):>13} | "
            f"{_fmt_percent(consistency_cmp.improvement_percent):>21} | "
            f"{verdict:<18}"
        )

    print("-" * 108)
    print(
        "gain > 0 means lower error after feedback; "
        "consistency reduction is an internal diagnostic, not forecast accuracy."
    )

    # --------------------------------------------------------
    # Most important result: final forecast accuracy
    # --------------------------------------------------------
    best = min(
        rows,
        key=lambda item: float(
            item[1].after
        ),
    )
    best_x, best_forecast, best_relation, best_consistency, _ = best

    baseline_rmse = float(base_evaluation.forecast.rmse)
    best_rmse = float(best_forecast.after)

    print()
    print("=" * 108)
    print("MOST IMPORTANT RESULT — DID FEEDBACK IMPROVE THE FINAL FORECAST?")
    print("=" * 108)
    print(
        f"No feedback RMSE         : {baseline_rmse:.10f}"
    )
    print(
        f"Best feedback RMSE       : {best_rmse:.10f} "
        f"(alpha={float(best_x.alpha):.2f}, "
        f"iterations={int(best_x.iterations)})"
    )
    print(
        f"Absolute RMSE change     : "
        f"{best_rmse - baseline_rmse:+.10f}"
    )
    print(
        f"Forecast improvement     : "
        f"{_fmt_percent(best_forecast.improvement_percent)}"
    )

    if best_forecast.status == "IMPROVED":
        print("CONCLUSION               : YES — final forecast accuracy improved.")
    elif best_forecast.status == "WORSE":
        print("CONCLUSION               : NO — feedback made the final forecast worse.")
    else:
        print("CONCLUSION               : NO — final forecast accuracy did not improve.")

    print()
    print("What changed in the best feedback setting:")
    print(
        f"  Relation RMSE          : "
        f"{best_relation.before:.8f} -> {best_relation.after:.8f} "
        f"({_fmt_percent(best_relation.improvement_percent)} improvement)"
    )
    print(
        f"  Consistency residual   : "
        f"{best_consistency.before:.8f} -> {best_consistency.after:.8f} "
        f"({_fmt_percent(best_consistency.improvement_percent)} reduction)"
    )
    print(
        f"  max |delta gamma|      : "
        f"{float(best_x.feedback.gamma_change_max):.6e}"
    )
    print(
        f"  max |delta forecast|   : "
        f"{float(best_x.forecast_change_max):.6e}"
    )

    if (
        best_forecast.status == "UNCHANGED"
        and best_relation.status == "IMPROVED"
    ):
        print()
        print(
            "Interpretation: feedback improved the pairwise relation fit / "
            "internal consistency, but this improvement did not propagate to "
            "gamma or the final system forecast."
        )

    print("=" * 108)



# ============================================================
# Selection-locked multivariate benchmark reporting
# ============================================================


def _fmt_num(x: float) -> str:
    try:
        x = float(x)
    except Exception:
        return "N/A"
    if not np.isfinite(x):
        return "N/A"
    ax = abs(x)
    if ax >= 1e5 or (ax > 0 and ax < 1e-4):
        return f"{x:.4e}"
    return f"{x:.6f}"


def print_locked_selection_report(
    relation_specs: List[Any],
    magnitude_spec: Any,
    system_names: List[str],
    *,
    title: str = "LOCKED MODEL SELECTION REPORT",
) -> None:
    """Show the train/validation-only decisions frozen before test evaluation."""
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)
    print(
        "All family / Null / reliability decisions below are frozen before the "
        "test split. Test windows may refit only the locked specification."
    )
    print("-" * 118)
    print(
        f"{'pair':<22} {'status':<8} {'family':<18} {'model':<28} "
        f"{'weight':>10} {'val-MSE':>13} {'val-nRMSE':>13}"
    )
    print("-" * 118)

    n_valid = 0
    weights = []
    nrmse = []

    for spec in relation_specs:
        i = int(getattr(spec, "i"))
        j = int(getattr(spec, "j"))
        pair_name = f"{system_names[i]}--{system_names[j]}"
        valid = bool(getattr(spec, "is_valid", False))
        status = "KEEP" if valid else "NULL"
        family = str(getattr(spec, "selected_family", "unknown"))
        model = str(getattr(spec, "selected_model", "unknown"))
        weight = float(getattr(spec, "weight", np.nan))
        val_loss = float(getattr(spec, "validation_loss", np.nan))
        val_nrmse = float(getattr(spec, "normalized_rmse", np.nan))

        if valid:
            n_valid += 1
            if np.isfinite(weight):
                weights.append(weight)
            if np.isfinite(val_nrmse):
                nrmse.append(val_nrmse)

        print(
            f"{pair_name:<22} {status:<8} {family:<18} {model:<28} "
            f"{_fmt_num(weight):>10} {_fmt_num(val_loss):>13} {_fmt_num(val_nrmse):>13}"
        )

    print("-" * 118)
    print(f"retained relations        : {n_valid}/{len(relation_specs)}")
    if weights:
        w = np.asarray(weights, dtype=float)
        print(
            "retained weight range     : "
            f"{np.min(w):.6f} .. {np.max(w):.6f} "
            f"(median={np.median(w):.6f})"
        )
    if nrmse:
        x = np.asarray(nrmse, dtype=float)
        print(
            "retained val-nRMSE        : "
            f"mean={np.mean(x):.6f}, median={np.median(x):.6f}, max={np.max(x):.6f}"
        )

    print()
    print("Locked magnitude predictor")
    print("-" * 118)
    print(
        f"family / model            : "
        f"{getattr(magnitude_spec, 'selected_family', 'unknown')} / "
        f"{getattr(magnitude_spec, 'selected_model', 'unknown')}"
    )
    print(
        f"validation MSE            : "
        f"{_fmt_num(getattr(magnitude_spec, 'validation_loss', np.nan))}"
    )
    print(
        f"validation nRMSE          : "
        f"{_fmt_num(getattr(magnitude_spec, 'normalized_rmse', np.nan))}"
    )
    print("=" * 118)


def print_locked_window_process_report(
    window: Any,
    system_names: List[str],
    *,
    horizons: Optional[List[int]] = None,
    title: Optional[str] = None,
) -> None:
    """Detailed per-window stability/process diagnostics for locked RESE.

    This report is designed to detect long-horizon numerical/dynamic explosion.
    It does not modify, clip, or recompute the forecast.
    """
    pred = np.asarray(getattr(window, "prediction_model_space"), dtype=float)
    truth = np.asarray(getattr(window, "truth_model_space"), dtype=float)
    H, N = pred.shape

    if horizons is None:
        candidates = [1, 24, 48, 96, H]
        horizons = sorted({h for h in candidates if 1 <= h <= H})

    relation_max = getattr(window, "relation_path_max_abs_by_horizon", None)
    magnitude = getattr(window, "magnitude_path", None)
    latent_max = getattr(window, "latent_max_abs_by_horizon", None)

    relation_max = (
        np.asarray(relation_max, dtype=float)
        if relation_max is not None
        else np.full(H, np.nan)
    )
    magnitude = (
        np.asarray(magnitude, dtype=float)
        if magnitude is not None
        else np.full(H, np.nan)
    )
    latent_max = (
        np.asarray(latent_max, dtype=float)
        if latent_max is not None
        else np.full(H, np.nan)
    )

    if title is None:
        title = f"WINDOW PROCESS REPORT — origin={getattr(window, 'origin', 'N/A')}"

    print()
    print("=" * 118)
    print(title)
    print("=" * 118)
    print(
        f"{'h':>5} {'MSE':>14} {'MAE':>14} {'max|pred|':>14} "
        f"{'max|truth|':>14} {'max|relation|':>15} {'max|u*|':>12} {'M_hat':>14}"
    )
    print("-" * 118)

    for h in horizons:
        k = int(h) - 1
        e = pred[k] - truth[k]
        print(
            f"{h:5d} "
            f"{_fmt_num(np.mean(e ** 2)):>14} "
            f"{_fmt_num(np.mean(np.abs(e))):>14} "
            f"{_fmt_num(np.max(np.abs(pred[k]))):>14} "
            f"{_fmt_num(np.max(np.abs(truth[k]))):>14} "
            f"{_fmt_num(relation_max[k] if k < len(relation_max) else np.nan):>15} "
            f"{_fmt_num(latent_max[k] if k < len(latent_max) else np.nan):>12} "
            f"{_fmt_num(magnitude[k] if k < len(magnitude) else np.nan):>14}"
        )

    abs_pred = np.abs(pred)
    flat_idx = int(np.nanargmax(abs_pred))
    h_idx, c_idx = np.unravel_index(flat_idx, abs_pred.shape)
    worst_pred = float(pred[h_idx, c_idx])
    worst_truth = float(truth[h_idx, c_idx])

    print("-" * 118)
    print(
        "largest absolute prediction: "
        f"h={h_idx + 1}, channel={system_names[c_idx]}, "
        f"pred={_fmt_num(worst_pred)}, truth={_fmt_num(worst_truth)}"
    )

    # Forecast-explosion diagnostic. Standardized benchmark forecasts in the
    # hundreds/thousands are almost certainly dynamic instability.
    max_abs_pred = float(np.nanmax(abs_pred))
    growth = (
        float(np.nanmax(abs_pred[-1]) / max(np.nanmax(abs_pred[0]), 1e-12))
        if H > 1
        else 1.0
    )
    if max_abs_pred > 50 or growth > 100:
        print(
            "STABILITY ALERT           : forecast path shows explosive growth; "
            "treat the aggregate benchmark score as a failure-mode diagnostic "
            "until the responsible locked relation/magnitude model is identified."
        )
        print(
            f"max|prediction|={_fmt_num(max_abs_pred)}, "
            f"last/first max-abs ratio={_fmt_num(growth)}"
        )
    else:
        print("stability check            : no obvious explosive path detected.")

    print("=" * 118)