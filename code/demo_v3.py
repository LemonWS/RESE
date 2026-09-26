"""demo_v10.py
================

Clean statistical Rel-ESE demo.

This runner intentionally contains no neural-relation branch.  Neural teacher /
student experiments remain in the dedicated ``demo_neural_*`` runner so the
ordinary statistical demo is easy to inspect and debug.

Pipeline
--------
file -> data_loader.py
     -> forecast_window_v1.py (input/output configuration)
     -> automatic/manual relation geometry selection
     -> pairwise_relation_v6.py
     -> relation_matrix.py
     -> equilibrium_solver_v3.py
     -> predictor_v5.py
     -> evaluation_v5.py
     -> experiment_report_v4.py

Automatic geometry
------------------
``representation='auto'`` scans the numeric dataset before fitting:
- all observations >= 0 -> ``log_ratio``
- any observation < 0   -> ``asinh``

For formal paper experiments, explicitly setting the representation is still
recommended so the experimental condition is fixed rather than data-driven.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from relation_representation import (
    AUTO,
    available_representations,
    get_representation_info,
    resolve_representation,
)
from forecast_window import (
    AUTO as AUTO_HISTORY,
    ForecastWindowConfig,
    effective_input_length,
)
from data_loader import load_system_data
from pairwise_systems import create_system_pairs
from pairwise_relation import estimate_pair_relation
from relation_matrix import RelationMatrixState, build_relation_matrix_from_results
from equilibrium_solver import EquilibriumResult, solve_equilibrium_state
from predictor import (
    PredictorResult,
    predict_magnitude_from_systems,
    reconstruct_from_equilibrium,
)
from evaluation import PipelineEvaluation, evaluate_pipeline
from experiment_report import (
    TimingReport,
    WallClock,
    print_experiment_report,
    print_end_to_end_runtime,
)


def _default_data_path() -> Path:
    root = Path(__file__).resolve().parent
    in_data_dir = root / "data" / "synthetic_ese_demo_v1.csv"
    beside_script = root / "synthetic_ese_demo_v1.csv"
    return in_data_dir if in_data_dir.exists() else beside_script


DEFAULT_DATA_PATH = _default_data_path()


@dataclass
class FileDemoResult:
    relation_mode: str
    representation: str
    requested_representation: str

    dates: np.ndarray
    system_names: List[str]
    origin: int
    origin_date: Any
    target_index: int
    target_date: Any
    horizon: int
    input_length: Optional[int]
    output_length: int
    history_mode: str

    Y_history: np.ndarray
    Y_input: np.ndarray
    y_true: np.ndarray

    relation_state: RelationMatrixState
    equilibrium: EquilibriumResult
    predictor: PredictorResult
    system_forecast: np.ndarray
    evaluation: PipelineEvaluation
    timing: TimingReport
    relation_results: Optional[List[Any]] = None


class _EvaluationView:
    pass


def _prepare_holdout(
    dates: np.ndarray,
    Y: np.ndarray,
    *,
    output_length: int,
    origin: Optional[int],
) -> tuple[int, int, np.ndarray, np.ndarray]:
    if output_length < 1:
        raise ValueError("output_length must be at least 1.")
    T = len(Y)
    origin = T - output_length if origin is None else int(origin)
    if origin < 20:
        raise ValueError(
            "The forecast origin must leave at least 20 historical observations."
        )
    target_index = origin + output_length - 1
    if target_index >= T:
        raise ValueError(
            f"origin={origin} with output_length={output_length} requires target index "
            f"{target_index}, but the dataset has only T={T} observations."
        )
    return (
        origin,
        target_index,
        np.asarray(Y[:origin], dtype=float),
        np.asarray(Y[target_index], dtype=float),
    )


def _build_statistical_relation_state(
    Y_history: np.ndarray,
    system_names: Sequence[str],
    timing: TimingReport,
    *,
    horizon: int,
    validation_ratio: float,
    allow_null: bool,
    min_improvement_over_null: float,
    weight_eta: float,
    validation_stride: int,
    max_validation_origins: Optional[int],
    min_validation_success_rate: float,
    representation: str,
    input_length: Optional[int],
    history_mode: str,
    verbose: bool,
) -> tuple[RelationMatrixState, List[Any]]:
    pairs = create_system_pairs(Y_history, list(system_names))
    relation_results: List[Any] = []

    for idx, pair in enumerate(pairs, start=1):
        if verbose:
            print(
                f"[relation {idx:>3}/{len(pairs)}] "
                f"{pair.system_i} -- {pair.system_j}: starting...",
                flush=True,
            )

        with timing.measure_pair():
            result = estimate_pair_relation(
                pair.y_i,
                pair.y_j,
                horizon=horizon,
                validation_ratio=validation_ratio,
                allow_null=allow_null,
                min_improvement_over_null=min_improvement_over_null,
                weight_eta=weight_eta,
                validation_method="rolling_origin",
                validation_stride=validation_stride,
                max_validation_origins=max_validation_origins,
                min_validation_success_rate=min_validation_success_rate,
                representation=representation,
                input_length=input_length,
                history_mode=history_mode,
            )
        relation_results.append(result)

        if verbose:
            r_text = f"{float(result.r): .6f}" if np.isfinite(result.r) else " NaN"
            print(
                f"[relation {idx:>3}/{len(pairs)}] "
                f"{pair.system_i} -- {pair.system_j}: done | "
                f"{result.selected_family}/{result.selected_model}, "
                f"r={r_text}, w={float(result.weight):.4f}, "
                f"valid={bool(result.is_valid)}, "
                f"time={timing.pair_relation_times[-1]:.4f}s",
                flush=True,
            )

    timing.relation_estimation = float(np.sum(timing.pair_relation_times))

    with timing.measure("relation_matrix"):
        relation_state = build_relation_matrix_from_results(
            pairs=pairs,
            relation_results=relation_results,
            n_systems=Y_history.shape[1],
            system_names=list(system_names),
            require_complete_input=True,
        )

    return relation_state, relation_results


def run_file_demo(
    file_path: str | Path,
    *,
    date_col: Optional[str] = None,
    input_length: Optional[int] = None,
    output_length: int = 1,
    history_mode: str = AUTO_HISTORY,
    origin: Optional[int] = None,
    representation: str = AUTO,
    relation_validation_ratio: float = 0.20,
    allow_null: bool = True,
    min_improvement_over_null: float = 0.0,
    relation_weight_eta: float = 1.0,
    relation_validation_stride: int = 1,
    relation_max_validation_origins: Optional[int] = 10,
    relation_min_validation_success_rate: float = 0.8,
    es_weight_threshold: float = 0.0,
    disconnected_policy: str = "error",
    predictor_validation_ratio: float = 0.20,
    predictor_validation_stride: int = 1,
    predictor_max_validation_origins: Optional[int] = 10,
    predictor_min_validation_success_rate: float = 0.8,
    verbose: bool = True,
    report: bool = True,
) -> FileDemoResult:
    timing = TimingReport()
    requested_representation = str(representation)
    window = ForecastWindowConfig(
        input_length=input_length,
        output_length=output_length,
        history_mode=history_mode,
    )
    output_length = window.output_length
    history_mode = window.resolved_history_mode

    with timing.measure("workflow_total"):
        with timing.measure("data_loading"):
            dates, Y, system_names = load_system_data(
                file_path=str(file_path),
                date_col=date_col,
            )
            Y = np.asarray(Y, dtype=float)

        with timing.measure("preparation"):
            origin, target_index, Y_history, y_true = _prepare_holdout(
                dates, Y, output_length=output_length, origin=origin
            )
            # The final fit uses either the full available history or exactly
            # the most recent L observations.  AUTO geometry scans this actual
            # model-input window rather than unrelated earlier observations.
            Y_input = window.slice(Y_history)
            representation = resolve_representation(
                Y_input,
                requested_representation,
            )

        info = get_representation_info(representation)
        minimum = float(np.min(Y_input))
        maximum = float(np.max(Y_input))
        auto_reason = (
            "negative observations detected -> signed-compressed asinh"
            if representation == "asinh" and requested_representation.lower() in {"auto", "automatic", "infer"}
            else "all observations are non-negative -> multiplicative log-ratio"
            if representation == "log_ratio" and requested_representation.lower() in {"auto", "automatic", "infer"}
            else "explicit user selection"
        )

        if verbose:
            print()
            print("=" * 76)
            print("Statistical Rel-ESE demo v10")
            print("=" * 76)
            print(f"Data file                 : {file_path}")
            print("Relation estimator        : statistical")
            print(f"Requested representation  : {requested_representation}")
            print(f"Selected representation   : {representation}")
            print(f"Geometry                  : {info.geometry}")
            print(f"Selection reason          : {auto_reason}")
            print(f"Input-window minimum      : {minimum:.6g}")
            print(f"Input-window maximum      : {maximum:.6g}")
            print(f"Available history shape   : {Y_history.shape}")
            print(f"Model input shape         : {Y_input.shape}")
            print(f"Input length              : {len(Y_input)}"
                  + (f" (requested {input_length})" if input_length is not None else " (full available)"))
            print(f"History mode              : {history_mode}")
            print(f"Output length / horizon   : {output_length}")
            print(f"Forecast origin           : index {origin - 1} / {dates[origin - 1]}")
            print(f"Target                    : index {target_index} / {dates[target_index]}")
            print(
                "Validation                : horizon-matched rolling origin "
                f"(relation max={relation_max_validation_origins}, "
                f"predictor max={predictor_max_validation_origins})"
            )
            print("=" * 76)
            print()

        relation_state, relation_results = _build_statistical_relation_state(
            Y_history,
            system_names,
            timing,
            horizon=output_length,
            validation_ratio=relation_validation_ratio,
            allow_null=allow_null,
            min_improvement_over_null=min_improvement_over_null,
            weight_eta=relation_weight_eta,
            validation_stride=relation_validation_stride,
            max_validation_origins=relation_max_validation_origins,
            min_validation_success_rate=relation_min_validation_success_rate,
            representation=representation,
            input_length=input_length,
            history_mode=history_mode,
            verbose=verbose,
        )

        with timing.measure("equilibrium_solver"):
            equilibrium = solve_equilibrium_state(
                relation_state,
                weight_threshold=es_weight_threshold,
                disconnected_policy=disconnected_policy,
                representation=representation,
            )

        with timing.measure("magnitude_predictor"):
            predictor = predict_magnitude_from_systems(
                Y_history,
                horizon=output_length,
                validation_ratio=predictor_validation_ratio,
                validation_method="rolling_origin",
                validation_stride=predictor_validation_stride,
                max_validation_origins=predictor_max_validation_origins,
                min_validation_success_rate=predictor_min_validation_success_rate,
                representation=representation,
                input_length=input_length,
                history_mode=history_mode,
            )

        with timing.measure("reconstruction"):
            system_forecast = reconstruct_from_equilibrium(
                equilibrium, predictor.magnitude
            )

        evaluation_view = _EvaluationView()
        evaluation_view.mode = "statistical"
        evaluation_view.relation_state = relation_state
        evaluation_view.equilibrium = equilibrium
        evaluation_view.predictor = predictor
        evaluation_view.system_forecast = system_forecast

        with timing.measure("evaluation"):
            evaluation = evaluate_pipeline(
                evaluation_view,
                y_true,
                relation_weight_threshold=es_weight_threshold,
                metadata={
                    "relation_mode": "statistical",
                    "requested_representation": requested_representation,
                    "representation": representation,
                    "data_file": str(file_path),
                    "input_length": input_length,
                    "effective_input_length": int(len(Y_input)),
                    "output_length": int(output_length),
                    "history_mode": history_mode,
                    "origin": int(origin),
                    "origin_date": str(dates[origin - 1]),
                    "target_index": int(target_index),
                    "target_date": str(dates[target_index]),
                },
            )

    result = FileDemoResult(
        relation_mode="statistical",
        representation=representation,
        requested_representation=requested_representation,
        dates=dates,
        system_names=list(system_names),
        origin=int(origin),
        origin_date=dates[origin - 1],
        target_index=int(target_index),
        target_date=dates[target_index],
        horizon=int(output_length),
        input_length=input_length,
        output_length=int(output_length),
        history_mode=history_mode,
        Y_history=np.asarray(Y_history, dtype=float),
        Y_input=np.asarray(Y_input, dtype=float),
        y_true=np.asarray(y_true, dtype=float),
        relation_state=relation_state,
        equilibrium=equilibrium,
        predictor=predictor,
        system_forecast=np.asarray(system_forecast, dtype=float),
        evaluation=evaluation,
        timing=timing,
        relation_results=relation_results,
    )

    if report:
        print_experiment_report(
            result,
            evaluation=result.evaluation,
            timing=result.timing,
            include_forecast=True,
            include_evaluation=True,
            include_timing=True,
            include_final_values=True,
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statistical history-driven Rel-ESE with pluggable relation geometry."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--date-col", default=None)
    parser.add_argument(
        "--input-length",
        type=int,
        default=None,
        help=(
            "Lookback length L. With the default history-mode=auto, setting L "
            "activates fixed sliding windows; omitting it preserves expanding history."
        ),
    )
    parser.add_argument(
        "--output-length",
        "--horizon",
        dest="output_length",
        type=int,
        default=3,
        help=(
            "Forecast endpoint H. Current Rel-ESE returns the system forecast "
            "at t+H; --horizon is kept as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--history-mode",
        choices=["auto", "sliding", "expanding"],
        default="auto",
        help=(
            "auto: input-length set -> sliding, otherwise expanding. "
            "sliding uses exactly the last L observations at every rolling origin."
        ),
    )
    parser.add_argument("--origin", type=int, default=None)
    parser.add_argument(
        "--representation",
        choices=list(available_representations(include_auto=True)),
        default=AUTO,
        help=(
            "auto: non-negative -> log_ratio; signed -> asinh. "
            "Manual options: log_ratio, asinh, signed_log, additive."
        ),
    )
    parser.add_argument("--disable-null", action="store_true")
    parser.add_argument("--weight-threshold", type=float, default=0.0)
    parser.add_argument("--validation-stride", type=int, default=1)
    parser.add_argument("--max-validation-origins", type=int, default=10)
    parser.add_argument("--min-validation-success-rate", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_path = Path(args.data).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Place the demo CSV under ./data/ or pass --data <path>."
        )

    with WallClock() as end_to_end_timer:
        result = run_file_demo(
            data_path,
            date_col=args.date_col,
            input_length=args.input_length,
            output_length=args.output_length,
            history_mode=args.history_mode,
            origin=args.origin,
            representation=args.representation,
            allow_null=not args.disable_null,
            es_weight_threshold=args.weight_threshold,
            relation_validation_stride=args.validation_stride,
            relation_max_validation_origins=args.max_validation_origins,
            relation_min_validation_success_rate=args.min_validation_success_rate,
            predictor_validation_stride=args.validation_stride,
            predictor_max_validation_origins=args.max_validation_origins,
            predictor_min_validation_success_rate=args.min_validation_success_rate,
            verbose=True,
            report=True,
        )
    result.timing.end_to_end_total = end_to_end_timer.elapsed
    print_end_to_end_runtime(result.timing)


if __name__ == "__main__":
    main()

