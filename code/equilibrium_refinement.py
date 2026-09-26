"""
equilibrium_refinement_v3.py
============================

Iterative robust equilibrium refinement for Rel-ESE.

Why this module exists
----------------------
The ordinary global ES solver already computes the exact weighted least-squares
solution for a fixed pairwise relation matrix R and fixed reliability matrix W.
Repeatedly solving the same problem therefore cannot improve the solution.

This module adds a *genuine* refinement loop by keeping the forecast pairwise
relations R fixed while updating only edge reliabilities according to how strongly
each edge disagrees with the globally coherent equilibrium state.

At iteration k:

    u^(k) = argmin_u sum_e w_e^(k) (B_e u - r_e)^2

    e_e^(k) = r_e - B_e u^(k)

    w_e^(k+1) = w_e^pred * g(e_e^(k))

where ``w^pred`` is the original validation reliability from the pairwise layer
and ``g`` is a robust consistency factor (Huber by default).

Important design choice
-----------------------
Pairwise relation forecasts ``r_ij`` are NEVER moved toward the ES projection.
Only their weights are refined.  This preserves the original relational evidence
and avoids the tautological feedback behavior in which repeatedly replacing
``r`` by ``u_i-u_j`` makes the residual vanish without adding forecast evidence.

The implementation operates on a copy of ``RelationMatrixState`` by default, so
calling code can retain the original pairwise reliability matrix for ablations
and diagnostics.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from relation_matrix import RelationMatrixState
from equilibrium_solver import EquilibriumResult, solve_equilibrium_state
from relation_representation import LOG_RATIO


HUBER = "huber"
CAUCHY = "cauchy"
EXPONENTIAL = "exponential"
SUPPORTED_METHODS = (HUBER, CAUCHY, EXPONENTIAL)


@dataclass(frozen=True)
class RefinementIteration:
    """Diagnostics for one completed reweighting iteration."""

    iteration: int
    latent_relative_change: float
    max_weight_relative_change: float

    weighted_rmse: float
    normalized_consistency_error: float
    mean_absolute_residual: float
    max_absolute_residual: float

    robust_scale: float
    cutoff: float
    n_used_edges: int
    n_downweighted_edges: int

    min_weight_ratio: float
    median_weight_ratio: float
    mean_weight_ratio: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "latent_relative_change": self.latent_relative_change,
            "max_weight_relative_change": self.max_weight_relative_change,
            "weighted_rmse": self.weighted_rmse,
            "normalized_consistency_error": self.normalized_consistency_error,
            "mean_absolute_residual": self.mean_absolute_residual,
            "max_absolute_residual": self.max_absolute_residual,
            "robust_scale": self.robust_scale,
            "cutoff": self.cutoff,
            "n_used_edges": self.n_used_edges,
            "n_downweighted_edges": self.n_downweighted_edges,
            "min_weight_ratio": self.min_weight_ratio,
            "median_weight_ratio": self.median_weight_ratio,
            "mean_weight_ratio": self.mean_weight_ratio,
        }


@dataclass
class EquilibriumRefinementResult:
    """Output of iterative robust equilibrium refinement."""

    method: str
    converged: bool
    n_iterations: int

    initial_equilibrium: EquilibriumResult
    equilibrium: EquilibriumResult
    relation_state: RelationMatrixState

    original_weights: np.ndarray
    final_weights: np.ndarray

    robust_scale: float
    tuning_constant: float
    cutoff: float
    blend: float
    minimum_weight_ratio: float

    latent_tolerance: float
    weight_tolerance: float
    history: List[RefinementIteration] = field(default_factory=list)

    @property
    def initial_weighted_rmse(self) -> float:
        return float(self.initial_equilibrium.weighted_rmse)

    @property
    def final_weighted_rmse(self) -> float:
        return float(self.equilibrium.weighted_rmse)

    @property
    def initial_normalized_consistency_error(self) -> float:
        return float(self.initial_equilibrium.normalized_consistency_error)

    @property
    def final_normalized_consistency_error(self) -> float:
        return float(self.equilibrium.normalized_consistency_error)

    def summary(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "initial_weighted_rmse": self.initial_weighted_rmse,
            "final_weighted_rmse": self.final_weighted_rmse,
            "initial_normalized_consistency_error": self.initial_normalized_consistency_error,
            "final_normalized_consistency_error": self.final_normalized_consistency_error,
            "robust_scale": self.robust_scale,
            "tuning_constant": self.tuning_constant,
            "cutoff": self.cutoff,
            "blend": self.blend,
            "minimum_weight_ratio": self.minimum_weight_ratio,
        }


def clone_relation_state(state: RelationMatrixState) -> RelationMatrixState:
    """Deep-copy a relation state without changing its numerical conventions."""
    if not isinstance(state, RelationMatrixState):
        raise TypeError("state must be a RelationMatrixState.")

    return RelationMatrixState(
        n_systems=int(state.n_systems),
        system_names=list(state.system_names),
        base_values=np.asarray(state.base_values, dtype=float).copy(),
        values=np.asarray(state.values, dtype=float).copy(),
        weights=np.asarray(state.weights, dtype=float).copy(),
        valid_mask=np.asarray(state.valid_mask, dtype=bool).copy(),
        edge_metadata=deepcopy(state.edge_metadata),
        iteration=int(state.iteration),
        feedback_history=deepcopy(state.feedback_history),
    )


def _edge_residuals(result: EquilibriumResult) -> np.ndarray:
    """Residual vector in exactly the used-edge order returned by the ES solver."""
    residuals = []
    for i, j in result.used_edge_index.T:
        value = result.residual_matrix[int(i), int(j)]
        residuals.append(float(value))
    return np.asarray(residuals, dtype=float)


def _initial_robust_scale(
    residuals: np.ndarray,
    relations: np.ndarray,
    weighted_rmse: float,
    *,
    floor_ratio: float = 1e-6,
) -> float:
    """Robust residual scale, fixed across refinement iterations.

    MAD is preferred.  Degenerate cases fall back to weighted RMSE, median
    absolute residual, and finally a tiny relation-scale floor.
    """
    residuals = np.asarray(residuals, dtype=float)
    relations = np.asarray(relations, dtype=float)

    finite_r = residuals[np.isfinite(residuals)]
    if finite_r.size == 0:
        raise ValueError("Cannot estimate refinement scale from empty residuals.")

    center = float(np.median(finite_r))
    mad = float(1.4826 * np.median(np.abs(finite_r - center)))
    median_abs = float(np.median(np.abs(finite_r)))

    finite_rel = relations[np.isfinite(relations)]
    relation_scale = (
        float(np.median(np.abs(finite_rel))) if finite_rel.size else 1.0
    )
    floor = max(1e-12, abs(relation_scale) * float(floor_ratio))

    for candidate in (mad, float(weighted_rmse), median_abs, floor):
        if np.isfinite(candidate) and candidate > floor:
            return float(candidate)

    return float(floor)


def _consistency_factor(
    residuals: np.ndarray,
    *,
    method: str,
    cutoff: float,
    minimum_weight_ratio: float,
) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    abs_e = np.abs(residuals)
    c = max(float(cutoff), 1e-15)

    if method == HUBER:
        factor = np.ones_like(abs_e)
        outside = abs_e > c
        factor[outside] = c / np.maximum(abs_e[outside], 1e-15)
    elif method == CAUCHY:
        factor = 1.0 / (1.0 + (abs_e / c) ** 2)
    elif method == EXPONENTIAL:
        factor = np.exp(-abs_e / c)
    else:
        raise ValueError(
            f"Unknown refinement method {method!r}; expected one of {SUPPORTED_METHODS}."
        )

    factor = np.clip(factor, float(minimum_weight_ratio), 1.0)
    return np.asarray(factor, dtype=float)


def _make_target_weight_matrix(
    state: RelationMatrixState,
    result: EquilibriumResult,
    base_weights: np.ndarray,
    factors: np.ndarray,
) -> np.ndarray:
    """Build an upper-triangular target W while leaving unused edges unchanged."""
    target = np.asarray(state.weights, dtype=float).copy()

    for idx, (i, j) in enumerate(result.used_edge_index.T):
        i = int(i)
        j = int(j)
        target[i, j] = float(base_weights[i, j] * factors[idx])

    return target


def refine_equilibrium_state(
    relation_state: RelationMatrixState,
    *,
    representation: str = LOG_RATIO,
    method: str = HUBER,
    max_iterations: int = 20,
    latent_tolerance: float = 1e-6,
    weight_tolerance: float = 1e-4,
    tuning_constant: float = 1.345,
    blend: float = 0.5,
    minimum_weight_ratio: float = 0.05,
    weight_threshold: float = 0.0,
    regularization: float = 0.0,
    disconnected_policy: str = "error",
    anchor_values: Optional[np.ndarray] = None,
    anchor_latent: Optional[np.ndarray] = None,
    anchor_epsilon: float = 1e-8,
    copy_state: bool = True,
    verbose: bool = False,
) -> EquilibriumRefinementResult:
    """Robustly refine the global ES by iteratively reweighting inconsistent edges.

    Parameters
    ----------
    relation_state:
        Pairwise relation matrix and validation-derived reliabilities.

    representation:
        Relation geometry passed unchanged to ``solve_equilibrium_state``.

    method:
        ``huber`` (default), ``cauchy``, or ``exponential`` consistency gate.

    max_iterations:
        Maximum number of weight-update / ES-solve iterations.  The initial
        one-shot ES is iteration 0 and is not counted here.

    latent_tolerance, weight_tolerance:
        Stop only when both the relative change in latent state and the maximum
        relative edge-weight change are below these thresholds.

    tuning_constant:
        Robust cutoff multiplier.  With Huber, 1.345 is the classical default.

    blend:
        Damping in [0,1].  At each iteration:

            W <- (1-blend) W + blend W_target

        ``W_target`` is always based on the ORIGINAL validation reliability,
        not recursively multiplied by previous consistency factors.

    minimum_weight_ratio:
        Lower bound for the consistency factor relative to each edge's original
        validation weight.  Keeping this positive helps preserve graph
        connectivity in the first experimental version.

    anchor_values, anchor_latent:
        Optional current-state anchor forwarded to ``equilibrium_solver_v4``.
        These are used only when ``disconnected_policy='st_component_anchor'``.

    Notes
    -----
    The relation values R are held fixed throughout.  This routine therefore
    implements robust global reconciliation rather than relation projection.
    """
    if not isinstance(relation_state, RelationMatrixState):
        raise TypeError("relation_state must be a RelationMatrixState.")

    method = str(method).strip().lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"method must be one of {SUPPORTED_METHODS}; got {method!r}."
        )
    if int(max_iterations) < 0:
        raise ValueError("max_iterations must be non-negative.")
    if latent_tolerance < 0 or weight_tolerance < 0:
        raise ValueError("convergence tolerances must be non-negative.")
    if tuning_constant <= 0:
        raise ValueError("tuning_constant must be positive.")
    if not 0.0 < blend <= 1.0:
        raise ValueError("blend must lie in (0, 1].")
    if not 0.0 <= minimum_weight_ratio <= 1.0:
        raise ValueError("minimum_weight_ratio must lie in [0, 1].")

    state = clone_relation_state(relation_state) if copy_state else relation_state
    original_weights = np.asarray(state.weights, dtype=float).copy()

    initial = solve_equilibrium_state(
        state,
        weight_threshold=weight_threshold,
        regularization=regularization,
        disconnected_policy=disconnected_policy,
        representation=representation,
        anchor_values=anchor_values,
        anchor_latent=anchor_latent,
        anchor_epsilon=anchor_epsilon,
    )

    residuals0 = _edge_residuals(initial)
    robust_scale = _initial_robust_scale(
        residuals0,
        initial.used_edge_relations,
        initial.weighted_rmse,
    )
    cutoff = float(tuning_constant) * float(robust_scale)

    if verbose:
        print()
        print("=" * 76)
        print("Iterative equilibrium refinement")
        print("=" * 76)
        print(f"method                    : {method}")
        print(f"max iterations            : {int(max_iterations)}")
        print(f"robust residual scale     : {robust_scale:.8g}")
        print(f"cutoff                    : {cutoff:.8g}")
        print(f"weight blend              : {blend:.4f}")
        print(f"minimum weight ratio      : {minimum_weight_ratio:.4f}")
        print(
            "initial consistency RMSE  : "
            f"{initial.weighted_rmse:.8g}"
        )
        print("=" * 76)

    if int(max_iterations) == 0:
        return EquilibriumRefinementResult(
            method=method,
            converged=True,
            n_iterations=0,
            initial_equilibrium=initial,
            equilibrium=initial,
            relation_state=state,
            original_weights=original_weights,
            final_weights=np.asarray(state.weights, dtype=float).copy(),
            robust_scale=robust_scale,
            tuning_constant=float(tuning_constant),
            cutoff=cutoff,
            blend=float(blend),
            minimum_weight_ratio=float(minimum_weight_ratio),
            latent_tolerance=float(latent_tolerance),
            weight_tolerance=float(weight_tolerance),
            history=[],
        )

    current = initial
    history: List[RefinementIteration] = []
    converged = False

    for iteration in range(1, int(max_iterations) + 1):
        residuals = _edge_residuals(current)
        factors = _consistency_factor(
            residuals,
            method=method,
            cutoff=cutoff,
            minimum_weight_ratio=minimum_weight_ratio,
        )

        target_weights = _make_target_weight_matrix(
            state,
            current,
            original_weights,
            factors,
        )

        before_weights = np.asarray(state.weights, dtype=float).copy()
        state.update_weights(
            target_weights,
            blend=float(blend),
            valid_only=True,
        )

        new = solve_equilibrium_state(
            state,
            weight_threshold=weight_threshold,
            regularization=regularization,
            disconnected_policy=disconnected_policy,
            representation=representation,
            anchor_values=anchor_values,
            anchor_latent=anchor_latent,
            anchor_epsilon=anchor_epsilon,
        )

        denom_u = max(float(np.linalg.norm(current.u)), 1e-12)
        latent_change = float(np.linalg.norm(new.u - current.u) / denom_u)

        valid_upper = np.triu(state.valid_mask, k=1)
        denom_w = np.maximum(np.abs(before_weights), 1e-12)
        rel_w_change = np.zeros_like(before_weights, dtype=float)
        rel_w_change[valid_upper] = (
            np.abs(state.weights[valid_upper] - before_weights[valid_upper])
            / denom_w[valid_upper]
        )
        max_weight_change = (
            float(np.max(rel_w_change[valid_upper]))
            if np.any(valid_upper)
            else 0.0
        )

        ratios = []
        for i, j in new.used_edge_index.T:
            i = int(i)
            j = int(j)
            base_w = float(original_weights[i, j])
            if base_w > 0:
                ratios.append(float(state.weights[i, j] / base_w))
        ratios_arr = np.asarray(ratios, dtype=float)

        if ratios_arr.size:
            min_ratio = float(np.min(ratios_arr))
            median_ratio = float(np.median(ratios_arr))
            mean_ratio = float(np.mean(ratios_arr))
            n_down = int(np.sum(ratios_arr < 1.0 - 1e-10))
        else:
            min_ratio = median_ratio = mean_ratio = 1.0
            n_down = 0

        record = RefinementIteration(
            iteration=iteration,
            latent_relative_change=latent_change,
            max_weight_relative_change=max_weight_change,
            weighted_rmse=float(new.weighted_rmse),
            normalized_consistency_error=float(new.normalized_consistency_error),
            mean_absolute_residual=float(new.mean_absolute_residual),
            max_absolute_residual=float(new.max_absolute_residual),
            robust_scale=robust_scale,
            cutoff=cutoff,
            n_used_edges=int(new.n_used_edges),
            n_downweighted_edges=n_down,
            min_weight_ratio=min_ratio,
            median_weight_ratio=median_ratio,
            mean_weight_ratio=mean_ratio,
        )
        history.append(record)

        if verbose:
            print(
                f"[refine {iteration:>2}/{int(max_iterations)}] "
                f"du={latent_change:.3e}, "
                f"dw={max_weight_change:.3e}, "
                f"consistency_rmse={new.weighted_rmse:.6g}, "
                f"downweighted={n_down}/{new.n_used_edges}, "
                f"min_w/base={min_ratio:.3f}",
                flush=True,
            )

        current = new

        if (
            latent_change <= float(latent_tolerance)
            and max_weight_change <= float(weight_tolerance)
        ):
            converged = True
            break

    if verbose:
        print("-" * 76)
        print(f"converged                 : {converged}")
        print(f"iterations                : {len(history)}")
        print(f"final consistency RMSE    : {current.weighted_rmse:.8g}")
        print(
            "final normalized error    : "
            f"{current.normalized_consistency_error:.8g}"
        )
        print("=" * 76)
        print()

    return EquilibriumRefinementResult(
        method=method,
        converged=converged,
        n_iterations=len(history),
        initial_equilibrium=initial,
        equilibrium=current,
        relation_state=state,
        original_weights=original_weights,
        final_weights=np.asarray(state.weights, dtype=float).copy(),
        robust_scale=robust_scale,
        tuning_constant=float(tuning_constant),
        cutoff=cutoff,
        blend=float(blend),
        minimum_weight_ratio=float(minimum_weight_ratio),
        latent_tolerance=float(latent_tolerance),
        weight_tolerance=float(weight_tolerance),
        history=history,
    )


# Backward-friendly short alias.
def refine_es(
    relation_state: RelationMatrixState,
    **kwargs: Any,
) -> EquilibriumRefinementResult:
    return refine_equilibrium_state(relation_state, **kwargs)


if __name__ == "__main__":
    # Small diagnostic: five consistent edges and one deliberately corrupted edge.
    from relation_matrix import PairwiseRelationEdge, build_relation_matrix

    u_true = np.array([0.75, 0.25, -0.25, -0.75], dtype=float)
    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            r = float(u_true[i] - u_true[j])
            if (i, j) == (0, 3):
                r += 2.0  # deliberately inconsistent relation
            edges.append(
                PairwiseRelationEdge(i=i, j=j, r=r, weight=1.0, is_valid=True)
            )

    state = build_relation_matrix(
        n_systems=4,
        edges=edges,
        system_names=["A", "B", "C", "D"],
        require_complete_input=True,
    )

    one_shot = solve_equilibrium_state(state, representation="additive")
    refined = refine_equilibrium_state(
        state,
        representation="additive",
        method="huber",
        max_iterations=30,
        blend=0.7,
        minimum_weight_ratio=0.02,
        verbose=True,
    )

    print("true u     :", np.round(u_true - np.mean(u_true), 6))
    print("one-shot u :", np.round(one_shot.u, 6))
    print("refined u  :", np.round(refined.equilibrium.u, 6))