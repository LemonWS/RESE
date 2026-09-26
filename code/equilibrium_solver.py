
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from relation_matrix import RelationMatrixState
from relation_representation import (
    LOG_RATIO,
    centered_latent_state,
    normalize_representation_name,
    reconstruct_from_latent,
    simplex_state_from_latent,
)


# ============================================================
# Result container
# ============================================================


@dataclass
class EquilibriumResult:
    """Result of global equilibrium-state recovery.

    Parameters
    ----------
    gamma : ndarray, shape (N,)
        Estimated equilibrium proportions for ``log_ratio`` mode.  For other
        geometries this compatibility field is NaN; use ``u`` plus
        representation-aware reconstruction.

    u : ndarray, shape (N,)
        Zero-sum latent relation coordinates satisfying r_ij ~= u_i-u_j.

    observed_relation_matrix : ndarray, shape (N,N)
        Full antisymmetric relation matrix supplied by RelationMatrixState.
        Missing/invalid edges are NaN.

    global_relation_matrix : ndarray, shape (N,N)
        Fully globally consistent matrix implied by u:

            R_ES[i,j] = u_i-u_j.

        This is defined for every pair, including pairs that were absent from
        the original relation graph.

    residual_matrix : ndarray, shape (N,N)
        Local-minus-global residual on edges actually used by the solver:

            E[i,j] = R_observed[i,j] - R_ES[i,j].

        Entries for unused edges are NaN.

    weighted_sse, weighted_mse, weighted_rmse : float
        Weighted local-global inconsistency diagnostics.

    normalized_consistency_error : float
        weighted_rmse divided by the weighted RMS magnitude of observed r.
        Zero means exact global consistency.

    connected : bool
        Whether the graph formed by used positive-weight edges is connected.

    globally_identified : bool
        True only when one global equilibrium is identified from the graph.

    components : list[list[int]]
        Connected components of the used relation graph.

    used_edge_index : ndarray, shape (2,E)
        Edges used by the solver.

    used_edge_relations, used_edge_weights : ndarray, shape (E,)
        Numerical edge observations supplied to weighted least squares.

    laplacian : ndarray, shape (N,N)
        Weighted graph Laplacian B^T W B.

    rhs : ndarray, shape (N,)
        Right-hand side B^T W r.

    diagnostics : dict
        Additional numerical diagnostics.
    """

    representation: str
    gamma: np.ndarray
    u: np.ndarray

    observed_relation_matrix: np.ndarray
    global_relation_matrix: np.ndarray
    residual_matrix: np.ndarray

    weighted_sse: float
    weighted_mse: float
    weighted_rmse: float
    normalized_consistency_error: float

    mean_absolute_residual: float
    max_absolute_residual: float

    connected: bool
    globally_identified: bool
    components: List[List[int]]

    used_edge_index: np.ndarray
    used_edge_relations: np.ndarray
    used_edge_weights: np.ndarray

    laplacian: np.ndarray
    rhs: np.ndarray

    system_names: List[str]
    use_base: bool
    weight_threshold: float
    solver_method: str

    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_systems(self) -> int:
        return int(len(self.u))

    @property
    def has_simplex_state(self) -> bool:
        return bool(self.representation == LOG_RATIO)

    def reconstruct(self, aggregate: float) -> np.ndarray:
        """Reconstruct system values using this latent ES and an aggregate M."""
        return reconstruct_from_latent(
            self.u, aggregate, representation=self.representation
        )

    @property
    def n_used_edges(self) -> int:
        return int(self.used_edge_relations.size)

    def target_relation_matrix(self) -> np.ndarray:
        """Return R_ES for `RelationMatrixState.apply_feedback(..., mode="target")`."""
        return self.global_relation_matrix.copy()

    def delta_relation_matrix(self) -> np.ndarray:
        """Return R_ES - R_observed on used edges for delta-style feedback.

        Unused entries are NaN.  This sign convention is directly suitable for

            state.apply_feedback(delta, mode="delta", ...).
        """
        delta = np.full_like(self.residual_matrix, np.nan, dtype=float)
        mask = np.isfinite(self.residual_matrix)
        delta[mask] = -self.residual_matrix[mask]
        np.fill_diagonal(delta, 0.0)
        return delta

    def summary(self) -> Dict[str, Any]:
        """Concise logging/demo summary."""
        return {
            "representation": self.representation,
            "n_systems": self.n_systems,
            "n_used_edges": self.n_used_edges,
            "connected": self.connected,
            "globally_identified": self.globally_identified,
            "weighted_rmse": self.weighted_rmse,
            "normalized_consistency_error": self.normalized_consistency_error,
            "mean_absolute_residual": self.mean_absolute_residual,
            "max_absolute_residual": self.max_absolute_residual,
            "solver_method": self.solver_method,
        }


# ============================================================
# Graph utilities
# ============================================================


def _connected_components(
    n_systems: int,
    edge_index: np.ndarray,
) -> List[List[int]]:
    """Return connected components using only supplied edges."""
    adjacency: List[List[int]] = [[] for _ in range(n_systems)]

    if edge_index.size:
        for i, j in edge_index.T:
            i = int(i)
            j = int(j)
            adjacency[i].append(j)
            adjacency[j].append(i)

    seen = np.zeros(n_systems, dtype=bool)
    components: List[List[int]] = []

    for start in range(n_systems):
        if seen[start]:
            continue

        stack = [start]
        seen[start] = True
        component: List[int] = []

        while stack:
            node = stack.pop()
            component.append(node)

            for neighbour in adjacency[node]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(neighbour)

        component.sort()
        components.append(component)

    return components


def _format_components(
    components: Sequence[Sequence[int]],
    system_names: Sequence[str],
) -> str:
    groups = []
    for comp in components:
        groups.append(
            "[" + ", ".join(f"{idx}:{system_names[idx]}" for idx in comp) + "]"
        )
    return ", ".join(groups)


# ============================================================
# Linear-algebra utilities
# ============================================================


def _build_incidence_matrix(
    n_systems: int,
    edge_index: np.ndarray,
) -> np.ndarray:
    """Create oriented incidence matrix B with +1 at i and -1 at j."""
    n_edges = edge_index.shape[1]
    B = np.zeros((n_edges, n_systems), dtype=float)

    for e, (i, j) in enumerate(edge_index.T):
        B[e, int(i)] = 1.0
        B[e, int(j)] = -1.0

    return B


def _solve_connected_kkt(
    laplacian: np.ndarray,
    rhs: np.ndarray,
    regularization: float,
) -> Tuple[np.ndarray, str]:
    """Solve connected weighted least squares with sum(u)=0 gauge."""
    n = len(rhs)

    L = laplacian.copy()
    if regularization > 0:
        L = L + float(regularization) * np.eye(n)

    # KKT system:
    #
    # [ L   1 ] [u]   [b]
    # [1^T  0 ] [λ] = [0]
    #
    # The final row enforces sum_i u_i = 0.
    K = np.zeros((n + 1, n + 1), dtype=float)
    K[:n, :n] = L
    K[:n, n] = 1.0
    K[n, :n] = 1.0

    target = np.zeros(n + 1, dtype=float)
    target[:n] = rhs

    try:
        solution = np.linalg.solve(K, target)
        method = "kkt_solve"
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(K, target, rcond=None)
        method = "kkt_lstsq"

    u = np.asarray(solution[:n], dtype=float)
    u -= np.mean(u)  # numerical cleanup of the gauge
    return u, method


def _solve_minimum_norm(
    laplacian: np.ndarray,
    rhs: np.ndarray,
    regularization: float,
) -> Tuple[np.ndarray, str]:
    """Diagnostic solution for a disconnected graph.

    The weighted Laplacian is singular by one dimension per connected
    component.  The Moore-Penrose solution chooses an arbitrary minimum-norm
    gauge for those component offsets.  Therefore the resulting cross-component
    gamma values are NOT globally identified.
    """
    n = len(rhs)

    if regularization > 0:
        L = laplacian + float(regularization) * np.eye(n)
        try:
            u = np.linalg.solve(L, rhs)
            method = "ridge_solve_disconnected"
        except np.linalg.LinAlgError:
            u, *_ = np.linalg.lstsq(L, rhs, rcond=None)
            method = "ridge_lstsq_disconnected"
    else:
        u = np.linalg.pinv(laplacian) @ rhs
        method = "laplacian_pinv_disconnected"

    u = np.asarray(u, dtype=float)
    u -= np.mean(u)
    return u, method


def _stable_softmax(u: np.ndarray) -> np.ndarray:
    shifted = np.asarray(u, dtype=float) - float(np.max(u))
    exp_u = np.exp(shifted)
    total = float(exp_u.sum())

    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Softmax normalization failed.")

    return exp_u / total


def _laplacian_diagnostics(laplacian: np.ndarray) -> Dict[str, float]:
    """Spectral diagnostics of the weighted graph Laplacian."""
    eigvals = np.linalg.eigvalsh(0.5 * (laplacian + laplacian.T))
    eigvals = np.asarray(eigvals, dtype=float)

    tol = max(1e-12, float(np.max(np.abs(eigvals))) * 1e-12)
    positive = eigvals[eigvals > tol]

    if positive.size:
        condition = float(positive.max() / positive.min())
        smallest_positive = float(positive.min())
        largest = float(positive.max())
    else:
        condition = np.inf
        smallest_positive = 0.0
        largest = 0.0

    # For a connected graph, lambda_2 is the algebraic connectivity.
    sorted_vals = np.sort(np.maximum(eigvals, 0.0))
    algebraic_connectivity = (
        float(sorted_vals[1]) if len(sorted_vals) >= 2 else 0.0
    )

    return {
        "laplacian_rank": int(np.linalg.matrix_rank(laplacian)),
        "smallest_positive_eigenvalue": smallest_positive,
        "largest_eigenvalue": largest,
        "spectral_condition_number": condition,
        "algebraic_connectivity": algebraic_connectivity,
    }



def _resolve_anchor_latent(
    *,
    n_systems: int,
    representation: str,
    anchor_values: Optional[np.ndarray],
    anchor_latent: Optional[np.ndarray],
    anchor_epsilon: float,
) -> np.ndarray:
    """Return a centered latent ST vector used to place disconnected components.

    Exactly one of ``anchor_values`` and ``anchor_latent`` may be supplied.
    ``anchor_values`` is transformed with the same relation geometry as the ES.
    """
    if anchor_values is not None and anchor_latent is not None:
        raise ValueError("Provide at most one of anchor_values and anchor_latent.")

    if anchor_latent is not None:
        v = np.asarray(anchor_latent, dtype=float)
        if v.ndim != 1 or len(v) != int(n_systems):
            raise ValueError(
                "anchor_latent must be a one-dimensional vector with length n_systems."
            )
        if not np.all(np.isfinite(v)):
            raise ValueError("anchor_latent must contain only finite values.")
        # Match the solver's global zero-sum gauge.
        return np.asarray(v - float(np.mean(v)), dtype=float)

    if anchor_values is None:
        raise ValueError(
            "disconnected_policy='st_component_anchor' requires anchor_values "
            "or anchor_latent."
        )

    values = np.asarray(anchor_values, dtype=float)
    if values.ndim != 1 or len(values) != int(n_systems):
        raise ValueError(
            "anchor_values must be a one-dimensional vector with length n_systems."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("anchor_values must contain only finite values.")

    return centered_latent_state(
        values,
        representation=representation,
        epsilon=float(anchor_epsilon),
    )


def _solve_disconnected_with_st_component_anchor(
    laplacian: np.ndarray,
    rhs: np.ndarray,
    components: List[List[int]],
    anchor_latent: np.ndarray,
    regularization: float,
) -> Tuple[np.ndarray, str, np.ndarray, np.ndarray]:
    """Identify disconnected component offsets from the current observed ST.

    Relation edges determine the within-component shape.  The otherwise
    unidentified component offsets are fixed by preserving each component's
    mean latent position from the current state:

        mean_{i in C_k} u_i = mean_{i in C_k} v_{t,i}.

    Equivalently, with indicator constraints C u = C v_t, the augmented KKT
    system is

        [ L + lambda I   C^T ] [u ] = [b    ]
        [ C              0   ] [mu]   [C v_t].

    Because the anchor latent vector is globally centered, these component
    constraints also imply sum_i u_i = 0.
    """
    n = len(rhs)
    k = len(components)

    C = np.zeros((k, n), dtype=float)
    for row, comp in enumerate(components):
        C[row, np.asarray(comp, dtype=int)] = 1.0

    target_sums = C @ np.asarray(anchor_latent, dtype=float)

    L = np.asarray(laplacian, dtype=float).copy()
    if regularization > 0:
        L = L + float(regularization) * np.eye(n)

    kkt = np.block(
        [
            [L, C.T],
            [C, np.zeros((k, k), dtype=float)],
        ]
    )
    target = np.concatenate([np.asarray(rhs, dtype=float), target_sums])

    try:
        solution = np.linalg.solve(kkt, target)
        method = "st_component_anchor_kkt"
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(kkt, target, rcond=None)
        method = "st_component_anchor_lstsq"

    u = np.asarray(solution[:n], dtype=float)
    # The constraints should already imply the same zero-sum gauge as v_t.
    # Recenter only at floating-point precision.
    u -= float(np.mean(u))

    achieved_sums = C @ u
    return u, method, target_sums, achieved_sums


# ============================================================
# Main solver
# ============================================================


def solve_equilibrium_state(
    relation_state: RelationMatrixState,
    *,
    use_base: bool = False,
    weight_threshold: float = 0.0,
    regularization: float = 0.0,
    disconnected_policy: str = "error",
    representation: str = LOG_RATIO,
    anchor_values: Optional[np.ndarray] = None,
    anchor_latent: Optional[np.ndarray] = None,
    anchor_epsilon: float = 1e-8,
) -> EquilibriumResult:
    """Recover a global equilibrium state from pairwise relations.

    Parameters
    ----------
    relation_state : RelationMatrixState
        Output of `relation_matrix.py`.  It may come from either the
        statistical relation path or the neural relation path.

    use_base : bool, default=False
        False:
            solve from the current working relation state R^(k).

        True:
            solve from the original pairwise relation state R^(0), ignoring
            any later feedback updates to `values`.

    weight_threshold : float, default=0.0
        A valid edge participates only when

            w_ij > weight_threshold.

        This is useful with soft relation weights, where near-null edges may
        have very small but nonzero weights.

    regularization : float, default=0.0
        Optional small L2 penalty on u for numerical experiments.  The exact
        method corresponds to 0.  Normally leave this at zero.

    disconnected_policy : {"error", "minimum_norm", "st_component_anchor"}, default="error"
        "error":
            refuse to report a global ES when the relation graph is
            disconnected.

        "minimum_norm":
            return the Moore-Penrose/ridge minimum-norm solution for diagnostic
            purposes.  `globally_identified` will be False because relative
            offsets between components are not identified.

        "st_component_anchor":
            preserve the current-state (ST) latent mean of each disconnected
            component. Pairwise forecasts determine within-component structure;
            ST supplies only the otherwise unidentified between-component
            offsets.

    anchor_values, anchor_latent:
        Current observed system state used only by ``st_component_anchor`` when
        the positive-weight relation graph is disconnected.  ``anchor_values``
        is transformed with the same relation geometry; ``anchor_latent`` may
        be supplied directly.  At most one may be provided.

    anchor_epsilon:
        Numerical epsilon used when transforming ``anchor_values``.

    Returns
    -------
    EquilibriumResult
        Global state, latent relation coordinates u, globally consistent
        pairwise relations, local-global residuals, graph diagnostics, and
        feedback-ready targets.

    Notes
    -----
    The objective is

        min_u sum_e w_e (B_e u - r_e)^2

    with sum(u)=0 for a connected graph.

    The zero-sum gauge only selects one coordinate representative of the same
    pairwise differences.  For a disconnected graph, ``st_component_anchor``
    adds one constraint per component,

        mean_{i in C_k} u_i = mean_{i in C_k} v_{t,i},

    so the relation forecasts still determine all within-component differences
    while current ST determines only the missing component offsets.
    ``log_ratio`` maps u to simplex shares; signed geometries reconstruct raw
    levels from u and the aggregate forecast.
    """
    if not isinstance(relation_state, RelationMatrixState):
        raise TypeError("relation_state must be a RelationMatrixState.")

    representation = normalize_representation_name(representation)

    if weight_threshold < 0:
        raise ValueError("weight_threshold must be non-negative.")

    if regularization < 0:
        raise ValueError("regularization must be non-negative.")

    disconnected_policy = str(disconnected_policy).lower()
    if disconnected_policy not in {"error", "minimum_norm", "st_component_anchor"}:
        raise ValueError(
            "disconnected_policy must be one of 'error', 'minimum_norm', "
            "or 'st_component_anchor'."
        )

    anchor_epsilon = float(anchor_epsilon)
    if not np.isfinite(anchor_epsilon) or anchor_epsilon <= 0:
        raise ValueError("anchor_epsilon must be a finite positive value.")

    n = relation_state.n_systems

    edge_index, edge_r, edge_weight, edge_valid = relation_state.edge_arrays(
        use_base=use_base,
        valid_only=False,
    )

    # A usable edge must be explicitly valid, finite, and carry weight above
    # the requested threshold.
    used = (
        edge_valid
        & np.isfinite(edge_r)
        & np.isfinite(edge_weight)
        & (edge_weight > float(weight_threshold))
    )

    used_edge_index = edge_index[:, used]
    used_r = np.asarray(edge_r[used], dtype=float)
    used_w = np.asarray(edge_weight[used], dtype=float)

    if used_r.size == 0:
        raise ValueError(
            "No positive-weight valid relation edges are available for "
            "equilibrium estimation."
        )

    components = _connected_components(n, used_edge_index)
    connected = len(components) == 1

    if not connected and disconnected_policy == "error":
        component_text = _format_components(
            components,
            relation_state.system_names,
        )
        raise ValueError(
            "The positive-weight relation graph is disconnected, so one global "
            "equilibrium state is not identifiable. Connected components: "
            f"{component_text}. Consider lowering `weight_threshold`, retaining "
            "more soft edges, or handling components separately."
        )

    B = _build_incidence_matrix(n, used_edge_index)

    # L = B^T W B and b = B^T W r without explicitly constructing a dense
    # diagonal W matrix.
    weighted_B = used_w[:, None] * B
    laplacian = B.T @ weighted_B
    rhs = B.T @ (used_w * used_r)

    anchor_used = False
    anchor_target_sums = np.asarray([], dtype=float)
    anchor_achieved_sums = np.asarray([], dtype=float)
    resolved_anchor_latent = None

    if connected:
        u, solver_method = _solve_connected_kkt(
            laplacian,
            rhs,
            regularization,
        )
        globally_identified = True
    elif disconnected_policy == "st_component_anchor":
        resolved_anchor_latent = _resolve_anchor_latent(
            n_systems=n,
            representation=representation,
            anchor_values=anchor_values,
            anchor_latent=anchor_latent,
            anchor_epsilon=anchor_epsilon,
        )
        (
            u,
            solver_method,
            anchor_target_sums,
            anchor_achieved_sums,
        ) = _solve_disconnected_with_st_component_anchor(
            laplacian,
            rhs,
            components,
            resolved_anchor_latent,
            regularization,
        )
        anchor_used = True
        # The relation graph alone is disconnected, but the augmented
        # relation + ST-anchor system identifies one global latent state.
        globally_identified = True
    else:
        u, solver_method = _solve_minimum_norm(
            laplacian,
            rhs,
            regularization,
        )
        globally_identified = False

    if not np.all(np.isfinite(u)):
        raise RuntimeError("Equilibrium solver produced non-finite latent scores.")

    # ``gamma`` remains for backward compatibility and is meaningful only for
    # the multiplicative/log-ratio representation.  Other geometries expose
    # the globally identified relative state through ``u`` and reconstruct
    # raw levels using their own inverse geometry plus the aggregate forecast.
    gamma = simplex_state_from_latent(u, representation)

    # The ES-implied relation is globally cycle-consistent for every pair.
    global_R = u[:, None] - u[None, :]
    np.fill_diagonal(global_R, 0.0)

    observed_R = relation_state.full_relation_matrix(
        use_base=use_base,
        invalid_value=np.nan,
    )

    residual_R = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(residual_R, 0.0)

    edge_fitted = np.empty_like(used_r)
    edge_residual = np.empty_like(used_r)

    for e, (i, j) in enumerate(used_edge_index.T):
        i = int(i)
        j = int(j)

        fitted = float(global_R[i, j])
        residual = float(used_r[e] - fitted)

        edge_fitted[e] = fitted
        edge_residual[e] = residual

        residual_R[i, j] = residual
        residual_R[j, i] = -residual

    total_weight = float(np.sum(used_w))
    weighted_sse = float(np.sum(used_w * edge_residual**2))
    weighted_mse = weighted_sse / total_weight
    weighted_rmse = float(np.sqrt(weighted_mse))

    mean_absolute_residual = float(np.mean(np.abs(edge_residual)))
    max_absolute_residual = float(np.max(np.abs(edge_residual)))

    weighted_rms_relation = float(
        np.sqrt(np.sum(used_w * used_r**2) / total_weight)
    )
    scale_floor = 1e-12
    normalized_consistency_error = float(
        weighted_rmse / max(weighted_rms_relation, scale_floor)
    )

    spectral = _laplacian_diagnostics(laplacian)

    diagnostics: Dict[str, Any] = {
        "representation": representation,
        "n_possible_edges": relation_state.n_possible_edges,
        "n_valid_edges_in_state": relation_state.n_valid_edges,
        "n_used_edges": int(used_r.size),
        "edge_density_used": float(
            used_r.size / relation_state.n_possible_edges
        ),
        "total_edge_weight": total_weight,
        "weighted_rms_observed_relation": weighted_rms_relation,
        "regularization": float(regularization),
        "relation_state_iteration": int(relation_state.iteration),
        "components_named": [
            [relation_state.system_names[idx] for idx in comp]
            for comp in components
        ],
        "n_components": int(len(components)),
        "relation_graph_connected": bool(connected),
        "st_component_anchor_used": bool(anchor_used),
        "identification_source": (
            "relations"
            if connected
            else (
                "relations+st_component_anchor"
                if anchor_used
                else "minimum_norm_not_globally_identified"
            )
        ),
        "component_anchor_target_sums": np.asarray(
            anchor_target_sums, dtype=float
        ).tolist(),
        "component_anchor_achieved_sums": np.asarray(
            anchor_achieved_sums, dtype=float
        ).tolist(),
        "component_anchor_max_abs_sum_error": (
            float(np.max(np.abs(anchor_achieved_sums - anchor_target_sums)))
            if anchor_used and anchor_target_sums.size
            else 0.0
        ),
        **spectral,
    }

    return EquilibriumResult(
        representation=representation,
        gamma=np.asarray(gamma, dtype=float),
        u=np.asarray(u, dtype=float),

        observed_relation_matrix=np.asarray(observed_R, dtype=float),
        global_relation_matrix=np.asarray(global_R, dtype=float),
        residual_matrix=np.asarray(residual_R, dtype=float),

        weighted_sse=weighted_sse,
        weighted_mse=weighted_mse,
        weighted_rmse=weighted_rmse,
        normalized_consistency_error=normalized_consistency_error,

        mean_absolute_residual=mean_absolute_residual,
        max_absolute_residual=max_absolute_residual,

        connected=connected,
        globally_identified=globally_identified,
        components=[list(comp) for comp in components],

        used_edge_index=np.asarray(used_edge_index, dtype=int),
        used_edge_relations=np.asarray(used_r, dtype=float),
        used_edge_weights=np.asarray(used_w, dtype=float),

        laplacian=np.asarray(laplacian, dtype=float),
        rhs=np.asarray(rhs, dtype=float),

        system_names=list(relation_state.system_names),
        use_base=bool(use_base),
        weight_threshold=float(weight_threshold),
        solver_method=solver_method,

        diagnostics=diagnostics,
    )


# ============================================================
# Convenience wrapper
# ============================================================


def solve_es(
    relation_state: RelationMatrixState,
    **kwargs: Any,
) -> EquilibriumResult:
    """Short alias for `solve_equilibrium_state(...)`."""
    return solve_equilibrium_state(relation_state, **kwargs)


# ============================================================
# Standalone smoke test
# ============================================================


if __name__ == "__main__":
    from relation_matrix import PairwiseRelationEdge, build_relation_matrix

    # Known equilibrium state.
    gamma_true = np.array([0.40, 0.30, 0.20, 0.10], dtype=float)
    u_true = np.log(gamma_true)

    edges = []
    for i in range(len(gamma_true)):
        for j in range(i + 1, len(gamma_true)):
            r_ij = float(u_true[i] - u_true[j])
            edges.append(
                PairwiseRelationEdge(
                    i=i,
                    j=j,
                    r=r_ij,
                    weight=1.0,
                    is_valid=True,
                )
            )

    state = build_relation_matrix(
        n_systems=4,
        edges=edges,
        system_names=["S1", "S2", "S3", "S4"],
        require_complete_input=True,
    )

    result = solve_equilibrium_state(state)

    print("true gamma:     ", gamma_true)
    print("estimated gamma:", np.round(result.gamma, 8))
    print("max |error|:    ", np.max(np.abs(result.gamma - gamma_true)))
    print("summary:        ", result.summary())
