"""
relation_matrix.py
==================

Construction and state management of the upper-triangular pairwise relation
matrix used by history-driven equilibrium state estimation.

This module deliberately separates relation estimation from global equilibrium
estimation.

Forward path (current implementation)
-------------------------------------
    pairwise relation estimators
        -> edge relations r_ij and reliability weights w_ij
        -> upper-triangular relation matrix R
        -> downstream global equilibrium-state solver

Feedback-ready path (future extension)
--------------------------------------
    pairwise relation layer
        -> R^(0)
        -> global layer / ES
        -> feedback correction Delta R^(k)
        -> R^(k+1)
        -> global layer / ES
        -> ...

The first pairwise stage can therefore later be interpreted as an edge/neuron
layer containing N(N-1)/2 independent pairwise units.  This module exposes
both matrix and edge-vector views so that a future graph/neural implementation
can use the same data structure without changing the pairwise estimators.

Important conventions
---------------------
1. Only unique unordered pairs i < j are stored as primary relations.
2. The canonical relation is directional:

       r_ij ~= log(gamma_i^* / gamma_j^*)

   so the implied reverse relation is

       r_ji = -r_ij.

3. Reliability is symmetric:

       w_ji = w_ij.

4. Invalid / rejected relations are represented by NaN in the relation matrix,
   weight 0, and valid=False.
5. The base pairwise estimates are preserved separately from the current
   relation state.  Feedback can alter the current state without destroying
   the original first-layer estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


# ============================================================
# Edge-level record
# ============================================================


@dataclass(frozen=True)
class PairwiseRelationEdge:
    """One unique pairwise relation corresponding to an upper-triangular cell.

    Parameters
    ----------
    i, j : int
        Zero-based system indices.  They may be supplied in either order; the
        builder canonicalizes them to i < j and changes the sign of r when the
        order is reversed.

    r : float
        Canonical directional pairwise relation.  For a valid edge,

            r ~= log(gamma_i^* / gamma_j^*).

    weight : float, default=1.0
        Reliability of the relation.  Invalid edges are forced to weight 0.

    is_valid : bool, default=True
        Whether the edge should participate in downstream global estimation.

    metadata : dict
        Optional relation-family/model diagnostics.  These values do not alter
        the matrix numerically but are kept for analysis and future feedback or
        neural routing mechanisms.
    """

    i: int
    j: int
    r: float
    weight: float = 1.0
    is_valid: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ============================================================
# Main relation-matrix state
# ============================================================


@dataclass
class RelationMatrixState:
    """Feedback-ready state of the pairwise relation matrix.

    The object keeps two numerical relation matrices:

    ``base_values``
        Immutable-by-convention output of the pairwise relation layer.

    ``values``
        Current working relation state.  Initially identical to
        ``base_values`` but may later be modified by feedback iterations.

    Only upper-triangular entries i < j are primary.  Lower-triangular entries
    are left as NaN internally and are generated on demand using antisymmetry.
    """

    n_systems: int
    system_names: List[str]

    base_values: np.ndarray
    values: np.ndarray
    weights: np.ndarray
    valid_mask: np.ndarray

    edge_metadata: Dict[Tuple[int, int], Dict[str, Any]] = field(default_factory=dict)

    iteration: int = 0
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    def __post_init__(self) -> None:
        n = self.n_systems

        if n < 2:
            raise ValueError("n_systems must be at least 2.")

        if len(self.system_names) != n:
            raise ValueError(
                f"system_names must contain {n} names; got {len(self.system_names)}."
            )

        for name, array in (
            ("base_values", self.base_values),
            ("values", self.values),
            ("weights", self.weights),
            ("valid_mask", self.valid_mask),
        ):
            if np.asarray(array).shape != (n, n):
                raise ValueError(
                    f"{name} must have shape ({n}, {n}); got {np.asarray(array).shape}."
                )

        # Defensive copies prevent accidental mutation through caller-owned arrays.
        self.base_values = np.asarray(self.base_values, dtype=float).copy()
        self.values = np.asarray(self.values, dtype=float).copy()
        self.weights = np.asarray(self.weights, dtype=float).copy()
        self.valid_mask = np.asarray(self.valid_mask, dtype=bool).copy()

        if len(set(self.system_names)) != n:
            raise ValueError("system_names must be unique.")

        self._enforce_upper_storage()

    def _enforce_upper_storage(self) -> None:
        """Enforce the module's upper-triangular storage convention."""
        n = self.n_systems

        for i in range(n):
            # Diagonal is structurally zero, not a learned relation.
            self.base_values[i, i] = 0.0
            self.values[i, i] = 0.0
            self.weights[i, i] = 0.0
            self.valid_mask[i, i] = False

            for j in range(i):
                # Lower triangle is derived, never primary storage.
                self.base_values[i, j] = np.nan
                self.values[i, j] = np.nan
                self.weights[i, j] = 0.0
                self.valid_mask[i, j] = False

        # Invalid upper edges must not influence downstream solvers.
        upper = np.triu(np.ones((n, n), dtype=bool), k=1)
        invalid = upper & (~self.valid_mask)
        self.base_values[invalid] = np.nan
        self.values[invalid] = np.nan
        self.weights[invalid] = 0.0

        if np.any(self.weights < 0):
            raise ValueError("Relation weights must be non-negative.")

    # --------------------------------------------------------
    # Structural properties
    # --------------------------------------------------------

    @property
    def n_possible_edges(self) -> int:
        """Number of unique pairs N(N-1)/2."""
        n = self.n_systems
        return n * (n - 1) // 2

    @property
    def n_valid_edges(self) -> int:
        """Number of currently valid upper-triangular relations."""
        return int(np.sum(np.triu(self.valid_mask, k=1)))

    @property
    def density(self) -> float:
        """Fraction of possible pairwise edges currently retained."""
        return self.n_valid_edges / self.n_possible_edges

    # --------------------------------------------------------
    # Matrix views
    # --------------------------------------------------------

    def upper_matrix(
        self,
        *,
        use_base: bool = False,
        copy: bool = True,
    ) -> np.ndarray:
        """Return the stored upper-triangular relation matrix.

        Lower-triangular entries are NaN.  Diagonal entries are zero.
        """
        matrix = self.base_values if use_base else self.values
        return matrix.copy() if copy else matrix

    def full_relation_matrix(
        self,
        *,
        use_base: bool = False,
        invalid_value: float = np.nan,
    ) -> np.ndarray:
        """Expand the triangular matrix to a full antisymmetric matrix.

        For every valid pair i < j,

            R[j, i] = -R[i, j].

        Invalid relations receive ``invalid_value`` in both directions.
        """
        upper = self.base_values if use_base else self.values
        n = self.n_systems
        full = np.full((n, n), invalid_value, dtype=float)
        np.fill_diagonal(full, 0.0)

        for i in range(n):
            for j in range(i + 1, n):
                if self.valid_mask[i, j] and np.isfinite(upper[i, j]):
                    full[i, j] = upper[i, j]
                    full[j, i] = -upper[i, j]

        return full

    def full_weight_matrix(self) -> np.ndarray:
        """Return the symmetric reliability matrix."""
        n = self.n_systems
        full = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(i + 1, n):
                if self.valid_mask[i, j]:
                    full[i, j] = self.weights[i, j]
                    full[j, i] = self.weights[i, j]

        return full

    # --------------------------------------------------------
    # Edge / neuron-layer views
    # --------------------------------------------------------

    def edge_arrays(
        self,
        *,
        use_base: bool = False,
        valid_only: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a deterministic edge representation of the matrix.

        Returns
        -------
        edge_index : np.ndarray, shape (2, E)
            Zero-based node pairs.  Pair ordering is lexicographic with i < j.

        edge_r : np.ndarray, shape (E,)
            Pairwise relation values.

        edge_weight : np.ndarray, shape (E,)
            Reliability weights.

        edge_valid : np.ndarray, shape (E,)
            Validity mask.

        Notes
        -----
        This representation is intentionally compatible with a future neural
        or graph layer in which each unique pair acts as one first-layer unit.
        """
        matrix = self.base_values if use_base else self.values

        pairs: List[Tuple[int, int]] = []
        relations: List[float] = []
        weights: List[float] = []
        validity: List[bool] = []

        for i in range(self.n_systems):
            for j in range(i + 1, self.n_systems):
                valid = bool(self.valid_mask[i, j])
                if valid_only and not valid:
                    continue

                pairs.append((i, j))
                relations.append(float(matrix[i, j]) if valid else np.nan)
                weights.append(float(self.weights[i, j]) if valid else 0.0)
                validity.append(valid)

        edge_index = np.asarray(pairs, dtype=int).T
        edge_r = np.asarray(relations, dtype=float)
        edge_weight = np.asarray(weights, dtype=float)
        edge_valid = np.asarray(validity, dtype=bool)

        return edge_index, edge_r, edge_weight, edge_valid

    def edge_vector(
        self,
        *,
        use_base: bool = False,
        valid_only: bool = False,
    ) -> np.ndarray:
        """Return only the deterministic vector of edge relations."""
        _, edge_r, _, _ = self.edge_arrays(
            use_base=use_base,
            valid_only=valid_only,
        )
        return edge_r

    def edge_pair_order(self, *, valid_only: bool = False) -> List[Tuple[int, int]]:
        """Return the pair order corresponding to ``edge_vector``."""
        edge_index, _, _, _ = self.edge_arrays(valid_only=valid_only)
        return [tuple(x) for x in edge_index.T.tolist()]

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    def get_edge_metadata(self, i: int, j: int) -> Dict[str, Any]:
        """Return metadata for one unordered pair."""
        i, j, _ = _canonical_pair(i, j, 0.0)
        return dict(self.edge_metadata.get((i, j), {}))

    # --------------------------------------------------------
    # Feedback-ready update interface
    # --------------------------------------------------------

    def apply_feedback(
        self,
        feedback: Union[np.ndarray, Sequence[float], Mapping[Tuple[int, int], float]],
        *,
        mode: str = "delta",
        step_size: float = 1.0,
        valid_only: bool = True,
        clip: Optional[Tuple[float, float]] = None,
        note: Optional[str] = None,
    ) -> None:
        """Update the current relation state using external/global feedback.

        This method intentionally does NOT define how feedback is computed.
        A future ES/global/neural layer can calculate corrections and pass them
        back here.  Keeping the update mechanism separate prevents the current
        forward-only implementation from hard-coding one feedback algorithm.

        Parameters
        ----------
        feedback : array-like or mapping
            Accepted forms:

            1. Edge vector of length N(N-1)/2 in lexicographic i<j order.
            2. N x N matrix; only upper-triangular entries are read.
            3. Mapping {(i, j): value, ...}.

        mode : {"delta", "target"}, default="delta"
            ``delta``:

                r^(k+1) = r^(k) + step_size * feedback

            ``target``:

                r^(k+1) = (1-step_size) r^(k)
                          + step_size * feedback_target

        step_size : float, default=1.0
            Feedback strength / damping coefficient.  For target mode it must
            lie in [0, 1].  For delta mode it must be non-negative.

        valid_only : bool, default=True
            If True, invalid/null edges are frozen.  This is the safest default
            because the downstream solver should not manufacture relationships
            that the first layer rejected.

        clip : (lower, upper), optional
            Optional numerical clipping after the update.

        note : str, optional
            Human-readable description stored in ``feedback_history``.
        """
        mode = mode.lower()
        if mode not in {"delta", "target"}:
            raise ValueError("mode must be either 'delta' or 'target'.")

        if step_size < 0:
            raise ValueError("step_size must be non-negative.")
        if mode == "target" and step_size > 1:
            raise ValueError("target-mode step_size must lie in [0, 1].")

        feedback_upper = self._coerce_feedback_to_upper(feedback)

        before = self.values.copy()

        for i in range(self.n_systems):
            for j in range(i + 1, self.n_systems):
                if valid_only and not self.valid_mask[i, j]:
                    continue

                fb = feedback_upper[i, j]
                if not np.isfinite(fb):
                    continue

                # A previously invalid edge can only be updated when explicitly
                # requested through valid_only=False.  In that case it becomes
                # a feedback-created working edge, but base_values is untouched.
                if not np.isfinite(self.values[i, j]):
                    if valid_only:
                        continue
                    if mode == "delta":
                        current = 0.0
                        self.values[i, j] = current + step_size * fb
                    else:
                        self.values[i, j] = fb
                    self.valid_mask[i, j] = True
                    # Feedback-created edges get zero confidence by default;
                    # a future algorithm may explicitly update reliability.
                    self.weights[i, j] = 0.0
                    continue

                if mode == "delta":
                    self.values[i, j] = self.values[i, j] + step_size * fb
                else:
                    self.values[i, j] = (
                        (1.0 - step_size) * self.values[i, j]
                        + step_size * fb
                    )

                if clip is not None:
                    lo, hi = clip
                    self.values[i, j] = float(np.clip(self.values[i, j], lo, hi))

        self.iteration += 1
        self.feedback_history.append(
            {
                "iteration": self.iteration,
                "mode": mode,
                "step_size": float(step_size),
                "note": note,
                "max_absolute_change": _max_abs_upper_change(before, self.values),
            }
        )
        self._enforce_upper_storage()

    def update_weights(
        self,
        weights: Union[np.ndarray, Sequence[float], Mapping[Tuple[int, int], float]],
        *,
        blend: float = 1.0,
        valid_only: bool = True,
    ) -> None:
        """Optionally update relation reliability independently of r values.

        This is provided for future feedback/gating mechanisms.  ``blend=1``
        replaces weights; smaller values damp the update.
        """
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must lie in [0, 1].")

        upper = self._coerce_feedback_to_upper(weights)

        for i in range(self.n_systems):
            for j in range(i + 1, self.n_systems):
                if valid_only and not self.valid_mask[i, j]:
                    continue
                new_w = upper[i, j]
                if not np.isfinite(new_w):
                    continue
                if new_w < 0:
                    raise ValueError("Relation weights must be non-negative.")
                self.weights[i, j] = (
                    (1.0 - blend) * self.weights[i, j] + blend * new_w
                )

        self._enforce_upper_storage()

    def reset_feedback(self, *, clear_history: bool = False) -> None:
        """Restore the working relation state to original pairwise estimates."""
        self.values = self.base_values.copy()
        self.iteration = 0
        if clear_history:
            self.feedback_history.clear()
        self._enforce_upper_storage()

    def _coerce_feedback_to_upper(
        self,
        feedback: Union[np.ndarray, Sequence[float], Mapping[Tuple[int, int], float]],
    ) -> np.ndarray:
        """Convert supported feedback formats to an N x N upper matrix."""
        n = self.n_systems
        out = np.full((n, n), np.nan, dtype=float)
        np.fill_diagonal(out, 0.0)

        if isinstance(feedback, Mapping):
            for (i, j), value in feedback.items():
                i2, j2, v2 = _canonical_pair(int(i), int(j), float(value))
                _validate_indices(i2, j2, n)
                out[i2, j2] = v2
            return out

        array = np.asarray(feedback, dtype=float)

        if array.shape == (n, n):
            for i in range(n):
                for j in range(i + 1, n):
                    out[i, j] = array[i, j]
            return out

        flat = array.reshape(-1)
        expected = self.n_possible_edges
        if len(flat) != expected:
            raise ValueError(
                "Feedback must be an N x N matrix, an edge vector of length "
                f"{expected}, or a mapping of pair indices; got shape {array.shape}."
            )

        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                out[i, j] = flat[k]
                k += 1

        return out

    # --------------------------------------------------------
    # Convenience summary
    # --------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return concise structural information for logging/demo use."""
        return {
            "n_systems": self.n_systems,
            "n_possible_edges": self.n_possible_edges,
            "n_valid_edges": self.n_valid_edges,
            "density": self.density,
            "feedback_iteration": self.iteration,
        }


# ============================================================
# Builders
# ============================================================


def build_relation_matrix(
    n_systems: int,
    edges: Iterable[PairwiseRelationEdge],
    *,
    system_names: Optional[Sequence[str]] = None,
    require_complete_input: bool = False,
) -> RelationMatrixState:
    """Build a feedback-ready upper-triangular relation matrix.

    Parameters
    ----------
    n_systems : int
        Number of systems N.

    edges : iterable[PairwiseRelationEdge]
        Pairwise relation records.  Each unordered pair may appear at most once.
        Reversed records j,i are accepted and canonicalized to i<j by changing
        the sign of r.

    system_names : sequence[str], optional
        Names of the systems.  Defaults to S1, ..., SN.

    require_complete_input : bool, default=False
        If True, require a record for every one of the N(N-1)/2 possible pairs,
        even when some records are invalid/null.  This is useful for verifying
        a full first-layer computation.  If False, missing pairs are simply
        represented as invalid edges.
    """
    if n_systems < 2:
        raise ValueError("n_systems must be at least 2.")

    if system_names is None:
        names = [f"S{i + 1}" for i in range(n_systems)]
    else:
        names = [str(x) for x in system_names]
        if len(names) != n_systems:
            raise ValueError(
                f"system_names must contain {n_systems} names; got {len(names)}."
            )

    base = np.full((n_systems, n_systems), np.nan, dtype=float)
    current = np.full((n_systems, n_systems), np.nan, dtype=float)
    weights = np.zeros((n_systems, n_systems), dtype=float)
    valid = np.zeros((n_systems, n_systems), dtype=bool)

    np.fill_diagonal(base, 0.0)
    np.fill_diagonal(current, 0.0)

    metadata: Dict[Tuple[int, int], Dict[str, Any]] = {}
    seen: set[Tuple[int, int]] = set()

    for edge in edges:
        if not isinstance(edge, PairwiseRelationEdge):
            raise TypeError(
                "Every item in edges must be a PairwiseRelationEdge instance."
            )

        i, j, r = _canonical_pair(edge.i, edge.j, edge.r)
        _validate_indices(i, j, n_systems)

        key = (i, j)
        if key in seen:
            raise ValueError(f"Duplicate pairwise relation supplied for pair {key}.")
        seen.add(key)

        is_valid = bool(edge.is_valid and np.isfinite(r))
        weight = float(edge.weight)

        if weight < 0:
            raise ValueError(f"Negative relation weight for pair {key}: {weight}")

        if not is_valid:
            r = np.nan
            weight = 0.0

        base[i, j] = r
        current[i, j] = r
        weights[i, j] = weight
        valid[i, j] = is_valid
        metadata[key] = dict(edge.metadata)

    expected = n_systems * (n_systems - 1) // 2
    if require_complete_input and len(seen) != expected:
        missing = [
            (i, j)
            for i in range(n_systems)
            for j in range(i + 1, n_systems)
            if (i, j) not in seen
        ]
        raise ValueError(
            f"Expected {expected} pair records but received {len(seen)}. "
            f"Missing pairs: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )

    return RelationMatrixState(
        n_systems=n_systems,
        system_names=names,
        base_values=base,
        values=current,
        weights=weights,
        valid_mask=valid,
        edge_metadata=metadata,
    )


def build_relation_matrix_from_results(
    pairs: Sequence[Any],
    relation_results: Sequence[Any],
    *,
    n_systems: Optional[int] = None,
    system_names: Optional[Sequence[str]] = None,
    require_complete_input: bool = True,
) -> RelationMatrixState:
    """Convenience bridge from pair objects + ``RelationResult`` objects.

    This function intentionally uses duck typing rather than importing
    ``pairwise_systems`` or ``pairwise_relation``.  It therefore keeps this
    module independent and avoids circular/tight coupling.

    Expected pair attributes
    ------------------------
        pair.i
        pair.j
        optionally pair.system_i, pair.system_j

    Expected result attributes
    --------------------------
        result.r
        result.weight
        result.is_valid
        optionally selected_family, selected_model, validation_loss,
        normalized_rmse, parameters, is_custom, null_loss,
        improvement_over_null.
    """
    if len(pairs) != len(relation_results):
        raise ValueError(
            "pairs and relation_results must have the same length; "
            f"got {len(pairs)} and {len(relation_results)}."
        )

    if len(pairs) == 0:
        raise ValueError("At least one pair/result is required.")

    inferred_max = max(max(int(p.i), int(p.j)) for p in pairs)
    inferred_n = inferred_max + 1
    if n_systems is None:
        n_systems = inferred_n
    elif n_systems < inferred_n:
        raise ValueError(
            f"n_systems={n_systems} is too small for observed pair index {inferred_max}."
        )

    edges: List[PairwiseRelationEdge] = []

    for pair, result in zip(pairs, relation_results):
        metadata = {
            "system_i": getattr(pair, "system_i", None),
            "system_j": getattr(pair, "system_j", None),
            "selected_family": getattr(result, "selected_family", None),
            "selected_model": getattr(result, "selected_model", None),
            "validation_loss": getattr(result, "validation_loss", None),
            "normalized_rmse": getattr(result, "normalized_rmse", None),
            "parameters": getattr(result, "parameters", None),
            "is_custom": getattr(result, "is_custom", None),
            "null_loss": getattr(result, "null_loss", None),
            "improvement_over_null": getattr(result, "improvement_over_null", None),
        }

        edges.append(
            PairwiseRelationEdge(
                i=int(pair.i),
                j=int(pair.j),
                r=float(getattr(result, "r")),
                weight=float(getattr(result, "weight", 1.0)),
                is_valid=bool(getattr(result, "is_valid", True)),
                metadata=metadata,
            )
        )

    return build_relation_matrix(
        n_systems=n_systems,
        edges=edges,
        system_names=system_names,
        require_complete_input=require_complete_input,
    )


# ============================================================
# Utility functions
# ============================================================


def _canonical_pair(i: int, j: int, r: float) -> Tuple[int, int, float]:
    """Convert an oriented pair to canonical i<j storage."""
    i = int(i)
    j = int(j)
    r = float(r)

    if i == j:
        raise ValueError("A pairwise relation must involve two distinct systems.")

    if i < j:
        return i, j, r

    return j, i, -r


def _validate_indices(i: int, j: int, n_systems: int) -> None:
    if not (0 <= i < j < n_systems):
        raise IndexError(
            f"Invalid canonical pair ({i}, {j}) for n_systems={n_systems}."
        )


def _max_abs_upper_change(before: np.ndarray, after: np.ndarray) -> float:
    n = before.shape[0]
    changes: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            a = before[i, j]
            b = after[i, j]
            if np.isfinite(a) and np.isfinite(b):
                changes.append(abs(b - a))
    return float(max(changes)) if changes else 0.0


# ============================================================
# Standalone demonstration
# ============================================================


if __name__ == "__main__":
    # Four systems -> six pairwise first-layer outputs.
    example_edges = [
        PairwiseRelationEdge(0, 1, 0.20, 0.95, True, {"model": "stable"}),
        PairwiseRelationEdge(0, 2, 0.50, 0.90, True, {"model": "trend"}),
        PairwiseRelationEdge(0, 3, 0.80, 0.85, True, {"model": "VECM"}),
        PairwiseRelationEdge(1, 2, 0.25, 0.80, True, {"model": "ARIMA"}),
        PairwiseRelationEdge(1, 3, 0.55, 0.75, True, {"model": "spline"}),
        PairwiseRelationEdge(2, 3, np.nan, 0.0, False, {"model": "null"}),
    ]

    state = build_relation_matrix(
        n_systems=4,
        edges=example_edges,
        system_names=["S1", "S2", "S3", "S4"],
        require_complete_input=True,
    )

    print("Summary:")
    print(state.summary())

    print("\nUpper-triangular relation matrix:")
    print(state.upper_matrix())

    print("\nFull antisymmetric relation matrix:")
    print(state.full_relation_matrix())

    edge_index, edge_r, edge_w, edge_valid = state.edge_arrays()
    print("\nEdge/neuron view:")
    print("edge_index=\n", edge_index)
    print("r=", edge_r)
    print("w=", edge_w)
    print("valid=", edge_valid)

    # Example of a future backward/global feedback correction.  This does not
    # prescribe how feedback should be produced; it only demonstrates that the
    # matrix state can receive it without recomputing first-layer relations.
    delta = np.zeros(state.n_possible_edges)
    delta[0] = 0.01
    state.apply_feedback(
        delta,
        mode="delta",
        step_size=0.5,
        note="demo global-feedback correction",
    )

    print("\nAfter one feedback update:")
    print(state.upper_matrix())
    print(state.feedback_history[-1])