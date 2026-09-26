
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from equilibrium_solver import EquilibriumResult
from relation_representation import (
    LOG_RATIO,
    centered_latent_state,
    normalize_representation_name,
    reconstruct_from_latent,
    simplex_state_from_latent,
)


DIRECT_ALPHA = "alpha"
PERSISTENCE_RHO = "rho"
SUPPORTED_MODES = (DIRECT_ALPHA, PERSISTENCE_RHO)


@dataclass(frozen=True)
class EquilibriumAdjustmentResult:
    """Finite-horizon state obtained by partial movement from ST toward ES."""

    representation: str
    mode: str
    horizon: int

    alpha_h: float
    rho: Optional[float]

    current_values: np.ndarray
    current_latent: np.ndarray
    equilibrium_latent: np.ndarray
    adjusted_latent: np.ndarray

    initial_disequilibrium_norm: float
    remaining_disequilibrium_norm: float
    remaining_disequilibrium_ratio: float

    equilibrium_tolerance: float
    practically_at_equilibrium: bool

    @property
    def movement_fraction(self) -> float:
        """Fraction of the initial ST->ES gap removed by the forecast horizon."""
        return float(self.alpha_h)

    @property
    def persistence_fraction(self) -> float:
        """Fraction of the initial disequilibrium remaining at horizon h."""
        return float(1.0 - self.alpha_h)

    @property
    def gamma(self) -> np.ndarray:
        """Simplex state for log-ratio geometry; NaNs for other geometries."""
        return simplex_state_from_latent(
            self.adjusted_latent,
            representation=self.representation,
        )

    def reconstruct(self, aggregate: float) -> np.ndarray:
        """Reconstruct system values using the adjusted latent state and M_hat."""
        return reconstruct_from_latent(
            self.adjusted_latent,
            float(aggregate),
            representation=self.representation,
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "representation": self.representation,
            "mode": self.mode,
            "horizon": self.horizon,
            "alpha_h": self.alpha_h,
            "rho": self.rho,
            "movement_fraction": self.movement_fraction,
            "persistence_fraction": self.persistence_fraction,
            "initial_disequilibrium_norm": self.initial_disequilibrium_norm,
            "remaining_disequilibrium_norm": self.remaining_disequilibrium_norm,
            "remaining_disequilibrium_ratio": self.remaining_disequilibrium_ratio,
            "equilibrium_tolerance": self.equilibrium_tolerance,
            "practically_at_equilibrium": self.practically_at_equilibrium,
        }


def _validate_scalar_unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]; got {value}.")
    return value


def resolve_horizon_adjustment(
    *,
    horizon: int,
    alpha_h: Optional[float] = None,
    rho: Optional[float] = None,
) -> tuple[str, float, Optional[float]]:
    """Resolve either direct horizon adjustment alpha_h or per-step rho.

    Exactly one of ``alpha_h`` and ``rho`` must be provided.
    """
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    if (alpha_h is None) == (rho is None):
        raise ValueError(
            "Provide exactly one of alpha_h or rho. "
            "alpha_h is the direct h-step movement fraction; rho is the "
            "per-step disequilibrium persistence."
        )

    if alpha_h is not None:
        alpha = _validate_scalar_unit_interval(alpha_h, "alpha_h")
        return DIRECT_ALPHA, alpha, None

    rho_value = _validate_scalar_unit_interval(rho, "rho")
    alpha = float(1.0 - rho_value ** horizon)
    # Guard floating-point roundoff at exact boundaries.
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return PERSISTENCE_RHO, alpha, rho_value


def adjust_state_toward_equilibrium(
    current_values: np.ndarray,
    equilibrium: EquilibriumResult,
    *,
    horizon: int,
    alpha_h: Optional[float] = None,
    rho: Optional[float] = None,
    equilibrium_tolerance: float = 0.01,
    epsilon: float = 1e-8,
) -> EquilibriumAdjustmentResult:
    """Move the current state partially toward a predicted Rel-ESE equilibrium.

    Parameters
    ----------
    current_values:
        Last observed system vector y_t.  It is transformed into the same
        centered latent coordinate used by the selected relation geometry.

    equilibrium:
        Predicted ES from ``equilibrium_solver_v3.solve_equilibrium_state``.
        This object is not modified and remains the equilibrium target.

    horizon:
        Forecast horizon h.  Forecasting always stops at h; it does not stop
        when a numerical convergence criterion is met.

    alpha_h:
        Direct fraction of disequilibrium removed by horizon h, in [0,1].
        alpha_h=1 reproduces the current full-equilibrium Rel-ESE forecast.

    rho:
        Alternative per-step disequilibrium persistence in [0,1].  When used,
        alpha_h = 1-rho**h.

    equilibrium_tolerance:
        Diagnostic threshold for normalized remaining disequilibrium.  This
        does NOT control the forecast horizon or trigger early stopping.
    """
    if not isinstance(equilibrium, EquilibriumResult):
        raise TypeError("equilibrium must be an equilibrium_solver_v4.EquilibriumResult.")

    representation = normalize_representation_name(equilibrium.representation)
    horizon = int(horizon)
    mode, resolved_alpha, resolved_rho = resolve_horizon_adjustment(
        horizon=horizon,
        alpha_h=alpha_h,
        rho=rho,
    )

    equilibrium_tolerance = float(equilibrium_tolerance)
    if not np.isfinite(equilibrium_tolerance) or equilibrium_tolerance < 0:
        raise ValueError("equilibrium_tolerance must be a finite non-negative value.")

    current_values = np.asarray(current_values, dtype=float)
    if current_values.ndim != 1:
        raise ValueError("current_values must be a one-dimensional system vector.")
    if len(current_values) != equilibrium.n_systems:
        raise ValueError(
            "current_values length must match equilibrium.n_systems; "
            f"got {len(current_values)} and {equilibrium.n_systems}."
        )
    if not np.all(np.isfinite(current_values)):
        raise ValueError("current_values must contain only finite values.")

    current_latent = centered_latent_state(
        current_values,
        representation=representation,
        epsilon=epsilon,
    )
    equilibrium_latent = np.asarray(equilibrium.u, dtype=float).copy()

    if current_latent.shape != equilibrium_latent.shape:
        raise RuntimeError("Current latent state and equilibrium latent state differ in shape.")

    adjusted_latent = (
        (1.0 - resolved_alpha) * current_latent
        + resolved_alpha * equilibrium_latent
    )
    # Both endpoints use a zero-sum gauge.  Recenter defensively for numerical
    # stability and compatibility with representation-aware reconstruction.
    adjusted_latent = np.asarray(
        adjusted_latent - float(np.mean(adjusted_latent)),
        dtype=float,
    )

    initial_gap = current_latent - equilibrium_latent
    remaining_gap = adjusted_latent - equilibrium_latent

    initial_norm = float(np.linalg.norm(initial_gap))
    remaining_norm = float(np.linalg.norm(remaining_gap))

    if initial_norm <= 1e-15:
        remaining_ratio = 0.0
    else:
        remaining_ratio = float(remaining_norm / initial_norm)

    practically_at_equilibrium = bool(
        remaining_ratio <= equilibrium_tolerance
    )

    return EquilibriumAdjustmentResult(
        representation=representation,
        mode=mode,
        horizon=horizon,
        alpha_h=resolved_alpha,
        rho=resolved_rho,
        current_values=current_values.copy(),
        current_latent=np.asarray(current_latent, dtype=float),
        equilibrium_latent=equilibrium_latent,
        adjusted_latent=adjusted_latent,
        initial_disequilibrium_norm=initial_norm,
        remaining_disequilibrium_norm=remaining_norm,
        remaining_disequilibrium_ratio=remaining_ratio,
        equilibrium_tolerance=equilibrium_tolerance,
        practically_at_equilibrium=practically_at_equilibrium,
    )


# Short alias for experiment scripts.
def adjust_state(
    current_values: np.ndarray,
    equilibrium: EquilibriumResult,
    **kwargs: Any,
) -> EquilibriumAdjustmentResult:
    return adjust_state_toward_equilibrium(current_values, equilibrium, **kwargs)


if __name__ == "__main__":
    # Small positive-data illustration.  This deliberately uses a hand-built
    # relation graph only to demonstrate the adjustment algebra.
    from relation_matrix import PairwiseRelationEdge, build_relation_matrix
    from equilibrium_solver_v4 import solve_equilibrium_state

    current = np.array([50.0, 30.0, 20.0])
    target_share = np.array([0.30, 0.35, 0.35])
    target_u = np.log(target_share) - np.mean(np.log(target_share))

    edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            edges.append(
                PairwiseRelationEdge(
                    i=i,
                    j=j,
                    r=float(target_u[i] - target_u[j]),
                    weight=1.0,
                    is_valid=True,
                )
            )

    state = build_relation_matrix(
        3,
        edges,
        system_names=["A", "B", "C"],
        require_complete_input=True,
    )
    es = solve_equilibrium_state(state, representation=LOG_RATIO)

    out = adjust_state_toward_equilibrium(
        current,
        es,
        horizon=5,
        rho=0.8,
    )

    print("ES gamma       :", np.round(es.gamma, 6))
    print("current gamma  :", np.round(current / current.sum(), 6))
    print("adjusted gamma :", np.round(out.gamma, 6))
    print("alpha_h        :", out.alpha_h)
    print("remaining ratio:", out.remaining_disequilibrium_ratio)
