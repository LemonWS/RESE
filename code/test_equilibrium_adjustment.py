"""Regression tests for equilibrium_adjustment_v1.py.

Run:
    python test_equilibrium_adjustment_v1.py
"""
from __future__ import annotations

import numpy as np

from relation_matrix import PairwiseRelationEdge, build_relation_matrix
from equilibrium_solver import solve_equilibrium_state
from equilibrium_adjustment import adjust_state_toward_equilibrium


def _positive_example():
    current = np.array([50.0, 30.0, 20.0], dtype=float)
    target_share = np.array([0.30, 0.35, 0.35], dtype=float)
    u = np.log(target_share) - np.mean(np.log(target_share))
    edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            edges.append(PairwiseRelationEdge(i, j, float(u[i] - u[j]), 1.0, True))
    state = build_relation_matrix(
        3, edges, system_names=["A", "B", "C"], require_complete_input=True
    )
    es = solve_equilibrium_state(state, representation="log_ratio")
    return current, es


def test_alpha_one_matches_full_es():
    current, es = _positive_example()
    out = adjust_state_toward_equilibrium(current, es, horizon=5, alpha_h=1.0)
    np.testing.assert_allclose(out.adjusted_latent, es.u, atol=1e-12)
    np.testing.assert_allclose(out.gamma, es.gamma, atol=1e-12)
    assert out.remaining_disequilibrium_ratio == 0.0


def test_alpha_zero_keeps_current_state():
    current, es = _positive_example()
    out = adjust_state_toward_equilibrium(current, es, horizon=5, alpha_h=0.0)
    np.testing.assert_allclose(out.gamma, current / current.sum(), atol=1e-10)
    assert abs(out.remaining_disequilibrium_ratio - 1.0) < 1e-12


def test_rho_horizon_formula():
    current, es = _positive_example()
    rho = 0.8
    h = 3
    out = adjust_state_toward_equilibrium(current, es, horizon=h, rho=rho)
    expected_alpha = 1.0 - rho**h
    assert abs(out.alpha_h - expected_alpha) < 1e-12
    assert abs(out.remaining_disequilibrium_ratio - rho**h) < 1e-12


def test_intermediate_state_lies_between_current_and_es_latent():
    current, es = _positive_example()
    alpha = 0.4
    out = adjust_state_toward_equilibrium(current, es, horizon=1, alpha_h=alpha)
    expected = (1.0 - alpha) * out.current_latent + alpha * es.u
    expected -= expected.mean()
    np.testing.assert_allclose(out.adjusted_latent, expected, atol=1e-12)


def main():
    tests = [
        test_alpha_one_matches_full_es,
        test_alpha_zero_keeps_current_state,
        test_rho_horizon_formula,
        test_intermediate_state_lies_between_current_and_es_latent,
    ]
    print("=" * 72)
    print("Equilibrium adjustment regression tests")
    print("=" * 72)
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    print("-" * 72)
    print(f"Passed {len(tests)}/{len(tests)} tests.")
    print("=" * 72)


if __name__ == "__main__":
    main()
