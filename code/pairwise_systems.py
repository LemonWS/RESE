"""
pairwise_systems.py

Construct unique pairwise combinations of systems from a
multi-system time-series matrix.

Input
-----
Y : np.ndarray, shape (T, N)

    T = number of time observations
    N = number of systems

Columns correspond to systems.

Output
------
Unique unordered system pairs

    (i, j), i < j

The number of pairs is

    N(N - 1) / 2

This module ONLY constructs system pairs.
It does not estimate pairwise relationships.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, List, Optional, Tuple

import numpy as np


# ============================================================
# 1. Pair data structure
# ============================================================

@dataclass(frozen=True)
class SystemPair:
    """
    Container for one unique pair of systems.

    Attributes
    ----------
    i : int
        Column index of the first system.

    j : int
        Column index of the second system.

    system_i : str
        Name of the first system.

    system_j : str
        Name of the second system.

    y_i : np.ndarray
        Historical observations of system i.
        Shape: (T,)

    y_j : np.ndarray
        Historical observations of system j.
        Shape: (T,)
    """

    i: int
    j: int

    system_i: str
    system_j: str

    y_i: np.ndarray
    y_j: np.ndarray


# ============================================================
# 2. Validate input
# ============================================================

def _validate_input(
    Y: np.ndarray,
    system_names: Optional[List[str]] = None
) -> Tuple[int, int]:
    """
    Validate the multi-system input matrix.

    Parameters
    ----------
    Y : np.ndarray
        Target matrix with shape (T, N).

    system_names : list[str], optional
        Names of the N systems.

    Returns
    -------
    T : int
        Number of observations.

    N : int
        Number of systems.
    """

    if not isinstance(Y, np.ndarray):
        raise TypeError(
            "Y must be a NumPy array."
        )

    if Y.ndim != 2:
        raise ValueError(
            f"Y must be a 2-dimensional array with shape (T, N). "
            f"Received shape: {Y.shape}"
        )

    T, N = Y.shape

    if T < 1:
        raise ValueError(
            "Y contains no time observations."
        )

    if N < 2:
        raise ValueError(
            "At least two systems are required "
            "to construct pairwise relationships."
        )

    if system_names is not None:

        if len(system_names) != N:
            raise ValueError(
                f"Number of system names ({len(system_names)}) "
                f"does not match number of systems ({N})."
            )

        if len(set(system_names)) != N:
            raise ValueError(
                "System names must be unique."
            )

    return T, N


# ============================================================
# 3. Number of pairwise combinations
# ============================================================

def number_of_pairs(N: int) -> int:
    """
    Calculate the number of unique unordered pairs.

    Parameters
    ----------
    N : int
        Number of systems.

    Returns
    -------
    int
        Number of unique system pairs.

    Notes
    -----
    The number is

        N(N - 1) / 2.
    """

    if N < 2:
        return 0

    return N * (N - 1) // 2


# ============================================================
# 4. Generate pair indices
# ============================================================

def generate_pair_indices(
    N: int
) -> List[Tuple[int, int]]:
    """
    Generate all unique system index pairs.

    Only pairs satisfying

        i < j

    are generated.

    Parameters
    ----------
    N : int
        Number of systems.

    Returns
    -------
    list[tuple[int, int]]
        List of pair indices.

    Example
    -------
    For N = 4:

        [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3)
        ]
    """

    if N < 2:
        return []

    return list(combinations(range(N), 2))


# ============================================================
# 5. Pair generator
# ============================================================

def iter_system_pairs(
    Y: np.ndarray,
    system_names: Optional[List[str]] = None
) -> Iterator[SystemPair]:
    """
    Iterate over all unique system pairs.

    This function is memory-efficient because pairs are
    produced one at a time instead of storing all pair
    histories in a large list.

    Parameters
    ----------
    Y : np.ndarray
        Multi-system target matrix.
        Shape: (T, N).

    system_names : list[str], optional
        Names of the systems.

        If None, default names are generated:

            S1, S2, ..., SN

    Yields
    ------
    SystemPair
        One unique pair of systems.
    """

    T, N = _validate_input(
        Y=Y,
        system_names=system_names
    )

    if system_names is None:
        system_names = [
            f"S{i + 1}"
            for i in range(N)
        ]

    for i, j in combinations(range(N), 2):

        yield SystemPair(
            i=i,
            j=j,
            system_i=system_names[i],
            system_j=system_names[j],
            y_i=Y[:, i],
            y_j=Y[:, j]
        )


# ============================================================
# 6. Return all pairs as a list
# ============================================================

def create_system_pairs(
    Y: np.ndarray,
    system_names: Optional[List[str]] = None
) -> List[SystemPair]:
    """
    Construct all unique system pairs.

    Parameters
    ----------
    Y : np.ndarray
        Multi-system data.
        Shape: (T, N).

    system_names : list[str], optional
        Names of systems.

    Returns
    -------
    list[SystemPair]
        All unique system pairs.
    """

    pairs = list(
        iter_system_pairs(
            Y=Y,
            system_names=system_names
        )
    )

    _, N = Y.shape

    expected = number_of_pairs(N)

    if len(pairs) != expected:
        raise RuntimeError(
            f"Unexpected number of pairs. "
            f"Expected {expected}, obtained {len(pairs)}."
        )

    return pairs


# ============================================================
# 7. Display pair structure
# ============================================================

def print_pair_summary(
    Y: np.ndarray,
    system_names: Optional[List[str]] = None
) -> None:
    """
    Print the pairwise structure without estimating relations.
    """

    T, N = _validate_input(
        Y=Y,
        system_names=system_names
    )

    if system_names is None:
        system_names = [
            f"S{i + 1}"
            for i in range(N)
        ]

    K = number_of_pairs(N)

    print("=" * 60)
    print("Pairwise System Structure")
    print("=" * 60)

    print(f"Observations : {T}")
    print(f"Systems      : {N}")
    print(f"Pairs        : {K}")

    print("-" * 60)

    for pair_id, (i, j) in enumerate(
        combinations(range(N), 2),
        start=1
    ):
        print(
            f"Pair {pair_id:>3}: "
            f"{system_names[i]} "
            f"<-> "
            f"{system_names[j]} "
            f"(columns {i}, {j})"
        )

    print("=" * 60)


# ============================================================
# 8. Example
# ============================================================

if __name__ == "__main__":

    # Example:
    # 5 observations
    # 4 systems

    Y = np.array(
        [
            [10, 20, 30, 40],
            [11, 21, 29, 42],
            [13, 22, 31, 45],
            [14, 25, 33, 47],
            [16, 27, 35, 50],
        ],
        dtype=float
    )

    system_names = [
        "S1",
        "S2",
        "S3",
        "S4"
    ]

    print_pair_summary(
        Y,
        system_names
    )

    pairs = create_system_pairs(
        Y,
        system_names
    )

    print("\nFirst pair:")
    print(pairs[0])

    print("\nHistory of first system:")
    print(pairs[0].y_i)

    print("\nHistory of second system:")
    print(pairs[0].y_j)