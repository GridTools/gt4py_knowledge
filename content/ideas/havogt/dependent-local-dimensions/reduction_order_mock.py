# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Reference semantics for chained-connectivity reductions (numpy mock).

Verifies the reduction-order claim of
``docs/development/investigations/dependent-local-dimensions.md`` §2 on the
padded-with-skip-values representation of ``edge_field(V2E)(E2V)``:

- both reduction orders agree *iff* the gathered validity mask
  ``v2e_valid[e2v[e, j]]`` is applied — a mask that depends on the *value*
  ``e2v[e, j]``, not on any dimension of the field;
- reducing the outer (parent) hop first without that dependency information
  is simply wrong (the intermediate has a dangling local dimension).
"""

from __future__ import annotations

import numpy as np


SKIP = -1

# small irregular mesh: 4 vertices, 5 edges; vertices have 2-3 incident edges
E2V_TABLE = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [3, 1]])
V2E_TABLE = np.array([[0, 2, 3], [0, 1, 4], [1, 2, SKIP], [3, 4, SKIP]])


def two_hop_block(ef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Padded block ``t[e, j, i] = ef[V2E[E2V[e, j], i]]`` and its gathered validity."""
    block = ef[V2E_TABLE[E2V_TABLE]]  # contains garbage where invalid (index -1 wraps)
    valid = V2E_TABLE[E2V_TABLE] != SKIP
    return block, valid


def sum_innermost_first(ef: np.ndarray) -> np.ndarray:
    """LIFO order: reduce V2E (innermost hop), then E2V."""
    block, valid = two_hop_block(ef)
    return np.where(valid, block, 0.0).sum(axis=2).sum(axis=1)


def sum_outer_first_gathered(ef: np.ndarray) -> np.ndarray:
    """Outer-first, but with the (materialized) gathered validity mask."""
    block, valid = two_hop_block(ef)
    return np.where(valid, block, 0.0).sum(axis=1).sum(axis=1)


def sum_outer_first_naive(ef: np.ndarray) -> np.ndarray:
    """Outer-first without dependency information: the dangling-V2E intermediate."""
    block, _ = two_hop_block(ef)
    return block.sum(axis=1).sum(axis=1)
