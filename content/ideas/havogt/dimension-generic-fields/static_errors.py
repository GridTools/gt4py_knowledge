# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Cases that must be *rejected* by mypy.

Each line carrying an EXPECT-ERROR marker comment must produce at least one
mypy error on exactly that line; any error on an unmarked line fails the test
(see `test_typed_dimensions.py`).
"""

from __future__ import annotations

from typed_dimensions import Dimension, DimensionKind, Dims, Field, Staggered, dual_operator


class I(Dimension): ...  # noqa: E742


class J(Dimension): ...


class K(Dimension, kind=DimensionKind.VERTICAL): ...


def staggered_is_not_unstaggered(a: Field[Dims[I], float]) -> None:
    b: Field[Dims[I], float] = a(I + 1 / 2)  # EXPECT-ERROR: result lives on Staggered[I]
    c: Field[Dims[Staggered[I]], float] = a(I + 1)  # EXPECT-ERROR: result lives on I
    del b, c


def shift_along_missing_dimension(a: Field[Dims[I], float]) -> None:
    a(J + 1)  # EXPECT-ERROR: field has no J dimension


def mixing_dual_grids(a: Field[Dims[I], float], b: Field[Dims[Staggered[I]], float]) -> None:
    a + b  # EXPECT-ERROR: fields live on dual grids


def wrong_dimension_kind(k: Field[Dims[K], float]) -> None:
    def vertical_only(f: Field[Dims[K], float]) -> Field[Dims[K], float]:
        return f

    def horizontal_only(f: Field[Dims[I], float]) -> Field[Dims[I], float]:
        return f

    vertical_only(k)
    horizontal_only(k)  # EXPECT-ERROR: K is not I


def roundtrip_must_go_through_the_dual(a: Field[Dims[I], float]) -> None:
    b = a(I + 1 / 2)
    b(I + 1 / 2)  # EXPECT-ERROR: b has no unstaggered I dimension anymore


def doubly_staggered_dimensions_are_unrepresentable(
    dd: Field[Dims[Staggered[Staggered[I]]], float],  # EXPECT-ERROR: Staggered requires a Dimension
) -> None: ...


@dual_operator  # EXPECT-ERROR: signature does not map X to Dual[X]
def not_dual_generic(f: Field[Dims[I], float]) -> Field[Dims[I], float]:
    return f


def weight_must_live_on_the_dual_grid(
    a: Field[Dims[I], float], same_grid_weight: Field[Dims[I], float]
) -> None:
    from static_checks import weighted_avg

    weighted_avg(a, same_grid_weight)  # EXPECT-ERROR: weight is on X, not Dual[X]
