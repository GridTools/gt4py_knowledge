# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Static (mypy-verified) usage of the typed-dimensions prototype.

Every statement in this file must type-check with a plain mypy (no GT4Py
plugin); `assert_type` pins down the *exact* inferred types. The companion
file `static_errors.py` contains the cases that must *fail* to type-check.

This file is checked by `test_typed_dimensions.py`; it is never executed.
"""

from __future__ import annotations

from typing import Any, TypeAlias, TypeVar, overload

from typed_dimensions import (
    Connectivity,
    Dimension,
    DimensionBase,
    DimensionKind,
    Dims,
    Dual,
    Field,
    Staggered,
    dual_operator,
)
from typing_extensions import TypeVarTuple, Unpack, assert_type


class I(Dimension): ...  # noqa: E742  # single-letter dimension names are the domain convention


class J(Dimension): ...


class K(Dimension, kind=DimensionKind.VERTICAL): ...


# A staggered dimension needs no separate declaration: `Staggered[I]` is both
# the type and (via materialization) the runtime dimension object. A type
# alias gives it a domain name.
Ihalf: TypeAlias = Staggered[I]


# --- Connectivity typing: integral vs half-integral offsets ----------------


def connectivity_types() -> None:
    assert_type(I + 1, Connectivity[I, I])
    assert_type(I - 2, Connectivity[I, I])
    # half-integral offsets move to the dual grid ...
    assert_type(I + 1 / 2, Connectivity[Staggered[I], I])
    assert_type(I - 1 / 2, Connectivity[Staggered[I], I])
    # ... and back (involution: never `Staggered[Staggered[I]]`)
    assert_type(Staggered[I] + 1 / 2, Connectivity[I, Staggered[I]])
    assert_type(Staggered[I] - 3 / 2, Connectivity[I, Staggered[I]])
    # integer offsets on a staggered dimension stay on the staggered grid
    assert_type(Staggered[I] + 1, Connectivity[Staggered[I], Staggered[I]])


# --- The motivating example (staggering round trip) -------------------------


def staggering_roundtrip(a: Field[Dims[I], float]) -> None:
    b = a(I + 1 / 2)
    assert_type(b, Field[Dims[Staggered[I]], float])
    b_via_alias: Field[Dims[Ihalf], float] = b  # the alias spelling is the same type

    c = b(Staggered[I] + 1 / 2)
    assert_type(c, Field[Dims[I], float])

    # ordinary shifts do not change the dimensions
    d = a(I + 1)
    assert_type(d, Field[Dims[I], float])

    del b_via_alias, c, d


# --- Positional substitution on higher-rank fields ---------------------------


def rank2_substitution(uv: Field[Dims[I, J], float]) -> None:
    assert_type(uv(I + 1), Field[Dims[I, J], float])
    assert_type(uv(J + 1 / 2), Field[Dims[I, Staggered[J]], float])
    assert_type(uv(I - 1 / 2), Field[Dims[Staggered[I], J], float])


def rank3_substitution(w: Field[Dims[I, J, K], float]) -> None:
    assert_type(w(K + 1 / 2), Field[Dims[I, J, Staggered[K]], float])
    assert_type(w(J + 2), Field[Dims[I, J, K], float])
    # chaining: stagger in I, then in K
    assert_type(
        w(I + 1 / 2)(K - 1 / 2),
        Field[Dims[Staggered[I], J, Staggered[K]], float],
    )


# --- A C-grid style finite difference ----------------------------------------


def gradient_to_staggered(p: Field[Dims[I, J], float]) -> Field[Dims[Staggered[I], J], float]:
    """Pressure gradient at the staggered (velocity) points."""
    return p(I + 1 / 2) - p(I - 1 / 2)


# --- Dimension-generic operators ---------------------------------------------

DimT = TypeVar("DimT", bound=Dimension)
Ds = TypeVarTuple("Ds")
DTypeT = TypeVar("DTypeT")


def double(f: Field[Dims[Unpack[Ds]], DTypeT]) -> Field[Dims[Unpack[Ds]], DTypeT]:
    """Generic in rank, dimensions and dtype."""
    return f * f


def to_staggered(
    f: Field[Dims[DimT], float], dim: type[DimT]
) -> Field[Dims[Staggered[DimT]], float]:
    """Staggering an operator that is *generic* in the dimension it staggers."""
    return f(dim + 1 / 2)


def use_generic_operators(
    a: Field[Dims[I], float], uv: Field[Dims[I, J], float], k: Field[Dims[K], float]
) -> None:
    assert_type(double(a), Field[Dims[I], float])
    assert_type(double(uv), Field[Dims[I, J], float])
    assert_type(to_staggered(a, I), Field[Dims[Staggered[I]], float])
    assert_type(to_staggered(k, K), Field[Dims[Staggered[K]], float])


# --- Dual-generic operators ---------------------------------------------------
#
# `avg` maps a field to its *dual* grid: I -> Staggered[I], but also
# Staggered[I] -> I. Python typing has no type-level function `Dual[X]`, but
# the same overload-pair trick that types `DimensionMeta.__add__` expresses
# it. Two equivalent spellings:

# (1) explicit: the user writes the overload pair (one body, two signature lines)


@overload
def avg_explicit(f: Field[Dims[Staggered[DimT]], float]) -> Field[Dims[DimT], float]: ...
@overload
def avg_explicit(f: Field[Dims[DimT], float]) -> Field[Dims[Staggered[DimT]], float]: ...
def avg_explicit(f: Field[Any, float]) -> Field[Any, float]:
    # The body is dual-generic, so the dimension is only known as "the field's
    # dimension"; recover it at runtime (`Any`: precision lives in the overloads).
    dim: Any = f.dims[0]
    return f(dim - 1 / 2) + f(dim + 1 / 2)


# (2) declarative: ONE natural signature with the `Dual[X]` marker; the overload
#     pair lives in the library (`DualUnaryOperator`), attached by the decorator.
#     In gt4py proper this would be folded into `@field_operator` itself.

AnyDimT = TypeVar("AnyDimT", bound=DimensionBase)  # ranges over staggered dims too


@dual_operator
def avg(f: Field[Dims[AnyDimT], float]) -> Field[Dims[Dual[AnyDimT]], float]:
    dim: Any = f.dims[0]
    return f(dim - 1 / 2) + f(dim + 1 / 2)


# `Dual[X]` also works in *parameter* positions, e.g. a weight that already
# lives on the target (dual) grid; the decorator picks the matching shape.


@dual_operator
def weighted_avg(
    a: Field[Dims[AnyDimT], float], weight: Field[Dims[Dual[AnyDimT]], float]
) -> Field[Dims[Dual[AnyDimT]], float]:
    dim: Any = a.dims[0]
    return weight * (a(dim + 1 / 2) + a(dim - 1 / 2))


def use_avg(a: Field[Dims[I], float], b: Field[Dims[Staggered[I]], float]) -> None:
    assert_type(avg(a), Field[Dims[Staggered[I]], float])  # to the dual grid ...
    assert_type(avg(b), Field[Dims[I], float])  # ... and from it
    assert_type(avg(avg(a)), Field[Dims[I], float])  # round trip
    assert_type(avg(avg(avg(a))), Field[Dims[Staggered[I]], float])
    # the explicit spelling types identically
    assert_type(avg_explicit(a), Field[Dims[Staggered[I]], float])
    assert_type(avg_explicit(b), Field[Dims[I], float])
    assert_type(avg_explicit(avg_explicit(a)), Field[Dims[I], float])
    # Dual[X] in parameter position: the weight lives on the result grid
    assert_type(weighted_avg(a, b), Field[Dims[Staggered[I]], float])
    assert_type(weighted_avg(b, a), Field[Dims[I], float])


# --- Vertical-only operators (kind-restricted by nominal typing) -------------


def vertical_only(f: Field[Dims[K], float]) -> Field[Dims[K], float]:
    return f


def use_vertical(k: Field[Dims[K], float]) -> None:
    assert_type(vertical_only(k), Field[Dims[K], float])
