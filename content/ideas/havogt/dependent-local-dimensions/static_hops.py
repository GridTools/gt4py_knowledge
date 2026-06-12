# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Static (mypy-verified) encoding of "dims as a topology path".

Companion to ``docs/development/investigations/dependent-local-dimensions.md``
§4.4, building on the typed-dimensions prototype: a local dimension is the
typed application of a neighbor connectivity (``Local[C]``); remap pushes a
hop onto the field's hop stack, ``neighbor_sum`` pops the innermost hop
(LIFO), and chain violations are rejected at the call site.

Checked by ``test_dependent_local_dims.py`` (mypy with ``MYPYPATH`` pointing
at the typed-dimensions prototype); never executed.
"""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar

from typed_dimensions import Dimension, DimensionBase, Dims
from typing_extensions import TypeVarTuple, Unpack, assert_type


DT = TypeVar("DT")
O = TypeVar("O", bound=DimensionBase)  # noqa: E741  # origin of a connectivity
C = TypeVar("C", bound=DimensionBase)  # codomain of a connectivity
ConnT = TypeVar("ConnT", bound="NeighborTable")
Hops = TypeVarTuple("Hops")
DimsT = TypeVar("DimsT", bound=Dims, covariant=True)


class NeighborTable(Generic[O, C]):
    """Typed neighbor connectivity: origin ``O`` (where the remapped field
    lives), codomain ``C`` (what the entries point to)."""


class Local(DimensionBase, Generic[ConnT]):
    """The local dimension induced by a neighbor connectivity (a 'hop')."""


class Vertex(Dimension): ...


class Edge(Dimension): ...


class V2E(NeighborTable[Vertex, Edge]): ...


class E2V(NeighborTable[Edge, Vertex]): ...


class Field(Protocol[DimsT, DT]):
    # remap: requires the field's origin to be the connectivity's codomain,
    # replaces it by the connectivity's origin and pushes the hop
    def __call__(
        self: Field[Dims[C, Unpack[Hops]], DT],
        conn: type[NeighborTable[O, C]],
    ) -> Field[Dims[O, Local[NeighborTable[O, C]], Unpack[Hops]], DT]: ...


def neighbor_sum(f: Field[Dims[Unpack[Hops], Local[ConnT]], DT]) -> Field[Dims[Unpack[Hops]], DT]:
    # `Unpack` may only precede a *fixed* tail element, so "reduce only the
    # innermost hop" (LIFO) is enforced by the type system's own restriction.
    raise NotImplementedError


def chain_and_lifo_reduction(ef: Field[Dims[Edge], float]) -> None:
    vf = ef(V2E)
    assert_type(vf, Field[Dims[Vertex, Local[NeighborTable[Vertex, Edge]]], float])

    two_hop = vf(E2V)  # Edge -E2V-> Vertex -V2E-> Edge
    assert_type(
        two_hop,
        Field[
            Dims[
                Edge,
                Local[NeighborTable[Edge, Vertex]],
                Local[NeighborTable[Vertex, Edge]],
            ],
            float,
        ],
    )

    # LIFO: pop the innermost hop (V2E) first, then E2V
    once = neighbor_sum(two_hop)
    assert_type(once, Field[Dims[Edge, Local[NeighborTable[Edge, Vertex]]], float])
    assert_type(neighbor_sum(once), Field[Dims[Edge], float])


def chain_violation(vf: Field[Dims[Vertex, Local[NeighborTable[Vertex, Edge]]], float]) -> None:
    # V2E's codomain is Edge but the field's origin is Vertex: rejected.
    # (`warn_unused_ignores` turns this ignore into an error if mypy ever
    # stops flagging the call, pinning the rejection.)
    vf(V2E)  # type: ignore[arg-type]
