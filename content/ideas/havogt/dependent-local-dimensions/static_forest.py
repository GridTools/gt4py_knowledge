# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Static (mypy-verified) encoding of "dims as a dependency forest" (proposal B).

Companion to ``docs/development/investigations/dependent-local-dimensions.md``
§5: dimension *order carries no semantics*; instead every local dimension
references its parent in its own type (``Local[V2E, Vertex]``,
``Local[V2E, Local[E2V, Edge]]`` — the user-facing ``V2EDim[Vertex]``
notation). Remap *reparents* the existing hops onto the new hop; reductions
are legal for any *leaf* of the dependency forest (so independent sibling
hops commute), and reducing a non-leaf is rejected.

The dims tuple below is only a canonical *normal form* of the set (parents
before children); Python's static type system has no unordered type
collections, so order-free semantics can only be checked against a canonical
spelling. Checked by ``test_dependent_local_dims.py``; never executed.
"""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar, overload

from typed_dimensions import Dimension, DimensionBase, Dims
from typing_extensions import assert_type


DT = TypeVar("DT")
O = TypeVar("O", bound=DimensionBase)  # noqa: E741  # origin of a connectivity
C = TypeVar("C", bound=DimensionBase)  # codomain of a connectivity
R = TypeVar("R", bound=DimensionBase)  # a root (non-local) dimension
P = TypeVar("P", bound=DimensionBase)  # parent of a local dimension
ConnT = TypeVar("ConnT", bound="NeighborTable")
C1 = TypeVar("C1", bound="NeighborTable")
C2 = TypeVar("C2", bound="NeighborTable")
DimsT = TypeVar("DimsT", bound=Dims, covariant=True)


class NeighborTable(Generic[O, C]):
    """Typed neighbor connectivity: origin ``O``, codomain ``C``."""


class Local(DimensionBase, Generic[ConnT, P]):
    """Local dimension induced by connectivity ``ConnT``, depending on parent ``P``.

    The parent is part of the *type*: the dependency forest is encoded by
    reference, not by position, so the dims collection is a set.
    """


class Vertex(Dimension): ...


class Edge(Dimension): ...


class V2E(NeighborTable[Vertex, Edge]): ...


class E2V(NeighborTable[Edge, Vertex]): ...


class V2V(NeighborTable[Vertex, Vertex]): ...


# convenient aliases for the assertions below
L_E2V = Local[NeighborTable[Edge, Vertex], Edge]
L_V2E_of_E2V = Local[NeighborTable[Vertex, Edge], Local[NeighborTable[Edge, Vertex], Edge]]
L_V2E = Local[NeighborTable[Vertex, Edge], Vertex]
L_V2V = Local[NeighborTable[Vertex, Vertex], Vertex]


class Field(Protocol[DimsT, DT]):
    # remap of a hop-free field: new root, hop hangs off it
    @overload
    def __call__(
        self: Field[Dims[C], DT], conn: type[NeighborTable[O, C]]
    ) -> Field[Dims[O, Local[NeighborTable[O, C], O]], DT]: ...
    # remap of a single-hop field: the existing hop is REPARENTED onto the new
    # hop (one overload per supported forest shape; bounded codegen)
    @overload
    def __call__(
        self: Field[Dims[C, Local[C1, C]], DT], conn: type[NeighborTable[O, C]]
    ) -> Field[
        Dims[O, Local[NeighborTable[O, C], O], Local[C1, Local[NeighborTable[O, C], O]]],
        DT,
    ]: ...
    def __call__(self, conn: type[NeighborTable]) -> Field[Dims, DT]: ...


# leaf-only reductions, one overload per supported forest shape:
@overload
def neighbor_sum(
    f: Field[Dims[R, Local[C1, R]], DT], axis: type[Local[C1, R]]
) -> Field[Dims[R], DT]: ...
@overload  # chain of two: only the child is a leaf
def neighbor_sum(
    f: Field[Dims[R, Local[C1, R], Local[C2, Local[C1, R]]], DT],
    axis: type[Local[C2, Local[C1, R]]],
) -> Field[Dims[R, Local[C1, R]], DT]: ...
@overload  # two siblings: both are leaves, either reduction order is legal
def neighbor_sum(
    f: Field[Dims[R, Local[C1, R], Local[C2, R]], DT], axis: type[Local[C1, R]]
) -> Field[Dims[R, Local[C2, R]], DT]: ...
@overload
def neighbor_sum(
    f: Field[Dims[R, Local[C1, R], Local[C2, R]], DT], axis: type[Local[C2, R]]
) -> Field[Dims[R, Local[C1, R]], DT]: ...
def neighbor_sum(f: Field[Dims, DT], axis: type[Local]) -> Field[Dims, DT]:
    raise NotImplementedError


def chain_with_reparenting(ef: Field[Dims[Edge], float]) -> None:
    vf = ef(V2E)
    assert_type(vf, Field[Dims[Vertex, L_V2E], float])

    two = vf(E2V)  # the V2E hop now hangs off the E2V hop: V2EDim[E2VDim[Edge]]
    assert_type(two, Field[Dims[Edge, L_E2V, L_V2E_of_E2V], float])

    # leaf first (the V2E hop), then its parent (the E2V hop)
    once = neighbor_sum(two, L_V2E_of_E2V)
    assert_type(once, Field[Dims[Edge, L_E2V], float])
    assert_type(neighbor_sum(once, L_E2V), Field[Dims[Edge], float])


def non_leaf_reduction_rejected(two: Field[Dims[Edge, L_E2V, L_V2E_of_E2V], float]) -> None:
    # reducing the middle of the chain (the V2E hop still depends on it) must
    # not type-check; `warn_unused_ignores` pins the rejection
    neighbor_sum(two, L_E2V)  # type: ignore[arg-type]


def independent_siblings_commute(prod: Field[Dims[Vertex, L_V2E, L_V2V], float]) -> None:
    # two hops off the same root (e.g. edges x vertex-neighbors of a vertex):
    # both are leaves, so both reduction orders are well-typed — the order-free
    # payoff that the path proposal (A) cannot represent at all
    a = neighbor_sum(prod, L_V2E)
    assert_type(a, Field[Dims[Vertex, L_V2V], float])
    b = neighbor_sum(prod, L_V2V)
    assert_type(b, Field[Dims[Vertex, L_V2E], float])
