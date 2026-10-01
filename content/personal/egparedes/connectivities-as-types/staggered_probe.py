"""Typing probe for the Cartesian-axis levels and the `Staggered[D]` bound.

Run with `mypy --strict --python-version 3.12` and `pyright --pythonversion 3.12`.

Expected: every line marked `EXPECT-ERROR` reported (the `Staggered[...]`
subscriptions as `[type-var]`, "Type argument ... must be a subtype of
CartesianAxisIndex"), and nothing else. The `must be ACCEPTED` lines are the
point of placing the two axis levels *below* `DimensionIndex` rather than above
it: a staggered dimension stays a `DimensionIndex`, so no `type[DimensionIndex]`
annotation in the gt4py tree widens, and `LocalDimensionIndex` keeps its
position.

Only the *static* declarations are modelled. At runtime `Staggered` is a
metaclass subscription building one interned class per base, and
`LocalDimensionIndex` is a real subclass of `DimensionIndex` so that eve's
`type[...]` validation is unaffected — see the main note.

Last run: mypy 2.3.1 and pyright 1.1.414 (2026-10-01). Both report exactly the
nine `EXPECT-ERROR` lines and nothing else (mypy emits two diagnostics for the
`Doubly` alias, pyright one), so every `must be ACCEPTED` line holds.
"""

from __future__ import annotations

from typing import ClassVar, TypeAlias

# ---------------- the hierarchy ----------------


class DimensionIndex:
    """The root. Mesh locations and geometry-less index spaces are direct subclasses."""

    value: int


class AnyCartesianAxisIndex(DimensionIndex):
    """Either cell class of a Cartesian axis: declared or derived."""


class CartesianAxisIndex(AnyCartesianAxisIndex):
    """A *declared* Cartesian axis. Only these have a staggered partner."""


class Staggered[D: CartesianAxisIndex](AnyCartesianAxisIndex):
    """The derived partner cell class of a declared axis."""

    base: ClassVar[type[CartesianAxisIndex]]


class LocalDimensionIndex(DimensionIndex):
    """A neighbor index. Not an axis, and unmoved relative to the root."""

    owner: ClassVar[type | None]


class Dims[*Ds]: ...


class Field[DimsT, DT]: ...


# ---------------- user declarations ----------------


class I(CartesianAxisIndex): ...  # noqa: E742  # single-letter dimension names are the domain convention


class J(CartesianAxisIndex): ...


class K(CartesianAxisIndex): ...


class C(DimensionIndex): ...  # a mesh location: no index arithmetic, no partner


class E(DimensionIndex): ...


class V2E:
    class Local(LocalDimensionIndex): ...


Khalf: TypeAlias = Staggered[K]  # a domain name for the partner is just an alias


# ---------------- P1: the rejections ----------------


def p1_doubly_staggered(
    f: Field[Dims[Staggered[Staggered[I]]], float],  # EXPECT-ERROR: [type-var]
) -> None: ...


def p1_staggered_local(
    f: Field[Dims[Staggered[V2E.Local]], float],  # EXPECT-ERROR: [type-var]
) -> None: ...


def p1_staggered_mesh_location(
    f: Field[Dims[Staggered[C]], float],  # EXPECT-ERROR: [type-var]
) -> None: ...


# also in alias position, not only in an annotation
Doubly: TypeAlias = Staggered[Staggered[K]]  # EXPECT-ERROR: [type-var]


# ---------------- P2: what must keep working (no widening) ----------------


def p2_any_dimension(d: type[DimensionIndex]) -> None: ...


p2_any_dimension(Staggered[I])  # must be ACCEPTED: staggered dims are still dimensions
p2_any_dimension(V2E.Local)  # must be ACCEPTED: locals unmoved
p2_any_dimension(C)  # must be ACCEPTED: mesh locations unmoved
p2_any_dimension(I)  # must be ACCEPTED


def p2_in_dims(f: Field[Dims[I, J, Staggered[K]], float]) -> None: ...  # ACCEPTED


def p2_through_alias(f: Field[Dims[Khalf], float]) -> None: ...  # ACCEPTED


# ---------------- P3: the axis levels discriminate ----------------


def p3_any_axis(d: type[AnyCartesianAxisIndex]) -> None: ...  # e.g. flip_staggered, check_dims


p3_any_axis(I)  # must be ACCEPTED
p3_any_axis(Staggered[I])  # must be ACCEPTED
p3_any_axis(C)  # EXPECT-ERROR: a mesh location is not an axis
p3_any_axis(V2E.Local)  # EXPECT-ERROR: a local dimension is not an axis


def p3_declared_axis(d: type[CartesianAxisIndex]) -> None: ...  # e.g. the Staggered constructor


p3_declared_axis(I)  # must be ACCEPTED
p3_declared_axis(Staggered[I])  # EXPECT-ERROR: derived, not declared
p3_declared_axis(C)  # EXPECT-ERROR: mesh location
p3_declared_axis(V2E.Local)  # EXPECT-ERROR: local dimension
