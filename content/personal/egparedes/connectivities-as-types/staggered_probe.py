"""Typing probe for the Cartesian-axis levels and the `Staggered[D]` bound.

Run with `mypy --strict --python-version 3.12` and `pyright --pythonversion 3.12`.

Expected: every line marked `EXPECT-ERROR` reported (the `Staggered[...]`
subscriptions as `[type-var]`, "Type argument ... must be a subtype of
CartesianAxisIndex"), and nothing else. The `must be ACCEPTED` lines are the
point of placing the two axis levels *below* `DimensionIndex` rather than above
it: a staggered dimension stays a `DimensionIndex`, so no `type[DimensionIndex]`
annotation in the gt4py tree widens, and `LocalDimensionIndex` keeps its
position. P4 covers the fourth row of the note's table of checks that become
static — index arithmetic restricted to an axis by a self-type on
`DimensionMeta.__add__` — including the two definition-site suppressions that
restriction costs (see the comment on `DimensionMeta`).

Only the *static* declarations are modelled. At runtime `Staggered` is a
metaclass subscription building one interned class per base, and
`LocalDimensionIndex` is a real subclass of `DimensionIndex` so that eve's
`type[...]` validation is unaffected — see the main note.

Last run: mypy 2.4.0 and pyright 1.1.414 (2026-10-01). Both report exactly the
eleven `EXPECT-ERROR` lines and nothing else — mypy 12 diagnostics (two for the
`Doubly` alias), pyright 11 — so every `must be ACCEPTED` line holds.
"""

from __future__ import annotations

from typing import ClassVar, TypeAlias

# ---------------- the hierarchy ----------------


class Connectivity: ...


class DimensionMeta(type):
    # Restricting index arithmetic to an axis needs a self-type on a metaclass
    # method. BOTH checkers reject that at the *definition* site, with different
    # diagnostics, so two separately-spelled suppressions are required:
    #   mypy    : "self" parameter missing for a non-static method
    #             (or an invalid type for self)            [misc]
    #   pyright : Type of parameter "cls" must be a supertype of its class
    #             "DimensionMeta"       (reportGeneralTypeIssues)
    # Both bind the annotated signature correctly at every call site — which is
    # what P4 below checks.
    def __add__(  # type: ignore[misc]
        cls: type[AnyCartesianAxisIndex],  # pyright: ignore[reportGeneralTypeIssues]
        offset: int | float,
    ) -> Connectivity:
        raise NotImplementedError


class DimensionIndex(metaclass=DimensionMeta):
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


# ---------------- P4: the `__add__` self-type restricts index arithmetic ----------------


_ok_axis = I + 1  # must be ACCEPTED
_ok_staggered = Staggered[I] + 1  # must be ACCEPTED: a staggered dim is still an axis
_ok_half = K + 0.5  # must be ACCEPTED
_bad_location = C + 1  # EXPECT-ERROR: a mesh location has no index arithmetic
_bad_local = V2E.Local + 1  # EXPECT-ERROR: a local dimension has no index arithmetic
