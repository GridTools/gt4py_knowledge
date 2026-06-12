# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Prototype: dimensions as types, with statically typed staggering.

This is a self-contained prototype accompanying
``docs/development/investigations/dimension-generic-fields.md``. It demonstrates
that the following are expressible in standard Python typing (verified by mypy,
see ``static_checks.py`` / ``static_errors.py`` / ``test_typed_dimensions.py``):

1. **Dimensions as types**: a concrete dimension is a *class* (``class I(Dimension)``),
   not an instance, so ``Field[Dims[I], float]`` is a real, plugin-free generic
   annotation and dimensions can be used as TypeVar bounds/values.
2. **Value-level use of dimension classes**: the metaclass gives the class object
   the same value-level behavior dimension *instances* have today
   (``I + 1`` builds a connectivity, ``str(I)``, equality by name, ...).
3. **Staggering as a type-level function**: ``Staggered[I]`` is the dual
   (staggered) counterpart of ``I``. ``I + 1/2`` is typed
   ``Connectivity[Staggered[I], I]`` and ``Staggered[I] + 1/2`` is typed
   ``Connectivity[I, Staggered[I]]`` — the dual-grid involution is encoded in a
   pair of overloads, so no doubly-staggered types ever arise.
4. **Shift typing as positional substitution**: ``field(conn)`` replaces exactly
   the matching dimension in the field's ``Dims[...]``, expressed with
   rank-bounded overloads (rank <= 3 here; generated code can extend this).

Runtime semantics implemented here are the minimum needed to make the prototype
executable (pure relabeling/translation, periodic via ``np.roll``); domain
shrinking and interpolation conventions are design-doc topics, not typing topics.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any, Callable, ClassVar, Generic, Protocol, TypeVar, cast, overload

import numpy as np
from typing_extensions import TypeVarTuple, Unpack


class DimensionKind(enum.Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    LOCAL = "local"


# `D` is bound to `Dimension` (the base of *user-declared, unstaggered*
# dimensions) so that `Staggered[Staggered[I]]` is statically unrepresentable;
# all other dimension TypeVars are bound to the root `DimensionBase` and thus
# range over staggered dimensions too.
D = TypeVar("D", bound="Dimension")
AnyD = TypeVar("AnyD", bound="DimensionBase")
NewD = TypeVar("NewD", bound="DimensionBase")
OldD = TypeVar("OldD", bound="DimensionBase")
D0 = TypeVar("D0", bound="DimensionBase")
D1 = TypeVar("D1", bound="DimensionBase")
D2 = TypeVar("D2", bound="DimensionBase")


class DimensionMeta(type):
    """
    Metaclass of all dimension classes.

    Provides the value-level behavior that today lives on `common.Dimension`
    *instances*, but on the class object itself: building connectivities with
    ``+``/``-``, name-based equality (interop with legacy instances), printing.
    """

    value: str
    kind: DimensionKind

    def __repr__(cls) -> str:
        return f"{cls.value}[{cls.kind.value}]"

    # Name-based equality mirrors today's `Dimension.__eq__` (which only
    # compares `value`) and makes dimension classes interoperate with legacy
    # `Dimension` instances during a migration period.
    def __eq__(cls, other: object) -> bool:
        if isinstance(other, DimensionMeta) or type(other).__name__ == "Dimension":
            return cls.value == other.value  # type: ignore[attr-defined]
        return NotImplemented

    def __ne__(cls, other: object) -> bool:
        # explicit metaclass lookup: `cls.__eq__` would find `object.__eq__`
        # on the dimension class itself instead of this method
        result = DimensionMeta.__eq__(cls, other)
        return result if result is NotImplemented else not result

    def __hash__(cls) -> int:
        return hash(cls.value)

    # The fractional-offset overloads encode staggering:
    # - integer offsets stay on the same grid,
    # - half-integral offsets move to the dual grid.
    # The `Staggered[D]` overload must come before the generic `D` overload so
    # that shifting *from* a staggered dimension resolves back to the base
    # dimension (involution) instead of producing `Staggered[Staggered[D]]`.
    #
    # The `type: ignore[misc]` comments silence a *definition-site* mypy
    # restriction ("Self argument missing for a non-static method"): mypy does
    # not accept `cls: type[D]` as the self-type of a metaclass method, but it
    # binds the annotated signatures correctly at every *call site* (verified
    # in `static_checks.py`), which is what matters here.
    @overload
    def __add__(cls: type[AnyD], offset: int) -> Connectivity[AnyD, AnyD]: ...  # type: ignore[misc]
    @overload
    def __add__(  # type: ignore[misc]
        cls: type[Staggered[D]], offset: float
    ) -> Connectivity[D, Staggered[D]]: ...
    @overload
    def __add__(  # type: ignore[misc]
        cls: type[D], offset: float
    ) -> Connectivity[Staggered[D], D]: ...
    def __add__(cls, offset: int | float) -> Connectivity[Any, Any]:
        if isinstance(offset, int):
            return Connectivity(cls, cls, offset)
        if offset % 1.0 != 0.5:
            raise ValueError(
                f"Only integral or half-integral offsets are supported, got '{offset}'."
            )
        return Connectivity(dual(cls), cls, offset)

    @overload
    def __sub__(cls: type[AnyD], offset: int) -> Connectivity[AnyD, AnyD]: ...  # type: ignore[misc]
    @overload
    def __sub__(  # type: ignore[misc]
        cls: type[Staggered[D]], offset: float
    ) -> Connectivity[D, Staggered[D]]: ...
    @overload
    def __sub__(  # type: ignore[misc]
        cls: type[D], offset: float
    ) -> Connectivity[Staggered[D], D]: ...
    def __sub__(cls, offset: int | float) -> Connectivity[Any, Any]:
        return cast("Connectivity[Any, Any]", cast(Any, cls) + (-offset))


class DimensionBase(metaclass=DimensionMeta):
    """
    Root of the dimension hierarchy: anything usable as an entry of ``Dims[...]``.

    There are exactly two kinds of subclasses: user-declared dimensions
    (subclasses of `Dimension`) and their staggered duals (`Staggered[...]`).
    Generic code ranging over *any* dimension binds TypeVars to this root.
    """

    value: ClassVar[str]
    kind: ClassVar[DimensionKind] = DimensionKind.HORIZONTAL

    def __init_subclass__(cls, /, kind: DimensionKind | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "value" not in cls.__dict__:
            cls.value = cls.__name__
        if kind is not None:
            cls.kind = kind

    def __new__(cls) -> DimensionBase:
        raise TypeError(
            f"Dimension '{cls.value}' is a type and cannot be instantiated;"
            " use the class object itself as the value."
        )


class Dimension(DimensionBase):
    """
    Base class of user-declared dimensions; a concrete dimension is a *subclass*,
    not an instance.

    Spelling::

        class I(Dimension): ...


        class K(Dimension, kind=DimensionKind.VERTICAL): ...

    The class object plays the role of today's `Dimension` instance everywhere
    a value is needed (`I + 1`, dict keys, `domain` entries, ...), while also
    being a proper type for annotations (`Field[Dims[I], float]`, TypeVar
    bounds, `Staggered[I]`).

    `Staggered`'s type parameter is bound to this class (not to the root
    `DimensionBase`), which makes ``Staggered[Staggered[I]]`` a *static* error
    in addition to the runtime rejection.
    """


_staggered_cache: dict[DimensionMeta, DimensionMeta] = {}


class Staggered(DimensionBase, Generic[D]):
    """
    Type-level constructor for the dual (staggered) counterpart of a dimension.

    ``Staggered[I]`` is usable both as a type (in annotations) and as a value
    (``__class_getitem__`` materializes a singleton dimension class), so the
    static and runtime views never diverge. Since ``D`` is bound to
    `Dimension`, ``Staggered[Staggered[I]]`` is rejected statically
    (``[type-var]``) as well as at runtime.
    """

    #: The dimension this staggered dimension is dual to.
    base: ClassVar[type[Dimension]]

    def __class_getitem__(cls, item: Any) -> Any:
        if cls is Staggered and isinstance(item, DimensionMeta):
            base: DimensionMeta = item
            if _is_staggered(base):
                raise TypeError(
                    f"'Staggered[{base.value}]': doubly staggered dimensions do not exist;"
                    " a full offset stays on the same grid, use an integer offset instead."
                )
            if base not in _staggered_cache:
                staggered = DimensionMeta(
                    f"Staggered[{base.value}]",
                    (Staggered,),
                    {"base": base, "value": f"{base.value}½", "kind": base.kind},
                )
                _staggered_cache[base] = staggered
            return _staggered_cache[base]
        return super().__class_getitem__(item)  # type: ignore[misc]  # typing machinery (TypeVar case)


def _is_staggered(dim: DimensionMeta) -> bool:
    # plain `bool` (not a TypeGuard) on purpose: callers need `dim` to stay a
    # `DimensionMeta` in both branches
    return issubclass(dim, Staggered)


def dual(dim: DimensionMeta) -> DimensionMeta:
    """Return the dual-grid counterpart of a dimension class (involution)."""
    if _is_staggered(dim):
        return cast(DimensionMeta, dim.base)  # type: ignore[attr-defined]  # staggered dims carry `base`
    return cast(DimensionMeta, Staggered[dim])  # type: ignore[valid-type]  # value-level use


@dataclasses.dataclass(frozen=True)
class Connectivity(Generic[NewD, OldD]):
    """
    Maps positions of `domain_dim` to positions of `codomain`.

    Mirrors `common.CartesianConnectivity[DomainDimT, DimT]`: applying it to a
    field replaces `codomain` by `domain_dim` in the field's dimensions
    (cf. `CartesianConnectivity.for_relocation`). The fractional part of
    `offset` encodes the grid change, the integral part the translation.
    """

    domain_dim: type[NewD]
    codomain: type[OldD]
    offset: float


ShapeTs = TypeVarTuple("ShapeTs")


class Dims(tuple[Unpack[ShapeTs]]):
    """Same as `common.Dims` today; its arguments are now real types."""


DimsT = TypeVar("DimsT", bound=Dims, covariant=True)
DimsT2 = TypeVar("DimsT2", bound=Dims)
DT = TypeVar("DT")


class Field(Protocol[DimsT, DT]):
    """
    Minimal `Field` protocol: shifts (`__call__`) and same-domain arithmetic.

    The `__call__` overloads implement *positional dimension substitution*:
    a `Connectivity[NewD, OldD]` argument replaces the unique dimension equal
    to `OldD` with `NewD`. One overload per (rank, position) pair is needed
    because a tuple type may contain at most one `TypeVarTuple`; ranks beyond
    the ones spelled here would be generated (rank <= 3 in this prototype).
    """

    @property
    def dims(self) -> tuple[DimensionMeta, ...]:
        """The dimension objects of this field (runtime view of `DimsT`)."""
        ...

    # rank 1
    @overload
    def __call__(
        self: Field[Dims[D0], DT], conn: Connectivity[NewD, D0]
    ) -> Field[Dims[NewD], DT]: ...
    # rank 2
    @overload
    def __call__(
        self: Field[Dims[D0, D1], DT], conn: Connectivity[NewD, D0]
    ) -> Field[Dims[NewD, D1], DT]: ...
    @overload
    def __call__(
        self: Field[Dims[D0, D1], DT], conn: Connectivity[NewD, D1]
    ) -> Field[Dims[D0, NewD], DT]: ...
    # rank 3
    @overload
    def __call__(
        self: Field[Dims[D0, D1, D2], DT], conn: Connectivity[NewD, D0]
    ) -> Field[Dims[NewD, D1, D2], DT]: ...
    @overload
    def __call__(
        self: Field[Dims[D0, D1, D2], DT], conn: Connectivity[NewD, D1]
    ) -> Field[Dims[D0, NewD, D2], DT]: ...
    @overload
    def __call__(
        self: Field[Dims[D0, D1, D2], DT], conn: Connectivity[NewD, D2]
    ) -> Field[Dims[D0, D1, NewD], DT]: ...
    def __call__(self, conn: Connectivity[Any, Any]) -> Field[Any, DT]: ...

    # Same-domain arithmetic: mixing fields on dual grids is a static error.
    # (Dims-promoting broadcast arithmetic is an orthogonal typing question,
    # see the design document.)
    def __add__(self: Field[DimsT2, DT], other: Field[DimsT2, DT] | DT) -> Field[DimsT2, DT]: ...
    def __sub__(self: Field[DimsT2, DT], other: Field[DimsT2, DT] | DT) -> Field[DimsT2, DT]: ...
    def __mul__(self: Field[DimsT2, DT], other: Field[DimsT2, DT] | DT) -> Field[DimsT2, DT]: ...


# --- Dual-generic operator support (library side) ---------------------------
#
# Python typing has no type-level function, so "the dual grid of X"
# (I -> Staggered[I], Staggered[I] -> I) cannot be computed in a user
# signature. Instead the library provides:
#   - `Dual[X]`, a *marker* type only meaningful in annotations of
#     dual-generic operators, and
#   - a decorator whose return type is a protocol carrying the
#     dual-resolving overload pair (written once per signature shape).
# Users then write a single natural signature; in gt4py proper this typing
# would live on `@field_operator` itself, so users add nothing at all.

AnyX = TypeVar("AnyX", bound="DimensionBase")


class Dual(DimensionBase, Generic[AnyX]):
    """
    Marker for "the dual-grid counterpart of dimension `AnyX`".

    Unlike `Staggered`, this is not a real dimension: it never materializes and
    is resolved away by the typing of `dual_operator` (statically) or by the
    DSL frontend at specialization time (`dual()`), where the reduction rule
    ``Dual[Dual[X]] = X`` can be applied for real.
    """


class DualUnaryOperator(Protocol[DT]):
    """A unary field operator mapping every field to its dual grid: ``X -> Dual[X]``."""

    @overload
    def __call__(self, f: Field[Dims[Staggered[D]], DT]) -> Field[Dims[D], DT]: ...
    @overload
    def __call__(self, f: Field[Dims[D], DT]) -> Field[Dims[Staggered[D]], DT]: ...


class DualBinaryOperator(Protocol[DT]):
    """A binary field operator of shape ``(X, Dual[X]) -> Dual[X]``.

    The second argument (e.g. a weight) already lives on the dual grid of the
    first; the call is rejected when both arguments live on the same grid.
    """

    @overload
    def __call__(
        self, f: Field[Dims[Staggered[D]], DT], g: Field[Dims[D], DT]
    ) -> Field[Dims[D], DT]: ...
    @overload
    def __call__(
        self, f: Field[Dims[D], DT], g: Field[Dims[Staggered[D]], DT]
    ) -> Field[Dims[Staggered[D]], DT]: ...


# One decorator, overloaded per recognized signature shape (in gt4py proper
# these overloads would live on `@field_operator`).
@overload
def dual_operator(
    fn: Callable[[Field[Dims[AnyX], DT]], Field[Dims[Dual[AnyX]], DT]],
) -> DualUnaryOperator[DT]: ...
@overload
def dual_operator(
    fn: Callable[[Field[Dims[AnyX], DT], Field[Dims[Dual[AnyX]], DT]], Field[Dims[Dual[AnyX]], DT]],
) -> DualBinaryOperator[DT]: ...
def dual_operator(fn: Callable[..., Any]) -> Any:
    """Give a dual-generic operator its precise (overloaded) call type."""
    return fn  # the returned protocol *is* the resolved meaning of `Dual`


@dataclasses.dataclass(frozen=True)
class NdField(Generic[DimsT, DT]):
    """
    Runtime implementation of the `Field` protocol (periodic boundaries).

    Conventions (position of staggered point ``i`` is ``i + 1/2``):
    given ``b = a(conn)``, ``b[p] = a[p + conn.offset]`` where positions are
    grid points of the respective dimensions. In index space this is a shift by
    ``ceil(offset)`` when reading from an unstaggered dimension and by
    ``floor(offset)`` when reading from a staggered one.
    """

    dimensions: tuple[DimensionMeta, ...]
    ndarray: np.ndarray

    def __post_init__(self) -> None:
        if len(self.dimensions) != self.ndarray.ndim:
            raise ValueError("Number of dimensions does not match array rank.")

    @property
    def dims(self) -> tuple[DimensionMeta, ...]:
        return self.dimensions

    def __call__(self, conn: Connectivity[Any, Any]) -> NdField[Any, DT]:
        if conn.codomain not in self.dimensions:
            raise ValueError(
                f"Field defined on '{self.dimensions}' has no dimension '{conn.codomain}'."
            )
        axis = self.dimensions.index(conn.codomain)
        if issubclass(conn.codomain, Staggered):
            index_shift = math.floor(conn.offset)
        else:
            index_shift = math.ceil(conn.offset)
        new_dims = tuple(
            cast(DimensionMeta, conn.domain_dim) if i == axis else d
            for i, d in enumerate(self.dimensions)
        )
        return NdField(new_dims, np.roll(self.ndarray, -index_shift, axis=axis))

    def _binop(self, other: NdField[Any, Any] | Any, op: Any) -> NdField[DimsT, DT]:
        if isinstance(other, NdField):
            if self.dimensions != other.dimensions:
                raise ValueError(
                    f"Fields are defined on different domains:"
                    f" '{self.dimensions}' and '{other.dimensions}'."
                )
            return NdField(self.dimensions, op(self.ndarray, other.ndarray))
        return NdField(self.dimensions, op(self.ndarray, other))

    def __add__(self, other: NdField[Any, Any] | Any) -> NdField[DimsT, DT]:
        return self._binop(other, np.add)

    def __sub__(self, other: NdField[Any, Any] | Any) -> NdField[DimsT, DT]:
        return self._binop(other, np.subtract)

    def __mul__(self, other: NdField[Any, Any] | Any) -> NdField[DimsT, DT]:
        return self._binop(other, np.multiply)
