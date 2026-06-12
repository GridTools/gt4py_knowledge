# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Prototype of a ``FieldData`` protocol separating field *data* from *domain* computations.

Status: **exploration prototype**, not wired into the public API. It battle-tests the
idea that all `Field` implementations (buffer-backed, function-backed, constant, lazy,
piecewise) can share a single generic `Field` implementation (:class:`DataField`) that
owns the :class:`common.Domain` and performs *all* domain computations, delegating the
value handling to an exchangeable :class:`FieldData` implementation.

Design
------

A :class:`FieldData` is a mapping from *absolute* integer coordinates to values:

- **positional**: axes are unnamed; the owning field's `Domain` provides the names and
  the axis order. The field layer translates named/absolute/relative index specs into
  positional, absolute items before talking to the data layer.
- **absolute**: coordinates are the same integers the field's `Domain` ranges contain.
  There is no "buffer index space" in the protocol. This is the central design decision
  and answers the question whether the buffer origin (domains not starting at 0) is in
  the way of such a protocol:

  1. A *relative* (0-based) data protocol cannot represent fields with a domain that is
     unbounded below (e.g. a boundary condition extending to infinity): there is no
     finite origin to rebase to. Absolute addressing handles finite and infinite
     domains uniformly.
  2. With absolute addressing the origin does not disappear, but it collapses into a
     private detail of :class:`NdArrayFieldData` (``origins``: the absolute coordinate
     of buffer index 0, per axis). No other component ever sees it; the domain layer
     performs pure set arithmetic on `UnitRange`s and never subtracts a start.
  3. The data's support may be a *superset* of the field's domain (it is "at least
     defined on the domain"). E.g. `restrict` only narrows the domain; narrowing the
     data is an optional storage optimization (`FieldData.restrict`), not a semantic
     requirement. A domain-translation `premap` is an O(1) origin shift.

The protocol is intentionally small. Operations needed by the field layer:

- ``materialize(box, xp)``  -- evaluate the data on a finite absolute box into a buffer;
  used for arithmetic on finite domains, `ndarray`/`asnumpy`/`as_scalar`.
- ``gather_values(indices, xp)`` -- evaluate at arbitrary absolute coordinate arrays;
  used for `premap` with gather connectivities (advanced indexing) and to compose data
  lazily. For a function-backed data this evaluates the function *without* ever
  materializing its (possibly infinite) domain.
- ``restrict(items)`` -- fix axes to absolute points (dropping them) and optionally
  narrow ranges (storage hint).
- ``translate(offsets)`` -- precompose with a translation; backs O(1) cartesian shifts.
- ``remap_axes(axis_map)`` -- transpose/insert value-broadcast axes; backs `broadcast`.

The protocol exposes *evaluation primitives*, deliberately not the field operations
themselves (no ``__add__`` etc. on the data): operations on the data would need a
pairwise answer for every combination of data kinds (ndarray+function, lazy+piecewise,
...) -- the N x N dispatch problem this refactoring is meant to remove from the
`Field` level would reappear one level down. Instead, ``gather_values`` is the
universal "call the field as a function at absolute coordinates" primitive, so any
data kind combines with any other for free (e.g. `premap` of a function field through
a neighbor table needs no function-field-specific code). Should a data kind ever be
able to combine smarter than evaluate-and-apply (constant folding, fusion), an
*optional* ``combine(op, *others)`` hook consulted before the generic path could be
added -- there is no concrete case yet.

In this model :class:`FunctionFieldData` is the semantic ground truth -- any data is
behaviorally a function from coordinates to values -- and the other implementations
are storage/performance specializations: a constant field is just a function ignoring
its coordinates (:func:`constant_data` is a helper, not a class), buffers materialize
as views instead of evaluating a closure, and :class:`LazyFieldData` adds memoization
plus the retained knowledge "this *will be* a buffer" (needed for future capabilities
a closure cannot answer: mutability, `__gt_buffer_info__`, zero-copy `ndarray`).

Notable consequences observed while battle-testing (see
``tests/next_tests/unit_tests/embedded_tests/test_field_data.py``):

- `sub_domain` + `_get_slices_from_domain_slice` of the current `NdArrayField` collapse
  into one absolute-item computation (:func:`_absolute_items`); there is no separate
  "slice the buffer" step in the field layer anymore.
- Pointwise ops compose uniformly for every data kind: ``a + b`` is
  ``lambda x: a(x) + b(x)`` (a composed :class:`FunctionFieldData` evaluating the
  operands via ``gather_values``). On *infinite* intersections this is the only option;
  on finite ones, materializing eagerly into a buffer (today's `NdArrayField`
  behavior) is a one-line policy in the shared layer
  (`EAGER_POINTWISE_ON_FINITE_DOMAINS`), not something the data implementations
  decide pairwise. Always-composing is correct but unmemoized: expression trees
  re-evaluate on every access, hence eager is the default.
- A lazy (deferred-materialization) field is just :class:`LazyFieldData` wrapping a
  factory; equivalently it could be a `FunctionFieldData` whose function is the
  ndarray lookup (``lambda *coords: thunk().gather_values(coords, xp)``).
- `concat_where` with an infinite boundary-condition field produces a field on an
  infinite domain backed by :class:`PiecewiseFieldData`; this is not representable
  with today's `NdArrayField`.

Out of scope of the prototype (notes for a real implementation):

- Mutability (`MutableField.__setitem__`) needs a writable-buffer capability on the
  data (only ndarray-backed data has it); likely a separate `MutableFieldData`
  protocol exposing the buffer plus its box.
- The array namespace (`xp`) is fixed to NumPy at the field layer here; the real
  implementation should carry it per field (e.g. taken from the data or an allocator)
  as `NdArrayField` subclasses do today.
- `__gt_buffer_info__` / DaCe interop only make sense for ndarray-backed data and
  would be an optional capability of the data implementation.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
from collections.abc import Callable, Sequence
from types import ModuleType

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.eve.extended_typing import Any, Optional, Protocol, cast, runtime_checkable
from gt4py.next import common
from gt4py.next.embedded import (
    common as embedded_common,
    exceptions as embedded_exceptions,
    nd_array_field,
)
from gt4py.next.ffront import experimental, fbuiltins


#: Positional, absolute restriction item: an absolute point (drops the axis) or an
#: absolute range (narrows the axis; possibly infinite).
DataItem = int | common.UnitRange

#: Positional, absolute region of `ndim` axes (ranges possibly infinite).
DataBox = tuple[common.UnitRange, ...]


@runtime_checkable
class FieldData(Protocol):
    """Mapping from *absolute*, *positional* integer coordinates to values.

    The support (set of coordinates with defined values) is an axis-aligned box, not
    exposed on the protocol; the owning field guarantees it only asks for coordinates
    inside the support (the field's domain is always contained in it).
    """

    @property
    def ndim(self) -> int: ...

    @property
    def dtype(self) -> core_defs.DType: ...

    def materialize(self, box: DataBox, xp: ModuleType) -> core_defs.NDArrayObject:
        """Evaluate on the finite absolute `box`, returning an array of the box's shape."""
        ...

    def gather_values(
        self, indices: tuple[core_defs.NDArrayObject, ...], xp: ModuleType
    ) -> core_defs.NDArrayObject:
        """
        Evaluate at absolute coordinates given as one index array per axis.

        All index arrays have the same (already broadcast) shape, which is also the
        shape of the result: ``result[p] = data(indices[0][p], ..., indices[ndim-1][p])``.
        """
        ...

    def restrict(self, items: tuple[DataItem, ...]) -> FieldData:
        """
        Restrict to absolute `items`, one per axis.

        An `int` item fixes the axis to that absolute point and drops it. A `UnitRange`
        item is a *storage hint*: the result's support must still cover it, but an
        implementation may shrink storage to it (or ignore it).
        """
        ...

    def translate(self, offsets: tuple[int, ...]) -> FieldData:
        """Precompose with a translation: ``new(x) = old(x + offsets)``."""
        ...

    def remap_axes(self, axis_map: tuple[Optional[int], ...]) -> FieldData:
        """
        Rearrange axes: output axis `k` reads input axis `axis_map[k]`.

        Every input axis must appear exactly once; `None` introduces a new axis on
        which the values do not depend (value-broadcast axis, infinite support).
        """
        ...


def _box_shape(box: DataBox) -> tuple[int, ...]:
    return tuple(len(r) for r in box)


@dataclasses.dataclass(frozen=True)
class NdArrayFieldData:
    """
    Buffer-backed :class:`FieldData`.

    `origins[i]` is the absolute coordinate of buffer index 0 along axis `i`; `None`
    marks a value-broadcast axis (buffer extent 1, every coordinate reads index 0,
    infinite support). The origin is **internal** to this class: it never crosses the
    `FieldData` interface.
    """

    buffer: core_defs.NDArrayObject
    origins: tuple[Optional[int], ...]

    def __post_init__(self) -> None:
        assert len(self.origins) == self.buffer.ndim
        assert all(
            origin is not None or size == 1 for origin, size in zip(self.origins, self.buffer.shape)
        )

    @property
    def ndim(self) -> int:
        return self.buffer.ndim

    @property
    def dtype(self) -> core_defs.DType:
        return core_defs.dtype(self.buffer.dtype.type)

    def materialize(self, box: DataBox, xp: ModuleType) -> core_defs.NDArrayObject:
        slices = []
        for rng, origin, size in zip(box, self.origins, self.buffer.shape, strict=True):
            assert common.UnitRange.is_finite(rng)
            if origin is None:
                slices.append(slice(0, 1))
            else:
                start, stop = rng.start - origin, rng.stop - origin
                assert 0 <= start <= stop <= size, "Box outside of the data's support."
                slices.append(slice(start, stop))
        result = xp.asarray(self.buffer[tuple(slices)])
        shape = _box_shape(box)
        return xp.broadcast_to(result, shape) if result.shape != shape else result

    def gather_values(
        self, indices: tuple[core_defs.NDArrayObject, ...], xp: ModuleType
    ) -> core_defs.NDArrayObject:
        take = tuple(
            xp.zeros_like(idx) if origin is None else idx - origin
            for idx, origin in zip(indices, self.origins, strict=True)
        )
        return xp.asarray(self.buffer)[take]

    def restrict(self, items: tuple[DataItem, ...]) -> NdArrayFieldData:
        slices: list[slice | int] = []
        new_origins = []
        for item, origin, size in zip(items, self.origins, self.buffer.shape, strict=True):
            if isinstance(item, common.UnitRange):
                if origin is None or not common.UnitRange.is_finite(item):
                    slices.append(slice(None))
                    new_origins.append(origin)
                else:
                    start, stop = item.start - origin, item.stop - origin
                    assert 0 <= start <= stop <= size, "Item outside of the data's support."
                    slices.append(slice(start, stop))
                    new_origins.append(item.start)
            else:
                idx = 0 if origin is None else item - origin
                assert 0 <= idx < size, "Item outside of the data's support."
                slices.append(idx)
        return NdArrayFieldData(self.buffer[tuple(slices)], tuple(new_origins))

    def translate(self, offsets: tuple[int, ...]) -> NdArrayFieldData:
        # new(x) = old(x + off) => new buffer index of x is x - (origin - off)
        new_origins = tuple(
            None if origin is None else origin - off
            for origin, off in zip(self.origins, offsets, strict=True)
        )
        return NdArrayFieldData(self.buffer, new_origins)

    def remap_axes(self, axis_map: tuple[Optional[int], ...]) -> NdArrayFieldData:
        input_axes = [a for a in axis_map if a is not None]
        assert sorted(input_axes) == list(range(self.ndim))
        buffer = self.buffer.transpose(input_axes)  # type: ignore[attr-defined]  # not part of `NDArrayObject`, but provided by all supported array libraries
        # `None` inserts a new size-1 axis; `slice(None)` consumes the next transposed axis
        buffer = buffer[tuple(None if a is None else slice(None) for a in axis_map)]
        origins = tuple(None if a is None else self.origins[a] for a in axis_map)
        return NdArrayFieldData(buffer, origins)


@dataclasses.dataclass(frozen=True)
class FunctionFieldData:
    """
    Function-backed :class:`FieldData` with (conceptually) unbounded support.

    `func` must accept `ndim` absolute coordinate arguments (arrays or scalars,
    mutually broadcastable) and evaluate vectorized, e.g. ``lambda i, j: np.sin(i) + j``.
    Since the values are computed, the dtype cannot be introspected and must be given.
    """

    func: Callable[..., Any]
    ndim: int
    dtype: core_defs.DType

    def materialize(self, box: DataBox, xp: ModuleType) -> core_defs.NDArrayObject:
        assert len(box) == self.ndim
        coords = tuple(
            xp.reshape(
                xp.arange(rng.start, rng.stop),
                tuple(-1 if i == axis else 1 for i in range(self.ndim)),
            )
            for axis, rng in enumerate(box)
        )
        result = xp.asarray(self.func(*coords), dtype=self.dtype.scalar_type)
        shape = _box_shape(box)
        return xp.broadcast_to(result, shape) if result.shape != shape else result

    def gather_values(
        self, indices: tuple[core_defs.NDArrayObject, ...], xp: ModuleType
    ) -> core_defs.NDArrayObject:
        assert len(indices) == self.ndim
        result = xp.asarray(self.func(*indices), dtype=self.dtype.scalar_type)
        if self.ndim > 0 and result.shape != indices[0].shape:
            result = xp.broadcast_to(result, indices[0].shape)
        return result

    def restrict(self, items: tuple[DataItem, ...]) -> FunctionFieldData:
        assert len(items) == self.ndim
        fixed = {
            axis: item for axis, item in enumerate(items) if not isinstance(item, common.UnitRange)
        }
        if not fixed:
            return self  # range items are only storage hints, nothing to narrow here

        def restricted(
            *coords: Any, _func: Callable = self.func, _fixed: dict[int, int] = fixed
        ) -> Any:
            coords_iter = iter(coords)
            return _func(
                *(
                    _fixed[axis] if axis in _fixed else next(coords_iter)
                    for axis in range(self.ndim)
                )
            )

        return FunctionFieldData(restricted, self.ndim - len(fixed), self.dtype)

    def translate(self, offsets: tuple[int, ...]) -> FunctionFieldData:
        assert len(offsets) == self.ndim

        def translated(*coords: Any, _func: Callable = self.func) -> Any:
            return _func(*(c + o for c, o in zip(coords, offsets, strict=True)))

        return FunctionFieldData(translated, self.ndim, self.dtype)

    def remap_axes(self, axis_map: tuple[Optional[int], ...]) -> FunctionFieldData:
        input_positions = {a: k for k, a in enumerate(axis_map) if a is not None}
        assert sorted(input_positions.keys()) == list(range(self.ndim))

        def remapped(*coords: Any, _func: Callable = self.func) -> Any:
            return _func(*(coords[input_positions[axis]] for axis in range(self.ndim)))

        return FunctionFieldData(remapped, len(axis_map), self.dtype)


def constant_data(value: core_defs.Scalar, ndim: int) -> FunctionFieldData:
    """Data of a constant field: a function ignoring its coordinates."""
    return FunctionFieldData(
        lambda *coords: value, ndim, core_defs.dtype(np.asarray(value).dtype.type)
    )


@dataclasses.dataclass(frozen=True)
class LazyFieldData:
    """
    Defers the construction of the underlying data until values are needed.

    Structural operations (`restrict`, `translate`, `remap_axes`) stay lazy; value
    operations (`materialize`, `gather_values`) force the factory exactly once.

    Note: equivalently a lazy buffer field is a :class:`FunctionFieldData` whose
    function is the array lookup, ``lambda *coords: factory().gather_values(coords, xp)``;
    this class merely adds memoization of the forced data.
    """

    factory: Callable[[], FieldData]
    ndim: int
    dtype: core_defs.DType

    @functools.cached_property
    def _forced(self) -> FieldData:
        data = self.factory()
        assert data.ndim == self.ndim and data.dtype == self.dtype
        return data

    @property
    def is_forced(self) -> bool:
        return "_forced" in self.__dict__

    def materialize(self, box: DataBox, xp: ModuleType) -> core_defs.NDArrayObject:
        return self._forced.materialize(box, xp)

    def gather_values(
        self, indices: tuple[core_defs.NDArrayObject, ...], xp: ModuleType
    ) -> core_defs.NDArrayObject:
        return self._forced.gather_values(indices, xp)

    def restrict(self, items: tuple[DataItem, ...]) -> LazyFieldData:
        new_ndim = sum(1 for item in items if isinstance(item, common.UnitRange))
        return LazyFieldData(lambda: self._forced.restrict(items), new_ndim, self.dtype)

    def translate(self, offsets: tuple[int, ...]) -> LazyFieldData:
        return LazyFieldData(lambda: self._forced.translate(offsets), self.ndim, self.dtype)

    def remap_axes(self, axis_map: tuple[Optional[int], ...]) -> LazyFieldData:
        return LazyFieldData(lambda: self._forced.remap_axes(axis_map), len(axis_map), self.dtype)


@dataclasses.dataclass(frozen=True)
class PiecewiseFieldData:
    """
    Data composed of disjoint pieces, each defined on its own absolute box.

    Backs the result of `concat_where` when a piece (e.g. an infinite boundary
    condition) prevents eager concatenation into a single buffer.
    """

    pieces: tuple[tuple[DataBox, FieldData], ...]

    def __post_init__(self) -> None:
        assert self.pieces
        assert all(data.ndim == self.ndim for _, data in self.pieces)
        assert all(len(box) == self.ndim for box, _ in self.pieces)

    @property
    def ndim(self) -> int:
        return self.pieces[0][1].ndim

    @property
    def dtype(self) -> core_defs.DType:
        dtypes = {data.dtype for _, data in self.pieces}
        assert len(dtypes) == 1, "Pieces with mixed dtypes are not supported."
        return next(iter(dtypes))

    def materialize(self, box: DataBox, xp: ModuleType) -> core_defs.NDArrayObject:
        out = xp.empty(_box_shape(box), dtype=self.dtype.scalar_type)
        filled = 0
        for piece_box, piece_data in self.pieces:
            sub = tuple(rng & piece_rng for rng, piece_rng in zip(box, piece_box, strict=True))
            if any(rng.is_empty() for rng in sub):
                continue
            out_slices = tuple(
                slice(sub_rng.start - rng.start, sub_rng.stop - rng.start)
                for sub_rng, rng in zip(sub, box, strict=True)
            )
            out[out_slices] = piece_data.materialize(sub, xp)
            filled += math.prod(_box_shape(sub))
        assert filled == math.prod(_box_shape(box)), "Box not covered by the pieces."
        return out

    def gather_values(
        self, indices: tuple[core_defs.NDArrayObject, ...], xp: ModuleType
    ) -> core_defs.NDArrayObject:
        out = None
        for piece_box, piece_data in self.pieces:
            # clamp coordinates into the piece's box so evaluation is in-support
            # everywhere; out-of-piece results are masked out below
            clamped = tuple(
                xp.clip(
                    idx,
                    rng.start if common.UnitRange.is_left_finite(rng) else None,
                    rng.stop - 1 if common.UnitRange.is_right_finite(rng) else None,
                )
                for idx, rng in zip(indices, piece_box, strict=True)
            )
            values = piece_data.gather_values(clamped, xp)
            if out is None:
                out = values
            else:
                mask = xp.ones(indices[0].shape, dtype=bool)
                for idx, rng in zip(indices, piece_box, strict=True):
                    if common.UnitRange.is_left_finite(rng):
                        mask &= idx >= rng.start
                    if common.UnitRange.is_right_finite(rng):
                        mask &= idx < rng.stop
                out = xp.where(mask, values, out)
        assert out is not None
        return out

    def restrict(self, items: tuple[DataItem, ...]) -> FieldData:
        new_pieces = []
        for piece_box, piece_data in self.pieces:
            piece_items: list[DataItem] = []
            keep = True
            for item, rng in zip(items, piece_box, strict=True):
                if isinstance(item, common.UnitRange):
                    intersection = item & rng
                    if intersection.is_empty() and not item.is_empty():
                        keep = False
                        break
                    piece_items.append(intersection)
                else:
                    if item not in rng:
                        keep = False
                        break
                    piece_items.append(item)
            if keep:
                new_box = tuple(it for it in piece_items if isinstance(it, common.UnitRange))
                new_pieces.append((new_box, piece_data.restrict(tuple(piece_items))))
        assert new_pieces
        if len(new_pieces) == 1:
            return new_pieces[0][1]
        return PiecewiseFieldData(tuple(new_pieces))

    def translate(self, offsets: tuple[int, ...]) -> PiecewiseFieldData:
        return PiecewiseFieldData(
            tuple(
                (
                    tuple(rng - off for rng, off in zip(box, offsets, strict=True)),
                    data.translate(offsets),
                )
                for box, data in self.pieces
            )
        )

    def remap_axes(self, axis_map: tuple[Optional[int], ...]) -> PiecewiseFieldData:
        return PiecewiseFieldData(
            tuple(
                (
                    tuple(common.UnitRange.infinite() if a is None else box[a] for a in axis_map),
                    data.remap_axes(axis_map),
                )
                for box, data in self.pieces
            )
        )


# -- The generic field implementation on top of FieldData --


def _absolute_items(domain: common.Domain, index: common.AnyIndexSpec) -> tuple[DataItem, ...]:
    """
    Resolve any index spec into one absolute item (point or range) per domain axis.

    This single absolute computation replaces both `embedded_common.sub_domain` (domain
    arithmetic) and `nd_array_field._get_slices_from_domain_slice` (buffer arithmetic)
    of the `NdArrayField` implementation: with absolute addressing there is no separate
    relative-slice step.
    """
    index = embedded_common.canonicalize_any_index_sequence(index)
    index_sequence = common.as_any_index_sequence(index)

    items: list[DataItem] = []
    if common.is_absolute_index_sequence(index_sequence):
        for dim, rng in domain:
            if (pos := embedded_common._find_index_of_dim(dim, index_sequence)) is None:
                items.append(rng)
                continue
            idx = index_sequence[pos][1]
            if isinstance(idx, common.UnitRange):
                if not idx <= rng:
                    raise embedded_exceptions.IndexOutOfBounds(
                        domain=domain, indices=index, index=index_sequence[pos], dim=dim
                    )
                items.append(idx)
            else:
                assert common.is_int_index(idx)
                if idx not in rng:
                    raise embedded_exceptions.IndexOutOfBounds(
                        domain=domain, indices=index, index=index_sequence[pos], dim=dim
                    )
                items.append(int(idx))
        return tuple(items)

    if common.is_relative_index_sequence(index_sequence):
        expanded = embedded_common._expand_ellipsis(index_sequence, len(domain))
        if len(domain) < len(expanded):
            raise IndexError(
                f"Can not access dimension with index {index} of 'Field' with {len(domain)} dimensions."
            )
        expanded += (slice(None),) * (len(domain) - len(expanded))
        for (dim, rng), idx in zip(domain, expanded, strict=True):
            if isinstance(idx, slice):
                try:
                    items.append(embedded_common._slice_range(rng, idx))
                except IndexError as ex:
                    raise embedded_exceptions.IndexOutOfBounds(
                        domain=domain, indices=index, index=idx, dim=dim
                    ) from ex
            else:
                assert common.is_int_index(idx)
                assert common.UnitRange.is_finite(rng)
                absolute = (rng.start if idx >= 0 else rng.stop) + idx
                if absolute not in rng:
                    raise embedded_exceptions.IndexOutOfBounds(
                        domain=domain, indices=index, index=idx, dim=dim
                    )
                items.append(int(absolute))
        return tuple(items)

    raise IndexError(f"Unsupported index type: '{index}'.")


def _origins_for(domain: common.Domain, shape: tuple[int, ...]) -> tuple[Optional[int], ...]:
    """Per-axis buffer origin for a buffer of `shape` laid out over `domain` (`None`: value-broadcast axis)."""
    origins: list[Optional[int]] = []
    for rng, size in zip(domain.ranges, shape, strict=True):
        if common.UnitRange.is_finite(rng) and len(rng) == size:
            origins.append(rng.start)
        else:
            assert size == 1, (
                "Buffer shape incompatible with domain: needs to match the domain shape "
                "or be 1 (value-broadcast axis)."
            )
            origins.append(None)
    return tuple(origins)


def _result_dtype(
    op: Callable[..., Any], operands: Sequence[FieldData | core_defs.Scalar], xp: ModuleType
) -> core_defs.DType:
    """Result dtype of pointwise `op`, probed on empty arrays (robust e.g. for `true_divide` on ints)."""
    probes = [
        xp.empty((0,), dtype=o.dtype.scalar_type) if isinstance(o, FieldData) else o
        for o in operands
    ]
    return core_defs.dtype(op(*probes).dtype.type)


def _broadcast_data_field(field: DataField, dims: Sequence[common.Dimension]) -> DataField:
    if field.domain.dims == tuple(dims):
        return field
    axis_map = tuple(field.domain.dim_index(dim) for dim in dims)
    assert all(d in dims for d in field.domain.dims)
    new_ranges = tuple(
        common.UnitRange.infinite() if a is None else field.domain.ranges[a] for a in axis_map
    )
    return DataField(
        common.Domain(dims=tuple(dims), ranges=new_ranges), field._data.remap_axes(axis_map)
    )


def as_data_field(field: common.Field) -> DataField:
    """Adapt any `Field` to a `DataField` (migration shim: views the buffer, no copy)."""
    if isinstance(field, DataField):
        return field
    ndarray = field.ndarray
    return DataField(
        field.domain, NdArrayFieldData(ndarray, _origins_for(field.domain, ndarray.shape))
    )


#: Policy switch for pointwise operations on *finite* domains: materialize eagerly into
#: a buffer (today's `NdArrayField` semantics) or keep composing functions. This is a
#: single decision in the shared field layer, orthogonal to the data implementations:
#: with `False`, `a + b` is always `lambda x: a(x) + b(x)` (via `gather_values`),
#: whatever kind of data `a` and `b` are. On infinite domains composition is the only
#: option. Eager is the default to bound expression re-evaluation and memory surprises.
EAGER_POINTWISE_ON_FINITE_DOMAINS: bool = True


def _pointwise(
    op: Callable[..., Any],
    *operands: common.Field | core_defs.Scalar,
    xp: ModuleType,
) -> DataField:
    """
    Apply a pointwise array op on the intersection of the operands' domains.

    The composed (function) result is universal; materializing on finite domains is an
    optimization policy (see `EAGER_POINTWISE_ON_FINITE_DOMAINS`), not a property of
    the operand data kinds.
    """
    field_operands = [as_data_field(o) for o in operands if isinstance(o, common.Field)]
    assert field_operands
    promoted_dims = common.promote_dims(*(f.domain.dims for f in field_operands))
    broadcasted = [_broadcast_data_field(f, promoted_dims) for f in field_operands]
    broadcasted_iter = iter(broadcasted)
    args: list[FieldData | core_defs.Scalar] = [
        next(broadcasted_iter)._data if isinstance(o, common.Field) else o for o in operands
    ]
    domain = embedded_common.domain_intersection(*(f.domain for f in broadcasted))

    if EAGER_POINTWISE_ON_FINITE_DOMAINS and common.Domain.is_finite(domain):
        box = tuple(domain.ranges)
        values = op(*(a.materialize(box, xp) if isinstance(a, FieldData) else a for a in args))
        data: FieldData = NdArrayFieldData(
            xp.asarray(values), tuple(r.start for r in domain.ranges)
        )
    else:

        def composed(*coords: Any) -> Any:
            return op(
                *(a.gather_values(coords, xp) if isinstance(a, FieldData) else a for a in args)
            )

        data = FunctionFieldData(composed, len(promoted_dims), _result_dtype(op, args, xp))
    return DataField(domain, data)


def _make_field_builtin(
    builtin_name: str, array_builtin_name: str, reverse: bool = False
) -> Callable[..., DataField]:
    def _builtin_op(*operands: common.Field | core_defs.Scalar) -> DataField:
        xp = DataField.array_ns
        if reverse:
            operands = operands[::-1]
        return _pointwise(getattr(xp, array_builtin_name), *operands, xp=xp)

    _builtin_op.__name__ = builtin_name
    return _builtin_op


@dataclasses.dataclass(frozen=True, eq=False)
class DataField(common.FieldBuiltinFuncRegistry, common.Field):
    # note: the registry mixin is deliberately first in the MRO so that its concrete
    # `__gt_builtin_func__` classmethod satisfies the bare `ClassVar` declared on the
    # `Field` protocol (otherwise mypy treats `DataField` as abstract).
    """
    Generic `Field`: a `Domain` plus any `FieldData`.

    All domain computations happen here (or in `embedded.common`), once, for every kind
    of data; the data implementation never sees a `Dimension` nor a buffer origin
    coming from outside.
    """

    _domain: common.Domain
    _data: FieldData

    #: Prototype simplification: materialization is always NumPy. The real
    #: implementation would carry the array namespace per instance (cf. `NdArrayField`).
    array_ns = np

    def __post_init__(self) -> None:
        assert len(self._domain) == self._data.ndim

    # -- constructors --

    @classmethod
    def from_array(
        cls,
        data: Any,
        /,
        *,
        domain: common.DomainLike,
        dtype: Optional[core_defs.DTypeLike] = None,
    ) -> DataField:
        domain = common.domain(domain)
        xp = cls.array_ns
        array = xp.asarray(
            data, dtype=None if dtype is None else core_defs.dtype(dtype).scalar_type
        )
        assert array.ndim == len(domain)
        return cls(domain, NdArrayFieldData(array, _origins_for(domain, array.shape)))

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        /,
        *,
        domain: common.DomainLike,
        dtype: core_defs.DTypeLike,
    ) -> DataField:
        domain = common.domain(domain)
        return cls(domain, FunctionFieldData(func, len(domain), core_defs.dtype(dtype)))

    @classmethod
    def constant(cls, value: core_defs.Scalar, /, *, domain: common.DomainLike) -> DataField:
        domain = common.domain(domain)
        return cls(domain, constant_data(value, len(domain)))

    @classmethod
    def from_lazy_array(
        cls,
        factory: Callable[[], Any],
        /,
        *,
        domain: common.DomainLike,
        dtype: core_defs.DTypeLike,
    ) -> DataField:
        domain_ = common.domain(domain)
        xp = cls.array_ns

        def make_data() -> FieldData:
            array = xp.asarray(factory(), dtype=core_defs.dtype(dtype).scalar_type)
            return NdArrayFieldData(array, _origins_for(domain_, array.shape))

        return cls(domain_, LazyFieldData(make_data, len(domain_), core_defs.dtype(dtype)))

    # -- properties --

    @property
    def domain(self) -> common.Domain:
        return self._domain

    @property
    def data(self) -> FieldData:
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        return self.domain.shape

    @property
    def dtype(self) -> core_defs.DType:
        return self._data.dtype

    @property
    def codomain(self) -> type[core_defs.Scalar]:
        return self.dtype.scalar_type

    @property
    def __gt_origin__(self) -> tuple[int, ...]:
        assert common.Domain.is_finite(self.domain)
        return tuple(-r.start for r in self.domain.ranges)

    @property
    def ndarray(self) -> core_defs.NDArrayObject:
        if not common.Domain.is_finite(self.domain):
            raise ValueError(
                f"Cannot materialize 'Field' with non-finite domain '{self.domain}'; restrict it first."
            )
        return self._data.materialize(tuple(self.domain.ranges), self.array_ns)

    def asnumpy(self) -> np.ndarray:
        return np.asarray(self.ndarray)

    def as_scalar(self) -> core_defs.Scalar:
        if self.domain.ndim != 0:
            raise ValueError(
                f"'as_scalar' is only valid on 0-dimensional 'Field's, got a {self.domain.ndim}-dimensional 'Field'."
            )
        return self.asnumpy()[()]  # type: ignore[return-value]  # ensured by the 0-d check

    def materialized(self) -> DataField:
        """
        Force evaluation into a buffer-backed field over this field's (finite) domain.

        The explicit "force" found in delayed-evaluation array libraries (Repa's
        ``computeS``, Dask's ``persist``, Accelerate's manifest arrays): bounds
        re-evaluation of composed function data and pins down memory.
        """
        if isinstance(self._data, NdArrayFieldData):
            return self
        return DataField(
            self.domain,
            NdArrayFieldData(self.ndarray, tuple(r.start for r in self.domain.ranges)),
        )

    # -- restriction --

    def restrict(self, index: common.AnyIndexSpec) -> DataField:
        items = _absolute_items(self.domain, index)
        new_domain = common.Domain(
            *(
                common.NamedRange(dim, item)
                for dim, item in zip(self.domain.dims, items)
                if isinstance(item, common.UnitRange)
            )
        )
        return DataField(new_domain, self._data.restrict(items))

    __getitem__ = restrict

    # -- premap --

    def premap(self, *connectivities: common.Connectivity | fbuiltins.FieldOffset) -> DataField:
        conn_fields = [
            c if isinstance(c, common.Connectivity) else c.as_connectivity_field()
            for c in connectivities
        ]
        is_gather = [isinstance(c, common.GatherConnectivity) for c in conn_fields]
        if not any(is_gather):
            return self._affine_premap(conn_fields)
        assert all(is_gather), "Mixing affine and gather connectivities is not allowed."
        return self._gather_premap(cast(list[common.GatherConnectivity], conn_fields))

    def _affine_premap(self, connectivities: Sequence[common.Connectivity]) -> DataField:
        """Domain relabel; for translations the data shift is O(1) on buffer-backed data."""
        new_domain = self.domain
        offsets = [0] * self.domain.ndim
        for connectivity in connectivities:
            dim = connectivity.codomain
            dim_idx = self.domain.dim_index(dim)
            if dim_idx is None:
                raise ValueError(
                    f"Incompatible index field expects a data field with dimension '{dim}'"
                    f" but got '{self.domain}'."
                )
            current_range = self.domain[dim_idx].unit_range
            new_ranges = connectivity.inverse_image(current_range)
            new_domain = new_domain.replace(dim_idx, *new_ranges)
            # new(x) = old(c(x)): probe the translation offset of `c` via the preimage of a
            # unit range; works also for infinite field domains (no domain starts involved)
            probe = connectivity.inverse_image(common.UnitRange(0, 1))
            assert len(probe) == 1 and len(probe[0].unit_range) == 1, (
                "Affine 'premap' expects a translation/relocation connectivity."
            )
            offsets[dim_idx] = -probe[0].unit_range.start
        return DataField(new_domain, self._data.translate(tuple(offsets)))

    def _gather_premap(self, connectivities: Sequence[common.GatherConnectivity]) -> DataField:
        """Advanced-indexing gather: index arrays stay in *absolute* coordinates."""
        xp = self.array_ns
        new_domain = _gather_output_domain(self.domain, connectivities)
        assert common.Domain.is_finite(new_domain), (
            "Prototype limitation: gather 'premap' requires a finite output domain."
        )
        conn_by_codomain = {c.codomain: c for c in connectivities}
        indices = tuple(
            nd_array_field._connectivity_index_array(conn_by_codomain[dim], new_domain, xp)
            if dim in conn_by_codomain
            else nd_array_field._identity_index_array(new_domain, dim, xp)
            for dim in self.domain.dims
        )
        values = self._data.gather_values(indices, xp)
        return DataField(
            new_domain, NdArrayFieldData(values, tuple(r.start for r in new_domain.ranges))
        )

    def __call__(
        self,
        index_field: common.Connectivity | fbuiltins.FieldOffset,
        *args: common.Connectivity | fbuiltins.FieldOffset,
    ) -> DataField:
        return functools.reduce(
            lambda field, connectivity: field.premap(connectivity), [index_field, *args], self
        )

    # -- operators --

    __abs__ = _make_field_builtin("abs", "abs")
    __neg__ = _make_field_builtin("neg", "negative")
    __pos__ = _make_field_builtin("pos", "positive")

    __add__ = __radd__ = _make_field_builtin("add", "add")
    __sub__ = _make_field_builtin("sub", "subtract")
    __rsub__ = _make_field_builtin("sub", "subtract", reverse=True)
    __mul__ = __rmul__ = _make_field_builtin("mul", "multiply")
    __truediv__ = _make_field_builtin("div", "divide")
    __rtruediv__ = _make_field_builtin("div", "divide", reverse=True)
    __floordiv__ = _make_field_builtin("floordiv", "floor_divide")
    __rfloordiv__ = _make_field_builtin("floordiv", "floor_divide", reverse=True)
    __pow__ = _make_field_builtin("pow", "power")
    __mod__ = _make_field_builtin("mod", "mod")
    __rmod__ = _make_field_builtin("mod", "mod", reverse=True)

    __ne__ = _make_field_builtin("not_equal", "not_equal")
    __eq__ = _make_field_builtin("equal", "equal")
    __gt__ = _make_field_builtin("greater", "greater")
    __ge__ = _make_field_builtin("greater_equal", "greater_equal")
    __lt__ = _make_field_builtin("less", "less")
    __le__ = _make_field_builtin("less_equal", "less_equal")

    def __and__(self, other: common.Field | core_defs.Scalar) -> DataField:
        if self.dtype == core_defs.BoolDType():
            return _make_field_builtin("logical_and", "logical_and")(self, other)
        raise NotImplementedError("'__and__' not implemented for non-'bool' fields.")

    __rand__ = __and__

    def __or__(self, other: common.Field | core_defs.Scalar) -> DataField:
        if self.dtype == core_defs.BoolDType():
            return _make_field_builtin("logical_or", "logical_or")(self, other)
        raise NotImplementedError("'__or__' not implemented for non-'bool' fields.")

    __ror__ = __or__

    def __xor__(self, other: common.Field | core_defs.Scalar) -> DataField:
        if self.dtype == core_defs.BoolDType():
            return _make_field_builtin("logical_xor", "logical_xor")(self, other)
        raise NotImplementedError("'__xor__' not implemented for non-'bool' fields.")

    __rxor__ = __xor__

    def __invert__(self) -> DataField:
        if self.dtype == core_defs.BoolDType():
            return _make_field_builtin("invert", "invert")(self)
        raise NotImplementedError("'__invert__' not implemented for non-'bool' fields.")


def _gather_output_domain(
    field_domain: common.Domain, connectivities: Sequence[common.GatherConnectivity]
) -> common.Domain:
    """
    Like `nd_array_field._gather_output_domain`, but supporting infinite codomain ranges.

    If the field is defined on the whole codomain (infinite range), the preimage is the
    connectivity's entire domain; `inverse_image` implementations currently require a
    finite image range, so that case is handled explicitly here.
    """
    domain = field_domain
    for conn in connectivities:
        cod = conn.codomain
        cod_range = domain[cod].unit_range
        if common.UnitRange.is_finite(cod_range):
            narrowed = {nr.dim: nr.unit_range for nr in conn.inverse_image(cod_range)}
        else:
            narrowed = {nr.dim: nr.unit_range for nr in conn.domain}
        introduced = [
            common.NamedRange(dim, rng) for dim, rng in narrowed.items() if dim not in domain.dims
        ]
        result: list[common.NamedRange] = []
        for nr in domain:
            if nr.dim == cod:
                if cod in narrowed:
                    result.append(common.NamedRange(cod, nr.unit_range & narrowed[cod]))
                result.extend(introduced)
            elif nr.dim in narrowed:
                result.append(common.NamedRange(nr.dim, nr.unit_range & narrowed[nr.dim]))
            else:
                result.append(nr)
        domain = common.Domain(*result)
    return domain


# -- builtins --


def _builtins_broadcast(
    field: common.Field | core_defs.Scalar, new_dimensions: tuple[common.Dimension, ...]
) -> common.Field:
    if isinstance(field, common.Field):
        return _broadcast_data_field(as_data_field(field), new_dimensions)
    raise AssertionError("Scalar case not reachable from 'fbuiltins.broadcast'.")


DataField.register_builtin_func(fbuiltins.broadcast, _builtins_broadcast)
DataField.register_builtin_func(fbuiltins.where, _make_field_builtin("where", "where"))
DataField.register_builtin_func(fbuiltins.minimum, _make_field_builtin("minimum", "minimum"))
DataField.register_builtin_func(fbuiltins.maximum, _make_field_builtin("maximum", "maximum"))


def _concat_where(
    mask_domain: common.Domain, true_field: common.Field, false_field: common.Field
) -> DataField:
    """
    `concat_where` on `FieldData`: pieces may be infinite (e.g. boundary conditions).

    A finite result is eagerly concatenated into one buffer (matching today's
    `NdArrayField`); an infinite result stays a `PiecewiseFieldData`, which the
    current `NdArrayField` cannot represent at all.
    """
    if mask_domain.ndim != 1:
        raise NotImplementedError(
            "'concat_where': Can only concatenate fields with a 1-dimensional domain."
        )
    dim = mask_domain.dims[0]

    t_field, f_field = as_data_field(true_field), as_data_field(false_field)
    promoted_dims = common.promote_dims(t_field.domain.dims, f_field.domain.dims)
    t_field = _broadcast_data_field(t_field, promoted_dims)
    f_field = _broadcast_data_field(f_field, promoted_dims)
    # intersect in the dimensions orthogonal to `dim` so all pieces share those ranges
    t_domain, f_domain = embedded_common.restrict_to_intersection(
        t_field.domain, f_field.domain, ignore_dims=dim
    )
    t_field, f_field = t_field[t_domain], f_field[f_domain]

    pieces: list[DataField] = []
    true_piece = embedded_common.domain_intersection(t_field.domain, mask_domain)
    if not true_piece.is_empty():
        pieces.append(t_field[true_piece])
    for inverted in nd_array_field._invert_domain(mask_domain):
        false_piece = embedded_common.domain_intersection(f_field.domain, inverted)
        if not false_piece.is_empty():
            pieces.append(f_field[false_piece])

    if not pieces:
        empty_ranges = (common.UnitRange(0, 0),) * len(promoted_dims)
        return DataField.from_array(
            np.empty((0,) * len(promoted_dims), dtype=t_field.dtype.scalar_type),
            domain=common.Domain(dims=tuple(promoted_dims), ranges=empty_ranges),
        )

    pieces.sort(key=lambda p: p.domain[dim].unit_range.start)
    for prev, curr in itertools.pairwise(pieces):
        left, right = prev.domain[dim].unit_range.stop, curr.domain[dim].unit_range.start
        if left > right:
            raise ValueError("Fields to concatenate must not overlap.")
        if left < right:
            raise embedded_exceptions.NonContiguousDomain(f"Cannot concatenate fields along {dim}.")

    stacked_range = common.UnitRange(
        pieces[0].domain[dim].unit_range.start, pieces[-1].domain[dim].unit_range.stop
    )
    new_domain = pieces[0].domain.replace(dim, common.NamedRange(dim, stacked_range))

    if len(pieces) == 1:
        return pieces[0]
    data = PiecewiseFieldData(tuple((tuple(p.domain.ranges), p._data) for p in pieces))
    if common.Domain.is_finite(new_domain):
        # eager concatenation, semantics as today's `NdArrayField`
        box = tuple(new_domain.ranges)
        return DataField(
            new_domain,
            NdArrayFieldData(
                data.materialize(box, DataField.array_ns), tuple(r.start for r in box)
            ),
        )
    return DataField(new_domain, data)


DataField.register_builtin_func(experimental.concat_where, _concat_where)  # type: ignore[arg-type]
