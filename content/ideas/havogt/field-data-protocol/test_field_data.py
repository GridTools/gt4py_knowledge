# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Battle tests for the `FieldData` protocol prototype (`embedded.field_data`).

Wherever cheap, results are cross-checked against the existing `NdArrayField`
implementation on identical inputs; the cases `NdArrayField` cannot represent
(infinite/function-backed/lazy/piecewise data) are checked against NumPy references.
"""

import numpy as np
import pytest

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.common import Dimension, DimensionKind, Domain, NamedIndex, NamedRange, UnitRange
from gt4py.next.embedded import exceptions as embedded_exceptions, field_data, nd_array_field
from gt4py.next.embedded.field_data import (
    DataField,
    FieldData,
    FunctionFieldData,
    LazyFieldData,
    NdArrayFieldData,
    PiecewiseFieldData,
)

I = Dimension("I")
J = Dimension("J")
K = Dimension("K", kind=DimensionKind.VERTICAL)

Vertex = Dimension("Vertex")
Edge = Dimension("Edge")
E2VDim = Dimension("E2V", kind=DimensionKind.LOCAL)


def np_field(values, *starts_per_dim):
    """`DataField` over `values` with domain starting at the given (possibly negative) starts."""
    values = np.asarray(values)
    dims = [I, J, K][: values.ndim]
    domain = Domain(
        dims=tuple(dims),
        ranges=tuple(UnitRange(s, s + n) for s, n in zip(starts_per_dim, values.shape)),
    )
    return DataField.from_array(values, domain=domain)


def reference_field(data_field: DataField) -> common.Field:
    """The same field as a classic `NdArrayField` (for cross-checking)."""
    return common._field(data_field.asnumpy(), domain=data_field.domain)


def assert_fields_equal(a: common.Field, b: common.Field):
    assert a.domain == b.domain
    np.testing.assert_array_equal(a.asnumpy(), b.asnumpy())


# -- restriction / origin handling --


def test_ndarray_respects_origin():
    values = np.arange(10.0)
    f = np_field(values, -3)  # domain I=[-3, 7)

    assert f.domain == common.domain({I: (-3, 7)})
    np.testing.assert_array_equal(f.asnumpy(), values)


@pytest.mark.parametrize(
    "index",
    [
        NamedRange(I, UnitRange(0, 4)),  # absolute
        (NamedRange(I, UnitRange(0, 4)),),
        slice(3, 7),  # relative
        np.s_[3:7],
    ],
)
def test_restrict_range(index):
    values = np.arange(10.0)
    f = np_field(values, -3)

    restricted = f[index]
    assert restricted.domain == common.domain({I: (0, 4)})
    np.testing.assert_array_equal(restricted.asnumpy(), values[3:7])
    # cross-check against NdArrayField
    assert_fields_equal(restricted, reference_field(f)[index])


@pytest.mark.parametrize(
    "index, expected",
    [
        (NamedIndex(I, 2), 5.0),  # absolute point
        ((5,), 5.0),  # relative, positive
        (-1, 9.0),  # relative, negative
    ],
)
def test_restrict_point_drops_axis(index, expected):
    f = np_field(np.arange(10.0), -3)

    restricted = f[index]
    assert restricted.domain.ndim == 0
    assert restricted.as_scalar() == expected


def test_restrict_out_of_bounds():
    f = np_field(np.arange(10.0), -3)
    with pytest.raises(embedded_exceptions.IndexOutOfBounds):
        f[NamedRange(I, UnitRange(0, 8))]
    with pytest.raises(embedded_exceptions.IndexOutOfBounds):
        f[:11]  # stop beyond the domain (no clipping, following the array API standard)


def test_restrict_2d_mixed():
    values = np.arange(20.0).reshape(4, 5)
    f = np_field(values, -2, 10)  # I=[-2, 2), J=[10, 15)

    restricted = f[NamedIndex(I, -1), NamedRange(J, UnitRange(12, 14))]
    assert restricted.domain == common.domain({J: (12, 14)})
    np.testing.assert_array_equal(restricted.asnumpy(), values[1, 2:4])
    assert_fields_equal(
        restricted, reference_field(f)[NamedIndex(I, -1), NamedRange(J, UnitRange(12, 14))]
    )


def test_restriction_does_not_copy():
    f = np_field(np.arange(10.0), -3)
    restricted = f[slice(2, 8)]
    assert restricted._data.buffer.base is f._data.buffer


# -- pointwise operations --


def test_binary_op_aligns_by_absolute_position():
    a_values, b_values = np.arange(10.0), np.arange(10.0, 20.0)
    a = np_field(a_values, -3)  # I=[-3, 7)
    b = np_field(b_values, 2)  # I=[2, 12)

    result = a + b
    assert result.domain == common.domain({I: (2, 7)})
    np.testing.assert_array_equal(result.asnumpy(), a_values[5:] + b_values[:5])
    # cross-check against NdArrayField
    assert_fields_equal(result, reference_field(a) + reference_field(b))


def test_binary_op_with_scalar_and_reverse():
    a = np_field(np.arange(10.0), -3)
    assert_fields_equal(a - 1.0, reference_field(a) - 1.0)
    assert_fields_equal(1.0 - a, 1.0 - reference_field(a))


def test_binary_op_broadcasts_dims():
    a = np_field(np.arange(4.0), 1)  # I=[1, 5)
    b_values = np.arange(3.0)
    b = DataField.from_array(b_values, domain=common.domain({J: (10, 13)}))

    result = a + b
    assert result.domain == common.domain({I: (1, 5), J: (10, 13)})
    np.testing.assert_array_equal(result.asnumpy(), a.asnumpy()[:, np.newaxis] + b_values)


def test_comparison_and_logical_ops():
    a = np_field(np.arange(10.0), -3)
    b = np_field(np.arange(10.0)[::-1].copy(), -3)
    mask = a > b
    assert mask.dtype == core_defs.BoolDType()
    assert_fields_equal(mask, reference_field(a) > reference_field(b))
    assert_fields_equal(
        mask & ~mask,
        (reference_field(a) > reference_field(b)) & ~(reference_field(a) > reference_field(b)),
    )


def test_op_with_value_broadcast_buffer():
    # buffer of extent 1 on an infinite axis (value-broadcast), as `NdArrayField` supports
    f = DataField.from_array(
        np.array([[1.0], [2.0]]),
        domain=Domain(NamedRange(I, UnitRange(0, 2)), NamedRange(J, UnitRange.infinite())),
    )
    g = np_field(np.arange(6.0).reshape(2, 3), 0, 5)
    result = f + g
    assert result.domain == g.domain
    np.testing.assert_array_equal(result.asnumpy(), np.array([[1.0], [2.0]]) + g.asnumpy())


def test_mixed_with_nd_array_field():
    # interop: DataField op NdArrayField via the `as_data_field` adapter
    a = np_field(np.arange(10.0), -3)
    b_ref = common._field(np.arange(10.0, 20.0), domain=common.domain({I: (2, 12)}))
    result = a + b_ref
    assert_fields_equal(result, reference_field(a) + b_ref)


# -- function fields & infinite domains --


def test_function_field_materializes_on_absolute_coordinates():
    f = DataField.from_function(lambda i: i**2, domain={I: None}, dtype=np.int64)  # infinite domain
    window = f[NamedRange(I, UnitRange(-3, 4))]
    np.testing.assert_array_equal(window.asnumpy(), np.arange(-3, 4) ** 2)


def test_function_field_op_with_finite_field():
    bc = DataField.from_function(lambda i: 100.0 + i, domain={I: None}, dtype=np.float64)
    interior = np_field(np.arange(5.0), 7)  # I=[7, 12)

    result = bc + interior
    assert result.domain == interior.domain
    np.testing.assert_array_equal(result.asnumpy(), (100.0 + np.arange(7, 12)) + np.arange(5.0))


def test_function_fields_stay_lazy_on_infinite_intersection():
    f = DataField.from_function(lambda i: 2 * i, domain={I: None}, dtype=np.int64)
    g = DataField.from_function(lambda i: i + 1, domain={I: None}, dtype=np.int64)

    result = f + g
    assert result.domain == common.domain({I: None})
    assert isinstance(result._data, FunctionFieldData)
    window = result[NamedRange(I, UnitRange(-2, 3))]
    np.testing.assert_array_equal(window.asnumpy(), 2 * np.arange(-2, 3) + np.arange(-2, 3) + 1)


def test_composed_pointwise_policy(monkeypatch):
    # the composed result `lambda x: a(x) + b(x)` is universal, not specific to
    # function fields: with the eager policy off, every pointwise op composes,
    # whatever data backs the operands, with identical values
    monkeypatch.setattr(field_data, "EAGER_POINTWISE_ON_FINITE_DOMAINS", False)

    a_values, b_values = np.arange(10.0), np.arange(10.0, 20.0)
    a = np_field(a_values, -3)  # I=[-3, 7)
    b = np_field(b_values, 2)  # I=[2, 12)
    fn = DataField.from_function(lambda i: 100.0 + i, domain={I: None}, dtype=np.float64)

    for lhs, rhs in [(a, b), (a, fn), (fn, a)]:
        result = lhs + rhs
        assert isinstance(result._data, FunctionFieldData)

    result = (a + b) * fn  # composition chains; nothing materialized yet
    assert result.domain == common.domain({I: (2, 7)})
    expected = (a_values[5:] + b_values[:5]) * (100.0 + np.arange(2, 7))
    np.testing.assert_array_equal(result.asnumpy(), expected)
    # and the chain stays correct through a translation premap (domain moves, values don't)
    shifted = result.premap(common.CartesianConnectivity.for_translation(I, 2))
    assert shifted.domain == common.domain({I: (0, 5)})
    np.testing.assert_array_equal(shifted.asnumpy(), expected)


def test_materialized_forces_into_buffer(monkeypatch):
    monkeypatch.setattr(field_data, "EAGER_POINTWISE_ON_FINITE_DOMAINS", False)
    a = np_field(np.arange(10.0), -3)
    fn = DataField.from_function(lambda i: 100.0 + i, domain={I: None}, dtype=np.float64)

    composed = (a + fn) * 2.0
    assert isinstance(composed._data, FunctionFieldData)
    forced = composed.materialized()
    assert isinstance(forced._data, NdArrayFieldData)
    assert forced.materialized() is forced  # idempotent
    assert_fields_equal(forced, composed)

    infinite = fn + fn
    with pytest.raises(ValueError, match="non-finite"):
        infinite.materialized()


def test_truediv_dtype_on_lazy_path():
    f = DataField.from_function(lambda i: i, domain={I: None}, dtype=np.int64)
    result = f / 2
    assert result.dtype == core_defs.Float64DType()  # probed, not `result_type` (would be int64)
    # note: relative indexing is ill-defined on an infinite domain, restrict absolutely
    np.testing.assert_array_equal(
        result[NamedRange(I, UnitRange(0, 4))].asnumpy(), np.arange(4) / 2
    )


def test_constant_field():
    c = DataField.constant(42.0, domain={I: None, J: None})
    f = np_field(np.zeros((2, 3)), 5, -5)
    result = c + f
    assert result.domain == f.domain
    np.testing.assert_array_equal(result.asnumpy(), np.full((2, 3), 42.0))


def test_function_field_2d_restrict_point_and_broadcast():
    f = DataField.from_function(lambda i, j: 10 * i + j, domain={I: None, J: None}, dtype=np.int64)
    line = f[NamedIndex(I, 3)]  # fix I=3, keep J infinite
    assert line.domain == common.domain({J: None})
    np.testing.assert_array_equal(
        line[NamedRange(J, UnitRange(0, 4))].asnumpy(), 10 * 3 + np.arange(4)
    )


def test_ndarray_of_infinite_field_raises():
    f = DataField.from_function(lambda i: i, domain={I: None}, dtype=np.int64)
    with pytest.raises(ValueError, match="non-finite"):
        _ = f.ndarray


# -- premap --


def test_affine_premap_translation_is_o1_and_correct():
    values = np.arange(10.0)
    f = np_field(values, -3)
    conn = common.CartesianConnectivity.for_translation(I, 2)

    shifted = f.premap(conn)
    # data is not copied, only the origin moves
    assert shifted._data.buffer is f._data.buffer
    # cross-check semantics against NdArrayField
    assert_fields_equal(shifted, reference_field(f).premap(conn))


def test_affine_premap_relocation():
    I_half = Dimension("I_half")
    f = np_field(np.arange(10.0), -3)
    conn = common.CartesianConnectivity.for_relocation(I, I_half)

    relocated = f.premap(conn)
    assert_fields_equal(relocated, reference_field(f).premap(conn))


def test_affine_premap_translation_of_function_field():
    f = DataField.from_function(lambda i: i**2, domain={I: None}, dtype=np.int64)
    conn = common.CartesianConnectivity.for_translation(I, 2)

    shifted = f.premap(conn)
    np.testing.assert_array_equal(
        shifted[NamedRange(I, UnitRange(0, 4))].asnumpy(), (np.arange(0, 4) + 2) ** 2
    )


@pytest.fixture
def e2v():
    table = np.array([[0, 2], [2, 4], [4, 0]])
    return common._connectivity(table, codomain=Vertex, domain={Edge: 3, E2VDim: 2})


def test_gather_premap_matches_nd_array_field(e2v):
    values = np.arange(5.0) * 10
    f = DataField.from_array(values, domain={Vertex: 5})

    gathered = f.premap(e2v)
    reference = common._field(values, domain=common.domain({Vertex: 5})).premap(e2v)
    assert_fields_equal(gathered, reference)


def test_gather_premap_of_function_field(e2v):
    # a function field over the (infinite) Vertex dimension gathered through a
    # neighbor table: impossible with `NdArrayField`, no materialization of the source
    f = DataField.from_function(lambda v: 100.0 + v, domain={Vertex: None}, dtype=np.float64)

    gathered = f.premap(e2v)
    assert gathered.domain == common.domain({Edge: 3, E2VDim: 2})
    np.testing.assert_array_equal(gathered.asnumpy(), 100.0 + e2v.ndarray)


# -- lazy fields --


def test_lazy_field_defers_and_memoizes():
    counter = {"calls": 0}

    def factory():
        counter["calls"] += 1
        return np.arange(10.0)

    f = DataField.from_lazy_array(factory, domain={I: (-3, 7)}, dtype=np.float64)
    assert counter["calls"] == 0

    # structural ops stay lazy
    window = f[slice(2, 8)]
    shifted = f.premap(common.CartesianConnectivity.for_translation(I, 1))
    assert isinstance(window._data, LazyFieldData) and isinstance(shifted._data, LazyFieldData)
    assert counter["calls"] == 0

    # value access forces exactly once (per lazy chain root)
    np.testing.assert_array_equal(window.asnumpy(), np.arange(10.0)[2:8])
    assert counter["calls"] == 1
    _ = window + np_field(np.ones(6), -1)
    assert counter["calls"] == 1  # memoized


# -- concat_where: the boundary-condition use case --


def test_concat_where_finite_matches_nd_array_field():
    cond_domain = common.domain({I: (0, 5)})
    t = np_field(np.arange(10.0), -3)
    f = np_field(np.arange(100.0, 110.0), 0)

    result = field_data._concat_where(cond_domain, t, f)
    reference = nd_array_field._concat_where(cond_domain, reference_field(t), reference_field(f))
    assert_fields_equal(result, reference)
    assert isinstance(result._data, NdArrayFieldData)  # finite => eager, buffer-backed


def test_concat_where_infinite_boundary_condition():
    # boundary condition extending to all points outside the interior:
    # currently not representable in embedded (motivating use case)
    interior = np_field(np.arange(10.0), 0)  # I=[0, 10)
    boundary = DataField.constant(-1.0, domain={I: None})

    result = field_data._concat_where(common.domain({I: (0, 10)}), interior, boundary)
    assert result.domain == common.domain({I: None})
    assert isinstance(result._data, PiecewiseFieldData)

    window = result[NamedRange(I, UnitRange(-5, 15))]
    expected = np.concatenate([np.full(5, -1.0), np.arange(10.0), np.full(5, -1.0)])
    np.testing.assert_array_equal(window.asnumpy(), expected)


def test_concat_where_piecewise_gather():
    # gather (premap) through a piecewise field: indices crossing the boundary
    interior = np_field(np.arange(10.0), 0)
    boundary = DataField.from_function(lambda i: -float(1) * i, domain={I: None}, dtype=np.float64)
    composed = field_data._concat_where(common.domain({I: (0, 10)}), interior, boundary)

    conn = common.CartesianConnectivity.for_translation(I, -3)
    shifted = composed.premap(conn)  # values now looked up at i - 3
    window = shifted[NamedRange(I, UnitRange(0, 6))]
    expected = np.array([3.0, 2.0, 1.0, 0.0, 1.0, 2.0])  # bc at [-3,0), interior at [0,3)
    np.testing.assert_array_equal(window.asnumpy(), expected)


def test_concat_where_2d_orthogonal_intersection():
    cond_domain = common.domain({I: (0, 2)})
    t = np_field(np.arange(12.0).reshape(4, 3), -1, 0)  # I=[-1,3), J=[0,3)
    f = np_field(np.arange(100.0, 120.0).reshape(4, 5), 0, -1)  # I=[0,4), J=[-1,4)

    result = field_data._concat_where(cond_domain, t, f)
    reference = nd_array_field._concat_where(cond_domain, reference_field(t), reference_field(f))
    assert_fields_equal(result, reference)


# -- protocol sanity --


@pytest.mark.parametrize(
    "data",
    [
        NdArrayFieldData(np.arange(6.0).reshape(2, 3), (0, -1)),
        FunctionFieldData(lambda i, j: i + j, 2, core_defs.Int64DType()),
        LazyFieldData(
            lambda: NdArrayFieldData(np.zeros((2, 2)), (0, 0)), 2, core_defs.Float64DType()
        ),
        PiecewiseFieldData(
            (
                ((UnitRange(0, 2), UnitRange(0, 2)), NdArrayFieldData(np.zeros((2, 2)), (0, 0))),
                ((UnitRange(2, 4), UnitRange(0, 2)), NdArrayFieldData(np.ones((2, 2)), (2, 0))),
            )
        ),
    ],
)
def test_implementations_satisfy_protocol(data):
    assert isinstance(data, FieldData)


def test_data_field_is_a_common_field():
    f = np_field(np.arange(4.0), 0)
    assert isinstance(f, common.Field)


def test_remap_axes_consistency_between_implementations():
    values = np.arange(6.0).reshape(2, 3)
    nd = NdArrayFieldData(values, (5, -1))
    fn = FunctionFieldData(lambda i, j: values[i - 5, j + 1], 2, core_defs.Float64DType())
    box = (UnitRange(6, 7), UnitRange(-1, 2), UnitRange(0, 1))
    axis_map = (0, 1, None)
    np.testing.assert_array_equal(
        nd.remap_axes(axis_map).materialize(box, np),
        fn.remap_axes(axis_map).materialize(box, np),
    )

    transposed_box = (UnitRange(-1, 2), UnitRange(5, 7))
    np.testing.assert_array_equal(
        nd.remap_axes((1, 0)).materialize(transposed_box, np),
        fn.remap_axes((1, 0)).materialize(transposed_box, np),
    )


def test_translate_consistency_between_implementations():
    values = np.arange(10.0)
    nd = NdArrayFieldData(values, (-3,))
    fn = FunctionFieldData(lambda i: values[i + 3], 1, core_defs.Float64DType())
    box = (UnitRange(-5, 3),)
    np.testing.assert_array_equal(
        nd.translate((2,)).materialize(box, np), fn.translate((2,)).materialize(box, np)
    )
