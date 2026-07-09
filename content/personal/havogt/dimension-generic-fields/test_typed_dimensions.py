# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for the typed-dimensions prototype.

Run with::

    uv run pytest docs/development/investigations/prototypes/typed_dimensions/

The static tests shell out to mypy (without the gt4py plugin, see `mypy.ini`):
`static_checks.py` must be clean, `static_errors.py` must error exactly on the
lines carrying the EXPECT-ERROR marker comment.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest
from typed_dimensions import Dimension, DimensionBase, DimensionKind, NdField, Staggered, dual


HERE = pathlib.Path(__file__).parent


class I(Dimension): ...  # noqa: E742


class J(Dimension): ...


class K(Dimension, kind=DimensionKind.VERTICAL): ...


# --- runtime behavior of dimension classes ----------------------------------


def test_dimension_class_attributes():
    assert I.value == "I"
    assert I.kind == DimensionKind.HORIZONTAL
    assert K.kind == DimensionKind.VERTICAL


def test_dimension_classes_are_not_instantiable():
    with pytest.raises(TypeError, match="cannot be instantiated"):
        I()


def test_name_based_equality_and_hashing():
    class OtherI(Dimension):
        value = "I"

    assert I == OtherI  # today's `Dimension("I") == Dimension("I")` semantics
    assert I != J
    assert hash(I) == hash(OtherI)
    assert {I: 1}[OtherI] == 1  # usable as dict keys (domains, offset providers)


def test_staggered_is_a_materialized_singleton_class():
    assert Staggered[I] is Staggered[I]
    assert isinstance(Staggered[I], type) and issubclass(Staggered[I], DimensionBase)
    assert not issubclass(Staggered[I], Dimension)  # staggered dims are not user-declarable dims
    assert Staggered[I].base is I
    assert Staggered[I].kind == I.kind
    assert Staggered[I] != I


def test_staggering_is_an_involution_at_the_type_constructor_level():
    assert dual(I) is Staggered[I]
    assert dual(Staggered[I]) is I
    with pytest.raises(TypeError, match="doubly staggered"):
        Staggered[Staggered[I]]


def test_connectivity_construction():
    conn = I + 1
    assert (conn.domain_dim, conn.codomain, conn.offset) == (I, I, 1)
    conn = I + 1 / 2
    assert (conn.domain_dim, conn.codomain, conn.offset) == (Staggered[I], I, 0.5)
    conn = Staggered[I] - 1 / 2
    assert (conn.domain_dim, conn.codomain, conn.offset) == (I, Staggered[I], -0.5)
    with pytest.raises(ValueError, match="half-integral"):
        I + 0.7  # statically a (claimed) staggering shift; rejected at runtime


# --- runtime behavior of shifts ----------------------------------------------


def make_ifield(data) -> NdField:
    return NdField((I,), np.asarray(data, dtype=float))


def test_plain_shift():
    a = make_ifield([0.0, 1.0, 2.0, 3.0])
    b = a(I + 1)
    assert b.dimensions == (I,)
    np.testing.assert_array_equal(b.ndarray, [1.0, 2.0, 3.0, 0.0])


def test_staggering_relabels_and_shifts():
    a = make_ifield([0.0, 1.0, 2.0, 3.0])
    b = a(I + 1 / 2)  # b[i] = a at position i + 1/2, i.e. a[i + 1] in index space
    assert b.dimensions == (Staggered[I],)
    np.testing.assert_array_equal(b.ndarray, [1.0, 2.0, 3.0, 0.0])

    c = b(Staggered[I] + 1 / 2)  # back to the unstaggered grid
    assert c.dimensions == (I,)
    # two +1/2 staggerings == one full shift
    np.testing.assert_array_equal(c.ndarray, a(I + 1).ndarray)


def test_staggering_roundtrip_identity():
    a = make_ifield([0.0, 1.0, 2.0, 3.0])
    back = a(I + 1 / 2)(Staggered[I] - 1 / 2)
    assert back.dimensions == (I,)
    np.testing.assert_array_equal(back.ndarray, a.ndarray)


def avg(f: NdField) -> NdField:
    """Runtime twin of the dual-generic `avg` in `static_checks.py` (one body, both grids)."""
    (dim,) = f.dims
    return f(dim - 1 / 2) + f(dim + 1 / 2)


def test_avg_to_staggered():
    a = make_ifield([0.0, 1.0, 2.0, 3.0])
    result = avg(a)
    assert result.dimensions == (Staggered[I],)
    # at staggered point i+1/2: a[i] + a[i+1] (periodic)
    np.testing.assert_array_equal(result.ndarray, [1.0, 3.0, 5.0, 3.0])


def test_avg_from_staggered():
    g = NdField((Staggered[I],), np.asarray([0.0, 1.0, 2.0, 3.0]))
    result = avg(g)
    assert result.dimensions == (I,)
    # at unstaggered point i: g[i-1] + g[i] (the staggered neighbors at i -+ 1/2)
    np.testing.assert_array_equal(result.ndarray, [3.0, 1.0, 3.0, 5.0])


def weighted_avg(a: NdField, w: NdField) -> NdField:
    """Runtime twin of the dual-generic `weighted_avg` in `static_checks.py`."""
    (dim,) = a.dims
    return w * (a(dim + 1 / 2) + a(dim - 1 / 2))


def test_weighted_avg():
    a = NdField((I,), np.asarray([0.0, 1.0, 2.0, 3.0]))
    w = NdField((Staggered[I],), np.asarray([1.0, 2.0, 3.0, 4.0]))
    result = weighted_avg(a, w)
    assert result.dimensions == (Staggered[I],)
    # at staggered point i+1/2: w[i] * (a[i+1] + a[i]) (periodic)
    np.testing.assert_array_equal(result.ndarray, [1.0, 6.0, 15.0, 12.0])

    back = weighted_avg(result, a)
    assert back.dimensions == (I,)


def test_avg_roundtrip_dims():
    a = make_ifield([0.0, 1.0, 2.0, 3.0])
    assert avg(avg(a)).dimensions == (I,)
    assert avg(avg(avg(a))).dimensions == (Staggered[I],)


def test_c_grid_gradient():
    a = make_ifield([0.0, 1.0, 4.0, 9.0])
    grad = a(I + 1 / 2) - a(I - 1 / 2)  # grad[i+1/2] = a[i+1] - a[i]
    assert grad.dimensions == (Staggered[I],)
    np.testing.assert_array_equal(grad.ndarray, [1.0, 3.0, 5.0, -9.0])


def test_rank2_substitution_at_runtime():
    a = NdField((I, J), np.zeros((3, 4)))
    assert a(J + 1 / 2).dimensions == (I, Staggered[J])
    assert a(I - 1 / 2).dimensions == (Staggered[I], J)


def test_mixing_dual_grids_raises():
    a = make_ifield([0.0, 1.0])
    with pytest.raises(ValueError, match="different domains"):
        a + a(I + 1 / 2)


def test_shift_along_missing_dimension_raises():
    a = make_ifield([0.0, 1.0])
    with pytest.raises(ValueError, match="no dimension"):
        a(J + 1)


# --- static behavior (mypy as the oracle) ------------------------------------


def run_mypy(*files: str) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", str(HERE / "mypy.ini"), *files],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout


def test_static_checks_pass():
    returncode, output = run_mypy("typed_dimensions.py", "static_checks.py")
    assert returncode == 0, f"expected mypy to pass, got:\n{output}"


def test_static_errors_rejected_exactly_where_expected():
    expected_lines = {
        lineno
        for lineno, line in enumerate((HERE / "static_errors.py").read_text().splitlines(), start=1)
        if "# EXPECT-ERROR" in line
    }
    assert expected_lines, "marker lines missing"

    _, output = run_mypy("static_errors.py")
    reported_lines = {
        int(match.group(1))
        for match in re.finditer(r"^static_errors\.py:(\d+): error:", output, re.M)
    }
    assert reported_lines == expected_lines, (
        f"mypy errors on lines {sorted(reported_lines)},"
        f" expected exactly {sorted(expected_lines)}:\n{output}"
    )
