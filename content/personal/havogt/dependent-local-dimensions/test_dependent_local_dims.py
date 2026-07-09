# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for the dependent-local-dimensions investigation.

Run with::

    uv run pytest docs/development/investigations/prototypes/dependent_local_dims/

Covers (see ``dependent-local-dimensions.md``):
- the reduction-order claim (§2) on the numpy reference mock,
- the status quo: current GT4Py rejects connectivity chains at decoration
  time (these tests document the limitation and will fail once it is lifted),
- the static "dims as a topology path" encoding (§4.4) via mypy.
"""

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import reduction_order_mock as mock

import gt4py.next as gtx
from gt4py.next import neighbor_sum


HERE = pathlib.Path(__file__).parent

Vertex = gtx.Dimension("Vertex")
Edge = gtx.Dimension("Edge")
# note the string coupling: the local dim *must* be named like the offset
V2EDim = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)
E2VDim = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)
V2E = gtx.FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
E2V = gtx.FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))


def make_offset_provider():
    return {
        "V2E": gtx.as_connectivity(
            domain={Vertex: 4, V2EDim: 3},
            codomain=Edge,
            data=mock.V2E_TABLE,
            skip_value=mock.SKIP,
        ),
        "E2V": gtx.as_connectivity(
            domain={Edge: 5, E2VDim: 2}, codomain=Vertex, data=mock.E2V_TABLE
        ),
    }


# --- §2: reduction-order semantics on the reference mock ----------------------


def test_orders_agree_with_gathered_validity():
    ef = np.arange(1.0, 6.0)
    np.testing.assert_array_equal(mock.sum_innermost_first(ef), mock.sum_outer_first_gathered(ef))


def test_outer_first_without_dependency_information_is_wrong():
    ef = np.arange(1.0, 6.0)
    assert not np.array_equal(mock.sum_innermost_first(ef), mock.sum_outer_first_naive(ef))


def test_innermost_first_reference_values():
    # hand-checkable on the 5-edge mesh: result[e] = sum over e's two vertices
    # of the sum of the values of that vertex's incident edges (values = 1..5)
    ef = np.arange(1.0, 6.0)
    np.testing.assert_array_equal(mock.sum_innermost_first(ef), [16.0, 13.0, 13.0, 17.0, 17.0])


# --- status quo: current GT4Py rejects chains ----------------------------------


def test_single_hop_reduction_matches_mock():
    @gtx.field_operator
    def sum_edges(
        ef: gtx.Field[gtx.Dims[Edge], gtx.float64],
    ) -> gtx.Field[gtx.Dims[Vertex], gtx.float64]:
        return neighbor_sum(ef(V2E), axis=V2EDim)

    ef = gtx.as_field({Edge: 5}, np.arange(1.0, 6.0))
    out = gtx.zeros({Vertex: 4}, dtype=gtx.float64)
    sum_edges(ef, out=out, offset_provider=make_offset_provider())
    expected = np.where(mock.V2E_TABLE != mock.SKIP, np.arange(1.0, 6.0)[mock.V2E_TABLE], 0).sum(
        axis=1
    )
    np.testing.assert_array_equal(out.asnumpy(), expected)


def test_status_quo_chained_connectivities_rejected():
    """Documents the current limitation; starts failing once chains land."""
    with pytest.raises(Exception, match="more than one dimension with DimensionKind 'LOCAL'"):

        @gtx.field_operator
        def double_remap(
            ef: gtx.Field[gtx.Dims[Edge], gtx.float64],
        ) -> gtx.Field[gtx.Dims[Edge], gtx.float64]:
            return neighbor_sum(neighbor_sum(ef(V2E)(E2V), axis=V2EDim), axis=E2VDim)


# --- §4.4: static encoding (mypy as the oracle) ---------------------------------


@pytest.mark.parametrize("static_file", ["static_hops.py", "static_forest.py"])
def test_static_encodings(static_file):
    """static_hops.py: proposal A (path/LIFO); static_forest.py: proposal B (forest)."""
    typed_dimensions_dir = HERE.parent / "typed_dimensions"
    env = os.environ | {"MYPYPATH": str(typed_dimensions_dir)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(typed_dimensions_dir / "mypy.ini"),
            static_file,
        ],
        cwd=HERE,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"expected mypy to pass, got:\n{result.stdout}"
