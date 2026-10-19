"""uxarray probe for the mesh-field design-space note (research appendix §4.3).

    uv run --no-project --with "uxarray==2026.9.1" python uxarray_probe.py

Last run: uxarray 2026.9.1 (source read at UXARRAY/uxarray c05cb5a). Expected output:

    connectivity dims: ('n_face', 'n_max_face_nodes') fill: -9223372036854775808
    shifted grids equal: False
    cross-mesh a + b keeps a's grid: True
    5-element n_face array on a 2-face grid accepted: True
    node -> face aggregation dims: ('n_face',)
    lon equal: True lat equal: False grids equal: True
    explicit one-based, standardized dtype/fill: [[1, 2, 3]] UGRID-normalized: [[0, 1, 2]]
    omitted origin, node zero unused: [[0, 1, 2]] UGRID-normalized: [[1, 2, 3]]

The last two assertions pin observed public-loader regressions, not desired
behaviour. The proposed loader contract in the note uses the UGRID-normalized
values, independently of dtype/fill conversion.
"""

import warnings

import numpy as np
import xarray as xr

import uxarray as ux
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

warnings.filterwarnings("ignore")

tri = np.array([[[0, 0], [10, 0], [0, 10]], [[10, 0], [10, 10], [0, 10]]], float)
g1 = ux.Grid.from_face_vertices(tri, latlon=True)
g2 = ux.Grid.from_face_vertices(tri + 5.0, latlon=True)
fnc = g1.face_node_connectivity
print("connectivity dims:", fnc.dims, "fill:", fnc.attrs.get("_FillValue"))
print("shifted grids equal:", g1 == g2)

# Every UxDataArray carries its grid, but plain arithmetic does not check it:
# the result silently takes the first operand's grid.
a = ux.UxDataArray(np.ones(2), dims=["n_face"], uxgrid=g1)
b = ux.UxDataArray(np.ones(2), dims=["n_face"], uxgrid=g2)
print("cross-mesh a + b keeps a's grid:", (a + b).uxgrid is g1)

# The field/grid dependency is not enforced on construction either.
x = ux.UxDataArray(np.ones(5), dims=["n_face"], uxgrid=g1)
print("5-element n_face array on a 2-face grid accepted:", x.sizes["n_face"] != g1.n_face)

# Topological aggregations pick the connectivity from the pair of locations.
d = ux.UxDataArray(np.ones(4), dims=["n_node"], uxgrid=g1)
print("node -> face aggregation dims:", d.topological_mean(destination="face").dims)

# Grid.__eq__ tests `not (lon.equals or lat.equals)`: a latitude-only change
# compares equal.
lat_shift = tri.copy()
lat_shift[..., 1] += 5.0
g3 = ux.Grid.from_face_vertices(lat_shift, latlon=True)
print(
    "lon equal:", g1.node_lon.equals(g3.node_lon),
    "lat equal:", g1.node_lat.equals(g3.node_lat),
    "grids equal:", g1 == g3,
)


def loader_dataset(dtype, attrs):
    """Minimal UGRID dataset; node zero exists but is not referenced."""
    return xr.Dataset(
        {
            "mesh": xr.DataArray(
                0,
                attrs={
                    "cf_role": "mesh_topology",
                    "topology_dimension": 2,
                    "node_coordinates": "lon lat",
                    "face_node_connectivity": "conn",
                },
            ),
            "lon": (
                "node", np.array([0.0, 10.0, 0.0, 10.0]),
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "lat": (
                "node", np.array([0.0, 0.0, 10.0, 10.0]),
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "conn": (("face", "slot"), np.array([[1, 2, 3]], dtype=dtype), attrs),
        }
    )


for label, ds, observed, normalized in [
    (
        "explicit one-based, standardized dtype/fill",
        loader_dataset(INT_DTYPE, {"_FillValue": INT_FILL_VALUE, "start_index": 1}),
        [[1, 2, 3]], [[0, 1, 2]],
    ),
    (
        "omitted origin, node zero unused",
        loader_dataset(np.int32, {}),
        [[0, 1, 2]], [[1, 2, 3]],
    ),
]:
    actual = ux.Grid.from_dataset(ds).face_node_connectivity.values.tolist()
    assert actual == observed, (label, actual, observed)
    print(label + ":", actual, "UGRID-normalized:", normalized)
