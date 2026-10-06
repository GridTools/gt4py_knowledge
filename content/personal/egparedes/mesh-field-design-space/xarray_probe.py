"""xarray probe for the mesh-field design-space note (research appendix §4.1).

Run from a directory that does not contain an `xarray/` source checkout:

    uv run --no-project --with xarray --with numpy python xarray_probe.py

Last run: xarray 2026.9.0. Expected output:

    gather dims: ('vertex', 'v2e_local')
    gathered values: [[10.0, 20.0, 30.0], [20.0, 30.0, 30.0]]
    masked sum: [60.0, 50.0]
    class-valued dims: (<class '__main__.E'>, <class '__main__.V'>)
    cell + face dims: ('x', 'x_face') shape: (4, 5)
    same name, different ticks, default join size: 0
    same name, different ticks, exact join: AlignmentError
"""

import numpy as np
import xarray as xr

# A neighbor table is plain data: gathering through it is vectorized indexing,
# and the result takes its dimensions from the table (the local dimension's
# name belongs to the table value). -1 is not a skip value: it wraps around.
e = xr.DataArray(np.array([10.0, 20.0, 30.0]), dims="edge")
v2e = xr.DataArray(np.array([[0, 1, 2], [1, 2, -1]]), dims=("vertex", "v2e_local"))
g = e.isel(edge=v2e)
print("gather dims:", g.dims)
print("gathered values:", g.values.tolist())
print("masked sum:", g.where(v2e != -1).sum("v2e_local").values.tolist())


# Dimensions are any hashable, so classes work in memory; distinct dimensions
# broadcast silently into an outer product.
class V: ...


class E: ...


a = xr.DataArray(np.ones(3), dims=(E,))
b = xr.DataArray(np.ones(2), dims=(V,))
print("class-valued dims:", (a + b).dims)

# Staggering by name: a cell field and a face field broadcast into 2-D.
c = xr.DataArray(np.ones(4), dims="x")
f = xr.DataArray(np.ones(5), dims="x_face")
print("cell + face dims:", (c + f).dims, "shape:", (c + f).shape)

# Staggering by ticks: the default inner join silently yields an empty result.
c1 = xr.DataArray(np.ones(4), dims="x", coords={"x": [0.5, 1.5, 2.5, 3.5]})
c2 = xr.DataArray(np.ones(4), dims="x", coords={"x": [0.0, 1.0, 2.0, 3.0]})
print("same name, different ticks, default join size:", (c1 + c2).sizes["x"])
with xr.set_options(arithmetic_join="exact"):
    try:
        c1 + c2
    except ValueError as ex:
        print("same name, different ticks, exact join:", type(ex).__name__)
