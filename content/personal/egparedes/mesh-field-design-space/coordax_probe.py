"""coordax probe for the mesh-field design-space note (research appendix §4.2).

    uv run --no-project --with "coordax==0.2.8" --with "jax[cpu]" --with chex python coordax_probe.py

Last run: coordax 0.2.8 (source read at neuralgcm/coordax 765bb88). Expected output:

    gather dims: ('vertex', 'v2e_local')
    gathered values: [[10.0, 20.0, 30.0], [20.0, 30.0, 30.0]]
    different ticks: ValueError
    cell + face dims: ('x', 'x_face')
    class-valued dim: TypeError
"""

import jax.numpy as jnp
import numpy as np

import coordax as cx

# Gathering through a table = untag the gathered axis, map, and let the
# table field's named dimensions become the result's.
e = cx.field(jnp.array([10.0, 20.0, 30.0]), cx.SizedAxis("edge", 3))
v2e = cx.field(jnp.array([[0, 1, 2], [1, 2, -1]]), "vertex", "v2e_local")
g = cx.cmap(lambda x, i: x[i])(e.untag("edge"), v2e)
print("gather dims:", g.dims)
print("gathered values:", np.asarray(g.data).tolist())

# Coordinates are compared by value: offset ticks on the same name are rejected.
cells = cx.field(jnp.ones(4), cx.LabeledAxis("x", np.arange(4) + 0.5))
nodes = cx.field(jnp.ones(4), cx.LabeledAxis("x", np.arange(4.0)))
try:
    cells + nodes
except ValueError as ex:
    print("different ticks:", type(ex).__name__)

# ...but differently named dimensions broadcast silently.
faces = cx.field(jnp.ones(5), "x_face")
print("cell + face dims:", (cells + faces).dims)


# Dimension names must be strings.
class V: ...


try:
    cx.field(jnp.ones(2), V)
except TypeError as ex:
    print("class-valued dim:", type(ex).__name__)
