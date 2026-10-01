---
title: "Connectivities as types — alignment with the UGRID and SGRID conventions"
author: egparedes
tags: [type-system, dimensions, local-dimensions, connectivities, unstructured, staggering, cw-complex, domain, ugrid, sgrid, conventions, interop, halos, prior-art]
created: 2026-10-01
status: draft
---

> **Appendix** to [[personal/egparedes/connectivities-as-types/connectivities-as-types|Connectivities as types]].
> An audit of this design against the two established community conventions for
> describing mesh topology and grid staggering in netCDF/CF data: **UGRID** for
> unstructured meshes and **SGRID** for structured staggered grids. It records
> where the design already agrees with them, the one place a convention is
> sharper than this note, and the concrete enhancements that follow. Some
> findings land on
> [[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos|a mesh concept with
> first-class halos]] and
> [[personal/havogt/field-data-protocol/field-data-protocol|the field data protocol]]
> rather than here; they are listed in [For the mesh and field-data
> proposals](#5-for-the-mesh-and-field-data-proposals) rather than edited into
> those notes.

Sources: [UGRID conventions](https://ugrid-conventions.github.io/ugrid-conventions/)
and [SGRID conventions](https://sgrid.github.io/sgrid/) (v0.3), read 2026-10-01.

**A caution before anything else.** Both are netCDF *interchange* conventions.
A location is a string attribute on a data variable and a mesh is a dummy
variable referenced by name — the exact pattern this proposal replaces with
types. What transfers is the **vocabulary and the invariants**, not the
mechanism. Nothing below suggests adopting netCDF attributes into the DSL.

## 1. The two conventions, structurally

**UGRID.** A dummy variable with `cf_role = "mesh_topology"`,
`topology_dimension` ∈ {1, 2, 3}, `node_coordinates`, and connectivity arrays
named by *entity pair*: `edge_node_connectivity` (required for 1D),
`face_node_connectivity` (2D), `volume_node_connectivity` (3D), plus optional
`face_edge_connectivity`, `face_face_connectivity`, `edge_face_connectivity`,
`boundary_node_connectivity`. Ragged connectivity is padded to a maximum
(`nMaxMesh2_face_nodes`) with a required `_FillValue`; each connectivity carries
`start_index` ∈ {0, 1}. A data variable carries `mesh = "Mesh2"` and
`location` ∈ {`node`, `edge`, `face`, `volume`}, or `location_index_set` for
data on a subset. Face corner nodes *should* be ordered anticlockwise as viewed
from above — a recommendation, not a requirement.

**SGRID.** A variable with `cf_role = "grid_topology"`, `topology_dimension` ∈
{2, 3}, `node_dimensions`, and `face_dimensions` (2D) or `volume_dimensions`
(3D); optionally `edge1_dimensions`, `edge2_dimensions`, `vertical_dimensions`,
and per-location coordinate attributes. A data variable carries
`grid = "MyGrid"` and `location` ∈ {`node`, `edge1`, `edge2`, `face`} in 2D, or
eight values in 3D. 1D is not specified, and the spec states the 3D option
"should not be used for layered grids, such as typical ocean and atmosphere
models. Use the 2d grid with vertical dimensions instead."

### The structural observation

**UGRID's locations are cell degrees; SGRID's are per-axis bit vectors.**

| | locations | reading |
| --- | --- | --- |
| UGRID | `node`, `edge`, `face`, `volume` | degree 0, 1, 2, 3 |
| SGRID 2D | `node`, `edge1`, `edge2`, `face` | 2² bit vectors over the two axes |
| SGRID 3D | `node`, `edge1`–`edge3`, `face1`–`face3`, `volume` | 2³ bit vectors |

Two standards bodies, working independently, split exactly along the
product-versus-non-product line that
[[personal/egparedes/connectivities-as-types/connectivities-as-types#cartesian-axis-dimensions|Cartesian axis dimensions]]
derives from the CW complex. SGRID **must** distinguish `edge1` from `edge2`
because a product complex has two edge families per 2D cell; UGRID **cannot**,
because an unstructured complex does not factor and only the degree survives.
That is the note's claim arriving as external evidence, and it is the reason the
two cases are encoded differently here: `Staggered[D]` per axis for the product
case, separate dimension classes for mesh locations.

On encoding, this design is ahead: `Dims[Staggered[I], J]` *derives* what SGRID
enumerates as 2ⁿ location names, and it does not stop at three axes.

## 2. Where the design already agrees

| Convention feature | This design | Verdict |
| --- | --- | --- |
| Connectivities named by entity pair (`face_node_connectivity`) | `NeighborConnectivity[Domain, Codomain]` — `face_node_connectivity` ≡ `NeighborConnectivity[Face, Node]` | isomorphic. The `Domain`/`Codomain` naming matches UGRID's reading direction, where `source`/`target` did not |
| `_FillValue` required wherever an entry may be absent — ragged valence, but also a missing boundary neighbour in `edge_face_connectivity` / `face_face_connectivity`; `nMaxMesh2_face_nodes` | `skip_value`, `max_neighbors`, `min_neighbors` on `NeighborTableType` | strictly more informative. ICON's icosahedron carries 12 pentagons among hexagons, so `V2E` is `min_neighbors=5, max_neighbors=6`; UGRID can only record max 6 plus fill. That is the sketch's example verbatim |
| `vertical_dimensions` uses **the same** padding syntax as `face_dimensions` | — | the vertical is just another axis pair. SGRID needs no "vertical kind", which supports the conclusion that `DimensionKind.VERTICAL` is a layout and scan artefact, not geometry (main note, open question 6) |
| `face_dimensions` specifies a padding **per axis** | a per-axis alignment, never grid-wide | supports the shape of the alignment follow-up (open question 5): a class keyword on the axis |
| 3D topology forbidden for layered grids; use 2D + `vertical_dimensions` | horizontal topology plus a separate vertical axis | validates the existing split |
| `topology_dimension` gates which locations exist | — | see [§4(a)](#a-take-ugrids-location-vocabulary-for-locationindex-and-move-degree-there) |

## 3. SGRID padding versus "extents are declared, never derived"

This is the sharpest interaction. Verbatim from the SGRID spec:

> the padding type may be one of the four literal strings: "none", "low",
> "high", or "both" depending on whether the face_dimension is one shorter than
> the corresponding node_dimension (padding:none), one longer than the
> corresponding node_dimension (padding:both), or of equal length with one extra
> value stored on the low or high end of the dimension

The format is `face_dimension1: node_dimension1 (padding: type1) …`, and
`layer_dimension: layer_interface_dimension (padding: type)` — derived dimension
first, node dimension second. Decoding each value into array positions (netCDF
dimension indices are 0-based; see the qualification under (i)) with `n` the node count:

| padding | cells | cell `j` spans nodes | cell `j` at node-position | as one absolute `UnitRange` under ADR 0026 |
| --- | --- | --- | --- | --- |
| `none` | n−1 | (j, j+1) | j + ½ | `[1, n)` |
| `low` | n | (j−1, j) | j − ½ | `[0, n)` |
| `high` | n | (j, j+1) | j + ½ | `[1, n+1)` |
| `both` | n+1 | (j−1, j) | j − ½ | `[0, n+1)` |

Three conclusions, and they do not all point the same way.

**(i) The decision that extents are declared rather than derived is vindicated,
and the range model is the more expressive of the two.** Each of the four values
is one absolute `UnitRange` under a *single* alignment convention. The range model
also reaches arbitrary halo depth, which `padding` cannot express at all, and had
the design derived `Staggered[D]`'s extent from `D`'s it would have been wrong for
three of SGRID's four cases.

Two qualifications on "more expressive". The `UnitRange` column above is a correct
*relative* decoding; its absolute origin is a choice, because SGRID does have a
mechanism for absolute numbering that `padding` alone does not show — integer
coordinate variables (`face(face)`, `node(node)`), which the Delft3D example uses
to number layer interfaces `0 … KMAX` while everything else counts from 1. So
netCDF *dimensions* are 0-based but SGRID *labels* need not be. What a `UnitRange`
still adds is that the origin travels with the dimension in one object rather than
in a separate coordinate variable, and that halo depth is unbounded.

**(ii) `padding: low` is this note's halo remark, standardized.** The main note
observes that `Staggered[I](0)` "is the first cell outside the complex — exactly
where a halo cell goes". That is `low`: equal length, one extra value stored at
the low end.

**(iii) Both alignments are standardized and in production, which strengthens
the alignment follow-up.** `low` and `both` place cell `j` at `j − ½`, which is
ADR 0026's convention; `none` and `high` place it at `j + ½`, the other one. The
`padding` attribute exists *because* real codes use both — ROMS, Delft3D and
WRF-ARW are the spec's own examples. Note the asymmetry this exposes: absolute
ranges give the right *point set* under either convention but not matching
*labels*, which is why parameterizing the alignment is about porting and
validation fidelity rather than expressiveness.

## 4. Enhancements for this proposal

### (a) Take UGRID's location vocabulary for `LocationIndex`, and move `degree` there

Open question 8 proposes a `LocationIndex` level without saying what it would
contain. UGRID supplies it: `Node`, `Edge`, `Face`, `Volume`, gated by
`topology_dimension`. The non-obvious part is where the cell degree belongs:

- For a **mesh location**, the degree is **canonical** — UGRID's four location
  names *are* degrees 0–3, and `topology_dimension` is the maximum present.
- For a **Cartesian axis**, the degree is **declarational** — nothing in the
  types says whether `KDim` indexes layers or interfaces, and SGRID does not
  record a degree either; it records `padding` instead.

So the static `degree` of open question 7, rejected there for a Cartesian axis,
belongs on `LocationIndex` instead. That redirects the question rather than
answering it, and it gives `LocationIndex` content worth having: a typed
`Node`/`Edge`/`Face`/`Volume` vocabulary with degrees makes a UGRID mesh bind by
construction, and makes "a primary, non-local dimension" expressible.

### (b) Validate a bound table against the codomain's range, not against `[0, size)`

UGRID standardizes `start_index` ∈ {0, 1} on every connectivity array, and
ICON4Py's tables come from Fortran. `NeighborTableType` carries `dtype`,
`skip_value` and `max_neighbors` but no index origin. Since a codomain dimension
has an absolute range, `check_neighbor_table` can validate entries against
`codomain`'s `UnitRange` rather than against `[0, size)`, and a 1-based table is
then simply a codomain with range `[1, n+1)`. That makes `start_index` a
non-issue at the cost of one comparison in one function, and it is the same
absolute-range argument as §3(i).

### (c) Record the orientation guarantee so incidence signs become derivable

UGRID says face corner nodes "should be specified in anticlockwise (also referred
to as counterclockwise) direction as viewed from above". Where that holds, the
incidence signs — the `d` of `div = ⋆d⋆` — are **derivable from
`face_node_connectivity`** instead of supplied. Note the modal verb: the
convention *recommends* the ordering rather than mandating it, so a consumer
cannot assume it — which is exactly why a gt4py-side declaration would have to
record the guarantee rather than infer it. The main note explains why
the Cartesian side needs no orientation data (a product of intervals is
canonically oriented per axis) and points at ICON's `geofac_div` for the
unstructured side, where the signs are materialized. UGRID shows the
unstructured side can be canonical too if a declaration records the ordering
guarantee — an `orientation=` marker on a `C2V`-shaped connectivity would let
[[personal/egparedes/discretization-independent-fd-syntax|the surface-syntax note]]
compute mimetic weights rather than receive them. Out of scope for the PR stack;
it is the cheapest bridge between the two notes.

### (d) External evidence for open question 6

UGRID had to add `face_dimension` and `edge_dimension` attributes purely to
disambiguate which netCDF dimension indexes the mesh element, because a global
ordering rule proved insufficient. The mechanisms differ — UGRID has to *record*
which stored axis is the element axis, whereas F4 *derives* layout from dimension
names through `order_dimensions` — but the premise that failed is the same in
both: that one global rule can fix dimension order. It supports the conclusion
that layout belongs to the field or the grid.

## 5. For the mesh and field-data proposals

Two findings land outside this note. They are recorded here rather than edited
into another contributor's working area.

**Fields on a non-contiguous subset of a location.** UGRID's
`location_index_set` (a variable with `cf_role = "location_index_set"`, plus
`mesh` and `location`) exists exactly for data defined on a subset of an entity
type. A gt4py `Domain` is a dense `UnitRange`, so this is inexpressible — yet
ICON needs it: owned versus halo entities, and per-region index lists. Neither
[[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos|the mesh proposal]]
nor
[[personal/havogt/field-data-protocol/field-data-protocol|the field data protocol]]
names it as a requirement, and the halo proposal's ownership metadata is the
natural home.

**A field should know its mesh.** Both conventions put `mesh=` / `grid=` on
*every* data variable. A gt4py field knows its dimensions but not its mesh,
which is precisely the unresolved multi-table case of the main note's open
question 2 — halo variants, rewritten `keep_skip_values` tables, several meshes
in one process. The conventions' answer is uniform: carry the grid identity with
the data, not only in a side dict.

## 6. The practical payoff

Aligning on vocabulary rather than mechanism makes a UGRID/SGRID reader on the
`gtx` side straightforward: entity classes and connectivity declarations come
from the convention, `start_index` resolves into the codomain range (b), ragged
valence into `min_neighbors`/`max_neighbors`/`skip_value`, SGRID `padding` into
the two members' absolute ranges (§3), and SGRID `location` into a `Dims[...]`
bit vector. That is downstream of the whole PR stack and is a goal, not a
deliverable of it.
