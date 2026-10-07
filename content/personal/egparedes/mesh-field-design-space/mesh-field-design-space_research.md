---
title: "Meshes, dimensions and fields — research: knowledge-base inputs and ecosystem survey"
author: egparedes
tags: [type-system, dimensions, mesh, fields, domain, connectivities, halos, boundary-conditions, distributed, ugrid, sgrid, cf-conventions, xarray, coordax, uxarray, prior-art]
created: 2026-10-06
status: draft
---

> **Appendix** to [[personal/egparedes/mesh-field-design-space/mesh-field-design-space|Meshes, dimensions and fields: three typed designs and how they layer]].
> The inputs the three proposals are built from: what each note of this knowledge
> base contributes, the ICON / ICON4Py constraints those notes record, how three
> Python libraries (xarray, coordax, uxarray) and three conventions (UGRID, SGRID,
> CF) model dimensions, meshes and fields, and where the idea that "a field should
> know its mesh" comes from. Library claims were checked against source and, where
> marked **(ran)**, by the probe scripts vendored next to this note.

## 1. Method and versions

- **Knowledge base**: every note touching dimensions, meshes, domains or fields was
  read in full (list in §2).
- **Libraries**, read at source and probed:
  - xarray `23c9dd1` (installed as 2026.9.0);
  - coordax `765bb88` (0.2.8);
  - uxarray `c05cb5a` (installed as 2026.9.1).

  Probes: [`xarray_probe.py`](xarray_probe.py), [`coordax_probe.py`](coordax_probe.py),
  [`uxarray_probe.py`](uxarray_probe.py). Each has its run command and expected
  output in its docstring.
- **Conventions**:
  - [UGRID 1.0](https://ugrid-conventions.github.io/ugrid-conventions/);
  - [SGRID 0.3](https://sgrid.github.io/sgrid/);
  - [CF-1.13](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.13/cf-conventions.html)
    (17 Dec 2025, the latest release), cited by section;
  - the CF data model paper, Hassell et al. 2017, *GMD* 10:4619–4646.

## 2. Knowledge-base inputs

| Note | What it contributes | Constraints and stances it records |
| --- | --- | --- |
| [[personal/egparedes/connectivities-as-types/connectivities-as-types\|Connectivities as types]] | Nominal dimension classes, identified by qualified name. `NeighborConnectivity[Domain, Codomain]` declarations with a nested `Local`. A table bound per call and checked at the boundary. `Staggered[D]` on a declared Cartesian axis. | Four strings that had to agree (tag, variable, local dimension, provider key). Counts and skip-value presence are mesh- and configuration-dependent, so they are taken from the bound table. |
| [[personal/egparedes/connectivities-as-types/connectivities-as-types_dependent-typing\|Dependent-typing appendix]] | The theoretical base: `Fin`, Π, Σ, refinements, evidence, enumeration, parameter vs index, validating at the boundary. | gt4py takes the "middle road": brands, validated constructors, checks at the boundary. |
| [[personal/havogt/dependent-local-dimensions/dependent-local-dimensions\|Dependent local dimensions]] | Chained connectivities. Path (LIFO) vs forest (leaf rule) typing of local dimensions; the forest is the true invariant. | Reduction order matters operationally (streaming, gathered masks). At most one live local dimension today. |
| [[personal/havogt/dimension-generic-fields/dimension-generic-fields\|Generic dimensions and staggering]] | Dimension `TypeVar`/`TypeVarTuple`; per-rank and per-position overloads for changing a dimension; `Dual[X]` via decorator overloads. | 4 generic vs 14 concrete operators on the shallow-water C-grid kernel: staggering without dimension genericity is a regression. |
| [[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos\|Mesh with first-class halos]] | A V/E/C vocabulary. Multi-hop relations as compositions that are kept materialized. Halo depth as a symbolic relation closure. Ownership triple plus execution classes. Exchange placement as forward dataflow. | Three causes of skip values: pentagons, LAM boundary, halo edge of reach. Neighbor order is load-bearing. "Keep the vertical out of the mesh". |
| [[personal/havogt/field-data-protocol/field-data-protocol\|Field data protocol]] | A field is a function over absolute, possibly infinite coordinates. The origin is private to the buffer. Support ⊇ domain. | One layer owns all domain arithmetic. |
| [[personal/havogt/boundary-condition-syntax/boundary-condition-syntax\|Boundary-condition syntax]] | Domains as conditions (`concat_where`); region literals `K[0]`, `K[1:nlev]`; restriction as decomposition. | ICON4Py horizontal conditions use named index symbols that mix boundary and halo bounds. `Domain.__bool__` must raise. |
| [[personal/havogt/scan-redesign/scan-redesign\|Scan redesign]] | Anchored vertical ranges (`KDim.start`/`KDim.stop`). Forward-anchored vertical domain inference. Static vertical windows as compile-time connectivities. | The vertical is never distributed. Half levels (n+1) are staggering. Bare negative indices are not allowed. |
| [[personal/havogt/dtype-generic-fields/dtype-generic-fields\|Dtype-generic fields]] | Declare statically, bind at call time, monomorphize at FOAST (two-level type theory). | Strict: exact match or error, no promotion. |
| [[personal/egparedes/discretization-independent-fd-syntax\|Discretization-independent FD syntax]] | Gather–weight–reduce normal form. Location typing over the de Rham complex. The mesh as a partial-evaluation input. Staggering as a declared family selector. | Only compatible staggerings (C, D) keep the structural guarantees. The vertical and limiters need escape hatches. "Halo exchange is never written by the user". |
| [[knowledge/software-engineering/principles\|Working principles]] | Deep modules, DRY, making implicit concepts explicit, bounded contexts and anticorruption layers. | Red flags: information leakage, buried rules, unexamined constraints. |

Related work still in open PRs at the time of writing:
- [GridTools/gt4py_knowledge#31](https://github.com/GridTools/gt4py_knowledge/pull/31):
  a default domain per program statement plus per-output restrictions. Its examples
  bound outputs with ICON4Py zone symbols such as `start_edge_nudging_level_2` and
  `end_edge_local`.
- [GridTools/gt4py_knowledge#25](https://github.com/GridTools/gt4py_knowledge/pull/25):
  JAX support for connectivities. It proposes a small, eagerly computed bounds
  descriptor per table, which is compile-time information about a runtime table.

## 3. ICON / ICON4Py constraints

From the ICON4Py surveys appended to the mesh and boundary-condition notes:

- **Closed set of horizontal offsets**:
  - 14 offsets, up to 4 hops (`C2E2C2E2C`, 9 neighbours).
  - 8 tables are read from the grid file (1-based, converted to 0-based); the multi-hop ones are derived.
  - Neighbour order is semantic: `icon_edge_order` exists because `compute_e_flx_avg` depends on it.
- **Zones and layout**:
  - `Zone` = `LATERAL_BOUNDARY..LEVEL_8`, `NUDGING`, `INTERIOR`, `HALO`, `HALO_LEVEL_2`, `LOCAL`.
    These are ports of ICON's `rl_start`/`rl_end` and `refin_ctrl`.
  - The local memory layout is `|lateral-boundary|nudging|interior|halo1|halo2|`.
  - LAM halos are not contiguous.
  - **One vocabulary covers both semantic boundary zones and decomposition zones.**
- **Halo lines**:
  - cells 2, vertices 2, edges 3; the third edge line exists only to make `C2E` complete;
  - multi-hop tables are incomplete in the outer halo;
  - granules compute into the halo by hand (`end_index(HALO_LEVEL_2)`);
  - exchanges (GHEX) are placed by hand.
- **Skip values**:
  - from pentagons (intrinsic arity), the LAM lateral boundary (semantic), and the edge of the halo (an artifact);
  - `keep_skip_values=False` rewrites tables to hide the artifact and cannot tell the three apart.
- **Vertical**:
  - full and half levels: `KHalfDim` is a separate dimension today;
  - n+1 interface quantities;
  - the vertical is never decomposed.
- **Outside V/E/C**: `LsqUnkDim` (a local axis with no table), `RBFDimension`, sparse
  coefficient fields read from the grid file, and dynamic vertical indirection.
- **Fortran granule path**: nproma-padded buffers, 1-based tables, start and end
  indices supplied by ICON, `halo_levels=None`.

## 4. Libraries

### 4.1 xarray

- **Dimension identity**: any hashable (`_Dim = Hashable`,
  `xarray/namedarray/_typing.py:72`). In practice it is a `str`, which netCDF requires.
  Classes work in memory **(ran)**.
- **Coordinates and indexes** are separate from the dimension name (the `Index`
  API, `xarray/core/indexes.py:39`). Arithmetic matches by name with a configurable
  join. The default `inner` join of two fields whose coordinate labels differ yields
  an **empty** result; `arithmetic_join="exact"` raises `AlignmentError` **(ran)**.
- **Connectivity** is plain data. `edge_field.isel(edge=v2e)` gathers, and the result
  takes the table's dimensions `('vertex', 'v2e_local')`, so the local dimension's
  name belongs to the table *value* **(ran)**.
- **No skip values**: `-1` wraps around to the last element; masking is manual
  **(ran)**.
- **No mesh, no staggering**: a cell field and a face field broadcast silently into
  an outer product **(ran)**.
- **No index-space origin**: positional indexing is always 0-based; `RangeIndex`
  is a label range.

### 4.2 coordax

- **Dimension identity**: `str` only. A class raises `TypeError` **(ran)**.
- **`Coordinate` objects**:
  - frozen, hashable, registered as *static* JAX pytree nodes, so they are part of
    the trace and compilation cache key (`coordax/coordinate_systems.py:79`);
  - `SizedAxis` checks the size, `LabeledAxis` checks tick content via `ArrayKey`
    (hash of the bytes);
  - custom subclasses can model grids, and a `CartesianProduct` combines axes.
- **Exact alignment**: `cmap` rejects two fields that use one name with unequal
  coordinates (`coordax/fields.py`, `_cmap_with_doc`). Offset ticks on the same name
  therefore raise **(ran)**, but differently named dimensions broadcast silently
  **(ran)**.
- **Connectivity**: untag the gathered axis and `cmap` a take. The result takes the
  table field's dimensions **(ran)**. There are no skip values and no ragged rows.
- **The mesh as an index**: a table stored *in* a `Coordinate` would be static, so it
  would brand fields by content and compile once per mesh. This is §7's "index"
  option, the opposite of gt4py's table-as-parameter.

### 4.3 uxarray

- **The mesh is `Grid`** (`uxarray/grid/grid.py:116`). It wraps a UGRID-encoded
  `xr.Dataset` with lazily derived connectivities (12 properties named by entity
  pair), geometry, and subsetting with `inverse_indices`. Readers exist for UGRID,
  MPAS, ICON, ESMF, Exodus, SCRIP, FESOM2 and HEALPix.
- **A fixed vocabulary removes the string-agreement problem**
  (`uxarray/conventions/ugrid.py`, `DIM_NAMES`):
  - entities are `n_node`/`n_edge`/`n_face`;
  - every connectivity owns its local dimension name, except `two`, which
    `edge_node_connectivity` and `edge_face_connectivity` share. That conflates two
    different slot spaces.
  - A field's location is inferred from its dimension name (`data_mapping`).
- **Loader normalization**: on load, `_FillValue` becomes one sentinel
  (`INT_FILL_VALUE`, the minimum of `intp`) and `start_index` is rebased to 0
  (`uxarray/io/_ugrid.py:206-252`). `Grid.validate()` is opt-in and never called
  internally.
- **Ragged degree**: topological aggregations partition faces by their true node
  count (`uxarray/core/aggregation.py`, `get_face_node_partitions`). They pick the
  connectivity from the pair of locations: `topological_mean(destination="face")`
  **(ran)**.
- **Fields know their mesh**: `UxDataArray.uxgrid` (`uxarray/core/dataarray.py:62-187`),
  propagated by `_copy`/`_replace`. However:
  - grid equality is checked only by `curl`, `divergence`, `scalardotgradient` and
    `concat` (`dataarray.py:1688, 1799, 1907`; `uxarray/core/api.py:651`);
  - plain `a + b` across two meshes silently keeps `a`'s grid **(ran)**;
  - a 5-element `n_face` array attaches to a 2-face grid without complaint **(ran)**.
- **Grid identity**: `is`, otherwise content equality of the node coordinates and
  `face_node_connectivity` (`Grid.__eq__`, `grid.py:791`). The test is written
  `not (lon.equals or lat.equals)`, so a latitude-only change compares **equal**
  **(ran)**.
- **Dual mesh** (`use_dual`, `uxarray/grid/dual.py`) in the full primal/dual sense,
  i.e. the Hodge-dual meaning of "dual".

### 4.4 Summary

| | xarray | coordax | uxarray | gt4py (connectivities as types) |
| --- | --- | --- | --- | --- |
| Dimension identity | hashable (usually `str`) | `str` | fixed UGRID names | nominal class, qualified name |
| Connectivity | data; the result takes the table's dimensions | data, via `cmap` | 12 lazily derived grid properties | declaration class plus a table bound per call |
| Local dimension owned by | the table value | the table value | the connectivity (fixed name), except `two` | the declaration (`V2E.Local`) |
| Skip values | none | none | one fill sentinel; true ragged degree in aggregations | `skip_value`, `max_neighbors`/`min_neighbors` |
| Mesh object | none | none (a `Coordinate` could act as one) | `Grid` | none (the provider dict) |
| A field knows its mesh | no | per-axis coordinates | `.uxgrid`, rarely checked | no |
| Static checking | none | none (trace time) | none | mypy and pyright |
| Staggering | none | offset ticks | primal/dual meshes | `Staggered[D]` |

## 5. Conventions

### 5.1 UGRID and SGRID

- **UGRID** describes a mesh as a placeholder variable (`cf_role = "mesh_topology"`)
  whose attributes name connectivity arrays by entity pair
  (`face_node_connectivity`, …):
  - ragged rows are padded with `_FillValue`;
  - `start_index` ∈ {0, 1};
  - a data variable carries `mesh` and `location`, or `location_index_set` for data
    on a subset;
  - face nodes *should* be anticlockwise.
- **SGRID**: structured topology with `node_dimensions`, `face_dimensions` and
  `padding` ∈ {none, low, high, both}. A data variable carries `grid` and `location`
  (`node`, `edge1`, `edge2`, `face`).
- **Locations**: UGRID locations are cell **degrees**; SGRID locations are per-axis
  **bit vectors**. The split follows whether the complex is a product or not
  (see [[personal/egparedes/connectivities-as-types/connectivities-as-types_conventions|the conventions appendix]]).

### 5.2 CF-1.13

- **Dimensions vs coordinates**:
  - a netCDF dimension is an index space (§2.4);
  - a coordinate variable shares its dimension's name;
  - auxiliary coordinates are attached through the `coordinates` attribute (§5).
- **The vertical** is recognized by pressure units or a `positive` attribute
  (§4.3); `axis="Z"` is optional.
  - Parametric vertical coordinates use `formula_terms`.
  - CF has **no** full/half-level concept. The CF-native link between levels and
    interfaces is the bounds of the level coordinate (§7.1.4, Example 7.3).
    Equating the two is an inference, not CF text.
- **Cells**:
  - `bounds` add a trailing vertex dimension (§7.1);
  - `cell_measures` (`area`/`volume`, §7.2);
  - `cell_methods` (§7.3) say whether a value is a point value (`point`) or a cell
    aggregate (`area: mean`); methods apply in order.
- **Domain construct** (§5.8, since CF-1.9): "A domain describes data locations and
  cell properties." A domain variable describes a domain "in the absence of any data
  values". In the data model (Appendix I), a **field is a data array plus a
  domain**, and the domain is "the only metadata construct that may also exist
  independently of a field construct".
- **Meshes in CF** (§1.6 and §5.9, since CF-1.11): UGRID 1.0 is incorporated by
  reference.
  - A data or domain variable binds to a mesh by reference, through `mesh` plus
    `location` (or `location_index_set`).
  - Otherwise, binding to a domain is structural: shared dimension names plus
    attributes.
  - Appendix I: whether two domains share a mesh "may be reliably determined by
    inspection". **Domain identity in CF is structural, not nominal.**
- **Staggering**: CF-1.13 never mentions SGRID or staggering. UGRID itself notes
  that CF offers no specific support for staggered data.
- **Ragged data**:
  - compression by gathering (§8.2);
  - contiguous and indexed ragged arrays for discrete sampling geometries (§9.3).
  - UGRID connectivities nevertheless use padded dense arrays.

## 6. "Every field should know its mesh"

### 6.1 Where the idea appears

**In this knowledge base:**
- [[personal/egparedes/connectivities-as-types/connectivities-as-types_conventions|Conventions appendix]]
  §5: stated outright, with UGRID/SGRID's per-variable `mesh=`/`grid=` as the precedent.
- [[personal/egparedes/connectivities-as-types/connectivities-as-types|Connectivities as types]],
  *Binding model* ("one table per declaration per binding context") and open
  question 2 (the multi-table case): the design choice that leaves a field without
  a mesh, and its consequence.
- [[personal/egparedes/connectivities-as-types/connectivities-as-types_dependent-typing|Dependent-typing appendix]]
  §7 (parameter or index, brands) and "Same type ⇒ same table" (mixing meshes is out
  of scope and would need a mesh-level brand).
- [[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos|Mesh proposal]]:
  "each field carries a per-entity valid-depth" and the open question about a
  "mesh-aware field".

**In the conventions and libraries:**
- UGRID `mesh` + `location`, SGRID `grid`, CF-1.11 §5.9;
- uxarray `.uxgrid`;
- coordax `Field.axes`, whose coordinates can carry a mesh;
- xarray: no equivalent.

### 6.2 Why each system wants it

| System | Rationale |
| --- | --- |
| UGRID / SGRID / CF | Interpreting files: a file may hold several meshes, and an `n_face` index means nothing without its topology. |
| uxarray | Convenience: methods on a field (integration, gradients, aggregations, plotting, regridding) need geometry and topology without being passed them. |
| coordax | Safety: exact alignment is checked at trace time. |
| Conventions appendix §5 | Telling tables apart when there are several meshes or rewritten tables. |
| Mesh proposal | Per-field halo-validity state for automatic exchange placement. |
| Dependent-typing appendix | Preventing a field computed on mesh A from being consumed on mesh B. |

### 6.3 What a mesh is in each system

| System | The mesh is | Two meshes are the same when | A field is tied to it by |
| --- | --- | --- | --- |
| UGRID | a placeholder variable naming connectivity arrays | same variable name | `mesh="Mesh2"` |
| CF data model | a domain construct (plus a domain topology since CF-1.11) | inspection (structural) | structural binding, or `mesh` reference |
| coordax | a frozen, hashable `Coordinate` | value equality (part of the cache key) | `Field.axes` |
| uxarray | `Grid`: UGRID dataset plus derived connectivities and geometry, mutable | `is`, else content equality | `.uxgrid` (rarely checked) |
| gt4py today | the `offset_provider` dict | — | nothing |

### 6.4 Dependent-typing reading

"A field knows its mesh" moves `g` from a *parameter* of the call to an *index*
carried with the field (appendix §7):
- **uxarray**: a `UxDataArray` is literally a Σ pair `(Grid g, data over Fin<g.n_face>)`.
  Neither half of the dependency is enforced (sizes, arithmetic).
- **coordax**: makes the mesh a static index checked at trace time, at the cost of
  one compilation per mesh.
- **UGRID**: makes it an index *by name*, the unbranded case.
- **gt4py today**: keeps it a parameter. That is what lets one compiled program serve
  many meshes, and why mixing meshes goes undetected.

The proposals in the main note sit on this spectrum: A keeps a parameter plus a
runtime brand; B and C carry the logical mesh with every field.

## 7. Precedents for global vs local semantics

- **Global semantics**:
  - PSyclone/LFRic: the algorithm layer is written as if serial; the PSy layer
    generates loop bounds and halo exchanges.
  - Legion/Regent: partitions with set algebra.
  - Chapel's global-view arrays: the halo is a cached "fluff" overlay.
  - The FD-syntax note's surface ("halo exchange is never written by the user").
- **Local semantics (SPMD)**:
  - MPI codes generally;
  - ICON (`sync_patch_array`);
  - Atlas (`haloExchange` on a partition's function space);
  - PETSc (`DMGlobalToLocal` scatters a global vector into a ghosted local one);
  - FMS/NEMO compute vs data domains;
  - gt4py and ICON4Py today: gt4py programs see local connectivities, and ICON4Py
    places exchanges.
