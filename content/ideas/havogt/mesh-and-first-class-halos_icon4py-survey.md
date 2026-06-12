---
title: "icon4py constraint survey for a mesh concept"
author: havogt
tags: [mesh, halos, unstructured, connectivities, offset-provider, icon4py, distributed, halo-exchange, research]
created: 2026-06-12
---

> **Status**: research appendix to
> [[ideas/havogt/mesh-and-first-class-halos|A mesh concept with first-class halos]].
> This is the in-repo constraint set a gt4py mesh concept must satisfy — the
> detailed version of §2.3 of the
> [[ideas/havogt/mesh-and-first-class-halos_research|prior-art survey]].

# Survey of icon4py for a gt4py "mesh" concept design

All paths are relative to the `icon4py` repository root, inspected at the checkout
current as of June 2026 (`model/common` at the head of the repo).

---

## 1. Connectivity inventory

**Single source of truth**: `model/common/src/icon4py/model/common/dimension.py`. All local dimensions are lines 18–32, all `FieldOffset`s lines 33–48. There are **no other `FieldOffset(` definitions anywhere** in model/, tools/, or bindings/ (verified by grep). No flattened sparse dims (CEDim/ECDim/ECVDim/CECDim) exist anymore.

Naming convention: `A2B` = from A-located field iteration to B-located data; `O` suffix = origin element prepended at sparse position 0.

| Offset | def (dimension.py) | target (iterated) | source (accessed) | max nbh | "O" | multi-hop | skip values |
|---|---|---|---|---|---|---|---|
| `C2E` | :34 | Cell | Edge | 3 | no | no | never |
| `C2V` | :36 | Cell | Vertex | 3 | no | no | never |
| `E2C` | :33 | Edge | Cell | 2 | no | no | LAM-boundary / distributed |
| `E2V` | :38 | Edge | Vertex | 2 | no | no | never |
| `V2E` | :37 | Vertex | Edge | 6 | no | no | **pentagons** (+LAM/halo) |
| `V2C` | :35 | Vertex | Cell | 6 | no | no | **pentagons** (+LAM/halo) |
| `V2E2V` | :46 | Vertex | Vertex | 6 | no | 2-hop | **pentagons** (+LAM/halo) |
| `C2E2C` | :43 | Cell | Cell | 3 | no | 2-hop | LAM/distributed |
| `C2E2CO` | :40 | Cell | Cell | 4 | **yes** | 2-hop | LAM/distributed |
| `E2C2V` | :39 | Edge | Vertex | 4 | no | 2-hop ("diamond") | LAM/distributed |
| `E2C2E` | :42 | Edge | Edge | 4 | no | 2-hop ("diamond") | LAM/distributed |
| `E2C2EO` | :41 | Edge | Edge | 5 | **yes** | 2-hop | LAM/distributed |
| `C2E2C2E` | :44 | Cell | Edge | 9 | no | 3-hop ("triangle edges") | LAM/distributed |
| `C2E2C2E2C` | :45 | Cell | Cell | 9 | no | 4-hop ("butterfly") | LAM/distributed |
| `Koff` | :47 | K | K | — | — | vertical offset (`Dimension` provider, not a table) | — |
| `KHalfOff` | :48 | KHalf | KHalf | — | — | vertical | — |

Also defined but **not a connectivity**: `LsqUnkDim` (:18, LOCAL kind) — number of least-squares unknowns; used as an extra axis of a 3-D coefficient field `domain=(CellDim, LsqUnkDim, C2E2CDim)` in `model/common/src/icon4py/model/common/interpolation/interpolation_factory.py:305`.

**Skip-value rules** are centralized in `model/common/src/icon4py/model/common/grid/icon.py`:

```python
CONNECTIVITIES_ON_BOUNDARIES = (C2E2C2EDim, E2CDim, C2E2CDim, C2E2CODim,
                                E2C2VDim, E2C2EDim, E2C2EODim, C2E2C2E2CDim)   # :34-43
CONNECTIVITIES_ON_PENTAGONS = (V2EDim, V2CDim, V2E2VDim)                       # :44
```
`_has_skip_values(offset, limited_area_or_distributed)` (icon.py:130): pentagon offsets *always* have skip values (12 pentagons on the icosahedral global grid); boundary offsets only for limited-area or distributed grids. `skip_value=-1` passed to `gtx.as_connectivity` accordingly (icon.py:190). There is an opt-out `keep_skip_values=False` (`GridConfig`, base.py:46) that **rewrites the tables**: `base._replace_skip_values` (base.py:180) replaces `-1` by the row max — explicitly documented as *"a workaround to account for the lack of a domain inference for unstructured neighbor accesses in GT4Py: when computing into temporaries we do not use the minimal domain (interior + respective halo), but the full domain"*. The Fortran-bindings path always sets `keep_skip_values=False` (bindings/common.py:254).

**Where tables come from** (`grid_manager.py`):
- Read from the ICON netCDF grid file (`gridfile.py` `ConnectivityName`, :198–228): `C2E2C`("neighbor_cell_index"), `V2E2V`("vertices_of_vertex"), `V2E`, `V2C`, `E2V`, `C2V`, `E2C`, `C2E`. Fixed sparse sizes from file dims `nv=3, ne=6, nc=2, no=4` (gridfile.py:156-180).
- **Derived** in `grid_manager._get_derived_connectivities` (:543–576): `E2C2V` via `_construct_diamond_vertices` (:579), `E2C2E` via `_construct_diamond_edges` (:648, with an `icon_edge_order` permutation to match ICON ordering — `compute_e_flx_avg` depends on it), `E2C2EO`/`C2E2CO` via `column_stack((arange, table))`, `C2E2C2E` via `_construct_triangle_edges` (:696), `C2E2C2E2C` via `_construct_butterfly_cells` (:728). Skip values handled via `_patch_with_dummy_lastline` (-1 wraps to a dummy row).

---

## 2. Grid / mesh classes

**`base.Grid`** (`model/common/src/icon4py/model/common/grid/base.py:65`) — frozen dataclass:

```python
id: str                                       # uuidOfHGrid from grid file
config: GridConfig
connectivities: gtx_common.OffsetProvider     # dict[str, NeighborTable|Dimension]
start_index: Callable[[h_grid.Domain], gtx.int32]
end_index:   Callable[[h_grid.Domain], gtx.int32]
```
`__post_init__` injects `connectivities["Koff"] = dims.KDim` (base.py:92, with a TODO(havogt) to replace `Koff[k]` by `KDim + k`). `size` (cached property) is *derived from connectivity shapes* plus configured num_{cells,edges,vertices,levels}. The connectivities dict **is** the gt4py `offset_provider`: granules call programs with `offset_provider=self._grid.connectivities` (e.g. diffusion.py:463 and ~30 more sites).

**`IconGrid`** (icon.py:118) adds `grid_params` (icosahedron root/level/radius or torus length/height; GeometryType enum matches `mo_grid_geometry_info.f90`) and `refinement_control: dict[Dimension, Field]`.

**`GridManager`** (`grid_manager.py:52`) reads the netCDF grid file: connectivities (Fortran 1-based → 0-based via `ToZeroBasedIndexTransformation`), refinement-control fields (`refin_ctl` per dim), coordinates, geometry fields. `_construct_decomposed_grid` (:388) pipeline: read global tables → `decomposer(cell_to_cell_neighbors, comm_size)` gives cell→rank map → `halo.get_halo_constructor(...)` builds `DecompositionInfo` → `_get_local_connectivities` slices global tables by `decomposition_info.global_index(...)` and renumbers via `halo.global_to_local` (halo.py:503, missing entries become `-1`) → derived connectivities computed *on the local (incl. halo) tables* → start/end indices computed from refinement + decomposition (`grid_refinement.compute_domain_bounds`). Limited-area + distributed is currently `NotImplementedError` (grid_manager.py:414).

**`GridGeometry`** (geometry.py:39) — a `factory.FieldSource` that computes geometry fields with stencils; field providers implement a `NeedsExchange` protocol (`states/factory.py:100`) so freshly computed fields get a halo exchange (geometry.py:831).

**start/end index & domain zones** (`horizontal.py`): `Domain = (dim, Zone)`; `grid.start_index(domain)`/`end_index(domain)` return int32 bounds into the boundary-ordered arrays. `Zone` enum (:111): `END, INTERIOR, HALO, HALO_LEVEL_2, LOCAL, LATERAL_BOUNDARY..LEVEL_8 (edges) / ..LEVEL_4 (cells/vertices), NUDGING, NUDGING_LEVEL_2 (edges only)`. These are direct ports of ICON's `rl_start/rl_end` double-index scheme; the docstring tables (:117–192) map every ICON constant (`min_rlcell_int-1` etc.) to a python index into `start_idx_c/e/v` arrays of size `_CELL_GRF=14`, `_EDGE_GRF=24`, `_VERTEX_GRF=13` (:49–51). Memory layout (described in grid_refinement.py:121): `|lateral-boundary rows|nudging|interior|halo1|halo2| END`.

**refin_ctrl** (`grid_refinement.py`): refinement value = row distance from lateral boundary (cells: ≤4 ordered; edges: ≤9; vertices: ≤4; `_GRID_REFINEMENT_BOUNDARY_WIDTH`, :111). `compute_domain_bounds` (:175) derives start/end per Domain from refinement values + decomposition halo masks; notable wart `_refinement_level_placed_with_halo` (:155): in LAM, *some halo points that are also lateral-boundary points are placed in the lateral-boundary section, not the halo section → halos are **not contiguous** in the LAM case* (the docstring calls the reason "mysterious").

`SimpleGrid` (simple.py:414) is a hand-written 18-cell/27-edge/9-vertex periodic torus test grid with all 13 tables literal and no skip values; start_index/end_index are trivially 0/size.

---

## 3. Halo / decomposition handling

**`DecompositionInfo`** (`decomposition/definitions.py:147`): per horizontal dimension stores `global_index` (local→global map), `owner_mask` (bool), `halo_levels` (int array of `DecompositionFlag`). Query API: `global_index(dim, EntryType.ALL|OWNED|HALO|HALO_LEVEL_1|HALO_LEVEL_2)`, `halo_level_mask`, `get_halo_size`.

**Halo levels** (`DecompositionFlag`, definitions.py:624) — quoted definitions:
- `OWNED = 0`
- `FIRST_HALO_LEVEL = 1`: cells sharing 1 edge with an OWNED cell; vertices/edges on owned cells but owned by the neighboring rank.
- `SECOND_HALO_LEVEL = 2`: cells sharing one vertex with an OWNED cell; vertices on first-level cells not on owned cells; edges sharing exactly one vertex with an owned cell.
- `THIRD_HALO_LEVEL = 3`: *"This type does not exist in ICON"* — edges only on second-level cells.

**`IconLikeHaloConstructor`** (`decomposition/halo.py:58`) builds this from a cell→rank map using only `C2E2C, E2C, C2E, C2V, V2C, V2E`. Its `__call__` docstring (halo.py:241–336, with ASCII art) is the most explicit statement of halo-depth ↔ connectivity-reach in the repo:
- *"constructs a halo similar to ICON which consists of **exactly** 2 cell-halo lines"*; level-2 cells are those completing the **dual-cell hexagon** of cutting-line vertices (explicitly noted to possibly differ from ICON and from c2e2c2e2c reach).
- Edge halo has 3 levels; the third is **ICON4Py-only**: *"This is the HALO LINE which makes the C2E connectivity complete (= without skip value) for a distributed setup."*
- Ownership of cutting-line edges/vertices: unique-owner convention "max rank owns" (`_update_owner_mask_by_max_rank_convention`, :170, citing `mo_decomposition_tools.f90`); arrays are then reordered so owned entries come first.

**GHEX integration** (`decomposition/mpi_decomposition.py:113`, `GHexMultiNodeExchange`): one `DomainDescriptor` + `pattern` per horizontal dimension, built from `global_index(ALL)` and `local_index(HALO)`; `HaloGenerator.from_gids(global_halo_idx)`. Exchange granularity is therefore **all halo lines of a dimension at once** — there is no per-level exchange. GPU-aware: `schedule_exchange(patterns, stream)` / `schedule_wait(stream)` with a CUDA-stream protocol (`Stream`, `DEFAULT_STREAM`, `BLOCK` in definitions.py); applied patterns cached per `(dim, buffer_info.hash_key)`. Fields larger than the grid (Fortran nproma padding) are sliced to grid size before exchange (`_slice_field_based_on_dim`, :183). `SingleNodeExchange` (definitions.py:422) is a no-op stand-in selected by `create_exchange` singledispatch on `ProcessProperties`.

**Where exchanges live today: manually in granule code**, mirroring ICON's `sync_patch_array` calls. Examples:
- `diffusion.py:752` `# 2. HALO EXCHANGE -- CALL sync_patch_array_mult u_vert and v_vert`; `:828` overlapping comm/compute via `handle_edge_comm = self._exchange(... full_exchange=False)` + later `halo_exchange_wait(...)` with comment *"need to do this here, since we currently only use 1 communication object"*; `:880` conditional skip *"halo exchange can be skipped in the case of NWP or AES physics because... there is another halo exchange after the physics"*.
- `solve_nonhydro.py:1147,1224,1232,1332,1408`; `advection.py:211,321,408`; init-time exchanges in `interpolation_fields.py`, `metric_fields.py`, `compute_zdiff_gradp.py`, `topography.py` (all blocking `stream=BLOCK`).
- The driver creates the exchange once (`driver/icon4py_driver.py:429`) and injects it into granules.

**Alternative pattern — compute into the halo instead of exchanging**: stencil run domains frequently end at `end_index(HALO)` / `end_index(HALO_LEVEL_2)` rather than `LOCAL` (diffusion.py:480 `"horizontal_end": self._edge_end_halo_level_2`, :536 `self._cell_end_halo`; solve_nonhydro.py:919–966). This works only because halo connectivity rows are valid up to a given reach — the implicit contract a mesh concept would make explicit.

**Fortran/granule path** (`bindings/grid_wrapper.py:66` `grid_init`): ICON passes 9 connectivities (`c2e,e2c,c2e2c,e2c2e,e2v,v2e,v2c,e2c2v,c2v` — note `e2v` is sliced `[:, 0:2]` from an e2c2v-shaped input, bindings/common.py:232), nproma-sized & 1-based (shrunk + offset in `construct_icon_grid`, common.py:185), the ICON `start_idx/end_idx` arrays directly (no refinement recomputation), plus owner masks & global indices per dim → `DecompositionInfo` with `halo_levels=None` (common.py:317–323, halo levels unavailable from Fortran). `C2E2CO`/`E2C2EO` are built by `add_origin` column_stack; `C2E2C2E`/`C2E2C2E2C`/`V2E2V` are *not available* in the granule path.

---

## 4. Usage patterns

**Stencil declaration**: stencils import offsets and local dims from `icon4py.model.common.dimension` and use them in three ways:
1. Neighbor reductions: `neighbor_sum(w(C2E2CO) * geofac_n2s, axis=C2E2CODim)` (diffusion `calculate_nabla2_for_w.py:20`); coefficients live in sparse fields `Field[Dims[CellDim, C2E2CODim], wpfloat]`.
2. Constant-index sparse access: `u_vert_wp(E2C2V)` then `v_n[E2C2VDim(0)]`, or composed per-neighbor shifts `theta_v(C2E2C[1])` (`calculate_nabla2_and_smag_coefficients_for_vn.py:40-60`, `truly_horizontal_diffusion_nabla_of_theta_over_steep_points.py:29-35`).
3. Vertical: `Koff[±k]` everywhere; `concat_where(KDim < last_lev, t(Koff[1]), t)`; and **dynamic vertical indirection** `theta_v(C2E2C[0])(as_offset(Koff, zd_vertoffset[C2E2CDim(0)]))` (`gt4py.next.experimental.as_offset`, used in diffusion + dycore, marked `uses_as_offset` in tests).

**Multi-hop heavy users**: RBF edge→cell interpolation uses `C2E2C2E` (9 nbh, `interpolation/stencils/edge_2_cell_vector_rbf_interpolation.py`, `rbf_interpolation.py:124`); advection cubic reconstruction uses `C2E2C2E2C` (9 nbh, `reconstruct_cubic_coefficients_svd.py:49-57`); `V2E2V` is read from the grid file but currently only used for geometry/IO, not in any stencil.

**Non-standard items / things that stress a pure Vertex/Edge/Cell mesh**:
- `Koff`/`KHalfOff`: vertical offsets ride in the same `offset_provider` dict as `Dimension` values (base.py:92); many programs run with `offset_provider={"Koff": dims.KDim}` only (muphys, diffusion utils).
- `KHalfDim` as a distinct vertical dimension (advection, dycore states) — staggering, not topology.
- `LsqUnkDim`: LOCAL-kind dim that is *not* a neighbor axis (3-D field Cell×LsqUnk×C2E2C).
- `RBFDimension` (`rbf_interpolation.py:20`): plain Python enum with stencil sizes CELL=9, EDGE=4, VERTEX=6 — conceptually "the sparse size of C2E2C2E / E2C2E / V2E".
- Geometry sparse fields over connectivity dims read directly from the grid file: `Field[EdgeDim, E2CDim]` edge-cell distances, `Field[VertexDim, V2EDim]` edge orientation (grid_manager.py:290-334) — sparse *data* fields sharing the connectivity's local dim.
- nproma padding: in the Fortran path field buffers are longer than the grid; tables get shrunk, exchanges slice.
- Two geometries (icosahedron and torus) with identical topology machinery; torus has no pentagons (SimpleGrid has zero skip values).
- Ordering constraints: derived tables must reproduce **ICON's neighbor ordering** (e.g. `icon_edge_order` in `_construct_diamond_edges`, grid_manager.py:679) because interpolation coefficients are order-dependent.
- Everything else *does* fit a Vertex/Edge/Cell + K model: every horizontal connectivity is between those three entity types, all composed offsets are precomputed into flat tables, and max-neighbor counts are statically known per offset.

**Summary of design-relevant facts**: (a) the offset_provider is already grid-owned and uniform (`grid.connectivities`), so a mesh object replacing it is a localized change; (b) skip values have three distinct causes (pentagons / LAM boundary / halo edge-of-reach) with an existing table-rewriting workaround that a halo-aware mesh with domain inference could eliminate; (c) halo depth is fixed (2 cell lines, 2 vertex lines, 3 edge lines) and explicitly tied to making 1-hop connectivities complete — multi-hop offsets (C2E2C2E2C, C2E2C2E) are *not* complete in the outer halo, which is exactly why outputs are restricted via start/end_index Zones; (d) exchanges are manual, per-dimension, all-halo-lines-at-once, GPU-stream-aware, with both overlap (start/wait) and skip optimizations hand-coded in granules.