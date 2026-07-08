---
title: "Mesh & first-class halos: prior-art survey"
author: havogt
tags: [mesh, halos, unstructured, connectivities, prior-art, research, distributed, halo-exchange]
created: 2026-06-12
status: draft
---

> **Status**: research appendix to
> [[personal/havogt/mesh-and-first-class-halos|A mesh concept with first-class halos]].

Research input for replacing `offset_provider` with a mesh concept restricted to
basic entities (Vertex, Edge, Face/Cell), with halo regions as a native concept
enabling automatic halo-exchange placement.

Sources: web research (verified-claim pipeline over ~21 primary sources) + survey of
`icon4py` (current connectivity/halo usage). Citations inline. The icon4py findings
summarized in §2.3 are recorded in full in the companion
[[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos_icon4py-survey|icon4py constraint survey]].

---

## 1. The design space, as prior art divides it

Prior art separates cleanly into three layers. No single system covers all three;
gt4py would be combining them.

| Layer | What it provides | Best prior art |
|---|---|---|
| **Mesh data structure** | entity types, typed connectivity tables, per-entity ownership/halo metadata | Atlas, MPAS, ICON, DUNE |
| **Topology formalism** | connectivity as composable incidence relations; halo/overlap as closure of compositions | Sieve/DMPlex, Logg (FEniCS), GrAL, Regent dependent partitioning |
| **Compiler placement** | derive required halo depth from kernel access patterns, insert/elide exchanges, track validity | PSyclone/LFRic (only full implementation), PyOP2, Liszt |

---

## 2. Mesh data-structure layer

### 2.1 ECMWF Atlas — the closest "restricted entities" precedent

- Conceptual model is exactly three entity classes — **Nodes, Cells, Edges** — with
  typed pairwise adjacency between them (Nodes: edge_/cell_connectivity; Cells:
  node_/edge_connectivity; Edges: node_/cell_connectivity). Connectivities are
  dedicated table classes, not a flat dict: fixed-arity `BlockConnectivity`,
  variable-arity `IrregularConnectivity` (node-to-X), `MultiBlockConnectivity`
  for hybrid-element meshes. [Deconinck et al., CPC 2017; arXiv:1908.06091, Fig. 24]
  - *Nuance (a stricter claim was refuted in verification):* the implementation
    generalizes by **codimension** (`facets_/ridges_/peaks_` members, edges aliased
    to facets in 2D). "Exactly three" is the conceptual model, not the class layout.
    Relevant if 3D unstructured is ever in scope: name entities by (co)dimension.
- **Halo/ghost is first-class via a minimal per-entity metadata triple**:
  `global_index`, `partition`, `remote_index` on every entity class. Ghost = entity
  whose `partition` ≠ local partition; `remote_index` locates it on the owner.
  These three fields alone suffice to construct all parallel communication
  (`parallel::HaloExchange`, `GatherScatter`) without user involvement.
  `mesh::Halo` and `IsGhostNode` are API concepts. [CPC 2017 §; verified 3-0]
- Halos are built by **incrementally growing the overlap by node-sharing elements**
  (halo level = one element-adjacency expansion); a "periodic overlap region"
  treats the periodic E-W boundary like an internal partition boundary.
  (`mesh::actions::BuildHalo`, `nb_elems` parameter.)
- **Halo depth is NOT derived** from access patterns: it is an explicit integer at
  FunctionSpace construction (`NodeColumns(mesh, Halo(1))`), and `haloExchange(field)`
  is a manual per-field call. Atlas is deliberately a data-structure framework;
  stencil analysis is left to client DSLs. This marks the boundary of what Atlas
  offers: the mesh/halo *model*, not the placement toolchain.

### 2.2 MPAS — a closed connectivity vocabulary suffices for production

- Exactly three entity types (cells, edges, vertices) on Voronoi/Delaunay duals.
  Connectivity is a **closed, required set of named tables** with fixed leading
  dims: `cellsOnCell`, `edgesOnCell`, `verticesOnCell` (maxEdges×nCells);
  `cellsOnEdge`, `verticesOnEdge` (2×nEdges); `cellsOnVertex`, `edgesOnVertex`
  (vertexDegree×nVertices); plus `edgesOnEdge` (maxEdges2×nEdges) for tangential
  velocity reconstruction. No arbitrary user-named tables. [MPAS Mesh Spec v1.0]
- Strong precedent that a fixed vocabulary covers a production atmosphere/ocean
  dycore. Note `edgesOnEdge` is their one "weird" multi-hop-ish member — analogous
  to icon4py's E2C2E/E2C2EO.
- Halo: configurable number of halo layers grown by cell adjacency (typ. 2–3 for
  the atmosphere core); exchanges are explicit `mpas_dmpar` calls in the dycore.
  (Not part of the verified-claim set; from general knowledge of the codebase.)

### 2.3 ICON / icon4py — the constraint set the mesh concept must satisfy

From the icon4py survey (full detail in the
[[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos_icon4py-survey|icon4py constraint survey]];
paths relative to the `icon4py` repository root):

- **Connectivity inventory is closed and small**: 14 horizontal offsets, all in
  `model/common/.../dimension.py:33-48`, every one between Cell/Edge/Vertex, up to
  4-hop (`C2E2C2E2C`, 9 neighbors). "O"-suffixed variants prepend the origin.
  Multi-hop tables are **precomputed** (derived in `grid_manager.py` from the
  1-hop tables read from the grid file).
- **Halo depth is fixed per entity type** and explicitly tied to connectivity reach
  (`decomposition/definitions.py:624`, `decomposition/halo.py:241-336`):
  cells: 2 lines (line 1 = shares an edge with owned; line 2 = shares a vertex);
  vertices: 2 lines; edges: 3 lines, where the **third edge line exists solely
  "to make the C2E connectivity complete (= without skip values) for a distributed
  setup"** and does not exist in ICON itself. This is the clearest in-repo statement
  of "halo depth = f(connectivity)".
- **Skip values have three distinct causes** (`grid/icon.py:34-44,130`):
  (a) pentagons — intrinsic topology, `V2E/V2C/V2E2V` always; (b) LAM lateral
  boundary; (c) halo edge-of-reach in distributed runs. There is a workaround
  (`keep_skip_values=False`, `base.py:180`) that rewrites tables, documented as
  compensating for *"the lack of a domain inference for unstructured neighbor
  accesses in GT4Py"*. A halo-aware mesh + domain inference eliminates cause (c)
  and the workaround.
- **Ownership convention**: entities on the cut line get unique owners by a
  max-rank convention (ported from `mo_decomposition_tools.f90`); owned entries
  are reordered first; memory layout is
  `|lateral-boundary|nudging|interior|halo1|halo2|`. Wart: in LAM, some halo
  points that are also lateral-boundary points land in the LB section → **halos
  are not contiguous in the LAM case** (`grid_refinement.py:155`).
- **Exchanges are manual today**: GHEX pattern per horizontal dimension, built from
  `global_index(ALL)` + `local_index(HALO)` (`mpi_decomposition.py:113`); always
  all halo lines of a dimension at once; hand-placed in granules mirroring ICON's
  `sync_patch_array`, including hand-coded overlap (`full_exchange=False` +
  `halo_exchange_wait`) and hand-reasoned skips ("can be skipped because there is
  another halo exchange after the physics").
- **The alternative to exchanging is computing into the halo**: stencil output
  domains routinely end at `end_index(HALO)` / `end_index(HALO_LEVEL_2)` instead of
  `LOCAL`. This works only because connectivity rows are valid up to a given reach —
  the implicit contract a mesh concept would make explicit.
- **Neighbor ordering is semantically load-bearing**: derived tables must reproduce
  ICON's ordering (e.g. `icon_edge_order` permutation in `_construct_diamond_edges`)
  because interpolation coefficients are order-dependent.
- Items outside the V/E/C model that ride along today: `Koff`/`KHalfOff` (vertical,
  in the same offset_provider dict), `LsqUnkDim` (LOCAL dim that is *not* a neighbor
  axis), sparse *data* fields over connectivity dims (`Field[EdgeDim, E2CDim]`
  distances read from the grid file), nproma padding in the Fortran path.

### 2.4 DUNE — naming for per-entity parallel status

The DUNE grid interface classifies every entity (per codimension) with a
`PartitionType`: **interior, border, overlap, front, ghost** — interior = uniquely
owned; border = on the boundary between interior and overlap (shared, owned by
convention); overlap = copies for redundant computation; ghost = copies only for
reading. [Bastian et al., "A generic grid interface for parallel and adaptive
scientific computing", Computing 82, 2008.] This five-way split is the most
fine-grained published vocabulary for what icon4py expresses with
`owner_mask` + `halo_levels` + zones, and it maps directly onto the
exec/non-exec distinction (§4.2).

### 2.5 Others (brief)

- **LFRic mesh**: dofs live on *all* entity types (volumes, faces, edges, vertices)
  of an extruded unstructured mesh; cell-based partitioning ⇒ shared dofs on
  partition boundaries whose ownership assignment "adds to the complexity of
  setting up the halo exchanges"; routing tables built from global ids of owned +
  halo dofs (ESMF initially, later YAXT). [arXiv:1809.07267]
- **p4est / t8code**: ghost layer built **on demand** rather than stored in the
  mesh, via generic codimension-1 subalgorithms; argues for modularity of mesh vs
  ghost construction. [Holke, Knapp, Burstedde, arXiv:1910.10641]
- **UGRID conventions**: file-format-level entity restriction (node/edge/face
  (+volume)) with named connectivities (`face_node_connectivity`,
  `edge_node_connectivity`, optional `face_edge`/`face_face`), `_FillValue` for
  missing neighbors, `start_index` attribute. No halo concept — but it is the
  natural serialization target for a V/E/C mesh and validates the naming scheme.
- **ESMF Mesh**: nodes + elements only (no first-class edges); halo via
  `ESMF_FieldHaloStore` routing handles. Less informative for this design.

---

## 3. Topology formalism layer — connectivity as algebra

### 3.1 Sieve / PETSc DMPlex — incidence as the first-class object

- Sieve "makes instances of the incidence relation, or arrows, the conceptual
  first-class objects": all entities of all dimensions are uniform "points" in a
  DAG (Hasse diagram); `cone`/`support` are the one-hop down/up incidence,
  `closure`/`star` their transitive closures. Mesh distribution and overlap
  construction are then **dimension-independent algorithms on the DAG**, and the
  same algorithm distributes mesh-associated data. [Knepley & Karpeev,
  arXiv:0908.4427; verified 3-0]
- **Overlap-k**: `DMPlexDistributeOverlap(dm, overlap, ...)` takes an integer
  number of adjacency levels, where **adjacency itself is user-configurable**
  via `DMSetAdjacency()` — FEM-style `closure(star(p))` vs FVM-style
  `support(cone(p))` — "chosen appropriate for the function representation on
  the mesh". I.e. *one* integer depth, but parameterized by *which composed
  incidence relation* defines one level. This is the principled version of
  "V2E needs one edge-halo level": a level is a relation application.
- Lesson: even if gt4py does not adopt the uniform-DAG representation (it costs
  memory and genericity gt4py doesn't need), the *algebraic* view — halo region =
  image of the owned set under a composed incidence relation — is the right
  foundation, and it is proven at scale in PETSc.

### 3.2 Logg / FEniCS — d→d′ incidence with derivation rules

"Efficient representation of computational meshes" [Logg, arXiv:1205.3081 / IJCSE
2009] represents mesh topology as the set of incidence relations d→d′ between
entities of topological dimensions d, d′, and shows **all of them are derivable
from the single cell→vertex relation** via three algorithms (Build, Transpose,
Intersection), computed lazily on demand. For gt4py: a mesh concept needs to
*store* only a minimal generating set (e.g. C2V or C2E+E2V) and can *derive* the
rest — which icon4py's `grid_manager._get_derived_connectivities` already does ad
hoc for the multi-hop tables. The formalism gives the derivation a name and a
completeness argument.

### 3.3 GrAL — grids as algebra, stencils as relation sequences

Berti's GrAL [iccs2002; thesis 2000] defines a generic grid algebra of incidence
sequences (e.g. cell→vertex→cell) and — in the distributed-grid part — derives
**overlap ranges from a stencil specified as a sequence of incidence hops**. It is
the earliest direct prior art for "declare the access pattern as a composition of
basic connectivities, compute the required overlap mechanically".

### 3.4 Regent/Legion — dependent partitioning: halos as images

Legion's dependent partitioning [Treichler et al., OOPSLA 2016; Slaughter thesis]
computes partitions as **images/preimages of pointer (relation) fields**: given a
partition of vertices and the V2E relation, `image(V2E, vertex_part)` *is* the
edge set each rank needs — ghost regions are *defined*, not hand-built, and the
runtime derives communication from them. This is the cleanest formal statement of
"a V2E connectivity requires a certain amount of edge halos": **edge halo =
image(owned vertices under V2E) minus owned edges**. Multi-hop composes by taking
images of images.

---

## 4. Compiler placement layer — deriving exchanges from access patterns

### 4.1 PSyclone/LFRic — the only full implementation (and its scars)

The single verified system that does what gt4py wants end-to-end. Mechanisms
[Adams et al., JPDC / arXiv:1809.07267; PSyclone source `src/psyclone/lfric.py`]:

- **Input**: declared kernel metadata — per-argument access descriptors
  (read/write/readwrite/increment), function space (⇒ continuity), iteration
  entity (`operates_on`), optional per-argument **stencil descriptors** with
  extent. From this + the loop's iteration space, dependence analysis inserts
  exchanges; originally conservative (any write ⇒ dirty), later only provably
  required ones.
- **Halo depth is a symbolic, first-class object**: `HaloDepth` holds a PSyIR
  expression (simplified via symbolic math) plus flags `max_depth`,
  `max_depth_m1`, `annexed_only` — *not* an integer. `HaloReadAccess` computes,
  per field per kernel per loop, the read depth *and* the access pattern from
  continuity, access type, stencil descriptor, and loop bounds.
- **Depth composition**: redundant computation to depth *r* + stencil extent *s*
  ⇒ required read depth **r + s** (sum); but for kernels that *must* iterate into
  the halo the rule is **max** — self-described in-source as "an ad-hoc fix",
  open issue stfc/PSyclone#2781. ⚠ Lesson: define composition algebraically from
  day one; bolting arity rules on later leaves permanent warts.
- **Runtime clean/dirty tracking**: per-field, per-depth flag;
  generated code does `IF (proxy%is_dirty(depth=1)) CALL proxy%halo_exchange(depth=1)`;
  loop bounds use `mesh%get_last_halo_cell(k)`. Compile-time placement +
  runtime guard composes well with conditional call paths.
- **Redundant computation** into the halo is a first-class *transformation*
  (trade exchange for compute), with the bookkeeping consequence that the
  outermost written halo level of a continuous field is only a **partial sum**
  ⇒ `clean_depth = written_depth − 1` (`dirty_outer`).
- **Annexed dofs**: a continuous field read in an owned-cells loop still touches
  halo dofs on owned-cell boundaries; with no annexed-only exchange mechanism,
  PSyclone conservatively issues a depth-1 exchange flagged `annexed_only`.
  (An LFRic config option iterates over annexed dofs redundantly instead, making
  them always-clean — redundant computation as a *policy* to reduce exchanges.)

**Key structural difference favoring gt4py**: LFRic's hard cases (annexed dofs,
partial-sum dirty-outer) all stem from *scatter/increment* writes on continuous
function spaces. gt4py's field view is **gather-only** — neighbor access only
reads; writes always go to the iteration entity. Halo validity therefore
propagates forward deterministically:
`valid_depth(out) = min over inputs of (valid_depth(in) ⊖ footprint)` —
a pure dataflow analysis, with an exchange inserted when required depth exceeds
valid depth. gt4py also already *has* exact per-kernel access footprints in the
IR (offset literals in ITIR), strictly better information than LFRic's
hand-written kernel metadata. Automatic placement is structurally *easier* here
than in the strongest prior art.

### 4.2 PyOP2/Firedrake — entity classes + lazy exchange + overlap

[op2.github.io/PyOP2/mpi.html, fetched & quoted]

- Four contiguous-in-memory entity classes per set:
  **core** ("owned, can be processed without accessing halo data"),
  **owned** ("owned, access halo data when processed"),
  **exec halo** ("off-processor, redundantly executed over because they touch
  owned entities"),
  **non-exec halo** ("off-processor, not processed, but read when computing the
  exec halo").
- **Lazy exchange driven by access descriptors**: writes (`INC/WRITE/RW`) mark
  halos stale; an exchange fires only when a later `READ/RW` needs it.
- **Comm/compute overlap is structural**: `halo_exchange_begin(); compute(core);
  halo_exchange_end(); compute(owned + exec_halo)` — the core/owned split exists
  *exactly* to enable overlap. icon4py hand-codes this today
  (diffusion.py:828); a mesh concept with core/owned classification gives it to
  the compiler for free.
- Halo depth itself is fixed by construction (grown via DMPlex overlap in
  Firedrake); "halos grow by about a factor two" due to exec+non-exec.
- Mapping to icon4py: *exec halo* ≡ the halo lines stencils compute into
  (`end_index(HALO_LEVEL_2)` pattern); *non-exec halo* ≡ the outermost line that
  only exists to complete connectivities (the icon4py-only 3rd edge line).
  The prior art validates making this distinction explicit per entity type.

### 4.3 Liszt — footprint inference by program analysis

Liszt [DeVito et al., SC'11] infers each kernel's **stencil** (set of accessed
mesh elements relative to the iteration element) by abstractly interpreting the
program over topology operators (vertices(cell), edges(vertex), …), then uses the
inferred footprint to build ghost regions for MPI partitioning (and coloring for
GPU). Demonstrates footprint inference from code rather than declarations —
which is gt4py's situation (footprints fall out of the IR, no user metadata
needed).

### 4.4 Runtime exchange layers (GHEX, YAXT)

GHEX (already icon4py's backend) consumes exactly Atlas-style metadata: domain
descriptors with global ids of all entities + halo ids
(`HaloGenerator.from_gids`), builds reusable patterns per (dimension, buffer);
GPU-stream-aware start/wait. YAXT (LFRic's current layer) likewise builds routing
from global ids of owned/halo dofs. ⇒ Whatever the mesh concept stores, it must
be able to emit: per entity type, (global_index, owner/partition, halo-level
mask) — the Atlas triple + level labels. icon4py's `DecompositionInfo` already
is this; it should become part of the mesh concept rather than a parallel object.

---

## 5. Synthesis — design lessons for gt4py

1. **The restricted entity vocabulary is well-precedented.** Atlas, MPAS, UGRID,
   ESMF all restrict to ~3 entity classes; MPAS proves a *closed named table set*
   suffices for a production dycore; icon4py's entire inventory (14 offsets) is
   already within V/E/C. The flexibility being given up (arbitrary named
   neighbor tables) is precisely what blocks halo reasoning today: an opaque
   table has no derivable reach.

2. **Declare multi-hop connectivities as compositions; keep them materialized.**
   `C2E2CO = id ∪ (C2E ∘ E2C)`, `E2C2V = E2C ∘ C2V`, etc. The algebra term gives
   the compiler the *reach* (⇒ required halo per entity type, by
   image-composition à la Regent/DMPlex/GrAL); the materialized table keeps
   runtime performance and — critically — **pins ICON's neighbor ordering**,
   which is semantically load-bearing (order-dependent interpolation
   coefficients). A composition declaration must therefore either fix a
   deterministic ordering rule or accompany the table, not replace it.
   ("O" variants = reflexive closure, a one-token algebra annotation.)

3. **Halo depth should be per-entity-type, symbolic, and computed as relation
   closure** — not a single integer on the mesh. ICON's structure (2 cell /
   2 vertex / 3 edge lines, the 3rd edge line existing to complete C2E) is
   *itself* the output of such a computation; the mesh concept should be able to
   both (a) *verify* a given precomputed halo supports a program's footprint and
   (b) *derive* the minimal halo for a given set of declared connectivities.
   PSyclone's sum-vs-max wart (#2781) is the cautionary tale for ad-hoc depth
   arithmetic.

4. **Ownership metadata: adopt the Atlas triple + level labels.**
   (global_index, owner_partition, remote_index) per entity, plus halo-level
   classification (icon4py `DecompositionFlag`) and ideally the DUNE/PyOP2-style
   execution classes (core / owned / exec-halo / non-exec-halo). That is
   simultaneously: what GHEX needs, what enables comm/compute overlap
   structurally, and what makes "compute into the halo instead of exchanging" a
   safe, checkable transformation instead of a convention.

5. **Automatic placement is a forward dataflow problem in gt4py** — easier than
   LFRic. Because the field view is gather-only, per-field *valid-depth* (per
   entity type) propagates through each kernel as
   `valid(out) = min_inputs(valid(in) shrunk by footprint)`; insert an exchange
   when a consumer needs more than is valid, or enlarge the output domain
   (redundant computation, PSyclone-style policy / current icon4py practice)
   when inputs permit. Runtime clean/dirty flags (LFRic) remain useful across
   program boundaries / conditional call paths — icon4py's hand-written
   "skippable exchange" comments are exactly the cases a dirty-flag would
   handle automatically.

6. **Skip values get a principled split.** Pentagon skips = intrinsic variable
   arity (Atlas `IrregularConnectivity`; UGRID `_FillValue`) — stays. LAM
   lateral-boundary skips = domain boundary — stays, but is a *mesh boundary*
   property, not a table accident. Halo edge-of-reach skips = artifact of
   missing domain inference — eliminated by lesson 5 (and with them the
   `keep_skip_values=False` table-rewriting workaround and its Fortran-path
   special case).

7. **Keep the vertical out of the mesh.** `Koff`/`KHalfOff` are structured-offset
   concepts that currently ride in the offset_provider dict; the mesh concept is
   horizontal topology. (Same for `LsqUnkDim`-style LOCAL dims and sparse data
   fields over connectivity dims — they reference the mesh's local dimensions
   but are field-type, not mesh, concerns.)

8. **Known hard spots to design for, from icon4py**: non-contiguous halos in the
   LAM case (halo ∩ lateral-boundary points); max-rank ownership convention on
   cut lines (must match ICON bit-for-bit for verification runs); the Fortran
   granule path where halo_levels are unavailable and start/end indices come
   from ICON directly; nproma-padded buffers larger than the mesh.

## 6. Open questions

- Fixed precomputed halo (ICON-style, capped) vs on-demand growth (Atlas
  BuildHalo / t8code): for ICON compatibility the halo is given; the compiler
  verifies and splits/exchanges. Should the concept *also* support derive-and-
  build for non-ICON meshes (it falls out of lesson 3(b))?
- Granularity of exchange: today GHEX exchanges all halo lines of a dimension at
  once; per-level (depth-parameterized, LFRic-style) exchange needs per-level
  patterns — worth the pattern-setup cost?
- Where does the valid-depth dataflow live — embedded (runtime checks on a
  mesh-aware field), compiled backends (ITIR pass), or both?
- Ordering: can a deterministic ordering rule for composed connectivities
  reproduce ICON's orderings (`icon_edge_order`), or must ordering remain a
  property of the materialized table?

## 7. Source index

- PSyclone/LFRic: arXiv:1809.07267 (Adams et al.); PSyclone `src/psyclone/lfric.py`
  (`HaloDepth`, `HaloReadAccess`, `HaloWriteAccess`); stfc/PSyclone#2781;
  psyclone.readthedocs.io (LFRic API).
- Atlas: Deconinck et al., CPC 2017 (doi 10.1016/j.cpc.2017.05.006);
  arXiv:1908.06091; `atlas/mesh/Connectivity.h`, `actions/BuildHalo.h`.
- MPAS: MPAS Mesh Specification v1.0 (mpas-dev.github.io).
- Sieve/DMPlex: Knepley & Karpeev arXiv:0908.4427; petsc.org DMPlex manual,
  `DMPlexDistributeOverlap`, `DMSetAdjacency`; Lange/Knepley/Gorman 2016
  (10.1137/15M1026092); arXiv:1506.06194 (overlapping mesh distribution).
- FEniCS incidence: Logg, arXiv:1205.3081.
- GrAL: Berti, ICCS 2002 (pubs.berti-cmm.de/docs/iccs2002.pdf).
- PyOP2: op2.github.io/PyOP2/mpi.html, /concepts.html; Firedrake arXiv:1506.07749.
- Liszt: DeVito et al., SC'11 (theory.stanford.edu/~aiken/publications/papers/sc11.pdf).
- Regent/Legion: dependent partitioning (Treichler et al., OOPSLA'16); Slaughter
  thesis (theory.stanford.edu/~aiken/publications/theses/slaughter.pdf).
- p4est/t8code ghost: arXiv:1910.10641 (Holke, Knapp, Burstedde).
- DUNE: Bastian et al., Computing 82 (2008), parts I & II.
- icon4py: `model/common/src/icon4py/model/common/dimension.py`, `grid/icon.py`,
  `grid/base.py`, `grid/grid_manager.py`, `grid/horizontal.py`,
  `grid/grid_refinement.py`, `decomposition/{definitions,halo,mpi_decomposition}.py`,
  `bindings/{common,grid_wrapper}.py`.

*Coverage note*: the verified-claim pipeline confirmed PSyclone/LFRic, Atlas,
MPAS, Sieve/DMPlex 3-0; PyOP2, DMPlex overlap, Liszt, t8code were filled by
direct fetches; DUNE, UGRID, ESMF, GrAL details, MPAS halo config and Regent
dependent-partitioning specifics are from primary-source knowledge, not
independently re-verified — double-check before quoting in an ADR. Not covered:
libMesh, MOAB/iMesh, PUMI/Omega_h, FESOM, deal.II, Trilinos STK (judged
lower-yield: FEM mesh databases / coupler conventions without halo-derivation
machinery).
