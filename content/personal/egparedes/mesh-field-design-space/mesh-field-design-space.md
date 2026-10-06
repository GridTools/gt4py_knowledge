---
title: "Meshes, dimensions and fields: three typed designs and how they layer"
author: egparedes
tags: [type-system, type-checking, dimensions, mesh, fields, domain, connectivities, local-dimensions, unstructured, staggering, halos, boundary-conditions, distributed, global-semantics, local-semantics, dependent-types, nominal-types, cw-complex, exterior-calculus, ugrid, sgrid, cf-conventions, xarray, coordax, uxarray, design-space, prior-art]
created: 2026-10-06
status: draft
---

> **TL;DR** Three directions for how `gt4py.next` could represent index spaces,
> meshes, connectivities and fields in idiomatic, statically typed Python. All three
> are grounded in the dependent-typing reading of mesh connectivities. They differ
> in where the mesh lives and in what a program *means*:
>
> - **A — Typed relations over a rank-local mesh.** *Local semantics* (SPMD). Types
>   name index spaces and relations, never meshes. A program runs per rank on a
>   validated local binding. `Owned`, exchange points and global reductions are
>   explicit; halo depth is not.
> - **B — Logical mesh object plus a runtime decomposition.** *Global semantics*. A
>   field is a pair (logical mesh, data). Boundary regions belong to the mesh; halos
>   belong to a separate `Distribution` that source code cannot name.
> - **C — Located fields over cell complexes.** *Global semantics*. A field is typed
>   by its cell class, neighborhoods are the complex's incidences, and boundaries are
>   the boundary subcomplex.
>
> All three separate the **logical mesh** (topology, geometry, boundary regions:
> semantic) from the **decomposition** (ownership, halo lines: implementation). They
> are not exclusive: C lowers to B, B lowers per rank to A, and A lowers to today's
> GTIR (see [Layering](#layering-the-three-proposals-as-levels-of-abstraction)).

> **Status**: draft. This note records a design brainstorming session and was
> drafted with AI assistance. It fixes concepts and directions, not implementation
> details; it is a map of the design space, not a decision.
>
> **Appendix**:
> [[personal/egparedes/mesh-field-design-space/mesh-field-design-space_research|Research: knowledge-base inputs and ecosystem survey]].
> It covers what each note of this knowledge base contributes, the ICON / ICON4Py
> constraints, how xarray, coordax and uxarray and the UGRID, SGRID and CF
> conventions model meshes and fields, and where "a field should know its mesh"
> comes from. The library claims are backed by the probes
> [`xarray_probe.py`](xarray_probe.py), [`coordax_probe.py`](coordax_probe.py) and
> [`uxarray_probe.py`](uxarray_probe.py).

## Context and goal

The goal is a model for mesh-based computations in gt4py that:
- works in idiomatic Python;
- uses static typing wherever Python can express it;
- reflects the concepts scientists reason with: index spaces, meshes, neighborhoods,
  boundaries, and fields located on a mesh.

The ground rules of the exploration:
- **The theoretical base** is the dependent-typing reading of connectivities in
  [[personal/egparedes/connectivities-as-types/connectivities-as-types_dependent-typing|the dependent-typing appendix]]
  of [[personal/egparedes/connectivities-as-types/connectivities-as-types|Connectivities as types]].
  Everything else in that proposal may be kept, changed or discarded.
- **The other notes in this knowledge base** are sources of ideas, equally
  discardable: dependent local dimensions, generic dimensions, the mesh with halos,
  the field data protocol, the boundary-condition and scan notes, and the FD-syntax
  note.
- **The scientific-Python ecosystem and community conventions** (xarray, coordax,
  uxarray; UGRID, SGRID, CF) supply context and motivation, not requirements.
- **Python 3.12+ is assumed.** PEP 695 generics are used freely. Python 3.13 (PEP 696
  TypeVar defaults, PEP 742 `TypeIs`) and 3.14 (PEP 649 deferred annotations) are
  used where they help.

## Theoretical base: dependent typing

### Concepts from the appendix

| Concept | Meaning | Role in the proposals |
| --- | --- | --- |
| `Fin<n>`, type family `Vert<g>` (§1) | an index space whose bound depends on a value | entity index spaces |
| Π type (§2) | a result type that depends on an argument value | a program over a mesh; the type of a relation's row |
| refinement (§3) | a type restricted by a predicate | boundary regions; partial relations |
| Σ type (§4) | a pair whose second component's type mentions the first | incidence spaces; a field paired with its mesh |
| evidence (§5) | a checked, erased proof | validation when a mesh is bound or built |
| enumeration (§6) | the bijection between local slots and incident entities | slot-addressed vs incidence-addressed sparse fields |
| parameter vs index, brand (§7) | does the type identify the graph? | where the mesh lives |
| validate at the boundary (§8) | check once, reuse the evidence | binding / mesh construction |

### The ideal

In a language with dependent types:

```text
Mesh g       = { entity counts; relations r : (v : A<g>) → Vec<B<g>, deg_r(v)>; geometry; boundary }
A<g>         = Fin<g.n_A>                           -- an entity index space is a type family over g
Field<g,L,T> = L<g> → T                             -- a function on one location of one mesh
Slot<g,r,v>  = Fin<deg_r(v)>                        -- a neighbor slot depends on g, r and v (Π)
Incid<g,r>   = Σ v : A<g>. Fin<deg_r(v)>           -- an incidence: a flattened sparse position (Σ)
program      : (g : Mesh) → Field<g,E,T> → Field<g,V,T>   -- Π over the mesh
```

### What Python can keep

Python types cannot mention values. Each fact about `g` must therefore be assigned
to a stage:

1. **Static**: the type checker sees it. Only names and nominal identity.
2. **Compile time**: the frontend specializes on it, for example counts, dtype and
   presence of skip values. This is the two-level pattern that dtype generics and
   connectivities-as-types already use.
3. **Run time**: tables, extents, geometry, decomposition.

Limits that shape every design (verified by probes elsewhere in this knowledge base):
- nominal classes are the only usable brand;
- there are no integer type parameters;
- there are no type-level functions, so changing a dimension inside `Dims[...]`
  needs one overload per rank and position (bounded code generation);
- `TypeVarTuple` is unbounded, and only one may appear per tuple;
- `M.Cell` is not valid in a type when `M` is a `TypeVar`. Families that are generic
  over a mesh must therefore use subscription (`Face[M]`), not nested attributes.

## Logical mesh and decomposition

### Two objects, never one

- **The logical mesh `g`**: the global entity sets, relations, geometry and
  **boundary regions**. A boundary region is a refinement of a global entity space,
  e.g. `{c : Cell<g> | refin_ctrl(c) = k}`: ICON's distance-to-lateral-boundary
  levels, the nudging zone, the model top and surface.
- **The decomposition `π` of `g`**:
  - ranks `r`, each with a local mesh `g_r`;
  - injections `ι_r : Cell<g_r> → Cell<g>` (the global index);
  - owned sets that partition `Cell<g>`;
  - **halos**: the closure of the owned set under the relations, minus the owned set;
  - exchange patterns.

| | Boundary regions | Halo lines |
| --- | --- | --- |
| What they are | subsets of the global entity sets at the edge of the model domain | cached copies of values owned by other ranks |
| Exist on a single node | yes | no |
| Change results | yes: external data enters there | no, apart from rounding |
| Depend on the partition | no: a property of the logical mesh | yes: a property of the decomposition |
| Decided by | the model author, in source | the backend or configuration, never the source |
| Effect on the index space | restricts the global index space | extends the local index space |

Boundary regions pulled back through `ι_r` are the same on every rank (`refin_ctrl`
is a global property). A halo has no type-level meaning: it is a cache whose validity
the implementation tracks.

### Three causes of skip values, three owners

The three causes are those identified by
[[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos|the mesh proposal]].

| Cause | Owner | Visible in source |
| --- | --- | --- |
| ragged arity (pentagons) | the relation's table type: `max_neighbors`, presence of skip values | yes |
| boundary partiality: a missing neighbor at the LAM lateral boundary | the logical mesh: a **partial relation** (declared, or taken from the bound table) | yes: reads must be guarded by a boundary condition or a domain that excludes the boundary |
| truncation at the edge of the halo | the decomposition | never observable |

### Global and local semantics

- **Global semantics.** A program means a function of `g`. The code cannot observe
  `π`: no ranks, no halos, no communication. The implementation must satisfy the
  **simulation condition** `field_r = field ∘ ι_r` wherever `field_r` is valid; the
  backend discharges it through domain inference and exchange placement.
  Precedents: PSyclone/LFRic's algorithm layer, Legion/Regent, Chapel's global-view
  arrays.
- **Local semantics (SPMD).** A program means a function of `g_r` and runs once per
  rank. The code sees the local mesh, and communication and global reductions are
  explicit. The programmer discharges the simulation condition, with tool support.
  Precedents: ICON (`sync_patch_array`), Atlas (`haloExchange`), PETSc
  (`DMGlobalToLocal`), and gt4py and ICON4Py today.

**Requirement on backends under either semantics.**
- Per-entity results must be **bitwise identical under any decomposition**.
  Neighbor reductions use the table order on every rank, so recomputing a value in
  the halo equals exchanging it.
- Only reductions over a distributed dimension (global sums), whose summation order
  depends on the partition, may differ, and only by rounding.

### Design axes

| Axis | Choices explored |
| --- | --- |
| Semantics of a program | local (A) / global (B, C) |
| Where the mesh lives (§7) | parameter of the call plus a runtime brand (A) / carried by every field (B) / inherited, plus mesh *kind* at compile time (C) |
| User vocabulary | index spaces and relations (A) / mesh schema, entities and regions (B) / locations and operators (C) |
| Local dimensions | dependent slots and incidence spaces (A, B) / hidden behind incidences (C) |
| Boundary regions | declared on entity spaces (A) / owned by the mesh (B) / boundary subcomplex (C) |

## Proposal A — Typed relations over a rank-local mesh

> **Semantics: local (SPMD).** Programs express the local view of one element of a
> decomposition. The decomposition is visible *in kind*: a local index space, an
> `Owned` region, exchange points, explicit global reductions. It is **not** visible
> *in quantity*: halo depth is never written in source; it is configured or derived
> from the reach of the relations a program uses.

### Thesis

Types talk about index spaces and the relations between them, never about meshes.
A program runs per rank on a **local binding**: a validated, immutable value built
once from that rank's tables. This is the execution model of gt4py and ICON4Py today
(ICON4Py passes local connectivities and places exchanges), made typed and safe. It
is also what the ICON Fortran granule path needs natively.

### Core concepts

- **Index spaces** are nominal classes in four kinds:
  - `Entity`: unstructured locations.
  - `Axis`: Cartesian, with index arithmetic. `Staggered[D]` is its involution
    (cell centres ↔ faces).
  - `Slot[R]`: the dependent neighbor-slot space of relation `R`.
  - `Label`: finite spaces with no topology, such as tracers, ensemble members or
    `LsqUnk`.

  Identity is the qualified Python name, as in connectivities as types.
- **Relations**: `class C2E(Relation[Cell, Edge])` is the static half of a table's
  type. Relations form an **algebra**:
  - transpose `C2E.T`;
  - composition `C2E @ E2C`;
  - reflexive closure, for the "O" variants.

  A term's *reach* (its number of incidence hops) is known statically. A relation
  is **partial** when some rows have no valid neighbor because of the domain
  boundary, e.g. `E2C` on a LAM mesh. Partiality is a fact of the bound table,
  declared or taken from the binding like the counts, and it forces a guard on every
  read.
- **Chains and reduction order**: slot spaces record their parent (the dependency
  forest of [[personal/havogt/dependent-local-dimensions/dependent-local-dimensions|dependent local dimensions]]).
  Only a leaf may be reduced, so `ef(V2E)(E2V)` is well typed and a wrong reduction
  order is a type error. `Slot[R]` is spelled by subscription so that code generic
  over relations can name it.
- **Incidence spaces are Σ types**: `Pairs[C2E]` is the flattened (cell,
  edge-of-cell) space, ICON's `CE`. A sparse field has two isomorphic addressings:
  - `Dims[Cell, Slot[C2E]]`: padded, CSR-like;
  - `Dims[Pairs[C2E]]`: flattened, COO-like.

  The bijection between them is derived from the table. Relations into a flattened
  space (`C2CE`) are therefore derived, and agree with `C2E`'s slot order by
  construction.
- **The local binding** `bind_local(...)` holds:
  - local tables, including halo rows;
  - local extents;
  - the injection `ι_r` and the owned set;
  - boundary regions pulled back to local index sets (possibly non-contiguous);
  - exchange patterns.

  It is validated once (§8) and immutable, so the evidence cannot go stale. Its
  brand is a pair: the identity of the **logical** mesh (equal on all ranks), plus
  the binding itself.
- **Halo depth is derived, not written.** The binding computes the depth a program
  needs from the reach of the relation terms it uses; configuration may override it.
- **Halo validity is tracked state.** Each field records how far beyond `Owned` it is
  valid. Reading through a relation whose reach exceeds that is an error, or
  triggers an exchange under an `auto` policy. A typestate can make "owned-valid" vs
  "halo-valid" static; the depth itself cannot be.
- **Domains in source** are `Owned(Cell)` combined with semantic regions. Halo
  levels cannot be named. The backend may extend a computation into the halo instead
  of exchanging, because halo values are caches by definition.
- **Exchange points are explicit and depth-free**: `exchange(f)` means "make `f`
  valid for its consumers". **Global reductions are explicit**: `global_sum(f)`.

### Boundary regions and halo lines

- **Boundary regions** are declared once on entity spaces of the logical mesh, as
  refinements. Each rank's binding pulls them back to local index sets.
- **Halos** exist in the local index space, so their *existence* is visible (anything
  outside `Owned`); their *depth* is not.
- A skip value from halo truncation can only be read outside the halo-valid region,
  which validity tracking rejects.

### Sketch

```python
class Cell(Entity): ...
class Edge(Entity): ...
class K(Axis): ...                                      # Staggered[K]: the half levels

class C2E(Relation[Cell, Edge], max_neighbors=3): ...
class E2C(Relation[Edge, Cell], max_neighbors=2): ...   # partial on a LAM mesh: boundary edges have one cell
C2E2C = C2E @ E2C                                       # a term: its reach (2 hops) is known statically
CE = Pairs[C2E]                                         # Σ c. Fin(deg c): the flattened cell-edge space

class LateralBoundary(BoundaryRegion[Cell]): ...        # semantic: a refinement of the logical Cell space

@field_operator
def div(vn: Field[Dims[Edge, K], float], w: Field[Dims[Cell, Slot[C2E]], float]) -> Field[Dims[Cell, K], float]:
    return neighbor_sum(vn(C2E) * w, axis=Slot[C2E])

@local_program
def step(vn: Field[Dims[Edge, K], float], w: Field[Dims[Cell, Slot[C2E]], float], out: Field[Dims[Cell, K], float]) -> float:
    div(vn, w, out=out, domain=Owned(Cell) & ~LateralBoundary.levels(1, 4))   # never HALO_LEVEL_2
    exchange(out)                                       # no depth: the binding knows the reach of later reads
    return global_sum(out)                              # explicit allreduce

local = bind_local(tables, extents, owned=owned, global_index=gidx, regions={LateralBoundary: refin_ctrl})
step(vn, w, out, binding=local)                         # halo depth: configured, or derived from reach
```

### Dependent-typing reading

- A program is Π over the local mesh `g_r`. The mesh is a **parameter** (§7), so
  types are `∀g` and never claim "same type ⇒ same mesh".
- The binding's brand restores mesh identity at run time.
- The simulation condition against the logical `g` is the programmer's obligation,
  partly discharged by validity tracking, which is evidence that the halo cache
  agrees with the owners.
- Σ is explicit as `Pairs[R]`.
- `Fin<deg>` depends on `R` nominally and on `v` through padding.
- Boundary regions are refinements on `g`, transported by `ι_r`.

### Relation to existing notes and the ecosystem

- **Keeps from [[personal/egparedes/connectivities-as-types/connectivities-as-types|connectivities as types]]**:
  nominal dimensions with qualified-name identity, `Staggered[D]`, the split between
  a declaration and its table, the boundary check.
- **Changes**:
  - `V2E.Local` becomes `Slot[R]` (friendlier to generics, at the cost of interning);
  - the owner/sharer rules for local dimensions become `Pairs[R]`;
  - the `offset_provider` dict becomes a `Binding` object.
- **Adopts** the forest of dependent local dimensions and the halo-validity dataflow
  of the mesh proposal.
- **Ecosystem**: closest to xarray and coordax (dimensions plus data, topology as
  data) and to Atlas, PETSc and MPI practice.

### Strengths and costs

**Strengths:**
- Matches how ICON and ICON4Py run today, so migration and Fortran interop are
  cheapest.
- Experts keep control over communication.
- The compiler needs no global index bookkeeping.
- One compiled program serves many meshes.
- Code is generic over meshes for free.

**Costs:**
- Halo depth stays out of source, but exchange points and `Owned` do appear in it,
  so the decomposition is not fully hidden.
- Correctness of the distribution rests on the programmer.
- A forgotten `global_sum` silently gives a rank-local result.
- The mesh never appears as a whole, so geometry and regions are loose parts of the
  binding.

## Proposal B — Logical mesh object plus a runtime decomposition

> **Semantics: global.** Programs express the computation on the logical mesh. The
> decomposition is a runtime attribute of the mesh, chosen by configuration or by
> the backend, and **cannot be named in a field operator or program**. A single-node
> run and an N-rank run use identical source.

### Thesis

The mesh is the central domain object. A field is a pair of a logical mesh and data
on one location of it, so it always knows its mesh. Boundary regions belong to the
mesh; halos belong to a separate `Distribution`. The backend derives halos, places
exchanges, chooses redundant computation instead of an exchange when that is
cheaper, and splits global reductions.

### Core concepts

- **Schema**: entity families are generic over a schema class, using the UGRID
  vocabulary: `Face[M]`, `Edge[M]`, `Node[M]`. `Face[Icon]` materializes an ordinary
  nominal entity class, interned like `Staggered[D]`. A schema brands its entities
  statically, so `Face[Icon]` and `Face[RadiationGrid]` are different types.
  Schemas also declare axes (vertical levels, with their staggered half levels),
  relations and boundary-region kinds.
- **Roles**: `class Parent(Icon)` and `class Child(Icon)` separate several meshes of
  one schema in one process (ICON's nests and its reduced radiation grid). A
  transfer between meshes is a relation between entities of different roles:
  `Relation[Face[Child], Face[Parent]]`.
- **The mesh instance is the logical mesh**: CF's domain construct, made executable.
  It is validated once at construction (§8) and immutable. It holds:
  - global relations, which need never be materialized globally, with their neighbor
    order pinned;
  - geometry: coordinates, areas, metric factors (CF bounds and cell measures);
  - axis extents;
  - **boundary regions**: lateral boundary levels, nudging zone, vertical anchors
    (top, surface);
  - partiality of relations at the boundary.

  Factories build it from UGRID or ICON grid files.
- **`Distribution`** is a separate object: partition, local meshes, the ownership
  triple from Atlas, halo depth per entity, exchange patterns (GHEX), execution
  classes. It has two modes:
  - **derived**: the backend computes the halo depth from the compiled programs'
    reach;
  - **given**: the ICON Fortran granule path, where ICON's decomposition and halo
    configuration are imported through a translation layer that also absorbs nproma
    padding, 1-based tables and `rl_start`/`rl_end`.

  Either way it is configuration, never model source.
- **Fields** carry `.mesh` and are allocated through it. Their brand is the identity
  of the **logical** mesh, so changing the halo depth changes neither the brand nor
  the source. Storage is per rank; type and meaning are global. A program finds its
  mesh from its field arguments and requires exactly one instance per role.
- **Domains in source** are semantic regions only, e.g.
  `grid.region(Face[Icon]) - LateralBoundary.levels(1, 4)`. There is no
  `INTERIOR`/`HALO` vocabulary. Reductions over entity dimensions are global by
  definition.
- **The compile key** is the mesh *signature* (schema, table types, region layout)
  plus the distribution's parameters where they affect the generated loops, never the
  instance. One compiled program serves many meshes.

### Boundary regions and halo lines

They are separated by object:
- boundary regions are part of `Mesh`, nameable and typed;
- halos are part of `Distribution`, invisible in source, derived or configured.

An unguarded read of a partial relation is a frontend error. Truncation skips are
made unobservable by domain inference plus halo derivation. ICON's local memory
order (by `refin_ctrl`, halos appended) becomes a layout detail of the given mode's
translation layer.

### Sketch

```python
class Icon(Mesh2D): ...                                     # schema: Face/Edge/Node[Icon] are its entity spaces
class C2E(Relation[Face[Icon], Edge[Icon]], max_neighbors=3): ...
class LateralBoundary(BoundaryRegion[Face[Icon]]): ...

def div[M: Mesh2D](vn: Field[Dims[Edge[M], K], float]) -> Field[Dims[Face[M], K], float]: ...

grid = Icon.from_ugrid("icon_grid.nc", distribution=config.distribution)   # Auto(), or given by ICON
vn = grid.zeros(Edge[Icon], K)                              # vn.mesh is grid (logical); storage is per rank
div(vn, out=out, domain=grid.region(Face[Icon]) - LateralBoundary.levels(1, 4))
mass = sum_over(out, axis=Face[Icon])                       # global by semantics; allreduce is inserted
```

### Dependent-typing reading

- A field is a Σ pair over the logical `g`, and the mesh is an **index** at two
  levels:
  - the schema or role is a static brand (a restricted `template<Graph G>`);
  - the instance is a runtime index.
- The mesh instance is the appendix's `GraphPackage`.
- The distribution is the family `(g_r, ι_r)`. The backend constructs the evidence
  for the simulation condition (domain inference plus exchange placement), so the
  guarantee moves from the programmer to the system.
- Boundary regions are refinement types on `g`. A partial relation is
  `Option`-valued in its Π type, which forces its `None` case to be handled.

### Relation to existing notes and the ecosystem

- **Takes from [[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos|the mesh proposal]]**
  the V/E/C vocabulary, compositions with reach, ownership and execution classes,
  and automatic exchange placement. Here they are split between `Mesh` and
  `Distribution`.
- **Resolves** open question 2 of connectivities as types (the multi-table case) by
  making a mesh identity, not a provider dict, the unit of binding.
- **Ecosystem**: the CF data model (a field is data plus a domain; data binds to a
  mesh by `mesh` and `location`), uxarray's `.uxgrid` but strict, Atlas's
  grid/mesh/function-space model, MPAS's closed set of tables, SGRID for structured
  axes.

### Strengths and costs

**Strengths:**
- Halo lines are fully out of source.
- Boundaries and halos are separated by construction.
- The same source runs on one node or on any decomposition.
- Automatic exchange placement becomes possible.
- One source of truth for extents, regions and geometry.
- Direct UGRID/CF input and output.

**Costs:**
- The backend must own domain inference *across* stencil and program boundaries,
  plus exchange placement: engineering on the scale of PSyclone.
- Distributed embedded execution needs a distributed runtime; embedded mode is
  likely single-node.
- The Fortran host path needs the translation layer.
- The schema risks becoming a god object.
- Code generic over meshes needs subscripted families and bounded overloads.
- Every field needs a mesh, so a `CartesianGrid` schema becomes mandatory even for
  test fields.

## Proposal C — Located fields over cell complexes

> **Semantics: global.** Programs express operators on the logical complex. The
> decomposition is invisible in source and handled as in B: a runtime
> `Distribution` of the complex, derived or given.

### Thesis

Every mesh is a cell complex. An unstructured horizontal complex is combined with
1-D axis complexes; a structured grid is a product of axes. A field's type is its
**location**, a cell class of that complex. Neighborhoods are the complex's signed
incidences, which the mesh realizes. Users write operators between locations; named
relations are the escape hatch. This is Proposal 3 of
[[personal/egparedes/discretization-independent-fd-syntax|the FD-syntax note]] made
the type model. Its distinctive content is the user surface and the boundary model;
it reuses B's split between logical mesh and distribution.

### Core concepts

- **Locations** come from two sources:
  - unstructured classes that carry a degree: `Vertex` (0), `Edge` (1), `Cell` (2)
    (UGRID's node, edge, face);
  - per-axis classes `D` and `Staggered[D]` (SGRID's bit vectors).

  Products are written `At[Edge, Half]`. Dimension tuples become derived layout. On a
  structured 2-D grid an `Edge` cochain is a **pair** of fields, `(I½, J)` and
  `(I, J½)`; that is the exterior-calculus answer, and the reason the FD-syntax note
  keeps `EDGE_N`/`EDGE_T` apart.
- **Canonical incidences come with the complex**, including orientation signs. The
  mesh records the enumeration and orientation; that needs UGRID's anticlockwise face
  order plus each edge's own node order.
- **Operators**:
  - `d`, the coboundary: pure topology, typed location to location;
  - `star`, the Hodge star: metric, supplied by the mesh;
  - reductions over incidences, `sum_over(vn, Edge → Cell)` (the same shape as
    uxarray's `topological_mean(destination="face")`);
  - reconstructions `to[Loc]`.

  Sparse fields live on incidences (`Field[At[Incidence[Cell, Edge]]]`, the Σ type);
  slot dimensions disappear from user types.
- **The boundary is first-class topology**:
  - A bounded complex has a **boundary subcomplex `∂g`**, and incidences are partial
    exactly at `∂g`.
  - ICON's `refin_ctrl` is a **distance-to-boundary filtration** of `g`, so the
    lateral boundary levels and the nudging zone are subcomplexes `Collar[k]`.
  - Boundary data lives on boundary locations (`At[Boundary[Edge], Full]`), so a
    boundary condition appears in an operator's signature.
- **The mesh kind is a compile-time index**: a product complex lowers to index offsets
  with no tables, a general complex lowers to tables. Weight strategies are mesh
  properties. Column primitives (scan, tridiagonal solve) stay orthogonal on the
  vertical axis complex.
- **Halos**: none in the type language. `d` and `star` are global operators; their
  reach (number of incidence hops) is what the backend uses to derive halo depth.

### Boundary regions and halo lines

- Boundaries are topology: `∂g`, collars, boundary-located fields. A boundary
  condition has a mathematical status (relative chains, or ghost cells on `∂g`).
- Halos are the distribution's business; nothing in the surface can express them.

### Sketch

```python
@field_operator
def div(vn: Field[At[Edge, Full], float], flux_in: Field[At[Boundary[Edge], Full], float]) -> Field[At[Cell, Full], float]:
    return star(d(star(vn), boundary=flux_in))          # the boundary condition is in the signature

@field_operator
def grad_n(p: Field[At[Cell, Full], float]) -> Field[At[Edge, Full], float]:
    return d(p) / dual_edge_length                       # same source on an ICON grid or a Cartesian box
```

### Dependent-typing reading

- The complex is `g`. Location types are the families `Cells<g, k>`, indexed by a
  statically known degree.
- `Incidence<g>` (§4) is a type.
- The §3 refinement (an exact neighbor list) holds by construction, because the
  incidence *is* the boundary map (§6's "inverted presentation").
- `∂g` and the collars are refinements defined by the topology; partiality happens
  exactly on `∂g`.
- `d∘d = 0` is a structural fact about the operators.
- The mesh kind is a compile-time index; the simulation condition is discharged by
  the backend, as in B.

### Relation to existing notes and the ecosystem

- **Makes the type model** out of the location typing of the FD-syntax note, with
  `Staggered[D]` from connectivities as types as the per-axis cell class.
- **Unifies** UGRID (degrees) and SGRID (bit vectors) as one complex model, which is
  the structural observation of
  [[personal/egparedes/connectivities-as-types/connectivities-as-types_conventions|the conventions appendix]].
- **Ecosystem**: CF's `cell_methods` point/cell distinction is the degree of a
  location; Firedrake/UFL function spaces, Decapodes, PyDEC and TRiSK are the
  precedents.

### Strengths and costs

**Strengths:**
- Halo lines are fully out of source.
- Boundary conditions are typed and structural.
- The same source runs on structured and unstructured meshes.
- Conservation and the mimetic identities are structural.
- The smallest user vocabulary.

**Costs:**
- All of B's backend burden.
- ICON's order-dependent, non-canonical stencils (`E2C2V` for RBF, `C2E2C2E2C`) need
  the escape hatch heavily.
- Only the compatible staggerings (C, D) get the guarantees; PMAP's collocated A-grid
  falls outside them.
- The location algebra needs overloads generated per degree.
- The mesh's geometry pipeline becomes part of the language contract.

## Comparison

| | A | B | C |
| --- | --- | --- | --- |
| **Semantics** | **local (SPMD)** | **global** | **global** |
| Center of gravity | algebra of index spaces | data model | user surface |
| Decomposition visible in source | `Owned`, exchange points, `global_sum` | nothing | nothing |
| Halo depth | configured, or derived from reach | derived or given; never source | as B |
| Exchange placement | explicit depth-free points, or per-program `auto` | automatic | automatic |
| Simulation condition discharged by | the programmer (with validity tracking) | the backend | the backend |
| Boundary regions | declared on entity spaces, pulled back per rank | owned by the mesh, named | `∂g`, collars, boundary-located fields |
| Partial relations | declared or bound; reads must be guarded | frontend error if unguarded | partial exactly on `∂g` |
| Global reductions | explicit `global_sum` | implicit | implicit |
| Local dimensions | `Slot[R]` forest plus `Pairs[R]` | as A, scoped to a schema | hidden; incidence fields |
| Brand | logical mesh plus local binding | logical mesh (distribution excluded) | as B |
| Several meshes in one process | runtime brands | typed roles plus relations between meshes | as B |
| Same source, structured and unstructured | no | no | yes |
| ICON Fortran host integration | native | translation layer, given mode | as B |
| Halo lines out of source | partly: depth only | fully | fully |
| Distance from the implemented stack | small | medium | large |
| Main risk | distribution correctness on the programmer | backend burden; god-object schema | escape hatches dominate ICON |

## Layering: the three proposals as levels of abstraction

The three proposals answer different questions: what the user writes (C), what a
mesh and a field are (B), and what the compiler and runtime operate on (A). They can
therefore coexist as **levels of one stack**, each lowering to the one below. Each
level is useful on its own, and none forces the levels above it.

```text
 L3  surface    C: located fields, mesh-invariant operators              global semantics
       │  canonical incidences → named relations of the schema;
       │  star / weights → geometry fields; Boundary[L] → boundary regions
 L2  model      B: logical mesh, fields with a mesh, semantic regions     global semantics
       │  + Distribution: one local binding per rank; regions → Owned ∩ local index sets;
       │  inferred halo validity → exchanges or redundant computation; global sums → local + allreduce
 L1  core       A: index spaces, relations, local binding                 local semantics (SPMD)
       │  relations → bound tables; Slot / Pairs → local dimensions; exchange → GHEX calls
 L0  execution  GTIR; gtfn / DaCe / embedded backends; FieldData buffers
```

### What each lowering does

| From → to | Lowering steps | Obligation |
| --- | --- | --- |
| **L3 → L2** | 1. A pair of locations resolves to a canonical relation of the schema, with signs from the recorded orientation. 2. `star` and weights become geometry fields of the mesh instance. 3. `Boundary[L]` fields and collars become boundary regions plus `concat_where`. 4. A structured product complex lowers incidences to `D ± ½` shifts, and an `Edge` cochain to a tuple of fields. | The L2 program computes the same discrete operator. Checkable by diffing the generated code against hand-written ICON4Py stencils, as the FD-syntax note proposes. |
| **L2 → L1** | Given a `Distribution`: 1. one local binding per rank (local tables with halo rows, local extents, `ι_r`). 2. Semantic regions become local index sets intersected with `Owned`. 3. Domain inference yields the halo validity each read needs; the backend inserts an exchange or extends the producer's domain into the halo (redundant computation), by cost. 4. Global reductions become a local reduction plus an allreduce. 5. The field brand becomes the binding brand, keeping the logical-mesh part. | **The simulation condition**: every L1 rank program reproduces the L2 program on its owned entities. This is the contract to test: run the same L2 program under a trivial and a nontrivial distribution and compare bitwise, except global reductions. |
| **L1 → L0** | Relations become bound tables keyed by relation tag (today's offset providers); slot and incidence spaces become local dimensions; `exchange` becomes communication-library calls; programs become GTIR. | Today's toolchain contract. |

With a **trivial distribution** (one rank), L2 → L1 is almost the identity: the local
binding is the logical mesh and there are no exchanges. A global-semantics stack
therefore costs nothing on a single node and keeps today's code path.

### Rules that keep the levels consistent

1. **One vocabulary across levels.** Entities, axes and relations are declared once
   as L1 nominal classes. L2 schemas group them (`Face[Icon]` materializes an L1
   `Entity` class), and L3 locations map onto them per mesh (`At[Cell, Full]` ↦
   `Dims[Face[M], K]`). Nothing is renamed between levels, so diagnostics and
   generated code use one set of names.
2. **Semantics is a property of a program, not of a field operator.**
   - A field operator with no reduction over a distributed dimension is
     *semantics-neutral*: its value on an entity depends only on that entity's
     neighborhood, so it gives the same result under global semantics and per rank
     with valid halos.
   - One library of field operators therefore serves L1 and L2. Only programs
     (domains, communication, global reductions) are global (`@program`) or local
     (`@local_program`), and never both.
3. **Escape hatches point downward only.**
   - L3 code may use L2 named relations (ICON's ordered multi-hop stencils).
   - L2 code may **not** use L1's `Owned` or `exchange`; that would break the rule
     that halos stay out of source.
   - Code that needs local control is written as an explicit L1 local program.
4. **Brands are continuous.** The logical-mesh identity created at L2 is the same one
   the L1 binding carries, so mixing meshes is caught at whichever level the fields
   meet.
5. **Each level owns its own checks**:
   - L3: the location algebra (de Rham signatures);
   - L2: mesh identity, unguarded partial relations, boundary regions;
   - L1: chain and leaf rules, binding validation, halo validity;
   - L0: today's type deduction.

### Who writes at which level

| Who | Level | Why |
| --- | --- | --- |
| Scientists writing new dynamical-core code | L3, or L2 for operators with no canonical form | mesh-invariant, boundary-aware, halo-free source |
| Model infrastructure (grid managers, I/O, configuration) | L2 | builds meshes, regions and distributions from UGRID / ICON files |
| Performance experts; MPI host integration (the ICON Fortran granule path) | L1 | explicit local control, or a given decomposition |
| Backend developers | L0, and the lowering passes | — |

### Where existing work sits

- **L1**:
  - connectivities as types (implemented; it needs a binding object, `Slot`/`Pairs`,
    partial relations, static relations);
  - dependent local dimensions;
  - dimension genericity;
  - the scan redesign's vertical windows.
- **L0 / L1**: the field data protocol (storage).
- **L2**:
  - the mesh proposal (the `Distribution`, plus the exchange-placement pass as the
    L2 → L1 lowering);
  - the boundary-condition syntax (semantic regions in `concat_where`);
  - the scan redesign's anchors (vertical boundary regions).
- **L3**: the FD-syntax note.
- **Today's gt4py is an L1/L0 system with local semantics**, with ICON4Py doing the
  L2 work by hand. The stack adds levels above it; it replaces nothing below.

### An incremental path

Each step is useful on its own:

1. **L1 core**: a binding object, `Slot`/`Pairs`, partial relations, static
   relations, halo-validity tracking.
2. **L2 on one node**: logical mesh, fields that know their mesh, semantic boundary
   regions, a trivial distribution. Regions replace zone index arithmetic for
   boundaries.
3. **L2, given distribution**: import ICON's decomposition with configured halo
   depth, and lower to L1 with exchanges at program boundaries.
4. **L2, derived distribution**: domain inference across programs, plus automatic
   exchange placement and redundant computation.
5. **L3 library**: emits L2 code, validated against hand-written ICON4Py stencils.

### Risks of layering

- **Two semantics in one system** need a hard boundary. The program-level flag (rule
  2) must be checked, not conventional; otherwise local idioms leak into global code.
- **Abstraction can hide ordering.** L3's canonical incidences must expose which
  enumeration they use, since ICON's results depend on neighbor order.
- **Each lowering is a correctness obligation** with its own test strategy; the
  L2 → L1 simulation condition is the most demanding.
- **The levels must not drift apart in vocabulary** (rule 1). A level that invents
  its own names for the same entity reintroduces the string-agreement problem that
  connectivities as types removed.

## Cross-cutting decisions

These hold whichever direction is chosen.

1. **Two kinds of region, and only two.**
   - **Semantic** regions (boundary levels, nudging, vertical anchors) are defined on
     the logical mesh and nameable in source.
   - **Decomposition** regions (owned, halo levels, execution classes) are nameable
     only under local semantics, and then only `Owned`.
2. **Coordinates are logical for meaning and local for storage.** The decomposition
   owns the mapping between them. This qualifies the "absolute coordinates" rule of
   [[personal/havogt/field-data-protocol/field-data-protocol|the field data protocol]]:
   absolute in the *logical* index space.
3. **Skip values have three owners**: arity in the relation's table type, boundary
   partiality in the logical mesh, truncation in the decomposition (never
   observable).
4. **Mesh identity never includes the decomposition.** Changing the halo depth is not
   a new mesh.
5. **Bitwise reproducibility under decomposition** is a backend requirement, except
   for global reductions.
6. **One meaning for subscripting a dimension.** Today `D[...]` is wanted for regions
   (`K[1:nlev]`), slot indices (`V2E[1]`) and type construction (`Staggered[K]`).
7. **Table entries index the codomain's logical extent**, with the index origin
   recorded or rebased on load (as uxarray's loader does).
8. **Some relations have tables known at compile time** (the scan redesign's
   vertical windows, Cartesian offsets). They need a binding mode of their own.
9. **"Vertical" and "distributed" are facts of the decomposition, not dimension
   kinds.** The vertical is never distributed, but its top and surface are semantic
   boundaries.
10. **Dimension genericity (`TypeVar`/`TypeVarTuple` over dimensions) is a
    prerequisite** for all three. Without it, staggering costs 14 operators where 4
    generic ones suffice
    ([[personal/havogt/dimension-generic-fields/dimension-generic-fields|generic dimensions]] §4.5).

## Open questions

1. **Global or local semantics for the DSL?** Keeping halo lines out of source points
   to global semantics; the cost is B's backend burden (domain inference across
   programs, automatic exchange placement).
2. **If global, how public is the L1 local view?** It could be internal only, or an
   explicit escape hatch for experts and for the Fortran host path.
3. **Should several meshes in one process be distinguished statically** (B's roles),
   or is a runtime brand enough (A)?
4. **Should users name connectivities** (A, B), or should neighborhoods follow from
   locations (C)?
5. **Closed or open vocabulary of entities and relations?**
   - Closed: the mesh proposal and MPAS argue for V/E/C and canonical relations.
   - Open: ICON4Py's ordered multi-hop tables argue for an algebra.
6. **Halo-validity typestate**: is a static owned-valid / halo-valid distinction
   worth its annotation cost in A?

## Relation to other notes and conflicts

- [[personal/egparedes/connectivities-as-types/connectivities-as-types|Connectivities as types]]
  provides the L1 base. **Conflicts**:
  - `V2E.Local` (nested, owned) vs `Slot[R]` (subscripted, generic-friendly), with
    `Pairs[R]` replacing the owner/sharer rules;
  - `offset_provider` vs a binding or mesh object;
  - its conventions appendix §5 lists "owned versus halo entities" as an ICON need
    for UGRID's `location_index_set`, which treats a decomposition concept as a
    semantic subset;
  - its open question 2 counts "halo variants" among the multi-table cases, whereas
    here the mesh identity excludes the decomposition.
- [[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos|A mesh concept with first-class halos]]:
  compatible with B, where it becomes the `Distribution` plus the L2 → L1 lowering;
  its skip-value split is adopted. **Conflict**: it treats sparse data fields over
  connectivity dimensions as "field-type, not mesh, concerns". A slot-addressed value
  is only meaningful relative to one table's enumeration (dependent-typing appendix
  §6), so sparse fields depend on the mesh more than dense ones do.
- [[personal/havogt/boundary-condition-syntax/boundary-condition-syntax|Boundary-condition syntax]]:
  its horizontal conditions use ICON4Py index symbols that mix boundary and halo
  bounds (`interior_idx <= CellDim < halo_idx`, `end_edge_halo`). Under B and C,
  only the semantic half is expressible. Its `K[0]` region literal competes for
  `D[...]` (decision 6).
- [[personal/havogt/scan-redesign/scan-redesign|Scan redesign]]: its anchors are
  vertical boundary regions and the vertical is never distributed; both agree. Its
  `KDim[:-1]` examples conflict with logical absolute coordinates. Its
  `WindowOffset` is a static relation (decision 8) written in the removed
  `FieldOffset` style.
- [[personal/havogt/field-data-protocol/field-data-protocol|Field data protocol]]:
  "support ⊇ domain" is how a halo looks at L0 (data beyond the compute domain,
  private like the origin). Its absolute coordinates are qualified by decision 2.
- [[personal/havogt/dependent-local-dimensions/dependent-local-dimensions|Dependent local dimensions]]:
  A adopts its forest. Its `Local[C]` spelling matches `Slot[R]`, not
  connectivities-as-types' `V2E.Local`.
- [[personal/havogt/dimension-generic-fields/dimension-generic-fields|Generic dimensions and staggering]]:
  dimension genericity is a prerequisite (decision 10). Its `Dual[X]` name is
  reserved here for the full Hodge dual (C's `star`), as the connectivities
  staggering appendix argues.
- [[personal/egparedes/discretization-independent-fd-syntax|Discretization-independent FD syntax]]:
  C is its Proposal 3 made the type model, and L3 is its front-end. Its "halo
  exchange is never written by the user" is global semantics.
- **Open PRs**:
  - [GridTools/gt4py_knowledge#31](https://github.com/GridTools/gt4py_knowledge/pull/31)
    (a default domain plus per-output restrictions) restricts outputs with zone
    symbols that mix nudging and local/halo bounds (`start_edge_nudging_level_2`,
    `end_edge_local`). Under B and C these become semantic regions only.
  - [GridTools/gt4py_knowledge#25](https://github.com/GridTools/gt4py_knowledge/pull/25)
    (JAX connectivities) adds a compile-time bounds descriptor of a runtime table, an
    instance of the stage split above.

## Method and sources

The proposals were developed in a design session that:
1. read the knowledge-base notes on dimensions, meshes, domains and fields in full;
2. compared connectivities as types with xarray, coordax and uxarray, at source and
   with probes;
3. traced where "a field should know its mesh" comes from;
4. surveyed the UGRID, SGRID and CF-1.13 conventions and the CF data model;
5. separated halo lines (implementation) from boundary regions (semantics).

Inputs, versions and verified claims are in
[[personal/egparedes/mesh-field-design-space/mesh-field-design-space_research|the research appendix]].
