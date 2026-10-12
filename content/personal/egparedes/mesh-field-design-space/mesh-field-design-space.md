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
> - **C — Cochain-typed fields over cell complexes.** *Global semantics*. A field is
>   typed by its cochain space: primal or dual side, a degree, and an integrated or
>   sampled representation. The operators are the coboundary and the Hodge star.
>   Boundary conditions are typed operations on the boundary subcomplex.
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
SlotPos<g,r> = Σ v : A<g>. Fin<deg_r(v)>           -- a slot position: the flattened, CSR-like sparse space (Σ)
Incid<g,r>   = Σ v : A<g>. { e : B<g> | e ∈ r(v) }  -- an incidence (§4): a (v, e) pair, the COO-like addressing
program      : (g : Mesh) → Field<g,E,T> → Field<g,V,T>   -- Π over the mesh
```

The enumeration (§6) is the bijection between slot positions and incidences.

### What Python can keep

Python types can mention literal constants (`Literal[3]`) but never runtime values,
and they have no type-level arithmetic: neither `g.n_A` nor `degree(g, v)` can
appear in a type. Each fact about `g` must therefore be assigned to a stage:

1. **Static**: the type checker sees it. Only names and nominal identity.
2. **Compile time**: the frontend specializes on it, for example counts, dtype and
   presence of skip values. This is the two-level pattern that dtype generics and
   connectivities-as-types already use.
3. **Run time**: tables, extents, geometry, decomposition.

Limits that shape every design (verified by probes elsewhere in this knowledge base):
- nominal classes are the only usable brand;
- integer type arguments are limited to `Literal` constants, with no arithmetic on
  them;
- there are no type-level functions, so changing a dimension inside `Dims[...]`
  needs one overload per rank and position (bounded code generation);
- `TypeVarTuple` is unbounded, and only one may appear per tuple;
- `M.Cell` with `M` a `TypeVar` yields no usable type: mypy rejects it, and pyright
  silently degrades it to `Unknown`. Families that are generic over a mesh must
  therefore use subscription (`Cell[M]`), not nested attributes.

## Logical mesh and decomposition

### Two objects, never one

- **The logical mesh `g`**: the global entity sets, relations, geometry and
  **boundary regions**. A boundary region is a refinement of a global entity space,
  e.g. `{c : Cell<g> | refin_ctrl(c) = k}`: ICON's distance-to-lateral-boundary
  levels, the nudging zone, the model top and surface.
- **The decomposition `π` of `g`**, defined for every distributed entity space `A`
  (cells, edges, vertices alike):
  - ranks `r`, each with a local mesh `g_r`;
  - injections `ι_r : A<g_r> → A<g>` (the global index);
  - owned sets `owned_r(A) ⊆ A<g_r>` whose images partition `A<g>`;
  - **halos of finite depth `d`**: the entities reachable from the owned sets in at
    most `d` hops of the relations the programs use, minus the owned set. `d` is
    configured, or derived from the reach of those relations. An unbounded closure
    would be the whole connected mesh. Halos imported from a host (B's *given*
    mode) need not be such a closure: ICON4Py's second cell halo line completes the
    dual-cell hexagon around cut vertices;
  - exchange patterns.

| | Boundary regions | Halo lines |
| --- | --- | --- |
| What they are | subsets of the global entity sets at the edge of the model domain | cached copies of values owned by other ranks |
| Exist on a single node | yes | no |
| Change results | yes: external data enters there | no, apart from rounding |
| Depend on the partition | no: a property of the logical mesh | yes: a property of the decomposition |
| Decided by | the model author, in source | the backend or configuration, never the source |
| Effect on the index space | restricts the global index space | extends the local index space |

Boundary-region membership is consistent across ranks. A local entity belongs to a
boundary region `R` exactly when its global image does (`R_r = ι_r⁻¹(R)`), so all
replicas of one global entity agree on every rank. `refin_ctrl` is a global property.
The local sets `R_r` themselves differ, since each lives in its own local index space.
A halo has no type-level meaning: it is a cache whose validity the implementation
tracks.

### Three causes of skip values, three owners

The three causes are those identified by
[[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos|the mesh proposal]].

| Cause | Owner | Visible in source |
| --- | --- | --- |
| ragged arity (pentagons) | the relation's table type: `max_neighbors`, presence of skip values | yes |
| boundary partiality: a fixed-width table slot with no neighbor at the LAM lateral boundary (`E2C` on a boundary edge), or a stencil that reaches outside the domain (`C2E2C` next to the boundary) | the logical mesh: a **partial relation** (declared, or taken from the bound table). The incidence itself is complete: a boundary edge simply has one cell | yes: reads must be guarded by a boundary condition or a domain that excludes the boundary |
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
- Per-entity results must be **bitwise identical under any decomposition**. This
  rests on two premises:
  - neighbor reductions use the table order on every rank;
  - generated code gives every entity the same instruction sequence, regardless of
    its position or of the compute-domain extent (no remainder-loop or
    FMA-contraction differences).

  Recomputing a value in the halo then equals exchanging it.
- **Reductions over a distributed dimension must themselves be reproducible**:
  independent of the partition, e.g. by a fixed reduction tree over global indices,
  or by exact or binned summation. Otherwise their rounding differences propagate to
  every result that depends on them, including control decisions (`mass > 0`). In
  that case bitwise identity holds only for computations that do not depend on such
  a reduction.

### Design axes

| Axis | Choices explored |
| --- | --- |
| Semantics of a program | local (A) / global (B, C) |
| Where the mesh lives (§7) | parameter of the call plus a runtime brand (A) / carried by every field (B) / inherited, plus mesh *kind* at compile time (C) |
| User vocabulary | index spaces and relations (A) / mesh schema, entities and regions (B) / cochain spaces and operators (C) |
| Local dimensions | dependent slots `Slot[R, P]` and slot-flattened spaces (A, B) / hidden behind cochain operators (C) |
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
  - `Slot[R, P]`: the dependent neighbor-slot space of relation `R` under parent `P`
    (the root dimension, or another slot).
  - `Label`: finite spaces with no topology, such as tracers, ensemble members or
    `LsqUnk`.

  Identity is the qualified Python name, as in connectivities as types.
- **Relations**: `class C2E(Relation[Cell, Edge])` is the static half of a table's
  type. A relation carries two things that must not be conflated:
  - **Reach**: a footprint algebra with set semantics (transpose, composition,
    union, reflexive closure), used for halo depth and domain inference. A term's
    reach is known statically.
  - **An ordered stencil**: the materialized table, which fixes origin exclusion,
    deduplication and neighbor order. ICON's `C2E2C` lists a cell's three
    edge-neighbors without the cell itself, `C2E2CO` prepends it, and ICON4Py
    permutes some neighbors explicitly. The path composition `C2E @ E2C` would list
    the cell itself three times.

  A named relation is therefore a table *annotated* with its footprint term, e.g.
  `class C2E2C(Relation[Cell, Cell], reach=C2E @ E2C)`. Derived tables (transposes,
  compositions) need an explicit ordering rule.

  A relation is **partial** when some rows have no neighbor in a fixed-width slot
  because of the domain boundary, e.g. `E2C` on a LAM mesh. Partiality is a fact of
  the bound table, declared or taken from the binding like the counts, and it forces
  a guard on every read.
- **Chains and reduction order**: slot spaces name their parent, as in the
  dependency forest of [[personal/havogt/dependent-local-dimensions/dependent-local-dimensions|dependent local dimensions]]
  (`Local[C, P]` there).
  - `vf(V2V)` has `Slot[V2V, Vertex]`.
  - `vf(V2V)(V2V)` has `Slot[V2V, Vertex]` and `Slot[V2V, Slot[V2V, Vertex]]`,
    because the earlier hop is re-parented under the new one.
  - Only a leaf may be reduced, so `ef(V2E)(E2V)` is well typed and a wrong
    reduction order is a type error.
  - Re-parenting is a type-level substitution, expressed by bounded overloads per
    forest shape (that note's prototype verifies depth 2).
  - `Slot[R, P]` is spelled by subscription so that code generic over relations can
    name it.
- **Slot-flattened spaces are Σ types**: `Pairs[C2E]` = Σ c. Fin(deg c) is the
  flattened (cell, slot) space, ICON's `CE`. A sparse field has two isomorphic
  addressings:
  - `Dims[Cell, Slot[C2E, Cell]]`: padded, CSR-like;
  - `Dims[Pairs[C2E]]`: flattened.

  The enumeration (§6) maps both to the incidence pairs. The bijection is derived
  from the table, so relations into a flattened space (`C2CE`) are derived too, and
  agree with `C2E`'s slot order by construction.
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
- **Validity is tracked as a region per field, not as a depth.**
  - A write on domain `D` makes the written entries valid; other entries keep
    whatever validity they had.
  - `exchange` makes halo replicas valid where their owners are valid. It only
    copies, so owned entries that were never written stay invalid.
  - A read through relation `R` on domain `D'` requires `R(D')` ⊆ valid. A violation
    is an error, or triggers an exchange under an `auto` policy.
  - A single depth cannot describe partial writes such as `Owned ∩ interior`. A
    typestate can mark owned-valid vs halo-valid statically; the regions cannot be
    static.
- **Domains in source** are `Owned(Cell)` combined with semantic regions. Halo
  levels cannot be named. The backend may extend a computation into the halo instead
  of exchanging, because halo values are caches by definition.
- **Exchange points are explicit and depth-free**: `exchange(f)` means "make `f`
  valid for its consumers".
- **Global reductions are explicit and program-level only**: `global_sum(f, domain=d)`
  with `d ⊆ Owned`. Each global entity is counted exactly once, and only the
  entities in `d` are read, so `d` must not exceed what was written. Field operators
  never reduce over a distributed dimension (layering rule 2).

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
class C2E2C(Relation[Cell, Cell], max_neighbors=3, reach=C2E @ E2C): ...   # table pins order, excludes the cell itself
CE = Pairs[C2E]                                         # Σ c. Fin(deg c): the flattened cell-slot space

class LateralBoundary(BoundaryRegion[Cell]): ...        # semantic: a refinement of the logical Cell space

@field_operator
def div(vn: Field[Dims[Edge, K], float], w: Field[Dims[Cell, Slot[C2E, Cell]], float]) -> Field[Dims[Cell, K], float]:
    return neighbor_sum(vn(C2E) * w, axis=Slot[C2E, Cell])

@local_program
def step(vn: Field[Dims[Edge, K], float], w: Field[Dims[Cell, Slot[C2E, Cell]], float], out: Field[Dims[Cell, K], float]) -> float:
    interior = Owned(Cell) & ~LateralBoundary.levels(1, 4)   # owned + semantic; never HALO_LEVEL_2
    div(vn, w, out=out, domain=interior)
    exchange(out)                                       # no depth: the binding knows the reach of later reads
    return global_sum(out, domain=interior)             # explicit allreduce over what was written

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
  - `V2E.Local` becomes `Slot[R, P]` (it carries the parent the forest needs, and is
    friendlier to generics, at the cost of interning);
  - the owner/sharer rules for local dimensions become `Pairs[R]`;
  - the `offset_provider` dict becomes a `Binding` object.
- **Adopts** the forest of dependent local dimensions. **Generalizes** the mesh
  proposal's per-field valid-depth to valid regions.
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

- **Schema**: entity families are generic over a schema class: `Cell[M]`, `Edge[M]`,
  `Vertex[M]` (UGRID's face, edge and node). These are the same names A and C use,
  so nothing is renamed between levels. `Cell[Icon]` materializes an ordinary
  nominal entity class, interned like `Staggered[D]`. A schema brands its entities
  statically, so `Cell[Icon]` and `Cell[RadiationGrid]` are different types.
  Schemas also declare axes (vertical levels, with their staggered half levels),
  relations and boundary-region kinds.
- **Roles**: `class Parent(Icon)` and `class Child(Icon)` separate several meshes of
  one schema in one process (ICON's nests and its reduced radiation grid). A
  transfer between meshes is a relation between entities of different roles:
  `Relation[Cell[Child], Cell[Parent]]`.
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
  `grid.region(Cell[Icon]) - LateralBoundary.levels(1, 4)`. There is no
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
class Icon(Mesh2D): ...                                     # schema: Cell/Edge/Vertex[Icon] are its entity spaces
class C2E(Relation[Cell[Icon], Edge[Icon]], max_neighbors=3): ...
class LateralBoundary(BoundaryRegion[Cell[Icon]]): ...

def div[M: Mesh2D](vn: Field[Dims[Edge[M], K], float]) -> Field[Dims[Cell[M], K], float]: ...

grid = Icon.from_ugrid("icon_grid.nc", distribution=config.distribution)   # Auto(), or given by ICON
vn = grid.zeros(Edge[Icon], K)                              # vn.mesh is grid (logical); storage is per rank
interior = grid.region(Cell[Icon]) - LateralBoundary.levels(1, 4)
div(vn, out=out, domain=interior)
mass = sum_over(out, axis=Cell[Icon], domain=interior)      # global by semantics; allreduce is inserted
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

## Proposal C — Cochain-typed fields over cell complexes

> **Semantics: global.** Programs express operators on the logical complex. The
> decomposition is invisible in source and handled as in B: a runtime
> `Distribution` of the complex, derived or given.

### Thesis

Every mesh is a cell complex. An unstructured horizontal complex is combined with
1-D axis complexes; a structured grid is a product of axes.

A field's type is a **cochain space**, not a bare location. It is a side (the primal
complex or its dual) with a degree, plus a **value representation**: integrated, or
sampled. Operators are the coboundary `d` (topology) and the Hodge star `star`
(metric), both supplied by the mesh. Named relations are the escape hatch.

This is Proposal 3 of
[[personal/egparedes/discretization-independent-fd-syntax|the FD-syntax note]] made
the type model. Its distinctive content is the user surface and the boundary model;
it reuses B's split between logical mesh and distribution.

### Core concepts

- **Primal and dual complex.**
  - The **primal** complex is the mesh as stored: `Vertex` (0), `Edge` (1), `Cell` (2),
    which are UGRID's node, edge and face.
  - Its **dual** has a dual k-cell for every primal (2−k)-cell: a dual vertex at each
    cell centre, a dual edge crossing each primal edge, a dual cell around each
    primal vertex.
  - Assumed: the horizontal complex is an oriented 2-manifold with boundary, whose
    orientation the mesh records. That needs UGRID's anticlockwise face order plus
    each edge's own node order.
- **Cochain spaces** are `Primal[k]` and `Dual[k]`, combined with axis classes as in
  `At[Dual[1], Full]`.
  - The storage location is *derived*: `Primal[k]` lives at primal k-cells, `Dual[k]`
    at primal (2−k)-cells. So `Dual[0]` lives at `Cell`, `Dual[1]` at `Edge`, and
    `Dual[2]` at `Vertex`.
  - Two fields stored at `Edge` can therefore have different types.
  - `Dual[1]` is notation. Python spells the degree as a `Literal`
    (`Dual[Literal[1]]`) or as one class per side and degree. Checked with mypy
    2.4.0 and pyright 1.1.414:
    - overloads of `d` infer `Dual[Literal[0]] → Dual[Literal[1]] → Dual[Literal[2]]`;
    - both checkers reject a `Primal[Literal[1]]` field where a `Dual[Literal[1]]`
      one is expected.
  - On a structured product complex, every degree also splits by per-axis bits. For
    example, a degree-1 cochain is a pair of fields, `(I½, J)` and `(I, J½)`. This
    direction split is orthogonal to the primal/dual split: `Primal[1]` and `Dual[1]`
    are each such a pair.
- **Value representation.**
  - **Integrated** values are the cochain proper: integrals over cells, which is what
    `d` acts on.
  - **Sampled** values (`Sampled[Dual[1]]`) are point values or cell means, which is
    what pointwise physics uses.
  - `integrate` and `sample` convert between the two by multiplying or dividing by
    the cell measure. For degree 0 they coincide.
- **Operators**:
  - `d : Primal[k] → Primal[k+1]` and `d : Dual[k] → Dual[k+1]`: signed incidence,
    pure topology;
  - `star : Primal[k] → Dual[2−k]` and back: metric ratios supplied by the mesh;
  - `integrate` / `sample`: metric;
  - reductions over named incidences, `sum_over(vn, Edge → Cell)` (the same shape as
    uxarray's `topological_mean(destination="face")`);
  - reconstructions `to[...]`.

  Sparse fields live on incidences (`Field[At[Incidence[Cell, Edge]]]`, the §4 Σ
  type). For example, ICON's `geofac_div` factors into a sign (topology) times a
  metric. Slot dimensions disappear from user types.
- **Staggering is a placement declaration** of physical variables onto cochain
  spaces (the FD-syntax note's §6.5 selector):
  - C-grid: pressure is `Dual[0]`, normal velocity is `Sampled[Dual[1]]`, vorticity
    is `Dual[2]`.
  - D-grid: tangential velocity is `Primal[1]`. A C-grid and a D-grid velocity are
    therefore different types, though both are stored at `Edge`.
  - A-grid: the collocated grid has no compatible placement (velocity is a
    vector-valued `Dual[0]`).
- **The boundary model separates three things**:
  - **Incidences are total.** On a bounded complex the signed incidence matrices and
    `d` are complete: a boundary edge simply has one cell. Partiality arises only in
    fixed-width table representations of cofaces (`E2C` with two slots), in stencils
    that reach exterior cells (`C2E2C` next to the boundary), and in the dual
    complex, whose cells around boundary vertices are truncated.
  - **Boundary conditions are typed operations.** Either they prescribe the
    restriction of a cochain to the boundary subcomplex `∂g`, as in
    `with_boundary(flux, flux_in)` with `flux_in: Field[At[Boundary[Primal[1]], Full]]`.
    Or they supply exterior ("ghost") values for stencils that cross `∂g`.
  - **Collars are semantic regions with an explicit construction.**
    - `Collar[k]` is the closure under faces of the top cells within `k`
      cell-adjacency hops of `∂g`. It is a subcomplex, and the collars are nested in
      `k`, forming a filtration.
    - ICON's `refin_ctrl` approximates the collars with per-kind thresholds (cells
      ≤ 4, edges ≤ 9, vertices ≤ 4), so mapping its levels onto collars needs a
      per-kind rule.
- **The mesh kind is a compile-time index**: a product complex lowers to index offsets
  with no tables, a general complex lowers to tables. Weight strategies are mesh
  properties. Column primitives (scan, tridiagonal solve) stay orthogonal on the
  vertical axis complex.
- **Halos**: none in the type language. `d` and `star` are global operators; their
  reach (number of incidence hops) is what the backend uses to derive halo depth.

### Boundary regions and halo lines

- Boundaries are topology: `∂g`, collars, boundary-located cochains. A boundary
  condition is a typed operation: a restriction to `∂g`, or an exterior extension.
- Halos are the distribution's business; nothing in the surface can express them.

### Sketch

```python
type P = At[Dual[0], Full]                   # cell-centred scalar, stored at Cell (degree 0: sampled = integrated)
type Vn = At[Sampled[Dual[1]], Full]         # C-grid normal velocity, point values, stored at Edge
type FluxIn = At[Boundary[Primal[1]], Full]  # prescribed flux on the boundary edges ∂g

@field_operator
def grad_n(p: Field[P, float]) -> Field[Vn, float]:
    return sample(d(p))                      # d: Dual[0] → Dual[1], i.e. p(c2) − p(c1); sample: ÷ dual edge length

@field_operator
def div(vn: Field[Vn, float], flux_in: Field[FluxIn, float]) -> Field[P, float]:
    flux = with_boundary(star(integrate(vn)), flux_in)   # Dual[1] →star Primal[1]: flux through primal edges
    return star(d(flux))                                 # Primal[1] →d Primal[2] →star Dual[0]: ÷ cell area
```

The same source runs on a Cartesian box, where `d` lowers to `D ± ½` shifts.

### Dependent-typing reading

- The complex is `g`. Cochain spaces are the families `Primal<g, k>` and
  `Dual<g, k>`, indexed by a statically known side and degree.
- `Incidence<g>` (§4) is a type.
- The §3 refinement (an exact neighbor list) holds by construction, because the
  incidence *is* the boundary map (§6's "inverted presentation").
- `d∘d = 0` is a structural fact; for example, curl ∘ grad = 0 on the C-grid
  placement.
- `∂g` and the collars are refinements defined by the topology, with collars defined
  by an explicit closure.
- The mesh kind is a compile-time index; the simulation condition is discharged by
  the backend, as in B.

### Relation to existing notes and the ecosystem

- **Makes the type model** out of the location typing of the FD-syntax note.
  - The note's `EDGE_N`/`EDGE_T` are `Sampled[Dual[1]]`/`Primal[1]` here: the
    primal/dual distinction, not the per-axis direction split.
  - The FD note names its complexes the other way round: its primal 0-cells are
    `CELL`.
  - `Staggered[D]` from connectivities as types is the per-axis cell class on product
    complexes.
- **Unifies** UGRID (degrees) and SGRID (bit vectors) as one complex model, which is
  the structural observation of
  [[personal/egparedes/connectivities-as-types/connectivities-as-types_conventions|the conventions appendix]].
- **Ecosystem**:
  - CF's `cell_methods` (point vs cell mean or sum) describes the value-representation
    axis (sampled vs integrated), independently of where a value is stored.
  - Firedrake/UFL function spaces, Decapodes, PyDEC and TRiSK are the precedents.

### Strengths and costs

**Strengths:**
- Halo lines are fully out of source.
- Boundary conditions are typed operations.
- The same source runs on structured and unstructured meshes.
- Conservation and the mimetic identities are structural.
- C-grid and D-grid variables are distinct types.

**Costs:**
- All of B's backend burden.
- Users must know each variable's side, degree and representation. The vocabulary
  is small but more abstract than locations.
- ICON's order-dependent, non-canonical stencils (`E2C2V` for RBF, `C2E2C2E2C`) need
  the escape hatch heavily.
- Only the compatible placements (C, D) get the guarantees; PMAP's collocated A-grid
  falls outside them.
- The cochain algebra needs overloads generated per side and degree.
- The mesh's geometry pipeline (dual complex, metric) becomes part of the language
  contract.

## Comparison

| | A | B | C |
| --- | --- | --- | --- |
| **Semantics** | **local (SPMD)** | **global** | **global** |
| Center of gravity | algebra of index spaces | data model | user surface |
| Decomposition visible in source | `Owned`, exchange points, `global_sum` | nothing | nothing |
| Halo depth | configured, or derived from reach | derived or given; never source | as B |
| Exchange placement | explicit depth-free points, or per-program `auto` | automatic | automatic |
| Simulation condition discharged by | the programmer (with validity tracking) | the backend | the backend |
| Boundary regions | declared on entity spaces, pulled back per rank | owned by the mesh, named | `∂g`, collars (explicit closures), boundary-located cochains |
| Partial relations | declared or bound; reads must be guarded | frontend error if unguarded | incidences are total; only fixed-width tables and exterior stencils are partial at `∂g` |
| Global reductions | explicit `global_sum` | implicit | implicit |
| Local dimensions | `Slot[R, P]` forest plus `Pairs[R]` | as A, scoped to a schema | hidden; incidence fields |
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
 L3  surface    C: cochain-typed fields, mesh-invariant operators        global semantics
       │  d → signed relations of the schema; star, sample, integrate → geometry fields;
       │  Boundary[...] → boundary regions
 L2  model      B: logical mesh, fields with a mesh, semantic regions     global semantics
       │  + Distribution: one local binding per rank; regions → Owned ∩ local index sets;
       │  inferred validity → exchanges or redundant computation; global sums → local + allreduce
 L1  core       A: index spaces, relations, local binding                 local semantics (SPMD)
       │  kernels → GTIR; exchanges and collectives → a host schedule (an execution plan)
 L0  execution  GTIR kernels + host schedule; gtfn / DaCe / embedded backends; FieldData buffers
```

### What each lowering does

| From → to | Lowering steps | Obligation |
| --- | --- | --- |
| **L3 → L2** | 1. Each `d` resolves to a signed relation of the schema (`d` on `Dual[0]` to `E2C` with signs, `d` on `Primal[1]` to `C2E` with signs), using the recorded orientation. 2. `star`, `sample` and `integrate` become geometry fields of the mesh instance. 3. `with_boundary` and collars become boundary regions plus `concat_where`. 4. A structured product complex lowers `d` to `D ± ½` shifts, and a degree-1 cochain to a tuple of fields. | The L2 program computes the same discrete operator. Checkable by diffing the generated code against hand-written ICON4Py stencils, as the FD-syntax note proposes. |
| **L2 → L1** | Given a `Distribution`: 1. one local binding per rank (local tables with halo rows, local extents, `ι_r`). 2. Semantic regions become local index sets intersected with `Owned`. 3. Domain inference yields the halo validity each read needs; the backend inserts an exchange or extends the producer's domain into the halo (redundant computation), by cost. 4. Global reductions become a local reduction plus an allreduce. 5. The field brand becomes the binding brand, keeping the logical-mesh part. | **The simulation condition**: every L1 rank program reproduces the L2 program on its owned entities. This is the contract to test: run the same L2 program under a trivial and a nontrivial distribution and compare bitwise, with reproducible global reductions (see the requirement on backends). |
| **L1 → L0** | Relations become bound tables keyed by relation tag (today's offset providers), and slot and incidence spaces become local dimensions. Field-operator calls become GTIR kernels. Each program becomes an **execution plan**: a host schedule that sequences the kernels with exchanges, collectives and returned scalars. GTIR has only `SetAt`/`IfStmt` statements and no exchange, allreduce or program return, so the schedule is either a host-level representation around GTIR or an extension of GTIR. | Today's toolchain contract for the kernels; a new contract for the schedule. |

With a **trivial distribution** (one rank), L2 → L1 is almost the identity: the local
binding is the logical mesh and there are no exchanges. A global-semantics stack
therefore costs nothing on a single node and keeps today's code path.

### Rules that keep the levels consistent

1. **One vocabulary across levels.** Entities, axes and relations are declared once
   as L1 nominal classes. L2 schemas group them: `Cell[Icon]` materializes an L1
   `Entity` class, which A's standalone sketch declares directly as `Cell`. L3
   cochain spaces are stored on them per mesh: `At[Dual[0], Full]` is stored as
   `Dims[Cell[M], K]`. Nothing is renamed between levels (`Cell`, `Edge`, `Vertex`
   throughout), so diagnostics and generated code use one set of names.
2. **Semantics is a property of a program, not of a field operator.**
   - A field operator with no reduction over a distributed dimension is
     *semantics-neutral*: its value on an entity depends only on that entity's
     neighborhood, so it gives the same result under global semantics and per rank
     with valid halos.
   - The converse fixes the boundary: a reduction over a distributed dimension means
     different things at the two levels (rank-local vs global). It is therefore
     **program-level only** (`global_sum` in L1, `sum_over` in L2 programs), and a
     field operator may not contain one.
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
  L2 work, and the communication schedule, by hand. The stack adds levels above it,
  plus a communication-aware execution plan around the GTIR kernels at L0. It
  replaces nothing below.

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
   recorded or rebased on load. uxarray's loader attempts this, but only
   conditionally, and it guesses the origin when `start_index` is absent (research
   appendix §4.3). A gt4py loader should normalize and validate the origin
   unconditionally.
8. **Some relations have tables known at compile time** (the scan redesign's
   vertical windows, Cartesian offsets). They need a binding mode of their own.
9. **Verticality belongs to the logical mesh, distribution to the decomposition,
   and neither is a dimension kind.**
   - *Which* axis is vertical, with its orientation, top and surface, is geometry of
     the logical mesh, declared by the schema or the axis. CF records it on the
     coordinate (`axis="Z"`, `positive`). The top and surface are semantic boundary
     regions.
   - *Whether* an axis is distributed is decided by the decomposition. In practice
     the vertical never is, which keeps vertical scans and reductions free of
     communication.
   - What remains of `DimensionKind` (layout order, the scan axis) belongs to the
     field and the program, as the
     [[personal/egparedes/connectivities-as-types/connectivities-as-types_staggering|connectivities staggering appendix]]
     §7 argues.
   - This separates the *geometric* vertical, which may be absent (a box with no
     preferred direction), from the *computational* roles: the sequential scan axis
     and the undecomposed axis. The appendix's "a role, not a geometry" holds for the
     roles only; see the conflicts section.
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
  - `V2E.Local` (nested, owned, no parent) vs `Slot[R, P]` (subscripted, carries
    its parent, generic-friendly), with `Pairs[R]` replacing the owner/sharer rules;
  - `offset_provider` vs a binding or mesh object;
  - its conventions appendix §5 lists "owned versus halo entities" as an ICON need
    for UGRID's `location_index_set`, which treats a decomposition concept as a
    semantic subset;
  - its open question 2 counts "halo variants" among the multi-table cases, whereas
    here the mesh identity excludes the decomposition;
  - its staggering appendix §1 says per-axis staggering keeps apart "the two edge
    families" that unstructured code recovers as the component label `EDGE_N` vs
    `EDGE_T`. The per-axis bits split edges by *direction*. `EDGE_N`/`EDGE_T` split
    them by *primal vs dual 1-form* (C-grid vs D-grid; `Sampled[Dual[1]]`/`Primal[1]`
    in Proposal C), and that split is orthogonal to direction. The FD-syntax note
    §3.3 makes the same identification;
  - its staggering appendix §7 calls the vertical "a role, not a geometry". Decision
    9 keeps that for the computational roles (scan axis, undecomposed axis) but
    places the geometric vertical in the logical mesh.
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
  A adopts its forest. Its `Local[C, P]` spelling, with a parent, matches
  `Slot[R, P]`, not connectivities-as-types' `V2E.Local`.
- [[personal/havogt/dimension-generic-fields/dimension-generic-fields|Generic dimensions and staggering]]:
  dimension genericity is a prerequisite (decision 10). Its `Dual[X]` name is
  reserved here for the full Hodge dual (C's `star`), as the connectivities
  staggering appendix argues.
- [[personal/egparedes/discretization-independent-fd-syntax|Discretization-independent FD syntax]]:
  C is its Proposal 3 made the type model, and L3 is its front-end. Its "halo
  exchange is never written by the user" is global semantics.
  - Its form-degree placement (`P(0)=CELL`, `P(1)=EDGE_N`, `P(2)=VERTEX`) is C's
    C-grid placement on the dual side: the FD note calls primal what C calls dual.
  - Its §3.3 identification of the per-axis bit vector with the `EDGE_N`/`EDGE_T`
    distinction is questioned above.
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
