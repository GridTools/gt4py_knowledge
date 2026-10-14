---
title: "Meshes, dimensions and fields: three typed designs and how they layer"
author: egparedes
tags: [type-system, type-checking, dimensions, mesh, fields, domain, connectivities, local-dimensions, unstructured, staggering, halos, boundary-conditions, distributed, global-semantics, local-semantics, dependent-types, nominal-types, cw-complex, exterior-calculus, cochains, reproducibility, ugrid, sgrid, cf-conventions, xarray, coordax, uxarray, design-space, prior-art]
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
> - **C — Cochain core with located fields.** *Global semantics*. Side/degree types
>   define an integrated cochain core; a located-field facade adds quantity and
>   sampling semantics through explicit conversions. Signed incidences and metric
>   maps implement operators; topological boundaries coexist with semantic masks.
>
> All three separate the **logical mesh** (topology, geometry, boundary regions:
> semantic) from the **decomposition** (ownership, halo lines: implementation). They
> are not exclusive: C lowers to B, B lowers per rank to A, and A lowers to a host
> execution plan with today's GTIR kernels (see [Layering](#layering-the-three-proposals-as-levels-of-abstraction)).

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
| enumeration (§6) | the bijection between local slots and one entity's incident entities | slot-addressed vs incidence-addressed sparse fields |
| parameter vs index, brand (§7) | does the type identify the graph? | where the mesh lives |
| validate at the boundary (§8) | check once, reuse the evidence | binding / mesh construction |

### The ideal

In a language with dependent types:

```text
Mesh g       = { entity counts; relations r : (v : A<g>) → Vec<B<g>, deg_r(v)>; geometry; boundary }
A<g>         = Fin<g.n_A>                           -- an entity index space is a type family over g
Field<g,L,T> = L<g> → T                             -- a function on one location of one mesh
Slot<g,r,v>  = Fin<deg_r(v)>                        -- a neighbor slot depends on g, r and v (Π)
SlotPos<g,r> = Σ v : A<g>. Fin<deg_r(v)>           -- a compact valid-slot occurrence (Σ)
Incid<g,r>   = Σ v : A<g>. { e : B<g> | e ∈ r(v) } -- a distinct endpoint pair (Σ)
program      : (g : Mesh) → Field<g,E,T> → Field<g,V,T>   -- Π over the mesh
```

Here `r(v)` enumerates valid occurrences, and `deg_r(v)` counts them, excluding
padding. Evaluation maps `SlotPos` to `Incid`; it is a bijection exactly when each
row is injective. Repeated paths keep distinct slot occurrences with the same
endpoint pair. Padded storage additionally needs the valid-raw-slot ↔ compact-rank
map specified in A; raw padding positions belong to neither semantic space.

### What Python can keep

Python supports specific literal types such as `Literal[3]`, but not arbitrary
runtime-dependent indices or general type-level arithmetic. Each fact about `g`
must therefore be assigned to a stage:

1. **Static**: the type checker sees it: nominal identity and expressible literal types.
2. **Compile time**: the frontend specializes on it, for example counts, dtype and
   presence of skip values. This is the two-level pattern that dtype generics and
   connectivities-as-types already use.
3. **Run time**: tables, extents, geometry, decomposition.

Limits that shape every design (verified by probes elsewhere in this knowledge base):
- nominal classes provide the mesh/dimension brands chosen here;
- there are no general integer/non-type parameters: `Sized[Literal[3]]` takes a
  *type* argument and does not make `Sized[3]` a supported value-parameter syntax;
- there are no type-level functions, so replacing a named dimension
  **wherever it occurs** inside `Dims[...]` — substitution at an arbitrary
  position, the general case — needs one overload per rank and position
  (bounded code generation). Replacing a dimension at a **fixed leading or
  trailing position** is the exception: a single generic signature
  `stagger[D, *Tail]` that maps `Field[Dims[D, *Tail]]` to
  `Field[Dims[Staggered[D], *Tail]]` covers every rank, preserving the
  unbounded variadic remainder (probe-verified with mypy and pyright);
- `TypeVarTuple` is unbounded, and a flat type-argument list takes at most one
  unpacked `TypeVarTuple`; nested tuple types are argument lists of their own;
- `M.Cell` is not valid in a type when `M` is a `TypeVar`. Families that are generic
  over a mesh must therefore use subscription (`Face[M]`), not nested attributes.

## Logical mesh and decomposition

### Two objects, never one

- **The logical mesh `g`**: the global entity sets, relations, geometry and
  **boundary regions**. A boundary region is a refinement of a global entity space,
  e.g. `{c : Cell<g> | refin_ctrl(c) = k}`: ICON's entity-specific lateral and
  feedback labels, the nudging zone, the model top and surface.
- **The decomposition `π` of `g`**, defined for every distributed entity space `A`
  (cells, edges, vertices alike):
  - ranks `r`, each with a local mesh `g_r`;
  - injections `ι_r : A<g_r> → A<g>` (the global index);
  - owned sets `owned_r(A) ⊆ A<g_r>` whose images partition `A<g>`;
  - **available support** `S_r(A)`: owned entities plus supplied halo entities,
    table rows and geometry, finite for each entity space;
  - **required footprints**: sources reachable from each consumer's domain through
    its relation terms. A finite hop bound `d` bounds this requirement, not the
    exact supplied halo. An unbounded closure would be the whole connected mesh;
  - exchange patterns.

| | Boundary regions | Halo lines |
| --- | --- | --- |
| What they are | semantic subsets: boundaries, collars, nudging and feedback zones | cached copies of values owned by other ranks |
| Exist on a single node | yes | no |
| Change results | yes: external data enters there | governed by the reproducibility contract below |
| Depend on the partition | no: a property of the logical mesh | yes: a property of the decomposition |
| Decided by | the model author, in source | the backend or configuration, never the source |
| Effect on the index space | restricts the global index space | extends the local index space |

Boundary-region membership is consistent across ranks. A local entity belongs to a
boundary region `R` exactly when its global image does (`R_r = ι_r⁻¹(R)`), so all
replicas of one global entity agree on every rank. `refin_ctrl` is a global property.
The local sets `R_r` themselves differ, since each lives in its own local index space.
A halo has no type-level meaning: it is a cache whose validity the implementation
tracks.

Given decompositions import `S_r` and validate that every required footprint is
covered; insufficient support is an error before execution. Derived decompositions
construct sufficient support (extra entities are allowed). A configured depth is
only a construction hint, never evidence of coverage. For example, ICON4Py's second
cell halo closes vertex hexagons; it is not exactly the `C2E2C2E2C` neighbourhood
([pinned halo source](https://github.com/C2SM/icon4py/blob/e9060a1898823d38028b5cbf7f506e460fc451d7/model/common/src/icon4py/model/common/decomposition/halo.py#L280)).
Coverage and field-value validity are separate checks.

### Three causes of skip values, three owners

The three causes are those identified by
[[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos|the mesh proposal]].

| Cause | Owner | Visible in source |
| --- | --- | --- |
| ragged arity (pentagons) | the relation's table type: `max_neighbors`, presence of skip values | yes |
| boundary partiality: a missing fixed-width coface slot (`E2C`) or exterior stencil value (`C2E2C`) | the logical mesh: a **partial relation** (declared or bound); a boundary edge's actual one-cell incidence is complete | yes: reads need a boundary condition or a domain excluding the missing footprint |
| truncation at the edge of the halo | the decomposition | never observable |

### Global and local semantics

- **Global semantics.** A program means a function of `g`. The code cannot observe
  `π`: no ranks, no halos, no communication. The implementation must satisfy the
  **simulation condition** `field_r = field ∘ ι_r` on valid values in reference
  arithmetic; floating-point comparison follows the contract below. The backend
  discharges it through domain inference and exchange placement.
  Precedents: PSyclone/LFRic's algorithm layer, Legion/Regent, Chapel's global-view
  arrays.
- **Local semantics (SPMD).** A program means a function of `g_r` and runs once per
  rank. The code sees the local mesh, and communication and global reductions are
  explicit. The programmer discharges the simulation condition, with tool support.
  Precedents: ICON (`sync_patch_array`), Atlas (`haloExchange`), PETSc
  (`DMGlobalToLocal`), and gt4py and ICON4Py today.

**Proposed reproducibility contract under either semantics.** For fixed inputs,
dtype, target and arithmetic policy, per-entity results independent of
non-reproducible collectives must be **bitwise identical under decomposition**.
Preserve neighbour enumeration *and* the arithmetic evaluation tree: contraction,
reassociation, and scalar/vector code generation must follow the same rounding
policy for exchanged and redundantly computed values, including remainder loops
and FMA contraction. Table order alone is insufficient.

Programs choose reproducible distributed reductions for the strict contract,
including local accumulation: a deterministic allreduce of already rounded
rank-local sums is insufficient. Candidate strategies are a fixed tree over global
indices, exact accumulators, or a specified reproducible binned algorithm, with
matching local accumulation and final rounding. Otherwise they accept
partition-dependent collective results and a **transitive exception for all their
data and control dependants**, including per-entity outputs. For example,
`sum([1e16, -1e16, 1])` is `1`, whereas grouping `[1e16]` and `[-1e16, 1]` gives
`0`; broadcasting the result or branching on `sum > 0.5` changes entity values.
The collective's initial rounding difference can therefore become a large
downstream difference. Simulation tests compare bitwise in the strict subset and
check the declared numerical/control behaviour of the excepted outputs separately.

### Design axes

| Axis | Choices explored |
| --- | --- |
| Semantics of a program | local (A) / global (B, C) |
| Where the mesh lives (§7) | parameter of the call plus a runtime brand (A) / carried by every field (B) / inherited, plus mesh *kind* at compile time (C) |
| User vocabulary | index spaces and relations (A) / mesh schema, entities and regions (B) / cochain core plus located quantity/sampling facade (C) |
| Local dimensions | dependent slots and slot-flattened Σ spaces (A, B) / hidden behind incidences and named conversions (C) |
| Boundary regions | declared on entity spaces (A) / owned by the mesh (B) / topology plus semantic masks (C) |

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
  - `Slot[R, Parent]`: the neighbour-slot space of `R` in an explicit ancestry.
  - `Label`: finite spaces with no topology, such as tracers, ensemble members or
    `LsqUnk`.

  Identity is the qualified Python name, as in connectivities as types.
- **Relations**: `class C2E(Relation[Cell, Edge])` is the static half of a table's
  type. Their **footprint algebra** includes:
  - transpose `C2E.T`;
  - composition `C2E @ E2C`;
  - reflexive closure, to include the source in the footprint.

  Reach is a conservative hop bound checked when a table is bound, or computed by
  frontend specialization of an expression. This algebra does **not** specify
  ordered stencil slots (see below). A relation
  is **partial** when a requested neighbour is absent because of the domain
  boundary, e.g. `E2C` on a LAM mesh. Partiality is a fact of the bound table,
  declared or taken from the binding like the counts, and it forces a guard on every
  potentially missing read.
- **Chains and reduction order**: slot spaces record their parent (the dependency
  forest of [[personal/havogt/dependent-local-dimensions/dependent-local-dimensions|dependent local dimensions]]).
  Each parent is a root entity or another slot, with the required origin type.
  Gathering `R: O → T` replaces root `T` by `O`, adds `Slot[R, O]`, and recursively
  reparents slots referring to `T` under that new slot. Only leaves may be reduced.
  Subscription keeps ancestry nameable in generic code; bounded generated overloads
  may enforce these rules statically, with the general forest checked by the DSL
  frontend. Ordinary Python generics alone do not prove the leaf rule.
- **Slot-flattened spaces are Σ types**: `Pairs[R]` realizes `SlotPos<g,r>`.
  For complete, injective triangular `C2E`, it is also the endpoint incidence
  space, ICON's `CE`. A sparse field has two addressings:
  - `Dims[Cell, Slot[C2E, Cell]]`: padded, CSR-like. Raw positions run over
    `Fin<max_neighbors>`; the dependent slot domain is only each row's
    non-skip positions.
  - `Dims[Pairs[C2E]]`: flattened, COO-like. Its positions are the Σ type
    `Σ v : Cell. Fin<deg(v)>`: one per valid slot, not one per raw position.

  These are **not** isomorphic as raw domains. A row `[10, skip, 11]` has three
  raw positions but degree two: with a non-trailing skip a valid raw position
  differs from its compact rank, and two buffers that differ only in padding
  carry the same compact values. The bijection derived from the table is
  between **valid slots** and compact positions — an explicit order-preserving
  raw-slot ↔ compact-rank map, carried by the binding like the tables
  themselves, with padded values ignored or canonicalized, or retained in a
  separately padded flat representation of the same width. Triangular `C2E` is
  complete and injective, so it needs no compaction and the two addressings
  coincide with the slot order; relations into a flattened space (`C2CE`) are
  derived and agree with `C2E`'s slot order by construction in exactly that
  case. In general `Pairs[R]` carries **occurrence identity**: a retained-path
  `C2E @ E2C` row `[self,n1,self,n2,self,n3]` contributes six Σ positions but
  only four endpoint pairs, and the slot index is what keeps independent path
  coefficients distinct (the ordered stencils below rely on exactly this).
  `Pairs[R]` corresponds bijectively to the endpoint space `Incid<g,r>` only
  when valid rows are injective — no repeated target within a row — as C's
  `Incidence[Cell, Edge]` is in the regular complexes assumed there.
- **The local binding** `bind_local(...)` holds:
  - local tables, including halo rows;
  - local extents;
  - the injection `ι_r` and the owned set;
  - boundary regions pulled back to local index sets (possibly non-contiguous);
  - exchange patterns.

  It is validated once (§8) and immutable, so the evidence cannot go stale. Its
  brand is a pair: the identity of the **logical** mesh (equal on all ranks), plus
  the binding itself.
- **Required support is derived, not written.** Consumer domains and relation reach
  determine footprints; the binding validates coverage by the supplied support,
  or a derived distribution grows it. Configuration cannot waive that check.
- **Value validity is tracked state.** Each field version has an initialized region
  `Valid(f, generation)` (which `Field.domain` may expose), plus freshness of cached
  replicas. A read is legal only if its whole footprint is valid at the required
  generation. Halo depth is a summary, not this proof: with `Owned={0,1,2,3}`,
  fields valid on all of `Owned` and only on `{2,3}` both have depth zero, but a
  read of `{1,2}` is legal only for the former. Partial writes preserve previously
  initialized values outside the written region if the API promises preservation;
  otherwise that region remains invalid. Owner updates invalidate affected replicas.
  `auto` may exchange valid owner values; it cannot initialize undefined owned values.
  A whole-owned-valid typestate is possible, but partial regions need runtime or
  frontend evidence.
- **Domains in source** are `Owned(Cell)` combined with semantic regions. Halo
  levels cannot be named. The backend may extend a computation into the halo instead
  of exchanging, because halo values are caches by definition.
- **Exchange points are explicit and depth-free**: `exchange(f)` means "make `f`
  valid for its consumers", rejecting requests whose owner values are undefined.
- **Global reductions are explicit and program-level only**:
  `global_sum(f, domain=d)` with `d ⊆ Owned`. Each selected global entity is counted
  once, and only the entities in `d` are read, so `d` must not exceed what was
  written. Reductions over a distributed dimension cannot occur inside a field
  operator; neighbourhood/slot reductions and undistributed-axis scans remain local
  operator primitives (layering rule 2).

### Slot ancestry and ordered stencils

Repeated hops are distinct selectors. For `vf(V2V)(V2V)` the dimensions are
`{Vertex, Slot[V2V, Vertex], Slot[V2V, Slot[V2V, Vertex]]}`. The last is the leaf;
reducing `Slot[V2V, Vertex]` first is illegal. Siblings such as
`ef(V2E) * vf(V2V)` have `{Vertex, Slot[V2E, Vertex], Slot[V2V, Vertex]}`:
both slots are leaves and either reduction order is legal (floating-point results
still follow the chosen evaluation tree). Two independent uses with identical
ancestry require explicit nominal occurrence labels; otherwise they align as the
same axis. Reparenting and sibling broadcasting are proposed contracts for gt4py.
The [[personal/havogt/dependent-local-dimensions/dependent-local-dimensions|dependent-local-dimensions prototype]]
§5.4 demonstrates the bounded-overload encoding under mypy for chains of depth
≤ 2 and two siblings, including reparenting and the leaf rule. General forest
handling remains a proposed frontend check.

Footprints are sets; stencils are ordered enumerations with a declared policy for
self inclusion, duplicate paths and skip slots. On a triangular cell,
`C2E @ E2C` enumerates `[self,n1,self,n2,self,n3]` if paths are retained, and its
set is `{self,n1,n2,n3}`. Selecting the other cell across each edge in `C2E` slot
order gives `[n1,n2,n3]`, retaining skip slots for missing exterior neighbours.
ICON's named `C2E2C` is a three-slot, self-excluding table: imports preserve its
recorded neighbour order and check any conversion from incidence tables.
`C2E2CO` prepends self. It is not
obtained just by taking reflexive closure of the already reflexive composition.
Other stencils explicitly choose path multiplicity or deduplication (and its order,
e.g. first occurrence); sparse coefficients share that exact enumeration.

Named materialized tables can carry checked footprint/reach annotations and an
ordering policy, including ICON's special multi-hop orders. An expression
`C2E @ E2C` is a staged value, **not automatically a Python type alias**: the
frontend can specialize it, or declarations/code generation can supply nominal
tags that ordinary checkers understand. Binding checks targets, omissions, skips
and enumeration against the declared construction, not just a hop count.

### Boundary regions and halo lines

- **Boundary regions** are declared once on entity spaces of the logical mesh, as
  refinements. Each rank's binding pulls them back to local index sets.
- **Halos** exist in the local index space, so their *existence* is visible (anything
  outside `Owned`); their *depth* is not.
- A composed read that reaches truncation is rejected by coverage/validity checks,
  even if its output domain lies inside `Owned`.

### Sketch

```python
class Cell(Entity): ...
class Edge(Entity): ...
class K(Axis): ...                                      # Staggered[K]: the half levels

class C2E(Relation[Cell, Edge], max_neighbors=3): ...
class E2C(Relation[Edge, Cell], max_neighbors=2): ...   # partial on a LAM mesh: boundary edges have one cell
# Footprint term (hop bound 2); binding checks the self-excluding enumeration.
class C2E2C(Relation[Cell, Cell], max_neighbors=3, reach=C2E @ E2C): ...
C2ESlot = Slot[C2E, Cell]                               # alias usable in annotations and as an axis value
CE = Pairs[C2E]                                         # Σ c. Fin(deg c): the flattened cell-edge space

class LateralBoundary(BoundaryRegion[Cell]): ...        # semantic: a refinement of the logical Cell space

@field_operator
def div(vn: Field[Dims[Edge, K], float], w: Field[Dims[Cell, C2ESlot], float]) -> Field[Dims[Cell, K], float]:
    return neighbor_sum(vn(C2E) * w, axis=C2ESlot)

@local_program
def step(vn: Field[Dims[Edge, K], float], w: Field[Dims[Cell, C2ESlot], float], out: Field[Dims[Cell, K], float]) -> float:
    interior = Owned(Cell) & ~LateralBoundary.levels(1, 4)   # owned + semantic; never HALO_LEVEL_2
    div(vn, w, out=out, domain=interior)
    exchange(out)                                       # no depth: the binding knows the reach of later reads
    return global_sum(out, domain=interior)             # explicit allreduce over what was written

local = bind_local(tables, extents, owned=owned, global_index=gidx, regions={LateralBoundary: refin_ctrl})
step(vn, w, out, binding=local)                         # supplied support must cover required footprints
```

All sketches use proposed API spellings. Here `step` is a **host execution plan**:
a GTIR kernel writes `interior`, an exchange runs only if later consumers need
remote values, and a host collective returns the scalar. This example's reduction
reads only owned `interior`, so its exchange can be elided. Today's program
lowering emits field-operator calls as `SetAt` statements, not this entire schedule
([pinned lowering](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/past_to_itir.py#L286)).
The pinned [GTIR statement model](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/iterator/ir.py#L124)
contains `SetAt` and `IfStmt`, with no host exchange, allreduce or program-return
primitive. This design adds the host-plan layer around existing kernels.

### Dependent-typing reading

- A program is Π over the local mesh `g_r`. The mesh is a **parameter** (§7), so
  types are `∀g` and never claim "same type ⇒ same mesh".
- The binding's brand restores mesh identity at run time.
- The simulation condition against the logical `g` is the programmer's obligation,
  partly discharged by validity tracking, which is evidence that the halo cache
  agrees with the owners.
- Σ is explicit as `Pairs[R]`.
- `Fin<deg>` depends on `R` nominally and on `v` through the row degree. A
  padded table realizes those semantic slots as the non-skip positions of
  `Fin<max_neighbors>`; raw positions are not the slot domain (see the
  incidence-space bullet above).
- Boundary regions are refinements on `g`, transported by `ι_r`.

### Relation to existing notes and the ecosystem

- **Keeps from [[personal/egparedes/connectivities-as-types/connectivities-as-types|connectivities as types]]**:
  nominal dimensions with qualified-name identity, `Staggered[D]`, the split between
  a declaration and its table, the boundary check.
- **Changes**:
  - `V2E.Local` becomes `Slot[R, Parent]` (generic-friendly, with contextual ancestry);
  - the owner/sharer rules for local dimensions become `Pairs[R]`;
  - the `offset_provider` dict becomes a `Binding` object.
- **Adopts** the forest and reparenting of dependent local dimensions; extends the
  mesh proposal's halo-depth dataflow to initialized regions and generations.
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
  - **derived**: the backend constructs sufficient support from the compiled
    programs' domain-specific footprints, possibly including extra entities;
  - **given**: the ICON Fortran granule path, where ICON's decomposition and halo
    configuration are imported through a translation layer that also absorbs nproma
    padding, 1-based tables and `rl_start`/`rl_end`. Coverage is checked against the
    required footprints; insufficient support rejects the plan or requires an
    explicitly enlarged host decomposition.

  Either way it is configuration, never model source.
  In the Fortran given mode, the host owns communication: the generated execution
  plan supplies synchronization and collective requirements for host hooks to
  execute. Global sums combine owned local partials through the host's allreduce,
  outside GTIR kernels, before returning a scalar. The adapter must establish one
  executor for each event; importing a partition alone does not insert communication.
- **Fields** carry `.mesh` and the valid-region/generation state described in A,
  and are allocated through it. Their brand is the identity
  of the **logical** mesh, so changing the halo depth changes neither the brand nor
  the source. Storage is per rank; type and meaning are global. A program finds its
  mesh from its field arguments and requires exactly one instance per role.
- **Domains in source** are semantic regions only, e.g.
  `grid.region(Face[Icon]) - LateralBoundary.levels(1, 4)`. There is no
  `INTERIOR`/`HALO` vocabulary. Reductions over entity dimensions are global by
  definition and belong to programs, not field operators.
- **The compile key** is the mesh *signature* (schema, table types, region layout)
  plus the distribution's parameters where they affect the generated loops, never the
  instance. One compiled program serves many meshes.

### Boundary regions and halo lines

They are separated by object:
- boundary regions are part of `Mesh`, nameable and typed;
- halos are part of `Distribution`, invisible in source, derived or configured.

An unguarded read of a partial relation is a frontend error. Truncation skips are
made unobservable by domain inference plus support-coverage and validity checks.
ICON's local memory order (by `refin_ctrl`, halos appended) becomes a layout detail
of the given mode's translation layer.

### Sketch

```python
class Icon(Mesh2D): ...                                     # schema: Face/Edge/Node[Icon] are its entity spaces
class C2E(Relation[Face[Icon], Edge[Icon]], max_neighbors=3): ...
class LateralBoundary(BoundaryRegion[Face[Icon]]): ...

def div[M: Mesh2D](vn: Field[Dims[Edge[M], K], float]) -> Field[Dims[Face[M], K], float]: ...

grid = Icon.from_ugrid("icon_grid.nc", distribution=config.distribution)   # Auto(), or given by ICON
vn = grid.zeros(Edge[Icon], K)                              # vn.mesh is grid (logical); storage is per rank
interior = grid.region(Face[Icon]) - LateralBoundary.levels(1, 4)
div(vn, out=out, domain=interior)
mass = sum_over(out, axis=Face[Icon], domain=interior)      # global by semantics; allreduce is inserted
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

## Proposal C — Cochain core with located fields

> **Semantics: global.** Programs express operators on the logical complex. The
> decomposition is invisible in source and handled as in B: a runtime
> `Distribution` of the complex, derived or given.

### Thesis

C has two connected layers: an **integrated cochain core**, typed by primal/dual
side and degree, and a **located-field facade**, typed by physical placement,
quantity and sampling. Named conversions connect them; the core's algebra alone
does not establish the physical interpretation of a coefficient. This develops
Proposal 3 of [[personal/egparedes/discretization-independent-fd-syntax|the FD-syntax note]]
and reuses B's logical mesh / distribution split. Named relations and
reconstructions remain escape hatches.

### Cochain core: side, degree and derived storage

Use the mesh-as-stored convention: primal `Vertex`, `Edge`, `Cell` have horizontal
degrees 0, 1, 2. An admissible geometric dual pairs their bulk DOFs as follows;
a bounded dual needs the additional completion/relative policy described below.
This table identifies storage, not point-sampling semantics or boundary closure.

| Degree | Primal cochain support | Dual cochain support | Canonical storage entity for the dual |
| --- | --- | --- | --- |
| 0 | vertices | dual vertices at cell centres | `Cell` |
| 1 | primal edges | crossing dual edges | `Edge` |
| 2 | primal cells | dual cells around primal vertices | `Vertex` |

Mathematically write these spaces `C_P^k(H)` and `C_D^k(H)` on a horizontal
complex `H`. The Python families are `Primal[Literal[0]]`, `Dual[Literal[1]]`,
etc.; raw `Dual[1]` is only mathematical shorthand, not a Python value parameter.
A core field uses `Field[At[Dual[Literal[1]], Full, Integrated], float]`, retaining
vertical staggering separately. `Integrated` names a cochain DOF: exact continuous
integration may be replaced by a declared numerical quadrature when constructing it.

Topology and metric have different signatures on a validated n-complex:

```text
d_P^k : C_P^k -> C_P^(k+1)       d_D^k : C_D^k -> C_D^(k+1)
star_P^k : C_P^k -> C_D^(n-k)    star_D^k : C_D^k -> C_P^(n-k)
```

`d` is the signed coboundary on one side. A diagonal circumcentric `star` uses
primal/dual measure ratios on an admissible orthogonal dual; a general mesh needs
a declared metric discretization. Inverses and orientation signs are part of that
contract, not inferred from storage. See [Hirani, §§3.6 and 4.1](https://hirani.web.illinois.edu/papers/Hirani2003_PhD_Caltech.pdf).
A Hodge result of degree zero is an algebraic coefficient; interpreting it as a
continuum point sample still requires the facade's sampling contract.

Finite overloads express supported transitions without type-level arithmetic.
For example, these are **stub signatures** for the horizontal dual complex:

```python
from typing import Literal, overload

D0 = Field[At[Dual[Literal[0]], Full, Integrated], float]
D1 = Field[At[Dual[Literal[1]], Full, Integrated], float]
D2 = Field[At[Dual[Literal[2]], Full, Integrated], float]

@overload
def d(f: D0) -> D1: ...
@overload
def d(f: D1) -> D2: ...
```

Generate analogous signatures per supported side/degree and vertical staggering,
plus `star` transitions. Both pinned checkers accept this bounded D0 → D1 → D2
encoding and reject wrong-side calls. This demonstrates signatures, not arbitrary
inference or an implemented operator. The terminal `d` into a zero space can be
omitted from the API; its mathematical existence is unchanged.

The mesh *kind* (product or general complex) is a compile-time index. Product
incidences lower to fixed index shifts without tables; general complexes lower
to signed tables. Metric and weight strategies are mesh properties.
On a structured 2-D box, each side's degree-1 space has two directional edge
blocks. This directional split is independent of both primal/dual side and
normal/tangential quantity.

### Located facade and conversion contracts

`At[CellScalar, Full, Point]` names placement, vertical staggering and sampling.
`EdgeNormal`, `EdgeTangential` and `EdgeScalar` distinguish quantities at the same
edge. `Point`, `CellMean` and `DualEdgeMean` distinguish representations; the facade
retains those tags through its named conversions and lowering. Side/degree alone
cannot authorize a relabelling of physical quantity or sampling.

Integrating a continuous scalar onto a 0-cell means evaluating it at that point;
a cell average stored at the same cell centre does not become such a point value
without reconstruction. Multiplying an exact mean by its integration measure gives
an integral; multiplying an arbitrary point sample by measure chooses a quadrature.
For example, on `[-1,1]`, the midpoint sample of `x²` times length is 0 while its
integral is 2/3. Similarly, a pressure difference of `x³` divided by length is the
mean derivative 1, while its midpoint derivative is 0. No generic `sample` or
`integrate` operation may erase these distinctions.

The operators below use **horizontal n=2 at each vertical level**, preserving
`Full`/`Half`. Full product differentiation on `H × K` has n=3 and
`d = d_H + (-1)^k d_K` on horizontal degree k: it produces horizontal/vertical
components and changes vertical staggering. It is a separate API, not an implicit
meaning of horizontal `d`. Scans and column solves remain orthogonal primitives.

For a horizontal C-grid with orthogonal primal/dual edges, fix a tangent `t_e` on
each primal edge and a normal `n_e` pointing to the right of `t_e` in the oriented
surface. Orient the paired dual edge along `n_e`. Then the proposed scalar/vector
conversions and operators are:

| Operation | Representation and factors |
| --- | --- |
| `as_dual0(p)` | cell-centre scalar samples → `C_D^0(H)` (0-cell integration is evaluation); cell averages require an explicit reconstruction first |
| `grad_n(p)` | `d_D^0(as_dual0(p))` is a dual-edge integrated difference; divide by dual-edge length to obtain the mean normal derivative along that dual edge; interpreting it as a point sample needs a declared reconstruction |
| `integrate_normal_flux(vn)` | an edge-normal sample/mean → primal-edge flux cochain `F_P ∈ C_P^1(H)`, with `F_P(e) = length(e) * vn(e)` under the declared edge quadrature |
| `div(vn)` | `d_P^1(F_P)` is net outward flux per primal cell; divide by primal area to obtain a cell-average divergence |
| `integrate_tangential(vt)` | edge-tangential data → primal circulation cochain `U_P(e) = length(e) * vt(e)` under its quadrature; `d_P^1(U_P)` gives integrated curl per primal face |

The incidence sign relative to a face boundary turns `F_P(e)` into outward flux.
Metric conversion between dual-edge velocity integrals and primal-edge fluxes must
include the length ratio and orientation sign; bare `star(d(star(vn)))` on point
values leaves these contracts unspecified. Nor can `d_P` take primal degree-2 cell
pressure to edges. The signatures above deliberately state which cochain is used.
A vertex-located curl instead needs the corresponding dual-face circulation
representation and reconstruction; `EdgeTangential` alone does not select it.
In particular, both normal **flux** and tangential **circulation** above are primal
1-cochains. Their shared side/degree does not make them the same quantity. A
staggering declaration chooses DOFs and conversions; N/T is not intrinsically a
primal/dual classification.

Normal flux and tangential circulation each use the structured directional blocks
with different quantity semantics. The facade resolves to the same cochain core
on a box or an unstructured mesh. Sparse incidence fields are endpoint Σ types
(`Incidence[Cell, Edge]`, the ideal's `Incid`); their enumeration is bijective with
A's valid-slot `Pairs[R]` in this duplicate-free case. Ordered path stencils with
repeats retain occurrence identity instead.

### Topological boundary and semantic regions

Assume an oriented regular manifold cell complex with a validated boundary
subcomplex `∂g` and a declared dual construction. Non-manifold or non-regular input
requires a separate incidence/boundary/dual policy. Primal boundary and coboundary
maps are **total on the finite complex**, including its boundary: a single triangle
has all three edges and `d_P^1 d_P^0 = 0`. A boundary edge's coface list is complete
with one cell; a two-slot `E2C` representation has a missing slot, and a two-sided
or composed stencil may need an exterior cell. These affect adjacency and dual
completion, not the existence of the finite incidence map.

Boundary conditions supply the traces/ghost values and geometry an operator
actually needs. A flux divergence needs boundary-edge flux; a two-sided
cell-pressure gradient needs a declared dual boundary completion and pressure
condition. Higher/composed stencils may need several exterior layers or restriction
of their evaluation domain. Naming `∂g` does not discharge that obligation.
Relative cochains are another option, with their own restricted operator contract.

Arbitrary **semantic masks** remain available alongside topology. For an explicitly
chosen monotone distance/support rule, define `Collar[k]` as the *downward closure*
of cumulatively selected cells through level k, together with `∂g`; validate
closure and nesting before calling these subcomplexes a filtration. Separate level
bands, nudging masks and parent-child feedback regions need not be subcomplexes.
ICON's `refin_ctrl` is imported as entity-specific semantic labels, not proof of
such a filtration: cell/vertex and edge counting differ, and negative values mark
nest overlap ([ICON v2.6.4 nesting, Appendix A1](https://gmd.copernicus.org/articles/15/7153/2022/#A1)).

Halos remain B's distribution support. Incidence and reconstruction footprints
bound required support; coverage and current-value validity are checked separately.

### Sketch

```python
# Proposed API: horizontal operators, with boundary policies supplied by the mesh.
@field_operator
def div(vn: Field[At[EdgeNormal, Full, Point], float],
        flux_bc: Field[At[Boundary[EdgeNormal], Full, Point], float]) -> Field[At[CellScalar, Full, CellMean], float]:
    flux = integrate_normal_flux(vn, boundary=flux_bc)  # C_P^1(H); use boundary samples in the edge quadrature
    return cell_average(d_primal(flux))                # C_P^2(H) / primal_area

@field_operator
def grad_n(p: Field[At[CellScalar, Full, Point], float],
           pressure_bc: BoundaryPressure[Full]) -> Field[At[EdgeNormal, Full, DualEdgeMean], float]:
    dp = d_dual(as_dual0(p), boundary=pressure_bc)       # C_D^0(H) -> C_D^1(H), with explicit completion
    return edge_normal_average(dp)                     # divide by dual_edge_length
```

Here `as_dual0(p)` returns core D0; `integrate_normal_flux` constructs a primal
1-cochain under the declared edge quadrature. `d_primal` and `d_dual` are
side-specific wrappers around core `d`, with the latter applying the declared
pressure boundary completion. `cell_average` and `edge_normal_average` return the
facade's tagged means, not point samples. Geometry, quadrature and any subsequent
mean-to-point reconstruction belong to the schema contract.

### Dependent-typing reading and guarantees

The complex is `g`; cochain families are indexed by side and degree as well as
mesh, while located sample fields also name quantity and sampling representation.
Incidences and semantic regions give Σ types and refinements. Validation checks
orientations, enumeration, boundary closure and any admissibility assumptions of
the metric map.

`d^2 = 0` and cancellation of interior fluxes are structural **in exact arithmetic**
for compatible incidence operators on one complex (or a consistently completed or
relative complex). Floating evaluation follows the reproducibility contract above;
these identities alone do not promise a bitwise-zero residual. Compatible C/D
staggerings obtain these guarantees only when their DOFs, primal/dual assignment,
conversions and operator sequence realize that complex. A primal/dual swap must
also swap signatures and metric conventions. Reconstruction, nonlinear operators,
limiters, boundary closures and ordered escape stencils have separate obligations;
the C/D label alone proves none of them. A collocated A-grid can use the surface
with explicit reconstructions, without automatically inheriting the identities.

### Relation to existing notes and the ecosystem

- **Develops** the FD-syntax note's location typing and `EDGE_N`/`EDGE_T`/`EDGE`
  distinction, making its shorthand de Rham formulas representation-specific.
  Its §6.4–6.5 convention calls the CELL → EDGE_N → VERTEX sequence primal;
  relative to our stored-mesh convention its paired support is dual 0 → 1 → 2.
  This is a naming/support map, not a proof of component, sampling or C/D guarantees.
  The directional/component conflict is detailed under [Conflicts](#relation-to-other-notes-and-conflicts).
- **Unifies** UGRID entity placement and SGRID product staggering through complex
  schemas, using `Staggered[D]` for per-axis classes. This extends the
  [[personal/egparedes/connectivities-as-types/connectivities-as-types_conventions|conventions appendix]];
  UGRID location alone does not choose primal/dual representation.
- **Ecosystem**: CF `cell_methods` describes sampling/aggregation independently of
  placement: point samples and area averages may both be face-located. Firedrake/UFL,
  Decapodes, PyDEC and TRiSK motivate the explicit function-space/cochain layer.

### Strengths and costs

**Strengths:**
- Halo lines stay out of source; boundary requirements appear in operator contracts.
- One operator surface resolves to structured and unstructured meshes.
- Compatible incidence sequences expose conservation and mimetic identities.

**Costs:**
- All of B's backend burden, plus representation/quantity typing and metric validation.
- Core authors must choose side/degree; facade authors must supply the associated
  conversion and sampling contracts.
- ICON's ordered non-canonical stencils (`E2C2V` for RBF, `C2E2C2E2C`) need escape hatches.
- Boundary completion and reconstructions require separate numerical validation.
- The location/cochain algebra needs bounded generated overloads; geometry is part
  of the language contract.

## Comparison

| | A | B | C |
| --- | --- | --- | --- |
| **Semantics** | **local (SPMD)** | **global** | **global** |
| Center of gravity | algebra of index spaces | data model | cochain core with a located-field facade |
| Decomposition visible in source | `Owned`, exchange points, `global_sum` | nothing | nothing |
| Halo support | given/derived; required footprints checked | derived or given; never source | as B |
| Exchange placement | explicit depth-free points, or per-program `auto` | automatic | automatic |
| Simulation condition discharged by | the programmer (with validity tracking) | the backend | the backend |
| Boundary regions | declared on entity spaces, pulled back per rank | owned by the mesh, named | `∂g`, validated cumulative collars, arbitrary masks |
| Partial relations | declared or bound; reads must be guarded | frontend error if unguarded | total incidences; exterior stencils need a closure/guard |
| Global reductions | explicit `global_sum` | implicit | implicit |
| Local dimensions | `Slot[R, Parent]` forest plus `Pairs[R]` | as A, scoped to a schema | hidden; incidence fields |
| Brand | logical mesh plus local binding | logical mesh (distribution excluded) | as B |
| Several meshes in one process | runtime brands | typed roles plus relations between meshes | as B |
| Same source, structured and unstructured | no | no | yes, for the shared operator/representation contract |
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
 L3  surface    C: located facade → typed cochain core                   global semantics
       │  quantity/sampling conversions; d → signed relations;
       │  star / quadrature → geometry operators; boundary policies → traces / extensions
 L2  model      B: logical mesh, fields with a mesh, semantic regions     global semantics
       │  + Distribution: one local binding per rank; regions → Owned ∩ local index sets;
       │  inferred halo validity → exchanges or redundant computation; global sums → local + allreduce
 L1  core       A: index spaces, relations, local binding                 local semantics (SPMD)
       │  relations → bound tables; Slot / Pairs → local dimensions
 L0  execution  host plan: GTIR kernel calls + communication + collectives + scalar returns
               gtfn / DaCe / embedded kernels; FieldData buffers
```

### What each lowering does

| From → to | Lowering steps | Obligation |
| --- | --- | --- |
| **L3 → L2** | The facade lowers through explicit conversions to the cochain core: dual d0 uses signed `E2C`, primal d1 uses signed `C2E`. Preserve quantity/sampling contracts. `star`, measures and quadrature become geometry operators/fields. Boundary policies become trace/ghost operations or restricted domains; masks can use `concat_where`. Product incidences lower to shifts and directional field tuples. | Same discrete operator, including boundary completion. Compare with hand-written stencils and check incidence/metric identities separately. |
| **L2 → L1** | Build rank bindings and intersect semantic regions with `Owned`. Check support coverage and initialized regions/generations; schedule exchanges or valid redundant computation. Split global reductions into owned local partials and a collective under the chosen reproducibility policy. Preserve the logical-mesh brand. | **Simulation condition** on valid owned values: compare trivial and nontrivial distributions bitwise for the strict subset; separately check non-reproducible collectives and all their data/control dependants under the declared policy. |
| **L1 → L0** | Relations become bound tables; slot/incidence spaces become local dimensions. Emit a host execution plan containing GTIR kernel calls, communication hooks, host collectives and scalar returns. | Today's kernel lowering can remain unchanged; the host-plan layer is proposed work, not existing whole-program GTIR support. |

With a **trivial distribution** (one rank), L2 → L1 is almost the identity: the local
binding is the logical mesh and there are no exchanges. A global-semantics stack
therefore keeps today's kernel code path on a single node.

### Rules that keep the levels consistent

1. **Canonical identities with explicit mappings.** L2 `Face[Icon]` materializes
   one L1 nominal entity class; an ICON declaration can use the typed alias
   `Cell = Face[Icon]` (an alias usable in types and at runtime). L3
   `At[CellScalar, Full, Point]` maps through the schema to
   `Dims[Face[M], K]` with its quantity/representation contract. Surface names can
   differ, but aliases/mappings resolve to canonical objects and diagnostics show
   the mapping; no safety check relies on strings coincidentally agreeing.
   Core `At[Dual[Literal[0]], Full, Integrated]` also uses `Dims[Face[M], K]`,
   but shared storage does not identify it with every facade field on those cells.
2. **Semantics is a property of a program, not of a field operator.**
   - The shared *pure, neighbourhood-local* operator subset is semantics-neutral
     when logical boundary interpretation, ordered stencils and arithmetic policy
     agree, and all inputs are valid. Operators depending on rank-local state or
     on collective results are outside that unconditional claim.
   - Reductions over distributed dimensions are **program-only**: `global_sum`
     in L1, global `sum_over` in L2/L3. They cannot appear inside field operators.
     Local neighbour-slot reductions and scans on undistributed axes remain valid
     field-operator primitives.
   - A library of that subset serves L1 and L2, with collectives placed in the
     host program plan. Programs (domains, communication, global reductions) are
     global (`@program`) or local
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
   - L3: placement/quantity and cochain algebra, conversions and boundary policies;
   - L2: mesh identity, unguarded partial relations, boundary regions;
   - L1: ancestry and leaf rules, ordering, binding coverage, valid regions/generations;
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
  - connectivities as types (implemented; the proposed additions are a binding
    object, contextual `Slot`/`Pairs`, partial relations and ordered/staged relations);
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

1. **L1 core**: a binding object, contextual `Slot`/`Pairs`, partial relations,
   ordered/staged relations, valid-region tracking and a host execution plan.
2. **L2 on one node**: logical mesh, fields that know their mesh, semantic boundary
   regions, a trivial distribution. Regions replace zone index arithmetic for
   boundaries.
3. **L2, given distribution**: import ICON's available support, validate required
   footprints, and lower to L1 with host-owned communication and collectives.
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
- **Mappings must preserve canonical identities** (rule 1). Unchecked name-only
  correspondence reintroduces the string-agreement problem.

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
5. **Reproducibility is an explicit arithmetic/collective policy.** The strict subset
   is bitwise stable under decomposition; non-reproducible collectives and all their
   data/control dependants share the transitive exception described above.
6. **One meaning for subscripting a dimension.** Today `D[...]` is wanted for regions
   (`K[1:nlev]`), slot indices (`V2E[1]`) and type construction (`Staggered[K]`).
7. **Table entries index the codomain's logical extent.** The proposed loader
   normalizes origin independently of dtype/fill conversion: honour explicit
   `start_index`, default omitted UGRID origin to zero, preserve fill slots, then
   validate codomain bounds. Never infer origin from the minimum referenced index.
   uxarray is a qualified precedent: its pinned loader nests rebasing under a
   dtype/fill branch and guesses the minimum when origin is omitted (research §4.3).
8. **Some relations have tables known at compile time** (the scan redesign's
   vertical windows, Cartesian offsets). They need a binding mode of their own.
9. **Verticality belongs to the logical mesh, distribution to the decomposition,
   and neither is a dimension kind.**
   - *Which* axis is vertical, with its orientation, top and surface, is geometry of
     the logical mesh, declared by the schema or the axis. CF records it on the
     coordinate (`axis="Z"`, `positive`). The top and surface are semantic boundary
     regions.
   - *Whether* an axis is distributed is decided by the decomposition. ICON leaves
     the vertical undistributed; a scan along a distributed axis would require an
     explicit communication algorithm or rejection.
   - What remains of `DimensionKind` (layout order, the scan axis) belongs to the
     field and the program, as the
     [[personal/egparedes/connectivities-as-types/connectivities-as-types_staggering|connectivities staggering appendix]]
     §7 argues. Its item 6, "Vertical is a role, not a geometry", refers to the
     scan/layout role of an otherwise generic axis. If read as denying physical
     vertical geometry, it conflicts with this note: physical axis, program scan
     role and decomposition choice are distinct. A Cartesian axis can carry a
     physical vertical interpretation without forcing a scan or a partition policy.
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
   worth its annotation cost in A, alongside the required valid regions/generations?

## Relation to other notes and conflicts

- [[personal/egparedes/connectivities-as-types/connectivities-as-types|Connectivities as types]]
  provides the L1 base. **Conflicts**:
  - `V2E.Local` (nested, owned) vs `Slot[R, Parent]` (subscripted, ancestry-aware), with
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
  vertical boundary regions; its undistributed-column assumption matches ICON. Its
  `KDim[:-1]` examples conflict with logical absolute coordinates. Its
  `WindowOffset` is a static relation (decision 8) written in the removed
  `FieldOffset` style.
- [[personal/havogt/field-data-protocol/field-data-protocol|Field data protocol]]:
  "support ⊇ domain" is how a halo looks at L0 (data beyond the compute domain,
  private like the origin). Its absolute coordinates are qualified by decision 2.
- [[personal/havogt/dependent-local-dimensions/dependent-local-dimensions|Dependent local dimensions]]:
  A adopts its forest, including `Local[C, P]` ancestry and reparenting as
  `Slot[R, Parent]`, rather than the path-only `Local[C]` spelling.
- [[personal/havogt/dimension-generic-fields/dimension-generic-fields|Generic dimensions and staggering]]:
  dimension genericity is a prerequisite (decision 10). Its per-axis `Dual[X]`
  name conflicts with C's `Dual[...]` cochain-side family. Here the axis
  involution is `Staggered[X]` and the Hodge map is `star`, following the
  staggering appendix's separation of axis staggering and full duality.
- [[personal/egparedes/discretization-independent-fd-syntax|Discretization-independent FD syntax]]:
  C develops its Proposal 3 and the §7.1 component distinctions. **Qualification**:
  its shorthand `d0` / `star d star` and C/D guarantees need the explicit side,
  representation and boundary contracts above. Its "halo exchange is never written
  by the user" is global semantics. Its §6.4–6.5 primal naming for the
  `P(0)=CELL`, `P(1)=EDGE_N`, `P(2)=VERTEX` placement corresponds to dual support
  in our stored-mesh convention; this correspondence does not identify N/T with
  side, or make C/D identities automatic.
- [[personal/egparedes/connectivities-as-types/connectivities-as-types_staggering|Staggering appendix]]:
  **Conflict in §1**: it identifies structured directional edge families with the
  `EDGE_N`/`EDGE_T` component distinction. FD-syntax §3.3 links its directional
  bit vector indirectly to the two edge families of §6.4; §7.1 explicitly defines
  N/T as normal/tangential physical components. Directional block, quantity and
  primal/dual side are independent choices: both normal flux and tangential
  circulation can be primal 1-cochains under C's convention. Physical vertical
  geometry is logical; scan/layout roles belong to programs and distribution to
  decompositions. Its §7 item 6 needs the qualification in decision 9.
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
