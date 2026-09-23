---
title: "Connectivities as types — background: dependent typing for mesh connectivities"
author: egparedes
tags: [type-system, type-checking, dependent-types, connectivities, local-dimensions, unstructured, offset-provider, nominal-types, prior-art]
created: 2026-09-23
status: draft
---

> **Appendix** to [[personal/egparedes/connectivities-as-types/connectivities-as-types|Connectivities as types]].
> Background for reviewers: what a connectivity *would* look like in a
> language with dependent types, and which parts of that picture the
> Python-level design keeps statically, which it moves to a bind-time check,
> and which it gives up. Part 1 is a condensed primer, adapted from the note
> *Dependent typing for imperative programmers — from graph connectivity to
> checked sparse computations*; part 2 maps it onto gt4py.

## Why this appendix

The main note makes several choices that look arbitrary in isolation: the
local dimension is *owned* by a connectivity (`V2E.Local`) rather than being a
free-standing name; the class `V2E` is a *declaration* that never holds data;
neighbor counts are optional and taken from the table; a table is checked at
the program boundary and then trusted. All of them are the Python
approximation of one idea from type theory: **the type of a local index
depends on a value** — the mesh, and the element whose neighbors are being
visited. Python's type system cannot mention values, so the design has to
decide what to keep of that dependency and where to check the rest.

[[personal/havogt/dependent-local-dimensions/dependent-local-dimensions|Dependent local dimensions]]
starts from the same observation ("`V2EDim` is a *dependent* dimension", §1)
and cites Dex's dependent index sets as prior art (§10). This appendix spells
out the vocabulary both notes lean on.

## Part 1 — A primer

The primer uses C-like pseudocode in which angle brackets accept types *or
values*, and later arguments may mention earlier ones. Graphs are immutable
values throughout.

### Running example

A mesh fragment `G` with four vertices and three edges, as a Boolean
vertex × edge incidence matrix:

```text
             edge 0  edge 1  edge 2
vertex 0       true   false   false
vertex 1       true   true    true
vertex 2       false  true    false
vertex 3       false  false   true
```

Its neighbor lists are `[0]`, `[0, 1, 2]`, `[1]`, `[2]`. The ordinary API
leaves every interesting condition unstated:

```text
List<int> neighbors(Graph g, int v);
double    weight(Graph g, int v, int e);
```

IDs must be in range, the list must be exact, and `weight` needs an incident
pair — `(0, 1)` has valid IDs but is not an incidence.

### 1. Put values in types

A **type family** selects a type from a parameter, which may be a value:

```text
Vector<T, n>    // exactly n elements
Fin<n>          // a natural number i with 0 <= i < n

type Vert<Graph g> = Fin<g.nV>;
type Edge<Graph g> = Fin<g.nE>;

double get(Nat n, Vector<double, n> a, Fin<n> i);   // the type of i is the bounds check
```

The checker reasons symbolically about `n`; it need not know its numeral.

### 2. Let arguments determine types (Π and Σ)

A **dependent function type** (Π type) lets an argument determine later
argument or result types:

```text
auto   row(Graph g, Vert<g> v) -> Vector<double, degree(g, v)>;
double at (Graph g, Vert<g> v, Fin<degree(g, v)> k);
```

`k` is a **local slot** in `v`'s row, not an edge ID, and its legal range
changes with `v`. A **dependent pair** (Σ type) packages a value with data
whose type mentions it:

```text
struct Row { Nat length; Vector<double, length> data; };
```

### 3. Refinements: say *exactly* the neighbors

A length-correct `Vector<Edge<g>, degree(g, v)>` still admits `[2]` for vertex
0. A **refinement** restricts a type by a predicate:

```text
type NeighborList<Graph g, Vert<g> v> =
    List<Edge<g>> es where
        noDuplicates(es) &&
        forall (Edge<g> e) (contains(es, e) == g.incident(v, e));
```

— soundness, completeness, uniqueness. Order is extra information: without
sorting, a row of degree `d` has `d!` valid orderings.

### 4. Proofs are evidence, and can be erased

`Proof<P>` is checked evidence for `P`, not a Boolean. A loop that scans the
incidence matrix establishes `NeighborList` by an invariant ("the result holds
exactly the incident edges examined so far"). Evidence is usually erased
during compilation; computational data that happens to appear in types —
lengths, offsets, indices — is not.

### 5. Restrict a field to valid incidences

```text
type IncidentEdge<Graph g, Vert<g> v> = Edge<g> e where g.incident(v, e);

struct Incidence<Graph g> {             // a Σ type: edge's type mentions vertex
    Vert<g> vertex;
    IncidentEdge<g, vertex> edge;
};

type IncidenceField<Graph g> = function(Incidence<g>) -> double;
```

The field assigns one value per *incidence*: `(0, 0)` and `(1, 0)` may carry
different weights although both name edge 0. The dependency restricts the
**domain**; the output is an ordinary `double`.

### 6. Edge IDs vs. local slots: the enumeration

The edges above one vertex (its **fiber**) can be addressed by edge ID or by
local slot. An exact neighbor list is a **bijection** between the two:

```text
type NeighborOrder<Graph g> =
    function(Vert<g> v) -> Bijection<Fin<degree(g, v)>, IncidentEdge<g, v>>;
```

Once an enumeration is chosen, a field has two equivalent addressings:

| Addressing | Type for a fixed graph `g` | Storage analogy |
| --- | --- | --- |
| vertex–edge incidence | `Incidence<g> -> double` | coordinate-based, like COO |
| vertex–local slot | `(v : Vert<g>) -> Fin<degree(g, v)> -> double` | row-and-slot, like CSR |

Degrees alone define the legal slots but not *which* edge sits in each;
**reordering the neighbors requires reordering every slot-addressed value
consistently.** A uniform degree reduces the ragged shape to a rectangular
array — the edge correspondence still matters. With `nbr : NeighborOrder<g>`
and `field : IncidenceField<g>`, a neighbor reduction is:

```text
double sumAtVertex(Vert<g> v) {
    double total = 0.0;
    for (Fin<degree(g, v)> k : range(degree(g, v)))
        total += field(Incidence<g>{v, nbr(v).forward(k)});
    return total;
}
```

### 7. Types indexed by a graph do not identify the graph

Indexing a type by `g` does **not** prove "same type ⇒ same graph": an alias
may ignore its index, and `Fin<g.nV>` distinguishes bounds, not two meshes
with the same vertex count. Preventing cross-mesh ID mixing needs an
**identity-preserving wrapper, or brand**.

### 8. Validate at boundaries, then reuse the evidence

Runtime input acquires a dependent type through a checked decision
(`makeIncidence(g, v, e) -> Option<Incidence<g>>`). A graph loaded at run time
travels with its checked operations:

```text
type NeighborsFor<Graph g> = function(Vert<g> v) -> NeighborList<g, v>;

struct GraphPackage {
    Graph graph;
    NeighborsFor<graph> neighbors;
    IncidenceField<graph> field;
};
```

The loader proves the contracts once; consumers reuse them. Evidence is valid
only for the graph it describes — a mutated or different graph needs a new
check. The trade-off is proof work at construction points in exchange for
guarantees everywhere downstream; validated constructors and branded ID
wrappers offer part of this discipline in ordinary languages.

## Part 2 — What gt4py can keep of it

### The dictionary

| Dependent-typing concept | gt4py today (v1.2.2) | This design |
| --- | --- | --- |
| `Vert<g>` — index type of a mesh entity | `Dimension("Vertex")`, identity by `(value, kind)` | the class `V(DimensionIndex)`; identity is the class (a **brand**, §7) |
| `Fin<degree(g, v)>` — local slot, depends on `g` *and* `v` | `V2EDim = Dimension("V2E", LOCAL)`, a free name | `V2E.Local`: depends on the **declaration** `V2E` (nominally, by nesting), not on `g` or `v` |
| `degree(g, v)` bounded by the padded row length | `max_neighbors` on the table's type | `max_neighbors` / `min_neighbors`: optional class keywords stored on `V2E.Local`, else taken from the table |
| `Option`-valued slots of a padded ragged row | `skip_value` (−1) in the table | same; if `min_neighbors` is declared, skip values must be present iff `min_neighbors < max_neighbors` |
| `NeighborOrder<g>` — the enumeration | a `NeighborTable` / `NdArrayConnectivityField` | unchanged: the table *is* the enumeration |
| `(v : Vert<g>) -> Fin<degree(g, v)> -> double` — slot-addressed field | `Field[Dims[V, V2EDim], float]` | `Field[Dims[V, V2E.Local], float]` |
| `Incidence<g>` — a Σ-typed position | an untyped `{"Vertex": 3, "V2E": 2}` dict | `MultiDimensionIndex(V(3), V2E.Local(2))`; component kinds checked at run time, not that `V2E.Local` belongs to `V` |
| the graph value `g` in the signature | a string-keyed `offset_provider` | a bound connectivity: `{V2E: table}` binds the class to one table per call |
| `GraphPackage` — graph plus checked operations | — | the provider after `check_offset_provider` (and, in [[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos\|the mesh proposal]], a Mesh object) |
| validate at the boundary, reuse evidence | no check; errors surface as `KeyError` or wrong results | `check_offset_provider` when a compiled variant is created and on embedded calls, then trusted by lowering and backends |

### What is kept statically

- **Which index space a local slot belongs to.** The essential dependency of
  §2 is that `k` in `at(g, v, k)` is not an arbitrary integer but a slot of
  *this* neighbor relation. `V2E.Local` keeps that part: a field over
  `(V, V2E.Local)` and one over `(V, V2V.Local)` are different types for mypy
  and pyright, and `neighbor_sum(..., axis=V2E.Local)` names the relation it
  reduces over. This is why the design insists on a *nested, explicitly
  declared* `Local` — it is the only spelling that is a real, distinct type for
  both checkers (probes P1b and P2 in
  [`typing_probe.py`](typing_probe.py)). A connectivity that *shares* or
  *adopts* a local dimension exposes it as a plain attribute, so annotations
  name the declaring class (`C2E.Local`, not `C2CE.Local`).
- **A brand per mesh entity (§7).** Nominal identity makes two independently
  declared `Vertex` classes distinct types, statically and at run time. The
  `(name, kind)` equality of [[shared/dimensions-as-types|dimensions as types]]
  is the unbranded `Fin<g.nV>` of §7: equal names, interchangeable indices.
  Nominal identity is not sufficient to separate two *meshes* that share the
  same declarations, though — see below.

### What moves to a bind-time check

Python has no value-indexed types, so the dependency on the table value `g`
cannot be expressed. The design splits the connectivity type the way §4
splits data from evidence:

- The **class** `V2E` is the static, erasable part: `Origin`, `Codomain`,
  `Local` and, optionally, the declared counts. It is what the frontend and the
  type checker see.
- The **table** is the value `g`. Binding happens per call through the
  provider (`{V2E: table}`), and `check_neighbor_table` is the "checked
  decision" of §8: it establishes that the table's domain is `(Origin, Local)`,
  its codomain `Codomain`, its dtype integral, its width equal to a declared
  `max_neighbors` and not below a declared `min_neighbors`, and — if
  `min_neighbors` is declared — its skip values consistent with it.
- **Undeclared counts are taken from the table** when a variant is compiled.
  In §4's terms the counts are computational data that appear in types and
  are *not* erased; they are specialised on per compiled variant rather than
  fixed in the source. The main note's *Binding model* explains why a
  static-only count would bind DSL source to one mesh family.

### What is given up (and why that is acceptable)

- **Soundness and completeness of the table (§3).** Nothing checks that a
  table lists *exactly* the incident entities, without duplicates. That is a
  property of the mesh generator, and gt4py never has the incidence relation
  independently of the table to check it against.
- **Per-element ragged degree.** `Fin<degree(g, v)>` becomes
  `Fin<max_neighbors>` plus skip values, i.e. the padded, rectangular
  representation of §6. `min_neighbors == max_neighbors` is exactly the
  uniform-degree case, which is why it is equivalent to "no skip values".
- **Same type ⇒ same table (§7).** `V2E` does not identify a mesh. Two
  different tables bound to `V2E` in two calls give two different, both
  well-typed, programs. This is deliberate — it is what lets one DSL source run
  on several meshes — and the one-table-per-class-per-call dict keeps it from
  becoming ambiguous *within* a call. Guarding against mixing meshes across
  calls (e.g. a field computed on mesh A consumed on mesh B) is out of scope;
  it would need a mesh-level brand, which is where
  [[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos|a mesh concept]]
  could take over.

### The enumeration consequence: shared local dimensions

§6's warning — slot-addressed values are only meaningful relative to one
enumeration — is the reason for the one genuinely new rule the implementation
added. `C2CE` (cell → flattened (cell, edge-of-cell) pairs) *shares*
`C2E.Local` so that its results combine with `C2E`-shaped sparse fields.
Sharing a slot type across two connectivities is only sound if both tables
enumerate each cell's fiber in the same order and pad it in the same places.
The type checker cannot see this; the bind-time check (`check_offset_provider`)
requires tables over one shared local dimension to have the same width and
skip-value presence and, for concrete tables, skip values at the same
positions, and reductions over the shared axis take that structure from any
bound table over it. Note what this does *not* check: that the two tables list
each cell's neighbors in the same *order* — §6's bijection — which is again a
property of the mesh generator.

### Evidence lifetime

§8's caveat — evidence is valid only for the graph it describes — applies to
the check's placement. For compiled backends, `check_offset_provider` runs when
a compiled variant is created, not on every call, because building a table's
type is too slow for the call path; variants are keyed by the *identity* of
the bound table objects, so binding a different table creates a new variant
and re-runs the check, while re-binding the same object reuses the evidence.
The remaining gap is the one §8 names for mutable data: a table modified in
place after the check keeps its old evidence. Embedded calls re-check every
time. Some entry points (the iterator-level `fendef`, DaCe orchestration) do
not run the check at all — evidence that is never established.

## Further reading

- [[personal/havogt/dependent-local-dimensions/dependent-local-dimensions|Dependent local dimensions and connectivity chains]]
  — the chain/forest generalisation and its prior-art section (Dex).
- A. Paszke et al., *Getting to the Point: Index Sets and
  Parallelism-Preserving Autodiff for Pointful Array Programming*, ICFP 2021 —
  Dex's typed index sets, the closest array-language analogue of `Fin<n>`
  dimensions.
- M. Noonan, *Ghosts of Departed Proofs (Functional Pearl)*, Haskell
  Symposium 2018 — carrying checked facts as phantom/brand types in a language
  without full dependent types, the technique `V2E.Local` is an instance of.
