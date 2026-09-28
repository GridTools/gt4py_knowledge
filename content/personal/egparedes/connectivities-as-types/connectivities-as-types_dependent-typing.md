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
> *Dependent types for the imperative programmer — a walkthrough built on one
> example: graph incidence*; part 2 maps it onto gt4py.

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
states none of the conditions that make it correct:

```text
List<int> neighbors(Graph g, int v);
double    weight(Graph g, int v, int e);
```

- `v` and `e` must be in range;
- `neighbors(g, v)` must return exactly the edges incident to `v`, each once;
- `weight` is meaningful only on incident pairs — `(0, 1)` has valid IDs but
  is not an incidence.

Each of these ends up as an assert, a bounds check or a comment. A dependent
type states it in the signature instead, and the caller discharges it.

### 1. A type can mention a value

`std::array<double, 8>` already has a value in its type, but the value must be
a compile-time constant. Dependent types drop that restriction: any value in
scope, including a mesh read at run time, can appear in a type. A **type
family** selects a type from such a value:

```text
Vector<T, n>    // exactly n elements
Fin<n>          // a natural number i with 0 <= i < n

type Vert<Graph g> = Fin<g.nV>;
type Edge<Graph g> = Fin<g.nE>;

double get(Nat n, Vector<double, n> a, Fin<n> i);   // the type of i is the bounds check
```

A `Fin<n>` is a plain integer at run time: the bound is a static claim, not a
stored field, and it travels with the value, so a function taking `Fin<n>`
never re-validates it. The checker reasons symbolically about `n`; it need not
know its numeral.

### 2. An argument's type can mention another argument (Π types)

A **dependent function type** (Π type) lets the *value* of an argument
determine the types of later arguments or of the result:

```text
auto   row(Graph g, Vert<g> v) -> Vector<double, degree(g, v)>;
double at (Graph g, Vert<g> v, Fin<degree(g, v)> k);
```

`k` is a **local slot** in `v`'s row, not an edge ID, and its legal range
changes with `v`. Ordinary signatures cannot express this dependency, and
ragged data needs it.

### 3. Refinements: say *exactly* the neighbors

A length-correct `Vector<Edge<g>, degree(g, v)>` still admits `[2]` for vertex
0. A **refinement** restricts a type by a predicate:

```text
type NeighborList<Graph g, Vert<g> v> =
    List<Edge<g>> es where
        noDuplicates(es) &&
        forall (Edge<g> e) (contains(es, e) == g.incident(v, e));
```

The predicate states three properties: soundness (every listed edge is
incident), completeness (every incident edge is listed) and uniqueness. Together
they force the length to be `degree(g, v)`. Order is extra information: a row
of degree `d` has `d!` valid orderings, and only an added `sorted(es)` makes
one of them canonical.

### 4. Dependent pairs: a later field mentions an earlier one (Σ types)

`struct Row { int n; double* data; };  // data points at n doubles` — the
comment is a dependent type. A **dependent pair** (Σ type) packages a value
with data whose type mentions it:

```text
struct Row { Nat length; Vector<double, length> data; };

type IncidentEdge<Graph g, Vert<g> v> = Edge<g> e where g.incident(v, e);

struct Incidence<Graph g> {             // the edge's type mentions the vertex
    Vert<g> vertex;
    IncidentEdge<g, vertex> edge;
};

type IncidenceField<Graph g> = function(Incidence<g>) -> double;
```

`Incidence<G>{0, 1}` is rejected, so the domain of an `IncidenceField<G>` is
exactly the set of true entries of the matrix. A Π type accepts *any* vertex;
a Σ type packages *one* vertex with something valid for it. The field assigns
one value per *incidence*: `(0, 0)` and `(1, 0)` may carry different weights
although both name edge 0. The dependency restricts the **domain**; the output
is an ordinary `double`.

### 5. Proofs are arguments, and are erased

A statement can itself be a type: `Proof<P>` has a value only when `P` holds.
It is checked evidence, not a Boolean. `false` is a valid `bool`, but a false
statement has no proof. Unfolding the `IncidentEdge` refinement of §4 turns
the evidence into an explicit argument:

```text
double weight(Graph g, Vert<g> v, Edge<g> e);                    // body: assert(g.incident(v, e))
double weight(Graph g, Vert<g> v, Edge<g> e, Proof<g.incident(v, e)> ok);
```

The assert fails at run time, inside the callee. The proof is demanded from
the caller when the call is checked, cannot be forged, and is erased during
compilation. It is rarely written by hand. A loop that scans the incidence
matrix establishes `NeighborList` through an invariant ("the result holds
exactly the incident edges examined so far"), and SMT-backed refinement
checkers (F\*, Liquid Haskell) discharge most such obligations automatically.
Only evidence is erased: computational data that appears in types (lengths,
offsets, indices) stays a run-time value.

### 6. Edge IDs vs. local slots: the enumeration

The edges above one vertex (its **fiber**) can be addressed by edge ID or by
local slot. An exact neighbor list is a **bijection** between the two. Its
codomain gives soundness, surjectivity gives completeness and injectivity gives
uniqueness:

```text
type NeighborOrder<Graph g> =
    function(Vert<g> v) -> Bijection<Fin<degree(g, v)>, IncidentEdge<g, v>>;
```

Once an enumeration is chosen, a field has two equivalent addressings, and
choosing between them means choosing a sparse layout:

| Addressing | Type for a fixed graph `g` | Needs | Storage analogy |
| --- | --- | --- | --- |
| vertex–edge incidence | `Incidence<g> -> double` | the incidence predicate | coordinate-based, like COO |
| vertex–local slot | `(v : Vert<g>) -> Fin<degree(g, v)> -> double` | the degree function | row-and-slot, like CSR |

Degrees alone define the legal slots but not *which* edge sits in each;
**reordering the neighbors requires reordering every slot-addressed value
consistently.** A uniform degree reduces the ragged shape to a rectangular
array, but the edge correspondence still matters. With `nbr : NeighborOrder<g>`
and `field : IncidenceField<g>`, a neighbor reduction is:

```text
double sumAtVertex(Vert<g> v) {
    double total = 0.0;
    for (Fin<degree(g, v)> k : range(degree(g, v)))
        total += field(Incidence<g>{v, nbr(v).forward(k)});
    return total;
}
```

Iterating over the range yields a legal slot by construction, and
`nbr(v).forward(k)` yields an edge together with its incidence evidence, so the
obligations are met without a single hand-written proof.

The presentation can also be inverted: make the neighbor lists primitive and
*derive* incidence from them.

```text
struct Graph {
    Nat nV, nE;
    function(Fin<nV> v) -> Nat degree;
    function(Fin<nV> v, Fin<degree(v)> k) -> Fin<nE> nbr;   // injective in k, for each v
};
```

Soundness and completeness then hold by definition, because the lists *are* the
graph. The type of `nbr` makes every entry a valid edge; what remains to prove
is that no row repeats one. Built from raw integer tables instead, both must be
established: every entry in range, and no repeats in a row.

### 7. Parameter or index: does the type identify the graph?

When the graph is an ordinary argument (a **parameter** of the call), all
graphs share one type, so passing a vertex of one mesh to `neighbors` on
another type-checks. To tell graphs apart, the graph must be an **index** of
the type, e.g. `template<Graph G> struct Mesh`. C++20 allows this only for
compile-time constants; dependent types also allow a mesh loaded at run time.

Even then, "same type ⇒ same graph" is a checking discipline, not a theorem:

- an alias may ignore its index;
- `Fin<g.nV>` distinguishes bounds, not two meshes with the same vertex count;
- two graphs bound at run time have the same type only if they are the same
  variable, or if a proof of their equality is supplied.

Preventing cross-mesh ID mixing needs an **identity-preserving wrapper, or
brand**.

### 8. Validate at boundaries, then reuse the evidence

The checks do not disappear; they move to the boundary where data enters.
Inside the program, indices come from constructs that carry their own
evidence (a range, a neighbor list, an incidence), so nothing there needs
validation. Runtime input acquires a dependent type once, through a checked
decision, which is the only way to construct the value:

```text
auto makeIncidence(Graph g, int v, int e) -> Option<Incidence<g>>;
```

A graph loaded at run time travels with its checked operations:

```text
type NeighborsFor<Graph g> = function(Vert<g> v) -> NeighborList<g, v>;

struct GraphPackage {
    Graph graph;
    NeighborsFor<graph> neighbors;
    IncidenceField<graph> field;
};
```

The loader establishes the contracts once, and consumers reuse them. Evidence
is valid only for the graph it describes; a mutated or different graph needs a
new check. Without full dependent types, a **middle road** keeps most of this
discipline and drops the proof burden: branded index types, validated
constructors, and assertions at the boundary.

## Part 2 — What gt4py can keep of it

gt4py takes §8's middle road: connectivity and dimension classes are the
brands, `check_offset_provider` is the validated constructor, and nothing is
proved.

### The dictionary

| Dependent-typing concept | gt4py today (v1.2.2) | This design |
| --- | --- | --- |
| `Vert<g>` — index type of a mesh entity | `Dimension("Vertex")`, identity by `(value, kind)` | the class `V(DimensionIndex)`; identity is the class (a **brand**, §7) |
| `Fin<degree(g, v)>` — local slot, depends on `g` *and* `v` | `V2EDim = Dimension("V2E", LOCAL)`, a free name | `V2E.Local`: depends on the **declaration** `V2E` (nominally, by nesting), not on `g` or `v` |
| `degree(g, v)` bounded by the padded row length | `max_neighbors` on the table's type | `max_neighbors` / `min_neighbors`: optional class keywords stored on `V2E.Local`, else taken from the table |
| `Option`-valued slots of a padded ragged row | `skip_value` (−1) in the table | same; if `min_neighbors` is declared, skip values must be present iff `min_neighbors < max_neighbors` |
| `NeighborOrder<g>` — the enumeration | a `NeighborTable` / `NdArrayConnectivityField` | unchanged: the table *is* the enumeration and, in §6's inverted presentation, the graph |
| `(v : Vert<g>) -> Fin<degree(g, v)> -> double` — slot-addressed field | `Field[Dims[V, V2EDim], float]` | `Field[Dims[V, V2E.Local], float]` |
| `Incidence<g>` — a Σ-typed position | an untyped `{"Vertex": 3, "V2E": 2}` dict | `MultiDimensionIndex(V(3), V2E.Local(2))`; checked at run time that `V2E.Local` indexes the neighbors of a `V` (the dependency of the second component on the first, but not on the value `3`) |
| the graph value `g` in the signature | a string-keyed `offset_provider` | a bound connectivity: `{V2E: table}` binds the class to one table per binding context |
| `GraphPackage` — graph plus checked operations | — | the provider after `check_offset_provider` (and, in [[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos\|the mesh proposal]], a Mesh object) |
| validate at the boundary, reuse evidence | no check; errors surface as `KeyError` or wrong results | `check_offset_provider` at every entry point, remembered per set of bound tables, then trusted by lowering and backends |

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
  *adopts* a local dimension spells it `Local: TypeAlias = C2E.Local`, so
  `C2CE.Local` and `C2E.Local` are the same type — one slot type, two
  relations enumerating it.
- **A brand per mesh entity (§7).** Nominal identity makes two independently
  declared `Vertex` classes distinct types, statically and at run time. The
  `(name, kind)` equality of the superseded [[shared/dimensions-as-types|dimensions as types]]
  is the unbranded `Fin<g.nV>` of §7: equal names, interchangeable indices.
  Nominal identity is not sufficient to separate two *meshes* that share the
  same declarations, though — see below.

### What moves to a bind-time check

Python has no value-indexed types, so the dependency on the table value `g`
cannot be expressed. The design splits the connectivity type the way §5
splits data from evidence:

- The **class** `V2E` is the static, erasable part: `Domain`, `Codomain`,
  `Local` and, optionally, the declared counts. It is what the frontend and the
  type checker see.
- The **table** is the value `g`. Binding happens per binding context through the
  provider (`{V2E: table}`), and `check_neighbor_table` is the "checked
  decision" of §8: it establishes that the table's domain is `(Domain, Local)`,
  its codomain `Codomain`, its dtype integral, its width equal to a declared
  `max_neighbors` and not below a declared `min_neighbors`, and — if
  `min_neighbors` is declared — its skip values consistent with it.
- **Undeclared counts are taken from the table** when a variant is compiled.
  In §5's terms the counts are computational data that appear in types and
  are *not* erased; they are specialised on per compiled variant rather than
  fixed in the source. The main note's *Binding model* explains why a
  static-only count would bind DSL source to one mesh family.

### What is given up (and why that is acceptable)

- **Validity of the table's entries (§6).** gt4py uses §6's inverted
  presentation: there is no incidence relation apart from the table, so
  soundness and completeness (§3) hold by definition. But its tables hold raw
  integers, not `Fin<nE>`, and both remaining conditions are unchecked: that
  every non-skip entry is a valid `Codomain` index, and that no row repeats
  one. The bind-time check reads a table's
  contents only to compare skip positions (below). Well-formed entries are a
  property of the mesh generator.
- **Per-element ragged degree.** `Fin<degree(g, v)>` becomes
  `Fin<max_neighbors>` plus skip values, i.e. the padded, rectangular
  representation of §6. `min_neighbors == max_neighbors` is exactly the
  uniform-degree case, which is why it is equivalent to "no skip values".
- **Same type ⇒ same table (§7).** `V2E` does not identify a mesh. Two
  different tables bound to `V2E` in two binding contexts give two different,
  both well-typed, programs: the table is a *parameter* of the binding context,
  not an *index* of the type. This is deliberate — it is what lets one DSL
  source run on several meshes — and the one-table-per-declaration dict keeps
  it from becoming ambiguous *within* a context. Guarding against mixing meshes across
  calls (e.g. a field computed on mesh A consumed on mesh B) is out of scope;
  it would need a mesh-level brand, which is where
  [[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos|a mesh concept]]
  could take over.

### The enumeration consequence: shared local dimensions

§6's warning — slot-addressed values are only meaningful relative to one
enumeration — is the reason for the shared-local-dimension rule, the one
relaxation of "one table per declaration per binding context". `C2CE` (cell → flattened (cell, edge-of-cell) pairs) *shares*
`C2E.Local` so that its results combine with `C2E`-shaped sparse fields.
Sharing a slot type across two connectivities is only sound if both tables
enumerate each cell's fiber in the same order and pad it in the same places.
The type checker cannot see this; the bind-time check (`check_offset_provider`)
requires tables over one shared local dimension to have the same width and
skip-value presence and — where a program is compiled — skip values at the
same positions, and reductions over the shared axis take that structure from any
bound table over it. Note what this does *not* check: that the two tables list
each cell's neighbors in the same *order* — §6's bijection — which is again a
property of the mesh generator.

### Evidence lifetime

§8's caveat — evidence is valid only for the graph it describes — applies to
the check's placement. `check_offset_provider` runs at every entry point, and
its result is remembered per set of bound tables, keyed by their *identity*:
binding a different table re-runs the check, re-binding the same object reuses
the evidence for the cost of one hash. The data-dependent part (comparing skip
positions of tables over a shared local dimension) runs only where a program
is compiled, not on the call path. Two gaps remain, both the kind §8 names for
mutable data: a table modified in place after the check keeps its old
evidence, and — as with the compiled-program cache, which is keyed the same
way — a freed table replaced by a new one at the same address could in
principle reuse the old evidence.

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
