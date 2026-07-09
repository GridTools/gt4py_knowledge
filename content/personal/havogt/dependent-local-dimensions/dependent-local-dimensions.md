---
title: Dependent local dimensions and connectivity chains
author: havogt
tags: [type-system, dimensions, unstructured, connectivities, local-dimensions, reduction, neighbor-sum, type-checking]
created: 2026-06-11
status: draft
---

> **TL;DR** Make an unstructured *local* dimension carry the connectivity it
> comes from, so that `V2EDim`'s dependence on `Vertex` is expressed in the
> type, **chained** connectivities (`edge_field(V2E)(E2V)`) become well-typed,
> and the resulting reductions get a type-checked **order**. Two designs: A —
> the dims list as an ordered topology *path* (LIFO reduction); B — dims as an
> order-free dependency *forest* (leaf-rule reduction, supports neighborhood
> products).

> **Prototype**: vendored alongside this document in
> [`dependent-local-dimensions/`](dependent-local-dimensions/) — a numpy mock
> verifying the reduction-order claim (§2) and mypy-verified static encodings of
> both proposals (§4.4: path, §5.4: forest).
>
> Related: builds directly on the dimension redesign in
> [[personal/havogt/dimension-generic-fields|Generic dimensions and statically typed staggering]]
> (dimensions as types, typed connectivities). References to "companion doc"
> below point there.

---

# Investigation: dependent local dimensions and connectivity chains

- **Status**: investigation / pre-ADR design proposal
- **Scope**: unstructured *local* dimensions in `gt4py.next`: expressing that
  `V2EDim` **depends on** `Vertex` (per-vertex neighbor count on irregular
  meshes), allowing **chained** connectivities (`edge_field(V2E)(E2V)`), and
  giving the resulting reductions a well-defined, type-checked **order**.
- **Context**: builds directly on the dimension redesign in
  [[personal/havogt/dimension-generic-fields|Generic dimensions and statically typed staggering]]
  (dimensions as types, typed connectivities) and on the terminology of
  [ADR 0019 — Connectivities](https://github.com/GridTools/gt4py/blob/main/docs/development/ADRs/next/0019-Connectivities.md).
- **Prototype**: [`dependent-local-dimensions/`](dependent-local-dimensions/)
  — a numpy mock verifying the reduction-order claim (§2) and mypy-verified
  static encodings of both proposals (§4.4: path, §5.4: forest).
- **Outcome**: two design proposals — A "dims as a topology path" (ordered
  hops, §4) and B "dims as a dependency forest" (order-free, §5) — with a
  comparison (§6); decision to be recorded as an ADR after review.

## 1. Problem statement

Applying `V2E` to an edge field produces `Field[Dims[Vertex, V2EDim]]`. Three
related deficiencies:

1. **The dependency is not expressed.** `V2EDim` is a *dependent* dimension:
   its extent varies with the current `Vertex` on irregular meshes
   (implemented by padding to `max_neighbors` + skip values). The type
   `Dimension("V2E", kind=LOCAL)` says none of this; the connection to
   `Vertex` (and to the `V2E` table that defines validity) exists only as a
   *string-equality convention* between the local dimension's name and the
   offset-provider key (see §3).
2. **Connectivities cannot be chained.** `edge_field(V2E)(E2V)` is
   semantically meaningful (for each edge, for each of its 2 vertices, the
   values on that vertex's edges) but rejected at decoration time:
   `ValueError: There are more than one dimension with DimensionKind 'LOCAL'.`
   (`common.order_dimensions`, `common.py:1361`).
3. **Reduction order is not expressible.** If the chained field existed as
   `Field[Dims[Edge, E2VDim, V2EDim]]`, nothing in the flat dims list says
   that `neighbor_sum` over `V2EDim` must happen *before* the one over
   `E2VDim` — the flat encoding cannot even represent which local dimension
   depends on which.

## 2. Verification: does reduction order matter?

Setup: `t = edge_field(V2E)(E2V)`, conceptually
`t[e, j, i] = ef[V2E[E2V[e, j], i]]` with `j` the `E2VDim` index and `i` the
`V2EDim` index. The validity of position `(e, j, i)` is
`v2e_table[e2v_table[e, j], i] != skip_value` — note it depends on the
**value** `e2v_table[e, j]` (which vertex), not on any dimension of the
field. We call reducing `V2EDim` first (the dimension created *last*)
**innermost-first** (LIFO).

**(a) Pure value semantics: order-insensitive — under a condition.**
`neighbor_sum`/`min_over`/`max_over` are commutative monoids and skip
positions contribute the identity element, so both orders produce the same
value *provided the implementation applies the correct validity mask*, which
for the outer-first order is a **gathered** mask (`v2e_valid[e2v[e, j]]`).
The prototype mock confirms this numerically — and also shows what happens
without the dependency information: after reducing `E2VDim` first, the
intermediate `Field[Dims[Edge, V2EDim]]` is *semantically dangling* — the
vertex that defined `V2EDim`'s extent and validity has been summed away, no
mask is derivable from the field's remaining dims, and the naive computation
is simply wrong (mock: `[16, 18, 18, 22, 22]` instead of
`[16, 13, 13, 17, 17]`).

**(b) Fast/compiled implementations: order matters operationally.**
Innermost-first is the only order that *streams*: the parent index is live in
the enclosing loop, so validity and table lookups are direct. This is exactly
the shape of the existing single-hop implementations:

- gtfn unrolls every reduction into a flat
  `for i in range(max_neighbors)` fold with a single reduction index
  (`gtfn_ir_to_gtfn_im_ir.py:152,172`) — the parent position is the (live)
  iteration point.
- The embedded reduction reconstructs the skip-value mask from the
  connectivity table broadcast over `(parent_origin_dim, local_dim)` and
  therefore *requires the parent dimension to still be present in the field*
  (`embedded/nd_array_field.py:946-956`, including the comment
  "assumes offset and local dimension have same name").

Outer-first requires materializing the padded multi-hop block *plus* gathered
validity masks: extra memory traffic, no fusion with the producing stencil,
and a transpose across a ragged structure (with CSR-like storage it is not
implementable at all without re-materialization). Additionally, floating-point
non-associativity makes the two orders bitwise different, so a deterministic
order is needed for reproducibility regardless.

**(c) The current code agrees, by prohibition.** All structural blockers
enforce "at most one live local dimension", i.e. they over-approximate the
LIFO discipline by forbidding chains entirely: `check_dims`/
`order_dimensions` (≤ 1 LOCAL dim), the embedded reduction
(`NotImplementedError` for > 1 local dim), ITIR's
`ListType(element_type, offset_type)` carrying exactly one hop, and the DaCe
lowering assuming one `offset_type` local dim per field. Note also that even
if two local dims were admitted, today's canonical dims order would sort them
*lexicographically by name* — unrelated to dependency order — and a repeated
hop (`vertex_field(V2V)(V2V)`) could not be represented at all (same
dimension twice in one dims list).

**Verdict**: the claim is verified. Reduction order is irrelevant only under
fully materialized, gathered validity — exactly what fast implementations
must avoid. The dependency structure ("`V2EDim` is a function of the position
in the dims to its left") both *defines* the legal order (innermost-first)
and is *absent* from the current flat dims encoding.

## 3. Accidental complexity in the current encoding

Worth fixing in the same stroke (all verified against the sources):

- **String coupling**: the local dimension's `value` must literally equal the
  offset-provider key — `Dimension("V2E", kind=LOCAL)`, not
  `Dimension("V2EDim", ...)` — because the embedded reduction does
  `get_offset(offset_provider, axis.value)`. Using a differently named local
  dim fails at runtime with `KeyError: "Offset 'V2EDim' not found..."`.
- **Split declaration**: every connectivity needs *two* user declarations
  kept consistent by hand — the local `Dimension` and the `FieldOffset`
  (`V2E = FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))`) — plus a
  matching offset-provider entry at call time.
- **Direction-obscuring names**: `FieldOffset.source` is the *codomain* (what
  the entries point into), `target[0]` the new origin. The proposal below
  uses `origin`/`codomain` throughout.

## 4. Proposal A — the dims list as a topology path

### 4.1 Core idea

A local dimension is not a free-standing dimension: it is **the application
of a neighbor connectivity**. Make that the representation:

- A connectivity is declared once, with its full type (dimensions-as-types
  world of the companion document):

  ```python
  class V2E(NeighborTable[Vertex, Edge], max_neighbors=6, has_skip_values=True): ...
  class E2V(NeighborTable[Edge, Vertex], max_neighbors=2): ...
  ```

  (origin `Vertex`: where the remapped field lives; codomain `Edge`: what the
  entries point to. The local dimension and the offset tag are *derived* —
  the string coupling and split declaration of §3 disappear.)

- The induced local dimension is `Local[V2E]` (spelled `V2E.dim` where a
  value is needed); it carries origin and codomain by construction.

- A field's dimensions are an **origin dimension plus a stack of hops**:

  ```
  Field[Dims[Edge]]                                # ef
  Field[Dims[Vertex, Local[V2E]]]                  # ef(V2E)
  Field[Dims[Edge, Local[E2V], Local[V2E]]]        # ef(V2E)(E2V)
  ```

  Well-formedness is a **chain condition**: reading left to right,
  `origin(hop₁) = origin dim of the field`, and
  `origin(hopₖ₊₁) = codomain(hopₖ)`; the element values live on
  `codomain(hopₙ)`. The dims list *is* a typed path through the mesh
  topology: `Edge —E2V→ Vertex —V2E→ Edge`.

This answers the `V2EDim[Vertex]` question in a strictly stronger form: the
dependency of `V2EDim` on `Vertex` is `origin(V2E) = Vertex`, implied by the
hop's type rather than annotated by the user; and in a chain, the *parent of
each hop is the preceding path position* — including the case where the
parent is itself a hop entry (the vertex `E2V[e, j]`), which a
`V2EDim[Vertex]` annotation could not distinguish. (Σ-type reading:
`Field[Dims[Vertex, Local[V2E]]] ≅ (Σ v:Vertex. Fin(deg v)) → DType`; the
type system tracks only the dependency *structure*, the extents — tables,
skip masks — remain runtime values, today padded, possibly CSR later. The
representation is an implementation detail behind the path type.)

### 4.2 Typing rules

- **Remap (push)**: applying connectivity `C: origin O', codomain O` to a
  field with origin dim `O` yields origin `O'` with `Local[C]` pushed onto
  the front of the hop stack:
  `Field[Dims[O, H₁…Hₙ]] → Field[Dims[O', Local[C], H₁…Hₙ]]`. The chain
  condition is maintained by construction; applying a connectivity whose
  codomain does not match the field's origin is a type error. (Cartesian and
  staggering shifts keep their existing rule — they replace the origin
  without pushing a hop.)
- **Reduction (pop, LIFO)**: `neighbor_sum`/`min_over`/`max_over` reduce the
  **innermost** (rightmost) hop only:
  `Field[Dims[O, H₁…Hₙ]] → Field[Dims[O, H₁…Hₙ₋₁]]`. The `axis=` argument
  becomes redundant — it can be made optional, retained as a checked
  assertion. Reducing a non-innermost hop is a type error whose message
  states the reason ("`Local[E2V]` cannot be reduced while `Local[V2E]`
  depends on it; reduce `Local[V2E]` first").
- **Pointwise operations** between multi-hop fields require identical
  `(origin, hop-path)`; broadcasting a single-hop/no-hop field against a
  longer matching prefix is the natural extension of today's zero-dim /
  dims-promotion rules (v0: exact match only, mirroring the strict policies
  of the companion documents).
- **Positional identity**: hops are identified by their position in the path,
  not by name — `vertex_field(V2V)(V2V)` is well-formed
  (`Field[Dims[Vertex, Local[V2V], Local[V2V]]]`), which no flat dims set can
  represent.
- **Relaxation (later)**: reducing hop `Hₖ` out of order is semantically
  sound iff every hop to its right has constant extent (no skip values), e.g.
  `E2V` with exactly 2 vertices per edge — loop interchange over rectangular
  extents. v0 enforces strict LIFO; the relaxation is a backend-driven
  follow-up.

### 4.3 Type-system (ts) representation

`ts`/`common` gain a `LocalDimension` carrying its connectivity type
(`NeighborConnectivityType` from ADR 0019) instead of today's
`Dimension(name, kind=LOCAL)` + name convention:

- `FieldType` keeps `dims: list[...]` but the validator replaces "at most one
  LOCAL dim" + lexicographic ordering with the **chain check** (hops are
  ordered by the path, not by name; concrete non-local dims keep today's
  canonical order).
- The remap and reduction rules of §4.2 replace the corresponding cases in
  `return_type_field` / `_visit_reduction`.
- ITIR already nests structurally — `ListType.element_type` is a `DataType`,
  so a two-hop field is `ListType(element_type=ListType(...))`; what is
  missing is transform/backend support, which the LIFO discipline makes
  tractable: a well-typed program only ever reduces the innermost list, so
  gtfn's flat fold generalizes to a *nested* fold loop in which every parent
  index is live (the streaming implementation of §2b), and the embedded
  reduction generalizes by composing validity masks through the live prefix
  of the path (the prototype mock is the reference semantics).

### 4.4 Static (mypy) layer

The encoding rides on dimensions-as-types and is **prototype-verified**
([`dependent-local-dimensions/static_hops.py`](dependent-local-dimensions/static_hops.py)):
`Local[C]` is a generic
dimension type over connectivity types; remap pushes the hop in front of a
`TypeVarTuple` tail; reduction pops the rightmost element:

```python
class Field(Protocol[DimsT, DT]):
    def __call__(
        self: Field[Dims[C, Unpack[Hops]], DT],
        conn: type[NeighborTable[O, C]],
    ) -> Field[Dims[O, Local[NeighborTable[O, C]], Unpack[Hops]], DT]: ...

def neighbor_sum(
    f: Field[Dims[Unpack[Hops], Local[ConnT]], DT],
) -> Field[Dims[Unpack[Hops]], DT]: ...
```

mypy infers the full two-hop chain for `ef(V2E)(E2V)`, accepts exactly the
LIFO reduction sequence, and rejects chain violations (applying `V2E` to a
vertex-origin field) at the call site. Two notes: the inferred hop type is
the structural `Local[NeighborTable[Vertex, Edge]]` rather than the nominal
`Local[V2E]` (sufficient for all checks; nominal fidelity would need
per-connectivity overloads), and "reduce only the innermost" is enforced
because `Unpack` can only precede a *fixed* tail element — the type system's
one-`TypeVarTuple` restriction works in our favor here.

## 5. Proposal B — order-free: dims as a dependency forest

Proposal A gives the hop segment of the dims list a *semantic order*. If the
design premise is instead that **dimension order carries no semantics** (dims
are a set), the dependency must move out of the position and into the
dimension type itself: every local dimension *names its parent*.

Note that this premise is less radical than it sounds for GT4Py: today
`check_dims` rejects non-canonically-ordered dims lists, i.e. order is
already a *normal form*, not information. Proposal B extends that stance
uniformly; proposal A would carve out an ordered sub-segment as an exception.

### 5.1 Core idea

```
Field[Dims[Edge]]                                      # ef
Field[Dims[Vertex, Local[V2E, Vertex]]]                # ef(V2E)
Field[Dims[Edge, Local[E2V, Edge],
           Local[V2E, Local[E2V, Edge]]]]              # ef(V2E)(E2V)
```

`Local[C, P]` is the local dimension induced by connectivity `C`, *depending
on* parent dimension `P` — where `P` is either a root (non-local) dimension
or another local dimension. This is exactly the originally suggested
notation, generalized: `V2EDim[Vertex]` ≙ `Local[V2E, Vertex]`, and after a
second hop `V2EDim[E2VDim[Edge]]` — the parent of the `V2E` hop is the `E2V`
*hop* (the position that holds the vertex), which a parent-*dimension*
annotation alone could not distinguish from the `Vertex` root of another
field.

The dims collection is a **set**; its elements form a **dependency forest**
(roots = non-local dims, every local dim points at its parent, all parents
present). No order is needed because identity and dependency are both carried
by the types: same-connectivity *chains* stay distinct through their ancestry
(`Local[V2V, Vertex]` vs `Local[V2V, Local[V2V, Vertex]]`).

### 5.2 Rules

- **Well-formedness**: every local dim's parent is in the set, and the
  parent's value type (codomain of the parent hop, or the root dim itself)
  equals the origin of the local dim's connectivity. Acyclicity holds by
  construction.
- **Remap (reparenting)**: applying `C: origin O', codomain O` to a field
  with root `O` replaces the root by `O'`, adds `Local[C, O']`, and
  **reparents** every dim that referenced `O` onto the new hop:
  `{O, Local[X, O], ...} → {O', Local[C, O'], Local[X, Local[C, O']], ...}`.
  (In A, hops never mention the root, so no rewriting is needed — this is
  the price B pays for referential dependency.)
- **Reduction (leaf rule)**: a dimension may be reduced iff it is a **leaf**
  of the forest — no other dimension depends on it. This is the *true
  operational invariant* behind §2 (a mask/extent is computable exactly while
  every dimension it depends on is still live); A's LIFO is its
  linearization. Consequences: on a chain the leaf is unique (same constraint
  as A); **independent sibling hops commute** — reducing either first is
  legal and the results agree (§2a, masks both keyed by the live shared
  parent — also operationally streamable). The `axis=` argument is therefore
  *required* (there is no "innermost" default), and checked to be a leaf.
- **Pointwise/broadcast**: dims are matched by type equality (ancestry
  included), order-free; implementations pick a canonical layout internally.
  Notably this admits the **sibling product**: `vf(V2E) * wf(V2V)` broadcasts
  over the shared root to
  `Field[Dims[Vertex, Local[V2E, Vertex], Local[V2V, Vertex]]]` — for each
  vertex a (ragged) matrix over its edges × its vertex neighbors. Proposal A
  cannot *represent* this (a path is linear); under B it falls out of the
  ordinary broadcast rule.

### 5.3 Costs and limits

- **Verbosity**: ancestry lives in the type; deep chains read poorly without
  sugar (aliases; rendering such as `V2E@E2V@Edge` in diagnostics).
- **Same-type siblings collide**: a set cannot hold `Local[V2E, Vertex]`
  twice, so the *self*-product "edges × edges of the same vertex" needs an
  explicit disambiguating alias/tag (the same restriction xarray and coordax
  have via unique dim names). A cannot represent siblings at all, so this is
  still a net expressiveness gain.
- **Static layer**: Python's type system has no unordered type collections,
  so order-free semantics can only be *spelled* in a canonical normal form
  (proposal: topological — parents before children — with today's
  kind-then-name rule as tie-breaker). Moreover the reparenting remap is a
  type-level substitution, which (as with `Dual`, companion doc §4) is not
  generically expressible — the static layer enumerates *forest shapes*
  (bounded codegen, one overload per shape) instead of A's simpler push.
- **ts representation**: `LocalDimension(connectivity, parent)` becomes
  recursive; `return_type_field` performs the reparenting substitution;
  `_visit_reduction` checks leaf-ness. Lowering is unchanged relative to A —
  a topological traversal of the forest gives the loop nest, leaves are the
  innermost loops — plus rectangular sibling loops for products.

### 5.4 Static (mypy) layer — prototype-verified

[`dependent-local-dimensions/static_forest.py`](dependent-local-dimensions/static_forest.py)
verifies the encoding for the realistic envelope (chains of depth ≤ 2, two
siblings): the **reparenting remap** infers the exact nested type for
`ef(V2E)(E2V)` (including rewriting the existing hop's parent inside the
result type — expressible per shape because the shape is fixed in each
overload), the **leaf rule** accepts leaf-then-parent on chains and *both*
orders on siblings (with exact result types), and reducing the middle of a
chain is rejected.

## 6. Choosing between A and B

| aspect | A — path | B — forest |
|---|---|---|
| order of dims | hops are semantically ordered (LIFO) | no semantics (set + canonical normal form) |
| dependency encoding | positional (on the prefix) | referential (`Local[C, P]` in the type) |
| user notation | `Local[V2E]` | `V2EDim[Vertex]`, `V2EDim[E2VDim[Edge]]` |
| sibling hops / neighborhood products | unrepresentable | supported (incl. commuting reductions) |
| legal reduction orders | exactly one (innermost) | all orders the dependency structure allows |
| `axis=` argument | redundant | required, leaf-checked |
| repeated connectivity | chains ✓ (positional identity) | chains ✓ (ancestry); same-parent siblings need tags |
| type verbosity | compact | grows with chain depth (sugar needed) |
| static (mypy) layer | full chain typing via simple push | per-shape overloads incl. reparenting (verified ≤ depth 2) |
| ts machinery | chain validator; no substitution | recursive parent; reparenting substitution |
| relation to status quo | introduces an ordered segment | extends "order is a normal form" uniformly |

Assessment: **B is the semantically cleaner model** — the leaf rule is the
true invariant of §2 and A's LIFO is just its linearization; B is also the
one consistent with treating dimension order as a normal form everywhere,
and it unlocks neighborhood products. Its costs are surface verbosity and a
heavier (but bounded and prototype-verified) static encoding. **A is the
smaller step** — lighter types, simpler rules — and is exactly the
restriction of B to linear forests with an implicit axis, so the substantial
groundwork is shared (single typed connectivity declaration, derived local
dims and offset tags, nested `ListType` lowering, gathered-mask embedded
semantics): landing the shared core first keeps the choice open, and a later
A→B migration is additive.

## 7. What this fixes

Both proposals fix every item; rows where they differ list both.

| Deficiency | Resolution |
|---|---|
| dependency of `V2EDim` on `Vertex` unexpressed | implied by `origin(V2E) = Vertex` in the hop type (B: additionally explicit, `V2EDim[Vertex]`) |
| `edge_field(V2E)(E2V)` rejected | well-typed: A — path `Edge —E2V→ Vertex —V2E→ Edge`; B — forest `{Edge, E2VDim[Edge], V2EDim[E2VDim[Edge]]}` |
| illegal reduction order undetectable | type error: A — only the innermost hop is reducible (LIFO); B — only forest leaves are reducible |
| local-dim/offset name string coupling | local dim derived from the connectivity |
| split declaration (`Dimension` + `FieldOffset`) | single typed connectivity declaration |
| repeated hop (`V2V` twice) unrepresentable | A — positional identity in the path; B — distinct ancestry |
| `axis=` argument redundancy | A — optional (checked); B — required (leaf-checked, fits order-free semantics) |

## 8. Open questions

1. **Multi-hop fields at operator boundaries**: the type can express
   `Field[Dims[Edge, Local[E2V], Local[V2E]]]` as a parameter/return, but the
   *storage* is a padded block whose validity needs the gathered masks of
   §2a. v0 proposal: multi-hop fields are intermediates within one operator
   (reductions must close the path before return); lifting this needs a
   sparse-field storage story (explicit mask or CSR).
2. **Interaction with dimension variables** (companion doc Part III): do
   `DimsVar`s range over hop stacks? v0: dim variables bind origin and
   non-local dims only; a dedicated `HopsVar` (path-polymorphic operators,
   e.g. a generic "reduce all the way down") is a follow-up with real use
   cases to be collected first.
3. **Staggering interplay**: orthogonal — the origin dimension of a path can
   be staggered (`Field[Dims[Staggered[I], Local[...]]]` is excluded for
   unstructured meshes anyway; vertical staggering composes with `Koff`).
4. **`max_neighbors` in the type?** Stays a compile-time parameter of the
   connectivity (as in `NeighborConnectivityType`), not a type parameter of
   the field; nothing in the rules above needs it statically.
5. **User-constructed sparse fields** (data already laid out as
   `Field[Dims[Vertex, Local[V2E]]]`, e.g. per-edge-of-vertex coefficients):
   supported as today (single-hop sparse fields exist), now with the
   validity source attached by type instead of by name.
6. **Dynamic offsets / `as_offset`**: out of scope; both models assume
   statically known connectivities (consistent with ADR 0019's
   compile-time `ConnectivityType`).
7. **(B) Canonical normal form**: exact specification of the topological
   spelling order (parents before children, kind-then-name tie-break) and
   whether annotations in non-canonical order are normalized or rejected.
8. **(B) Disambiguation tags** for same-type siblings (the self-product
   "edges x edges of one vertex"): alias subclassing vs. an explicit tag
   parameter; how the tag surfaces in diagnostics and storage names.

## 9. Staged plan

1. **U0 — ts `LocalDimension` + chain validator** (inert): represent the
   connectivity in the local dimension, derive the offset tag, keep current
   single-hop behavior; delete the name-equality convention.
2. **U1 — typed connectivities + `Local[...]`** in `common`: rides on stages
   D0/D1 of the dimension redesign (single declaration, static remap typing).
3. **U2 — embedded multi-hop**: generalize `premap` and `_make_reduction` to
   gathered prefix masks (reference semantics = prototype mock); feature
   tests for `ef(V2E)(E2V)` with LIFO reductions, including a repeated-hop
   case.
4. **U3 — compiled multi-hop**: GTIR nested `ListType` through the
   transforms; gtfn nested-fold codegen; DaCe equivalent; exclusion-matrix
   markers per ADR 0015 until each backend lands.

Stages U0-U3 are *shared* between proposals A and B (the surface semantics —
ordered hops vs. forest, implicit vs. required `axis=` — only affects the
validator and the deduction rules in U0/U2). The A-vs-B decision must be
taken in the ADR for U0, but an A→B migration later is additive (a path is a
linear forest).

## 10. Prior art

- **Dex** treats ragged structures with dependent index sets
  (`Fin (deg v)`), validating the Σ-type reading; its tables-indexed-by-pairs
  are exactly the path positions used here.
- **JAX `segment_sum`** is the CSR realization of the innermost-first
  reduction (flattened values + segment ids = parent indices); a useful
  target shape for a future non-padded runtime representation.
- **coordax** (see companion doc §6) has no ragged story: named axes are
  rectangular; `cmap` cannot vectorize over a dependent extent — promoting
  the dependency into the dimension *type* is precisely what its value-level
  encoding cannot do.
- **xarray / coordax order-free alignment** is the precedent for proposal
  B's matching-by-identity (and for its unique-name restriction on
  same-type siblings); B transplants that model to the *type* level and adds
  the dependency forest on top.
