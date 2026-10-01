---
title: "Connectivities as types: one declaration for offset, local dimension and provider key"
author: egparedes
tags: [type-system, type-checking, dimensions, local-dimensions, connectivities, offset-provider, unstructured, neighbor-sum, reduction, frontend, foast, gtir, embedded, gtfn, dace, nominal-types, dependent-types, metaclass, serialization, fingerprint, staggering, axis-dimensions, cw-complex, exterior-calculus, domain, dimension-kind, ugrid, sgrid, migration, adr, tech-debt, shared-local-dimensions, implemented]
created: 2026-09-17
updated: 2026-10-01
status: draft
---

> **TL;DR** A neighbor connectivity in `gt4py.next` is spread over three
> user-authored objects that must agree by *string equality* — the
> `FieldOffset` tag, the local `Dimension` name and the `offset_provider` key —
> plus a fourth, hidden one: the Python variable the `FieldOffset` is bound to.
> Make the connectivity a **class** that *contains* its local dimension, is its
> own provider key, and whose identity — like every dimension's (ADR 0028) — is
> its qualified Python name. In the IR it is named by its local dimension's
> tag, so shifts, reductions and sparse arguments all find the table under one
> string:
>
> ```python
> class V2E(NeighborConnectivity[V, E]):
>     class Local(LocalDimensionIndex): ...
>
> neighbor_sum(a(V2E), axis=V2E.Local)
> program(a, out=out, offset_provider={V2E: v2e_table})
> ```
>
> `FieldOffset`, bare-name provider keys (`{"V2E": ...}`) and the
> `V2EDim`-next-to-`V2E` naming convention disappear. Of the ten cross-object
> identity constraints in the current tree, six (A1–A5 and A9, the embedded
> position-dict key) dissolve by construction, three (A6–A8) collapse into a
> single bind-time check, and one (A10, the `AxisLiteral` round-trip) remains,
> re-expressed over `tag` instead of `Dimension.value`.

> **Implementation.** The design below is implemented on the stacked
> GridTools/gt4py PRs listed in [Implementation](#implementation) (top branch
> `connectivities-as-types-8-typed-positions`), with ADR 0028 (dimensions as
> nominal types) and ADR 0029 (connectivities as types). The *Problem* section
> and the constraint catalogue in the appendix
> [[personal/egparedes/connectivities-as-types/connectivities-as-types_research|Tag and name constraints — full catalogue]]
> describe gt4py `main` (`b3c53fa7e`, v1.2.2), the tree the stack is based on.
> For the type-theoretic background — why the local dimension is owned by the
> connectivity, and what a Python type can and cannot say about a neighbor
> table — see
> [[personal/egparedes/connectivities-as-types/connectivities-as-types_dependent-typing|Background: dependent typing for mesh connectivities]].

> **Relation to existing proposals.** This is the *single-hop base* that
> [[personal/havogt/dependent-local-dimensions/dependent-local-dimensions|Dependent local dimensions and
> connectivity chains]] §6 calls "the shared core" and §9 stages as U0/U1: one
> typed connectivity declaration, derived local dimension, derived offset tag.
> It takes no position on chains, reduction order or the path-vs-forest
> choice, but it does diverge from that proposal's U0/U1 in two places
> (`min_neighbors` instead of `has_skip_values`; a nested `V2E.Local` instead
> of a generic `Local[C]`) — see [Open questions](#open-questions--follow-ups).
> The remap typing it needs is the `Connectivity[NewD, D0]` rule of
> [[personal/havogt/dimension-generic-fields/dimension-generic-fields|Generic dimensions and statically
> typed staggering]], whose Part II also proposed the `Staggered[D]` shape
> used here. It **supersedes** [[shared/dimensions-as-types|Dimensions as
> types]]: the dimension design below (ADR 0028) replaces that note's
> `(name, kind)` value identity and interning registry with type identity, and
> is stated here in full. It overlaps with
> [[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos|A mesh concept with
> first-class halos]], which proposes replacing the same `offset_provider`
> dict with a Mesh object; this note re-types the dict and should be read as
> the typed substrate that proposal can bind through. The staggering half
> draws its vocabulary from
> [[personal/egparedes/discretization-independent-fd-syntax|A discretization-independent surface
> syntax for finite-difference computations]] §3.3 (half-index notation) and
> §6.5 (the Arakawa placement map); that note's location types sit above this
> one, and its §3.3 cross-link back to here was added alongside this note, so it
> is a pointer rather than independent corroboration.

## Problem / motivation

Line references in this section are against gt4py `main` at `b3c53fa7e`
(v1.2.2, 2026-09-03), relative to `src/gt4py/next/`.

### Four strings, one dict lookup

To use one connectivity on an unstructured mesh, a user today writes:

```python
V2EDim = Dimension("V2E", kind=DimensionKind.LOCAL)                 # (N3) local dim name
V2E    = FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))   # (N1) tag, (N2) variable name
...
program(..., offset_provider={"V2E": table})                        # (N4) provider key
```

Four independently authored strings — N1 the tag, N2 the Python variable
name, N3 the local dimension's name, N4 the dict key — all of which must be
identical, none of which is checked against the others at declaration time.
Every consumer does `common.get_offset(offset_provider, <some string>)`
(`common.py:1200-1209`), and *which* string it uses depends on the execution
path and the operation:

| Path | Operation | String used | Where |
| --- | --- | --- | --- |
| embedded | shift `a(V2E[1])` | N1 `FieldOffset.value` | `ffront/fbuiltins.py:494` |
| embedded | `neighbor_sum(..., axis=V2EDim)` | N3 `axis.value` | `embedded/nd_array_field.py:983` — with the comment `# assumes offset and local dimension have same name` |
| compiled | shift | **N2 `foast.Name.id`** | `ffront/foast_to_gtir.py:305, 331` |
| compiled | reduction | N3 `ListType.offset_type.value` | `iterator/transforms/unroll_reduce.py:47` |
| compiled | sparse field *argument* | N3 `dim.value` | `codegens/gtfn/gtfn_module.py:95`; `dace/lowering/gtir_to_sdfg.py:581` |

Two of these were confirmed by execution against v1.2.2: embedded and compiled
execution key the *same* program on *different* strings, and a reduction whose
offset tag differs from its local-dimension name raises
`KeyError: "Offset 'Neigh' not found in offset provider."` in embedded while the
roundtrip backend passes it. GridTools/gt4py#1789 lifted the tag = local-dimension
requirement only for the *shift* path under gtfn. The transcripts, the
regression-test coverage and what was read rather than run are in
[[personal/egparedes/connectivities-as-types/connectivities-as-types_research|the constraints appendix]] §7.

### The declaration duplicates the type

`FieldOffset(value, source=S, target=(T, L))` carries *exactly* the
information in `NeighborConnectivityType(domain=(T, L), codomain=S)` plus a
name — with inverted vocabulary (`FieldOffset.source` is the connectivity's
**codomain**; `target` is its **domain**) and no cross-check between the two.
Three of the identity constraints in the catalogue exist purely to keep this
duplicate in sync, and none is validated eagerly: a mismatch surfaces as a
`KeyError`, a bare `assert`, or — per the #1789 test docstring — silently
wrong results.

### The Cartesian path already solved this

`field(IDim + 1)` builds a `CartesianConnectivity` (`common.py:1241`) and
lowers to `itir.CartesianOffset(domain: AxisLiteral, codomain: AxisLiteral)`
(`iterator/ir.py:99-101`): the IR node carries the two **dimensions**. There
is no *separate* offset tag and no provider lookup. Dimension *names* still
cross gtfn and DaCe as strings (`AxisLiteral.value`, `TagDefinition`,
`i_<dim>_gtx_<kind>` map variables), so the format constraints F5/F8/F9 apply
to both paths; the identity constraints A1–A5 apply to the unstructured path
only. Unstructured shifts still lower to `itir.OffsetLiteral(value: str)` and
a dict lookup. Every constraint above
lives on the unstructured side only, and the reason is not conceptual — it is
that the neighbor *table* has to be supplied at run time, and a string was
the easiest key.

### Concept count

The appendix inventories the vocabulary: **~25 distinct concepts** across 10
layers (5 type-system representations of "an offset", 4 IR node kinds, 8
runtime connectivity classes, 4 provider aliases, 2 connectivity type classes,
2 declaration classes) for
what is one idea — *a mapping between two index spaces, plus a name for it*.

## Proposal

### Concepts

| Concept | Role | Replaces |
| --- | --- | --- |
| `DimensionIndex` | a dimension is a subclass, an index along it an instance (ADR 0028); the root — mesh locations, axes and local dimensions are all subclasses, so every `type[DimensionIndex]` annotation in the tree keeps its meaning | `Dimension` instances |
| `CartesianAxisIndex(AnyCartesianAxisIndex)` | a **declared Cartesian axis**: an index space with integer index arithmetic and exactly one staggered partner (see [Cartesian axis dimensions](#cartesian-axis-dimensions)). `AnyCartesianAxisIndex` is either cell class of such an axis, declared or derived | — (new; today `kind != LOCAL` is the nearest approximation, and it does not separate a Cartesian axis from a mesh location) |
| `Staggered[D: CartesianAxisIndex]` | the axis's derived partner, at the half-integer positions of `D`; a real, interned class (see [`Staggered[D]`](#staggeredd)) | the `_Staggered` name prefix |
| `LocalDimensionIndex(DimensionIndex)` | a local dimension: **owned** (declared nested in, or adopted by, its connectivity), or **owner-less** (optionally with `size=n`, which sets `max_neighbors = min_neighbors = n`); carries the neighbor counts | `Dimension(..., kind=LOCAL)`, `_CONST_DIM` |
| `NeighborConnectivity[Domain, Codomain]` | the **static part of a neighbor table's type**: the shift handle in DSL code and the provider key; contains its `Local`; holds no data and is never instantiated | `FieldOffset`, the `V2EDim` convention |
| `MultiDimensionIndex[D, *Ls]` | a position in the product of a primary and local dimensions, a tuple of indices; user-facing, nothing in the toolchain requires it | — (new; the iterator-embedded position dicts are keyed by dimension classes, not name strings) |

A declaration is not a concept beside the neighbor table: it is the half of
the table's type that a Python type can express. The "true" type of a local
index is `Fin<degree(table, v)>` — it depends on the table *value* and on the
element whose neighbors are visited — and Python types cannot mention values.
The declaration keeps what a checker, the IR and the fingerprint can see
(*which* neighbor relation an index belongs to, as the nominal type
`V2E.Local`, plus the domain, codomain and any declared counts); the bound
table completes it at each binding ([Binding model](#binding-model)), and the
value-dependent part becomes a validate-at-the-boundary check whose result is
reused downstream. The
[[personal/egparedes/connectivities-as-types/connectivities-as-types_dependent-typing|dependent-typing appendix]]
develops this with a worked example and lists what is kept, checked and
given up. The split is what lets DSL code be written, type-checked and
compiled before any mesh is loaded, and one compiled program serve many
meshes.

Nothing is an instance of `V2E`: `NeighborConnectivity` is **not** a
`Connectivity`, and instantiating a declaration raises. The table stays a
`Connectivity` implementation and is *checked against* the declaration.
`Field.premap`/`__call__` accept either a table or a declaration, as they
accepted a `FieldOffset` (which was not a `Connectivity` either).

`LocalDimensionIndex` subclasses `DimensionIndex` rather than a separate
root: `Dimension` is `type[DimensionIndex]` and eve validates `type[X]` by
`issubclass`, so a sibling root would force every `type[DimensionIndex]`
annotation in the tree to widen and then admit local dimensions everywhere
anyway. The tree tells local dimensions apart by a runtime `kind` check, and
constructors whose parameter must be a primary dimension
(`NeighborConnectivity[Domain, Codomain]`, `MultiDimensionIndex`) check that
it is not local. `Staggered[D]` instead takes its parameter from a narrower
level *below* the root, which is what lets it reject a local dimension
statically at no widening cost — see [Cartesian axis dimensions](#cartesian-axis-dimensions).

### What stays, and why

The runtime and compile-time connectivity objects of ADR 0019 stay. They are
*produced by* or *checked against* the declaration instead of being authored
in parallel with it, and the two that duplicated the declaration are renamed
and slimmed so that each owns only what the others cannot know:

| Artifact | What it is | Role after this proposal |
| --- | --- | --- |
| `Connectivity` | the runtime protocol: a `Field` of integer indices with a `codomain`, which `field(conn)` / `Field.premap` dispatch on | unchanged. It unifies neighbor tables, Cartesian shifts and `as_offset` index fields, so embedded execution has one remap entry point. Essential. |
| `CartesianConnectivity` | the `Connectivity` behind `I + 1` and `I + 0.5`; carries the ADR 0026 index arithmetic | unchanged. The Cartesian counterpart of a bound table: it needs no declaration and no provider. Essential. |
| `NeighborTable` / `NdArrayConnectivityField` | the data: a 2-D table over `(Domain, Local)` with values in `Codomain` | unchanged; `check_neighbor_table` checks it against the declaration. Essential. |
| `NeighborTableType` (was `NeighborConnectivityType`) | the type of a table *bound to a declaration* | renamed and slimmed; see below |
| `ts.ShiftType` (was `ts.OffsetType`) | the frontend type of a shift *argument* in DSL code | renamed and re-fielded; see below |

`NeighborTableType` holds exactly the facts a declaration may leave open: element
`dtype`, the skip-value sentinel and `max_neighbors`; `domain` and `codomain` are
derived from the declaration. It is built where a table is bound
(`check_neighbor_table`, `offset_provider_to_type`) or given directly for
ahead-of-time compilation (`table_types`), never authored next to the declaration.
A table alone cannot name its declaration — a sharer's table has the owner's domain
dims and a different codomain — so the record is built from the provider key,
taking whichever of the local dimension's `owner` or `sharers` the key names. A
table bound under a name no declaration answers to (hand-written IR) gets the
structural base `ConnectivityType`, which is also what `NeighborTable.__gt_type__()`
returns and what types general connectivity values such as the index field of
`as_offset`. It is what fingerprints a compiled variant: the owner's and a sharer's
record over one table fingerprint differently.

`ts.ShiftType` is what `V2E`, `V2E[1]`, `I + 1`, `K + 0.5` and `as_offset(K, f)`
type as during FOAST deduction, and what the lowering reads to emit `OffsetLiteral`
/ `CartesianOffset`. Fields: `domain` (a tuple — one dimension for Cartesian shifts
and `V2E[i]`, two for `V2E`), `codomain`, and `tag` (`None` for untagged Cartesian
shifts), printed as `Shift[tag: E -> (V, V2E.Local)]`. It carries no counts, dtype
or skip value because the frontend never needs them, and it is gone once
`foast_to_gtir` has read its `tag`. The name avoids `ts.ConnectivityType`, which
`common` already uses; "shift" is the word this note, the frontend errors and users
use for `a(V2E)`. The iterator-level `it_ts.OffsetLiteralType` /
`it_ts.CartesianOffsetType` keep their names: they type IR literals.

### Sketch

```python
# ── dimensions (ADR 0028) ─────────────────────────────────────────────────
class DimensionMeta(type):
    @property
    def tag(cls) -> Tag:                       # identity, and the IR spelling
        return f"{cls.__module__}.{cls.__qualname__}"
    kind: DimensionKind
    __hash__ = type.__hash__                   # __eq__ stays for `I == 5` -> Domain
    def __add__(                               # I + 1, K + 0.5; a Cartesian axis only
        cls: type[AnyCartesianAxisIndex], offset: int | float
    ) -> Connectivity: ...


class DimensionIndex(metaclass=DimensionMeta):
    value: int                                 # an index along the dimension

type Dimension = type[DimensionIndex]          # PEP 695: `Dimension("I")` is not callable


class AnyCartesianAxisIndex(DimensionIndex):   # either cell class of a Cartesian axis;
    ...                                        #   `D + 1`, `as_offset`, ranges need this


class CartesianAxisIndex(AnyCartesianAxisIndex):
    ...                                        # a *declared* axis; `Staggered` takes these


class LocalDimensionIndex(DimensionIndex):     # kind is LOCAL; `kind=LOCAL` elsewhere is an error
    owner: ClassVar[type[NeighborConnectivity] | None]   # None: owner-less; set by the owner
    sharers: ClassVar[tuple[type[NeighborConnectivity], ...]]   # later declarations over it
    max_neighbors: ClassVar[int | None]        # the counts live here, not on the connectivity
    min_neighbors: ClassVar[int | None]        # min == max  <=>  no skip values
    def __init_subclass__(cls, *, size: int | None = None): ...   # size: max = min = size


class MultiDimensionIndex[D: DimensionIndex, *Ls](tuple[D, *Ls]): ...   # shape checked at runtime


# the TYPE_CHECKING form; at runtime a metaclass subscription builds the
# interned classes, since a PEP 695 alias fails `issubclass` (ADR 0026)
class Staggered[D: CartesianAxisIndex](AnyCartesianAxisIndex): ...

Staggered[KDim]                                # KDim's derived partner;
                                               #   Staggered[Staggered[KDim]] is a type error


# ── connectivities (ADR 0029) ─────────────────────────────────────────────
class ConnectivityMeta(type):
    tag: Tag                                   # the declaration's qualified name
    offset_tag: Tag                            # how the IR names it, see "Identity"
    domain: Dimension; codomain: Dimension     # read from the base subscription
    def __call__(cls, *a, **kw) -> NoReturn: ...        # no instances
    def __getitem__(cls, item): ...            # V2E[1] = bound_table()[Local(1)] in embedded;
                                               # NC[V, E] forwarded to __class_getitem__
    def __gt_type__(cls) -> ts.ShiftType: ...   # Shift[offset_tag: Codomain -> (Domain, Local)]
    def bound_table(cls) -> NeighborTable: ... # the table bound in the current embedded run


class NeighborConnectivity[Domain: DimensionIndex, Codomain: DimensionIndex](
    metaclass=ConnectivityMeta
):
    # every subclass declares `Local` (a nested class or a TypeAlias); it is
    # deliberately *not* annotated here, see "The local dimension"
    def __init_subclass__(cls, *, max_neighbors=None, min_neighbors=None): ...
    # writes Local.owner and the counts onto Local; subclassing a declaration is an error


def check_neighbor_table(connectivity, table_or_type) -> NeighborTableType: ...   # A6–A8 as one check


# ── user code ──────────────────────────────────────────────────────────────
class V(DimensionIndex): ...                   # mesh locations: no index arithmetic and
class E(DimensionIndex): ...                   #   no staggered partner
class C(DimensionIndex): ...
class CE(DimensionIndex): ...                  # flattened (cell, edge-of-cell) pairs

class KDim(CartesianAxisIndex): ...            # a Cartesian axis: `KDim + 1`, Staggered[KDim]


class V2E(NeighborConnectivity[V, E], max_neighbors=6, min_neighbors=5):   # counts optional
    class Local(LocalDimensionIndex): ...                            # explicit, always


class C2E(NeighborConnectivity[C, E]):
    class Local(LocalDimensionIndex): ...


class C2CE(NeighborConnectivity[C, CE]):       # a flattened sparse pattern:
    Local: typing.TypeAlias = C2E.Local        # shares C2E's neighbor axis


class LsqUnk(LocalDimensionIndex, size=3): ...                       # owner-less local axis


@field_operator
def op(a: Field[Dims[E], float]) -> Field[Dims[V], float]:
    return neighbor_sum(a(V2E), axis=V2E.Local) + a(V2E[0])


program(a, out=out, offset_provider={V2E: v2e_table})
a(as_offset(KDim, k_offsets))                 # was as_offset(Koff, ...)
```

Vocabulary: a declaration is a mapping from one index space to another —
`Domain` is the dimension it maps from (the dimension a remapped field lands
on), `Codomain` the dimension its entries point into. A bound table is a
field over `(Domain, Local)` with values in `Codomain`, i.e. the *table's*
domain is the declaration's domain extended by the local axis:
`table.domain.dims == (V2E.domain, V2E.Local)`. `FieldOffset.source`/`target`
obscured the direction (`source` was the codomain);
[[personal/havogt/dependent-local-dimensions/dependent-local-dimensions|dependent local dimensions]]
§3 uses `origin` for the same role, which is avoided here because `origin`
already names a buffer's origin in gt4py (`__gt_origin__`).

### Identity is the qualified Python name

A dimension's or connectivity's identity is the Python type; its tag is
`f"{cls.__module__}.{cls.__qualname__}"` (except `Staggered[D]`, whose tag
embeds its base's full tag, rule 1; a connectivity is *named in the IR* by its
`offset_tag`, rule 7). The static view (checkers see nominal types) and the
runtime view (equality is `is`) agree by construction, and the tag is a
valid, unique IR string. This is the point on which this note supersedes
[[shared/dimensions-as-types|dimensions as types]] and GridTools/gt4py#2844,
which chose `(name, kind)` value equality plus an interning registry so that
the test tree's many independently declared `IDim`s stay interchangeable.
Under value equality the `typing` subscription cache aliases
`Field[Dims[I]] is Field[Dims[I2]]` for two distinct same-named classes, so
the static and runtime views disagree exactly there; under type identity
that aliasing disappears. The cost is that the many function-local dummy
dimensions in `tests/next_tests` move to module level. The consequences,
stated as rules:

1. **Reconstruction from the IR is an import.** `resolve(tag)` imports the
   longest importable module prefix and walks the rest as attributes, the way
   `pickle` references a class, so a nested `V2E.Local` resolves naturally and a
   redefined declaration resolves to the new class. `AxisLiteral` stores only the
   tag. Memoization, `resolve_loaded` and the `Staggered[D]` grammar: [[personal/egparedes/connectivities-as-types/connectivities-as-types_identity|the identity appendix]].
2. **Types reaching the IR must be importable.** `<locals>` in a qualname is
   rejected at class creation (for dimensions and connectivities); pickle's
   own `save_global` stays the authoritative check. Interactive `__main__`
   (REPL, notebooks, `python -c`) matters — every notebook declares its
   dimensions there — so the process-pool runner (`BUILD_JOBS_MODE=process`)
   detects a job that references such a class and, with a warning,
   **compiles it in the calling thread** instead of failing in a spawn worker.
3. **No registry.** Classes pickle by reference. The one narrow `copyreg` hook is
   for `Staggered[D]`, whose bracketed qualname `save_global` cannot look up.
4. **Fingerprints depend on qualified names**, so a redefinition under the same
   name does not reuse artifacts and the ADR 0023 build cache invalidates on
   module renames.
5. **Codegen names need injective mangling.** `codegen_name(tag)` is a prefix
   escape (`_`→`_u`, `.`→`_d`, `[`→`_l`, `]`→`_r`); the obvious alternative is
   not injective.
6. **Staggering is part of the dimension change.** The `_Staggered` name prefix
   needs the name→class lookup that type identity removes, so `Staggered[D]` is a
   real class and part of ADR 0028 (PR A), with its parameter bounded on a
   declared [Cartesian axis](#cartesian-axis-dimensions).

   Rules 3–6 in full — the hook, the fingerprint deconstructors, the exhaustive
   mangling test and the `_Staggered` history — are in [[personal/egparedes/connectivities-as-types/connectivities-as-types_identity|the identity appendix]].
7. **The IR names a connectivity by its `offset_tag`.** That is its local
   dimension's tag when it declares the local dimension: shifts, reductions
   and sparse arguments then all find the table under **one** string, so the
   backends' A3/A4 lookups are correct without touching them. A connectivity
   *sharing* another one's local dimension is named by its own tag, since the
   local tag already names the owner's table; for that case the backends have
   a fallback lookup, `connectivity_key_over` (see *The local dimension*).
8. **Same-named declarations are different connectivities.** Two
   declarations with the same qualified name from different modules, or a
   redefinition, are simply different classes; no warning is issued, and
   `check_neighbor_table` hints "was the declaration redefined?" when a table
   built for one is bound to the other.

### Cartesian axis dimensions

A **Cartesian axis** is an index space with integer index arithmetic and
exactly one staggered partner — the dimensions `CartesianConnectivity` acts
on. One axis of a Cartesian grid is a 1-dimensional CW complex and has exactly
two cell classes; a declared axis and its `Staggered[·]` name those two, and
`Staggered` is the involution that swaps them. The types do not fix which class
has degree 0. In more than one dimension the grid is the product complex of its
axes, so a cell class is one bit per axis and `Dims[...]` *is* that bit vector —
finer than a form degree, and the reason mesh locations are separate classes
instead: a per-axis bit exists exactly when the complex factors as a product.

Two facts shape the hierarchy. A staggered dimension **is itself** a Cartesian
axis — the other cell class of the same complex — so the asymmetry between `D`
and `Staggered[D]` is *declarational*, not semantic: one is written by the user,
the other derived, and there is exactly one partner to derive. And `Staggered[D]`
needs no orientation data, because a product of intervals is canonically
oriented per axis.

That derivation, the alignment convention and its antisymmetry, why extents are
declared rather than derived, and the per-declaration axis-versus-mesh-location
decision are in [[personal/egparedes/connectivities-as-types/connectivities-as-types_staggering|the staggering appendix]]; its external
corroboration is in [[personal/egparedes/connectivities-as-types/connectivities-as-types_conventions|the conventions appendix]].

Hence three levels, all *below* `DimensionIndex`, so that no
`type[DimensionIndex]` annotation in the tree changes and `LocalDimensionIndex`
keeps its position:

```
DimensionIndex                            # the root; the tree's annotations say this
├── AnyCartesianAxisIndex                 # a cell class of a Cartesian axis
│   ├── CartesianAxisIndex                # ← declared; what users subclass
│   └── Staggered[D: CartesianAxisIndex]  # derived
├── LocalDimensionIndex                   # a neighbor index; not an axis
└── (direct subclasses)                   # mesh locations V/E/C, and index spaces
                                          #   with no geometry (tracer, ensemble)
```

Four checks the tree performs at runtime, or not at all, become static:

| Rejected | Today | With `CartesianAxisIndex` |
| --- | --- | --- |
| `Staggered[Staggered[K]]` | runtime `TypeError` | static, at the bound |
| `Staggered[V2E.Local]` | runtime `TypeError` | static, at the bound |
| `Staggered[C]` — no half-cells on an unstructured mesh | nothing rejects it | static, at the bound |
| `C + 1`, `V2E.Local + 1` — index arithmetic off an axis | runtime `kind` check, or a late failure | static, via `DimensionMeta.__add__`'s self-type |

The last row costs two definition-site suppressions, one per checker. P4 of
[`staggered_probe.py`](staggered_probe.py) pins it down and
[[personal/egparedes/connectivities-as-types/connectivities-as-types_typing|the typing appendix]] records both diagnostics.

`kind` **shrinks.** It carries four jobs today: `LOCAL` marks a neighbor axis;
`order_dimensions` sorts by `(kind, as_non_staggered(dim).value)`, so it decides
memory layout order (constraint F4); `order_dimensions` also uses it to enforce
at most one `LOCAL` dimension; and `VERTICAL` names the scan axis and, by
convention, the undecomposed one. The first and third are now class facts —
`kind == LOCAL` iff `issubclass(d, LocalDimensionIndex)` — so **`LOCAL` leaves the
enum**: `kind` becomes `HORIZONTAL | VERTICAL` and `order_dimensions` sorts by
`(is_local, kind, base.tag)` with `is_local` read from the class (PR B, where
`LocalDimensionIndex` lands). What is left is the layout key and the "vertical"
role, which belong to the field and to the program respectively — see
[Open questions](#open-questions--follow-ups).


### `Staggered[D]`

A staggered dimension sits at the **half-integer positions** of a base
dimension (ADR 0026): `Staggered[K](0)` is the point half a cell *below*
`K(0)`. Cell- and edge-located fields on a Cartesian grid then carry
different dimensions and cannot be combined by accident. The semantics of
ADR 0026 — the position convention, the shift rule and the backend treatment
— are unchanged; what changes is the *encoding*, from a reserved `_Staggered`
prefix on the dimension name to a class, whose parameter is the declared axis
of [Cartesian axis dimensions](#cartesian-axis-dimensions).

- **Runtime representation.** `Staggered[K]` is a real, interned class:
  `Staggered` has a metaclass whose `__getitem__` builds one class per base
  and caches it by base identity (with `setdefault`, so concurrent
  compilation threads see one class). Its base is `AnyCartesianAxisIndex`, and it is
  deliberately *not* a subclass of `K`, so a `Field[Dims[Staggered[K]]]` is
  rejected where a `Field[Dims[K]]` is required; it has `base = K` and `K`'s
  kind. Staggering anything that is not a declared `CartesianAxisIndex` — a local
  dimension, a mesh location, an already-staggered class — and subclassing a
  staggered class directly are `TypeError`s. A PEP 695 generic cannot serve
  here — it would be a `typing` alias, failing `issubclass` and eve's
  `type[...]` validation — but under `TYPE_CHECKING` `Staggered` *is* declared
  as one, so mypy and pyright accept `Staggered[K]` in annotations and in
  `Field[Dims[Staggered[K]], float]`. `is_staggered`, `flip_staggered` and
  `as_non_staggered` keep their meaning and read `base`.
- **Doubly staggered dimensions are unrepresentable, statically.** The
  `TYPE_CHECKING` declaration is `class Staggered[D: CartesianAxisIndex](AnyCartesianAxisIndex)`,
  and `Staggered[K]` is an `AnyCartesianAxisIndex` but not a `CartesianAxisIndex`, so
  `Staggered[Staggered[K]]` is a `[type-var]` error for both checkers rather
  than only a runtime `TypeError` — as are `Staggered[V2E.Local]` and
  `Staggered[C]`. Because the extra levels sit below `DimensionIndex`,
  `Staggered[K]` stays a `DimensionIndex` and no annotation widens; verified in
  [`staggered_probe.py`](staggered_probe.py). Nothing stronger is claimed. The
  equation `Staggered[Staggered[K]] = K` is semantically right — `Staggered` is
  an involution on the two cell classes — but Python typing has no type-level
  reduction, so `Field[Dims[K]]` and a hypothetical
  `Field[Dims[Staggered[Staggered[K]]]]` would be mutually incompatible nominal
  types. Making the nested form unrepresentable sidesteps the equation instead
  of asserting it.
- **Embedded.** `K + 0.5` builds, through `connectivity_for_cartesian_shift`,
  a `CartesianConnectivity` whose codomain is `flip_staggered(K)`, with the
  `+1` index correction the position convention requires when shifting *out
  of* an unstaggered dimension: `K + 0.5` maps `K(i)` to `Staggered[K](i+1)`
  (position `i+½`), `Staggered[K] + 0.5` maps `Staggered[K](i)` to `K(i)`.
  The frontend accepts only integer and half-integer literal offsets and
  types the shift as an `OffsetType` at type-deduction time; statically,
  `K + 0.5` is a plain `Connectivity`. `check_dims` rejects a dimension and
  its staggered counterpart in one field or domain, and `order_dimensions`
  orders by the base, so a staggered field has the layout of its base.
- **IR, serialization and fingerprints.** The tag is
  `gt4py.next.common.Staggered[<base tag>]` — the one tag that is not a
  qualified name. Its grammar, the `copyreg` hook and fingerprinting through
  the base are in [[personal/egparedes/connectivities-as-types/connectivities-as-types_identity|the identity appendix]].
- **Backends.** gtfn emits the staggered tag as a C++ alias of its base tag
  (`_add_staggered_aliases`), since its shift primitive can offset along an
  axis but not rename it. DaCe names map variables by the base
  (`as_non_staggered` in `get_map_variable`), so map fusion sees one
  iteration space for a field and its staggered counterpart.

[[personal/havogt/dimension-generic-fields/dimension-generic-fields|Generic dimensions and statically typed staggering]]
Part II proposed this `Staggered[D]` shape and the root/user split that makes the
doubly staggered type unrepresentable. What differs here, and the constraint
reserving `dual` for the full Hodge dual, are in [[personal/egparedes/connectivities-as-types/connectivities-as-types_staggering|the staggering appendix]].

### Binding model

The connectivity *type* conceptually depends on the *table*: two different
tables assign different neighbors to the same domain element. The model:

- The class `V2E` is a **declaration**. Its static part — `V2E.domain`,
  `V2E.codomain`, `V2E.Local`, and *optionally* the counts
  `V2E.Local.max_neighbors` / `V2E.Local.min_neighbors` (they are stored on
  the local dimension; `V2E.max_neighbors` does not exist) — is what
  compilation sees. Declared counts are a constraint on the table; undeclared
  ones are simply taken from the bound table's `NeighborTableType` when
  a variant is compiled. Binding never writes anything back to the class.
- Why the counts are not static-only. Two in-tree and in-ICON facts: the
  arity of the same connectivity varies per mesh (`fvm_nabla_setup.py` sizes
  `V2E` from the atlas mesh), and skip-value presence is
  *configuration*-dependent in ICON (`icon.py:130`: pentagon offsets have skip
  values on the icosahedron but not on the torus), so a static `min_neighbors`
  would force duplicate classes or always-on skip handling.
- **One table per declaration per binding context.** A binding context is
  one program or field-operator call, one `compile`, or one embedded
  context. Within it, the provider `{V2E: table}` binds exactly one table to
  `V2E` — the dict makes this structural, and `check_offset_provider`
  verifies it (a tag bound twice, through a class key and a string key, is an
  error). Across contexts the same declaration may be bound to different
  tables: tests bind several meshes per process, ICON runs the same
  declarations on the torus and the icosahedron, and this is what lets one
  compiled program serve many meshes. The one relaxation is at the
  local-dimension level: connectivities that *share* a local dimension bind
  several tables over one local axis in one context, subject to the
  consistency checks below.
- **Normalized at the entry points.** `as_tag_keyed_offset_provider`
  rewrites the provider to the form the IR uses, keyed by `offset_tag`. It is
  called *strictly* by `Program.__call__`, `FieldOperator.__call__`,
  `FieldOperatorFromFoast.__call__`, `compile` and
  `CompilationOptions.connectivities`, and *non-strictly* by
  `embedded.context.update`, the iterator `fendef` and DaCe
  `get_sdfg_conn_args`.
  Strict means that a key which is neither a declaration nor a dotted tag
  string — e.g. `"V2E"`, the removed `FieldOffset` spelling — raises
  `TypeError`; the iterator-level hooks accept any string, since hand-written
  IR names offsets freely. A dotted string is accepted as an
  already-normalized tag everywhere, and class and string keys can be mixed.
  Lowering, the backends and the compiled-program cache keep seeing
  tag-keyed providers — like hand-written IR — so the backends' accesses *by
  connectivity tag* are unchanged; those keyed by a *local dimension*
  (reductions, sparse and list arguments) find the table through
  `connectivity_key_over`.
- **Validated at the boundary.** `check_offset_provider` runs at *every* entry
  point, including the iterator `fendef`, `FieldOperatorFromFoast` and the DaCe
  orchestration's `CompilationOptions.connectivities`. Its result is remembered
  per set of bound tables (hashed by their `id`, as the compiled-program cache
  keys its variants), so a repeated call with the same tables costs one hash;
  like that cache, the memo can in principle skip a check when a freed table is
  replaced at the same address. Reading the tables — comparing skip positions,
  below — happens only where a program is compiled, not on the call path. It
  calls `check_neighbor_table` per declaration, which checks the domain
  `(Domain, Local)` and the codomain by class identity (with a hint when a
  declaration looks redefined), an integral dtype, `max_neighbors` equal to
  the table's width and `min_neighbors` not above it if declared, and — only
  when `min_neighbors` is declared — that the table has a skip value iff
  `min_neighbors <` its width. Skip values are judged on the table's *type*:
  the number of valid neighbors per row is never counted. `check_offset_provider`
  additionally requires all tables over one local dimension to agree on width
  and skip-value presence and — where a program is compiled — on the skip
  positions of the concrete tables, compared on the device the tables live on.
  A key that names the *connectivity* instead of the declaration (`{V2E.tag:
  table}`, which is the IR name of a declaration only when it shares another's
  local dimension) is rejected with a pointer to `{V2E: table}`. A string key that is
  not a qualified name at all is remembered as such, so it costs one import
  attempt in total rather than one per call.
- The class never holds data; the table crosses process boundaries as it did.

### The local dimension

- `V2E.Local` **must be given explicitly** in the class body — a nested
  `class Local(LocalDimensionIndex): ...`, or an existing local dimension to
  adopt or share (below); `__init_subclass__` verifies it (a missing `Local`
  is a `TypeError`) and sets `Local.owner`. There is no generated form. This
  is settled by running both checkers on
  [`typing_probe.py`](typing_probe.py): an explicitly
  declared nested `Local` is a real, distinct type under mypy `--strict` and
  pyright (`Field[V, V_E2E.Local]` vs `Field[V, E_V2V.Local]`, differing only
  in the local dimension, is an `arg-type` error, probe P1b), while a `Local`
  generated in `__init_subclass__` and
  annotated `ClassVar[type[LocalDimensionIndex]]` on the base is rejected by
  both as *not valid as a type* — the very error class ADR 0028 exists to
  remove. Two lines per connectivity buy a real type; ICON4Py has 16
  declarations. Its tag is `<owner tag>.Local`, unique by construction.
- `Local` is **not** passed through the base subscription
  (`NeighborConnectivity[V, "V2E.Local", E]`): mypy accepts that string
  forward reference, pyright reports `Class definition for "V2E" depends on
  itself`. `__init_subclass__` reads it from the class body instead.
- **`Local` is annotated nowhere**, and an adopted or shared `Local` must be a
  `TypeAlias`. Both are forced by what mypy and pyright accept in an annotation
  position; [[personal/egparedes/connectivities-as-types/connectivities-as-types_typing|the typing appendix]] has the diagnostics, and
  [`typing_probe.py`](typing_probe.py) the probes.
- `max_neighbors` / `min_neighbors` are class keywords, like `kind` on a
  dimension, and optional (see Binding model). They are *not* type
  parameters: Python has no integer-valued type parameters, and nothing
  static needs the count — what must be in the type is *which* local
  dimension.
- **Owner-less local dimensions** exist: `class LsqUnk(LocalDimensionIndex, size=3)`
  has `owner = None`, `min == max == size`, no skip values, and never appears
  in the provider — unless a connectivity later adopts it, which makes it
  owned. `size` is optional; an owner-less local without it has no counts. ICON4Py needs this today: `LsqUnkDim` (`dimension.py:18`,
  LOCAL kind, "number of least-squares unknowns") is the middle axis of a
  coefficient field over `(CellDim, LsqUnkDim, C2E2CDim)` and indexes no
  table; `RBFDimension` is an enum of stencil sizes that plays the same role.
  Such axes want the *storage and layout* treatment of a sparse dimension,
  not a connectivity; declaring them non-LOCAL instead would make them domain
  dimensions requiring a range in every program domain and would change
  ICON4Py's memory layout for those fields. (Backend support for sparse fields
  over an owner-less axis *other than* `ConstList` is not part of the stack:
  such a field still needs some bound table over its axis.)
- `_CONST_DIM` is replaced by the owner-less
  `common.ConstList(LocalDimensionIndex, size=1)`, one class used by iterator
  embedded and the DaCe lowering with identity checks, and refused as an
  adopted `Local`. (Type inference still
  represents a constant list as `ListType(offset_type=None)`.)
- `neighbor_sum(..., axis=V2E.Local)` — the local axis stays **explicit**.
  `axis=V2E` as sugar is deliberately not offered; it would blur the two
  concepts the design separates. Passing it is a type error in DSL code.

### `MultiDimensionIndex`

- `MultiDimensionIndex(Vertex(3), V2E.Local(2))` is a position in the product
  of a primary and local dimensions. It is a `tuple` subclass, so it indexes a
  field or a table directly (`v2e_table[position]`, a 0-d field), and compares
  and hashes like the plain tuple; `pickle` and `copy` keep it a
  `MultiDimensionIndex`, tuple operations such as slicing do not. It is a
  user-facing typed index; nothing in the toolchain requires it.
- `*Ls` is unconstrained because `TypeVarTuple` cannot carry a bound, so the
  constructor checks the shape at runtime: one non-local index, then *at least
  one* local index, and an owned local dimension must index the neighbors of the
  primary index's dimension — `MultiDimensionIndex(Edge(1), V2E.Local(1))` is a
  `TypeError`, since `V2E` indexes the neighbors of a vertex. An owner-less
  local axis (`LsqUnk`) is accepted next to any primary dimension.
- It is a position in a *table*, not in a declaration: a declaration is not a
  `Connectivity` and cannot be indexed.
- The iterator-level embedded execution keys its positions by dimension
  classes instead of tag strings and steps along a local dimension with an
  explicit `SparseAxis(dim)`; `SparseTag` is gone.

### Effect per layer

| Layer | Before | After |
| --- | --- | --- |
| Frontend declaration | `Dimension(LOCAL)` + `FieldOffset` + provider key | one class |
| Frontend types | `ts.OffsetType(source, target)`, tag dropped | `ts.ShiftType(domain, codomain, tag)` produced by the class (`__gt_type__`), `tag` set to `offset_tag`; `V2E.Local` in DSL code types as the local dimension |
| FOAST → GTIR | shift tag = **Python variable name** | tag = `offset_tag`; the variable name is irrelevant (GridTools/gt4py#2898) |
| ITIR | `OffsetLiteral(str)` + dict lookup; `AxisLiteral(value, kind)` | `OffsetLiteral(offset_tag)`, unchanged node; `AxisLiteral(value)` with `kind` derived |
| ITIR types | `ListType.offset_type: Dimension` | unchanged (already the local dim) |
| Embedded field | `get_offset(provider, axis.value)` | table over the local dimension (`connectivity_key_over`); `V2E.bound_table()` |
| Embedded iterator | positions keyed by name strings; `SparseTag` | positions keyed by dimension classes; `SparseAxis(dim)` |
| gtfn | `dim.value` lookups; `TagDefinition` per tag and per `neighbor_dim` | still a `TagDefinition` per offset tag (and per differing `neighbor_dim`, i.e. for sharers), named by `codegen_name(tag)`; sparse-argument lookups through `connectivity_key_over` |
| DaCe | ~10 `offset_type.value` lookups; `Dimension(offset, LOCAL)` synthesized | `codegen_name` for array/symbol names; `connectivity_key_over`; a local dimension sized from its own table (`local_dimension_size`); no synthesis |
| roundtrip backend | emits `gtx.Dimension("<name>")` / `offset("<tag>")` as source text | emits `<mangled> = gtx.resolve("<tag>")` for axes and `<mangled> = offset("<tag>")` for offsets |
| Provider | `Mapping[str, NeighborTable]`; `offset_provider_type: Mapping[str, NeighborConnectivityType]` | public: `{V2E: table}`, and `table_types={V2E: NeighborTableType(...)}` (`TableTypesLike`, normalized to `TableTypes = Mapping[Tag, NeighborTableType]`) for ahead-of-time compilation (a dotted tag string is also accepted; the type stays `Mapping[Any, …]`); internal: keyed by `offset_tag`, normalized at the entry points; no deprecation window (a migration script for ICON4Py instead) |

### What it deletes

`FieldOffset` (both forms) and its export; bare-name provider keys (rejected
at the strict entry points); the `_Staggered` prefix and its string sniffing
(replaced by the `Staggered[D]` class and a small `<owner>[<base>]` tag
grammar); `_CONST_DIM` as a magic name; the stored `AxisLiteral.kind`;
`SparseTag`; `NamedIndex` and the dimension half of the mypy plugin;
`DimensionIndex(kind=LOCAL)` as a way to declare a local dimension, and the
`LOCAL` member of `DimensionKind` itself (derived from `LocalDimensionIndex`); and the
convention that `V2EDim = Dimension("V2E", LOCAL)` must sit next to
`V2E = FieldOffset("V2E", ...)`; the names `ts.OffsetType` (now
`ts.ShiftType`, with `domain`/`codomain` instead of the inverted
`source`/`target`), `NeighborConnectivityType` (now `NeighborTableType`) and
the parameter `offset_provider_type` (now `table_types`).

**Kept** (the concepts, not always the names — `ts.ShiftType` is the renamed and
re-fielded `ts.OffsetType`): the frontend shift type and the runtime
connectivity objects (see
[What stays, and why](#what-stays-and-why)); `iterator.runtime.offset("...")`
(the iterator-level API names offsets by string, like the IR); and
string-keyed providers *below* the entry points.

The Cartesian `FieldOffset` goes together with the unstructured one:
Cartesian shifts were already `Dim + i`, and `as_offset` takes the
dimension, `as_offset(KDim, field)`.


## Alternatives considered

Eleven alternatives are recorded in [[personal/egparedes/connectivities-as-types/connectivities-as-types_alternatives|the alternatives appendix]],
and the six concerning the axis levels and `Staggered[D]` in
[[personal/egparedes/connectivities-as-types/connectivities-as-types_staggering|the staggering appendix]]. The three a reviewer is most likely
to raise: keep every concept and fix only which string wins; `(name, kind)` value
equality with an interning registry; and static-only `max_neighbors` /
`min_neighbors`.

## Relation to gt4py ADRs

- **ADR 0019 (Connectivities).** The part naming `FieldOffset` as the frontend
  identifier is superseded by ADR 0029; the `Connectivity` / `NeighborTable` /
  `ConnectivityType` vocabulary is kept ([What stays](#what-stays-and-why)).
- **ADR 0026 (Staggered dimensions).** Its encoding is superseded by
  `Staggered[D]` ([above](#staggeredd)); its semantics are unchanged, and
  [Cartesian axis dimensions](#cartesian-axis-dimensions) states the structure they encode.
- **ADR 0028 (Dimensions as nominal types)** and **ADR 0029 (Connectivities
  as types)** record this design in the gt4py tree.
- **ADR 0023 (build cache).** Fingerprints depend on qualified names, so the
  cache invalidates on module renames (Identity, rule 4).

## Open questions / follow-ups

1. **Naming.** `NeighborConnectivity[Domain, Codomain]`, `Local`,
   `max_neighbors` / `min_neighbors`, and `offset_tag` for the IR name.
   Convergence with
   [[personal/havogt/dependent-local-dimensions/dependent-local-dimensions|dependent local dimensions]]
   (`Dim`, `has_skip_values`, `source_dim`/`neighbor_dim`) is still to be
   discussed on the PRs; the names become public API in PR B.
2. **The mesh proposal.**
   [[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos|A mesh concept with
   first-class halos]] keeps `Koff`/`LsqUnkDim` outside the Mesh object and
   replaces the same `offset_provider` dict this note re-types. The two are
   compatible if the Mesh binds typed connectivity classes to tables, but
   that has to be said in one of the two documents, and the multi-table case
   (halo variants, rewritten `keep_skip_values` tables, several meshes in one
   process) is addressed by neither yet.
3. **Closure variable resolution.** The N2 leak (Python variable name
   becoming the IR tag) is fixed in `foast_to_gtir` (GridTools/gt4py#2898); the resolution
   mechanism of
   [[personal/havogt/closure-variable-resolution|Closure variable resolution]]
   would be its natural home.
4. **Chain proposals.** Their static encodings need rewriting to `C.Local`
   ([[personal/egparedes/connectivities-as-types/connectivities-as-types_typing#3-relation-to-localv2e|typing appendix]] §3).
5. **Parameterizing the alignment.** A per-axis `stagger=` class keyword would
   remove the one cosmetic cost of choosing which member of a pair to declare. It
   has to be static — it changes the integer correction
   `connectivity_for_cartesian_shift` bakes in, hence the emitted stencil — and it
   hits the rule-4 fingerprint hole. See [[personal/egparedes/connectivities-as-types/connectivities-as-types_staggering|the staggering appendix]].
   Not in this stack.
6. **`kind`'s remaining two jobs.** Once `LOCAL` leaves the enum, `kind` is a
   layout sort key and a name for the scan axis — properties of the field and of
   the program, not of the dimension. Every candidate *addition* belongs to the
   grid, the program or the range, so `kind` should shrink rather than grow; see
   [[personal/egparedes/connectivities-as-types/connectivities-as-types_staggering|the staggering appendix]] §7.
7. **A cell `degree` on the axis.** Left out: it duplicates what the absolute
   ranges already say, needs periodicity (which cannot be static), and has no
   consumer until an exterior-calculus surface exists. A degree is *canonical* for
   a mesh location and only *declarational* for an axis, so it belongs on the
   `LocationIndex` of the next item; see [[personal/egparedes/connectivities-as-types/connectivities-as-types_staggering|the staggering appendix]]
   and [[personal/egparedes/connectivities-as-types/connectivities-as-types_conventions|conventions]] §4a.
8. **A `LocationIndex` level for mesh locations.** `V`, `E`, `C` stay direct
   `DimensionIndex` subclasses, so "a primary, non-local dimension" is still not a
   type. UGRID supplies the vocabulary and the degrees
   ([[personal/egparedes/connectivities-as-types/connectivities-as-types_conventions|conventions appendix]] §4a). Out of scope here.
9. **`DimensionMeta.__eq__` versus "equality is `is`".** [Identity](#identity-is-the-qualified-python-name)
   states that the runtime view of identity is `is`, while the sketch keeps
   `__eq__` overridden so that `I == 5` builds a `Domain` (with
   `__hash__ = type.__hash__`). A metaclass `__eq__` that does not return a bool
   makes `I == J` not a comparison, and dict lookups on a hash collision would
   call `bool(Domain)`. Class-keyed providers (`{V2E: table}`) and the `is`-based
   identity story both depend on this being benign; it is not established here,
   and the ADR 0028 implementation should say which of the two rules wins.
10. **The index origin of a bound table.** `NeighborTableType` records no index
    origin, while UGRID standardizes `start_index` ∈ {0, 1}. `check_neighbor_table`
    could validate entries against the codomain's absolute range instead of
    `[0, size)` ([[personal/egparedes/connectivities-as-types/connectivities-as-types_conventions|conventions appendix]] §4b).

## Implementation

GridTools/gt4py#2898 (merged 2026-09-23) fixed the N2 leak first: unstructured
shifts lower with the offset's own tag, shift lowering is type-driven so
module-qualified offsets (`a(mod.V2E)`) work, and a regression matrix
{shift, `neighbor_sum`} × {tag ≠ variable name, tag ≠ local-dimension name}
guards it. The rest is a stack of four GridTools/gt4py PRs, each green in CI
on its own, each based on the previous one:

| PR | Branch | What |
| --- | --- | --- |
| A: GridTools/gt4py#2899 | `connectivities-as-types-2-dimension-classes` | `feat[next]`: a concrete dimension is a class identified by its qualified name (ADR 0028); `codegen_name`; `CartesianAxisIndex` / `AnyCartesianAxisIndex` and `Staggered[D: CartesianAxisIndex]`; interactive-`__main__` fallback; `NamedIndex` and the dimension half of the mypy plugin removed; `ConstList` with `size=1` replaces the `_CONST_DIM` aliases; `AxisLiteral` drops `kind`; printing IR never imports (`resolve_loaded`); offset tag = local dimension tag = provider key in the tree |
| B: GridTools/gt4py#2907 | `connectivities-as-types-3-neighbor-connectivity` | `feat[next]`: `NeighborConnectivity[Domain, Codomain]`, `LocalDimensionIndex` (and `DimensionKind.LOCAL` removed, derived from the class), `check_neighbor_table`, `local_dimension_of`, declaration fingerprinting; usable in the DSL; shared local dimensions, in the declarations and in the backends (`connectivity_key_over`, DaCe `local_dimension_size`); `NeighborTableType` and `ts.ShiftType`; pyright in the typing nox session; ADR 0029 |
| C: GridTools/gt4py#2910 | `connectivities-as-types-6-class-keyed-providers` | `feat[next]!`: the tree and docs declare connectivities as classes; class-keyed offset providers with `check_offset_provider` at every entry point; `FieldOffset` removed; `as_offset(dim, field)`; `table_types` replaces `offset_provider_type`; migration script |
| D: GridTools/gt4py#2912 | `connectivities-as-types-8-typed-positions` | `refactor[next]`: `MultiDimensionIndex` and typed embedded positions |

The four PRs form GitHub stack GridTools/gt4py#2917. `NeighborTableType`,
`TableTypes` and `ts.ShiftType` land in B, `table_types` in C; ADR 0028 and
ADR 0029 are in A and B respectively.

**ICON4Py migration.** `scripts/python/migrate_connectivities.py` (PR C)
rewrites `Dimension(...)` and `FieldOffset(...)` declarations (connectivities
adopt their existing local dimensions, so `C2EDim` etc. keep working and
`C2CE` becomes a sharer), removes Cartesian offsets and rewrites their uses
(`Koff[1]` → `KDim + 1`, `as_offset(KDim, ...)`, imports), and reports what
it cannot decide from the source. At ICON4Py `89b4967` (2026-09-21) the
script rewrites 4 files (`dimension.py` and 3 stencil modules) and reports 46
connectivity provider keys, 2 removable Cartesian provider entries and 10
`isinstance(..., Dimension)` sites in 5 modules. Declaring counts is left to
ICON4Py.

## Appendices

- [[personal/egparedes/connectivities-as-types/connectivities-as-types_staggering|Cartesian axes, the CW-complex reading and alignment]]:
  why an axis has exactly two cell classes and `Dims[...]` is the product complex's
  bit vector; the alignment convention and its antisymmetry; why extents are
  declared; the six axis-level alternatives and the staggering follow-ups.
- [[personal/egparedes/connectivities-as-types/connectivities-as-types_identity|What type identity costs the IR, pickle and codegen]]:
  the mechanics behind Identity rules 1 and 3–6 — reconstruction by import, the one
  `copyreg` hook, fingerprinting, the injective codegen mangling and the
  `Staggered[D]` tag grammar.
- [[personal/egparedes/connectivities-as-types/connectivities-as-types_typing|Decisions made by running mypy and pyright]]:
  why `Local` is annotated nowhere and must be a `TypeAlias` when shared, what the
  `__add__` self-type costs, and why `Local[C]` and `V2E.Local` cannot be
  reconciled. The two probes are its attachments.
- [[personal/egparedes/connectivities-as-types/connectivities-as-types_alternatives|Alternatives considered]]:
  the alternatives rejected on grounds other than the axis levels.
- [[personal/egparedes/connectivities-as-types/connectivities-as-types_research|Tag and name constraints — full catalogue]]:
  the five name spaces; the 10 cross-object identity constraints (A1–A10),
  9 name-format constraints (F1–F9) and 7 structural constraints (S1–S7),
  each with `file:line`; per-context tables for embedded, IR, gtfn and DaCe;
  the complete concept inventory (L0–L9); class-hierarchy and name-flow
  diagrams; the two executed experiments.
- [[personal/egparedes/connectivities-as-types/connectivities-as-types_dependent-typing|Background: dependent typing for mesh connectivities]]:
  a primer on value-dependent types (Π and Σ types, refinements, evidence,
  local slots vs. incidences) on a small mesh, and a mapping of each concept
  onto this design — what is kept statically, what moves to the bind-time
  check, and what is given up.
- [`typing_probe.py`](typing_probe.py): the mypy /
  pyright probes behind the `Local` decisions (explicit nested class works,
  also when only the local dimension differs; generated `ClassVar` form and
  `Local[C]` reconciliation do not). Last run with mypy 2.3.1 and pyright
  1.1.414.
- [[personal/egparedes/connectivities-as-types/connectivities-as-types_conventions|Alignment with the UGRID and SGRID conventions]]:
  an audit against the two community conventions for mesh topology and grid
  staggering — why UGRID's locations are cell degrees while SGRID's are per-axis
  bit vectors, how SGRID's four `padding` values decode into single absolute
  ranges, and the enhancements that follow (a UGRID vocabulary for
  `LocationIndex`, validating a bound table against its codomain's range,
  deriving incidence signs from a recorded node ordering).
- [`staggered_probe.py`](staggered_probe.py): the mypy / pyright probes behind
  [Cartesian axis dimensions](#cartesian-axis-dimensions) — `Staggered[Staggered[I]]`,
  `Staggered[V2E.Local]` and `Staggered[C]` are `[type-var]` errors, while
  `Staggered[I]` stays usable where `type[DimensionIndex]` is required and in
  `Dims[...]`, so no annotation in the tree widens.
