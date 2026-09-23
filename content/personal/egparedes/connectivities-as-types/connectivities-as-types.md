---
title: "Connectivities as types: one declaration for offset, local dimension and provider key"
author: egparedes
tags: [type-system, type-checking, dimensions, local-dimensions, connectivities, offset-provider, unstructured, neighbor-sum, reduction, frontend, foast, gtir, embedded, gtfn, dace, nominal-types, dependent-types, metaclass, serialization, fingerprint, staggering, migration, adr, tech-debt, shared-local-dimensions, implemented]
created: 2026-09-17
updated: 2026-09-23
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
> identity constraints in the current tree, five (A1–A5) dissolve by
> construction and three (A6–A8) collapse into a single bind-time check.

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
> the typed substrate that proposal can bind through.

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

Two of these were confirmed by execution against v1.2.2:

```text
# MyOff = FieldOffset("TAGNAME", ...)              -- tag != variable name
embedded:  offset_provider={"TAGNAME": conn} -> OK ; {"MyOff": conn}   -> KeyError 'TAGNAME'
roundtrip: offset_provider={"MyOff": conn}   -> OK ; {"TAGNAME": conn} -> KeyError 'MyOff'

# Off = FieldOffset("Off", source=E, target=(V, Neigh))   -- tag != local dim name
neighbor_sum(a(Off), axis=Neigh), offset_provider={"Off": conn}
embedded:  KeyError: "Offset 'Neigh' not found in offset provider."
```

The first shows embedded and compiled execution key the *same* program on
*different* strings. The second shows that GridTools/gt4py#1789 ("gtfn with
offset name != local dimension name"), which lifted the N1 = N3 requirement,
only did so for the *shift* path under gtfn; its regression test
(`tests/next_tests/regression_tests/ffront_tests/test_offset_dimensions_names.py`)
covers exactly `a(Off[1])` on `GTFN_CPU` and nothing else. The embedded
reduction still requires it (run); the roundtrip backend *passes* the same
program, and the compiled gtfn pipeline is expected to fail through
`unroll_reduce.py:47` — established by reading, not run. Sparse field
arguments require it in both gtfn and DaCe (by reading).

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
runtime connectivity classes, 4 provider aliases, 2 declaration classes) for
what is one idea — *a mapping between two index spaces, plus a name for it*.

## Proposal

### Concepts

| Concept | Role | Replaces |
| --- | --- | --- |
| `DimensionIndex` | a dimension is a subclass, an index along it an instance (ADR 0028) | `Dimension` instances |
| `Staggered[D]` | the dimension at the half-integer positions of `D`; a real, interned class (see [`Staggered[D]`](#staggeredd)) | the `_Staggered` name prefix |
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
(`NeighborConnectivity[Domain, Codomain]`, `Staggered[D]`,
`MultiDimensionIndex`) check that it is not local.

### What stays, and why

The runtime and compile-time connectivity objects of ADR 0019 stay. They are
*produced by* or *checked against* the declaration instead of being authored
in parallel with it:

| Artifact | What it is | Role after this proposal |
| --- | --- | --- |
| `Connectivity` | the runtime protocol: a `Field` of integer indices with a `codomain`, which `field(conn)` / `Field.premap` dispatch on | unchanged. It unifies neighbor tables, Cartesian shifts and `as_offset` index fields, so embedded execution has one remap entry point. Essential. |
| `CartesianConnectivity` | the `Connectivity` behind `I + 1` and `I + 0.5`; carries the ADR 0026 index arithmetic | unchanged. The Cartesian counterpart of a bound table: it needs no declaration and no provider. Essential. |
| `NeighborTable` / `NdArrayConnectivityField` | the data: a 2-D table over `(Domain, Local)` with values in `Codomain` | unchanged; `check_neighbor_table` checks it against the declaration. Essential. |
| `NeighborConnectivityType` | the *table's* type, `table.__gt_type__()`: `domain`, `codomain`, `dtype`, `skip_value`, `max_neighbors`; produced by the table, never authored | what compilation consumes instead of data (ADR 0019), what `compile(offset_provider_type=...)` takes when no table exists yet, and what fingerprints a compiled variant. Once the bind-time check has passed, its `domain`/`codomain` duplicate the declaration; only `dtype`, `skip_value` and `max_neighbors` are information a declaration may lack. Necessary in role; its shape could be slimmed to (declaration, dtype, skip value, width) in a follow-up over its 13 consumer modules. |
| `ts.OffsetType` | the frontend type of a shift *argument* in DSL code — what `V2E`, `V2E[1]`, `I + 1`, `I + 0.5` and `as_offset(K, f)` type as during FOAST type deduction, and what the lowering reads to emit `OffsetLiteral` / `CartesianOffset` | produced by `V2E.__gt_type__()`, with a `tag` field carrying `offset_tag`. Some type for these expressions is unavoidable. Its `source`/`target` fields are the inverted vocabulary of the Problem section and show in error messages; renaming them to `codomain`/`domain` is a follow-up. |

### Sketch

```python
# ── dimensions (ADR 0028) ─────────────────────────────────────────────────
class DimensionMeta(type):
    @property
    def tag(cls) -> Tag:                       # identity, and the IR spelling
        return f"{cls.__module__}.{cls.__qualname__}"
    kind: DimensionKind
    __hash__ = type.__hash__                   # __eq__ stays for `I == 5` -> Domain
    def __add__(cls, offset: int | float) -> Connectivity: ...   # I + 1, K + 0.5


class DimensionIndex(metaclass=DimensionMeta):
    value: int                                 # an index along the dimension

type Dimension = type[DimensionIndex]          # PEP 695: `Dimension("I")` is not callable


class LocalDimensionIndex(DimensionIndex):     # kind is LOCAL; `kind=LOCAL` elsewhere is an error
    owner: ClassVar[type[NeighborConnectivity] | None]   # None: owner-less; set by the owner
    max_neighbors: ClassVar[int | None]        # the counts live here, not on the connectivity
    min_neighbors: ClassVar[int | None]        # min == max  <=>  no skip values
    def __init_subclass__(cls, *, size: int | None = None): ...   # size: max = min = size


class MultiDimensionIndex[D: DimensionIndex, *Ls](tuple[D, *Ls]): ...   # shape checked at runtime

Staggered[KDim]                                # an interned, real dimension class (ADR 0026)


# ── connectivities (ADR 0029) ─────────────────────────────────────────────
class ConnectivityMeta(type):
    tag: Tag                                   # the declaration's qualified name
    offset_tag: Tag                            # how the IR names it, see "Identity"
    domain: Dimension; codomain: Dimension     # read from the base subscription
    def __call__(cls, *a, **kw) -> NoReturn: ...        # no instances
    def __getitem__(cls, item): ...            # V2E[1] = bound_table()[Local(1)] in embedded;
                                               # NC[V, E] forwarded to __class_getitem__
    def __gt_type__(cls) -> ts.OffsetType: ...  # Codomain -> (Domain, Local), tag=offset_tag
    def bound_table(cls) -> NeighborTable: ... # the table bound in the current embedded run


class NeighborConnectivity[Domain: DimensionIndex, Codomain: DimensionIndex](
    metaclass=ConnectivityMeta
):
    # every subclass declares `Local` (a nested class or a TypeAlias); it is
    # deliberately *not* annotated here, see "The local dimension"
    def __init_subclass__(cls, *, max_neighbors=None, min_neighbors=None): ...
    # writes Local.owner and the counts onto Local; subclassing a declaration is an error


def check_neighbor_table(connectivity, table_or_type) -> None: ...   # A6–A8 as one check


# ── user code ──────────────────────────────────────────────────────────────
class V(DimensionIndex): ...
class E(DimensionIndex): ...
class C(DimensionIndex): ...
class CE(DimensionIndex): ...                  # flattened (cell, edge-of-cell) pairs


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
already names a buffer's origin in gt4py (`__gt_origin__`). Note that the
branch still spells the parameter `Origin` / `V2E.origin`; see
[Implementation](#implementation).

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
   longest importable module prefix and walks the rest as attributes (nested
   classes like `V2E.Local` resolve naturally), the way `pickle` references a
   class. Only where the module path ends is memoized: the attribute walk is
   repeated, so a declaration redefined under the same name (a re-run
   notebook cell) resolves to the new class. `AxisLiteral` stores only the tag
   and derives `kind` and `dim` from it (the `TODO` at `iterator/ir.py:93`);
   printing IR never imports (`resolve_loaded`). `Staggered[D]` tags use the
   small grammar `<owner tag>[<base tag>]`, resolved by subscripting the owner.
2. **Types reaching the IR must be importable.** `<locals>` in a qualname is
   rejected at class creation (for dimensions and connectivities); pickle's
   own `save_global` stays the authoritative check. Interactive `__main__`
   (REPL, notebooks, `python -c`) matters — every notebook declares its
   dimensions there — so the process-pool runner (`BUILD_JOBS_MODE=process`)
   detects a job that references such a class and, with a warning,
   **compiles it in the calling thread** instead of failing in a spawn worker.
3. **No registry.** Classes pickle by reference. One narrow `copyreg` hook
   exists, for `Staggered[D]`, whose bracketed qualname `save_global` cannot
   look up; it reduces to the base dimension and re-interns.
4. **Fingerprints depend on qualified names.** Dimensions are fingerprinted by
   reference (`Staggered[D]` through its base); a connectivity declaration
   additionally by its domain, codomain, `Local` and counts, so a redefinition
   under the same name does not reuse artifacts. Consequently the ADR 0023
   build cache invalidates on module renames.
5. **Codegen names need injective mangling.** `codegen_name(tag)` is a prefix
   escape — `_`→`_u`, `.`→`_d`, `[`→`_l`, `]`→`_r` — used by gtfn, DaCe, the
   roundtrip backend and the nanobind bindings; its inverse
   `from_codegen_name` is used by DaCe to parse names back. The simpler
   "escape `__`, then replace `.`" is **not** injective (`".."` and `"_"`
   collide); the prefix escape is tested exhaustively over its alphabet up
   to length 4.
6. **Staggering is part of the dimension change.** The `_Staggered` name
   prefix needs the name→class lookup that type identity removes, so
   `Staggered[D]` is a real class and part of ADR 0028 (PR 2); see
   [`Staggered[D]`](#staggeredd).
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

### `Staggered[D]`

A staggered dimension sits at the **half-integer positions** of a base
dimension (ADR 0026): `Staggered[K](0)` is the point half a cell *below*
`K(0)`. Cell- and edge-located fields on a Cartesian grid then carry
different dimensions and cannot be combined by accident. The semantics of
ADR 0026 — the position convention, the shift rule and the backend treatment
— are unchanged; what changes is the *encoding*, from a reserved `_Staggered`
prefix on the dimension name to a class.

- **Runtime representation.** `Staggered[K]` is a real, interned class:
  `Staggered` has a metaclass whose `__getitem__` builds one class per base
  and caches it by base identity (with `setdefault`, so concurrent
  compilation threads see one class). It is deliberately *not* a subclass of
  `K`, so a `Field[Dims[Staggered[K]]]` is rejected where a `Field[Dims[K]]`
  is required; it has `base = K` and `K`'s kind. Staggering a local
  dimension, staggering twice, or subclassing a staggered class directly is a
  `TypeError`. A PEP 695 generic cannot serve here — it would be a `typing`
  alias, failing `issubclass` and eve's `type[...]` validation — but under
  `TYPE_CHECKING` `Staggered` *is* declared as one, so mypy and pyright accept
  `Staggered[K]` in annotations and in `Field[Dims[Staggered[K]], float]`.
  `is_staggered`, `flip_staggered` and `as_non_staggered` keep their meaning
  and read `base`.
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
- **IR and serialization.** The tag is
  `gt4py.next.common.Staggered[<base tag>]` — the one tag that is not a
  qualified name. `resolve` parses the `<owner>[<base>]` grammar and
  subscripts the owner, the IR pretty-parser accepts it, and `codegen_name`
  escapes the brackets. `pickle` cannot look up a bracketed qualname, so one
  `copyreg` hook reduces a staggered class to its base and re-subscripts on
  load; fingerprints are likewise taken through the base.
- **Backends.** gtfn emits the staggered tag as a C++ alias of its base tag
  (`_add_staggered_aliases`), since its shift primitive can offset along an
  axis but not rename it. DaCe names map variables by the base
  (`as_non_staggered` in `get_map_variable`), so map fusion sees one
  iteration space for a field and its staggered counterpart.

[[personal/havogt/dimension-generic-fields/dimension-generic-fields|Generic dimensions and statically typed staggering]]
Part II proposed this `Staggered[D]` shape with the same shift convention.
Its static involution overloads (typing `Staggered[K] + 0.5` as landing on
`K` for the checkers) are not part of the stack.

### Binding model

The connectivity *type* conceptually depends on the *table*: two different
tables assign different neighbors to the same domain element. The model:

- The class `V2E` is a **declaration**. Its static part — `V2E.domain`,
  `V2E.codomain`, `V2E.Local`, and *optionally* the counts
  `V2E.Local.max_neighbors` / `V2E.Local.min_neighbors` (they are stored on
  the local dimension; `V2E.max_neighbors` does not exist) — is what
  compilation sees. Declared counts are a constraint on the table; undeclared
  ones are simply taken from the bound table's `NeighborConnectivityType` when
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
  table}`, which is the IR name only of a connectivity that shares a local
  dimension) is rejected with a pointer to `{V2E: table}`. A string key that is
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
- **`Local` is annotated nowhere**, and that is load-bearing. An annotation on
  the base (`Local: ClassVar[type[LocalDimensionIndex]]`) or on the metaclass
  makes a declaration's `Local` a *variable* for the checkers, so
  `Field[Dims[V, V2E.Local], float]` is rejected — by pyright for a nested
  `Local` ("Variable not allowed in type expression"), by mypy for an adopted
  or shared one ("not valid as a type"). A real nested `Local` on the base is
  not an option either: pyright reports an incompatible override in every
  declaration. With no annotation anywhere, all three spellings are types for
  both checkers.
  The cost is that `conn.Local` is not an attribute the checkers know for a
  *generic* `conn`: library code reads it through
  `common.local_dimension_of(conn)`, and code that must name a local dimension
  generically uses a `TypeVar` bound to `LocalDimensionIndex`. This is checked
  for both checkers in CI: the mypy cases in `typing_tests/test_next.yaml`, and
  a pyright run over `typing_tests/pyright_probes.py` in the same nox session
  (pyright is what catches the annotation regression; mypy accepts it).
- A declaration can instead **adopt** a module-level local dimension, written
  `Local: typing.TypeAlias = V2EDim`; the local keeps its own tag, which is then
  the connectivity's `offset_tag`. This is also how the migration script
  rewrites existing `FieldOffset`s without renaming anything. Adopting is *not*
  side-effect free: the adopter becomes the local's `owner` and writes its
  counts onto the module-level class, and ownership is first-come — the first
  connectivity declared over a local dimension owns it, later ones share it. A
  declaration *redefined* under the same tag (a re-run notebook cell) takes
  ownership over again rather than becoming a sharer, and the counts it repeats
  are then checked against the local dimension's own `size=`, not against what
  the stale owner wrote. `ConstList` cannot be adopted: it belongs to no
  connectivity.
- A connectivity can **share** another one's local dimension
  (`class C2CE(NeighborConnectivity[C, CE]): Local: typing.TypeAlias =
  C2E.Local`). ICON4Py's flattened sparse offsets (`C2CE`, `E2ECV`, `E2EC`,
  `C2CEC`) need exactly this, because their results must combine with
  `C2E`-shaped sparse fields. The
  owner stays the first declaration over the local (here `C2E`); the sharer
  must have the same domain (its codomain is free), any counts it repeats must
  equal the owner's, and reductions over the shared axis take the neighbor
  structure from any bound table over it (`connectivity_key_over`: the owner's
  key if bound, else the smallest bound key over that local, which raises
  `KeyError` if there is none).
- **Write an adopted or shared `Local` as a `TypeAlias`.** `Local: TypeAlias =
  C2E.Local` is what keeps mypy treating `C2CE.Local` as a type; with a plain
  assignment it is "not valid as a type" there (pyright accepts either, but
  widens the plain form to `LocalDimensionIndex`). With the alias, annotations
  may name the local dimension through any of its spellings — `C2CE.Local`,
  `C2E.Local`, `V2EDim` — and they are one type.
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
| Frontend types | `ts.OffsetType(source, target)`, tag dropped | `ts.OffsetType` produced by the class (`__gt_type__`), its `tag` field set to `offset_tag`; `V2E.Local` in DSL code types as the local dimension |
| FOAST → GTIR | shift tag = **Python variable name** | tag = `offset_tag`; the variable name is irrelevant (PR 1) |
| ITIR | `OffsetLiteral(str)` + dict lookup; `AxisLiteral(value, kind)` | `OffsetLiteral(offset_tag)`, unchanged node; `AxisLiteral(value)` with `kind` derived |
| ITIR types | `ListType.offset_type: Dimension` | unchanged (already the local dim) |
| Embedded field | `get_offset(provider, axis.value)` | table over the local dimension (`connectivity_key_over`); `V2E.bound_table()` |
| Embedded iterator | positions keyed by name strings; `SparseTag` | positions keyed by dimension classes; `SparseAxis(dim)` |
| gtfn | `dim.value` lookups; `TagDefinition` per tag and per `neighbor_dim` | still a `TagDefinition` per offset tag (and per differing `neighbor_dim`, i.e. for sharers), named by `codegen_name(tag)`; sparse-argument lookups through `connectivity_key_over` |
| DaCe | ~10 `offset_type.value` lookups; `Dimension(offset, LOCAL)` synthesized | `codegen_name` for array/symbol names; `connectivity_key_over`; a local dimension sized from its own table (`local_dimension_size`); no synthesis |
| roundtrip backend | emits `gtx.Dimension("<name>")` / `offset("<tag>")` as source text | emits `<mangled> = gtx.resolve("<tag>")` for axes and `<mangled> = offset("<tag>")` for offsets |
| Provider | `Mapping[str, NeighborTable]` | public: `{V2E: table}` (a dotted tag string is also accepted; the type stays `Mapping[Any, …]`); internal: keyed by `offset_tag`, normalized at the entry points; no deprecation window (a migration script for ICON4Py instead) |

### What it deletes

`FieldOffset` (both forms) and its export; bare-name provider keys (rejected
at the strict entry points); the `_Staggered` prefix and its string sniffing
(replaced by the `Staggered[D]` class and a small `<owner>[<base>]` tag
grammar); `_CONST_DIM` as a magic name; the stored `AxisLiteral.kind`;
`SparseTag`; `NamedIndex` and the dimension half of the mypy plugin;
`DimensionIndex(kind=LOCAL)` as a way to declare a local dimension; and the
convention that `V2EDim = Dimension("V2E", LOCAL)` must sit next to
`V2E = FieldOffset("V2E", ...)`.

**Kept**: `ts.OffsetType` and the runtime connectivity objects (see
[What stays, and why](#what-stays-and-why)); `iterator.runtime.offset("...")`
(the iterator-level API names offsets by string, like the IR); and
string-keyed providers *below* the entry points.

The Cartesian `FieldOffset` goes together with the unstructured one:
Cartesian shifts were already `Dim + i`, and `as_offset` takes the
dimension, `as_offset(KDim, field)`.

### Relation to `Local[V2E]`

[[personal/havogt/dependent-local-dimensions/dependent-local-dimensions|Dependent local dimensions]]
spells the local dimension `Local[V2E]` (a generic parametrized by the
connectivity); this proposal spells it `V2E.Local` (a class nested in the
connectivity). They **cannot be reconciled for a type checker**: making
`Local.__class_getitem__` return `V2E.Local` unifies them at runtime only.
[`typing_probe.py`](typing_probe.py) shows both mypy
and pyright treating `Field[V, V2E.Local]` and `Field[V, Local[V2E]]` as
incompatible in both directions. One spelling has to win, and this proposal
picks `V2E.Local`: the nested form is what makes the tag unique with no
registry, and the counts belong to the owner. The cost is honest and falls
on the chain proposals: their mypy-verified encodings
(`static_hops.py`: `Local[NeighborTable[O, C]]`, `Local[ConnT]`) are built
on subscripting `Local`, and a `TypeVar` cannot be subscripted for a nested
attribute (`Local[ConnT]` fails at runtime with `'TypeVar' object has no
attribute 'Local'`). Those encodings need rewriting to `C.Local`, which
works for concrete `C` and needs a protocol or overloads for the generic
hop-stack case. The semantics of the chain proposals are unaffected; their
static encoding is.

## Alternatives considered

**Keep every concept; fix only which string wins.** Emit `FieldOffset.value`
instead of `foast.Name.id` in `foast_to_gtir`, validate `value == target[-1].value`
eagerly, extend the #1789 regression test. Two PRs, low risk — and the
`foast_to_gtir` part is a real bug that is fixed regardless (PR 1).
But it *enforces* the tangle rather than removing it; no concept goes away.

**A declaration object with a derived local dimension, keeping `Dimension`
instances.** Same shape as this proposal minus static typing and minus type
identity: `local_dim.value == name` by construction, provider keyed by the
declaration object. Superseded by building on dimensions as types instead:
once dimensions are classes, the nested-class form is both simpler and
statically meaningful.

**`(name, kind)` value equality with an interning registry** (the design of
[[shared/dimensions-as-types|dimensions as types]] and GridTools/gt4py#2844).
Keeps independently declared same-named dimensions interchangeable and
avoids the importability rule. Rejected because it decouples the Python
type's identity from the IR's, needs a registry plus `copyreg` plus a custom
fingerprint deconstructor to paper over that gap, and cannot give a nested
`V2E.Local` a unique name without further convention. See *Identity*.

**Integer type parameters for `max_neighbors`** (`LocalDimensionIndex[F: int, M: int]`).
Would need `Literal[6]` type arguments and `Final[F]` over a `TypeVar`;
checkers gain nothing. Class keywords instead.

**Data on the class** (`data: ConnectivityField | Unbound`). Process-global
mutable state; tests bind several meshes per process, and the compile cache
assumes the table travels separately from the type. Binding per call, keyed by
the class, keeps the invariant structural.

**`axis=V2E` as sugar.** Rejected for now; keep the local axis explicit.

**Static-only `max_neighbors` / `min_neighbors`.** Simplest, and correct for
fixed-arity meshes. Rejected as the *only* mode because it binds DSL source
to one mesh family (`fvm_nabla_setup.py` sizes `V2E` from the atlas mesh) and
because skip-value presence is configuration-dependent in ICON, so a static
`min_neighbors` forces either duplicate classes or always-on skip handling.
Declared counts stay available as a constraint; see Binding model.

**A generated `Local` as the default, explicit declaration optional.**
Rejected after running the checkers: a `ClassVar`-typed generated `Local` is
not usable in an annotation under mypy or pyright
([`typing_probe.py`](typing_probe.py), probe P2).

**`Local[C]` as the spelling, with `V2E.Local` an alias.** Rejected: the two
are distinct types for every checker (probe P3); see Relation to `Local[V2E]`.

**Owner-less local axes as non-LOCAL dimensions.** Rejected: they would
become domain dimensions with a range in every program domain and lose the
sparse storage treatment, changing ICON4Py's layout for `LsqUnkDim` fields.

**A `PEP 695` generic for `Staggered[D]`.** Rejected: the subscription would
be a `typing` alias, not a class, failing `issubclass` and eve's `type[...]`
validation and unable to carry a tag. The metaclass subscription builds a
real class; the PEP 695 form is used only under `TYPE_CHECKING`.

## Relation to gt4py ADRs

- **ADR 0019 (Connectivities).** The part naming `FieldOffset` as the frontend
  identifier is superseded by ADR 0029; the `Connectivity` / `NeighborTable` /
  `ConnectivityType` vocabulary is kept ([What stays](#what-stays-and-why)).
- **ADR 0026 (Staggered dimensions).** Its encoding is superseded by
  `Staggered[D]` ([above](#staggeredd)); its semantics are unchanged.
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
   discussed on the PRs; the names become public API in PR 3. The branch still
   spells the domain parameter `Origin` (see Implementation).
2. **Slimming the kept types.** `NeighborConnectivityType` could become
   (declaration, dtype, skip value, width), and `ts.OffsetType`'s
   `source`/`target` fields could be renamed `codomain`/`domain`; both are
   follow-ups outside the stack ([What stays](#what-stays-and-why)).
3. **The mesh proposal.**
   [[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos|A mesh concept with
   first-class halos]] keeps `Koff`/`LsqUnkDim` outside the Mesh object and
   replaces the same `offset_provider` dict this note re-types. The two are
   compatible if the Mesh binds typed connectivity classes to tables, but
   that has to be said in one of the two documents, and the multi-table case
   (halo variants, rewritten `keep_skip_values` tables, several meshes in one
   process) is addressed by neither yet.
4. **Closure variable resolution.** The N2 leak (Python variable name
   becoming the IR tag) is fixed in `foast_to_gtir` (PR 1); the resolution
   mechanism of
   [[personal/havogt/closure-variable-resolution|Closure variable resolution]]
   would be its natural home.
5. **Chain proposals.** Their static encodings need rewriting to `C.Local`
   ([Relation to `Local[V2E]`](#relation-to-localv2e)).

## Implementation

A stack of GridTools/gt4py PRs, each green in CI on its own; the top branch
is `connectivities-as-types-8-typed-positions`.

| # | PR | What |
| --- | --- | --- |
| 1 | GridTools/gt4py#2898 | `fix[next]`: lower unstructured shifts with the offset's own tag (the N2 leak); adds `ts.OffsetType.tag` and makes shift lowering type-driven, so module-qualified offsets (`a(mod.V2E)`) work (supersedes #2730); regression matrix {shift, `neighbor_sum`} × {tag ≠ variable name, tag ≠ local-dimension name}; invalid shift arguments are located `DSLError`s |
| 2 | GridTools/gt4py#2899 | `feat[next]`: a concrete dimension is a class identified by its qualified name (ADR 0028); `codegen_name`; `Staggered[D]`; interactive-`__main__` fallback; `NamedIndex` and the dimension half of the mypy plugin removed; offset tag = local dimension tag = provider key (`FieldOffset(V2EDim.tag, ...)`, `{V2EDim.tag: table}`) |
| 3 | GridTools/gt4py#2907 | `feat[next]`: `NeighborConnectivity`, `LocalDimensionIndex`, `check_neighbor_table`, `local_dimension_of`, declaration fingerprinting; usable in the DSL; shared local dimensions; pyright in the typing nox session; ADR 0029 |
| 4 | GridTools/gt4py#2908 | `feat[next]`: the test tree and docs declare connectivities as classes; iterator-embedded `shift`/`neighbors` accept them; `DimensionIndex(kind=LOCAL)` rejected; two-target `FieldOffset` deprecated |
| 5 | GridTools/gt4py#2909 | `feat[next]`: backends support connectivities sharing a local dimension (`connectivity_key_over`, DaCe `local_dimension_size`); lifts PR 1's xfails |
| 6 | GridTools/gt4py#2910 | `feat[next]!`: class-keyed offset providers, `check_offset_provider` at every entry point; `FieldOffset` removed; `as_offset(dim, field)`; migration script |
| 7 | GridTools/gt4py#2911 | `refactor[next]`: `ConstList` with `size=1` replaces the `_CONST_DIM` aliases by identity checks; `AxisLiteral` drops `kind`; printing IR never imports (`resolve_loaded`) |
| 8 | GridTools/gt4py#2912 | `refactor[next]`: `MultiDimensionIndex` and typed embedded positions |

**Pending on the branch:** the domain type parameter and attribute are
spelled `Origin` / `V2E.origin` there; this note uses `Domain` / `V2E.domain`
(see *Sketch*, Vocabulary).

**ICON4Py migration.** `scripts/python/migrate_connectivities.py` (PR 6)
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
