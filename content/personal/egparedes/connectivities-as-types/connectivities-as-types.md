---
title: "Connectivities as types: one declaration for offset, local dimension and provider key"
author: egparedes
tags: [type-system, type-checking, dimensions, local-dimensions, connectivities, offset-provider, unstructured, neighbor-sum, reduction, frontend, foast, gtir, embedded, gtfn, dace, nominal-types, metaclass, serialization, fingerprint, staggering, migration, adr, tech-debt]
created: 2026-09-17
status: draft
---

> **TL;DR** A neighbor connectivity in `gt4py.next` is spread over three
> user-authored objects that must agree by *string equality* — the
> `FieldOffset` tag, the local `Dimension` name and the `offset_provider` key —
> plus a fourth, hidden one: the Python variable the `FieldOffset` is bound to.
> Make the connectivity a **class** that *contains* its local dimension, is its
> own provider key, and whose identity — like every dimension after
> [[shared/dimensions-as-types|dimensions as types]] — is its qualified Python
> name, which is also its IR tag:
>
> ```python
> class V2E(NeighborConnectivity[V, E]):
>     class Local(LocalDimensionIndex): ...
>
> neighbor_sum(a(V2E), axis=V2E.Local)
> program(a, out=out, offset_provider={V2E: v2e_table})
> ```
>
> `FieldOffset`, `ts.OffsetType`, the string-keyed `OffsetProvider` and the
> `V2EDim`-next-to-`V2E` naming convention disappear. Of the ten cross-object
> identity constraints in the current tree, five (A1–A5) dissolve by
> construction and three (A6–A8) collapse into a single bind-time check.

> **Status**: draft, AI-assisted. Derived from a full audit of the gt4py tree
> at `b3c53fa7e` (v1.2.2, 2026-09-03); every `file:line` reference was
> re-derived from that tree, and the two behaviours the proposal leans on
> hardest were confirmed by *running* them, not by reading. The complete
> constraint catalogue and concept inventory are in the appendix
> [[personal/egparedes/connectivities-as-types/connectivities-as-types_research|Tag and name constraints — full catalogue]].

> **Relation to existing proposals.** This is the *single-hop base* that
> [[personal/havogt/dependent-local-dimensions|Dependent local dimensions and
> connectivity chains]] §6 calls "the shared core" and §9 stages as U0/U1: one
> typed connectivity declaration, derived local dimension, derived offset tag.
> It takes no position on chains, reduction order or the path-vs-forest
> choice, but it does diverge from that proposal's U0/U1 in two places
> (`min_neighbors` instead of `has_skip_values`; a nested `V2E.Local` instead
> of a generic `Local[C]`) — see [Open questions / conflicts](#open-questions--conflicts).
> The remap typing it needs is the `Connectivity[NewD, D0]` rule of
> [[personal/havogt/dimension-generic-fields|Generic dimensions and statically
> typed staggering]]. It builds on [[shared/dimensions-as-types|Dimensions as
> types]] and its implementation in GridTools/gt4py#2844, and **conflicts with
> one of that proposal's decisions** (identity semantics). It also overlaps
> with [[personal/havogt/mesh-and-first-class-halos|A mesh concept with
> first-class halos]], which proposes replacing the same `offset_provider`
> dict with a Mesh object; this note re-types the dict and should be read as
> the typed substrate that proposal can bind through.

## Problem / motivation

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

### Five concepts

| Concept | Role | Replaces |
| --- | --- | --- |
| `DimensionBaseIndex` | root of all dimension types; `value` is the index position | the `DimensionBase` root [[shared/dimensions-as-types|dimensions as types]] deferred |
| `DimensionIndex` | user-declarable primary dimension (as in #2844) | `Dimension` instances |
| `LocalDimensionIndex` | a local dimension: either **owned**, declared nested in its connectivity, or **owner-less** with a fixed `size`; carries `max_neighbors` / `min_neighbors` when they are known statically | `Dimension(..., kind=LOCAL)`, `_CONST_DIM`, and `NeighborConnectivityType.max_neighbors` / `skip_value` |
| `NeighborConnectivity[Origin, Codomain]` | the connectivity **class**: type, shift handle and provider key in one object; contains its `Local` | `FieldOffset`, `runtime.Offset`, `ts.OffsetType`, the `V2EDim` convention |
| `MultiDimensionIndex[D, *Ls]` | position in the product of a primary and one or more local dimensions | untyped position dicts keyed by name strings |

`Connectivity` (the runtime protocol), `CartesianConnectivity`,
`NeighborTable` / `NdArrayConnectivityField` (the data) and
`NeighborConnectivityType` (the compile-time record) all **stay**. What
changes is that the compile-time record is *produced by* the class instead
of being authored in parallel with it.

### Sketch

```python
# ── dimensions ─────────────────────────────────────────────────────────────
class DimensionMeta(type):
    @property
    def tag(cls) -> Tag:                       # identity, and the IR spelling
        return f"{cls.__module__}.{cls.__qualname__}"
    kind: DimensionKind
    # __eq__/__hash__: default type identity. No registry, no copyreg.


class DimensionBaseIndex[T](metaclass=DimensionMeta):
    value: T


class DimensionIndex(DimensionBaseIndex[int]): ...                  # user-declarable


class LocalDimensionIndex(DimensionBaseIndex[int]):
    kind = DimensionKind.LOCAL
    owner: ClassVar[type[NeighborConnectivity] | None]   # None: owner-less, fixed extent
    max_neighbors: ClassVar[int | None]    # None: completed at bind time
    min_neighbors: ClassVar[int | None]    # min == max  <=>  no skip values

    def __init_subclass__(cls, *, size: int | None = None): ...   # owner-less form


class MultiDimensionIndex[D: DimensionIndex, *Ls](DimensionBaseIndex[tuple[D, *Ls]]):
    """Position in the product of a primary dimension and its local dimensions."""
    value: tuple[D, *Ls]                   # *Ls checked at runtime to be LocalDimensionIndex


class Staggered[D: DimensionIndex](DimensionBaseIndex[int]): ...    # replaces the _Staggered prefix


# ── connectivities ─────────────────────────────────────────────────────────
class Connectivity[Src, Dst]: ...   # Protocol; parametrization changes, see Open Q6

class CartesianConnectivity[D: DimensionIndex](Connectivity[D, D]): ...   # semantics unchanged


class NeighborConnectivity[Origin: DimensionIndex, Codomain: DimensionIndex](
    Connectivity[MultiDimensionIndex[Origin, LocalDimensionIndex], Codomain],
    metaclass=ConnectivityMeta,
):
    Local: type[LocalDimensionIndex]   # MUST be declared nested by every subclass (see below)

    def __init_subclass__(
        cls, *, max_neighbors: int | None = None, min_neighbors: int | None = None
    ): ...   # verifies `cls.Local` exists and is a LocalDimensionIndex; sets Local.owner = cls
    # ConnectivityMeta provides: tag (qualified name), __gt_type__() -> NeighborConnectivityType
    # (complete only once the counts are known), __getitem__ so that V2E[1] is a
    # single-neighbor Connectivity, default type identity.


# ── user code ──────────────────────────────────────────────────────────────
class V(DimensionIndex): ...
class E(DimensionIndex): ...


class V2E(NeighborConnectivity[V, E], max_neighbors=6, min_neighbors=5):   # counts optional
    class Local(LocalDimensionIndex): ...                            # explicit, always


class LsqUnk(LocalDimensionIndex, size=3): ...                       # owner-less local axis
class ConstList(LocalDimensionIndex, size=1): ...                    # replaces _CONST_DIM


@field_operator
def op(a: Field[Dims[E], float]) -> Field[Dims[V], float]:
    return neighbor_sum(a(V2E), axis=V2E.Local)                      # local axis stays explicit


program(a, out=out, offset_provider={V2E: v2e_table})
```

Vocabulary: `Origin` is the dimension the remapped field lives on
(`conn.domain[0]`), `Codomain` is what the table entries point into
(`conn.codomain`) — the terms of ADR 0019 and of
[[personal/havogt/dependent-local-dimensions|dependent local dimensions]] §3,
chosen precisely because `FieldOffset.source`/`target` obscure direction.

### Identity is the qualified Python name

A dimension's or connectivity's identity is the Python type; its tag is
`f"{cls.__module__}.{cls.__qualname__}"`. The static view (checkers see
nominal types) and the runtime view (equality is `is`) agree by
construction, and the tag is a valid, unique IR string. This is a deliberate
departure from [[shared/dimensions-as-types|dimensions as types]] and #2844,
which chose `(name, kind)` value equality plus an interning registry so that
the test tree's many independently declared `IDim`s stay interchangeable.
The position taken here: *existing tests that redeclare dummy dimensions can
be fixed in other ways; they should not constrain the design of the
concepts.* The consequences, stated as rules:

1. **Reconstruction from the IR is an import.** `dimension(tag, kind)` is
   replaced by `resolve(tag)`: `importlib.import_module(module)` then walk
   the `qualname` (nested classes like `V2E.Local` resolve naturally). The
   IR references a Python type exactly the way `pickle` references a class.
   `AxisLiteral.kind` becomes redundant, which the `TODO` at
   `iterator/ir.py:93` has wanted since the beginning. Parametrized types
   (`Staggered[D]`, rule 6) have no importable `qualname`; their tag needs a
   small grammar (`<qualname>[<tag>]`) and `resolve` must parse it and
   materialize the class — the one place where "resolution is an import"
   is not literally true.
2. **Types reaching the IR must be importable.** An early heuristic at class
   creation catches the common mistake:

   ```python
   def __init_subclass__(cls, **kw):
       if "<locals>" in cls.__qualname__:
           raise TypeError(f"'{cls.__qualname__}' must be declared at module level.")
   ```

   It is neither necessary nor sufficient — `type("Dyn", ...)` inside a
   function passes, a class `del`'d after creation passes — so the
   authoritative check remains pickle's own `save_global` at pickle time,
   exactly as for any class. The `spawn`-based compile workers of
   `otf/runners.py` re-execute the main script as `__mp_main__`, so classes
   declared in a *file's* `__main__` resolve there (provided the script has
   an `if __name__ == "__main__":` guard, which the pool already requires).
   What does not resolve is *interactive* `__main__`: the REPL, notebooks,
   `python -c`. Documented limitation, not fixed.
3. **No registry, no `copyreg`.** Classes pickle by reference, which is
   correct for module-level types; #2844's `copyreg.pickle(DimensionMeta, ...)`
   and `_DIMENSION_REGISTRY` are removed again. Nothing equivalent creeps
   back in, because string-keyed providers are not accepted at all (there is
   no deprecation window), so no name→class table is ever needed.
4. **Cache fingerprints depend on module paths.** #2844 adds a
   `fingerprinting.py` deconstructor keyed on `(tag, kind)` (recorded in
   ADR 0028) *specifically so that* a dimension is not fingerprinted by
   qualified name. Under type identity it is, by definition; renaming a
   module invalidates compiled artifacts (ADR 0023's cache). Correct for
   this model; a consequence to record, not a reversal of ADR 0023 itself.
5. **Codegen names need injective mangling.** `generated::<tag>_t` (gtfn) and
   DaCe symbols cannot contain dots. One shared `codegen_name(tag)`, injective
   (escape existing `__` before replacing `.`), used by both backends; gtfn's
   existing `TagDefinition.alias` mechanism can shorten where needed.
6. **Staggering comes into scope.** `flip_staggered` builds
   `Dimension(f"_Staggered{name}")` from a string (`common.py:1452-1457`).
   There is no such type to import, so ADR 0026's prefix cannot survive
   type identity; `Staggered[D]` becomes a real parametrized dimension type.
   This is the `DimensionBase` extension point named in
   [[shared/dimensions-as-types|dimensions as types]], now with a concrete
   requirement.

### Binding model

The connectivity *type* conceptually depends on the *table*: two different
tables assign different neighbors to the same origin element. The model:

- The class `V2E` is a **type constructor**. Its static part — `Origin`,
  `Codomain`, `Local`, and *optionally* `max_neighbors` / `min_neighbors` —
  is what compilation sees. When the counts are declared, they are a
  constraint the table must satisfy. When they are not, they are **completed
  at bind time**: from the table's shape and skip value in the JIT flow, or
  from a `NeighborConnectivityType` supplied through `connectivities=` in the
  AOT flow (`ffront/decorator.py:188-208`), which already passes exactly that
  record without any data. `NeighborConnectivityType` is derived from class
  plus binding, and is complete only once both are known.
- Why the counts are not static-only. Two in-tree and in-ICON facts: the
  arity of the same connectivity varies per mesh (`fvm_nabla_setup.py:99`
  sizes `V2E` from `edges_per_node`, computed from the atlas mesh), so a
  static count binds the DSL source to one mesh family and makes
  `Field[Dims[Vertex, V2E.Local]]` signatures unshareable across meshes; and
  skip-value presence is *configuration*-dependent in ICON (`icon.py:130`:
  pentagon offsets have skip values on the icosahedron but not on the torus;
  boundary offsets only on limited-area or distributed grids), so a static
  `min_neighbors` would force ICON4Py to either declare two classes per
  boundary offset or always compile skip-mask handling, defeating its
  `keep_skip_values=False` table rewrite (`base.py:180`). Declared counts
  remain the right choice for fixed-arity meshes, and the compile cache is
  unaffected either way: `NeighborConnectivityType` is part of the key today
  and stays so.
- A **bound connectivity** is the class plus a table. Within one program
  invocation there is exactly one table per class, so every occurrence of
  `V2E` in that invocation denotes the same mapping. The provider
  `{V2E: table}` *is* the binding, and a dict makes one-table-per-class
  structural.
- The table is validated against the static part **once, at bind time**:
  shape `(n, max_neighbors)`, integral dtype, skip values present iff
  `min_neighbors < max_neighbors`. This one runtime check replaces the whole
  declaration-vs-provider family of constraints (A6–A8 in the appendix); they
  do not vanish, they become a single validated invariant.
- The class never holds data. `_ConnectivityFileRef` and the compile cache
  (ADR 0023) keep working unchanged: the type part is fingerprinted by
  qualified name, the table crosses process boundaries as it does today.

### The local dimension

- `V2E.Local` **must be declared explicitly** as a nested
  `class Local(LocalDimensionIndex): ...`; `__init_subclass__` verifies it
  and sets `Local.owner`. There is no generated form. This is settled by
  running both checkers on
  [`typing_probe.py`](connectivities-as-types/typing_probe.py): an explicitly
  declared nested `Local` is a real, distinct type under mypy `--strict` and
  pyright (`Field[V, V_E2E.Local]` vs `Field[E, E_V2V.Local]` is an
  `arg-type` error), while a `Local` generated in `__init_subclass__` and
  annotated `ClassVar[type[LocalDimensionIndex]]` on the base is rejected by
  both as *not valid as a type* — the very error class
  [[shared/dimensions-as-types|dimensions as types]] exists to remove. Two
  lines per connectivity buy a real type; ICON4Py has 16 declarations. Its
  tag is `<owner tag>.Local`, unique by construction.
- `Local` is **not** passed through the base subscription
  (`NeighborConnectivity[V, "V2E.Local", E]`): mypy accepts that string
  forward reference, pyright reports `Class definition for "V2E" depends on
  itself`. The metaclass reads `cls.Local` after the class body has run; on
  Python 3.13 a PEP 696 default type parameter is the typed alternative.
- `max_neighbors` / `min_neighbors` are class keywords, exactly as #2844 does
  `kind`, and optional (see Binding model). They are *not* type parameters:
  Python has no integer-valued type parameters, and nothing static needs the
  count — what must be in the type is *which* local dimension.
- **Owner-less local dimensions** exist: `class LsqUnk(LocalDimensionIndex, size=3)`
  has `owner = None`, `min == max == size`, no skip values, and never appears
  in the provider. ICON4Py needs this today: `LsqUnkDim` (`dimension.py:18`,
  LOCAL kind, "number of least-squares unknowns") is the middle axis of a
  coefficient field over `(CellDim, LsqUnkDim, C2E2CDim)` and indexes no
  table; `RBFDimension` is an enum of stencil sizes that plays the same role.
  Such axes want the *storage and layout* treatment of a sparse dimension,
  not a connectivity; declaring them non-LOCAL instead would make them domain
  dimensions requiring a range in every program domain and would change
  ICON4Py's memory layout for those fields. Backends handle the owner-less
  form exactly as they handle `_CONST_DIM` today, generalized from size 1 to
  size *n*.
- `_CONST_DIM` (`iterator/embedded.py:220`) is therefore the owner-less
  `ConstList(LocalDimensionIndex, size=1)`, replacing a magic name that is
  special-cased at twelve use sites today (six in `iterator/embedded.py`,
  six in the DaCe lowering).
- `neighbor_sum(..., axis=V2E.Local)` — the local axis stays **explicit**.
  `axis=V2E` as sugar is deliberately not offered; it would blur the two
  concepts the design separates.

### `MultiDimensionIndex`

Included so the final picture is complete, even though it is the last piece
to implement and nothing else depends on it:

- It is the **index type of a sparse-field position**. `V2E.Local(2)` alone
  cannot address an element; `(V(3), V2E.Local(2))` can. Today that pair is a
  dict keyed by dimension-name strings (`iterator/embedded.py:597-616`).
- It is the **domain index of a `NeighborConnectivity`**, which is what makes
  `Connectivity[MultiDimensionIndex[Origin, Local], Codomain]` a real
  index-to-index function type.
- `*Ls` is unconstrained because `TypeVarTuple` cannot carry a bound; a
  runtime check in `__init_subclass__` covers what the checker cannot.

### Effect per layer

| Layer | Today | Under the proposal |
| --- | --- | --- |
| Frontend declaration | `Dimension(LOCAL)` + `FieldOffset` + provider key | one class |
| Frontend types | `ts.OffsetType(source, target)`, tag dropped | `ConnectivityType`, produced by the class (`TODO` at `type_specifications.py:74`) |
| FOAST → GTIR | shift tag = **Python variable name** (`foast_to_gtir.py:305, 331`) | tag = `cls.tag`; the variable name is irrelevant |
| ITIR | `OffsetLiteral(str)` + dict lookup | node carries the local dim's tag, resolved by import — the `CartesianOffset` pattern |
| ITIR types | `ListType.offset_type: Dimension` | unchanged (already the local dim) |
| Embedded field | `get_offset(provider, axis.value)` | `axis.owner`, bound table from context |
| Embedded iterator | positions keyed by name strings; `SparseTag` | `MultiDimensionIndex` positions (last step) |
| gtfn | `dim.value` lookups for sparse args; `TagDefinition` per tag and per `neighbor_dim` | `local_dim.owner`; `codegen_name(tag)` |
| DaCe | ~10 `offset_type.value` lookups; `Dimension(offset, LOCAL)` synthesized at `gtir_to_sdfg_lambda.py:1155` | `local_dim.owner`; no synthesis |
| roundtrip backend | emits `gtx.Dimension("<name>", kind=...)` / `offset("<tag>")` as *source text* (`runners/roundtrip.py:175-179`) | emits imports of the qualified names, via `codegen_name` for the Python identifier |
| Provider | `Mapping[str, NeighborTable]` | `Mapping[type[NeighborConnectivity], NeighborTable]`; **no string keys** and no deprecation window (a scripted migration for ICON4Py instead) |

### What it deletes

`FieldOffset` as the unstructured declaration; `runtime.Offset` as its base
(the `TODO` at `fbuiltins.py:467-470`); `ts.OffsetType`; the four
`OffsetProvider*` aliases in their
string-keyed form; `dimension(tag, kind)`, the interning registry and the
`copyreg` hook from #2844; the `_Staggered` prefix and its string sniffing;
`_CONST_DIM` as a magic name; and the convention that
`V2EDim = Dimension("V2E", LOCAL)` must sit next to `V2E = FieldOffset("V2E", ...)`.

**Deleted only together with its replacement:** the single-target
(Cartesian) `FieldOffset` form. Cartesian *shifts* already use
`Dimension + int` (GridTools/gt4py#2411), but `as_offset(offset, field)`
(`ffront/experimental.py:17`) requires a Cartesian `FieldOffset`
(`type_deduction.py:956-967`, S6 in the appendix), is used in 5 test modules
plus the `Ioff`/`Koff`/`EdgeOffset` fixtures in `cases_utils.py:163-169`, and
ICON4Py calls it in diffusion and the dycore (`as_offset(Koff, ...)`, with
`offset_provider={"Koff": KDim}`). Since there is no deprecation window, a
dimension-based signature — `as_offset(KDim, field)` is the obvious
candidate — must land in the **same release** that removes `FieldOffset`,
and the migration script must rewrite these call sites too.

### Relation to `Local[V2E]`

[[personal/havogt/dependent-local-dimensions|Dependent local dimensions]]
spells the local dimension `Local[V2E]` (a generic parametrized by the
connectivity); this proposal spells it `V2E.Local` (a class nested in the
connectivity). They **cannot be reconciled for a type checker**: making
`Local.__class_getitem__` return `V2E.Local` unifies them at runtime only.
[`typing_probe.py`](connectivities-as-types/typing_probe.py) shows both mypy
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
`foast_to_gtir` part is a real bug that should land regardless. But it
*enforces* the tangle rather than removing it; no concept goes away.

**A declaration object with a derived local dimension, keeping `Dimension`
instances** (the "Option B" that led here). Same shape as this proposal minus
static typing and minus type identity: `local_dim.value == name` by
construction, provider keyed by the declaration object. Superseded by
building on dimensions-as-types instead: once dimensions are classes, the
nested-class form is both simpler and statically meaningful.

**`(name, kind)` value equality with an interning registry** (the #2844
design). Keeps independently declared same-named dimensions interchangeable
and avoids the importability rule. Rejected here because it decouples the
Python type's identity from the IR's, needs a registry plus `copyreg` plus a
custom fingerprint deconstructor to paper over that gap, and cannot give a
nested `V2E.Local` a unique name without further convention. See conflicts.

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
([`typing_probe.py`](connectivities-as-types/typing_probe.py), probe P2).

**`Local[C]` as the spelling, with `V2E.Local` an alias.** Rejected: the two
are distinct types for every checker (probe P3); see Relation to `Local[V2E]`.

**Owner-less local axes as non-LOCAL dimensions.** Rejected: they would
become domain dimensions with a range in every program domain and lose the
sparse storage treatment, changing ICON4Py's layout for `LsqUnkDim` fields.

## Open questions / conflicts

**Conflicts, stated explicitly:**

- **[[shared/dimensions-as-types|Dimensions as types]] / #2844 — identity
  semantics.** That proposal decides `(name, kind)` equality, an interning
  factory (`Dimension("I") is Dimension("I")`), and a value-based
  `__reduce__`. Its reasons are concrete and should be weighed, not waved
  away: `tests/next_tests` declares a dimension named `"I"` 46 times and
  `IDim` in 45 files (its Step 0); ADR 0028 records that nominal identity
  "changes the meaning of `Dimension("I") == Dimension("I")`, which downstream
  code relies on"; and its "first declaration wins" residual means
  `Dimension("I") is <some module's I>` is already *not* guaranteed today.
  This proposal reverses all three decisions in favour of type identity. One
  argument in its favour that the shared note itself raises as a known
  residual: under `(name, kind)` equality the `typing` subscription cache
  aliases `Field[Dims[I]] is Field[Dims[I2]]` for two distinct same-named
  classes, so the static and runtime views disagree exactly there; under type
  identity that aliasing disappears. If accepted, `shared/dimensions-as-types.md`
  needs a revision and #2844 should stop hardening the registry and `copyreg`
  paths. The test-tree consequence is real: the many function-local dummy
  dimensions in `tests/next_tests` must move to module level (a larger sweep
  than #2845).
- **ADR 0028 / #2844 (fingerprinting of dimensions).** #2844 adds a
  `(tag, kind)` fingerprint deconstructor; under type identity dimensions are
  fingerprinted by qualified name instead, and ADR 0023's cache invalidates
  on module renames. A consequence to record in the new ADR.
- **[[personal/havogt/mesh-and-first-class-halos|A mesh concept with
  first-class halos]].** Proposes replacing the `offset_provider` dict with a
  Mesh object and keeps `Koff`/`LsqUnkDim` outside it. This note re-types the
  same dict. The two are compatible if the Mesh binds typed connectivity
  classes to tables, but that has to be said in one of the two documents,
  and the multi-table case (halo variants, rewritten `keep_skip_values`
  tables, several meshes in one process) is addressed by neither yet.
- **ADR 0026 (Staggered dimensions).** The `_Staggered` name prefix cannot
  survive type identity; `Staggered[D]` is required, not optional.
- **ADR 0019 (Connectivities).** The part naming `FieldOffset` as the
  frontend identifier is superseded; the `Connectivity` / `NeighborTable` /
  `ConnectivityType` vocabulary is kept and strengthened.
- **[[personal/havogt/closure-variable-resolution|Closure variable
  resolution]].** The N2 leak (Python variable name becoming the IR tag) is a
  closure-variable-resolution bug in `foast_to_gtir`; that proposal's
  resolution mechanism should be the vehicle for the fix.

**Open:**

1. **Naming.** `NeighborConnectivity[Origin, Codomain]` vs the sketch's
   `StaticMultiConnectivity[F, L, T]` vs the existing
   `NeighborConnectivityType.source_dim`/`neighbor_dim` (a third vocabulary
   this note would otherwise add to the two it criticizes); `Local` vs `Dim`;
   `min_neighbors` vs `has_skip_values`. Converge with
   [[personal/havogt/dependent-local-dimensions|dependent local dimensions]]
   before either lands.
2. **How `Local` reaches the base's parametrization.** Discovered by the
   metaclass after the class body (works on 3.12, untyped in the base), or a
   PEP 696 default type parameter (3.13, typed)? AGENTS.md invites 3.13
   features where they improve a design; this is a candidate.
3. **`__main__` in spawn workers.** Document, or detect and fall back to
   in-process compilation with a warning?
4. **ICON4Py migration.** `gtx.FieldOffset`, `Dimension(..., LOCAL)`,
   `as_offset(Koff, ...)` and string-keyed providers (`{"V2E": ..., "Koff": KDim}`)
   are all public and all change in **one release, with no deprecation
   window**. ICON4Py's keys are bare names, so no import-based shim could
   have resolved them anyway; the migration is a script over
   `dimension.py` (16 offsets, 15 local dims), the `offset_provider`
   construction in `grid/base.py`, and the `as_offset` call sites. `LsqUnkDim`
   becomes an owner-less `LocalDimensionIndex` with `size=`. What the script
   cannot decide is whether each connectivity declares its counts or leaves
   them to bind time; the ICON skip-value rules (`icon.py:130`) suggest
   leaving `min_neighbors` open for the boundary offsets.
5. **Duplicate declarations.** With type identity, two modules declaring
   `class V2E(...)` are two connectivities. Is a warning on same-`__name__`
   collisions in one program useful, or noise?
6. **What is an instance of `V2E`?** Today `Connectivity` is
   `Field[DimsT, IntegralScalar]` parametrized by `Dims` plus a codomain
   (`common.py:990`), and `CartesianConnectivity` is a dataclass with
   instances. The sketch's `Connectivity[Src, Dst]` is a different
   parametrization, and `V2E` is a data-less type constructor subclassing a
   runtime protocol. Either instances of `V2E` are the bound tables (so
   `{V2E: table}` becomes `V2E(table)`), or `NeighborConnectivity` is not a
   `Connectivity` subclass at all and only *produces* one at bind time. The
   sketch leaves this open and should not.

## Staging

Each step is landable separately and leaves the tree green.

0. **Fix the N2 leak now, independently.** `foast_to_gtir.py:305, 331` emit
   the offset's own tag; extend `test_offset_dimensions_names.py` to
   `neighbor_sum`, embedded and DaCe so A3 is *visible*. This is a bug today.
1. **#2844 lands**, then the `DimensionBaseIndex` root and type identity
   (registry and `copyreg` removed, importability rule added,
   `resolve(tag)` by import). Revision of `shared/dimensions-as-types.md`.
2. **`NeighborConnectivity` + `LocalDimensionIndex`** (owned and owner-less)
   in `common`; object-keyed provider; `FieldOffset` and string keys removed
   outright; bind-time completion and validation of the counts.
3. **`ts.OffsetType` → `ConnectivityType`**; `as_offset` gets its
   dimension-based signature in the same step, so the Cartesian `FieldOffset`
   form can go with the rest.
4. **Backends** switch from `local_dim.value` lookups to `local_dim.owner`,
   one file at a time (`gtfn_module.py`, `unroll_reduce.py`, `gtir_to_sdfg*.py`,
   `nd_array_field.py`, `iterator/embedded.py`, and the source-emitting
   `runners/roundtrip.py`). `codegen_name` introduced.
5. **`Staggered[D]`**, superseding ADR 0026; `_CONST_DIM` becomes the
   owner-less `ConstList` local dimension.
6. **`MultiDimensionIndex`** and typed embedded positions.
7. **Typing tests** (`typing_tests/test_next.yaml`, pyright): `Field[Dims[V, V2E.Local], float]`,
   a `TypeVar` bound to `LocalDimensionIndex`, a negative case mixing two
   connectivities' locals — the probes in
   [`typing_probe.py`](connectivities-as-types/typing_probe.py) are the
   starting set.
8. **Test-tree migration** as its own PR (`toy_connectivity.py`,
   `cases_utils.py`, `fvm_nabla_setup.py` are the fixture modules everything
   imports).

An ADR in gt4py (next in sequence after 0028) should record steps 1–2, since
they reverse two recorded decisions and change a public-API shape.

## Appendices

- [[personal/egparedes/connectivities-as-types/connectivities-as-types_research|Tag and name constraints — full catalogue]]:
  the five name spaces; the 10 cross-object identity constraints (A1–A10),
  9 name-format constraints (F1–F9) and 7 structural constraints (S1–S7),
  each with `file:line`; per-context tables for embedded, IR, gtfn and DaCe;
  the complete concept inventory (L0–L9); class-hierarchy and name-flow
  diagrams; the two executed experiments.
- [`typing_probe.py`](connectivities-as-types/typing_probe.py): the mypy /
  pyright probes behind the `Local` decisions (explicit nested class works;
  generated `ClassVar` form and `Local[C]` reconciliation do not).
