---
title: "Connectivities as types: one declaration for offset, local dimension and provider key"
author: egparedes
tags: [type-system, dimensions, local-dimensions, connectivities, offset-provider, unstructured, neighbor-sum, reduction, frontend, foast, gtir, embedded, gtfn, dace, nominal-types, metaclass, serialization, fingerprint, staggering, migration, adr, tech-debt]
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
> class V2E(NeighborConnectivity[V, E], max_neighbors=6): ...
>
> neighbor_sum(a(V2E), axis=V2E.Local)
> program(a, out=out, offset_provider={V2E: v2e_table})
> ```
>
> `FieldOffset`, `ts.OffsetType`, the string-keyed `OffsetProvider` and the
> `V2EDim`-next-to-`V2E` naming convention disappear. Seven of the ten
> cross-object identity constraints in the current tree become tautologies.

> **Status**: draft, AI-assisted. Derived from a full audit of the gt4py tree
> at `b3c53fa7e` (v1.2.2, 2026-09-03); every `file:line` reference was
> re-derived from that tree, and the two behaviours the proposal leans on
> hardest were confirmed by *running* them, not by reading. The complete
> constraint catalogue and concept inventory are in the appendix
> [[personal/egparedes/connectivities-as-types/connectivities-as-types_research|Tag and name constraints — full catalogue]].

> **Relation to existing proposals.** This is the *single-hop base* that
> [[personal/havogt/dependent-local-dimensions|Dependent local dimensions and
> connectivity chains]] §9 calls "the shared core" (stages U0/U1): one typed
> connectivity declaration, derived local dimension, derived offset tag. It
> takes no position on chains, reduction order or the path-vs-forest choice.
> It builds on [[shared/dimensions-as-types|Dimensions as types]] and its
> implementation in GridTools/gt4py#2844, and **conflicts with one of that
> proposal's decisions** (identity semantics) — see
> [Open questions / conflicts](#open-questions--conflicts).

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
covers exactly `a(Off[1])` on `GTFN_CPU` and nothing else. Reductions and
sparse arguments still require it.

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
(`iterator/ir.py:99-101`): the IR node carries **dimensions**. No tag, no
provider entry, no lookup. Unstructured shifts still lower to
`itir.OffsetLiteral(value: str)` and a dict lookup. Every constraint above
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
| `LocalDimensionIndex` | a local dimension; always **owned** by a connectivity; carries `max_neighbors` / `min_neighbors` | `Dimension(..., kind=LOCAL)` + `NeighborConnectivityType.max_neighbors` / `skip_value` |
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
    owner: ClassVar[type[NeighborConnectivity]]
    max_neighbors: ClassVar[int]
    min_neighbors: ClassVar[int]           # min == max  <=>  no skip values


class MultiDimensionIndex[D: DimensionIndex, *Ls](DimensionBaseIndex[tuple[D, *Ls]]):
    """Position in the product of a primary dimension and its local dimensions."""
    value: tuple[D, *Ls]                   # *Ls checked at runtime to be LocalDimensionIndex


class Staggered[D: DimensionIndex](DimensionBaseIndex[int]): ...    # replaces the _Staggered prefix


# ── connectivities ─────────────────────────────────────────────────────────
class Connectivity[Src, Dst]: ...                                    # Protocol, as today

class CartesianConnectivity[D: DimensionIndex](Connectivity[D, D]): ...   # unchanged


class NeighborConnectivity[Origin: DimensionIndex, Codomain: DimensionIndex](
    Connectivity[MultiDimensionIndex[Origin, "Local"], Codomain],
    metaclass=ConnectivityMeta,
):
    Local: ClassVar[type[LocalDimensionIndex]]   # generated in __init_subclass__ unless declared

    def __init_subclass__(cls, *, max_neighbors: int, min_neighbors: int | None = None): ...
    # ConnectivityMeta provides: tag (qualified name), __gt_type__() -> NeighborConnectivityType,
    # __getitem__ so that V2E[1] is a single-neighbor Connectivity, default type identity.


# ── user code ──────────────────────────────────────────────────────────────
class V(DimensionIndex): ...
class E(DimensionIndex): ...
class V2E(NeighborConnectivity[V, E], max_neighbors=6): ...         # V2E.Local generated


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
   `iterator/ir.py:93` has wanted since the beginning.
2. **Types reaching the IR must be importable.** Enforced at class creation:

   ```python
   def __init_subclass__(cls, **kw):
       if "<locals>" in cls.__qualname__:
           raise TypeError(f"'{cls.__qualname__}' must be declared at module level.")
   ```

   A class declared in `__main__` resolves in-process but not in the
   `spawn`-based compile workers of `otf/runners.py`, where `__main__` is a
   different module — the same restriction stdlib pickle has, for the same
   reason. Documented limitation, not fixed.
3. **No registry, no `copyreg`.** Classes pickle by reference, which is
   correct for module-level types; #2844's `copyreg.pickle(DimensionMeta, ...)`
   and `_DIMENSION_REGISTRY` are removed again.
4. **Cache fingerprints depend on module paths.** ADR 0023's deconstructor
   keyed dimensions on `(tag, kind)` *specifically so that* a dimension was
   not fingerprinted by qualified name. Under type identity it is, by
   definition; renaming a module invalidates compiled artifacts. Correct for
   this model, and a recorded reversal.
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
  `Codomain`, `Local`, `max_neighbors`, `min_neighbors` — is what compilation
  sees, and `NeighborConnectivityType` is derived from it. **Static only**:
  `max_neighbors` is declared on the class, not inferred from data.
- A **bound connectivity** is the class plus a table. Within one program
  invocation there is exactly one table per class, so every occurrence of
  `V2E` in that invocation denotes the same mapping. The provider
  `{V2E: table}` *is* the binding, and a dict makes one-table-per-class
  structural.
- The table is validated against the static part **once, at bind time**:
  shape `(n, max_neighbors)`, integral dtype, skip values present iff
  `min_neighbors < max_neighbors`. This one check replaces the whole
  declaration-vs-provider family of constraints (A6–A8 in the appendix).
- The class never holds data. `_ConnectivityFileRef` and the compile cache
  (ADR 0023) keep working unchanged: the type part is fingerprinted by
  qualified name, the table crosses process boundaries as it does today.

### The local dimension

- `V2E.Local` is **generated** in `__init_subclass__` unless the user declares
  a nested `class Local(LocalDimensionIndex): ...` — the explicit form gives a
  *statically* distinct type when a signature needs one; the generated form
  is annotated `type[LocalDimensionIndex]` on the base and covers everything
  else. Its tag is `<owner tag>.Local`, unique by construction.
- `max_neighbors` / `min_neighbors` are class keywords, exactly as #2844 does
  `kind`. They are *not* type parameters: Python has no integer-valued type
  parameters, and nothing static needs the count — what must be in the type
  is *which* local dimension.
- `_CONST_DIM` (`iterator/embedded.py:220`) becomes a real
  `LocalDimensionIndex` with no owner and `max_neighbors = 1`, replacing a
  magic name that is special-cased at seven lookup sites today.
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
| Provider | `Mapping[str, NeighborTable]` | `Mapping[type[NeighborConnectivity], NeighborTable]`; string keys accepted with a warning during deprecation, resolved by import |

### What it deletes

`FieldOffset`; `runtime.Offset` as its base (the `TODO` at
`fbuiltins.py:467-470`); `ts.OffsetType`; `is_cartesian_offset` and the
single-target `FieldOffset` form (Cartesian shifts already use `Dimension + int`
after GridTools/gt4py#2411); the four `OffsetProvider*` aliases in their
string-keyed form; `dimension(tag, kind)`, the interning registry and the
`copyreg` hook from #2844; the `_Staggered` prefix and its string sniffing;
`_CONST_DIM` as a magic name; and the convention that
`V2EDim = Dimension("V2E", LOCAL)` must sit next to `V2E = FieldOffset("V2E", ...)`.

### Relation to `Local[V2E]`

[[personal/havogt/dependent-local-dimensions|Dependent local dimensions]]
spells the local dimension `Local[V2E]` (a generic parametrized by the
connectivity); this proposal spells it `V2E.Local` (a class nested in the
connectivity). These should be **the same object**: `Local.__class_getitem__`
can return `V2E.Local`, so both spellings denote one type and the chain
proposals (`Local[V2E]` pushed onto a hop stack, or `Local[C, P]` with a
parent) sit on top unchanged. Nesting is preferred for the *declaration* side
because the local dimension's `max_neighbors`/`min_neighbors` belong to the
owner, and because the nested form is what makes the tag unique with no
registry.

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

**Bound (data-inferred) `max_neighbors`.** Rejected for now; static only. Can
be revisited if a generic-mesh consumer needs it, at the cost that such a
class's `NeighborConnectivityType` is not derivable before binding.

## Open questions / conflicts

**Conflicts, stated explicitly:**

- **[[shared/dimensions-as-types|Dimensions as types]] / #2844 — identity
  semantics.** That proposal decides `(name, kind)` equality, an interning
  factory (`Dimension("I") is Dimension("I")`), and a value-based
  `__reduce__`. This proposal reverses all three in favour of type identity.
  If accepted, `shared/dimensions-as-types.md` needs a revision and #2844
  should stop hardening the registry and `copyreg` paths. The test-tree
  consequence is real: the many function-local dummy dimensions in
  `tests/next_tests` must move to module level (a larger sweep than #2845).
- **ADR 0023 (Fingerprinting).** Dimensions become fingerprinted by
  qualified name. Deliberate; must be recorded.
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
   `StaticMultiConnectivity[F, L, T]`; `Local` vs `Dim`; `min_neighbors` vs
   `has_skip_values`. Converge with
   [[personal/havogt/dependent-local-dimensions|dependent local dimensions]]
   before either lands.
2. **Declared vs generated `Local`.** Is the optional explicit nested class
   worth its rule surface, or is `type[LocalDimensionIndex]` on the base
   enough for every real signature?
3. **`__main__` in spawn workers.** Document, or detect and fall back to
   in-process compilation with a warning?
4. **ICON4Py surface.** `gtx.FieldOffset`, `Dimension(..., LOCAL)` and
   string-keyed providers are all public. The deprecation window will
   dominate the schedule; the string-key compatibility shim (resolve via
   import, warn) is what makes it tolerable.
5. **Duplicate declarations.** With type identity, two modules declaring
   `class V2E(...)` are two connectivities. Is a warning on same-`__name__`
   collisions in one program useful, or noise?

## Staging

Each step is landable separately and leaves the tree green.

0. **Fix the N2 leak now, independently.** `foast_to_gtir.py:305, 331` emit
   the offset's own tag; extend `test_offset_dimensions_names.py` to
   `neighbor_sum`, embedded and DaCe so A3 is *visible*. This is a bug today.
1. **#2844 lands**, then the `DimensionBaseIndex` root and type identity
   (registry and `copyreg` removed, importability rule added,
   `resolve(tag)` by import). Revision of `shared/dimensions-as-types.md`.
2. **`NeighborConnectivity` + `LocalDimensionIndex`** in `common`;
   `FieldOffset` kept as a deprecated alias that constructs one;
   object-keyed provider accepted alongside string-keyed.
3. **`ts.OffsetType` → `ConnectivityType`**; drop Cartesian `FieldOffset`.
4. **Backends** switch from `local_dim.value` lookups to `local_dim.owner`,
   one file at a time (`gtfn_module.py`, `unroll_reduce.py`, `gtir_to_sdfg*.py`,
   `nd_array_field.py`, `iterator/embedded.py`). `codegen_name` introduced.
5. **`Staggered[D]`**, superseding ADR 0026; `_CONST_DIM` as a real type.
6. **`MultiDimensionIndex`** and typed embedded positions.
7. **Typing tests** (`typing_tests/test_next.yaml`, pyright): `Field[Dims[V, V2E.Local], float]`,
   a `TypeVar` bound to `LocalDimensionIndex`, a negative case mixing two
   connectivities' locals.
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
