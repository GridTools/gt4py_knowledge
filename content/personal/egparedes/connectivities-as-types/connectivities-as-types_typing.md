---
title: "Connectivities as types — decisions made by running mypy and pyright"
author: egparedes
tags: [type-system, type-checking, local-dimensions, dependent-types, nominal-types]
created: 2026-10-01
status: draft
---

> **Appendix** to [[personal/egparedes/connectivities-as-types/connectivities-as-types|Connectivities as types]].
> Every place in the design where the shape of a declaration was decided by what a
> checker accepts rather than by taste: why `Local` is annotated nowhere, why an
> adopted or shared `Local` must be a `TypeAlias`, what the `__add__` self-type
> costs, and why `Local[C]` and `V2E.Local` cannot be reconciled. The probes are
> [`typing_probe.py`](typing_probe.py) and
> [`staggered_probe.py`](staggered_probe.py).

## 1. The `__add__` self-type costs two suppressions

The last row needs `__add__` declared with `cls: type[AnyCartesianAxisIndex]`.
Both checkers bind that signature correctly at every call site and both reject it
at the *definition* site, with different diagnostics — mypy `[misc]` ("self"
parameter missing for a non-static method), pyright
`reportGeneralTypeIssues` ("Type of parameter `cls` must be a supertype of its
class `DimensionMeta`") — so it costs two separately-spelled suppressions. P4 of
[`staggered_probe.py`](staggered_probe.py) pins this down; the same restriction
is what
[[personal/havogt/dimension-generic-fields/dimension-generic-fields|generic dimensions]]
suppresses for its overload pairs.

## 2. Why `Local` is annotated nowhere, and must be a `TypeAlias` when shared

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
  connectivity declared over a local dimension owns it, later ones share it and
  are recorded in its `sharers`. A
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

## 3. Relation to `Local[V2E]`

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

## 4. Alternatives these decided

**A generated `Local` as the default, explicit declaration optional.**
Rejected after running the checkers: a `ClassVar`-typed generated `Local` is
not usable in an annotation under mypy or pyright
([`typing_probe.py`](typing_probe.py), probe P2).

**`Local[C]` as the spelling, with `V2E.Local` an alias.** Rejected: the two
are distinct types for every checker (probe P3); see Relation to `Local[V2E]`.
