---
title: Dimensions as types
description: "Make a concrete gt4py.next dimension a type (`class I(gtx.Dimension)`) instead of an instance, so `Field[Dims[I], float64]` is a real annotation for plain mypy — and so that staggering, dimension variables and dependent local dimensions can land on top without revisiting the base."
author: havogt
tags: [type-system, dimensions, type-checking, mypy, mypy-plugin, nominal-types, metaclass, migration, frontend, foast, extension-point]
created: 2026-08-05
status: proposed
---

> **TL;DR** A concrete dimension is currently an *instance*
> (`I = Dimension("I")`), so `Field[Dims[I], float64]` is not a valid static
> annotation and gt4py ships a mypy plugin to paper over it. Make a dimension a
> *class* (`class I(gtx.Dimension): ...`) whose class object carries today's
> value-level behaviour via a metaclass. That is the entire proposal — but the
> shape is chosen so that staggering, dimension variables and dependent local
> dimensions all land on top of it without coming back to change it.

> **Prototype**: mypy-verified, vendored under
> [[personal/havogt/dimension-generic-fields|Generic dimensions and statically
> typed staggering]] (`dimension-generic-fields/typed_dimensions.py`,
> `static_checks.py`, `static_errors.py`, `test_typed_dimensions.py`; mypy 1.19,
> no plugin). That document is the wider investigation this one is extracted
> from; everything beyond Part I stays there.

## Scope

| In | Out |
| --- | --- |
| A concrete dimension is a type | Staggering syntax and index conventions |
| Value-level behaviour preserved on the class object | Dimension variables / dim-generic operators |
| Old- and new-style dimensions coexist during migration | Dependent local dimensions and connectivity chains |
| Dimension types usable as `TypeVar` bounds | Anything about `Dims` ordering semantics |

The out-column items each have their own proposal. This one is deliberately the
smallest thing that unblocks all of them.

## Problem

`I = Dimension("I")` is an instance (`common.py:95`), so `Field[Dims[I], T]` is
not a valid annotation — mypy reports `Variable "I" is not valid as a type
[valid-type]`. That is GridTools/gt4py#2503, and the shipped answer is a
compatibility shim: `src/gt4py/next/type_system/mypy_plugin.py` (wired in via
`pyproject.toml` `[tool.mypy] plugins = [...]`), which substitutes placeholder
classes `_DimA`–`_DimD` and then `_AnyDim` for dimension arguments.

What that costs:

- At most **four distinct dimensions** are tracked per run, *globally*, keyed by
  argument name — so same-named dimensions in different modules collide and
  everything past the fourth becomes `_AnyDim`.
- **No `TypeVar`s over dimensions**, so no dimension-generic operator can ever be
  typed.
- **Every downstream project must install the plugin**, and only mypy is served;
  pyright users get nothing.
- The plugin also *blurs dtypes* (`float` vs `float32` vs `float64`) to suppress
  a different class of false positive, which erases exactly the information
  dtype generics need.

**Success criterion**: with this proposal landed, the dimension half of
`mypy_plugin.py` is deleted rather than maintained, and client code type-checks
without a gt4py-specific plugin.

## Design

```python
class DimensionBase(metaclass=DimensionMeta):   # root: anything usable in Dims[...]
    value: ClassVar[str]                        # defaults to the class name
    kind: ClassVar[DimensionKind] = DimensionKind.HORIZONTAL


class Dimension(DimensionBase): ...             # base of *user-declared* dimensions


class I(Dimension): ...
class K(Dimension, kind=DimensionKind.VERTICAL): ...
```

- **The class object is the value.** There is no instance level (`__new__`
  raises). Wherever a `Dimension` instance flows today — domains, offset
  providers, `FieldType.dims`, IR nodes — the class object flows instead.
- **`DimensionMeta` carries the value-level API**: `__add__`/`__sub__` building
  connectivities, `__call__` → `NamedIndex`, comparison operators → `Domain`,
  `__repr__`. Binary operators on a class object dispatch through its metaclass,
  so this is the only place they can live.
- **Name-based `__eq__`/`__hash__` on the metaclass** reproduces today's
  `Dimension.__eq__` (which compares `.value` only) and makes a new-style class
  compare and hash equal to an old-style instance of the same name. This is the
  migration mechanism: `Domain` dicts, offset-provider lookups and
  `FieldType.dims` comparisons work across both styles, so subsystems — and
  downstream code such as icon4py — migrate dimension by dimension.

## The extension point

This is the part that makes the proposal worth landing on its own.

The two-level split — a root `DimensionBase` (anything usable inside `Dims[...]`)
above `Dimension` (the base of user-declared dimensions) — exists so that
**parameterized dimension types** can be usable in `Dims[...]` without being
user-declarable. Both known follow-ups need exactly that, and neither is decided
here:

- **`Staggered[D]`** — a dimension parameterized by a dimension. Binding its
  parameter to `Dimension` rather than the root is what makes
  `Staggered[Staggered[I]]` statically unrepresentable.
- **`Local[C]`** — a local dimension parameterized by the *connectivity* it comes
  from, so `V2EDim`'s dependence on `Vertex` lives in the type
  ([[personal/havogt/dependent-local-dimensions|Dependent local dimensions and
  connectivity chains]], §4.4/§5.4 — both designs are mypy-verified on top of
  this base).

Dimension variables (`TypeVar("D", bound=Dimension)`, `TypeVarTuple` over dims)
need only that dimensions *are* types, which is the proposal itself.

So the forward-compatibility claim is narrow and checkable: **the root admits
parameterized dimension types, and both candidate users of that slot have
working prototypes.** Nothing else about them is settled here.

## Relation to the shipped `_Staggered` name prefix

gt4py already ships staggered dimensions (`feat[next]: staggered fields`, #2667,
ADR 0026), encoding "staggered" as a `_Staggered` prefix on the dimension
*name*. That ADR's rationale is sound *given today's model*: a dimension is
identified by its name and appears as a `common.Dimension`, as an
`itir.AxisLiteral`, and as a plain string in generated backend code, so a flag on
`Dimension` would have to be threaded through all three.

Dimensions-as-types changes that premise — identity becomes nominal, and the name
becomes what crosses the IR boundary rather than what *is* the dimension.
(`itir.AxisLiteral` is `(value: str, kind)` and already carries
`TODO(havogt): Refactor to use declare Axis/Dimension at the Program level`.)

**Step 1 keeps the prefix and costs nothing.** `Staggered[I]` materializes a class
whose `value` is `"_StaggeredI"`; because the metaclass compares by name, every
existing staggering helper keeps working untouched — `is_staggered`,
`flip_staggered`, `as_non_staggered`, `check_dims` and `order_dimensions` read
only `.value`/`.kind`, and gtfn's `_add_staggered_aliases` rebuilds a `Dimension`
from a string. (Established by reading those call sites, not by running a
migration.)

**Step 2 is a separate change behind this surface.** Once staggering is answered
by the type, `is_staggered`'s `dim.value.startswith("_Staggered")` sniffing has no
callers and the encoding stops being a user-visible convention. Being precise
about what does *not* go away: distinct axes still need distinct identifiers at
the IR and backend boundary, so some mangling survives there. What disappears is
the string sniffing and the fact that users can observe — or collide with — the
prefix.

## Migration

Each step is landable separately, with name-based equality carrying the interop:

1. Land `DimensionMeta` / class-style `Dimension` in `common`, with name-based
   interop; keep `Dimension("I")` working as a factory.
2. Move gt4py-internal dimension definitions (tests, docs, `fbuiltins`) to class
   style; add `typing_tests` coverage; drop the `_DimA`–`_DimD` machinery from
   the mypy plugin for class-style dimensions.
3. Deprecate instance-style definitions; `Dimension("I")` eventually warns.

## Open questions and risks

1. **The metaclass-overload mypy quirk.** mypy rejects generic self-types on
   metaclass methods at the *definition* site but applies them correctly at every
   *call* site, so the operator overloads carry `# type: ignore[misc]`, pinned by
   `static_checks.py`. **pyright is untested and must be checked before
   committing** — the checker-agnostic fallback is free functions (`shift(I, 1)`),
   which changes notation but not the design.
2. **eve serialization of dimension classes** inside IR nodes: dataclass
   instances serialize structurally today; classes need a `value`/`kind`-based
   representation. Straightforward, since both are class attributes — but it is
   real work and touches every IR node embedding a `common.Dimension`.
3. **Migration surface is the largest cost item.** Name-based interop bounds the
   risk but needs a dedicated test layer exercising mixed old/new dimensions in
   one program.
4. **`typing` internals cache subscriptions by hash**, so hash-equal-but-distinct
   dimensions could alias. Only reachable for same-named distinct dimensions,
   which are rejected anyway.
5. **Dependent local dimensions stay open** and are expected to. The one question
   that reaches back here is whether dimension variables range over hop stacks
   ([[personal/havogt/dependent-local-dimensions|dependent local dimensions]]
   §8.2); the v0 answer — dimension variables bind origin and non-local dims only
   — leaves this proposal unaffected either way.

## Related

- [[personal/havogt/dimension-generic-fields|Generic dimensions and statically
  typed staggering]] — the investigation this is extracted from; keeps staggering
  (Part II) and dimension variables (Part III).
- [[personal/havogt/dependent-local-dimensions|Dependent local dimensions and
  connectivity chains]] — the second consumer of the parameterized-dimension slot.
- [[personal/havogt/dtype-generic-fields|Dtype-generic fields in gt4py.next]] —
  shares the binding/monomorphization machinery, and is the other half of what
  the mypy plugin currently blurs.
