---
title: Dimensions as types
description: "Make a concrete gt4py.next dimension a type (`class I(gtx.Dimension)`) instead of an instance, so `Field[Dims[I], float64]` is a real annotation for plain mypy — and so that staggering, dimension variables and dependent local dimensions can land on top without revisiting the base."
author: havogt
tags: [type-system, dimensions, type-checking, mypy, mypy-plugin, pyright, nominal-types, metaclass, migration, serialization, frontend, foast, extension-point]
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
> `static_checks.py`, `static_errors.py`, `test_typed_dimensions.py`; mypy 2.3,
> no plugin). That document is the wider investigation this one is extracted
> from; everything beyond Part I stays there. The Part I surface has since also
> been cross-checked on pyright and against the shipped `gtx.Field`/`gtx.Dims`
> — see [Verification status](#verification-status).

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

`I = Dimension("I")` is an instance (`common.py:79`), so `Field[Dims[I], T]` is
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

    def __new__(cls, val: int) -> NamedIndex:   # `I(0)`, *not* instantiation
        return NamedIndex(cls, val)


class Dimension(DimensionBase): ...             # base of *user-declared* dimensions


class I(Dimension): ...
class K(Dimension, kind=DimensionKind.VERTICAL): ...
```

- **The class object is the value.** There is no instance level. Wherever a
  `Dimension` instance flows today — domains, offset providers, `FieldType.dims`,
  IR nodes — the class object flows instead.
- **`DimensionMeta` carries the value-level API**: `__add__`/`__sub__` building
  connectivities, comparison operators → `Domain`, `__repr__`, `__eq__`/`__hash__`,
  `__reduce__`. Binary operators on a class object dispatch through its metaclass,
  so this is the only place they can live.
- **`I(0)` → `NamedIndex` lives on `DimensionBase.__new__`, not on
  `DimensionMeta.__call__`.** mypy ignores a metaclass `__call__` and types `I(0)`
  as an instantiation of `I` (`Too many arguments for "I"  [call-arg]`), which
  would make this proposal *introduce* a false positive where none exists today.
  `__new__` may return any type, both checkers honour its return type, and
  `type.__call__` skips `__init__` because a `NamedIndex` is not an instance of
  `I` — so the same one line serves the static and the runtime view. This also
  supersedes the earlier "`__new__` raises" formulation: there is still no
  instance level, `__new__` simply never returns one.
- **Equality and hashing must include `kind`, not just `value`.** Today's
  `Dimension` is a frozen dataclass whose `__eq__` compares `value` *and* `kind`,
  with a dataclass-generated `__hash__` over both; `Dimension("I")` and
  `Dimension("I", VERTICAL)` are neither equal nor hash-equal. The vendored
  prototype's name-only `__eq__`/`__hash__` is therefore *not* semantics-preserving
  and must not be copied verbatim.
- **`DimensionMeta.__reduce__` is load-bearing, not a nicety.** `eve.utils.content_hash`
  pickles to fingerprint build-cache entries, and `otf/runners.py` ships compilation
  tasks to a `spawn`-based `ProcessPoolExecutor` through `pickle`. A class made by the
  `Dimension("I")` compatibility factory is dynamically created and therefore
  unpicklable (`PicklingError: attribute lookup ... failed`), and a statically declared
  one pickles *by reference*, which would make fingerprints depend on the defining
  module where today they are value-based. Reducing to `(Dimension, (value, kind))`
  restores both properties at once.

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

**Step 1 keeps the prefix and costs almost nothing.** `Staggered[I]` must
materialize a class whose `value` is `"_StaggeredI"`; because the metaclass
compares by name, the *reading* helpers keep working untouched — `is_staggered`,
`as_non_staggered`, `check_dims` and `order_dimensions` only ever look at
`.value`/`.kind`, and gtfn's `_add_staggered_aliases` rebuilds a `Dimension` from
a string. Two caveats, both established by reading the call sites:

- The vendored prototype sets `value = f"{base.value}½"`, not `"_StaggeredI"`.
  As vendored it would break `is_staggered`'s prefix test and, through it,
  `order_dimensions` and `check_dims`. Whichever way this is settled, the
  prototype and this document have to agree before Part II starts.
- `flip_staggered` is not a pure reader: it *constructs* `Dimension(value, kind)`.
  Under class-style dimensions it returns a factory-made class rather than
  `Staggered[I]`, which is fine only because equality is by name — one more
  reason the compatibility factory has to intern (see Migration, step 0).

**Step 2 is a separate change behind this surface.** Once staggering is answered
by the type, `is_staggered`'s `dim.value.startswith("_Staggered")` sniffing has no
callers and the encoding stops being a user-visible convention. Being precise
about what does *not* go away: distinct axes still need distinct identifiers at
the IR and backend boundary, so some mangling survives there. What disappears is
the string sniffing and the fact that users can observe — or collide with — the
prefix.

## Migration

Each step is landable separately, with name-based equality carrying the interop.
Sizes below are measured against the tree at the time of writing.

**Step 0 — settle identity semantics. Blocking, ~0 lines.** Static checkers see
dimension classes *nominally*; runtime equality is *structural* (name + kind).
That mismatch is load-bearing rather than academic: `tests/next_tests` declares a
dimension literally named `"I"` 46 times, and `IDim` in 45 distinct files, and
today's plugin hides this by keying its `_DimA`–`_DimD` substitution on the
*argument name*, so all 45 collapse to one placeholder. Afterwards mypy will
treat them as 45 distinct types. Recommended answer: keep **name + kind**
equality and make the `Dimension(value, kind)` compatibility factory **intern**,
so `Dimension("I") is Dimension("I")`. Do *not* add duplicate-name rejection here
— nothing rejects same-named dimensions today, so it is a new behaviour that
would break those 45 files at once; defer it to step 4 behind a warning.

**Step 1 — `DimensionMeta` and class-style `Dimension` in `common`.** Name-based
interop; `Dimension("I")` keeps working as an interning factory. Carries the
`__new__` → `NamedIndex`, `kind`-aware `__eq__`/`__hash__` and `__reduce__` points
from [Design](#design). One prerequisite outside `common`: `eve.type_validation`
has no `type[...]` case, so a `type[Dimension]` annotation currently degrades to
"is any class" — a `DataModel` field annotated `type[common.Dimension]` accepts
`int`. Adding that case is small and localized, and step 2 depends on it.
*Acceptance*: `nox -s test_next` green with no call-site changes outside `common`.

**Step 2 — the annotation and guard sweep. This is the bulk of the work.**
`Dimension` as an annotation means "an instance" and has to become
`type[Dimension]`; `isinstance(x, Dimension)` has to become
`isinstance(x, DimensionMeta)`. In `src/gt4py/next` alone that is ~109 annotation
sites and 41 `isinstance` guards, plus ~355 more occurrences in `tests/` and ~35
in `docs/`/`examples/`. Mechanical and greppable, but it is larger than steps 1
and 3 combined and should be its own PR. Two sites need specific attention rather
than a sweep:

- `type_system/type_specifications.py` — `DimensionType.dim`, `IndexType.dim`,
  `OffsetType.source`/`target`, `ListType.offset_type` and `FieldType.dims` are
  eve `DataModel` fields, so they are where the missing `type[...]` validator
  case actually bites.
- `ffront/fbuiltins.py` — `_type_conversion`'s `elif t is common.Dimension:
  return ts.DimensionType` maps builtin signatures and must learn
  `type[Dimension]`.

Until this step lands, the runtime guards fire immediately: `Domain(dims=(I,), ...)`
raises `ValueError: 'dims' argument needs to be a 'tuple[Dimension, ...]'`.

**Step 3 — migrate internal definitions and add typing coverage.** Tests, docs and
`fbuiltins` to class style. New `typing_tests` cases for dimension-mismatch
rejection, `TypeVar` bounds over dimensions, unparameterized `Dims`, and `I(0)` —
each run under **both** mypy and pyright, since pyright coverage is the new
capability this buys and nothing currently exercises it. Gate the plugin's
dimension hooks *off* rather than deleting them; downstream code mid-migration
still needs them, and the `fullname.endswith("Dim")` hook means deleting early
breaks any dimension that has not been converted yet.

**Step 4 — deprecate.** `Dimension("I")` warns, then the dimension half of
`mypy_plugin.py` is deleted. Duplicate-name rejection, if wanted, lands here.

## Open questions and risks

1. **The metaclass-overload mypy quirk — closed.** mypy rejects generic
   self-types on metaclass methods at the *definition* site but applies them
   correctly at every *call* site, so the operator overloads carry
   `# type: ignore[misc]`, pinned by `static_checks.py`. Re-verified on mypy 2.3
   (2026-08-05): both the quirk and the suppression still behave as described.
   pyright has now been checked too (1.1.411) and needs none of those
   suppressions — see [Verification status](#verification-status). The
   checker-agnostic fallback (free functions, `shift(I, 1)`) is **not** needed.
   What did turn up in its place is the metaclass `__call__` problem, now folded
   into [Design](#design).
2. **eve handling of dimension classes**, in two distinct places. *Validation*:
   `eve.type_validation` has no `type[...]` case and silently weakens
   `type[Dimension]` to "is any class" — a `DataModel` with a
   `dim: type[common.Dimension]` field accepts `int` today. *Serialization*:
   dataclass instances serialize structurally; classes need a `value`/`kind`-based
   representation, which `DimensionMeta.__reduce__` supplies. Note that
   `itir` nodes themselves do **not** embed `common.Dimension` — `AxisLiteral` is
   `(value: str, kind)` — so the affected surface is `type_specifications.py`,
   not the IR.
3. **Migration surface is the largest cost item**, and it is bigger than "move
   the dimension definitions": the annotation/guard sweep of step 2 is ~109
   annotation sites and 41 `isinstance` guards in `src/gt4py/next` alone, before
   tests and docs. Name-based interop bounds the *risk*, not the *size*, and it
   still needs a dedicated test layer exercising mixed old/new dimensions in one
   program.
4. **`typing` internals cache subscriptions by hash**, so hash-equal-but-distinct
   dimensions *do* alias — confirmed: for two distinct classes both named `I`,
   `Field[Dims[I]] is Field[Dims[I2]]` evaluates to `True`, and the annotation
   silently binds whichever was subscripted first. The earlier note that
   same-named distinct dimensions "are rejected anyway" is wrong: nothing in the
   tree rejects them, and `tests/next_tests` relies on that (see Migration,
   step 0). Under name+kind equality the aliasing is *consistent* with runtime
   semantics, but mypy will still see the two as distinct types, so the
   static and runtime views disagree exactly here.
5. **Dependent local dimensions stay open** and are expected to. The one question
   that reaches back here is whether dimension variables range over hop stacks
   ([[personal/havogt/dependent-local-dimensions|dependent local dimensions]]
   §8.2); the v0 answer — dimension variables bind origin and non-local dims only
   — leaves this proposal unaffected either way.

## Verification status

Checked against gt4py `main` (`ee1bb4f6a`) on 2026-08-26, mypy 2.3.0 and pyright
1.1.411.

- **The premise reproduces on the shipped types**, not only on the prototype's
  own `Field`/`Dims`. With no plugin, `IDim = gtx.Dimension("IDim")` in
  `gtx.Field[gtx.Dims[IDim], gtx.float64]` gives
  `Variable "IDim" is not valid as a type  [valid-type]`, while
  `class IDim(Dimension)` type-checks *and* rejects passing a
  `Field[Dims[JDim], float64]` where `Field[Dims[IDim], float64]` is expected.
- **The vendored prototype still passes mypy 2.3.0** with its own `mypy.ini`
  (`Success: no issues found in 2 source files`).
- **pyright accepts the whole Part I surface with zero diagnostics** — metaclass
  `__repr__`/`__hash__`/`__add__`/`__sub__`/comparisons with `cls: type[AnyD]`
  generic self-types, `__init_subclass__`, `Dims[...]`, and TypeVars ranging over
  dimensions. It needs none of mypy's `# type: ignore[misc]` suppressions.
  pyright *does* report 8 errors on the vendored prototype, but all 8 are
  `dual()`, which is Part II and out of scope here.
- **`I(0)` via a metaclass `__call__` fails on mypy** (`Too many arguments for
  "I"  [call-arg]`, revealed type `I`) while pyright resolves it correctly;
  moving it to `DimensionBase.__new__` satisfies mypy, pyright and the runtime.
- **A factory-made dimension class is unpicklable** as-is, and
  `type[Dimension]` passes eve `DataModel` validation for any class at all.

Not yet checked: behaviour of the sweep in step 2 under an actual migration, and
whether ICON4Py has dimension usages outside the patterns present in this tree.

## Related

- [[personal/havogt/dimension-generic-fields|Generic dimensions and statically
  typed staggering]] — the investigation this is extracted from; keeps staggering
  (Part II) and dimension variables (Part III).
- [[personal/havogt/dependent-local-dimensions|Dependent local dimensions and
  connectivity chains]] — the second consumer of the parameterized-dimension slot.
- [[personal/havogt/dtype-generic-fields|Dtype-generic fields in gt4py.next]] —
  shares the binding/monomorphization machinery, and is the other half of what
  the mypy plugin currently blurs.
