---
title: Closure variable resolution in gt4py.next
author: havogt
tags: [frontend, foast, past, closure-variables, name-resolution, constants, builtins, aliasing, gtir, lowering]
created: 2026-06-12
status: draft
---

> **TL;DR** Make closure-variable references in the `gt4py.next` frontend
> resolve **by value, not by source-level name**, so that module-prefixed
> builtins (`gtx.where`), aliased operators (`helper as h`), module/`Final`
> scalar constants (`np.pi`), and re-exports all work — by resolving every
> reference to its Python value right after parsing and canonicalizing the AST.
> Errors move from late, internal stages to decoration time.

> **Prototype**: a *partial* prototype of strategy **A** below (the
> resolve-early / canonicalize approach) lives on the gt4py branch
> [`claude/elegant-cannon-13ew06`](https://github.com/graitools/gt4py/tree/claude/elegant-cannon-13ew06),
> together with a proposed ADR (`docs/development/ADRs/next/0023-Closure_Variable_Resolution.md`)
> that catalogs the limitations and compares strategies in full. This document
> distills that ADR into a knowledge-base idea; consult the branch for the
> detailed case table and the prototype code.

---

## Problem / motivation

Field operators and programs are Python functions that may refer to names which
are neither parameters nor locals — *closure variables*. The frontend captures
them with `inspect.getclosurevars()` into a flat `dict[str, Any]` keyed by the
**source-level name** (`ffront.source_utils.get_closure_vars_from_function`).

The crucial — and fragile — property of the current design is that **identity in
all later stages is the source-level name**:

- Builtin calls are recognized by comparing `foast.Name.id` against
  `fbuiltins.*_BUILTIN_NAMES` (in `foast_passes.type_deduction` and
  `foast_to_gtir`).
- Calls to other operators lower to `SymRef(<name used at the call site>)`,
  while the callee registers its IR function definition under its original `def`
  name (`GTCallable.__gt_gtir__()`), linked at program level in `past_to_itir`.
- Offsets lower to `OffsetLiteral(<call-site name>)` and are looked up by that
  name in the `offset_provider` at run time.
- All closure variables of all transitively referenced operators are merged into
  one flat namespace (`transform_utils._get_closure_vars_recursively`), which
  raises if the same name binds to different values anywhere in the call tree.

Any reference whose source-level name differs from the "canonical" name
therefore breaks — each case in a different stage, with a different (often
internal) error. Only `FrozenNamespace`/`enum` attributes were resolved by value
today (the former `ClosureVarFolding` pass).

A second, orthogonal problem is *when* values are captured: at decoration time.
Embedded execution instead runs the original Python function and sees the *live*
values, so closing over a mutable binding has **diverging semantics** between
embedded and compiled execution (or simply fails in compiled mode for scalars).

### Symptoms (verified on `main`, 2026-06)

References that **should** be part of the model but fail today, with how late /
how obscure the error is:

| Case | Behavior on `main` | Stage |
| --- | --- | --- |
| Module-prefixed builtin `gtx.where(...)` | `DSLError: Functions can only be called directly.` | FOAST type deduction |
| Aliased builtin `my_where = where` | `AssertionError: ... not fully typed.` (internal) | FOAST type deduction |
| Module-prefixed operator `helpers.helper(a)` | `DSLError: Functions can only be called directly.` | FOAST type deduction |
| Aliased operator `helper as h2; h2(a)` | `EveValueError: Symbols {SymbolRef('h2')} not found.` (internal) | GTIR validation |
| Same operator referenced under two names | `EveValueError: Multiple definitions of symbol 'helper'` (internal) | GTIR validation |
| Scalar from a module attribute `np.pi` | `AssertionError: Unreachable` (internal) | FOAST→GTIR lowering |
| Module-level scalar marked `typing.Final` | `EveValueError: Symbols {SymbolRef('CONST')} not found.` (internal) | GTIR validation |

By contrast, `Dimension` via a module prefix and `FrozenNamespace`/`enum`
attribute constants already work. The full case table (incl. the
should-be-*rejected* cases that today produce internal errors instead of clear
decoration-time ones — mutable globals, callables in containers, mutated
captured bindings) is on the branch.

## Proposal

### The model: what may be captured

A closure reference must be a **constant reference expression** — a name, or a
chain of attribute accesses rooted at a name — that resolves at decoration time
to a value of a supported kind:

- a GT4Py builtin (function, type constructor),
- a `GTCallable` (field operator, scan operator, …),
- a `Dimension` or `FieldOffset`,
- a namespace used as a prefix (module, `FrozenNamespace`, `enum` class),
- a scalar **constant**, where "constant" means the binding is immutable or
  promised to be: a `FrozenNamespace`/`enum` attribute, a module attribute, or a
  module-level global annotated `typing.Final`.

**Identity is by value, not by name**: aliases, re-exports and module-prefixed
references of the same value are interchangeable. Mutable bindings (plain
globals, nonlocals) and mutable containers are out of the model; data values
(fields, arrays) are not capturable (see the stretch goal).

### Strategy A — resolve names to values at decoration, canonicalize (chosen)

Keep the existing capture-at-decoration-time pipeline, but immediately after
parsing resolve every closure reference (including attribute chains) to its
Python *value* and canonicalize the AST:

- constants → folded into `Constant` nodes,
- builtins → rewritten to the canonical builtin name,
- attribute-derived references to operators/dimensions/offsets → rewritten to a
  synthesized, value-derived name registered as an extra closure variable,
- direct references (incl. aliases) keep their source name; the program lowering
  registers IR function definitions under the **reference name** instead of the
  definition name — aliases then work without rewriting, and user-facing names
  stay in IR and error messages.

**Pros**: small, local change (two frontend passes + the function registration
in `past_to_itir`); fixes all catalogued cases; errors move to decoration time
with source locations; no change to stages, caching, embedded execution, or
backends; IR stays readable.

**Cons**: keeps snapshot-at-decoration semantics (mitigated by the `Final`
rule); constants are baked into the AST; the flat merged namespace remains, so
two *directly* referenced operators with the same source name but different
values still collide (aliasing is the documented workaround); the AST is no
longer purely syntactic.

### Why A now, B later

**Strategy A now, with [Strategy B](#alternatives-considered) as the evolution
path.** A resolves every catalogued limitation except the no-alias same-name
operator collision, is reviewable incrementally, and does not paint us into a
corner: the canonicalization pass is exactly where B's opaque reference ids
would later be introduced, and the key-based function registration in
`past_to_itir` is the seed of B's link step.

## Alternatives considered

- **Strategy B — late binding via an explicit closure environment ("linker").**
  References in FOAST/PAST point to opaque, unique reference ids; a
  `ClosureEnvironment` maps ids to values; binding happens at program-assembly
  time (function definitions get final names, mangled only on real collision;
  `SymRef`s remapped; constants substituted at lowering, not into the AST).
  *Pros*: fully value-based identity (even the same-source-name collision
  disappears); AST stays syntactic; enables re-binding (`with_closure(...)`),
  staleness checks, and is the natural foundation for capturing fields. *Cons*: a
  cross-cutting refactor of every stage that assumes the flat name-keyed dict;
  higher migration cost; harder to land incrementally. **The intended evolution
  of A.**
- **Strategy C — minimal name-mangling patches** (dedup identical defs, rename
  `GTCallable` closure vars when unambiguous, fold module attributes). Smallest
  diff, but does not fix the underlying name-identity problem and piles up
  special cases that A subsumes.
- **Strategy D — trace-based capture (à la JAX).** Closures resolve for free via
  ordinary Python semantics, but it is a different frontend architecture
  deliberately *not* chosen for GT4Py (ADR 0001 — Field View Frontend Design):
  tracing loses source locations and type annotations and restricts control
  flow. Out of scope.

## Open questions / conflicts

- **Same-name collision without aliases**: two *directly* referenced same-named
  operators still raise under A; the message now suggests aliasing as the
  workaround. Fully solving it requires B's link step.
- **`typing.Final` for nonlocals** cannot be verified (function-local
  annotations are not retained at runtime), so the factory-function pattern
  needs a `FrozenNamespace` or a future explicit opt-in (e.g.
  `gtx.constant(...)`).
- **Local shadowing** of a canonical builtin name (a local `where` together with
  `gtx.where`) is detected only when the local is visible in the symbol table.
- **pyright / non-mypy checkers** and serialization concerns are unaffected by A
  (it touches only frontend passes), but B would intersect with stage
  fingerprinting and `GTCallable.__gt_closure_vars__`.
- No direct conflict with the other ideas here, but it shares the frontend
  binding/lowering surface with
  [[personal/havogt/dtype-generic-fields|Dtype-generic fields in gt4py.next]] and
  [[personal/havogt/dimension-generic-fields|Generic dimensions and statically typed staggering]]
  (the `foast_specialize`/monomorphization machinery and the closure-var
  fingerprint that compiled-program caching keys on).

## Appendices

### Stretch goal: closing over read-only fields

Capturing field *data* differs in kind: the value cannot be folded into the IR,
it must reach the backend at run time. A clean design answers three questions —
**marking** (explicit, e.g. `typing.Final[gtx.Field[...]]` or a
`gtx.constant_field(field)` wrapper that can freeze/copy-on-capture and expose
`__gt_type__`); **mechanism** (a captured read-only field becomes a *hidden
parameter* of the operator via a `GTCallable.__gt_implicit_args__()`, reusing
the entire existing argument machinery; embedded execution needs no change);
and **binding time / caching** (bind the *binding* not the buffer, so the
compiled cache key includes only the field's type/domain descriptor — zero
recompilation cost). A separate opt-in (`gtx.constant_field(field, freeze=True)`)
could bake the buffer in as a true constant, layered on `StaticArg`-style
descriptors (ADR 0021 — Argument Descriptors). This fits A (the canonicalization
pass introduces the hidden parameter) and becomes more natural under B.

### Source material

- Proposed ADR and **partial prototype of strategy A** on gt4py branch
  [`claude/elegant-cannon-13ew06`](https://github.com/graitools/gt4py/tree/claude/elegant-cannon-13ew06)
  (`docs/development/ADRs/next/0023-Closure_Variable_Resolution.md`). The
  prototype adds `foast_passes.closure_var_resolution.ClosureVarResolution`
  (replacing `ClosureVarFolding`), a PAST counterpart
  `past_passes.closure_var_canonicalization.ClosureVarCanonicalization`,
  reference-name function registration in `past_to_itir.past_to_gtir`, a
  data-typed-closure-var rejection in `foast_to_gtir`, and
  `source_utils.get_final_closure_var_names` for `typing.Final` detection.
- ADR 0001 — Field View Frontend Design (why not strategy D).
- ADR 0021 — Argument Descriptors (the `freeze=True` field-capture variant).
