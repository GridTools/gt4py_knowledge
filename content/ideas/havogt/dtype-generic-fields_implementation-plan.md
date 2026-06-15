---
title: "Dtype-generic fields: implementation roadmap (PR series)"
author: havogt
tags: [type-system, generics, dtype, frontend, foast, past, monomorphization, planning, roadmap]
created: 2026-06-15
---

> **Appendix** to [[ideas/havogt/dtype-generic-fields|Dtype-generic fields in gt4py.next]].
> The proposal is the *design*; this is the *implementation roadmap* — the ordered
> series of PRs that lands it, for reviewers to follow. It is **not** the ADR
> (which records the decision) and lives outside the gt4py source tree.

## Goal (recap)

Let field operators be generic in their field **dtype**, spelled with a native
value-constrained `TypeVar` so the same annotation is meaningful to mypy and to
the DSL frontend; specialize per concrete signature (monomorphize) at call time.

```python
FloatT = typing.TypeVar("FloatT", gtx.float32, gtx.float64)

@gtx.field_operator
def diffusion(a: gtx.Field[gtx.Dims[I, J], FloatT],
              b: gtx.Field[gtx.Dims[I, J], FloatT]) -> gtx.Field[gtx.Dims[I, J], FloatT]:
    return a - b

diffusion(a32, b32, out=o32, offset_provider={})  # compiles a float32 variant
diffusion(a64, b64, out=o64, offset_provider={})  # compiles a float64 variant
diffusion(a32, b64, out=o64, offset_provider={})  # error: inconsistent binding of FloatT
```

A working end-to-end prototype of Stages 0+1 exists on branch
`claude/gt4py-generics-investigation-aycs14`; the PRs below land it cleanly (and
extend it to Stage 2, which the prototype does not cover).

## The PR series

PRs are opened **staggered** (one after the next); each builds on the previous.

| # | PR | Stage | What it does | Status |
|---|----|-------|--------------|--------|
| 1 | `refactor[next]: normalize dtype predicates` | pre-0 | Route the field-aware dtype predicates (`is_floating_point`/`is_integral`/`is_logical`) through one helper; fix a latent non-f-string error message. Inert, behavior-preserving (pinned by doctests). | **implemented** |
| 2 | `feat[next]: dtype-generic type system (Stage 0)` | 0 | `ts.TypeVarType`; `type_info.is_generic`; `bind_type_vars`/`substitute_type_vars`; TypeVar-aware predicates/`promote`; rewire `CompiledProgramsPool._is_generic`. Lands **ADR 0023**. Inert — nothing produces a `TypeVarType` yet. | **implemented** |
| 3 | `feat[next]: generic field operators (Stage 1)` | 1 | `from_type_hint` TypeVar translation; FOAST type-checking of generic bodies + guard rails; the `foast_specialize` monomorphization step; **direct call + embedded execution**. A single generic operator runs end-to-end (verified across the full backend matrix incl. GPU). | **implemented** |
| 4 | `feat[next]: generic operators in programs (Stage 2)` | 2 | Generic operator **called from a `@program`** (incl. several dtypes in one program): bind-and-substitute the fieldop signature *before* the existing checks (so zero-dim promotion only ever sees concrete types — this also obviates a separately-planned `promote_zero_dims` change), a PAST monomorphization pass (`__gt_specialize__` + name mangling), wired in `past_to_itir`. Verified across the matrix incl. GPU. | **implemented** |

Follow-ups beyond the series (each independently landable): generic operator
called from **another operator** (FOAST-level specialization); eager
`.compile()`-all-constraints; per-builtin TypeVar audits (`astype(x, T)`,
`where`/`concat_where`, `broadcast`, reductions).

> Note: the originally-planned standalone "binding-aware `promote_zero_dims`" PR
> dissolved during implementation — substitute-first in the Stage 2 fieldop checks
> means `promote_zero_dims` never sees a type variable, so there is nothing to
> change there.

## Why the Stage 1 / Stage 2 boundary is where it is

Stage 1 deliberately stops at "a **single** generic operator works (direct call +
embedded)" and **rejects** calling a generic operator from a `@program` or from
another operator (clean decoration-time error). Two reasons, both structural:

- **Embedded needs no wiring.** `FieldOperator.__call__`'s embedded branch runs
  the raw Python function — NumPy ignores the `TypeVar` annotations — so a generic
  operator that type-checks at decoration time already executes correctly in
  embedded mode. There is no "type-checks but can't run" half-state.
- **Type-deduction is decoration-time for every operator.** The TypeVar-aware
  FOAST deduction must succeed at *import* (errors-at-decoration UX), independent
  of backend. It cannot be split from the feature.

So the genuinely clean cut is single-operator (Stage 1) vs operator-in-program
(Stage 2), not "type-check vs wire". Stage 2 is the larger design risk (no
prototype) and gets its own PR.

## Out of scope / deferred (so reviewers don't flag them as gaps)

- **Generic scan operators** — rejected with a clear message in Stage 1 (needs
  `init: T` coercion semantics).
- **`bound=` TypeVars** — Stage 1 accepts only *value-constrained* TypeVars
  (finite variant set ⇒ eager precompilation possible).
- **Strict no-promotion** — `a * 2.0` on a generic `a` is a decoration-time error
  (D3); `astype(x, T)` is the designated remediation, a fast-follow.
- **Dimension variables / generic dims** — separate effort
  ([[ideas/havogt/dimension-generic-fields|generic dimensions]]); the shared
  binding utilities stay `dtype`-scoped here and are widened there.
- **mypy-plugin un-blurring** of `float32`/`float64` — follow-up.

See [[ideas/havogt/dtype-generic-fields|the proposal]] §7/§8 for the full risk and
out-of-scope discussion, and ADR 0023 (once it lands) for the recorded decision.
