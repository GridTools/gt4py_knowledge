---
title: Bounds checking via domain containment in gt4py.next
author: havogt
tags: [bounds-checking, domain-inference, trace-shifts, out-of-bounds, validation, concat_where, unstructured, neighbor-tables, compiled-backends, gtfn, dace, embedded, type-system, staggering, static-analysis, runtime-checks, halos, errors]
created: 2026-06-16
---

> **TL;DR** Domain inference already back-propagates the output domain `D_out`
> through every accessor's shifts to compute, for each input field `f`, the
> domain `D_read(f)` it is actually read at — and then **throws that result
> away** at the program boundary. Keep it, and two features fall out of the same
> data: (1) a **domain-containment check** `D_read(f) ⊆ D_provided(f)` that turns
> today's silent out-of-bounds buffer reads on the compiled backends (`gtfn`,
> `dace`) into a deterministic `OutOfBoundsError` at first call, with no
> per-element overhead; and (2) a **public API to query the required input
> domains** of a program/field-operator. The check is a *spectrum*, not one
> mechanism — type-level guard (earliest), static containment at first call,
> symbolic/runtime containment, per-element sanitizer (last resort) — and which
> layer fires depends on how much is known statically. This proposal maps the
> spectrum onto the existing machinery, embeds the in/out-field check of
> [GridTools/gt4py#2182](https://github.com/GridTools/gt4py/pull/2182) as its
> first sibling, and lists the refactors that should land first.

> **Status**: draft proposal / design exploration. Drafted with AI assistance
> from a deep read of the `gt4py.next` toolchain. Not implemented; no prototype
> yet. The motivating bug is real (reduced below).

> **Related (in this repo)**:
> - [[ideas/havogt/dimension-generic-fields|Generic dimensions and statically typed staggering]]
>   — the **type-level guard** (Layer 0 below): typing a half-level field on
>   `Staggered[K]` / a distinct `KHalfDim` makes "full-level field used where a
>   half-level extent is required" a *type* error, before any domain is known.
>   This is the pending "type `z_mc`/`z_ifc`/`ddqz` on `KHalfDim`" cleanup the
>   feature request mentions.
> - [[ideas/havogt/mesh-and-first-class-halos|A mesh concept with first-class halos]]
>   — the **forward dual**: `valid(out) = min over inputs (valid(in) ⊖ footprint)`
>   is the same arithmetic run forwards for automatic halo-exchange placement;
>   composed-connectivity *reach* replaces reading raw neighbor tables, and the
>   "skip values split" there is exactly the false-positive hazard for the
>   unstructured containment check here.
> - [[ideas/havogt/field-data-protocol|A FieldData protocol for gt4py.next embedded fields]]
>   — why **embedded** is harmless today (domain-tracked, intersection
>   semantics) and how infinite-domain `concat_where` results (the boundary
>   branch) must be checked piece-wise rather than against a naive union.
> - [[ideas/havogt/boundary-condition-syntax|Frontend syntax sugar for boundary conditions over concat_where]]
>   and [[ideas/havogt/scan-redesign|Redesign of the vertical scan]] — the
>   `concat_where` boundary idioms that produce the read-domain over-/under-shoot
>   this check both relies on and must not false-positive on.

> **Upstream references**:
> [PR #2182 — Check inout field](https://github.com/GridTools/gt4py/pull/2182)
> (the first `trace_shifts`-based validation pass; embedded below),
> [issue #2205 — domain inference with concat_where at domain boundaries](https://github.com/GridTools/gt4py/issues/2205)
> (inference over-approximation; bounds the check's precision),
> [ADR 20 — Runtime domains](https://github.com/GridTools/gt4py/blob/main/docs/development/ADRs/next/0020-Runtime-domains.md)
> (`get_domain_range`), [ADR 21 — Argument Descriptors](https://github.com/GridTools/gt4py/blob/main/docs/development/ADRs/next/0021-Argument-Descriptors.md)
> (`FieldDomainDescriptor`, the first-call concrete-domain hook),
> [ADR 22 — Limitations of embedded concat_where](https://github.com/GridTools/gt4py/blob/main/docs/development/ADRs/next/0022-Limitations-of-embedded-concat_where.md).

---

# Bounds checking via domain containment

## Problem / motivation

A `gt4py.next` `field_operator`/`program` computes an output on an iteration
domain `D_out`. Domain inference
(`src/gt4py/next/iterator/transforms/infer_domain.py`) already back-propagates
`D_out` through each accessor's offsets to obtain the required read domain
`D_read(f)` for every input field `f` — that is how it sets the per-`as_fieldop`
domains the compiled backends consume. What is missing is a **containment
check**: for every input `f`, verify

```
D_read(f) ⊆ D_provided(f)         # D_provided(f) = f's actual origin .. origin+shape
```

and, if it fails, raise a clear `OutOfBoundsError` instead of emitting a kernel
that indexes past the buffer.

### The bug, reduced

```python
from gt4py import next as gtx
from gt4py.next import Dims, Field
from gt4py.next.experimental import concat_where

KDim = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)

@gtx.field_operator
def f(z_mc: Field[Dims[KDim], float], nlev: gtx.int32) -> Field[Dims[KDim], float]:
    # false branch (k >= nlev) reads bare z_mc; at k == nlev that is z_mc[nlev]
    return concat_where(KDim < nlev, 0.0, z_mc)

# call:  out  domain [0, nlev+1)  (half-level)   -> D_out          = [0, nlev+1)
#        z_mc domain [0, nlev)    (full-level)   -> D_provided(z_mc) = [0, nlev)
#
# inference: z_mc (offset 0) is live on the false region {nlev}
#   -> D_read(z_mc) ∋ nlev   but   nlev ∉ [0, nlev)   ==>  OUT OF BOUNDS
```

- **embedded** (`backend=None`): harmless **but silent**. Embedded fields carry
  their domain and use intersection semantics
  (`embedded/nd_array_field.py`, `embedded/common.py:domain_intersection`), so
  `z_mc` is simply never indexed at `nlev`; the false region `{nlev}` intersects
  `[0, nlev)` to empty. No crash — but also no signal that the requested output
  point had no input, so the result silently differs from what the compiled path
  would attempt.
- **gtfn / dace**: emit `z_mc[nlev]`, a raw read past the allocation. The
  lowering carries no guard — gtfn lowers the access to a bare `deref`, and the
  dace lowering computes the memlet subset as `domain_ranges[dim][0] - origin`
  with no containment assertion. Silent at small sizes; on GPU at large sizes it
  crosses an unmapped page → `cudaErrorIllegalAddress`.

The goal is to make this a deterministic, well-located error on **every** backend
— a static version of the safety embedded already provides.

### The key observation: the data already exists and is discarded

`infer_domain.infer_expr` returns `(transformed_expr, accessed_domains)`, where
`accessed_domains: AccessedDomains = dict[str, DomainAccess]`
(`infer_domain.py:53`, `:437`) maps **each input symbol** to the
`SymbolicDomain` it is read at — or to `DomainAccessDescriptor.UNKNOWN` (dynamic
shift, extent unknown) or `DomainAccessDescriptor.NEVER` (`infer_domain.py:31-53`).
This is exactly `D_read(f)`. But `infer_program` (`infer_domain.py:556`) discards
it: it threads the per-statement result only to attach domains to `as_fieldop`
nodes (`expr.annex.domain`, `:517`) and returns the transformed program — the
program-boundary mapping `param → D_read` is computed and thrown away
(`_infer_stmt` at `:540` ignores the returned dict).

So **both** features the colleague asks for are the same datum surfaced at two
boundaries:

| Feature | What it is |
| --- | --- |
| **Bounds check** | assert `D_read(f) ⊆ D_provided(f)` for each program input `f` |
| **Query API** | return `{f: D_read(f)}` to the user |

Step zero for either is: **stop discarding `AccessedDomains` at the program
boundary.**

## How `D_read` is computed today (the engine both features ride on)

`_extract_accessed_domains` (`infer_domain.py:124`) drives it:

1. `trace_shifts.trace_stencil(stencil, num_args)`
   (`transforms/trace_shifts.py:332`) returns, per argument, the **set of shift
   chains** that reach a `deref` of that argument — e.g.
   `{(), (⟪Koff, 1⟫,), (⟪E2V, 0⟫,)}`. This is the "shift chain from trace
   shifts" the request names. A dynamic shift becomes `Sentinel.VALUE`.
2. For each chain, `SymbolicDomain.translate(D_out, shift, offset_provider, …)`
   (`ir_utils/domain_utils.py:163`) maps the output domain to the read domain:
   - **Cartesian** offset `(Koff, 1)`: pure symbolic shift of the range
     start/stop by the integer (`SymbolicRange.translate`, `domain_utils.py:43`).
     Works even when bounds are symbolic.
   - **Unstructured** offset `(E2V, n)`:
     `_unstructured_translate_range_statically` (`domain_utils.py:61`) **reads the
     actual neighbor table** (`connectivity.ndarray[start:stop, nb_index]`) and
     returns the contiguous hull `[accessed.min(), accessed.max()+1)`. This is
     the **static extent analysis via the neighbor table**, and it already does
     two relevant things: it **warns on skip-value (out-of-bounds sentinel)
     hits** (`domain_utils.py:92-101`, with a `TODO` to *"turn this into a
     configurable error"*) and warns when the hull is sparse
     (non-contiguous, `:107-119`). It requires concrete integer ranges
     (`assert isinstance(start_expr, itir.Literal)`).
   - `Sentinel.VALUE` (dynamic offset): cartesian → `NotImplementedError`;
     `infer_domain` catches this earlier and marks the input `UNKNOWN`
     (`infer_domain.py:138-140`).
3. The chains' results are unioned (`_domain_union`, `infer_domain.py:91`), and
   `_infer_concat_where` (`:357`) intersects each branch's domain with the
   branch's region (`domain_intersection(d, promoted_cond)`, `:379`) — this is
   what localizes `z_mc` to the false region `{nlev}` in the reduced bug.

The upshot: **the static analysis the request asks for already exists**, scattered
across `trace_shifts` + `SymbolicDomain.translate`. What is missing is (a)
retaining the per-input result at the program boundary, and (b) asserting
containment against the provided fields.

## When are `D_read` and `D_provided` both concrete? (the "first call")

`D_provided(f)` enters through the argument-descriptor mechanism (ADR 21). When a
program is compiled with `static_domains=True`, `_make_compiled_programs_pool`
(`ffront/decorator.py:124`) registers a `FieldDomainDescriptor`
(`otf/arguments.py:115`) for every field parameter
(`_field_domain_descriptor_mapping_from_func_type`, `decorator.py:201`); the
descriptor extracts `field.domain` (`arguments.py:120`) at first call and keys the
compiled variant on it. At that moment the field's concrete origin..origin+shape
is a compile-time constant, and `ReplaceGetDomainRangeWithConstants`
(`transforms/replace_get_domain_range_with_constants.py`) folds the ADR-20
`get_domain_range(field, dim)` builtin into integer literals.

So at **first-call compilation of a variant with `static_domains=True`**, both
`D_read(f)` (from inference) and `D_provided(f)` (from the descriptor) are known
integers — the containment check is a handful of integer comparisons per input
per dimension. This is the request's "no runtime overhead" sweet spot, and it is
*already* the moment the toolchain compiles a variant for a given
compile-time-arg signature.

## Proposal: a spectrum of checks

The check is not one mechanism. It is a layered set, ordered by *how early* it
fires and *how much* must be known. The request's three cases map directly:

| Layer | Fires at | Engine | Catches | Cost | Request's case |
| --- | --- | --- | --- | --- | --- |
| **0 — type-level guard** | type-check / decoration | dims-as-types, `Staggered[K]` / `KHalfDim` | *kind* mismatch (full- vs half-level) | none | "other (type distinction)" |
| **1 — static containment** | first-call compile (`static_domains`) | `trace_shifts` + `SymbolicDomain.translate`, concrete bounds | most real OOB on structured + unstructured | none per call | **static extent analysis via neighbor table** |
| **2 — symbolic / runtime containment** | program entry, per call | inferred `D_read` kept symbolic vs `get_domain_range(f,·)` | OOB with non-static sizes / runtime-scalar regions | O(#inputs × #dims) per call | **runtime checks via infer_domain** |
| **3 — per-element sanitizer** | in-kernel (debug) | emitted guards / backend sanitizer | dynamic shifts, data-dependent gathers (`UNKNOWN`) | high (debug only) | "other" |

### Layer 0 — type-level guard (earliest, narrowest)

If a half-level field were typed on a dimension distinct from the full-level
`KDim` (`Staggered[K]`, or a nominal `KHalfDim`), then using a full-level `z_mc`
where a half-level extent is required is a **type error** at decoration time —
before any concrete domain exists. This is the strongest guard because it rejects
a whole *class* of staggering bugs at design time, and it is exactly the
"type `z_mc`/`z_ifc`/`ddqz` on `KHalfDim`" cleanup. The full design (dimensions
as types, `Staggered[D]` as an involutive type constructor, the value-level
`a(K + 1/2)` staggering, and a prototype) is
[[ideas/havogt/dimension-generic-fields|its own proposal]]; this one only notes
the relationship. Its limit: it catches the *grid* mismatch, not an arbitrary
off-by-one **within** one dimension (e.g. reading `[0, n+1)` from a `[0, n)`
field that *is* on the right grid) — that residue is what Layers 1–2 exist for.

### Layer 1 — static containment at first call (the primary mechanism)

A new validation pass, modeled on `CheckInOutField` (PR #2182, see below), runs
**right after** `infer_domain.infer_program`
(`pass_manager.py:181`) and **before** `concat_where.transform_to_as_fieldop`
(`:189`) — the same slot #2182 chose, and the one where inference precision is
highest (before `concat_where` is lowered to `if_` and loses its region
restriction; see Open questions and #2205). For each program input it compares the
retained `D_read(f)` against `D_provided(f)` from the `FieldDomainDescriptor`:

- **Both concrete** (`static_domains=True`): integer `⊆` test; on failure raise
  `OutOfBoundsError` naming the field, dimension, required range, and provided
  range. Zero per-call cost — the variant simply fails to compile.
- **Unstructured**: reuse `_unstructured_translate_range_statically`, and
  **promote its existing skip-value `warnings.warn` to the same diagnostic** (the
  code already has the `TODO` to make it configurable). The neighbor-table hull
  is the read domain; containment against `D_provided` is the assertion.
- **Bounds symbolic** (sizes not static): defer to Layer 2 rather than guessing a
  symbolic `⊆`.

### Layer 2 — symbolic / runtime containment (the generalization)

When `static_domains=False`, or when `D_read(f)` depends on a runtime scalar
(`nlev`), keep `D_read(f)` **symbolic** — expressed in terms of
`get_domain_range(out, dim)` (ADR 20) and the scalar params — and **emit** a cheap
guard at program entry that compares it against `get_domain_range(f, dim)` and
raises. This is `O(#inputs × #dims)` integer comparisons **once per call**,
nothing per-element and nothing in the hot loop. It is the compiled-path analogue
of embedded's intersection, and it is what makes the check work without
`static_domains`. (The emitted guard is itself just IR; `ReplaceGetDomainRange…`
collapses it to constants whenever it *can*, so Layer 2 degrades into Layer 1
exactly when domains turn out static.)

### Layer 3 — per-element sanitizer (last resort, debug only)

Static analysis cannot cover the irreducible residue: dynamic shifts
(`trace_shifts.Sentinel.VALUE` → inference yields `UNKNOWN`,
`infer_domain.py:138`) and data-dependent gathers. For those, an **opt-in debug
mode** emits per-access guards (or leans on `cuda-compute-sanitizer` / ASan).
High overhead, off by default; the only layer that catches `UNKNOWN`. Being
honest that this residue exists is part of the design.

## The public API: querying required input domains

The same retained `AccessedDomains`, surfaced:

```python
# given an output domain (+ any region-determining scalars), what must each input cover?
required: dict[str, gtx.Domain] = f.required_domains(
    out=gtx.domain({KDim: (0, nlev + 1)}), nlev=nlev, offset_provider=op
)
# -> {"z_mc": Domain(KDim: [0, nlev+1))}   # which then fails ⊆ [0, nlev)
```

Building blocks all exist: `infer_expr` computes it; `ReplaceGetDomainRangeWith
Constants` concretizes it; `FieldDomainDescriptor` shows how scalars/domains reach
the toolchain. Design choices to settle: return the **public** `common.Domain`
(not the internal `SymbolicDomain`); represent `UNKNOWN`/`NEVER` in the public
result (an enum or `None`); and decide whether the query requires a concrete out
domain + scalars (returns `Domain`s) or accepts symbols (returns symbolic
expressions). This is independently useful — users can size allocations correctly,
and can *see* the over-computation behind issue #2205 — and it is the exact datum
the check consumes, so the two ship together.

## Embedding PR #2182 (the in/out-field check) as the first sibling

[PR #2182](https://github.com/GridTools/gt4py/pull/2182) adds `CheckInOutField`,
a pass that — within a `SetAt` — runs `trace_shifts.trace_stencil` over the
stencil and raises `ValueError` if a field that is **written** is also **read with
a (non-trivial) offset** (the aliasing/in-place hazard). It is, structurally, the
*first member of a family of `trace_shifts`-based safety passes*:

- same engine (`trace_shifts.trace_stencil` over a `SetAt`),
- same pipeline slot (inserted right after `infer_domain` in
  `apply_common_transforms`),
- same shape (trace shifts → validate against a per-field fact → raise).

The bounds-containment check is the **second member**: where #2182 asks *"is a
shifted read aliased with the write target?"*, containment asks *"is a shifted
read within the provider's extent?"* — both are predicates over the same traced
shift chains. #2182 also already (a) refactored the shared
`is_tuple_expr_of` helper into `common_pattern_matcher`
(used to peel `make_tuple`/`tuple_get` down to the `SymRef` inputs — which the
containment check needs too, to attribute a read domain to a concrete parameter),
and (b) added `static_domain_sizes` to `CompiledProgramsPool`. In other words the
validation-pass scaffolding is being built right now; **bounds-containment should
ride on it, not fork it.** Concretely: land #2182, lift its
`extract_subexprs` / `filter_shifted_args` / target-attribution helpers into a
small shared `trace_shifts`-validation utility, and implement both checks against
that utility.

## Refactorings first (preferred order)

Per the standing preference to clean up before landing a feature, and because the
check is only as sound as the inference under it:

1. **Retain `AccessedDomains` at the program boundary.** Have `infer_program`
   return (or stash) the `param → D_read` mapping it already computes and
   currently discards (`infer_domain.py:540, 556`). Pure refactor; unblocks both
   the API and the check. *(No behavior change — a test pinning the returned map
   is the spec.)*
2. **Tighten `concat_where` inference precision (issue #2205).** Inference
   over-approximates the read domain when a `concat_where` has been lowered to
   `if_` before inference (both branches get the *full* domain, `_infer_if`,
   `:340`, with no region intersection — the `keep_existing_domains` comment at
   `:461` documents exactly this "unnecessary overcomputation and out-of-bounds
   accesses" hazard). A containment check on an over-approximated `D_read` would
   raise **false** `OutOfBoundsError`s. Either run the check strictly on the
   `concat_where` form (before `transform_to_as_fieldop`, `pass_manager.py:189`),
   or fix the precision first. This is a prerequisite, not an optional polish.
3. **Consolidate the existing OOB detection.** `_unstructured_translate_range_
   statically` already detects skip-value hits and non-contiguous hulls and
   `warnings.warn`s with a `TODO` to make it "a configurable error"
   (`domain_utils.py:92-119`). Route it through one diagnostic so there is a
   single "this read is out of bounds" site, shared with the new check.
4. **Land + generalize #2182's validation-pass infra** into a shared
   `trace_shifts`-validation utility (above), so the bounds pass is a sibling.
5. **Add the public `Domain`-returning inference query** (surface
   `AccessedDomains`). With (1) done this is thin, and it makes the check a
   user-inspectable computation rather than a black box.

Only then add the containment assertion (Layers 1–2) and, separately, the
type-level guard (Layer 0, which is the
[[ideas/havogt/dimension-generic-fields|dimensions-as-types]] track).

## Alternatives considered

- **A single runtime check in the kernel (per element).** Rejected as the
  *primary* mechanism: it pays per-element cost for what is a per-(field,dim)
  fact. Kept only as Layer 3 (debug) for the `UNKNOWN` residue static analysis
  cannot reach.
- **Rely on the type system alone (Layer 0).** Strongest where it applies, but it
  cannot see an off-by-one within a correctly-grided dimension, and it depends on
  the (sizeable) dimensions-as-types work. It is a *companion*, not a substitute.
- **Forward dataflow instead of backward containment.** The
  [[ideas/havogt/mesh-and-first-class-halos|mesh proposal]] computes
  `valid(out) = min over inputs (valid(in) ⊖ footprint)` *forwards* to place halo
  exchanges. It is the same arithmetic; backward containment is the local, static,
  no-mesh-concept-required version, and the natural first step. They share the
  `SymbolicDomain` algebra and should not diverge.
- **Do nothing for compiled; tell users to test in embedded.** The status quo.
  Embedded's intersection masks the bug (silently), so it does not even surface in
  embedded testing — the compiled-only crash is precisely the failure mode this
  removes.
- **Always compile with `static_domains=True`.** Would make Layer 1 universal but
  forces a recompile per distinct field-domain signature (variant explosion for
  varying sizes). Layer 2 exists exactly so the check does not require this.

## Open questions / conflicts

- **Inference precision bounds the check (issue #2205).** False positives are the
  main risk; see refactor (2). The check must run where `D_read` is tightest.
- **Unstructured hull over-approximation.** `_unstructured_translate_range_
   statically` returns the contiguous hull `[min, max+1)`; containment against the
  hull is *sound* (hull ⊇ real accesses ⇒ hull ⊆ provided ⇒ real ⊆ provided) but
  can **false-positive** when the hull crosses the buffer end though no real
  access does — notably when **skip values** (`-1`) pollute `max`. Skip values
  must be masked out of the hull before the bounds compare. This is the same
  "skip values are an artifact, not a real access" split called out in the
  [[ideas/havogt/mesh-and-first-class-halos|mesh proposal]].
- **Symbolic `⊆` is undecidable in general.** `a + nlev ⊆ b`-style comparisons
  with free symbols cannot always be settled at compile time; Layer 2 (emit a
  runtime guard) is the principled fallback rather than a heuristic.
- **Infinite read domains.** A boundary branch can yield an unbounded `D_read`
  (e.g. `concat_where(K < 0, 0.0, field)` — the constant extends to ±∞;
  `is_finite` is `False`, `domain_utils.py:316`). A finite buffer can never
  contain an infinite read domain — but in the `concat_where` case the infinite
  part reads the *constant*, not the buffer. The check must therefore apply
  **per `concat_where` piece** (the `PiecewiseFieldData` view in the
  [[ideas/havogt/field-data-protocol|FieldData protocol]]), not to the naive
  union. Ties into ADR 22.
- **Should embedded also raise?** Embedded is harmless but *silent*: it
  intersects the read away and quietly produces a smaller result. Making the
  check fire in embedded too would give one contract across backends — but the
  [[ideas/havogt/field-data-protocol|FieldData protocol]] is changing embedded's
  domain semantics (representable infinite domains), so coordinate before wiring a
  check into the embedded path.
- **Halos are intentional over-provision, not a bug.** A halo'd input deliberately
  provides `D_provided ⊋ D_compute`; the check is against `D_provided`
  (origin..origin+shape, which *includes* the halo), so a correct halo passes. The
  principled long-term home for "is the halo deep *enough*" is the mesh proposal's
  forward `valid()` dataflow; backward containment is its static, per-call shadow.
- **Error type.** Add a compiled-path `OutOfBoundsError` consistent with the
  embedded `IndexOutOfBounds` (`embedded/exceptions.py:13`); both should sit under
  `GT4PyError`, and the compiled one should carry the field name, dimension,
  required range, provided range, and — via the `DSLError` machinery
  (`errors/exceptions.py`) — the source location of the offending accessor.
- **`field_operator` vs `program` granularity.** A bare `field_operator` has no
  output-domain until embedded/called; the query API and the check are most
  natural at `program` (or first-call) granularity. Decide whether
  `field_operator.required_domains(out=…)` is offered as a convenience that
  supplies a hypothetical `D_out`.

## Appendices

None yet. Natural follow-ups: a prior-art survey of OOB/extent checking in
array/stencil DSLs (Halide bounds inference & `bound`/`bound_extent`, Devito
`subdomain` extents, Lift/RISE range analysis, polyhedral `isl` containment,
PSyclone/LFRic halo-depth checks), and a vendored prototype of the
`AccessedDomains`-retention refactor (1) plus a Layer-1 check on the reduced bug.
