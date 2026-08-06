---
title: Surface syntax for per-output compute domains
author: havogt
tags: [frontend, past, domain, multiple-output-domains, tuples, named-collections, embedded, dsl-design, icon4py, concat_where, slicing, diagnostics, out-arguments]
created: 2026-08-06
status: draft
---

> **TL;DR** Since [#2225](https://github.com/GridTools/gt4py/pull/2225) every element of a tuple
> output can have its own compute domain, written as a **tuple of domain dicts that mirrors `out`
> positionally**. icon4py is now migrating programs onto this feature, and the first review comment
> on the first migration PR was *"is this positional only (prone to mistakes?)? or can be indexed by
> kw somehow?"*. This note answers that question. The literal reading — key the domains by the out
> field — **cannot work**, because in embedded mode the program body is real Python and `Field` is
> unhashable. The reading that does work is to stop pairing two parallel structures at all and
> **attach the domain to the out element it governs**: `out=(rho, theta, pgrad[{EdgeDim: (a, b)}])`.
> That is the same `field[domain]` restriction embedded execution already performs, it needs **no
> backend work** (identical GTIR), and it composes with a distributed `domain=` for the common
> single-domain case. Around it: seven verified defects in today's `out`/`domain` handling (including
> a **silent out-of-bounds write on gtfn**), and a staged plan whose first stage is a few days of
> frontend work that unblocks the migration.

> **Status**: draft proposal / design exploration. Drafted with AI assistance. The
> "what is possible today" claims are **empirically verified** on `gt4py@43746f13a`, not read off
> the source — see the appendix for the probes and their output.

> **Appendix**:
> [[personal/havogt/output-domain-syntax/output-domain-syntax_current-state|Verified current state]] —
> the capability matrix across the three execution paths, the probe scripts and their raw output
> (including the out-of-bounds demonstration), the icon4py survey and its script, and the
> implementation map.

> **Related (in this repo)**:
> [[personal/havogt/boundary-condition-syntax|Boundary-condition syntax over `concat_where`]] is the
> mirror image of this note — it moves region handling *into* the operator body, while the icon4py
> migration that triggered this note moves it *out* to the call site. The two must share one region
> vocabulary; §7 discusses the conflict.
> [[personal/havogt/scan-redesign|Scan redesign]] proposes `KDim.start`/`KDim.stop` anchored indices,
> which are the same anchors §6.3 needs for runtime-valued relative bounds.
> [[personal/havogt/mesh-and-first-class-halos|Mesh with first-class halos]] would replace the
> `start_edge_nudging_level_2`-style integer parameters with named zones — the long-horizon version
> of Option 7.
> [[personal/havogt/field-data-protocol|FieldData protocol]] owns the embedded restriction
> machinery (`sub_domain`) that Option 3 extends.

---

# Surface syntax for per-output compute domains

- **Status**: draft proposal / design exploration (not implemented)
- **Authors**: Hannes Vogt (@havogt), drafted with AI assistance
- **Created**: 2026-08-06

## 1. Problem / motivation

### 1.1 The trigger

[icon4py#1405](https://github.com/C2SM/icon4py/pull/1405) migrates
`compute_rho_theta_pgrad_and_update_vn` onto multiple output domains. The single `domain=` dict
becomes a four-element tuple:

```python
out=(
    rho_at_edges_on_model_levels,
    theta_v_at_edges_on_model_levels,
    horizontal_pressure_gradient,
    next_vn,
),
domain=(
    {dims.EdgeDim: (horizontal_start, horizontal_end),        dims.KDim: (vertical_start, vertical_end)},
    {dims.EdgeDim: (horizontal_start, horizontal_end),        dims.KDim: (vertical_start, vertical_end)},
    {dims.EdgeDim: (start_edge_nudging_level_2, end_edge_local), dims.KDim: (vertical_start, vertical_end)},
    {dims.EdgeDim: (start_edge_lateral_boundary, end_edge_local), dims.KDim: (vertical_start, vertical_end)},
),
```

> **@jcanton**: is this positional only (prone to mistakes?)? or can be indexed by kw somehow?
> **@havogt**: currently only this is possible

The reviewer is right to be uneasy. In the real file the two tuples are ~5 lines apart, the
correspondence is carried entirely by list position, and three of the four dicts differ from each
other in exactly one token. Nothing checks that the *n*-th domain was meant for the *n*-th field;
the type checker cannot help, because all four entries have the same type.

### 1.2 Why the pressure is rising now

Two things make this more than a cosmetic complaint.

**(a) The migration has started.** icon4py is converting programs to per-output domains as a
deliberate campaign. Whatever the spelling is, it is about to be replicated across the dycore.

**(b) Per-output domains are *replacing* `concat_where` guards.** This is the part that changes the
character of the feature. In #1405 the operator body loses a guard:

```python
# before: the validity region is a `concat_where` inside the operator
horizontal_pressure_gradient = concat_where(
    (start_edge_nudging_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_local),
    _compute_horizontal_pressure_gradient(...),
    broadcast(wpfloat("0.0"), (dims.EdgeDim, dims.KDim)),   # dummy value, never consumed
)

# after: the validity region is an output domain at the call site
horizontal_pressure_gradient = _compute_horizontal_pressure_gradient(...)
```

That is a genuine improvement — no over-computation, no dummy branch, no fill value that only exists
to make the types work out. It also means the **region vocabulary migrates from the operator body to
the call site**, so per-output domains stop being an exotic feature used by three call sites and
become the normal way to express "this output is only valid here". A syntax that is merely tolerable
at three call sites is not good enough at thirty.

Note the asymmetry the migration creates: in the body the region was written
`(start_edge_nudging_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_local)`; at the call site the
*same region* must be re-spelled `{dims.EdgeDim: (start_edge_nudging_level_2, end_edge_local)}`, in a
tuple slot whose position encodes which output it belongs to. Same concept, two notations, and the
link between them is positional.

### 1.3 Evidence: how the feature is used today

Survey of all 274 `@gtx.program`s in icon4py (script and raw output in the appendix):

| | |
| --- | --- |
| call statements in program bodies | 289 |
| … carrying an explicit `domain=` | 231 |
| … whose `domain=` is a tuple (multiple output domains) | 3 |
| … with a subscripted (`out=f[...]`) out element | 2 |
| programs with more than one statement | 4 |
| redundant repetitions of an identical domain literal *within* one program | 1 |

Two conclusions that constrain the design:

- **The single-domain case dominates and must stay first-class.** 231 of 289 statements name one
  domain for all outputs. Any proposal that forces per-output spelling on them is a regression.
- **Cross-statement repetition is a non-problem.** Programs are effectively single-statement, so a
  scoped construct (`with domain(...):` around a program body, a program-level default) would buy
  almost nothing. This kills an option that looks attractive in the abstract; see Option 8.

The three existing multi-domain call sites, however, are large:

| call site | out leaves | domain leaves | distinct domains | lines of `domain=` |
| --- | --- | --- | --- | --- |
| `muphys_run` | 8 | 13 | 2 | 64 |
| `graupel_run` | 8 | 13 | 2 | 64 |
| `compute_perturbed_quantities_and_interpolation` | 9 | 9 | 6 | 50 |
| `compute_rho_theta_pgrad_and_update_vn` (#1405) | 4 | 4 | 3 | 18 |

`graupel_run` writes **thirteen** domain dicts of which **two** are distinct, spread over 64 lines,
and annotates them with `# t_out`, `# q_out`, `# pflx`, … comments — a hand-maintained index into
the positional correspondence. `compute_perturbed_quantities_and_interpolation` interleaves prose
comments explaining *why* a particular output needs a wider halo, which is exactly the information
that wants to live next to the field it describes.

## 2. What is possible today

Three unrelated mechanisms decide where a statement writes. All claims verified; see appendix §A.

| form | frame of reference | bounds | dimensions | per-output |
| --- | --- | --- | --- | --- |
| `out=f` | the full domain of `f` | — | — | yes, since #2225 |
| `out=f[1:-1]` | **relative** to `f`'s own `start`/`stop` | integer **literals only** | **all**, in field order | yes, per element |
| `domain={I: (lo, hi)}` | **absolute** index space | arbitrary integer expressions | **all**, in field order | tuple mirroring `out` |

`domain=` and slicing are mutually exclusive (`_ensure_no_sliced_field`), so the two axes cannot be
combined on one call.

### Verified defects

- **D1 — No partial broadcast.** A single dict may stand for a whole tuple only at the *top* level;
  at any nested position it raises `Domain dict cannot map to tuple outputs`. This is why
  `graupel_run` must write six identical dicts for the nested `q_out`.
- **D2 — Three different contracts for the same keyword.** The compiled program path requires the
  domain to list **all** dimensions **in field order**; the *same program* run embedded accepts a
  partial domain; the direct field-operator call accepts partial *and* unordered (it goes through
  `out[domain]`). The strictest of the three is the one users hit in production.
- **D3 — Absolute domains are unchecked on gtfn.** With `out` a 4-row field and
  `domain={IDim: (0, 6)}`, gtfn writes 6 rows — two rows past the field, into the neighbouring
  field's memory, silently. Embedded and roundtrip raise `IndexOutOfBounds` for the same program.
  (Appendix §A.3 is a reproducer that observes the corrupted bytes.)
- **D4 — Runtime bounds force the absolute form.** `out=f[lo:-1]` with a program parameter raises a
  raw `TypeError: 'Slice.lower' must be Constant`. The rule "a negative bound counts from the end"
  is decided *syntactically* from the literal's sign, so it does not generalise to runtime values —
  the restriction is structural, not accidental.
- **D5 — Slices look absolute and are not.** `f[1:5]` means `[f.start + 1, f.start + 5)`, not
  `[1, 5)`. For a field whose domain starts at 0 the two coincide, which is exactly how this
  footgun stays hidden until someone passes a field with an offset origin.
- **D6 — A domain cannot be named.** Program bodies allow no assignments, closure variables of type
  `dict` are rejected (`Type <class 'dict'> not supported`), and `gtx.domain(...)` cannot be called
  in PAST. The two distinct domains of `graupel_run` therefore have to be written out thirteen times.
- **D7 — `Subscript.value` must be a `Name`.** This blocks chaining (`f[a][b]`) and blocks
  subscripting a named-collection field (`state.u[...]`), even though `out=state.u` itself works.
- **D8 — Internal errors leak.** Bare `AssertionError`/`TypeError`/`ValueError` without source
  location for: literal-only slice bounds, wrong dimension order, `out=f[{...}]`, closure-variable
  domains, `...` in a slice, a slice step.

D1 and D2 are what the icon4py migration hits first; D3 is the one that should worry us most.

## 3. Design axes

Naming the axes makes the options comparable rather than a list of syntaxes.

1. **Where is the domain written?** Call-site keyword · attached to the out element · declared in the
   operator.
2. **How is the correspondence to an output established?** Position · name/key · syntactic adjacency.
3. **What is the frame of reference?** Absolute indices · relative to the field · anchored
   (`Dim.start + 1`).
4. **How do multiple specifications compose?** Innermost overrides · intersection.
5. **Is a specification complete?** All dimensions required · partial allowed (unspecified dimensions
   inherited from the field).
6. **Does it survive embedded execution?** In embedded mode a `@gtx.program` body is executed as
   plain Python, so any surface form must be *evaluable* — this axis eliminates options rather than
   ranking them.
7. **What does it cost?** Frontend-only (identical GTIR) · new IR/backend support.

## 4. Options

### Option 1 — Minimal repair of the current spelling

Allow a dict at any nesting level to stand for a whole subtree (fixes D1); accept partial and
unordered domains in the compiled path, matching what embedded and direct calls already do (D2);
turn the leaked internal errors into `DSLError`s with spans (D8).

`graupel_run`'s 13 leaves collapse to 3 (one for the whole tuple, then per-output overrides — except
that "override" is not expressible; see below), and #1405 gains nothing at all: its four domains are
genuinely distinct, so the positional tuple stays.

- **Pros.** Days of work, purely additive, no new syntax to teach, no backend involvement. Removes
  the worst structural duplication immediately.
- **Cons.** Does not answer the review comment. The correspondence stays positional, and partial
  broadcast makes it *harder* to read, not easier: with a dict covering a subtree, the reader must
  now infer how many leaves each entry covers before counting positions.
- **Verdict.** Necessary but not sufficient — the correctness half (D2, D8) should ship regardless;
  the partial-broadcast half is better subsumed by Option 3.

### Option 2 — Key the domains by the out field (the literal reading of the review comment)

```python
out=(rho, theta, pgrad, vn),
domain={
    pgrad: {EdgeDim: (start_nudging_2, end_local), KDim: (v0, v1)},
    vn:    {EdgeDim: (start_lb, end_local),        KDim: (v0, v1)},
    ...:   {EdgeDim: (h0, h1),                     KDim: (v0, v1)},   # default
},
```

**This cannot work.** In embedded mode the program body runs as real Python, and a dict literal is
evaluated for real: `Field.__hash__` does not exist (`TypeError: unhashable type: 'numpy.ndarray'`)
because `Field.__eq__` is element-wise and returns a field. A form that works only in the compiled
path and raises `TypeError` in embedded is not a candidate.

Even setting that aside: the same field appearing twice in `out` would collide, nested tuple
elements have no symbol to key on, and the keys would have to be validated against `out` anyway —
so it buys naming at the cost of a second consistency check.

- **Verdict.** Rejected, on a hard constraint. Worth recording precisely because it is the obvious
  first answer to the question that was actually asked.

### Option 3 — Attach the domain to the out element  *(recommended core)*

Stop maintaining two parallel structures. A statement's compute domain is a property of its write
target, so write it there:

```python
out=(
    rho_at_edges_on_model_levels,
    theta_v_at_edges_on_model_levels,
    horizontal_pressure_gradient[{dims.EdgeDim: (start_edge_nudging_level_2, end_edge_local)}],
    next_vn[{dims.EdgeDim: (start_edge_lateral_boundary, end_edge_local)}],
),
domain={dims.EdgeDim: (horizontal_start, horizontal_end), dims.KDim: (vertical_start, vertical_end)},
```

`domain=` keeps its meaning for the common case and **distributes** over the out tree; a per-element
subscript **narrows** it further. Correspondence is syntactic adjacency: the domain is on the field,
so there is nothing to mis-pair. #1405's 4+4 structure becomes one shared domain plus two local
narrowings — and the two vertical ranges, which never varied, are written once.

`graupel_run`:

```python
out=(t_out, q_out, pflx,
     pr[surface_k], ps[surface_k], pi[surface_k], pg[surface_k], pre[surface_k]),
domain={dims.CellDim: (horizontal_start, horizontal_end), dims.KDim: (vertical_start, vertical_end)},
```

with `surface_k = {dims.KDim: (vertical_end - 1, vertical_end)}` spelled inline (five times, one line
each) until Option 6 makes it nameable. 64 lines → ~10.

**Why this spelling and not another.** `f[domain]` is not new syntax: it is what embedded execution
*already does* to implement `domain=` — `decorator.py` turns `domain=` into
`tree_map(lambda f, d: f[d])(out, domain)`, and the embedded operator call ends in
`target[domain] = source[domain]`. Today's `out=f[1:-1]` is the same operation with a relative
index. Option 3 makes the surface agree with the semantics that are already implemented one layer
down.

- **Pros.** Answers the review comment (no positions to count). Domain sits next to the field it
  governs, and so do the `#`-comments explaining it. Composes with the dominant single-`domain=`
  form instead of replacing it. Identical GTIR — **no backend work**, in contrast to #2225 which
  needed a substantial dace effort. Works in embedded with one small, principled extension
  (`Field.__getitem__` must accept a `DomainLike` mapping; today `f[{I: (1, 5)}]` raises
  `IndexError: Unsupported index type` while `f[gtx.domain({I: (1, 5)})]` works).
  Lifts D1 (nothing to broadcast), D3 (a restriction cannot leave the field), D7 (the
  `Subscript.value` widening it requires also enables `state.u[...]`).
- **Cons.** Does not help when `out` is a *single symbol* denoting a tuple or named collection
  (`out=out_state`) — there is no element to attach to without destructuring. `domain=` as a
  structural tuple must therefore stay for that case, so we end up with two mechanisms rather than
  one. The dict-in-subscript reads a little dense; §6.2 discusses alternatives.
- **Verdict.** Recommended as the core.

### Option 4 — Named collections with keyed domains

When the operator returns a named collection, names *do* exist, and both of these become
well-defined:

```python
domain=Diagnostics(rho={...}, pgrad={...}),          # mirror the collection type
out=Diagnostics(rho=rho, pgrad=pgrad[{...}]),        # or attach, via Option 3
```

`Field`-keyed dicts fail on hashability; *string*-keyed ones would need names, which is exactly what
a named collection supplies. #2428 already made named collections work with multiple output domains,
but the domain tuple is still matched **positionally in element order** — there is a
`TODO(havogt)` in `past_passes/type_deduction.py` saying so.

- **Pros.** The only form that gives real keyword indexing. Self-documenting; reorder-safe.
- **Cons.** Requires the operator to return a named collection. icon4py's dycore returns plain
  tuples; converting is a larger change than the syntax problem it solves, and the muphys/graupel
  collections are inputs, not outputs. Constructing a collection inside a PAST body is not currently
  supported (only inside field operators).
- **Verdict.** Complementary, not a substitute. Worth doing as a follow-up: replace the positional
  `TODO` with true by-name matching, so that named-collection users get keyed domains for free.

### Option 5 — Declare the output domain in the operator

The information "the pressure gradient is only valid on `[nudging_2, local)`" is a property of the
computation, not of the call site — that is why it lived in a `concat_where` before #1405. So let the
operator keep saying it, without the dummy branch:

```python
@gtx.field_operator
def _compute_rho_theta_pgrad_and_update_vn(..., start_edge_nudging_level_2: int32, end_edge_local: int32):
    ...
    return (
        rho_at_edges,
        theta_v_at_edges,
        restrict(pgrad, (start_edge_nudging_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_local)),
        restrict(next_vn, start_edge_lateral_boundary <= dims.EdgeDim),
    )
```

The call site then keeps a single `domain=` and the migration touches no call sites at all. Note that
the zone bounds are *already* parameters of the operator (they were the `concat_where` conditions),
so no new information has to be plumbed.

- **Pros.** Single source of truth; the region is named by the local variable it constrains; reuses
  the existing `concat_where` condition vocabulary verbatim, so #1405's diff becomes
  `concat_where(cond, x, 0.0)` → `restrict(x, cond)` — strictly less code, same shape. Embedded is
  free (`restrict(f, d) == f[d]`).
- **Cons.** In a compiled backend the output domain is not just where values land — it is **where
  kernels execute**. The statement's domain is the iteration space that is handed to the backend, so
  moving a narrowing inside the operator body means the launch bounds are now determined by an
  expression in the body rather than by the statement. That is a much larger change than a surface
  one: it needs a `restrict` builtin (GTIR has `as_fieldop(stencil, domain)` and `concat_where`, but
  no first-class domain-narrowing primitive), domain inference must intersect it with the target
  domain and propagate it to the inputs, and every backend must derive its launch configuration from
  the result. Restriction (slicing generally) inside field operators is a plausible future
  direction, but it is a separate, bigger project. Secondary: it hides the write extent from the
  call site, which is where halo policy is decided, and it cannot express a *call-site-specific*
  narrowing.
- **Verdict.** Not a competitor to Option 3 on this timescale. Keep as a future direction to be taken
  up on its own terms, not as part of fixing the output-domain surface.

### Option 6 — Domains as first-class values

Make `Domain` a type that a program parameter can have, and/or allow naming a domain in a program
body:

```python
@gtx.program
def prog(..., interior: gtx.Domain, surface: gtx.Domain):
    fop(..., out=(t, q, pr[surface], ps[surface]), domain=interior)
```

- **Pros.** Fixes D6 head-on: the two distinct domains of `graupel_run` become two names. Callers
  already compute these bounds from the grid (`grid.start_index(cell_domain(Zone...))`), so passing a
  `Domain` instead of four `int32`s is closer to the caller's actual data. `ts.DomainType` already
  exists in the type system, and `get_domain_range` (ADR 20) shows the IR can carry domain values.
  Interacts well with `static_domains` JIT specialisation.
- **Cons.** The largest scope of anything here: a domain-typed parameter must cross the compiled
  program ABI, be fingerprinted, and be describable to the argument-descriptor machinery (ADR 21).
  Introduces a second way to pass bounds alongside the existing scalars, which icon4py would have to
  migrate to *again*.
- **Verdict.** Right direction, wrong order. Revisit after Option 3 has settled the surface.

### Option 7 — Named zones from the mesh concept

`out=vn[edges(Zone.NUDGING_LEVEL_2, Zone.LOCAL)]` — the horizontal bounds in every icon4py program
are grid zones that have been flattened into `int32` parameters. See
[[personal/havogt/mesh-and-first-class-halos|mesh with first-class halos]].

- **Pros.** Collapses both the parameter list and the domain spelling; the halo reasoning becomes
  checkable instead of commented.
- **Cons.** Depends on a mesh concept that does not exist yet.
- **Verdict.** Long horizon. Option 3's subscript is the right place to plug it in later — one more
  spelling of "a restriction", not a new mechanism.

### Option 8 — Do nothing new

Two sub-variants. **(a) Split the call**: one statement per domain group. This is what was done
before #2225 and it re-runs the operator per group, which is exactly the cost #2225 removed —
unless the toolchain fuses them, which it does not guarantee. **(b) A scoped default**
(`with domain(...):` around several statements, or a program-level default): the survey kills it —
programs are single-statement, so there is nothing to scope over.

- **Verdict.** Rejected, but (b) is worth recording so it is not re-proposed; the intuition that
  "the domain repeats a lot" is right *within* a statement and wrong *across* statements.

### Comparison

| | correspondence | embedded-safe | fixes D1 | fixes D3 | backend work | cost |
| --- | --- | --- | --- | --- | --- | --- |
| **O1** repair | positional | ✓ | ✓ | ✗ | none | days |
| **O2** field-keyed | by key | **✗ (unhashable)** | ✓ | ✗ | none | — |
| **O3** attached | **adjacency** | ✓ (+1 small ext.) | n/a | ✓ | **none** | 1–2 weeks |
| **O4** named collections | by key | ✓ | ✓ | ✗ | none | weeks (+ user migration) |
| **O5** in-operator | adjacency (in body) | ✓ | n/a | ✓ | **IR + inference** | prototype first |
| **O6** domain values | n/a | ✓ | n/a | ✓ | ABI, descriptors | months |
| **O7** zones | n/a | ✓ | n/a | ✓ | mesh concept | long horizon |
| **O8** status quo | positional | ✓ | ✗ | ✗ | none | zero |

## 5. Recommendation

**Stage 0 — correctness, ship independently (days).** The half of Option 1 that is not a syntax
question: accept partial and unordered domains in the compiled path so that the three execution
paths agree (D2); replace the leaked internal exceptions with `DSLError`s carrying spans (D8);
document that slices are field-relative (D5). None of this changes any working program.

**Stage 1 — Option 3 (1–2 weeks, frontend only).** Widen `past.Subscript.value` from `Name` to
`Expr` (D7); accept a domain mapping as a subscript on an out element; define `domain=` as
distribution over the out tree; define composition as intersection (§6.1); extend
`common.Field.__getitem__` to accept a `DomainLike` mapping so the same source runs embedded. Keep
the positional `domain=` tuple working — it stays the only option when `out` is a single symbol.

**Stage 2 — close the out-of-bounds hole (D3).** With Option 3 in place the *frontend* can lower a
restriction against `get_domain_range` and clip by construction for the cases that are unbounded by
construction (omitted dimensions, one-sided conditions). For explicitly bounded requests, clipping
silently computes less than asked, so the honest behaviour is an error — which needs either a
Python-level check in the compiled-program argument layer (cheap where the bounds are plain
parameters) or backend support for an assertion (general, but per-backend work). Decide by measuring
how many icon4py bounds are plain parameters; §7 OQ3.

**Stage 3 — by-name matching for named collections (Option 4).** Replace the positional
named-collection `TODO` in `past_passes/type_deduction.py` with matching on field names.

Option 5 is explicitly **not** staged here: because the statement domain is the kernel launch domain,
restriction inside a field operator is a separate project with backend-wide impact, not a step in
this one.

Do **not** take Options 6 or 7 as prerequisites — they change what a bound *is*, not how an output is
paired with one, and both remain available on top of Option 3's subscript.

## 6. Semantics of the recommended core

### 6.1 Composition

For each leaf of the out tree:

```
compute_domain(leaf) = leaf_field.domain ∩ attached_restriction ∩ distributed_domain
```

Intersection, not override, so that the operation is associative and `f[a][b]` is well-defined.
Dimensions not mentioned by a restriction are inherited from the field. A restriction that is
*bounded* in a dimension and not contained in the field's range is an **error**, matching
`embedded/common.py::_absolute_sub_domain`; an *unbounded* one (a one-sided condition, an omitted
dimension) is clipped to the field, matching `concat_where`.

The rule to check against Option 3's examples: `graupel_run`'s `pr[{KDim: (v1-1, v1)}]` intersected
with `domain={CellDim: (h0, h1), KDim: (v0, v1)}` gives `{CellDim: (h0, h1), KDim: (v1-1, v1)}` — the
narrowing wins in K because it is a subset, and the shared domain supplies Cell. Override-per-dimension
would give the same answer here and differ only when the two are not nested, where intersection is
the safer reading of a subscript that is spelled as a restriction.

### 6.2 How a restriction is spelled

Three candidate spellings, not mutually exclusive:

- **(a) Mapping literal** — `pgrad[{EdgeDim: (a, b)}]`. Canonical: it is `common.DomainLike`, the
  same vocabulary as `domain=`, `gtx.domain`, `allocate` and `as_field`. Partial by nature,
  order-free. **Recommended for Stage 1.**
- **(b) Domain conditions** — `pgrad[(a <= EdgeDim) & (EdgeDim < b)]`. This is the `concat_where`
  vocabulary, already valid Python (`Dimension.__le__` returns a `Domain`, `Domain.__and__`
  intersects) and already a `ts.DomainType` in FOAST. It is *the* spelling the icon4py migration is
  translating away from, so supporting it verbatim would make the migration diff nearly mechanical.
  Needs `ast.Compare`/`BoolOp` support in PAST (today: `Unsupported Python syntax: 'ast.Compare'`).
  **Recommended for Stage 2.**
- **(c) Dimension-indexed literals** — `pgrad[EdgeDim[a:b]]`, the `K[0]` / `K[1:nlev]` region
  literals from [[personal/havogt/boundary-condition-syntax|the boundary-condition note]]. Most
  compact; requires `Dimension.__getitem__`. Only worth adding if that note adopts them, and then in
  both places at once.

Existing relative slices (`f[1:-1]`) stay exactly as they are — literal-only sugar. They are *not*
generalised to runtime bounds, because the "negative means from the end" rule has no runtime
equivalent (D4); the general answer is anchored bounds, §6.3.

### 6.3 Anchored bounds

`f[{K: (K.start + 1, K.stop - 1)}]` expresses "one in from each end" with the same mechanism as an
absolute range, and lowers directly to the existing `get_domain_range` builtin (ADR 20). It makes
`f[1:-1]` pure sugar and removes the literal-only restriction without inventing a sign convention.
[[personal/havogt/scan-redesign|The scan redesign]] independently proposes `KDim.start`/`KDim.stop`;
these should be **one** feature, decided once.

### 6.4 Compatibility

Every form in §2 keeps working. Stage 1 is purely additive. Stage 0 turns errors into non-errors
(partial/unordered domains) and improves messages. The only intended breakage is Stage 2's rejection
of a domain that exceeds the out field — today that is an unchecked memory write, so it has no
correct users; still, it should land behind a warning first and be measured against icon4py.

## 7. Open questions / conflicts

- **OQ1 — Two mechanisms, permanently?** Option 3 cannot serve `out=out_state`, so the positional
  `domain=` tuple survives for that case. Is that acceptable, or should the single-symbol case be
  forced to destructure (`out=(state.u, state.v[...])`, which Option 3's `Subscript.value` widening
  enables anyway)?
- **OQ2 — Direction of travel.** This note moves region information *to* the call site;
  [[personal/havogt/boundary-condition-syntax|boundary-condition syntax]] moves it *into* the
  operator body, and Option 5 sits in between. All three are defensible for different regions (a
  physical boundary condition belongs in the body; a halo policy belongs at the call site), but they
  must not end up with three different notations for the same set of indices. Fixing the shared
  region vocabulary is more urgent than choosing between them.
- **OQ3 — Where does the bounds check live?** Frontend clip via `get_domain_range` (free, silent),
  Python-level check in the argument layer (loud, only where bounds are plain parameters), or a
  backend assertion (general, per-backend). Needs a count of how many icon4py domain bounds are plain
  parameters versus expressions.
- **OQ4 — What would in-operator restriction cost end to end?** Option 5's blocker is that the
  statement domain is the kernel launch domain, so the question is not only whether domain inference
  accepts a `restrict`, but how each backend derives its launch configuration once the narrowing
  lives in the body. Worth scoping as its own effort.
- **OQ5 — Named-collection matching.** Replacing element-order matching with by-name matching (the
  `TODO` in `past_passes/type_deduction.py`) is a small change with a compatibility question: is any
  existing code relying on positional matching for a named collection?
- **OQ6 — Does `out=f[...]` want to grow into tuple-element writes?** `out=tup[0]` is currently a type
  error. Allowing it would let a program write into part of a tuple parameter — adjacent to this
  proposal, and it would make the subscript node mean two different things (restriction and
  projection) unless the type decides.
