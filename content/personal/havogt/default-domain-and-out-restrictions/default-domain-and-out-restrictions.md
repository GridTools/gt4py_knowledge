---
title: A default domain plus per-output restrictions
author: havogt
tags: [frontend, past, domain, multiple-output-domains, out-arguments, tuples, named-collections, embedded, dsl-design, icon4py, concat_where, slicing]
created: 2026-08-06
status: draft
---

> **TL;DR** Replace the tree-shaped `domain=` argument introduced by
> [#2225](https://github.com/GridTools/gt4py/pull/2225) with **one domain per statement plus
> per-output restrictions written inside `out`**. `domain=` is a *default*: it applies to every
> output, and is only legal when all outputs live on the same dimensions. An output that needs
> something else says so where it is named — `out=(rho, theta, pgrad[{EdgeDim: (a, b)}])` — and
> restricts per dimension, so the dimensions it does not mention keep the default. One resolution
> rule covers every case; the positional tuple, the partial-broadcast question and the
> slicing-excludes-`domain` rule all disappear. Frontend only: the emitted GTIR is unchanged.

> **Status**: draft design, not implemented. Drafted with AI assistance. Claims about today's
> behaviour were **executed, not read off the source** — see the appendix.

> **Appendix**:
> [[personal/havogt/default-domain-and-out-restrictions/default-domain-and-out-restrictions_current-state|Verified current state]]
> — what `out` and `domain=` accept across the three execution paths (they disagree), the
> out-of-bounds reproducer, why domains keyed by the out field cannot work, the icon4py survey of
> 274 programs, and an implementation map.

> **Origin**: [icon4py#1405 (review)](https://github.com/C2SM/icon4py/pull/1405#discussion_r3729957759)
> — *"is this positional only (prone to mistakes?)? or can be indexed by kw somehow?"* — asked while
> icon4py is migrating programs onto the current spelling.

> **Related (in this repo)**:
> [[personal/havogt/boundary-condition-syntax|Boundary-condition syntax over `concat_where`]] moves
> region handling *into* the operator body, where this note moves it to the call site; both must
> share one region vocabulary (OQ6).
> [[personal/havogt/scan-redesign|Scan redesign]] proposes `KDim.start`/`KDim.stop` anchored indices
> — the same anchors a runtime-valued relative bound would need (OQ7).
> [[personal/havogt/field-data-protocol|FieldData protocol]] owns the embedded restriction machinery
> (`sub_domain`) that §7 extends.

---

# A default domain plus per-output restrictions

- **Status**: draft design (not implemented)
- **Authors**: Hannes Vogt (@havogt), drafted with AI assistance
- **Created**: 2026-08-06

## 1. The rule

A statement's compute domain is resolved **per output field and per dimension**, from three sources,
narrowest last:

```
for each out leaf F, for each dimension d of F:

    range(d) = domain_of(F)[d]                  # the field's own extent
    if  domain=      mentions d:  range(d) ∩= domain[d]
    if  restriction  mentions d:  range(d) ∩= restriction[d]
```

with one validation: a **bounded** bound supplied by the user must lie inside the field's extent in
that dimension; an **unbounded** one (a one-sided condition, or a dimension left out) clamps to it.
So the intersections above are no-ops for valid input, and the check is what turns today's silent
out-of-bounds write into an error.

Everything else follows from this rule:

- `domain=` alone → today's single-domain behaviour, unchanged.
- Neither → each output uses its own field domain, unchanged.
- A restriction alone → that output is exactly as restricted.
- Both → the restriction wins in the dimensions it mentions, the default holds elsewhere.

## 2. Surface

### 2.1 `domain=` — the default

Spelling unchanged: `domain={EdgeDim: (h0, h1), KDim: (v0, v1)}`. Two changes in meaning:

- It is a **default**, not a per-output specification, so it never has tuple structure. The
  positional `domain=(d0, d1, …)` form is retired (§6).
- It may be **partial**: dimensions it does not mention keep the field's extent. This aligns the
  compiled path with embedded execution and direct field-operator calls, which already accept
  partial and unordered domains.

**Constraint.** `domain=` is only legal if **all out leaves have the same set of dimensions**, and
its keys are a subset of that set. A shared domain over outputs on different dimensions has no
meaning, and today's alternative — silently ignoring a dimension the field does not have — hides
typos. Outputs on differing dimensions must restrict individually.

### 2.2 `out=…[…]` — the restriction

Three spellings, all meaning "narrow this output":

| form | example | dimensions | bounds |
| --- | --- | --- | --- |
| domain mapping | `pgrad[{EdgeDim: (a, b)}]` | those named | arbitrary integer expressions |
| domain condition | `pgrad[(a <= EdgeDim) & (EdgeDim < b)]` | those named | arbitrary; one-sided allowed |
| slice | `pgrad[1:-1, :]` | all, positionally | integer literals only |

The mapping is canonical: it is `common.DomainLike`, the same vocabulary as `domain=`, `gtx.domain`,
`allocate` and `as_field`. The condition form is the `concat_where` vocabulary — already valid
Python (`Dimension.__le__` returns a `Domain`, `Domain.__and__` intersects) and already a
`ts.DomainType` in FOAST — which matters because the icon4py migration is translating guards written
exactly like that out of operator bodies. Slices keep working as they do today, field-relative and
literal-only; under the rule of §1, `:` needs no special case, since "the whole field" intersected
with the default *is* the default.

Restrictions compose: `f[{KDim: (a, b)}][{CellDim: (c, d)}]` is the intersection of both, so
`Subscript.value` must accept an expression rather than only a name — which also makes
`state.u[{…}]` parse.

### 2.3 Reaching the leaves

A restriction attaches to a `Field`. When `out` is a single symbol denoting a tuple or a named
collection, its leaves must be projected first:

```python
out=(t[0], t[1][{KDim: (v0, v1)}])          # tuple: project by index
out=(state.u, state.v[{KDim: (v0, v1)}])    # named collection: project by name
```

`out=state.u` already works. Tuple projection does not: `past.Subscript` types as `value.type`, so
`t[0]` has the tuple's type and the call fails the `out` type check. The fix is to dispatch on the
value's type — `FieldType` + slices/mapping/condition → restriction (type unchanged), `TupleType` +
integer literal → element type. As a side effect this also enables writing into part of a tuple
parameter.

A rebuilt tuple must satisfy the operator's return type; for a named collection that currently fails
(*"Expected keyword argument 'out' to be of type `NamedTuple{u: …, v: …}`, got `tuple[…]`"*), so
either the check accepts a structurally matching tuple literal, or the collection is reconstructed —
see OQ2.

## 3. Worked examples

### 3.1 icon4py#1405

```python
out=(
    rho_at_edges_on_model_levels,
    theta_v_at_edges_on_model_levels,
    horizontal_pressure_gradient[{dims.EdgeDim: (start_edge_nudging_level_2, end_edge_local)}],
    next_vn[{dims.EdgeDim: (start_edge_lateral_boundary, end_edge_local)}],
),
domain={dims.EdgeDim: (horizontal_start, horizontal_end), dims.KDim: (vertical_start, vertical_end)},
```

18 lines of `domain=` become one, the vertical range is written once instead of four times, and the
two outputs that differ say so next to their own names. With the condition spelling the deleted
`concat_where` guard reappears verbatim:

```python
horizontal_pressure_gradient[(start_edge_nudging_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_local)]
```

### 3.2 `graupel_run` / `muphys_run`

Thirteen domain leaves, two distinct, 64 lines — the K range of the five surface outputs is the only
thing that varies:

```python
out=(t_out, q_out, pflx,
     pr[{dims.KDim: (vertical_end - 1, vertical_end)}],
     ps[{dims.KDim: (vertical_end - 1, vertical_end)}],
     pi[{dims.KDim: (vertical_end - 1, vertical_end)}],
     pg[{dims.KDim: (vertical_end - 1, vertical_end)}],
     pre[{dims.KDim: (vertical_end - 1, vertical_end)}]),
domain={dims.CellDim: (horizontal_start, horizontal_end),
        dims.KDim: (vertical_start, vertical_end)},
```

The horizontal range comes from the default for all eight outputs, including the nested six-element
`q_out`, which needs no mention at all. This is the case that per-dimension resolution buys: a
restriction naming only `KDim` leaves `CellDim` alone.

### 3.3 Outputs on different dimensions

`fop_different_fields` (an `IField` and a `JField`) and `prog_two_vertical_dims` (`KDim` and
`KHalfDim`) have no meaningful shared domain. Under §2.1 they may not use `domain=` at all:

```python
out=(out_b[{JDim: (0, j_size)}], out_a[{IDim: (0, i_size)}])
```

which is what the positional tuple was already saying, minus the pairing by position.

## 4. Composition: why narrowing only

The rule of §1 intersects. The alternative is **override** — a restriction replaces the default in
the dimensions it names, so it could also *widen* relative to it. Override is more permissive and,
for the narrowing cases above, indistinguishable. It was rejected for one reason:

**Override cannot be implemented in embedded execution.** An embedded `@gtx.program` body runs as
plain Python, so `pgrad[{KDim: (a, b)}]` evaluates to an ordinary restricted field view. The
information "the author set `KDim` explicitly and said nothing about `CellDim`" is gone by the time
`field_operator_call` sees the argument — a view carries a domain, not a record of which dimensions
were chosen. Intersection needs no such record: the embedded path composes `view.domain ∩ default`
per leaf and gets the right answer. Override would need the out-restriction to evaluate to a
*marker* rather than a view, which means a distinguishable spelling (`gtx.at(f, …)`, `f.at[…]`) and a
new object in the embedded data model.

Nothing is lost in expressiveness: for any set of desired per-output domains, take `domain=` to be
one that contains them all (or omit it) and restrict each output to what it needs. Only convenience
is lost — the case "this output is the default plus one extra row" must name the range in full
rather than adjusting the default.

If the widening idiom turns out to matter in practice, the escape hatch is §2.2's third row: a
dedicated `at` spelling that survives into embedded as a marker. That decision can be made later
without disturbing this design; see OQ1.

## 5. Validation and errors

Every one of these is a `DSLError` with a source span:

| condition | error |
| --- | --- |
| `domain=` given and out leaves differ in their dimension sets | *"'domain' applies to all outputs and requires them to share dimensions; got `Field[[IDim]]` and `Field[[JDim]]`. Restrict each output individually."* |
| `domain=` names a dimension the outputs do not have | *"Dimension 'KDim' is not a dimension of the output(s)."* |
| a restriction names a dimension the field does not have | same, per output |
| a bounded range is not inside the field's extent | *"Domain `(0, 8)` for 'IDim' is outside the extent `(0, 6)` of out field 'out'."* — today this is an unchecked write on gtfn, see the appendix |
| duplicate dimension in a mapping, non-integer bound, wrong tuple length | as today, but with spans |

Where the containment check runs is the one open cost: fully static when the bounds are literals,
otherwise a Python-level check in the compiled-program argument layer where the bounds are plain
program parameters, and a backend assertion in the general case. Empty results stay legal — an empty
domain is absorbing — but an *attachment disjoint from the default* is almost certainly a mistake and
deserves at least a warning.

## 6. Migration

Today's tree-shaped `domain=(d0, d1, …)` is retired. Rewriting is mechanical — each dict moves from
its tuple slot onto the out element at the same position — and every known call site gets shorter.
Affected: three icon4py programs plus icon4py#1405, and in gt4py
`test_multiple_output_domains.py`, `test_named_collections.py` and `test_domain.py`.

**Timing matters.** icon4py is converting call sites onto the positional form right now, so each
week of delay adds rewrites. Suggested sequence:

1. Land the restriction syntax and the default-domain rule; keep the positional tuple accepted with
   a deprecation warning.
2. Convert icon4py.
3. Remove the tuple form.

The rest of today's surface is unaffected: `out=f`, `out=f[1:-1]` and a single `domain=` keep working
unchanged, which covers 231 of the 289 icon4py call statements and all of gt4py's other tests.

## 7. Implementation

Frontend only — the emitted GTIR is what it is today, so no backend, dace or gtfn work.

| step | where |
| --- | --- |
| `Subscript.value: Name → Expr`; allow `Dict` and comparison expressions as a subscript | `ffront/program_ast.py`, `ffront/func_to_past.py` |
| type-dispatch `visit_Subscript`: field → restriction, tuple + int literal → projection | `ffront/past_passes/type_deduction.py` |
| replace `_validate_domain_out` (tree matching) with: single-domain validation, the same-dimensions check, per-restriction validation; delete `_ensure_no_sliced_field` | same file |
| resolve per leaf and per dimension; emit `named_range` with the resolved bounds, clamping unbounded sides via `get_domain_range` | `ffront/past_to_itir.py` (`_visit_stencil_call_out_arg`, `_construct_itir_domain_arg`) |
| `Field.__getitem__` accepts a `DomainLike` mapping (today `f[{I: (1, 5)}]` raises, `f[gtx.domain(...)]` works) | `common.py` / `embedded/common.py` |
| compose `view.domain ∩ default` per leaf instead of requiring containment of the default | `embedded/operators.py` (`field_operator_call`) |
| same resolution for the direct field-operator call path | `ffront/decorator.py` |

Tests to add, none of which exist today: partial and unordered `domain=` on the compiled path; a
restriction naming a subset of dimensions combined with a default; outputs on different dimensions
rejecting `domain=`; the containment error; tuple projection; and the compiled-vs-embedded agreement
for all of the above. A `uses_multiple_output_domains` marker is missing from
`tests/next_tests/definitions.py` and should be added with them.

## 8. Open questions

- **OQ1 — Is widening wanted?** §4 rejects override because embedded cannot reconstruct
  explicitness from a view. If the "default plus one row" idiom shows up, the fix is a distinguishable
  out-restriction spelling (`gtx.at(f, …)` / `f.at[…]`) that evaluates to a marker. Worth deciding
  only with a real example in hand.
- **OQ2 — Rebuilt outputs versus their declared type.** Should a tuple literal of the right shape
  satisfy an `out` whose type is a named collection, or should the collection be reconstructed
  (`out=State(u=…, v=…)`, which PAST cannot currently call)? The first is convenient and drops the
  field names; the second keeps them and needs constructor support in PAST.
- **OQ3 — Where does the containment check live?** Static where bounds are literals; otherwise the
  argument layer or a backend assertion. Needs a count of how many icon4py bounds are plain
  parameters rather than expressions.
- **OQ4 — Same dimension *set* or same *order*?** §2.1 says set, since a domain is dimension-keyed.
  Outputs with identical dimensions in different layouts would then share a `domain=`; confirm that
  is intended.
- **OQ5 — Scan outputs.** A scan's outputs may legitimately differ along the scan axis
  (`graupel_run`'s surface-only outputs). The rule handles it, but the interaction with the
  vertical-range machinery in `embedded/operators.py::_get_vertical_range` needs checking.
- **OQ6 — Region vocabulary.** The condition spelling of §2.2 is shared with `concat_where` and with
  [[personal/havogt/boundary-condition-syntax|boundary-condition syntax]]. Those must not diverge:
  whatever "a region" looks like there should be exactly what an out restriction accepts here.
- **OQ7 — Relative bounds with runtime values.** Slices stay literal-only, because "a negative bound
  counts from the end" is decided from the literal's sign and has no runtime equivalent. The mapping
  form covers the need with absolute bounds, but "one in from each end of *this field*" then has no
  short spelling. Anchored bounds — `f[{KDim: (KDim.start + 1, KDim.stop - 1)}]`, lowered via the
  existing `get_domain_range` builtin — would supply it, and are the same anchors
  [[personal/havogt/scan-redesign|the scan redesign]] proposes. One feature, to be decided once.
