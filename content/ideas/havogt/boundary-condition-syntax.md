---
title: Frontend syntax sugar for boundary conditions over concat_where
author: havogt
tags: [frontend, foast, boundary-conditions, concat_where, regions, syntax-sugar, if-elif-else, match, embedded, cartesian-parity, dsl-design, prior-art, python-versions, peps, piece-algebra, metaclass, replay]
created: 2026-06-16
---

> **Early design ideas.** This is an exploratory brainstorm — a menu of mostly mutually exclusive
> possibilities to react to, not a committed design, a recommendation of record, or a plan to
> implement. Nothing here has been prototyped or agreed on; expect rough edges, overlap, and dead
> ends.

> **TL;DR** `gt4py.next` writes boundary conditions as `concat_where(domain, true, false)`.
> Real code (icon4py) builds a multi-region column as a **chain of reassignments** or a
> **nest** of `concat_where` calls, which reads poorly. `match` / `if`-`elif`-`else` would be
> nicer but cannot run in **embedded** mode (the field-operator body executes as plain Python,
> and a `Domain` condition is not a `bool`). `gt4py.cartesian` solves the analogous problem with
> a `with horizontal(region[...])` statement, but only because cartesian has **no embedded mode**
> — it always AST-parses. This note proposes **nine** surface syntaxes that desugar to
> `concat_where` and discusses which can serve *both* execution modes. The unifying trick: a
> construct works in embedded **iff it desugars to `concat_where` calls before the body runs** —
> which is possible exactly when the construct carries an unambiguous *syntactic marker* (with two
> notable loopholes: `class`-body capture and replay execution).

> **Status**: draft proposal / design exploration. Drafted with AI assistance, including a
> web survey of how other DSLs encode boundary conditions and a survey of real `concat_where`
> usage in [icon4py](https://github.com/C2SM/icon4py). Not implemented.

> **Appendices**:
> [[ideas/havogt/boundary-condition-syntax_research|Cross-DSL prior-art survey]] — how
> Devito, FEniCS/Firedrake, Halide, Lift/RISE, YASK, GridTools/STELLA/Dawn, Pochoir, ExaSlang,
> Regent/Legion, Chapel, NEMO/FMS, Mathematica, Rust/Swift/Python `match`, and APL encode
> "interior vs boundary", plus a verified Python language-feature watch (3.14/3.15/3.16, dead
> PEPs) and precedents for the implementation tricks (AST/bytecode rewriting, tracer-refuses-bool,
> replay, `__prepare__`). · [[ideas/havogt/boundary-condition-syntax_icon4py-survey|icon4py
> concat_where survey]] — ~40 real call sites, the chained/nested idioms, and the inline
> `# TODO: Use if-else statement instead`.

> **Related (in this repo)**:
> [[ideas/havogt/scan-redesign|Redesign of the vertical scan]] proposes boundary handling
> *inside scans* via "peelable index conditionals" and strawman anchored indices
> `KDim.start`/`KDim.stop`; this note is the orthogonal **field-operator frontend** layer.
> [[ideas/havogt/field-data-protocol|FieldData protocol]] is the orthogonal **embedded
> representation** layer (it makes unbounded-domain `concat_where` representable).

---

# Frontend syntax sugar for boundary conditions over `concat_where`

- **Status**: draft proposal / design exploration (not implemented)
- **Authors**: Hannes Vogt (@havogt), drafted with AI assistance
- **Created**: 2026-06-16

## Problem / motivation

`gt4py.next` expresses domain/region boundary conditions with the `concat_where` builtin:

```python
concat_where(domain_condition, true_field, false_field)
```

Unlike pointwise `where(mask_field, a, b)`, the first argument is a **`Domain`** (not a mask
field), the two branches may live on **different sub-domains**, and the result is a
*concatenation* along one dimension (see `src/gt4py/next/ffront/experimental.py`). A condition
like `KDim < 5` *is* a `Domain`: `Dimension.__lt__` returns
`Domain(dims=(KDim,), ranges=(UnitRange(-inf, 5),))` (`common.py:111-135`). Conditions combine
with `&` (intersection) and `|` (union), and `==` yields a one-index range.

This is a fine *primitive*, but it is a poor *surface syntax* for the thing people actually
write: "compute X in the interior, Y at the model top, Z at the surface." Two pain points show
up in real code (see the [[ideas/havogt/boundary-condition-syntax_icon4py-survey|icon4py survey]],
~40 call sites):

**1. Chained reassignment.** The dominant idiom is to refine one variable region-by-region.
From icon4py `compute_weight_factors.py` (`_compute_wgtfac_c`), a three-region column:

```python
wgt_fac_c = concat_where(
    (dims.KDim > 0) & (dims.KDim < nlev),
    _compute_wgtfac_c_inner(z_ifc),
    z_ifc,
)
wgt_fac_c = concat_where(dims.KDim == 0,    _compute_wgtfac_c_0(z_ifc=z_ifc),    wgt_fac_c)
wgt_fac_c = concat_where(dims.KDim == nlev, _compute_wgtfac_c_nlev(z_ifc=z_ifc), wgt_fac_c)
```

**2. Nesting.** A multi-way (three-or-more-region) split is written as a *nest* of
`concat_where` calls, read inside-out. From icon4py
`compute_edge_diagnostics_for_dycore_and_update_vn.py` (`apply_on_vertical_level`), a textbook
`if/elif/else`:

```python
return concat_where(
    dims.KDim < nflatlev,
    on_flatlevels,
    concat_where(nflat_gradp + 1 <= dims.KDim, below_flatgradp, between_flat_and_flatgradp),
)
```

The wish is explicit in the icon4py source — `apply_diffusion_to_vn.py` literally carries
`# TODO(): Use if-else statement instead` above a `concat_where`-in-a-ternary.

What we want is something like

```python
match KDim:
    case == 0:    wgt = top(z_ifc)
    case == nlev: wgt = surface(z_ifc)
    case _:       wgt = inner(z_ifc)
```

or chained `if`/`elif`/`else`. The cartesian sibling DSL already has a beloved construct for
exactly this — `with horizontal(region[I[0], J[-1]])` and `with computation(...), interval(0, 1)`
(designed in GDP-0002). **But neither `match` nor `region` is directly possible in `next`**, and
the reason is the crux of this note.

## The core constraint: embedded execution runs the raw Python body

`gt4py.next` field operators run in **two** modes:

- **compiled** — the decorator parses the function to FOAST/GTIR and lowers it to a backend.
  Here the body is *traced*, never executed as written, so we have total freedom (like cartesian).
- **embedded** — `backend=None`. The decorator **executes the original Python function**
  directly on `Field`s (`decorator.py:408`: `self.definition_stage.definition(*args, **kwargs)`).
  `concat_where(...)` works here only because it is a *function call* dispatching to an embedded
  implementation (`nd_array_field.py:_concat_where`).

`gt4py.cartesian` has **no embedded mode**: `region`, `interval`, `computation`, `I`, `J` are
inert runtime stubs (no-op context managers; `region.__getitem__` is `pass`) and the *real*
meaning lives entirely in an `ast.parse` of the source (`gtscript_frontend.py`). It never has to
run as Python, so it can use statement syntax freely.

`next` does not have that luxury. A statement-based sugar (`if`, `match`, `with`) over a **domain
condition** has a fundamental problem in embedded mode:

- `if KDim == 0:` runs `bool(KDim == 0)` = `bool(Domain(...))` — there is no sensible truth value,
  and even if there were, native control flow takes **one** branch and assigns a **whole** field,
  which is *not* the spliced-along-a-dimension semantics of `concat_where`.
- The FOAST frontend already proves the point: it *parses* `if`/`else` into `IfStmt`
  (`func_to_foast.py:422`) and ternaries into `TernaryExpr`, but type-deduction **requires the
  condition to be a scalar `bool`** (`type_deduction.py:362-373`) and lowers it to a scalar
  select `im.if_` — not `concat_where`. `and`/`or` are rejected (use `&`/`|`); chained compares
  `0 < KDim < 5` are rejected with an auto-suggestion to write `(KDim > 0) & (KDim < 5)`.

So the design space is governed by one rule:

> **A statement/expression sugar can serve _embedded_ execution iff it is desugared into
> `concat_where(...)` calls _before the body runs_** — i.e. by a source-to-source AST rewrite in
> the `@field_operator` decorator (the same `ast` trick cartesian uses, but emitting
> `concat_where` so the rewritten function still executes natively). For **compiled** execution
> any of these can additionally/instead be lowered in FOAST.

And a corollary that discriminates the proposals:

> **A decoration-time AST rewrite can only fire on an unambiguous _syntactic_ marker** (it has no
> type information). `with region(<cond>):`, `out[<cond>] = ...`, and a `cases(...)` call are
> unmistakable. A bare `if <cond>:` is **not** — it is syntactically identical to a scalar
> `if param > 0:`, so it cannot be rewritten without types, which are only available in the
> compiled path. This makes `if`/`match` the *nicest-looking* but *hardest-to-unify* options.
> Two loopholes bypass the rule without a marker or a rewrite: **replay execution**
> (Proposal 1d) and **`class`-body capture** (Proposal 9).

The cross-DSL survey ([[ideas/havogt/boundary-condition-syntax_research|appendix]]) confirms the
two archetypes worth stealing from: **domain-splicing** systems (YASK `IF_DOMAIN`, Devito
`Eq(subdomain=)`, GridTools per-interval `apply`, Mathematica `DirichletCondition[value, pred]`)
attach a *different computation to a sub-region* — exactly `concat_where`; **pointwise boundary
modes** (Halide `BoundaryConditions`, Lift `Pad.Boundary.Clamp`) re-index out-of-bounds reads and
are a different feature. We want the first kind.

---

## Proposal 1 — `if` / `elif` / `else` over domain conditions

*(Python feature (mis)used: `if` statements + the existing FOAST `IfStmt` node + operator-overloaded `Dimension` comparisons that already return `Domain`.)*

```python
@field_operator
def wgtfac_c(z_ifc: fa.CellKField[wpfloat]) -> fa.CellKField[wpfloat]:
    if KDim == 0:
        wgt = top(z_ifc)
    elif KDim == nlev:
        wgt = surface(z_ifc)
    else:
        wgt = inner(z_ifc)
    return wgt
```

**Semantics.** An `if`/`elif`/`else` whose condition is a `Domain` lowers to nested
`concat_where`, first-match-wins (the `elif` chain becomes the `false` branch nest, the `else` is
the innermost false field). The variable(s) assigned in every branch become the result — FOAST's
`IfStmt` *already* enforces "same symbols, same types in both branches"
(`type_deduction.py:375-383`), which is exactly the constraint we need.

**Compiled.** Small, surgical frontend change: in `visit_IfStmt`, if the condition types as
`DomainType` instead of scalar `bool`, lower to `im.concat_where` instead of `im.if_`. The
`&`/`|` condition algebra and the `(A & B) → nested concat_where` canonicalization already exist
(`iterator/transforms/concat_where/canonicalize_domain_argument.py`).

**Embedded.** This is the hard case (see the corollary). A bare `if KDim == 0:` cannot be
AST-rewritten without types, and it cannot run *correctly* natively — worse, today it runs
**silently wrong** (verified): `Domain` defines no `__bool__`, so `bool(KDim == 0)` falls back to
`Sequence.__len__` and is always `True` — the true branch is unconditionally taken. (`Field`
already raises "The truth value of a Field is ambiguous"; `Domain` deserves the same guard
regardless of what happens to this proposal.) Options, none free:
(a) **compiled-only** — reject domain-`if` in embedded with a clear error (acceptable in that
cartesian regions are *also* compiled-only, but a real regression for a DSL that sells embedded);
(b) **`Domain.__bool__` raises** a guiding error ("use `concat_where`/`cases`, or a backend") so
embedded fails loudly rather than wrongly — the exact strategy of JAX
(`TracerBoolConversionError`, pointing to `lax.cond`) and `torch.fx` ("symbolically traced
variables cannot be used as inputs to control flow"); (c) require a tiny disambiguating marker,
e.g. `if domain(KDim == 0):`, which restores a syntactic hook for the rewrite (but then it is no
longer "just `if`"); (d) **replay (multi-shot) execution** — run the body once per branch
decision, with `Domain.__bool__` returning a *recorded real `bool`* (`True` on one run, `False` on
the next), then assemble the per-run results with `concat_where` over the recorded conditions.
This is legal Python (`__bool__` must return an actual `bool` — and here it does), needs no source
access, and works today; it costs one body execution per region and is sound only because field
operators are pure. Precedent: concolic testers like CrossHair repeatedly re-execute a function,
forcing each branch boolean and "remember[ing] what decisions it has made so that it can make
different decisions on future executions"; the functional-pearl form is *thermometer
continuations* (Koppel, Scherer, Solar-Lezama 2018: "replaying the entire computation from the
start, guiding it using a recording"). No array DSL is known to do this — array DSLs either refuse
`bool()` or rewrite — which makes it novel, not disqualified. Worked example below.

**Pros.** Most natural, most Pythonic, zero new vocabulary; reuses `IfStmt` wholesale; reads
exactly like the icon4py `# TODO: Use if-else statement instead` wish; collapses the *nested*
cases cleanly; matches Mathematica `Piecewise`'s ordered first-true model. Good for multi-statement
branches.

**Cons.** The embedded story is the hard part — the scalar-`if` vs domain-`if` ambiguity is
irreducible at the AST level, forcing compiled-only, a marker (which erases the elegance), or
replay (which costs one execution per region and leans on a purity contract). Mixing scalar
control-flow `if` and domain `if` in one operator is confusing. `else` is effectively mandatory
(the result must be defined everywhere).

### Option (d) in detail: replay execution, worked

Concrete trace for the three-region example at the top of this proposal. The decorator wraps the
embedded call in a recorder; during a run, `Domain.__bool__` consults an oracle that returns
`True` at a fresh fork and forced values on scheduled re-runs:

| run | forced decisions | `bool(KDim == 0)` | `bool(KDim == nlev)` | body computes |
| --- | --- | --- | --- | --- |
| 1 | *(none)* | `True` | *not reached* | `wgt = top(z_ifc)` |
| 2 | `[False]` | `False` | `True` | `wgt = surface(z_ifc)` |
| 3 | `[False, False]` | `False` | `False` | `wgt = inner(z_ifc)` |

Folding the run results over the recorded conditions produces exactly the nest a user writes
today, with first-match semantics:

```python
concat_where(KDim == 0, run1, concat_where(KDim == nlev, run2, run3))
```

The whole recorder is small (sketch):

```python
def replay(body, *args):
    runs = []                                    # (decisions, result) per completed run

    def explore(forced):
        decisions = []
        def oracle(domain):                      # Domain.__bool__ delegates here during a run
            take = forced[len(decisions)] if len(decisions) < len(forced) else True
            decisions.append((domain, take))
            return take
        with _domain_bool_oracle(oracle):
            result = body(*args)                 # one ordinary embedded execution
        runs.append((decisions, result))
        for i in range(len(forced), len(decisions)):          # schedule unexplored siblings
            prefix = [t for _, t in decisions[:i]] + [False]  # fresh forks defaulted True → flip
            if not _path_is_empty(decisions[:i], flipped=prefix[-1]):
                explore(prefix)

    explore([])
    return _fold(runs)    # nested concat_where over the decision tree
```

Properties, concretely:

- **Cost is nonempty paths, not conditions.** An `if`/`elif`/…/`else` chain with *n* conditions
  has *n+1* paths → *n+1* runs, each computing only its own branch — about the same total field
  work as the equivalent `concat_where` nest, which also evaluates every branch eagerly. The real
  overhead is re-running the *shared prefix* of the body once per run; hoisting shared code above
  the first domain-`if` (or memoizing field ops) mitigates.
- **Empty-path pruning kills the exponential.** *Sequential* domain-`if`s fork multiplicatively
  (two paint-style `if`s → 4 paths), but a scheduled path whose accumulated region is empty —
  intersect the forks' domains, complements for `False` decisions — is skipped *without running*
  (`_path_is_empty` above). For disjoint regions (`KDim == 0`, `KDim == nlev`) the both-`True`
  path is empty, so paint-style sequential `if`s stay at n+1 runs. The genuinely multiplicative
  case — a `KDim` condition nested with an `EdgeDim` condition — yields exactly the nonempty
  sub-boxes that any lowering of a 2-D decomposition must produce anyway.
- **Scalar control flow just works.** A real `bool` (`if limited_area:`) is not intercepted; each
  run takes its actual branch. icon4py's `apply_diffusion_to_vn` pattern (a Python flag selecting
  between two `concat_where`s) replays only the domain forks inside the taken scalar branch.
  Early `return` inside a branch works — each run's return value is its path's result — and tuple
  returns make it multi-output for free (assembled tuple-wise).
- **The contract is purity + determinism.** The body must produce the same fork sequence given the
  same decisions (field operators satisfy this by design); side effects repeat once per run.
  Failures can be reported *with the failing path's region attached* ("while computing the
  `KDim == nlev` branch…") — arguably better error UX than today's single execution.

---

## Proposal 2 — `match` / `case` over a dimension

*(Python feature (mis)used: PEP 634 structural pattern matching — and a hard look at its limits.)*

```python
match KDim:
    case _ if KDim < nflatlev:          # guards only — see below
        vn = on_flatlevels
    case _ if nflat_gradp + 1 <= KDim:
        vn = below_flatgradp
    case _:
        vn = between
```

or, leaning into PEP 635's own escape hatch, **custom matchable region objects**:

```python
match KDim:
    case below(nflatlev):               # __match_args__ on a region pattern class
        vn = on_flatlevels
    case at_or_above(nflat_gradp + 1):
        vn = below_flatgradp
    case _:
        vn = between
```

**The honest finding (from the [[ideas/havogt/boundary-condition-syntax_research|prior-art survey]]).**
Python's `match` has **no range patterns** — unlike Rust (`0..=5 =>`) or Swift (`case 0...5:`). PEP
634 defines only literal/capture/wildcard/value/group/sequence/mapping/class patterns; PEP 635
explicitly rejects range patterns and steers users to **guards** (`case x if 0 <= x < 5`) or
**custom pattern objects**. So a `match` over `KDim` *degrades to guarded `if`/`elif` with extra
ceremony*, and the subject/scrutinee is awkward (you "match" a `Dimension`, then guard on a
`Domain`). The `case _:` wildcard is a clean default, and the structure *looks* exhaustive, but
Python won't check exhaustiveness.

Three verified sharp edges (CPython grammar, unchanged through 3.15): a region literal cannot even
be *written* in a pattern — `case K[0]:` is a **SyntaxError** (value patterns must be dotted
names); a bare `case K:` does not compare against `K`, it silently **captures** the subject into a
new local named `K`; and `case range(0, 5):` parses but is a *class/isinstance* pattern, not a
range test. Every road leads to guards or custom `__match_args__` classes.

**Compiled.** Needs a new `visit_Match` in FOAST lowering `case`→nested `concat_where`. The
custom-pattern variant additionally needs the frontend to understand the region pattern classes.

**Embedded.** Same wall as Proposal 1 (subject + guards execute natively, `bool(Domain)`), plus
`match` is *less* AST-detectable as "domain dispatch" than a `with`/subscript marker. The
custom-pattern form *is* a syntactic marker (`case below(n):`) and could be AST-rewritten, which
is its one redeeming property.

**Pros.** Familiar; `case _:` is a natural else; custom region patterns (`case Interior():`) could
read beautifully and are PEP-635-blessed; would age well if Python ever adds range patterns.

**Cons.** Native `match` buys *no expressive power over `if`/`elif`* for ranges (guards only) while
adding ceremony and a conceptually wrong scrutinee — **the weakest option for the plain form**.
A new frontend node for what is a linear `(region → value)` table. Same embedded breakage. I rank
this last unless we commit to the custom-matchable-object variant (which is really Proposal 4 in
disguise).

---

## Proposal 3 — `with region(...)` blocks (the cartesian-`region` analog)

*(Python feature (mis)used: `with` + context managers as a syntactic marker; decoration-time AST rewrite to `concat_where`.)*

```python
@field_operator
def wgtfac_c(z_ifc: fa.CellKField[wpfloat]) -> fa.CellKField[wpfloat]:
    with regions() as wgt:
        with region(KDim == 0):
            wgt = top(z_ifc)
        with region(KDim == nlev):
            wgt = surface(z_ifc)
        with otherwise():
            wgt = inner(z_ifc)
    return wgt
```

**Mechanism.** `with region(<domain>):` is an unambiguous syntactic marker, so the decorator can
AST-rewrite each block to `wgt = concat_where(<domain>, <block result>, <accumulator>)` **without
type information** — therefore it works **uniformly in embedded and compiled** (rewrite once,
before the body runs). This is the cartesian trick, but emitting `concat_where` so the rewritten
function still executes natively in embedded. `otherwise()` supplies the innermost default.

**Pros.** Direct parity with the cartesian `region` people already like (easiest migration path);
the marker gives a *single* implementation for both modes; groups multi-statement boundary code
visually; `otherwise()` is an explicit default. Mirrors Devito `Eq(subdomain=)`, YASK `IF_DOMAIN`,
ExaSlang `apply bc`.

**Cons.** `with`-blocks read as imperative mutation but the result is functional — a slight
conceptual mismatch (also true of cartesian). Capturing the per-block assignment requires the AST
rewrite (a runtime-only context manager *cannot* intercept `wgt = ...`, since `with` does not trap
assignments) — so this proposal is inseparable from shipping the rewriter. Cartesian's subtle "a region executes only where consumed in the
compute domain" semantics would *not* carry over (next has explicit domains), so it is parity in
*looks* more than in *meaning*.

---

## Proposal 4 — A fluent / varargs / dict builder (no AST magic)

*(Python feature (mis)used: varargs, dict literals with `Ellipsis` as the default, or chained methods — a pure-runtime fold to nested `concat_where`.)*

```python
# (a) varargs of (condition, value) + default=  — recommended
wgt = cases(
    (KDim == 0,    top(z_ifc)),
    (KDim == nlev, surface(z_ifc)),
    default=inner(z_ifc),
)

# (b) dict literal, Ellipsis as the "else"
wgt = piecewise({
    KDim == 0:    top(z_ifc),
    KDim == nlev: surface(z_ifc),
    ...:          inner(z_ifc),
})

# (c) fluent builder
wgt = (when(KDim == 0, top(z_ifc))
       .when(KDim == nlev, surface(z_ifc))
       .otherwise(inner(z_ifc)))
```

**Mechanism.** Pure sugar that folds the `(condition, value)` pairs right-to-left into nested
`concat_where`. Because it is *just a function call*, **embedded works with zero new machinery** —
no AST rewrite, no `__bool__` hacks, the call dispatches to `_concat_where` exactly like a
hand-written nest. For **compiled**, it must be a recognized builtin: the varargs `cases(...)` is
easiest to type and lower; the dict form needs the frontend to parse a `Dict` literal with an
`Ellipsis` default; the fluent form needs the frontend to trace method chains (hardest).

**Pros.** Lowest risk and most incremental; **the only proposal that needs nothing special for
embedded**; explicit and greppable; `default=` / `...` / `.otherwise()` is an unambiguous else;
the varargs form is the literal `(region → value)` table that Mathematica `Piecewise` and OpenFOAM
`boundaryField` validate as a recurring idiom; easy to attach good errors (missing default,
overlapping/duplicate regions). A natural semantic core that the statement sugars (1, 3, 5) can all
desugar *into*.

**Cons.** Not "native" syntax — it is still a function, so it does not feel like language-level
control flow. All branch expressions are evaluated eagerly (same as today's nested `concat_where`:
a boundary stencil computes over the whole column and is then sliced — no short-circuit). The dict
form's keys are `Domain` objects (must be hashable; `...`-as-default is cute but a little obscure).
It flattens the *visual* nesting but not the *semantic* nesting.

---

## Proposal 5 — Domain-subscript assignment ("painting")

*(Python feature (mis)used: `__setitem__` with a `Domain` subscript on an output accumulator — like numpy masked assignment, and the closest match to how people mentally build BCs.)*

```python
@field_operator
def wgtfac_c(z_ifc: fa.CellKField[wpfloat]) -> fa.CellKField[wpfloat]:
    wgt = undefined(z_ifc)                 # or seed with the interior directly
    wgt[...]          = inner(z_ifc)       # base / default layer
    wgt[KDim == 0]    = top(z_ifc)         # paint the model top
    wgt[KDim == nlev] = surface(z_ifc)     # paint the surface
    return wgt
```

**Mechanism.** Each `wgt[<domain>] = value` desugars to `wgt = concat_where(<domain>, value, wgt)`
— **last paint wins** (the intuitive "start from the interior, stamp the boundaries" model), with
`wgt[...] = base` as the default layer. The subscripted-assignment target with a `Domain` index is
an unambiguous syntactic marker, so it AST-rewrites uniformly across embedded and compiled. (In
embedded it can even be done at runtime: make the accumulator a small builder whose
`__setitem__(domain, val)` runs `self._f = concat_where(domain, val, self._f)`; though typing
`return wgt` then needs care, the AST-rewrite route is cleaner.)

**Variant — annotated-assignment marker.** `wgt: at[K[0]] = top(z_ifc)` puts the region in the
*annotation* slot instead of the subscript. A verified language fact makes this attractive:
annotations on local variables inside functions "have no runtime effect — they're discarded at
compile-time" (PEP 526; reconfirmed unchanged by PEP 649 for Python 3.14+). The marker is
therefore free at runtime, purely syntactic, and cannot collide with real element indexing the
way `wgt[<domain>]` can. The trap: *un*-rewritten embedded silently executes the plain assignment
(each line overwrites the whole field), so the decorator must rewrite unconditionally — annotated
assignments must never fall through. `wgt: Annotated[fa.CellKField[wpfloat], K[0]] = ...` is the
type-checker-clean spelling, at the price of verbosity.

**This collapses icon4py's dominant idiom 1:1.** The three-line
`wgt_fac_c = concat_where(...); wgt_fac_c = concat_where(...); wgt_fac_c = concat_where(...)` chain
*is already* "paint region after region onto one variable", read inside-out. Domain-subscript
assignment is that idiom with the noise removed — you paint `wgt[K[0]] = top(...)` directly, in
source order.

**Pros.** Matches the real chained-reassignment idiom most directly (biggest practical win on
actual code); reads like numpy `a[mask] = ...` and cartesian region-assignment; "last paint wins"
is intuitive for BCs and needs no mandatory single `else` (the base layer is the default); the
marker gives one implementation for both modes; incremental
(add one region at a time). Echoes APL `⌺`/Devito region-restricted assignment.

**Cons.** Tension with SSA/immutability: field operators are single-assignment and functional, and
`wgt[d] = v` *looks* like mutation — needs the accumulator object or AST rewrite to stay
functional. The seed/base (`undefined(...)` then a mandatory `wgt[...] =`, or a "must cover all
points" check) needs defining and validating. "Last paint wins" is the *opposite* precedence of
`if`/`match` "first match wins" — shipping both would need a clearly documented distinction.
`wgt[KDim == 0]` (domain condition) overloads the same `__setitem__` that `wgt[0]` (index) would
use, so the two must be unambiguous.

---

## Proposal 6 — `concat[regions](branches)`: a positional N-way splice

*(Python feature (mis)used: `__getitem__` to take a list of region literals, then `__call__` to take one branch per region — an expression, a multi-arm `concat_where` in a single call. From old design notes.)*

```python
# region literals: K[0] ≡ "level 0", K[1:nlev] ≡ "levels 1 .. nlev-1", K[nlev] ≡ "level nlev"
result = concat[K[0], K[1:nlev], K[nlev]](lower, interior, upper)

# or leave the interior implicit:
result = concat[K[0], K[nlev]](lower, upper, default=interior)
```

**A region-literal notation worth adopting everywhere.** `K[...]` is sugar for a `Domain`: `K[0]` ≡
`KDim == 0`, `K[1:nlev]` ≡ `(KDim >= 1) & (KDim < nlev)`, `K[nlev]` ≡ `KDim == nlev`. It mirrors
cartesian's `I[0]` / `J[-1]` axis-subscript and Python slicing, and is markedly cleaner than
spelling out comparisons. Any proposal here can use it (e.g. Proposal 1's `if K[0]:`, Proposal 5's
`out[K[0]] = ...`); it is introduced here because it is what makes the positional form read well.

**Mechanism.** `concat[r0, r1, r2]` captures the tuple of region literals via `__getitem__` and
returns a callable; calling it with the branches matches branch *i* to region *i* by position and
folds to nested `concat_where`. It is an **expression** (a call), so embedded works for free; for
compiled the frontend recognizes the `concat[...](...)` builtin.

**Language notes.** Subscripts cannot take keyword arguments — PEP 637 was **rejected** — so
`default=` can never move inside the brackets (`concat[K[0], default=inner]` is a SyntaxError
forever); it must stay a call argument as shown. Star-unpacking inside subscripts is legal since
Python 3.11 (PEP 646), so computed region lists work: `concat[*regions](*branches)`. The Draft
PEP 718 ("subscriptable functions") would, if it ever lands, make the subscript-then-call shape a
typed first-class citizen.

**Pros.** The most compact N-way form — every region and the whole splice in one expression; no
statement/SSA/scope concerns; embedded-safe with no AST rewrite; the `K[1:nlev]` slice is far nicer
than `(KDim >= 1) & (KDim < nlev)`. A disjoint region list reads as a **partition** of the column
(each level in exactly one region), so there is no precedence to reason about.

**Cons.** Positional region↔branch matching gets error-prone past a few regions — you read the
region list, then count into the branch list, and a misalignment is silent. Regions and values are
physically separated. `concat[...](...)` (subscript-then-call) is unusual Python that
linters/type-checkers/readers may trip on. Needs a coverage rule (regions must tile the output, or a
`default=`). A new notation (`K[...]`) to learn, though it pays for itself.

---

## Proposal 7 — `with concatenated() as result:` (a scoped region builder)

*(Python feature (mis)used: a `with` scope that introduces and finalizes a result builder, combining region blocks with subscript painting. From old design notes; refines Proposals 3 and 5.)*

```python
@field_operator
def boundary(...):
    with concatenated() as result:
        result[K[0]]      = top(...)          # one-liner painting for simple regions
        result[K[1:nlev]] = inner(...)
        with result(K[nlev]) as upper:        # a block (+ sub-handle) for multi-statement regions
            tmp = some(...)
            upper[...] = surface(tmp)
    return result
```

(Two pure variants also work: every region as a `with result(K[...]) as sub:` block, or every
region as a `result[K[...]] = ...` one-liner.)

**Mechanism.** `concatenated()` is a context manager that yields a builder `result` and assembles
the recorded regions into a `concat_where` structure on `__exit__`. `result[region] = expr` records
a one-liner; `with result(region) as sub:` opens a sub-scope whose local handle `sub` (or `result`
itself) is painted with `sub[...] = expr`. The enclosing `with concatenated()` is an unambiguous
marker, so one decoration-time AST rewrite serves embedded and compiled alike.

**Pros.** The explicit scope **removes Proposal 5's seed/SSA ambiguity** — `result` is introduced
and finalized by `concatenated()`, so there is no question of where the accumulator comes from. It
is the **best of Proposals 3 and 5**: one-liners for simple regions, blocks (with a named
sub-handle) for multi-statement boundary code. The regions read top-to-bottom as a domain
decomposition, and a disjoint region set is a clean **partition** (each region assigned once — no
precedence). Unambiguous marker → uniform embedded/compiled.

**Cons.** The most machinery of any proposal: a context-manager protocol, per-region sub-handles,
and the AST rewrite. `result` plays three roles — the scope handle, the block opener
`result(region)`, and the painter `result[region] =` — which must be specified carefully or it
confuses. Per-region `with` is verbose when most regions are one-liners (the mixed form mitigates).
Needs a coverage rule (regions tile the domain, or an explicit `otherwise()`), and typing the
partially-built `result` through the scope is non-trivial.

---

## Proposal 8 — Restrict-then-concatenate: a piece algebra

*(Python feature (mis)used: `field[domain]` restriction plus a variadic `concat(...)` — regions attached to values instead of listed separately.)*

```python
wgt = concat(
    top(z_ifc)[K[0]],
    inner(z_ifc)[K[1:nlev]],
    surface(z_ifc)[K[nlev]],
)
```

**Mechanism.** `field[region]` is the restriction embedded fields already support (`field[domain]`
slices a field to a sub-domain), and `concat(*pieces)` concatenates fields with disjoint, adjacent
domains. This is arguably not sugar *over* `concat_where` but a decomposition *of* it: the embedded
`_concat_where` already works exactly this way internally (restrict both branches, then `_concat`
the slices), and `concat_where(d, t, f) ≡ concat(t[d], f[~d])`. An expression → embedded works
essentially for free; compiled needs a `concat` builtin and restriction typing.

**Pros.** Fixes Proposal 6's positional-alignment hazard: each region is attached to its value, so
nothing can silently misalign. Mathematically clean — restriction + disjoint union, the same mental
model as assembling one array from disjoint labeled pieces in xarray (`combine_by_coords`).
Partition semantics, no precedence. Reuses existing restriction syntax; no new statement forms.

**Cons.** Scalar branches need a wrapper (`broadcast(0.0)[K[0]]` — a bare scalar has no
`__getitem__`). Naive embedded computes each branch on its full domain before restricting (same as
today; compiled domain inference avoids the waste). The tempting operator spelling `piece | piece`
is **blocked**: `__or__` on fields already means elementwise boolean `or`, so overloading it by
domain-disjointness would be ambiguous — the variadic `concat(...)` call is the honest form.
Coverage/adjacency errors surface only at assembly time.

---

## Proposal 9 — `class`-body capture (the no-rewrite statement block)

*(Python feature (mis)used: PEP 3115 metaclass `__prepare__` namespaces — the only Python statement whose body executes against a namespace the library controls.)*

```python
@field_operator
def boundary(z_ifc: fa.CellKField[wpfloat]) -> fa.CellKField[wpfloat]:
    class wgt(concatenated):            # ← not actually a class: binds the assembled field
        this[K[0]]      = top(z_ifc)
        this[K[1:nlev]] = inner(z_ifc)
        this[K[nlev]]   = surface(z_ifc)
    return wgt
```

**Mechanism.** Documented runtime semantics, no AST access anywhere: a metaclass's `__prepare__`
returns the mapping used as the class-body namespace and observes **every name binding, in order**
(verified: rebindings included) — this is exactly how `enum` tracks member definition order via its
`_EnumDict` and how Django's declarative `Model` bodies work. The namespace can inject magic names
(`this`), and the metaclass call may return **any object** — the `class` statement just binds the
result, so `wgt` becomes the assembled *field*, not a class. Multi-statement regions work as
ordinary temporaries in the body, and class bodies can read enclosing function locals (`z_ifc`
resolves).

**Pros.** The **only statement-block form that works in embedded with no AST rewrite at all** —
pure PEP 3115 semantics. Ordered observation supports painting or partition; multi-statement
regions are natural; `class wgt(concatenated):` is also a crisp syntactic marker for the compiled
path.

**Cons.** Deeply weird: a `class` that isn't a class will surprise every reader, linter, and type
checker (`this` is undefined to mypy; the binding's type is wrong). One verified scoping trap: a
name assigned *inside* the class body and read before its assignment falls back to *globals*, not
the enclosing function scope — shadowing an argument inside the block misbehaves. The compiled
frontend would have to accept `class` statements inside field operators. Best treated as an
**existence proof** — the bar any AST-rewrite statement form must beat — rather than a
recommendation.

### Multiple output fields

First, the general fact: `concat_where` already accepts **tuples** (and named collections) as
branches — icon4py assigns `(rho, theta) = concat_where(cond, helper(...), (0, 0))`, and the
compiled pipeline has a dedicated `expand_tuple_args` transform — so *every* proposal here is
multi-output-capable by passing tuple branches. The painting forms even get it in plain Python:
`rho[K[0]], theta[K[0]] = helper(...)` is ordinary tuple assignment with subscript targets. What
`class`-body capture adds are three distinct *groupings*:

**Tuple values** (cheapest — the "class" binds a tuple; unpack after the block):

```python
class rho_theta(concatenated):
    this[lateral_zone] = _compute_horizontal_advection_of_rho_and_theta(...)  # returns a tuple
    this[...]          = (broadcast(0.0, ...), broadcast(0.0, ...))
rho, theta = rho_theta
```

**Field-major** (named handles via `this.<name>`; the result is a named collection):

```python
class fields(concatenated):
    this.rho[K[0]]        = rho_top(...)
    this.rho[K[1:nlev]]   = rho_inner(...)
    this.theta[K[0]]      = theta_top(...)
    this.theta[K[1:nlev]] = theta_inner(...)
return fields.rho, fields.theta
```

**Region-major** (a nested class per region, plain assignments per field — no other proposal has
this grouping). The namespace observes tuple unpacking from a single helper call, which is exactly
the real icon4py shape (one helper computing several fields for one zone):

```python
class fields(concatenated):
    class lateral(region[start_edge_lb_7 <= EdgeDim]):
        rho, theta = _compute_horizontal_advection_of_rho_and_theta(...)
    class elsewhere(otherwise):
        rho   = broadcast(0.0, ...)
        theta = broadcast(0.0, ...)
rho, theta = fields.rho, fields.theta
```

Even `class lateral(region[...])` with a non-class base works at pure runtime: PEP 560
`__mro_entries__` substitutes a marker base and the original domain stays readable via
`__orig_bases__` — the same machinery `Generic[T]` bases use. Two additional cautions: the
namespace must **not** auto-create unknown names (loads fall through it to the enclosing
scope/globals — that is exactly how `z_ifc` and `top` resolve, so swallowing misses would break
name resolution), and coverage checking becomes per-field (every field defined on every region,
or a per-field default).

---

## Design axes (how the proposals compare)

| Proposal | Python feature | Form | AST-rewritable marker | Embedded cost | Precedence | Default / else | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **P1** `if`/`elif` | `if` / `IfStmt` | statement | **no** (≡ scalar `if`) | ✗ (types) / replay | first-match | `else:` | med–high |
| **P2** `match` | PEP 634 match | statement | weak (guards) / yes (custom pat) | ✗ | first-match | `case _:` | high |
| **P3** `with region` | `with` / ctx-mgr | statement | **yes** | via rewrite | first-match | `otherwise()` | medium |
| **P4** builder | varargs / dict / chain | expression | n/a (already a call) | **✓ free** | first-match | `default=` / `...` | **low** |
| **P5** paint `out[d]=` | `__setitem__` | statement | **yes** | via rewrite / builder | **last-paint** | base `out[...]` | medium |
| **P6** `concat[..](..)` | `__getitem__` + `__call__` | expression | n/a (already a call) | **✓ free** | **partition** | `default=` / tiling | low–med |
| **P7** `with concatenated()` | `with` + `__setitem__` | statement | **yes** | via rewrite / builder | partition / last-paint | `otherwise()` / base | med–high |
| **P8** `concat(pieces...)` | restriction `[]` + varargs call | expression | n/a (already a call) | **✓ free** | partition | full-domain piece | low |
| **P9** `class`-body capture | metaclass `__prepare__` (PEP 3115) | statement | yes (`class X(concatenated)`) | **✓ free (no rewrite)** | partition / last-paint | base paint | high (weirdness) |

## How regions compose: first-match, last-paint, partition

The proposals differ in what happens to a point that more than one region could touch — the
"Precedence" column above. There are three models; all three still lower to nested `concat_where`
and differ only in the contract the *surface* signs the user up for.

**Painting (overlapping, order matters).** Start from a base and stamp regions on top, like numpy
masked assignment; where regions overlap, a later write wins over an earlier one.

```python
out[...]     = inner(...)    # base layer: the whole column
out[K[0]]    = top(...)      # stamp over level 0
out[K[nlev]] = surface(...)  # stamp over the last level
```

The base `out[...]` is the default. This is Proposal 5, and exactly icon4py's real idiom
(`wgt = concat_where(...); wgt = concat_where(..., wgt)` — each call refines the previous result).
Flexible and incremental, but reordering can change the result when regions overlap.

**Partition (disjoint tiling, order irrelevant).** The regions are mutually exclusive and together
cover every point exactly once.

```python
out = concat[K[0], K[1:nlev], K[nlev]](top, inner, surface)  # disjoint; tile the whole column
```

No point is touched twice, so there is no precedence and order does not matter — the natural reading
of Proposals 6/7. Cleaner, and potentially cheaper (nothing is computed-then-overwritten), but it
carries a **coverage obligation**: every point must fall in some region (or an explicit
`default=`/`otherwise()`), and the tooling must *check* completeness and disjointness.

**First-match (overlapping, first wins).** Ordered like `if`/`elif`/`else`: the first region that
matches a point wins. This is Proposals 1/2/4 (and a hand-written `concat_where` nest) — the same
"ordered + overlap allowed" shape as painting, but with the opposite priority.

| Model | Overlap? | Who wins | Surface examples |
| --- | --- | --- | --- |
| **first-match** | allowed | the *first* matching region | `if`/`elif`/`else` (P1), `match` (P2), `cases(...)` (P4), a `concat_where` nest |
| **last-paint** | allowed | the *last* write | `out[domain] = ...` (P5) |
| **partition** | none (by construction) | — (each point once) | `concat[...](...)` (P6), `with concatenated()` (P7), `concat(pieces...)` (P8), `class` body (P9) |

first-match and last-paint are both ordered-with-overlap but opposite in priority; partition is
unordered-without-overlap with a coverage requirement. The open design question is which single
model to commit to — shipping several risks leaving users unsure which write wins.

## Recommendation

A pragmatic, layered path rather than a single winner:

1. **Ship an embedded-safe expression core first — Proposal 4 (`cases(...)`), Proposal 6
   (`concat[...](...)`), and/or Proposal 8 (`concat(pieces...)`).** All fold to `concat_where`
   with no AST surgery and run in embedded immediately; `cases` reads best for a handful of
   labelled regions, `concat[...]` for a compact positional partition, and pieces (P8) when each
   value should carry its own region (no alignment hazard). The statement sugars can desugar
   *into* this core.
2. **Make Proposal 7 (`with concatenated() as result:`) the headline statement sugar.** It subsumes
   the painting of Proposal 5 and the region blocks of Proposal 3, fixes Proposal 5's seed/SSA
   ambiguity with an explicit scope, and its marker yields one decoration-time AST rewrite serving
   *both* embedded and compiled. Proposal 5's bare `out[domain] =` painting is the lighter option
   when no enclosing scope is wanted.
3. **Adopt the `K[...]` region-literal notation everywhere** — `K[0]` / `K[1:nlev]` is cleaner than
   `KDim == 0` / `(KDim >= 1) & (KDim < nlev)` and mirrors cartesian's axis-subscript.
4. **Offer Proposal 1 (`if`/`elif`/`else`) as the most Pythonic surface for branching cases**, once
   the scalar-vs-domain `if` ambiguity is resolved — most likely type-directed lowering in compiled
   plus a loud `Domain.__bool__` error (or an explicit `if domain(...)` marker) in embedded.
5. **Keep Proposal 3 (`with region`) as the cartesian-parity option**, and treat **Proposal 2
   (`match`) as lowest priority** — Python's missing range patterns make the plain form a more
   verbose `if`/`elif` (and `case K[0]:` is a SyntaxError); only the custom-matchable-region
   variant is interesting, and it collapses into Proposal 4/6. **Proposal 9 (`class`-body
   capture)** stays documented as an existence proof — the one no-rewrite statement block — not a
   recommendation.

In every form a region is an arbitrary `Domain`-valued expression — a single comparison, an
`&`/`|` combination, a multi-dimensional condition, or a named index bound — and the sugar composes
the regions into the appropriate `concat_where` structure, so the user writes a flat sequence of
regions and never hand-builds the nest.

## Python feature watch (3.14, 3.15, and beyond)

Statuses verified against peps.python.org and the 3.14/3.15 "What's New" (2026-07); links and
quotes in the [[ideas/havogt/boundary-condition-syntax_research|research appendix]].

- **Shipped in 3.14.** **Template strings (PEP 750, Final)**: interpolations evaluate eagerly, but
  each `Interpolation` also carries the **source text** of its expression — Python's first
  sanctioned quasi-quotation channel. A region DSL could exploit it
  (`concat(t"{K[0]}: {top(z_ifc)}")` — values eager for embedded, expression text available to a
  compiled path), but code-in-strings is a readability regression; noted and rejected.
  **Deferred annotations (PEP 649/749, Final)** change nothing here: function-local annotations
  "have no runtime effect — they're discarded at compile-time", which is exactly what the
  Proposal 5 annotated-assignment variant relies on.
- **Shipped in 3.15.** `lazy import` (PEP 810; a soft keyword special to import statements — not
  reusable by libraries) and star-unpacking in comprehensions (PEP 798). Nothing that helps region
  dispatch.
- **3.16 pipeline.** Nothing syntax-level accepted yet. Drafts worth watching: **PEP 718**
  subscriptable functions — would make Proposal 6's `concat[...](...)` a typed first-class shape;
  PEP 653 (match semantics) — still no range patterns anywhere.
- **Dead ends that bound the design space.** **PEP 335** (overloadable booleans): *Rejected*, and
  `__bool__` returning a non-`bool` raises `TypeError` — so `if`/`and`/`or` can never be
  intercepted, only type-lowered (compiled), replayed (Proposal 1d), or rewritten. **PEP 637**
  (keywords in subscripts): *Rejected*. **PEP 638** (syntactic macros): Draft, dormant since 2020 —
  no macro escape hatch is coming. **PEP 3150 / PEP 403** (block-as-expression): Deferred
  indefinitely. **Range patterns for `match`**: no PEP exists.

Net: nothing shipped or on the horizon changes the core rule — statement forms need a syntactic
marker + rewrite (or replay); expression forms work today.

## Prior art (summary; details in the appendix)

From the [[ideas/havogt/boundary-condition-syntax_research|cross-DSL survey]]:

- **Domain-splicing (like `concat_where`)** — the model to imitate: **YASK** `EQUALS … IF_DOMAIN sd0`
  with `first_domain_index(x)+5` / `last_domain_index(x)-3` (per-equation sub-domain condition —
  strikingly close to *both* `concat_where` and cartesian `region`); **Devito** `Eq(…, subdomain=…)`
  with per-axis `('left'|'right'|'middle', thickness)`; **GridTools/STELLA/Dawn** per-`interval`
  `apply()` overloads and `vertical_region(k_lo,k_hi)` (the lineage cartesian descends from);
  **Mathematica** `DirichletCondition[value, predicate]` and `Piecewise[{{v,c},…}, default]`
  (predicate-names-a-region; ordered first-true); the **FEM** family (`DirichletBC`, `ds(marker)`,
  `on(label)`, boundary-id). **ExaSlang** puts the BC in the field type (`Field<…,Neumann>`) and
  applies it with `apply bc to u`.
- **Pointwise boundary modes (a different feature)**: **Halide** `BoundaryConditions::{repeat_edge,
  mirror_image,constant_exterior}` + `clamp`; **Lift/RISE** `Pad(l,r,Pad.Boundary.{Clamp,Mirror,
  Wrap})`; **Pochoir** `Pochoir_Boundary_*` (a fallback *value* for out-of-domain reads, explicitly
  *not* a domain split).
- **Language pattern-matching**: **Rust**/**Swift** have first-class range patterns + guards +
  exhaustiveness — the ergonomic ideal; **Python `match` has none of the range part** (PEP 634/635),
  which is decisive for Proposal 2. **APL `⌺`** threads a boundary-offset token into the kernel —
  an orthogonal "the function is told where it sits" idea.

The recurring cross-system idiom is an ordered list of `(region predicate → branch)` pairs with a
default — which is precisely Proposals 4, 6, and 8, and what Proposals 1/3/5/7/9 are sugar over.

## Open questions / conflicts

- **Embedded vs compiled parity is the whole game.** The cleanest unification is a decoration-time
  AST rewrite to `concat_where` for the marker-bearing forms (3, 5, 7, custom-pattern 2). Is adding
  an `ast`-rewriting pass to the `next` frontend acceptable, given `next` deliberately avoids
  cartesian's "parse, never run" model? (Precedent that it is production-viable: pytest rewrites
  every collected test module's AST at import time, and gt4py.cartesian itself is
  `inspect.getsource` + `ast.parse`.) Proposal 9 shows a no-rewrite statement block exists, at the
  price of class-statement weirdness. This is the central decision.
- **Purity contract for replay.** Option (d) of Proposal 1 re-executes the body once per region;
  that is sound only if field-operator bodies are pure and deterministic. Today that is a de-facto
  convention — is it a contract we can state and enforce?
- **Precedence model.** Which single composition model — first-match, last-paint, or partition
  (see *How regions compose* above) — should the sugar commit to? Shipping more than one risks
  leaving users unsure which write wins; pick one, or make the difference syntactically obvious.
- **Interaction with [[ideas/havogt/scan-redesign|scan redesign]].** That note already proposes
  `concat_where(KDim == 0, bc, recurrence)` *inside scans* and strawman anchored indices
  `KDim.start`/`KDim.stop`. The sugar here is orthogonal but should share the condition vocabulary;
  a robust anchored end-index notation (`KDim == KDim.stop - 1` rather than `KDim == nlev - 1`)
  would make every proposal here read better.
- **Interaction with [[ideas/havogt/field-data-protocol|FieldData protocol]].** Unbounded boundary
  results (a scalar boundary extending past the compute domain) are an orthogonal *embedded
  representation* concern handled by that note; the frontend sugar does not affect it, but the
  runtime-builder variant of Proposal 5 must cooperate with whatever backs such a field.
- **Type checking partial assignments.** Proposals 3 and 5 introduce a variable that is only fully
  defined after several blocks/paints. How does FOAST type and SSA-check the intermediate states?
- **Scope.** Should the sugar also cover the pointwise `where` (mask) case, or stay
  `concat_where`-only? Keeping it domain-only avoids conflating the two selection models.

## Appendices

- [[ideas/havogt/boundary-condition-syntax_research|Cross-DSL prior-art survey]] — full,
  source-cited write-up of every framework above.
- [[ideas/havogt/boundary-condition-syntax_icon4py-survey|icon4py `concat_where` survey]] — the
  ~40 real call sites, the chained/nested idioms, condition shapes, and the inline if/else TODOs.
