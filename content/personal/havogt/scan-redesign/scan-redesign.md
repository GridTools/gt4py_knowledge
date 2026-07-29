---
title: Redesign of the vertical scan in gt4py.next
author: havogt
tags: [scan, vertical, reduction, boundary-conditions, windows, k-caches, fusion, embedded, jax, frontend]
created: 2026-06-11
status: draft
---

> **TL;DR** Replace `scan_operator` with a level-slice scan
> (`jax.lax.scan` semantics: explicit field-valued `init`, carry/output
> separation, range from inputs), K-local views for relative level access,
> a vertical window construct for scan + sub-reduce, fixed-op vertical
> reductions with the fold as the scan's final carry, and scan fusion as a
> toolchain obligation.

- **Status**: draft, for discussion.
- **Markers**: **NEEDS REFINEMENT** = examined this iteration, design still
  open; **NEEDS REVIEW** = section not yet revisited.
- **Updated**: 2026-06-16 (full detail in git history). Latest iteration: two
  output models (A free-output / B carry+projector, fold = projector-less); the
  scan/fold combinator alongside the decorator; vertical-forward range with a
  fixed-anchor rule; `include_initial` scoped to projector output, its N+1
  domain deferred to staggering; the staggering dependency; NEEDS-* markers.
  Doc then compressed; §3 reordered (constructs → fusion → rollout); a
  design-at-a-glance table added.
- **Companion documents**:
  [[personal/havogt/scan-redesign/scan-redesign_research|research notes]] (current
  implementation, prior art with sources, the scan + sub-reduce case study);
  [[personal/havogt/scan-redesign/scan-redesign_examples|worked examples]] (before/after for
  muphys, the PMAP stride-2 solve, the physics_patterns patterns, the
  sedimentation references).
- **Dependency (staggering)**: the boundary-condition and output design must be
  finalized *after* the staggering / half-level model
  ([[personal/havogt/dimension-generic-fields|generic dimensions]], `Staggered[D]`).
  Staggering shapes scan boundaries — `include_initial`'s n→n+1
  cell-to-interface emission (§3.3.2), the `KDim.start`/`KDim.stop` anchors
  across staggered grids (§3.3.1), and range deduction when inputs are
  cell-centred and outputs live on interfaces (§3.2, OQ2). Do not lock the
  boundary-condition design until staggering is settled.
  **NEEDS REFINEMENT** (blocked on staggering).

**Why.** Vertical recurrences in weather/climate models (vertical integrals,
tridiagonal solvers, sedimentation) are served today by `scan_operator`, which
has an implicit scan range, a scalar element-function that is slow in embedded
execution and needs type-promotion magic, and clumsy boundary handling. We
propose a *level-slice* scan with `jax.lax.scan`-compatible semantics —
call-site `init` (field-valued), carry/output separation, range from *inputs*
with explicit override — plus *K-local views* (relative level access without
delay-line carries), a *vertical window* for scan + sub-reduce, fixed-op
*vertical reductions* (the fold being the scan's final carry), and scan fusion
as a toolchain obligation. We reject whole-column kernels and GTScript interval
dispatch for v1, but keep the scalar-element body as an open variant (a
`scalar_operator`, elementally promotable with `map`); whether per-column
control flow uses `where` or native `if` is open (§4.1).

**Design at a glance.**

| Construct | Solves | § | Status |
| --- | --- | --- | --- |
| Level-slice scan + output models (A/B) | P2, P3 (scalar body; carry = output) | §3.1 | NEEDS REFINEMENT (model A/B, OQ4) |
| Scan range (vertical-forward, fixed anchor) | P1 (implicit range) | §3.2 | settled this iteration |
| Boundary conditions (peel / `init` / compose) | P4 (clumsy BCs) | §3.3 | NEEDS REVIEW; `include_initial` NEEDS REFINEMENT |
| Vertical windows (scan + sub-reduce) | P5 | §3.4 | NEEDS REVIEW |
| Vertical reductions & the fold | P7 | §3.5 | NEEDS REVIEW |
| Scan fusion (toolchain obligation) | P6 (mega-operators) | §3.6 | NEEDS REVIEW |
| Library layer (`cumsum`, `tridiagonal_solve`, …) | ergonomics | §3.7 | — |
| Execution model per backend | — | §3.8 | — |
| Migration | adoption | §3.9 | — |
| Body granularity (scalar vs slice; `map`) | P2 (control flow) | §4.1 | NEEDS REFINEMENT (OQ11–12) |

## 1. Background: the current scan and its problems

The current API is a scalar per-element function with carry threading
(`src/gt4py/next/ffront/decorator.py`):

```python
@gtx.scan_operator(axis=KDim, forward=True, init=0.0)
def cumsum(carry: float, val: float) -> float:
    return carry + val
```

Problems (file/line references in research §A):

- **P1 — Implicit scan range.** The vertical range comes from the *output*
  domain via a context variable (`closure_column_range` in
  `embedded/context.py`, consumed in `embedded/operators.py` and
  `iterator/embedded.py`) — the opposite of input-driven domain propagation,
  fragile (a placeholder `NamedRange` patched at call time), and ill-defined
  when the output is consumed on a different domain.
- **P2 — Scalar element body.** The pass maps scalars to scalars while call
  sites pass fields. Pro: per-column Python `if` in embedded. Cons: embedded is
  O(n_h · n_k) interpreter iterations
  (`embedded/operators.py::ScanOperator.__call__`); the type system needs
  promotion/demotion (`ffront/type_info.py::_scan_param_promotion`); and it maps
  to no array-level scan.
- **P3 — Carry is the output.** Whatever is carried is emitted at every level
  and vice versa. The ICON-like tridiagonal test
  (`tests/next_tests/integration_tests/multi_feature_tests/ffront_tests/test_icon_like_scan.py`)
  returns a *dummy bool field* because the `first_level` flag lives in the
  carry; conversely a final carry (e.g. surface precip flux) isn't retrievable
  without materializing a full 3D field.
- **P4 — Boundary conditions are clumsy.** Top/bottom cases are flags in the
  carry plus a branch run at *every* level, then repositioned by program-level
  slicing:

  ```python
  class State(NamedTuple):
      z_q_new: float
      w_new: float
      first_level: bool


  @gtx.scan_operator(axis=KDim, forward=True, init=State(z_q_new=0.0, w_new=0.0, first_level=True))
  def _scan(state: State, w: float, z_q: float, z_a: float, z_b: float, z_c: float) -> State:
      z_g = z_b + z_a * state.z_q_new
      z_q_new = (0.0 - z_c) * z_g
      w_new = z_a * state.w_new * z_g
      return (
          State(z_q_new=z_q, w_new=w, first_level=False)
          if state.first_level
          else State(z_q_new=z_q_new, w_new=w_new, first_level=False)
      )


  # program level: dummy output + slicing
  _solve_nonhydro_stencil_52_like(..., out=(z_q[:, 1:], w[:, 1:], dummy[:, 1:]))
  ```

- **P5 — No scan + sub-reduce.** Explicit sedimentation (Seifert–Beheng
  two-moment), PPM flux integration, ICON two-moment microphysics need, per
  level, a *reduction over a bounded window of neighbouring levels* — sometimes
  inside a recurrence. Today only emulable with hand-rolled ring buffers in the
  carry (research §C).
- **P6 — Production scans degenerate into one mega-operator.** The icon4py
  muphys graupel scheme (research §A.7) fuses *five* recurrences (four
  independent species sedimentations — rain/snow/ice/graupel — plus a
  temperature/energy recurrence coupled only via same-level outputs) into one
  `scan_operator` with a **21-component carry**, because separate scans can't
  share per-level intermediates or be relied on to fuse. Of the 21, two are
  pass-through outputs and `rho` is a *delay line* (carried only because the
  body can't read an input at k−1; the lookahead `t_kp1` is precomputed
  outside as `concat_where(KDim < last_lev, t(Koff[1]), t)`). The scalar body
  also forces `*_scalar` field-operator twins, recursion-limit workarounds, and
  ~650 lines of DaCe cleanup hooks. The older `_icon_graupel_scan` is the end
  state: a 28-tuple carry with an in-carry level counter,
  `sys.setrecursionlimit(350000)`, 13/28 outputs discarded.
- **P7 — No vertical reductions.** Field view reduces only over *local*
  (neighbor) dimensions; a column integral, column max (vertical CFL), or level
  search (tropopause, cloud top, PBL) must be a scan with
  accumulator/flag/counter carries emitting a full 3D field, or escapes to host
  NumPy (research §A.9).

## 2. Goals and non-goals

Goals: (1) explicit scan range, consistent with input-driven domain
propagation; (2) fast embedded (array ops per level) and a direct `jax.lax.scan`
mapping; (3) declarative, branch-free, carry-clean boundary conditions; (4)
carry and per-level output separate, final carry retrievable; (5) bounded
scan + sub-reduce (sedimentation, PPM) declaratively; (6) direct access to
neighbouring *input* levels without delay-line carries or external pre-shifts;
(7) cheap, composable small scans with fusion as a toolchain obligation (no more
21-component carries); (8) a migration path (`scan_operator` as sugar); (9)
column reductions (integrals, min/max, level/arg searches) as one-liners (P7).

Non-goals for v1 (see §4.7–4.8):

- **Parallel-in-k** (`associative_scan` lowering): sequential-k /
  parallel-over-columns is production-proven (GridTools, ICON-ACC); k-extents
  are ~80–137 while horizontal parallelism is abundant.
- **Data-dependent trip counts** (`while`-scans, early exit): bounded masked
  windows cover the known cases on GPU-friendly terms.
- **Distributed scans**: the vertical axis is never distributed.
- **Horizontal / global reductions** (mesh-wide sums, inner products): only the
  vertical is in scope — it is node-local (never MPI-distributed) whereas the
  horizontal is distributed, so vertical-only reductions stay collective-free
  and sidestep distributed-reduction semantics. Rejecting a reduction over a
  distributed axis is a required (implicit) guard.

## 3. Proposed design

### 3.1 Core primitive: a level-slice scan

The body maps *level slices* to level slices — a level slice is a field over
the call-site horizontal domain (K removed). It takes the carry and one slice
per scanned input, and returns the new carry plus (per the output model below) a
per-level output:

```
scan body:  (carry, x_k, ...) -> (carry', y_k)
scan call:  (xs..., init)     -> (final_carry, ys)
```

This is `jax.lax.scan` semantics (the ecosystem standard — also PyTorch's
higher-order `scan`, PyTorch/XLA; research §B.1, §B.3), the scan axis being the
declared vertical dimension instead of the leading array axis.

**Two output models** (both in scope; the call site chooses):

- **(A) Free output (`jax.lax.scan` form).** The body returns `(carry', y_k)`,
  computing `y_k` from the new carry *and* the level inputs — maximally
  expressive; an output needing a level input is never forced through the carry.

  ```python
  @gtx.scan(axis=KDim, forward=True)
  def cumsum_step(carry: float, x: float) -> tuple[float, float]:
      s = carry + x
      return s, s  # (new carry, output at this level)
  ```

- **(B) Carry + projector.** The body returns only `carry'`; a *projector*
  `carry' -> y_k` says what to write. Three cases on one mechanism: **identity**
  projector (default) = today's carry-is-output `scan_operator`; **non-identity**
  = a function/part of the carry (carry richer than output, e.g. carry
  `(running_sum, running_prod)` writing only the sum); **no projector** = the
  *fold* (only the final carry, §3.5). The projector sees only the carry, so an
  output depending on a level input must be threaded through the carry — the one
  gap vs (A).

  ```python
  @gtx.scan(axis=KDim, forward=True)  # default identity projector → carry is the output
  def cumsum_step(carry: float, x: float) -> float:
      return carry + x
  ```

(B) is the **minimal delta from today** and the IR-honest shape: the per-level
write is a pure function of the carry, which any backend column loop emits
directly. On the now-primary DaCe backend a scan already lowers to a sequential
loop region (`gtir_to_sdfg_scan.py`, whose GTIR scan form is slated to change
for exactly this), so the projector is the write statement inside the loop and
the fold is the same loop with no write; GTFN expresses it as a `scan_pass`
projector (`column_stage.hpp`). (A) earns its keep only when an output genuinely
depends on a level input — and the surveyed cases (cumsum, the Thomas sweep,
even muphys, whose per-level `(x, p)` outputs *are* carry components) are all
(B). Either way, carry/output separation removes P3's dummy outputs and makes
the final carry (surface fluxes) retrievable.

A consuming field operator (model (A)):

```python
@gtx.field_operator
def vertical_integral(
    x: gtx.Field[gtx.Dims[Cell, KDim], float],
) -> gtx.Field[gtx.Dims[Cell, KDim], float]:
    total, integral = cumsum_step(x, init=0.0)
    # total: Field[[Cell], float] — final carry (e.g. a surface flux)
    # integral: Field[[Cell, KDim], float]
    return integral
```

**Spelling: decorator and combinator.** The ITIR scan builtin is *already* a
higher-order function — `scan(step, forward, init)` applied to the args, axis
handled at the program level (`foast_to_gtir.py`). The `@gtx.scan` decorator is
sugar over it; the same shape as a **call-site combinator** is where the
projector, fold, and direction naturally live:

```python
final_carry, ys = gtx.scan(step, axis=KDim, forward=True, project=proj)(xs..., init=...)
final_carry      = gtx.fold(step, axis=KDim, forward=True)(xs..., init=...)
```

Both spellings are kept. The step is an ordinary traced operator — a slice
`field_operator` or a scalar `scalar_operator` (§4.1) — reusable outside a scan.
`gtx.fold` is the general sequential reduction underlying §3.5; its contract
must *not* foreclose a future parallel (`associative`) scan/reduce lowering
(§4.7). Picking model (A) vs (B) and the `project=`/`fold` spelling is OQ4.

> **NEEDS REFINEMENT** — output model (A) vs (B) and the `project=`/`fold` +
> decorator-vs-combinator spelling (open question 4).

Semantics and type rules:

- **Slice typing.** Arguments are level slices (or K-local views, below); the
  body runs once per level on whole slices, not once per element.
- **Carry structure** must be stable across iterations (as in JAX): the returned
  carry has the type/structure of `init` after broadcasting.
- **`init` is a call-site argument** — scalar, tuple, or a *field* (level slice,
  or broadcastable). A decorator default (`init=0.0`) remains possible for closed
  algorithms.
- **Control flow.** Data-dependent branching is `where`; Python `if` on field
  values is rejected at type checking (as in `field_operator`). Conditions on the
  *scan index only* get special treatment (§3.3).
- **Backward scans** keep `forward=`; outputs stay index-aligned with inputs
  (as `lax.scan(reverse=True)`).
- **Simple / carry-is-output** = model (B) identity projector — the migration
  target. **Fold** = model (B) with no output: only the final carry, k-less
  (research §A.9 tropopause argmin); the general order-pinned sequential
  reduction (§3.5). The discriminator is the projector (identity / non-identity /
  absent), not the return annotation — so simple and fold aren't two readings of
  one `-> carry` signature.

#### Typing of body arguments, and access to neighbouring levels

Two coupled questions: what the annotations *say*, and how the body reads an
input at k−1 / k+1. Today the second is "you don't" — muphys carries `rho` as a
delay line and precomputes `t_kp1` with a clamped `Koff[1]` shift (P6). Options:

- *(a) element-type-as-slice* — annotate `x: float` ("level slice of whatever
  the caller passes"): terse and polymorphic, but the annotations aren't the
  types the body computes with (a milder rerun of P2's promotion dishonesty),
  and no level access.
- *(b) honest field annotations* — `x: gtx.Field[Dims[Cell], float]` (or
  `Dims[*Hs]` with dimension variables): honest, but loses horizontal
  polymorphism or needs variadic-generic machinery
  ([[personal/havogt/dimension-generic-fields|generic dimensions]]); still no level
  access.
- *(c) **K-local views** (proposed)* — scanned inputs arrive as views of the
  column positioned at the current level, indexable by *relative* K-offset;
  indexing yields a level slice. Carry and outputs stay plain values.

```python
@gtx.scan(axis=KDim, forward=True)
def step(carry: float, t: gtx.KView[float], rho: gtx.KView[float]) -> float:
    # t[0]: current level; t[+1]: lookahead; rho[-1]: previous level
    return f(carry, t[0], t[+1], rho[0], rho[-1])
```

This is, deliberately, *iterator/local-view style along K only* — "field view
horizontally, local view vertically": a scan is the one construct with an
inherent current position. It isn't new in the stack — the ITIR scan pass
already receives shiftable, dereferenceable iterators (see the scan in
`tests/next_tests/unit_tests/iterator_tests/transforms_tests/test_domain_inference.py`,
which shifts its argument before deref); only ffront's `scan_operator` flattens
them to scalars. GTScript's `field[0,0,-1]`, Halide update definitions, and
GridTools fill-type k-caches are the same idea (research §B.6, §B.8). Semantics
and lowering:

- Offsets are static; the toolchain infers the per-input *vertical extent*
  (cartesian-style) and the scan range shrinks/masks at column ends — same as
  `Koff` shifts. The analysis is **guard-refined**: a read under an index
  conditional (§3.3) constrains only the levels where that guard holds (an
  `x[-2]` read in a last-level-only branch must not clip the column start),
  matching cartesian's per-`interval` precision (research §A.8).
- Inputs are read-only, so **lookahead (`t[+1]`) in a forward scan is legal** —
  muphys's external `t_kp1` becomes `t[+1]`. Reading *outputs* at an offset stays
  a carry concern (§4.6).
- Compiled backends map views to fill k-caches/registers; embedded maps `x[o]`
  at level k to the slice at k+o (one array op per level); JAX passes one extra
  pre-shifted `xs` leaf per offset — `lax.scan` semantics preserved.
- Sugar: an element-typed parameter (`x: float`) is an auto-dereferenced
  `x[0]`-only view, so simple bodies stay terse and option (a) becomes a
  documented projection of (c), not a lie.

We recommend (c) with the (a)-style sugar; (b) stays compatible (an author may
pin concrete slice types) but isn't required.

The ICON-like forward sweep from P3/P4, rewritten (boundary handling moved out,
§3.3):

```python
@gtx.scan(axis=KDim, forward=True)
def w_fwd_sweep(
    carry: tuple[float, float], z_q: float, w: float, z_a: float, z_b: float, z_c: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    z_q_km1, w_km1 = carry
    z_g = z_b + z_a * z_q_km1
    z_q_new = -z_c * z_g
    w_new = z_a * w_km1 * z_g
    return (z_q_new, w_new), (z_q_new, w_new)
```

No `first_level` flag, no dummy output, no per-iteration branch.

### 3.2 Scan range: vertical flows forward, horizontal stays backward

Domain inference in `next` is *backward* (the output domain back-propagates to
inputs, `infer_domain.py`); the scan even pulls its vertical range from the
output access domain (`_extract_vertical_dims`), the ITIR twin of embedded's
`closure_column_range`. That is sound only for *translation-covariant* ops,
where shifting the output request equals shifting the input accesses. A scan is
translation-covariant in its parallel dims but **not** in the scan dim: moving
the seed re-seeds the recurrence and changes every level downstream of it. So
the scan's domain splits:

- **Horizontal — backward, as usual.** A consumer's horizontal-shift request
  propagates into the producer like any field op.
- **Vertical — forward and anchored.** The vertical range is intrinsic to the
  scan (its K-bearing inputs, or an explicit `k_range`), never back-inferred:

```python
_, (z_q_res, w_res) = w_fwd_sweep(
    z_q, w, z_a, z_b, z_c, init=(z_q0, w0), k_range=KDim[1:]
)  # strawman override (k_range spelling: OQ3)
```

Rules:

- **Inputs bound the range; output + direction set what is needed.** The
  K-bearing inputs give the upper bound; the carry direction + requested output
  give the needed range (a forward scan writing up to level `b` must compute from
  the seeded end to `b` — it can skip the suffix beyond `b`, never the prefix the
  carry threads through; mirror for backward). Semantics = scan the available
  input range, write the requested sub-range; pruning the unneeded suffix is a
  *toolchain optimization*, valid only when the final carry and pruned outputs
  are dead.
- **The seed anchor is fixed and immovable.** A downstream *vertical* shift of
  the scan output (`scan_out(Koff[-1])`) is a read on the materialized K-range —
  a **vertical fusion barrier**, never a range expansion; below the anchor
  (forward) or above it (backward) is an out-of-range read (error / skip-mask),
  never an anchor move. So the scan range must be sized up front to cover every
  level a consumer needs. (Counterpart of the in-body offset reads: input offsets
  are KView §3.1, in-progress-output offsets are history carries §4.6.)
- **No K-bearing input ⇒ explicit range required** (pure recurrence; JAX
  `xs=None`/`length=`).
- **Guard-refined extent, not a naive intersection.** With KView offsets the
  range shrinks by the offsets, refined *per guarded region* — an input on a
  sub-range is legal if every region's reads stay in its domain. Production case:
  the PMAP back substitution (research §A.8), a full-column scan over a temporary
  defined one level short, well-formed because the last-level branch reads it
  only at offset −2. The formal fixpoint rule (anchors resolve against the range
  being deduced) is OQ9; `k_range` sidesteps it case by case.

Semantic change: "scan exactly over the requested output" becomes "scan over the
defined inputs, write the requested output". For the common cases (output range
== input range, or a sub-range like the `[:, 1:]` idiom) behaviour is unchanged
but now *predictable from the call site*, and the output-driven vertical range
(`_extract_vertical_dims`, `closure_column_range`) disappears.

### 3.3 Boundary conditions

> **NEEDS REVIEW** — not yet revisited this iteration (peeling guarantee,
> anchors); §3.3.2 is separately flagged.

Three mechanisms, in expected order of use:

1. **Peelable index conditionals** — the workhorse, for boundary values computed
   from the scanned fields themselves (the ICON-like case). `concat_where` on the
   scan index inside the body:

   ```python
   @gtx.scan(axis=KDim, forward=True)
   def step(carry: float, x: float) -> float:
       return concat_where(KDim == 0, bc_expr(x), recurrence(carry, x))
   ```

   Since the condition depends only on the loop index, the toolchain guarantees
   *loop partitioning*: compiled backends split the k-loop (GridTools/Dawn
   intervals, Halide loop partitioning — research §B.5, §B.6), embedded hoists the
   branch out. One function; no steady-state branch; no input slicing or output
   stitching — the peeled first iteration seeds the carry and writes the first
   level. Subsumes cartesian `interval(...)` in field-view clothing, reusing an
   existing builtin. (An earlier `at_level(f, KDim, 0)` accessor for feeding
   first-level values into `init` is unnecessary — this mechanism covers it.)
   Two refinements production cartesian demands (research §A.8):

   - **Anchored index expressions.** Conditions (and `k_range` bounds) may
     reference both ends: `KDim.start`, `KDim.stop` (strawman), so
     `KDim == KDim.stop - 1` is the last level and `KDim < KDim.start + 2` the
     first two. Bare negative indices aren't an option (`next` domains may start
     negative), so end-relative access is anchored. Anchors resolve against the
     scan range; muphys's runtime `last_lev` becomes redundant.
   - **The guarantee is loop partitioning, not first-iteration peeling.**
     Per-level evaluation stays the semantics (columns shorter than the boundary
     regions stay well-defined); the obligation: collect all `anchor ± const`
     split points in the body's index conditions, partition the range into maximal
     regions where every condition is statically resolved, and emit one branch-free
     loop (or peeled iteration) per region. Nested conditionals and multi-level
     boundary regions fall out of the same partition (Dawn interval splitting /
     Halide partitioning). Runtime-offset conditions stay legal but unguaranteed
     (in-loop branch). Extent analysis and range deduction run *per region* (§3.1,
     §3.2) — so partitioning is also what makes guard-refined well-formedness
     checkable.

2. **Field-valued `init`, optionally emitted — inclusive/exclusive scans.** When
   the boundary value comes from *outside* the scanned fields (surface pressure
   from another component, a prescribed model-top flux), it is the `init` — a
   scalar or k-less slice, evaluated once. `include_initial=True` additionally
   *emits* the init, one level upstream of the range — the array-API
   `cumulative_sum(include_initial=...)` design (research §B.2):

   ```python
   # hydrostatic-like integration: n cell increments -> n+1 interface values
   _, p = pressure_step(dz_rho_g, init=p_top, include_initial=True)
   ```

   With `y[k] = f(y[k-1], x[k])`, `y[k0] = init`, the "BC at the first level"
   pattern *is* an inclusive scan over `K[k0+1:]` whose init is emitted at `k0`;
   the *exclusive* scan is the same primitive with the last level dropped. Both
   ship as library wrappers (§3.7: `exclusive_cumsum`, `scan_with_initial`), not
   core variants. The emitted level is also the natural producer of *staggered*
   outputs — interface fluxes (n+1) from cell quantities (n), "the flux at the
   model top is the boundary condition". Two constraints, from §3.2 and the
   staggering dependency:

   - **Well-defined only for carry-is-output / projector scans (model (B),
     §3.1).** `include_initial` emits `project(init)`; in model (A) the emitted
     level has no level-input for `y = g(carry, x_k)`, so it's undefined.
     Restricted to projector scans.
   - **The +1 level's home is deferred to staggering.** Extending the output one
     level past the input range is §3.2's out-of-range write — *unless* the extra
     level lives on a different dimension. Cell→interface staggering is exactly
     that: n cells → n+1 interfaces on `Staggered[KDim]` (a dimension change), and
     §3.2 holds cleanly. Until staggering lands (`Staggered[D]`,
     [[personal/havogt/dimension-generic-fields|generic dimensions]]; the Dependency
     note above), `include_initial` on the same dimension is an array-API stopgap
     and this design is not final (OQ2). **NEEDS REFINEMENT** (blocked on
     staggering).

3. **Composition outside the scan** with `concat_where` over K regions — possible
   because the scan range is explicit (§3.2). Replaces the program-level
   `out=(z_q[:, 1:], ...)` idiom and is the escape hatch for boundary regions
   thicker than one level with genuinely different algorithms.

### 3.4 Vertical windows and scan + sub-reduce

> **NEEDS REVIEW** — windows not yet revisited this iteration (open question 6).

Targets `out[k] = reduce_{m in window(k)} f(inputs at k, k-m, k-m±1, ...)` —
explicit sedimentation (COSMO/ICON two-moment), PPM flux integration, ICON
two-moment microphysics. Case-study insight (research §C): when the window
extent is a compile-time constant `M` (max vertical Courant number in
sedimentation; max departure-cell count in PPM), this is **not a scan** — it is
a vertical stencil with a static window plus a reduction, and any recurrence
present (cumulative geometric height) is a separate ordinary `cumsum`. The
physics formulas also self-mask: out-of-reach contributions are exactly zero.

So: **vertical window offsets**, reusing the existing unstructured machinery
(local dimensions, sparse fields, `neighbor_sum`, skip-value masking) rather than
a new loop. A window offset is a compile-time K "connectivity":

```python
KWin = gtx.WindowOffset("KWin", source=KDim, offsets=range(0, -4, -1))  # k, k-1, k-2, k-3
KWinUp = gtx.WindowOffset(
    "KWinUp", source=KDim, offsets=range(-1, -5, -1), local_dim=KWin.local_dim
)  # k-1 ... k-4
```

`phi(KWin)` yields a field with a local dimension of static size 4, entry `m`
holding `phi[k + offsets[m]]` — exactly like a sparse neighbor field. The COSMO
"new explicit scheme" (research §C.1, eq. 12):

```python
@gtx.field_operator
def box_sedim_flux(v: CellKF, z: CellKF, phi: CellKF, dt: float) -> CellKF:
    z_up = minimum(z(KWinUp), z + abs(v(KWin)) * dt)  # z broadcasts over the window dim
    z_low = minimum(z(KWin), z_up)
    return -neighbor_sum(phi(KWin) * (z_up - z_low), axis=KWin.local_dim)
```

a direct vectorized transcription of the reference (`box_sedim_naive_func`,
§C.2) — no ring buffer, no carry, no index arithmetic. Design points:

- **Boundaries.** Entries past the field's K-range are skip-masked (neutral
  element) — the right default, since the physics wants no contribution from
  above the model top. (Or shrink the output by intersection, as for `Koff`
  shifts.)
- **Inside scans.** Individual offsets inside a body are already KView (§3.1:
  `x[-2]`, `x[+1]`); the window adds the *reduction over the window*
  (`neighbor_sum` over the local dim, allowed on view arguments too, `x(KWin)`),
  for windowed contributions that interact with running state. Backends lower both
  to fill k-caches / ring buffers (research §B.6); embedded, to shifted views —
  the rolling-buffer transform done by hand in the gist (`box_sedim_scanlike`),
  now automatic.
- **Data-dependent extents** = a static bound + masking (`where`/self-masking),
  the GPU-friendly form. The work-efficient *scatter* formulation is intentionally
  unsupported: it writes multiple output levels per step, breaking
  one-output-per-level and any future parallel-in-k (§C.3–C.4).
- **Reading the in-progress output** (`ys(Koff[-1])`, GTScript/Halide style) is
  not core: an order-m recurrence is an m-tuple/windowed carry. Offset-output reads
  stay later sugar over a window carry (§4.6).

### 3.5 Vertical reductions and the fold form

> **NEEDS REVIEW** — vertical reductions not yet revisited this iteration
> (open question 10).

Field view reduces only over local dimensions (P7). The missing vertical
reduction splits into two tiers (research §B.10):

1. **The general sequential fold is already in the design** — a scan whose
   per-level output is discarded, returning only the final carry (§3.1 fold form).
   Order-pinned, arbitrary data-dependent combine, bit-reproducible by
   construction. muphys's surface fluxes are this (§3.6.3).
2. **Fixed-op reductions** as dimension-removing builtins over a non-local axis:
   `sum`, `max`, `min` (extending — or renaming, OQ10 —
   `neighbor_sum`/`max_over`/`min_over`), plus `argmin`, `argmax`, `any`, `all`.
   No user-defined combine (for a custom combine, write the fold). Mirrors the
   array API (every fixed-op reduction, no general reduce/scan; research §B.2,
   §B.10) and gives embedded a one-call `xp.sum(...)`.

```python
twv = gtx.sum(qv * rho * dz, axis=KDim)  # column integral: Field[[Cell], float]
itpl = gtx.argmin(t, axis=KDim)  # level search: Field[[Cell], int32]
rcs = gtx.sum(  # masked, per-column variable depth (research §A.9)
    where(gtx.index(KDim) < nroot, dz * gx, 0.0), axis=KDim
)
```

- **Order is unspecified** (the XLA `Reduce` contract; NumPy `sum` is itself
  pairwise). Compiled backends accumulate sequentially in the vertical executor
  (GridTools has no reduction construct — a column sum *is* a forward
  k-accumulation; research §B.10), so only the embedded array path may
  reassociate. Codes that must pin the order write the fold (tier 1) — PSyclone's
  reproducible-reduction answer.
- **Range rules are §3.2's** (input K-range, `k_range` override, guard
  refinement); the output loses the K dimension.
- **`gtx.index(KDim)`** exposes the level index as a value — required for masked
  variable-depth reductions (`k < nroot`, PBL heights; research §A.9) and for
  arg-reductions as raw folds (the argmin carry `(t_min, k)`). GTIR already has
  `index(dim)` (`iterator/builtins.py`); only the ffront exposure is missing.
  Anchored conditions (§3.3) are for *static* dispatch; `gtx.index` for
  *data-dependent* level arithmetic.

**Combined scan + reduce** — four compositions, none a new user construct:

| Composition | Example | Answer |
| --- | --- | --- |
| fold as result | muphys surface fluxes | final carry (tier 1) |
| reduce over a scan's output | CAPE: parcel scan emits buoyancy; masked integral | compose; §3.6 fuses into one k-loop |
| reduce feeding a scan | column total used per level | two passes (true dependence) |
| bounded sub-reduce in a scan | sedimentation | §3.4 windows |

Row 2's IR target is a column executor hosting scan *and* reduction channels in
one vertical loop — Futhark's `screma` (research §B.10); GTFN's `ScanExecution`
grows reduce slots, DaCe adds loop-region accumulators. Row 3 ("reduce, then
consume the total at every level") has two backend strategies: (a) *materialize
k-less* — the reduction writes a k-less temporary the consumer reads broadcast
(two traversals, one 2D temp, consumer direction free); (b) *carry-propagate* — a
second pass in the *opposite* direction starts where the reduction ended, keeping
the total in the carry (register / flush k-cache), so a fused consumer is fully
register-resident (the classic forward-scan + backward-scan idiom, now a backend
peephole). (b) wins only when the consumer runs (or can be flipped to run)
opposite to the reduction; unfused it materializes a 3D temporary and loses to
(a). (b) also bridges backends that can't yet write k-less outputs from a column
pass (emit the total at every level, alias one). Both are sequential in k, so
reproducibility is unchanged. (This backend-created second pass is *not* the
opposing-direction fusion of OQ5.)

### 3.6 Composability and scan fusion

> **NEEDS REVIEW** — fusion guarantees not yet revisited this iteration
> (open question 5).

muphys (P6, research §A.7) shows what happens when small scans aren't viable:
users fuse by hand. Four-part response:

1. **Small scans are cheap to write.** Slice bodies call ordinary field operators
   directly — the `*_scalar` helper twins (`_fall_speed_scalar`,
   `_T_from_internal_energy_scalar`) disappear. Carry shrinks to the genuinely
   recurrent state: KView removes the delay-line slots (`rho` at k−1, the `*_kup`
   caches), carry/output separation removes pass-through slots, index conditionals
   remove in-carry level counters and first-iteration flags.
2. **Fusion is the toolchain's obligation, not the user's.** Scans over the same
   axis/direction/range with same-level or carried dependencies (the muphys
   temperature scan consumes the species scans' same-level flux outputs) fuse into
   one k-loop. GTFN already merges scans at the stencil level; DaCe fuses loop
   regions. The redesign specifies it: *the muphys decomposition — four species
   scans + one temperature scan, written separately — must compile to a single
   vertical loop* (an acceptance test). It extends to scan → reduction chains over
   the same range (§3.5): a parcel-ascent scan + masked column integral (CAPE) is
   one k-loop with an extra accumulator — Futhark's `screma`/`redomap` (research
   §B.10) is the precedent that a fused map+scan+reduce executor is standard.
3. **Surface outputs are final carries.** muphys's `pr, ps, pi, pg, pre` (sliced
   from the last level of full-column outputs today) are simply components of the
   returned final carry (§3.1), allocated as such.
4. **Work-skipping becomes a transformation concern.** muphys's
   `if current_level_activated:` (skip all physics where nothing is active,
   identity else-branch) leaks into user code and needs ~650 lines of DaCe hooks
   to remove the self-copies. In the slice world it is `where`-masking;
   masking optimizations including "else-branch is identity" detection move into
   the toolchain (where the muphys TODO already wants them).

### 3.7 Library layer

Ship the common cases so users rarely write raw scans:

- `gtx.lib.cumsum(x, axis=KDim, forward=True, include_initial=False)` and
  `exclusive_cumsum(...)` — lowered to `xp.cumulative_sum` (array API 2023.12) in
  embedded/array backends, a trivial scan elsewhere.
- `gtx.lib.scan_with_initial(step, xs, init)` — the §3.3.2 inclusive-with-emitted-
  init pattern, named.
- `gtx.lib.tridiagonal_solve(a, b, c, d)` — Thomas (forward sweep + back
  substitution) as *one* operation: naive backends → two scans; optimized GPU →
  vendor (cuSPARSE `gtsv2`-family) or per-column Thomas in registers. Precedent:
  `jax.lax.linalg.tridiagonal_solve`, LFRic CMA operators (research §B.7). Pins
  numerics and removes the most awkward scan from user code. Textbook case only —
  production solves are often pre-factored (PMAP, §A.8, hoists the elimination
  coefficients out of a GCR iteration, re-running only the sweeps) or non-standard
  (stride-2 interleaving), which stay raw-scan (so the primitive must stay
  ergonomic for hand-written sweeps); a `factor`/`solve_factored` pair is a possible
  later extension.
- `gtx.lib.windowed_scan(...)` — §3.1 + §3.4 for bounded-history recurrences.

### 3.8 Execution model per backend

| Backend | Lowering |
| --- | --- |
| Embedded NumPy/CuPy | Python loop over k (~100), one vectorized slice expression per level — O(n_k) array calls instead of O(n_h · n_k) scalar calls. |
| Embedded JAX | Direct `jax.lax.scan` over K (body is slice-level) — scans become `jit`/`vmap`/`grad`-compatible; vertical solvers differentiable for free. |
| GTFN | Unchanged model: the slice body is scalarized per column in the existing `vertical_executor` k-loop. Peelable conditionals → interval splitting; windows → k-caches. |
| DaCe | As today: sequential loop region nested in horizontal maps (`gtir_to_sdfg_scan.py`); carry/output separation removes dummy-output temporaries. |

The frontend simplifies: `_scan_param_promotion` and the scalar↔field
special-casing (`ffront/type_info.py`) disappear; the ITIR `scan` builtin gains
an explicit init/range and a carry/output pair instead of carry-is-output.

### 3.9 Migration

- `@gtx.scan_operator(axis, forward, init)` stays one deprecation cycle as sugar.
  If the slice body wins (§4.1), the scalar body becomes a simple-form slice body
  (Python `if` on data values rewritten to `where` where possible, else a
  deprecation error with a fix-it hint); if the `scalar_operator` variant is
  retained, migration is the identity — old scalar bodies become `scalar_operator`
  steps passed to the combinator, `if` and all.
- The `out=field[:, 1:]` idiom keeps working (an output restriction under §3.2).
- Scan tests (`test_icon_like_scan.py`, `test_execution.py::*scan*`) are the
  migration spec; each gets a rewritten twin before the old path is removed.

## 4. Discussion: decisions, pros and cons, alternatives

### 4.1 Body granularity: scalar element vs level slice (vs whole column)

> **NEEDS REFINEMENT** — body granularity / `if`-vs-`where`, `scalar_operator`,
> elemental `map`, and the JAX-lowering performance comparison (open
> questions 11–12).

An **open design axis, not a settled choice.** Both the scalar-element body
(`scalar_operator`, native `if`) and the level-slice body (`field_operator`,
`where`) are on the table; the crux is `if` vs `where`.

| | scalar element (`scalar_operator`) | level slice (`field_operator`) | whole column (§4.8) |
| --- | --- | --- | --- |
| per-column `if` | native Python `if` — but **not** under the `vmap ∘ lax.scan` lowering below | `where` only (branch-free) | native |
| embedded NumPy/CuPy | array ops if branch-free; Python per element only where `if` is data-dependent | array op per level | array op per col. batch |
| JAX lowering | `vmap`(orthogonal) ∘ `lax.scan` — canonical NWP pattern (§B.1), **not** "none" | array `lax.scan` | manual |
| reuse in field context | `*_scalar` twins (P6) — **removed once `map` exists** (below) | calls field operators directly | n/a |
| type system | scalar→field lifting made explicit by `scan`/`map` (replaces hidden P2 promotion) | plain broadcasting | column types |
| backend codegen | per-column loop | scalarize per column (mechanical) | compiler must rediscover |

**`if` vs `where` (the crux).** The slice body forces branching into `where`:
branch-free and GPU-friendly, but both arms are evaluated. The scalar body keeps
native `if`: how column physics is naturally written (muphys
`if current_level_activated`, branchy microphysics), and on a GPU the skip pays
off when a whole warp of columns is inactive. The earlier draft's "always
`where`, treat work-skipping as a masking transformation (§3.6.4)" stance is one
position; the counter-position is that some physics is clearer and faster with
`if`, so the scalar body stays first-class. Both kept; the decision is open (§5).

**Measured: native `if` and the `vmap ∘ lax.scan` lowering are mutually
exclusive.** The scalar body's `if` survives eager embedded execution
(NumPy/CuPy, eager JAX) and the compiled backends' per-column loop — that is
where the warp-skipping argument holds — but not under the JAX lowering this
table pairs it with, because the predicate is a tracer and Python `if` needs a
concrete bool: `TracerBoolConversionError: Attempted boolean conversion of traced
array with shape bool[]`. The `bool[]` matters: `vmap ∘ lax.scan` *does* restore
scalar shape (the slice-body failure is the different `ValueError: truth value of
an array with more than one element is ambiguous`), but not concreteness, and no
JAX transform makes a data-dependent branch concrete. So `if` vs `where` is not
purely a body-granularity axis, it is also a lowering axis: under JAX the body
must be branch-free whichever granularity wins, and the scalar body's `if`
advantage argues for the compiled backends only.

**`scalar_operator` + elemental `map`.** Two historical arguments against scalar
bodies dissolve once a scalar kernel can be *lifted* explicitly. We float a `map`
that promotes a `scalar_operator` to act point-wise on fields — Fortran
`elemental` procedures are the exact reference (one scalar definition, callable on
scalars or arrays). With `map`: the `*_scalar` twin explosion (P6) disappears —
write the kernel once, `map` it for field use and pass it to `scan`/`fold` for
column use — and the `_scan_param_promotion` "dishonesty" (P2) becomes explicit,
named lifting rather than hidden machinery. So `scalar_operator` is not just a
migration shim: with `map` it is a reusable, composable primitive that *also*
enables scalar `if`. (`map` spelling/semantics is OQ12.)

**Performance is unsettled and must be measured.** The same scan semantics admits
at least three JAX lowerings whose GPU performance can differ substantially: (i)
an **array `lax.scan`** with horizontal-slice-valued carry (slice body); (ii) a
**`vmap`(orthogonal axes) ∘ `lax.scan`** with scalar carry (scalar body,
JAX-canonical); (iii) a **full-field** version with explicit per-level array ops
and no scan primitive. The winner for NWP shapes (k≈100, large horizontal) is open
and a prerequisite for choosing the default body granularity (§5).

A first data point, CPU only, 64×64×80, all variants verified against a scalar
reference loop: array `lax.scan` **0.70 ms**, `vmap ∘ lax.scan` **0.81 ms**,
today's per-element double Python loop **122 ms**. So (i) and (ii) sit within
noise of each other at ~150× the status quo, i.e. keeping the body scalar costs
nothing measurable here. This does not answer the question the paragraph poses —
it is CPU rather than GPU, and 4096 columns is small against NWP horizontals —
but it does suggest the choice will not be decided by (i)-vs-(ii) throughput
alone.

Whole-column kernels (write the k-loop yourself) stay the §4.8 escape hatch:
maximal freedom, but the toolchain loses the recurrence structure (no interval
splitting, k-caching, `lax.scan` mapping, future associative lowering), so out of
scope for v1.

### 4.2 Carry/output separation

Pro: removes dummy outputs (P3), gives final carries (surface fluxes), matches
JAX/PyTorch signatures, lets the Thomas sweep carry `(c', d')` while emitting what
back-substitution needs. Con: one more tuple level in the general form, mitigated
by the simple form (§3.1). Rejected: keep carry-is-output and add a separate
final-carry accessor — fixes only half of P3 and diverges from the ecosystem
signature.

### 4.3 Range from inputs + override

Pro: consistent with field-view ops, call-site-predictable, removes the
context-variable plumbing, composable with `concat_where`. Con: pure recurrences
need an explicit range (acceptable — no other source of truth); behaviour change
for exotic cases where the output domain exceeded the inputs (was implicit and
error-prone, now an explicit error). Rejected: keep output-domain deduction — the
"not clear how this should be done" problem.

### 4.4 Boundary conditions: init + peelable conditionals vs interval dispatch

We considered GTScript-style interval dispatch as the API
(`@gtx.scan(intervals={KDim[0:1]: body0, KDim[1:]: body})`). Pro: explicit, proven
in cartesian. Con: multiplies function definitions for what is usually one
differing expression, interacts badly with carry typing (every body must agree),
and `concat_where`-in-body + field `init` express the same with existing
vocabulary. The PMAP preconditioner (research §A.8) — five intervals across two
sweeps — transcribes to nested anchored conditionals without loss (examples §7).
The peeling *guarantee* (not just optimization) is load-bearing — it must be
specced and tested, else users are back to per-iteration branches.

### 4.5 Windows via local dimensions vs a dedicated loop construct

Reusing sparse-field machinery: no new IR, embedded nearly free (stacked shifted
views), masking inherited from skip values, and it works *outside* scans too
(where most uses live — §C). Con: window size is compile-time static; two offsets
sharing a local dimension (`KWin`/`KWinUp`) need a small extension to offset
declarations. Rejected: a bounded `for m in range(M)` body — more familiar, but
introduces general loops into the DSL with all their analysis burden, for no added
expressivity.

### 4.6 Offset reads of the in-progress output: `history` carries

KView (§3.1) covers offset reads of *inputs*. For the *in-progress output*
(GTScript's `field[0,0,-1]` in FORWARD, Halide update definitions) the core stays
first-order: an order-m recurrence is an m-tuple carry (research §B.8) — easier to
type, lower, differentiate. But the PMAP preconditioner (§A.8) shows order > 1
isn't exotic: its forward elimination reads `ff[0,0,-2]`, its back substitution
`rp[0,0,2]` — two interleaved even/odd chains — and a 2-tuple carry reintroduces
for outputs the delay-line idiom this proposal removes for inputs. So the sugar is
part of the design: `@gtx.scan(..., history=m)` presents the carry as a read-only
K-local view of the last m carries (`ff[-1]`, `ff[-2]` forward; `rp[+1]`, `rp[+2]`
backward); the body returns just the new value (simple form: also the per-level
output; general form: the view is of the carry). `init` then supplies m levels
(structural-only when a boundary region seeds the recurrence, §3.3.1). It desugars
to the m-tuple carry — no new IR — and lowers to existing k-cache machinery
(GridTools flush caches, window m).

### 4.7 Parallel-in-k: deliberately not in v1

Linear recurrences and tridiagonal sweeps are associative-reformulable (Blelloch;
`lax.associative_scan`; PyTorch `associative_scan` — research §B.1, §B.3, §B.8),
but NWP k-extents are ~100, horizontal parallelism saturates GPUs (the
ICON/GridTools production model is sequential-k), the reformulation changes
floating-point results (reproducibility), and PyTorch's still-prototype status
after 2+ years shows the cost. The door stays open: the primitive's semantics are
sequential, and an `associative=True` hint or pattern recognition (cumsum, linear
recurrence) can later unlock parallel lowering with no user-visible change
(Futhark `scan`/`loop`, Halide `rfactor`).

### 4.8 `while`-scans: rejected

Data-dependent trip counts (the COSMO paper's work-efficient inner `while`) don't
vectorize over columns, lose `vmap`/AD in JAX, and complicate every backend. The
masked bounded window is the vectorized form production codes use anyway (the
paper's own NEC-vectorized variant effectively does this). Unbounded extents
pre-chunk or take multiple passes.

The SHiELD melting pattern (research §A.9) marks the documented boundary: an inner
data-dependent m-loop with `break`s, scatter writes at both k and m, and
sequential feedback in *both* loops (`qice[k]` exhausted across the inner loop,
`temperature[m]` re-read by later k iterations). It is not reformulable as a masked
bounded-window gather without changing the algorithm (the order-dependent `min`
caps read intermediate state). The honest answer: reformulate (COSMO's boxtracking
did, §C.1) or, if such patterns accumulate during migration, revisit the §4.1
whole-column kernel as an explicit escape hatch rather than stretching the scan.

## 5. Open questions

01. **K-view annotation spelling and honesty.** `gtx.KView[float]` strawman; does
    the `x: float` auto-deref sugar (§3.1) stay honest enough for frontend style,
    or should views be mandatory in signatures whenever offsets are used?
02. **Staggered outputs of `include_initial`.** The emitted-init level produces
    n+1 outputs from n inputs (interface vs cell-center). Interaction with
    half-level dimensions (icon4py's `KHalfDim`) and §3.2 range deduction needs a
    dedicated design; the statically typed staggering (`Staggered[D]`) of
    [[personal/havogt/dimension-generic-fields|generic dimensions]] is one candidate.
03. **Range override syntax.** `k_range=KDim[1:]` vs the `concat_where` domain API
    (`KDim > 0`) vs program-style named ranges. Should align with field-view domain
    slicing and share the `KDim.start`/`KDim.stop` anchors of §3.3 (research §A.8).
04. **Choosing the output model (§3.1 (A) vs (B)).** (A) returns `(carry, y)`; (B)
    returns the carry alone plus a projector (identity / non-identity / absent =
    fold). Reading this off the return shape is subtle ((A) collides with a
    2-component carry), so an explicit marker (a `project=` / `fold` spelling, tied
    to the decorator-vs-combinator question) may be safer. The projector already
    disambiguates simple-vs-fold; only (A)-vs-(B) remains.
05. **Fusion guarantees.** §3.6 makes scan fusion an obligation — spec or
    best-effort? Which compositions are *guaranteed* (same range/direction,
    same-level coupling) vs merely attempted (different ranges, opposing directions
    as in Thomas forward+backward)? Pin before users rely on decomposed scans being
    performance-neutral.
06. **Window declaration ergonomics.** Final `WindowOffset` spelling, shared local
    dims, and interaction with `offset_provider` (these are compile-time, not
    runtime, connectivities). The window's local dimension is exactly a
    connectivity-carrying local dimension per
    [[personal/havogt/dependent-local-dimensions|dependent local dimensions]]; the two
    designs should agree.
07. **Naming.** `gtx.scan` vs `scan_operator`; `KView` vs `ColumnView`;
    `WindowOffset` vs `KWindow`; `gtx.lib` vs `gtx.stdlib`; `history=` vs
    `carry_window=` (§4.6); `KDim.start/.stop` vs `gtx.first/last(KDim)` (§3.3).
08. **NamedCollection carries.** Today's `NamedTuple` carries (muphys's nested
    `IntegrationState`) should keep working in the slice world (collections of
    slices); verify against `arguments.extract` handling.
09. **Guard-refined range deduction, formally.** §3.1–§3.3 want the scan range to
    be the maximal range on which every guard-refined read is satisfied, but
    anchored guards resolve against the very range being deduced — a fixpoint.
    Existence/uniqueness of that maximum, the error story when a guarded read is
    unsatisfiable, and when `k_range` is *required* need a precise spec. The PMAP
    back substitution (examples §7) is the acceptance test.
10. **Vertical-reduction spelling and order.** Extend
    `neighbor_sum`/`max_over`/`min_over` to non-local axes vs new
    `gtx.sum`/`gtx.max`/`gtx.min`; the `gtx.index(KDim)` exposure; and whether
    unspecified association needs an explicit opt-in/opt-out flag or "write the
    fold" suffices as the reproducibility story (§3.5).
11. **Body granularity, `if` vs `where`, and `scalar_operator` (§4.1).** Keep the
    scalar-element body first-class — native `if` (compiled backends and eager
    embedded only — *not* under `vmap ∘ lax.scan`, §4.1), the `vmap ∘ lax.scan`
    lowering, elemental `map` reuse — alongside the slice body, or commit to
    slice-only with `where`? The empirical JAX lowering comparison (array
    `lax.scan` vs `vmap ∘ lax.scan` vs full-field) should drive the default; a
    first CPU measurement (§4.1) puts the two scan lowerings within noise, so the
    decision likely turns on GPU behaviour and on how much the `if` matters for
    the compiled backends.
12. **Elemental `map` (§4.1).** Spelling and semantics of promoting a
    `scalar_operator` point-wise over fields (the Fortran-`elemental` analog), how
    it composes with `scan`/`fold`, and whether it fully subsumes the `*_scalar`
    twin pattern.

## 6. Consequences

Easier: writing and reading vertical solvers (no flag-in-carry idioms, no delay
lines, no `*_scalar` twins); embedded debugging at realistic sizes; JAX
differentiation of vertical physics; declarative sedimentation/PPM; column
diagnostics (integrals, level searches) as one-liners; reasoning about scan
domains; decomposing muphys-scale schemes into per-process scans without a
performance penalty (given §3.6). Tracing also shallows out (slice bodies replace
the deeply recursive scalar expansion behind icon4py's `sys.setrecursionlimit`).
Harder/cost: a frontend+IR+all-backends change with a deprecation cycle; per-column
`if` users may need to move to `where` if the slice body wins the §4.1 decision
(work-skipping then becomes a masking transformation); the peeling guarantee, KView
extents, window lowering, and the fusion obligation are new backend responsibilities
(GTFN intervals, scan merging, and k-caches already exist; DaCe needs loop-region
splitting and fusion).

If accepted, the decisions in §3.1–§3.4 and §3.6 should each land as a numbered ADR
under `docs/development/ADRs/next/` referencing this proposal; this document stays
the design rationale.
