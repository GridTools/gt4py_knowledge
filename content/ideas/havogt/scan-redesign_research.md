---
title: "Vertical scan redesign: research notes"
author: havogt
tags: [scan, vertical, reduction, prior-art, research]
created: 2026-06-11
---

- **Status**: companion document to
  [[ideas/havogt/scan-redesign|the scan redesign proposal]]

This document records the research behind the proposal: an analysis of the
current implementation (§A), a survey of prior art with sources (§B), and a
case study of vertical scan + sub-reduce algorithms (§C). §D condenses
everything into the design drivers used in the proposal.

## A. The current scan in `gt4py.next`

### A.1 User-facing API and frontend representation

`scan_operator` (`src/gt4py/next/ffront/decorator.py`, ~lines 814–864) takes
`axis`, `forward`, `init` at decoration time and builds a
`foast.ScanOperator` node (`ffront/field_operator_ast.py`, ~lines 239–247)
holding `axis`, `forward`, `init` as constants plus the scalar definition
function. There is no dedicated ITIR node: FOAST lowering
(`ffront/foast_to_gtir.py`, ~lines 121–148) produces
`as_fieldop(scan(definition, forward, init))(*args)` around the `scan`
builtin (`iterator/builtins.py`). The type is
`ts_ffront.ScanOperatorType(axis, definition)`
(`ffront/type_specifications.py`).

### A.2 Scalar↔field promotion

Because the definition is scalar but call sites pass fields,
`ffront/type_info.py` carries dedicated machinery
(`_scan_param_promotion`, ~lines 166–213; carry special-casing in argument
canonicalization, ~lines 115–128) that promotes scalar parameter types to
field types with dims harvested from the actual arguments, and demotes on
return. This is the "weird promotion/demotion" cited in the problem
statement: the declared types of the scan pass are systematically not the
types it is called with.

### A.3 Embedded execution paths

There are two:

1. **Field-view embedded** (`embedded/operators.py::ScanOperator.__call__`,
   lines 34–88): computes `domain_intersection` of the args, takes the scan
   range from the `closure_column_range` context variable (set in
   `field_operator_call` from the **out** field's domain, lines 97–133),
   then runs

   ```python
   for hpos in iterate_domain(non_scan_domain):  # every horizontal point
       for k in scan_range:  # every level
           acc = self.fun(acc, *scalar_args_at(hpos, k))
   ```

   i.e. O(n_horizontal · n_k) Python interpreter iterations with per-element
   tuple packing/unpacking — the documented embedded-performance pain.
   Notably line 58–60: if the scan axis is not in the inputs at all, it is
   *added* from the context range ("even if the scan dimension is not in the
   input, we can scan over it") — the pure-recurrence case that makes
   input-based deduction non-trivial.

2. **Iterator embedded** (`iterator/embedded.py`, `scan` builtin at ~lines
   1625–1640, used by the `roundtrip` backend): same loop structure at the
   iterator level, column range from
   `embedded_context.get_closure_column_range()`, populated by
   `_extract_column_range` (~lines 1717–1728) which patches a placeholder
   `NamedRange` whose dim is the column axis with start/stop from the
   output domain.

In both paths the **scan range is back-derived from the output domain via a
context variable** — the design wart the proposal's §3.2 removes.

### A.4 Compiled backends

- **GTFN** (`program_processors/codegens/gtfn/codegen.py`): dedicated
  `Scan`/`ScanExecution`/`ScanPassDefinition` nodes rendered into
  `vertical_executor(axis)().arg(...).scans.execute()` — a per-column
  sequential k-loop in the GridTools C++ execution model, with scan fusion at
  the stencil level.
- **DaCe**
  (`program_processors/runners/dace/lowering/gtir_to_sdfg_scan.py`): no
  native scan in DaCe; the scan dimension becomes a nested SDFG with a
  sequential loop region, horizontal dimensions become parallel Maps, scan
  arguments are passed as full columns by value.
- Domain inference (`iterator/transforms/infer_domain.py`, ~lines 487–506)
  special-cases scans to retain vertical dimensions when propagating domains.

Both compiled backends already implement exactly the execution model the
proposal keeps (sequential k per column, parallel horizontal), so the rewrite
is frontend/IR-heavy, not backend-heavy.

### A.5 Boundary-condition evidence

`tests/next_tests/integration_tests/multi_feature_tests/ffront_tests/test_icon_like_scan.py`
(an ICON `solve_nonhydro` stencil-52 extract) is the canonical specimen, and
exhibits three pain points at once:

- a `first_level: bool` flag inside the carry `NamedTuple`, branched on at
  *every* level;
- a **dummy bool output field** materialized over the full 3D domain, only
  because carry == output forces the flag to be emitted (one test variant
  even exists specifically because backends needed the unused output
  written);
- program-level slicing `out=(z_q[:, 1:], w[:, 1:], dummy[:, 1:])` to place
  the result, coupling the scan to its caller's indexing.

ADR 0022 (`docs/development/ADRs/next/0022-Limitations-of-embedded-concat_where.md`)
documents the `concat_where` domain-expression API (`KDim == 0`, `KDim > 0`,
contiguous 1D regions only) — the existing vocabulary the proposal reuses for
peelable in-body conditionals and outside-the-scan composition.

### A.6 `gt4py.cartesian` for contrast

GTScript (`src/gt4py/cartesian/gtscript.py`) provides
`with computation(FORWARD|BACKWARD|PARALLEL)` and `with interval(a, b)`:
non-overlapping vertical regions each with their own statements, and offset
reads of already-updated fields (`field[0,0,-1]` in FORWARD) for
recurrences. Boundary conditions are separate interval blocks — no flags, no
runtime branches — and backends (GridTools) implement them as split k-loops
with k-caches. This is the proven UX target for boundary handling; what it
lacks is the explicit carry/init story and any array-ecosystem interop.

### A.7 Production evidence: icon4py muphys and microphysics

Survey of `C2SM/icon4py` at commit `515763c` (2026-06-12),
https://github.com/C2SM/icon4py — the largest production users of
`scan_operator`. GitHub code search finds 8 `scan_operator` uses repo-wide;
the two microphysics packages dominate.

**muphys** (`model/atmosphere/subgrid_scale_physics/muphys/`, port of the
one-moment "muphys" scheme): exactly one scan, `_precip_and_t` in
`implementations/graupel.py`, and it is the canonical *fused* scan:

- **Carry**: nested NamedTuples (`IntegrationState` containing four
  `PrecipStateQx(x, p, vc, activated)` — rain/snow/ice/graupel — plus
  `TempState(t, eflx, activated)`, `rho`, `pflx_tot`), i.e. **21 scalar
  components**. Body signature: carry + 10 arguments (one a 6-component `Q`
  NamedTuple) = 15 scalar inputs, four of them precomputed per-level boolean
  mask fields (`kmin_r = q.r > qmin` etc., built outside as `CellKField[bool]`).
- **Five recurrences in one scan**: the four species sedimentation
  recurrences are mutually independent (each reads only its own carry slot
  plus shared `rho`/`zeta`); they are instantiated from a helper
  `precip_qx_level_update(previous_level_q, previous_level_rho, prefactor, exponent, offset, …)` called four times with per-species constants — a
  hand-made abstraction for "one species' scan body". The fifth recurrence
  (temperature/energy flux) couples to the others only through *same-level*
  outputs (`pr=r_update.p`, `pflx_tot=s_update.p + i_update.p + g_update.p`).
  Nothing about the physics requires one scan; the construct does.
- **Carry as workaround channel**: of 21 components, `t_state.t` and
  `pflx_tot` are never read from `previous_level` — pure output channels
  (carry == output). `rho` is carried solely as a **delay line**: the flux
  average needs ρ at k−1 and a scan body cannot read an input at an offset.
  The symmetric problem — lookahead — is solved *outside* the scan:
  `t_kp1 = concat_where(dims.KDim < last_lev, t(Koff[1]), t)` precomputed
  and passed as an extra input.
- **Scalar-body tax**: `core/properties.py` / `core/thermo.py` maintain
  duplicated `*_scalar` twins of field operators (`_fall_speed_scalar`,
  docstring "Compute the scalar fall speed (can be used in scan operator)";
  `_T_from_internal_energy_scalar`, "scalar version callable from
  scan_operator") — and the scan body *still* manually inlines one "in order
  to avoid scan_operator -> field_operator" (graupel.py:155).
- **Work-skipping branch**: the body wraps everything in
  `if current_level_activated:` with an identity else-branch, annotated
  `# TODO(): … Can be made unnecessary with future transformations.`
  (graupel.py:217). The identity branch generates per-level self-copies that
  ~650 lines of hand-written DaCe hooks
  (`implementations/graupel_dace_hooks.py`: `remove_self_copy_inside_scan`,
  `rename_intermediate_access_nodes`) exist to repair, plus in/out buffer
  aliasing flagged "not best GT4Py practice".
- **Surface outputs by program slicing**: `pr, ps, pi, pg, pre` are written
  with `dims.KDim: (vertical_end - 1, vertical_end)` domains in the
  `@gtx.program` and allocated as single-level fields — i.e. the *final
  carry*, extracted manually.
- **Tracing scale**: drivers wrap execution in a `recursion_limit(10**4)` /
  `(10**5)` context manager, "TODO(havogt): make an option in gt4py?".

**Old microphysics** (`microphysics/stencils/graupel_stencils.py`,
`_icon_graupel_scan`): the same pattern without the NamedTuple hygiene — a
**flat 28-tuple carry** preceded by `sys.setrecursionlimit(350000)`
("The limit has to be manually set to a huge value for a big scan
operator"). Carry slots include five `*_kup` caches of previous-level
*derived* values (`rho_kup`, `qvsw_kup`, …— delay lines again), four fall
speeds, an explicit level counter (`k_lev = k_lev + 1`) used for
`is_surface = True if k_lev == ground_level else False` branching at four
places, and `cloud top distance`. The wrapper field operator unpacks all 28
outputs and **discards 13 of them with `_`**. Ground fluxes are computed by
two separate post-scan field operators over sliced domains
(`(ground_level, num_levels)` / `(model_top, ground_level)`) consuming the
scan's by-product outputs.

**Other icon4py scans** (completing the survey): `diagnose_pressure.py` —
backward scan, carry `(pressure, pressure_interface, is_first_level: bool)`
with the bool as first-iteration flag and one slot never read back;
`metric_fields.py::_compute_param` — carry `(gtx.int32, bool)` maintaining a
hand-rolled k counter purely to compare against level bounds;
`solve_tridiagonal_matrix_for_w_{forward_sweep,back_substitution}.py` —
clean, natural Thomas-algorithm scans (carry `(c', d')` / scalar), the
motivating case for a `tridiagonal_solve` primitive (§B.7).

Implications fed back into the proposal: P6, goals 6–7, K-local views
(§3.1), final-carry surface outputs, and the scan-fusion obligation (§3.6).

### A.8 Production evidence: PMAP Helmholtz preconditioner (`gt4py.cartesian`)

Survey of `PMAP-Project/PMAP` (https://github.com/PMAP-Project/PMAP),
`src/pmap/helmholtz/preconditioner.py` — production *cartesian* evidence,
complementing §A.6's description of the mechanism with what a real GTScript
vertical solve looks like. This matters because cartesian users are the
second migration audience (§4.6), and their canonical solve is deliberately
*not* textbook Thomas. The `_tridiagonal` stencil is the vertical line solve
of a line-relaxation preconditioner, applied `nitr` times inside a GCR-style
iteration:

```python
@stencil
def _tridiagonal(rs, b33, ee, dnmi, denratm, rp):
    with computation(FORWARD):
        with interval(0, 2):
            ff = rs[0, 0, 0] * dnmi[0, 0, 0]
        with interval(2, -1):
            zff = rs[0, 0, 0] + denratm[0, 0, 0] * b33[0, 0, -1] * ff[0, 0, -2]
            ff = zff * dnmi[0, 0, 0]

    with computation(BACKWARD):
        with interval(-1, None):
            rp[0, 0, 0] = (
                ff[0, 0, -2] * 2.0 * denratm[0, 0, 0] * b33[0, 0, -1] + rs[0, 0, 0]
            ) * dnmi[0, 0, 0]
        with interval(-2, -1):
            rp[0, 0, 0] = ff / (1.0 - ee[0, 0, 0])
        with interval(0, -2):
            rp[0, 0, 0] = ee[0, 0, 0] * rp[0, 0, 2] + ff
```

Four load-bearing observations:

- **Stride-2 recurrences via offset reads of the in-progress output.**
  `ff[0,0,-2]` in FORWARD and `rp[0,0,2]` in BACKWARD: two interleaved
  even/odd elimination chains, i.e. an order-2 recurrence — the GTScript
  feature the proposal keeps out of the core (§4.6), used in production and
  not at order 1. Transcribed with a tuple carry this becomes
  `carry = (ff, ff_km1)` where the second slot exists only to be "ff two
  levels ago" — exactly the delay-line idiom the proposal eliminates for
  inputs (§A.7's `rho`), reappearing for outputs.
- **Pre-factored coefficients.** `dnmi` (inverse denominators), `ee`,
  `denratm` are computed once outside the solver; only the cheap sweeps run
  per preconditioner application (and `nitr` times per call). A
  fixed-signature `tridiagonal_solve(a, b, c, d)` (§3.7, §B.7) does not
  apply — the raw scan primitive must stay ergonomic for pre-factored and
  non-standard solves.
- **Multi-level boundary regions at both ends, end-anchored.**
  `interval(0, 2)` peels *two* levels with one body; the BACKWARD pass peels
  the *last two* levels with two different bodies; and the FORWARD pass
  stops one level short (`interval(2, -1)` — `ff` is never computed at the
  last level; its elimination step is folded into the BACKWARD last-level
  body). Every boundary except `interval(0, 2)` is *end*-relative.
- **Interval-local extents.** The BACKWARD pass covers the full column while
  consuming `ff`, which is defined only on `[0, n-1)`; this is well-formed
  only because the last-level interval reads `ff[0,0,-2]`, never `ff` at
  the current level. Similarly `b33[0,0,-1]` is read only for `k >= 2`.
  Cartesian gets this precision for free — each `interval` block is its own
  code with its own accesses; a body expressing the same dispatch with
  `concat_where` must not lose it to a global extent analysis over all
  branches.

Implications fed back into the proposal: end-anchored index conditions and
the partition-based peeling guarantee (§3.3), guard-refined extents and
range deduction (§3.1–§3.3, open question 9), `history` carries (§4.6), and
the §3.7 library caveat. Worked transcription in the examples companion
(§7).

### A.9 Production evidence: GridTools `physics_patterns` (Météo-France, NOAA)

The two open PRs in https://github.com/GridTools/physics_patterns (a
repository collecting vertical physics patterns for exactly this design
space) contribute three patterns and one boundary case.

**PR #1 — Météo-France PHYEX `condensation.F90`** (tropopause inversion +
mixing length):

- *Tropopause level = argmin over K*: a k-loop tracking `(ZTMIN, ITPL)` —
  the running minimum temperature plus its level index. As a scan: final
  carry only, no per-level output (the fold form, proposal §3.1/§3.5),
  and the body needs the level *index as a value* to store `ITPL = JK`
  (today's workaround is the hand-rolled in-carry counter of §A.7's
  `metric_fields.py::_compute_param`). As a builtin:
  `argmin(t, axis=KDim)`.
- *Gather at a computed K index*: the mixing-length condition reads
  `PZZ(JIJ, ITPL(JIJ))` — the height at the per-column tropopause level.
  Not a scan concern: an `as_index`-style computed-index gather
  (field-valued connectivity, subject of a separate design discussion),
  composing with the scan as a k-less input.
- *Conditional recurrence*: `ZL(k) = 0.6·ZL(k−1)` above 0.9× tropopause
  height, else `min(600, z − z_sfc)`, seeded with `ZL = 20` at the first
  level — a clean slice-body scan (`where` with a data-dependent
  per-column condition, init/`include_initial` seeding). Transcribes
  without new machinery (examples §8). The runtime `IKL = ±1` level-order
  genericity is a Fortran portability artifact; the DSL fixes the
  ordering convention.

**PR #2 — NOAA `lsm.py`** (Noah land-surface): a k-loop with per-column
trip count (`for k in range(nroot[i, j])`; the PBL variant varies at
runtime), accumulating a k-less soil conductance. The vectorized form is a
full-range reduction masked by `k < nroot` — validating §4.8's
masked-bound stance, but the mask compares the level index against a
*field*, which anchored conditions (proposal §3.3) cannot express: it
needs `gtx.index(KDim)` as a value (proposal §3.5). `zsoil` is again
gathered at the computed index `nroot`.

**PR #2 — NOAA `microphysics.py`** (SHiELD melting during sedimentation):
the boundary case. Upward k-loop over ice source levels; per-k inner
*downward* m-loop with two data-dependent `break`s; scatter writes at both
k and m (`qice[k] -= …`, `qrain[m] += …`, temperature at both); and
sequential feedback in *both* loops — `qice[k]` is exhausted across the
inner loop and `temperature[m]` is updated and re-read by later k
iterations. Strictly harder than the COSMO sedimentation of §C: the
order-dependent `min` caps read intermediate state, so it is *not*
reformulable as a masked bounded-window gather without changing the
algorithm. Regime 3 of §C.3 *with feedback*; fed back into §4.8 as the
documented scope boundary.

Implications fed back into the proposal: P7 and §3.5 (vertical reductions,
`argmin`, `gtx.index`, the fold form), §4.8 (scope boundary and the
escape-hatch discussion), and the cross-reference to the computed-index
gather design. Worked transcriptions in the examples companion (§8–§10).

## B. Prior art

### B.1 JAX

**`jax.lax.scan(f, init, xs=None, length=None, reverse=False, unroll=1)`**
with `f :: (c, a) -> (c, b)`, scanning the leading axis of the `xs` pytree,
returning `(final_carry, stacked_ys)`. Reference semantics from the docs:

```python
def scan(f, init, xs):
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, np.stack(ys)
```

Salient properties for our design:

- carry must keep fixed structure/shape/dtype across iterations (init defines
  it);
- carry and stacked output are **separate** concepts;
- `xs=None` + `length=` gives pure recurrences — note this forces an explicit
  length, mirroring our "pure recurrences must state a range" rule;
- `reverse=True` keeps `ys` index-aligned with `xs` (our `forward=False`);
- static trip count and shapes; no early exit (that is `lax.while_loop`,
  which loses reverse-mode AD and output stacking);
- `f` operates on array *slices*, so batching over columns is `vmap` — the
  canonical NWP-shaped pattern is "vmap over columns, scan over k", and under
  `vmap` the carry simply becomes a per-column array. No scalar↔field
  promotion problem exists by construction.

**`jax.lax.associative_scan(fn, elems, reverse=False, axis=0)`**: parallel
prefix (Blelloch-style, O(n) work / O(log n) depth) for *associative* `fn`;
linear recurrences `x[k] = a[k]·x[k-1] + b[k]` are associative over `(a, b)`
pairs, so vertical integrals and even the Thomas forward sweep (2×2
matrix/Möbius form, with stability caveats) can be cast into it.

Sources:

- https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
- https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html
- https://github.com/jax-ml/jax/discussions/19114 (vmap×scan guidance)
- https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html

### B.2 Array API standard / NumPy

- `cumulative_sum(x, *, axis, dtype, include_initial=False)` entered the
  standard in 2023.12 (RFC data-apis/array-api#597); `include_initial=True`
  prepends the identity, yielding length N+1 — the exclusive/inclusive scan
  with explicit initial value, exactly the staggered-grid (interface vs
  cell-center) shape. `cumulative_prod` followed in 2024.12 (RFC #598).
- **No general scan** (reduce-with-carry) exists in the standard and no open
  proposal was found; the standard avoids higher-order functions because
  function-valued arguments sit poorly with lazy/graph array libraries.
  Consequence: portable embedded lowering exists only for `cumsum`/`cumprod`;
  a general scan must remain a GT4Py-level construct.
- NumPy's `np.ufunc.accumulate` generalizes only over binary ufuncs; custom
  Python ufuncs run at interpreter speed.
- **Inclusive/exclusive duality.** With `y[k] = f(y[k-1], x[k])`: the
  *inclusive* scan emits the carry after consuming `x[k]`; the *exclusive*
  scan emits the carry before (so `y[k0] = init`, and
  `exclusive[k] = inclusive[k-1]`); `include_initial=True` is "inclusive
  plus the init emitted one level upstream", subsuming both
  (exclusive = include_initial minus the last level). For NWP this duality
  *is* the cell-center/interface staggering: n cell increments integrate to
  n+1 interface values whose first entry is the boundary condition. It also
  reframes the "boundary condition at the first level" idiom: `y[0] = bc; y[k] = f(y[k-1], x[k])` is precisely an inclusive scan over `[1, n)` with
  its init emitted at 0 — one primitive, no special case (proposal §3.3).

Sources:

- https://data-apis.org/array-api/latest/API_specification/generated/array_api.cumulative_sum.html
- https://github.com/data-apis/array-api/issues/597 ,
  https://github.com/data-apis/array-api/issues/598 ,
  https://github.com/data-apis/array-api/issues/797
- https://numpy.org/doc/stable/reference/generated/numpy.ufunc.accumulate.html

### B.3 PyTorch

`torch._higher_order_ops.scan.scan(combine_fn, init, xs, *, dim=0, reverse=False) -> (carry, ys)` — a JAX-shaped general scan, pytree support,
fixed carry structure, **prototype** (requires `torch.compile`, documented
miscompile risk). `torch.associative_scan(combine_fn, xs, dim, reverse, combine_mode)` — prototype, CUDA-codegen-only efficient path, no autograd,
open correctness issues (e.g. pytorch/pytorch#137943). PyTorch/XLA ships its
own JAX-shaped `scan`/`scan_layers`.

Two lessons: (1) the whole Python array ecosystem converged on the
`(carry, x) -> (carry, y)` signature; (2) compiling a *general* scan is hard
(prototype after 2+ years) while fixed-op cumulatives are trivial — support
for keeping the general primitive semantically simple and sequential.

Sources:

- https://docs.pytorch.org/docs/main/higher_order_ops/scan.html
- https://docs.pytorch.org/docs/main/higher_order_ops/associative_scan.html
- https://github.com/pytorch/pytorch/pull/119430 ,
  https://github.com/pytorch/pytorch/issues/137943
- https://docs.pytorch.org/xla/master/features/scan.html

### B.4 Futhark

`scan (f: t -> t -> t) (neutral: t) (arr: []t)` requires `f` associative
with a neutral element — that requirement is what licenses parallel-prefix
codegen. Non-associative recurrences use the explicitly sequential
`loop pat = init for i < n do body` construct. Futhark thus encodes the
**two-tier design** — restricted-but-parallelizable `scan` vs general
sequential `loop` — in the language itself. (Futhark also published
reverse-mode AD of scan, relevant if differentiable vertical solvers become a
requirement.)

Sources:

- https://futhark-book.readthedocs.io/en/latest/language.html
- https://futhark.readthedocs.io/en/stable/performance.html
- https://futhark-lang.org/publications/pldi17.pdf
- https://dl.acm.org/doi/fullHtml/10.1145/3652561.3652575

### B.5 Halide

Reductions/recurrences are *update definitions* over an `RDom`: e.g.
`f(x) = f(x-1) + in(x)` traversed serially in order, with pure dimensions
schedulable in parallel around it — the exact "parallel columns, serial k"
factorization, expressed as a *scheduling* property. Two transferable ideas:

- **`rfactor`** (Suriana, Adams & Kamil, CGO'17): the scheduler may split a
  serial reduction into parallel partial reductions when it can synthesize an
  associative operator — associativity as a scheduling concern, invisible in
  the algorithm definition.
- **Boundary conditions** (`BoundaryConditions::repeat_edge`,
  `constant_exterior`, …) are wrappers that make out-of-range reads
  well-defined, and Halide's **loop partitioning** specializes the
  steady-state interior so border handling costs nothing in the inner loop —
  the formal version of the "peeling guarantee" in the proposal.

Sources:

- https://halide-lang.org/tutorials/tutorial_lesson_09_update_definitions.html
- https://halide-lang.org/tutorials/tutorial_lesson_18_parallel_associative_reductions.html
- https://halide-lang.org/docs/class_halide_1_1_r_dom.html
- https://halide-lang.org/docs/namespace_halide_1_1_boundary_conditions.html
- https://andrew.adams.pub/par.pdf

### B.6 Weather & climate DSLs

- **GTScript / gt4py.cartesian**: `computation(FORWARD|BACKWARD)` +
  `interval(...)` vertical regions + offset reads of already-updated values;
  see §A.6. Docs: https://gridtools.github.io/gt4py/latest/gtscript.html
- **GridTools C++**: same vertical execution model plus **k-caches** — ring
  buffers in registers/shared memory for vertical loop-carried dependencies,
  with `fill`/`flush` IO policies; the user-manual worked example is a
  tridiagonal solver (FORWARD elimination interval + BACKWARD substitution).
  https://gridtools.github.io/gridtools/latest/user_manual/user_manual.html
- **STELLA** (COSMO GPU dycore predecessor) pioneered the model; vertical
  advection/tridiagonal solve is a canonical benchmark.
  https://github.com/GridTools/stencil_benchmarks
- **Dawn/GTClang**: `vertical_region(k_start, k_end)` as the SIR construct;
  the compiler does interval splitting/merging, k-cache assignment, fusion.
  https://github.com/MeteoSwiss-APN/dawn ,
  https://spcl.inf.ethz.ch/Publications/.pdf/osuna-dawn.pdf
- **PSyclone / LFRic**: kernels with `operates_on = CELL_COLUMN` — users
  write plain k-loops over a whole column, PSyclone generates horizontal
  parallelism ("vmap over columns" by codegen). Additionally **CMA
  (Column-wise Matrix Assembly) operators**: columnwise banded
  assemble/apply/inverse-apply as first-class kernel types — vertical solvers
  as built-in operator objects rather than user scans.
  https://psyclone.readthedocs.io/en/stable/dynamo0p3.html ,
  https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.3880
- **ICON GPU port**: OpenACC over Fortran; vertical recurrences are serial
  k-loops inside `!$ACC PARALLEL LOOP` over columns. No scan abstraction;
  production evidence that sequential-k/parallel-columns is the right GPU
  execution model.
  https://egusphere.copernicus.org/preprints/2022/egusphere-2022-152/egusphere-2022-152-manuscript-version4.pdf

### B.7 Tridiagonal solvers as primitives

- `jax.lax.linalg.tridiagonal_solve(dl, d, du, b)` — batched; lowers to
  cuSPARSE `gtsv2` on GPU, Thomas/LAPACK on CPU; JVP rule exists; batching
  currently loops rather than using `gtsv2StridedBatch` (perf issue
  jax-ml/jax#28544).
- cuSPARSE `gtsv2` / `gtsv2StridedBatch` / `gtsv2_nopivot`: vendor batched
  solvers (cyclic-reduction/PCR based).
- GPU literature: Kim et al., *A Scalable Tridiagonal Solver for GPUs*
  (hybrid Thomas+PCR); Merrill/Garland-era CR/PCR analyses; PaScaL_TDMA for
  multi-GPU. For NWP shapes (k≈100, huge batch) a thread-per-column Thomas
  sweep is usually optimal — the primitive's value is **ergonomics and pinned
  numerics**, not raw speed.
- Most stencil DSLs do *not* offer it built in (users hand-write two sweeps);
  LFRic CMA and JAX are the exceptions and the model to follow.

Sources:

- https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.tridiagonal_solve.html
- https://github.com/jax-ml/jax/issues/6830 ,
  https://github.com/jax-ml/jax/pull/25787 ,
  https://github.com/jax-ml/jax/issues/28544
- http://impact.crhc.illinois.edu/shared/papers/hee-seok2011.pdf
- https://www.sciencedirect.com/science/article/abs/pii/S0010465523001303
- https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

### B.8 Windowed scans and higher-order recurrences

- Any order-m recurrence reduces to a first-order scan with an m-tuple carry
  (standard JAX idiom; Blelloch, *Scans and Linear Recurrences*,
  http://www.cs.cmu.edu/~scandal/alg/scan.html). Higher-order *linear*
  recurrences remain parallelizable: Maleki & Burtscher, PLDI'16
  (https://userweb.cs.txstate.edu/~burtscher/papers/pldi16.pdf) and
  ASPLOS'18 (https://dl.acm.org/doi/pdf/10.1145/3296957.3173168).
- GTScript/Halide instead expose **offset reads of the in-progress output**
  (`f[0,0,-1]`, `f(x-2)`) with k-caches/registers making the window fast —
  the ergonomic alternative surface syntax; the compiler infers the
  equivalent carry depth.
- **Local-view precedent inside gt4py itself**: at the ITIR level the scan
  pass already receives *iterators*, not values — they can be shifted and
  dereferenced (see the scan in
  `tests/next_tests/unit_tests/iterator_tests/transforms_tests/test_domain_inference.py`,
  whose pass is `λ(init, it). deref(shift("Ioff", 1)(it))`). Only the ffront
  `scan_operator` flattens arguments to scalars. A K-local-view body
  (proposal §3.1) therefore maps onto the *existing* IR; GridTools fill-type
  k-caches (§B.6) are the corresponding execution mechanism. This also
  resolves the asymmetry that read-only inputs admit *lookahead* (`x[+1]` in
  a forward scan — used by muphys via external precomputation, §A.7) which
  no carry encoding can express.
- Merrill & Garland, *Single-pass Parallel Prefix Scan with Decoupled
  Look-back* (NVIDIA NVR-2016-002,
  https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf)
  is the CUB `DeviceScan` building block if a backend ever parallelizes in k.

### B.9 Boundary-condition-friendly constructs (survey)

| Mechanism                                 | Where                                                    | Character                                                        |
| ----------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------- |
| Vertical regions / interval dispatch      | GTScript `interval`, Dawn `vertical_region`              | declarative, compiler splits k-loop, zero steady-state cost      |
| Explicit init carry                       | JAX/Futhark; array-API `include_initial`                 | BC = initial state, evaluated once, no branch                    |
| Step-index predication                    | JAX (`where`/`cond` on an index in `xs`), manual peeling | works, but is exactly the clumsy pattern; no first-class support |
| Out-of-range wrappers + loop partitioning | Halide `BoundaryConditions::*`                           | reads beyond bounds defined; interior specialized                |

The synthesis adopted by the proposal: explicit init (row 2) for state
seeding **plus** peel-guaranteed index conditionals (rows 1+4 combined) for
differing first/last-level computations.

### B.10 Reductions: prior art

- **XLA `Reduce`**: arbitrary combine computation, but "the evaluation
  order of the reduction function is arbitrary and may be
  non-deterministic"; the function must tolerate reassociation. The order
  freedom *is* the semantic payload distinguishing reduce from the
  sequential while/scan. JAX accordingly exposes both tiers: `lax.scan`'s
  final carry (order-pinned fold) and `lax.reduce`/`jnp.sum`
  (reorderable).
  https://openxla.org/xla/operation_semantics
- **Futhark**: `reduce` requires associativity + a neutral element
  (licensing parallel codegen); sequential recurrences use `loop` (§B.4).
  Its compiler IR fuses the combined forms into **`screma`** — a
  generalized scan-reduce-map second-order combinator with `redomap`
  (map+reduce) and `scanomap` (map+scan) as special cases; the fusion
  engine is built around it. The lesson for proposal §3.5: combined
  scan+reduce is an IR/fusion concern, not a user-facing construct.
  https://hackage.haskell.org/package/futhark-0.20.6/docs/Futhark-IR-SOACS-SOAC.html ,
  https://dl.acm.org/doi/10.1145/2935323.2935326
- **Array API standard**: *all* fixed-op reductions are standardized
  (`sum`, `prod`, `min`, `max`, `argmin`, `argmax`, `any`, `all`) while no
  general reduce/scan is — §B.2's finding seen from the other side.
  Fixed-op vertical reductions therefore get a portable one-call embedded
  lowering, unlike the general scan.
  https://data-apis.org/array-api/latest/API_specification/statistical_functions.html
- **NumPy** `sum` uses pairwise summation — even the reference array
  library does not evaluate in index order; order-unspecified is the
  ecosystem default for reductions.
  https://numpy.org/doc/stable/reference/generated/numpy.sum.html
- **GridTools 2** has no reduction construct at all; a column sum is a
  FORWARD computation accumulating through a k-cache — i.e. compiled
  backends lower a vertical reduce as a sequential k-accumulator in the
  existing vertical executor, bit-reproducible by construction.
  https://gridtools.github.io/gridtools/latest/user_manual/user_manual.html
- **PSyclone/LFRic** treats reductions as first-class kernel metadata
  (global sums, inner products) and offers a *reproducible* reduction mode
  (padded per-thread partials) — evidence that the weather/climate
  community demands an explicit reproducibility answer, which proposal
  §3.5 gives as "write the fold".
  https://psyclone.readthedocs.io/en/stable/dynamo0p3.html

## C. Case study: scan + sub-reduce in the vertical

### C.1 The COSMO/ICON two-moment sedimentation scheme

Source: U. Blahak, *New implementation of explicit hydrometeor sedimentation
in the Seifert-Beheng 2-moment bulk microphysical scheme*, DWD, 2020-02-24,
http://www.cosmo-model.org/content/model/documentation/core/docu_sedi_twomom.pdf

Setting: bulk sedimentation `∂(ρq)/∂t = −∂/∂z [v(ρq)·ρq]`, explicit
first-order scheme, fluxes `P` on half levels, k increasing top→bottom:

```
ϕ_k(i+1) = ϕ_k(i) − (P_{k−1/2} − P_{k+1/2})·Δt / Δz_k        (eq. 3)
```

The time integral of the flux through a face is rewritten (via a δ-function
argument) as a *height integral over the levels above*, masked to the
fall-distance reach (eqs. 6–7):

```
P(z_f)·Δt ≈ −∫_{z_f}^{∞} ϕ(z)·F(z) dz ,   F = 1 if 0 < z − z_f < |v(z)|Δt else 0
```

Two discretizations:

- **Current scheme** (eq. 9–11, `-DSEDI_VECTORIZED`): approximate `v(z)` by
  the face value `v_{k+1/2}` everywhere above, giving a *boxcar* mask and a
  running-Courant recurrence `c_{k−l} = (c_{k−l+1} − 1)·Δz_{k−l+1}/Δz_{k−l}`
  with summation `P_{k+1/2}Δt ≈ −Σ_l ϕ_{k−l}·min[c_{k−l}, 1]·Δz_{k−l}`,
  stopped when `c < 1`. Vectorizes (NEC), but produces unphysical
  "rain congestion" pulses at Courant > 1.

- **New scheme** ("boxtracking", eq. 12): no approximation of `v(z)`;
  per-level **gather over the levels above**:

  ```
  P_{k+1/2}·Δt ≈ −Σ_{l=0}^{k−1} ϕ_{k−l} · (z_up − z_low)
  z_up  = min[ z_{k+1/2−l−1} , z_{k+1/2} + |v_{k−l}|Δt ]
  z_low = min[ z_{k+1/2−l}   , z_up ]
  ```

  Note the **self-masking**: when level k−l is out of reach,
  `z_up = z_{k+1/2} + |v_{k−l}|Δt = z_low` and the contribution is exactly 0.
  The paper's reference implementation is the work-efficient **scatter**
  form: an outer loop over source levels k with an inner
  `while Δsum < |v_k|Δt` loop scattering `−ϕ_k·(z_up − z_low)` into the
  fluxes of the faces below. The paper notes this form "does not properly
  vectorize" on NEC — i.e. the work-efficient form is the
  vectorization-hostile one, the gather form is the vectorizable one.

### C.2 The gist: six formulations and the key transformation

https://gist.github.com/havogt/c52d19f2f7557c2c048d12be7330bda8
(`test_sedim_box_standalone.py`) implements the box scheme six ways; the
interesting axis of variation:

| Variant                 | Form                                                               | Complexity            | Notes                                                                                   |
| ----------------------- | ------------------------------------------------------------------ | --------------------- | --------------------------------------------------------------------------------------- |
| `box_sedim_paper`       | scatter + data-dependent `while`                                   | O(n·M̄) work-efficient | the paper's algorithm; serial, scatter writes                                           |
| `box_sedim_naive`       | gather, full triangle `for m in range(k+1)`                        | O(n²)                 | direct eq. 12                                                                           |
| `box_sedim_naive_break` | gather + monotonicity `break`                                      | O(n·M̄)                | early exit ⇒ not vectorizable                                                           |
| `box_sedim_naive_func`  | gather written as `Σ_m f(shift_origin(·, k), m)`                   | O(n²)/O(n·M)          | **the windowed-stencil formulation** — per-output sum of a function of K-shifted inputs |
| `box_sedim_scanlike`    | forward k-loop + **rolling buffers** (`z`: 4, `v`: 3, `phi`: 3)    | O(n·M), M=3 fixed     | bounded window ⇒ single pass, O(M) state                                                |
| `box_sedim_scan`        | same, packaged as `scan(scanpass, range, init=(None, buffers...))` | O(n·M)                | shows the carry: ring buffers + level index `k` passed in                               |

The load-bearing observations:

1. **If the window is bounded** (max Courant number ⇒ contributions reach at
   most M levels), the algorithm needs only the last M levels of `z`, `v`,
   `phi` at each k. `box_sedim_scanlike` carries them in ring buffers — i.e.
   the scan + sub-reduce *is* an ordinary first-order scan with an O(M) carry.
2. But in this particular scheme the contributions don't depend on any
   *running* state at all (no P feedback, the apparent recurrence is only the
   cumulative height, which is itself a `cumsum` of `Δz` and here a plain
   input `z`). So the bounded form is **not even a scan** — it is a vertical
   stencil with a static window and a sum-reduction:
   `P[k] = −Σ_{m=0..M} ϕ[k−m]·(z_up(k,m) − z_low(k,m))`. The ring buffers in
   the gist are a *memory-locality optimization* (k-caches), not a semantic
   necessity — exactly the thing a compiler, not the user, should introduce.
3. The level-index conditionals in `box_sedim_scanlike` (`if k <= 1`,
   `elif k == 2`) are pure *array-boundary masking* of the window (don't read
   above the model top) — skip-value semantics, not physics.
4. The data-dependent extent (`while`/`break` variants) buys work-efficiency
   at the cost of vectorization — the same trade the paper documents on NEC.
   The GPU/SIMD-friendly version is the masked fixed bound, relying on
   self-masking (observation in §C.1) for correctness.

### C.3 Pattern classification

Generalizing, "scan + sub-reduce in the vertical" decomposes into:

```
out[k] = reduce_{m ∈ W(k)} g( x[k], x[k−m], x[k−m±1], …, s[k] )
s[k]   = scan-state (optional), e.g. cumulative height, running mass
```

with three regimes:

1. **Static bounded window, no state feedback** (sedimentation gather with
   CFL bound; PPM flux integration with bounded departure range): a windowed
   stencil + reduction. Needs: K-window access, masking at array boundaries,
   a reduction over the window. *No scan construct required*; any cumulative
   quantities (heights from Δz) are separate `cumsum`s composed in front.
2. **Bounded window + genuine recurrence** (the window contribution feeds or
   reads running state; higher-order recurrences): a scan whose carry
   includes a ring buffer of the last M slices, and/or window access to
   inputs inside the body. The gist's `box_sedim_scan` shows the manual
   encoding; the toolchain should derive it from "window access inside a
   scan body".
3. **Data-dependent extent / scatter** (the paper's work-efficient form;
   unbounded Courant numbers): inner `while` over a data-dependent range, or
   scatter-writes to multiple output levels per step. Hostile to
   vectorization, vmap, AD, and parallel-in-k; production vector codes
   already avoid it. Covered by regime 1/2 plus a static bound `M = ⌈max CFL⌉` and masking; genuinely unbounded cases fall back to multiple passes
   or host-side chunking.

### C.4 Cost analysis of the masked-bounded form

Work: O(n_k · M) vs O(n_k · M̄) for the data-dependent form (M̄ = mean
extent ≤ M). For sedimentation M is small (the gist uses 3; max CFL in the
paper's experiments ~10 only in the degenerate thin-layer near-ground case);
the overhead factor M/M̄ is bounded and is paid in exchange for full
column-vectorization — the same trade GPU tridiagonal/PCR literature and the
NEC-vectorized COSMO variant make. Memory: regime-1 windowed stencils read
each input O(M) times without caching; ring-buffer/k-cache lowering restores
O(1) reads per element — GridTools k-caches (§B.6) are precisely this and
already exist in the GTFN execution substrate.

### C.5 Other instances of the pattern

- **PPM / semi-Lagrangian flux-form advection** (Colella & Woodward 1984;
  Lin & Rood 1996; FV3's vertical remapping): piecewise-parabolic
  reconstruction is a small-window vertical stencil; the flux through a face
  integrates the reconstruction over the departure region, which spans a
  data-dependent but CFL-bounded number of cells — regime 1 with a `cumsum`
  (cumulative mass/height) composed in front. The integer-offset "how many
  whole cells does the trajectory cross" logic becomes a window-masked sum.
- **ICON two-moment microphysics sedimentation**
  (`mo_2mom_mcrph_processes.f90`, `lboxtracking` variants): the production
  Fortran of §C.1, applied simultaneously to mass and number densities
  (tuple-valued `ϕ`) with moment-coupled fall speeds — same structure, tuple
  element types.
- **Precipitation-flux style schemes generally** (implicit/explicit
  column microphysics with flux limiting): forward scan for the flux with
  carry = flux entering the cell, occasionally needing the previous M levels
  of hydrometeor state — regime 2.

### C.6 Mapping to the proposed constructs

| Regime              | Construct (proposal §)                                | Embedded lowering                                                                               | Compiled lowering                                 |
| ------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| 1: windowed stencil | `WindowOffset` + `neighbor_sum` + skip-masking (§3.4) | stacked shifted views / `sliding_window_view`; JAX: slices, fully `jit`/`vmap`-able             | unrolled window in the k-loop; k-caches for reuse |
| 2: windowed scan    | window access inside `@gtx.scan` body (§3.1+§3.4)     | per-level slice expressions; JAX: `lax.scan` with ring-buffer carry synthesized by the frontend | fill-type k-caches in the scan loop               |
| 3: data-dependent   | static bound + masking (§4.8); not supported natively | —                                                                                               | —                                                 |

The sedimentation example in proposal §3.4 demonstrates regime 1 end-to-end:
the proposal's `box_sedim_flux` is a line-by-line vectorization of the
gist's `box_sedim_naive_func` with the window machinery absorbing
`shift_origin`, the triangle bound, and the boundary `if`s.

## D. Synthesis: design drivers

| #   | Finding                                                                                                                                             | Source                                       | Consequence in proposal                                                 |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------- |
| 1   | Ecosystem converged on `(carry, x) → (carry, y)` with explicit `init`, `reverse`                                                                    | §B.1, §B.3                                   | §3.1 signature; JAX-lowerable embedded                                  |
| 2   | Slice-level bodies make batching/vectorization implicit and kill promotion magic                                                                    | §B.1 (vmap), §B.6 (LFRic)                    | §3.1 slice typing; §4.1                                                 |
| 3   | Only `cumsum`/`cumprod` are portable array-API scans; general scan stays a DSL construct                                                            | §B.2                                         | §3.7 library lowering; embedded keeps a k-loop                          |
| 4   | General scans are hard to compile; keep core semantics sequential and simple                                                                        | §B.3, §B.4                                   | §2 non-goals; §4.7                                                      |
| 5   | Interval/vertical regions + loop partitioning are the proven BC mechanism; init-as-state the proven seeding mechanism                               | §B.5, §B.6, §B.9                             | §3.3 three-mechanism design with peeling guarantee                      |
| 6   | Sequential-k / parallel-columns is the production GPU model; parallel-in-k is a later scheduling concern                                            | §B.6 (ICON, GridTools), §B.5 (rfactor), §B.4 | §2, §4.7                                                                |
| 7   | Tridiagonal solve should be a named primitive, not a user scan                                                                                      | §B.7                                         | §3.7                                                                    |
| 8   | Order-m recurrences = first-order scan + m-carry; offset-output reads are sugar over it                                                             | §B.8                                         | §4.6                                                                    |
| 9   | Bounded scan+sub-reduce = windowed stencil (+ separate cumsum); ring buffers are a compiler optimization, boundary `if`s are skip-masking           | §C.2                                         | §3.4 `WindowOffset` reusing sparse-field machinery                      |
| 10  | Data-dependent extents trade vectorization for work-efficiency; production vector codes use masked bounds                                           | §C.1, §C.4                                   | §4.8 rejection of `while`-scans                                         |
| 11  | Production scans degenerate into mega-operators: manual fusion, delay-line/pass-through carry slots, `*_scalar` helper twins, bespoke backend hooks | §A.7                                         | P6; goals 6–7; §3.6 fusion obligation                                   |
| 12  | Previous-level *and* lookahead input access is real demand; carry delay lines and external pre-shifts are the workarounds                           | §A.7, §B.8                                   | §3.1 K-local views ("local view vertically")                            |
| 13  | Inclusive/exclusive/`include_initial` unify first-level BCs with staggered outputs                                                                  | §B.2                                         | §3.3 mechanism 2; `gtx.lib` wrappers                                    |
| 14  | Surface diagnostics are final carries, today extracted by last-level program slicing                                                                | §A.7                                         | §3.1 `(final_carry, ys)` return                                         |
| 15  | Production cartesian solves are pre-factored, stride-2, with multi-level end-anchored intervals; a textbook-Thomas library call doesn't cover them  | §A.8                                         | §3.3 anchors + partition guarantee; §3.7 caveat; §4.6 `history` carries |
| 16  | Interval dispatch makes extents interval-local for free; conditionals-in-body must match this precision via guard refinement                        | §A.8, §A.6                                   | §3.1–§3.3 guard-refined extents; open question 9                        |
| 17  | Column reductions (integrals, level searches, masked variable depth, column maxima) are real demand; today flag/counter carries or host NumPy       | §A.9, §A.7                                   | P7; §3.5 fixed-op reductions; `gtx.index`                               |
| 18  | Ecosystem two-tier: order-unspecified fixed-op reduce + order-pinned sequential fold; combined scan+reduce is an IR/fusion concern (`screma`)       | §B.10                                        | §3.5; §3.1 fold form; §3.6 scan→reduce fusion                           |
| 19  | Data-dependent scatter with feedback (SHiELD melting) exists in production; not window-reformulable — a documented scope boundary                   | §A.9                                         | §4.8; §4.1 escape-hatch discussion                                      |
