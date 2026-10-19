---
title: Redesign of the vertical scan in gt4py.next
author: sf-n, tehrengruber
tags:
  - scan
  - vertical
  - reduction
  - boundary-conditions
  - windows
  - k-caches
  - fusion
  - embedded
  - jax
  - frontend
  - patterns
  - tridiagonal
  - sedimentation
  - migration
created: 2026-10-19
status: draft
---

> **This iteration: the pattern catalog.** Every vertical-recurrence pattern we
> have evidence for, each reduced to the smallest *runnable* column program
> that still exhibits it. No API is proposed here — the catalog is the input
> against which a design (ours, or
> [[personal/havogt/scan-redesign|havogt's scan redesign]]) is measured.

- **Status**: draft, collection phase.
- **Companion documents**:
  [[personal/tehrengruber/scan-design/scan-occurences-pmap|the PMAP occurrence survey]]
  (every `with computation(FORWARD/BACKWARD)` in the PMAP dycore, cloud
  microphysics and radiation, with scan/field-op verdicts and the three ported
  stencils);
  [[personal/havogt/scan-redesign|havogt's scan redesign]] with its
  [[personal/havogt/scan-redesign/scan-redesign_research|research notes]] and
  [[personal/havogt/scan-redesign/scan-redesign_examples|worked examples]]
  (prior art, the sedimentation case study, the muphys analysis).

## How to read this catalog

**Minimal form.** Each pattern is given as a NumPy function over **one
column** of $n$ levels — the smallest program that still needs the pattern. No
gt4py construct appears: the point is to state *what must be expressible*,
independently of how. The forms are executable and type-annotated, so they can
serve directly as the reference side of an acceptance suite for whatever API we
end up with. Bodies are concrete stand-ins (`0.5 * s + x[k]` where production
code has a page of physics); only the data flow is the pattern.

Every snippet below runs as written (NumPy 2.5, Python 3.14) and is clean
under `mypy --strict`; the asserted equalities in [A4](#a4), [B4](#b4),
[B6](#b6), [D1](#d1) and [F1–F5](#f) were checked numerically. Shared preamble:

```python
import numpy as np
import numpy.typing as npt

Column = npt.NDArray[np.float64]  # one column: shape (n,), n = number of levels
Mask = npt.NDArray[np.bool_]
Stack = npt.NDArray[np.float64]  # several columns: shape (m, n)
```

**Scan order.** The loop index `k` runs in **scan order**: `k = 0` is the level
the scan visits first, $k = n - 1$ the last. Direction (`FORWARD`/`BACKWARD`,
ground→space or top→bottom) is therefore *not* part of a pattern — it is a
relabelling. A forward and a backward sweep of the same shape are one entry.

**Loop, not vectorized.** The `for k` loop is deliberate: it is the sequential
dependence the pattern is about. Where a pattern turns out *not* to be
sequential, the array-level form is given alongside and the two are asserted
equal ([B6](#b6), [D1](#d1), [F1–F5](#f)) — that equality is exactly the
"this is not a scan" claim.

**No carries unless unavoidable.** A value that lives in an array is read by
index — `out[k - 1]`, `ff[k - 2]`, `x[k - 1]`, `y[k + 1]` — not threaded
through a scalar. Carries appear *only* where there is nothing to read from: an
accumulator that is never emitted per level ([B1](#b1)–[B5](#b5)), a boolean
latch, a counter. The delay-line tuples, level counters and pass-through slots
production code is full of are encodings forced by today's construct, so they
belong in the prose and in the
[workarounds table](#cross-cutting-workarounds), not in a statement of the
pattern. Equivalently: **carry width is a property of a design, not of a
pattern** — which is what makes it a fair thing to measure designs by. For the
same reason boundary rows are *peeled* out of the loop here (`out[0] = …`
before it) rather than branched on inside it: the peeled form is the one a
design should be able to recover.

**Horizontal.** Omitted throughout. Every pattern is per-column and
embarrassingly parallel across columns; where a quantity is column-constant
(a 2D field in the DSL) it appears as a plain scalar parameter (`depth`,
`k_start`) or scalar return value.

### Axes of variation

The catalog is a basis over these axes; a pattern earns an entry only by being
new on at least one of them.

| Axis | Values seen in production |
| --- | --- |
| Carry width | 0 (none) · 1 scalar · small tuple (2–5) · 21–28 components (muphys, cloudsc) |
| Carry content | accumulator · previous *output* (order 1) · previous *outputs* (order $m$) · previous/next *input* (delay line, lookahead) · boolean flag · level counter · level index · ring buffer |
| Output consumption | per level (3D) · final value only (column-constant) · both |
| Body uniformity | uniform · start-anchored region(s) · end-anchored region(s) · data-dependent anchor · per-band (interval dispatch) |
| Neighbour access | current level only · input at $k \pm 1$ · bounded window of $M$ levels |
| Trip count | full K range · static sub-range · per-column data-dependent |
| Coupling | single recurrence · sweep pair (forward then backward) · N independent sweeps — composition itself needs no new syntax, so it earns no entry |
| Reduction op | none · fixed (sum/prod/min/max/any/all) · arg · user-defined combine |

### Index

| ID          | Pattern                                            | Carry today            | Result    | Evidence                                        |
| ----------- | -------------------------------------------------- | ---------------------- | --------- | ----------------------------------------------- |
| [A1](#a1)   | Running accumulation                               | accumulator            | per level | `hydrostatic_exner`, `_trapezoidal_z`, `cumsum` |
| [A2](#a2)   | Affine propagation $y_k = a_k \cdot y_{k-1} + b_k$ | 1 scalar               | per level | radiation fluxes/albedo (×9)                    |
| [A3](#a3)   | Coupled multi-component recurrence                 | tuple                  | per level | Thomas forward elimination (×7)                 |
| [A4](#a4)   | Reverse sweep over a forward sweep's fields        | 1 scalar               | per level | Thomas back-substitution (×7)                   |
| [A5](#a5)   | Order-$m$ / strided recurrence                     | $m$ output levels      | per level | PMAP stride-2 `_tridiagonal`                    |
| [A6](#a6)   | Previous/next *input* level                        | delay-line slots       | per level | muphys `rho`, `t_kp1`; PMAP `b33`               |
| [A7](#a7)   | Monotone activation flag (work skipping)           | flag + state           | per level | muphys species, cloud searches                  |
| [A8](#a8)   | Mixed carry: per-level + column state              | tuple                  | both      | cloudsc main loops (×3)                         |
| [B1](#b1)   | Fixed-op column reduction                          | accumulator            | final     | column integrals, `cc_max`, `v_mat_prod`        |
| [B2](#b2)   | Arg-reduction                                      | value + index          | final     | `i_cloud_top`, PHYEX tropopause                 |
| [B3](#b3)   | First-hit / last-hit level search                  | flag + value           | final     | cloud base/top, PBL height, `tropopause`        |
| [B4](#b4)   | Masked variable-depth reduction                    | accumulator            | final     | Noah LSM root zone                              |
| [B5](#b5)   | Value at a data-dependent level                    | widened carry          | final     | PHYEX `PZZ(ITPL)`, Noah `zsoil(nroot)`          |
| [B6](#b6)   | Column total consumed per level                    | — (two passes)         | per level | `eta_levels`, `tropopause` broadcast            |
| [C1](#c1)   | Start-anchored boundary row                        | + counter              | —         | every tridiagonal solve, ICON `first_level`     |
| [C2](#c2)   | End-anchored / multi-level regions                 | + counter              | —         | PMAP stride-2 (4 regions), `_solve_itcs`        |
| [C3](#c3)   | Init from outside the column                       | field-valued init      | —         | `diagnose_pressure`, surface BCs                |
| [C4](#c4)   | Cross-sweep handover                               | extra field            | —         | PMAP `rp_top`                                   |
| [C5](#c5)   | Data-dependent start level                         | flag + index           | per level | radiation `i_cloud_top` (×4)                    |
| [D1](#d1)   | Bounded window sub-reduce, no state                | none (window only)     | per level | COSMO boxtracking, PPM                          |
| [D2](#d2)   | Bounded window inside a recurrence                 | state + $M$ ring slots | per level | precip-flux schemes                             |
| [E1](#e1)   | Recurrence under an extra data dimension           | ×G                     | per level | radiation g-points ($O(100)$)                   |
| [X1](#x1)   | $O(K^2)$ double loop, multi-level writes, feedback | —                      | —         | SHiELD melting                                  |
| [X2](#x2)   | Data-dependent `while` over a table                | —                      | —         | `f_calc_rh_index`                               |
| [X3](#x3)   | Scatter formulation of a flux                      | —                      | —         | COSMO reference implementation                  |
| [F1–F5](#f) | Fake scans (not recurrences at all)                | —                      | —         | ~85% of all PMAP occurrences                    |

The **carry today** column is how `scan_operator` must encode the pattern now,
not a property of the pattern itself — the minimal forms below carry nothing
they do not have to. Entries are **frontend** patterns: something that has to
be *expressible*. Shapes whose only open question is how the toolchain
executes them are collected under
[composition](#composition-expressible-today-obligations-on-the-toolchain)
instead.

---

## A. Recurrences with per-level output

### A1 {#a1} Running accumulation

$s_k = s_{k-1} \oplus x_k$ with a fixed associative $\oplus$, emitted at every level.

```python
def running_sum(x: Column) -> Column:
    out = np.empty_like(x)
    out[0] = x[0]  # init 0.0 folded into the peeled first level
    for k in range(1, x.size):
        out[k] = out[k - 1] + x[k]
    return out
```

The accumulator *is* the output, so it is read back rather than carried; a
design that cannot read its own output needs one scalar of state. Variants:
`*` (cumulative product), `max`. The *only* pattern the array API covers
(`np.cumulative_sum`), hence the only one a backend can be assumed to have a
primitive for.

Evidence: `hydrostatic_exner` (hydrostatic pressure integral),
`_trapezoidal_z` (column integral — same recurrence, only the final value
consumed → [B1](#b1)), geometric height from `dz` in sedimentation
([D1](#d1)).

### A2 {#a2} Affine propagation

$y_k = a_k \cdot y_{k-1} + b_k$ — the workhorse of two-stream radiation
(transmittance × incoming flux + local source), and of every
back-substitution.

```python
def propagate(a: Column, b: Column, y0: float = 0.0) -> Column:
    out = np.empty_like(b)
    out[0] = a[0] * y0 + b[0]
    for k in range(1, b.size):
        out[k] = a[k] * out[k - 1] + b[k]
    return out
```

$a_k$, $b_k$ are *inputs* (precomputed per level), so the body is pointwise in
them — the pattern is "one multiply-add of the previous output". A design needs
a 1-scalar carry for it, or offset access to the output being written.

Evidence: nine radiation kernels (LW/SW adding, no-scatter up/down, total
albedo, LW derivatives, plus their unified twins); `_solve_itcs` backward half.

### A3 {#a3} Coupled multi-component recurrence

Several carried components, each depending on the others — the irreducible
tuple carry. Minimal instance: Thomas forward elimination of a tridiagonal
system $a_k x_{k-1} + b_k x_k + c_k x_{k+1} = d_k$.

```python
def thomas_forward(a: Column, b: Column, c: Column, d: Column) -> tuple[Column, Column]:
    ee, ff = np.empty_like(b), np.empty_like(b)
    ee[0], ff[0] = -c[0] / b[0], d[0] / b[0]  # a[0] == 0: nothing below
    for k in range(1, b.size):
        den = b[k] + a[k] * ee[k - 1]
        ee[k] = -c[k] / den
        ff[k] = (d[k] - a[k] * ff[k - 1]) / den
    return ee, ff
```

Two mutually dependent components, both also per-level outputs (consumed by
[A4](#a4)) — so this is a 2-scalar carry in any design that cannot read the
output it is writing (6 for PMAP's vector diffusion: three coupled systems).

Evidence: `_solve_itcs` ×3, `_nabla_z_cmp4/6_solve`, `_tridiagonal_init` ×2,
ICON `solve_tridiagonal_matrix_for_w_forward_sweep`.

### A4 {#a4} Reverse sweep over a forward sweep's fields

A second scan, opposite direction, whose inputs are the first scan's per-level
outputs. Structurally [A2](#a2); listed separately because the body shape
differs — it consumes two fields and carries nothing but its own output (the
*pairing* with [A3](#a3) is composition, see
[below](#composition-expressible-today-obligations-on-the-toolchain)).

```python
def thomas_backward(ee: Column, ff: Column) -> Column:
    x = np.empty_like(ff)
    x[-1] = ff[-1]  # ee[-1] == 0: nothing above
    for k in reversed(range(ff.size - 1)):
        x[k] = ee[k] * x[k + 1] + ff[k]
    return x
```

Checks against a dense solve: `thomas_backward(*thomas_forward(a, b, c, d))`
equals `np.linalg.solve(np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1), d)`.

### A5 {#a5} Order-m / strided recurrence

The body reads its own output $m$ levels back, $m > 1$ — two interleaved
elimination chains in PMAP's pre-factored Helmholtz solve ($m = 2$).

```python
def strided(rs: Column, g: Column, dnmi: Column) -> Column:
    ff = np.empty_like(rs)
    ff[:2] = rs[:2] * dnmi[:2]  # the two chain starts: a two-level peeled region
    for k in range(2, rs.size):
        ff[k] = (rs[k] + g[k] * ff[k - 2]) * dnmi[k]
    return ff
```

The dependence is one index read, `ff[k - 2]`. An order-1 scan encodes it as an
$m$-wide carry shifted every step (PMAP's ported version threads
`(ff, ff_km1)` by hand) — equivalent, but the shift is the user's to write
unless a design offers offset access to the output. Combines with
[C1](#c1)/[C2](#c2): an order-$m$ recurrence needs an $m$-level start
region.

Evidence: `helmholtz/preconditioner.py` and `poisson/preconditioner.py`
`_tridiagonal(_init)` (offsets $\pm 2$) — **ported and validated** on gtfn with a
hand-shifted 5-tuple carry.

### A6 {#a6} Previous/next *input* level inside the body

The body needs $x_{k-1}$ (previous level) or $y_{k+1}$ (lookahead) of an
*input*, and the corresponding global shift is invalid at the column end — so
today the value is either carried as a delay line or pre-shifted outside with
a clamp.

```python
def with_neighbour_inputs(x: Column, y: Column) -> Column:
    out = np.empty_like(x)
    n = x.size
    out[0] = x[0] - 0.5 * (x[0] + y[1])  # x[k-1] clamped at the column start
    for k in range(1, n):
        y_next = y[min(k + 1, n - 1)]  # lookahead, clamped at the column end
        out[k] = 0.5 * out[k - 1] + x[k] - 0.5 * (x[k - 1] + y_next)
    return out
```

Both reads are plain indexing here; a scan body sees only the current level, so
each (input, offset) pair costs either a carry slot or an external pre-shift —
pure plumbing, no physics. A lookahead is only a *read* of an input, so it is
legal in a forward scan. Note the clamps: the shift is well defined on the
interior and needs a decision at exactly one end.

Evidence: muphys carries `rho` solely as a delay line and precomputes
`t_kp1 = concat_where(KDim < last_lev, t(Koff[1]), t)` outside; PMAP's
stride-2 solve carries `b33` because the wrapper shift `b33(KDim - 1)` is
evaluated out of domain at $k = 0, 1$; `_cloud_cover_maximum_random` carries
`cf`, `zcr`. Half-level inputs are the exception — `bbz(KDim ∓ 0.5)` exists
for every cell, no carry needed.

### A7 {#a7} Monotone activation flag (work skipping)

A latch in the carry: once a predicate has held at any level already visited,
the expensive body runs for the rest of the column; before that, the output is
the identity.

```python
def activated(q: Column, mask: Mask) -> Column:
    out = np.empty_like(q)
    active: Mask = np.logical_or.accumulate(mask)  # the latch is itself an A1 scan
    out[0] = 0.5 * q[0] if active[0] else q[0]
    for k in range(1, q.size):
        if active[k]:
            out[k] = 0.5 * (out[k - 1] + q[k])  # stand-in for the expensive body
        else:
            out[k] = q[k]  # identity branch
    return out
```

The latch is a cumulative `or` — i.e. an [A1](#a1) over `mask`, hoistable out
of the body entirely, which is what the form above does and what muphys's
in-carry `activated` flag cannot. What remains irreducible is the gated body,
whose `else` branch is the *identity* — the load-bearing fact (muphys needs
~650 lines of DaCe cleanup hooks to recover it); a redesign should let this be
*stated* rather than inferred.

Evidence: muphys `PrecipStateQx.activated` ×4 species; the "not found yet"
gate of [B3](#b3) is the same latch, read inverted.

### A8 {#a8} Mixed carry: per-level outputs + column state

One scan whose carry holds both level-coupled quantities (consumed per level)
and accumulated column state (consumed only at the end).

```python
def mixed(a: Column, x: Column) -> tuple[Column, np.float64]:
    out = np.empty_like(x)
    out[0] = x[0]
    total = out[0]  # the only genuine carry: never emitted per level
    for k in range(1, x.size):
        out[k] = a[k] * out[k - 1] + x[k]  # per-level, level-coupled
        total = max(total, out[k])  # column-constant accumulation
    return out, total  # both consumed
```

The asymmetry is the pattern: the per-level part needs no state beyond the
output it is writing, the column part has nowhere to live but a carry.

Evidence: the cloudsc main loops (cy49r1, cy50r1, dwarf) — per-level fall/sink
fluxes of five species *plus* six column accumulators
(`covptot/covpclr/covpmax/cldtopdist/rainfrac_toprfz/rainliq`); muphys surface
precipitation fluxes. Today the column state is recovered by slicing the last
level of a materialized 3D field at program level.

## B. Column reductions (final value only)

### B1 {#b1} Fixed-op column reduction

[A1](#a1) with only the final value consumed — a column-constant result, i.e.
the K dimension is *removed*.

```python
def column_sum(x: Column) -> np.float64:
    s = np.float64(0.0)
    for k in range(x.size):
        s = s + x[k]
    return s  # one value per column, no 3D output
```

Ops seen: `sum` (column integrals, broadband g-point sums), `max` (cloud
cover, vertical CFL), `prod` (random-overlap cloud cover), `any`/`all`
(masks). Order matters numerically: a bit-reproducible column integral must
pin the sequential order, so `np.sum`'s pairwise summation is *not* a drop-in.

Evidence: `_trapezoidal_z` (**ported**), `_cloud_cover_maximum/_random`,
`i_cloud_top`, `v_mat_prod`. Today: no vertical reduction exists, so each is a
scan writing a full 3D temporary plus a host-side `tmp[:, :, -1]` slice — an
operator-level port, not a stencil-level one.

### B2 {#b2} Arg-reduction

The reduction result is the *level* at which an extremum occurs.

```python
def argmin(x: Column) -> int:
    best, k_best = x[0], 0
    for k in range(x.size):
        if x[k] < best:
            best, k_best = x[k], k
    return k_best
```

Carry: value + index — and the index requires the level number to be
available inside the body, which today it is not (hence hand-rolled counters).

Evidence: PHYEX tropopause (`ZTMIN`, `ITPL`), radiation `i_cloud_top`
(max-reduction over a level index).

### B3 {#b3} First-hit / last-hit level search

Reduce a predicate to "the value at the first (or last) level where it holds",
with a not-found fallback.

```python
def first_hit(h: Column, p: Mask, fill: float) -> np.float64:
    found = False
    val = np.float64(fill)
    for k in range(h.size):
        if not found and p[k]:  # drop `not found` -> last hit instead
            found, val = True, h[k]
    return val
```

Carry: flag + value. The flag is required because the fallback (`fill_value`)
is a *runtime* parameter and cannot be an init constant; it also removes the
fragile `val == fill_value` float comparison production code uses as its
"not found" sentinel.

Evidence: `_cloud_base_height` (first hit), `_cloud_top_height/_temperature`,
`tropopause` (last hit), `_boundary_layer_height` (`rib > 0.25`),
`wind_bl_height` (level bracketing a target height → interpolation between two
adjacent levels, i.e. a 2-level carry).

### B4 {#b4} Masked variable-depth reduction

The trip count is per-column *data*, so no static range applies; the
vectorizable answer is a full-range masked reduction.

```python
def masked_sum(w: Column, depth: int) -> np.float64:  # depth: per-column, runtime
    s = np.float64(0.0)
    for k in range(w.size):
        if k < depth:  # masking, not early exit
            s = s + w[k]
    return s


def masked_sum_vectorized(w: Column, depth: int) -> np.float64:
    return np.float64(np.sum(np.where(np.arange(w.size) < depth, w, 0.0)))  # level index as a value
```

Costs $O(n)$ instead of $O(\mathit{depth})$; the trade is deliberate (early exit kills
vectorization). Needs the level index as a value.

Evidence: Noah LSM root-zone conductance (`nroot`, runtime-varying in the PBL
variant).

### B5 {#b5} Value at a data-dependent level

A gather at an index that is itself a column reduction — `z[k_best]`. Two
encodings: a second pass gathering at a computed index, or widening the
arg-reduction carry so the payload travels with the index.

```python
def value_at_argmin(x: Column, z: Column) -> tuple[int, np.float64]:
    best, k_best, z_best = x[0], 0, z[0]
    for k in range(x.size):
        if x[k] < best:
            best, k_best, z_best = x[k], k, z[k]  # carry the payload too
    return k_best, z_best
```

Evidence: PHYEX `PZZ(JIJ, ITPL(JIJ))` (height of the tropopause level, then
used as a per-level threshold), Noah `zsoil(nroot)`, radiation's use of
`i_cloud_top` ([C5](#c5)).

### B6 {#b6} Column total consumed at every level

A genuine two-pass dependence: reduce the column, then use the
column-constant in a pointwise per-level expression.

```python
def normalize_by_total(x: Column) -> Column:
    total = column_sum(x)  # pass 1: K removed
    return x / total  # pass 2: pointwise, the total broadcast back over K
```

Not a recurrence in the second pass — but production code writes it as a scan
that broadcasts a carried value through the column ([F1](#f)).

Evidence: `eta_levels` (divide by surface pressure), `tropopause` broadcast,
`_dev/cloudsc2` `trpaus`. A fused consumer running *opposite* to the reduction
can keep the total in a register instead of materializing it — a backend
concern, not a user-visible pattern.

## C. Boundary conditions and level-dependent bodies

These are not recurrences of their own; they are modifiers that any A/B
pattern can carry. All of them are spelled with `interval` dispatch in
`gt4py.cartesian` and with in-carry flags or counters in `gt4py.next` today.

### C1 {#c1} Start-anchored boundary row

The first level of the scan runs a *different* body (the boundary-condition
row of a solve, an initialization from other fields).

```python
def start_anchored(x: Column) -> Column:
    out = np.empty_like(x)
    out[0] = x[0]  # the boundary-condition row: a different body
    for k in range(1, x.size):
        out[k] = 0.5 * out[k - 1] + x[k]  # the uniform interior body
    return out
```

The form above is what a design should produce: the row is *peeled*, so the
loop body is branch-free. Today it is instead a `first_level: bool` in the
carry (ICON `test_icon_like_scan`, plus a *dummy bool output field* and
program-level output slicing), or an `int32` counter compared against 0 —
either way the branch runs at every level.

### C2 {#c2} End-anchored and multi-level regions

Regions anchored at the *far* end of the scan, and regions several levels
deep — `interval(-1, None)`, `interval(-2, -1)`, `interval(0, 2)`.

```python
def regions(x: Column) -> Column:  # n >= 5
    out = np.empty_like(x)
    n = x.size
    out[:2] = x[:2]  # two-level start region
    for k in range(2, n - 2):  # branch-free interior
        out[k] = 0.5 * out[k - 1] + x[k]
    out[n - 2] = out[n - 3] - x[n - 2]  # second to last: its own body
    out[n - 1] = out[n - 2] + x[n - 1]  # last level: another body
    return out
```

Four regions, four bodies, one branch-free loop — the partition a design must
produce (peeling "the first iteration" is not enough). Five regions occur in
one PMAP solve. The end anchor
needs the column length inside the body (today: `k_top = last_idx[2] -
first_idx[2]` injected as a scalar and compared against a carried counter; or
muphys smuggling a runtime `last_lev` in). Extents must be
**guard-refined**: an `x[-2]` read that only occurs inside the $k = n - 1$
branch must not clip the scan at the column start.

### C3 {#c3} Init from outside the column

The seed is not a compile-time constant but a field — surface pressure, a
surface flux, a lower boundary condition.

```python
def seeded(dz_rho_g: Column, p_surface: float) -> Column:
    out = np.empty_like(dz_rho_g)
    out[0] = p_surface * np.exp(dz_rho_g[0])  # the boundary value *is* the init
    for k in range(1, dz_rho_g.size):
        out[k] = out[k - 1] * np.exp(dz_rho_g[k])
    return out  # n values; np.concatenate([[p_surface], out]) is the n+1 inclusive form
```

Two sub-cases: the exclusive form above ($n$ outputs) and the inclusive form
($n+1$ values — the boundary value itself is emitted, which lands on the
half-level/interface grid and therefore couples to the staggering design).
Today `init` must be compile-time, so the field value is smuggled in as a
scanned input plus a `first_level` flag that selects it.

Evidence: icon4py `diagnose_pressure` (carry `(p, p_ifc, is_first_level)`,
init `(0.0, 0.0, True)`, one slot never read back); `_solve_itcs`'s surface-flux
BC; `_cloud_base_height`'s runtime `fill_value` ([B3](#b3)).

### C4 {#c4} Cross-sweep handover

One sweep needs a value derived from levels the *other* sweep has already
visited and it has not.

```python
def strided_solve(rs: Column, g: Column, dnmi: Column) -> Column:  # n >= 3
    n, ff, rp = rs.size, np.empty_like(rs), np.empty_like(rs)

    ff[:2] = rs[:2] * dnmi[:2]  # sweep 1 (upward) = A5
    for k in range(2, n):
        ff[k] = (rs[k] + g[k] * ff[k - 2]) * dnmi[k]

    # sweep 2 (downward) starts here, and its very first row needs ff two levels
    # *ahead* of itself -- a level sweep 2 only reaches at the end of its own loop
    rp[n - 1] = (ff[n - 3] * 2.0 * g[n - 1] + rs[n - 1]) * dnmi[n - 1]
    rp[n - 2] = ff[n - 2]
    for k in reversed(range(n - 2)):
        rp[k] = g[k] * rp[k + 2] + ff[k]
    return rp
```

Both loops are shown because the pattern *is* the dependence between them: at
its first iteration sweep 2 reads `ff[n - 3]` and `g[n - 1]`, levels it has not
reached yet. NumPy just indexes them — the two sweeps communicate through the
whole `ff` array. A scan body sees only the current level and its carry, so the
value has to be computed in the sweep where it *is* visible and handed over as
a field: PMAP's forward sweep writes `rp_top` at every level, and the backward
sweep reads it at its first one.

Evidence: PMAP's stride-2 back-substitution, whose top row needs `ff[k-2]` and
`b33[k-1]` from below (**ported** — `rp_top` is the handover field).
Generalizes: anything a sweep needs from ahead of itself, boundary rows being
where it shows up, since that is where "ahead" is furthest away.

### C5 {#c5} Data-dependent start level

The recurrence starts mid-column, at a level that is runtime data (cloud top),
with the levels before it passed through.

```python
def start_at(x: Column, k_start: int, seed: float) -> Column:
    # k_start: per column, any level in 0 .. n-1, or n for "this column never starts"
    out = np.zeros_like(x)
    if k_start < x.size:
        out[k_start] = seed  # the recurrence starts mid-column, at a level that is data
    for k in range(k_start + 1, x.size):
        out[k] = 0.5 * out[k - 1] + x[k]
    return out
```

`k_start` is **unbounded**: any level of the column, differing per column and
per timestep, because it is itself a [B2](#b2) arg-reduction (the uppermost
cloudy layer; read back as in [B5](#b5)). It is not a small or near-boundary
value, so nothing static can be peeled, and the number of levels after it
ranges over all of 0…n. A column where the predicate never holds (clear sky)
must also be legal — then the recurrence never starts and the whole output is
the pass-through value, which is why the form above tolerates
`k_start == n`.

NumPy can simply start its loop at `k_start`; a scan has to run the full range
and gate the body on a started-flag, since the trip count must stay static. So
this is *not* [B4](#b4)'s data-dependent trip count — the range is full, the
*body* is gated — and it needs the level index as a value.

Evidence: radiation `lw:664+689` and `lw:718+730` (above-cloud flux up, cloudy
flux down, both keyed on `i_cloud_top`, two scans each that should fuse into
one); PHYEX mixing length (gated on the height at the tropopause level).

## D. Windowed patterns (scan + sub-reduce)

### D1 {#d1} Bounded window sub-reduce, no state feedback

Per level, a reduction over a *bounded* window of neighbouring levels:
$y_k = \bigoplus_{m=0}^{M-1} g(x_k,\, x_{k-m},\, x_{k-m-1})$. With a static
bound $M$
this is **not a recurrence at all** — it is a vertical stencil with a window
plus a reduction, which is what the two equal forms below demonstrate.

```python
def box_flux(phi: Column, z: Column, v: Column, dt: float, m_max: int) -> Column:
    out = np.zeros_like(phi)
    for k in range(phi.size):
        acc = np.float64(0.0)
        for m in range(m_max):  # static bound = max vertical Courant number
            if k - m - 1 < 0:
                continue  # window reaches past the model top: skip-masked
            z_up = min(z[k - m - 1], z[k] + abs(v[k - m]) * dt)
            z_low = min(z[k - m], z_up)
            acc += phi[k - m] * (z_up - z_low)  # out-of-reach terms are exactly 0
        out[k] = -acc
    return out


def box_flux_vectorized(phi: Column, z: Column, v: Column, dt: float, m_max: int) -> Column:
    n = phi.size
    acc = np.zeros_like(phi)
    for m in range(min(m_max, n - 1)):  # unrolled window: no k-loop, no carry, no state
        dst, src, above = slice(m + 1, n), slice(1, n - m), slice(0, n - 1 - m)
        z_up = np.minimum(z[above], z[dst] + np.abs(v[src]) * dt)
        z_low = np.minimum(z[src], z_up)
        acc[dst] += phi[src] * (z_up - z_low)
    return -acc
```

Three facts from the COSMO case study: the formula **self-masks** (out-of-reach
levels contribute exactly zero, so no early exit is needed for correctness);
the rolling buffers production code uses are a *memory* optimization
(k-caches), not semantics; and any cumulative quantity in the window
(geometric height from `dz`) is a separate [A1](#a1) composed in front.

Evidence: COSMO/ICON two-moment "boxtracking" sedimentation (eq. 12), ICON
`mo_2mom_mcrph_processes.f90` (tuple-valued `phi`: mass *and* number density),
PPM / semi-Lagrangian vertical remapping (FV3),
`ppm/common/cell_face_values.py` in its reconstruction part. Today: hand-rolled
ring buffers in a scan carry.

### D2 {#d2} Bounded window inside a recurrence

Same window, but the contribution feeds or reads running state, so the scan is
genuine. The window itself is over *inputs*, hence plain slicing here; a scan
body encodes it as $M$ extra carry slots (a ring buffer) — which, as the
sedimentation case study notes, is a k-cache a compiler should introduce, not a
user.

```python
def windowed_scan(x: Column, m_max: int) -> Column:
    out = np.empty_like(x)
    out[0] = x[0]
    for k in range(1, x.size):
        window = x[max(k - m_max + 1, 0) : k + 1]  # the last M input levels
        out[k] = 0.5 * out[k - 1] + np.sum(window)  # sub-reduce feeding the state
    return out
```

Evidence: precipitation-flux schemes where the flux entering a cell is carried
*and* the hydrometeor state of the previous $M$ levels is needed.

## E. Extra dimensions

### E1 {#e1} Recurrence under an extra data dimension

The recurrence is per `(column, g-point)`: a vertical scan applied pointwise
over an extra, non-spatial dimension of $O(100)$ entries. Tuple unrolling does
not scale at that size.

```python
def per_gpoint(a: Stack, b: Stack) -> Stack:  # shape (n_gpoints, n), n_gpoints ~ 100-200
    return np.stack([propagate(a[g], b[g]) for g in range(a.shape[0])])
```

Evidence: all radiation flux kernels (`N_GPOINTS_LW/SW`); PMAP pins a gt4py
*fork* (`loops-over-data-dims`) for the cartesian versions, and the broadband
sums additionally need a reduction over that dense dimension.

## X. Out of scope — documented boundaries

### X1 {#x1} $O(K^2)$ double loop with multi-level writes and feedback

An outer loop over source levels, an inner loop over the levels below, writing
at **both** indices, with data-dependent `break`s and feedback through the
written state.

```python
T_MELT = 273.15


def melt(qice: Column, qrain: Column, temperature: Column) -> None:
    for k in range(qice.size):  # source levels, upward
        if qice[k] <= 0.0:
            continue
        for m in range(k - 1, -1, -1):  # receiving levels, downward
            if temperature[m] < T_MELT:
                break  # data-dependent exit
            amount = min(qice[k], temperature[m] - T_MELT)  # reads *updated* state
            qice[k] -= amount  # write at k
            qrain[m] += amount  # write at m
            temperature[m] -= amount  # ... and re-read by later k iterations
            if qice[k] <= 0.0:
                break
```

Not a first-order recurrence under any reindexing: $O(K^2)$ traversal, writes to
levels other than the current one, and the order-dependent `min` cap reads
intermediate state — so no masked bounded window ([D1](#d1)) computes the same
answer. Options: reformulate in flux form (what COSMO did for exactly this
physics — it changes the operation order, so results differ and need
validation), or keep it as sequential column code outside the DSL.

Evidence: SHiELD microphysics melting (`physics_patterns` PR #2). Nothing in
PMAP currently has this shape.

### X2 {#x2} Data-dependent `while` over a lookup table

Not vertical at all, but it blocks the same ports: a search over fixed bins
inside a per-level function, plus indexed access into a lookup table.

```python
def rh_lookup(rh: float, bounds: Column, table: Column) -> np.float64:
    # bounds: ascending upper edge of each humidity bin, last one +inf
    # table: one tabulated coefficient per bin (N_RH rows, compile-time constant)
    i = 0
    while rh > bounds[i]:  # (1) data-dependent search for the bin
        i += 1
    return np.float64(table[i])  # (2) what the index is for: a computed-index gather
```

The index selects a row of a *static* table of aerosol optical properties per
humidity bin — mass extinction coefficient, single-scattering albedo,
asymmetry factor (`rh_lower.A[irh]` in the source) — so the pattern has two
halves and a `field_operator` supports neither: the data-dependent `while`,
and the gather at a computed index.

Both are bounded by a compile-time constant, so the pair unrolls into one
masked select over the bins — $\sum_i \mathrm{where}(\text{first covering bin
is } i,\ \mathit{table}_i,\ 0)$— which is branch-free and needs no indexing;
alternatively the index field is precomputed outside. Evidence:
`functions_aerosols.py::f_calc_rh_index`.

### X3 {#x3} Scatter formulation of a flux

The work-efficient form of [D1](#d1): an outer loop over source levels with an
inner `while` depositing contributions into the faces below — $O(n \cdot \bar{M})$
instead of $O(n \cdot M)$, but it writes several output levels per step and does not vectorize
(the COSMO paper documents exactly this on NEC). Deliberately out of scope;
the gather form with a static bound is the supported one.

## F. {#f} Fake scans — written as recurrences, aren't

Roughly 85% of the `with computation(FORWARD/BACKWARD)` blocks surveyed in
PMAP need no recurrence at all. They are catalogued because a redesign is
partly judged by how obviously it steers these *away* from a scan. Each row's
"rewrite" column is the array-level form the loop is equal to.

| ID | Shape | Scan-shaped original | Rewrite |
| --- | --- | --- | --- |
| F1 | **Broadcast of one level** — the carry copies a single value through the column | `s = x[-1]`; `for k: out[k] = y[k] / s` | `out = y / x[-1]` ([B6](#b6)) |
| F2 | **Carry equals an existing field** — the carried temporary provably equals an input at `k-1` | `for k: out[k] = y[k] - prev; prev = y[k]` | `out[1:] = y[1:] - y[:-1]` |
| F3 | **Self-read across disjoint intervals** — edge levels read interior levels written by *another* interval | interior pass, then `out[0] = out[1]`, `out[-1] = out[-2]` | two passes: interior, then edge fill |
| F4 | **Ordering only because of input shifts** — the body reads *inputs* at $k \pm 1$, never its own output | `for k: out[k] = x[k] + x[k+1]` | `out[:-1] = x[:-1] + x[1:]` + boundary select |
| F5 | **Single-level write** — `interval(0, 1)` only; the sweep is a no-op ordering | `out[0] = f(x[0])`, nothing else | a pure 2D computation, no K at all |

Evidence and per-case rewrites: see
[[personal/tehrengruber/scan-design/scan-occurences-pmap|the PMAP survey appendix]].

## Composition: expressible today, obligations on the toolchain

Three shapes that look like entries but are not *frontend* patterns: in each
one the recurrences involved are already in the catalog, composing them needs
nothing new to say, and what is actually at stake is whether the toolchain
fuses them. They are recorded here because the evidence is what forces the
fusion requirement.

- **Several independent recurrences over one column, sharing per-level work.**
  Each needs only its own output; only the shared per-level expression
  (`shared` below) is common.

  ```python
  def independent_recurrences(q: Stack, rho: Column) -> Stack:
      out = np.empty_like(q)
      out[:, 0] = q[:, 0]
      for k in range(1, q.shape[1]):
          shared = np.sqrt(rho[k])  # per-level work used by all of them, computed once
          out[:, k] = out[:, k - 1] * shared + q[:, k]  # independent recurrences
      return out
  ```

  Writing them separately is legal today; it is *performance* that is not, so
  production fuses by hand. icon4py muphys graupel is four species
  sedimentations plus a temperature/energy recurrence in **one**
  `scan_operator` with a **21-component carry** (2 pass-through slots, 1 delay
  line); its predecessor `_icon_graupel_scan` had a 28-tuple carry, an in-carry
  level counter, `sys.setrecursionlimit(350000)` and 13 of 28 outputs
  discarded; cloudsc is the same at ~2000 lines with unrolled species loops.
  **This is the scalability case**: every entry in the catalog is small, and
  production code is their composition — so a design is only usable if the
  small forms compose without hand-fusion.
- **Opposite-direction pair** ([A3](#a3) then [A4](#a4), as in every
  tridiagonal solve — 7 in PMAP, plus radiation's down/up flux pairs and
  `tropopause`): two scans, written as two scans. The intermediate `ee`, `ff`
  are materialized as full columns, which is unavoidable without re-deriving
  them; fusion across opposite directions is generally not possible.
- **Same-direction cascade** (recurrence → pointwise → reduction, e.g.
  sedimentation's `cumsum` of `dz` in front of the window ([D1](#d1)),
  `_trapezoidal_z`'s weights between integrand and column sum, radiation's
  per-level optics before the flux scan): plain function composition in the
  frontend. Whether the intermediate ever exists, or the chain becomes one
  k-loop, is the open fusion question.

One frontend-visible residue of composition *is* real and recorded in the
table below: gtfn requires a scan to be the whole expression of a closure, so
taking one element of a tuple-valued scan needs a trivial exact `* 1.0`.

## Cross-cutting workarounds

Every item below is something production code carries *because* the current
construct lacks a feature — a checklist of things a redesign should make
disappear, independent of the patterns above.

| Workaround | Caused by | Seen in |
| --- | --- | --- |
| `int32` level counter in the carry | no level index inside the body | PMAP ports, `_icon_graupel_scan`, every arg-reduction |
| `first_level: bool` in the carry + branch at every level | no boundary-region dispatch | ICON `test_icon_like_scan`, `diagnose_pressure` |
| Runtime `last_lev` / `k_top` scalar smuggled in | no end anchor | muphys, PMAP ports |
| Delay-line carry slot for an input | no neighbour-level read in the body | muphys `rho`, PMAP `b33` |
| External pre-shift with a clamp (`concat_where(KDim < last_lev, t(Koff[1]), t)`) | same, for lookahead | muphys `t_kp1` |
| Pass-through carry slot (carried only to be emitted) | carry *is* the output | muphys `pflx_tot`, `t` |
| Dummy output field + program-level output slicing | same, inverted | ICON `test_icon_like_scan` |
| 3D materialization + last-level slice for a column-constant | no vertical reduction, no K-removing output | `_trapezoidal_z`, cloud base/top, muphys surface fluxes |
| `where`-blend instead of a runtime `if` (`enabled * new + (1 - enabled) * old`) | runtime scalar conditions | PMAP `_update_sflux_next` |
| `* 1.0` to give a scan its own closure | gtfn scan-at-top-level-only | all three PMAP ports |
| `*_scalar` field-operator twins | scalar element body | muphys |
| `sys.setrecursionlimit(350000)`, ~650 lines of DaCe repair hooks | 21–28-component carries, identity `else` branches | muphys, `_icon_graupel_scan` |
| `broadcast(x2d, (IDim, JDim, KDim))` for every 2D argument | all scan arguments must carry K | PMAP ports |
| Manual fusion into one mega-scan | no fusion guarantee across scans | muphys, cloudsc |

## Next steps

1. **Validate the catalog against the design axes** — for each entry: can it
   be written without any workaround from the table above, and how wide does
   its carry get?
2. **Promote the minimal forms to an acceptance suite** — they are runnable
   NumPy; pairing each with its desired DSL spelling (and with the equalities
   already noted for [A4](#a4), [B4](#b4), [B6](#b6), [D1](#d1),
   [F1–F5](#f)) gives a test suite that is independent of the eventual API.
3. **Decide the scope line** — [X1](#x1)–[X3](#x3) are the claimed boundary;
   confirm that nothing else in PMAP, icon4py or `physics_patterns` crosses it.
4. **Then** compare designs: this document's successor should be the API
   proposal measured against the entries, cross-checked with
   [[personal/havogt/scan-redesign|havogt's scan redesign]] (whose constructs
   already target A1–A6, B1–B6, C1–C3 and D1–D2 explicitly — the overlap
   is deliberate, and conflicts should be called out per entry rather than per
   document).
