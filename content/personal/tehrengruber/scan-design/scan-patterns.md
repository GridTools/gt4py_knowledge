---
title: Vertical scan pattern catalog
author: sf-n, tehrengruber
tags:
  - scan
  - vertical
  - reduction
  - boundary-conditions
  - windows
  - fusion
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
> against which a design
> ([[personal/tehrengruber/scan-design/scan-design|ours]], or
> [[personal/havogt/scan-redesign|havogt's scan redesign]]) is measured.

- **Status**: draft, collection phase.
- **Companion documents**:
  [[personal/tehrengruber/scan-design/scan-design|the design document]]
  (how the patterns can be expressed in gt4py; the cross-cutting workarounds
  table);
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
independently of how. The examples are **fabricated** — minimal in text and
code — but each carries everything its pattern needs; only the data flow is
the point.

Shared preamble:

```python
from typing import Literal

import numpy as np

Column = np.ndarray[tuple[int], np.dtype[np.float64]]  # one column: shape (n,), n = levels
Mask = np.ndarray[tuple[int], np.dtype[np.bool_]]
Stack = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # several columns: shape (m, n)
```

**References.** Every entry ends with a `References:` line of
`[project/component/N]` labels; each label links straight to the cited code
line (commit-pinned), and the URLs live in one place,
[References](#references). Every true scan of PMAP and icon4py is cited in
the entry whose pattern it instantiates; pace's 161 true scans are cited by
one exemplar per family.

**Horizontal.** Omitted throughout. Every pattern is per-column and
embarrassingly parallel across columns; where a quantity is column-constant
(a 2D field in the DSL) it appears as a plain scalar parameter (`depth`,
`k_start`) or scalar return value.

### Axes of variation

The catalog is a basis over these axes; a pattern earns an entry only by being
new on at least one of them.

| Axis               | Values seen in production                                                                                                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Carry width        | 0 (none) · 1 scalar · small tuple (2–5) · 21–28 components (muphys, cloudsc)                                                                                                          |
| Carry content      | accumulator · previous *output* (order 1) · previous *outputs* (order $m$) · previous/next *input* (delay line, lookahead) · boolean flag · level counter · level index · ring buffer |
| Output consumption | per level (3D) · final value only (column-constant) · both                                                                                                                            |
| Body uniformity    | uniform · start-anchored region(s) · end-anchored region(s) · data-dependent anchor · per-band (interval dispatch)                                                                    |
| Neighbour access   | current level only · input at $k \pm 1$ · bounded window of $M$ levels                                                                                                               |
| Trip count         | full K range · static sub-range · per-column data-dependent                                                                                                                           |
| Coupling           | single recurrence · sweep pair (forward then backward) · N independent sweeps — composition itself needs no new syntax, so it earns no entry                                          |
| Reduction op       | none · fixed (sum/prod/min/max/any/all) · arg · user-defined combine                                                                                                                  |

### Index

| ID        | Pattern                                            | Carry today            | Result    | References                                    |
| --------- | -------------------------------------------------- | ---------------------- | --------- | --------------------------------------------- |
| [A1](#a1) | Recurrence on own output (any body, width, order)  | shifted state tuple    | per level | solves, radiation fluxes, integrals, stride-2 |
| [A2](#a2) | Monotone activation flag (work skipping)           | flag + state           | per level | muphys species, cloud searches                |
| [B1](#b1) | Fixed-op column reduction                          | accumulator            | final     | column integrals, `cc_max`, `v_mat_prod`      |
| [B2](#b2) | Arg-reduction + value at the found level           | value + index          | final     | `i_cloud_top`, PHYEX `ZTMIN`/`PZZ(ITPL)`      |
| [B3](#b3) | First-hit / last-hit level search                  | flag + value           | final     | cloud base/top, PBL height, `tropopause`      |
| [B4](#b4) | Masked variable-depth reduction                    | accumulator            | final     | Noah LSM root zone                            |
| [C1](#c1) | Boundary rows: start/end-anchored regions          | + counter              | —         | PMAP stride-2 (5 regions), `_solve_itcs`      |
| [C2](#c2) | Init from outside the column                       | field-valued init      | —         | `diagnose_pressure`, surface BCs              |
| [C3](#c3) | Data-dependent start level                         | flag + index           | per level | radiation `i_cloud_top` (×4)                  |
| [D1](#d1) | Bounded window sub-reduce, no state                | none (window only)     | per level | COSMO boxtracking, PPM                        |
| [D2](#d2) | Bounded window inside a recurrence                 | state + $M$ ring slots | per level | precip-flux schemes                           |
| [E1](#e1) | Recurrence under an extra data dimension           | ×G                     | per level | radiation g-points ($O(100)$)                 |
| [X1](#x1) | $O(K^2)$ double loop, multi-level writes, feedback | —                      | —         | SHiELD melting                                |
| [X2](#x2) | Data-dependent `while` over a table                | —                      | —         | `f_calc_rh_index`                             |

The **carry today** column is how `scan_operator` must encode the pattern now,
not a property of the pattern itself — the minimal forms below carry nothing
they do not have to. Entries are **frontend** patterns: something that has to
be *expressible*. Shapes whose only open question is how the toolchain
executes them are collected under
[composition](#composition-expressible-today-obligations-on-the-toolchain)
instead.

---

## A. Recurrences with per-level output

### A1 {#a1} Recurrence on its own output

$\mathit{state}_k = f(\mathit{state}_{k-1},\ \ldots,\ \mathit{state}_{k-m},\ \mathit{inputs}_k)$,
$m$ small and static (usually 1).

```python
# example 1
def recurrence(x: Column, y: Column) -> tuple[Column, Column]:
    u, v = np.empty_like(x), np.empty_like(x)
    u[:2], v[:2] = x[:2], y[:2]  # m-level start region: the boundary pattern C1
    for k in range(2, x.size):
        u[k] = x[k] - 0.5 * u[k - 1] * v[k - 1]
        v[k] = y[k] + u[k - 2] * v[k - 1]
    return u, v
```

**Staggered output.** The recurrence may live on the dual grid: $n$ cell
inputs produce $n + 1$ interface values, the first being the init itself
([C2](#c2)) [icon4py/diagnostics/1], [icon4py/thermodynamics/1]:

```python
# example 2 — pseudo-numpy: half indices mark the interface (staggered) grid
p[-1/2] = p_surface
for k in range(n):
    p[k + 1/2] = p[k - 1/2] + rho[k] * g * dz[k]
```

The same on integer indices:

```python
# example 3
def inclusive(w: Column, seed: float) -> Column:  # n inputs -> n + 1 outputs
    out = np.empty(w.size + 1)
    out[0] = seed  # the init is itself the first output level
    for k in range(w.size):
        out[k + 1] = out[k] + w[k]
    return out
```

**Neighbour input levels.** The body may also read *inputs* at $k \pm 1$ — a
lookahead is only a read, legal in a forward scan; each offset shrinks the
range the body is defined on by one level (group C's business):

```python
# example 4
def neighbour_inputs(x: Column, y: Column) -> Column:
    out = np.zeros_like(x)  # boundary levels: group C
    for k in range(1, x.size - 1):
        out[k] = 0.5 * out[k - 1] + x[k - 1] - y[k + 1]
    return out
```

References (every true scan of this shape; pace families by one exemplar):

- running integral $u_k = u_{k-1} + x_k$ [pmap/dycore/4],
  `_accumulate_from_top` [icon4py/common/2]; pace's hydrostatic/height prefix
  sums [pace/fv3/4], including a 2D-carry variant that materializes only the
  derived per-level product [pace/fv3/5];
- affine flux propagation $y_k = a_k \cdot y_{k-1} + b_k$: LW adding
  [pmap/radiation/1], flux down [pmap/radiation/5], no-scatter up
  [pmap/radiation/6], total albedo LW [pmap/radiation/7] / SW
  [pmap/radiation/8];
- Thomas pairs: `_solve_itcs` ×3 [pmap/dycore/1], [pmap/dycore/11],
  [pmap/dycore/12] (the last: three coupled systems); icon4py's shared solver
  — three sweep/substitution pairs — [icon4py/common/1], wrapped by the
  w-solver [icon4py/dycore/2]; pace: `sim1_solver` [pace/fv3/1],
  `tridiag_solve` (+ a column-masked variant) [pace/ndsl/1],
  `tridit`/`tridi2`/`tridin` (second RHS in a tracer data dimension)
  [pace/shield/13], the remap spline edges [pace/fv3/2], `updatedzd`'s edge
  profile (coefficients from a host-side K loop) [pace/fv3/3], and the
  satmedmf setups staging the next level's row in 2D carries [pace/shield/4];
- stride-2 ($m = 2$): helmholtz/poisson `_tridiagonal` [pmap/dycore/2],
  [pmap/dycore/13] and `_tridiagonal_init` [pmap/dycore/14],
  [pmap/dycore/15], the helmholtz pair ported and validated;
- running clamp against own output, $u_k = \max(u_{k+1} + c,\ x_k)$:
  `update_dz_c` [pace/fv3/6];
- time-implicit sedimentation $q_k = (q_k + d_{k-1} q_{k-1}) / (dz_k + d_k)$
  with flux prefix sums, ×4 species [pace/shield/1];
- negative-tracer borrow cascades — the carry is the deficit passed to the
  next level [pace/fv3/7];
- convective parcel recurrences (entrainment/detrainment; all
  [C3](#c3)-bounded): updrafts [pace/shield/2], downdrafts seeded at the
  radiation top [pace/shield/3];
- the dry-adjustment `m_loop` — one backward sweep, a ~15-field carry, five
  anchored top rows ([C1](#c1)) [pace/fv3/8].

### A2 {#a2} Monotone activation flag (work skipping)

```python
# example 5
def activated(q: Column, mask: Mask) -> Column:
    out = np.empty_like(q)
    active: Mask = np.logical_or.accumulate(mask)  # the latch is itself an A1 scan
    out[0] = 0.5 * q[0] if active[0] else q[0]
    for k in range(1, q.size):
        # the active branch stands in for the expensive body; the else is the identity
        out[k] = 0.5 * (out[k - 1] + q[k]) if active[k] else q[k]
    return out
```

Muphys's in-carry `activated` flag is the unhoisted encoding of the latch,
with ~650 lines of repair hooks to recover the identity `else`; a redesign
should let it be *stated* rather than inferred.

References: the per-species `activated` latch [icon4py/muphys/2]; pace's
`stop_k` melt latch and ×4 `no_fall` latches [pace/shield/5]; the `flg` latch
inside every pace level search, and the "not found yet" gate of [B3](#b3) —
the same latch, read inverted.

## B. Column reductions (final value only)

### B1 {#b1} Fixed-op column reduction

[A1](#a1) accumulation with only the final value consumed.

```python
# example 6
def column_sum(x: Column) -> np.float64:
    return np.sum(x)  # K removed: one value per column
```

Ops seen: `sum` (column integrals, broadband g-point sums), `max` (cloud
cover, vertical CFL), `prod` (random-overlap cloud cover), `any`/`all`
(masks). One numerical caveat: `np.sum` is pairwise, so a bit-reproducible
column integral additionally pins the accumulation order — whether the
reduction order is specified is a design decision, not part of the pattern.

References: `_trapezoidal_z` (**ported**) [pmap/dycore/5], the cloud-cover
maximum [pmap/dycore/6] and random-overlap product [pmap/dycore/16], the
`v_mat_prod` cumulative product [pmap/radiation/12]; four independent energy
integrals in one field operator, the column value at the last level
[icon4py/tmx/1]; pace's masked momentum/pressure sums, re-broadcast by copy
scans [pace/fv3/9], `fix_tracer`'s mass sums and rescale [pace/fv3/7], and
NDSL's `column_max`/`column_min` library helpers [pace/ndsl/2].

### B2 {#b2} Arg-reduction

```python
# example 7
def argmin_gather(x: Column, z: Column) -> tuple[np.float64, Mask]:
    best, k_best = x[0], 0
    for k in range(1, x.size):
        if x[k] < best:
            best, k_best = x[k], k
    return z[k_best], np.arange(x.size) < k_best
```

References: the tropopause — argmin of temperature, then the height at that
level used as a per-level threshold [phyex/1]; `i_cloud_top`, a max-reduction
consumed as a start level ([C3](#c3)) [pmap/radiation/3]; `kb`/`hmax` — argmax
of moist static energy, the index consumed as a gather address throughout the
scheme [pace/shield/6]; the argmin of radiative cooling → `krad`
[pace/shield/7]; NDSL's `column_max`/`column_min` return (value, index)
[pace/ndsl/2].

### B3 {#b3} First-hit / last-hit level search

```python
# example 8
def first_hit(h: Column, p: Mask, fill: float) -> np.float64:
    for k in range(h.size):  # reversed(range(...)) -> last hit
        if p[k]:
            return np.float64(h[k])
    return np.float64(fill)
```

A data-dependent trip count does not vectorize; the vectorizable form runs the
full column with a monotone latch ([A2](#a2)).

References: cloud base height [pmap/dycore/7] (first hit), cloud top height
[pmap/dycore/17] and temperature [pmap/dycore/18] (last hit), the tropopause
[pmap/microphysics/5] (last hit), the PBL height (`rib > 0.25`, interpolated
between the bracketing levels) [pmap/dycore/8], and `wind_bl_height`
[pmap/dycore/19] (the same bracketing → interpolation between two adjacent
levels, i.e. a 2-level carry); the level-bracketing `_compute_param` — counter
+ latch carry; production runs the equivalent host-side loop
[icon4py/metrics/1]; pace's convection/PBL searches — the LFC `kbcon` with a
not-found fallback [pace/shield/9], critical-Richardson bracketing +
interpolation [pace/shield/8], cloud-deck top/base `lcld`/`kcld`
[pace/shield/7].

### B4 {#b4} Masked variable-depth reduction

```python
# example 9
def masked_sum(w: Column, depth: int) -> np.float64:  # depth: per-column, runtime
    s = np.float64(0.0)
    for k in range(depth):  # data-dependent trip count
        s = s + w[k]
    return s


def masked_sum_vectorized(w: Column, depth: int) -> np.float64:
    return np.float64(np.sum(np.where(np.arange(w.size) < depth, w, 0.0)))
```

References: the root-zone conductance, `nroot` runtime-varying in the PBL
variant [noaa/lsm/1]; seven two-index integrals in pace's shallow-convection
scheme (TKE mean, CIN, cloud work function, …) with runtime per-column bounds
`kb`/`kbcon`/`ktcon` [pace/shield/10].

## C. Boundary conditions and level-dependent bodies

These are not recurrences of their own; they are modifiers that any A/B
pattern can carry.

### C1 {#c1} Boundary rows: start- and end-anchored regions

The first and last levels of the scan run *different* bodies — the
boundary-condition rows of a solve, initializations from other fields — and a
region may be several levels deep (`interval(0, 2)`, `interval(-2, -1)`).

```python
# example 10
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

This partition is what a design must produce — peeling "the first iteration"
is not enough, and the end anchor needs the column length inside the body.
Extents must be **guard-refined**: an
`x[-2]` read that only occurs inside the $k = n - 1$ branch must not clip the
scan at the column start — the stride-2 back-substitution is exactly this, its
top row reading the forward sweep's `ff` and `b33` at offsets no other level
touches (**ported**).

References: the `first_level` boundary row [icon4py/dycore/1]; the surface
rows of the PMAP solves and their ports [pmap/dycore/10]; five regions in one
solve [pmap/dycore/2]; the end anchor [icon4py/muphys/5]; `sim1_solver`'s
distinct start/end rows [pace/fv3/1]; `m_loop`'s five anchored top rows
[pace/fv3/8].

### C2 {#c2} Init from outside the column

The seed is not a compile-time constant but a field — surface pressure, a
surface flux, a lower boundary condition.

```python
# example 11
def seeded(dz_rho_g: Column, p_surface: float) -> Column:
    out = np.empty_like(dz_rho_g)
    out[0] = p_surface * np.exp(dz_rho_g[0])  # the boundary value *is* the init
    for k in range(1, dz_rho_g.size):
        out[k] = out[k - 1] * np.exp(dz_rho_g[k])
    return out
```

Whether the seed itself is also emitted — the $n + 1$-output form — is
[A1](#a1)'s staggered output, independent of where the seed comes from.
References: `diagnose_pressure` — carry `(p, p_ifc, is_first_level)`, init
`(0.0, 0.0, True)`, one slot never read back [icon4py/diagnostics/1], its live
successor `_scan_pressure` [icon4py/thermodynamics/1]; `_solve_itcs`'s
surface-flux BC [pmap/dycore/1]; pace's surface seeds — `hs`/`zs` under the
height integrals [pace/fv3/4], `ws` in `sim1_solver` [pace/fv3/1], `qs` in the
remap spline [pace/fv3/2]; [B3](#b3)'s runtime `fill_value`.

### C3 {#c3} Data-dependent start level

```python
# example 12
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
cloudy layer; read back as in [B2](#b2)). It is not a small or near-boundary
value, so nothing static can be peeled. A column where the predicate never holds (clear sky) must also be legal.

References: the above-cloud and cloudy flux scans, keyed on `i_cloud_top`
[pmap/radiation/2] and [pmap/radiation/11]; the mixing-length recurrence,
gated on the height at the tropopause level [phyex/2]; every pace
convection/PBL recurrence, bounded by `kb`/`kbcon`/`ktcon`/`kpbl` index fields
[pace/shield/2] — sharpest `mfscu_s3`, *seeded* at the runtime radiation-top
level [pace/shield/3].

## D. Windowed patterns (scan + sub-reduce)

### D1 {#d1} Bounded window sub-reduce, no state feedback

Per level, a reduction over a *bounded* window of neighbouring levels:
$y_k = \bigoplus_{m=0}^{M-1} g(x_k,\, x_{k-m},\, x_{k-m-1})$. With a static
bound $M$ this is **not a recurrence at all** — per level a pure gather over
the window and a reduction, no carried state:

```python
# example 13
def box_flux(phi: Column, z: Column, v: Column, dt: float, m_max: int) -> Column:
    out = np.empty_like(phi)
    for k in range(phi.size):
        lo = max(k - m_max + 1, 1)  # M = max Courant number; clipped at the model top
        z_up = np.minimum(z[lo - 1 : k], z[k] + np.abs(v[lo : k + 1]) * dt)
        z_low = np.minimum(z[lo : k + 1], z_up)
        out[k] = -np.sum(phi[lo : k + 1] * (z_up - z_low))
    return out
```

Three facts from the COSMO case study: the formula **self-masks** (out-of-reach
levels contribute exactly zero, so no early exit is needed for correctness);
the rolling buffers production code uses are a *memory* optimization, not
semantics; and any cumulative quantity in the window
(geometric height from `dz`) is a separate [A1](#a1) accumulation composed in front.

References: the two-moment "boxtracking" sedimentation (eq. 12; tuple-valued
$\phi$: mass *and* number density) [cosmo/1]; the PPM reconstruction in
`ppm/common/cell_face_values.py` [pmap/dycore/9]; the asymptotic-mixing-length
walk — per level a *runtime*-bounded window sub-reduce, beyond D1's static
bound (worst case $O(K^2)$) [pace/shield/11].

### D2 {#d2} Bounded window inside a recurrence

Same window, but the contribution feeds or reads running state, so the scan is
genuine. A scan body encodes the window as $M$ extra carry slots (a ring
buffer) — an optimization the toolchain should introduce, not the user.

```python
# example 14
def windowed_scan(x: Column, m_max: int) -> Column:
    out = np.empty_like(x)
    out[0] = x[0]
    for k in range(1, x.size):
        window = x[max(k - m_max + 1, 0) : k + 1]  # the last M input levels
        out[k] = 0.5 * out[k - 1] + np.sum(window)  # sub-reduce feeding the state
    return out
```

References: none pinned — precipitation-flux schemes where the flux entering a
cell is carried *and* the hydrometeor state of the previous $M$ levels is
needed.

## E. Extra dimensions

### E1 {#e1} Recurrence under an extra data dimension

The recurrence is per `(column, g-point)`: a vertical scan applied pointwise
over an extra, non-spatial dimension of $O(100)$ entries. Tuple unrolling does
not scale at that size.

```python
# example 15
# 140 = RRTMG LW; in PMAP the count is read from the gas-optics file at runtime
GPointField = np.ndarray[tuple[Literal[140], int], np.dtype[np.float64]]  # (g-point, K)


def per_gpoint(a: GPointField, b: GPointField) -> GPointField:
    out = np.empty_like(b)
    out[:, 0] = b[:, 0]
    for k in range(1, b.shape[1]):
        out[:, k] = a[:, k] * out[:, k - 1] + b[:, k]  # one A1 per (column, g-point)
    return out
```

References: all flux kernels (`N_GPOINTS_LW/SW`) — the cartesian versions pin a
gt4py *fork* (`loops-over-data-dims`), and the broadband sums additionally
need a reduction over that dense dimension [pmap/radiation/1]. pace has no
in-DSL instance: its g-point workload is delegated to an external package, and
tracers ride a data dimension indexed one slot per call [pace/shield/13].

## X. Out of scope — documented boundaries

### X1 {#x1} $O(K^2)$ double loop with multi-level writes and feedback

```python
# example 16
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

No masked bounded window ([D1](#d1)) computes the same answer. Options:
reformulate in flux form (what COSMO did for exactly this physics — it changes
the operation order, so results differ and need validation), or keep it as
sequential column code outside the DSL.

References: the melting loop [noaa/shield/1]; its live form `sedi_melt` — a
stencil with dynamic-offset writes at the traversed levels, a host-side Python
fallback kept beside it [pace/shield/12]. Nothing in PMAP currently has this
shape.

### X2 {#x2} Data-dependent `while` over a lookup table

Not vertical at all, but it blocks the same ports:

```python
# example 17
def rh_lookup(rh: float, bounds: Column, table: Column) -> np.float64:
    # bounds: ascending upper edge of each humidity bin, last one +inf
    # table: one tabulated coefficient per bin (N_RH rows, compile-time constant)
    i = 0
    while rh > bounds[i]:  # (1) data-dependent search for the bin
        i += 1
    return np.float64(table[i])  # (2) what the index is for: a computed-index gather
```

The table holds aerosol optical properties per humidity bin (`rh_lower.A[irh]`
in the source).

Both halves are bounded by a compile-time constant, so the pair unrolls into one
masked select over the bins — $\sum_i \mathrm{where}(\text{first covering bin
is } i,\ \mathit{table}_i,\ 0)$ — which is branch-free and needs no indexing;
alternatively the index field is precomputed outside.

References: `f_calc_rh_index` [pmap/radiation/4].

## Compound examples {#compound}

**Tridiagonal solve** — the shape of the `_solve_itcs` forward sweep:

```python
# example 18
def compound(x: Column, y: Column, surface: float) -> Column:
    out = np.empty_like(x)
    n = x.size
    out[0] = surface * x[0]  # C1 peeled first row, seeded by data (C2)
    for k in range(1, n - 1):  # branch-free interior
        out[k] = 0.5 * out[k - 1] + x[k] * y[k + 1]  # A1 recurrence + lookahead input
    out[n - 1] = out[n - 2] + x[n - 1]  # C1 end-anchored row: no y[n] to read
    return out
```

The opposite-direction partner completing the pair; `f` materializes as a full
column — fusion across directions is not possible:

```python
# example 19
def back_substitution(e: Column, f: Column) -> Column:  # f: the forward sweep's output
    out = np.empty_like(f)
    out[-1] = f[-1]
    for k in range(f.size - 2, -1, -1):
        out[k] = e[k] * out[k + 1] + f[k]
    return out
```

References: [pmap/dycore/1]; the same pair in every tridiagonal solve — 7
across the PMAP dycore ([pmap/dycore/11], [pmap/dycore/12] counting its three
systems, [pmap/dycore/2], [pmap/dycore/13]) — in radiation's down/up flux
pairs [pmap/radiation/5], [pmap/radiation/6], and in `tropopause`
[pmap/microphysics/5].

**Reduction feeding per-level consumers** ([B1](#b1) → pointwise) — two
passes, no scan; production writes the broadcast as one:

```python
# example 20
def normalized(x: Column) -> Column:
    return x / np.sum(x)  # the column-constant broadcast back over K
```

References: `eta_levels` [pmap/microphysics/2], the `tropopause` broadcast
[pmap/microphysics/5], `_dev/cloudsc2`'s `trpaus`.

## Composition: expressible today, obligations on the toolchain

Nothing here is a new frontend pattern — every example only composes catalog
entries; what the evidence forces is the fusion obligation. Each example's
two forms are equal: the first is what should be *written*, the second the
one k-loop the toolchain should *produce*.

**Independent recurrences sharing per-level work** ([A1](#a1) × N):

```python
# example 21
def species(q: Column, rho: Column) -> Column:
    out = np.empty_like(q)
    out[0] = q[0]
    for k in range(1, q.size):
        out[k] = out[k - 1] * np.sqrt(rho[k]) + q[k]  # shared work, recomputed per species
    return out
```

```python
# example 21 (fused)
def species_fused(q: Stack, rho: Column) -> Stack:  # = np.stack([species(qi, rho) for qi in q])
    out = np.empty_like(q)
    out[:, 0] = q[:, 0]
    for k in range(1, q.shape[1]):
        shared = np.sqrt(rho[k])  # one k-loop, the shared work computed once
        out[:, k] = out[:, k - 1] * shared + q[:, k]
    return out
```

Production fuses by hand — the scalability case: a design is only usable if
the small forms compose without hand-fusion.

References: the muphys graupel scan [icon4py/muphys/1] — four species plus a
temperature/energy recurrence in one `scan_operator`, 21-component carry
(predecessor `_icon_graupel_scan` [icon4py/microphysics/1]: 28-tuple, in-carry
level counter, `sys.setrecursionlimit(350000)`, 13 of 28 outputs discarded);
cloudsc [pmap/microphysics/1] — the same at ~2000 lines, species loops
unrolled; the tmx energy integrals are the *written* form — four separate
accumulations in one field operator [icon4py/tmx/1].

**A recurrence and a reduction over its output** ([A1](#a1) → [B1](#b1), the
same-direction cascade):

```python
# example 22
def peak(a: Column, x: Column) -> tuple[Column, np.float64]:
    out = np.empty_like(x)
    out[0] = x[0]
    for k in range(1, x.size):
        out[k] = a[k] * out[k - 1] + x[k]  # A1
    return out, np.float64(np.max(out))  # B1 over its per-level output
```

```python
# example 22 (fused)
def peak_fused(a: Column, x: Column) -> tuple[Column, np.float64]:  # = peak, one pass
    out = np.empty_like(x)
    out[0] = x[0]
    total = out[0]  # the accumulator: the only state never emitted per level
    for k in range(1, x.size):
        out[k] = a[k] * out[k - 1] + x[k]
        total = max(total, out[k])
    return out, total
```

References: the cloudsc main loops — five per-level fall/sink fluxes plus six
column accumulators — cy49r1 [pmap/microphysics/1], cy50r1
[pmap/microphysics/3], dwarf [pmap/microphysics/4]; SW fluxes
[pmap/radiation/9], LW derivatives [pmap/radiation/10]; pointwise stages in
the cascade: sedimentation's `cumsum` of `dz` ([D1](#d1)), `_trapezoidal_z`'s
weights [pmap/dycore/5], radiation's per-level optics [pmap/radiation/1].

**Horizontal-offset producer feeding a scan** (pointwise with a horizontal
offset → [A1](#a1)) — no surveyed scan body reads a horizontal offset; only
this split form occurs:

```python
# example 23
def div_then_scan(flux: Stack, a: Column) -> Stack:
    g: Stack = flux[1:, :] - flux[:-1, :]  # horizontal-offset producer, materialized
    out = np.empty_like(g)
    out[:, 0] = g[:, 0]
    for k in range(1, g.shape[1]):
        out[:, k] = a[k] * out[:, k - 1] + g[:, k]  # A1 per column
    return out
```

```python
# example 23 (fused)
def div_then_scan_fused(flux: Stack, a: Column) -> Stack:  # = div_then_scan
    out: Stack = np.empty((flux.shape[0] - 1, flux.shape[1]))
    out[:, 0] = flux[1:, 0] - flux[:-1, 0]
    for k in range(1, flux.shape[1]):
        # the offset read moves into the k-loop: it gathers *inputs* of the
        # neighbouring columns, so the columns stay independent
        out[:, k] = a[k] * out[:, k - 1] + (flux[1:, k] - flux[:-1, k])
    return out
```

References: `_rhs_fluxdiv_xyz` [pmap/dycore/20] feeding `_solve_itcs`'s
`fluxdiv` argument [pmap/dycore/1]; every PMAP solve consumes such a
horizontally produced right-hand side.

## References {#references}

Each label links to the cited line at a pinned commit. New icon4py labels
(`common`/`tmx`/`metrics`/`thermodynamics`/`microphysics`) are pinned @
`f8032fd` (2026-10-20 main); pace labels are pinned to the submodule commits of
pace @ `43886fa` (pyFV3 @ `9df3c68`, pySHiELD @ `dc85a88`, NDSL @ `baf4e79`)
and cite one exemplar per family — the sweep found 161 true scans among 389
`FORWARD`/`BACKWARD` blocks (the rest are fake scans).

- [pmap/dycore/1] — `_solve_itcs` — implicit vertical diffusion, the Thomas pair (`diffusion/scalar.py`)
- [pmap/dycore/2] — stride-2 `_tridiagonal`, five interval regions (`helmholtz/preconditioner.py`)
- [pmap/dycore/3] — the `b33[0, 0, -1]` read in `_tridiagonal`
- [pmap/dycore/4] — `hydrostatic_exner` — running pressure integral
- [pmap/dycore/5] — `_trapezoidal_z` — column integral (`operators/integral.py`)
- [pmap/dycore/6] — `_cloud_cover_maximum(_random)` (`operators/cloud_cover.py`)
- [pmap/dycore/7] — `_cloud_base_height` and the other level searches
- [pmap/dycore/8] — `_boundary_layer_height`
- [pmap/dycore/9] — PPM `_cell_face_values_z_4th`
- [pmap/dycore/10] — the `precompilation` branch the scan ports target — its pinned HEAD still carries the pre-port TODOs; the validated ports are not pushed yet
- [pmap/dycore/11] — `_solve_itcs`, the terrain-following variant (`diffusion/scalar_z.py`)
- [pmap/dycore/12] — `_solve_itcs`, three coupled systems (`diffusion/vector.py`)
- [pmap/dycore/13] — the poisson stride-2 `_tridiagonal`
- [pmap/dycore/14] — helmholtz `_tridiagonal_init`, stride-2 LU factorisation
- [pmap/dycore/15] — poisson `_tridiagonal_init`
- [pmap/dycore/16] — `_cloud_cover_maximum_random`
- [pmap/dycore/17] — `_cloud_top_height`
- [pmap/dycore/18] — `_cloud_top_temperature`
- [pmap/dycore/19] — `wind_bl_height`
- [pmap/dycore/20] — `_rhs_fluxdiv_xyz` — horizontal flux divergence, PARALLEL (`diffusion/scalar.py`)
- [pmap/microphysics/1] — the cloudsc cy49r1 main loop (cy50r1/dwarf alike)
- [pmap/microphysics/2] — `eta_levels`
- [pmap/microphysics/3] — the cloudsc cy50r1 main loop
- [pmap/microphysics/4] — the cloudsc dwarf main loop
- [pmap/microphysics/5] — `tropopause`
- [pmap/radiation/1] — LW adding, one of the nine affine flux kernels (branch `real_cases_radiation`)
- [pmap/radiation/2] — above-cloud flux, started at `i_cloud_top`
- [pmap/radiation/3] — the `i_cloud_top` max-reduction
- [pmap/radiation/4] — `f_calc_rh_index`
- [pmap/radiation/5] — LW flux down
- [pmap/radiation/6] — LW no-scatter flux up
- [pmap/radiation/7] — LW total albedo
- [pmap/radiation/8] — SW total albedo
- [pmap/radiation/9] — SW fluxes, 2D carries alongside per-level fluxes
- [pmap/radiation/10] — LW derivatives, 2D carry
- [pmap/radiation/11] — cloudy flux down, started at `i_cloud_top`
- [pmap/radiation/12] — the `v_mat_prod` cumulative product
- [icon4py/muphys/1] — the muphys graupel `scan_operator` `_precip_and_t`, 21-component carry (`graupel.py` @ `515763c`)
- [icon4py/muphys/2] — the per-species `activated` latch
- [icon4py/muphys/3] — the `rho` delay-line carry slot
- [icon4py/muphys/4] — the external `t_kp1` pre-shift with a clamp
- [icon4py/muphys/5] — the runtime `last_lev` argument
- [icon4py/dycore/1] — gt4py's `test_icon_like_scan.py`, modeled on icon4py's `solve_tridiagonal_matrix_for_w_*`
- [icon4py/dycore/2] — the w-solver Thomas pair: `solve_tridiagonal_matrix_for_w_forward_sweep.py` (back-substitution alongside)
- [icon4py/diagnostics/1] — `diagnose_pressure.py` @ `515763c` (removed from `main` since)
- [icon4py/common/1] — the shared Thomas solver, three sweep/substitution pairs (`math/tridiagonal.py`)
- [icon4py/common/2] — `_accumulate_from_top` (`math/vertical_operations.py`)
- [icon4py/tmx/1] — `_compute_end_of_step_diagnostics` — four energy integrals in one field operator
- [icon4py/metrics/1] — `_compute_param` — level-bracketing search; the production path is a host-side loop (`metrics/metric_fields.py`)
- [icon4py/thermodynamics/1] — `_scan_pressure` — `diagnose_pressure`'s live successor (`physics/thermodynamics/compute_pressure.py`)
- [icon4py/microphysics/1] — `_icon_graupel_scan` — 28-tuple carry, `sys.setrecursionlimit(350000)` (`microphysics/stencils/graupel_stencils.py`)
- [pace/fv3/1] — `sim1_solver` — semi-implicit sound-wave solver: Thomas pair + w recurrence, boundary rows, `ws` seed
- [pace/fv3/2] — `set_initial_vals` — cubic-spline edge-value tridiagonal, `qs` seed (`remap_profile.py`)
- [pace/fv3/3] — `cubic_spline_interpolation_from_layer_center_to_interfaces` — coefficients from a host-side K loop (`updatedzd.py`)
- [pace/fv3/4] — `precompute`/`finalize` — hydrostatic pressure/height prefix sums, `hs` seed (`riem_solver_c.py`)
- [pace/fv3/5] — `edge_pe_update` — running pressure in a 2D carry, per-level product emitted (`pk3_halo.py`)
- [pace/fv3/6] — `update_dz_c` — `gz = max(gz[+1] + DZ_MIN, gz)` running clamp (`updatedzc.py`)
- [pace/fv3/7] — `fix_tracer` — negative-tracer borrow cascade + mass-sum rescale (`fillz.py`)
- [pace/fv3/8] — `m_loop` — Richardson dry adjustment, ~15-field carry, five top rows (`fv_subgridz.py`)
- [pace/fv3/9] — `ray_fast_wind_compute` — masked column sums, broadcast back by copy scans (`ray_fast.py`)
- [pace/ndsl/1] — `tridiag_solve` / `masked_tridiag_solve` (`ndsl/stencils/tridiag.py`)
- [pace/ndsl/2] — `column_max`/`column_min` — library column reductions via `while` + absolute-level reads
- [pace/shield/1] — `warm_rain` — time-implicit sedimentation recurrences + flux prefix sums, ×4 species (`gfs_microphysics.py`)
- [pace/shield/2] — `stencil_static7` — updraft parcel recurrence (`samfshalconv.py`)
- [pace/shield/3] — `mfscu_s3` — downdraft recurrence seeded at the runtime `krad` level (`pbl/mfscu.py`)
- [pace/shield/4] — `tke_tridiag_matrix_ele_comp` — next level's row staged in 2D carries (`pbl/satmedmfvdiff.py`)
- [pace/shield/5] — the `stop_k` melt latch (`no_fall` ×4 alike) (`gfs_microphysics.py`)
- [pace/shield/6] — `stencil_static0` — argmax `kb`/`hmax` (`samfshalconv.py`)
- [pace/shield/7] — `stratocumulus` — argmin `krad`, cloud-deck searches `lcld`/`kcld` (`pbl/satmedmfvdiff.py`)
- [pace/shield/8] — `mrf_pbl_scheme_part1` — critical-Richardson bracketing + interpolation
- [pace/shield/9] — `stencil_static1` — LFC search `kbcon`, not-found fallback (`samfshalconv.py`)
- [pace/shield/10] — `stencil_static10` — CIN integral, one of seven two-index integrals (`samfshalconv.py`)
- [pace/shield/11] — `compute_asymptotic_mixing_length` — runtime-bounded window walk (`pbl/satmedmfvdiff.py`)
- [pace/shield/12] — `sedi_melt` — the melting loop as a stencil, dynamic-offset writes; Python fallback `sedi_melt_python` alongside (`gfdl_cld_microphysics/sedimentation.py`)
- [pace/shield/13] — `tridit`/`tridi2`/`tridin` — tridiagonal with a tracer-data-dimension RHS (`pbl/tridiag.py`)
- [phyex/1] — the tropopause `ITPL`/`ZTMIN` argmin (`condensation.F90`, [physics_patterns PR #1](https://github.com/GridTools/physics_patterns/pull/1))
- [phyex/2] — the mixing-length recurrence gated on the tropopause height
- [noaa/lsm/1] — the Noah LSM `nroot` loop ([physics_patterns PR #2](https://github.com/GridTools/physics_patterns/pull/2))
- [noaa/shield/1] — the SHiELD melting double loop (physics_patterns PR #2)
- [cosmo/1] — U. Blahak, *New implementation of explicit hydrometeor sedimentation* (DWD, 2020), eq. 12; production Fortran `mo_2mom_mcrph_processes.f90` in the [ICON model](https://gitlab.dkrz.de/icon/icon-model)

## Next steps

1. **Validate the catalog against the design axes** — for each entry: can it
   be written without any workaround from the
   [[personal/tehrengruber/scan-design/scan-design|design document]]'s table,
   and how wide does its carry get?
2. **Promote the minimal forms to an acceptance suite** — they are runnable
   NumPy; pairing each with its desired DSL spelling (and with the equality
   already noted for [B4](#b4)) gives a test suite that is independent of the
   eventual API.
3. **Decide the scope line** — [X1](#x1) and [X2](#x2) are the claimed
   boundary; confirm that nothing else in PMAP, icon4py or `physics_patterns`
   crosses it.
4. **Then** compare designs:
   [[personal/tehrengruber/scan-design/scan-design|the design document]]
   should be the API proposal measured against the entries, cross-checked with
   [[personal/havogt/scan-redesign|havogt's scan redesign]] (whose constructs
   already target A1–A2, B1–B4, C1–C3 and D1–D2 explicitly — the overlap
   is deliberate, and conflicts should be called out per entry rather than per
   document).

[pmap/dycore/1]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/diffusion/scalar.py#L421
[pmap/dycore/2]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/helmholtz/preconditioner.py#L53
[pmap/dycore/3]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/helmholtz/preconditioner.py#L65
[pmap/dycore/4]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/operators/hydrostatic_exner.py#L48
[pmap/dycore/5]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/operators/integral.py#L105
[pmap/dycore/6]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/operators/cloud_cover.py#L195
[pmap/dycore/7]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/operators/cloud_cover.py#L248
[pmap/dycore/8]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/diagnostics/boundary_layer_height.py#L87
[pmap/dycore/9]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/ppm/common/cell_face_values.py#L319
[pmap/dycore/10]: https://github.com/PMAP-Project/PMAP/tree/fbdad0b0fa4cc9fccd1cc180ea2460e4935f7d63
[pmap/microphysics/1]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/physics/cloud_microphysics/cloudsc/cy49r1/_stencils/cloudsc_cy49r1.py#L380
[pmap/microphysics/2]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/physics/cloud_microphysics/diagnostics.py#L18
[pmap/radiation/1]: https://github.com/PMAP-Project/PMAP/blob/aa9eeea8fc564e659e5dd6d2f1b31b3f3467bef5/src/pmap/physics/radiation/ecrad_cy49r1/_stencils/calc_radiative_fluxes_lw.py#L395
[pmap/radiation/2]: https://github.com/PMAP-Project/PMAP/blob/aa9eeea8fc564e659e5dd6d2f1b31b3f3467bef5/src/pmap/physics/radiation/ecrad_cy49r1/_stencils/calc_radiative_fluxes_lw.py#L664
[pmap/radiation/3]: https://github.com/PMAP-Project/PMAP/blob/aa9eeea8fc564e659e5dd6d2f1b31b3f3467bef5/src/pmap/physics/radiation/ecrad_cy49r1/_stencils/calc_optical_properties.py#L212
[pmap/radiation/4]: https://github.com/PMAP-Project/PMAP/blob/aa9eeea8fc564e659e5dd6d2f1b31b3f3467bef5/src/pmap/physics/radiation/ecrad_cy49r1/_stencils/functions_aerosols.py#L11
[icon4py/muphys/1]: https://github.com/C2SM/icon4py/blob/515763c3e42776c35a7751ce7173c412392dbec4/model/atmosphere/subgrid_scale_physics/muphys/src/icon4py/model/atmosphere/subgrid_scale_physics/muphys/implementations/graupel.py#L175
[icon4py/muphys/2]: https://github.com/C2SM/icon4py/blob/515763c3e42776c35a7751ce7173c412392dbec4/model/atmosphere/subgrid_scale_physics/muphys/src/icon4py/model/atmosphere/subgrid_scale_physics/muphys/implementations/graupel.py#L64
[icon4py/muphys/3]: https://github.com/C2SM/icon4py/blob/515763c3e42776c35a7751ce7173c412392dbec4/model/atmosphere/subgrid_scale_physics/muphys/src/icon4py/model/atmosphere/subgrid_scale_physics/muphys/implementations/graupel.py#L79
[icon4py/muphys/4]: https://github.com/C2SM/icon4py/blob/515763c3e42776c35a7751ce7173c412392dbec4/model/atmosphere/subgrid_scale_physics/muphys/src/icon4py/model/atmosphere/subgrid_scale_physics/muphys/implementations/graupel.py#L511
[icon4py/muphys/5]: https://github.com/C2SM/icon4py/blob/515763c3e42776c35a7751ce7173c412392dbec4/model/atmosphere/subgrid_scale_physics/muphys/src/icon4py/model/atmosphere/subgrid_scale_physics/muphys/implementations/graupel.py#L488
[icon4py/dycore/1]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/multi_feature_tests/ffront_tests/test_icon_like_scan.py#L43
[icon4py/diagnostics/1]: https://github.com/C2SM/icon4py/blob/515763c3e42776c35a7751ce7173c412392dbec4/model/common/src/icon4py/model/common/diagnostic_calculations/stencils/diagnose_pressure.py#L15
[phyex/1]: https://github.com/GridTools/physics_patterns/blob/d54724e0aac8c423aa8818688f381dd13e11e1bb/meteo-france/tropopause_inversion_and_mixing_length/condensation.F90#L89
[phyex/2]: https://github.com/GridTools/physics_patterns/blob/d54724e0aac8c423aa8818688f381dd13e11e1bb/meteo-france/tropopause_inversion_and_mixing_length/condensation.F90#L110
[noaa/lsm/1]: https://github.com/GridTools/physics_patterns/blob/2207ae909c0387b5208bb4eb573850f029788a95/NOAA_physics/lsm.py#L22
[noaa/shield/1]: https://github.com/GridTools/physics_patterns/blob/2207ae909c0387b5208bb4eb573850f029788a95/NOAA_physics/microphysics.py#L59
[cosmo/1]: http://www.cosmo-model.org/content/model/documentation/core/docu_sedi_twomom.pdf
[pmap/dycore/11]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/diffusion/scalar_z.py#L178
[pmap/dycore/12]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/diffusion/vector.py#L767
[pmap/dycore/13]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/poisson/preconditioner.py#L51
[pmap/dycore/14]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/helmholtz/preconditioner_coefficients.py#L174
[pmap/dycore/15]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/poisson/preconditioner_coefficients.py#L101
[pmap/dycore/16]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/operators/cloud_cover.py#L217
[pmap/dycore/17]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/operators/cloud_cover.py#L279
[pmap/dycore/18]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/operators/cloud_cover.py#L309
[pmap/dycore/19]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/boundary_conditions/cell_perturbation_method.py#L336
[pmap/dycore/20]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/diffusion/scalar.py#L403
[pmap/microphysics/3]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/physics/cloud_microphysics/cloudsc/cy50r1/_stencils/cloudsc_cy50r1.py#L384
[pmap/microphysics/4]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/physics/cloud_microphysics/cloudsc/dwarf/_stencils/cloudsc_dwarf.py#L326
[pmap/microphysics/5]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/physics/cloud_microphysics/diagnostics.py#L36
[pmap/radiation/5]: https://github.com/PMAP-Project/PMAP/blob/aa9eeea8fc564e659e5dd6d2f1b31b3f3467bef5/src/pmap/physics/radiation/ecrad_cy49r1/_stencils/calc_radiative_fluxes_lw.py#L419
[pmap/radiation/6]: https://github.com/PMAP-Project/PMAP/blob/aa9eeea8fc564e659e5dd6d2f1b31b3f3467bef5/src/pmap/physics/radiation/ecrad_cy49r1/_stencils/calc_radiative_fluxes_lw.py#L449
[pmap/radiation/7]: https://github.com/PMAP-Project/PMAP/blob/aa9eeea8fc564e659e5dd6d2f1b31b3f3467bef5/src/pmap/physics/radiation/ecrad_cy49r1/_stencils/calc_radiative_fluxes_lw.py#L595
[pmap/radiation/8]: https://github.com/PMAP-Project/PMAP/blob/aa9eeea8fc564e659e5dd6d2f1b31b3f3467bef5/src/pmap/physics/radiation/ecrad_cy49r1/_stencils/calc_radiative_fluxes_sw.py#L474
[pmap/radiation/9]: https://github.com/PMAP-Project/PMAP/blob/aa9eeea8fc564e659e5dd6d2f1b31b3f3467bef5/src/pmap/physics/radiation/ecrad_cy49r1/_stencils/calc_radiative_fluxes_sw.py#L527
[pmap/radiation/10]: https://github.com/PMAP-Project/PMAP/blob/aa9eeea8fc564e659e5dd6d2f1b31b3f3467bef5/src/pmap/physics/radiation/ecrad_cy49r1/_stencils/calc_radiative_fluxes_lw.py#L791
[pmap/radiation/11]: https://github.com/PMAP-Project/PMAP/blob/aa9eeea8fc564e659e5dd6d2f1b31b3f3467bef5/src/pmap/physics/radiation/ecrad_cy49r1/_stencils/calc_radiative_fluxes_lw.py#L718
[pmap/radiation/12]: https://github.com/PMAP-Project/PMAP/blob/aa9eeea8fc564e659e5dd6d2f1b31b3f3467bef5/src/pmap/physics/radiation/ecrad_cy49r1/_stencils/calc_optical_properties.py#L162
[icon4py/dycore/2]: https://github.com/C2SM/icon4py/blob/fa2de0fe0fe69e1f01ac4667c56fa3c433c0ed0b/model/atmosphere/dycore/src/icon4py/model/atmosphere/dycore/stencils/solve_tridiagonal_matrix_for_w_forward_sweep.py#L19
[icon4py/common/1]: https://github.com/C2SM/icon4py/blob/f8032fd51b104a4fc6322c31cb5a2ad55316203a/model/common/src/icon4py/model/common/math/tridiagonal.py#L23
[icon4py/common/2]: https://github.com/C2SM/icon4py/blob/f8032fd51b104a4fc6322c31cb5a2ad55316203a/model/common/src/icon4py/model/common/math/vertical_operations.py#L417
[icon4py/tmx/1]: https://github.com/C2SM/icon4py/blob/f8032fd51b104a4fc6322c31cb5a2ad55316203a/model/atmosphere/subgrid_scale_physics/tmx/src/icon4py/model/atmosphere/subgrid_scale_physics/tmx/stencils/energy_update_stencils.py#L71
[icon4py/metrics/1]: https://github.com/C2SM/icon4py/blob/f8032fd51b104a4fc6322c31cb5a2ad55316203a/model/common/src/icon4py/model/common/metrics/metric_fields.py#L898
[icon4py/thermodynamics/1]: https://github.com/C2SM/icon4py/blob/f8032fd51b104a4fc6322c31cb5a2ad55316203a/model/common/src/icon4py/model/common/physics/thermodynamics/compute_pressure.py#L60
[icon4py/microphysics/1]: https://github.com/C2SM/icon4py/blob/f8032fd51b104a4fc6322c31cb5a2ad55316203a/model/atmosphere/subgrid_scale_physics/microphysics/src/icon4py/model/atmosphere/subgrid_scale_physics/microphysics/stencils/graupel_stencils.py#L40
[pace/fv3/1]: https://github.com/NOAA-GFDL/pyFV3/blob/9df3c68fc15d5361903fa7a0ea007b63e7d0ed9f/pyfv3/stencils/sim1_solver.py#L11
[pace/fv3/2]: https://github.com/NOAA-GFDL/pyFV3/blob/9df3c68fc15d5361903fa7a0ea007b63e7d0ed9f/pyfv3/stencils/remap_profile.py#L139
[pace/fv3/3]: https://github.com/NOAA-GFDL/pyFV3/blob/9df3c68fc15d5361903fa7a0ea007b63e7d0ed9f/pyfv3/stencils/updatedzd.py#L160
[pace/fv3/4]: https://github.com/NOAA-GFDL/pyFV3/blob/9df3c68fc15d5361903fa7a0ea007b63e7d0ed9f/pyfv3/stencils/riem_solver_c.py#L12
[pace/fv3/5]: https://github.com/NOAA-GFDL/pyFV3/blob/9df3c68fc15d5361903fa7a0ea007b63e7d0ed9f/pyfv3/stencils/pk3_halo.py#L9
[pace/fv3/6]: https://github.com/NOAA-GFDL/pyFV3/blob/9df3c68fc15d5361903fa7a0ea007b63e7d0ed9f/pyfv3/stencils/updatedzc.py#L59
[pace/fv3/7]: https://github.com/NOAA-GFDL/pyFV3/blob/9df3c68fc15d5361903fa7a0ea007b63e7d0ed9f/pyfv3/stencils/fillz.py#L11
[pace/fv3/8]: https://github.com/NOAA-GFDL/pyFV3/blob/9df3c68fc15d5361903fa7a0ea007b63e7d0ed9f/pyfv3/stencils/fv_subgridz.py#L223
[pace/fv3/9]: https://github.com/NOAA-GFDL/pyFV3/blob/9df3c68fc15d5361903fa7a0ea007b63e7d0ed9f/pyfv3/stencils/ray_fast.py#L36
[pace/ndsl/1]: https://github.com/NOAA-GFDL/NDSL/blob/baf4e79565501b937d774a6b8dc1494302fd1c51/ndsl/stencils/tridiag.py#L6
[pace/ndsl/2]: https://github.com/NOAA-GFDL/NDSL/blob/baf4e79565501b937d774a6b8dc1494302fd1c51/ndsl/stencils/column_operations.py#L8
[pace/shield/1]: https://github.com/NOAA-GFDL/pySHiELD/blob/dc85a881d7f72ccfb71dd88c040ba304bb97207c/pyshield/stencils/gfs_microphysics.py#L270
[pace/shield/2]: https://github.com/NOAA-GFDL/pySHiELD/blob/dc85a881d7f72ccfb71dd88c040ba304bb97207c/pyshield/stencils/shallow_convection/samfshalconv.py#L676
[pace/shield/3]: https://github.com/NOAA-GFDL/pySHiELD/blob/dc85a881d7f72ccfb71dd88c040ba304bb97207c/pyshield/stencils/pbl/mfscu.py#L142
[pace/shield/4]: https://github.com/NOAA-GFDL/pySHiELD/blob/dc85a881d7f72ccfb71dd88c040ba304bb97207c/pyshield/stencils/pbl/satmedmfvdiff.py#L1084
[pace/shield/5]: https://github.com/NOAA-GFDL/pySHiELD/blob/dc85a881d7f72ccfb71dd88c040ba304bb97207c/pyshield/stencils/gfs_microphysics.py#L694
[pace/shield/6]: https://github.com/NOAA-GFDL/pySHiELD/blob/dc85a881d7f72ccfb71dd88c040ba304bb97207c/pyshield/stencils/shallow_convection/samfshalconv.py#L301
[pace/shield/7]: https://github.com/NOAA-GFDL/pySHiELD/blob/dc85a881d7f72ccfb71dd88c040ba304bb97207c/pyshield/stencils/pbl/satmedmfvdiff.py#L555
[pace/shield/8]: https://github.com/NOAA-GFDL/pySHiELD/blob/dc85a881d7f72ccfb71dd88c040ba304bb97207c/pyshield/stencils/pbl/satmedmfvdiff.py#L356
[pace/shield/9]: https://github.com/NOAA-GFDL/pySHiELD/blob/dc85a881d7f72ccfb71dd88c040ba304bb97207c/pyshield/stencils/shallow_convection/samfshalconv.py#L415
[pace/shield/10]: https://github.com/NOAA-GFDL/pySHiELD/blob/dc85a881d7f72ccfb71dd88c040ba304bb97207c/pyshield/stencils/shallow_convection/samfshalconv.py#L820
[pace/shield/11]: https://github.com/NOAA-GFDL/pySHiELD/blob/dc85a881d7f72ccfb71dd88c040ba304bb97207c/pyshield/stencils/pbl/satmedmfvdiff.py#L719
[pace/shield/12]: https://github.com/NOAA-GFDL/pySHiELD/blob/dc85a881d7f72ccfb71dd88c040ba304bb97207c/pyshield/stencils/gfdl_cld_microphysics/sedimentation.py#L290
[pace/shield/13]: https://github.com/NOAA-GFDL/pySHiELD/blob/dc85a881d7f72ccfb71dd88c040ba304bb97207c/pyshield/stencils/pbl/tridiag.py#L6
