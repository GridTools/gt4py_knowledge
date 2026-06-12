---
title: "Vertical scan redesign: worked examples"
author: havogt
tags: [scan, vertical, reduction, examples]
created: 2026-06-12
---

- **Status**: companion document to
  [[ideas/havogt/scan-redesign|the scan redesign proposal]] and its
  [[ideas/havogt/scan-redesign_research|research notes]]

Before/after code for the main scan use cases under the proposed design.
All "proposed" code uses the strawman syntax of the proposal (§3.1–§3.4,
§3.8) and is **illustrative, not validated**: names, the `k_range=` spelling,
and `KView` indexing are open questions (§5 of the proposal). The physics in
the muphys sketch is transcribed for structure, not correctness.

## 1. Cumulative sum, and the final carry as a surface diagnostic

Today:

```python
@gtx.scan_operator(axis=KDim, forward=True, init=0.0)
def cumsum(carry: float, val: float) -> float:
    return carry + val
```

Proposed (simple form — body returns only the carry, call returns only `ys`):

```python
@gtx.scan(axis=KDim, forward=True)
def cumsum(carry: float, val: float) -> float:
    return carry + val


integral = cumsum(x, init=0.0)
```

General form, when the column total is itself a result (e.g. accumulated
surface flux) — impossible today without reading the last level of a full 3D
field:

```python
@gtx.scan(axis=KDim, forward=True)
def cumsum_step(carry: float, val: float) -> tuple[float, float]:
    s = carry + val
    return s, s


total, integral = cumsum_step(x, init=0.0)
# total:    Field[[Cell], float]      — the final carry
# integral: Field[[Cell, KDim], float]
```

## 2. ICON-like tridiagonal forward sweep (boundary condition at level 0)

Today (`tests/next_tests/integration_tests/multi_feature_tests/ffront_tests/test_icon_like_scan.py`):

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


# program level: dummy bool output and output slicing
_solve_nonhydro_stencil_52_like(..., out=(z_q[:, 1:], w[:, 1:], dummy[:, 1:]))
```

Proposed (mechanism §3.3.1 — peelable index conditional; no flag, no dummy
output, no slicing; the toolchain peels `k == 0` out of the loop):

```python
class State(NamedTuple):
    z_q: float
    w: float


@gtx.scan(axis=KDim, forward=True)
def w_fwd_sweep(carry: State, z_q: float, w: float, z_a: float, z_b: float, z_c: float) -> State:
    z_g = z_b + z_a * carry.z_q
    step = State(z_q=(0.0 - z_c) * z_g, w=z_a * carry.w * z_g)
    return concat_where(KDim == 0, State(z_q=z_q, w=w), step)


z_q_res, w_res = w_fwd_sweep(z_q, w, z_a, z_b, z_c, init=State(z_q=0.0, w=0.0))
```

(`init` is still required to fix the carry structure; it is never *used*
because the peeled first iteration overwrites it — backends drop it.)

## 3. Backward scan with a first-level flag → `include_initial`

Today (icon4py `diagnose_pressure.py`, paraphrased): a backward scan with
carry `(pressure, pressure_interface, is_first_level: bool)` and
`init=(0.0, 0.0, True)`; the body selects `surface_pressure` instead of the
carried interface pressure when the flag is set, then carries `False`
forever; the `pressure` slot is never read back (output channel only).

Proposed (mechanism §3.3.2 — the boundary value comes from *outside* the
scanned fields, so it is the `init`; the flag and the dead slot disappear):

```python
@gtx.scan(axis=KDim, forward=False)
def pressure_step(p_interface: float, dz_rho_g: float) -> tuple[float, tuple[float, float]]:
    p_interface_above = p_interface * exp(...)  # hydrostatic increment
    p_full = sqrt(p_interface * p_interface_above)  # e.g. interface -> full level
    return p_interface_above, (p_full, p_interface_above)


_, (pressure, pressure_ifc) = pressure_step(dz_rho_g, init=surface_pressure)
```

With `include_initial=True` the surface value itself appears in the output —
n cell inputs, n+1 interface outputs, "the value at the boundary *is* the
init" (the exclusive/inclusive duality, research §B.2):

```python
_, p_ifc = pressure_step(dz_rho_g, init=surface_pressure, include_initial=True)
```

## 4. Thomas algorithm: don't write it at all

```python
w = gtx.lib.tridiagonal_solve(a, b, c, d)  # proposal §3.5
```

Today this is two hand-written scans (icon4py
`solve_tridiagonal_matrix_for_w_forward_sweep.py` /
`..._back_substitution.py`).

## 5. Sedimentation (scan + sub-reduce, regime 1): vertical window

The COSMO "boxtracking" flux (research §C.1, eq. 12), today inexpressible
without hand-rolled ring buffers in a carry tuple. Proposed (§3.4), with the
geometric height composed in front as a `cumsum` when only `dz` is given:

```python
KWin = gtx.WindowOffset("KWin", source=KDim, offsets=range(0, -4, -1))  # k .. k-3
KWinUp = gtx.WindowOffset(
    "KWinUp", source=KDim, offsets=range(-1, -5, -1), local_dim=KWin.local_dim
)  # k-1 .. k-4


@gtx.field_operator
def box_sedim_flux(v: CellKF, dz: CellKF, phi: CellKF, dt: float) -> CellKF:
    z = gtx.lib.cumsum(dz, axis=KDim, forward=False)  # height of lower cell face
    z_up = minimum(z(KWinUp), z + abs(v(KWin)) * dt)
    z_low = minimum(z(KWin), z_up)
    return -neighbor_sum(phi(KWin) * (z_up - z_low), axis=KWin.local_dim)
```

Window entries reaching above the model top are skip-masked (contribute the
neutral element 0); contributions from levels out of physical reach are
already zero by self-masking of the formula. The window bound (here 4) is
the static maximum vertical Courant number. The reference implementations
this transcribes are in the appendix.

## 6. muphys graupel: decomposing the fused scan

Today (`C2SM/icon4py`,
`model/atmosphere/subgrid_scale_physics/muphys/implementations/graupel.py`
at commit `515763c`; analysis in research §A.7) — one scan, five
recurrences, 21 carry components:

```python
class PrecipStateQx(NamedTuple):
    x: ta.wpfloat
    p: ta.wpfloat
    vc: ta.wpfloat
    activated: bool


class TempState(NamedTuple):
    t: ta.wpfloat
    eflx: ta.wpfloat
    activated: bool


class IntegrationState(NamedTuple):
    r: PrecipStateQx
    s: PrecipStateQx
    i: PrecipStateQx
    g: PrecipStateQx
    t_state: TempState
    rho: ta.wpfloat  # delay line: rho at k-1
    pflx_tot: ta.wpfloat  # output channel, never read back


@gtx.scan_operator(axis=dims.KDim, forward=True, init=IntegrationState(...))
def _precip_and_t(
    previous_level: IntegrationState,
    t: ta.wpfloat,
    t_kp1: ta.wpfloat,  # lookahead, precomputed outside:
    #   concat_where(KDim < last_lev, t(Koff[1]), t)
    rho: ta.wpfloat,
    q: Q_scalar,  # 6 species components
    mask_r: bool,
    mask_s: bool,
    mask_i: bool,
    mask_g: bool,
    dt: ta.wpfloat,
    dz: ta.wpfloat,
) -> IntegrationState: ...
```

with the per-species recurrence factored into a helper called four times
(excerpt; note `previous_level_rho` — the delay line in use — and the
work-skipping branch):

```python
@gtx.field_operator
def precip_qx_level_update(previous_level_q: PrecipStateQx, previous_level_rho: ta.wpfloat,
                           prefactor: ta.wpfloat, ...) -> PrecipStateQx:
    current_level_activated = previous_level_q.activated | mask
    ...
    rhox_prev = (previous_level_q.x + q) * wpfloat(0.5) * previous_level_rho
    if current_level_activated:
        vt = (previous_level_q.vc * prefactor * power((rhox_prev + offset), exponent)
              if previous_level_q.activated else wpfloat(0.0))
        x = (zeta * (flx_eff - flx_partial)) / ((wpfloat(1.0) + zeta * vt) * rho)
        p = (x * rho * vt + flx_partial) * wpfloat(0.5)
    else:
        x = q
        p = wpfloat(0.0)
    return PrecipStateQx(x=x, p=p, vc=vc, activated=current_level_activated)
```

Surface outputs `pr, ps, pi, pg, pre` are extracted in the `@gtx.program` by
slicing the last level (`dims.KDim: (vertical_end - 1, vertical_end)`).

Proposed decomposition — one scan *per species* plus one for temperature;
under §3.8 the toolchain must fuse these into a single k-loop:

```python
class PrecipState(NamedTuple):
    x: float
    p: float
    vc: float
    activated: bool


@gtx.scan(axis=KDim, forward=True)
def species_step(
    carry: PrecipState,
    q: float,
    rho: gtx.KView[float],         # rho[-1] replaces the delay-line carry slot
    vc: float,                     # per-level scale factor, computed outside
    zeta: float,
    mask: bool,
    prefactor: float, exponent: float, offset: float,
) -> tuple[PrecipState, tuple[float, float]]:
    activated = carry.activated | mask
    rho_x = q * rho[0]
    flx_eff = rho_x / zeta + 2.0 * carry.p
    flx_partial = minimum(rho_x * vc * prefactor * (rho_x + offset) ** exponent, flx_eff)
    rhox_prev = (carry.x + q) * 0.5 * rho[-1]
    vt = where(carry.activated, carry.vc * prefactor * (rhox_prev + offset) ** exponent, 0.0)
    x = where(activated, zeta * (flx_eff - flx_partial) / ((1.0 + zeta * vt) * rho[0]), q)
    p = where(activated, (x * rho[0] * vt + flx_partial) * 0.5, 0.0)
    return PrecipState(x=x, p=p, vc=vc, activated=activated), (x, p)


@gtx.scan(axis=KDim, forward=True)
def temperature_step(
    eflx: float,
    t: gtx.KView[float],           # t[+1] replaces the precomputed t_kp1 input
    pr: float, pflx_tot: float,    # same-level outputs of the species scans
    q: Q, qliq: float, qice: float,
    rho: float, dz: float, dt: float,
) -> tuple[float, float]: ...


@gtx.field_operator
def graupel(...):
    ...
    r_final, (qr, pr) = species_step(q.r, rho, vc_r, zeta, mask_r, *RAIN_PARAMS, init=...)
    s_final, (qs, ps) = species_step(q.s, rho, vc_s, zeta, mask_s, *SNOW_PARAMS, init=...)
    i_final, (qi, pi) = species_step(q.i, rho, vc_i, zeta, mask_i, *ICE_PARAMS, init=...)
    g_final, (qg, pg) = species_step(q.g, rho, vc_g, zeta, mask_g, *GRAUPEL_PARAMS, init=...)
    eflx_final, t_out = temperature_step(
        t, pr, ps + pi + pg, q, qr + q.c, qs + qi + qg, rho, dz, dt, init=0.0
    )
    # surface diagnostics are final carries — no last-level program slicing:
    return t_out, (qr, qs, qi, qg), (r_final.p, s_final.p, i_final.p, g_final.p, eflx_final)
```

What disappeared: the 21-component shared carry (each scan carries only its
own 4 floats / 1 float), the `rho` delay line (`rho[-1]`), the external
`t_kp1` precomputation (`t[+1]`), the pass-through `t`/`pflx_tot` slots
(carry/output separation), the `*_scalar` helper twins (slice bodies call
field operators), the program-level surface slicing (final carries), and
the `if current_level_activated:` identity else-branch with its ~650 lines
of DaCe repair hooks (`where`-masking; "else is identity" becomes a
transformation concern). What it costs: correctness now depends on the §3.8
fusion obligation — five scans sharing `rho`, `zeta`, masks must compile to
one vertical loop, which is the acceptance test of the rewrite.

## 7. PMAP stride-2 tridiagonal solve (`gt4py.cartesian` migration)

`PMAP-Project/PMAP`, `src/pmap/helmholtz/preconditioner.py` (research §A.8):
the vertical line solve of a line-relaxation Helmholtz preconditioner,
applied `nitr` times per call inside a GCR-style iteration. Deliberately not
textbook Thomas: the coefficients are pre-factored outside the solver (so
§3.5's `tridiagonal_solve` does not apply), the recurrences are stride-2
(two interleaved even/odd elimination chains), and there are two-level
boundary regions at both column ends.

Today (GTScript):

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

Proposed (anchored conditions and partition peeling §3.3.1, `history`
carries §4.6, K-local views §3.1; transcribed for structure, not validated):

```python
@gtx.scan(axis=KDim, forward=True, history=2)
def fwd_elim(
    ff: gtx.KView[float], rs: float, dnmi: float, denratm: float, b33: gtx.KView[float]
) -> float:
    # ff is the carry-history view (§4.6): ff[-2] = own output two levels up
    return concat_where(
        KDim < KDim.start + 2,
        rs * dnmi,
        (rs + denratm * b33[-1] * ff[-2]) * dnmi,
    )


@gtx.scan(axis=KDim, forward=False, history=2)
def back_subst(
    rp: gtx.KView[float],
    ff: gtx.KView[float],
    ee: float,
    rs: float,
    dnmi: float,
    denratm: float,
    b33: gtx.KView[float],
) -> float:
    return concat_where(
        KDim == KDim.stop - 1,
        (ff[-2] * 2.0 * denratm * b33[-1] + rs) * dnmi,
        concat_where(
            KDim == KDim.stop - 2,
            ff[0] / (1.0 - ee),
            ee * rp[+2] + ff[0],
        ),
    )


@gtx.field_operator
def tridiagonal(rs: IJKF, b33: IJKF, ee: IJKF, dnmi: IJKF, denratm: IJKF) -> IJKF:
    ff = fwd_elim(rs, dnmi, denratm, b33, init=0.0, k_range=KDim[:-1])
    return back_subst(ff, ee, rs, dnmi, denratm, b33, init=0.0)
```

(`init` is structural in both scans — the peeled boundary regions seed the
chains, so it is never read and backends drop it. The `k_range` spelling
for "all but the last level" is open question 3.)

What this exercises beyond the earlier examples:

- **End-anchored conditions** (`KDim == KDim.stop - 1`, `KDim.stop - 2`):
  four of the five boundary regions are end-relative; GTScript spells them
  `interval(-1, None)` / `interval(-2, -1)`, muphys smuggles a runtime
  `last_lev` scalar in — anchors replace both.
- **Multi-level, nested peels**: a two-level region with one body at the
  start of the forward scan; two single-level regions with *different*
  bodies at the start of the backward scan, written as nested
  `concat_where`. The §3.3.1 partition guarantee must produce three
  branch-free loop sections for `back_subst`, not "peel the first
  iteration".
- **Guard-refined extents** (§3.1, §3.2, open question 9): `b33[-1]` is
  read only where `KDim >= KDim.start + 2` holds, so it must not clip the
  forward scan at the column start; `ff` is defined on one level less than
  the backward scan range, legal because the last-level branch reads only
  `ff[-2]`. A global extent analysis over all branches would reject both.
- **`history` carries** (§4.6): the stride-2 recurrences read the scan's
  own output at offset −2 (forward) / +2 (backward); `history=2` expresses
  them without hand-threading `(ff, ff_km1)` delay-line tuples.
- **Pre-factored solve**: `dnmi`, `ee`, `denratm` are factored once outside
  the GCR loop; only the sweeps re-run. The raw primitive, not
  `gtx.lib.tridiagonal_solve`, carries this case (§3.5).
- **Opposite-direction pair**: `ff` is materialized between the two scans,
  exactly as the GTScript 3D temporary is today — no regression, and a
  concrete instance of the fusion question (proposal §5.5: opposing
  directions are "attempted", not guaranteed).

## 8. Tropopause inversion + mixing length (PHYEX)

`GridTools/physics_patterns` PR #1 (Météo-France), from PHYEX
`condensation.F90` (research §A.9). Today: Fortran k-loops — an argmin
tracking `(ZTMIN, ITPL)` (minimum temperature and its level), then a
mixing-length loop whose recurrence branch depends on the height of the
per-column tropopause level `PZZ(JIJ, ITPL(JIJ))`.

Proposed — the level search is a reduction (§3.9), the height lookup a
computed-index gather (separate design discussion), the recurrence an
ordinary scan with a data-dependent `where`:

```python
@gtx.scan(axis=KDim, forward=True)  # ground -> space
def mixing_length_step(zl_below: float, zz: float, z_sfc: float, z_tpl: float) -> float:
    zzz = zz - z_sfc
    free_tropo = minimum(600.0, zzz)  # boundary-layer-limited length scale
    return where(zzz > 0.9 * (z_tpl - z_sfc), 0.6 * zl_below, free_tropo)


@gtx.field_operator
def mixing_length(t: IJKF, zz: IJKF, z_sfc: IJF) -> IJKF:
    itpl = gtx.argmin(t, axis=KDim)  # tropopause level: Field[[I, J], int32]
    z_tpl = zz(gtx.as_index(KDim, itpl))  # height at that level: Field[[I, J], float]
    return mixing_length_step(
        zz, z_sfc, z_tpl, init=20.0, k_range=KDim[1:], include_initial=True
    )  # ZL = 20 at the first level, emitted (§3.3 mechanism 2)
```

What disappeared: the in-carry `(t_min, k)` tracking with a hand-rolled
level counter (the §A.7 `_compute_param` idiom) and a full-3D
materialization for a column-constant quantity (`itpl`, `z_tpl` are
k-less). What it needs: `argmin` (§3.9) and the `as_index` gather
(separate design discussion; strawman spelling).

## 9. Per-column variable-depth accumulation (Noah LSM)

`GridTools/physics_patterns` PR #2 (NOAA), `lsm.py` (research §A.9): a
k-loop running to a per-column depth `nroot` (root zone; the PBL variant
varies at runtime), accumulating a k-less soil conductance. The trip count
is data, so no anchored condition applies — the §4.8 answer is a masked
full-range reduction, with the level index as a value (§3.9):

```python
@gtx.field_operator
def soil_conductance(zsoil: IJKF, sh2o: IJKF, smcwlt: IJF, smcref: IJF, nroot: IJF_int) -> IJF:
    gx = minimum(1.0, maximum(0.0, (sh2o - smcwlt) / (smcref - smcwlt)))
    dz = concat_where(KDim == KDim.start, zsoil, zsoil - zsoil(Koff[-1]))
    z_root = zsoil(gtx.as_index(KDim, nroot))  # computed-index gather again
    return gtx.sum(where(gtx.index(KDim) < nroot, dz / z_root * gx, 0.0), axis=KDim)
```

One masked reduction instead of a triple loop; the per-column bound costs
O(n_k) masked work instead of O(nroot) — the §4.8 vectorization trade.

## 10. SHiELD melting during sedimentation: the scope boundary

`GridTools/physics_patterns` PR #2 (NOAA), `microphysics.py` (research
§A.9). Upward k-loop over ice source levels; for each k an inner downward
m-loop melts and precipitates into the cells below, with two
data-dependent `break`s, scatter writes at both k and m (`qice[k] -= …`,
`qrain[m] += …`, temperature at both), and sequential feedback in both
loops (`qice[k]` exhausted across the inner loop; `temperature[m]`
updated and re-read by later k iterations).

There is deliberately no "after" code: the order-dependent `min` caps read
intermediate state, so no masked bounded-window gather computes the same
answer — research §C.3's regime 3 *with feedback*, outside the proposed
constructs (proposal §4.8). The options are algorithm reformulation (what
COSMO's boxtracking did for the same physics, research §C.1) or — if such
patterns accumulate during migration — revisiting the whole-column escape
hatch (proposal §4.1, §4.8).

## Appendix: sedimentation reference implementations

Transcribed from https://gist.github.com/havogt/c52d19f2f7557c2c048d12be7330bda8
(`test_sedim_box_standalone.py`), preserved here because it is the canonical
executable specification of the scan + sub-reduce pattern (research §C.2)
and a future acceptance test for §3.4. License of the original: WTFPL
(Copyright (c) 2021, ETH Zurich); reference:
http://www.cosmo-model.org/content/model/documentation/core/docu_sedi_twomom.pdf

```python
from typing import Iterable

import numpy as np
import numpy.typing as npt


def box_sedim_naive(v: npt.NDArray, z: npt.NDArray, phi: npt.NDArray, dt: npt.DTypeLike):
    Pbar = np.zeros_like(v)
    iter = 0
    for k in range(v.shape[0]):
        for m in range(k + 1):
            z_up = min(z[k - m - 1], z[k] + np.abs(v[k - m]) * dt)
            z_low = min(z[k - m], z_up)
            Pbar[k] -= phi[k - m] * (z_up - z_low)
            iter += 1
    print(f"iter = {iter}")
    return Pbar


class rolling_buffer:
    def __init__(self, size):
        self.data = np.zeros(size)

    def __getitem__(self, index):
        return self.data[index]

    def next(self, value):
        self.data = np.roll(self.data, 1)
        self.data[0] = value
        return self.data


def box_sedim_scanlike(v: npt.NDArray, z: npt.NDArray, phi: npt.NDArray, dt: npt.DTypeLike):
    def delta(z_state, v_state, m):
        z_up = min(z_state[1 + m], z_state[0] + np.abs(v_state[m]) * dt)
        z_low = min(z_state[m], z_up)
        return z_up - z_low

    Pbar = np.zeros_like(v)
    z_state = rolling_buffer(4)
    v_state = rolling_buffer(3)
    phi_state = rolling_buffer(3)
    for k in range(v.shape[0]):
        z_state.next(z[k])
        v_state.next(v[k])
        phi_state.next(phi[k])
        if k <= 1:
            Pbar[k] = -phi_state[0] * delta(z_state, v_state, 0)
        elif k == 2:
            Pbar[k] = -phi_state[0] * delta(z_state, v_state, 0) - phi_state[1] * delta(
                z_state, v_state, 1
            )
        else:
            Pbar[k] = (
                -phi_state[0] * delta(z_state, v_state, 0)
                - phi_state[1] * delta(z_state, v_state, 1)
                - phi_state[2] * delta(z_state, v_state, 2)
            )
    return Pbar


def box_sedim_scanpass(k, state, v, z, phi, dt):  # k for simplicity
    def delta(z_state, v_state, m):
        z_up = min(z_state[1 + m], z_state[0] + np.abs(v_state[m]) * dt)
        z_low = min(z_state[m], z_up)
        return z_up - z_low

    _, z_state, v_state, phi_state = state
    z_state.next(z)
    v_state.next(v)
    phi_state.next(phi)
    if k <= 1:
        return -phi_state[0] * delta(z_state, v_state, 0), z_state, v_state, phi_state
    elif k == 2:
        return (
            -phi_state[0] * delta(z_state, v_state, 0) - phi_state[1] * delta(z_state, v_state, 1),
            z_state,
            v_state,
            phi_state,
        )
    else:
        return (
            -phi_state[0] * delta(z_state, v_state, 0)
            - phi_state[1] * delta(z_state, v_state, 1)
            - phi_state[2] * delta(z_state, v_state, 2),
            z_state,
            v_state,
            phi_state,
        )


def scan(scanpass, range_, init):
    def get_n(n):
        def foo(arg):
            if isinstance(arg, Iterable):
                return arg[n]
            else:
                return arg

        return foo

    def foo(*args):
        Pbar = np.zeros(len(range_))
        state = init
        for k in range_:
            state = scanpass(k, state, *list(map(get_n(k), args)))
            Pbar[k] = state[0]  # for simplicity
        return Pbar

    return foo


def box_sedim_scan(v: npt.NDArray, z: npt.NDArray, phi: npt.NDArray, dt: npt.DTypeLike):
    return scan(
        box_sedim_scanpass,
        range(v.shape[0]),
        (None, rolling_buffer(4), rolling_buffer(3), rolling_buffer(3)),
    )(v, z, phi, dt)


class shift_origin:
    def __init__(self, arr: npt.NDArray, offset: int):
        self.arr = arr
        self.offset = offset

    def __getitem__(self, index):
        return self.arr[index + self.offset]


def box_sedim_naive_func(v: npt.NDArray, z: npt.NDArray, phi: npt.NDArray, dt: npt.DTypeLike):
    Pbar = np.zeros_like(v)
    iter = 0
    for k in range(v.shape[0]):

        def foo(m, v, z, phi, dt):
            nonlocal iter
            z_up = min(z[-m - 1], z[0] + np.abs(v[-m]) * dt)
            z_low = min(z[-m], z_up)
            iter += 1
            return -phi[-m] * (z_up - z_low)

        Pbar[k] = sum(
            foo(m, shift_origin(v, k), shift_origin(z, k), shift_origin(phi, k), dt)
            for m in range(k + 1)
        )
    print(f"iter = {iter}")
    return Pbar


def box_sedim_naive_break(v: npt.NDArray, z: npt.NDArray, phi: npt.NDArray, dt: npt.DTypeLike):
    Pbar = np.zeros_like(v)
    iter = 0
    for k in range(v.shape[0]):
        for m in range(k):
            if z[k - m] > z[k] + np.abs(v[k - m] * dt):
                break
            z_up = min(z[k - m - 1], z[k] + np.abs(v[k - m]) * dt)
            z_low = min(z[k - m], z_up)
            Pbar[k] -= phi[k - m] * (z_up - z_low)
            iter += 1
    print(f"iter = {iter}")
    return Pbar


def box_sedim_paper(v: npt.NDArray, delta_z: npt.NDArray, phi: npt.NDArray, dt: npt.DTypeLike):
    Pbar = np.zeros_like(v)
    for k in range(v.shape[0]):
        m = 0
        delta_sum = 0.0
        while delta_sum < np.abs(v[k]) * dt and k + m < v.shape[0]:
            # z_low = 0.0
            z_up = min(delta_z[k], -delta_sum + np.abs(v[k]) * dt)
            Pbar[k + m] -= phi[k] * z_up
            m += 1
            if k + m < v.shape[0]:
                delta_sum += delta_z[k + m]
    return Pbar


def test_box_sedim():
    z = np.asarray(range(11, 0, -1))
    delta_z = -(z[1:] - z[:-1])
    v = np.array([0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.5, 0.5])
    dt = 1.0
    phi = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    ref = np.asarray([-0.05, -0.05, -0.1, -0.1, -0.1, -0.2, -0.2, -0.2, -0.25, -0.15])

    paper = box_sedim_paper(v, delta_z, phi, dt)
    naive = box_sedim_naive(v, z, phi, dt)
    naive_break = box_sedim_naive_break(v, z, phi, dt)
    naive_func = box_sedim_naive_func(v, z, phi, dt)
    scan_like = box_sedim_scanlike(v, z, phi, dt)
    scanned = box_sedim_scan(v, z, phi, dt)

    print(paper)
    print(naive)
    print(naive_break)
    print(naive_func)
    print(scan_like)
    print(scanned)

    assert np.allclose(paper, ref)
    assert np.allclose(paper[1:], naive[1:])  # TODO check bounds
    assert np.allclose(paper[1:-1], naive_break[1:-1])  # TODO check bounds
    assert np.allclose(paper[1:], naive_func[1:])  # TODO check bounds
    assert np.allclose(paper[1:], scan_like[1:])  # TODO check bounds


test_box_sedim()
```
