---
title: Redesign of the vertical scan in gt4py.next
author: sf-n, tehrengruber
tags: [scan, vertical, reduction, boundary-conditions, windows, k-caches, fusion, embedded, jax, frontend]
created: 2026-06-11
status: draft
---

# Scan patterns for the gt4py.cartesian → gt4py.next port

Classification of every `with computation(FORWARD/BACKWARD)` on `real_cases_radiation` in the
dycore, `physics.cloud_microphysics` and `physics.radiation`.

**Rule of thumb:** a `scan_operator` is needed only if the block
(a) reads a field at `[0,0,-1]` (FORWARD) / `[0,0,+1]` (BACKWARD) that is *written in the same
computation* (vertical recurrence), or
(b) reduces the column into a 2D result (sum / max / cumulative product / "first/last level where …").
Everything else — single-level `interval(0, 1)` writes, per-band formulas, offset reads of
*inputs* — becomes a `field_operator` (+ `concat_where` for interval boundaries), matching the
convention on the `precompilation` branch (true scans stayed cartesian there, marked
`TODO: port to next fo when scans are available`, until the first three were ported — see
"Status" below).

**Verdicts:** SCAN = port as `scan_operator` (see "How the true scans map…"). FIELD_OP = rewrite
as one or more gt4py.next `field_operator`s; the "how" letters (a)–(e) in the tables refer to the
rewrite patterns in the appendix:
(a) host-side slice at operator level + `broadcast` · (b) replace carry by `field(KDim - 1)` read · (c) two-step interior +
edge fill · (d) input shifts + `concat_where` per interval band · (e) pure 2D field_op over
`(IDim, JDim)`.

## Summary

| Area                               | Occurrences | Need scan                      | Field-operator |
| ---------------------------------- | ----------- | ------------------------------ | -------------- |
| Dycore (`src/pmap`, excl. physics) | 80          | 16 stencils                    | rest           |
| `physics.cloud_microphysics`       | 29          | 4 (+4 in `_dev`)               | rest           |
| `physics.radiation` (ecrad_cy49r1) | 58          | 26 blocks (~10 unique kernels) | rest           |

**Status (precompilation branch, gtfn, gt4py fork `pmap_staging6`):** ported and validated
against the cartesian stencils — `_solve_itcs` (`@ported_stencil`), helmholtz `_tridiagonal`
(in `Preconditioner_next`), `_trapezoidal_z` (`ColumnIntegral_next` / `DomainIntegral_next`),
plus `_solve_itcs`'s surface-flux update as a 2D field_op (`_update_sflux_next`). The sketches
for these show the *working* code; all other sketches are untested.

---

## 1. Dycore — 16 scan stencils

| File                                                  | Stencil                                            | Dir | Recurrence / carried state                                                               |
| ----------------------------------------------------- | -------------------------------------------------- | --- | ---------------------------------------------------------------------------------------- |
| `diffusion/scalar.py:440,476`                         | `_solve_itcs`                                      | F+B | Thomas solve: `ee, ff` at `[-1]`; back-sub `psi = ee·psi[+1] + ff` — **ported**          |
| `diffusion/scalar_z.py:197,231`                       | `_solve_itcs`                                      | F+B | Same Thomas solve                                                                        |
| `diffusion/vector.py:797,887`                         | `_solve_itcs`                                      | F+B | Three coupled Thomas solves (`ee_*, ff_*`)                                               |
| `operators/nabla.py:353–365`                          | `_nabla_z_cmp4_solve`                              | F+B | Compact-scheme Thomas solve                                                              |
| `operators/nabla.py:485–508`                          | `_nabla_z_cmp6_solve`                              | F+B | Same, per-band coefficients                                                              |
| `helmholtz/preconditioner.py:61,68`                   | `_tridiagonal`                                     | F+B | **Stride-2** recurrence (`ff[0,0,-2]`, `rp[0,0,+2]`) → carry needs 2 levels — **ported** |
| `poisson/preconditioner.py:59,66`                     | `_tridiagonal`                                     | F+B | Same stride-2 solve                                                                      |
| `helmholtz/preconditioner_coefficients.py:186`        | `_tridiagonal_init`                                | F   | Stride-2 LU factorisation (`ee[0,0,-2]`)                                                 |
| `poisson/preconditioner_coefficients.py:109`          | `_tridiagonal_init`                                | F   | Same                                                                                     |
| `operators/hydrostatic_exner.py:59`                   | `hydrostatic_exner`                                | B   | `pexhyd = pexhyd[+1] + …` (top-down hydrostatic integral)                                |
| `operators/integral.py:140`                           | `_trapezoidal_z`                                   | B   | 2D column sum `ci_psi +=` — **ported** (operator level)                                  |
| `operators/cloud_cover.py:203,225`                    | `_cloud_cover_maximum/_random`                     | F   | 2D carry: `max` / cumulative product                                                     |
| `operators/cloud_cover.py:259,290,319`                | `_cloud_base/top_height`, `_cloud_top_temperature` | F   | First/last qualifying level (2D carry)                                                   |
| `diagnostics/boundary_layer_height.py:102`            | `_boundary_layer_height`                           | F   | First level with `rib > 0.25`; also reads own temporaries at `[-1]`                      |
| `boundary_conditions/cell_perturbation_method.py:424` | `wind_bl_height`                                   | F   | Level search bracketing `bl_height` into 2D                                              |

**Field-operator rewrites (no scan):**

| Stencils | How |
|---|---|
| All `interval(0, 1)` stencils in `operators/basic.py`, `surface_layer.py`, `resolved_statistics.py`, `pressure_diagnostics.py`, `relations.py`, `forces.py`, `spherical.py`, `timeseries.py`, `time_dependent_data.py`, `integral.py:151,159`, cell-perturbation setup | (e) pure 2D field_op over `(IDim, JDim)`; inputs sliced to their k=0 / surface level by the caller (`operators/basic.py` on `precompilation` shows the template) |
| `scaling_richardson_number` (`cell_perturbation_method.py:453`) | (d) one field_op, the three one-sided/centred differences selected by nested `concat_where(KDim == 0, …, concat_where(KDim == nz - 1, …, centred))` |
| `ppm/common/cell_face_values.py` (6 bands + 2 edge copies) | (d)+(c): field_op #1 computes all interior bands via nested `concat_where` on `KDim`; field_op #2 fills k=0 / k=nz−1 from the shifted result (sketch in appendix (c)) |
| rhs/sflux blocks around each Thomas solve (`scalar[_z].py:437/479`, `vector.py:792/892`) | `rhs`: pointwise expression in the wrapping field_op *before* the scan pair. `sflux`: a 2D update from `psi` at k=0 — a field_op cannot drop K, but the operator can pass that level as a zero-copy 2D view of the next field (`psi.local_field.next_field[KDim(k_bottom)]`) into a pure 2D field_op (`_update_sflux_next`, called on the `(I, J)` index space) — **ported** |

## 2. cloud_microphysics — 4 scans (+ `_dev`)

| File | Stencil | Dir | Recurrence / carried state |
|---|---|---|---|
| `cloudsc/cy49r1/_stencils/cloudsc_cy49r1.py:389` | main loop | B | Precip fluxes from level above: `fallsink_q*[+1]`, `q*n[+1]`, `rdtgdp[+1]`; 2D carries `tmp_covptot/covpclr/covpmax/cldtopdist/rainfrac_toprfz/rainliq` |
| `cloudsc/cy50r1/_stencils/cloudsc_cy50r1.py:384` | main loop | B | Same pattern as cy49r1 |
| `cloudsc/dwarf/_stencils/cloudsc_dwarf.py:326` | main loop | B | Same: `pfpls*[+1]`, `anew[+1]`, same 2D carries |
| `diagnostics.py:37` | `tropopause` | F | Last level where flag set (`trpaus[-1]` carry) |

**Field-operator rewrites (no scan):**

| Stencils | How |
|---|---|
| Setup blocks before each main loop (`cloudsc_cy49r1.py:208`, `cy50r1:207`, `dwarf:177`) | Split in two field_ops: (1) 3D level-local temporaries (`a, tp1, q*, rho, dp, rdtgdp`) — only input shift is `in_aph(KDim - 1)`; (2) 2D zero-init of the `tmp_*` carriers, which becomes the scan's `init` tuple instead of a stencil |
| FSD block (`cloudsc_cy49r1.py:2466`) | Pointwise 3D field_op — all reads at `[0,0,0]`, the BACKWARD order was irrelevant |
| `cloud_satadj` (both cycles) | (d) one field_op: `concat_where(KDim < NCLDTOP, 0.0, f_cloud_satadj(...))`; `in_aph(KDim + 1)` is an input shift. The `tmp_aph_s` init is a surface slice (a) |
| `cloud_supersatcheck`, driver `setup` (`gridlength`) | (e) pure 2D field_ops |
| `kessler/kessler.py` (all 5 blocks) | (d) sedimentation flux from input shifts `rhod/rr/zcr(KDim ± 1)` + `concat_where(KDim == nz - 1, bottom_formula, interior)`; `rhods`/`prect` are 2D field_ops (e) |
| `diagnostics.py:21,45` (`eta_levels`, tropopause broadcast) | (a) surface/top slice (host side) + `broadcast` (sketch in appendix (a)); the tropopause FWD half stays a scan (operator-level port, 2D result) |

**`_dev` (low priority):** `cloudsc2.py:390,406` are true scans (`trpaus[+1]`; `covptot/out_rfl/
out_sfl[+1]`); `:401` is a broadcast; `:488` is a pure shift of inputs → field_op.
`kessler_sedimentation.py:34` is intended as a scan but see "suspicious code" below.

## 3. radiation (ecrad_cy49r1) — ~10 unique scan kernels

`calc_radiative_fluxes_unified.py` duplicates the lw/sw/optical-properties stencils; verdicts
transfer 1:1 (unified line in parentheses).

| File:line | Kernel | Dir | Recurrence / carried state |
|---|---|---|---|
| `calc_optical_properties.py:162` (u:606) | cloud cover | B | Cumulative product `v_mat_prod *= v_matrix` |
| `calc_optical_properties.py:212` (u:654) | cloud top index | B | 2D max-reduction `i_cloud_top` |
| `calc_radiative_fluxes_lw.py:395` (u:704) | adding, up | F | `albedo[-1]`, `source[-1]` |
| `calc_radiative_fluxes_lw.py:419` (u:728) | flux down | B | `flux_dn[+1]` |
| `calc_radiative_fluxes_lw.py:449` (u:758) | no-scatter up | F | `flux_up[-1]` |
| `calc_radiative_fluxes_lw.py:595` (u:906) | total albedo LW | F | `total_albedo[-1]`, `total_source[-1]` |
| `calc_radiative_fluxes_lw.py:664+689` (u:975+1000) | above-cloud flux up | B+F | 2D carry `tmp_flux_up`, started at data-dependent `i_cloud_top` → fuse into one scan |
| `calc_radiative_fluxes_lw.py:718+730` (u:1029+1041) | cloudy flux down | B | 2D carries `tmp_flux_dn/up/below`, started at `i_cloud_top` → fuse |
| `calc_radiative_fluxes_lw.py:791` (u:1103) | LW derivatives | F | 2D carry `lw_deriv` (unified fuses in a SW top-slice part that can split out as field_op) |
| `calc_radiative_fluxes_sw.py:474` (u:1391) | total albedo SW | F | `total_albedo[-1]` (+clear, +direct) |
| `calc_radiative_fluxes_sw.py:527` (u:1444) | SW fluxes | B | 2D carries `tmp_flux_dn`, `tmp_below`, `tmp_flux_dn_clear` |

**Field-operator rewrites (no scan):**

| Stencils | How |
|---|---|
| `albedo.py` (6), `cloud_overlap_param.py`, `effective_radius.py`, `zenith_angle.py`, `calc_radiative_fluxes_2d.py` | (e) pure 2D field_ops; the band/canopy loops become data dims or unrolled tuples (see hazards) |
| Per-layer optics (`calc_optical_properties.py:78`, `lw:229`, `sw:235`, u:400/1152) | Pointwise 3D field_ops; only input shifts (`pressure_hl`/`temperature_hl(KDim ± 1)`). The 2D aerosol temporaries rewritten per level are really per-level values → make them 3D locals |
| Fixed-level top/surface extractions (`lw:467,585`, `sw:208,440`, u:364/776/895/1357/1544) | Per-level part (g-point sums, Planck surface source) as pointwise field_op; the TOA/surface 2D outputs are fixed-index slices of it (a) — not data-dependent, no scan |
| `calc_heating_rates.py` (all 4 blocks, incl. `:326`) | `:326` per pattern (b): `hr[k] = f·(x(KDim − 1) − x)` with `concat_where(KDim > 0, …)` — the `tmp_*_above` carry equals `updated_flux_*_net[k−1]`. The other blocks are surface/TOA slices (a)/(e) |
| Driver `setup_data:783` | (c) two-step: interior `t_hl` from the PARALLEL block, then boundary levels patched via `concat_where(KDim == 0 / nz, shifted interior, t_hl)` |

---

## How the true scans map to gt4py.next `scan_operator`s

A `scan_operator` is a per-column sequential loop: it gets the *carry* (state from the previous
level) plus the current level's inputs, and returns the new carry. 2D `tmp_*` fields and
cartesian self-reads at `[0,0,∓1]` both become entries in the carry tuple. There is only one
scan mechanism — a first-order recurrence `state_k = f(state_{k−1}, inputs_k)` — and *every*
vertical recurrence can be written that way (deeper recurrences like the stride-2 tridiagonals
just widen the carry tuple). The distinct patterns occurring in PMAP:

| # | Pattern | Carry | Result used | Occurrences |
|---|---|---|---|---|
| 1 | Thomas forward elimination | `(k, ee, ff)` per system | per level (3D) | `_solve_itcs` ×3 (scalar **ported**; vector: 3 coupled systems), `_nabla_z_cmp4/6`, `_tridiagonal_init` |
| 2 | Thomas back-substitution | solution `x` | per level (3D) | second half of every solve in 1 |
| 3 | Stride-2 recurrence | **two** levels of state (+ what the other sweep needs from unvisited levels) | per level (3D) | helmholtz/poisson `_tridiagonal(_init)` (offsets `±2`) — helmholtz **ported** |
| 4 | Running integral | accumulated sum | per level (3D) | `hydrostatic_exner` |
| 5 | Column sum | accumulator | final only → 2D (host-side slice) | `_trapezoidal_z` — **ported** |
| 6 | Column max | accumulator | final only → 2D | `_cloud_cover_maximum`, `i_cloud_top` (opt_props:212) |
| 7 | Cumulative product | product | final only → 2D | `_cloud_cover_maximum_random`, `v_mat_prod` (opt_props:162) |
| 8 | First-hit level search | value + "found" sentinel | final only → 2D | `_cloud_base_height`, `_boundary_layer_height`, `wind_bl_height` |
| 9 | Last-hit level search | value, overwritten on condition | final only → 2D | `_cloud_top_height/_temperature`, `tropopause` |
| 10 | Linear propagation `x = a·x_prev + b` | flux(es)/albedo/source | per level (3D) | radiation lw:395/419/449/595/791, sw:474/527 (+ unified twins) |
| 11 | Mid-column start (init injected at `k == i_cloud_top`) | flux + started-flag | per level (3D) | lw:664+689, 718+730 (+ unified) |
| 12 | Multi-field cascade, mixed carry | per-level fluxes **+** accumulated 2D state | per level *and* final 2D | cloudsc cy49r1/cy50r1/dwarf main loops; `_dev` cloudsc2 core, kessler_sedimentation |

They differ only in what the carry holds and whether per-level values or only the final one are
used; 4 vs 5 is the same recurrence with a different consumer, 8 vs 9 differ only in the gate,
and 12 mixes 10 with 5–7 in one carry tuple. Representative sketches:

**A. Thomas solve (patterns 1–3; 4 is the same with a sum as carry)** → one forward + one
backward scan, carry = the sweep variables `(ee, ff)` plus a level counter.

*The Thomas algorithm.* An implicit vertical discretisation couples each level only to its two
neighbours, giving one tridiagonal system per column,

    a_k·x_{k−1} + b_k·x_k + c_k·x_{k+1} = d_k,    k = 0 … n−1,    a_0 = c_{n−1} = 0,

with `a` the sub-diagonal (coupling to the level below), `b` the diagonal, `c` the
super-diagonal (coupling to the level above) and `d` the right-hand side. Gaussian
elimination specialised to this shape is two sweeps:

1. *Forward elimination (upward).* Suppose the level below is already expressed through the
   current one, `x_{k−1} = ee_{k−1}·x_k + ff_{k−1}`. Substituting into row k gives
   `(b_k + a_k·ee_{k−1})·x_k + c_k·x_{k+1} = d_k − a_k·ff_{k−1}`, i.e. the same form one level up:

       ee_k = −c_k / (b_k + a_k·ee_{k−1}),      ff_k = (d_k − a_k·ff_{k−1}) / (b_k + a_k·ee_{k−1}),

   started at the surface with `ee_0 = −c_0/b_0`, `ff_0 = d_0/b_0` (`a_0 = 0`: nothing below).
   Level k needs only `(ee, ff)` of level k−1 → a **forward scan** with that pair as carry.
2. *Back-substitution (downward).* At the top `c_{n−1} = 0`, so `ee_{n−1} = 0` and
   `x_{n−1} = ff_{n−1}`; then `x_k = ee_k·x_{k+1} + ff_k` level by level → a **backward scan**
   whose carry is `x_{k+1}`.

(Sign convention as in `scalar.py`: `x_k = ee·x_{k+1} + ff` with `ee = −c/(…)`; textbooks often
write `x_k = ff − ee′·x_{k+1}` with `ee′ = −ee`.)

**Ported**: `diffusion/scalar.py:440` (`_solve_itcs`) — implicit vertical diffusion of `psi` with
`bbz` the diffusion coefficient on the half levels (`[k−½]` = face below cell k, cartesian
`bbz[0,0,0]`; `[k+½]` = face above, `bbz[0,0,1]`) and `dti = 1/dt`. The coefficients are

    a_k = −bbz[k−½],   b_k = rho_k·dti + bbz[k−½] + bbz[k+½],   c_k = −bbz[k+½],
    d_k = rhs_k = rho_k·psi_k^old·dti + fluxdiv_k,

so `b_k + a_k·ee_{k−1} = rho·dti + bbz_lo + bbz_up − bbz_lo·ee_{k−1}` is exactly the cartesian
`1/zdenomi`, `ee_k = −c_k·zdenomi = bbz_up·zdenomi` and `ff_k = (rhs + bbz_lo·ff_{k−1})·zdenomi`
(lines 459–463). The surface row (`interval(0, 1)`) folds the surface-flux boundary condition
into `b_0` and `d_0` — `b_0 = rho·dti + (1+sbc_a)·(bbz_up + sflux_enabled·(1−explicit)·beta·dzi·k_rho_sfc)`,
`c_0 = −(1+sbc_a)·bbz_up`, `d_0 = rhs ∓ …sflux/psi_sfc terms` — and the top row
(`interval(-1, None)`) folds the top boundary condition into `a_{n−1} = −(1+tbc_a)·bbz_lo`,
`b_{n−1} = rho·dti + (1+tbc_a)·bbz_lo`, with `c_{n−1} = 0` ⇒ `ee = 0`. The code as it runs on
`precompilation`:

```python
# No level index exists inside a scan: the carry counts the levels (k = 0 at the first level
# of the compute domain); k_top = last_idx[2] - first_idx[2] comes from porting.unify.
@scan_operator(axis=KDim, forward=True, init=(int32(0), float_type(0.0), float_type(0.0)))
def _solve_itcs_forward(carry: tuple[int32, float_type, float_type],
                        rhs: float_type, rho: float_type,
                        bbz_lo: float_type,   # bbz[0,0,0] → bbz(KDim - 0.5)  (lower face)
                        bbz_up: float_type,   # bbz[0,0,1] → bbz(KDim + 0.5)  (upper face)
                        sflux: float_type, k_rho_sfc: float_type, psi_sfc: float_type,
                        beta: float_type, dti: float_type, dzi: float_type,
                        sbc_a: float_type, tbc_a: float_type,           # bools arrive as 0.0/1.0
                        explicit_sflux: float_type, sflux_enabled: float_type,
                        k_top: int32) -> tuple[int32, float_type, float_type]:
    k, ee_km1, ff_km1 = carry         # ← was ee[0,0,-1], ff[0,0,-1]
    one = float_type(1.0)
    if k == int32(0):                 # ← was interval(0, 1): surface BC row
        zdenomi = one / (rho * dti + (one + sbc_a)
                         * (bbz_up + sflux_enabled * (one - explicit_sflux) * beta * dzi * k_rho_sfc))
        ee = (one + sbc_a) * bbz_up * zdenomi
        if explicit_sflux == one:
            ff = (rhs - (one + sbc_a) * beta * sflux * dzi) * zdenomi
        else:
            ff = (rhs + sflux_enabled * (one + sbc_a) * beta * dzi * k_rho_sfc * psi_sfc) * zdenomi
    elif k == k_top:                  # ← was interval(-1, None): top BC row, ee = 0
        zdenomi = one / (-(one + tbc_a) * bbz_lo * ee_km1 + rho * dti + (one + tbc_a) * bbz_lo)
        ee = float_type(0.0)
        ff = (rhs + (one + tbc_a) * bbz_lo * ff_km1) * zdenomi
    else:                             # ← was interval(1, -1)
        zdenomi = one / (-bbz_lo * ee_km1 + rho * dti + bbz_up + bbz_lo)
        ee = bbz_up * zdenomi
        ff = (rhs + bbz_lo * ff_km1) * zdenomi
    return k + int32(1), ee, ff       # the returned tuple is the next level's carry

@scan_operator(axis=KDim, forward=False, init=float_type(0.0))
def _solve_itcs_backward(psi_kp1: float_type, ee: float_type, ff: float_type) -> float_type:
    return ee * psi_kp1 + ff          # ← was computation(BACKWARD): psi = ee*psi[0,0,1] + ff

# The wrapper: scans are called like field_operators, with full fields; gt4py applies the
# scalar body column by column. porting.unify injects first_idx/last_idx and adapts the call
# to the stencil convention (indices, index_space, ..., subdomain=...).
@porting.unify(outputs=["psi"], inout_arg=True, kw_only=True, add_first_idx=True, add_last_idx=True)
@field_operator(backend=build_config.backend_next, grid_type=GridType.CARTESIAN, static_domains=True)
def _solve_itcs_next(psi: IJKField, rho: IJKField, fluxdiv: IJKField, bbz: IJKHalfField,
                     sflux: IJField, k_rho_sfc: IJField, psi_sfc: IJField,
                     beta: float_type, dti: float_type, dzi: float_type,
                     sbc_a: float_type, tbc_a: float_type,
                     explicit_sflux: float_type, sflux_enabled: float_type,
                     first_idx: tuple[int32, int32, int32],
                     last_idx: tuple[int32, int32, int32]) -> IJKField:
    rhs = rho * psi * dti + fluxdiv                       # ← line 437, pointwise pre-work
    k_top = last_idx[2] - first_idx[2]
    _, ee, ff = _solve_itcs_forward(
        rhs, rho, bbz(KDim - 0.5), bbz(KDim + 0.5),
        broadcast(sflux, (IDim, JDim, KDim)),             # scan args must all have KDim
        broadcast(k_rho_sfc, (IDim, JDim, KDim)), broadcast(psi_sfc, (IDim, JDim, KDim)),
        beta, dti, dzi, sbc_a, tbc_a, explicit_sflux, sflux_enabled, k_top)
    return _solve_itcs_backward(ee, ff)                   # a scalar scan may be the return value

@porting.ported_stencil(_solve_itcs_next)   # switches via build_config.porting_* / porting_variant()
def _solve_itcs(...):                        # the cartesian stencil, unchanged except for the
    ...                                      # sflux block, which became the 2D field_op below

# The sflux update reads psi on the lowest level only: a field_op cannot drop K, but the
# next field can be indexed at one level (a zero-copy 2D view) and fed to a 2D field_op.
@porting.unify(outputs=["sflux"], inout_arg=True, kw_only=True)
@field_operator(backend=build_config.backend_next, grid_type=GridType.CARTESIAN, static_domains=True)
def _update_sflux_next(psi_bottom: IJField, sflux: IJField, k_rho_sfc: IJField, psi_sfc: IJField,
                       beta: float_type, explicit_sflux: float_type, sflux_enabled: float_type) -> IJField:
    one = float_type(1.0)
    enabled = (one - explicit_sflux) * sflux_enabled   # `if not explicit_sflux and sflux_enabled`
    updated = (one - beta) * sflux + beta * (psi_bottom - psi_sfc) * k_rho_sfc
    return enabled * updated + (one - enabled) * sflux # exact 0/1 blend, no runtime scalar `if`

# in Scalar.__call__, right after the solve:
k_bottom = self.indices.index_spaces[I, J, K].subset["physical"][0, 0, 0][2]
_update_sflux_next(self.indices, (I, J), psi_bottom=psi.local_field.next_field[KDim(k_bottom)],
                   sflux=sflux, k_rho_sfc=k_rho_sfc, psi_sfc=psi_sfc, beta=beta,
                   explicit_sflux=explicit_sflux, sflux_enabled=sflux_enabled, subdomain="physical")
```

Notes: the three `interval` bands become `if`s on the counter; the half-level `bbz` pair
becomes two scalar args shifted with `(KDim ∓ 0.5)` in the wrapper (the faces of cell k exist
for every k, so these shifts stay inside the domain); `psi_top = ff` (cartesian line 474) has
no counterpart — it is the backward scan's first iteration, since `ee = 0` at the top gives
`psi = 0·init + ff`; the bool flags arrive as `float_type` 0/1 (unify converts them) and are
used arithmetically as in cartesian; the `rhs` assembly lives in the wrapper, and the final
`sflux` update is the 2D field_op above, fed with the K-indexed view — no cartesian code is
left in the solve. Validated: `tests/benchmarks/test_diffusion.py` (`TestScalar`,
`Test_solve_itcs`, `test_update_sflux_next_formula`) — next vs cartesian to 1e-10 for all 16
flag combinations.

**Stride-2 variant (pattern 3) — ported:** `helmholtz/preconditioner.py:61` `_tridiagonal`
(poisson is identical) eliminates two interleaved odd/even-level chains, so it reads
`ff[0,0,-2]` and `rp[0,0,2]`. The carry holds the last **two** levels and is shifted by one each
step. Two things a naive sketch gets wrong, which the working code (`preconditioner_next.py`)
handles: the cartesian reads of `b33[0,0,-1]` / `ff[0,0,-2]` happen only for `k >= 2`, but a
wrapper shift `b33(KDim - 1)` is evaluated for *every* level, i.e. outside the domain at
k = 0, 1 — so the scan carries `b33` too; and the top row of the back-substitution needs
`ff[k-2]`, `b33[k-1]` from *below*, which the backward sweep has not visited — so the forward
sweep computes that value (`rp_top`, meaningful only at the last level) and hands it over:

```python
@scan_operator(axis=KDim, forward=True,
               init=(int32(0), float_type(0.0), float_type(0.0), float_type(0.0), float_type(0.0)))
def _tridiagonal_forward(carry: tuple[int32, float_type, float_type, float_type, float_type],
                         rs: float_type, b33: float_type, dnmi: float_type, denratm: float_type):
    k, ff_km1, ff_km2, b33_km1, _rp_top = carry       # ← ff[0,0,-1], ff[0,0,-2], b33[0,0,-1]
    if k < int32(2):                                   # ← interval(0, 2): both chain starts
        ff = rs * dnmi
    else:                                              # ← interval(2, -1)
        ff = (rs + denratm * b33_km1 * ff_km2) * dnmi
    rp_top = (ff_km2 * float_type(2.0) * denratm * b33_km1 + rs) * dnmi   # top row of the back-sub
    return k + int32(1), ff, ff_km1, b33, rp_top       # shift the two-level window

@scan_operator(axis=KDim, forward=False, init=(int32(0), float_type(0.0), float_type(0.0)))
def _tridiagonal_backward(carry: tuple[int32, float_type, float_type],
                          ff: float_type, ee: float_type, rp_top: float_type):
    j, rp_kp1, rp_kp2 = carry                          # ← rp[0,0,1], rp[0,0,2]; j counts from the top
    if j == int32(0):                                  # ← interval(-1, None)
        rp = rp_top
    elif j == int32(1):                                # ← interval(-2, -1)
        rp = ff / (float_type(1.0) - ee)
    else:                                              # ← interval(0, -2)
        rp = ee * rp_kp2 + ff
    return j + int32(1), rp, rp_kp1

@porting.unify(outputs=["rp"])
@field_operator(backend=build_config.backend_next, grid_type=GridType.CARTESIAN, static_domains=True)
def _tridiagonal_next(rs: IJKField, b33: IJKField, ee: IJKField, dnmi: IJKField,
                      denratm: IJKField) -> IJKField:
    _k, ff, _ff_km1, _b33_km1, rp_top = _tridiagonal_forward(rs, b33, dnmi, denratm)
    _j, rp, _rp_kp1 = _tridiagonal_backward(ff, ee, rp_top)
    # gtfn accepts a scan only as the whole expression of a closure, not an element of its
    # tuple result; the (exact) multiplication gives the scan its own closure.
    return rp * float_type(1.0)
```

Cartesian leaves `ff` unset at the last level (`interval(2, -1)`) and never reads it; the scan
computes it there too, harmlessly. `preconditioner.py` is ported at *operator* level
(`@ported_operator(Preconditioner_next)`), so the cartesian `_tridiagonal` stays a plain
`@stencil` and `Preconditioner_next` calls `_tridiagonal_next`. `ee`, `dnmi`, `denratm` come
from `_tridiagonal_init` (preconditioner_coefficients), which would port the same way with
carry `(k, ee_km1, ee_km2, …)` — not done yet, nor the poisson twin. Validated:
`tests/benchmarks/test_helmholtz_preconditioner.py` (`TestPreconditioner`, `Test_tridiagonal`).

**B. Column reduction / level search into 2D (patterns 5–9)** — gt4py.next has no K-reduction
builtin and no field_op can drop the K dimension, so: scan into a 3D temporary, then take the
last-processed level on the host at operator level (`out["covering"] = tmp["covering"][:, :, -1]`
for a forward scan, `[:, :, 0]` for a backward one). These are therefore **operator-level ports**
(`@ported_operator`), not `@ported_stencil`s:

```python
@scan_operator(axis=KDim, forward=True, init=float_type(0.0))
def cc_max(carry: float_type, cf: float_type) -> float_type:
    return maximum(carry, cf)
# 3D result, answer at the last level → the operator copies tmp["covering"][:, :, -1] into the
# 2D field (or broadcasts it back, cf. fake scans (a)). "First level where…" gates on the carry.
```

**Ported**: `operators/integral.py:140` `_trapezoidal_z` (pattern 5; trapezoidal weights 0.5 at
both ends, 1.0 interior). Keeping the weights *outside* the scan (`concat_where` on
`first_idx`/`last_idx`, same operation order as `_integf`) leaves a scalar carry, so no counter
is needed and the scan is a closure of its own (`operators/integral_next.py`):

```python
@scan_operator(axis=KDim, forward=False, init=float_type(0.0))
def _column_sum_down(acc_kp1: float_type, integrand: float_type) -> float_type:
    return acc_kp1 + integrand                       # ← the 2D ci_psi was the carry

@porting.unify(outputs=["ci_psi"], kw_only=True, add_first_idx=True, add_last_idx=True)
@field_operator(backend=build_config.backend_next, grid_type=GridType.CARTESIAN, static_domains=True)
def _trapezoidal_z_next(psi: IJKField, rho: IJKField, gac: IJKField, dz: float_type,
                        first_idx: tuple[int32, int32, int32],
                        last_idx: tuple[int32, int32, int32]) -> IJKField:
    full = psi * rho * gac * dz                      # ← _integf(1.0, ...)
    half = float_type(0.5) * psi * rho * gac * dz    # ← _integf(0.5, ...)
    integrand = concat_where(KDim == first_idx[2], half,
                             concat_where(KDim == last_idx[2], half, full))
    return _column_sum_down(integrand)

# in ColumnIntegral_next / DomainIntegral_next (@ported_operator):
with managed_temporary(op.halo_exchanger_pool, (I, J, K)) as (ci_psi_3d,):
    _trapezoidal_z_next(op.indices, (I, J, K), psi=psi, rho=rho, gac=op.gac, dz=op.dz,
                        ci_psi=ci_psi_3d, subdomain="covering")
    ci_psi["covering"] = ci_psi_3d["covering"][:, :, 0]   # backward sweep: total on the lowest level
```

Notes: the cartesian `=` (not `+=`) at the top interval means `init=0.0` needs no pre-zeroing.
The sum is order-independent mathematically, but keep `forward=False` to reproduce cartesian's
summation order (floating-point addition is not associative). The ×0.5 on the lateral
boundaries (`_trapezoidal_xy_wgt`) is a pure 2D field_op (`ci_psi * 0.5`) called on the
`(I, J)` index space, which has `prescribed_boundary` subdomains of its own. Validated:
`tests/benchmarks/test_integral.py` (`TestColumnIntegral`, `TestDomainIntegral`).

And for `operators/cloud_cover.py:203` `_cloud_cover_maximum` (pattern 6, masked by a height
band; the 2D in-place `cc` is the carry; not yet ported):

```python
@scan_operator(axis=KDim, forward=True, init=float_type(0.0))
def _cc_max_scan(cc_km1: float_type,
                 cf: float_type, zcr: float_type, zorog: float_type,
                 lower_bound: float_type, upper_bound: float_type) -> float_type:
    z_ag = zcr - zorog
    if z_ag >= lower_bound and z_ag < upper_bound:
        return maximum(cc_km1, cf)      # ← interval(1,None): cc = max(cc, cf), in band only
    return cc_km1                       # out of band: keep the accumulated value

# field_op: return _cc_max_scan(cf, zcr, broadcast(zorog, ...), lb, ub)   (scalar carry: fine)
# operator: cc["covering"] = tmp["covering"][:, :, -1]                     (forward: last level)
```

Notes: the whole `interval(0, 1)` block (cc = cf in band, else 0) collapses into `init=0.0` —
since `cf ≥ 0`, `max(0, cf) = cf`, so no counter is needed at all. `zorog` (2D) is broadcast
in the wrapper. `_cloud_cover_maximum_random` (:225, pattern 7) has the same shape with the
random-overlap accumulation as body; its reads of `cf`/`zcr` at `[0,0,-1]` only occur for
`k >= 1`, so they are **carried** (previous level's values), not wrapper shifts.

And for `operators/cloud_cover.py:259` `_cloud_base_height` (pattern 8, first-hit search —
cartesian gates on `cloud_base_height == fill_value` to mean "not found yet"; not yet ported).
A scan `init` must be compile-time, but `fill_value` is a runtime parameter — so carry an
explicit `(found, height)` pair instead of the sentinel:

```python
@scan_operator(axis=KDim, forward=True, init=(False, 0.0))
def _cbh_scan(carry: tuple[bool, float_type],
              cf: float_type, rliquid: float_type, rice: float_type,
              zcr: float_type, zorog: float_type,
              cf_threshold: float_type, cct: float_type) -> tuple[bool, float_type]:
    found, height = carry
    if (not found) and cf >= cf_threshold and (rliquid + rice) >= cct:
        return True, zcr - zorog            # ← first cloudy level wins
    return found, height                    # already found, or not cloudy: pass through

# field_op: return the whole tuple (two 3D outputs), or one element through `* float_type(1.0)`
#           — gtfn wants the scan as the whole closure expression (see lessons)
# operator: cloud_base_height["covering"] = where(found["covering"][:, :, -1],
#                                                 h["covering"][:, :, -1], fill_value)   # host side
```

Notes: the `interval(0, 1)` block again collapses into the init — at k=0 the carry is
`(False, 0.0)`, so the uniform body reproduces the "set or fill_value" start. The bool-flag
carry also removes the fragile float `== fill_value` comparison; `fill_value` is applied in a
2D `where` on the host after slicing the last level. A `bool` in the carry is untested on
gtfn — a `float_type` 0/1 flag is the fallback. `_cloud_top_height/_temperature` (:290/:319,
pattern 9) are the same scan **without** the `not found` gate — the last qualifying level
simply overwrites.

**C. Flux propagation with several carried fields (patterns 10–12)** — one scan, carry = tuple
of everything level-coupled (not yet ported; the sketch applies the validated rules):

Concretely for pattern 10, the no-scattering LW pair `calc_radiative_fluxes_lw.py:419/449`
(helpers `f_calc_fluxes_no_scattering_lw_down/_up/_toa/_surf`; levels are numbered
ground → space, so BACKWARD = physically downward from TOA):

```python
# The cartesian [0,0,1] reads of transmittance/source_dn happen for k < top only; as wrapper
# shifts trans(KDim + 1) they would be out of the domain at the top → carry them instead
# (they are the values of the level processed just before). init = 0 reproduces f_…_toa.
@scan_operator(axis=KDim, forward=False, init=(float_type(0.0), float_type(0.0), float_type(0.0)))
def _lw_flux_dn(carry: tuple[float_type, float_type, float_type],
                trans: float_type, source_dn: float_type) -> tuple[float_type, float_type, float_type]:
    flux_dn_kp1, trans_kp1, source_dn_kp1 = carry
    flux_dn = trans_kp1 * flux_dn_kp1 + source_dn_kp1   # ← flux_dn = T[k+1]·flux_dn[k+1] + S[k+1]
    return flux_dn, trans, source_dn

@scan_operator(axis=KDim, forward=True, init=(int32(0), float_type(0.0)))
def _lw_flux_up(carry: tuple[int32, float_type],
                trans: float_type, source_up: float_type,
                emission_surf: float_type, albedo_surf: float_type,
                flux_dn: float_type) -> tuple[int32, float_type]:
    k, flux_up_km1 = carry
    if k == int32(0):                                    # ← f_…_surf: reflection + emission
        flux_up = emission_surf + albedo_surf * flux_dn
    else:
        flux_up = trans * flux_up_km1 + source_up        # ← flux_up = T[k]·flux_up[k−1] + S[k]
    return k + int32(1), flux_up
```

Notes: the TOA row (`interval(-1, None)`: `flux_dn = 0`) collapses into the init; the surface
row needs a counter branch because it reads other fields (`flux_dn` from the first scan,
surface albedo/emission broadcast from 2D) — and since both scans then return tuples, the
wrapper returns `flux_up * float_type(1.0)` (closure rule, see lessons). These fields carry a
g-point data dimension (`for n_gpoint in range(N_GPOINTS_LW)`): make it a proper dimension in
next and the K-scan applies pointwise across it — one scalar column per `(i, j, g-point)`.
The aerosol-scattering variant (`f_adding_ica_*`, lw:395/419) is the same shape with carry
`(albedo, source)` resp. `(flux_dn, flux_up)`. For pattern 12, the carry grows to everything
level-coupled:

```python
@scan_operator(axis=KDim, forward=False, init=(0.0, ..., 0.0))
def cloudsc_main(carry, ...):
    fallsink_qi_kp1, qin_kp1, ..., covptot, ... = carry   # per-level + accumulated 2D state
    ...
    return (fallsink_qi, qin, ..., covptot_new, ...)
```

For cloudsc, the carry is large: `fallsink_q{i,l,r,s,v}`, `q*n`, `rdtgdp` from the level above
plus the 2D state `covptot/covpclr/covpmax/cldtopdist/rainfrac_toprfz/rainliq` (the final 2D
values are host-side slices of the last processed level). Keep the scan minimal: the per-level
setup blocks and the FSD/output blocks are field_ops before/after it. Radiation starts keyed on
the data-dependent `i_cloud_top` (lw:664/718) fold into the scan as a branch on
`k == i_cloud_top - first_idx[2]` (counter vs. broadcast 2D index) instead of a separate pass.

**Reality check:** patterns 1–12 are expressible as scans by construction (first-order
recurrences); only pattern 13 is not. But expressible ≠ smooth on the pinned toolchain:
(a) the radiation g-point data dimension is the real risk — `pyproject.toml` already pins a
gt4py *fork* (`loops-over-data-dims`) for the cartesian radiation stencils, mainline next has
neither data-dim loops nor dense-dimension reductions (needed for the broadband sums, lw:467),
and O(100) g-points rule out tuple unrolling; (b) tuple carries, `if/elif` bodies, `int32`
counters and scans feeding scans are **validated** on gtfn with the Cartesian grid type (the
three ports below); untested so far: bools in carries and the DaCe backend; (c) cloudsc as one
scan means fully unrolling the species loops in a ~2000-line body — possible, but a major
effort; (d) cartesian fused each stencil into one kernel — the ports add a field_op pass
before/after each scan and a host-side slice for 2D results; whether that re-fuses depends on
the optimizer. The `precompilation` TODOs ("when scans are available") reflected these
practical gaps, not a missing scan concept; three of them are resolved now.

**Lessons from the first three ports** (`_solve_itcs`, helmholtz `_tridiagonal`, `_trapezoidal_z`
on the `precompilation` branch, gt4py fork `pmap_staging6`, gtfn backend, Cartesian grid type):
- **gtfn accepts a scan only as the *whole* expression of a closure.** Returning one element
  of a tuple-valued scan (`_, rp, _ = scan(...); return rp`) fails with "Scans are only
  supported at the top level of a stencil closure". A scalar-carry scan is fine as the
  return value; with a tuple carry (any counter or two-level recurrence) give the scan its own
  closure by returning through a trivial, exact pointwise op (`return rp * float_type(1.0)`),
  or feed it into the next scan/field_op (scan → scan works).
- **No level index inside a scan:** carry an `int32` counter (0 at the first level of the
  compute domain) and compare it with `k_top = last_idx[2] - first_idx[2]`, both injected by
  `porting.unify(add_first_idx=True, add_last_idx=True)`. **Casts only on literals:**
  `float_type(last_idx[2] - first_idx[2])` is rejected, so keep the counter `int32`
  (`k + int32(1)`, `k == int32(0)`).
- **No shifts inside a scan, and wrapper shifts hit the whole column:** a cartesian read
  like `b33[0,0,-1]` (or radiation's `transmittance[0,0,1]`) that only occurs for some
  levels becomes an out-of-domain access when written as `b33(KDim - 1)` in the wrapping
  field_op. Carry the previous level's value in the scan instead, and compute anything the
  *other* sweep needs from levels it has not visited (the top row of a back-substitution) in
  the sweep that has them, passing it along as a field. Half-level fields are the exception:
  `bbz(KDim ∓ 0.5)` exists for every cell.
- **2D results need an operator-level port** (`@ported_operator`): a scan yields a 3D field
  and no field_op can drop the K dimension. Run the scan into a 3D temporary and copy the
  last-processed level on the host (`ci_psi["covering"] = tmp["covering"][:, :, 0]`), or
  equivalently index the next field at that level — `tmp.local_field.next_field[KDim(k)]` is a
  zero-copy 2D view with the right domain — and copy it with a 2D field_op. A 2D *update*
  that only reads one known level of a 3D field (`_solve_itcs`'s `sflux` from `psi` at k=0)
  needs no copy at all: pass that view as an input of a pure 2D field_op. Put per-level
  weights/masks outside the scan (`concat_where` on `first_idx`/`last_idx`) so the carry
  stays scalar where possible.
- Scan arguments must all have the K dimension (`broadcast` 2D fields); bool flags arrive as
  `float_type` 0/1 through `porting.unify` and are used arithmetically as in cartesian.
- Keep the cartesian operation order in the body; then the variants agree to ~1e-13, the
  rest being FMA contraction differences between the two code generators — compare with
  rtol ~1e-10, not bitwise.
- Toolchain: `GT4PY_BUILD_JOBS=0` (serial in-process gtfn builds — the spawned worker pool
  dies on the laptop setup) and `PMAP_NUM_THREADS_COMPILATION=1`; unbounded parallel C++
  compilation crashed a 16 GB machine.

**General rules (as validated by the ports):** scans are scalar-in/scalar-out per level
(fields become scalars inside; every argument must carry the K dimension → `broadcast` 2D
fields in the wrapper); `interval` bands become `if`s on an `int32` level counter in the carry
(no level index exists inside a scan; bounds from `porting.unify`'s `first_idx`/`last_idx`);
only the recurrence belongs in the scan — pointwise pre/post-processing (rhs assembly, weights,
g-point sums) stays in field_ops so the toolchain can parallelize/fuse them; a 2D result of a
scan (`ci_psi`, cloud base/top) is taken from the 3D temporary at one level at operator level
(host-side slice or `next_field[KDim(k)]` view + 2D field_op); a 2D update that reads one
level of a 3D input (`sflux`) takes that view directly.

---

## External patterns: GridTools/physics_patterns PRs #1 and #2

Cross-check against the two open PRs in [GridTools/physics_patterns](https://github.com/GridTools/physics_patterns/pulls).

**PR #1 — Météo-France tropopause inversion + mixing length (`condensation.F90`):** both loops
match the catalog.
- Tropopause search (minimum temperature → `ZTMIN(ij)`, `ITPL(ij)`): pattern **6** as argmin —
  carry = `(t_min, k_min)`, identical in structure to `i_cloud_top`.
- Mixing length `ZL(k) = 0.6·ZL(k−1)` above `0.9·z(ITPL)`: conditional pattern **10** (with
  `b = 0`). The gate needs `PZZ` *at the data-dependent level* `ITPL(ij)` — avoid a K-gather by
  widening the argmin carry to `(t_min, k_min, pzz_at_min)` (same device as pattern 11).

**PR #2 — Noah LSM + SHiELD microphysics:**
- Noah LSM root-zone sum (`lsm.py`): pattern **5** with a *per-column* loop extent `nroot(ij)` →
  masked accumulation `where(k < nroot, acc + gx·w, acc)`; `zsoil[nroot]` is again
  value-at-data-dependent-index, precomputed by the same gated carry. Known variant.
- SHiELD melting loop (`microphysics.py`): **NEW — pattern 13, not expressible as a scan.**
  A double k-loop: for each ice-bearing level k (bottom-up), an inner loop walks down through
  levels m below, melting ice from k into rain at m and updating `qice/qrain/temperature/cvm`
  at **both k and m**, with data-dependent `break`s. A `scan_operator` is a first-order
  recurrence that writes only the current level — O(K²) traversal with multi-level writes does
  not fit. Port options: (a) reformulate in flux form — a downward scan carrying the pending
  melt flux and depositing per level (changes the operation order, so results differ at
  round-off to answer-changing level — needs validation); (b) keep as sequential column code
  outside gt4py.next (host fallback / embedded function). Nothing in PMAP currently has this
  shape — the cloudsc loops (pattern 12) stay first-order because melt/precip interact only
  through the carried flux.

## Appendix: "fake scans" — written as recurrences, rewritable as field_operators

**a) Broadcast of a single value** — the carry just copies one value through the column.
- `cloud_microphysics/diagnostics.py:21` (`eta_levels`): `surface = surface[0,0,1]` only propagates
  `in_aph[surface]` up → `out_eta[k] = in_ap[k] / aph_surf`. Slice the surface level to 2D outside
  (or via domain restriction) and do one pointwise field_op:

  ```python
  # cartesian:                              # next:
  with computation(BACKWARD):               @field_operator
      with interval(-1, None):              def eta_levels(in_ap: IJKField,
          surface = in_aph[0, 0, 1]                        aph_surf: IJField) -> IJKField:
          out_eta = in_ap / surface             return in_ap / broadcast(aph_surf,
      with interval(0, -1):                                             (IDim, JDim, KDim))
          surface = surface[0, 0, 1]        # aph_surf: host-side aph["covering"][:, :, -1]
          out_eta = in_ap / surface
  ```
- `diagnostics.py:45` (`tropopause`, BWD half): broadcasts the final value of the FWD pass from the
  last level down. The FWD half stays a scan; then slice its last level to 2D and broadcast as above.
- `_dev/cloudsc2.py:401`: same `trpaus` broadcast, same fix.

**b) The carry equals an existing field** — `radiation/…/calc_heating_rates.py:326`.
`tmp_sw_above` is initialised to `updated_flux_sw_net` at the top and set to
`updated_flux_sw_net[0,0,-1]` after each level, so `tmp_sw_above == updated_flux_sw_net[k]` always:

```python
# cartesian (carried 2D tmp_sw_above):     # next (pure offset read, no carry):
with computation(BACKWARD):                @field_operator
    with interval(1, None):                def heating_rate_sw(flux: IJKField,
        sw_below = flux[0, 0, -1]                              f2hr: IJKField) -> IJKField:
        out_hr_sw = f2hr * (                   return concat_where(
            sw_below - tmp_sw_above)               KDim > 0,
        tmp_sw_above = sw_below                    f2hr * (flux(KDim - 1) - flux),
                                                   float_type(0.0))
```

`tmp_sw_above == updated_flux_sw_net[k]` at every level, so it is a pure `KDim - 1` read.
Same for the lw/tot/clear variants; the `tmp_*_above` temporaries vanish.

**c) Self-written field read across *disjoint* intervals, no recurrence** — split in two field_ops:
compute the interior first, then patch the edges from the shifted result via `concat_where`.
- `ppm/common/cell_face_values.py:397,400`: edge levels copy `psi_face_z[0,0,±1]` written by the
  interior bands.

  ```python
  # step 1: interior, one formula per band       # step 2: fill the edges from the interior
  face = concat_where(KDim == 1, f_k1(psi),      face = concat_where(KDim == 0, face(KDim + 1),
         concat_where(KDim == 2, f_k2(psi),             concat_where(KDim == nz - 1,
                      f_interior(psi)))                              face(KDim - 1), face))
  ```
- `ecrad_cy49r1_driver.py:783` (`setup_data`): BWD block only patches boundary levels using interior
  `t_hl[±1]` written by the preceding PARALLEL block → same two-step pattern.

**d) FORWARD/BACKWARD only because cartesian needs an order** — body reads *inputs* at `[0,0,±1]`,
never its own output → plain shifts (+ `concat_where` at K boundaries).

```python
# cartesian:                               # next:
with computation(FORWARD):                 sed = concat_where(
    with interval(0, -1):                      KDim < nz - 1,
        sed = g(rhod[0,0,1], rr[0,0,1],        g(rhod(KDim + 1), rr(KDim + 1), rhod, rr),
                rhod, rr)                      g_bottom(rhod, rr))
    with interval(-1, None):
        sed = g_bottom(rhod, rr)
```
- `kessler/kessler.py:128`: sedimentation flux reads `rhod/rr/zcr` (inputs) at `±1` (the sketch above).
- `_dev/cloudsc2.py:488` (`cloudsc2_fluxes`): reads input fluxes `in_rfl/in_sfl[0,0,1]`.
- `cell_perturbation_method.py:453` (`scaling_richardson_number`): per-band one-sided/centred
  differences → nested `concat_where`.
- cloudsc setup blocks (`cloudsc_cy49r1.py:208` etc.): read only `in_aph[0,0,-1]` (input); the BWD
  order just sequences the 2D init before the main loop — irrelevant once split into separate
  field_ops feeding the scan.

**e) `interval(0, 1)` single-level writes** (albedo, surface_layer, operators/basic, statistics, …):
2D computations where FWD is a no-op ordering → 2D field_ops over `(IDim, JDim)` only, as already
done on the `precompilation` branch (`operators/basic.py` there). If the output must stay 3D,
update in place with `concat_where(KDim == 0, new_value, out)`.

```python
# cartesian:                               # next (pure 2D):
with computation(FORWARD),                 @field_operator
     interval(0, 1):                       def buoyancy_flux(shf: IJField,
    bf = g * (shf + 0.61 * t * lhf)                          lhf: IJField,
                                                             t_sfc: IJField) -> IJField:
                                               return g * (shf + 0.61 * t_sfc * lhf)
```

## Other porting hazards (not scans, found during the sweep)

- `radiation/…/_stencils/functions_aerosols.py:18` (`f_calc_rh_index`): a data-dependent `while`
  loop over relative-humidity bins inside a gtscript function. gt4py.next field_operators support
  neither `while` nor indexed access into lookup tables (`rh_lower.A[irh]`) — needs an unrolled
  search over the fixed `N_RH` bins, or precomputing the index field outside.
- Several radiation scans branch on `klevel` / `i_cloud_top` (absolute K positions). There is
  no level index inside a `scan_operator`: carry an `int32` counter (0 at the first level of
  the compute domain) and compare it with `i_cloud_top - first_idx[2]` (`i_cloud_top` as a
  broadcast 2D field; `first_idx`/`last_idx` are injected by `porting.unify`).
- Higher-dimensional data axes (g-points/bands as data dims, e.g. `albedo.py`, flux stencils):
  next has no data-dimension loops in field_ops — these become extra named dimensions or unrolled
  tuples; decide per stencil before porting.

## Suspicious code found while classifying (check before porting)

- `operators/nabla.py:489/494` — intervals `(1,2)` and `(1,-2)` overlap at k=1; the boundary
  coefficient never takes effect (second interval probably meant `(2,-2)`).
- `diagnostics/boundary_layer_height.py` — `h_ag[0,0,-1]` read at k=1 but never set at k=0.
- `calc_heating_rates.py:218` — reads `cos_sza_no_average` at k=0, but it is only written at
  `interval(-1, None)`.
- `_dev/_stencils/kessler_sedimentation.py` — FORWARD loop reads temporaries `rho_s`/`vt` at
  `[0,0,+1]` before they are computed; likely should be BACKWARD.
- `_dev/_stencils/cloudsc2.py` — the two `f_core` calls pass inconsistent argument lists
  (`in_tnd_cml_t` only in one).
