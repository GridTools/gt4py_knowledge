---
title: "icon4py concat_where usage survey"
author: havogt
tags: [concat_where, boundary-conditions, icon4py, dycore, diffusion, metrics, vertical, lateral-boundary, research]
created: 2026-06-16
---

> **Status**: research appendix to
> [[ideas/havogt/boundary-condition-syntax|Frontend syntax sugar for boundary conditions over concat_where]].
> A survey of real `concat_where` usage in [C2SM/icon4py](https://github.com/C2SM/icon4py),
> drafted with AI assistance via the GitHub code-search API + `raw.githubusercontent.com`. Code is
> verbatim unless marked `...`. Pinned to icon4py commit `515763c…` (default-branch HEAD at survey
> time) and gt4py `ffe777e…`; permalink base
> `https://github.com/C2SM/icon4py/blob/515763c3e42776c35a7751ce7173c412392dbec4/<path>`.

# `concat_where` in icon4py — the patterns the sugar must serve

## Scale and distribution

`concat_where` (imported everywhere as `from gt4py.next.experimental import concat_where`) appears
at **~40+ source call sites across ~13 files**, concentrated in the **dycore** (solve_nonhydro +
velocity advection), with secondary use in **diffusion**, **common/metrics**, **advection**, and
**muphys (graupel)**. icon4py defines pytest markers `uses_concat_where` / `infinite_concat_where`
and xfails such tests on the embedded backend ("Embedded backend does not support concat_where.").

| Subpackage | File | ≈ # `concat_where` |
| --- | --- | --- |
| dycore | `stencils/vertically_implicit_dycore_solver.py` | 9 |
| dycore | `stencils/compute_cell_diagnostics_for_dycore.py` | 7 |
| dycore | `stencils/compute_advection_in_vertical_momentum_equation.py` | 5 |
| dycore | `stencils/compute_edge_diagnostics_for_dycore_and_update_vn.py` | 4 (1 nested) |
| dycore | `stencils/compute_diagnostics_from_normal_wind.py` | 2 |
| dycore | `stencils/compute_horizontal_velocity_quantities.py` | 1 |
| dycore | `stencils/compute_advection_in_horizontal_momentum_equation.py` | 1 |
| diffusion | `stencils/apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence.py` | 3 |
| diffusion | `stencils/apply_diffusion_to_vn.py` | 2 (1 nested) |
| diffusion | `diffusion_utils.py` | 1 |
| common/metrics | `metric_fields.py` | 5 (incl. a 3-chain) |
| common/metrics | `compute_weight_factors.py` | 3 (chain) |
| advection | `stencils/compute_ppm_all_face_values.py` | 3 (chain) |
| advection | `stencils/compute_ppm_slope.py` | 1 |
| muphys | `…/muphys/implementations/graupel.py` | 1 |

## The two idioms (both verbose)

### Idiom A — chained reassignment to one variable (the dominant form)

A multi-region column is built by reassigning the same variable, region after region. This is the
idiom **Proposal 5 (domain-subscript painting) collapses 1:1** and **Proposal 4 (`cases`) flattens**.

`compute_weight_factors.py`, `_compute_wgtfac_c` — interior / top (`KDim==0`) / surface
(`KDim==nlev`):

```python
wgt_fac_c = concat_where(
    (0 < dims.KDim) & (dims.KDim < nlev),  # noqa: SIM300 [yoda-conditions]
    _compute_wgtfac_c_inner(z_ifc),
    z_ifc,
)
wgt_fac_c = concat_where(dims.KDim == 0,    _compute_wgtfac_c_0(z_ifc=z_ifc),    wgt_fac_c)
wgt_fac_c = concat_where(dims.KDim == nlev, _compute_wgtfac_c_nlev(z_ifc=z_ifc), wgt_fac_c)
```

`metric_fields.py`, `_compute_ddqz_z_half` — a 3-call chain refining interior / `==nlev`:

```python
ddqz_z_half = concat_where((dims.KDim > 0) & (dims.KDim < nlev), 0.0, 2.0 * (z_ifc - z_mc))
ddqz_z_half = concat_where(
    (0 < dims.KDim) & (dims.KDim < nlev),  # noqa: SIM300 [yoda-conditions]
    z_mc(Koff[-1]) - z_mc,
    ddqz_z_half,
)
ddqz_z_half = concat_where(dims.KDim == nlev, 2.0 * (z_mc(Koff[-1]) - z_ifc), ddqz_z_half)
```

`compute_ppm_all_face_values.py` — PPM advection start/end levels (`slevp1|elev` / `slev` /
`elevp1`):

```python
p_face = concat_where(
    (dims.KDim == slevp1) | (dims.KDim == elev),
    _compute_ppm_quadratic_face_values(p_cc, p_cellhgt_mc_now),
    p_face_in,
)
p_face = concat_where(dims.KDim == slev,    p_cc,            p_face)
p_face = concat_where(dims.KDim == elevp1,  p_cc(Koff[-1]),  p_face)
# TODO(dastrm): slev/elev and vertical_start/end are redundant
```

`compute_cell_diagnostics_for_dycore.py` repeats `concat_where(dims.KDim >= 1, interpolate(...),
prior_value)` for `rho`, `theta_v`, perturbed `theta_v`, buoyancy — "skip the model top (level 0)".
`vertically_implicit_dycore_solver.py` has long runs of consecutive `concat_where` each refining one
boundary of the column.

### Idiom B — nested `concat_where` (the else-branch is another `concat_where`)

These are textbook `if`/`elif`/`else` written inside-out — what **Proposals 1/2/3 collapse**.

`compute_edge_diagnostics_for_dycore_and_update_vn.py`, `apply_on_vertical_level` — a 3-way vertical
split:

```python
@gtx.field_operator
def apply_on_vertical_level(
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
    on_flatlevels: fa.EdgeKField[ta.wpfloat],
    between_flat_and_flatgradp: fa.EdgeKField[ta.wpfloat],
    below_flatgradp: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    return concat_where(
        dims.KDim < nflatlev,
        on_flatlevels,
        concat_where(nflat_gradp + 1 <= dims.KDim, below_flatgradp, between_flat_and_flatgradp),
    )
```

i.e. `if KDim < nflatlev: on_flatlevels elif KDim >= nflat_gradp+1: below_flatgradp else: between`.

`apply_diffusion_to_vn.py`, `_apply_diffusion_to_vn` — nested over the `limited_area` flag, **with
the explicit wish in a comment**:

```python
# TODO(): Use if-else statement instead
vn = (
    concat_where(
        dims.EdgeDim >= start_2nd_nudge_line_idx_e,
        _apply_nabla2_and_nabla4_to_vn(...),
        _apply_nabla2_to_vn_in_lateral_boundary(...),
    )
    if limited_area
    else concat_where(
        dims.EdgeDim >= start_2nd_nudge_line_idx_e,
        _apply_nabla2_and_nabla4_global_to_vn(...),
        vn,
    )
)
```

## Condition shapes observed

- **Single comparison**: `KDim < N` (`< nlev`, `< nlev-1`, `< nflatlev`, `< last_lev`), `KDim > 0`,
  `KDim >= 1`, `nflatlev <= KDim`, and equalities `KDim == 0` / `== nlev` / `== elev` / `== slev`.
- **Banded `&` (AND)**: `(0 < KDim) & (KDim < nlev)`; `(KDim > 0) & (KDim < end_index_of_damping_layer + 1)`
  (Rayleigh damping band); `(KDim >= maximum(2, end_index_of_damping_layer - 2)) & (KDim < nlev - 3)`.
- **`|` (OR) of equalities**: `(KDim == slevp1) | (KDim == elev)`.
- **Horizontal half-open ranges over index symbols**:
  `(start_edge_lateral_boundary_level_7 <= EdgeDim) & (EdgeDim < end_edge_halo)`;
  `(interior_idx <= CellDim) & (CellDim < halo_idx)`; `CellDim >= lateral_boundary_level_2`.
- **Mixed vertical+horizontal** in one condition:
  `(0 < KDim) & (KDim < nrdmax) & (interior_idx <= CellDim) & (CellDim < halo_idx)` (4 clauses).
- The `# noqa: SIM300 [yoda-conditions]` marker recurs on `(0 < dims.KDim)` — because the
  **dimension must be on the left** of the comparison to produce a `Domain`, half the conditions are
  written "backwards". A sugar that paints `KDim == 0` directly removes this papercut.

## Vertical vs horizontal

- **Vertical (≈90% of sites)** — model top (`KDim == 0`, `KDim >= 1`), surface/bottom (`KDim == nlev`,
  `KDim < nlev`, `KDim == nlev-1`, PPM `elev`/`elevp1`), half-/full-level interpolation transitions
  and terrain-following→flat transition (`nflatlev`/`nflat_gradp`), damping-layer bands
  (`end_index_of_damping_layer`/`nrdmax`/`nshift`), moisture start (`kstart_moist`).
- **Horizontal / lateral (smaller but important)** — limited-area lateral-boundary and nudging zones
  on edges (`start_edge_lateral_boundary`, `start_edge_lateral_boundary_level_7`, `end_edge_nudging`,
  `start_edge_nudging_level_2`, `start_2nd_nudge_line_idx_e`), interior-vs-halo on cells
  (`interior_idx`/`halo_idx`, `end_edge_halo`), lateral-boundary level on cells
  (`CellDim >= lateral_boundary_level_2`). These index symbols are the icon4py port of ICON's
  `rl_start`/`rl_end` + `refin_ctrl` zone scheme.

Names are self-documenting about intent: helper functions like
`_compute_contravariant_correction_of_w_for_lower_boundary`, `_apply_nabla2_to_vn_in_lateral_boundary`,
`_apply_nabla2_to_w_in_upper_damping_layer`, `_compute_wgtfac_c_0` / `_compute_wgtfac_c_nlev`
(top/surface).

## Direct evidence of the wish, and related gt4py issues

- **The if/else wish is in the source**: `# TODO(): Use if-else statement instead`
  (`apply_diffusion_to_vn.py`) — verbatim. There is **no** public GitHub issue/ADR proposing
  `if/elif/else` or `match` sugar in either repo; the desire is expressed only via this inline TODO
  and the prevalence of hand-written chained/nested `concat_where`. So
  [[ideas/havogt/boundary-condition-syntax|the proposal]] documents a gap the code clearly motivates
  but that has not been written up upstream.
- **`&` already becomes nested `concat_where` internally**:
  `src/gt4py/next/iterator/transforms/concat_where/canonicalize_domain_argument.py` rewrites
  `concat_where(d1 & d2, a, b) → concat_where(d1, concat_where(d2, a, b), b)`. So every banded
  `(A <= KDim) & (KDim < B)` already lowers to a nest — the sugar would just make the *surface* match
  the semantics.
- **`Dimension.__ne__(int)` raises** (gt4py `common.py`): "produces two disjoint domains, which is
  not supported. Use `concat_where(dim < value, …) concat_where(dim > value, …)`, see ADR 22" — the
  enforcement behind Idiom B's manual nesting.
- **Open gt4py bugs triggered by these patterns**:
  - [gt4py#2205](https://github.com/GridTools/gt4py/issues/2205) — `concat_where(KDim == nlev-1, …)`
    is broken (domain inference over-extends K by one level). Referenced verbatim by a
    `# TODO(havogt): …` in `compute_advection_in_vertical_momentum_equation.py`.
  - [gt4py#2265](https://github.com/GridTools/gt4py/issues/2265) — SDFG lowering fails when a
    `concat_where` branch starts/ends outside the target domain; triggered by two icon4py dycore
    stencil tests.
  These matter to [[ideas/havogt/boundary-condition-syntax|the proposal]]: nicer surface syntax must
  not paper over (and ideally should surface better errors for) the underlying domain-inference and
  end-anchor (`nlev-1`) limitations.

## Take-aways for the syntax design

1. The **chained-reassignment idiom dominates** real code → domain-subscript painting (Proposal 5)
   and the `cases(...)` builder (Proposal 4) hit the most code.
2. **Nested `concat_where` for 3-way splits** is common and reads inside-out → `if/elif/else`
   (Proposal 1) is the most natural collapse, and one icon4py TODO asks for exactly it.
3. Conditions are simple (`==`, `<`, `>=`, banded `&`, occasional `|`) and overwhelmingly **1-D
   over `KDim`**, so the sugar does not need to solve general multi-dim domains to be useful — it
   mainly needs to remove chaining/nesting noise and the Yoda-condition papercut.
4. Horizontal/lateral conditions use **named index symbols** (`start_edge_lateral_boundary`, …), so
   the sugar must accept arbitrary `Domain`-valued expressions, not just literal `KDim == 0`.
