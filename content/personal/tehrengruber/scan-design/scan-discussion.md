## 2026.10.30 T, H, E, S

TODOs:

1. Let's find an example of an inclusive scan where the init value is a good concept (on any level).
```
@gtx.scan_operator(axis=dims.KDim, forward=True, init=(wpfloat("0.0"), wpfloat("0.0")))
def _solve_tridiagonal_matrix_forward_sweep(
    state_kminus1: tuple[wpfloat, wpfloat],
    a: wpfloat,
    b: wpfloat,
    c: wpfloat,
    d: wpfloat,
) -> tuple[wpfloat, wpfloat]:
    q_kminus1, d_prime_kminus1 = state_kminus1
    normalization = wpfloat("1.0") / (b + a * q_kminus1)
    return (wpfloat("0.0") - c) * normalization, (d - a * d_prime_kminus1) * normalization
```
Look if there is (by research elsewhere) a deductive answer to the question if the init value in the inclusive scan can be removed without losing expressiveness.

Answer (2026.11.17): the literature has two kinds of scan.

- Associative scans (prefix sums): inclusive needs no init (Haskell `scanl1`,
  `np.cumsum`, C++17 `inclusive_scan` without `init`); exclusive needs the
  identity (Blelloch; Futhark `scan` requires a neutral element). C++17
  `inclusive_scan` with `init` folds it into every element, so by
  associativity it adds nothing. Removing the init loses nothing here.
- General recurrences with carry type ≠ element type (our bodies): `init` is
  mandatory everywhere (`mapAccumL`, `jax.lax.scan`, a Fortran loop); no
  init-free variant exists. The only init-free scan, `scanl1`, needs carry
  type = element type and is defined as
  `scanl1 f xs = scanl f (head xs) (tail xs)`: seed from the first input,
  body from the second level, i.e. the exclusive scan with the init read from
  the inputs (scratchpad below).

Hence for our bodies the init value of the inclusive scan can be removed
without loss iff a first-level rule replaces it: `out[0] = g(x[K(0)])` (the
init of the exclusive scan as an expression of the inputs), since
`g(x0) = f(s, x0)` recovers any seed `s`. Without such a rule the tridiagonal
sweep is not expressible.

Examples of an inclusive scan where the init value is a good concept, i.e.
the body gives the correct first level when it reads the init as the carry:

- The sweep above: with `init=(0, 0)` the body at level 0 computes
  `(-c / b, d / b)`, the correct first row. Likewise composition.2 (backward)
  with `init=0.0`: `e * 0 + f = f` is the correct last row.
- C2 hydrostatic pressure: `init=p_surface` is the pressure below level 0,
  and the body at level 0 integrates from it.
- Reductions B1–B4: `init` is the identity of the reduction (`0.0` for a sum,
  `(np.inf, K(0))` for argmin, `False` for first hit). Nothing is written per
  level, so the exclusive reading "init is the first output level" does not
  apply at all.

Example where the init value is not a good concept: a seeded boundary row
`out[0] = seed` (two-sided BC). No init makes the body `0.5 * out[-1] + x[0]`
return `seed`, so the inclusive scan needs an `if` on the level instead.

Sources: [cppreference inclusive_scan](https://en.cppreference.com/w/cpp/algorithm/inclusive_scan),
[Data.List](https://hackage-content.haskell.org/package/base-4.22.0.0/docs/Data-List.html),
[jax.lax.scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html),
[Futhark scan](https://futhark-lang.org/examples/scan-reduce.html),
[GPU Gems 3 ch. 39](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html).

2. Let's take a look at how the patterns we target here are expressed in Jax, Torch, maybe even Fortran and how do they compare.

Answer for Fortran (2026.11.20), from [ICON](https://gitlab.dkrz.de/icon/icon-model)
at `7de8533`. JAX and Torch skipped for now.

The universal Fortran form is a sequential `DO jk` loop (`!$ACC LOOP SEQ`)
around a vectorized `DO jc` loop (`!$ACC LOOP GANG VECTOR`): k-sequential,
horizontally parallel, i.e. the `scan_operator` model. Per column:

| Pattern | Fortran form | Where (file:line) |
| --- | --- | --- |
| A1 recurrence | previous level read *from the output array itself*, no carry variable: `w(jc,jk) = w(jc,jk) + w(jc,jk+1)*z_q(jc,jk)`; `pres_ifc(jc,jk) = pres_ifc(jc,jk+1)*EXP(...)` | `mo_solve_nonhydro.f90:2412`, `mo_nh_diagnose_pres_temp.f90:569` |
| A1 with non-output state | 1D carry arrays over the horizontal, updated at the end of each level for the next: `zprvr(iv)`, `zvzr(iv)` (precipitation flux, fall speed) | `gscp_graupel.f90:1383-1390, 1406-1411` |
| A1.4 lookahead | by index, guarded by the end anchor: `qr(iv,k+1)` under `IF (k /= ke)` | `gscp_graupel.f90:1383, 1406` |
| A2 latch | logical arrays over `jc` guarding the whole body: `IF (lbelow_zmax(jc) .AND. .NOT. lfound_inversion(jc))`; `lfound(i)` set in one pass, consumed and reset in a second | `mo_util_phys.f90:917`, `mo_opt_nwp_diagnostics.f90:271-282` |
| B1 reduction | accumulate in place into the 2D result, zeroed before the loop (inclusive, identity init): `tot_cld_vi(jc,jb,iqv) = tot_cld_vi(...) + z_help*tot_cld(jc,jk,jb,iqv)` | `mo_nwp_diagnosis.f90:1171-1191` |
| B2/B3 arg-/first-hit | latch + stored index: `lfound_inversion(jc) = .true.; k_inversion(jc) = jk+1`; snow line interpolated at the hit. No `EXIT` inside a vectorized column loop; `EXIT` only in loops without `jc` | `mo_util_phys.f90:923-924`, `mo_opt_nwp_diagnostics.f90:280-282`, `mo_nwp_diagnosis.f90:1147-1150` |
| C1 boundary rows, dycore | peeled: boundary levels written before the loop, loop over the interior: `z_q(jc,1) = 0`, `w(jc,1,jb) = 0` or `w_ubc`, `w(jc,nlevp1,jb) = w_concorr_c`, then `DO jk = 2, nlev` and `DO jk = nlev-1, 2, -1` | `mo_solve_nonhydro.f90:2274, 2286, 2297, 2310, 2386, 2407` |
| C1 boundary rows, physics | `IF` on the index inside the body: `IF (k == ke)`, `IF (k /= ke)`, `k > k_start .AND. k < ke` | `gscp_graupel.f90:813, 1100, 1383` |
| C2 field seed | computed from the inputs and *written as a level*: `pres_sfc(jc) = p0ref*EXP(...)`; `pres_ifc(jc,nlev+1) = pres_sfc(jc)` (the exclusive init as an input expression); zeroed carries for the inclusive case (`zpkr(iv) = 0`) | `mo_nh_diagnose_pres_temp.f90:545-549`, `gscp_graupel.f90:562-570` |
| C3 data-dependent start | a scalar in the loop bounds: `DO jk = kstart_moist, nlev`, `k_start` argument; a per-column `k_start(jc,jb)` only in serial column-wise code (`jc` outer, `jk` inner) | `mo_nwp_diagnosis.f90:1176`, `gscp_graupel.f90:629`, `mo_vertical_grid.f90:1953, 2038` |
| D1 window | hand-written shift register: `dthetadz(jc,1) = dthetadz(jc,2); dthetadz(jc,2) = dthetadz(jc,3)` (the `delay line`) | `mo_util_phys.f90:931-932` |
| E1 extra dimension | one accumulator per species written out (`tot_cld_vi(:,:,iqv/iqc/iqi)`); 2-moment sedimentation is one subroutine call per particle type, one pass each (no fusion) | `mo_nwp_diagnosis.f90:1185-1189`, `mo_2mom_mcrph_processes.f90:5094` |
| composition.1/2 Thomas | forward sweep writes `w` and the 2D helper `z_q` in one loop, back substitution a second loop reading `z_q`; `z_q` materialized | `mo_solve_nonhydro.f90:2386-2414` |
| X1 double loop | `sedi_icon_core`: inner `DO ... kk = kk-1 ... EXIT` over the levels above, exit on `MAXVAL` over the vector; `CALL finish(... "not supported on GPU")` | `mo_2mom_mcrph_processes.f90` (`sedi_icon_core`) |

Comparison with the design:

- The carry is the output: Fortran reads `w(jc,jk+1,jb)` from the array it
  writes. `KView` carry `out[1]` is the direct transcription; explicit carry
  arrays appear only for non-output state (A2 tuple slots).
- Both boundary styles exist: the dycore peels rows and writes the boundary
  level before the loop (exclusive init, field expression), the physics
  branches on `k` inside the body (if-style). Where performance matters, rows
  are peeled.
- The level index is free: `k` serves conditions and lookahead; the end
  anchor is the scalar `ke`/`nlev` (`grid_info.nz`).
- Latches and first hit are `IF` guards around the whole body, same cost
  model as A2 today; no per-column early exit in vectorized loops.
- Loop bounds are explicit everywhere (`DO jk = 2, nlev`): the counterpart
  of `range=` at the call, never inferred.
- Delay lines and manual accumulators are workarounds in Fortran too (D1, B1).
- A per-column start level is used as a loop bound only in serial code; in
  vectorized code it is a scalar: supports "a mask, not a domain" (C3).

### Scratchpad
Exclusive scan using in the init what was previously an input value
```
@scan_pass
def scan_pass(out: KView[float], inp: KView[float]):
  return inp
  
@fo
def fo():
  scan_pass(inp, init=2*inp[K(0)])
```
## 2026.11.17 T, S

TODO: 
3. Scan design on GTIR: scan could look like concat_where, i.e., pass the ranges and the values for these ranges

Moving this from the scan-design.md, because it is not reviewed yet:

### What each variant needs

F = frontend syntax, T = type system, D = domain/extent analysis, L = lowering.

| Variant                      | Needs                                                                                                                                                                                                                                                                       |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| inclusive scan               | F: nothing (today's semantics). T: `init` has the carry type. L: nothing. A first level that differs from the body needs index access (index (a)/(b)).                                                                                                                        |
| exclusive scan               | F: field-valued `init`, K-literal slicing (`x[K(0)]`, `f[K(-1)]`, `x[K(0):K(2)]`), `m`-level init forms. T: `init` checked against the carry annotation, `m` levels of `T` per `KView[T, m]` slot (plain slots of a tuple carry keep the inclusive init, A2). D: the init occupies the first `m` levels of the range, the body runs on the rest; `range=`/`extend` when the output is longer than the inputs (A1.3). L: write the init levels, start the body at level `m`. |
| BC, init                     | The exclusive scan plus a runtime init (init (a)–(d)). D: `range=` for the body's sub-range. Two-sided: `concat_where` + `interior(Koff[-1])` outside (composition.1), or `end=` (A1.4).                                                                                   |
| BC, if                       | F: index access (index (a)/(b)); `if` on index conditions; `init` optional. T: index type `K`, end anchor. D: guard-refined extents (`out[-1]` under `level != K(0)`). L: the k-loop must be split into one branch-free loop per region of the index conditions (level 0, interior, last level), as GridTools intervals do; this must be guaranteed, not an optional optimization, else every interior level evaluates the boundary branches (A2 today). |
| init (a) nested def          | F: nested `def` in `field_operator`, closure capture. T: captured variables typed from the enclosing scope. L: hoisting of the nested scan.                                                                                                                                   |
| init (b) call-site `init=`   | F: `init=` keyword at the call, decorator init optional. T: checked against the carry annotation. L: nothing, `init` is already a scan argument.                                                                                                                               |
| init (c) string              | F: string -> implicit keyword parameter; expression parser if expressions are allowed. T: type from the carry annotation, no static checking. L: nothing.                                                                                                                       |
| init (d) `scan_pass` + `scan` | F: `scan_pass` decorator; `scan` builtin taking the body, its inputs, `init`, `axis`, `forward`, `range`. T: builtin typed from the body's signature (inputs, carry). L: nothing.                                                                                               |
| index (a) `index_var`        | F: decorator keyword, parameter removed from the call signature. T: index type `K`. L: as index (b), the parameter is bound to GTIR `index(K)`.                                                                                                                                 |
| index (b) builtin            | F: `index(dim)` in scan (and field operator) bodies. T: index type `K`. L: GTIR `index(dim)` (an integer field over `dim`), read at the current level inside the scan.                                                                                                         |
| output domain                | F: `range=`, `extend`, `named_range`, `x.domain[K]` slicing. D: call-site anchored: `range=` else the full K range of the K-bearing inputs, never inferred backward from the consumer; replaces `_extract_vertical_dims`/`closure_column_range`; data-dependent range rejected or lowered to a mask. |
| kview inputs                 | F: `KView[T]` annotation, literal-offset subscripts (`x[-1]`, `x[1]`), slices (`x[0:m]`). T: `KView[T]`, static-offset `__getitem__`, `x: T` sugar, K-less inputs as plain values, window slices over a local dimension. D: per-input extents, guard-refined, checked against the range (no implicit shrinking). L: k-caches (GTFN), pre-shifted leaves (JAX), shifted views (embedded); half offsets blocked on staggering. |
| carry (a) `KView[T, m]`      | F: `KView[T, m]` annotation with an integer parameter, `out[-k]` subscripts. T: `KView[T, m]`, `m` static, direction-dependent offsets, `m`-level init forms. D: init written as the first `m` levels. L: `m`-tuple desugar, flush k-caches, non-output slots not materialized. |
| carry (b) `history=m`        | F: decorator keyword. T, D, L: as carry (a), with one `m` for the whole carry.                                                                                                                                                                                                |
| carry (c) tuple carry        | nothing (today).                                                                                                                                                                                                                                                             |

`fold=True` (nothing written, K-less result) and `exit_condition` (early exit,
B3) are independent of the carry declaration and apply to plain carries too
(B2, B3).

Cheapest consistent set: `KView` inputs + carry (a) `KView[T, m]` (exclusive
scan), call-site `init=` (b), `index` builtin (b), `range=` at the call else
the full K range of the inputs, `concat_where` outside for the rest.

### Cross-cutting workarounds

Every item below is something production code carries *because* the current
construct lacks a feature — a checklist of things a redesign should make
disappear, independent of the catalog's patterns.

| Workaround                                                                       | Caused by                                         | Seen in                                                                                                             |
| -------------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `int32` level counter in the carry                                               | no level index inside the body                    | the PMAP ports [pmap/dycore/10], `_icon_graupel_scan` [icon4py/muphys/1]                                            |
| `first_level: bool` in the carry + branch at every level                         | no boundary-region dispatch                       | `test_icon_like_scan` [icon4py/dycore/1], `diagnose_pressure` [icon4py/diagnostics/1]                               |
| Runtime `last_lev` / `k_top` scalar smuggled in                                  | no end anchor                                     | `last_lev` [icon4py/muphys/5], the PMAP ports [pmap/dycore/10]                                                      |
| Delay-line carry slot for an input                                               | no neighbour-level read in the body               | `rho` [icon4py/muphys/3], `b33` [pmap/dycore/3]                                                                     |
| External pre-shift with a clamp (`concat_where(KDim < last_lev, t(Koff[1]), t)`) | same, for lookahead                               | `t_kp1` [icon4py/muphys/4]                                                                                          |
| Pass-through carry slot (carried only to be emitted)                             | carry *is* the output                             | `pflx_tot`, `t` [icon4py/muphys/1]                                                                                  |
| Dummy output field + program-level output slicing                                | same, inverted                                    | `test_icon_like_scan` [icon4py/dycore/1]                                                                            |
| 3D materialization + last-level slice for a column-constant                      | no vertical reduction, no K-removing output       | the ports of `_trapezoidal_z` [pmap/dycore/5] and cloud base/top [pmap/dycore/7], surface fluxes [icon4py/muphys/1] |
| `where`-blend instead of a runtime `if` (`enabled * new + (1 - enabled) * old`)  | runtime scalar conditions                         | `_update_sflux_next` [pmap/dycore/10]                                                                               |
| `*_scalar` field-operator twins                                                  | scalar element body                               | [icon4py/muphys/1]                                                                                                  |
| `sys.setrecursionlimit(350000)`, ~650 lines of repair hooks                      | 21–28-component carries, identity `else` branches | [icon4py/muphys/1]                                                                                                  |
| Handover field (`rp_top` emitted at every level, read at one)                    | a boundary branch cannot read inputs at an offset | the port of the stride-2 back-substitution [pmap/dycore/2]                                                          |
| `broadcast(x2d, (IDim, JDim, KDim))` for every 2D argument                       | all scan arguments must carry K                   | the PMAP ports [pmap/dycore/10]                                                                                     |
| Manual fusion into one mega-scan                                                 | no fusion guarantee across scans                  | [icon4py/muphys/1], cloudsc [pmap/microphysics/1]                                                                   |
