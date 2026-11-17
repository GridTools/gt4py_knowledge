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

2. Let's take a look at how the pattern we target here are expressed in Jax, Torch, maybe even Fortran and how do they compare.


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
| BC, init                     | F: field-valued init; K-literal field slicing (`x[K(0)]`, `x[K(0):K(2)]`, `f[K(-1)]`); runtime init (one of the three below); `m`-level init. T: init vs carry annotation. D: init written as the first `m` levels; `range=`/`extend`. Two-sided: `concat_where` + `(Koff[-1])` outside, or `end=` (A1.4). |
| BC, if                       | F: index access (one of the two below); `if` on index conditions; `init` optional. T: index type `K`, end anchor. D: guard-refined extents (`out[-1]` under `level != K(0)`). L: the k-loop must be split into one branch-free loop per region of the index conditions (level 0, interior, last level), as GridTools intervals do; this must be guaranteed, not an optional optimization, else every interior level evaluates the boundary branches (A2 today). |
| init (a) nested def          | F: nested `def` in `field_operator`, closure capture. T: captured variables typed from the enclosing scope. L: hoisting; body not reusable.                                                                                                                                   |
| init (b) call-site `init=`   | F: `init=` keyword at the call, decorator init optional. T: checked against the carry annotation. L: nothing (init is already a scan argument).                                                                                                                                |
| init (c) string              | F: string -> implicit keyword parameter; expression parser if expressions allowed. T: type from the carry annotation. No static checking.                                                                                                                                       |
| index (a) `index_var`        | F: decorator keyword, parameter removed from the call signature. T: index type `K`.                                                                                                                                                                                           |
| index (b) builtin            | F: `index(dim)` in scan (and field operator) bodies. T: index type `K`. L: GTIR `index`.                                                                                                                                                                                       |
| output domain                | F: `range=`, `extend`, `named_range`, `x.domain[K]` slicing. D: forward, call-site anchored K inference replacing `_extract_vertical_dims`/`closure_column_range`; data-dependent range rejected or lowered to a mask.                                                            |
| kview inputs                 | T: `KView[T]`, static-offset `__getitem__`, `x: T` sugar, K-less inputs as plain values, window slices over a local dimension. D: per-input extents, guard-refined. L: k-caches (GTFN), pre-shifted leaves (JAX), shifted views (embedded); half offsets blocked on staggering. |
| kview carry                  | T: `KView[T, m]`, `m` static, direction-dependent offsets, `m`-level init forms. D: init written as the first `m` levels; `fold=True`. L: `m`-tuple desugar, flush k-caches, non-output slots not materialized, `exit_condition` early exit (B3).                                                |

Cheapest consistent set: `KView` inputs + carry, call-site `init=` (b),
`index` builtin (b), `range=` with forward inference, `concat_where` outside
for the rest.

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
