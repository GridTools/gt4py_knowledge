---
title: Redesign of the vertical scan in gt4py.next
author: sf-n, tehrengruber
tags:
  - scan
  - vertical
  - frontend
  - fusion
  - migration
  - workarounds
created: 2026-10-20
status: draft
---

> How the patterns of the
> [[personal/tehrengruber/scan-design/scan-patterns|pattern catalog]] can be
> expressed in gt4py. Reference labels are those of the catalog's
> [[personal/tehrengruber/scan-design/scan-patterns#references|references]].

## Representability with today's `scan_operator`

Each catalog example, classified: can today's **`gt4py.next`**
(`scan_operator` + field operators + program-level code — *not*
`gt4py.cartesian`) express it, with which workarounds, at what cost.

Verdicts:

| Verdict                        | Meaning                                                                 |
| ------------------------------ | ----------------------------------------------------------------------- |
| ✓                              | expressible cleanly                                                     |
| ✓ (fold)                       | clean only because the boundary row folds *algebraically* into the init |
| poor readability               | expressible; the workarounds hurt readability only                      |
| poor readability & performance | expressible; the workarounds hurt readability and cost performance      |
| poor performance               | expressible cleanly; the execution wastes work                           |
| ✗                              | not expressible                                                         |
Opinions:
not ok: We have to solve this, before writing code that way.
temporarily: Doesn't need to be solved right away.
ok: We can use the workaround for now and let users use it. 

Workaround / blocker labels (each is a row of the
[workarounds table](#cross-cutting-workarounds)):

Shared preamble (`Column`/`Mask` as in
[[personal/tehrengruber/scan-design/scan-patterns|the catalog]]):

```python
import numpy as np
import gt4py.next as gtx
from gt4py.next import broadcast, exp, where
from gt4py.next.experimental import concat_where

IDim = gtx.Dimension("I")
KDim = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
Koff = gtx.FieldOffset("Koff", source=KDim, target=(KDim,))
IField = gtx.Field[gtx.Dims[IDim], float]
IKField = gtx.Field[gtx.Dims[IDim, KDim], float]
IKBoolField = gtx.Field[gtx.Dims[IDim, KDim], bool]
```

#### `delay line`

delay-line carry slot(s) for an input read at $k - m$

```python
@gtx.scan_operator(axis=KDim, forward=True, init=(0.0, 0.0))
def _delayed(state: tuple[float, float], x: float) -> tuple[float, float]:
    out, x_km1 = state
    return out + x_km1, x


@gtx.field_operator
def delayed(x: IKField) -> IKField:
    out, _ = _delayed(x)
    return out
```

also: [`discarded carry`](#discarded-carry)

```python
def delayed(x: Column) -> Column:
    out = np.zeros_like(x)
    for k in range(1, x.size):
        out[k] = out[k - 1] + x[k - 1]
    return out
```

Opinion: Ok for now.
#### `pre-shift`

external shift + clamp (materializes a shifted copy)

```python
@gtx.scan_operator(axis=KDim, forward=True, init=0.0)
def _lookahead(state: float, x_kp1: float) -> float:
    return state + x_kp1


@gtx.field_operator
def lookahead(x: IKField, nlev: gtx.int32) -> IKField:
    return _lookahead(concat_where(KDim < nlev - 1, x(Koff[1]), x))
```

also: [`column length`](#column-length)

```python
def lookahead(x: Column) -> Column:
    out = np.empty_like(x)
    s = 0.0
    for k in range(x.size):
        s = s + x[min(k + 1, x.size - 1)]
        out[k] = s
    return out
```

Opinion: Syntax not ok, performance medium (measurement needed).

#### `level counter`

`int32` level counter in the carry

```python
@gtx.scan_operator(axis=KDim, forward=True, init=(0.0, 0))
def from_level(
    state: tuple[float, gtx.int32], x: float, k_start: gtx.int32
) -> tuple[float, gtx.int32]:
    s, k = state
    return s + x if k >= k_start else s, k + 1
```

Note: The index needs to be passed from the very outermost field operator through to where you need the index.

also: [`discarded carry`](#discarded-carry)

```python
def from_level(x: Column, k_start: int) -> Column:
    out = np.empty_like(x)
    s = 0.0
    for k in range(x.size):
        if k >= k_start:
            s = s + x[k]
        out[k] = s
    return out
```

Opinion: Syntax ok. Performance temporarily.
#### `column length`

runtime column length smuggled in (end anchor)

```python
@gtx.scan_operator(axis=KDim, forward=True, init=(0.0, 0))
def last_row(
    state: tuple[float, gtx.int32], x: float, nlev: gtx.int32
) -> tuple[float, gtx.int32]:
    s, k = state
    return 0.5 * s if k == nlev - 1 else s + x, k + 1
```

also: [`level counter`](#level-counter), [`discarded carry`](#discarded-carry)

```python
def last_row(x: Column) -> Column:
    out = np.empty_like(x)
    s = 0.0
    for k in range(x.size - 1):
        s = s + x[k]
        out[k] = s
    out[-1] = 0.5 * s
    return out
```

Opinion: Syntax & performance ok.
#### `seed input`

field-valued init via a level counter + the seed as an extra input

```python
@gtx.scan_operator(axis=KDim, forward=True, init=(0.0, 0))
def _seeded(state: tuple[float, gtx.int32], x: float, seed: float) -> tuple[float, gtx.int32]:
    out, k = state
    return seed if k == 0 else out + x, k + 1


@gtx.field_operator
def seeded(x: IKField, seed: IField) -> IKField:
    out, _ = _seeded(x, broadcast(seed, (IDim, KDim)))
    return out
```

also: [`level counter`](#level-counter), [`broadcast`](#broadcast), [`discarded carry`](#discarded-carry)

```python
def seeded(x: Column, seed: float) -> Column:
    out = np.empty_like(x)
    out[0] = seed
    for k in range(1, x.size):
        out[k] = out[k - 1] + x[k]
    return out
```

Opinion: Syntax not ok, performance not ok (measurement would be useful)
#### `broadcast`

2D argument broadcast to 3D

```python
@gtx.scan_operator(axis=KDim, forward=True, init=0.0)
def _scaled_sum(state: float, x: float, c: float) -> float:
    return state + c * x


@gtx.field_operator
def scaled_sum(x: IKField, c: IField) -> IKField:
    return _scaled_sum(x, broadcast(c, (IDim, KDim)))
```

```python
def scaled_sum(x: Column, c: float) -> Column:  # c: column-constant
    out = np.empty_like(x)
    s = 0.0
    for k in range(x.size):
        s = s + c * x[k]
        out[k] = s
    return out
```

Opinion: Syntax temporarily, performance unclear (needs measurement)
#### `last-level slice`

3D materialization + last-level slice outside the DSL (K removal)

```python
@gtx.scan_operator(axis=KDim, forward=True, init=0.0)
def _running_sum(state: float, x: float) -> float:
    return state + x


# outside the DSL, not even in a program
_running_sum(x, out=tmp, offset_provider={})
column_sum = tmp.ndarray[:, -1]
```

```python
def column_sum(x: Column) -> np.float64:
    return np.sum(x)
```

Opinion: Syntax not ok, performance not ok.

Note: This doesn't allow fusing beyond this pattern.
#### `pass-through carry`

pass-through carry slot (carry *is* the output)

```python
@gtx.scan_operator(axis=KDim, forward=True, init=(0.0, 0.0))
def doubled_sum(state: tuple[float, float], x: float) -> tuple[float, float]:
    s, _ = state
    return s + x, 2.0 * (s + x)
```

also: [`discarded carry`](#discarded-carry)

```python
def doubled_sum(x: Column) -> Column:
    tmp = np.empty_like(x)
    tmp[0] = x[0]
    for k in range(1, x.size):
        tmp[k] = tmp[k - 1] + x[k]
    return tmp * 2
```

Note: Performance-only workaround.
Opinion: Syntax not ok.
#### `discarded carry`

a non-output carry slot, discarded in the wrapping field operator —
implied by any non-output state: a [`delay line`](#delay-line), a
[`level counter`](#level-counter), a search latch, …

```python
@gtx.scan_operator(axis=KDim, forward=True, init=(0.0, False))
def _latched(state: tuple[float, bool], x: float, mask: bool) -> tuple[float, bool]:
    _, hit = state
    hit = hit | mask
    return x if hit else 0.0, hit


@gtx.field_operator
def latched(x: IKField, mask: IKBoolField) -> IKField:
    out, _ = _latched(x, mask)
    return out
```


```python
def latched(x: Column, mask: Mask) -> Column:
    out = np.empty_like(x)
    hit = False
    for k in range(x.size):
        hit = hit or bool(mask[k])
        out[k] = x[k] if hit else 0.0
    return out
```
Opinion: Syntax ok. Performance ok (as long as the discarded output is not materialized).
#### `compile-time unroll`

compile-time unrolling (window terms, table bins)

```python
@gtx.field_operator
def lookup(r: IKField, b0: float, b1: float, t0: float, t1: float, t2: float) -> IKField:
    return where(r <= b0, t0, where(r <= b1, t1, t2))
```

```python
def lookup(r: float, b0: float, b1: float, t0: float, t1: float, t2: float) -> float:
    bounds, table = [b0, b1, np.inf], [t0, t1, t2]
    i = 0
    while r > bounds[i]:
        i += 1
    return table[i]
```

Opinion: Syntax Temporarily. Performance ok.
#### `manual fusion`

performance only via manual fusion into one scan

```python
@gtx.scan_operator(axis=KDim, forward=True, init=(0.0, 0.0))
def two_sums(state: tuple[float, float], x: float, y: float) -> tuple[float, float]:
    sa, sb = state
    return sa + x, sb + y
```

```python
def two_sums(x: Column, y: Column) -> tuple[Column, Column]:
    a, b = np.empty_like(x), np.empty_like(y)
    sa, sb = 0.0, 0.0
    for k in range(x.size):
        sa = sa + x[k]
        a[k] = sa
    for k in range(y.size):
        sb = sb + y[k]
        b[k] = sb
    return a, b
```
Opinion: Syntax ok. Performance ok.
#### `outside the DSL`

a step escapes the DSL entirely (plain numpy)

```python
@gtx.field_operator
def _divide(x: IKField, total: IField) -> IKField:
    return x / broadcast(total, (IDim, KDim))


# outside the DSL, not even in a program
_running_sum(x, out=tmp, offset_provider={})
total = gtx.as_field((IDim,), tmp.ndarray[:, -1])
_divide(x, total, out=out, offset_provider={})
```

also: [`last-level slice`](#last-level-slice), [`broadcast`](#broadcast)

```python
def normalized(x: Column) -> Column:
    return x / np.sum(x)
```
Opinion: Temporarily (case dependent). Performance not ok.
#### `no: data dimension`

dense extra dimension — the scalar body only offers tuple unrolling, which explodes at $O(100)$

#### `no: scatter`

writes at other levels with feedback, $O(K^2)$

Classification:

| Ex  | Pattern                       | Verdict                        | Workarounds / blocker                                                                                                                                  |
| --- | ----------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | A1, order 2, coupled          | poor readability               | [`delay line`](#delay-line) ×2, [`level counter`](#level-counter)                                                                                      |
| 2   | A1 staggered ($n \to n+1$)    | poor readability               | [`level counter`](#level-counter), [`pass-through carry`](#pass-through-carry), [`outside the DSL`](#outside-the-dsl)                                  |
| 3   | A1 staggered, integer indices | poor readability               | [`level counter`](#level-counter), [`pass-through carry`](#pass-through-carry), [`outside the DSL`](#outside-the-dsl)                                  |
| 4   | A1 + neighbour inputs         | poor readability & performance | [`delay line`](#delay-line), [`pre-shift`](#pre-shift)                                                                                                 |
| 5   | A2 latch                      | poor performance               | — (both branches evaluated at every level: no work skipping)                                                                                           |
| 6   | B1 reduction                  | poor readability & performance | [`last-level slice`](#last-level-slice)                                                                                                                |
| 7   | B2 arg-reduction              | poor readability & performance | [`level counter`](#level-counter), [`pass-through carry`](#pass-through-carry), [`last-level slice`](#last-level-slice)                                |
| 8   | B3 first hit                  | poor readability & performance | [`last-level slice`](#last-level-slice)                                                                                                                |
| 9   | B4 masked depth               | poor readability & performance | [`level counter`](#level-counter), [`last-level slice`](#last-level-slice)                                                                             |
| 10  | C1 four regions               | poor readability               | [`level counter`](#level-counter), [`column length`](#column-length) (stride-2 back-substitution: + a handover field → poor readability & performance) |
| 11  | C2 field seed                 | poor readability & performance | [`seed input`](#seed-input), [`broadcast`](#broadcast)                                                                                                 |
| 12  | C3 data-dependent start       | poor readability & performance | [`level counter`](#level-counter), [`broadcast`](#broadcast)                                                                                           |
| 13  | D1 bounded window             | poor readability & performance | [`compile-time unroll`](#compile-time-unroll) ×M (field op) or [`delay line`](#delay-line) ×M                                                          |
| 14  | D2 window + state             | poor readability & performance | [`delay line`](#delay-line) ×M                                                                                                                         |
| 15  | E1 dense data dim             | ✗                              | [`no: data dimension`](#no-data-dimension)                                                                                                             |
| 16  | X1 melting double loop        | ✗                              | [`no: scatter`](#no-scatter)                                                                                                                           |
| 17  | X2 `while` over table         | poor readability & performance | [`compile-time unroll`](#compile-time-unroll) (all bins evaluated) or [`outside the DSL`](#outside-the-dsl)                                            |
| 18  | compound forward sweep        | poor readability & performance | [`seed input`](#seed-input), [`pre-shift`](#pre-shift), [`column length`](#column-length), [`level counter`](#level-counter)                           |
| 19  | back substitution             | ✓ (fold)                       | —                                                                                                                                                      |
| 20  | B1 → pointwise broadcast      | poor readability & performance | [`last-level slice`](#last-level-slice), [`outside the DSL`](#outside-the-dsl)                                                                         |
| 21  | independent recurrences × N   | poor readability & performance | [`manual fusion`](#manual-fusion), [`discarded carry`](#discarded-carry) ×13 (unfused: shared work recomputed, one pass per scan)                      |
| 22  | A1 → B1, one pass             | poor readability & performance | [`pass-through carry`](#pass-through-carry), [`last-level slice`](#last-level-slice), [`manual fusion`](#manual-fusion)                                |
| 23  | horizontal producer → scan    | ✓                              | — (unfused: the intermediate materializes)                                                                                                             |

## Cross-cutting workarounds

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

[pmap/dycore/2]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/helmholtz/preconditioner.py#L53
[pmap/dycore/3]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/helmholtz/preconditioner.py#L65
[pmap/dycore/5]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/operators/integral.py#L105
[pmap/dycore/7]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/operators/cloud_cover.py#L248
[pmap/dycore/10]: https://github.com/PMAP-Project/PMAP/tree/fbdad0b0fa4cc9fccd1cc180ea2460e4935f7d63
[pmap/microphysics/1]: https://github.com/PMAP-Project/PMAP/blob/3041d4686f463d877ba51009b4e449435c693683/src/pmap/physics/cloud_microphysics/cloudsc/cy49r1/_stencils/cloudsc_cy49r1.py#L380
[icon4py/muphys/1]: https://github.com/C2SM/icon4py/blob/515763c3e42776c35a7751ce7173c412392dbec4/model/atmosphere/subgrid_scale_physics/muphys/src/icon4py/model/atmosphere/subgrid_scale_physics/muphys/implementations/graupel.py#L175
[icon4py/muphys/3]: https://github.com/C2SM/icon4py/blob/515763c3e42776c35a7751ce7173c412392dbec4/model/atmosphere/subgrid_scale_physics/muphys/src/icon4py/model/atmosphere/subgrid_scale_physics/muphys/implementations/graupel.py#L79
[icon4py/muphys/4]: https://github.com/C2SM/icon4py/blob/515763c3e42776c35a7751ce7173c412392dbec4/model/atmosphere/subgrid_scale_physics/muphys/src/icon4py/model/atmosphere/subgrid_scale_physics/muphys/implementations/graupel.py#L511
[icon4py/muphys/5]: https://github.com/C2SM/icon4py/blob/515763c3e42776c35a7751ce7173c412392dbec4/model/atmosphere/subgrid_scale_physics/muphys/src/icon4py/model/atmosphere/subgrid_scale_physics/muphys/implementations/graupel.py#L488
[icon4py/dycore/1]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/multi_feature_tests/ffront_tests/test_icon_like_scan.py#L43
[icon4py/diagnostics/1]: https://github.com/C2SM/icon4py/blob/515763c3e42776c35a7751ce7173c412392dbec4/model/common/src/icon4py/model/common/diagnostic_calculations/stencils/diagnose_pressure.py#L15
