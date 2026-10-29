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
| poor performance               | expressible cleanly; the execution wastes work                          |
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

| Ex            | Pattern                       | Verdict                        | Workarounds / blocker                                                                                                                                  |
| ------------- | ----------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A1.1          | A1, order 2, coupled          | poor readability               | [`delay line`](#delay-line) ×2, [`level counter`](#level-counter)                                                                                      |
| A1.2          | A1 staggered ($n \to n+1$)    | poor readability               | [`level counter`](#level-counter), [`pass-through carry`](#pass-through-carry), [`outside the DSL`](#outside-the-dsl)                                  |
| A1.3          | A1 staggered, integer indices | poor readability               | [`level counter`](#level-counter), [`pass-through carry`](#pass-through-carry), [`outside the DSL`](#outside-the-dsl)                                  |
| A1.4          | A1 + neighbour inputs         | poor readability & performance | [`delay line`](#delay-line), [`pre-shift`](#pre-shift)                                                                                                 |
| A2            | A2 latch                      | poor performance               | — (both branches evaluated at every level: no work skipping)                                                                                           |
| B1            | B1 reduction                  | poor readability & performance | [`last-level slice`](#last-level-slice)                                                                                                                |
| B2            | B2 arg-reduction              | poor readability & performance | [`level counter`](#level-counter), [`pass-through carry`](#pass-through-carry), [`last-level slice`](#last-level-slice)                                |
| B3            | B3 first hit                  | poor readability & performance | [`last-level slice`](#last-level-slice)                                                                                                                |
| B4            | B4 masked depth               | poor readability & performance | [`level counter`](#level-counter), [`last-level slice`](#last-level-slice)                                                                             |
| C1            | C1 four regions               | poor readability               | [`level counter`](#level-counter), [`column length`](#column-length) (stride-2 back-substitution: + a handover field → poor readability & performance) |
| C2            | C2 field seed                 | poor readability & performance | [`seed input`](#seed-input), [`broadcast`](#broadcast)                                                                                                 |
| C3            | C3 data-dependent start       | poor readability & performance | [`level counter`](#level-counter), [`broadcast`](#broadcast)                                                                                           |
| D1            | D1 window + state             | poor readability & performance | [`delay line`](#delay-line) ×M                                                                                                                          |
| E1            | E1 dense data dim             | ✗                              | [`no: data dimension`](#no-data-dimension)                                                                                                             |
| composition.1 | tridiagonal forward sweep     | poor readability & performance | [`seed input`](#seed-input), [`pre-shift`](#pre-shift), [`column length`](#column-length), [`level counter`](#level-counter)                           |
| composition.2 | back substitution             | ✓ (fold)                       | —                                                                                                                                                      |
| composition.3 | B1 → pointwise broadcast      | poor readability & performance | [`last-level slice`](#last-level-slice), [`outside the DSL`](#outside-the-dsl)                                                                         |
| composition.4 | independent recurrences × N   | poor readability & performance | [`manual fusion`](#manual-fusion), [`discarded carry`](#discarded-carry) ×13 (unfused: shared work recomputed, one pass per scan)                      |
| composition.5 | A1 → B1, one pass             | poor readability & performance | [`pass-through carry`](#pass-through-carry), [`last-level slice`](#last-level-slice), [`manual fusion`](#manual-fusion)                                |
| composition.6 | horizontal producer → scan    | ✓                              | — (unfused: the intermediate materializes)                                                                                                             |
| X1            | X1 melting double loop        | ✗                              | [`no: scatter`](#no-scatter)                                                                                                                           |
| X2            | X2 `while` over table         | poor readability & performance | [`compile-time unroll`](#compile-time-unroll) (all bins evaluated) or [`outside the DSL`](#outside-the-dsl)                                            |
| X3            | X3 bounded window             | poor readability & performance | [`compile-time unroll`](#compile-time-unroll) ×M (field op) or [`delay line`](#delay-line) ×M                                                           |



#### [[personal/tehrengruber/scan-design/scan-patterns#a1|A1.1]]

```python
# example A1.1
def recurrence(x: Column, y: Column) -> tuple[Column, Column]:
    u, v = np.empty_like(x), np.empty_like(x)
    u[:2], v[:2] = x[:2], y[:2]  # m-level start region: the boundary pattern C1
    for k in range(2, x.size):
        u[k] = x[k] - 0.5 * u[k - 1] * v[k - 1]
        v[k] = y[k] + u[k - 2] * v[k - 1]
    return u, v

@scan_operator(index_var="level", init=(0.0, 0.0))
def recurrence(state: tuple[KView[float], KView[float]], level: K, x: KView[float], y: KView[float]):
  u, v = state
  if level < K(2):
    return x, y
  else:
    return (x-0.5*u[-1]*v[-1], y + u[-2] * v[-1])

@scan_operator(init=(x[K(0):K(2)], y[K(0):K(2)]))
def recurrence(state: tuple[KView[float, 2], KView[float, 2]], x: KView[float], y: KView[float]):
  u, v = state
  return (x-0.5*u[-1]*v[-1], y + u[-2] * v[-1])
```

#### [[personal/tehrengruber/scan-design/scan-patterns#a1|A1.2]]

```python
# example A1.2 — pseudo-numpy: half indices mark the interface (staggered) grid
p[-1/2] = 0.0
for k in range(n):
    p[k + 1/2] = p[k - 1/2] + rho[k] * g * dz[k]
    
@scan_operator(init=0.0)
def example_a1_2(p: KView[float, 1], rho: KView[float], g: float, dz: KView[float]):
	return p[-1] + rho[-1/2] * g * dz[-1/2]

```

#### [[personal/tehrengruber/scan-design/scan-patterns#a1|A1.3]]

```python
# example A1.3
def inclusive(w: Column, seed: float) -> Column:  # n inputs -> n + 1 outputs
    out = np.empty(w.size + 1)
    out[0] = seed  # the init is itself the first output level
    for k in range(1, w.size+1):
        out[k] = out[k-1] + w[k-1]
    return out
    
@scan_operator(index_var="level", init=0.0)
def inclusive(out: KView[float], level: K, seed: float, w: KView[float]):
    if level==K(0):
      return seed
    else:
      return out[-1] + w[-1]
      
inclusive(seed, w, range=extend(w.domain[K], 1))

@field_operator
def inclusive(seed: float, w: Field)-> Field
	@scan_operator(init=seed)
	def inclusive_pass(out: KView[float], w: KView[float]):
	    return out[-1] + w[-1]
	return inclusive_pass(w, range=extend(w.domain[K], 1))
      
inclusive(seed, w)
# 1 in KView[float, 1] describes number of previous levels that can be accessed (the first level gets init value)
```

#### [[personal/tehrengruber/scan-design/scan-patterns#a1|A1.4]]

```python
# example A1.4
def neighbour_inputs(y: Column) -> Column:
    out = np.zeros_like(y)
    for k in range(1, y.size - 1):  # the lookahead shrinks the output range by one level
        out[k] = 0.5 * out[k-1] + y[k + 1]
    return out

@field_operator
def neighbour_inputs(...)
	@scan_operator(index_var="level")
	def scan(out: KView[float], level: K, x: KView[float], y: KView[float]):
	    if level==K(0) or level==K(mesh.num_levels-1):
	      return 0
		else:
		  return 0.5 * out[-1] + x[-1] - y[1]
    
    return scan(x, y, range=domain) # TODO: complete domain
    
@field_operator
def neighbour_inputs(...)
	@scan_operator(init=0.0, end=0.0)
	def scan(out: KView[float], x: KView[float], y: KView[float]):
		return 0.5 * out[-1] + x[-1] - y[1]
    
    return scan(x, y, range=domain) # TODO: complete domain
```

#### [[personal/tehrengruber/scan-design/scan-patterns#a2|A2]]

```python
# example A2
def activated(q: Column, mask: Mask) -> Column:
    out = np.empty_like(q)
    active: Mask = np.logical_or.accumulate(mask)  # the latch is itself an A1 scan
    out[0] = 0.5 * q[0] if active[0] else q[0]
    for k in range(1, q.size):
        # the active branch stands in for the expensive body; the else is the identity
        out[k] = 0.5 * (out[k - 1] + q[k]) if active[k] else q[k]
    return out

@scan_operator(index_var="level", init=(False, 0.0))
def activated(state, level: K, q: KView[float], mask: KView[float]):
	active, out = state
	active |= mask[0]
	if level == K(0):
	    return active, 0.5 * q[0] if active else q[0]
	else:
		return active, 0.5 * (out[-1] + q[0]) if active else q[0]

@scan_operator(init=(mask[0], where(mask[0], 0.5 * q[0] , q[0])))
def activated(state, q: KView[float], mask: KView[float]):
	active, out = state
	active |= mask[0]
	return active, 0.5 * (out[-1] + q[0]) if active else q[0]
```

#### [[personal/tehrengruber/scan-design/scan-patterns#b1|B1]]

```python
# example B1
def column_sum(x: Column) -> np.float64:
    return np.sum(x)  # K removed: one value per column
    
gtx.sum(x, axis=K)
```

#### [[personal/tehrengruber/scan-design/scan-patterns#b2|B2]]

```python
# example B2
def argmin_gather(x: Column, z: Column, q: Column) -> Column:
    best, k_best = x[0], 0
    for k in range(1, x.size):
        if x[k] < best:
            best, k_best = x[k], k
    # the index as a region bound, the gathered value as a per-level threshold
    return np.where(np.arange(x.size) < k_best, np.minimum(q, z[k_best]), q)
    
@scan_operator(fold=True, index_var="level", init=(x[0], K(0)))
def argmin_gather_selection(state, level: K, x: KView[float], y: KView[float]):
	best, k_best = state
	if x[0] < best:
		return x[0], level
	else:
		return best, k_best
			
@field_operator
def argmin_gather(x: Field, y: Field) -> float, Field:
	best, k_best = argmin_gather_selection(x, y)
	return concat_where(x.domain[K] < k_best, minimum(q, z[k_best]), q)
```

#### [[personal/tehrengruber/scan-design/scan-patterns#b3|B3]]

```python
# example B3
def first_hit(h: Column, p: Mask, fill: float) -> np.float64:
    for k in range(h.size):  # reversed(range(...)) -> last hit
        if p[k]:
            return np.float64(h[k])
    return np.float64(fill)

@scan_operator(exit_condition=True, init=None)
def scan(_: tuple[bool, float], h: KView[float], p: KView[float], fill: float):
	if p[0]:
		return True, h
	return False, fill
```

#### [[personal/tehrengruber/scan-design/scan-patterns#b4|B4]]

```python
# example B4
def masked_sum(w: Column, depth: int) -> np.float64:  # depth: per-column, runtime
    s = np.float64(0.0)
    for k in range(depth):  # data-dependent trip count
        s = s + w[k]
    return s

gtx.sum(w[depth:])
```

#### [[personal/tehrengruber/scan-design/scan-patterns#c1|C1]]

```python
# example C1
def regions(x: Column) -> Column:  # n >= 5
    out = np.empty_like(x)
    n = x.size
    out[:2] = x[:2]  # two-level start region
    for k in range(2, n - 2):  # branch-free interior
        out[k] = 0.5 * out[k - 1] + x[k]
    out[n - 2] = out[n - 3] - x[n - 2]  # second to last: its own body
    out[n - 1] = out[n - 2] + x[n - 1]  # last level: another body
    return out

@scan_operator(index_var="level", init=0.0)
def regions(out: KView[float], level: K, x: KView[float], grid_info: GridInfo):
    if level < K(2):
        return x[0]
    elif K(grid_info.nz - 2) <= level:
        return out[-1] - x[0]
    else:
        return 0.5 * out[-1] + x[0]
        
@field_operator()
def regions(...)
	@scan_operator(init=x[K(0):K(2)])
	def regions(out: KView[float], x: KView[float], grid_info: GridInfo):
		return 0.5 * out[-1] + x[0]
	out = regions(x, grid_info, range=x.domain[:-2])
	out = concat_where(K < (grid_info.nz - 2), out, out[K-1] - x)
```
#### [[personal/tehrengruber/scan-design/scan-patterns#c2|C2]]

```python
# example C2
def seeded(dz_rho_g: Column, p_surface: float) -> Column:
    out = np.empty_like(dz_rho_g)
    out[0] = p_surface * np.exp(dz_rho_g[0])  # the boundary value *is* the init
    for k in range(1, dz_rho_g.size):
        out[k] = out[k - 1] * np.exp(dz_rho_g[k])
    return out

@field_operator
def seeded(dz_rho_g: Field, p_surface: Field) -> Field:
	@scan_operator(init=p_surface)
	def seeded_pass(p: float, dz_rho_g: float) -> float:
		return p * exp(dz_rho_g)
    return seeded_pass(dz_rho_g)
```
#### [[personal/tehrengruber/scan-design/scan-patterns#c3|C3]]

```python
# example C3
def start_at(x: Column, k_start: int, seed: float) -> Column:
    # k_start: per column, any level in 0 .. n-1, or n for "this column never starts"
    out = np.zeros_like(x)
    if k_start < x.size:
        out[k_start] = seed  # the recurrence starts mid-column, at a level that is data
    for k in range(k_start + 1, x.size):
        out[k] = 0.5 * out[k - 1] + x[k]
    return out

@scan_operator(index_var="level", init=0.0)
def start_at(out: KView[float], k_start: K, level: K, x: KView[float], seed: float):
    if level == k_start:
        return seed
    else:
        return 0.5 * out[-1] + x[0]

# k_start is per-column data: the output domain itself is data-dependent
start_at(k_start, x, seed, range=named_range(K, k_start, x.domain[K].stop))

@field_operator
def seeded(k_start, x, seed: Field) -> Field:
	@scan_operator(init=seed)
	def start_at(out: KView[float], x: KView[float]):
		return 0.5 * out[-1] + x[0]
	return start_at(x, range=named_range(K, k_start, x.domain[K].stop))

# k_start is per-column data: the output domain itself is data-dependent
seeded(k_start, x, seed)
```
#### [[personal/tehrengruber/scan-design/scan-patterns#d1|D1]]

```python
# example D1
def windowed_scan(x: Column, m_max: int) -> Column:  # x carries m_max - 1 lookahead levels
    n = x.size - m_max + 1
    out = np.empty(n)
    out[0] = np.sum(x[0:m_max])
    for k in range(1, n):
        out[k] = 0.5 * out[k - 1] + np.sum(x[k : k + m_max])  # lookahead window
    return out

@scan_operator(index_var="level", init=0.0)
def windowed_scan(out: KView[float], level: K, x: KView[float], m_max: int):
    window = gtx.sum(x[0:m_max])  # lookahead window over the input
    if level == K(0):
        return window
    else:
        return 0.5 * out[-1] + window

# x is defined on K(0, n + m_max - 1): the input carries the lookahead levels
windowed_scan(x, m_max, range=named_range(K, 0, n))

@scan_operator(init=gtx.sum(x[0:m_max]))
def windowed_scan(out: KView[float], x: KView[float], m_max: int):
    window = gtx.sum(x[0:m_max])  # lookahead window over the input
    return 0.5 * out[-1] + window

# x is defined on K(0, n + m_max - 1): the input carries the lookahead levels
windowed_scan(x, m_max, range=named_range(K, 0, n))

```
#### [[personal/tehrengruber/scan-design/scan-patterns#e1|E1]]

```python
# example E1
# 140 = RRTMG LW; in PMAP the count is read from the gas-optics file at runtime
ComponentField = np.ndarray[tuple[Literal[140], int], np.dtype[np.float64]]  # (component, K)


def per_component(a: ComponentField, b: ComponentField) -> ComponentField:
    out = np.empty_like(b)
    out[:, 0] = b[:, 0]
    for k in range(1, b.shape[1]):
        out[:, k] = a[:, k] * out[:, k - 1] + b[:, k]  # one A1 per (column, component)
    return out

# the dense dimension as a tuple of fields, the body a tuple comprehension
@scan_operator(index_var="level", init=(0.0,) * N_COMPONENTS)
def per_component(out: tuple[KView[float], ...], level: K, a: tuple[KView[float], ...], b: tuple[KView[float], ...]):
    if level == K(0):
        return tuple(b_i[0] for b_i in b)
    else:
        return tuple(a_i[0] * o[-1] + b_i[0] for a_i, o, b_i in zip(a, out, b))
        
# the dense dimension as a tuple of fields, the body a tuple comprehension
@scan_operator(init=tuple(b_i[K(0)] for b_i in b))
def per_component(out: tuple[KView[float], ...], a: tuple[KView[float], ...], b: tuple[KView[float], ...]):
	return tuple(a_i[0] * o[-1] + b_i[0] for a_i, o, b_i in zip(a, out, b))
```
#### [[personal/tehrengruber/scan-design/scan-patterns#composition|composition.1]]

```python
# example composition.1
def forward_sweep(x: Column, y: Column, surface: float) -> Column:
    out = np.empty_like(x)
    n = x.size
    out[0] = surface * x[0]  # C1 peeled first row, seeded by data (C2)
    for k in range(1, n - 1):  # branch-free interior
        out[k] = 0.5 * out[k - 1] + x[k] * y[k + 1]  # A1 recurrence + lookahead input
    out[n - 1] = out[n - 2] + x[n - 1]  # C1 end-anchored row: no y[n] to read
    return out

@scan_operator(index_var="level", init=0.0)
def forward_sweep(out: KView[float], level: K, x: KView[float], y: KView[float], surface: float, grid_info: GridInfo):
    if level == K(0):
        return surface * x[0]
    elif level == K(grid_info.nz - 1):
        return out[-1] + x[0]  # no y[1] to read: the branch bounds the extent
    else:
        return 0.5 * out[-1] + x[0] * y[1]

@scan_operator(init=surface * x[K(0)])
def _forward_sweep(out: KView[float], x: KView[float], y: KView[float], surface: float):
    return 0.5 * out[-1] + x[0] * y[1]

@field_operator
def forward_sweep(x: Field, y: Field, surface: Field) -> Field:
    interior = _forward_sweep(x, y, surface, range=x.domain[K][:-1])
    # the end anchor as a domain split: the last row reads the interior's final value
    return concat_where(x.domain[K][:-1], interior, interior(Koff[-1]) + x)
```

#### [[personal/tehrengruber/scan-design/scan-patterns#composition|composition.2]]

```python
# example composition.2
def back_substitution(e: Column, f: Column) -> Column:  # f: the forward sweep's output
    out = np.empty_like(f)
    out[-1] = f[-1]
    for k in range(f.size - 2, -1, -1):
        out[k] = e[k] * out[k + 1] + f[k]
    return out

@scan_operator(index_var="level", forward=False, init=0.0)
def back_substitution(out: KView[float], level: K, e: KView[float], f: KView[float], grid_info: GridInfo):
    if level == K(grid_info.nz - 1):
        return f[0]
    else:
        return e[0] * out[1] + f[0]

@scan_operator(forward=False, init=f[K(-1)])  # the last level, visited first
def back_substitution(out: KView[float], e: KView[float], f: KView[float]):
    return e[0] * out[1] + f[0]
```

#### [[personal/tehrengruber/scan-design/scan-patterns#composition|composition.3]]

```python
# example composition.3
def normalized(x: Column) -> Column:
    return x / np.sum(x)  # the column-constant broadcast back over K

x / gtx.sum(x, axis=K)
```

#### [[personal/tehrengruber/scan-design/scan-patterns#composition|composition.4]]

```python
# example composition.4
def species(q: Column, rho: Column) -> Column:
    out = np.empty_like(q)
    out[0] = q[0]
    for k in range(1, q.size):
        out[k] = out[k - 1] * np.sqrt(rho[k]) + q[k]  # shared work, recomputed per species
    return out

@scan_operator(index_var="level", init=0.0)
def species(out: KView[float], level: K, q: KView[float], rho: KView[float]):
    if level == K(0):
        return q[0]
    else:
        return out[-1] * sqrt(rho[0]) + q[0]

@scan_operator(init=q[K(0)])
def species(out: KView[float], q: KView[float], rho: KView[float]):
    return out[-1] * sqrt(rho[0]) + q[0]

# written per species; fusing the calls (and the shared sqrt) is the toolchain's job
a, b = species(q_a, rho), species(q_b, rho)
```

```python
# example composition.4 (fused)
def species_fused(q: Stack, rho: Column) -> Stack:  # = np.stack([species(qi, rho) for qi in q])
    out = np.empty_like(q)
    out[:, 0] = q[:, 0]
    for k in range(1, q.shape[1]):
        shared = np.sqrt(rho[k])  # one k-loop, the shared work computed once
        out[:, k] = out[:, k - 1] * shared + q[:, k]
    return out
```

#### [[personal/tehrengruber/scan-design/scan-patterns#composition|composition.5]]

```python
# example composition.5
def peak(a: Column, x: Column) -> tuple[Column, np.float64]:
    out = np.empty_like(x)
    out[0] = x[0]
    for k in range(1, x.size):
        out[k] = a[k] * out[k - 1] + x[k]  # A1
    return out, np.float64(np.max(out))  # B1 over its per-level output

@scan_operator(index_var="level", init=0.0)
def running(out: KView[float], level: K, a: KView[float], x: KView[float]):
    if level == K(0):
        return x[0]
    else:
        return a[0] * out[-1] + x[0]

@scan_operator(init=x[K(0)])
def running(out: KView[float], a: KView[float], x: KView[float]):
    return a[0] * out[-1] + x[0]

@field_operator
def peak(a: Field, x: Field):
    out = running(a, x)
    return out, gtx.max(out, axis=K)  # one pass if the toolchain fuses scan and reduction
```

```python
# example composition.5 (fused)
def peak_fused(a: Column, x: Column) -> tuple[Column, np.float64]:  # = peak, one pass
    out = np.empty_like(x)
    out[0] = x[0]
    total = out[0]  # the accumulator: the only state never emitted per level
    for k in range(1, x.size):
        out[k] = a[k] * out[k - 1] + x[k]
        total = max(total, out[k])
    return out, total
```

#### [[personal/tehrengruber/scan-design/scan-patterns#composition|composition.6]]

```python
# example composition.6
def div_then_scan(flux: Stack, a: Column) -> Stack:
    g: Stack = flux[1:, :] - flux[:-1, :]  # horizontal-offset producer, materialized
    out = np.empty_like(g)
    out[:, 0] = g[:, 0]
    for k in range(1, g.shape[1]):
        out[:, k] = a[k] * out[:, k - 1] + g[:, k]  # A1 per column
    return out

@scan_operator(init=x[K(0)])  # running, as in composition.5
def running(out: KView[float], a: KView[float], x: KView[float]):
    return a[0] * out[-1] + x[0]

@field_operator
def div_then_scan(flux: Field, a: Field) -> Field:
    g = flux(Ioff[1]) - flux  # horizontal-offset producer
    return running(a, g)  # fuses into the scan's k-loop: columns stay independent
```

```python
# example composition.6 (fused)
def div_then_scan_fused(flux: Stack, a: Column) -> Stack:  # = div_then_scan
    out: Stack = np.empty((flux.shape[0] - 1, flux.shape[1]))
    out[:, 0] = flux[1:, 0] - flux[:-1, 0]
    for k in range(1, flux.shape[1]):
        # the offset read moves into the k-loop: it gathers *inputs* of the
        # neighbouring columns, so the columns stay independent
        out[:, k] = a[k] * out[:, k - 1] + (flux[1:, k] - flux[:-1, k])
    return out
```

#### [[personal/tehrengruber/scan-design/scan-patterns#x1|X1]]

```python
# example X1
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

#### [[personal/tehrengruber/scan-design/scan-patterns#x2|X2]]

```python
# example X2
def rh_lookup(rh: float, bounds: Column, table: Column) -> np.float64:
    # bounds: ascending upper edge of each humidity bin, last one +inf
    # table: one tabulated coefficient per bin (N_RH rows, compile-time constant)
    i = 0
    while rh > bounds[i]:  # (1) data-dependent search for the bin
        i += 1
    return np.float64(table[i])  # (2) what the index is for: a computed-index gather
```

#### [[personal/tehrengruber/scan-design/scan-patterns#x3|X3]]

```python
# example X3
def box_flux(phi: Column, z: Column, v: Column, dt: float, m_max: int) -> Column:
    out = np.empty_like(phi)
    for k in range(phi.size):
        lo = max(k - m_max + 1, 1)  # M = max Courant number; clipped at the model top
        z_up = np.minimum(z[lo - 1 : k], z[k] + np.abs(v[lo : k + 1]) * dt)
        z_low = np.minimum(z[lo : k + 1], z_up)
        out[k] = -np.sum(phi[lo : k + 1] * (z_up - z_low))
    return out
```


## Requirements

Sketch vocabulary as in the examples above: `KView[T]` scanned input,
`KView[T, m]` carry with `m` readable previous levels, `K(n)` level literal,
`range=` output domain. Related: havogt's
[[personal/havogt/scan-redesign/scan-redesign|scan redesign]] (§3.1 K-local
views, §3.3 boundaries, §4.6 `history=m`).



### Boundary conditions

```python
def two_sided(x: Column, seed: float, top: float) -> Column:
    out = np.empty_like(x)
    out[0] = seed  # bottom BC
    for k in range(1, x.size - 1):
        out[k] = 0.5 * out[k - 1] + x[k]
    out[-1] = top  # top BC
    return out
```

#### Option 1: Using init:

```python
@field_operator
def two_sided(x: IKField, seed: float, top: float) -> IKField:
    @scan_operator(init=seed)
    def _interior(out: KView[float], x: KView[float]) -> float:
        return 0.5 * out[-1] + x[0]

    interior = _interior(x, range=x.domain[K][:-1])  # K[0] written from init, body on K[1:-1]
    return concat_where(x.domain[K][:-1], interior, top)  # top BC outside the scan
```

- can only be used for one-sided boundary conditions and only if they are the same on all levels -> otherwise concat_where needs to be used
- get rid of index_var="level"
- better compatibility with JAX
- need to a solution to pass runtime values to scan_operator (see below)

#### Option 2: Using if-condition:

```python
@scan_operator(index_var="level", init=0.0)  # init never read
def two_sided(out: KView[float], level: K, x: KView[float], seed: float, top: float, grid_info: GridInfo) -> float:
    if level == K(0):
        return seed
    elif level == K(grid_info.nz - 1):
        return top
    else:
        return 0.5 * out[-1] + x[0]


two_sided(x, seed, top, grid_info)
```

- more flexible and easier to understand but harder to check correctness
- init values will not be used
- need some way to access index inside the operator (see below)

### Passing runtime values to init

#### Option 1: Define scan_operator inside field_operator:

```python
@field_operator
def two_sided(x: IKField, seed: float) -> IKField:
    @scan_operator(init=seed)
    def _interior(out: KView[float], x: KView[float]) -> float:
        return 0.5 * out[-1] + x[0]

    return _interior(x, range=x.domain[K][:-1])
```

#### Option 2: Separate decorator and pass of scan_operator:

```python
@scan_operator(axis=KDim, forward=True)  # axis/forward stay in the decorator, init moves to the call
def _interior(out: KView[float], x: KView[float]) -> float:
    return 0.5 * out[-1] + x[0]


@field_operator
def two_sided(x: IKField, seed: float, top: float) -> IKField:
    interior = _interior(x, init=seed, range=x.domain[K][:-1])  # init checked against the carry annotation
    return concat_where(x.domain[K][:-1], interior, top)


two_sided(x, seed, top)
```

A decorator default (`@scan_operator(axis=KDim, forward=True, init=0.0)`)
stays possible; the call-site `init=` overrides it.

#### Option 3: Pass init as String -> becomes implicit parameter of the function (Type?):

```python
@scan_operator(init="seed")
def _interior(out: KView[float], x: KView[float]) -> float:
    return 0.5 * out[-1] + x[0]


@field_operator
def two_sided(x: IKField, seed: float) -> IKField:
    return _interior(x, seed=seed, range=x.domain[K][:-1])  # type of `seed` = type of the carry
```

For a `KView[float, m]` carry with `m > 1` or a tuple carry the implicit parameter would have to be `m` levels or a tuple, which a bare name cannot express.
### Accessing index inside the operator

(currently used only in B2 and if-condition variant for BCs)

```python
def argmin(x: Column) -> int:
    best, k_best = np.inf, 0
    for k in range(x.size):
        if x[k] < best:
            best, k_best = x[k], k
    return k_best
```

#### Option 1: Index_var parameter:

```python
@scan_operator(fold=True, index_var="level", init=(np.inf, K(0)))
def argmin(state: tuple[float, K], level: K, x: KView[float]) -> tuple[float, K]:
    best, k_best = state
    if x[0] < best:
        return x[0], level
    return best, k_best
```

#### Option 2: Introduce Index builtin:

```python
@scan_operator(fold=True, init=(np.inf, K(0)))
def argmin(state: tuple[float, K], x: KView[float]) -> tuple[float, K]:
    best, k_best = state
    level = index(K)  # GTIR already has `index(dim)` (used by the concat_where transformation)
    if x[0] < best:
        return x[0], level
    return best, k_best
```

`index_var` and the `index` builtin only differ in how the body obtains the
current level. Both need the same index type `K` for what the body does with it:
- literals: `K(0)` (BCs, B2)
- comparison with the running index: `level == K(0)`, `level < K(2)` (A1.1)
- `±const` arithmetic: `K(grid_info.nz - 2) <= level` (C1)
- end anchor for the last level: a runtime scalar `K(grid_info.nz - 1)` or an anchor resolved against the scan range (`K.stop - 1`)
- conversion from an `int32` field: `level == k_start` with `k_start` per column (C3)
- usable as carry and output element type: `(best, k_best)` in B2

### Define scan output domain

 `range=` at the call else full K range of its inputs

```python
inclusive(w, range=extend(w.domain[K], 1))                 # one level more, filled by init (A1.3)
_forward_sweep(x, y, surface, range=x.domain[K][:-1])      # explicit sub-range (composition.1)
start_at(x, range=named_range(K, k_start, x.domain[K].stop))  # k_start a field: a mask, not a domain (C3)
```

### kview in the inputs


```python
def body(out: KView[float], x: KView[float], c: float, mask: KView[bool]) -> float:
    x[0]       # current level
    x[-1]      # previous level
    x[1]       # lookahead
    x[0:m]     # window over a local dimension
    x[-1/2]    # staggered neighbour (A1.2)
    c          # K-less input
```

### kview in the carry

```python
def body(out: KView[float, m], ...) -> float  # out[-1..-m] forward, out[1..m] backward; out[0] unreadable
```

- `KView[T, m]`: `m` is the number of previously written levels the body may
  read (`out[-1]` is the value returned one level ago, the return value is the
  implicit `out[0]`) and the number of levels the init supplies. A missing
  count means `m = 1`; it is written when more is needed (`KView[float, 2]`
  in A1.1).
- the `m` init levels are *written*: they are the first `m` levels of the
  range, the body runs on the rest (composition.2/4, A1.1). Forms: scalar or
  K-less field (broadcast), tuple of `m`, field slice of `m` levels.
- plain carry (`state: float`, C2): init is the carry before the first level,
  not written (today's semantics).
- tuple carry mixes both (A2); non-output slots never materialized.
- `fold=True`: nothing written, final carry returned K-less (B2, B3).

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
