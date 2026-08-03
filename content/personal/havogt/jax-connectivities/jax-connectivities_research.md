---
title: JAX connectivities — research appendix
author: havogt
tags: [jax, connectivities, premap, pytree, tracing, distributed, autodiff, prior-art]
created: 2026-08-03
status: draft
---

> Appendix to [[personal/havogt/jax-connectivities/jax-connectivities|JAX support for
> connectivities and premap]]. Measurements against **jax/jaxlib 0.6.2**, CPU-only
> jaxlib. Claims are labelled **[M]** measured, **[D]** documented, **[S]** read in
> source, **[I]** inferred.

## Measured comparison

Field 60 000×f64, table 120 000×2 int32 (0.96 MB):

| | closure (today) | identity handle | bounds descriptor |
| --- | --- | --- | --- |
| jaxpr consts | `[(120000,2) int32]` | `[]` | `[]` |
| StableHLO text | 1.92 MB | small | **2.4 kB** |
| table in `func @main` | no | yes | **yes** |
| retraces, same conn object | 1 | 1 | 1 |
| retraces, reconstructed conn | 2 | 2 | **1** |
| retraces, different mesh, same shape+bounds | 3 | 3 | **1** |
| `grad` | — | works | works (`sum(grad) == table.size`) |

**[M]** The covering-path equivalence was brute-forced over 18 458 `(table,
image_range)` pairs (random 2-D tables, with and without `skip_value=-1`) against the
real `_hyperslice`: **0 mismatches** in the 2 531 covering cases. Without the
`skip_value ∉ image_range` guard, 176 mismatch — all from ranges containing `-1`.

**[M]** Of non-covering pairs, only **8.9%** yield a valid narrowing; the rest raise
`_hyperslice`'s "non-contiguous or empty".

## Corrections to earlier measurements

**[M]** The apparent 2× blow-up (6.4 MB table → 12.8 MB StableHLO) is an artefact of
`.as_text()` rendering decimal ASCII. `module.operation.write_bytecode()` is ~1×. The
cost is nonetheless real: **[S]** that bytecode is `sha256`'d for the persistent
compilation-cache key on every lowering (`jax/_src/cache_key.py:214-219`). Peak RSS
growth measured ≈110 MB for a 16 MB constant.

**[S]** Constant inlining is unconditional in 0.6.2: `core.is_literalable` requires
rank 0 (`jax/_src/core.py:510`), so every shaped closure array becomes a constvar, and
`mlir.lower_jaxpr_to_fun` emits `ir_constant` for each
(`jax/_src/interpreters/mlir.py:1765`). No size threshold, no flag.

**[D]** Upstream fix `JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=True` hoists large constants
into arguments, jax ≥ 0.7.1 —
[PR #30180](https://github.com/jax-ml/jax/pull/30180),
[docs](https://docs.jax.dev/en/latest/internals/constants.html). Default `False`.
Tracking issue [#3220](https://github.com/google/jax/issues/3220) open since 2020.

**[S]** Multi-node is refused outright for closures: `jax/_src/array.py:1100-1114` —
*"Closing over jax.Array that spans non-addressable (non process local) devices is not
allowed. Please pass such arrays as arguments to the function."*

Diagnostic available today: `JAX_CAPTURED_CONSTANTS_WARN_BYTES` (default 2 GB) plus
`JAX_CAPTURED_CONSTANTS_REPORT_FRAMES=-1` gives a per-constant capture-site report
(`jax/_src/interpreters/mlir.py:1111-1153`, jax 0.6.1, absent from the CHANGELOG).

## JAX mechanisms

**`ensure_compile_time_eval`** — **[S]** since 0.4.36 its body is
`with config.eager_constant_folding(True): yield` (`jax/_src/core.py:1383-1444`); the
mechanism is one branch in `partial_eval.JaxprTrace.process_primitive`
(`jax/_src/interpreters/partial_eval.py:1989-1992`): with no `Tracer` inputs, bind on
`core.eval_trace` instead of staging.

- **[M]** The docstring is **wrong** about raising `ConcretizationTypeError` when eager
  evaluation is impossible — tracer-valued ops silently fall through to staging.
- **[M]** Works under `grad`, `vmap`, `while_loop`, simple `shard_map`. **Broken under
  `lax.scan`**: `NotImplementedError: Evaluation rule for 'empty' not implemented`
  (cf. [#29996](https://github.com/jax-ml/jax/issues/29996)). Unverified on ≥0.7.
- **[D]** `shard_map` caveat from a maintainer
  ([discussion #31461](https://github.com/jax-ml/jax/discussions/31461)): it *"creates
  `Manual` HloShardings that don't really work in the execution part"*; workaround
  `with jax.sharding.use_abstract_mesh(AbstractMesh((), ())):`.

**Pytree aux data** — **[D]** the
[custom-pytree docs](https://docs.jax.dev/en/latest/custom_pytrees.html) require
"meaningful hashing and equality" and say the hash enters the jit cache key.
**[M] The hash part is false in jaxlib 0.6.2**: `PyTreeDef.__hash__` ignores custom-node
aux data entirely — identical hashes for aux `'a'`/`'b'`/`1`/`2`/`None`. Unhashable aux
(a `list`) works and never raises. Correctness rests entirely on `__eq__`; do not rely
on this, the docs assert the opposite.

**[M]** A *bare* array in aux data works on the first call and raises on the second:
`ValueError: Exception raised while checking equality of metadata fields of pytree …
(Note: arrays cannot be passed as metadata fields!)`. Identity short-circuits the first
comparison, so single-call tests hide it. **[M]** `Connectivity.__eq__` raises
`TypeError` by design (`common.py:1039`), so a `Connectivity` can never be aux data.

**[D]** `register_dataclass`'s own docstring (`jax/_src/tree_util.py:934-947`):
*"Metadata fields must be static, hashable, immutable objects … metadata fields cannot
contain `jax.Array` or `numpy.ndarray` objects."*

**Cache retention** — **[S]** `jax/_src/util.py:317-337`: `weakref_lru_cache` holds a
weak ref to the **first argument only**, strong refs to the rest, `maxsize=2048`.
`_infer_params_cached(fun, jit_info, signature, …)` makes the treedef (hence aux data)
strongly held. **[M]** aux data survives `jax.clear_caches()` and `f._clear_cache()`;
only `del f` frees it, and `gc.get_referrers` cannot see the holder. Related:
[#16278](https://github.com/jax-ml/jax/issues/16278),
[#11448](https://github.com/jax-ml/jax/issues/11448).

**`static_argnums`** — **[S]** `jax/_src/api_util.py:116-129` boxes static args
requiring `hash(val)` and `type(a) is type(b) and a == b`. **[M]** With a gt4py field:
*"static arguments should be comparable using `__eq__`"* then `ValueError: The truth
value of an array with more than one element is ambiguous`. Canonical issue
[#24204](https://github.com/jax-ml/jax/issues/24204) — errors on the *second* call.

**Donation** — **[M]** irrelevant for index arrays: `UserWarning: Some donated buffers
were not usable`. **[I]** Donation needs an output to alias into; a read-only table has
none.

## Prior art

The rule, consistent across every library surveyed: **arrays are children; only small,
hashable, value-typed descriptors are aux data; and no library derives a shape from
index-array contents inside `jit`.**

| library | aux data | children | static shapes | recompiles avoided |
| --- | --- | --- | --- | --- |
| **jraph** | *nothing* — plain `NamedTuple` | all 7 fields incl. `senders`/`receivers` | user declares ints to `pad_with_graphs` (*"do not support jax.jit"*); `segment_sum(num_segments=)` | topology is pure data → zero recompiles |
| **jax-md** | `max_occupancy` (**content-derived int**), `format`, `cell_size` | `idx`, `reference_position`, `error.code` | eager `allocate()` computes `int(occupancy * 1.25)` | fixed `[N, max_occupancy]`; recompile only on re-`allocate` |
| **e3nn-jax** | `irreps` (representation type) | `(array,)` | `jnp.where(…, size=)`; `jnp.unique(size=x.shape[0])  # Pigeonhole` | one treedef per irreps type |
| **Equinox** | `field(static=True)` values | everything `is_array` | n/a | `filter_jit`; **warns** if an array reaches a static field |
| **flax** | `pytree_node=False` fields | `variables` collections | Module is the static half | `nnx` **hard-errors** since 0.12.0 on data in a static attribute |
| **jax-cfd** | `(offset, grid)`; `Grid` is tuples of ints/floats | `(data,)` | structured — coordinates regenerated via `jnp.arange` | tiny descriptor is the whole key |
| **jax-fem** | *none* | element values, not indices | gather/scatter **hoisted out** of the kernel | inner kernel is index-free |
| **PyG** (contrast) | `EdgeIndex._sparse_size`, `Index._dim_size` | the index tensor | eager `int(edge_index.max())+1`, memoized | eager tolerates the sync; `torch.compile` **graph-breaks**, remedy is "pass the size explicitly" |

Smoking guns, read in source:

```python
# jraph/_src/models.py:166-168
# Equivalent to jnp.sum(n_node), but jittable
sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
```
```python
# jax_md/partition.py:1090-1096  -- reachable only from allocate(), NOT jit
max_occupancy = int(occupancy * capacity_multiplier + _extra_capacity)
```
```python
# jax_md/partition.py:1253
"Converts a sparse neighbor list to dense ids. Cannot be JIT."
```

The two closest precedents are **jax-md**'s allocate/update split — `allocate` *"cannot
be compiled, since it uses the values of positions to infer the shapes"*, with
`max_occupancy` as a content-derived static int documented as *"Changing this will
invoke a recompilation"* — and **PyG**'s `EdgeIndex`, which exists precisely to attach
`_sparse_size` next to the index tensor.

**jax-cfd** shows the structured-grid escape gt4py does not have: coordinates are
*regenerated* under jit. An unstructured mesh has no such compression.

## Skip values and masking

| library | sentinel | made safe by |
| --- | --- | --- |
| jraph | padding edges → node 0 | padding nodes carry zero features; `arange(static) < traced` masks |
| jax-md | index `N` (one past end) | gather clamping, then `out *= mask`; `safe_mask` double-`where` |
| e3nn-jax | `-1`, optionally remapped via `fill_src` | `mode="promise_in_bounds"` — refuses to tolerate it silently |
| **gt4py** | `-1`, relying on wraparound | masked *after* the gather, in `_make_reduction` |

**[S/M]** gt4py's `-1` wraps in JAX exactly as in NumPy —
`jax/_src/numpy/indexing.py:188-200` emits `select(index < 0, index + axis_size, index)`
for traced indices too. Note `jnp.take`'s `mode='fill'` does **not** catch it: after
normalisation the index is in-bounds.

**[M]** `_gather_premap` subtracts the domain start from the index array
(`nd_array_field.py:679`), so with a field domain starting at 2 the sentinel `-1`
becomes `-3` — a *different* wrapped element. Harmless for the primal, but it selects
which element receives spurious cotangent.

### The NaN-gradient bug

**[M]** With `x = [4, 9, 16, -1]`, `tbl = [[0,1],[2,-1]]`:

```
primal : 9.0                              <- correct
grad   : [0.25  0.1667  0.125  nan]
```

Both remedies verified: the double-`where` sanitising the **operand** (jax-md's
`safe_mask`; [JAX FAQ](https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where)),
or sanitising the **index** before the gather. The latter is the better fit — one
`where` in `_gather_premap`, and it also removes the domain-start shift above.

## Distributed

**[M]** Auto-sharding a gather with sharded operand *and* sharded indices (8 CPU
devices): the optimised HLO contains `all-gather(%param), replica_groups=[1,8]` — every
device materialises the entire global field. Confirmed at N=64 and N=2²⁰.
**[S]** XLA's `gather_scatter_handler.cc` has five partitioning methods; the
communication-free ones need iota-like monotone indices, so an unstructured table falls
back to all-gather or a whole-output all-reduce. **`jit` auto-sharding will not scale;
`shard_map` is mandatory.**

**[S/M]** In explicit-sharding mode gather has *no* sharding rule
(`jax/_src/lax/slicing.py:1950-1963`): *"Use `.at[...].get(out_sharding=)` to provide
output PartitionSpec for the gather indexing."*

**[D]** There is no halo-exchange example in the 0.6.2 `shard_map` tutorial — only a
passing mention under `ppermute`. The working example is pmap-based
([Wave_Equation.ipynb](https://github.com/jax-ml/jax/blob/main/cloud_tpu_colabs/Wave_Equation.ipynb)).
**[I]** Per-shard index renumbering has no documented JAX pattern; ICON already does it,
so this is a port rather than research.

**[M]** The decisive constraint: inside `shard_map` the child is a per-shard tracer of
local shape while an aux-data table reference keeps the **global** shape. Any path
reading the aux table instead of the child silently sees global data — which is exactly
what the identity-handle design does in `inverse_image`. The bounds descriptor has no
such path.

## Differentiability

**[S]** `_gather_transpose_rule` is `scatter_add` (`jax/_src/lax/slicing.py:2007-2027`);
`ad.defjvp(gather_p, _gather_jvp_rule, None)` marks indices non-differentiable, so index
provenance is irrelevant to `grad`.

- **`unique_indices` must never be set true.** **[S]** Comment at `slicing.py:1989`:
  *"We don't consume `unique_indices` directly in gather(), only in its transpose."* A
  connectivity is non-unique by construction; setting it leaves the forward pass correct
  and **silently drops gradient contributions**.
- **[D]** Scatter-add on GPU is non-deterministic for conflicting updates
  (`--xla_gpu_deterministic_ops=true` fixes it at a cost,
  [#17844](https://github.com/jax-ml/jax/issues/17844)); performance issue
  [#8637](https://github.com/jax-ml/jax/issues/8637) open since 2021. **[I]** The VJP of
  an edge→vertex gather is the high-contention case; the standard remedy is a
  `custom_vjp` whose backward uses the inverse connectivity. Future work, not a blocker.
- **[D]** OOB gather clamps but OOB scatter is *dropped*, and reverse-mode AD
  [does not preserve OOB semantics](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing).
  gt4py's `-1` is in-bounds after normalisation, so it does not benefit.

## Non-issue

**[M]** `_gather_premap` builds one full-output-shape index array per field dimension,
which looked like a scalability problem. It is not — XLA fuses it: for a 3-D gather
(8000×2×64 output) compiled `bytes accessed` is **10.3 MB for both** `_gather_premap`
and an idiomatic `jnp.take(arr, tbl, axis=0)`. CPU only; worth one GPU re-check, but no
rewrite indicated.
