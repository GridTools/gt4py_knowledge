---
title: JAX support for connectivities and premap
author: havogt
tags:
  [
    jax,
    embedded,
    connectivities,
    premap,
    unstructured,
    domain-inference,
    pytree,
    tracing,
    distributed,
    autodiff,
    prior-art,
  ]
created: 2026-08-03
status: draft
---

> **TL;DR** Make `premap` work under `jax.jit` by giving a connectivity a small,
> eagerly-computed **bounds descriptor** (`value_min`, `value_max`, `support_box`) as
> pytree *aux data* and putting the neighbour table in pytree *children*. Domain
> inference then answers from integers instead of from table contents, so the table
> becomes a runtime argument rather than a compiled-in constant, and one compiled
> program serves every mesh of the same shape and bounds.

## Problem

`gt4py.next`'s embedded backend can run on JAX arrays, and a `Field` already crosses a
`jax.jit` boundary cleanly: the `Domain` is static pytree aux data, the array is the
single traced child. Connectivities do not, and `premap` — the gather through a
neighbour table — is the operation that breaks.

`premap` does two things with the table, and they pull in opposite directions:

- **domain inference.** `inverse_image` derives the *output domain* from the table's
  **contents** via `_hyperslice`, which uses `nonzero`/`min`/`max`/`any`/`all` and
  returns Python `slice` objects. This needs concrete values.
- **the gather itself.** Needs the table on device, as a runtime argument, for meshes
  of ICON scale.

Under `jax.jit` you cannot have both with the obvious encodings. A connectivity
captured as a **closure** keeps the table concrete but JAX inlines it into the
compiled module as a constant. A connectivity registered as a plain **pytree node**
makes the table a runtime argument but hands `inverse_image` a tracer.

There is a second, independent obstacle: `_hyperslice` is *untraceable by
construction*, because `jnp.nonzero` has a data-dependent output shape and cannot be
staged without an explicit `size=`. Whatever else changes, inference must run eagerly.

## Constraints

1. **The inference result depends on `image_range`**, which comes from the field being
   premapped and differs between calls. It cannot be precomputed once per connectivity.
2. **The table is large.** ICON meshes have several connectivities of millions of rows.
   Anything that scales with table size per compiled program is disqualified.
3. **icon4py is distributed.** Whatever is static must be meaningful per rank.
4. **Gathers must be differentiable** — 4D-var is the motivating use case.
5. `Field.__eq__` is elementwise by design, so fields and connectivities can never be
   JAX `static_argnums` (which require `__eq__` to be a real predicate).

## Design

### The bounds descriptor

At construction, one eager pass over the table records:

```
value_min     minimum over non-skip entries
value_max     maximum over non-skip entries
support_box   per-axis bounding box of non-skip *positions*
```

All Python ints. This is a property of the connectivity's *type* in the same sense
`max_neighbors` already is, and it is the same move jax-md makes with `max_occupancy`
and PyG with `EdgeIndex._sparse_size`: measure the table once eagerly, carry the small
answer as metadata.

### Two paths in `inverse_image`

**Covering path** — when `image_range` covers `[value_min, value_max]` *and* does not
contain `skip_value`:

```
return domain.slice_at[support_box]
```

Pure integer arithmetic. No contact with the buffer, so it behaves identically under
`jit`, `grad`, `vmap` and `shard_map`, and for NumPy and CuPy.

This is **exact, not an approximation** — see
[[personal/havogt/jax-connectivities/jax-connectivities_research|the research appendix]]
for the brute-force check (0 mismatches in 2 531 covering cases out of 18 458 pairs).
The `skip_value ∉ image_range` guard is load-bearing: without it, 176 of those cases
disagree.

**Narrowing path** — when the field's domain does not cover the table's image. Keep
today's `_hyperslice` under `jax.ensure_compile_time_eval()`, which evaluates eagerly
at trace time. Under `jit` with a traced table this raises a clear
`TracerBoolConversionError`, which is the intended contract rather than a defect.

The covering case is the one that matters in practice: icon4py allocates input fields
over the full horizontal range, and program `domain=` restrictions apply to *outputs*.
Of the non-covering pairs, only ~9% even yield a valid narrowing — the rest already
raise "non-contiguous or empty".

### Pytree registration

```
children  = (table,)
aux_data  = (domain, codomain, skip_value, bounds)
```

Everything in aux data is value-typed and small, which matters more than it looks:
in jaxlib 0.6.2 `PyTreeDef.__hash__` **ignores custom-node aux data entirely**, so
treedefs collide and are separated by a linear `__eq__` scan on every call. Aux data
`__eq__` must therefore be O(1) — a frozen dataclass of ints, never an array or a
lazily-computed property.

## Why not the alternatives

**Connectivity as a jit closure** (today). The table is inlined as
`stablehlo.constant`; in 0.6.2 this is unconditional, with no size threshold. Fatal
independently of size: a closure constant is replicated on every device, invisible to
`in_shardings`, not donatable, and JAX refuses outright to close over an array spanning
non-addressable devices. Multi-node is impossible. A fix exists upstream
(`JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS`, jax ≥ 0.7.1) but is off by default and not
available at our pinned version.

**Identity-handled concrete table in aux data.** Correct, and the honest encoding of
"the output domain depends on this exact table" — but it retraces per connectivity
*object* rather than per mesh (`restrict()` and `as_connectivity_field()` mint fresh
objects), and it pins the table alive in JAX's compilation cache in a way
`jax.clear_caches()` does not release. The bounds descriptor removes both.

The decisive difference: with bounds, **two different meshes of the same shape and
bounds share one compiled program**. Neither alternative can do that.

## Independent defect found while researching this

`_gather_premap` relies on `-1` wrapping around, masking only later inside
`_make_reduction`. Under `grad` this produces **NaN gradients**, because the
sentinel gathers a real element which then flows through the operator's
nonlinearities. This is a correctness blocker for 4D-var and is unrelated to
everything above.

The fix is to sanitise the *index* before the gather (`where(table != skip, table, 0)`)
rather than the operand — one `where`, done once in `_gather_premap`, and it also
removes an existing oddity where the domain-start offset shifts `-1` to a different
wrapped element. **Worth doing on its own, ahead of any JAX work.**

Related: `_make_reduction` masks against the module-level `common._DEFAULT_SKIP_VALUE`
instead of the connectivity's own `skip_value`.

## Risks and open questions

- **Distributed is the biggest unknown.** The bounds describe the *global* index space.
  Under `shard_map` the traced child is a per-rank table, possibly locally renumbered,
  while bounds and domain stay global. Bounds and table must then be constructed
  together per rank. This needs designing explicitly, not discovering. Note also that
  auto-sharding a gather all-gathers the whole operand — `shard_map` is mandatory, and
  in explicit-sharding mode `lax.gather` has no sharding rule at all, so
  `_gather_premap` would need an `.at[].get(out_sharding=…)` route.
- **`as_offset` is an explicit non-goal.** It builds a connectivity from a runtime
  field, so no eager bounds exist. Under `jit` it must either fail loudly or take a
  declared output domain.
- **`ensure_compile_time_eval` is broken under `lax.scan`** in 0.6.2. Only affects the
  narrowing fallback, but a user wrapping a program in `lax.scan` for time-stepping
  would hit an opaque error.
- **The traced connectivity must be installed in the embedded context**, not merely
  passed to `premap` — `_make_reduction` reads the offset provider from the contextvar.
  The design silently degrades to closure constants if a caller forgets, so this
  belongs in the contract for `embedded/context.py`.
- `restrict()` on a traced table cannot recompute bounds; the constructor must accept
  their absence.

## Sketch of the change

`common.py` — add the value-typed bounds descriptor (consider hanging it off
`NeighborConnectivityType`, next to `max_neighbors`). Strengthen the `Connectivity`
contract: `inverse_image` is answerable without inspecting the buffer whenever the
image range covers the value range. Document that a `Connectivity` may never itself be
aux data (`__eq__` raises by design).

`nd_array_field.py` — populate the descriptor in `from_array`; split `inverse_image`
into covering and fallback paths; sanitise skip values in `_gather_premap`; fix
`_make_reduction`'s skip value; register `JaxArrayConnectivityField` as a pytree node.

The regression test that matters, because it is the property nothing else provides and
the one that will silently rot: **two different tables of the same shape and bounds
must produce exactly one trace, and correct results for both.**

## Related

- [[personal/havogt/mesh-and-first-class-halos|A mesh concept with first-class halos]] —
  overlaps directly on connectivities, offset providers, domain inference and the
  distributed story; a mesh object owning its connectivities would be the natural home
  for the bounds descriptor.
- [[personal/havogt/dependent-local-dimensions|Dependent local dimensions and connectivity chains]] —
  local dimensions and reductions over them are what consume `premap`'s output.
- [[personal/havogt/scan-redesign|Redesign of the vertical scan]] — the same
  static-metadata-versus-traced-data tension, and the same `lax.scan` caveat.
- [[personal/havogt/jax-connectivities/jax-connectivities_research|Research appendix]] —
  measurements, prior-art survey, and JAX-internals citations.
