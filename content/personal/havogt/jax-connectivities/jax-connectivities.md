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

> **TL;DR** Make `premap` work under `jax.jit` by registering a connectivity as a pytree
> whose **child** is the neighbour table and whose **aux data** carries a handle to the
> same buffer, keyed on `id(ndarray)`. Domain inference reads the handle — concrete at
> trace time — so it works for **narrowing** as well as covering ranges, while the table
> travels as a runtime argument instead of being compiled into the module.

## Problem

A `Field` already crosses a `jax.jit` boundary cleanly: `Domain` is static pytree aux
data, the array is the traced child. Connectivities do not, and `premap` is where it
breaks, because it uses the table for two things that pull opposite ways:

- **domain inference** — `inverse_image` derives the *output domain* from the table's
  **contents** (`_hyperslice`: `nonzero`/`min`/`max`/`any`/`all` → Python `slice`s). Needs
  concrete values.
- **the gather** — needs the table on device as a runtime argument, at ICON scale.

A connectivity captured as a **closure** keeps contents concrete but JAX inlines the
whole table into the compiled module. Registered as a **plain pytree node** it becomes a
runtime argument but inference receives a tracer.

Compounding this, `_hyperslice` is untraceable *by construction*: `jnp.nonzero` has a
data-dependent output shape and cannot be staged without an explicit `size=`. Inference
must run eagerly no matter what else changes.

**The domain generally does not cover the connectivity's image.** Narrowing is the normal
case, not the exception. Any design whose fast path handles only the covering case is
partial support and is rejected.

## Constraints

1. **Inference depends on `image_range`**, which comes from the field being premapped and
   differs between calls — it cannot be precomputed once per connectivity.
2. **Narrowing is the common case.** See above. This is the constraint that decides the
   design.
3. **The table is large** — millions of rows, several per mesh. Nothing may scale with
   table size per compiled program.
4. **Gathers must be differentiable** — 4D-var is the motivating use case.
5. `Field.__eq__` is elementwise by design, so fields and connectivities can never be
   `static_argnums` (which require `__eq__` to be a real predicate).

## Design

Register `JaxArrayConnectivityField` as a pytree node:

```
children  = (table,)
aux_data  = (domain, codomain, skip_value, Handle(table))
```

`Handle` is a thin wrapper whose `__eq__`/`__hash__` key on **`id(table)` — the buffer,
not the connectivity object**. `inverse_image` reads the table through the handle, which
is an ordinary Python reference and therefore concrete during tracing, and runs today's
`_hyperslice` under `jax.ensure_compile_time_eval()`.

The same buffer is referenced twice in different capacities: as aux data it is read at
trace time only and never staged; as the child it is passed at call time and gathered
from on device. No duplication.

**Verified for genuine narrowing.** Field on `Vertex[0:30000)`, table 120 000×2 with image
spanning `[0,60000)`:

```
jaxpr consts : []
main params  : %arg0: tensor<30000xf64>, %arg1: tensor<120000x2xi32>
StableHLO    : 1.4 kB
out domain   : Domain(Edge=(0:60000), E2V[local]=(0:2))
```

Narrowing occurred (`Edge` 120 000 → 60 000), the table stayed a jit **argument**, nothing
was inlined. The `restrict()` that follows slices the *traced* child with Python slices,
which is fully traceable.

### Why keying on the buffer matters

Retrace granularity is per **buffer**, not per wrapper object: reconstructing the
connectivity around the same array does not retrace; re-uploading the array does.
gt4py's normal flow already reuses buffers — `FieldOffset.as_connectivity_field()`
memoizes on `id(offset_definition)` and returns the same object, fed from the user's
stable offset-provider dict. So in practice this is **one trace per mesh**.

A content digest would additionally let two separately-uploaded copies of the same table
share a compiled program. It costs a full hash per new buffer and should not be added
speculatively.

### Optional, free optimisation

An eagerly-computed `(value_min, value_max, support_box)` descriptor lets `inverse_image`
skip the table read *when the range happens to cover* — pure integer arithmetic, exact
(brute-forced: 0 mismatches in 2 531 covering cases, given a `skip_value ∉ image_range`
guard). This is a shortcut inside the general path, never a fallback boundary, so it
changes no contract. Worth having; not load-bearing.

## Rejected alternatives

**Connectivity as a jit closure** (today). The table is inlined as `stablehlo.constant` —
unconditional in jax 0.6.2, no size threshold. Fatal independently of size: closure
constants are replicated per device, invisible to `in_shardings`, not donatable, and JAX
*refuses outright* to close over an array spanning non-addressable devices, so multi-node
is impossible. An upstream fix exists (`JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS`, jax ≥ 0.7.1)
but is off by default and postdates our pin.

**Static bounds descriptor as the design.** Exact only in the covering case, falling back
to eager inference otherwise — i.e. precisely the partial support that is out of scope.
Demoted to the optimisation above.

**Caller-supplied output domain.** The `segment_sum(num_segments=)` / jraph / e3nn-jax
analogy does *not* transfer: those declare a trivial quantity the caller already knows,
whereas a narrowed gt4py domain is a non-trivial function of table contents that nobody
upstream of `inverse_image` knows. Declaring it means the caller re-runs the same O(n)
scan by hand, and field view gives up automatic domain inference — a core promise
(ADRs 0010, 0020). Keep it as an explicit **escape hatch for `as_offset`**, whose table
is built from a runtime field so no eager metadata can ever exist.

**Host-mirrored per-row/block bounds.** A summary is not a distinct design — any O(n)
summary is not O(1)-comparable and needs the same handle treatment. Its one real benefit
is that host-side inference needs no `ensure_compile_time_eval`, dodging the `lax.scan`
and `shard_map` caveats below; if that is wanted, **mirror the whole table on host rather
than summarising**, since per-row min/max is not exact (a row straddling `image_range`
without intersecting it is indistinguishable from one that does, so it fails
conservatively — partial support again). A block decomposition
`(row_boundary, vmin, vmax)` exploiting ICON's renumbering would be O(#blocks) and
value-typed, restoring cross-mesh program sharing; it needs a measurement of #blocks on a
real grid and a fix for the straddling gap. Future work.

## Two independent defects, both blocking

Neither is caused by this design; both must be fixed regardless, and the second becomes
unavoidable once narrowing is the norm.

**NaN gradients from skip values.** `_gather_premap` relies on `-1` wrapping around and
masks only later in `_make_reduction`, so under `grad` the sentinel gathers a real element
that flows through the operator's nonlinearities: `grad = [0.25, 0.167, 0.125, nan]`. Fix
is one `where` sanitising the *index* before the gather — which also removes an existing
oddity where the domain-start offset shifts `-1` onto a different wrapped element.

**`neighbor_sum` after a narrowing `premap` is already broken — in NumPy too.**
`_make_reduction` broadcasts the full context table against the narrowed field:
`operands could not be broadcast together with shapes (4,2) (2,2)`. It must restrict the
offset definition to the field's domain before masking — a static slice given the Domain,
so it stays traceable. Related: it masks against the module-level
`common._DEFAULT_SKIP_VALUE` rather than the connectivity's own `skip_value`.

## Risks and open questions

- **Mesh replacement retains old tables.** The table is a jit argument, so the caller
  keeps it alive anyway and aux data adds *zero* incremental retention (a per-rank R2B7
  decomposition is ~25 MB across ~10 tables, shared by all programs). But old compiled
  programs pin old tables, and `jax.clear_caches()` does not release them — only dropping
  the jitted callables does. Matters for ensemble and nested-grid workflows.
- **`ensure_compile_time_eval` is broken under `lax.scan`** in 0.6.2
  (`NotImplementedError: Evaluation rule for 'empty' not implemented`). If gt4py programs
  get wrapped in `lax.scan` for time-stepping this is live, and it is the one concrete
  argument for the host-mirror variant.
- **`shard_map` is deferred, not solved.** icon4py runs MPI with per-rank local index
  spaces and halos (GHEX), not JAX's global-array model; there the handle *is* the
  rank-local table and the inferred domain is rank-local, which is correct. The aux/child
  mismatch only arises under `shard_map` over a JAX global array, and that world raises
  the prior question of what a `Domain` means globally — a larger decision that should not
  drive connectivity representation. Note `ensure_compile_time_eval` under `shard_map`
  produces `Manual` HloShardings that reportedly do not work.
- `restrict()` on a traced table cannot recompute metadata; the constructor must tolerate
  its absence.
- The traced connectivity must be **installed in the embedded context**, not merely passed
  to `premap` — `_make_reduction` reads the offset provider from the contextvar, so the
  design silently degrades to closure constants if a caller forgets.

## Sketch of the change

`common.py` — document that a `Connectivity` may never itself be aux data (`__eq__` raises
by design, producing a confusing `ValueError` from `jaxlib`). Optionally hang the bounds
descriptor off `NeighborConnectivityType`, next to `max_neighbors`.

`nd_array_field.py` — add the buffer-keyed handle; register `JaxArrayConnectivityField`
as a pytree node; route `inverse_image` through the handle; sanitise skip indices in
`_gather_premap`; restrict the offset definition in `_make_reduction` and use the
connectivity's own `skip_value`.

Tests worth pinning: a **narrowing** `premap` under `jit` with the table as an argument
and nothing in `jaxpr.consts`; `neighbor_sum` after a narrowing `premap` (currently
failing on NumPy); and `grad` through a gather with skip values producing no NaNs.

## Related

- [[personal/havogt/mesh-and-first-class-halos|A mesh concept with first-class halos]] —
  overlaps directly; a mesh owning its connectivities is the natural home for any
  precomputed metadata, and owns the local-index-space question deferred above.
- [[personal/havogt/dependent-local-dimensions|Dependent local dimensions and connectivity chains]] —
  local dimensions and reductions consume `premap`'s output.
- [[personal/havogt/scan-redesign|Redesign of the vertical scan]] — same static-metadata
  versus traced-data tension, same `lax.scan` caveat.
- [[personal/havogt/jax-connectivities/jax-connectivities_research|Research appendix]] —
  measurements, prior-art survey, JAX-internals citations.
