---
title: A FieldData protocol for gt4py.next embedded fields
author: havogt
tags: [fields, domain, data, protocol, embedded, function-fields, boundary-conditions, materialization, lazy, concat_where, origin, prior-art]
created: 2026-06-12
---

> **TL;DR** Split the embedded `common.Field` into two layers: **one generic
> field implementation** that owns the `Domain` and does *all* domain
> computations (broadcast, intersection, restrict, premap, `concat_where`
> decomposition) once, for every kind of field, and a **small `FieldData`
> protocol** for the value part with exchangeable implementations (buffer,
> function, lazy, piecewise). A `FieldData` is a mapping from **absolute,
> positional** integer coordinates to values, so the buffer origin collapses
> into a private detail of the buffer implementation and infinite domains
> (boundary conditions extending past the computational domain) become
> representable. This removes the N×N pairwise-dispatch problem between field
> kinds.

> **Status**: draft proposal, validated by a prototype (not merged into the
> public gt4py API). Drafted with AI assistance.

> **Prototype**: vendored alongside this document in
> [`field-data-protocol/`](field-data-protocol/) — a full implementation of the
> `common.Field` protocol (`field_data.py`) cross-checked against today's
> `NdArrayField` on identical inputs (`test_field_data.py`). The source of these
> files lives on the gt4py branch `claude/gt4py-fielddata-protocol-p1q45k` under
> `src/gt4py/next/embedded/field_data.py` and
> `tests/next_tests/unit_tests/embedded_tests/test_field_data.py`.

> **Appendix**:
> [[ideas/havogt/field-data-protocol_research|Prior-art survey]] — separating
> field *data* from *domain* in array abstractions (Halide, Repa, pull/push
> arrays, Accelerate, dask, xarray, Devito, OffsetArrays.jl, GridTools SID, isl).

> **Related**: the infinite-`concat_where` motivation comes from
> [ADR 22 — Limitations of embedded `concat_where`](https://github.com/GridTools/gt4py/blob/main/docs/development/ADRs/next/0022-Limitations-of-embedded-concat_where.md).
> Touches the embedded execution path discussed in
> [[ideas/havogt/scan-redesign|Redesign of the vertical scan in gt4py.next]]
> (boundary conditions, embedded evaluation).

---

# Proposal: A `FieldData` protocol for `gt4py.next` embedded fields

- **Status**: draft proposal (validated by a prototype, not merged into the public API)
- **Authors**: Hannes Vogt (@havogt), drafted with AI assistance
- **Created**: 2026-06-12

## Problem

The original plan for embedded execution was one `common.Field` implementation per
kind of field: today's `NdArrayField` (buffer-backed), plus e.g. a `FunctionField`
(an old, unmerged PR) for fields backed by a function instead of a buffer — needed for
fields with an infinite domain such as a boundary condition extending beyond the
computational domain (the first unsupported case in ADR 22).

This turned out to be the wrong cut: every such implementation re-implements the same
*domain* computations (broadcast, intersection, sub-domain on `restrict`, `premap`
domain logic, `concat_where` decomposition) and only differs in how *values* are
produced. Worse, operations must be defined pairwise between implementations
(`NdArrayField + FunctionField = ?`), which scales quadratically in the number of
field kinds.

## Proposal

Split `Field` into two layers:

- **One generic field implementation** (`DataField` in the prototype) owning the
  `Domain` and performing *all* domain computations, once, for every kind of field.
- **A small `FieldData` protocol** for the value part, with exchangeable
  implementations. A `FieldData` is a mapping from **absolute, positional** integer
  coordinates to values: axes are unnamed (the owning field's `Domain` names and
  orders them), and coordinates are the same integers the `Domain` ranges contain.

```python
class FieldData(Protocol):
    @property
    def ndim(self) -> int: ...
    @property
    def dtype(self) -> core_defs.DType: ...

    def materialize(self, box, xp) -> NDArrayObject:
        """Evaluate on a finite absolute box into an array of the box's shape."""

    def gather_values(self, indices, xp) -> NDArrayObject:
        """Evaluate at absolute coordinates given as one index array per axis."""

    def restrict(self, items) -> FieldData:
        """Fix axes to absolute points (dropping them); ranges are storage hints."""

    def translate(self, offsets) -> FieldData:
        """Precompose with a translation: ``new(x) = old(x + offsets)``."""

    def remap_axes(self, axis_map) -> FieldData:
        """Transpose axes / introduce value-broadcast axes (backs `broadcast`)."""
```

| `Field` operation                      | domain layer (shared)                    | data operation                               |
| -------------------------------------- | ---------------------------------------- | -------------------------------------------- |
| `restrict` / `__getitem__`             | resolve any index spec to absolute items | `restrict`                                   |
| pointwise ops (`+`, `<`, `where`, ...) | dim promotion, broadcast, intersection   | `materialize` or compose via `gather_values` |
| `broadcast`                            | dim bookkeeping, infinite new ranges     | `remap_axes`                                 |
| `premap` (affine / cartesian shift)    | `inverse_image` relabeling               | `translate` (O(1) for buffers)               |
| `premap` (gather / neighbor table)     | output-domain computation, index arrays  | `gather_values`                              |
| `concat_where`                         | mask inversion, piece intersection       | per-piece data, `PiecewiseFieldData`         |
| `ndarray` / `asnumpy` / `as_scalar`    | finiteness check                         | `materialize`                                |

### Absolute coordinates and the buffer origin

The central design question was whether the buffer origin (domains not starting at 0)
is in the way of such a protocol. Answer: **no — provided the protocol speaks absolute
coordinates**, for these reasons:

1. A *relative* (0-based) data protocol cannot represent fields with a domain that is
   unbounded below (boundary condition extending to infinity): there is no finite
   origin to rebase to. Absolute addressing handles finite and infinite domains
   uniformly.
2. The origin does not disappear; it collapses into a private detail of the buffer
   implementation (`NdArrayFieldData.origins`: the absolute coordinate of buffer index
   0, per axis, `None` for value-broadcast axes). No other component ever sees it; the
   domain layer performs pure set arithmetic on `UnitRange`s and never subtracts a
   start. As a consequence, today's parallel computations `sub_domain` (domain
   arithmetic) and `_get_slices_from_domain_slice` (buffer arithmetic) collapse into a
   single absolute-item resolution.
3. The data's support may be a *superset* of the field's domain. `restrict` only
   narrows the domain; narrowing the data is an optional storage optimization, and a
   domain-translation `premap` is an O(1) origin shift.

### Data implementations

`FunctionFieldData` is the semantic ground truth — any field data is behaviorally a
function from coordinates to values — and the other implementations are
storage/performance specializations:

- `NdArrayFieldData` — buffer + private per-axis origins; materializes as views.
- `FunctionFieldData` — vectorized function evaluated on absolute coordinates;
  unbounded support; a *constant* field is just a function ignoring its coordinates
  (a helper, not a class).
- `LazyFieldData` — defers building the data until values are needed; adds memoization
  over the equivalent function encoding (`lambda *coords: factory().gather_values(coords, xp)`) and retains the knowledge "this will be a
  buffer" (needed for capabilities a closure cannot answer: mutability,
  `__gt_buffer_info__`, zero-copy `ndarray`).
- `PiecewiseFieldData` — disjoint absolute boxes, each with its own data; backs
  `concat_where` results whose domain is infinite.

### Operations live on the field, not on the data

The protocol exposes *evaluation primitives*, deliberately not the operations
themselves: operations on the data would need a pairwise answer for every combination
of data kinds — the N×N dispatch problem this refactoring removes from the `Field`
level would reappear one level down. Instead `gather_values` is the universal "call
the field as a function at absolute coordinates" primitive, so any data kind combines
with any other for free (e.g. `premap` of a function field through a neighbor table
required no function-field-specific code in the prototype). Should a data kind ever be
able to combine smarter than evaluate-and-apply (constant folding, fusion), an
optional `combine(op, *others)` hook consulted before the generic path can be added.

### Materialization is a policy, not a property of the data

`a + b` is universally `lambda x: a(x) + b(x)` (a composed `FunctionFieldData`
evaluating the operands via `gather_values`). On infinite domain intersections this is
the only representation; on finite ones, eagerly materializing into a buffer (today's
`NdArrayField` semantics) is a single policy switch in the shared layer
(`EAGER_POINTWISE_ON_FINITE_DOMAINS`, default eager: composed chains are unmemoized
and re-evaluate on every access). An explicit `DataField.materialized()` forces any
field into a buffer-backed one — the `force`/`compute` of delayed-evaluation array
libraries (see prior-art survey).

### `concat_where` with infinite boundary conditions

`concat_where(I < 0, boundary, field)` where `boundary` extends to infinity — the
first unsupported case of ADR 22 — becomes representable: a finite result is eagerly
concatenated into one buffer (today's behavior); an infinite result is a field on an
infinite domain backed by `PiecewiseFieldData`, which restricts, gathers, and
materializes correctly across the piece boundary. Note `PiecewiseFieldData` also hints
at a representation for the "union of boxes" *support* discussed in ADR 22, while the
field's `Domain` itself stays hypercubic.

## Validation (battle tests)

The prototype implements the full `common.Field` protocol and is cross-checked against
`NdArrayField` on identical inputs for: restriction (absolute/relative/mixed indices,
negative origins, no-copy views), pointwise alignment by absolute position,
scalar/reverse operators, dim broadcasting, comparisons, affine and gather `premap`,
finite `concat_where`, and mixed `DataField`/`NdArrayField` expressions (via a no-copy
`as_data_field` adapter). New capabilities are checked against NumPy references:
operations between infinite function fields, gathering a function field through a
neighbor table without materializing it, deferred (lazy) buffer fields with
exactly-once evaluation, the composed-pointwise policy, and `concat_where` with an
infinite boundary condition.

Battle-testing also caught one instructive bug: deriving the affine-`premap` data
shift from domain starts silently fails for infinite domains; the offset must be
probed from the connectivity's `inverse_image`, keeping the computation
domain-start-free.

## Out of scope / open questions

- **Mutability**: `MutableField.__setitem__` needs a writable-buffer capability that
  only buffer-backed data has; likely a separate `MutableFieldData` protocol.
- **Array namespace**: the prototype fixes `xp = numpy` at the field layer; the real
  implementation should carry it per instance (from the data or an allocator), as
  `NdArrayField` subclasses do today.
- **Buffer interop**: `__gt_buffer_info__` / DaCe descriptors as an optional data
  capability.
- **Gather `premap` with infinite output domains** (index arrays are materialized in
  the prototype); a lazy gather is expressible but unimplemented.
- **Connectivities**: `NdArrayConnectivityField` (and `inverse_image`) should be
  rebased onto `FieldData` in the same way; untouched in the prototype.
- **Lazy result dtype** is probed by applying the op to empty arrays, which assumes
  ufunc-like operations.

## Migration sketch

1. Land `FieldData` + `DataField` alongside `NdArrayField`; interop is already
   guaranteed by the `as_data_field` shim (and `DataField` results entering
   `NdArrayField` code paths through `.ndarray`).
2. Re-express `NdArrayField` as `DataField` + `NdArrayFieldData` (per array
   namespace), moving the builtin registry over; delete the duplicated domain logic
   (`_get_slices_from_domain_slice`, broadcast/intersection helpers).
3. Introduce function/constant fields publicly (constructors and/or `concat_where`
   producing them), then revisit ADR 22's restrictions.
