---
title: "Prior art: separating field data from domain in array abstractions"
author: havogt
tags: [fields, domain, data, prior-art, function-fields, materialization, lazy, origin, embedded]
created: 2026-06-12
---

> Background research appendix for
> [[ideas/havogt/field-data-protocol|A FieldData protocol for gt4py.next embedded fields]].
> Indexed only from that proposal, not from the top-level index.

# Prior art: separating field *data* from *domain* in array abstractions

Background research for the [`FieldData` protocol proposal](field-data-protocol)
(2026-06-12). The questions we looked for prior art on:

1. Arrays/fields represented as *functions over an index domain* (function fields,
   infinite domains, boundary conditions).
2. The *delayed vs. manifest* duality and who decides when to materialize.
3. Where the *buffer origin* (index spaces not starting at 0) lives in comparable
   designs.
4. Domain/index-set algebra as a separate layer.

## 1. Arrays as functions over an (infinite) integer domain

**Halide** is the strongest precedent for the semantic model. Halide represents
images as pure functions defined over an *infinite integer domain*; storage, bounds,
and evaluation order are a separate concern (the "schedule"). Boundary conditions are
ordinary combinators that turn a finite image into a function defined everywhere —
`Halide::BoundaryConditions::constant_exterior`, `repeat_edge`, `repeat_image`,
`mirror_image`, `mirror_interior` — exactly the "boundary condition extending to all
points outside the domain" use case motivating our function fields. The
algorithm/schedule split also mirrors our "composition is semantics, materialization
is policy" stance.
Sources: [Halide project page (PLDI 2012)](https://people.csail.mit.edu/jrk/halide12/),
[CACM 2018 paper](http://graphics.stanford.edu/papers/halide-cacm18/halide-cacm18.pdf),
[`BoundaryConditions` API docs](https://halide-lang.org/docs/namespace_halide_1_1_boundary_conditions.html),
[boundary-conditions tests](https://github.com/halide/Halide/blob/main/test/correctness/boundary_conditions.cpp).

**Repa** (Haskell) makes the data representation a *type index* on the array:
delayed `D` arrays are literally `ADelayed sh (sh -> a)` — a shape plus a function
from index to value — while manifest `U`/`B` arrays are concrete memory. All
operations are written once against a `Source` interface (our `materialize`/
`gather_values` analog); `computeS`/`computeP` force a delayed array into a manifest
one. Repa deliberately makes the representation choice *explicit and
programmer-controlled* because compiler heuristics proved fragile — supporting our
explicit policy switch and `materialized()` over implicit magic.
Sources: [Repa delayed representation docs](https://hackage.haskell.org/package/repa-3.4.1.4/docs/Data-Array-Repa-Repr-Delayed.html),
[Repa API docs](https://hackage.haskell.org/package/repa/docs/Data-Array-Repa.html),
[Keller et al., *Regular, shape-polymorphic, parallel arrays in Haskell* (ICFP 2010)](https://www.cs.tufts.edu/~nr/cs257/archive/simon-peyton-jones/RArrays.pdf),
[Repa tutorial](https://wiki.haskell.org/Numeric_Haskell:_A_Repa_Tutorial).

**Pull/push arrays** (Claessen, Sheeran, Svensson — Obsidian/Feldspar line of work):
a *pull array* is a length plus a function from index to value; composition fuses by
construction and `force` is the only way to stop fusion. Push arrays are the dual
(functions that write), interesting for a future `MutableFieldData`. Our composed
pointwise result `lambda x: op(a(x), b(x))` is precisely pull-array fusion.
Sources: [*Expressive array constructs in an embedded GPU kernel programming language* (DAMP 2012)](https://www.researchgate.net/publication/254003957_Expressive_array_constructs_in_an_embedded_GPU_kernel_programming_language),
[Svensson & Svenningsson, *Defunctionalizing Push Arrays* (FHPC 2014)](https://svenssonjoel.github.io/writing/defuncEmb.pdf).

**Accelerate** (Haskell, embedded GPU language) keeps the same delayed/manifest
duality but lets the *compiler* fuse element-wise operations and manifest only at
non-fusible/collective operations — the fully-automatic end of the policy spectrum.
Sources: [Accelerate API docs](https://hackage.haskell.org/package/accelerate/docs/Data-Array-Accelerate.html),
[fusion transformation source](https://hackage.haskell.org/package/accelerate-0.13.0.0/src/Data/Array/Accelerate/Trafo/Fusion.hs).

## 2. Deferred materialization as an explicit user-facing operation

**Dask** arrays are lazy task graphs; nothing materializes until the user calls
`compute()` (bring into local memory) or `persist()` (materialize but keep the
abstraction). The lesson: a successful lazy array API still exposes an explicit
force, both for memory control and to bound re-evaluation — adopted as
`DataField.materialized()`.
Sources: [`dask.array.Array.persist`](https://docs.dask.org/en/stable/generated/dask.array.Array.persist.html),
[managing computation](https://distributed.dask.org/en/stable/manage-computation.html).

## 3. Wrapped/labeled arrays in scientific Python

**xarray** separates the labeled object (`Variable`: dims + attrs) from an inner
duck array, and maintains an internal stack of *lazy indexing* wrapper classes
(`ExplicitlyIndexed`, `LazilyIndexedArray`, `LazilyVectorizedIndexedArray`) that defer
loading/indexing — effectively a private, buffer-oriented "field data" layer. There is
a long-standing wish to extract it as a stand-alone duck-array package
([issue #5081](https://github.com/pydata/xarray/issues/5081)), confirming the demand
for exactly this kind of component. xarray has no concept of an index-space origin
(coordinates are data, positional indexing is always 0-based), so it does not answer
our origin question.
Sources: [xarray internal design](https://docs.xarray.dev/en/latest/internals/internal-design.html),
[`LazilyIndexedArray` docs](https://docs.xarray.dev/en/stable/generated/xarray.core.indexing.LazilyIndexedArray.html),
[`indexing.py` source](https://github.com/pydata/xarray/blob/main/xarray/core/indexing.py).

**Devito** (finite-difference DSL) separates the symbolic `Function` from its `Grid`,
and each `Function`'s storage distinguishes *domain*, *halo*, and *padding* regions —
i.e. the allocated data support is deliberately a superset of the computational
domain, with the offset bookkeeping hidden inside the object. This matches our
"support ⊇ domain, origin is private to the data" rule.
Sources: [*Architecture and performance of Devito*](https://arxiv.org/pdf/1807.03032),
[Devito DSL overview](https://www.devitoproject.org/examples/userapi/01_dsl.html).

## 4. Buffer origin in comparable designs

**OffsetArrays.jl** (Julia) is the direct precedent for non-zero origins: a
lightweight wrapper around any parent `AbstractArray` plus an index offset; indexing
translates wrapper (absolute) indices to parent indices. The offset is entirely
internal to the wrapper — code consuming the array sees only its axes. Julia's
ecosystem experience also documents the cost of *leaking* custom axes into generic
code (libraries assuming 1-based indexing break), reinforcing that the origin should
be encapsulated at exactly one layer.
Sources: [OffsetArrays.jl](https://github.com/JuliaArrays/OffsetArrays.jl),
[internals docs](https://juliaarrays.github.io/OffsetArrays.jl/stable/internals/),
[Julia devdocs on custom indices](https://docs.julialang.org/en/v1/devdocs/offset-arrays/).

**GridTools C++ / SID concept** (in-house): the stencil composition library consumes
fields through the *Stencil Iterable Data* concept — an origin pointer plus strides,
independent of the concrete storage — rather than using the storage library directly;
GT4Py's `__gt_origin__` plays the same role at the Python boundary. Same lesson at a
lower level: computation code is written against an access concept, the
origin/layout lives behind it.
Sources: [GridTools internal documentation](https://gridtools.github.io/gridtools/latest/internal/internal.html),
[GridTools user manual](https://gridtools.github.io/gridtools/latest/user_manual/user_manual.html).

## 5. Domain algebra as a separate layer

The **polyhedral model / isl** manipulates iteration domains as first-class sets of
integer points with affine constraints, fully separated from any data layout — the
mature, general version of our `Domain`/`UnitRange` set arithmetic (and of ADR 22's
open question about unions of boxes, which isl handles natively).
Sources: [Verdoolaege, *isl: An Integer Set Library for the Polyhedral Model*](https://link.springer.com/chapter/10.1007/978-3-642-15582-6_49),
[isl primer](https://www.jeremykun.com/2025/10/19/isl-a-primer/).

## Synthesis: what the prototype adopts (and where it deviates)

| Prior art                       | Lesson adopted in the prototype                                                                                                                         |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Halide                          | fields as functions over an *infinite absolute* integer domain; boundary conditions are just functions; semantics vs. materialization split             |
| Repa / pull arrays              | delayed (`FunctionFieldData`) vs. manifest (`NdArrayFieldData`) behind one source interface; explicit, programmer-controlled forcing (`materialized()`) |
| Accelerate / dask               | materialization as policy (automatic ↔ explicit spectrum; our `EAGER_POINTWISE_ON_FINITE_DOMAINS` default is the conservative point)                    |
| xarray                          | one labeled wrapper over exchangeable inner data; lazy wrappers compose; demand exists for a reusable layer                                             |
| Devito                          | allocated support ⊇ computational domain; offsets are internal bookkeeping                                                                              |
| OffsetArrays.jl / GridTools SID | the origin is a private translation in the data/access layer; never leak it through the interface                                                       |
| isl                             | domain set algebra as an independent, shared layer                                                                                                      |

Deviations worth noting:

- Halide/Accelerate stage programs and *compile* fused pipelines; our embedded
  execution evaluates immediately, so composed chains are unmemoized closures — hence
  the eager-on-finite default and the explicit `materialized()`.
- Pull-array fusion assumes pure element-wise producers; gather (`premap` through
  neighbor tables) is handled by the separate `gather_values` primitive, closer to
  Accelerate's element-wise/collective split.
- None of the surveyed systems combine *named dimensions*, *absolute (possibly
  infinite) per-dimension ranges*, and *exchangeable data representations* in one
  design; the combination appears to be novel, but every ingredient has a precedent.
