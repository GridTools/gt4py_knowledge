---
title: gt4py Knowledge Base
---

Design ideas and proposals for [gt4py](https://github.com/GridTools/gt4py). Drop
an idea, cross-check it against what already exists, and surface conflicts early.

This index is the map of everything here. Each entry lists **keywords** for the
topics a document discusses — scan them to find overlapping or conflicting ideas.
See `AGENTS.md` in the repository root for how to add a proposal and keep this
index current. (Keep entries and their keywords in sync with each document's `tags`.)

## Shared

Proposals the group broadly agrees are implementation-ready.

- [[shared/external-memory-for-dace-arrays/external-memory-for-dace-arrays|External workspace memory for DaCe temporary arrays]] — keywords: dace, backend, gpu, memory, temporary-arrays, workspace, cuda, hip, mempool, persistent, external, allocation, icon4py, performance

<!-- Entry format:
- [[shared/<slug>|Title]] — keywords: keyword1, keyword2, keyword3
-->

## Personal

Work-in-progress proposals, organized by contributor.

### havogt

- [[personal/havogt/dtype-generic-fields|Dtype-generic fields in gt4py.next]] — keywords: type-system, generics, dtype, frontend, foast, past, monomorphization, field-operators, type-checking
- [[personal/havogt/dimension-generic-fields|Generic dimensions and statically typed staggering]] — keywords: type-system, generics, dimensions, staggering, type-checking, mypy, frontend, foast, monomorphization, unstructured
- [[personal/havogt/dependent-local-dimensions|Dependent local dimensions and connectivity chains]] — keywords: type-system, dimensions, unstructured, connectivities, local-dimensions, reduction, neighbor-sum, type-checking
- [[personal/havogt/closure-variable-resolution|Closure variable resolution in gt4py.next]] — keywords: frontend, foast, past, closure-variables, name-resolution, constants, builtins, aliasing, gtir, lowering
- [[personal/havogt/scan-redesign|Redesign of the vertical scan in gt4py.next]] — keywords: scan, vertical, reduction, boundary-conditions, windows, k-caches, fusion, embedded, jax, frontend
- [[personal/havogt/mesh-and-first-class-halos|A mesh concept with first-class halos]] — keywords: mesh, halos, unstructured, connectivities, offset-provider, domain-inference, distributed, halo-exchange, prior-art
- [[personal/havogt/field-data-protocol|A FieldData protocol for gt4py.next embedded fields]] — keywords: fields, domain, data, protocol, embedded, function-fields, boundary-conditions, materialization, lazy, concat_where, origin, prior-art
- [[personal/havogt/boundary-condition-syntax|Frontend syntax sugar for boundary conditions over concat_where]] — keywords: frontend, foast, boundary-conditions, concat_where, regions, syntax-sugar, if-elif-else, match, embedded, cartesian-parity, dsl-design, prior-art, python-versions, peps, piece-algebra, metaclass, replay

### edopao

- [[personal/edopao/refactor-gtir-to-sdfg-lowering|Refactoring the GTIR-to-SDFG lowering visitor]] — keywords: dace, backend, gtir, sdfg, lowering, refactoring, visitor-pattern, translator-registry, readability, maintainability

### egparedes

- [[personal/egparedes/layered-architecture|A layered architecture for gt4py with enforced public/internal decoupling]] — keywords: architecture, layering, tach, modularity, public-api, semi-public, internal, packaging, dependencies, refactoring, infrastructure, over-engineering, otf, workflow, dsl, config, allocators, caching, fingerprint, compiled-program, process-pool, factory-boy, instrumentation, adr, jax, decoupling, migration, tech-debt
- [[personal/egparedes/discretization-independent-fd-syntax|A discretization-independent surface syntax for finite-difference computations]] — keywords: finite-difference, finite-volume, mesh-invariant, structured, unstructured, exterior-calculus, dec, mimetic, de-rham, location-typing, stencil, weight-generation, rbf-fd, gfdm, moment-matching, conservation, field-operators, icon, pace, pmap, fv3, ifs-fvm, arakawa-staggering, c-grid, d-grid, a-grid, dynamical-core, hevi, vertical-solve, elliptic-solve, mpdata, semi-implicit, miura, atlas, prior-art, dsl-design

<!-- Entry format (index only the main proposal; link appendices/implementations
     from within the proposal document itself, not here):
- [[personal/havogt/<slug>|Title]] — keywords: keyword1, keyword2
-->
