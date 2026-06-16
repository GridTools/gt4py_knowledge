---
title: gt4py Knowledge Base
---

Design ideas and proposals for [gt4py](https://github.com/GridTools/gt4py). Drop
an idea, cross-check it against what already exists, and surface conflicts early.

This index is the map of everything here. Each entry lists **keywords** for the
topics a document discusses — scan them to find overlapping or conflicting ideas.
See `AGENTS.md` in the repository root for how to add a proposal and keep this
index current. (Keep entries and their keywords in sync with each document's `tags`.)

## Accepted

Proposals the group broadly agrees on but that are not yet concrete enough to
implement (may have gaps; may be retired if flaws surface).

_None yet._

<!-- Entry format:
- [[accepted/<slug>|Title]] — keywords: keyword1, keyword2, keyword3
-->

## Ideas

Work-in-progress proposals, organized by contributor.

### havogt

- [[ideas/havogt/dtype-generic-fields|Dtype-generic fields in gt4py.next]] — keywords: type-system, generics, dtype, frontend, foast, past, monomorphization, field-operators, type-checking
- [[ideas/havogt/dimension-generic-fields|Generic dimensions and statically typed staggering]] — keywords: type-system, generics, dimensions, staggering, type-checking, mypy, frontend, foast, monomorphization, unstructured
- [[ideas/havogt/dependent-local-dimensions|Dependent local dimensions and connectivity chains]] — keywords: type-system, dimensions, unstructured, connectivities, local-dimensions, reduction, neighbor-sum, type-checking
- [[ideas/havogt/closure-variable-resolution|Closure variable resolution in gt4py.next]] — keywords: frontend, foast, past, closure-variables, name-resolution, constants, builtins, aliasing, gtir, lowering
- [[ideas/havogt/scan-redesign|Redesign of the vertical scan in gt4py.next]] — keywords: scan, vertical, reduction, boundary-conditions, windows, k-caches, fusion, embedded, jax, frontend
- [[ideas/havogt/mesh-and-first-class-halos|A mesh concept with first-class halos]] — keywords: mesh, halos, unstructured, connectivities, offset-provider, domain-inference, distributed, halo-exchange, prior-art
- [[ideas/havogt/field-data-protocol|A FieldData protocol for gt4py.next embedded fields]] — keywords: fields, domain, data, protocol, embedded, function-fields, boundary-conditions, materialization, lazy, concat_where, origin, prior-art
- [[ideas/havogt/boundary-condition-syntax|Frontend syntax sugar for boundary conditions over concat_where]] — keywords: frontend, foast, boundary-conditions, concat_where, regions, syntax-sugar, if-elif-else, match, embedded, cartesian-parity, dsl-design, prior-art

### egparedes

- [[ideas/egparedes/layered-architecture|A layered architecture for gt4py with enforced public/internal decoupling]] — keywords: architecture, layering, tach, modularity, public-api, internal, packaging, dependencies, refactoring, infrastructure, over-engineering, otf, workflow, config, allocators, caching, fingerprint, instrumentation, jax, decoupling, migration, tech-debt

<!-- Entry format (index only the main proposal; link appendices/implementations
     from within the proposal document itself, not here):
- [[ideas/havogt/<slug>|Title]] — keywords: keyword1, keyword2
-->
