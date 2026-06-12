---
title: A mesh concept with first-class halos
author: havogt
tags: [mesh, halos, unstructured, connectivities, offset-provider, domain-inference, distributed, halo-exchange, prior-art]
created: 2026-06-12
---

> **TL;DR** Replace the opaque `offset_provider` dict with a first-class **mesh**
> concept restricted to a small entity vocabulary (Vertex, Edge, Cell), in which
> connectivities are typed relations between entities and **halos are native**.
> Because every connectivity then has a *derivable reach*, the compiler can treat
> required halo depth as a per-entity-type quantity computed by composing
> incidence relations, and **place halo exchanges automatically** instead of
> leaving them hand-coded in granules. Prior art (Atlas, MPAS, PSyclone/LFRic,
> Sieve/DMPlex, Regent) validates every piece of this individually; gt4py would be
> the first to combine the entity-restricted mesh, the algebraic halo, and the
> automatic placement in one DSL.

> **Status**: research / pre-ADR. This document is a *direction*, not yet a
> concrete design. Its job is to record what prior art establishes and to fix the
> design vocabulary, so a real proposal (and eventually an ADR in the gt4py repo)
> can be written against it.

> **Appendices**:
> - [[ideas/havogt/mesh-and-first-class-halos_research|Prior-art survey]] — the
>   verified survey of ~21 primary sources across the three design layers (mesh
>   data structure, topology formalism, compiler placement), with citations.
> - [[ideas/havogt/mesh-and-first-class-halos_icon4py-survey|icon4py constraint survey]]
>   — a detailed audit of `icon4py`'s current connectivity, grid, halo, and
>   exchange machinery: the constraint set any mesh concept must satisfy.

---

## Problem / motivation

In `gt4py.next` today, unstructured topology enters a program through the
**`offset_provider`**: a flat `dict[str, NeighborTable | Dimension]` mapping an
offset name (`"C2E"`, `"V2E2V"`, `"Koff"`, …) to either a materialized neighbor
table or a vertical `Dimension`. This dict is the entire mesh model. It has three
deficiencies that this idea targets:

1. **An opaque table has no derivable reach.** `C2E2C2E2C` is just a 9-column
   `int` array as far as the toolchain is concerned. Nothing records that it is
   `C2E ∘ E2C ∘ C2E ∘ E2C` — a 4-hop composition — so nothing can compute how far
   into a halo a stencil using it reaches. The flexibility of arbitrary named
   tables is *precisely* what blocks halo reasoning.

2. **Halos are not a gt4py concept at all.** In the primary consumer
   ([icon4py](https://github.com/C2SM/icon4py)), halo levels, ownership, and the
   per-dimension partition metadata live in a *parallel* `DecompositionInfo`
   object next to the grid, and halo exchanges are **hand-placed in granule code**
   (mirroring ICON's `sync_patch_array`), including hand-coded comm/compute
   overlap and hand-reasoned "this exchange can be skipped" comments. The compiler
   sees none of it.

3. **Missing domain inference forces a workaround that mutates the mesh.** Because
   the toolchain cannot infer the minimal valid domain for an unstructured access,
   icon4py ships `keep_skip_values=False`, which **rewrites connectivity tables**
   (replacing skip values `-1` by the row maximum) so that computing into a too-
   large domain does not read invalid neighbors. Its own docstring calls this *"a
   workaround to account for the lack of a domain inference for unstructured
   neighbor accesses in GT4Py"*. The mesh concept proposed here is the thing that
   would make that workaround unnecessary.

The structural opportunity: gt4py's field view is **gather-only** — neighbor
access only *reads*; writes always go to the iteration entity. That makes halo
validity a pure *forward dataflow* property, strictly easier to analyze than the
scatter/increment writes that give the strongest prior art (PSyclone/LFRic) its
hardest cases. gt4py also already has exact per-kernel access footprints in the
IR (offset literals in ITIR) — better information than the hand-written kernel
metadata LFRic relies on. Automatic placement is *structurally easier* here than
in any system that has shipped it.

## Proposal

A mesh concept built from four design lessons, each grounded in prior art (see
the [[ideas/havogt/mesh-and-first-class-halos_research|survey]] for sources). The
intent is to fix the vocabulary, not to settle the API.

### 1. Restrict to a small entity vocabulary

Vertex / Edge / Cell (Face), with connectivities as typed relations *between*
those entities. This is well-precedented: **Atlas** (Nodes/Cells/Edges with typed
pairwise adjacency), **MPAS** (a *closed, named* table set — `cellsOnCell`,
`edgesOnCell`, … — proven sufficient for a production atmosphere/ocean dycore),
**UGRID**, and **ESMF** all restrict to ~3 entity classes. icon4py's *entire*
inventory — 14 horizontal offsets — is already within V/E/C. The flexibility
being given up (arbitrary named neighbor tables) is the flexibility that blocks
halo reasoning.

> Nuance from Atlas: its implementation generalizes by *codimension*
> (`facets/ridges/peaks`), so "exactly three entities" is the conceptual model,
> not the class layout. Relevant only if 3D unstructured ever enters scope — then
> name entities by (co)dimension.

### 2. Declare multi-hop connectivities as compositions; keep them materialized

`C2E2CO = id ∪ (C2E ∘ E2C)`, `E2C2V = E2C ∘ C2V`, etc. The **algebra term gives
the compiler the reach** (⇒ required halo per entity type, by image-composition
à la Regent/DMPlex/GrAL); the **materialized table keeps runtime performance** and
— critically — **pins ICON's neighbor ordering**, which is semantically
load-bearing (interpolation coefficients are order-dependent, e.g. the
`icon_edge_order` permutation). A composition declaration must therefore either
fix a deterministic ordering rule *or* accompany the table, not replace it. ("O"
variants = reflexive closure — a one-token algebra annotation.) icon4py already
*derives* these multi-hop tables ad hoc in `grid_manager`; the formalism (Logg's
Build/Transpose/Intersection over a minimal generating set) gives the derivation
a name and a completeness argument.

### 3. Halo depth is per-entity-type, symbolic, and a relation closure

**Not** a single integer on the mesh. ICON's own structure — 2 cell halo lines,
2 vertex lines, 3 edge lines, where the 3rd edge line *exists solely to make C2E
complete in a distributed setup* — is itself the output of such a computation.
The mesh concept should be able to both **(a) verify** that a given precomputed
halo supports a program's footprint, and **(b) derive** the minimal halo for a
set of declared connectivities. PSyclone represents halo depth as a *symbolic*
first-class object (a `HaloDepth` PSyIR expression, not an `int`); its sum-vs-max
composition wart (self-described "ad-hoc fix", open issue
[stfc/PSyclone#2781](https://github.com/stfc/PSyclone/issues/2781)) is the
cautionary tale: **define depth composition algebraically from day one.**

### 4. Ownership metadata: the Atlas triple + execution classes

Per entity, store `(global_index, owner_partition, remote_index)` — the minimal
metadata from which Atlas builds *all* its parallel communication, and exactly
what GHEX (icon4py's backend) consumes via `HaloGenerator.from_gids`. Add halo-
level labels (icon4py's `DecompositionFlag`) and ideally the DUNE/PyOP2 execution
classes (**core / owned / exec-halo / non-exec-halo**). That single classification
is simultaneously: what GHEX needs, what makes comm/compute overlap *structural*
instead of hand-coded (PyOP2's core/owned split exists exactly for this; icon4py
hand-codes it in `diffusion.py`), and what turns "compute into the halo instead of
exchanging" from a fragile convention into a checkable transformation. icon4py's
`DecompositionInfo` already *is* this — it should become part of the mesh rather
than a parallel object.

### The payoff: automatic exchange placement as forward dataflow

With reach (lesson 2), symbolic per-entity depth (lesson 3), and execution classes
(lesson 4), exchange placement becomes a forward dataflow pass. Each field carries
a per-entity **valid-depth**; through each kernel,

```
valid(out) = min over inputs of ( valid(in) ⊖ footprint )
```

and the compiler inserts an exchange when a consumer needs more than is valid, or
*enlarges the output domain* (redundant computation — PSyclone's policy, and
icon4py's current hand-rolled practice of writing to `end_index(HALO_LEVEL_2)`)
when inputs permit. Runtime clean/dirty flags (LFRic) remain useful across program
boundaries and conditional call paths — they are exactly icon4py's hand-written
"skippable exchange" cases, made automatic.

This also gives **skip values a principled split**: pentagon skips (intrinsic
variable arity — stays), LAM lateral-boundary skips (a *mesh boundary* property —
stays), and halo edge-of-reach skips (an *artifact* of missing domain inference —
**eliminated**, and with them the `keep_skip_values=False` table-rewriting
workaround).

## Alternatives considered

- **Keep `offset_provider`, add a side-table of reaches.** Bolting reach metadata
  onto opaque tables without an entity model re-creates PSyclone's #2781 situation:
  ad-hoc arithmetic with no algebra to fall back on. Rejected as the *direction*,
  though it may be a migration step.
- **Adopt the uniform-DAG representation (Sieve/DMPlex) wholesale.** Treating all
  entities of all dimensions as undifferentiated "points" buys dimension-
  independence gt4py does not need, at a memory and genericity cost it does not
  want. The *algebraic* view (halo = image of the owned set under a composed
  incidence relation) is the keeper; the uniform representation is not.
- **Derive-and-build halos on demand (Atlas `BuildHalo`, t8code).** For ICON
  compatibility the halo is *given* and the compiler verifies/splits it. On-demand
  growth is attractive for non-ICON meshes and falls out of lesson 3(b) — but it is
  a second mode, not the primary one. See open questions.
- **Leave exchanges hand-placed (status quo).** This is what icon4py does today and
  it works, but it is exactly the hand-coded overlap, hand-reasoned skips, and
  table-rewriting workaround this idea aims to retire.

## Open questions / conflicts

- **Fixed vs on-demand halo.** ICON-style capped precomputed halo (compiler
  verifies + splits + exchanges) vs Atlas/t8code on-demand growth. Should the
  concept support *both*? Derive-and-build falls out of lesson 3(b) for free.
- **Exchange granularity.** GHEX today exchanges *all* halo lines of a dimension at
  once; per-level (depth-parameterized, LFRic-style) exchange needs per-level
  patterns — worth the pattern-setup cost?
- **Where does the valid-depth dataflow live** — embedded (runtime checks on a
  mesh-aware field), compiled backends (an ITIR pass), or both?
- **Ordering.** Can a deterministic ordering rule for composed connectivities
  reproduce ICON's orderings (`icon_edge_order`), or must ordering remain a property
  of the materialized table (forcing "composition + table", not "composition
  replaces table")?
- **Known hard spots to design for** (from icon4py): non-contiguous halos in the LAM
  case (halo ∩ lateral-boundary points); the max-rank ownership convention on cut
  lines (must match ICON bit-for-bit for verification runs); the Fortran granule
  path where halo levels are unavailable and start/end indices come from ICON
  directly; nproma-padded buffers larger than the mesh.
- **Keep the vertical out of the mesh.** `Koff`/`KHalfOff` are structured-offset
  concepts riding in the `offset_provider` dict today; the mesh concept is
  *horizontal topology*. (Same for `LsqUnkDim`-style LOCAL dims and sparse *data*
  fields over connectivity dims — they reference the mesh's local dimensions but are
  field-type, not mesh, concerns.)

### Relation to other proposals in this garden

- [[ideas/havogt/dependent-local-dimensions|Dependent local dimensions and connectivity chains]]
  — strongly overlapping. That proposal makes a *local* dimension carry the
  connectivity it derives from (`V2EDim` depends on `Vertex`) and gives chained
  reductions a typed order. This mesh idea is the *runtime/distributed* half of the
  same picture: where dependent-local-dimensions types the connectivity in the field
  type system, the mesh concept gives those same connectivities a **reach** and the
  halos that reach implies. A composed connectivity like `edge_field(V2E)(E2V)` there
  is a composed incidence relation here — the two should share one algebra.
- [[ideas/havogt/dimension-generic-fields|Generic dimensions and statically typed staggering]]
  — the dimension-as-type / typed-connectivity substrate this concept would build on.
