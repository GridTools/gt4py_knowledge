---
title: A layered architecture for gt4py with enforced public/internal decoupling
author: egparedes
tags: [architecture, layering, tach, modularity, public-api, internal, packaging, dependencies, refactoring, infrastructure, over-engineering, otf, workflow, config, allocators, jax, decoupling]
created: 2026-06-15
---

> **TL;DR** Give gt4py four explicit, machine-checked layers — **public API →
> core logic → infrastructure → utils** (top depends down, never up) — and move
> *all* implementation under a private `gt4py._internal` package while the public
> packages (`gt4py`, `gt4py.next`, `gt4py.cartesian`, …) become thin re-export
> facades (JAX's `_src/` pattern). Enforce the layering with
> [tach](https://docs.gauge.sh/usage/layers/) in CI. Along the way, cut the
> over-engineered *infrastructure* code (the 5-class `otf` workflow combinators,
> factory-boy backend builders, the allocator Protocol/`TypeIs` zoo, the duplicated
> `config`/caching) down to plain callables, dataclasses, and a small registry.

> **Status**: draft proposal — a structural/architectural direction, not a merged
> change. Drafted with AI assistance against the current `main` of
> [GridTools/gt4py](https://github.com/GridTools/gt4py).

> **Appendix**:
> [[ideas/egparedes/layered-architecture_research|Current-state analysis & references]] —
> the concrete dependency tangles, the infrastructure over-engineering catalog
> (with file paths), a `tach` layers primer, the JAX `_src` reference, and the
> third-party dependency inventory.

---

## Problem / motivation

gt4py has grown into **two frameworks in one** (the legacy cartesian GTScript
stack and the modern `next` field-operator stack) sharing a foundation (`eve`,
`_core`, `storage`). After years of growth the package layout no longer
communicates *intent*, and nothing prevents the layout from decaying further:

- **No public/internal boundary.** Everything is importable. `gt4py/__init__.py`
  exports only `eve` and `storage`; `gt4py.next.__init__` exports ~60 builtins,
  the decorators, and the runners — but a user (or a downstream project like
  icon4py) can just as easily reach into `gt4py.next.otf.workflow` or
  `gt4py.next.iterator.transforms.pass_manager`. The "public API" is whatever
  people happened to import, so **every internal refactor risks breaking
  someone**. The only privacy signal is an underscore prefix, applied
  inconsistently.

- **Dependencies point in every direction.** `next/backend.py` imports
  `next/ffront/`, while `ffront/decorator.py` imports `backend` and `otf` — a
  cycle papered over with `TYPE_CHECKING` guards. The `otf` package mixes
  *DSL-agnostic build plumbing* (cmake/ninja, caching, bindings) with *IR-aware
  translation steps*, so "compilation infrastructure" and "core IR" are
  impossible to separate by import.

- **The same concepts are implemented two-to-four times.** `cartesian/config.py`
  vs `next/config.py`; `cartesian/caching.py` vs `next/otf/compilation/cache.py`;
  `cartesian/backend/base.py` vs `next/backend.py`; and **four** overlapping type
  systems (`_core/definitions.py`, `next/type_system/`,
  `next/iterator/type_system/`, `cartesian/type_hints.py`). There is almost no
  code shared between cartesian and next, and no structural rule that *would*
  encourage sharing.

- **The infrastructure layer is over-engineered.** The `otf` pipeline is built
  from five generic, variance-annotated workflow classes and chain/replace mixins
  to express what is a *linear* `translate → bind → compile → decorate` pipeline;
  backends are assembled with factory-boy `LazyAttribute`/`Trait`/`SubFactory`
  trees to populate plain dataclasses; allocators are guarded by six `Protocol`s
  plus ~ten `TypeIs` functions around two magic methods. (Full catalog with file
  paths in the [[ideas/egparedes/layered-architecture_research|research appendix]].)

The cost is paid every day: slow onboarding, fragile refactors, accidental
coupling, and an unclear contract for downstream users. **Now** is a good time
because gt4py is already dropping <3.12 Python support and reworking subsystems
(see the related `next` proposals below) — a structural baseline makes those
reworks cheaper and keeps the public surface stable while internals move.

## Proposal

Two changes, applied together and kept honest by a tool:

1. **Four conceptual layers**, with a strict top-down dependency rule.
2. **A public/internal split**: implementation lives in `gt4py._internal`; the
   public packages are facades. Then **`tach` enforces both** in CI.

### The four layers

Top-down; a layer may import from layers **below** it, never above. (This is the
ordering the task asks for; it also matches JAX: thin public API over a
tracing/transform core, over lowering/compilation services, over generic utils.)

| Layer | Responsibility | May import | Contents (today's modules) |
| --- | --- | --- | --- |
| **public_api** | The supported import surface. Re-exports only; defines deprecations. | core, infra, utils (to re-export) | `gt4py`, `gt4py.next`, `gt4py.cartesian`, `gt4py.eve`, `gt4py.storage` (the `__init__` facades) |
| **core** | The gt4py *domain*: field/dimension/domain model, the IRs (FOAST/PAST/ITIR; GTIR/OIR) and their passes, type systems, embedded execution, code generators (IR → source text), and runners that *orchestrate* compile+run. | infrastructure, utils | `next/{common,ffront,iterator,type_system,embedded}`, `next/program_processors/codegens`, the runner orchestration; `cartesian/{frontend,gtc,backend,stencil_*}` |
| **infrastructure** | DSL-agnostic *services* that know nothing about IR: configuration, caching/hashing, the build system (cmake/ninja + importer), C++ bindings, the pipeline/step machinery, device/buffer/allocator management, instrumentation. | utils | `next/otf/{compilation,binding}`, `next/otf/workflow`, `config`, allocators (`storage` + `next/custom_layout_allocators`), `next/instrumentation` |
| **utils** | Domain-agnostic foundations with no gt4py knowledge. | (nothing internal) | `eve` (datamodel/visitor/codegen framework), `_core` (scalar/device types, `filecache`, `locking`, `ndarray_utils`), pure-Python helpers |

**The key structural insight** is the core/infrastructure cut *inside* `otf`.
Today `otf` is one package that both knows the iterator IR (translation steps)
and runs cmake. Split it:

- **IR → source text** (codegen) is **core**: it understands the IR.
- **source text → loadable module** (build, bindings, cache, the generic
  pipeline that chains steps) is **infrastructure**: it manipulates strings,
  files, and callables and never imports an IR node.
- A **runner** is **core**: it holds the IR + arguments and *calls* the
  infrastructure build service.

This single cut dissolves the `backend ↔ ffront` cycle (infrastructure stops
depending on the DSL) and makes the cartesian/next *infrastructure* genuinely
shareable without forcing their *cores* to merge.

### Public/internal decoupling (`_internal`, the JAX `_src` pattern)

All implementation moves under a private root; the public packages re-export from
it and **nothing in `_internal` imports a public package**. Representative target
tree (illustrative, not line-exact):

```
src/gt4py/
  __init__.py            # facade: re-exports + deprecation __getattr__
  next/__init__.py       # facade -> gt4py._internal.next (+ named runners/allocators)
  cartesian/__init__.py  # facade -> gt4py._internal.cartesian
  eve/__init__.py        # facade -> gt4py._internal.eve     (framework kept importable)
  storage/__init__.py    # facade -> infra allocator constructors

  _internal/
    utils/   eve/   defs/            # layer: utils   (defs = today's _core)
    infra/                           # layer: infrastructure
      config.py  caching.py  pipeline.py
      build/   bindings/   allocators/   instrumentation/
    next/                            # layer: core
      common/  type_system/  ffront/  iterator/  embedded/  codegens/  runners/
    cartesian/                       # layer: core
      gtscript/  frontend/  gtc/  backend/  stencil/
```

Public names stay **byte-for-byte stable** — `gtx.field_operator`, `gtx.Field`,
`gtx.gtfn_cpu`, `gt4py.eve.NodeVisitor` — only their *provenance* moves under
`_internal`. Concretely, adopt JAX's three conventions (see appendix):

- **Explicit re-exports**: `from gt4py._internal.next.ffront.decorator import field_operator as field_operator`
  in the facade (the `as name` form marks it public for type checkers).
- **Module-level deprecation shims** via `__getattr__` in the facade, so
  relocating or retiring a symbol emits a warning instead of an `AttributeError`.
- **Concrete types at the boundary** (no speculative generics in the public
  signatures).

### Enforce it with `tach`

A single `tach.toml` at the repo root declares the four layers and pins each
package to one. Higher layers may import lower; the reverse fails `tach check`
(run it in CI alongside mypy/ruff). Sketch:

```toml
# tach.toml — gt4py layered architecture
# Ordered high -> low: a layer may import only from layers listed after it.
layers = ["public_api", "core", "infrastructure", "utils"]

# --- public_api: facades only; nothing depends on these ---
[[modules]]
path = "gt4py"
layer = "public_api"
[[modules]]
path = "gt4py.next"
layer = "public_api"
[[modules]]
path = "gt4py.cartesian"
layer = "public_api"
[[modules]]
path = "gt4py.eve"
layer = "public_api"
[[modules]]
path = "gt4py.storage"
layer = "public_api"

# --- core: the two frontends are siblings and must NOT import each other ---
[[modules]]
path = "gt4py._internal.next"
layer = "core"
[[modules]]
path = "gt4py._internal.cartesian"
layer = "core"

# --- infrastructure: DSL-agnostic services ---
[[modules]]
path = "gt4py._internal.infra"
layer = "infrastructure"

# --- utils: domain-agnostic foundations ---
[[modules]]
path = "gt4py._internal.eve"
layer = "utils"
[[modules]]
path = "gt4py._internal.defs"
layer = "utils"
[[modules]]
path = "gt4py._internal.utils"
layer = "utils"
```

The layer rule does the heavy lifting (no `depends_on` edge is declared *into*
`public_api`, and neither core sibling lists the other, so a `next ↔ cartesian`
or `infra → core` import is rejected). `tach` can also be adopted incrementally:
start permissive, ratchet to `strict` per module as the moves land.

### Simplify the infrastructure while moving it

The relocation is the moment to delete accidental complexity. Each item below is
a service that, once in the `infra` layer, should be *small* (details and current
file paths in the appendix):

1. **Pipeline, not combinators.** Replace `otf/workflow.py`'s five classes
   (`NamedStepSequence`, `MultiWorkflow`, `StepSequence`, `CachedStep`,
   `SkippableStep`) and the chain/replace mixins with `Workflow = Callable[[T], U]`,
   one small frozen `Pipeline` dataclass that calls its steps in order, and a
   `compose()` helper. Use `dataclasses.replace` instead of a `.replace()` mixin;
   make caching one `cached(step, key=...)` wrapper instead of a class hierarchy.
2. **Builders, not factory-boy.** Drop the `factory` dependency; replace
   `*WorkflowFactory` (`LazyAttribute`/`Trait`/`SubFactory`) with plain builder
   functions and `dataclasses.field(default_factory=...)`. A `cached: bool`
   parameter replaces a `Trait`.
3. **A registry, not a Protocol zoo.** Replace the six allocator `Protocol`s and
   ~ten `TypeIs` guards in `custom_layout_allocators.py` with one allocator
   registry keyed by device type (`register(device, allocator)` / `get(device)`).
4. **One config.** Merge `next/config.py` and `cartesian/config.py` into
   `infra/config.py` (single env-var-driven module); both frontends read it.
5. **One hashing contract.** Collapse the two fingerprinting paths
   (`compilation_hash` using `hash()` vs `fingerprint_compilable_program` using
   `content_hash()`) into one documented function in `infra/caching.py`.
6. **Concrete types.** Flatten the 5-level `TypeVar → TypeAlias → Protocol →
   Generic → class` chains in `otf/definitions.py` to concrete dataclasses, and
   inline the `toolchain.py` adapter dataclasses (`DataOnlyAdapter`, …) into
   `functools.partial`/small wrappers.

None of these change behavior or performance; they remove indirection that the
layering makes unnecessary (e.g. generic variance existed largely to thread one
pipeline through many call sites — a single `Pipeline` type needs none of it).

## Alternatives considered

- **Lint rules / conventions only (no `_internal` move).** Cheapest, but it does
  not *decouple* anything: internals stay importable and the API contract stays
  implicit. Rejected — the whole point is a checkable boundary.
- **`_src` naming (JAX) vs `_internal`.** Functionally identical; this proposal
  uses `_internal` (clearer to newcomers; the task asks for it). Equivalent under
  `tach`.
- **Per-package `_internal` (`next/_internal`, `cartesian/_internal`) vs one
  shared `gt4py._internal` root.** Per-package keeps moves local but re-fragments
  the shared infrastructure and makes the `tach` mapping noisier. Recommended: the
  **shared root**, so each layer is one top-level package under `_internal` and
  cross-frontend infra/utils live once.
- **Merge the cartesian and next pipelines now.** Out of scope and risky. The
  layering *enables* sharing the infrastructure layer later without forcing a core
  merge; that can proceed independently once the boundary exists.
- **Import Linter / pydeps / hand-rolled checks instead of `tach`.** Viable, but
  `tach`'s first-class *layers* feature expresses "higher-may-import-lower"
  directly and runs fast in CI. (Comparison in the appendix.)

## Open questions / conflicts

- **`eve` / `_core` placement.** Both are placed in **utils** (domain-agnostic).
  But `eve` is currently a *publicly used* framework (re-exported from
  `gt4py/__init__.py`, imported widely downstream). Keep a public `gt4py.eve`
  facade over `gt4py._internal.eve`? Or treat `eve` as a semi-public utils package
  with its own narrower contract? Open.
- **Is `ffront` semi-public?** Some advanced users build on FOAST/PAST stages.
  Decide whether a curated slice is promoted to public_api or stays core.
- **Migration sequencing.** *Facade-first* (add re-export shims at today's paths,
  then move files behind them) keeps every import working throughout and is the
  recommended order; *move-first* is shorter but breaks downstreams in one step.
- **Dependency hygiene.** Should `dace` and `factory` leave the core dependency
  set (→ optional / removed respectively)? The layering makes `dace` a pluggable
  infra/core backend and removes `factory` entirely (item 2 above).
- **Relation to in-flight `next` proposals — complementary, not conflicting.**
  These all assume the *public* `gt4py.next` import names, which the facade keeps
  stable, so this proposal is a substrate beneath them rather than a competitor:
  - [[ideas/havogt/field-data-protocol|A FieldData protocol for embedded fields]] —
    also a *layering* split, one level down (Field = domain layer + data
    protocol); lands naturally inside `_internal/next/embedded`.
  - [[ideas/havogt/mesh-and-first-class-halos|A mesh concept with first-class halos]] —
    a public-API redesign (replacing `offset_provider`); the facade is exactly
    where such a surface change should be staged.
  - [[ideas/havogt/scan-redesign|Redesign of the vertical scan]] — uses JAX
    (`jax.lax.scan`) as its semantic reference, mirroring this proposal's use of
    JAX as the *architectural* reference.

## Appendices

- [[ideas/egparedes/layered-architecture_research|Current-state analysis & references]] —
  detailed dependency tangles, the full infrastructure over-engineering catalog
  with file paths and code snippets, a `tach` layers primer with a worked
  `tach.toml`, the JAX `_src`/facade reference, and the third-party dependency
  inventory (which subsystems pull `dace`, `cupy`, `jax`, `gridtools-cpp`,
  `nanobind`, `factory`, …).
