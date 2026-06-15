---
title: A layered architecture for gt4py with enforced public/internal decoupling
author: egparedes
tags: [architecture, layering, tach, modularity, public-api, internal, packaging, dependencies, refactoring, infrastructure, over-engineering, otf, workflow, config, allocators, jax, decoupling, migration]
created: 2026-06-15
---

> **TL;DR** Give gt4py four explicit, machine-checked layers — **public API →
> core logic → infrastructure → utils** (top depends down, never up) — and move
> implementation under a private `gt4py._internal` package while the public
> package `gt4py.next` becomes a thin re-export facade (JAX's `_src/` pattern).
> Enforce the layering with [tach](https://docs.gauge.sh/usage/layers/) in CI.
> Along the way, cut the over-engineered *infrastructure* code (the 5-class `otf`
> workflow combinators, factory-boy backend builders, the allocator
> Protocol/`TypeIs` zoo, the duplicated `config`/caching) down to plain callables,
> dataclasses, and a small registry. **Scope for the first steps:** restructure
> `gt4py.next` and the shared infrastructure/utils; **the `eve` package is
> dissolved into two internal subpackages** — general Python utilities and a
> toolkit for defining/processing DSL/IR trees — with the `eve` name retired (it
> has no out-of-tree users, so neither its name nor its public API is preserved);
> **`gt4py.cartesian` stays as-is** and is only updated to *consume* the new shared
> infrastructure — the design lets it be folded into the same layers later.

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
   public package is a facade. Then **`tach` enforces both** in CI.

### Scope of this proposal

The end state is repo-wide, but the first steps are deliberately bounded so each
is small and reviewable:

- **In scope now:** restructure **`gt4py.next`** (its core logic) and extract the
  **shared infrastructure** and **utils** layers it sits on.
- **The `eve` package is dissolved and renamed.** It is not used outside gt4py,
  and today it bundles two unrelated concerns: general Python helpers, and the
  machinery for *defining and traversing* IR/AST trees. We do **not** preserve its
  name or public API. It is split into two subpackages, both in the **utils**
  layer:
  - **`gt4py._internal.utils`** — general Python utilities (collection/iterator
    helpers, typing helpers, exceptions, pattern matching) with no gt4py knowledge.
  - **`gt4py._internal.irtools`** — the toolkit for *defining and processing*
    DSL/IR trees: typed node datamodels, the `Node` base, visitors/translators,
    tree walks, symbol-table traits, and template-based code generation. (This is
    the *framework for building* IRs — distinct from any concrete IR such as
    `next`'s `iterator.ir` and its `ir_utils` helpers.) `irtools` may import
    `utils`.
- **`gt4py.cartesian` is left in its current shape** for now. The only change to
  it is mechanical: repoint its imports/usage at the **shared internal
  infrastructure** it already has analogues for (config, the cmake/ninja build
  toolchain, file-cache primitives, allocators). Cartesian keeps its own
  higher-level caching strategy and IR/backends untouched; its internal layering is
  **deferred** — but the target architecture is designed so cartesian can be folded
  into `gt4py._internal.cartesian` later **without reworking the layers** (see
  *Implementation steps*, final step).

### The four layers

Top-down; a layer may import from layers **below** it, never above. This top-down
ordering also matches JAX: thin public API over a tracing/transform core, over
lowering/compilation services, over generic utils.

| Layer | Responsibility | May import | Contents (today's modules) |
| --- | --- | --- | --- |
| **public_api** | The supported import surface. Re-exports only; defines deprecations. | core, infra, utils (to re-export) | `gt4py.next`, `gt4py.storage` (the thin `__init__` facades) |
| **core** | The gt4py *domain*: field/dimension/domain model, the IRs (FOAST/PAST/ITIR) and their passes, type systems, embedded execution, code generators (IR → source text), and runners that *orchestrate* compile+run. | infrastructure, utils | `next/{common,ffront,iterator,type_system,embedded}`, `next/program_processors/codegens`, the runner orchestration; plus **`gt4py.cartesian` as-is** (user-facing but tagged *core*, not public_api, because it is the whole implementation rather than a thin facade — until it is split in step 8). |
| **infrastructure** | DSL-agnostic *services* that know nothing about IR: configuration, caching/hashing, the build system (cmake/ninja + importer), C++ bindings, the pipeline/step machinery, device/buffer/allocator management, instrumentation. | utils | `next/otf/{compilation,binding}`, `next/otf/workflow`, `config`, allocators (`storage` + `next/custom_layout_allocators`), `next/instrumentation` |
| **utils** | Domain-agnostic foundations with no gt4py knowledge: general Python helpers, plus a toolkit for defining/processing IR trees (datamodels, node base + traversal, codegen). | (nothing internal) | `utils` (general helpers) and `irtools` (IR/AST toolkit) — **both carved out of today's `eve`**, whose name is retired; `defs` (today's `_core`: scalar/device types, `filecache`, `locking`, `ndarray_utils`) |

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

Implementation moves under a private root; the public package re-exports from it
and **nothing in `_internal` imports a public package**. Representative target
tree (illustrative, not line-exact):

```
src/gt4py/
  __init__.py            # facade: re-exports + deprecation __getattr__
  next/__init__.py       # facade -> gt4py._internal.next (+ named runners/allocators)
  storage/__init__.py    # facade -> infra allocator constructors
  cartesian/            # UNCHANGED for now; only its imports of shared infra are
                        #   updated to gt4py._internal.infra.* (no _internal move yet)

  _internal/
    utils/   irtools/   defs/        # layer: utils
                                     #   utils   = general Python helpers (from eve)
                                     #   irtools = IR/AST toolkit (from eve): datamodels,
                                     #             Node base, visitors, trees, traits, codegen
                                     #   defs    = today's _core
    infra/                           # layer: infrastructure
      config.py  caching.py  pipeline.py
      build/   bindings/   allocators/   instrumentation/
    next/                            # layer: core
      common/  type_system/  ffront/  iterator/  embedded/  codegens/  runners/
    # cartesian/  -> added in a later step, same layering (see Implementation steps)
```

Public names stay **byte-for-byte stable** — `gtx.field_operator`, `gtx.Field`,
`gtx.gtfn_cpu` — only their *provenance* moves under `_internal`. The `eve` package
is the one exception: being internal (no out-of-tree users), it gets no public
facade and is dissolved into `_internal/utils` + `_internal/irtools` with its name
retired. Concretely, adopt JAX's three conventions for the public surface (see
appendix):

- **Explicit re-exports**: `from gt4py._internal.next.ffront.decorator import field_operator as field_operator`
  in the facade (the `as name` form marks it public for type checkers).
- **Module-level deprecation shims** via `__getattr__` in the facade, so
  relocating or retiring a symbol emits a warning instead of an `AttributeError`.
- **Concrete types at the boundary** (no speculative generics in the public
  signatures).

### Enforce it with `tach`

A single `tach.toml` at the repo root declares the four layers and pins each
package to one. Higher layers may import lower; the reverse fails `tach check`
(run it in CI alongside mypy/ruff). The sketch below is the **target** mapping
(after the moves); step 1 lands an equivalent file pointing at *today's* paths and
tightens it as modules move:

```toml
# tach.toml — gt4py layered architecture (target mapping, gt4py.next + shared layers)
# Ordered high -> low: a layer may import only from layers listed after it.
layers = ["public_api", "core", "infrastructure", "utils"]

# --- public_api: facades only; nothing depends on these ---
[[modules]]
path = "gt4py.next"
layer = "public_api"
[[modules]]
path = "gt4py.storage"
layer = "public_api"

# --- core: the gt4py.next domain (cartesian joins here later) ---
[[modules]]
path = "gt4py._internal.next"
layer = "core"

# gt4py.cartesian is NOT restructured yet. Tag it at the core layer so tach still
# forbids it from reaching into gt4py._internal.next and pins its allowed
# dependencies to infra + utils. Its internal sub-layering is deferred.
[[modules]]
path = "gt4py.cartesian"
layer = "core"

# --- infrastructure: DSL-agnostic services shared by next and cartesian ---
[[modules]]
path = "gt4py._internal.infra"
layer = "infrastructure"

# --- utils: domain-agnostic foundations (today's eve, split + renamed) ---
# Cross-layer (downward) imports need no declaration; same-layer ones do, so the
# one intra-utils edge (irtools -> utils) is named explicitly.
[[modules]]
path = "gt4py._internal.utils"      # general Python helpers
layer = "utils"
[[modules]]
path = "gt4py._internal.irtools"    # IR/AST toolkit (datamodels, nodes, visitors, codegen)
layer = "utils"
depends_on = ["gt4py._internal.utils"]
[[modules]]
path = "gt4py._internal.defs"       # today's _core
layer = "utils"
```

The layer rule does the heavy lifting: downward imports (e.g. `core → infra`,
`infra → utils`) are allowed without any `depends_on` declaration, while *upward*
imports fail `tach check`. Two consequences worth calling out: nothing may import
the `public_api` layer (so no internal module can depend on the facades), and
`gt4py.cartesian` and `gt4py._internal.next` sit in the **same** layer with no edge
between them — and tach requires same-layer edges to be explicit — so a
`cartesian ↔ next-internals` import is rejected. (The lone intentional same-layer
edge, `irtools → utils`, is the declared exception above.) `tach` is adopted
incrementally: start permissive, ratchet to `strict` per module as the moves land.

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

## Implementation steps

Adopt the architecture **facade-first**: at every step the old import paths keep
working (via thin re-export shims), so the existing gt4py test suite is a complete
regression guard and each step is an independently mergeable, reviewable PR. The
two invariants checked on every step are **(a)** the full test suite passes and
**(b)** `tach check` passes. A guiding rule keeps reviews easy: **move first,
simplify second** — a pure relocation behind a shim has a behaviour-empty diff;
the simplification then lands as its own small, readable PR.

1. **Land `tach`, permissively.** Add `tach` to the dev/CI dependencies and a
   `tach.toml` that declares the four layers and tags modules *at their current
   paths*, non-strict. CI now reports layer violations without blocking. No code
   moves. *(Config-only PR.)*
2. **Create the `gt4py._internal` skeleton.** Add empty layer packages
   (`_internal/{utils,irtools,defs,infra,next}`) with `__init__.py` only. No moves
   yet; establishes the destination and the `tach` module paths.
3. **Extract the `utils` layer.**
   - 3a. Move `_core` → `_internal/defs`; leave `gt4py._core` as a re-export shim.
   - 3b. Dissolve `eve`: move its general Python helpers (collection/iterator,
     typing, exceptions, pattern matching) → `_internal/utils`, and its IR/AST
     toolkit (datamodels, `Node` base, visitors, trees, traits, codegen) →
     `_internal/irtools`. Because `eve` is **internal**, the public `gt4py.eve`
     export in `gt4py/__init__.py` is dropped (optionally a one-release deprecation
     shim re-exporting from the two new locations); update all in-tree `gt4py.eve`
     imports to the new paths. The split can land as two sub-PRs (`utils` first,
     then `irtools`, which depends on it).
4. **Build the shared `infra` layer — one service per PR**, each leaving a
   re-export shim at the old path *and* updating both `next` and `cartesian` to the
   new import form (this is the only change cartesian receives):
   - 4a. **config** → `infra/config.py`; `next/config.py` and `cartesian/config.py`
     become shims that re-export it. *(Simplification: unify into one module.)*
   - 4b. **build / cache / bindings** from `next/otf/{compilation,binding}` →
     `infra/{build,bindings,caching}`; shim `next/otf/...`. *(Simplification:
     single hashing contract.)* Cartesian is repointed only at the shared build
     toolchain and file-cache primitives; its higher-level `CachingStrategy` (which
     caches `StencilObject`s, not compiled programs) is left as-is for now.
   - 4c. **pipeline** from `next/otf/workflow.py` → `infra/pipeline.py`. *(Move as a
     shim first; then the follow-up PR collapses the 5 classes to
     `Workflow = Callable` + `Pipeline` + `compose`, and removes factory-boy.)*
   - 4d. **allocators** (`storage` + `next/custom_layout_allocators`) →
     `infra/allocators` with a device registry; shim old paths; repoint
     `gt4py.storage` constructors. *(Simplification: drop the Protocol/`TypeIs`
     zoo.)*
   - 4e. **instrumentation** → `infra/instrumentation`; shim.
5. **Move `gt4py.next` core under `_internal.next`, sub-package by sub-package**
   (`common` → `type_system` → `iterator` → `ffront` → `embedded` →
   `program_processors/codegens` → runners), each leaving a shim at the old path so
   nothing breaks mid-migration. The dependency direction is now enforceable:
   codegen/runners (core) call `infra` build services; `infra` no longer imports IR
   — the `backend ↔ ffront` cycle is broken here.
6. **Turn `gt4py/next/__init__.py` into the real facade.** Replace its internals
   with explicit `from gt4py._internal.next... import x as x` re-exports plus a
   `__getattr__` deprecation shim. Same for `gt4py/__init__.py` and
   `gt4py/storage/__init__.py`. Public import names are unchanged.
7. **Tighten and clean up.** Flip `tach` layers/modules to `strict` as each move
   lands; mark the old-path shims `deprecated` in `tach` and on a removal timeline;
   delete shims once downstreams (e.g. icon4py) have migrated.
8. **(Later, out of first scope) Fold in `gt4py.cartesian`.** Move its
   frontend/IR/backends under `_internal/cartesian` (core layer) using the same
   facade-first pattern. No layer changes are needed — cartesian already consumes
   the shared `infra` from step 4, so this is purely relocation + a `cartesian`
   facade.

## Alternatives considered

- **Lint rules / conventions only (no `_internal` move).** Cheapest, but it does
  not *decouple* anything: internals stay importable and the API contract stays
  implicit. Rejected — the whole point is a checkable boundary.
- **`_src` naming (JAX) vs `_internal`.** Functionally identical; this proposal
  uses `_internal` as the clearer name for newcomers. Equivalent under `tach`.
- **Per-package `_internal` (`next/_internal`, `cartesian/_internal`) vs one
  shared `gt4py._internal` root.** Per-package keeps moves local but re-fragments
  the shared infrastructure and makes the `tach` mapping noisier. Recommended: the
  **shared root**, so each layer is one top-level package under `_internal` and
  cross-frontend infra/utils live once.
- **Restructure `cartesian` now vs defer it.** Deferred (this proposal). Doing
  cartesian's internal layering at the same time multiplies the diff and the risk
  for little gain, since cartesian is in maintenance. Updating only its *imports*
  of the shared infra (step 4) is low-risk and is the prerequisite that lets it be
  folded in later (step 8) without any layer rework.
- **Keep `eve` public vs dissolve it.** Dissolved into `_internal/utils` +
  `_internal/irtools` (name retired): it has no known out-of-tree users, and the
  package mixed two concerns (general helpers vs the IR/AST toolkit) that belong in
  distinct subpackages. Keeping the `eve` name (even internally) would preserve that
  conflation and invite confusion with concrete IRs. (If an external user surfaces,
  a thin public facade over the two subpackages can be re-added later — the reverse
  is harder.)
- **Merge the cartesian and next pipelines.** Out of scope and risky. The layering
  *enables* sharing the infrastructure layer without forcing a core merge; that can
  proceed independently once the boundary exists.
- **Import Linter / pydeps / hand-rolled checks instead of `tach`.** Viable, but
  `tach`'s first-class *layers* feature expresses "higher-may-import-lower"
  directly and runs fast in CI. (Comparison in the appendix.)

## Open questions / conflicts

- **`irtools` naming and granularity.** The IR/AST toolkit is named `irtools`
  (general helpers go to `utils`); both sit in the utils layer with
  `irtools → utils`. Residual bikeshed: is `irtools` distinct enough from `next`'s
  `iterator.ir`/`ir_utils`, and is the generic `datamodels` typed-data library
  general enough to belong in `utils` rather than `irtools`? Leaning: keep
  `datamodels` with `irtools` (it is the node-definition substrate), revisit the
  name in review.
- **Is `ffront` semi-public?** Some advanced users build on FOAST/PAST stages.
  Decide whether a curated slice is promoted to public_api or stays core.
- **Shim lifetime.** How long do the old-path re-export shims (steps 3–6) stay,
  and on what deprecation timeline? Driven mainly by downstream (icon4py) migration.
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
