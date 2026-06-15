---
title: A layered architecture for gt4py with enforced public/internal decoupling
author: egparedes
tags: [architecture, layering, tach, modularity, public-api, internal, packaging, dependencies, refactoring, infrastructure, over-engineering, otf, workflow, config, allocators, caching, fingerprint, instrumentation, jax, decoupling, migration, tech-debt]
created: 2026-06-15
---

> **TL;DR** Give gt4py four explicit, machine-checked layers — **public API →
> core logic → infrastructure → utils** (top depends down, never up) — and move
> implementation under a private `gt4py._internal` package while the public
> package `gt4py.next` becomes a thin re-export facade (JAX's `_src/` pattern).
> Enforce the layering with [tach](https://docs.gauge.sh/usage/layers/) in CI.
> Along the way, cut the over-engineered *infrastructure* code down to plain
> callables, dataclasses, and a small registry, and **pay down the `otf`
> compilation-pipeline debt**: one canonical fingerprint, a sanctioned
> stage-inspection/dump API, honest combinator typing, plain-code backend builders
> (no factory-boy), a tightened argument model, and one config. **Scope for the first steps:** restructure
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

- **The infrastructure layer is over-engineered, and the `otf` compilation
  pipeline most of all.** `otf` is a generic, statically-typed function-composition
  framework (`Workflow[StartT, EndT]`, `NamedStepSequence`, `CachedStep`, …) that
  met its original goal — backends share the binding/build/import stages instead of
  being monoliths — but accreted concrete friction:
  - The promised **static type safety is largely illusory**: five variance-annotated
    workflow classes and chain/replace mixins express what is a *linear*
    `translate → bind → compile → decorate` pipeline.
  - **Configuration goes through a second framework** (`factory_boy`) with
    stringly-typed `__`-path overrides, trait duplication, and `type: ignore`s.
  - **Caching is spread over four uncoordinated layers with three different key
    derivations** — including `id()`-based offset-provider hashing — and no
    documented contract for which key is used when.
  - There is **no sanctioned way to inspect or re-run intermediate stages**, so
    consumers (e.g. the dace runner's `program.py`) resort to duck-typed unwrapping
    of workflow internals.
  - Allocators are guarded by six `Protocol`s plus ~ten `TypeIs` functions around
    two magic methods. (Full catalog with file paths in the
    [[ideas/egparedes/layered-architecture_research|research appendix]].)

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
| **core** | The gt4py *domain*: field/dimension/domain model, the IRs (FOAST/PAST/ITIR) and their passes, type systems, embedded execution, code generators (IR → source text), and runners that *orchestrate* compile+run. | infrastructure, utils | `next/{common,ffront,iterator,type_system,embedded}`, `next/program_processors/codegens`, the runner orchestration; plus **`gt4py.cartesian` as-is** (user-facing but tagged *core*, not public_api, because it is the whole implementation rather than a thin facade — until it is split in step 9). |
| **infrastructure** | DSL-agnostic *services* that know nothing about IR: configuration, caching/hashing, the build system (cmake/ninja + importer), C++ bindings, the pipeline/step machinery, device/buffer/allocator management, instrumentation. | utils | `next/otf/{compilation,binding}`, `next/otf/workflow`, `config`, allocators (`storage` + `next/custom_layout_allocators`), `next/instrumentation` |
| **utils** | Domain-agnostic foundations with no gt4py knowledge: general Python helpers, plus a toolkit for defining/processing IR trees (datamodels, node base + traversal, codegen). | (nothing internal) | `utils` (general helpers) and `irtools` (IR/AST toolkit) — **both carved out of today's `eve`**, whose name is retired; `defs` (today's `_core`: scalar/device types, `filecache`, `locking`, `ndarray_utils`) |

**The key structural insight** is the core/infrastructure cut *inside* `otf`.
Today `otf` is one package that both knows the iterator IR (translation steps)
and runs cmake. Split it:

- **IR → source text** (codegen) is **core**: it understands the IR.
- **source text → loadable module** (build, bindings, cache, the generic
  pipeline that chains steps) is **infrastructure**: it manipulates strings,
  files, and callables and never imports an IR node.
- A **runner** is **core**: it drives compilation and then executes the result.
  The per-backend object that bundles the compile pipeline (today `Backend`,
  renamed `Toolchain` in the cleanup below) is likewise **core**: it *composes*
  core codegen steps with infra build/pipeline/allocator services. Because that
  composition happens in core and the generic `Pipeline` only ever receives
  callables, **infrastructure can host the pipeline without importing any core
  step** — which is the whole point of the cut.

This cut removes the dependency that runs *through* the infrastructure:
`ffront`'s import of `otf` becomes an import of `infra` (downward, legal), and the
generic build/pipeline services stop importing any IR, so **infrastructure no
longer depends on the DSL**. The residual `backend ↔ ffront` coupling becomes a
*core-internal* concern (both live in `_internal.next`), addressed by the
sanctioned `lower()` boundary (below) rather than by the layer rule alone. The cut
also makes the cartesian/next *infrastructure* genuinely shareable without forcing
their *cores* to merge.

### Why four layers (not more, not fewer)

Four is deliberate, not inherited. It captures the two highest-value boundaries —
**public_api ↔ core** (the `_internal` decoupling) and **core ↔ infrastructure**
(the cut that takes the DSL→build dependency out and makes infra shareable) — with
`utils` as the uncontroversial floor. It also matches the reference: JAX is
effectively public (`jax/`) → core/`_src` transforms → lowering/compilation →
utils.

- **Fewer drops a boundary that is doing work.** Merging public_api into core
  defeats the decoupling outright; merging infrastructure into utils loses the
  core/infra cut *and* conflates DSL-agnostic gt4py *services* (build, allocators,
  cache) with domain-agnostic *generic Python* (`irtools`, helpers) — different
  audiences, different churn.
- **More, as global layers, is premature.** The tempting refinement splits `core`
  into a stack (`runners → codegens → frontend → IR → domain`). But `tach` layers
  are a **total order**, and `gt4py.next`'s core still has cycles (the
  `backend ↔ ffront` one, and likely `type_system ↔ common`, `ffront ↔ iterator`).
  A finer *layer* split would fail `tach check` until those are refactored — out of
  first-steps scope. Express that finer ordering instead as per-module `depends_on`
  edges *within* the `core` layer (`codegens → iterator`, `runners → codegens,
  infra`, …) and promote a sub-stack to a real layer only once its cycles are paid
  down.
- **A fifth "domain-model" layer below infrastructure would be wrong.** It is
  tempting because a few infra services touch domain vocabulary (see below), but
  the domain model (`Field`/`Dimension`/`Domain`/`Connectivity`) *is* DSL knowledge
  — it cannot sit beneath a layer defined as DSL-agnostic. The right fix is the
  reverse: keep the domain model in core and make those infra services
  domain-agnostic (next section, items 1 and 8).

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
# Cross-layer (downward) imports need no declaration; same-layer ones do, so any
# intra-utils edge is named explicitly (here just irtools -> utils; add others,
# e.g. defs -> utils, only if the move turns one up).
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

### Clean up and simplify the infrastructure (especially the `otf` pipeline)

The relocation is the moment to delete accidental complexity. Most of it lives in
the `otf` compilation pipeline, and most can be paid down **in place** — each item
below is independently landable and behavior-preserving (details and file paths in
the appendix). Items **1–4 are high value at low risk and worth doing first**; they
also build foundations (one fingerprint, a stage-inspection seam) that the layering
and any deeper redesign reuse, so they never need re-doing.

1. **One canonical fingerprint** *(small, high value)*. Replace the multiple
   cache-key derivations with a single content-based
   `fingerprint(program_def) -> str` — GTIR fingerprint, argument types,
   offset-provider *types*, backend-relevant config, gt4py version — used by the
   translation cache, the executor cache step, and (via a `fingerprint → build-dir`
   index) the build cache, so warm starts skip translation entirely. Replace the
   `id()`-based offset-provider hashing with content hashing of the provider *type*
   plus an identity fast path. Lands in `infra/caching` — and so must be
   **domain-agnostic**: core extracts the type descriptors (arg types,
   offset-provider types) and passes plain hashable data down; `infra/caching`
   hashes data, never importing `common`/connectivity types (otherwise it would be
   an illegal infra→core edge).
2. **A sanctioned stage-inspection / dump API** *(small, high value)*. Add an
   optional `on_stage(name, artifact)` observer to the pipeline and a
   `GT4PY_DUMP_STAGES=<dir>` switch that writes each intermediate artifact (GTIR
   text, generated source, SDFG JSON) per program (MLIR `print-ir-tree-dir` style).
   Promote the lowering entry point to a supported method —
   `Toolchain.lower(definition, compile_time_args) -> stage artifacts` — so
   consumers stop duck-typed-unwrapping pipeline internals. This sanctioned
   `lower()` boundary is exactly the **core ↔ infrastructure seam** (IR → source vs
   source → module): it improves observability *and* realizes the layering.
3. **`explain_cache_misses` logging** *(small)*. Under a debug flag, each cache
   layer logs its key, hit/miss, and the first differing key component — turning
   "why did this recompile?" into a one-line answer.
4. **Plain-code backend builders, not factory-boy** *(medium, high value)*. A
   backend becomes a frozen dataclass built by a `make_*_toolchain(...)` function
   with ordinary keyword arguments (a `cached: bool` flag replaces a `Trait`). This
   removes the trait duplication, the stringly-typed `__`-path overrides and the
   `type: ignore`s, lets the shared "cached translation / naming / device wiring" be
   deduplicated into one helper used by both the gtfn and dace backends, and drops
   the `factory` runtime dependency.
5. **Pipeline, not combinators — honest typing first.** Make the combinators honest
   in place: drop the unused generality (`SkippableStep`, rarely-used chain
   combinators), declare each pipeline's step order as an explicit class-level tuple
   instead of via type-annotation reflection, and document the sequence as
   runtime-typed. The architectural target, reached when the pipeline moves into
   `infra/pipeline`, is to collapse the remaining classes
   (`NamedStepSequence`, `MultiWorkflow`, `StepSequence`, `CachedStep`) to
   `Workflow = Callable[[T], U]` + one small frozen `Pipeline` + `compose()`, with
   `dataclasses.replace` instead of a replace-mixin and one `cached(step, key=...)`
   wrapper.
6. **Tighten the argument model and the types** *(medium)*. Split `CompileTimeArgs`
   into a true compile-time part (types, descriptors, offset-provider *types*) and
   an explicitly-named bridge for the passes that still need runtime tables; remove
   the dead `column_axis`; restrict `eval`/`exec`-based codegen to the one measured
   hot path (the argument-descriptor cache key) and use ordinary code elsewhere.
   Then flatten the `otf/definitions.py` type-alias chains
   (`TypeVar → TypeAlias → Protocol → Generic → class`) to concrete dataclasses and
   inline the `toolchain.py` adapter dataclasses (`DataOnlyAdapter`, …) into
   `functools.partial`/small wrappers.
7. **One config** *(small)*. Merge `next/config.py` and `cartesian/config.py` into a
   single env-var-driven `infra/config.py` that both frontends read.
8. **An allocator registry, not a Protocol zoo** *(small)*. Replace the six
   allocator `Protocol`s and ~ten `TypeIs` guards in `custom_layout_allocators.py`
   with one registry keyed by device type (`register(device, allocator)` /
   `get(device)`), keeping at most one Protocol for the call signature. Moving this
   into `infra` requires making it **domain-agnostic**: today
   `__gt_allocate__(domain: common.Domain, …)` takes core vocabulary, which would
   be an illegal infra→core edge. Have the infra allocator take a shape /
   aligned-index / device (defs-level data) and let **core** do the `Domain → shape`
   translation before calling it — dumb generic allocator below, domain-aware
   wrapper above (the JAX buffer-vs-array split).
9. **A naming pass** *(small)*. Align the vocabulary with the (eventual) ADR: the
   per-backend object `Backend → Toolchain`, `executor → compile_pipeline`, and the
   docstring stage names. (Throughout this proposal, *toolchain* is that per-backend
   compile-pipeline object, while a *runner* is the core component that drives a
   toolchain to compile and then execute a program.)

None of these change behavior or performance; they remove indirection the layering
makes unnecessary (e.g. the generic variance existed largely to thread one pipeline
through many call sites — a single `Pipeline` needs none of it). Items 1–4 remove
the friction that makes the pipeline "hard to follow, hard to extend"; items 5–9
reduce the number of concepts a contributor must hold.

## Implementation steps

Adopt the architecture **facade-first**: at every step the old import paths keep
working (via thin re-export shims), so the existing gt4py test suite is a complete
regression guard and each step is an independently mergeable, reviewable PR. The
two invariants checked on every step are **(a)** the full test suite passes and
**(b)** `tach check` passes. A guiding rule keeps reviews easy: **move first,
simplify second** — a pure relocation behind a shim has a behaviour-empty diff; the
simplification then lands as its own small, readable PR. The one deliberate
exception is step 2: the `otf` cleanups are independent of the layering and pay for
themselves immediately, so they land *before* the moves (and the relayering keeps
them — they are not re-entrenched).

1. **Land `tach`, permissively.** Add `tach` to the dev/CI dependencies and a
   `tach.toml` that declares the four layers and tags modules *at their current
   paths*, non-strict. CI now reports layer violations without blocking. No code
   moves. *(Config-only PR.)*
2. **Pay down the `otf` compilation-pipeline debt in place** — independent of the
   layering, each its own PR, high-value items first: the one canonical
   fingerprint; the `lower()` + `GT4PY_DUMP_STAGES` stage-inspection API;
   `explain_cache_misses` logging; plain-code backend builders (drop `factory_boy`);
   honest combinator typing; the argument-model split; and the naming pass
   (`Backend → Toolchain`, `executor → compile_pipeline`). None blocks the others.
   Done now, step 5 then relocates already-clean code. *(Maps to the simplification
   items above; items 1–4 there are the ones to do first.)*
3. **Create the `gt4py._internal` skeleton.** Add empty layer packages
   (`_internal/{utils,irtools,defs,infra,next}`) with `__init__.py` only. No moves
   yet; establishes the destination and the `tach` module paths.
4. **Extract the `utils` layer.**
   - 4a. Move `_core` → `_internal/defs`; leave `gt4py._core` as a re-export shim.
   - 4b. Dissolve `eve`: move its general Python helpers (collection/iterator,
     typing, exceptions, pattern matching) → `_internal/utils`, and its IR/AST
     toolkit (datamodels, `Node` base, visitors, trees, traits, codegen) →
     `_internal/irtools`. Because `eve` is **internal**, the public `gt4py.eve`
     export in `gt4py/__init__.py` is dropped (optionally a one-release deprecation
     shim re-exporting from the two new locations); update all in-tree `gt4py.eve`
     imports to the new paths. The split can land as two sub-PRs (`utils` first,
     then `irtools`, which depends on it).
5. **Build the shared `infra` layer — one service per PR**, each leaving a
   re-export shim at the old path *and* updating both `next` and `cartesian` to the
   new import form (this is the only change cartesian receives):
   - 5a. **config** → `infra/config.py`; `next/config.py` and `cartesian/config.py`
     become shims that re-export it. *(Simplification 7: unify into one module.)*
   - 5b. **build / cache / bindings** from `next/otf/{compilation,binding}` →
     `infra/{build,bindings,caching}`; shim `next/otf/...`. The canonical
     `fingerprint` built in place in step 2 moves here. Cartesian is repointed only at the shared
     build toolchain and file-cache primitives; its higher-level `CachingStrategy`
     (which caches `StencilObject`s, not compiled programs) is left as-is for now.
   - 5c. **pipeline** from `next/otf/workflow.py` → `infra/pipeline.py`; this is
     where the honest-but-still-class-based combinators from step 2 collapse to
     `Workflow = Callable` + `Pipeline` + `compose`.
   - 5d. **allocators** (`storage` + `next/custom_layout_allocators`) →
     `infra/allocators` with a device registry; shim old paths; repoint
     `gt4py.storage` constructors. *(Simplification 8: drop the Protocol/`TypeIs`
     zoo.)*
   - 5e. **instrumentation** → `infra/instrumentation`; shim. (The `on_stage`
     observer from step 2 is the natural hook surface here.)
6. **Move `gt4py.next` core under `_internal.next`, sub-package by sub-package**
   (`common` → `type_system` → `iterator` → `ffront` → `embedded` →
   `program_processors/codegens` → runners), each leaving a shim at the old path so
   nothing breaks mid-migration. The dependency direction is now enforceable:
   codegen/runners (core) call `infra` build services via the sanctioned `lower()`
   seam; `infra` imports no IR, so the DSL→infrastructure dependency is gone and any
   residual `backend ↔ ffront` coupling is contained within the core layer.
7. **Turn `gt4py/next/__init__.py` into the real facade.** Replace its internals
   with explicit `from gt4py._internal.next... import x as x` re-exports plus a
   `__getattr__` deprecation shim. Same for `gt4py/__init__.py` and
   `gt4py/storage/__init__.py`. Public import names are unchanged.
8. **Tighten and clean up.** Flip `tach` layers/modules to `strict` as each move
   lands; mark the old-path shims `deprecated` in `tach` and on a removal timeline;
   delete shims once downstreams (e.g. icon4py) have migrated.
9. **(Later, out of first scope) Fold in `gt4py.cartesian`.** Move its
   frontend/IR/backends under `_internal/cartesian` (core layer) using the same
   facade-first pattern. No layer changes are needed — cartesian already consumes
   the shared `infra` from step 5, so this is purely relocation + a `cartesian`
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
  of the shared infra (step 5) is low-risk and is the prerequisite that lets it be
  folded in later (step 9) without any layer rework.
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
- **A full staged-compilation-API redesign instead of the in-place cleanups.** The
  fuller alternative replaces the generic workflow framework with an explicit
  staged API: typed stage objects (`Program → LoweredProgram → CompiledArtifact →
  Executable`), a two-method toolchain protocol, one fingerprint, one artifact
  store, and instrumentation at every stage boundary. It is attractive — and the
  core/infra cut plus the `lower()` seam already point straight at it — but it is a
  larger commitment. This proposal takes the incremental path: the cleanups
  (steps 2/5) are valuable on their own and remain correct even if the staged
  redesign is later pursued, so they are not wasted work either way. The staged API
  is the natural follow-on once the boundary and a single fingerprint exist.
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
- **Type-system consolidation is enabled, not done.** The four overlapping type
  systems are *placed* by the layering (`_core/definitions` → `defs`/utils;
  `next/type_system` and `iterator/type_system` → core; `cartesian/type_hints`
  stays with cartesian) but not *merged*. The shared `utils`/`defs` floor makes a
  later consolidation possible; it is deliberately out of scope for the first steps.
- **Shim lifetime.** How long do the old-path re-export shims (steps 4–7) stay,
  and on what deprecation timeline? Driven mainly by downstream (icon4py) migration.
- **When to promote a `core` sub-layer.** The finer intra-core ordering
  (`domain → IR → frontend → codegens → runners`) is carried as per-module
  `depends_on` edges within the single `core` layer for now, because the existing
  core cycles (`backend ↔ ffront`, etc.) would make a finer *tach layer* split fail.
  Once those cycles are paid down (a natural step-8 activity), decide whether to
  promote a sub-stack — most plausibly the `domain` model or the `codegens` — to a
  real layer.
- **How far to collapse the combinators.** The honest-typing cleanup (step 2) and
  the full `Callable` + `Pipeline` collapse (step 5c) are two points on one path.
  The conservative endpoint keeps `NamedStepSequence` as a documented runtime-typed
  sequence; the aggressive one removes it. Decide whether the move goes all the way
  or stops at "honest".
- **ADR impact.** The factory-boy removal and the naming pass touch gt4py's ADR
  0011 (the `otf` workflow framework) and ADR 0017 (toolchain configuration);
  landing them implies a short successor ADR in the gt4py repo recording the change,
  not only this garden note.
- **Dependency hygiene.** Should `dace` and `factory` leave the core dependency
  set (→ optional / removed respectively)? The layering makes `dace` a pluggable
  infra/core backend and removes `factory` entirely (simplification 4 above).
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
