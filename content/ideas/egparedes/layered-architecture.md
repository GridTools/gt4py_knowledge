---
title: A layered architecture for gt4py with enforced public/internal decoupling
author: egparedes
tags: [architecture, layering, tach, modularity, public-api, semi-public, internal, packaging, dependencies, refactoring, infrastructure, over-engineering, otf, workflow, dsl, config, allocators, caching, fingerprint, compiled-program, factory-boy, instrumentation, jax, decoupling, migration, tech-debt]
created: 2026-06-15
updated: 2026-07-02
---

> **TL;DR** Give gt4py four explicit, machine-checked layers — **public API →
> core logic → infrastructure → utils** (top depends down, never up) — and move
> implementation under a private `gt4py._internal` package while the public
> package `gt4py.next` becomes a thin re-export facade (JAX's `_src/` pattern).
> **Revision 2, re-baselined on `main` as of 2026-07-02.** Since revision 1,
> gt4py has *already* adopted [tach](https://docs.gauge.sh/) in CI (as a flat
> five-package import DAG) and *already* landed the canonical fingerprinting
> engine, a redesigned two-tier caching contract, and default-on persistent
> translation caching. The target architecture is unchanged; the plan is
> re-based: evolve the existing `tach.toml` toward layers (cycle fixes →
> intra-`next` modules → layers), finish the remaining `otf` debt items
> (plain-code backend builders replacing factory-boy, honest combinator typing,
> the argument-model split, a sanctioned stage-inspection/dump API, cache-miss
> diagnostics, one config, the `Backend → Toolchain` naming), and then do the
> `_internal` moves. Scope is unchanged: restructure `gt4py.next` plus shared
> infrastructure/utils; **`eve` is dissolved** into a semi-public
> general-purpose `gt4py.utils` library and an internal `dsltools` toolkit
> (name retired); **`gt4py.cartesian` stays as-is** and is only repointed at
> the shared infrastructure, foldable into the same layers later.

> **Status**: draft proposal, **revision 2 (2026-07-02)** — supersedes the
> 2026-06-15 draft. A structural/architectural direction, not a merged change.
> Drafted with AI assistance against the current `main` of
> [GridTools/gt4py](https://github.com/GridTools/gt4py) (post-1.1.11).

> **Appendix**:
> [[ideas/egparedes/layered-architecture_research|Current-state analysis & references (rev. 2)]] —
> what changed on `main` since revision 1, the verified dependency structure
> (with corrections to rev. 1's claims), the `otf`/caching/factory-boy/allocator
> catalogs with file paths, the updated dependency inventory, and the
> tach/JAX references.

---

## What has already happened (rev. 1 → main)

Revision 1 proposed nine simplification items and nine implementation steps.
Roughly a third has since landed upstream — partly matching the proposal,
partly overtaking it. Re-baselining honestly:

| Rev. 1 item | Status on `main` (2026-07-02) |
| --- | --- |
| Step 1: land `tach`, permissively | **Done, differently**: flat `exact = true` DAG over the five top-level packages, enforced in pre-commit/CI, socially institutionalized in `AGENTS.md`. No layers yet; `forbid_circular_dependencies` commented out (blocked by the `cartesian ⇄ storage` cycle). |
| Simplification 1: one canonical fingerprint | **Largely done, and better**: new `next/fingerprinting.py` engine (strict/lenient fingerprinters, content-based, xxhash); old `compilation_hash` / `fingerprint_compilable_program` deleted; `CachedStep` redesigned (`in_memory`/`persistent`, step+input fingerprint halves, cache-version salt); persistent translation caching **on by default** (the `*_cached` backend variants were deleted); backends excluded from program fingerprints via field metadata. Residuals: a strict/lenient mismatch in `CachedStep.persistent`, a vestigial `key_function`, cartesian untouched (appendix §5.3). |
| Simplification 2: stage-inspection / dump API | **Not done** — no observer, no `GT4PY_DUMP_STAGES`; the dace `program.py` duck-typed reach-in survives verbatim. `Backend.compile()` now exists as a sanctioned AOT entry, which helps. |
| Simplification 3: `explain_cache_misses` | **Not done** — all cache layers miss silently. |
| Simplification 4: plain-code builders, drop factory-boy | **Not done** — 7 `factory.Factory` classes, stringly `__`-path overrides; but `make_dace_backend()` already wraps the factory in the proposed plain-function shape, and pyproject now pins factory-boy as mypy-broken. Counter-current: `roundtrip.py` carries `TODO: introduce factory`. |
| Simplification 5: honest combinator typing | **Marginal** — `StepSequence`'s inner `__Steps` wrapper removed; all 8 combinator classes remain; `SkippableStep` now has **zero users**; `NamedStepSequence` still derives step order by annotation reflection. |
| Simplification 6: argument-model split | **Not done** — `CompileTimeArgs` still carries the runtime `offset_provider` (in-tree `TODO(havogt)`, blocked on the temporaries pass) and the suspected-dead `column_axis` (in-tree TODO). eval/exec codegen **grew** (`compiled_program.py`, `named_collections.py`). |
| Simplification 7: one config | **Not done** — `next/config.py` and `cartesian/config.py` fully separate, divergent env-var namespaces. |
| Simplification 8: allocator registry | **Partially moot** *(rev. 1 overcounted)*: the module has 2 Protocols + 6 `TypeIs` guards and a `device_allocators` registry **already exists**. The real remaining issue: `__gt_allocate__(domain: common.Domain, …)` keeps allocators core-bound. |
| Simplification 9: naming pass (`Backend → Toolchain`) | **Not done**, but now *agreed in-tree*: `backend.py:143` carries `TODO(tehrengruber): Rename … Backend -> Toolchain`. |
| (unforeseen) | **New subsystems** rev. 1 never placed: `otf/compiled_program.py` (`CompiledProgramsPool`: JIT/AOT variants, static-argument specialization, async compilation, instrumentation hooks), `otf/options.py`, `named_collections.py`, the public `gtx.typing` namespace, `instrumentation/hook_machinery`. The removal of the `decoration` step (pipeline is now `translation → bindings → compilation`, with per-backend `CompilationArtifact.load()`) moved the code *toward* the staged-API end state. |

Two structural facts from rev. 1 also needed correction (appendix §3): the
`backend ↔ ffront` relationship is a **strict one-way layering**
(`decorator → backend → ffront-lowering`), not a cycle — the genuine guarded
cycles are `iterator.runtime ⇄ backend`, `common ⇄ ffront.fbuiltins`,
`otf.stages ⇄ otf.definitions`, and `cartesian ⇄ storage`; and the "four type
systems with ~0 sharing" was overstated — `iterator/type_system` genuinely
extends `next/type_system`; the real duplication is `ScalarKind` vs
`DTypeKind` plus cartesian's separate world.

## Problem / motivation

gt4py remains **two frameworks in one** (the legacy cartesian GTScript stack
and the modern `next` field-operator stack) on a shared foundation (`eve`,
`_core`, `storage`). The adoption of `tach` froze the *inter-package* structure
— cartesian and `next` are verifiably decoupled now, and new cross-package
edges require a reviewed `tach.toml` change. That solves decay at the top
level. What it does not solve:

- **No public/internal boundary.** Everything is importable.
  `gt4py/__init__.py` still exports only `eve` and `storage`;
  `gt4py.next.__init__` re-exports ~100 names — but a downstream project (e.g.
  icon4py) can just as easily reach into `gt4py.next.otf.workflow` or
  `gt4py.next.iterator.transforms`. There is **no deprecation machinery at
  all** (no module `__getattr__` anywhere in the public `__init__`s), so every
  internal rename is a potential downstream break. The public API is still
  whatever people happened to import.

- **Inside `gt4py.next`, structure is convention, not contract.** The whole
  package is a single tach module. The core stack is actually well-ordered
  today (`decorator → backend → lowering → IR`), but nothing prevents a new
  upward import, and the four known cycles are papered with `TYPE_CHECKING`
  guards and in-tree TODOs. `otf` still mixes DSL-aware modules
  (`definitions`, `stages`, `compiled_program`) with DSL-agnostic build
  plumbing (`workflow`, `compilation/`, `binding/`, `code_specs`, `options`) in
  one package — although the split now runs along a *verifiably sharp line*
  (appendix §4.1), it is invisible in the package structure.

- **The same concepts are still implemented twice.** `cartesian/config.py` vs
  `next/config.py` (different env-var namespaces, different cache dirs);
  `cartesian/caching.py` vs the otf fingerprint caches;
  `cartesian/backend/base.py` (ABC) vs `next/backend.py` (dataclass). The
  shared floor (`eve`/`_core`/`storage`) exists but hosts no shared
  *services*.

- **The infrastructure over-engineering shrank but did not go away.** The
  caching/fingerprinting corner is genuinely fixed — one engine, an explicit
  two-tier durability contract, warm starts that skip translation. Still
  standing: factory-boy as the backend-construction framework (now
  acknowledged in-tree as mypy-broken), eight workflow combinator classes for
  what are linear pipelines (one provably dead), Protocol/Generic adapter
  scaffolding and `TypeVar → TypeAlias → Protocol → Generic → class` chains in
  `otf`, no way to observe intermediate stages without duck-typing into
  workflow internals, and caches that miss silently. Meanwhile eval/exec-based
  code generation *expanded* into two more modules — each site individually
  motivated, collectively a growing debuggability tax with no stated policy.

The cost profile is unchanged: slow onboarding, fragile refactors, an unclear
contract for downstream users — plus, new since rev. 1, a live example of what
missing observability costs (the dace `__sdfg__` reach-in) and a live example
of how much cleaner things get when a service is built engine-down
(`fingerprinting`). **The moment is still right**: subsystem reworks are in
flight (see related proposals below), the team already accepted
machine-checked import contracts, and the fingerprinting work demonstrated the
exact core/infrastructure split this proposal generalizes.

## Proposal

Unchanged in substance from revision 1; restated compactly with the updated
placements:

1. **Four conceptual layers** with a strict top-down dependency rule.
2. **A public/internal split**: implementation lives in `gt4py._internal`; the
   public packages become facades. **`tach` — already in the repo — enforces
   both**, evolving the existing flat DAG into layers.

### Scope

- **In scope now:** restructure **`gt4py.next`** (core logic) and extract the
  **shared infrastructure** and **utils** layers it sits on.
- **The `eve` package is dissolved and renamed** (unchanged from rev. 1, and
  the code has started moving this way on its own: the fingerprint helpers
  already left eve for `next/fingerprinting.py`). Its two audiences become two
  packages in the **utils** layer:
  - **`gt4py.utils`** — general Python helpers (collection/iterator utilities,
    typing helpers, exceptions, pattern matching, `content_hash`,
    `singledispatcher`, …) with no gt4py knowledge; an **independent,
    semi-public** library living *outside* `_internal` (no facade), under a
    documented but lighter stability contract, dependency-isolated so it can
    later be extracted wholesale into its own distribution.
  - **`gt4py._internal.dsltools`** — the toolkit for *defining and processing*
    DSL/syntax trees: typed node datamodels, the `Node` base,
    visitors/translators, tree walks, symbol-table traits, template-based
    codegen. Internal only; may import `utils`. The `eve` name is retired.
- **`gt4py.cartesian` stays in its current shape.** Only its imports are
  repointed at the shared infrastructure it has analogues for (config, the
  cmake/ninja build toolchain, file-cache primitives, allocators). Its
  internal layering is deferred; the target is designed so it folds into
  `gt4py._internal.cartesian` later without layer rework.

### The four layers

Top-down; a layer may import from layers **below** it, never above. The table
reflects the July 2026 tree, including the modules that are new or newly
placed since rev. 1 (in bold):

| Layer | Responsibility | May import | Contents (today's modules) |
| --- | --- | --- | --- |
| **public_api** | The supported import surface. Re-exports only; owns deprecations (module `__getattr__`). | core, infra, utils (to re-export) | `gt4py.next`, `gt4py.storage` facades; **`next/typing.py`** (the public typing-only namespace — already facade-shaped) |
| **core** | The gt4py *domain*: field/dimension/domain model, the IRs (FOAST/PAST/ITIR) and passes, type systems, embedded execution, codegens (IR → source), and the runner/orchestration components that drive compile+run. | infrastructure, utils | `next/{common,ffront,iterator,type_system,embedded}`, `next/program_processors/codegens` + runner orchestration, **`otf/{definitions,stages,arguments,options,compiled_program}`** (the DSL-aware side of otf, incl. `CompiledProgramsPool`), **`named_collections`**, **`instrumentation/hooks.py`** (imports `decorator`/`compiled_program`), the per-IR fingerprint *deconstructors* (`iterator/ir.py`, `ffront/stages.py`); plus `gt4py.cartesian` as-is (tagged core, not public_api, until folded in) |
| **infrastructure** | DSL-agnostic *services* that never import an IR: configuration, caching/hashing, the build system (cmake/ninja + importer), C++ bindings, the pipeline/step machinery, device/buffer/allocator management, instrumentation machinery. | utils | `next/otf/{compilation,binding}`, `next/otf/{workflow,toolchain,code_specs,cpp_utils}`, **`next/fingerprinting.py`** (the engine — its only gt4py deps are eve/utils-level), `config`, allocators (`storage` + `next/custom_layout_allocators`, once made domain-agnostic), **`instrumentation/{hook_machinery,metrics,gpu_profiler}`** |
| **utils** | Domain-agnostic foundations with no gt4py knowledge: the semi-public `gt4py.utils` library and the `dsltools` tree toolkit (both from `eve`), plus `defs` (today's `_core`). | (nothing internal) | `gt4py.utils` (semi-public, standalone), `gt4py._internal.dsltools`, `gt4py._internal.defs` (today's `_core`: scalar/device types, `filecache`, `locking`, `ndarray_utils`) |

**The key structural insight is unchanged — and now empirically validated.**
The core/infrastructure cut inside `otf` separates *IR → source text* (core)
from *source text → loadable module* (infrastructure), with runners/toolchain
composition in core so the generic pipeline machinery never imports an IR.
Two things strengthened the case since rev. 1:

- The cut line is now **verifiable and sharp** (appendix §4.1): `workflow`,
  `toolchain`, `code_specs`, `options`, `cpp_utils`, `binding/`,
  `compilation/` already import no IR; only `definitions`, `stages`,
  `compiled_program` (and `recipes`, transitively) are DSL-aware. Extraction
  is relocation, not disentangling — the one intra-`otf` cycle
  (`stages ⇄ definitions`) is the only surgery required.
- **`fingerprinting` is the pattern, proven.** The engine is generic; domain
  knowledge enters only through per-type deconstructor overrides supplied at
  core call sites (`lenient_ir_fingerprinter` in `iterator/ir.py`,
  `semantic_fingerprinter` in `ffront/stages.py`). That is exactly the
  "hashing primitive in infra, `program_def → data` extraction in core"
  contract rev. 1 specified for the cache — implemented before the layer
  exists to host it. The layering gives it (and future services) a home and a
  rule.

One placement is new: **`CompiledProgramsPool`** (`otf/compiled_program.py`)
is the *runner orchestration* component — it sits above `Backend.compile()`,
knows ffront stages and instrumentation, and manages JIT/AOT variant caches.
It is unambiguously **core** (runners), and it is the natural consumer of the
sanctioned stage API below.

### Why four layers (not more, not fewer)

Unchanged from rev. 1, compressed: four captures the two highest-value
boundaries — **public_api ↔ core** (the `_internal` decoupling) and
**core ↔ infrastructure** (the cut that frees the build/caching/allocator
services from the DSL and makes them shareable with cartesian) — with `utils`
as the floor; it matches the JAX reference. Fewer merges a boundary that does
work; more is premature while core-internal cycles remain (tach layers are a
total order — a finer split would fail `tach check` today). The finer
intra-core ordering (`domain → IR → frontend → codegens → runners`) is
expressed as per-module `depends_on` edges *within* the core layer — which the
re-based plan now introduces as its own early step (step 3 below), since the
verified `decorator → backend → lowering` ordering makes it mostly declarative
today. A fifth "domain-model" layer below infrastructure stays wrong for the
same reason as before: `Field`/`Domain`/`Connectivity` *is* DSL knowledge; the
fix is making the few domain-touching services (allocators, cache-key
extraction) take plain data — the latter is already done, the former is item 7
below.

### Public/internal decoupling (`_internal`, the JAX `_src` pattern)

Unchanged. Implementation moves under a private root; the public facades
re-export from it; nothing in `_internal` imports a facade (carve-out:
`gt4py.utils`, which is not a facade). Representative target tree:

```
src/gt4py/
  __init__.py            # facade: re-exports + deprecation __getattr__
  next/__init__.py       # facade -> gt4py._internal.next (+ named runners/allocators)
  next/typing.py         # public typing-only namespace (already exists; stays public)
  storage/__init__.py    # facade -> infra allocator constructors
  utils/                 # layer: utils — standalone, semi-public utility library
                         #   (general helpers from eve; content_hash, singledispatcher, ...)
  cartesian/             # UNCHANGED for now; only repointed at _internal.infra services

  _internal/
    dsltools/   defs/                # layer: utils (internal members)
    infra/                           # layer: infrastructure
      config.py  caching.py  fingerprinting.py  pipeline.py
      build/  bindings/  allocators/  instrumentation/
    next/                            # layer: core
      common/  type_system/  ffront/  iterator/  embedded/
      codegens/  runners/            # runners incl. compiled-programs pool + toolchains
```

Public names stay byte-for-byte stable (`gtx.field_operator`, `gtx.Field`,
`gtx.gtfn_cpu`, `gtx.typing.Program`, `gtx.wait_for_compilation`); only their
provenance moves. Adopt JAX's three conventions (appendix §10): explicit
`from ... import x as x` re-exports; module-level `__getattr__` deprecation
shims (gt4py has none today — this is also the missing tool for retiring
*any* public name, independent of this proposal); concrete types at the
boundary.

### Enforce it with `tach` — by evolving the file that already exists

Rev. 1's "land tach" step is done; the enforcement plan becomes an evolution
path for the in-tree `tach.toml`:

1. **Fix the `cartesian ⇄ storage` cycle** (the in-file TODO) and flip
   `forbid_circular_dependencies = true`.
2. **Declare intra-`next` modules** with `depends_on` edges — making today's
   *de facto* ordering a contract: `ffront.decorator`-level orchestration →
   `backend`/runners → codegens → IR/transforms → `type_system` → `common`,
   with `otf`'s DSL-agnostic half depending on nothing DSL-side. This step
   surfaces the three intra-`next` cycles as explicit debts to pay
   (`iterator.runtime ⇄ backend`, `common ⇄ fbuiltins`,
   `otf.stages ⇄ otf.definitions`).
3. **Switch the top of the file to `layers`** once the moves land:

```toml
# tach.toml — target state (gt4py.next + shared layers; cartesian joins later)
layers = ["public_api", "core", "infrastructure", "utils"]

[[modules]]
path = "gt4py.next"                  # facade
layer = "public_api"
[[modules]]
path = "gt4py.storage"               # facade
layer = "public_api"

[[modules]]
path = "gt4py._internal.next"
layer = "core"
[[modules]]
path = "gt4py.cartesian"             # unrestructured; pinned so it can't reach next internals
layer = "core"

[[modules]]
path = "gt4py._internal.infra"
layer = "infrastructure"

[[modules]]
path = "gt4py.utils"                 # semi-public, standalone, extraction-ready
layer = "utils"
[[modules]]
path = "gt4py._internal.dsltools"
layer = "utils"
depends_on = ["gt4py.utils"]
[[modules]]
path = "gt4py._internal.defs"        # today's _core
layer = "utils"
```

The layer rule does the heavy lifting: upward imports fail `tach check`;
nothing may import the facades; `cartesian` and `_internal.next` sit in the
same layer with no declared edge, so cross-imports are rejected. Ratchet
`strict` per module as moves land; mark old-path shims `deprecated`.

### Clean up and simplify — the re-based catalog

The relocation remains the moment to delete accidental complexity, and the
guiding rule stands: **move first, simplify second** — except for the in-place
`otf` debt, which pays for itself immediately and lands before any move.
Renumbered by current priority; statuses per the table above and appendix
§§4-7.

1. **Finish the fingerprint/caching residuals** *(small; the big item is
   done)*. Fix the `CachedStep.persistent` step-fingerprinter to honor its
   documented strict contract (or fix the docstring — decide which is wrong);
   delete the vestigial `GTFNBackendFactory.Params.key_function`; document the
   two-tier (lenient-in-memory / strict-persistent) contract in one place as
   *the* caching policy. Cartesian adoption of the engine is deferred with the
   rest of cartesian.
2. **A sanctioned stage-inspection / dump API** *(small, high value — now the
   top observability gap)*. Add an optional `on_stage(name, artifact)`
   observer to the pipeline and a `GT4PY_DUMP_STAGES=<dir>` switch writing
   each intermediate artifact (GTIR text, generated source, SDFG JSON) per
   program, MLIR `print-ir-tree-dir` style. Promote a supported
   `Toolchain.lower(definition, compile_time_args) -> stage artifacts` /
   `translate_only` entry so `dace/program.py.__sdfg__` stops duck-typing
   through `backend.executor.translation.step` (the reach-in is quoted in
   appendix §4.8; `Backend.compile()` is the precedent for sanctioned
   entries). The observer treats artifacts opaquely, so infra writes bytes
   without importing IR types — this API *is* the core ↔ infra seam.
3. **`explain_cache_misses` diagnostics** *(small)*. Under a debug flag, each
   cache layer (`CachedStep.cache_key`, the build-folder cache, the
   compiled-programs pool) logs its key, hit/miss, and the first differing key
   component. The unified fingerprint made keys comparable; this makes them
   *inspectable*. Also turn the pool's JIT-disabled miss (`RuntimeError`) into
   an error that says *which* static-args key was missing.
4. **Plain-code backend builders, not factory-boy** *(medium, high value)*.
   Unchanged in substance: a backend/toolchain becomes a frozen dataclass built
   by a `make_*_toolchain(...)` function with ordinary keyword arguments; the
   shared cached-translation/naming/device wiring deduplicates into one helper
   used by gtfn and dace; the `factory` dependency goes away. New evidence
   makes this cheaper and more urgent: `make_dace_backend()` already has the
   right signature (it just delegates to the factory internally — invert it),
   pyproject pins factory-boy as mypy-broken, and `roundtrip.py` already
   builds `Backend(...)` as plain code. **Conflict to resolve explicitly**:
   `roundtrip.py:283` carries `TODO(tehrengruber): introduce factory` — the
   team should decide the direction once, in an ADR, not per-module.
5. **Pipeline, not combinators — honest typing first** *(medium)*. Immediate:
   delete `SkippableStep` (zero users). Then: declare `NamedStepSequence` step
   order as an explicit class-level tuple instead of annotation reflection;
   fold `MultiWorkflow` (one subclass) into its single user or keep it as a
   documented runtime-typed dispatcher. Architectural target when the
   machinery moves to `infra/pipeline`: `Workflow = Callable[[T], U]`, one
   small frozen `Pipeline`, `compose()`, `dataclasses.replace`, and the
   existing `cached(step, key=...)` role played by today's (kept) `CachedStep`.
   The `toolchain.py` adapters (`DataOnlyAdapter`, …) become
   `functools.partial`/3-line wrappers; the `definitions.py` alias chains
   flatten to concrete dataclasses where variance is not exercised.
6. **Tighten the argument model — and set an eval/exec policy** *(medium)*.
   Split `CompileTimeArgs` into a true compile-time part and an
   explicitly-named runtime bridge for the temporaries pass (the in-tree
   `TODO(havogt)` is the blocker to clear); remove `column_axis` after
   confirming the in-tree suspicion that it is dead. New since rev. 1: the
   eval/exec-generated-code technique spread to ~10 sites across
   `compiled_program.py`, `arguments.py`, and `named_collections.py`. Keep the
   measured hot paths (the argument-descriptor cache key), but adopt a stated
   policy — a hot-path allowlist with a comment pointing at the measurement —
   so the technique stops spreading by default.
7. **Domain-agnostic allocators** *(small; re-scoped)*. The registry exists;
   what remains is the layering blocker: have the infra-level allocator take
   shape / aligned-index / device (defs-level data) and let **core** do the
   `Domain → shape` translation before calling it (the JAX buffer-vs-array
   split). Fold the 6 `TypeIs` guards into at most one entry-point check.
8. **One config** *(small)*. Merge `next/config.py` and `cartesian/config.py`
   into a single env-var-driven `infra/config.py`; reconcile the divergent
   namespaces (`GT4PY_*` vs `GT_*`) with deprecation warnings for the old
   variable names.
9. **A naming and docs pass** *(small)*. Execute the in-tree TODO:
   `Backend → Toolchain`, `executor → compile_pipeline` (or the TODO's
   `backend_transforms` — align with the ADR), `transforms →
   frontend_transforms`; document the stage names. Also fix the stale
   architecture docs shipped in-tree: `iterator/ARCHITECTURE.md` describes a
   layout (`fencil_processors/`, `closure()`) that no longer exists, and
   `iterator/README.md` is empty — misleading for exactly the
   contributors/agents the new `AGENTS.md` files target.
10. **Pay down the genuine cycles** *(small each; prerequisite for layers)*.
    `iterator.runtime ⇄ backend` (in-tree TODO exists), `common ⇄
    ffront.fbuiltins`, `otf.stages ⇄ otf.definitions`, and `cartesian ⇄
    storage` (the `tach.toml` TODO). Each removal is independently landable
    and unlocks `forbid_circular_dependencies = true` and, later, the layer
    declarations.

None of these change behavior or performance; items 1-3 build the
observability/diagnostics foundation that both the layering and any deeper
staged-compilation redesign reuse.

## Implementation steps

Facade-first, as before: at every step old import paths keep working via
shims, the full test suite passes, and `tach check` passes. Each step is an
independently mergeable PR. Step 0 records what is already done so the plan
tracks reality:

0. ~~**Land `tach`.**~~ **Done** (flat DAG, CI-enforced).
   ~~**One canonical fingerprint.**~~ **Done** (engine + two-tier contract +
   default-on persistent translation caching).
1. **Cycle fixes + `forbid_circular_dependencies = true`.** Fix
   `cartesian ⇄ storage`, then the three intra-`next` cycles (item 10). Flip
   the flag in `tach.toml`. *(Small PRs, config + targeted refactors.)*
2. **Finish the in-place `otf`/infra debt** — items 1-9 above, high-value
   first (2, 3, 4), each its own PR. None blocks the others; all are
   independent of the moves and are not re-entrenched by them.
3. **Declare intra-`next` tach modules.** Encode today's verified ordering
   (`decorator/orchestration → backend/runners → codegens → IR → type_system →
   common`; otf split by DSL-awareness) as `depends_on` edges. From here on,
   the internal structure cannot silently regress while the moves proceed.
   *(Config-only PR; possible because step 1 removed the cycles.)*
4. **Create the skeleton.** Standalone `gt4py/utils/` + empty
   `_internal/{dsltools,defs,infra,next}`.
5. **Extract the `utils` layer.** 5a: `_core` → `_internal/defs` (shim at old
   path). 5b: dissolve `eve` → `gt4py.utils` (general helpers, incl.
   `content_hash`/`singledispatcher`) + `_internal/dsltools` (datamodels,
   `Node`, visitors, trees, traits, codegen); drop the public `gt4py.eve`
   export (one-release deprecation shim); update in-tree imports. The
   fingerprint-helpers precedent shows eve pieces move out cleanly.
6. **Build the shared `infra` layer — one service per PR**, each with a shim
   and with cartesian repointed where it has an analogue: 6a config (item 8);
   6b build/cache/bindings from `next/otf/{compilation,binding}` (+ the
   `fingerprinting` engine, whose per-IR deconstructors stay in core); 6c
   pipeline from `next/otf/workflow.py`+`toolchain.py` (where item 5's
   collapse completes); 6d allocators with the domain-agnostic signature
   (item 7); 6e instrumentation machinery (`hook_machinery`, `metrics`;
   `hooks.py` stays core), hosting the `on_stage` observer from item 2.
7. **Move `gt4py.next` core under `_internal.next`**, sub-package by
   sub-package (`common` → `type_system` → `iterator` → `ffront` → `embedded`
   → `codegens` → runners incl. `compiled_program`), each behind a shim.
8. **Turn the public `__init__`s into real facades** (explicit re-exports +
   `__getattr__` deprecation shims). Public import names unchanged.
9. **Tighten:** flip `tach` to layers + per-module `strict`; deprecate and
   later delete shims on a downstream-driven timeline (icon4py).
10. **(Later) Fold in `gt4py.cartesian`** under `_internal/cartesian` (core),
    reusing the infra from step 6 — pure relocation, no layer rework.

## Alternatives considered

- **Lint rules / conventions only (no `_internal` move).** Rejected as in
  rev. 1: internals stay importable and the contract stays implicit. The tach
  DAG now in-tree is exactly this halfway point — valuable, but it protects
  package boundaries, not the public/internal boundary.
- **Import Linter / pydeps / hand-rolled checks instead of `tach`.** Settled
  by events: tach is adopted and institutionalized in-tree.
- **`_src` (JAX) vs `_internal` naming; per-package `_internal` vs one shared
  root.** Unchanged: `_internal`, one shared root.
- **Restructure `cartesian` now vs defer.** Deferred, unchanged.
- **Keep `eve` public vs dissolve it.** Dissolve, unchanged — reinforced by
  eve already shedding its fingerprint helpers to `next` and gaining
  general-purpose utilities (`singledispatcher`) that clearly belong in a
  utils library rather than an "IR framework".
- **A full staged-compilation-API redesign instead of in-place cleanups.**
  Still the attractive larger commitment; still deferred. The gap has
  narrowed from both sides since rev. 1: `decoration`'s removal made the
  pipeline stage-shaped (`translation → bindings → compilation` +
  `CompilationArtifact.load()`), and `Backend.compile()` +
  `CompiledProgramsPool` sketch the runner half. Items 2 and 9 (stage API +
  naming) are deliberately the largest strides toward it that don't require
  the redesign.
- **Wrap factory-boy better instead of removing it** (the `make_dace_backend`
  status quo). Rejected: it hides the stringly `__`-overrides at one call
  site while keeping the framework, the trait duplication, the mypy
  breakage, and the dependency. The wrapper's existence is evidence the plain
  function is the right interface — so make it the implementation.

## Open questions / conflicts

- **Builder direction needs a team decision.** This proposal says *remove*
  factory-boy (item 4); `roundtrip.py:283` says *introduce* it. Both
  positions are in-tree today. Resolve once, in an ADR — per gt4py's own
  `AGENTS.md`, backend-interface changes require one.
- **`CachedStep.persistent` strict/lenient mismatch** (appendix §5.3): code
  or docstring — which is the intent? Worth an upstream issue independent of
  this proposal.
- **Where does `CompiledProgramsPool` sit long-term?** This proposal places
  it in core/runners. If the staged-compilation API is pursued, the pool is
  its natural driver — decide whether the stage API (item 2) should be
  designed against the pool's needs from the start.
- **eval/exec policy** (item 6): which of the ~10 current sites are measured
  hot paths, and who signs off on new ones?
- **`dsltools` naming and granularity; semi-public `gt4py.utils` contract;
  is `ffront` semi-public; shim lifetime.** All unchanged from rev. 1.
- **Type-system consolidation** remains enabled-not-done; the rev. 2 analysis
  sharpens the target: unify `ScalarKind` (next) with `DTypeKind` (`_core`)
  first, since `iterator/type_system` already extends `next/type_system`.
- **Dependency hygiene:** should `dace` (required install dep, lazily
  imported, one backend) become optional, and `factory` be removed (item 4)?
  The `>=2.0.0a4` pin on a pre-release dace strengthens the case for
  optionality.
- **ADR impact.** Items 2, 4, 5, 9 touch ADR 0011 (workflow framework) and
  0017 (toolchain configuration); the root `AGENTS.md` now *requires* ADRs
  for architectural changes, so the landing sequence should include short
  successor ADRs in the gt4py repo — this garden note is not the record.
- **Python floor.** gt4py still supports 3.10 (the anticipated <3.12 drop has
  not happened); nothing in this proposal depends on it, but the facade work
  is a natural moment to schedule the bump.
- **Relation to in-flight `next` proposals — complementary, not conflicting**
  (unchanged): [[ideas/havogt/field-data-protocol|A FieldData protocol for embedded fields]]
  (a layering split one level down, lands inside `_internal/next/embedded`);
  [[ideas/havogt/mesh-and-first-class-halos|A mesh concept with first-class halos]]
  (a public-API redesign — the facade is where such a change is staged);
  [[ideas/havogt/scan-redesign|Redesign of the vertical scan]] (JAX as
  semantic reference, mirroring this proposal's architectural reference).

## Appendices

- [[ideas/egparedes/layered-architecture_research|Current-state analysis & references (rev. 2)]] —
  what changed on `main` between the two revisions; the verified dependency
  structure with rev-1 corrections; the full `otf`, caching/fingerprinting,
  factory-boy, and allocator catalogs with file paths and code snippets; the
  updated third-party dependency inventory; the tach-in-gt4py status and
  layers primer; and the JAX `_src`/facade reference.
