---
title: "Layered architecture — current-state analysis & references (rev. 2)"
author: egparedes
tags: [architecture, layering, tach, infrastructure, over-engineering, otf, workflow, dependencies, dsl, semi-public, jax, caching, fingerprint, compiled-program, factory-boy, research]
created: 2026-06-15
updated: 2026-07-02
draft: false
status: draft
---

> Appendix to [[personal/egparedes/layered-architecture|A layered architecture for
> gt4py with enforced public/internal decoupling]]. Background only — the
> recommendation lives in the main proposal. **Revision 2**: findings re-verified
> against the current `main` of
> [GridTools/gt4py](https://github.com/GridTools/gt4py) as of **2026-07-02**
> (post-1.1.11); paths are under `src/gt4py/`. Where revision 1 (2026-06-15) got
> a fact wrong or the code has since changed, the correction is called out
> explicitly with a **[rev-1 correction]** or **[changed since rev. 1]** marker.

---

## 1. What changed on `main` since the rev. 1 analysis

The repo moved noticeably between mid-June and July 2026. The changes land
squarely on rev. 1's items — partly implementing them, partly overtaking them:

- **`tach` is adopted and CI-enforced** — a `tach.toml` at the repo root
  declares an *exact* import DAG over the five top-level packages
  (`_core`, `eve`, `storage`, `cartesian`, `next`), and a pre-commit hook runs
  `tach check` (`.pre-commit-config.yaml:101-105`; `tach>=0.23.0` in the `lint`
  dependency group). The root `AGENTS.md` institutionalizes it: *new import
  edges require an explicit `tach.toml` edit plus PR justification*. This is
  rev. 1's step 1, in a flat-modules (not layered) form.
  `forbid_circular_dependencies` is present but commented out, with a
  `TODO(egparedes)` naming the `cartesian ↔ storage` cycle as the blocker.
- **A canonical fingerprinting engine landed**: the new
  `next/fingerprinting.py` (756 lines) provides deterministic content-based
  hashing via a generic *deconstruction + catamorphism* design, with two
  canonical instances — `strict_fingerprinter` (cross-process deterministic,
  for persistent caches) and `lenient_fingerprinter` (single-process, for
  in-memory caches). The old ad-hoc pair in `otf/stages.py`
  (`compilation_hash` built on `hash()`, `fingerprint_compilable_program`
  built on eve's `content_hash`) is **gone**. This is most of rev. 1's
  simplification 1 (details and residual issues in §5).
- **`CachedStep` was redesigned** around the engine: explicit
  `input_fingerprinter` + `step_fingerprinter` halves, sanctioned
  `CachedStep.in_memory(...)` / `CachedStep.persistent(...)` constructors, and
  a `BUILD_CACHE_VERSION_ID` salt.
- **Translation caching is now on by default**: the `run_gtfn_cached` /
  `run_gtfn_gpu_cached` backend variants were deleted and the public
  `gtfn_cpu`/`gtfn_gpu` aliases repointed at the plain runners, which now carry
  a persistent (`FileCache`-backed) translation cache keyed by the strict
  fingerprint (`gtfn.py`, `dace/workflow/factory.py`).
- **The `decoration` pipeline step was removed**: `OTFCompileWorkflow` is now
  three steps (`translation → bindings → compilation`, `otf/recipes.py:16-24`),
  and the runtime argument conversion moved into per-backend
  `CompilationArtifact.load()` implementations (structural
  `CompilationArtifact` Protocol in `otf/stages.py:140-156`;
  e.g. `GTFNCompilationArtifact.load()` wraps `convert_args`).
- **Backends are excluded from program fingerprints** via dataclass field
  metadata — `backend: ... = dataclasses.field(metadata=utils.gt4py_metadata(fingerprint=False))`
  in `ffront/decorator.py` — replacing three bespoke `singledispatch`
  fingerprint registrations. The `gt4py_metadata` helper lives in
  `next/utils.py` (`GT4PY_CLASS_METADATA_NS`).
- **eve slimmed slightly and gained engine support**: the pickle-based
  fingerprint helpers (`NodeFingerprinter`, `skipping_fields_node_pickler`,
  `custom_pickler*`) were **removed from `eve/concepts.py`/`eve/utils.py`** and
  superseded by the `next/fingerprinting.py` toolkit; `eve/utils.py` gained
  `get_fully_qualified_name`, `singledispatcher`, `merge_dispatchers` (used by
  the engine) and `content_hash` had its pickle protocol pinned
  (`_CONTENT_HASH_PICKLE_PROTOCOL = 5`) for cross-version reproducibility.
  `datamodels/core.py` gained `DataModelABC` (virtual-subclass hook).
- **Minor honest-typing cleanups**: `StepSequence` lost its inner `__Steps`
  wrapper dataclass (now a plain `steps: tuple[Workflow, ...]`).
- **New agent/architecture docs**: root `AGENTS.md`, `next/AGENTS.md` (module
  map, embedded-vs-compiled modes, conventions), plus
  `iterator/ARCHITECTURE.md` and `iterator/README.md` — note the latter two are
  **stale**: `ARCHITECTURE.md` still describes `fencil_processors/`,
  `closure(...)` and `backend_executor.py` (none exist), and `README.md` is an
  empty heading.

Not moved: no `_internal`, no facades, no eve dissolution, config still
duplicated, factory-boy still the backend-builder mechanism, `Backend` not
renamed, no stage-dump or cache-diagnostics facility. Python floor is still
3.10 (`requires-python = '>=3.10, <3.15, ...'`) — the anticipated <3.12 drop
has not happened yet.

## 2. Current package map (2026-07-02)

```
gt4py/
  __init__.py        # exports only: eve, storage (unchanged; no __getattr__/deprecations)
  _core/             # definitions (DType/Device/Scalar), filecache, locking, ndarray_utils, types
  eve/               # datamodels / NodeVisitor / codegen / traits framework + general utils
  storage/           # TensorBuffer + allocators core; storage/cartesian layout registry
  cartesian/         # legacy GTScript stack (config, caching, frontend/, gtc/, backend/)
  next/              # modern field-operator stack
    common.py  config.py  backend.py  constructors.py  field_utils.py  utils.py
    custom_layout_allocators.py
    fingerprinting.py        # NEW: content-hash engine (strict/lenient fingerprinters)
    named_collections.py     # NEW-ish: eval-codegen extract/construct for tuple/dataclass args
    typing.py                # NEW-ish: public typing-only namespace (gtx.typing)
    type_system/             # type_specifications, type_info, type_translation, mypy_plugin
    ffront/                  # decorator, func_to_foast/past, *_ast, *_passes/, foast_to_gtir, past_to_itir, stages
    iterator/                # ir.py, builtins, embedded, transforms/ (~40 passes), type_system/
    otf/                     # workflow, toolchain, definitions, stages, recipes, arguments,
                             #   code_specs, options, compiled_program, cpp_utils,
                             #   binding/ (cpp_interface, nanobind), compilation/ (cmake, cache, importer)
    program_processors/      # codegens/gtfn, runners/{gtfn,roundtrip,dace}, formatters
    embedded/                # nd_array_field, operators, context
    instrumentation/         # hook_machinery, hooks, metrics, gpu_profiler
    errors/  experimental/
tach.toml                    # NEW: exact flat import DAG over the five packages, CI-enforced
```

Proposed layer assignment (see main proposal): unchanged in shape from rev. 1,
extended to place the modules that are new or newly understood —
`fingerprinting` → *infrastructure* (engine; per-IR deconstructors stay in
core); `otf/compiled_program` + `otf/arguments` + `otf/options` → *core*
(runner orchestration / type-system-aware argument model);
`instrumentation/hook_machinery` + `metrics` → *infrastructure* but
`instrumentation/hooks.py` → *core* (it imports `ffront.decorator` and
`otf.compiled_program`); `named_collections` → *core* (type-system-aware);
`typing` → *public_api*.

## 3. Dependency structure — verified facts

### 3.1 The inter-package DAG is now machine-checked

`tach.toml` (root, `exact = true`): `eve → {}`; `_core → {eve}`;
`storage → {_core, eve}`; `next → {_core, eve, storage}`;
`cartesian → {_core, eve, storage}`. Verified by grep: **zero import edges
between `cartesian` and `next` in either direction** (the only "hits" are a
docstring in `next/common.py` and comments in a dace transformation). The
decay-prevention half of rev. 1's motivation is thus already handled at
package granularity; everything *inside* `gt4py.next` remains unenforced.

### 3.2 `backend ↔ ffront` — **[rev-1 correction]** a layering, not a cycle

Rev. 1 claimed a `backend ↔ ffront` cycle "papered over with `TYPE_CHECKING`
guards". The current code (and closer reading) shows a strict one-way
layering:

- `next/backend.py:16-27` imports the ffront *lowering* submodules
  (`foast_to_gtir`, `past_to_itir`, `stages`, …) plus `iterator.ir` and `otf` —
  unconditionally.
- `next/ffront/decorator.py:29-51` imports `next.backend` and
  `otf.{arguments, compiled_program, options, toolchain}` — unconditionally.
- `backend.py` never imports `decorator` or `compiled_program`; the lowering
  submodules import neither `decorator` nor `backend`.

So the real structure is `decorator → backend → {ffront lowering, otf}` — the
decorator is the top of the core stack, not a cycle participant. Neither
`backend.py` nor `decorator.py` nor `ffront/stages.py` uses `TYPE_CHECKING` at
all.

### 3.3 The genuine cycles (all TYPE_CHECKING/lazy-import guarded)

Only 9 modules in `gt4py.next` use `TYPE_CHECKING`; four sites guard real
cycles:

| Cycle | Guard site | Note |
| --- | --- | --- |
| `iterator.runtime ⇄ backend` | `iterator/runtime.py:28-30` + lazy re-import at `:88` | in-code `TODO(tehrengruber): remove circular dependency` |
| `common ⇄ ffront.fbuiltins` | `common.py:716-717` | low-level domain model referencing frontend builtins |
| `otf.stages ⇄ otf.definitions` | `otf/stages.py:30-33` | intra-`otf`; `definitions.py:15` imports `stages` unconditionally |
| `errors.formatting ⇄ errors.exceptions` | `errors/formatting.py:22-23` | intra-package, benign |

Plus the **`cartesian ⇄ storage`** cycle acknowledged in `tach.toml`'s TODO —
the one blocker for `forbid_circular_dependencies = true`.

### 3.4 Backends: still hard-wired, no registry

`next/__init__.py:97-98` wires `gtfn_cpu`/`gtfn_gpu` (now the non-`_cached`
runners — **[changed since rev. 1]**) and `itir_python`; dace backends are
module-level singletons in `runners/dace/workflow/backend.py`, picked up
lazily in `decorator.py:419-424` via `try: import ... except ImportError`.
`DEFAULT_BACKEND: Backend | None = None` (`decorator.py:54`) selects embedded
execution by default. No central registry; adding a backend still means
editing `__init__`.

### 3.5 Type systems — **[rev-1 correction]** four exist, but two genuinely share

- `_core/definitions.py` — scalar/dtype/device foundation (`DType`,
  `DTypeKind`, `DeviceType`, `NDArrayObject`), used everywhere.
- `next/type_system/` — DSL `TypeSpec` hierarchy **built on eve datamodels**,
  with its own `ScalarKind` enum (distinct from `_core.DTypeKind`;
  `type_translation.py` bridges them).
- `next/iterator/type_system/` — **extends** `next/type_system`
  (`type_specifications.py:12` imports and subclasses `ts.TypeSpec`/`DataType`)
  — real sharing, not a parallel implementation.
- `cartesian/type_hints.py` — trivial (26 lines, three Protocols), standalone.

The real duplication is (a) `ScalarKind` vs `DTypeKind` and (b) cartesian's
separate world; rev. 1's "four type systems, ~0 sharing" overstated it.

### 3.6 Cartesian/next duplication (unchanged)

`cartesian/config.py` vs `next/config.py` (§7); `cartesian/caching.py`
(pickle/hashlib + `StencilID`) vs the otf fingerprint caches;
`cartesian/backend/base.py` (`Backend(abc.ABC)` with ClassVars, subclassed per
backend) vs `next/backend.py` (frozen composition-of-workflows dataclass). No
shared base anywhere; the only shared floor is `eve`/`_core`/`storage`.

## 4. The `otf` pipeline today

### 4.1 Module inventory: the IR-aware/IR-agnostic split is already sharp

Grep of IR/frontend imports across `otf/`:

| IR/DSL-**aware** (imports `iterator.ir` or `ffront`) | IR-**agnostic** |
| --- | --- |
| `definitions.py` (ffront stages + itir), `stages.py` (itir), `compiled_program.py` (ffront), `recipes.py` (transitively, via `definitions`) | `workflow.py`, `toolchain.py`, `code_specs.py`, `options.py`, `cpp_utils.py`, `arguments.py` (type-system-aware only), all of `binding/`, all of `compilation/` |

The core/infrastructure cut proposed in the main document now runs along an
existing, verifiable line — extracting the right column into an `infra` layer
requires no disentangling, only relocation (plus the `otf.stages ⇄
otf.definitions` cycle fix).

### 4.2 Workflow combinators: 8 classes, one now provably dead

`otf/workflow.py` (360 lines) still defines `Workflow` (Protocol),
`ChainableWorkflowMixin`, `ReplaceEnabledWorkflowMixin`, `NamedStepSequence`,
`MultiWorkflow`, `StepSequence`, `CachedStep`, `SkippableStep`.

- **`SkippableStep` has zero users** in `src/` — deletable today.
- **`MultiWorkflow` has exactly one user** (`backend.Transforms`), which
  implements an explicit `step_order(inp)` via a `match` on the input stage
  type (`backend.py:98-137`) — reasonable, but a whole abstract class for one
  subclass.
- **`NamedStepSequence.step_order` still uses annotation reflection**
  (`typing.get_type_hints` + `issubclass(origin, Workflow)` over dataclass
  fields, `workflow.py:146-161`) to derive step order from field declaration
  order.
- `StepSequence` was simplified (**[changed since rev. 1]**: inner `__Steps`
  holder removed).
- `CachedStep` is the redesigned piece (§5.2).

### 4.3 `toolchain.py`: the envelope + three adapters (unchanged shape)

`ConcreteArtifact(Generic[DefT, ArgsT])` — the `(data, args)` pair threaded
through the pipeline — plus `DataOnlyAdapter`, `ArgsOnlyAdapter`,
`StripArgsAdapter`, each a frozen dataclass inheriting
`ChainableWorkflowMixin + ReplaceEnabledWorkflowMixin + Workflow[...] +
Generic[...]` to lift a step over one half of the envelope
(`toolchain.py:30-66`). Still composition dressed as Protocol/Generic
scaffolding; still ~3 lines of logic per adapter.

### 4.4 `definitions.py`: the alias chains persist

`ConcreteProgramDef: TypeAlias = toolchain.ConcreteArtifact[IRDefinitionT,
ArgsDefinitionT]`; `CompilableProgramDef: TypeAlias =
ConcreteProgramDef[itir.Program, arguments.CompileTimeArgs]`; step contracts
`TranslationStep`/`BindingStep`/`CompilationStep` as Protocols over these
(`definitions.py:22-73`). Reading the pipeline still means unfolding
`TypeVar → TypeAlias → Protocol → Generic → class`.

### 4.5 Argument model: the straddle persists, with in-tree acknowledgments

`arguments.CompileTimeArgs` (`arguments.py:142-176`) still mixes compile-time
`TypeSpec`s with the **runtime** `offset_provider` —
`# TODO(havogt): replace with common.OffsetProviderType once the temporary
pass doesn't require the runtime information` (`:148`) — and still carries
`column_axis`, which `compiled_program.py:627` suspects is dead
(`TODO ... seems to be unused, even for programs with scans`).

### 4.6 eval/exec-based codegen has **grown** — **[changed since rev. 1]**

Inventory of `eval`/`exec` on generated source strings:

- `otf/arguments.py`: `ArgStaticDescriptor.from_value` (`:62-66`),
  `make_primitive_value_args_extractor` (`:279-280`).
- `otf/compiled_program.py`: 6 sites — `_get_type_of_param_expr:223`,
  `_make_argument_descriptors:244`, `_convert_to_argument_descriptor_context:290`,
  `_argument_descriptor_cache_key_from_args:501` (the measured hot path),
  `_argument_descriptor_cache_key_from_descriptors:522`,
  `_validate_argument_descriptor_mapping:554`.
- `next/named_collections.py`: extractor/constructor factories
  (`make_named_collection_extractor:126`, `make_named_collection_constructor:255`, …).

Rev. 1 proposed restricting this technique to the one measured hot path; the
codebase went the other way. Whether each site is hot enough to justify the
debuggability cost is an open question for the main proposal, not a settled
defect.

### 4.7 `compiled_program.py`: a new orchestration subsystem (not in rev. 1)

`CompiledProgramsPool` (`compiled_program.py:318-681`) manages per-program
compiled variants keyed by
`CompiledProgramsKey = (*hashable_arg_descriptors, id(offset_provider),
specialization)`: JIT vs AOT (`enable_jit`, `otf/options.py`
`CompilationOptions`), static-argument specialization via
`ArgStaticDescriptor`s, **async compilation** through a `ThreadPoolExecutor`
(with public `wait_for_compilation` re-exported from `next/__init__`), and
instrumentation via `hook_machinery` event/context hooks. It sits *above*
`Backend.compile()` (`backend.py:154-160`), which is itself new-ish: a
sanctioned AOT entry `Backend.compile(program, compile_time_args) →
ExecutableProgram`. Architecturally this is the "runner orchestration"
component of the core layer — and it is the natural consumer of a future
sanctioned stage API.

### 4.8 Stage inspection: still none; the dace reach-in persists

No `on_stage` observer, no `GT4PY_DUMP_STAGES`, no stage-dump switch exists
anywhere in `next/` (the only dump facility is metrics:
`GT4PY_DUMP_METRICS_AT_EXIT`). The single duck-typed reach-in survives at
`runners/dace/program.py:79-95`:

```python
otf_workflow = self.backend.executor
assert hasattr(otf_workflow, "translation")
otf_workflow_translation = (
    otf_workflow.translation.step
    if isinstance(otf_workflow.translation, workflow.CachedStep)
    else otf_workflow.translation
)
sdfg = dace.SDFG.from_json(
    otf_workflow_translation.replace(disable_itir_transforms=True, ...)(gtir_stage).source_code
)
```

`Backend.executor` is typed only as `Workflow[CompilableProgramDef,
CompilationArtifact]`, so this relies on it *actually* being an
`OTFCompileWorkflow` (hence the `assert hasattr` and `type: ignore`). It also
mutates a frozen stage via `object.__setattr__` (`program.py:62-69`).
`config.DEBUG`-gated `devtools.debug(...)` dumps exist at exactly two points
(`past_to_itir.py:145`, `iterator/runtime.py:71`).

## 5. Caching & fingerprinting today

### 5.1 The engine (`next/fingerprinting.py`)

Design: one generic bottom-up fold (`catabolize`, iterative, cycle-tolerant
via depth-relative back-references) over per-type one-level
**deconstructions**, aggregated with xxhash (`fingerprint_aggregator`).
Deconstructor tables are composable (`make_deconstructor` with per-type
overrides over a fallback; MRO-dispatched via eve's new `singledispatcher`).
Two canonical instances with an explicit durability contract:

- `strict_fingerprinter` — functions/types/modules identified by *verified*
  fully-qualified name (rejects `<lambda>`/`<locals>`/shadowed globals);
  deterministic **across processes**; keys persistent caches.
- `lenient_fingerprinter` — tolerates non-importable objects (hashes code
  objects/closures); **single-process only**; keys in-memory caches.

Domain knowledge enters only through overrides supplied at the *call sites in
core*: `iterator/ir.py:26-30` builds `lenient_ir_fingerprinter` with
`skipping_fields_node_deconstructor("location", "type")`;
`ffront/stages.py:39-70` builds `semantic_fingerprinter` with a custom
`FunctionType` deconstructor. **This is precisely the "engine in
infrastructure, descriptors from core" contract rev. 1 asked for** — already
implemented, just not yet *located* in an infrastructure package.

### 5.2 Cache layers and keys (one engine, several keys — by design, mostly)

| Cache | Key derivation | Tier |
| --- | --- | --- |
| ffront stage caches (7× `CachedStep` in `func_to_foast`, `foast_to_gtir`, `past_to_itir`, …) | `semantic_fingerprinter` input half + lenient step half | in-memory |
| Executor cache key helper | `fast_compilable_program_fingerprinter` (`otf/stages.py:36-58`): lenient over (semantic IR fingerprint, args, kwargs, **offset providers by `id()`**, column_axis) | in-memory |
| Persistent translation caches (gtfn `gtfn.py:149`, dace `workflow/factory.py:45`) | `CachedStep.persistent` with `compilable_program_fingerprinter = strict_fingerprinter` → `_core.filecache.FileCache` | persistent |
| Build-artifact folders (`otf/compilation/cache.py:34-67`) | `slug_{strict_fingerprinter(ext_source)}_{BUILD_CACHE_VERSION_ID}` + `build_context_id = strict_fingerprinter(builder_factory)` | persistent |
| `CompiledProgramsPool` variants | tuple key: eval-extracted static-arg values + `hash_offset_provider_items_by_id` + `content_hash` specialization | in-memory |
| `FileCache` filenames | `eve.utils.content_hash(key)` (pickle-protocol-pinned) | persistent |
| cartesian (`cartesian/caching.py`) | separate `StencilID`/`shashed_id` scheme | persistent |

The by-`id()` offset-provider hashing survives **only in the in-memory tier**
and is now *documented as deliberate* (cheap identity comparison; binding
order baked into generated bindings — docstring at `otf/stages.py:44-51`).
Warm starts genuinely skip translation now: persistent translation cache +
strict-fingerprint build folders.

### 5.3 Residual issues found in this pass

1. **`CachedStep.persistent()` wires the *lenient* step fingerprinter**
   (`workflow.py:322`) while the class docstring (`workflow.py:243-248`)
   states persistent caches "require `strict_fingerprinter` … so the on-disk
   keys stay reproducible across processes". The *input* half is strict; the
   *step* half is not. Either the docstring or the code is wrong — worth an
   upstream issue either way.
2. **Vestigial `key_function`**: `GTFNBackendFactory.Params.key_function =
   stages.fast_compilable_program_fingerprinter` (`gtfn.py:193`) is declared
   but wired to nothing.
3. `compiled_program.py` keeps a metrics-source cache keyed by `id(pool)` with
   a `weakref.finalize` guard (`:61-77`) — correct but another bespoke keying.
4. **No cache diagnostics anywhere**: `CachedStep.__call__` computes silently
   on `KeyError`; a JIT-disabled pool miss raises a bare `RuntimeError`. The
   "why did this recompile?" question still has no one-line answer.
5. Cartesian caching is untouched by all of the above.

## 6. Backend construction: factory-boy, still — with new evidence

- 7 `factory.Factory` subclasses: `GTFNCompilerFactory`,
  `GTFNCompileWorkflowFactory`, `GTFNBackendFactory` (gtfn.py),
  `DaCeBackendFactory`, `DaCeWorkflowFactory`, `DaCeTranslationStepFactory`,
  `DaCeCompilationStepFactory` (dace/workflow/). All of `Trait`, `SubFactory`,
  `LazyAttribute`, `LazyFunction`, `SelfAttribute` in active use.
- Stringly-typed `__`-path overrides remain the configuration mechanism, e.g.
  `GTFNBackendFactory(name_postfix="_imperative",
  otf_workflow__translation__use_imperative_backend=True)` (`gtfn.py:210`) and
  six `otf_workflow__bare_translation__*` overrides inside
  `make_dace_backend` (`dace/workflow/backend.py:116-121`).
- **`make_dace_backend(...)` is already a plain-function wrapper** producing
  the four dace backends — i.e. the `make_*_toolchain()` shape rev. 1
  proposed exists, but as a facade *over* the factory instead of a
  replacement.
- **pyproject pins factory-boy as broken under mypy**:
  `# factory-boy is broken, see https://github.com/FactoryBoy/factory_boy/pull/1114`
  with `module = "factory.*"` overrides — the type-safety cost is now
  acknowledged in-tree.
- **In-tree counter-current**: `roundtrip.py` builds its four backends by
  directly instantiating `next_backend.Backend(...)` — plain code — but
  carries `# TODO(tehrengruber): introduce factory` (`roundtrip.py:283`). The
  team is pulling in both directions; needs an explicit decision.
- `factory-boy>=3.3.3` and `dace>=2.0.0a4` are still **required** runtime
  dependencies (dace is imported lazily/optionally at runtime,
  `decorator.py:419-424`, yet is a hard install dep).

## 7. Allocators & config — **[rev-1 correction]** smaller than described

`next/custom_layout_allocators.py` (276 lines) has **2** Protocols
(`FieldBufferAllocatorProtocol`, `FieldBufferAllocatorFactoryProtocol`) and
**6** `TypeIs` guards (`is_field_allocator[_for]`,
`is_field_allocator_factory[_for]`, `is_field_allocation_tool[_for]`) — rev. 1
counted "six Protocols plus ~ten TypeIs". A device registry **already
exists**: `device_allocators: dict[core_defs.DeviceType,
FieldBufferAllocatorProtocol]` (`:188`), populated for CPU/CUDA/ROCM at import
time. What keeps the module core-bound is unchanged:
`__gt_allocate__(domain: common.Domain, ...)` takes the *domain* (`:42-49`),
with `BaseFieldBufferAllocator` doing `domain.shape` / `domain.dims` /
absolute-to-relative index translation internally — the domain→shape
translation rev. 1 wanted lifted into core. `next` field allocation is built
on `gt4py.storage.allocators` (`TensorBuffer`, layout maps), which cartesian
also uses via its own `storage/cartesian` layout registry — the low level is
shared; the field-level abstraction is duplicated.

**Config** remains fully duplicated with divergent conventions:
`next/config.py` (env namespace `GT4PY_*`; DEBUG, build-cache dir/lifetime/
version-id, CMake build type, metrics, JIT default; typed helpers
`env_flag_to_bool/int`) vs `cartesian/config.py` (namespaces `GT_*`,
`GT4PY_CARTESIAN_*`, `GT4PY_COMPILE_*`; gridtools-cpp include paths, OpenMP
flags, `build_settings`/`cache_settings` dicts; `.gt_cache` vs
`.gt4py_cache`). No shared code, no cross-imports.

## 8. Third-party dependency inventory (updated)

Required runtime deps now: `numpy`, `attrs`, `array-api-compat`, `black`
(codegen formatting), `boltons`, `cached-property`, `click`, `cmake`,
`cytoolz`/`toolz`, **`dace>=2.0.0a4`** (one backend; lazily imported but
required), `deepdiff`, `devtools`, **`factory-boy`** (builders only; mypy-broken,
see §6), `filelock`, `frozendict`, `gridtools-cpp`, `jinja2` **and** `mako`
(two template engines), `lark`, `nanobind` **and** `pybind11` (two binding
generators), `ninja`, `ordered-set`, `packaging`, `setuptools`, `tabulate`,
`typing-extensions`, `versioningit`, `xxhash` (the fingerprint aggregator).
Optional extras: `cartesian`/`next` (= `jax,standard,testing`), `cuda12/13`,
`rocm6/7`, `jax[-cuda*]`. Dev: `tach>=0.23.0` in the `lint` group, run by
pre-commit.

Layering observations (mostly unchanged): `dace` and `factory` sit in core
deps but serve one backend / one construction pattern; two template engines
and two binding generators still reflect the cartesian/next split; hashing
surface is now consolidated on `xxhash`+`content_hash` (better than rev. 1's
three-way spread).

## 9. `tach` in gt4py today, and the layers feature

gt4py already runs `tach check` in pre-commit/CI over the flat five-package
DAG (§3.1) — adoption is no longer a proposal, it is practice, including the
social contract in `AGENTS.md` (edit `tach.toml` + justify in the PR). What
the current file does **not** use yet:

- `forbid_circular_dependencies = true` (blocked by `cartesian ⇄ storage`);
- module boundaries *inside* `gt4py.next` (the whole package is one tach
  module today);
- the [layers](https://docs.gauge.sh/usage/layers/) feature: an ordered
  `layers = [...]` list where each module is pinned to a layer and
  lower→higher imports fail `tach check`; cross-layer downward imports need no
  `depends_on`, same-layer edges must be explicit; per-module `strict` limits
  cross-boundary imports to declared public members; `deprecated` marks edges
  that warn without failing (useful for migration shims).

The main proposal's step "land tach" is therefore replaced by an **evolution
path of the existing file**: fix the storage cycle → forbid cycles → declare
intra-`next` modules → switch the top of the file to `layers`.

## 10. JAX as the architectural reference (unchanged)

Still the model for the public/internal split; summarized (details in rev. 1
remain accurate):

1. **`_src/` holds everything; public namespaces only re-export** (one-way
   `jax/* → jax/_src/*`). gt4py's `_internal` plays the `_src` role.
2. **Explicit `from x import y as y` re-exports** — the redundant `as y` marks
   names public for type checkers and makes the surface an auditable list.
3. **Deprecation via module `__getattr__`** — relocated/retired names warn
   instead of `AttributeError`. Note gt4py currently has **no** module
   `__getattr__` anywhere in its public `__init__`s, so there is no
   deprecation machinery to lean on yet.
4. **Domain-grouped public submodules** over thin `_src` implementations.
5. **Concrete types at the boundary**; generic/variance machinery stays
   internal.

gt4py's `next/typing.py` (public typing-only aliases, re-exported as
`gtx.typing`) is a first, small step toward a deliberate public surface — the
facade proposal generalizes it.
