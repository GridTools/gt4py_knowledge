---
title: Dissolving gt4py.next.otf into a toolchain core and DSL-agnostic build infrastructure
author: egparedes
tags: [otf, toolchain, workflow, pipeline, refactoring, architecture, layering, modularity, backend, bindings, build-system, caching, runners, compiled-program, factory-boy, naming, observability, stage-inspection, adr, tach, tech-debt]
created: 2026-07-28
status: draft
---

> **TL;DR** Retire the `gt4py.next.otf` package by splitting it along the line
> its own code already draws. The DSL-aware half (stage vocabulary, argument
> model, options, compiled-programs pool, task preparation) becomes the
> **toolchain core**, merged with today's `next/backend.py` under the name the
> team already agreed on (`Backend → Toolchain`, ADR 0017's own term). The
> DSL-agnostic half (pipeline machinery, source/artifact models, build
> systems, bindings, compilation runners) becomes the shared **build
> infrastructure** of [[personal/egparedes/layered-architecture|the layered
> architecture]]. On the way, the thirteen-abstraction workflow-combinator
> framework collapses into plain typed composition (`Step` alias +
> `CachedStep` + explicit `__call__` dataclasses), the `stages ⇄ definitions`
> cycle disappears by merging the conjoined pair, and the seam gets its
> missing sanctioned interface (stage observer + `Toolchain.translate`),
> killing the dace `__sdfg__` reach-in. Everything is phased: **simplify in
> place first, relocate once, shims once.** All behavior is preserved.

> **Status**: draft, AI-assisted, verified against the local gt4py checkout at
> `1eda83a79` (one commit past `da2c6df41`, the rev. 3 baseline of the parent
> proposal). Every usage count below was re-derived from that tree, not
> quoted from memory.

> **Parent proposal**:
> [[personal/egparedes/layered-architecture|A layered architecture for gt4py]]
> — this note *executes* its items 2 (stage-inspection API), 5 (pipeline, not
> combinators), 4 (plain builders — phase 6 below), the naming half of item 9
> (the stale-docs half stays with the parent), the `otf.stages ⇄
> otf.definitions` cycle of item 10 (the other three cycles stay with the
> parent), and the `key_function`-deletion half of item 1 (phase 0); it also
> refines the layer placement of `otf/binding` with a new finding (§ "Two
> corrections"). Item 1's `CachedStep.persistent` half and items 3, 6, 7, 8
> stay with the parent. Background catalogs live in
> [[personal/egparedes/layered-architecture/layered-architecture_research|the
> parent's research appendix]] (§4 in particular) and are not repeated here.

## Problem / motivation

`gt4py.next.otf` (~3.9 kLOC, 23 modules) is the on-the-fly compilation
toolchain: everything between an IR and an executable Python callable, plus —
by accretion — a good part of what sits above and around that bracket. The
package works, but its structure actively resists change. Each pain point
below is a named red flag from
[[knowledge/software-engineering/principles|Working Principles]] §6, with the
evidence verified in the current tree.

1. **Scope drift / unmapped context.** ADR 0011 chartered OTF as "everything
   necessary to go from an *IR representation* to an executable Python
   function" — the frontend is explicitly outside the bracket. Today the
   package hosts `compiled_program.py` (imports `ffront.stages`, sits *above*
   `Backend.compile`), `definitions.py` (imports the ffront stage types) and
   the argument model — all above its own charter. ADR 0017 already names the
   larger unit: "**toolchain** ⊇ OTF pipeline + transformation passes +
   lowerings + parsers". The code grew into a toolchain package while keeping
   the OTF name; even the in-tree `next/AGENTS.md` describes `otf/` as the
   "on-the-fly compilation *toolchain*". The vocabulary exists; the package
   boundaries ignore it.
2. **Conjoined modules + a genuine cycle.** `definitions.py` defines step
   contracts over types from `stages.py`, while `stages.py` type-checks
   against `definitions.py` behind a `TYPE_CHECKING` guard
   (`stages.py:30-33`) — neither is understandable alone, and the cycle is
   one of the four blocking `forbid_circular_dependencies = true` in
   `tach.toml`. The same cycle formally taints `runners.py` (its only
   DSL-side contact is the `CompilationArtifact` protocol it imports from the
   itir-importing `stages.py`).
3. **A shallow combinator tower.** `workflow.py` + `toolchain.py` export
   thirteen abstractions — nine in `workflow.py` (`Workflow`,
   `ChainableWorkflowMixin`, `ReplaceEnabledWorkflowMixin`,
   `NamedStepSequence`, `MultiWorkflow`, `StepSequence`, `CachedStep`,
   `SkippableStep`, `make_step`), plus the `ConcreteArtifact` envelope and
   three adapters in `toolchain.py` — to express *function composition*. Measured
   against actual use (full consumer map re-derived for this note):
   `Workflow` appears ~38 times, essentially always as a type annotation —
   i.e. it is `Callable[[S], T]` with extra steps; `CachedStep` is the one
   deep member (7 ffront stage caches + the gtfn/dace persistent translation
   caches); `MultiWorkflow` has exactly one subclass (`backend.Transforms`);
   `StepSequence` has one non-test user (`past_passes/linters.py`);
   `make_step` has three, each immediately re-wrapped; `SkippableStep` has
   zero; the three adapters have five call sites totalling ~9 lines of logic.
   Worse, the machinery *defeats* the static typing ADR 0011 prized:
   `NamedStepSequence.__call__` runs an `Any`-typed reflection loop over
   `typing.get_type_hints` (`workflow.py:139-161`), so mypy never checks that
   the composed steps actually fit together.
4. **Vague and colliding names.** Five module names for one concept's pieces
   (`workflow`, `toolchain`, `definitions`, `stages`, `recipes`);
   `ConcreteArtifact` for what is a (definition, arguments) pair;
   `Backend.executor` for a pipeline; and two distinct `CompilationError`
   classes (`otf/compilation/compiler.py:140` shadows
   `next/errors/exceptions.CompilationError` — every external reference
   resolves to the latter).
5. **Special–general mixture, and a hidden layering.** The DSL-aware /
   DSL-agnostic split inside `otf` is verifiably sharp (parent appendix §4.1)
   but invisible in the package structure — generic build plumbing sits
   beside itir-importing orchestration in one flat namespace, and nothing
   stops a new upward import. Meanwhile the genuinely shared services keep
   being built *as one-offs* (fingerprinting, runners, crash-consistent
   caches) because there is no infrastructure home to put them in.
6. **The seam has no sanctioned interface.** The one consumer that needs an
   intermediate stage (`runners/dace/program.py:79-95`) duck-types through
   `backend.executor.translation`, `assert hasattr`, unwraps `CachedStep`,
   and mutates the envelope's frozen `CompileTimeArgs` via
   `object.__setattr__` (`program.py:62-69`) — the cost of having no
   observer/inspection API at the core↔infrastructure seam (parent item 2).
7. **Small DRY debts.** `options.py` maintains a `TypedDict` mirror of
   `CompilationOptions` kept in sync by a runtime `assert` (noted, not fixed
   here: the decorator's `Unpack[TypedDict]` kwargs typing requires the
   mirror as long as Python cannot derive a `TypedDict` from a dataclass);
   the vestigial `GTFNBackendFactory.Params.key_function` still points at
   `fast_compilable_program_fingerprinter`, which has no other production
   user left.

**Why now**: the parent proposal's step 2 ("finish the in-place `otf` debt")
is explicitly scheduled *before* any `_internal` move, and ADRs 0023–0025
just demonstrated, three times in two months, that services built engine-down
in exactly this shape land well. This note is the concrete design for that
step plus the relocation map that follows it.

## Two corrections to the parent proposal's placement table

Reading the code end-to-end surfaced two facts the parent's rev. 3 analysis
missed; both refine (not contradict) its layer table:

- **`otf/binding/` is not DSL-agnostic — it is type-system-bound.**
  `binding/interface.py` gives every `Parameter` a `ts.TypeSpec`;
  `binding/cpp_interface.py` and `binding/nanobind.py` pattern-match
  `ts.ScalarType / FieldType / TupleType` and read `common.DimensionKind`
  (`nanobind.py:22,40,205-230`); `cpp_utils.py` maps `ts.ScalarType` → C++
  type strings. "IR-agnostic" (never imports itir/ffront — true) is not
  "DSL-agnostic" (never imports the type system — false). Moving `binding/`
  to infrastructure as-is would drag `next/type_system` (and a corner of
  `common`) down with it. The fix is the same pattern the parent already
  applies to allocators and fingerprinting — *the service takes plain data;
  core does the translation* — detailed below as the **bindings parameter
  surgery**.
- **`otf/compilation/` really is shared infrastructure, with the evidence to
  prove it**: gtfn *subclasses* `CPPCompiler`/`CPPCompilationArtifact`, dace
  consumes `compilation.cache` and `compilation.common`, and an external
  script (`scripts/python/dace_determinism.py`) reads
  `cache.CACHE_FOLDER_NAME_PATTERN` from the installed package. Two backends
  + external tooling = a real seam, not a hypothetical one. (The nanobind
  generator, by contrast, has exactly one production consumer — gtfn — but it
  is ADR 0011's designed "shared bindings step" for any future
  nanobind-bound C++ backend, and dace independently constructs
  `interface.Parameter`/`Function`, so the *interface model* is shared even
  where the generator is not.)

## Design — considered twice, then chosen

Three genuinely different decompositions were sketched before committing
(principles §2, "design it twice"); ADR 0011's own alternatives section
provides a fourth, historical data point.

| Decomposition | Idea | Verdict |
| --- | --- | --- |
| **A. Split along the knowledge line** (chosen) | DSL-aware half → toolchain core (merged with `backend.py`); DSL-agnostic half → shared build infrastructure (the parent's `infra` layer). | Matches the verified sharp line, the parent's end-state tree, and ADR 0017's vocabulary. Makes the build services shareable with `cartesian` later. Cost: two package homes instead of one. |
| **B. One `toolchain` package, restructured but unsplit** | Rename `otf` → `toolchain`, absorb `backend.py`, clean up modules, draw intra-package tach edges. | Cheaper, single home, and tach *could* still enforce the internal line. Rejected as the end state because it forfeits what the split is *for*: an infrastructure layer that `cartesian` can share and that the `_internal` move can lift wholesale — B would have to be reorganized a second time. Adopted, however, as the **phasing**: phases 0–4 below are exactly "B applied in place". |
| **C. Push special-purpose code down into the backends** | If bindings/build systems serve only gtfn, move them under `program_processors`; shrink `otf` to generic contracts. | Rejected as organizing principle: the premise is false for `compilation/` (two backends + external tooling, see above) and half-false for bindings (shared interface model). Its *lens* is kept: it produced the nanobind finding and keeps the door open to relocating truly single-backend pieces later. |
| (ADR 0011's alternatives) | God-typed pipeline; ad-hoc function composition; untyped step lists. | Its requirements — named steps, static typing, composition-time customization — are preserved below *without* the combinator machinery; see the supersession table. |

**Choice: A for the destination, B for the path.** Simplify and re-partition
in place (each intermediate state complete and consistent), then relocate
once, when the parent's `_internal` skeleton exists — shims are paid exactly
once.

## Target structure

### After the in-place phases (0–4): still two roots, honest contents

```
next/backend.py         # Toolchain (ex Backend) + Transforms (plain code) + CompilePipeline (ex recipes)
next/otf/
  # DSL-aware half (future toolchain core)
  stages.py             # definition-stage union + CompilableProgram + step type aliases
                        #   (absorbs definitions.py; the generic pair type does NOT live here — see below)
  arguments.py          # argument model (unchanged home; parent item 6 applies here)
  options.py            # CompilationOptions
  compiled_program.py   # CompiledProgramsPool, wait_for_compilation
  compilation_tasks.py  # main-side task preparation
  # DSL-agnostic half (future infra)
  workflow.py           # Step alias + CachedStep + ProgramWithArgs pair + stage-observer hook
  artifacts.py          # NEW: code specs (ex code_specs.py) + ProgramSource/BindingSource/
                        #   ExtensionSource + CompilationArtifact + ExecutableProgram + BuildSystemProject
  runners.py            # unchanged; its otf-side import narrows to artifacts.py → formally IR-free
  binding/  compilation/  cpp_utils.py   # unchanged until phase 5; binding/ pending the surgery
  # deleted: definitions.py, toolchain.py, recipes.py, code_specs.py (all shimmed one release)
```

### After relocation (phase 5, = parent steps 6b/6c/6f + 7)

```
gt4py/_internal/infra/            # layer: infrastructure
  pipeline.py         ← otf/workflow.py
  artifacts.py        ← otf/artifacts.py
  bindings/           ← otf/binding/ (+ cpp_utils, post-surgery)
  build/              ← otf/compilation/   (compiler, cache, build_data, importer, common, build_systems/)
  runners.py          ← otf/runners.py
gt4py/_internal/next/toolchain/   # layer: core (runner orchestration)
  toolchain.py        ← next/backend.py    (Toolchain, Transforms, CompilePipeline)
  stages.py           ← otf/stages.py
  arguments.py        ← otf/arguments.py
  options.py          ← otf/options.py
  pool.py             ← otf/compiled_program.py
  tasks.py            ← otf/compilation_tasks.py
```

File renames (`compiled_program → pool`, `binding → bindings`,
`compilation → build`, `workflow → pipeline`) deliberately ride the
relocation, where per-module shims exist anyway — the in-place phases change
*contents*, never import paths, except where a module is deleted outright
(shimmed one release).

### Module map (old → new, with layer and the deciding fact)

| Today | Interim (phases 0–4) | Final (phase 5) | Layer | Deciding fact |
| --- | --- | --- | --- | --- |
| `otf/workflow.py` | shrinks to `Step` + `CachedStep` + observer | `infra/pipeline.py` | infrastructure | generic; `CachedStep` is the one deep combinator |
| `otf/toolchain.py` | **deleted** (pair type → `workflow.py`; adapters inlined) | — | — | 3 adapters, 9 lines of logic, 5 call sites; the pair type must stay in a DSL-neutral bottom module because `ffront/stages.py:32` imports it at module level |
| `otf/definitions.py` | **merged into** `stages.py` | — | — | conjoined with `stages.py`; the cycle dies with the merge |
| `otf/stages.py` (def-side) | `stages.py` | `toolchain/stages.py` | core | imports itir + ffront stages |
| `otf/stages.py` (source/artifact side) + `code_specs.py` | **new** `artifacts.py` | `infra/artifacts.py` | infrastructure | no IR imports; what any build knows |
| `otf/recipes.py` | `CompilePipeline` in `backend.py` | `toolchain/toolchain.py` | core | typed against `CompilableProgram` |
| `otf/arguments.py` | unchanged | `toolchain/arguments.py` | core | type-system-aware throughout |
| `otf/options.py` | unchanged | `toolchain/options.py` | core | consumed by decorator/pool |
| `otf/compiled_program.py` | unchanged | `toolchain/pool.py` | core | orchestration above `Toolchain.compile` |
| `otf/compilation_tasks.py` | unchanged | `toolchain/tasks.py` | core | decides what only the main process can do |
| `otf/runners.py` | imports `artifacts.py` | `infra/runners.py` | infrastructure | ADR 0024's execution service; IR-free after the merge |
| `otf/compilation/` | drops its `definitions` import + `CompilationStep` base (phase 1); otherwise unchanged | `infra/build/` | infrastructure | multi-backend + external-tooling seam; `compiler.py:17,90` is its only IR-side edge today |
| `otf/binding/` + `cpp_utils.py` | unchanged | `infra/bindings/` | infrastructure **after surgery** | type-system-bound today (see below) |
| `next/backend.py` | `Toolchain` (+ aliases) | `toolchain/toolchain.py` | core | the toolchain root object |

## The interfaces, concretely

### Pipeline machinery: three exports instead of thirteen

```python
# infra/pipeline.py (interim: otf/workflow.py)
type Step[S, T] = Callable[[S], T]          # the whole "framework"

@dataclasses.dataclass(frozen=True)
class CachedStep[S, T, H]:                   # kept as-is: the one deep module
    ...  # in_memory() / persistent() / cache_key() / __call__ — unchanged interface

@dataclasses.dataclass                       # NOT frozen: freezing the envelope would be a
class ProgramWithArgs[DefT, ArgsT]:          #   behavior change (it is mutable today), deferred
    definition: DefT                          # ex toolchain.ConcreteArtifact / .data
    args: ArgsT
```

The generic pair type lives *here*, in the DSL-neutral bottom module — not in
the core `stages.py` — because `ffront/stages.py:32` parameterizes it at
module-import time (the four `Concrete*Def` aliases), while the merged
`stages.py` imports `ffront.stages` for the definition union. Today's
`toolchain.py` plays exactly this bottom-module role; putting the pair
anywhere DSL-aware would recreate the `stages ⇄ ffront.stages` cycle one seam
over.

Named pipelines become frozen dataclasses with an **explicit, fully typed**
`__call__` — composition you can read *and* that mypy actually checks
(today's reflection loop is `Any`-typed):

```python
@dataclasses.dataclass(frozen=True)
class CompilePipeline:                       # ex recipes.OTFCompileWorkflow
    translation: Step[CompilableProgram, ProgramSource]
    bindings: Step[ProgramSource, ExtensionSource]
    compilation: Step[ExtensionSource, CompilationArtifact]

    def __call__(self, program: CompilableProgram) -> CompilationArtifact:
        source = self.translation(program);  _notify_stage("translation", source)
        ext = self.bindings(source);         _notify_stage("bindings", ext)
        artifact = self.compilation(ext);    _notify_stage("compilation", artifact)
        return artifact
```

Customization stays composition-time, per ADR 0011, via plain
`dataclasses.replace(pipeline, translation=...)` — the
`ReplaceEnabledWorkflowMixin` was always a pass-through to it.
`backend.Transforms` gets the same treatment: its `match`-based `step_order`
becomes a `match`-based `__call__` that calls the steps directly and threads
`(definition, args)` explicitly. The three adapters serve **two** consumer
groups, and both are in the rewrite inventory of the combinator phase: the
`Transforms` pipeline itself, and the *per-step external callers* that today
wrap a stage into a pair only to unwrap `.data` from the result —
`decorator.py` (`past_lint` at `:270`, `func_to_past` at `:295`,
`func_to_foast` at `:611`) and `foast_to_past.OperatorToProgram` (`:86`).
For those callers the bare data-only steps are strictly *simpler*: call the
step, drop the wrap/unwrap dance. The ffront `adapted_*_factory` wrappers
shrink to their bare step factories (the `CachedStep.in_memory` wrapping is
unchanged; it caches the data-only function, so keys don't change).
`make_step`/`StepSequence`/`.chain` rewrites are one-liners at their 4 call
sites.

What ADR 0011's decisions become (the successor ADR's core table):

| ADR 0011 requirement | Kept by |
| --- | --- |
| Named steps, order visible | dataclass fields + explicit `__call__` (order now *literally* visible) |
| Statically typed composition | explicit `__call__` — checked end-to-end, unlike the reflection loop |
| Customization at composition, not via flags | `dataclasses.replace` on frozen pipelines |
| Steps compose across backends | `Step[S, T]` is `Callable` — every existing step already satisfies it |
| Linear workflows | unchanged (`Transforms` keeps its input-dependent step *selection*) |

### The stage vocabulary: one core module, one infra module

```python
# toolchain/stages.py (core; interim otf/stages.py — absorbs definitions.py)
type ProgramDefinition = (DSLFieldOperatorDef | DSLProgramDef | FOASTOperatorDef
                          | PASTProgramDef | itir.Program)   # ex IRDefinitionT TypeVar

@dataclasses.dataclass                        # freezing deferred, like the pair type: the dace
class CompilableProgram:                      #   mutation site lives until phase 3
    definition: itir.Program                  # ex CompilableProgramDef alias — now concrete
    args: CompileTimeArgs                     # (JAX convention); crosses the process boundary (ADR 0024)

type TranslationStep[C: SourceCodeSpec] = Step[CompilableProgram, ProgramSource[C]]
# BindingStep protocol: deleted (zero users); CompilationStep: alias, same pattern
```

Classes that today *inherit* the step protocols drop those bases in the same
phase-1 PR — structural typing already covers them: `CPPCompiler` subclasses
`definitions.CompilationStep` (`compiler.py:90`, with the import at
`compiler.py:17`), and dace's `DaCeTranslationStep`/`DaCeCompiler` and gtfn's
`GTFNTranslationStep` do the equivalent. The `compiler.py` edit is not
optional polish: it is what actually removes `otf/compilation/`'s only
IR-side import edge, and a `type` alias cannot be subclassed, so leaving the
base in place would break the moment the protocol becomes an alias.

```python
# infra/artifacts.py (interim otf/artifacts.py) — no IR, no ts after the surgery
SourceCodeSpec and friends          # ex code_specs.py, unchanged
ProgramSource / BindingSource / ExtensionSource      # ex stages.py, unchanged shape
CompilationArtifact (Protocol: load() -> ExecutableProgram)
ExecutableProgram = Callable
BuildSystemProject                  # internal-only protocol, kept for the two build systems
```

`fast_compilable_program_fingerprinter` and its sole (vestigial) consumer
`GTFNBackendFactory.Params.key_function` are deleted;
`compilable_program_fingerprinter` (the strict alias used by both persistent
translation caches) moves next to `CachedStep`'s users' call sites unchanged.
The duplicate `compiler.CompilationError` is flagged for retirement, but
**not** as a behavior-preserving cleanup: the two classes differ in base
(`RuntimeError` vs `GT4PyError(Exception)`) and in message formatting, so
unifying them changes what `except` clauses catch at `compiler.py:123` — it
is queued as its own small decision in the open questions, outside the
no-behavior-change ladder.

### Toolchain: the deep root object, with the missing sanctioned entries

```python
# toolchain/toolchain.py (interim next/backend.py)
@dataclasses.dataclass(frozen=True)
class Toolchain(Generic[DeviceTypeT]):        # ex Backend — executes the in-tree TODO
    name: str
    frontend: Step[ProgramWithArgs, CompilableProgram]        # ex .transforms
    backend: Step[CompilableProgram, CompilationArtifact]     # ex .executor; a CompilePipeline for gtfn/dace
    allocator: next_allocators.FieldBufferAllocatorProtocol[DeviceTypeT]

    def compile(self, definition, compile_time_args) -> ExecutableProgram: ...   # as today
    def translate(self, definition, compile_time_args) -> ProgramSource:
        """Sanctioned partial run: frontend + translation only.

        Narrows `self.backend` to the standard `CompilePipeline` shape;
        raises a clear error on monolithic backends (which support no stage
        inspection today either).
        """
```

Two deliberate hardenings, both enabled by the consumer map:

- `backend` stays **structurally** typed: gtfn and dace already have the
  three-step `CompilePipeline` shape, but **roundtrip does not** —
  `Roundtrip` (`roundtrip.py:257-280`) is a monolithic single step with no
  `ProgramSource` intermediate, so forcing it into the shape would be a
  real (small) redesign, not a rename; it is listed as an open question.
  `translate()` therefore narrows to `CompilePipeline` in one typed place
  and raises a clear error for monolithic backends — still strictly better
  than the `assert hasattr` at `dace/program.py:83`, which it deletes.
- `translate()` takes **no per-call step options** — reconfiguration stays
  composition-time, ADR 0011's own rule (and a bare `Step` callable has no
  `.replace` for an options API to lean on). A caller needing a variant
  translation step builds a variant pipeline with `dataclasses.replace`;
  `CachedStep.step` is part of `CachedStep`'s interface for exactly this.
  Concretely, dace's `__sdfg__` path holds a dedicated translate-only
  pipeline built once next to its backend definition, instead of
  duck-typing into the default one and mutating its stages.

The **stage observer** rides the existing `instrumentation/hook_machinery`
(an event hook `stage_hook(name, artifact)` emitted by `CompilePipeline` and
`Transforms` between steps; artifacts are treated opaquely, so infrastructure
never imports an IR). `GT4PY_DUMP_STAGES=<dir>` registers a subscriber that
writes each artifact per program, MLIR `print-ir-tree-dir` style. This *is*
parent item 2, designed against the machinery this note reshapes.

### Bindings parameter surgery (the infra enabler)

`binding/nanobind.py` already contains its own internal `DimensionSpec`
node; the surgery promotes that shape to the interface and removes the
`ts.TypeSpec` field from `interface.Parameter`:

```python
# infra/bindings vocabulary — defs-level data only (no ts, no common):
ScalarParameter(name, dtype: core_defs.DType)
BufferParameter(name, dtype: core_defs.DType, dims: tuple[DimSpec, ...])
    # DimSpec(name, horizontal: bool, static_stride: int | None)
TupleParameter(name, elements)
```

Core-side codegens (gtfn_module, dace translation) translate `ts.TypeSpec` →
these when they build `ProgramSource` — they already own the ts knowledge.
`cpp_utils.pytype_to_cpptype` becomes a `core_defs.DType` → C++ mapping in
`infra/bindings`. This is the **third application of the established
pattern** (fingerprinting deconstructors, domain-agnostic allocators):
generic engine below, per-domain extraction above. It is required only for
the phase-5 `bindings/` move; nothing else waits on it.

### Naming (decided once, in the successor ADR)

| Today | Proposed | Alternatives noted |
| --- | --- | --- |
| `Backend` | `Toolchain` | settled by ADR 0017's definition + in-tree TODO |
| `Backend.transforms` | `Toolchain.frontend` | `frontend_transforms` (the TODO's literal) |
| `Backend.executor` | `Toolchain.backend` | `backend_transforms` (TODO), `compile_pipeline` |
| `OTFCompileWorkflow` | `CompilePipeline` | `BackendPipeline` |
| `ConcreteArtifact` / `.data` | `ProgramWithArgs` / `.definition` | `ProgramCall` (ADR 0011's original), `StagedProgram` |
| `CompilableProgramDef` | `CompilableProgram` (concrete class) | — |
| `otf` (package) | dissolved: `toolchain` + `infra` | — |

Class renames land with one-release deprecation aliases (`Backend =
Toolchain`); `gtx.*` user-facing names (`gtx.gtfn_cpu`,
`gtx.wait_for_compilation`) are byte-for-byte unchanged throughout.

## Implementation plan — each phase an independently green PR

Gates for every PR, per gt4py's own `AGENTS.md`: full `test_next` suite,
`uv run tach check`, `pre-commit` (ruff/mypy), no behavior change. Conventional
title `refactor[next]: ...`.

0. **Trivial deletions.** `SkippableStep` — together with its section in
   `docs/user/next/advanced/WorkflowPatterns.md`, which still documents it
   as user-facing API (its "zero users" are zero *code* users) — plus
   `key_function` + `fast_compilable_program_fingerprinter` (the second
   half of parent item 1) and the `BindingStep` protocol. *(Zero risk.)*
   The duplicate `CompilationError` is deliberately **not** here — see the
   open questions.
1. **Merge `definitions.py` into `stages.py`; extract `artifacts.py`; move
   the pair type to `workflow.py`.** Kills the `stages ⇄ definitions` cycle
   *without recreating it against `ffront.stages`* (the pair stays in the
   DSL-neutral bottom module); drops the step-protocol base classes in
   `compiler.py`, dace, and gtfn (structural typing suffices) — the edit
   that makes `runners.py` *and* `compilation/` formally IR-free; in-repo
   imports updated; all four deleted modules shimmed (incl. `code_specs`).
2. **Naming.** `Backend → Toolchain` + field renames with deprecation
   aliases; `next/AGENTS.md` + tutorials updated. **Lands with the
   successor ADR to 0011/0017** (append-only policy; the supersession table
   above is its core), which also records phases 3-4.
3. **Observability.** `stage_hook` + `GT4PY_DUMP_STAGES` +
   `Toolchain.translate`; delete the dace reach-in
   (`dace/program.py:62-95`), replaced by a dace-owned translate-only
   pipeline. *(Parent item 2, done.)* **Ordering constraint: must land
   before phase 4** — the reach-in is the last external consumer of the
   mixins' `.replace`.
4. **Pipeline, not combinators.** Explicit `__call__` in
   `CompilePipeline`-né-`OTFCompileWorkflow` and `Transforms` (which take
   over emitting the phase-3 hooks); delete `MultiWorkflow`,
   `NamedStepSequence`, `StepSequence`, `make_step`, `chain`, both mixins,
   all three adapters; introduce `Step`; simplify the seven ffront
   factories **and their per-step external callers** (`decorator.py` ×3,
   `foast_to_past.OperatorToProgram`); rewrite the two factory-boy
   `Meta.model` usages against the new class; update **both** advanced
   guides — `HackTheToolchain.md` *and* `WorkflowPatterns.md` (the latter
   teaches `StepSequence.chain`, `make_step`, `ChainableWorkflowMixin`, and
   `CachedStep` at length).
5. **Relocation** (when the parent's `_internal` skeleton lands — its steps
   6b/6c/6f/7): move per the module map, shims at all old `otf.*` paths,
   `scripts/python/dace_determinism.py` repointed, tach modules declared per
   the final layers. **Enabler PR**: the bindings parameter surgery,
   immediately before or with the `bindings/` move.
6. **Plain builders** (parent item 4, executed here; unblocked by phases 2
   and 4): `make_*_toolchain()`
   functions replacing the eight factory-boy factories; supersedes ADR 0017's
   factory-boy decision; resolves the in-tree conflict with
   `roundtrip.py:283`'s `TODO: introduce factory` the other way.

Phases 0–4 are valuable and complete even if the `_internal` move never
happens (decomposition B as fallback); phase 5 costs one `git mv` + shims
because everything is already on the right side of the line.

Suggested tach ratchet after phase 1 (interim, before layers exist —
requires the parent's step 3 intra-`next` module declarations):

```toml
[[modules]]
path = "gt4py.next.otf.workflow"      # + artifacts, runners, compilation
depends_on = [ { path = "gt4py._core" }, { path = "gt4py.eve" },
               { path = "gt4py.next.config" }, { path = "gt4py.next.fingerprinting" },
               { path = "gt4py.next.otf.artifacts" } ]   # no IR-side edges — true once
                                                         # phase 1 drops compiler.py's
                                                         # `definitions` base-class import
```

## What this deliberately does *not* do

- **No behavior or performance change anywhere** — cache keys, fingerprints,
  compiled artifacts, and public `gtx.*` names are all preserved (the ffront
  `CachedStep`s keep caching the same functions with the same
  fingerprinters).
- **No staged-compilation-API redesign** — `translate()` + the observer are
  the largest strides toward it that don't require one (parent's stance,
  unchanged).
- **No `cartesian` changes** — but after phase 5 the build/cache/bindings
  services finally have a home `cartesian` can be repointed at (the
  per-service repointing of parent step 6; the full fold-in under
  `_internal` is its step 10).
- **No argument-model surgery** — `CompileTimeArgs`' runtime
  `offset_provider` straddle (parent item 6) stays blocked on the
  temporaries-pass TODO; it lives in `arguments.py`, which this proposal
  only relocates.

## Alternatives considered

- **Single restructured `toolchain` package (B)** — rejected as end state,
  adopted as phasing; see the design table.
- **Backend-owned push-down (C)** — rejected; premise false for the build
  half, lens retained.
- **Keep the combinators but trim (delete only dead ones)** — rejected:
  the tower is a shallow-module archetype (interface ≈ implementation), and
  every recorded ADR 0011 requirement survives its removal; trimming would
  keep the reading cost and the `Any`-typed composition.
- **Type-parameterize instead of the bindings surgery** (make `Parameter`
  generic over a type-representation parameter) — rejected: keeps the
  coupling, adds variance machinery of exactly the kind being deleted.
- **Rename `otf` in place without splitting** — rejected: fixes the
  vocabulary, forfeits the layer seam; and the split is where the
  simplifications stop regressing (tach can only enforce a line that exists
  in the module graph).
- **Big-bang single PR** — rejected: the phase ladder keeps every
  intermediate state shippable and independently revertable.

## Open questions / conflicts

- **Names needing a team vote** (in the successor ADR): `frontend`/`backend`
  vs the TODO's `frontend_transforms`/`backend_transforms`;
  `ProgramWithArgs` vs `ProgramCall`; `pool.py` vs `compiled_programs.py`.
- **Shim lifetime**: which `otf.*` deep paths does icon4py (or other
  downstream) actually import? Needs the audit the parent schedules for its
  step 9; until then all old paths get shims.
- **Should roundtrip be factored into the three-step shape?** `Roundtrip`
  is a monolithic single step today, which is why `Toolchain.backend` stays
  structurally typed and `translate()` narrows at runtime. Factoring its
  `_generate_source` into a real translation step would permit the concrete
  `CompilePipeline` annotation (a stronger, fully static seam) — small but
  genuine redesign work; needs the backend owners' call.
- **The `CompilationError` twins**: `otf/compilation/compiler.py:140`
  (subclasses `RuntimeError`) shadows `errors.exceptions.CompilationError`
  (subclasses `GT4PyError`, rewrites the message). Unifying them is
  desirable but changes what downstream `except` clauses catch and what the
  message looks like — a deliberate, small, *behavior-affecting* decision to
  schedule separately from this proposal's no-behavior-change ladder.
- **Does the observer need to see *frontend* stages too** (FOAST/PAST), or
  only the compile pipeline's three? The hook design supports both; the
  question is which stage names become API.
- **Conflicts with in-flight work**: the factory-boy direction conflict
  (`roundtrip.py:283` vs parent item 4) must be settled in the ADR before
  phase 6; [[personal/edopao/refactor-gtir-to-sdfg-lowering|the GTIR-to-SDFG
  lowering refactor]] happens *inside* the dace translation step and is
  untouched by this proposal (same seam, opposite sides);
  [[personal/egparedes/layered-architecture|the parent]] owns everything
  this note marks as "parent item N".
- **ADR logistics**: one successor ADR covering 0011 + 0017 + naming, or
  two? (Both are marked *valid* today and are drifted from the code they
  describe — 0011 still names `otf.step_types`, which no longer exists.)
