---
title: "Layered architecture — current-state analysis & references"
author: egparedes
tags: [architecture, layering, tach, infrastructure, over-engineering, otf, workflow, dependencies, jax, research]
created: 2026-06-15
draft: false
---

> Appendix to [[ideas/egparedes/layered-architecture|A layered architecture for
> gt4py with enforced public/internal decoupling]]. Background only — the
> recommendation lives in the main proposal. Findings are against the current
> `main` of [GridTools/gt4py](https://github.com/GridTools/gt4py); paths are under
> `src/gt4py/`.

---

## 1. Current package map

```
gt4py/
  __init__.py        # exports only: eve, storage
  _core/             # DType/Device/Scalar defs, filecache, locking, ndarray_utils
  eve/               # datamodel / NodeVisitor / codegen / traits framework
  storage/           # TensorBuffer, allocators (cartesian-flavoured)
  cartesian/         # legacy GTScript stack
    config.py  caching.py  stencil_builder.py  stencil_object.py  gtscript.py
    frontend/        # GTScript -> DEFIR -> GTIR
    gtc/             # GTIR, OIR, {numpy,dace,gtcpp} codegen, passes/
    backend/         # base.py (Backend ABC), numpy/dace/gtcpp/debug backends
  next/              # modern field-operator stack
    common.py  config.py  backend.py  constructors.py
    custom_layout_allocators.py
    type_system/             # type_info, type_specifications, type_translation, mypy_plugin
    ffront/                  # decorator, func_to_foast/past, *_ast, *_passes/, foast_to_gtir, past_to_itir
    iterator/                # ir.py, builtins, embedded, transforms/ (~40 passes), type_system/
    otf/                     # workflow, toolchain, definitions, stages, recipes,
                             #   binding/ (cpp_interface, nanobind), compilation/ (cmake, cache, importer)
    program_processors/      # codegens/gtfn, runners/{gtfn,roundtrip,dace}, formatters
    embedded/                # nd_array_field, operators, context
    instrumentation/         # metrics, hooks, gpu_profiler
    errors/
```

Proposed layer assignment (see main proposal for the rule and for the
**first-steps scope**: `gt4py.next` + shared infra/utils now; `eve` becomes
internal; `gt4py.cartesian` is left as-is and only repointed at the shared infra,
folded into the layers later):

| Layer | Gets |
| --- | --- |
| utils | `eve` (now **internal** — IR-tree framework, name not preserved), `_core` (→ `defs`), pure-python helpers |
| infrastructure | `next/otf/{compilation,binding}`, `next/otf/workflow`+`toolchain`, `config`, allocators (`storage` + `next/custom_layout_allocators`), `next/instrumentation` — shared by next and cartesian |
| core | `next/{common,ffront,iterator,type_system,embedded}`, `next/program_processors/codegens`, runner orchestration. *(cartesian's frontend+IR+backends join here in the later step.)* |
| public_api | the `__init__` facades of `gt4py`, `gt4py.next`, `gt4py.storage`. `gt4py.cartesian` stays public but unrestructured; `gt4py.eve` is dropped (internal) |

## 2. Dependency tangles (why the layering is needed)

- **`backend ↔ ffront` cycle.** `next/backend.py` imports from `next/ffront/`;
  `next/ffront/decorator.py` imports `next.backend` and `next.otf`. Both sit in
  the same conceptual area and the cycle is only survivable via `TYPE_CHECKING`
  guards. Splitting *codegen/runner (core)* from *build service (infrastructure)*
  removes the upward edge: infrastructure no longer needs the DSL.
- **`otf` straddles two layers.** `next/otf/` contains both IR-aware translation
  steps (need `iterator.ir`) and DSL-agnostic build/cache/binding plumbing. As one
  package it can never be cleanly placed; as two (`codegens` in core,
  `infra/build`+`infra/bindings`+`infra/pipeline`) it can.
- **Backends are discovered by import, not registered.** `gtfn_cpu`/`gtfn_gpu`
  are hard-wired in `next/__init__`; the dace runner is imported lazily inside
  `decorator.py`. No central registry; adding a backend means editing `__init__`.
- **Four type systems, ~0 sharing.** `_core/definitions.py` (scalars/devices),
  `next/type_system/` (DSL types), `next/iterator/type_system/` (ITIR types),
  `cartesian/type_hints.py` (GTScript types). Overlapping concepts, separate code.
- **Dual config/caching/backend bases.** `cartesian/config.py` vs `next/config.py`;
  `cartesian/caching.py` vs `next/otf/compilation/cache.py`;
  `cartesian/backend/base.py` (ABC) vs `next/backend.py` (dataclass). No shared base.

## 3. Infrastructure over-engineering catalog

### 3.1 `otf/workflow.py` — five classes for a linear pipeline (~283 lines)

```python
class Workflow(Protocol[StartT_contra, EndT_co]):
    def __call__(self, inp: StartT_contra) -> EndT_co: ...

class ChainableWorkflowMixin(Workflow[StartT, EndT_co], Protocol[StartT, EndT_co]):
    def chain(self, next_step: Workflow[EndT_co, NewEndT]) -> ChainableWorkflowMixin[StartT, NewEndT]:
        return make_step(self).chain(next_step)
```

Plus `NamedStepSequence`, `MultiWorkflow` (abstract `step_order()` overridden per
subclass), `StepSequence`, `CachedStep`, `SkippableStep`, and a
`ReplaceEnabledWorkflowMixin`. The actual usage (`recipes.OTFCompileWorkflow`,
`backend.Backend`) is a fixed linear pipeline:

```python
@dataclasses.dataclass(frozen=True)
class OTFCompileWorkflow(workflow.NamedStepSequence):
    translation: definitions.TranslationStep
    bindings:    workflow.Workflow[stages.ProgramSource, stages.CompilableProject]
    compilation: workflow.Workflow[stages.CompilableProject, stages.ExecutableProgram]
    decoration:  workflow.Workflow[stages.ExecutableProgram, stages.ExecutableProgram]
```

**Simplification.** `Workflow = Callable[[T], U]`; one `Pipeline` dataclass that
calls steps in order; `compose(*steps)`; `cached(step, key=...)` wrapper; use
`dataclasses.replace` instead of a replace-mixin. The contravariant/covariant
TypeVars exist mostly to thread the combinators and disappear with them.

### 3.2 `otf/toolchain.py` — composition dressed as Protocol classes

```python
@dataclasses.dataclass(frozen=True)
class DataOnlyAdapter(ChainableWorkflowMixin, ReplaceEnabledWorkflowMixin,
                      workflow.Workflow[ConcreteArtifact[S, ArgsT], ConcreteArtifact[T, ArgsT]],
                      Generic[ArgsT, S, T]):
    step: workflow.Workflow[S, T]
    def __call__(self, inp): return ConcreteArtifact(data=self.step(inp.data), args=inp.args)
```

**Simplification.** A `functools.partial` or a 3-line wrapper function; no
Protocol/Generic scaffolding.

### 3.3 factory-boy backend builders

`program_processors/runners/dace/workflow/factory.py` (and siblings) build plain
dataclasses through `factory.Factory` with `LazyFunction`/`LazyAttribute`/
`Trait`/`SubFactory`:

```python
class DaCeWorkflowFactory(factory.Factory):
    class Meta: model = recipes.OTFCompileWorkflow
    class Params:
        auto_optimize: bool = False
        cached_translation = factory.Trait(
            translation=factory.LazyAttribute(lambda o: workflow.CachedStep(o.bare_translation, ...)))
    translation = factory.LazyAttribute(lambda o: o.bare_translation)
    compilation = factory.SubFactory(DaCeCompilationStepFactory, ...)
```

**Simplification.** A builder function with ordinary defaults + a `cached: bool`
flag returning the dataclass directly; removes the `factory` runtime dependency.

### 3.4 `custom_layout_allocators.py` — Protocol + `TypeIs` zoo

Six `Protocol`s (`FieldBufferAllocatorProtocol`,
`FieldBufferAllocatorFactoryProtocol`, …) plus ~ten runtime guards such as:

```python
def is_field_allocator(obj) -> TypeIs[FieldBufferAllocatorProtocol]:
    return hasattr(obj, "__gt_device_type__") and hasattr(obj, "__gt_allocate__")
```

all validating two magic methods. **Simplification.** A registry
`dict[DeviceType, Allocator]` with `register(device, alloc)` / `get(device)`;
keep one Protocol for the allocator call signature if useful, drop the guards.

### 3.5 Two hashing paths for the same data

`otf/stages.py` has `compilation_hash(...)` built on `hash((...))` *and*
`fingerprint_compilable_program(...)` built on `content_hash((...))` (an eve
util) — for the same compilable program, with no documented contract for which to
use when. **Simplification.** One function, one strategy, documented.

### 3.6 Type-alias chains in `otf/definitions.py`

```python
ConcreteProgramDef: TypeAlias = toolchain.ConcreteArtifact[IRDefinitionT, ArgsDefinitionT]
CompilableProgramDef: TypeAlias = ConcreteProgramDef[itir.Program, arguments.CompileTimeArgs]
class TranslationStep(workflow.ReplaceEnabledWorkflowMixin[CompilableProgramDef,
                      stages.ProgramSource[CodeSpecT]], Protocol[CodeSpecT]): ...
```

Reading the pipeline means unfolding `TypeVar → TypeAlias → Protocol → Generic →
class`. **Simplification.** Concrete dataclasses (e.g. a `CompilableProgram`
holding `data: itir.Program` + `args: CompileTimeArgs`) where variance is not
actually exercised.

### 3.7 Dataclass containers that are really records

~38 `@dataclass(frozen=True)` containers across `otf/` (`ProgramSource`,
`BindingSource`, `CompilableProject`, …) with no methods — `NamedTuple`/
`TypedDict`-shaped. Not urgent, but worth collapsing where they cross the
infra boundary.

## 4. Third-party dependency inventory (which subsystems pull what)

Core runtime deps include: `numpy`, `attrs`, `array-api-compat`, `boltons`,
`cytoolz`/`toolz`, `frozendict`, `xxhash`, `lark` (GTScript parsing), `jinja2` +
`mako` (two template engines), `black` (codegen formatting), `cmake` + `ninja`
(build), `nanobind` + `pybind11` (two binding generators), `gridtools-cpp` (GTFN
backend), `dace` (heavy, DaCe backend — currently non-optional), `factory-boy`
(only used to build dataclasses), `filelock`, `devtools`, `tabulate`,
`cached-property`, `typing-extensions`.

Optional groups: `next`/`cartesian` pull `jax`, `scipy`, `hypothesis`;
`cuda12/13`, `rocm6/7`, `jax-cuda12/13` pull `cupy` / `jax[cuda]`.

Layering observations:
- `dace` and `factory` sit in *core* deps but serve *one* backend / *one* builder
  pattern respectively — candidates to make optional (dace) or remove (factory).
- Two template engines (`jinja2`, `mako`) and two binding generators (`nanobind`,
  `pybind11`) reflect the cartesian/next split; the infra layer is where one of
  each could eventually win.
- `xxhash` + `content_hash` + `hash()` is more hashing surface than the single
  caching contract (§3.5) needs.

## 5. `tach` layers primer

[`tach`](https://docs.gauge.sh/) is a fast, import-graph linter for Python that
enforces module boundaries with **no runtime cost** — it parses imports and checks
them against declared rules (run in CI next to ruff/mypy).

Its [layers](https://docs.gauge.sh/usage/layers/) feature is exactly the
"higher-may-import-lower" rule: declare an ordered `[[layers]]` list, assign each
module a `layer`, and `tach check` rejects any import from a lower layer up to a
higher one. Same-layer imports can be allowed or required-explicit; `strict` can
be enabled per module so only declared public members are importable across
boundaries; dependencies can be marked `deprecated` to warn without failing
(useful during migration). A minimal example:

```toml
# Ordered high -> low: each layer may import only from layers after it.
layers = ["public_api", "core", "infrastructure", "utils"]

[[modules]]
path = "myapp.api"
layer = "public_api"
[[modules]]
path = "myapp.core"
layer = "core"
[[modules]]
path = "myapp.infra"
layer = "infrastructure"
[[modules]]
path = "myapp.utils"
layer = "utils"
```

With this, `myapp.utils` importing `myapp.core` fails; `myapp.core` importing
`myapp.utils` passes. The full gt4py mapping is in the main proposal. Adoption is
incremental: start non-strict, add modules and tighten as the `_internal` moves
land. Alternatives (Import Linter, pydeps, hand-rolled AST checks) can express
similar constraints but lack a first-class ordered-layers primitive.

## 6. JAX as the architectural reference

JAX's layout is the model for the public/internal split:

```
jax/
  __init__.py          # ~100 explicit re-exports: from jax._src.api import jit as jit
  numpy/__init__.py    # ~400 explicit re-exports from jax._src.numpy.*
  lax/__init__.py      # re-exports from jax._src.lax.*
  _src/                # ALL implementation
    api.py  core.py  numpy/  lax/  ...
```

Conventions worth copying:

1. **`_src/` holds everything; the public namespace only re-exports.** The
   dependency is one-way (`jax/*` → `jax/_src/*`, never back). gt4py's `_internal`
   plays the `_src` role.
2. **Explicit `from x import y as y`** re-exports — the redundant `as y` marks the
   name public for type checkers (PEP 484) and makes the public surface an
   auditable list.
3. **Deprecation via module `__getattr__`.** Old/relocated names are intercepted
   at import with a warning instead of breaking, letting internals move freely:

   ```python
   _deprecations = {"old_name": ("Use new_name instead.", new_name)}
   def __getattr__(name):
       if name in _deprecations:
           msg, value = _deprecations[name]
           warnings.warn(msg, DeprecationWarning, stacklevel=2)
           return value
       raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
   ```

4. **Domain-grouped public submodules** (`jax.numpy`, `jax.lax`, `jax.random`)
   over thin `_src` implementations — the analogue of `gt4py.next` /
   `gt4py.cartesian` facades over `gt4py._internal.{next,cartesian}`.
5. **Concrete types at the boundary**; the generic/variance machinery (where it
   exists) stays internal. This is the principle behind infrastructure
   simplifications §3.1, §3.2 and §3.6.

The net lesson: JAX keeps a large, fast-moving internal codebase shippable by
making the *public surface a deliberate, thin, explicitly-enumerated facade* and
keeping every cross-cutting service concrete. gt4py can adopt the same with
`_internal` + facades, and have `tach` guarantee the one-way dependency that JAX
maintains by convention.
