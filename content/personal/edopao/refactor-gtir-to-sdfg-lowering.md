---
title: Refactoring the GTIR-to-SDFG lowering visitor
author: edopao
tags: [dace, backend, gtir, sdfg, lowering, refactoring, visitor-pattern, translator-registry, readability, maintainability]
created: 2026-07-09
status: draft
---

> **TL;DR** Refactor the GTIR-to-SDFG lowering code in
> `src/gt4py/next/program_processors/runners/dace/lowering/` so that its two
> visitor layers dispatch through a single registry of small, self-contained
> translator classes. The aim is to make the **lowering implementation** easier
> to read, maintain, and extend; the generated SDFGs remain unchanged.

**Code baseline:** <https://github.com/edopao/gt4py/tree/dace_refactor_lowering>

## Problem / motivation

The lowering from GTIR to DaCe SDFG is implemented in
`src/gt4py/next/program_processors/runners/dace/lowering/` by **two cooperating
visitor layers**:

| Layer | Class | File | Responsibility |
| --- | --- | --- | --- |
| Program / fieldview | `GTIRToSDFG` | `gtir_to_sdfg_fieldview.py` (~1,453 lines) | Lowers `Program`, `SetAt`, `Lambda`, `FunCall`, `Literal`, `SymRef` to SDFG states and nested SDFGs. |
| Stencil / iterator | `LambdaToIterator` | `gtir_to_sdfg_iterator.py` (~1,926 lines) | Lowers the body of a field-operator lambda (iterators, shifts, reductions, local maps) to tasklets and memlets. |

Both inherit from `eve.NodeVisitor` and the separation between program-level and
stencil-level lowering is a good one. However, as the backend has grown, two
issues make the module hard to maintain:

1. **Two dispatch mechanisms are mixed and duplicated.** Both visitor layers mix
   structural pattern-matching checks (e.g. `isinstance(node.fun, gtir.Lambda)`,
   `cpm.is_applied_as_fieldop(node)`) with a string-based dispatch table
   (`_BUILTIN_HANDLERS: dict[str, str]`). A contributor must understand both and
   decide which path to extend when adding a new builtin.
2. **Primitive translators are free functions behind thin methods.** The outer
   visitor contains many one-line wrapper methods such as `_visit_as_fieldop`,
   which call free functions in separate modules and pass `self` explicitly as
   the SDFG builder. This separates dispatch from implementation, but the free
   functions are harder to test in isolation and harder to discover than a
   class-based registry.
3. **`LambdaToIterator` is too large.** At nearly 2,000 lines it mixes iterator
   mechanics, local control flow, reductions, tuple/list builtins, and generic
   math builtins in one file.
4. **Module responsibilities are blurred.** `gtir_to_sdfg_fieldview.py` contains
   the `IteratorBuilder` / `SDFGBuilder` protocols, `SubgraphContext`, the
   `GTIRToSDFG` visitor, and miscellaneous helpers. The file is a bottleneck for
   anyone trying to understand the public boundary of the lowering layer.

The goal is an internal refactor that leaves the produced SDFGs and the public
entry points unchanged, but makes the visitor pattern significantly easier to
read, extend, and test.

## Proposal

Move from a mixed structural/string dispatch model to a **single registry of
small translator classes** at each visitor layer. Each translator declares both
"I match this node" and "here is how I lower it."

### Core abstraction

```python
class BuiltinTranslator(abc.ABC):
    @abc.abstractmethod
    def matches(self, node: gtir.FunCall, ctx: SubgraphContext) -> bool: ...

    @abc.abstractmethod
    def translate(
        self,
        node: gtir.FunCall,
        ctx: SubgraphContext,
        builder: SDFGBuilder,
    ) -> gtir_to_sdfg_types.FieldopResult: ...


@dataclasses.dataclass
class TranslatorRegistry:
    translators: list[BuiltinTranslator] = dataclasses.field(default_factory=list)

    def register(self, translator: BuiltinTranslator) -> None:
        self.translators.append(translator)

    def translate(self, node, ctx, builder):
        for translator in self.translators:
            if translator.matches(node, ctx):
                return translator.translate(node, ctx, builder)
        raise LoweringError(f"No translator for {node}", node=node)
```

Order of registration encodes priority: structural translators (e.g.
`as_fieldop`) register before generic name-based ones, so the registry replaces
both dispatch mechanisms with a single `translate()` call.

### Restructure the two layers

Keep the existing two-layer architecture, but make each layer thin and delegate
all `FunCall` lowering to a registry.

**Outer layer (`GTIRToSDFG`)** keeps only:

- `visit_Program`
- `visit_SetAt`
- `visit_Lambda` (nested-SDFG construction)
- `visit_Literal`, `visit_SymRef`
- `visit_FunCall` → delegates to a `FieldviewTranslatorRegistry`

**Fieldview translators** (one class per builtin, small files):

- `AsFieldopTranslator`
- `IfTranslator`
- `ConcatWhereTranslator`
- `IndexTranslator`
- `MakeTupleTranslator`
- `TupleGetTranslator`
- `ScalarExprTranslator` (fallback)

**Inner layer (`LambdaToIterator`)** delegates `FunCall` to an
`IteratorTranslatorRegistry`.

**Iterator translators**:

- `DerefTranslator`, `ShiftTranslator`, `NeighborsTranslator`
- `MapTranslator`, `ReduceTranslator`
- `ListGetTranslator`, `MakeConstListTranslator`
- `IfIteratorTranslator`
- `MakeTupleTranslator`, `TupleGetTranslator`
- `GenericBuiltinTranslator` (fallback for math/comparison ops)

### Split files by concern

Suggested layout after the refactor:

```
lowering/
├── __init__.py
├── builder.py              # IteratorBuilder, SDFGBuilder protocols
├── context.py              # SubgraphContext
├── types.py                # FieldopData, SymbolicData
├── domain.py
├── codegen.py
├── utils.py
├── program_visitor.py      # GTIRToSDFG (outer visitor)
├── fieldview/
│   ├── __init__.py
│   ├── registry.py
│   ├── as_fieldop.py
│   ├── if_.py
│   ├── concat_where.py
│   ├── index.py
│   └── tuple_.py
├── iterator/
│   ├── __init__.py
│   ├── registry.py
│   ├── visitor.py          # LambdaToIterator
│   ├── deref.py
│   ├── shift.py
│   ├── neighbors.py
│   ├── map_.py
│   ├── reduce.py
│   ├── list_get.py
│   ├── if_.py
│   └── tuple_.py
└── errors.py               # LoweringError with debug info
```

No file should exceed ~600 lines. This layout keeps protocols, context, types,
and utilities separate from the visitors themselves, and splits the two large
files into focused modules.

### Improve context handling and diagnostics

Make `SubgraphContext` support scoped symbols explicitly:

```python
with ctx.push_symbol_scope(symbols) as lambda_ctx:
    lambda_result = self.visit(node.expr, ctx=lambda_ctx)
```

This replaces the current implicit symbol shadowing and makes it easier to
reason about which symbols are visible where.

Add a dedicated lowering exception:

```python
class LoweringError(Exception):
    def __init__(self, msg, node=None):
        super().__init__(msg)
        self.node = node
        self.location = node.location if node else None
```

This improves diagnostics when a GTIR construct is not yet supported.

## Migration plan

Refactor in four phases, keeping the dace lowering tests green at each step.

### Phase 1 — Extract registries without changing behavior

- Replace the string dispatch tables with `TranslatorRegistry` in both visitor
  layers.
- Keep existing free functions as the implementation of each translator class.
- Run the dace lowering tests.

### Phase 2 — Split oversized files

- Move iterator builtins into `lowering/iterator/`.
- Move fieldview builtins into `lowering/fieldview/`.
- Keep `program_visitor.py` and `iterator/visitor.py` thin.
- Run the dace lowering tests.

### Phase 3 — Convert primitives to translator classes

- Turn the free functions in `gtir_to_sdfg_primitives.py`,
  `gtir_to_sdfg_scan.py`, and `gtir_to_sdfg_concat_where.py` into
  `BuiltinTranslator` classes.
- Delete the `PrimitiveTranslator` Protocol if it becomes redundant.
- Run the dace lowering tests.

### Phase 4 — Context and error cleanup

- Add the symbol-scope stack to `SubgraphContext`.
- Add `LoweringError`.
- Add targeted unit tests for individual translators.

## What to keep

- **Two-layer visitor.** The program-level / stencil-level split correctly models
  the two semantic levels of GTIR.
- **`SDFGBuilder` Protocol.** It is the right boundary between the two layers.
- **Rich data descriptors.** `FieldopData`, `IteratorExpr`, `ValueExpr`,
  `MemletExpr`, `SymbolicData` make the iterator state explicit and should stay.

## Alternatives considered

1. **Keep the current visitor and only add docstrings.** Docstrings help, but
   they do not address the duplicated dispatch mechanisms or the oversized
   files.

2. **Merge the two visitor layers into one.** A single visitor would remove one
   dispatch step, but it would also erase the useful separation between
   program-level control flow and stencil-level iterator lowering. The two-layer
   model should stay.

3. **Introduce a custom IR between GTIR and SDFG.** This could decouple the
   frontend from DaCe but adds a whole new representation to maintain. It is
   out of scope for a readability-focused refactor.

## Open questions / conflicts

- **Overlap with
  [[personal/havogt/closure-variable-resolution|Closure variable resolution in
  gt4py.next]].** Closure resolution changes the GTIR that reaches the lowering
  module. The refactor should assume the current GTIR shape and be updated if
  closure resolution introduces new node types or changes symbol handling.

- **Overlap with
  [[personal/havogt/scan-redesign|Redesign of the vertical scan]].** A new scan
  representation will require a new or updated translator in the registry. The
  refactored structure should make this addition local.

- **Overlap with
  [[personal/havogt/field-data-protocol|A FieldData protocol]].** Changes to how
  fields and boundaries are represented may affect how arrays are allocated and
  origins are computed. The `SDFGBuilder` Protocol should isolate these changes
  from the translators.

- **Overlap with
  [[personal/edopao/external-memory-for-dace-arrays|External workspace memory
  for DaCe temporary arrays]].** Memory-mode transformations are applied after
  lowering and are outside the scope of this refactor, but the builder Protocol
  should continue to expose the transient-allocation hooks that the external
  memory pass relies on.

- **Overlap with
  [[personal/egparedes/layered-architecture|A layered architecture for gt4py]].**
  The refactor stays inside the DaCe runner/infrastructure layer. Splitting the
  module into `builder.py`, `context.py`, and `types.py` should make the
  internal boundaries clearer without affecting the public API.

- **Golden SDFG tests.** To ensure the refactor does not silently change
  generated code, add golden SDFG comparisons for a few representative programs
  before starting. The comparison should be structural (arrays, maps, states,
  memlets) rather than a raw string diff, because DaCe identifiers and ordering
  can be sensitive to small code changes.

- **Visitor base class.** Keep the existing `eve.NodeVisitor`-style class-based
  dispatch for the top-level visitors. Centralize the `FunCall` dispatch through
  the registry so that adding a new builtin only requires adding one translator
  class and registering it.
