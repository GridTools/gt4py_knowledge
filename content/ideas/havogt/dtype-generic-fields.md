---
title: Dtype-generic fields in gt4py.next
author: havogt
tags: [type-system, generics, dtype, frontend, foast, past, monomorphization, field-operators]
created: 2026-06-11
---

> **TL;DR** Let field operators (and the programs calling them) be generic in
> their field **dtype**, spelled with a native Python value-constrained
> `TypeVar` so the same annotation is meaningful to mypy and to the DSL
> frontend. Specialize per concrete signature (monomorphize) at call time,
> reusing the machinery the codebase already grew for scan operators.

> **Prototype**: a working end-to-end prototype (Stage 0 + Stage 1: the
> `ts.TypeVarType`, the bind/substitute utilities, the `foast_specialize`
> toolchain step, direct-call and embedded execution, plus feature tests)
> lives directly in the gt4py source tree on branch
> [`claude/gt4py-generics-investigation-aycs14`](https://github.com/graitools/gt4py/tree/claude/gt4py-generics-investigation-aycs14).
> Because it is a set of changes across `src/gt4py/next/`, it cannot be vendored
> into this knowledge garden as standalone material — follow the branch link to
> read the code. This document is the design draft that accompanies it.
>
> Related: **dims genericity** is tracked as a separate follow-up (see §8); a
> parallel exploration of dimension genericity lives on branch
> [`claude/gt4py-generic-dimensions-csn6tx`](https://github.com/graitools/gt4py/tree/claude/gt4py-generic-dimensions-csn6tx).

---

- **Status**: investigation / pre-ADR design proposal
- **Scope**: fields generic in their **dtype** (first step towards general
  generics support in `gt4py.next`), with focus on the frontend (FOAST/PAST)
  aspects that are in the way and should be improved before landing the
  feature.
- **Outcome**: a staged implementation plan; the actual decision should be
  recorded as `docs/development/ADRs/next/0023-Dtype-Generic-Operators.md`
  in the gt4py repo once reviewed.

## 1. Goal

Allow field operators (and transitively programs calling them) to be generic
in the field dtype, spelled with Python's native generics so the same
annotation is meaningful to mypy and to the DSL frontend:

```python
from typing import TypeVar
import gt4py.next as gtx

T = TypeVar("T", gtx.float32, gtx.float64)  # value-constrained TypeVar

@gtx.field_operator
def diffusion(
    a: gtx.Field[gtx.Dims[I, J], T],
    b: gtx.Field[gtx.Dims[I, J], T],
) -> gtx.Field[gtx.Dims[I, J], T]:
    return a - b

diffusion(a32, b32, out=o32, offset_provider={})  # compiles a float32 variant
diffusion(a64, b64, out=o64, offset_provider={})  # compiles a float64 variant
diffusion(a32, b64, out=o64, offset_provider={})  # error: inconsistent binding of 'T'
```

Non-goals of the first step (but kept forward-compatible, see §8): dims/rank
genericity, generic scan operators, `bound=`-style open constraint sets,
`astype(x, T)`, PEP 696 dtype defaults.

## 2. Current state of the frontend

Everything below was verified against the current sources (file:line
references are as of this investigation; see the prototype branch).

### 2.1 The static (Python typing) side half-exists already

`common.Field` is a runtime-introspectable generic protocol,
`class Field(GTFieldInterface, Protocol[DimsT, core_defs.ScalarT])`
(`src/gt4py/next/common.py:765`). `Field[Dims[I, J], T]` with a module-level
`TypeVar` is therefore *already* a valid, mypy-visible annotation today. The
gap is entirely on the DSL side: translating that annotation into the internal
type system and checking/lowering code that uses it.

### 2.2 Annotation translation rejects TypeVars

`type_translation.from_type_hint`
(`src/gt4py/next/type_system/type_translation.py:166-259`) translates
`Field[...]` annotations in a `match` over the canonical origin type. The
dtype argument must translate to a `ts.ScalarType`; a `TypeVar` instance is
not a `type`, falls through all cases and raises
`ValueError("'{type_hint}' type is not supported.")`, re-raised as
"Field dtype argument must be a scalar type". This function is the single
insertion point for TypeVar translation. (Note: it is wrapped in
`optional_lru_cache` keyed on the annotation object — the same `TypeVar`
object always maps to the same result, and distinct same-named TypeVars are
distinct keys, so the cache is helpful rather than a hazard.)

### 2.3 The internal type system has deferral but no *identity*

`ts.DeferredType(constraint=...)`
(`src/gt4py/next/type_system/type_specifications.py:41`) expresses "some type,
maybe constrained", and is accepted by `type_info.is_concretizable` /
`is_compatible_type`. But two `DeferredType` instances are unrelated: the type
system cannot express "the *same* unknown dtype in two parameters and the
return type", which is the essence of generics. This is the root cause of the
existing scan-operator hack (§2.6).

### 2.4 FOAST: eager, bottom-up, concrete-only type deduction

- Decoration is eager: `FieldOperator.__post_init__`
  (`src/gt4py/next/ffront/decorator.py:594-596`) forces FOAST parsing and full
  type deduction at decoration time. This is a deliberate UX choice (errors at
  import time) and worth preserving.
- `FieldOperatorTypeDeduction`
  (`src/gt4py/next/ffront/foast_passes/type_deduction.py`) deduces types
  bottom-up from concrete parameter types, unifying via `type_info.promote`.
  There is no constraint solving or unification with variables.
- The completeness validator
  (`FieldOperatorTypeDeductionCompletnessValidator`, same file, l.128-145)
  only rejects types failing `type_info.is_concrete`, which today only rejects
  `ts.DeferredType` (`src/gt4py/next/type_system/type_info.py:47-53`). A new
  TypeSpec node with identity would pass through unchanged — the validator is
  *not* the obstacle; the deduction rules (`promote`, dtype predicates,
  builtins) are.
- Literals are typed concretely: `visit_Constant`
  (`foast_passes/type_deduction.py:1045`) calls
  `type_translation.from_value`, so a Python `float` literal is `float64`.
  This makes `generic_field + 1.0` a real design point (§5, D3).

### 2.5 PAST: call checking is not binding-aware

`ProgramTypeDeduction.visit_Call`
(`src/gt4py/next/ffront/past_passes/type_deduction.py:225-278`) checks a
program's call to a field operator via `type_info.accepts_args` (boolean /
exception interface) and compares `type_info.return_type` against the `out`
argument with strict equality. `return_type` for a `FunctionType` returns
`func_type.returns` *unsubstituted*
(`src/gt4py/next/type_system/type_info.py:605-612`). Neither API can return
"this call is valid *under the binding T := float32*", which Stage 2 needs.

### 2.6 Existing genericity-adjacent machinery (the good news)

The codebase already contains most of the runtime mechanism, grown ad-hoc for
scan operators, plus several TODOs asking for exactly this feature:

- **Scan hack**: `type_in_program_context`
  (`src/gt4py/next/ffront/type_info.py:335-389`) gives scan operators a
  program-context signature whose args are `DeferredType(constraint=None)`,
  with `TODO(tehrengruber): What we actually want is a generic type here, but we don't have that concept yet.` (l.378). A sibling TODO at l.200 asks for a
  generic field type. Note this hack encodes *dims* genericity (scalar scan
  args become fields of caller-determined dims) — dtype TypeVars alone do not
  replace it, but the *detection and plumbing* can be unified now.
- **Call-time specialization pool**: `CompiledProgramsPool`
  (`src/gt4py/next/otf/compiled_program.py`) detects "generic" programs by
  sniffing for `DeferredType` params (`_is_generic`, l.457-474, with a TODO
  that the concept "is not properly reflected in the type system"), computes a
  per-call `arg_specialization_key` from
  `type_translation.from_value(arg)` (l.403-416), and `_compile_variant`
  passes the concrete types into the toolchain as `CompileTimeArgs`
  (l.612-631). The pool's docstring already anticipates "the type of a generic
  program". **The monomorphization cache the feature needs exists.**
- **Concrete-args seam in the toolchain**: `OperatorToProgram`
  (`src/gt4py/next/ffront/foast_to_past.py:97-179`) builds the implicit
  program for a direct operator call from the **concrete** `CompileTimeArgs`
  types. The assert at l.116-118 compares the concrete `out` type against the
  operator's declared return type — pinning down exactly where a generic FOAST
  must already have been specialized.
- **The missing piece is FOAST-side**: in `past_to_itir.py` (l.83-94) closure
  field operators are lowered via `gt_callable.__gt_gtir__()` with *no
  argument context*; the FOAST stays decoration-time-typed forever.
- **GTIR is not the blocker**: the iterator-IR type inference
  (`src/gt4py/next/iterator/type_system/type_synthesizer.py`) already
  tolerates `DeferredType` per builtin. Still, we propose never lowering
  generic GTIR (§5, D4).
- **Embedded mode is nearly free**: `FieldOperator.__call__` without a backend
  executes the original Python definition directly
  (`src/gt4py/next/ffront/decorator.py:646-691`); generic operators work in
  embedded mode as soon as decoration-time checks tolerate generic signatures.
- **mypy plugin tension**: `src/gt4py/next/type_system/mypy_plugin.py`
  currently *blurs* `float32`/`float64` → `builtins.float` (and int variants
  likewise) and substitutes dims placeholders. It erases exactly the
  information dtype generics track statically; un-blurring is follow-up work
  (§8).
- **eve generics**: `eve.GenericDataModel`/`GenericNode` exist (used in
  `gt4py.cartesian` GTC); a previous attempt at a generic FOAST `Symbol` was
  disabled due to nested-specialization issues
  (`src/gt4py/next/ffront/field_operator_ast.py:69`). The design below does
  not need generic eve datamodels.
- **Related issues**: [#1415](https://github.com/GridTools/gt4py/issues/1415)
  (the `Field[Dims[...], DType]` annotation form this design rides on) and
  [#1416](https://github.com/GridTools/gt4py/issues/1416) (dtype hierarchy
  cleanup in `_core.definitions`).

## 3. Prior art

### 3.1 Spelling: how to write a dtype-generic annotation

- **numpy.typing** converged on exactly the pattern proposed here: a
  module-level bounded/constrained TypeVar used inside a real generic class —
  `ScalarT = TypeVar("ScalarT", bound=np.floating)`; `NDArray[ScalarT]`
  ([numpy.typing docs](https://numpy.org/doc/stable/reference/typing.html)).
  The older `NBitBase` precision-genericity mechanism is deprecated in favor
  of this.
- **PEP 695** (`def op[T: (float32, float64)](...)`, Python 3.12+) cannot be
  *required* since GT4Py supports Python 3.10-3.14, but old-style
  `TypeVar("T", float32, float64)` is semantically identical for type
  checkers, runtime-introspectable via `typing.get_type_hints` + `get_args`
  (the TypeVar survives literally in the args), and forward-compatible: PEP
  695 syntax produces the same runtime objects via `__type_params__`. Caveat:
  PEP 695 TypeVars evaluate bounds/constraints *lazily* — do not assume plain
  attributes ([PEP 695](https://peps.python.org/pep-0695/)).
- **PEP 696** TypeVar defaults (`typing_extensions >= 4.12` on 3.10+) would
  later allow "unparameterized `Field` means `float64`"
  ([PEP 696](https://peps.python.org/pep-0696/)).
- **jaxtyping** achieves dtype polymorphism via dtype *groups* as runtime-
  checked constraints (`Float[Array, "n m"]`), deliberately hiding dtype from
  static checkers ([docs](https://docs.kidger.site/jaxtyping/)). **torchtyping
  died** of static-checker invisibility (its author recommends jaxtyping and
  wrote a [retrospective](https://kidger.site/thoughts/jaxtyping/)). Lesson:
  keep the annotation a real generic that mypy can see.

### 3.2 Implementation strategy: monomorphization at call time, everywhere

Every comparable system specializes per concrete signature at call time:
Numba (lazy per-signature compile + specialization cache + ranked dispatch,
[docs](https://numba.readthedocs.io/en/stable/reference/jit-compilation.html)),
Taichi templates ([docs](https://docs.taichi-lang.org/docs/meta)), Triton
per-signature compilation, TensorFlow `tf.function` trace caches keyed on
dtypes/shapes ([paper](https://arxiv.org/abs/1903.01855)), DaCe symbol solving
per call
([docs](https://spcldace.readthedocs.io/en/stable/frontend/daceprograms.html)).
The theory agrees: with a simply-typed object language (which GTIR is),
monomorphization is the natural elaboration of staged polymorphism
([Kovács, *Staged Compilation with Two-Level Type Theory*, ICFP
2022](https://arxiv.org/pdf/2209.09729)).

### 3.3 Pitfalls reported by prior art

- **Silent promotion**: Numba's safe-conversion ranking silently turns `int32`
  args into `float64` kernels — for weather/climate code,
  exact-match-or-error is the right default.
- **Unconstrained templates scale badly**: Taichi's `ti.template()` (no
  constraint surface) led to poor diagnostics and a type-system redesign
  ([taichi#7495](https://github.com/taichi-dev/taichi/issues/7495)). Prefer
  constrained TypeVars.
- **Cache-key completeness**: the specialization cache key must include the
  full substitution (the TF Eager paper discusses trace-cache keying bugs).
  GT4Py's `arg_specialization_key` already hashes all argument types.
- **typing-runtime drift**: `typing`/`typing_extensions` version differences
  bit eve before ([#968](https://github.com/GridTools/gt4py/issues/968));
  keep TypeVar introspection inside `eve.extended_typing`.
- **mypy holes**: constrained dtype TypeVars in numpy-like generics have known
  mypy gaps ([mypy#17228](https://github.com/python/mypy/issues/17228));
  budget for extending the existing GT4Py mypy plugin rather than expecting
  full static enforcement on day one.

## 4. Proposed design

### 4.1 New type: `ts.TypeVarType`

```python
class TypeVarType(DataType):
    name: str                            # identity within one operator signature
    constraints: tuple[ScalarType, ...]  # from TypeVar.__constraints__, written order
```

- Subclasses `DataType` so it fits *unchanged* into `FieldType.dtype`
  (widened to `ScalarType | ListType | TypeVarType`), `TupleType.types`,
  `NamedCollectionType` members, and `foast.Symbol`'s
  `Union[SymbolT, ts.DeferredType]` (whose `DataTypeT` is bound to
  `ts.DataType`).
- Identity is the name, scoped to one operator signature; two *distinct*
  TypeVar objects with the same name in one signature are detected and
  rejected at parse time. As an eve frozen DataModel it gets deterministic
  `eq`/`hash`/`content_hash` for free (needed for pool cache keys and
  `fingerprint_stage`).
- `ts.DeferredType` is *not* replaced: it keeps meaning "not yet inferred",
  while `TypeVarType` means "universally quantified over the constraint set".

### 4.2 Decisions

- **D1 — Decoration-time body checking: full symbolic check with opaque
  `TypeVarType`.** The body is type-checked once, at decoration time, with `T`
  treated as an opaque scalar. Rejected alternatives: *skip-until-
  instantiation* breaks the decoration-time-errors UX and `__gt_type__`
  consumers (PAST needs a generic signature at program decoration anyway);
  *finite monomorph-check* (instantiate every constraint member) duplicates
  compile work at import time and reports errors in instantiated vocabulary
  ("float32 mismatch") instead of the user's ("T"). The finite check is still
  valuable — as a *test-suite* cross-check of substitution soundness.
- **D2 — v0 supports value-constrained TypeVars only**
  (`TypeVar("T", float32, float64)`), not `bound=`-only. Constraint sets make
  dtype predicates decidable (`is_arithmetic(T)` ⇔ all members arithmetic) and
  the variant set finite, enabling eager precompilation via the existing
  `CompiledProgramsPool.compile()` API. `bound=` is follow-up (§8).
- **D3 — Strict no-promotion policy.** `promote(T, T) = T`; mixing `T` with a
  concrete scalar/dtype (including literals: `a * 2.0`) is a decoration-time
  error that names the TypeVar, the concrete type, and the remediation.
  `astype(x, T)` is the designated remediation but is itself a fast-follow,
  not v0 (it requires a `ConstructorType` over `TypeVarType`; the
  `_visit_astype` path currently asserts a concrete `ScalarType`). This will
  be the #1 ergonomics complaint — we pre-commit to strict-first with a named
  follow-up ("generic literals" that adapt if exactly representable in all
  constraint members), rather than inheriting Numba-style silent promotion.
- **D4 — Monomorphize at FOAST level; never lower generic GTIR.**
  Specialization is direct type substitution over the typed FOAST (cheap),
  with a full re-run of `FieldOperatorTypeDeduction` as a soundness backstop
  under `__debug__`. Rejected: lowering generic FOAST and concretizing at GTIR
  level — `foast_to_gtir` bakes dtypes into literals and `cast_` calls, GTIR
  has no syntax for "dtype of param x", and genericity would leak into every
  GTIR transform.
- **D5 — Binding becomes a first-class `type_info` utility.**
  `bind_type_vars(params, args) -> dict[str, ts.ScalarType]` (structural
  match, consistency check, exact-match policy, good error strings) and
  `substitute_type_vars(t, binding)` (explicit recursion over every TypeSpec).
  `accepts_args` keeps its boolean interface; callers needing the binding use
  the new API.

### 4.3 Call-path designs

**(i) Direct `op(a, b, out=o, ...)` with a backend.**
`FieldOperator.__call__` → `CompiledProgramsPool`. `_is_generic` (extended via
the new first-class predicate, §6) is true; the per-call
`arg_specialization_key` already keys the cache on the full substitution;
`_compile_variant` already forwards concrete types as `CompileTimeArgs`. The
only new piece is a **`foast_specialize` toolchain step** in
`backend.Transforms.step_order` (`src/gt4py/next/backend.py`), inserted after
`func_to_foast` in both the `DSLFieldOperatorDef` and `FOASTOperatorDef`
branches: compute the binding by matching `inp.args` against the FOAST
signature, substitute throughout the tree, no-op when not generic. Everything
downstream (`OperatorToProgram` including its l.116 return-type assert,
`past_lint`, `past_to_itir`) then runs unchanged on a concrete artifact.

**(ii) Generic operator called inside a concrete `@program`** (the hard part).
At program decoration the binding is fully static: the program's params are
concrete, so `function_signature_incompatibilities_fieldop` and
`return_type_fieldop` (`src/gt4py/next/ffront/type_info.py:84-153`) become
bind-and-substitute (binding computed *before* `promote_zero_dims`, whose
dtype-equality comparison would otherwise misfire on `Field[[], T]` params vs
scalar args). For lowering, where `__gt_gtir__()` has no argument context, add
a **PAST monomorphization pass** run inside `past_to_gtir` before
`ProgramLowering.apply` (closure-var values are available there): for every
`past.Call` whose callee type is generic, recompute the binding from the typed
call-site args, mangle the callee name per binding (e.g.
`diffusion__float32`), and swap the closure-var entry for a specialized
callable obtained via a new `GTCallable.__gt_specialize__(binding)` method
(implemented by `decorator.FieldOperator` and `foast_to_past.ItirShim` by
applying the FOAST substitution pass and renaming the node). Two different
bindings of one operator in one program naturally become two GTIR
`FunctionDefinition`s; the existing lowering loop is untouched.

**(iii) Embedded mode.** Nearly free: the original Python definition is
executed on real fields. Requirements: decoration tolerates generic signatures
(Stage 1) and `past_process_args._validate_args` / the `__debug__` validation
in `Program.__call__` go through the binding-aware checks (Stage 2). Locked in
by running the feature tests across the backend matrix including `None`
(embedded).

## 5. What is "in the way": pre-landing frontend improvements (Stage 0)

These are independently landable, each PR-sized, and valuable on their own:

1. **First-class genericity predicate** — add
   `type_info.is_generic(t: ts.TypeSpec) -> bool` (recursive; true for
   `DeferredType` now, `TypeVarType` later) and replace the `DeferredType`
   sniffing in `CompiledProgramsPool._is_generic`
   (`otf/compiled_program.py:457-474`), resolving its TODO. The scan
   `DeferredType(constraint=None)` hack in `type_in_program_context` *stays*
   for now — it encodes dims genericity and is only fully removable with dim
   variables (§8) — but its detection routes through the new predicate and its
   TODO is updated to reference the ADR.
1. **`ts.TypeVarType` + bind/substitute utilities** (D5) — inert until
   `from_type_hint` produces them, so safely landable ahead of the feature.
   Unit tests for hashing/equality/`content_hash` determinism.
1. **Builtin-deduction audit / guard rails** — sweep
   `foast_passes/type_deduction.py` and `ffront/type_info.py` for silent
   `isinstance(..., ts.ScalarType)` assumptions on dtypes (`_visit_astype`,
   reductions, `where`, math builtins, `with_altered_scalar_kind`,
   `promote_zero_dims`) and convert them into explicit, well-worded
   `DSLError`s per `CODING_GUIDELINES.md`. Pure error-quality work that
   de-risks Stage 1: anything unaudited fails loudly instead of mistyping.

## 6. Staged implementation plan

**Stage 1 — minimal dtype generics: direct operator calls + embedded.**

- `type_translation.from_type_hint`: a branch for
  `isinstance(canonical_type, typing.TypeVar)` reading `__constraints__`
  (reject empty-constraint / `bound=`-only TypeVars in v0 with a clear
  message); relax the Field-dtype check to accept `TypeVarType`. Detect two
  distinct same-named TypeVars in one signature during `func_to_foast`
  annotation processing.
- `type_info.promote` + dtype predicates (`is_arithmetic`, `is_logical`,
  `is_floating_point`, `is_integral`): `T` with itself promotes to itself;
  `T` with anything else raises (D3); predicates evaluate over the constraint
  set (D2).
- New FOAST pass `ffront/foast_passes/specialize_type_vars.py` (eve
  `NodeTranslator` mapping every node `type` through `substitute_type_vars`;
  re-runs the completeness validator; full re-deduction under `__debug__`).
- `backend.py`: `foast_specialize` workflow step + `step_order` insertion.
- Guard: decorating a *scan* operator with TypeVar annotations raises "not yet
  supported" (don't silently mistype `init` against `T`).
- Tests: `tests/next_tests/unit_tests/` for type translation (TypeVar
  translation and error cases), type deduction (symbolic body typing,
  `T` × concrete mixing errors, tuple-of-`T` params), bind/substitute; new
  feature test file
  `tests/next_tests/integration_tests/feature_tests/ffront_tests/test_generic_dtype_operators.py`
  across the backend matrix using the `cases` framework, with a new
  `USES_GENERIC_DTYPE` marker added to `tests/next_tests/definitions.py` (per
  ADR 0015; no exclusions expected since backends only ever see concrete GTIR,
  but the marker must exist so a backend *can* be excluded). Cover: two calls
  with different dtypes hit two pool variants, cache-key distinctness, eager
  `.compile()` of all constraint members, embedded mode.

**Stage 2 — generic operators called from concrete programs.**

- Binding-aware `function_signature_incompatibilities_fieldop` /
  `return_type_fieldop` in `ffront/type_info.py` (order: canonicalize → bind →
  substitute → existing zero-dim promotion and checks); PAST `visit_Call`'s
  `out` mismatch error prints the binding.
- New `ffront/past_passes/monomorphize_generic_calls.py` + invocation in
  `past_to_itir.past_to_gtir`; name-mangling helper;
  `GTCallable.__gt_specialize__` in `ffront/gtcallable.py`, implemented by
  `FieldOperator` and `ItirShim`.
- Tests: unit tests for binding deduction and double-instantiation with
  mangling, conflicting-binding errors; feature test with one program calling
  the same operator at `float32` and `float64` in one body, full matrix.

**Stage 3+ — follow-ups (each its own design slice).** Generic *programs*
(a PAST-level mirror of `foast_specialize`; the pool path already handles
them); `astype(x, T)` and generic scalar constructors; generic scan operators
(needs `init: T` coercion semantics — nothing in Stages 0-2 hardcodes
`FieldOperatorType` in the utilities); `bound=` TypeVars (infinite constraint
sets, predicates by bound, no eager precompile); **dims genericity** (the true
fix for the scan `DeferredType` hack and `_scan_param_promotion`'s fabricated
`Dimension("...")`); mypy-plugin un-blurring of `float32`/`float64`
(coordinate with #1415/#1416); PEP 696 dtype defaults.

## 7. Risks and open questions

1. **TypeVarType identity & hashing** — name-keyed identity per signature;
   same-named distinct TypeVar objects rejected at parse time; constraint
   tuple order preserved as written (same TypeVar object ⇒ same order, so
   `content_hash` determinism holds).
1. **Literal mixing ergonomics** — `a * 2.0` errors in v0 (D3); expect this to
   dominate user feedback; the ADR should pre-commit to the strict policy with
   the named "generic literals" follow-up.
1. **Builtin long tail** — `where`, `broadcast`, reductions, `concat_where`,
   math builtins, neighbor fields (`ListType.element_type` substitution),
   `NamedCollectionType` members of type `T`. Mitigated by the Stage 0 guard
   rails: unaudited paths fail loudly, then widen incrementally.
1. **Int-literal magnitude typing** — `type_translation.from_value` picks
   `int32` vs `int64` by value magnitude; a Python `int` scalar argument may
   bind `T` surprisingly. Exact-match policy makes it visible; document it.
1. **Zero-dim promotion ordering** — `promote_zero_dims` compares dtypes by
   equality; must run *after* binding (or be TypeVar-aware).
1. **Substitution soundness** — substitution-without-re-deduction is sound
   only if every deduction rule is parametric in the dtype; covered by the
   `__debug__` re-deduction backstop plus the finite monomorph cross-check in
   the test suite (D1).
1. **DaCe `Program` override** — `decorator.py` swaps in a DaCe-specific
   `Program` class; verify the pool/specialization path is shared, or exclude
   `dace` in Stage 1 via the `USES_GENERIC_DTYPE` marker.

## 8. Proposed ADR

`docs/development/ADRs/next/0023-Dtype-Generic-Operators.md`, covering: the
user-facing spelling (module-level value-constrained TypeVars; why not strings
or `Annotated` forms; relation to #1415); the `ts.TypeVarType` design
(`DataType` subclass, name identity, constraint sets, relation to — and
non-replacement of — `DeferredType`); decisions D1-D5 with rejected
alternatives; the monomorphization strategy (FOAST-level substitution, the two
call paths, name mangling, cache-key story); the strict promotion policy; and
an explicit out-of-scope list with forward-compatibility notes (generic scan,
`bound=`, dim variables, PEP 696, mypy-plugin un-blurring).
