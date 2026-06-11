---
title: "Dtype-generic fields: prior-art research"
author: havogt
tags: [type-system, generics, dtype, prior-art, research, monomorphization, typevar, pep695, numpy, jax, numba]
created: 2026-06-11
---

> **Appendix** to [[ideas/havogt/dtype-generic-fields|Dtype-generic fields in gt4py.next]]
> (§3 there is the short version). This document records the full prior-art
> research that informed the design: Python's own generics mechanisms, how
> comparable array/DSL frameworks handle dtype genericity, the static-typing
> situation for dtype-generic numpy code, the theory behind staged/two-level
> typing, and GT4Py's own related issues.

## 1. Python's generics mechanisms at runtime

### 1.1 Old-style TypeVars (still accepted)

A module-level value-constrained TypeVar used inside a real generic class still
works, and was the only portable option before Python 3.12:

```python
FloatT = TypeVar("FloatT", float32, float64)        # value-constrained
# or: ScalarT = TypeVar("ScalarT", bound=np.floating)  # bounded (open set)

def op(a: Field[Dims[I, J], FloatT]) -> Field[Dims[I, J], FloatT]: ...
```

Runtime introspection works without any special machinery: the TypeVar
survives literally inside `typing.get_args(...)` of a parameterized generic
(after `typing.get_type_hints`), and its constraint set / bound are plain
attributes (`__constraints__`, `__bound__`).

Semantics relevant to the design: a *value-constrained* TypeVar
(`TypeVar("T", float32, float64)`) requires each use of the generic to resolve
to **exactly one** of the listed types — semantically much closer to
monomorphization than `bound=` (which admits any subtype). This is why the
v0 implementation supports only value-constrained TypeVars: the variant set is
finite and decidable.

- [typing module documentation](https://docs.python.org/3/library/typing.html)
- [typing-inspection (pydantic team)](https://typing-inspection.pydantic.dev/latest/usage/)
  — a library encapsulating version-dependent edge cases of runtime annotation
  inspection; useful reference if GT4Py's handling ever needs to leave
  `eve.extended_typing`
  ([announcement discussion](https://discuss.python.org/t/typing-inspection-a-library-to-inspect-type-annotations-at-runtime/81301)).

### 1.2 PEP 695 (Python 3.12+): `def op[T: (float32, float64)](...)`

PEP 695 syntax is the **preferred** spelling now that GT4Py targets 3.12+, and
is forward-compatible with the old-style spelling: it produces the same runtime
`TypeVar` objects, reachable via the new `__type_params__` attribute, and
`Generic[T]` base classes are implied. It also scopes the type parameter to the
operator, matching the per-signature identity the design wants. Two runtime
caveats when introspecting them:

- **Lazy bounds/constraints**: PEP 695 TypeVars store bounds and constraints
  as code objects evaluated on first access of `__bound__`/`__constraints__`
  — they are not plain attributes, and the lazy scoping rules differ subtly
  from old-style TypeVars.

- The implementation details are documented in
  [Jelle Zijlstra's PEP 695 implementation write-up](https://jellezijlstra.github.io/pep695.html)
  and the [CPython tracking issue](https://github.com/python/cpython/issues/103763).

- [PEP 695](https://peps.python.org/pep-0695/)

### 1.3 PEP 696 (TypeVar defaults)

`TypeVar("DT", float32, float64, default=float64)` would let an
unparameterized `Field` mean `Field[..., float64]` — attractive for backwards
compatibility of annotations that don't mention a dtype. Native in `typing`
from Python 3.13 (Python 3.12 still needs `typing_extensions >= 4.12`); runtime
introspection via `__default__` / `has_default()` with a `NoDefault` sentinel.

Known pitfalls: defaults interact awkwardly with optional/implicit
parameterization in user code
([cthoyt's "default typing dilemma" post](https://cthoyt.com/2025/04/19/python-default-typing-dilemma.html)),
and pydantic needed careful special-casing to detect defaults robustly across
CPython/typing_extensions variants
([pydantic PR #9426](https://github.com/pydantic/pydantic/pull/9426)).

- [PEP 696](https://peps.python.org/pep-0696/) ·
  [typing_extensions docs](https://typing-extensions.readthedocs.io/en/stable/)

## 2. How other frameworks handle dtype genericity

### 2.1 jaxtyping — constraint groups, runtime checking, static invisibility

The closest analogue for annotation *spelling*: `Float[Array, "n m"]`, where
the dtype group is a hierarchy (`Shaped > Num > Inexact > Float > Float32/Float64/BFloat16`, ...). Two design points directly relevant to GT4Py:

- *Dtype groups as bounds*: users get "any float" genericity without TypeVars
  — the annotation is a constraint checked at runtime, and each call is
  implicitly specialized. This is the lighter-weight alternative to TypeVars,
  but it cannot express "these two arguments have the *same* dtype", which is
  the essence of the GT4Py use case.
- *Static/runtime split*: static checkers see only the array part of
  `Float[T, "n m"]` (mypy/pyright treat it as plain `T`/`Array`), while
  jaxtyping's runtime isinstance machinery checks dtype groups and shapes —
  since v0.2.32 including TypeVar bounds/constraints, with cross-argument
  consistency enforced by a per-call dictionary. jaxtyping thus deliberately
  *sacrifices* static dtype checking; GT4Py's `Field[Dims[...], T]` spelling
  keeps the dtype statically visible instead.

JAX itself deliberately punted on dtype/shape generics in its core annotations
([JEP 12049, type-annotation roadmap](https://docs.jax.dev/en/latest/jep/12049-type-annotations.html)).

- [jaxtyping docs](https://docs.kidger.site/jaxtyping/) ·
  [array annotations](https://docs.kidger.site/jaxtyping/api/array/) ·
  [v0.2.32 release notes (TypeVar support)](https://github.com/patrick-kidger/jaxtyping/releases/tag/v0.2.32)

### 2.2 Numba — the canonical call-time monomorphization model

With no signature, `@jit` is lazy: each call computes a type signature from
the concrete arguments, reuses a compatible specialization if one exists,
otherwise compiles a new one keyed on that signature. Eager mode
(`@jit("float64(float64)")`, or a list of signatures) restricts to declared
monomorphs — the analogue of GT4Py's `.compile()` precompilation, which
value-constrained TypeVars make finite.

Dispatch among specializations ranks candidates by
`(unsafe conversions, safe conversions, same-kind promotions, exact matches)`
and is resolved at compile time for nested jitted calls. Documented pitfall:
implicit promotion (e.g. `int` → `float64`) can silently change precision —
the reason the GT4Py design chose **exact-match-or-error** binding.

- [Numba JIT compilation reference](https://numba.readthedocs.io/en/stable/reference/jit-compilation.html) ·
  [polymorphic dispatching internals](https://numba.readthedocs.io/en/stable/developer/dispatching.html)

### 2.3 Taichi — templates without a constraint surface

`ti.template()` argument hints make every kernel a template; instantiation is
keyed on a "template signature" derived from the concrete field passed (dtype
and shape are compile-time metadata), with binary reuse when the signature
repeats. Notably, Taichi found that "untyped template + specialize on call"
without a declared constraint surface scales poorly for diagnostics, and wrote
a type-system redesign issue — evidence in favor of constrained TypeVars over
an "anything goes" template marker.

- [metaprogramming docs](https://docs.taichi-lang.org/docs/meta) ·
  [kernel compilation lifecycle](https://docs.taichi-lang.org/docs/compilation) ·
  [type-system redesign issue taichi#7495](https://github.com/taichi-dev/taichi/issues/7495)

### 2.4 Triton — per-signature compilation plus constexpr

Kernels compile per signature (`signature={0: "*fp32", ...}`) plus
`tl.constexpr` values; dtype genericity is achieved by writing the kernel
against whatever pointer dtype arrives and specializing per launch. Dtypes can
also be passed as `tl.constexpr` arguments, with known foot-guns in how
constexpr literals get typed
([triton#6251](https://github.com/triton-lang/triton/issues/6251)).

- [Triton semantics](https://triton-lang.org/main/python-api/triton-semantics.html) ·
  [PyTorch blog: Triton kernel compilation stages](https://pytorch.org/blog/triton-kernel-compilation-stages/)

### 2.5 DaCe — symbolic shapes, concrete dtypes, same endpoint

`@dace.program` annotations use `dace.float64[N, M]` with *symbolic shapes*
(`dace.symbol`); the frontend solves for symbol values from concrete argument
shapes at call time, and JIT mode takes argument types from the call. The
dtype itself is fixed per annotation — dtype genericity is obtained by
re-parsing/specializing per call. So DaCe demonstrates symbol-parametric
shapes but reaches the same monomorphization endpoint for dtypes. (Relevant
since DaCe is a GT4Py backend.)

- [Writing DaCe programs in Python](https://spcldace.readthedocs.io/en/stable/frontend/daceprograms.html)

### 2.6 torchtyping — the cautionary tale

`TensorType["batch", "channels", float]` is effectively deprecated; the author
explicitly recommends jaxtyping for PyTorch because torchtyping monkey-patched
typeguard and was invisible/hostile to static checkers. Lesson: a dtype
annotation scheme that static checkers cannot process at all eventually gets
abandoned — keep the annotation a real generic that mypy can see.

- [torchtyping README](https://github.com/patrick-kidger/torchtyping) ·
  [author's retrospective](https://kidger.site/thoughts/jaxtyping/)

## 3. Static typing of dtype-generic numpy code

`numpy.typing.NDArray[ScalarT]` is an alias for
`np.ndarray[tuple[Any, ...], np.dtype[ScalarT]]`, generic in the scalar type.
The documented numpy pattern for dtype-generic functions is exactly a bounded
TypeVar:

```python
ScalarT = TypeVar("ScalarT", bound=np.floating)
def f(a: NDArray[ScalarT]) -> NDArray[ScalarT]: ...
```

The older `NBitBase` precision-genericity mechanism is deprecated (since
numpy 2.3) in favor of `typing.overload` or scalar-bounded TypeVars. mypy
support has known holes — e.g. it fails to flag some incompatible-dtype
parameterizations ([mypy#17228](https://github.com/python/mypy/issues/17228))
— and numpy ships its own mypy plugin for the rest. Takeaway: numpy converged
on the same spelling proposed for GT4Py, validating
`Field[Dims[...], DTypeT]`; and full static enforcement should not be
promised — extending GT4Py's existing mypy plugin
(`src/gt4py/next/type_system/mypy_plugin.py`, which currently *blurs*
`float32`/`float64` → `float`) is follow-up work.

- [numpy.typing docs](https://numpy.org/doc/stable/reference/typing.html)

## 4. Staged / two-level typing in embedded DSLs (theory)

- **Lightweight Modular Staging** (Rompf & Odersky): the foundational pattern
  — use the *host* type system (`Rep[T]`) to distinguish binding times;
  generic DSL code lives in the host language, specialized code is generated
  at staging time.
  ([CACM 2012 paper](https://dl.acm.org/doi/10.1145/2184319.2184345),
  [Stanford lecture notes](https://web.stanford.edu/class/cs442/lectures_unrestricted/cs442-lms.pdf),
  [Building-Blocks for Performance Oriented DSLs](https://arxiv.org/pdf/1109.0778))
- **Staged Compilation with Two-Level Type Theory** (Kovács, ICFP 2022):
  formalizes the key point — when the *object* language is simply typed
  (every object-level type statically known, as in GTIR), **monomorphization
  is the natural elaboration**: metaprograms may be polymorphic, but each
  staged output is fully concrete. This is the theoretical justification for
  "generic at decoration time, monomorphic per call-time specialization".
  ([arXiv 2209.09729](https://arxiv.org/pdf/2209.09729))
- **TensorFlow Eager / `tf.function`**: the de-facto industry standard for
  call-time monomorphization in Python — a trace cache keyed on argument
  dtypes/shapes; the paper discusses exactly the trace-cache-keying
  correctness problem that GT4Py's `arg_specialization_key` addresses.
  ([arXiv 1903.01855](https://arxiv.org/abs/1903.01855))
- **BuildIt** (C++): two-stage execution combining partial evaluation,
  analysis and codegen via host-language overloading.
  ([paper](https://arxiv.org/pdf/2601.02653))
- MLIR Python frontends repeat the pattern — parse Python at decoration time,
  lower with concrete types at call/compile time:
  [PyDSL (MLIR open meeting slides)](https://mlir.llvm.org/OpenMeetings/2023-12-21-PyDSL.pdf),
  [nelli](https://arxiv.org/pdf/2307.16080).

## 5. GT4Py's own related issues

No existing GridTools/gt4py issue specifically requests dtype-generic field
operators, but several are adjacent groundwork or recorded pitfalls:

- [#1415 "Fix Field type annotations in gt4py.next"](https://github.com/GridTools/gt4py/issues/1415)
  — the move to syntactically valid `Field[Dims[A, B, C], DType]` (mypy-clean,
  PEP 646-ready). The dtype-TypeVar design rides on this annotation form.
- [#1416 "Cleanup dtype hierarchy in \_core.definitions"](https://github.com/GridTools/gt4py/issues/1416)
  — a dtype-group/bound mechanism (à la jaxtyping's `Float`) would build on
  this hierarchy; relevant for the future `bound=` extension.
- [#214](https://github.com/GridTools/gt4py/issues/214) — historical cartesian
  bug: `Field[float, IJK]` returned an *instance* from `__class_getitem__`,
  breaking `Optional[...]`, `get_type_hints` and every typing wrapper. The
  generic alias must be a real `__class_getitem__` product.
- [#565](https://github.com/GridTools/gt4py/issues/565) — `ClassVar[T]` is
  illegal per PEP 526 (typing-spec compliance issue in eve).
- [#968](https://github.com/GridTools/gt4py/issues/968) —
  `typing_extensions` behavior changes around `Any`/`TypeVar` broke eve's
  runtime type validation once before: pin/test against `typing_extensions`
  versions, and keep TypeVar introspection inside `eve.extended_typing`.

## 6. Summary of design takeaways

1. **Spelling**: a value-constrained (or, later, bounded) type parameter inside
   the real generic `Field` class — spelled with PEP 695 `[T: (...)]` syntax
   (preferred at the 3.12+ floor) or an equivalent module-level TypeVar — the
   convergent solution of numpy.typing, jaxtyping-with-TypeVar and PEP 695;
   statically visible to mypy, runtime-introspectable, and forward-compatible
   with PEP 696 defaults.
1. **Strategy**: monomorphization at call time, everywhere — Numba, Taichi,
   Triton, TF `tf.function`, DaCe, and two-level type theory all agree. For
   GT4Py: check the body once with the TypeVar held abstract; bind from
   concrete `FieldType.dtype`s at call time; key the compilation cache on the
   substitution; lower fully concrete programs. Value-constrained TypeVars
   make the variant set finite, enabling eager precompilation.
1. **Pitfalls adopted into the design**: exact-match-or-error binding (no
   silent promotion); never hide the dtype from static checkers; prefer
   constrained TypeVars over unconstrained template markers; include the full
   substitution in cache keys; treat runtime TypeVar handling as
   version-sensitive; don't expect full mypy enforcement without plugin work.
