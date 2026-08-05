---
title: Generic dimensions and statically typed staggering in gt4py.next
author: havogt
tags: [type-system, generics, dimensions, staggering, type-checking, mypy, frontend, foast, monomorphization, unstructured]
created: 2026-06-11
status: draft
---

> **TL;DR** Redesign `Dimension` so that concrete dimensions are *types*
> (`class I(gtx.Dimension)`), usable directly in static type checking without a
> mypy plugin. On top of that: statically typed **staggering**
> (`b: Field[Dims[Staggered[I]]] = a(I + 1/2)`) via a `Staggered[D]` type
> constructor, and **dimension variables** (`TypeVar`/`TypeVarTuple` over
> dimensions) so field operators can be generic in their dimensions.

> **Part I has been extracted** into [[shared/dimensions-as-types|Dimensions as
> types]]. §3 below is now a stub keeping only what Parts II and III refer back
> to; the requirements, rejected alternatives, migration plan and risks of the
> base live in the shared document. Parts II and III stay here.

> **Prototypes**: the self-contained static-expressibility prototype (mypy 1.19,
> no plugin) is vendored alongside this document in
> [`dimension-generic-fields/`](dimension-generic-fields/) (see §8.A). The
> type-system extension (`ts.DimensionVar`/`ts.DimsVar`, real `src/` changes plus
> unit tests, §8.B) cannot be vendored here — it lives on the gt4py branch
> [`claude/gt4py-generic-dimensions-csn6tx`](https://github.com/graitools/gt4py/tree/claude/gt4py-generic-dimensions-csn6tx).
>
> Related: builds on [[personal/havogt/dtype-generic-fields|Dtype-generic fields in gt4py.next]]
> (shared binding/monomorphization machinery, Part III) and is the companion of
> [[personal/havogt/dependent-local-dimensions|Dependent local dimensions and connectivity chains]]
> (unstructured local dimensions on top of this redesign).

---

# Investigation: generic dimensions and statically typed staggering in `gt4py.next`

- **Status**: investigation / pre-ADR design proposal
- **Scope**: extends the dtype-generics investigation
  ([[personal/havogt/dtype-generic-fields|dtype-generic fields]]) to **dimensions**:
  (I) redesigning `Dimension` so concrete dimensions are *types* usable in
  static type checking, (II) statically typed **staggering**
  (`b: Field[Dims[Staggered[I]]] = a(I + 1/2)`), and (III) dimension
  *variables* (dim-generic operators) in the DSL type system.
- **Prototypes**: the critical pieces are implemented and tested — see §8 for
  what exactly is proven by the vendored
  [`dimension-generic-fields/`](dimension-generic-fields/) prototype (static
  expressibility, mypy-verified) and the `ts.DimensionVar`/`ts.DimsVar`
  extension of the type system (unit-tested, inert, on the gt4py branch).
- **Outcome**: a staged design proposal; decisions should be recorded as ADRs
  once reviewed (suggested: `0027-Dimensions-As-Types.md`,
  `0028-Dimension-Generic-Operators.md`).

## 1. Goals

```python
import gt4py.next as gtx
from typing import TypeVar
from typing_extensions import TypeVarTuple, Unpack


class I(gtx.Dimension): ...  # (I) a dimension that *is* a type


class K(gtx.Dimension, kind=gtx.DimensionKind.VERTICAL): ...


a: gtx.Field[gtx.Dims[I], gtx.float64]
b: gtx.Field[gtx.Dims[gtx.Staggered[I]], gtx.float64] = a(I + 1 / 2)  # (II) staggering
c: gtx.Field[gtx.Dims[I], gtx.float64] = b(gtx.Staggered[I] + 1 / 2)

T = TypeVar("T", gtx.float32, gtx.float64)
Ds = TypeVarTuple("Ds")


@gtx.field_operator  # (III) dims- and dtype-generic
def double(f: gtx.Field[gtx.Dims[Unpack[Ds]], T]) -> gtx.Field[gtx.Dims[Unpack[Ds]], T]:
    return f + f
```

All three are meaningful both to mypy (no GT4Py plugin required) and to the
DSL frontend. The previous conclusion that staggering "is not expressible in
Python typing" is revisited and **refuted** — but only under the dimension
redesign of §3; with today's instance-dimensions it indeed is not.

## 2. Current state

- **A concrete dimension is an instance, not a type**:
  `I = Dimension("I")` (`src/gt4py/next/common.py:95`). `Field[Dims[I], T]`
  is therefore not a valid static annotation; the mypy plugin
  (`src/gt4py/next/type_system/mypy_plugin.py`) papers over this by
  substituting up to four placeholder classes `_DimA`–`_DimD` (then `_AnyDim`)
  for dimension instances. Consequences: at most 4 distinct dims are tracked
  per run *globally* (keyed by argument name, so same-named dims in different
  modules collide), TypeVars over dimensions are impossible, and every
  downstream user must install the plugin.
- **`Dims` is already variadic** (`class Dims(tuple[Unpack[ShapeTs]])`,
  `common.py:57`) and `Field` is already a generic protocol
  (`Field(Protocol[DimsT, core_defs.ScalarT])`, `common.py:765`) — the static
  side is prepared; only the dimension arguments are not types.
- **Shift typing is dimension substitution.** A field call `field(offset)` is
  checked by `function_signature_incompatibilities_field` and typed by
  `return_type_field` (`type_system/type_info.py`), which replaces
  `OffsetType.source` by `OffsetType.target` in the dims list. The type
  system therefore *already* treats shifts as a dims-level substitution; only
  the *static* (mypy) side cannot express it.
- **Staggering exists at runtime**:
  `CartesianConnectivity.for_relocation(old, new)` (`common.py:1322`)
  relabels a field from one dimension to another in embedded execution
  (`premap`), and `CartesianConnectivity` is already generic in
  `[DomainDimT, DimT]`. There is no DSL syntax and no static typing for it.
- **Dims genericity is faked where needed**: the scan-operator program-context
  signature uses `DeferredType(constraint=None)` for what is actually
  "field with caller-determined dims" (`ffront/type_info.py:378`, with TODO),
  and `_scan_param_promotion` fabricates `Dimension("...")` placeholders
  (`ffront/type_info.py:200`).
- **From the dtype branch** (prerequisite for Part III): `ts.TypeVarType`,
  `type_info.is_generic`, `bind_type_vars`/`substitute_type_vars`, the
  `foast_specialize` toolchain step, and the `CompiledProgramsPool`
  call-time monomorphization path.

## 3. Part I — dimensions as types

**Extracted** to [[shared/dimensions-as-types|Dimensions as types]], together
with its requirements, the rejected alternatives, the migration plan and the
mypy-plugin removal it enables. Repeated here only insofar as Parts II and III
below refer back to it:

```python
class DimensionBase(metaclass=DimensionMeta):  # root: anything usable in Dims[...]
    value: ClassVar[str]
    kind: ClassVar[DimensionKind] = DimensionKind.HORIZONTAL


class Dimension(DimensionBase): ...  # base of user-declared dimensions


class I(Dimension): ...
class K(Dimension, kind=DimensionKind.VERTICAL): ...
```

- The **class object is the value**, and `DimensionMeta` carries the value-level
  API (`__add__`/`__sub__`, `__call__` → `NamedIndex`, comparisons → `Domain`) —
  which is where the staggering overloads of §4.2 attach.
- The **two-level root/user split** is the extension point Parts II and III use:
  `Staggered[...]` (§4.2) subclasses `DimensionBase` but not `Dimension`, and
  every generic dimension TypeVar (§5.1) is bound to the root.
- Dimensions are **nominal**, which is what lets the overload *argument matching*
  of §4.1 discriminate positions at all.

## 4. Part II — statically typed staggering

### 4.1 Why it was "not expressible", and what changed

A shift `a(I + 1/2)` maps `Field[Dims[..., I, ...]]` to
`Field[Dims[..., Istag, ...]]` — a **type-level substitution inside a
variadic tuple**. Python typing has no type-level `Map`/`Replace` over a
`TypeVarTuple`, and a tuple type may contain at most one unpacked
`TypeVarTuple`, so no single signature can express "replace `I` wherever it
occurs". With instance-dimensions (everything is `Dimension`) there is also
no nominal distinction to dispatch on. Hence the earlier conclusion.

Two ingredients of the redesign make it expressible:

1. **Dimensions are nominal types** (Part I), so overload *argument matching*
   can discriminate positions: an overload requiring
   `Connectivity[NewD, D1]` simply does not match when the connectivity's
   codomain is not the field's second dimension.
2. **Substitution is spelled per (rank, position)**: one `__call__` overload
   per position in each supported rank. This is bounded codegen, not
   hand-written combinatorics: ranks ≤ N need N(N+1)/2 overloads (10 for
   N = 4, covering I/J/K + one local/extra dimension); unstructured remaps
   (one dim → two dims) add one overload family analogously.

```python
class Field(Protocol[DimsT, DT]):
    @overload
    def __call__(
        self: Field[Dims[D0, D1], DT], conn: Connectivity[NewD, D0]
    ) -> Field[Dims[NewD, D1], DT]: ...
    @overload
    def __call__(
        self: Field[Dims[D0, D1], DT], conn: Connectivity[NewD, D1]
    ) -> Field[Dims[D0, NewD], DT]: ...

    ...
```

This is independently valuable: it makes *every* shift (`a(Koff[1])`,
unstructured remaps) statically typed, not just staggering.

### 4.2 `Staggered[D]`: the dual grid as a type constructor

```python
class Staggered(Dimension, Generic[D]):
    base: ClassVar[type[Dimension]]
```

- `Staggered[I]` is **both** the static type and (via `__class_getitem__`
  materializing a cached singleton class) the runtime dimension object — the
  static and runtime views cannot diverge. A domain name is just an alias:
  `Ihalf: TypeAlias = Staggered[I]`.
- **Staggering is an involution**, encoded in overload pairs on
  `DimensionMeta.__add__` (the `Staggered[D]` overload *before* the generic
  `D` overload, so `Staggered[I] + 1/2` yields `Connectivity[I, Staggered[I]]`
  and `Staggered[Staggered[I]]` never arises; constructing it explicitly
  raises):

```python
@overload
def __add__(cls: type[Staggered[D]], offset: int) -> Connectivity[Staggered[D], Staggered[D]]: ...
@overload
def __add__(cls: type[D], offset: int) -> Connectivity[D, D]: ...
@overload
def __add__(cls: type[Staggered[D]], offset: float) -> Connectivity[D, Staggered[D]]: ...
@overload
def __add__(cls: type[D], offset: float) -> Connectivity[Staggered[D], D]: ...
```

- **Doubly staggered types are unrepresentable.** The hierarchy has two
  levels: a root `DimensionBase` (anything usable in `Dims[...]`; all generic
  dimension TypeVars are bound to it) and `Dimension(DimensionBase)`, the
  base of *user-declared* dimensions. `Staggered`'s parameter is bound to
  `Dimension`, so `Staggered[Staggered[I]]` is a **static** `[type-var]`
  error ("Type argument ... must be a subtype of Dimension"), not just a
  runtime `TypeError`. Three layers of defense, all prototype-verified:
  (1) the annotation cannot be written; (2) no API operation infers it —
  the involution overloads always reduce (`Staggered[I] + 1/2` is
  `Connectivity[I, Staggered[I]]`); (3) runtime materialization rejects it.
  Note what is *not* claimed: the type-level equation
  `Staggered[Staggered[I]] = I` is **not expressible** (Python typing has no
  type-level reduction) — `Field[Dims[I], DT]` and a hypothetical
  `Field[Dims[Staggered[Staggered[I]]], DT]` would be mutually incompatible
  nominal types. Making the type unrepresentable sidesteps the equation
  entirely: the only fixed points are `D` and `Staggered[D]`, matching the
  physics (there are exactly two grids per dimension).
- The notation is exactly the desired `a(I + 1/2)`: `1/2` is a `float`, and
  `int` vs `float` is statically distinguishable. **Caveat**: float *values*
  are not (no `Literal` for floats), so `I + 0.7` statically claims to be a
  staggering shift and is only rejected at runtime (`ValueError`). If this is
  considered too weak, a dedicated `HALF` singleton type
  (`I + HALF`, `I - HALF + 1`) restores full static soundness at the cost of
  the literal notation; both can coexist.
- `Connectivity[NewD, OldD]` mirrors the existing
  `CartesianConnectivity[DomainDimT, DimT]` (domain dim = new, codomain =
  old); a staggering shift is `for_relocation` + translation, carried by one
  fractional `offset` value (fractional part = grid change, integral part =
  translation).
- **Dual-generic operators.** Operators like the C-grid average are generic
  in the *dual*, not in `Staggered[X]`: for `X = I` the result lives on
  `Staggered[I]`, for `X = Staggered[I]` on `I`. There is no type-level
  `Dual[X]` in Python typing, but the same overload pair used for `__add__`
  expresses it — one implementation body, two signature lines, generic at
  every call site (prototype-verified, including `avg(avg(a))` round trips):

  ```python
  @overload
  def avg(f: Field[Dims[Staggered[D]], float]) -> Field[Dims[D], float]: ...
  @overload
  def avg(f: Field[Dims[D], float]) -> Field[Dims[Staggered[D]], float]: ...
  def avg(f):
      (dim,) = f.dims
      return f(dim - 1 / 2) + f(dim + 1 / 2)
  ```

  The overload pair does **not** have to be user-written, though: a `Dual[X]`
  marker type plus a decorator whose argument type *recognizes* dual-generic
  signatures and whose return type is a library-provided protocol carrying
  the overload pair gives the same precision from one natural signature
  (also prototype-verified, including rejection of non-dual signatures at
  the decorator):

  ```python
  @dual_operator                      # in gt4py: folded into @field_operator
  def avg(f: Field[Dims[X], float]) -> Field[Dims[Dual[X]], float]:
      ...
  ```

  `Dual[X]` works in *parameter* positions too, e.g. a weight already living
  on the target grid (prototype-verified, including rejection of a weight on
  the wrong grid at the call site):

  ```python
  @dual_operator
  def weighted_avg(
      a: Field[Dims[X], float], weight: Field[Dims[Dual[X]], float]
  ) -> Field[Dims[Dual[X]], float]:
      ...
  ```

  The library writes one resolving protocol per supported signature *shape*
  (unary `X -> Dual[X]`, binary `(X, Dual[X]) -> Dual[X]`, ...), and the
  decorator is overloaded across the shapes — the same bounded-codegen
  philosophy as the rank/position overloads. In gt4py proper this typing
  belongs on `@field_operator` itself, so DSL users add nothing.

  Note the asymmetry with the DSL: the *internal* type system is ours, so
  there `Dual` can be a first-class type operator with the reduction rule
  `Dual[Dual[X]] = X` (or equivalently: applied eagerly during binding,
  since monomorphization makes `X` concrete before any result type is
  needed). The same `Dual[X]` annotation thus serves both layers: mypy
  resolves it through the decorator overloads, the frontend through
  `dual()` at specialization time — no per-direction duplication anywhere.

### 4.3 Semantics (value level)

Convention: staggered point `i` of `Staggered[I]` sits at position `i + 1/2`
of `I`. `b = a(conn)` means `b[p] = a[p + conn.offset]` in *position* space;
in index space this is a shift by `ceil(offset)` reading from an unstaggered
and `floor(offset)` reading from a staggered dimension. Useful identities
(all covered by runtime tests in the prototype):

- `a(I + 1/2)(Staggered[I] + 1/2) == a(I + 1)` — two half shifts = one full.
- `a(I + 1/2)(Staggered[I] - 1/2) == a` — round trip.
- `a(I + 1/2) - a(I - 1/2)` — C-grid finite difference, lives on
  `Staggered[I]`.

`a(I + 1/2)` is *pure relabeling + translation* (no interpolation);
averaging is written explicitly, e.g.
`0.5 * (u(I + 1/2) + u(I - 1/2))`. Domain handling on bounded (non-periodic)
domains follows the same position arithmetic (result range =
`{i : i + offset ∈ range(a)}`, i.e. half-open ranges shrink/shift by
ceil/floor); the prototype sidesteps this with periodic `np.roll`, the real
implementation reuses the existing `premap`/domain machinery which already
handles translation.

### 4.4 DSL integration (design, not prototyped)

Staggering needs **no genericity machinery** in the DSL type system — once
`Staggered[I]` materializes as an ordinary concrete dimension, the existing
concrete typing handles it:

- **FOAST**: `visit_BinOp` on `dimension ± literal` (today
  `type_deduction.py:604`, integers only) gains the fractional case,
  producing `ts.OffsetType(source=I, target=(Staggered[I],))` resp.
  `(source=Staggered[I], target=(I,))`. `return_type_field` already performs
  the dims substitution for arbitrary source/target. The lowering of the
  offset value keeps the integral part as today's cartesian offset; the grid
  change is encoded in the (auto-registered) relocation offset provider, so
  GTIR and the backends only ever see ordinary dimensions (e.g. `I½`) and
  integer shifts — **backends are unaffected**.
- **Embedded**: `I + 1/2` constructs the relocation+translation
  `CartesianConnectivity`, which `premap` supports today.
- The `Field`/`Connectivity` overloads of §4.1/4.2 live in `common.py` (or a
  generated `.pyi`) and replace the current untyped
  `__call__(...) -> Field` / plugin-blurred annotations.

### 4.5 Evidence: staggering *without* Part III is a regression

§4.4 is right that staggering needs no genericity to be *typed*. But typing it is
not the same as using it, and measuring a real kernel shows staggering alone makes
the code worse, not better.

Measured 2026-08-05 against gt4py v1.2.0, using the staggered-dimension feature as
merged (`feat[next]: staggered fields`, #2667) — so this is the shipped semantics,
not the `Staggered[D]` spelling proposed in §4.2.

The shallow water model (NCAR/SWM, an Arakawa C-grid kernel) applies four
operations — `avg_x`, `avg_y`, `delta_x`, `delta_y` — across fields at four grid
locations: cell centres `p`, `h`; x-faces `u`, `cu`; y-faces `v`, `cv`; corners `z`.
Today it needs **8 operators**, four of them distinguished only by a `_staggered`
name suffix and a flipped `±1` shift, with correctness resting on the author
calling the right one.

Rewriting with staggered dimensions requires **14**, because a field operator is
concrete in *every* dimension, not just the one being shifted. `avg_x` is applied to
`p` on `(I, J)` and to `cv` on `(I, JHalf)`; one operator cannot serve both:

```
avg_x(p  : (I, J)     ) -> accepted
avg_x(cv : (I, JHalf) ) -> ValueError: Incompatible 'Domain' in assignment
```

Enumerating every distinct signature the timestep actually uses:

| operation | required signatures | n |
| --- | --- | --- |
| `avg_x` | `(I,J)→(I½,J)`, `(I,J½)→(I½,J½)`, `(I½,J)→(I,J)`, `(I½,J½)→(I,J½)` | 4 |
| `avg_y` | `(I,J)→(I,J½)`, `(I½,J)→(I½,J½)`, `(I,J½)→(I,J)`, `(I½,J½)→(I½,J)` | 4 |
| `delta_x` | `(I,J)→(I½,J)`, `(I,J½)→(I½,J½)`, `(I½,J)→(I,J)` | 3 |
| `delta_y` | `(I,J)→(I,J½)`, `(I½,J)→(I½,J½)`, `(I,J½)→(I,J)` | 3 |

With the dimension variables of §5.1 the same four operations are **4** operators,
each generic in the dimensions it does not touch:

```python
def avg_x(f: Field[Dims[I, Unpack[Ds]], T]) -> Field[Dims[Staggered[I], Unpack[Ds]], T]: ...
```

So the progression is **4 generic / 8 by convention today / 14 with staggering
alone**. The type-safety argument also weakens at 14: fourteen near-identical
three-line bodies differing only in dimension annotations invite exactly the
copy-paste error the types were meant to catch.

§9 already has this ordering right: **D4** (staggering, value path) sits after
**D3** (dim-generic operators), so a user following the plan never meets staggering
without `Ds`. The measurement above is what that ordering is protecting against, and
it argues the two must not be decoupled — D4 pulled ahead of D3 is a net loss on
real kernels, not merely an incomplete feature.

Worth noting that gt4py **has already shipped the value path** (#2667) while D3 does
not exist. The staggered dimensions available today are therefore in exactly the
premature state this section measures: usable, correct, and — on a C-grid kernel —
worse than the `_staggered`-suffix convention they replace. That is an argument for
prioritising `Ds`, not for withdrawing staggering.

### 5.1 Spelling

```python
D = TypeVar("D", bound=Dimension)  # one unknown dimension
Ds = TypeVarTuple("Ds")  # an unknown *list* of dimensions


def op(f: Field[Dims[Unpack[Ds]], T]) -> Field[Dims[Unpack[Ds]], T]: ...
def col(f: Field[Dims[D, K], T]) -> Field[Dims[D, K], T]: ...
```

Native generics again (mypy-visible, PEP 646; `Unpack[Ds]` spelled `*Ds` on
3.11+). Note an asymmetry with dtype TypeVars forced by today's
instance-dimensions: **value-constrained dimension TypeVars
(`TypeVar("D", I, J)`) are impossible** because TypeVar constraints must be
types — they become expressible exactly when Part I lands. Until then,
`bound=Dimension` (open constraint set) is the only form, which inverts the
dtype-design situation (D2 there: finite constraints only).

### 5.2 Type-system representation (prototyped on this branch)

`ts.DimensionVar` (single) and `ts.DimsVar` (variadic) are
**`common.Dimension` subclasses, not `TypeSpec`s** —
deliberately asymmetric to `ts.TypeVarType`:

- They slot into `FieldType.dims` unchanged, so the large body of code that
  merely *carries* dimensions keeps working; only code requiring concreteness
  distinguishes them (`isinstance`), and `type_info.is_generic` reports
  fields containing them as generic.
- This *formalizes* the existing scan hack: `_scan_param_promotion`'s
  fabricated `Dimension("...")` and the `DeferredType(constraint=None)`
  program-context signature become `Field[Dims[DimsVar("Ds")], dtype]` — the
  type finally says what the hack meant ("field over caller-determined dims,
  with this dtype"), and the dims-genericity TODOs in `ffront/type_info.py`
  get their real fix.
- `FieldType._dims_validator` skips the canonical-ordering check while
  variables are present (order is only defined after substitution; the
  validator re-fires on the substituted, concrete type).
- Identity is name-based per signature, like `TypeVarType`. Name capture
  between a `DimensionVar("D")` and a concrete `Dimension("D")` is possible
  in principle; parse-time rejection of same-named distinct TypeVars (already
  required by the dtype design) extends to dims, and `__str__` marks
  variables (`D!`, `*Ds!`) in diagnostics.

`type_translation.from_type_hint` translates `TypeVar(bound=Dimension)` →
`DimensionVar` and `Unpack[TypeVarTuple]` → `DimsVar` (at most one variadic
per dims list; anything else in a `Dims[...]` is rejected with a specific
error).

### 5.3 Binding and substitution (prototyped on this branch)

`type_info.bind_type_vars` / `substitute_type_vars` (from the dtype design,
D5) gain dims rules under a single binding environment
`name → ScalarType | Dimension | tuple[Dimension, ...]`:

- A `DimsVar` splits the parameter dims into `prefix, *Ds, suffix`; the
  argument's dims must cover prefix+suffix, the middle binds to `Ds`
  (possibly empty; scalar args bind it to `()`).
- `DimensionVar`s bind positionally; inconsistent bindings across parameters
  raise with the variable name (same wording policy as dtype).
- Structural mismatches (wrong rank) leave variables unbound — reported by
  the signature checks, mirroring the dtype design.
- Substitution expands `DimsVar` in place and re-validates ordering via the
  `FieldType` validator.

### 5.4 What is different from dtype generics

- **Open genericity**: `bound=Dimension` has no finite variant set, so eager
  `.compile()` of all variants is impossible — only call-time
  monomorphization (the `CompiledProgramsPool` path and `foast_specialize`
  step work unchanged: the pool already keys on full argument types,
  including dims).
- **Deduction rules**: shifting a field whose shifted dimension is a
  variable, `promote_dims` across distinct variables, `broadcast`, `domain=`
  interaction — v0 policy mirrors dtype D3: operations whose dims result is
  not derivable symbolically are decoration-time errors naming the variable
  (`f(Ioff[1])` requires `I` to appear concretely in the signature; `f + g`
  requires identical generic dims expressions). Everything dims-preserving
  (arithmetic with same-var operands, math builtins, `where` with matching
  vars, scalar broadcast) works symbolically.
- **Staggering and dim vars compose for free**: `Staggered[D]` with `D` a
  variable is just a dims entry `Staggered[DimensionVar]` after Part I+II;
  binding resolves `D` and materializes the concrete staggered dimension —
  no new machinery (the prototype's `to_staggered` shows the static side
  already works).

## 6. Related work: coordax

[coordax](https://github.com/neuralgcm/coordax) ("Coordinate Axes for JAX",
by the NeuralGCM / JAX-CFD / Xarray authors) solves an overlapping problem —
dimension-labeled fields for simulation codes — with an instructive mix of
shared spirit and opposite mechanism.

**The coordax model.** `Field` = array + `dims: tuple[str | None, ...]` +
`axes: dict[str, Coordinate]`. A dimension's *identity* is a **string**; an
axis may also be `None` ("positional"). `Coordinate` objects are frozen
dataclasses registered as *static* JAX pytree nodes carrying `dims`, `shape`
and optional tick fields (`SizedAxis` checks only size, `LabeledAxis` checks
tick equality, custom subclasses model e.g. spectral grids). Operations
require **exact coordinate alignment** — deliberately no Xarray-style
auto-alignment, which the authors call out as wrong for simulation code.
Dimension-generic code is written *dynamically*: `untag` a named axis into a
positional one, apply a plain-array function via `cmap` (which `vmap`s over
all remaining named axes), `tag` the result.

**Similar in spirit.**

- *Strictness*: coordax's exact-alignment / no-auto-align stance is the same
  philosophy as our strict no-promotion policy (dtype D3, dims §5.4) — make
  domain mismatches errors, not conversions.
- *Checks before run time*: their static-pytree `Coordinate`s give
  "compile-time" (JAX-trace-time) consistency checks; our decoration-time
  FOAST checking plays the same role one stage earlier, and the dims-as-types
  layer adds a stage coordax does not have at all (see below).
- *Identity/metadata split*: coordax separates cheap per-axis identity
  (string) from rich coordinate metadata (`Coordinate` with ticks). GT4Py has
  the same split — `Dimension` (identity + kind) vs `Domain`/`NamedRange` and
  offset providers (the actual coordinates) — and this design keeps it:
  Part I promotes only the *identity* half to the type level; ranges/ticks
  remain runtime values. A coordax `Coordinate` maps to (dimension, range)
  pairs, not to `Dimension` itself.
- *Pointwise-over-named-axes semantics*: `cmap`'s "apply over positional,
  vectorize over named" is essentially the field-view semantics gt4py field
  operators already have (pointwise over the field's dims with implicit
  broadcast).

**Opposite mechanism, and what that buys/costs.**

- Identity as *strings and values* (coordax) vs *nominal types* (this
  design). Coordax has **no static-typing story**: `cx.Field` is opaque to
  mypy — wrong-dimension code trace-fails at run/trace time. That is the
  torchtyping/jaxtyping lesson from the dtype investigation again, and the
  gap Part I exists to close. Conversely, coordax needs no rank-bounded
  overloads and no metaclass machinery — its flexibility (axes appearing/
  disappearing under `vmap`, `None` axes) comes precisely from *not* putting
  dims in types.
- Dimension genericity: dynamic erasure (`untag`/`cmap`, rank polymorphism
  via `vmap`) vs typed variables + monomorphization (Part III). The coordax
  pattern needs no signature-level machinery but moves all errors to trace
  time and erases dim safety *inside* the generic function body; our DSL
  keeps the body symbolically checked (§5.4).
- Staggering: not first-class in coordax; the idiom (explicit in its
  sibling JAX-CFD, whose `GridArray` carries a value-level
  `offset=(0.5, ...)`) is coordinate *metadata* — a staggered field is a
  `LabeledAxis`/custom `Coordinate` with shifted ticks, and mixing grids
  fails the runtime tick-equality check. Part II promotes exactly this
  metadata bit to a type (`Staggered[I]`, `Dual[X]`), making the same
  mistake a static + decoration-time error and giving shifts a *type-level*
  effect (`a(I + 1/2)` changing the dims type has no coordax analogue).

**How coordax could fit.** Not as a dependency — gt4py's `Field`/`Domain`
already cover its runtime role — but three concrete touchpoints:

1. *Interop*: a lossless `gt4py.Field ↔ cx.Field` mapping is cheap and
   useful for the JAX-backed embedded path and for analysis pipelines
   (dimension class → dim name; `NamedRange`/staggering → a `Coordinate`
   subclass whose ticks encode the half-offset; `kind` → coordinate
   metadata). Mirrors coordax's own "lossless to/from Xarray" goal.
2. *Validation of the identity/metadata split* (above): coordax arrived at
   the same factoring independently — supporting evidence that promoting
   dimension *identity* (not coordinates) to types is the right cut.
3. *A `cmap`-shaped escape hatch* is what §5.4's v0 restrictions would send
   users to when an operation's dims-effect is not expressible symbolically:
   the embedded API could expose an explicit untag/apply/tag idiom for
   power users rather than silently widening the typed surface.

## 7. Python version considerations

GT4Py currently supports 3.10–3.14; the design is *not* required to hold back
to 3.10 if a newer version buys expressiveness. Audit result: **every
load-bearing feature works on 3.10** (nominal classes, metaclass operators,
overloads, `TypeVarTuple`/`Unpack` via `typing_extensions`); newer versions
add ergonomics, not power:

- **3.11 (PEP 646 native)**: `Field[Dims[*Ds], T]` spelling instead of
  `Unpack[Ds]`; runtime `typing.TypeVarTuple` (also removes the 3.10
  `typing_extensions.Unpack`-passes-`isinstance(…, TypeVar)` wart found while
  prototyping).
- **3.12 (PEP 695)**: `class Field[*Ds, DT]`, `def op[D: Dimension](...)` —
  much nicer generic-operator spelling, same semantics. Caveat: PEP 695
  `type Ihalf = Staggered[I]` aliases are `TypeAliasType` objects, **not**
  usable as runtime values (`Ihalf + 1/2` would not dispatch through
  `DimensionMeta`); the plain assignment alias `Ihalf = Staggered[I]`
  (which our `__class_getitem__` materializes to a real class) remains the
  recommended spelling on all versions.
- **3.13 (PEP 696 TypeVar defaults)**: `Field[Dims[I, J]]` defaulting the
  dtype parameter (already noted in the dtype investigation); PEP 742
  `TypeIs` gives precise both-branch narrowing for `is_staggered`-style
  guards (we currently avoid narrowing deliberately).
- **3.14 (PEP 649/749 deferred annotations + `annotationlib`)**: removes the
  `from __future__ import annotations` requirement and makes forward
  references in dimension/field declarations robust; `annotationlib`'s
  `FORWARDREF` format is a cleaner basis for `from_type_hint`'s runtime
  introspection. No change in type-system expressiveness.
- **3.15 and beyond (future directions, not yet usable)**:
  - **PEP 747 `TypeForm`** (accepted for typing_extensions/3.15-era
    checkers): lets `from_type_hint`-style APIs be *typed* (`TypeForm[T]`),
    improving our own tooling, not the DSL surface.
  - **PEP 718 subscriptable functions** (draft): would allow *explicit*
    specialization of generic operators (`diffusion[float32]`,
    `op[Dims[I, J]]`) — a natural fit for the eager-`.compile()` API and
    worth tracking for the ADR's forward-compatibility section.
  - **Higher-kinded types / type-level functions** (no PEP exists, and none
    is on the typing council's roadmap): this is the feature that would make
    "replace `I` by `Staggered[I]` anywhere in a variadic `Dims`" directly
    expressible and the rank-bounded overload generation unnecessary. Until
    then the overload encoding of §4.1 is the only game in town — which is
    exactly why it is designed as bounded *codegen* rather than a hand-kept
    API surface.

## 8. What the prototypes prove

**A. Static expressibility** — [`dimension-generic-fields/`](dimension-generic-fields/)
(self-contained, vendored next to this document; run with
`pytest content/personal/havogt/dimension-generic-fields/` and
`mypy --config-file content/personal/havogt/dimension-generic-fields/mypy.ini ...`;
mypy 1.19 without any plugin):

- `typed_dimensions.py`: `DimensionMeta`/`Dimension`/`Staggered`/
  `Connectivity`/`Field` exactly as in §3–4 (~150 lines of typing surface).
- `static_checks.py` (mypy-clean, `assert_type`-pinned): the §1 staggering
  example verbatim; involution typing; rank-1/2/3 positional substitution
  incl. chaining; C-grid gradient; rank-/dim-/dtype-generic operators via
  `TypeVarTuple`; a dimension-generic staggering operator.
- `static_errors.py` (every marked line must error, no other line may):
  staggered↔unstaggered assignment mismatches both ways, shift along a
  missing dimension, mixing dual grids in arithmetic, wrong dimension
  (kind) as argument, re-staggering without going through the dual grid,
  and unrepresentability of `Staggered[Staggered[I]]` (§4.2).
- `test_typed_dimensions.py`: runtime semantics (involution, round trip,
  two-halves-equal-one-full, C-grid gradient, name-based legacy interop,
  rejection of `I + 0.7` and `Staggered[Staggered[I]]`) plus the two
  mypy-as-oracle tests above.

**B. Type-system extension** (real `src/` changes, inert until the frontend
produces them, mypy-clean, all existing tests pass):

- `ts.DimensionVar`/`ts.DimsVar` + validator relaxation
  (`type_specifications.py`), `is_generic`/`bind_type_vars`/
  `substitute_type_vars` extensions (`type_info.py`), annotation translation
  (`type_translation.py`).
- Unit tests: `tests/next_tests/unit_tests/type_system_tests/test_dimension_vars.py`
  (translation incl. error cases, genericity predicate, variadic/positional
  binding, consistency errors, scalar args, end-to-end `FunctionType`
  bind+substitute).

## 9. Staged plan

Stages 0–2 of the dtype plan are prerequisites for the *frontend* stages here
(the binding utilities are shared); Part I/II stages are independent of dtype.

1. **Stage D0 — dimensions as types in `common`** (Part I): specified in
   [[shared/dimensions-as-types|Dimensions as types]]; prerequisite for every
   stage below.
2. **Stage D1 — typed shifts & staggering, static side** (Part II):
   `Staggered`, typed `Connectivity`, generated `Field.__call__` overloads;
   remove the dims hack from the mypy plugin for class-style dims.
3. **Stage D2 — dim variables in the type system** (Part III, §5.2–5.3):
   done on this branch; landable behind the same "inert until frontend"
   property as dtype Stage 0.
4. **Stage D3 — dim-generic operators, frontend**: `from_type_hint` is done;
   extend FOAST deduction (symbolic dims rules incl. the v0 error policy),
   reuse `foast_specialize`; embedded + direct-call path first (pool already
   keys on dims), program-call path second — both mirror dtype Stages 1–2.
   Replace the scan `DeferredType`/`Dimension("...")` hacks.
5. **Stage D4 — staggering, value path** (Part II §4.4): FOAST
   `dim ± fraction` typing, relocation offset providers, embedded `premap`
   wiring, domain conventions; ADR for the position convention.

## 10. Risks and open questions

1. **The metaclass-overload mypy quirk** (see
   [[shared/dimensions-as-types|Dimensions as types]]): call-site behavior is
   correct but the def-site suppression could break on a mypy upgrade;
   pinned by tests, with a notation-only fallback. **pyright is untested** —
   must be checked before committing to the `I + 1/2` notation (the
   fallback functions are checker-agnostic).
2. **Float-literal staggering offsets** are value-checked only at runtime
   (`I + 0.7`); decide literal notation vs. `HALF` token (§4.2) in the ADR.
3. **Overload-set size vs. checker performance**: N(N+1)/2 overloads per
   substitution operation is small (≤ 15), but `Field` has many operators;
   measure mypy runtime on a large downstream consumer before generalizing.
4. **Migration surface of Part I** is the largest cost item overall; it is
   tracked in [[shared/dimensions-as-types|Dimensions as types]], not here.
5. **Canonical dims ordering with variables**: `Dims[D, K]` assumes the
   binding of `D` sorts before `K`; substitution re-validates, so a "wrongly
   ordered" binding today raises a validator error — decide whether to
   re-sort instead (probably not: order is semantic) or improve the message.
6. **Name capture** between same-named variables and concrete dims (§5.2):
   parse-time rejection policy must be specified in the ADR (including
   collisions between dtype and dim variables sharing a name).
7. **Staggered domains on bounded grids** (§4.3): the ceil/floor convention
   is consistent but interacts with `domain=` in programs and with
   `concat_where`; needs its own design slice in Stage D4.
8. **Scan-operator unification**: replacing the scan hack with
   `DimsVar` changes user-visible error messages and the `__gt_type__` of
   scan operators in program context; coordinate with the dtype Stage 1
   guard ("generic scan not yet supported") so the two features do not
   trip over each other.
