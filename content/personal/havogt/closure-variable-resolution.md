---
title: Closure variable resolution in gt4py.next
author: havogt
tags: [frontend, foast, past, closure-variables, name-resolution, constants, builtins, aliasing, gtir, lowering, contextvar, ambient, fields, mesh, static-args]
created: 2026-06-12
status: draft
---

> **TL;DR** Make closure-variable references in the `gt4py.next` frontend
> resolve **by value, not by source-level name**, so that module-prefixed
> builtins (`gtx.where`), aliased operators (`helper as h`), module/`Final`
> scalar constants (`np.pi`), and re-exports all work — by resolving every
> reference to its Python value right after parsing and canonicalizing the AST.
> Errors move from late, internal stages to decoration time.

> **Prototype**: a *partial* prototype of strategy **A** below (the
> resolve-early / canonicalize approach) lives on the gt4py branch
> [`claude/elegant-cannon-13ew06`](https://github.com/graitools/gt4py/tree/claude/elegant-cannon-13ew06),
> together with a proposed ADR (`docs/development/ADRs/next/0023-Closure_Variable_Resolution.md`)
> that catalogs the limitations and compares strategies in full. This document
> distills that ADR into a knowledge-base idea; consult the branch for the
> detailed case table and the prototype code.

---

## Problem / motivation

Field operators and programs are Python functions that may refer to names which
are neither parameters nor locals — *closure variables*. The frontend captures
them with `inspect.getclosurevars()` into a flat `dict[str, Any]` keyed by the
**source-level name** (`ffront.source_utils.get_closure_vars_from_function`).

The crucial — and fragile — property of the current design is that **identity in
all later stages is the source-level name**:

- Builtin calls are recognized by comparing `foast.Name.id` against
  `fbuiltins.*_BUILTIN_NAMES` (in `foast_passes.type_deduction` and
  `foast_to_gtir`).
- Calls to other operators lower to `SymRef(<name used at the call site>)`,
  while the callee registers its IR function definition under its original `def`
  name (`GTCallable.__gt_gtir__()`), linked at program level in `past_to_itir`.
- Offsets lower to `OffsetLiteral(<call-site name>)` and are looked up by that
  name in the `offset_provider` at run time.
- All closure variables of all transitively referenced operators are merged into
  one flat namespace (`transform_utils._get_closure_vars_recursively`), which
  raises if the same name binds to different values anywhere in the call tree.

Any reference whose source-level name differs from the "canonical" name
therefore breaks — each case in a different stage, with a different (often
internal) error. Only `FrozenNamespace`/`enum` attributes were resolved by value
today (the former `ClosureVarFolding` pass).

A second, orthogonal problem is *when* values are captured: at decoration time.
Embedded execution instead runs the original Python function and sees the *live*
values, so closing over a mutable binding has **diverging semantics** between
embedded and compiled execution (or simply fails in compiled mode for scalars).

### Symptoms (verified on `main`, 2026-06)

References that **should** be part of the model but fail today, with how late /
how obscure the error is:

| Case | Behavior on `main` | Stage |
| --- | --- | --- |
| Module-prefixed builtin `gtx.where(...)` | `DSLError: Functions can only be called directly.` | FOAST type deduction |
| Aliased builtin `my_where = where` | `AssertionError: ... not fully typed.` (internal) | FOAST type deduction |
| Module-prefixed operator `helpers.helper(a)` | `DSLError: Functions can only be called directly.` | FOAST type deduction |
| Aliased operator `helper as h2; h2(a)` | `EveValueError: Symbols {SymbolRef('h2')} not found.` (internal) | GTIR validation |
| Same operator referenced under two names | `EveValueError: Multiple definitions of symbol 'helper'` (internal) | GTIR validation |
| Scalar from a module attribute `np.pi` | `AssertionError: Unreachable` (internal) | FOAST→GTIR lowering |
| Module-level scalar marked `typing.Final` | `EveValueError: Symbols {SymbolRef('CONST')} not found.` (internal) | GTIR validation |

By contrast, `Dimension` via a module prefix and `FrozenNamespace`/`enum`
attribute constants already work. The full case table (incl. the
should-be-*rejected* cases that today produce internal errors instead of clear
decoration-time ones — mutable globals, callables in containers, mutated
captured bindings) is on the branch.

## Proposal

### The model: what may be captured

A closure reference must be a **constant reference expression** — a name, or a
chain of attribute accesses rooted at a name — that resolves at decoration time
to a value of a supported kind:

- a GT4Py builtin (function, type constructor),
- a `GTCallable` (field operator, scan operator, …),
- a `Dimension` or `FieldOffset`,
- a namespace used as a prefix (module, `FrozenNamespace`, `enum` class),
- a scalar **constant**, where "constant" means the binding is immutable or
  promised to be: a `FrozenNamespace`/`enum` attribute, a module attribute, or a
  module-level global annotated `typing.Final`.

**Identity is by value, not by name**: aliases, re-exports and module-prefixed
references of the same value are interchangeable. Mutable bindings (plain
globals, nonlocals) and mutable containers are out of the model; data values
(fields, arrays) are not capturable (see the stretch goal).

### Strategy A — resolve names to values at decoration, canonicalize (chosen)

Keep the existing capture-at-decoration-time pipeline, but immediately after
parsing resolve every closure reference (including attribute chains) to its
Python *value* and canonicalize the AST:

- constants → folded into `Constant` nodes,
- builtins → rewritten to the canonical builtin name,
- attribute-derived references to operators/dimensions/offsets → rewritten to a
  synthesized, value-derived name registered as an extra closure variable,
- direct references (incl. aliases) keep their source name; the program lowering
  registers IR function definitions under the **reference name** instead of the
  definition name — aliases then work without rewriting, and user-facing names
  stay in IR and error messages.

**Pros**: small, local change (two frontend passes + the function registration
in `past_to_itir`); fixes all catalogued cases; errors move to decoration time
with source locations; no change to stages, caching, embedded execution, or
backends; IR stays readable.

**Cons**: keeps snapshot-at-decoration semantics (mitigated by the `Final`
rule); constants are baked into the AST; the flat merged namespace remains, so
two *directly* referenced operators with the same source name but different
values still collide (aliasing is the documented workaround); the AST is no
longer purely syntactic.

### Why A now, B later

**Strategy A now, with [Strategy B](#alternatives-considered) as the evolution
path.** A resolves every catalogued limitation except the no-alias same-name
operator collision, is reviewable incrementally, and does not paint us into a
corner: the canonicalization pass is exactly where B's opaque reference ids
would later be introduced, and the key-based function registration in
`past_to_itir` is the seed of B's link step.

## Alternatives considered

- **Strategy B — late binding via an explicit closure environment ("linker").**
  References in FOAST/PAST point to opaque, unique reference ids; a
  `ClosureEnvironment` maps ids to values; binding happens at program-assembly
  time (function definitions get final names, mangled only on real collision;
  `SymRef`s remapped; constants substituted at lowering, not into the AST).
  *Pros*: fully value-based identity (even the same-source-name collision
  disappears); AST stays syntactic; enables re-binding (`with_closure(...)`),
  staleness checks, and is the natural foundation for capturing fields. *Cons*: a
  cross-cutting refactor of every stage that assumes the flat name-keyed dict;
  higher migration cost; harder to land incrementally. **The intended evolution
  of A.**
- **Strategy C — minimal name-mangling patches** (dedup identical defs, rename
  `GTCallable` closure vars when unambiguous, fold module attributes). Smallest
  diff, but does not fix the underlying name-identity problem and piles up
  special cases that A subsumes.
- **Strategy D — trace-based capture (à la JAX).** Closures resolve for free via
  ordinary Python semantics, but it is a different frontend architecture
  deliberately *not* chosen for GT4Py (ADR 0001 — Field View Frontend Design):
  tracing loses source locations and type annotations and restricts control
  flow. Out of scope.

## Open questions / conflicts

- **Same-name collision without aliases**: two *directly* referenced same-named
  operators still raise under A; the message now suggests aliasing as the
  workaround. Fully solving it requires B's link step.
- **`typing.Final` for nonlocals** cannot be verified (function-local
  annotations are not retained at runtime), so the factory-function pattern
  needs a `FrozenNamespace` or a future explicit opt-in (e.g.
  `gtx.constant(...)`).
- **Local shadowing** of a canonical builtin name (a local `where` together with
  `gtx.where`) is detected only when the local is visible in the symbol table.
- **pyright / non-mypy checkers** and serialization concerns are unaffected by A
  (it touches only frontend passes), but B would intersect with stage
  fingerprinting and `GTCallable.__gt_closure_vars__`.
- No direct conflict with the other ideas here, but it shares the frontend
  binding/lowering surface with
  [[personal/havogt/dtype-generic-fields|Dtype-generic fields in gt4py.next]] and
  [[personal/havogt/dimension-generic-fields|Generic dimensions and statically typed staggering]]
  (the `foast_specialize`/monomorphization machinery and the closure-var
  fingerprint that compiled-program caching keys on).

## Appendices

### Stretch goal: closing over read-only fields

Capturing field *data* differs in kind: the value cannot be folded into the IR,
it must reach the backend at run time. A clean design answers three questions —
**marking** (explicit, e.g. `typing.Final[gtx.Field[...]]` or a
`gtx.constant_field(field)` wrapper that can freeze/copy-on-capture and expose
`__gt_type__`); **mechanism** (a captured read-only field becomes a *hidden
parameter* of the operator via a `GTCallable.__gt_implicit_args__()`, reusing
the entire existing argument machinery; embedded execution needs no change);
and **binding time / caching** (bind the *binding* not the buffer, so the
compiled cache key includes only the field's type/domain descriptor — zero
recompilation cost). A separate opt-in (`gtx.constant_field(field, freeze=True)`)
could bake the buffer in as a true constant, layered on `StaticArg`-style
descriptors (ADR 0021 — Argument Descriptors). This fits A (the canonicalization
pass introduces the hidden parameter) and becomes more natural under B.

### Stretch goal: ambient values bound at execution time

The stretch goal above binds at *decoration* time. The variant worth recording
binds later: a user declares a value once, globally, and any field operator
reaches it without it appearing in a signature — a `ContextVar` filled at
program-execution time (JIT time for compiled backends).

> **Prototyped**: gt4py branch
> [`ambient-offset-provider`](https://github.com/havogt/gt4py/pull/71) (fork PR).
> Everything below marked *measured* comes from there; everything marked *open*
> is still open.

The motivating case is mesh properties. They are scalars on a Cartesian grid
(`dx`, `dy`) and fields on an unstructured one (connectivities plus their
weights); both must today be carried explicitly — as extra operator parameters,
or as the `offset_provider` threaded down the call chain. Every operator between
the caller and the one that actually needs a weight has to name it.

#### Surface

One rule for everything ambient: **the declaration is the key**.

```python
V2E = gtx.FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
dx = gtx.Static[float]              # a value, folded into the generated code
nu = gtx.Extern[float]              # a value, passed at runtime


@gtx.field_operator
def delta_x(f: IJField) -> IJField:
    return (1.0 / dx) * (f(I + 1) - f)     # never a parameter


prog(f, out, bind={V2E: connectivity, dx: 0.5})   # or: with gtx.bind(dx, 0.5): ...
```

A `FieldOffset` is *already* a declaration — it names the offset and fixes its
source and target — so it binds exactly like a value, and a program called
without an `offset_provider` assembles one from the bound offsets. A container
may declare what it supplies, with the class attribute holding the declaration
and the instance attribute the value:

```python
class Mesh:
    V2E = V2E              # this mesh supplies the V2E connectivity
    dx = physics.dx


prog(f, out, bind=Mesh(...))
```

Binding by declaration *identity* rather than by attribute name is what lets a
container supply the very offset an operator refers to, rather than one that
merely shares its name — and it means a container attribute need not be named
after the declaration at all (*measured*: an attribute called `spacing` resolves
`physics.dx` correctly, across modules).

`bind=` is sugar over the `ContextVar`, scoped to one call, so the two spellings
compose rather than compete.

#### How it works

- **A declaration carries its type.** `Static[float]` implements `__gt_type__`,
  and `type_translation.from_value` already dispatches on that — so an operator
  referring to an ambient value type-checks when it is *defined*, with nothing
  bound. **No type-system change was needed.** An untyped placeholder fails with
  `DSLTypeError: Unexpected object ...`, and the failure is temporal, not
  spatial: it happens in the same file, so putting the declaration in another
  module changes nothing.
- **The reference becomes a synthesised program parameter**, added once in
  `func_to_past`. From there it travels the ordinary path: type checking,
  lowering, `static_params` and the compiled-program key all treat it as an
  argument, and only the value is supplied per call.
- **The two forms differ in one place only** — whether that parameter is listed
  as static. `Extern[T]` stays a runtime argument; `Static[T]` is a static
  argument, so the *existing* fold in `past_to_itir` bakes it in and the
  *existing* `StaticArg` key specialises on it. *Measured* on gtfn and dace with
  two values: `Static` compiles 2 variants, `Extern` 1, both correct.
- **Connectivities identify by content, not by `id`.** `gtx.freeze(conn)` caches
  a content hash once; `hash_offset_provider_items_by_id` prefers it. *Measured*:
  3 compiled variants drop to 2 when two structurally identical meshes stop
  being keyed apart by object identity — that function's own docstring warns it
  "could generate different hashes for two offset providers that are
  semantically equal".

Three things it turned out **not** to need, each of which was assumed at some
point in the design: a new FOAST specialization step (the existing static-argument
fold suffices); threading the parameter into operator signatures and call sites
(a free symbol in a lowered operator resolves against the *program's* parameters,
and both gtfn and dace codegen fine that way); and a per-operator inspection pass
(`transform_utils._get_closure_vars_recursively` already collects declarations
transitively through nested operators).

#### Transition: retiring the second binding rule

The prototype first grew *two* binding mechanisms with different identity rules:
connectivities were harvested from a bound object by attribute **name** (a
`Namespace`), values were keyed by declaration **identity**. Same surface, two
rules — and the name-based half needed a collision check, because two namespaces
could each offer an offset called `V2E`.

Unifying on declarations removes `Namespace`, the attribute harvesting and that
collision check outright, and it is worth doing in two steps:

1. **Move only the binding surface.** Offsets bind by declaration; the
   `offset_provider` is still assembled from the bound ones and passed exactly as
   today, so the frontend, the IR and the backends are untouched. *This step is
   implemented.*
2. **Then consider making connectivities ordinary ambient values**, i.e. an
   `Extern[Connectivity]` that becomes a synthesised parameter like any other,
   at which point `offset_provider` stops being a separate concept. This is not
   obviously right and should not be assumed: `offset_provider` carries things
   the parameter path does not model today (its *type* drives domain inference
   and connectivity typing, and `arguments.py` still notes the temporary pass
   needs the runtime object). Worth a separate look rather than a follow-through.

The staging matters because step 1 is a pure surface change with no risk to the
toolchain, while step 2 touches how connectivities are typed and inferred.

#### Binding time

"Static for the lifetime of the application" is the wrong unit; the useful one is
**static for a jitted program**. That makes the two forms a declaration-site
choice of what identifies a value in the jit key: `Static[T]` puts the *value*
there, `Extern[T]` only its type. Which also dissolves an apparent conflict with
[[personal/havogt/jax-connectivities/jax-connectivities|the JAX connectivities proposal]] — a
connectivity wants descriptor-identity (one program per mesh *class*), a scalar
wants value-identity, and nothing forces one policy on both.

#### Open

- **Canonical identifiers.** Synthesised parameters currently take the closure
  variable's *local* name, so two operators naming the same declaration
  differently, or two modules that both use `dx`, collide. Declarations should be
  named explicitly (`Static[float]("dx")`) with a FOAST rename pass; a generated
  counter will not do, since the name lands in the compiled signature and would
  shift with import order.
- **Ambient fields** (`mesh.edge_length`) are not implemented, but now look like
  an `Extern[Field[...]]` — a synthesised parameter with no folding, i.e. the
  read-only-field stretch goal above with a later binding time.
- **Immutability.** `freeze(readonly=True)` would make the cached content hash
  trustworthy, but the gtfn bindings are generated with mutable `ndarray`
  parameters and reject a read-only array outright, so it is off by default.
- **Debuggability**, unchanged: errors when nothing is bound must be good, and
  there should be a way to see what an operator depends on ambiently.

The ceiling this aims at is a mesh concept built on top, where connectivities
and weights are properties of an ambient mesh rather than arguments — see
[[personal/havogt/mesh-and-first-class-halos|A mesh concept with first-class halos]].
It would also be a plausible substrate for
[[personal/egparedes/discretization-independent-fd-syntax|A discretization-independent
surface syntax]], whose mesh-invariant surface needs weights and connectivity to
come from a declared mesh property rather than from operator arguments (that
proposal is a higher-level surface layered on gt4py's core concepts, not a
replacement for them).

### Source material

- Proposed ADR and **partial prototype of strategy A** on gt4py branch
  [`claude/elegant-cannon-13ew06`](https://github.com/graitools/gt4py/tree/claude/elegant-cannon-13ew06)
  (`docs/development/ADRs/next/0023-Closure_Variable_Resolution.md`). The
  prototype adds `foast_passes.closure_var_resolution.ClosureVarResolution`
  (replacing `ClosureVarFolding`), a PAST counterpart
  `past_passes.closure_var_canonicalization.ClosureVarCanonicalization`,
  reference-name function registration in `past_to_itir.past_to_gtir`, a
  data-typed-closure-var rejection in `foast_to_gtir`, and
  `source_utils.get_final_closure_var_names` for `typing.Final` detection.
- ADR 0001 — Field View Frontend Design (why not strategy D).
- ADR 0021 — Argument Descriptors (the `freeze=True` field-capture variant).
