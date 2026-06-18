---
title: The "open type slot" problem space in gt4py.next
author: havogt
tags: [type-system, generics, dtype, dimensions, deferred-type, type-variables, monomorphization, scan, foast, frontend]
created: 2026-06-17
---

> **TL;DR** `gt4py.next` has **two** open-type-slot placeholders today — the
> anonymous `DeferredType` ("some type, maybe constrained") and the named,
> value-constrained `TypeVarType` from the dtype-generics work (gt4py ADR 0023).
> Between them they wear at least **six semantic hats**, which are really points
> in a **three-axis space** (binding mode × sort × constraint). The fix is to
> split the placeholders along their main axis into three constructs that live in
> three different places — *absence* → `Optional[TypeSpec]`, a *universal
> parameter* → a kinded `TypeScheme`/`TypeParam`/`ParamRef`, a *value-dependent /
> computed return* → an explicit *type rule* on builtins. A tempting shortcut —
> *merging* the two placeholders into a single class (explored on a branch as a
> candidate "ADR 0024") — consolidates along the **wrong** axis and is evaluated
> and set aside in §4. With the split, both `DeferredType` and any "is this slot
> open?" discriminator disappear, taking the scan / dimension-genericity hack
> with them.

> **Status**: design analysis / forward-looking proposal (not a decision). It
> deliberately ignores backwards-compatibility so the target design is stated in
> its purest form; a pragmatic migration path is sketched at the end.
>
> **Baseline.** This analysis assumes today's mainline gt4py state:
> `DeferredType` and `TypeVarType` are **two separate classes** (the
> dtype-generics work of gt4py ADR 0023 added `TypeVarType` and deliberately left
> `DeferredType` in place). A branch additionally explores *merging* the two — a
> candidate "ADR 0024" — which is **not in mainline gt4py**; it is treated here as
> one possible direction (§4), not as the status quo.
>
> **Related & recommended sequencing.** Builds on the dtype-generics work
> ([[ideas/havogt/dtype-generic-fields|Dtype-generic fields]] / gt4py ADR 0023,
> which supplies `TypeVarType`). Its decisive payoff — **one *kinded* polymorphism
> mechanism for dtype *and* dimensions** — only materializes once dimensions can
> appear as type parameters, i.e. once dimensions are *proper types*
> ([[ideas/havogt/dimension-generic-fields|Generic dimensions]], Part I), so it is
> **recommended only after (or co-designed with) that work** (§8). The motivating
> R5 "scan hack" also drives [[ideas/havogt/scan-redesign|the scan redesign]].

This is a forward-looking analysis of how `gt4py.next` represents *not-yet-known* and
*generic* types, why the current representation feels patched, and what a clean-slate
design would look like. It is intended to inform — not pre-decide — the future generic
work (notably dimension genericity). It deliberately ignores backwards-compatibility so
the target design is stated in its purest form; a pragmatic migration path is sketched at
the end.

## TL;DR

The system uses two placeholder types — the anonymous `DeferredType` and the named,
value-constrained `TypeVarType` (gt4py ADR 0023) — to express at least **six different
things**. Those six things are points in a **three-axis space** (binding mode × sort ×
constraint), and the current two-class design projects them onto those two overloaded
classes in an ad-hoc way (and a proposed merge, §4, would project them onto *one*). The
fix is to split the placeholder along its main axis into three constructs that live in
three different places:

- **absence** of a type → `Optional[TypeSpec]` (kept *out* of the type language),
- a **universal parameter** → a kinded `TypeScheme` / `TypeParam` / `ParamRef`,
- a **value-dependent / computed return** → an explicit *type rule* on builtins.

When that is done, both `DeferredType` and any "is this slot still open?" discriminator
disappear, and the scan / dimension-genericity hack dissolves with them.

## 1. What is actually being conflated

Tracing every construction and consumption site, the two placeholders wear six semantic
hats.

| #  | Role | Representative site | What it really is |
|----|------|---------------------|-------------------|
| R1 | Uninitialized `.type` on an AST node before deduction runs | `Expr.type = DeferredType(constraint=None)` default (`ffront/field_operator_ast.py`) | **Absence** of information (⊥); an IR-completeness artifact, not a type |
| R2 | Category-bounded placeholder / expectation | `func_to_past` / `func_to_foast` set `DeferredType(constraint=ProgramType/FunctionType/TupleType)` | A **kind expectation** used for assertions |
| R3 | "Inference gave up / legitimately polymorphic intermediate", tolerated | iterator `inference.py` returns `DeferredType(None)` for undeclared symbols, scan-tuple results, `tuple_get` on deferred | An **unsolved existential** the iterator IR is content to leave unsolved (also absence) |
| R4 | Named, value-constrained dtype generic | `TypeVarType(name, constraints)` | Genuine **universal polymorphism** (∀ over a finite scalar set), monomorphized per call |
| R5 | Dimension genericity for scans | `ffront/type_info.py` rewrites the *entire* program-context signature to `DeferredType(None)` | Faked **∀ over dimensions**: there is no concept, so all structure is erased |
| R6 | Value-dependent builtin return **and** its dispatch trigger | `astype` returns `DeferredType(constraint=(FieldType, ScalarType, TupleType))`; `visit_Call` routes to `_visit_astype` precisely because `not is_concrete(return_type)` | A **type-level function of a value** *plus* an overloaded control-flow signal |

R4 is the dtype-generics feature ([[ideas/havogt/dtype-generic-fields|Dtype-generic
fields]] / gt4py ADR 0023); R5 is the dims-genericity hole that both
[[ideas/havogt/dimension-generic-fields|Generic dimensions]] (Part III) and
[[ideas/havogt/scan-redesign|the scan redesign]] need fixed. Everything else (R1, R2, R3,
R6) sits on `DeferredType`.

The decisive evidence that R5 is a known hack is in `ffront/type_info.py`, in the function
that builds a scan operator's program-context signature:

```python
# TODO(tehrengruber): What we actually want is a generic type here, but we don't
#  have that concept yet.
as_deferred_type_with_same_structure = type_info.tree_map_type(
    lambda _: ts.DeferredType(constraint=None)
)
```

Every parameter and return of the scan signature is replaced by `DeferredType(None)` —
both the dimensions *and* the dtypes are erased — purely because there is no way to say
"generic in the dimensions".

Three observations:

- **R1, R2, R3 are all "absence"**, dressed up as a type so it can sit in a non-optional
  `.type` field and flow through `isinstance` checks.
- **R4 and R5 are the same phenomenon** (universal parametric polymorphism) at two
  different **sorts** (dtype vs dimensions) — but R4 got a real construct (`TypeVarType`)
  and R5 got erasure-to-absence (`DeferredType`).
- **R6 is a third, unrelated thing** (a computed/dependent return) wedged onto
  `DeferredType` and additionally abused as a dispatch flag.

## 2. The three orthogonal axes

Every use above is a point in a 3-axis space; the current design projects it onto two
overloaded classes (and a proposed merge, §4, would project it onto one).

- **Axis A — binding mode (the quantifier).** *absent / not-computed* (R1–R3) vs
  *universal parameter* (R4, R5) vs *value-dependent* (R6). In type theory, "a hole to be
  solved" (existential metavariable) and "a parameter to be instantiated" (universal `∀`)
  are **dual** — opposite quantifiers. Today the two live in *different* classes
  (`DeferredType` is the existential, `TypeVarType` the universal), which is the one thing
  the current design gets right on this axis. The temptation (§4) is to merge them into one
  class tagged by `name is None`; that hand-rolled tag — distinguishing an existential from
  a universal — would be the smell. The clean design instead **keeps Axis A as the primary
  split** and lines it up with three distinct homes (§5).
- **Axis B — sort / kind (what the slot ranges over).** a *dtype*, a *dimension* / *set of
  dimensions*, a *whole type of some category*, a *callable type*. `TypeVarType` is
  dtype-only; `DeferredType.constraint` ranges over "TypeSpec category". Neither can say
  "a dimension".
- **Axis C — constraint.** none / finite value-set (`OneOf`) / category bound (`Bounded`) /
  predicate. `TypeVarType.constraints` (value-set) and `DeferredType.constraint` (category)
  are two points on this single axis that today live as two separate fields on two separate
  classes.

The patchiness is structural: two classes spanning a 3-D space (or, after a merge, one)
means every new corner needs a new special case.

## 3. Why the current two-class design is patchy

The two-class split is right about Axis A (existential vs universal in separate classes)
but ad-hoc about everything else:

- It gives `DeferredType` a `constraint` field that overlaps conceptually with
  `TypeVarType.constraints` — Axis C smeared across both classes (a category bound on one,
  a value set on the other), with no shared vocabulary.
- It makes `DeferredType` do R1/R2/R3/R5/R6 all at once — five unrelated jobs on one
  anonymous class.
- It makes `TypeVarType` a `DataType` subclass and dtype-only, so R5 (dimension genericity)
  has nowhere to go but erasure-to-`DeferredType` — the scan hack.

Two classes, six jobs. The separation exists, but along only one of the three axes, and the
other two leak across the class boundary in ad-hoc ways. That is what a clean design must
fix — and, crucially, what makes a *merge* look attractive (the two classes are confusingly
similar) even though merging is the wrong response (§4).

## 4. A tempting consolidation: merging the two placeholders (the "ADR 0024" idea)

Set side by side, `DeferredType` and `TypeVarType` differ on only two axes: **identity**
(`TypeVarType` has a `name`; `DeferredType` is anonymous) and **constraint generality**
(`TypeVarType` is value-constrained to a finite scalar set; `DeferredType` is
category-bounded or unconstrained). A `DeferredType` is then *almost* a `TypeVarType` with
`name is None` and a generalized constraint. That near-coincidence invites a merge, and a
gt4py branch implements one as a candidate **ADR 0024** (*not* in mainline gt4py):

- a single `TypeVarType(TypeSpec)` with `name: Optional[str] = None` (the discriminator,
  `name is None` ⇔ deferred), `bound` for the category constraint, and `constraints` for
  the value set;
- `DeferredType(...)` retained as a thin **factory** returning
  `TypeVarType(name=None, bound=...)`, so the ~25 construction sites are unchanged;
- an `is_deferred(x)` predicate replacing `isinstance(x, DeferredType)` everywhere.

It is locally attractive: one concept, one class, one predicate, and a clean home for the
`bound=`-constrained generic that ADR 0023 deferred.

**But it consolidates along the wrong axis.** It collapses **Axis A** — the existential (a
hole to be *solved*) and the universal (a parameter to be *instantiated*) — into a single
class discriminated by `name is None`. That hand-rolled tag is exactly the
existential-vs-universal distinction, now smuggled back in as a runtime flag instead of a
type. Concretely it:

- reintroduces the distinction as scattered runtime guards (`is_type_var`-style checks at
  ~8 sites);
- permits illegal field combinations the class can no longer rule out
  (named-without-constraints; `name=None` *with* value constraints);
- creates a footgun: the natural `isinstance(x, ts.TypeVarType)` now also matches deferred
  types, and because the constraint predicates do `all(... for c in constraints)`, an
  **empty** constraint set is *vacuously true* — a deferred type silently passes checks
  meant for value-constrained variables.

**Verdict.** The merge is a reasonable *local* cleanup and a plausible stepping stone, but
not the target: it makes the two-class confusion go away by erasing the one distinction
(Axis A) that the clean design wants to make load-bearing. The right move is to consolidate
by **splitting along Axis A** into three homes (§5), not by collapsing it into one class.
If the merge lands first, the migration path (§7) shows how it still becomes a stepping
stone rather than a dead end.

## 5. Proposed clean-slate design

**Core principle: stop modeling "an open type slot" as one thing. Split it along Axis A
into three constructs that live in three different places.**

### Construct 1 — "Not computed yet" is *absence*, kept out of the type language

R1/R2/R3 should not be types at all. An AST node's type becomes `Optional[TypeSpec]` (or,
pragmatically, a single `Unknown` sentinel that is *documented as algorithm-only*, carries
**no** constraint, and is asserted away by the existing completeness validator before any
"final typed IR").

- The category bound (R2) disappears: the deduction pass already knows structurally what
  kind of node it is visiting; a sanity check becomes an `assert`, not a stored value.
- The iterator's "legitimately unknown" (R3) becomes `None` — which is honest: `None`
  cannot be silently `isinstance`-matched as a real type, so the merge's vacuous-true
  footgun (§4) never arises and even today's "a `DeferredType` flowing as if it were a
  type" becomes a mypy error rather than a runtime surprise.

Payoff: the type language now contains **only real types and type parameters** — no dummies
flowing around.

### Construct 2 — Polymorphism is a first-class, *kinded* scheme (unifies R4 **and** R5)

Introduce standard `∀`-machinery, kinded so it covers dtype *and* dimension genericity with
one mechanism:

```text
TypeParam   = (name, kind, constraint)            # a bound variable of a scheme
   kind        ∈ { DType, Dims, Type, … }          # Axis B, extensible
   constraint  ∈ { Unconstrained, OneOf{…}, Bounded(category) }   # Axis C, unified

ParamRef(TypeSpec)  = reference to a TypeParam by name   # occurrence inside a body
TypeScheme          = (params: tuple[TypeParam, …], body: TypeSpec)   # type of a generic operator
```

- Dtype-generic field operator (the [[ideas/havogt/dtype-generic-fields|dtype-generics]]
  feature, R4):
  `TypeScheme([TypeParam("T", DType, OneOf{f32, f64})], FunctionType(… FieldType(dims=[I, J], dtype=ParamRef("T")) …))`.
- **Dimension-generic scan** (the real fix for R5, the goal of
  [[ideas/havogt/dimension-generic-fields|Generic dimensions]] Part III and a prerequisite
  of [[ideas/havogt/scan-redesign|the scan redesign]]):
  `TypeScheme([TypeParam("D", Dims, Unconstrained)], FunctionType(… FieldType(dims=ParamRef("D") + [axis], …) …))`.
  The program-context signature is now a proper scheme whose dims-parameter binds from the
  concrete call arguments, instead of erasing all structure to `DeferredType(None)`.
- Monomorphization = `instantiate(scheme, {"T": f32, "D": [I, J]})` — one substitution
  routine over `ParamRef`s, kind-agnostic. This is exactly the generalization ADR 0023 said
  it was deferring ("widening the binding environment to dimensions and generalizing
  same-name rejection across type-parameter kinds").

`ParamRef` (universal, instantiated) is now a **different type from absence** (`None` /
`Unknown`), for a *principled* reason (the quantifier), not an ad-hoc class boundary or a
`name is None` flag. `is_generic` becomes "contains a `ParamRef` / is a `TypeScheme`";
`is_concrete` becomes "no `ParamRef`". No "deferred" special-casing — whether or not the
merge of §4 ever happens.

One honest modeling consequence: making dims genericity real means `FieldType.dims` must
admit a parameter in dimension position (e.g. `list[Dimension] | ParamRef`, or a dims-list
that can contain a "rest" parameter). That is the crux of the work — but it is strictly more
honest than erasing the field's structure to a placeholder, and it is the thing that
actually unblocks generic scans. **It also presupposes that a dimension can be carried as a
type parameter at all** — see §8.

### Construct 3 — Value-dependent builtins carry an explicit type rule (R6)

`astype` / `where` / constructors should not encode "compute my return specially" as a
`DeferredType` return that `visit_Call` sniffs via `not is_concrete`. A polymorphic builtin
should carry an explicit **type rule** — a function from argument types to result type —
that deduction invokes directly. This removes both halves of R6: the return is computed by
the rule (no placeholder), and dispatch is driven by *which builtin it is* (no
`is_concrete`-as-flag overload). It also makes the builtin's polymorphism inspectable rather
than implicit.

### How the six roles land

- R1, R2, R3 → **absence** (`Optional` / `Unknown`), out of the type language.
- R4, R5 → **`TypeScheme` + kinded `TypeParam` / `ParamRef`**, one mechanism, dtype *and* dims.
- R6 → **explicit type rule on builtins**, computed returns.

The type language ends up with a sharp three-way separation that maps exactly onto Axis A:
*real types*, *universal parameters* (`ParamRef`, bound by a `TypeScheme`), and *the absence
of a type*. Illegal states (named-without-constraints, deferred-with-value-constraints, a
dtype type-variable used in a dimension position) become unrepresentable rather than
validator-policed — and these are precisely the illegal states the merge of §4 would
instead *allow*.

## 6. Honest trade-offs

- **Not behavior-preserving, and bigger than the merge of §4.** It touches the IR (`.type`
  becomes optional), the deduction passes, monomorphization (now kinded), and introduces
  scheme/param/ref plus a small kind enum. The `Optional[.type]` migration is the invasive
  part (every `.type` access must handle `None`); it is the price of making "absence"
  honest, and it can be staged behind a temporary `Unknown` sentinel.
- **Kinds add ceremony.** For a system that *today* only ships dtype generics, "kinded
  parameters" looks like over-engineering — but the named next step *is* dimension
  genericity, so it is precisely the investment that pays off, and it can start with just
  `{DType, Dims}`.
- **Schemes add an indirection** (operator types become `TypeScheme(params, body)` rather
  than a bare `FunctionType`), rippling into anything that inspects operator signatures.
  That is real churn, but it makes "this operator is generic in X" a first-class, queryable
  fact instead of something inferred by scanning for placeholders.

## 7. Pragmatic migration path

The clean design is not fresh-start in practice; it can be reached incrementally — and if
the merge of §4 lands first, that merge becomes a stepping stone rather than a dead end.
Each step stands on its own and removes one of the six hats:

1. **Introduce `TypeScheme` + kinded `TypeParam`** and migrate the existing dtype
   `TypeVarType` to a `DType`-kinded parameter bound by a scheme. (The current
   [[ideas/havogt/dtype-generic-fields|dtype-generics]] work is most of the way here
   already; this holds whether `TypeVarType` is still its own class or has been merged with
   `DeferredType` per §4.)
2. **Add the `Dims` kind and delete the scan program-context erasure (R5).** Highest-value
   cleanup: it removes a known hack and proves the kind system on a second sort. **Requires
   dimensions to be proper types** ([[ideas/havogt/dimension-generic-fields|Generic
   dimensions]] Part I).
3. **Replace builtin deferred-returns with explicit type rules (R6).**
4. **Migrate `.type` to optional / `Unknown` and delete `DeferredType` entirely (R1–R3).**
   Most invasive, done last.

One-line summary: the system is trying to express *absence*, *a universal parameter*, and
*a computed return* with placeholder types that differ on the wrong axes. Give each its own
home — absence as `Optional`, parameters as a kinded `TypeScheme`, computed returns as type
rules — and both `DeferredType` and any "open slot" discriminator disappear, taking the
scan / dimension hack with them.

## 8. Relation to other proposals and recommended sequencing

This document is a *representation-level* analysis; the concrete generics features it would
re-home live in other proposals. Putting it in context:

- **[[ideas/havogt/dtype-generic-fields|Dtype-generic fields]] (gt4py ADR 0023).** Supplies
  role R4 and the `ts.TypeVarType` this analysis dissects. Construct 2's `DType`-kinded
  `TypeParam` is the clean-slate form of exactly that type variable; migration step 1 starts
  from the dtype work as it stands.
- **The merge idea (§4).** A gt4py branch explores merging `DeferredType` into
  `TypeVarType` as a candidate "ADR 0024"; it is **not in mainline gt4py**. This analysis
  evaluates it (§4) and sets it aside: a defensible *local* cleanup that nonetheless
  consolidates along the wrong axis. The recommendation here is orthogonal to whether that
  merge lands — the clean design absorbs it either way (§7).
- **[[ideas/havogt/dimension-generic-fields|Generic dimensions]].** The pivotal dependency.
  Construct 2's decisive payoff is **one kinded mechanism for dtype *and* dimensions**, and
  its `Dims`-kinded `TypeParam` / `ParamRef`-in-dimension-position is meaningless unless a
  dimension can be carried as a type parameter — i.e. unless **dimensions are proper types**
  (that proposal's Part I, `class I(gtx.Dimension)`). Hence the recommended sequencing
  below.
- **[[ideas/havogt/scan-redesign|Scan redesign]].** The R5 erasure is the running example of
  the whole analysis; the scan redesign needs dims genericity to drop
  `_scan_param_promotion` and the `DeferredType(None)` program-context signature.

### Recommended sequencing

**Pursue this redesign only after — or co-designed with — the *dimensions-as-types* work**
([[ideas/havogt/dimension-generic-fields|Generic dimensions]] Part I), for three reasons:

1. **The payoff is the unification, and the unification needs dimensions.** A kinded
   `TypeScheme` over `{DType}` *alone* is mostly ceremony on top of today's `TypeVarType`
   (§6). Its value appears when the second kind, `Dims`, joins — and `Dims` requires
   dimensions that can appear as type parameters.
2. **Step 2 of the migration path is gated on it.** Deleting the scan erasure (R5) — the
   single most valuable cleanup — is literally "add the `Dims` kind", which cannot exist
   before dimensions are types.
3. **It avoids a wasted intermediate representation.** Building the scheme machinery first
   and retrofitting dimensions later risks a second migration; doing dimensions-as-types
   first lets the scheme arrive already kinded over both sorts.

### A conflict to surface

There is a genuine **representational tension** with
[[ideas/havogt/dimension-generic-fields|Generic dimensions]] §5.2, worth calling out
explicitly (surfacing such conflicts is the point of this garden):

- *Generic dimensions* prototypes dims variables as `ts.DimensionVar` / `ts.DimsVar` that
  subclass `common.Dimension` (**not** `TypeSpec`), *deliberately asymmetric* to
  `ts.TypeVarType`, so the large body of code that merely carries dimensions keeps working
  and only concreteness-checking sites change.
- *This analysis* argues for the opposite: a **single, symmetric, kinded** parameter
  (`TypeParam` of kind `DType` or `Dims`) so dtype and dims genericity share one mechanism
  and the dtype/dims asymmetry disappears.

These are two points on a spectrum, not a contradiction: the `DimensionVar`/`DimsVar`
asymmetry is the *pragmatic, incrementally-landable* form (and an excellent stepping stone),
while the kinded `TypeScheme` is the *purest target* this analysis states with
backwards-compatibility ignored on purpose. Deciding between "keep dims variables asymmetric
forever" and "converge on one kinded parameter" is the open question the two documents
jointly pose — and it should be answered as part of the dimensions-as-types ADR so the two
efforts do not lay down conflicting type-system foundations.
