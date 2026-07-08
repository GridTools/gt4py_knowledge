---
title: "Cross-DSL prior art for boundary-condition syntax"
author: havogt
tags: [boundary-conditions, regions, prior-art, dsl-design, stencil, fem, pattern-matching, halide, devito, gridtools, research, python-versions, peps, ast-rewriting, replay, metaclass]
created: 2026-06-16
---

> **Status**: research appendix to
> [[ideas/havogt/boundary-condition-syntax|Frontend syntax sugar for boundary conditions over concat_where]].
> A web survey (drafted with AI assistance, with source URLs) of how other languages, DSLs, and
> frameworks encode "do X in the interior, Y on the boundary / top level / lateral edge." Quotes
> and signatures are copied from primary sources; URLs are inline. Where a source could not be
> verified verbatim it is flagged.

# How other systems encode interior-vs-boundary

## The one distinction that matters

Across everything surveyed, boundary handling splits into **two archetypes**:

- **Domain-splicing / region-restricted** — a *different computation is attached to a sub-region*.
  The branches may have different shapes/domains and even different physics. **This is exactly what
  `gt4py.next` `concat_where` and `gt4py.cartesian` `region`/`interval` do**, and it is the family
  worth stealing surface syntax from.
- **Pointwise boundary mode** — one uniform kernel runs everywhere and out-of-bounds *reads* are
  re-indexed (clamp / mirror / wrap) or a fallback value is substituted. Same shape, no region
  split. A genuinely different feature (closer to gt4py's pointwise `where` + halo handling).

| System | Boundary primitive | Splice or pointwise? |
| --- | --- | --- |
| **YASK** `IF_DOMAIN` | per-equation sub-domain condition | **splice** |
| **Devito** `SubDomain`/`SubDimension` | `Eq(…, subdomain=…)`, per-axis thickness | **splice** |
| **GridTools / STELLA** | per-`interval` `apply()` overloads, `extent`/halo | **splice** |
| **Dawn / GTClang** | `vertical_region(k_lo,k_hi)`, `boundary_condition` functor | **splice** |
| **gt4py.cartesian** | `with interval(...)`, `with horizontal(region[...])` | **splice** |
| **ExaSlang** | BC in field type + `apply bc to f` | **splice** (generated boundary sweep) |
| **FEniCS / Firedrake / FreeFEM / MFEM / deal.II** | `DirichletBC`/`ds(marker)`/`on(label)`/boundary-id | **splice** (separate term per marked region) |
| **Mathematica** | `DirichletCondition[val, pred]`, `Piecewise` | **splice** (predicate names a region) |
| **Regent / Legion** | `partition`/`image`/`preimage` subregions | **splice** (distinct index partitions) |
| **Halide** | `BoundaryConditions::*`, `clamp`, `select` | **pointwise** (loop-split under the hood) |
| **Lift / RISE** | `Pad(l, r, Pad.Boundary.{Clamp,Mirror,Wrap})` | **pointwise** |
| **Taichi** | `if` in kernel, `ti.select`, `ti.ndrange` bounds | **pointwise** (or restrict iteration) |
| **Pochoir** | `Pochoir_Boundary_*` fallback value function | **pointwise** (one iteration domain) |
| **Futhark** | hand-written `if interior then … else …`, clamp | **either** (idiom is a domain split) |
| **Chapel `StencilDist`** | `fluff`, `updateFluff` | pointwise overlay (halo = cached cells) |
| **NEMO / GFDL FMS** | `lbc_lnk` / `mpp_update_domains` halo, loop bounds | communication + index ranges |

---

## 1. Stencil / HPC DSLs (the closest relatives)

### YASK (Intel) — `IF_DOMAIN`, the closest analog of all

YASK restricts an equation to a spatial sub-domain with a per-equation boolean condition built from
`first_domain_index` / `last_domain_index` — simultaneously the analog of `concat_where`'s `KDim <
n` *and* cartesian's `I[0]`/`I[-1]`:

```cpp
auto sd0 = (x >= first_domain_index(x) + 5) && (x <= last_domain_index(x) - 3);
A(t+1, x) EQUALS  u IF_DOMAIN  sd0;
A(t+1, x) EQUALS -u IF_DOMAIN !sd0;
```

A different *expression* (even a different stencil order) is bound to each sub-domain via mutually
exclusive conditions; the low-level form is `yc_equation_node::set_cond()`. Caveat: sub-domain
conditions are evaluated once at `prepare_solution()`, so they are purely spatial (cannot depend on
the step index). Sources: <http://intel.github.io/yask/api/html/classyask_1_1yc__equation__node.html>,
<https://github.com/intel/yask/blob/master/src/stencils/TestStencils.cpp>.

### Devito — `SubDomain` / `SubDimension`

Boundary regions are objects whose `define()` restricts each dimension by a positional tuple —
`('left', N)`, `('right', N)`, `('middle', N, M)`, or the bare dimension for full extent:

```python
class Left(SubDomain):
    name = 'left'
    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', self.pmls), y: y}

eq_v      = Eq(v.forward, v - dt*grad(p)/density,         subdomain=main)
eq_v_left = Eq(v.forward, (1-d_l)*v - dt*grad(p)/density, subdomain=left)
```

Different physics per region as *separate `Eq` objects over differently-shaped subdomains* — the
most gt4py-like FEM-adjacent model. The underlying primitive `SubDimension.{left,middle,right}` is a
restricted parent dimension with a `thickness`. Sources:
<https://www.devitoproject.org/examples/userapi/03_subdomains.html>,
<https://www.devitoproject.org/examples/userapi/04_boundary_conditions.html>.

### GridTools / STELLA, Dawn / GTClang, gt4py.cartesian (one lineage)

The family gt4py.cartesian descends from expresses the vertical split as **per-interval `apply`
overloads** dispatched at compile time:

```cpp
GT_FUNCTION static void apply(Eval const& eval, first_interval)  { /* top region   */ }
GT_FUNCTION static void apply(Eval const& eval, second_interval) { /* other levels */ }
```

with `axis<2> my_axis(5, 10)`, `full_interval::first_level` / `last_level`, and horizontal halos via
`extent<>` + `halo_descriptor`. Dawn/GTClang lower `vertical_region(k_start, k_end) { … }` blocks
into interval-tagged SIR stages; horizontal BCs are a *separate* `boundary_condition` functor.
gt4py.cartesian's own `with interval(0,1)` / `with horizontal(region[I[0], J[-1]])` is the same
concept — and its Tridiagonal/Thomas solver shows the canonical "first level / all-other levels"
splice. Sources:
<https://gridtools.github.io/gridtools/latest/user_manual/user_manual.html>,
<https://github.com/MeteoSwiss-APN/dawn/blob/master/dawn/examples/tutorial/README.md>,
<https://gridtools.github.io/gt4py/latest/gtscript.html>.

### ExaStencils / ExaSlang — boundary condition is part of the field type

```text
Field rho < global, CellLayout, Neumann > [2] @all
...
apply bc to u
```

The BC is declared once in the field's type (`None` / `Neumann(order)` / a Dirichlet expression /
`applyBoundaries()`), and `apply bc to <field>` triggers a *generated separate sweep* over the
boundary layer — the most declarative of the stencil DSLs.
Source: <https://www.exastencils.fau.de/sphinx/documentation/user/ExaSlang4.html>.

### Halide / Lift / RISE / Taichi — pointwise boundary *modes* (the other archetype)

Halide wraps a `Func` so out-of-bounds reads are re-indexed, then relies on the `likely` intrinsic
to loop-split interior vs edge automatically:

```cpp
Func clamped = BoundaryConditions::repeat_edge(input);   // or constant_exterior / mirror_image
Expr xc = clamp(x, 0, input.width()-1);                  // the manual form it wraps
```

Lift/RISE is a single pointwise policy `Map(f) ∘ Slide(size,step) ∘ Pad(l, r,
Pad.Boundary.{Clamp,Mirror,MirrorUnsafe,Wrap,WrapUnsafe})` — one boundary mode for the whole array,
*not* a domain split. Taichi has no boundary construct at all: `if` on the index inside the kernel,
`ti.select(cond,a,b)`, or restrict iteration with `ti.ndrange`. Sources:
<https://github.com/halide/Halide/blob/main/src/BoundaryConditions.h>,
<https://github.com/lift-project/lift/blob/master/src/main/ir/ast/Pad.scala>,
<https://docs.taichi-lang.org/docs/operator>.

### Pochoir — a fallback *value* function (a useful contrast)

```cpp
Pochoir_Boundary_2D(aperiodic_2D, arr, t, i, j)
    return 0;                       // Dirichlet: value for any off-domain read
Pochoir_Boundary_End
a.Register_Boundary(aperiodic_2D);
```

There is **one** iteration domain; the boundary function is invoked per out-of-range *read* from
inside the interior kernel and returns a scalar. The HOTPAR paper notes it "do[es] not handle
spatial boundaries that change with time" — for a genuine region split you must put `if` checks in
the kernel. A clear example of *not* being a domain splice.
Source: <https://www.cs.stonybrook.edu/~rezaul/papers/pochoir-HOTPAR-11.pdf>,
<https://github.com/Pochoir/Pochoir/blob/master/examples/tb_heat_2D_NP.cpp>.

### Futhark — no primitive; the idiom is a domain split

No builtin stencil/boundary; the canonical Gaussian-blur is exactly "interior stencil vs boundary
copy" as a per-point `if`:

```futhark
map (\col -> if row>0 && row<rows-1 && col>0 && col<cols-1
             then new_value channel row col
             else channel[row,col])
    (iota cols)
```

The performance variant splits into separate maps over sub-ranges joined by `concat`/`scatter` —
a hand-rolled `concat_where`. Source: <https://futhark-lang.org/examples/gaussian-blur.html>.

### Regent / Legion, Chapel — distributed region/halo (orthogonal)

Regent/Legion build **distinct logical subregions** with dependent partitioning
(`partition(equal, …)` for the interior, `create_partition_by_restriction` / `image` / `preimage`
for ghosts) — separate index sets, set-algebra `|`/`&`/`-`. Chapel's `StencilDist` instead makes
the halo a *cached overlay of the same array* (`fluff=(1,1)`, `A.updateFluff()`), iterated only over
the `boundingBox`. These are about distribution/halo exchange, not in-kernel boundary expression.
Sources: <https://regent-lang.org/reference/>, <https://legion.stanford.edu/tutorial/multiple.html>,
<https://chapel-lang.org/docs/modules/dists/StencilDist.html>.

---

## 2. PDE / FEM frameworks — boundary as a separate marked object

Every FEM framework attaches the BC to a marker-identified subregion (a separate equation / integral
term / DOF constraint), never a same-shape pointwise select:

- **FEniCS / DOLFINx**: strong `dirichletbc` on located DOFs; weak Neumann/Robin as boundary
  integrals `*ds(marker)`; `MeshTags` carry integer markers; `ufl.conditional` exists but is for
  *coefficients*, not the boundary split.
  <https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html>
- **Firedrake**: `DirichletBC(V, value, subdomain_id)` (`id`, tuple, or `"on_boundary"`/`"top"`);
  `ds(id)`. <https://www.firedrakeproject.org/boundary_conditions.html>
- **FreeFEM**: `border C(...){ … label=1; }` then `on(1, u=0)` and `int1d(Th, label)(…)`.
  <https://doc.freefem.org/tutorials/poisson.html>
- **MFEM**: `bdr_attributes` + `ess_bdr` marker array + `GetEssentialTrueDofs`.
  <https://github.com/mfem/mfem/blob/master/examples/ex1.cpp>
- **deal.II**: `boundary_id` + `interpolate_boundary_values` / `AffineConstraints`.
  <https://github.com/dealii/dealii/blob/master/examples/step-3/step-3.cc>

The cross-FEM takeaway: "do X interior, Y on boundary" is universally a **separate object on a
lower-dimensional, marker-identified subregion** — a domain-splice, *not* `where(cond, A, B)`.

### Spectral — Dedalus (tau method)

BCs are *extra algebraic equations* on a lower-dimensional object (a Chebyshev endpoint / `z=const`
slice), added distinctly from the bulk PDE:

```python
problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
problem.add_equation("b(z=0) = Lz")     # boundary condition
problem.add_equation("b(z=Lz) = 0")
```

Source: <https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_2d_rayleigh_benard.html>.

---

## 3. Climate/weather Fortran frameworks (how the incumbents do it)

- **ICON** (the model icon4py reimplements): lateral boundary / nudging / halo zones are addressed
  by **row-based loop bounds** — `rl_start`/`rl_end` + `refin_ctrl` + `get_indices_c/e/v`. This is
  the conceptual origin of icon4py's `start_edge_lateral_boundary` / `end_edge_nudging` index
  symbols that now appear as `concat_where` conditions.
- **OpenFOAM**: declarative per-patch dictionary — each named patch picks a `type` that selects a
  distinct `fvPatchField` subclass (`fixedValue`, `zeroGradient`, `inletOutlet`, …) with its own
  `updateCoeffs()`/`evaluate()`:

  ```text
  boundaryField {
      inlet  { type fixedValue;  value uniform (10 0 0); }
      outlet { type zeroGradient; }
      upperWall { type noSlip; }
  }
  ```
  Genuinely *different computations* per patch, not just different constants. Source:
  <https://doc.cfd.direct/openfoam/user-guide-v13/boundary-conditions>.
- **NEMO** `lbc_lnk('name', field, 'T'|'U'|'V'|'F', ±1.)` — a sign-aware MPI **halo exchange**;
  physical open boundaries (BDY: Flow-Relaxation + Flather) are a *separate* list of rim points
  (`nambdy_index`). **GFDL FMS** `mpp_domains` — compute domain `isc:iec` (interior loop) vs data
  domain `isd:ied` (allocation incl. halo), with `mpp_update_domains` filling the halo. Halos are
  *communication + loop bounds*, not in-kernel selects.

---

## 4. Language features for region/interval dispatch (the "(mis)use Python" angle)

### Rust / Swift — first-class range patterns + guards (the ergonomic ideal)

```rust
match altitude {
    TROPOSPHERE_MIN..=TROPOSPHERE_MAX => "troposphere",   // bounds may be named consts
    STRATOSPHERE_MIN..=STRATOSPHERE_MAX => "stratosphere",
    _ => "outer space, maybe",
}
```

Inclusive `a..=b` / half-open `a..b` range patterns, `|` or-patterns, `if` guards, `@` bindings, and
**compiler-enforced exhaustiveness** (missing region = compile error). Swift is equivalent
(`case 0...5, 22...:` + `where`). This is what a "match on a region index" *should* read like.
Sources: <https://doc.rust-lang.org/reference/expressions/match-expr.html>,
<https://docs.swift.org/swift-book/documentation/the-swift-programming-language/controlflow/>.

### Python `match` — **no range patterns** (decisive for Proposal 2)

PEP 634 defines only literal / capture / wildcard / value / group / sequence / mapping / class
patterns. There is **no** range or comparison pattern. The only ways to express an interval are a
**guard**:

```python
match value:
    case x if 0 <= x < 10: ...     # == if/elif with extra ceremony
    case x if 10 <= x < 20: ...
```

or, per PEP 635's own recommendation, **custom matchable objects** (`case InRange(0, 6):` via
`__match_args__`). PEP 635 verbatim: *"Rather than creating a special-case syntax for ranges, it was
decided that allowing custom pattern objects … would be more flexible and less ambiguous."* So a
native `match` over a dimension degrades to guarded `if`/`elif`. Sources:
<https://peps.python.org/pep-0634/>, <https://peps.python.org/pep-0635/>.

### Mathematica — predicate-names-a-region + `Piecewise`

```wolfram
DirichletCondition[u[x, y] == 0, x == 0 && y == 0]     (* BC where predicate is True *)
Piecewise[{{v1, c1}, {v2, c2}, ...}, default]          (* ordered first-true; default fills gaps *)
```

`DirichletCondition`/`NeumannValue` bind a value to the boundary part where a **coordinate
predicate** holds ("any logical combination of equalities and inequalities"); `Piecewise` is the
ordered `(value, condition)` table with a default — the visual template for Proposal 4. Sources:
<https://reference.wolfram.com/language/ref/DirichletCondition.html>,
<https://reference.wolfram.com/language/ref/Piecewise.html>.

### MATLAB `pdepe`, APL `⌺`

`pdepe`'s boundary function returns `(pl,ql)` for the left and `(pr,qr)` for the right boundary —
region selection by **return-tuple position**. APL's stencil `⌺` invokes the kernel dyadically with
a left argument `⍺` that says *how much of this window is edge-fill and on which side* (`0 0` =
interior, `1 ¯1` = a corner) — the kernel is *told its boundary offset* and can branch on it. A novel
"thread the position in" idea, orthogonal to `concat_where`. Sources:
<https://www.mathworks.com/help/matlab/ref/pdepe.html>,
<https://help.dyalog.com/19.0/Content/Language/Primitive%20Operators/Stencil.htm>.

---

## 5. Python itself: shipped, upcoming, and rejected (verified 2026-07)

Statuses verified against `peps.python.org` (PEP index API + individual pages) and the official
3.14/3.15 "What's New"; grammar claims re-checked live on CPython 3.13 (no relevant grammar changed
in 3.14/3.15).

**Shipped:**

- **PEP 750 — template strings** (Final, 3.14): `t"..."` evaluates interpolations **eagerly**
  ("evaluated immediately … not deferred or wrapped in lambdas"; delayed evaluation is an
  explicitly Rejected Idea), but each `Interpolation` carries `expression: str`, "the original text
  of the interpolation" — a limited, sanctioned quasi-quotation channel.
  <https://peps.python.org/pep-0750/>
- **PEP 649 / PEP 749 — deferred annotations** (Final, 3.14): module/class annotations become
  lazy, but **function-local annotations are untouched**: "these annotations have no runtime
  effect–they're discarded at compile-time" (PEP 649), echoing PEP 526: "Annotations for local
  variables will not be evaluated: `def f(): x: NonexistentName  # No error`". This is the fact the
  annotated-assignment marker variant of Proposal 5 relies on.
  <https://peps.python.org/pep-0649/> <https://peps.python.org/pep-0526/>
- **PEP 810 — explicit lazy imports** (Final, 3.15): `lazy import json`; `lazy` is a soft keyword
  special to import statements only — not reusable by libraries. Irrelevant to region dispatch.
  <https://peps.python.org/pep-0810/>
- **PEP 798 — unpacking in comprehensions** (Final, 3.15): `[*L for L in lists]`. Mildly nice for
  piece lists; nothing more. <https://peps.python.org/pep-0798/>
- **PEP 646** (Final, 3.11): star-unpacking is legal inside subscripts — `concat[*regions]` works
  (verified live). <https://peps.python.org/pep-0646/>

**Pipeline:** nothing syntax-level is accepted for 3.16 yet. Drafts worth watching: **PEP 718
subscriptable functions** (would make `concat[...](...)` a typed first-class shape), PEP 653
(match-semantics refinement — still no range patterns), PEP 822 (dedented strings).

**Rejected / dormant — the walls of the design space:**

- **PEP 335 — overloadable boolean operators**: *Rejected* (Guido, 2012: "I really dislike that
  the PEP adds to the bytecode for all uses of these operators even though almost no call sites
  will ever need the feature"). Verified live: `__bool__` returning a non-`bool` raises
  `TypeError: __bool__ should return bool` — `if`/`and`/`or` are un-interceptable forever.
  <https://peps.python.org/pep-0335/>
- **PEP 637 — keyword arguments in subscripts**: *Rejected*. `concat[K[0], default=x]` is a
  SyntaxError (verified live). <https://peps.python.org/pep-0637/>
- **PEP 638 — syntactic macros** (`name!` AST transforms): *Draft*, dormant since 2020.
  <https://peps.python.org/pep-0638/>
- **PEP 3150 (`given`) / PEP 403 (`@in`)** — block-as-expression: both *Deferred* ("primarily due
  to the lack of compelling real world use cases"). <https://peps.python.org/pep-3150/>
- **Pattern matching**: no range-pattern PEP exists. Verified live: `case K[0]:` is a
  **SyntaxError**; value patterns must be dotted names; a bare `case K:` *captures*; and
  `case range(0, 5):` parses as a class/isinstance pattern, not a range test.
  <https://peps.python.org/pep-0634/>

## 6. Precedents for the implementation tricks

- **Decoration-time AST rewriting is production-proven.** pytest: "Reporting details about a
  failing assertion is achieved by rewriting assert statements before they are run", via an import
  hook writing new `.pyc` files (<https://docs.pytest.org/en/stable/how-to/assert.html>). MacroPy:
  "transformations on the abstract syntax tree (AST) of a Python program at *import time*"
  (<https://github.com/lihaoyi/macropy>). And gt4py.cartesian itself is `inspect.getsource` +
  `ast.parse` (`src/gt4py/cartesian/utils/meta.py`, `frontend/gtscript_frontend.py`).
- **Bytecode-level alternative.** Numba compiles from **bytecode**, not source: "Numba is a
  compiler for Python bytecode…"; its first stage recovers control flow from the bytecode
  (<https://numba.readthedocs.io/en/stable/developer/architecture.html>). Relevant when source is
  unavailable (REPL, exec).
- **Tracer-refuses-bool** (the model for `Domain.__bool__` raising): JAX
  `TracerBoolConversionError` — "occurs when a traced value in JAX is used in a context where a
  boolean value is expected", pointing users to `lax.cond` (<https://docs.jax.dev/en/latest/errors.html>);
  `torch.fx` — symbolic tracing "does not currently support *dynamic control flow*", raising
  "symbolically traced variables cannot be used as inputs to control flow"
  (<https://docs.pytorch.org/docs/stable/fx.html>).
- **Replay / multi-shot execution** (the model for Proposal 1's option d): thermometer
  continuations — Koppel, Scherer, Solar-Lezama, "Capturing the Future by Replaying the Past
  (Functional Pearl)", PACMPL 2(ICFP) 2018: "replaying the entire computation from the start,
  guiding it using a recording" (<https://doi.org/10.1145/3236771>). In Python, the concolic
  verifier **CrossHair** "repeatedly calls the function you want to analyze", forcing branch
  booleans and "remember[ing] what decisions it has made so that it can make different decisions
  on future executions" (<https://crosshair.readthedocs.io/en/latest/how_does_it_work.html>). No
  array DSL is known to use replay — they refuse `bool()` or rewrite instead.
- **`class`-body capture** (the model for Proposal 9): PEP 3115 — "`__prepare__` returns a
  dictionary-like object which is used to store the class member definitions during evaluation of
  the class body" (<https://peps.python.org/pep-3115/>); the metaclass call may return any object,
  which the `class` statement simply binds (datamodel: the result "is bound in the local
  namespace"). Real DSL precedent: `enum`'s `EnumMeta.__prepare__` returns `_EnumDict` (ordered,
  duplicate-tracking member capture); Django's declarative `Model` bodies via `ModelBase`.
  Verified empirically: the namespace observes every binding in order (rebindings included), a
  metaclass returning `42` binds the class name to `42`, and class bodies read enclosing function
  locals — with the caveat that a name *assigned in* the body and read before assignment falls
  back to globals, not the enclosing function.
- **Model-capture `with` blocks** (why Proposal 3/7 need the rewrite): PyMC — "all PyMC objects
  introduced in the indented code block below the `with` statement are added to the model behind
  the scenes" (<https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html>)
  — but registration happens at object *construction*; PEP 343 gives a `with` block only
  `__enter__`/`__exit__`, so plain assignments to local names trigger no hook.
- **Piece assembly as an array idiom** (Proposal 8): xarray `combine_by_coords` — "Attempt to
  auto-magically combine the given datasets … into one by using dimension coordinates"
  (<https://docs.xarray.dev/en/stable/generated/xarray.combine_by_coords.html>).

## Synthesis for the proposal

1. The model to imitate is **domain-splicing**: YASK `IF_DOMAIN`, Devito `subdomain=`, GridTools
   per-interval `apply`, Mathematica `DirichletCondition[val, pred]` — all bind *a different
   computation to a region named by a predicate/bounds*. `concat_where` is already in this family;
   we only need nicer surface syntax.
2. The recurring surface idiom is an **ordered list of `(region predicate → branch)` pairs with a
   default** (Mathematica `Piecewise`, OpenFOAM `boundaryField`, FEM stacked BCs). That is exactly
   Proposal 4's `cases(...)`, and Proposals 1/3/5 are statement sugar over it.
3. **Python `match` is the weakest fit** (no range patterns; guards-only) — flagged so we do not
   over-invest in Proposal 2.
4. Two ideas from the edges are worth remembering even if not adopted now: ExaSlang's
   *BC-in-the-type + `apply bc`* (declarative), and APL's *boundary-offset token threaded into the
   kernel* (orthogonal to selection).
5. Python's own trajectory (§5) does not change the picture: 3.14/3.15 add nothing for region
   dispatch (t-strings' source-text capture is the one new primitive, and it trades Python syntax
   for strings), and the enabling PEPs are rejected (335, 637), dormant (638), or deferred
   (3150/403). The durable implementation options are the tricks in §6 — AST rewriting, replay,
   and `__prepare__` capture.

### Source-reliability notes

GitHub raw files, official docs, and language references were quoted verbatim. A few PDFs (Dawn,
Lift CGO'18, dpl2016) and some vendor doc portals (Wolfram `reference.wolfram.com`, ESI
`openfoam.com`, parts of dealii.org) returned 403/were image- or FlateDecode-encoded; in those cases
the API names/snippets were taken from the corresponding source repositories or non-blocked mirrors
and are noted as such above. STELLA has no public extractable source (paywalled paper); its role is
cited only as the GridTools/cartesian ancestor.
