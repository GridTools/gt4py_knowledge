---
title: A discretization-independent surface syntax for finite-difference computations
author: egparedes
tags: [finite-difference, finite-volume, mesh-invariant, structured, unstructured, exterior-calculus, dec, mimetic, de-rham, location-typing, stencil, weight-generation, rbf-fd, gfdm, moment-matching, conservation, field-operators, icon, pace, pmap, fv3, ifs-fvm, arakawa-staggering, c-grid, d-grid, a-grid, dynamical-core, hevi, vertical-solve, elliptic-solve, mpdata, semi-implicit, miura, atlas, prior-art, dsl-design]
created: 2026-06-30
---

# A Discretization-Independent Surface Syntax for Finite-Difference Computations in Weather and Climate

*A design study: from the computational patterns of atmospheric dynamical cores, through the mathematical notations and programming syntaxes used to express them, to a concrete proposal for a mesh-invariant front-end that lowers to GT4Py.next. The proposal is developed against the ICON nonhydrostatic dynamical core and then generalized across grid staggerings to cover the other production GT4Py cores — PACE (FV3, D-grid) and PMAP (IFS-FVM, co-located A-grid).*

> **Scope note (this revision).** The first version of this document specialized to ICON's triangular C-grid. But GT4Py is the shared backend of at least three dynamical cores with *different Arakawa staggerings*: ICON (C-grid), PACE / FV3 (D-grid, with C-grid advective winds), and PMAP / IFS-FVM (co-located A-grid finite volume with a 3D semi-implicit solve). The location-typed surface of Proposal 3 quietly *is* a C-grid commitment. This revision makes that relationship explicit (§6.4), generalizes the proposals so staggering becomes a declared, pluggable property rather than a baked-in assumption (§6.5), and works the generalization through PACE and PMAP (§9).

---

## Contents

1. The scientific domain
2. Computational patterns and the unifying invariant
3. Mathematical notations (by discretization method)
4. Programming syntaxes (by discretization method)
5. Landscape: what exists, and the actual gap
6. The proposals: a mesh-invariant FD surface (incl. Arakawa staggering and its generalization)
7. Specification of the user-written surface
8. Application to the ICON dynamical core (C-grid)
9. Beyond ICON: staggering-parametric application to PACE (D-grid) and PMAP (A-grid)
10. Synthesis and open questions
11. References

---

## 1. The scientific domain

Physics-based numerical weather prediction (NWP) and climate modelling integrate the equations of atmospheric (and oceanic) motion forward in time. The atmospheric dynamical core solves, in the modern nonhydrostatic case, the **fully compressible Euler equations** for momentum, mass continuity, and thermodynamics, coupled to an equation of state. These continuous PDEs are discretized in space onto a grid and advanced by a time-stepping scheme, with separately-treated "physics" parametrizations (radiation, microphysics, turbulence) supplying source terms.

Spatial discretization uses one of a small number of methods — **finite differences (FD)**, **finite volumes (FV)**, **finite elements (FE)**, spectral methods — over a mesh that is either:

- **structured** (a logically Cartesian lattice; topology is implicit in integer index arithmetic), or
- **unstructured** (an arbitrary cell/edge/vertex complex; topology is explicit in connectivity tables).

The two mesh families dominate different parts of the field. Classical limited-area and many global models (COSMO, WRF, FV3) are structured; the current generation of global icosahedral/cubed-sphere models (ICON, MPAS, LFRic) are horizontally unstructured. A recurring and important asymmetry: icosahedral models such as ICON are **unstructured in the horizontal but structured in the vertical** (terrain-following levels), so the vertical direction retains index arithmetic even when the horizontal does not.

### The central observation

The computations are *conceptually identical* on both mesh families — a divergence is a divergence — but must be *expressed differently* in code. Nearly every structured-vs-unstructured difference reduces to a single axis:

> **How is adjacency resolved?** Structured grids encode topology *implicitly* in index offsets (`i±1`, `j±1`). Unstructured grids encode it *explicitly* in connectivity tables (neighbour lists) plus precomputed **geometric factors** (edge lengths, cell areas, orientation signs).

Everything that follows — every operator, every performance characteristic, every abstraction boundary — is a consequence of "arithmetic offset" versus "table lookup + geometric weight." This document is, in effect, an extended argument that because the *conceptual* operator is shared, a single high-level syntax can express both, with the mesh supplying the resolution of adjacency at compile time.

---

## 2. Computational patterns and the unifying invariant

The following patterns recur across atmospheric dynamical cores, ordered from trivial to involved. For each, the structured and unstructured expressions are contrasted. ICON / GT4Py.next vocabulary is used where it is the natural idiom.

| # | Pattern | Structured expression | Unstructured expression |
|---|---|---|---|
| 1 | **Pointwise / map** (EOS, source terms) | elementwise array op | identical — embarrassingly parallel, no neighbour access |
| 2 | **Local neighbour access** | strided offset `phi[i+1,j]` | indirect gather `phi[E2C[e,0]]` (index is itself a lookup) |
| 3 | **Gradient (∇)** | centred difference `(phi[i+1]−phi[i])/dx` | edge-normal difference `(phi[c1]−phi[c0])/dual_edge_len` |
| 4 | **Divergence (∇·)** — FV/Gauss | sum of staggered flux differences | surface integral: `Σ_{e∈C2E} u_n·edge_len·orient / area` |
| 5 | **Curl / vorticity (∇×)** | cross-difference at corner | circulation around dual cell: `Σ_{e∈V2E} u_t·edge_len·sign / dual_area` |
| 6 | **Laplacian / diffusion (∇², ∇⁴)** | fixed 5-/7-point stencil | `div(grad(·))`, or a precomputed `nabla2` coefficient stencil; ∇⁴ widens to second-ring |
| 7 | **Staggered interpolation / reconstruction** | 2-point average `0.5*(phi[i]+phi[i+1])` | weighted neighbour sum with precomputed coefficients; RBF/least-squares for full vectors |
| 8 | **Vertical column operations** | `k±1` offsets; sequential tridiagonal sweep (Thomas) | **identical** — vertical is structured regardless of horizontal mesh |
| 9 | **Neighbour reductions** (sum/min/max/avg) | hand-unrolled fixed-arity | `neighbor_sum` / `min_over` / `max_over` over a connectivity |
| 10 | **Flux-form conservative advection + limiters** | directional flux differences + slope limiter | reconstruct → edge fluxes → divergence; limiter needs min/max over neighbours |
| 11 | **Implicit / elliptic solves** | banded matrices, geometric multigrid, FFT | general sparse (CSR), Krylov + algebraic multigrid; ICON sidesteps via HEVI split |
| 12 | **Halo exchange / domain decomposition** | contiguous halo slabs from Cartesian partition | graph partitioning (METIS) + explicit halo index lists; depth set by widest stencil |
| 13 | **Scatter-vs-gather / write conflicts** | naturally gather; no races | edge-scatter races on shared cells → needs coloring/atomics, or reformulate as gather |

### Cross-cutting performance characteristics

- **Memory locality.** Structured access is contiguous and vectorizes cleanly; unstructured gather/scatter has lower arithmetic intensity per byte and resists SIMD. Mitigation is mesh reordering (space-filling curves: Hilbert/Morton; or Cuthill–McKee) to make connectivity lookups cache-local — a step with no structured analogue.
- **Geometric factors as data.** The defining unstructured trait is that metric terms (areas, edge lengths, orientations, interpolation weights) are precomputed *fields* consumed by every operator, whereas structured codes fold them into constants (`1/dx`). This trades flops for memory traffic and makes unstructured kernels typically memory-bound.

### The unifying invariant

After normalization, **every finite-difference/finite-volume operator on either mesh family has the same shape:**

> **gather → weight → reduce**: gather a fixed-arity neighbour set, multiply by a per-neighbour weight, reduce (almost always a sum).

This is the key fact the entire design rests on. An operator is determined by three things, and structured vs. unstructured differ only in *how each is obtained*, never in *what it is*:

1. **Support** — which neighbours participate (structured: index offsets; unstructured: an *n*-ring of a connectivity).
2. **Weights** — one coefficient per neighbour (structured: Taylor/geometric constants; unstructured: *solved* from a local consistency condition).
3. **Reduction** — combine weighted neighbours (identical in both).

This `gather-weight-reduce` normal form aligns exactly with the shape of a **GT4Py.next field operator**, which is why GT4Py.next is the natural lowering target throughout this document.

---

## 3. Mathematical notations (by discretization method)

Scientific papers and documentation express these patterns in a spectrum of notations, from maximally mesh-agnostic to maximally explicit. Each is associated with particular discretization methods. The running example is the **divergence**:

$$(\nabla\cdot\mathbf{u})_c = \frac{1}{A_c}\sum_{e\,\in\,\text{edges}(c)} n_{c,e}\,u_e\,\ell_e$$

with $u_e$ the edge-normal velocity, $\ell_e$ the edge length, $n_{c,e}=\pm1$ the orientation sign, $A_c$ the cell area.

### 3.1 Continuous vector calculus — *method-agnostic*

$$\nabla\cdot\mathbf{u} = \partial_x u + \partial_y v,\qquad \nabla\phi,\qquad \zeta=\mathbf{k}\cdot(\nabla\times\mathbf{u}),\qquad \nabla^2\phi=\nabla\cdot\nabla\phi$$

The starting point before any grid is committed; the structured/unstructured fork has not yet happened.

### 3.2 Finite-volume integral form (Gauss) — *FV, but the bridge to all methods*

$$\int_A \nabla\cdot\mathbf{u}\,dA = \oint_{\partial A}\mathbf{u}\cdot\mathbf{n}\,d\ell \;\;\longrightarrow\;\; (\nabla\cdot\mathbf{u})_c \approx \frac{1}{A_c}\sum_{e\in\partial c} u_{n,e}\,n_{c,e}\,\ell_e$$

Still mesh-agnostic, but already in the form both discretizations inherit. A structured paper collapses $\partial c$ to four named faces; an unstructured paper leaves it as a variable-cardinality sum. *Same line of mathematics; the fork is whether $\partial c$ has fixed cardinality.*

### 3.3 Structured index / staggered notation — *FD on structured grids*

The half-index form is the lingua franca of C-grid documentation (Arakawa & Lamb; Durran):

$$(\nabla\cdot\mathbf{u})_{i,j} = \frac{u_{i+\frac12,j}-u_{i-\frac12,j}}{\Delta x} + \frac{v_{i,j+\frac12}-v_{i,j-\frac12}}{\Delta y}$$

Its companion in documentation is the **stencil molecule** — a figure that *only exists because the neighbour pattern is fixed*:

```
        φ(i,j+1)
            |
φ(i-1,j) — φ(i,j) — φ(i+1,j)      (5-point Laplacian)
            |
        φ(i,j-1)
```

There is no analogous molecule for an unstructured operator: the "stencil" is a variable-valence patch, so unstructured papers draw a representative hexagon/triangle and rely on the connectivity-sum form instead.

### 3.4 Connectivity-sum notation — *FV / mimetic FD on unstructured grids*

The explicit-topology form, with an orientation array doing the work the index sign did above. ICON-class papers (Wan et al. 2013; Zängl et al. 2015) and MPAS/TRiSK papers (Ringler et al. 2010) write essentially:

$$D_c = \frac{1}{A_c}\sum_{e\in EC(c)} n_{c,e}\,\ell_e\,u_e$$

with $EC(c)$ the edges-of-cell set and $n_{c,e}$ a tabulated incidence sign. This *is* the mathematical statement of "the loop has a connectivity table," and is the notation closest to the actual Fortran.

### 3.5 Discrete operator / sparse-matrix notation — *unifying, method-agnostic*

Promote the operators to matrices: $\mathsf{D}$ (cells×edges), $\mathsf{G}$ (edges×cells), $\mathsf{C}$ (vertices×edges):

$$\mathbf{d}=\mathsf{D}\,\mathbf{u},\qquad \mathbf{g}=\mathsf{G}\,\phi,\qquad \mathsf{L}=\mathsf{D}\,\mathsf{G}$$

Now structured-vs-unstructured is *purely a sparsity-pattern statement*: $\mathsf{D}$ is **banded/Toeplitz** on a structured grid, **general CSR** on an unstructured one — the same linear map. Mimetic/compatible schemes add the relation $\mathsf{G}=-\mathsf{D}^{\!\top}$ (under the proper mass-weighted inner product), which holds in both cases. This view also maps cleanly to implementation: the operator is a matrix-free stencil-apply or an SpMV, a choice orthogonal to mesh type.

### 3.6 Discrete Exterior Calculus / mimetic notation — *mimetic FD / FV / DEC*

The deepest answer to "conceptually identical, expressed differently." DEC writes velocity as a 1-form and factors every operator into **topology × metric**:

$$\text{div} = \star\, d\, \star,\qquad d = \text{signed incidence matrix (topology only)},\qquad \star = \text{Hodge star (diagonal metric ratios)}$$

The punchline: **the discrete exterior derivative $d$ is the *same* signed-incidence operator for both mesh families** — regular/banded when structured, general sparse when not — and *all* geometry lives in $\star$. "Structured vs. unstructured" reduces, in this notation, to the regularity of one incidence matrix. This is the framing of TRiSK and the Thuburn–Cotter mimetic literature, and is why those schemes port between grids cleanly.

### 3.7 The finite-element clarification

A correction worth stating explicitly, because it is a common slip. **Firedrake and FEniCS are finite-*element* frameworks**; their Unified Form Language (UFL) expresses weak forms integrated over elements. They are **not** a finite-difference syntax — there is no index arithmetic, no Taylor-difference operator, no structured-grid stencil. The mesh is treated as an unstructured FE complex regardless of how regular it looks.

The reason FE nonetheless recurs in FD discussions is genuine but conceptual: **compatible (mimetic) finite elements were designed to reproduce the mimetic properties of staggered C-grid FD/FV schemes** (the Cotter–Shipton / Thuburn–Cotter line, underpinning Gusto and the Met Office's LFRic). The sharp relationships are the lowest-order equivalences:

- **RT0 (velocity) + DG0 (pressure) on quads**, mass-lumped, recovers a scheme essentially equivalent to the MAC / Arakawa C-grid staggered FV scheme.
- **The same construction on triangles/hexagons** generalizes TRiSK; Cotter & Shipton (2012) derive TRiSK-type schemes as a mixed FE method — the explicit bridge.
- **Finite Element Exterior Calculus** (Arnold–Falk–Winther) is the rigorous frame, and is the *same* de Rham / discrete-differential-form story as the DEC $\text{div}=\star d\star$ factorization.

So at lowest order, mimetic FD, the C-grid/TRiSK FV schemes, and compatible mixed FE are different *derivations* of overlapping discrete operators. FE belongs in a discussion *adjacent* to FD cores — an alternative dynamical-core technology targeting the same operators and conservation properties — not as another syntax for finite differences.

### Notation summary

| Notation | Primary method association | Mesh-explicitness |
|---|---|---|
| Continuous vector calculus | all | none (pre-grid) |
| FV integral (Gauss) form | FV (and bridge to FD/FE) | none |
| Structured index / staggered | **FD** (structured) | implicit offsets |
| Connectivity-sum | **FV / mimetic FD** (unstructured) | explicit tables |
| Sparse-matrix / discrete operator | all (unifying) | sparsity pattern only |
| DEC / mimetic ($d$, $\star$) | **mimetic FD / FV / DEC** | one incidence matrix |
| UFL weak form | **FE** (compatible/mixed FE → mimetic) | fully implicit (mesh object) |

---

## 4. Programming syntaxes (by discretization method)

Implementations span the same spectrum, from maximally explicit (Fortran connectivity loops, where every gather is visible) to maximally implicit (UFL `div(u)`, where the mesh has vanished from the source). The divergence example recurs.

### 4.1 Fortran, structured — *FD/FV* (COSMO / WRF / CROCO idiom)

```fortran
do j = jstart, jend
  do i = istart, iend
    div(i,j) = (u(i+1,j) - u(i,j)) * rdx &
             + (v(i,j+1) - v(i,j)) * rdy   ! C-grid: u at i+1/2 stored as u(i+1)
  end do
end do
```

Topology implicit in offsets; metric folded into constants.

### 4.2 Fortran, unstructured — *FV / mimetic FD*

MPAS keeps connectivity arrays visible:

```fortran
do iCell = 1, nCells
  div(iCell) = 0.0_RKIND
  do i = 1, nEdgesOnCell(iCell)
    iEdge = edgesOnCell(i, iCell)
    div(iCell) = div(iCell) + edgeSignOnCell(i,iCell) * u(iEdge) * dvEdge(iEdge)
  end do
  div(iCell) = div(iCell) / areaCell(iCell)
end do
```

ICON hand-unrolls the fixed triangle valence and pre-bakes $n_{c,e}\ell_e/A_c$ into a single coefficient field (`geofac_div`):

```fortran
div(jc,jk,jb) =                                                  &
    p_vn(iidx(jc,jb,1),jk,iblk(jc,jb,1)) * geofac_div(jc,1,jb) + &
    p_vn(iidx(jc,jb,2),jk,iblk(jc,jb,2)) * geofac_div(jc,2,jb) + &
    p_vn(iidx(jc,jb,3),jk,iblk(jc,jb,3)) * geofac_div(jc,3,jb)
```

### 4.3 NumPy / array — *FD/FV*

```python
# structured (slicing / periodic roll)
div = (u[1:, :] - u[:-1, :]) / dx + (v[:, 1:] - v[:, :-1]) / dy

# unstructured gather form (race-free)
div = np.sum(u_n[c2e] * geofac_div, axis=1) / cell_area      # c2e: (n_cells, 3)

# unstructured scatter form (edge loop) — races on shared cells; needs add.at / segment_sum
flux = u_n * edge_length
div = np.zeros(n_cells)
np.add.at(div, e2c[:, 0],  flux)
np.add.at(div, e2c[:, 1], -flux)
div /= cell_area
```

### 4.4 GT4Py / GT4Py.next — *stencil DSL (FD/FV), structured and unstructured*

GT4Py originated as a structured (Cartesian) stencil DSL in the STELLA → GridTools lineage. The `gt4py.next` subpackage unifies structured and unstructured at the field-operator level: the *same* surface takes Cartesian offsets (`Ioff[1]`) or connectivity reductions (`E2C`, `C2E`).

```python
# structured backend (offsets, constant geometric weight)
@field_operator
def div_u(u: Field[[E], float]) -> Field[[C], float]:
    return (u(Ioff[1]) - u(Ioff[0])) / dx + (u(Joff[1]) - u(Joff[0])) / dy

# unstructured backend (C2E reduction, weight field)
@field_operator
def div_u(u: Field[[E], float]) -> Field[[C], float]:
    return neighbor_sum(u(C2E) * geofac_div, axis=C2EDim)
```

This is precisely the declarative rewrite of the ICON hand-unrolled Fortran above; `geofac_div` is identical.

### 4.5 Devito — *symbolic FD (structured only)*

```python
from devito import Grid, Function
u = Function(name='u', grid=grid); v = Function(name='v', grid=grid)
div = u.dx + v.dy           # symbolic; lowered to an optimized C stencil
```

Regular grids only — no connectivity concept, which is why it is compact and why it does not cover the unstructured case. Primarily used in seismic imaging.

### 4.6 UFL / Firedrake / FEniCS — *FE (compatible/mixed → mimetic)*

```python
from firedrake import div, dx, TestFunction
q = TestFunction(V)
F = phi * div(u) * q * dx          # 'div(u)' is the same token on any mesh
```

Completely mesh-agnostic by construction: swapping `UnitSquareMesh(...)` for an icosahedral mesh changes nothing in this line. This is "conceptually the same" taken to its logical end — the *source* is mesh-invariant, assembly handles the rest. But it is **finite element**, not finite difference (§3.7).

### 4.7 PSyclone / LFRic — *FE / mimetic, performance-portable* (PSyKAl separation)

The science is written for a single column/cell with connectivity passed in; iteration, halos, and parallelism are code-generated in the "PSy" layer.

```fortran
call invoke( divergence_kernel_type(div_field, u_field) )
```

The operator's *mesh traversal* is generated, not written — the same kernel retargets CPU/GPU without the scientist editing it. Conceptually adjacent to gt4py.next (separate the math from the iteration), different surface syntax.

### 4.8 Discretization-agnostic and DEC DSLs

- **Finch.jl** — a Julia DSL solving PDEs in a discretization-agnostic way, currently **FE and FV**, with code generation to multiple targets (Julia, MATLAB, C++/MPI).
- **Decapodes.jl / CombinatorialSpaces.jl / DiagrammaticEquations.jl** (AlgebraicJulia) — a **DEC** embedded DSL: diagrammatic exterior-calculus equations compiled to runnable Julia. Simplicial complexes (triangulated manifolds) only.
- **PyDEC** — the **DEC** matrix realization: exterior derivative as signed coboundary, Hodge star as diagonal, all as sparse matrices; lowest-order (Whitney) only, no codegen.
- **Medusa** (C++) and the **`rbf`** library (Python) — **meshless / RBF-FD / generalized FD**: strong-form mesh-free solvers that *synthesize* stencil weights by least-squares / polyharmonic-spline collocation.

### Framework-level data structures and compilers

- **Atlas** (ECMWF) — a unified **field/mesh data structure** spanning structured global grids, unstructured meshes, and structured regional grids, with the grid-vs-mesh distinction (implicit i,j indexing vs. explicit connectivity) built into the type system. Provides parallel data structures, not an operator surface.
- **PETSc DMPlex** — general unstructured-topology substrate (incidence/DAG machinery).
- **Open Earth Compiler** — an MLIR `stencil` dialect lowering stencil programs to GPU code; in practice structured-grid in the published work. The `stencil` dialect is now part of **xDSL** and a shared compilation stack across Devito / Open Earth / PSyclone.
- **Dawn / STELLA** — the structured-stencil DSLs that GT4Py descends from.

### Syntax summary

| Syntax | Method | Mesh support | Mesh-explicitness |
|---|---|---|---|
| Fortran loops (COSMO/WRF) | FD/FV | structured | explicit offsets |
| Fortran loops (MPAS/ICON) | FV / mimetic FD | unstructured | explicit connectivity |
| NumPy | FD/FV | both (by idiom) | explicit |
| Devito | FD | structured | symbolic, implicit |
| GT4Py.next | FD/FV (stencil) | **both** | offsets *or* connectivity |
| PSyclone/LFRic | FE / mimetic | unstructured | generated traversal |
| UFL/Firedrake/FEniCS | **FE** | both | fully implicit |
| Finch.jl | FE/FV | both | implicit |
| Decapodes.jl | DEC | unstructured (simplicial) | implicit (diagram) |
| PyDEC | DEC | both (matrix) | sparsity only |
| Medusa / rbf | meshless FD (RBF-FD/GFDM) | unstructured/scattered | solver, not DSL |
| Atlas | data structure | both | grid-vs-mesh typed |

---

## 5. Landscape: what exists, and the actual gap

A "Firedrake for finite differences" — or a "Devito for both mesh types" — would need four ingredients:

1. a **location-typed calculus surface** (operators that hide placement and connectivity);
2. **pluggable weight generation** (mimetic *and* moment-matched);
3. a **portable stencil compiler** spanning structured and unstructured;
4. a **unified field/mesh data structure**.

The research landscape shows that **all four ingredients exist — but in disconnected communities, never assembled into one stack.**

| Ingredient | Where it exists today | Limitation as a complete solution |
|---|---|---|
| Location-typed exterior-calculus surface | Decapodes.jl (DEC eDSL, compiles to code); PyDEC (matrix) | simplicial/unstructured only; generates Julia/SpMV, not portable stencils; PyDEC lowest-order, no codegen |
| Pluggable weight generation | PyDEC / CombinatorialSpaces (mimetic); Medusa, `rbf` (moment-matched RBF-FD/GFDM) | solvers, not a DSL front-end; do not emit field operators; do not unify the two strategies under one surface |
| Portable structured+unstructured stencil compiler | GT4Py.next (EXCLAIM); shared MLIR `stencil` dialect (xDSL) | GT4Py.next carries the types but no calculus surface above them; MLIR stencil dialect is structured in practice |
| Unified field/mesh data structure | Atlas (ECMWF) | data structures and parallelism only; no lowered operator surface |
| Discretization-agnostic surface (adjacent) | Finch.jl (FE/FV); UFL (FE only) | not FD-on-structured-and-unstructured; do not target a portable stencil backend |

### The precise gap

What does **not** exist as an assembled stack is:

> a mesh-invariant, **location-typed exterior/vector-calculus surface** (Decapodes proves the surface), whose **weight generation is a pluggable mesh property** — mimetic via incidence×Hodge (PyDEC/CombinatorialSpaces) vs. moment-matched via RBF-FD/GFDM least-squares (Medusa/`rbf`) — lowering through a **structured-and-unstructured portable stencil compiler** (GT4Py.next), over a **unified field/mesh data structure** (Atlas).

The reason the gap persists is **sociological, not technical**. The DEC/applied-category-theory community (Julia, correctness- and composition-first) generates its own code and has flagged HPC solvers as future work. The EXCLAIM/GT4Py community (performance-first) built the surface bottom-up from the IR with a Fortran-shaped field-operator API, not an exterior-calculus one. The meshless-FD community are numerical analysts shipping solvers, not DSL authors. The three required competencies — categorical/mimetic surface design, performance stencil compilation, and meshless weight theory — sit in three disjoint communities. The opening is an integration and co-design problem across three skill sets, not a missing algorithm.

The lowest-friction realization is therefore **not** a greenfield DSL but a **thin exterior-calculus front-end emitting GT4Py.next field operators**, with `geofac_div`-style coefficient fields produced by a mesh-build step that may be either mimetic (what ICON already does) or moment-matched (for high-order experiments). This reuses the entire EXCLAIM backend investment and makes the structured/unstructured distinction a property of the mesh object — the one part of the design no existing tool contradicts.

---

## 6. The proposals: a mesh-invariant FD surface

Three proposals were developed. They are separable layers; the third is the implicit "glue" and is orthogonal to whether one chooses the first or second for the operators themselves. All three rest on the `gather-weight-reduce` invariant of §2 and target GT4Py.next.

### Proposal 1 — Exterior-calculus surface (conservation-first)

Fields are *k*-forms on a primal/dual complex. Primitives are `d` (exterior derivative — **pure topology**), `star` (Hodge — **pure metric**), `wedge`. Everything in vector calculus is derived.

```python
mesh = Mesh.structured(shape=(nx, ny, nz))      # or Mesh.unstructured(grid_file)
phi  = Field(mesh, form=0)        # scalar potential   (cells/vertices)
u    = Field(mesh, form=1)        # flux / circulation (edges)

grad_phi = d(phi)                 # 0-form -> 1-form
div_u    = codiff(u)              # = star(d(star(u)))
lap_phi  = codiff(d(phi))         # Hodge Laplacian
```

**Strengths.** `d` *is* the signed incidence matrix and is identical on both meshes; all geometry lives in `star`. Conservation (`d∘d=0`, telescoping fluxes) is *structural*, not an accuracy property to verify. Cleanest possible answer to "conceptually identical, expressed differently."

**Weaknesses.** Gives the mimetic / FV-flavoured operators (the C-grid/TRiSK family) only — "finite difference" in the mimetic sense (Shashkov–Lipnikov–Manzini), not arbitrary-order Taylor stencils. Cannot express 6th-order interior stencils. **This proposal largely already exists as Decapodes.jl** — the novel content collapses to pointing the DEC surface at a structured backend and a performance stencil compiler instead of a Julia SpMV path.

### Proposal 2 — Consistency-driven stencil synthesis (accuracy-first)

The genuinely "FD" proposal: weights are **synthesized**, not literal. Declare a differential operator and an order; weights are produced by moment-matching against local neighbour geometry.

```python
Dx  = Diff(dx,        order=2)      # ∂/∂x, 2nd order
Lap = Diff(laplacian, order=2)
dphi = Dx(phi);  lap = Lap(phi)
```

The mechanism: impose that the weighted neighbour sum differentiates monomials exactly up to degree *p*,

$$\sum_{k\in\text{support}} w_k\,(x_k - x_0)^{\alpha} = \partial^\alpha\big[(x-x_0)^\alpha\big]\big|_{x_0},\qquad |\alpha|\le p.$$

On a structured grid this has a constant closed-form solution — the Taylor table (recovering Devito's coefficients). On an unstructured cloud it is a small per-point least-squares solve at mesh-compile time, emitting a per-location weight field. *Runtime is then identical*; only the *provenance* of the weights differs. This is the precise sense of "Devito for both."

**Strengths.** Arbitrary order; boundaries unify (a one-sided stencil is just an asymmetric support, weights re-solve against whatever neighbours exist). **The method and solvers already exist** (RBF-FD/GFDM; Medusa, `rbf`).

**Weaknesses.** Not strictly conservative. The existing tools are solvers, not a mesh-invariant operator surface lowering to a portable stencil compiler — so the white space here is "a DSL front-end over a known method," not new numerics. Production-grade neighbour selection, polynomial conditioning, and hyperviscosity stabilization are a Medusa-sized subproject.

### Proposal 3 — Location typing over the de Rham complex (the implicit glue)

Operators carry signatures `Location -> Location`; composition is type-checked; the required connectivity is *inferred*. The user never names a connectivity or an intermediate staggered location.

```python
grad : Field[Cell]       -> Field[EdgeNormal]
div  : Field[EdgeNormal]  -> Field[Cell]
curl : Field[EdgeTangent] -> Field[Vertex]

res = div(grad(phi))      # Cell -> EdgeNormal -> Cell; connectivity inferred
```

The legal compositions are exactly the de Rham complex, which guarantees placement inference is unambiguous and terminating — Firedrake's function-space type-checking, transposed from FE spaces onto FD location types. Where two operators meet at incompatible placements, the inference inserts the staggered interpolation (itself a synthesized stencil).

**Strengths/weaknesses.** Implemented twice in disconnected communities (GT4Py.next location types at the IR level; UFL function spaces for FE), but never as a typed *exterior/vector-calculus* surface spanning structured and unstructured FD that lowers to stencils.

### Unified architecture

Three layers; the mesh is the partial-evaluation input that collapses the top into the bottom:

- **Surface** — exterior- or vector-calculus expressions, location-typed (Proposals 1 + 3).
- **Synthesis IR** — every operator rewritten to `(support, weight_field, reduction)`; weights from incidence×Hodge *or* moment-matching (Proposal 2). *Mesh-specialized.*
- **Backend** — GT4Py.next field operators: Cartesian offsets for structured, connectivity reductions for unstructured.

The crux that lets one source serve both the conservative-low-order and the high-order-FD user: **make the weight strategy a mesh-level property, not a surface-level one.**

```python
mesh = Mesh.unstructured(grid, weights="mimetic")    # incidence × Hodge → conservative, low order
mesh = Mesh.unstructured(grid, weights="gfdm:p=2")   # least-squares     → higher order, not conservative
```

Same `div(u)` in the source; the discrete realization is chosen by the mesh.

### Where the abstraction genuinely strains

These are the parts a clean exterior-calculus pitch tends to paper over:

- **Vertical column operations do not fit.** The tridiagonal implicit solve and vertical scans are not local-stencil reductions, and the vertical is structured regardless of horizontal mesh. The honest design keeps *separate* column primitives (`scan`, `solve_tridiag`) exposed orthogonally.
- **Limiters break linearity.** Flux-limited advection has solution-dependent weights (min/max over neighbours), so it is not a fixed weighted sum. The surface must expose raw neighbour-reduction primitives (`min_over`, `max_over`) as an escape hatch.
- **Weight strategies are not interchangeable.** Mimetic gives conservation; moment-matched gives order; very few schemes give both. A one-line mesh switch is ergonomically nice but invites silently trading away a conservation law — the choice probably deserves to be loud.
- **Compile-time weight solves cost something.** Per-point least-squares for high-order unstructured operators is real mesh-build work, and the resulting weight fields are extra runtime memory traffic — the flops-for-bandwidth trade made systematic.

### 6.4 Arakawa staggering and the de Rham commitment

Proposal 3 hides connectivity and placement behind location-typed operators. It is worth being explicit about what those types encode, because it is a *choice of Arakawa grid*, made by construction rather than stated as an assumption. The typing never writes "C-grid" anywhere, yet the form-degree ↔ location assignment it hardcodes —

$$\text{0-form}\to\text{CELL},\qquad \text{velocity 1-form}\to\text{EDGE\_N},\qquad \text{2-form (vorticity)}\to\text{VERTEX}$$

— *is* the Arakawa C-grid. The C-grid is the unique staggering that realizes the discrete de Rham complex: normal velocity on edges is exactly the 1-form DOF, so `grad: CELL→EDGE_N`, `div: EDGE_N→CELL`, `curl: EDGE_T→VERTEX` close into a subcomplex with `div∘curl = 0` and `curl∘grad = 0` holding *structurally*. This is precisely why the conservation and mimetic guarantees earlier were free rather than verified — they are the de Rham identities, and they exist because the staggering is C. "Location typing over de Rham" and "C-grid" are the same commitment stated two ways.

The consequence for the Arakawa family {A, B, C, D, E, Z}, grid by grid:

| Grid | Prognostic velocity DOF | de Rham fit? | Consequence for the surface |
|---|---|---|---|
| **C** | normal component on edges (1-form, primal) | native | works verbatim; guarantees structural |
| **D** | tangential component on edges (1-form, dual) | native (dual) | works with `EDGE_T` prognostic + dual connectivity — a **mesh swap** (primal↔dual), *not* a language change |
| **A** | full vector, collocated at centres | breaks | velocity is a vector 0-form at `CELL`; operators collapse to collocated centred stencils (`CELL→CELL`), skip the nearest neighbour, admit the 2Δx checkerboard mode → needs stabilization outside the de Rham algebra |
| **B** | full vector at vertices (corners) | breaks | vector 0-form on the dual; own computational modes; different signatures, no mimetic closure |
| **E** | rotated B/C hybrid | breaks | no single clean form-degree↔location assignment |
| **Z** | velocity *diagnosed* from ζ, δ via Helmholtz inversion | out of scope | the velocity 1-form is never a state variable; the core op is a **global elliptic solve**, not a local stencil |

The sharp statement of what Proposal 3 actually unifies over is therefore: **mesh topology, not staggering.** Holding the staggering fixed at C, the typing is genuinely agnostic to triangles vs. hexagons vs. quads and to structured vs. unstructured — that is the real content of the structured/unstructured unification. It does *not* range over the Arakawa family; choosing a discrete de Rham subcomplex *fixes* the staggering, and there is no single operator set that gives all Arakawa grids at once. In mimetic-FE terms this is the familiar fact that RT0+DG0 (a compatible pair) recovers the C-grid, while equal-order/collocated elements (the A-grid analogue) are an *incompatible* pair needing pressure stabilization.

Crucially, the C-grid restriction is **principled, not accidental**: the C/D branch is where the good wave dispersion and the absence of a stationary checkerboard mode live, so baking the surface into the de Rham staggering bakes it into the staggering with the desirable properties. Supporting A/B is not a small generalization of the *same* language; it is a separate, less clean instantiation with collocated operators and explicit computational-mode stabilization, forfeiting the structural conservation argument that was the main reason to type by location.

### 6.5 Generalizing the surface to arbitrary staggerings

The generalization is not "make the operators clever enough to span all grids" — no such operator set exists. It is to **factor the surface so the staggering becomes a declared, pluggable property that selects an operator family**, with only the compatible branch enjoying the free guarantees. Three changes:

**(a) Separate the abstract form-degree algebra from concrete placement.** The surface is written in form degrees ($0$-form scalar, $1$-form flux/velocity, $2$-form); a mesh-supplied **placement map** $P:\text{form-degree}\to(\text{HLoc},\text{component})$ realizes them. The staggering *is* the placement map:

```python
# staggering is a declared mesh property that selects a placement map + operator family
mesh = Mesh.unstructured(grid, staggering="C")   # P(0)=CELL, P(1)=EDGE_N, P(2)=VERTEX
mesh = Mesh.unstructured(grid, staggering="D")   # P(0)=CELL, P(1)=EDGE_T, P(2)=VERTEX (dual)
mesh = Mesh.structured(shape, staggering="A")    # P(0)=P(1)=P(2)=CELL  (collocated → degenerate)
```

The user still writes `div(grad(phi))` in form-degree terms; the placement map decides where each form lives and which connectivity the operators use. C and D differ only in the map (primal vs. dual 1-form), which is why the D-grid "just works."

**(b) Two operator families, selected by placement compatibility.**

- *Compatible staggerings (C, D)* — the placement map is injective across form degrees, so the operators are the de Rham incidence×Hodge and form an exact subcomplex. Conservation and the mimetic identities are structural. The specification of §7 applies verbatim; the mesh chooses primal (C) or dual (D).
- *Collocated / incompatible staggerings (A, B)* — the placement map is non-injective (multiple form degrees at one location). The operators degenerate into **collocated reconstructions** (Green–Gauss or least-squares gradient; FV divergence over reconstructed face values), and the mesh **must additionally supply a stabilization primitive** — a filter, a pressure/divergence stabilization, or the non-oscillatory correction that the advection scheme (e.g. MPDATA) already carries. The de Rham guarantees are *replaced* by explicit stabilization, exactly as the numerics themselves do it.

An important precision on conservation: collocated finite-volume schemes (FVM/PMAP) remain **locally mass-conserving** because conservation comes from the *flux form*, not from the staggering — the same reason Miura upwind stays conservative in §8.8. What the A-grid loses is the discrete de Rham structure (`div curl = 0`, the skew-adjoint grad/div pairing) and gains computational modes; it does **not** lose mass conservation. The generalization must state this cleanly rather than implying A-grids are "non-conservative."

**(c) Extend the escape families with two staggering-driven primitives.** The §7 escape set (`ddz`, `scan`, `solve_tridiag`, `reduce_over`) is sufficient for ICON's HEVI but not for the other cores:

```
to_stagger[S2](Field @ placement S1) -> Field @ placement S2   # inter-placement reconstruction
solve_elliptic(operator, rhs) -> Field                          # global Helmholtz/Poisson inversion
```

- `to_stagger` covers the **CD / two-grid** case (FV3/PACE), where the model advances D-grid winds but reconstructs C-grid advective winds each half-step. In the surface this is not a new grid but *two placements of velocity plus a class-(2) reconstruction between them* — which the existing `EDGE_N`/`EDGE_T` distinction already almost expresses.
- `solve_elliptic` covers the **co-located semi-implicit** case (PMAP/IFS-FVM, and the Z-grid): a 3D semi-implicit FV step needs a global elliptic (Helmholtz) solve for the pressure update, delegated to a Krylov/GCR backend. ICON's HEVI deliberately *avoids* this (only per-column tridiagonal); FVM *requires* it. So the escape-hatch set is genuinely model-dependent, and the generalized surface must expose the global elliptic solve as a first-class primitive — co-equal with `solve_tridiag`, not a bolt-on.

The honest framing of the generalization: **the surface spans the Arakawa family, but only the compatible C/D branch retains the free structural guarantees; the collocated A/B branch trades them for explicit stabilization and a global elliptic solve — which is exactly the trade the underlying numerics make.** A staggering kwarg that *transparently* retargeted across A/C/D would be a lie; a staggering declaration that *selects a whole operator family and escape-hatch set* is honest and is what the three real GT4Py cores actually need.

---

## 7. Specification of the user-written surface

This section specifies the surface language a user writes, refined to the form needed for a real dynamical core. The invariant to keep in mind: **every operator, after normalization, is `reduce(weights ⊙ gather(field, connectivity))`** — the shape of a GT4Py.next field operator. The front-end's job is to (a) type-check the composition, (b) put each operator in that normal form, and (c) let the *mesh* decide where connectivity and weights come from.

### 7.1 The type system: locations are the types

A field's type is **horizontal placement × vertical staggering**:

```
HLoc  ::= CELL | EDGE_N | EDGE_T | EDGE | VERTEX
VLoc  ::= FULL | HALF
Field[HLoc, VLoc]
```

`EDGE_N`/`EDGE_T` are the normal/tangential *components* on an edge; `EDGE` is a *scalar* at the edge midpoint. Distinguishing them lets `div` demand `EDGE_N` by type and refuse a tangential field, and makes `0.5*(vn**2 + vt**2)` type to `EDGE` (a scalar), not to a component. `VLoc` encodes the Lorenz grid: scalars at `FULL`; `w` and all interface quantities at `HALF`. Parameters (Coriolis `f`, metric factors, reference profiles) are fields with fixed location, bound at mesh-build.

### 7.2 Three operator classes + two escape families

The surface has exactly three ways to *combine* fields, plus two escape hatches for what provably does not fit a local linear-stencil algebra. Keeping these five categories distinct makes the lowering total and the strain points visible.

**(1) Linear mimetic horizontal operators** — pure de Rham edges; never touch `VLoc`; the only operators lowering to `neighbor_sum` with mesh weights.

```
grad_n : Field[CELL,   V] -> Field[EDGE_N, V]     # ∂/∂n  (d0)
div    : Field[EDGE_N, V] -> Field[CELL,   V]     # ∇·    (⋆d⋆)
curl   : Field[EDGE_T, V] -> Field[VERTEX, V]     # ∇×    (d1 on dual)
```

**(2) Placement coercions (reconstructions)** — any location change that is not one of the three operators; *also* gather-weight-reduce, but with interpolation/RBF weights rather than incidence×Hodge.

```
to[H2](Field[H1, V]) -> Field[H2, V]              # RBF / linear horizontal reconstruction
to_half(Field[H, FULL]) -> Field[H, HALF]         # interface interpolation
to_full(Field[H, HALF]) -> Field[H, FULL]
```

**(3) Pointwise maps** — arithmetic / elementwise functions between *co-located* fields; nonlinearity is fine (this is where the surface departs from pure DEC, which it must).

```
(+ - * /), pow, exp, ...   require identical [HLoc, VLoc]; result same location
```

**Escape family (4): vertical/column primitives** — orthogonal to the horizontal algebra; carry the `k`-dependency the local stencil cannot.

```
ddz   : Field[H, FULL] <-> Field[H, HALF]          # vertical difference (k±1)
scan(f_init, f_step, direction)                    # sequential k recurrence
solve_tridiag(a, b, c, rhs) -> Field[H, V]         # Thomas, per column
```

**Escape family (5): neighbour reductions** — for solution-dependent (limiter) stencils.

```
reduce_over[conn](Field, op = min | max | sum)
```

The categorical claim that makes this honest: (1) and (2) are *both* `reduce(w ⊙ gather(f, conn))` and differ only in weight provenance, so both lower to identical GT4Py.next shape. (3) is a map. (4) and (5) are explicitly *not* in the linear-stencil normal form, which is why they are named separately rather than smuggled in.

### 7.3 Typing rules

Composition of class-(1) operators is checked against the de Rham signatures; a mismatch is an error, except a *known staggering* mismatch, where the required class-(2) coercion is inserted automatically. Pointwise ops require exact co-location and otherwise raise — they never silently interpolate (an implicit reconstruction inside a product is how conservation errors hide). `ddz`/`scan`/`solve_tridiag` are the only operators allowed to change `VLoc` by recurrence. **Halo exchange is never written by the user**; the compiler inserts it after any write to a distributed field that a subsequent wider stencil reads.

### 7.4 Mesh and weight binding

The mesh is the partial-evaluation input, carrying the weight strategy *per operator class* (ICON genuinely uses different strategies for different operators):

```python
mesh = Mesh.unstructured(
    "icon_grid.nc",
    weights = {
        div:    Mimetic(),                    # geofac_div  — conservation
        grad_n: Mimetic(),
        curl:   Mimetic(),                    # geofac_rot
        to[EDGE_T]: RBF(stencil="rbf_vec"),   # v_t reconstruction
        to[EDGE_N]: RBF(),                    # ζ vertex→edge, etc.
    },
)
```

Swapping `Mesh.structured(...)` here — and nothing in the application — retargets to a Cartesian backend, with `div`/`grad_n` collapsing to offset arithmetic and RBF reconstructions becoming trivial averages.

### 7.5 Tendency declaration and the dynamics/driver boundary

The surface expresses *spatial tendencies* (RHS) and the *coefficients* of implicit systems. It does **not** express time stepping. The predictor–corrector loop, substepping, `nnow⇄nnew` swap, and tendency averaging are ordinary Python orchestration in a driver that calls compiled tendency functions — the same scope boundary Firedrake draws (UFL expresses the form; the time loop is Python).

```python
@tendency
def ddt_rho(s: State) -> Field[CELL, FULL]: ...   # returns a field; driver integrates it
```

### 7.6 Out of scope, on purpose

Not in the language: time-stepping control flow, MPI/partitioning, the *adjoint* of the halo exchange (forward-only here), and the *internal* numerics of genuinely model-specific kernels — RBF coefficient synthesis, the `igradp` hydrostatic correction, Miura back-trajectory flux, Thomas coefficient assembly. Those appear as *named primitives with mesh-supplied weights*, not as user-derived expressions. That boundary is the point: the mimetic skeleton is portable and checkable; the model-specific pieces are honestly quarantined.

---

## 8. Application to the ICON dynamical core (C-grid)

ICON (ICOsahedral Nonhydrostatic) solves the fully compressible nonhydrostatic Euler equations on a triangular Arakawa C-grid (Lorenz vertical grid), with vector-invariant horizontal momentum, a two-time-level predictor–corrector scheme, and **horizontally-explicit / vertically-implicit (HEVI)** integration — the vertical sound-wave terms are implicit (a per-column tridiagonal Thomas solve), everything else explicit. The dynamical core is substepped `ndyn_substeps` (default 5) times per model step.

The ICON dycore is close to an ideal stress test for the surface because it exercises every operator class and every escape hatch: the mimetic skeleton (flux-form `div`, the Exner gradient), reconstructions (RBF `v_t`, vertex→edge `ζ`), pointwise nonlinearity (vector-invariant Coriolis, the Exner coupling, EOS), the vertical column solve (HEVI), and limiter reductions (transport).

### 8.1 Prognostic equation set (Zängl et al. 2015)

Prognostic variables follow Gassmann & Herzog (2008):

| Symbol | Meaning | Location | Vertical |
|---|---|---|---|
| $v_n$ | horizontal velocity normal to edge | edge | full |
| $w$ | vertical velocity | cell | half |
| $\rho$ | density | cell | full |
| $\theta_v$ | virtual potential temperature | cell | full |
| $\pi$ | Exner function | cell | full |

**(D1) Horizontal momentum** (edges, full levels):
$$\frac{\partial v_n}{\partial t} + \frac{\partial K_h}{\partial n} + (\zeta + f)\,v_t + w\,\frac{\partial v_n}{\partial z} = -\,c_{pd}\,\theta_v\,\frac{\partial \pi}{\partial n} + F(v_n)$$

**(D2) Vertical momentum** (cells, half levels):
$$\frac{\partial w}{\partial t} + v_h\cdot\nabla w + w\,\frac{\partial w}{\partial z} = -\,c_{pd}\,\theta_v\,\frac{\partial \pi}{\partial z} - g$$

**(D3) Mass continuity** (cells, full levels):
$$\frac{\partial \rho}{\partial t} + \nabla\cdot(\mathbf{v}\rho) = 0$$

**(D4) Thermodynamic** (cells, full levels):
$$\frac{\partial (\rho\theta_v)}{\partial t} + \nabla\cdot(\mathbf{v}\rho\theta_v) = \tilde{Q}$$

**(D5) Exner-pressure reformulation of (D4)** for implicit vertical acoustics:
$$\frac{\partial \pi}{\partial t} + \frac{R_d}{c_{vd}}\,\frac{\pi}{\rho\theta_v}\,\nabla\cdot(\mathbf{v}\rho\theta_v) = \hat{Q}$$

**(EOS)** diagnostic closure:
$$\pi = \left(\frac{R_d\,\rho\,\theta_v}{p_0}\right)^{R_d/c_{vd}}, \qquad p = p_0\,\pi^{c_{pd}/R_d}, \qquad T = \theta_v\,\pi$$

### 8.2 State declarations

```python
vn       = Field[EDGE_N, FULL]    # D1 prognostic
w        = Field[CELL,   HALF]    # D2 prognostic
rho      = Field[CELL,   FULL]    # D3 prognostic
rhotheta = Field[CELL,   FULL]    # ρθ_v, D4 prognostic
exner    = Field[CELL,   FULL]    # π, D5 prognostic/diagnostic
f        = Field[EDGE_N, FULL]    # Coriolis parameter (mesh parameter)
```

### 8.3 The horizontally-explicit tendencies

Each term maps to exactly one operator class (annotated). **(D1)** is the hardest case for a mimetic surface — only two of its five terms are mimetic operators; the rest are reconstruction + pointwise nonlinearity:

```python
@tendency
def ddt_vn_apc(s) -> Field[EDGE_N, FULL]:
    vt   = to[EDGE_T](s.vn)                       # (2) RBF reconstruction → v_t
    ke   = 0.5*(s.vn**2 + vt**2)                  # (3) pointwise → Field[EDGE, FULL]
    Kh   = to[CELL](ke)                           # (2) edge→cell reconstruction
    zeta = curl(vt)                               # (1) mimetic → Field[VERTEX, FULL]
    zeta_e = to[EDGE_N](zeta)                      # (2) vertex→edge reconstruction
    w_e    = to[EDGE_N](to_full(s.w))              # (2) w to edges, full level

    horiz_ke = grad_n(Kh)                          # (1) ∂K_h/∂n  — mimetic grad
    coriolis = (zeta_e + s.f) * vt                 # (3) (ζ+f) v_t — pointwise nonlinear
    vert_adv = w_e * ddz(s.vn)                     # (4)+(3) w ∂v_n/∂z — column × pointwise
    return -(horiz_ke + coriolis + vert_adv)

@tendency
def ddt_vn_pgrad(s) -> Field[EDGE_N, FULL]:
    theta_v_e = to[EDGE_N](s.rhotheta / s.rho)     # (2)+(3)
    return -CPD * theta_v_e * grad_n(s.exner)      # (1) ∂π/∂n × (3) pointwise
```

**(D3) mass continuity** is the cleanest equation — pure flux-form conservation:

```python
@tendency
def ddt_rho(s) -> Field[CELL, FULL]:
    rho_e    = to[EDGE_N](s.rho)                   # (2) reconstruction (Miura in ICON)
    mass_flx = s.vn * rho_e                         # (3) edge mass flux
    horiz    = div(mass_flx)                         # (1) ∇·(vρ) — mimetic, conservative
    rho_ic   = to_half(s.rho)                       # (2) vertical interp
    vert     = ddz(s.w * rho_ic)                    # (4) vertical flux divergence
    return -(horiz + vert)
```

**(D4)** is structurally identical with `rhotheta`. **(D5)** reuses the (D4) divergence under a pointwise nonlinear coefficient — class-(3) wrapping class-(1):

```python
@tendency
def ddt_exner(s) -> Field[CELL, FULL]:
    div_flux = div(s.vn * to[EDGE_N](s.rhotheta))   # (1) same ∇·(vρθ_v)
    coeff    = (RD/CVD) * s.exner / s.rhotheta       # (3) pointwise nonlinear
    return -coeff * div_flux                          # + Qhat (physics source)
```

### 8.4 The vertically-implicit coupling (the HEVI seam)

(D2) and the acoustic parts of (D3)/(D5) are coupled implicitly in the vertical. The spec handles this by assembling tridiagonal coefficients *from vertical operators* and handing them to `solve_tridiag` — the column primitive, not the operator algebra:

```python
def vertical_implicit_update(s, dt) -> State:
    pgrad_z = CPD * to_half(s.rhotheta / s.rho) * ddz(s.exner)   # (4)+(2)+(3)
    rhs_w   = s.w + dt*(ddt_w_adv(s) - pgrad_z - G)              # explicit part of (D2)
    a, b, c = assemble_acoustic_tridiag(s, dt)   # named primitive, mesh metrics
    w_new   = solve_tridiag(a, b, c, rhs_w)      # (4) Thomas, per column
    rho_new, exner_new = backsub_acoustic(s, w_new, dt)   # (4) scan
    return s.with(w=w_new, rho=rho_new, exner=exner_new)
```

The clean statement: **everything horizontal is a class-(1) tendency; everything vertical-acoustic is a class-(4) column solve; the two meet only through shared fields.** This is exactly the ICON HEVI design, and it falls out of the type system rather than being imposed.

### 8.5 EOS closure — pure pointwise

```python
@diagnostic
def exner_from_state(s) -> Field[CELL, FULL]:
    return (RD * s.rho * (s.rhotheta/s.rho) / P0) ** (RD/CVD)   # (3) only
```

Pure pointwise nonlinearity — the case pure-DEC frameworks (Decapodes, PyDEC) have no natural home for, and the reason the surface treats pointwise nonlinearity as first-class.

### 8.6 The driver — orchestration above the language

```python
def dyn_substep(s, dt):
    apc1 = ddt_vn_apc(s)                               # predictor velocity_advection
    vn_star = s.vn + dt*(apc1 + ddt_vn_pgrad(s) + ddt_vn_phy)
    # <-- compiler inserts halo exchange on vn_star here: the edge mass flux below
    #     reads a div stencil wider than vn_star's owned region
    s1 = vertical_implicit_update(s.with(vn=vn_star), dt)

    apc2 = ddt_vn_apc(s1)                              # corrector velocity_advection
    apc_avg = 0.5*(apc1 + apc2)                        # time-centering — pure Python
    vn_new = s.vn + dt*(apc_avg + ddt_vn_pgrad(s1)
                        + divergence_damping(s1.vn) + ddt_vn_phy)   # (1)/(5) filter
    return vertical_implicit_update(s1.with(vn=vn_new), dt)

def model_step(state, dt, n=5):
    for _ in range(n):
        state = dyn_substep(state, dt/n)
    state = tracer_advection(state, mass_flx_avg)      # class (1)+(5)
    state = horizontal_diffusion(state)                # ∇⁴ via div(grad(·)) twice
    state = fast_physics(state)                        # cell-only pointwise + column
    return rediagnose_exner(state)
```

### 8.7 End-to-end lowering of (D3) to GT4Py.next

(D3) is taken fully through the pipeline. The schedule from §8.3 normalizes (post-order) into two independent chains feeding a final pointwise combine:

```
horizontal:  reconstruct[E2C] →  mul  →  reduce[C2E, +, geofac_div]
vertical:    interp_half[Koff] →  mul  →  ddz[Koff]
combine:     negate(horiz + vert)                         @ CELL, FULL
```

The mesh materializes four weight fields; **only here does structured vs. unstructured diverge.** For the unstructured mesh with `weights={div: Mimetic(), to[EDGE_N]: Linear()}`:

```
c_lin_e[e, j]         # E2C: inverse-distance cell→edge interp (½,½ on equidistant edges)
geofac_div[c, i]      # C2E: orientation[c,i] · edge_length / cell_area   (= ⋆d⋆)
wgtfac_c[c, k]        # vertical linear interp weight
inv_ddqz_z_full[c,k]  # 1/Δz at full level k
```

`geofac_div` *is* the `⋆d⋆` of the surface, materialized: the orientation sign is the incidence `d`, the `edge_length/area` ratio is the Hodge `⋆`.

**Unstructured emission** (GT4Py.next; ICON4Py-idiomatic), as small individually-testable field operators:

```python
@gtx.field_operator
def _reconstruct_rho_to_edge(rho, c_lin_e):                   # surface: to[EDGE_N](rho)
    return neighbor_sum(rho(E2C) * c_lin_e, axis=E2CDim)

@gtx.field_operator
def _horizontal_divergence(mass_flx_e, geofac_div):           # surface: div(mass_flx)
    return neighbor_sum(mass_flx_e(C2E) * geofac_div, axis=C2EDim)

@gtx.field_operator
def _rho_to_half_levels(rho, wgtfac_c):                       # surface: to_half(rho)
    return wgtfac_c * rho + (1.0 - wgtfac_c) * rho(Koff[-1])

@gtx.field_operator
def _vertical_flux_divergence(w, rho_ic, inv_ddqz_z_full):    # surface: ddz(w * rho_ic)
    z_mass_flx = w * rho_ic
    return (z_mass_flx(Koff[1]) - z_mass_flx) * inv_ddqz_z_full

@gtx.field_operator
def _ddt_rho(vn, rho, w, c_lin_e, geofac_div, wgtfac_c, inv_ddqz_z_full):
    rho_e      = _reconstruct_rho_to_edge(rho, c_lin_e)
    mass_flx_e = vn * rho_e
    horiz      = _horizontal_divergence(mass_flx_e, geofac_div)
    rho_ic     = _rho_to_half_levels(rho, wgtfac_c)
    vert       = _vertical_flux_divergence(w, rho_ic, inv_ddqz_z_full)
    return -(horiz + vert)
```

The `@program` wrapper carries the iteration domain and is where the **halo exchange auto-inserts**: `_horizontal_divergence` reads `mass_flx_e(C2E)`, so each owned cell needs its three edge values valid including the boundary ring — the compiler emits an edge-field exchange before the reduction.

**Structured emission** — same schedule, dispatched on `mesh.kind`, offsets instead of connectivity:

```python
@gtx.field_operator
def _ddt_rho_cartesian(u, v, w, rho, dxi, dyi, wgtfac_c, inv_ddqz_z_full):
    rho_x = 0.5 * (rho + rho(Ioff[1]))               # to[EDGE_N] collapsed to 2-pt avg
    rho_y = 0.5 * (rho + rho(Joff[1]))
    fx = u * rho_x;  fy = v * rho_y                  # (3) pointwise flux
    horiz = (fx - fx(Ioff[-1])) * dxi + (fy - fy(Joff[-1])) * dyi   # div as offset diffs
    rho_ic = wgtfac_c * rho + (1.0 - wgtfac_c) * rho(Koff[-1])
    z_flx  = w * rho_ic
    vert   = (z_flx(Koff[1]) - z_flx) * inv_ddqz_z_full
    return -(horiz + vert)
```

The structural identity is explicit: `neighbor_sum(mass_flx_e(C2E)*geofac_div, axis=C2EDim)` became `(fx−fx(Ioff[-1]))·dxi + (fy−fy(Joff[-1]))·dyi`. The C2E reduction with `geofac_div` weights collapsed into signed offset differences with `1/dx`; the vertical chain is *byte-identical* between backends (the vertical was structured all along). **Clinching point:** running a Cartesian mesh through the *mimetic* weight generator yields `geofac_div = {+1/dx, −1/dx, +1/dy, −1/dy}` (orientation × face-length / cell-area on a unit quad) — the structured emission is the unstructured one with the weights constant-folded. That is the whole thesis, concrete on the conservation equation.

**Verification.** Because the emitted operators are small, each diffs against a known-good ICON4Py stencil: `_horizontal_divergence` should be token-identical to ICON4Py's divergence stencil given the same `geofac_div`; `_reconstruct_rho_to_edge` matches `cell_2_edge_interpolation` with `c_lin_e`; `_rho_to_half_levels` matches the `wgtfac_c` interface interpolation. Validation is *structural equality* against verified stencils, not a numerical-tolerance argument. Conservation is then a separate structural check: `neighbor_sum` over cells of `cell_area · horiz` telescopes over shared edges (each interior edge appears in two cells with opposite orientation), summing to boundary fluxes ≈ 0 — to machine precision, by construction of `geofac_div`, independent of resolution.

### 8.8 The Miura reconstruction swap

ICON does **not** use the centred `c_lin_e` for `rho_e` — it uses Miura second-order upwind. The thesis: **only the reconstruction slot changes; the divergence operator is byte-identical.** The clean chain `reconstruct[E2C, c_lin_e] → mul → reduce[C2E]` becomes:

```
lsq_gradient[C2E2C, lsq_pinv]  ┐
btraj(vn, vt, dt) → distv      ├→ upwind_eval[E2C, where(vn>0)] → mul → reduce[C2E, geofac_div]
                               ┘                                          └──── UNCHANGED ────┘
```

Surface call, unchanged shape, heavier strategy:

```python
rho_e = to[EDGE_N](s.rho, strategy=Miura(order=2, dt=dt_sub))
```

Three new sub-operators appear. The **least-squares cell gradient** lowers cleanly (a `C2E2C` reduction with pseudo-inverse weights — ICON4Py's `recon_lsq_cell_l` family):

```python
@gtx.field_operator
def _lsq_cell_gradient(rho, lsq_pseudoinv_1, lsq_pseudoinv_2):
    drho = rho(C2E2C) - rho
    z_grad_1 = neighbor_sum(lsq_pseudoinv_1 * drho, axis=C2E2CDim)
    z_grad_2 = neighbor_sum(lsq_pseudoinv_2 * drho, axis=C2E2CDim)
    return z_grad_1, z_grad_2
```

The **back-trajectory** is dynamic (recomputed each substep from `vn, vt, dt`; the departure barycenter is `x_e − ½Δt·v`), so part of the reconstruction's weight is now *state*, not static mesh data. The **upwind barycenter evaluation** is the term that genuinely leaves the fixed-weight normal form — both candidate cells are evaluated, and `where` selects on the sign of the normal mass flux:

```python
@gtx.field_operator
def _reconstruct_rho_miura(rho, z_grad_1, z_grad_2,
                           distv_0_x, distv_0_y, distv_1_x, distv_1_y, vn):
    rho_e0 = rho(E2C[0]) + distv_0_x * z_grad_1(E2C[0]) + distv_0_y * z_grad_2(E2C[0])
    rho_e1 = rho(E2C[1]) + distv_1_x * z_grad_1(E2C[1]) + distv_1_y * z_grad_2(E2C[1])
    return where(vn > 0.0, rho_e0, rho_e1)
```

The composing operator imports `_horizontal_divergence`, `_rho_to_half_levels`, `_vertical_flux_divergence` **unchanged**. The diff against the centred version is exactly three new field operators plus a wider input list, and *nothing downstream of `mass_flx_e`*. That locality is the payoff of putting reconstruction behind `to[EDGE_N]`.

**What changed, and the key property distinction:**

- **Weight provenance split** — `c_lin_e` (Hodge-flavoured) gives way to least-squares pseudo-inverse weights. Mimetic and moment-matched weight theories now coexist *in one equation, on adjacent operators* — the "pluggable weight strategy" claim cashing out per-operator.
- **Static → dynamic weights** — the btraj distances depend on `vn, vt, dt`, recomputed every substep.
- **Fixed gather → data-dependent branch** — the `where(vn>0, …)` is the formal exit from `reduce(w ⊙ gather)`; the documented escape, used.
- **Wider halo** — the lsq gradient reads `C2E2C`, so the compiler provisions a deeper exchange (second-ring `rho`) than the centred first-ring; the same auto-insertion mechanism, widened because the stencil widened.
- **Conservation kept, skew-adjointness lost** — the subtle and important one. **Mass conservation survives untouched**: it comes from the *flux form* (the `div` telescopes regardless of how `rho_e` was built), so Miura is still conservative to machine precision. What is lost is the mimetic *skew-adjointness* — the centred reconstruction is (up to mass weighting) the transpose pairing with `div` to make advection energy-neutral (`grad = −divᵀ`); upwind deliberately breaks that to add implicit diffusion for stability. The swap trades the energy-conserving/adjoint-clean property for monotonicity-friendliness, and does **not** trade away mass conservation. The flux form is precisely what protects the latter.

### 8.9 Honest accounting: clean algebra vs. quarantined primitives

Expressed as clean surface algebra (portable, checkable against the mathematics): the flux-form divergences of (D3)/(D4), the Exner coupling (D5), both pressure gradients, the vorticity `curl`, hyperdiffusion as `div(grad(·))²`, the EOS, and the entire pointwise structure of the momentum equation. These are where structured-vs-unstructured genuinely vanishes into the mesh argument.

Invoked as named primitives with mesh-supplied weights (model-specific, *not* derived in the surface): RBF tangential reconstruction `to[EDGE_T]`; vertex→edge vorticity; the `igradp_method` hydrostatic correction (not a clean `grad_n` over steep terrain — a custom edge stencil supplied as the `grad_n` weight strategy in the orography region); Miura back-trajectory flux; the acoustic tridiagonal assembly and Thomas solve; PPM vertical transport and its monotone limiters (class-5 `reduce_over` plus a column reconstruction).

**The biggest caveat to keep in view:** the implicit vertical solve is not peripheral — it runs `2n` times per step and couples three prognostic equations. A surface that only elegantly expressed the explicit horizontal operators would address perhaps half of where the dycore spends its complexity. The specification treats `solve_tridiag`/`scan` as first-class for that reason; the column solver is **co-equal** with the mimetic algebra here, not subordinate to it. Likewise, (D3) is the *most favourable* equation for the abstraction: the divergence and vertical term lower beautifully, while the *reconstruction* (Miura) is where ICON's real-world numerics start pushing on the clean picture.

---

## 9. Beyond ICON: staggering-parametric application to PACE (D-grid) and PMAP (A-grid)

ICON exercises the *compatible* branch of §6.5 in its purest form. The two other production GT4Py cores land on different points of the Arakawa spectrum and therefore exercise the generalization's other branches. Both are real GT4Py users (Dahm et al. 2023 for PACE; Ubbiali et al. 2025 for PMAP), so this is not a hypothetical portability claim.

### 9.1 The three cores, side by side

| Model (GT4Py port) | Numerics | Grid topology | Arakawa staggering | §6.5 branch | Distinctive primitive needed |
|---|---|---|---|---|---|
| **ICON** (icon-exclaim) | mimetic FD/FV, vector-invariant, HEVI | triangular icosahedral (unstructured) | **C** | compatible (primal) | `solve_tridiag` (per-column) |
| **PACE** (Pace / FV3GFS, SHiELD) | finite-volume, Lin–Rood, vertically-Lagrangian | cubed-sphere (structured, 6 tiles) | **D** (+ C advective winds) | compatible (dual) + two-grid | `to_stagger` (D↔C each half-step) |
| **PMAP** (IFS-FVM) | finite-volume, MPDATA, 3D semi-implicit | octahedral / quad (structured **and** unstructured) | **A** (co-located) | collocated + stabilized | `solve_elliptic` (global Helmholtz) + stabilization |

### 9.2 PACE / FV3 — the D-grid and the two-grid step

FV3 is the compatible branch on the *dual*: its prognostic winds are the **D-grid** tangential components, and it advances them with a vertically-Lagrangian finite-volume scheme (Lin 2004; Putman & Lin 2007). The surface change from ICON is a placement swap, not a language change:

```python
mesh = Mesh.structured(cubed_sphere, staggering="D")   # P(1) = EDGE_T
u_d  = Field[EDGE_T, FULL]                              # prognostic D-grid wind
div_d = div(u_d)                                        # same operator, dual connectivity
```

`div`, `grad`, `curl` retain their signatures and their structural guarantees — the placement map routes them through dual connectivity. What FV3 adds is the **two-grid (CD) step**: each acoustic sub-step reconstructs C-grid advective winds from the D-grid state, computes fluxes on the C-grid, then updates the D-grid winds. That is exactly the `to_stagger` escape:

```python
u_c = to_stagger[EDGE_N](u_d)          # (2) D→C reconstruction of advective winds
mass_flx = u_c * to[EDGE_N](rho)        # fluxes built on the C-grid placement
ddt_rho  = -div(mass_flx)               # divergence closes on cells — conservative
# ... update u_d from the C-grid fluxes (vorticity-flux form)
```

The point: the CD structure is not a new grid type in the surface — it is *two placements of one velocity field plus a reconstruction between them*, which the `EDGE_N`/`EDGE_T` distinction already almost expresses. FV3's mass conservation is the same flux-form argument as ICON's (the `div` telescopes); the vertically-Lagrangian vertical coordinate is an additional per-column remap that slots in as a `scan`-class primitive (a Lagrangian-to-Eulerian reconstruction), orthogonal to the horizontal algebra just like ICON's tridiagonal solve.

The honest strain for PACE is different from ICON's: FV3's stencils are wide (PPM reconstruction, corner transport upwind) and its cubed-sphere edges/corners carry special-cased metric terms, so a large fraction of the operators are *reconstructions* rather than mimetic `div`/`grad`. The Pace port already reports that the FV3 core "suits the two stencil classes GT4Py targets well, but not exactly" — the surface would inherit precisely that: the conservation skeleton stays clean, the reconstructions and the panel-edge handling are the quarantined, mesh-supplied part.

### 9.3 PMAP / IFS-FVM — the collocated branch

PMAP is the port of ECMWF's IFS-FVM: a non-hydrostatic, locally-conserving, **finite-volume** core with **co-located (A-grid)** variables on the median-dual of the octahedral grid, MPDATA non-oscillatory advection, and a **3D semi-implicit** time integration (Smolarkiewicz et al. 2016; Kühnlein et al. 2019). It is the collocated branch of §6.5, and it is the case that most stresses the abstraction — deliberately, because it is where the de Rham guarantees do *not* apply.

```python
mesh = Mesh.unstructured(octahedral, staggering="A", stabilization=MPDATA())
u = Field[CELL, FULL]   # full velocity vector, collocated at nodes (0-form)
```

Three consequences fall directly out of the generalization:

1. **Operators degenerate to collocated reconstructions.** `grad(phi)` is no longer an edge difference but a node-centred Green–Gauss / least-squares gradient over the neighbour ring; `div(u)` is a node-centred FV divergence over reconstructed face values. These are class-(2)/class-(1) *reconstructions*, not the mimetic incidence×Hodge, and the surface must select them from the `staggering="A"` declaration.
2. **Conservation is retained, de Rham structure is not.** FVM is locally conserving via the flux form — so the surface's conservation check still holds — but `div∘curl = 0` and the skew-adjoint pairing do not, and the checkerboard mode is controlled by MPDATA's non-oscillatory correction, supplied as the mesh `stabilization` primitive. Stating this precisely matters: PMAP is conservative *and* collocated; the two are not in tension.
3. **The elliptic solve is first-class.** The 3D semi-implicit step forms a global Helmholtz problem for the pressure update, solved by a preconditioned GCR/Krylov iteration. Where ICON's HEVI reduces the implicit coupling to independent per-column tridiagonals (`solve_tridiag`), PMAP needs `solve_elliptic` over the full 3D domain — a global, communication-heavy operation the surface delegates to a Krylov backend but must expose as a primitive:

```python
# semi-implicit pressure update (schematic)
rhs   = assemble_helmholtz_rhs(state, dt)     # from divergence of provisional momentum
dpi   = solve_elliptic(helmholtz_operator, rhs)   # global GCR — the FVM analogue of Thomas
u     = u - dt * grad(dpi)                     # collocated gradient correction
```

This is the sharpest illustration of why the escape-hatch set must be *staggering-dependent*: the same document that treats `solve_tridiag` as co-equal with the mimetic algebra for ICON must treat `solve_elliptic` as co-equal for PMAP. A surface that only offered the column solve would express ICON's implicit coupling and be unable to express PMAP's at all.

### 9.4 What the three-core view establishes

The generalization is validated less by elegance than by the fact that three independently-developed cores, already sharing GT4Py as a backend, populate exactly the three branches the design predicts: compatible-primal (ICON/C), compatible-dual-plus-two-grid (PACE/D+C), and collocated-stabilized-plus-elliptic (PMAP/A). The mimetic conservation skeleton — flux-form `div`, and mass conservation as a telescoping-flux property — is common to all three and stays clean in the surface. The staggering declaration is what selects placement, operator family, stabilization, and the implicit-solve primitive; it is not a cosmetic kwarg but the switch that picks which of three genuinely different discrete worlds the same form-degree source compiles into. That the switch is *loud* — that A-grid forfeits de Rham structure and demands `solve_elliptic`, visibly — is a feature: it keeps the surface from promising a portability it cannot honestly deliver.

---

## 10. Synthesis and open questions

The argument of this document, compressed: the computational patterns of atmospheric dynamical cores reduce to a single `gather-weight-reduce` normal form whose only mesh-dependence is the provenance of support and weights; that normal form is the shape of a GT4Py.next field operator; therefore a mesh-invariant, location-typed exterior/vector-calculus surface can lower to GT4Py.next with the structured/unstructured distinction confined to a mesh-build weight-generation step. The pieces to build this all exist — Decapodes (surface), PyDEC/CombinatorialSpaces and Medusa/`rbf` (the two weight strategies), GT4Py.next (the portable backend), Atlas (the data structure) — but in disconnected communities; the contribution would be integration, realized most cheaply as a thin front-end over GT4Py.next rather than a greenfield DSL.

Applied to ICON, the surface expresses the mimetic conservation skeleton and the thermodynamics cleanly and portably, while honestly quarantining the model-specific numerics (RBF reconstructions, terrain corrections, Miura flux, the column solver) as named primitives. The clean/quarantined boundary becomes *explicit in the code* rather than tangled — which is the practical value even before any performance portability is realized.

The design has two orthogonal unification axes, and it is worth keeping them distinct. The first — **mesh topology** (structured ↔ unstructured) — is genuinely transparent: the same source compiles to Cartesian offsets or connectivity reductions with the difference confined to weight provenance. The second — **grid staggering** (A/B/C/D/Z) — is *not* transparent and should not pretend to be: the location typing of Proposal 3 is a C-grid commitment, its dual gives D, and the collocated grids (A/B) and the vorticity-divergence grid (Z) require a different operator family, a stabilization primitive, and in the semi-implicit case a global elliptic solve. §6.5 makes staggering a declared property that *selects* an operator family and escape-hatch set rather than a kwarg that silently retargets, and §9 shows the three production GT4Py cores — ICON (C), PACE (D+C), PMAP (A) — populating exactly the three branches this predicts. The common core across all of them is the flux-form conservation skeleton; that is the invariant worth building the surface around.

Open questions worth flagging:

- **Weight-strategy pluggability vs. correctness.** Making mimetic↔moment-matched a mesh kwarg is ergonomic but lets a user silently trade away conservation. The split should probably be loud, and conservation-relevant operators flagged.
- **The emitter's real integration point.** Emitting field-operator *source* is the legible representation; the production path (generated source vs. building GTIR/ITIR via the builder API) determines whether domain inference and halo-insertion come for free — that is the part of the design sketched rather than pinned.
- **Production-grade moment-matched weights.** Neighbour selection, polynomial conditioning, and hyperviscosity stabilization for high-order unstructured operators are a Medusa-sized subproject, not the dozen illustrative lines.
- **Differentiable halo exchange.** Out of scope here (forward-only), but the adjoint of a halo exchange is its *transpose* (cotangents scatter-add onto owning ranks), not the same exchange run backward — a known trap for hand-rolled wrappers and a prerequisite for any AD-through-dycore ambition.
- **Staggering as a family selector, not a kwarg.** The clean part of the generalization is the compatible C/D branch (a placement-map swap). The open engineering question is the collocated branch: how much of the A-grid operator family (Green–Gauss/least-squares gradients, FV divergence over reconstructed faces) and its stabilization can be shared machinery versus per-model, and whether `solve_elliptic` can present a backend-agnostic interface (GCR, multigrid, FFT-on-structured) the way `solve_tridiag` does for the column case. PACE's two-grid `to_stagger` is the easiest non-ICON target; PMAP's global elliptic solve is the hardest and the one that most tests whether "co-equal escape hatch" is real or aspirational.

A natural first milestone: implement the surface, typing, normalization, and the mimetic backend over the existing ICON connectivities, and diff the emitted (D3) field operators against the hand-written ICON4Py stencils for structural equality. That is checkable against ground truth, and validates the central claim before any moment-matched or structured-backend work is layered on.

---

## 11. References

### Domain, equations, and the ICON dynamical core

- Zängl, G., Reinert, D., Rípodas, P., Baldauf, M. (2015). *The ICON (ICOsahedral Nonhydrostatic) modelling framework of DWD and MPI-M: Description of the non-hydrostatic dynamical core.* Q. J. R. Meteorol. Soc. 141, 563–579. doi:10.1002/qj.2378.
- Gassmann, A., Herzog, H.-J. (2008). *Towards a consistent numerical compressible non-hydrostatic model using generalized Hamiltonian tools.* Q. J. R. Meteorol. Soc. 134, 1597–1613.
- Zängl, G. (2012). *Extending the numerical stability limit of terrain-following coordinate models over steep slopes.* Mon. Weather Rev. 140, 3722–3733.
- Wan, H., Giorgetta, M. A., Zängl, G., et al. (2013). *The ICON-1.2 hydrostatic atmospheric dynamical core on triangular grids – Part 1.* Geosci. Model Dev. 6, 735–763.
- Miura, H. (2007). *An upwind-biased conservative advection scheme for spherical hexagonal–pentagonal grids.* Mon. Weather Rev. 135, 4038–4044.
- Lauritzen, P. H., et al. (2013). *A standard test case suite for two-dimensional linear transport on the sphere.* Geosci. Model Dev. 6.
- Zängl, G., Reinert, D., Prill, F. (2022). *Grid refinement in ICON v2.6.4.* Geosci. Model Dev. 15, 7153–7176.
- Giorgetta, M. A., et al. (2018). *ICON-A, the atmosphere component of the ICON Earth system model: I. Model description.* J. Adv. Model. Earth Syst. 10. doi:10.1029/2017MS001242.
- Borchert, S., et al. (2019). *The upper-atmosphere extension of the ICON general circulation model (ua-icon-1.0).* Geosci. Model Dev. 12, 3541–3569.
- Dipankar, A., et al. (2015). *Large eddy simulation using the general circulation model ICON.* J. Adv. Model. Earth Syst. 7, 963–986.
- Giorgetta, M. A., et al. (2022). *The ICON-A model for direct QBO simulations on GPUs (icon-cscs).* Geosci. Model Dev. 15, 6985–7016.
- Ullrich, P. A., et al. (2017). *DCMIP2016: a review of non-hydrostatic dynamical core design and intercomparison.* Geosci. Model Dev.
- Prill, F., Reinert, D., Rieger, D., Zängl, G. *ICON Tutorial — Working with the ICON Model* (DWD).
- ICON4Py numerical documentation (C2SM). https://c2sm.github.io/icon4py/
- ICON model documentation (MPI-M / DWD). https://code.mpimet.mpg.de/projects/iconpublic/wiki/Documentation

### Mimetic / compatible discretization, TRiSK, DEC, FEEC

- Ringler, T. D., Thuburn, J., Klemp, J. B., Skamarock, W. C. (2010). *A unified approach to energy conservation and potential vorticity dynamics for arbitrarily-structured C-grids.* J. Comput. Phys. 229, 3065–3090. (TRiSK; MPAS.)
- Cotter, C. J., Shipton, J. (2012). *Mixed finite elements for numerical weather prediction.* J. Comput. Phys. 231, 7076–7091. (TRiSK as a mixed FE method.)
- Thuburn, J., Cotter, C. J. (2012). *A framework for mimetic discretization of the rotating shallow-water equations on arbitrary polygonal grids.* SIAM J. Sci. Comput. 34, B203–B225.
- Arnold, D. N., Falk, R. S., Winther, R. (2006). *Finite element exterior calculus, homological techniques, and applications.* Acta Numerica 15, 1–155.
- Bochev, P. B., Hyman, J. M. (2006). *Principles of mimetic discretizations of differential operators.* In *Compatible Spatial Discretizations*, IMA Vol. 142, Springer, 89–119.
- Bell, N., Hirani, A. N. (2012). *PyDEC: Software and Algorithms for Discretization of Exterior Calculus.* ACM Trans. Math. Softw. 39(1), 3:1–3:41. doi:10.1145/2382585.2382588. arXiv:1103.3076. https://github.com/hirani/pydec

### Meshless / RBF-FD / generalized finite differences

- Slak, J., Kosec, G. (2021). *Medusa: A C++ Library for Solving PDEs Using Strong-Form Mesh-Free Methods.* ACM Trans. Math. Softw. 47(3). doi:10.1145/3450966.
- `rbf` — Python package for RBF and RBF-FD. https://rbf.readthedocs.io/
- Shankar, V. (2017). *The overlapped radial basis function-finite difference (RBF-FD) method: A generalization of RBF-FD.* J. Comput. Phys. 342, 211–228.
- Shankar, V., Fogelson, A. L. (2018). *Hyperviscosity-based stabilization for RBF-FD discretizations of advection–diffusion equations.* J. Comput. Phys. 372, 616–639.
- Benito, J. J., Ureña, F., Gavete, L. (2001). *Influence of several factors in the generalized finite difference method.* Appl. Math. Modelling 25, 1039–1053.

### DSLs, compilers, and software frameworks

- Deconinck, W., Bauer, P., Diamantakis, M., et al. (2017). *Atlas: A library for numerical weather prediction and climate modelling.* Comput. Phys. Commun. 220, 188–204. doi:10.1016/j.cpc.2017.07.006. https://github.com/ecmwf/atlas
- GridTools / GT4Py. https://github.com/GridTools/gt4py
- Dipankar, A., et al. (2026). *Toward exascale climate modelling: a python DSL approach to ICON's (icosahedral non-hydrostatic) dynamical core (icon-exclaim v0.2.0).* Geosci. Model Dev. 19, 713–729. doi:10.5194/gmd-19-713-2026. https://gmd.copernicus.org/articles/19/713/2026/
- Dahm, J. P. S., et al. (2023). *Pace v0.1: A Python-based Performance-Portable Implementation of the FV3 Dynamical Core.* Geosci. Model Dev. https://egusphere.copernicus.org/preprints/2022/egusphere-2022-943/
- Ubbiali, S., Kühnlein, C., Krieger, N., Papritz, L., Vollenweider, G., et al. (2025). *PMAP — the Portable Model for multi-scale Atmospheric Prediction* (Python/GT4Py port of IFS-FVM; ECMWF / ETH Zürich / CSCS). Model-development page: https://iac.ethz.ch/group/atmospheric-dynamics/focus/model-development.html

### Other dynamical cores and numerical methods (PACE / FV3, PMAP / IFS-FVM)

- Lin, S.-J. (2004). *A "vertically Lagrangian" finite-volume dynamical core for global models.* Mon. Weather Rev. 132, 2293–2307. (FV3; D-grid, vertically-Lagrangian.)
- Putman, W. M., Lin, S.-J. (2007). *Finite-volume transport on various cubed-sphere grids.* J. Comput. Phys. 227, 55–78.
- Harris, L. M., Lin, S.-J. (2013). *A two-way nested global-regional dynamical core on the cubed-sphere grid.* Mon. Weather Rev. 141, 283–306. (FV3 C-grid/D-grid two-grid step.)
- Colella, P., Woodward, P. R. (1984). *The Piecewise Parabolic Method (PPM) for gas-dynamical simulations.* J. Comput. Phys. 54, 174–201.
- Smolarkiewicz, P. K., Deconinck, W., Hamrud, M., Kühnlein, C., et al. (2016). *A finite-volume module for simulating global all-scale atmospheric flows.* J. Comput. Phys. 314, 287–304. (IFS-FVM; co-located A-grid, MPDATA.)
- Kühnlein, C., Deconinck, W., Klein, R., Malardel, S., et al. (2019). *FVM 1.0: A nonhydrostatic finite-volume dynamical core for the IFS.* Geosci. Model Dev. 12, 651–676. doi:10.5194/gmd-12-651-2019.
- Smolarkiewicz, P. K., Margolin, L. G. (1998). *MPDATA: A finite-difference solver for geophysical flows.* J. Comput. Phys. 140, 459–480. (Non-oscillatory advection / stabilization.)
- Ben-Nun, T., et al. (2022). *Productive Performance Engineering for Weather and Climate Modeling with Python.* arXiv:2205.04148.
- Gysi, T., et al. (2021). *Domain-Specific Multi-Level IR Rewriting for GPU: The Open Earth Compiler.* ACM TACO. doi:10.1145/3469030. arXiv:2005.13014. https://github.com/spcl/open-earth-compiler
- *A shared compilation stack for distributed-memory parallelism in stencil DSLs.* arXiv:2404.02218 (2024).
- *An MLIR Lowering Pipeline for Stencils at Wafer-Scale.* arXiv:2601.17754. (xDSL stencil dialect.)
- Clement, V., et al. *The CLAW DSL: Abstractions for Performance Portable Weather and Climate Models.* (PASC.)
- Kalman, ... Bonaventura, L., et al. (2024). [STELLA / GridTools history and DSL discussion.] Comput. Phys. Commun. 295, 108993.
- Müller, A., Jumah, N., et al. *GGDML / AIMES* — general grid definition and manipulation language (Fortran extension). arXiv:1710.08616 (related Hybrid Fortran context).
- Rathgeber, F., et al. (2016). *Firedrake: automating the finite element method by composing abstractions.* ACM Trans. Math. Softw. 43(3), 1–27. doi:10.1145/2998441.
- Alnæs, M. S., Logg, A., Ølgaard, K. B., Rognes, M. E., Wells, G. N. *Unified Form Language (UFL): a domain-specific language for weak formulations of partial differential equations.* ACM Trans. Math. Softw.
- Heisler, E., et al. (2022). *Finch: Domain Specific Language and Code Generation for Finite Element and Finite Volume in Julia.* ICCS 2022. doi:10.1007/978-3-031-08751-6_9. Extended: *Multi-discretization DSL and code generation for differential equations*, J. Comput. Sci. (2023). https://github.com/paralab/Finch
- Morris, L., et al. (2024). *Decapodes: A Diagrammatic Tool for Representing, Composing, and Computing Spatialized Partial Differential Equations.* arXiv:2401.17432. https://github.com/AlgebraicJulia/Decapodes.jl
- Morris, L., Rauta, G., Carlson, K., Fairbanks, J. (2025). *Porous Convection in the Discrete Exterior Calculus with Geometric Multigrid.* arXiv:2508.12501. (CombinatorialSpaces.jl multigrid.)
- LFRic / PSyclone (Met Office). *PSyKAl separation for performance-portable NWP.*
- *Domain-specific implementation of high-order Discontinuous Galerkin methods in spherical geometry* (GT4Py frontend extension). arXiv:2303.11767.

### Standard texts

- Arakawa, A., Lamb, V. R. (1977). *Computational design of the basic dynamical processes of the UCLA general circulation model.* Methods Comput. Phys. 17, 173–265. (C-grid staggering.)
- Durran, D. R. (2010). *Numerical Methods for Fluid Dynamics, with Applications to Geophysics*, 2nd ed. Springer.
- Shashkov, M. (1996). *Conservative Finite-Difference Methods on General Grids.* CRC Press. (Mimetic FD.)

---

*Document compiled from a design conversation. Code is illustrative pseudocode / sketch-level GT4Py.next, intended to convey structure rather than to compile unmodified; API spellings (e.g. `Field[[...], wpfloat]`, offset providers) vary by GT4Py.next version. Equation signs and index conventions for the vertical operators follow a self-consistent choice and should be pinned against `mo_solve_nonhydro` before use. The staggering characterizations of the three GT4Py cores (ICON = C-grid; PACE/FV3 = D-grid with C-grid advective winds; PMAP/IFS-FVM = co-located A-grid) are drawn from the model literature cited in §11; the surface-level treatment of their operator families and escape hatches (§6.5, §9) is a design proposal, not a description of existing code — no such staggering-parametric front-end exists today (§5).*
