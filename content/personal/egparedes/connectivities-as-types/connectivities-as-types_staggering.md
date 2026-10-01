---
title: "Connectivities as types — Cartesian axes, the CW-complex reading and alignment"
author: egparedes
tags: [type-system, dimensions, staggering, cw-complex, exterior-calculus, domain, dimension-kind, type-checking]
created: 2026-10-01
status: draft
---

> **Appendix** to [[personal/egparedes/connectivities-as-types/connectivities-as-types|Connectivities as types]].
> Background and full derivations for the staggering half of the design: why a
> Cartesian axis has exactly two cell classes, what the product complex buys over a
> form degree, why the alignment convention is the user's to pick, why extents are
> declared rather than derived, and the alternatives and follow-ups that follow from
> all of it. External corroboration from the UGRID and SGRID conventions is in the
> [[personal/egparedes/connectivities-as-types/connectivities-as-types_conventions|conventions appendix]].

## 1. A Cartesian axis is two cell classes

A **Cartesian axis** is an index space with integer index arithmetic and
exactly one staggered partner — the dimensions `CartesianConnectivity` acts on.
Read one axis of a Cartesian grid as a
1-dimensional CW complex — a path when the axis is bounded, a cycle when it is
periodic. Such a complex has exactly two cell classes, of degree 0 and 1, and a
declared axis together with its `Staggered[·]` name those two classes.
`Staggered` is the **involution that swaps them**. The types do not say which
class has degree 0, and they need not: the ADR 0026 position convention
(`Staggered[X](i)` at `i - 1/2`) fixes only their *relative* alignment. Both
assignments occur. A vertex-indexed structured grid declares the 0-cells, so
`Staggered[I](i)` is the 1-cell `[I(i-1), I(i)]`; ICON's Lorenz grid declares
the 1-cells, `KDim` being the full levels (layer midpoints) while
`Staggered[KDim]` are the half levels (interfaces, the 0-cells).

In more than one dimension the grid is the **product** complex of its axes, so
a cell class is a bit vector over axes — one bit per axis, "does this cell span
it?" — and the cell degree is the number of set bits. `Dims[...]` *is* that bit
vector:

| `Dims[...]` | cell | degree |
| --- | --- | --- |
| `Dims[I, J]` | vertex | 0 |
| `Dims[Staggered[I], J]` | edge along `I` | 1 |
| `Dims[I, Staggered[J]]` | edge along `J` | 1 |
| `Dims[Staggered[I], Staggered[J]]` | face | 2 |

The `cell` and `degree` columns assume the first assignment above — every
declared axis indexes its 0-cells. Under the other assignment the column reads
upside down on that axis; the *encoding* is unaffected, which is the point.

Per-axis staggering is therefore *more* informative than a form degree, which
in two dimensions conflates the two edge families a C-grid must keep apart —
the distinction that has to be recovered by hand, as a component label, wherever
no product structure exists (`EDGE_N` vs `EDGE_T` in
[[personal/egparedes/discretization-independent-fd-syntax|the surface-syntax note]]
§7.1). It is also why the structured and unstructured cases are encoded
differently here at all: a per-axis bit exists exactly when the complex factors
as a product, and mesh locations (`V`, `E`, `C`) are separate dimension classes
because it does not.

Two consequences shape the hierarchy:

- **A staggered dimension is itself a discretized axis.** It is the other cell
  class of the same complex: it carries fields, has a range, and takes integer
  offsets. So the asymmetry between `D` and `Staggered[D]` is not semantic but
  *declarational* — one is written by the user, the other derived from it, and
  there is exactly one partner to derive. That is what makes the derivation
  well-founded, and it is the level `Staggered`'s parameter must name.
- **`Staggered[D]` needs no orientation data.** A 1-cell in a complex is
  oriented, which is what makes the coboundary signed. A product of intervals
  has a canonical orientation per axis (increasing index), so the sign is
  implicit in the stencil the user writes. The unstructured side has no such
  canonical choice, which is why incidence signs are materialized there (ICON's
  `geofac_div` carries them). The absence is a consequence of the product
  structure, not an omission.

## 2. Alignment

**The alignment is antisymmetric, so the convention is the user's to pick.** The
ADR 0026 convention places `Staggered[D](i)` at `i - 1/2` in `D`'s coordinates;
equivalently, it places `D(j)` at `j + 1/2` in `Staggered[D]`'s. Both conventions
appear in production codes — the SGRID convention's four `padding` values
enumerate both, `low` and `both` placing a cell at `j - 1/2` while `none` and
`high` place it at `j + 1/2`
([[personal/egparedes/connectivities-as-types/connectivities-as-types_conventions|conventions appendix]]
§3) — and either is reachable by choosing *which member of the pair to
declare*. The default puts the partner half a cell *below*; to have it half a cell
*above* your main field, declare the axis the **partner** lives on and alias your
main field onto the derived member. With `class IFace(CartesianAxisIndex)` and
`ICell: TypeAlias = Staggered[IFace]`, an `IFace` field sits half a cell above the
`ICell` field of the same index. The pair is symmetric in the implementation —
`order_dimensions` orders by the base either way, `check_dims` is symmetric,
fingerprints go through the base, gtfn aliases the staggered tag to its base — so
the only residual cost is cosmetic: the derived member's *tag* reads
`...Staggered[IFace]` in the IR, in codegen names and in error messages. Stencil
code built from relative shifts is convention-independent in any case
(`p(S + 1/2) - p(S - 1/2)` is the same difference either way, and `±1/2` reaches
both neighbours); the convention bites only on absolute index correspondence with
a reference implementation and on boundary index ranges.

## 3. Extents

**Extents are declared, never derived.** A path complex has `n` 0-cells and
`n - 1` 1-cells; a cycle has `n` and `n`. So the two members of a pair do not in
general have the same range, and for a bounded, halo-free complex *which one is
wider is the degree assignment* (periodicity or halo padding destroys that
correspondence — see §7):
with `I` the 0-cells on `[0, n)`, the interior 1-cells are `[I(i-1), I(i)]` for
`i` in `[1, n)`, and `Staggered[I](0)` is the first cell outside the complex —
exactly where a halo cell goes; with `K` the 1-cells on `[0, n)` (ICON's layers),
the interfaces run `[0, n+1)`. Both are expressible without any new type-level
concept because a `Domain` holds absolute `UnitRange(start, stop)`s rather than
sizes — see
[[personal/havogt/field-data-protocol/field-data-protocol|the field data protocol]]
for why absolute addressing is the right domain model — and the periodic case
needs nothing extra, the two ranges simply coincide in size. The invariant a grid
builder owes is that the two ranges *interleave*; their sizes then differ by at
most one when the complex is closed, and halo extension relaxes even that (two of
SGRID's four bounded cases, `low` and `high`, have equal sizes). An absolute range
is strictly more expressive than SGRID's `padding` enum here: each of its four
values is one `UnitRange` under a single alignment convention, and
`padding: low` is exactly the halo cell named above. No component of this design
sees both ranges at once, so enforcing it belongs to whatever
constructs both
([[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos|a mesh
concept with first-class halos]]). Note also that `order_dimensions` giving a
staggered field the *layout order* of its base says nothing about its *shape*:
the two are independent.


## 4. Axis versus mesh location is a per-declaration decision

The class level also draws a distinction `kind` never did: `IDim` and `CellDim`
are both `HORIZONTAL` and only the first is an axis. That is why the migration
cannot be inferred from the source — whether a dimension is an axis or a mesh
location is a decision per declaration (about ten in ICON4Py's `dimension.py`;
near-uniformly `CartesianAxisIndex` in `tests/next_tests`). The migration script
decides per declaration, in this order:

1. **`CartesianAxisIndex`** on any of this axis evidence: `kind=VERTICAL`; the
   source of a *Cartesian* `FieldOffset`; the first argument of `as_offset(...)` or
   `flip_staggered(...)`; **index arithmetic `D ± <number>` in the source**, which
   v1.2.2 code could already write as `KDim + 1`; or a string constant
   `"_Staggered<old Dimension value>"`, the old encoding of a staggered partner.
2. **`DimensionIndex`** on mesh-location evidence: the source or the target domain
   dimension of a *neighbor* `FieldOffset`.
3. Otherwise **`DimensionIndex`**, reported as "declare it as `CartesianAxisIndex`
   if it is a Cartesian axis (index arithmetic, `Staggered`)". A structured
   horizontal axis with no Cartesian offset in the source lands here, because
   nothing in the source distinguishes it from a mesh location.

A dimension carrying **both** kinds of evidence is a conflict: the script declares
it `DimensionIndex` and reports it for a hand decision rather than choosing, since
the two are mutually exclusive by construction.

## 5. Relation to `dimension-generic-fields`, and one naming constraint

[[personal/havogt/dimension-generic-fields/dimension-generic-fields|Generic dimensions and statically typed staggering]]
Part II proposed this `Staggered[D]` shape with the same shift convention, and
the root/user split that makes the doubly staggered type unrepresentable; here
that split is motivated as the axis concept above and placed below
`DimensionIndex` so that no annotation widens. Its static involution overloads
(typing `Staggered[K] + 0.5` as landing on `K` for the checkers) are not part
of the stack, so statically a half-integer shift is a plain `Connectivity`. Its
`gradient_to_staggered` example, `p(Staggered[I] + 1/2) - p(Staggered[I] - 1/2)`,
is the coboundary of a 0-cochain, which is why it lands on the partner class.

**One naming constraint.** That prototype spells the partner-swapping operation
`dual()` / `Dual[X]`. In a complex the Hodge dual flips *every* axis bit (a
primal k-cell pairs with a dual (n−k)-cell) while `Staggered[D]` flips one, so
the two coincide only in one dimension. `dual` stays reserved for the full
Hodge dual — the sense
[[personal/egparedes/discretization-independent-fd-syntax|the surface-syntax note]]
§6.5 uses for the primal/dual C↔D grid swap — and the per-axis operation keeps
gt4py's existing name, `flip_staggered`.

## 6. Alternatives considered

**A `PEP 695` generic for `Staggered[D]`.** Rejected: the subscription would
be a `typing` alias, not a class, failing `issubclass` and eve's `type[...]`
validation and unable to carry a tag. The metaclass subscription builds a
real class; the PEP 695 form is used only under `TYPE_CHECKING`.

**A sibling root above `DimensionIndex`** — `AnyDimensionIndex` with
`DimensionIndex` and `Staggered[D]` as siblings, which is where
[[personal/havogt/dimension-generic-fields/dimension-generic-fields|generic dimensions]]
puts the split. Gives the same static ban, but `Staggered[K]` then stops being
a `DimensionIndex`, so every annotation and `issubclass` guard that must accept
a staggered dimension has to widen, and the static and runtime lattices diverge
unless the widening is mirrored at runtime. The `AnyCartesianAxisIndex` / `CartesianAxisIndex`
levels sit below the root instead and cost nothing at those sites.

**A structural discriminator instead of the axis levels** — bound `Staggered`'s
parameter to a `Protocol` carrying `__gt_staggered__: ClassVar[Literal[False]]`
that the staggered class overrides with `Literal[True]`. Rejected: it needs a
suppressed incompatible-override, reports an opaque diagnostic, hangs a
meaningless attribute on every dimension, and buys only the ban — there is no
axis concept, so `C + 1` and `Staggered[C]` stay unchecked.

**A second partner constructor** — `Staggered[D]` for `i - 1/2` and
`StaggeredAbove[D]` for `i + 1/2`, instead of a per-axis alignment. Rejected: it
gives three cell classes per axis where an axis has two, so
`Field[Dims[Staggered[I], StaggeredAbove[I]]]` — the same physical axis twice —
becomes expressible and needs a new `check_dims` rule; and it makes
`flip_staggered` partial, since `flip_staggered(I)` would have two candidate
answers, so `I + 0.5` could not compute its own codomain and neither
`as_non_staggered` nor the typed `dual()` would be well defined. It also spells a
per-*pair* fact at every use site, so two modules can disagree about one grid, and
it duplicates the whole bracketed-tag apparatus (a second interned family, tag
grammar, `copyreg` hook and codegen escape). Parameterizing the alignment, if it
is done at all, is one class keyword on the declared axis — see
[[personal/egparedes/connectivities-as-types/connectivities-as-types#open-questions--follow-ups|Open questions]].

**Declaring both members of a pair** (`class ICell(CartesianAxisIndex, dual_of=IFace,
offset=+0.5)`). The explicit form, rejected for the reason this note exists: it
reintroduces a two-object agreement, loses the property that a partner needs no
separate declaration, and loses the involution by construction.

**`Staggered[D: AnyCartesianAxisIndex]`**, bounding on any cell class rather than on a
declared axis. The semantically tempting reading, since a staggered dimension
*is* an axis — but it re-admits `Staggered[Staggered[K]]`. The bound has to name
the declared level.

## 7. Open follow-ups

Numbered as in the main note's *Open questions*.

5. **Parameterizing the alignment.** The ADR 0026 convention is hard-coded, while
   SGRID's `padding` attribute exists precisely because production codes use both
   alignments
   ([[personal/egparedes/connectivities-as-types/connectivities-as-types_conventions|conventions appendix]]
   §3). The antisymmetry above makes both conventions reachable by choosing which member of
   the pair to declare, at a cosmetic cost in the derived member's tag; a per-axis
   class keyword (`class X(CartesianAxisIndex, stagger=Align.ABOVE)`) would remove
   even that. It would have to be **static, on the axis**: it changes the integer
   correction `connectivity_for_cartesian_shift` bakes into the
   `CartesianConnectivity`, hence the emitted stencil, so unlike periodicity and
   extents it cannot be a grid property. Its footprint is one `ClassVar` plus `+1`
   vs `+0` in that one function — the backends only ever see an integer offset
   along an axis — against three costs: `a(Staggered[I] + 1/2)` stops being
   readable without the declaration, the frontend and embedded shift test matrix
   doubles, and Identity rule 4 bites, since dimensions fingerprint *by reference*
   and flipping the keyword in place would not invalidate the ADR 0023 cache. That
   last hole is pre-existing (`kind` has it too) and would argue for folding
   declaration-time dimension attributes into the fingerprint. Not in this stack.

7. **A cell `degree` on the axis.** Making the degree assignment static
   (`degree=0|1`) would let a coboundary be typed generically
   (`d⁰: Field[deg 0] -> Field[deg 1]`) and make the range invariant derivable
   rather than declared. Left out for now because **nothing consumes it** until an
   exterior-calculus surface exists (Proposal 1
   of
   [[personal/egparedes/discretization-independent-fd-syntax|the surface-syntax note]]).
   Absolute ranges do **not** determine the
   degree in general: a periodic axis gives both classes `[0, n)` under either
   assignment, and halo padding makes the relative widths arbitrary. A static degree
   needs no periodicity and would not be ill-defined — it is *deriving extents* that
   needs the degree together with a periodicity and halo policy. Revisit if that
   surface lands — but note that a degree is *canonical* for a mesh
   location and only *declarational* for a Cartesian axis, so it belongs on the
   `LocationIndex` of the next item rather than here
   ([[personal/egparedes/connectivities-as-types/connectivities-as-types_conventions|conventions appendix]]
   §4a).
