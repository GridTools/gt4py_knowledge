---
title: "Connectivities as types — slides"
author: egparedes
tags: [type-system, connectivities, local-dimensions, offset-provider, dimensions, staggering, slides]
created: 2026-10-01
status: draft
paginate: true
---

## Rendering this deck

```console
npx -y @marp-team/marp-cli@4 --no-stdin \
    connectivities-as-types_slides.md --html -o /tmp/deck.html
```

- **`--no-stdin` is required** — without it marp-cli waits on stdin and looks like a hang
- add `--pdf` for a PDF; that path needs a Chromium-family browser
- the first run downloads marp-cli and may take several minutes, later runs under a second
- the file also reads as an ordinary page in the garden, scrolling rule to rule

→ [Appendices](connectivities-as-types#appendices)

---

# Connectivities as types

One declaration for offset, local dimension and provider key

ADR 0029 (dimensions) · ADR 0030 (connectivities) · GridTools/gt4py#2917

> **Appendix** to [Connectivities as types](connectivities-as-types), summarising that note as of commit `8292eff`. Every slide links the section it compresses and this deck adds no facts of its own, so where the two disagree the note wins. Rendering instructions are on the slide before this one.

---

## Today: four strings must agree

```python
V2EDim = Dimension("V2E", kind=DimensionKind.LOCAL)                # (N3) local dim name
V2E    = FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))  # (N1) tag, (N2) variable
program(..., offset_provider={"V2E": table})                       # (N4) provider key
```

Four independently authored strings. None checked against the others at declaration time. A fifth constraint is hidden: the Python *variable* name.

→ [Four strings, one dict lookup](connectivities-as-types#four-strings-one-dict-lookup)

---

## Which string wins depends on the path

| Path | Operation | String used |
| --- | --- | --- |
| embedded | shift `a(V2E[1])` | N1 `FieldOffset.value` |
| embedded | `neighbor_sum(axis=V2EDim)` | N3 `axis.value` |
| compiled | shift | **N2 `foast.Name.id`** |
| compiled | reduction | N3 `ListType.offset_type.value` |
| compiled | sparse field argument | N3 `dim.value` |

Confirmed by execution: embedded and compiled key the *same* program on *different* strings.

→ [Four strings, one dict lookup](connectivities-as-types#four-strings-one-dict-lookup)

---

## Proposed: the connectivity is a class

```python
class V2E(NeighborConnectivity[V, E], max_neighbors=6, min_neighbors=5):
    class Local(LocalDimensionIndex): ...

neighbor_sum(a(V2E), axis=V2E.Local)
program(a, out=out, offset_provider={V2E: v2e_table})
```

The class *contains* its local dimension and *is* its own provider key. Nothing is ever an instance of `V2E`; it holds no data.

→ [Concepts](connectivities-as-types#concepts) · [Sketch](connectivities-as-types#sketch)

---

## Identity is the qualified Python name

`tag = f"{cls.__module__}.{cls.__qualname__}"`

- the static view (nominal types) and the runtime view (`is`) agree by construction
- reconstruction from the IR is an **import**, the way `pickle` references a class
- so `<locals>` dimensions are rejected, and a redefinition is a *different* connectivity
- the IR names a connectivity by its `offset_tag` = its local dimension's tag, so shifts, reductions and sparse arguments find one table under **one** string

→ [Identity](connectivities-as-types#identity-is-the-qualified-python-name)

---

## What a Python type can and cannot say

The "true" type of a local index is `Fin<degree(table, v)>` — it depends on the table *value*. Python types cannot mention values.

- **in the type**: *which* neighbor relation (`V2E.Local`), domain, codomain, declared counts
- **at bind time**: `check_neighbor_table` — dtype, skip value, actual width
- **given up**: per-element valence

That split is what lets DSL code be written, checked and compiled before any mesh is loaded, and one compiled program serve many meshes.

→ [Binding model](connectivities-as-types#binding-model)

---

## Dimensions: four levels, one root

```
DimensionIndex                            # the root; every annotation says this
├── AnyCartesianAxisIndex                 # a cell class of a Cartesian axis
│   ├── CartesianAxisIndex                # ← declared; what users subclass
│   └── Staggered[D: CartesianAxisIndex]  # derived
├── LocalDimensionIndex                   # a neighbor index; not an axis
└── (direct subclasses)                   # mesh locations V/E/C
```

Below the root, so nothing widens and `LocalDimensionIndex` does not move.

→ [Cartesian axis dimensions](connectivities-as-types#cartesian-axis-dimensions)

---

## Four checks become static

| Rejected | Today | With the axis levels |
| --- | --- | --- |
| `Staggered[Staggered[K]]` | runtime `TypeError` | static |
| `Staggered[V2E.Local]` | runtime `TypeError` | static |
| `Staggered[C]` | nothing rejects it | static |
| `C + 1`, `V2E.Local + 1` | runtime `kind` check | static |

No half-cells on an unstructured mesh. Verified with mypy and pyright in `staggered_probe.py`.

→ [Cartesian axis dimensions](connectivities-as-types#cartesian-axis-dimensions)

---

## Why two levels, not one

An axis of a Cartesian grid is a 1-dimensional CW complex: exactly **two** cell classes. A declared axis and its `Staggered[·]` name those two; `Staggered` swaps them.

So a staggered dimension *is itself* an axis — the asymmetry is **declarational**, not semantic. That is the level the bound must name, and it is why there is nothing to nest.

In *n* dimensions the grid is the product complex, so `Dims[...]` is one staggering bit per axis — finer than a form degree.

→ [Staggering appendix](connectivities-as-types_staggering)

---

## What it deletes

`FieldOffset` · bare-name provider keys · the `_Staggered` name prefix · `_CONST_DIM` as a magic name · `SparseTag` · `NamedIndex` · the dimension half of the mypy plugin · `DimensionKind.LOCAL` · the `V2EDim`-next-to-`V2E` convention

Of ten cross-object identity constraints: **six dissolve**, three collapse into one bind-time check, one remains.

→ [What it deletes](connectivities-as-types#what-it-deletes) · [Effect per layer](connectivities-as-types#effect-per-layer)

---

## Where it stands

| PR | Contents |
| --- | --- |
| A #2899 | dimensions as classes, `Staggered[D]`, ADR 0029 |
| B #2907 | `NeighborConnectivity`, `LocalDimensionIndex`, ADR 0030 |
| C #2910 | class-keyed providers, `FieldOffset` removed, migration script |
| D #2912 | `MultiDimensionIndex`, typed embedded positions |

ICON4Py: the script rewrites 4 files and reports 46 provider keys to decide.

→ [Implementation](connectivities-as-types#implementation)

---

## Open, and worth arguing about

- **naming** — convergence with the chain proposals is still open
- **the mesh proposal** — does a `Mesh` bind the typed declarations?
- **`kind`** — four jobs in one enum; it should shrink, not grow
- **`__eq__` vs "equality is `is`"** — asserted in one place, overridden in another
- **alignment** — `Staggered[D](i)` at `i − ½` is hard-coded; both conventions exist in production

→ [Open questions](connectivities-as-types#open-questions--follow-ups)
