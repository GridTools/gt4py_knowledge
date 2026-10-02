---
title: "Connectivities as types — alternatives considered"
author: egparedes
tags: [type-system, connectivities, local-dimensions, offset-provider, nominal-types]
created: 2026-10-01
status: draft
---

> **Appendix** to [[personal/egparedes/connectivities-as-types/connectivities-as-types|Connectivities as types]].
> The alternatives rejected on grounds other than the axis levels, which are in the
> [[personal/egparedes/connectivities-as-types/connectivities-as-types_staggering|staggering appendix]]. Each entry states the alternative, then why
> it loses.

## Alternatives

**Keep every concept; fix only which string wins.** Emit `FieldOffset.value`
instead of `foast.Name.id` in `foast_to_gtir`, validate `value == target[-1].value`
eagerly, extend the #1789 regression test. Two PRs, low risk — and the
`foast_to_gtir` part is a real bug, fixed regardless, in GridTools/gt4py#2898.
But it *enforces* the tangle rather than removing it; no concept goes away.

**A declaration object with a derived local dimension, keeping `Dimension`
instances.** Same shape as this proposal minus static typing and minus type
identity: `local_dim.value == name` by construction, provider keyed by the
declaration object. Superseded by building on dimensions as types instead:
once dimensions are classes, the nested-class form is both simpler and
statically meaningful.

**`(name, kind)` value equality with an interning registry** (the design of
[[shared/dimensions-as-types|dimensions as types]] and GridTools/gt4py#2844).
Keeps independently declared same-named dimensions interchangeable and
avoids the importability rule. Rejected because it decouples the Python
type's identity from the IR's, needs a registry plus `copyreg` plus a custom
fingerprint deconstructor to paper over that gap, and cannot give a nested
`V2E.Local` a unique name without further convention. See [[personal/egparedes/connectivities-as-types/connectivities-as-types#identity-is-the-qualified-python-name|Identity]].

**Integer type parameters for `max_neighbors`** (`LocalDimensionIndex[F: int, M: int]`).
Would need `Literal[6]` type arguments and `Final[F]` over a `TypeVar`;
checkers gain nothing. Class keywords instead.

**Data on the class** (`data: ConnectivityField | Unbound`). Process-global
mutable state; tests bind several meshes per process, and the compile cache
assumes the table travels separately from the type. Binding per call, keyed by
the class, keeps the invariant structural.

**`axis=V2E` as sugar.** Rejected for now; keep the local axis explicit.

**Static-only `max_neighbors` / `min_neighbors`.** Simplest, and correct for
fixed-arity meshes. Rejected as the *only* mode because it binds DSL source
to one mesh family (`fvm_nabla_setup.py` sizes `V2E` from the atlas mesh) and
because skip-value presence is configuration-dependent in ICON, so a static
`min_neighbors` forces either duplicate classes or always-on skip handling.
Declared counts stay available as a constraint; see [[personal/egparedes/connectivities-as-types/connectivities-as-types#binding-model|Binding model]].

**Owner-less local axes as non-LOCAL dimensions.** Rejected: they would
become domain dimensions with a range in every program domain and lose the
sparse storage treatment, changing ICON4Py's layout for `LsqUnkDim` fields.
