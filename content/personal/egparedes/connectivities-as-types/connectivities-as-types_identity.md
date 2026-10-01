---
title: "Connectivities as types — what type identity costs the IR, pickle and codegen"
author: egparedes
tags: [type-system, dimensions, nominal-types, serialization, fingerprint, metaclass, tech-debt]
created: 2026-10-01
status: draft
---

> **Appendix** to [[personal/egparedes/connectivities-as-types/connectivities-as-types|Connectivities as types]].
> The mechanics behind rules 1 and 3-6 of *Identity is the qualified Python name*:
> reconstruction by import, the one `copyreg` hook, fingerprinting, the injective
> codegen mangling, and the `Staggered[D]` tag grammar.

## Rule 1 — reconstruction from the IR is an import

1. **Reconstruction from the IR is an import.** `resolve(tag)` imports the
   longest importable module prefix and walks the rest as attributes (nested
   classes like `V2E.Local` resolve naturally), the way `pickle` references a
   class. Only where the module path ends is memoized: the attribute walk is
   repeated, so a declaration redefined under the same name (a re-run
   notebook cell) resolves to the new class. `AxisLiteral` stores only the tag
   and derives `kind` and `dim` from it as read-only properties (the `TODO` at
   `iterator/ir.py:93`); the derived `kind` is optional, since a local
   dimension's is `None`;
   printing IR never imports (`resolve_loaded`). `Staggered[D]` tags use the
   small grammar `<owner tag>[<base tag>]`, resolved by subscripting the owner.

## Rules 3-6 — registry, fingerprints, mangling, staggering

3. **No registry.** Classes pickle by reference. One narrow `copyreg` hook
   exists, for `Staggered[D]`, whose bracketed qualname `save_global` cannot
   look up; it reduces to the base dimension and re-interns.
4. **Fingerprints depend on qualified names.** Dimensions are fingerprinted by
   reference (`Staggered[D]` through its base); a connectivity declaration
   additionally by its domain, codomain, `Local` and counts, so a redefinition
   under the same name does not reuse artifacts. Consequently the ADR 0023
   build cache invalidates on module renames.
5. **Codegen names need injective mangling.** `codegen_name(tag)` is a prefix
   escape — `_`→`_u`, `.`→`_d`, `[`→`_l`, `]`→`_r` — used by gtfn, DaCe, the
   roundtrip backend and the nanobind bindings; its inverse
   `from_codegen_name` is used by DaCe to parse names back. The simpler
   "escape `__`, then replace `.`" is **not** injective (`".."` and `"_"`
   collide); the prefix escape is tested exhaustively over its alphabet up
   to length 4.
6. **Staggering is part of the dimension change.** The `_Staggered` name
   prefix needs the name→class lookup that type identity removes, so
   `Staggered[D]` is a real class and part of ADR 0029 (PR A), together with
   the [[personal/egparedes/connectivities-as-types/connectivities-as-types#cartesian-axis-dimensions|Cartesian axis dimensions]] its parameter is bounded on; see
   [[personal/egparedes/connectivities-as-types/connectivities-as-types#staggeredd|`Staggered[D]`]].

## The `Staggered[D]` tag

- **IR and serialization.** The tag is
  `gt4py.next.common.Staggered[<base tag>]` — the one tag that is not a
  qualified name. `resolve` parses the `<owner>[<base>]` grammar and
  subscripts the owner, the IR pretty-parser accepts it, and `codegen_name`
  escapes the brackets. `pickle` cannot look up a bracketed qualname, so one
  `copyreg` hook reduces a staggered class to its base and re-subscripts on
  load; fingerprints are likewise taken through the base.
