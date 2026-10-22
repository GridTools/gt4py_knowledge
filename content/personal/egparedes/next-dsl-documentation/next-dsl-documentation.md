---
title: "GT4Py Next DSL documentation: specification snapshot and generation process"
author: egparedes
tags: [documentation, dsl, public-api, frontend, type-system, fields, domain, dimensions, connectivities, field-operators, scan, embedded, validation, provenance]
created: 2026-10-22
status: draft
---

> **Proposal.** Maintain a paired language specification and Python API reference
> for `gt4py.next`, with explicit source provenance, contradictory definitions,
> adopted interpretations, and bounded validation evidence. The complete initial
> snapshot is preserved in the two appendices below. Apply the
> [[personal/egparedes/next-dsl-documentation/next-dsl-documentation#errata-for-this-baseline|baseline errata]]
> when reading either snapshot; preservation does not imply that every original
> claim is correct.

## Motivation and scope

The Python-facing computation DSL is described across implementation, tests,
user/developer documentation, ADRs, and examples. Those sources sometimes disagree.
A single account of syntax, typing, execution, and host-call contracts helps
readers distinguish implemented behavior from design intent and future proposals.
Keeping a language specification and an operation-oriented API reference together
also makes contradictions between their descriptions easier to find.

This is a personal **draft documentation proposal and reference snapshot** for
discussion and maintenance. It has not been adopted as GT4Py's normative standard
or architecture. The AI-assisted documents require human
review before adoption. Independent AI-agent review is neither human adoption nor
backend certification. No runtime source or tests
are changed by this proposal.

The scope is the Python-facing `gt4py.next` computation language and Python
interface: fields and domains, field operators, scans, programs, builtins,
construction, invocation, and compilation controls. Internal iterator and
functional IRs supply implementation evidence; they are not additional user
languages specified here. The separate Cartesian DSL is outside this scope.

## The paired documents

- [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification|Computation DSL specification]]:
  precise prose, an abstract grammar, typing rules, and operational semantics where
  useful; a C1–C22 conflict register in section 8; evidence coverage and focused
  observations in section 9.
- [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_api-reference|DSL API reference]]:
  category indexes and per-operation contracts, signatures, returns, limitations,
  examples, and direct links to the relevant C1–C22 interpretations. Its organization
  follows the category/per-operation style of Python Array API documentation; it
  does not claim Array API conformance.

Current behavior and proposed resolutions must remain visibly distinct. A broad
annotation, a historical ADR, or frontend acceptance alone does not establish
executable support. An interpretation in the conflict register is a documented
reading or recommendation, not a runtime fix.

## Provenance and preservation

The described implementation baseline is
[`f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f`](https://github.com/GridTools/gt4py/tree/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f).
The two source documents were committed together in
[`66b811b78f34cb16a95ebe9f29bb5243f7e48faf`](https://github.com/GridTools/gt4py/commit/66b811b78f34cb16a95ebe9f29bb5243f7e48faf).
That documentation commit and the implementation baseline serve different roles:
the former identifies the reviewed text; the latter identifies the code and tests
that its claims describe.

| Reviewed document | Original at documentation commit | SHA-256 of original bytes |
| --- | --- | --- |
| Language specification | [Source document](https://github.com/GridTools/gt4py/blob/66b811b78f34cb16a95ebe9f29bb5243f7e48faf/docs/development/next/dsl-specification.md) | `bd9c93eb41480b7d18a46bf962c8c0a7e31a0b601ce9390130d5734390f90cc1` |
| API reference | [Source document](https://github.com/GridTools/gt4py/blob/66b811b78f34cb16a95ebe9f29bb5243f7e48faf/docs/development/next/dsl-api-reference.md) | `ef2fe1d4fc52bf0e9cf30c38c54a30e5bcbc1ecc943c58654a342c0a6ad2663a` |

The documentation branch has been deleted, and its commit is not retained by a
branch or tag. Those original-document and commit links resolved during the
independent migration review on 2026-10-22, but GitHub may eventually remove the
unreferenced objects. They are convenience links, not an availability guarantee.
The complete copies in this repository and the original SHA-256 values above are
the lasting record. The implementation baseline is retained in GT4Py's `main`
history; the appendices' code, test, and documentation evidence links point there.

Both appendices include their full reviewed source bodies. Migration adds draft
frontmatter and a context notice, changes relative evidence links into immutable
GitHub `blob` links for files and `tree` links for directories at the implementation
baseline, and changes companion-document links into full-path Quartz wikilinks.
Existing heading fragments are retained and checked against their targets. These
changes preserve all reviewed technical claims, examples, C1–C22 entries, and API
interpretations. Reading the preserved documents and their implementation evidence
does not depend on retaining the original documentation branch. The copied bodies
also preserve their original formatting.

The original hashes identify the pre-migration documents, not the bytes of these
appendices after frontmatter and link changes. A byte-for-byte inverse of the
recorded migration substitutions reconstructs both original bodies. This is a
content-preservation check, not a new semantic review.

## Documentation method and evidence policy

Build the language account from three complementary sources: user/developer
reference material and examples; frontend, data-model, embedded, and lowering
implementation; and unit, integration, and regression tests. Use precise prose,
abstract grammar, typing rules, and operational semantics where these clarify a
contract. Organize the companion API by category and operation, with signatures,
parameters, results, limitations, and examples. Both documents describe the same
implementation revision.

Cross-check each contract across those sources. Record disagreements explicitly,
with links to the competing definitions and the interpretation used by the API.
Keep implemented behavior separate from future recommendations. Historical ADRs
provide design intent, test skips and expected failures bound executable support,
and frontend acceptance alone does not establish backend support. Use small
frontend or execution probes where reading sources leaves uncertainty.

For maintenance, check the language rules and per-operation entries together.
Update relevant conflict/errata entries with the contract, and preserve the
baseline evidence when comparing revisions. Host-call limitations are already
recorded in C2/C4/C20–C22: output-domain differences, legacy offsets and providers,
compilation restrictions, incomplete direct embedded validation, and the distinction
between omitted and explicit `None` domains. They must qualify the affected API
entries rather than becoming implicit coercion or support guarantees.

### Evidence read, behavior executed, and limits

The source investigation inspected the cell sources of all 21 Next-relevant
notebooks, including workshop slides, exercises/solutions, the Jupyter introduction, and the
Laplacian comparison; it did not rerun the notebooks in full. Specification
section 9 records documentation, implementation, and representative test coverage.
Reading those test trees is distinct from running their cases.

The recorded execution evidence includes six concrete API examples with NumPy
embedded fields; four decorated computation examples also ran with `itir_python`.
Focused probes reproduced multiple-output-domain and scalar/zero-dimensional
boolean differences and checked specific frontend and host-interface limitations.
The source snapshots report those observations; the original probe scripts and
logs were not retained, and supplemental independent notebook sampling was
incomplete. They do not provide a fully replayable investigation archive.

The full nox suite and native C++/GPU/DaCe/JAX/CuPy backend matrices were not run.
Numerical edge behavior, arbitrary aliasing, closure mutation, and exotic dtype,
declaration-signature, and remapping paths retain the limitations stated in the
appendices and errata. This proposal makes no backend conformance claim. Detailed
review chronology, preservation checks, and migration validation belong in the
[proposal PR](https://github.com/GridTools/gt4py_knowledge/pull/40).

## Errata for this baseline

These corrections and qualifications take precedence over the preserved snapshot
text. They concern the same implementation baseline, not proposed runtime changes.
The original bodies and their C1–C22 register remain intact for provenance; a
maintained successor should incorporate these corrections into both documents.

The behavioral qualifications below were verified against baseline source and
focused NumPy embedded/`itir_python` probes. Those probes used dependencies from
an existing local environment rather than the baseline's lockfile; they establish
the observed paths, not a backend matrix. Probe scripts are not knowledge-base
artifacts. Source-reference and wording clarifications require no new runtime
behavior.

### E1. Chained assignment is not supported

Specification sections 2.1–2.2 overstate chained name assignment. At this baseline,
`b = c = a` fails during decoration with `AssertionError`: the
[single-assignment-target pass][simple-assign] creates a synthetic load without a
source location, while the [dialect parser][dialect] requires one. Treat chained
assignment as unsupported and remove `chained-name-targets` from the supported
`targets` production. Use separate assignments instead. The pass's intended
transformation does not establish executable support.

### E2. Local annotated assignment is partial

The same sections and the API's computation-statement table must qualify local
annotated assignment. Builtin scalar forms,
such as `x: float = 1.0`, work. General Python annotation resolution is not
available: [local annotation evaluation][fo-parser] uses the DSL builtin context
plus captured closure variables. [Source capture][source] deliberately excludes
names used only in annotations. Thus `Dims`, a module prefix such as `gtx.`, or a
dimension used only in a local annotation can raise a raw `NameError` during
decoration, even if imported at module level. Once the annotation resolves,
mismatches with the assigned value are DSL type errors. Skipped
or expected-failure annotation cases in the [parser tests][fo-parser-tests] and
[lowering tests][fo-lowering-tests] do not establish general support. Prefer
ordinary inferred bindings unless the exact annotated form is supported.

### E3. Fixed-shift constants have restricted folding

Specification section 5.2 and the API Cartesian-shift entry must not imply that a
plain captured integer such as `SHIFT = 1` is usable in `field(I + SHIFT)`.
[Closure folding][closure-fold] supports attributes of the recognized frozen
namespace/enum objects; it does not turn arbitrary captured scalar names into
the literal node required by fixed-shift typing. Use an inline literal or a
supported folded attribute. This is separate from runtime dynamic offsets.

### E4. A not-equal region cannot be constructed embedded

Specification section 5.4 and C16 already record the rejection; read the looser
region-table and API `concat_where` wording strictly. `d != n` is frontend syntax
with backend-specific support; evaluating it directly in field-level embedded
execution raises `NotImplementedError` in [Dimension comparison][model] before
`concat_where` runs. It is not merely a potentially noncontiguous result.
For the corresponding selection, use `concat_where(d == n, f, t)` or
`concat_where(d < n, t, concat_where(d > n, t, f))`, subject to the ordinary
branch-domain and contiguity requirements. This selects `f` at `n` and `t` on
both sides without constructing a disjoint condition.

### E5. Field dimensions must already have canonical order

In specification section 1.1 and the API dimension/type entries,
[dimension checks][model] reject noncanonical order
as well as duplicates, multiple local dimensions, and base/staggered coexistence.
For horizontal `I` and vertical `K`, using `Dims[K, I]` in a DSL annotation raises
`InvalidAnnotationError` at decoration; use `Dims[I, K]`. Canonical promotion does
not imply automatic reordering of
an invalid field annotation. Host `Domain` construction has different checks.

### E6. Scan configuration has defaults

Specification sections 2, 5.5, and 7 must distinguish required `axis` from optional
configuration. The [scan decorator][decorator] defaults to `forward=True` and
`init=0.0`. The initial value must still match the carry and return type exactly:
the default is a scalar `float64` initializer, not an implicit initializer for
integer, differently typed, or product carries. Supply a compatible explicit
`init` for those cases. The API signature already displays the defaults.

### E7. Missing-neighbor evidence is in the embedded field implementation

C10's execution-side evidence should cite [embedded fields][array-field],
particularly remapping's `skip_value is None or skip_value == -1` assertion and
`_make_reduction`'s comparison against `_DEFAULT_SKIP_VALUE`. The latter uses the
fixed sentinel rather than arbitrary connectivity metadata. This corrects the
`builtins` evidence attribution; the portable `-1` interpretation is unchanged.

### E8. Export coverage does not cover every host utility

The [public exports][public] also include `Device`, `DeviceType`,
`CUPY_DEVICE_TYPE`, and `is_scalar_type`, as well as the `program_processors`
submodule. These host device-selection, scalar-classification, and execution
facilities are outside this snapshot's per-operation reference; consult the
utilities' [core definitions][core-definitions]. Constructor `device=` parameters
take `Device` values. Their omission must not imply
that they are unexported or that the API page inventories every package export.

### E9. Complete the domain grammar production

The undefined `domain-entry` in specification section 2.1 is an editorial omission.
Read the mapping production as follows, with the existing static domain and
output-structure restrictions still applying:

```ebnf
domain-spec  = "{" domain-entry { "," domain-entry } "}" | domain-tuple ;
domain-entry = dimension ": (" integer-expression "," integer-expression ")" ;
```

### E10. The power diagnostic is printed

In specification section 4.2 and the API `power` entry, the integral-argument
[typing diagnostic][typing] uses `print` to standard output during frontend type
deduction when all operands are integral, not `warnings.warn`. It is diagnostic
text, not a Python warning category or a promise of a particular warnings-filter
behavior. The dtype and numerical limitations in C5/C14 remain applicable.

### E11. Clarify zero-dimensional domains and signed literals

The API's domain with no axes contains exactly one coordinate in the mathematical
model, the empty tuple `()`; it is not an empty domain. This does not describe
Python sequence-membership behavior on `Domain`. In C13, “unsigned operand first” means the
positive literal magnitude is classified before unary minus; it does not mean
an unsigned integer dtype is inferred. Specification section 3.1's examples of
negative integer minima explain that distinction.

### E12. Read interpretation notes as whole-entry qualifications

C20/C21/C22 interpretation notes qualify their entire API entry even where they
appear before `Returns`. A maintained version should place them at the end of
the affected entries for consistent reading. Their position does not relax
return, parameter, or error contracts.

### Evidence clarification: grid deduction

Grid deduction needs no semantic correction: [program lowering][p-lowering]
calls `transform_utils._deduce_grid_type`, defined in the already cited
[transform utilities][transform]. Both the implementation and caller are useful
evidence for specification section 1.1.

## Related proposals and explicit differences

These notes have their own baselines and adoption states. Their proposed changes
must not be silently folded into this implementation snapshot.

- [[personal/egparedes/connectivities-as-types/connectivities-as-types|Connectivities as types]]
  replaces string-tag `FieldOffset` declarations/provider keys and value-based
  dimension identity with typed declarations and nominal identity. That conflicts
  with the snapshot's `(name, kind)` dimension identity and current string-provider
  model. The note reports an implementation stack; this snapshot does not assert
  those changes are present at its baseline. It supersedes the earlier
  [[shared/dimensions-as-types|Dimensions as types]] proposal, whose class-based
  declaration syntax is also absent from this snapshot.
- [[personal/egparedes/mesh-field-design-space/mesh-field-design-space|Meshes, dimensions and fields]]
  explores typed relations over a rank-local mesh, a logical mesh with runtime
  decomposition, and a cochain core with located fields. These redesigns overlap
  the snapshot's fields, domains, connectivities, and dimensions (section 1,
  C4, C15), and would change the corresponding representation or host interface.
  They are future design alternatives, not contracts of this baseline.
- [[personal/havogt/mesh-and-first-class-halos/mesh-and-first-class-halos|A mesh concept with first-class halos]]
  replaces the opaque `offset_provider` dictionary with a mesh object and proposes
  compiler-managed halo exchanges. This conflicts with the current provider
  contracts in C4/C20; neither the mesh interface nor automatic halo exchange is
  promised by this snapshot.
- [[personal/havogt/dependent-local-dimensions/dependent-local-dimensions|Dependent local dimensions and connectivity chains]]
  makes local dimensions carry their connectivity and explores typed reduction
  order over multiple hops. That extends the snapshot's single-local-dimension
  restrictions and reduction rules (section 5, C8, C9). Neither multi-hop local
  dimensions nor the proposed reduction-order typing is established here.
- [[personal/havogt/scan-redesign/scan-redesign|Scan redesign]] proposes a level-slice
  scan with different carry/output and boundary-handling choices. The snapshot
  describes the current scalar-recurrence `scan_operator`; the proposed scan is
  not an alternative syntax already accepted by it.
- [[personal/havogt/closure-variable-resolution|Closure variable resolution]] seeks
  value-based canonicalization and a clearer capture model. The snapshot records
  current direct-name restrictions and same-named recursive capture conflicts,
  including unresolved capture-lifetime differences in C11. The intended future
  resolution must not be read as current behavior.
- [[personal/havogt/field-data-protocol/field-data-protocol|FieldData protocol]]
  separates domain algebra from value production and explores lazy or unbounded
  field representations. It could address embedded limitations documented in C7
  and C16; the snapshot does not promise that representation or its prototype API.
- [[personal/havogt/boundary-condition-syntax/boundary-condition-syntax|Boundary-condition syntax]]
  explores syntax over `concat_where`. The snapshot retains the experimental
  primitive and its embedded region limitations; the proposed surface constructs
  are not part of its grammar.
- [[personal/havogt/dtype-generic-fields/dtype-generic-fields|Dtype-generic fields]]
  and [[personal/havogt/dimension-generic-fields/dimension-generic-fields|Generic dimensions and statically typed staggering]]
  extend typing and specialization. Their proposed generic signatures must remain
  separate from the concrete annotation contracts supported by this baseline.
- [[personal/egparedes/layered-architecture/layered-architecture|Layered architecture]]
  and [[personal/egparedes/otf-toolchain-split|OTF toolchain split]] concern public
  boundaries and host infrastructure. The reference documents relevant invocation
  and compilation controls without adopting either proposed architectural split.

## Maintenance proposal and open questions

Keep the documents paired: an API contract change should update the relevant
language rule, conflict entry, API interpretation, evidence references, and
validation record together. Preserve this baseline's provenance when comparing a
new revision; explicitly distinguish newly observed behavior from recommendations.
If the team adopts a normative specification later, move that decision into the
appropriate GT4Py documentation/ADR process with human review.

Open questions are who maintains the paired documents, how often to refresh the
baseline, which C1–C22 resolutions warrant runtime work, and what backend/numerical
conformance coverage is required before promising a stronger contract. This
proposal seeks discussion of that documentation process; it does not select
runtime fixes or settle the competing designs linked above.

[simple-assign]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/ast_passes/simple_assign.py
[dialect]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/dialect_parser.py
[fo-parser]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/func_to_foast.py
[source]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/source_utils.py
[fo-parser-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/unit_tests/ffront_tests/test_func_to_foast.py
[fo-lowering-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/unit_tests/ffront_tests/test_foast_to_gtir.py
[closure-fold]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/foast_passes/closure_var_folding.py
[model]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/common.py
[decorator]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/decorator.py
[array-field]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/embedded/nd_array_field.py
[public]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/__init__.py
[core-definitions]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/_core/definitions.py
[typing]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/foast_passes/type_deduction.py
[p-lowering]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/past_to_itir.py
[transform]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/transform_utils.py
