---
title: "GT4Py Next DSL documentation: specification snapshot and documentation method"
author: egparedes
tags: [documentation, dsl, public-api, frontend, type-system, fields, domain, dimensions, connectivities, field-operators, scan, embedded, validation, provenance]
created: 2026-10-22
status: draft
---

> **Proposal.** Maintain a paired language specification and Python API reference
> for `gt4py.next`, with explicit source provenance, contradictory definitions,
> adopted interpretations, and bounded validation evidence. The initial versions
> of both documents are the two appendices below.

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

## Baseline

Both documents describe GT4Py `main` at
[`f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f`](https://github.com/GridTools/gt4py/tree/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f),
the revision they were written from. Their code, test, and documentation
evidence links point at that commit.

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
Update relevant conflict entries with the contract, and keep the baseline
evidence when comparing revisions. Host-call limitations are already
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
The appendices report those observations; the original probe scripts and
logs were not retained, and supplemental independent notebook sampling was
incomplete. They do not provide a fully replayable investigation archive.

The full nox suite and native C++/GPU/DaCe/JAX/CuPy backend matrices were not run.
Numerical edge behavior, arbitrary aliasing, closure mutation, and exotic dtype,
declaration-signature, and remapping paths retain the limitations stated in the
appendices. This proposal makes no backend conformance claim. The review
chronology is in the
[proposal PR](https://github.com/GridTools/gt4py_knowledge/pull/40).

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
validation record together. Record the new baseline revision when refreshing; explicitly distinguish newly observed behavior from recommendations.
If the team adopts a normative specification later, move that decision into the
appropriate GT4Py documentation/ADR process with human review.

Open questions are who maintains the paired documents, how often to refresh the
baseline, which C1–C22 resolutions warrant runtime work, and what backend/numerical
conformance coverage is required before promising a stronger contract. This
proposal seeks discussion of that documentation process; it does not select
runtime fixes or settle the competing designs linked above.
