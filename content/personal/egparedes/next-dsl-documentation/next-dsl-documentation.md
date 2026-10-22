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
> snapshot is preserved in the two appendices below.

## Motivation and scope

The Python-facing computation DSL is described across implementation, tests,
user/developer documentation, ADRs, and examples. Those sources sometimes disagree.
A single account of syntax, typing, execution, and host-call contracts helps
readers distinguish implemented behavior from design intent and future proposals.
Keeping a language specification and an operation-oriented API reference together
also makes contradictions between their descriptions easier to find.

This is a personal **draft documentation proposal and reference snapshot** for
discussion and maintenance. It has not been adopted as GT4Py's normative standard
or architecture. Independent agent approval of the source documents is not human
review and does not change the proposal's draft status. No runtime source or tests
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

| Reviewed document | Immutable original | SHA-256 of original bytes |
| --- | --- | --- |
| Language specification | [Source document](https://github.com/GridTools/gt4py/blob/66b811b78f34cb16a95ebe9f29bb5243f7e48faf/docs/development/next/dsl-specification.md) | `bd9c93eb41480b7d18a46bf962c8c0a7e31a0b601ce9390130d5734390f90cc1` |
| API reference | [Source document](https://github.com/GridTools/gt4py/blob/66b811b78f34cb16a95ebe9f29bb5243f7e48faf/docs/development/next/dsl-api-reference.md) | `ef2fe1d4fc52bf0e9cf30c38c54a30e5bcbc1ecc943c58654a342c0a6ad2663a` |

Both appendices include their full reviewed source bodies. Migration adds draft
frontmatter and a context notice, changes relative evidence links into immutable
GitHub `blob` links for files and `tree` links for directories at the implementation
baseline, and changes companion-document links into full-path Quartz wikilinks.
Existing heading fragments are retained and checked against their targets. These
changes preserve all reviewed technical claims, examples, C1–C22 entries, and API
interpretations. No link depends on the temporary source branch remaining present.
The copied bodies also preserve their original formatting.

The original hashes identify the pre-migration documents, not the bytes of these
appendices after frontmatter and link changes. A byte-for-byte inverse of the
recorded migration substitutions reconstructs both original bodies. This is a
content-preservation check, not a new semantic review.

## How the source documents were generated

The following summarizes the original investigation and review history. The
appendices' evidence sections and immutable source references expose its technical
basis; the execution and review history here is a process record, not a claim that
those checks were repeated during migration.

1. **Scope and research plan.** The user limited the investigation to the
   Python-facing DSL and Python interface, requesting precise prose, grammar,
   typing rules, and operational semantics where useful. Concrete language and
   interface research was delegated to two subagents, with at most two running
   concurrently and frequent saved checkpoints. They reviewed implementation,
   tests, user/developer docs, ADRs, examples, and workshop notebook cell sources.
   All 21 Next-relevant notebooks had their cell sources inspected; the notebooks
   were not all rerun. Sections 8 and 9 of the specification record disagreements
   and coverage rather than treating all sources as equally authoritative.
2. **Initial adversarial review and companion reference.** An independent Astra
   review at medium reasoning effort found four issues in the initial formal
   specification; those findings were fixed and verified. A second document was
   then organized as an API reference. Direct contradiction links and explicit
   adopted interpretations were added at affected API entries.
3. **Independent cross-document review.** A separate codex-senior-developer review
   using `gpt-6.1-sol` at xhigh reasoning effort checked both documents for internal
   coherence, agreement with each other, and fidelity to repository documentation,
   code, and tests. It found seven issues: six medium and one low. The corrections
   are summarized below.
4. **Corrections and focused re-review.** All seven were corrected consistently in
   both documents. C2 and C4 were qualified; C20–C22 were added with matching API
   interpretation notes. Focused review round 1 confirmed all seven findings were
   resolved and requested one remaining source-reference correction: C20 needed
   `_compile_variant` in place of `add_variant`. Round 2 explicitly **APPROVED** the
   correction, verified unchanged API content and equality between the reviewed,
   staged, and working-tree files, and reported no remaining findings within scope.
5. **Continuity and preservation.** Low usage limits interrupted the review.
   Checkpoints and two-hour recovery heartbeats preserved and resumed the same
   reviewer. Final hash and staged-content checks preserved the approved documents.
   These operational recovery steps establish continuity of review; they do not
   add semantic or backend coverage. No new subagents were used for this migration.

| Review correction | Contract preserved in the appendices |
| --- | --- |
| Descriptor-only `.compile()` providers | Nonempty descriptor-only mappings are rejected; runtime table mappings are required on the documented path. |
| Direct scan precompilation | Direct scan `.compile()` is unavailable; a concretely annotated wrapping program is the documented ahead-of-time route. |
| Embedded scan output domains | Embedded scans require a shared domain/column interval; per-leaf domain tuples are not a universal workaround. |
| Legacy indexed Cartesian `FieldOffset` | It retains tagged-provider behavior and is not equivalent to self-contained dimension shifts. |
| Captured program output slices | Slice bounds are literal-only; explicit runtime domains are the qualified alternative. |
| Direct embedded runtime validation | Captured type rules remain caller obligations; invalid direct embedded calls are not uniformly rejected. |
| Explicit operator `domain=None` | Embedded convenience differs from compiled calls; omission requests output-derived domains across the documented common interface. |

## Consistency checks and their limits

### Original semantic and document validation

The original work reported the following checks on the final reviewed content:

- Required GT4Py staged pre-commit hooks passed: mdformat, applicable generic file
  checks, and tach; Python-only hooks were skipped for the documentation-only
  change. The final documentation validator checked 281 specification and 159 API
  local links/anchors, 20 tables, and 9 Python fences, including Python AST parsing
  and direct API issue-link coverage for every C1–C22 entry. Earlier API
  completeness checks covered 50 exported builtin/type symbols and 37 individual
  function entries.
- All six concrete API Python examples ran in NumPy embedded execution. The four
  decorated computation examples also ran with `itir_python`. The multiple-output
  domain and scalar/zero-dimensional boolean discrepancies in specification section
  9.3 were reproduced; the reviewer additionally ran focused probes for the seven
  findings above.
- Review checked internal coherence, cross-document synchronization, evidence
  fidelity, and final file identity. Approval is an independent **agent review**,
  not human approval or all-backend certification.

The full notebooks, full nox suite, and native C++/GPU/DaCe/JAX/CuPy backend matrices
were not run as part of this work. Numerical edge behavior, arbitrary aliasing,
closure mutation, and exotic dtype, declaration-signature, and remapping paths
remain bounded by the limitations recorded in the appendices. Reading a test tree
or an ADR is distinct from executing its cases or certifying all backends.

### Knowledge-base migration validation

The following migration checks passed. They cover preservation, links, document
structure, and repository integration; they do not repeat the original runtime
probes:

- Both source files matched the committed blobs and approved SHA-256 values.
  Reversing only the documented link substitutions, after removing the added
  frontmatter/context, reproduced every original byte: 1,518 specification lines
  and 1,850 API-reference lines.
- Immutable source targets were verified against 89 Git objects, covering the
  baseline evidence and original-document provenance, with `blob` for files and
  `tree` for directories. All 94 wikilinks and 55 additional internal heading
  links resolved, including C1–C22 API interpretation links and internal API
  navigation. These are local Git-object and document checks; they do not claim
  every external URL was network-tested.
- All 22 tables passed raw row-column checks, and all 9 Python fences parsed as
  Python ASTs. Markdown whitespace and frontmatter checks passed. The index has
  exactly one main-proposal entry with keywords matching the main note's tags.
  Appendices remain reachable from this note without separate index entries.

## Related proposals and explicit differences

These notes have their own baselines and adoption states. Their proposed changes
must not be silently folded into this implementation snapshot.

- [[personal/egparedes/connectivities-as-types/connectivities-as-types|Connectivities as types]]
  replaces string-tag `FieldOffset` declarations/provider keys and value-based
  dimension identity with typed declarations and nominal identity. That conflicts
  with the snapshot's `(name, kind)` dimension identity and current string-provider
  model. The note reports an implementation stack; this snapshot does not assert
  those changes are present at its baseline.
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
