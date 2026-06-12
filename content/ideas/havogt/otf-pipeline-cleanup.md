---
title: OTF pipeline cleanup (incremental improvements)
author: havogt
tags: [otf, workflow, compilation, caching, fingerprint, backend, factory_boy, dace, gtfn, cleanup, tech-debt]
created: 2026-06-12
---

> **TL;DR** A set of independently-landable cleanups to the existing
> `gt4py.next.otf` compilation framework — one canonical fingerprint, a
> sanctioned stage-inspection/dump API, cache-miss logging, replacing
> `factory_boy` backend config with plain code, honest combinator typing, a
> tightened argument model, and a naming pass — each worth doing **on its own**,
> none of them blocking (or blocked by) the larger staged-API redesign.

> **Context**: this is the §6 "Incremental improvements" track extracted from
> the OTF pipeline redesign proposal that lives in the gt4py source tree at
> [`docs/development/proposals/otf-pipeline-redesign.md`](https://github.com/graitools/gt4py/blob/claude/wizardly-allen-00cgi0/docs/development/proposals/otf-pipeline-redesign.md)
> (with the prior-art survey in the companion
> [`otf-pipeline-redesign-prior-art.md`](https://github.com/graitools/gt4py/blob/claude/wizardly-allen-00cgi0/docs/development/proposals/otf-pipeline-redesign-prior-art.md)).
> That document proposes both these cleanups **and** a full redesign towards an
> explicit staged compilation API; this note carries the cleanup half on its own
> so it can be tracked, cross-checked, and picked up incrementally without
> committing to the redesign.

---

## Problem / motivation

GT4Py's compiled backends are driven by `gt4py.next.otf`: a generic,
statically-typed function-composition framework (`Workflow[StartT, EndT]`,
`NamedStepSequence`, `CachedStep`, …) that pipes a program definition from DSL
AST to a compiled Python extension. The design (ADR 0011) met its original goal
— backends share the binding, build-system, and import stages instead of being
monoliths — but a review of the code and of how each backend actually uses the
framework surfaced concrete friction that does **not** require a redesign to
fix:

- The promised static type safety is largely illusory.
- Configuration goes through a second framework (`factory_boy`) with
  stringly-typed `__`-path overrides.
- Caching is spread over four uncoordinated layers with three different key
  derivations (including `id()`-based offset-provider hashing).
- There is no sanctioned way to inspect or re-run intermediate stages —
  consumers (e.g. `dace/program.py`) resort to duck-typed unwrapping of workflow
  internals.

These are maintenance and extensibility pains ("hard to follow, hard to
extend") that can be paid down in place. The cleanups below remove the worst of
them while leaving the door open to the bigger redesign.

## Proposal

Seven incremental improvements, ordered by value/effort (as in §6 of the source
proposal). Each is independently landable.

1. **Single fingerprint function** *(small, high value)*. Define one canonical
   `fingerprint(program_def) -> str` (content-based: GTIR fingerprint, arg
   types, offset-provider *types*, backend-relevant config, gt4py version) and
   use it for the translation cache, the executor `CachedStep`, and — via a
   `fingerprint → build dir` index — the build cache, so warm starts skip
   translation entirely. Replace `id()`-based offset-provider hashing with
   content hashing of the type plus an identity fast path.
2. **`inspect`/dump API + stage instrumentation** *(small, high value)*. Add an
   optional observer to `NamedStepSequence.__call__`/`MultiWorkflow`
   (`on_stage(name, artifact)`) plus a `GT4PY_DUMP_STAGES=<dir>` config that
   writes each artifact (GTIR text, generated source, SDFG JSON) per program
   (MLIR `print-ir-tree-dir`-style). Expose
   `Backend.lower(definition, compile_time_args) -> stage artifacts` as a
   supported method so `dace/program.py` can stop unwrapping workflow internals.
3. **`explain_cache_misses` logging** *(small)*. Each cache layer logs its key,
   hit/miss, and the first differing key component under a debug flag.
4. **Replace `factory_boy` with plain code** *(medium, high value)*. A backend
   becomes a frozen dataclass / `make_*_backend()` function with keyword
   arguments — the existing `make_dace_backend` wrapper already proves the need.
   This removes the trait duplication, the `__`-path strings, and the
   `type: ignore`s, and lets the shared "cached translation", naming, and device
   wiring be deduplicated into one helper used by gtfn and dace. (Supersedes
   part of ADR 0017; config stays in `gt4py.next.config`.)
5. **Honest typing for the combinators** *(small)*. Drop the unused combinator
   generality (`SkippableStep`, rarely-used `StepSequence.chain` chains);
   document `NamedStepSequence` as runtime-typed; declare step order explicitly
   (a class-level tuple) instead of via type-annotation reflection.
6. **Fix the argument model edges** *(medium)*. Split `CompileTimeArgs` into a
   true compile-time part (types, descriptors, offset-provider *types*) and an
   explicitly-named bridge for passes that still need runtime tables; remove dead
   `column_axis`; restrict `eval`/`exec` codegen to the measured hot path
   (`_argument_descriptor_cache_key_from_args`) and use ordinary code elsewhere.
7. **Align names with the ADR (or the ADR with the names)** *(small)*. One
   rename pass (`Backend` → `Toolchain`, `executor` → `compile_pipeline`,
   docstring stage names) plus a short successor ADR recording the changes.

Items 1–3 directly remove the two pains that triggered the parent proposal
(hard to follow, hard to extend) at low risk. Items 4–7 reduce the number of
concepts contributors must hold in their heads.

## Alternatives considered

- **The full redesign** — replace the generic workflow framework with an
  explicit staged compilation API (typed stage objects
  `Program → LoweredProgram → CompiledArtifact → Executable`, a two-method
  backend protocol, one fingerprint, one artifact store, instrumentation at
  every stage boundary). That is the §7 track of the source proposal. It is a
  larger commitment; these cleanups are deliberately the lower-risk path and
  are valuable even if the redesign is rejected. Notably items 1 and 2 build
  foundations (one fingerprint, stage instrumentation) the redesign would reuse,
  so the tracks are complementary rather than mutually exclusive.
- **Do nothing / wait for the redesign** — leaves the `id()`-keyed caching,
  `factory_boy` override strings, and internal-unwrapping in place indefinitely;
  the parent proposal's recommendation is explicitly to do items 1–3 *now*
  regardless of the redesign decision.

## Open questions / conflicts

- **Relation to the redesign.** This note and the staged-API redesign both live
  in the same source document; they should be tracked together so the cleanup
  work doesn't accidentally re-entrench abstractions the redesign would remove.
  When prioritising, prefer cleanups that the redesign would also want
  (fingerprint, stage instrumentation).
- **ADR impact.** Items 4 and 7 touch ADR 0017 (toolchain configuration) and
  imply a short successor ADR in the gt4py repo. Anyone picking these up should
  record the change there rather than only here.
- No conflicts with other notes currently in this garden.
