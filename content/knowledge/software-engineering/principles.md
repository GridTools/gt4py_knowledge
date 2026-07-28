---
title: Working Principles
description: "Distilled software design and engineering practice — from prime directives on complexity and modularity through domain modelling, architecture, and daily craft, down to a red-flag checklist."
tags: [software-design, principles, complexity, modularity, dry, domain-modelling, architecture, code-review, checklist]
---

Distilled software design and engineering practice: the rules worth applying when
writing or reviewing code, from the prime directives down to a red-flag checklist you
can run over a diff.

## 1. Prime directives

Everything below elaborates these.

1. **Complexity is the enemy, and it is incremental.** Nothing that adds complexity is
   "too small to matter"; zero tolerance — fix broken windows when you see them.
2. **Working code isn't enough.** Program strategically: invest continually (~10–20% of
   effort) in design improvements instead of taking the fastest path that works.
3. **The domain is the heart of the software.** Build a shared model with domain
   experts, speak it as one ubiquitous language, and bind model and implementation so
   each refines the other.
4. **Design is iterative discovery, not plan execution.** Goals, requirements, and
   constraints are discovered while designing; you never start from a fixed goal and
   search a known space of alternatives. Use tracer
   bullets, prototypes, and early releases to find the target; expect and welcome model
   breakthroughs.
5. **DRY.** Every piece of knowledge has one authoritative representation — in code,
   schema, docs, and build alike.
6. **Design for reading, not writing.** If a reviewer or future reader must work hard,
   the design is wrong, whatever the effort saved at writing time.
7. **You can't write perfect software — be paranoid.** Contracts, assertions, crashing
   early: a dead program does less damage than a crippled one.
8. **There are no final decisions.** Keep decisions reversible; abstractions live longer
   than details.

## 2. Modules and interfaces

- Make modules **deep**: simple interface, powerful implementation. The interface is the
  cost, the functionality the benefit; a simple interface matters more than a simple
  implementation.
- **Hide information.** Each module encapsulates knowledge that appears in its
  implementation but not its interface. Never leak a design decision across modules, and
  never decompose by order of execution — decompose by knowledge.
- Make modules **somewhat general-purpose**: the implementation serves today's need; the
  interface serves a class of needs. Separate general-purpose from special-purpose code
  — specialization lives in higher layers.
- **Different layer, different abstraction.** Adjacent layers with similar abstractions
  (pass-through methods, thin wrappers) signal a wrong decomposition.
- **Pull complexity downward.** It is better for the module developer to suffer than the
  module's users; avoid configuration parameters that export your problem to the caller.
- **Eliminate effects between unrelated things.** Write "shy" code: a method calls only
  its own object, its parameters, objects it creates, and its components (Law of
  Demeter). A requirement change should touch one module.
- Bring code together when it shares information or is always used together; separate
  general from special; split a method only if the pieces are independently
  understandable — conjoined methods are worse than one long one.
- **Define errors out of existence:** redesign APIs so the exceptional case is normal
  (unset a missing variable is a no-op; deleting an open file just works). Mask or
  aggregate the exceptions that remain; the fewer places that must handle errors, the
  simpler the system. Reserve exceptions for the truly exceptional.
- **Design it twice.** Sketch at least two genuinely different decompositions before
  committing; the comparison teaches even when the first idea wins.
- **Put abstractions in code, details in metadata.** Business policy and technology
  choices are configuration, not engineering. Separate views from models.
- The unit of development is an **abstraction, not a feature**: when a change needs a new
  abstraction, design the abstraction cleanly, don't bolt on the feature.

## 3. Domain modelling (tactical)

- Name every code element with the exact terms domain experts use; when the language
  changes in conversation, rename the code — a language change is a model change.
  Awkward phrasing in design discussion is a model smell.
- Keep a dedicated **domain layer** free of UI, application, and infrastructure
  concerns; keep the application layer thin — coordination only, no business rules.
- Classify every domain object: **ENTITY** (thread of identity), **VALUE OBJECT**
  (immutable descriptive whole, equality by attributes), or stateless **SERVICE** (an
  operation that belongs to no object). Never let services drain entities into an
  anemic model. Package by domain concept, not technical tier.
- Define an **AGGREGATE** boundary and root for each entity cluster: outside references
  point only to roots; invariants are enforced within the boundary at each commit;
  cross-aggregate consistency is asynchronous. Build with invariant-enforcing
  **FACTORIES**; access through one **REPOSITORY** per root that truly needs global
  access.
- **Make implicit concepts explicit.** When experts use a word with no counterpart in
  the code — or a rule hides in a conditional — reify it as a class, method, or
  **SPECIFICATION**.
- **Supple design:** intention-revealing names; command–query separation (queries have
  no side effects, commands return nothing); complex logic in immutable values with
  operations closed under their own type; explicit assertions/post-conditions; class
  boundaries along the domain's natural **conceptual contours**; intricate computation
  in standalone classes.
- **Refactor toward deeper insight:** refactor on model grounds, not just code smells;
  treat recurring "unexpected" requirement changes and un-killable bugs as symptoms of
  a wrong model.

## 4. Architecture and strategy

- Total model unification of a large system is infeasible. Name each model's **BOUNDED
  CONTEXT**, draw the **CONTEXT MAP** as it actually is, and choose every inter-context
  relationship consciously (shared kernel ↔ customer/supplier ↔ conformist ↔
  anticorruption layer ↔ separate ways ↔ open host service).
- Integration is always expensive: first ask whether **SEPARATE WAYS** suffices; when
  integrating with a legacy or foreign model, translate through an **ANTICORRUPTION
  LAYER** rather than absorbing its representation.
- Identify the small **CORE DOMAIN** that differentiates the system; put the best
  long-term effort there; justify everything else by how it supports the core; offload
  **GENERIC SUBDOMAINS**.
- Large-scale structure is optional and must **evolve**: an ill-fitting structure is
  worse than none; keep it minimal and expressed in the ubiquitous language. Strategy
  must come from (or in tight feedback with) the application teams — no ivory tower.
- Guard **conceptual integrity** — unity, economy, clarity. It comes from one empowered
  chief designer (or a two-person team) with genuine design authority, not from
  committee negotiation; talk about the design concept itself, not its derivative
  documents.
- The hardest part of design is deciding **what** to design: help the client discover
  what they want; set goals iteratively. Use the formal design vocabulary (goal,
  desiderata, constraints, budgeted resource) for communication, not as a process.
- **Requirements need weights and an advocate.** Committee-gathered requirements accrete
  into an obese, unweighted wish list with nobody speaking for the product as a whole;
  rank goals, and contract for design separately from construction with explicit
  re-contracting points.
- Make the **user model explicit — better wrong than vague.** Write down who the users
  are, what they know, and what they are trying to do; an articulated guess can be
  corrected, an unspoken assumption cannot.
- Identify the actual **budgeted resource** (rarely money — latency, bytes, schedule,
  attention), track it publicly, and let one person control it; it can switch
  mid-project. List constraints explicitly up front so you notice when one disappears —
  sometimes the breakthrough is removing a constraint, not designing around it.
- **Study exemplars before designing** — originality is no excuse for ignorance.
  Specify properties, not implementations; grow a new design from a complete exemplar by
  progressive truthfulness, every intermediate state complete and consistent (the design
  analogue of tracer bullets).
- **Great designs come from great designers, not process.** Process raises the floor but
  suppresses the ceiling — keep it out of conception, leave room for bold decisions
  (they account for most of the goodness of outcomes), and record the rationale — the
  whys — as you design.
- **Stay in contact with use and implementation.** The besetting mistake of expert
  designers is confidently designing the wrong thing; design divorced from its users
  and its builders rots.

## 5. Construction and daily practice

- **Names:** precise, consistent, formed from the ubiquitous language; if a precise name
  is hard to find, the design is unclear.
- **Comments** describe what is not obvious from the code — interface comments for
  users (no implementation detail), precision comments for units/invariants/ownership,
  why-comments for rationale. Write them first, as a design tool: a hard-to-describe
  entity is a badly designed entity.
- **Obvious code:** judged by the reader, not the writer. Consistency — in names,
  style, interfaces, invariants — is leverage; don't "improve" a convention without
  changing it everywhere.
- **Don't program by coincidence.** Rely only on documented behavior; prove assumptions
  with real data and boundary conditions; understand generated/wizard code before
  incorporating it; assume the bug is in your code, not the compiler.
- **Design by contract:** preconditions, postconditions, invariants; assert the
  impossible (and leave assertions on); crash early rather than limp; the allocator of
  a resource deallocates it.
- **Test ruthlessly and automatically:** design to test; test against the contract;
  tests run with every build; state coverage over code coverage; once a human finds a
  bug, a test finds it forever after. Coding ain't done 'til all the tests run.
- **Automate everything repeatable:** builds, generation, checks — a script does the
  same thing in the same order every time; keep knowledge in plain text under version
  control; write code that writes code.
- **Refactor early, refactor often** — but don't mix refactoring with behavior change;
  when modifying existing code, leave the design better than you found it, never
  "just make it work".
- **Estimate** before you start (schedule and algorithmic cost alike), then measure
  against reality and iterate the schedule with the code.
- When stuck on an "impossible" problem, don't think outside the box — **find the box**:
  enumerate the real constraints and ask "does it have to be done this way? at all?"
- **Requirements are needs, not policy or UI.** Dig for them, work with a user, keep
  them abstract, maintain a project glossary.

## 6. Red-flag checklist

Stop and reconsider the design when you see:

- **Shallow module** — interface barely simpler than implementation.
- **Information leakage** — one design decision reflected in several modules.
- **Temporal decomposition** — structure follows execution order, not knowledge.
- **Overexposure** — common use forces awareness of rare features.
- **Pass-through method** — same signature relayed one level down.
- **Repetition** — nontrivial code repeated; knowledge represented twice.
- **Special–general mixture** — special-purpose code entangled with general.
- **Conjoined methods** — can't understand one without the other.
- **Comment repeats code / implementation leaks into interface docs.**
- **Vague name / hard to pick name / hard to describe** — the design, not the
  vocabulary, is at fault.
- **Nonobvious code** — meaning not readily understood; needs redesign, not comments
  alone.
- **Broken windows** — known-bad code left standing, licensing further decay.
- **Train wrecks** — `a.getB().getC().doD()`: coupling to structure you don't own.
- **Coincidental correctness** — it works but nobody can say why.
- **Manual procedure** — a human repeating what a script should do.
- **Anemic domain model** — entities as data bags, logic drained into services.
- **Language drift** — code vocabulary diverging from the experts' speech; experts
  can't follow the model's core.
- **Buried rule** — a business rule expressed only as a conditional, not a named
  concept.
- **Unmapped context** — models blending across an undeclared boundary; translation
  happening ad hoc.
- **Wrong-model churn** — "unexpected" requirement changes recurring in the same spot.
- **Unexamined constraint** — designing around a constraint nobody has re-validated.
- **Advocate-less wish list** — requirements accreted by committee, unweighted, with
  nobody advocating for the product as a whole.
- **Vague user model** — design proceeding on unspoken assumptions about the users
  instead of an explicit, falsifiable one.
