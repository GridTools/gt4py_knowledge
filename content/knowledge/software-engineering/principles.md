---
title: Working Principles
description: "The operating synthesis of the software-engineering knowledge base — the rules the four books converge on, each tagged with its source."
tags: [software-design, principles, complexity, modularity, dry, domain-modelling, architecture, code-review, checklist, knowledge-base]
---

The operating synthesis of the knowledge base. Consult this when writing or reviewing
code; follow the source tags into the per-book notes when a rule needs its full
rationale. Tags: **(PoSD n)** = Ousterhout principle n / **(PoSD Ch n)** = chapter, in
[[knowledge/software-engineering/philosophy-of-software-design|A Philosophy of Software Design]];
**(Tip n)** = Pragmatic Programmer tip, in
[[knowledge/software-engineering/pragmatic-programmer|The Pragmatic Programmer]];
**(DDD Ch n)** in [[knowledge/software-engineering/domain-driven-design|Domain-Driven Design]];
**(Brooks Ch n)** in [[knowledge/software-engineering/design-of-design|The Design of Design]].

## 1. Prime directives

The four books converge on these; everything else elaborates them.

1. **Complexity is the enemy, and it is incremental.** Nothing that adds complexity is
   "too small to matter"; zero tolerance — fix broken windows when you see them.
   (PoSD 1; Tip 4)
2. **Working code isn't enough.** Program strategically: invest continually (~10–20% of
   effort) in design improvements instead of taking the fastest path that works.
   (PoSD 2, 3; Tip 47)
3. **The domain is the heart of the software.** Build a shared model with domain
   experts, speak it as one ubiquitous language, and bind model and implementation so
   each refines the other. (DDD Ch 1–3; Tip 17, 54)
4. **Design is iterative discovery, not plan execution.** Goals, requirements, and
   constraints are discovered while designing — the rational model is false. Use tracer
   bullets, prototypes, and early releases to find the target; expect and welcome model
   breakthroughs. (Brooks Ch 1–5; Tip 15, 16; DDD Ch 8)
5. **DRY.** Every piece of knowledge has one authoritative representation — in code,
   schema, docs, and build alike. (Tip 11; PoSD red flag 6)
6. **Design for reading, not writing.** If a reviewer or future reader must work hard,
   the design is wrong, whatever the effort saved at writing time. (PoSD 14, 16)
7. **You can't write perfect software — be paranoid.** Contracts, assertions, crashing
   early: a dead program does less damage than a crippled one. (Tip 30–33)
8. **There are no final decisions.** Keep decisions reversible; abstractions live longer
   than details. (Tip 14, 53)

## 2. Modules and interfaces

- Make modules **deep**: simple interface, powerful implementation. The interface is the
  cost, the functionality the benefit; a simple interface matters more than a simple
  implementation. (PoSD 4, 6)
- **Hide information.** Each module encapsulates knowledge that appears in its
  implementation but not its interface. Never leak a design decision across modules, and
  never decompose by order of execution — decompose by knowledge. (PoSD Ch 5)
- Make modules **somewhat general-purpose**: the implementation serves today's need; the
  interface serves a class of needs. Separate general-purpose from special-purpose code
  — specialization lives in higher layers. (PoSD 7, 8)
- **Different layer, different abstraction.** Adjacent layers with similar abstractions
  (pass-through methods, thin wrappers) signal a wrong decomposition. (PoSD 9)
- **Pull complexity downward.** It is better for the module developer to suffer than the
  module's users; avoid configuration parameters that export your problem to the caller.
  (PoSD 10)
- **Eliminate effects between unrelated things.** Write "shy" code: a method calls only
  its own object, its parameters, objects it creates, and its components (Law of
  Demeter). A requirement change should touch one module. (Tip 13, 36)
- Bring code together when it shares information or is always used together; separate
  general from special; split a method only if the pieces are independently
  understandable — conjoined methods are worse than one long one. (PoSD Ch 9)
- **Define errors out of existence:** redesign APIs so the exceptional case is normal
  (unset a missing variable is a no-op; deleting an open file just works). Mask or
  aggregate the exceptions that remain; the fewer places that must handle errors, the
  simpler the system. Reserve exceptions for the truly exceptional. (PoSD 11; Tip 34)
- **Design it twice.** Sketch at least two genuinely different decompositions before
  committing; the comparison teaches even when the first idea wins. (PoSD 12)
- **Put abstractions in code, details in metadata.** Business policy and technology
  choices are configuration, not engineering. Separate views from models. (Tip 37, 38, 42)
- The unit of development is an **abstraction, not a feature**: when a change needs a new
  abstraction, design the abstraction cleanly, don't bolt on the feature. (PoSD 15)

## 3. Domain modelling (tactical)

- Name every code element with the exact terms domain experts use; when the language
  changes in conversation, rename the code — a language change is a model change.
  Awkward phrasing in design discussion is a model smell. (DDD Ch 2)
- Keep a dedicated **domain layer** free of UI, application, and infrastructure
  concerns; keep the application layer thin — coordination only, no business rules.
  (DDD Ch 4)
- Classify every domain object: **ENTITY** (thread of identity), **VALUE OBJECT**
  (immutable descriptive whole, equality by attributes), or stateless **SERVICE** (an
  operation that belongs to no object). Never let services drain entities into an
  anemic model. Package by domain concept, not technical tier. (DDD Ch 5)
- Define an **AGGREGATE** boundary and root for each entity cluster: outside references
  point only to roots; invariants are enforced within the boundary at each commit;
  cross-aggregate consistency is asynchronous. Build with invariant-enforcing
  **FACTORIES**; access through one **REPOSITORY** per root that truly needs global
  access. (DDD Ch 6)
- **Make implicit concepts explicit.** When experts use a word with no counterpart in
  the code — or a rule hides in a conditional — reify it as a class, method, or
  **SPECIFICATION**. (DDD Ch 9)
- **Supple design:** intention-revealing names; command–query separation (queries have
  no side effects, commands return nothing); complex logic in immutable values with
  operations closed under their own type; explicit assertions/post-conditions; class
  boundaries along the domain's natural **conceptual contours**; intricate computation
  in standalone classes. (DDD Ch 10)
- **Refactor toward deeper insight:** refactor on model grounds, not just code smells;
  treat recurring "unexpected" requirement changes and un-killable bugs as symptoms of
  a wrong model. (DDD Ch 8, 13)

## 4. Architecture and strategy

- Total model unification of a large system is infeasible. Name each model's **BOUNDED
  CONTEXT**, draw the **CONTEXT MAP** as it actually is, and choose every inter-context
  relationship consciously (shared kernel ↔ customer/supplier ↔ conformist ↔
  anticorruption layer ↔ separate ways ↔ open host service). (DDD Ch 14)
- Integration is always expensive: first ask whether **SEPARATE WAYS** suffices; when
  integrating with a legacy or foreign model, translate through an **ANTICORRUPTION
  LAYER** rather than absorbing its representation. (DDD Ch 14)
- Identify the small **CORE DOMAIN** that differentiates the system; put the best
  long-term effort there; justify everything else by how it supports the core; offload
  **GENERIC SUBDOMAINS**. (DDD Ch 15)
- Large-scale structure is optional and must **evolve**: an ill-fitting structure is
  worse than none; keep it minimal and expressed in the ubiquitous language. Strategy
  must come from (or in tight feedback with) the application teams — no ivory tower.
  (DDD Ch 16–17)
- Guard **conceptual integrity** — unity, economy, clarity. It comes from one empowered
  chief designer (or a two-person team) with genuine design authority, not from
  committee negotiation; talk about the design concept itself, not its derivative
  documents. (Brooks Ch 1, 6)
- The hardest part of design is deciding **what** to design: help the client discover
  what they want; set goals iteratively. Use the rational-model vocabulary (goal,
  desiderata, constraints, budgeted resource) for communication, not as a process.
  (Brooks Ch 1–3)
- **Requirements need weights and an advocate.** Committee-gathered requirements accrete
  into an obese, unweighted wish list with nobody speaking for the product as a whole;
  rank goals, and contract for design separately from construction with explicit
  re-contracting points. (Brooks Ch 4–5)
- Make the **user model explicit — better wrong than vague.** Write down who the users
  are, what they know, and what they are trying to do; an articulated guess can be
  corrected, an unspoken assumption cannot. (Brooks Ch 9)
- Identify the actual **budgeted resource** (rarely money — latency, bytes, schedule,
  attention), track it publicly, and let one person control it; it can switch
  mid-project. List constraints explicitly up front so you notice when one disappears —
  sometimes the breakthrough is removing a constraint, not designing around it.
  (Brooks Ch 3, 10–11)
- **Study exemplars before designing** — originality is no excuse for ignorance.
  Specify properties, not implementations; grow a new design from a complete exemplar by
  progressive truthfulness, every intermediate state complete and consistent (the design
  analogue of tracer bullets). (Brooks Ch 13, 17; Tip 15)
- **Great designs come from great designers, not process.** Process raises the floor but
  suppresses the ceiling — keep it out of conception, leave room for bold decisions
  (they account for most of the goodness of outcomes), and record the rationale — the
  whys — as you design. (Brooks Ch 16, 19–20, Part VI)
- **Stay in contact with use and implementation.** The besetting mistake of expert
  designers is confidently designing the wrong thing; design divorced from its users
  and its builders rots. (Brooks Ch 14–15; DDD Ch 3)

## 5. Construction and daily practice

- **Names:** precise, consistent, formed from the ubiquitous language; if a precise name
  is hard to find, the design is unclear. (PoSD Ch 14; DDD Ch 2)
- **Comments** describe what is not obvious from the code — interface comments for
  users (no implementation detail), precision comments for units/invariants/ownership,
  why-comments for rationale. Write them first, as a design tool: a hard-to-describe
  entity is a badly designed entity. (PoSD 13, Ch 12–15)
- **Obvious code:** judged by the reader, not the writer. Consistency — in names,
  style, interfaces, invariants — is leverage; don't "improve" a convention without
  changing it everywhere. (PoSD Ch 17–18)
- **Don't program by coincidence.** Rely only on documented behavior; prove assumptions
  with real data and boundary conditions; understand generated/wizard code before
  incorporating it; assume the bug is in your code, not the compiler. (Tip 26, 27, 44, 50)
- **Design by contract:** preconditions, postconditions, invariants; assert the
  impossible (and leave assertions on); crash early rather than limp; the allocator of
  a resource deallocates it. (Tip 31–33, 35)
- **Test ruthlessly and automatically:** design to test; test against the contract;
  tests run with every build; state coverage over code coverage; once a human finds a
  bug, a test finds it forever after. Coding ain't done 'til all the tests run.
  (Tip 48, 49, 62–66)
- **Automate everything repeatable:** builds, generation, checks — a script does the
  same thing in the same order every time; keep knowledge in plain text under version
  control; write code that writes code. (Tip 20, 23, 29, 61)
- **Refactor early, refactor often** — but don't mix refactoring with behavior change;
  when modifying existing code, leave the design better than you found it, never
  "just make it work". (Tip 47; PoSD Ch 16)
- **Estimate** before you start (schedule and algorithmic cost alike), then measure
  against reality and iterate the schedule with the code. (Tip 18, 19, 45, 46)
- When stuck on an "impossible" problem, don't think outside the box — **find the box**:
  enumerate the real constraints and ask "does it have to be done this way? at all?"
  (Tip 55)
- **Requirements are needs, not policy or UI.** Dig for them, work with a user, keep
  them abstract, maintain a project glossary. (Tip 51–54)

## 6. Red-flag checklist

Stop and reconsider the design when you see:

- **Shallow module** — interface barely simpler than implementation. (PoSD)
- **Information leakage** — one design decision reflected in several modules. (PoSD)
- **Temporal decomposition** — structure follows execution order, not knowledge. (PoSD)
- **Overexposure** — common use forces awareness of rare features. (PoSD)
- **Pass-through method** — same signature relayed one level down. (PoSD)
- **Repetition** — nontrivial code repeated; knowledge represented twice. (PoSD; Tip 11)
- **Special–general mixture** — special-purpose code entangled with general. (PoSD)
- **Conjoined methods** — can't understand one without the other. (PoSD)
- **Comment repeats code / implementation leaks into interface docs.** (PoSD)
- **Vague name / hard to pick name / hard to describe** — the design, not the
  vocabulary, is at fault. (PoSD)
- **Nonobvious code** — meaning not readily understood; needs redesign, not comments
  alone. (PoSD)
- **Broken windows** — known-bad code left standing, licensing further decay. (Tip 4)
- **Train wrecks** — `a.getB().getC().doD()`: coupling to structure you don't own. (Tip 36)
- **Coincidental correctness** — it works but nobody can say why. (Tip 44)
- **Manual procedure** — a human repeating what a script should do. (Tip 61)
- **Anemic domain model** — entities as data bags, logic drained into services. (DDD Ch 5)
- **Language drift** — code vocabulary diverging from the experts' speech; experts
  can't follow the model's core. (DDD Ch 2–3)
- **Buried rule** — a business rule expressed only as a conditional, not a named
  concept. (DDD Ch 9)
- **Unmapped context** — models blending across an undeclared boundary; translation
  happening ad hoc. (DDD Ch 14)
- **Wrong-model churn** — "unexpected" requirement changes recurring in the same spot.
  (DDD Ch 13)
- **Unexamined constraint** — designing around a constraint nobody has re-validated.
  (Brooks Ch 3, 11)
- **Advocate-less wish list** — requirements accreted by committee, unweighted, with
  nobody advocating for the product as a whole. (Brooks Ch 4)
- **Vague user model** — design proceeding on unspoken assumptions about the users
  instead of an explicit, falsifiable one. (Brooks Ch 9)
