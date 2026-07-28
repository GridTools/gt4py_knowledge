---
title: Domain-Driven Design — Eric Evans (2003)
description: "Book notes: build a shared domain model with the experts, speak it as a ubiquitous language, bind it to the implementation, and protect it with bounded contexts."
tags: [domain-modelling, ubiquitous-language, model-driven-design, entities, value-objects, aggregates, repositories, factories, bounded-context, strategic-design, core-domain, refactoring, supple-design, book-notes]
---

The heart of software is its ability to solve domain-related problems for its user, so
the most demanding complexity to tame is the domain itself, not the technology. The
book's discipline: build a shared, rigorous model of the domain with the domain experts;
speak it as a UBIQUITOUS LANGUAGE in code, diagrams, and conversation; bind the model
directly to the implementation so that refining one refines the other; and keep
refactoring toward deeper insight until the design becomes supple. At scale, protect
model integrity by drawing explicit BOUNDED CONTEXTs and choosing every inter-context
relationship deliberately, and focus the best effort on a distilled CORE DOMAIN. Pattern
names below are the book's own vocabulary (SMALL CAPS); each entry gives the problem,
the prescription, and a clarifying example.

Structure: Part I — Putting the Domain Model to Work (Ch 1–3); Part II — The Building
Blocks of a Model-Driven Design (Ch 4–7); Part III — Refactoring Toward Deeper Insight
(Ch 8–13); Part IV — Strategic Design (Ch 14–17).

## Ch 1 — Crunching Knowledge

Effective domain models are produced by "knowledge crunching": developers and domain experts collaboratively sifting a torrent
of information, trying and discarding organizing ideas until a small set of abstract concepts makes sense of the mass.
The model is not gathered from experts and handed to programmers; it is distilled jointly, disciplined by a feedback loop
through running code.

- Ingredients of effective modeling (from the PCB net-routing project, where a developer with no electronics
  background built a probe-simulation tool together with circuit designers):
  1. Bind model and implementation early, via a crude prototype, and maintain the bond through every iteration.
  2. Cultivate a language based on the model, so anyone can compose sentences from model terms and be understood
     without translation.
  3. Develop a knowledge-rich model — objects with behavior and enforced rules, not just a data schema.
  4. Distill: drop concepts that are not useful or central; when a needed concept is entangled with an unneeded one,
     find a new model that separates them.
  5. Brainstorm and experiment — spoken scenarios are a laboratory of the model; the ear quickly detects clarity or
     awkwardness of expression.
- Waterfall fails because knowledge flows one way (expert → analyst → programmer) and never accumulates. Iteration
  without abstraction also fails: teams build feature after feature but never learn the principles behind them, so
  powerful new features never unfold as corollaries of older ones.
- All projects leak knowledge: people leave, teams reorganize, outsourced code arrives without understanding. Unless
  code and documents express hard-earned knowledge in usable form, it is lost when the oral tradition breaks;
  productive teams practice continuous learning and grow a stable core of self-educated members.
- Knowledge-rich design goes beyond "find the nouns": business activities and rules are as central as entities, and
  making rules explicit forces the team to clarify, reconcile, or scope out inconsistencies experts navigate
  unconsciously. Example: a 10% overbooking rule buried as a guard clause (`voyage.capacity() * 1.1`) in a booking
  method is invisible to business experts; an explicit `OverbookingPolicy` class (a policy — STRATEGY) makes the
  rule a distinct, verifiable business concept and closes the feedback loop with experts.
- Deep models rarely lie on the surface. On the container-shipping project, months of crunching shifted the model
  from "moving containers from place to place" to "transfers of responsibility for cargo between parties" (carriers,
  bills of lading, release of payments) — the Itinerary object stayed, but the model changed profoundly. You cannot
  know in advance where crunching will end up; even dropped early models (the probe simulation) start the process:
  shared language, team learning, and the code feedback loop.

**Apply:**
- Model with domain experts in live sessions; change the model whenever it cannot carry an important scenario.
- Make every business rule an explicit, named element (e.g. a policy object), never a buried conditional.
- Prototype domain behavior early, test-driven, with no UI or persistence, to close the feedback loop.
- Distill continuously: delete model concepts that stop pulling their weight.
- Expect deep model shifts; do not freeze the first workable abstraction.

## Ch 2 — Communication and the Use of Language

A project needs one language, built on the domain model, used by everyone in speech, diagrams, documents, and code.
Translation between a business dialect and a technical dialect blunts communication, hides model schisms, and makes
knowledge crunching anemic.

**UBIQUITOUS LANGUAGE** —
- Problem: domain experts use their jargon while technical team members have their own language tuned for design;
  the terminology of day-to-day discussion is disconnected from the terminology in the code; translation is inexact,
  bilingual members become bottlenecks, and schisms form unnoticed between team members using terms differently.
- Therefore: Use the model as the backbone of a language. Commit the team to exercising that language relentlessly
  in all communication within the team and in the code. Use the same language in diagrams, writing, and especially
  speech. Iron out difficulties by experimenting with alternative expressions, which reflect alternative models.
  Then refactor the code, renaming classes, methods, and modules to conform to the new model. Resolve confusion
  over terms in conversation, just as we come to agree on the meaning of ordinary words. Recognize that a change
  in the UBIQUITOUS LANGUAGE is a change to the model. Domain experts should object to terms or structures that
  are awkward or inadequate to convey domain understanding; developers should watch for ambiguity or inconsistency
  that will trip up design.
- Example: two cargo-router dialogs — one in tables/rows/Booleans, one where "a Routing Service finds an Itinerary
  that satisfies a Route Specification." The second is concise, precise, and intelligible to user and developer alike.
- Addendum (modeling out loud): Play with the model as you talk about the system. Describe scenarios out loud using
  the elements and interactions of the model, combining concepts in ways allowed by the model. Find easier ways to
  say what you need to say, and take those new ideas back down to the diagrams and code. Rough edges are easy to hear.

- One team, one language: do not "shield" business experts from the model — if sophisticated domain experts don't
  understand the model, something is wrong with the model. Technical and business jargons persist as extensions,
  never as alternative vocabularies for the same domain concepts. Write requirements, use cases, and acceptance
  tests in the model's language.
- Diagrams are communication aids, not the model. Sketch three to five objects central to the issue; comprehensive
  UML diagrams are simultaneously too complete and too incomplete — they cannot state the meaning of concepts or
  what objects are meant to do. Prefer a text document illustrated with selective, simplified diagrams over a
  diagram annotated with text. The model is not the diagram.
- Documents should complement code and speech, not restate what code already specifies exactly. A document must work
  for a living: if its terms don't show up in conversations and code — if it has no impact on the UBIQUITOUS
  LANGUAGE — it is not fulfilling its purpose; archive it rather than maintain it by sheer discipline. Code is the
  closest document to the ground, but its message (names, structure) is not guaranteed accurate; writing code that
  says the right thing as well as does the right thing takes discipline.
- Explanatory models: one model underlies implementation, design, and team communication, but additional
  non-technical views (pictures, metaphors, deliberately non-UML drawings) may be used purely to teach the domain,
  kept visibly distinct from the design model — e.g. a hand-drawn timeline of port operations and cargo at rest
  explaining the shipping-route class diagram.

**Apply:**
- Name code elements with the exact terms domain experts use; when a term changes in conversation, rename the code
  — a language change is a model change.
- Treat awkward phrasing in design talk as a model smell; experiment aloud with alternatives before changing code.
- Reject designs whose core the domain experts cannot follow.
- Keep diagrams minimal and selective; put durable detail in code plus a short text narrative.
- Delete or archive any document whose vocabulary no longer appears in speech or code.

## Ch 3 — Binding Model and Implementation

A model that is not tightly bound to the running code is worthless, however "correct" it is — and code without a model
behind it does useful things without explaining its actions. Analysis and design must not be separate models, and
modelers must not be separate people from programmers.

**MODEL-DRIVEN DESIGN** —
- Problem: methodologies that keep an analysis model distinct from the design lose the crunched knowledge at handoff;
  developers are forced to invent new abstractions, complex mappings between model and design cannot be maintained,
  and a deadly divide opens where insight gained in each activity never feeds the other. (Cautionary tales: a
  wall-sized, "correct" analysis diagram abandoned by developers for an ad hoc design; the end product was
  indistinguishable from a project that never modeled at all.)
- Therefore: Design a portion of the software system to reflect the domain model in a very literal way, so that
  mapping is obvious. Revisit the model and modify it to be implemented more naturally in software, even as you seek
  to make it reflect deeper insight into the domain. Demand a single model that serves both purposes well, in
  addition to supporting a robust UBIQUITOUS LANGUAGE. Draw from the model the terminology used in the design and
  the basic assignment of responsibilities. The code becomes an expression of the model, so a change to the code may
  be a change to the model; its effect must ripple through the rest of the project's activities. When a model seems
  impractical to implement, or fails to express the key domain concepts, search for a new one that does both.
- Requires a modeling paradigm with tool support: object-oriented programming fits because objects directly implement
  model constructs; purely procedural languages do not — the program becomes data manipulation capturing little meaning.
- Example: PCB layout rules. Engineers' scripts inferred "bus" by sorting net-name files — the concept existed only
  implicitly in string matches. An explicit Bus/Net model (assignRule, assignedRules) made the core functionality
  almost trivial, unit-testable, and extensible to the 20+ real operations, where scripts restarted per file format.

- Letting the bones show: the user's mental model and the implementation model should also be one. Internet Explorer
  "Favorites" pretend to be a list of site names but are actually files, so filename restrictions surface as baffling
  errors. Either reveal the underlying model to the user or change the implementation to match the illusion; a design
  based on a model reflecting the domain can expose its bones, yielding consistent, predictable behavior.

**HANDS-ON MODELERS** —
- Problem: separating modeling from programming (the manufacturing metaphor: skilled engineers design, laborers
  assemble) kills MODEL-DRIVEN DESIGN. Model intent is lost in handoff; implementation feedback never reaches the
  modeler, so models turn impractical (a hands-off architect's model was reduced to a mere data structure). If the
  people who write code do not feel responsible for the model, the model has nothing to do with the software.
  Programmers are modelers, whether anyone likes it or not.
- Therefore: Any technical person contributing to the model must spend some time touching the code, whatever primary
  role he or she plays on the project. Anyone responsible for changing code must learn to express a model through
  the code. Every developer must be involved in some level of discussion about the model and have contact with
  domain experts. Those who contribute in different ways must consciously engage those who touch the code in a
  dynamic exchange of model ideas through the UBIQUITOUS LANGUAGE.

**Apply:**
- Maintain one model per part of the system, serving analysis and design; if implementation strains, change the model.
- Make every class and operation play a conceptual role in the model; reject "fancy data structures" plus procedures.
- Treat code changes as model changes: propagate renames and restructurings to diagrams, documents, vocabulary.
- Never let anyone model without touching code, or code without contact with the model and domain experts.
- Align the user-visible model with the implementation model; don't paper over it with an imperfect illusion.

## Ch 4 — Isolating the Domain

Domain code is a small fraction of a system but carries its value; unless it is decoupled from UI, persistence, and
other technical concerns, model-driven design is impractical — the model's elements cannot be seen as a system when
diffused through the mass of the program.

**LAYERED ARCHITECTURE** —
- Problem: in typical OO programs, UI, database, and support code get written directly into business objects, and
  business logic leaks into widgets and database scripts, because that is the short-run easy path. Then business
  rules can only be changed by meticulously tracing UI and database code, automated testing is awkward, and
  coherent, model-driven objects become impractical.
- Therefore: Partition a complex program into layers. Develop a design within each layer that is cohesive and that
  depends only on the layers below. Follow standard architectural patterns to provide loose coupling to the layers
  above. Concentrate all the code related to the domain model in one layer and isolate it from the user interface,
  application, and infrastructure code. The domain objects, free of the responsibility of displaying themselves,
  storing themselves, and managing application tasks, can be focused on expressing the domain model.
- The four conventional layers:
  - User Interface (Presentation): shows information, interprets user (or other-system) commands.
  - Application: defines the jobs the software does and directs expressive domain objects to work out problems.
    Kept thin: no business rules or business state; only task coordination and task-progress state.
  - Domain (Model): represents business concepts, business-situation state, and business rules. The heart of
    business software.
  - Infrastructure: generic technical capabilities — messaging, persistence, drawing widgets, frameworks.
- Example: online-banking funds transfer — the rule "every credit has a matching debit" belongs to the domain layer,
  not the application layer; the UI could be replaced by an XML wire request without touching lower layers.
- Relating layers: dependencies point downward only; lower layers communicate upward through callbacks or OBSERVERS
  (MVC and descendants for the UI). Infrastructure typically offers capabilities as SERVICES: the application knows
  *when* to send a message, not *how*.
- Architectural frameworks: apply selectively to hard problems, not one-size-fits-all (early J2EE "every domain
  object is an entity bean" bogged down performance and development); minimalism keeps business objects expressive.
- DDD requires only one layer to exist: the isolated domain layer, where the model lives. Isolating the domain
  implementation is a prerequisite for domain-driven design.

**SMART UI (anti-pattern)** —
- Context/problem: a project delivering simple functionality, dominated by data entry and display, with few business
  rules, staffed without advanced object modelers. For such projects, model-driven design with layers imposes a
  learning curve and overhead that will sink them before delivery.
- Therefore, when circumstances warrant: Put all the business logic into the user interface. Chop the application
  into small functions and implement them as separate user interfaces, embedding the business rules into them. Use a
  relational database as a shared repository of the data. Use the most automated UI building and visual programming
  tools available.
- Advantages: immediate productivity; little training needed; prototypes-into-production; small modules with
  predictable schedules; 4GL tools work well. Disadvantages: integration only through the database; no reuse or
  abstraction of business logic (rules duplicated per operation); refactoring hits a ceiling; complexity buries you
  — no graceful path to richer behavior.
- It is a mutually exclusive fork: you cannot migrate out except by replacing whole applications, so choose
  consciously. A team committed to MODEL-DRIVEN DESIGN must isolate a domain layer from the first iteration.
  Bottom line: any architecture that isolates domain-related code in a cohesive, loosely coupled domain design can
  support domain-driven design.

**Apply:**
- Keep a dedicated domain layer with zero dependencies on UI, application, or infrastructure concerns.
- Keep the application layer thin — coordination and task-progress state only; push every business rule and
  business-state decision into domain objects.
- Call infrastructure through narrow SERVICE interfaces from application/domain code; never let infrastructure
  initiate action in the domain.
- Adopt framework features selectively; drop any that force domain objects into unnatural shapes.
- Choose consciously: full domain isolation from day one, or a deliberate SMART UI — never an unplanned middle.

## Ch 5 — A Model Expressed in Software

The building blocks that express a model in code: constrained associations, plus three patterns of model elements —
ENTITIES (identity), VALUE OBJECTS (descriptive attributes), SERVICES (operations) — organized into MODULES that are
themselves part of the model.

- Associations: for every traversable association in the model, there is a mechanism in software with the same
  properties (pointer, collection, or encapsulated database lookup — all can reflect the same model). Bidirectional
  many-to-many associations abound in early models but complicate implementation and communicate little. Make them
  tractable by (1) imposing a traversal direction, (2) adding a qualifier to reduce multiplicity, (3) eliminating
  nonessential associations. Constraining direction captures domain insight: country→president, not
  president→country; qualifying by period makes it one-to-one ("Who was U.S. president in 1790?"). Example:
  Brokerage Account→Investment qualified by stock symbol (one investment per stock) simplifies both the in-memory
  and SQL implementations. Retained bidirectionality then signals a genuine semantic feature.

**ENTITIES (a.k.a. Reference Objects)** —
- Problem: some objects are not defined by their attributes but represent a thread of identity running through time
  and across distinct representations; they must be matched even when attributes differ and distinguished even when
  attributes match. Mistaken identity leads to data corruption. Language-level identity (`==`, memory location) is
  too fragile — instances are recreated on every database retrieval or network transmission.
- Therefore: When an object is distinguished by its identity, rather than its attributes, make this primary to its
  definition in the model. Keep the class definition simple and focused on life cycle continuity and identity.
  Define a means of distinguishing each object regardless of its form or history. Be alert to requirements that call
  for matching objects by attributes. Define an operation guaranteed to produce a unique result for each object,
  possibly by attaching a symbol guaranteed unique. The means of identification may come from outside or be an
  arbitrary identifier created by and for the system, but it must correspond to the identity distinctions in the
  model. The model must define what it means to be the same thing.
- Examples: two same-amount, same-day bank deposits are distinct transactions — ENTITIES; check numbers exist to
  match the same transaction across bank statement and check registry. Stadium seats are ENTITIES under assigned
  seating but not under general admission: entity-hood depends on what the application cares about.
- Modeling ENTITIES: strip the definition to the most intrinsic characteristics — those that identify it or are
  commonly used to find or match it — plus behavior essential to the concept; move everything else into associated
  ENTITIES and VALUE OBJECTS. ENTITIES tend to coordinate the operations of objects they own.
- Identity operation: attribute combinations sometimes work (newspaper name + city + date — beware edge cases);
  otherwise attach a unique ID, immutable for life, preserved across flattening and reconstitution. Uniqueness may
  need to span systems (tracking numbers, externally issued IDs — imperfect); attribute matching often needs human
  confirmation. IDs that don't correspond to meaningful domain distinctions only confuse.

**VALUE OBJECTS** —
- Problem: giving every object identity hurts performance, adds analytical work, and muddles the model by making all
  objects look the same. Many objects merely describe things: a child drawing cares about a marker's color, not
  which marker it is.
- Therefore: When you care only about the attributes of an element of the model, classify it as a VALUE OBJECT.
  Make it express the meaning of the attributes it conveys and give it related functionality. Treat the VALUE OBJECT
  as immutable. Don't give it any identity, and avoid the design complexities necessary to maintain ENTITIES.
- The attributes should form a conceptual whole (WHOLE VALUE): street + city + postal code is one Address value,
  not three loose Person fields. VALUE or ENTITY depends on the domain: Address is a VALUE for a mail-order company,
  an ENTITY for the postal service's routing hierarchy or an electric utility (roommates ordering service must be
  recognized as one location).
- VALUES may be composed of other VALUES and may reference ENTITIES (a scenic Route value linking two city ENTITIES
  via a highway); they are often passed as parameters, often transient, and serve as attributes of ENTITIES.
- Immutability makes sharing and reference-passing safe and frees copy-vs-share decisions to be purely technical
  (FLYWEIGHT sharing of interchangeable outlets; denormalized copies for database clustering). Mutable
  implementation is tolerable only for performance (frequent change, expensive creation, clustering) and only if the
  object is never shared.
- Bidirectional associations between two VALUE OBJECTS are meaningless; if one seems necessary, an unrecognized
  identity is probably hiding — reconsider whether it is an ENTITY.

**SERVICES** —
- Problem: some domain concepts are intrinsically activities or actions, not things. Forcing such an operation onto
  an ENTITY or VALUE distorts the object's definition, swamps simple objects, and tangles dependencies; the phony
  alternative is meaningless "Manager" doer-objects — while giving up entirely slides into procedural programming.
- Therefore: When a significant process or transformation in the domain is not a natural responsibility of an ENTITY
  or VALUE OBJECT, add an operation to the model as a standalone interface declared as a SERVICE. Define the
  interface in terms of the language of the model and make sure the operation name is part of the UBIQUITOUS
  LANGUAGE. Make the SERVICE stateless.
- Three marks of a good SERVICE: (1) the operation relates to a domain concept that is not a natural part of an
  ENTITY or VALUE OBJECT; (2) the interface is defined in terms of other elements of the domain model; (3) the
  operation is stateless. Named for an activity — a verb rather than a noun — yet with a defined responsibility.
- Partition services by layer: funds transfer is a *domain* SERVICE (significant business rules, banking language;
  it asks the Account and Ledger objects to do most of the work); exporting transactions to a spreadsheet is an
  *application* SERVICE (no business rules); sending the notification e-mail is an *infrastructure* SERVICE.
- Medium-grained, stateless SERVICES also control interface granularity, ease reuse and distribution, and prevent
  fine-grained domain-interaction logic from leaking into the application layer — but used judiciously, they must
  not strip ENTITIES and VALUE OBJECTS of all behavior.

**MODULES (a.k.a. Packages)** —
- Problem: everyone uses modules, but few treat them as part of the model. Carved by technical category or frozen
  early, they fragment concepts; cognitive overload — not technology — is the primary motivation for modularity.
  Coupling and cohesion are about concepts, not just code metrics.
- Therefore: Choose MODULES that tell the story of the system and contain a cohesive set of concepts. This often
  yields low coupling between MODULES, but if it doesn't, look for a way to change the model to disentangle the
  concepts, or for an overlooked concept that might bring the elements together meaningfully. Seek low coupling in
  the sense of concepts that can be understood and reasoned about independently of each other. Refine the model
  until it partitions according to high-level domain concepts and the code is decoupled as well. Give the MODULES
  names that become part of the UBIQUITOUS LANGUAGE — if the model tells a story, the MODULES are chapters.
- When trade-offs arise, favor conceptual clarity even at the cost of more inter-module references. Refactor
  modules along with the model despite the friction, or module structure fossilizes an obsolete model.
- Resist infrastructure-driven packaging that splits one conceptual object across tiers and packages (J2EE entity
  bean + session bean in separate packages; a four-tiers-per-object project delivered an anemic model): if
  partitioning conventions pull apart the elements implementing a concept, the code no longer reveals the model,
  and the framework uses up all the partitioning a mind can stitch together. Use packaging to separate the domain
  layer from other code; otherwise leave domain developers free to package by domain meaning, keeping all code
  implementing one conceptual object in the same module unless there is a real, immediate distribution need.

- Modeling paradigms: objects dominate because they balance simplicity and sophistication and are mature; some
  domains (intense math, global logical rules) fit other paradigms, and rules engines or workflow can be mixed in.
  Four rules for mixing: don't fight the dominant implementation paradigm; lean on the UBIQUITOUS LANGUAGE to hold
  the heterogeneous model together; don't get hung up on UML; be skeptical the extra tool pulls its weight. Exhaust
  the dominant paradigm first.

**Apply:**
- Classify every domain object explicitly as ENTITY or VALUE OBJECT; give ENTITIES an immutable, domain-meaningful
  identity; make VALUE OBJECTS immutable conceptual wholes with equality by attributes.
- Constrain every association: pick a traversal direction, qualify to reduce multiplicity, delete associations the
  application doesn't need; keep bidirectionality only when it is a domain fact.
- Put behavior on an ENTITY or VALUE first; only when it belongs to none, declare a stateless domain SERVICE named
  in the ubiquitous language with domain objects as parameters and results.
- Package by domain concept, not by pattern, tier, or technical category; module names must be domain words.
- Do not let services strip entities and values of behavior (anemic model); do not add anything to a domain object
  unrelated to the concept it expresses.

## Ch 6 — The Life Cycle of a Domain Object

Long-lived objects with state changes and interdependencies pose two challenges: maintaining integrity throughout the
life cycle, and preventing life-cycle management from swamping the model. Three patterns answer them: AGGREGATES
(integrity boundaries), FACTORIES (creation and reconstitution), REPOSITORIES (finding and retrieval). FACTORIES and
REPOSITORIES do not express model concepts, yet they are part of the domain layer's design responsibility.

**AGGREGATES** —
- Problem: it is difficult to guarantee consistency of changes in a model with complex associations. Invariants
  apply to closely related groups of objects, not just discrete objects, yet cautious locking makes users interfere
  pointlessly and the web of relationships gives no limit to the effect of a change. Where does an object made of
  other objects begin and end? (Delete a Person — does the Address go too, when roommates may share it?)
- Therefore: Cluster the ENTITIES and VALUE OBJECTS into AGGREGATES and define boundaries around each. Choose one
  ENTITY to be the root of each AGGREGATE, and control all access to the objects inside the boundary through the
  root. Allow external objects to hold references to the root only. Transient references to internal members can be
  passed out for use within a single operation only. Because the root controls access, it cannot be blindsided by
  changes to the internals, making it practical to enforce all invariants of the AGGREGATE in any state change.
- Transaction rules: the root has global identity and is ultimately responsible for checking invariants; internal
  ENTITIES have local identity, unique only within the AGGREGATE; nothing outside the boundary may hold a reference
  to anything inside except the root (copies of VALUE OBJECTS may be handed out freely — they carry no association
  back); only AGGREGATE roots are obtainable directly by database query, all else by traversal; internal objects may
  hold references to other AGGREGATE roots; deletion removes everything inside the boundary at once; when a change
  to any object inside commits, all invariants of the whole AGGREGATE must be satisfied.
- Invariants *within* an AGGREGATE are enforced with the completion of each transaction; rules that span AGGREGATES
  are not expected to be up-to-date at all times — resolve them asynchronously (events, batch) within a specified time.
- Examples: a car (root, VIN) encloses its tires (ENTITIES with rotation history, but nobody queries a tire to find
  its car); an engine (engraved serial number, tracked independently) may be its own root. Purchase order: the
  invariant "sum of line items ≤ PO approval limit" was silently broken by two users editing different line items —
  PO plus items form one AGGREGATE, locked as a unit; Part stays outside (high contention, fewer changes), with
  price *copied* into the line item, matching the business fact that price changes don't rewrite existing POs.

**FACTORIES** —
- Problem: assembling a complex object or AGGREGATE is a job unrelated to what the assembled object does (a car
  engine doesn't assemble itself, and you don't need the assembly robot along while driving). Putting creation on
  the product overloads it; making the client assemble internals breaches encapsulation, couples the client to
  concrete classes, and — if the client is in the application layer — leaks domain responsibility out of the domain
  layer entirely.
- Therefore: Shift the responsibility for creating instances of complex objects and AGGREGATES to a separate object,
  which may itself have no responsibility in the domain model but is still part of the domain design. Provide an
  interface that encapsulates all complex assembly and that does not require the client to reference the concrete
  classes of the objects being instantiated. Create entire AGGREGATES as a piece, enforcing their invariants.
- Two requirements of any good FACTORY: (1) each creation method is atomic and enforces all invariants of the
  created object or AGGREGATE — if a correct object cannot be produced, raise an exception, never return a
  malformed value; (2) the FACTORY is abstracted to the type desired, not the concrete classes created.
- Placement: a FACTORY METHOD on the aggregate root to add elements inside the boundary; a FACTORY METHOD on an
  object closely involved in spawning another it doesn't own (Brokerage Account creates Trade Orders, embedding its
  identity and trading rules); otherwise a standalone FACTORY producing a whole AGGREGATE, handing out the root.
- A plain public constructor is fine when: the class is the type (no hierarchy/polymorphism); the client cares about
  the implementation (e.g. choosing a STRATEGY); all attributes are available so no creation nests inside; the
  construction is simple. It must meet the same standard — atomic, all invariants satisfied. Keep constructors dead
  simple; never call constructors within constructors of other classes.
- Invariant logic may live in the FACTORY to keep the product simple — appealing for AGGREGATE-spanning rules and
  creation-time-only rules (identity assignment; anything immutable thereafter).
- ENTITY FACTORIES take just the essential attributes and handle identity; VALUE OBJECT FACTORIES must emit the
  complete, final, immutable product. Reconstitution FACTORIES differ from creation: no new ID is assigned (identity
  attributes are inputs), and invariant violations must be repaired somehow, not simply rejected — the object exists.

**REPOSITORIES** —
- Problem: to use an object you must hold a reference; creation and traversal are not enough (a team that tried
  all-traversal access built an endless tangle). But scattered database queries swamp client code in technology,
  tempt developers to bypass AGGREGATE roots and encapsulation, drain domain rules into query code, and reduce
  ENTITIES and VALUE OBJECTS to data containers — the model becomes irrelevant. Meanwhile only a subset of objects
  should be globally accessible at all: AGGREGATE roots not conveniently reached by traversal.
- Therefore: For each type of object that needs global access, create an object that can provide the illusion of an
  in-memory collection of all objects of that type. Set up access through a well-known global interface. Provide
  methods to add and remove objects, which encapsulate the actual insertion or removal of data in the data store.
  Provide methods that select objects based on some criteria and return fully instantiated objects or collections
  whose attribute values meet the criteria, thereby encapsulating the actual storage and query technology. Provide
  REPOSITORIES only for AGGREGATE roots that actually need direct access. Keep the client focused on the model,
  delegating all object storage and access to the REPOSITORIES.
- Advantages: an intention-revealing model for obtaining persistent objects; decoupling from persistence technology
  and data sources; communicated design decisions about object access; easy in-memory dummies for testing.
- Queries: hard-coded methods (by identity, attribute values, ranges; even counts and sums the model intends to
  tally) are cheap and always legitimate; SPECIFICATION-based query frameworks add declarative flexibility.
- Cautions: clients ignore the implementation, but developers must not — performance implications can be extreme
  (an "all objects" query once pulled an entire production database into memory). Leave transaction control to the
  client; the REPOSITORY inserts and deletes but does not commit. Don't fight your frameworks: seek affinities
  (entity bean ≈ AGGREGATE root; EJB Home ≈ REPOSITORY) and let go of specifics where the framework is antagonistic.
- FACTORY vs. REPOSITORY: the FACTORY makes new objects; the REPOSITORY finds old ones. Reconstitution of a stored
  object is not creation of a new conceptual object — it is the same ENTITY, mid-life-cycle, even if a new instance
  is made. A REPOSITORY may delegate reconstitution to a FACTORY. Avoid "find or create": the distinction between
  new and existing is usually important in the domain, and clients wanting a VALUE can just ask a FACTORY.
- Designing objects for relational databases: when the database is primarily the object store, keep data model and
  object model close — sacrifice some object richness and some relational norms (selective denormalization) rather
  than maintain divergent models. A table row should contain an object (perhaps with AGGREGATE subsidiaries); a
  foreign key should translate to an ENTITY reference; table and column names should correspond meticulously to the
  UBIQUITOUS LANGUAGE. Outside processes must not write to the object store (they bypass invariants). A legacy or
  external schema is effectively another model; divergence is legitimate only as a conscious choice for clear reasons.

**Apply:**
- Define an AGGREGATE boundary and root for every ENTITY cluster; hold references only to roots from outside; query
  only for roots; delete the whole boundary at once.
- Enforce every intra-aggregate invariant at each commit; design cross-aggregate rules as delayed, asynchronous
  consistency.
- Draw boundaries from contention and change-frequency realities; copy values (like price) into the aggregate when
  the business snapshot demands it.
- Create complex objects and aggregates only through atomic, invariant-enforcing FACTORIES (or equally disciplined
  simple constructors); never let clients assemble internals or reference concrete product classes.
- Add one REPOSITORY per aggregate root that genuinely needs global access — no more; keep transaction commits in
  the caller.
- Keep object–relational mappings simple, transparent, and named in the ubiquitous language.

## Ch 7 — Using the Language: An Extended Example

A cargo-shipping system (track handling of cargo, book cargo, auto-invoice at a handling milestone) shows the Part II
patterns combined. Model: Cargo; Customers in qualified roles (shipper, receiver, payer); a Delivery Specification
(goal: destination, arrival date); a Delivery History of Handling Events (loading, unloading, customs...), each event
referencing a Carrier Movement between Locations. Design moves demonstrated:

- Delivery Specification extracted as a VALUE OBJECT (SPECIFICATION pattern): declutters Cargo and states explicitly
  that the means of delivery is undetermined but must meet the goal.
- LAYERED ARCHITECTURE first: three thin application coordinators (Tracking Query, Booking Application, Incident
  Logging) that ask questions the domain layer answers.
- ENTITY vs. VALUE with domain-driven identity: Customer (existing company-wide ID), Cargo (tracking ID), Handling
  Event (Cargo ID + completion time + type), Carrier Movement (schedule code), Location (internal ID); Delivery
  History is an ENTITY with identity borrowed from its Cargo; Delivery Specification and Role are VALUES.
- Associations constrained by business need: Handling Event → Carrier Movement only (we track cargo, not ships); no
  Customer→Cargo collection (query instead); one deliberate Cargo↔Delivery History↔Handling Event cycle kept in the
  model, with implementation choices (collection now, query later) that don't change the model.
- AGGREGATES: Customer, Location, Carrier Movement are their own roots (shared by many cargoes); Cargo's boundary
  encloses Delivery History and Delivery Specification; Handling Event becomes its own root because handling has
  meaning apart from the cargo. REPOSITORIES strictly from application requirements: Customer, Location, Carrier
  Movement, Cargo — none for Handling Event until a requirement demands it.
- Scenario walk-throughs cross-check decisions: changing destination = replace the Delivery Specification VALUE;
  "repeat business" copies a Cargo prototype respecting the boundary — new empty Delivery History, new tracking ID,
  shared references to outside ENTITIES (Customers), nothing outside the boundary affected.
- Refactoring for contention: adding a Handling Event updated Delivery History's collection, dragging the Cargo
  AGGREGATE into every logging transaction; replacing the collection with a query via a new Handling Event
  Repository made event entry contention-free and Delivery History derivable — a design change entirely inside the
  same model, enabled by the aggregate boundary.
- MODULES by domain story (Customer, Shipping, Billing), not by pattern — packages named entities/values/services
  tell "the story of what the developer was reading," with high coupling and low cohesion.
- Integrating an external Sales Management System for allocation checking: an Allocation Checker SERVICE acts as an
  ANTICORRUPTION LAYER translating between models; a new Enterprise Segment VALUE OBJECT (the dimensions along
  which the business is segmented) enriches the domain instead of importing the other system's category strings;
  the acceptance rule moves out of the Booking Application into the domain; segment derivation lives with the object
  that knows the segmentation rules (Allocation Checker), not the object that merely holds the data (Cargo).

**Apply:**
- Derive repositories, aggregate boundaries, and association directions from concrete application scenarios, and
  re-walk the scenarios after every design change.
- When integrating a foreign system, translate at a boundary service expressed in your model's terms; grow your
  model with a new concept rather than absorbing the other system's representation.
- Place a rule on the object that knows the rule, not the object that merely has the data it applies to.

## Ch 8 — Breakthrough

Returns from refactoring are not linear: small continuous refactorings fight entropy, but the most important
insights arrive abruptly as a breakthrough — a rush of change to a model that corresponds on a deeper level to
the realities and priorities of the users, where versatility and explanatory power suddenly increase even as
complexity evaporates. A breakthrough is not a technique but an event; the skill is recognizing it and
deciding how to deal with it.

- Case study (syndicated loans): a workable model tied Facility shares and Loan shares together. Symptoms of
  the mismatch: unexpected requirements kept complicating the design (Loan Adjustments bolted on), and
  rounding inconsistencies resisted ever-more-complex algorithms. The breakthrough: shares of the Loan and
  shares of the Facility can change independently. "Investments" and "Loan Investments" were special cases of
  one fundamental concept — shares of any divisible value — modeled as an abstract **Share Pie** with "shares
  math." Special-case logic disappeared; the invented pseudo-concept "loan investment" (which business experts
  had never understood) was deleted; the persistent rounding problems were pulled out by the roots.
- Symptom checklist for a subtly wrong model: requirements keep arriving that don't fit; complexity grows
  without converging on solid functionality; experts call your diagrams "too technical"; terms in the design
  don't exist in the business's language.
- A deep model, once found, becomes the unifying theme of the application and enters the UBIQUITOUS LANGUAGE —
  used by developers, experts, marketers, and customers alike.
- Decision economics: refactoring to the deep model had no stable intermediate stopping points, no tests, an
  exhausted team, and a hard deadline — and was still the lower-risk choice long term (three weeks to parity;
  forward movement without it would be slow, and the change much harder once there was an installed base).
- Don't chase breakthroughs directly; you cannot schedule them. Set the stage: crunch knowledge, cultivate the
  UBIQUITOUS LANGUAGE, make implicit concepts explicit (Ch 9), make the design suppler (Ch 10), distill the
  model. Clarity is the usual precursor of a breakthrough.
- Breakthroughs cascade: weeks after Share Pie, a missing ENTITY ("Transaction") became visible — the clearer
  field of vision exposed the next round of implicit concepts, and development accelerated at the stage where
  most projects bog down.

**Apply:**
- Treat recurring "unexpected" requirement changes and un-killable edge-case bugs as symptoms of a wrong model
  relationship, not as normal churn.
- Delete concepts the domain experts don't recognize; they are usually artifacts of incomplete understanding.
- When a candidate deep model appears, walk every known scenario through it on a whiteboard before coding.
- Weigh deep-model refactorings on long-run risk, not just short-run schedule; the change gets more expensive
  once code has an installed base.
- After any significant model improvement, re-examine the design — clearer vision exposes the next missing
  concept.

## Ch 9 — Making Implicit Concepts Explicit

Deep models grow out of a repeated move: recognize a concept hinted at in discussion or present implicitly in
the design, and represent it explicitly in the model with objects or relationships. The breakthrough usually
comes only after several important concepts have been made explicit and refactored repeatedly.

How to dig out implicit concepts:

- **Listen to language.** Terms domain experts use that are absent from the design are a warning sign — doubly
  so when both experts and developers use vocabulary that is nowhere in the design — and an opportunity to
  improve the model. Hints: a term that succinctly states something complicated; experts diplomatically
  correcting your word choice; puzzled looks that vanish when you use a particular phrase. Not "nouns are
  objects": a new word is a lead to follow with knowledge crunching. Example: shipping experts kept saying
  "itinerary"; the data existed only as table rows plus report logic. Reifying an Itinerary object made the
  Routing Service interface expressive, decoupled routing from the database tables, moved domain logic out of
  the report into the domain layer, and expanded the UBIQUITOUS LANGUAGE.
- **Scrutinize awkwardness.** Dig where procedures do complicated things that are hard to explain and every
  new requirement adds complexity. Example: an ever-more-complex Interest Calculator with special cases for
  late payments; probing conversation surfaced "accrual basis accounting," and refactoring to daily Accruals
  posted to ledgers decoupled accrual from payment, unified fees and interest, moved ledger knowledge into the
  domain layer, and made new variations easy to add as Accrual Schedules.
- **Contemplate contradictions.** When two factual statements by experts seem to conflict (beyond terminology
  or misunderstanding), contemplating how both could apply to the same external reality can reveal a deeper
  model (Galileo's inertial-frames thought experiment is the archetype).
- **Read the book.** Mine the literature of the domain — many fields have thinkers who have already organized
  and abstracted its practice (the accounting-book version of the accrual story), plus software-oriented
  sources (analysis patterns). You still must crunch with your own experts, but you start from a coherent,
  deeply considered view instead of reinventing the field.
- **Try, try again.** Follow half a dozen leads before one is worth trying; replace it later as knowledge
  crunching serves up better ideas. A modeler cannot afford to get attached to his own ideas. Each experiment
  leaves the design more supple; trying to avoid missteps yields a lower-quality result on less experience.

Less obvious categories of concept worth modeling explicitly:

- **EXPLICIT CONSTRAINTS** —
  (a) Constraints emerge implicitly and get lost inside operations; as rules grow they overwhelm the object
  they apply to, or have no good home in any existing object. Warning signs: evaluating the constraint needs
  data that doesn't fit the object's definition; related rules recur in objects that aren't otherwise a
  family; conversation revolves around the constraints but the implementation hides them in procedural code.
  (b) Factor the constraint into its own method with an intention-revealing name, giving it room to grow
  (`constrainedToCapacity()` in the Bucket example); when it still doesn't fit, factor it out into an explicit
  object, or model it as a set of objects and relationships.
  (c) The shipping overbooking policy (book 10% more than capacity) extracted into an Overbooking Policy
  class — a business rule visible in diagrams and code instead of buried in a guard clause.
- **PROCESSES AS DOMAIN OBJECTS** —
  (a) Processes that exist in the domain make awkward object designs when left implicit; yet procedures must
  not become a prominent aspect of the model.
  (b) Express the process explicitly as a SERVICE encapsulating the complex algorithm; when there is more than
  one way to carry out the process, make the algorithm (or a key part) an object in its own right — a STRATEGY.
  (c) Litmus test: is this something the domain experts talk about, or just part of the mechanism of the
  computer program? Only the former belongs in the model.
- **SPECIFICATION** —
  (a) Business rules often don't fit the responsibility of any obvious ENTITY or VALUE OBJECT, and their
  variety and combinations overwhelm the basic meaning of the domain object (`Invoice.isDelinquent()`
  ballooning with grace periods, account status, collection policy). Moving the rules out of the domain layer
  is worse: it leaves a dead data object and domain code that no longer expresses the model. Full
  logic-programming predicates in objects are a major undertaking and too general to communicate intent.
  (b) Therefore: Create explicit predicate-like VALUE OBJECTS for specialized purposes. A SPECIFICATION is a
  predicate that determines if an object does or does not satisfy some criteria. It states a constraint on the
  state of another object, which may or may not be present, and it keeps the rule in the domain layer.
  (c) DelinquentInvoiceSpecification, built with an evaluation date, exposing `isSatisfiedBy(Invoice)`; a
  FACTORY can configure it from other sources (customer account, policy database) without coupling Invoice.

Three uses of SPECIFICATION — conceptually one, even when implementations diverge; without the pattern the
same rule shows up in different, possibly contradictory guises:

1. Validation — test an individual object to see if it fulfills some need or is ready for some purpose.
2. Selection/querying — select objects from a collection (`repository.selectSatisfying(spec)`). SQL is a
   natural way to write SPECIFICATIONS; either the SPEC carries the query (risking table details leaking into
   the domain layer), or — better — the SPEC delegates via double dispatch,
   `spec.satisfyingElementsFrom(repository)`: SQL lives in the REPOSITORY while the SPEC keeps the essential
   declaration of the rule. Performance trade-offs are implementation choices that leave the model unchanged.
3. Building to order (generating) — describe criteria for objects not yet present, like a fighter-jet
   procurement spec: it constrains the product without designing it. A generator whose interface takes a
   descriptive SPEC decouples implementation from interface, communicates its rules explicitly, leaves the
   request in the client's hands, and is easier to test — the same SPEC that constrains creation can validate
   the output (an ASSERTION). Book example: the chemical warehouse packer — each Chemical carries a
   ContainerSpecification (TNT → armored, ammonia → ventilated); the WarehousePacker SERVICE interface carries
   the ASSERTION that after `pack()`, every Drum's ContainerSpecification is satisfied by its Container.

- Clearing logjams with working prototypes: with implementation decoupled from interface, a naive dozen-line
  PrototypePacker satisfying the same interface and ASSERTIONS let the application team and the optimization
  team work in parallel with early user feedback; integrating the real Packer later was a breeze because it
  was written to the same well-characterized interface.

**Apply:**
- When experts use a word that has no counterpart in the code, reify it — as class, method, or relationship —
  and keep renaming until code matches speech.
- Extract every non-trivial rule into an intention-revealing method; when it needs foreign data or recurs
  across unrelated objects, promote it to a constraint object or SPECIFICATION.
- Model a domain process the experts talk about as a SERVICE or STRATEGY; keep mechanical procedures hidden.
- Use one SPECIFICATION concept for validation, querying, and building-to-order rather than scattering the
  same rule in three forms.
- Before designing a generator or solver, write the validation of its output first and publish it as the
  ASSERTION of the interface.
- Read the domain's literature before inventing concepts the field has already refined.

## Ch 10 — Supple Design

Software has to serve developers before it can serve users: a supple design is a pleasure to work with,
inviting to change — the complement to deep modeling. It serves two roles: the client developer, who should
flexibly use a minimal set of loosely coupled concepts to express a range of domain scenarios with predictable
results, and the maintainer, for whom the effects of code must be transparently obvious. Overengineered
"flexibility" layers are the opposite; supple designs are usually simple, and simple is not easy.

- **INTENTION-REVEALING INTERFACES** —
  (a) If a developer must consider the implementation of a component in order to use it, the value of
  encapsulation is lost; a purpose inferred from the implementation may hold only by chance, corrupting the
  conceptual basis of the design as developers work at cross-purposes.
  (b) Therefore: Name classes and operations to describe their effect and purpose, without reference to the
  means by which they do what they promise. This relieves the client developer of the need to understand the
  internals. Names should conform to the UBIQUITOUS LANGUAGE so team members can quickly infer their meaning.
  Write a test for a behavior before creating it, to force your thinking into client-developer mode. In public
  interfaces, state relationships and rules but not how they are enforced; describe events and actions but not
  how they are carried out; pose the question, but don't present the means by which the answer shall be found.
  (c) Paint example: `paint(Paint)` is unguessable without reading the code; writing the client test first
  drives the rename to `mixIn(Paint)`, `getVolume()`.
- **SIDE-EFFECT-FREE FUNCTIONS** —
  (a) Operations are commands (change observable state) or queries. With deep nesting of calls, even
  intentional changes become "side effects" in every sense: the caller must understand an operation's
  implementation and all its delegations to anticipate the result, defeating abstraction and placing a low
  ceiling on the feasible richness of behavior.
  (b) Therefore: Place as much of the logic of the program as possible into functions — operations that return
  results with no observable side effects. Strictly segregate commands (methods that modify observable state)
  into very simple operations that do not return domain information. Further control side effects by moving
  complex logic into VALUE OBJECTS when a concept fitting the responsibility presents itself; the side effect
  can often be eliminated entirely by deriving a new VALUE OBJECT instead of changing existing state. VALUE
  OBJECTS are immutable, so apart from initializers all their operations are functions.
  (c) The color math moves from the mutating `Paint.mixIn()` into immutable
  `PigmentColor.mixedWith(other, ratio)`, returning a new PigmentColor — safe to combine, easy to test.
- **ASSERTIONS** —
  (a) Commands remain, and when their side effects are defined only implicitly by their implementation,
  designs with much delegation become a tangle of cause and effect, understandable only by tracing execution;
  interfaces don't restrict side effects, so two subclasses of one interface can have different ones — so much
  for abstraction and polymorphism.
  (b) Therefore: State post-conditions of operations and invariants of classes and AGGREGATES. If ASSERTIONS
  cannot be coded directly in your programming language, write automated unit tests for them; write them into
  documentation or diagrams where it fits the project's style. Seek models with coherent sets of concepts,
  which lead a developer to infer the intended ASSERTIONS, accelerating the learning curve and reducing the
  risk of contradictory code. Assertions describe state, not procedure, so tests are easy: setup establishes
  preconditions; then post-conditions are checked.
  (c) Honestly stating `mixIn()`'s post-condition ("p2's volume is unchanged") exposed a counter-intuitive
  model; the awkwardness pointed to missing concepts, and splitting StockPaint from MixedPaint left one
  trivial command and common-sense invariants.
- **CONCEPTUAL CONTOURS** —
  (a) Chopping functionality fine, lumping it large, or seeking uniform granularity are all
  oversimplifications: monoliths duplicate functionality and mix concepts; excessive breakdown complicates the
  client and can lose the concept entirely — half a uranium atom is not uranium. It isn't just grain size that
  counts, but where the grain runs.
  (b) Therefore: Decompose design elements (operations, interfaces, classes, and AGGREGATES) into cohesive
  units, taking into consideration your intuition of the important divisions in the domain. Observe the axes
  of change and stability through successive refactorings and look for the underlying CONCEPTUAL CONTOURS
  that explain these shearing patterns. Align the model with the consistent aspects of the domain that make it
  a viable area of knowledge in the first place. Ask of each decision: expedient of the current code, or echo
  of a contour of the underlying domain? Keep whole values whole (paint users combine complete paints, never
  individual pigments).
  (c) The accrual model absorbed an unanticipated requirement (early/late-payment rules) as a local extension
  to the single Payment class — not foresight, but alignment with underlying domain concepts. Localized
  refactorings signal model fit; a requirement forcing broad restructuring says understanding needs refinement.
- **STANDALONE CLASSES** —
  (a) Every association, argument type, and return value is a dependency; with each one added, the effort of
  understanding a class snowballs, and implicit concepts count as much as explicit references. Even within a
  MODULE, interpretation difficulty grows wildly with dependencies.
  (b) Prescription: refine the model until every remaining connection represents something fundamental to the
  concepts; in an important subset, reduce dependencies to zero, yielding a class fully understandable by
  itself along with primitives and basic library concepts. Every dependency is suspect until proven basic to
  the concept behind the object. Eliminate all nonessential dependencies (those within a module, or between
  naturally tightly coupled pairs, are less harmful). Try to factor the most intricate computations into
  STANDALONE CLASSES, perhaps by modeling VALUE OBJECTS held by the more connected classes. Don't dumb the
  model down to primitives to get there.
  (c) PigmentColor: color can be considered without paint; the class holding most of the computational
  complexity can be studied and tested alone.
- **CLOSURE OF OPERATIONS** —
  (a) Interfaces stripped down to primitives are impoverished, but many unnecessary dependencies — even entire
  concepts — get introduced at interfaces.
  (b) Therefore: Where it fits, define an operation whose return type is the same as the type of its
  argument(s). If the implementer has state that is used in the computation, then the implementer is
  effectively an argument of the operation, so the argument(s) and return value should be of the same type as
  the implementer. Such an operation is closed under the set of instances of that type: a high-level interface
  with no dependency on other concepts, trivially chained (as multiplication is closed under real numbers; as
  XSLT maps XML to XML). Mostly an opportunity on VALUE OBJECTS — ENTITIES are rarely computation results.
  Half-closures — the extra type is a primitive or basic library class — give much of the benefit: Smalltalk's
  `select:` returns a Collection, versus Java's extraneous Iterator concept.
  (c) `PigmentColor.mixedWith()` is closed under PigmentColors; SharePie `plus`/`minus` under SharePies.

Declarative design and declarative style:

- True declarative design (programs as executable specifications via reflection, code generation, or rule
  engines) is a Holy Grail with recurring pitfalls: declaration languages not expressive enough plus
  frameworks that resist extension; code generation that merges destructively with handwritten code and
  cripples iteration; rule engines whose performance-tuning "control predicates" reintroduce side effects. The
  common result is dumbing-down of the model as developers enact design triage inside framework limits. The
  greatest delivered value: narrowly scoped frameworks automating one tedious, error-prone aspect
  (persistence, O-R mapping) while leaving complete design freedom. Domain-specific languages bind tightest to
  the UBIQUITOUS LANGUAGE but are hard to refine iteratively and split framework builders from app builders.
- A declarative style of design needs none of that machinery: once a design has INTENTION-REVEALING
  INTERFACES, SIDE-EFFECT-FREE FUNCTIONS, and ASSERTIONS, you are edging into declarative territory —
  combinable elements that communicate their meaning, with characterized or no observable effects, deliver
  many of the benefits of declarative design.
- **COMPOSITE SPECIFICATION (SPECIFICATION extended declaratively)** — A SPECIFICATION is a predicate, so
  SPECS combine with `and()`, `or()`, `not()`; logical operations are closed under predicates, so combinations
  exhibit CLOSURE OF OPERATIONS: `ventilated.and(armored)` for a volatile explosive;
  `(ventilated.not()).and(armored.not())` declares a "cheap container" rule for sand. Implement as a COMPOSITE
  of leaf and operator SPECS — or, when fine-grained objects are too costly, as an encoded expression
  interpreted at runtime: same model, very different implementations. Don't build full generality when AND
  alone suffices. Subsumption — `spec.subsumes(other)`, logical implication: any candidate satisfying the new
  SPEC also satisfies the old — answers "which chemicals' handling became more stringent," even for chemicals
  not in inventory. General implication proofs are hard, but special cases are easy: parameterized SPECS
  define their own rule (MinimumAgeSpecification: compare thresholds), and AND-only composites reduce to a
  leaf-superset check. Avoid combining subsumption with OR/NOT.

Angles of attack:

- Carve off subdomains: you can't make a whole system supple at once. Pull out a part viewable as specialized
  math, or complex state-change rules into a validation framework; each step leaves the remainder smaller and
  clearer, partly declarative. A big impact on one area beats spreading efforts thin.
- Draw on established formalisms: long-refined conceptual systems such as accounting or arithmetic are clean,
  combinable by clear rules, and easy to understand. Capstone "Shares Math" example: Loan payment distribution
  refactored stepwise — separate the command from the side-effect-free calculation; make the implicit
  whole/part concept explicit as Share Pie; make Share Pie an immutable VALUE OBJECT with closed operations
  `plus()`, `minus()`, `prorated()` — until Loan methods read like conceptual definitions of business
  transactions ("applying a principal payment means subtracting the payment from the loan, share by share")
  and analytical features safely reuse the same functions. A familiar formalism doesn't have to be invented or
  learned — provided the design stays carefully consistent with its rules so people are not misled.

**Apply:**
- Name every public element for effect and purpose in the UBIQUITOUS LANGUAGE, never for mechanism; write the
  client-side test first to force the naming.
- Enforce command-query separation: queries and calculations have no observable side effects; commands stay
  trivial and return no domain data.
- Push complex logic into immutable VALUE OBJECTS whose operations return new values; prefer operations closed
  under the value's own type.
- State post-conditions and invariants for every command and AGGREGATE, as assertions or unit tests; if the
  honest assertion sounds weird, suspect a missing concept and split the model.
- Justify every dependency as fundamental to the concept; drive the most intricate computations into
  standalone, self-contained classes.
- Decompose along domain contours, not uniform granularity; treat any requirement forcing broad restructuring
  as a signal to deepen the model.
- Hunt for established formalisms (arithmetic, predicate logic, accounting) hiding in the domain and factor
  that part out under the formalism's rules.

## Ch 11 — Applying Analysis Patterns

Analysis patterns let you cut through expensive trial and error by starting from models others have carried
through implementation and maintenance — but they feed the knowledge-crunching process; they are not
out-of-the-box solutions.

- **ANALYSIS PATTERNS** —
  (a) Deep models come from lots of learning, talking, and trial and error; without prior art, teams reinvent
  (and mis-invent) concepts whole fields have refined, and discover implementation consequences the hard way.
  (b) Fowler's definition: "Analysis patterns are groups of concepts that represent a common construction in
  business modeling. It may be relevant to only one domain or it may span many domains." They are conceptual,
  not technological — yet at their best they combine model insights with discussion of design directions and
  implementation and maintenance consequences, avoiding the deadly divide between analysis and design that is
  antithetical to MODEL-DRIVEN DESIGN. They offer valuable leads and cleanly abstracted vocabulary, not
  answers.
  (c) From Fowler's "Inventory and Accounting": Account holds an append-only history of Entries (value changes
  only by inserting Entries; balance is their combined effect — computed or cached, an implementation decision
  encapsulated by the Account interface); Transaction moves money between Accounts under double-entry
  conservation; Posting Rules make cross-account dependency explicit (a rule triggered by a new Entry in its
  input Account derives and inserts an Entry in its output Account), with three firing modes — eager,
  account-based, posting-rule-based — whose names in the UBIQUITOUS LANGUAGE matter as much as the objects.
- In the interest-tracking story, developers tried Fowler's Transaction, found through expert conversation
  that it misfit (accrual and payment are separate postings), kept Account and Entry, subclassed Entry into
  Payment and Accrual — letting the pattern's vocabulary enrich the UBIQUITOUS LANGUAGE. The result often
  resembles the documented form but adapted; sometimes it doesn't obviously relate at all, yet was stimulated
  by the pattern's insight. Analysis patterns also expose blind spots: the nightly batch script had never been
  considered domain-oriented; Posting Rules revealed its implicit domain logic, and the batch became a thin
  layer sending a few self-explanatory messages to domain objects.
- Reality forces calculated compromises (concrete Entry subclasses per relational table because of the O-R
  mapping and table-readability standards); make them and move on without abandoning MODEL-DRIVEN DESIGN.
- When you use a term from a well-known analysis pattern, keep the basic concept it designates intact however
  much the superficial form changes: the pattern may embed understanding that avoids problems, and widely
  understood terms enhance the UBIQUITOUS LANGUAGE. If your definitions later diverge, change the names.
- This is knowledge reuse, not code reuse: a framework is a complete working whole; an analysis pattern is a
  kit of model fragments that focuses on the most critical and difficult decisions, illuminates alternatives,
  and anticipates downstream consequences that are expensive to discover for yourself.

**Apply:**
- Before modeling a well-trodden area (accounting, inventory, scheduling), check documented analysis patterns
  for a starting model and vocabulary.
- Validate every borrowed model element against your own domain experts; discard parts that misfit rather
  than force the pattern.
- If you adopt a pattern's term, preserve its established meaning — or rename when your concept diverges.
- Suspect "mechanical" scripts and batch jobs of hiding domain logic; give them a domain model and reduce
  them to thin orchestration.

## Ch 12 — Relating Design Patterns to the Model

Some, not all, technical design patterns can serve as domain patterns — but only by reading them on two levels
simultaneously: as technical design patterns in the code and as conceptual patterns in the model. The only
requirement is that the pattern say something about the conceptual domain, not just be a technical solution
to a technical problem.

- **STRATEGY (a.k.a. POLICY)** —
  (a) Domain processes often have more than one legitimate way of being done; describing the options makes
  the process definition clumsy, and the actual behavioral alternatives are obscured as they mix in with the
  rest of the behavior.
  (b) Therefore: Factor the varying part of a process into a separate "strategy" object in the model. Factor
  apart a rule and the behavior it governs. Implement the rule or substitutable process following the STRATEGY
  design pattern. Multiple versions of the strategy object represent different ways the process can be done.
  As a domain pattern the emphasis shifts from substituting algorithms to expressing a concept — usually an
  actual business process or policy rule. The design pattern's implementation experience still applies (e.g.
  share stateless strategy objects if object count becomes a problem).
  (c) Routing Service fastest-vs-cheapest: a Leg Magnitude Policy passed as a parameter replaces conditionals
  in every computation — "the Routing Service chooses an Itinerary with a minimum total magnitude of the Legs
  based on the chosen STRATEGY."
- **COMPOSITE** —
  (a) Important domain objects are composed of parts made of parts, occasionally nesting to arbitrary depth,
  where parts are conceptually the same kind of thing as the whole. Unreflected in the model: behavior
  duplicated at each level, rigid nesting, different client interfaces per level, complicated recursion for
  aggregated information.
  (b) Therefore: Define an abstract type that encompasses all members of the COMPOSITE. Methods that return
  information are implemented on containers to return aggregated information about their contents; "leaf"
  nodes implement them based on their own values. Clients deal with the abstract type and need not distinguish
  leaves from containers. First verify the fit: a true whole-part hierarchy, an abstraction under which all
  parts truly are the same conceptual type. The power lies in rigorous operational symmetry — the same
  behavior at every structural level — not the structure alone.
  (c) Shipping Routes: expert-visible segments and door legs complicated the model until every level — route,
  segment, leg — was seen as "a movement of a container from one point to another": routes made of routes
  restored uniform traversal and enabled splicing and arbitrary nesting. A design pattern should be applied
  only when it is needed — the team did fine without COMPOSITE until those distinctions appeared.
- Counter-example — FLYWEIGHT: sharing instances of a limited set of VALUE OBJECTS is purely an implementation
  option (available for VALUE OBJECTS, never ENTITIES) with no correspondence to the domain model. COMPOSITE
  applies to both model and implementation — the essential trait of a domain pattern.

**Apply:**
- Use a design pattern in the domain layer only when it fits both an implementation need and a genuine domain
  concept; then exploit the pattern's documented implementation experience.
- Model alternative business processes or policies as explicit STRATEGY objects instead of threading
  conditionals through a service.
- Before applying COMPOSITE, verify parts and whole are truly one conceptual type with operations meaningful
  symmetrically at every level.
- Don't apply patterns speculatively; introduce them when the domain distinctions that motivate them appear.

## Ch 13 — Refactoring Toward Deeper Insight

Refactoring toward deeper insight superimposes a broader process on conventional micro-refactoring. Three
focal points: live in the domain; keep looking at things a different way; maintain an unbroken dialog with
domain experts.

- Initiation: it can begin with awkward code whose root is sensed to be in the domain model — but also, in a
  departure from conventional refactoring, when the code looks tidy: if the language of the model is
  disconnected from the domain experts, if new requirements are not fitting in naturally, or when a
  developer's learning reveals an opportunity for a more lucid or useful model. Seeing the trouble spot is
  often the hardest and most uncertain part.
- Exploration teams: assemble four or five people on the fly — the initiators plus developers good at that
  kind of problem or strong at modeling, and a domain expert if there are subtleties. Brainstorm 30–90
  minutes: sketch diagrams, walk scenarios with the objects, make sure the expert understands the model and
  finds it useful; then code it, or sleep on it and reconvene. Keys: self-determination (small ad hoc teams,
  no long-term structure); scope and sleep (two or three short meetings spaced over a few days — if stuck,
  you're taking on too much; pick a smaller aspect); exercising the UBIQUITOUS LANGUAGE — the session's real
  product is a refinement of the LANGUAGE, which the developers then formalize in code.
- Prior art: feed knowledge crunching with domain literature, analysis patterns (subtle concepts and avoided
  mistakes, but no cookbook recipe), design patterns where they fit both implementation need and model
  concept, and established formalisms (arithmetic, predicate logic) adapted for tight, readily understood
  models.
- A design for developers: software is for developers too. A supple design communicates its intent, makes the
  effects of running and changing code easy to anticipate, and limits mental overload by reducing dependencies
  and side effects. It rests on a deep model fine-grained only where most critical to users — flexibility
  where change is most common, simplicity elsewhere.
- Timing (refactoring economics): if you wait until you can make a complete justification for a change, you've
  waited too long — the project is already incurring heavy costs, and postponed changes get harder as code
  becomes more elaborated and embedded. The risk of changing code and the developer time are visible; the risk
  of keeping an awkward design and the cost of working around it are not. Demanding justification makes an
  already difficult thing impossibly difficult and squelches refactoring or drives it underground. Therefore,
  refactor when: (1) the design does not express the team's current understanding of the domain; (2) important
  concepts are implicit in the design and you see a way to make them explicit; or (3) you see an opportunity
  to make some important part of the design suppler. Limits: don't refactor the day before a release; don't
  introduce supple-design demonstrations of technical virtuosity that fail to cut to the core of the domain;
  don't introduce a "deeper model" you couldn't convince a domain expert to use, however elegant. Don't be
  absolute — but push beyond the comfort zone in the direction of favoring refactoring.
- Crisis as opportunity: development follows punctuated equilibrium — long steady refinement interrupted by
  short bursts of rapid change. A breakthrough often first looks like a crisis: a gaping hole in what the
  model can express, an opaque critical area, statements that are just wrong. That perception means the team
  has reached a new level of understanding: from the elevated viewpoint the old model looks poor, and a far
  better one can be conceived. Then steady refinement begins again.

**Apply:**
- Refactor on model grounds, not only code smells: tidy code with the wrong language, or requirements that
  won't fit, are triggers too.
- For non-obvious model problems, run short, small, expert-included brainstorming sessions spaced over a few
  days instead of solo redesign or standing committees.
- Don't demand full cost-benefit justification for each refactoring; default toward refactoring whenever
  understanding has outgrown the design.
- Respect the limits: no refactoring right before release, no virtuosity for its own sake, no "deep model"
  the experts wouldn't adopt.
- When the model suddenly looks broken, treat the crisis as evidence of new understanding and mine it for the
  deeper model.

## Ch 14 — Maintaining Model Integrity

The most fundamental requirement of a model is internal consistency ("unification"): every term unambiguous, no contradictory rules. Total unification of the domain model of a large system is not feasible or cost-effective — multiple models will exist, driven as much by politics and team organization as by technical need. Instead of drifting into accidental fragmentation, explicitly decide which parts of the system will diverge, mark the boundaries, and choose the relationship between models consciously. The chapter's cautionary opening: two teams unknowingly shared one `Charge` class with two incompatible meanings (billing customers vs. paying vendors), producing corrupt data and crashes — a "false cognate."

**BOUNDED CONTEXT** — Multiple models are in play on any large project; when code based on distinct models is combined, software becomes buggy and communication becomes confused, because it is unclear in what context a model applies. Therefore: explicitly define the context within which a model applies. Explicitly set boundaries in terms of team organization, usage within specific parts of the application, and physical manifestations such as code bases and database schemas. Keep the model strictly consistent within these bounds, but don't be distracted or confused by issues outside. Each CONTEXT has its own dialect of the UBIQUITOUS LANGUAGE; integration across boundaries always involves translation. Example: a shipping booking model's CONTEXT included the booking application and the schema (model-driven), but excluded the legacy tracking system and the casually coordinated voyage-schedule team — recognizing the latter as a separate context was the biggest win. Note: BOUNDED CONTEXTS are not MODULES — modules organize elements *within* one model; separate namespaces inside one CONTEXT can actually hide model fragmentation.

- Symptoms of splintering within a supposed single context: interfaces that don't match, unexpected behavior, and — earliest — confusion of language. Two failure categories: **duplicate concepts** (same concept implemented twice, updated and reanalyzed in two places) and **false cognates** (same term, different meanings — subtler and more harmful).

**CONTINUOUS INTEGRATION** — When many people work in the same BOUNDED CONTEXT the model tends to fragment; breaking into ever-smaller contexts loses integration and coherency. Therefore: institute a process of merging all code and other implementation artifacts frequently, with automated tests to flag fragmentation quickly. Relentlessly exercise the UBIQUITOUS LANGUAGE to hammer out a shared view of the model as the concepts evolve in different people's heads. It operates at two levels: integration of model concepts (constant communication) and integration of implementation (reproducible merge/build, automated tests, small upper limit on lifetime of unintegrated changes). CI is essential only *within* a BOUNDED CONTEXT — do not pay this cost across contexts.

**CONTEXT MAP** — An individual BOUNDED CONTEXT gives no global view; people blur edges and connections bleed into each other; code reuse between CONTEXTS is a hazard. Therefore: identify each model in play on the project and define its BOUNDED CONTEXT, including implicit models of non-object subsystems. Name each BOUNDED CONTEXT and make the names part of the UBIQUITOUS LANGUAGE. Describe the points of contact between the models, outlining explicit translation for any communication and highlighting any sharing. Map the existing terrain; take up transformations later. The map must reflect reality, not the ideal — "put a dragon on the map" where things are entangled; change the map only after reality changes. Example: booking vs. Network Traversal Service — two models of shipping operations connected by a small two-way translator object (RouteSpecification ↔ location-code list, node IDs ↔ Itinerary) jointly maintained and heavily tested by both teams. Test contact points between contexts intensively — "trust, but verify."

**SHARED KERNEL** — Full CI across teams may be too much overhead, but uncoordinated teams on closely related apps produce work that doesn't fit together, duplicating effort and losing the common language. Therefore: designate some subset of the domain model that the two teams agree to share (model plus associated code and database design). This explicitly shared stuff has special status and shouldn't be changed without consultation with the other team. Integrate a functional system frequently, but less often than each team's internal CI, running both teams' tests at each merge. The kernel is often the CORE DOMAIN or GENERIC SUBDOMAINS. Goal: reduce (not eliminate) duplication and ease integration.

**CUSTOMER/SUPPLIER DEVELOPMENT TEAMS** — When one subsystem feeds another (all dependencies one way), the downstream team can be helpless before upstream priorities, while upstream fears breaking downstream. Therefore: establish a clear customer/supplier relationship between the two teams. In planning sessions, make the downstream team play the customer role to the upstream team; negotiate and budget tasks for downstream requirements so that everyone understands the commitment and schedule. Jointly develop automated acceptance tests that validate the expected interface; add them to the upstream team's continuous-integration suite, freeing upstream to change without fear. Works when both teams answer to the same management (or a genuine commercial relationship exists). Example: yield-analysis (data warehouse, own models) downstream of booking — separate CONTEXTS by tooling and model, formalized in iteration planning.

**CONFORMIST** — When upstream has no motivation to serve downstream (different hierarchies, indifferent supplier), altruistic promises won't be fulfilled and downstream plans built on them fail. Three paths: abandon the dependency (SEPARATE WAYS); build your own model behind an ANTICORRUPTION LAYER if the upstream design is unusable; or, if upstream's design is decent and compatible: therefore, eliminate the complexity of translation between BOUNDED CONTEXTS by slavishly adhering to the model of the upstream team. This cramps the downstream's style and rarely yields the ideal model, but enormously simplifies integration and gives a shared UBIQUITOUS LANGUAGE with the supplier. Following isn't always bad: with a large-interface off-the-shelf component, conform to its model — its design has knowledge crunched into it, and you may be dragged into a better design. Conform wholeheartedly: extension only, no modification of the followed model.

**ANTICORRUPTION LAYER** — When a new system must have a large interface with a legacy or external system, the difficulty of relating the two models can overwhelm the new model, corrupting it with ad hoc accommodations to foreign semantics; even "primitive data" carries model meaning. Therefore: create an isolating layer to provide clients with functionality in terms of their own domain model. The layer talks to the other system through its existing interface, requiring little or no modification to the other system. Internally, the layer translates in both directions as necessary between the two models. It is a *conceptual* translator, not a data-transport mechanism. Implementation: a combination of FACADES (simplified interface in the *other* system's model), ADAPTERS (one per SERVICE, converting requests), and translator objects (the actual conceptual/data conversion), plus whatever communication mechanism is needed; can be bidirectional. Example: a minimal new booking application passes shipments through an ACL to the legacy system, allowing incremental replacement release by release. The Great Wall analogy: isolation permits regulated commerce while blocking corruption — but walls are expensive; weigh cost against benefit.

**SEPARATE WAYS** — Integration is always expensive, and sometimes the benefit is small; it forces compromises and coordination overhead. If two functional parts don't call each other's functionality, share touched objects, or share data, integration may be unnecessary — features related in a use case need not be integrated in the model. Therefore: declare a BOUNDED CONTEXT to have no connection to the others at all, allowing developers to find simple, specialized solutions within this small scope. Features can still be composed at the middleware/UI level (links on an intranet page, buttons on a desktop). Cost: models developed in isolation are very hard to merge later. Example: a stuck insurance project delivered several small standalone adjuster tools "almost overnight" once it stopped forcing integration.

**OPEN HOST SERVICE** — When a subsystem must integrate with many others, custom translators for each bog the team down in maintenance. Therefore: define a protocol that gives access to your subsystem as a set of SERVICES. Open the protocol so that all who need to integrate with you can use it. Enhance and expand the protocol to handle new integration requirements, except when a single team has idiosyncratic needs — then use a one-off translator to augment the protocol for that special case so the shared protocol stays simple and coherent. Pays off only when the subsystem's resources form a cohesive service set and integrators are numerous.

**PUBLISHED LANGUAGE** — Direct translation into or out of existing domain models is a poor interchange medium: those models are complex, poorly factored, undocumented, and freezing one as an interchange format blocks its evolution. Therefore: use a well-documented shared language that can express the necessary domain information as a common medium of communication, translating as necessary into and out of that language. "Published" means readily available and documented well enough that independent interpretations are compatible. Examples: Chemical Markup Language (CML), an XML dialect for chemistry, unlocked shared tooling (the JUMBO browser) for everyone; the DB2 SQL interface as a published language for a persistence port. Keep the published language distinct from the host's internal model so refactoring stays free.

- The blind men and the elephant: with no integration, disagreeing models don't matter; minimal integration only requires translating the few shared aspects (the elephant's location); fuller unification almost always means creating a *new* model (wall + trees + rope + snake → animal with body, legs, tail, trunk). Successful unification hinges on minimalism — dropping wrong implications matters more than adding features.
- Choosing a context strategy: decisions must be made (or at least understood) team-wide, and often above team level; politics frequently trumps technical merit — map reality first, transform pragmatically. Forces favoring larger contexts: smoother user-task flow, one model to understand, translation is hard, shared language. Favoring smaller: less communication overhead, easier CI, less demand for versatile abstraction, room for specialized jargons/dialects. Roughly one team per BOUNDED CONTEXT (one team can own several; several teams sharing one is hard).
- External systems: first consider SEPARATE WAYS; if integration is essential, choose CONFORMIST (peripheral extension, large interface, tolerable model) or ANTICORRUPTION LAYER (major new system, small interface, or bad upstream design). Don't assume an external/legacy system is internally unified — check.
- Specialized jargons of user groups can justify separate contexts and dialects, but beware rationalizing quirky parochial models; a later deep model may unify dialects — accept the opportunity when it arises, don't plan on it.
- Deployment couples to context strategy: SHARED KERNEL demands coordinated releases; translation layers mark the hot spots; SEPARATE WAYS is simplest. Feasibility of deployment should feed back into where boundaries are drawn.
- The spectrum trades seamless functional integration against coordination/communication overhead: single CONTEXT with CI → SHARED KERNEL → CUSTOMER/SUPPLIER → CONFORMIST → ANTICORRUPTION LAYER → SEPARATE WAYS. More ambitious unification requires more control over all subsystems involved.
- Transformations are incremental game plans, too big for one refactoring: SEPARATE WAYS → SHARED KERNEL (set up merge process and empty shared test suite first; start with a small, duplicated, non-CORE subdomain; joint 2–4 person modeling group; remove obsolete translations); SHARED KERNEL → full CI (harmonize processes, circulate team members, distill both models, merge the CORE fast — high-overhead phase); phasing out a legacy (iteration by iteration move functions into favored systems, extend then prune the ACL, ignore dead legacy modules if excision is impractical); ad hoc protocols → OPEN HOST SERVICE → PUBLISHED LANGUAGE (prefer an industry standard; else base the interchange language on the host's distilled CORE, e.g., in XML — but do not equate interchange language and host model).

**Apply:**
- Never combine or reuse code across model boundaries without explicit translation; treat cross-context reuse as a hazard.
- Before any modeling work, draw the CONTEXT MAP as it actually is, name every context, put the names in the ubiquitous language, and only then plan changes.
- Within each context, enforce continuous integration (frequent merges, automated tests, constant exercise of the shared language); across contexts, define the relationship explicitly using the spectrum (shared kernel / customer-supplier / conformist / ACL / open host + published language / separate ways).
- Choose the relationship by control and cooperation: cooperative teams → shared kernel or customer/supplier; uncooperative but usable upstream → conformist; uncooperative and messy → anticorruption layer; no real need → separate ways.
- Test interfaces between contexts with jointly owned automated suites; treat translator objects as jointly maintained property.
- Make boundary changes as staged transformations over many iterations, never as one big reorganization.

## Ch 15 — Distillation

Even an isolated domain layer of a large system can be unmanageably complex. Distillation separates the mixture to extract the essence: the CORE DOMAIN — the part of the model that differentiates the application and makes it worth building. Strategic distillation gives everyone the big picture, focuses effort and top talent on the highest-value part, and guides refactoring, outsourcing, and buy-vs-build decisions. Techniques escalate in commitment: DOMAIN VISION STATEMENT → HIGHLIGHTED CORE → GENERIC SUBDOMAINS → COHESIVE MECHANISMS → SEGREGATED CORE → ABSTRACT CORE.

**CORE DOMAIN** — In a large system the essence of the model, the real business asset, gets obscured among many necessary components; the harsh reality is that not all parts of the design will be equally refined, and skilled developers gravitate to interesting technical or generic problems, leaving the differentiating part to the least experienced. Therefore: boil the model down. Find the CORE DOMAIN and provide a means of easily distinguishing it from the mass of supporting model and code. Bring the most valuable and specialized concepts into sharp relief. Make the CORE small. Apply top talent to the CORE DOMAIN, and recruit accordingly. Spend the effort in the CORE to find a deep model and develop a supple design; justify investment in any other part by how it supports the distilled CORE. Cautionary example: a syndicated-loan system where the talented developer built an elegant peripheral commenting feature while the mission-critical loan module became an incomprehensible tangle. The CORE is point-of-view dependent (a money model is generic for most apps, CORE for a currency trader) and evolves through iterations. It can rarely be bought or outsourced — the value comes from a stable team accumulating specialized knowledge; a framework that constrains your CORE is losing you an asset.

**GENERIC SUBDOMAINS** — Some parts of the model add complexity without capturing specialized knowledge (org charts, accounting, time zones); they clog the model and absorb the best developers. Therefore: identify cohesive subdomains that are not the motivation for your project. Factor out generic models of these subdomains and place them in separate MODULES. Leave no trace of your specialties in them. Give their development lower priority than the CORE, and avoid assigning your core developers to them (they'd gain no domain knowledge); consider off-the-shelf solutions or published models. Four sourcing options with trade-offs: off-the-shelf (mature but integration/evaluation cost), published design or model (e.g., Fowler's Analysis Patterns; use formalized fields like accounting outright — implement only the self-consistent subset you need), outsourced implementation (frees core team, forces interface-oriented design; demand delivered unit tests, write acceptance tests), in-house (best fit, ongoing burden). Generic does NOT mean reusable: don't design for reuse — model reuse matters more than code reuse — but stay strictly within the generic concept; industry-specific traces impede growth and belong in the CORE or a specialized subdomain. Example (two time-zone efforts): a shipping project correctly assigned a short-term contractor to adapt the BSD zone database *after* the CORE was mature and requirements clear; an insurance project burned its long-term developers on a speculative, general time-zone model while the CORE went undeveloped. Risk management corollary: don't let "get something end-to-end running" seduce you into building a peripheral subsystem first — except with proven skills in a familiar domain, base the first-cut system on some part of the CORE DOMAIN, however simple, because the CORE is where unexpected difficulty and project-killing risk live.

**DOMAIN VISION STATEMENT** — Early on there is no model to point at, and later a shallow way to explain the system's value is still needed; critical model concerns may span BOUNDED CONTEXTS. Therefore: write a short description (about one page) of the CORE DOMAIN and the value it will bring — the "value proposition." Ignore those aspects that do not distinguish this domain model from others. Show how the domain model serves and balances diverse interests. Keep it narrow. Write this statement early and revise it as you gain new insight. Use it during all phases to guide resource allocation and modeling choices. Content is about the model's domain value ("the model can represent passenger priorities and airline booking strategies and balance them by flexible policies"), never UI, performance, or technology stack.

**HIGHLIGHTED CORE** — A vision statement identifies the CORE only broadly; different people pick different elements, and mentally re-filtering the model wastes concentration. Structural change is ideal but often impractical short-term. Therefore, in either of two lightweight forms: (1) the *distillation document* — write a very brief document (three to seven sparse pages) that describes the CORE DOMAIN and the primary interactions among CORE elements; minimalist, understandable by nontechnical members, an entry point, not a design doc. (2) the *flagged CORE* — flag the elements of the CORE DOMAIN within the primary repository of the model, without particularly trying to elucidate its role; make it effortless for a developer to know what is in or out of the CORE (page tabs and yellow highlighter turned a 200-page consortium model into an asset; UML stereotypes or code annotations work too). Process tool: if a change would force the distillation document to change, it's a CORE change — consult the team before, notify everyone after; all other changes can proceed with full autonomy.

**COHESIVE MECHANISMS** — Computations sometimes reach a complexity that bloats the design: the conceptual "what" is swamped by the mechanistic "how." Therefore: partition a conceptually COHESIVE MECHANISM into a separate lightweight framework. Particularly watch for formalisms or well-documented categories of algorithms. Expose the capabilities of the framework with an INTENTION-REVEALING INTERFACE. Now the other elements of the domain can focus on expressing the problem, delegating the intricacies of the solution to the framework. Example: an org-chart model delegated "who in this chain of command can approve this?" to a graph-traversal framework using standard textbook graph terminology; the model just declared people as nodes and relationships as edges. Distinction: a GENERIC SUBDOMAIN expresses some aspect of how the team sees the domain; a COHESIVE MECHANISM solves a computational problem the model poses — "a model proposes; a COHESIVE MECHANISM disposes." Exception: a proprietary mechanism that *is* the product's value (e.g., a bank's risk-rating algorithms) can belong to the CORE. Full-circle refactorings that reabsorb a mechanism are common and fine — the end state is a deeper model that still differentiates facts, goals, and mechanisms. Distillation payoff: with mechanisms encapsulated behind intention-revealing, side-effect-free, assertion-characterized interfaces, CORE code approaches a declarative style.

**SEGREGATED CORE** — Model elements partially serving the CORE and partially supporting it, plus tight coupling to generic elements, choke the CORE; designers can't see the most important relationships. Therefore: refactor the model to separate the CORE concepts from supporting players (including ill-defined ones) and strengthen the cohesion of the CORE while reducing its coupling to other code. Factor all generic or supporting elements into other objects and place them into other packages, even if this means refactoring the model in ways that separate highly coupled elements. Steps: identify a CORE subdomain (use the distillation document) → move classes to a module named for the unifying concept → sever non-expressive data/functionality into other packages → refactor the CORE module to sharpen its relationships → repeat. Sacrificing the cohesion of a supporting module to gain CORE cohesion is a net win. It is a whole-team decision requiring ongoing shared definition of what is CORE. Example: shipping model — the vision statement ("increase visibility of operations… fulfill customer requirements") ruled out billing as CORE and produced a `Delivery` package; segregation itself yielded insights (Customer Agreement constrains Handling Step, attach the agreement directly to Cargo so Customer drops out of the CORE).

**ABSTRACT CORE** — Even the CORE model can be too detailed to communicate the big picture, and heavy interaction between subdomain modules forces either many cross-references or obscure indirection. Therefore: identify the most fundamental concepts in the model and factor them into distinct classes, abstract classes, or interfaces. Design this abstract model so that it expresses most of the interaction between significant components. Place this abstract overall model in its own MODULE, while the specialized, detailed implementation classes are left in their own MODULES defined by subdomain. Specialized modules then reference the ABSTRACT CORE, not each other. This is refactoring to deeper insight, not a mechanical move of frequently referenced classes; done well, the ABSTRACT CORE reads like the distillation document rendered in rigorous code.

- Deep models distill: distillation is not only partitioning; it is continuously refactoring the CORE toward a deep model and supple design that expresses the domain in combinable elements.
- Choosing refactoring targets on a poorly factored system: neither "refactor everything" nor purely pain-driven fixing works. When pain-driven, check whether the root involves the CORE or its relationship to a supporting element and fix that first; when free, spend effort on segregating the CORE and purifying supporting subdomains into GENERIC form — most bang per refactoring buck.

**Apply:**
- Identify the CORE DOMAIN explicitly and early; write a one-page domain vision statement and keep a 3–7 page distillation document (or flag CORE elements in code) current.
- Put the strongest long-term developers, paired with domain experts, on the CORE; give generic subdomains to contractors, off-the-shelf products, or published models.
- Base the first end-to-end slice of a new system on the CORE, not on an easy peripheral subsystem.
- Factor generic subdomains out with no trace of your specialty in them; do not gold-plate them for reuse.
- Extract complex computations into cohesive mechanisms behind intention-revealing interfaces so the CORE reads declaratively; keep a mechanism in the CORE only if it is itself the proprietary value.
- Prioritize refactorings by CORE impact; treat any change that alters the distillation document as requiring team consultation.

## Ch 16 — Large-Scale Structure

In a big system, even well-modularized, distilled models leave developers unable to see the forest for the trees: uncoordinated local decisions produce a jumble no one can interpret as a whole. A large-scale structure is a language of high-level concepts and/or rules that establishes a design pattern for the entire system, letting the role of any part be understood without knowing its details. It usually spans BOUNDED CONTEXTS, lives mostly outside UML and code, and — unlike the CONTEXT MAP — is optional: impose one only when a fitting structure is found, because an ill-fitting structure is worse than none. Opening story: a satellite-communications simulator regained intelligibility by layering the whole design around aspects of the communications system.

**EVOLVING ORDER** — Design free-for-alls produce systems no one can understand or maintain, but up-front architectures freeze assumptions, straitjacket developers, and get dumbed-down or subverted. The problem is not guiding rules but their rigidity and source. Therefore: let this conceptual large-scale structure evolve with the application, possibly changing to a completely different type of structure along the way. Don't overconstrain the detailed design and model decisions that must be made with detailed knowledge. A structure should be minimal ("less is more"), leave contexts freedom to vary locally, tolerate flagged exceptions (too many exceptions means change or discard it), and never require heroic battles to amend.

**SYSTEM METAPHOR** — Software designs are abstract and hard to grasp; developers and users need a tangible shared view. Therefore: when a concrete analogy to the system emerges that captures the imagination of team members and seems to lead thinking in a useful direction, adopt it as a large-scale structure. Organize the design around this metaphor and absorb it into the UBIQUITOUS LANGUAGE. But because all metaphors are inexact, continually reexamine the metaphor for overextension or inaptness, and be ready to drop it if it gets in the way. Example: the network "firewall" shaped an industry and made products interchangeable and comprehensible — yet also produced barriers that were insufficiently selective and blind to inside threats. The XP notion of a "naive metaphor" (the domain model itself) should be retired: a mature domain model is anything but naive, and the UBIQUITOUS LANGUAGE already fills the metaphor's role when no apt analogy exists.

**RESPONSIBILITY LAYERS** — With every object's responsibilities handcrafted, there is no uniformity and no way to handle large swaths of the domain together; yet many domains have natural stratification — concepts that change at different rates and for different reasons against backgrounds of other concepts. Therefore: look at the conceptual dependencies in your model and the varying rates and sources of change of different parts of your domain. If you identify natural strata in the domain, cast them as broad abstract responsibilities. These responsibilities should tell a story of the high-level purpose and design of your system. Refactor the model so that the responsibilities of each domain object, AGGREGATE, and MODULE fit neatly within the responsibility of one layer. (Use the relaxed variant: a layer may access any lower layer.) Recurring layers for enterprise systems, top to bottom of dependency: **Decision Support** (what action should be taken? analysis, planning tools), **Policy** (what are the rules and goals? passive constraints on other layers), **Commitment** (what have we promised? goals like policy, but emerging from operations — prominent in service businesses), **Operations** (what is being done? current reality of activity), **Potential/Capability** (what could we do? resources, fixed assets, vendor contracts — prominent in asset-heavy businesses). Keep to four or five layers at most; the structure must be ferociously distilled. Layer choice is a business-modeling decision, judged by *storytelling*, *conceptual dependency* (upper meaningful against lower; lower meaningful alone), and *conceptual contours* (layers absorb shearing between different rates/sources of change). Example: the shipping model split into Operations (Cargo, Route Specification, Itinerary), Capability (Transit Leg, Customer — because the business cultivates long-term relationships), then Decision Support (Router); the "is preferred" flag on Transport Leg violated the layering and was refactored into an explicit Route Bias Policy — a structure grounded in deep domain understanding pushes the model toward clarity. Once adopted, the structure constrains new design: an Operations object (Cargo) may not depend on a Decision Support service (HazMat routing policy) — move the responsibility (Router collects applicable policies) even if the local design is no better, because project-wide consistency is worth modest local trade-offs. Higher layers observe lower ones via events to avoid upward dependencies (part in wrong machine → event → policy layer reacts).

**KNOWLEDGE LEVEL** — In applications where the roles and relationships among ENTITIES vary by installation or at runtime (org structures, payroll rules), a static model overconstrains while a fully flexible one underconstrains and can't enforce the organization's own rules; objects sprout type references and multi-use attributes; "nestled into our model is another model that is about our model." Therefore: create a distinct set of objects that can be used to describe and constrain the structure and behavior of the basic model. Keep these concerns separate as two "levels," one very concrete, the other reflecting rules and knowledge that a user or superuser is able to customize. This is the REFLECTION pattern (base level / meta level) applied to the domain layer — but built of ordinary objects, deliberately *not* fully general, and not a layer: dependencies run both directions between levels. Example: Employee Type (superuser-edited) constrains Employee (daily-edited); "an Employee Type is assigned to either Retirement Plan or either payroll" exposed the implicit Payroll concept, which was factored out once the level split made it visible. Use sparingly: a complex knowledge level makes configurators into de facto meta-programmers, and knowledge-level changes still force migration of operational objects.

**PLUGGABLE COMPONENT FRAMEWORK** — When many applications must interoperate on the same abstractions but are designed independently, translation among many BOUNDED CONTEXTS limits integration, and a SHARED KERNEL is infeasible for teams that don't work closely. Therefore: distill an ABSTRACT CORE of interfaces and interactions and create a framework that allows diverse implementations of those interfaces to be freely substituted. Likewise, allow any application to use those components, so long as it operates strictly through the interfaces of the ABSTRACT CORE. The hub is an ABSTRACT CORE within a SHARED KERNEL; multiple contexts can hide behind component interfaces; a PUBLISHED LANGUAGE can serve as the plug-in interface. Downsides: very hard to design (requires a deep, mature model — it should never be a project's first or second large-scale structure; the best examples follow several fully developed specialized applications), and it freezes refinement of the ABSTRACT CORE, limiting applications that need a different approach. Example: SEMATECH's CIM Framework for semiconductor manufacturing — abstract interfaces (Process Machine etc.) plus interaction rules let any vendor's machine-control component plug into any conforming MES application. Analogy: the AIDS Memorial Quilt — a few simple panel rules (3×6 feet, durable fabric, one person per panel) let thousands contribute independently to one integrated whole.

- How restrictive should a structure be? A spectrum from loose (SYSTEM METAPHOR) to restrictive (PLUGGABLE COMPONENT FRAMEWORK). Extra rules (e.g., prescribed event-based communication between layers) buy uniformity and better fit of parts, at the cost of flexibility — and rigid communication paths may be impractical across heterogeneous BOUNDED CONTEXTS. Resist the temptation to build frameworks to enforce the structure; its most important contribution is conceptual coherence and domain insight — each structural rule should make development easier.
- Refactoring toward a fitting structure: a useful structure can only be found from deep domain understanding, which arrives iteratively — so expect to refactor the structure in throughout the life cycle, and fearlessly rethink it. Cost controls: *minimalism* (address only the most serious concerns; start loose — a metaphor or a couple of responsibility layers); *communication and self-discipline* (the whole team must follow the structure; its terms must enter the UBIQUITOUS LANGUAGE, because tests and code don't enforce it); *restructuring yields supple design* (each transformation makes the next easier — the leather-jacket effect: repeatedly transformed models identify and loosen their principal axes of change); *distillation lightens the load* (a distilled CORE leaves less to restructure; fit GENERIC SUBDOMAINS into single layers or single components).

**Apply:**
- Do not impose a large-scale structure by default; adopt one only when the system has outgrown modules-plus-distillation, and choose the loosest structure that restores comprehensibility.
- Derive structure from the domain's natural stratification (rates and sources of change, conceptual dependency), and require it to tell the business's story; refactor model elements that straddle layers into explicit concepts.
- Treat the structure as evolving: flag exceptions, and change or discard the structure when exceptions multiply or it forces many awkward designs.
- Put the structure's vocabulary into the ubiquitous language and rely on team discipline, not enforcement frameworks.
- Use a knowledge level only where user-configurable rules would otherwise distort the model; keep it specialized, not general.
- Attempt a pluggable component framework only after multiple mature applications exist in the domain.

## Ch 17 — Bringing the Strategy Together

The three strategic principles — context, distillation, and large-scale structure — are complementary, not substitutes. Strategy must be assessed against the project as it is, decided in tight feedback with application development, and kept minimal and evolving.

- Combining structure and contexts: a large-scale structure can live inside one BOUNDED CONTEXT (raising the complexity ceiling of a unified model) or span the CONTEXT MAP, organizing the relationships among contexts with one project-wide vocabulary. A legacy system that ignores the structure can still be *characterized* by it — e.g., its FACADE services each assigned to a layer. Different structures per context are possible but erode the structure's unifying value.
- Combining structure and distillation: layers clarify the relationships among CORE DOMAIN modules and GENERIC SUBDOMAINS; and the structure itself (e.g., the potential/operations/policy/decision-support split) may be part of the CORE DOMAIN — a distilled business insight, especially valuable when the project spans many contexts and no model object has project-wide meaning.
- Assessment first. Before strategizing: (1) Can you draw a consistent CONTEXT MAP? (2) Is there a UBIQUITOUS LANGUAGE rich enough to help development? (3) Is the CORE DOMAIN identified; is there a DOMAIN VISION STATEMENT? (4) Does the project's technology work for or against MODEL-DRIVEN DESIGN? (5) Do developers have the necessary technical skill? (6) Are developers knowledgeable about, and interested in, the domain?
- Who sets the strategy — two working models (rejecting wisdom-from-on-high): *emergent structure from application development* (a self-disciplined XP-style team, usually with an informal hands-on leader/coach as arbiter and communicator; loose inter-team committees when few, comparable, committed teams are involved), and *a customer-focused architecture team* (peers with application teams, discovering patterns alongside developers, getting hands dirty — different in every activity from an ivory-tower team even if identical on the org chart).
- Six essentials for strategic design decision making: (1) **Decisions must reach the entire team** — a strategy nobody follows is irrelevant; community-made decisions often propagate better than decreed ones. (2) **The decision process must absorb feedback** — only application developers have the depth of project/domain knowledge that subtle organizing principles require; rotate architects through application teams. (3) **The plan must allow for evolution** — set-in-stone top decisions hobble response to change; EVOLVING ORDER applies to strategy itself. (4) **Architecture teams must not siphon off all the best and brightest** — application building takes design skill; strategy teams need domain knowledge, not just technical stars; keep architecture part-time if needed. (5) **Strategic design requires minimalism and humility** — almost everything gets in the way of something; pare organizing principles down to what significantly improves clarity; your best idea will obstruct someone. (6) **Objects are specialists; developers are generalists** — don't split people into designers-who-don't-know-the-business and experts-who-don't-touch-technology; strategic design emerges from application design and must be done by people working with the code.
- The same goes for technical frameworks: evolution, minimalism, and involvement with application teams keep frameworks helpful. "Don't write frameworks for dummies" — team divisions assuming some developers can't design will fail; a good framework earns respect for its users, encapsulating drudgery while handing over powerful abstractions.
- Beware the master plan: Alexander's critique — master plans create totalitarian, not organic, order; they are simultaneously too precise (the totality) and not precise enough (the details), become obsolete, and alienate the community. Prefer a shared set of principles applied to every act of piecemeal growth.

**Apply:**
- Start any strategic engagement with the six-question assessment; fix the context map and language before anything else.
- Locate strategic decision-making with (or in tight rotation through) the application teams; never accept architecture handed down without a feedback loop.
- Keep every project-spanning rule minimal and revisable; prefer principles for piecemeal growth over master plans.
- Staff application teams with strong designers and strategy work with domain knowledge; resist elite-team skimming.
- Let one large-scale structure organize the context map where possible, and check whether the structure itself is a CORE-DOMAIN insight worth distilling.

## Conclusion

Project epilogues show that success is measured by sustained useful evolution, not by design stasis: a deep model enables new insight, and a supple design facilitates the ongoing change that is software's nature (the team that transformed an inherited ABSTRACT CORE "almost beyond recognition" was the design succeeding, not failing). The shipping project's partial failure traces to a culture that resisted iteration, released too late, and never closed the feedback loop from implementation back to the model; the Evant story shows a small skilled team with a supple, domain-driven code base absorbing enormous new demands and saving the company.

- The defining characteristic of a domain-driven project: priority on understanding the target domain and incorporating that understanding into the software — conscious cultivation of language, dissatisfaction with the current model as learning deepens, continuous refinement seen as opportunity and an ill-fitting model as risk, and design skill taken seriously.
- Between conservative patchworks of small applications and doomed big-bang unification there is a third way: piecemeal growth of big, richly functional systems on a deep model and supple design.
- Tools help thought; they cannot replace it — "creating good software is a learning and thinking activity"; attempts to automate what must be the product of thought are naive and counterproductive.

Appendix material — *The Use of Patterns in This Book*: pattern names (small caps in the book) are meant to become vocabulary, enabling economical discussion the way "kitchen" and "three-bedroom, two-bath" compress house design; standard elements prevent quirky, Peugeot-like designs only specialists can maintain. Pattern format: context, problem discussion, problem summary, "Therefore:" solution summary, consequences. A glossary defines the book's terms (unification: internal consistency of a model, each term unambiguous, no contradictory rules; strategic design: modeling and design decisions that apply to large parts of the system, decided at team level; supple design; deep model; etc.). References ground the strategic patterns in Alexander (pattern language, piecemeal growth), Fowler (Analysis Patterns — KNOWLEDGE LEVEL source), Buschmann et al. (layers, REFLECTION), Gamma et al. (FACADE, ADAPTER), Beck (XP), Fayad & Johnson (domain frameworks).

**Apply:**
- Judge a design by how well it supports continued change and deeper insight, not by how long it survives unchanged.
- Close the loop: feed implementation problems (performance, scaling) back into the model instead of patching code away from it.
- Release working software early and iterate; late exposure of model problems makes them expensive and politically unfixable.
- Use the pattern names as working vocabulary in design discussions so decisions stay economical and communicable.
