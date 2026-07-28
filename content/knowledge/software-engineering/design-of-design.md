---
title: The Design of Design — Frederick P. Brooks Jr. (2010)
description: "Book notes: the rational model of design is false, design iterates on its own goals, and conceptual integrity comes from a single architect working for the user."
tags: [design-process, conceptual-integrity, architecture, requirements, iteration, collaboration, constraints, exemplars, esthetics, design-tools, case-studies, book-notes]
---

Essays on how design processes really work, written to prod designers and design
managers into thinking hard about designing complex systems. The central corrective: the
Rational Model of design (given goals, explore a known tree of alternatives, maximize a
utility function) is false in every assumption — real design iterates on the goals
themselves, and the hardest part of design is deciding what to design. Great designs
come from great designers, not from process; conceptual integrity — unity, economy,
clarity — is the mark of greatness and is best achieved by a single architect (or a
two-person team) working for the user. The essays span process models, collaboration,
perspectives (constraints, budgeted resources, esthetics, exemplars), tools for
designers, growing great designers, and seven case studies from houses to System/360.

Structure: Part I — Models of Designing (Ch 1–5); Part II — Collaboration and
Telecollaboration (Ch 6–7); Part III — Design Perspectives (Ch 8–16); Part IV — A
Computer Scientist's Dream System for Designing Houses (Ch 17–18); Part V — Great
Designers (Ch 19–20); Part VI — Case Studies (Ch 21–27); Ch 28 — Recommended Reading.

## Preface

- Aim: prod designers and design project managers into thinking hard about the *process* of designing complex systems — an engineer's viewpoint, focused on utility and effectiveness but also efficiency and elegance.
- Audience: designers of many kinds (systematic design excluding intuition yields pedestrian follow-ons; intuitive design without system yields flawed fancies — how to weld the two?), design project managers (who must design their design process rather than replicate an oversimplified academic model or jury-rig one), and design researchers (challenged to address the larger questions again, even where social-science rigor fails).
- Core conviction: there are constants across design processes in very different media — Brooks designed in five (computer architecture, software, houses, books, organizations) and found the mental processes, iterations, constraints, and human interactions strikingly alike. "A science of design" is an impossible and misleading goal; this skepticism licenses opinionated essays grounded in intuition and experience rather than a monograph.
- Why the design process matters: gaps between best, average, and semi-competent practice are wide in every discipline; as much as a third of design cost is rework, the correction of mistakes. Mediocre design provably wastes the world's resources.
- The process has changed rapidly since WWII, and the changes are rarely discussed: team design is now the norm; teams are geographically dispersed; designers are divorced from both use and implementation (they can no longer build with their own hands what they design); designs live in computer models instead of drawings; formal processes are taught and often mandated.
- Retrospective observation across the case studies: the boldest design decisions, whoever made them, accounted for a high fraction of the goodness of the outcome. Made sometimes from vision, sometimes from desperation, they were always gambles — extra investment in hopes of a much better result.

## Ch 1 — The Design Question (Part I: Models of Designing)

Design processes have invariant properties across media, so designers in different fields can learn about their own craft by comparing experiences (Bacon's hypothesis). Design is the mental formulation — a plan formed in the mind for later execution — distinct from the artifact itself.

- OED essentials of *design*: plan, in the mind, for later execution. A design (noun) is a created object preliminary to and distinct from the thing being designed.
- Sayers's three aspects of creation: the Idea (formulation of conceptual constructs), the Energy/Implementation (in real media), and the Interaction (with users in real uses).
- These map to Brooks's essence (mental crafting of the conceptual construct) vs. accident/incident (the implementation process). Creation is complete only when someone uses the thing.
- The design can be complete before realization begins — Mozart: "Everything has been composed, just not yet written down." But for most makers, the incompletenesses and inconsistencies of our ideas become clear only during implementation; writing, experimentation, "working out" are essential disciplines even for theoreticians.
- The three phases operate recursively: implementation creates a space in which another design cycle occurs (Mozart implements the opera on paper; the conductor, interacting with it, conceives an interpretation, implements it with orchestra; the audience's interaction completes the process).
- The Design Concept is a real entity: an invisible shared Platonic ideal that architect and client gesture toward, distinct from any drawing or any particular detail therein. The constant concern is the conceptual integrity of the developing design.
- For the System/360 architecture team, the "real" System/360 was the Design Concept — a Platonic ideal computer; the physical models under construction were Plato's shadows; its most complete and faithful embodiment was the prose and diagrams of the programmer's manual, *Principles of Operation*, not silicon and steel.
- Value of naming the Design Concept: (1) great designs have conceptual integrity — unity, economy, clarity; they not only work, they delight (Vitruvius); (2) talking about the Concept per se, rather than derivative representations or partial details, vastly aids team communication. Unity of concept is the goal; it is achieved only by much conversation. Moviemakers use storyboards to keep design conversations on the Concept, not implementation details.
- Detailing surfaces conflicting versions of the Concept and forces resolution — resolve such conflicts by asking which choice better fits the Concept. Example: S/360's decimal datatype could have followed the integer or the character-string type; it was modeled on the character string, the form familiar to the largest community actually using decimal data (IBM 1401 users).
- Multiple concurrent implementations of one architecture drove the common architecture toward generality and cleanliness, insulating it from small cost-saving compromises.
- Scope distinctions: system design (engineer's view: utility, effectiveness, efficiency, elegance) vs. artistic design (delight, conveying meaning). Routine design (codified decision trees, automatable — 50-foot highway bridges, compilers for established languages on new platforms), adaptive design (modifying a preceding design for new purposes), and original design — this book's subject.

**Apply:**
- Treat the Design Concept as a first-class entity; discuss it directly in the team instead of arguing only about representations and details.
- When a detailed decision has strong arguments both ways, decide by which option better fits the Design Concept.
- Expect implementation to expose flaws in the idea; build/write early to "work out" the concept.
- Distinguish routine/adaptive work from original design and choose your process accordingly.

## Ch 2 — How Engineers Think of Design — The Rational Model

Engineers carry a clear, usually implicit, orderly model of design: fix a goal, desiderata, a utility function, and constraints, then systematically search a tree-structured design space, choosing at each node so as to optimize utility subject to feasibility.

- Elements of the model:
  - **Goal**: the primary objective ("build a beach house to take advantage of wind and wave").
  - **Desiderata**: secondary objectives — each conceived as saturating (each extra square foot of window adds less utility), with the terms roughly summed linearly.
  - **Utility function**: weights the desiderata by importance; the thing to be optimized.
  - **Constraints**: some binary (10-foot setback), some elastic with steeply rising penalties (schedule), some blithely concealing terrifying complexity ("must satisfy all the building codes").
  - **Budgets**: many constraints take the form of a fixed resource allocated among design elements — often not dollars (feet of ocean frontage; bits in a control register or instruction format; memory bandwidth; working days for Y2K fixes).
- Design as tree search: each decision narrows the space; at each node other paths could have been taken; at dead ends one backtracks. Search algorithms are well known and cleanly describable — for exhaustive search. Designers actually *satisfice* (Simon: make good enough without optimizing); engineers approximate depth-first search guided by hunches, experience, consistency, and esthetic taste (building architects tend to search more laterally among alternatives at every level).
- Two branch types in decision trees: **attribute branches** (independent sub-questions — each must be specified) and **alternative branches** (mutually exclusive options — choose one).
- The model was independently formulated at least three times, proving its naturalness: German mechanical engineering (Pahl & Beitz, the most widely used exposition), Simon's *Sciences of the Artificial* ("the theory of design is that general theory of search through large combinatorial spaces" — motivated by wanting to automate design via AI), and Royce's seven-step Waterfall.
- Royce introduced the waterfall as a straw man he then argued against, but many cited and followed the straw man rather than his more sophisticated models — "I made that mistake myself in my younger days, and publicly repented of it later." Royce's iteration is carefully limited: each step iterates with the immediately preceding and succeeding steps "but rarely with the more remote steps."
- What's right with the model: any systematization is a great step forward over "let's just start coding, or building." It provides clear steps for planning, definable milestones, project organization and staffing suggestions, a single vocabulary for team and stakeholders, and teachability — it tells novices where to begin.
- Further genuine advantages: early explicit statement of goals, desiderata, and constraints helps a team avoid wandering and breeds unification on purposes; planning the whole process before coding avoids much wasted effort; casting design as search of a space lifts designers' eyes beyond their personal experience.
- But the Rational Model is much too simplistic, even in Simon's rich version.

**Apply:**
- State goal, desiderata, constraints, and utility weighting explicitly at project start, even knowing they will change.
- Use the model as scaffolding for planning, milestones, and shared vocabulary — never as a literal description of how design will proceed.
- Satisfice deliberately: define "good enough" rather than pursuing an optimum through a combinatorial space.

## Ch 3 — What's Wrong with This Model?

The Rational Model describes how we think design ought to work, not how it works in real life; every one of its assumptions fails for original design, and following it slavishly produces bloated products and schedule/budget/performance disasters.

- **We don't really know the goal when we start.** The most serious shortcoming: "The hardest part of design is deciding what to design."
  - Brooks's missile-company story: weeks of daily report revisions ("That's fine — it is what I asked for — but could you change it so that...?") taught him that "the most useful service I was performing for my client was helping him decide what he really wanted."
  - Hence: "A chief service of a designer is helping clients discover what they want designed."
  - Rapid prototyping is an essential tool for formulating precise requirements; not only is design iterative — the design-goal-setting process is itself iterative. Knowing complete product requirements up front is a quite rare exception, not the norm, despite the literature treating "product requirements" as a given.
- **We usually don't know the decision tree — we discover it as we go.** Original designs have enough novelty in goal, desiderata, constraints, and fabrication technology that no one can map the tree a priori. Few designers work in depth on even 100 projects in a working life, so no individual has explored the discipline's tree. Engineers, unlike scientists, rarely explore alternatives not clearly on the way to a solution. Designers make a decision, then see the alternatives it opens and forecloses.
- **The nodes are not single decisions but tentative complete designs.** Choices in one branch are linked to those in others by exclusion, affinity, or trade-off; at each node one faces a choice among multiple tentative complete designs. The ordering of decisions matters greatly (Parnas: put the decisions least apt to change nearest the root, making the design flexible — a fundamental aim of both object-oriented design and agile methods). The combinatorics boggle the mind, as with chess move trees.
- **The goodness function cannot be evaluated incrementally.** Many goodness measures (performance, cost) depend heavily on subsequent detailing all the way to the leaves. So designers estimate and trim the tree as they go:
  - From experience, direct and surrogate ("the B5000 explored descriptor architecture; the performance hit was too great") — "the most potent reason to study design history is to learn what doesn't work, and why."
  - From simple estimators (square-foot costs, instruction mixes) — dangerous, because the approximation may lop off feasible branches: an architect quoted high costs for pushing out a wall under an already-committed roof, when the marginal cost was actually very low. "One can often get something for nothing, if one has previously bought nothing for something."
- **The desiderata and their weightings keep changing.** Schön: the situation "talks back," and the designer reflects-in-action on the very construction of the problem. As trade-offs are pondered, a new understanding of the whole problem emerges and weightings shift; the client's understanding grows too.
  - A seemingly low-weight desideratum discovered deep into design ("Where will guests at meetings put their coats?") tipped the big scales and moved the Master Bedroom to the other end of the house.
  - Opportunities appear to add goodness at very low marginal cost, bringing in desiderata never on the original list — which then acquire value worth preserving (Sitterson Hall's serendipitous conference-hosting suite, contemplated in no program, now a feature any revision would preserve).
- **The constraints keep changing.** Even with everything else fixed, design would still be iterative:
  - The environment changes (new setback rules, annual electrical-code updates, a vendor withdraws a chip). The world keeps changing while we design.
  - Discovery changes them (builders hit solid rock; chip cooling newly becomes a constraint).
  - Constraints also *disappear* — and deep in design we usually neither notice nor remember which alternatives they foreclosed.
  - Remedy: list known constraints explicitly at the start in the **design program** (a document prepared with the client setting forth goal, desiderata, constraints — not the same as a contractual requirements statement). Explicit listing smokes constraints out early and radically improves the odds of recognizing when one goes away. Periodically rescan: "Can this constraint now be removed because the world has changed? Can it be entirely circumvented by working outside the design space?"
  - Breakthroughs sometimes come entirely outside the design space: Brooks solved an intractable setback-vs.-Music-Room conflict by buying a 5-foot strip of land from his neighbor — cheaper and faster than a variance, and it liberated other parts of the design.
- **Designers just don't work that way.** The most devastating critique, though hardest to prove.
  - Cross: empirical studies repeatedly find "intuitive" features of design ability the most effective and relevant; "in practice, designing seems to proceed by oscillating between sub-solution and sub-problem areas, as well as by decomposing the problem and combining sub-solutions" — not in staged sequence.
  - Schön: Technical Rationality (heir of discredited positivism) ignores problem *setting* — "in real-world practice, problems do not present themselves as givens. They must be constructed from the materials of problematic situations which are puzzling, troubling, and uncertain."
  - Cray: "I'm supposed to be a scientific person, but I use intuition more than logic in making basic decisions."
- **Yet the model persists** — enshrined in VDI-2221 (German mechanical engineering standard) and DoD-STD-2167A; Dorst (2006): Simon's rational problem-solving paradigm is still dominant in design methodology. Even sympathetic teachers (Wallace: "I put up the Pahl and Beitz diagram... And then my very next slide says, 'But this is not the way real designers work'") may not be echoed by younger teachers without design experience.
- **Why the wrong model matters:** designers are right-brained, spatial people (Brooks's talent test: "Where is next November?" — strong candidates have a spatial calendar model); process models live in our minds as diagrams and subconsciously influence much of our thinking, so a deficient model hinders us in ways we cannot fully know and can barely suspect. We mis-educate successors, teaching modes of working we don't follow ourselves. And the model "leads us to demand up-front statements of design requirements... to believe that such can be formulated... to make contracts with one another on the basis of this enshrined ignorance."
- "The Waterfall Model is wrong and harmful; we must outgrow it."

**Apply:**
- Treat goal-discovery as part of the job: prototype early and iterate with the client to find out what they really want.
- Write a design program listing goal, desiderata, and all known constraints; rescan it regularly for constraints that have vanished or can be removed by acting outside the design space.
- Order decisions so the ones least likely to change are made first; keep the design flexible below them.
- Expect and welcome the situation's "back-talk": let late-discovered requirements re-open earlier decisions rather than patching around them.
- Never contract or commit on the assumption that requirements are complete.

## Ch 4 — Requirements, Sin, and Contracts

Requirements set by committee before design begins produce grossly obese wish lists with no advocate for the product itself; contracts — necessary because humans are fallen — are what force this too-early binding of goals and best explain the Waterfall Model's persistence.

- The LHX/Comanche horror story: a next-generation light attack helicopter's requirements — fly fast, low, at night, armed and armored — plus, without a change of inflection, "it must ferry itself across the Atlantic." The requirements committee had (as best Brooks recalls) neither an aircraft engineer nor a helicopter pilot — mostly bureaucrats skilled at representing their groups in inter-group negotiations. No one had weighed what the extreme requirement cost elsewhere in the design; its extremity didn't even bother the briefer.
- Committee dynamics: each player has a wish list garnered from constituents, an ego, and a reputation staked on getting his list adopted; logrolling is endemic — "I won't naysay your wish, if you won't naysay mine."
- "Who advocates in the requirements process for the product itself — its conceptual integrity, its efficiency, its economy, its robustness? Often, no one." In a Waterfall process, requirements are set before design begins, so an architect can offer only opinion "unbuttressed as yet by facts."
- The result: a grossly obese set of requirements, the union of many wish lists, assembled without constraints, neither prioritized nor weighted — the social forces forbid the painful conflicts of weighting, much less prioritizing.
- Designers then implicitly weight the official requirements with their own personal user models; the failure to weight decouples designers from the deep user knowledge the requirements-setters actually possess.
- Committee specification tends to produce products that are too rich (Detroit cars, bloated software, unbuildable IRS and FBI systems) — perhaps why super-ambitious software systems are so prone to total disaster. For OS/360, Brooks as Project Manager rejected the Marketing Division committee's requirements document as totally impractical and had a quite small team of architects, marketers, and implementers extract the essence.
- Speed, smallness, robustness, ease of use have no ardent champions in a requirements committee — a particular feature's effect on them can't be known early, while individual features have ardent champions. And designers themselves cause bloat too, adding things to "complete" the design or make it consistent (Boehm's experiment; Brooks did it on Stretch).
- Fighting requirements bloat and creep ("by both birth control and infanticide"), per the Air Force Studies Board:
  - Dramatically successful programs of yore had a few clear overriding objectives plus schedule urgency; top-level requirements were broken down into sub-requirements *during* development under hard-driving, capable managers continually balancing function against schedule and cost. Development times have since doubled or tripled while "oversight" layers replaced urgency.
  - Define clear key performance parameters by Milestone A, but detailed requirements only between Milestone A and B — "getting to a state of clear and complete system-level requirements requires the interaction with potential contractors that occurs between Milestones A and B."
  - Top recommendation: appoint early strong, seasoned, domain-knowledgeable managers who stay through initial delivery, empowered to "tailor standardized processes and procedures as they feel is necessary."
  - Use a Requirements Traceability Matrix: every detailed requirement must derive from one or more initial overall requirements — nothing sneaks in from a user representative's request or a designer's desire to do something clever, novel, and putatively useful.
  - Schedule urgency is the best defense against requirements creep (Brooks's own best defense in system building too).
- **Sin.** With ideal actors — an ungreedy client, an architect who is truly the client's agent, a builder delivering best value/cost on schedule, honesty and excellent communication all around — cost-plus gives the client most value per dollar, design-build is fastest, and an explicit Spiral process yields the best-suited product. These conditions never hold: "Because humans are fallen, we cannot trust each other's motivations. Because humans are fallen, we cannot communicate perfectly." Hence "get it in writing" — written agreements for clarity, enforceable contracts for protection from others' misdeeds and our own temptations. Organizations often behave worse than any member would.
- It is the necessity for contracts that forces the too-early binding of goals, requirements, and constraints — and thus best explains the Waterfall's persistence. Contracts also open new wrongdoing: "Low-ball on the contract; make it up on the change orders."
- The centuries-old architectural contracting model separates the contract for design from the contract for construction:
  - The client develops a *program*, not a specification.
  - He contracts with an architect on an hourly or percentage basis for *services*, not a specified product.
  - The architect elicits a fuller program from client, users, stakeholders; does a conceptual design approximating the reconciliation of program with budget, schedule, and code — a first prototype, conceptually tested by stakeholders.
  - After iteration: design development, then construction drawings and specifications.
  - Only then does the client enter a fixed-price contract for the product.
- Projects with close client-architect-contractor trust, well-understood design challenges, or a pressing hurry justifying higher risk can conflate this into concurrent, pipelined design-build: produce first the detailed drawings the contractor needs first (long-lead steel, site work, foundations). System projects meeting those conditions should proceed likewise — the challenge is identifying the build order and long-lead components.

**Apply:**
- Never let a stakeholder committee fix detailed requirements before design begins; extract a few top-level requirements and develop the rest iteratively during design.
- Appoint an empowered advocate for the product itself — its integrity, economy, robustness — in every requirements process.
- Weight and prioritize requirements explicitly; trace every detailed requirement back to a top-level one and kill the ones that snuck in.
- Contract for design services separately from product construction; fix price only once the design exists.
- Preserve schedule urgency — it is the best defense against requirements creep.

## Ch 5 — What Are Better Design Process Models?

Since a dominant process model will exist regardless (communication and teaching demand one), the pressing need is to *replace* the Waterfall with a less misleading model, not merely augment it; Boehm's Spiral Model is the most promising, to be developed further with explicit contracting points.

- Cross & Dorst on what real creative design is: "not a matter of first fixing the problem and then searching for a satisfactory solution concept; instead... developing and refining together both the formulation of the problem and ideas for its solution, with constant iteration of analysis, synthesis and evaluation processes between the two 'spaces' — problem and solution."
- Criteria for a better model: emphasize progressive discovery and evolution of requirements; be memorably visualized so it can be readily taught and understood by team and stakeholders (designers are spatial thinkers — they learn, think, and talk in terms of a model with a clear geometric picture); still facilitate contracting among fallen humans.
- "Let 100 models bloom" fails: the ubiquity and damage of the Waterfall despite decades of criticism convinces Brooks a dominant model will exist; Simon's problem-solving model likewise occasioned much wasted effort in blind alleys.
- **Co-Evolution Model** (Maher, Poon, Boulanger): problem space and solution space co-evolve with interchange of information between them, both incrementally generated and incrementally evaluated. Emphasizes requirements discovery; memorable image. But: not comprehensive (no build-test-field-maintain-extend), the geometry doesn't suggest convergence, and it has no milestones or contracting points. Attractive but not sufficient.
- Over-rich models (Hickling's Whirligig, with cycles and epicycles) defy understanding, much less memorization and facile use.
- **Raymond's Bazaar Model** (Open Source): a member of the using/creating community sees a need, builds a module or bug fix for his own work, and gifts it; the incentive is prestige; a market mechanism among free goods selects the best modules and fixes; hundreds of simultaneous testers smoke out bugs sooner. Brooks's six observations on when it works:
  - It is an evolutionary model — the system grows by adding components each meeting a need discovered by a user-designer.
  - The gift-prestige economy works for people who are being otherwise fed — the gifts are by-products of revenue-producing work.
  - Hence more tools than applications; results need only be good enough for the builder's own purpose; "market" selection is the quality control.
  - Linux is hardly a random pile: "Linus Torvalds has been an overarching intellectual force for conceptual integrity," a functional specification already existed (UNIX), and an overall system design existed.
  - Its conspicuous success derives directly from the builders being the users: requirements, desiderata, criteria, and taste come unbidden from their own experience — "the whole requirements determination is implicit, hence skipped." Doubtful where builders know user needs only secondhand.
  - Cathedral processes — carefully architected, tightly controlled, meticulously tested — are still needed: "Would you use the Open Source process to build the new national air-traffic control system?"
- **Boehm's Spiral Model** (1988): successive repetitions of the same activities radiating outward; suggests progress; emphasizes prototyping — starting with user-interface prototypes and user testing long before an operational prototype is possible; easily understood and memorable; accepted even in DoD procurement; risk management is the focus of Boehm's later development of it.
- Denning & Dargan's critique — still designer- and product-centered rather than user- and action-centered — is answered: a development model is principally used by developers, so designer-centered is appropriate; with Boehm, advocate "frequent but not continuous interaction with representative users, with successive prototypes as the vehicles."
- The way forward: embrace and develop the Spiral — "punctuating the spiral with explicit contracting points, augmented with clear specification of what can be contracted, with what certainties, and with what explicit distribution of risk."
- Summary argument of Chapters 2–5: a formal design process model is needed (organizing work, communication, teaching); it must be visual/geometric; the Rational Model occurs naturally and was independently formalized (Simon, Pahl & Beitz, Royce); it is highly misleading — it doesn't reflect what real designers do; the bad model matters (too-early binding of requirements → bloated products and schedule/budget/performance disasters); it persists because of seductive logical simplicity and because builders and clients need contracts; the Spiral is the most promising replacement.

**Apply:**
- Run design as a spiral: repeated cycles of prototype, user test, risk assessment, and refinement — with contracting points placed on the spiral, each specifying what is committed, with what certainty, and who bears which risk.
- Let problem formulation and solution co-evolve; schedule iteration between the two spaces rather than freezing the problem first.
- Prototype the user interface and test with users long before an operational prototype exists.
- Reserve bazaar-style processes for cases where the builders are the users; use cathedral processes for high-assurance systems.

## Ch 6 — Collaboration in Design (Part II: Collaboration and Telecollaboration)

Team design is now the modern standard, for good reasons, but collaboration is not good per se: most great works of the human mind were made by one mind, or two working closely, and the grave danger of team design is loss of conceptual integrity — achieving it in a team is a management feat.

- Why design shifted from solo to teams:
  - Technological sophistication: no naive technologies are left (CFD to mix shampoo without tearing the emulsion; computed airflow in cotton pickers); explosive sophistication forces specialization — no one can follow all the subfields anymore — so state-of-the-art design needs masters of various crafts.
  - Hurry to market: first-to-market can expect roughly 40% long-run share and a profit bubble; team design becomes necessary when it accelerates delivery.
- Costs of collaboration — "Many hands make light work — often. But many hands make more work — always." No design task is perfectly partitionable, few highly so:
  - **Partitioning cost**: crisp interface definition is a lot of work, slighted at peril; interfaces need continual interpretation; gaps and inconsistencies must be reconciled; commonality and standardization must be established; then integration — "cut to plan; bang to fit."
  - **Learning/teaching cost**: each of n collaborators must come up to speed on goals, desiderata, constraints, utility function: work = n·l + d, not l + d; and someone with the vision must teach rather than design.
  - **Communication cost** during design, to be sure the pieces fit.
  - **Change control cost**: each designer makes only changes affecting his own part or negotiated with affected owners; substantial cost, but the cost of not having formal change control is much greater.
- Conceptual integrity is the challenge: much of what we consider elegance is the integrity and consistency of concepts; it delights and yields ease of learning and use — the tool does what one expects. Component principles: orthogonality, propriety, generality. The solo designer produces integrity subconsciously — he tends to make each microdecision the same way each time.
- Many great engineering designs are still principally one mind's work: Cray's genius flowed from total personal mastery over the whole design — architecture to circuits, packaging, cooling — and consequent freedom to trade across all domains; he repeatedly took his team away into solitude and built the CDC 6600 with 35 people "including the janitor." The recurring pattern for truly innovative products: physical isolation, small teams, intense concentration, leadership by one mind (Spitfire at Hursley House; Lockheed's Skunk Works; IBM's closed Boca Raton lab for the PC).
- Modern design as "interdisciplinary negotiation" among peers? "NO! If conceptual integrity is the final goal, negotiation among peers is the classic recipe for bloated products! The result is design by committee, where none dare say 'No' to another's suggestion."
- **A system architect**: the most important single way to ensure conceptual integrity in team design is to empower a single system architect — competent in the relevant technologies, experienced in the sort of system, with a clear vision, who really cares about conceptual integrity. The architect serves during the entire design process as agent, approver, and advocate for the user and all stakeholders. The real user is often not the purchaser; marketers represent the purchaser, engineers and manufacturers are represented — "only the architect represents the users."
- **One user-interface designer**: a major system needs an architecture team, so the integrity challenge recurses; the user interface, the user's crucial system component, must be tightly controlled by one mind — "if one architect can't master it, one user can't either" (at Google, one VP maintained personal control of the page format and home page). Iverson on why APL is easy to use: "It does what you expect it to do" — consistency, orthogonality, propriety, generality, parsimony.
- Cautionary tale — IBM's Future Series: brilliant, experienced architects, hours of grand vision, and to the request "let me talk to the architect who understands it all" the answer was "There isn't one." "I knew then that the project was doomed — the system would collapse of its own weight." The 800-page user manual confirmed it: how could any user master such an interface?
- Where plural minds genuinely add value:
  - **Determining needs and desiderata**: a small team beats an individual at studying an unmet need or an existing system — several minds ask different questions, pick up different unsaid things, observe different aspects of how users work today (videotape and re-watch the observations); the hard task is to flush out the implicit objectives and constraints stakeholders don't recognize they have.
  - **Conceptual exploration**: explore solutions early (as long as no one gets wedded to any), because concrete postulated solutions elicit hitherto unspoken desiderata and constraints. Brainstorm collectively — but note Dornburg's Sandia experiment: individuals produced at least as many ideas, and of significantly *higher quality* (originality, feasibility, effectiveness), than groups working together.
  - **Design competitions**: with concretely stated shared constraints and objectives, competitions harness multiple designers' creativity (Brunelleschi's dome, 1419). The S/360 competition — 13 one-to-three-person teamlets against fixed rules after a demoralizing cost estimate — was "immensely invigorating and fruitful": it re-energized everyone, involved each person deeply in all aspects, produced consensus, and produced a good design (the two best designs, from non-communicating teams, were surprisingly alike).
  - **Product fights** (unplanned competitions) follow a five-act script: teams meet and unanimously find "no real overlap"; reality appears (forecast or skeptical boss); each team expands its design to cover the *whole* of the other's market; each woos supporters; shootout before an executive. Usually best shortened by early action from a skeptical boss, though occasionally the best way to get a thorough, impassioned exploration of two approaches.
  - **Design review**: the phase where collaboration is most valuable, even necessary. Each disciplinary specialist must review the documents alone first (careful review takes time, reflection, references — documents, not view-graphs: PowerPoint's "vague bullets enable each participant to interpret the information as he pleases; they also facilitate the suppression of embarrassing but crucial details"). Then multidisciplinary group review with a review team *larger* than the design team: builders, maintainers, sample users, marketers, reliability, safety. One specialist's spoken concern triggers another's (the shipyard foreman's one-piece rolled cylinder replacing 20 engineered pieces; the oil-rig maintenance foreman: "we can paint it in the workshop... but where it goes, we'll never be able to paint it again" — the engineers redesigned the whole vicinity). Use rich graphical representations, mock-ups, or virtual environments so every reviewer can visualize the product.
- Dissent noted: participatory-design advocates argue users should co-design; feasible and prudent for buildings, but for mass-market products user participation is inherently limited to a small sample, conditioned by representativeness and the designer's vision. Others claim team design was always the norm.
- **The fantasy of collaborative design**: the CSCW literature imagines a team jointly seeing a model and directly effecting changes, the design forming bit by bit. "The fantasy model of collaborative design reflects a monumental unconcern about conceptual integrity. Jill pats the design here; Jim nudges it there; Jack patches it yonder. It is spontaneous; it is collaborative; and it produces poor designs... we have a scornful name for it — committee design." Tools that encourage committee design do more harm than good.
- How collaborators really design: each part of a design has at any time **one owner**, who works alone preparing a proposal, meets his collaborators for what is in effect a micro-session of design review, then retires to work out the detailed consequences. A rejected alternative's proposer withdraws and develops it; the session reconvenes to choose, fuse, or strike off in a third direction.
- Concurrent activity requires design control absent from solo design: monitoring that owners don't collide (Jack's air ducts vs. Jill's steam pipes), a conflict-resolution procedure, and version control against a single time-stamped baseline. The client admiral may move a bulkhead only in a playpen copy of the model — never the standard version, which embodies constraints and compromises he cannot know.
- "Conceptual design, especially, must not be collaborative": once exploration is past and a basic theme selected, conceptual integrity must rule — "a design flows from a chief designer, supported by a design team, not partitioned among one." If the theme hits a blind alley, return to collaborative exploration until a new basic scheme is selected.
- **Two-person teams are magical.** Even in conceptual design, pairs acting uno animo can be more fruitful than solo designers. Pair programming: initial productivity below two working separately, but error rates radically reduced — and since perhaps 40% of design effort is rework, net productivity is higher and products more robust. Pair dynamics differ from both solo and multi-person work: rapid informal interchange, short bursts of holding the floor, a single thread of idea development, "two pencils may move over the same paper with neither collision nor contradiction." Articulating one's thinking — why as well as what — brings quicker perception of one's own fallacies. (Torrance 1970: dyads produced twice as many original ideas, of twice the originality.) Pair sessions still need interspersed solo ones — to detail, document, and prepare proposals.
- Closing cautions for toolsmiths: real design is always more complex than imagined and always explodes into countless details; real team design always requires change control; and "no amount of collaboration eliminates the need for the 'dreariness of labor and the loneliness of thought.'"

**Apply:**
- Empower a single system architect as user-advocate and keeper of conceptual integrity; never design by negotiation among peers.
- Put the user interface under the control of one mind.
- Collaborate hard where many minds pay: requirements elicitation, early exploration, competitions, and above all multidisciplinary design review (review team larger than design team; documents studied solo before the meeting).
- Give every part of the design exactly one owner at any time; run joint sessions as micro design reviews of an owner's proposal.
- Use pairs for high-stakes conceptual work; institute formal change control from the moment concurrent design begins.

## Ch 7 — Telecollaboration

Distributed design will only increase, but space, time-zone, and cultural barriers are real; telecollaboration works only when built on face-to-face history, meticulously clean interfaces, and mostly low-tech tools.

- Why teams telecollaborate: super-specialized skills aren't available in every city; specialists live where they please and work elsewhere; Earth's rotation advances work around the clock on day shifts; cost disparities make skills radically cheaper via outsourcing; international politics partitions work among nations (Airbus 380 development split across France, Germany, Britain, Spain).
- Airbus's working recipe: full telecommunications, resident "ambassador" engineers on the remote site, and a daily company plane carrying live people both ways — "none of these collaboration aids could be omitted." The A380 also shows the pitfall: French/British teams on CATIA Release 5, German/Spanish on Release 4 — wiring harnesses needed larger radii than the other team's conduits provided; ~22 months of painful delays.
- System/360 precedent (1961–65): seven computers concurrently developed in four labs across three countries; 40+ new I/O devices in seven more labs. Coordination radically aided by a technical innovation — the meticulous definition of a standard logical, electrical, and mechanical interface for attaching any I/O device to any computer (which needed its own small architecture team). Management: IBM's first full-time transatlantic phone line, resident participants exchanged between labs, thousands of calls and documents, many pair-wise face meetings, and annual two-week whole-team meetings settling hanging conflicts — some 200 at one session.
- The verdict: highly successful, but "distributed development of a unified product is work! Moreover, the distribution per se creates a lot of extra work! We sorely underestimated the immense importance of the informal communication channels at work within collocated teams, until we experienced their absence. Space barriers are real! Time-zone barriers are real, sometimes more so... And cultural barriers are very real."
- **Face-to-face time is crucial.** The most successful telecollaborations are built on extensive face-time histories, and even these require some face time during the collaboration; "absent such histories, travel is worth what it costs in money and time." Among the most fruitful dollars Brooks spent at IBM: a bus taking the S/360 administrative staff to lunch with their headquarters counterparts — "familiar voices hitherto faceless"; the lubrication beat more pressure for cooperation. Boeing collocated the 777's distributed teams for weeks as design started. People instinctively know this — despite videoconferencing, planes still carry business travelers.
- **Clean interfaces** among remotely designed components: defining them is hard; the job doesn't end at definition — continual question-and-answer interpretation of the semantics proves necessary; changes must be made, controlled, and widely communicated. Management must also design in advance a mechanism for resolving differences of opinion or taste — "there is no substitute for authority."
- Payoff of clean interfaces: errors and rework, though touching a small fraction of a design, may account for half its cost; vague or sloppy interface errors surface late, during integration — nastier to find, costlier to fix, impacting the whole schedule. Clean interfaces also enhance the joy of the work: each designer gets ownership, "the privilege of signing a piece of work."
- Low tech often suffices — in order of potency:
  - **The document**: the most potent telecollaboration technology; formal prose and drawings "carry the precision that demands study, enables critique, stimulates interaction." Brooks & Blaauw wrote their 1,200-page book largely by mailed drafts — but atop nine years of daily face time, deep knowledge of each other's style and manners, and shared convictions; even so, quarterly phone meetings and semiannual three-day face meetings were needed. Those focused on the uncracked nuts: "when a text paragraph couldn't be made to work, it was always because we didn't know what we were talking about."
  - **The telephone**: an even bigger breakthrough than email; email is extemporaneous writing with no vocal inflection and no instant give-and-take; instant messaging is a poor substitute for telephony.
  - **Telephone-plus-shared-document**: vastly more powerful than either alone — real-time interaction heads off misunderstanding, and having to agree word-by-word forces the collaborative facing of issues that would otherwise be missed. (The staff engineer 150 feet from a high-bandwidth videoconference room preferred phone plus shared drawing at his desk.)
- Videoconferencing: adopted far more slowly and less extensively than predicted. Remaining gaps: field of view (committee-to-committee meetings), simultaneous viewing of speaker and document, spreading materials on a table, symmetric shared whiteboards, private plus shared markings, resolution (can't share a full page or read faces well), depth cues. Most valuable where facial expressions and body language matter: screening stranger job applicants, vital issues, an insecure participant, different organizational or national cultures.
- Tools pushed, not pulled: most telecollaboration tools spring from a technical idea rather than analysis of a collaboration pattern or need (in a web search, 49 of the first 50 entries were on tools, not on collaboration in design). "Effective toolsmithing always starts with the user and the task" — best done when the toolsmith has a real user with a real task that must be done; then buggy prototypes will not satisfy, and critical feedback is immediate and blunt.

**Apply:**
- Budget real travel and collocated time, especially at project start and for standing conflicts; do not pretend tools replace face time.
- Invest heavily in meticulously defined component interfaces, with a Q&A channel for semantics, change control, and a pre-agreed authority for settling differences.
- Keep the whole distributed team on identical tool versions.
- Default to shared documents and phone-plus-document; reserve video for situations where faces matter.

## Ch 8 — Rationalism versus Empiricism in Design (Part III: Design Perspectives)

Can I, by sufficient thought alone, design a complex object correctly? Rationalists say yes; empiricists say no. Brooks is a dyed-in-the-wool empiricist: testing and iteration are in practice necessary — but careful thought helps.

- The crux goes deeper than method — it is one's view of the nature of man as creator:
  - The rationalist: man is inherently sound, subject to mistakes, perfectible by education; after right education, maturing experience, and careful-enough thought, a designer can make a flawless design. The methodology task: learn to reason a design into flawlessness.
  - The empiricist: man is inherently flawed, subject repeatedly to temptation and error; anything he makes will be flawed. The methodology task: learn to determine the flaws by experiment, so one can iterate on the design.
- Lineage: Aristotle deducing that heavier objects fall faster vs. Galileo's experiments; Descartes vs. Locke; French science's beautiful logical structures vs. the British empirical tradition.
- In software: Dijkstra's rationalist position — a program is a mathematical object; design it correct and prove it, and that will suffice. Brooks grants that a program is pure thought-stuff and in principle can be designed perfectly: "The difficulty is not with the design medium but with designers." Humans err in defining objectives, in architecture, in algorithms and data structures, in code.
- Firm faith in fallibility prescribes the methodology: design, early prototypes, early user testing, iterative incremental implementation, testing on a rich bank of test cases, regression testing after changes.
- Brooks's own existence proof and its limit: only twice did a program of his run correctly the first time; the 1953 case followed meticulous, ad-nauseam desk-checking under extreme motivation (two one-hour machine shots per semester). "Yes, in principle it is possible. With real people and real-scale contemporary software, it is not sustainable."
- Formal proof of correctness: exactly proper for secure OS kernels, where high assurance is needed and, if the kernel is right, damage from error or malice elsewhere can be contained. Caveats: proofs themselves have historically been found fallacious — proof's real advantage is that its reasoning differs in form from design reasoning, so the odds are radically improved that the same mistake will not slip past both scrutinies; proof effort is on the order of the work of building the program; and "no proof can show that the original objectives for the program were right" (Lufthansa 2904 at Warsaw: the code followed the specification, which was wrong for the unexpected circumstances).
- The practical middle way — Mills's cleanroom: expose every aspect of a design to intense group scrutiny; the designer explains to the assembled group why the design is correct while they challenge his arguments and their implicit assumptions. "Formal proof of correctness is usually infeasible; abandoning all effort at systematic verification (the more common extreme) is dangerous: Mills's systematic but non-formal group scrutiny of logical argument seems to me a wise and practical balance."
- Other design domains don't attempt correctness proofs (one cannot prove theorems about materials and their faults, or spaces and their suitability) but use extensive empirical design verification: stress, vibration, and acoustic analyses; walk-throughs running use scenarios on designed buildings; snow/hurricane/earthquake analyses; hardware simulated at circuit, logic, and program-execution levels; operating systems executed dead-slow on simulators of unbuilt computers.
- Consequence of rich analysis: more iteration and more certain verification of the design against the goals specified — "but none of these analyses and simulations addresses the rightness of the goals or the validity of the assumptions about the environment."

**Apply:**
- Assume every design you make is flawed; build the process (prototypes, user tests, incremental implementation, rich test banks, regression tests) to find the flaws early.
- Reserve formal verification for small, critical kernels; elsewhere use structured group scrutiny of the designer's correctness argument.
- Remember that no amount of verification validates the objectives or the environmental assumptions — test those against reality separately.

## Ch 9 — User Models — Better Wrong than Vague

Every design decision is guided by assumptions about users and uses, consciously or not; write them down explicitly — and where facts run out, guess — because "an articulated guess beats an unspoken assumption," and wrong assumptions will be questioned while vague ones won't.

- Experienced designers begin by writing down exactly what they *know* about the user, the purposes of use, and the modes of use; wise designers also write down what they *don't know but assume*. With multiple applications or user sets, describe each and define explicit weightings among them (a use model is a weighted collection of use cases).
- The more detailed and particularized the assumptions, the more occasion for early detailed thinking — thinking that would have been required later anyway; doing it early forfends mistakes.
- Very few designers actually do this; the need follows from modern design's peculiar characteristics — team design and complex tools:
  - **Team design** creates the all-new requirement that the *entire team* share the same user and use model. Members silently believe they share assumptions (each heard the leader's charge; each read the goals document; all are expert), but each person's different experience with similar systems yields a distinct implicit model. "Microdecisions too minor ever to be discussed will be made differently, and conceptual integrity will be lost."
  - OS/360 reflects two quite inconsistent debugging philosophies — one assuming batch use, one assuming time-shared terminals — never a conscious decision, merely subgroups holding differing use models. "The result was bloat and incoherence."
  - **Complexity**: even a shovel needs explicitness (coal, dirt, grain, or snow? child, woman, or man? casual user or laborer?) — how much more a truck, a spreadsheet, an academic building. And the more complex the design, the less likely the designers are domain experts who could do the users' jobs, making implicit models much more dangerous.
- Making models explicit rudely confronts the designer with how much he doesn't know — "an unmitigated good," forcing early the questions he might not otherwise ask until much later. Field facts (from representative users) just drive harder questions: representativeness, ranges, distributions, rates of change, the situation five or ten years out.
- When reasonable inquiry runs out: **guess** — postulate a complete set of attributes and values, with guessed frequency distributions, to develop complete, explicit, shared user and use models. Benefits:
  - Guessing forces very careful thought about the expected user set.
  - Writing values down exposes them to debate — "it is easier to criticize something concrete than to create," so the whole team contributes, differences in the designers' user images surface, and other unrecognized assumptions surface too.
  - Enumeration helps everyone see which decisions depend on which user-set properties.
  - It enables sensitivity analysis: which assumptions matter, and how much? Where important decisions hinge on a particular guess, it is worth the cost to develop a better estimate.
- In the end many assumptions remain debatable and unverifiable; "the chief architect must own — and make known — the set the team goes with."
- The alternative is worse: the vague designer substitutes himself for the user, "designing for what he assumes he would want if he were the user. But he isn't." Therefore: wrong explicit assumptions are much better than vague ones — "wrong ones will perhaps be questioned; vague ones won't."

**Apply:**
- Before designing, write an explicit user and use model: who the users are, their purposes, modes of use, and the weighting across user classes — including stated assumptions where facts are missing.
- Where data ends, guess concrete values and frequency distributions rather than staying vague; publish the guesses for attack.
- Have the chief architect own the assumption set and make it known to the whole team.
- Run sensitivity checks: identify which decisions hinge on which assumptions, and buy better estimates only where it matters.

## Ch 10 — Inches, Ounces, Bits, Dollars — The Budgeted Resource

Every design has at least one scarce resource to be rationed; for conceptual integrity — especially in team design — "name the scarce resource explicitly, track it publicly, control it firmly."

- Designers often talk as if cost or performance/cost were what they optimize; that is often not how they act. Usually one limiting resource dominates and others appear as desiderata or constraints.
- The budgeted resource is often not dollars: inches of oceanfront in a beach house; ounces of payload in a spacecraft or backpack; memory bandwidth in any von Neumann architecture; nanoseconds of timing tolerance in GPS; calendar days on an asteroid-interception project; resident kernel memory in OS/360; program hours at a conference; pages in a proposal; power and stored energy on a satellite; heat in a high-performance chip; water on western farmland; student learning hours in a curriculum; political power in an organization's constitution; seconds in a film; track-access hours in Underground maintenance; format bits in a computer architecture.
- Even dollars have flavors: manufacturing cost dominates mass-market PCs; development cost dominates supercomputers made by the dozens.
- Surrogates for dollars (square feet for buildings; register/cache bits for chip area) have advantages: simpler; usable before the surrogate/dollar ratio is known; more stable; they harness previous design experience. But they lead astray when used after they cease being appropriate — chip designers thought in area well after wiring length or pin count became the critical resource.
- The budgeted resource changes with technology (chip area → I/O pins → power dissipation; Cray: "Refrigeration is the key to supercomputer design") — and mid-project, just because we get smarter:
  - OS/360's team was sure memory space was the constraint (12K resident kernel in the tightest case), so everyone read small chunks from disk with great frequency. Ruthrauff built a performance simulator *early*; the first results were horrifying — five Fortran statements compiled per minute on the second-fastest model. "That day, the project's budgeted resource switched from memory bytes to disk accesses." (Counterpoint: Digitek's compiler used a dense interpreted representation — losing decode time but avoiding the true constraint entirely, winning tenfold.)
- Corollary actions:
  - **Identify explicitly**, right after enumerating objectives and constraints. It is usually a resource of the *design*, not the design process (skill allocation is crucial to the project but is not a property of the design; schedule becomes the budgeted resource only when it is intrinsic — the asteroid interception, the race to be first to market).
  - **Track publicly**: the whole team must continually know the current budget; each sub-team and member must know how many milliwatts or disk accesses their part may use.
  - **Control firmly**: exactly one person controls budgeting and rebudgeting, keeping a small personal kitty for late allocation, "just as a general keeps some reserves for dispatch to the hottest part of the battle."
- The exemplary budget-keepers share three traits — total system overview, cautious stinginess, and inventiveness of alternatives within the existing design: Blaauw with Program Status Word bits in S/360; Iverson, who desired conceptual integrity above all and so made *the number of distinct language concepts* APL's budgeted resource; Mayer as "the dragon guarding Google's look and feel."
- Note the generalization: the budgeted resource need not be physical — making concept-count the rationed commodity is a direct mechanism for conceptual integrity.

**Apply:**
- At project start, name the scarce resource explicitly (it is often not money); publish each sub-team's allocation and keep the running budget continuously visible.
- Put budget control in exactly one person's hands, with a held-back reserve for late needs.
- Verify the assumed budget with early measurement (build the performance simulator first); be ready to switch the budgeted resource when evidence or technology shifts.
- Watch for stale surrogates — periodically re-ask what actually limits the design.

## Ch 11 — Constraints Are Friends

Constraints shrink the designer's search space, focusing and speeding design, and often stimulate completely fresh creation; consequently "a general-purpose product is harder to design well than a special-purpose one."

- "Form is liberating." Unconstrained assignments ("write an essay on whatever you want") make design harder, not easier. Examples: Michelangelo's David, carved from a cracked block abandoned as unusable — the defects stimulated a different artistic concept; Wren's 50 London churches, each constrained by site, environment, old foundations, and the eastward-altar rule — 27 survive, each a different invented solution; the Blue Ridge Parkway viaduct, elegant because it had to touch ground as little as possible; Bach, who "preferred to work within a given framework and accept the challenges it imposed."
- Artificial constraints have the nice property that one is free to relax them; ideally they push one into an unexplored corner of the design space, stimulating creativity. But any constraint set may push into an empty corner where no conceivable design works. Therefore carefully distinguish:
  - Real constraints.
  - **Obsolete once-real constraints**: the experienced designer, "like a lion accustomed to pace the confines of its cage," obeys by habit constraints made obsolete by technological advances (squeezing software into cramped memory long after memory was cheap).
  - **Constraints misperceived as real**: the nine-dots puzzle (the line may leave the imagined square); Strassen's seven-multiplication matrix product (discard the assumption that operations must be on vectors).
  - Intentional artificial constraints.
- The FAA 9020 story: MITRE's architects, shopping for function and reliability, mistakenly specified a redundancy *topology* as an essential constraint. IBM's first proposal — standard Model 50s whose I/O system reused the same hardware — more than met all performance and reliability requirements with half the equipment, but was rejected for not matching the topology. The delivered system was, by unchallenged analysis, considerably *less* reliable (twice the components, more connectors), but it met the specified constraints.
- The moral: "When you specify something to be designed, tell what properties you need, not how they are to be achieved." If implementation approaches are given as constraints, better solutions are cut off; "the designer confronted with false constraints should fight back!"
- **The design paradox**: since the hardest part of designing is deciding what to design, and constraints pre-narrow the space, the more specialized the purpose, the easier the design task:
  - In one sense the general task is "easier" — with no constraints there are no criteria for excellence, so it is easier to do a *mediocre* general-purpose design than a mediocre special-purpose one.
  - But for an *excellent* design the special-purpose job is easier: the first task of any design process is narrowing the design space, and the constrained goal has already accomplished much of it.
  - A good designer can sketch a general-purpose computer architecture in days — the decision list is known (formats, addressing, datatypes, operations, sequencing, supervision, I/O). But an excellent general-purpose architecture demands a user model much harder to craft: study *each* of a whole set of applications, then weight them across the entire application set, the entire set of intended implementations, and the decades of lifetime a new architecture must contemplate (S/360 was predicted to live 25 years; it passed 45 with no end in sight).
  - Same paradox in software: a special-purpose language is straightforward; a general-purpose one demands delicate balancing of expressive power, generality, and parsimony — "restraint is so much easier to practice in the special-purpose design."
  - And in buildings: a superb bedroom is easier than superb public living spaces — more functions, more scenarios, more furnishing options.
  - Teaching corollary: students assigned special-purpose architecture projects "cannot offer gas and platitudes; the application and user analysis must be precise" — and they often do excellent work impossible for them on an unconstrained task.
- Net: "if the task originally seems unconstrained, first think harder about what is really desired, about the user and use models, and you will probably find some narrowing constraints, to the benefit of both designer and user."

**Apply:**
- Welcome constraints; use them to narrow the space before designing, and hunt for narrowing constraints whenever a task seems unconstrained.
- Audit the constraint list: which are real, which obsolete, which misperceived, which self-imposed and relaxable?
- Specify what properties you need, never how to achieve them; push back when handed implementation as a constraint.
- Prefer the specialized design task; treat "general-purpose" as demanding deep multi-application user modeling, not as a way to skip it.

## Ch 12 — Esthetics and Style in Technical Design

Delight is coequal with firmness and usefulness (Vitruvius: *firmitas, utilitas, venustas*); logical beauty in technical artifacts rests on parsimony, structural clarity, metaphor, and above all consistency — and style is the bundle of microdecisions, consistently made, that gives work recognizability and comprehensibility.

- Elegance in programs, languages, and computers is not visual: a "clean" computer refers to a property of its logical structure; delight may be purely intellectual, as with a beautiful proof.
- **Parsimony** — "accomplishing a great deal with few elements" — matters (Lisp's tiny core with elegant extensibility and composability, vs. Visual Basic's complexity and poor extensibility), but is not the whole story:
  - Adding an "unnecessary" component like index registers can radically improve performance and performance/cost.
  - Van der Poel's one-operation computer was provably sufficient yet very difficult to program; effective use required a library of nonobvious idioms — the delight of a crossword puzzle, "a construct of intentional complexity and no intended utility," not elegant design.
  - APL one-liner sport is the same: "programming languages exist to facilitate the writing — and the much more frequent reading — of programs, not to serve as puzzles."
- **Structural clarity**: a direct route from what one wants to say to how one says it; the basic structural concept of the design plainly evident and, if not logically straightforward, easily explained. (Natural language, meeting real needs, is far from parsimonious — English is ~50% redundant.)
- **Metaphor**: familiar, simple metaphors aid both elegance and comprehensibility, especially in user interfaces (the Macintosh Desktop; VisiCalc's spreadsheet).
- **Consistency underlies all principles of quality**: "a good architecture is consistent in the sense that, given a partial knowledge of the system, one can predict the remainder." The mere decision to include a square-root operation should almost fully define it — formats, precision, range, rounding, exceptions all as elsewhere. Touchstones for the truly consistent solution: brevity of description, simplicity of code generation, suitability for many implementations.
- Three principles derive from consistency:
  - **Orthogonality — do not link what is independent**: a change in one orthogonal function has no observable effect on any other (violated if the alarm works only when the clock face is lighted).
  - **Propriety — do not introduce what is immaterial**: functions must meet essential requirements; the opposite is extraneousness (the gearshift is extraneous to driving — implementation leaking into the user interface; twos-complement's unique zero vs. the extraneous signed zero of other notations, with its unexpected rule-cascades). Parsimony is a subset of propriety; so is transparency — implementation producing no visible side effects (pipelining invisible to the programmer).
  - **Generality — do not restrict what is inherent**: the ability to use a function for many ends. "It expresses the professional humility of the designer, his conviction that users will be inventive beyond his imagination and that needs may change beyond his ability to forecast... When you don't know, grant freedom." (Intel 8080A's Restart, general enough that its most frequent use became subroutine return.) Achieved via open-endedness, completeness of function sets, decomposition into orthogonal components, composability.
- Consistency is reinforcing and self-teaching — it confirms expectations — and it reconciles ease of learning (wants simplicity) with ease of use (wants richness): make the simple architecture a subset of the complex one (fixed-point a subset of floating-point) so comprehension grows naturally.
- **Style**: recognizability of a maker from the work (a computer "looks as if" Cray designed it; the Federalist authors conclusively identified by details of prose style; programmers identify each other's code; WWII radio operators recognized each broadcaster's Morse "fist").
- Working definition: "Style is a set of different repeated microdecisions, each made the same way whenever it arises, even though the context may be different," with related microdecisions resolved in related ways. Hypothesized mechanism: habit as the human economization of mental effort — absent substantial reasons, we make the same microdecision the same way every time. Style has more to do with details of design than with the main purpose or thrust.
- Clarity of style: consistency achieved across a wide range of macro- and microdecisions, describable economically; recognition follows. "A clear style may not be a good style; a muddled one never is." "Somehow, consistency brings clarity, and clarity brings delight." A mishmash of styles delights no one.
- Style consistently applied is a component — even if only the "dress" — of conceptual integrity; beyond delight, it aids comprehensibility, which begets ease of initial learning, ease of use, ease of recollection after disuse, ease of maintenance, ease of extension.
- Properties of styles: specification is remarkably costly (Fowler spends ~2,800 words defining proper use of "the"; the Chicago Manual runs near a thousand pages); specification is inherently hierarchical (dialect/diction → tone → prosody → usage → punctuation → layout); styles evolve over time, even an individual's (early vs. late Turner; Early/Decorated/Perpendicular Gothic).
- **To get a consistent style — document it.** A solo ten-page paper keeps style naturally; a book, or any multi-author work, needs a growing style sheet (the 1,200-page *Computer Architecture* grew a 19-page writing style sheet beyond the Chicago Manual and the publisher's house style). Beyond given standards (SAE catalogs, subroutine libraries), each particular design occasions lots of unprescribed microdecisions.
- Teams must also document the design's intent — "say why, capturing the designers' intent so that the later maintainer will not in ignorance loose a vital stone from the edifice's arch" — and internally document the myriad microdecisions composing the visible design, to preserve conceptual integrity during maintenance.
- How to achieve a good style: "the prescription is simple; the methods, straightforward; the work, arduous" — study other designers' styles intentionally and practice working in them (forces close attention to detail; can even produce great works — Respighi, Kreisler); write opinions about what styles you like and why; practice, practice, practice; revise, looking for stylistic inconsistencies; and for your products, choose designers with clear styles and good taste demonstrated in previous works.

**Apply:**
- Design for consistency above all: partial knowledge of your system should predict the rest.
- Enforce orthogonality (don't link the independent), propriety (don't add the immaterial), and generality (don't restrict the inherent; when you don't know, grant freedom).
- Keep a living style sheet of microdecisions for any multi-person or long-running work; record the why behind decisions for maintainers.
- Prefer readability and directness over cleverness; treat idiomatic compression as puzzle-sport, not elegance.
- Study and imitate other designers' styles deliberately as training; revise your work hunting stylistic inconsistencies.

## Ch 13 — Exemplars in Design

Few designs are all-new; exemplars provide safe models for new designs, implicit checklists of design tasks, warnings of potential mistakes, and launching pads for radical designs — and mastering the corpus of one's craft is a mark of great designers, not a crutch.

- "The vast field of possibility can only be searched if you have some idea in advance of what you are looking for. Without prestructures of some kind, you cannot know where to look, or whether you have found what you are looking for" (Hillier & Penn).
- Great designers invested great effort in precedents: Palladio measured and documented Rome's surviving monuments — from that tedious, unsung labor sprang his own designs and a book that fathered an enduring style; Jefferson studied Palladio's books and the buildings of Paris; Bach took unpaid leave and walked 250 miles to study Buxtehude for six months (losing his job for overstaying) — "his surpassing excellence came from comprehending and using the techniques of his predecessors, not ignoring them."
- Amateurs vs. professionals: the amateur uses whatever exemplars he happens to have encountered; the trained professional has been exposed to whole libraries spanning eras, styles, and schools, ideally with expert-guided tours highlighting noteworthy characteristics. Much computer and software design suggests personal-experience-only exemplars — even trained professionals aren't studying what's available.
- Evidence in computers: architectures reveal the machine on which the architect first had substantial programming experience (early DEC minis flavored by Whirlwind; S/360 by the 704 and 1401; early micros by the PDP-11), plus corporate memory — architects know their company's predecessors far better than competitors'.
- Software: mass products evolve by generation like computers; custom applications and operating systems historically reflect chiefly their designers' own experience. Patterns (Gamma at component level; Buschmann at system-structure level) began cross-fertilization; "we need more descriptions of whole systems, explaining system concepts."
- **Study rationales, not just artifacts**: manuals give the whats; for the whys one needs the technical papers — which mostly skimp on whys. Designs' own creators rarely explicate them (too busy on the next design), and published rationales, "like reports of military victories, are always after the fact, and they are usually rationalized... far more rational in retrospect than was the actual design process. For most of us, that process was rich with potholes, blind alleys, mistaken turns, and alterations of goals."
- The good rationales cluster around technology births and revolutions, when approaches vary widely and debates are hot. In computer architecture: Burks–Goldstine–von Neumann 1946 ("the most important computer paper ever written... The coverage is complete; the reasoning, compelling"); EDSAC and Manchester; Stretch and CDC 6600; virtual memory (Atlas); the minicomputer revolution; the microcomputer and RISC revolutions.
- Revolution pattern worth noting: at each turn the comfortable incumbents — "fat, dumb, and happy" — missed the revolution (mainframe makers missed minis; mini makers missed micros; DEC did not survive).
- A discipline's maturation ladder for exemplars: **collection** (Bell & Newell; Hennessy & Patterson; Blaauw & Brooks's "computer zoo") → **criticism** of particular exemplars → **comparative analysis** (assess how well the designing met each design's *own* objectives — criticizing goal selection doesn't help future designers) → **Rules of Good Practice** derived from examples → handbooks and ultimately standards. Computer design has progressed far along this path; software design is way behind — describing an OS architecture is perhaps two orders of magnitude more complex than describing a computer.
- Systematizing exemplars is scholarship, not design; scholars and designers differ in taste and temperament, and engineering academia undervalues the systematizer's work.
- On laziness, originality, and pride — emphatically not slavish copying, but:
  - "The designer should know well the exemplars of his craft, their strengths, their weaknesses. Originality is no excuse for ignorance."
  - "In engineering, if not in the arts, gratuitous innovation (that is, not anticipated to be 'better' in some useful sense) is a foolish idea and a selfish indulgence of pride — because of the unavoidable risk of unintended downside consequences."
  - "Designers who master the styles of their predecessors have more treasures upon which their originality can draw."
  - Those who just copy draw only on the current and fashionable (lazy Bauhaus; mediocre prairie-style ranches); mastering the corpus takes enthusiasm and diligence, not laziness.
- On novelty: delight lies in the superior elegance of a new solution to an old problem, not novelty per se — the delight of a Leatherman or a cable-stayed bridge does not fade with use, whereas "mere novelty is a cheat for satisfaction. The seven-day wonder grows old." "He who seeks originality is apt to find novelty, but not permanence of delight. On the other hand, he who seeks to make designs that really work is most apt to come up with new designs of enduring value, almost as a by-product."
- Pride — the desire to make a name for oneself — "infects all design, and ruins much" (Babel; Ozymandias; Sitterson Hall's "original" roof spoiling a possible quadrangle).

**Apply:**
- Before designing, deliberately study the exemplars of the domain — including documented rationales and especially failures; know their strengths and weaknesses.
- Distrust published design rationales as smoothed retrospectives; seek accounts of the blind alleys and mistaken turns.
- Innovate only where you anticipate real improvement; adopt proven solutions elsewhere to avoid unintended downside risk.
- Pursue designs that really work; let originality arrive as a by-product.

## Ch 14 — How Expert Designers Go Wrong

"The besetting mistake of expert designers is not designing the thing wrong, but designing the wrong thing." Amateurs make many little mistakes; professionals, when they goof, do it in a big way — and success breeds the overconfidence that causes it.

- Professional-scale failures: bridges that collapse during construction, houses with no stairs between stories, computers that radically waste memory bandwidth, programming languages too rich to be learned.
- Petroski's cycle after each revolution in materials or technique: designers tread cautiously at first → master the new approach → extend it boldly, often forgetting the underlying assumptions → overreach in boldness and self-confidence, pressed by hubris and competitiveness. (A documented ~30-year rhythm between major bridge collapses; Petroski predicted another was due — the I-35W Minneapolis collapse proved him right.)
- A major cause: a new generation trained in the technique from the start, never having suffered the birth pangs whose controversy probed assumptions — hence much less conscious of the assumptions and caveats, and unconscious of how the technique fits into the whole armory. The professional "is apt to be familiar with the trees, doesn't see the woods, and is slow to ask, 'Does what I am doing make sense in the large?'" — so preoccupied with doing the thing right that he fails to ask "Am I doing the right thing?" (S/360's superb, seasoned architects rejected automatic memory management — remedied "almost before the paint was dry.")
- "Success is dangerous for the professional designer. Failure stimulates analysis, scrutiny, rethinking. Success stimulates confidence both in design technique and in oneself. Both trusts may be misplaced."
- Case study — OS/360 Job Control Language, "the worst computer programming language ever devised by anybody, anywhere," developed under Brooks's supervision ("there is blame enough to go around among all the supervisory levels"), still in use in essentially the same form ~45 years later. The flaws:
  - The biggest: JCL *is* a programming language — interpreted and executed at schedule time — "but it was not perceived as such by its designers."
  - One schedule-time language imposed across all programming languages, so every user had to know two languages; what was wanted was a schedule-time *capability* within each programming language (like compile-time facilities), letting each programmer work in a single language, specifying actions for compile, schedule, and run time.
  - Modeled on assembler syntax — chosen just as assembler jobs fell to ~1% of all jobs: "a major paradigm shift had happened, and it wasn't recognized." (And with enough deviations that knowing assembler didn't mean knowing JCL.)
  - Card-column-dependent, built around the punched card "just as it galloped into obsolescence" — while the very same system pushed terminal access.
  - Proudly only six verbs, while the needed functions far exceeded six: "With an imposed 'elegant' simplicity not up to the actual complexity inherent in the task at hand, the complexity inevitably breaks out in jury-rigged solutions" — declaration parameters doing verbish things (DISP commanding dataset disposition).
  - Almost no branching (an afterthought via a parameter), no iteration, no clean subroutine call — the designers never imagined such actions in a schedule-time script.
  - Consequence for users: JCL so hard that working scripts were copied blindly; programs archived as "dusty decks" with their attendant JCL.
- How experts got there: "The professionals who designed JCL brought too much experience to the task. Their familiarity with what they thought to be the problem blocked their thinking about it afresh, in its wider setting. In this case, following an exemplar brought disaster." The key thinkers came fresh from the far simpler 1410/7010 batch OS and framed the task as "a few control cards for the scheduler" — every word wrong: *few* → too few verbs; *cards* → an obsolescing medium baked into the concept; *control cards* → separately interpreted, near-independent actions, hence no control flow. "Stating it wrongly led to wrong thinking throughout the design... Our basic problem was a pedestrian vision."
- Because it was never recognized as a language, JCL was never *designed* — it just grew; every new schedule-time function got loaded on as another keyword parameter. Had it been recognized as a system language, the team's expert language designers could have designed it as one.
- Lessons learned (near-verbatim):
  1. Study failure examples even more carefully than you study successes.
  2. Watch yourself after success — it stimulates confidence in the design technique, in the design, and in oneself; all may lead to overconfidence.
  3. Think at the top level about the object you are designing and its assumptions about the environment in which it will be used. Is a paradigm shift under way? Will your assumptions still be valid a decade hence? Are you designing the right thing?

**Apply:**
- Before detailing, interrogate the framing itself: what is this thing really, and is a paradigm shift invalidating your assumptions? Ask "Am I designing the right thing?" not just "Am I designing it right?"
- Study failures more carefully than successes; after a success, deliberately re-examine the assumptions your confidence rests on.
- Beware imposed simplicity below the task's inherent complexity — it will break out as jury-rigged complexity elsewhere.
- Beware your most relevant experience: the previous project is the most seductive false exemplar.
- If an artifact is really a language (or a system), recognize it and design it as one, with the appropriate experts.

## Ch 15 — The Divorce of Design

The 20th century progressively divorced designers from both implementation (they can no longer build what they design) and use (they bring little personal user experience); both links narrow radically in bandwidth — "communication between people is always much poorer than communication within a person" — and the divorces must be recognized and deliberately, effortfully mitigated.

- Then vs. now: Edison fabricated all his inventions in his lab; Ford made his own car; the Wrights built their airplane with their own hands. Today no computer engineer makes his own chips; no airplane designer masters the manufacturing processes or the stabilizing software; architects of hospitals or crematoria bring little user experience and must elicit behavior from representatives or — worse — surrogates removed a step or two from real users; few naval architects have commanded a ship. A generation ago, senior engineers had taken cars apart under the shade tree, held ham licenses, or came through sandwich/co-op programs interleaving hands-on industrial work with study.
- Why: (1) stunning advances in implementation technologies demand full-time specialization and protracted learning; (2) designed objects are now so complex that design alone demands specialization and all the designer's energies — "there are now few unsophisticated technologies" (even the Twinkie's manufacturing is complex).
- Fallout: miscommunication abounds — elegant buildings hard to work in; reactor control panels operators find confusing; over-specified implementations costing far more than needed with little added function.
- Happy exceptions prove the diagnosis: software engineering is young enough that system architects were once programmers; designers of personal products (iPod, iPhone, cars) are first of all users, their own use-vision illuminating the design; UNIX and Linux designers start with their own needs and build tools for their own use — accounting for both the use success and the user passion.
- **Remedy 1 — use-scenario experience.** Even a small amount is better than none; even a good simulation is better than none (full-scale mock-ups for kitchen or cockpit dry runs; virtual environments).
  - Brooks apprenticed two weeks as a computer operator before designing the Stretch console — "immensely informative"; it led to the first program-controlled operator console. (Honest coda: the overly fancy console was rarely used as envisioned; the experience bore full fruit later in the leaner OS/360 terminal design.)
  - Kruchten on Canada's air-traffic-control system: all software people sent to ATC classes and days sitting beside live controllers; ATC specialists sent to OO design and Ada courses — until there was enough common vocabulary to work together.
  - Early user exposure keeps humbling: after a decade building a room-filling walk-in protein display, the first user's third-session request — "May I have a chair?" — "A decade's work shot down by one sentence!" The navigation benefit wasn't worth the physical labor. Radiologists preferred sitting and rotating the virtual patient over walking around it. "Almost invariably I have made wrong assumptions about how they would use the new tool."
- **Remedy 2 — close interaction with users via incremental development and iterative delivery (Mills).** Build a minimal-function version that works; give it to users to use or test-drive; iterate. The best way to stay close to users from the very start; works for mass-market products via user samples.
- **Remedy 3 — concurrent engineering.** Designers should dig personally into actual implementation: "even an isolated and unrepresentative implementation experience can wonderfully inform a designer's often idealized or inchoate vision." Danger: a modest sample experience unduly influences design if it's all the designer has — so the best balance is concurrent engineering, with true implementers intimately involved in the design process, their broad experience balancing the designer's limited samples ("in the software field, this same practice sometimes is called just an agile method"). Pulling implementers forward makes demands too: they may need richer visuals (even virtual environments) to foresee gotchas from plans and sections.
- **Remedy 4 — education of designers.** Curricula must include techniques for and practice at understanding users' needs. Gould & Lewis's durable principles: early and continual focus on users by *direct contact from the outset* (many designers think they're doing this when they're merely reading profiles, "presenting," "reviewing," or "verifying" designs with users late); empirical measurement of usage; iterative design. Implementation experience at the machine shop, the job site, actually building the software, is equally crucial. This argues for more project courses with real outside users, even at the expense of book learning: "advanced methods can be self-taught when needed. Gut instincts are harder to acquire." (Projects should be useful if successful but not necessary — the team must be allowed to fail.)

**Apply:**
- Get direct user contact from the outset — apprentice in the users' world, watch real work, run scenarios in mock-ups; never settle for profiles and late-stage design reviews.
- Deliver minimal working versions early and iterate on real user reactions; expect your use assumptions to be wrong.
- Involve real implementers in the design (concurrent engineering), and personally sample the implementation work — while remembering your sample is unrepresentative.
- When crossing a domain divide, invest in cross-training both ways until designer and domain expert share a working vocabulary.

## Ch 16 — Representing Designs' Trajectories and Rationales

Designers should document not only the whats of a design but the whys and the trajectory by which it was reached — priceless for learning and for maintainers ("Be careful how you fix what you don't understand") — but capturing this is much harder than it appears, as Brooks and Razzaque's failed experiment reconstructing a real house-design tree showed.

- The representation problem: knowledge is a nonplanar web (Bush's Memex insight); every practical representation linearizes it — cut edges until the graph is a tree (imposing a hierarchy where there was none), map the tree to a line (usually depth-first), then restore cut links with auxiliary structures (a book: contents tree + index chains; a library: shelf order + author/title/subject indices; Wikipedia's instant rich cross-linking is a genuinely new intellectual tool). Design spaces have the same web structure; design *processes* are inherently harder still.
- The experiment: transcribe the 235-page contemporaneous prose log of the Brooks house-wing design into a decision tree with rationale, using Compendium. The transcription scheme broke repeatedly; each fix meant starting over from page one. Root cause, realized late: no rigorous operational definition of "design tree" existed. "What we learned in trying to reconstruct the design tree is more revealing than the tree itself... This exercise was an experiment that failed."
- Insight 1 — **Design isn't just to satisfy requirements, but to uncover them.** The architect's pavilion proposal was rejected because the resulting house had no central place for the family to congregate and would cost a precious oak — but neither the central place nor the oak were recognized requirements until the proposal was analyzed. "We see this pattern again and again in the log. Design work doesn't just satisfy requirements, it elicits them. A good design process will encourage this phenomenon, rather than suppress it." (Confirms Schön's back-talk theory.)
- Insight 2 — **Design isn't simply selecting from alternatives, but realizing their existence.** Alternatives can't simply be enumerated: some are obvious or borrowed from exemplars, but others require breakthroughs. After much analysis, two Music Room configurations both failed; a third was discovered and instantly liked ("Config C — A real way forward!"); the land purchase was likewise not an on-tree option. "A major part of design is realizing that design options exist."
- Insight 3 — **The tree itself changes as the design changes.** The same rooms hang under different high-level nodes at different design stages; the organizational structure of the tree embodies design decisions already made. Short-lived high-level explorations ("flip house end for end") remain intrusively visible forever in a static tree; abandoned early alternatives differ from later ones only in affecting larger portions of the tree — far-reaching changes become fewer as the design stabilizes. Documenting this evolution "requires a new dynamic tool that doesn't yet exist," tracking not only leaf-ward growth but nodes and sub-trees cut from one branch and grafted onto another.
- Insight 4 — **Tree of decisions vs. tree of designs.** In a decision tree, the final product is a *set of leaves* — hard to visualize the best complete design on any given day. The tree of designs (each node a family of products alike in all decisions down to that node; each leaf a complete design) is conceptually clarifying but combinatorially enormous (n independent binary questions → 2^n nodes) — implausible for a human to construct. Instructive analogy: in agile development, each nightly build is a node in the tree of designs — the best complete design thus far. (Parnas's program-families paper uses the tree of designs as its basic framework.)
- Insight 5 — **Modular vs. tightly integrated designs.** Decision options are rarely independent (Music Room placement constrained Studies, Living Room, and Kitchen), forcing awkward compound alternatives in the tree. Tight dependency makes revision hard (Alexander); a designer may rightly trade design quality against ease of future modification (Parnas), and may also trade design quality for speed and ease of the design process itself. "Modular designs are more readily represented as decision trees. Indeed, this may be what we mean by a modular design." But complete modularity has costs: optimized designs have components achieving multiple goals (the unibody car — lighter and stronger than ladder-frame, but a ladder-framed pickup converts to an SUV more easily).
- Tooling cautions: a structured annotation tool used *during* design "will restrict the ease of having vague ideas, impeding conceptual design — in much the same way a CAD tool is too precise for the quick exploration of creative ideas, whereas sketches allow the designer to be vague." Task-analysis tools and PERT-based project-management tools don't represent decision trees (PERT implicitly assumes the major design decisions are already made). The IBIS/gIBIS/Compendium lineage exists for decision rationale, but even for after-the-fact reconstruction Compendium required work-arounds and couldn't accommodate the tree's size; a generic diagramming tool might do better.
- **DRed at Rolls-Royce — the success story** in computer-aided design rationale: a gIBIS-like capture tool in wide real use (~600 engineers, ~30% of RR engineers, trained; used for conceptual and detailed design, by designers, reviewers, and downstream manufacturing engineers, without facilitators). Why adoption succeeded:
  - A strong rationale-capture *culture* preexisted: engineers were already required to write prose design-rationale reports; the enabling management rule allowed a DRed document *instead* — much easier to do. (BAE Systems, without such a culture, did not widely adopt it.)
  - A dedicated two-person user–builder link (one academic, one corporate point person) trained users, gave support, and filtered/prioritized feedback — in Brooks's view a major role in the success.
  - Typical mode: rationale captured on a whiteboard during a design meeting, then one person formalizes it into DRed.
  - DRed charts stay useful in reviews, but are deliberately *not* under formal revision control: "if DRed were under it, DRed wouldn't be used."
  - Unplanned bonus use: steering and documenting field-fault diagnosis ("here's the data on when, where, and how the engine quit — now, what caused it?").

**Apply:**
- Keep a contemporaneous design log recording decisions and their whys; hand the rationale to maintainers so they don't in ignorance loose a vital stone from the arch.
- Run a process that welcomes requirement-elicitation-by-designing and the discovery of new alternatives, rather than suppressing them as scope churn.
- Consciously trade integration (optimized, multi-goal components) against modularity (ease of future revision, tree-representable structure); know which you're choosing and why.
- If adopting rationale-capture tooling, make it easier than the documentation it replaces, keep it out of heavyweight revision control, and staff a dedicated user–builder feedback link.
- Keep conceptual design sketch-loose; don't impose precise structured tools while ideas still need to be vague.

## Ch 17 — A Computer Scientist's Dream System for Designing Houses — Mind to Machine (Part IV)

Thought experiment: specify the ideal designer's workstation (for house design, generalizable to any domain), focusing on the channel from the designer's mind into the machine. The designer's mind is paramount; the tool's job is to let the designer utter a vision into existence with minimal friction — and new tools can change how one thinks.

- **Progressive truthfulness** (Whitted's technique) — the central proposal. Instead of starting from blank paper and refining upward, start from a *fully detailed exemplar that merely resembles* what is wanted, then adjust attribute after attribute toward the mental vision: "Give me the Three-Bedroom Georgian House. Face it north. Mirror-flip it. Make the living room 14 feet wide. Make the exterior white stucco."
  - At every step you have a complete, internally consistent prototype.
  - Because the prototype is always fully detailed, visual and aural perceptions of it never mislead.
  - Starting from exemplars with consistent style means consistency is the designer's to lose, not to achieve.
  - This is also the program of natural science: models successively approaching the existing creation. In artifact design, the process of designing itself changes the mental ideal being approached — progressive truthfulness radically helps.
- Good design remains top-down: identify key ideas / functional spaces / data structure and algorithm first, then refine.
- Great designers rarely start from scratch; they build on rich inheritance from predecessors and wrestle borrowed ideas into a design with conceptual integrity and coherent style.
  - The "exemplars limit creativity" objection fails: Brunelleschi, Le Corbusier, Gehry, Gaudí all trained by studying precedents. Like Bach, they innovated from mastery, not ignorance.
- **The model library** is the system's foundation and its main hazard.
  - Bad models, too few models, too narrow a variety will limit the emerging designs more than anything else — worst at the beginning.
  - The library must be hierarchically browsable: as it grows, even experts will be at home in parts, passably competent in others, explorers elsewhere.
  - Users and groups need individualized synonym dictionaries that supplement and override the system dictionary; nomenclature can never be centrally rationalized (cf. the joint Army–Navy–Air Force database, where even "What time is it?" had three answers).
- **Noun-verb command rhythm.** Design utterances are imperative sentences: verb + object noun (+ selecting adjectives, + adverbial modifiers). Match input mechanism to grammatical role:
  - Nouns: point or sketch. Verbs: voice is the natural mode (limited vocabulary, wide voice tolerance, rich synonyms, user-modifiable dictionary), with menus and keyboard equivalents always kept active as fallbacks.
  - The one-handed WIMP rhythm (point at noun, travel to verb menu, travel back) loses the user's place; keyboard verb shortcuts for the left hand are a brilliant fix — the novice keeps menus, the expert acquires shortcuts one verb at a time by personal frequency.
  - Quantitative adverbs ("how wide? how many? spaced how?") need precision: auxiliary verbs (Snap To, Align With), small customizable menus, and a numeric keypad that stays out on the desk.
  - Choices exhibit strong locality — many alternatives exist, but most choices come from a small personal subset, so personal palettes and customizable menus are essential.
- Two-handed input (Buxton): dominant hand makes precise manipulations; non-dominant hand provides framing context; both together give approximate dimension ("so big").
- Pointing is powerful but not sufficient: the expert GRIP chemist preferred keying a three-digit residue number he knew by heart over pointing into a 3-D tangle. Experts name what they know; support naming as well as pointing.
- 2-D sketching stays primary even for 3-D artifacts — the retina is 2-D, flat surfaces guide the hand, and sketching appears essential to the thought process. Provide a pen pad sensing position and pressure.
- Viewpoint specification deserves dedicated devices tuned to change frequency (even rarely changed parameters should change dynamically and smoothly):
  - The EyeBall: 6-DOF tracker in a sliced billiard ball glided over the plan, heavily favoring x, y, yaw — the parameters people actually change continually; clutch button for floor changes with no visual discontinuity.
  - The "Toothpick" 2-DOF joystick for exterior view direction, plus default buttons for elevations and three-quarter views.
  - Automatic scene-rocking (torsion-pendulum mode when idle) exploits the kinetic depth effect — stronger than stereopsis — so the designer perceives massing while merely thinking.
- Device design lesson from GRIP: 21 degrees of freedom across devices, but no device overloaded with more than three; users instinctively reached to known physical locations to change values.

**Apply:**
- Start designs from the best fully detailed existing exemplar and mutate it toward the vision; keep every intermediate state complete and consistent.
- Master and study precedents before innovating; treat a curated library of good prior designs as core infrastructure, and worry most about its quality and breadth.
- Give experts fast shortcuts learnable one at a time alongside novice-friendly discoverable menus; never remove the fallback path.
- Match each input/interaction mechanism to the kind of information conveyed (identity, position, quantity, choice); don't overload one mechanism.

## Ch 18 — A Computer Scientist's Dream System for Designing Houses — Machine to Mind

The return channel: what the machine must display to the designer's mind. Mind-machine collaboration demands a two-way channel; the eyes are the broadband path, but the ears serve situation awareness and alerts, and haptic/olfactory senses reach deeper levels of consciousness.

- **Multiple concurrent windows, always.** One active window forces the wasteful universal loop observed in all real design work:
  1. Study a big spatial chunk for context. 2. Zoom in. 3. Create or manipulate some local portion. 4. Zoom out. 5. Repeat.
  - Provide context view and detail view simultaneously, switched by movement of eyes rather than hand. No serious design shop has any excuse for being display-constrained.
- The concurrent views:
  - *Drafting/drawing view* — electronic drafting table; work surface separated from the display so hand and arm no longer obscure the view; eye-resolution display; layered drawings with controllable transparency to focus attention on one aspect while maintaining conceptual context.
  - *Context view* — a second full-size screen, not a thumbnail (one needs to see details in it too); normally the entire plan; during library selection it shows the library tree while the other shows the placement target.
  - *3-D view* — the design as currently specified, always in full detail; an auxiliary creation tool for a seated designer with controls at hand (small dome or single 3-D window, not an immersive CAVE, which was designed for viewing rather than creation); mode-switchable stereo so glasses aren't worn constantly.
  - *Workbook view* — the designer's log. Designers arrive with in-progress designs and action plans; they leave with updated designs and plans, automatic action logs (enabling backtracking via version control), and dictated notes recording **what was tried and why, what was rejected and why, what was kept and why**.
    - The whys cannot be captured from the action log. They aid refreshment after interruption, recall the branching thought-trails skipped during exploration, and are priceless for new team members and the designer's project heirs.
    - Two page-corner inserts continuously show the current cost estimate and the current value of the budgeted commodity.
  - *Specification view* — prose specifications growing to finality contemporaneously with the drawings, not afterward. Feasible under progressive truthfulness because specs are highly stylized and each library exemplar carries its own specs to be modified in place (product databases like Sweets supply components). Spec-to-drawing propagation is tractable; drawing-to-spec is a research problem.
- Audio display: recorded playground sounds made a mediocre architectural visualization leap to life. Plant sound sources (TV, traffic, children) and listen during walkthroughs for pleasure and nuisance; a gridded sound-intensity plot suggests where to listen.
- Haptics: seems to reach the gut like no other modality, yet no plausible use of existing haptics technology was found for this system — honesty about a fashionable technology.
- Generalization to software: a dream system for building software needs the rich starting library, the Design view, Context view, Workbook view, and a **Test Cases view**, all suitably cross-linked (the 3-D apparatus is domain-specific).
- Feasibility: buildable now, affordable at least for larger firms — but only **incrementally, with continual trials by real designers**. Any all-at-once project to build such a superficially pondered system would almost surely fail.
  - The hardest part is assembling a good initial library of starting models; an Open Source prestige-incentive model could seed it, with automatic usage-based winnowing (if it isn't viewed, out it goes).

**Apply:**
- Work with a detail view and a full-size context view visible simultaneously; never force zoom-toggling through one window.
- Keep a design workbook capturing rationale — what was tried/rejected/kept and *why* — alongside the automatic history.
- Grow specifications and tests in step with the design, starting from the exemplar's, not as an afterthought.
- Keep the budgeted commodity's running total permanently visible while designing.
- Build ambitious tools incrementally under continuous real-user trial; never all-at-once from a superficially pondered spec.

## Ch 19 — Great Designs Come from Great Designers (Part V)

Not from great design processes. Products with passionate fan clubs were, almost without exception, produced *outside* formal product processes (as were the atomic bomb, nuclear submarine, ballistic missile, stealth airplane, Spitfire, penicillin — each by a small team set apart); process-produced products lack fans. Process controls cost and prevents failure, but by its nature suppresses greatness — so the job is to protect and empower great designers within necessary process.

- Why product processes stifle great design — each "by its very nature":
  - Process is *conservative*: it brings similar things into one orderly framework, so the really dissimilar, highly innovative thing doesn't fit (the PC was not the same thing as the 1960s glass house).
  - Process aims at *predictability*: a product roughly defined by business needs before any great designer has spent time on the problem, delivered at a stated time and price. Predictability and great design are not friends.
  - Process *fights the last war*: it encodes tactics that worked before, both irrelevant for a product addressing a new need (the iPhone was not the same thing as a mobile phone).
  - Process is *veto-oriented*: many expert watchdogs, each paid to avoid a separate failure cause, each separately biased toward finding reasons not to proceed.
  - Consensus mechanisms *take off the sharp edges by forcing compromises — but the sharp edges are the cutting edges*.
  - Rules accrete without any force for elimination: each mistake-experience begets new rules or approvals; bureaucracies become more Byzantine as organizations succeed and grow.
  - Consensus *eats the resource*: meetings take time, and great designers' time is exceedingly precious.
- S/360 Model 20 story: a talented, scrupulous Böblingen lab had never gotten a product into IBM's main line because it conscientiously followed the 100-page Corporate Product Procedure. Successful managers elsewhere made bold, well-chosen *exceptions* to the rules. An experienced procedure-manipulator was sent to manage; the talent was unlocked; the product became phenomenally successful.
- Why have process at all: corporate approval, tapping experience to catch oversights, agreed schedule and budget are inescapable. The trick is to **hold process off long enough for great design to occur**, so lesser issues are debated once the great design is on the table — rather than smothering it in the cradle.
- Product processes are properly designed for **follow-on products**, which rightly dominate:
  - Use reveals shortcomings to correct; users find unexpected uses that enlarge the concept incrementally; demonstrated usefulness creates demand for a more capable product; yet popularity breeds lock-in — users want what they know, not revolution.
  - Follow-ons are highly constrained, must be selected from many possible directions, and must be monitored; the definition → forecast → cost → price cycle does this efficiently.
  - But forecasting and cost estimating depend on experience with *similar* products. For innovation, one must step outside of process.
- Process raises the floor, never the ceiling:
  - CMM-style discipline brings up the low end and average of practice — valuable, especially in software, where average practice lags best practice unusually far.
  - No amount of process improvement can raise the ceiling. Great designs come from talented people doing hard work. Apple: "We're at Level 1 [CMM] and will always be" — with results that speak for themselves.
- How S/360 harnessed unavoidable process without being stifled: core design team insulated from normal oversight; strong support from several levels of bold managers; word from the top that this known-revolutionary gamble would require bending normal processes; exceptional recruiting power; enough money; and high-talent process people properly pushing back weekly.
- Designing a process that permits greatness:
  - Explicitly identify the matters of fundamental importance and constrain *those, and those only*. A process is a protective mechanism: protect the crown jewels securely, but eschew building high fences around the garbage cans — protectors instinctively overprotect.
  - Provide easy and swift exception mechanisms: exercisable at the appeal of any project manager with the approval of only one sufficiently high-level boss. "All rules can be broken."
- Talent is wildly unevenly distributed, and no two people have the same bundle. A wise leader draws responsibility boxes around the people he has and can get, rather than putting people into abstractly ideal boxes.
  - Structures must cold-bloodedly recognize that people who have done great designs are the most likely to do more if entrusted with freedom and authority.
- Great designers require **bold leaders who demand innovation**: the top leader must passionately want innovative products (IBM under both Watsons; Apple under Jobs — less so between his reigns).
- **Entrust each design to a chief designer** — conceptual integrity, the most important attribute of a great design, comes from one or a few minds working uno animo:
  - The manager must not second-guess the design (a real temptation — the manager is often a designer too, but design and management are different jobs, and his attention is fragmented).
  - The chief designer has complete authority over the design and ranks sociologically equal to the project manager.
  - Shield him from outside watchbirds and time diversions.
  - Provide tools and help as *he* sees the need — what he is doing is of prime importance.

**Apply:**
- For genuinely new products, work outside the standard product process; apply process to follow-ons, where it belongs.
- Delay process gates until a great design exists to be debated.
- Constrain only the explicitly identified matters of fundamental importance, and build a fast one-boss exception path into every rule system.
- Appoint a chief designer with full design authority equal in rank to the project manager; don't second-guess, do shield, do resource.
- Use process discipline (reviews, maturity checklists) to raise the floor of practice — never expect it to produce excellence.

## Ch 20 — Where Do Great Designers Come From?

Great designers must be deliberately taught, recruited, grown, managed, and protected — with the same institutional seriousness organizations already apply to growing managers. Design skill is mastered only through critiqued practice, not lectures.

- **Teach by critiqued practice** (Schön): "Technical Rationality" — basic science first, application skills second — is dead wrong for teaching professions.
  - Medicine (clinics, grand rounds), architecture (studio in all years), law, ministry, and the craft guilds all converged independently on critiqued practice. The PhD dissertation is exactly this method for teaching research.
  - Engineering and especially software education waste the most precious commodity — student time — on lectures and prescribed labs. Start critiqued design in the freshman year, concurrent with science; use co-op/sandwich programs.
  - Effective style education: do a well-constrained design *in the style of* a master (a fugue in the style of Bach, an architecture in the style of Cray, a building in the style of Wren) and have a discerning mentor critique the stylistic inconsistencies. This demands a boldness in free-form subjective criticism that science-centered faculties lack.
  - Students can mentor each other; the most effective way to learn other design styles is to undertake to teach them.
- **Recruit for design brilliance**, not for the recruiter's own job profile:
  - Managers subconsciously assess candidates by "could he do my job well?" — favoring the articulate, meeting-effective leader and overlooking the introvert, the slow-spoken, the unconventional. Brilliant designers come in these packages too.
  - Select by looking at portfolios of the design work itself (Microsoft has candidates craft programs), not oral presentations about the work.
- **Grow them deliberately**, as organizations grow executives (and as Moses, David, and Paul were grown — planned formative apprenticeships):
  - Identify promising talents early; track them; assign mentors; rotate assignments for planned variety.
  - The course of a young designer's career should itself be *designed* — for variety, depth of involvement, spiraling challenge and responsibility.
  - The most fruitful early assignments often place young designers inside the *user* organizations of what they will design (Brooks's stints in a payroll shop, rocket-trajectory computing, cryptanalysis, telephone switching, and as an apprentice computer operator gave visceral understanding of requirements).
- **Make the dual ladder real and honorable**: a technical career path whose compensation *and sociological status* match management's.
  - Market forces equalize salaries; prestige requires strong proactive measures — equal offices, equal staff support, reverse-biased raises when duties change.
  - Managers, being human, are inclined to consider their own tasks the more difficult and important; countering that takes deliberate assessment of what makes creativity happen.
- **Plan formal educational experiences** throughout the career:
  - Reason 1, continual retooling: technical education obsolesces fast; a good teacher's balanced overview can double learning efficiency versus journal-reading and conference-going.
  - Reason 2, deepening and broadening by studying good and bad designs of predecessors and contemporaries: outside education brings detachment — company-sponsored education, like shop culture, emphasizes its own traditions.
  - Ted Codd story: sending a promising engineer for a mid-career PhD — at the time an incidental personnel action — led to the relational database, a Turing Award, and the principal application of IBM's most profitable line. Brooks's most productive single act as a manager.
- **Plan sabbaticals** outside the organization (loan to a customer, university teaching, agency assignment): preventing stagnation of creative people is a great investment.
- **Manage them imaginatively** — the Cocke–Gomory story:
  - John Cocke — couldn't manage a group, rarely published, mostly thought and talked to bright people — produced instruction pipelining, global compiler optimization, and RISC, each worthy of a Turing Award, through collaborators who captured and implemented his ideas.
  - The second genius was Ralph Gomory, who built an organization and management style that enabled each person to contribute in the way best suiting his particular bundle of talents — treating each great mind *differently*, according to its nature and needs, and paying Cocke as IBM Research's highest contributor.
- **Protect them fiercely**:
  - From *distraction*: design productivity requires flow — an uninterrupted state of high creativity and concentration. Meetings, phone calls, emails, rules, staff bureaucracies who make rules to simplify their own jobs, customers, and visitors all destroy it. Countermeasures: quiet mornings; IBM's closed Boca Raton PC lab that even relevant corporate staff could not enter; Brooks closed the S/360 project to all non-project visitors for four months; Airbus UK's Technical Director simply answered "No" to a request to speak with his chief designer.
  - From *managers*: a mediocre or insecure manager fails to recognize the jewels on his team, resents the better or better-paid designer, and smothers creativity with petty put-downs. Higher management must actively change the first-line manager — raise his vision of his own enabling role, train him in encouragement and leadership.
  - From *managing*: organizational culture drags great designers into management, where their potential dies. Seymour Cray repeatedly and deliberately pulled small teams into seclusion ("thirty-five people including the janitor") and disentangled himself from all management to keep designing — three times over.
- **Growing yourself as a designer** — you alone are responsible for your growth program:
  - Constantly sketch designs; make some fully detailed, for the devil is in the details and many a grand scheme has foundered on a little submerged rock. Keep a notebook of patterns encountered and invented (Leonardo's Notebooks).
  - Seek knowledgeable criticism of your designs.
  - Study exemplars and precedents with humility: precedents that survived long criticism have some deep excellence; master the excellence that has gone before even if your muse then drives you elsewhere. Prefer designs that passed a real-money test over published-only ones.
  - **Assume competence**: the right question about a precedent is "What led such a smart designer to do that?" — never "Why did he do such a fool thing?" The answer lurks in the designer's objectives and constraints, and discovering it brings new insights.
  - Listen to and read contemporary designers on their own work; cast your studies of others' designs into a common format with a short critique of each.
  - Self-education exercise: design a 1,000 ft² house floor plan for a given family and site, keeping a dated journal of questions, decisions, and reasons — then analyze the constraints deduced, the budgeted commodity and how it was managed, the desiderata followed, how alternatives were compared, and your design trajectory.

**Apply:**
- Learn and teach design through critiqued practice on real designs, including exercises in the styles of masters.
- Recruit from portfolios of actual design work; deliberately look past presentation polish.
- Plan designers' careers as deliberately as managers': mentors, rotations through user organizations, outside formal education, sabbaticals, an honored dual ladder.
- Guard designers' flow ruthlessly, and keep great designers out of management.
- When studying any precedent design, assume competence and reconstruct the objectives and constraints that made the decision sensible.

## Ch 21 — Case Study: Beach House "View/360" (Part VI)

Part VI framing (applies to all seven case studies): the boldest design decisions, whoever made them, accounted for much of the goodness of the outcomes. They were due sometimes to vision, sometimes to desperation. They were always gambles — extra investment in hopes of a much better result.

A family-designed-and-built oceanfront vacation house (Caswell Beach, NC; shell 1972, completed 1997), documenting how very many decisions even a simple, understandable structure requires and the considerations affecting each.
- Bold decision: place the house as close to the ocean as the warranty-deeded lot allowed — about 40′ forward of all neighbors, at somewhat greater wash-away risk.
- The **budgeted resource turned out to be inches of ocean frontage** (hence view and breeze). Frontage was rationed room by room through explicit priority argument: Living Room first (and in the corner with the best view), dorms before Guest Room, Master Bedroom last (a sleeping space, not a living space).
- Decisions cascade through corollaries: no air conditioning → breeze paramount → casement scoop windows → steel crank mechanisms corrode in salt spray → replaced by year 35. The failure: inadequate weighting of *maintenance* in a long-life project, plus inattention to all materials of construction. Tower for view → roof ridge capped at Tower windowsill height. Windows everywhere → structure needed added anti-skew bracing.
- Serendipity: a spiral staircase forced by floor-space cramping became a piece of spatial art — given a necessity, feature it as sculpture.
- In-construction changes (moving the eating area after mock-up studies, omitting a planned partition, adding ground-floor exterior showers) substantially improved delight and commodity. But not all opportunities created by changes were exploited: the west deck consumed 42″ of the critical frontage budget for a purpose (outside access to the bath) that the later shower decision had eliminated. When one decision removes a constraint or desideratum, re-examine every decision that depended on it.
- Failures: neither the amateur nor the professional architect thought carefully about placing pilings under the centers of weight with roughly equal loads — uneven settling in sand required structural remediation in years 25 and 28. A consulting architect's back-to-fundamentals veto ("If you can design a house to keep the rain out, you will have done well") killed a romantic breaking-wave roofline that would have leaked.
- Hurricanes validated the firmness goals; the "throw one away" reflection: with hindsight, exploit the true budgeted resource down to the last inch, and remove the deck once its justifying requirement vanished.
- Stated general lessons (apply to every substantial design project — hardware, software, or buildings):
  1. Check your professional architect's work very carefully, and ask for rationale. Even honest, competent, conscientious architects make mistakes.
  2. Inspect often and thoroughly during construction. Even honest, competent, conscientious builders make mistakes.
  3. Think hard about all aspects of maintenance. One maintains any successful design a long time.

## Ch 22 — Case Study: House Wing Addition

A 1987–1992 home addition (Chapel Hill) with a 235-page contemporaneous design log later encoded as a formal decision tree; illustrates the interplay of design and the *discovery* of requirements.
- Bold decision (made mid-process, from desperation, when nothing was working): **defer the budget constraint; design for function; then value-engineer**. The $100K ≈ 1,000 ft² constraint was inhibiting all thinking; dropping it radically freed the design. Same principle proven in computer graphics: the way to a cost-effective system is to make an effective one and cost-reduce it, not to make a cheap one and augment it until useful.
- Bold decision (late, revealed by a use case): move the Master Bedroom into the midst of the public spaces. Running the biweekly 40-person student-meeting scenario against a congealed design surfaced "where do the winter coats go?" — the room they currently went in had been designed away. A low-frequency use case revealed a hitherto unperceived requirement and reorganized the whole plan (and let the East End redesign be abandoned entirely).
- Key decision: buy a 5′ strip of land from the neighbor when extensive iteration could not reconcile the Music Room's required shape with the town's setback constraint — sometimes the right move is to change the constraint itself.
- Process features that worked:
  - Partitioning the house into three almost-separate design problems proved liberating; all later design used it.
  - Two design-and-build phases, years apart, kept design and supervision manageable.
  - ~60 months of design versus 9 months of construction; design continued until the designers were satisfied, unconstrained by a construction schedule.
  - Wide consultation; sliding-door integration of Music Room and Living Room to serve occasional recitals without merging the rooms permanently; Alexander's "daylight on two or preferably three sides" pattern applied scrupulously to every new room.
- Verification incidents: a draftsman misaligned the foundation drawing against the floor plan (found after pouring); window heights right on plans proved wrong on site once the falling terrain was visible during framing — the sort of problem plans and elevations never reveal but a virtual-environment simulation would have shown during design.
- After 17 years: no "wish we had done that differently" list. The redesigned house also met needs the owners didn't know they had — for products, this is the usual case, not the exception.
- Stated general lessons:
  1. Spend time on design — far more per unit than seems cost-effective. (OS/360 would have benefited greatly from more design time before implementation; the product would not have cost more in total.)
  2. Talk many times, lengthily, with the principal user(s), showing prototypes they can understand.
  3. Run lots of use scenarios.
  4. Double-check the work of professionals — architects, draftsmen, decorators. Make sure you understand it and that it is accurate.

## Ch 23 — Case Study: Kitchen Remodeling

Phase II of the same remodeling (1995–1996): a small, tightly constrained kitchen redesign demonstrating the distinct power of each design tool in the spectrum.
- Bold decisions: move the exterior wall outward (transformed the design by relaxing the critical width budget) and cut a door through 8″ of brick between Kitchen and Living Room (expensive; transformed the traffic pattern of the whole house — almost all east-west traffic now uses it).
- Explicit budget rationing: the north-south width at the pinch point was itemized — eating table ≥30″, traffic passage ≥24″, sink island ≥24″, sink-stove work space ≥36″ (mock-ups later proved 44″), stove and counter ≥27″ — totaling more than the existing 11′11″. The arithmetic *proved* the wall had to move or the passage be sacrificed.
- Pruning the design tree: extensive study of every "move the basement stairs" alternative ended in rejecting the whole branch — a critical moment that radically narrowed the field of possible designs (Simon's tree search: explore instances, fail, go up a level, rule out the subtree).
- Tool ladder, each adding value the others did not:
  - Sketches, then a layered CAD model (the CAD file served as the design document) at multiple scales.
  - A contemporaneous design log capturing rationale and the wanderings toward each decision.
  - An isometric drawing kit; cardboard scale models (viewable from any angle, richer than isometrics).
  - **Full-scale mock-ups exercised with use scenarios** — cardboard boxes for the exterior push-out, tables and sawhorses for counters — the only satisfactory way found to establish minimum tolerable spacings and the ease bought by measured relaxations.
  - **Virtual-environment walkthrough** — revealed that planned hanging cabinets over the island broke up the visual space and made the kitchen feel small and cramped (redesigned to keep the shelf space elsewhere); also flagged an intrusive hanging lamp and confirmed the mural and diagonal flooring. Invisible in plans and even mock-ups.
- VE versus mock-ups: VEs will become cheaper and easier; mock-ups won't. But touchable mock-ups add presence and spatial learning that visuals alone do not (Insko's passive-haptics experiments), so mock-ups stay worth their cost for intensively used spaces and widely replicated ones.
- Stated general lessons:
  1. The kitchen is the most important room in the house and rewards extensive design work — invest design effort where usage intensity is highest.
  2. The happy outcome (only minor regrets after 14 years) came from designers-as-users — realistic, representative use cases, as with Linux — plus generous design time spent testing pseudo-prototypes (mock-ups, VE models) against extensive use cases. Most projects need a larger share of total schedule devoted to design.
  3. Very wide consultation with friends yielded crucial good ideas, including the basic configuration.
  4. Full-scale mock-ups, together with use scenarios, proved invaluable.
  5. Virtual-environment technology provided important information beyond floor plans and even mock-ups, especially about visual space and the feel of the room.

## Ch 24 — Case Study: System/360 Architecture

IBM's 1961–1964 replacement of six mutually incompatible product lines with one strictly compatible computer family — "IBM's $5,000,000,000 gamble," "you bet your company."
- Boldest decision (CEO Thomas J. Watson, Jr.): drop all further development of all six existing product lines in favor of one new line, exposing the installed customer base to competitors' compatible machines.
- Bold decision: make the six new computers strictly upward- **and downward**-binary-compatible with exactly one architecture.
- Bold decision: base the architecture on the 8-bit byte, obsoleting all existing I/O and auxiliary devices, even card punches. The most hotly debated decision; settled on the future application promise of the lowercase alphabet. It changed computer architecture completely and permanently.
- Compatibility as discipline: strict two-way compatibility protected the low end from functional deficiency and the high end from excess — like a strict page limit yielding cleaner, more effective writing. The careful conceptual separation of architecture, implementation, and realization made compatibility definable.
- Recovery from failure: six months into a stack architecture, performance evaluations showed it killed the low end (stack in main memory, not registers). Amdahl proposed an **internal design competition**: two teams independently produced base-register solutions; the competition produced concurrence on many issues, spotlighted the crucial differences, and powerfully lifted morale — junior architects competed on the same basis as the distinguished.
- Other significant decisions:
  - 24-bit addresses, reluctantly, with provisions for a future jump to 32 — but Branch-and-Link was inadvertently designed to use the reserved upper 8 bits. **The leader had failed to indoctrinate the whole team strongly enough with the expansion vision, and no review caught it** — a clear example of the danger of team designs.
  - Standard logical/electrical/mechanical I/O interface for attaching all devices — radically reduced configuration and software costs.
  - Full supervisory provisions (interruption system, memory protection, privileged mode, timer) so an operating system could control the machine without manual intervention.
  - Mandatory end-to-end single-error detection despite no evident customer willingness to pay — professional responsibility, because people had begun trusting computer answers.
  - Microprogrammed implementation mandated unless conventional logic showed a 33% performance/cost advantage; microcode enabled the 7090 emulator and the near-overnight 1401 emulator that won the decisive internal product fight and solved the customer-conversion nightmare.
- Mistakes owned: rejecting virtual memory (expert designers going wrong in a big way; rectified in S/370); the decimal datatype (hardware cheap; software cost and conceptual complexity not); failing to see that an I/O channel is just another computer (Cray's CDC 6600 peripheral processors were the elegant embodiment); the irregular SS instruction format; the missing floating-point guard digit that forced field modification.
- Outcome: the architecture endures 45+ years on, still backward-compatible; commercially a major success; judged by Gordon Bell the most intellectually influential computer in history; licensed and copied worldwide.
- Stated general lessons:
  1. Allow plenty of project time for design. It makes the product much better and useful longer, and might even make delivery sooner by reducing rework.
  2. Multiple concurrent implementations of one architecture strongly protect the architecture from bad compromises — with only one implementation, it is always easier and cheaper to change the manual than the machine.
  3. A design competition when the first design runs aground is very fruitful: concurrence, clarity on crucial differences, and team morale.
  4. For totally new designs, devote early design effort to establishing metrics for performance and essential properties, plus approximate cost surrogates.
  5. Market forecasting methodology is designed for follow-on products, not radical innovations; designers of totally new products should spend early effort getting forecasters on board with the new concepts.

## Ch 25 — Case Study: IBM Operating System/360

The first second-generation software support package (1961–1965): one operating system, compilers, and utilities for the entire compatible family. The successes were architectural concepts still alive today; the failures teach the costs of removed constraints, missing conceptual control, and insufficient design time.
- Bold decisions:
  - One software package for the whole range of computers and configurations, generated to fit varied memory sizes and I/O.
  - Mandate a random-access device (disk) for operating-system residence — the biggest single difference in design concepts; modules could be small, function-specific, rolled in on demand.
  - Require no operator: the OS, not the operator, controls the computer (the console is just another I/O device) — while the same OS can also be configured for full operator control.
  - Multitasking: concurrent safe execution of independent, *untrusted* programs, enabled by the S/360 supervisory hardware. Full-generality multiprogramming proved much harder than expected.
- Most important innovation (Brooks's judgment): the standard *software* I/O interface complementing the standard hardware interface — **device-independent I/O** through abstract access methods, with dataset names bound to particular datasets, devices, and media only at scheduling time.
  - Scheduling time explicitly recognized as a binding occasion distinct from compile time (rigid) and run time (overhead) — modules linked and datasets bound by the Scheduler.
  - Result: weekend reconfigurations routine; most applications rerun without recompilation.
- System structure mirrored its diverse ancestry — Supervisor from interrupt handlers, Scheduler from tape-based job schedulers, Data Management from I/O subroutine libraries: structure follows evolutionary history.
- Design weaknesses named:
  - **Too rich**: disk residence removed the size constraint that had disciplined earlier OS designers, and functional goodies of marginal usefulness flooded in — featuritis.
  - System-wide shared control blocks readable and writable by every programmer: adopting Parnas's information hiding (published 1971, embodied today in object orientation) would have avoided much construction grief and all subsequent maintenance grief.
  - JCL — "the worst programming language ever designed by anybody anywhere ... under my management": the very concept was wrong — seen as "a few control cards to precede the job," never recognized as a programming language.
  - Missing the virtual-memory boat made the retrofit more difficult and costly than designing it in.
  - Complexity carried from history: count-key-data variable blocks instead of fixed-length blocks; an unnecessarily complex device–control-unit–channel attachment tree; three sequential access methods where one could have served.
  - The best batch debugging system ever designed — totally obsolete from birth (interactive terminals were the future).
- Process weaknesses: it should have been built in PL/I — a high-level language would have been just as fast, far cleaner and more reliable, and built more swiftly; and rigid architectural control over all interfaces should have been maintained, with external declarations included from libraries rather than crafted anew in each module.
- Stated general lessons:
  1. Give the system architect full authority over the design — not doing so was a "multimillion-dollar mistake."
  2. Take the time necessary for sound design and prototyping, whatever the schedule pressures — the project completes sooner, not later. Chapters 21–24 show the benefits of enough design time; this case illustrates the opposite.

## Ch 26 — Case Study: Book Design of *Computer Architecture: Concepts and Evolution*

Design thinking applied to a book (Blaauw & Brooks, ~1971–1997): scope, structure, and sequence are design decisions like any others — and schedule is part of design goodness.
- Bold decision: hold to a narrow, precise definition of computer architecture — exactly the properties visible to the programmer, excluding speed; distinguish architecture, implementation, realization — against the drift toward loose usage. The precision is what makes program compatibility definable.
- Bold decision: a "zoo" of 30 computer architectures described in a rigorously standardized format (highlights and peculiarities, context, drawn programming model, enumerated design decisions, precise formal descriptions).
- Bold decision: build, test, and publish executable APL simulators of every zoo machine. Building the simulators forced close scrutiny and added great precision to the descriptions — even though few people ever ran them. A formal executable model disciplines the describer regardless of its audience.
- Matrix organization: every design decision treated twice — once systematically by decision domain, once in the context of all the interrelated decisions of a specific machine. Sequence is the hardest single design decision in expository writing: the graph of interrelated concepts must be cut to a tree to map onto linear text; needing two orders, they used both, with heavy cross-referencing.
- The 80-some decision trees link into one vast unified tree — a formalism that treats design as search of a well-defined space, the very model this book argues against; the author's view broadened after the formalism was built.
- Stated lessons:
  1. A less ambitious work sooner would have been more useful to the profession. The assumption of an essentially unlimited time and effort budget was a major mistake; the book (26 years in the making) came out too late for maximum influence, and teaching from it now leads with the vivid zoo examples rather than the systematic theory.
  2. Book writing has logarithmic convergence: checking the last uncertain facts, fixing the last glitched figures, verifying the last obscure references consume inordinate proportions of total effort — the hardest little tasks get put off until the end.

## Ch 27 — Case Study: A Joint Computer Center Organization (TUCC)

An organization as a designed artifact: Duke, NC State, and UNC-Chapel Hill pooled resources (1964–1992) to co-own one high-performance computation center, with no prior organizational model to copy. The **budgeted commodity in this design was decision-making power**: how to protect each owner's distinct vital interests while enabling efficient decisions.
- Bold decision: a jointly owned center at a neutral site, motivated by the quadratic performance/price curve — spending n times as much bought at least n² times the computing — economics strong enough to justify overcoming the real difficulties of co-ownership.
- Governance design decisions:
  - Careful separation of policy (board of directors, meeting monthly) from operations (staff under a Director/CEO).
  - Board of ten: three chosen by and from each campus, plus the Director — small enough to work, big enough to represent various segments of each campus.
  - Voting by **simple majority of members, not of institutions** — deliberately chosen to make the board think as a unit and discourage division by institutional affiliation. Unanimous consent was rejected early: requiring consensus makes decisions too hard.
  - **"Issues of Fundamental Importance"** — explicitly enumerated in the by-laws (selecting/discharging the Director; annual budget increase over 10%; amending articles or by-laws) — required unanimity of *institutions* (each institution's vote decided by two-thirds of its delegation). Chapter 19's principle embodied: constrain the crown jewels, and only those.
  - **Escape hatch**: any owner could elevate any issue to fundamental status through a deliberately onerous procedure (delegation tables it a month; the institution's CEO elevates by letter). So any institution could, with deliberate effort, stop any action inimical to its vital interests.
  - Rotating two-year chairmanship among the owners. Power explicitly rationed between staff and board, majority and minority, academic and administrative users.
- Outcomes:
  - The escape hatch was never used — its mere presence was a great psychological comfort and prevented sessions where any group felt trapped and forced to fight for its life.
  - Staff and board both operated as a single enterprise. Divisions of opinion ran along role lines — the three finance representatives often voted together, as did the three computer-center directors and the three faculty users — not along school lines.
  - The model flexed to unequal usage shares via ad hoc capacity deals, preserving the form of equal ownership; unequal shares were eventually formalized with use-fraction-triggered board representation.
  - The organization worked for 18 years, through two Directors and three mainframe generations; it was obsoleted by the minicomputer revolution, not by governance failure. A side program (NCCOP) seeded computing at ~100 other North Carolina institutions with free time, terminals, and circuit-rider mentors.
- Stated lessons learned:
  1. Careful and explicit identification, at the beginning, of the vital interests of each partner and of the facility's director was a big help in arriving quickly at agreed organizational mechanics.
  2. An ultimate appeal procedure, though not easy to invoke, assured each participant it wouldn't be trodden upon.
  3. Recognize the differing interests *within* each partner and get them represented in each delegation — divisions then run by area of responsibility, not by school.
  4. A governing board easily becomes a rubber stamp for management; monthly meetings were necessary to avoid that hazard.
  5. CEOs tend to fill board meetings with presentations rather than real issues, and fail to use board members severally as advisers in their areas of expertise — a real loss.

**Apply (Chs 21–27, the case studies):**
- Identify the true budgeted commodity of the design early (frontage inches, kitchen width, address bits, decision power) and ration it by explicit argument; re-check dependent decisions whenever a constraint changes.
- Make the bold gamble that most improves the outcome; boldness from desperation (a design competition, buying land, moving a wall) counts as much as boldness from vision.
- Design first for function with the budget deferred, then value-engineer; never let a numeric constraint strangle exploration.
- Run many concrete use scenarios — including low-frequency ones — against congealed designs; they surface requirements nothing else will.
- Escalate through the cheapest tool that answers the current question: sketch → CAD → scale model → full-scale mock-up → VE walkthrough; keep a rationale log throughout.
- Spend generously on design time before implementation; indoctrinate the whole team in the guiding vision so reviews can catch violations, and verify professionals' work.
- When constraints prove intractable, consider changing the constraint itself (buy the land, move the wall, raise the minimum memory).
- In organizational design, enumerate each party's vital interests, protect only those with strong consensus rules, decide everything else by simple majority, and provide a hard-to-invoke ultimate appeal.

## Ch 28 — Recommended Reading

Works Brooks singles out as exceptionally valuable on the design process as such:
- Blaauw & Brooks, *Computer Architecture: Concepts and Evolution* (1997) — architecture/implementation/realization distinction; design trees; what makes an architecture good.
- Boehm, *Software Engineering* (2007) — indispensable collected papers across software design, management, and research.
- Brooks, *The Mythical Man-Month* (1975/1995) — "No Silver Bullet" separates design problems into essential and accidental; 1975–1995 retrospective.
- Burks, Goldstine & von Neumann (1946), "Preliminary discussion of the logical design of an electronic computing instrument" — the most important computer paper ever written; stunningly comprehensive.
- Cross, *Designerly Ways of Knowing* (2006) — the devastating empirical critique of Simon: real designers don't work by rational search, and here are the studies showing it.
- DeMarco & Lister, *Peopleware* (1987) — research results on the nontechnical factors affecting design quality (flow, teams, environment).
- Hales (1987/1991), *An Analysis of the Engineering Design Process in an Industrial Context* — the most complete published documentation of a real, substantial design process, by a co-designer serving as scholarly observer.
- Hennessy & Patterson, *Computer Architecture: A Quantitative Approach* — the definitive text on designing computer architectures; shows the convergence to a standard architecture.
- Hoffman & Weiss (eds.), *Software Fundamentals: Collected Papers by David L. Parnas* (2001) — the other indispensable software-design collection (information hiding et al.).
- Mills (1971), "Top-down programming in large systems" — teaches and argues for incremental design and programming.
- Royce (1970), "Managing the development of large software systems" — the classic paper describing *and decrying* the Waterfall Model; advocates an alternative.
- Schön, *The Reflective Practitioner* (1983) — design as reflective practice; the foundation for critiqued-practice education.
- Simon, *The Sciences of the Artificial* (1969/1996) — the most influential and articulate proposal of the Rational Model for design.
- Winograd et al. (eds.), *Bringing Design to Software* (1996) — a very helpful collection including important papers.
- Wozniak, *iWoz* (2006) — an engineer's-engineer autobiography with many insights into design.

**Apply:**
- For deepening design judgment, prioritize Schön, Cross, Parnas (Hoffman & Weiss), Peopleware, and Mythical Man-Month; read Simon as the model to argue with, not the method to follow.
