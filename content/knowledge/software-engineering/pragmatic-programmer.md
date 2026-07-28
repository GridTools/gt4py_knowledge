---
title: The Pragmatic Programmer — Andrew Hunt & David Thomas (1999)
description: "Book notes: the pragmatic attitude plus its load-bearing ideas — DRY, orthogonality, reversibility, crash early, automation and testing — with all 70 tips verbatim."
tags: [software-design, dry, orthogonality, reversibility, design-by-contract, assertions, testing, automation, refactoring, tracer-bullets, prototyping, tooling, book-notes]
---

Pragmatic programming is an attitude before it is a technique: take responsibility for
your work and career, think critically about what you are doing while you are doing it,
and improve continuously (kaizen). The craft rests on a few load-bearing ideas — DRY
(one authoritative representation for every piece of knowledge), orthogonality
(eliminate effects between unrelated things), reversibility (there are no final
decisions), crash-early paranoia (design by contract, assertions), and ruthless
automation and testing — applied through concrete daily practices. The book's essence is
its 70 numbered tips; every one is transcribed verbatim below at its point of
appearance, and the complete quick-reference list closes the file.

## Ch 1 — A Pragmatic Philosophy

From the preface — what makes a Pragmatic Programmer:

- Pragmatism: no best tool/language/OS exists, only solutions more appropriate to particular circumstances. Choose from a broad background instead of being wedded to one technology; adjust your approach continuously as work progresses.
- Shared traits: early adopter/fast adapter; inquisitive (collect little facts); critical thinker (never accept "because that's the way it's done"); realistic (understand the underlying difficulty of problems); jack of all trades.
- Programming is a craft; engineering discipline still leaves room for individual craftsmanship ("We who cut mere stones must always be envisioning cathedrals").
- Kaizen: continuous small daily improvements. Every day refine existing skills and add new tools; results compound over years like the Eton lawns.

> **Tip 1: Care About Your Craft**

> **Tip 2: Think! About Your Work**

Never run on autopilot — critique every decision, every day, in real time.

### The Cat Ate My Source Code

Take responsibility for yourself, your career, and your project; admit ignorance and mistakes honestly and directly.

- Responsibility is something you actively agree to; analyze risks beyond your control before accepting it. You may decline responsibility for an impossible situation.
- When you accept responsibility, expect to be held accountable. When you err, admit it and offer options, not excuses.
- Don't blame vendors, languages, management, or coworkers — provide solutions. No backup when the disk crashes is your fault.
- Before delivering bad news, rehearse it (tell the rubber duck or the cat first): does the excuse sound reasonable or stupid? Anticipate "have you tried X?" and try it first.
- Options instead of excuses: propose refactoring, prototyping, better testing, automation, or ask for resources/help.

> **Tip 3: Provide Options, Don't Make Lame Excuses**

### Software Entropy

Software rot follows the Broken Window Theory: one unrepaired broken window (bad design, wrong decision, poor code) signals nobody cares, and decay accelerates.

- Fix each broken window as soon as it is discovered.
- If there's no time to fix it properly, board it up: comment out offending code, show a "Not Implemented" message, substitute dummy data — visible action that prevents further damage.
- Neglect accelerates rot faster than any other factor.
- Converse (the firefighters who laid a mat before dragging hoses): in a pristine codebase people take extra care not to make the first mess — even during a raging deadline. Don't be the first to break a window.

> **Tip 4: Don't Live with Broken Windows**

### Stone Soup and Boiled Frogs

Two morals: be the catalyst that gets change started, and don't let gradual change cook you unnoticed.

- Stone soup tactic against "start-up fatigue": work out what you can reasonably ask for, build it well, show it, then say "of course, it would be better if we added..." — people find it easier to join an ongoing success.
- ("It's easier to ask forgiveness than it is to get permission." — Grace Hopper)
- Boiled-frog failure mode: disasters start too small to notice; overruns happen a day at a time; systems drift feature by feature; patch upon patch until nothing original remains. Unlike broken windows (people stop caring), the frog simply doesn't notice the change.
- Constantly review what's happening around you, not just what you personally are doing.

> **Tip 5: Be a Catalyst for Change**

> **Tip 6: Remember the Big Picture**

### Good-Enough Software

You can't produce perfect software, so discipline yourself to write software that's good enough — for users, future maintainers, and your own peace of mind. "Good enough" never means sloppy: systems must still meet users' requirements.

- Give users a say in deciding when the product is good enough; scope and quality belong in the requirements themselves.
- Domains differ: pacemakers and widely disseminated low-level libraries need near-perfection; a new product faces marketing promises, user schedules, and cash flow instead.
- Great software today is often preferable to perfect software tomorrow; early feedback leads to a better final solution.
- Know when to stop (programming is like painting): don't spoil a good program with overembellishment and overrefinement. It will never be perfect.

> **Tip 7: Make Quality a Requirements Issue**

### Your Knowledge Portfolio

Knowledge and experience are your most important professional assets — and they are expiring assets. Manage them like a financial portfolio:

- Invest regularly (habit matters more than amount); diversify (the more different things you know, the more valuable you are); manage risk (mix conservative standards with risky high-reward tech); buy low, sell high (learn emerging tech before it's popular); review and rebalance periodically.
- Concrete goals: learn at least one new language every year; read a technical book each quarter; read nontechnical books too; take classes; participate actively in user groups; experiment with different environments; stay current with journals; get wired (online sources).
- The learning process itself expands thinking, even if you never use the technology; cross-pollinate ideas into your current project.
- When you don't know an answer, take it as a personal challenge to find it — ask a guru, search, go to the library; always have something to read in dead moments.
- Think critically about what you read and hear; beware vendor/media hype and zealots — top billing can be bought.

> **Tip 8: Invest Regularly in Your Knowledge Portfolio**

> **Tip 9: Critically Analyze What You Read and Hear**

### Communicate!

A good idea is an orphan without effective communication; you're communicating only if you're conveying information.

Checklist (the book's summary):
- Know what you want to say — plan it, write an outline, refine until it gets the message across.
- Know your audience — their needs, interests, capabilities. WISDOM acrostic: What do you want them to learn? What is their Interest in what you've got to say? How Sophisticated are they? How much Detail do they want? Whom do you want to Own the information? How can you Motivate them to listen?
- Choose your moment — make it relevant in time; ask "Is this a good time to talk about...?"
- Choose a style — formal briefing vs. chat, report vs. memo; if in doubt, ask. Pushing back ("that needs pages, not a paragraph") is communication too.
- Make it look good — presentation matters; use style sheets, check spelling.
- Involve your audience — share early drafts; the process often matters more than the document.
- Be a listener — if you don't listen to them, they won't listen to you; turn meetings into dialog.
- Get back to people — always respond, even if just "I'll get back to you later."
- E-mail rules: proofread, check spelling, keep format simple (plain text), minimal quoting with attribution, don't flame, check recipients, archive it — e-mail is forever.

> **Tip 10: It's Both What You Say and the Way You Say It**

**Apply:**
- Own every outcome you commit to; when something breaks, bring options, never excuses.
- Fix or board up every broken window the day you find it.
- Ship something small and working to catalyze change instead of asking permission for the whole vision.
- Treat quality as an explicit requirement negotiated with users; stop polishing when it's good enough.
- Invest in learning weekly (new language yearly, technical book quarterly) and vet everything you read critically.
- Before communicating anything important, plan what to say, tailor it to the audience, and pick the moment.

## Ch 2 — A Pragmatic Approach

### The Evils of Duplication

Knowledge changes constantly, so maintenance is not a discrete phase but a routine part of all development. The DRY principle: EVERY PIECE OF KNOWLEDGE MUST HAVE A SINGLE, UNAMBIGUOUS, AUTHORITATIVE REPRESENTATION WITHIN A SYSTEM. Duplicated knowledge means changes must be repeated — and it's not whether you'll forget, it's when.

> **Tip 11: DRY—Don't Repeat Yourself**

The four categories of duplication:
- **Imposed** — the environment seems to require it. Counter with active code generation: build multiple representations (client/server structures, DB-schema classes, docs embedding code) from one metadata source at every build, never as a one-time conversion. Comments duplicating code are duplication: keep low-level knowledge in the code, reserve comments for high-level explanation (bad code requires lots of comments). Generate tests/docs from the authoritative source (e.g., acceptance tests generated from the spec document). Language-imposed duplication (C/C++ headers): document interface in headers, implementation details in source.
- **Inadvertent** — design mistakes: unnormalized data (driver stored on both Truck and DeliveryRoute — normalize per the business model), or mutually dependent fields (Line storing start, end, and length — make length computed). Caching may deliberately violate DRY for performance; localize the violation inside the class and expose only accessors. Always use accessor functions for object attributes (Meyer's Uniform Access).
- **Impatient** — copy-paste and literal constants because it seems easier now. "Shortcuts make for long delays" (Y2K). Easy to detect; takes discipline.
- **Interdeveloper** — hardest to detect; whole functionality duplicated across a team (10,000 programs each with its own SSN validation). Counter at high level with clear design, strong technical lead, well-understood division of responsibility; at module level with active frequent communication: forums, a project librarian, a central place for utility code, and reading each other's source.

> **Tip 12: Make It Easy to Reuse**

If reusing isn't easier than writing it yourself, people won't reuse.

### Orthogonality

Two things are orthogonal when changes in one do not affect the other (DB code vs. UI). Nonorthogonal systems (the helicopter: every control input has secondary effects) are inherently harder to change and control — no local fix exists.

> **Tip 13: Eliminate Effects Between Unrelated Things**

- Design self-contained components: independent, single well-defined purpose (cohesion). Benefits: productivity (localized changes, easier testing, reuse; combining orthogonal components multiplies functionality per unit effort) and reduced risk (diseased code is isolated, system less fragile, better tested, less vendor lock-in).
- Teams: measure orthogonality by how many people must be involved in discussing each change — the more, the less orthogonal. Separate infrastructure from application; give each component a subteam.
- Design test: "If I dramatically change the requirements behind a particular function, how many modules are affected?" Answer should be one. Also: don't rely on properties of things you can't control (e.g., phone number as customer ID).
- Layered architectures are a powerful orthogonal design; each layer uses only the abstractions below.
- Toolkits/libraries: ask whether adopting one imposes changes on code that shouldn't be there (transparent persistence is orthogonal; special object access is not). Declarative metadata (EJB transactions) and AOP keep cross-cutting concerns out of code.
- Coding techniques for maintaining orthogonality:
  - Keep code decoupled — write "shy" modules that don't reveal anything unnecessary and don't rely on other modules' implementations; if you need to change an object's state, get the object to do it (Law of Demeter).
  - Avoid global data — pass context explicitly (constructor parameters, context structs); beware Singletons used as globals.
  - Avoid similar functions — duplicated head/tail with different middles signals a structural problem (see Strategy pattern).
  - Refactor constantly; be perpetually critical of your code's structure.
- Testing as an orthogonality probe: if building a unit test drags in much of the system, the module isn't decoupled. At each bug fix, assess how localized the fix is; tag fixes in source control and trend the number of files touched per fix.
- Documentation is orthogonal too: content vs. presentation (style sheets).
- DRY minimizes duplication within a system; orthogonality minimizes interdependency between components. Use both together.

### Reversibility

Critical decisions aren't easily reversible, yet requirements, users, vendors, and platforms change faster than software gets developed. Prepare so decisions are written in sand, not stone.

- With every critical decision the project commits to a narrower target; hedge by keeping the code flexible: DRY, decoupling, metadata.
- Abstract third-party products behind well-defined interfaces (database as "persistence service" — then you can switch vendors mid-stream); deployment model (stand-alone vs. client-server vs. n-tier) should be a configuration change, not a rewrite.
- If a dependency must be sprinkled through code, express it in metadata and inject it with an automatic mechanism — whatever is added automatically can be removed automatically.

> **Tip 14: There Are No Final Decisions**

### Tracer Bullets

For new systems with vague requirements and many unknowns, get from requirement to some aspect of the final system quickly, visibly, and repeatably — like tracer fire that shows where you're actually hitting so you can adjust aim, instead of one big up-front calculation.

> **Tip 15: Use Tracer Bullets to Find the Target**

- Build a thin end-to-end skeleton first: every architectural component present and connected, each doing a trivial version of its job (e.g., one query flowing UI → library → server → SQL). Then flesh out all components in parallel.
- Tracer code is NOT disposable: written for keeps, with full error checking, structure, documentation, and self-checks — it is simply not fully functional yet.
- Advantages: users see something working early (and contribute feedback); developers get a structure to work in; you have an integration platform (integrate daily, not big-bang); you always have something to demonstrate; progress is measurable use case by use case (no "95% complete" monoliths).
- Misses are expected: the point is cheap, fast adjustment — a small body of code has low inertia.
- Tracer code vs. prototype: a prototype explores specific aspects and is thrown away and recoded properly; tracer code is lean but complete and forms part of the skeleton of the final system. Prototyping is the reconnaissance before the first tracer bullet is fired.

### Prototypes and Post-it Notes

Prototypes analyze and expose risk cheaply, targeting specific aspects. Their value is in the lessons learned, not the code.

> **Tip 16: Prototype to Learn**

- Prototype anything risky, unproven, critical, experimental, or uncomfortable: architecture, new functionality in an existing system, structure/contents of external data, third-party tools or components, performance, UI design.
- Prototypes need not be code: Post-it notes for workflow/application logic, whiteboard or paint-program mockups for UI.
- Details you may ignore in a prototype: correctness (dummy data), completeness (one input, one menu item), robustness (missing error checking, crashes allowed), style (little documentation).
- Use a high-level scripting language (Perl/Python/Tcl) and glue existing components; use GUI builders for interfaces.
- Architectural prototype questions: Are responsibilities of major components well defined and appropriate? Are collaborations well defined? Is coupling minimized? Potential duplication sources? Are interface definitions and constraints acceptable? Does every module have an access path to the data it needs, when it needs it? (The last yields the most surprises.)
- How NOT to use them: make sure everyone knows the code is disposable and cannot be completed — otherwise management may deploy the prototype. If the environment risks misinterpreting prototype code, use tracer bullets instead.

### Domain Languages

Program close to the problem domain: the language of the problem domain suggests the programming solution.

> **Tip 17: Program Close to the Problem Domain**

- Always write code using the domain vocabulary; when users produce well-bounded statements, consider a mini-language that expresses exactly what they want — first as a specification, possibly later executable.
- Benefits: new requirements become small edits at the domain level; domain-specific validation gives errors in users' vocabulary instead of "syntax error: undeclared identifier".
- Remember secondary users too — operations, config/test managers, support and maintenance programmers — each can get a mini-environment (e.g., screen-scraping scripts in the maintenance programmer's domain).
- Implementation options: line-oriented, easily parsed formats (switch statements or regexes) — the most common in practice; formal grammars via BNF plus a parser generator (yacc/bison, javaCC); or extending an existing language (embed Python etc.).
- Data languages produce data structures (config: sendmail.cf, Windows .rc files); imperative languages are executed, with statements and control constructs.
- Stand-alone vs. embedded: a specification language can generate artifacts used at build time (schema → SQL, C, web pages, XML); an embedded interpreted language changes application behavior without recompiling.
- Trade-off: complex, readable grammars cost more to implement but repay in extendibility and maintenance; simple grammars can be cryptic. Most applications outlive expectations — bias toward the more readable language.

### Estimating

Learn to estimate well enough to have an intuitive feel for feasibility and magnitudes.

> **Tip 18: Estimate to Avoid Surprises**

- All answers are estimates; first ask what accuracy the context requires. The units of the answer convey precision: quote 1–15 days in days, 3–8 weeks in weeks, 8–30 weeks in months, 30+ weeks — think hard before estimating at all ("about six months" ≠ "130 working days").
- Basic trick before modeling: ask someone who's already done it.
- Process: understand what's being asked (state scope in the answer: "assuming no traffic accidents..."); build a bare-bones mental model (model building may reveal a cheaper variant Y of the requested X); break the model into components with parameters; assign values, concentrating on the parameters that matter most (multipliers over addends); calculate, varying critical parameters (spreadsheet), and hedge the answer in terms of them.
- Strange answers with correct arithmetic mean your model or understanding is wrong — valuable information.
- Keep a log of estimates and check how you did; when one is wrong, find out why — that improves the next one.
- Project schedules: often only gained by experience on that same project. Iterate: check requirements, analyze risk, design/implement/integrate, validate with users; refine iteration counts after each increment. Don't nail down the whole plan up front.
- The single correct answer when asked for an estimate: "I'll get back to you." Estimates made at the coffee machine come back to haunt you.

> **Tip 19: Iterate the Schedule with the Code**

**Apply:**
- Give every piece of knowledge exactly one authoritative home; generate all derived forms automatically at build time.
- When you're tempted to copy-paste or hardcode a constant again, stop and extract it.
- For each module ask: if this requirement changes dramatically, how many modules must change? Refactor toward "one".
- Hide every third-party product and deployment decision behind an abstract interface.
- Start unfamiliar projects with a thin end-to-end tracer skeleton and grow it; use throwaway prototypes only to answer specific risk questions, and label them disposable.
- Never estimate on the spot: say "I'll get back to you", model it, quote units that match your confidence, and record the estimate for later review.

## Ch 3 — The Basic Tools

Tools amplify talent. Start with a basic, generally applicable toolset and let need drive acquisitions; don't confine yourself to one IDE's cozy interface.

### The Power of Plain Text

The base material of programming is knowledge, and the best persistent format for it is plain text: printable characters readable and understandable directly by people (structured formats like XML/HTML count).

> **Tip 20: Keep Knowledge in Plain Text**

- Human-readable ≠ human-understandable: `<SSNO>123-45-6789</SSNO>` beats `Field19=467abe`. Make it self-describing.
- Binary formats divorce data from its meaning — the context to parse them lives only in the application. Plain text can be a self-describing data stream that outlives the program that created it.
- Drawbacks (larger storage, costlier processing) rarely dominate; obscure binary is not more secure — encrypt passwords, checksum configs with a secure hash instead.
- Benefits: insurance against obsolescence (parse with only partial format knowledge); leverage (every tool operates on text — version control, diff, sum, editors, filters); easier testing (synthetic test data and regression outputs trivially manipulated with diff/Perl).
- The Unix philosophy — small sharp tools over a common line-oriented plain-text substrate; plain text is the lowest common denominator in heterogeneous environments.

### Shell Games

The command shell is the programmer's workbench: invoke and combine tools with pipes in ways their authors never dreamt of, and script whatever you do often.

> **Tip 21: Use the Power of Command Shells**

- GUI benefit is WYSIWYG; the disadvantage is WYSIAYG — what you see is all you get. GUIs cap you at what the designer intended; ad hoc queries and automation need the command line (e.g., `find . -name '*.c' -newer Makefile -print`; `find ... | xargs grep 'java.awt'`).
- Invest in one shell, learn it deeply, program it; on Windows use Cygwin or UWIN for Unix tools (mind case sensitivity, spaces, path separators).

### Power Editing

Text is the basic raw material; you must manipulate it effortlessly.

> **Tip 22: Use a Single Editor Well**

- Choose one editor, know it thoroughly, use it for all editing tasks (code, docs, memos, e-mail) — keystrokes become reflex, the editor becomes an extension of your hand. Prefer one available on all your platforms.
- Required features: configurable (fonts, colors, keybindings — keystrokes beat mouse), extensible (new languages/formats, compiler integration), programmable (macros or built-in scripting).
- Valuable language-aware features: syntax highlighting (mistyped keywords jump out), auto-completion, auto-indentation (enforces consistent project style), boilerplate templates, help tie-in, IDE-like compile/debug and error navigation.
- Proficiency test: single keystrokes to move by word/line/block/function; sort a region in place; if you use only basic features of many editors — pick one powerful editor and learn it well.

### Source Code Control

An SCCS is a giant project-wide UNDO key and time machine: every change to source and documentation tracked, any previous version recoverable.

> **Tip 23: Always Use Source Code Control**

- Always — even for a one-person, one-week project, a "throwaway" prototype, or non-source material. Put everything under it: documentation, phone lists, memos, makefiles, build/release procedures, scripts.
- Beyond undo: answers who changed what, diffs between versions, change volume per release, most-changed files — invaluable for bug-tracking, audit, performance, quality.
- Identify releases so any release can be regenerated exactly; manage branches (fix bugs on the release branch, develop on trunk, merge fixes across); central repository enables archiving; concurrent editing with merge works well in practice.
- Hidden benefit: automatic, repeatable product builds — pull latest source, build and run regression tests nightly, rebuild the source as of any date. No manual copy-to-build-area procedures.
- If the team won't use one, run your own private repository anyway and evangelize.

### Debugging

Debugging is just problem solving; attack it as such, without ego, denial, or finger-pointing.

> **Tip 24: Fix the Problem, Not the Blame**

> **Tip 25: Don't Panic**

- It doesn't matter whose fault the bug is — it's your problem. Step back, think about what could cause the symptoms; never waste a neuron on "that can't happen" — it did.
- Resist fixing symptoms: the fault is often several steps removed from what you observe; find the root cause.
- Before starting: compile clean at maximum warning levels — don't hand-hunt what the compiler could find.
- Gather data accurately; bug reports through third parties lose detail — watch the reporting user in action (the brush-stroke story: the programmer only ever tested strokes bottom-left to top-right). Test boundary conditions AND realistic end-user usage patterns.
- Make the bug reproducible — ideally with a single command, not 15 steps; isolating the reproduction often reveals the fix.
- Visualize your data: print variable=value, draw structures by hand, or use visualizing debuggers (DDD).
- Tracing ("got here", "x = 2") beats debuggers wherever time matters: concurrent processes, real-time systems, event-based applications. Keep trace messages in a regular, machine-parseable format (e.g., log opens/closes to find a resource leak).
- Corrupt variable? Examine the memory around it for a clue.
- Rubber ducking: explain the problem step by step to someone (or something) — verbalizing assumptions surfaces the error.
- Process of elimination: the bug is far more likely in your application code than in the OS, compiler, or a third-party library. "select is broken" is almost never true — if you see hoof prints, think horses, not zebras.
- If you "changed only one thing" and it broke, that thing is likely responsible. With no obvious start, binary-search the code/steps for where symptoms appear.
- Surprise bugs mean one or more of your cherished assumptions is wrong — don't gloss over a routine because you "know" it works.
- After the fix: ask why it wasn't caught earlier; amend tests; add earlier parameter checks; search for the same bug elsewhere; if fixing took long, build better hooks/log analyzers; if caused by a wrong assumption, discuss with the whole team.

> **Tip 26: "select" Isn't Broken**

> **Tip 27: Don't Assume It—Prove It**

Debugging checklist:
- Is the problem being reported a direct result of the underlying bug, or merely a symptom?
- Is the bug really in the compiler? The OS? Or in your code?
- If you explained this problem in detail to a coworker, what would you say?
- If the suspect code passes its unit tests, are the tests complete enough? What happens if you run the unit test with this data?
- Do the conditions that caused this bug exist anywhere else in the system?

### Text Manipulation

Text manipulation languages (Perl, Ruby, Python, Tcl, awk/sed) are the programmer's router: messy but powerful general-purpose transformers.

> **Tip 28: Learn a Text Manipulation Language**

- They hack up utilities and prototype ideas 5–10x faster than conventional languages — the multiplier that makes 30-minute experiments feasible.
- Proven uses: database schema maintenance (one schema file → SQL, data dictionary, C access libraries, integrity checks, web docs, XML); generating property accessors; knitting/converting test data; extracting tested code excerpts into documents (DRY); parsing C headers to generate another language's bindings as part of the build; generating web documentation.

### Code Generators

When you must repeat the same thing over and over, build a jig: write a code generator.

> **Tip 29: Write Code That Writes Code**

- **Passive** generators run once; output becomes an ordinary editable source file (new-file templates with boilerplate; one-off language conversions — they needn't be 100% accurate, fix the rest by hand; precomputed lookup tables). They save typing.
- **Active** generators run at every build from a single knowledge source; output is disposable. They are a necessity for DRY: one representation (e.g., a DB schema) converted into all needed forms (structs, classes) — schema changes then break compilation instead of production. Works only if generation is part of the build process.
- Use active generation whenever two disparate environments must share knowledge (database ↔ code; two languages sharing message structures — generate both from a language-neutral representation).
- Generators needn't be complex (a simple parser plus print statements) and needn't generate code — any text output (HTML, XML) counts.

**Apply:**
- Store every durable artifact — config, data, docs, metadata — as self-describing plain text.
- Automate any multi-step manual task as a shell script the second time you do it.
- Master one editor and one text-manipulation/scripting language deeply; use them everywhere.
- Put everything you type under version control and make builds fully automatic and repeatable from it.
- Debug systematically: reproduce with one command, read the data, trace over guessing, suspect your own code first, and prove every assumption.
- When the same knowledge must exist in two forms, generate one from the other at build time.

## Ch 4 — Pragmatic Paranoia

> **Tip 30: You Can't Write Perfect Software**

Accept it as an axiom and turn it into an advantage: drive defensively. Everyone else's code may not live up to your standards, and inputs may be invalid — so validate, assert, check consistency. Pragmatic Programmers go further: they don't trust themselves either, and code in defenses against their own mistakes.

### Design by Contract

DBC (Meyer, Eiffel): document and verify the rights and responsibilities of software modules to ensure correctness. A correct program does no more and no less than it claims to do.

- **Preconditions**: what must be true for the routine to be called — its requirements. A routine must never be called with violated preconditions; passing good data is the caller's responsibility.
- **Postconditions**: what the routine guarantees; the state of the world when it's done. Having a postcondition implies the routine terminates — no infinite loops.
- **Class invariants**: conditions always true from the caller's perspective; may be broken during internal processing but must hold whenever control returns to the caller. Never give unrestricted write access to data participating in the invariant.
- The contract: if the caller meets all preconditions, the routine guarantees all postconditions and invariants on completion. Failure by either side is a bug and triggers an agreed remedy (raise exception, terminate). Therefore never use preconditions for user-input validation.
- Write lazy code: be strict in what you accept before you begin, and promise as little as possible in return.
- Inheritance: contracts enforce the Liskov Substitution Principle ("Subclasses must be usable through the base class interface without the need for the user to know the difference"). Specify the contract once in the base class; a subclass may accept at least as much (or more) and guarantee at least as much (or more).
- Postconditions using passed parameters need unchanging parameters (final/`variable@pre`/Eiffel `old`); assertion conditions must be side-effect free.
- Even without language support, DBC pays as a design technique: enumerating input domain, boundary conditions, and what the routine promises — and doesn't promise — at design time is a huge leap; contracts as comments still help when trouble strikes. Plain assertions emulate DBC only partially (no inheritance propagation, no "old" values, unchecked runtime/libraries). Preprocessors (iContract for Java, Nana for C/C++) generate checking code.
- DBC and crashing early: an Eiffel `sqrt` with precondition `>= 0` reports "sqrt_arg_must_be_positive" with a stack trace at the call site, rather than propagating NaN to fail mysteriously later.
- Who checks? With language support, neither party — the runtime tests the precondition after invocation but before entry; explicit checking must be done by the caller, and the routine can be designed secure in the knowledge its input is in range.
- Loop invariants: a generalized statement of the loop's eventual goal, valid before the loop and on every iteration — kills fencepost/off-by-one errors; code as assertions or use as design/documentation.
- Semantic invariants: inviolate, requirement-driven laws central to a thing's meaning (debit-card switch: "ERR IN FAVOR OF THE CONSUMER" — never process a transaction twice); state them clearly, make them prominent, and don't confuse them with changeable policy.

> **Tip 31: Design with Contracts**

### Dead Programs Tell No Lies

All errors give you information. Don't rationalize that an error "can't happen" — if there is an error, something very, very bad has already happened, and your program is no longer viable.

> **Tip 32: Crash Early**

- Crash, don't trash: a dead program normally does a lot less damage than a crippled one that continues writing corrupted data to a vital database.
- Java's model: unexpected runtime problems throw RuntimeException and, uncaught, halt with a stack trace. Without exceptions, handle errors yourself, e.g., a C CHECK macro wrapping calls that should never fail and aborting with file/line/expected/got.
- Every case/switch needs a default clause — you want to know when the "impossible" happens.
- When immediate exit is inappropriate (resources held, logs, open transactions), release/tidy first — but the principle stands: once the impossible has happened, terminate as soon as possible.

### Assertive Programming

Don't practice the self-deception of "this can never happen." If it can't happen, check it anyway.

> **Tip 33: If It Can't Happen, Use Assertions to Ensure That It Won't**

- Whenever you think "of course that could never happen," add code to check it — assert null-pointer expectations, verify algorithm results (e.g., array actually sorted).
- Assertion conditions must be side-effect free (the `iter.nextElement()` inside an ASSERT skipping elements — a Heisenbug); never put must-execute code inside an assert (it may be compiled out).
- Assertions are not error handling: never assert on conditions that legitimately occur (user input).
- Your assert needn't call exit: it may raise an exception, longjmp, or call an error handler — but the dying-milliseconds code must not rely on the data that triggered the failure.
- Leave assertions turned on in production. The "turn them off after testing" argument wrongly assumes testing finds all bugs and forgets that the production world is dangerous. Turning them off is crossing a high wire without a net because you once made it across in practice. If a specific assertion has real performance cost, make only that one optional and keep the rest.

### When to Use Exceptions

Checking every possible error return breeds deeply nested, obscured code; exceptions move error handling to a single place and leave the normal flow of control clear. But exceptions must be reserved for unexpected events.

> **Tip 34: Use Exceptions for Exceptional Problems**

- Test: assume an uncaught exception terminates the program and ask "Will this code still run if I remove all the exception handlers?" If "no", exceptions are being used in nonexceptional circumstances.
- Example: opening /etc/passwd failing is exceptional (it should exist) — let FileNotFoundException propagate; opening a user-specified file that may not exist is not — check existence and return false.
- Why: an exception is an immediate, nonlocal transfer of control — a cascading goto; programs using exceptions for normal processing get spaghetti readability and tighter coupling between routines and callers (broken encapsulation).
- Error handlers (registered routines called for a category of errors) are an alternative or complement — useful in languages without exceptions, or to centralize handling (e.g., wrapping RMI objects in a non-remote class so clients register a handler instead of catching RemoteException everywhere).

### How to Balance Resources

Resource usage (memory, transactions, threads, files, timers, windows) follows allocate → use → deallocate. Have a consistent plan.

> **Tip 35: Finish What You Start**

- The routine or object that allocates a resource should be responsible for deallocating it. Anti-pattern: readCustomer opens a file into a global, writeCustomer closes it — a later conditional skips the close and production collapses with too many open files. Refactor so open and close live in the same routine, visibly balanced.
- Nested allocations: (1) deallocate in the opposite order of allocation, so you don't orphan resources that reference each other; (2) when allocating the same set of resources in different places, always allocate in the same order to prevent deadlock.
- OO: encapsulate a resource in a class — constructor acquires, destructor releases (scope-bound); helps especially where exceptions can interfere with deallocation.
- C++ exceptions: freeing in both the normal path and the catch violates DRY; instead use stack objects destroyed automatically on block exit, a wrapper class, or `auto_ptr` for dynamic objects.
- Java: garbage collection is lazy and finalize is unreliable; use try/finally — the finally clause runs however the try block exits (exception or return) and is the place to release/delete.
- When the pattern doesn't fit (dynamic structures linked into larger ones): establish a semantic invariant for who owns the data. On deallocating a top-level structure choose explicitly and implement consistently: (1) it recursively frees contained substructures; (2) it is simply deallocated, orphaning unreferenced contents; (3) it refuses to deallocate while it still contains substructures.
- Trust no one, including yourself: build wrappers that track allocations/deallocations and check the balance at logical points (e.g., top of a server's main request loop); use leak checkers (Purify, Insure++).

**Apply:**
- For every nontrivial routine, decide and document preconditions, postconditions, and invariants — strict about inputs, promising little.
- Validate the "impossible" with always-on assertions (side-effect free); never assert on legitimate runtime conditions.
- On detecting an impossible state, terminate as soon as cleanly possible — never limp on and trash data.
- Reserve exceptions for cases where the program couldn't proceed without the handler; use error returns for expected failures.
- Free every resource in the routine/object that acquired it; release in reverse order, acquire shared sets in one fixed order, and use finally/destructor-scope mechanisms.
- Add a default to every switch and a check to every "can't fail" call.

## Ch 5 — Bend, or Break

Write loose, flexible code so decisions stay reversible. The chapter's tools: minimize coupling between modules, move details out of code into metadata, break dependencies on time and ordering, separate models from views, and let modules exchange data anonymously via blackboards.

### Decoupling and the Law of Demeter

Write "shy" code: don't reveal yourself to others, don't interact with too many people. Organize code into cells (modules) and limit interaction between them, so a compromised module can be replaced without dragging others down.

- Symptom of trouble: traversing object relationships directly, e.g. `aSelection.getRecorder().getLocation().getTimeZone()`. This couples the caller to three classes; an unrelated change anywhere in that chain breaks you. Instead ask directly for what you need (`someSelection.getTimeZone()`) and let each object delegate internally.
- **Law of Demeter for functions** — any method of an object should call only methods belonging to:
  1. itself
  2. any parameters that were passed in to the method
  3. any objects it created
  4. any directly held component objects
- Warning signs of dependency explosion: link commands longer than the test program; "simple" changes propagating through unrelated modules; developers afraid to change code because they don't know what will be affected.
- Evidence: classes with large response sets (number of functions directly invoked by the class's methods) are more error-prone. Demeter shrinks the response set.
- Cost: you write many small wrapper/forwarding methods (runtime and space overhead). Like denormalizing a DB schema for speed, you may deliberately reverse Demeter and couple modules for performance — acceptable only if the coupling is known and agreed.
- Physical decoupling matters too (Lakos): manage dependencies among files, directories, and libraries; avoid cyclic dependencies; prefer forward declarations (`class Date;`) over `#include` in headers so builds stay fast.

> **Tip 36: Minimize Coupling Between Modules**

### Metaprogramming

Details — especially frequently changing ones — mess up code; every edit risks a new bug. Get details out of the code entirely: program for the general case and put the specifics in metadata, outside the compiled code base.

- Make systems highly configurable — not just colors and prompts but choice of algorithms, database products, middleware, UI style. These should be configuration options, not engineering.
- Metadata = any data that describes the application: how it should run, what resources it should use. Access it at runtime, not compile time. Represent it as plain text (key/value files, property files) or an embedded scripting/mini-language for more power.
- Benefits: forces a decoupled, more abstract design; customize without recompiling (including emergency workarounds in production); metadata can be closer to the problem domain; one engine can drive several products with different metadata.
- Business policy and rules change more than anything else — keep them in the most flexible format: config values (e.g. supplier payment terms), a mini-language, or a rules engine.
- When to reload config: long-running servers should be able to reread and apply metadata while running; a quick-restarting GUI app may only need startup reads.
- Example: EJB deployment descriptors specify transaction/thread/load-balancing behavior as metadata, keeping bean code free of that machinery.
- Without metadata your code can't adapt — "dodo-code" goes extinct.

> **Tip 37: Configure, Don't Integrate**

> **Tip 38: Put Abstractions in Code, Details in Metadata**

### Temporal Coupling

Time is a design element: concurrency (things at the same time) and ordering (relative position in time). Linear "do this then always that" thinking produces temporal coupling — method A must be called before B, only one report at a time — which is inflexible and unrealistic.

- Workflow: during requirements analysis, capture user workflow with UML activity diagrams (actions, arrows, synchronization bars). Find what can happen simultaneously versus what must be strictly ordered; use the diagram to maximize parallelism among activities that could run in parallel but don't.
- Architecture: design independent concurrent components communicating via work queues (e.g. an OLTP pipeline of input tasks → app servers → database handler, all asynchronous). The hungry consumer model — independent consumer tasks pulling from a shared work queue — gives cheap load balancing; a bogged-down task doesn't stall the others.
- You have created services: independent, concurrent objects behind well-defined, consistent interfaces.
- Design for concurrency even if you don't deploy it: protect global/static variables (and ask why they're global); ensure objects are in a valid state whenever they could be called — beware separate constructor+initialize routines; class invariants help. Concurrency constraints fight programming by coincidence.
- Cleaner interfaces result: C's `strtok` (hidden static state, call-order dependency, can't parse two strings at once) versus Java's `StringTokenizer` (per-instance state, thread-safe, no surprises).
- Deployment: with concurrency designed in, you can deploy standalone, client-server, or n-tier as a configuration choice. Retrofitting concurrency into a nonconcurrent app is much harder.

> **Tip 39: Analyze Workflow to Improve Concurrency**

> **Tip 40: Design Using Services**

> **Tip 41: Always Design for Concurrency**

### It's Just a View

Separate the data (model) from every interpretation of it (views), and synchronize with events instead of hard-wired knowledge, so modules only "hear what they want to hear."

- Event = a message saying "something interesting just happened." The sender needs no explicit knowledge of receivers; multiple receivers each pursue their own agenda.
- Don't route all events through one routine (violates encapsulation, increases coupling, breeds giant case statements). Use publish/subscribe: objects register for exactly the events they need and are never sent events they don't need. Variants: peer-to-peer, a centralized "software bus" dispatching to listeners, broadcast for critical events (cf. CORBA Event Service push/pull modes).
- MVC: Model — abstract data, no knowledge of views or controllers. View — an interpretation of the model (a subset, not necessarily graphical); subscribes to model changes and controller events. Controller — controls the view and feeds the model new data; publishes events to both.
- Payoff: multiple views of one model, common viewers over many models, multiple controllers; one of the cheapest ways to maintain reversibility. Example: Java's JTree — supply any `TreeModel` and the widget works; change rendering via `TreeCellRenderer` without touching other code.
- Views can themselves become models for higher-level viewers (baseball-reporting network: score/stat/trivia viewers feed a scheduler, which feeds teleprompter/caption/Web formatters). Networks of models and viewers; add debugging views to inspect models cheaply.
- Residual coupling: publishers and subscribers still share interface definitions — blackboards remove even that.

> **Tip 42: Separate Views from Models**

### Blackboards

A blackboard is a shared space where producers and consumers of knowledge exchange data anonymously and asynchronously — like detectives posting facts on a case board. Nobody needs to know who else exists; participants may come and go; anything may be posted.

- Modern implementations: JavaSpaces / T Spaces (tuple spaces, from Linda). Operations: `read` (search and retrieve), `write` (put an item in), `take` (read + remove), `notify` (callback when a matching object is written). Retrieval by partial match of fields (templates/wildcards) or by subtype; atomic operations and distributed transactions.
- You can store live objects, not just data, so algorithms can be designed as a flow of objects.
- One consistent interface to the blackboard replaces a combinatorial explosion of unique inter-module APIs.
- Ideal fit: loan/mortgage processing — no guaranteed data-arrival order, distributed contributors, asynchronous feeds, data dependencies, new data triggering new rules. Blackboard + rules engine: posting a fact triggers applicable rules; rule output posts back and triggers more.
- Partition large blackboards into zones/interest groups or hierarchies when they get cluttered.

> **Tip 43: Use Blackboards to Coordinate Workflow**

**Apply:**
- In any method, only call methods on yourself, your parameters, objects you created, or your direct components — never chain through returned objects.
- Add a delegating method rather than reaching two objects deep; hide structural knowledge behind the owning object's interface.
- Move volatile details (business rules, tunable choices, environment specifics) into plain-text config or a mini-language; code the general case.
- Never depend on call ordering or hidden shared state; keep objects valid at every observable moment and design APIs as if they'll be called concurrently.
- Model your domain data once and attach independent views/listeners via publish/subscribe instead of pushing updates by hand.

## Ch 6 — While You Are Coding

Coding is not mechanical transcription of design; it demands continuous decisions. Developers who don't actively think about their code are programming by coincidence.

### Programming by Coincidence

Don't rely on luck and accidental successes (the soldier who probes a minefield, finds nothing, and marches confidently to his death). If you don't know why your code works, you won't know why it fails.

- Accidents of implementation: relying on undocumented error/boundary behavior, calling routines in the wrong order or context because it "seems to work" (Fred's `paint(); invalidate(); validate(); revalidate(); repaint(); paintImmediately();`). Reasons not to leave it alone: it may not really work; the boundary condition is accidental; undocumented behavior changes with the next release; extra calls slow the code and add bug risk.
- Accidents of context: assuming a GUI exists, English-speaking users, a tty, writable local disk — things your environment happens to provide but doesn't guarantee.
- Implicit assumptions: rarely documented, often conflicting between developers; testing is especially prone to false causality — don't assume it, prove it.
- For code others call: good modularization, small well-documented interfaces, explicit contracts. For routines you call: rely only on documented behavior; if you must rely on more, document the assumption.
- **How to Program Deliberately** (checklist):
  - Stay aware of what you're doing.
  - Don't code blindfolded (unfamiliar tech or unclear requirements invite coincidence).
  - Proceed from a plan.
  - Rely only on reliable things; if unsure, assume the worst.
  - Document your assumptions (Design by Contract).
  - Test assumptions as well as code (write assertions).
  - Prioritize your effort — spend time on the important/hard parts first.
  - Don't be a slave to history: don't let existing code dictate future code; be ready to refactor.

> **Tip 44: Don't Program by Coincidence**

### Algorithm Speed

Estimate the resources (time, memory) algorithms use as a routine habit — whenever you write loops or recursion, check that the runtime is sensible for the input sizes you'll face.

- Big-O gives a worst-case upper bound as input size n grows; drop low-order terms and constant factors (so a fast O(n²) may still beat a slow O(n²) — the notation won't tell you).
- Common orders: O(1) constant (array access); O(lg n) logarithmic (binary search); O(n) linear (sequential search); O(n lg n) (average quicksort, heapsort); O(n²) (selection/insertion sort); O(n³) (naive matrix multiply); O(2ⁿ) exponential (traveling salesman, set partitioning).
- Common-sense estimation: simple loop over n → O(n); nested loops → O(n×m), typically O(n²) for simple sorts; halving the working set each iteration ("binary chop") → O(lg n); partition + work on halves + combine → O(n lg n); anything over permutations → factorial/exponential, use heuristics.
- Ask how big n can get. If bounded, you know the cost; if it depends on external factors (overnight batch size, list of people), consider what large values do to runtime and memory.
- Test in practice: run with varying input sizes and plot the curve (three or four points reveal the shape); use profilers to count step executions. Theory misses practical effects: thrashing when memory runs out, sort routines degrading on presorted input. The only timing that counts is real code, production environment, real data.
- Best isn't always best: for small inputs a simple insertion sort beats writing and debugging a quicksort; watch out for high setup costs; and confirm something is a genuine bottleneck before optimizing (beware premature optimization).

> **Tip 45: Estimate the Order of Your Algorithms**

> **Tip 46: Test Your Estimates**

### Refactoring

Software is gardening, not building construction — organic, continuously tended. Rewriting, reworking, and re-architecting code is refactoring, and it's a normal, necessary activity, not a failure.

- **When to refactor** — as soon as anything strikes you as "wrong":
  - Duplication (DRY violation)
  - Nonorthogonal design
  - Outdated knowledge (requirements drifted, your understanding grew)
  - Performance (functionality needs to move to improve it)
- Time pressure is not an excuse: fail to refactor now and the fix later costs far more (more dependencies). Medical analogy for the boss: it's a growth — removing it now is cheap; waiting makes surgery more expensive and dangerous. If you can't refactor immediately, put it on the schedule and warn users of the affected code.
- How to refactor without harm (Fowler):
  1. Don't refactor and add functionality at the same time.
  2. Have good tests before you begin; run them as often as possible.
  3. Take short, deliberate steps (move a field, fuse two similar methods); test after each step to avoid prolonged debugging.
- Make incompatible changes break the build so all old clients surface immediately.
- Fix it and everything that depends on it — don't live with broken windows.

> **Tip 47: Refactor Early, Refactor Often**

### Code That's Easy to Test

Build testability into software from the start (like chips designed with Built-In Self Test and Test Access Mechanisms) and test each piece thoroughly before wiring pieces together.

- Unit test = code that exercises a module in isolation: establish an artificial environment, invoke routines, check results against known values or previous runs (regression).
- Test against contract: write cases that verify the unit honors its contract — this tells you both whether the code meets the contract and whether the contract means what you think. Cover a wide range of cases and boundary conditions (e.g. sqrt: negative arg rejected, zero boundary accepted, results within epsilon across the range).
- For dependent modules, test subcomponents' contracts first (LinkedList in full, Sort in full, then module A that uses them). Then a failure in A's test points at A, not its parts — big debugging savings.
- Design to test: when you design a module, design its contract and the code to test that contract; writing tests before implementation lets you try out the interface before committing to it.
- Locate unit tests conveniently (in the module or a nearby subdirectory) — if tests aren't easy to find, they won't be used. Accessible test code also gives users examples of usage and a regression base. Shipping tests in the product (e.g. per-class `main`, `#ifdef __TEST__`) enables field diagnostics.
- Test harness capabilities: standard setup/cleanup; selection of individual or all tests; analysis of output for expected results; standardized failure reporting. Tests should be composable (suites of subtests to any depth); see xUnit/JUnit.
- Formalize ad hoc tests: any test improvised during debugging goes into the permanent unit test — if code broke once, it will break again.
- Build a test window for deployed software: consistently formatted log files (parseable), a hot-key diagnostic status window, or a built-in HTTP server exposing internal status, log entries, and a debug panel.
- Testing is more cultural than technical (Perl's `make test` standard): instill a testing culture — tests in a standard place with expected output — regardless of language.

> **Tip 48: Design to Test**

> **Tip 49: Test Your Software, or Your Users Will**

### Evil Wizards

Wizards that generate reams of code you don't understand leave you programming by coincidence: the generated code is interwoven with yours, you own it, and you can't maintain or debug what you don't understand.

- Wizards are fine as a starting point only if you understand every line they produce; unlike libraries or OS services, wizard output is not factored behind a tidy interface — it becomes your application.
- No one should produce code they don't fully understand.

> **Tip 50: Don't Use Wizard Code You Don't Understand**

**Apply:**
- Before shipping code that "works," be able to say why it works; delete lucky-looking calls and replace assumptions with assertions or documented contracts.
- Rely only on documented behavior of libraries; when you can't, write the assumption down next to the code.
- For every loop/recursion, note its big-O and the realistic bound on n; measure with real data before and after optimizing.
- Refactor in small tested steps the moment you spot duplication or misfit — never mixed with feature work, never without tests.
- Write the test (against the contract, including boundaries) with or before the code, keep it runnable with one command, and promote every debugging probe into the suite.
- Understand any generated code line-by-line before committing it.

## Ch 7 — Before the Project

Establish ground rules before the project starts: dig out real requirements, identify true constraints, know when to start, avoid the specification spiral, and keep methodologies in their place.

### The Requirements Pit

Requirements aren't gathered from the surface — they're buried beneath layers of assumptions, misconceptions, and politics. A requirement is a statement of something that needs to be accomplished.

- Distinguish requirements from business policy. "Only an employee's supervisors and personnel may view that employee's records" embeds policy. Make the requirement general ("Only authorized users may access an employee record"), document the policy separately, and hyperlink the two; the developer then builds an access-control system and policy changes become metadata updates, not code changes. Requirements stated this way naturally lead to metadata-driven systems.
- Requirements vs. UI: "The system must let you choose a loan term" is a requirement; "we need a list box" is only a requirement if users absolutely must have a list box. Discover the underlying reason users do a thing, not just the way they currently do it; document the why — it informs daily implementation decisions. (Cautionary tale: Brian Eno's ultimate mixing board failed because its interface ignored engineers' existing tactile skills — sometimes the interface is the system; successful tools adapt to the hands that use them.)
- Become a user: work a week on the help desk or in the warehouse. It reveals how the system will really be used and builds trust with users.
- Document with use cases (goal-driven, per Cockburn's template): characteristic information (goal in context, scope, level, preconditions, success/failed end conditions, primary actor, trigger), main success scenario, extensions, variations, related information (priority, performance target, frequency, super/subordinate use cases, actors and channels), schedule, open issues. The template captures nonfunctional requirements and user comments, and doubles as a meeting agenda; use cases nest hierarchically. Don't be a slave to notation (stick-figure diagrams can't carry this density); use whatever communicates with your audience.
- Don't overspecify: good requirements documents remain abstract — the simplest statement that accurately reflects the business need. Requirements are not architecture, not design, not UI. Requirements are need.
- See further, via abstraction not prophecy: Y2K came from failing to see beyond current business practice plus DRY violations — a DATE abstraction should have owned the two-digit representation.
- Manage scope creep: track requirements — who requested, who approved, count of approved requests — and show each new feature's schedule impact to sponsors; "just one more feature" becomes visibly the fifteenth this month.
- Maintain a project glossary — one place defining all specific terms and vocabulary; everyone from end users to support staff uses it. Projects fail when people call the same thing by different names or different things by the same name.
- Publish requirements as a hypertext document on an internal Web site: sponsors read the high level, programmers drill down — better than a two-inch binder nobody reads.

> **Tip 51: Don't Gather Requirements—Dig for Them**

> **Tip 52: Work with a User to Think Like a User**

> **Tip 53: Abstractions Live Longer than Details**

> **Tip 54: Use a Project Glossary**

### Solving Impossible Puzzles

The secret to "impossible" problems (Gordian knots) is to identify the real constraints — as opposed to preconceived notions — and find a solution within them.

- Absolute constraints must be honored, however distasteful; but many apparent constraints aren't real. "Thinking outside the box" is misleading: the trick is to find the box — the true boundary of constraints — which may be far larger than you assume.
- Method: enumerate all possible avenues, however stupid-sounding; then go through the list and prove why each path can't be taken. Can you prove it? (The Trojan horse: "through the front door" was surely dismissed as suicide.)
- Categorize and prioritize constraints; like woodworkers cutting the longest pieces first, identify the most restrictive constraints first and fit the rest within them.
- **Cutting the Gordian Knot** — when a problem seems much harder than it should be, ask:
  - Is there an easier way?
  - Am I solving the right problem, or distracted by a peripheral technicality?
  - Why is this thing a problem?
  - What is making it so hard to solve?
  - Does it have to be done this way?
  - Does it have to be done at all?
- Often a reinterpretation of the requirements makes a whole set of problems vanish.

> **Tip 55: Don't Think Outside the Box—Find the Box**

### Not Until You're Ready

Great performers know when to start and when to wait. A nagging doubt when you sit down to start is accumulated experience talking — heed it; given time it will crystallize into something addressable.

- Distinguish good judgment from procrastination by prototyping the area that worries you:
  - If you quickly feel bored, your reluctance was probably just fear of starting — drop the prototype and do the real development.
  - If the prototype triggers a revelation that a basic premise was wrong, you'll see how to fix it — the doubt was justified and you've saved the team wasted effort.
- Remember while prototyping why you're doing it — don't drift into serious development on prototype code.

> **Tip 56: Listen to Nagging Doubts—Start When You're Ready**

### The Specification Trap

Specification reduces a requirement to the point where a programmer's skill can take over — it's communication. But ever-more-detailed specs hit diminishing, then negative, returns.

- Why exhaustive specs fail: no spec captures every detail and nuance (and users don't know exactly what they need until they see a running system); natural language can't express operations precisely (try writing instructions for tying shoelaces); and the straightjacket effect — a design leaving the coder no interpretation room robs the effort of skill and art, and hides options that only become apparent during coding.
- Treat requirements gathering, design, and implementation as facets of one process — delivery of a quality system; let specification and implementation flow into each other with feedback from implementation and testing. Distrust environments where they happen in isolation.
- Incredibly detailed specs are legitimately demanded for contracts, life-critical systems, and published interfaces/libraries.
- Beware specs layered on specs with no supporting implementation or prototype — it's easy to specify the unbuildable. Don't let specifications become security blankets: at some point, start coding; if the team is wrapped up in specs, break them out with prototyping or tracer-bullet development.

> **Tip 57: Some Things Are Better Done than Described**

### Circles and Arrows

Formal methods and methodologies (CASE, waterfall, spiral, ER, UML, ...) are tools, not masters. Blindly adopting any technique without putting it in the context of your practices and capabilities is a recipe for disappointment.

- Shortcomings: diagrams are meaningless to end users, so there's no real user checking — only the designers' interpretation (prefer showing users a prototype they can play with); formal methods encourage specialization and us-vs-them between designers and coders — prefer understanding the whole system; most methods model static relationships, but dynamic metadata-driven systems need relationships knitted together at runtime.
- Research (Glass): the hype around every method (4GLs, CASE, formal methods, OO, ...) was overblown; benefits appear only after a significant productivity/quality dip during adoption. Never underestimate the cost of adopting new tools; treat first projects with a new technique as a learning experience.
- Use formal methods when analysis says you need them — but remember who's in charge. Extract the best from each methodology and meld it into your own continuously improving set of working practices. Don't bow to the false authority of a method or a tool's price tag: an acre of class diagrams is still a fallible interpretation. "The class diagram is the application, the rest is mechanical coding" marks a doomed project.

> **Tip 58: Don't Be a Slave to Formal Methods**

> **Tip 59: Expensive Tools Do Not Produce Better Designs**

**Apply:**
- For each stated requirement, ask whether it's a need, a policy, or a UI preference; keep the need abstract and push policy toward metadata.
- Record why users need each feature, keep a shared glossary, and track every scope change with its schedule impact.
- When stuck, list all avenues and force yourself to prove each "impossible" one is truly barred; question whether the task must be done this way, or at all.
- Resolve nagging doubts by prototyping the scary part; let boredom or revelation decide.
- Prototype instead of writing another layer of spec; start coding when further description adds nothing.

## Ch 8 — Pragmatic Projects

Project-scale versions of the same philosophy: pragmatic teams, automation of every procedure, project-wide testing, documentation as a first-class product, and managing what success looks like.

### Pragmatic Teams

Pragmatic techniques multiply when the whole team applies them. Recast the individual practices at team level.

- No broken windows: quality is a team issue; a "quality officer" role is ridiculous — quality comes only from every member's individual contributions.
- Boiled frogs: teams get boiled more easily than individuals (everyone assumes someone else is handling it). Appoint a chief water tester to constantly check for increased scope, decreased timescales, added features, new environments — anything not in the original agreement; keep metrics on new requirements. You needn't reject change, just be aware of it.
- Communicate as one entity: great teams have a distinct personality, prepared meetings, crisp consistent documents — one voice externally (lively debate internally). Marketing trick: brand the project — a name, a zany logo used on memos and reports gives identity and something memorable to associate with the work.
- DRY at team scale: appoint a project librarian to coordinate documentation and repositories and to spot impending duplication; or appoint focal points per functional aspect (date handling → Mary, DB schema → Fred). Use groupware/newsgroups to archive Q&A.
- Organize around functionality, not job functions: analysis/design/coding/testing are views of the same problem and can't happen in isolation. Split into small, cohesive, largely self-contained teams, each owning a functional aspect (including infrastructure like DB access layer or help subsystem); teams self-organize internally with agreed commitments to each other. Apply the same criteria used to modularize code — contracts, decoupling, orthogonality — to isolate teams from change (change DB vendor → only the database team is hit). Warning sign: two subteams working on the same module/class. Ownership increases commitment.
- Requires responsible developers and strong leadership: at least two "heads" — technical (development philosophy and style, assigns responsibilities, arbitrates, watches for commonality that reduces orthogonality) and administrative (project manager: resources, progress reporting, business priorities, external ambassador). Larger projects add a librarian, tool builders, operational support.
- Automation: appoint tool builders to construct makefiles, shell scripts, editor templates, utilities that automate project drudgery.
- Know when to stop adding paint: give individuals just enough structure to shine and deliver; don't over-manage.

> **Tip 60: Organize Around Functionality, Not Job Functions**

### Ubiquitous Automation

Whatever recurs — build, release, review paperwork — must be automatic. Manual procedures leave consistency to chance; repeatability isn't guaranteed when steps are open to interpretation. People aren't as repeatable as computers, nor should they be.

- Scripts (shell/batch) execute the same instructions in the same order every time and go under source control so procedure changes are auditable. Cautionary tale: many-page manual IDE-install instructions produced subtly different developer machines and machine-specific bugs.
- Schedule unattended tasks with cron/at: nightly builds, backups, report generation, Web site maintenance.
- Compile via makefiles (even inside IDEs): scripted, automatic, with hooks for code generation and regression tests — checkout, build, test, ship with a single command. Use make dependency rules to regenerate derived files (e.g. XML → Java → class) so DRY holds. Beware recursive makefiles: each invocation can't see the others' dependencies, so you get needless work or missed rebuilds; build and test dependencies may need separate hierarchies.
- A build takes an empty directory plus a known environment and builds the project from scratch to the final deliverable: (1) check out from the repository; (2) build from scratch, stamped with a version; (3) create a distributable image with ship-exact layout, permissions, docs, README; (4) run specified tests. Run it nightly with the full test suite so a regression is caught close to the code change that caused it. Final (ship) builds differ (locked/tagged repository, different optimization flags) — use a separate target like `make final`; anything compiled differently must be fully retested.
- Automate administrivia with content-driven workflow: generate the project Web site from the repository (docs extracted from code, requirements, design docs, nightly build results, test and performance reports, coding metrics) published automatically as a nightly-build step or check-in hook — the Web view is just a view (DRY). Automate approval workflows, e.g. a `/* Status: needs_review */` marker collected by a script that posts lists, mails reviewers, changes status to `reviewed` after Web sign-off. (Glass: code inspection works; review meetings don't.)
- Don't be the cobbler's barefoot children: use cron, make, scripting languages to build your own tools; let the computer do the repetitious and mundane.

> **Tip 61: Don't Use Manual Procedures**

### Ruthless Testing

Find your bugs now, so you don't endure others finding them later. Fishing metaphor: unit tests are fine nets for minnows, integration tests are coarse nets for sharks; patch the net whenever a fish escapes.

- Test as soon as code exists. Teams using automated tests succeed far more than teams with shelved test plans. "Code a little, test a little" — write test code with (or before) production code; a good project may have more test code than production code, and it's cheaper in the long run.
- **What to test** (aspects of testing):
  - Unit testing — foundation of everything else; all modules must pass their own tests first.
  - Integration testing — do the subsystems honor their contracts with each other? Often the single largest source of bugs; an extension of unit testing.
  - Validation and verification — users told you what they wanted, but is it what they need? A bug-free answer to the wrong question is useless; mind real end-user access patterns.
  - Resource exhaustion, errors, and recovery — behavior at real-world limits: memory, disk space, CPU bandwidth, wall-clock time, disk bandwidth, network bandwidth, color palette, video resolution. When it fails, does it fail gracefully, saving state and work?
  - Performance testing — stress/load with realistic volumes of users, connections, transactions per second; is it scalable? May need specialized hardware/software to simulate load.
  - Usability testing — with real users under real conditions; does the software fit the user like an extension of the hand? Failure to meet usability criteria is as big a bug as dividing by zero.
- Design/methodology testing via metrics: cyclomatic complexity, inheritance fan-in/fan-out, response set, class coupling ratios; compare each module against the population — outliers without a good excuse indicate potential problems.
- **How to test**: regression tests (compare current output against previous/known values — run every kind of test as a regression); test data — both real-world data (reveals requirement misunderstandings; "typical" surprises you) and synthetic data (volume, boundary conditions like Feb 29 / huge records / foreign postal codes, statistical properties like every-third-failure or presorted input); GUI testing — use tooling but chiefly decouple app logic from the GUI so logic is testable without it; nondeterministic output may need manual interpretation; test the tests — deliberately cause the bug and make sure the test complains; consider a project saboteur who plants bugs in a copy of the tree to verify the tests catch them; test thoroughly — coverage tools help, but hitting every line isn't the point: the number of program states matters (a three-line function on two 0–999 ints has a million states), and traversal order can matter most of all.
- **When to test**: as soon as any production code exists; automatically (results interpreted automatically too); as frequently as possible and always before check-in (some SCCS like Aegis enforce it). Tests needing special setup (stress) run on a regular schedule with resources allocated.
- Tightening the net — the single most important testing concept: if a bug slips through existing tests, add a new test to trap it next time. Once a human finds a bug, no human should ever find it again — no exceptions, no matter how trivial or how loudly someone claims it can't recur.

> **Tip 62: Test Early. Test Often. Test Automatically.**

> **Tip 63: Coding Ain't Done 'Til All the Tests Run**

> **Tip 64: Use Saboteurs to Test Your Testing**

> **Tip 65: Test State Coverage, Not Code Coverage**

> **Tip 66: Find Bugs Once**

### It's All Writing

Documentation is integral to development, not an afterthought. Code and documentation are two views of the same model; apply all the pragmatic principles (DRY, orthogonality, MVC, automation) to documents. All documentation mirrors the code — when they disagree, the code is what matters.

- Comments should say why — purpose, goal, engineering trade-offs, decisions made, alternatives discarded. The code already shows how; commenting the how violates DRY. Reasonable level: module-level header, comments on significant data/type declarations, brief per-class and per-method headers describing use and non-obvious behavior (JavaDoc level for parameters).
- Names: meaningful, spelled out (`connectionPool` not `cp`); no `foo`/`doit`/`manager`/`stuff`; Hungarian notation is inappropriate in OO systems; misleading names are worse than meaningless ones (`getData` that writes to disk — the Stroop effect: names are deeply meaningful to the brain).
- Keep out of source comments: exported-function lists, revision history (that's what SCCS is for), lists of files used, the file's own name — all better derived by tools. Do include the code's owner/author: responsibility keeps people honest.
- Executable documents: when the same information exists in several forms (spec, SQL schema, record structure), pick one authoritative source (the model) and generate the other views automatically (markup + scripts; or make the document subordinate to another representation and import on every print). The only way to change the schema becomes changing the document; spec, schema, and code cannot disagree.
- Generate API documentation from source with JavaDoc/DOC++ — the source is the model, printed/Web docs are views.
- Technical writers should honor the same principles: DRY, orthogonality, model-view, automation and scripting — don't throw material over the wall.
- Publish online, hyperlinked, with a date stamp or version number on each page; paper is a snapshot that's out of date at printing. Keep presentation independent of content (markup like DocBook/HTML + XSL/CSS, or word-processor styles) so one source yields report, slides, online help, Web pages.
- Documentation built along with the nightly build, under source control — never a second-class citizen.

> **Tip 67: Treat English as Just Another Programming Language**

> **Tip 68: Build Documentation In, Don't Bolt It On**

### Great Expectations

A project's success is measured by how well it meets its users' expectations, not by the deliverable in absolute terms — falling below expectations is failure, and (like the child expecting a cheap doll) wildly overshooting them can be too.

- Communicate expectations throughout development: work with users toward a common understanding of the process and deliverable, including expectations they haven't verbalized — don't "manage" (control) their hopes, and never lose sight of the business problem. Tracer bullets and prototypes are the key techniques: both give users something visible and train mutual communication.
- Go the extra mile: surprise—delight—users with slightly more than they expected. Cheap delighters: balloon/ToolTip help, keyboard shortcuts, a quick-reference guide, colorization, log file analyzers, automated installation, system-integrity checking tools, running multiple versions for training, a splash screen customized for their organization. Superficial, not feature bloat — each says the team cared. Just don't break the system adding them.

> **Tip 69: Gently Exceed Your Users' Expectations**

### Pride and Prejudice

Pragmatic Programmers accept responsibility and do work they can be proud of. Sign your work, as craftsmen of an earlier age did.

- Ownership must not become territoriality or prejudice for your code and against coworkers: treat others' code with respect (Golden Rule); mutual respect is critical. Communal ownership (XP) works too, backed by practices like pair programming that guard against anonymity's dangers.
- Anonymity breeds sloppiness, mistakes, sloth, and bad code — being a cog with lame status-report excuses. Your signature should be recognized as an indicator of quality: solid, well written, tested, documented.

> **Tip 70: Sign Your Work**

**Apply:**
- Script every recurring procedure (build, test, publish, review bookkeeping) and put the scripts under source control; if it's manual, it's broken.
- Keep a one-command from-scratch build that checks out, builds, tests, and packages; run it nightly with the full suite.
- Never check in without running the tests; when any bug escapes, add a test that would have caught it before fixing it.
- Test states and boundaries, not lines: feed both real-world and synthetic data, and sabotage the code occasionally to prove the tests notice.
- Comment the why with an owner's name; generate all derived docs and views from one authoritative source.
- Watch for scope drift explicitly, keep users' expectations aligned as you go, then deliver a small delighter on top.

## The 70 Tips (complete list)

1. **Care About Your Craft** — Why spend your life developing software unless you care about doing it well?
2. **Think! About Your Work** — Turn off the autopilot and take control. Constantly critique and appraise your work.
3. **Provide Options, Don't Make Lame Excuses** — Instead of excuses, provide options. Don't say it can't be done; explain what can be done.
4. **Don't Live with Broken Windows** — Fix bad designs, wrong decisions, and poor code when you see them.
5. **Be a Catalyst for Change** — You can't force change on people. Instead, show them how the future might be and help them participate in creating it.
6. **Remember the Big Picture** — Don't get so engrossed in the details that you forget to check what's happening around you.
7. **Make Quality a Requirements Issue** — Involve your users in determining the project's real quality requirements.
8. **Invest Regularly in Your Knowledge Portfolio** — Make learning a habit.
9. **Critically Analyze What You Read and Hear** — Don't be swayed by vendors, media hype, or dogma. Analyze information in terms of you and your project.
10. **It's Both What You Say and the Way You Say It** — There's no point in having great ideas if you don't communicate them effectively.
11. **DRY—Don't Repeat Yourself** — Every piece of knowledge must have a single, unambiguous, authoritative representation within a system.
12. **Make It Easy to Reuse** — If it's easy to reuse, people will. Create an environment that supports reuse.
13. **Eliminate Effects Between Unrelated Things** — Design components that are self-contained, independent, and have a single, well-defined purpose.
14. **There Are No Final Decisions** — No decision is cast in stone. Instead, consider each as being written in the sand at the beach, and plan for change.
15. **Use Tracer Bullets to Find the Target** — Tracer bullets let you home in on your target by trying things and seeing how close they land.
16. **Prototype to Learn** — Prototyping is a learning experience. Its value lies not in the code you produce, but in the lessons you learn.
17. **Program Close to the Problem Domain** — Design and code in your user's language.
18. **Estimate to Avoid Surprises** — Estimate before you start. You'll spot potential problems up front.
19. **Iterate the Schedule with the Code** — Use experience you gain as you implement to refine the project time scales.
20. **Keep Knowledge in Plain Text** — Plain text won't become obsolete. It helps leverage your work and simplifies debugging and testing.
21. **Use the Power of Command Shells** — Use the shell when graphical user interfaces don't cut it.
22. **Use a Single Editor Well** — The editor should be an extension of your hand; make sure your editor is configurable, extensible, and programmable.
23. **Always Use Source Code Control** — Source code control is a time machine for your work—you can go back.
24. **Fix the Problem, Not the Blame** — It doesn't really matter whether the bug is your fault or someone else's—it is still your problem, and it still needs to be fixed.
25. **Don't Panic When Debugging** — Take a deep breath and THINK! about what could be causing the bug.
26. **"select" Isn't Broken** — It is rare to find a bug in the OS or the compiler, or even a third-party product or library. The bug is most likely in the application.
27. **Don't Assume It—Prove It** — Prove your assumptions in the actual environment—with real data and boundary conditions.
28. **Learn a Text Manipulation Language** — You spend a large part of each day working with text. Why not have the computer do some of it for you?
29. **Write Code That Writes Code** — Code generators increase your productivity and help avoid duplication.
30. **You Can't Write Perfect Software** — Software can't be perfect. Protect your code and users from the inevitable errors.
31. **Design with Contracts** — Use contracts to document and verify that code does no more and no less than it claims to do.
32. **Crash Early** — A dead program normally does a lot less damage than a crippled one.
33. **Use Assertions to Prevent the Impossible** — Assertions validate your assumptions. Use them to protect your code from an uncertain world.
34. **Use Exceptions for Exceptional Problems** — Exceptions can suffer from all the readability and maintainability problems of classic spaghetti code. Reserve exceptions for exceptional things.
35. **Finish What You Start** — Where possible, the routine or object that allocates a resource should be responsible for deallocating it.
36. **Minimize Coupling Between Modules** — Avoid coupling by writing "shy" code and applying the Law of Demeter.
37. **Configure, Don't Integrate** — Implement technology choices for an application as configuration options, not through integration or engineering.
38. **Put Abstractions in Code, Details in Metadata** — Program for the general case, and put the specifics outside the compiled code base.
39. **Analyze Workflow to Improve Concurrency** — Exploit concurrency in your user's workflow.
40. **Design Using Services** — Design in terms of services—independent, concurrent objects behind well-defined, consistent interfaces.
41. **Always Design for Concurrency** — Allow for concurrency, and you'll design cleaner interfaces with fewer assumptions.
42. **Separate Views from Models** — Gain flexibility at low cost by designing your application in terms of models and views.
43. **Use Blackboards to Coordinate Workflow** — Use blackboards to coordinate disparate facts and agents, while maintaining independence and isolation among participants.
44. **Don't Program by Coincidence** — Rely only on reliable things. Beware of accidental complexity, and don't confuse a happy coincidence with a purposeful plan.
45. **Estimate the Order of Your Algorithms** — Get a feel for how long things are likely to take before you write code.
46. **Test Your Estimates** — Mathematical analysis of algorithms doesn't tell you everything. Try timing your code in its target environment.
47. **Refactor Early, Refactor Often** — Just as you might weed and rearrange a garden, rewrite, rework, and re-architect code when it needs it. Fix the root of the problem.
48. **Design to Test** — Start thinking about testing before you write a line of code.
49. **Test Your Software, or Your Users Will** — Test ruthlessly. Don't make your users find bugs for you.
50. **Don't Use Wizard Code You Don't Understand** — Wizards can generate reams of code. Make sure you understand all of it before you incorporate it into your project.
51. **Don't Gather Requirements—Dig for Them** — Requirements rarely lie on the surface. They're buried deep beneath layers of assumptions, misconceptions, and politics.
52. **Work with a User to Think Like a User** — It's the best way to gain insight into how the system will really be used.
53. **Abstractions Live Longer than Details** — Invest in the abstraction, not the implementation. Abstractions can survive the barrage of changes from different implementations and new technologies.
54. **Use a Project Glossary** — Create and maintain a single source of all the specific terms and vocabulary for a project.
55. **Don't Think Outside the Box—Find the Box** — When faced with an impossible problem, identify the real constraints. Ask yourself: "Does it have to be done this way? Does it have to be done at all?"
56. **Start When You're Ready** — You've been building experience all your life. Don't ignore niggling doubts.
57. **Some Things Are Better Done than Described** — Don't fall into the specification spiral—at some point you need to start coding.
58. **Don't Be a Slave to Formal Methods** — Don't blindly adopt any technique without putting it into the context of your development practices and capabilities.
59. **Costly Tools Don't Produce Better Designs** — Beware of vendor hype, industry dogma, and the aura of the price tag. Judge tools on their merits.
60. **Organize Teams Around Functionality** — Don't separate designers from coders, testers from data modelers. Build teams the way you build code.
61. **Don't Use Manual Procedures** — A shell script or batch file will execute the same instructions, in the same order, time after time.
62. **Test Early. Test Often. Test Automatically.** — Tests that run with every build are much more effective than test plans that sit on a shelf.
63. **Coding Ain't Done 'Til All the Tests Run** — 'Nuff said.
64. **Use Saboteurs to Test Your Testing** — Introduce bugs on purpose in a separate copy of the source to verify that testing will catch them.
65. **Test State Coverage, Not Code Coverage** — Identify and test significant program states. Just testing lines of code isn't enough.
66. **Find Bugs Once** — Once a human tester finds a bug, it should be the last time a human tester finds that bug. Automatic tests should check for it from then on.
67. **English is Just a Programming Language** — Write documents as you would write code: honor the DRY principle, use metadata, MVC, automatic generation, and so on.
68. **Build Documentation In, Don't Bolt It On** — Documentation created separately from code is less likely to be correct and up to date.
69. **Gently Exceed Your Users' Expectations** — Come to understand your users' expectations, then deliver just that little bit more.
70. **Sign Your Work** — Craftsmen of an earlier age were proud to sign their work. You should be, too.
