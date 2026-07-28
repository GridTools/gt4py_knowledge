---
title: A Philosophy of Software Design — John Ousterhout (2nd ed., 2021)
description: "Book notes: complexity is the enemy of software design; fight it with deep modules, information hiding, and a strategic mindset."
tags: [software-design, complexity, modularity, deep-modules, abstraction, information-hiding, interfaces, comments, naming, error-handling, code-review, book-notes]
---

The greatest limitation in writing software is our ability to understand the systems we create; therefore complexity is the single enemy of software design, and the whole discipline reduces to minimizing it. Complexity is anything about a system's structure that makes it hard to understand and modify, and it is caused by dependencies and obscurity. It accumulates incrementally through hundreds of small decisions, so it must be fought continuously with a strategic (investment) mindset rather than a tactical get-it-working mindset. The main structural weapon is the deep module: a unit with a simple interface hiding a powerful implementation, so that developers face only a small fraction of total complexity at any time. Design is never done — software is developed incrementally, and every modification is an opportunity to improve or degrade the design.

## Ch 1 — Introduction (It's All About Complexity)

Since software has no physical limits, the binding constraint is human understanding; the goal of design is to keep systems simple enough to understand and evolve. There are two ways to fight complexity: eliminate it (simpler, more obvious code, fewer special cases) and encapsulate it (modular design, so no one faces it all at once).

- Waterfall fails for software because a large system cannot be fully visualized before building; problems surface mid-implementation when the design is frozen, and patching around them explodes complexity.
- Incremental development works because software is malleable: design, implement, and evaluate a small subset, fix design problems while the system is small, repeat. Consequence: design is never done, and continuous redesign is part of the job.
- Design skill is learned by recognizing red flags — signs that code is more complicated than necessary — and trying alternative designs until the red flag disappears.
- Every principle has limits; taking any design idea to its extreme leads somewhere bad. Good designs balance competing ideas.

**Apply:**
- Treat every task as a design task; plan to spend part of your time improving existing design.
- When a design looks complicated, try a different approach and compare, rather than pushing through.
- When you see a red flag, stop and search for an alternative design that removes it — don't just note it.

## Ch 2 — The Nature of Complexity

Complexity is anything related to the structure of a software system that makes it hard to understand and modify. It is felt by readers, not writers: if others find your code complex, it is complex.

- Cost view: in a complex system even small improvements take a lot of work; in a simple system larger improvements take less effort. Overall complexity is each part's complexity weighted by how often developers touch that part — isolating complexity where it is never seen is almost as good as eliminating it.
- Three symptoms:
  - **Change amplification** — a seemingly simple change requires modifications in many places (e.g., a banner color hard-coded in every page of a website vs. one shared variable).
  - **Cognitive load** — how much a developer must know to complete a task (e.g., a C function that returns allocated memory the caller must remember to free). More lines of code can be *simpler* if they lower cognitive load; line count is not a complexity metric.
  - **Unknown unknowns** — it is not obvious which code must be modified or what information is needed (e.g., a few pages also hard-code a darker emphasis shade derived from the banner color — nothing points you to them). This is the worst symptom: the only "cure" is reading every line.
- Two causes:
  - **Dependencies** — code that cannot be understood or modified in isolation. Dependencies can't be eliminated (every interface creates them); the goal is fewer, simpler, more obvious dependencies.
  - **Obscurity** — important information that is not obvious: vague names, missing units, undocumented invariants, inconsistency. Needing extensive documentation is itself a red flag that the design isn't right.
- Complexity is incremental: it accumulates in lots of small chunks, each individually defensible; once accumulated it is very hard to remove. This demands zero tolerance.

**Apply:**
- Judge every change by whether it adds dependencies or obscurity, however small.
- Prefer designs that turn many hidden dependencies into one obvious one.
- Optimize for the reader; if a reviewer says code is complex or nonobvious, believe them.
- Aim for an obvious system: a developer can quickly guess what to do and be confident the guess is right.

## Ch 3 — Working Code Isn't Enough (Strategic vs. Tactical Programming)

Tactical programming — getting the current feature working as fast as possible — makes good design impossible, because every task adds a bit of complexity that compounds. Strategic programming makes a great design the primary goal (which also happens to work) and treats design as continuous investment.

- Tactical shortcuts feel individually reasonable; collectively they turn code into a mess that takes months to clean up, so no one ever cleans it up. Beware the "tactical tornado": prolific programmers who pump out working code and leave destruction for others to clean up.
- Most code is written by extending existing code, so the most important job of a developer is to facilitate future extensions.
- Invest 10–20% of development time continuously in design: proactively (try alternative designs, imagine future changes, write good docs) and reactively (when a design problem surfaces, fix it, don't patch around it). Payback estimate: 6–18 months; after that the investment is free.
- Technical debt is worse than financial debt: you pay back more than you borrowed, and most of it is never fully repaid.
- Startup pressure doesn't excuse tactical work: spaghetti code is nearly impossible to fix later, payoff for good design comes quickly, and messy code bases repel the best engineers (Facebook "move fast and break things" vs. Google/VMware strategic cultures — both can succeed, but working strategically is more fun).

**Apply:**
- Never accept "it works" as the finish line; the finish line is "the design is good and it works."
- Budget a fixed slice (10–20%) of every task for design improvement; do it today, not after the crunch — there is always another crunch.
- When you find a design problem, fix it now rather than patching around it.

## Ch 4 — Modules Should Be Deep

Modular design divides a system into modules (classes, subsystems, services) whose interfaces are much simpler than their implementations. The best modules are deep: lots of functionality behind a simple interface.

- A module = interface + implementation. The interface is everything a developer in another module must know to use it — formal parts (signatures, types) plus informal parts (behavior, usage constraints), which are usually larger and can only be described in comments.
- An abstraction is a simplified view of an entity that omits unimportant details. It fails two ways: including unimportant details (extra cognitive load) or omitting important ones (obscurity — a **false abstraction** that looks simple but isn't).
- Depth is cost/benefit: the benefit of a module is its functionality; its cost to the system is its interface. Interfaces are good, but more or larger interfaces are not better.
- Canonical deep interface: Unix file I/O — five system calls (`open`, `read`, `write`, `lseek`, `close`) hiding hundreds of thousands of lines (on-disk layout, permissions, caching, scheduling, device drivers), unchanged for decades. A garbage collector is even deeper: it has no interface at all and *shrinks* the system's interface.
- Shallow module: interface complexity ≈ implementation complexity (a linked-list class; a one-line `addNullValueForAttribute` method whose documentation is longer than its code).
- **Classitis**: the mistaken view that "classes are good, so more classes are better." Many small classes are individually simple but their interfaces accumulate into system-level complexity plus boilerplate. Example: Java file I/O needing three chained objects (`FileInputStream` + `BufferedInputStream` + `ObjectInputStream`), where forgetting the buffering wrapper silently kills performance.
- Make the common case simple: if an interface has many features but most users need only a few, its effective complexity is only that of the commonly used features (Unix defaults to sequential I/O; `lseek` exists but sequential users never see it).

🚩 **Red Flag: Shallow Module** — a shallow module is one whose interface is complicated relative to the functionality it provides; the benefit (not having to learn its internals) is negated by the cost of learning its interface. Small modules tend to be shallow.

**Apply:**
- Design each class/method to hide as much functionality as possible behind the simplest possible interface.
- Don't split code into more classes/methods just to make each one small; judge decomposition by total system complexity.
- Provide the common case by default (e.g., buffering); make rare options invisible to those who don't need them.
- Reject new interface elements that don't eliminate more complexity than they add.

## Ch 5 — Information Hiding (and Leakage)

The main technique for creating deep modules is information hiding (Parnas): each module encapsulates a few design decisions — data structures, algorithms, formats, assumptions — that appear nowhere in its interface. Hiding simplifies the interface and lets the decision change without touching other modules.

- Declaring variables `private` is not information hiding if getters/setters expose them anyway.
- Partial hiding also helps: features needed by only a few users should be accessed through separate methods so the common case doesn't see them.
- **Information leakage**: a design decision reflected in multiple modules (e.g., two classes both knowing a file format). Back-door leakage (not via interfaces) is more pernicious because it's invisible. Fix by merging the affected classes or by extracting the shared knowledge into one class with a genuinely abstract interface.
- **Temporal decomposition**: structuring code by the runtime order of operations ("read, then parse, then write") instead of by knowledge. The same knowledge is needed at multiple points in time, so it leaks. Design modules around pieces of knowledge, not around execution order.
- HTTP-server lessons: reading and parsing a request belong in one class (you can't find the end of a request without parsing it). Returning the internal `Map` of parameters (`getParams()`) is shallow and exposes the representation; `getParameter(name)` / `getIntParameter(name)` hide it and do more work per call. Making a class slightly larger often improves hiding: bring all code for a capability together and raise the level of the interface.
- Defaults are partial information hiding: the class should "do the right thing" without being asked (auto-fill HTTP response version and Date). The best features are the ones you get without even knowing they exist.
- Don't hide information that is genuinely needed outside the module (e.g., tunable performance parameters a caller must set) — but first try to make it not needed (auto-tuning).

🚩 **Red Flag: Information Leakage** — information leakage occurs when the same knowledge is used in multiple places, such as two different classes that both understand the format of a particular type of file.

🚩 **Red Flag: Temporal Decomposition** — in temporal decomposition, execution order is reflected in the code structure: operations that happen at different times are in different methods or classes. If the same knowledge is used at different points in execution, it gets encoded in multiple places, resulting in information leakage.

🚩 **Red Flag: Overexposure** — if the API for a commonly used feature forces users to learn about other features that are rarely used, this increases the cognitive load on users who don't need the rarely used features.

**Apply:**
- When designing a module, list the knowledge it encapsulates; if a decision shows up in two modules, restructure.
- Decompose around knowledge, never around the order operations happen at runtime.
- Never return or accept internal data structures across an interface.
- Give every option a sensible default; ask whether callers could actually pick a better value than the module can.
- Also hide information *within* a class: minimize the number of places each instance variable is used.

## Ch 6 — General-Purpose Modules are Deeper

Over-specialization may be the single greatest cause of complexity; general-purpose interfaces are simpler, deeper, need less code, and hide information better — even when used for only one purpose.

- Sweet spot: **somewhat general-purpose** — the module's functionality reflects today's needs, but its interface does not; the interface supports multiple uses without being hard to use for the current one.
- Editor text-class example: special-purpose methods `backspace(cursor)`, `delete(cursor)`, `deleteSelection(sel)` leaked UI concepts into the text class, produced many shallow one-caller methods, and hid information the UI actually needed (a false abstraction). The general API — `insert(position, text)`, `delete(start, end)`, `changePosition(position, numChars)` — is smaller, obvious, and reusable; backspace becomes `text.delete(text.changePosition(cursor, -1), cursor)`.
- Three questions to find the balance: (1) What is the simplest interface that covers all my current needs? (fewer methods without losing capability, so long as each stays simple). (2) In how many situations will this method be used? (a method designed for one call site is too special-purpose). (3) Is this API easy to use for my current needs? (if you must write lots of wrapper code, it's too general/low-level — e.g., single-character insert/delete would force loops everywhere).
- Push specialization **up** (application-specific behavior into top-level classes, as with the UI code) or **down** (device drivers implement a general block-read/block-write interface so device quirks never reach the OS core).
- Editor undo example: extract the general mechanism (a `History` class managing a list of `History.Action` objects with `undo`/`redo` and fences for grouping) from the special-purpose actions (text insert/delete implemented by the text class, selection/cursor actions by the UI). Each layer works without understanding the others.
- Eliminate special cases in code: design the normal case so edge cases fall out automatically — represent "no selection" as an empty selection, so selection code needs no existence checks.

**Apply:**
- Design interfaces for the underlying capability, not for the specific feature that prompted them.
- Count call sites: a public method used at exactly one place is suspect.
- Replace clusters of special-purpose methods with one general method when each signature stays simple.
- Separate general-purpose mechanism from special-purpose policy; push specialization toward the application top or the driver bottom.
- Choose representations in which edge cases are handled by the normal-case code (empty ranges, zero counts).

## Ch 7 — Different Layer, Different Abstraction

In a well-designed system each layer provides a different abstraction from the layers above and below (file → block cache → device driver; TCP byte stream → best-effort packets). Adjacent layers with similar abstractions signal a bad class decomposition.

- **Pass-through methods** do nothing but invoke a method with a similar signature (a student `TextDocument` class where 13 of 15 public methods just forwarded to `TextArea`). They add interface without functionality and mean responsibility is split confusingly. Fixes: let callers use the lower class directly, redistribute functionality, or merge the classes. The interface to a piece of functionality should be in the same class that implements it.
- Same-signature methods are fine when each adds distinct value: dispatchers (choose which method handles a request) and multiple implementations of one interface (device drivers) — the latter actually reduce cognitive load.
- **Decorators** (wrappers) invite shallow classes full of pass-throughs (Java `BufferedInputStream`, `ScrollableWindow`). Before writing one, ask: could the feature go in the underlying class (buffering belongs in file I/O), in the use case, in an existing decorator, or in an independent class? Wrappers are justified mainly to adapt an unmodifiable external interface.
- Interface should differ from implementation: a text class stored as lines but exposing `getLine`/`putLine` forced every caller to split and join lines; a character/range-oriented interface over line-based storage encapsulates that work — the difference between interface and internals is the class's value.
- **Pass-through variables** travel down long call chains through methods that don't use them (a `cert` argument threaded from `main` to a socket call). Best available fix: a **context object** holding all per-instance global state, stored as an instance field in major objects and passed only to constructors. Contexts have global-variable drawbacks (nonobvious dependencies, thread-safety), so keep contents immutable and disciplined — but they beat the alternatives and keep multiple system instances possible in one process (vital for tests).

🚩 **Red Flag: Pass-Through Method** — a pass-through method is one that does nothing except pass its arguments to another method, usually with the same API as the pass-through method. This typically indicates that there is not a clean division of responsibility between the classes.

**Apply:**
- When two adjacent classes expose similar signatures, redraw the responsibility boundary or merge them.
- Put a feature's interface in the class that implements the feature.
- Prefer adding functionality to the underlying class over writing a decorator.
- Make a class's interface abstraction different from its internal representation.
- Replace variables threaded through call chains with a context object passed via constructors.

## Ch 8 — Pull Complexity Downwards

When a module faces unavoidable complexity, handle it inside the module rather than exporting it to users: modules have more users than developers, and a simple interface matters more than a simple implementation.

- Punting is tempting — throw an exception, add a configuration parameter — but it amplifies complexity: every caller/administrator now deals with the problem instead of one implementer.
- Text-class example: a line-oriented interface was easy to implement but forced all higher-level code to split/join lines; a character/range interface moved that complexity down into one place.
- Configuration parameters move complexity upward and often can't be set well by users anyway; they also go stale. A transport protocol can *compute* its retry interval from measured response times — better than any hand-set value, and self-adjusting. Before exporting a parameter ask: "will users be able to determine a better value than we can determine here?" If you must have one, give it a sensible default.
- Limits: pull complexity down only if (a) it is closely related to the module's existing functionality, (b) it simplifies things elsewhere in the application, and (c) it simplifies the module's interface. (Pulling backspace-key knowledge into the text class fails these tests — that's leakage, not depth.)

**Apply:**
- When you hit something hard, solve it inside the module instead of exposing it through the interface.
- Treat each configuration parameter as a design failure to justify; compute values automatically where possible, default them otherwise.
- Accept a more complex implementation whenever it buys a simpler interface.

## Ch 9 — Better Together Or Better Apart?

Combine or separate pieces of functionality to minimize overall complexity, not component size. Subdividing adds its own complexity: more components to track, more interfaces, more management code, separation of related code, and duplication.

- Bring code together when it is closely related: it shares information (both depend on a document syntax); it is used together *bidirectionally*; it overlaps conceptually under one category (string manipulation); one piece can't be understood without the other.
- Bring together if information is shared (the HTTP read+parse methods both needed the request format; merging made code shorter and simpler).
- Bring together if it simplifies the interface (merged modules can eliminate intermediate interfaces and do things automatically — combined FileInputStream+buffering would make buffering invisible).
- Bring together to eliminate duplication (factor repeated snippets into one method if it has a simple signature; or restructure so the snippet executes in one place — a `goto`-to-cleanup error path in C is legitimate for escaping nested code).
- Separate general-purpose from special-purpose code (the text class provides general text operations; UI operations like "delete selection" live in the UI layer).
- Cursor/selection counter-example: merging them into one object helped no caller and complicated the implementation; separate `Position`-based objects were simpler — combine only what is truly related.
- Logging counter-example: hoisting one-line log statements into a separate logger class with one method per error site added interfaces and forced readers to flip between files; log at the point of detection.
- Method splitting: length alone is not a reason to split; developers break methods up too much, producing extra interfaces and conjoined pieces. Each method should do one thing and do it completely. Good splits either (a) factor out a cleanly separable, ideally reusable subtask (readable without knowing the parent, and vice versa), or (b) split one over-broad interface into two simpler ones that most callers use singly. If callers must invoke both pieces, or readers flip back and forth, the split failed.
- Against Clean Code's "functions should be tiny": below a few dozen lines, further shrinking doesn't help readability but multiplies interfaces and conjoined functions. Depth beats length: first make functions deep, then make them short enough to read easily — never sacrifice depth for length.

🚩 **Red Flag: Repetition** — if the same piece of code (or code that is almost the same) appears over and over again, that's a red flag that you haven't found the right abstractions.

🚩 **Red Flag: Special-General Mixture** — this red flag occurs when a general-purpose mechanism also contains code specialized for a particular use of that mechanism. This creates information leakage between the mechanism and the particular use case: future modifications to the use case are likely to require changes to the underlying mechanism as well.

🚩 **Red Flag: Conjoined Methods** — it should be possible to understand each method independently. If you can't understand the implementation of one method without also understanding the implementation of another, that's a red flag.

**Apply:**
- Decide split-vs-join by total complexity: best information hiding, fewest dependencies, deepest interfaces.
- Merge code that shares knowledge or is always used together; keep independent things apart.
- Don't split a method by line count; split only to create a cleanly separable subtask or genuinely simpler interfaces.
- If understanding one piece requires reading another, put them back together.

## Ch 10 — Define Errors Out Of Existence

Exceptions are a disproportionate source of complexity: handlers are hard to write, hard to test, rarely executed, and breed secondary exceptions. Reduce the number of places exceptions must be handled — ideally to zero — by redefining semantics.

- Exceptions thrown by a class are part of its interface; many exceptions = shallow class. An exception propagates up the stack, so it complicates not just the caller but higher levels too. Throwing is easy; handling is hard. Over-defensive "detect and report everything" styles multiply complexity; >90% of catastrophic failures in distributed data-intensive systems were caused by incorrect error handling.
- Technique 1 — **Define errors out of existence**: change the operation's definition so the "error" case is normal behavior. Tcl `unset` should "ensure the variable no longer exists" (no error when absent) rather than "delete the variable." Unix file deletion vs. Windows: Unix marks an in-use file for deletion and lets existing users keep reading/writing — two whole error classes vanish. Java `substring` should just return the overlapping characters for out-of-range indices (as Python slices do) — simpler API, more functionality, deeper method. Fewer errors defined ⇒ simpler software; the best bug-reduction strategy is simplification, not more error checks.
- Technique 2 — **Mask exceptions**: detect and handle the condition at a low level so higher levels never see it (TCP retransmits lost packets; NFS hangs and retries rather than surfacing server outages, because no application could do anything better). Masking pulls complexity downward and deepens the class.
- Technique 3 — **Exception aggregation**: handle many exceptions with one handler high up rather than one handler per call site (web server: let `NoSuchParameter` and friends propagate to the top-level dispatcher, which turns any such exception's message into an error response; new methods plug in with no new handlers). Also "error promotion": RAMCloud handles a corrupted object by crashing the server and using the already-necessary crash-recovery path — one general mechanism replaces many special-purpose ones (fine because corruption is rare; don't promote frequent errors).
- Technique 4 — **Just crash**: for errors that are rare and essentially unhandleable (out of memory, disk hard errors, internal inconsistencies), print diagnostics and abort (`ckalloc` wrapper around `malloc`). What's crash-worthy depends on the application — a replicated storage system must recover from I/O errors.
- Taking it too far: only define away/mask information that genuinely isn't needed outside the module. A network module that swallowed all errors made robust applications impossible. Decide what's important: hide what isn't; expose what is.

**Apply:**
- Before adding a throw, try to redefine the API so the situation is normal behavior with well-defined semantics.
- Handle what you can at the lowest level (mask); let the rest propagate to one aggregate handler near the top of the request loop.
- Distinguish request-aborting exceptions from system-fatal ones; catch the former in exactly one place.
- Crash with a clear message on rare, unhandleable errors instead of threading them through every caller.
- Never swallow errors that callers need for correctness.

## Ch 11 — Design it Twice

Your first design idea is unlikely to be the best one: for every major design decision, sketch at least two radically different alternatives before committing.

- Rough out each alternative's most important methods only; then list pros and cons. The most important criterion for an interface is ease of use for higher-level software; also compare interface simplicity, generality, and enabled implementation efficiency (text class: line-oriented vs. character-oriented vs. range-oriented APIs — the comparison exposes that both losers push text manipulation upward, pointing to the range API).
- Even when one alternative seems clearly right, designing a second teaches you why; if all alternatives are bad, their shared problems drive a new design.
- Apply at every level: interface first, then implementation (for implementations the goals are simplicity and performance), also subsystem decomposition. Costs an hour or two for a class — trivial against the implementation time it saves.
- Smart-people trap: those whose first idea was always good enough never learned to consider a second; hard problems make that habit fail. Trying multiple designs is not an admission of weakness — the problems are just hard. The comparison process itself builds design skill.

**Apply:**
- For every new class/module interface, write down two or more meaningfully different designs and compare pros/cons before coding.
- Judge interface alternatives primarily by ease of use for callers.
- Design interface and implementation as separate design-it-twice exercises.

## Ch 12 — Why Write Comments? The Four Excuses

Comments are essential to abstraction — without them you cannot hide complexity — and writing them well improves design. The four excuses for not commenting all fail.

- "Good code is self-documenting": false. Only signatures are expressible in code; a method's high-level behavior, meaning of results, side effects, preconditions, rationale, invariants are not. Expecting users to read implementations destroys the abstraction (all complexity exposed) and pressures you into hordes of tiny shallow methods.
- "No time": comments cost at most ~10% of development time and pay for themselves in maintainability; abstraction comments pay immediately as design tools.
- "They get stale": keeping comments current is cheap if documentation is not duplicated and lives next to the code; code reviews catch drift.
- "All comments I've seen are worthless": most comments are indeed mediocre, but writing solid ones is a learnable skill.
- What comments are for: capturing information that was in the designer's mind but can't be represented in code — from hardware quirks up to class rationale. Good comments attack cognitive load and unknown unknowns, and clarify dependencies while filling obscurity gaps.
- Against Clean Code's "comments are failures": replacing a comment with a method named `isLeastRelevantMultipleOfNextLargerPrimeFactor` conveys less than a sentence of English; comments and code carry different kinds of information, and both are needed.

**Apply:**
- Comment what cannot be expressed in code: abstractions, rationale, constraints, units, invariants.
- Treat missing interface comments as missing abstraction, not missing polish.
- Don't contort code (tiny methods, mega-names) to avoid writing prose.

## Ch 13 — Comments Should Describe Things that Aren't Obvious from the Code

The guiding principle: comments describe what isn't obvious from the code — either more precise than the code or more abstract than the code, never at the same level.

- Pick conventions (Javadoc/Doxygen/godoc style). Four comment categories: **interface** (before class/method declarations — the abstraction), **data structure member** (each field), **implementation** (inside methods — what/why), **cross-module** (dependencies spanning modules). The first two matter most: comment every class, every method, every instance variable; it's easier to comment everything than to argue about exceptions.
- Test for a useless comment: could someone write it just by looking at the adjacent code (or by rewording the name)? Then it adds nothing. First step to a good comment: use different words than the name, adding meaning (e.g., `textHorizontalPadding`: "blank space to leave on the left and right sides of each line of text, in pixels").
- **Lower-level comments add precision** — best for variable declarations: units, inclusive/exclusive bounds, meaning of null, ownership of resources, invariants. Document what a variable *represents* (nouns), not how it is manipulated (verbs).
- **Higher-level comments enhance intuition** — best inside methods and for interfaces: state overall intent ("Try to append the current key hash onto an existing RPC to the desired server that hasn't been sent yet") so readers can judge whether the code is correct; "how we get here" comments explain why code executes.
- Interface comments: class comment gives the overall abstraction, what an instance represents, and limitations; method comment gives behavior as perceived by callers, each argument and return value (precisely), side effects, exceptions, preconditions. If interface comments must describe the implementation, the class is shallow.
- Implementation comments: what and why, not how — label major blocks and nontrivial loops abstractly; document tricky rationale and bug-fix motivations (reference the bug tracker rather than duplicating it).
- Cross-module decisions: put the documentation where developers will naturally see it (RAMCloud's `Status` enum lists every place a new status must be added); when no natural home exists, keep a central `designNotes` file with topic sections and put one-line pointers ("See 'Zombies' in designNotes") at each dependent site.

🚩 **Red Flag: Comment Repeats Code** — if the information in a comment is already obvious from the code next to the comment, then the comment isn't helpful. One example of this is when the comment uses the same words that make up the name of the thing it is describing.

🚩 **Red Flag: Implementation Documentation Contaminates Interface** — this red flag occurs when interface documentation, such as that for a method, describes implementation details that aren't needed in order to use the thing being documented.

**Apply:**
- Write comments at a different level than the code: precise details for declarations, abstract intent for blocks and interfaces.
- For every variable: units, bounds, null-meaning, ownership, invariants.
- Keep interface comments free of implementation; put implementation notes inside the method.
- For each cross-module decision, choose one discoverable home for the documentation and point to it from the other sites.
- If a reviewer says something is nonobvious, clarify it — don't argue.

## Ch 14 — Choosing Names

Names are a form of documentation and of abstraction; mediocre names accumulate into system complexity, and a single ambiguous name can cause severe bugs.

- Sprite bug: the name `block` used for both physical disk blocks and logical file blocks; a logical number used where a physical one was needed zeroed an unrelated disk block, taking six months to find. `fileBlock`/`diskBlock` (or distinct types) would have prevented it.
- Create an image: ask "if someone sees this name in isolation, how closely can they guess what it refers to?" Two or three words maximum — pick the words that matter most.
- Be precise: avoid generic names (`getCount` → `numActiveIndexlets`; `x`,`y` for character positions → `charIndex`, `lineIndex`; `blinkStatus` → `cursorVisible`; boolean names should be predicates). Names can also be too *specific* (`delete(Range selection)` → `range`, since any range can be deleted). Generic loop variables `i`, `j` are fine when the whole scope is visible; the greater the distance between declaration and use, the longer the name should be.
- Be consistent: one name per purpose, used for that purpose and nothing else, with a definition narrow enough that all uses behave the same; add prefixes for multiples (`srcFileBlock`, `dstFileBlock`).
- Avoid extra words: no generic nouns (`fileObject`), no type encodings/Hungarian notation, no repeating the class name in a field.
- Against Go's ultra-short-name culture: readability is determined by readers, not writers; the same short name for multiple things (`ch`, `d`) invites `block`-style confusion.

🚩 **Red Flag: Vague Name** — if a variable or method name is broad enough to refer to many different things, then it doesn't convey much information to the developer and the underlying entity is more likely to be misused.

🚩 **Red Flag: Hard to Pick Name** — if it's hard to find a simple name for a variable or method that creates a clear image of the underlying object, that's a hint that the underlying object may not have a clean design.

**Apply:**
- Don't settle for a name that is "reasonably close"; make it precise, unambiguous, intuitive.
- Name booleans as predicates; never let one name serve two meanings anywhere in the system.
- Strip words that add no information; lengthen names as their scope grows.
- If naming something is hard, suspect the design — consider refactoring the entity instead of forcing a name.

## Ch 15 — Write The Comments First (Use Comments As Part Of The Design Process)

Write comments at the beginning of the process, not after coding: delayed comments never get written, or get written by someone mentally checked out, repeating the code.

- Order for a new class: class interface comment → interface comments + signatures for key public methods (bodies empty) → iterate until the structure feels right → declarations + comments for key instance variables → fill in bodies, adding implementation comments as you go. New methods get their interface comment before their body. When the code is done, the comments are done.
- Comments are a design tool: they are the only way to fully capture abstractions, so writing them early lets you review and tune the abstractions before implementing. Comments are a canary in the coal mine of complexity — if a method or variable needs a long comment, the abstraction is probably wrong; compare interface comment against implementation to gauge depth.
- Early comments are also more fun (recording design as you invent it) and roughly free: comment typing is ~5% of development time, and stabler abstractions reduce code rework.

🚩 **Red Flag: Hard to Describe** — the comment that describes a method or variable should be simple and yet complete. If you find it difficult to write such a comment, that's an indicator that there may be a problem with the design of the thing you are describing.

**Apply:**
- Write the interface comment before implementing any class or method.
- Use comment difficulty as a design signal: a long or convoluted comment means redesign, not wordsmithing.
- Never leave a backlog of "comments to add later."

## Ch 16 — Modifying Existing Code

Most of a system's design is determined by its evolution, not its initial conception; complexity creeps in through modifications unless every change is made strategically.

- The tactical modification mindset — "smallest possible change that does what I need" — accumulates special cases and dependencies. Ideal standard: after each change, the system has the structure it would have had if it had been designed from the start with that change in mind. If the current design is no longer the best one for the change, refactor.
- Whenever you touch code, leave the design at least a little better; if you're not making it better, you are probably making it worse. Under real deadline constraints, ask "is this the best I can possibly do given my constraints?" and schedule deferred refactorings explicitly.
- Maintaining comments: keep them **near the code** they describe (interface comment next to the method body, not in a distant header; users read Doxygen/IDE output, not headers). Spread implementation comments to the narrowest scope covering their code; the farther a comment sits from its code, the more abstract it should be.
- Comments belong in the code, not the commit log: information needed by future developers (e.g., the subtle bug a change fixes) must live in the code, or someone will undo the fix.
- Avoid duplication: document each design decision exactly once in the most obvious place; elsewhere, use pointers ("See comment in xyz"). Don't re-document another module's decisions at call sites, and don't restate external documentation — reference it.
- Check the diffs before committing: verify every change is reflected in the documentation (also catches leftover debug code and TODOs).
- Higher-level comments survive code change better than detailed ones.

**Apply:**
- Before a fix, ask what the design *should* be given the new requirement; refactor toward it rather than patching.
- Improve something in the design every time you modify code.
- Put change rationale in the code itself; treat the commit message as a copy at best.
- On every commit, diff-scan for comments invalidated by the change.

## Ch 17 — Consistency

Consistency means similar things are done in similar ways and dissimilar things in different ways; it creates cognitive leverage (learn once, apply everywhere) and makes assumptions safe.

- Levels: names (Ch 14), coding style guides, interfaces with multiple implementations, design patterns (e.g., MVC), invariants (properties always true, which shrink special-case reasoning).
- Ensuring it: **document** conventions (style guide, conspicuous location); **enforce** with automated pre-commit checkers (the line-terminator script that fixed a chronic CRLF problem) and nit-picky code reviews; **"When in Rome"** — before deciding anything in existing code, look for an established pattern and follow it.
- Don't change existing conventions: a "better idea" is not sufficient to introduce inconsistency; the value of consistency almost always exceeds the delta between approaches. Change only if you have significant new information *and* it's worth updating every old use so no trace of the old convention remains.
- Taking it too far: forcing dissimilar things to look the same (same name for different things, wrong design pattern) creates confusion — consistency only pays when "if it looks like an x, it really is an x."

**Apply:**
- Before writing new code in an area, read neighboring code and mimic its conventions.
- Automate convention enforcement (linters, pre-commit hooks) rather than relying on memory.
- Don't "improve" a convention unless you'll migrate every existing use.
- Never make different things look similar.

## Ch 18 — Code Should be Obvious

Obvious code can be read quickly with first guesses about behavior being correct; nonobviousness means the reader lacks needed information. Obviousness is judged by readers — if a reader says it's not obvious, it isn't.

- Makes code more obvious: good names, consistency, judicious white space (blank lines between labeled blocks, formatted parameter docs, spaces within statements), and comments that supply exactly the missing information.
- Makes code less obvious:
  - Event-driven programming — control flow is invisible (handlers invoked indirectly); compensate by documenting in each handler's interface comment when it is invoked.
  - Generic containers (`Pair`, `std::pair`) — `result.getKey()`/`getValue()` say nothing; define a small purpose-specific struct/class with meaningful names instead. General rule: **software should be designed for ease of reading, not ease of writing**.
  - Declaring a variable as one type and allocating another (`List` declared, `ArrayList` allocated) misleads readers about performance/thread-safety.
  - Code that violates reader expectations (a `main` that doesn't exit because a constructor spawned threads) — document departures from convention where readers will see them.
- Three ways to make code obvious: reduce the information needed (abstraction, eliminating special cases); exploit information readers already have (conventions, expectations); present the needed information in the code (names, strategic comments).

🚩 **Red Flag: Nonobvious Code** — if the meaning and behavior of code cannot be understood with a quick reading, it is a red flag. Often this means that there is important information that is not immediately clear to someone reading the code.

**Apply:**
- Format for the reader: blank lines between logical blocks, each opened by a summary comment.
- Replace generic containers/tuples with named types.
- Match declared and allocated types; when code must defy expectations, say so where the reader will look.
- Accept the reader's verdict on obviousness and fix the code, not the reader.

## Ch 19 — Software Trends

Evaluate every methodology and pattern against one question: does it actually reduce complexity in large systems?

- **Inheritance**: interface inheritance (signatures only, many implementations) fights complexity — the more implementations an interface has, the deeper it is. Implementation inheritance (inherited method bodies) reduces duplication but creates parent-child dependencies and information leakage through shared instance variables; in the worst case understanding any class requires the whole hierarchy. Prefer composition (helper classes); if unavoidable, keep parent-managed state private to the parent.
- **Agile development**: incremental and iterative development matches how good designs emerge, but agile's feature-focus risks tactical programming. The increments of development should be abstractions, not features: defer an abstraction until a feature needs it, then design it cleanly and somewhat general-purpose all at once.
- **Unit tests**: enormously valuable — they enable refactoring; without a test suite, structural improvements are too risky and design mistakes never get corrected (Tcl's byte-code compiler rewrite shipped with one post-alpha bug thanks to its suite).
- **Test-driven development**: writing tests before code focuses attention on making features work rather than finding the best design — tactical programming pure and simple. Exception: when fixing a bug, first write a failing unit test, then fix.
- **Design patterns**: fine as proven solutions, but over-application is the risk — don't force a problem into a pattern when a custom approach is cleaner; more patterns is not better.
- **Getters/setters**: shallow methods that expose implementation; better not to expose instance variables at all.

**Apply:**
- Prefer composition over implementation inheritance; use interface inheritance for depth.
- Develop in increments of abstractions, not features.
- Maintain a strong unit-test suite specifically so you can refactor.
- Write the failing test first for bug fixes; don't let TDD drive design.
- Avoid exposing state via getters/setters; challenge every new paradigm on complexity grounds.

## Ch 20 — Designing for Performance

Clean design and high performance are compatible: simpler code is usually faster, and the path to speed is simplicity applied to the critical path.

- Baseline attitude: don't micro-optimize everything (slows development, adds complexity), but don't ignore performance either ("death by a thousand cuts" leaves a 5–10x slower system with no single fix). Instead learn which operations are fundamentally expensive — network round trips (10–50 µs in-datacenter), disk I/O (5–10 ms), flash (10–100 µs), dynamic allocation, cache misses — via micro-benchmarks, and choose naturally efficient designs when they're equally simple (hash table over ordered map; array of structs over array of pointers).
- If efficiency requires significant complexity: start simple and optimize later unless you have clear evidence performance matters (RAMCloud committed to kernel bypass up front because measurements showed kernel networking couldn't meet its latency goal).
- Measure before (and after) modifying: intuition about performance is unreliable, even for experts. Measure deep enough to find where time actually goes; re-measure after the change, and back out changes that don't produce measurable speedup (unless they simplified the system).
- Design around the critical path: write down "the ideal" — the minimum code that must execute in the common case, ignoring existing structure — then find the cleanest design that stays close to it. Collapse layers, remove special cases from the critical path; ideally one `if` up front detects all special cases (which are handled off the path, structured for simplicity, not speed).
- RAMCloud Buffer example: the original allocation path crossed three same-signature methods (shallow layers — a red flag) and tested 6 conditions; the redesign handled the common case in one method with a single test (`availableAppendBytes` encoding three special cases as zero), got 2x faster, and shrank the class by 20%.

**Apply:**
- Know the rough cost of network hops, disk/flash I/O, allocation, and cache misses; pick the cheap design when it's equally clean.
- Never optimize on intuition; measure first, re-measure after, revert non-improvements.
- For hot code, define the ideal minimal critical path, then design the module around it.
- Move special-case handling off the critical path behind a single up-front test.

## Ch 21 — Decide What Matters

Good design separates what matters from what doesn't: structure the system around the important things, emphasize them, and hide or minimize everything else. This idea underlies abstraction, naming, and performance design alike.

- Find what matters by looking for **leverage**: a solution that solves many problems (the general text-insert/delete interface vs. a `backspace` method), or one piece of knowledge that explains many behaviors (an invariant). Comparing multiple candidates ("design it twice") makes the important one easier to spot; when unsure, hypothesize, commit, and learn from whether the hypothesis held.
- Minimize what matters: fewer required constructor parameters, defaults for common usage, information hidden in modules, exceptions handled entirely at low levels, configuration computed automatically — each removes something from everyone's plate.
- Emphasize what matters via **prominence** (interface docs, names, parameters of heavily used methods), **repetition** (key ideas recur), and **centrality** (the device-driver interface sits at the heart of the OS). De-emphasize the rest: hidden, rarely encountered, structurally uninfluential.
- Two mistakes: treating too many things as important (cluttered interfaces, irrelevant parameters, buffered/unbuffered distinctions nobody wants — shallow classes result) and failing to recognize something important (hidden essentials, recreated functionality, unknown unknowns).
- "Good taste" — the ability to distinguish what is important from what isn't — is central to being a good designer; the principle also applies to technical writing and beyond.

**Apply:**
- For each design, explicitly identify the few things that matter; build the structure around them.
- Shrink the "matters" list aggressively: defaults, hiding, automation.
- Put important things where they'll be seen; keep unimportant things out of interfaces entirely.
- When unsure what matters, pick a hypothesis, build, and review the outcome.

## Ch 22 — Conclusion

Everything reduces to complexity: its causes (dependencies, obscurity), its red flags (information leakage, unneeded error conditions, generic names), the remedies (deep and generic classes, defining errors out of existence, separating interface from implementation documentation), and the investment mindset that funds them. The cost is extra work early in a project — real, but quickly repaid: carefully defined modules get reused, clear documentation saves future time, and design skill compounds until good design costs little more than quick-and-dirty design. Good designers spend their time designing, which is fun; poor designers spend it chasing bugs in brittle code.

**Apply:**
- Accept slower early progress as the price of compounding speed later.
- Measure your growth as a designer by how quickly you produce simple, obvious structures.

## Summary of Design Principles

1. Complexity is incremental: you have to sweat the small stuff.
2. Working code isn't enough.
3. Make continual small investments to improve system design.
4. Modules should be deep.
5. Interfaces should be designed to make the most common usage as simple as possible.
6. It's more important for a module to have a simple interface than a simple implementation.
7. General-purpose modules are deeper.
8. Separate general-purpose and special-purpose code.
9. Different layers should have different abstractions.
10. Pull complexity downward.
11. Define errors out of existence.
12. Design it twice.
13. Comments should describe things that are not obvious from the code.
14. Software should be designed for ease of reading, not ease of writing.
15. The increments of software development should be abstractions, not features.
16. Separate what matters from what doesn't matter and emphasize the things that matter.

## Summary of Red Flags

1. **Shallow Module**: the interface for a class or method isn't much simpler than its implementation.
2. **Information Leakage**: a design decision is reflected in multiple modules.
3. **Temporal Decomposition**: the code structure is based on the order in which operations are executed, not on information hiding.
4. **Overexposure**: an API forces callers to be aware of rarely used features in order to use commonly used features.
5. **Pass-Through Method**: a method does almost nothing except pass its arguments to another method with a similar signature.
6. **Repetition**: a nontrivial piece of code is repeated over and over.
7. **Special-General Mixture**: special-purpose code is not cleanly separated from general-purpose code.
8. **Conjoined Methods**: two methods have so many dependencies that it's hard to understand the implementation of one without understanding the implementation of the other.
9. **Comment Repeats Code**: all of the information in a comment is immediately obvious from the code next to the comment.
10. **Implementation Documentation Contaminates Interface**: an interface comment describes implementation details not needed by users of the thing being documented.
11. **Vague Name**: the name of a variable or method is so imprecise that it doesn't convey much useful information.
12. **Hard to Pick Name**: it is difficult to come up with a precise and intuitive name for an entity.
13. **Hard to Describe**: in order to be complete, the documentation for a variable or method must be long.
14. **Nonobvious Code**: the behavior or meaning of a piece of code cannot be understood easily.
