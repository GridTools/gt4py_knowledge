---
title: GT4Py Next computation DSL specification
author: egparedes
tags: [documentation, dsl, public-api, frontend, type-system, fields, domain, dimensions, connectivities, field-operators, scan, embedded, validation, provenance]
created: 2026-10-22
status: draft
---

> **Snapshot context.** Appendix to [[personal/egparedes/next-dsl-documentation/next-dsl-documentation|the documentation proposal and generation process]].
> The reviewed source body follows in full. This personal draft records the stated
> GT4Py revision; it is not an accepted normative standard. Migration adds this
> frontmatter/context, relocates evidence links to immutable source URLs, and uses
> knowledge-base wikilinks for companion documents. Technical claims are preserved.

# gt4py.next computation DSL specification

## Status, scope, and evidence

This document specifies the Python-facing computation language implemented by
`gt4py.next` at repository revision
`f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f`. It covers field operators, scan
operators, programs, and their interface with Python. Internal representations
are evidence for these semantics, not additional user languages specified here.
The package remains experimental; this is a descriptive specification of this
checkout, not a promise of compatibility with later releases.

Rules stated below describe the frontend and its intended mathematical model,
with observed execution differences called out explicitly. A rejection by the
frontend is different from a limitation of a particular backend. A historical
ADR records design intent; it does not establish current support by itself.
Recommendations in the conflict register are proposals for interpreting or
reconciling existing sources, not changes made to the implementation.

The investigation follows three evidence streams: user documentation and
examples (including notebook cells), frontend/data-model/embedded/lowering
implementation, and unit/integration/regression tests. The final source index
records their coverage. Accepted and rejected examples in tests are executable
evidence, subject to their backend skips and expected failures. An untested
implementation path is identified as such where it affects the contract.

### Notation

`d` is a named dimension; `D` is an ordered sequence of dimensions; `τ` is a
scalar dtype; `F[D, τ]` is a field type; `T₀ × … × Tₙ` is a fixed product type.
`Γ ⊢ e : T` means that expression `e` has type `T` in environment `Γ`.
`R(d) = [a, b)` is an integer interval. `p|D` projects a labeled index onto
`D`. Mathematical rules describe values, not allocation or fusion decisions.

A rule applies only when its side conditions hold. Unless a section explicitly
states an execution difference, storage must contain every element read and
outputs must contain every element written. No rule implies automatic halo
exchange, padding, clipping, or boundary conditions.

## 1. Semantic objects

### 1.1 Dimensions and grids

A dimension is the immutable pair `(name, kind)`. Kinds are `HORIZONTAL`
(default), `LOCAL`, and `VERTICAL`. Equality includes both name and kind;
separately created equal dimensions identify the same axis. Dimensions label
logical coordinates, not array positions or units of physical distance.
Horizontal dimensions may represent Cartesian coordinates or unstructured
location sets such as cells, edges, and vertices. A local dimension labels
neighbor slots; a vertical dimension may carry a sequential scan.

Canonical field dimension order groups horizontal, local, then vertical
axes, ordering base names lexicographically within a group. Dimension promotion
merges compatible dimension sequences into this order. Field/DSL dimension
checks reject duplicates, more than one local dimension, and coexistence of a
base dimension and its staggered counterpart. A host `Domain` object has weaker
construction checks and is not proof that a field annotation is admissible.

For non-staggered `d`, `flip_staggered(d)` represents locations half a cell below
`d`: its integer index `i` denotes physical position `i - 1/2` in base-grid
coordinates. `is_staggered` and `as_non_staggered` inspect/normalize this pairing.
The `_Staggered` name prefix is implementation encoding; names such as `IHalf`
in examples are Python variables, not a required suffix. Staggering preserves
kind. It changes labeled coordinate spaces, not array index types.

`GridType.CARTESIAN` and `GridType.UNSTRUCTURED` select grid constraints.
Unstructured offsets/local dimensions require the unstructured grid model;
unstructured programs can also perform Cartesian-like calculations. When the
user does not select a grid type, the frontend deduces it from referenced
dimensions, offsets, and operators.

Evidence: [data model][model] (`Dimension`, `order_dimensions`, `check_dims`,
`promote_dims`, stagger helpers), [grid deduction][transform],
[dimension tests][dimension-tests], [staggered tests][staggered-tests], and
[ADR 26][adr26].

### 1.2 Ranges, domains, and coordinates

A unit range is a contiguous integer interval `[a, b)` with stride one.
`UnitRange` normalizes `a >= b` to the empty range `[0, 0)`. Semantic bounds
may be infinite, for example on broadcast dimensions; allocated storage needs
finite extents. A domain is an ordered product

```text
R = {d₁: [a₁,b₁), …, dₙ: [aₙ,bₙ)}
|R| = {p | aᵢ ≤ p[dᵢ] < bᵢ for every i}.
```

The domain with no dimensions contains the single empty coordinate. A domain
with an empty constituent interval has no coordinates. Bounds are runtime
properties, not part of a field's static type.

Domain intersection aligns by dimension labels, adds dimensions present on only
one side, and intersects shared intervals. Equivalently, an absent dimension
has an unbounded interval for this operation. This is the domain algebra of
pointwise field evaluation; it is not NumPy positional shape broadcasting.

The host `domain(...)` conversion accepts a `Domain`, a mapping, or an ordered
sequence of `(Dimension, range-like)` pairs. An integral extent `n` means
`[0,n)`; a pair `(a,b)` means absolute bounds; a Python range must have unit
stride. These conveniences are broader than captured program domain syntax
(section 6).

`d(i)` is a named absolute index. Relative host array indexing and program
output slices use positions relative to existing bounds; explicit domain pairs
use logical coordinates. For a field with logical interval `[-2,8)`, the slice
`[0:3]` selects `[-2,1)`, whereas domain `{d: (0,3)}` selects `[0,3)`.

Evidence: [data model][model] (`UnitRange`, `unit_range`, `Domain`, `domain`),
[model tests][model-tests] (empty ranges/domains and domain intersections), and
[domain tests][domain-tests].

### 1.3 Scalars, fields, and collections

A scalar is a value of a concrete dtype. A field `F[D,τ]` is a partial function
from labeled coordinates in a domain over `D` to scalar values of dtype `τ`.
Storage layout, strides, padding, device, and allocator do not change this
mathematical value. The Python spelling is `gtx.Field[gtx.Dims[d₁, …, dₙ], τ]`;
`gtx.Field[[d₁, …, dₙ], τ]` also occurs in existing code.

A zero-dimensional field `F[[],τ]` contains one value but remains a field.
A scalar and a zero-dimensional field are not generally identical: only a
scalar boolean controls `if`. `.as_scalar()` is a host extraction operation.
At call boundaries, `promote_zero_dims` allows a matching scalar for an annotated
zero-dimensional field, recursively inside supported collections. It does not
implicitly convert a scalar argument into an arbitrary nonzero-dimensional
field.

Tuples are fixed, heterogeneous products, possibly nested. They can contain
scalars and fields and can be arguments, local bindings, results, and scan
carries. A tuple of fields is not a field with tuple-valued elements. Supported
typed named collections additionally preserve member names and original Python
type. Their admissibility is checked by the named-collection/type-translation
layer; arbitrary Python objects are not DSL values. Member access,
construction, unpacking, and selected structure-preserving builtins operate on these
statically known products (section 3.4).

Evidence: [type specifications][types], [type translation][type-translation],
[call compatibility][call-types] (`promote_zero_dims`), [named collections][named],
[tuple tests][tuple-tests], [named-collection tests][named-tests], and
[ADR 13][adr13].

### 1.4 Connectivities and missing neighbors

A connectivity is an integer index map `C : R_C → Z_d`, with codomain dimension
`d`. A neighbor table has origin dimension `a`, local dimension `l`, and
codomain `d`; its entries `C[a,l]` identify neighbors in `d`. Multiple slots
may contain the same neighbor; multiplicity matters in reductions.

A declaration `FieldOffset(tag, source=d, target=(a,l))` names such a remapping.
Its `source` is the dimension read in the input field; its `target` describes
the replacement dimensions in the result. By contrast, a neighbor table's
`source_dim` names its origin `a`. These two uses of “source” have opposite
orientations and must not be interchanged.

The runtime `offset_provider` maps tags to neighbor tables. Its compile-time
descriptor includes domain/codomain dimensions, integer dtype, skip metadata,
and maximum neighbor count; actual table entries are runtime data. Current
Cartesian dimension shifts describe themselves and need no provider entry.

A declared skip value, normally `-1`, denotes an absent neighbor. Local
reductions omit skipped slots. Ordinary fixed-neighbor dereferences need valid
indices; no portable value is specified for reading a skipped neighbor or an
out-of-bounds table entry. The connectivity is not an automatic boundary
condition.

Evidence: [connectivity model][model] (`Connectivity`, `NeighborTable`,
`NeighborConnectivityType`, `OffsetProvider`, `get_offset`),
[constructors] (`as_connectivity`), [ADR 19][adr19], and
[reduction tests][reduction-tests].

## 2. Language layers and syntax

Python code outside decorated functions is ordinary host Python. Decorated
functions form three related constructs:

| Construct                                   | Meaning                                                | Result/effect                                                                                 |
| ------------------------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `@field_operator`                           | Pure composition of operations on whole fields/scalars | Returns a DSL value inside an operator; writes through `out` at an output-writing entry point |
| `@scan_operator(axis=K, forward=b, init=z)` | A scalar recurrence lifted over columns                | Field/collection of fields containing successive updated carries                              |
| `@program`                                  | Ordered composition of output-writing operator calls   | Mutates supplied output fields; returns `None`                                                |

These are source-captured, statically analyzed Python subsets, not arbitrary
Python functions accelerated opportunistically. The same frontend validation
runs when selecting field-level embedded execution. Ordinary host methods such
as `.ndarray`, `.asnumpy()`, allocation, or I/O do not thereby become DSL
operations.

### 2.1 Abstract grammar

The following grammar describes constructs, not Python precedence or indentation
lexing. Python's parser supplies those; static side conditions below further
restrict the productions. `suite` means an indented sequence of statements.
`call-arguments` are statically bound positional/keyword arguments, not arbitrary
Python argument expansion.

```ebnf
field-definition = "def" name "(" typed-parameters ")"
                   [ "->" data-type ] ":" field-suite ;
field-suite      = { field-statement } ;
field-statement  = assignment | conditional | "return" expression
                 | "pass" | docstring ;
assignment       = targets [ ":" data-type ] "=" expression ;
targets          = name | tuple-pattern | chained-name-targets ;
conditional      = "if" expression ":" field-suite
                   { "elif" expression ":" field-suite }
                   [ "else" ":" field-suite ] ;
expression       = scalar-literal | name | qualified-reference
                 | tuple-expression | collection-construction
                 | unary-op expression
                 | expression binary-op expression
                 | expression comparison-op expression
                 | expression "if" expression "else" expression
                 | callable "(" call-arguments ")"
                 | field "(" remappings ")"
                 | expression "[" restricted-index "]" ;
unary-op         = "+" | "-" | "~" | "not" ;
binary-op        = "+" | "-" | "*" | "/" | "//" | "%" | "**"
                 | "&" | "|" | "^" ;
comparison-op    = "==" | "!=" | "<" | "<=" | ">" | ">=" ;

program-suite    = { operator-name "(" call-arguments ","
                   "out=" output [ ", domain=" domain-spec ] ")" } ;
output           = field-name | field-name "[" output-slices "]"
                 | output-collection ;
domain-spec      = "{" dimension ": (" integer-expression ","
                   integer-expression ")" { "," domain-entry } "}"
                 | domain-tuple ;
```

The program production abbreviates argument ordering and a trailing comma;
it does not require an ordinary argument before `out`. Domain tuples must
match output structure. Supported module-qualified dimensions, offsets, and constants may be resolved
from closure references. Callable calls require directly bound names: forms
such as `gtx.broadcast(...)` or `module.fieldop(...)` in a captured body are
currently rejected, as pinned by [module-import tests][import-tests]. Import
the callable into a directly bound name outside the DSL function.

Evidence: [field parser][fo-parser], [program parser][p-parser],
[dialect parser][dialect], [frontend AST passes][ast-passes],
[field parser tests][fo-parser-tests], and [program parser tests][p-parser-tests].

### 2.2 Binding, assignments, and control flow

Every ordinary DSL function parameter needs a supported type annotation.
A field-operator return annotation is optional; when supplied, it must equal the
deduced return type. The safe signature subset is ordinary positional-or-keyword
parameters. Full Python support for positional-only parameters, keyword-only
parameters, defaults, `*args`, and `**kwargs` must not be inferred from the
Python function object: the current parsers primarily consume `ast.arguments.args`
(see conflict C12).

Locals bind immutable values; rebinding a name expresses a new value, not a
mutation of field storage. The frontend converts bindings to static single
assignment form. It supports annotated assignment, chained name assignment,
and tuple/named-collection unpacking; a starred unpacking target receives a
tuple. Assignment through a field subscript or object attribute is not a
field-operator operation. Function/operator definition names cannot shadow
reserved DSL builtin names; this explicit check does not establish a separate
blanket prohibition on every local/parameter spelling.

An `if` statement or conditional expression requires scalar `bool`, not a
boolean field or zero-dimensional boolean field. Only the chosen branch
contributes its runtime result, but both branches must be statically valid.
All reachable paths of a field operator return, and all returned types agree
exactly. Conditional-expression alternatives likewise have equal types; no
branch dtype/dimension promotion is implied. A new name is available after a
branch only when the binding/SSA analysis can establish it on all relevant
paths with compatible types. `where` provides elementwise field selection.

Closure references may name supported constants, dimensions, offsets, callable
DSL objects, and typed collections. Module attribute references are resolved
against captured Python objects; arbitrary method calls or Python callbacks
are not implied. Closure-mutation behavior differs by execution mode
(section 7 and C11).

### 2.3 Rejected Python constructs

The supported subsets do not include loops (`for`, `while`), comprehensions,
generators, lambdas, exception handling, context managers, assertions, classes,
pattern matching, import statements in the DSL body, or nested ordinary Python
function definitions. Augmented assignment such as `+=` is also rejected.
Arbitrary expression statements are invalid in field
operators. Boolean `and` and `or` are rejected even for scalar booleans; use
boolean `&` and `|`. Chained comparisons such as `a < b < c` are rejected;
write the individual comparisons with `&` and explicit parentheses.

Programs have a smaller subset: output-writing field/scan-operator calls in
sequence. Programs do not introduce local temporaries with assignment, branch,
loop, return a value, or call another program. Field expressions belong in
field operators. Scalar expressions used for call arguments or bounds use the
restricted program expression grammar: scalar arithmetic, negative scalar
literals, and supported scalar `minimum`/`maximum`, not arbitrary field
computation or unrestricted unary expressions.

Evidence: [dialect rejection catalogue][dialect], [diagnostic guide][diagnostics],
[diagnostic tests][diagnostic-tests], [parser tests][fo-parser-tests]
(`test_domain_chained_comparison_failure`,
`test_field_chained_comparison_failure`), and [program parser][p-parser].

## 3. Static types

### 3.1 Type universe and dtype identity

```text
τ ::= bool | int8 | int16 | int32 | int64
    | uint8 | uint16 | uint32 | uint64 | float32 | float64
T ::= τ | F[D,τ] | (T₀, …, Tₙ) | NamedProduct[type, members: T]
```

Dimensions, domains, offsets, callable types, and constructors additionally
have compile-time/meta-level types. Their presence does not make them
arbitrary runtime field elements. Strings are represented by the type system
and can occur as scalar-constructor literals; this is not a promise of string
field arithmetic.

The Python `int` annotation/constructor maps to signed `int64`, and `float` to
`float64`, independent of platform integer width. In contrast, a plain Python
integer value is inferred as `int32` when representable, otherwise `int64`;
values outside signed 64-bit range are rejected. Source integer constants are
classified before unary signs: unary minus preserves its operand's dtype.
Consequently the source expression `-2147483648` has type `int64` because its
positive operand exceeds int32, and `-9223372036854775808` is rejected because
its positive operand exceeds int64. A host value already equal to either
negative minimum is classified by its signed value instead. A decimal floating
literal is `float64`. NumPy scalar constants preserve their concrete kind. `IndexType` aliases `int32`; frontend alias support
for constructor spelling is incomplete. Export/type-enum availability of
small/unsigned integer widths is broader than the subset handled by some
`as_dtype`/`from_dtype` and backend paths. These paths commonly support only
`bool`, `int32`, `int64`, `float32`, and `float64`; portable use of other widths
requires backend-specific evidence (C13).

Scalar constructors are intended for typed literals. Reserved constructor
spellings enforce literal arguments; use `astype` for data conversions.
Constructor aliases may bypass the initial spelling check, but literal-only
validation also occurs during lowering. They do not establish a general
computed-value conversion interface.

Evidence: [scalar/type definitions][types], [type translation][type-translation]
(`get_scalar_kind`), [builtins], [typing rules][typing], and
[conversion tests][conversion-tests].

### 3.2 Dimension promotion

Write `dims(τ)=[]`, `dims(F[D,τ])=D`, and `dtype(T)=τ` for primitive scalar/field
types. `merge(D₁,D₂)` is dimension promotion under the constraints of section
1.1. For admissible ordinary pointwise arithmetic:

```text
Γ ⊢ a : T₁     Γ ⊢ b : T₂     dtype(T₁) = dtype(T₂) = τ
τ is arithmetic     D = merge(dims(T₁), dims(T₂))
────────────────────────────────────────────────────────────
Γ ⊢ a op b : τ                       if both operands are scalars
Γ ⊢ a op b : F[D,τ]                  otherwise
```

Shared dimensions align by label, not positional index. A scalar or a field
missing an axis is constant along that axis. At shared axes the evaluated
domain is the intersection of available ranges. Distinct extents on shared
axes are not a request to broadcast length one positionally.

There is **no general implicit dtype promotion**. For example, combining an
`int64` field and a small integer literal (`int32`) requires an explicitly
typed literal or `astype`; an `int32` field plus `1` needs no dtype conversion.
Combining float32 and float64 is likewise not accepted by the ordinary rule.
Bool is not an arithmetic dtype.

`**` currently bypasses the ordinary result-promotion path and is a recorded
exception, not a model for other binary operations (C5).

Evidence: [type utilities][type-info] (`promote`), [typing rules][typing],
[model] (`promote_dims`), and [type deduction tests][typing-tests].

### 3.3 Calls and exact matching

Callable signatures determine argument count, names, supported keyword binding,
and structural types. In captured calls, passing an argument with the wrong
dtype, dimension list, or collection structure is an error; dimension promotion of an expression does
not make every field argument interchangeable with every annotation. The
zero-dimensional scalar compatibility of section 1.3 is a specific exception.
Return annotations constrain the full inferred type. Output fields must match
the computed result's dimensions/dtype/collection structure; a domain
restriction does not change a field's type. A scalar return is not generally
assignable to a Field output through the compiled frontend: use `broadcast`
to produce a statically field-valued result.

These matching rules are also caller obligations at host entry points. Direct
field-level embedded operator calls do not perform all runtime dtype, dimension,
or nominal-collection checks; invalid calls can execute or cast on assignment.
This accidental acceptance is not supported coercion (C21).

Scan calls have a distinct promotion rule: a scalar recurrence parameter may
receive a matching scalar or field leaf, and the result is lifted over the
merged invocation dimensions plus the scan axis (section 5.5).

Evidence: [call compatibility][call-types], [type utilities][type-info],
[argument interface tests][arg-tests], and [scalar argument tests][scalar-tests].

### 3.4 Products and named collections

Tuple element access uses a statically known integer index; negative tuple
indices are supported. Tuple slicing and dynamic tuple indexing are not a
portable DSL operation. Tuple unpacking is statically expanded, including one
starred target. For named collections, supported Python declarations include
typed named tuples and admissible dataclasses/custom named-collection types.
Dataclasses must satisfy the constructor/type layer's restrictions: at least
one member, no custom `__init__`, no `InitVar`, and no default-valued fields.

Collection construction and member access preserve the nominal collection
type; plain tuples and distinct named types are not freely interchangeable.
`where` recurses through compatible products, including named collections;
`astype` supports scalars, fields, and recursive tuples, without equivalent
named-collection frontend support. These are products
of DSL data, without arbitrary Python object behavior, member mutation, or
user-defined methods.

Evidence: [named collections][named], [type translation][type-translation]
(`make_type`, `make_constructor_type`), [unpacking pass][unpack],
[named-collection tests][named-tests], and [tuple tests][tuple-tests].

## 4. Pointwise operations and builtins

### 4.1 Evaluation rule

For a pointwise operation on fields/scalars, let `R` be the intersection of
operand domains under the dimension algebra, treating scalars as constant.
For each `p ∈ |R|`:

```text
(op(e₁,…,eₙ))(p) = op_scalar(e₁(p|dims(e₁)), …, eₙ(p|dims(eₙ))).
```

This equation specifies values and dimensional alignment. It does not prescribe
intermediate materialization, iteration order, fusion, floating-point
reassociation, or evaluation of invalid coordinates outside `R`.

### 4.2 Operator and builtin inventory

The table lists the Python-facing function inventory of `fbuiltins`, together
with expression operators. “Same type” includes dimension preservation;
“promote” means dimension promotion with matching dtypes, not dtype widening.
The experimental functions are specified separately in section 5.

| Spelling                                                | Operand constraint                                                                                 | Static result / meaning                                                                                                                                            |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Unary `+`, `-`; `abs`, `neg`                            | Arithmetic scalar/field                                                                            | Same type; identity, negation, magnitude, negation respectively                                                                                                    |
| `+`, `-`, `*`, `/`, `//`                                | Arithmetic scalar/fields with matching dtype                                                       | Promoted dimensions, same dtype; `/` does not acquire a floating static type automatically                                                                         |
| `%`                                                     | Ordinary promotion plus integral right operand (therefore both integral under matching-dtype rule) | Same dtype over promoted dimensions                                                                                                                                |
| `**`                                                    | Arithmetic base and exponent; special frontend result path                                         | Base type is returned directly; floating base with integral scalar exponent is tested; additional exponent dimensions are not validated/promoted consistently (C5) |
| `==`, `!=`, `<`, `<=`, `>`, `>=`                        | Matching arithmetic dtypes                                                                         | `bool` scalar/field over promoted dimensions; bool equality is not covered by this arithmetic comparison rule                                                      |
| `&`, `\|`, `^`                                          | Bool scalar/fields                                                                                 | Bool conjunction, disjunction, exclusive-or over promoted dimensions; not integer bitwise arithmetic                                                               |
| `~`, `not`                                              | Bool scalar/field in frontend typing                                                               | Logical negation; prefer `~` on fields because `not field` is an explicitly disputed accidental acceptance (C6)                                                    |
| `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`       | Floating scalar/field                                                                              | Same floating type                                                                                                                                                 |
| `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh` | Floating scalar/field                                                                              | Same floating type                                                                                                                                                 |
| `sqrt`, `exp`, `log`, `gamma`, `cbrt`                   | Floating scalar/field                                                                              | Same floating type                                                                                                                                                 |
| `floor`, `ceil`, `trunc`                                | Floating scalar/field                                                                              | Same floating dtype, rather than an integer result                                                                                                                 |
| `isfinite`, `isinf`, `isnan`                            | Floating scalar/field                                                                              | Bool over the same dimensions                                                                                                                                      |
| `minimum`, `maximum`, `fmod`, `power`                   | Arithmetic scalar/fields under ordinary promotion                                                  | Matching dtype over merged dimensions; integral `power` emits a result-type consistency warning                                                                    |
| `astype(value, dtype)`                                  | Scalar/field or recursively nested tuple                                                           | Convert every primitive leaf to `dtype`, preserving dimensions and tuple structure                                                                                 |
| `broadcast(value, dimensions)`                          | Scalar/field; target dimensions contain every input dimension                                      | Field of the same dtype over exactly the target dimensions                                                                                                         |
| `where(mask, true_value, false_value)`                  | Bool field mask; matching branch structure and corresponding leaf dtypes                           | Recursive field result with branch/mask dimension promotion                                                                                                        |
| `neighbor_sum(value, axis=L)`                           | Local-neighbor field                                                                               | Sum valid neighbors; remove `L`, preserve static dtype                                                                                                             |
| `max_over(value, axis=L)`, `min_over(value, axis=L)`    | Local-neighbor field                                                                               | Maximum/minimum valid neighbors; remove `L`, preserve static dtype                                                                                                 |

Scalar constructors `bool`, `int`, `float`, the explicitly sized numeric
constructors, and `tuple` are also exposed names. The collection/literal/type
rules apply to these names; their export is not an unrestricted re-export of
all Python constructor behavior.

Numerical edge cases are only partly specified by the repository. In
particular, integer `/` has integer static type while embedded array true
division can produce floating values. Negative modulo and floor division carry
backend exclusions. Division by zero may yield IEEE values or raise, and the
basic tests accept both. Overflow/narrowing, NaN and signed-zero propagation,
and reproducible accumulation order have no uniform complete contract here.
Use explicit types and the selected backend's established behavior; C14 records
the unresolved numerical contract.

Evidence: [builtin definitions][builtins], [typing rules][typing],
[expression lowering][fo-lowering], [math test data][math-data],
[math execution tests][math-tests], [unary tests][unary-tests],
[basic feature tests][basic-tests], and [backend exclusions][matrix].

### 4.3 Broadcasting and conversion

For `Γ ⊢ v : T` of dtype `τ`, if `dims(T) ⊆ D` and `D` is admissible:

```text
Γ ⊢ broadcast(v, D) : F[D,τ]
broadcast(v,D)(p) = v(p|dims(T)).
```

Broadcasting introduces no coordinate dependence on the added dimensions.
Embedded scalar broadcast currently returns its scalar unchanged rather than
materializing the statically described field (C7); later output assignment may
still give the expected values. This matters to field-only operations on the
broadcast result, particularly zero-dimensional ones.

`astype` changes dtype explicitly, not domain or coordinate dependence. For a
tuple it converts leaves independently while retaining tuple structure. Named
collections lack the corresponding generic frontend acceptance.
It is the supported operation for computed-value conversion, unlike the
literal constructors. A cast does not repair missing input coordinates or
change connectivity bounds.

Evidence: [typing rules][typing], [builtins],
[embedded field implementation][array-field], [broadcast tests][broadcast-tests],
and [conversion tests][conversion-tests].

### 4.4 Elementwise selection

For primitive branches `t`, `f`, and boolean field `m`, let
`D = merge(dims(m), dims(t), dims(f))`. Corresponding branch dtypes must agree.
Then `where(m,t,f)` has type `F[D,τ]` and, on the common valid domain:

```text
where(m,t,f)(p) = t(p|dims(t))  if m(p|dims(m))
                = f(p|dims(f)) otherwise.
```

For products the rule applies to corresponding leaves, preserving the product
and any named type. Even scalar branch leaves become statically field-valued
through the mask. A boolean scalar is not the ordinary `where` mask type.

`where` selects among field expressions; it does not promise lazy Python
short-circuit evaluation of branch expressions. Both expressions must be
well-typed and valid for the required evaluation region. In particular,
`where` is not a general mechanism for hiding an out-of-bounds shift in the
unselected branch. Region-based concatenation has a different domain rule
(section 5.4).

Evidence: [typing rules][typing] (`_visit_where`), [where tests][where-tests],
[tuple tests][tuple-tests], and [Quickstart].

## 5. Coordinate and sequential operations

### 5.1 General remapping

For a field `f` containing dimension `d`, and connectivity `c` with codomain
`d`, remapping substitutes that coordinate:

```text
f(c)(p) = f(p with its d-coordinate replaced by c(p|dims(c))).
```

Its type replaces `d` by the connectivity's domain dimensions and retains
unaffected dimensions, subject to dimension constraints. Connectivity entries
must name valid input coordinates for every access. Halo allocation belongs
to the caller; domain inference does not synthesize halo values.

For `Off = FieldOffset(tag, source=d, target=(a,l))` and provider table `C`:

```text
f(Off)(a,l,…)    = f(C[a,l],…)    ; result replaces d by (a,l)
f(Off[q])(a,…)   = f(C[a,q],…)    ; result replaces d by a.
```

The first gathers all neighbor slots; the second selects a fixed neighbor.
A Cartesian offset needs a displacement, so the analogous unindexed
single-target `f(Off)` is rejected. A field already carrying a local axis
can select a slot as `field[LocalDim(q)]`, removing that axis. This restricted
field subscript differs from tuple element access and host array slicing.

Multiple Cartesian remappings may occur in one call or chained calls. Fixed
unstructured remaps can compose. A remapping call that creates a local neighbor
dimension permits only one remapping argument; arbitrary mixtures are not
established by the generic equation. Runtime selection of an unstructured
neighbor slot is not a general supported operation.

Evidence: [type utilities][type-info] (`return_type_field`,
`function_signature_incompatibilities_field`), [expression lowering][fo-lowering]
(`_visit_shift`), [shift tests][shift-tests], [reduction tests][reduction-tests],
and [local-field tests][local-tests].

### 5.2 Cartesian and staggered shifts

For integral literal displacement `n`:

```text
f(d+n)(…,i,…) = f(…,i+n,…)
dom(f(d+n))[d] = dom(f)[d] - n.
```

`d-n` reverses the displacement. Use dimension arithmetic for current fixed
Cartesian shifts. Indexed single-target `FieldOffset` remains accepted by parts
of the frontend but retains legacy tagged-provider behavior; it is not an
equivalent portable, self-contained shift spelling (C4). This limitation does
not remove the single-target declaration used by `as_offset` below.
Closure folding can supply a literal;
a runtime scalar displacement is not the same syntax as a fixed shift.

For half-integral `h`, `f(d+h)` reads the opposite staggered dimension and
returns dimension `d`. With result index `i`, the input integer index is
`i + floor(h) + 1` when `d` is nonstaggered, and `i + floor(h)` when `d` is
staggered. Thus a field on `IHalf = flip_staggered(I)` read as `f(I+0.5)` uses
`IHalf(i+1)`, while a field on `I` read as `f(IHalf+0.5)` uses `I(i)`.
Other fractional parts are rejected. Integer shifts retain staggering.

Experimental `as_offset(Off,q)` from `gt4py.next.ffront.experimental` makes a
coordinate-dependent Cartesian offset. `Off` is single-target with equal
source/target dimension `d`; `q` is an integral field containing `d`:

```text
f(as_offset(Off,q))(p) = f(p with d := p[d] + q(p)).
```

All dynamically chosen input coordinates must be valid. The current static
result-dimension rule does not uniformly account for extra dimensions of `q`;
treat such extra-dimensional cases as unresolved rather than a general
broadcasting guarantee. This API does not establish arbitrary dynamic
unstructured neighbor selection.

Evidence: [Cartesian connectivity construction][model],
[experimental builtins][experimental], [typing rules][typing],
[shift tests][shift-tests], and [staggered tests][staggered-tests].

### 5.3 Neighbor reductions

The portable reduction model is a reduction over one `LOCAL` neighbor axis
with a known connectivity association. For valid-slot set
`S(a) = {l | C[a,l] is not the skip sentinel}`:

```text
neighbor_sum(v, axis=L)(a,…) = Σ {v(a,l,…) | l ∈ S(a)}
max_over(v, axis=L)(a,…)     = max {v(a,l,…) | l ∈ S(a)}
min_over(v, axis=L)(a,…)     = min {v(a,l,…) | l ∈ S(a)}.
```

The result removes `L`; other dimensions and the static dtype remain.
Repeated neighbor indices contribute once per slot. These operations are not
a general horizontal/vertical global reduction API. Frontend axis checking is
weaker than this portable model: it checks presence, whereas embedded execution
requires `LOCAL` (C8).

Missing neighbors are excluded rather than replaced by caller-provided physical
boundary values. The implementation disagrees about min/max identities for
all-skipped neighborhoods and infinite data; no uniform result is specified
for those cases (C9). The frontend's preserved sum dtype may also differ from
array-library accumulator widening.

Externally supplied local coefficient fields are supported in expressions with
shifted neighborhood fields. Reduction of a standalone external sparse/local
field is explicitly skipped because compiled unrolling lacks the corresponding
neighborhood evidence. Sparse output writes and arbitrary mismatches between
offset tag and local dimension name also have backend exclusions. Use the
feature matrix, not the broad static `Field` type, to determine portability.

Evidence: [reduction lowering][fo-lowering] (`_visit_neighbor_sum`,
`_visit_max_over`, `_visit_min_over`), [embedded reductions][array-field]
(`_make_reduction`), [reduction tests][reduction-tests],
[local-field tests][local-tests], and [backend exclusions][matrix].

### 5.4 Region selection: experimental `concat_where`

`concat_where(R,t,f)` is imported from `gt4py.next.ffront.experimental`.
Its condition is a coordinate region, not a boolean field. Dimension/index
comparisons describe regions; with `n` of `IndexType` (`int32`):

| Expression | Region along `d`                                            |
| ---------- | ----------------------------------------------------------- |
| `d < n`    | `(-∞, n)`                                                   |
| `d <= n`   | `(-∞, n+1)`                                                 |
| `d > n`    | `[n+1, +∞)`                                                 |
| `d >= n`   | `[n, +∞)`                                                   |
| `d == n`   | `[n, n+1)`                                                  |
| `d != n`   | Complement of `[n, n+1)`; not generally embedded-compatible |

Here the `d > n` row means integer indices `i >= n+1`; infinite endpoints are
only notation for unbounded intervals. Reversed comparisons reverse the
relation. Parenthesized domain `&`, `|`, `^` expressions have frontend typing,
but disjoint and multidimensional regions are not uniformly executable.
Coordinate arithmetic such as `d % 2` is not an index-field expression.

For primitive branches and a one-dimensional region `R` along dimension `d`,
let `Q` be the intersection of the branch domains projected onto all dimensions
other than `d`. Missing branch dimensions are unbounded for this operation.
Extend `Q` without restriction along `d` when intersecting it below:

```text
Q = projection_except_d(dom(t)) ∩ projection_except_d(dom(f))
dom(concat_where(R,t,f)) = Q ∩ ((dom(t) ∩ R) ∪ (dom(f) ∩ complement(R)))
concat_where(R,t,f)(p)   = t(p) if p ∈ R, otherwise f(p).
```

This accounts for the orthogonal intersection performed before embedded
concatenation. For example, true/false branches on `{I:(0,1), J:(0,3)}` and
`{I:(1,2), J:(1,4)}` with `R = I < 1` produce `{I:(0,2), J:(1,3)}` rather than
the nonrectangular union of their unrestricted contributions. Missing branch
dimensions follow broadcasting; products are processed leafwise with matching
structure/dtypes.
Each branch need only cover its contributed region, so nonoverlapping input
fields can concatenate. This is materially different from `where`'s common
branch-domain requirement.

Embedded execution requires a one-dimensional condition and a contiguous
resulting domain; interior gaps raise `NonContiguousDomain`. It cannot express
every compiled disjoint condition, including `d != n`. Empty branches,
infinite broadcast domains, scalar branches, and more elaborate concatenations
have feature-specific exclusions. The frontend accepts some region forms that
embedded execution cannot represent; section 8 records this boundary rather
than treating all typed regions as portable.

Evidence: [experimental definitions][experimental], [typing rules][typing],
[embedded domain operations][array-field], [concat tests][concat-tests], and
[ADR 22][adr22].

### 5.5 Scans

A declaration

```python
@gtx.scan_operator(axis=K, forward=True, init=z)
def recurrence(carry: C, x: X) -> C: ...
```

defines a scalar recurrence lifted over a vertical axis `K`. `forward` is a
boolean, and `K` must have kind `VERTICAL`. The first parameter, return value,
and `init` have exactly matching scalar/product types. `C` may be a nested
tuple or supported named collection with scalar leaves. The body uses the
scalar DSL subset and may compose lowerable scalar operators; it is not an
arbitrary Python loop or a field remapping of the carry.

At invocation the caller supplies inputs after `carry`; the implementation
supplies the carry. Scalar pass parameters accept matching scalar/field leaves
at the field-level call. Invocation dimensions merge input dimensions and `K`.
Each scalar carry leaf becomes a field over that merged set, even if no input
contains `K`. Collection inputs and outputs participate as composite values.

For each orthogonal coordinate `p` and requested interval `[l,u)`:

```text
forward:  c(l-1) = z
          c(k)   = recurrence(c(k-1), x(p,k), …), k=l,…,u-1
          result(p,k) = c(k)

backward: c(u)   = z
          c(k)   = recurrence(c(k+1), x(p,k), …), k=u-1,…,l
          result(p,k) = c(k).
```

The scan is inclusive: it stores the updated carry. `init` resets independently
for every column and invocation, including an invocation on a subdomain. It is
not implicitly taken from a neighboring output value outside that subdomain.
The enclosing output/domain context supplies the scan interval.

Embedded scans require a single shared domain/column interval. Explicit
per-leaf domain tuples can fail, both at direct host entry and when called from
a program; use a shared domain or separate invocations (C2). Direct scan
operators can invoke/JIT but cannot be explicitly precompiled with `.compile()`.
For ahead-of-time compilation, wrap the scan in a program with concrete field
annotations and compile that program (section 7.4).

All scans in one lowered program must share the same scan axis. General
horizontal/local scans, per-point initial state, multiple scan axes, and
arbitrary nested scans are not a portable extension of this rule. Scalar-only
scans exist, but embedded execution excludes scan-without-field-input cases.
Iterator-level nested-scan tests do not establish Python DSL nesting support.

Evidence: [scan typing][typing] (`visit_ScanOperator`), [call rules][call-types]
(`return_type_scanop`, `_scan_param_promotion`), [embedded operators][operators],
[program lowering][p-lowering] (`_column_axis`), [scan tests][scan-tests],
[named-collection tests][named-tests], and [scan regression][scan-regression].

## 6. Programs, domains, and writes

For nonconflicting output storage, an operator call denotes evaluation followed
by assignment over the selected domain. If program state is `σ`, result is
`v`, output field is `o`, and selected domain is `R`:

```text
σ'(o,p) = v(p)     for p ∈ |R|
σ'(o,p) = σ(o,p)   for p outside |R|.
```

For products apply the rule to each output leaf and its selected domain.
Program statements execute in source order; a later statement observes writes
from earlier statements. Local field-operator assignments do not write host
fields. The equation does not resolve conflicting overlapping writes within
one call (section 6.3).

### 6.1 Explicit domains

Captured program `domain=` accepts a nonempty dictionary literal of dimensions
to integral `(start, stop)` expressions. It names absolute, half-open logical
ranges. Current lowering requires keys to match output dimensions **in order**;
omitted, extra, or reordered axes are not a supported shorthand. Runtime scalar
parameters can supply bounds; field bounds need not all be static compilation
constants.

A single outer domain dictionary may constrain all output leaves, or a domain
tuple can recursively match the output structure and supply different leaf
domains. Within nested structures, a dictionary is not a general substitute for
an entire nested tuple of domains. Embedded scans require a single shared domain
and do not support this per-leaf tuple workaround (section 5.5). Each write domain must be covered by its
output and by all input reads required by the expression, including shifted
and dynamically indexed coordinates.

### 6.2 Output slices and omitted domains

A program output may be a field name or a restricted relative slice of it.
Captured output slicing requires one slice per dimension with its step
omitted. Integer indexing, ellipsis, and explicit steps (including `::1`)
are not the output-domain syntax. For current output bounds `[a,b)`, a nonnegative bound `n` means `a+n`,
a negative bound means `b+n`, and an omitted bound selects the corresponding
endpoint. `domain=` cannot be combined with sliced output arguments.

Captured slice bounds must be integer literals (including supported negative
literal spellings) or omitted. Runtime names and arithmetic expressions such as
`1+1` are unsupported bounds. Use an explicit absolute domain for runtime bounds,
for example `domain={I: (0,n)}`; account for nonzero logical origins when
replacing a relative slice.

Without a domain restriction, lowered execution derives a domain independently
for each output leaf. Field-level embedded execution instead intersects all
output-leaf domains before assignment. Equal output domains give the same
choice; unequal extents can differ (C2). Equal domains avoid that difference;
explicit per-leaf domain tuples also do so for ordinary field operators.
Embedded scans instead require a single shared domain/column interval.

Output fields must match result leaf types without a general scalar-to-field result conversion. Lowering rejects local output dimensions, and sparse output tests
have backend exclusions. Host construction of such a field is not proof that
a portable program may write it.

Evidence: [program typing][p-typing], [program lowering][p-lowering]
(`_compute_field_slice`, `_visit_slice_bound`, `_visit_stencil_call_out_arg`),
[embedded assignment][operators] (`_get_out_domain`, `_tuple_assign_field`),
[domain tests][domain-tests], [program tests][program-tests], and
[multiple-output domains][multi-domain-tests].

### 6.3 Aliasing, boundaries, and execution obligations

Some in-place calls are established by tests, including pointwise output reuse
and scans writing an input. This does not specify every case of shifted reads
sharing output storage, intersecting views, or multiple aliased output leaves.
In particular, distinct allocations in tuple-output tests do not establish a
simultaneous in-place swap guarantee. The coherent mathematical interpretation
would evaluate right-hand sides from the call's input state, but full snapshot
semantics for arbitrary aliases is a recommendation, not a verified contract.
Use independent outputs or caller-managed temporaries where overlapping reads
and writes could affect results.

The caller supplies valid halos, valid connectivity entries, and sufficient
output coverage. The DSL does not perform mesh communication, time stepping,
physical boundary conditions, or dynamic allocation as implicit effects of an
operator. A domain restricts writes and determines required reads; it does not
turn an invalid read into a defined value. Error timing and out-of-bounds
behavior can differ between validation, embedded indexing, and native code;
no portable value is specified for invalid accesses.

Evidence: [scan tests][scan-tests] (`test_scalar_scan`),
[domain tests][domain-tests] (`test_single_value_field`),
[decorator tests][decorator-tests], and [multiple-output tests][multi-domain-tests].

## 7. Python interface

### 7.1 Construction and storage

| Host API                                           | Contract relevant to the DSL                                                                                                                                                |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `empty`, `zeros`, `ones`, `full`                   | Allocate fields on an explicit domain. Default dtype is float64 except `full`, which derives dtype from the fill value.                                                     |
| `as_field(domain_or_dimensions, data, ...)`        | Copy supported array data into new field storage, with optional dtype/device/allocator selection. There is no `copy=False` contract.                                        |
| `as_connectivity(..., codomain=d, skip_value=...)` | Copy integer index data and attach connectivity metadata. Default auto-detection selects `-1` when present, else no skip sentinel; explicit `None` disables auto-detection. |
| `Field.domain`, `.dtype`                           | Expose logical bounds/dimensions and dtype.                                                                                                                                 |
| `.ndarray`, `.asnumpy()`, `.as_scalar()`           | Host storage access, host NumPy conversion, and zero-dimensional scalar extraction.                                                                                         |
| Host field indexing and mutable-field assignment   | Relative/named/absolute views and storage updates according to the Field protocol; not unrestricted DSL subscript/store syntax.                                             |

With a dimension-only `as_field` argument, data shape supplies `[0,n)` bounds.
An origin `origin[d]=o` changes bounds to `[-o,n-o)`: array index `o` is logical
index zero. A concrete domain cannot be combined with a nonempty origin, and
its extents must match data shape. Default dtype follows input data; default
allocation follows its device, otherwise CPU. Namespace allocators such as
NumPy/CuPy/JAX and GT4Py allocator/backend objects are host configuration.
`aligned_index` is ignored with a warning for namespace allocation.

Constructors copy, but subsequent field slices may share underlying storage.
The general `as_connectivity` signature admits broader shapes and skip
sentinels than the portable compiled neighbor-table boundary. Current remapping
paths require `None` or `-1` skip metadata; constructing another sentinel does
not establish that it can execute (C10).

Evidence: [constructors], [Field protocols][model],
[constructor tests][constructor-tests], and [embedded field tests][array-tests].

### 7.2 Decoration, capture, and execution modes

`program`, `field_operator`, and `scan_operator` consume Python function objects.
They retrieve source and resolve annotations using Python introspection and
captured globals/nonlocals. Nonlocals override globals. Syntax/type validation
runs during frontend construction even for embedded execution. Arbitrary
dynamically generated functions without retrievable source are not a guaranteed
entry path.

Captured constants and qualified namespace values are folded for lowering;
enums and supported frozen namespaces have test coverage. Embedded execution
calls the original Python function and may see later changes to its environment,
whereas parsed/cached compiled stages retain captured values. Uniform closure
mutation/recompilation semantics are not specified. Treat closure values as
immutable configuration; pass changing values explicitly.

Immutability alone does not guarantee closure composition. Recursive closure
collection currently rejects a parent and referenced child that capture the
same name with unequal values, raising `NotImplementedError` during collection
or lowering even though Python gives them distinct lexical scopes. Use distinct
captured names or explicit parameters for such compositions. Evidence:
[recursive closure collection][transform] (`_get_closure_vars_recursively`) and
[program lowering][p-lowering].

`program` and `field_operator` accept backend/grid/compilation configuration
and can be used bare or with arguments. `scan_operator` requires its scan
configuration and accepts backend/grid selection, but does not expose the same
compilation-option keywords at decoration time; its resulting entry point can
use `.with_compilation_options`. An omitted backend uses `DEFAULT_BACKEND`,
initially `None`; explicit `None` chooses field-level embedded execution.
A backend value is a `Backend` object, not a backend-name string. Configured
copies are produced by `.with_backend`, `.with_grid_type`, and
`.with_compilation_options`.

| Execution selection                                   | Meaning                                                                                                                                |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `backend=None`                                        | Original Python field-operator/program execution over registered field implementations, typically NumPy/CuPy/JAX; no iterator lowering |
| `gtx.itir_python`                                     | Lower to iterator IR and execute the roundtrip Python backend                                                                          |
| `gtx.gtfn_cpu`, `gtx.gtfn_gpu`, other backend objects | Lower and run generated native/backend code subject to backend feature support                                                         |

An operator called inside a program/operator uses the enclosing execution
context; its own configured backend applies when it is the host entry point.
Field-level embedded execution is distinct from executing lowered iterator IR
in Python. Neither one is a universal oracle for every accepted construct.

Evidence: [decorators][decorator], [source capture][source],
[closure folding][closure-fold], [closure tests][closure-tests],
[public API][public], and [decorator tests][decorator-tests].

### 7.3 Calling convention

| Call site                                   | Required behavior                                                         |
| ------------------------------------------- | ------------------------------------------------------------------------- |
| Ordinary Python calls a program             | Supply typed parameter arguments; writes output arguments; returns `None` |
| Ordinary Python calls a field/scan operator | Supply `out=`; optional `domain=` controls writes; returns `None`         |
| A program calls a field/scan operator       | `out=` is required; optional domain/slice determines writes               |
| A field operator calls another operator     | Expression call returns a DSL value; no `out=`                            |

Omitting `offset_provider` at a host entry point becomes `{}`. Programs also
normalize explicit `None` to `{}`; a field/scan operator requires a mapping
when the keyword is explicitly supplied, so `None` is not equivalent there.
Named neighbor offsets require the necessary table mapping; Cartesian-only
computations can omit it. Nested calls inherit the provider from the execution
context. Omitting `out` on a host operator call is diagnosed even in embedded
mode; internal expression calls return values because they run inside an
existing context.

Arguments support positional/keyword binding for ordinary annotated parameters.
Captured calls validate names, count, dtype, dimensions, and product structure.
Host callers must also supply matching types, but direct field-level embedded
operator calls bypass some runtime type checks and may execute or cast invalid
arguments/results during assignment (C21).
`out` is a field or recursively matching output collection. The compiled frontend requires field result leaves matching the output
fields. Embedded assignment can store a scalar (for example the scalar
`broadcast` shortcut), but that is not a general scalar-return entry-point rule. Host domain arguments use domain conversion interfaces; the captured
program literal syntax remains the narrower rule of section 6.

Omit `domain` to infer writes from outputs. Explicit `domain=None` is treated
as omission only by field-level embedded operator execution; compiled operator
host calls pass it to domain conversion and reject it (C22).

Evidence: [decorators][decorator] (`Program.__call__`, `FieldOperator.__call__`),
[operator-to-program adaptation][fo-to-past], [embedded call wrapper][operators],
and [argument interface tests][arg-tests].

### 7.4 Specialization and compilation

`.compile(offset_provider=..., **static_arguments)` requires a non-None
backend and runtime neighbor-table mappings, or a list of such mappings. An
empty mapping suffices when no tables are used; connectivity configuration may
supply the mapping instead. It submits variants
and returns the same entry-point object, updating its compiled-variant pool.
Although the public annotation admits provider type descriptors, nonempty
descriptor-only mappings are currently rejected, including when supplied
through connectivity configuration (C20). Static
argument values are supplied as lists, producing combinations of variants.
`static_params` specializes selected scalar parameters; `static_domains`
specializes field bounds. Otherwise bounds may be runtime data.

Direct scan operators cannot be explicitly precompiled through this interface:
their generic invocation types are not supplied by `.compile()`. Wrap the scan
in a program with concrete field annotations and compile that program instead.
Direct scan invocation can still compile a variant through JIT (C20).

`enable_jit` controls compilation of a missing variant at invocation. Disabling
it requires a matching precompiled variant. The global `gtx.wait_for_compilation()` waits for
submitted compilation work. These host controls do not change the meaning of
a well-defined DSL expression.

`Program.with_bound_args(**kwargs)` creates a reduced interface with bound
scalar arguments. Unknown names and repeated bound arguments are errors;
explicit `.compile()` on `ProgramWithBoundArgs` is currently unimplemented.
Compilation options, caches, workers, allocator selection, and backend builders
are host infrastructure. Caller-owned DaCe workspace memory has backend-specific
lifetime/concurrency requirements, not a DSL allocation primitive.

Evidence: [decorators][decorator] (`compile`, `with_static_params`,
`with_bound_args`), [compilation options][options], [compiled pool][compiled],
[compiled-program tests][compiled-tests], [bound-argument tests][bound-tests],
[ADR 27][adr27], and [advanced guides][advanced].

### 7.5 Errors and conformance

Unsupported syntax, undeclared names, invalid annotations, incompatible types,
and invalid call signatures are DSL errors. The frontend uses `DSLError` and
subclasses with source locations and structured explanatory data. The exact
rendered message is a diagnostic interface, not a different language meaning.
Host construction/conversion errors can instead use Python exceptions; some
unsupported lower/backend paths still fail later with assertions or
`NotImplementedError`.

Frontend acceptance is necessary but does not establish backend conformance.
Feature tests run a backend matrix with explicit skips/expected failures.
Use [definitions.py][matrix] and the feature tests together. Iterator-only
features (`lift`, generic iterator reductions, nested iterator scans, and
internal runtime-domain primitives) are not Python DSL constructs simply
because they occur in the same test tree.

Evidence: [diagnostic guide][diagnostics], [diagnostic tests][diagnostic-tests],
[backend matrix][matrix], and [ADR 15][adr15].

## 8. Contradictions, interpretation, and unresolved contracts

This register separates disagreements from ordinary absence of support. The
recommendations are the most coherent interpretation of the collected evidence;
they are not implemented by this documentation change. “Portable” below means
avoiding the particular disagreement, not certification of every backend.

### C1. Historical frontend/domain API versus the accepted API

**Definitions:** [ADR 1][adr1] rejects an explicit field-operator `domain=`
interface in favor of output slicing and presents AST-root decorator entry
points. Current [decorators][decorator], [domain tests][domain-tests], and
[workshop material][slides] use Python-function decorators and explicit domains.
**Interpretation:** retain the ADR as architecture history; the current surface
includes explicit absolute domains and relative output slicing. Do not revive
its rejected syntax decision as a current prohibition.

### C2. Default domains of multiple outputs

**Definitions:** [ADR 10][adr10] requires same-shaped/same-sliced outputs and
infers a shared domain from the first field. Current [program lowering][p-lowering]
(`_visit_stencil_call_out_arg`) derives each output-leaf domain independently;
[multiple-output tests][multi-domain-tests] and [named outputs][named-tests]
accept distinct explicit leaf domains. [Embedded execution][operators]
(`_get_out_domain`) instead intersects all output domains when none is supplied.
**Observation:** the probe in section 9 produces different writes in embedded
and `itir_python` execution. **Interpretation:** per-leaf domains are the
coherent model consistent with the current multiple-output API. Use equal output
domains or, for ordinary field operators, explicit per-leaf domain tuples until default execution agrees;
the old blanket same-shape restriction is not current frontend law.
Embedded scans are an additional limitation: `_get_vertical_range` in
[embedded execution][operators] produces a tuple of ranges for a domain tuple,
while `ScanOperator.__call__` expects one range. Use a single shared
domain/column interval or separate invocations for embedded scans, including
scans called from a program; per-leaf tuples are not a universal workaround.

### C3. `where` dtype promotion

**Definitions:** [Quickstart], “Using conditionals on fields,”
says dimensions and dtype are promoted. [Type promotion][type-info] and
[frontend typing][typing] require matching corresponding leaf dtypes.
**Interpretation:** dimensions promote; dtypes do not implicitly widen. Use
`astype` explicitly. Product leaves may have different dtypes from one another,
but each true/false leaf pair must match; there is no single global product dtype.

### C4. Offset-provider values and optionality

**Definitions:** the history in [ADR 19][adr19] includes dimension-valued
Cartesian provider entries and older provider classes. The current
[model] defines provider elements as neighbor tables; dimension shifts
are self-describing. [Quickstart] says host operator calls require
both `out` and `offset_provider`, while [decorators][decorator] and
[call-interface tests][arg-tests] allow an omitted provider.
**Interpretation:** provide a tag-to-neighbor-table mapping when a named neighbor
lookup needs it; otherwise omission means `{}`. `out` remains required at host
operator entry points. Explicit `offset_provider=None` is accepted by programs,
while operator calls require a mapping when supplied. Old dimension-valued
provider examples do not establish a portable current API. Indexed single-target
`FieldOffset` is still accepted by [frontend typing][typing] and appears in
[diagnostic hints][diagnostic-tests], but [lowering][fo-lowering] emits a tagged
shift and [embedded offset handling][builtins] looks up a neighbor table. It is
not equivalent to a self-contained Cartesian shift: use `field(I+n)` or
`field(I-n)` for fixed shifts. Single-target declarations used by `as_offset`
remain a separate supported form.

### C5. `**` versus ordinary promotion and `power`

**Definitions:** [typing] returns the base type immediately for `**`,
while `power` follows common-dtype/dimension promotion. The parser test
`test_binary_pow` in [field parser tests][fo-parser-tests] deliberately allows
a floating field raised to an integral scalar exponent. A focused parser probe
also accepts `float64[I] ** int32[J]` and reports only `float64[I]`, losing the
exponent's extra dimension. **Interpretation:** preserve the tested floating
base/integral scalar exponent behavior. Additional field-exponent dimension
and compatibility checks are needed before a general field-exponent rule can
be promised; indiscriminately imposing equal dtypes on `**` would contradict
the intentional scalar exponent case.

### C6. Boolean negation across Python and lowered execution

**Definitions:** [typing/lowering][typing] accept field `not` and scalar/field
`~` as logical negation. [Unary tests][unary-tests] explicitly xfail field `not`
as accidental support that should be rejected. Python `not field` truth
conversion and Python `~True == -2` differ from elementwise/logical semantics.
The scalar-to-0d probe in section 9 reproduces the latter difference through
an accepted call. **Interpretation:** use `not` for scalar bool and `~` for
actual bool fields. Field `not` acceptance and Python-bool inversion must not
be used to infer a coherent cross-mode boolean contract.

### C7. Broadcast fields and scalar-to-zero-dimensional compatibility

**Definitions:** [ADR 13][adr13] describes annotation-driven implicit scalar
broadcast. [Call typing][call-types] implements scalar/0d compatibility, but
[embedded operators][operators] pass the original scalar to the Python body.
Separately, the scalar [broadcast implementation][builtins] returns its input
unchanged despite a statically field-valued result. Field-only behavior can
therefore differ even when simple arithmetic agrees.
**Interpretation:** retain the distinct scalar/field types and the explicit
compatibility rule. Abstract `broadcast` denotes a constant field; the scalar
embedded representation is a shortcut with observable limitations. Pass an
actual zero-dimensional field when field operations matter. A uniform wrapper
or representation-independent primitive semantics would be needed to turn
annotation compatibility into uniform runtime behavior.

### C8. Reduction-axis acceptance and sparse-only reductions

**Definitions:** [frontend typing][typing] checks that the requested axis
exists, whereas [embedded reductions][array-field] require it to be `LOCAL`,
and lowering constructs neighborhood-list reductions. The external-local-field
suite [explicitly skips standalone local-field reduction][local-tests].
**Interpretation:** specify neighbor reductions over a local axis with known
connectivity association, spelling `axis=` explicitly. Neither arbitrary
global reductions nor standalone sparse-only reductions follow from the weaker
static check. The positional-axis Python signature is also broader than the
frontend implementation, which accesses `kwargs['axis']`.

### C9. Reduction identity, skipped slots, and dtype

**Definitions:** [lowering][fo-lowering] seeds max/min with finite dtype
minimum/maximum (`arithmetic_bounds` in [type utilities][type-info]);
[embedded reduction][array-field] fills skipped slots with global input extrema.
All-skipped neighborhoods and all-`-inf` maximum/all-`+inf` minimum can therefore
differ. Array-library integer sum can widen while frontend result typing
preserves input dtype. **Interpretation:** normal nonempty finite neighborhoods
follow section 5.3. Recommend uniform typed identities (zero for sum,
`-∞`/`+∞` for floating max/min, integer extrema), an explicit empty-neighborhood
rule, and an explicit accumulator dtype. Current sources do not settle these
edge cases uniformly, including NaN behavior.

### C10. General skip-sentinel API versus executable remapping

**Definitions:** [constructors] and connectivity metadata accept
integral `skip_value` arguments; [embedded remapping][array-field] and
[FieldOffset handling][builtins] require `None` or `-1` on current paths.
A constructor/remapping probe accepts a table with `skip_value=-99`, then
raises `AssertionError` on remapping it. **Interpretation:** `-1` is the
portable missing-neighbor sentinel; the generic constructor signature is
broader than established execution support. Either execution should honor
metadata generally or construction should reject unsupported sentinels earlier.

### C11. Closure lifetime

**Definitions:** [source capture][source] and cached frontend stages retain
captured values for lowering; [embedded execution][operators] calls Python
functions that can observe changed globals/nonlocals. No common reparse or
invalidation rule establishes mutation semantics. **Interpretation:** closure
values should be immutable configuration; dynamic values are parameters.
There is also a concrete composition restriction independent of mutation:
[recursive closure collection][transform] rejects unequal same-named captures
in a parent and referenced child. A child capturing `factor=1` and a program
capturing `factor=2` can construct successfully, then fail lowering with
`NotImplementedError`. Distinct capture names or explicit parameters avoid that
restriction. Python lexical scope independence is therefore not fully preserved
by current recursive collection. The lifetime behavior remains unresolved;
no rule establishes that every mode snapshots every Python object.

### C12. Python declaration signatures

**Definitions:** Python functions can declare defaults, positional-only,
keyword-only, and variadic parameters; [frontend parsers][fo-parser] primarily
consume ordinary `node.args`, constructing positional-or-keyword signatures.
[Interface tests][arg-tests] validate permutations of ordinary supplied arguments,
not every Python declaration form. **Interpretation:** annotate ordinary
parameters and supply all values for portable declarations. Do not characterize
all other forms as consistently rejected at decoration time: some are omitted
or interpreted differently by the captured path versus Python invocation.

### C13. Exported dtypes versus conversion/backend coverage

**Definitions:** [exports][public], `ScalarKind`, and scalar classification
include small and unsigned integer widths; [type conversion helpers][type-translation]
`as_dtype`/`from_dtype` cover only bool/int32/int64/float32/float64.
**Interpretation:** type representation and export do not certify all execution,
allocator, or conversion paths. State the wider type universe, and require
backend evidence for the smaller/unsigned widths; do not label them categorically
unparseable. Plain integer values also use int32/int64 inference, while the
`int` annotation denotes int64: this distinction is current behavior, not an
implicit contextual conversion rule. Signed source expressions classify the
unsigned operand first, so source-negative dtype minima differ from host values
as described in section 3.1; the representable-value rule alone does not describe
every source spelling.

### C14. Numerical semantics

**Definitions:** [frontend typing][typing] keeps integer `/` integer, unlike
normal Python/array true division. [Basic tests][basic-tests] tolerate both
IEEE inf/NaN and `ZeroDivisionError`; [backend exclusions][matrix] cover floor
division and negative modulo. **Interpretation:** no uniform Python/NumPy
numerical contract can be inferred. A future specification should decide
integer division dtype, signed division/modulo, narrowing/overflow, exceptional
values, NaN/signed-zero selection, and reduction ordering explicitly. Until
then retain the static rules and label these runtime cases backend-dependent.

### C15. Dimension identity and domain-map ordering

**Definitions:** [ADR 26][adr26] describes dimensions as identified by name;
[Dimension equality][model] and [grid tests][dimension-tests] distinguish kind
as well. Domains are conceptually keyed by dimension, yet
[initialized-domain lowering][p-lowering] checks dictionary key order against
output dimensions. **Interpretation:** equality is `(name,kind)`. Keep the
current order constraint for accepted program domains, while recommending
key-based lookup as the coherent future map semantics. Backend-generated tags
may impose additional practical name constraints not established here.

### C16. Region algebra versus embedded interval representation

**Definitions:** [frontend region typing][typing] accepts some disjoint and
multidimensional expressions; [concat tests][concat-tests] exercise compiled
`!=` and mark embedded exclusions. Embedded dimension inequality rejects a
disjoint region; concat requires one-dimensional, contiguous representable
output ([ADR 22][adr22]). **Interpretation:** the abstract coordinate-region
algebra is wider than the common execution subset. Keep those distinctions
explicit; one-dimensional contiguous concatenations are the shared model,
not every well-typed region expression.

### C17. Backend selector examples and internal representation terminology

**Definitions:** some [decorator docstrings][decorator] show a string such as
`backend="roundtrip"`, but execution expects a backend object with transforms.
Historical tuple IR ADRs discuss equivalences involving tuples inside fields,
while current frontend values are products of scalar-valued fields.
**Interpretation:** use backend objects such as `gtx.itir_python`; IR tuple
representations do not create Python fields with arbitrary tuple-valued elements.
The same boundary excludes internal `get_domain_range`, `lift`, and generic
iterator primitives from this DSL specification.

### C18. Aliases and dimension-changing remaps

**Definitions:** existing [in-place tests][scan-tests] establish selected reuse
cases; the sources do not establish arbitrary overlapping-output or shifted
snapshot semantics. [Remap lowering][fo-lowering] visits multiple remaps, while
[type result deduction][type-info] bases substitution on the first remap on
some paths. Ordinary fixed Cartesian shifts leave dimensions unchanged, masking
this discrepancy; multi-axis staggering or extra-dimensional dynamic offsets
need more coverage. **Interpretation:** preserve tested in-place and fixed-shift
cases, but do not generalize to arbitrary aliases or dimension-changing remap
combinations. Recommend snapshot right-hand-side semantics and complete remap
dimension deduction as future rules backed by dedicated tests.

### C19. Named-collection declaration and conversion limits

**Definitions:** [type-translation diagnostics][type-translation] describe
restricted dataclasses without custom initialization; the structural hook in
[named collections][named] checks field metadata but does not directly prove
all custom-init exclusions. Locally defined collection classes are explicitly
xfailed in [named tests][named-tests] because lowering requires module-qualified
resolution. `where`/`concat_where` accept named products, whereas `astype`'s
frontend signature is limited to scalars, fields, and tuples.
**Interpretation:** use module-resolvable data-only named collections satisfying
the documented restrictions. Do not extend one builtin's structure support to
all others, or infer arbitrary dataclass support from incomplete rejection.

### C20. Compilation annotations versus executable precompilation

**Definitions:** [decorators][decorator] annotate provider arguments with runtime
tables or provider type descriptors. The [compiled pool][compiled]
(`_compile_variant`, `compile`) explicitly rejects nonempty descriptor-only mappings.
It also rejects direct scan precompilation without argument types; the public
`.compile()` path supplies none, and [compiled-program tests][compiled-tests]
explicitly xfail scan precompilation. **Interpretation:** use runtime table
mappings (or `{}` when no tables are needed), including configured connectivities.
Compile a concretely annotated program wrapping a scan for ahead-of-time use;
direct scan invocation/JIT remains available. Broader annotations or inherited
methods do not establish implementation support.

### C21. Captured type rules versus embedded host-call validation

**Definitions:** [frontend typing][typing] enforces captured-call compatibility,
but direct embedded [decorator calls][decorator] pass actual values to the
original Python function. [Embedded assignment][operators] does not consistently
check collection types and can cast values into output storage. Program host
calls have a separate debug validation path. **Interpretation:** exact matching
remains a caller obligation; accidental embedded acceptance is not implicit
coercion or a supported relaxation of the DSL type rules. Do not promise uniform
runtime rejection or error timing for invalid direct host calls.

### C22. Omitted versus explicit None operator domains

**Definitions:** [embedded calls][operators] treat `domain=None` like omission.
Compiled host calls in [decorators][decorator] convert any explicitly supplied
domain through [domain conversion][model], which rejects `None`.
**Interpretation:** omit the keyword to request output-derived domains.
Document explicit `None` as an embedded-only convenience, not a common default
value accepted by all operator entry points.

## 9. Evidence coverage and focused observations

### 9.1 Documentation and examples reviewed

| Repository source                                            | Knowledge used / boundary                                                                                                               |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| [Quickstart]                                                 | Fields, pure operators, imperative programs, connectivities, local reductions, where, tuples, pseudo-Laplacian; wording conflicts C3/C4 |
| [WorkflowPatterns][advanced], [HackTheToolchain][hack]       | Toolchain stages, configurable/cached workflows, builders; host infrastructure rather than new DSL syntax                               |
| [Workshop slides 1–4][slides]                                | Fields/decorators/programs; Cartesian and neighbor shifts/reductions; where/domains; scan recurrence and composition                    |
| [Workshop exercises and solutions][exercises] 1–5            | Addition, divergence, gradient, curl, vector Laplace; array/field conversion, orientation/local coefficients, provider wiring           |
| [Workshop exercises and solutions][exercises] 6–8            | Where/domain halos, tuple scan carry, diffusion composition; exercises intentionally incomplete, solutions supply intent                |
| [Workshop helpers][helpers], [Jupyter introduction][jupyter] | Host mesh/allocation helpers and notebook usage; no additional language productions                                                     |
| [Cartesian/Next Laplacian comparison][lap-example]           | Named Cartesian shifts and storage interoperability in the example, not syntax compatibility between the two DSLs                       |
| [Diagnostic guide][diagnostics]                              | Error vocabulary, rejected-construct hints, source-location expectations                                                                |

All 21 Next-relevant notebooks (workshop slides, exercises/solutions, Jupyter
introduction, and the Laplacian comparison) had cell sources inspected. They
were not rerun in full. `examples/cartesian/*` belongs to the separate Cartesian
DSL and is not normative evidence for Next syntax.

The [Next ADR collection][adrs] was reviewed as follows:

- **Direct semantic evidence:** 0001 (frontend), 0010 (domains), 0013 (scalar/0d),
  0019 (connectivities), 0020 (runtime domains), 0022 (embedded concat limits),
  0026 (staggering), cross-checked against current implementation/tests.
- **Internal-language history:** 0002 (deprecated lowering), 0003/0004 (tuple
  representation), 0005 (IR extensions), 0006/0008 (C++ indices/domains),
  0007 (deprecated processors), 0009 (superseded backend integration). Their
  primitives and historical restrictions do not enlarge/restrict the current
  Python language without current evidence.
- **Execution/host boundary:** 0011/0012 (OTF), 0014 (DaCe), 0015 (test
  exclusions), 0016/0017 (backend/configuration), 0018 (SDFG transforms),
  0021 (argument descriptors), 0023 (fingerprints), 0024 (runners),
  0025 (cache), 0027 (workspace), 0028 (builders). Prototype restrictions
  there are not treated as current universal frontend limitations.

### 9.2 Implementation and test coverage

| Topic                                | Primary implementation                                                                | Representative tests                                                                                                                                                |
| ------------------------------------ | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Syntax, binding, errors              | [Parsers][fo-parser], [program parser][p-parser], [dialect], [AST passes][ast-passes] | [Field parser][fo-parser-tests], [program parser][p-parser-tests], [diagnostics][diagnostic-tests]                                                                  |
| Types, literal inference, dimensions | [Types], [translation][type-translation], [promotion][type-info], [model]             | [Typing][typing-tests], [model][model-tests], [dimension/grid][dimension-tests]                                                                                     |
| Operators, conditionals, conversions | [Builtin signatures][builtins], [typing], [lowering][fo-lowering]                     | [Math][math-tests], [unary][unary-tests], [basic][basic-tests], [conditionals][conditional-tests], [conversions][conversion-tests]                                  |
| Tuples, named collections, closures  | [Products][named], [unpacking][unpack], [closure folding][closure-fold]               | [Tuples][tuple-tests], [named products][named-tests], [closure/import][closure-tests]                                                                               |
| Broadcast, where, region selection   | [Typing], [experimental], [embedded field][array-field]                               | [Broadcast][broadcast-tests], [where][where-tests], [concat][concat-tests]                                                                                          |
| Remaps, staggering, dynamic offsets  | [Connectivity model][model], [result typing][type-info], [lowering][fo-lowering]      | [Cartesian shifts][shift-tests], [staggering][staggered-tests], [offset regression][offset-regression]                                                              |
| Local fields and reductions          | [Typing], [embedded reductions][array-field], [lowering][fo-lowering]                 | [Reductions][reduction-tests], [external local fields][local-tests]                                                                                                 |
| Scan recurrence/composition          | [Scan types][call-types], [embedded scans][operators], [program lowering][p-lowering] | [Scans][scan-tests], [ICON-like scan][icon-scan], [scan-init regression][scan-regression]                                                                           |
| Domains, sequential writes           | [Program typing][p-typing], [lowering][p-lowering], [embedded assignment][operators]  | [Domains][domain-tests], [programs][program-tests], [multiple outputs][multi-domain-tests]                                                                          |
| Host construction/invocation         | [Constructors], [decorators][decorator], [compiled pool][compiled]                    | [Constructors][constructor-tests], [embedded fields][array-tests], [call interface][arg-tests], [bound arguments][bound-tests], [compiled variants][compiled-tests] |
| End-to-end compositions              | Frontend/embedded/lowering layers above                                               | [Laplacian, FVM, scan, and multiple-output examples][multi-tests]                                                                                                   |
| Backend support limits               | [Test exclusion definitions][matrix]                                                  | Feature markers plus per-backend skips/xfails, read with each test                                                                                                  |

This index records investigation coverage, not a claim that every test in these
trees was executed. Low-level iterator tests were consulted only as supporting
implementation evidence; their complete internal language is out of scope.

### 9.3 Focused cross-mode and frontend probes

The following small cases were run against this checkout with `uv`, Python
3.12.13, NumPy embedded fields, and the `itir_python` lowered backend. Native
C++/GPU/DaCe variants were not exercised. No runtime code was changed.

For default tuple-output domains, the essential case is:

```python
import numpy as np
import gt4py.next as gtx

I = gtx.Dimension("I")


@gtx.field_operator
def two(a: gtx.Field[gtx.Dims[I], gtx.int32]):
    return a + 1, a + 2


if __name__ == "__main__":
    for backend in (None, gtx.itir_python):
        a = gtx.as_field([I], np.arange(5, dtype=np.int32))
        x = gtx.as_field([I], np.full(5, -7, dtype=np.int32))
        y = gtx.as_field([I], np.full(3, -7, dtype=np.int32))
        two.with_backend(backend)(a, out=(x, y))
        print(x.asnumpy(), y.asnumpy())
```

| Mode          | `x` after call      | `y` after call |
| ------------- | ------------------- | -------------- |
| Embedded      | `[1, 2, 3, -7, -7]` | `[2, 3, 4]`    |
| `itir_python` | `[1, 2, 3, 4, 5]`   | `[2, 3, 4]`    |

A second operator demonstrates scalar/0d compatibility without uniform wrapping:

```python
@gtx.field_operator
def complement(x: gtx.Field[gtx.Dims[()], bool]) -> gtx.Field[gtx.Dims[()], bool]:
    return ~x


# In the guarded entry point above, for each backend:
# out = gtx.as_field([], np.array(False))
# complement.with_backend(backend)(True, out=out)
```

Embedded assignment produces `True`, whereas `itir_python` produces `False`.
Passing an actual bool Field avoids the Python-scalar inversion involved here.
A third probe constructs a neighbor table with entries `[[0,-99],[1,2]]` and
`skip_value=-99`: construction succeeds, then direct embedded remapping raises
`AssertionError`.

Five focused parser/lowering probes additionally established:

- `float64[I] ** int32[J]` is accepted with deduced result `float64[I]`;
- floating field to integral scalar exponent is accepted;
- `float64("inf")` is accepted and lowered;
- integer field `/ 2` retains integer inferred dtype;
- augmented assignment (`+=`) is rejected as unsupported syntax.

These observations substantiate the conflict register; they are not an
all-backend conformance suite. The remaining gaps are explicit in sections
6.3 and 8: arbitrary aliasing, extreme/empty reductions, complete numeric edge
semantics, closure invalidation, richer signatures, and dimension-changing
remap combinations.

[adr1]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/development/ADRs/next/0001-Field_View_Frontend_Design.md
[adr10]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/development/ADRs/next/0010-Domain_in_Field_View.md
[adr13]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/development/ADRs/next/0013-Scalar_vs_0d_Fields.md
[adr15]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/development/ADRs/next/0015-Test_Exclusion_Matrices.md
[adr19]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/development/ADRs/next/0019-Connectivities.md
[adr22]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/development/ADRs/next/0022-Limitations-of-embedded-concat_where.md
[adr26]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/development/ADRs/next/0026-Staggered_Dimensions.md
[adr27]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/development/ADRs/next/0027-External_Workspace_Memory.md
[adrs]: https://github.com/GridTools/gt4py/tree/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/development/ADRs/next
[advanced]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/user/next/advanced/WorkflowPatterns.md
[arg-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_arg_call_interface.py
[array-field]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/embedded/nd_array_field.py
[array-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/unit_tests/embedded_tests/test_nd_array_field.py
[ast-passes]: https://github.com/GridTools/gt4py/tree/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/ast_passes
[basic-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_basic.py
[bound-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_bound_args.py
[broadcast-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_broadcast.py
[builtins]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/fbuiltins.py
[call-types]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/type_info.py
[closure-fold]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/foast_passes/closure_var_folding.py
[closure-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_closure_vars.py
[compiled]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/otf/compiled_program.py
[compiled-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_compiled_program.py
[concat-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_concat_where.py
[conditional-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_conditionals.py
[constructor-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/unit_tests/test_constructors.py
[constructors]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/constructors.py
[conversion-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_type_conversion.py
[decorator]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/decorator.py
[decorator-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_decorator.py
[diagnostic-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/unit_tests/ffront_tests/test_diagnostic_messages.py
[diagnostics]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/development/next/error-messages.md
[dialect]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/dialect_parser.py
[dimension-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/unit_tests/ffront_tests/test_decorator_domain_deduction.py
[domain-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_domain.py
[exercises]: https://github.com/GridTools/gt4py/tree/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/user/next/workshop/exercises
[experimental]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/experimental.py
[fo-lowering]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/foast_to_gtir.py
[fo-parser]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/func_to_foast.py
[fo-parser-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/unit_tests/ffront_tests/test_func_to_foast.py
[fo-to-past]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/foast_to_past.py
[hack]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/user/next/advanced/HackTheToolchain.md
[helpers]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/user/next/workshop/exercises/helpers.py
[icon-scan]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/multi_feature_tests/ffront_tests/test_icon_like_scan.py
[import-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_import_from_mod.py
[jupyter]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/user/next/workshop/jupyter_intro.ipynb
[lap-example]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/examples/lap_cartesian_vs_next.ipynb
[local-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_external_local_field.py
[math-data]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/math_builtin_test_data.py
[math-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_math_builtin_execution.py
[matrix]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/definitions.py
[model]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/common.py
[model-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/unit_tests/test_common.py
[multi-domain-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/multi_feature_tests/ffront_tests/test_multiple_output_domains.py
[multi-tests]: https://github.com/GridTools/gt4py/tree/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/multi_feature_tests/ffront_tests
[named]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/named_collections.py
[named-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_named_collections.py
[offset-regression]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/regression_tests/ffront_tests/test_offset_dimensions_names.py
[operators]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/embedded/operators.py
[options]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/otf/options.py
[p-lowering]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/past_to_itir.py
[p-parser]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/func_to_past.py
[p-parser-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/unit_tests/ffront_tests/test_func_to_past.py
[p-typing]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/past_passes/type_deduction.py
[program-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_program.py
[public]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/__init__.py
[quickstart]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/user/next/QuickstartGuide.md
[reduction-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_reductions.py
[scalar-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_scalar_args.py
[scan-regression]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/regression_tests/ffront_tests/test_cse_extraction_scan_init.py
[scan-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_scan.py
[shift-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_cartesian_shifts.py
[slides]: https://github.com/GridTools/gt4py/tree/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/user/next/workshop/slides
[source]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/source_utils.py
[staggered-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_staggered.py
[transform]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/transform_utils.py
[tuple-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_tuples.py
[type-info]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/type_system/type_info.py
[type-translation]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/type_system/type_translation.py
[types]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/type_system/type_specifications.py
[typing]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/foast_passes/type_deduction.py
[typing-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/unit_tests/ffront_tests/test_type_deduction.py
[unary-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_math_unary_builtins.py
[unpack]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/foast_passes/iterable_unpack.py
[where-tests]: https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests/test_where.py
