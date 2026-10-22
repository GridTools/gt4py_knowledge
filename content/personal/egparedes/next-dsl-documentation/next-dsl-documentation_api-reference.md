---
title: GT4Py Next DSL API reference
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
>
> **Known corrections.** Read [[personal/egparedes/next-dsl-documentation/next-dsl-documentation#errata-for-this-baseline|baseline errata]] first.
> The text below is a preserved snapshot; the errata take precedence.

# gt4py.next DSL API reference

This page describes the Python-facing computation API of `gt4py.next` at
revision `f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f`. It is organized as an API
reference, following the category indexes and per-operation entries used by the
[Python Array API standard](https://data-apis.org/array-api/latest/API_specification/index.html).
The requirements and behavior described here are GT4Py's own; this page does
not claim Array API conformance.

The companion [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification|DSL specification]] contains the formal
language model, repository evidence, and conflict analysis. This reference
presents the supported call contracts. **Current limitations** identify observed
restrictions or differences between execution modes; a frontend accepting an
expression does not establish support on every backend. Proposed resolutions
in the companion's conflict register are not additional implemented APIs.

**Interpretation used** notes link affected contracts directly to the relevant
issue in [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#8-contradictions-interpretation-and-unresolved-contracts|Section 8 of the DSL specification]]
and state the interpretation adopted here. Notes in a category introduction
apply to every entry in that category.

## API index

| Category                                                  | Objects and operations                                                                                              |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| [Conventions](#conventions)                               | Namespaces, signatures, type notation, execution contexts                                                           |
| [Data types and promotion](#data-types-and-promotion)     | `bool`, signed/unsigned integer types, floating types, `int`, `float`, `IndexType`, tuples, named collections       |
| [Dimensions and domains](#dimensions-and-domains)         | `Dimension`, `DimensionKind`, `Dims`, `UnitRange`, `unit_range`, `Domain`, `domain`, `GridType`, staggering helpers |
| [Field object](#field-object)                             | Attributes, `asnumpy`, `as_scalar`, operators, indexing, remapping                                                  |
| [Broadcasting](#broadcasting)                             | Dimension alignment, domain intersection, scalar expansion                                                          |
| [Creation functions](#creation-functions)                 | `empty`, `zeros`, `ones`, `full`, `as_field`, `as_connectivity`, `FieldConstructor`                                 |
| [Elementwise functions](#elementwise-functions)           | Arithmetic, trigonometric, hyperbolic, exponential, rounding, and predicate functions                               |
| [Conversion and selection](#conversion-and-selection)     | `astype`, `broadcast`, `where`                                                                                      |
| [Connectivity and remapping](#connectivity-and-remapping) | `Connectivity`, `CartesianConnectivity`, `FieldOffset`, Cartesian shifts, neighbor access                           |
| [Neighbor reductions](#neighbor-reductions)               | `neighbor_sum`, `max_over`, `min_over`                                                                              |
| [Computation definitions](#computation-definitions)       | `field_operator`, `scan_operator`, `program`, permitted statements                                                  |
| [Invocation and execution](#invocation-and-execution)     | `out`, `domain`, `offset_provider`, output slices, backend selection, compilation methods                           |
| [Experimental functions](#experimental-functions)         | `as_offset`, `concat_where`                                                                                         |
| [Errors and support](#errors-and-support)                 | Validation, numerical limits, backend restrictions, source references                                               |

## Conventions

### Namespaces

Public names below are available from `gt4py.next` unless explicitly qualified.
Host code commonly uses `import gt4py.next as gtx`. Inside a captured computation,
callable names must be directly bound: import `where`, for example, outside the
function and call `where(...)` inside it. A captured call to `gtx.where(...)`
is currently rejected. Supported qualified references to dimensions, offsets,
and constants have separate resolution rules.

`as_offset` and `concat_where` are available from
`gt4py.next.ffront.experimental`. Python allocation, object construction,
backend selection, and compilation control occur outside captured bodies.

### Signatures and type notation

`/` marks positional-only parameters; `*` marks keyword-only parameters. The
signatures retain GT4Py parameter names and ordering, rather than substituting
Array API signatures. Some captured call forms impose extra restrictions, noted
on their entries. Displayed generic types are documentation notation:

| Notation            | Meaning                                                                           |
| ------------------- | --------------------------------------------------------------------------------- |
| `Scalar`            | A supported scalar value with a concrete dtype                                    |
| `Field[D, T]`       | A field with ordered dimensions `D` and scalar dtype `T`                          |
| `Number`            | Arithmetic scalar or field; excludes boolean operands                             |
| `Floating`          | Float32/float64 scalar or field                                                   |
| `Boolean`           | Boolean scalar or field                                                           |
| `TupleValue`        | A fixed, recursively nested tuple of supported values                             |
| `Product`           | A supported tuple or named collection, recursively containing scalar/field leaves |
| `DomainLike`        | A `Domain`, mapping of dimensions to ranges, or sequence of dimension/range pairs |
| `Output`            | Field or nested output collection matching an operator's field result             |
| `OffsetProvider`    | Mapping from offset tags to runtime neighbor tables                               |
| `<default-backend>` | Omission uses the configured default backend, initially `None`                    |
| `<auto>`            | Omit the argument to request automatic detection; not an exported value           |

`float64` shown as an allocation default denotes the equivalent dtype default.
Call forms using `*args` and `**kwargs` document an entry point accepting the
parameters of the wrapped definition; they do not permit variadic parameter
declarations inside the DSL.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c12-python-declaration-signatures|C12]]).** Declaration signatures use annotated ordinary positional-or-keyword parameters with all values supplied. Other Python declaration forms are not promised to work or to be rejected uniformly; wrapper call signatures do not enlarge this subset.

### Common requirements

1. Field dimensions identify logical coordinates. Array-axis positions and
   physical memory strides do not determine dimension identity.
2. Arithmetic operands normally have matching dtypes. Dimension broadcasting
   does not perform implicit dtype conversion.
3. Inputs contain every coordinate read, including shifted and dynamically
   selected coordinates. Callers supply halos and valid connectivity indices.
4. Output fields contain every coordinate written. Values outside the selected
   output domain remain unchanged.
5. Unless stated otherwise, elementwise functions preserve input dtype and
   align dimensions by the [broadcasting rules](#broadcasting).
6. Floating-point accuracy, overflow, exceptional arithmetic, and reduction
   order have no complete common-backend contract. An entry's mathematical
   description specifies the operation; it does not promise bitwise identical
   output or Array API special-case rules.

## Data types and promotion

### Scalar types

| Name                                  | Category               | Meaning                                                   |
| ------------------------------------- | ---------------------- | --------------------------------------------------------- |
| `bool`                                | Boolean                | Logical false or true; distinct from arithmetic integers  |
| `int8`, `int16`, `int32`, `int64`     | Signed integer         | Signed integer dtype of the indicated width               |
| `uint8`, `uint16`, `uint32`, `uint64` | Unsigned integer       | Unsigned integer dtype of the indicated width             |
| `float32`, `float64`                  | Floating point         | Floating dtype of the indicated width                     |
| `int`                                 | Python alias spelling  | Maps to `int64` as an annotation/constructor              |
| `float`                               | Python alias spelling  | Maps to `float64` as an annotation/constructor            |
| `IndexType`                           | Index annotation alias | Alias of `int32`; constructor alias support is incomplete |

**Current limitations.** All listed sized types are exported and represented
in the type system. Some conversion and execution paths support only `bool`,
`int32`, `int64`, `float32`, and `float64`; smaller/unsigned widths require
backend-specific validation. Complex, object, and general string field
arithmetic are not established DSL operations.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c13-exported-dtypes-versus-conversionbackend-coverage|C13]]).** The exported type universe is documented separately from execution coverage. Small and unsigned widths require evidence for the selected conversion, allocation, and backend paths.

### Scalar construction and literals

```text
T(literal) -> scalar of dtype T
```

**Parameters.** `T` is a supported scalar constructor. `literal` is a literal
value accepted by that constructor. Typed string literals such as
`float64("inf")` are supported; general string computation is not implied.

**Returns.** A scalar with the requested dtype.

**Notes.** Use [astype](#astype) for a computed value. Constructor aliases do
not establish an unrestricted alternative to literal-only conversion.
A decimal floating literal has dtype float64. An ordinary Python integer value
is classified as int32 when representable, otherwise int64. Source constants
are classified before unary signs: `-2147483648` consequently has type int64,
and `-9223372036854775808` is rejected because its positive operand exceeds
int64. NumPy scalar constants retain their dtype.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c13-exported-dtypes-versus-conversionbackend-coverage|C13]]).** Integer value inference, the `int` annotation, and source-literal parsing retain their distinct current rules. An annotation does not contextually convert a literal, and a negative source literal is classified through its positive operand first.

### Type promotion rules

For ordinary arithmetic, both scalar dtypes must be equal. Scalars expand over
field dimensions; fields combine dimensions according to the broadcasting
rules. An arithmetic result has that same dtype. Comparisons instead produce
bool. There is no automatic int-to-float or float32-to-float64 conversion.

Thus `int32_field + 1` is type-compatible, but `int64_field + 1` requires an
explicit int64 literal or conversion. The `**` operator has a separate result
rule, described under [field operators](#field-operators).

Captured call parameters match annotated dtypes, dimensions, and product structure.
An expression's dimension broadcasting does not make field annotations freely
interchangeable. A scalar can match a zero-dimensional field parameter of the
same dtype, but field-only operations can behave differently in embedded
execution because the scalar is not always wrapped. Supply an actual
zero-dimensional field when field behavior matters.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c3-where-dtype-promotion|C3]], [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c5--versus-ordinary-promotion-and-power|C5]]).** Dimension promotion does not imply dtype widening, including in `where`. The tested floating-base/integral-scalar-exponent case for `**` remains an exception to ordinary matching-dtype rules; `power` follows the ordinary rules.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c7-broadcast-fields-and-scalar-to-zero-dimensional-compatibility|C7]]).** Scalar/zero-dimensional argument compatibility is retained without promising runtime wrapping. Use an actual zero-dimensional field when field operations are required.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c21-captured-type-rules-versus-embedded-host-call-validation|C21]]).** Host callers must also supply matching types. Direct embedded operator calls do not enforce all runtime dtype, dimension, or nominal-collection checks; accidental acceptance or assignment casts do not establish supported coercion.

### Tuples and named collections

`tuple[T0, T1, ...]` describes a fixed product; the literal `(a, b)` constructs
one in a computation. Products may be nested. They are tuples of field/scalar
values, rather than fields whose elements are arbitrary Python tuples.

Typed named tuples and supported dataclasses/custom named collections preserve
member names and their original Python type. Use module-resolvable,
data-only declarations. Dataclasses have restrictions including nonempty
members and no default-valued fields, `InitVar`, or custom initialization as
specified by their validation contract. Locally defined collection classes
have known lowering limitations.

Tuple indexing uses a literal integer, including a negative index. Unpacking
supports one starred target, whose result is a tuple. Named members are accessed
as product projections. `where` and `concat_where` support matching named
products. `astype` supports recursive tuples without equivalent named-product
frontend support. Scan calls have special scalar-to-field product matching.

See [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#3-static-types|static types]] and
[[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c19-named-collection-declaration-and-conversion-limits|collection limitations]].

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c17-backend-selector-examples-and-internal-representation-terminology|C17]], [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c19-named-collection-declaration-and-conversion-limits|C19]]).** Products contain scalar values or scalar-valued fields; internal IR tuple representations do not add tuple-valued field elements. Named collections use module-resolvable, data-only declarations satisfying the documented restrictions, and support is stated per builtin. Incomplete validation does not establish broader dataclass support.

## Dimensions and domains

### Dimension

```text
Dimension(value: str, kind: DimensionKind = DimensionKind.HORIZONTAL) -> Dimension
```

Constructs an immutable dimension label.

**Parameters.** `value` is its name; `kind` identifies horizontal locations,
local neighbor slots, or a vertical axis.

**Returns.** A dimension whose equality is determined by both `value` and
`kind`, independent of Python object identity.

**Operations.** `d(i)` forms a named absolute index. `d + n` and `d - n` form
Cartesian remappings. Dimension/index comparisons form coordinate regions for
`concat_where`; they do not construct general numerical index fields.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c15-dimension-identity-and-domain-map-ordering|C15]]).** Dimension identity is the pair `(value, kind)`, following current equality rather than the historical name-only description.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c16-region-algebra-versus-embedded-interval-representation|C16]]).** Region expressions describe coordinate sets, but the shared `concat_where` contract is one-dimensional with a contiguous result. Broader typed region algebra does not imply embedded support.

### DimensionKind

The enumeration members are `HORIZONTAL`, `LOCAL`, and `VERTICAL`.
Canonical field dimension order is horizontal, local, vertical, then
lexicographic base name within each kind. Field/DSL checks reject duplicate
dimensions, multiple local dimensions, and a base dimension together with its
staggered counterpart. Host `Domain` construction has weaker checks.

### Dims

```text
Field[Dims[I, J, K], float64]
Field[Dims[()], float64]
```

Supplies the ordered dimensions of a field annotation. `Dims[()]` denotes a
zero-dimensional field. The legacy spelling `Field[[I, J, K], float64]` also
appears in the API. Bounds are runtime properties, not annotation parameters.

### UnitRange and unit_range

```text
UnitRange(start: integer, stop: integer) -> UnitRange
unit_range(r: RangeLike) -> UnitRange
```

**Parameters.** `start` and `stop` are absolute integer bounds. `r` can be an
existing range object, a unit-step Python range, a bound pair, or an integer
extent. An extent `n` denotes `[0,n)`. `None` denotes an unbounded range;
`None` in a bound pair denotes the corresponding unbounded endpoint.

**Returns.** The half-open range `[start,stop)` with step one. Empty ranges
normalize to `[0,0)`. Semantic unbounded ranges can be represented internally;
material allocation requires finite extents.

**Attributes.** `.start`, `.stop` expose bounds. Python `len` requires finite
bounds. UnitRange does not describe strided computation domains.

### Domain and domain

```text
Domain(*named_ranges, dims=None, ranges=None) -> Domain
domain(domain_like: DomainLike) -> Domain
```

**Parameters.** Construct `Domain` from named ranges, or provide both `dims`
and `ranges` of equal length. Use `domain` to convert convenient mappings and
sequences; for example `domain({I: (0, 8), K: range(2, 5)})`.

**Returns.** An ordered product of named unit ranges. `.dims`, `.ranges`,
`.shape`, and `.ndim` expose its layout and extents. A domain with no axes has
one empty coordinate; any empty constituent interval makes a domain empty.

**Operations.** Domain intersection aligns dimensions and intersects shared
ranges. A missing dimension is treated as unbounded. Domains use absolute
coordinates; output slices use relative bounds.

**Errors.** Duplicate dimensions, mismatched dimension/range counts, or invalid
range representations are not valid domains. Do not mix positional named ranges
with `dims`/`ranges` construction.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c15-dimension-identity-and-domain-map-ordering|C15]]).** Domains retain their ordered dimensions. Captured program domain dictionaries must follow output-dimension order; the proposed order-independent map interpretation is not used as current behavior.

### GridType

`GridType.CARTESIAN` selects Cartesian constraints;
`GridType.UNSTRUCTURED` permits unstructured neighbor relations and
Cartesian-like calculations. Omission on a computation definition requests
grid inference from dimensions, offsets, and referenced operators. Explicit
Cartesian selection rejects an unstructured requirement.

### flip_staggered

```text
flip_staggered(dim: Dimension) -> Dimension
```

**Parameters.** `dim` is a base or staggered dimension.

**Returns.** Its counterpart, preserving kind. Applying the function twice
recovers an equal dimension. For nonstaggered `I`, index `i` on
`flip_staggered(I)` has position `i - 1/2` in base-grid coordinates.

### is_staggered

```text
is_staggered(dim: Dimension) -> bool
```

**Parameters.** `dim` is a dimension.

**Returns.** Whether it is a staggered dimension. Use this helper rather than
depending on the internal name encoding.

### as_non_staggered

```text
as_non_staggered(dim: Dimension) -> Dimension
```

**Parameters.** `dim` is a dimension.

**Returns.** Its base dimension, or `dim` itself when already nonstaggered.

Source: [common.py](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/common.py).

## Field object

### Field

```text
Field[Dims[d0, ...], T]
```

A field represents scalar values of dtype `T` on a domain of named coordinates.
`Field` is a protocol and annotation, not the storage allocation constructor;
use the [creation functions](#creation-functions) to obtain array-backed fields.
A scalar and a zero-dimensional field remain distinct values.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c7-broadcast-fields-and-scalar-to-zero-dimensional-compatibility|C7]], [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c17-backend-selector-examples-and-internal-representation-terminology|C17]]).** A scalar and a zero-dimensional field are distinct, even where argument matching permits substitution. Field elements are scalar; tuple results are products of fields or scalars.

### Attributes

| Attribute        | Meaning                                       | Availability                                                         |
| ---------------- | --------------------------------------------- | -------------------------------------------------------------------- |
| `field.domain`   | Logical bounds and dimensions                 | Host interface                                                       |
| `field.dtype`    | Dtype descriptor, including its scalar type   | Host interface                                                       |
| `field.ndarray`  | Backing array for array-backed fields         | Host interface; not guaranteed for every abstract field/connectivity |
| `field.codomain` | Value type, or a dimension for a connectivity | Host interface                                                       |

Host properties and methods do not automatically become captured DSL expressions.
Storage layout and device selection are independent of logical dimension labels.

### asnumpy

```text
field.asnumpy() -> numpy.ndarray
```

**Returns.** The field's values as a NumPy array on the host. This is an
interoperability method, not a captured computation. Do not infer a universal
copy/alias guarantee from the return type.

### as_scalar

```text
field.as_scalar() -> Scalar
```

**Returns.** The value of a zero-dimensional field.

**Errors.** Extraction requires a zero-dimensional field. A nonzero-dimensional
field with a single element is not thereby a scalar. Python truth testing of a
Field raises a truth-value error; use explicit scalar extraction on the host
or `where` for field selection.

### Field operators

The table describes the captured DSL operators. Their scalar/field dimensions
follow the broadcasting rules unless noted.

| Expression                                               | Accepted operands                               | Result                                                                                |
| -------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------- |
| `+x`, `-x`                                               | Arithmetic scalar/field                         | Same type; identity or negation                                                       |
| `x + y`, `x - y`, `x * y`                                | Matching arithmetic dtypes                      | Same dtype, merged dimensions                                                         |
| `x / y`, `x // y`                                        | Matching arithmetic dtypes                      | Same inferred dtype, merged dimensions; numerical caveats below                       |
| `x % y`                                                  | Matching integral dtypes                        | Same dtype, merged dimensions                                                         |
| `x ** y`                                                 | Arithmetic base/exponent                        | Base type in current frontend; scalar integral exponents of floating bases are tested |
| `x == y`, `x != y`, `x < y`, `x <= y`, `x > y`, `x >= y` | Matching arithmetic dtypes                      | Bool over merged dimensions                                                           |
| `x & y`, `x \| y`, `x ^ y`                               | Bool operands                                   | Conjunction, disjunction, exclusive-or over merged dimensions                         |
| `~x`                                                     | Actual bool field for consistent field behavior | Logical field negation                                                                |
| `not x`                                                  | Scalar bool for consistent scalar behavior      | Logical scalar negation                                                               |

**Current limitations.** Integer `/` retains integer static type, while embedded
array true division may produce a floating value. Floor division and negative
modulo have backend exclusions. `**` does not consistently account for extra
exponent dimensions. Field `not` is accidentally accepted on some paths;
Python scalar `~True` differs from logical negation. Bool equality is outside
the ordinary arithmetic comparison rule. In-place augmented assignments such
as `+=` are not captured DSL statements.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c5--versus-ordinary-promotion-and-power|C5]]).** `**` keeps the current base-type result and the tested floating-base/integral-scalar-exponent case. No general rule for broadcasting extra field-exponent dimensions is promised.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c6-boolean-negation-across-python-and-lowered-execution|C6]]).** Use `not` for scalar bool and `~` for actual bool fields. Accidental acceptance of field `not` or scalar inversion is not a cross-mode logical-negation contract.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c14-numerical-semantics|C14]]).** The table records static result types. Integer division and exceptional arithmetic retain their execution-mode caveats; Python, NumPy, or Array API numerical behavior is not assumed uniformly.

### Indexing and writes

| Context                   | Supported form                              | Meaning                                                               |
| ------------------------- | ------------------------------------------- | --------------------------------------------------------------------- |
| Tuple expression          | `values[q]` with literal integer `q`        | Product projection                                                    |
| Field-operator expression | `field[LocalDim(q)]`                        | Select a local component and remove that dimension                    |
| Program output            | `out[slice0, ...]`, one slice per dimension | Relative restriction of the write domain; slice steps must be omitted |
| Host array-backed field   | Relative and named absolute indexing        | Field restriction/view according to the concrete implementation       |
| Host mutable field        | `out[index] = value`                        | Storage mutation outside the captured language                        |

Captured field-operator subscripts are not general NumPy indexing. Captured
program output slices do not accept integer indices, ellipsis, or explicit steps
(including `::1`). Tuple dynamic indexing/slicing is not established. The host
`restrict`/`premap` methods implement field restriction/remapping; captured
remapping uses [field-call syntax](#field-remapping).

Captured output slice bounds must be integer literals (including supported
negative literals) or omitted. Runtime names and computed expressions such as
`1+1` are unsupported; use an explicit absolute domain for runtime bounds.

Slices may alias storage. Some in-place calls are tested, but arbitrary shifted
input/output overlap and overlapping tuple outputs have no uniform snapshot
contract. Use independent outputs or explicit temporaries in such cases.

Source: [Field protocol](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/common.py),
[embedded fields](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/embedded/nd_array_field.py), and
[[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#4-pointwise-operations-and-builtins|field operations]].

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c18-aliases-and-dimension-changing-remaps|C18]]).** Selected tested in-place uses remain supported, but arbitrary shifted aliases or overlapping outputs have no promised snapshot semantics. Independent outputs or explicit temporaries avoid relying on the proposed future rule.

## Broadcasting

### Dimension and domain algorithm

1. Validate each operand's ordered dimensions.
2. Form the union of dimension labels in canonical order.
3. On a dimension present in multiple fields, intersect their ranges.
4. On a dimension absent from an operand, that operand is constant along the
   dimension. A scalar is constant along all field dimensions.
5. Apply the scalar operation at each coordinate, projecting the coordinate
   onto each operand's dimensions.

For example, a field on `I:[0,4)` plus a field on `K:[0,3)` has dimensions
`(I,K)` and domain `I:[0,4) × K:[0,3)`. Two fields on `I:[0,4)` and `I:[1,5)`
combine on `I:[1,4)`. A length-one range does not invoke NumPy positional
broadcasting on a shared dimension.

Broadcasting preserves dtype; the operation's dtype constraints still apply.
Explicit [broadcast](#broadcast) adds requested dimensions. Tuple structures
are processed only by operations that explicitly support them.

### Evaluation domains

Ordinary pointwise expressions require the common domain of their operands.
`where` does not promise short-circuit branch evaluation and cannot generally
hide an invalid shifted access. `concat_where` instead chooses contributed
coordinate regions, intersecting orthogonal branch domains first.

The result's expression domain and the output's requested write domain are
related but distinct. A program must have valid input reads for its requested
writes; output-domain selection does not fill, clip, or exchange halo values.

## Creation functions

All creation functions are **host APIs**. `dtype` accepts a supported dtype-like
object. `allocator` accepts an array namespace or GT4Py allocator/factory/backend;
`device` selects the allocation device. With neither supplied, ordinary
allocation uses a CPU default. Input conversion can infer the device from data.

`aligned_index` is a sequence of named absolute indices used as an alignment
reference by custom allocators. Array-namespace allocation ignores it with a
warning. Allocator/device incompatibilities are errors. A backing array's
physical layout does not alter the logical domain.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c13-exported-dtypes-versus-conversionbackend-coverage|C13]]).** A dtype being exported does not guarantee that every constructor, conversion helper, or allocator accepts it. Smaller and unsigned widths need validation on the chosen host path and backend.

### empty

```text
empty(domain: DomainLike, dtype=float64, *, aligned_index=None,
      allocator=None, device=None) -> Field
```

Creates an uninitialized field.

**Parameters.** `domain` gives finite dimensions and bounds; `dtype` gives the
value type. Allocation keywords follow the common creation rules above.

**Returns.** A newly allocated field of the specified domain and dtype.
Its initial element values are undefined.

### zeros

```text
zeros(domain: DomainLike, dtype=float64, *, aligned_index=None,
      allocator=None, device=None) -> Field
```

**Parameters.** `domain`, `dtype`, and allocation keywords as for `empty`.

**Returns.** A newly allocated field filled with zero of the requested dtype.

### ones

```text
ones(domain: DomainLike, dtype=float64, *, aligned_index=None,
     allocator=None, device=None) -> Field
```

**Parameters.** `domain`, `dtype`, and allocation keywords as for `empty`.

**Returns.** A newly allocated field filled with one of the requested dtype.

### full

```text
full(domain: DomainLike, fill_value: Scalar, dtype=None, *, aligned_index=None,
     allocator=None, device=None) -> Field
```

**Parameters.** `domain` gives the finite domain; `fill_value` is the value
assigned to every element. Omitted `dtype` is inferred from the fill value's
type. Allocation keywords follow `empty`.

**Returns.** A newly allocated field of the selected dtype, filled with the
converted fill value. Host dtype inference here is not the frontend's
small-integer-literal inference rule.

### as_field

```text
as_field(domain: DomainLike | sequence[Dimension], data, dtype=None, *,
         origin=None, aligned_index=None, allocator=None, device=None) -> Field
```

Copies supported array data into field storage.

**Parameters.** `data` supplies shape, dtype, and element values. `domain` may
be a concrete domain or just dimension labels. With labels, each data extent
`n` supplies `[0,n)`; `origin[d]=o` changes it to `[-o,n-o)`. With no `dtype`,
use the input data dtype. Allocation keywords follow the common rules.

**Returns.** A field with copied storage and the requested logical domain.
There is no `copy=False` argument.

**Errors.** Shape and domain extents must agree. The number of dimension labels
must equal data rank. Origin keys must be valid dimensions; a nonempty `origin`
cannot accompany a concrete domain.

**Example.**

```python
import numpy as np
import gt4py.next as gtx

I = gtx.Dimension("I")
f = gtx.as_field([I], np.arange(5, dtype=np.float64), origin={I: 1})
assert f.domain == gtx.domain({I: (-1, 4)})
```

### as_connectivity

```text
as_connectivity(domain: DomainLike | sequence[Dimension], codomain: Dimension,
                data, dtype=None, *, origin=None, aligned_index=None,
                allocator=None, device=None, skip_value=<auto>) -> Connectivity
```

Copies an integer index array and attaches connectivity metadata.

**Parameters.** `domain` and conversion/allocation options follow `as_field`.
`codomain` is the dimension indexed by stored values. `skip_value` identifies
missing neighbors. Omission detects `-1` when present in data, otherwise uses
`None`; explicit `None` disables this detection.

**Returns.** An integer-valued connectivity. A two-dimensional origin/local
neighbor table can be supplied through `offset_provider`.

**Current limitations.** Execution paths require `None` or `-1` skip metadata;
constructing a table with another sentinel can succeed but remapping can fail.
Arbitrary connectivity shapes accepted by construction are broader than the
compiled neighbor-table interface.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c10-general-skip-sentinel-api-versus-executable-remapping|C10]]).** Use `-1` for missing neighbors and `None` when no missing-neighbor sentinel is used. Constructor acceptance of other integral sentinels is not executable-remapping support.

### FieldConstructor

```text
FieldConstructor(allocator=None, aligned_index=None, device=None) -> FieldConstructor
constructor.empty(domain, *, dtype=float64) -> Field
constructor.zeros(domain, *, dtype=float64) -> Field
constructor.ones(domain, *, dtype=float64) -> Field
constructor.full(domain, fill_value, *, dtype=None) -> Field
constructor.as_field(domain, data, *, dtype=None, origin=None) -> Field
```

**Parameters.** Constructor arguments bind allocation policy. Method arguments
have the corresponding creation-function meanings; method dtypes are keyword-only.

**Returns.** Reusable construction configuration, whose methods return fields.
This is a host utility, not an allocation operation within the DSL.

Source: [constructors.py](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/constructors.py).

## Elementwise functions

Each entry below is available from `gt4py.next` and callable by a directly
bound name inside a field operator. Parameters shown before `/` are positional.
Unary functions act on each scalar/field element. Binary functions combine
matching dtypes using dimension promotion and common domains. The mathematical
meaning applies on the function's valid numerical domain; exceptions, NaNs,
infinities, signed zero, overflow, and accuracy remain subject to the
[numerical support notes](#numerical-support).

Source for all entries in this category:
[builtin definitions](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/fbuiltins.py) and
[frontend typing](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/foast_passes/type_deduction.py).

| Group                   | Functions                                                                                                  |
| ----------------------- | ---------------------------------------------------------------------------------------------------------- |
| Arithmetic              | [abs](#abs), [neg](#neg), [minimum](#minimum), [maximum](#maximum), [fmod](#fmod), [power](#power)         |
| Trigonometric           | [sin](#sin), [cos](#cos), [tan](#tan), [arcsin](#arcsin), [arccos](#arccos), [arctan](#arctan)             |
| Hyperbolic              | [sinh](#sinh), [cosh](#cosh), [tanh](#tanh), [arcsinh](#arcsinh), [arccosh](#arccosh), [arctanh](#arctanh) |
| Exponential and special | [sqrt](#sqrt), [exp](#exp), [log](#log), [gamma](#gamma), [cbrt](#cbrt)                                    |
| Rounding                | [floor](#floor), [ceil](#ceil), [trunc](#trunc)                                                            |
| Predicates              | [isfinite](#isfinite), [isinf](#isinf), [isnan](#isnan)                                                    |

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c14-numerical-semantics|C14]]).** For every entry in this category, the mathematical operation and static dtype are specified separately from numerical edge behavior. Backend-dependent exceptions, exceptional values, and accuracy are not replaced by an assumed Array API contract.

### abs

```text
abs(value: Number, /) -> Number
```

**Parameters.** `value` is a nonboolean arithmetic scalar or field.

**Returns.** The absolute value of each element. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### arccos

```text
arccos(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The inverse cosine in radians. The real-valued mathematical domain is `[-1,1]`. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### arccosh

```text
arccosh(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The inverse hyperbolic cosine. The real-valued mathematical domain is `[1,+∞)`. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### arcsin

```text
arcsin(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The inverse sine in radians. The real-valued mathematical domain is `[-1,1]`. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### arcsinh

```text
arcsinh(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The inverse hyperbolic sine of each element. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### arctan

```text
arctan(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The inverse tangent in radians. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### arctanh

```text
arctanh(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The inverse hyperbolic tangent. Its finite real-valued mathematical domain is `(-1,1)`. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### cbrt

```text
cbrt(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The real cube root of each element, including negative elements. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### ceil

```text
ceil(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The least integer-valued floating value no less than the input. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

**Notes.** The result keeps its floating dtype. Use `astype` for an integer dtype.

### cos

```text
cos(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The cosine of each element; angles are in radians. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### cosh

```text
cosh(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The hyperbolic cosine of each element. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### exp

```text
exp(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The base-e exponential of each element. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### floor

```text
floor(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The greatest integer-valued floating value no greater than the input. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

**Notes.** The result keeps its floating dtype. Use `astype` for an integer dtype.

### fmod

```text
fmod(lhs: Number, rhs: Number, /) -> Number
```

**Parameters.** `lhs` and `rhs` are arithmetic scalars/fields with equal dtypes.

**Returns.** The remainder with a quotient truncated toward zero: mathematically `lhs - trunc(lhs/rhs) * rhs`, for nonzero `rhs`. The result has the common dtype
and merged field dimensions on the common domain; two scalar inputs give a scalar.

**Notes.** Floating operands are accepted here; the captured `%` operator has
an integral-operand restriction. Negative operands also distinguish truncation
remainder from floor-based modulo.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c14-numerical-semantics|C14]]).** The mathematical remainder above uses a truncated quotient. It does not establish uniform exceptional arithmetic or remove the separate backend restrictions on integral `%` and floor division.

### gamma

```text
gamma(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The gamma function of each element, away from its poles at nonpositive integers. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### isfinite

```text
isfinite(value: Floating, /) -> Boolean
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** True where an element is finite and false where it is infinite or NaN. The result has bool dtype and the same dimensions/domain as a field input,
or is a scalar bool for scalar input.

### isinf

```text
isinf(value: Floating, /) -> Boolean
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** True where an element is positive or negative infinity. The result has bool dtype and the same dimensions/domain as a field input,
or is a scalar bool for scalar input.

### isnan

```text
isnan(value: Floating, /) -> Boolean
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** True where an element is NaN. The result has bool dtype and the same dimensions/domain as a field input,
or is a scalar bool for scalar input.

### log

```text
log(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The natural logarithm of each positive element. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### maximum

```text
maximum(lhs: Number, rhs: Number, /) -> Number
```

**Parameters.** `lhs` and `rhs` are arithmetic scalars/fields with equal dtypes.

**Returns.** The larger of the two corresponding values. The result has the common dtype
and merged field dimensions on the common domain; two scalar inputs give a scalar.

**Current limitations.** NaN and signed-zero selection are not specified
uniformly across execution modes.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c14-numerical-semantics|C14]]).** Ordinary ordered values use the larger-value rule. NaN and signed-zero selection remain unresolved across modes rather than inheriting Array API special cases.

### minimum

```text
minimum(lhs: Number, rhs: Number, /) -> Number
```

**Parameters.** `lhs` and `rhs` are arithmetic scalars/fields with equal dtypes.

**Returns.** The smaller of the two corresponding values. The result has the common dtype
and merged field dimensions on the common domain; two scalar inputs give a scalar.

**Current limitations.** NaN and signed-zero selection are not specified
uniformly across execution modes.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c14-numerical-semantics|C14]]).** Ordinary ordered values use the smaller-value rule. NaN and signed-zero selection remain unresolved across modes rather than inheriting Array API special cases.

### neg

```text
neg(value: Number, /) -> Number
```

**Parameters.** `value` is a nonboolean arithmetic scalar or field.

**Returns.** The additive inverse of each element, `-value`. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### power

```text
power(lhs: Number, rhs: Number, /) -> Number
```

**Parameters.** `lhs` and `rhs` are arithmetic scalars/fields with equal dtypes.

**Returns.** The first operand raised to the second operand. The result has the common dtype
and merged field dimensions on the common domain; two scalar inputs give a scalar.

**Current limitations.** Integral operands emit a frontend warning about result-type
consistency. This function uses ordinary matching-dtype promotion, unlike the
`**` operator's special base-type result rule.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c5--versus-ordinary-promotion-and-power|C5]]).** `power` requires matching dtypes and promotes dimensions normally. The base-type rule and mixed scalar exponent case of `**` do not extend to this function.

### sin

```text
sin(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The sine of each element; angles are in radians. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### sinh

```text
sinh(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The hyperbolic sine of each element. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### sqrt

```text
sqrt(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The nonnegative square root of each nonnegative element. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### tan

```text
tan(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The tangent of each element; angles are in radians. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### tanh

```text
tanh(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The hyperbolic tangent of each element. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

### trunc

```text
trunc(value: Floating, /) -> Floating
```

**Parameters.** `value` is a floating scalar or field.

**Returns.** The integer-valued floating value obtained by truncation toward zero. The static result preserves input dtype, dimensions, and domain; a scalar
input gives a scalar result.

**Notes.** The result keeps its floating dtype. Use `astype` for an integer dtype.

## Conversion and selection

### astype

```text
astype(value: Scalar | Field | TupleValue, type_: type, /) -> Scalar | Field | TupleValue
```

Converts values to an explicit dtype.

**Parameters.** `value` is a scalar, field, or recursively nested tuple.
`type_` is the destination scalar type.

**Returns.** A value with every primitive leaf converted to `type_`. Field
dimensions/domains and tuple structure are preserved. Scalar leaves remain
scalar; field leaves remain field-valued.

**Constraints.** Named collections lack the corresponding generic frontend
acceptance. Conversion does not add dimensions, extend a domain, or repair an
invalid input read. Allocation/copy identity is not part of this expression's
contract. Exceptional/narrowing casts have backend-dependent edge behavior.

**See also.** [Scalar construction](#scalar-construction-and-literals),
[type promotion](#type-promotion-rules).

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c13-exported-dtypes-versus-conversionbackend-coverage|C13]], [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c19-named-collection-declaration-and-conversion-limits|C19]]).** Conversion is specified for scalars, fields, and recursive tuples, subject to dtype/backend coverage. Named-product support in selection functions does not extend to `astype`.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c14-numerical-semantics|C14]]).** Exceptional and narrowing conversions remain backend-dependent; no uniform overflow or cast-edge rule is added here.

### broadcast

```text
broadcast(field: Scalar | Field, dims: tuple[Dimension, ...], /) -> Field
```

Adds coordinate-independent dimensions to a value.

**Parameters.** `field` is a scalar or field. `dims` is the exact ordered target
dimension tuple and contains every input dimension.

**Returns.** A field with the input dtype and target dimensions. At coordinate
`p`, the value is the input at the projection of `p` onto its input dimensions.
Added axes introduce no coordinate dependence. `broadcast(value, ())` has a
zero-dimensional field type.

**Errors.** Dropping an input dimension or supplying invalid dimensions is not
an admissible broadcast.

**Current limitations.** The embedded scalar implementation returns the original
scalar as a representation shortcut. Subsequent output assignment may produce
the expected values, but field-only behavior of that intermediate result is not
uniform. This does not relax the static field result or target-dimension rule.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c7-broadcast-fields-and-scalar-to-zero-dimensional-compatibility|C7]]).** The abstract result is a constant field along added dimensions. The embedded scalar shortcut is recorded as a representation limitation, not a guarantee that subsequent field-only operations work uniformly.

### where

```text
where(mask: Field[..., bool], true_field: Scalar | Field | Product,
      false_field: Scalar | Field | Product, /) -> Field | Product
```

Selects values elementwise using a boolean field.

**Parameters.** `mask` is a boolean field. The two branches have matching
product structure and equal dtypes at corresponding leaves. Named products
also match named type. Scalar branches are permitted; a scalar mask is not the
ordinary frontend mask contract.

**Returns.** A field at each primitive result leaf, with dimensions promoted
from that branch pair and the mask. Select `true_field` wherever the mask is
true and `false_field` elsewhere. Scalar branch leaves become field-valued.
Corresponding branches need equal dtypes, but different leaves of one tuple may
have different dtypes.

**Notes.** The branch expressions must be valid over their required common
domain. There is no guarantee of short-circuit evaluation or protection from
invalid shifted reads in an unselected branch. Use
[concat_where](#concat_where) when selecting coordinate regions with differing
branch coverage.

Source: [builtins](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/fbuiltins.py),
[typing](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/foast_passes/type_deduction.py), and
[[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#44-elementwise-selection|selection semantics]].

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c3-where-dtype-promotion|C3]]).** Only dimensions promote. Each true/false leaf pair must have the same dtype; distinct leaf pairs may have different dtypes. Convert explicitly with `astype` when needed.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c19-named-collection-declaration-and-conversion-limits|C19]]).** Matching supported named products are accepted here, without implying that other builtins accept the same structures.

## Connectivity and remapping

### Connectivity

```text
Connectivity[Dims[...], Dimension]
```

An integer-valued coordinate map with a domain and a codomain dimension.
`Connectivity` is a protocol; use `as_connectivity` for table data.
Applying a connectivity to a field substitutes its integer values for the
field's codomain coordinate. Unaffected field dimensions survive.

Neighbor-table providers use an origin/location dimension and a local slot
dimension as their domain, and the neighbor-location dimension as codomain.
`skip_value` identifies an absent entry; ordinary dereferences require valid
entries. Duplicated neighbor entries retain their multiplicity.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c4-offset-provider-values-and-optionality|C4]], [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c10-general-skip-sentinel-api-versus-executable-remapping|C10]]).** Runtime providers map tags to neighbor tables, and executable missing-neighbor metadata uses `-1` or `None`. Historical dimension-valued providers and arbitrary accepted sentinel metadata are not treated as the current execution contract.

### CartesianConnectivity

```text
CartesianConnectivity(domain_dim: Dimension, offset: int = 0, *,
                      codomain: Dimension | None = None) -> Connectivity
```

**Parameters.** `domain_dim` is the result-coordinate dimension. `offset` is
the integral displacement applied to its index. `codomain` is the input-field
dimension, defaulting to `domain_dim`.

**Returns.** The map `i -> i + offset`, optionally relabeling the dimension.
It represents a translation without a material neighbor table. Array-conversion
methods are not supported by this symbolic connectivity. In DSL expressions,
prefer dimension shift notation described below.

### FieldOffset

```text
FieldOffset(value: str, source: Dimension,
            target: tuple[Dimension] | tuple[Dimension, Dimension]) -> FieldOffset
```

Declares a named remapping.

**Parameters.** `value` is the offset-provider tag. `source` is the dimension
read in the input field. `target` is the replacement dimension tuple in the
result. For a neighbor offset `(A,L)`, `L` must have kind `LOCAL`.

**Returns.** A declaration resolved against the runtime offset provider when
used in a computation.

**Notes.** This `source` is the table's **codomain**, not its origin dimension.
For example, `FieldOffset("E2C", source=Cell, target=(Edge,E2CDim))` reads cell
values and produces edge/local-slot values. The provider table maps
`(Edge,E2CDim)` to `Cell`. Current Cartesian dimension shifts are self-contained
and do not require dimension-valued provider entries.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c4-offset-provider-values-and-optionality|C4]]).** Named neighbor offsets resolve through table-valued provider entries. Cartesian dimension shifts carry their own meaning and need no historical dimension-valued provider entry.

Indexed single-target `FieldOffset` retains legacy tagged-provider behavior;
frontend acceptance does not make it equivalent to a self-contained Cartesian
shift. Use `field(I+n)` or `field(I-n)` for fixed Cartesian shifts. A
single-target declaration used by `as_offset` is a separate supported form.

### Field remapping

```text
field(connectivity, *additional_connectivities) -> Field
field(offset) -> Field
field(offset[q]) -> Field
```

**Parameters.** The connectivity's codomain must be a dimension of `field`.
For neighbor offset `O` with provider table `C[A,L]`, bare `O` selects every
neighbor slot; `O[q]` selects fixed slot `q`.

**Returns.** For bare neighbor access, `field(O)(a,l,...) = field(C[a,l],...)`:
replace the input location dimension by `(A,L)`. For a fixed slot, remove `L`
and return `field(C[a,q],...)`. `field[LocalDim(q)]` similarly selects a slot
from an already local/sparse field.

**Constraints.** A dereferenced neighbor must be valid and within input bounds;
there is no value assigned to an ordinary skipped-neighbor read. A remap
creating a local dimension allows only one remapping argument. Fixed neighbor
remaps may compose; general dynamic unstructured slot selection is not an
established operation. Single-target Cartesian offset declarations cannot be
used as bare all-neighbor offsets. Their indexed form is accepted by parts of
the frontend but retains legacy tagged-provider execution limitations.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c4-offset-provider-values-and-optionality|C4]]).** Use dimension arithmetic for fixed Cartesian shifts; indexed single-target `FieldOffset` is not an equivalent portable, self-contained spelling. This does not restrict single-target declarations used by `as_offset`.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c10-general-skip-sentinel-api-versus-executable-remapping|C10]], [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c18-aliases-and-dimension-changing-remaps|C18]]).** Missing-neighbor metadata is limited to `-1` or `None`, and ordinary reads still require a valid neighbor. Tested fixed remaps do not establish support for arbitrary dimension-changing remap combinations.

### Cartesian shift expressions

```text
field(I + n) -> Field
field(I - n) -> Field
```

**Parameters.** `I` is the result dimension and `n` is a fixed literal
(or a closure-folded literal) displacement. Integral displacement reads the
same dimension; half-integral displacement reads its staggered counterpart.
A runtime displacement field uses [as_offset](#as_offset).

**Returns.** For an integral shift, result coordinate `i` reads input `i+n`,
and the corresponding input interval is translated by `-n`. For a half-integral
shift `h`, the input integer index is `i + floor(h) + 1` when the result
dimension is nonstaggered, and `i + floor(h)` when it is staggered. Other
fractional parts are rejected.

**Current limitations.** Multiple fixed Cartesian shifts preserve dimensions
and can appear together. Dimension-changing combinations, including multiple
staggered remaps, need additional validation because result typing does not
uniformly incorporate every remapping.

**Example.** A neighboring sum with explicit input coverage:

```python
import numpy as np
import gt4py.next as gtx

I = gtx.Dimension("I")


@gtx.field_operator
def adjacent_sum(a: gtx.Field[gtx.Dims[I], gtx.float64]):
    return a + a(I + 1)


a = gtx.as_field([I], np.arange(4, dtype=np.float64))
out = gtx.zeros({I: 3})
adjacent_sum(a, out=out)
np.testing.assert_array_equal(out.asnumpy(), [1.0, 3.0, 5.0])
```

Source: [connectivity model](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/common.py),
[FieldOffset](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/fbuiltins.py), and
[[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#51-general-remapping|remapping specification]].

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c18-aliases-and-dimension-changing-remaps|C18]]).** Multiple dimension-preserving fixed shifts retain their established behavior. Multi-axis staggering and other dimension-changing combinations are not generalized from those cases because result typing can omit later remaps.

## Neighbor reductions

These operations reduce one local neighbor axis with a known connectivity
association. Their results remove that axis and retain other dimensions.
Use `axis=` explicitly in captured code: the Python signature permits a
positional axis, but frontend reduction handling requires the keyword.

Missing neighbors are excluded. Each valid slot contributes once, even if
multiple slots reference the same neighbor. These functions do not define
arbitrary global reductions over horizontal/vertical axes. The frontend's
axis-presence check is broader than the local-axis execution contract.
Standalone external sparse/local fields cannot generally be reduced without
shifted-neighborhood evidence for compiled unrolling.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c8-reduction-axis-acceptance-and-sparse-only-reductions|C8]]).** The following three functions reduce a `LOCAL` neighborhood axis with known connectivity association, using `axis=`. Weaker frontend acceptance does not establish global-axis or standalone sparse-only reductions.

### neighbor_sum

```text
neighbor_sum(field: Field, /, axis: Dimension) -> Field
```

**Parameters.** `field` contains the local axis `axis` and the neighborhood
association needed by the backend.

**Returns.** The sum of valid neighbor-slot values at every remaining coordinate.
The static dtype is preserved; the selected dimension is removed. Lowering uses
a typed zero initial value.

**Current limitations.** Array-library accumulation may widen integer dtype.
Do not infer a general cross-mode accumulator dtype from frontend result typing.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c9-reduction-identity-skipped-slots-and-dtype|C9]]).** Sum retains its input static dtype and typed-zero lowering seed. Actual integer accumulation can widen in embedded execution; an explicit uniform accumulator-dtype policy remains a proposal.

### max_over

```text
max_over(field: Field, /, axis: Dimension) -> Field
```

**Parameters.** `field` and `axis` satisfy the common reduction requirements.

**Returns.** The maximum valid neighbor value at every remaining coordinate,
with the input static dtype and the local axis removed.

**Current limitations.** Lowering seeds with the dtype's finite minimum,
whereas embedded skip handling uses a global input extremum. All-skipped
neighborhoods and all-negative-infinity neighborhoods need not agree. NaN
handling has no uniform contract.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c9-reduction-identity-skipped-slots-and-dtype|C9]]).** The maximum rule describes ordinary nonempty finite neighborhoods. All-skipped, all-negative-infinity, and NaN cases retain the observed mode differences; the proposed uniform negative-infinity seed is not current behavior.

### min_over

```text
min_over(field: Field, /, axis: Dimension) -> Field
```

**Parameters.** `field` and `axis` satisfy the common reduction requirements.

**Returns.** The minimum valid neighbor value at every remaining coordinate,
with the input static dtype and the local axis removed.

**Current limitations.** Lowering seeds with the dtype's finite maximum,
whereas embedded skip handling uses a global input extremum. All-skipped
neighborhoods and all-positive-infinity neighborhoods need not agree. NaN
handling has no uniform contract.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c9-reduction-identity-skipped-slots-and-dtype|C9]]).** The minimum rule describes ordinary nonempty finite neighborhoods. All-skipped, all-positive-infinity, and NaN cases retain the observed mode differences; the proposed uniform positive-infinity seed is not current behavior.

### Neighbor reduction example

```python
import numpy as np
import gt4py.next as gtx
from gt4py.next import neighbor_sum

Cell = gtx.Dimension("Cell")
Edge = gtx.Dimension("Edge")
E2CDim = gtx.Dimension("E2C", kind=gtx.DimensionKind.LOCAL)
E2C = gtx.FieldOffset("E2C", source=Cell, target=(Edge, E2CDim))


@gtx.field_operator
def sum_adjacent_cells(a: gtx.Field[gtx.Dims[Cell], gtx.float64]):
    return neighbor_sum(a(E2C), axis=E2CDim)


table = gtx.as_connectivity([Edge, E2CDim], Cell, np.array([[0, 1], [1, 2]], dtype=np.int32))
a = gtx.as_field([Cell], np.array([2.0, 3.0, 5.0]))
out = gtx.zeros({Edge: 2})
sum_adjacent_cells(a, out=out, offset_provider={"E2C": table})
np.testing.assert_array_equal(out.asnumpy(), [5.0, 8.0])
```

**Support notes.** Some paths resolve reduction connectivity by local dimension
name, while others use the offset tag. Equal tag/local names avoid that specific
difference. Sparse output writes and sparse-only reductions have additional
backend exclusions. See [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#53-neighbor-reductions|reduction semantics]]
and the [backend matrix](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/definitions.py).

## Computation definitions

All definition functions consume Python functions whose source is retrievable.
Annotations, source names, and captured constants are resolved at frontend
construction, including when selecting embedded execution. The supported
parameter declaration subset is annotated ordinary positional-or-keyword
parameters with all argument values supplied. Full Python support for defaults,
positional-only, keyword-only, and variadic declarations is not established.

Use immutable captured configuration and explicit parameters for changing data.
Recursive closure collection also rejects unequal same-named captures in a
parent and referenced child, even if both are immutable. Distinct capture names
or explicit parameters avoid that restriction. Some source capture is cached;
changing a Python global does not have a uniform cross-mode invalidation rule.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c1-historical-frontenddomain-api-versus-the-accepted-api|C1]], [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c12-python-declaration-signatures|C12]]).** The definitions below decorate Python functions, using the supported ordinary annotated parameter subset. Historical AST-root entry points and full Python declaration flexibility are not the current contract.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c11-closure-lifetime|C11]]).** Captured values are immutable configuration and changing data are explicit arguments. Distinct names or parameters avoid unequal same-named captures across composed definitions. No common snapshot or mutation-invalidation policy is assumed.

### field_operator

```text
field_operator(definition=None, *, backend=<default-backend>, grid_type=None,
               **compilation_options) -> FieldOperator | decorator
```

Defines a pure computation on scalar/field values.

**Parameters.** `definition` is an annotated Python function, supplied directly
or by decorator syntax. `backend` is a backend object or explicit `None` for
embedded execution; omission selects the configured default. `grid_type`
selects or requests inference of grid constraints. Supported compilation options
are described under [with_compilation_options](#with_compilation_options).

**Returns.** A field-operator object, or a decorator awaiting the definition.
Its definition returns a scalar, field, or supported product internally.
An optional return annotation must match the inferred type exactly.

**Body requirements.** Assignments bind local values and may rebind names;
they do not mutate field storage. Every reachable path returns a value, with
matching return types. Ordinary function/operator definition names cannot
shadow reserved DSL builtin names. Source-captured callable references must be
lowerable DSL callables; arbitrary Python callbacks are not supported.

**Invocation.** Inside another operator, an expression call returns its value.
At a host entry point or in a program, provide `out=` matching field result
leaves. A scalar result is not a general substitute for a Field output; use
`broadcast` to make a statically field-valued result.

### scan_operator

```text
scan_operator(definition=None, *, axis: Dimension, forward: bool = True,
              init=0.0, backend=<default-backend>, grid_type=None)
    -> FieldOperator | decorator
```

Lifts a scalar recurrence over columns of a vertical dimension.

**Parameters.** `axis` has kind `VERTICAL`. `forward=True` traverses increasing
indices; `False` traverses decreasing indices. `init` is the initial carry.
The definition's first parameter is the carry; its type, the return type, and
`init` agree exactly. Carry values may be scalar or supported scalar products.
The remaining parameters receive the per-point input values.

**Returns.** A scan operator using the field-operator invocation protocol.
At invocation, omit the carry argument; scalar recurrence parameters accept
matching scalar/field leaves. Result scalar leaves are lifted to fields over
the merged invocation dimensions plus `axis`. The axis is included even when
absent from input fields.

**Semantics.** For requested interval `[l,u)` and each orthogonal coordinate:

```text
forward:  carry = init
          for k in l, ..., u-1:
              carry = definition(carry, inputs_at_k...)
              result_at_k = carry

backward: carry = init
          for k in u-1, ..., l:
              carry = definition(carry, inputs_at_k...)
              result_at_k = carry
```

This pseudocode specifies evaluation order, not permitted Python loop syntax
inside the definition. The scan is inclusive and resets `init` for every
column and invocation, including a subdomain. The output/domain context
supplies the scan interval.

**Current limitations.** All scans in one lowered program share an axis.
General horizontal/local scans, per-point initial state, arbitrary nested scan
bodies, and multi-axis scans are not portable features. Some scalar-only scans
lack embedded support. The decorator does not accept the compilation-option
keywords of `field_operator`; configure the resulting object's methods instead.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c2-default-domains-of-multiple-outputs|C2]]).** Embedded scans require one shared domain/column interval, including when called from a program. Explicit per-leaf domain tuples can fail; use a shared domain or separate invocations.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c20-compilation-annotations-versus-executable-precompilation|C20]]).** Direct scans support invocation/JIT but cannot be explicitly precompiled through `.compile()`. Wrap a scan in a program with concrete field annotations and compile that program for ahead-of-time use.

**Example.**

```python
import numpy as np
import gt4py.next as gtx

K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)


@gtx.scan_operator(axis=K, forward=True, init=0.0)
def prefix_sum(carry: float, x: float) -> float:
    return carry + x


x = gtx.as_field([K], np.array([1.0, 2.0, 3.0]))
out = gtx.zeros({K: 3})
prefix_sum(x, out=out)
np.testing.assert_array_equal(out.asnumpy(), [1.0, 3.0, 6.0])
```

### program

```text
program(definition=None, *, backend=<default-backend>, grid_type=None,
        **compilation_options) -> Program | decorator
```

Defines an ordered sequence of output-writing operator calls.

**Parameters.** `definition`, `backend`, `grid_type`, and compilation options
have the corresponding `field_operator` meanings.

**Returns.** A program object, or a decorator awaiting the definition.
Calling a program writes its output arguments and returns `None`.

**Body requirements.** A program contains direct field/scan-operator calls with
`out=`, optionally `domain=` or restricted output slices. Statements execute in
source order, so later calls observe earlier writes. Programs do not introduce
local assignment, branches, loops, value returns, or calls to other programs.
Full field expressions are placed inside field operators. Restricted scalar
arithmetic and `minimum`/`maximum` may occur in program call/domain arguments.

**Example.**

```python
import numpy as np
import gt4py.next as gtx

I = gtx.Dimension("I")


@gtx.field_operator
def twice(a: gtx.Field[gtx.Dims[I], gtx.float64]):
    return a + a


@gtx.program
def apply_twice(
    a: gtx.Field[gtx.Dims[I], gtx.float64],
    out: gtx.Field[gtx.Dims[I], gtx.float64],
):
    twice(a, out=out, domain={I: (1, 3)})


a = gtx.as_field([I], np.array([1.0, 2.0, 3.0, 4.0]))
out = gtx.full({I: 4}, -1.0)
apply_twice(a, out)
np.testing.assert_array_equal(out.asnumpy(), [-1.0, 4.0, 6.0, -1.0])
```

### Statements and control flow

| Form                                                                                                             | Field/scan operator body                                                  | Program body                        |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ----------------------------------- |
| Local assignment, annotation, tuple unpacking                                                                    | Supported; value bindings, no field stores                                | Not supported                       |
| `return expression`                                                                                              | Required on all reachable paths; common return type                       | No value return                     |
| `if`/`elif`/`else`, conditional expression                                                                       | Scalar bool condition; branches statically checked; matching result types | Not supported                       |
| Elementwise conditional                                                                                          | `where` with bool Field mask                                              | Put in a field operator             |
| `for`, `while`, comprehensions, generators                                                                       | Not supported                                                             | Not supported                       |
| `and`, `or`, chained comparisons                                                                                 | Not supported; use parenthesized comparisons with boolean `&`/`\|`        | Not general program expressions     |
| Augmented assignment, attribute/subscript stores                                                                 | Not supported                                                             | Writes through operator `out=` only |
| Imports, nested ordinary functions, lambdas, exceptions, context managers, assertions, classes, pattern matching | Not supported in captured bodies                                          | Not supported in captured bodies    |

`if` selects a scalar branch; both branches must still be statically valid.
Zero-dimensional bool fields are fields and cannot control `if`. Locals after
branches must be valid on the relevant paths according to binding/type analysis.
Docstrings are removed during parsing; `pass` is a no-op in field-operator bodies.
The [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#21-abstract-grammar|formal grammar]] describes the
supported subset in more detail.

Source: [decorators](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/decorator.py),
[field parser](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/func_to_foast.py), and
[program parser](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/func_to_past.py).

## Invocation and execution

### Operator calls

```text
operator(*args, out: Output, domain=<omitted>, offset_provider=<omitted>,
         enable_jit=None, **kwargs) -> None
operator(*args, **kwargs) -> DSL value             # inside an operator body
```

These are supported call forms; the wrapper's Python implementation uses
`**kwargs` to receive `out`, `domain`, and `offset_provider`.

**Parameters.** `args` and ordinary `kwargs` bind the definition's parameters.
`out` receives field-valued result leaves and matches dtype, dimensions, and
collection structure. `domain` restricts writes. An omitted `offset_provider`
becomes an empty mapping; an explicitly supplied operator provider must be a
mapping, so `None` is not equivalent. `enable_jit` overrides whether a missing
compiled variant may be created, otherwise following configured options.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c21-captured-type-rules-versus-embedded-host-call-validation|C21]]).** Matching argument and output types are caller obligations. Direct field-level embedded operator calls bypass some runtime type checks and can execute invalid calls or cast on assignment; this is not a supported coercion API.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c22-omitted-versus-explicit-none-operator-domains|C22]]).** Omit `domain` to infer writes from outputs. Explicit `domain=None` acts like omission only in field-level embedded execution; compiled operator host calls reject it during domain conversion.

**Returns.** `None` for output-writing calls. Expression calls inside another
operator return the DSL value and inherit the enclosing execution context.
Host operator calls require `out` even in embedded mode.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c1-historical-frontenddomain-api-versus-the-accepted-api|C1]], [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c4-offset-provider-values-and-optionality|C4]]).** Host calls require `out` and permit explicit `domain=`. An omitted provider means `{}`; an explicitly supplied provider must be a mapping. The historical mandatory-provider and domain-prohibition descriptions are not used.

### Program calls

```text
program_object(*args, offset_provider=None, enable_jit=None, **kwargs) -> None
```

**Parameters.** `args`/`kwargs` bind the program's declared arguments.
Output fields are program parameters used by its body; there is no universal
extra `out` parameter independent of the definition. Omitted or explicit
`None` provider becomes `{}`. A neighbor computation needs all referenced
offset tables. `enable_jit` has the same override meaning as for operator calls.

**Returns.** `None`, after performing the program's writes.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c4-offset-provider-values-and-optionality|C4]]).** Omitted and explicit `None` providers both mean `{}` for programs. Referenced neighbor offsets still require tables; the operator-call restriction on explicitly supplied `None` does not apply here.

### Output-domain rules

| Selection                                        | Meaning                                                                                                  |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| Explicit program `domain={d: (start,stop), ...}` | Absolute half-open bounds; nonempty dictionary with integral bounds, matching output dimensions in order |
| Output slice                                     | Relative literal bounds on the current output field; steps omitted, one slice per dimension              |
| Domain tuple                                     | Recursively corresponding per-leaf output domains; embedded scans require a shared domain instead        |
| Omitted domain in lowered execution              | Each output leaf supplies its own domain                                                                 |
| Omitted domain in field-level embedded execution | Intersection of all output-leaf domains                                                                  |

A program cannot combine sliced outputs with explicit `domain=`. For output
bounds `[a,b)`, nonnegative slice bound `n` means `a+n`; negative `n` means
`b+n`. An omitted bound selects the corresponding endpoint. A dictionary in a
nested domain tuple is not a general replacement for a nested tuple of domains.
Host domain conversion forms are broader than captured program dictionary syntax.
An outer single domain dictionary may apply to every matching output leaf.
Slice bounds must be integer literals (including supported negative literals)
or omitted, not runtime names or expressions such as `1+1`. For runtime bounds,
use an explicit absolute domain such as `domain={I: (0,n)}`; account for nonzero
logical origins when replacing a relative slice.

**Current limitations.** Unequal tuple-output extents with an omitted domain
can produce different writes between modes. Use equal domains or, for ordinary
field operators, explicit per-leaf domains to avoid that difference. Embedded
scans require a single shared domain/column interval and can fail on domain
tuples, including when called from a program. Local/sparse output dimensions have
lowering/backend restrictions. No domain rule allocates missing halos or
specifies arbitrary overlapping output writes.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c1-historical-frontenddomain-api-versus-the-accepted-api|C1]], [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c15-dimension-identity-and-domain-map-ordering|C15]]).** Explicit absolute domains and relative output slicing are current alternatives, subject to the combination restriction above. Captured domain dictionaries retain the output-dimension key-order constraint.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c2-default-domains-of-multiple-outputs|C2]]).** Per-leaf domains are the model for explicit multiple-output selection. The distinct omitted-domain rules of lowered and embedded execution remain documented behavior. Equal domains or explicit per-leaf tuples avoid the default-mode disagreement for ordinary field operators; there is no blanket same-shape requirement. Embedded scans instead need a shared domain or separate invocations.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c18-aliases-and-dimension-changing-remaps|C18]]).** Domain selection does not establish snapshot semantics for overlapping inputs or outputs. The aliasing limits under indexing and writes still apply.

### Backends

| Selection              | Execution                                                                                               |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| `None`                 | Field-level embedded execution of the original Python definitions over registered field implementations |
| `gtx.itir_python`      | Lowering followed by roundtrip execution of iterator IR in Python                                       |
| `gtx.gtfn_cpu`         | GridTools native CPU backend                                                                            |
| `gtx.gtfn_gpu`         | GridTools GPU backend                                                                                   |
| Other `Backend` object | The selected backend's transformation/execution pipeline                                                |

Pass backend objects rather than name strings. A nested operator uses its
caller's execution context; its configured backend applies at its own host
entry point. These backends differ in supported features and numerical cases;
Python roundtrip is distinct from field-level embedded execution.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c17-backend-selector-examples-and-internal-representation-terminology|C17]]).** Backend selectors are objects (or `None`), not historical string names. Internal iterator execution and representation choices do not add Python DSL primitives or tuple-valued field elements.

### with_backend

```text
entry.with_backend(backend: Backend | None) -> configured copy
```

**Parameters.** `backend` is the desired backend object or `None`.

**Returns.** A configured copy of the program/operator. It does not submit a
compilation by itself.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c17-backend-selector-examples-and-internal-representation-terminology|C17]]).** Pass a backend object or `None`; docstring examples using backend-name strings are treated as historical examples rather than an accepted selector interface.

### with_grid_type

```text
entry.with_grid_type(grid_type: GridType) -> configured copy
```

**Parameters.** `grid_type` selects Cartesian or unstructured constraints.

**Returns.** A program/operator copy configured with that grid type.
An incompatible explicit grid selection is an error.

### with_compilation_options

```text
entry.with_compilation_options(**compilation_options) -> configured copy
```

**Parameters.** Supported options are:

| Option                           | Meaning                                                                                      |
| -------------------------------- | -------------------------------------------------------------------------------------------- |
| `enable_jit: bool`               | Allow compilation of missing variants during a call; initial policy comes from configuration |
| `static_params: sequence[str]`   | Names of scalar arguments to specialize                                                      |
| `static_domains: bool`           | Specialize field bounds; default false                                                       |
| `connectivities: OffsetProvider` | Connectivity configuration available to compilation workflows                                |

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c20-compilation-annotations-versus-executable-precompilation|C20]]).** For explicit compilation, configured connectivities must supply runtime neighbor tables (or `{}` when none are needed). Nonempty provider-type-only mappings are rejected despite broader annotations.

**Returns.** A configured copy. These options affect staging/execution; they
do not add DSL syntax or change an expression's defined mathematical value.

### with_static_params

```text
program_object.with_static_params(*static_params: str | None) -> Program
```

**Parameters.** Names of scalar parameters to specialize. No arguments, or a
single `None`, clears the set; otherwise use parameter names.

**Returns.** A configured program copy.

### with_bound_args

```text
program_object.with_bound_args(**kwargs) -> ProgramWithBoundArgs
```

**Parameters.** Scalar program arguments to bind by name.

**Returns.** A program with a reduced call interface. Unknown parameter names
and supplying an already bound argument again are errors. Explicit `.compile()`
on this bound-program wrapper is currently unimplemented.

### compile

```text
entry.compile(offset_provider=None, **static_args) -> same entry
```

Submits variants to the entry point's compiled-program pool.

**Parameters.** `offset_provider` is a runtime neighbor-table mapping, or a list
of such mappings. Supply a mapping (possibly `{}` when no tables are needed)
unless connectivity configuration supplies it. Each `static_args` value is a
list of candidate scalar/product values; combinations form specialized variants.
The entry must have a non-None backend.

**Returns.** The same object, with variants added to its pool; this differs
from the configured-copy behavior of `with_*` methods.

**Errors.** Compiling without a backend raises `RuntimeError`. Missing
connectivity/provider information raises `ValueError`; non-list static argument
values raise `TypeError`. If JIT is disabled, subsequent invocation needs a
matching compiled variant.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c20-compilation-annotations-versus-executable-precompilation|C20]]).** Nonempty provider type descriptors are currently rejected with `ValueError`, despite their accepted annotation. Direct scan operators also cannot be precompiled through this interface, which does not supply their required argument types. For ahead-of-time scan compilation, compile a wrapping program with concrete field annotations; direct scan invocation/JIT remains available.

### wait_for_compilation

```text
wait_for_compilation() -> None
```

**Returns.** `None` after all tracked submitted compilations finish.
This is the namespace-level `gtx.wait_for_compilation()` function.

**Errors.** Failed compilation exceptions are surfaced; multiple failures can
be grouped. Waiting does not itself execute the resulting programs.

Source: [entry points](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/decorator.py),
[compilation options](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/otf/options.py), and
[compiled variants](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/otf/compiled_program.py).

## Experimental functions

Import these names from `gt4py.next.ffront.experimental` outside the captured
function, then use directly bound names inside it. Their accepted type forms
are broader than uniformly executable cases.

### as_offset

```text
as_offset(offset: FieldOffset, field: Field, /) -> Connectivity
```

Constructs a coordinate-dependent Cartesian displacement for field remapping.

**Parameters.** `offset` is a single-target Cartesian offset with equal source
and target dimension `d`. `field` has integral dtype and contains `d`; its values
are the displacements at each result coordinate.

**Returns.** A connectivity for which `a(as_offset(offset,q))(p)` reads
`a` at the coordinate obtained by replacing `p[d]` with `p[d] + q(p)`.
All selected input coordinates must exist.

**Current limitations.** General dynamic unstructured neighbor selection is not
established. Extra dimensions on the displacement field are not consistently
reflected in static result typing. Check the selected backend's dynamic-offset
support and transitive halo requirements.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c18-aliases-and-dimension-changing-remaps|C18]]).** The coordinate-dependent displacement meaning is retained, but extra displacement dimensions are not promised to propagate correctly through remap typing. Complete dimension deduction is a proposed future rule, not an assumption of this entry.

### concat_where

```text
concat_where(domain: Domain, true_field: Scalar | Field | Product,
             false_field: Scalar | Field | Product, /) -> Field | Product
```

Assembles values selected by coordinate region.

**Parameters.** `domain` is a one-dimensional region for the common embedded
contract. Branches have matching structure and corresponding dtypes, with
dimension broadcasting. They need only cover their selected contributions.

**Returns.** At coordinates in `domain`, select the true branch; elsewhere
select the false branch. Let `d` be the region dimension, `T` and `F` the branch
domains, and `Q` their projected intersection on all dimensions except `d`:

```text
Q = projection_except_d(T) ∩ projection_except_d(F)
result_domain = Q ∩ ((T ∩ domain) ∪ (F ∩ complement(domain))).
```

Extend `Q` without restriction along `d` for the final intersection. Missing
branch dimensions broadcast; products are processed leafwise. Orthogonal
intersection occurs before concatenation, so differently bounded branches do
not create an arbitrary nonrectangular union.

**Region expressions.** With `n` of `IndexType` (int32), `d < n`, `d <= n`,
`d > n`, `d >= n`, and `d == n` select their corresponding integer-coordinate
intervals. `d != n` selects a complement with a gap and has embedded limitations.
Chained comparisons are rejected; parenthesized region combinations have
frontend typing but not uniform execution support.

**Errors and limitations.** Embedded execution requires a one-dimensional
condition and a contiguous resulting domain. Interior gaps raise
`NonContiguousDomain`. Some disjoint/multidimensional conditions pass frontend
typing but cannot execute embedded. Empty branches, scalar/unbounded broadcasts,
and more elaborate region combinations have feature-specific exclusions.

**Example.** The result intersects the orthogonal `J` ranges:

```python
import numpy as np
import gt4py.next as gtx
from gt4py.next.ffront.experimental import concat_where

I = gtx.Dimension("I")
J = gtx.Dimension("J")
t = gtx.as_field({I: (0, 1), J: (0, 3)}, np.ones((1, 3), dtype=np.float64))
f = gtx.as_field({I: (1, 2), J: (1, 4)}, np.zeros((1, 3), dtype=np.float64))
result = concat_where(I < 1, t, f)
assert result.domain == gtx.domain({I: (0, 2), J: (1, 3)})
np.testing.assert_array_equal(result.asnumpy(), [[1.0, 1.0], [0.0, 0.0]])
```

Source: [experimental definitions](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/ffront/experimental.py),
[embedded implementation](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/src/gt4py/next/embedded/nd_array_field.py),
and [concat limitations](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/development/ADRs/next/0022-Limitations-of-embedded-concat_where.md).

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c16-region-algebra-versus-embedded-interval-representation|C16]]).** The shared execution contract uses a one-dimensional condition and contiguous result. Disjoint or multidimensional expressions accepted by typing remain backend-specific rather than part of that common subset.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c19-named-collection-declaration-and-conversion-limits|C19]]).** Matching supported named products are selected leafwise here. This structure support does not imply named-product conversion support in `astype`.

## Errors and support

### Validation

Invalid syntax, unresolved names, annotation mismatches, incompatible dtypes,
invalid dimensions, and invalid calls are rejected by the DSL frontend with
`DSLError` or a subclass. The diagnostic includes source information when
available. Host constructors and execution controls may raise ordinary Python
exceptions. Some accepted-but-unimplemented paths fail later with assertions
or `NotImplementedError`; frontend acceptance alone is not a support test.

See the [diagnostic guide](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/docs/development/next/error-messages.md) and
[[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#23-rejected-python-constructs|unsupported syntax]].

### Numerical support

The static dtype rules are specified above. Current implementations do not
establish a complete uniform contract for integer division, overflow/narrowing,
negative modulo, exceptional scalar arithmetic, NaN/signed-zero selection,
extreme/empty reductions, or floating-point reassociation. Some division-by-zero
tests permit either IEEE values or exceptions. Do not assume the special cases
or accuracy requirements of the Array API standard merely from a shared
function name.

**Interpretation used ([[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c9-reduction-identity-skipped-slots-and-dtype|C9]], [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c14-numerical-semantics|C14]]).** Static types and ordinary mathematical results are retained while unsettled runtime edge cases remain explicit. Uniform reduction identities, accumulator dtypes, and numerical special-case rules are recommendations in the companion, not guarantees added by this API page.

### Backend support and related specification

Use [feature exclusions](https://github.com/GridTools/gt4py/blob/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/definitions.py) together with
[frontend feature tests](https://github.com/GridTools/gt4py/tree/f3b3ab28ecfc4de6ef914d6e7bed17a5447fd43f/tests/next_tests/integration_tests/feature_tests/ffront_tests)
for backend support. Iterator-only facilities are outside this API. Allocator,
cache, runner, and backend workspace management are host infrastructure.

The companion specification records the evidence and interpretation of current
disagreements:

- [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#8-contradictions-interpretation-and-unresolved-contracts|Dtype, power, boolean, and broadcast rules]].
- [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#6-programs-domains-and-writes|Output domains and aliasing]].
- [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#c11-closure-lifetime|Closure capture and composition]].
- [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#53-neighbor-reductions|Reduction semantics]].
- [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#54-region-selection-experimental-concat_where|Experimental region semantics]].
- [[personal/egparedes/next-dsl-documentation/next-dsl-documentation_specification#9-evidence-coverage-and-focused-observations|Repository evidence and focused observations]].

This page is a reference for the reviewed checkout. It is not a new public API,
a backend conformance certification, or an adoption of the companion's proposed
future resolutions.
