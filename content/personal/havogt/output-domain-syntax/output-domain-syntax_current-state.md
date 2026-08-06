---
title: Output domains — verified current state
author: havogt
tags: [frontend, past, domain, multiple-output-domains, embedded, icon4py, verification, diagnostics]
created: 2026-08-06
status: draft
---

> Appendix to [[personal/havogt/output-domain-syntax|Surface syntax for per-output compute domains]].
> Everything here was **executed**, not read off the source. Base:
> `GridTools/gt4py@43746f13a` (2026-08), CPython 3.13, backends `itir_python` (roundtrip),
> `gtfn_cpu`, and embedded.

## A. Capability matrix

Three execution paths accept `out` / `domain`, and they do **not** agree:

| | compiled program (`gtfn`, `roundtrip`) | embedded program (`backend=None`) | direct field-operator call |
| --- | --- | --- | --- |
| `out=f` | ✓ | ✓ | ✓ |
| `out=f[1:-1]` (literal slices) | ✓ | ✓ | ✓ (xfail on JAX embedded) |
| `out=f[lo:-1]` (runtime bound) | ✗ `TypeError: 'Slice.lower' must be Constant` | ✗ (same, at decoration) | ✗ (same) |
| `out=f[1:-1]` with fewer slices than dims | ✗ `Too many indices` | ✗ | ✗ |
| `out=f[1:-1][1:-1]` (chained) | ✗ `'Subscript.value' must be Name` | ✗ | ✗ |
| `out=f[{I: (1, 5)}]` (mapping subscript) | ✗ `AssertionError` in `_compute_field_slice` | ✗ `IndexError: Unsupported index type` | ✗ (same) |
| `out=f[gtx.domain({I: (1, 5)})]` | ✗ (not callable in PAST) | **✓** (plain Python) | **✓** |
| `out=state.u` (named-collection field) | ✓ | ✓ | ✓ |
| `out=tup[0]` (tuple element) | ✗ type error | ✗ | ✗ |
| `domain={I: …}` all dims, in order | ✓ | ✓ | ✓ |
| `domain={J: …}` **partial** | **✗** `Dimensions in out field and field domain are not equivalent` | **✓** | **✓** |
| `domain={J: …, I: …}` **unordered** | **✗** (same error) | **✓** | **✓** |
| `domain=` + slicing on the same call | ✗ `Either only domain or slicing allowed` | ✗ (rejected at decoration) | n/a |
| `domain=` exceeding the out field | **silent OOB write (gtfn)** / `IndexOutOfBounds` (roundtrip) | `IndexOutOfBounds` | `IndexOutOfBounds` |
| `domain=` as a closure variable | ✗ `Type <class 'dict'> not supported` | ✗ | ✓ (ordinary Python) |
| `domain=gtx.domain(...)` | ✗ `Functions must be referenced by their name` | ✓ | ✓ |
| single dict broadcast over a **nested** tuple out | ✓ (top level only) | ✓ | ✓ |
| dict covering a **subtree** of a tuple out | **✗** `Domain dict cannot map to tuple outputs` | ✓ | ✓ |

The compiled path — the one that runs in production — is the strictest in every row where they
differ.

## B. Reference-frame facts

Verified with an out field whose domain is `IDim ∈ [1, 7)` (non-zero origin):

- `domain={IDim: (2, 4)}` writes buffer rows 1–2, i.e. **absolute coordinates 2–3**. Absolute.
- `out=f[0:2]` writes buffer rows 0–1, i.e. **absolute coordinates 1–2**. Relative to the field's
  own `start`.

So `f[1:5]` is `[f.start + 1, f.start + 5)` and only *looks* absolute. A negative bound counts from
`stop` — decided from the literal's sign in `_visit_slice_bound`, which is why runtime bounds cannot
be admitted without a new convention.

## C. The out-of-bounds reproducer

`domain=` is a promise, not a derivation: on gtfn nothing relates it to the out field's actual
extent. The following writes two rows past the field, into another field's buffer, and reports
success.

```python
@gtx.field_operator
def fop(a: IJField) -> IJField:
    return a + a

@gtx.program(backend=gtx.gtfn_cpu)
def prog(a: IJField, out: IJField):
    fop(a, out=out, domain={IDim: (0, 6), JDim: (0, 6)})   # `out` will be a 4-row field

big = gtx.as_field(gtx.domain({IDim: (0, 6), JDim: (0, 6)}), np.full((6, 6), -1.0))
small = common._field(big.ndarray[0:4], domain=gtx.domain({IDim: (0, 4), JDim: (0, 6)}))
prog(gtx.as_field(gtx.domain({IDim: (0, 6), JDim: (0, 6)}), np.full((6, 6), 1.0)), small)
print(big.ndarray[:, 0])
```

```
small domain: Domain(IDim[horizontal]=(0:4), JDim[horizontal]=(0:6)) shares memory: True
rows of the underlying 6-row buffer after the call:
[2. 2. 2. 2. 2. 2.]        # rows 4 and 5 belong to `big`, not to `small`
```

The same program on `roundtrip`, and the same `domain=` in embedded or on a direct call, raises:

```
IndexOutOfBounds: Out of bounds: slicing Domain(IDim[horizontal]=(0:6), JDim[horizontal]=(0:6))
with index `Domain(IDim[horizontal]=(0:8), …)`, `IDim[horizontal]=(0:8)` is out of bounds …
```

because those paths go through `embedded/common.py::_absolute_sub_domain`, which requires
containment. The compiled path has no equivalent.

## D. Why field-keyed domains cannot work

```python
>>> f = gtx.as_field([I], np.zeros(4))
>>> hash(f)
TypeError: unhashable type: 'numpy.ndarray'
>>> type(f == f)
<class 'gt4py.next.embedded.nd_array_field.NumPyArrayField'>     # element-wise, not a bool
```

An embedded `@gtx.program` body executes as plain Python, so `domain={pgrad: {...}}` is a real dict
construction and raises. This is a hard constraint on the surface, not an implementation detail.

## E. icon4py survey

274 `@gtx.program`s, AST-based (script below):

```
programs: 274
call statements in program bodies: 289
  ... with an explicit `domain=`: 231
  ... whose `domain=` is a tuple (multiple output domains): 3
  ... with a sliced/subscripted `out` element: 2
redundant repetitions of an identical domain literal within one program: 1
(#domain-carrying statements, #distinct domains) -> #programs:
   (1, 1): 216      (2, 1): 1      (2, 2): 5      (3, 3): 1
```

The multi-domain call sites:

```
muphys.py:93   program=muphys_run
   out leaves=8  domain leaves=13 distinct=2   domain spans 64 lines, whole stmt 76 lines
graupel.py:737 program=graupel_run
   out leaves=8  domain leaves=13 distinct=2   domain spans 64 lines, whole stmt 77 lines
compute_cell_diagnostics_for_dycore.py:263 program=compute_perturbed_quantities_and_interpolation
   out leaves=9  domain leaves=9  distinct=6   domain spans 50 lines, whole stmt 84 lines
```

plus `compute_rho_theta_pgrad_and_update_vn` from
[icon4py#1405](https://github.com/C2SM/icon4py/pull/1405): 4 out leaves, 4 domain leaves, 3 distinct,
18 lines.

Survey script:

```python
import ast, pathlib
ROOT = pathlib.Path("<icon4py>/model")

def is_program(node):
    return any("program" in ast.unparse(d) and "field_operator" not in ast.unparse(d)
               for d in node.decorator_list)

def leaves(n):
    if isinstance(n, ast.Tuple):
        for e in n.elts: yield from leaves(e)
    else: yield n

for path in ROOT.rglob("*.py"):
    try: tree = ast.parse(path.read_text())
    except SyntaxError: continue
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or not is_program(fn): continue
        for s in fn.body:
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)): continue
            kw = {k.arg: k.value for k in s.value.keywords}
            d = kw.get("domain")
            if not isinstance(d, ast.Tuple): continue
            ls = [ast.unparse(x) for x in leaves(d)]
            print(f"{path.name}:{s.lineno} program={fn.name}  out leaves="
                  f"{len(list(leaves(kw['out'])))} domain leaves={len(ls)} "
                  f"distinct={len(set(ls))} lines={d.end_lineno - d.lineno + 1}")
```

## F. Implementation map

Where the behaviour lives, for anyone implementing the staged plan:

| concern | location |
| --- | --- |
| `out` / `domain` validation, the "no slicing with domain" rule, tuple-structure matching | `ffront/past_passes/type_deduction.py` (`_validate_domain_out`, `_ensure_no_sliced_field`) |
| positional named-collection matching | same file, `TODO(havogt)` above `out.types` |
| slice → domain lowering, dimension-order check, `get_domain_range` emission | `ffront/past_to_itir.py` (`_visit_stencil_call_out_arg`, `_construct_itir_domain_arg`, `_visit_slice_bound`, `_compute_field_slice`) |
| `Subscript.value: Name`, `Slice.lower/upper: Optional[Constant]` | `ffront/program_ast.py` |
| PAST parsing of subscripts, dicts, slices | `ffront/func_to_past.py` |
| direct field-operator call: `domain=` → `out[domain]` | `ffront/decorator.py` (~line 714) |
| embedded call: `target[domain] = source[domain]` | `embedded/operators.py` (`field_operator_call`, `_tuple_assign_field`) |
| absolute vs relative restriction, containment check | `embedded/common.py` (`sub_domain`, `_absolute_sub_domain`, `_relative_sub_domain`) |
| `DomainLike`, `Domain.slice_at`, `Dimension.__lt__` → `Domain` | `common.py` |
| GTIR domain access | `get_domain_range` builtin, ADR 20 |

Existing tests: `feature_tests/ffront_tests/test_domain.py`,
`multi_feature_tests/ffront_tests/test_multiple_output_domains.py` (the #2225 suite: nested tuples,
two vertical dims, unstructured, temporaries, direct calls),
`feature_tests/ffront_tests/test_named_collections.py::test_named_collection_with_multiple_output_domains`,
and six `test_domain_exception_*` cases in `unit_tests/ffront_tests/test_func_to_past.py`. Feature
markers: `uses_program_with_sliced_out_arguments` (xfailed on JAX embedded),
`uses_scalar_in_domain_and_fo`. There is **no** marker for multiple output domains and no test for
the partial/unordered-domain rejection, the OOB case, or the compiled-vs-embedded divergence.

## G. Relevant history

- ADR 10 (2022) *Domain in Field View* — introduced multiple `out` fields "without runtime checks,
  requiring that all fields are sliced in the same way … This restriction should be lifted, but as
  the domain handling is not final, we decided to implement this pragmatic solution for now."
  D3 is that decision, still in place.
- ADR 20 (2025) *Runtime domains* — `get_domain_range(field, dim)`; notes that a builtin returning a
  field's *entire* domain was withdrawn for lack of a clean design.
- [#2209](https://github.com/GridTools/gt4py/pull/2209) prototype → [#2225](https://github.com/GridTools/gt4py/pull/2225)
  *Multiple output domains* (merged 2025-11-03; large dace-lowering component) →
  [#2428](https://github.com/GridTools/gt4py/pull/2428) named collections.
