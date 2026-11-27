---
title: "A Kokkos C++ backend for gt4py.next: reuse the gtfn pipeline, resolve iterators, own the accessor"
author: egparedes
tags: [kokkos, backend, gtfn, gtfn-ir, cpp, codegen, performance-portability, performance, gpu, cuda, hip, openmp, iterator-resolution, field-accessor, temporary-arrays, mempool, scan, vertical, unstructured, connectivities, bindings, nanobind, build-system, otf, toolchain, benchmarks, acceptance-criteria, effort-estimate, adr, prior-art]
created: 2026-11-27
status: draft
---

> **TL;DR** Add a Kokkos C++ backend to `gt4py.next` that keeps gt4py's
> Python-side gtfn pipeline (`apply_common_transforms` → `GTFN_lowering` →
> `gtfn_ir`, OTF workflow, cache, CMake, test matrix) and **contains no
> GridTools C++**. There is no dedicated Kokkos IR. The three new pieces are:
> - an **iterator-resolution pass** on `gtfn_ir`, which emits explicit index
>   expressions for every stencil and scan body;
> - a **gt4py-owned pointer + strides accessor** inside kernels, with Kokkos
>   Views used only at the binding boundary;
> - **pooled, aligned, non-zeroed temporaries**, with loop shapes kept in
>   per-execution-space header traits so the generated source is identical on
>   every platform.
>
> Proposed acceptance bar: per-cell parity with gtfn within 2 % on GPU and
> 3 % on CPU, plus a tighter per-family mean bound. Central estimate: about
> 30–37 person-weeks for CPU + CUDA, starting with a two-week spike on
> generated kernels.

> [!warning] What this document is, and is not
> - **A recommendation, not a decision to start.** It records the common
>   position that three independent, agent-based study groups signed after
>   two analysis rounds, one benchmark run and a reconciliation round (see the
>   [[personal/egparedes/kokkos-backend/kokkos-backend_research|research appendix]]).
>   It is AI-generated and not yet reviewed by a person. The decisions in
>   [Open decisions](#open-decisions) are still the author's to make.
> - **Parity is not demonstrated for a backend.** The measurements compare
>   **hand-written** Kokkos kernels, which imitate what a generator could
>   emit, against `gtfn_cpu`/`gtfn_gpu`. They were taken on **one node**
>   (NVIDIA GH200: Grace CPU + Hopper GPU) with **one toolchain** (Kokkos
>   5.2.2, GCC 14.2.0, CUDA 12.9.41). There is one known GPU miss: vertical
>   solves with temporaries, at 1.03 at the largest size. No code generator
>   exists, and generated code is unmeasured.
> - **Marks.** **[M]** marks a measured statement and **[I]** an inferred one
>   (from reading code or reasoning about measurements), as in the sources.
>   Each design item also carries its agreement status:
>   - **AGREED**: all three groups hold it.
>   - **PROPOSED**: the groups differed, and this is the merge they signed,
>     with dissents recorded.
>   - **OPEN**: the decision is not yet made.
> - **Pinned versions.** All gt4py file references are to commit
>   [`c56efffdd`](https://github.com/GridTools/gt4py/tree/c56efffdd02380d98439be3f7877d7d9a287c8d5),
>   with gridtools-cpp 2.3.9.

## Requirements

These are the requirements the study was asked to meet, in the order they were
set:

1. **R1. A new `gt4py.next` backend targeting Kokkos C++.** The hypothesis to
   test was that it looks much like gtfn and can reuse most of the gtfn
   infrastructure.
2. **R2. The generated code is self-contained: no GridTools C++ at all.** Only
   gt4py's Python-side gtfn and OTF infrastructure may be reused.
3. **R3. Performance-portable.** Kokkos primitives handle the
   performance-relevant parts, and the generated source is (mostly) identical
   on every platform, as gtfn's is today.
4. **R4. Performance at least on par with gtfn on CPU and GPU** (≤ 1.0x run
   time). Portability matters equally.

R2 and R4 together shape most of the design. R2 removes the cheapest route (a
Kokkos executor inside `gridtools::fn`). R4, once measured, removes the
assumption that a thin C++ iterator header is free.

## Problem / motivation

`gtfn` is gt4py's C++ backend. On the Python side it is mostly gt4py code
(lowering to `gtfn_ir`, OTF stages, bindings and CMake generation). On the C++
side everything sits on GridTools: `gridtools::fn` executors and `scan_pass`,
SIDs, `hymap`, `sid::composite`, the nanobind SID adapter, and the GridTools
CPU and GPU backends. One group counted that a generated gtfn program pulls in
about 160–184 GridTools headers, roughly 21–29k lines **[M, `g++ -H`]**.
Performance portability today means "whatever `gridtools::fn` provides for
CPU and CUDA/HIP".

A Kokkos backend would hand execution spaces, dispatch, policies and memory
spaces to a widely used performance-portability layer. In return gt4py must
own everything GridTools did for it on the C++ side.

**The hypothesis holds for the Python side and fails for the C++ side.** The
pipeline shape carries over unchanged. Everything that turns `gtfn_ir` into
C++ must be rewritten, along with the C++ runtime it relies on. The places
where GridTools strings leak into gt4py's Python code are listed in
[the research appendix](#appendices). The ones on the codegen path are:

| Where (at `c56efffdd`) | GridTools leak |
|---|---|
| `codegens/gtfn/codegen.py` (332 lines) | templates built on `gridtools::fn`: executors, `scan_pass`, `deref`/`shift`, `hymap`, `sid::composite`, `allocate_global_tmp`, `GT_FUNCTION` |
| `codegens/gtfn/gtfn_module.py:95` | `gridtools::sid::dimension_to_tuple_like` for sparse-field arguments |
| `codegens/gtfn/gtfn_module.py:125-132` | `gridtools::fn::sid_neighbor_table::as_neighbor_table`, `gridtools::hymap::keys` for connectivities |
| `codegens/gtfn/gtfn_module.py:213-214, 230-256` | backend header, backend type and library name (`gridtools/fn/backend/{gpu,naive}.hpp`, `gridtools_{cpu,gpu}`) |
| `codegens/gtfn/itir_to_gtfn_ir.py:723` | `::gridtools::tuple<…>` as the C++ type of tuple temporaries |
| `otf/binding/nanobind.py:100, 181-203, 267-274` | `gridtools::nanobind::as_sid`, `stride_spec`, `shift_sid_origin`, `rename_numbered_dimensions`, `fn::make_tuple`, `tuple_util::get`, GridTools includes |
| `otf/binding/nanobind.py:144` | `cudaDeviceSynchronize()` hard-coded for device programs |
| `otf/compilation/build_systems/cmake_lists.py:79-115` | a closed `match` over library names (`nanobind`, `gridtools_cpu`, `gridtools_gpu`), whose in-tree TODO says "do not add more libraries here … refer to ADR-0016" |

## Proposal

### Overview

```text
FOAST/PAST ──► GTIR ──► apply_common_transforms ──► GTFN_lowering ──► gtfn_ir          (all reused)
                                                                         │
                                                       iterator-resolution pass        (new, Python)
                                                                         │
                                                                "resolved gtfn_ir"
                                                                         │
                                                       Kokkos code generator            (new, next to codegens/gtfn/)
                                                                         │
             generated .cpp  +  gt4py-owned runtime header (package data)  +  GridTools-free nanobind bindings
                                                                         │
             CMake (find_package(Kokkos)) ──► compiledb / cache / OTF compile step      (reused, parameterised)
                                                                         │
                                                       runners/kokkos.py                (new)
```

### 1. Pipeline: reuse the gtfn Python side, no second IR

- **AGREED.** Reuse `apply_common_transforms`, `GTFN_lowering` and `gtfn_ir`
  unchanged. Also reuse the OTF workflow (translation → bindings →
  compilation), the persistent translation cache, compiledb, CMake generation,
  the toolchain builders, argument conversion and the test matrix with its
  exclusion lists.
- **New:**
  - a Kokkos translation step that takes the same constructor options as
    `GTFNTranslationStep`;
  - a Kokkos code generator in `codegens/kokkos/`;
  - a gt4py-owned runtime header;
  - GridTools-free nanobind bindings;
  - `runners/kokkos.py`.
- **AGREED.** There is no second lowering from GTIR and no dedicated Kokkos IR
  in the baseline. No measured miss needs information that `gtfn_ir` lacks.
  The extra information the backend does need can be re-attached alongside
  `gtfn_ir`: types from GTIR, connectivity properties and dimension kinds.
  A dedicated IR keeps value only as headroom, for example scan fusion with
  scratch memory or grouping launches. Its extra cost was estimated
  differently by each group: +8.5–15, +6–12 and about 0–3 person-weeks.
- **PROPOSED (name).** Call it the "iterator resolution pass", producing
  "resolved `gtfn_ir`". It is not presented as a new IR.

### 2. Iterator resolution pass

- **AGREED (moved into the baseline after the measurements).** A pass over
  `gtfn_ir` resolves `shift`, `deref` and `can_deref` into explicit index
  expressions and neighbor-table loads. The code generator emits these as
  straight-line C++.
- **Why.** No iterator form was fast on both targets for neighbor accesses
  **[M]**:
  - gtfn's literal form ("shift twice, then a ternary") is 1.6–2.0x slower
    than explicit indexing on GPU;
  - the "bind the shifted iterator once" form is 2.2–3.8x slower on CPU. In
    its assembly GCC 14.2 keeps the iterator object in stack memory inside
    the innermost loop **[M]**. That this is the cause is **[I]**: the
    magnitudes are not fully explained;
  - explicit indexing was best, or within 1 % of best, in every cell.

  For Cartesian stencils and column-per-iteration scans, iterator and explicit
  forms are within 1 % on GPU and 0.91–1.03 on CPU **[M]**.
- **AGREED (fallback).** Some bodies the pass may not resolve. These are
  emitted in gtfn's literal iterator form from a thin header, flagged at
  codegen time and reported. Such programs are outside the performance bar
  until measured.
- **PROPOSED (scope).** One resolver handles all stencil and scan bodies,
  including Cartesian shifts, which are its trivial case. Excluding them would
  save nothing and keep two emission paths. Scan bodies are included because
  of the CPU evidence:
  - CPU column chunks need explicit bodies; iterator chunks are 1.3–2.9x
    slower than the iterator column loop **[M]**;
  - two groups measured gtfn's per-column scan shape above 1.0 on x86
    (1.03–1.08 and 1.29–1.33 on workstations; unexplained), so chunks may be
    needed for parity there.

  *Dissents:* one group keeps scan bodies as headroom (+2–3 person-weeks).
  Two analysts of another group want the iterator form kept for Cartesian
  kernels.
- **Feasibility [M, first round].** One group's prototype closure-aware
  partial evaluator resolved 581 of 584 distinct programs of the gtfn CPU test
  suite after transforms. The 3 failures were `no_transforms` variants. A
  second group found 0 unresolvable programs in 102 dumps.

  ITIR CSE does leave iterator-typed `let`s of applied shifts in stencil
  bodies (`iterator/transforms/cse.py`), so the pass must evaluate
  iterator-valued bindings rather than just pattern-match `deref(shift(…))`.

What changes in the emitted body, for a V2E neighbor sum with skip values
(illustrative, not generator output):

```cpp
// gtfn today (literal iterator form, executed by gridtools::fn)
acc = can_deref(shift(e, V2E, 0_c)) ? acc + deref(shift(e, V2E, 0_c)) : acc;  // ×6

// resolved gtfn_ir, emitted as straight-line C++ over the gt4py accessor
const int n0 = v2e(v, 0);
acc = (n0 != skip) ? acc + e(n0, k) : acc;                                    // ×6
```

### 3. Field accessor inside kernels

- **AGREED (design).** Inside kernels, use a gt4py-owned accessor made of the
  data pointer and precomputed strides. The stride of the first horizontal
  dimension is statically 1, and indices are `int`. Kokkos unmanaged Views are
  used **only at the binding boundary**, for ownership, memory space and
  debugging.
- **Rejected in kernels: `LayoutStride` Views.** They cost up to +10 % on
  stencils and up to 2x on the neighbor chain **[M]**.
- **Inputs whose first horizontal stride is not 1** remain supported through a
  runtime-unit-stride instantiation of the pointer accessor. That
  instantiation is outside the performance bar.
- **Why not `LayoutLeft` Views.** A `LayoutLeft` View ties the pointer on
  stencils and unstructured kernels. On GPU scans it costs about 5 %, with
  explicit bodies as well (`tri` 1.07 against 1.02, `tritmp` 1.10 against
  1.05, ratios to gtfn) **[M]**, and the scans have no margin to pay that.
- **AGREED (offset width chosen at bind time).** Use 32-bit offsets when the
  allocation has fewer than 2^31 elements, otherwise a 64-bit-offset
  instantiation. The 64-bit-index, 64-bit-offset form ties the 32-bit pointer
  on GPU scans **[M]**. It costs 12–15 % on GPU `lap2` **[M]**. Both numbers
  were measured with the iterator body. The combination
  of `int` index and 64-bit offsets with explicit bodies is unmeasured (spike
  item 5).
- **OPEN (decision 1).** This uses fewer Kokkos primitives than "Views
  everywhere". Kokkos abstracts execution, dispatch, policies, memory spaces
  and temporaries, but not the address arithmetic in kernels. All three groups
  recommend accepting that.

```cpp
// sketch of the accessor (gt4py-owned runtime header)
template <class T, class Off /* int32_t or int64_t, chosen at bind time */, int Rank>
struct field {
  T* __restrict__ p;            // points at the first compute point
  Off s[Rank];                  // s[0] == 1 statically on the fast path; runtime-unit-stride variant otherwise
  KOKKOS_FORCEINLINE_FUNCTION T& operator()(int i, int j, int k) const { return p[i + j * s[1] + k * s[2]]; }
};
```

### 4. Temporaries

- **AGREED.** Temporaries come from a pool on both targets and are never
  zero-filled. Per-call allocation is 2.7–3.4x slower on GPU **[M]**. On CPU
  at 72 threads, a per-call allocation is only 0.93–1.02 against an
  allocator-fixed gtfn (`gtfn_fixed`, see [Evidence](#evidence)), while the
  pool is 0.34–0.62 against it **[M]**.
- **AGREED (layout).** Temporaries are compact. The first compute point sits
  on a 256-byte boundary and rows are padded to 128 bytes.
  - Alignment is the largest single factor measured, at 2–4 % **[M]**.
  - The evidence is one kernel (`tritmp`, GPU) at widths that are multiples
    of 32. Other widths, `hdiff`/`tmpst` and the CPU are unmeasured.
- Tuple-valued temporaries need a gt4py-owned tuple type in place of
  `::gridtools::tuple` (`itir_to_gtfn_ir.py:723`).
- See [[shared/external-memory-for-dace-arrays/external-memory-for-dace-arrays|External workspace memory for DaCe temporaries]]
  for the overlap with the DaCe backend's workspace design.

### 5. Loop shapes live in header traits; the generated text is identical

- **AGREED.** Loop shapes are defined per execution space in header traits,
  so the generated text is identical on every platform. The generated file
  names only `Kokkos::DefaultExecutionSpace` and its memory space. The
  platform comes from the Kokkos installation and the architecture flags.
  A unit test guards that the CPU and GPU configurations produce identical
  source, so build caches do not fork silently.

| | CPU (OpenMP) | GPU (CUDA) |
|---|---|---|
| stencils | `RangePolicy` over rows (k, j), plain inner unit-stride loop; unstructured: k × 1024-point blocks | `MDRangePolicy`, tile {32,8,1}, `Iterate::Left` (gtfn's block shape) |
| functor members | copied to locals before the inner loop (without this, unstructured kernels are 1.12–1.40x slower **[M]**) | — |
| not the default | `MDRangePolicy` (1.0–2.1x slower **[M]**; may change with Kokkos 5.3, see [[personal/egparedes/kokkos-backend/kokkos-backend_versions\|versions appendix]]) | flat `RangePolicy` for stencils (1.04–1.42x slower **[M]**) |
| no-ops, left out | `omp simd` on the inner loop | `IndexType<int>`, `LaunchBounds`, `Kokkos::fence` vs device sync |
| dispatch | named functors (`KOKKOS_LAMBDA` unmeasured) | same |

- **Scans (AGREED).** One GPU thread, or one CPU iteration, per column, with a
  sequential vertical loop. Never `Kokkos::parallel_scan`: gt4py scans are
  arbitrary, non-associative recurrences.
- **PROPOSED (GPU scan launch shape).** A header trait offers two modes:
  stages launched separately as gtfn does, or fused per column.
  - **Default: two kernels**, the measured shape (tiled `MDRangePolicy`, one
    thread per column), until a pre-registered rerun decides.
  - The fused flat-range variant reached 0.997 / 0.979 on `tritmp`, but it was
    picked after the run from 12 combinations, and fusion has the opposite
    sign on other cells.
  - Fusing is legal only if no stage reads an earlier stage's output at a
    horizontal shift. The codegen checks this.

  *Dissent:* one group would default to the fused flat range now. The
  two-kernel default ships solves with temporaries outside the bar (1.030 at
  the largest size) until the rerun.
- **PROPOSED (CPU scan shape).** A header trait offers both modes, and both
  are implemented in the baseline. The defaults are per architecture until the
  spike decides:
  - **x86:** explicit column chunks (B = 8);
  - **aarch64:** gtfn's two-pass column loop.

  Evidence:
  - The column loop is at parity on Grace (0.99–1.00 at 1–72 threads, one cell
    at 1.002) **[M]** but above 1.0 on x86 in two groups' workstation runs
    (unexplained).
  - Explicit chunks are below 1.0 in every measured cell, but the best chunk
    size varies with CPU and thread count **[M]**.
- **AGREED.** No mesh-ordering knob is needed for the bar. With gtfn's tile,
  randomly numbered tables are at parity (0.96–1.01) **[M]**. Tile and policy
  stay overridable per program. A numbering-aware heuristic is headroom.
  *Dissent:* two analysts want it now.

### 6. Runtime, bindings and build

These points come from the first analysis round. No measurement contradicted
them, but none were tested beyond the benchmark build.

- **One Kokkos runtime per process.** Kokkos must be a shared library
  (`libkokkoscore`), because every generated program is its own extension
  module.
  - Initialization is lazy, thread-safe and guarded by
    `Kokkos::is_initialized()`. A second `initialize` aborts. If someone else
    initialized Kokkos, the backend attaches to that runtime.
  - The backend never calls `Kokkos::finalize()`: Python-held Views and the
    temporary pool can outlive any exit hook.
  - Compile workers never initialize Kokkos.
  - A build-time guard rejects a user environment that links a second, static
    Kokkos.
- **Streams.** Kokkos' default CUDA instance uses its own stream. The baseline
  runs one device on the default instance and fences after each call
  (`Kokkos::fence()` replaces `cudaDeviceSynchronize()`). Passing CuPy's
  stream (`Kokkos::Cuda(stream)`) is later production work, and the ordering
  against CuPy streams must be documented.
- **Bindings.** Keep the nanobind generator. Replace the SID construction
  (`as_sid`, `shift_sid_origin`, `make_tuple`) with the accessor and unmanaged
  Views. DLPack strides are in elements. The device-consistency check stays in
  Python.
- **Build.**
  - Generated CMake uses `find_package(Kokkos)`. That needs the ADR 0016
    redesign of the closed library `match` in `cmake_lists.py`, not a fourth
    `case`.
  - Kokkos 5.x requires C++20, while the generated CMake pins `cxx_std_17`
    (`cmake_lists.py:66`). The Kokkos target propagates C++20, but the pin
    should become a parameter.
  - On CUDA, use Kokkos' default CXX mode (`nvcc_wrapper` /
    `kokkos_launch_compiler`) and replay it through compiledb.
  - The cache key includes a fingerprint of the Kokkos installation.
- **Device-code rules.**
  - The generated code must not need `--expt-relaxed-constexpr`. Kokkos'
    `Kokkos_ENABLE_CUDA_CONSTEXPR` defaults to off, while GridTools sets the
    flag.
  - gtfn's builtin mapping (`std::plus<>{}`, `std::numeric_limits`, …) is
    therefore replaced by `Kokkos::` math functions, `Kokkos::min`/`max`
    (same tie and NaN behaviour as `std::min`/`max`; never `fmin`/`fmax`),
    and inf/NaN helpers pinned in one place in the header.
- **R2 guard in CI.** No `gridtools` token may appear in generated sources or
  bindings.
- **Provisioning.** No PyPI wheel ships Kokkos with a CMake config, and the
  Debian/Ubuntu package (Kokkos 3.7) is too old. For the baseline, users provide
  `Kokkos_ROOT`. CI needs a cached source build on GitHub CPU runners and a
  CSCS image for CUDA. DeviceType has only CPU, CUDA and ROCM, so SYCL is out
  of scope without core changes.

## Evidence

All numbers below are from hand-written Kokkos kernels against gtfn, through
gt4py's real toolchain, on one GH200 node. Kokkos 5.2.2 shared, GCC 14.2.0,
CUDA 12.9.41, no `-mcpu`. Ratios are Kokkos time / gtfn time, so below 1
means Kokkos is faster. Every timed variant matched gtfn bit for bit **[M]**.

Full setup, method, all tables and limits are in the
[[personal/egparedes/kokkos-backend/kokkos-backend_benchmarks|benchmark appendix]].
The kernel and harness sources are in `kokkos-backend/bench/`.

**What was benchmarked.** All fields are `float64`.

- **Kernels:**
  - `lap` is a Cartesian Laplacian and `lap2` is `lap(lap)`.
  - `hdiff` is flux-limited horizontal diffusion (4 stencils, 3 temporaries)
    and `tmpst` is two statements with 1 temporary.
  - `tri` is a tridiagonal vertical solve (forward and backward sweep) writing
    to caller fields; `tritmp` keeps its intermediates as temporaries.
  - `v2e` sums over the 6 edges of each vertex, with 5 % skip values (-1).
    `chain` goes vertex → edges → vertices (E2V inside V2E).
- **Sizes:**
  - `large` is a 512×512×80 Cartesian domain, or a mesh of 160k vertices and
    480k edges with 80 levels.
  - `xlarge` (GPU only) is 1024×1024×80, or 1.05M vertices and 3.1M edges.
  - Every large working set exceeds the CPU's last-level cache.
- **Mesh numbering:**
  - The meshes are periodic triangulated grids. Each is used with two
    numberings of its vertices and edges, so the same topology gives two
    memory-access patterns.
  - **Local numbering** is grid order. Neighbors of a point get nearby
    indices, so neighbor-table lookups touch nearby memory, and a GPU block or
    CPU row reads few distinct cache lines **[I]**. This is the locality-ordered case
    a well-ordered real mesh aims for.
  - **Random numbering** applies a random permutation to the same mesh.
    Neighbor indices are scattered, so almost every neighbor access is a cache
    or memory-sector miss, and run time is dominated by memory traffic rather
    than by the kernel code **[I]**. It is a deliberate worst case.
  - **Why the distinction matters.** On random tables the GPU block shape,
    not the source form, decides the time **[M]**: a flat `RangePolicy` ran
    `v2e` in 0.35x gtfn's time at xlarge but 0.88x at large. gtfn's 32×8 tile
    is not optimal there **[I]**.
  - This is why §5 has no mesh-ordering knob: with gtfn's tile, Kokkos is at
    parity on random tables. It is also why the acceptance bar treats the two
    orderings as separate families.
  - A real ICON grid was not tested (spike item 4).

**GPU, large / xlarge, candidate shapes [M]**

| family | kernel | ratio vs stock gtfn | at or below 1.0? |
|---|---|---|---|
| Cartesian stencils | `lap`, `lap2` | 0.93–0.95, 0.76–0.81 | yes |
| stencils with temporaries | `hdiff`, `tmpst` | 0.98–0.99 | yes, **no margin** |
| vertical solve | `tri`, two kernels (gtfn shape) | 1.00 / 1.01 | parity; xlarge interval above 1.0 |
| vertical solve with temporaries | `tritmp`, two kernels, compact aligned temporaries | **1.004 / 1.030** | **no**; the 3 % is unexplained |
| neighbor reductions, local numbering | `v2e`, `chain`, explicit | 0.38–0.58 (vs `gtfn_us`: 0.49–0.64) | yes |
| neighbor reductions, random numbering | `v2e`, `chain`, explicit | 0.96–1.01 | parity |

**CPU, one socket (72 Neoverse-V2 cores), large, at 1 / 8 / 72 threads [M]**

| kernel | candidate | ratio vs stock gtfn |
|---|---|---|
| `lap` / `lap2` | rows, explicit | 0.33 / 0.39 / 0.93 and 0.51 / 0.51 / 0.84 |
| `tri` | two-pass column loop (gtfn shape) | 0.99 / 0.99 / 1.00 |
| `tri` | explicit column chunks (16) | 0.65 / 0.74 / 0.87 |
| `v2e` / `chain`, local | rows, explicit | 0.63 / 0.64 / 0.78 and 0.56 / 0.57 / 0.57 |
| `hdiff`, `tmpst`, `tritmp` | pooled temporaries | 0.10–0.52 against stock gtfn; **0.34–0.62 against `gtfn_fixed`** |

How to read these numbers:

- **At 36–72 threads, CPU margins on kernels with temporaries are mostly
  gtfn's own handicap.**
  - Reading the headers shows that GridTools allocates every CPU temporary
    twice and zero-fills it (`sid/allocator.hpp`, `fn/backend/naive.hpp` of
    gridtools-cpp 2.3.9) **[I]**.
  - `gtfn_fixed` emulates the upstream fix of about two lines: one
    uninitialized allocation, applied by `bench/patch_gt.py`. With it, a
    factor of 5.9 of `hdiff`'s 10x margin at 72 threads disappears **[M]**.
  - At 1 thread, temporaries explain only 1.14x of a 2.3x margin; the kernel
    code explains the rest **[M]**.
  - These margins must not be quoted as a Kokkos advantage. All groups
    recommend submitting that fix upstream regardless of this proposal
    (decision 3).
- **Most CPU cells at 1–36 threads come from one allocation**, so allocation
  spread is not covered there. The spread between allocations is up to
  2.5–2.7 % on GPU and 2–5 % on CPU (worst 14 % at 1–18 threads, 28 % on one
  `gtfn_fixed` cell) **[M]**.
- **1-thread CPU margins reflect gtfn's own loop nest** (`omp parallel for
  collapse(3)`, K outermost), which gtfn could fix **[I]**.
- **Source form moves times by 2–4x [M]**, which is why generated code must be
  measured before any claim of parity. The iterator results are GCC 14.2 and
  one header. "Explicit is safe" is equally single-compiler and
  single-architecture, and rests on four kernel families.
- **Version sensitivity** (desk research, not part of the signed position;
  [[personal/egparedes/kokkos-backend/kokkos-backend_versions|versions appendix]]):
  - Kokkos 5.3.0 may weaken "row loops beat `MDRangePolicy` on CPU";
  - older Kokkos (below 5.2) is slower on GPU `MDRangePolicy`;
  - Clang or `-mcpu=neoverse-v2` could move the unstructured ratios either
    way;
  - nothing is known for CUDA 13.x or AMD.

## Acceptance bar

- **AGREED.** A blanket "≤ 1.0x on every cell" is not testable. Cells sit at
  0.98–1.03, and the spread between allocations is of the same size. The bar
  must be stated per kernel family, with a tolerance.
- **PROPOSED (wording).** This merges one group's per-cell rule with the other
  two groups' family mean:

  > Families: Cartesian stencils; stencils with temporaries; vertical solves;
  > vertical solves with temporaries; neighbor reductions on locality-ordered
  > meshes; neighbor reductions on randomly numbered meshes. Targets: one GPU;
  > one CPU socket at 1 thread, 8 threads and all cores. Sizes whose working
  > set exceeds the last-level cache. For every cell (kernel, size, target,
  > gating baseline) the median over at least three allocations of the
  > Kokkos/gtfn ratio is ≤ 1 + ε, and for every family and target the
  > geometric mean of its cells is ≤ 1 + ε/2. Cells in (1.00, 1 + ε] are
  > listed with a cause. Sizes that fit in cache are reported as time per call
  > and not gated.

  The family mean is an additional condition. It cannot offset a per-cell
  miss.
- **PROPOSED (ε).** ε = 0.02 on GPU and 0.03 on CPU. *Dissents:* one group
  would take 0.05 on CPU and revisit ε once three allocations exist. Two
  analysts of another group would take 1 % on GPU and 5–6 % on CPU, applied to
  the upper 95 % bound.
- **AGREED (baselines).** Gate on stock `gtfn_cpu`/`gtfn_gpu` for all
  families. For unstructured families, also gate on gtfn with its existing
  unit-stride option (`gtfn_us`, `unstructured_horizontal_has_unit_stride`),
  since the Kokkos backend assumes a static unit stride. Use the same compiler
  and flags on both sides.
- **PROPOSED (`gtfn_fixed`).** Report `gtfn_fixed` for every family with
  temporaries and expect ≤ 1.0 against it. It becomes gating once the
  allocator fix is upstream. *Dissent:* one group would gate on it now.
- **Consequence on today's data [M].**
  - GPU `tritmp` in gtfn's two-kernel shape misses at xlarge (1.030).
  - The literal-iterator V2E sum misses against `gtfn_us` (1.03–1.05).
  - Explicit V2E passes (0.57–0.64).

## Effort and first step

**PROPOSED.** For CPU + CUDA at the bar above, the central estimate is about
30–37 person-weeks, with a range of 22–45. The per-group central estimates
were about 30, about 36 and about 36–37. The last group puts a floor of
about 31 under it and considers 22 unreachable at this bar.

| scope | person-weeks |
|---|---|
| CPU + CUDA at the acceptance bar (central) | 30–37 (range 22–45) |
| HIP on top | +2–3 |
| full production incl. HIP, CI and stream interop (one group's own milestone definition, not comparable) | ≈ 50 |
| a dedicated IR instead of resolved `gtfn_ir` | +0–15, depending on the group, for headroom only |

- **Why the estimate grew.** First-round estimates were about 22–27 central
  for comparable scopes. They assumed a thin iterator header and `LayoutStride`
  Views, and are superseded. Relative to the previous round, the signed figure
  is about 4–8 higher, from three changes:
  - the resolver moving from contingency into the baseline, including scan
    bodies;
  - the aligned temporary pool;
  - the scan traits.
- **AGREED: a two-week spike first.** One analyst asks for three weeks. The
  benchmark run already replaced the hand-written-kernel parts of earlier
  spike plans. Still to measure, in this order:
  1. **Generated kernels** from the resolver on the benchmark cells, CPU and
     GPU, since source form moved times by 2–4x.
  2. **A pre-registered rerun** of GPU scan shapes and temporary layouts. It
     includes widths that are not multiples of 32, `hdiff`/`tmpst` with
     aligned temporaries, and the cause of the 3 % `tritmp` gap.
  3. **x86 and Clang**, with `-mcpu`/`-march` on both sides. This covers the
     per-column scan discrepancy and the chunk size.
  4. **A real workload.** One real program with several executions and
     temporaries, one real ICON grid, and many small launches with host
     overhead separated out.
  5. **The `int`-index, 64-bit-offset accessor** on GPU scans.
  6. **A pooled gtfn** without zero-fill.
  7. **Multi-socket CPU.**
  8. **Compile time** per program.
  9. **AMD**, if HIP is in scope.

  Every gating cell needs at least three allocations. The versions appendix
  suggests three additions: Kokkos 5.3.0 on the same GCC/CUDA; Clang 23 with
  libomp and `-mcpu=neoverse-v2` on both sides, plus a GCC control with the
  same flags; and the newest CUDA 13.x.

## Open decisions

These are the author's to make. The text above is a recommendation for each.

1. **Pointer accessor or Views everywhere.** Accept the gt4py-owned pointer
   accessor in kernels, or use Kokkos Views everywhere. All groups recommend
   the accessor, because Views cost about 5 % on GPU scans, which have no
   margin (§3).
2. **Tolerance and wording of the bar.** Any tolerance relaxes "at least on
   par with gtfn (≤ 1.0x)". The questions are whether the per-cell plus
   family-mean rule, ε = 0.02 / 0.03, and the gating baselines are acceptable.
3. **The GridTools fix.** Whether to upstream the roughly two-line GridTools
   temporary-allocation fix now. All groups recommend it regardless of the
   Kokkos decision. It also decides when `gtfn_fixed` becomes gating.
4. **Minimum Kokkos version.**
   - Any 5.x pin forces C++20, while gt4py's generated CMake pins C++17.
   - The suggestion is 5.2.2, the measured version, raised to 5.3.0 after a
     rerun shows no regression.
   - Writing the header C++17-clean so Kokkos 4.7 also works was one
     analyst's untested option.
5. **How Kokkos is provided.** There is no PyPI wheel. The baseline assumes a
   user-provided install (`Kokkos_ROOT`) as a shared library, one runtime per
   process, never finalized. The CI provisioning route must be chosen.
6. **Whether HIP is in scope.** It adds 2–3 person-weeks plus AMD
   measurements, and no number from this study carries over to AMD.
7. **What "self-contained" allows.** All groups assumed that R2 permits a
   small gt4py-owned C++ header shipped as package data. The fallback is to
   inline the header into every generated translation unit, at a compile-time
   cost.
8. **How visible Kokkos is in the generated file.** This is partly settled.
   - **Excluded by the measurements:** spelling the policies out in generated
     text ("Y2"). A bare `MDRangePolicy` costs 1.0–2.1x on CPU, so the policy
     type and the CPU loop shape must come from per-space header traits (§5)
     to keep the text identical.
   - **Still open:** whether the generated file shows the launch line
     `Kokkos::parallel_for(label, gk::policy<Space, Rank>(…), gk::kernel<Space>(functor))`
     ("Y1"), or hides the launch in a header call as gtfn does ("Y0").
     - Two groups recommend Y1. The third describes a codegen-side launch
       plan plus header traits without choosing between the two.
     - The signed position takes no explicit stance, and this was never put
       to sign-off.
   - **Under Y1** the visible `parallel_for` line still delegates the CPU row
     loop to a header adapter (`gk::kernel<Space>`), because a bare policy
     cannot express that shape.
9. **Whether to start at all.** The go/no-go comes after the spike, on
   generated-kernel numbers.

## Alternatives considered

The position evolved over four stages:

1. **First round (three groups, analysis only).** Two groups kept `gtfn_ir`
   with a thin C++ iterator header. One proposed a new explicit-index IR
   lowered from GTIR.
2. **Second round** asked whether a dedicated IR would be better for
   performance. All three groups answered no, after first CPU measurements
   against `gtfn_cpu`, and the group that had proposed a new IR reversed.
3. **The GH200 run** contradicted "the iterator is free" for neighbor accesses
   and chunked scans. It showed that `LayoutStride` Views are costly and that
   `LayoutLeft` Views cost about 5 % on GPU scans. It also found the GPU miss
   on scans with temporaries.
4. **The reconciliation round** merged the positions.

Details are in the [[personal/egparedes/kokkos-backend/kokkos-backend_research|research appendix]].

| Alternative | Verdict |
|---|---|
| Kokkos executor inside `gridtools::fn` (plug a Kokkos backend into GridTools' executor layer) | cheapest (about 7–14 person-weeks, or 1–3 by one estimate), but **violates R2** |
| Re-implement the `gridtools::fn` API on Kokkos (a shim) | rebuilds about 6–10k lines of GridTools semantics, hides parallelism in the runtime and has template-heavy compile times; dominated |
| DaCe (SDFG → Kokkos) | DaCe has no Kokkos target, and generated code would need DaCe runtime headers; a new DaCe target is larger than this project |
| pykokkos | Python → Kokkos translator that bypasses GTIR |
| MLIR/EmitC, `std::mdspan` | no Kokkos dialect; Kokkos already fills the role |
| New explicit-index IR lowered from GTIR (round 1, one group) | duplicates the 732-line `itir_to_gtfn_ir.py` lowering with no measured gain; withdrawn by its proposers in round 2 |
| Dedicated Kokkos IR after `gtfn_ir` | +0–15 person-weeks for headroom only (scan fusion with scratch, launch grouping); no measured miss needs it |
| Keep gtfn's iterators, thin C++ header (round-1 baseline of two groups) | **measured out** for neighbor accesses (1.6–2.0x on GPU, 2.2–3.8x on CPU, depending on the form) and chunked scans (1.3–2.9x); kept only as a flagged fallback |
| `Kokkos::View<…, LayoutStride, Unmanaged>` in kernels (round-1 baseline) | up to +10 % on stencils, up to 2x on `chain` |
| `LayoutLeft` Views in kernels | ties the pointer except on GPU scans (+≈ 5 %); open decision 1 |
| `Kokkos::parallel_scan` for vertical scans | requires associativity; gt4py scans are arbitrary recurrences |
| `MDRangePolicy` on CPU | 1.0–2.1x slower than row loops on Kokkos 5.2.2; may change with 5.3.0 |
| Per-call (zero-initialized) temporaries | 2.7–3.4x on GPU; worse than `gtfn_fixed` on CPU from 18 threads |
| `KOKKOS_LAMBDA` kernels | unmeasured; named functors avoid nvcc's extended-lambda restrictions with gtfn's generic-lambda skeleton |

## Relation to other proposals

- **[[personal/egparedes/otf-toolchain-split|Dissolving gt4py.next.otf]]: overlap,
  with a sequencing question.**
  - That proposal treats the nanobind bindings generator as ADR 0011's shared
    bindings step for "any future nanobind-bound C++ backend", while noting it
    has exactly one consumer. This backend would be that second consumer.
  - Today, however, `otf/binding/nanobind.py` emits GridTools SID code
    (`as_sid`, `shift_sid_origin`, `fn::make_tuple`). A Kokkos backend must
    either parameterize the generator or fork it, and forking would undercut
    the infra split.
  - Its **bindings parameter surgery**, `BufferParameter(…, dims:
    DimSpec(name, horizontal, static_stride))`, is the right vocabulary for the
    accessor's static unit stride. It would also need a place for the
    bind-time offset width.
  - If the split lands first, the Kokkos bindings target `infra/bindings`.
    Otherwise they add a second backend to migrate.
- **[[personal/egparedes/layered-architecture/layered-architecture|Layered architecture]].**
  - The Kokkos code generator and the resolution pass are core
    (`program_processors/codegens`), next to gtfn.
  - The shipped C++ header is package data of that codegen.
  - Generated CMake via `find_package(Kokkos)` belongs to the build
    infrastructure, and forces the ADR 0016 library-adaptor redesign that the
    in-tree TODO in `cmake_lists.py` asks for.
  - Nothing here conflicts with the layering.
- **[[shared/external-memory-for-dace-arrays/external-memory-for-dace-arrays|External workspace memory for DaCe temporaries]]
  (ADR 0027 in gt4py): overlap.**
  - That design gives DaCe temporaries a caller-provided workspace, sized
    once, owned by the caller and reused across programs.
  - The Kokkos pool (§4) solves the same per-call-allocation problem
    internally.
  - **Open question:** should the Kokkos backend expose the same external
    workspace interface, so that icon4py drives both backends identically? If
    so, the aligned layout rules of §4 must hold inside the caller's buffer.
- **[[personal/havogt/scan-redesign/scan-redesign|Scan redesign]] and
  [[personal/tehrengruber/scan-design/scan-design|scan design]]: potential
  conflict.**
  - The scan traits (§5) and the resolver assume today's `scan_operator`
    lowering to gtfn column stages (`scan_pass`, launched per stage).
  - Level-slice scans, K-local views, vertical windows and "scan fusion as a
    toolchain obligation" would change what reaches `gtfn_ir`.
  - The fusion-legality check proposed here (no horizontal shift of an
    earlier stage's output) duplicates what the scan redesign assigns to the
    toolchain. It should live in one place, upstream of both C++ backends.
- **[[personal/egparedes/connectivities-as-types/connectivities-as-types|Connectivities as types]]
  (tagged `implemented`): dependency.**
  - It changes how gtfn emits offset tags and neighbor-table keys:
    `TagDefinition` per offset tag, `connectivity_key_over`, and the
    `gtfn_module.py:95` sparse-argument path.
  - Its PR stack was not merged at the study commit. `c56efffdd` has no
    `connectivity_key_over`, and the benchmark programs still declare
    `FieldOffset`s.
  - The Kokkos translation step reuses gtfn's argument processing. If the
    stack lands first, the parameterization must follow it rather than the
    line numbers above.
- **[[personal/egparedes/next-dsl-documentation/next-dsl-documentation|DSL specification]].**
  The Kokkos backend must match gtfn's semantics, for example skip values and
  `can_deref`. The bit-for-bit comparison with gtfn used here is a useful
  conformance gate alongside the test matrix.

## Recorded dissents

The groups are numbered here (1–3); their composition is process detail.
The sign-off of draft 3 was given by each group's orchestrator. Two groups did
not re-consult their analysts for that round; the third consulted four
analysts in the previous round.

| Group | Remaining dissents on the signed text |
|---|---|
| 1 | The two-kernel GPU default ships solves with temporaries outside the bar (1.030 at xlarge) until the rerun. Central effort about 30 (one analyst about 31). |
| 2 | Scan bodies in the resolver cost +2–3 person-weeks over neighbor accesses only. The CPU spread between allocations at 1–18 threads is 1–5 % (worst 14 %), so ε should be revisited once three allocations exist. |
| 3 | Two analysts keep the iterator form for Cartesian kernels. The group would gate CPU families with temporaries on `gtfn_fixed` now. One analyst proposes ε of 1 % on GPU and 5–6 % on CPU on the upper 95 % bound. Effort floor about 31, central about 36–37 with the chunk emitter in the baseline; 22 is not reachable at this bar. Two analysts want a mesh-ordering heuristic now. One analyst asks for a three-week spike. |

## Appendices

- [[personal/egparedes/kokkos-backend/kokkos-backend_benchmarks|Benchmarks]]:
  setup, compiler flags, baselines, kernels and sizes, method, every
  candidate table, the findings (with the reconciliation round's corrections
  applied), the temporary-layout follow-up, limits and how far to trust the
  numbers.
- [[personal/egparedes/kokkos-backend/kokkos-backend_versions|Kokkos and compiler version sensitivity]]:
  desk research on how far another Kokkos version, CPU compiler, CUDA release
  or AMD would move the results, and the suggested minimum pin.
- [[personal/egparedes/kokkos-backend/kokkos-backend_research|Research]]:
  how the study was run; the first-round findings that still stand (the
  GridTools leak inventory, Kokkos lifetime, build and provisioning); the
  superseded first-round designs and estimates; and side findings in gt4py.
- `kokkos-backend/bench/`: the hand-written Kokkos benchmark kernels
  (`kk/`), the gt4py baseline programs (`progs.py`, `progs_fixed.py`), the
  GridTools allocator patch (`patch_gt.py`) and the harness (`bench.py`,
  `cases.py`, `ct.py`, `asm.py`, `gtfn_compile.py`). They are **benchmark
  kernels, not generator output**. They define the source forms that the
  findings compare.
