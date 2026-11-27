---
title: "Kokkos backend — research: how the study was run, first-round findings and superseded designs"
author: egparedes
tags: [kokkos, backend, gtfn, gtfn-ir, bindings, nanobind, build-system, otf, gridtools, effort-estimate, prior-art, research]
created: 2026-11-27
status: draft
---

> **Appendix** to [[personal/egparedes/kokkos-backend/kokkos-backend|A Kokkos C++ backend for gt4py.next]].
> This appendix covers three things:
> - how the recommendation was reached;
> - the first-round findings that still stand (the GridTools leak inventory, the
>   reuse inventory, Kokkos lifetime, build and provisioning);
> - the first-round designs and estimates that the measurements **superseded**.
>
> Where this appendix and the proposal disagree, the proposal (the signed common
> position) wins. File references are to gt4py
> [`c56efffdd`](https://github.com/GridTools/gt4py/tree/c56efffdd02380d98439be3f7877d7d9a287c8d5).

## 1. How the study was run

Three independent study groups worked on the same brief. Each group was a set of
agent-based analysts coordinated by an orchestrating agent, and each analyst had
the gt4py source. A separate agent compared the groups and coordinated the
reconciliation. All of this is AI-generated analysis, so the proposal stays at
`status: draft` until a person reviews it. Which models were used is process
detail and is not recorded here.

| Stage | What happened | Outcome |
|---|---|---|
| Round 1 | Analysis only, against requirements R1-R4 of the proposal. Nothing was compiled for a GPU. | Two groups kept `gtfn_ir` with a thin C++ iterator header; one proposed a new explicit-index IR lowered from GTIR. Central estimates of about 22-27 person-weeks for comparable scopes. |
| Round 2 | Question: "would a dedicated Kokkos IR be better for performance?" Groups ran first CPU measurements against `gtfn_cpu` on workstations. | All three answered no. The group that had proposed a new IR reversed. |
| Benchmark run | Hand-written Kokkos kernels against `gtfn_cpu`/`gtfn_gpu` on one GH200 node: seven allocations, about 55 minutes of node time ([[personal/egparedes/kokkos-backend/kokkos-backend_benchmarks\|benchmark appendix]]). | Contradicted "the iterator is free" for neighbor accesses and chunked scans. Showed `LayoutStride` Views are costly and `LayoutLeft` Views cost about 5 % on GPU scans. Found the GPU miss on vertical solves with temporaries. |
| Reconciliation, round A | Each group revised its position against the measurements. | Three position papers. |
| Reconciliation, round B | The comparing agent merged the three positions into one text (three drafts, with ACK/NACK per item). | Signed by all three groups, with the dissents listed in the proposal. Each signature was the group orchestrator's reply. Two groups did not re-consult their analysts in this round; the third had consulted four analysts in round A. |
| After sign-off | Desk survey of Kokkos and compiler version sensitivity ([[personal/egparedes/kokkos-backend/kokkos-backend_versions\|versions appendix]]). | Not part of the signed text. |

## 2. First-round verdict on the hypothesis

All three groups gave the same verdict: **the hypothesis holds for the Python
pipeline and fails for the code generator.**

- **Reusable as is.** These parts carry over unchanged:
  - the GTIR pass pipeline (`apply_common_transforms`, including
    `unroll_reduce`, temporaries and fusion) and `GTFN_lowering`;
  - the OTF workflow (stages, recipes, translation → bindings → compilation),
    `compiled_program`, `CPPCompiler`, the cache and compiledb;
  - argument conversion (`convert_args`, `extract_connectivity_args`),
    allocators, the backend/toolchain builders (ADR 0016/0017/0028), and the
    test harness with its exclusion matrix, including the compile-free formatter
    gate.
- **Needs parameterisation.** These are the GridTools strings in Python code,
  listed in §3.
- **Must be rewritten.** The whole C++ side:
  - the `codegens/gtfn/codegen.py` templates and the `gridtools::fn` runtime
    they rely on;
  - the SID half of the nanobind bindings.

  gtfn's semantics (iterators, `shift`/`deref`, `can_deref`, executors) live in
  GridTools C++: about 1.5k lines for `fn` and 3.1k for `sid`, plus `meta`,
  `common`, `hymap` and `tuple_util`. A generated gtfn program pulls in about
  160-184 GridTools headers, roughly 21-29k lines, measured by one group with
  `g++ -H`/`-M`.

## 3. GridTools leak inventory (verified at `c56efffdd`)

| File:line | What | Change for Kokkos |
|---|---|---|
| `program_processors/codegens/gtfn/codegen.py` (332 lines, 21 lines mentioning `gridtools`) | templates on `gridtools::fn`: `stencil_executor`/`vertical_executor`, `scan_pass`, `deref`/`shift`/`can_deref`, `hymap`, `sid::composite`, `allocate_global_tmp`, `GT_FUNCTION`, `_c` literals | new Kokkos codegen next to it |
| `codegens/gtfn/gtfn_module.py:95` | `gridtools::sid::dimension_to_tuple_like<…>` for sparse-field arguments | parameterise (overridable constants) |
| `codegens/gtfn/gtfn_module.py:125-132` | `gridtools::fn::sid_neighbor_table::as_neighbor_table<…>`, `gridtools::hymap::keys<…>::make_values` | parameterise |
| `codegens/gtfn/gtfn_module.py:213-214` | `#include <gridtools/fn/backend/…>`, `#include <gridtools/sid/dimension_to_tuple_like.hpp>` | parameterise |
| `codegens/gtfn/gtfn_module.py:230-256` | backend header, backend type (`gridtools::fn::backend::gpu<…>{}` / `naive{}`), library name (`gridtools_gpu` / `gridtools_cpu`) | parameterise |
| `codegens/gtfn/itir_to_gtfn_ir.py:723` | `::gridtools::tuple<…>` as the C++ type of tuple temporaries | gt4py-owned tuple type |
| `otf/binding/nanobind.py:100` | `gridtools::nanobind::dynamic_size` in the nanobind shape | replace |
| `otf/binding/nanobind.py:144` | `cudaDeviceSynchronize()` for device programs | `Kokkos::fence()` |
| `otf/binding/nanobind.py:181-203` | `stride_spec`, `as_sid`, `shift_sid_origin`, `rename_numbered_dimensions`, `fn::make_tuple`, `tuple_util::get` | accessor + unmanaged Views |
| `otf/binding/nanobind.py:213-221` | `static_stride=1` for every horizontal dimension when `unstructured_horizontal_has_unit_stride` is set | reuse (see §7 for a side finding) |
| `otf/binding/nanobind.py:267-274` | GridTools includes (`sid/composite.hpp`, `fn/unstructured.hpp`, `storage/adapter/nanobind_adapter.hpp`, …) | replace |
| `otf/compilation/build_systems/cmake_lists.py:66` | `target_compile_features(… cxx_std_17)` | parameterise (Kokkos 5.x: C++20) |
| `otf/compilation/build_systems/cmake_lists.py:79-115` | closed `match` on `nanobind`, `gridtools_cpu`, `gridtools_gpu`; in-tree TODO: "do not add more libraries here … refer to ADR-0016" | ADR 0016 library adaptor |
| `otf/compilation/cache.py:49` | `TRANSLATION_CACHE_BACKENDS = ("dace", "gtfn")` | add the Kokkos backend |

The first-round groups also pointed at `itir_to_gtfn_ir.py:59-60` and `:717-731`
for small constants. Those line ranges were not re-checked for this appendix.

## 4. Runtime, build and provisioning findings

None of these were contradicted by the measurements. Only the benchmark build
itself was exercised: shared Kokkos 5.2.2 libraries, a host build and a CUDA
build, `nvcc_wrapper`. Several items are therefore still spike items.

- **Lifetime.**
  - A shared `libkokkoscore` is a hard requirement. A static PIC Kokkos gives
    one runtime per extension module.
  - Initialization is lazy and thread-safe, guarded by `is_initialized()`, and
    attaches if someone else initialized Kokkos. A second `initialize` aborts.
  - v1 never calls `finalize()`, because Python-held Views and the temporary
    pool can outlive any exit hook. One first-round analyst proposed
    "finalize only if owned"; that was superseded.
  - Compile workers never initialize.
  - There is a build-time guard and a clear error if the environment statically
    links a second Kokkos.
  - **Untested:** two separately compiled program modules sharing one runtime,
    a CPU module and a CUDA module loaded in one process, and OpenMP + CUDA
    spaces from one install. These are spike items.
- **Streams.** Kokkos' default CUDA instance creates its own stream with
  `cudaStreamCreate`. v1 runs on one device, on the default instance, and fences
  after each call. Ordering against CuPy streams must be documented. Passing
  CuPy's stream (`Kokkos::Cuda(stream)`) is production work.
- **Bindings.**
  - DLPack/nanobind strides are in elements: `nanobind/ndarray.h:289`, and the
    GridTools adapter copies them undivided.
  - gt4py's allocators make the first horizontal dimension unit-stride on CPU
    and GPU, with padded runtime strides elsewhere (for example (I, J, K)
    10×12×7 → strides (1, 16, 192)).
  - Iterating with the horizontal dimension as rank 0 matters: the wrong order
    costs 1.4-2.5x on CPU (first-round toy measurement).
- **Build.**
  - Kokkos' default CXX mode for CUDA (`nvcc_wrapper` /
    `kokkos_launch_compiler`) must be replayable through compiledb. The CMake
    language mode needs CMake ≥ 3.25.2 and a non-default Kokkos option, so it
    is optional and comes later.
  - `Kokkos_ARCH_*` replaces `CMAKE_CUDA_ARCHITECTURES`.
  - The compiledb `_write_files` step carries only static libraries, so Kokkos
    must be an external `find_package` install, not `FetchContent`.
  - The cache key needs a fingerprint of the Kokkos install.
- **Device code.**
  - Kernels are named functors with `KOKKOS_FUNCTION operator()`. Generic
    lambdas appear only inside device code. nvcc's extended-lambda
    restrictions clash with gtfn's generic-lambda skeleton.
  - The generated code must not need `--expt-relaxed-constexpr`.
    `Kokkos::min`/`max` have been in `namespace Kokkos` since 3.7, with
    `(b < a) ? b : a` semantics, the same as `std::min`/`max`. inf/NaN come
    from `Kokkos::Experimental::infinity_v`/`quiet_NaN_v` behind a gt4py
    helper.
- **Provisioning.**
  - No PyPI wheel ships Kokkos with a CMake config. `pykokkos-base` is
    sdist-only and builds Kokkos itself.
  - The Debian/Ubuntu package (3.7) is too old. Spack, conda or source builds
    remain.
  - GPU CI is only available at CSCS. One analyst asked that the Linux CPU-CI
    Kokkos install (spack or a cached source build) be priced explicitly.
  - `DeviceType` has only CPU, CUDA and ROCM, so SYCL is out of scope without
    core changes.
- **"Self-contained" (R2).** All groups read it as Kokkos + std + nanobind + a
  small gt4py-owned header shipped as package data. If that is not acceptable,
  the fallback is to inline the header into each generated translation unit,
  at a compile-time cost only.

## 5. First-round designs and estimates (superseded)

The table below records the designs and estimates as they stood before the
measurements. The measurements changed the accessor, the iterator treatment and
the temporaries, and the estimates rose accordingly. See the proposal for the
signed figures.

| Group | Round-1 design | Round-1 estimate (person-weeks) | What the measurements changed |
|---|---|---|---|
| 1 | `gtfn_ir` + new codegen; header-only runtime with a thin value-type iterator; unmanaged `LayoutStride` Views at the allocation base with a static-stride fast path; temporaries as managed Views from a caching pool; `MDRangePolicy` everywhere (tile 32×8×1 from a launch plan) | CPU + CUDA 16.5-28, central about 22; HIP +2-3; the GridTools plug-in route about 7.5-20 (indicative) | iterator resolved in Python; pointer accessor; row loops on CPU |
| 2 | "X-T2 + Y1": `gtfn_ir`, stencil bodies keep gtfn's shape against a lean 1.0-1.5k-line iterator header, the compiler does the partial evaluation; launches spelled `Kokkos::parallel_for(label, gk::policy<Space, Rank>(lo, hi), gk::kernel<Space>(functor))`, with the policy, host loop-shape adapter, tile and index type from header traits; a custom pointer + strides accessor for user fields (already preferred over `LayoutStride` from CPU toys) | MVP (correctness only) about 20-34, likely about 26; production (HIP, ≤ 1.2x gtfn, robustness) about 31-54, likely about 40 | X-T3 (explicit index expressions from a Python partial evaluator) moved from fallback to baseline |
| 3 | sibling backend: GTIR → new explicit-index Kokkos IR → C++ via eve `TemplatedGenerator`; rolled neighbor loops with an explicit skip-value mask (`unroll_reduce=False`); `KOKKOS_LAMBDA` by default; `LayoutStride` Views with a dim-0 unit-stride fast path; per-call `WithoutInitializing` temporaries in the CPU MVP, a cached allocator in the GPU phase | MVP (CPU, Cartesian only) about 12; full (CPU + CUDA, unstructured, scans, parity test matrix, performance comparable to gtfn) about 27 (18-42); HIP +2; maintenance 0.3-0.5 FTE | reversed to `gtfn_ir` in round 2; pooled temporaries; named functors |

The signed text puts its figure of 30-37 (22-45) at about 4-8 above the
previous round's estimates. It attributes the increase to three changes:
- the resolver moving into the baseline, including scan bodies;
- the aligned pool;
- the scan traits.

Group 2's estimate is broken down further below.

### Group 2: how visible is Kokkos in the generated file?

| Option | Generated file contains | Extra MVP effort | Same text on all platforms |
|---|---|---|---|
| Y0 (like gtfn) | functors with `KOKKOS_INLINE_FUNCTION` and a `gk::launch<Space>(…)` call; `parallel_for` and policies only in the header | 0 | yes |
| Y1 (group 2's recommendation) | `Kokkos::parallel_for(label, gk::policy<Space,Rank>(lo,hi), gk::kernel<Space>(functor))`; the `gk::kernel` host adapter wraps the functor into the outer-range / inner-horizontal loop shape, which a bare policy cannot express | +0.5-3.5 including the adapter | yes |
| Y2 | fully spelled `Kokkos::MDRangePolicy<Space, Rank<2,Left,Left>, IndexType<int>>` and typed Views | +1-3 | only with one policy for all spaces; 0-26 % CPU risk otherwise |

The measurements exclude Y2: a bare `MDRangePolicy` costs 1.0-2.1x on CPU, so
the policy type and the CPU loop shape must come from header traits.

Between Y0 and Y1, the signed position takes no stance, and the choice was never
put to sign-off. Where each group stands:
- Group 2 recommended Y1 in its first consensus and again in its IR proposal.
- Group 3 moved to Y1 in its IR proposal, with `parallel_for` spelled out and
  the policy and loop shape from traits, and listed it as a user decision.
- Group 1 describes a codegen-side launch plan plus header traits without
  choosing between the two.

The proposal lists the remaining choice as open decision 8.

### Group 2: first-round decision gate (for reference)

| Gate | Pass | Fail → effect |
|---|---|---|
| G1: nvcc accepts the generated shapes as named functors | compiles | emit concrete functors or explicit indexing: +4-8 person-weeks |
| G2: compile time per program | ≤ 2x gtfn GPU (≲ 60 s) | header slimming: +2-5 |
| G3: GPU kernel time vs hand-written index code | ≤ 1.3x | 1.3-1.5x: production +2-4; > 1.5x: explicit indexing for hot paths, +5-10 |
| G4: CuPy stream ordering | correct, no extra sync | +1 |
| G5: CXX-mode Kokkos CUDA build through gt4py's CMake/compiledb | works | +1-2 |

G3 was effectively run by the benchmark: the iterator form exceeded 1.5x for
neighbor accesses in some forms. That is why explicit indexing is now the
baseline. G1, G2 (for generated code), G4 and G5 remain open.

## 6. First-round risks (merged)

Ranked differently by each group:

1. **GPU performance parity** of Kokkos launches against GridTools' tuned GPU
   backend. This is now partly measured: see the benchmark appendix.
2. **The Kokkos lifetime** with many extension modules per Python process, and
   CuPy interop.
3. **Kokkos provisioning and packaging, and CI.** There is no wheel, a shared
   library is required, and GPU runners are scarce.
4. **The long tail of the test matrix.** Tuples, local and sparse dimensions,
   scans with tuples, dynamic offsets and `concat_where` are the largest source
   of variance for bring-up. For the resolver, iterator `let`s,
   `concat_where` and domain expressions are the coverage risk.
5. **Device standard-library portability** (builtins).
6. **Compile time** of Kokkos headers per generated program. One-kernel units
   took 6.9-8.5 s, of which headers alone took 6.6-6.9 s **[M]**. That is not
   a backend-level comparison.
7. **Binding correctness**: strides, origins and the static-stride fast path.
   Silent corruption calls for golden tests.
8. **Two C++ backends** to keep in sync with GTIR evolution.

## 7. Side findings in gt4py (pre-existing, independent of Kokkos)

- **`horizontal_first_layout_mapper`'s docstring is wrong.** The docstring
  (`custom_layout_allocators.py:161`) says horizontal dimensions "change the
  slowest (i.e. larger strides)". In practice the first horizontal dimension is
  contiguous [verified by two groups].
- **The unit-stride option breaks Cartesian fields.**
  `unstructured_horizontal_has_unit_stride=True` gives static stride 1 to
  **every** horizontal dimension, including Cartesian I and J
  (`nanobind.py:213-221`). For a Cartesian (I, J) field the adapter would then
  throw `std::invalid_argument` at call time. The throw was read from
  `nanobind_adapter.hpp`, not reproduced. The fix is about 0.1 person-weeks.
- **CSE leaves iterator-typed `let`s.** `iterator/transforms/cse.py` collects
  applied shifts (only partial `shift(offsets)` calls are excluded), producing
  iterator-typed `let`s such as `(λ(_cs_0) → …)(⟪I,1⟫(inp))`. This is harmless
  for gtfn and relevant for the resolution pass. A separate upstream fix was
  recommended by one analyst.
- **Two unverified notes from one analyst.** The compiledb key lacks
  `build_context_id`, and `cmake.py` mutates `cmake_extra_flags` on CUDA.
