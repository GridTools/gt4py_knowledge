---
title: "Kokkos backend — how far could another Kokkos version or compiler move the results?"
author: egparedes
tags: [kokkos, benchmarks, performance, performance-portability, gpu, cuda, hip, openmp, compilers, versions, prior-art]
created: 2026-11-27
status: draft
---

> **Appendix** to [[personal/egparedes/kokkos-backend/kokkos-backend|A Kokkos C++ backend for gt4py.next]].
> This is desk research done after the study groups signed their common position, so it is **not
> part of the signed text**. Nothing was built or run. It asks how far the
> [[personal/egparedes/kokkos-backend/kokkos-backend_benchmarks|benchmark]] numbers (Kokkos 5.2.2,
> GCC 14.2.0, CUDA 12.9.41, no `-mcpu`) carry over to other Kokkos versions, compilers and
> vendors. "Our" and "I" below refer to the benchmark run and to the researcher who wrote this.
> Section references such as "(b)", "(d)" and "(e)" point to section 3 of the benchmark appendix.
> **[D]** = stated in the linked source (including source code read or compared); **[I]** = inference.

## What this means for the proposal

- **[I] Kokkos 5.3.0** removes an index narrowing that stopped GCC vectorising host
  `MDRangePolicy` (documented; no speedup figure found). "Row loops beat MDRange on CPU" may
  therefore weaken on 5.3. The row-loop candidates themselves are unaffected.
- **[D] The GPU `MDRangePolicy`** was rewritten in 5.1 and changed again in 5.2, so the GPU stencil
  ratios should not be assumed for older Kokkos versions **[I]**.
- **[D] The View implementation** changed repeatedly from 4.7 to 5.3, so "a `LayoutLeft` View ties
  the pointer" is established for 5.2.2 only **[I]**.
- **[I] GCC 14.2 observations.** The iterator-on-stack and unvectorised-chunk results are GCC 14.2
  observations. Clang and `-mcpu=neoverse-v2` could move the unstructured ratios either way; no
  controlled comparison was found.
- **Not covered.** Nothing was found on CUDA 12.9 against 13.x for such kernels, and AMD is
  unmeasured.
- **Suggested spike additions.** Three runs:
  - Kokkos 5.3.0 on the same GCC and CUDA;
  - Clang with libomp and `-mcpu=neoverse-v2` on both sides, plus a GCC control with the same flags;
  - the newest CUDA 13.x.
- **Suggested minimum pin.** Kokkos 5.2.2, raised to 5.3.0 if that rerun shows no regression. Any
  5.x pin implies C++20 (open decision 4 of the proposal).

Desk research, 2026-11-27; nothing was built or run. Baseline: Kokkos 5.2.2, GCC 14.2.0,
CUDA 12.9.41. **[D]** = stated in the linked source (including source code I read or compared).
**[I]** = my inference. Confidence is in the claim *as it applies to our kernels*.

**Reading rule [I].** A Kokkos version change moves only the Kokkos side of every ratio. A compiler
change moves gtfn too; it changes a verdict mainly where the result depends on source form
(section 3 (b), (d), (e)) or where the margin is small (`hdiff`/`tmpst` 0.98-0.99, `tri`
1.00-1.02, `tritmp` 1.03-1.05).

## Summary

| # | change | direction, size | conf. | moves | conclusion it could overturn |
|---|---|---|---|---|---|
| 1 | Kokkos 5.3.0: host `MDRangePolicy` can auto-vectorise with GCC | MDRange faster on CPU; size unpublished | mech. high, size none | Kokkos | (d) "rows beat MDRange"; "64-bit MDRange 0.58-0.95 of int" |
| 2 | Kokkos older than 5.1 / 5.2: GPU MDRange | older is slower; one 3D stencil gained 43 % in 5.2 alone | medium | Kokkos | GPU stencil verdicts, if an older pin were allowed |
| 3 | `View` implementation differs by version | operator cost for our kernels unknown elsewhere | medium | Kokkos | (c) `LayoutLeft` View = pointer (shown for 5.2.2 only) |
| 4 | GCC misses vectorisation of some `int`-indexed loops (open bug) | scalar instead of vector loop | high (bug), low (ours) | both, by source form | (b), emitter design |
| 5 | Non-GCC host compiler changes Kokkos' OpenMP schedule | sign and size unknown | high (code), low (effect) | Kokkos | (a), (d) CPU multi-thread |
| 6 | Another CPU compiler | up to 3.3x spread in one user report (x86) | low | both, unequally | (b), (e) chunked scans, thin CPU margins |
| 7 | `-mcpu=neoverse-v2` (SVE2, gathers) | either sign; gathers can lose | low | both | `v2e`, `chain`, chunked scans |
| 8 | libgomp vs libomp | libgomp parallel-region start is dearer at 72+ threads | medium | both | small size at 36-72 threads |
| 9 | CUDA 13.x instead of 12.9 | found no data | none | both | thin GPU margins; (b) GPU |
| 10 | Clang as CUDA compiler | mostly near nvcc; one Kokkos kernel 26-28 % slower | low | both | same as 9 |
| 11 | AMD / HIP | not measured; Kokkos and ROCm both have version-specific regressions | low | - | none measured |

CPU row-loop candidates are largely insulated from item 1: the `RangePolicy` loop in
[`Kokkos_OpenMP_Parallel_For.hpp`](https://github.com/kokkos/kokkos/blob/5.3.0/core/src/OpenMP/Kokkos_OpenMP_Parallel_For.hpp)
is unchanged from 5.2.2 to 5.3.0 **[D]**. GPU tiled candidates use MDRange and are exposed.

One default differs below 5.2 **[D]**: a policy without `IndexType` used the unsigned `size_type`
([5.1.0](https://github.com/kokkos/kokkos/blob/5.1.0/core/src/impl/Kokkos_AnalyzePolicy.hpp)) and
uses the signed `index_type` from
[5.2](https://github.com/kokkos/kokkos/blob/5.2.2/core/src/impl/Kokkos_AnalyzePolicy.hpp), for
`RangePolicy` too. **[I]** Row loops that rely on the default could vectorise differently on old pins.

## 1. Kokkos versions

Releases: 5.0.x, 5.1.0 (March 2026), 5.2.0 (July), 5.2.2 (September), 5.3.0 (11 November 2026),
the newest ([releases](https://github.com/kokkos/kokkos/releases)) **[D]**. Changelog references
below are to the [5.3.0 CHANGELOG](https://github.com/kokkos/kokkos/blob/5.3.0/CHANGELOG.md).

### MDRange on the host

* **[D]** Up to 5.2.x, `point_type` is `int64`
  ([source](https://github.com/kokkos/kokkos/blob/5.2.2/core/src/KokkosExp_MDRangePolicy.hpp)), so a
  functor taking `int` gets a narrowing conversion in the inner loop, which stops GCC vectorising
  ([PR 9351](https://github.com/kokkos/kokkos/pull/9351)).
* **[D]** 5.3.0 uses `index_type` there
  ([PR 9650](https://github.com/kokkos/kokkos/pull/9650)). Vectorisation needs signed functor
  arguments and an `IndexType` of the same width
  ([maintainer comment](https://github.com/kokkos/kokkos/issues/8652#issuecomment-6791519452)). The
  host default is signed 64-bit
  ([source](https://github.com/kokkos/kokkos/blob/5.3.0/core/src/Kokkos_HostSpace.hpp)), so **[I]**
  default policy plus `int` arguments still narrows.
* **[I]** This probably explains our "64-bit index MDRange is faster than `int`": the 64-bit build
  has no narrowing. On 5.3.0 the `int` build should catch up. Tiling overhead remains, so MDRange
  need not reach row loops. `ivdep` on these loops still needs
  `Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION`, off by default
  ([source](https://github.com/kokkos/kokkos/blob/5.3.0/core/src/impl/KokkosExp_Host_IterateTile.hpp)) **[D]**.
* **[D]** 5.3.0 also rejects an MDRange whose total extent does not fit `index_type`
  (PR 9650). **[I]** `IndexType<int>` over 2^31 or more points now fails at construction.
* **[D]** The host default-tile logic (2 per rank, full innermost range) differs only in types
  between 5.2.2 and
  [5.3.0](https://github.com/kokkos/kokkos/blob/5.3.0/core/src/KokkosExp_MDRangePolicy.hpp) (my diff), so **[I]** 5.3.0
  alone should not remove the 3-9x cost of the all-default policy.
* The paper announced for this work (P3HPC at SC 2026) I could not obtain; only its
  [artifact instructions](https://github.com/kokkos/code-examples/tree/main/papers/kokkos-mdrange-2026).

### MDRange on CUDA / HIP

* **[D]** 5.1.0 rewrote device iteration so tile = thread block
  ([PR 8638](https://github.com/kokkos/kokkos/pull/8638): rank-3 STREAM add, A100, CUDA 12.4, 34 to
  22 registers) and changed default tiles, rank 3 from (16,2,2) to (32,2,4)
  ([PR 8731](https://github.com/kokkos/kokkos/pull/8731)).
* **[D]** 5.2.0 skips the grid-stride loop when the grid covers the range
  ([PR 9142](https://github.com/kokkos/kokkos/pull/9142)). PolyBench heat-3d, A100, CUDA 12.8: 815
  GB/s and 48 registers on 5.1, 1169 and 28 with the change, native CUDA 1189.
* **[I]** On any pin below 5.2.0, our 0.95-0.99 GPU stencil ratios should not be assumed. Our
  "default tile" rows describe 5.1+ defaults only; explicit `{32,8,1}` is unaffected by defaults.
* **[D]** 5.3.0: reduction grid sizes and sub-warp blocks (changelog PRs 9523, 9339). **[I]** No
  effect expected on our `parallel_for`; the narrower `point_type` could shift GPU code slightly.
* Launch bounds: found no relevant change (we measured no effect).

### `View`

* **[D]** 4.7.0 introduced the `mdspan`-based View; 4.7.1 reverted the default to the old one
  "to avoid performance regressions"; 5.0.0 re-enabled it with an index-computation workaround
  ([PR 8476](https://github.com/kokkos/kokkos/pull/8476)).
* **[D]** 5.0.0 still defaults to the old View for nvcc 12.9
  ([CMake](https://github.com/kokkos/kokkos/blob/5.0.0/cmake/kokkos_enable_options.cmake)); 5.0.1
  switches; 5.0.2 fixes slow `subview` ([PR 8822](https://github.com/kokkos/kokkos/pull/8822), about
  3x in its reproducer); 5.2.0 adds explicit `index_type`; 5.2.1 fixes an inverted `NDEBUG` test in
  `mdspan` precondition checks (changelog). A 2-2.5x application slowdown from
  unmanaged `DynRankView` construction in kernels, seen on 5.1.1, was closed as fixed in 5.2.1
  ([issue 9368](https://github.com/kokkos/kokkos/issues/9368)). 5.3.0 removes the old View.
* **[D]** On H100 the old View was reported consistently faster in a 4D test, while another tester
  found differences within noise ([issue 8430](https://github.com/kokkos/kokkos/issues/8430)).
* `LayoutStride` cost across versions: found nothing.

### OpenMP, launch, allocation, requirements

* **[D]** For static `RangePolicy`, Kokkos omits the OpenMP chunk clause under GCC ("leads to
  performance regression") but passes `schedule(static, chunk)` for other compilers except NVHPC
  ([source](https://github.com/kokkos/kokkos/blob/5.3.0/core/src/OpenMP/Kokkos_OpenMP_Parallel_For.hpp)),
  with an [automatic chunk](https://github.com/kokkos/kokkos/blob/5.2.2/core/src/Kokkos_ExecPolicy.hpp).
  The branch exists from at least 4.7.1. **[I]** Under Clang, 40960 rows on 72 threads are dealt
  round-robin in chunks of 16, unlike gtfn's static split; set the chunk size or check locality.
* **[D]** 5.0.0 launches via `__grid_constant__` (nvcc, CUDA 12.2+, not Clang); per its author it
  matters only for functors above about 200 bytes ([PR 8529](https://github.com/kokkos/kokkos/pull/8529)).
  **[I]** Could shift only small-size GPU ratios.
* **[D]** `cudaMallocAsync` has been off by default since 4.5
  ([PR 7353](https://github.com/kokkos/kokkos/pull/7353)), still opt-in in
  [5.3.0](https://github.com/kokkos/kokkos/blob/5.3.0/core/src/Cuda/Kokkos_CudaSpace.cpp).
  **[I]** Matters only for per-call allocation, not pooled temporaries.
* **[D]** [Minimums](https://kokkos.org/kokkos-core-wiki/get-started/requirements.html) for 5.x:
  C++20, GCC 10.4, Clang 14 (CPU), Clang 15 with CUDA 11.8, nvcc 12.2, NVC++ 22.3, ROCm 6.2,
  Arm Clang 20.1. 4.x: C++17, GCC 8.2, nvcc 11.0, ROCm 5.2. CUDA 13.1 needs a compile fix present
  in 4.7.3 and 5.0.2 (changelog).
* No published version-to-version benchmark of host `RangePolicy` or flat GPU `RangePolicy` found.

## 2. CPU compilers (mainly Neoverse V2)

* **[D]** [GCC bug 128968](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=128968) (filed 23 November
  2026, open): an `int` loop whose index is `int(j + offset)`, with unsigned or 64-bit `offset`, is
  not vectorised by GCC mainline; Clang 23 vectorises it. **[I]** Generated code mixing `int`
  indices and wide offsets can hit this. Our `vl` rows (0.95-1.01 of pointer) show no sign of it.
* **[I]** Iterator objects kept on the stack and the unvectorised chunked iterator scan (1.3-3.8x at
  large size) are GCC 14.2 observations. Another compiler or release could remove or move them;
  I found no matching bug report or fix. gtfn is iterator code too, so its time may also move.
* **[D]** One user report, one EPYC 7543 core, same Views, three `RangePolicy` stencils with inner
  loops: GCC 12.4 1.04 s, Intel 2024.2 2.33 s, NVHPC 25.9 3.46 s; three MDRange kernels 3.1 / 3.2 /
  3.7 s ([comment](https://github.com/kokkos/kokkos/issues/8652#issuecomment-3503953150)). Not
  controlled, not aarch64.
* Aliasing: **[D]** adding `Kokkos::Restrict` to Views was reported as "not much of a help" for
  MDRange vectorisation ([comment](https://github.com/kokkos/kokkos/issues/8652#issuecomment-3806599344)).
  **[I]** Our 1.12-1.40x penalty for not copying the functor into locals depends on alias analysis,
  so it is compiler-sensitive. I searched no compiler documentation on `restrict`.
* **[D]** Grace has 128-bit SVE2/NEON units; in a 13-application study GCC 11.4 auto-vectorised
  better than Arm Compiler 23.10 for most, and memory-bound cases gained little
  ([paper](https://arxiv.org/html/2505.09462v1)).
* **[D]** NVIDIA advises `-mcpu=neoverse-v2` (GCC 12.3+, LLVM 16+) and recent compilers
  ([guide](https://nvidia.github.io/grace-cpu-benchmarking-guide/developer/languages/c-c++.html)).
  Kokkos' Grace option adds `-mcpu=neoverse-v2 -msve-vector-bits=128`
  ([CMake](https://github.com/kokkos/kokkos/blob/5.2.2/cmake/kokkos_arch.cmake)); our builds did not.
  **[I]** Contiguous `float64` stays 2 lanes, so expect modest stencil gains; gtfn must get the same
  flags or the comparison is unfair.
* **[D]** Gathers: a recent LLVM change that emits SVE gathers made a reduced kernel 2.34x slower on
  Neoverse V2 ([LLVM 245634](https://github.com/llvm/llvm-project/issues/245634), open). V2 gather
  throughput is 1/4 cache line per cycle ([microbenchmarks](https://arxiv.org/pdf/2409.08108)).
  **[I]** `-mcpu` newly allows gathers in neighbor loops, so unstructured ratios could move either way.
* **[D]** libgomp's `parallel` start-up overhead at 72/144 threads is much higher than libomp's
  ([GCC 119588](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=119588), open; partial work committed
  to GCC trunk in October 2026). **[I]** Matters for many small kernels, on both sides.
* **[D]** GCC defaults to `-ffp-contract=fast` in GNU mode
  ([GCC](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)); Clang defaults to `on`
  ([Clang](https://clang.llvm.org/docs/UsersManual.html)). **[I]** Set it explicitly on both sides
  under Clang (gtfn passes none).
* x86-64 **[I]**: without `-march`, builds get SSE2 only, so our CPU ratios do not transfer to
  AVX2/AVX-512. Found no x86 comparison for these patterns beyond the user report above.
* Found nothing usable: a GCC 12-16 vs LLVM 17-23 stencil comparison on Neoverse V2; NVHPC, Arm
  Compiler or Cray CCE on these patterns on aarch64; C++17 vs C++20 effects (not searched).
  **[D]** CUDA 13.5 deprecates Arm Compiler for Linux 24.10 as host compiler
  ([notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)).

## 3. GPU compilers

* **[D]** CUDA is at 13.5 ([archive](https://developer.nvidia.com/cuda-toolkit-archive)). Its
  [notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) say Blackwell and
  newer use a newer LLVM-based backend by default, pre-Blackwell (so Hopper) can opt in
  (`-nvvm-latest`, for nvcc via `-dlto --nvlink-options`), and performance can differ between
  backend versions.
* **[D]** A [forum report](https://forums.developer.nvidia.com/t/364169) shows nvcc 13.2 spilling
  heavily for `sm_120` but not `sm_90`. Not a 12.9 comparison.
* **[D]** Compiler notes of 13.0 move `libNVVM` to LLVM 20 and add opt-in register spilling to
  shared memory ([13.0](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)).
  [12.9.1](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html) fixes
  wrong address code (large immediates, since 12.8, SM90), and
  [13.3](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-toolkit-release-notes/index.html) a
  reconvergence miscompile in nested divergence (since 12.8). **[I]** nvcc 12.9.41 looks like the
  12.9.0 release, so prefer a later one; our outputs did match gtfn bit for bit.
* Read: compiler feature/issue sections of 12.9.0, 12.9.1, 13.0.0-13.3.0 and 13.5 (not 11.x, early
  12.x or 13.4). They give no performance figure for stencil-like code and no note on load hoisting
  or 32/64-bit index arithmetic; I found no 12.9 vs 13.x Hopper comparison. **[I]** Our unexplained 0.4-3 % `tritmp` gap and the 1.6-2.0x
  neighbor-iterator penalty (load overlap) are code-generation effects that a toolkit change could
  move either way.
* **[D]** Clang 20.1 vs nvcc through Kokkos on GH200 (CUDA 12.5) and A100: native backends "mostly
  close", but the TestSNAP `compute_yi` kernel was 26-28 % slower with Clang
  ([paper](https://tianshilei.me/doc/iwomp-2025.pdf)). Similar SASS and speed for a plain CUDA FIR
  kernel, where load scheduling by ptxas decided register pressure
  ([LLVM 239427](https://github.com/llvm/llvm-project/issues/239427)).
* **[D]** For CUDA 12.9, Clang
  [22](https://github.com/llvm/llvm-project/blob/release/22.x/clang/include/clang/Basic/Cuda.h) is
  partial and
  [23](https://github.com/llvm/llvm-project/blob/release/23.x/clang/include/clang/Basic/Cuda.h)
  full; 20 and 21 stop earlier. **[I]** Use Clang 23.
* AMD **[D]**: Kokkos HIP block-size choices changed in 4.6 and 5.0 (one reduction regression,
  fixed in 5.0), 5.2 fixed a `RangePolicy` regression, async allocation is off for ROCm 7+
  (changelog), and PR 9142 covers HIP. A non-Kokkos kernel ran several times slower on ROCm 7.14
  than 7.2.4 from an LLVM spill regression
  ([report](https://github.com/moonmath-ai/amd-kernels/issues/5)). **[I]** No GH200 number carries
  over to AMD. Found no ROCm-version comparison for Kokkos stencils.

## 4. Recommendation

**Next three benchmark runs [I]**, each with gtfn rebuilt where the compiler or flags change:

1. **Kokkos 5.3.0, same GCC 14.2 / CUDA 12.9.41.** Cheapest; isolates items 1-3. CPU: MDRange with
   `IndexType` matched to the argument type, against rows. Rerun `vl`/`vs` and the thin GPU cells.
2. **CPU: LLVM/Clang 23 (or newest available) with libomp, `-mcpu=neoverse-v2 -ffp-contract=fast`
   on both sides**, plus a GCC 14.2 control with the same two flags to separate flags from compiler.
   Cover iterator forms, chunked scans, `v2e`/`chain`, small size at 72 threads, and rows with
   default and explicit chunk size. Check gathers in the assembly.
3. **GPU: newest available CUDA 13.x nvcc, Kokkos from run 1, GCC host**, all GPU kernels, recording
   registers per kernel. I found no statement that Kokkos is tested with 13.4/13.5.

Deferred: GCC 15/16 (after 2), Clang-CUDA, and ROCm (its own study if AMD is a target).

**Minimum Kokkos pin [I]: 5.2.2.** It is what we measured; it has both GPU MDRange improvements and
the 5.2.1 precondition fix. Raise the floor to **5.3.0** if run 1 shows no regression, or at once if
the emitter uses host MDRange. Caveats: 5.3.0 is 16 days old with no patch release, and one SIMD
regression was reported against it (AVX2 mask conversion, closed,
[issue 9783](https://github.com/kokkos/kokkos/issues/9783)). Either pin imposes C++20. Whatever the
version, an emitter should keep functor argument types equal to the policy `IndexType`, prefer
explicit indexing for neighbors, and check vectorisation per compiler in CI.
