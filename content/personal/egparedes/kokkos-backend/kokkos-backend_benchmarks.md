---
title: "Kokkos backend — benchmarks: hand-written Kokkos kernels against gtfn on one GH200 node"
author: egparedes
tags: [kokkos, gtfn, benchmarks, performance, gpu, cuda, openmp, field-accessor, iterator-resolution, temporary-arrays, scan, unstructured]
created: 2026-11-27
status: draft
---

> **Appendix** to [[personal/egparedes/kokkos-backend/kokkos-backend|A Kokkos C++ backend for gt4py.next]].
> This is the benchmark report the proposal rests on, kept here because the original run
> directory was temporary. It compares **hand-written** Kokkos kernels against `gtfn_cpu` and
> `gtfn_gpu` on one node with one toolchain; **no code generator exists**. Node, job and account
> identifiers, file paths of the original run and the job log are removed. The raw per-allocation
> data (JSON lines, build logs, assembly archives) are **not** preserved here; the kernel and harness
> sources are, under `kokkos-backend/bench/`. Text below is the report as written, except for the
> scrubbing and the correction notes marked **Correction (reconciliation round)**.

## Corrections from the reconciliation round

The three study groups checked this report against its own tables and agreed on these corrections.
They take precedence over the text below.

1. **Section 3(c) and the verdict table, View vs pointer on GPU scans.** The `LayoutLeft` View
   penalty on GPU scans is present **with explicit bodies as well**, not only with the iterator body
   (`tri` 1.07 vs 1.02 for the pointer, `tritmp` 1.10 vs 1.05; verified against the section 2.3
   tables).
2. **Section 3(a), CPU.** "Every gtfn-shaped candidate ≤ 1.0" holds for the **literal** iterator form
   only. The bound-once iterator form is 1.25-2.38x.
3. **Section 4, "iterator chunks match explicit chunks".** That claim came from one first-round
   study group (an iterator over pointers, on x86) and was withdrawn by that group as a general
   claim.
4. **Reading limits** all three groups add:
   - The iterator results are for one header and one compiler.
   - "Explicit is safe" is equally single-compiler and single-architecture, and rests on four
     kernel families.
   - Most CPU cells at 1-36 threads come from one allocation.
   - The 0.35x random-table cell looks like a gtfn baseline pathology.
   - The 1-thread CPU margins reflect gtfn's own loop nest, which gtfn could fix.
5. **Kernels with temporaries on CPU.** These margins are largely gtfn's double allocation and
   zero-fill (section 3(f)). They must not be quoted as a Kokkos advantage until the GridTools
   allocator fix is upstream.

## Sources kept with this proposal

| file under `kokkos-backend/bench/` | content |
|---|---|
| `kk/gk.hpp` | the small runtime of the hand-written kernels: accessors (`ptr`, `ptr64`, `vl`, `vs`), launch shapes, temporary pool, variant registry |
| `kk/cart.hpp`, `kk/unstr.hpp`, `kk/scan.hpp` | the Cartesian, unstructured and scan kernels in their iterator, explicit and pointer-walking forms |
| `kk/reg_*.cpp`, `kk/api.cpp`, `kk/CMakeLists.txt` | variant registration, the C entry point timed by the harness, the build |
| `progs.py`, `progs_fixed.py` | the gt4py.next programs used as gtfn baselines |
| `patch_gt.py` | the GridTools 2.3.9 header edit behind `gtfn_fixed` (single, uninitialised temporary allocation) |
| `bench.py`, `cases.py` | correctness check and interleaved timing of gtfn and the Kokkos variants |
| `ct.py`, `asm.py`, `gtfn_compile.py` | compile-time and register collection, single-variant assembly, gtfn compilation |

All numbers come from one node of a CSCS cluster (Grace CPU + Hopper GPU), one
toolchain, and hand-written Kokkos kernels that imitate what a `gtfn_ir` to Kokkos emitter could
produce. No code generator exists yet. Statements are marked **[M]** (measured here) or **[I]**
(inferred, from reading code or from reasoning about the measurements).

## 1. Setup

### Node and software [M]

| item | value |
|---|---|
| node | one GH200 node, the same node for every run |
| CPU | 4 sockets x 72 Neoverse-V2 cores (288 cores, no SMT), L1d 64 KiB, L2 1 MiB per core, L3 114 MiB per socket, 64 KiB pages, THP `always`, kernel 6.4.0-150600.23.143-64kb |
| NUMA | CPU memory in nodes 0-3 (one per socket, about 120 GB each); GPU memory shows up as extra nodes |
| GPU | NVIDIA GH200 120GB (97871 MiB), GPU 0 only; driver 595.91.07, max SM clock 1980 MHz, power limit 900 W, persistence mode off |
| environment | uenv `prgenv-gnu/25.11:v1`, GCC 14.2.0, CUDA 12.9.41, CMake 4.4.2, Ninja 1.13 |
| Python | 3.12.15, numpy 2.5.1, cupy-cuda12x 14.1.1, nanobind 2.13.0 |
| gt4py | `c56efffdd02380d98439be3f7877d7d9a287c8d5`, gridtools-cpp 2.3.9 |
| Kokkos | 5.2.2 (`8e15454`), shared libraries; a host build (OpenMP + Serial, no architecture flag) and a CUDA build (CUDA + OpenMP, `HOPPER90`, `nvcc_wrapper`) |

### Compiler flags [M, from the compile databases]

| build | command line |
|---|---|
| gtfn CPU | `g++ -O3 -DNDEBUG -fPIC -fopenmp -ffunction-sections -fdata-sections` (no `-std`, so gnu++17 and `-ffp-contract=fast`; no `-march`) |
| gtfn GPU | `nvcc -forward-unknown-to-host-compiler -DGT_PP_VARIADICS=1 -O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_90,code=[compute_90,sm_90] -Xcompiler=-fPIC --expt-relaxed-constexpr -ffunction-sections -fdata-sections` |
| Kokkos CPU | `g++ -O3 -DNDEBUG -std=gnu++20 -fPIC -ffp-contract=fast -fopenmp` |
| Kokkos GPU | `nvcc_wrapper -O3 -DNDEBUG -std=c++20 -fPIC -ffp-contract=fast -Xptxas -v -extended-lambda -Wext-lambda-captures-this -arch=sm_90 -fopenmp` |

Differences that remain between the two sides: C++17 (gtfn) vs C++20 (Kokkos), and the CUDA
code-generation switches listed above.

### gtfn baselines [M unless marked]

All baselines run through the real toolchain (`program.with_backend(...)`), with fields from the
backend's allocator (numpy on CPU, cupy on GPU).

| name | what it is |
|---|---|
| `gtfn` | stock `run_gtfn` / `run_gtfn_gpu`; **all ratios are against this** unless a column says otherwise |
| `gtfn_us` | unstructured only: `GTFNConfig(unstructured_horizontal_has_unit_stride=True)`; the fair comparison for Kokkos pointer variants, which also assume unit stride |
| `gtfn_fixed` | CPU only, kernels with temporaries: GridTools headers patched for the build (`bench/patch_gt.py`) so temporaries are allocated once and not zero-filled; a marker string confirms the patched allocator is in the three modules |

What the generated gtfn code does **[I, from reading the generated C++ and GridTools headers]**:

* `lap2` is one fused stencil; `hdiff` is 4 stencils with 3 temporaries; `tmpst` is 2 with 1.
* A vertical solve is one `vertical_executor` with two column stages, and GridTools launches
  **each stage separately** (`fn/run.hpp`, `run_column_stages`): forward sweep over all columns,
  then backward sweep. On the GPU the object file indeed has two device functions **[M]**. So the
  Kokkos `*.split` variants are the gtfn-shaped ones; the merged ones fuse more than gtfn does.
* CPU stencils: `omp parallel for collapse(3)`, K outermost, per-point pointer shifting. Temporaries
  are allocated on every call, twice, and zero-filled (`sid/allocator.hpp`).
* GPU: thread blocks 32x8(x1), temporaries from a caching allocator.
* Temporaries are compact: the compute domain plus the needed extent, no padding, I fastest
  (`sid/contiguous.hpp`).
* `v2e` unrolls `can_deref(shift(...)) ? acc + deref(shift(...)) : acc`; `chain` is one fused
  kernel that shifts again for every access.

### Kernels and sizes [M]

All fields are `float64`. Sizes are compute domains.

| name | content | `large` | `xlarge` (GPU only) | `small` |
|---|---|---|---|---|
| `lap`, `lap2`, `hdiff`, `tmpst` | Cartesian laplacian, lap(lap), flux-limited diffusion with 3 temporaries, two statements with 1 temporary | 512x512x80 (allocation 518x518x80, about 170 MB per field) | 1024x1024x80 | 32x32x80 |
| `tri`, `tritmp` | tridiagonal solve, forward sweep with tuple state and backward sweep; `tri` writes cp, dp to caller fields, `tritmp` keeps them as temporaries | same | same | same |
| `v2e` | vertex sum over 6 edges, 5 % of the table entries are skip values (-1) | 160k vertices, 480k edges, K=80 | 1.05M vertices, 3.1M edges | 1024 vertices, 3072 edges |
| `chain` | vertex -> edges -> vertices (E2V inside V2E) | same mesh | same mesh | same mesh |

Meshes are periodic triangulated grids, once with grid-order ("local") numbering and once with a
random permutation ("random"); the random tables are a worst case. The working set of every large
kernel exceeds the 114 MB L3 of a socket **[I, computed]**: one Cartesian field is about 170 MB;
the mesh edge field is 307 MB and the vertex field 102 MB (below L3 on its own), tables 4-5 MB, so
the smallest mesh working set (`chain`, two vertex fields) is 205 MB. `small` fits in cache.

### Kokkos variants

Name pattern `<accessor>.<style>.<shape>[.<temporaries>|.split]`.

* accessor: `ptr` pointer + `int` strides, static unit stride; `ptr64` the same with 64-bit policy
  index and offsets; `vl` unmanaged `LayoutLeft` View over the padded extents; `vs` unmanaged
  `LayoutStride` View; `naive` = `vs`, iterator, all-default `MDRangePolicy` (negative control).
  `ptr` with `int` offsets is only valid below 2^31 elements; `vl` is the variant that is safe
  without that restriction (`int` index, 64-bit offsets).
* style: `it` iterator-style functor (an object with `shift`/`deref`), `ex` explicit indexing.
  `v2e`: `it` shifts twice per neighbor with a ternary like the gtfn code, `it1` binds the shifted
  iterator once and branches; `chain`: `it` binds once, `itlit` re-shifts like the gtfn code; `ex2`
  explicit but re-reads the table at each use; `pw` pointer-walking scan column.
* GPU shapes: `mdtlb` `MDRangePolicy` tile {32,8,1} or {32,8}, `Iterate::Left`, `IndexType<int>`,
  `LaunchBounds<256>`; `mdt` without launch bounds; `mddef` all defaults; `mddefint` defaults plus
  `IndexType<int>`; `range` flat `RangePolicy`; `col*` one thread per column with the same choices.
* CPU shapes: `rows` `RangePolicy` over (k, j) rows, or (k, 1024-point blocks), with a plain inner
  unit-stride loop and the functor copied into locals; `rowsnh` without that copy; `rowssimd`
  with `omp simd`; `mdleft` / `mddef` `MDRangePolicy`; scans `col` (one column per iteration) and
  `chunk8/16/32/64` (a block of columns, k outer, `omp simd` inner loop; `chunk32` explicit only).
* temporaries (default: a pre-allocated pool, one full field-shaped padded buffer per temporary):
  `.alloc` `kokkos_malloc` on every call, `.view` zero-initialised View per call, `.viewnoinit`.
* temporary layouts (GPU `tritmp` only, two follow-up runs): `.lay-c0`, `.lay-c3`,
  `.lay-p0`, `.lay-p3`. `c` = compact `ni x nj x nk` (strides 1, `ni`, `ni*nj`, as gtfn
  allocates), `p` = the strides of the caller fields; `0` / `3` = elements between a 256-byte
  boundary and the first compute point. Applied to `ptr.it.colmdtlb` (merged and `.split`) and
  `ptr.it.colrange`; same kernel code, pooled buffers.

### Method [M]

* Same process, same input buffers for gtfn and Kokkos. Before timing, every variant's output
  (pre-filled with NaN; pool poisoned with NaN) is compared with gtfn: **all variants at all timed
  sizes agree bit for bit** (largest relative difference 0).
* 3 warm-up calls per variant; then rounds that run every variant once in random order (seed 42);
  a sample is the median of 1, 3, 10 or 30 consecutive calls depending on kernel duration; 21 rounds,
  or at least 15 once a cell has used its time budget.
* Both times are C++ `std::chrono` intervals that start after argument conversion in Python,
  nanobind or ctypes, so Python call overhead and compilation are excluded. gtfn: gt4py's own
  `compute` metric (generated binding); the interval holds building the GridTools data structures
  from the arrays, temporary allocation, the launch(es) and, on GPU, `cudaDeviceSynchronize`.
  Kokkos (`bench/kk/api.cpp`): accessor construction, temporary pool look-up, the launch(es) and
  `cudaDeviceSynchronize` (GPU) or `Kokkos::fence` (CPU). The device is synchronised before every
  call. Using `Kokkos::fence` on the GPU instead changes
  nothing measurable (`+kfence` rows).
* Ratio = median(variant) / median(gtfn) per allocation, 95 % bootstrap interval over whole rounds.
  Condensed tables show the median over allocations and the envelope of the intervals.
* CPU runs: `numactl --cpunodebind=0 --membind=0`, `OMP_PLACES=cores`, `OMP_PROC_BIND=close`,
  `OMP_WAIT_POLICY=active`, `OPENBLAS_NUM_THREADS=1`. GPU runs: `CUDA_VISIBLE_DEVICES=0`, one host
  thread, same binding. GPU strides are padded to 16 elements (Cartesian), CPU to 8 (the backend
  allocators' defaults, used for both sides).

## 2. Results

Cell format: `ratio [lo, hi]` or `time in ms / ratio [lo, hi]`. Ratio = variant / gtfn; the
interval is the envelope of the per-allocation 95 % intervals; `*` = one allocation only.
2.1 gives the candidates side by side, 2.2 the device-code and host-code facts, 2.3 every variant at
the large sizes, 2.4 the follow-up on the layout of the GPU `tritmp` temporaries. The small-size
tables of every variant, the direct variant-to-variant comparisons and the per-allocation tables
of the original report are not reproduced here.

### 2.1 Candidates against gtfn

**Size large, local numbering** (ratio to gtfn; below 1 means Kokkos is faster)

| kernel | Kokkos variant (GPU / CPU) | GPU large | GPU xlarge | CPU 1 | CPU 2 | CPU 4 | CPU 8 | CPU 18 | CPU 36 | CPU 72 |
|---|---|---|---|---|---|---|---|---|---|---|
| lap | ptr.it.mdtlb / ptr.it.rows | 0.95 [0.94, 0.96] | 0.93 [0.93, 0.94]* | 0.33 [0.33, 0.34]* | 0.34 [0.33, 0.34]* | 0.37 [0.36, 0.38]* | 0.39 [0.39, 0.40]* | 0.55 [0.55, 0.56]* | 0.82 [0.81, 0.83]* | 0.95 [0.90, 1.00] |
| lap | ptr.ex.mdtlb / ptr.ex.rows | 0.95 [0.94, 0.96] | 0.93 [0.93, 0.93]* | 0.33 [0.33, 0.33]* | 0.34 [0.33, 0.34]* | 0.37 [0.36, 0.37]* | 0.39 [0.38, 0.39]* | 0.55 [0.54, 0.56]* | 0.81 [0.81, 0.82]* | 0.93 [0.90, 0.98] |
| lap | vl.it.mdtlb / vl.it.rows | 0.95 [0.94, 0.96] | 0.93 [0.92, 0.93]* | 0.33 [0.33, 0.33]* | 0.33 [0.33, 0.34]* | 0.37 [0.37, 0.38]* | 0.39 [0.39, 0.40]* | 0.55 [0.55, 0.56]* | 0.82 [0.81, 0.82]* | 0.94 [0.90, 0.99] |
| lap2 | ptr.it.mdtlb / ptr.it.rows | 0.81 [0.78, 0.82] | 0.77 [0.76, 0.78]* | 0.51 [0.51, 0.51]* | 0.51 [0.51, 0.51]* | 0.51 [0.51, 0.51]* | 0.51 [0.51, 0.52]* | 0.51 [0.51, 0.52]* | 0.54 [0.53, 0.56]* | 0.86 [0.83, 0.93] |
| lap2 | ptr.ex.mdtlb / ptr.ex.rows | 0.80 [0.79, 0.82] | 0.76 [0.75, 0.78]* | 0.51 [0.51, 0.51]* | 0.51 [0.51, 0.51]* | 0.51 [0.51, 0.52]* | 0.51 [0.50, 0.51]* | 0.50 [0.50, 0.51]* | 0.53 [0.52, 0.55]* | 0.84 [0.83, 0.90] |
| lap2 | vl.it.mdtlb / vl.it.rows | 0.80 [0.78, 0.82] | 0.76 [0.75, 0.78]* | 0.51 [0.51, 0.51]* | 0.51 [0.51, 0.51]* | 0.51 [0.51, 0.52]* | 0.51 [0.51, 0.51]* | 0.51 [0.51, 0.52]* | 0.54 [0.53, 0.57]* | 0.84 [0.82, 0.88] |
| hdiff | ptr.it.mdtlb / ptr.it.rows | 0.99 [0.98, 1.00] | 0.99 [0.97, 1.00]* | 0.44 [0.43, 0.44]* | 0.37 [0.36, 0.37]* | 0.27 [0.27, 0.28]* | 0.19 [0.18, 0.19]* | 0.14 [0.14, 0.14]* | 0.11 [0.11, 0.12]* | 0.10 [0.10, 0.11] |
| hdiff | ptr.ex.mdtlb / ptr.ex.rows | 0.99 [0.98, 1.00] | 0.99 [0.97, 1.00]* | 0.44 [0.44, 0.44]* | 0.37 [0.37, 0.37]* | 0.28 [0.27, 0.28]* | 0.19 [0.19, 0.19]* | 0.14 [0.14, 0.14]* | 0.11 [0.11, 0.12]* | 0.10 [0.10, 0.11] |
| tmpst | ptr.it.mdtlb / ptr.it.rows | 0.98 [0.98, 0.98] | 0.98 [0.98, 0.99]* | 0.43 [0.42, 0.43]* | 0.36 [0.35, 0.36]* | 0.27 [0.27, 0.28]* | 0.19 [0.19, 0.19]* | 0.16 [0.15, 0.16]* | 0.13 [0.12, 0.13]* | 0.13 [0.12, 0.13] |
| tmpst | ptr.ex.mdtlb / ptr.ex.rows | 0.98 [0.97, 0.98] | 0.98 [0.98, 0.99]* | 0.43 [0.42, 0.43]* | 0.36 [0.35, 0.36]* | 0.27 [0.27, 0.28]* | 0.19 [0.18, 0.19]* | 0.15 [0.15, 0.16]* | 0.13 [0.12, 0.13]* | 0.13 [0.11, 0.13] |
| tri | ptr.it.colmdtlb.split / ptr.it.col.split | 1.00 [0.99, 1.00] | 1.01 [1.00, 1.01] | 0.99 [0.99, 1.00] | 0.99 [0.98, 1.00] | 0.99 [0.98, 1.01] | 0.99 [0.95, 1.01]* | 0.99 [0.95, 1.02] | 1.00 [0.98, 1.01]* | 1.00 [0.99, 1.01] |
| tri | ptr.it.colmdtlb / ptr.it.col | 1.02 [1.01, 1.02] | 1.02 [1.02, 1.02] | 0.99 [0.98, 1.00] | 0.99 [0.98, 1.01] | 1.02 [0.98, 1.06] | 0.99 [0.95, 1.00]* | 0.95 [0.90, 0.97] | 0.85 [0.84, 0.86]* | 0.85 [0.83, 0.88] |
| tri | ptr.pw.colmdtlb.split / ptr.ex.col | 1.02 [1.01, 1.02]* | 1.00 [1.00, 1.01]* | 0.99 [0.99, 1.00] | 0.99 [0.99, 1.01] | 1.02 [0.98, 1.05] | 0.99 [0.95, 1.02]* | 0.97 [0.90, 1.03] | 0.85 [0.83, 0.86]* | 0.85 [0.83, 0.87] |
| tri | ptr.it.colrange / ptr.it.chunk16 | 1.01 [1.01, 1.01] | 0.99 [0.99, 0.99] | 2.85 [2.81, 2.87] | 2.83 [2.79, 2.87] | 2.86 [2.78, 2.96] | 2.73 [2.62, 2.76]* | 2.55 [2.40, 2.60] | 1.93 [1.90, 1.95]* | 1.15 [1.11, 1.19] |
| tri | - / ptr.ex.chunk16 | - | - | 0.65 [0.64, 0.66] | 0.66 [0.65, 0.67] | 0.69 [0.67, 0.72] | 0.74 [0.71, 0.74]* | 0.89 [0.82, 0.92] | 0.97 [0.95, 0.98]* | 0.87 [0.85, 0.91] |
| tri | - / ptr.ex.chunk32 | - | - | 0.43 [0.43, 0.44]* | 0.44 [0.43, 0.44]* | 0.47 [0.46, 0.48]* | - | 0.62 [0.61, 0.63]* | - | 0.84 [0.84, 0.85]* |
| tritmp | ptr.it.colmdtlb.split / - | 1.03 [1.03, 1.03] | 1.05 [1.05, 1.06] | - | - | - | - | - | - | - |
| tritmp | ptr.it.colmdtlb / ptr.it.col | 1.05 [1.05, 1.05] | 1.07 [1.06, 1.07] | 0.52 [0.52, 0.53] | 0.48 [0.47, 0.48] | 0.41 [0.40, 0.42] | 0.33 [0.33, 0.34]* | 0.22 [0.20, 0.23] | 0.13 [0.13, 0.14]* | 0.13 [0.11, 0.14] |
| tritmp | ptr.pw.colmdtlb.split / ptr.ex.col | 1.05 [1.05, 1.05]* | 1.05 [1.05, 1.06]* | 0.53 [0.52, 0.53] | 0.48 [0.47, 0.48] | 0.41 [0.38, 0.43] | 0.33 [0.33, 0.33]* | 0.22 [0.20, 0.24] | 0.15 [0.13, 0.16]* | 0.13 [0.11, 0.14] |
| tritmp | ptr.it.colrange / ptr.it.chunk16 | 1.04 [1.04, 1.04] | 1.03 [1.02, 1.04] | 1.43 [1.41, 1.45] | 1.28 [1.25, 1.29] | 1.07 [1.01, 1.11] | 0.86 [0.85, 0.86]* | 0.53 [0.50, 0.55] | 0.29 [0.29, 0.29]* | 0.16 [0.15, 0.17] |
| tritmp | - / ptr.ex.chunk16 | - | - | 0.34 [0.33, 0.35] | 0.31 [0.31, 0.32] | 0.27 [0.25, 0.28] | 0.24 [0.24, 0.24]* | 0.19 [0.17, 0.20] | 0.14 [0.14, 0.14]* | 0.13 [0.12, 0.14] |
| tritmp | - / ptr.ex.chunk32 | - | - | 0.24 [0.24, 0.24]* | 0.21 [0.21, 0.22]* | 0.18 [0.18, 0.19]* | - | 0.13 [0.12, 0.13]* | - | 0.11 [0.11, 0.11]* |
| v2e | ptr.it.mdtlb / ptr.it.rows | 0.92 [0.91, 0.93] | 0.93 [0.92, 0.94] | 0.70 [0.70, 0.70]* | 0.70 [0.69, 0.70]* | 0.70 [0.70, 0.70]* | 0.70 [0.70, 0.71]* | 0.71 [0.70, 0.72]* | 0.75 [0.72, 0.80]* | 0.82 [0.80, 0.83] |
| v2e | ptr.it1.mdtlb / ptr.it1.rows | 0.59 [0.58, 0.60]* | 0.58 [0.57, 0.58]* | 2.38 [2.38, 2.39]* | 2.33 [2.32, 2.33]* | 2.32 [2.31, 2.32]* | 2.33 [2.32, 2.34]* | 2.30 [2.29, 2.32]* | 2.27 [2.22, 2.33]* | 2.03 [1.96, 2.08] |
| v2e | ptr.ex.mdtlb / ptr.ex.rows | 0.58 [0.57, 0.58] | 0.57 [0.56, 0.58] | 0.63 [0.63, 0.63]* | 0.64 [0.63, 0.64]* | 0.64 [0.64, 0.64]* | 0.64 [0.64, 0.64]* | 0.65 [0.64, 0.66]* | 0.72 [0.67, 0.74]* | 0.78 [0.76, 0.81] |
| v2e | vl.ex.mdtlb / vl.ex.rows | 0.58 [0.57, 0.58] | 0.57 [0.56, 0.57]* | 0.63 [0.63, 0.63]* | 0.64 [0.63, 0.64]* | 0.65 [0.64, 0.65]* | 0.64 [0.64, 0.65]* | 0.65 [0.64, 0.66]* | 0.68 [0.66, 0.74]* | 0.78 [0.76, 0.81] |
| chain | ptr.itlit.mdtlb / ptr.itlit.rows | 0.66 [0.66, 0.67]* | 0.77 [0.76, 0.78]* | 0.60 [0.60, 0.60]* | 0.60 [0.60, 0.60]* | 0.60 [0.60, 0.61]* | 0.60 [0.60, 0.61]* | 0.57 [0.57, 0.57]* | 0.62 [0.59, 0.62]* | 0.61 [0.59, 0.64] |
| chain | ptr.it.mdtlb / ptr.it.rows | 0.43 [0.42, 0.43] | 0.40 [0.40, 0.41] | 1.29 [1.28, 1.29]* | 1.30 [1.29, 1.30]* | 1.29 [1.29, 1.30]* | 1.29 [1.29, 1.30]* | 1.21 [1.21, 1.22]* | 1.30 [1.24, 1.34]* | 1.25 [1.20, 1.32] |
| chain | ptr.ex.mdtlb / ptr.ex.rows | 0.38 [0.38, 0.39] | 0.39 [0.38, 0.39] | 0.56 [0.56, 0.56]* | 0.56 [0.56, 0.56]* | 0.57 [0.57, 0.57]* | 0.57 [0.57, 0.57]* | 0.53 [0.53, 0.54]* | 0.60 [0.56, 0.60]* | 0.57 [0.55, 0.60] |
| chain | vl.ex.mdtlb / vl.ex.rows | 0.38 [0.38, 0.39] | 0.39 [0.38, 0.39]* | 0.56 [0.56, 0.56]* | 0.56 [0.56, 0.56]* | 0.57 [0.56, 0.57]* | 0.57 [0.57, 0.57]* | 0.54 [0.53, 0.54]* | 0.58 [0.55, 0.58]* | 0.56 [0.54, 0.59] |

**Size large, random numbering** (ratio to gtfn; below 1 means Kokkos is faster)

| kernel | Kokkos variant (GPU / CPU) | GPU large | GPU xlarge | CPU 1 | CPU 2 | CPU 4 | CPU 8 | CPU 18 | CPU 36 | CPU 72 |
|---|---|---|---|---|---|---|---|---|---|---|
| v2e | ptr.it.mdtlb / ptr.it.rows | 0.99 [0.99, 1.00] | 1.00 [1.00, 1.00] | 0.83 [0.83, 0.83]* | 0.83 [0.83, 0.83]* | 0.83 [0.82, 0.83]* | 0.81 [0.81, 0.82]* | 0.79 [0.79, 0.80]* | 0.72 [0.71, 0.72]* | 0.82 [0.81, 0.83] |
| v2e | ptr.it1.mdtlb / ptr.it1.rows | 0.96 [0.96, 0.96]* | 1.00 [1.00, 1.00]* | 2.75 [2.75, 2.75]* | 2.75 [2.75, 2.76]* | 2.73 [2.70, 2.74]* | 2.60 [2.59, 2.62]* | 2.54 [2.52, 2.55]* | 2.48 [2.45, 2.50]* | 1.98 [1.96, 2.00] |
| v2e | ptr.ex.mdtlb / ptr.ex.rows | 0.96 [0.96, 0.96] | 1.00 [1.00, 1.00] | 0.82 [0.81, 0.82]* | 0.81 [0.81, 0.81]* | 0.81 [0.80, 0.81]* | 0.79 [0.79, 0.81]* | 0.76 [0.76, 0.77]* | 0.67 [0.67, 0.68]* | 0.80 [0.78, 0.81] |
| v2e | vl.ex.mdtlb / vl.ex.rows | 0.96 [0.96, 0.96] | 1.00 [1.00, 1.00]* | 0.82 [0.82, 0.82]* | 0.81 [0.81, 0.81]* | 0.81 [0.80, 0.81]* | 0.79 [0.79, 0.80]* | 0.76 [0.76, 0.77]* | 0.68 [0.67, 0.68]* | 0.79 [0.79, 0.81] |
| chain | ptr.itlit.mdtlb / ptr.itlit.rows | 0.99 [0.99, 0.99]* | 1.00 [1.00, 1.00]* | 0.69 [0.69, 0.70]* | 0.69 [0.69, 0.69]* | 0.70 [0.69, 0.70]* | 0.73 [0.72, 0.73]* | 0.73 [0.73, 0.73]* | 0.71 [0.70, 0.71]* | 0.72 [0.71, 0.73] |
| chain | ptr.it.mdtlb / ptr.it.rows | 1.00 [1.00, 1.00] | 1.01 [1.01, 1.01] | 1.05 [1.05, 1.05]* | 1.05 [1.05, 1.06]* | 1.05 [1.05, 1.06]* | 1.06 [1.05, 1.06]* | 1.06 [1.05, 1.06]* | 1.06 [1.06, 1.07]* | 1.07 [1.06, 1.08] |
| chain | ptr.ex.mdtlb / ptr.ex.rows | 1.01 [1.01, 1.01] | 1.01 [1.01, 1.01] | 0.69 [0.69, 0.69]* | 0.69 [0.69, 0.69]* | 0.69 [0.69, 0.69]* | 0.72 [0.72, 0.72]* | 0.72 [0.72, 0.72]* | 0.70 [0.69, 0.70]* | 0.71 [0.70, 0.71] |
| chain | vl.ex.mdtlb / vl.ex.rows | 1.01 [1.00, 1.01] | 1.01 [1.01, 1.01]* | 0.69 [0.69, 0.69]* | 0.69 [0.69, 0.69]* | 0.69 [0.69, 0.69]* | 0.72 [0.71, 0.72]* | 0.72 [0.72, 0.72]* | 0.70 [0.69, 0.70]* | 0.71 [0.69, 0.71] |

**Size small, local numbering** (ratio to gtfn; below 1 means Kokkos is faster)

| kernel | Kokkos variant (GPU / CPU) | GPU small | CPU 1 | CPU 2 | CPU 4 | CPU 8 | CPU 18 | CPU 36 | CPU 72 |
|---|---|---|---|---|---|---|---|---|---|
| lap | ptr.it.mdtlb / ptr.it.rows | 0.68 [0.63, 0.74] | 0.37 [0.37, 0.37]* | 0.37 [0.37, 0.38]* | 0.36 [0.36, 0.36]* | 0.41 [0.41, 0.42]* | 0.57 [0.55, 0.57]* | 0.73 [0.72, 0.74]* | 0.90 [0.90, 0.91] |
| lap | ptr.ex.mdtlb / ptr.ex.rows | 0.68 [0.64, 0.73] | 0.37 [0.37, 0.37]* | 0.37 [0.37, 0.37]* | 0.36 [0.36, 0.36]* | 0.41 [0.41, 0.42]* | 0.55 [0.53, 0.56]* | 0.71 [0.70, 0.72]* | 0.89 [0.89, 0.90] |
| lap | vl.it.mdtlb / vl.it.rows | 0.69 [0.65, 0.74] | 0.32 [0.32, 0.32]* | 0.33 [0.33, 0.34]* | 0.31 [0.31, 0.31]* | 0.36 [0.36, 0.37]* | 0.53 [0.51, 0.54]* | 0.71 [0.70, 0.71]* | 0.89 [0.89, 0.90] |
| lap2 | ptr.it.mdtlb / ptr.it.rows | 0.69 [0.66, 0.73] | 0.56 [0.56, 0.56]* | 0.55 [0.55, 0.55]* | 0.55 [0.55, 0.55]* | 0.59 [0.58, 0.59]* | 0.66 [0.66, 0.66]* | 0.75 [0.74, 0.75]* | 0.88 [0.88, 0.89] |
| lap2 | ptr.ex.mdtlb / ptr.ex.rows | 0.69 [0.65, 0.74] | 0.56 [0.56, 0.56]* | 0.55 [0.55, 0.56]* | 0.56 [0.55, 0.56]* | 0.58 [0.58, 0.59]* | 0.66 [0.65, 0.66]* | 0.75 [0.75, 0.75]* | 0.88 [0.87, 0.89] |
| lap2 | vl.it.mdtlb / vl.it.rows | 0.69 [0.65, 0.74] | 0.56 [0.56, 0.56]* | 0.55 [0.55, 0.55]* | 0.56 [0.56, 0.56]* | 0.59 [0.58, 0.59]* | 0.66 [0.66, 0.67]* | 0.75 [0.75, 0.77]* | 0.89 [0.88, 0.90] |
| hdiff | ptr.it.mdtlb / ptr.it.rows | 0.77 [0.73, 0.81] | 0.43 [0.43, 0.43]* | 0.39 [0.38, 0.39]* | 0.32 [0.32, 0.32]* | 0.26 [0.26, 0.26]* | 0.24 [0.24, 0.24]* | 0.27 [0.27, 0.27]* | 0.37 [0.36, 0.38] |
| hdiff | ptr.ex.mdtlb / ptr.ex.rows | 0.77 [0.73, 0.81] | 0.43 [0.43, 0.43]* | 0.39 [0.38, 0.39]* | 0.32 [0.32, 0.32]* | 0.26 [0.26, 0.26]* | 0.24 [0.24, 0.24]* | 0.27 [0.27, 0.27]* | 0.38 [0.37, 0.38] |
| tmpst | ptr.it.mdtlb / ptr.it.rows | 0.72 [0.70, 0.75] | 0.37 [0.37, 0.38]* | 0.34 [0.34, 0.34]* | 0.27 [0.27, 0.27]* | 0.24 [0.24, 0.24]* | 0.25 [0.25, 0.26]* | 0.32 [0.32, 0.32]* | 0.43 [0.42, 0.44] |
| tmpst | ptr.ex.mdtlb / ptr.ex.rows | 0.72 [0.69, 0.75] | 0.37 [0.37, 0.38]* | 0.34 [0.33, 0.34]* | 0.27 [0.27, 0.27]* | 0.24 [0.24, 0.24]* | 0.25 [0.25, 0.26]* | 0.32 [0.32, 0.33]* | 0.44 [0.43, 0.44] |
| tri | ptr.it.colmdtlb.split / ptr.it.col.split | 0.96 [0.94, 0.97] | 0.99 [0.99, 0.99] | 0.98 [0.98, 0.99] | 0.97 [0.97, 0.98] | 0.97 [0.97, 0.97]* | 0.99 [0.98, 1.00] | 0.97 [0.96, 0.97]* | 0.97 [0.96, 0.97] |
| tri | ptr.it.colmdtlb / ptr.it.col | 0.92 [0.90, 0.92] | 1.04 [1.03, 1.05] | 1.04 [1.04, 1.04] | 1.03 [1.03, 1.03] | 1.01 [1.01, 1.01]* | 0.90 [0.89, 0.91] | 0.92 [0.90, 0.93]* | 0.67 [0.66, 0.68] |
| tri | ptr.pw.colmdtlb.split / ptr.ex.col | 0.96 [0.96, 0.96]* | 1.04 [1.03, 1.05] | 1.04 [1.04, 1.04] | 1.03 [1.03, 1.04] | 1.01 [1.00, 1.01]* | 0.90 [0.89, 0.91] | 0.92 [0.88, 0.92]* | 0.66 [0.64, 0.67] |
| tri | ptr.it.colrange / ptr.it.chunk16 | 0.89 [0.88, 0.90] | 3.19 [3.17, 3.20] | 3.20 [3.20, 3.21] | 3.18 [3.16, 3.19] | 3.09 [3.08, 3.09]* | 2.76 [2.73, 2.77] | 2.02 [2.02, 2.03]* | 1.16 [1.15, 1.18] |
| tri | - / ptr.ex.chunk16 | - | 0.33 [0.32, 0.33] | 0.35 [0.34, 0.35] | 0.36 [0.36, 0.36] | 0.41 [0.41, 0.41]* | 0.52 [0.52, 0.53] | 0.44 [0.43, 0.44]* | 0.44 [0.44, 0.45] |
| tri | - / ptr.ex.chunk32 | - | 0.43 [0.43, 0.43]* | 0.47 [0.46, 0.47]* | 0.49 [0.49, 0.49]* | - | 0.62 [0.61, 0.62]* | - | 0.52 [0.52, 0.52]* |
| tritmp | ptr.it.colmdtlb.split / - | 0.97 [0.96, 0.98] | - | - | - | - | - | - | - |
| tritmp | ptr.it.colmdtlb / ptr.it.col | 0.91 [0.91, 0.93] | 0.81 [0.76, 0.86] | 0.77 [0.74, 0.81] | 0.67 [0.67, 0.68] | 0.55 [0.55, 0.56]* | 0.43 [0.43, 0.44] | 0.40 [0.39, 0.40]* | 0.28 [0.28, 0.29] |
| tritmp | ptr.pw.colmdtlb.split / ptr.ex.col | 0.97 [0.96, 0.97]* | 0.81 [0.76, 0.86] | 0.77 [0.74, 0.81] | 0.67 [0.67, 0.68] | 0.55 [0.55, 0.55]* | 0.43 [0.42, 0.43] | 0.39 [0.38, 0.39]* | 0.28 [0.27, 0.28] |
| tritmp | ptr.it.colrange / ptr.it.chunk16 | 0.89 [0.88, 0.90] | 2.47 [2.34, 2.62] | 2.38 [2.28, 2.48] | 2.07 [2.06, 2.09] | 1.68 [1.68, 1.69]* | 1.30 [1.29, 1.30] | 0.80 [0.80, 0.80]* | 0.48 [0.48, 0.49] |
| tritmp | - / ptr.ex.chunk16 | - | 0.25 [0.24, 0.27] | 0.26 [0.25, 0.27] | 0.23 [0.23, 0.24] | 0.23 [0.23, 0.23]* | 0.25 [0.24, 0.25] | 0.18 [0.18, 0.19]* | 0.18 [0.18, 0.19] |
| tritmp | - / ptr.ex.chunk32 | - | 0.29 [0.29, 0.34]* | 0.32 [0.32, 0.33]* | 0.32 [0.31, 0.32]* | - | 0.29 [0.29, 0.29]* | - | 0.22 [0.22, 0.22]* |
| v2e | ptr.it.mdtlb / ptr.it.rows | 0.72 [0.69, 0.75] | 0.63 [0.61, 0.64]* | 0.62 [0.62, 0.63]* | 0.59 [0.59, 0.59]* | 0.57 [0.57, 0.57]* | 0.64 [0.62, 0.65]* | 0.73 [0.72, 0.73]* | 0.87 [0.86, 0.89] |
| v2e | ptr.it1.mdtlb / ptr.it1.rows | 0.65 [0.64, 0.66]* | 3.86 [3.85, 3.88]* | 3.62 [3.61, 3.63]* | 3.61 [3.60, 3.62]* | 3.41 [3.40, 3.44]* | 3.44 [3.37, 3.48]* | 3.29 [3.26, 3.31]* | 2.23 [2.16, 2.30] |
| v2e | ptr.ex.mdtlb / ptr.ex.rows | 0.62 [0.59, 0.65] | 0.60 [0.59, 0.61]* | 0.58 [0.58, 0.59]* | 0.56 [0.56, 0.57]* | 0.54 [0.54, 0.55]* | 0.61 [0.59, 0.61]* | 0.70 [0.70, 0.71]* | 0.86 [0.84, 0.87] |
| v2e | vl.ex.mdtlb / vl.ex.rows | 0.64 [0.61, 0.66] | 0.59 [0.59, 0.60]* | 0.59 [0.58, 0.59]* | 0.56 [0.56, 0.57]* | 0.54 [0.54, 0.55]* | 0.61 [0.60, 0.62]* | 0.72 [0.71, 0.72]* | 0.86 [0.84, 0.88] |
| chain | ptr.itlit.mdtlb / ptr.itlit.rows | 0.71 [0.71, 0.72]* | 0.54 [0.54, 0.54]* | 0.54 [0.54, 0.54]* | 0.54 [0.54, 0.54]* | 0.55 [0.55, 0.55]* | 0.58 [0.58, 0.58]* | 0.72 [0.71, 0.72]* | 0.74 [0.72, 0.77] |
| chain | ptr.it.mdtlb / ptr.it.rows | 0.59 [0.57, 0.62] | 1.50 [1.50, 1.50]* | 1.56 [1.56, 1.56]* | 1.55 [1.54, 1.55]* | 1.54 [1.53, 1.54]* | 1.56 [1.55, 1.56]* | 1.76 [1.76, 1.76]* | 1.61 [1.56, 1.66] |
| chain | ptr.ex.mdtlb / ptr.ex.rows | 0.58 [0.56, 0.60] | 0.52 [0.52, 0.52]* | 0.52 [0.52, 0.52]* | 0.52 [0.52, 0.52]* | 0.53 [0.53, 0.53]* | 0.56 [0.56, 0.56]* | 0.69 [0.69, 0.69]* | 0.73 [0.71, 0.75] |
| chain | vl.ex.mdtlb / vl.ex.rows | 0.59 [0.56, 0.62] | 0.52 [0.52, 0.52]* | 0.52 [0.52, 0.52]* | 0.52 [0.52, 0.53]* | 0.53 [0.53, 0.53]* | 0.56 [0.56, 0.56]* | 0.69 [0.69, 0.70]* | 0.74 [0.72, 0.76] |

**Size small, random numbering** (ratio to gtfn; below 1 means Kokkos is faster)

| kernel | Kokkos variant (GPU / CPU) | GPU small | CPU 1 | CPU 2 | CPU 4 | CPU 8 | CPU 18 | CPU 36 | CPU 72 |
|---|---|---|---|---|---|---|---|---|---|
| v2e | ptr.it.mdtlb / ptr.it.rows | 0.76 [0.73, 0.81] | 0.59 [0.59, 0.59]* | 0.59 [0.59, 0.59]* | 0.54 [0.54, 0.54]* | 0.55 [0.55, 0.55]* | 0.61 [0.61, 0.62]* | 0.72 [0.72, 0.73]* | 0.87 [0.85, 0.88] |
| v2e | ptr.it1.mdtlb / ptr.it1.rows | 0.78 [0.76, 0.79]* | 3.48 [3.47, 3.49]* | 3.27 [3.27, 3.29]* | 3.18 [3.17, 3.18]* | 3.15 [3.12, 3.16]* | 3.14 [3.11, 3.17]* | 3.22 [3.20, 3.24]* | 2.23 [2.15, 2.30] |
| v2e | ptr.ex.mdtlb / ptr.ex.rows | 0.74 [0.71, 0.79] | 0.56 [0.56, 0.56]* | 0.55 [0.54, 0.55]* | 0.52 [0.51, 0.52]* | 0.52 [0.52, 0.53]* | 0.59 [0.58, 0.59]* | 0.70 [0.69, 0.70]* | 0.85 [0.83, 0.86] |
| v2e | vl.ex.mdtlb / vl.ex.rows | 0.76 [0.72, 0.80] | 0.56 [0.56, 0.56]* | 0.54 [0.54, 0.55]* | 0.52 [0.52, 0.52]* | 0.53 [0.52, 0.53]* | 0.59 [0.59, 0.60]* | 0.71 [0.70, 0.71]* | 0.86 [0.85, 0.88] |
| chain | ptr.itlit.mdtlb / ptr.itlit.rows | 0.78 [0.78, 0.79]* | 0.54 [0.54, 0.54]* | 0.53 [0.53, 0.53]* | 0.53 [0.53, 0.53]* | 0.53 [0.53, 0.53]* | 0.56 [0.56, 0.56]* | 0.71 [0.71, 0.71]* | 0.74 [0.72, 0.78] |
| chain | ptr.it.mdtlb / ptr.it.rows | 0.70 [0.67, 0.74] | 1.46 [1.45, 1.46]* | 1.49 [1.49, 1.50]* | 1.50 [1.50, 1.50]* | 1.46 [1.46, 1.46]* | 1.48 [1.48, 1.49]* | 1.75 [1.75, 1.76]* | 1.60 [1.54, 1.66] |
| chain | ptr.ex.mdtlb / ptr.ex.rows | 0.70 [0.67, 0.73] | 0.52 [0.52, 0.52]* | 0.51 [0.51, 0.51]* | 0.51 [0.51, 0.51]* | 0.51 [0.51, 0.52]* | 0.54 [0.54, 0.55]* | 0.69 [0.69, 0.69]* | 0.73 [0.70, 0.76] |
| chain | vl.ex.mdtlb / vl.ex.rows | 0.71 [0.68, 0.74] | 0.52 [0.52, 0.52]* | 0.51 [0.51, 0.51]* | 0.51 [0.51, 0.51]* | 0.52 [0.51, 0.52]* | 0.54 [0.54, 0.55]* | 0.69 [0.69, 0.70]* | 0.74 [0.72, 0.77] |

### 2.2 Device and host code

#### GPU registers, static instruction counts, compile time [M]

| unit | compile s | kernel | registers/thread | SASS instructions | global loads | global stores |
|---|---|---|---|---|---|---|
| gtfn/chain | 4.2 |  | 26 | 355 | 38 | 1 |
| gtfn/hdiff | 5.5 |  | 26 | 111 | 6 | 1 |
|  |  |  | 22 | 77 | 4 | 1 |
|  |  |  | 20 | 75 | 4 | 1 |
|  |  |  | 23 | 75 | 5 | 1 |
| gtfn/lap | 3.5 |  | 25 | 82 | 5 | 1 |
| gtfn/lap2 | 3.5 |  | 32 | 151 | 15 | 1 |
| gtfn/tmpst | 4.6 |  | 20 | 71 | 4 | 1 |
|  |  |  | 16 | 75 | 2 | 1 |
| gtfn/tri | 5.6 | backward | 32 | 502 | 58 | 29 |
|  |  | forward | 48 | 644 | 20 | 10 |
| gtfn/tritmp | 6.2 | backward | 31 | 498 | 58 | 29 |
|  |  | forward | 48 | 597 | 20 | 10 |
| gtfn/v2e | 3.8 |  | 24 | 275 | 13 | 1 |
| kokkos/chain/ex | 7.5 | ChainB | 32 | 161 | 30 | 1 |
| kokkos/chain/ex2 | 7.6 | ChainB | 32 | 161 | 30 | 1 |
| kokkos/chain/it | 7.4 | ChainB | 32 | 177 | 30 | 1 |
| kokkos/chain/itlit | 7.4 | ChainB | 18 | 281 | 38 | 1 |
| kokkos/hdiff/ex | 8.6 | FlxB | 18 | 56 | 4 | 1 |
|  |  | FlyB | 20 | 63 | 4 | 1 |
|  |  | HoutB | 26 | 70 | 6 | 1 |
|  |  | LapB | 18 | 58 | 5 | 1 |
| kokkos/hdiff/it | 8.6 | FlxB | 18 | 56 | 4 | 1 |
|  |  | FlyB | 20 | 63 | 4 | 1 |
|  |  | HoutB | 26 | 70 | 6 | 1 |
|  |  | LapB | 18 | 59 | 5 | 1 |
| kokkos/headers_only/- | 6.7 |  | - | - | - | - |
| kokkos/lap/ex | 7.4 | LapB | 18 | 58 | 5 | 1 |
| kokkos/lap/it | 7.5 | LapB | 18 | 59 | 5 | 1 |
| kokkos/lap2/ex | 7.4 | Lap2B | 32 | 94 | 13 | 1 |
| kokkos/lap2/it | 7.5 | Lap2B | 32 | 97 | 13 | 1 |
| kokkos/tmpst/ex | 7.7 | S1B | 14 | 49 | 2 | 1 |
|  |  | S2B | 20 | 57 | 4 | 1 |
| kokkos/tmpst/it | 7.8 | S1B | 14 | 49 | 2 | 1 |
|  |  | S2B | 20 | 58 | 4 | 1 |
| kokkos/tri/ex | 7.5 | both sweeps | 52 | 808 | 78 | 39 |
| kokkos/tri/it | 7.6 | both sweeps | 52 | 808 | 78 | 39 |
| kokkos/tri/pw | 7.5 | both sweeps | 44 | 481 | 62 | 31 |
| kokkos/tri_split/it | 7.8 | backward | 32 | 302 | 58 | 29 |
|  |  | forward | 50 | 550 | 20 | 10 |
| kokkos/tri_split/pw | 7.7 | backward | 32 | 286 | 58 | 29 |
|  |  | forward | 40 | 218 | 4 | 2 |
| kokkos/tritmp/it | 7.6 | both sweeps | 52 | 808 | 78 | 39 |
| kokkos/tritmp_split/it | 8.0 | backward | 32 | 302 | 58 | 29 |
|  |  | forward | 50 | 550 | 20 | 10 |
| kokkos/tritmp_split/pw | 8.0 | backward | 32 | 286 | 58 | 29 |
|  |  | forward | 40 | 218 | 4 | 2 |
| kokkos/v2e/ex | 7.4 | V2EB | 28 | 89 | 12 | 1 |
| kokkos/v2e/ex2 | 7.5 | V2EB | 28 | 89 | 12 | 1 |
| kokkos/v2e/it | 7.4 | V2EB | 14 | 126 | 12 | 1 |
| kokkos/v2e/it1 | 7.4 | V2EB | 30 | 108 | 12 | 1 |

#### Host code generation [M]

Counts in the kernel loop function of a single-variant translation unit; time ratio from the large
local case at 8 threads. "Stack" = loads / stores through the stack pointer or an alias of it.
Counted on the assembly of single-variant translation units (see `bench/asm.py`).

| variant | ratio to gtfn, 8 thr | instructions | stack | vector `fdiv` |
|---|---|---|---|---|
| v2e.ptr.ex.rows | 0.64 | 73 | 0 / 0 | - |
| v2e.ptr.it.rows (literal) | 0.70 | 106 | 3 / 3 | - |
| v2e.ptr.it1.rows | 2.33 | 146 | 39 / 35 | - |
| chain.ptr.ex.rows | 0.57 | 117 | 0 / 0 | - |
| chain.vl.ex.rows | 0.57 | 116 | 0 / 0 | - |
| chain.ptr.itlit.rows | 0.60 | 150 | 3 / 3 | - |
| chain.ptr.it.rows | 1.29 | 200 | 42 / 37 | - |
| chain.vl.it.rows | 1.83 | 194 | 29 / 45 | - |
| chain.ptr64.it.rows | 3.69 | 188 | 37 / 36 | - |
| tri.ptr.ex.chunk16 | 0.74 | 707 | 127 / 86 | 30 |
| tri.vl.ex.chunk16 | 0.73 | 685 | 112 / 96 | 30 |
| tri.ptr64.it.chunk16 | 0.95 | 344 | 45 / 74 | 0 |
| tri.vl.it.chunk16 | 1.74 | 479 | 115 / 107 | 0 |
| tri.ptr.it.chunk16 | 2.73 | 321 | 53 / 54 | 0 |

In vectorised code stack traffic is spill code around a wide loop body, so the counts are only
comparable between variants of the same shape.

### 2.3 All variants, large sizes

#### `lap`: GPU (ms / ratio to gtfn)

| variant | large | xlarge |
|---|---|---|
| gtfn | 0.125 / 1.00 [1.00, 1.00] | 0.491 / 1.00 [1.00, 1.00]* |
| naive | 0.127 / 1.01 [1.00, 1.02] | 0.487 / 0.99 [0.99, 1.00]* |
| ptr.ex.mddef | 0.125 / 1.00 [0.99, 1.01]* | 0.477 / 0.97 [0.97, 0.98]* |
| ptr.ex.mdt | 0.119 / 0.95 [0.94, 0.96] | 0.455 / 0.93 [0.92, 0.93]* |
| ptr.ex.mdtlb | 0.119 / 0.95 [0.94, 0.96] | 0.456 / 0.93 [0.93, 0.93]* |
| ptr.it.mddef | 0.125 / 1.00 [0.99, 1.01] | 0.482 / 0.98 [0.98, 0.99]* |
| ptr.it.mddefint | 0.126 / 1.00 [0.99, 1.01] | 0.48 / 0.98 [0.97, 0.98]* |
| ptr.it.mdt | 0.12 / 0.96 [0.94, 0.96] | 0.457 / 0.93 [0.93, 0.94]* |
| ptr.it.mdtlb | 0.119 / 0.95 [0.94, 0.96] | 0.456 / 0.93 [0.93, 0.94]* |
| ptr.it.mdtlb+kfence | 0.12 / 0.96 [0.94, 0.96] | 0.455 / 0.93 [0.93, 0.93]* |
| ptr.it.range | 0.14 / 1.11 [1.09, 1.14] | 0.587 / 1.20 [1.18, 1.23]* |
| ptr64.it.mdtlb | 0.121 / 0.97 [0.95, 0.98] | 0.469 / 0.96 [0.95, 0.96]* |
| vl.ex.mdtlb | 0.119 / 0.95 [0.94, 0.96] | 0.454 / 0.93 [0.92, 0.93]* |
| vl.it.mdtlb | 0.119 / 0.95 [0.94, 0.96] | 0.456 / 0.93 [0.92, 0.93]* |
| vs.it.mdtlb | 0.121 / 0.97 [0.95, 0.97] | 0.463 / 0.94 [0.94, 0.96]* |

#### `lap`: CPU, large (ms / ratio to gtfn)

| variant | 1 thr | 2 thr | 4 thr | 8 thr | 18 thr | 36 thr | 72 thr |
|---|---|---|---|---|---|---|---|
| gtfn | 33.3 / 1.00 [1.00, 1.00]* | 17.1 / 1.00 [1.00, 1.00]* | 8.43 / 1.00 [1.00, 1.00]* | 4.36 / 1.00 [1.00, 1.00]* | 1.96 / 1.00 [1.00, 1.00]* | 1.06 / 1.00 [1.00, 1.00]* | 0.808 / 1.00 [1.00, 1.00] |
| naive | - | - | 29.7 / 3.53 [3.50, 3.55]* | 15.6 / 3.56 [3.54, 3.65]* | 9.19 / 4.70 [4.64, 4.76]* | 4.99 / 4.72 [4.68, 4.74]* | 3.11 / 3.85 [3.74, 3.94] |
| ptr.ex.mdleft | 15.9 / 0.48 [0.47, 0.48]* | 11.2 / 0.65 [0.64, 0.67]* | 6.68 / 0.79 [0.75, 0.82]* | 3.37 / 0.77 [0.74, 0.80]* | 1.54 / 0.79 [0.77, 0.83]* | 1.1 / 1.04 [1.02, 1.08]* | 0.823 / 1.02 [0.95, 1.08] |
| ptr.ex.rows | 11 / 0.33 [0.33, 0.33]* | 5.75 / 0.34 [0.33, 0.34]* | 3.08 / 0.37 [0.36, 0.37]* | 1.69 / 0.39 [0.38, 0.39]* | 1.08 / 0.55 [0.54, 0.56]* | 0.858 / 0.81 [0.81, 0.82]* | 0.755 / 0.93 [0.90, 0.98] |
| ptr.it.mddef | - | - | 29.5 / 3.51 [3.48, 3.54]* | 15.5 / 3.54 [3.53, 3.62]* | 9.29 / 4.75 [4.66, 4.82]* | 5.02 / 4.75 [4.71, 4.79]* | 3.12 / 3.86 [3.75, 3.96] |
| ptr.it.mdleft | 15.9 / 0.48 [0.47, 0.48]* | 11.4 / 0.67 [0.66, 0.67]* | 6.65 / 0.79 [0.76, 0.80]* | 3.31 / 0.76 [0.73, 0.78]* | 1.56 / 0.80 [0.77, 0.86]* | 1.15 / 1.09 [1.03, 1.09]* | 0.827 / 1.02 [0.96, 1.08] |
| ptr.it.rows | 11.1 / 0.33 [0.33, 0.34]* | 5.78 / 0.34 [0.33, 0.34]* | 3.13 / 0.37 [0.36, 0.38]* | 1.71 / 0.39 [0.39, 0.40]* | 1.08 / 0.55 [0.55, 0.56]* | 0.868 / 0.82 [0.81, 0.83]* | 0.77 / 0.95 [0.90, 1.00] |
| ptr.it.rowsnh | 10.9 / 0.33 [0.33, 0.33]* | 5.66 / 0.33 [0.33, 0.34]* | 3.06 / 0.36 [0.36, 0.37]* | 1.68 / 0.38 [0.38, 0.39]* | 1.08 / 0.55 [0.55, 0.56]* | 0.864 / 0.82 [0.81, 0.83]* | 0.761 / 0.94 [0.90, 0.98] |
| ptr.it.rowssimd | 11 / 0.33 [0.33, 0.33]* | 5.74 / 0.33 [0.33, 0.34]* | 3.1 / 0.37 [0.36, 0.37]* | 1.69 / 0.39 [0.38, 0.40]* | 1.08 / 0.55 [0.55, 0.56]* | 0.865 / 0.82 [0.81, 0.83]* | 0.766 / 0.95 [0.90, 1.00] |
| ptr64.it.mdleft | 10.7 / 0.32 [0.32, 0.33]* | 9.1 / 0.53 [0.52, 0.54]* | 5.25 / 0.62 [0.62, 0.64]* | 2.63 / 0.60 [0.59, 0.64]* | 1.39 / 0.71 [0.69, 0.75]* | 1.05 / 0.99 [0.96, 1.04]* | 0.837 / 1.04 [0.98, 1.09] |
| ptr64.it.rows | 11 / 0.33 [0.33, 0.33]* | 5.67 / 0.33 [0.33, 0.34]* | 3.05 / 0.36 [0.36, 0.37]* | 1.67 / 0.38 [0.38, 0.40]* | 1.08 / 0.55 [0.54, 0.55]* | 0.866 / 0.82 [0.81, 0.83]* | 0.76 / 0.94 [0.90, 0.98] |
| vl.ex.rows | 11 / 0.33 [0.33, 0.33]* | 5.74 / 0.33 [0.33, 0.34]* | 3.08 / 0.37 [0.36, 0.37]* | 1.69 / 0.39 [0.38, 0.40]* | 1.08 / 0.55 [0.54, 0.56]* | 0.864 / 0.82 [0.81, 0.82]* | 0.757 / 0.94 [0.90, 0.99] |
| vl.it.mdleft | 15.1 / 0.45 [0.45, 0.46]* | 11.1 / 0.65 [0.64, 0.66]* | 6.49 / 0.77 [0.74, 0.81]* | 3.16 / 0.72 [0.71, 0.75]* | 1.55 / 0.79 [0.77, 0.83]* | 1.1 / 1.04 [1.03, 1.05]* | 0.821 / 1.02 [0.96, 1.08] |
| vl.it.rows | 11 / 0.33 [0.33, 0.33]* | 5.69 / 0.33 [0.33, 0.34]* | 3.14 / 0.37 [0.37, 0.38]* | 1.7 / 0.39 [0.39, 0.40]* | 1.08 / 0.55 [0.55, 0.56]* | 0.864 / 0.82 [0.81, 0.82]* | 0.758 / 0.94 [0.90, 0.99] |
| vs.it.rows | 11.1 / 0.33 [0.33, 0.34]* | 5.69 / 0.33 [0.33, 0.34]* | 3.11 / 0.37 [0.36, 0.38]* | 1.7 / 0.39 [0.38, 0.40]* | 1.09 / 0.55 [0.55, 0.56]* | 0.866 / 0.82 [0.81, 0.82]* | 0.769 / 0.95 [0.90, 1.00] |

#### `lap2`: GPU (ms / ratio to gtfn)

| variant | large | xlarge |
|---|---|---|
| gtfn | 0.192 / 1.00 [1.00, 1.00] | 0.75 / 1.00 [1.00, 1.00]* |
| naive | 0.174 / 0.91 [0.88, 0.92] | 0.656 / 0.88 [0.85, 0.90]* |
| ptr.ex.mddef | 0.163 / 0.85 [0.84, 0.86]* | 0.605 / 0.81 [0.80, 0.82]* |
| ptr.ex.mdt | 0.154 / 0.80 [0.79, 0.82] | 0.584 / 0.78 [0.76, 0.80]* |
| ptr.ex.mdtlb | 0.154 / 0.80 [0.79, 0.82] | 0.571 / 0.76 [0.75, 0.78]* |
| ptr.it.mddef | 0.164 / 0.85 [0.84, 0.87] | 0.611 / 0.82 [0.81, 0.83]* |
| ptr.it.mddefint | 0.163 / 0.85 [0.84, 0.87] | 0.61 / 0.81 [0.81, 0.83]* |
| ptr.it.mdt | 0.155 / 0.80 [0.79, 0.83] | 0.577 / 0.77 [0.76, 0.79]* |
| ptr.it.mdtlb | 0.155 / 0.81 [0.78, 0.82] | 0.574 / 0.77 [0.76, 0.78]* |
| ptr.it.mdtlb+kfence | 0.155 / 0.80 [0.79, 0.82] | 0.573 / 0.76 [0.76, 0.77]* |
| ptr.it.range | 0.215 / 1.12 [1.10, 1.15] | 0.815 / 1.09 [1.08, 1.10]* |
| ptr64.it.mdtlb | 0.174 / 0.90 [0.89, 0.94] | 0.66 / 0.88 [0.87, 0.90]* |
| vl.ex.mdtlb | 0.153 / 0.80 [0.78, 0.82] | 0.578 / 0.77 [0.75, 0.80]* |
| vl.it.mdtlb | 0.153 / 0.80 [0.78, 0.82] | 0.569 / 0.76 [0.75, 0.78]* |
| vs.it.mdtlb | 0.171 / 0.89 [0.87, 0.91] | 0.633 / 0.84 [0.84, 0.86]* |

#### `lap2`: CPU, large (ms / ratio to gtfn)

| variant | 1 thr | 2 thr | 4 thr | 8 thr | 18 thr | 36 thr | 72 thr |
|---|---|---|---|---|---|---|---|
| gtfn | 59.3 / 1.00 [1.00, 1.00]* | 29.8 / 1.00 [1.00, 1.00]* | 15 / 1.00 [1.00, 1.00]* | 7.54 / 1.00 [1.00, 1.00]* | 3.38 / 1.00 [1.00, 1.00]* | 1.77 / 1.00 [1.00, 1.00]* | 0.931 / 1.00 [1.00, 1.00] |
| naive | - | - | 52.5 / 3.51 [3.50, 3.53]* | 27.2 / 3.60 [3.59, 3.62]* | 12.8 / 3.78 [3.72, 3.89]* | 8.29 / 4.68 [4.43, 4.84]* | 5.25 / 5.64 [5.37, 5.94] |
| ptr.ex.mdleft | 54.4 / 0.92 [0.92, 0.92]* | 28.2 / 0.94 [0.94, 0.95]* | 14.6 / 0.98 [0.97, 0.99]* | 7.38 / 0.98 [0.97, 0.98]* | 3.27 / 0.97 [0.96, 0.99]* | 1.75 / 0.99 [0.97, 1.06]* | 0.96 / 1.03 [1.01, 1.06] |
| ptr.ex.rows | 30.1 / 0.51 [0.51, 0.51]* | 15.2 / 0.51 [0.51, 0.51]* | 7.61 / 0.51 [0.51, 0.52]* | 3.83 / 0.51 [0.50, 0.51]* | 1.7 / 0.50 [0.50, 0.51]* | 0.937 / 0.53 [0.52, 0.55]* | 0.783 / 0.84 [0.83, 0.90] |
| ptr.it.mddef | - | - | 52.5 / 3.51 [3.48, 3.52]* | 27.2 / 3.61 [3.59, 3.62]* | 12.4 / 3.66 [3.64, 3.75]* | 8.32 / 4.70 [4.46, 4.84]* | 5.04 / 5.42 [5.34, 5.50] |
| ptr.it.mdleft | 54.4 / 0.92 [0.92, 0.92]* | 28.4 / 0.95 [0.94, 0.96]* | 14.7 / 0.98 [0.98, 0.99]* | 7.4 / 0.98 [0.98, 0.99]* | 3.31 / 0.98 [0.97, 0.99]* | 1.79 / 1.01 [0.97, 1.04]* | 0.976 / 1.05 [1.03, 1.10] |
| ptr.it.rows | 30.2 / 0.51 [0.51, 0.51]* | 15.3 / 0.51 [0.51, 0.51]* | 7.66 / 0.51 [0.51, 0.51]* | 3.87 / 0.51 [0.51, 0.52]* | 1.73 / 0.51 [0.51, 0.52]* | 0.96 / 0.54 [0.53, 0.56]* | 0.804 / 0.86 [0.83, 0.93] |
| ptr.it.rowsnh | 30.2 / 0.51 [0.51, 0.51]* | 15.3 / 0.51 [0.51, 0.51]* | 7.61 / 0.51 [0.51, 0.51]* | 3.83 / 0.51 [0.50, 0.51]* | 1.72 / 0.51 [0.51, 0.52]* | 0.961 / 0.54 [0.53, 0.57]* | 0.787 / 0.85 [0.83, 0.88] |
| ptr.it.rowssimd | 30.2 / 0.51 [0.51, 0.51]* | 15.3 / 0.51 [0.51, 0.51]* | 7.63 / 0.51 [0.51, 0.52]* | 3.82 / 0.51 [0.50, 0.51]* | 1.75 / 0.52 [0.51, 0.52]* | 0.956 / 0.54 [0.53, 0.56]* | 0.807 / 0.87 [0.84, 0.92] |
| ptr64.it.mdleft | 31.3 / 0.53 [0.53, 0.53]* | 17.2 / 0.58 [0.57, 0.58]* | 9.71 / 0.65 [0.64, 0.66]* | 5 / 0.66 [0.66, 0.67]* | 2.28 / 0.68 [0.67, 0.69]* | 1.32 / 0.75 [0.72, 0.78]* | 0.841 / 0.90 [0.87, 0.96] |
| ptr64.it.rows | 30.2 / 0.51 [0.51, 0.51]* | 15.2 / 0.51 [0.51, 0.51]* | 7.61 / 0.51 [0.51, 0.51]* | 3.82 / 0.51 [0.50, 0.51]* | 1.7 / 0.50 [0.50, 0.51]* | 0.958 / 0.54 [0.53, 0.58]* | 0.788 / 0.85 [0.83, 0.89] |
| vl.ex.rows | 30.1 / 0.51 [0.51, 0.51]* | 15.2 / 0.51 [0.51, 0.51]* | 7.61 / 0.51 [0.50, 0.51]* | 3.8 / 0.50 [0.50, 0.51]* | 1.71 / 0.51 [0.50, 0.52]* | 0.977 / 0.55 [0.53, 0.58]* | 0.788 / 0.85 [0.82, 0.93] |
| vl.it.mdleft | 54.7 / 0.92 [0.92, 0.92]* | 28.3 / 0.95 [0.94, 0.95]* | 14.7 / 0.98 [0.97, 0.98]* | 7.33 / 0.97 [0.97, 0.98]* | 3.29 / 0.97 [0.97, 0.99]* | 1.75 / 0.99 [0.98, 1.03]* | 0.959 / 1.03 [1.01, 1.05] |
| vl.it.rows | 30 / 0.51 [0.51, 0.51]* | 15.1 / 0.51 [0.51, 0.51]* | 7.68 / 0.51 [0.51, 0.52]* | 3.84 / 0.51 [0.51, 0.51]* | 1.73 / 0.51 [0.51, 0.52]* | 0.955 / 0.54 [0.53, 0.57]* | 0.784 / 0.84 [0.82, 0.88] |
| vs.it.rows | 30 / 0.51 [0.51, 0.51]* | 15.2 / 0.51 [0.51, 0.51]* | 7.6 / 0.51 [0.50, 0.51]* | 3.81 / 0.51 [0.50, 0.51]* | 1.71 / 0.50 [0.50, 0.51]* | 0.938 / 0.53 [0.52, 0.59]* | 0.806 / 0.87 [0.83, 0.93] |

#### `hdiff`: GPU (ms / ratio to gtfn)

| variant | large | xlarge |
|---|---|---|
| gtfn | 0.652 / 1.00 [1.00, 1.00] | 2.51 / 1.00 [1.00, 1.00]* |
| naive | 0.664 / 1.02 [1.01, 1.03] | 2.57 / 1.02 [1.00, 1.03]* |
| ptr.ex.mddef | 0.661 / 1.01 [1.00, 1.02]* | 2.56 / 1.02 [1.00, 1.02]* |
| ptr.ex.mdt | 0.646 / 0.99 [0.98, 1.00] | 2.49 / 0.99 [0.97, 1.00]* |
| ptr.ex.mdtlb | 0.646 / 0.99 [0.98, 1.00] | 2.49 / 0.99 [0.97, 1.00]* |
| ptr.it.mddef | 0.66 / 1.01 [1.00, 1.02] | 2.55 / 1.02 [1.00, 1.02]* |
| ptr.it.mddefint | 0.66 / 1.01 [1.00, 1.02] | 2.56 / 1.02 [1.00, 1.02]* |
| ptr.it.mdt | 0.647 / 0.99 [0.98, 1.00] | 2.5 / 1.00 [0.98, 1.00]* |
| ptr.it.mdtlb | 0.646 / 0.99 [0.98, 1.00] | 2.49 / 0.99 [0.97, 1.00]* |
| ptr.it.mdtlb+kfence | 0.646 / 0.99 [0.98, 1.00] | 2.49 / 0.99 [0.97, 0.99]* |
| ptr.it.mdtlb.alloc | 2.37 / 3.64 [3.60, 3.67] | 7.88 / 3.14 [3.05, 3.19]* |
| ptr.it.mdtlb.view | 2.52 / 3.86 [3.81, 3.89] | 8.47 / 3.37 [3.29, 3.40]* |
| ptr.it.mdtlb.viewnoinit | 2.33 / 3.57 [3.53, 3.59] | 7.91 / 3.14 [3.04, 3.17]* |
| ptr.it.range | 0.686 / 1.05 [1.04, 1.06] | 2.71 / 1.08 [1.04, 1.10]* |
| ptr64.it.mdtlb | 0.653 / 1.00 [0.99, 1.01] | 2.52 / 1.00 [0.98, 1.01]* |
| vl.ex.mdtlb | 0.646 / 0.99 [0.98, 1.00] | 2.49 / 0.99 [0.97, 1.00]* |
| vl.it.mdtlb | 0.646 / 0.99 [0.98, 1.00] | 2.49 / 0.99 [0.97, 0.99]* |
| vs.it.mdtlb | 0.649 / 1.00 [0.99, 1.00] | 2.51 / 1.00 [0.98, 1.00]* |

#### `hdiff`: CPU, large (ms / ratio to gtfn)

| variant | 1 thr | 2 thr | 4 thr | 8 thr | 18 thr | 36 thr | 72 thr |
|---|---|---|---|---|---|---|---|
| gtfn | 177 / 1.00 [1.00, 1.00]* | 108 / 1.00 [1.00, 1.00]* | 77.5 / 1.00 [1.00, 1.00]* | 61.5 / 1.00 [1.00, 1.00]* | 50.8 / 1.00 [1.00, 1.00]* | 50.7 / 1.00 [1.00, 1.00]* | 55.3 / 1.00 [1.00, 1.00] |
| gtfn_fixed | 155 / 0.88 [0.87, 0.88]* | 79.8 / 0.74 [0.74, 0.74]* | 41.8 / 0.54 [0.53, 0.54]* | 22.4 / 0.36 [0.36, 0.37]* | 11.6 / 0.23 [0.23, 0.23]* | 8.52 / 0.17 [0.17, 0.17]* | 9.28 / 0.17 [0.17, 0.17] |
| naive | - | - | 149 / 1.92 [1.84, 1.95]* | 76.6 / 1.24 [1.22, 1.38]* | 40.5 / 0.80 [0.79, 0.81]* | 25.3 / 0.50 [0.49, 0.51]* | 18.6 / 0.34 [0.32, 0.35] |
| ptr.ex.mdleft | 112 / 0.63 [0.63, 0.63]* | 66.8 / 0.62 [0.61, 0.63]* | 36.8 / 0.48 [0.47, 0.48]* | 19.6 / 0.32 [0.32, 0.33]* | 9.81 / 0.19 [0.19, 0.20]* | 6.81 / 0.13 [0.13, 0.14]* | 5.42 / 0.10 [0.09, 0.10] |
| ptr.ex.rows | 77.8 / 0.44 [0.44, 0.44]* | 39.7 / 0.37 [0.37, 0.37]* | 21.5 / 0.28 [0.27, 0.28]* | 11.7 / 0.19 [0.19, 0.19]* | 7.12 / 0.14 [0.14, 0.14]* | 5.77 / 0.11 [0.11, 0.12]* | 5.73 / 0.10 [0.10, 0.11] |
| ptr.it.mddef | - | - | 150 / 1.94 [1.85, 1.96]* | 79.6 / 1.29 [1.23, 1.34]* | 40.4 / 0.79 [0.78, 0.81]* | 25 / 0.49 [0.49, 0.50]* | 18.2 / 0.33 [0.32, 0.34] |
| ptr.it.mdleft | 112 / 0.63 [0.63, 0.64]* | 66.2 / 0.61 [0.61, 0.62]* | 37.1 / 0.48 [0.47, 0.48]* | 19.7 / 0.32 [0.32, 0.33]* | 9.82 / 0.19 [0.19, 0.20]* | 6.84 / 0.13 [0.13, 0.14]* | 5.41 / 0.10 [0.09, 0.10] |
| ptr.it.rows | 77.2 / 0.44 [0.43, 0.44]* | 39.5 / 0.37 [0.36, 0.37]* | 21.2 / 0.27 [0.27, 0.28]* | 11.6 / 0.19 [0.18, 0.19]* | 7.13 / 0.14 [0.14, 0.14]* | 5.74 / 0.11 [0.11, 0.12]* | 5.72 / 0.10 [0.10, 0.11] |
| ptr.it.rows.alloc | 86 / 0.49 [0.48, 0.49]* | 44.4 / 0.41 [0.41, 0.42]* | 25.6 / 0.33 [0.33, 0.33]* | 14.7 / 0.24 [0.24, 0.24]* | 9.62 / 0.19 [0.19, 0.19]* | 8.36 / 0.16 [0.16, 0.17]* | 9.35 / 0.17 [0.17, 0.17] |
| ptr.it.rows.view | 100 / 0.57 [0.56, 0.57]* | 60.8 / 0.56 [0.56, 0.57]* | 29.3 / 0.38 [0.37, 0.38]* | 17.3 / 0.28 [0.28, 0.29]* | 13 / 0.26 [0.25, 0.26]* | 11.2 / 0.22 [0.22, 0.22]* | 12.1 / 0.22 [0.21, 0.23] |
| ptr.it.rows.viewnoinit | 86.1 / 0.49 [0.48, 0.49]* | 44.8 / 0.42 [0.41, 0.42]* | 25.6 / 0.33 [0.32, 0.33]* | 14.8 / 0.24 [0.24, 0.25]* | 9.68 / 0.19 [0.19, 0.20]* | 8.4 / 0.17 [0.16, 0.17]* | 9.31 / 0.17 [0.17, 0.17] |
| ptr.it.rowsnh | 77.8 / 0.44 [0.44, 0.44]* | 39.5 / 0.37 [0.36, 0.37]* | 21.2 / 0.27 [0.27, 0.28]* | 11.6 / 0.19 [0.18, 0.19]* | 7.12 / 0.14 [0.14, 0.14]* | 5.72 / 0.11 [0.11, 0.12]* | 5.71 / 0.10 [0.10, 0.11] |
| ptr.it.rowssimd | 75.7 / 0.43 [0.43, 0.43]* | 38.4 / 0.36 [0.35, 0.36]* | 20.7 / 0.27 [0.26, 0.27]* | 11.4 / 0.19 [0.18, 0.19]* | 6.91 / 0.14 [0.13, 0.14]* | 5.67 / 0.11 [0.11, 0.11]* | 5.66 / 0.10 [0.10, 0.11] |
| ptr64.it.mdleft | 77.2 / 0.44 [0.43, 0.44]* | 52.5 / 0.49 [0.48, 0.49]* | 29.1 / 0.37 [0.37, 0.38]* | 15.8 / 0.26 [0.25, 0.26]* | 8.65 / 0.17 [0.17, 0.17]* | 6.3 / 0.12 [0.12, 0.13]* | 5.66 / 0.10 [0.10, 0.11] |
| ptr64.it.rows | 75.7 / 0.43 [0.43, 0.43]* | 38.6 / 0.36 [0.36, 0.36]* | 20.8 / 0.27 [0.26, 0.27]* | 11.4 / 0.18 [0.18, 0.19]* | 6.9 / 0.14 [0.13, 0.14]* | 5.65 / 0.11 [0.11, 0.11]* | 5.67 / 0.10 [0.10, 0.10] |
| vl.ex.rows | 76.1 / 0.43 [0.43, 0.43]* | 38.4 / 0.36 [0.35, 0.36]* | 20.8 / 0.27 [0.27, 0.27]* | 11.5 / 0.19 [0.19, 0.19]* | 7 / 0.14 [0.14, 0.14]* | 5.73 / 0.11 [0.11, 0.11]* | 5.73 / 0.10 [0.10, 0.11] |
| vl.it.mdleft | 121 / 0.68 [0.68, 0.69]* | 71.7 / 0.67 [0.66, 0.67]* | 39.5 / 0.51 [0.50, 0.51]* | 20.5 / 0.33 [0.33, 0.34]* | 10.1 / 0.20 [0.20, 0.20]* | 6.71 / 0.13 [0.13, 0.13]* | 5.32 / 0.10 [0.09, 0.10] |
| vl.it.rows | 76.1 / 0.43 [0.43, 0.43]* | 38.3 / 0.36 [0.35, 0.36]* | 20.9 / 0.27 [0.27, 0.27]* | 11.5 / 0.19 [0.18, 0.19]* | 7.12 / 0.14 [0.14, 0.14]* | 5.74 / 0.11 [0.11, 0.11]* | 5.72 / 0.10 [0.10, 0.11] |
| vs.it.rows | 77.8 / 0.44 [0.44, 0.44]* | 39.2 / 0.36 [0.36, 0.37]* | 21.3 / 0.28 [0.27, 0.28]* | 11.7 / 0.19 [0.19, 0.20]* | 7.08 / 0.14 [0.14, 0.14]* | 5.72 / 0.11 [0.11, 0.11]* | 5.71 / 0.10 [0.10, 0.11] |

Same cells, ratio to `gtfn_fixed`:

| variant | 1 thr | 2 thr | 4 thr | 8 thr | 18 thr | 36 thr | 72 thr |
|---|---|---|---|---|---|---|---|
| naive | - | - | 3.56 [3.43, 3.62]* | 3.42 [3.37, 3.79]* | 3.48 [3.44, 3.54]* | 2.97 [2.93, 2.99]* | 2.00 [1.93, 2.08] |
| ptr.ex.mdleft | 0.72 [0.72, 0.72]* | 0.84 [0.83, 0.85]* | 0.88 [0.87, 0.89]* | 0.87 [0.87, 0.88]* | 0.84 [0.84, 0.85]* | 0.80 [0.79, 0.81]* | 0.59 [0.55, 0.61] |
| ptr.ex.rows | 0.50 [0.50, 0.50]* | 0.50 [0.49, 0.50]* | 0.51 [0.51, 0.52]* | 0.52 [0.52, 0.53]* | 0.61 [0.61, 0.62]* | 0.68 [0.67, 0.68]* | 0.62 [0.60, 0.64] |
| ptr.it.mddef | - | - | 3.59 [3.42, 3.63]* | 3.55 [3.36, 3.64]* | 3.47 [3.42, 3.50]* | 2.93 [2.90, 2.95]* | 1.96 [1.88, 2.05] |
| ptr.it.mdleft | 0.72 [0.72, 0.73]* | 0.83 [0.82, 0.84]* | 0.89 [0.88, 0.89]* | 0.88 [0.87, 0.89]* | 0.84 [0.84, 0.85]* | 0.80 [0.79, 0.81]* | 0.58 [0.56, 0.61] |
| ptr.it.rows | 0.50 [0.50, 0.50]* | 0.49 [0.49, 0.50]* | 0.51 [0.50, 0.51]* | 0.52 [0.51, 0.53]* | 0.61 [0.61, 0.62]* | 0.67 [0.66, 0.68]* | 0.62 [0.60, 0.63] |
| ptr.it.rows.alloc | 0.55 [0.55, 0.56]* | 0.56 [0.55, 0.56]* | 0.61 [0.61, 0.62]* | 0.66 [0.65, 0.66]* | 0.83 [0.82, 0.83]* | 0.98 [0.97, 0.99]* | 1.01 [0.99, 1.04] |
| ptr.it.rows.view | 0.65 [0.64, 0.65]* | 0.76 [0.75, 0.77]* | 0.70 [0.70, 0.70]* | 0.77 [0.76, 0.78]* | 1.12 [1.10, 1.14]* | 1.31 [1.28, 1.32]* | 1.31 [1.28, 1.35] |
| ptr.it.rows.viewnoinit | 0.55 [0.55, 0.56]* | 0.56 [0.56, 0.57]* | 0.61 [0.61, 0.62]* | 0.66 [0.65, 0.67]* | 0.83 [0.82, 0.86]* | 0.99 [0.97, 0.99]* | 1.00 [0.97, 1.03] |
| ptr.it.rowsnh | 0.50 [0.50, 0.50]* | 0.49 [0.49, 0.50]* | 0.51 [0.50, 0.51]* | 0.52 [0.51, 0.53]* | 0.61 [0.60, 0.62]* | 0.67 [0.66, 0.68]* | 0.62 [0.59, 0.63] |
| ptr.it.rowssimd | 0.49 [0.49, 0.49]* | 0.48 [0.48, 0.49]* | 0.50 [0.49, 0.50]* | 0.51 [0.50, 0.51]* | 0.59 [0.59, 0.60]* | 0.66 [0.66, 0.67]* | 0.61 [0.60, 0.63] |
| ptr64.it.mdleft | 0.50 [0.50, 0.50]* | 0.66 [0.65, 0.66]* | 0.70 [0.69, 0.70]* | 0.70 [0.70, 0.71]* | 0.74 [0.73, 0.75]* | 0.74 [0.73, 0.75]* | 0.61 [0.59, 0.64] |
| ptr64.it.rows | 0.49 [0.49, 0.49]* | 0.48 [0.48, 0.49]* | 0.50 [0.49, 0.50]* | 0.51 [0.50, 0.51]* | 0.59 [0.59, 0.60]* | 0.66 [0.65, 0.67]* | 0.61 [0.59, 0.62] |
| vl.ex.rows | 0.49 [0.49, 0.49]* | 0.48 [0.48, 0.49]* | 0.50 [0.49, 0.51]* | 0.52 [0.51, 0.53]* | 0.60 [0.59, 0.62]* | 0.67 [0.66, 0.68]* | 0.62 [0.60, 0.63] |
| vl.it.mdleft | 0.78 [0.77, 0.78]* | 0.90 [0.89, 0.90]* | 0.94 [0.94, 0.95]* | 0.92 [0.91, 0.92]* | 0.87 [0.86, 0.88]* | 0.79 [0.78, 0.79]* | 0.57 [0.54, 0.60] |
| vl.it.rows | 0.49 [0.49, 0.49]* | 0.48 [0.48, 0.48]* | 0.50 [0.50, 0.51]* | 0.52 [0.51, 0.52]* | 0.61 [0.60, 0.62]* | 0.67 [0.66, 0.68]* | 0.62 [0.60, 0.63] |
| vs.it.rows | 0.50 [0.50, 0.51]* | 0.49 [0.49, 0.50]* | 0.51 [0.50, 0.52]* | 0.52 [0.52, 0.53]* | 0.61 [0.60, 0.62]* | 0.67 [0.66, 0.68]* | 0.62 [0.60, 0.63] |

#### `tmpst`: GPU (ms / ratio to gtfn)

| variant | large | xlarge |
|---|---|---|
| gtfn | 0.27 / 1.00 [1.00, 1.00] | 1.02 / 1.00 [1.00, 1.00]* |
| naive | 0.273 / 1.01 [1.01, 1.01] | 1.04 / 1.02 [1.01, 1.02]* |
| ptr.ex.mddef | 0.271 / 1.01 [1.00, 1.01]* | 1.03 / 1.02 [1.01, 1.02]* |
| ptr.ex.mdt | 0.264 / 0.98 [0.97, 0.98] | 1 / 0.98 [0.98, 0.99]* |
| ptr.ex.mdtlb | 0.264 / 0.98 [0.97, 0.98] | 1 / 0.98 [0.98, 0.99]* |
| ptr.it.mddef | 0.272 / 1.01 [1.00, 1.01] | 1.03 / 1.02 [1.01, 1.02]* |
| ptr.it.mddefint | 0.272 / 1.01 [1.00, 1.01] | 1.03 / 1.02 [1.01, 1.02]* |
| ptr.it.mdt | 0.264 / 0.98 [0.97, 0.98] | 1 / 0.98 [0.98, 0.99]* |
| ptr.it.mdtlb | 0.264 / 0.98 [0.98, 0.98] | 1 / 0.98 [0.98, 0.99]* |
| ptr.it.mdtlb+kfence | 0.264 / 0.98 [0.98, 0.98] | 1 / 0.98 [0.98, 0.99]* |
| ptr.it.mdtlb.alloc | 0.893 / 3.31 [3.28, 3.34] | 2.83 / 2.78 [2.76, 2.83]* |
| ptr.it.mdtlb.view | 0.941 / 3.49 [3.47, 3.51] | 3.03 / 2.98 [2.94, 3.00]* |
| ptr.it.mdtlb.viewnoinit | 0.88 / 3.26 [3.24, 3.29] | 2.84 / 2.80 [2.74, 2.81]* |
| ptr.it.range | 0.275 / 1.02 [1.01, 1.03] | 1.06 / 1.04 [1.03, 1.05]* |
| ptr64.it.mdtlb | 0.267 / 0.99 [0.98, 0.99] | 1.01 / 0.99 [0.99, 1.00]* |
| vl.ex.mdtlb | 0.264 / 0.98 [0.98, 0.98] | 1 / 0.98 [0.98, 0.99]* |
| vl.it.mdtlb | 0.264 / 0.98 [0.98, 0.98] | 1 / 0.98 [0.98, 0.99]* |
| vs.it.mdtlb | 0.266 / 0.98 [0.98, 0.99] | 1.01 / 0.99 [0.98, 0.99]* |

#### `tmpst`: CPU, large (ms / ratio to gtfn)

| variant | 1 thr | 2 thr | 4 thr | 8 thr | 18 thr | 36 thr | 72 thr |
|---|---|---|---|---|---|---|---|
| gtfn | 72.3 / 1.00 [1.00, 1.00]* | 42.9 / 1.00 [1.00, 1.00]* | 29.4 / 1.00 [1.00, 1.00]* | 23.2 / 1.00 [1.00, 1.00]* | 18.1 / 1.00 [1.00, 1.00]* | 17.4 / 1.00 [1.00, 1.00]* | 17.5 / 1.00 [1.00, 1.00] |
| gtfn_fixed | 63.5 / 0.88 [0.86, 0.89]* | 32.7 / 0.76 [0.76, 0.78]* | 16.8 / 0.57 [0.56, 0.59]* | 9.04 / 0.39 [0.38, 0.40]* | 4.62 / 0.26 [0.25, 0.26]* | 3.54 / 0.20 [0.20, 0.21]* | 3.62 / 0.21 [0.19, 0.21] |
| naive | - | - | 57.8 / 1.97 [1.91, 2.02]* | 32.8 / 1.42 [1.36, 1.46]* | 17 / 0.94 [0.92, 0.96]* | 10.3 / 0.59 [0.57, 0.61]* | 7.17 / 0.41 [0.37, 0.43] |
| ptr.ex.mdleft | 34.3 / 0.47 [0.47, 0.48]* | 22.1 / 0.51 [0.51, 0.53]* | 12.8 / 0.43 [0.43, 0.45]* | 6.62 / 0.29 [0.28, 0.29]* | 3.72 / 0.21 [0.20, 0.21]* | 2.73 / 0.16 [0.15, 0.16]* | 2.41 / 0.14 [0.13, 0.14] |
| ptr.ex.rows | 30.8 / 0.43 [0.42, 0.43]* | 15.3 / 0.36 [0.35, 0.36]* | 7.93 / 0.27 [0.27, 0.28]* | 4.36 / 0.19 [0.18, 0.19]* | 2.78 / 0.15 [0.15, 0.16]* | 2.18 / 0.13 [0.12, 0.13]* | 2.2 / 0.13 [0.11, 0.13] |
| ptr.it.mddef | - | - | 58.3 / 1.98 [1.95, 2.03]* | 33.2 / 1.43 [1.39, 1.48]* | 17.2 / 0.95 [0.93, 0.97]* | 10.6 / 0.61 [0.59, 0.62]* | 7.22 / 0.41 [0.37, 0.43] |
| ptr.it.mdleft | 34.2 / 0.47 [0.47, 0.48]* | 22.4 / 0.52 [0.51, 0.53]* | 12.7 / 0.43 [0.42, 0.44]* | 6.79 / 0.29 [0.28, 0.30]* | 3.71 / 0.21 [0.20, 0.21]* | 2.75 / 0.16 [0.15, 0.16]* | 2.45 / 0.14 [0.13, 0.15] |
| ptr.it.rows | 30.8 / 0.43 [0.42, 0.43]* | 15.4 / 0.36 [0.35, 0.36]* | 7.96 / 0.27 [0.27, 0.28]* | 4.38 / 0.19 [0.19, 0.19]* | 2.82 / 0.16 [0.15, 0.16]* | 2.22 / 0.13 [0.12, 0.13]* | 2.25 / 0.13 [0.12, 0.13] |
| ptr.it.rows.alloc | 32.8 / 0.45 [0.45, 0.46]* | 17.1 / 0.40 [0.39, 0.40]* | 9.42 / 0.32 [0.32, 0.33]* | 5.72 / 0.25 [0.24, 0.25]* | 3.73 / 0.21 [0.20, 0.21]* | 3.33 / 0.19 [0.18, 0.20]* | 3.68 / 0.21 [0.19, 0.22] |
| ptr.it.rows.view | 38.1 / 0.53 [0.52, 0.54]* | 22.8 / 0.53 [0.52, 0.54]* | 10.8 / 0.37 [0.36, 0.38]* | 6.73 / 0.29 [0.28, 0.30]* | 4.82 / 0.27 [0.26, 0.28]* | 4.41 / 0.25 [0.25, 0.26]* | 4.59 / 0.26 [0.25, 0.28] |
| ptr.it.rows.viewnoinit | 32.7 / 0.45 [0.45, 0.46]* | 17.2 / 0.40 [0.40, 0.41]* | 9.52 / 0.32 [0.32, 0.34]* | 5.89 / 0.25 [0.24, 0.26]* | 3.75 / 0.21 [0.20, 0.21]* | 3.33 / 0.19 [0.18, 0.20]* | 3.66 / 0.21 [0.20, 0.22] |
| ptr.it.rowsnh | 30.9 / 0.43 [0.42, 0.43]* | 15.4 / 0.36 [0.36, 0.36]* | 8.01 / 0.27 [0.27, 0.28]* | 4.39 / 0.19 [0.19, 0.19]* | 2.82 / 0.16 [0.15, 0.16]* | 2.22 / 0.13 [0.12, 0.13]* | 2.25 / 0.13 [0.12, 0.13] |
| ptr.it.rowssimd | 30.8 / 0.43 [0.42, 0.43]* | 15.4 / 0.36 [0.35, 0.36]* | 7.94 / 0.27 [0.27, 0.28]* | 4.37 / 0.19 [0.18, 0.19]* | 2.79 / 0.15 [0.15, 0.16]* | 2.18 / 0.13 [0.12, 0.13]* | 2.23 / 0.13 [0.12, 0.13] |
| ptr64.it.mdleft | 29.6 / 0.41 [0.40, 0.42]* | 20.7 / 0.48 [0.48, 0.49]* | 11.7 / 0.40 [0.39, 0.41]* | 6.22 / 0.27 [0.26, 0.27]* | 3.55 / 0.20 [0.19, 0.20]* | 2.61 / 0.15 [0.15, 0.15]* | 2.43 / 0.14 [0.13, 0.14] |
| ptr64.it.rows | 30.8 / 0.43 [0.42, 0.43]* | 15.4 / 0.36 [0.36, 0.36]* | 7.98 / 0.27 [0.27, 0.28]* | 4.37 / 0.19 [0.18, 0.19]* | 2.81 / 0.16 [0.15, 0.16]* | 2.23 / 0.13 [0.12, 0.13]* | 2.25 / 0.13 [0.12, 0.13] |
| vl.ex.rows | 30.8 / 0.43 [0.42, 0.43]* | 15.4 / 0.36 [0.35, 0.36]* | 7.95 / 0.27 [0.27, 0.28]* | 4.37 / 0.19 [0.18, 0.19]* | 2.81 / 0.16 [0.15, 0.16]* | 2.2 / 0.13 [0.12, 0.13]* | 2.21 / 0.13 [0.12, 0.13] |
| vl.it.mdleft | 33.7 / 0.47 [0.46, 0.47]* | 21.9 / 0.51 [0.51, 0.52]* | 12.7 / 0.43 [0.42, 0.44]* | 6.76 / 0.29 [0.28, 0.30]* | 3.71 / 0.21 [0.20, 0.21]* | 2.75 / 0.16 [0.15, 0.16]* | 2.41 / 0.14 [0.13, 0.14] |
| vl.it.rows | 30.8 / 0.43 [0.42, 0.43]* | 15.4 / 0.36 [0.35, 0.36]* | 7.94 / 0.27 [0.27, 0.28]* | 4.38 / 0.19 [0.18, 0.19]* | 2.81 / 0.16 [0.15, 0.16]* | 2.17 / 0.13 [0.12, 0.13]* | 2.22 / 0.13 [0.11, 0.13] |
| vs.it.rows | 30.9 / 0.43 [0.42, 0.43]* | 15.3 / 0.36 [0.35, 0.36]* | 7.94 / 0.27 [0.27, 0.28]* | 4.38 / 0.19 [0.18, 0.19]* | 2.76 / 0.15 [0.15, 0.16]* | 2.19 / 0.13 [0.12, 0.13]* | 2.21 / 0.13 [0.11, 0.13] |

Same cells, ratio to `gtfn_fixed`:

| variant | 1 thr | 2 thr | 4 thr | 8 thr | 18 thr | 36 thr | 72 thr |
|---|---|---|---|---|---|---|---|
| naive | - | - | 3.45 [3.35, 3.51]* | 3.63 [3.49, 3.76]* | 3.67 [3.60, 3.74]* | 2.92 [2.88, 3.00]* | 1.98 [1.88, 2.10] |
| ptr.ex.mdleft | 0.54 [0.54, 0.54]* | 0.67 [0.66, 0.69]* | 0.76 [0.75, 0.78]* | 0.73 [0.72, 0.74]* | 0.81 [0.79, 0.81]* | 0.77 [0.76, 0.80]* | 0.67 [0.63, 0.70] |
| ptr.ex.rows | 0.49 [0.48, 0.49]* | 0.47 [0.46, 0.47]* | 0.47 [0.47, 0.48]* | 0.48 [0.48, 0.49]* | 0.60 [0.58, 0.61]* | 0.61 [0.59, 0.63]* | 0.61 [0.57, 0.64] |
| ptr.it.mddef | - | - | 3.48 [3.42, 3.56]* | 3.68 [3.58, 3.80]* | 3.71 [3.62, 3.78]* | 3.00 [2.95, 3.07]* | 2.00 [1.89, 2.11] |
| ptr.it.mdleft | 0.54 [0.53, 0.55]* | 0.68 [0.67, 0.69]* | 0.76 [0.74, 0.77]* | 0.75 [0.74, 0.76]* | 0.80 [0.79, 0.82]* | 0.78 [0.76, 0.80]* | 0.68 [0.64, 0.71] |
| ptr.it.rows | 0.48 [0.48, 0.49]* | 0.47 [0.46, 0.47]* | 0.47 [0.47, 0.48]* | 0.48 [0.48, 0.49]* | 0.61 [0.59, 0.62]* | 0.63 [0.59, 0.64]* | 0.62 [0.58, 0.66] |
| ptr.it.rows.alloc | 0.52 [0.51, 0.52]* | 0.52 [0.51, 0.53]* | 0.56 [0.55, 0.58]* | 0.63 [0.62, 0.64]* | 0.81 [0.79, 0.81]* | 0.94 [0.92, 0.98]* | 1.02 [0.98, 1.07] |
| ptr.it.rows.view | 0.60 [0.59, 0.60]* | 0.70 [0.68, 0.71]* | 0.64 [0.63, 0.66]* | 0.74 [0.73, 0.76]* | 1.04 [1.03, 1.08]* | 1.25 [1.23, 1.29]* | 1.27 [1.23, 1.34] |
| ptr.it.rows.viewnoinit | 0.52 [0.51, 0.52]* | 0.53 [0.51, 0.53]* | 0.57 [0.56, 0.59]* | 0.65 [0.63, 0.66]* | 0.81 [0.80, 0.82]* | 0.94 [0.93, 0.97]* | 1.01 [0.97, 1.05] |
| ptr.it.rowsnh | 0.49 [0.48, 0.49]* | 0.47 [0.46, 0.48]* | 0.48 [0.47, 0.48]* | 0.49 [0.48, 0.49]* | 0.61 [0.60, 0.62]* | 0.63 [0.62, 0.65]* | 0.62 [0.59, 0.66] |
| ptr.it.rowssimd | 0.49 [0.48, 0.49]* | 0.47 [0.46, 0.48]* | 0.47 [0.47, 0.48]* | 0.48 [0.47, 0.49]* | 0.60 [0.60, 0.61]* | 0.62 [0.60, 0.64]* | 0.62 [0.58, 0.65] |
| ptr64.it.mdleft | 0.47 [0.46, 0.47]* | 0.63 [0.62, 0.64]* | 0.70 [0.69, 0.71]* | 0.69 [0.68, 0.69]* | 0.77 [0.75, 0.78]* | 0.74 [0.72, 0.76]* | 0.67 [0.64, 0.70] |
| ptr64.it.rows | 0.49 [0.48, 0.49]* | 0.47 [0.46, 0.47]* | 0.48 [0.47, 0.48]* | 0.48 [0.48, 0.49]* | 0.61 [0.60, 0.62]* | 0.63 [0.62, 0.64]* | 0.62 [0.59, 0.65] |
| vl.ex.rows | 0.49 [0.48, 0.49]* | 0.47 [0.46, 0.48]* | 0.47 [0.47, 0.48]* | 0.48 [0.47, 0.49]* | 0.61 [0.59, 0.62]* | 0.62 [0.59, 0.64]* | 0.61 [0.58, 0.65] |
| vl.it.mdleft | 0.53 [0.53, 0.54]* | 0.67 [0.66, 0.68]* | 0.76 [0.74, 0.77]* | 0.75 [0.73, 0.76]* | 0.80 [0.79, 0.81]* | 0.78 [0.76, 0.80]* | 0.67 [0.64, 0.70] |
| vl.it.rows | 0.49 [0.48, 0.49]* | 0.47 [0.46, 0.47]* | 0.47 [0.47, 0.48]* | 0.48 [0.47, 0.49]* | 0.61 [0.59, 0.61]* | 0.61 [0.59, 0.63]* | 0.62 [0.57, 0.65] |
| vs.it.rows | 0.49 [0.48, 0.49]* | 0.47 [0.46, 0.47]* | 0.47 [0.47, 0.48]* | 0.48 [0.47, 0.49]* | 0.60 [0.59, 0.61]* | 0.62 [0.59, 0.64]* | 0.61 [0.56, 0.65] |

#### `tri`: GPU (ms / ratio to gtfn)

| variant | large | xlarge |
|---|---|---|
| gtfn | 0.523 / 1.00 [1.00, 1.00] | 2.04 / 1.00 [1.00, 1.00] |
| naive | 0.532 / 1.02 [1.02, 1.02] | 2.08 / 1.02 [1.02, 1.02]* |
| ptr.ex.colmdt | 0.531 / 1.02 [1.01, 1.02] | 2.08 / 1.02 [1.02, 1.02]* |
| ptr.ex.colmdtlb | 0.532 / 1.02 [1.01, 1.02] | 2.08 / 1.02 [1.02, 1.02] |
| ptr.it.colmddef | 0.529 / 1.01 [1.01, 1.01] | 2.07 / 1.01 [1.01, 1.01]* |
| ptr.it.colmdt | 0.531 / 1.02 [1.01, 1.02] | 2.08 / 1.02 [1.02, 1.02]* |
| ptr.it.colmdtlb | 0.532 / 1.02 [1.01, 1.02] | 2.08 / 1.02 [1.02, 1.02] |
| ptr.it.colmdtlb+kfence | 0.531 / 1.02 [1.01, 1.02] | 2.08 / 1.02 [1.02, 1.02]* |
| ptr.it.colmdtlb.split | 0.521 / 1.00 [0.99, 1.00] | 2.06 / 1.01 [1.00, 1.01] |
| ptr.it.colrange | 0.529 / 1.01 [1.01, 1.01] | 2.02 / 0.99 [0.99, 0.99] |
| ptr.it.colrangelb | 0.528 / 1.01 [1.01, 1.01] | 2.01 / 0.99 [0.99, 0.99]* |
| ptr.pw.colmdtlb | 0.532 / 1.02 [1.02, 1.02]* | 2.09 / 1.03 [1.02, 1.03]* |
| ptr.pw.colmdtlb.split | 0.53 / 1.02 [1.01, 1.02]* | 2.05 / 1.00 [1.00, 1.01]* |
| ptr.pw.colrange | 0.528 / 1.01 [1.01, 1.01]* | 2.01 / 0.99 [0.99, 0.99]* |
| ptr64.it.colmdtlb | 0.53 / 1.02 [1.01, 1.02] | 2.08 / 1.02 [1.02, 1.02]* |
| vl.ex.colmdtlb | 0.557 / 1.07 [1.06, 1.07] | 2.14 / 1.05 [1.05, 1.06]* |
| vl.it.colmdtlb | 0.556 / 1.07 [1.06, 1.07] | 2.14 / 1.05 [1.05, 1.06]* |
| vs.it.colmdtlb | 0.534 / 1.02 [1.02, 1.03] | 2.1 / 1.03 [1.03, 1.03]* |

#### `tri`: CPU, large (ms / ratio to gtfn)

| variant | 1 thr | 2 thr | 4 thr | 8 thr | 18 thr | 36 thr | 72 thr |
|---|---|---|---|---|---|---|---|
| gtfn | 157 / 1.00 [1.00, 1.00] | 79.1 / 1.00 [1.00, 1.00] | 39.2 / 1.00 [1.00, 1.00] | 20.5 / 1.00 [1.00, 1.00]* | 9.83 / 1.00 [1.00, 1.00] | 6.58 / 1.00 [1.00, 1.00]* | 5.91 / 1.00 [1.00, 1.00] |
| naive | - | - | 89.3 / 2.23 [2.21, 2.25]* | 47.8 / 2.33 [2.23, 2.35]* | 25.2 / 2.53 [2.42, 2.55]* | 20.8 / 3.16 [3.09, 3.26]* | 20.1 / 3.43 [3.36, 3.52] |
| ptr.ex.chunk16 | 102 / 0.65 [0.64, 0.66] | 52.5 / 0.66 [0.65, 0.67] | 27.2 / 0.69 [0.67, 0.72] | 15.1 / 0.74 [0.71, 0.74]* | 8.7 / 0.89 [0.82, 0.92] | 6.37 / 0.97 [0.95, 0.98]* | 5.1 / 0.87 [0.85, 0.91] |
| ptr.ex.chunk32 | 67.9 / 0.43 [0.43, 0.44]* | 34.5 / 0.44 [0.43, 0.44]* | 17.9 / 0.47 [0.46, 0.48]* | - | 6 / 0.62 [0.61, 0.63]* | - | 4.99 / 0.84 [0.84, 0.85]* |
| ptr.ex.chunk64 | 81.9 / 0.52 [0.52, 0.52]* | 41.1 / 0.52 [0.52, 0.52]* | 21.7 / 0.57 [0.56, 0.58]* | - | 8.02 / 0.83 [0.81, 0.84]* | - | 5.14 / 0.87 [0.86, 0.87]* |
| ptr.ex.chunk8 | 79.6 / 0.51 [0.50, 0.51]* | 40.3 / 0.51 [0.51, 0.51]* | 21 / 0.55 [0.55, 0.56]* | - | 7.55 / 0.78 [0.76, 0.79]* | - | 4.99 / 0.84 [0.84, 0.85]* |
| ptr.ex.col | 156 / 0.99 [0.99, 1.00] | 78.6 / 0.99 [0.99, 1.01] | 39.8 / 1.02 [0.98, 1.05] | 20.4 / 0.99 [0.95, 1.02]* | 9.51 / 0.97 [0.90, 1.03] | 5.59 / 0.85 [0.83, 0.86]* | 5.04 / 0.85 [0.83, 0.87] |
| ptr.it.chunk16 | 448 / 2.85 [2.81, 2.87] | 224 / 2.83 [2.79, 2.87] | 112 / 2.86 [2.78, 2.96] | 56.1 / 2.73 [2.62, 2.76]* | 25 / 2.55 [2.40, 2.60] | 12.7 / 1.93 [1.90, 1.95]* | 6.77 / 1.15 [1.11, 1.19] |
| ptr.it.chunk16.split | 665 / 4.23 [4.21, 4.25] | 334 / 4.22 [4.18, 4.25] | 167 / 4.27 [4.13, 4.42] | 90 / 4.38 [4.20, 4.42]* | 37.6 / 3.83 [3.61, 3.92] | 18.9 / 2.88 [2.83, 2.89]* | 9.9 / 1.69 [1.63, 1.73] |
| ptr.it.chunk64 | 459 / 2.92 [2.89, 2.93] | 230 / 2.91 [2.88, 2.94] | 115 / 2.94 [2.85, 3.04] | 57.7 / 2.81 [2.70, 2.84]* | 25.9 / 2.64 [2.49, 2.70] | 13.2 / 2.01 [1.98, 2.03]* | 7.14 / 1.23 [1.19, 1.26] |
| ptr.it.chunk8 | 451 / 2.87 [2.84, 2.90] | 226 / 2.86 [2.81, 2.91] | 113 / 2.90 [2.81, 3.00] | 56.6 / 2.75 [2.64, 2.78]* | 25.3 / 2.58 [2.42, 2.65] | 12.7 / 1.94 [1.91, 1.96]* | 6.63 / 1.12 [1.10, 1.15] |
| ptr.it.col | 155 / 0.99 [0.98, 1.00] | 78.5 / 0.99 [0.98, 1.01] | 39.8 / 1.02 [0.98, 1.06] | 20.3 / 0.99 [0.95, 1.00]* | 9.33 / 0.95 [0.90, 0.97] | 5.61 / 0.85 [0.84, 0.86]* | 5.03 / 0.85 [0.83, 0.88] |
| ptr.it.col.split | 156 / 0.99 [0.99, 1.00] | 78.4 / 0.99 [0.98, 1.00] | 38.8 / 0.99 [0.98, 1.01] | 20.3 / 0.99 [0.95, 1.01]* | 9.73 / 0.99 [0.95, 1.02] | 6.55 / 1.00 [0.98, 1.01]* | 5.93 / 1.00 [0.99, 1.01] |
| ptr.pw.col | 156 / 0.99 [0.98, 1.00] | 78.6 / 0.99 [0.99, 1.00] | 39.8 / 1.02 [0.98, 1.05] | 20.5 / 1.00 [0.97, 1.03]* | 9.4 / 0.96 [0.91, 0.99] | 5.59 / 0.85 [0.84, 0.86]* | 5.05 / 0.85 [0.84, 0.87] |
| ptr64.it.chunk16 | 133 / 0.85 [0.81, 0.89] | 70.3 / 0.89 [0.87, 0.91] | 36.7 / 0.94 [0.90, 0.99] | 19.6 / 0.95 [0.91, 0.97]* | 9.1 / 0.93 [0.88, 1.01] | 5.75 / 0.87 [0.86, 0.89]* | 4.81 / 0.81 [0.80, 0.86] |
| ptr64.it.col | 156 / 0.99 [0.99, 0.99]* | 78.3 / 0.99 [0.98, 1.00]* | 40.2 / 1.00 [0.99, 1.03]* | 21.1 / 1.03 [0.96, 1.07]* | 9.82 / 0.98 [0.93, 1.09]* | 5.61 / 0.85 [0.83, 0.86]* | 4.94 / 0.84 [0.83, 0.85] |
| vl.ex.chunk16 | 100 / 0.64 [0.62, 0.65] | 51.8 / 0.65 [0.64, 0.66] | 26.7 / 0.68 [0.67, 0.70] | 14.9 / 0.73 [0.70, 0.73]* | 8.59 / 0.87 [0.82, 0.90] | 6.32 / 0.96 [0.94, 0.97]* | 5.09 / 0.87 [0.85, 0.90] |
| vl.it.chunk16 | 286 / 1.82 [1.82, 1.82]* | 142 / 1.80 [1.78, 1.80]* | 71.2 / 1.78 [1.77, 1.80]* | 35.7 / 1.74 [1.67, 1.75]* | 16.2 / 1.63 [1.55, 1.64]* | 8.3 / 1.26 [1.24, 1.27]* | 4.9 / 0.84 [0.82, 0.85] |
| vl.it.col | 156 / 0.99 [0.99, 1.00]* | 78.3 / 0.99 [0.98, 1.00]* | 39.7 / 0.99 [0.98, 1.00]* | 20.5 / 1.00 [0.95, 1.02]* | 9.57 / 0.96 [0.92, 1.04]* | 5.59 / 0.85 [0.84, 0.86]* | 4.93 / 0.84 [0.83, 0.85] |
| vs.it.chunk16 | 293 / 1.86 [1.86, 1.86]* | 140 / 1.77 [1.75, 1.77]* | 75.7 / 1.89 [1.88, 1.91]* | 35.6 / 1.73 [1.66, 1.75]* | 15.9 / 1.59 [1.52, 1.61]* | 8.12 / 1.23 [1.21, 1.24]* | 5.04 / 0.86 [0.83, 0.90] |

#### `tritmp`: GPU (ms / ratio to gtfn)

| variant | large | xlarge |
|---|---|---|
| gtfn | 0.505 / 1.00 [1.00, 1.00] | 1.96 / 1.00 [1.00, 1.00] |
| naive | 0.532 / 1.05 [1.05, 1.05] | 2.09 / 1.07 [1.07, 1.07]* |
| ptr.ex.colmdt | 0.531 / 1.05 [1.05, 1.05] | 2.09 / 1.07 [1.07, 1.07]* |
| ptr.ex.colmdtlb | 0.531 / 1.05 [1.05, 1.05] | 2.08 / 1.07 [1.06, 1.07] |
| ptr.it.colmddef | 0.528 / 1.04 [1.04, 1.05] | 2.07 / 1.06 [1.06, 1.06]* |
| ptr.it.colmdt | 0.53 / 1.05 [1.05, 1.05] | 2.09 / 1.07 [1.07, 1.07]* |
| ptr.it.colmdtlb | 0.53 / 1.05 [1.05, 1.05] | 2.08 / 1.07 [1.06, 1.07] |
| ptr.it.colmdtlb+kfence | 0.531 / 1.05 [1.05, 1.05] | 2.09 / 1.07 [1.07, 1.07]* |
| ptr.it.colmdtlb.alloc | 1.71 / 3.37 [3.35, 3.40] | 5.61 / 2.87 [2.86, 2.89]* |
| ptr.it.colmdtlb.lay-c0 | 0.509 / 1.01 [1.00, 1.01] | 1.99 / 1.02 [1.01, 1.02] |
| ptr.it.colmdtlb.lay-c3 | 0.518 / 1.03 [1.02, 1.03] | 2.05 / 1.05 [1.04, 1.05] |
| ptr.it.colmdtlb.lay-p0 | 0.512 / 1.01 [1.01, 1.02] | 2.01 / 1.03 [1.02, 1.03] |
| ptr.it.colmdtlb.lay-p3 | 0.531 / 1.05 [1.05, 1.05] | 2.08 / 1.06 [1.06, 1.07] |
| ptr.it.colmdtlb.split | 0.521 / 1.03 [1.03, 1.03] | 2.06 / 1.05 [1.05, 1.06] |
| ptr.it.colmdtlb.split.lay-c0 | 0.507 / 1.00 [1.00, 1.01] | 2.02 / 1.03 [1.03, 1.04] |
| ptr.it.colmdtlb.split.lay-c3 | 0.505 / 1.00 [1.00, 1.00] | 2.03 / 1.03 [1.03, 1.04] |
| ptr.it.colmdtlb.split.lay-p0 | 0.509 / 1.01 [1.01, 1.01] | 2.01 / 1.03 [1.02, 1.03] |
| ptr.it.colmdtlb.split.lay-p3 | 0.521 / 1.03 [1.03, 1.03] | 2.06 / 1.05 [1.05, 1.06] |
| ptr.it.colmdtlb.view | 1.8 / 3.56 [3.54, 3.60] | 5.96 / 3.05 [3.04, 3.07]* |
| ptr.it.colmdtlb.viewnoinit | 1.68 / 3.33 [3.30, 3.35] | 5.62 / 2.88 [2.84, 2.91]* |
| ptr.it.colrange | 0.525 / 1.04 [1.04, 1.04] | 2.01 / 1.03 [1.02, 1.04] |
| ptr.it.colrange.lay-c0 | 0.504 / 1.00 [0.99, 1.00] | 1.92 / 0.98 [0.98, 0.98] |
| ptr.it.colrange.lay-c3 | 0.511 / 1.01 [1.01, 1.01] | 1.98 / 1.01 [1.00, 1.01] |
| ptr.it.colrange.lay-p0 | 0.506 / 1.00 [1.00, 1.00] | 1.93 / 0.99 [0.98, 0.99] |
| ptr.it.colrange.lay-p3 | 0.525 / 1.04 [1.04, 1.04] | 2.01 / 1.03 [1.02, 1.03] |
| ptr.it.colrangelb | 0.525 / 1.04 [1.04, 1.04] | 2.02 / 1.03 [1.03, 1.04]* |
| ptr.pw.colmdtlb | 0.53 / 1.05 [1.05, 1.05]* | 2.13 / 1.09 [1.08, 1.09]* |
| ptr.pw.colmdtlb.split | 0.53 / 1.05 [1.05, 1.05]* | 2.06 / 1.05 [1.05, 1.06]* |
| ptr.pw.colrange | 0.525 / 1.04 [1.04, 1.04]* | 2.02 / 1.03 [1.03, 1.04]* |
| ptr64.it.colmdtlb | 0.53 / 1.05 [1.05, 1.05] | 2.09 / 1.07 [1.07, 1.07]* |
| vl.ex.colmdtlb | 0.556 / 1.10 [1.10, 1.10] | 2.17 / 1.11 [1.11, 1.11]* |
| vl.it.colmdtlb | 0.556 / 1.10 [1.10, 1.10] | 2.18 / 1.11 [1.11, 1.12]* |
| vs.it.colmdtlb | 0.535 / 1.06 [1.05, 1.06] | 2.11 / 1.08 [1.08, 1.08]* |

#### `tritmp`: CPU, large (ms / ratio to gtfn)

| variant | 1 thr | 2 thr | 4 thr | 8 thr | 18 thr | 36 thr | 72 thr |
|---|---|---|---|---|---|---|---|
| gtfn | 294 / 1.00 [1.00, 1.00] | 165 / 1.00 [1.00, 1.00] | 99.9 / 1.00 [1.00, 1.00] | 62.6 / 1.00 [1.00, 1.00]* | 45.8 / 1.00 [1.00, 1.00] | 41.7 / 1.00 [1.00, 1.00]* | 39.7 / 1.00 [1.00, 1.00] |
| gtfn_fixed | 254 / 0.87 [0.86, 0.87] | 130 / 0.79 [0.77, 0.81] | 66.2 / 0.66 [0.63, 0.69] | 36.3 / 0.58 [0.58, 0.58]* | 18 / 0.39 [0.37, 0.41] | 13.6 / 0.33 [0.32, 0.33]* | 16 / 0.41 [0.32, 0.43] |
| naive | - | - | 92.3 / 0.95 [0.94, 0.96]* | 50.3 / 0.80 [0.80, 0.81]* | 26.2 / 0.60 [0.60, 0.61]* | 19.4 / 0.46 [0.46, 0.47]* | 23.1 / 0.59 [0.58, 0.61] |
| ptr.ex.chunk16 | 99.9 / 0.34 [0.33, 0.35] | 52 / 0.31 [0.31, 0.32] | 26.9 / 0.27 [0.25, 0.28] | 15.1 / 0.24 [0.24, 0.24]* | 8.56 / 0.19 [0.17, 0.20] | 5.92 / 0.14 [0.14, 0.14]* | 5.06 / 0.13 [0.12, 0.14] |
| ptr.ex.chunk32 | 69.4 / 0.24 [0.24, 0.24]* | 34.9 / 0.21 [0.21, 0.22]* | 18.6 / 0.18 [0.18, 0.19]* | - | 6.19 / 0.13 [0.12, 0.13]* | - | 4.68 / 0.11 [0.11, 0.11]* |
| ptr.ex.chunk64 | 83.5 / 0.28 [0.28, 0.29]* | 42.4 / 0.26 [0.26, 0.26]* | 22 / 0.21 [0.21, 0.22]* | - | 7.99 / 0.17 [0.16, 0.17]* | - | 4.86 / 0.12 [0.11, 0.12]* |
| ptr.ex.chunk8 | 78.7 / 0.27 [0.27, 0.27]* | 39.5 / 0.24 [0.24, 0.24]* | 21.1 / 0.20 [0.20, 0.21]* | - | 7.37 / 0.15 [0.15, 0.16]* | - | 4.65 / 0.11 [0.11, 0.11]* |
| ptr.ex.col | 155 / 0.53 [0.52, 0.53] | 78.7 / 0.48 [0.47, 0.48] | 41.2 / 0.41 [0.38, 0.43] | 20.7 / 0.33 [0.33, 0.33]* | 9.97 / 0.22 [0.20, 0.24] | 6.1 / 0.15 [0.13, 0.16]* | 5.01 / 0.13 [0.11, 0.14] |
| ptr.it.chunk16 | 421 / 1.43 [1.41, 1.45] | 211 / 1.28 [1.25, 1.29] | 107 / 1.07 [1.01, 1.11] | 53.6 / 0.86 [0.85, 0.86]* | 24.1 / 0.53 [0.50, 0.55] | 12.1 / 0.29 [0.29, 0.29]* | 6.39 / 0.16 [0.15, 0.17] |
| ptr.it.chunk16.alloc | 442 / 1.50 [1.49, 1.51]* | 220 / 1.32 [1.30, 1.33]* | 112 / 1.16 [1.14, 1.16]* | 56.7 / 0.91 [0.90, 0.91]* | 26.5 / 0.61 [0.60, 0.61]* | 16.7 / 0.40 [0.40, 0.41]* | 16.2 / 0.42 [0.41, 0.44] |
| ptr.it.chunk16.view | 432 / 1.47 [1.46, 1.47]* | 228 / 1.36 [1.35, 1.37]* | 112 / 1.16 [1.15, 1.17]* | 57.4 / 0.92 [0.91, 0.93]* | 27.8 / 0.64 [0.63, 0.64]* | 16.1 / 0.39 [0.38, 0.40]* | 10.6 / 0.27 [0.26, 0.30] |
| ptr.it.chunk16.viewnoinit | 434 / 1.47 [1.47, 1.48]* | 222 / 1.33 [1.31, 1.34]* | 112 / 1.16 [1.14, 1.16]* | 57.1 / 0.91 [0.91, 0.92]* | 26.4 / 0.61 [0.60, 0.61]* | 16.6 / 0.40 [0.39, 0.40]* | 15.8 / 0.41 [0.39, 0.43] |
| ptr.it.chunk64 | 429 / 1.46 [1.45, 1.47] | 215 / 1.30 [1.27, 1.32] | 108 / 1.09 [1.03, 1.13] | 54.5 / 0.87 [0.87, 0.88]* | 25 / 0.55 [0.51, 0.57] | 13 / 0.31 [0.30, 0.32]* | 7.2 / 0.18 [0.17, 0.20] |
| ptr.it.chunk8 | 429 / 1.46 [1.43, 1.49] | 214 / 1.30 [1.26, 1.33] | 109 / 1.09 [1.04, 1.13] | 54.1 / 0.86 [0.86, 0.87]* | 24.4 / 0.53 [0.50, 0.56] | 12.2 / 0.29 [0.29, 0.30]* | 6.47 / 0.16 [0.15, 0.18] |
| ptr.it.col | 154 / 0.52 [0.52, 0.53] | 78.8 / 0.48 [0.47, 0.48] | 41.1 / 0.41 [0.40, 0.42] | 20.7 / 0.33 [0.33, 0.34]* | 10.1 / 0.22 [0.20, 0.23] | 5.52 / 0.13 [0.13, 0.14]* | 5.04 / 0.13 [0.11, 0.14] |
| ptr.it.col.alloc | 175 / 0.59 [0.59, 0.60]* | 93.3 / 0.56 [0.55, 0.57]* | 47.8 / 0.49 [0.49, 0.50]* | 25.5 / 0.41 [0.40, 0.41]* | 12.8 / 0.29 [0.29, 0.30]* | 9.96 / 0.24 [0.24, 0.27]* | 15 / 0.39 [0.37, 0.40] |
| ptr.it.col.view | 170 / 0.58 [0.57, 0.58]* | 94.1 / 0.56 [0.55, 0.57]* | 45.3 / 0.47 [0.46, 0.48]* | 24.3 / 0.39 [0.39, 0.39]* | 13.4 / 0.31 [0.30, 0.31]* | 9.79 / 0.23 [0.22, 0.25]* | 9.8 / 0.25 [0.25, 0.26] |
| ptr.it.col.viewnoinit | 175 / 0.59 [0.59, 0.60]* | 93.2 / 0.56 [0.55, 0.56]* | 47.8 / 0.49 [0.49, 0.51]* | 25.8 / 0.41 [0.41, 0.42]* | 13.2 / 0.30 [0.29, 0.31]* | 10.2 / 0.24 [0.24, 0.27]* | 14.9 / 0.38 [0.37, 0.40] |
| ptr.pw.col | 154 / 0.53 [0.52, 0.53] | 79.3 / 0.48 [0.47, 0.49] | 41.4 / 0.41 [0.40, 0.43] | 20.9 / 0.33 [0.33, 0.35]* | 10.2 / 0.22 [0.21, 0.25] | 5.88 / 0.14 [0.13, 0.16]* | 5.04 / 0.13 [0.11, 0.14] |
| ptr64.it.chunk16 | 125 / 0.42 [0.41, 0.44] | 67.3 / 0.41 [0.39, 0.43] | 34.8 / 0.35 [0.32, 0.38] | 18.8 / 0.30 [0.30, 0.31]* | 8.84 / 0.19 [0.18, 0.22] | 5.25 / 0.13 [0.12, 0.13]* | 4.84 / 0.13 [0.11, 0.13] |
| ptr64.it.col | 155 / 0.53 [0.52, 0.53]* | 79.4 / 0.47 [0.47, 0.48]* | 40.2 / 0.42 [0.41, 0.42]* | 20.8 / 0.33 [0.33, 0.34]* | 9.99 / 0.23 [0.22, 0.25]* | 6.1 / 0.15 [0.13, 0.16]* | 5.21 / 0.13 [0.13, 0.14] |
| vl.ex.chunk16 | 97.4 / 0.33 [0.33, 0.34] | 51.4 / 0.31 [0.30, 0.32] | 26.7 / 0.27 [0.25, 0.28] | 15 / 0.24 [0.24, 0.24]* | 8.5 / 0.19 [0.17, 0.20] | 5.92 / 0.14 [0.14, 0.14]* | 5.08 / 0.13 [0.11, 0.14] |
| vl.it.chunk16 | 272 / 0.92 [0.92, 0.93]* | 135 / 0.81 [0.80, 0.82]* | 68.7 / 0.71 [0.70, 0.71]* | 34.6 / 0.55 [0.55, 0.56]* | 15.6 / 0.36 [0.35, 0.36]* | 8.23 / 0.20 [0.19, 0.22]* | 5.29 / 0.14 [0.13, 0.14] |
| vl.it.col | 154 / 0.52 [0.52, 0.53]* | 79 / 0.47 [0.47, 0.48]* | 39.8 / 0.41 [0.41, 0.42]* | 21 / 0.34 [0.33, 0.34]* | 9.73 / 0.22 [0.22, 0.24]* | 5.88 / 0.14 [0.13, 0.15]* | 5.25 / 0.13 [0.13, 0.14] |
| vs.it.chunk16 | 270 / 0.92 [0.91, 0.92]* | 135 / 0.81 [0.80, 0.81]* | 72.7 / 0.75 [0.74, 0.76]* | 35.7 / 0.57 [0.57, 0.57]* | 15.3 / 0.35 [0.35, 0.36]* | 8.33 / 0.20 [0.20, 0.20]* | 5.22 / 0.13 [0.13, 0.14] |

Same cells, ratio to `gtfn_fixed`:

| variant | 1 thr | 2 thr | 4 thr | 8 thr | 18 thr | 36 thr | 72 thr |
|---|---|---|---|---|---|---|---|
| naive | - | - | 1.39 [1.39, 1.40]* | 1.39 [1.38, 1.40]* | 1.48 [1.47, 1.50]* | 1.42 [1.40, 1.43]* | 1.44 [1.38, 1.49] |
| ptr.ex.chunk16 | 0.39 [0.39, 0.40] | 0.40 [0.40, 0.40] | 0.41 [0.40, 0.41] | 0.42 [0.41, 0.42]* | 0.48 [0.45, 0.49] | 0.43 [0.43, 0.44]* | 0.34 [0.31, 0.36] |
| ptr.ex.chunk32 | 0.27 [0.27, 0.28]* | 0.27 [0.26, 0.27]* | 0.28 [0.28, 0.28]* | - | 0.34 [0.32, 0.35]* | - | 0.34 [0.34, 0.35]* |
| ptr.ex.chunk64 | 0.33 [0.33, 0.33]* | 0.32 [0.32, 0.33]* | 0.33 [0.33, 0.33]* | - | 0.44 [0.41, 0.45]* | - | 0.36 [0.35, 0.36]* |
| ptr.ex.chunk8 | 0.31 [0.31, 0.31]* | 0.30 [0.30, 0.30]* | 0.32 [0.32, 0.33]* | - | 0.40 [0.39, 0.42]* | - | 0.34 [0.34, 0.34]* |
| ptr.ex.col | 0.61 [0.61, 0.61] | 0.60 [0.59, 0.62] | 0.62 [0.60, 0.66] | 0.57 [0.56, 0.58]* | 0.55 [0.54, 0.60] | 0.45 [0.41, 0.48]* | 0.33 [0.31, 0.35] |
| ptr.it.chunk16 | 1.65 [1.64, 1.67] | 1.62 [1.60, 1.64] | 1.62 [1.60, 1.63] | 1.48 [1.47, 1.48]* | 1.34 [1.29, 1.36] | 0.89 [0.87, 0.89]* | 0.40 [0.39, 0.50] |
| ptr.it.chunk16.alloc | 1.74 [1.74, 1.74]* | 1.70 [1.69, 1.70]* | 1.69 [1.68, 1.70]* | 1.56 [1.56, 1.57]* | 1.50 [1.48, 1.51]* | 1.23 [1.21, 1.24]* | 1.01 [0.98, 1.04] |
| ptr.it.chunk16.view | 1.70 [1.70, 1.70]* | 1.76 [1.75, 1.76]* | 1.69 [1.69, 1.70]* | 1.58 [1.58, 1.59]* | 1.57 [1.55, 1.59]* | 1.18 [1.17, 1.23]* | 0.66 [0.65, 0.71] |
| ptr.it.chunk16.viewnoinit | 1.71 [1.71, 1.71]* | 1.71 [1.71, 1.71]* | 1.69 [1.68, 1.70]* | 1.57 [1.57, 1.58]* | 1.50 [1.48, 1.50]* | 1.22 [1.20, 1.23]* | 0.98 [0.97, 1.02] |
| ptr.it.chunk64 | 1.69 [1.68, 1.69] | 1.65 [1.63, 1.66] | 1.64 [1.63, 1.65] | 1.50 [1.50, 1.51]* | 1.39 [1.33, 1.41] | 0.96 [0.93, 0.96]* | 0.45 [0.44, 0.54] |
| ptr.it.chunk8 | 1.69 [1.67, 1.71] | 1.64 [1.64, 1.65] | 1.65 [1.63, 1.67] | 1.49 [1.49, 1.50]* | 1.36 [1.30, 1.38] | 0.90 [0.88, 0.93]* | 0.40 [0.39, 0.49] |
| ptr.it.col | 0.61 [0.60, 0.61] | 0.60 [0.59, 0.62] | 0.62 [0.60, 0.65] | 0.57 [0.56, 0.58]* | 0.56 [0.52, 0.60] | 0.41 [0.39, 0.43]* | 0.34 [0.31, 0.35] |
| ptr.it.col.alloc | 0.69 [0.69, 0.69]* | 0.72 [0.72, 0.73]* | 0.72 [0.71, 0.73]* | 0.70 [0.70, 0.71]* | 0.72 [0.71, 0.74]* | 0.73 [0.72, 0.81]* | 0.93 [0.92, 0.94] |
| ptr.it.col.view | 0.67 [0.67, 0.67]* | 0.73 [0.72, 0.73]* | 0.68 [0.68, 0.69]* | 0.67 [0.66, 0.67]* | 0.76 [0.74, 0.77]* | 0.72 [0.69, 0.76]* | 0.61 [0.59, 0.63] |
| ptr.it.col.viewnoinit | 0.69 [0.68, 0.70]* | 0.72 [0.71, 0.72]* | 0.72 [0.72, 0.74]* | 0.71 [0.70, 0.73]* | 0.74 [0.72, 0.76]* | 0.75 [0.72, 0.81]* | 0.93 [0.92, 0.94] |
| ptr.pw.col | 0.61 [0.60, 0.61] | 0.61 [0.60, 0.62] | 0.63 [0.60, 0.65] | 0.58 [0.57, 0.60]* | 0.56 [0.54, 0.61] | 0.43 [0.39, 0.48]* | 0.33 [0.31, 0.34] |
| ptr64.it.chunk16 | 0.49 [0.47, 0.51] | 0.52 [0.49, 0.55] | 0.53 [0.51, 0.55] | 0.52 [0.51, 0.53]* | 0.49 [0.45, 0.53] | 0.39 [0.38, 0.40]* | 0.32 [0.30, 0.36] |
| ptr64.it.col | 0.61 [0.61, 0.61]* | 0.61 [0.61, 0.61]* | 0.61 [0.60, 0.61]* | 0.57 [0.56, 0.59]* | 0.56 [0.54, 0.61]* | 0.45 [0.39, 0.48]* | 0.32 [0.31, 0.34] |
| vl.ex.chunk16 | 0.38 [0.38, 0.39] | 0.39 [0.39, 0.40] | 0.40 [0.39, 0.41] | 0.41 [0.41, 0.42]* | 0.47 [0.44, 0.49] | 0.43 [0.43, 0.44]* | 0.34 [0.31, 0.36] |
| vl.it.chunk16 | 1.07 [1.07, 1.07]* | 1.04 [1.04, 1.05]* | 1.04 [1.03, 1.04]* | 0.95 [0.95, 0.96]* | 0.88 [0.87, 0.89]* | 0.60 [0.59, 0.67]* | 0.33 [0.31, 0.34] |
| vl.it.col | 0.61 [0.61, 0.61]* | 0.61 [0.61, 0.61]* | 0.60 [0.60, 0.61]* | 0.58 [0.57, 0.58]* | 0.55 [0.54, 0.58]* | 0.43 [0.39, 0.47]* | 0.33 [0.31, 0.34] |
| vs.it.chunk16 | 1.06 [1.06, 1.06]* | 1.04 [1.04, 1.04]* | 1.10 [1.09, 1.10]* | 0.98 [0.98, 0.99]* | 0.86 [0.85, 0.88]* | 0.61 [0.60, 0.62]* | 0.32 [0.31, 0.34] |

#### `v2e, local numbering`: GPU (ms / ratio to gtfn)

| variant | large | xlarge |
|---|---|---|
| gtfn | 0.243 / 1.00 [1.00, 1.00] | 1.55 / 1.00 [1.00, 1.00] |
| gtfn_us | 0.217 / 0.89 [0.89, 0.90] | 1.37 / 0.88 [0.88, 0.89] |
| naive | 0.176 / 0.72 [0.72, 0.73] | 1.49 / 0.96 [0.95, 0.97] |
| ptr.ex.mddef | 0.134 / 0.54 [0.54, 0.55]* | 0.911 / 0.59 [0.58, 0.59] |
| ptr.ex.mdt | 0.14 / 0.58 [0.57, 0.58] | 0.884 / 0.56 [0.56, 0.57]* |
| ptr.ex.mdtlb | 0.14 / 0.58 [0.57, 0.58] | 0.884 / 0.57 [0.56, 0.58] |
| ptr.ex2.mdtlb | 0.14 / 0.57 [0.57, 0.58]* | 0.883 / 0.56 [0.56, 0.57]* |
| ptr.it.mddef | 0.172 / 0.71 [0.70, 0.72] | 1.47 / 0.94 [0.93, 0.95] |
| ptr.it.mddefint | 0.173 / 0.71 [0.70, 0.72] | 1.47 / 0.94 [0.93, 0.95]* |
| ptr.it.mdt | 0.225 / 0.92 [0.91, 0.93] | 1.45 / 0.93 [0.92, 0.93]* |
| ptr.it.mdtlb | 0.224 / 0.92 [0.91, 0.93] | 1.45 / 0.93 [0.92, 0.94] |
| ptr.it.mdtlb+kfence | 0.225 / 0.92 [0.91, 0.94] | 1.45 / 0.93 [0.92, 0.94] |
| ptr.it.range | 0.179 / 0.73 [0.73, 0.75] | 1.68 / 1.08 [1.07, 1.09] |
| ptr.it1.mddef | 0.137 / 0.56 [0.56, 0.57]* | 0.933 / 0.60 [0.59, 0.60]* |
| ptr.it1.mdtlb | 0.145 / 0.59 [0.58, 0.60]* | 0.904 / 0.58 [0.57, 0.58]* |
| ptr64.it.mdtlb | 0.225 / 0.93 [0.92, 0.94] | 1.47 / 0.94 [0.93, 0.94]* |
| vl.ex.mdtlb | 0.14 / 0.58 [0.57, 0.58] | 0.884 / 0.57 [0.56, 0.57]* |
| vl.it.mdtlb | 0.211 / 0.87 [0.86, 0.88] | 1.36 / 0.87 [0.87, 0.88]* |
| vs.it.mdtlb | 0.229 / 0.94 [0.93, 0.95] | 1.48 / 0.95 [0.94, 0.95]* |

#### `v2e, local numbering`: CPU, large (ms / ratio to gtfn)

| variant | 1 thr | 2 thr | 4 thr | 8 thr | 18 thr | 36 thr | 72 thr |
|---|---|---|---|---|---|---|---|
| gtfn | 64.8 / 1.00 [1.00, 1.00]* | 32.5 / 1.00 [1.00, 1.00]* | 16.4 / 1.00 [1.00, 1.00]* | 8.18 / 1.00 [1.00, 1.00]* | 3.71 / 1.00 [1.00, 1.00]* | 1.95 / 1.00 [1.00, 1.00]* | 1.13 / 1.00 [1.00, 1.00] |
| gtfn_us | 54.8 / 0.85 [0.84, 0.85]* | 27.5 / 0.85 [0.84, 0.85]* | 13.9 / 0.85 [0.85, 0.85]* | 6.98 / 0.85 [0.84, 0.86]* | 3.17 / 0.85 [0.85, 0.86]* | 1.66 / 0.85 [0.84, 0.87]* | 1.01 / 0.89 [0.87, 0.92] |
| naive | - | - | 31.5 / 1.93 [1.91, 1.93]* | 16.1 / 1.96 [1.95, 1.97]* | 7.4 / 1.99 [1.98, 2.01]* | 4.06 / 2.08 [2.05, 2.13]* | 2.58 / 2.28 [2.24, 2.33] |
| ptr.ex.mdleft | 56.7 / 0.88 [0.87, 0.88]* | 28.5 / 0.87 [0.87, 0.88]* | 14.3 / 0.87 [0.87, 0.88]* | 7.15 / 0.87 [0.87, 0.88]* | 4.35 / 1.17 [1.16, 1.18]* | 2.93 / 1.50 [1.48, 1.54]* | 1.53 / 1.36 [1.32, 1.39] |
| ptr.ex.rows | 41 / 0.63 [0.63, 0.63]* | 20.7 / 0.64 [0.63, 0.64]* | 10.5 / 0.64 [0.64, 0.64]* | 5.25 / 0.64 [0.64, 0.64]* | 2.41 / 0.65 [0.64, 0.66]* | 1.4 / 0.72 [0.67, 0.74]* | 0.881 / 0.78 [0.76, 0.81] |
| ptr.ex.rowsnh | 53.3 / 0.82 [0.82, 0.83]* | 26.8 / 0.82 [0.82, 0.83]* | 13.6 / 0.83 [0.82, 0.83]* | 6.79 / 0.83 [0.83, 0.83]* | 3.08 / 0.83 [0.83, 0.84]* | 1.61 / 0.83 [0.81, 0.85]* | 0.987 / 0.87 [0.85, 0.89] |
| ptr.ex2.rows | 40.9 / 0.63 [0.63, 0.64]* | 20.6 / 0.63 [0.63, 0.64]* | 10.5 / 0.64 [0.64, 0.65]* | 5.24 / 0.64 [0.64, 0.64]* | 2.4 / 0.65 [0.64, 0.65]* | 1.33 / 0.68 [0.66, 0.71]* | 0.881 / 0.78 [0.77, 0.79] |
| ptr.it.mddef | - | - | 30.9 / 1.88 [1.87, 1.90]* | 15.7 / 1.92 [1.91, 1.93]* | 7.25 / 1.95 [1.94, 1.97]* | 4 / 2.05 [2.02, 2.10]* | 2.57 / 2.27 [2.23, 2.31] |
| ptr.it.mdleft | 53.2 / 0.82 [0.82, 0.82]* | 26.6 / 0.82 [0.82, 0.82]* | 13.4 / 0.82 [0.81, 0.82]* | 6.7 / 0.82 [0.82, 0.82]* | 4.07 / 1.10 [1.09, 1.11]* | 2.78 / 1.42 [1.39, 1.49]* | 1.46 / 1.29 [1.26, 1.33] |
| ptr.it.rows | 45.1 / 0.70 [0.70, 0.70]* | 22.7 / 0.70 [0.69, 0.70]* | 11.4 / 0.70 [0.70, 0.70]* | 5.74 / 0.70 [0.70, 0.71]* | 2.64 / 0.71 [0.70, 0.72]* | 1.46 / 0.75 [0.72, 0.80]* | 0.925 / 0.82 [0.80, 0.83] |
| ptr.it.rowsnh | 51.2 / 0.79 [0.79, 0.79]* | 25.8 / 0.79 [0.79, 0.79]* | 13 / 0.79 [0.79, 0.80]* | 6.46 / 0.79 [0.79, 0.80]* | 2.98 / 0.80 [0.79, 0.82]* | 1.6 / 0.82 [0.81, 0.85]* | 0.988 / 0.87 [0.85, 0.91] |
| ptr.it1.rows | 154 / 2.38 [2.38, 2.39]* | 75.7 / 2.33 [2.32, 2.33]* | 38 / 2.32 [2.31, 2.32]* | 19 / 2.33 [2.32, 2.34]* | 8.55 / 2.30 [2.29, 2.32]* | 4.42 / 2.27 [2.22, 2.33]* | 2.29 / 2.03 [1.96, 2.08] |
| ptr64.it.rows | 45.5 / 0.70 [0.70, 0.70]* | 22.8 / 0.70 [0.70, 0.70]* | 11.5 / 0.70 [0.70, 0.70]* | 5.78 / 0.71 [0.70, 0.71]* | 2.64 / 0.71 [0.71, 0.72]* | 1.44 / 0.74 [0.72, 0.77]* | 0.925 / 0.82 [0.80, 0.83] |
| vl.ex.rows | 40.9 / 0.63 [0.63, 0.63]* | 20.7 / 0.64 [0.63, 0.64]* | 10.6 / 0.65 [0.64, 0.65]* | 5.26 / 0.64 [0.64, 0.65]* | 2.42 / 0.65 [0.64, 0.66]* | 1.33 / 0.68 [0.66, 0.74]* | 0.886 / 0.78 [0.76, 0.81] |
| vl.it.rows | 45.3 / 0.70 [0.70, 0.70]* | 22.7 / 0.70 [0.69, 0.70]* | 11.5 / 0.70 [0.70, 0.70]* | 5.73 / 0.70 [0.70, 0.71]* | 2.67 / 0.72 [0.71, 0.72]* | 1.42 / 0.73 [0.72, 0.80]* | 0.929 / 0.82 [0.81, 0.84] |
| vs.it.rows | 45 / 0.69 [0.69, 0.70]* | 22.6 / 0.69 [0.69, 0.70]* | 11.4 / 0.70 [0.69, 0.71]* | 5.72 / 0.70 [0.69, 0.70]* | 2.62 / 0.71 [0.70, 0.71]* | 1.49 / 0.76 [0.72, 0.78]* | 0.924 / 0.82 [0.80, 0.85] |

Same cells, ratio to `gtfn_us`:

| variant | 1 thr | 2 thr | 4 thr | 8 thr | 18 thr | 36 thr | 72 thr |
|---|---|---|---|---|---|---|---|
| naive | - | - | 2.27 [2.25, 2.28]* | 2.30 [2.29, 2.32]* | 2.34 [2.33, 2.35]* | 2.45 [2.42, 2.47]* | 2.56 [2.50, 2.62] |
| ptr.ex.mdleft | 1.03 [1.03, 1.04]* | 1.03 [1.03, 1.04]* | 1.03 [1.02, 1.03]* | 1.03 [1.02, 1.04]* | 1.37 [1.36, 1.39]* | 1.76 [1.75, 1.78]* | 1.52 [1.48, 1.56] |
| ptr.ex.rows | 0.75 [0.74, 0.75]* | 0.75 [0.75, 0.75]* | 0.76 [0.75, 0.76]* | 0.75 [0.75, 0.76]* | 0.76 [0.76, 0.77]* | 0.84 [0.79, 0.86]* | 0.87 [0.85, 0.91] |
| ptr.ex.rowsnh | 0.97 [0.97, 0.98]* | 0.97 [0.97, 0.98]* | 0.98 [0.97, 0.98]* | 0.97 [0.97, 0.98]* | 0.97 [0.97, 0.98]* | 0.97 [0.95, 0.99]* | 0.98 [0.95, 1.00] |
| ptr.ex2.rows | 0.75 [0.74, 0.75]* | 0.75 [0.74, 0.75]* | 0.75 [0.75, 0.76]* | 0.75 [0.75, 0.76]* | 0.76 [0.75, 0.77]* | 0.80 [0.78, 0.82]* | 0.87 [0.86, 0.89] |
| ptr.it.mddef | - | - | 2.22 [2.20, 2.23]* | 2.25 [2.24, 2.27]* | 2.29 [2.28, 2.30]* | 2.41 [2.39, 2.43]* | 2.55 [2.48, 2.61] |
| ptr.it.mdleft | 0.97 [0.96, 0.97]* | 0.97 [0.96, 0.97]* | 0.96 [0.96, 0.97]* | 0.96 [0.95, 0.97]* | 1.29 [1.28, 1.30]* | 1.67 [1.64, 1.72]* | 1.45 [1.40, 1.48] |
| ptr.it.rows | 0.82 [0.82, 0.82]* | 0.82 [0.82, 0.82]* | 0.82 [0.82, 0.83]* | 0.82 [0.81, 0.83]* | 0.83 [0.83, 0.84]* | 0.88 [0.85, 0.94]* | 0.92 [0.89, 0.94] |
| ptr.it.rowsnh | 0.93 [0.93, 0.94]* | 0.93 [0.93, 0.94]* | 0.93 [0.93, 0.94]* | 0.93 [0.92, 0.93]* | 0.94 [0.93, 0.96]* | 0.96 [0.95, 1.00]* | 0.98 [0.95, 1.02] |
| ptr.it1.rows | 2.81 [2.81, 2.82]* | 2.75 [2.74, 2.75]* | 2.73 [2.72, 2.74]* | 2.73 [2.71, 2.75]* | 2.70 [2.69, 2.72]* | 2.66 [2.62, 2.70]* | 2.27 [2.19, 2.34] |
| ptr64.it.rows | 0.83 [0.83, 0.83]* | 0.83 [0.82, 0.83]* | 0.82 [0.82, 0.83]* | 0.83 [0.82, 0.84]* | 0.83 [0.83, 0.84]* | 0.87 [0.84, 0.91]* | 0.92 [0.89, 0.93] |
| vl.ex.rows | 0.75 [0.74, 0.75]* | 0.75 [0.75, 0.75]* | 0.76 [0.75, 0.76]* | 0.75 [0.75, 0.76]* | 0.76 [0.75, 0.77]* | 0.80 [0.78, 0.86]* | 0.88 [0.85, 0.92] |
| vl.it.rows | 0.83 [0.82, 0.83]* | 0.82 [0.82, 0.83]* | 0.82 [0.82, 0.83]* | 0.82 [0.82, 0.83]* | 0.84 [0.83, 0.85]* | 0.86 [0.84, 0.93]* | 0.92 [0.90, 0.95] |
| vs.it.rows | 0.82 [0.82, 0.82]* | 0.82 [0.82, 0.82]* | 0.82 [0.82, 0.83]* | 0.82 [0.81, 0.83]* | 0.83 [0.82, 0.84]* | 0.90 [0.84, 0.91]* | 0.92 [0.89, 0.96] |

GPU, ratio to `gtfn_us`:

| variant | large | xlarge |
|---|---|---|
| naive | 0.81 [0.80, 0.82] | 1.08 [1.07, 1.09] |
| ptr.ex.mddef | 0.61 [0.61, 0.61]* | 0.66 [0.65, 0.67] |
| ptr.ex.mdt | 0.65 [0.64, 0.65] | 0.64 [0.63, 0.64]* |
| ptr.ex.mdtlb | 0.64 [0.64, 0.65] | 0.64 [0.63, 0.65] |
| ptr.ex2.mdtlb | 0.64 [0.64, 0.64]* | 0.64 [0.63, 0.64]* |
| ptr.it.mddef | 0.79 [0.78, 0.80] | 1.07 [1.06, 1.08] |
| ptr.it.mddefint | 0.79 [0.79, 0.80] | 1.06 [1.05, 1.07]* |
| ptr.it.mdt | 1.03 [1.02, 1.04] | 1.05 [1.04, 1.06]* |
| ptr.it.mdtlb | 1.03 [1.02, 1.04] | 1.05 [1.04, 1.06] |
| ptr.it.mdtlb+kfence | 1.03 [1.02, 1.04] | 1.06 [1.04, 1.06] |
| ptr.it.range | 0.82 [0.81, 0.83] | 1.23 [1.21, 1.24] |
| ptr.it1.mddef | 0.63 [0.62, 0.63]* | 0.67 [0.67, 0.68]* |
| ptr.it1.mdtlb | 0.66 [0.66, 0.67]* | 0.65 [0.65, 0.66]* |
| ptr64.it.mdtlb | 1.04 [1.03, 1.05] | 1.06 [1.05, 1.07]* |
| vl.ex.mdtlb | 0.64 [0.64, 0.65] | 0.64 [0.63, 0.64]* |
| vl.it.mdtlb | 0.97 [0.96, 0.98] | 0.99 [0.98, 0.99]* |
| vs.it.mdtlb | 1.05 [1.04, 1.06] | 1.07 [1.06, 1.08]* |

#### `v2e, random numbering`: GPU (ms / ratio to gtfn)

| variant | large | xlarge |
|---|---|---|
| gtfn | 0.606 / 1.00 [1.00, 1.00] | 11.5 / 1.00 [1.00, 1.00] |
| gtfn_us | 0.604 / 1.00 [0.99, 1.00] | 11.5 / 1.00 [1.00, 1.00] |
| naive | 0.523 / 0.86 [0.86, 0.87] | 9.55 / 0.83 [0.83, 0.83] |
| ptr.ex.mddef | 0.521 / 0.86 [0.85, 0.86]* | 9.52 / 0.83 [0.83, 0.83] |
| ptr.ex.mdt | 0.582 / 0.96 [0.96, 0.96] | 11.5 / 1.00 [1.00, 1.00]* |
| ptr.ex.mdtlb | 0.582 / 0.96 [0.96, 0.96] | 11.5 / 1.00 [1.00, 1.00] |
| ptr.ex2.mdtlb | 0.585 / 0.96 [0.96, 0.96]* | 11.5 / 1.00 [1.00, 1.00]* |
| ptr.it.mddef | 0.522 / 0.86 [0.86, 0.87] | 9.55 / 0.83 [0.83, 0.83] |
| ptr.it.mddefint | 0.522 / 0.86 [0.86, 0.87] | 9.55 / 0.83 [0.83, 0.83]* |
| ptr.it.mdt | 0.602 / 0.99 [0.99, 1.00] | 11.5 / 1.00 [1.00, 1.00]* |
| ptr.it.mdtlb | 0.602 / 0.99 [0.99, 1.00] | 11.5 / 1.00 [1.00, 1.00] |
| ptr.it.mdtlb+kfence | 0.602 / 0.99 [0.99, 1.00] | 11.5 / 1.00 [1.00, 1.00] |
| ptr.it.range | 0.53 / 0.88 [0.87, 0.88] | 3.96 / 0.35 [0.34, 0.35] |
| ptr.it1.mddef | 0.521 / 0.86 [0.86, 0.86]* | 9.52 / 0.83 [0.83, 0.83]* |
| ptr.it1.mdtlb | 0.584 / 0.96 [0.96, 0.96]* | 11.5 / 1.00 [1.00, 1.00]* |
| ptr64.it.mdtlb | 0.602 / 0.99 [0.99, 1.00] | 11.5 / 1.00 [1.00, 1.00]* |
| vl.ex.mdtlb | 0.582 / 0.96 [0.96, 0.96] | 11.5 / 1.00 [1.00, 1.00]* |
| vl.it.mdtlb | 0.6 / 0.99 [0.99, 0.99] | 11.5 / 1.00 [1.00, 1.00]* |
| vs.it.mdtlb | 0.602 / 0.99 [0.99, 1.00] | 11.5 / 1.00 [1.00, 1.00]* |

#### `v2e, random numbering`: CPU, large (ms / ratio to gtfn)

| variant | 1 thr | 2 thr | 4 thr | 8 thr | 18 thr | 36 thr | 72 thr |
|---|---|---|---|---|---|---|---|
| gtfn | 234 / 1.00 [1.00, 1.00]* | 117 / 1.00 [1.00, 1.00]* | 59.8 / 1.00 [1.00, 1.00]* | 32.1 / 1.00 [1.00, 1.00]* | 16.9 / 1.00 [1.00, 1.00]* | 14.3 / 1.00 [1.00, 1.00]* | 12.3 / 1.00 [1.00, 1.00] |
| gtfn_us | 201 / 0.86 [0.86, 0.86]* | 101 / 0.86 [0.86, 0.86]* | 51.3 / 0.86 [0.85, 0.87]* | 27.2 / 0.85 [0.84, 0.85]* | 14 / 0.83 [0.83, 0.84]* | 11.1 / 0.78 [0.77, 0.78]* | 10.7 / 0.87 [0.86, 0.88] |
| naive | - | - | 73.1 / 1.22 [1.21, 1.23]* | 37.9 / 1.18 [1.17, 1.19]* | 18.9 / 1.12 [1.12, 1.13]* | 12.5 / 0.87 [0.87, 0.88]* | 8.77 / 0.71 [0.70, 0.73] |
| ptr.ex.mdleft | 203 / 0.87 [0.87, 0.87]* | 101 / 0.86 [0.86, 0.87]* | 51.5 / 0.86 [0.85, 0.87]* | 27.4 / 0.85 [0.85, 0.86]* | 17.7 / 1.05 [1.05, 1.06]* | 15.3 / 1.07 [1.06, 1.07]* | 9.98 / 0.81 [0.80, 0.82] |
| ptr.ex.rows | 191 / 0.82 [0.81, 0.82]* | 95 / 0.81 [0.81, 0.81]* | 48.2 / 0.81 [0.80, 0.81]* | 25.5 / 0.79 [0.79, 0.81]* | 12.9 / 0.76 [0.76, 0.77]* | 9.66 / 0.67 [0.67, 0.68]* | 9.85 / 0.80 [0.78, 0.81] |
| ptr.ex.rowsnh | 198 / 0.85 [0.85, 0.85]* | 98.9 / 0.84 [0.84, 0.85]* | 50.4 / 0.84 [0.83, 0.85]* | 26.6 / 0.83 [0.83, 0.84]* | 13.7 / 0.81 [0.81, 0.82]* | 10.6 / 0.74 [0.74, 0.75]* | 10.5 / 0.85 [0.84, 0.86] |
| ptr.ex2.rows | 191 / 0.82 [0.81, 0.82]* | 95 / 0.81 [0.81, 0.81]* | 48.2 / 0.81 [0.80, 0.81]* | 25.5 / 0.79 [0.79, 0.80]* | 12.9 / 0.77 [0.76, 0.77]* | 9.67 / 0.68 [0.67, 0.68]* | 9.83 / 0.80 [0.79, 0.81] |
| ptr.it.mddef | - | - | 73.4 / 1.23 [1.21, 1.24]* | 38 / 1.18 [1.18, 1.20]* | 19 / 1.13 [1.13, 1.14]* | 12.6 / 0.88 [0.87, 0.88]* | 8.77 / 0.71 [0.70, 0.73] |
| ptr.it.mdleft | 196 / 0.84 [0.83, 0.84]* | 97.4 / 0.83 [0.83, 0.83]* | 49.4 / 0.83 [0.82, 0.83]* | 26.1 / 0.81 [0.81, 0.82]* | 16.8 / 1.00 [0.99, 1.01]* | 14.1 / 0.98 [0.98, 0.99]* | 9.12 / 0.74 [0.73, 0.75] |
| ptr.it.rows | 195 / 0.83 [0.83, 0.83]* | 97.2 / 0.83 [0.83, 0.83]* | 49.5 / 0.83 [0.82, 0.83]* | 26.1 / 0.81 [0.81, 0.82]* | 13.3 / 0.79 [0.79, 0.80]* | 10.3 / 0.72 [0.71, 0.72]* | 10.1 / 0.82 [0.81, 0.83] |
| ptr.it.rowsnh | 195 / 0.83 [0.83, 0.83]* | 97.4 / 0.83 [0.83, 0.83]* | 49.6 / 0.83 [0.82, 0.84]* | 26.3 / 0.82 [0.81, 0.83]* | 13.4 / 0.79 [0.79, 0.80]* | 10.3 / 0.72 [0.71, 0.72]* | 10.1 / 0.82 [0.81, 0.84] |
| ptr.it1.rows | 643 / 2.75 [2.75, 2.75]* | 323 / 2.75 [2.75, 2.76]* | 163 / 2.73 [2.70, 2.74]* | 83.5 / 2.60 [2.59, 2.62]* | 42.8 / 2.54 [2.52, 2.55]* | 35.5 / 2.48 [2.45, 2.50]* | 24.4 / 1.98 [1.96, 2.00] |
| ptr64.it.rows | 194 / 0.83 [0.83, 0.83]* | 96.9 / 0.83 [0.83, 0.83]* | 49.2 / 0.82 [0.81, 0.83]* | 26.1 / 0.81 [0.81, 0.82]* | 13.3 / 0.79 [0.78, 0.80]* | 10.1 / 0.71 [0.70, 0.71]* | 10.1 / 0.82 [0.80, 0.83] |
| vl.ex.rows | 191 / 0.82 [0.82, 0.82]* | 95.1 / 0.81 [0.81, 0.81]* | 48.2 / 0.81 [0.80, 0.81]* | 25.4 / 0.79 [0.79, 0.80]* | 12.9 / 0.76 [0.76, 0.77]* | 9.69 / 0.68 [0.67, 0.68]* | 9.77 / 0.79 [0.79, 0.81] |
| vl.it.rows | 195 / 0.83 [0.83, 0.83]* | 97.1 / 0.83 [0.83, 0.83]* | 49.4 / 0.83 [0.82, 0.83]* | 26.2 / 0.82 [0.81, 0.82]* | 13.4 / 0.79 [0.79, 0.80]* | 10.2 / 0.71 [0.71, 0.72]* | 10.1 / 0.82 [0.81, 0.84] |
| vs.it.rows | 195 / 0.83 [0.83, 0.83]* | 97.2 / 0.83 [0.83, 0.83]* | 49.5 / 0.83 [0.82, 0.83]* | 26.1 / 0.81 [0.81, 0.82]* | 13.4 / 0.79 [0.79, 0.80]* | 10.3 / 0.72 [0.71, 0.72]* | 10.2 / 0.83 [0.82, 0.84] |

Same cells, ratio to `gtfn_us`:

| variant | 1 thr | 2 thr | 4 thr | 8 thr | 18 thr | 36 thr | 72 thr |
|---|---|---|---|---|---|---|---|
| naive | - | - | 1.42 [1.42, 1.43]* | 1.39 [1.39, 1.40]* | 1.35 [1.35, 1.35]* | 1.12 [1.12, 1.13]* | 0.82 [0.81, 0.84] |
| ptr.ex.mdleft | 1.01 [1.01, 1.01]* | 1.01 [1.01, 1.01]* | 1.00 [1.00, 1.01]* | 1.01 [1.00, 1.01]* | 1.26 [1.26, 1.27]* | 1.37 [1.36, 1.38]* | 0.94 [0.93, 0.95] |
| ptr.ex.rows | 0.95 [0.95, 0.95]* | 0.95 [0.94, 0.95]* | 0.94 [0.93, 0.95]* | 0.94 [0.93, 0.95]* | 0.92 [0.91, 0.92]* | 0.87 [0.86, 0.87]* | 0.92 [0.91, 0.93] |
| ptr.ex.rowsnh | 0.99 [0.99, 0.99]* | 0.98 [0.98, 0.99]* | 0.98 [0.98, 0.99]* | 0.98 [0.98, 0.99]* | 0.98 [0.97, 0.98]* | 0.96 [0.95, 0.96]* | 0.98 [0.97, 0.99] |
| ptr.ex2.rows | 0.95 [0.95, 0.95]* | 0.95 [0.94, 0.95]* | 0.94 [0.93, 0.94]* | 0.94 [0.93, 0.94]* | 0.92 [0.92, 0.93]* | 0.87 [0.86, 0.87]* | 0.92 [0.91, 0.93] |
| ptr.it.mddef | - | - | 1.43 [1.42, 1.44]* | 1.40 [1.39, 1.40]* | 1.36 [1.35, 1.36]* | 1.13 [1.12, 1.13]* | 0.82 [0.81, 0.84] |
| ptr.it.mdleft | 0.98 [0.97, 0.98]* | 0.97 [0.97, 0.97]* | 0.96 [0.96, 0.97]* | 0.96 [0.96, 0.97]* | 1.20 [1.20, 1.20]* | 1.27 [1.26, 1.27]* | 0.86 [0.84, 0.87] |
| ptr.it.rows | 0.97 [0.97, 0.97]* | 0.97 [0.97, 0.97]* | 0.96 [0.96, 0.97]* | 0.96 [0.95, 0.97]* | 0.95 [0.95, 0.96]* | 0.93 [0.92, 0.93]* | 0.95 [0.94, 0.96] |
| ptr.it.rowsnh | 0.97 [0.97, 0.97]* | 0.97 [0.97, 0.97]* | 0.97 [0.96, 0.97]* | 0.97 [0.96, 0.97]* | 0.96 [0.95, 0.96]* | 0.92 [0.92, 0.93]* | 0.95 [0.94, 0.97] |
| ptr.it1.rows | 3.21 [3.20, 3.21]* | 3.21 [3.20, 3.21]* | 3.18 [3.16, 3.20]* | 3.07 [3.06, 3.08]* | 3.05 [3.03, 3.06]* | 3.19 [3.16, 3.21]* | 2.29 [2.26, 2.31] |
| ptr64.it.rows | 0.97 [0.97, 0.97]* | 0.96 [0.96, 0.97]* | 0.96 [0.95, 0.96]* | 0.96 [0.96, 0.97]* | 0.95 [0.94, 0.95]* | 0.91 [0.90, 0.92]* | 0.95 [0.93, 0.96] |
| vl.ex.rows | 0.95 [0.95, 0.95]* | 0.95 [0.94, 0.95]* | 0.94 [0.93, 0.95]* | 0.94 [0.93, 0.94]* | 0.92 [0.91, 0.92]* | 0.87 [0.87, 0.88]* | 0.92 [0.91, 0.93] |
| vl.it.rows | 0.97 [0.97, 0.97]* | 0.97 [0.97, 0.97]* | 0.96 [0.96, 0.97]* | 0.96 [0.96, 0.97]* | 0.95 [0.95, 0.95]* | 0.92 [0.91, 0.92]* | 0.95 [0.94, 0.96] |
| vs.it.rows | 0.97 [0.97, 0.97]* | 0.97 [0.97, 0.97]* | 0.96 [0.96, 0.97]* | 0.96 [0.96, 0.97]* | 0.95 [0.95, 0.96]* | 0.93 [0.92, 0.93]* | 0.95 [0.95, 0.97] |

GPU, ratio to `gtfn_us`:

| variant | large | xlarge |
|---|---|---|
| naive | 0.87 [0.86, 0.87] | 0.83 [0.83, 0.83] |
| ptr.ex.mddef | 0.86 [0.86, 0.86]* | 0.83 [0.83, 0.83] |
| ptr.ex.mdt | 0.96 [0.96, 0.96] | 1.00 [1.00, 1.00]* |
| ptr.ex.mdtlb | 0.96 [0.96, 0.97] | 1.00 [1.00, 1.00] |
| ptr.ex2.mdtlb | 0.96 [0.96, 0.96]* | 1.00 [1.00, 1.00]* |
| ptr.it.mddef | 0.87 [0.86, 0.87] | 0.83 [0.83, 0.83] |
| ptr.it.mddefint | 0.87 [0.86, 0.87] | 0.83 [0.83, 0.83]* |
| ptr.it.mdt | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00]* |
| ptr.it.mdtlb | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| ptr.it.mdtlb+kfence | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| ptr.it.range | 0.88 [0.87, 0.88] | 0.35 [0.34, 0.35] |
| ptr.it1.mddef | 0.86 [0.86, 0.86]* | 0.83 [0.83, 0.83]* |
| ptr.it1.mdtlb | 0.96 [0.96, 0.96]* | 1.00 [1.00, 1.00]* |
| ptr64.it.mdtlb | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00]* |
| vl.ex.mdtlb | 0.96 [0.96, 0.97] | 1.00 [1.00, 1.00]* |
| vl.it.mdtlb | 0.99 [0.99, 1.00] | 1.00 [1.00, 1.00]* |
| vs.it.mdtlb | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00]* |

#### `chain, local numbering`: GPU (ms / ratio to gtfn)

| variant | large | xlarge |
|---|---|---|
| gtfn | 0.349 / 1.00 [1.00, 1.00] | 2.39 / 1.00 [1.00, 1.00] |
| gtfn_us | 0.253 / 0.73 [0.72, 0.74] | 1.88 / 0.79 [0.77, 0.81] |
| naive | 0.205 / 0.59 [0.58, 0.60] | 1.82 / 0.76 [0.74, 0.78] |
| ptr.ex.mddef | 0.13 / 0.37 [0.37, 0.38]* | 0.961 / 0.40 [0.39, 0.41] |
| ptr.ex.mdt | 0.134 / 0.38 [0.38, 0.39] | 0.923 / 0.39 [0.38, 0.40]* |
| ptr.ex.mdtlb | 0.134 / 0.38 [0.38, 0.39] | 0.925 / 0.39 [0.38, 0.39] |
| ptr.ex2.mdtlb | 0.134 / 0.38 [0.38, 0.39]* | 0.914 / 0.39 [0.38, 0.39]* |
| ptr.it.mddef | 0.149 / 0.43 [0.42, 0.43] | 0.994 / 0.42 [0.41, 0.43] |
| ptr.it.mddefint | 0.149 / 0.43 [0.42, 0.44] | 0.974 / 0.41 [0.41, 0.42]* |
| ptr.it.mdt | 0.151 / 0.43 [0.42, 0.44] | 0.958 / 0.40 [0.40, 0.41]* |
| ptr.it.mdtlb | 0.149 / 0.43 [0.42, 0.43] | 0.963 / 0.40 [0.40, 0.41] |
| ptr.it.mdtlb+kfence | 0.15 / 0.43 [0.43, 0.44] | 0.963 / 0.40 [0.40, 0.41] |
| ptr.it.range | 0.165 / 0.47 [0.46, 0.48] | 1.6 / 0.67 [0.66, 0.68] |
| ptr.itlit.mdtlb | 0.232 / 0.66 [0.66, 0.67]* | 1.82 / 0.77 [0.76, 0.78]* |
| ptr64.it.mdtlb | 0.22 / 0.63 [0.62, 0.64] | 1.84 / 0.78 [0.77, 0.79]* |
| vl.ex.mdtlb | 0.134 / 0.38 [0.38, 0.39] | 0.918 / 0.39 [0.38, 0.39]* |
| vl.it.mdtlb | 0.15 / 0.43 [0.43, 0.44] | 0.956 / 0.40 [0.40, 0.41]* |
| vs.it.mdtlb | 0.215 / 0.62 [0.61, 0.63] | 1.83 / 0.77 [0.76, 0.78]* |

#### `chain, local numbering`: CPU, large (ms / ratio to gtfn)

| variant | 1 thr | 2 thr | 4 thr | 8 thr | 18 thr | 36 thr | 72 thr |
|---|---|---|---|---|---|---|---|
| gtfn | 121 / 1.00 [1.00, 1.00]* | 60.5 / 1.00 [1.00, 1.00]* | 30.4 / 1.00 [1.00, 1.00]* | 15.3 / 1.00 [1.00, 1.00]* | 7.23 / 1.00 [1.00, 1.00]* | 3.41 / 1.00 [1.00, 1.00]* | 1.82 / 1.00 [1.00, 1.00] |
| gtfn_us | 98.6 / 0.82 [0.82, 0.82]* | 55.1 / 0.91 [0.91, 0.91]* | 27.6 / 0.91 [0.91, 0.91]* | 13.9 / 0.91 [0.90, 0.91]* | 5.56 / 0.77 [0.77, 0.77]* | 2.82 / 0.83 [0.79, 0.84]* | 1.44 / 0.79 [0.76, 0.83] |
| naive | - | - | 238 / 7.82 [7.81, 7.83]* | 120 / 7.89 [7.87, 7.91]* | 53 / 7.33 [7.31, 7.36]* | 27.6 / 8.10 [7.73, 8.12]* | 13.4 / 7.38 [7.18, 7.67] |
| ptr.ex.mdleft | 96 / 0.79 [0.79, 0.80]* | 48.1 / 0.80 [0.79, 0.80]* | 24.2 / 0.80 [0.80, 0.80]* | 12.2 / 0.80 [0.80, 0.80]* | 7.29 / 1.01 [1.01, 1.01]* | 4.91 / 1.44 [1.38, 1.44]* | 2.47 / 1.36 [1.32, 1.40] |
| ptr.ex.rows | 67.7 / 0.56 [0.56, 0.56]* | 34 / 0.56 [0.56, 0.56]* | 17.2 / 0.57 [0.57, 0.57]* | 8.7 / 0.57 [0.57, 0.57]* | 3.85 / 0.53 [0.53, 0.54]* | 2.04 / 0.60 [0.56, 0.60]* | 1.04 / 0.57 [0.55, 0.60] |
| ptr.ex.rowsnh | 94.8 / 0.78 [0.78, 0.79]* | 47.5 / 0.79 [0.79, 0.79]* | 24 / 0.79 [0.79, 0.79]* | 12.1 / 0.79 [0.79, 0.79]* | 5.36 / 0.74 [0.74, 0.75]* | 2.71 / 0.80 [0.76, 0.81]* | 1.42 / 0.78 [0.76, 0.81] |
| ptr.ex2.rows | 67.8 / 0.56 [0.56, 0.56]* | 34.1 / 0.56 [0.56, 0.57]* | 17.2 / 0.57 [0.57, 0.57]* | 8.69 / 0.57 [0.57, 0.57]* | 3.86 / 0.53 [0.53, 0.54]* | 1.99 / 0.58 [0.55, 0.59]* | 1.03 / 0.57 [0.54, 0.59] |
| ptr.it.mddef | - | - | 58.8 / 1.93 [1.92, 1.97]* | 31.7 / 2.08 [2.05, 2.10]* | 14.3 / 1.98 [1.94, 2.02]* | 8.3 / 2.43 [2.32, 2.44]* | 4.35 / 2.39 [2.31, 2.49] |
| ptr.it.mdleft | 163 / 1.35 [1.35, 1.35]* | 82.6 / 1.37 [1.36, 1.37]* | 41.3 / 1.36 [1.36, 1.36]* | 20.7 / 1.36 [1.35, 1.36]* | 13.2 / 1.82 [1.82, 1.83]* | 8.28 / 2.43 [2.32, 2.44]* | 4.31 / 2.37 [2.30, 2.40] |
| ptr.it.rows | 155 / 1.29 [1.28, 1.29]* | 78.4 / 1.30 [1.29, 1.30]* | 39.3 / 1.29 [1.29, 1.30]* | 19.7 / 1.29 [1.29, 1.30]* | 8.77 / 1.21 [1.21, 1.22]* | 4.42 / 1.30 [1.24, 1.34]* | 2.27 / 1.25 [1.20, 1.32] |
| ptr.it.rowsnh | 165 / 1.36 [1.36, 1.37]* | 81.1 / 1.34 [1.34, 1.34]* | 40.8 / 1.34 [1.34, 1.35]* | 20.4 / 1.34 [1.34, 1.35]* | 9.05 / 1.25 [1.25, 1.26]* | 4.63 / 1.36 [1.30, 1.36]* | 2.41 / 1.32 [1.26, 1.41] |
| ptr.itlit.rows | 72.6 / 0.60 [0.60, 0.60]* | 36.4 / 0.60 [0.60, 0.60]* | 18.3 / 0.60 [0.60, 0.61]* | 9.2 / 0.60 [0.60, 0.61]* | 4.1 / 0.57 [0.57, 0.57]* | 2.11 / 0.62 [0.59, 0.62]* | 1.11 / 0.61 [0.59, 0.64] |
| ptr64.it.rows | 387 / 3.20 [3.20, 3.21]* | 224 / 3.71 [3.70, 3.71]* | 112 / 3.70 [3.69, 3.70]* | 56.2 / 3.69 [3.68, 3.70]* | 24.7 / 3.42 [3.41, 3.43]* | 10.8 / 3.17 [3.03, 3.17]* | 5.87 / 3.22 [3.02, 3.38] |
| vl.ex.rows | 67.7 / 0.56 [0.56, 0.56]* | 34.1 / 0.56 [0.56, 0.56]* | 17.2 / 0.57 [0.56, 0.57]* | 8.68 / 0.57 [0.57, 0.57]* | 3.87 / 0.54 [0.53, 0.54]* | 1.96 / 0.58 [0.55, 0.58]* | 1.02 / 0.56 [0.54, 0.59] |
| vl.it.rows | 218 / 1.81 [1.80, 1.81]* | 111 / 1.84 [1.84, 1.84]* | 55.6 / 1.83 [1.82, 1.83]* | 27.8 / 1.83 [1.82, 1.84]* | 12.6 / 1.74 [1.74, 1.75]* | 6.23 / 1.83 [1.75, 1.85]* | 3.2 / 1.76 [1.69, 1.84] |
| vs.it.rows | 311 / 2.57 [2.57, 2.57]* | 156 / 2.58 [2.58, 2.58]* | 78.5 / 2.58 [2.58, 2.59]* | 39.3 / 2.57 [2.57, 2.58]* | 17.6 / 2.44 [2.43, 2.45]* | 8.83 / 2.59 [2.48, 2.61]* | 4.47 / 2.46 [2.38, 2.57] |

Same cells, ratio to `gtfn_us`:

| variant | 1 thr | 2 thr | 4 thr | 8 thr | 18 thr | 36 thr | 72 thr |
|---|---|---|---|---|---|---|---|
| naive | - | - | 8.61 [8.58, 8.63]* | 8.68 [8.64, 8.71]* | 9.53 [9.49, 9.57]* | 9.79 [9.69, 9.84]* | 9.31 [9.24, 9.42] |
| ptr.ex.mdleft | 0.97 [0.97, 0.98]* | 0.87 [0.87, 0.88]* | 0.88 [0.88, 0.88]* | 0.88 [0.88, 0.88]* | 1.31 [1.31, 1.32]* | 1.74 [1.72, 1.75]* | 1.71 [1.69, 1.75] |
| ptr.ex.rows | 0.69 [0.69, 0.69]* | 0.62 [0.62, 0.62]* | 0.62 [0.62, 0.63]* | 0.63 [0.63, 0.63]* | 0.69 [0.69, 0.70]* | 0.73 [0.70, 0.73]* | 0.72 [0.71, 0.74] |
| ptr.ex.rowsnh | 0.96 [0.96, 0.96]* | 0.86 [0.86, 0.87]* | 0.87 [0.87, 0.87]* | 0.87 [0.87, 0.87]* | 0.96 [0.96, 0.97]* | 0.96 [0.95, 0.98]* | 0.98 [0.96, 1.01] |
| ptr.ex2.rows | 0.69 [0.69, 0.69]* | 0.62 [0.62, 0.62]* | 0.62 [0.62, 0.63]* | 0.63 [0.62, 0.63]* | 0.69 [0.69, 0.70]* | 0.71 [0.69, 0.71]* | 0.71 [0.69, 0.73] |
| ptr.it.mddef | - | - | 2.13 [2.11, 2.17]* | 2.29 [2.26, 2.31]* | 2.58 [2.52, 2.62]* | 2.94 [2.90, 2.96]* | 3.02 [2.99, 3.05] |
| ptr.it.mdleft | 1.66 [1.66, 1.66]* | 1.50 [1.50, 1.50]* | 1.49 [1.49, 1.50]* | 1.49 [1.49, 1.50]* | 2.37 [2.36, 2.38]* | 2.94 [2.90, 2.96]* | 2.99 [2.88, 3.11] |
| ptr.it.rows | 1.57 [1.57, 1.58]* | 1.42 [1.42, 1.43]* | 1.42 [1.42, 1.43]* | 1.42 [1.41, 1.43]* | 1.58 [1.57, 1.58]* | 1.57 [1.55, 1.62]* | 1.58 [1.54, 1.62] |
| ptr.it.rowsnh | 1.67 [1.67, 1.67]* | 1.47 [1.47, 1.48]* | 1.48 [1.47, 1.48]* | 1.47 [1.47, 1.48]* | 1.63 [1.62, 1.63]* | 1.64 [1.62, 1.65]* | 1.67 [1.62, 1.72] |
| ptr.itlit.rows | 0.74 [0.74, 0.74]* | 0.66 [0.66, 0.66]* | 0.66 [0.66, 0.67]* | 0.66 [0.66, 0.67]* | 0.74 [0.73, 0.74]* | 0.75 [0.74, 0.75]* | 0.77 [0.75, 0.79] |
| ptr64.it.rows | 3.92 [3.92, 3.93]* | 4.07 [4.07, 4.08]* | 4.07 [4.06, 4.08]* | 4.06 [4.04, 4.07]* | 4.44 [4.42, 4.45]* | 3.83 [3.78, 3.85]* | 4.07 [3.78, 4.36] |
| vl.ex.rows | 0.69 [0.69, 0.69]* | 0.62 [0.62, 0.62]* | 0.62 [0.62, 0.63]* | 0.63 [0.62, 0.63]* | 0.70 [0.69, 0.70]* | 0.70 [0.69, 0.71]* | 0.71 [0.70, 0.73] |
| vl.it.rows | 2.21 [2.21, 2.22]* | 2.02 [2.02, 2.03]* | 2.01 [2.01, 2.02]* | 2.01 [2.00, 2.02]* | 2.27 [2.26, 2.28]* | 2.21 [2.18, 2.24]* | 2.21 [2.17, 2.26] |
| vs.it.rows | 3.15 [3.15, 3.15]* | 2.83 [2.83, 2.84]* | 2.84 [2.84, 2.85]* | 2.83 [2.82, 2.84]* | 3.17 [3.16, 3.18]* | 3.13 [3.09, 3.16]* | 3.10 [3.07, 3.14] |

GPU, ratio to `gtfn_us`:

| variant | large | xlarge |
|---|---|---|
| naive | 0.81 [0.79, 0.82] | 0.97 [0.95, 0.99] |
| ptr.ex.mddef | 0.51 [0.51, 0.52]* | 0.51 [0.50, 0.52] |
| ptr.ex.mdt | 0.53 [0.52, 0.53] | 0.49 [0.48, 0.50]* |
| ptr.ex.mdtlb | 0.53 [0.52, 0.54] | 0.49 [0.48, 0.50] |
| ptr.ex2.mdtlb | 0.53 [0.52, 0.54]* | 0.49 [0.48, 0.49]* |
| ptr.it.mddef | 0.59 [0.57, 0.60] | 0.53 [0.51, 0.54] |
| ptr.it.mddefint | 0.59 [0.58, 0.60] | 0.52 [0.51, 0.53]* |
| ptr.it.mdt | 0.60 [0.58, 0.60] | 0.51 [0.50, 0.52]* |
| ptr.it.mdtlb | 0.59 [0.57, 0.60] | 0.51 [0.50, 0.52] |
| ptr.it.mdtlb+kfence | 0.59 [0.59, 0.60] | 0.51 [0.50, 0.52] |
| ptr.it.range | 0.65 [0.63, 0.66] | 0.85 [0.84, 0.86] |
| ptr.itlit.mdtlb | 0.91 [0.90, 0.92]* | 0.97 [0.95, 0.98]* |
| ptr64.it.mdtlb | 0.87 [0.86, 0.88] | 0.98 [0.96, 0.99]* |
| vl.ex.mdtlb | 0.53 [0.53, 0.54] | 0.49 [0.48, 0.49]* |
| vl.it.mdtlb | 0.59 [0.59, 0.60] | 0.51 [0.50, 0.52]* |
| vs.it.mdtlb | 0.85 [0.84, 0.86] | 0.97 [0.96, 0.98]* |

#### `chain, random numbering`: GPU (ms / ratio to gtfn)

| variant | large | xlarge |
|---|---|---|
| gtfn | 0.63 / 1.00 [1.00, 1.00] | 11.2 / 1.00 [1.00, 1.00] |
| gtfn_us | 0.629 / 1.00 [1.00, 1.00] | 11.2 / 1.00 [1.00, 1.00] |
| naive | 0.735 / 1.17 [1.16, 1.17] | 9.6 / 0.86 [0.86, 0.86] |
| ptr.ex.mddef | 0.832 / 1.32 [1.32, 1.32]* | 10.2 / 0.91 [0.91, 0.91] |
| ptr.ex.mdt | 0.635 / 1.01 [1.01, 1.01] | 11.3 / 1.01 [1.01, 1.01]* |
| ptr.ex.mdtlb | 0.635 / 1.01 [1.01, 1.01] | 11.3 / 1.01 [1.01, 1.01] |
| ptr.ex2.mdtlb | 0.634 / 1.01 [1.01, 1.01]* | 11.3 / 1.01 [1.01, 1.01]* |
| ptr.it.mddef | 0.814 / 1.29 [1.29, 1.29] | 10.1 / 0.91 [0.91, 0.91] |
| ptr.it.mddefint | 0.814 / 1.29 [1.29, 1.29] | 10.1 / 0.91 [0.91, 0.91]* |
| ptr.it.mdt | 0.631 / 1.00 [1.00, 1.00] | 11.3 / 1.01 [1.01, 1.01]* |
| ptr.it.mdtlb | 0.631 / 1.00 [1.00, 1.00] | 11.3 / 1.01 [1.01, 1.01] |
| ptr.it.mdtlb+kfence | 0.63 / 1.00 [1.00, 1.00] | 11.3 / 1.01 [1.01, 1.01] |
| ptr.it.range | 1.38 / 2.19 [2.19, 2.19] | 11.2 / 1.00 [0.99, 1.01] |
| ptr.itlit.mdtlb | 0.624 / 0.99 [0.99, 0.99]* | 11.2 / 1.00 [1.00, 1.00]* |
| ptr64.it.mdtlb | 0.626 / 0.99 [0.99, 1.00] | 11.2 / 1.00 [1.00, 1.00]* |
| vl.ex.mdtlb | 0.634 / 1.01 [1.00, 1.01] | 11.3 / 1.01 [1.01, 1.01]* |
| vl.it.mdtlb | 0.631 / 1.00 [1.00, 1.00] | 11.3 / 1.01 [1.01, 1.01]* |
| vs.it.mdtlb | 0.626 / 0.99 [0.99, 1.00] | 11.2 / 1.00 [1.00, 1.00]* |

#### `chain, random numbering`: CPU, large (ms / ratio to gtfn)

| variant | 1 thr | 2 thr | 4 thr | 8 thr | 18 thr | 36 thr | 72 thr |
|---|---|---|---|---|---|---|---|
| gtfn | 661 / 1.00 [1.00, 1.00]* | 343 / 1.00 [1.00, 1.00]* | 177 / 1.00 [1.00, 1.00]* | 89.3 / 1.00 [1.00, 1.00]* | 41.4 / 1.00 [1.00, 1.00]* | 22.1 / 1.00 [1.00, 1.00]* | 13 / 1.00 [1.00, 1.00] |
| gtfn_us | 510 / 0.77 [0.77, 0.77]* | 265 / 0.77 [0.77, 0.77]* | 137 / 0.78 [0.77, 0.78]* | 71.6 / 0.80 [0.80, 0.80]* | 33.6 / 0.81 [0.81, 0.81]* | 17.2 / 0.78 [0.78, 0.78]* | 10.4 / 0.80 [0.79, 0.81] |
| naive | - | - | 289 / 1.63 [1.62, 1.64]* | 142 / 1.59 [1.58, 1.60]* | 62.3 / 1.50 [1.50, 1.51]* | 32.1 / 1.45 [1.45, 1.46]* | 15.7 / 1.21 [1.20, 1.22] |
| ptr.ex.mdleft | 487 / 0.74 [0.74, 0.74]* | 250 / 0.73 [0.73, 0.73]* | 130 / 0.73 [0.73, 0.74]* | 68.8 / 0.77 [0.77, 0.77]* | 41.8 / 1.01 [1.00, 1.01]* | 27.5 / 1.25 [1.24, 1.25]* | 14.6 / 1.13 [1.12, 1.14] |
| ptr.ex.rows | 457 / 0.69 [0.69, 0.69]* | 235 / 0.69 [0.69, 0.69]* | 122 / 0.69 [0.69, 0.69]* | 64.4 / 0.72 [0.72, 0.72]* | 29.9 / 0.72 [0.72, 0.72]* | 15.4 / 0.70 [0.69, 0.70]* | 9.16 / 0.71 [0.70, 0.71] |
| ptr.ex.rowsnh | 481 / 0.73 [0.72, 0.73]* | 247 / 0.72 [0.72, 0.72]* | 128 / 0.73 [0.72, 0.73]* | 67.6 / 0.76 [0.75, 0.76]* | 31.7 / 0.77 [0.76, 0.77]* | 16.3 / 0.74 [0.74, 0.74]* | 9.85 / 0.76 [0.75, 0.77] |
| ptr.ex2.rows | 456 / 0.69 [0.69, 0.69]* | 235 / 0.68 [0.68, 0.69]* | 122 / 0.69 [0.69, 0.69]* | 64.3 / 0.72 [0.72, 0.72]* | 29.9 / 0.72 [0.72, 0.72]* | 15.4 / 0.70 [0.70, 0.70]* | 9.16 / 0.71 [0.69, 0.71] |
| ptr.it.mddef | - | - | 128 / 0.72 [0.72, 0.73]* | 56.3 / 0.63 [0.63, 0.63]* | 24.4 / 0.59 [0.59, 0.59]* | 13.8 / 0.62 [0.62, 0.63]* | 6.6 / 0.51 [0.49, 0.52] |
| ptr.it.mdleft | 714 / 1.08 [1.08, 1.08]* | 369 / 1.08 [1.07, 1.08]* | 190 / 1.07 [1.07, 1.08]* | 97 / 1.09 [1.08, 1.09]* | 59.4 / 1.43 [1.43, 1.44]* | 39.9 / 1.81 [1.80, 1.81]* | 21.1 / 1.63 [1.61, 1.65] |
| ptr.it.rows | 696 / 1.05 [1.05, 1.05]* | 361 / 1.05 [1.05, 1.06]* | 186 / 1.05 [1.05, 1.06]* | 94.5 / 1.06 [1.05, 1.06]* | 43.8 / 1.06 [1.05, 1.06]* | 23.5 / 1.06 [1.06, 1.07]* | 14 / 1.07 [1.06, 1.08] |
| ptr.it.rowsnh | 710 / 1.07 [1.07, 1.08]* | 368 / 1.07 [1.07, 1.07]* | 190 / 1.07 [1.07, 1.07]* | 96.2 / 1.08 [1.07, 1.08]* | 44.6 / 1.08 [1.07, 1.08]* | 23.9 / 1.08 [1.07, 1.08]* | 14.1 / 1.09 [1.08, 1.10] |
| ptr.itlit.rows | 459 / 0.69 [0.69, 0.70]* | 237 / 0.69 [0.69, 0.69]* | 123 / 0.70 [0.69, 0.70]* | 64.8 / 0.73 [0.72, 0.73]* | 30.2 / 0.73 [0.73, 0.73]* | 15.6 / 0.71 [0.70, 0.71]* | 9.34 / 0.72 [0.71, 0.73] |
| ptr64.it.rows | 970 / 1.47 [1.46, 1.47]* | 499 / 1.46 [1.45, 1.46]* | 253 / 1.43 [1.43, 1.43]* | 128 / 1.43 [1.42, 1.43]* | 58.4 / 1.41 [1.41, 1.41]* | 30.8 / 1.40 [1.39, 1.40]* | 17.9 / 1.38 [1.37, 1.39] |
| vl.ex.rows | 457 / 0.69 [0.69, 0.69]* | 235 / 0.69 [0.69, 0.69]* | 122 / 0.69 [0.69, 0.69]* | 64.2 / 0.72 [0.71, 0.72]* | 29.9 / 0.72 [0.72, 0.72]* | 15.4 / 0.70 [0.69, 0.70]* | 9.16 / 0.71 [0.69, 0.71] |
| vl.it.rows | 826 / 1.25 [1.25, 1.25]* | 427 / 1.25 [1.24, 1.25]* | 220 / 1.24 [1.24, 1.24]* | 111 / 1.24 [1.24, 1.25]* | 51.5 / 1.24 [1.24, 1.25]* | 27.4 / 1.24 [1.23, 1.24]* | 16.3 / 1.26 [1.24, 1.26] |
| vs.it.rows | 1034 / 1.56 [1.56, 1.56]* | 526 / 1.54 [1.54, 1.54]* | 268 / 1.51 [1.51, 1.52]* | 135 / 1.51 [1.51, 1.52]* | 61.7 / 1.49 [1.48, 1.49]* | 32.5 / 1.47 [1.47, 1.48]* | 19 / 1.46 [1.45, 1.48] |

Same cells, ratio to `gtfn_us`:

| variant | 1 thr | 2 thr | 4 thr | 8 thr | 18 thr | 36 thr | 72 thr |
|---|---|---|---|---|---|---|---|
| naive | - | - | 2.11 [2.09, 2.11]* | 1.98 [1.98, 1.99]* | 1.86 [1.85, 1.87]* | 1.86 [1.86, 1.87]* | 1.51 [1.50, 1.52] |
| ptr.ex.mdleft | 0.96 [0.95, 0.96]* | 0.95 [0.94, 0.95]* | 0.95 [0.94, 0.95]* | 0.96 [0.96, 0.96]* | 1.24 [1.24, 1.25]* | 1.60 [1.59, 1.61]* | 1.41 [1.40, 1.42] |
| ptr.ex.rows | 0.90 [0.90, 0.90]* | 0.89 [0.89, 0.89]* | 0.89 [0.89, 0.89]* | 0.90 [0.90, 0.90]* | 0.89 [0.89, 0.89]* | 0.89 [0.89, 0.90]* | 0.88 [0.87, 0.89] |
| ptr.ex.rowsnh | 0.94 [0.94, 0.94]* | 0.93 [0.93, 0.94]* | 0.94 [0.93, 0.94]* | 0.94 [0.94, 0.95]* | 0.95 [0.94, 0.95]* | 0.95 [0.95, 0.95]* | 0.95 [0.94, 0.96] |
| ptr.ex2.rows | 0.89 [0.89, 0.90]* | 0.89 [0.88, 0.89]* | 0.89 [0.88, 0.89]* | 0.90 [0.90, 0.90]* | 0.89 [0.89, 0.90]* | 0.90 [0.89, 0.90]* | 0.88 [0.87, 0.89] |
| ptr.it.mddef | - | - | 0.93 [0.93, 0.94]* | 0.79 [0.78, 0.79]* | 0.73 [0.73, 0.73]* | 0.80 [0.79, 0.80]* | 0.64 [0.62, 0.65] |
| ptr.it.mdleft | 1.40 [1.40, 1.40]* | 1.39 [1.39, 1.40]* | 1.38 [1.38, 1.39]* | 1.35 [1.35, 1.36]* | 1.77 [1.77, 1.78]* | 2.32 [2.31, 2.32]* | 2.03 [2.02, 2.06] |
| ptr.it.rows | 1.37 [1.36, 1.37]* | 1.36 [1.36, 1.37]* | 1.36 [1.35, 1.36]* | 1.32 [1.31, 1.32]* | 1.31 [1.30, 1.31]* | 1.36 [1.36, 1.37]* | 1.34 [1.33, 1.35] |
| ptr.it.rowsnh | 1.39 [1.39, 1.40]* | 1.39 [1.39, 1.39]* | 1.38 [1.38, 1.39]* | 1.34 [1.34, 1.35]* | 1.33 [1.33, 1.34]* | 1.39 [1.38, 1.39]* | 1.36 [1.35, 1.37] |
| ptr.itlit.rows | 0.90 [0.90, 0.90]* | 0.90 [0.89, 0.90]* | 0.90 [0.89, 0.90]* | 0.90 [0.90, 0.91]* | 0.90 [0.90, 0.90]* | 0.91 [0.90, 0.91]* | 0.90 [0.89, 0.90] |
| ptr64.it.rows | 1.90 [1.90, 1.91]* | 1.88 [1.88, 1.89]* | 1.85 [1.84, 1.85]* | 1.78 [1.78, 1.79]* | 1.74 [1.74, 1.75]* | 1.79 [1.78, 1.79]* | 1.72 [1.71, 1.73] |
| vl.ex.rows | 0.90 [0.90, 0.90]* | 0.89 [0.89, 0.89]* | 0.89 [0.89, 0.89]* | 0.90 [0.89, 0.90]* | 0.89 [0.89, 0.90]* | 0.89 [0.89, 0.90]* | 0.88 [0.87, 0.89] |
| vl.it.rows | 1.62 [1.62, 1.62]* | 1.61 [1.61, 1.62]* | 1.60 [1.59, 1.61]* | 1.55 [1.55, 1.56]* | 1.53 [1.53, 1.54]* | 1.59 [1.58, 1.60]* | 1.57 [1.56, 1.58] |
| vs.it.rows | 2.03 [2.02, 2.03]* | 1.99 [1.99, 1.99]* | 1.95 [1.95, 1.96]* | 1.89 [1.88, 1.89]* | 1.84 [1.83, 1.85]* | 1.89 [1.88, 1.89]* | 1.83 [1.82, 1.84] |

GPU, ratio to `gtfn_us`:

| variant | large | xlarge |
|---|---|---|
| naive | 1.17 [1.17, 1.17] | 0.86 [0.86, 0.86] |
| ptr.ex.mddef | 1.33 [1.32, 1.33]* | 0.91 [0.91, 0.91] |
| ptr.ex.mdt | 1.01 [1.01, 1.01] | 1.01 [1.01, 1.01]* |
| ptr.ex.mdtlb | 1.01 [1.01, 1.01] | 1.01 [1.01, 1.01] |
| ptr.ex2.mdtlb | 1.01 [1.01, 1.01]* | 1.01 [1.01, 1.01]* |
| ptr.it.mddef | 1.29 [1.29, 1.30] | 0.91 [0.91, 0.91] |
| ptr.it.mddefint | 1.29 [1.29, 1.30] | 0.91 [0.91, 0.91]* |
| ptr.it.mdt | 1.00 [1.00, 1.00] | 1.01 [1.01, 1.01]* |
| ptr.it.mdtlb | 1.00 [1.00, 1.00] | 1.01 [1.01, 1.01] |
| ptr.it.mdtlb+kfence | 1.00 [1.00, 1.00] | 1.01 [1.01, 1.01] |
| ptr.it.range | 2.19 [2.19, 2.20] | 1.00 [0.99, 1.01] |
| ptr.itlit.mdtlb | 0.99 [0.99, 0.99]* | 1.00 [1.00, 1.00]* |
| ptr64.it.mdtlb | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00]* |
| vl.ex.mdtlb | 1.01 [1.01, 1.01] | 1.01 [1.01, 1.01]* |
| vl.it.mdtlb | 1.00 [1.00, 1.00] | 1.01 [1.01, 1.01]* |
| vs.it.mdtlb | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00]* |

### 2.4 Follow-up: layout of the GPU `tritmp` temporaries

#### Every shape and layout (two follow-up runs) [M]

Offset = elements between a 256-byte boundary and the first compute point.

**large**: gtfn `tri` 0.5228 ms, `tritmp` 0.5054 ms.

| Kokkos shape | temporaries | ms | / gtfn `tritmp` | per run | / Kokkos `tri`, same shape |
|---|---|---|---|---|---|
| two kernels | caller fields (`tri`) | 0.521 | 0.997 [0.995, 0.998] (vs gtfn `tri`) | 0.997 / 0.996 | 1 |
| two kernels | pool variant of the earlier runs | 0.5212 | 1.031 [1.029, 1.034] | 1.030 / 1.032 | 1.000 |
| two kernels | caller-field strides, offset 3 | 0.5205 | 1.030 [1.028, 1.031] | 1.030 / 1.030 | 0.999 |
| two kernels | caller-field strides, offset 0 | 0.5093 | 1.008 [1.006, 1.010] | 1.007 / 1.009 | 0.978 |
| two kernels | compact, offset 3 | 0.5051 | 0.999 [0.997, 1.002] | 0.998 / 1.001 | 0.969 |
| two kernels | compact, offset 0 (gtfn-like) | 0.5074 | 1.004 [1.001, 1.007] | 1.003 / 1.005 | 0.974 |
| merged | caller fields (`tri`) | 0.5331 | 1.020 [1.018, 1.021] (vs gtfn `tri`) | 1.020 / 1.020 | 1 |
| merged | pool variant of the earlier runs | 0.5299 | 1.048 [1.046, 1.052] | 1.047 / 1.050 | 0.994 |
| merged | caller-field strides, offset 3 | 0.5308 | 1.050 [1.048, 1.052] | 1.051 / 1.050 | 0.996 |
| merged | caller-field strides, offset 0 | 0.5124 | 1.014 [1.011, 1.016] | 1.013 / 1.015 | 0.961 |
| merged | compact, offset 3 | 0.5185 | 1.026 [1.024, 1.028] | 1.025 / 1.027 | 0.973 |
| merged | compact, offset 0 (gtfn-like) | 0.5089 | 1.007 [1.004, 1.009] | 1.006 / 1.008 | 0.955 |
| merged, flat range | caller fields (`tri`) | 0.5292 | 1.012 [1.011, 1.014] (vs gtfn `tri`) | 1.013 / 1.012 | 1 |
| merged, flat range | pool variant of the earlier runs | 0.5246 | 1.038 [1.035, 1.041] | 1.036 / 1.040 | 0.991 |
| merged, flat range | caller-field strides, offset 3 | 0.525 | 1.039 [1.037, 1.040] | 1.038 / 1.039 | 0.992 |
| merged, flat range | caller-field strides, offset 0 | 0.5065 | 1.002 [1.000, 1.004] | 1.001 / 1.003 | 0.957 |
| merged, flat range | compact, offset 3 | 0.5115 | 1.012 [1.009, 1.014] | 1.011 / 1.013 | 0.966 |
| merged, flat range | compact, offset 0 (gtfn-like) | 0.504 | 0.997 [0.995, 1.000] | 0.996 / 0.999 | 0.952 |
| merged, explicit | pool variant of the earlier runs | 0.5303 | 1.049 [1.047, 1.052] | 1.048 / 1.050 | - |

**xlarge**: gtfn `tri` 2.044 ms, `tritmp` 1.958 ms.

| Kokkos shape | temporaries | ms | / gtfn `tritmp` | per run | / Kokkos `tri`, same shape |
|---|---|---|---|---|---|
| two kernels | caller fields (`tri`) | 2.058 | 1.007 [1.003, 1.010] (vs gtfn `tri`) | 1.008 / 1.006 | 1 |
| two kernels | pool variant of the earlier runs | 2.06 | 1.052 [1.048, 1.056] | 1.050 / 1.053 | 1.001 |
| two kernels | caller-field strides, offset 3 | 2.06 | 1.052 [1.047, 1.057] | 1.048 / 1.056 | 1.001 |
| two kernels | caller-field strides, offset 0 | 2.01 | 1.027 [1.024, 1.032] | 1.025 / 1.028 | 0.977 |
| two kernels | compact, offset 3 | 2.026 | 1.035 [1.029, 1.040] | 1.032 / 1.038 | 0.985 |
| two kernels | compact, offset 0 (gtfn-like) | 2.018 | 1.030 [1.026, 1.036] | 1.029 / 1.032 | 0.981 |
| merged | caller fields (`tri`) | 2.089 | 1.022 [1.020, 1.024] (vs gtfn `tri`) | 1.022 / 1.023 | 1 |
| merged | pool variant of the earlier runs | 2.082 | 1.063 [1.059, 1.067] | 1.061 / 1.065 | 0.996 |
| merged | caller-field strides, offset 3 | 2.083 | 1.063 [1.057, 1.069] | 1.059 / 1.068 | 0.997 |
| merged | caller-field strides, offset 0 | 2.011 | 1.027 [1.023, 1.031] | 1.025 / 1.030 | 0.963 |
| merged | compact, offset 3 | 2.051 | 1.048 [1.042, 1.053] | 1.044 / 1.051 | 0.982 |
| merged | compact, offset 0 (gtfn-like) | 1.994 | 1.018 [1.014, 1.024] | 1.015 / 1.022 | 0.954 |
| merged, flat range | caller fields (`tri`) | 2.022 | 0.989 [0.986, 0.992] (vs gtfn `tri`) | 0.988 / 0.991 | 1 |
| merged, flat range | pool variant of the earlier runs | 2.01 | 1.027 [1.023, 1.029] | 1.025 / 1.028 | 0.994 |
| merged, flat range | caller-field strides, offset 3 | 2.013 | 1.028 [1.021, 1.034] | 1.023 / 1.033 | 0.996 |
| merged, flat range | caller-field strides, offset 0 | 1.931 | 0.986 [0.981, 0.991] | 0.983 / 0.990 | 0.955 |
| merged, flat range | compact, offset 3 | 1.976 | 1.009 [1.004, 1.014] | 1.005 / 1.013 | 0.978 |
| merged, flat range | compact, offset 0 (gtfn-like) | 1.918 | 0.979 [0.975, 0.984] | 0.977 / 0.982 | 0.949 |
| merged, explicit | pool variant of the earlier runs | 2.083 | 1.064 [1.060, 1.068] | 1.062 / 1.066 | - |

**small**: gtfn `tri` 0.06633 ms, `tritmp` 0.06606 ms.

| Kokkos shape | temporaries | ms | / gtfn `tritmp` | per run | / Kokkos `tri`, same shape |
|---|---|---|---|---|---|
| two kernels | caller fields (`tri`) | 0.06397 | 0.964 [0.959, 0.970] (vs gtfn `tri`) | 0.969 / 0.960 | 1 |
| two kernels | pool variant of the earlier runs | 0.06357 | 0.962 [0.956, 0.969] | 0.965 / 0.960 | 0.994 |
| two kernels | caller-field strides, offset 3 | 0.06353 | 0.962 [0.956, 0.970] | 0.965 / 0.958 | 0.993 |
| two kernels | caller-field strides, offset 0 | 0.06309 | 0.955 [0.950, 0.963] | 0.959 / 0.952 | 0.986 |
| two kernels | compact, offset 3 | 0.06342 | 0.960 [0.954, 0.967] | 0.963 / 0.957 | 0.991 |
| two kernels | compact, offset 0 (gtfn-like) | 0.06289 | 0.952 [0.947, 0.960] | 0.954 / 0.950 | 0.983 |
| merged | caller fields (`tri`) | 0.06084 | 0.917 [0.913, 0.921] (vs gtfn `tri`) | 0.919 / 0.915 | 1 |
| merged | pool variant of the earlier runs | 0.06043 | 0.915 [0.912, 0.921] | 0.915 / 0.914 | 0.993 |
| merged | caller-field strides, offset 3 | 0.06045 | 0.915 [0.911, 0.921] | 0.917 / 0.914 | 0.994 |
| merged | caller-field strides, offset 0 | 0.05985 | 0.906 [0.902, 0.913] | 0.908 / 0.904 | 0.984 |
| merged | compact, offset 3 | 0.06041 | 0.915 [0.912, 0.919] | 0.915 / 0.914 | 0.993 |
| merged | compact, offset 0 (gtfn-like) | 0.05925 | 0.897 [0.892, 0.903] | 0.898 / 0.896 | 0.974 |
| merged, flat range | caller fields (`tri`) | 0.05892 | 0.888 [0.884, 0.893] (vs gtfn `tri`) | 0.891 / 0.886 | 1 |
| merged, flat range | pool variant of the earlier runs | 0.05861 | 0.887 [0.884, 0.893] | 0.888 / 0.886 | 0.995 |
| merged, flat range | caller-field strides, offset 3 | 0.05864 | 0.888 [0.884, 0.893] | 0.890 / 0.885 | 0.995 |
| merged, flat range | caller-field strides, offset 0 | 0.05802 | 0.878 [0.874, 0.884] | 0.880 / 0.876 | 0.985 |
| merged, flat range | compact, offset 3 | 0.05851 | 0.886 [0.884, 0.890] | 0.886 / 0.885 | 0.993 |
| merged, flat range | compact, offset 0 (gtfn-like) | 0.05771 | 0.874 [0.869, 0.879] | 0.875 / 0.872 | 0.979 |
| merged, explicit | pool variant of the earlier runs | 0.06039 | 0.914 [0.910, 0.921] | 0.915 / 0.913 | - |

## 3. Findings

Every ratio below is Kokkos time / gtfn time (below 1: Kokkos faster) from the tables in section 2 and
the original per-allocation data. "large / xlarge" are the two GPU sizes.

### Verdicts in one table

| claim | verdict |
|---|---|
| (a) a gtfn-shaped Kokkos kernel reaches <= 1.0x | CPU: yes for the right source form (parity on some cells). GPU: yes for Cartesian stencils; **not met by the gtfn-shaped tridiagonal solve with temporaries** (1.03-1.05 with field-shaped pool temporaries; 1.004 / 1.030 with compact, aligned ones, see (e)); a flat-range merged kernel with aligned temporaries does reach it (0.997 / 0.979); parity (1.00 / 1.01) for the solve with caller fields; unstructured depends on source form and baseline |
| (b) iterator-style = explicit indexing | Yes on GPU stencils and column scans (within 1 %). **No for neighbor accesses**: some iterator forms are 1.6-2.0x slower than explicit on GPU, others 1.5-3.8x slower on CPU at the large size (up to 6.5x at the small size), and the form that is fast on one is slow on the other |
| (c) unmanaged View = raw pointer | Yes for `LayoutLeft` (explicit indexing: 0.95-1.01 CPU, within 1 % on GPU stencils/unstructured at the large sizes; GPU scans, measured with the iterator body only: +3-5 % large, +2 % small). `LayoutStride` costs up to 10 % (`lap2`) and up to 1.9-2.0x on `chain` iterator code |
| (d) loop shape, tile, index type matter | Yes. CPU: row loops beat `MDRangePolicy` (on local data). GPU: tile choice is a few percent on stencils, large and of either sign on random tables; `IndexType<int>` and launch bounds changed nothing; 64-bit index and offsets cost up to 1.9x (GPU) and 2.9x (CPU), both on `chain` iterator code |
| (e) scans | GPU: with caller fields no Kokkos shape beats gtfn by more than about 1 % at the large sizes; aligning the first compute point of the temporaries gains 2-4 %. CPU: gtfn-shaped two-pass column loop = parity; explicit column chunks help at 1-18 threads (best 0.43 of gtfn on large `tri`); **iterator chunks are 1.3-2.9x slower than the iterator column loop** (1.2-4.4x slower than explicit chunks; large size, more at the small one) |
| (f) CPU margin comes from gtfn's temporaries | At 36-72 threads mostly yes (`hdiff` at 72: factor 5.9 of 10). At 1 thread mostly no (factor 1.1 of 2.3) |

> **Correction (reconciliation round).** Row (c): the GPU-scan View penalty also applies to
> explicit bodies (see the corrections at the top).

### (a) Does a gtfn-shaped Kokkos kernel reach <= 1.0x?

"gtfn-shaped" = iterator-style functor, pointer + strides, one launch per gtfn launch.

**GPU [M]**

| kernel | iterator, large / xlarge | explicit, large / xlarge | at or below 1.0? |
|---|---|---|---|
| lap | 0.95 / 0.93 | 0.95 / 0.93 | yes |
| lap2 | 0.81 / 0.77 | 0.80 / 0.76 | yes |
| hdiff | 0.99 / 0.99 | 0.99 / 0.99 | yes (no margin) |
| tmpst | 0.98 / 0.98 | 0.98 / 0.98 | yes (no margin) |
| tri, two kernels (gtfn-shaped) | 1.00 / 1.01 | not built | parity; xlarge interval above 1.0 |
| tri, one merged kernel | 1.02 / 1.02 | 1.02 / 1.02 | **no** |
| tritmp, two kernels, field-shaped pool temporaries | **1.03 / 1.05** | not built | **no** |
| tritmp, merged, field-shaped pool temporaries | 1.05 / 1.07 | 1.05 / 1.07 | **no** |
| tritmp, two kernels, compact aligned temporaries (follow-up) | **1.004 / 1.030** | not built | **no** (interval above 1.0 at both) |
| tritmp, merged flat range, compact aligned temporaries (follow-up) | 0.997 / 0.979 | not built | yes; picked out after the run |
| v2e local, literal gtfn form | 0.92 / 0.93 (**1.03 / 1.05 vs `gtfn_us`**) | 0.58 / 0.57 (0.64 / 0.64) | against stock gtfn only |
| chain local, literal | 0.66 / 0.77 (0.91 / 0.97 vs `gtfn_us`) | 0.38 / 0.39 (0.53 / 0.49) | yes |
| v2e random, literal | 0.99 / 1.00 | 0.96 / 1.00 | parity |
| chain random, literal | 0.99 / 1.00 | 1.01 / 1.01 | parity (explicit about 1 % above) |

* The best scan shape found (one thread per column, flat `RangePolicy`, merged) gives `tri`
  1.01 / 0.99 and `tritmp` 1.04 / 1.03. With the field-shaped pool temporaries, `tritmp` misses
  the bar in every launch shape measured at the large sizes, with narrow intervals (about
  +-0.5 %). Kokkos takes the same time with caller fields and with those temporaries, whereas gtfn
  gets faster with its own. Aligned temporaries recover part or all of that, depending on the
  launch shape (section (e)).
* Small size (32x32x80): all GPU candidates are below 1.0 (0.58-0.97), but gtfn's own small timings
  move by up to 12 % between allocations. The gaps are 3-6 us per call **[M]** (e.g. `lap` 7.8 vs
  11.4 us, explicit `chain` 8.7 vs 15.0 us). Both timed regions contain per-call host work
  (section 1, Method), which was not separated from device time, so these ratios should be read
  mostly as a lower fixed cost per call, not as faster kernels **[I]**.
* Turning on gt4py's existing unit-stride option (`gtfn_us`) makes gtfn itself 0.73-0.89x on local
  unstructured kernels at the large GPU sizes **[M]**. A Kokkos backend with a static unit stride
  should be judged against that.

**CPU, one socket [M]** (large size, ratio at 1 / 8 / 72 threads)

> **Correction (reconciliation round).** "gtfn-shaped iterator candidates ≤ 1.0" holds for the
> literal iterator form only; the bound-once form is 1.25-2.38x (bold rows below).

| kernel | candidate | 1 thr | 8 thr | 72 thr |
|---|---|---|---|---|
| lap | rows, explicit | 0.33 | 0.39 | 0.93 (about 1.0, **not resolved**: spread between allocations up to 11 %) |
| lap2 | rows, explicit | 0.51 | 0.51 | 0.84 |
| hdiff | rows, explicit, pool | 0.44 | 0.19 | 0.10 |
| tmpst | rows, explicit, pool | 0.43 | 0.19 | 0.13 |
| tri | column, two passes (gtfn-shaped) | 0.99 | 0.99 | 1.00 |
| tri | explicit, chunks of 16 | 0.65 | 0.74 | 0.87 |
| tri | explicit, chunks of 32 (not run at 8; value at 4) | 0.43 | 0.47 | 0.84 |
| tritmp | column, explicit, pool | 0.52 | 0.33 | 0.13 |
| tritmp | explicit, chunks of 16, pool | 0.34 | 0.24 | 0.13 |
| tritmp | explicit, chunks of 32, pool (not run at 8; value at 4) | 0.24 | 0.18 | 0.11 |
| v2e local | rows, explicit | 0.63 | 0.64 | 0.78 |
| v2e local | rows, iterator, literal gtfn form (gtfn-shaped) | 0.70 | 0.70 | 0.82 |
| chain local | rows, explicit | 0.56 | 0.57 | 0.57 |
| chain local | rows, iterator, literal gtfn form (gtfn-shaped) | 0.60 | 0.60 | 0.61 |
| **v2e local** | **rows, iterator bound once** | **2.38** | **2.33** | **2.03** |
| **chain local** | **rows, iterator bound once** | **1.29** | **1.29** | **1.25** |
| **tri** | **iterator, chunks of 16** | **2.85** | **2.73** | **1.15** |

* For the Cartesian stencils the gtfn-shaped iterator rows equal the explicit rows above
  (`it` / `ex` 0.99-1.03; at 72 threads `lap` 0.95 and `lap2` 0.86 against gtfn).
* Over all measured 1-72 thread and both-size cells, the gtfn-shaped iterator candidates (`rows` stencils and `v2e`, literal `chain`, two-pass `tri`, column `tritmp`) are at or below 1.0
  against gtfn except one cell at 1.002 [0.99, 1.01] (large `tri`, 72 threads). Against
  `gtfn_fixed` one cell is above: small `tritmp` at 36 threads, 1.02.
* **Merging** the two sweeps into one column loop misses on small `tri` at 1-4 threads: 1.03-1.04
  for iterator, explicit and pointer-walking columns alike; the two-pass form there is 0.97-0.99
  (built for the iterator only). The explicit chunked scans are below 1.0 in every cell and chunk
  size measured.
* The ratio depends strongly on thread count: gtfn_cpu catches up on `lap`/`lap2` when the socket
  becomes bandwidth-bound, and falls further behind on kernels with temporaries.

**[I]** On this CPU the bar is reachable on every kernel tested, with a clear margin on most; the
two-pass `tri` and `lap` at 72 threads are at parity only. On this GPU it is reachable for the
stencils, with 5-24 % margin on `lap`/`lap2` but only 1-2 % on `hdiff`/`tmpst`. `tri` is at
parity. For `tritmp` the gtfn-shaped two-kernel form stays 0.4 % (large) and 3 % (xlarge) slower
even with compact, aligned temporaries; only a flat-range merged kernel, identified after the run,
reaches <= 1.0 at both sizes. A plan that requires <= 1.0x should align temporaries and expect to
tune the launch shape of vertical solves.

### (b) Iterator-style vs explicit indexing

**GPU [M]** (`it` / `ex` at the same shape)

| kernels | large | xlarge | small |
|---|---|---|---|
| lap, lap2, hdiff, tmpst | 1.00-1.01 | 1.00 | 1.00-1.01 |
| tri, tritmp (also pointer-walking) | 1.00 | 1.00-1.02 | 0.99-1.00 |
| v2e local, literal form (shifts twice, ternary) | **1.60** | **1.64** | 1.16 |
| v2e local, bound once with `if` | 1.03 | 1.02 | 1.02 |
| chain local, literal form (re-shifts per access) | **1.73** | **1.98** | 1.21 |
| chain local, bound once | 1.11 | 1.04 | 1.02 |
| v2e / chain random (all forms) | 0.98-1.03 | 0.99-1.00 | 0.98-1.08 |

* Re-reading the neighbor table at every use is not the cause: the explicit variant that does so
  (`ex2`) equals `ex` (0.99-1.01) **[M]**.
* Register use per thread of the launched kernel **[M]**: `lap` 18 = 18, `lap2` 32 = 32, `tri` 52 =
  52 (iterator = explicit); `v2e` literal **14** (126 instructions) vs explicit 28 (89) vs bound once
  30 (108); `chain` literal **18** (281) vs bound once 32 (177) vs explicit 32 (161). The slow forms
  use few registers and more instructions, which suggests their loads are not overlapped
  **[I, no GPU profile taken]**.
* Compile time **[M]**: one-kernel translation units take 6.9-8.5 s, headers alone 6.6-6.9 s;
  iterator and explicit differ by less than the 1 s run-to-run variation.

**CPU [M]** (`it` / `ex`, large, 1-72 threads)

| code | ratio |
|---|---|
| Cartesian stencils; column-per-iteration scans | 0.91-1.03 |
| v2e literal / chain literal | 1.02-1.10 / 1.00-1.07 |
| **v2e bound once** | **2.5-3.8** |
| **chain bound once** | **2.2-2.3** (random tables 1.5) |
| **chunked scan, iterator** | **1.2-1.3 at 72 threads, 2.0 at 36, 2.8-4.4 at 1-18** |

Small size: `v2e` bound once 2.6-6.5, `chain` bound once 2.2-3.0, chunked iterator scan 2.6-9.8
(all over 1-72 threads); literal forms 1.02-1.08 **[M]**.

Why, from the generated assembly (single-variant translation units compiled with the library's
flags, so the link to the timed library is **[I]**):

* In the slow forms GCC 14.2 keeps the iterator object in **stack memory inside the innermost loop**
  **[M]**: stack loads/stores in the kernel loop function are 39/35 for `v2e` bound-once and 42/37
  for `chain` bound-once, against 0/0 for explicit and 3/3 for the literal
  forms.
* The chunked iterator scan is **not vectorised** (no vector `fdiv`, GCC reports no vectorised loop);
  the explicit one is (30 vector `fdiv`) **[M]**.
* The magnitudes are **not fully explained**. The unvectorised `ptr64.it.chunk16` runs about 3x
  faster than `ptr.it.chunk16` (0.31-0.45 of it up to 36 threads) with a similar loop; in the `int`
  build the loop reloads the upper halves of 64-bit stack stores, so a store-forwarding penalty is a
  plausible cause **[I, no hardware counters]**. `chain` with `ptr`, `vl`, `ptr64` iterators keeps
  similar stack traffic yet runs at 2.3x, 3.2x and about 6x the time of the explicit kernel.

**[I]** The cost of the iterator style is not a fixed overhead but sensitivity to the exact source
form. Explicit index expressions for neighbor accesses were best or close to best on both
architectures in every case; an emitter that keeps `gtfn_ir` iterators needs a per-compiler check
that they are scalar-replaced.

### (c) View vs pointer [M]

> **Correction (reconciliation round).** The `LayoutLeft` penalty on GPU scans is present with
> explicit bodies too (`tri` 1.07 vs 1.02, `tritmp` 1.10 vs 1.05); the table's "measured with the
> iterator body only" is wrong.

| comparison (large) | GPU | CPU |
|---|---|---|
| `vl` / `ptr`, explicit | 0.99-1.01 stencils and unstructured | 0.95-1.01 |
| `vl` / `ptr`, iterator | 0.94-1.01; scans 1.03-1.05 | 0.97-1.06, except `chain` 1.41 and chunked scan 0.63-0.83 (both stack-bound code) |
| `vs` / `ptr`, iterator | 1.00-1.02, except `lap2` 1.10 and `chain` local 1.44 (xlarge 1.92) | 0.98-1.01, except `chain` 2.0 and chunked scan 0.63-0.88 |
| `ptr64` / `ptr`, iterator | `lap2` 1.12-1.15, `chain` local 1.47-1.93, others <= 1.03 | rows <= 1.01 except `chain` 2.4-2.9 (random tables 1.3-1.4) |

`ptr` with `int` offsets is valid only below 2^31 elements. `vl` (`int` index, 64-bit offsets,
static unit stride) has no such limit and **is the safe shape the data support**. `ptr64` changes
index and offset width together, so its cost cannot be assigned to one of them.

### (d) Loop shape, tile, index type [M]

**GPU** (relative to `mdtlb`: tile {32,8,1} / {32,8}, `int`, `LaunchBounds<256>`)

| change | effect at large / xlarge |
|---|---|
| all-default `MDRangePolicy` | stencils 1.02-1.07 slower; local `v2e` 0.77 / 1.01; random tables 0.83-0.89 faster, but `chain` random 1.29 / 0.90 |
| `IndexType<int>` added to the default | no change (1.00): the default index is already 4 bytes |
| no `LaunchBounds` | no change (1.00) |
| flat `RangePolicy` | stencils 1.04-1.42 slower; unstructured anywhere from **0.34 to 2.19** (see section 4) |
| `naive` (`LayoutStride` View, iterator, all defaults) | 1.03-1.14 stencils; 0.79-1.92 unstructured |
| `Kokkos::fence` instead of device sync | no change |

On random numbering the block shape (how many levels and columns one block touches) decides the
time; gtfn's 32x8 is not the best choice there **[I]**. This only matters for badly ordered meshes.

**CPU** (relative to `rows`)

| change | effect, large |
|---|---|
| `MDRangePolicy<IndexType<int>>`, tiled left | 1.0-2.1x slower up to 36 threads; 0.9-1.9x at 72; explicit `chain` up to 2.4x |
| all-default `MDRangePolicy` | stencils 3-9x slower, local unstructured 1.5-2.8x; but **`chain` random 0.47-0.69 (faster)** |
| functor not copied into locals (`rowsnh`) | stencils none; unstructured explicit 1.12-1.40x slower, iterator 1.00-1.14x |
| `omp simd` on the inner loop | none |
| 64-bit index in row loops | none (except `chain` iterator, above) |
| 64-bit index in `MDRangePolicy` (stencils) | 0.58-0.95 of `int` up to 36 threads |

A negative control (`naive`) fails by factors of 3-9 on CPU stencils but only 3-14 % on GPU stencils.

### (e) Scans

**GPU** gtfn_gpu launches one kernel per sweep **[I, GridTools `run_column_stages`]**; its object
file has two device functions **[M]**. **[M]** Splitting vs merging in Kokkos changes at most 3 % at large sizes (split
slightly faster) and costs 5 % at the small size. Thread-per-column `RangePolicy` is 0.97-0.99 of
the tiled `MDRangePolicy`; pointer-walking columns equal indexed ones (0.99-1.02); the default tile
is 0.95-1.00; launch bounds do nothing. Per-call temporaries are 2.7-3.4x slower than a pool.

Where the `tritmp` miss comes from. **[M]** With field-shaped pool temporaries the Kokkos
two-kernel variant takes the same time as with caller fields (large 0.521 / 0.521 ms; xlarge
2.058 / 2.060 ms), while gtfn is 3.3 % (large: 0.523 -> 0.505 ms) and 4.2 % (xlarge:
2.044 -> 1.958 ms) **faster with its own temporaries**. The two Kokkos kernels use 50 / 32
registers per thread against gtfn's 48 / 32, with the same static global load and store counts
(forward 20 / 10, backward 58 / 29), so register pressure is an unlikely cause.

**Follow-up on the temporary layout** (two runs; every shape and size in section
2.4; pass criterion written down before the run). The pool
temporaries have the strides of the caller fields, and their first compute point lies 24 bytes
after a 256-byte boundary (caller fields: 152 bytes) **[M, printed addresses]**. gtfn allocates
compact `ni x nj x nk` temporaries whose first compute point is the first element of the allocation
**[I, headers]**. Same kernel code, pooled buffers, four layouts; offset = elements between a
256-byte boundary and the first compute point. Time / gtfn `tritmp`, large / xlarge **[M]** (median
of these two runs; the table in (a) takes all allocations, hence small differences for the pool
variants):

| temporaries | two kernels (gtfn-shaped) | merged | merged, flat range |
|---|---|---|---|
| caller-field strides, offset 3 (reproduces the pool) | 1.030 / 1.052 | 1.050 / 1.063 | 1.039 / 1.028 |
| compact, offset 3 | 0.999 / 1.035 | 1.026 / 1.048 | 1.012 / 1.009 |
| caller-field strides, offset 0 | 1.008 / 1.027 | 1.014 / 1.027 | 1.002 / 0.986 |
| compact, offset 0 (gtfn-like) | **1.004 / 1.030** | 1.007 / 1.018 | 0.997 / 0.979 |

* **The fixed criterion is not met [M].** The two-kernel compact, aligned variant is 1.004
  [1.001, 1.007] at large and 1.030 [1.026, 1.036] at xlarge. Layout closes most of the gap at
  large (1.031 -> 1.004) and about 40 % of it at xlarge (1.052 -> 1.030). **The rest is
  unexplained.**
* Consistency check **[M]**: that variant is 2.6 % (large) and 1.9 % (xlarge) faster than Kokkos
  `tri` of the same shape; gtfn gains 3.3 % and 4.2 %.
* Alignment of the first compute point is the largest single factor **[M]**: offset 0 instead of 3
  gains 2-4 % with caller-field strides and 0-3 % with compact strides. Compact strides gain 1-3 %
  at offset 3 and at most 1 % at offset 0.
* The flat-range merged kernel with aligned temporaries is at or just above gtfn at large and below
  it at xlarge (compact 0.997 [0.995, 1.000] / 0.979 [0.975, 0.984]; caller-field strides
  1.002 / 0.986). It was picked out after the run among twelve combinations, and it is not the
  launch structure of gtfn.
* Small size: layouts change 1-2 %, all variants stay below gtfn **[M]**.
* **[I]** For a backend: put the first compute point of every temporary on an aligned boundary and
  pad the rows so that all rows stay aligned (128 bytes sufficed here). Compact temporaries are
  row-aligned in these runs only because `ni` (32, 512, 1024) is a multiple of 32. The mechanism is
  plausibly fewer memory sectors per warp; nothing was profiled. Not tested: other sizes,
  `hdiff`/`tmpst`, aligned caller fields.

**CPU** gtfn_cpu also runs the two sweeps as separate passes over all columns **[I, from
`run_column_stages`]**. Ratios **[M]**:

| large `tri`, ratio to gtfn | 1 thr | 2 | 4 | 18 | 72 |
|---|---|---|---|---|---|
| column per iteration, two passes (gtfn-shaped) | 0.99 | 0.99 | 0.99 | 0.99 | 1.00 |
| column per iteration, merged | 0.99 | 0.99 | 1.02 | 0.97 | 0.85 |
| explicit, chunk 8 | 0.51 | 0.51 | 0.55 | 0.78 | 0.84 |
| explicit, chunk 16 | 0.65 | 0.66 | 0.69 | 0.89 | 0.87 |
| explicit, chunk 32 | **0.43** | **0.44** | **0.47** | **0.62** | 0.84 |
| explicit, chunk 64 | 0.52 | 0.52 | 0.57 | 0.83 | 0.87 |
| iterator, chunk 16 | 2.85 | 2.83 | 2.86 | 2.55 | 1.15 |

* Merging the sweeps alone gains nothing below 18 threads and about 15 % at 36-72.
* Explicit column chunks (merged sweeps) gain up to a factor 2.3 over one column per iteration at
  low thread counts. At 72 threads they gain nothing over the merged column loop (0.85) and 16 %
  over the two-pass one. At the small size the better of chunk 8 and 16 gives 0.33-0.44 of
  gtfn at 1-72 threads.
* The chunk size matters and not monotonically: at large, 16 is the worst of the explicit sizes
  at 1-4 threads. Cause unknown; chunk 8, 32 and 64 come from one allocation.
* With temporaries (`tritmp`, large) explicit chunk 32 gives 0.24 / 0.21 / 0.18 / 0.13 / 0.11 of
  gtfn at 1 / 2 / 4 / 18 / 72 threads, mostly because gtfn_cpu allocates and zero-fills on every call (see (f)).
* Iterator-style chunk sizes 8/16/64 are all slow (within 4 % of each other up to 18 threads); an
  iterator chunk of 16 takes 2.4-2.9x the time of the iterator column loop up to 18 threads on
  large `tri` and `tritmp`, 1.3x at 72.
* The earlier study's range of 0.27-0.48x for chunked columns on 1-8 threads is matched in part:
  explicit chunk 32 reaches 0.43-0.47 on large `tri` and 0.18-0.24 on `tritmp` (1-4 threads); the
  iterator form did not get there.
* Pointer-walking columns equal indexed ones; 64-bit offsets change nothing for one column per
  iteration.

### (f) How much of the CPU margin is gtfn's temporary handling?

[M], large size. `gtfn_fixed` = gtfn with single, uninitialised allocation per call.

| kernel, threads | Kokkos pool / gtfn | gtfn_fixed / gtfn | Kokkos pool / gtfn_fixed | Kokkos per-call alloc / gtfn_fixed |
|---|---|---|---|---|
| hdiff, 1 | 0.44 | 0.88 | 0.50 | 0.55 |
| hdiff, 8 | 0.19 | 0.36 | 0.52 | 0.66 |
| hdiff, 18 | 0.14 | 0.23 | 0.61 | 0.83 |
| hdiff, 72 | 0.10 | 0.17 | 0.62 | 1.01 |
| tmpst, 1 | 0.43 | 0.88 | 0.48 | 0.52 |
| tmpst, 8 | 0.19 | 0.39 | 0.48 | 0.63 |
| tmpst, 18 | 0.16 | 0.26 | 0.61 | 0.81 |
| tmpst, 72 | 0.13 | 0.21 | 0.62 | 1.02 |
| tritmp, 1 | 0.52 | 0.87 | 0.61 | 0.69 |
| tritmp, 8 | 0.33 | 0.58 | 0.57 | 0.70 |
| tritmp, 18 | 0.22 | 0.39 | 0.56 | 0.72 |
| tritmp, 72 | 0.13 | 0.41 (0.32-0.43) | 0.34 | 0.93 |

(`tritmp` uses the column-per-iteration iterator variant.)

* At 72 threads, `hdiff`: of the 10x margin, a factor 5.9 disappears when gtfn stops allocating
  twice and zero-filling; the remaining 1.6x needs the pool. With a per-call allocation Kokkos is
  no faster than `gtfn_fixed` (0.94-1.02 at 36-72 threads for `hdiff`/`tmpst`).
* At 1 thread, temporaries explain only 1.14x of a 2.3x margin; the kernel code explains about 2x.
* gtfn_cpu barely scales on `hdiff`: 177 ms (1 thread), 61.5 ms (8), 55.3 ms (72). That fits a
  serial allocation + zero-fill on every call **[I]**.
* A zero-initialised `Kokkos::View` per call (`.view`) is worse than `gtfn_fixed` for `hdiff` and
  `tmpst` from 18 threads on (1.04-1.31).
* **[I]** Both fixes (no zero-fill, pooling) can be made in gtfn too; this part of the margin is not
  an argument for Kokkos. No pooled gtfn variant was measured.

## 4. Limits, surprises, and how far to trust the numbers

### Not measured

* One node type, one toolchain (GCC 14.2, CUDA 12.9.41, Kokkos 5.2.2). No AMD GPU, no x86, no clang,
  no `-march=native`.
* No code generator exists: the Kokkos kernels are hand-written. A generated kernel can differ in
  source form, which section 3 (b) shows to matter.
* One kernel family per pattern, `float64` only; no halo exchange, no MPI, no programs with many
  small launches, so launch overhead of a real program is only covered by the small sizes.
* Kokkos temporaries come from a pool of full field-shaped padded buffers prepared in advance. No
  real pool or allocator of a backend was built, and **no pooled gtfn variant** was measured.
* Temporary layouts were varied only for GPU `tritmp`, at domain widths that are multiples of 32;
  not for `hdiff`/`tmpst`, the CPU or aligned caller fields. Why the two-kernel Kokkos solve still
  trails gtfn by 3 % at xlarge with gtfn-like temporaries is unknown (section 3 (e)).
* No lambda variant: every Kokkos kernel is a functor struct; `KOKKOS_LAMBDA` was not compared.
* Two-pass (`.split`) scans exist only for iterator and pointer-walking columns on GPU and for the
  iterator `tri` column loop on CPU; there is no explicit two-pass scan.
* At the small GPU size, host overhead per call and device time were not separated.
* CPU: one socket only (up to 72 threads); nothing across NUMA domains. `naive` and the
  default-`MDRangePolicy` variants were not run at 1 and 2 threads (left out for run time).
* No hardware counters and no GPU profile. Register counts and instruction counts are static.
  gtfn's CPU assembly was not examined.
* Compile time: only single translation units, 8 compiled in parallel. Kokkos one-kernel unit
  6.9-8.5 s, gtfn bindings unit 3.5-6.3 s **[M]**; they do not contain the same things, so this is
  not a backend-level compile-time comparison.
* `ptr64` changes index and offset width together.

### Surprising or suspicious

All reproduced in two allocations unless marked otherwise **[M]**:

1. **GPU, random numbering, `xlarge`:** a flat `RangePolicy` runs `v2e` in **0.35x** gtfn's time
   (3.96 vs 11.5 ms); at `large` the same variant is 0.88x. For `chain` it is 1.00x at `xlarge` but
   2.19x at `large`. The all-default `MDRangePolicy` is 0.83-0.91x at `xlarge` where the gtfn-like
   tile is 1.00x. Locality across levels (pages, cache lines touched per block) is a plausible
   cause **[I, no profile]**. Gtfn-shaped launches are not optimal on badly ordered meshes.
2. GPU `tritmp`: gtfn is faster with its temporaries than with caller fields. Kokkos with
   field-shaped temporaries is not; with aligned ones it gains too, but less than gtfn in the
   two-kernel form (section 3 (e)).
3. CPU `chain`, random numbering: the all-default `MDRangePolicy` is 0.47-0.69x of the row loop,
   although it is 1.5-1.9x slower on local numbering (4-36 threads: one allocation).
4. CPU explicit chunk sizes on large `tri`: 32 fastest, 16 slowest (chunk 8, 32, 64: one
   allocation).
5. `gtfn_fixed` on `tritmp` at 72 threads moves between 0.32x and 0.43x of gtfn across three
   allocations; this is the largest spread in the data (28 % on the ratio).
6. gtfn's GPU timings at the small size move by up to 12 % between allocations (large: under 2 %).
7. gtfn_cpu `hdiff` stops scaling after 8 threads (61.5 ms at 8 threads, one allocation; 55.3 ms
   at 72).
* CPU assembly explains which variants are slow but not by how much (section 3 (b)).
* (Correction, reconciliation round: the claim below came from one first-round group and was withdrawn by it as a general claim.)
* The study claim that an iterator body in a column-chunk executor matches the explicit one (about
  0.27x gtfn_cpu, measured by its authors on x86 with another iterator header) **was not
  reproduced** with this iterator implementation on this compiler.

### How far to trust

* Correctness: every timed variant matched gtfn bit for bit **[M]**.
* Within one allocation the 95 % intervals are usually +-1-2 %. Between allocations **[M]**
  (cells measured twice or more): GPU `large` ratios differ by 0.3 % in the
  median (worst 2.7 %), GPU `xlarge` (52 cells) by 0.4 % (worst 2.5 %); CPU at 72 threads by 2-3 %
  (worst 28 %, see above); CPU 1-18 threads (21 scan cells each) by 1-5 % (worst 14 %);
  **GPU small by 6 % in the median (worst 14 %)** because gtfn's own small timings move
  (the two follow-up runs agree within 1 % there).
* Cells marked `*` come from a single allocation (most CPU cells at 1-36 threads). All runs used
  the same node, so node-to-node variation is not covered.
* Treat differences below about 5 % as real only where both allocations agree and the interval
  excludes 1.0; treat CPU `lap`/`lap2` at 72 threads as bandwidth-limited.
