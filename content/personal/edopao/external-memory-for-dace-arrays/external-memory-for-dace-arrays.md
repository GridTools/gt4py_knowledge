---
title: External workspace memory for DaCe temporary arrays
description: "Use application-provided buffers for gt4py.next DaCe temporary arrays via AllocationLifetime.External, avoiding cudaMallocAsync/cudaFreeAsync overhead while staying within GPU memory limits."
author: edopao
tags: [dace, backend, gpu, memory, temporary-arrays, workspace, cuda, hip, mempool, persistent, external, allocation, icon4py, performance]
created: 2026-07-08
status: final
---

> **TL;DR** Add an external memory mode to the gt4py.next DaCe backend where
> temporary arrays are backed by an **application-provided workspace buffer**
> (`dace.AllocationLifetime.External`). This removes the per-SDFG-call
> `cudaMallocAsync`/`cudaFreeAsync` (or HIP equivalent) overhead without
> requiring every SDFG's temporaries to be persistently resident at once.

## Problem / motivation

A typical weather/climate driver calls many gt4py programs inside a time loop.
When each program is compiled with the DaCe backend it becomes one SDFG, and
each SDFG may contain many maps → many GPU kernels. Inside an SDFG, **transient
arrays** carry intermediate results from one kernel to the next.

DaCe gives transients three lifetime options. gt4py currently exposes four
practical modes, selected by overlapping boolean flags:

| Mode | Lifetime | Allocation API |
| --- | --- | --- |
| `scoped` | SDFG scope | regular `cudaMalloc` / `cudaFree` |
| `pool` (default) | SDFG scope | `cudaMallocAsync` / `cudaFreeAsync` |
| `persistent` | Process | allocate once |
| `external` | External workspace | caller-provided buffer |

On CPU the `scoped`/`pool` distinction does not exist; CPU transients always
use heap allocation at SDFG scope unless `persistent` or `external` is selected.

The current default uses **stream-ordered allocation** on the default stream.
That is much cheaper than plain `cudaMalloc`/`cudaFree`, but on AMD the async
HIP APIs have larger overhead, and even on NVIDIA the cost is non-zero at scale.

We need a mode that:

1. avoids per-call allocation overhead,
2. keeps peak memory bounded to **one SDFG's** transient set, not all SDFGs
   combined, and
3. fits naturally into the existing gt4py.next DaCe toolchain without breaking
   the legacy pool or persistent modes.

## Proposal

Introduce an **external workspace** mode. The application provides one
pre-allocated buffer per device (CPU heap, GPU global). gt4py maps each SDFG's
transient arrays into a slice of that buffer using DaCe's
`AllocationLifetime.External`.

```python
import dace
import numpy as np

sdfg = dace.SDFG("external_memory_example")
sdfg.add_array("a", [M, N], dace.float64)
sdfg.add_array("e", [M, N], dace.float64)
sdfg.add_transient(
    "t0", [M, N], dace.float64,
    lifetime=dace.AllocationLifetime.External,
)
```

After compilation the caller queries the required workspace size, hands in a
buffer, and runs:

```python
csdfg.initialize(a=a, e=e)
sizes = csdfg.get_workspace_sizes()          # {StorageType.CPU_Heap: ...}
wsp = np.zeros(sizes[StorageType.CPU_Heap] // 8)  # bytes -> float64
csdfg.set_workspace(StorageType.CPU_Heap, wsp)
csdfg(a=a, e=e)
```

The DaCe-generated code no longer allocates or frees `t0`; it indexes into the
external workspace.

### High-level gt4py integration

Add `external_memory_allocator` to the DaCe backend factory:

```python
from gt4py.next.program_processors.runners.dace import (
    DaCeBackendFactory,
)

backend = DaCeBackendFactory(
    gpu=True,
    auto_optimize=True,
    otf_workflow__compilation__external_memory_allocator=my_allocator,
)
```

`my_allocator(device_type: core_defs.DeviceType, size: int)` returns a writable
buffer with at least `size` bytes:

- CPU: any writable buffer (NumPy ndarray, …).
- GPU: a device array (CuPy or ROCm equivalent).

The allocator is stored as a callable in the backend factory and in
`CompiledDaceProgram`, not as the array itself, so the objects remain
serializable as long as the callable itself is pickleable (e.g. a module-level
function). The actual workspace buffer is materialized at run time by calling
the allocator with the size returned from `csdfg.get_workspace_sizes()`.

### Four-mode memory policy

The current flags are two booleans (`make_persistent`, `gpu_memory_pool`) that
select among overlapping behaviors. It is cleaner to replace them with one
explicit policy that covers all four meaningful choices for transient lifetime
and allocation strategy:

```python
transient_memory_mode: Literal["scoped", "pool", "persistent", "external"] = "pool"
```

| Mode | Lifetime | Allocation API | When to use |
| --- | --- | --- | --- |
| `scoped` | SDFG scope | `cudaMalloc` / `cudaFree` | Baseline; no pool, no persistent state. |
| `pool` (default) | SDFG scope | `cudaMallocAsync` / `cudaFreeAsync` | Current default; low per-call overhead. |
| `persistent` | Process | allocate once | When a single SDFG dominates and memory permits. |
| `external` | External workspace | caller-provided buffer | Multi-program drivers where persistent would exceed GPU memory. |

Exactly one mode is active. The `make_persistent` and `gpu_memory_pool` flags
can be removed; the policy is encoded in the array lifetimes and therefore in
the build fingerprint, so the cache distinguishes all four cases naturally.

### Post-processing placement

External lifetime should be applied as a post-processing transformation, like
`gt_gpu_apply_mempool()` and `gt_make_transients_persistent()`. In fact these
three passes touch the same set of arrays and can be merged into a single
"choose transient lifetime" pass.

### Storage and serialization

`DaCeCompilationArtifact` and `CompiledDaceProgram` carry the allocator
callable:

```python
external_memory_allocator: Callable[[core_defs.DeviceType, int], np.ndarray | cp.ndarray] | None
```

No workspace buffer is serialized. In `CompiledDaceProgram.construct_arguments`,
the allocator is called with the size returned by `csdfg.get_workspace_sizes()`
to obtain the CPU and/or GPU workspace buffers, which are then validated and
installed with `csdfg.set_workspace()` before `fast_call()`.

### Optional: stream synchronization for multi-stream SDFGs

This project deliberately keeps all SDFGs on a single execution stream; two
SDFGs never run concurrently, so the same workspace buffer can safely be reused.

However, **inside** one SDFG DaCe may schedule maps on multiple internal
streams. As long as every internal stream is synchronized back to the default
stream before the SDFG returns, this is safe with external memory. A small
CUDA/HIP tasklet at SDFG exit records events on each internal stream and waits
on them from the default stream:

```cpp
for (int i = 0; i < __state->context.num_streams; i++) {
    if (__state->gpu_context->internal_streams[i] != cudaStreamDefault) {
        cudaEventRecord(dace_external_events[i], __state->gpu_context->internal_streams[i]);
        cudaStreamWaitEvent(cudaStreamDefault, dace_external_events[i], 0);
    }
}
```

The events are created once during SDFG initialization and destroyed during
cleanup, so they are not leaked on every SDFG invocation:

```python
sdfg.append_global_code("cudaEvent_t *dace_external_events = nullptr;", location="cuda")
sdfg.append_init_code(
    """
    dace_external_events = new cudaEvent_t[__state->context.num_streams];
    for (int i = 0; i < __state->context.num_streams; i++) {
        cudaEventCreate(&dace_external_events[i]);
    }
    """,
    location="cuda",
)
sdfg.append_exit_code(
    """
    for (int i = 0; i < __state->context.num_streams; i++) {
        cudaEventDestroy(dace_external_events[i]);
    }
    delete[] dace_external_events;
    dace_external_events = nullptr;
    """,
    location="cuda",
)
```

The HIP target uses the same structure with `hipEvent_t`, `hipEventCreate`,
`hipEventRecord`, `hipEventDestroy`, and `hipStreamWaitEvent`.

These snippets are compatible with DaCe code generation: `append_global_code`,
`append_init_code`, and `append_exit_code` emit host-side code in the generated
CUDA/HIP translation unit, and `__state->context.num_streams` together with
`__state->gpu_context->internal_streams` reflects the runtime stream layout.

This enables the new multi-stream DaCe code generator
(`gt4py/pull/2591`) without risking workspace corruption. It is a secondary
goal; the primary goal is external allocation itself.

## Alternatives considered

1. **Keep the pool default.** Already the status quo. It works but leaves
   alloc/free overhead and blocks safe intra-SDFG multi-stream execution.
2. **Make everything persistent.** Rejected because the combined transient
   memory of all SDFGs exceeds GPU memory for realistic multi-program drivers.
3. **One workspace per SDFG, allocated by gt4py.** Rejected because the
   application often already owns a memory-pool or slab allocator; letting the
   caller provide the buffer avoids double bookkeeping and keeps gt4py out of
   long-lived GPU memory management.

## Open questions / conflicts

- **Device portability.** The current write-up centers on CUDA/CuPy. ROCm
  support needs the same allocator contract but returns a ROCm array object.
  The generated synchronization code must be emitted for both `cuda` and `hip`
  targets.
- **Buffer validation.** We should assert that a GPU workspace buffer is really
  resident on the device the SDFG will use, and that CPU buffers are writable.
  CuPy's `arr.data.device` (or similar) can be checked.
- **Size stability.** `construct_arguments` queries the required workspace size
  via `csdfg.get_workspace_sizes()` and asks the allocator for a buffer of at
  least that size. If the allocator cannot satisfy the request it should raise a
  clear error.
- **Multi-SDFG sharing of one workspace.** The shaped project suggests sizing
  the buffer for the largest SDFG and reusing it across all SDFGs. This only
  works if SDFGs are executed sequentially on the same stream. We should
  document this constraint explicitly and consider whether a per-SDFG workspace
   slice registry is ever needed.
- **Interaction with memory modes.** `transient_memory_mode` replaces the legacy
  `make_persistent` and `gpu_memory_pool` booleans entirely. The factory should
  reject any combination of the old flags with the new enum to avoid ambiguous
  configurations. Migration: `make_persistent=True` → `"persistent"`;
  `gpu_memory_pool=False` → `"scoped"`; `gpu_memory_pool=True` (default) →
  `"pool"`.
- **Lifetime of CPU vs GPU transients.** For CPU targets the external storage is
  `StorageType.CPU_Heap`; for GPU targets it is `StorageType.GPU_Global`. A
  single workspace buffer per storage type is sufficient because CPU and GPU
  transients are never interleaved in the same SDFG execution path.
- **Allocator picklability.** `CompiledDaceProgram` stores only the allocator
  callable. The allocator must therefore be pickleable; documenting this contract
  and failing clearly at build time if a closure or lambda is supplied avoids
  obscure serialization errors later.

## Related work

- DaCe external-memory test that motivates the approach: [`external_memory_test.py`](external_memory_test.py)
  in this directory.
- gt4py layering and build-cache direction:
  [[personal/egparedes/layered-architecture|A layered architecture for gt4py with
  enforced public/internal decoupling]] — this change lives in the
  infrastructure/runner layer; it should respect the core/infrastructure split
  and not leak IR-specific concepts into the allocator contract.
- New DaCe code generator enabling intra-SDFG multi-stream execution:
  `gt4py/pull/2591`.
- Frontend/embedded field-data redesign
  ([[personal/havogt/field-data-protocol|A FieldData protocol]]) and scan redesign
  ([[personal/havogt/scan-redesign|Redesign of the vertical scan]]) are orthogonal;
  they affect what temporaries exist, not how the DaCe backend stores them.

## Appendix

- [[external-memory-for-dace-arrays_shaped-project|Shaped project brief]]
