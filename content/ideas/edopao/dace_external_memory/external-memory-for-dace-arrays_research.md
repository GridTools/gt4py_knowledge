---
title: External workspace memory for DaCe temporary arrays — implementation notes
description: "Detailed design notes for external-memory mode in the gt4py.next DaCe backend: allocator contract, validation, code-generator integration, and testing plan."
author: edopao
tags: [dace, backend, gpu, memory, temporary-arrays, workspace, implementation, testing]
created: 2026-07-08
---

This is an implementation appendix for
[[external-memory-for-dace-arrays|External workspace memory for DaCe temporary arrays]].

## 1. The four-mode unification

The legacy flags `make_persistent` and `gpu_memory_pool` can be replaced by a
single enum because their Cartesian product maps cleanly onto the four modes:

| `make_persistent` | `gpu_memory_pool` | Equivalent mode | Notes |
| --- | --- | --- | --- |
| `False` | `False` | `scoped` | Plain alloc/free per SDFG call. |
| `False` | `True` | `pool` | Stream-ordered async pool (current default). |
| `True` | `False` | `persistent` | Allocate once, live for process. |
| `True` | `True` | rejected | Factory rejects any combination of old flags with the new enum. |

This removes the need to validate combinations of two booleans and makes the
choice explicit in the SDFG fingerprint.

## 2. Code locations to touch

| Area | Files (current `main`) | Change |
| --- | --- | --- |
| Auto-optimization | `src/gt4py/next/program_processors/runners/dace/transformations/auto_optimize.py` | Replace `make_persistent: bool` / `gpu_memory_pool: bool` with `transient_memory_mode: Literal["scoped", "pool", "persistent", "external"]`. Add `gt_apply_external_memory()`. |
| GPU utilities | `.../dace/transformations/gpu_utils.py`, `utils.py` | Merge mempool/persistent/external passes into one transient-lifetime pass. |
| Backend factory | `.../runners/dace.py` | Add `external_memory_allocator` parameter; forward through factory chain. |
| Compilation step | `.../dace/compilation.py` | Add allocator field; forward it to the compilation artifact. |
| Compiled program | `.../dace/compiled_program.py` | Store allocator callable; call it in `construct_arguments()` to obtain workspaces, validate sizes, and `set_workspace()`. |
| Translation | `.../dace/translation.py` | Ensure external mode sets `AllocationLifetime.External` on generated transients.
| Tests | `tests/next_tests/unit_tests/dace_tests/test_dace_backend.py`, `test_dace_compilation.py` | Parameterize all four memory modes; add external-memory tests. |

## 3. Workspace allocation at run time

`CompiledDaceProgram` stores only the allocator callable; no workspace buffer is
serialized. At run time `construct_arguments` queries the required size and asks
the allocator for a buffer of at least that size.

```python
@dataclasses.dataclass(frozen=True)
class CompiledDaceProgram:
    ...
    external_memory_allocator: Callable[[core_defs.DeviceType, int], np.ndarray | cp.ndarray] | None
    ...
```

The allocator must be pickleable (e.g. a module-level function) so that
`CompiledDaceProgram` can be serialized.

## 4. Validation checklist

In `CompiledDaceProgram.construct_arguments`:

1. Call `csdfg.get_workspace_sizes()`.
2. For each required storage type (`CPU_Heap`, `GPU_Global`), call the stored
   allocator with the required size and assert `buffer.nbytes >= required_size`.
3. For GPU, assert the buffer's device matches the execution device.
4. Call `csdfg.set_workspace(storage_type, buffer)` before building the call
   argument vectors.

## 5. Testing plan

- **Unit:** build a tiny SDFG with `AllocationLifetime.External`, assert no
  `new`/`delete` in generated code, assert `set_workspace` is present.
- **Integration:** run a gt4py program through the DaCe backend with each of
  the four `transient_memory_mode` values and check correctness.
- **Regression:** ensure `scoped`, `pool`, and `persistent` still work
  unchanged.
- **Serialization:** pickle/unpickle `CompiledDaceProgram` and verify the
  allocator callable is restored; after unpickling, run the program and verify
  the allocator is invoked to create the workspace.

## 6. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Workspace too small for a later SDFG | Validate sizes at run time; fail with a clear message. |
| Concurrent SDFG execution corrupts workspace | Document single-stream constraint; add debug-only synchronization asserts if possible. |
| GPU device mismatch | Validate `buffer.device` in `construct_arguments`. |
| Pickle failure | Store only the allocator callable; require it to be pickleable. |
| HIP codegen differences | Emit the synchronization snippet behind the same target guard DaCe uses for CUDA vs HIP. |
| Old flags mixed with new enum | Reject in factory; provide migration message. |
