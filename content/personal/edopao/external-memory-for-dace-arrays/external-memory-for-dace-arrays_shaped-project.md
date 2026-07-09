---
title: Gt4Py[next] External Memory for DaCe Temporary Arrays
author: edopao
tags: [dace, backend, gpu, memory, temporary-arrays, workspace, shaped-project]
created: 2026-07-08
status: draft
---

# Gt4Py[next] External Memory for DaCe Temporary Arrays

- Shaped by: Edoardo
- Appetite (FTEs, weeks):
- Developers: <!-- Filled in at the betting table unless someone is specifically required here -->

## Problem

A typical application will call several GT4Py programs at runtime.
When running on the Dace backend, each program is lowered to one SDFG. Each SDFG can contain several `Maps`, therefore several GPU kernels.
Transient arrays are used to store intermediate fields within one SDFG and use the data produced by one GPU kernel as input to another GPU kernel.
In current design, the lifetime of these transient arrays is the SDFG scope.

Each GT4Py program is typically called in a time loop, in weather simulations, so the SDFG is executed multiple times.
The most efficient deployment, from the perspective of execution time, would be to allocate the transient arrays with the `persistent` lifetime. That means all intermediate fields are allocated at initialization time, before the first SDFG call, and are never deallocated until the application exits.
However, for real use cases with multiple GT4Py programs, the total memory required by all SDFGs for intermediate fields does not fit in the available GPU memory. The current design of the SDFG lowering was targeting this limitation: the intermediate fields have the SDFG scope lifetime, which means they are allocated and deallocated each time the SDFG is called, in order to keep the total memory utilization across all SDFGs within the limit.
However, to limit the cost of allocation/deallocation, we use Stream Ordered Memory Allocation with the aynchronous APIs (`cudaMallocAsync` and `cudaFreeAsync`). The current design allocates memory from the memory pol on the default stream, and it works with mutiple GPU kernels because all kernels are executed on the queue associated with the default stream.

The current design has 2 limitations: 1. the equivalent HIP APIs on the AMD runtime have a larger overhead and 2. although much lighter that regular `cudaMalloc` and `cudaFree`, the asynchrounous APIs still have some overhead.

## Proposed solution

In this project we want to implement a solution based on the `external` lifetime of transient arrays, which means to configure the SDFG with memory allocated externally.
Below is an example SDFG and dace test:
```python
def test_external_memory_2d_builder():
    """Tests external memory allocation with 2D arrays built via the SDFG
    Builder API.

    The SDFG computes:
        t0 <= a + b
        t1 <= t0 + c
        t2 <= t1 + d
        e  <= t2

    where ``t0``, ``t1`` and ``t2`` are transient arrays with
    ``dace.AllocationLifetime.External`` (i.e. their memory is provided by the
    caller through the external-memory workspace), while ``a``, ``b``, ``c``,
    ``d`` and ``e`` are regular global arrays. All arrays are 2D.
    """
    M, N = 4, 6

    sdfg = dace.SDFG('external_memory_2d_builder')

    # Global arrays (inputs and output)
    sdfg.add_array('a', [M, N], dace.float64)
    sdfg.add_array('b', [M, N], dace.float64)
    sdfg.add_array('c', [M, N], dace.float64)
    sdfg.add_array('d', [M, N], dace.float64)
    sdfg.add_array('e', [M, N], dace.float64)

    # Transient arrays with external memory
    sdfg.add_transient('t0', [M, N], dace.float64, lifetime=dace.AllocationLifetime.External)
    sdfg.add_transient('t1', [M, N], dace.float64, lifetime=dace.AllocationLifetime.External)
    sdfg.add_transient('t2', [M, N], dace.float64, lifetime=dace.AllocationLifetime.External)

    # All computations happen in a single state. The four maps are chained
    # through access nodes for the external transients t0, t1 and t2.
    state = sdfg.add_state('compute')

    # t0 = a + b
    me0, mx0 = state.add_map('map_t0', dict(i=f'0:{M}', j=f'0:{N}'))
    t0_tasklet = state.add_tasklet('t0_op', {'_a', '_b'}, {'_out'}, '_out = _a + _b')
    state.add_memlet_path(state.add_read('a'), me0, t0_tasklet, dst_conn='_a', memlet=dace.Memlet('a[i, j]'))
    state.add_memlet_path(state.add_read('b'), me0, t0_tasklet, dst_conn='_b', memlet=dace.Memlet('b[i, j]'))
    t0_acc = state.add_access('t0')
    state.add_memlet_path(t0_tasklet, mx0, t0_acc, src_conn='_out', memlet=dace.Memlet('t0[i, j]'))

    # t1 = t0 + c
    me1, mx1 = state.add_map('map_t1', dict(i=f'0:{M}', j=f'0:{N}'))
    t1_tasklet = state.add_tasklet('t1_op', {'_a', '_b'}, {'_out'}, '_out = _a + _b')
    state.add_memlet_path(t0_acc, me1, t1_tasklet, dst_conn='_a', memlet=dace.Memlet('t0[i, j]'))
    state.add_memlet_path(state.add_read('c'), me1, t1_tasklet, dst_conn='_b', memlet=dace.Memlet('c[i, j]'))
    t1_acc = state.add_access('t1')
    state.add_memlet_path(t1_tasklet, mx1, t1_acc, src_conn='_out', memlet=dace.Memlet('t1[i, j]'))

    # t2 = t1 + d
    me2, mx2 = state.add_map('map_t2', dict(i=f'0:{M}', j=f'0:{N}'))
    t2_tasklet = state.add_tasklet('t2_op', {'_a', '_b'}, {'_out'}, '_out = _a + _b')
    state.add_memlet_path(t1_acc, me2, t2_tasklet, dst_conn='_a', memlet=dace.Memlet('t1[i, j]'))
    state.add_memlet_path(state.add_read('d'), me2, t2_tasklet, dst_conn='_b', memlet=dace.Memlet('d[i, j]'))
    t2_acc = state.add_access('t2')
    state.add_memlet_path(t2_tasklet, mx2, t2_acc, src_conn='_out', memlet=dace.Memlet('t2[i, j]'))

    # e = t2
    me3, mx3 = state.add_map('map_e', dict(i=f'0:{M}', j=f'0:{N}'))
    e_tasklet = state.add_tasklet('e_op', {'_in'}, {'_out'}, '_out = _in')
    state.add_memlet_path(t2_acc, me3, e_tasklet, dst_conn='_in', memlet=dace.Memlet('t2[i, j]'))
    state.add_memlet_path(e_tasklet, mx3, state.add_write('e'), src_conn='_out', memlet=dace.Memlet('e[i, j]'))

    # Test that no heap allocation is generated for the external transients
    code = sdfg.generate_code()[0].clean_code
    assert 'new double' not in code
    assert 'delete[]' not in code
    assert 'set_workspace' in code

    # Compile and verify the external workspace size
    csdfg = sdfg.compile()

    a = np.random.rand(M, N)
    b = np.random.rand(M, N)
    c = np.random.rand(M, N)
    d = np.random.rand(M, N)
    e = np.zeros((M, N))

    csdfg.initialize(a=a, b=b, c=c, d=d, e=e)
    sizes = csdfg.get_workspace_sizes()
    assert sizes == {dace.StorageType.CPU_Heap: 3 * M * N * 8}

    # Provide the external workspace and run
    wsp = np.zeros(3 * M * N)
    csdfg.set_workspace(dace.StorageType.CPU_Heap, wsp)
    csdfg(a=a, b=b, c=c, d=d, e=e)

    # Verify the output and the intermediate results stored in the workspace
    ref_t0 = a + b
    ref_t1 = ref_t0 + c
    ref_t2 = ref_t1 + d

    assert np.allclose(e, ref_t2)
    assert np.allclose(wsp[:M * N].reshape(M, N), ref_t0)
    assert np.allclose(wsp[M * N:2 * M * N].reshape(M, N), ref_t1)
    assert np.allclose(wsp[2 * M * N:3 * M * N].reshape(M, N), ref_t2)
```

The interesting section is this:
```python
    csdfg.initialize(a=a, b=b, c=c, d=d, e=e)
    sizes = csdfg.get_workspace_sizes()
    assert sizes == {dace.StorageType.CPU_Heap: 3 * M * N * 8}

    # Provide the external workspace and run
    wsp = np.zeros(3 * M * N)
    csdfg.set_workspace(dace.StorageType.CPU_Heap, wsp)
```

The SDFG API allows to get the requirement of external CPU heap memory.
We would need to customize the DaCe backend with an optional allocator callable. The allocator is provided by the application (ICON4Py) and invoked by GT4Py at run time to obtain the workspace buffer for the compiled SDFG.
The provided memory allocation should be large enough to store all intermediate fields in one SDFG, therefore it must be sized for the largest SDFG.
The code in `compilation.py` should probably do something like:
```python
class CompiledDaceProgram:
    ...
    # The `CompiledDaceProgram` object stores only the allocator callable so it
    # remains picklable. The actual workspace buffers are allocated at run time.
    external_memory_allocator: Callable[[core_defs.DeviceType, int], np.ndarray | cp.ndarray] | None

    def construct_arguments(self, **kwargs: Any) -> None:
        """
        This function will process the arguments and store the processed argument
        vectors in `self.csdfg_args`, to call them use `self.fast_call()`.
        """
        if (sizes := self.sdfg_program.get_workspace_sizes()):
            self.sdfg_program.initialize(**kwargs)
            for memory_type in [dace.StorageType.CPU_Heap, dace.StorageType.GPU_Global]:
                if (size := sizes.get(memory_type, 0)) == 0:
                    continue
                wsp = self.external_memory_allocator(memory_type, size)
                if wsp.nbytes < size:
                    raise ValueError(f"The SDFG memory requirement for {memory_type} is {size}, provided external memory is {wsp.nbytes} bytes")
                self.sdfg_program.set_workspace(memory_type, wsp)

        with dace.config.set_temporary("compiler", "allow_view_arguments", value=True):
            csdfg_argv, csdfg_init_argv = self.sdfg_program.construct_arguments(**kwargs)

        self.csdfg_argv = [*csdfg_argv]
        self.csdfg_init_argv = csdfg_init_argv
```

Note that this design proposal, to reuse a preallocated memory region for temporary arrays across all SDFGs, only works if all SDFGs run on the same GPU stream (e.g. the default stream).
Using multiple streams could result in incorrect result, because the same memory could be accessed by GPU kernels running concurrently on different streams.
Therefore, **support for running multiple SDFGs concurrently on different streams is outside the scope of this project**.

On the contrary, this design enables a different kind of improvement, still related to multi-stream execution.
As said above, two SDFGs will never run concurrently in the proposed design. However, one SDFG usually contains several Maps, which result in several GPU kernels.
As long as all used streams synchronize on the default stream, before leaving the SDFG, the other internal Maps can be scheduled on different streams by DaCe.
The synchronization can be implemented by some custom code that records an event on each used stream and then waits for those events on the default stream.

By using external memory, we no longer depend on `cudaMallocAsync` and `cudaFreeAsync`, so we do not have the problem of mapping GPU kernels to the stream memory pools.
Therefore, we can enable multi-stream execution in the [new DaCe code generator](github.com/GridTools/gt4py/pull/2591).
However, this can be implemented as a secondary task. The primary objective of this project should be the external memory allocation to remove the overhead of `cudaMallocAsync` and `cudaFreeAsync`, or the equivalent HIP functions.

Note that the external mode should be added as a post-processing step analogous to `gt_gpu_apply_mempool()` and `gt_make_transients_persistent()`.
We could probably merge these functions.

Two design points require special attention, which wil be discussed in separate sections below:
- **Storage for external memory.** How should the external memory array be stored? Depending on the cpu/gpu target, it should be a numpy/cupy array.
- **Stream synchronization.** Prototype the SDFG begin/exit code to synchronize all streams on the default stream.

### Storage for external memory

The storage for external memory should come from the application. The backend
factory and the compiled program store a callable allocator rather than the
buffer itself, so both remain serializable as long as the allocator is
pickleable. The allocator is invoked at run time with the size returned by
`get_workspace_sizes()` to obtain the actual workspace buffer.

Somewhere in application code:
```
def external_memory_allocator(device_type: core_defs.DeviceType, size: int) -> np.ndarray | cp.ndarray:
   ...

backend_descr["external_memory_allocator"] = external_memory_allocator
custom_backend = backend_factory(**backend_descr)
```

The function `external_memory_allocator()` will return an array with at least `size` bytes:
- CPU: any writable buffer.
- GPU: a device array (CuPy/ROCm).

If the requested size is not available, the function should raise an exception.

In GT4Py, we can setup the dace backend:
```
DaCeBackendFactory(
    gpu=gpu,
    auto_optimize=auto_optimize,
    ...
    otf_workflow__compilation__external_memory_allocator=external_memory_allocator,
)
```

In compilation.py:
```
@dataclasses.dataclass(frozen=True)
class DaCeCompiler():
    ...
    external_memory_allocator: Callable[[core_defs.DeviceType, int], np.ndarray | cp.ndarray] | None = None,

    def __call__(
        self,
        inp: stages.ExtensionSource[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
    ) -> DaCeCompilationArtifact:
        with gtx_wfdcommon.dace_context(
            device_type=self.device_type,
            cmake_build_type=self.cmake_build_type,
        ):
            ...
            with locking.lock(sdfg_build_folder):
                sdfg.compile(validate=False, return_program_handle=False)

        return DaCeCompilationArtifact(
            build_folder=sdfg_build_folder,
            library_path=library_path,
            sdfg_json=json.dumps(inp.program_source.source_code),
            binding_source_code=inp.binding_source.source_code,
            bind_func_name=self.bind_func_name,
            device_type=self.device_type,
            external_memory_allocator=self.external_memory_allocator,
        )

```

The `AllocationLifetime.External` attribute is stored on the array descriptors, inside the SDFG, therefore it is part of the source fingerprint. This ensures the build cache distinguishes pool/persistent/external modes.

### Stream synchronization

Stream synchronization is only applicable to SDFGs with GPU schedule.
In this case, we want to terminate the SDFG with a C++ tasklet that records an
event on each internal stream and makes the default stream wait on it:

```
// Record an event in each stream and make the default stream wait on it
for (int i = 0; i < __state->context.num_streams; i++) {
    if (__state->gpu_context->internal_streams[i] != cudaStreamDefault) {
        cudaEventRecord(dace_external_events[i], __state->gpu_context->internal_streams[i]);
        cudaStreamWaitEvent(cudaStreamDefault, dace_external_events[i], 0);
    }
}
```

The events must be created once during SDFG initialization and destroyed during
cleanup to avoid leaking one event per SDFG invocation. Declare a global event
array, create the events in the init routine, and destroy them in the exit
routine:

```
sdfg.append_global_code("cudaEvent_t *dace_external_events = nullptr;", location="cuda")
sdfg.append_init_code("""
    dace_external_events = new cudaEvent_t[__state->context.num_streams];
    for (int i = 0; i < __state->context.num_streams; i++) {
        cudaEventCreate(&dace_external_events[i]);
    }
""", location="cuda")
sdfg.append_exit_code("""
    for (int i = 0; i < __state->context.num_streams; i++) {
        cudaEventDestroy(dace_external_events[i]);
    }
    delete[] dace_external_events;
    dace_external_events = nullptr;
""", location="cuda")
```

This code example is for the CUDA target; the HIP target uses `hipEvent_t`,
`hipEventCreate`, `hipEventRecord`, `hipEventDestroy`, and `hipStreamWaitEvent`.
The event array is sized at initialization time using the actual number of
internal streams, so it does not require a compile-time constant `N`.

These snippets are compatible with DaCe code generation: `append_global_code`,
`append_init_code`, and `append_exit_code` emit host-side code in the generated
CUDA/HIP translation unit, and `__state->context.num_streams` together with
`__state->gpu_context->internal_streams` reflects the runtime stream layout.

## Related code changes

We want to add this new mechanism, but keep the legacy modes.
Therefore, we introduce a four-state memory policy in `gt_auto_optimize` and enforce exactly one mode:
```
transient_memory_mode: Literal["scoped", "pool", "persistent", "external"] = "pool"
```

This means that when customizing the compilation step of the dace workflow, we also need to customize the `optimization_args` of the translation step.

The signature of the function that customizes the dace workflow should be updated:
```
make_dace_backend(..., external_memory_allocator=...)
```

where `external_memory_allocator` is default `None`.


## [AI-generated] Recommended implementation outline

1. Auto-optimize transformation
  - Replace make_persistent: bool and gpu_memory_pool: bool with transient_memory_mode: Literal["scoped", "pool", "persistent", "external"] = "pool" in gt_auto_optimize().
  - Add gt_apply_external_memory(sdfg, storage) helper in gpu_utils.py or utils.py.
  - Update _gt_auto_post_processing() to dispatch the four modes.
2. Backend factory
  - Add external_memory_allocator to make_dace_backend().
  - Forward it through DaCeBackendFactory to DaCeCompilationStepFactory and DaCeCompiler.
3. Compilation
  - Add `external_memory_allocator` to `DaCeCompilationArtifact` and `CompiledDaceProgram`.
  - In `CompiledDaceProgram.construct_arguments()`, query sizes, call the allocator, validate the buffer, and call `set_workspace()`.
4. Translation
  - Ensure the external-memory mode sets AllocationLifetime.External on the appropriate transients. For GPU targets the storage is GPU_Global; for CPU it is CPU_Heap (the default).
5. Tests
  - Extend test_dace_backend.py to cover the new transient_memory_mode="external" parameterization.
  - Add a unit test in test_dace_compilation.py verifying that an SDFG with External transients is built and that `set_workspace` is present in the generated code.
6. Optional: multi-stream sync
  - Prototype in a separate PR; it touches the same files but is logically independent.