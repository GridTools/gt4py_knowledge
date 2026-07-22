# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "dace==v2.0.0-alpha4",
#   "numpy",
#   "pytest",
# ]
# ///
#
# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests external memory allocation.
"""
import dace
import numpy as np
import pytest


@pytest.mark.parametrize('symbolic', (False, True))
def test_external_mem(symbolic):
    N = dace.symbol('N') if symbolic else 20

    @dace.program
    def tester(a: dace.float64[N]):
        workspace = dace.ndarray([N], dace.float64, lifetime=dace.AllocationLifetime.External)

        workspace[:] = a
        workspace += 1
        a[:] = workspace

    sdfg = tester.to_sdfg()

    # Test that there is no allocation
    code = sdfg.generate_code()[0].clean_code
    assert 'new double' not in code
    assert 'delete[]' not in code
    assert 'set_external_memory' in code

    a = np.random.rand(20)

    if symbolic:
        extra_args = dict(N=20)
    else:
        extra_args = {}

    # Test workspace size
    csdfg = sdfg.compile()
    csdfg.initialize(a, **extra_args)
    sizes = csdfg.get_workspace_sizes()
    assert sizes == {dace.StorageType.CPU_Heap: 20 * 8}

    # Test setting the workspace
    wsp = np.random.rand(20)
    csdfg.set_workspace(dace.StorageType.CPU_Heap, wsp)

    ref = a + 1

    csdfg(a, **extra_args)

    assert np.allclose(a, ref)
    assert np.allclose(wsp, ref)


def test_external_twobuffers():
    N = dace.symbol('N')

    @dace.program
    def tester(a: dace.float64[N]):
        workspace = dace.ndarray([N], dace.float64, lifetime=dace.AllocationLifetime.External)
        workspace2 = dace.ndarray([2], dace.float64, lifetime=dace.AllocationLifetime.External)

        workspace[:] = a
        workspace += 1
        workspace2[0] = np.sum(workspace)
        workspace2[1] = np.mean(workspace)
        a[0] = workspace2[0] + workspace2[1]

    sdfg = tester.to_sdfg()
    csdfg = sdfg.compile()

    # Test workspace size
    a = np.random.rand(20)
    csdfg.initialize(a=a, N=20)
    sizes = csdfg.get_workspace_sizes()
    assert sizes == {dace.StorageType.CPU_Heap: 22 * 8}

    # Test setting the workspace
    wsp = np.random.rand(22)
    csdfg.set_workspace(dace.StorageType.CPU_Heap, wsp)

    ref = a + 1
    ref2 = np.copy(a)
    s, m = np.sum(ref), np.mean(ref)
    ref2[0] = s + m

    csdfg(a=a, N=20)

    assert np.allclose(a, ref2)
    assert np.allclose(wsp[:-2], ref)
    assert np.allclose(wsp[-2], s)
    assert np.allclose(wsp[-1], m)


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
    assert 'set_external_memory' in code

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


@pytest.mark.gpu
def test_external_memory_2d_builder_gpu():
    """Tests external memory allocation with 2D arrays built via the SDFG
    Builder API for the GPU target.

    The SDFG computes:
        t0 <= a + b
        t1 <= t0 + c
        t2 <= t1 + d
        e  <= t2

    where ``t0``, ``t1`` and ``t2`` are transient arrays with
    ``dace.AllocationLifetime.External`` (i.e. their memory is provided by the
    caller through the external-memory workspace), while ``a``, ``b``, ``c``,
    ``d`` and ``e`` are regular global arrays. All arrays are 2D and live in
    ``dace.StorageType.GPU_Global``.
    """
    import cupy as cp

    M, N = 4, 6

    sdfg = dace.SDFG('external_memory_2d_builder_gpu')

    # Global arrays (inputs and output)
    sdfg.add_array('a', [M, N], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('b', [M, N], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('c', [M, N], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('d', [M, N], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('e', [M, N], dace.float64, storage=dace.StorageType.GPU_Global)

    # Transient arrays with external memory
    sdfg.add_transient('t0', [M, N], dace.float64, storage=dace.StorageType.GPU_Global,
                       lifetime=dace.AllocationLifetime.External)
    sdfg.add_transient('t1', [M, N], dace.float64, storage=dace.StorageType.GPU_Global,
                       lifetime=dace.AllocationLifetime.External)
    sdfg.add_transient('t2', [M, N], dace.float64, storage=dace.StorageType.GPU_Global,
                       lifetime=dace.AllocationLifetime.External)

    # All computations happen in a single state. The four maps are chained
    # through access nodes for the external transients t0, t1 and t2.
    state = sdfg.add_state('compute')

    # t0 = a + b
    me0, mx0 = state.add_map('map_t0', dict(i=f'0:{M}', j=f'0:{N}'), schedule=dace.ScheduleType.GPU_Device)
    t0_tasklet = state.add_tasklet('t0_op', {'_a', '_b'}, {'_out'}, '_out = _a + _b')
    state.add_memlet_path(state.add_read('a'), me0, t0_tasklet, dst_conn='_a', memlet=dace.Memlet('a[i, j]'))
    state.add_memlet_path(state.add_read('b'), me0, t0_tasklet, dst_conn='_b', memlet=dace.Memlet('b[i, j]'))
    t0_acc = state.add_access('t0')
    state.add_memlet_path(t0_tasklet, mx0, t0_acc, src_conn='_out', memlet=dace.Memlet('t0[i, j]'))

    # t1 = t0 + c
    me1, mx1 = state.add_map('map_t1', dict(i=f'0:{M}', j=f'0:{N}'), schedule=dace.ScheduleType.GPU_Device)
    t1_tasklet = state.add_tasklet('t1_op', {'_a', '_b'}, {'_out'}, '_out = _a + _b')
    state.add_memlet_path(t0_acc, me1, t1_tasklet, dst_conn='_a', memlet=dace.Memlet('t0[i, j]'))
    state.add_memlet_path(state.add_read('c'), me1, t1_tasklet, dst_conn='_b', memlet=dace.Memlet('c[i, j]'))
    t1_acc = state.add_access('t1')
    state.add_memlet_path(t1_tasklet, mx1, t1_acc, src_conn='_out', memlet=dace.Memlet('t1[i, j]'))

    # t2 = t1 + d
    me2, mx2 = state.add_map('map_t2', dict(i=f'0:{M}', j=f'0:{N}'), schedule=dace.ScheduleType.GPU_Device)
    t2_tasklet = state.add_tasklet('t2_op', {'_a', '_b'}, {'_out'}, '_out = _a + _b')
    state.add_memlet_path(t1_acc, me2, t2_tasklet, dst_conn='_a', memlet=dace.Memlet('t1[i, j]'))
    state.add_memlet_path(state.add_read('d'), me2, t2_tasklet, dst_conn='_b', memlet=dace.Memlet('d[i, j]'))
    t2_acc = state.add_access('t2')
    state.add_memlet_path(t2_tasklet, mx2, t2_acc, src_conn='_out', memlet=dace.Memlet('t2[i, j]'))

    # e = t2
    me3, mx3 = state.add_map('map_e', dict(i=f'0:{M}', j=f'0:{N}'), schedule=dace.ScheduleType.GPU_Device)
    e_tasklet = state.add_tasklet('e_op', {'_in'}, {'_out'}, '_out = _in')
    state.add_memlet_path(t2_acc, me3, e_tasklet, dst_conn='_in', memlet=dace.Memlet('t2[i, j]'))
    state.add_memlet_path(e_tasklet, mx3, state.add_write('e'), src_conn='_out', memlet=dace.Memlet('e[i, j]'))

    # Test that no GPU allocation is generated for the external transients
    code = sdfg.generate_code()[0].clean_code
    assert 'cudaMalloc' not in code
    assert 'cudaFree' not in code
    assert 'set_external_memory' in code

    # Compile and verify the external workspace size
    csdfg = sdfg.compile()

    a = cp.random.rand(M, N)
    b = cp.random.rand(M, N)
    c = cp.random.rand(M, N)
    d = cp.random.rand(M, N)
    e = cp.zeros((M, N))

    csdfg.initialize(a=a, b=b, c=c, d=d, e=e)
    sizes = csdfg.get_workspace_sizes()
    assert sizes == {dace.StorageType.GPU_Global: 3 * M * N * 8}

    # Provide the external workspace and run
    wsp = cp.zeros(3 * M * N)
    csdfg.set_workspace(dace.StorageType.GPU_Global, wsp)
    csdfg(a=a, b=b, c=c, d=d, e=e)

    # Verify the output and the intermediate results stored in the workspace
    ref_t0 = a + b
    ref_t1 = ref_t0 + c
    ref_t2 = ref_t1 + d

    assert cp.allclose(e, ref_t2)
    assert cp.allclose(wsp[:M * N].reshape(M, N), ref_t0)
    assert cp.allclose(wsp[M * N:2 * M * N].reshape(M, N), ref_t1)
    assert cp.allclose(wsp[2 * M * N:3 * M * N].reshape(M, N), ref_t2)


if __name__ == '__main__':
    test_external_mem(False)
    test_external_mem(True)
    test_external_twobuffers()
    test_external_memory_2d_builder()
    try:
        test_external_memory_2d_builder_gpu()
    except ImportError as e:
        print(f"GPU test skipped: {e}")
