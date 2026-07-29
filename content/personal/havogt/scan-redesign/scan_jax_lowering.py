# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Reproducer for the two §4.1 claims about JAX lowerings of the vertical scan.

Status: **exploration prototype**, standalone. Requires `jax` and `numpy`; the
optional final check additionally requires `gt4py`.

It compares the scan lowerings §4.1 lists against today's embedded implementation
(`embedded/operators.py::ScanOperator`, a per-column *and* per-level Python loop),
and answers two questions:

1. Which lowerings admit a scalar Python `if` in the body? Only the eager ones.
   `vmap ∘ lax.scan` restores scalar *shape* but not *concreteness*, so the `if`
   still fails — with `TracerBoolConversionError` on a `bool[]` predicate, rather
   than the slice-body's "truth value of an array ... is ambiguous".
2. How do the lowerings compare on throughput? On CPU, (i) and (ii) are within
   noise of each other.

The strategies mirror the implementations discussed in §4.1:

- `ref_double_loop`   -- today's `ScanOperator`: scalar body, loop over columns and levels
- `vectorized_loop`   -- `ScanOperatorVectorized`: slice body, Python loop over K
- `jax_scan_slices`   -- lowering (i): array `lax.scan`, slice body
- `vmap_scan_scalar`  -- lowering (ii): `vmap`(horizontal) ∘ `lax.scan`, scalar body
"""

from __future__ import annotations

import time

import jax
import numpy as np
from jax import lax, numpy as jnp


jax.config.update("jax_enable_x64", True)

NX, NY, NZ = 64, 64, 80

_rng = np.random.default_rng(0)
A = _rng.standard_normal((NX, NY, NZ))
B = _rng.standard_normal((NX, NY, NZ))


def body_if(carry, a, b):
    """Scalar body as column physics is naturally written: data-dependent `if`."""
    if a > 0.0:
        return carry + a * b
    return carry * 0.5


def body_where(carry, a, b):
    """Same semantics, branch-free."""
    return jnp.where(a > 0.0, carry + a * b, carry * 0.5)


def body_where_np(carry, a, b):
    return np.where(a > 0.0, carry + a * b, carry * 0.5)


def ref_double_loop():
    """Today's embedded `ScanOperator`: scalar body, Python loop over columns and levels."""
    out = np.empty((NX, NY, NZ))
    for i in range(NX):
        for j in range(NY):
            acc = 0.0
            for k in range(NZ):
                acc = body_if(acc, A[i, j, k], B[i, j, k])
                out[i, j, k] = acc
    return out


def vectorized_loop(body):
    """`ScanOperatorVectorized`: body receives horizontal slices, one Python loop over K."""
    acc = np.zeros((NX, NY))
    out = np.empty((NX, NY, NZ))
    for k in range(NZ):
        acc = body(acc, A[..., k], B[..., k])
        out[..., k] = acc
    return out


def jax_scan_slices(body):
    """Lowering (i): array `lax.scan` over K with a horizontal-slice-valued carry."""
    a = jnp.asarray(np.moveaxis(A, -1, 0))
    b = jnp.asarray(np.moveaxis(B, -1, 0))

    def step(carry, xs):
        acc = body(carry, xs[0], xs[1])
        return acc, acc

    _, ys = lax.scan(step, jnp.zeros((NX, NY)), (a, b))
    return jnp.moveaxis(ys, 0, -1)


def vmap_scan_scalar(body):
    """Lowering (ii): `lax.scan` runs the scalar body per column, `vmap` over the horizontal."""

    def column(a_col, b_col):
        def step(carry, xs):
            acc = body(carry, xs[0], xs[1])
            return acc, acc

        _, ys = lax.scan(step, jnp.float64(0.0), (a_col, b_col))
        return ys

    out = jax.vmap(column)(jnp.asarray(A.reshape(-1, NZ)), jnp.asarray(B.reshape(-1, NZ)))
    return out.reshape(NX, NY, NZ)


def _report(label, fn, ref=None):
    try:
        out = np.asarray(jax.block_until_ready(fn()))
        match = "" if ref is None else f"  match={np.allclose(out, ref)}"
        print(f"  {label:42s} ok{match}")
    except Exception as e:
        print(f"  {label:42s} {type(e).__name__}: {str(e).strip().splitlines()[0][:64]}")


def main():
    t = time.perf_counter()
    ref = ref_double_loop()
    t_ref = time.perf_counter() - t
    print(f"reference: today's ScanOperator (scalar body, native `if`)   {t_ref * 1e3:.0f} ms\n")

    print("scalar python `if` in the body:")
    _report("vectorized loop  (slice body)", lambda: vectorized_loop(body_if), ref)
    _report("lax.scan slices  (i)", lambda: jax_scan_slices(body_if), ref)
    _report("vmap(lax.scan)   (ii)", lambda: vmap_scan_scalar(body_if), ref)

    print("\nbranch-free `where` body:")
    _report("vectorized loop  (slice body)", lambda: vectorized_loop(body_where_np), ref)
    _report("lax.scan slices  (i)", lambda: jax_scan_slices(body_where), ref)
    _report("vmap(lax.scan)   (ii)", lambda: vmap_scan_scalar(body_where), ref)

    # Run-to-run spread on CPU is comparable to the gap between the two lowerings and
    # the ordering is not stable, so report the best batch rather than a single mean.
    print(f"\njitted, best of 7 batches of 10 after warmup (grid {NX}x{NY}x{NZ}):")
    for label, fn in [
        ("lax.scan slices  (i)", lambda: jax_scan_slices(body_where)),
        ("vmap(lax.scan)   (ii)", lambda: vmap_scan_scalar(body_where)),
    ]:
        jitted = jax.jit(fn)
        jax.block_until_ready(jitted())
        batches = []
        for _ in range(7):
            t = time.perf_counter()
            for _ in range(10):
                jax.block_until_ready(jitted())
            batches.append((time.perf_counter() - t) / 10)
        print(
            f"  {label:42s} {min(batches) * 1e3:6.2f} ms"
            f"   (spread {min(batches) * 1e3:.2f}-{max(batches) * 1e3:.2f})"
        )
    print(f"  {'today (per-element python loop)':42s} {t_ref * 1e3:6.0f} ms")

    _gt4py_embedded_if_check()


def _gt4py_embedded_if_check():
    """The scalar `if` really does work in embedded execution today (needs gt4py)."""
    try:
        import gt4py.next as gtx
        from gt4py.next import common
    except ImportError:
        print("\n(skipping the embedded gt4py check: gt4py not importable)")
        return

    I = gtx.Dimension("I")
    K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)

    @gtx.scan_operator(axis=K, forward=True, init=0.0)
    def with_if(carry: float, a: float) -> float:
        if a > 0.0:
            return carry + a
        return carry * 0.5

    a = gtx.as_field([I, K], np.array([[1.0, -2.0, 3.0, -4.0]]))
    out = gtx.zeros(common.domain({I: 1, K: 4}), dtype=np.float64, allocator=None)
    with_if(a, out=out, offset_provider={})
    print(f"\nembedded gt4py scan_operator with scalar `if` -> {out.asnumpy().tolist()}")


if __name__ == "__main__":
    main()
