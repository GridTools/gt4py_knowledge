"""Compile the gtfn programs of one device into the persistent gt4py cache.

Usage: python gtfn_compile.py cpu|gpu stock|fixed [kernel ...]

`stock` compiles `gtfn` (and `gtfn_us` for unstructured kernels), `fixed` compiles `gtfn_fixed`
against patched GridTools headers. Never run `fixed` while another compilation is running.
"""

import contextlib
import sys
import time

import cases
import patch_gt


def main():
    device, which = sys.argv[1:3]
    kernels = sys.argv[3:] or (cases.KERNELS if which == "stock" else list(cases.FIXED))
    for kernel in kernels:
        case = cases.Case(kernel, "tiny", "local", device)
        for name, prog in case.gtfn_variants().items():
            if (name == "gtfn_fixed") != (which == "fixed"):
                continue
            ctx = patch_gt.patched() if which == "fixed" else contextlib.nullcontext()
            t0 = time.perf_counter()
            with ctx:
                case.call(prog)
            print(
                f"{device} {kernel} {name}: first call {time.perf_counter() - t0:.1f}s "
                f"field strides {[tuple(s // 8 for s in f.ndarray.strides) for f in case.fields]} "
                f"table strides "
                f"{[tuple(s // t.ndarray.itemsize for s in t.ndarray.strides) for t in case.tables]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
