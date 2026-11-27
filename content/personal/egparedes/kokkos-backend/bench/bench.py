"""Check and time gtfn and the Kokkos variants on the same buffers, interleaved, in one process.

Per cell (kernel, size, table locality):
1. correctness: outputs are filled with NaN, each variant runs once and is compared with gtfn;
2. warm-up: three calls per variant;
3. timing: `reps` rounds; every round runs all variants once, in a random order. A sample is the
   median of `n_inner` individually timed calls.

gtfn is timed by gt4py itself (`run_cpp_duration`: C++ clock around the generated program plus
`cudaDeviceSynchronize` on the GPU). Kokkos is timed by the same clock in C++ around the kernel
launches plus the closing synchronisation.
"""

import argparse
import ctypes as ct
import json
import os
import random
import re
import statistics
import sys
import time


os.environ.setdefault("GT4PY_COLLECT_METRICS_LEVEL", "10")

import numpy as np

from gt4py.next.instrumentation import metrics

import cases


class GKArgs(ct.Structure):
    _fields_ = [
        ("f", ct.c_void_p * 8),
        ("fs", (ct.c_int64 * 3) * 8),
        ("fe", (ct.c_int64 * 3) * 8),
        ("t", ct.c_void_p * 2),
        ("ts", (ct.c_int64 * 2) * 2),
        ("te", (ct.c_int64 * 2) * 2),
        ("ni", ct.c_int32),
        ("nj", ct.c_int32),
        ("nk", ct.c_int32),
        ("h", ct.c_int32),
    ]


def load(path, nthreads):
    lib = ct.CDLL(path)
    lib.gk_init.argtypes = [ct.c_int]
    lib.gk_info.restype = ct.c_char_p
    lib.gk_variants.argtypes = [ct.c_char_p]
    lib.gk_variants.restype = ct.c_char_p
    lib.gk_run.argtypes = [ct.c_char_p, ct.c_char_p, ct.POINTER(GKArgs), ct.c_int]
    lib.gk_run.restype = ct.c_double
    lib.gk_init(nthreads)
    return lib


def pointer(arr):
    return arr.data.ptr if hasattr(arr.data, "ptr") else arr.ctypes.data


def make_args(case):
    g = GKArgs()
    for n, field in enumerate(case.fields):
        a = field.ndarray
        g.f[n] = pointer(a)
        for d in range(a.ndim):
            g.fs[n][d] = a.strides[d] // a.itemsize
            g.fe[n][d] = a.shape[d]
    for n, tab in enumerate(case.tables):
        a = tab.ndarray
        assert a.dtype == np.int32
        g.t[n] = pointer(a)
        for d in range(2):
            g.ts[n][d] = a.strides[d] // a.itemsize
            g.te[n][d] = a.shape[d]
    g.ni, g.nj, g.nk = case.dims
    g.h = 0 if case.kernel in cases.UNSTRUCTURED else cases.H
    return g


class Runner:
    def __init__(self, case, lib):
        self.case, self.lib = case, lib
        self.args = make_args(case)
        self.kernel = case.kernel.encode()
        self.gtfn = case.gtfn_variants()
        if case.device == "gpu":
            self.sync = case.xp.cuda.runtime.deviceSynchronize
        else:
            self.sync = lambda: None

    def time_gtfn(self, prog):
        self.sync()
        before = {k: len(s.metrics["compute"].samples) for k, s in metrics.sources.items()}
        self.case.call(prog)
        new = [
            s.metrics["compute"].samples[-1]
            for k, s in metrics.sources.items()
            if len(s.metrics["compute"].samples) > before.get(k, 0)
        ]
        assert len(new) == 1, f"expected one new gtfn sample, got {len(new)}"
        return new[0]

    def time(self, name):
        """Run variant `name` once and return the elapsed seconds."""
        if name in self.gtfn:
            return self.time_gtfn(self.gtfn[name])
        base, _, flag = name.partition("+")
        t = self.lib.gk_run(self.kernel, base.encode(), ct.byref(self.args), int(flag == "kfence"))
        if t < 0:
            raise RuntimeError(f"{name} failed")
        return t

    def outputs(self):
        for n in self.case.outputs:
            yield self.case.fields[n].ndarray

    def run_checked(self, name):
        for a in self.outputs():
            a[...] = np.nan
        self.lib.gk_pool_poison()
        self.sync()
        self.time(name)
        self.sync()
        return [np.array(a.get() if hasattr(a, "get") else a, copy=True) for a in self.outputs()]


def compare(ref, got):
    res = {"ok": True, "maxabs": 0.0, "maxrel": 0.0, "points": 0}
    for r, g in zip(ref, got):
        mask = ~np.isnan(r)
        if not np.array_equal(mask, ~np.isnan(g)):
            return {"ok": False, "why": "written points differ", "points": int(mask.sum())}
        diff = np.abs(r[mask] - g[mask])
        scale = max(1.0, float(np.abs(r[mask]).max()))
        res["maxabs"] = max(res["maxabs"], float(diff.max()))
        res["maxrel"] = max(res["maxrel"], float(diff.max()) / scale)
        res["points"] += int(mask.sum())
    res["ok"] = res["maxrel"] <= 1e-11 and res["points"] > 0
    return res


def expected_points(case):
    ni, nj, nk = case.dims
    n = ni * nk if case.kernel in cases.UNSTRUCTURED else ni * nj * nk
    return n * len(case.outputs)


def run_cell(case, lib, opts, out):
    run = Runner(case, lib)
    kokkos = lib.gk_variants(case.kernel.encode()).decode().split(",")
    if opts.variants:
        kokkos = [v for v in kokkos if re.search(opts.variants, v)]
    names = list(run.gtfn) + kokkos
    if case.device == "gpu" and not opts.no_kfence:
        names += [v + "+kfence" for v in kokkos if v in ("ptr.it.mdtlb", "ptr.it.colmdtlb")]
    cell = {
        "device": case.device,
        "threads": opts.threads,
        "kernel": case.kernel,
        "size": case.size,
        "locality": case.locality,
        "dims": case.dims,
        "tag": opts.tag,
    }
    label = f"[{case.device} t={opts.threads}] {case.kernel}/{case.size}/{case.locality}"

    ref = run.run_checked("gtfn")
    checks = {"gtfn": compare(ref, ref)}
    assert checks["gtfn"]["points"] == expected_points(case), "gtfn did not write its domain"
    for name in names[1:]:
        try:
            checks[name] = compare(ref, run.run_checked(name))
        except Exception as e:
            checks[name] = {"ok": False, "why": repr(e)}
    bad = [n for n, c in checks.items() if not c["ok"]]
    worst = max(c.get("maxrel", 0.0) for c in checks.values())
    print(f"{label}: {len(names)} variants, max rel diff {worst:.2e}, failed {bad}", flush=True)
    names = [n for n in names if n not in bad]
    for n in bad:
        out.write(json.dumps({**cell, "variant": n, "check": checks[n], "samples": []}) + "\n")

    if opts.check_only:
        for n in names:
            out.write(json.dumps({**cell, "variant": n, "check": checks[n], "samples": []}) + "\n")
        out.flush()
        return

    for n in names:
        for _ in range(3):
            run.time(n)
    probe = statistics.median(run.time("gtfn") for _ in range(5))
    n_inner = 30 if probe < 1e-4 else 10 if probe < 1e-3 else 3 if probe < 1e-2 else 1
    samples = {n: [] for n in names}
    order = random.Random(42)
    start = time.perf_counter()
    for rep in range(opts.reps):
        round_ = names[:]
        order.shuffle(round_)
        for n in round_:
            samples[n].append(statistics.median(run.time(n) for _ in range(n_inner)))
        if rep + 1 >= opts.min_reps and time.perf_counter() - start > opts.cell_seconds:
            break
    for n in names:
        out.write(
            json.dumps(
                {**cell, "variant": n, "n_inner": n_inner, "check": checks[n], "samples": samples[n]}
            )
            + "\n"
        )
    out.flush()
    base = statistics.median(samples["gtfn"])
    ratios = " ".join(
        f"{n}={statistics.median(s) / base:.2f}" for n, s in samples.items() if n != "gtfn"
    )
    print(
        f"{label}: gtfn {base * 1e3:.4f} ms, reps {len(samples['gtfn'])}, n_inner {n_inner}, "
        f"{time.perf_counter() - start:.1f}s | {ratios}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("device", choices=["cpu", "gpu"])
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernels", default=",".join(cases.KERNELS))
    ap.add_argument("--sizes", default="large,small")
    ap.add_argument("--localities", default="local,random")
    ap.add_argument("--variants", default=None, help="regex selecting Kokkos variants")
    ap.add_argument("--reps", type=int, default=21)
    ap.add_argument("--min-reps", type=int, default=15)
    ap.add_argument("--cell-seconds", type=float, default=1e9, help="stop after min-reps beyond")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--no-kfence", action="store_true")
    ap.add_argument("--tag", default="")
    opts = ap.parse_args()
    opts.threads = int(os.environ.get("OMP_NUM_THREADS", "1"))

    lib = load(opts.lib, opts.threads if opts.device == "cpu" else 0)
    print(lib.gk_info().decode(), flush=True)
    failures = []
    with open(opts.out, "a") as out:
        for kernel in opts.kernels.split(","):
            locs = opts.localities.split(",") if kernel in cases.UNSTRUCTURED else ["local"]
            for size in opts.sizes.split(","):
                for loc in locs:
                    try:
                        case = cases.Case(kernel, size, loc, opts.device)
                        run_cell(case, lib, opts, out)
                        del case
                    except Exception as e:
                        failures.append((kernel, size, loc, repr(e)))
                        print(f"CELL FAILED {kernel}/{size}/{loc}: {e!r}", flush=True)
                    lib.gk_pool_clear()
                    if opts.device == "gpu":
                        import cupy

                        cupy.get_default_memory_pool().free_all_blocks()
    print("failed cells:", failures, flush=True)
    lib.gk_finalize()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
