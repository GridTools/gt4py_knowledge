"""Compile time and GPU register usage, iterator vs explicit index (base GPU variant per kernel).

Usage: python ct.py <gk compile_commands.json> <kernel source dir> <outdir>

Each Kokkos kernel is compiled alone (one explicit template instantiation per translation unit)
with the flags of the kernel library. gtfn programs are recompiled with their own build command.
`cuobjdump -res-usage` gives registers per device kernel.
"""

import concurrent.futures as cf
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time


STYLES = {
    "it": "Iter",
    "ex": "Expl",
    "it1": "Iter1",
    "ex2": "Expl2",
    "itlit": "IterLit",
    "pw": "Walk",
}
KERNEL_STYLES = {
    "lap": ("it", "ex"),
    "lap2": ("it", "ex"),
    "hdiff": ("it", "ex"),
    "tmpst": ("it", "ex"),
    "tri": ("it", "ex", "pw"),
    "tri_split": ("it", "pw"),
    "tritmp": ("it",),
    "tritmp_split": ("it", "pw"),
    "v2e": ("it", "it1", "ex", "ex2"),
    "chain": ("it", "itlit", "ex", "ex2"),
}


def instantiation(kernel, style):
    s = STYLES[style]
    cart = f"gk::cart::{s}, gk::Ptr3<int>, gk::MdTile<int, true>"
    uns = f"gk::unstr::{s}, gk::Ptr2<int>, gk::TabPtr<int>, gk::HMdTile<int, true>"
    col = (
        f"gk::scan::{s}, gk::Ptr3<int>, gk::scan::PerColumn<gk::ColMdTile<int, true>>"
    )
    return {
        "lap": f"gk::cart::run_lap<{cart}>",
        "lap2": f"gk::cart::run_lap2<{cart}>",
        "hdiff": f"gk::cart::run_hdiff<{cart}, gk::TmpPool>",
        "tmpst": f"gk::cart::run_tmpst<{cart}, gk::TmpPool>",
        "tri": f"gk::scan::run_tri<{col}, false>",
        "tri_split": f"gk::scan::run_tri<{col}, true>",
        "tritmp": f"gk::scan::run_tritmp<{col}, gk::TmpPool, false>",
        "tritmp_split": f"gk::scan::run_tritmp<{col}, gk::TmpPool, true>",
        "v2e": f"gk::unstr::run_v2e<{uns}>",
        "chain": f"gk::unstr::run_chain<{uns}>",
    }[kernel]


def run(cmd, cwd):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return time.perf_counter() - t0, p


def swap(cmd, out, src):
    args = shlex.split(cmd)
    args[args.index("-o") + 1] = out
    args[args.index("-c") + 1] = src
    return args


def res_usage(obj):
    p = subprocess.run(["cuobjdump", "-res-usage", obj], capture_output=True, text=True)
    return p.stdout


def sass_counts(obj):
    """Return {device function: {instructions, global loads, global stores}} from the SASS."""
    p = subprocess.run(["cuobjdump", "-sass", obj], capture_output=True, text=True)
    out, current = {}, None
    for line in p.stdout.splitlines():
        text = line.strip()
        if text.startswith("Function :"):
            current = out.setdefault(text.split(":", 1)[1].strip(), {"n": 0, "ldg": 0, "stg": 0})
            continue
        m = re.match(r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", text)
        if current is None or not m or m.group(1) == "NOP":
            continue
        op = m.group(1)
        current["n"] += 1
        current["ldg"] += op.startswith(("LDG", "LD.", "LDGSTS"))
        current["stg"] += op.startswith(("STG", "ST."))
    return out


def main():
    cc, srcdir, outdir = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    work = pathlib.Path(os.environ["W"]) / "ct"
    work.mkdir(exist_ok=True)
    entry = next(e for e in json.load(open(cc)) if e["file"].endswith("reg_scan.cpp"))
    jobs = {}
    for kernel, styles in KERNEL_STYLES.items():
        for style in styles:
            src = work / f"ct_{kernel}_{style}.cpp"
            src.write_text(
                f'#include "{srcdir}/scan.hpp"\n#include "{srcdir}/unstr.hpp"\n'
                f"template void {instantiation(kernel, style)}(GKArgs const &);\n"
            )
            jobs[f"kokkos/{kernel}/{style}"] = (src, str(src.with_suffix(".o")), entry)
    src = work / "ct_headers_only.cpp"
    src.write_text(f'#include "{srcdir}/scan.hpp"\n#include "{srcdir}/unstr.hpp"\n')
    jobs["kokkos/headers_only/-"] = (src, str(src.with_suffix(".o")), entry)
    cache = pathlib.Path(os.environ["GT4PY_BUILD_CACHE_DIR"]) / ".gt4py_cache"
    for d in sorted(cache.glob("p_*_pyext_*")):
        ccd = json.load(open(d / "compile_commands.json"))
        e = next((e for e in ccd if "_bindings" in e["file"]), None)
        if e is None:
            continue
        # only the CUDA builds
        if "nvcc" not in e["command"]:
            continue
        kernel = d.name.split("_pyext_")[0][2:]
        # gtfn records "directory" relative to the program cache folder
        e = {**e, "directory": str((d / e["directory"]).resolve())}
        jobs[f"gtfn/{kernel}/{d.name[-16:]}"] = (None, str(work / f"{d.name}.o"), e)

    def one(item):
        name, (src, obj, e) = item
        if src is None:
            f = e["file"] if os.path.isabs(e["file"]) else os.path.join(e["directory"], e["file"])
            cmd = swap(e["command"], obj, f)
        else:
            cmd = swap(e["command"], obj, str(src))
        dt, p = run(cmd, e["directory"])
        rec = {"name": name, "seconds": dt, "rc": p.returncode, "cmd": " ".join(cmd)}
        if p.returncode:
            rec["err"] = p.stderr[-3000:]
        else:
            rec["res_usage"] = res_usage(obj)
            rec["sass"] = sass_counts(obj)
            rec["ptxas"] = [l for l in p.stderr.splitlines() if "registers" in l or "Function" in l]
        return rec

    with cf.ThreadPoolExecutor(max_workers=8) as ex, open(outdir / "compile.jsonl", "w") as out:
        for rec in ex.map(one, jobs.items()):
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(rec["name"], f"{rec['seconds']:.1f}s rc={rec['rc']}", flush=True)


if __name__ == "__main__":
    main()
