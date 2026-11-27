"""Host code generation study for variant pairs that behave differently on the CPU.

Usage: python asm.py <gk-cpu compile_commands.json> <kernel source dir> <outdir>

Each variant is instantiated alone, compiled to assembly with the kernel library's flags plus
`-fopt-info-vec`, and summarised: lines GCC reports as vectorised, and NEON vector / scalar
floating point instruction counts of the outlined OpenMP loop bodies.
"""

import json
import pathlib
import re
import shlex
import subprocess
import sys


def spec():
    out = {}
    uns = "gk::unstr::{s}, {a}, {t}, gk::HRows<{i}>"
    ptr = ("gk::Ptr2<int>", "gk::TabPtr<int>", "int")
    acc = {
        "ptr": ptr,
        "ptr64": ("gk::Ptr2<long>", "gk::TabPtr<long>", "long"),
        "vl": ("gk::ViewL2", "gk::TabViewL", "int"),
    }
    for kernel, variants in {
        "v2e": [("ptr", "Iter"), ("ptr", "Iter1"), ("ptr", "Expl")],
        "chain": [
            ("ptr", "Iter"),
            ("ptr", "IterLit"),
            ("ptr", "Expl"),
            ("ptr64", "Iter"),
            ("vl", "Iter"),
            ("vl", "Expl"),
        ],
    }.items():
        for a, s in variants:
            acc_t, tab_t, i = acc[a]
            args = uns.format(s=s, a=acc_t, t=tab_t, i=i)
            out[f"{kernel}.{a}.{s}.rows"] = f"gk::unstr::run_{kernel}<{args}>"
    cols = {
        "chunk16": "gk::scan::Chunk<16>",
        "col": "gk::scan::PerColumn<gk::ColRange<{i}>>",
    }
    for a, acc_t, i in [
        ("ptr", "gk::Ptr3<int>", "int"),
        ("ptr64", "gk::Ptr3<long>", "long"),
        ("vl", "gk::ViewL3", "int"),
    ]:
        for s in ("Iter", "Expl"):
            if a == "ptr64" and s == "Expl":
                continue
            args = f"gk::scan::{s}, {acc_t}, {cols['chunk16']}"
            out[f"tri.{a}.{s}.chunk16"] = f"gk::scan::run_tri<{args}, false>"
    out["tri.ptr.Iter.chunk16.split"] = (
        "gk::scan::run_tri<gk::scan::Iter, gk::Ptr3<int>, gk::scan::Chunk<16>, true>"
    )
    return out


def main():
    cc, srcdir, outdir = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]) / "asm"
    outdir.mkdir(exist_ok=True)
    entry = next(e for e in json.load(open(cc)) if e["file"].endswith("reg_scan.cpp"))
    summary = {}
    for name, inst in spec().items():
        src = outdir / f"{name}.cpp"
        src.write_text(
            f'#include "{srcdir}/scan.hpp"\n#include "{srcdir}/unstr.hpp"\n'
            f"template void {inst}(GKArgs const &);\n"
        )
        args = shlex.split(entry["command"])
        asm = outdir / f"{name}.s"
        args[args.index("-o") + 1] = str(asm)
        args[args.index("-c") + 1] = str(src)
        args += ["-S", "-fopt-info-vec-optimized"]
        p = subprocess.run(args, cwd=entry["directory"], capture_output=True, text=True)
        if p.returncode:
            summary[name] = {"error": p.stderr[-2000:]}
            continue
        vec = sorted(
            {
                re.sub(r"^.*/", "", line)
                for line in p.stderr.splitlines()
                if "optimized:" in line and "/kk/" in line
            }
        )
        # kernel loops live in functions whose mangled name involves the gk namespace
        # (outlined OpenMP regions and any body GCC did not inline)
        counts = {"insns": 0, "vector_fp": 0, "scalar_fp": 0, "loads": 0, "stores": 0, "calls": 0}
        functions, inside = {}, None
        for line in asm.read_text().splitlines():
            label = re.match(r"^(_Z\S+):$", line)
            if label:
                inside = label.group(1) if "2gk" in label.group(1) else None
                continue
            if inside and line.strip().startswith(".size"):
                inside = None
            m = re.match(r"^\t([a-z][a-z0-9.]*)\t?(.*)$", line)
            if not (inside and m):
                continue
            op, rest = m.groups()
            functions[inside] = functions.get(inside, 0) + 1
            counts["insns"] += 1
            if op.startswith("f"):
                counts["vector_fp" if re.search(r"v\d+\.2d", rest) else "scalar_fp"] += 1
            counts["loads"] += op.startswith("ld")
            counts["stores"] += op.startswith("st")
            counts["calls"] += op == "bl"
        counts["functions"] = dict(sorted(functions.items(), key=lambda kv: -kv[1])[:6])
        summary[name] = {"vectorized": vec, **counts}
        shown = {k: v for k, v in counts.items() if k != "functions"}
        print(name, shown, f"{len(vec)} vectorised-loop reports", flush=True)
    (outdir.parent / "asm_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
