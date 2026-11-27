"""Remove the double allocation and the zero-fill of gtfn_cpu temporaries (GridTools 2.3.9).

The edit is applied to the node-local copy of the `gridtools_cpp` headers only while the
`pf_*` programs are compiled, then reverted.
"""

import contextlib
import pathlib

import gridtools_cpp


INC = pathlib.Path(gridtools_cpp.get_include_dir()) / "gridtools"

EDITS = {
    "sid/allocator.hpp": [
        ("                    auto ptr = self.m_impl(sizeof(type) * size);\n", "")
    ],
    "fn/backend/naive.hpp": [
        (
            "    namespace naive_impl_ {\n",
            "    namespace naive_impl_ {\n"
            "        inline std::unique_ptr<char[]> gt_tmp_uninit(std::size_t n) {\n"
            '            __attribute__((used)) static char const marker[] = "GT_TMP_PATCH";\n'
            "            return std::unique_ptr<char[]>(new char[n]);\n"
            "        }\n",
        ),
        ("sid::allocator(&std::make_unique<char[]>)", "sid::allocator(&gt_tmp_uninit)"),
    ],
}


@contextlib.contextmanager
def patched():
    originals = {}
    try:
        for name, edits in EDITS.items():
            path = INC / name
            originals[path] = text = path.read_text()
            for old, new in edits:
                assert text.count(old) == 1, f"{name}: expected exactly one {old!r}"
                text = text.replace(old, new)
            path.write_text(text)
        yield
    finally:
        for path, text in originals.items():
            path.write_text(text)
