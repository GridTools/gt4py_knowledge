"""Input data, meshes and gtfn calls shared by all benchmark runs."""

import numpy as np

import gt4py.next as gtx
from gt4py.next.program_processors.runners import gtfn

import progs as P
import progs_fixed as PF


H = P.H
NK = 80

CART_SIZES = {
    "xlarge": (1024, 1024, NK),
    "large": (512, 512, NK),
    "small": (32, 32, NK),
    "tiny": (12, 10, 7),
}
# periodic triangulated grid nx * ny: nv = nx*ny vertices, ne = 3*nv edges
MESH_SIZES = {
    "xlarge": (1024, 1024, NK),
    "large": (400, 400, NK),
    "small": (32, 32, NK),
    "tiny": (11, 9, 7),
}

KERNELS = ["lap", "lap2", "hdiff", "tmpst", "tri", "tritmp", "v2e", "chain"]
UNSTRUCTURED = ("v2e", "chain")
FIXED = {"hdiff": PF.pf_hdiff, "tmpst": PF.pf_tmpst, "tritmp": PF.pf_tritmp}
SKIP_FRACTION = 0.05


def make_mesh(nx, ny, locality, seed=1234):
    """Return (v2e [nv, 6] with -1 skips, e2v [ne, 2]) of a periodic triangulated grid."""
    nv = nx * ny
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    x, y = x.ravel(), y.ravel()  # vertex id = x + nx * y

    def vid(i, j):
        return (i % nx) + nx * (j % ny)

    v = vid(x, y)
    e2v = np.empty((3 * nv, 2), dtype=np.int32)
    for d, (dx, dy) in enumerate(((1, 0), (0, 1), (1, 1))):
        e2v[3 * v + d, 0] = v
        e2v[3 * v + d, 1] = vid(x + dx, y + dy)
    v2e = np.empty((nv, 6), dtype=np.int32)
    for d, (dx, dy) in enumerate(((1, 0), (0, 1), (1, 1))):
        v2e[v, d] = 3 * v + d
        v2e[v, 3 + d] = 3 * vid(x - dx, y - dy) + d
    # same skip pattern (in topological position) for both numberings
    rng = np.random.default_rng(seed)
    v2e[rng.random(v2e.shape) < SKIP_FRACTION] = -1
    if locality == "random":
        pv = rng.permutation(nv).astype(np.int32)
        pe = rng.permutation(3 * nv).astype(np.int32)
        new_v2e = np.empty_like(v2e)
        new_v2e[pv] = np.where(v2e < 0, -1, pe[np.where(v2e < 0, 0, v2e)])
        new_e2v = np.empty_like(e2v)
        new_e2v[pe] = pv[e2v]
        v2e, e2v = new_v2e, new_e2v
    else:
        assert locality == "local"
    return v2e, e2v


class Case:
    """One kernel at one size on one device: fields, tables and the gtfn call.

    Attributes:
        fields: input/output fields in the order the Kokkos library expects.
        outputs: indices into `fields` that are compared against gtfn.
        tables: neighbor connectivities in the order the Kokkos library expects.
        dims: (ni, nj, nk) for cartesian cases, (nv, ne, nk) for unstructured ones.
    """

    def __init__(self, kernel, size, locality, device):
        self.kernel, self.size, self.locality, self.device = kernel, size, locality, device
        self.backend = gtfn.run_gtfn if device == "cpu" else gtfn.run_gtfn_gpu
        if device == "gpu":
            import cupy

            self.xp = cupy
        else:
            self.xp = np
        self.rng = np.random.default_rng(7)
        self.tables = []
        if kernel in UNSTRUCTURED:
            self._unstructured()
        else:
            self._cartesian()

    def _field(self, domain, values=None):
        f = gtx.zeros(domain, dtype=np.float64, allocator=self.backend)
        if values is not None:
            f.ndarray[...] = self.xp.asarray(values)
        return f

    def _cartesian(self):
        ni, nj, nk = CART_SIZES[self.size]
        self.dims = (ni, nj, nk)
        shape = (ni + 2 * H, nj + 2 * H, nk)
        dom = {P.I: shape[0], P.J: shape[1], P.K: shape[2]}

        def inp(lo=0.0, hi=1.0):
            return self._field(dom, lo + (hi - lo) * self.rng.random(shape))

        k = self.kernel
        if k in ("lap", "lap2"):
            self.fields = [inp(), self._field(dom)]
            self.outputs = [1]
            prog = P.p_lap if k == "lap" else P.p_lap2
        elif k == "hdiff":
            self.fields = [inp(), inp(0.0, 0.1), self._field(dom)]
            self.outputs = [2]
            prog = P.p_hdiff
        elif k == "tmpst":
            self.fields = [inp(), inp(), self._field(dom)]
            self.outputs = [2]
            prog = P.p_tmpst
        elif k in ("tri", "tritmp"):
            abcd = [inp(), inp(4.0, 5.0), inp(), inp()]
            if k == "tri":
                self.fields = abcd + [self._field(dom) for _ in range(3)]
                self.outputs = [4, 5, 6]
                prog = P.p_tri
            else:
                self.fields = abcd + [self._field(dom)]
                self.outputs = [4]
                prog = P.p_tritmp
        else:
            raise ValueError(k)
        self.prog = prog.with_backend(self.backend)
        self.args = tuple(self.fields) + self.dims
        self.offset_provider = {}

    def gtfn_variants(self):
        """Return {name: program} of the gtfn baselines available for this case."""
        out = {"gtfn": self.prog}
        if self.kernel in UNSTRUCTURED:
            # same program, binding built with a static unit stride in the horizontal
            cfg = gtfn.GTFNConfig(
                gpu=self.device == "gpu", unstructured_horizontal_has_unit_stride=True
            )
            us = gtfn.make_gtfn_toolchain(cfg, name_postfix="_us")
            out["gtfn_us"] = self.prog.with_backend(us)
        if self.kernel in FIXED and self.device == "cpu":
            out["gtfn_fixed"] = FIXED[self.kernel].with_backend(self.backend)
        return out

    def _unstructured(self):
        nx, ny, nk = MESH_SIZES[self.size]
        nv, ne = nx * ny, 3 * nx * ny
        self.dims = (nv, ne, nk)
        v2e, e2v = make_mesh(nx, ny, self.locality)
        v2e_c = gtx.as_connectivity(
            [P.Vertex, P.V2EDim], P.Edge, v2e, skip_value=-1, allocator=self.backend
        )
        domv = {P.Vertex: nv, P.K: nk}
        dome = {P.Edge: ne, P.K: nk}
        if self.kernel == "v2e":
            self.fields = [self._field(dome, self.rng.random((ne, nk))), self._field(domv)]
            self.tables = [v2e_c]
            self.offset_provider = {"V2E": v2e_c}
            prog = P.p_v2e
        else:
            e2v_c = gtx.as_connectivity([P.Edge, P.E2VDim], P.Vertex, e2v, allocator=self.backend)
            self.fields = [self._field(domv, self.rng.random((nv, nk))), self._field(domv)]
            self.tables = [v2e_c, e2v_c]
            self.offset_provider = {"E2V": e2v_c, "V2E": v2e_c}
            prog = P.p_chain
        self.outputs = [1]
        self.prog = prog.with_backend(self.backend)
        self.args = tuple(self.fields) + self.dims

    def call(self, prog):
        prog(*self.args, offset_provider=self.offset_provider)
