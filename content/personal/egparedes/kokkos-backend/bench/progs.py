"""gt4py.next programs used as gtfn baselines (gt4py commit c56efffdd).

All programs take their compute-domain sizes as scalar parameters so that one
compiled artifact serves the small and the large size.

Cartesian fields are allocated over (ni + 2*H, nj + 2*H, nk) with H = 3 and the
compute domain is [H, H + n) in I and J.
"""

import gt4py.next as gtx
from gt4py.next import neighbor_sum, where


H = 3

I = gtx.Dimension("I")
J = gtx.Dimension("J")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
F = gtx.Field[gtx.Dims[I, J, K], gtx.float64]
IT = gtx.int32

Vertex = gtx.Dimension("Vertex")
Edge = gtx.Dimension("Edge")
V2EDim = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)
E2VDim = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)
V2E = gtx.FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
E2V = gtx.FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))
VF = gtx.Field[gtx.Dims[Vertex, K], gtx.float64]
EF = gtx.Field[gtx.Dims[Edge, K], gtx.float64]


# -- Cartesian ---------------------------------------------------------------
@gtx.field_operator
def lap_op(a: F) -> F:
    return a(I + 1) + a(I - 1) + a(J + 1) + a(J - 1) - 4.0 * a


@gtx.field_operator
def lap2_op(a: F) -> F:
    return lap_op(lap_op(a))


@gtx.field_operator
def flx_op(lp: F, a: F) -> F:
    f = lp(I + 1) - lp
    return where(f * (a(I + 1) - a) > 0.0, 0.0, f)


@gtx.field_operator
def fly_op(lp: F, a: F) -> F:
    f = lp(J + 1) - lp
    return where(f * (a(J + 1) - a) > 0.0, 0.0, f)


@gtx.field_operator
def hdiff_op(a: F, c: F) -> F:
    lp = lap_op(a)
    fx = flx_op(lp, a)
    fy = fly_op(lp, a)
    return a - c * (fx - fx(I - 1) + fy - fy(J - 1))


@gtx.field_operator
def s1_op(a: F, b: F) -> F:
    return a * b + a


@gtx.field_operator
def s2_op(t: F) -> F:
    return 0.5 * (t(I + 1) + t(I - 1)) + 0.25 * (t(J + 1) + t(J - 1))


@gtx.field_operator
def tmpst_op(a: F, b: F) -> F:
    return s2_op(s1_op(a, b))


@gtx.program
def p_lap(a: F, out: F, ni: IT, nj: IT, nk: IT):
    lap_op(a, out=out, domain={I: (3, ni + 3), J: (3, nj + 3), K: (0, nk)})


@gtx.program
def p_lap2(a: F, out: F, ni: IT, nj: IT, nk: IT):
    lap2_op(a, out=out, domain={I: (3, ni + 3), J: (3, nj + 3), K: (0, nk)})


@gtx.program
def p_hdiff(a: F, c: F, out: F, ni: IT, nj: IT, nk: IT):
    hdiff_op(a, c, out=out, domain={I: (3, ni + 3), J: (3, nj + 3), K: (0, nk)})


@gtx.program
def p_tmpst(a: F, b: F, out: F, ni: IT, nj: IT, nk: IT):
    tmpst_op(a, b, out=out, domain={I: (3, ni + 3), J: (3, nj + 3), K: (0, nk)})


# -- Vertical scans ----------------------------------------------------------
@gtx.scan_operator(axis=K, forward=True, init=(0.0, 0.0))
def tri_fwd(
    state: tuple[float, float], a: float, b: float, c: float, d: float
) -> tuple[float, float]:
    den = b - a * state[0]
    return (c / den, (d - a * state[1]) / den)


@gtx.scan_operator(axis=K, forward=False, init=0.0)
def tri_bwd(x_kp1: float, cp: float, dp: float) -> float:
    return dp - cp * x_kp1


@gtx.field_operator
def tri_op(a: F, b: F, c: F, d: F) -> F:
    cp, dp = tri_fwd(a, b, c, d)
    return tri_bwd(cp, dp)


# caller-provided cp/dp: two scan executions, no temporaries
@gtx.program
def p_tri(a: F, b: F, c: F, d: F, cp: F, dp: F, x: F, ni: IT, nj: IT, nk: IT):
    tri_fwd(a, b, c, d, out=(cp, dp), domain={I: (3, ni + 3), J: (3, nj + 3), K: (0, nk)})
    tri_bwd(cp, dp, out=x, domain={I: (3, ni + 3), J: (3, nj + 3), K: (0, nk)})


# cp/dp are program temporaries
@gtx.program
def p_tritmp(a: F, b: F, c: F, d: F, x: F, ni: IT, nj: IT, nk: IT):
    tri_op(a, b, c, d, out=x, domain={I: (3, ni + 3), J: (3, nj + 3), K: (0, nk)})


# -- Unstructured ------------------------------------------------------------
@gtx.field_operator
def v2e_op(e: EF) -> VF:
    return neighbor_sum(e(V2E), axis=V2EDim)


@gtx.field_operator
def e2v_op(a: VF) -> EF:
    return neighbor_sum(a(E2V), axis=E2VDim)


@gtx.field_operator
def chain_op(a: VF) -> VF:
    return neighbor_sum(e2v_op(a)(V2E), axis=V2EDim)


@gtx.program
def p_v2e(e: EF, out: VF, nv: IT, ne: IT, nk: IT):
    v2e_op(e, out=out, domain={Vertex: (0, nv), K: (0, nk)})


@gtx.program
def p_chain(a: VF, out: VF, nv: IT, ne: IT, nk: IT):
    chain_op(a, out=out, domain={Vertex: (0, nv), K: (0, nk)})
