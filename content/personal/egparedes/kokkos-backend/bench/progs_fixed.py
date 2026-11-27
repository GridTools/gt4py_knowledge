"""Clones of the programs with temporaries, compiled against patched GridTools headers."""

import gt4py.next as gtx

from progs import F, I, IT, J, K, hdiff_op, tmpst_op, tri_op


@gtx.program
def pf_hdiff(a: F, c: F, out: F, ni: IT, nj: IT, nk: IT):
    hdiff_op(a, c, out=out, domain={I: (3, ni + 3), J: (3, nj + 3), K: (0, nk)})


@gtx.program
def pf_tmpst(a: F, b: F, out: F, ni: IT, nj: IT, nk: IT):
    tmpst_op(a, b, out=out, domain={I: (3, ni + 3), J: (3, nj + 3), K: (0, nk)})


@gtx.program
def pf_tritmp(a: F, b: F, c: F, d: F, x: F, ni: IT, nj: IT, nk: IT):
    tri_op(a, b, c, d, out=x, domain={I: (3, ni + 3), J: (3, nj + 3), K: (0, nk)})
