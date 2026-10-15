//! Launchers of the decode FFN fusions (`MatmulGlu`, `RmsNormAddMatmulGlu`; kernels/metal/gemv_glu.metal): the gate / up
//! projections of one `x` with the GLU as the epilogue, and for the second op the residual add and pre-norm as the prologue.
//! One dispatch replaces `rmsnorm_add` + `gemv_pair` + `silu_glu`. Bitwise the unfused Metal sequence when the operands take
//! 16-byte (or, for 8-byte aligned shards, 8-byte) loads; any other alignment runs the 2-byte-load variant (correct, within
//! tolerance of the unfused sequence, not bitwise).

use crate::dense::{Bind, Grid};
use crate::kernels::{GLU_NORM_VARIANTS, GLU_VARIANTS, QKV_VARIANTS};
use crate::MetalBackend;
use intarsia_core::{DType, Error, OpCall, Result};

/// Kernel arguments of `gemv_glu_t` (`GluP` in gemv_glu.metal; 48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GluP {
    pub k: u32,
    pub n: u32,
    pub wg_skip: u32,
    pub wu_skip: u32,
    pub x_skip: u32,
    pub y_skip: u32,
    pub res_skip: u32,
    pub nw_skip: u32,
    pub sum_skip: u32,
    pub eps: f32,
    pub w_off: f32,
    pub pad: u32,
}

/// Threads per threadgroup of the GLU shaders (one threadgroup per output row, eight simdgroups).
pub(crate) const GLU_THREADS: usize = 256;
/// Most hidden elements the norm prologue keeps in threadgroup memory (`xt` in gemv_glu.metal).
pub const GLU_NORM_MAX_K: usize = 4096;

fn addr(b: &Bind) -> usize {
    b.off + 2 * b.skip as usize
}

impl MetalBackend {
    /// `a = silu(x Wg^T) * (x Wu^T)` (`norm == false`: inputs x, wg, wu; output a) or the same on
    /// `RmsNormAdd(x, res, w)` (inputs x, res, w, wg, wu; outputs a, sum).
    pub(crate) fn launch_matmul_glu(&self, call: &OpCall, norm: bool) -> Result<()> {
        let bf = DType::Bf16;
        let (i, o) = (&call.inputs, &call.outputs);
        let (x, wg, wu, a) = if norm { (&i[0], &i[3], &i[4], &o[0]) } else { (&i[0], &i[1], &i[2], &o[0]) };
        if x.shape.len() != 2 || x.shape[0] != 1 || wg.shape.len() != 2 {
            return Err(Error::invalid("metal glu: needs x [1,K], w [N,K]"));
        }
        let (k, n) = (wg.shape[1], wg.shape[0]);
        if n == 0 || k == 0 {
            return Ok(());
        }
        if norm && k > GLU_NORM_MAX_K {
            return Err(Error::invalid(format!("metal glu: the norm prologue takes K <= {GLU_NORM_MAX_K}, got {k}")));
        }
        let (xb, wgb, wub, ab) = (
            self.bind(x, bf, false, "x")?,
            self.bind(wg, bf, false, "w_gate")?,
            self.bind(wu, bf, false, "w_up")?,
            self.bind(a, bf, true, "a")?,
        );
        // vector loads when W (and x, when it is read as vectors) allow: the unfused GEMV's own choice
        let vec_ok = |align: usize, kmod: usize| {
            k % kmod == 0
                && addr(&wgb) % align == 0
                && addr(&wub) % align == 0
                && (norm || addr(&xb) % align == 0)
        };
        let (idx, wskip) = if vec_ok(16, 8) {
            (0, false)
        } else if vec_ok(8, 4) {
            (1, false)
        } else {
            (2, true)
        };
        let names = if norm { GLU_NORM_VARIANTS } else { GLU_VARIANTS };
        let mut p = GluP {
            k: k as u32,
            n: n as u32,
            y_skip: ab.skip,
            ..Default::default()
        };
        // the vector variants are bound at the exact (aligned) address; the 2-byte one at the 4-byte aligned base + skip
        let wbind = |b: &Bind| -> Bind {
            if wskip {
                Bind { mtl: b.mtl.clone(), off: b.off, skip: b.skip }
            } else {
                Bind { mtl: b.mtl.clone(), off: addr(b), skip: 0 }
            }
        };
        let (wgb2, wub2) = (wbind(&wgb), wbind(&wub));
        p.wg_skip = if wskip { wgb.skip } else { 0 };
        p.wu_skip = if wskip { wub.skip } else { 0 };
        if norm {
            // x (blk) is read as bf16 elements by the prologue
            p.x_skip = xb.skip;
            let (res, nw, sum) = (
                self.bind(&i[1], bf, false, "res")?,
                self.bind(&i[2], bf, false, "w")?,
                self.bind(&o[1], bf, true, "sum")?,
            );
            p.res_skip = res.skip;
            p.nw_skip = nw.skip;
            p.sum_skip = sum.skip;
            p.eps = call.attrs.eps;
            p.w_off = call.attrs.w_offset;
            let xb2 = Bind { mtl: xb.mtl.clone(), off: xb.off, skip: xb.skip };
            return self.dispatch(
                names[idx],
                &[&wgb2, &wub2, &xb2, &res, &nw, &ab, &sum],
                &p,
                Grid::Groups { x: n, y: 1, threads: GLU_THREADS },
            );
        }
        let xb2 = if wskip { Bind { mtl: xb.mtl.clone(), off: xb.off, skip: xb.skip } } else { Bind { mtl: xb.mtl.clone(), off: addr(&xb), skip: 0 } };
        p.x_skip = if wskip { xb.skip } else { 0 };
        self.dispatch(names[idx], &[&wgb2, &wub2, &xb2, &ab, &ab, &ab, &ab], &p, Grid::Groups { x: n, y: 1, threads: GLU_THREADS })
    }

    /// `RmsNormAddMatmulQkv`: inputs x, res, w, wq, wk, wv; outputs q, k, v, sum.
    pub(crate) fn launch_norm_qkv(&self, call: &OpCall) -> Result<()> {
        let bf = DType::Bf16;
        let (i, o) = (&call.inputs, &call.outputs);
        if i[0].shape.len() != 2 || i[0].shape[0] != 1 {
            return Err(Error::invalid("metal qkv: needs x [1,H]"));
        }
        let h = i[0].shape[1];
        let (nq, nk, nv) = (i[3].shape[0], i[4].shape[0], i[5].shape[0]);
        if h > GLU_NORM_MAX_K {
            return Err(Error::invalid(format!("metal qkv: the norm prologue takes H <= {GLU_NORM_MAX_K}, got {h}")));
        }
        if h == 0 || nq + nk + nv == 0 {
            return Ok(());
        }
        let w = [
            self.bind(&i[3], bf, false, "wq")?,
            self.bind(&i[4], bf, false, "wk")?,
            self.bind(&i[5], bf, false, "wv")?,
        ];
        let vec_ok = |align: usize, kmod: usize| h % kmod == 0 && w.iter().all(|b| addr(b) % align == 0);
        let (idx, wskip) = if vec_ok(16, 8) {
            (0, false)
        } else if vec_ok(8, 4) {
            (1, false)
        } else {
            (2, true)
        };
        let wb: Vec<Bind> = w
            .iter()
            .map(|b| if wskip { Bind { mtl: b.mtl.clone(), off: b.off, skip: b.skip } } else { Bind { mtl: b.mtl.clone(), off: addr(b), skip: 0 } })
            .collect();
        let (x, res, nw) = (self.bind(&i[0], bf, false, "x")?, self.bind(&i[1], bf, false, "res")?, self.bind(&i[2], bf, false, "w")?);
        let (yq, yk, yv, sum) = (
            self.bind(&o[0], bf, true, "q")?,
            self.bind(&o[1], bf, true, "k")?,
            self.bind(&o[2], bf, true, "v")?,
            self.bind(&o[3], bf, true, "sum")?,
        );
        let p = QkvP {
            k: h as u32,
            nq: nq as u32,
            nk: nk as u32,
            nv: nv as u32,
            wq_skip: if wskip { w[0].skip } else { 0 },
            wk_skip: if wskip { w[1].skip } else { 0 },
            wv_skip: if wskip { w[2].skip } else { 0 },
            x_skip: x.skip,
            res_skip: res.skip,
            nw_skip: nw.skip,
            yq_skip: yq.skip,
            yk_skip: yk.skip,
            yv_skip: yv.skip,
            sum_skip: sum.skip,
            eps: call.attrs.eps,
            w_off: call.attrs.w_offset,
        };
        self.dispatch(
            QKV_VARIANTS[idx],
            &[&wb[0], &wb[1], &wb[2], &x, &res, &nw, &yq, &yk, &yv, &sum],
            &p,
            Grid::Groups { x: nq + nk + nv, y: 1, threads: GLU_THREADS },
        )
    }
}

/// Kernel arguments of `gemv_qkv_norm_t` (`QkvP` in gemv_glu.metal; 64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct QkvP {
    pub k: u32,
    pub nq: u32,
    pub nk: u32,
    pub nv: u32,
    pub wq_skip: u32,
    pub wk_skip: u32,
    pub wv_skip: u32,
    pub x_skip: u32,
    pub res_skip: u32,
    pub nw_skip: u32,
    pub yq_skip: u32,
    pub yk_skip: u32,
    pub yv_skip: u32,
    pub sum_skip: u32,
    pub eps: f32,
    pub w_off: f32,
}
