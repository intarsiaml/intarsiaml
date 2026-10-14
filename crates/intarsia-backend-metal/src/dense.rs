//! Launch planning and encoding of the dense ops (everything but the batch-1 `Matmul` GEMV, which lives in `lib.rs`):
//! `Embedding`, `RmsNorm`, `RmsNormAdd`, `Add`, `QkNorm`, `Rope`, `SiluGlu`, `Argmax`, `KvAppend`, `AttnPrefill`,
//! `AttnDecode` (kernels/metal/attention.metal), the tiled GEMM for
//! `Matmul` / `Logits` and the f32-output GEMV for `Logits` with M == 1. Shaders: kernels/metal/{dense_ops,gemm_bf16,
//! gemv_f32out}.metal; semantics: the cpu-ref oracle (crates/intarsia-backend-cpu/src/reference.rs).
//!
//! Every tensor is bound at its byte offset rounded down to 4 bytes and the 0 or 1 leftover bf16 element goes to the
//! shader as a `skip` (kernels/metal/common.h). f32 / i32 tensors must be 4-byte aligned. Dtypes are checked here, so
//! a launch of a kernel on operands it does not take is an `Error::Invalid`, not garbage.

use crate::kernels::*;
use crate::{set_bytes, GemvParams, MetalBackend, MtlBuffer};
use intarsia_core::backend::TensorRef;
use intarsia_core::{DType, Error, Op, OpCall, Result};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

/// Kernel arguments of the dense shaders (`OpP` in kernels/metal/common.h; 64 bytes). Field use is per kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OpP {
    pub dims: [u32; 8],
    pub skips: [u32; 5],
    pub f0: f32,
    pub f1: f32,
    pub pad: u32,
}

/// One tensor bound to a shader argument: the MTLBuffer, the 4-byte aligned byte offset, the leftover elements.
struct Bind {
    mtl: MtlBuffer,
    off: usize,
    skip: u32,
}

/// Threads per threadgroup of the row-reduction kernels (rmsnorm, argmax): a multiple of 32, at most 1024.
const ROW_THREADS: usize = 256;
/// Threads per threadgroup of `argmax_f32_v4`.
const ARGMAX_V4_THREADS: usize = 1024;
/// Threads per threadgroup of the GEMM (4 simdgroups).
const GEMM_THREADS: usize = 128;
const GEMM_TILE: usize = 32;
/// Query rows per threadgroup of `attn_prefill_bf16` (one simdgroup each; `ATT_PF_BQ` in attention.metal). The
/// threadgroup has `ATTN_THREADS` threads; the decode kernels use `ATTN_DEC_SG` simdgroups.
const ATTN_PF_BQ: usize = 8;
const ATTN_THREADS: usize = 256;
/// Simdgroups per threadgroup of the decode kernels (`ATT_DEC_SG` in attention.metal: keep in sync, the shader's
/// threadgroup arrays are sized by it).
const ATTN_DEC_SG: usize = 16;

enum Grid {
    /// `n` threads in a line (`dispatchThreads`: the last threadgroup may be partial).
    Threads(usize),
    /// `x` by `y` threadgroups of `threads` threads.
    Groups { x: usize, y: usize, threads: usize },
}

fn dim(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    /// The low-latency row-norm kernel for rows of `h` elements as (pipeline, threads per threadgroup): it keeps the
    /// 256-thread kernel's element-to-thread assignment, so results are bitwise those of `rmsnorm_bf16` /
    /// `rmsnorm_add_bf16`. Rows of up to 8192 elements; `None` (the 256-thread kernel) for wider ones.
    fn norm_ex(&self, h: usize, names: &[&'static str; 3]) -> Option<(&'static str, usize)> {
        let (name, threads) = match h {
            0..=ROW_THREADS => (names[0], h.next_multiple_of(32)),
            ..=4096 => (names[1], ROW_THREADS),
            ..=8192 => (names[2], ROW_THREADS),
            _ => return None,
        };
        (self.pso(name)?.maxTotalThreadsPerThreadgroup() >= threads).then_some((name, threads))
    }

    fn bind(&self, t: &TensorRef, dt: DType, write: bool, what: &str) -> Result<Bind> {
        if t.dtype != dt {
            return Err(Error::invalid(format!("metal {what}: expected {dt}, got {}", t.dtype)));
        }
        let mb = self.mbuf(&t.buf)?;
        if write && mb.read_only {
            return Err(Error::backend(format!("metal {what}: output buffer is read-only (wrapped host memory)")));
        }
        let addr = mb.base + t.offset;
        let es = dt.size_bytes();
        if addr % es != 0 {
            return Err(Error::invalid(format!("metal {what}: {dt} tensor at byte {addr} is not {es}-byte aligned")));
        }
        Ok(Bind { mtl: mb.mtl.clone(), off: addr & !3, skip: ((addr & 3) / es) as u32 })
    }

    /// Encode one dispatch of shader `name`: `binds` at indices 0.., then `p` as inline constants.
    fn dispatch(&self, name: &str, binds: &[&Bind], p: &OpP, grid: Grid) -> Result<()> {
        let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
        self.with_encoder(|enc| {
            enc.setComputePipelineState(pso);
            for (i, b) in binds.iter().enumerate() {
                // SAFETY: the offset lies inside the buffer (TensorRef bounds were checked by `OpCall::check`).
                unsafe { enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i) };
            }
            set_bytes(enc, p, binds.len());
            match grid {
                Grid::Threads(n) => {
                    let w = pso.maxTotalThreadsPerThreadgroup().min(256);
                    enc.dispatchThreads_threadsPerThreadgroup(
                        MTLSize { width: n, height: 1, depth: 1 },
                        MTLSize { width: w, height: 1, depth: 1 },
                    );
                }
                Grid::Groups { x, y, threads } => enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: x, height: y, depth: 1 },
                    MTLSize { width: threads, height: 1, depth: 1 },
                ),
            }
        })
    }

    /// Launch one of the dense kernels (everything `kernels::TABLE` lists except `matmul_bf16_gemv`).
    pub(crate) fn launch_dense(&self, kernel: &str, call: &OpCall) -> Result<()> {
        let known = TABLE.iter().any(|&(n, op)| n == kernel && op == call.op);
        if !known || kernel == MATMUL_BF16_GEMV {
            return Err(Error::NoKernel(format!("metal kernel '{kernel}' for op '{}'", call.op)));
        }
        let (i, o, a) = (&call.inputs, &call.outputs, &call.attrs);
        let bf = DType::Bf16;
        match kernel {
            EMBEDDING_BF16 => {
                let (t, v, h) = (i[0].shape[0], i[1].shape[0], i[1].shape[1]);
                let n = t * h;
                if n == 0 {
                    return Ok(());
                }
                let (ids, table, out) = (
                    self.bind(&i[0], DType::I32, false, "ids")?,
                    self.bind(&i[1], bf, false, "table")?,
                    self.bind(&o[0], bf, true, "out")?,
                );
                let p = OpP {
                    dims: [dim(t, "T")?, dim(h, "H")?, dim(v, "V")?, 0, 0, 0, 0, 0],
                    skips: [0, table.skip, out.skip, 0, 0],
                    ..Default::default()
                };
                dim(n, "T*H")?;
                self.dispatch(kernel, &[&ids, &table, &out], &p, Grid::Threads(n))
            }
            RMSNORM_BF16 | QK_NORM_BF16 => {
                let h = *i[0].shape.last().unwrap();
                let rows: usize = i[0].shape[..i[0].shape.len() - 1].iter().product();
                if rows * h == 0 {
                    return Ok(());
                }
                let (x, w, out) = (
                    self.bind(&i[0], bf, false, "x")?,
                    self.bind(&i[1], bf, false, "w")?,
                    self.bind(&o[0], bf, true, "out")?,
                );
                let p = OpP {
                    dims: [dim(h, "H")?, 0, 0, 0, 0, 0, 0, 0],
                    skips: [x.skip, w.skip, out.skip, 0, 0],
                    f0: a.eps,
                    f1: a.w_offset,
                    pad: 0,
                };
                if let Some((names, threads)) = self.norm_ex(h, &RMSNORM_EX) {
                    return self.dispatch(names, &[&x, &w, &out], &p, Grid::Groups { x: rows, y: 1, threads });
                }
                self.dispatch(
                    // qk_norm is rmsnorm over rows = T*N of width D
                    RMSNORM_BF16,
                    &[&x, &w, &out],
                    &p,
                    Grid::Groups { x: rows, y: 1, threads: ROW_THREADS },
                )
            }
            RMSNORM_ADD_BF16 => {
                let h = i[0].shape[1];
                let rows = i[0].shape[0];
                if rows * h == 0 {
                    return Ok(());
                }
                let x = self.bind(&i[0], bf, false, "x")?;
                let res = self.bind(&i[1], bf, false, "res")?;
                let w = self.bind(&i[2], bf, false, "w")?;
                let normed = self.bind(&o[0], bf, true, "normed")?;
                let sum = self.bind(&o[1], bf, true, "sum")?;
                let p = OpP {
                    dims: [dim(h, "H")?, 0, 0, 0, 0, 0, 0, 0],
                    skips: [x.skip, res.skip, w.skip, normed.skip, sum.skip],
                    f0: a.eps,
                    f1: a.w_offset,
                    pad: 0,
                };
                if let Some((names, threads)) = self.norm_ex(h, &RMSNORM_ADD_EX) {
                    return self.dispatch(
                        names,
                        &[&x, &res, &w, &normed, &sum],
                        &p,
                        Grid::Groups { x: rows, y: 1, threads },
                    );
                }
                self.dispatch(
                    kernel,
                    &[&x, &res, &w, &normed, &sum],
                    &p,
                    Grid::Groups { x: rows, y: 1, threads: ROW_THREADS },
                )
            }
            ADD_BF16 | SILU_GLU_BF16 => {
                let n = i[0].numel();
                if n == 0 {
                    return Ok(());
                }
                let (x, y, out) = (
                    self.bind(&i[0], bf, false, "a")?,
                    self.bind(&i[1], bf, false, "b")?,
                    self.bind(&o[0], bf, true, "out")?,
                );
                let p = OpP {
                    dims: [dim(n, "elements")?, 0, 0, 0, 0, 0, 0, 0],
                    skips: [x.skip, y.skip, out.skip, 0, 0],
                    ..Default::default()
                };
                self.dispatch(kernel, &[&x, &y, &out], &p, Grid::Threads(n))
            }
            ROPE_BF16 => {
                let (t, n, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
                let total = t * n * d;
                if total == 0 {
                    return Ok(());
                }
                let rd = if a.rotary_dim == 0 { d } else { a.rotary_dim };
                let (x, out) = (self.bind(&i[0], bf, false, "x")?, self.bind(&o[0], bf, true, "out")?);
                let (pos, use_dev) = self.pos_bind(call)?;
                let p = OpP {
                    dims: [
                        dim(t, "T")?,
                        dim(n, "N")?,
                        dim(d, "D")?,
                        dim(rd, "rotary_dim")?,
                        use_dev,
                        dim(a.pos0, "pos0")?,
                        0,
                        0,
                    ],
                    skips: [x.skip, out.skip, 0, 0, 0],
                    f0: a.theta,
                    f1: 0.0,
                    pad: 0,
                };
                dim(total, "T*N*D")?;
                self.dispatch(kernel, &[&x, &out, pos.as_ref().unwrap_or(&x)], &p, Grid::Threads(total))
            }
            KV_APPEND_BF16 => {
                let (t, nkv, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
                let cap = o[0].shape[1];
                let total = t * nkv * d;
                if total == 0 {
                    return Ok(());
                }
                let (src, cache) = (self.bind(&i[0], bf, false, "new")?, self.bind(&o[0], bf, true, "cache")?);
                let (pos, use_dev) = self.pos_bind(call)?;
                let p = OpP {
                    dims: [
                        dim(t, "T")?,
                        dim(nkv, "Nkv")?,
                        dim(d, "D")?,
                        dim(cap, "cap")?,
                        use_dev,
                        dim(a.pos0, "pos0")?,
                        0,
                        0,
                    ],
                    skips: [src.skip, cache.skip, 0, 0, 0],
                    ..Default::default()
                };
                dim(total, "T*Nkv*D")?;
                self.dispatch(kernel, &[&src, &cache, pos.as_ref().unwrap_or(&src)], &p, Grid::Threads(total))
            }
            ARGMAX_F32 | ARGMAX_BF16 => {
                let (r, v) = (i[0].shape[0], i[0].shape[1]);
                if r == 0 {
                    return Ok(());
                }
                let dt = if kernel == ARGMAX_F32 { DType::F32 } else { bf };
                let (l, out) = (self.bind(&i[0], dt, false, "logits")?, self.bind(&o[0], DType::I32, true, "out")?);
                let p = OpP {
                    dims: [dim(v, "V")?, 0, 0, 0, 0, 0, 0, 0],
                    skips: [l.skip, 0, 0, 0, 0],
                    ..Default::default()
                };
                if kernel == ARGMAX_F32 && v % 4 == 0 && l.skip == 0 && l.off % 16 == 0 {
                    return self.dispatch(
                        ARGMAX_F32_V4,
                        &[&l, &out],
                        &p,
                        Grid::Groups { x: r, y: 1, threads: ARGMAX_V4_THREADS },
                    );
                }
                self.dispatch(kernel, &[&l, &out], &p, Grid::Groups { x: r, y: 1, threads: ROW_THREADS })
            }
            ATTN_PREFILL_BF16 | ATTN_DECODE_BF16 => {
                let (t, n, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
                let (nkv, cap) = (i[1].shape[0], i[1].shape[1]);
                if t * n * d == 0 {
                    return Ok(());
                }
                if d % 2 != 0 || d > 256 {
                    return Err(Error::invalid(format!("{kernel}: head_dim {d} must be even and at most 256")));
                }
                let decode = kernel == ATTN_DECODE_BF16;
                let (q, kc, vc, out) = (
                    self.bind(&i[0], bf, false, "q")?,
                    self.bind(&i[1], bf, false, "k cache")?,
                    self.bind(&i[2], bf, false, "v cache")?,
                    self.bind(&o[0], bf, true, "out")?,
                );
                let (pos, use_dev) = if decode { self.pos_bind(call)? } else { (None, 0) };
                let p = OpP {
                    dims: [
                        dim(t, "T")?,
                        dim(n, "N")?,
                        dim(nkv, "Nkv")?,
                        dim(d, "D")?,
                        dim(cap, "cap")?,
                        dim(a.pos0, "pos0")?,
                        use_dev,
                        0,
                    ],
                    skips: [q.skip, kc.skip, vc.skip, out.skip, 0],
                    f0: a.scale,
                    f1: 0.0,
                    pad: 0,
                };
                dim(t * n * d, "T*N*D")?;
                dim(nkv * cap * d, "Nkv*cap*D")?;
                if decode {
                    let (ka, va) = (self.mbuf(&i[1].buf)?.base + i[1].offset, self.mbuf(&i[2].buf)?.base + i[2].offset);
                    let wide = (d / 8).is_power_of_two() && d % 8 == 0 && ka % 16 == 0 && va % 16 == 0;
                    self.dispatch(
                        if wide { ATTN_DECODE_BF16_V8 } else { kernel },
                        &[&q, &kc, &vc, &out, pos.as_ref().unwrap_or(&q)],
                        &p,
                        Grid::Groups { x: n, y: 1, threads: ATTN_DEC_SG * 32 },
                    )
                } else {
                    self.dispatch(
                        kernel,
                        &[&q, &kc, &vc, &out],
                        &p,
                        Grid::Groups { x: t.div_ceil(ATTN_PF_BQ), y: n, threads: ATTN_THREADS },
                    )
                }
            }
            MATMUL_BF16 | LOGITS_BF16 => {
                let (m, k, n) = (i[0].shape[0], i[0].shape[1], i[1].shape[0]);
                if m * n == 0 {
                    return Ok(());
                }
                let (od, shader) = if call.op == Op::Logits { (DType::F32, GEMM_O_F32) } else { (bf, GEMM_O_BF16) };
                let (x, w, y) = (
                    self.bind(&i[0], bf, false, "x")?,
                    self.bind(&i[1], bf, false, "w")?,
                    self.bind(&o[0], od, true, "y")?,
                );
                let p = OpP {
                    dims: [dim(m, "M")?, dim(n, "N")?, dim(k, "K")?, 0, 0, 0, 0, 0],
                    skips: [x.skip, w.skip, y.skip, 0, 0],
                    ..Default::default()
                };
                self.dispatch(
                    shader,
                    &[&x, &w, &y],
                    &p,
                    Grid::Groups { x: n.div_ceil(GEMM_TILE), y: m.div_ceil(GEMM_TILE), threads: GEMM_THREADS },
                )
            }
            LOGITS_BF16_GEMV => {
                let (m, k, n) = (i[0].shape[0], i[0].shape[1], i[1].shape[0]);
                if m != 1 {
                    return Err(Error::invalid("logits_bf16_gemv: needs M == 1"));
                }
                if n == 0 {
                    return Ok(());
                }
                let (x, w, y) = (
                    self.bind(&i[0], bf, false, "x")?,
                    self.bind(&i[1], bf, false, "w")?,
                    self.bind(&o[0], DType::F32, true, "y")?,
                );
                let (xa, wa) = (self.mbuf(&i[0].buf)?.base + i[0].offset, self.mbuf(&i[1].buf)?.base + i[1].offset);
                let params = GemvParams {
                    k: dim(k, "K")?,
                    n: dim(n, "N")?,
                    w_skip: w.skip,
                    x_skip: x.skip,
                    y_skip: y.skip,
                    pad: [0; 3],
                };
                let (shader, rows, threads) = if k % 8 == 0 && xa % 16 == 0 && wa % 16 == 0 {
                    (GEMV_F32O_V, 4, 128)
                } else {
                    (GEMV_F32O_SCALAR, 8, 256)
                };
                let pso = self.pso(shader).ok_or_else(|| Error::backend(format!("metal: no pipeline '{shader}'")))?;
                self.with_encoder(|enc| {
                    crate::encode_gemv(
                        enc,
                        pso,
                        (&w.mtl, w.off),
                        (&x.mtl, x.off),
                        (&y.mtl, y.off),
                        &params,
                        n.div_ceil(rows) as u32,
                        threads,
                    )
                })
            }
            other => Err(Error::NoKernel(format!("metal kernel '{other}' for op '{}'", call.op))),
        }
    }

    /// The device-side position tensor of `Rope` / `KvAppend` (`OpCall::pos`) as a binding plus the shader's flag.
    fn pos_bind(&self, call: &OpCall) -> Result<(Option<Bind>, u32)> {
        match &call.pos {
            Some(p) => Ok((Some(self.bind(p, DType::I32, false, "pos")?), 1)),
            None => Ok((None, 0)),
        }
    }
}
