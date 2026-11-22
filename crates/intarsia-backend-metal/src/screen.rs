//! The screened greedy head on Metal (lane ink-fp8; contract `intarsia_core::screen`): `head_screen_f32` and
//! `screen_argmax_f32` (kernels/metal/screen.metal), and the gather, which is the serving 1-row head kernel itself in its
//! runtime index mode (brief 14 option B): `logits_gather_f32o_v` launches `gemv_f32o_v` (the bf16 head's
//! `logits_bf16_gemv`), `logits_gather_p12` launches P12's 1-row `p12_gemv_f32o` instance (the P12 head's
//! `matmul_p12_logits_gemv`). One function and pipeline for head and gather, so a gathered logit is the head's by
//! construction.

use crate::dense::Grid;
use objc2_metal::MTLComputeCommandEncoder;
use crate::MetalBackend;
use intarsia_core::{Exactness, KernelImpl, Tol};
use intarsia_core::{DType, Error, Op, OpCall, Result};

pub const HEAD_SCREEN_F32: &str = "head_screen_f32";
pub const LOGITS_GATHER_F32O_V: &str = "logits_gather_f32o_v";
pub const LOGITS_GATHER_P12: &str = "logits_gather_p12";
pub const SCREEN_ARGMAX_F32: &str = "screen_argmax_f32";

pub const KERNELS: [(&str, Op); 4] = [
    (HEAD_SCREEN_F32, Op::HeadScreen),
    (LOGITS_GATHER_F32O_V, Op::LogitsGather),
    (LOGITS_GATHER_P12, Op::LogitsGather),
    (SCREEN_ARGMAX_F32, Op::ScreenArgmax),
];

/// The screen's own shaders (the gathers run the head kernels' pipelines, compiled with those).
pub fn shader_names() -> Vec<&'static str> {
    vec![HEAD_SCREEN_F32, SCREEN_ARGMAX_F32]
}

/// Threads per threadgroup of the two row kernels (one threadgroup per activation row).
const ROW_THREADS: usize = 1024;

/// f32 outputs of the gather: accumulation-order noise against the cpu (kernels.rs `F32_TOL`).
const F32_TOL: Tol = Tol::new(1e-3, 2e-4);

pub fn register(k: KernelImpl) -> KernelImpl {
    let bf = [DType::Bf16];
    match k.name.as_str() {
        // every kernel here computes each activation row on its own: rows-exact
        HEAD_SCREEN_F32 => k.acts(bf).exact(Exactness::Exact).tol(Tol::EXACT).rows_exact(),
        // gemv_f32o_v in its index mode: one activation row (the screen serves 1-row heads), its 16-byte form (K % 8 == 0;
        // alignment is checked at launch). Registered rows-exact like the other screen kernels (the screen runs its calls so)
        LOGITS_GATHER_F32O_V => k.acts(bf).weights(bf).when(|s| s.m == 1 && s.k % 8 == 0).exact(Exactness::Deterministic).tol(F32_TOL).rows_exact(),
        // P12's 1-row p12_gemv_f32o in its index mode (K % 16 == 0, the P12 kernels' form)
        LOGITS_GATHER_P12 => k.acts(bf).weights([DType::U8]).when(|s| s.m == 1 && s.k % 16 == 0).exact(Exactness::Deterministic).tol(F32_TOL).rows_exact(),
        SCREEN_ARGMAX_F32 => k.acts([DType::F32]).exact(Exactness::Exact).tol(Tol::EXACT).rows_exact(),
        other => unreachable!("screen kernel {other}"),
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct P {
    a: u32,
    b: u32,
    c: u32,
    pad: u32,
}

fn dim(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal screen: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    pub(crate) fn launch_screen(&self, kernel: &str, call: &OpCall) -> Result<()> {
        let (i, o) = (&call.inputs, &call.outputs);
        let whole = |b: &crate::dense::Bind, what: &str| -> Result<()> {
            if b.skip != 0 {
                return Err(Error::invalid(format!("metal {kernel}: {what} must be 4-byte aligned")));
            }
            Ok(())
        };
        match (kernel, call.op) {
            (HEAD_SCREEN_F32, Op::HeadScreen) => {
                let (t, v, c, d) = (i[0].shape[0], i[0].shape[1], o[0].shape[1], i[1].shape[1]);
                if t == 0 {
                    return Ok(());
                }
                let a = self.bind(&i[0], DType::F32, false, "a")?;
                let h = self.bind(&i[1], DType::Bf16, false, "h")?;
                let beta = self.bind(&i[2], DType::F32, false, "beta")?;
                let cand = self.bind(&o[0], DType::I32, true, "cand")?;
                let cnt = self.bind(&o[1], DType::I32, true, "cnt")?;
                whole(&h, "h")?;
                let p = P { a: dim(v, "V")?, b: dim(d, "D")?, c: dim(c, "C")?, pad: 0 };
                self.dispatch(HEAD_SCREEN_F32, &[&a, &h, &beta, &cand, &cnt], &p, Grid::Groups { x: t, y: 1, threads: ROW_THREADS })
            }
            (LOGITS_GATHER_F32O_V, Op::LogitsGather) => {
                let (t, d, v, c) = (i[0].shape[0], i[0].shape[1], i[1].shape[0], i[2].shape[1]);
                if t != 1 || c == 0 {
                    return Err(Error::invalid(format!("metal {kernel}: one activation row and a candidate cap (T {t}, C {c})")));
                }
                if v == 0 {
                    return Ok(());
                }
                let (xa, wa) = (self.mbuf(&i[0].buf)?.base + i[0].offset, self.mbuf(&i[1].buf)?.base + i[1].offset);
                if d % 8 != 0 || xa % 16 != 0 || wa % 16 != 0 {
                    return Err(Error::invalid(format!(
                        "metal {kernel}: needs D % 8 == 0 and 16-byte aligned h and W (gemv_f32o_v's form; D {d}, h {xa}, W {wa})"
                    )));
                }
                let w = self.bind(&i[1], DType::Bf16, false, "W")?;
                let x = self.bind(&i[0], DType::Bf16, false, "h")?;
                let y = self.bind(&o[0], DType::F32, true, "y")?;
                let cand = self.bind(&i[2], DType::I32, false, "cand")?;
                let cnt = self.bind(&i[3], DType::I32, false, "cnt")?;
                // gemv_f32o_v's parameters as logits_bf16_gemv passes them, plus the index mode (pad0 = the cap)
                let params = crate::GemvParams { k: dim(d, "D")?, n: dim(v, "V")?, w_skip: w.skip, x_skip: x.skip, y_skip: y.skip, pad: [dim(c, "C")?, 0, 0] };
                let name = crate::kernels::GEMV_F32O_V;
                let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
                self.with_encoder(|enc| {
                    // SAFETY: cand / cnt are their own bindings, inside their buffers (OpCall::check).
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&cand.mtl), cand.off, 4);
                        enc.setBuffer_offset_atIndex(Some(&cnt.mtl), cnt.off, 5);
                    }
                    // the head's grid (logits_bf16_gemv: 4 rows per threadgroup of 128): every row in the fallback
                    crate::encode_gemv(enc, pso, (&w.mtl, w.off), (&x.mtl, x.off), (&y.mtl, y.off), &params, v.div_ceil(4) as u32, 128)
                })
            }
            (LOGITS_GATHER_P12, Op::LogitsGather) => self.launch_p12_logits_gather(call),
            (SCREEN_ARGMAX_F32, Op::ScreenArgmax) => {
                let (t, c, v) = (i[0].shape[0], i[0].shape[1], i[2].shape[1]);
                if t == 0 {
                    return Ok(());
                }
                let cand = self.bind(&i[0], DType::I32, false, "cand")?;
                let cnt = self.bind(&i[1], DType::I32, false, "cnt")?;
                let y = self.bind(&i[2], DType::F32, false, "y")?;
                let tok = self.bind(&o[0], DType::I32, true, "tok")?;
                let p = P { a: dim(v, "V")?, b: 0, c: dim(c, "C")?, pad: 0 };
                self.dispatch(SCREEN_ARGMAX_F32, &[&cand, &cnt, &y, &tok], &p, Grid::Groups { x: t, y: 1, threads: ROW_THREADS })
            }
            _ => Err(Error::invalid(format!("metal: {kernel} does not run {}", call.op))),
        }
    }
}
