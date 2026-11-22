//! The screened greedy head on Metal (lane ink-fp8; contract `intarsia_core::screen`): `head_screen_f32` and
//! `screen_argmax_f32` (kernels/metal/screen.metal) and `logits_gather_f32o_v` (kernels/metal/gemv_f32out.metal, next to
//! the `gemv_f32o_v` whose per-row arithmetic it repeats, so a gathered logit is bitwise the serving head's).

use crate::dense::Grid;
use crate::MetalBackend;
use intarsia_core::backend::{Exactness, KernelImpl, Tol};
use intarsia_core::{DType, Error, Op, OpCall, Result};

pub const HEAD_SCREEN_F32: &str = "head_screen_f32";
pub const LOGITS_GATHER_F32O_V: &str = "logits_gather_f32o_v";
pub const SCREEN_ARGMAX_F32: &str = "screen_argmax_f32";

pub const KERNELS: [(&str, Op); 3] =
    [(HEAD_SCREEN_F32, Op::HeadScreen), (LOGITS_GATHER_F32O_V, Op::LogitsGather), (SCREEN_ARGMAX_F32, Op::ScreenArgmax)];

pub fn shader_names() -> Vec<&'static str> {
    vec![HEAD_SCREEN_F32, LOGITS_GATHER_F32O_V, SCREEN_ARGMAX_F32]
}

/// Threads per threadgroup of the two row kernels (one threadgroup per activation row).
const ROW_THREADS: usize = 1024;
/// Threadgroups (of 4 simdgroups) per activation row of the gather: 1024 simdgroups stream the full head in the fallback.
const GATHER_GROUPS: usize = 256;

/// f32 outputs of the gather: accumulation-order noise against the cpu (kernels.rs `F32_TOL`).
const F32_TOL: Tol = Tol::new(1e-3, 2e-4);

pub fn register(k: KernelImpl) -> KernelImpl {
    let bf = [DType::Bf16];
    match k.name.as_str() {
        // every kernel here computes each activation row on its own: rows-exact
        HEAD_SCREEN_F32 => k.acts(bf).exact(Exactness::Exact).tol(Tol::EXACT).rows_exact(),
        // the 16-byte form of gemv_f32o_v: K % 8 == 0 (alignment is checked at launch)
        LOGITS_GATHER_F32O_V => k.acts(bf).weights(bf).when(|s| s.k % 8 == 0).exact(Exactness::Deterministic).tol(F32_TOL).rows_exact(),
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
                if t == 0 || v == 0 {
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
                let p = P { a: dim(d, "D")?, b: dim(v, "V")?, c: dim(c, "C")?, pad: 0 };
                let groups = GATHER_GROUPS.min(v.div_ceil(4));
                self.dispatch(LOGITS_GATHER_F32O_V, &[&w, &x, &y, &cand, &cnt], &p, Grid::Groups { x: groups, y: t, threads: 128 })
            }
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
