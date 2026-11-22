//! Inkling-2-Mini decode controller fusion on Metal (lane ink-dfuse; kernels/metal/inkling2_ctrl.metal, compiled into the
//! metal4.0 library for its device-scope fences). Contract: `intarsia_core::inkling2::ops` (`InkRoute`, op 76).
//!
//! `ink_route_bf16`: the router GEMV and `MoeRoute` in one dispatch. Grid `(ceil(E / 4), T)` threadgroups of 4 simdgroups;
//! simdgroup s of threadgroup (j, t) computes logit `4j + s` of row t exactly as `gemv_f32o_v` / `gemv_f32o_scalar` do (the
//! same form choice as `logits_bf16_gemv`), the last threadgroup of row t to finish (a counter of the sync arena, sync.rs)
//! routes the row exactly as `ink_moe_route`. So the outputs are bitwise `Logits` + `MoeRoute` on Metal, and a row never
//! depends on the call's other rows (rows-exact by construction: no separate verify twin). Without the metal4.0 library the
//! kernel is not registered and the model graph keeps `Logits` + `MoeRoute`.

use crate::{set_bytes, MetalBackend};
use intarsia_core::dsv41::ops::ScoreFunc;
use intarsia_core::{DType, Error, Exactness, KernelImpl, Op, OpCall, Result, ShapeInfo, Tol};
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

/// `Op::InkRoute`, bf16 x and router weight (any D, E <= 1024, K <= 32, every score function).
pub const INK_ROUTE_BF16: &str = "ink_route_bf16";

/// (kernel, op) entries of `kernels::TABLE`.
pub const KERNELS: [(&str, Op); 1] = [(INK_ROUTE_BF16, Op::InkRoute)];

const ROUTE: &str = "ink_route";
/// Shader entry points of kernels/metal/inkling2_ctrl.metal (all in the metal4.0 library).
pub const METAL4_SHADERS: [&str; 4] = [ROUTE, "ink_blend_tail_g1", "ink_blend_tail_g8", "ink_blend_tail_g16"];

/// Threads per threadgroup of `ink_blend_tail` (`BT_THREADS`: the h GEMV's 8 simdgroups, 256 channels per gate item).
pub const BLEND_TAIL_THREADS: usize = 256;
/// Most rows of a blend chunk the tail takes (decode, verify, batch); longer chunks keep the three dispatches.
pub const BLEND_TAIL_ROWS_MAX: usize = 16;

/// 0 = `INTARSIA_INK_BLEND_TAIL` (`1` / `on`: on; unset / `0` / `off`: off, the default until the lane's timing lands), 1 = forced
/// on, 2 = forced off ([`set_blend_tail`]).
static BLEND_TAIL: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Force the fused blend tail on or off for this process, `None` = back to the environment (tests and benchmarks: both give
/// the same bits).
pub fn set_blend_tail(on: Option<bool>) {
    BLEND_TAIL.store(match on { None => 0, Some(true) => 1, Some(false) => 2 }, std::sync::atomic::Ordering::Relaxed);
}

fn blend_tail_env() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("INTARSIA_INK_BLEND_TAIL").ok().as_deref().map(str::trim), Some("1") | Some("on")))
}

/// Whether a blend chunk of `n` rows (controller rank `g`, width `d`) takes the fused tail: the metal4.0 library, at most
/// [`BLEND_TAIL_ROWS_MAX`] rows, G % 8 == 0 and G <= 128 (the gate's register rows), and the switch on.
pub(crate) fn blend_tail_on(be: &MetalBackend, n: usize, g: usize, d: usize) -> bool {
    let on = match BLEND_TAIL.load(std::sync::atomic::Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => blend_tail_env(),
    };
    on && be.has_metal4() && (1..=BLEND_TAIL_ROWS_MAX).contains(&n) && g % 8 == 0 && (8..=128).contains(&g) && d > 0
}

/// The tail shader for controller rank `g` (the gate's `GC`, as `ink_blend_gate_g*`).
pub(crate) fn blend_tail_shader(g: usize) -> &'static str {
    match g / 8 {
        0..=1 => "ink_blend_tail_g1",
        2..=8 => "ink_blend_tail_g8",
        _ => "ink_blend_tail_g16",
    }
}

/// Logits per threadgroup (`RT_SG` in the shader).
const RT_SG: usize = 4;
const MAX_EXPERTS: usize = 1024;
const MAX_K: usize = 32;
/// Most rows of one call (one sync counter per row).
pub const ROUTE_ROWS_MAX: usize = 1024;
/// Weights and logits are f32 (the `moe_route_f32` budget).
const ROUTE_TOL: Tol = Tol::new(1e-3, 2e-4);

/// Scratch: the f32 logits `[T, E]` when the call does not take them as an output.
fn route_scratch(s: &ShapeInfo) -> usize {
    s.m * s.experts * 4
}

/// Registration of a kernel of [`KERNELS`] (`kernels::all`).
pub fn register(k: KernelImpl) -> KernelImpl {
    match k.name.as_str() {
        INK_ROUTE_BF16 => k
            .acts([DType::Bf16])
            .weights([DType::Bf16])
            .when(|s| (1..=ROUTE_ROWS_MAX).contains(&s.m) && s.experts <= MAX_EXPERTS && (1..=MAX_K).contains(&s.top_k) && s.top_k <= s.experts)
            .exact(Exactness::Deterministic)
            .tol(ROUTE_TOL)
            .scratch(route_scratch),
        other => unreachable!("inkctrl: no registration for {other}"),
    }
}

/// `InkRouteP` of the shader (64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct InkRouteP {
    t: u32,
    e: u32,
    k: u32,
    d: u32,
    norm_topk: u32,
    sfunc: u32,
    vec: u32,
    ngr: u32,
    sx: u32,
    sw: u32,
    rank: u32,
    pad: u32,
    scale: f32,
    temp: f32,
    fpad: [f32; 2],
}

/// `INTARSIA_INK_ROUTE_RANK=0` / `off`: route with the selection loop of `ink_moe_route` (K dependent shuffle rounds on one
/// simdgroup) instead of the rank selection (every thread of the last threadgroup ranks its experts; elementwise score functions
/// only, softmax always takes the loop). The same bits for finite logits; an A/B knob.
fn route_rank() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| !matches!(std::env::var("INTARSIA_INK_ROUTE_RANK").ok().as_deref().map(str::trim), Some("0") | Some("off")))
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal ink_route: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    /// The controller-fusion kernels.
    pub(crate) fn launch_inkctrl(&self, kernel: &str, call: &OpCall) -> Result<()> {
        if !self.has_metal4() {
            return Err(Error::NoKernel(format!("metal kernel '{kernel}' needs the metal4.0 library")));
        }
        match kernel {
            INK_ROUTE_BF16 if call.op == Op::InkRoute => self.launch_ink_route_fused(call),
            _ => Err(Error::NoKernel(format!("metal kernel '{kernel}' for op '{}'", call.op))),
        }
    }

    fn launch_ink_route_fused(&self, call: &OpCall) -> Result<()> {
        let ([x, w, bias], [wo, ids, lg]) = (call.inputs.as_slice(), call.outputs.as_slice()) else {
            return Err(Error::invalid("metal ink_route: expected 3 inputs / 3 outputs"));
        };
        let (t, d, e, k) = (x.shape[0], x.shape[1], w.shape[0], call.attrs.top_k);
        if e > MAX_EXPERTS || k == 0 || k > MAX_K || k > e || t > ROUTE_ROWS_MAX {
            return Err(Error::invalid(format!(
                "metal ink_route: T = {t}, E = {e}, K = {k} (needs T <= {ROUTE_ROWS_MAX}, E <= {MAX_EXPERTS}, K in 1..={MAX_K})"
            )));
        }
        if t == 0 {
            return Ok(());
        }
        let bf = DType::Bf16;
        let xb = self.bind(x, bf, false, "x")?;
        let wb = self.bind(w, bf, false, "w")?;
        let bb = self.bind(bias, DType::F32, false, "bias")?;
        let wob = self.bind(wo, DType::F32, true, "weights")?;
        let ib = self.bind(ids, DType::I32, true, "ids")?;
        let lb = if lg.shape == [0] {
            let s = call.scratch.as_ref().ok_or_else(|| Error::invalid("metal ink_route: no logits output and no scratch"))?;
            let have = s.numel() * s.dtype.size_bytes();
            if have < t * e * 4 {
                return Err(Error::invalid(format!("metal ink_route: scratch of {have} bytes, needs {}", t * e * 4)));
            }
            let sc = self.bind(s, s.dtype, true, "scratch")?;
            if sc.skip != 0 {
                return Err(Error::invalid("metal ink_route: the scratch must be 4-byte aligned"));
            }
            sc
        } else {
            self.bind(lg, DType::F32, true, "logits")?
        };
        // `logits_bf16_gemv`'s form choice (the same bits per logit)
        let (xa, wa) = (self.mbuf(&x.buf)?.base + x.offset, self.mbuf(&w.buf)?.base + w.offset);
        let vec = d % 8 == 0 && xa % 16 == 0 && wa % 16 == 0;
        let ctr = self.sync_counters(t)?;
        let ngr = e.div_ceil(RT_SG);
        let a = &call.attrs;
        let p = InkRouteP {
            t: u32_of(t, "T")?,
            e: e as u32,
            k: k as u32,
            d: u32_of(d, "D")?,
            norm_topk: a.norm_topk as u32,
            sfunc: match a.dsv.score_func {
                ScoreFunc::Softmax => 0,
                ScoreFunc::Sigmoid => 1,
                ScoreFunc::SqrtSoftplus => 2,
            },
            vec: vec as u32,
            ngr: ngr as u32,
            sx: if vec { 0 } else { xb.skip },
            sw: if vec { 0 } else { wb.skip },
            rank: route_rank() as u32,
            scale: a.scale,
            temp: a.temperature,
            ..Default::default()
        };
        let pso = self.pso(ROUTE).ok_or_else(|| Error::backend(format!("metal: no pipeline '{ROUTE}'")))?;
        let binds = [&xb, &wb, &bb, &wob, &ib, &lb, &ctr];
        self.with_encoder(|enc| {
            enc.setComputePipelineState(pso);
            for (i, b) in binds.iter().enumerate() {
                // SAFETY: offsets lie inside the buffers (TensorRef bounds checked by `OpCall::check`, scratch and the
                // sync arena by this launcher).
                unsafe { enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i) };
            }
            set_bytes(enc, &p, binds.len());
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: ngr, height: t, depth: 1 },
                MTLSize { width: RT_SG * 32, height: 1, depth: 1 },
            );
        })
    }
}
