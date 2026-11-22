//! Launchers of the Inkling-2-Mini attention ops (`InkKvPrologue`, `RelSinkAttn`; contract `intarsia_core::inkling2::ops`,
//! shaders kernels/metal/inkling2_attn.metal with the design notes in their header; lane ink-m-attn).
//!
//! * `ink_kv_prologue_bf16`: one dispatch (plus `ink_ring_write_bf16` when a long chunk would overwrite conv-ring rows other
//!   rows still read: T > Rr - K + 1).
//! * `rel_sink_attn_bf16`: stage 1 (`ink_attn_s1_g{G}`, (kv head, split, row) threadgroups), the combine when a row can have
//!   more than one split, and `ink_kv_write_bf16` when the cache write could race the reads (ring, T > Cr - window + 1).
//!   Any T; rows-exact (row t of a T-row call is bitwise the 1-row call at its position, host or device position): the split
//!   count and the key partition are functions of the row's position and the window only.
//!
//! All dispatches of a call are encoded in one `with_encoder` (one launch); the serial encoder orders them.

use crate::dense::Bind;
use crate::{set_bytes, MetalBackend, Pso};
use intarsia_core::{DType, Error, OpCall, Result, ShapeInfo};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

pub const INK_KV_PROLOGUE_BF16: &str = "ink_kv_prologue_bf16";
pub const REL_SINK_ATTN_BF16: &str = "rel_sink_attn_bf16";

const RING_WRITE: &str = "ink_ring_write_bf16";
const S1: [&str; 4] = ["ink_attn_s1_g1", "ink_attn_s1_g2", "ink_attn_s1_g3", "ink_attn_s1_g4"];
const COMB: &str = "ink_attn_comb_bf16";
const KV_WRITE: &str = "ink_kv_write_bf16";

/// `INK_MAXS` / `INK_SPLIT_KEYS` / `INK_NSG` of the shader.
pub const MAX_SPLIT: usize = 32;
const SPLIT_KEYS: usize = 64;
const NSG: usize = 4;

pub fn shader_names() -> Vec<&'static str> {
    let mut v = vec![INK_KV_PROLOGUE_BF16, RING_WRITE, COMB, KV_WRITE];
    v.extend(S1);
    v
}

/// Kernel arguments (`InkP` in the shader).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct InkP {
    t: u32,
    h: u32,
    hk: u32,
    d: u32,
    dr: u32,
    e: u32,
    k: u32,
    rr: u32,
    cr: u32,
    pos0: u32,
    use_dev: u32,
    flags: u32,
    tau_floor: u32,
    window: u32,
    nq: u32,
    pw: u32,
    qk_clip: f32,
    rel_clip: f32,
    eps: f32,
    tau_alpha: f32,
    scale: f32,
    cscale: f32,
    f2: f32,
    f3: f32,
    sk: [u32; 16],
}

fn u(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal ink attn: {what} = {v} exceeds u32")))
}

/// Shader: `ink_nsplit`.
pub fn nsplit(vis: usize) -> usize {
    vis.div_ceil(SPLIT_KEYS).clamp(1, MAX_SPLIT)
}

/// Shapes `ink_kv_prologue_bf16` takes (d_rel <= 32 and the weight dtypes are checked at launch).
pub fn prologue_ok(s: &ShapeInfo) -> bool {
    s.n_kv_heads > 0 && s.n_heads % s.n_kv_heads == 0 && s.n_heads / s.n_kv_heads <= 4 && (1..=256).contains(&s.head_dim)
}

/// Shapes `rel_sink_attn_bf16` takes.
pub fn attn_ok(s: &ShapeInfo) -> bool {
    s.n_kv_heads > 0
        && s.n_heads % s.n_kv_heads == 0
        && (1..=4).contains(&(s.n_heads / s.n_kv_heads))
        && matches!(s.head_dim, 8 | 16 | 32 | 64 | 128)
        && s.m >= 1
}

/// Scratch of `rel_sink_attn_bf16`: partial accumulators `[T][H][MAX_SPLIT][D]` + `(max, sum)` `[T][H][MAX_SPLIT]`, f32.
pub fn attn_scratch(s: &ShapeInfo) -> usize {
    s.m * s.n_heads * MAX_SPLIT * (s.head_dim + 2) * 4
}

fn encode(
    enc: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
    pso: &Pso,
    binds: &[&Bind],
    p: &InkP,
    groups: Option<(usize, usize, usize, usize)>,
    threads: usize,
) {
    enc.setComputePipelineState(pso);
    for (i, b) in binds.iter().enumerate() {
        // SAFETY: offsets lie inside the buffers (TensorRef bounds were checked by `OpCall::check`).
        unsafe { enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i) };
    }
    set_bytes(enc, p, binds.len());
    match groups {
        Some((x, y, z, tpg)) => enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: x, height: y, depth: z },
            MTLSize { width: tpg, height: 1, depth: 1 },
        ),
        None => {
            let w = pso.maxTotalThreadsPerThreadgroup().min(256);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: threads, height: 1, depth: 1 },
                MTLSize { width: w, height: 1, depth: 1 },
            );
        }
    }
}

impl MetalBackend {
    fn ink_pso(&self, name: &str) -> Result<&Pso> {
        self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))
    }

    fn ink_aligned16(&self, t: &intarsia_core::backend::TensorRef, what: &str) -> Result<()> {
        let a = self.mbuf(&t.buf)?.base + t.offset;
        if a % 16 != 0 {
            return Err(Error::invalid(format!("metal rel_sink_attn_bf16: {what} must be 16-byte aligned")));
        }
        Ok(())
    }

    pub(crate) fn launch_inkattn(&self, kernel: &str, call: &OpCall) -> Result<()> {
        match kernel {
            INK_KV_PROLOGUE_BF16 => self.launch_ink_prologue(call),
            REL_SINK_ATTN_BF16 => self.launch_rel_sink(call),
            _ => Err(Error::NoKernel(format!("metal kernel '{kernel}' for op '{}'", call.op))),
        }
    }

    fn launch_ink_prologue(&self, call: &OpCall) -> Result<()> {
        let (i, o, a) = (&call.inputs, &call.outputs, &call.attrs);
        let ink = &a.ink;
        let bf = DType::Bf16;
        let (t, h, hk, d, dr) = (i[0].shape[0], a.n_heads, a.n_kv_heads, a.head_dim, ink.d_rel);
        if t == 0 {
            return Ok(());
        }
        let k = i[3].shape[0];
        let e = i[7].shape[1];
        let rr = o[4].shape[0];
        if dr > 32 || h / hk > 4 || d > 256 {
            return Err(Error::invalid(format!("metal {INK_KV_PROLOGUE_BF16}: d_rel {dr} > 32, group > 4 or head_dim > 256")));
        }
        let names = ["pq", "pk", "pv", "k taps", "v taps", "q scale", "r scale", "rel proj"];
        let mut binds: Vec<Bind> = Vec::with_capacity(15);
        for (x, w) in i.iter().take(8).zip(names) {
            binds.push(self.bind(x, bf, false, w)?);
        }
        for (x, w) in o.iter().take(6).zip(["q", "rel", "k", "v", "k ring", "v ring"]) {
            binds.push(self.bind(x, bf, true, w)?);
        }
        let (pos, use_dev) = self.pos_bind(call)?;
        let pos_b = match pos {
            Some(b) => b,
            None => self.bind(&i[0], bf, false, "pos (unused)")?,
        };
        let fused = t + k <= rr + 1;
        let mut p = InkP {
            t: u(t, "T")?,
            h: u(h, "H")?,
            hk: u(hk, "Hk")?,
            d: u(d, "D")?,
            dr: u(dr, "dr")?,
            e: u(e, "E")?,
            k: u(k, "K")?,
            rr: u(rr, "Rr")?,
            pos0: u(a.pos0, "pos0")?,
            use_dev,
            flags: u32::from(ink.dyn_conv) | if fused { 4 } else { 0 },
            tau_floor: u(ink.tau_floor, "tau_floor")?,
            nq: u(i[0].shape[1], "Nq")?,
            pw: u(i[1].shape[1], "pk width")?,
            qk_clip: ink.qk_clip,
            rel_clip: ink.rel_clip,
            eps: a.eps,
            tau_alpha: ink.tau_alpha,
            cscale: if ink.dyn_conv { ((2 * k + 1) as f32).powf(-0.5) } else { ((k + 1) as f32).powf(-0.5) },
            ..Default::default()
        };
        for (s, b) in p.sk.iter_mut().zip(&binds) {
            *s = b.skip;
        }
        u(t * hk * d * 2, "T*Ckv")?;
        let threads = d.max(dr).max(32).next_multiple_of(32);
        let pso = self.ink_pso(INK_KV_PROLOGUE_BF16)?;
        let pso_rw = self.ink_pso(RING_WRITE)?;
        let all: Vec<&Bind> = binds.iter().chain(std::iter::once(&pos_b)).collect();
        let rw: Vec<&Bind> = vec![&binds[1], &binds[2], &binds[12], &binds[13], &pos_b];
        self.with_encoder(|enc| {
            encode(enc, pso, &all, &p, Some((hk, t, 1, threads)), 0);
            if !fused {
                encode(enc, pso_rw, &rw, &p, None, t * hk * d);
            }
        })
    }

    fn launch_rel_sink(&self, call: &OpCall) -> Result<()> {
        let (i, o, a) = (&call.inputs, &call.outputs, &call.attrs);
        let bf = DType::Bf16;
        let (t, h, hk, d) = (i[0].shape[0], a.n_heads, a.n_kv_heads, a.head_dim);
        if t == 0 {
            return Ok(());
        }
        let g = h / hk;
        let cr = o[1].shape[1];
        let e = i[5].shape[2];
        let ring = a.ink.kv_ring;
        for (x, w) in [(&i[0], "q"), (&i[1], "k"), (&i[2], "v"), (&o[1], "k cache"), (&o[2], "v cache")] {
            self.ink_aligned16(x, w)?;
        }
        let q = self.bind(&i[0], bf, false, "q")?;
        let kn = self.bind(&i[1], bf, false, "k")?;
        let vn = self.bind(&i[2], bf, false, "v")?;
        let kc = self.bind(&o[1], bf, true, "k cache")?;
        let vc = self.bind(&o[2], bf, true, "v cache")?;
        let rel = self.bind(&i[5], bf, false, "rel")?;
        let pq = self.bind(&i[6], bf, false, "pq")?;
        let out = self.bind(&o[0], bf, true, "out")?;
        let (pos, use_dev) = self.pos_bind(call)?;
        let pos_b = match pos {
            Some(b) => b,
            None => self.bind(&i[0], bf, false, "pos (unused)")?,
        };
        // grid height: the most splits a row of the call can have
        let gy = if use_dev == 1 {
            nsplit(if a.window > 0 { a.window.min(cr) } else { cr })
        } else {
            let last = a.pos0 + t - 1;
            let lo = if a.window > 0 { (last + 1).saturating_sub(a.window) } else { 0 };
            nsplit(last - lo + 1)
        };
        let scratch = match &call.scratch {
            Some(s) => s,
            None if gy == 1 => &i[0],
            None => return Err(Error::invalid("metal rel_sink_attn_bf16: the scratch workspace (OpCall::scratch) is missing")),
        };
        if gy > 1 && scratch.numel() * scratch.dtype.size_bytes() < t * h * MAX_SPLIT * (d + 2) * 4 {
            return Err(Error::invalid("metal rel_sink_attn_bf16: scratch too small"));
        }
        let sc = self.bind(scratch, scratch.dtype, true, "scratch")?;
        if sc.skip != 0 || (self.mbuf(&scratch.buf)?.base + scratch.offset) % 4 != 0 {
            return Err(Error::invalid("metal rel_sink_attn_bf16: scratch must be 4-byte aligned"));
        }
        let fused = !ring || t + a.window <= cr + 1;
        let mut p = InkP {
            t: u(t, "T")?,
            h: u(h, "H")?,
            hk: u(hk, "Hk")?,
            d: u(d, "D")?,
            dr: u(a.ink.d_rel, "dr")?,
            e: u(e, "E")?,
            cr: u(cr, "Cr")?,
            pos0: u(a.pos0, "pos0")?,
            use_dev,
            flags: if ring { 2 } else { 0 } | if fused { 4 } else { 0 },
            window: u(a.window, "window")?,
            nq: u(i[6].shape[1], "Nq")?,
            scale: a.scale,
            ..Default::default()
        };
        for (s, b) in p.sk.iter_mut().zip([&q, &kn, &vn, &kc, &vc, &rel, &pq, &out]) {
            *s = b.skip;
        }
        u(t * h * MAX_SPLIT * (d + 2), "scratch floats")?;
        u(hk * cr * d, "cache elements")?;
        let s1 = self.ink_pso(S1[g - 1])?;
        let comb = self.ink_pso(COMB)?;
        let wr = self.ink_pso(KV_WRITE)?;
        self.with_encoder(|enc| {
            encode(enc, s1, &[&q, &kn, &vn, &kc, &vc, &rel, &pq, &out, &pos_b, &sc], &p, Some((hk, gy, t, NSG * 32)), 0);
            if gy > 1 {
                encode(enc, comb, &[&pq, &out, &pos_b, &sc], &p, Some((h, t, 1, d.max(32).next_multiple_of(32))), 0);
            }
            if !fused {
                encode(enc, wr, &[&kn, &vn, &kc, &vc, &pos_b], &p, None, t * hk * d);
            }
        })
    }
}
