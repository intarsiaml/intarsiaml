//! Row-table kernels of batched decode (lane mr3, `OpCall::rows`): the one-row (`T = 1`) decode calls of the solo step that
//! touch per-sequence state, `GdnConv`, `GdnDeltaRule`, `KvAppend`, `QkNormRopeKv` and `AttnDecode`, over ALL rows of a batched
//! step in one dispatch. Row `r` of the activations is the single input row of sequence `r`; the state buffers of row `r`
//! (recurrent state, K / V cache) are named by a table of device addresses (`RowTab`, written by the host before every step),
//! and each row's position comes from the position vector (`OpCall::pos`, i32 `[rows]`).
//!
//! Every kernel here is the solo kernel's arithmetic for one row (kernels/metal/{gdn,dense_ops,qk_rope_kv,
//! attention_splitkv}.metal, "Row tables" sections), so row `r` is bitwise the solo launch on sequence `r`
//! (`tests/row_tables.rs`). Attention reproduces the HOST-position work partition of the solo launch per row (split count,
//! simdgroup count), which is what the solo decode step runs.
//!
//! Residency: the kernels reach the state buffers through addresses, not bindings, so Metal must be told they are in use:
//! [`MetalBackend::device_address`] registers every buffer it hands out (weakly), and a row-table dispatch declares the
//! registered set resident in the encoder it is encoded into (`useResources`, once per encoder and per change of the set).

use crate::attn::{decode_splits, head_split, policy, MAX_GROUP, MAX_SG, MAX_SPLIT, TG_FLOATS};
use crate::dense::{Bind, OpP};
use crate::{set_bytes, MetalBackend};
use intarsia_core::{DType, Error, Op, OpCall, Result, ShapeInfo};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

/// Rows a row-table kernel takes (the model's `BATCH_MAX_ROWS`).
pub const ROWTAB_MAX: usize = 16;

pub const GDN_CONV_BF16_ROWS: &str = "gdn_conv_bf16_rows";
pub const GDN_DELTA_RULE_BF16_ROWS: &str = "gdn_delta_rule_bf16_rows";
pub const KV_APPEND_BF16_ROWS: &str = "kv_append_bf16_rows";
pub const QK_NORM_ROPE_KV_BF16_ROWS: &str = "qk_norm_rope_kv_bf16_rows";
pub const ATTN_DECODE_BF16_SPLITKV_ROWS: &str = "attn_decode_bf16_splitkv_rows";
/// Rope with one position per token (`Rope` call whose `pos` holds one i32 per token): the unfused batched chain.
pub const ROPE_BF16_TOKPOS: &str = "rope_bf16_tokpos";

const CONV_ROWS_SHADERS: [&str; 7] = [
    "gdn_conv_bf16_rows_k2",
    "gdn_conv_bf16_rows_k3",
    "gdn_conv_bf16_rows_k4",
    "gdn_conv_bf16_rows_k5",
    "gdn_conv_bf16_rows_k6",
    "gdn_conv_bf16_rows_k7",
    "gdn_conv_bf16_rows_k8",
];
const DELTA_ROWS_SHADERS: [(usize, &str); 4] = [
    (32, "gdn_delta_rule_bf16_rows_dk32"),
    (64, "gdn_delta_rule_bf16_rows_dk64"),
    (128, "gdn_delta_rule_bf16_rows_dk128"),
    (256, "gdn_delta_rule_bf16_rows_dk256"),
];
const QK_ROWS: &str = "qk_norm_rope_kv_bf16_rows";
const ATTN_ROWTAB_SHADERS: [&str; 5] =
    ["attn_decode_rowtab_g1", "attn_decode_rowtab_g2", "attn_decode_rowtab_g4", "attn_decode_rowtab_g8", "attn_decode_rowtab_combine"];
const KV_ROWS: &str = "kv_append_bf16_rows";

pub fn shader_names() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = vec![QK_ROWS, KV_ROWS, ROPE_BF16_TOKPOS];
    v.extend(CONV_ROWS_SHADERS);
    v.extend(DELTA_ROWS_SHADERS.iter().map(|d| d.1));
    v.extend(ATTN_ROWTAB_SHADERS);
    v
}

/// Shapes the row-table kernels take on top of the solo kernel's own predicate: 1..=`ROWTAB_MAX` rows.
pub fn rows_in_range(s: &ShapeInfo) -> bool {
    (1..=ROWTAB_MAX).contains(&s.m)
}

pub fn conv_ok(s: &ShapeInfo) -> bool {
    rows_in_range(s) && crate::gdn::conv_ok(s)
}

pub fn delta_ok(s: &ShapeInfo) -> bool {
    rows_in_range(s) && crate::gdn::delta_ok(s)
}

/// Split-KV decode shapes with 1..=`ROWTAB_MAX` query rows.
pub fn attn_ok(s: &ShapeInfo) -> bool {
    rows_in_range(s) && crate::attn::splitkv_ok(&ShapeInfo { m: 1, ..s.clone() })
}

/// Scratch of the attention row-table kernel: one `splitkv_scratch` slab per row.
pub fn attn_scratch(s: &ShapeInfo) -> usize {
    crate::attn::splitkv_scratch(s) * s.m.clamp(1, ROWTAB_MAX)
}

/// Kernel arguments of `gdn_delta_rule_bf16_rows_dk*` (`GdnDeltaRowsP` in gdn.metal; 56 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct DeltaRowsP {
    hk: u32,
    hv: u32,
    dv: u32,
    stride: u32,
    /// Leading bf16 elements of qkv, a, b, A_log, dt_bias, o.
    skips: [u32; 6],
    index: u32,
    byte_off: u32,
    eps: f32,
    pad: u32,
}

/// Kernel arguments of `qk_norm_rope_kv_bf16_rows` (`QkRopeRowsP` in qk_rope_kv.metal; 80 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct QkRowsP {
    rows: u32,
    n: u32,
    nkv: u32,
    d: u32,
    rd: u32,
    cap: u32,
    stride: u32,
    index: u32,
    /// Leading bf16 elements of q, k, v, q_w, k_w, q_out.
    skips: [u32; 6],
    eps: f32,
    w_offset: f32,
    theta: f32,
    pad: u32,
}

fn dim(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal: {what} = {v} exceeds u32")))
}

/// The row table of a call, bound.
struct Tab {
    b: Bind,
    stride: u32,
    index: u32,
    byte_off: u32,
}

/// Grid of a row-table dispatch.
enum G3 {
    Threads(usize),
    Groups { x: usize, y: usize, z: usize, threads: usize },
}

impl MetalBackend {
    /// Encode one dispatch of `name` into the stream's encoder with the registered indirect buffers resident.
    fn dispatch_rows<P: Copy>(&self, name: &str, binds: &[&Bind], p: &P, grid: G3) -> Result<()> {
        let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
        self.with_encoder_ind(|enc| {
            enc.setComputePipelineState(pso);
            for (i, b) in binds.iter().enumerate() {
                // SAFETY: the offset lies inside the buffer (TensorRef bounds were checked by `OpCall::check`).
                unsafe { enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i) };
            }
            set_bytes(enc, p, binds.len());
            match grid {
                G3::Threads(n) => {
                    let w = pso.maxTotalThreadsPerThreadgroup().min(256);
                    enc.dispatchThreads_threadsPerThreadgroup(
                        MTLSize { width: n, height: 1, depth: 1 },
                        MTLSize { width: w, height: 1, depth: 1 },
                    );
                }
                G3::Groups { x, y, z, threads } => enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: x, height: y, depth: z },
                    MTLSize { width: threads, height: 1, depth: 1 },
                ),
            }
        })
    }

    fn row_tab(&self, call: &OpCall) -> Result<Tab> {
        let r = call.rows.as_ref().ok_or_else(|| Error::invalid(format!("metal {}: a row-table kernel needs OpCall::rows", call.op)))?;
        let b = self.bind(&r.ptrs, DType::U8, false, "row table")?;
        if b.skip != 0 || b.off % 8 != 0 {
            return Err(Error::invalid("metal row table: the address table must be 8-byte aligned"));
        }
        if r.byte_off % 2 != 0 {
            return Err(Error::invalid("metal row table: byte_off must be even"));
        }
        Ok(Tab { b, stride: dim(r.stride, "table stride")?, index: dim(r.index, "table index")?, byte_off: dim(r.byte_off, "byte_off")? })
    }

    /// The per-row positions of a row-table call (i32 `[rows]`), bound.
    fn row_pos(&self, call: &OpCall, rows: usize) -> Result<Bind> {
        let p = call.pos.as_ref().ok_or_else(|| Error::invalid(format!("metal {}: a row-table call needs OpCall::pos", call.op)))?;
        if p.numel() != rows {
            return Err(Error::invalid(format!("metal {}: {} positions for {rows} rows", call.op, p.numel())));
        }
        let b = self.bind(p, DType::I32, false, "pos")?;
        if b.skip != 0 {
            return Err(Error::invalid("metal row table: the position vector must be 4-byte aligned"));
        }
        Ok(b)
    }

    pub(crate) fn launch_rowtab(&self, kernel: &str, call: &OpCall) -> Result<()> {
        let (i, o, a) = (&call.inputs, &call.outputs, &call.attrs);
        let bf = DType::Bf16;
        match (kernel, call.op) {
            (GDN_CONV_BF16_ROWS, Op::GdnConv) => {
                let (rows, c, k) = (i[0].shape[0], i[0].shape[1], i[1].shape[1]);
                let Some(&shader) = CONV_ROWS_SHADERS.get(k.wrapping_sub(2)) else {
                    return Err(Error::invalid(format!("metal {kernel}: {k} taps (2..=8 supported)")));
                };
                if rows * c == 0 {
                    return Ok(());
                }
                let tab = self.row_tab(call)?;
                let (x, w, y) =
                    (self.bind(&i[0], bf, false, "x")?, self.bind(&i[1], bf, false, "taps")?, self.bind(&o[0], bf, true, "y")?);
                dim(rows * c, "R*C")?;
                dim(c * k, "C*K")?;
                let p = OpP {
                    dims: [dim(rows, "R")?, dim(c, "C")?, dim(k, "K")?, tab.stride, tab.index, tab.byte_off, 0, 0],
                    skips: [x.skip, w.skip, y.skip, 0, 0],
                    ..Default::default()
                };
                self.dispatch_rows(
                    shader,
                    &[&x, &w, &y, &tab.b],
                    &p,
                    G3::Groups { x: c.div_ceil(crate::gdn::CONV_TG), y: 1, z: rows, threads: crate::gdn::CONV_TG },
                )
            }
            (GDN_DELTA_RULE_BF16_ROWS, Op::GdnDeltaRule) => {
                let d = intarsia_core::gdn::DeltaDims::from_call(call);
                let Some(&(_, shader)) = DELTA_ROWS_SHADERS.iter().find(|s| s.0 == d.dk) else {
                    return Err(Error::invalid(format!("metal {kernel}: dk = {} (32/64/128/256 supported)", d.dk)));
                };
                if d.t * d.hv * d.dv == 0 {
                    return Ok(());
                }
                if d.hk == 0 || d.hv % d.hk != 0 {
                    return Err(Error::invalid(format!("metal {kernel}: {} value heads, {} key heads", d.hv, d.hk)));
                }
                let tab = self.row_tab(call)?;
                let (qkv, ga, gb, al, db) = (
                    self.bind(&i[0], bf, false, "qkv")?,
                    self.bind(&i[1], bf, false, "a")?,
                    self.bind(&i[2], bf, false, "b")?,
                    self.bind(&i[3], bf, false, "A_log")?,
                    self.bind(&i[4], bf, false, "dt_bias")?,
                );
                let out = self.bind(&o[0], bf, true, "o")?;
                dim(d.t * d.conv_dim(), "R*C")?;
                let p = DeltaRowsP {
                    hk: dim(d.hk, "Hk")?,
                    hv: dim(d.hv, "Hv")?,
                    dv: dim(d.dv, "dv")?,
                    stride: tab.stride,
                    skips: [qkv.skip, ga.skip, gb.skip, al.skip, db.skip, out.skip],
                    index: tab.index,
                    byte_off: tab.byte_off,
                    eps: a.eps,
                    pad: 0,
                };
                self.dispatch_rows(
                    shader,
                    &[&qkv, &ga, &gb, &al, &db, &out, &tab.b],
                    &p,
                    G3::Groups {
                        x: d.dv.div_ceil(crate::gdn::DELTA_SG),
                        y: d.hv,
                        z: d.t,
                        threads: 32 * crate::gdn::DELTA_SG,
                    },
                )
            }
            (KV_APPEND_BF16_ROWS, Op::KvAppend) => {
                let (rows, nkv, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
                let cap = o[0].shape[1];
                let total = rows * nkv * d;
                if total == 0 {
                    return Ok(());
                }
                let tab = self.row_tab(call)?;
                let pos = self.row_pos(call, rows)?;
                let src = self.bind(&i[0], bf, false, "new")?;
                dim(total, "R*Nkv*D")?;
                dim(nkv * cap * d, "Nkv*cap*D")?;
                let p = OpP {
                    dims: [dim(rows, "R")?, dim(nkv, "Nkv")?, dim(d, "D")?, dim(cap, "cap")?, tab.stride, tab.index, 0, 0],
                    skips: [src.skip, 0, 0, 0, 0],
                    ..Default::default()
                };
                self.dispatch_rows(KV_ROWS, &[&src, &tab.b, &pos], &p, G3::Threads(total))
            }
            (QK_NORM_ROPE_KV_BF16_ROWS, Op::QkNormRopeKv) => {
                let (rows, n, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
                let (nkv, cap) = (i[1].shape[1], o[1].shape[1]);
                if rows * n * d == 0 {
                    return Ok(());
                }
                if d % 2 != 0 || d > 256 {
                    return Err(Error::invalid(format!("{kernel}: head_dim {d} must be even and at most 256")));
                }
                let tab = self.row_tab(call)?;
                let pos = self.row_pos(call, rows)?;
                let binds = [
                    self.bind(&i[0], bf, false, "q")?,
                    self.bind(&i[1], bf, false, "k")?,
                    self.bind(&i[2], bf, false, "v")?,
                    self.bind(&i[3], bf, false, "q_w")?,
                    self.bind(&i[4], bf, false, "k_w")?,
                    self.bind(&o[0], bf, true, "q_out")?,
                ];
                let rd = if a.rotary_dim == 0 { d } else { a.rotary_dim };
                let mut skips = [0u32; 6];
                for (s, b) in skips.iter_mut().zip(&binds) {
                    *s = b.skip;
                }
                let groups = rows * (n + 2 * nkv);
                dim(groups, "R*(N+2*Nkv)")?;
                let p = QkRowsP {
                    rows: dim(rows, "R")?,
                    n: dim(n, "N")?,
                    nkv: dim(nkv, "Nkv")?,
                    d: dim(d, "D")?,
                    rd: dim(rd, "rotary_dim")?,
                    cap: dim(cap, "cap")?,
                    stride: tab.stride,
                    index: tab.index,
                    skips,
                    eps: a.eps,
                    w_offset: a.w_offset,
                    theta: a.theta,
                    pad: 0,
                };
                let threads = d.next_multiple_of(32);
                let pso = self.pso(QK_ROWS).ok_or_else(|| Error::backend(format!("metal: no pipeline '{QK_ROWS}'")))?;
                if pso.maxTotalThreadsPerThreadgroup() < threads {
                    return Err(Error::backend(format!("{QK_ROWS}: {threads} threads exceed the pipeline limit")));
                }
                self.with_encoder_ind(|enc| {
                    enc.setComputePipelineState(pso);
                    for (idx, b) in binds.iter().enumerate() {
                        // SAFETY: the offset lies inside the buffer (TensorRef bounds were checked by `OpCall::check`).
                        unsafe { enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, idx) };
                    }
                    // SAFETY: as above.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&tab.b.mtl), tab.b.off, 6);
                        enc.setBuffer_offset_atIndex(Some(&pos.mtl), pos.off, 7);
                    }
                    set_bytes(enc, &p, 8);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize { width: groups, height: 1, depth: 1 },
                        MTLSize { width: threads, height: 1, depth: 1 },
                    );
                })
            }
            (ATTN_DECODE_BF16_SPLITKV_ROWS, Op::AttnDecode) => self.launch_attn_rowtab(call),
            _ => Err(Error::NoKernel(format!("metal kernel '{kernel}' for op '{}'", call.op))),
        }
    }

    /// `AttnDecode` over the rows of a batched step, each on its own cache (`attn_decode_rowtab_*`). Stage 1 runs
    /// `Nkv * HS x max_split x rows` threadgroups of the largest simdgroup count any row can take, stage 2 combines the
    /// partials; both are one launch. The split count and simdgroup count of every row are the solo host-position launch's,
    /// derived on the GPU from the row's position.
    fn launch_attn_rowtab(&self, call: &OpCall) -> Result<()> {
        let (i, o, a) = (&call.inputs, &call.outputs, &call.attrs);
        let bf = DType::Bf16;
        let (rows, n, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
        let (nkv, cap) = (i[1].shape[0], i[1].shape[1]);
        if rows * n * d == 0 {
            return Ok(());
        }
        if !(nkv > 0 && n % nkv == 0 && n / nkv <= MAX_GROUP && matches!(d, 8 | 16 | 32 | 64 | 128 | 256)) {
            return Err(Error::invalid(format!("metal {ATTN_DECODE_BF16_SPLITKV_ROWS}: {n} query / {nkv} kv heads of {d}")));
        }
        let qa = self.mbuf(&i[0].buf)?.base + i[0].offset;
        if qa % 16 != 0 {
            return Err(Error::invalid("metal row-table attention: q must be 16-byte aligned"));
        }
        let tab = self.row_tab(call)?;
        let pos = self.row_pos(call, rows)?;
        let q = self.bind(&i[0], bf, false, "q")?;
        let out = self.bind(&o[0], bf, true, "out")?;
        let grp = n / nkv;
        let hs = head_split(grp, d);
        let nkvv = nkv * hs;
        // a full cache bounds every row's split count (the shader derives each row's own from its position and idles the
        // surplus threadgroups); the policy is monotone in the visible keys and capped at MAX_SPLIT, as is the solo launch's
        let max_split = decode_splits(cap, nkvv, MAX_SPLIT);
        let need = rows * n * max_split * (d + 2) * 4;
        let scratch = call.scratch.as_ref().ok_or_else(|| {
            Error::invalid("metal attn_decode_bf16_splitkv_rows: the scratch workspace (OpCall::scratch) is missing")
        })?;
        if scratch.numel() * scratch.dtype.size_bytes() < need {
            return Err(Error::invalid(format!(
                "metal attn_decode_bf16_splitkv_rows: scratch of {} bytes, need {need}",
                scratch.numel() * scratch.dtype.size_bytes()
            )));
        }
        let sc = self.bind(scratch, DType::U8, true, "scratch")?;
        if sc.skip != 0 || (self.mbuf(&scratch.buf)?.base + scratch.offset) % 16 != 0 {
            return Err(Error::invalid("metal attn_decode_bf16_splitkv_rows: scratch must be 16-byte aligned"));
        }
        let (min_keys, target) = policy();
        let u = |v: usize, w: &str| dim(v, w);
        u(nkv * cap * d, "Nkv*cap*D")?;
        u(rows * n * max_split * (d + 2), "R*N*max_split*(D+2)")?;
        let p = OpP {
            dims: [u(target, "target")?, u(n, "N")?, u(nkv, "Nkv")?, u(d, "D")?, u(cap, "cap")?, 0, 0, u(max_split, "max_split")?],
            skips: [u(rows, "R")?, u(hs, "head split")?, tab.stride, out.skip, tab.index],
            f0: a.scale,
            f1: 0.0,
            pad: u(min_keys, "min_keys")?,
        };
        let gp = (grp / hs).next_power_of_two();
        let nsg = MAX_SG.min((TG_FLOATS / (gp * d)).max(1));
        let (name1, name2) = (ATTN_ROWTAB_SHADERS[gp.trailing_zeros() as usize], ATTN_ROWTAB_SHADERS[4]);
        let pso1 = self.pso(name1).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name1}'")))?;
        let pso2 = self.pso(name2).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name2}'")))?;
        self.with_encoder_ind(|enc| {
            // SAFETY: offsets lie inside the buffers (TensorRef bounds were checked by `OpCall::check`; the scratch size above).
            unsafe {
                enc.setComputePipelineState(pso1);
                for (idx, b) in [&q, &tab.b, &out, &pos, &sc].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, idx);
                }
            }
            set_bytes(enc, &p, 5);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: nkvv, height: max_split, depth: rows },
                MTLSize { width: nsg * 32, height: 1, depth: 1 },
            );
            // SAFETY: as above.
            unsafe {
                enc.setComputePipelineState(pso2);
                for (idx, b) in [&out, &pos, &sc].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, idx);
                }
            }
            set_bytes(enc, &p, 3);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n, height: rows, depth: 1 },
                MTLSize { width: d, height: 1, depth: 1 },
            );
        })
    }
}
