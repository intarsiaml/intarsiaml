//! Kernel table and device dispatch of the DeepSeek-V4.1 ops (lane `dsv41/cuda-attn`: hyper-connections, router, clamped
//! SwiGLU, activation fake-quantisation, rope tail, window ring, sparse attention, compressor pooling, indexer;
//! kernels/cuda/dsv41_*.cu, ABI kernels/cuda/ik_cuda_abi_dsv41.h). The contract is docs/design/dsv41-ops.md; each kernel is
//! admitted against the cpu reference (`intarsia-backend-cpu`'s `dsv41_ref`) by the ordinary sweep
//! (`intarsia kernels test --backend cuda --op <op>`). The table is data and compiles on every host; the dispatch
//! (`launch_dsv41`, called from `device.rs::launch`) needs CUDA.
//!
//! Every launcher takes bf16, f16 or f32 activations (the sweep runs bf16 and f32); the model runs bf16. `transpose01`,
//! `engram_gather` and `engram_merge` (kernels/cuda/dsv41_engram.cu) were added by lane `dsv41/tp2`; transpose01 takes any dtype.

use crate::kernels::ARCH;
use intarsia_core::{DType, Exactness, KernelImpl, Op, ShapeInfo, Tol};

pub const TABLE: &[(&str, Op)] = &[
    ("hc_expand", Op::HcExpand),
    ("hc_mixes", Op::HcMixes),
    ("hc_pre", Op::HcPre),
    ("hc_post", Op::HcPost),
    ("hc_sublayer", Op::HcSublayer),
    ("moe_route", Op::MoeRoute),
    ("silu_glu_clamp", Op::SiluGluClamp),
    ("act_qdq", Op::ActQdq),
    ("rope_tail", Op::RopeTail),
    ("window_kv", Op::WindowKv),
    ("attn_sparse", Op::AttnSparse),
    ("compress_pool", Op::CompressPool),
    ("indexer_blocks", Op::IndexerBlocks),
    ("indexer_topk", Op::IndexerTopk),
    ("transpose01", Op::Transpose01),
    ("engram_gather", Op::EngramGather),
    ("engram_merge", Op::EngramMerge),
    ("logits_f16w", Op::Logits),
    ("matmul_f16w", Op::Matmul),
];

/// bf16-rounded outputs: a one-ulp flip is allowed (the backend's shared budget).
const BF16_TOL: Tol = Tol::new(1.5e-2, 6e-3);
/// f32 outputs: accumulation-order noise of f32 math against the f64 reference.
const F32_TOL: Tol = Tol::new(1e-3, 2e-4);

const FLOATS: [DType; 3] = [DType::Bf16, DType::F16, DType::F32];

// ---- hc_mixes split policy (mirrors `hc_nsplit` in kernels/cuda/dsv41_hc.cu) ----
const HC_ROWS: usize = 4;
const HC_MIN_SPLIT_ELEMS: usize = 512;
const HC_TARGET_BLOCKS: usize = 192;

/// Splits of hc_mixes' partial-dot kernel for `rows` rows of `sd` flattened elements.
pub fn hc_nsplit(rows: usize, sd: usize) -> usize {
    let rb = rows.div_ceil(HC_ROWS).max(1);
    let mx = (sd / HC_MIN_SPLIT_ELEMS).max(1);
    HC_TARGET_BLOCKS.div_ceil(rb).clamp(1, mx)
}

/// hc_mixes scratch: f32 partials `[T, nsplit, (2+S)S + 1]` (ShapeInfo: m = T, n = S, k = D).
fn hc_mixes_scratch(s: &ShapeInfo) -> usize {
    let m = (2 + s.n) * s.n;
    s.m * hc_nsplit(s.m, s.n * s.k) * (m + 1) * 4
}

// ---- hc_sublayer split policy (mirrors `hc_sub_nsplit` in kernels/cuda/dsv41_hc_sub.cu) ----
const HC_SUB_TILE_ROWS: usize = 16;
const HC_SUB_CHUNK: usize = 128;
const HC_SUB_TARGET_BLOCKS: usize = 1536;

/// Splits of hc_sublayer's partial kernel for `rows` rows of `sd` flattened elements.
pub fn hc_sub_nsplit(rows: usize, sd: usize) -> usize {
    let tiles = rows.div_ceil(HC_SUB_TILE_ROWS).max(1);
    HC_SUB_TARGET_BLOCKS.div_ceil(tiles).clamp(1, sd.div_ceil(HC_SUB_CHUNK).max(1))
}

/// hc_sublayer scratch: f32 partials `[T, nsplit, (2+S)S + 1]` (ShapeInfo: m = T, n = S, k = D).
fn hc_sub_scratch(s: &ShapeInfo) -> usize {
    let m = (2 + s.n) * s.n;
    s.m * hc_sub_nsplit(s.m, s.n * s.k) * (m + 1) * 4
}

// ---- attn_sparse split policy (mirrors kernels/cuda/dsv41_attn.cu) ----
/// Most query rows that run split-KV (above: one block per query and head group, no scratch).
pub const ATTN_SPLIT_ROWS: usize = 16;
/// Most key splits per query.
pub const ATTN_MAX_SPLIT: usize = 32;

/// attn_sparse scratch: f32 `acc [T,H,nsplit,D]` + `(m, l) [T,H,nsplit]` for T <= 16 at the most splits, so it does not
/// depend on the window or the index count (ShapeInfo: m = T, n_heads = H, head_dim = D).
fn attn_scratch(s: &ShapeInfo) -> usize {
    if s.m > ATTN_SPLIT_ROWS {
        0
    } else {
        s.m * s.n_heads * ATTN_MAX_SPLIT * (s.head_dim + 2) * 4
    }
}

// ---- indexer tiling ----
/// Budget of the indexer's per-tile score buffer (f32 `[rows, Nk]`, rows = budget / (4 Nk), at least one query).
pub const INDEX_SCORE_BUDGET: usize = 128 << 20;

/// Indexer scratch: one tile of head-summed score rows. A decode step needs `4 Nk` bytes (512 KiB at Nk = 128K); a prefill
/// chunk is cut into tiles of `INDEX_SCORE_BUDGET / (4 Nk)` queries, so the buffer never exceeds the budget (or one row).
/// The launcher takes as many query rows per tile as the scratch it is given holds, so any buffer of at least `4 Nk` bytes
/// serves a call. `indexer_topk` with few long rows (T <= 16, Nk >= 16384) also gets the multi-block threshold select's
/// buffers (histogram, candidates: 41 KiB per row); without them it falls back to the one-block select (ShapeInfo: m = T,
/// kv_len = Nk).
pub fn index_scratch(s: &ShapeInfo) -> usize {
    let row = s.kv_len.max(1) * 4;
    let rows = (INDEX_SCORE_BUDGET / row).clamp(1, s.m.max(1));
    let mut bytes = rows * row;
    if s.m <= INDEX_FAST_ROWS && s.kv_len >= INDEX_FAST_MIN_N {
        bytes = bytes.div_ceil(256) * 256 + s.m * INDEX_FAST_EXTRA_PER_ROW + INDEX_FAST_SLACK;
    }
    bytes
}

// ---- the multi-block threshold select (mirrors kernels/cuda/dsv41_index.cu: kFastRows, kFastMinN, fast_extra) ----
const INDEX_FAST_ROWS: usize = 16;
const INDEX_FAST_MIN_N: usize = 16384;
/// Histogram (2048 u32) + meta (16 B) + candidates (4096 x 8 B) per row.
const INDEX_FAST_EXTRA_PER_ROW: usize = 2048 * 4 + 16 + 4096 * 8;
const INDEX_FAST_SLACK: usize = 1024;

fn pow2_le(v: usize, max: usize) -> bool {
    v.is_power_of_two() && v <= max
}

pub fn all() -> Vec<KernelImpl> {
    TABLE
        .iter()
        .map(|(name, op)| {
            let k = KernelImpl::new(*op, *name, "cuda").arch([ARCH]).acts(FLOATS).exact(Exactness::Deterministic);
            match *name {
                // copies: bitwise the reference
                "hc_expand" => k.exact(Exactness::Exact).tol(Tol::EXACT),
                "hc_mixes" => k.when(|s| s.n >= 1 && s.n <= 4).tol(F32_TOL).scratch(hc_mixes_scratch),
                "hc_sublayer" => k
                    .when(|s| (1..=4).contains(&s.n) && (s.n * s.k) % 4 == 0 && s.k <= 12288)
                    .tol(BF16_TOL)
                    .tol_f32(F32_TOL)
                    .scratch(hc_sub_scratch),
                "hc_pre" | "hc_post" => k.when(|s| s.n >= 1 && s.n <= 8).tol(BF16_TOL).tol_f32(F32_TOL),
                "moe_route" => k.acts([DType::F32]).when(|s| s.experts <= 1024 && (1..=32).contains(&s.top_k)).tol(F32_TOL),
                "silu_glu_clamp" => k.tol(BF16_TOL).tol_f32(F32_TOL),
                // quantise / dequantise: the reference's f32 arithmetic step for step (bitwise in practice)
                "act_qdq" => k.tol(BF16_TOL).tol_f32(F32_TOL),
                "rope_tail" => k.tol(BF16_TOL).tol_f32(F32_TOL),
                "window_kv" => k.exact(Exactness::Exact).tol(Tol::EXACT),
                "attn_sparse" => k
                    .when(|s| pow2_le(s.head_dim, 512) && s.head_dim >= 32)
                    .tol(BF16_TOL)
                    .tol_f32(F32_TOL)
                    .scratch(attn_scratch),
                "compress_pool" => k.acts([DType::F32]).tol(BF16_TOL).tol_f32(F32_TOL),
                // u8 mask / i32 indices: must match exactly (selection on f32 scores of exactly representable fp4 q / k)
                "indexer_blocks" | "indexer_topk" => k
                    .when(|s| (1..=32).contains(&s.n_heads) && s.head_dim % 8 == 0 && s.n_heads * s.head_dim <= 8192)
                    .tol(BF16_TOL)
                    .scratch(index_scratch),
                // byte movement by element size, any dtype (f32 / bf16 / f16 activations, i32 route tables): bitwise the reference
                "transpose01" => KernelImpl::new(*op, *name, "cuda").arch([ARCH]).exact(Exactness::Exact).tol(Tol::EXACT),
                // e4m3 payload times a power of two in f32 (the reference's arithmetic); y rounded to its dtype like the reference's
                "engram_gather" => k.acts([DType::U8]).exact(Exactness::Exact).tol(Tol::EXACT),
                "engram_merge" => k.tol(BF16_TOL).tol_f32(F32_TOL),
                // f16-weight dense linears (router, indexer weights_proj): warp per output, f32 accumulation
                "logits_f16w" | "matmul_f16w" => k.weights([DType::F16]).when(|s| s.m <= 65535).tol(BF16_TOL).tol_f32(F32_TOL),
                other => unreachable!("dsv41 kernel {other} has no registration"),
            }
        })
        .collect()
}

#[cfg(ik_cuda)]
mod launch {
    use crate::device::CudaBackend;
    use crate::ffi::{status_name, IkScratch, IkTensor};
    use intarsia_core::backend::StreamId;
    use intarsia_core::dsv41::ops::{QdqFmt, QdqScale, ScoreFunc};
    use intarsia_core::{Error, Op, OpCall, Result};
    use std::ffi::{c_int, c_void};

    type T = *const IkTensor;
    type TM = *mut IkTensor;
    type SC = *const IkScratch;
    type S = *mut c_void;

    extern "C" {
        fn ik_hc_expand(x: T, y: TM, s: S) -> c_int;
        fn ik_hc_mixes(x: T, f: T, scale: T, base: T, eps: f32, iters: i64, hc_eps: f32, pre: TM, post: TM, comb: TM, sc: SC, s: S) -> c_int;
        fn ik_hc_pre(x: T, mix: T, y: TM, s: S) -> c_int;
        fn ik_hc_post(x: T, res: T, post: T, comb: T, y: TM, s: S) -> c_int;
        fn ik_hc_sublayer_mode(
            mode: c_int,
            x: T,
            f: T,
            scale: T,
            base: T,
            pre_in: T,
            norm_w: T,
            post_x: T,
            post_mix: T,
            post_comb: T,
            eps: f32,
            w_offset: f32,
            iters: i64,
            hc_eps: f32,
            n: TM,
            pre: TM,
            post: TM,
            comb: TM,
            h_out: TM,
            sc: SC,
            s: S,
        ) -> c_int;
        fn ik_moe_route(lg: T, bias: T, top_k: i64, scale: f32, norm: i32, temp: f32, sf: i32, w: TM, ids: TM, s: S) -> c_int;
        fn ik_silu_glu_clamp(g: T, u: T, limit: f32, y: TM, s: S) -> c_int;
        fn ik_act_qdq(x: T, fmt: i32, block: i64, sfmt: i32, ratio: i64, y: TM, pos: T, s: S) -> c_int;
        fn ik_rope_tail(x: T, f: T, pos0: i64, stride: i64, ratio: i64, inverse: i32, y: TM, pos: T, s: S) -> c_int;
        fn ik_window_kv(kv: T, window: i64, pos0: i64, win: TM, ring: TM, pos: T, s: S) -> c_int;
        fn ik_attn_sparse(
            q: T,
            win: T,
            cmp: T,
            idx: T,
            sink: T,
            window: i64,
            pos0: i64,
            ratio: i64,
            scale: f32,
            o: TM,
            sc: SC,
            pos: T,
            s: S,
        ) -> c_int;
        fn ik_compress_pool(kv: T, score: T, ratio: i64, pos0: i64, out: TM, kv_ring: TM, score_ring: TM, pos: T, s: S) -> c_int;
        fn ik_indexer_blocks(
            q: T,
            k: T,
            w: T,
            keep: i64,
            block: i64,
            ratio: i64,
            pos0: i64,
            scale: f32,
            mask: TM,
            sc: SC,
            pos: T,
            s: S,
        ) -> c_int;
        fn ik_transpose01(x: T, y: TM, s: S) -> c_int;
        fn ik_logits_f16w(x: T, w: T, y: TM, s: S) -> c_int;
        fn ik_matmul_f16w(x: T, w: T, y: TM, s: S) -> c_int;
        fn ik_engram_gather(rec: T, y: TM, s: S) -> c_int;
        fn ik_engram_merge(h: T, kv: T, q_w: T, k_w: T, eps: f32, out: TM, s: S) -> c_int;
        fn ik_indexer_topk(
            q: T,
            k: T,
            w: T,
            cand: T,
            top_k: i64,
            block: i64,
            ratio: i64,
            pos0: i64,
            scale: f32,
            idx: TM,
            sc: SC,
            pos: T,
            s: S,
        ) -> c_int;
    }

    impl CudaBackend {
        /// `Some(result)` when `kernel` is a DeepSeek-V4.1 launcher of this lane, `None` otherwise.
        pub(crate) fn launch_dsv41(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Option<Result<()>> {
            let (_, op) = super::TABLE.iter().find(|(n, _)| *n == kernel)?;
            Some(self.launch_dsv41_inner(kernel, *op, call, stream))
        }

        fn launch_dsv41_inner(&self, kernel: &str, op: Op, call: &OpCall, stream: StreamId) -> Result<()> {
            if op != call.op {
                return Err(Error::invalid(format!("kernel '{kernel}' implements {op}, not {}", call.op)));
            }
            self.bind()?;
            let s = self.stream(stream)?;
            let inp: Vec<IkTensor> = call.inputs.iter().map(Self::tensor).collect::<Result<_>>()?;
            let mut out: Vec<IkTensor> = call.outputs.iter().map(Self::tensor).collect::<Result<_>>()?;
            let scratch = match &call.scratch {
                Some(t) => {
                    let ik = Self::tensor(t)?;
                    Some(IkScratch { data: ik.data, bytes: t.nbytes() as i64 })
                }
                None => None,
            };
            let sc = scratch.as_ref().map_or(std::ptr::null(), |p| p as *const IkScratch);
            let pos_t: Option<IkTensor> = call.pos.as_ref().map(Self::tensor).transpose()?;
            let pos = pos_t.as_ref().map_or(std::ptr::null(), |t| t as *const IkTensor);
            let a = &call.attrs;
            let d = &a.dsv;
            let i = |n: usize| &inp[n] as *const IkTensor;
            let o = out.as_mut_ptr();
            let on = |n: usize| o.wrapping_add(n);
            let fmt = match d.qdq_fmt {
                QdqFmt::Fp8E4m3 => 0,
                QdqFmt::Fp4E2m1 => 1,
            };
            let sfmt = match d.qdq_scale {
                QdqScale::Ue8m0 => 0,
                QdqScale::E4m3 => 1,
                QdqScale::F32 => 2,
            };
            let sf = match d.score_func {
                ScoreFunc::Softmax => 0,
                ScoreFunc::Sigmoid => 1,
                ScoreFunc::SqrtSoftplus => 2,
            };
            let pos0 = a.pos0 as i64;
            // device position: the compressed row ratio of the positioned ops (rope_tail's group start, act_qdq's row offset,
            // attn_sparse's visible compressed rows); a host-position launch ignores it
            let ratio = d.ratio.max(1) as i64;
            // SAFETY: tensors are bounds-checked views of live device buffers (held by `call`); the launchers validate
            // dtype / rank / shape / scratch and return an error status instead of faulting, and never synchronise or allocate.
            let rc = unsafe {
                match kernel {
                    "hc_expand" => ik_hc_expand(i(0), on(0), s),
                    "hc_mixes" => ik_hc_mixes(i(0), i(1), i(2), i(3), a.eps, d.iters as i64, d.hc_eps, on(0), on(1), on(2), sc, s),
                    "hc_pre" => ik_hc_pre(i(0), i(1), on(0), s),
                    "hc_post" => ik_hc_post(i(0), i(1), i(2), i(3), on(0), s),
                    "hc_sublayer" => ik_hc_sublayer_mode(
                        d.hc_mode as c_int,
                        i(0),
                        i(1),
                        i(2),
                        i(3),
                        i(4),
                        i(5),
                        i(6),
                        i(7),
                        i(8),
                        a.eps,
                        a.w_offset,
                        d.iters as i64,
                        d.hc_eps,
                        on(0),
                        on(1),
                        on(2),
                        on(3),
                        on(4),
                        sc,
                        s,
                    ),
                    "moe_route" => {
                        ik_moe_route(i(0), i(1), a.top_k as i64, a.scale, a.norm_topk as i32, a.temperature, sf, on(0), on(1), s)
                    }
                    "silu_glu_clamp" => ik_silu_glu_clamp(i(0), i(1), d.limit, on(0), s),
                    "act_qdq" => ik_act_qdq(i(0), fmt, d.qdq_block as i64, sfmt, ratio, on(0), pos, s),
                    "rope_tail" => ik_rope_tail(i(0), i(1), pos0, d.pos_stride as i64, ratio, d.inverse as i32, on(0), pos, s),
                    "window_kv" => ik_window_kv(i(0), a.window as i64, pos0, on(0), on(1), pos, s),
                    "attn_sparse" => {
                        ik_attn_sparse(i(0), i(1), i(2), i(3), i(4), a.window as i64, pos0, ratio, a.scale, on(0), sc, pos, s)
                    }
                    "compress_pool" => ik_compress_pool(i(0), i(1), d.ratio as i64, pos0, on(0), on(1), on(2), pos, s),
                    "indexer_blocks" => ik_indexer_blocks(
                        i(0),
                        i(1),
                        i(2),
                        a.top_k as i64,
                        d.block as i64,
                        d.ratio as i64,
                        pos0,
                        a.scale,
                        on(0),
                        sc,
                        pos,
                        s,
                    ),
                    "indexer_topk" => ik_indexer_topk(
                        i(0),
                        i(1),
                        i(2),
                        i(3),
                        a.top_k as i64,
                        d.block as i64,
                        d.ratio as i64,
                        pos0,
                        a.scale,
                        on(0),
                        sc,
                        pos,
                        s,
                    ),
                    "transpose01" => ik_transpose01(i(0), on(0), s),
                    "logits_f16w" => ik_logits_f16w(i(0), i(1), on(0), s),
                    "matmul_f16w" => ik_matmul_f16w(i(0), i(1), on(0), s),
                    "engram_gather" => ik_engram_gather(i(0), on(0), s),
                    "engram_merge" => ik_engram_merge(i(0), i(1), i(2), i(3), a.eps, on(0), s),
                    other => return Err(Error::NoKernel(format!("cuda kernel '{other}'"))),
                }
            };
            if rc != 0 {
                return Err(Error::backend(format!(
                    "ik_{kernel} returned {} ({rc}); inputs {:?} shapes {:?}",
                    status_name(rc),
                    call.inputs.iter().map(|t| t.dtype).collect::<Vec<_>>(),
                    call.inputs.iter().map(|t| t.shape.clone()).collect::<Vec<_>>()
                )));
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_scratch_holds_a_row_and_stays_in_budget() {
        let s = |m, nk| ShapeInfo { m, kv_len: nk, ..ShapeInfo::default() };
        assert_eq!(index_scratch(&s(1, 4096)), 4096 * 4);
        assert_eq!(index_scratch(&s(1, 131072)), 131072 * 4 + INDEX_FAST_EXTRA_PER_ROW + INDEX_FAST_SLACK);
        assert_eq!(index_scratch(&s(4096, 131072)), INDEX_SCORE_BUDGET);
        assert_eq!(index_scratch(&s(4096, 1 << 26)), (1 << 26) * 4, "one row at least");
        for nk in (1..300_000).step_by(997) {
            let b = index_scratch(&s(4096, nk));
            assert!(b >= nk * 4 && b <= INDEX_SCORE_BUDGET.max(nk * 4), "nk {nk}");
        }
    }

    #[test]
    fn hc_sub_nsplit_policy() {
        assert_eq!(hc_sub_nsplit(1, 4 * 5120), 160);
        assert_eq!(hc_sub_nsplit(4096, 4 * 5120), 6);
        assert_eq!(hc_sub_nsplit(17, 256), 2);
    }

    #[test]
    fn hc_nsplit_policy() {
        assert_eq!(hc_nsplit(1, 4 * 5120), 40);
        assert_eq!(hc_nsplit(4096, 4 * 5120), 1);
        assert_eq!(hc_nsplit(1, 256), 1);
    }
}
