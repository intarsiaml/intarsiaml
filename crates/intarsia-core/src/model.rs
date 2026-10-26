//! The generic decoder (ARCHITECTURE 3.5 / 3.6): walks the recipe's layer
//! pattern and calls ops through the backend. Nothing here is model-family
//! specific: shapes, tensor names, norms, qk-norm, tied head and rope theta all
//! come from the recipe + `config.json`.
//!
//! Dataflow per block (pre-norm, residual stream `h`):
//! ```text
//!   n = norm(h)                      first block: RmsNorm
//!   n, h = rmsnorm_add(prev_out, h)  later blocks: fused residual add + norm
//!   prev_out = block(n)
//! ```
//! and one final `rmsnorm_add` with the model's final norm before the head.

use crate::backend::{Backend, Buffer, GraphHandle, MemKind, TensorRef};
use crate::comm::Comm;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::memplan::MemoryPlan;
use crate::ops::{Op, OpAttrs, OpCall, ShapeInfo};
use crate::quant::moe::{MoeScratch, MoeWeights, NormFold, OpRunner};
use crate::quant::{Fp8Act, QTensor, QuantLoadOpts};
use crate::recipe::{LinearDims, ModelDims, Recipe};
use crate::registry::Registry;
use crate::state::{PeerState, SequenceState, SlotSpec, StateSlot};
use crate::telemetry::Telemetry;
use crate::weights::{load_tensor, load_tensor_shard, Shard, ShardAxis, WeightSource};
use std::cell::Cell;
use std::sync::{Arc, Mutex};
use std::time::Instant;

mod ar_norm;
mod batch;
mod coll_trace;
mod gdn_batch;
mod gdn_block;
mod gemma;
pub(crate) mod feedback;
mod graph_step;
mod spec_graph;
pub use spec_graph::SpecPass;
pub(crate) mod dcoco;
pub(crate) mod dflash;
pub(crate) mod mtp;
pub use dflash::{DflashConfig, DflashLoad};
pub use mtp::DraftScript;
pub use batch::TpBatchMutant;
use gdn_block::{ExtraWs, GdnBlock};
use gemma::{GemmaLayer, GemmaWs};
use graph_step::GraphState;

/// Macro prefill passes run by any model of the process (diagnostics and tests; [`Model::macro_passes`] is the per-model count).
pub static MACRO_PASSES_TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The ops a macro pass runs once per chunk ([`Model::macro_split`]): `INTARSIA_PREFILL_MACRO_SPLIT=<op,op,..>` (op names as
/// `Op::parse`; `none` = nothing) replaces the default, the vendor-GEMM matmul family. A diagnostic knob for bisecting a row-variant op.
fn macro_split_ops() -> Result<Vec<Op>> {
    match std::env::var("INTARSIA_PREFILL_MACRO_SPLIT") {
        Err(_) => Ok(vec![Op::Matmul, Op::MatmulFp8, Op::MatmulFp8Pair, Op::MatmulPair]),
        Ok(v) if v.trim() == "none" => Ok(vec![]),
        Ok(v) => v
            .split(',')
            .map(|n| Op::parse(n.trim()).ok_or_else(|| Error::invalid(format!("INTARSIA_PREFILL_MACRO_SPLIT: unknown op '{n}'"))))
            .collect(),
    }
}

/// Rows of logits computed/downloaded at a time when all positions are requested.
const LOGIT_ROWS: usize = 16;

#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// Upper bound on context length (tokens); clamped by measured free memory.
    pub max_seq: usize,
    /// Fraction of measured free memory the KV cache may use.
    pub kv_fraction: f32,
    /// Tokens per forward pass (prefill chunk / activation workspace rows).
    pub prefill_chunk: usize,
    pub act_dtype: Option<DType>,
    pub kv_dtype: Option<DType>,
    /// Time every Nth step per op (synchronises the device); 1 = every step; 0 = never.
    pub profile_every: u32,
    /// Bytes this process may use in total (weights + workspace + KV); the KV cache is
    /// clamped to what is left, so a fleet lease sizes the KV, not the box's free memory.
    pub mem_budget: Option<u64>,
    /// FP8 projections: activations quantised with the checkpoint's static scale (`static`, W8A8) or kept
    /// 16-bit (`none`, W8A16). `None` = the recipe's `[defaults] fp8_act`.
    pub fp8_act: Option<Fp8Act>,
    /// MXFP4 projections: dynamic MXFP4 activations (`dynamic`, W4A4) or 16-bit (`none`, W4A16). `None` = what the
    /// checkpoint declares (Quark `input_tensors`), W4A16 when it declares nothing.
    pub mxfp4_act: Option<crate::quant::Mxfp4Act>,
    /// Replay the single-token decode step as a captured graph when the backend supports it (`--no-graphs` turns
    /// this off; profiled steps always run eagerly). Same kernels, same results as the eager path.
    pub graphs: bool,
    /// Tensor parallelism: run the row-local fused decode ops on each rank's shard when the backend can
    /// (`Backend::fusion_under_tp`). The CLI turns it on by default for CUDA and for HIP at tp > 1 (`INTARSIA_TP_FUSION=0` opts
    /// out, `=1` asks for it anywhere); the library default is off. Measured bitwise to unfused on the 35B tp=2 gates
    /// (docs/results/2026-10-21-tp2-cuda-graphs.md, docs/results/2026-10-21-h1-tp2-hip.md).
    pub tp_fusion: bool,
    /// Tensor parallelism: split the lm_head over the ranks by vocabulary rows (rank r computes the logits of rows
    /// `[r V/w, (r+1) V/w)` and one all-gather puts the halves into rank 0's `[rows, V]` logits, bitwise those of the
    /// unsplit head: every logit is one row's dot product). Needs a communicator that implements `all_gather`
    /// (`INTARSIA_TP_VOCAB=0` opts out, `=1` asks for it anywhere). The CLI turns it on by default for HIP at tp=2; the
    /// library default is off, and a head that cannot be split (vocabulary not a multiple of the communicator's gather quantum x world, a format
    /// without a row-sharded loader) keeps the replicated head on rank 0.
    pub tp_vocab: bool,
    /// Tensor parallelism: fuse each block-end all-reduce with the standalone residual add + rmsnorm that follows it
    /// (`Comm::all_reduce_add_rmsnorm`), when the communicator can and `tp_fusion` is on at world 2. Bit for bit the unfused
    /// pair (`model/ar_norm.rs`). The CLI sets it from `INTARSIA_TP_ARNORM` (default off); the library default is off.
    pub tp_arnorm: bool,
    /// Load the checkpoint's multi-token-prediction head (recipe `[mtp]`) as the `--spec mtp` drafter. Off: nothing under
    /// the head's prefix is read and the model is exactly the one without it.
    pub mtp: bool,
    /// Most drafts per verify pass the MTP drafter is sized for (`--spec-k`): the per-row recurrent state buffers of a
    /// verify pass are reserved in the memory plan and allocated with each sequence for this many rows. 0: the
    /// largest k ([`crate::spec::SPEC_K_MAX`]).
    pub mtp_k: usize,
    /// `--mtp-draft-vocab N`: the MTP drafter's argmax runs over the first N rows of the lm_head (token ids 0..N, the
    /// frequent end of a BPE vocabulary) instead of all of them, so a draft costs N/V of a head sweep. The verify pass keeps
    /// the full head, so output stays the spec-off output and only acceptance can change. 0: the full head.
    pub mtp_draft_vocab: Option<usize>,
    /// Load a DFlash draft checkpoint (`--spec dflash --spec-draft DIR`, `model/dflash.rs`). None: nothing is read, no op is
    /// added and the model is exactly the one without it.
    pub dflash: Option<DflashLoad>,
    /// Macro prefill pass size in rows (see [`Model::macro_rows`]): `None` = `INTARSIA_PREFILL_MACRO` (unset = off); `Some(0)` off.
    pub prefill_macro: Option<usize>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            max_seq: 4096,
            kv_fraction: 0.5,
            prefill_chunk: 256,
            act_dtype: None,
            kv_dtype: None,
            profile_every: 1,
            mem_budget: None,
            fp8_act: None,
            mxfp4_act: None,
            graphs: true,
            tp_fusion: false,
            tp_vocab: false,
            tp_arnorm: false,
            mtp: false,
            mtp_k: 0,
            mtp_draft_vocab: None,
            dflash: None,
            prefill_macro: None,
        }
    }
}

/// The finished frequency table of a scaled rotary embedding on the device ([`crate::rope::RopeScaling::freqs`]).
#[derive(Clone)]
struct RopeTable {
    /// f32 `[rotary_dim / 2]`.
    inv_freq: TensorRef,
    /// Multiplier on cos / sin (YaRN's attention factor).
    mscale: f32,
}

struct AttnBlock {
    pre_norm: TensorRef,
    /// `[n_heads * head_dim * (1 + gated), hidden]`: with `gated` each head's rows are `q` then its output gate.
    wq: QTensor,
    wk: QTensor,
    wv: QTensor,
    wo: QTensor,
    q_norm: Option<TensorRef>,
    k_norm: Option<TensorRef>,
    /// Projection biases (`[n_heads * head_dim]`, `[n_kv_heads * head_dim]` x2), added right after the projection.
    q_bias: Option<TensorRef>,
    k_bias: Option<TensorRef>,
    v_bias: Option<TensorRef>,
    slot: usize,
    /// `q` carries a per-head output gate (`attn_out * sigmoid(gate)` before `wo`).
    gated: bool,
    /// Leading values of each head that rotate (`partial_rotary_factor * head_dim`); 0 = the whole head.
    rotary_dim: usize,
}

/// Dense SwiGLU FFN. The projections are in any supported weight format (bf16, or a quantised checkpoint such as the
/// MXFP4 Qwen3.8-27B); a tensor-parallel rank holds bf16 slices (the quantised formats refuse to shard).
struct FfnBlock {
    pre_norm: TensorRef,
    gate: QTensor,
    up: QTensor,
    down: QTensor,
}

/// Sparse MoE FFN (router + routed experts + gated shared expert); the activation scratch is shared in the workspace.
struct MoeBlock {
    pre_norm: TensorRef,
    w: MoeWeights,
}

enum Block {
    Attn(AttnBlock),
    Gdn(GdnBlock),
    Ffn(FfnBlock),
    Moe(MoeBlock),
    /// A whole Gemma 4 decoder layer (`model/gemma.rs`): attention, dense MLP and MoE with sandwich norms.
    Gemma(GemmaLayer),
}

/// Bytes of `Workspace::pos`: [`POS_TAB_ENTRIES`] i32 positions.
const POS_TAB_ENTRIES: usize = 96;
const POS_TAB_BYTES: usize = POS_TAB_ENTRIES * 4;
/// First entry of the verify rows' positions (`P + r`), the MTP head's catch-up rows (`P - n + i`) and its chained rows
/// (`P + i`) in `Workspace::pos`. Each region holds up to 16 consecutive entries.
const POS_VERIFY: usize = 0;
const POS_CATCH_UP: usize = 16;
const POS_CHAIN: usize = 32;

struct Workspace {
    max_tokens: usize,
    ids: Buffer,
    h: [Buffer; 2],
    n: Buffer,
    blk: Buffer,
    q: [Buffer; 2],
    k: [Buffer; 2],
    v: Buffer,
    g: Buffer,
    u: Buffer,
    a: Buffer,
    logits: Buffer,
    tok: Buffer,
    /// Tensor parallelism: f32 `[max_tokens, hidden]` partial results of the row-parallel projections (o, down),
    /// reduced into the bf16 activation buffer with a single rounding. None: bf16 partials reduced in place.
    part: Option<Buffer>,
    /// `i32` position table (rewritten before every replay of a captured step). Entry 0 is the position of the token a
    /// captured decode step processes; a captured speculative step ([`Model::spec_step`]) uses the regions
    /// [`POS_VERIFY`], [`POS_CATCH_UP`] and [`POS_CHAIN`], each a run of consecutive positions.
    pos: Buffer,
    /// Buffers of the gated-attention and GatedDeltaNet blocks (empty when the model has none).
    x: ExtraWs,
    /// Activations of the MoE blocks (one set, reused by every layer).
    moe: Option<MoeScratch>,
}

/// Runs the MoE block's ops through `Model::exec` (profiling, plan recording, kernel scratch).
struct ExecRunner<'a> {
    m: &'a Model,
    profile: bool,
}

impl OpRunner for ExecRunner<'_> {
    fn run(&self, op: Op, inputs: Vec<TensorRef>, outputs: Vec<TensorRef>, attrs: OpAttrs) -> Result<()> {
        self.m.exec(op, inputs, outputs, attrs, self.profile)
    }
    fn run_fused(&self, op: Op, inputs: Vec<TensorRef>, outputs: Vec<TensorRef>, attrs: OpAttrs) -> Result<bool> {
        self.m.try_fused(op, inputs, outputs, attrs, self.profile)
    }
    /// Only in a batched pass (batching on and the pass row-exact): the callers use it to take a fused op the solo step would not take at
    /// this row count, which must not change anything with batching off (`plan show` and the solo / prefill / verify paths stay as on main).
    fn fused_available(&self, op: Op, inputs: &[TensorRef], outputs: &[TensorRef], attrs: &OpAttrs) -> bool {
        self.m.batch_rows >= 2
            && self.m.verifying()
            && self.m.fused_ok(&OpCall::new(op, inputs.to_vec(), outputs.to_vec(), attrs.clone()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadMode {
    /// Fill the KV cache only.
    None,
    /// Logits of the last position.
    LastLogits,
    /// Greedy token of the last position (argmax on device, 4 bytes back).
    LastArgmax,
    /// Logits of every position, streamed through a callback.
    AllLogits,
    /// Batched decode rows only (`Model::forward_batch`, device-side sampling, R3b): the `k` largest logits of the last position
    /// and their token ids, in the sampler's order (`StepOut::TopK`), instead of the whole `[V]` row. Exact for a sampler with
    /// `top_k == k`: `Sampler::sample_topk` draws the token host sampling would.
    LastTopK(usize),
}

#[derive(Debug)]
pub enum StepOut {
    None,
    Logits(Vec<f32>),
    Token(u32),
    /// The `k` largest logits of the row and their token ids, best first (`HeadMode::LastTopK`).
    TopK { ids: Vec<u32>, vals: Vec<f32> },
}

pub type RowCallback<'a> = &'a mut dyn FnMut(usize, &[f32]) -> Result<()>;

/// Views of the attention scratch for some rows: the two ping-pong q buffers `[t, n_heads, head_dim]`, the two k
/// buffers and the v buffer `[t, n_kv_heads, head_dim]` (see [`Model::attn_rows`]).
pub(super) struct TopkBufs {
    ids: Buffer,
    vals: Buffer,
    nan: Buffer,
    scratch: Buffer,
}

struct AttnBufs {
    q: [TensorRef; 2],
    k: [TensorRef; 2],
    v: TensorRef,
}

/// What a layer pass runs on: one sequence's next `t` tokens, or one decode token of each of several sequences.
#[derive(Clone, Copy)]
enum Seqs<'a> {
    One(&'a SequenceState),
    Batch(&'a [&'a SequenceState]),
    /// Several prompts' next chunks in one pass (`Model::forward_packed`): the row-wise ops run once over all rows, attention and
    /// the GatedDeltaNet recurrence once per segment on that segment's own state.
    Packed(&'a [Seg<'a>]),
}

/// One prompt of a packed prefill pass: its state and its rows `[lo, lo + len)` of the packed activations, which sit at
/// positions `pos..pos + len` of the sequence (`st.len` for a pass of different prompts; later for the second and further
/// chunks of ONE prompt in a macro pass, [`Model::forward_macro`], whose segments share a state and run in order).
#[derive(Clone, Copy)]
pub(crate) struct Seg<'a> {
    pub(crate) st: &'a SequenceState,
    pub(crate) lo: usize,
    pub(crate) len: usize,
    pub(crate) pos: usize,
}

impl AttnBufs {
    /// Rows `[lo, lo + n)` of every buffer (a packed pass's segment).
    fn rows(&self, lo: usize, n: usize) -> AttnBufs {
        AttnBufs {
            q: [self.q[0].rows(lo, n), self.q[1].rows(lo, n)],
            k: [self.k[0].rows(lo, n), self.k[1].rows(lo, n)],
            v: self.v.rows(lo, n),
        }
    }
    /// Row `r` of every buffer, as a one-row set.
    fn row(&self, r: usize) -> AttnBufs {
        AttnBufs {
            q: [self.q[0].rows(r, 1), self.q[1].rows(r, 1)],
            k: [self.k[0].rows(r, 1), self.k[1].rows(r, 1)],
            v: self.v.rows(r, 1),
        }
    }
}

pub struct Model {
    pub backend: Arc<dyn Backend>,
    pub registry: Registry,
    /// The model's dimensions (the checkpoint's, not one rank's).
    pub dims: ModelDims,
    /// What this rank computes with: heads, KV heads and the FFN width divided by the tensor-parallel degree
    /// (equal to `dims` without tensor parallelism). Plan shapes and the workspace follow these.
    pub ldims: ModelDims,
    pub recipe: Recipe,
    pub act: DType,
    pub kv: DType,
    pub plan: MemoryPlan,
    pub weights_bytes: u64,
    embed: TensorRef,
    final_norm: TensorRef,
    /// Replicated head: only rank 0 holds it (and runs the final norm's consumer), the other ranks stop after the last layer.
    /// Vocab-parallel head (`vocab_par`): every rank holds its `V / world` rows of it.
    lm_head: Option<QTensor>,
    /// The lm_head is split over the ranks by vocabulary rows (`LoadOptions::tp_vocab`): every rank runs its half and the
    /// halves are all-gathered into rank 0's `ws.logits`.
    vocab_par: bool,
    /// The first `LoadOptions::mtp_draft_vocab` rows of `lm_head` (the MTP drafter's cheaper head), or None.
    draft_head: Option<QTensor>,
    layers: Arc<Vec<Vec<Block>>>,
    /// Scratch of the Gemma 4 layers (`None` for every other model).
    gemma: Option<GemmaWs>,
    slot_specs: Vec<SlotSpec>,
    ws: Workspace,
    telemetry: Arc<Telemetry>,
    profile_every: u32,
    /// Kernel scratch workspace shared by all ops (sized by plan resolution; ops run in stream order).
    scratch: Option<Buffer>,
    /// Installed by [`Model::dry_run`]: `exec` records `(op, shape, kernel)` instead of launching.
    recorder: Mutex<Option<crate::plan::Recorder>>,
    /// Result of plan resolution (kernel of every op/shape the model needs).
    pub resolved: crate::plan::ResolvedPlan,
    /// Captured decode step (CUDA graph); see `model/graph_step.rs`.
    graphs: GraphState,
    /// While a graph is being captured: the ops it records (replays count them in telemetry).
    capture_ops: Mutex<Option<Vec<Op>>>,
    /// While a graph is being captured: the collectives it records.
    capture_comms: std::sync::atomic::AtomicU64,
    /// Scaled rotary embedding (llama3 / yarn): the frequency table every attention block's rope reads. `None` = the
    /// plain `Rope` op, which computes its own frequencies from `theta`.
    rope_table: Option<RopeTable>,
    /// Row-local fusion under tensor parallelism is on (`LoadOptions::tp_fusion`, world > 1, backend capable).
    tp_fusion: bool,
    /// Block-end all-reduces are deferred to the norm that follows (`model/ar_norm.rs`).
    tp_arnorm: bool,
    /// The all-reduce deferred by `tp_arnorm`, until the next op decides how it runs.
    ar_defer: Mutex<Option<ar_norm::DeferredAr>>,
    /// Fused all-reduce + rmsnorm collectives run eagerly / recorded into a graph (printed when the model is dropped).
    ar_fused_eager: std::sync::atomic::AtomicU64,
    ar_fused_captured: std::sync::atomic::AtomicU64,
    /// Tensor parallelism: this rank's place in its group (rank 0 / world 1 without it).
    tp: TpRank,
    /// `INTARSIA_TP_NO_COMM=1`: skip the collectives, to time a decode step without them (the output is garbage).
    no_comm: bool,
    /// `INTARSIA_TP_TEST_BUG=scale:F[@RANK]` (test only, accuracy-gate sensitivity): before every all-reduce, rank `RANK`
    /// (default 1) multiplies its partial by `F`. The gate must fail on it (docs/accuracy.md, "TP self-golden budget").
    test_bug: Option<(usize, f32)>,
    /// Ranks 1.. of the group, each a full shard with its own backend, weights, workspace and KV (rank 0 only).
    peers: Vec<Model>,
    /// Tensor parallelism across processes (rank 0 only): the other rank runs in another process, driven over a
    /// connection (`crate::tp_remote`, docs/design/35b-tp2.md). Exclusive with `peers`.
    remote: Option<Arc<crate::tp_remote::RemotePeer>>,
    /// Set for the duration of a speculative verify pass ([`Model::forward_verify`]): every op call is marked
    /// `OpCall::exact_rows`, so the registry gives it a row-exact kernel (or fails). Never set otherwise.
    exact_rows: std::sync::atomic::AtomicBool,
    /// Inside an MTP head pass over several rows ([`Model::mtp_layer_at`]): attention runs one decode call per row on a backend
    /// that asks for it ([`Backend::verify_attn_per_row`]), so the eager head's rows are the captured step's bit for bit.
    head_rows_attn: std::sync::atomic::AtomicBool,
    /// Inside the DFlash draft block ([`Model::dflash_draft`]): attention is not causal, every row reads all keys up to the
    /// block's last position (one decode call per row at that position).
    noncausal_rows: std::sync::atomic::AtomicBool,
    /// Verify passes run their attention one decode call per row on every backend (exact by construction), as HIP does by
    /// `Backend::verify_attn_per_row`: set with the DFlash drafter, whose 16-row passes the multi-row call is not bitwise for
    /// on CUDA (docs/results/2026-10-25-dc-dflash-dcoco.md section 10.1).
    verify_rows_attn: bool,
    /// `i32 [max rows]` argmax output of a verify pass; allocated by [`Model::enable_verify`] (spec on only).
    verify_tok: Option<Buffer>,
    /// Most rows of a verify pass; 0 = speculative verify off (the default: plan and kernels as without it).
    verify_rows: usize,
    /// Most sequences of a batched decode step ([`Model::forward_batch`]); 0 = batching off (the default).
    batch_rows: usize,
    /// Opt-in (`Model::allow_recurrent_batch`, cpu backend only): batched decode of hybrid models, i.e. GatedDeltaNet
    /// (recurrent) blocks, mixture-of-experts blocks and gated attention. Off by default: `enable_batch` then accepts
    /// dense attention-only models exactly as before.
    recurrent_batch: bool,
    /// `i32 [BATCH_MAX_ROWS]` argmax output of a batched step; allocated by [`Model::enable_batch`].
    batch_tok: Option<Buffer>,
    /// Row tables of batched decode (R3a, `OpCall::rows`): `u8 [BATCH_MAX_ROWS, 2 * slots, 8]` device addresses of every row's
    /// state buffers (entries `2j`, `2j + 1` of row `r`: recurrent buffer of slot `j`, or its K and V cache), and `i32
    /// [BATCH_MAX_ROWS]` positions. Rewritten by the host before each batched step, so one recording of the step (a graph)
    /// serves any set of sequences. `Some` only on a backend that supports them and unless `INTARSIA_ROW_TABLES=0`.
    /// The backend stream this model's work is enqueued on: 0, or 1 on a decode twin ([`Model::decode_twin`]).
    stream: crate::backend::StreamId,
    row_tab: Option<Buffer>,
    batch_pos: Option<Buffer>,
    /// Device-side top-k sampling buffers (R3b): ids `i32 [BATCH_MAX_ROWS, 64]`, values `f32 [..]`, NaN flags `i32 [BATCH_MAX_ROWS]`
    /// and the kernel's scratch. `Some` when the backend has `topk_rows` and `INTARSIA_DEVICE_SAMPLING` is `1` (opt-in, default off).
    topk: Option<TopkBufs>,
    /// Batched decode with a vocab-parallel head (`vocab_par`, every rank): the gather tile `f32 [world, BATCH_MAX_ROWS * V /
    /// world]`, rank-major as the all-gather leaves it; rank 0 then copies it into `ws.logits` as row-major `[b, V]`
    /// ([`Model::batch_head`]). Allocated by [`Model::enable_batch`].
    batch_tile: Option<Buffer>,
    /// Test hook (`Model::set_tp_batch_mutant`, tests only): a deliberate protocol bug a tensor-parallel peer applies to its
    /// share of a batched step, which the batched-vs-solo tests must catch. 0 = none (always, outside those tests).
    tp_batch_mutant: std::sync::atomic::AtomicU8,
    /// `INTARSIA_TP_COLL_TRACE=1`: the ledger of passes and collectives (model/coll_trace.rs).
    coll_trace: coll_trace::CollTrace,
    /// Device-side token feedback in greedy decode (`model/feedback.rs`): the backend's default, `INTARSIA_TOKEN_FEEDBACK`,
    /// or [`Model::set_token_feedback`]; only ever true on a backend that supports it.
    token_feedback: std::sync::atomic::AtomicBool,
    /// Dispatches of a speculative feedback step submitted before the previous token is read (`model/feedback.rs`).
    feedback_hold: usize,
    /// Opt-in prefill tail merge in rows (`INTARSIA_PREFILL_TAIL_MERGE`, already clamped; 0 = off). The workspace holds
    /// `prefill_chunk + tail_merge` rows when set.
    tail_merge: usize,
    /// Prefill tail peel in rows (recipe `[tuning.<backend>] prefill_peel`, `INTARSIA_PREFILL_PEEL`; 0 = off):
    /// [`crate::scheduler::peel_tail`] in [`Model::forward_chunks`].
    prefill_peel: usize,
    /// Opt-in macro prefill pass size in rows (`INTARSIA_PREFILL_MACRO`, [`LoadOptions::prefill_macro`]; a multiple of the chunk,
    /// at least two chunks, else 0 = off): [`Model::forward`] of a recurrent-state sequence cuts its prompt into the usual chunks and
    /// runs up to this many rows of consecutive chunks as ONE layer-by-layer pass (`Seqs::Packed` with one state, each chunk a segment),
    /// so the row-wise ops (the routed experts above all) read their weights once for all the chunks; attention, KV append and the
    /// GatedDeltaNet recurrence still run chunk by chunk. The workspace holds `macro_rows + tail_merge` rows.
    macro_rows: usize,
    /// What the workspace was sized for at load ([`Model::set_macro_rows`] stays within it).
    macro_cap: usize,
    /// The ops a macro pass runs once per chunk instead of once over all rows: the vendor GEMMs (cuBLAS / cuBLASLt pick their
    /// tiling, and so their summation order, by the row count), whose bits a chunk-sized call must reproduce. Everything else a
    /// macro pass batches is row-wise bitwise (norms, the pinned expert and shared-expert kernels, quantisers, top-k).
    macro_split: Vec<Op>,
    /// The rows of the macro pass being run, `(lo, len)` per chunk, `Some` only inside [`Model::run_macro_group`].
    macro_ctx: Mutex<Option<Vec<(usize, usize)>>>,
    /// Macro passes run (groups of two or more chunks), for tests and the serve log.
    macro_passes: std::sync::atomic::AtomicU64,
    /// Opt-in (`INTARSIA_PREFILL_SKIP_DEAD=1`, [`Model::set_skip_dead`]): a prefill chunk whose rows feed no head skips
    /// the last layer's work after its last cache write (the attention core and output projection, the FFN / MoE block,
    /// the final norm). Single rank only. Every cache and recurrent state byte is unchanged (nothing skipped writes them).
    skip_dead: std::sync::atomic::AtomicBool,
    /// Set by [`Model::forward_inner`] around a chunk that feeds no head (read by `run_layers`).
    tail_dead: std::sync::atomic::AtomicBool,
    /// The MTP drafter head (`LoadOptions::mtp`, `model/mtp.rs`).
    mtp: Option<mtp::MtpHead>,
    /// The DFlash block drafter (`LoadOptions::dflash`, `model/dflash.rs`).
    dflash: Option<dflash::DflashHead>,
    /// Per-row recurrent state buffers each sequence gets for verify passes (`LoadOptions::mtp_k`, MTP head only).
    mtp_k: usize,
    /// Measured-cost gate of `--spec mtp` (`spec::CostParams`): backend default, recipe `[mtp]`, `INTARSIA_SPEC_COSTGATE`.
    spec_cost: crate::spec::CostParams,
}

/// A rank's place in a tensor-parallel group and its handle for collectives.
#[derive(Clone)]
pub struct TpRank {
    pub rank: usize,
    pub world: usize,
    pub comm: Option<Arc<dyn Comm>>,
}

impl TpRank {
    pub fn single() -> TpRank {
        TpRank { rank: 0, world: 1, comm: None }
    }
}

/// `INTARSIA_TP_TEST_BUG=scale:F[@RANK]` -> (rank, factor). Unset or empty: no bug.
fn parse_tp_test_bug(v: Option<&str>) -> Result<Option<(usize, f32)>> {
    let Some(v) = v.map(str::trim).filter(|v| !v.is_empty()) else { return Ok(None) };
    let bad = || Error::invalid(format!("INTARSIA_TP_TEST_BUG={v} (expected scale:FACTOR[@RANK])"));
    let spec = v.strip_prefix("scale:").ok_or_else(bad)?;
    let (f, r) = match spec.split_once('@') {
        Some((f, r)) => (f, r.parse::<usize>().map_err(|_| bad())?),
        None => (spec, 1),
    };
    let f: f32 = f.parse().map_err(|_| bad())?;
    if !f.is_finite() {
        return Err(bad());
    }
    Ok(Some((r, f)))
}

fn resolve(src: &dyn WeightSource, name: &str) -> Result<String> {
    if src.info(name).is_some() {
        return Ok(name.to_string());
    }
    let w = format!("{name}.weight");
    if src.info(&w).is_some() {
        return Ok(w);
    }
    // an EXL3 linear layer (an EXL3 lm_head) has no `.weight`: the module prefix names it
    if src.info(&format!("{name}.trellis")).is_some() {
        return Ok(name.to_string());
    }
    Err(Error::model(format!("tensor '{name}' (or '{name}.weight') not found in checkpoint")))
}

impl Model {
    /// Tokens per forward pass (the prefill chunk); callers split long prompts on it to stay responsive.
    pub fn prefill_chunk(&self) -> usize {
        self.ws.max_tokens
    }

    /// Turn the dead-tail skip of non-final prefill chunks on or off (`INTARSIA_PREFILL_SKIP_DEAD`).
    pub fn set_skip_dead(&self, on: bool) {
        self.skip_dead.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Opt-in tail merge threshold in rows (`INTARSIA_PREFILL_TAIL_MERGE`; 0 = off, the default). See
    /// [`crate::scheduler::next_prefill_chunk_tm`].
    pub fn tail_merge(&self) -> usize {
        self.tail_merge
    }

    /// Prefill tail peel in rows (0 = off, the default; [`crate::scheduler::peel_tail`]).
    pub fn prefill_peel(&self) -> usize {
        self.prefill_peel
    }

    /// Rows of a macro prefill pass (0 = off, the default; `INTARSIA_PREFILL_MACRO`, see the field). Callers that feed a long
    /// prompt to [`Model::forward`] in steps use [`crate::scheduler::macro_step_rows`] with it.
    pub fn macro_rows(&self) -> usize {
        self.macro_rows
    }

    /// Change the macro pass size after load (tests and tools that compare schedules on one set of weights): `rows` is
    /// clamped to what the workspace was sized for at load (`LoadOptions::prefill_macro` / `INTARSIA_PREFILL_MACRO`) and
    /// rounded down to whole chunks; 0 turns the schedule off.
    pub fn set_macro_rows(&mut self, rows: usize) {
        self.macro_rows = crate::scheduler::effective_macro_rows(rows.min(self.macro_cap), self.ws.max_tokens);
    }

    /// Macro passes (groups of two or more chunks) this model has run.
    pub fn macro_passes(&self) -> u64 {
        self.macro_passes.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn load(
        backend: Arc<dyn Backend>,
        recipe: &Recipe,
        dims: &ModelDims,
        src: &dyn WeightSource,
        opts: &LoadOptions,
        telemetry: Arc<Telemetry>,
    ) -> Result<Model> {
        Self::load_shard(backend, recipe, dims, src, opts, telemetry, TpRank::single())
    }

    /// Load rank `tp.rank` of a `tp.world`-way tensor-parallel model onto `backend`: column-parallel projections
    /// (q, k, v, gate, up) keep their rows for this rank's heads / FFN slice, row-parallel ones (o, down) keep
    /// their columns, and only that slice is uploaded. `dims` are the checkpoint's. With `TpRank::single()` this is
    /// the plain model.
    pub fn load_shard(
        backend: Arc<dyn Backend>,
        recipe: &Recipe,
        dims: &ModelDims,
        src: &dyn WeightSource,
        opts: &LoadOptions,
        telemetry: Arc<Telemetry>,
        tp: TpRank,
    ) -> Result<Model> {
        recipe.validate()?;
        let global = dims;
        let local_dims = dims.sharded(tp.world)?;
        let dims = &local_dims;
        let b: &dyn Backend = &*backend;
        let weights_bytes = Cell::new(0u64);
        let load = |tmpl: &str, layer: usize, shape: Option<&[usize]>| -> Result<TensorRef> {
            let name = resolve(src, &Recipe::expand(tmpl, layer))?;
            let t = load_tensor(b, src, &name, shape)?;
            weights_bytes.set(weights_bytes.get() + t.nbytes() as u64);
            Ok(t)
        };
        // a linear layer in whatever format the checkpoint holds it (bf16, fp8, ...), checked against `[n, k]`
        let mxfp4_act = crate::quant::mxfp4::effective_act(opts.mxfp4_act, src.quant_scheme().and_then(|s| s.mxfp4_act()));
        let qopts = QuantLoadOpts { fp8_act: opts.fp8_act.unwrap_or(recipe.defaults.fp8_act), mxfp4_act, shard: None };
        // `n`, `k`: the shape this rank holds (for a sharded weight, its slice of the checkpoint's `[n, k]`)
        let load_qs = |tmpl: &str, layer: usize, n: usize, k: usize, axis: Option<ShardAxis>| -> Result<QTensor> {
            let prefix = Recipe::expand(tmpl, layer);
            let shard = axis.filter(|_| tp.world > 1).map(|axis| Shard { axis, rank: tp.rank, world: tp.world });
            let q = crate::quant::load_linear(src, &prefix, b, &QuantLoadOpts { shard, ..qopts })?;
            // what the checkpoint declares (hf_quant_config.json) must match what was stored and what we implement
            if let Some(scheme) = src.quant_scheme() {
                scheme.check_linear(&prefix, q.format)?;
            }
            if (q.n, q.k) != (n, k) {
                return Err(Error::model(format!("tensor '{prefix}' is [{}, {}], expected [{n}, {k}]", q.n, q.k)));
            }
            weights_bytes.set(weights_bytes.get() + q.weight_bytes() as u64);
            Ok(q)
        };
        let load_q = |tmpl: &str, layer: usize, n: usize, k: usize| load_qs(tmpl, layer, n, k, None);
        // a dense weight of this rank's shape `expect` in whatever slice the rank keeps (row splits take any rank of tensor)
        let load_sharded = |tmpl: &str, layer: usize, axis: ShardAxis, expect: &[usize]| -> Result<TensorRef> {
            if tp.world == 1 {
                return load(tmpl, layer, Some(expect));
            }
            let name = resolve(src, &Recipe::expand(tmpl, layer))?;
            let t = load_tensor_shard(b, src, &name, Shard { axis, rank: tp.rank, world: tp.world })?;
            if t.shape != expect {
                return Err(Error::model(format!("tensor '{name}' shard is {:?}, expected {expect:?}", t.shape)));
            }
            weights_bytes.set(weights_bytes.get() + t.nbytes() as u64);
            Ok(t)
        };

        let embed = load(&recipe.top.embed, 0, Some(&[dims.vocab, dims.hidden]))?;
        let act = match opts.act_dtype {
            Some(d) => d,
            None if embed.dtype.is_float() && embed.dtype != DType::Fp8E4m3 && embed.dtype != DType::Fp8E5m2 => {
                embed.dtype
            }
            None => DType::Bf16,
        };
        let kv = match opts.kv_dtype {
            Some(d) => d,
            None => recipe.kv_dtype()?,
        };
        let final_norm = load(&recipe.top.final_norm, 0, Some(&[dims.hidden]))?;

        // lm_head: own tensor, or tied to the embedding table.
        // (the head may be quantised: Qwen3.6 NVFP4 ships `lm_head.{weight,weight_scale,weight_scale_2,input_scale}`)
        let lh = &recipe.top.lm_head;
        let own = resolve(src, &lh.tensor).ok();
        // tied: the head is the embedding table itself (the arms below, in order)
        let tied = dims.tie_embeddings && lh.tied_to.is_some();
        // Vocab-parallel head: every rank holds `V / world` rows of the head, in rank order. A tied head is the matching
        // row range of the replicated embedding table (a view: no second copy; its bytes must stay 16-aligned for the
        // vector GEMV paths).
        let vh = dims.vocab / tp.world.max(1);
        let mut vocab_par = opts.tp_vocab
            && tp.world > 1
            && tp.comm.as_ref().and_then(|c| c.all_gather_quantum()).is_some_and(|q| dims.vocab % tp.world == 0 && (vh * 4) % q == 0)
            && (!tied || (embed.shape == [dims.vocab, dims.hidden] && (vh * dims.hidden * embed.dtype.size_bytes()) % 16 == 0));
        let dense = |t: TensorRef, n: usize| QTensor { format: crate::quant::BF16, n, k: dims.hidden, parts: vec![t], act_quant: false };
        let load_head = |vocab_par: bool| -> Result<Option<QTensor>> {
            if tp.rank != 0 && !vocab_par {
                return Ok(None);
            }
            // the rows this rank's head covers
            let (n, axis) = if vocab_par { (vh, Some(ShardAxis::Rows)) } else { (dims.vocab, None) };
            Ok(Some(match (&own, &lh.tied_to) {
                (Some(_), _) if !dims.tie_embeddings => load_qs(&lh.tensor, 0, n, dims.hidden, axis)?,
                // the config says the head is its own tensor but it is not in the checkpoint (renamed, re-exported):
                // falling back to the embedding table would serve garbage logits without any error
                (None, _) if !dims.tie_embeddings => {
                    return Err(Error::model(format!(
                        "lm_head tensor '{}' (or '{}.weight') is missing from the checkpoint and the config does not tie the \
                         embeddings (tie_word_embeddings is false or absent); if the checkpoint really ties them, say so in config.json",
                        lh.tensor, lh.tensor
                    )))
                }
                (_, Some(_)) if vocab_par => dense(embed.rows(tp.rank * vh, vh), vh),
                (_, Some(_)) => dense(embed.clone(), dims.vocab),
                (Some(_), None) => load_qs(&lh.tensor, 0, n, dims.hidden, axis)?,
                (None, None) => {
                    return Err(Error::model(format!(
                        "lm_head tensor '{}' missing and the recipe has no tied_to",
                        lh.tensor
                    )))
                }
            }))
        };
        let bytes_before = weights_bytes.get();
        let lm_head = match load_head(vocab_par) {
            // a format without a row-sharded loader (EXL3): keep the replicated head
            Err(Error::Unsupported(_)) if vocab_par => {
                weights_bytes.set(bytes_before);
                vocab_par = false;
                load_head(false)?
            }
            r => r?,
        };

        // scaled rotary embedding: the finished frequency table, uploaded once (rotary_dim as the attention blocks resolve it)
        let rope_table = if dims.rope_scaling.is_scaled() {
            let rd = if (dims.partial_rotary_factor - 1.0).abs() < 1e-6 {
                dims.head_dim
            } else {
                (dims.head_dim as f64 * dims.partial_rotary_factor as f64).round() as usize
            };
            if rd == 0 || rd % 2 != 0 || rd > dims.head_dim {
                return Err(Error::Unsupported(format!("rotary dim {rd} of head_dim {} (must be even)", dims.head_dim)));
            }
            let freqs = dims.rope_scaling.freqs(dims.rope_theta as f64, rd).expect("scaled rope has a table");
            let bytes: Vec<u8> = freqs.inv_freq.iter().flat_map(|f| f.to_le_bytes()).collect();
            let buf = crate::backend::upload_new(b, &bytes)?;
            weights_bytes.set(weights_bytes.get() + bytes.len() as u64);
            Some(RopeTable { inv_freq: TensorRef::new(buf, DType::F32, &[rd / 2]), mscale: freqs.attention_factor })
        } else {
            None
        };

        let qdim = dims.n_heads * dims.head_dim;
        let kvdim = dims.n_kv_heads * dims.head_dim;
        let mut layers = Vec::new();
        let mut slot_specs = Vec::new();
        for li in 0..dims.n_layers {
            let mut blocks = Vec::new();
            for name in recipe.layer.pattern_for(li, &dims.layer_types)? {
                let pre = recipe
                    .layer
                    .norms
                    .get(&format!("pre_{name}"))
                    .ok_or_else(|| Error::model(format!("recipe layer.norms.pre_{name} missing")))?;
                match name.as_str() {
                    "attn" => {
                        let a = recipe.layer.attn.as_ref().unwrap();
                        if a.kind != "gqa" {
                            return Err(Error::Unsupported(format!(
                                "attention kind '{}' is not implemented in this build",
                                a.kind
                            )));
                        }
                        if a.rope != "default" {
                            return Err(Error::Unsupported(format!("rope variant '{}'", a.rope)));
                        }
                        // partial rotary embeddings: only the first `rotary_dim` values of each head rotate
                        let rotary_dim = if (dims.partial_rotary_factor - 1.0).abs() < 1e-6 {
                            0
                        } else {
                            let rd = (dims.head_dim as f64 * dims.partial_rotary_factor as f64).round() as usize;
                            if rd == 0 || rd % 2 != 0 || rd > dims.head_dim {
                                return Err(Error::Unsupported(format!(
                                    "partial_rotary_factor {} of head_dim {} gives rotary dim {rd} (must be even)",
                                    dims.partial_rotary_factor, dims.head_dim
                                )));
                            }
                            rd
                        };
                        let gate = usize::from(a.gated);
                        if a.gated && (a.q_bias.is_some() || a.k_bias.is_some() || a.v_bias.is_some()) {
                            return Err(Error::Unsupported("projection biases on a gated attention block".into()));
                        }
                        // a bias in the checkpoint that the recipe does not map would be dropped without any error
                        for (proj, mapped) in [(&a.q, &a.q_bias), (&a.k, &a.k_bias), (&a.v, &a.v_bias)] {
                            let name = format!("{}.bias", Recipe::expand(proj, li));
                            if mapped.is_none() && src.info(&name).is_some() {
                                return Err(Error::model(format!(
                                    "the checkpoint has '{name}' but the recipe maps no bias for it ([layer.attn] q_bias / k_bias / v_bias)"
                                )));
                            }
                        }
                        let slot = slot_specs.len();
                        slot_specs.push(SlotSpec::AttnKv {
                            n_kv_heads: dims.n_kv_heads,
                            head_dim: dims.head_dim,
                            dtype: kv,
                        });
                        blocks.push(Block::Attn(AttnBlock {
                            pre_norm: load(pre, li, Some(&[dims.hidden]))?,
                            wq: load_qs(&a.q, li, qdim * (1 + gate), dims.hidden, Some(ShardAxis::Rows))?,
                            wk: load_qs(&a.k, li, kvdim, dims.hidden, Some(ShardAxis::Rows))?,
                            wv: load_qs(&a.v, li, kvdim, dims.hidden, Some(ShardAxis::Rows))?,
                            wo: load_qs(&a.o, li, dims.hidden, qdim, Some(ShardAxis::Cols))?,
                            q_norm: a.q_norm.as_ref().map(|n| load(n, li, Some(&[dims.head_dim]))).transpose()?,
                            k_norm: a.k_norm.as_ref().map(|n| load(n, li, Some(&[dims.head_dim]))).transpose()?,
                            q_bias: a.q_bias.as_ref().map(|n| load_sharded(n, li, ShardAxis::Rows, &[qdim])).transpose()?,
                            k_bias: a.k_bias.as_ref().map(|n| load_sharded(n, li, ShardAxis::Rows, &[kvdim])).transpose()?,
                            v_bias: a.v_bias.as_ref().map(|n| load_sharded(n, li, ShardAxis::Rows, &[kvdim])).transpose()?,
                            slot,
                            gated: a.gated,
                            rotary_dim,
                        }));
                    }
                    "gdn" => {
                        let g = recipe.layer.gdn.as_ref().unwrap();
                        if g.kind != "gated_delta_net" {
                            return Err(Error::Unsupported(format!(
                                "linear attention kind '{}' is not implemented in this build",
                                g.kind
                            )));
                        }
                        let lin: LinearDims = dims.linear.ok_or_else(|| {
                            Error::model("the recipe has a gdn block but config.json has no linear attention dimensions")
                        })?;
                        let (c, vdim, hv) = (lin.conv_dim(), lin.v_heads * lin.v_head_dim, lin.v_heads);
                        let slot = slot_specs.len();
                        slot_specs.push(SlotSpec::Recurrent { bytes: gdn_block::state_bytes(&lin) });
                        // tensor parallelism (docs/design/35b-tp2.md): by key-head group. `lin` is this rank's (heads
                        // divided); the fused q/k/v projection and the conv channels are `[q | k | v]` of the checkpoint's
                        // widths, each group split on its own. v head h reads k head h / (Hv / Hk), so a contiguous run of
                        // k heads carries its own v heads.
                        let qkv_groups = ShardAxis::RowGroups([
                            lin.k_heads * lin.k_head_dim * tp.world,
                            lin.k_heads * lin.k_head_dim * tp.world,
                            vdim * tp.world,
                        ]);
                        let conv = if tp.world == 1 {
                            load(&g.conv, li, None)?
                        } else {
                            let name = resolve(src, &Recipe::expand(&g.conv, li))?;
                            let t = load_tensor_shard(b, src, &name, Shard { axis: qkv_groups, rank: tp.rank, world: tp.world })?;
                            weights_bytes.set(weights_bytes.get() + t.nbytes() as u64);
                            t
                        };
                        if conv.numel() != c * lin.conv_kernel {
                            return Err(Error::model(format!(
                                "conv taps {:?} do not match [{c}, {}]",
                                conv.shape, lin.conv_kernel
                            )));
                        }
                        let rows = Some(ShardAxis::Rows);
                        blocks.push(Block::Gdn(GdnBlock {
                            pre_norm: load(pre, li, Some(&[dims.hidden]))?,
                            in_qkv: load_qs(&g.in_qkv, li, c, dims.hidden, Some(qkv_groups))?,
                            in_z: load_qs(&g.in_z, li, vdim, dims.hidden, rows)?,
                            in_b: load_qs(&g.in_b, li, hv, dims.hidden, rows)?,
                            in_a: load_qs(&g.in_a, li, hv, dims.hidden, rows)?,
                            conv: conv.reshaped(&[c, lin.conv_kernel]),
                            a_log: load_sharded(&g.a_log, li, ShardAxis::Rows, &[hv])?,
                            dt_bias: load_sharded(&g.dt_bias, li, ShardAxis::Rows, &[hv])?,
                            norm: load(&g.norm, li, Some(&[lin.v_head_dim]))?,
                            out: load_qs(&g.out, li, dims.hidden, vdim, Some(ShardAxis::Cols))?,
                            slot,
                            lin,
                        }));
                    }
                    "moe" => {
                        let mc = dims.moe.as_ref().ok_or_else(|| {
                            Error::model("the recipe has a moe block but config.json has no MoE dimensions")
                        })?;
                        let m = recipe.layer.other.get("moe").ok_or_else(|| Error::model("recipe [layer.moe] missing"))?;
                        if m.get("kind").and_then(|v| v.as_str()) != Some("qwen3_5_moe") {
                            return Err(Error::Unsupported(format!("moe kind {:?} is not implemented in this build", m.get("kind"))));
                        }
                        let tmpl = m
                            .get("prefix")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| Error::model("recipe [layer.moe] needs a prefix"))?;
                        // tensor parallelism: inside every expert (`mc` is this rank's config, widths divided)
                        let w = MoeWeights::load_shard(b, src, &Recipe::expand(tmpl, li), mc, &qopts, tp.rank, tp.world)?;
                        weights_bytes.set(weights_bytes.get() + w.weight_bytes() as u64);
                        blocks.push(Block::Moe(MoeBlock { pre_norm: load(pre, li, Some(&[dims.hidden]))?, w }));
                    }
                    "gemma" => {
                        let (Some(gd), Some(mc)) = (dims.gemma.as_ref(), dims.moe.as_ref()) else {
                            return Err(Error::model("the recipe has a gemma block but config.json is not a gemma4_text config"));
                        };
                        if tp.world > 1 {
                            return Err(Error::Unsupported("tensor parallelism of Gemma 4".into()));
                        }
                        let gcfg = recipe.layer.other.get("gemma").ok_or_else(|| Error::model("recipe [layer.gemma] missing"))?;
                        if gcfg.get("kind").and_then(|v| v.as_str()) != Some("gemma4") {
                            return Err(Error::Unsupported(format!("gemma block kind {:?}", gcfg.get("kind"))));
                        }
                        let ptmpl = gcfg
                            .get("prefix")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| Error::model("recipe [layer.gemma] needs a prefix (the layer module, `...layers.{i}`)"))?;
                        let lp = Recipe::expand(ptmpl, li);
                        let global = dims.layer_types.get(li).map(String::as_str) == Some("full_attention");
                        if !global && dims.layer_types.get(li).map(String::as_str) != Some("sliding_attention") {
                            return Err(Error::model(format!("gemma4 layer {li} has type {:?} (sliding_attention or full_attention)", dims.layer_types.get(li))));
                        }
                        let (nh, nkv, hd) = if global { (dims.n_heads, gd.global_kv_heads, gd.global_head_dim) } else { (dims.n_heads, dims.n_kv_heads, dims.head_dim) };
                        let h = dims.hidden;
                        let norm = |suffix: &str| load(&format!("{lp}.{suffix}"), li, Some(&[h]));
                        let lq = |suffix: &str, n: usize, k: usize| load_q(&format!("{lp}.{suffix}"), li, n, k);
                        let slot = slot_specs.len();
                        slot_specs.push(SlotSpec::AttnKv { n_kv_heads: nkv, head_dim: hd, dtype: kv });
                        let v_in_ckpt = src.info(&format!("{lp}.self_attn.v_proj.weight")).is_some();
                        let wv = if global && gd.k_eq_v {
                            if v_in_ckpt {
                                return Err(Error::model(format!("{lp}.self_attn.v_proj exists but config says attention_k_eq_v")));
                            }
                            None
                        } else {
                            Some(lq("self_attn.v_proj", nkv * hd, h)?)
                        };
                        let ones = {
                            let bytes = gemma::encode_floats(act, &vec![1.0f32; hd])?;
                            TensorRef::new(crate::backend::upload_new(b, &bytes)?, act, &[hd])
                        };
                        // router input norm weight = router.scale * hidden^-0.5 (HF applies both after a scale-less RMSNorm)
                        let router_norm = {
                            let name = format!("{lp}.router.scale");
                            let info = src.info(&name).ok_or_else(|| Error::model(format!("tensor '{name}' not found in checkpoint")))?;
                            if info.shape != [h] {
                                return Err(Error::model(format!("tensor '{name}' is {:?}, expected [{h}]", info.shape)));
                            }
                            let root = (h as f32).powf(-0.5);
                            let v: Vec<f32> = crate::weights::to_f32_vec(info.dtype, src.bytes(&name)?)?.into_iter().map(|x| x * root).collect();
                            let bytes = gemma::encode_floats(act, &v)?;
                            TensorRef::new(crate::backend::upload_new(b, &bytes)?, act, &[h])
                        };
                        let layer_scalar = {
                            let name = format!("{lp}.layer_scalar");
                            let info = src.info(&name).ok_or_else(|| Error::model(format!("tensor '{name}' not found in checkpoint")))?;
                            crate::weights::to_f32_vec(info.dtype, src.bytes(&name)?)?.first().copied().ok_or_else(|| Error::model(format!("tensor '{name}' is empty")))?
                        };
                        if let Some(scheme) = src.quant_scheme() {
                            let m = format!("{lp}.experts.0.gate_proj");
                            let declared = scheme.algo_of(&m).map(crate::quant::scheme::Algo::parse);
                            if declared == Some(crate::quant::scheme::Algo::Nvfp4W4A4) && !recipe.defaults.w4a4_as_w4a16 {
                                return Err(Error::Unsupported(format!(
                                    "'{m}' is declared quant_algo NVFP4 (weights AND 4-bit activations) in hf_quant_config.json; this engine \
                                     runs it weight-only (W4A16), different from the declared scheme and from vLLM. Set \
                                     [defaults] w4a4_as_w4a16 = true in the recipe to accept that"
                                )));
                            }
                        }
                        let moe = MoeWeights::load_gemma(b, src, &lp, mc, &qopts)?;
                        weights_bytes.set(weights_bytes.get() + moe.weight_bytes() as u64);
                        blocks.push(Block::Gemma(GemmaLayer {
                            nh,
                            nkv,
                            hd,
                            rotary_dim: if global { (gd.global_rotary_factor * hd as f32) as usize } else { 0 },
                            theta: if global { gd.global_theta } else { dims.rope_theta },
                            window: if global { 0 } else { gd.sliding_window },
                            slot,
                            input_norm: load(pre, li, Some(&[h]))?,
                            post_attn_norm: norm("post_attention_layernorm")?,
                            pre_ffn_norm: norm("pre_feedforward_layernorm")?,
                            post_ffn_norm: norm("post_feedforward_layernorm")?,
                            pre_ffn2_norm: norm("pre_feedforward_layernorm_2")?,
                            post_ffn1_norm: norm("post_feedforward_layernorm_1")?,
                            post_ffn2_norm: norm("post_feedforward_layernorm_2")?,
                            q_norm: load(&format!("{lp}.self_attn.q_norm"), li, Some(&[hd]))?,
                            k_norm: load(&format!("{lp}.self_attn.k_norm"), li, Some(&[hd]))?,
                            ones,
                            wq: lq("self_attn.q_proj", nh * hd, h)?,
                            wk: lq("self_attn.k_proj", nkv * hd, h)?,
                            wv,
                            wo: lq("self_attn.o_proj", h, nh * hd)?,
                            gate: lq("mlp.gate_proj", dims.intermediate, h)?,
                            up: lq("mlp.up_proj", dims.intermediate, h)?,
                            down: lq("mlp.down_proj", h, dims.intermediate)?,
                            router_norm,
                            moe,
                            layer_scalar,
                        }));
                    }
                    "ffn" => {
                        let f = recipe.layer.ffn.as_ref().unwrap();
                        if f.kind != "dense" || f.act != "silu_glu" {
                            return Err(Error::Unsupported(format!(
                                "ffn kind '{}' / act '{}' is not implemented in this build",
                                f.kind, f.act
                            )));
                        }
                        blocks.push(Block::Ffn(FfnBlock {
                            pre_norm: load(pre, li, Some(&[dims.hidden]))?,
                            gate: load_qs(&f.gate, li, dims.intermediate, dims.hidden, Some(ShardAxis::Rows))?,
                            up: load_qs(&f.up, li, dims.intermediate, dims.hidden, Some(ShardAxis::Rows))?,
                            down: load_qs(&f.down, li, dims.hidden, dims.intermediate, Some(ShardAxis::Cols))?,
                        }));
                    }
                    other => {
                        return Err(Error::Unsupported(format!(
                            "block kind '{other}' is not implemented in this build"
                        )))
                    }
                }
            }
            layers.push(blocks);
        }

        // activation workspace
        // tensor parallelism: INTARSIA_TP_PARTIAL=bf16 reduces bf16 partials in place (two roundings per reduce, like
        // vLLM); the default keeps the partials in f32 and rounds once, as one unsharded matmul does
        let part_f32 = match std::env::var("INTARSIA_TP_PARTIAL").ok().as_deref() {
            None | Some("") | Some("f32") => true,
            Some("bf16") => false,
            Some(v) => return Err(Error::invalid(format!("INTARSIA_TP_PARTIAL={v} (f32 or bf16)"))),
        };
        if opts.tp_fusion && tp.world > 1 && !b.fusion_under_tp() {
            return Err(Error::Unsupported(format!(
                "fused ops under tensor parallelism (INTARSIA_TP_FUSION) on backend '{}' (Backend::fusion_under_tp)",
                b.name()
            )));
        }
        let chunk = opts.prefill_chunk.max(1);
        let tail_merge = crate::scheduler::effective_tail_merge(
            crate::scheduler::parse_tail_merge(std::env::var(crate::scheduler::TAIL_MERGE_ENV).ok().as_deref())?,
            chunk,
        );
        // the recipe's prefill tail peel for this backend, `INTARSIA_PREFILL_PEEL` over it (0 = off)
        let prefill_peel =
            match crate::scheduler::parse_peel(std::env::var(crate::scheduler::PEEL_ENV).ok().as_deref())? {
                Some(v) => v,
                None => recipe.resolve_prefill_peel(b.name()).0,
            };
        // the opt-in macro prefill pass: several chunks of one prompt in one layer-by-layer pass (hybrid models, one rank)
        // (`LoadOptions::prefill_macro`, then `INTARSIA_PREFILL_MACRO`, then the recipe's `[tuning.<backend>] prefill_macro`; never
        // more rows than the context holds)
        let macro_rows = crate::scheduler::effective_macro_rows(
            match opts.prefill_macro {
                Some(v) => v,
                None => match crate::scheduler::parse_macro_env(std::env::var(crate::scheduler::MACRO_ENV).ok().as_deref())? {
                    Some(v) => v,
                    None => recipe.resolve_prefill_macro(b.name()).0,
                },
            }
            .min(opts.max_seq),
            chunk,
        );
        // one rank only (tensor-parallel passes are not packed), and no MTP head or DFlash draft (its feature taps are per chunk): the drafter's workspace and its plan probes are
        // chunk-sized, and a speculative sequence never takes a macro pass (`macro_plan`)
        let macro_rows = if tp.world > 1 || opts.mtp || opts.dflash.is_some() { 0 } else { macro_rows };
        // the workspace (and so the memory plan below) holds the merged tail chunk: `chunk + tail_merge` rows, and a macro
        // pass of `macro_rows` (+ the merged tail)
        let mt = chunk.max(macro_rows) + tail_merge;
        let esz = act.size_bytes();
        // the MTP drafter head (`--spec mtp`): read only when asked for, so without it the model is unchanged
        let mtp_k = if opts.mtp_k == 0 { crate::spec::SPEC_K_MAX } else { opts.mtp_k.min(crate::spec::SPEC_K_MAX) };
        let mtp_head = if opts.mtp {
            let spec = recipe.mtp.as_ref().ok_or_else(|| Error::Unsupported("--spec mtp: the recipe has no [mtp] head".into()))?;
            if tp.world > 1 {
                return Err(Error::Unsupported("--spec mtp under tensor parallelism".into()));
            }
            let main_attn = layers.iter().flatten().find_map(|b| match b {
                Block::Attn(a) => Some(a),
                _ => None,
            });
            let t0 = Instant::now();
            let head = mtp::load_head(
                spec,
                b,
                src,
                dims,
                recipe,
                main_attn,
                &|name, shape| load(name, 0, shape),
                &|name, n, k| load_q(name, 0, n, k),
                &|n| weights_bytes.set(weights_bytes.get() + n),
                mt + Self::VERIFY_MAX_ROWS + 1,
            )?;
            eprintln!("mtp: drafter head '{}' loaded in {:.1} s (routed experts quantised to nvfp4)", spec.prefix, t0.elapsed().as_secs_f64());
            Some(head)
        } else {
            None
        };
        // the DFlash drafter (`--spec dflash`): read only when asked for
        let dflash_head = match &opts.dflash {
            Some(dl) => {
                if tp.world > 1 {
                    return Err(Error::Unsupported("--spec dflash under tensor parallelism".into()));
                }
                let main_attn = layers.iter().flatten().find_map(|b| match b {
                    Block::Attn(a) => Some(a),
                    _ => None,
                });
                let t0 = Instant::now();
                let head = dflash::load_head(dl, b, dims, recipe, main_attn, &|n| weights_bytes.set(weights_bytes.get() + n), mt, act)?;
                eprintln!(
                    "dflash: draft of {} layers (taps {:?}, block {}) loaded in {:.1} s",
                    head.layers.len(),
                    head.taps,
                    head.block,
                    t0.elapsed().as_secs_f64()
                );
                Some(head)
            }
            None => None,
        };
        let mut ws_bytes = 0u64;
        let mut wb = |bytes: usize| -> Result<Buffer> {
            ws_bytes += bytes as u64;
            b.alloc(bytes.max(16), MemKind::Device)
        };
        let (has_gdn, has_gated) = layers.iter().flatten().fold((false, false), |(g, a), blk| match blk {
            Block::Gdn(_) => (true, a),
            Block::Attn(x) => (g, a || x.gated),
            _ => (g, a),
        });
        let x = gdn_block::alloc_extra(&mut wb, mt, esz, dims, has_gdn, has_gated)?;
        let mut moe_bytes = 0u64;
        let moe = match dims.moe.as_ref().filter(|_| layers.iter().flatten().any(|bl| matches!(bl, Block::Moe(_)))) {
            Some(mc) => {
                let si = mc.shared_inter.max(1);
                moe_bytes = (mt * (mc.experts * esz + mc.top_k * 8 + 2 * mc.hidden * esz + 3 * si * esz + 2 * esz)) as u64;
                Some(MoeScratch::new(b, mc, mt, act)?)
            }
            None => None,
        };
        let gemma_ws = match &dims.gemma {
            Some(gd) if layers.iter().flatten().any(|bl| matches!(bl, Block::Gemma(_))) => Some(GemmaWs::alloc(&mut wb, mt, esz, dims, gd)?),
            _ => None,
        };
        let ws = Workspace {
            max_tokens: chunk,
            ids: wb(mt * 4)?,
            h: [wb(mt * dims.hidden * esz)?, wb(mt * dims.hidden * esz)?],
            n: wb(mt * dims.hidden * esz)?,
            blk: wb(mt * dims.hidden * esz)?,
            q: [wb(mt * qdim * esz)?, wb(mt * qdim * esz)?],
            k: [wb(mt * kvdim * esz)?, wb(mt * kvdim * esz)?],
            v: wb(mt * kvdim * esz)?,
            g: wb(mt * dims.intermediate * esz)?,
            u: wb(mt * dims.intermediate * esz)?,
            a: wb(mt * dims.intermediate * esz)?,
            logits: wb(LOGIT_ROWS * dims.vocab * 4)?,
            tok: wb(16)?,
            part: if tp.world > 1 && part_f32 { Some(wb(mt * dims.hidden * 4)?) } else { None },
            pos: wb(POS_TAB_BYTES)?,
            x,
            moe,
        };
        // recurrent state is fixed-size: the live copy plus the prefix-reuse snapshots count against the budget
        let rec_bytes: u64 = slot_specs
            .iter()
            .map(|s| match s {
                SlotSpec::Recurrent { bytes } => *bytes as u64,
                _ => 0,
            })
            .sum();
        ws_bytes += moe_bytes;
        ws_bytes += rec_bytes * (1 + crate::state::RECURRENT_SNAPSHOTS as u64);
        if mtp_head.is_some() {
            // per-row recurrent states of a verify pass (up to SPEC_K_MAX saved rows) and the drafter's hidden-row queue
            ws_bytes += rec_bytes * mtp_k as u64;
            ws_bytes += ((mt + Self::VERIFY_MAX_ROWS + 1) * dims.hidden * esz) as u64;
        }

        let kv_per_tok: u64 = slot_specs
            .iter()
            .map(|s| match s {
                SlotSpec::AttnKv { n_kv_heads, head_dim, dtype } => {
                    (2 * n_kv_heads * head_dim * dtype.size_bytes()) as u64
                }
                _ => 0,
            })
            .sum::<u64>()
            // the drafter head's own attention cache
            + if mtp_head.is_some() { (2 * dims.n_kv_heads * dims.head_dim * kv.size_bytes()) as u64 } else { 0 }
            // the DFlash draft layers' context caches
            + dflash_head.as_ref().map_or(0, |h| (h.layers.len() * 2 * dims.n_kv_heads * dims.head_dim * kv.size_bytes()) as u64);
        let mut plan = MemoryPlan::compute_budgeted(
            &b.info(),
            weights_bytes.get(),
            ws_bytes,
            kv_per_tok,
            opts.max_seq,
            dims.max_positions,
            opts.kv_fraction,
            opts.mem_budget,
        )?;
        if opts.tp_vocab && tp.world > 1 && !vocab_par {
            plan.notes.push("vocab-parallel lm_head requested but not possible here (the communicator has no in-place all_gather for this size, the vocabulary does not split, or the head format has no row-sharded loader): the head stays on rank 0".into());
        }
        let registry = Registry::new(b.name(), &b.info(), b.kernels());
        if let Some(line) = b.layout_report() {
            eprintln!("{line}");
        }
        let token_feedback = feedback::initial(b.token_feedback(), std::env::var("INTARSIA_TOKEN_FEEDBACK").ok().as_deref());
        let feedback_hold = feedback::hold_from_env(std::env::var("INTARSIA_TOKEN_FEEDBACK_HOLD").ok().as_deref());

        let no_comm = std::env::var_os("INTARSIA_TP_NO_COMM").is_some_and(|v| v != "0");
        let test_bug = parse_tp_test_bug(std::env::var("INTARSIA_TP_TEST_BUG").ok().as_deref())?;
        let dv = recipe.resolve_mtp_draft_vocab(opts.mtp_draft_vocab, b.name());
        let draft_vocab = dv.value;
        if opts.mtp && opts.mtp_draft_vocab.is_none() && draft_vocab > 0 {
            eprintln!("--mtp-draft-vocab {draft_vocab} ({}; --mtp-draft-vocab 0 drafts with the full head)", dv.source);
        }
        let draft_head = if opts.mtp && draft_vocab > 0 {
            match lm_head.as_ref().filter(|_| !vocab_par) {
                None => {
                    if opts.mtp_draft_vocab.is_some() {
                        eprintln!(
                            "--mtp-draft-vocab {}: rank {} of the tensor-parallel group keeps no lm_head (only rank 0 drafts); ignored here",
                            draft_vocab, tp.rank
                        );
                    }
                    None
                }
                Some(h) => match h.row_prefix(draft_vocab) {
                    Some(d) => Some(d),
                    None => {
                        eprintln!(
                            "--mtp-draft-vocab {}: not applicable to this lm_head ({} rows, format {:?}; needs a multiple of 8 below the row count, \
                             not EXL3); drafting with the full head",
                            draft_vocab, h.n, h.format
                        );
                        None
                    }
                },
            }
        } else {
            None
        };
        let spec_cost = if dflash_head.is_some() {
            // the DFlash drafter: gate off by default (built-in values, the recipe's `[dflash]` table, then the env var)
            crate::spec::CostParams::DFLASH.with_dflash_recipe(recipe.dflash.as_ref())
        } else {
            crate::spec::CostParams::for_backend(b.name()).with_recipe(recipe.mtp.as_ref())
        }
        .with_env(std::env::var("INTARSIA_SPEC_COSTGATE").ok().as_deref());
        let mut model = Model {
            backend,
            registry,
            dims: global.clone(),
            ldims: dims.clone(),
            recipe: recipe.clone(),
            act,
            kv,
            plan,
            weights_bytes: weights_bytes.get(),
            embed,
            final_norm,
            lm_head,
            draft_head,
            layers: Arc::new(layers),
            gemma: gemma_ws,
            slot_specs,
            ws,
            telemetry,
            profile_every: opts.profile_every,
            scratch: None,
            recorder: Mutex::new(None),
            resolved: Default::default(),
            // at world > 1 also gated on a capture-safe communicator (`GraphState::usable`)
            // Gemma 4 runs its layers eagerly (no captured decode step yet)
            graphs: GraphState::new(opts.graphs && dims.gemma.is_none()),
            capture_ops: Mutex::new(None),
            capture_comms: std::sync::atomic::AtomicU64::new(0),
            rope_table,
            tp_fusion: opts.tp_fusion && tp.world > 1,
            vocab_par,
            tp_arnorm: ar_norm::enabled(opts, &tp, no_comm, test_bug.is_some()),
            ar_defer: Mutex::new(None),
            ar_fused_eager: Default::default(),
            ar_fused_captured: Default::default(),
            tp,
            no_comm,
            test_bug,
            peers: Vec::new(),
            remote: None,
            exact_rows: std::sync::atomic::AtomicBool::new(false),
            head_rows_attn: std::sync::atomic::AtomicBool::new(false),
            noncausal_rows: std::sync::atomic::AtomicBool::new(false),
            verify_rows_attn: dflash_head.is_some(),
            verify_tok: None,
            verify_rows: 0,
            batch_rows: 0,
            recurrent_batch: false,
            batch_tok: None,
            stream: 0,
            row_tab: None,
            batch_pos: None,
            topk: None,
            batch_tile: None,
            tp_batch_mutant: std::sync::atomic::AtomicU8::new(0),
            coll_trace: coll_trace::CollTrace::new(std::env::var("INTARSIA_TP_COLL_TRACE").is_ok_and(|v| v == "1")),
            token_feedback: std::sync::atomic::AtomicBool::new(token_feedback),
            feedback_hold,
            tail_merge,
            prefill_peel,
            macro_rows,
            macro_cap: macro_rows,
            macro_split: macro_split_ops()?,
            macro_ctx: Mutex::new(None),
            macro_passes: std::sync::atomic::AtomicU64::new(0),
            skip_dead: std::sync::atomic::AtomicBool::new(std::env::var("INTARSIA_PREFILL_SKIP_DEAD").is_ok_and(|v| v == "1")),
            tail_dead: std::sync::atomic::AtomicBool::new(false),
            spec_cost,
            mtp: mtp_head,
            dflash: dflash_head,
            mtp_k,
        };
        let tp_comm = model.tp.comm.clone();
        model.coll_trace.set_comm(tp_comm);
        // the f32-partial reduce's landing buffer, sized for the largest chunk now: a captured decode graph keeps its address
        if let (Some(c), Some(_)) = (&model.tp.comm, &model.ws.part) {
            c.reserve(mt * dims.hidden, 0)?;
        }
        // resolve the kernel of every (op, shape) now: a missing kernel is a load error
        crate::plan::resolve(&mut model)?;
        Ok(model)
    }

    /// Apply plan options (`--reference`, admission, `require_admitted`) and re-resolve. `reference`
    /// is the oracle backend for admission (the cpu backend); required when `opts.admit` is set.
    pub fn configure_plan(
        &mut self,
        opts: &crate::plan::PlanOptions,
        reference: Option<&dyn Backend>,
    ) -> Result<crate::plan::AdmitSummary> {
        self.graphs.invalidate(); // kernel choices change
        self.registry.reference_only = opts.force_reference;
        self.registry.require_admitted = false;
        let mut summary = Default::default();
        if opts.admit {
            let r = reference.ok_or_else(|| Error::invalid("plan admission needs a reference backend"))?;
            summary = crate::plan::admit(self, r, opts)?;
        }
        self.registry.require_admitted = opts.require_admitted;
        crate::plan::resolve(self)?;
        for i in 0..self.peers.len() {
            self.peers[i].registry.adopt_status(&self.registry);
            self.peers[i].graphs.invalidate();
            crate::plan::resolve(&mut self.peers[i])?;
        }
        Ok(summary)
    }

    /// Run the real forward path for `t` tokens at `pos0` with a recorder installed: every `exec`
    /// selects its kernel and is logged, nothing launches, nothing is uploaded or downloaded.
    pub fn dry_run(&self, phase: crate::plan::Phase, t: usize, pos0: usize) -> Result<crate::plan::Recorder> {
        let st = crate::plan::phantom_state(&self.slot_specs, self.plan.kv_cap_tokens, pos0);
        *self.recorder.lock().unwrap() = Some(crate::plan::Recorder { phase: Some(phase), t, pos0, ..Default::default() });
        let ids = vec![0u32; t];
        let verify = phase == crate::plan::Phase::Verify;
        let batch = phase == crate::plan::Phase::Batch;
        if verify || batch {
            self.exact_rows.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let run = (|| -> Result<()> {
            if batch {
                // `t` sequences, each `pos0` tokens long: the real batched pass, recorded
                return self.batch_probe(t, pos0);
            }
            let hn = self.run_chunk(&st, &ids, false)?;
            // the MTP drafter head runs after every pass (catch-up over up to t rows, chained drafts of one row)
            self.mtp_dry_run(t, pos0)?;
            self.dflash_dry_run()?;
            if verify {
                return self.verify_head(&hn, t, false);
            }
            // the heads generation and accuracy use: argmax of the last row, logits of up to LOGIT_ROWS rows
            if self.lm_head.is_some() {
                self.head_argmax(&hn, t - 1, false)?;
                if let Some(r) = self.recorder.lock().unwrap().as_mut() {
                    r.extra = true;
                }
                self.head_logits(&hn, 0, t.min(LOGIT_ROWS), false)?;
            }
            Ok(())
        })();
        self.exact_rows.store(false, std::sync::atomic::Ordering::Relaxed);
        let rec = self.recorder.lock().unwrap().take().unwrap_or_default();
        run?;
        Ok(rec)
    }

    /// Most rows a verify pass ([`Model::forward_verify`]) takes: one logits buffer of rows.
    pub const VERIFY_MAX_ROWS: usize = LOGIT_ROWS;

    /// Enable speculative verify passes of up to `max_rows` rows (2..=[`Model::VERIFY_MAX_ROWS`]). Call before
    /// [`Model::configure_plan`]: plan resolution then also dry-runs [`crate::plan::Phase::Verify`] passes (every row
    /// count, at the start and the end of the context), so a verify op without a row-exact kernel fails the load,
    /// admission covers the verify kernels and the scratch workspace is sized for them. Single-rank models only.
    pub fn enable_verify(&mut self, max_rows: usize) -> Result<()> {
        if !self.peers.is_empty() || self.tp.world > 1 || self.lm_head.is_none() {
            return Err(Error::Unsupported("speculative verify needs a single-rank model".into()));
        }
        let most = Self::VERIFY_MAX_ROWS.min(self.ws.max_tokens);
        if !(2..=most).contains(&max_rows) {
            return Err(Error::invalid(format!("verify rows {max_rows} outside 2..={most}")));
        }
        if self.verify_tok.is_none() {
            // [0, rows): the argmax of each row; [rows, 2 rows): the pass's input ids (a captured speculative step reads its
            // drafts back with the outputs, `Model::spec_step`)
            self.verify_tok = Some(self.backend.alloc(Self::VERIFY_MAX_ROWS * 8, MemKind::Device)?);
        }
        self.verify_rows = max_rows;
        Ok(())
    }

    /// Most rows a verify pass may have (0: speculative verify not enabled).
    /// The measured-cost gate parameters for `--spec mtp` requests (single-sequence engine).
    pub fn spec_cost_params(&self) -> crate::spec::CostParams {
        self.spec_cost
    }

    pub fn verify_rows(&self) -> usize {
        self.verify_rows
    }

    /// Speculative verify pass: run `ids` (`[last token, drafts...]`, 2..=rows given to
    /// [`Model::enable_verify`]) as one forward with every op on its row-exact kernel, and return the greedy
    /// token of every row. Row `t`'s logits are bitwise those a single-token decode step at position
    /// `st.len + t` would compute, so its argmax is the token sequential greedy decode would pick there. All rows
    /// are committed; the caller truncates the rejected ones (attention reads keys `<= st.len + t` only, so rows
    /// past the accepted length are never seen and are overwritten by the next append).
    pub fn forward_verify(&self, st: &mut SequenceState, ids: &[u32]) -> Result<Vec<u32>> {
        Self::check_forward(st, ids, HeadMode::LastArgmax, false)?;
        let q = ids.len();
        let Some(tok) = &self.verify_tok else {
            return Err(Error::invalid("forward_verify before enable_verify"));
        };
        if !(2..=self.verify_rows).contains(&q) {
            return Err(Error::invalid(format!("forward_verify of {q} rows (enabled for 2..={})", self.verify_rows)));
        }
        let profile = self.profile_every > 0 && self.telemetry.steps_total() % self.profile_every as u64 == 0;
        // recurrent slots (GatedDeltaNet): the delta rule runs row by row and the state after every row but the last is
        // kept (`SpecRows`), so the caller can roll back to any accepted length (`SequenceState::spec_rollback`)
        let recurrent = st.has_recurrent();
        if recurrent {
            let need = st.recurrent_bytes();
            st.spec_rows.saved = 0;
            st.spec_rows.base = st.len;
            st.spec_rows.bufs.retain(|b| b.bytes() >= need);
            while st.spec_rows.bufs.len() < q - 1 {
                let b = self.backend.alloc(need.max(16), MemKind::Device)?;
                st.spec_rows.bufs.push(b);
            }
        }
        self.exact_rows.store(true, std::sync::atomic::Ordering::Relaxed);
        let run = (|| -> Result<TensorRef> {
            let hn = self.run_chunk(st, ids, profile)?;
            self.verify_head(&hn, q, profile)?;
            Ok(hn)
        })();
        self.exact_rows.store(false, std::sync::atomic::Ordering::Relaxed);
        let hn = match run {
            Ok(hn) => hn,
            Err(e) => {
                if recurrent {
                    // the recurrent slots were updated in place part-way: they match no token prefix
                    let _ = st.reset(&*self.backend);
                }
                return Err(e);
            }
        };
        let mut b = vec![0u8; q * 4];
        self.backend.download(tok, 0, &mut b, self.stream)?;
        st.commit(ids);
        if recurrent {
            st.spec_rows.saved = q - 1;
        }
        self.telemetry.step();
        // the drafter queues every row; the caller's rollback drops the rejected ones
        self.mtp_after_forward(st, &hn, q, false, profile)?;
        // the DFlash context takes every row too; the caller's rollback shortens it with the target
        self.dflash_after_forward(st, q, profile)?;
        Ok(b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32).collect())
    }

    /// lm_head of rows `[0, q)` into `ws.logits` (`[q, V]` f32) and their argmax into `verify_tok` (`i32 [q]`).
    fn verify_head(&self, hn: &TensorRef, q: usize, profile: bool) -> Result<()> {
        let v = self.dims.vocab;
        let tok = self.verify_tok.as_ref().ok_or_else(|| Error::invalid("verify head without its buffer"))?;
        let logits = self.view(&self.ws.logits, DType::F32, &[q, v]);
        self.proj(&hn.rows(0, q), self.head()?, &logits, profile)?;
        self.exec(Op::Argmax, vec![logits], vec![self.view(tok, DType::I32, &[q])], OpAttrs::default(), profile)
    }

    /// Make sure the scratch workspace holds at least `bytes` (0 = none needed).
    pub fn reserve_scratch(&mut self, bytes: usize) -> Result<()> {
        if bytes == 0 || self.scratch.as_ref().is_some_and(|b| b.bytes() >= bytes) {
            return Ok(());
        }
        self.graphs.invalidate(); // a captured graph holds the old scratch address
        self.scratch = None; // free the old one first
        self.scratch = Some(self.backend.alloc(bytes.max(16), MemKind::Device)?);
        self.plan.workspace_bytes += bytes as u64;
        Ok(())
    }

    fn scratch_view(&self, need: usize) -> Result<TensorRef> {
        match &self.scratch {
            Some(b) if b.bytes() >= need => Ok(TensorRef::new(b.clone(), DType::U8, &[need])),
            Some(b) => Err(Error::Memory(format!("kernel needs {need} scratch bytes but the plan reserved {}", b.bytes()))),
            None => Err(Error::Memory(format!("kernel needs {need} scratch bytes but the plan reserved none"))),
        }
    }

    /// Block kinds of every layer in plan vocabulary ("attention", "dense_ffn", ...).
    pub fn layer_kinds(&self) -> Vec<Vec<&'static str>> {
        self.layers
            .iter()
            .map(|blocks| {
                blocks
                    .iter()
                    .map(|b| match b {
                        Block::Attn(_) => "attention",
                        Block::Gdn(_) => "gated_delta_net",
                        Block::Ffn(_) => "dense_ffn",
                        Block::Moe(_) => "moe",
                        Block::Gemma(_) => "gemma4_layer",
                    })
                    .collect()
            })
            .collect()
    }

    /// The slots a sequence state of this model holds (rank 0's; `crate::prefix_store` lays its entries out by them).
    pub fn slot_specs(&self) -> &[SlotSpec] {
        &self.slot_specs
    }

    /// Whether this model runs tensor-parallel (local peers or a remote rank).
    pub fn is_tp(&self) -> bool {
        !self.peers.is_empty() || self.remote.is_some()
    }

    /// Allocate a fresh sequence state with the planned KV capacity.
    pub fn new_state(&self) -> Result<SequenceState> {
        let mut st = SequenceState::new(&*self.backend, &self.slot_specs, self.plan.kv_cap_tokens)?;
        if self.batch_rows > 0 {
            // a batching engine's slots hold their recurrent snapshots from the start (reserved in `enable_batch`'s memory
            // plan): no device allocation inside the scheduler loop
            st.reserve_snapshots(&*self.backend, self.batch_snapshots())?;
        }
        st.mtp = self.new_mtp_state(self.plan.kv_cap_tokens)?;
        st.dflash = self.new_dflash_state(self.plan.kv_cap_tokens)?;
        if st.mtp.is_some() && st.has_recurrent() {
            // allocated once here (and reserved in the memory plan), not in the first verify pass
            let need = st.recurrent_bytes().max(16);
            for _ in 0..self.mtp_k {
                st.spec_rows.bufs.push(self.backend.alloc(need, MemKind::Device)?);
            }
        }
        for p in &self.peers {
            let mut ps = SequenceState::new(&*p.backend, &p.slot_specs, self.plan.kv_cap_tokens)?;
            if p.batch_rows > 0 {
                // a batched tensor-parallel slot: the peer's share of the recurrent snapshots, reserved by its own enable_batch
                ps.reserve_snapshots(&*p.backend, p.batch_snapshots())?;
            }
            st.peers.push(PeerState { backend: p.backend.clone(), st: ps });
        }
        if let Some(r) = &self.remote {
            st.remote = Some(r.new_state()?);
        }
        Ok(st)
    }

    /// Make this rank-0 shard of a 2-way group drive its other rank in another process (`crate::tp_remote`): its
    /// communicator must already connect the two processes, and the other process runs [`crate::tp_remote::serve`] on
    /// the other end of `peer`'s connection.
    pub fn attach_remote(&mut self, peer: Arc<crate::tp_remote::RemotePeer>) -> Result<()> {
        if self.tp.rank != 0 || self.tp.world != 2 || !self.peers.is_empty() || self.tp.comm.is_none() {
            return Err(Error::invalid("a remote rank attaches to rank 0 of a 2-way group with a communicator and no local peers"));
        }
        if self.vocab_par {
            return Err(Error::Unsupported("a vocab-parallel head with a rank in another process (INTARSIA_TP_VOCAB=0)".into()));
        }
        self.remote = Some(peer);
        Ok(())
    }

    /// Every rank must hold the same number of tokens: clamp this shard's KV capacity to `cap` (the smallest any rank of
    /// the group can hold, agreed in `crate::tp_remote::handshake`) and re-resolve the plan for it.
    pub fn clamp_kv_cap(&mut self, cap: usize) -> Result<()> {
        if self.plan.kv_cap_tokens > cap {
            self.plan.notes.push(format!("context clamped from {} to {cap} tokens: the smallest rank decides", self.plan.kv_cap_tokens));
            self.plan.kv_cap_tokens = cap;
            self.plan.kv_total_bytes = cap as u64 * self.plan.kv_bytes_per_token;
            crate::plan::resolve(self)?;
        }
        Ok(())
    }

    /// A non-primary rank's forward pass over `ids` (the layers, no head), for a rank that runs in its own process
    /// (`crate::tp_remote::serve`). `graph`: rank 0 replays its decode graph in this step, so this rank does too. Same
    /// checks and failure reset as [`Model::forward`].
    pub fn forward_rank(&self, st: &mut SequenceState, ids: &[u32], graph: bool) -> Result<()> {
        Self::check_forward(st, ids, HeadMode::None, false)?;
        if self.vocab_par {
            return Err(Error::Unsupported("a vocab-parallel head with a rank in another process (INTARSIA_TP_VOCAB=0)".into()));
        }
        let r = self.run_peer(st, ids, HeadMode::None, graph);
        if r.is_err() && st.has_recurrent() {
            let _ = st.reset(&*self.backend);
        }
        r
    }

    /// Dtype of the row-parallel partial results the collective sums ("f32" or "bf16"); None without tensor parallelism.
    pub fn tp_partials(&self) -> Option<&'static str> {
        (self.tp.world > 1).then(|| if self.ws.part.is_some() { "f32" } else { "bf16" })
    }

    /// Whether the lm_head is split over the ranks by vocabulary rows (`LoadOptions::tp_vocab`, and possible here).
    pub fn vocab_parallel(&self) -> bool {
        self.vocab_par
    }

    /// Tensor-parallel degree (1 without it).
    pub fn tp_world(&self) -> usize {
        self.tp.world
    }

    /// Every rank's backend, rank 0 first.
    pub fn rank_backends(&self) -> Vec<Arc<dyn Backend>> {
        std::iter::once(self.backend.clone()).chain(self.peers.iter().map(|p| p.backend.clone())).collect()
    }

    /// Rank `r`'s shard (rank 0 is `self`).
    pub fn rank_model(&self, r: usize) -> Option<&Model> {
        if r == 0 {
            Some(self)
        } else {
            self.peers.get(r - 1)
        }
    }

    /// Collective counters of rank 0 (None without tensor parallelism).
    pub fn comm_stats(&self) -> Option<crate::comm::CommStats> {
        self.tp.comm.as_ref().map(|c| c.stats())
    }

    /// Name of the collective implementation ("host", "hip-p2p", ...), None without tensor parallelism.
    pub fn comm_name(&self) -> Option<String> {
        self.tp.comm.as_ref().map(|c| c.name().to_string())
    }

    /// Assemble a tensor-parallel model from its loaded shards (`shards[r]` is rank `r`): the KV capacity becomes
    /// the smallest any rank can hold (every rank must keep the same number of tokens), the peers follow rank 0's
    /// kernel selection, and the plan is re-resolved for the common capacity.
    pub fn from_shards(mut shards: Vec<Model>) -> Result<Model> {
        if shards.is_empty() {
            return Err(Error::invalid("a tensor-parallel group needs at least one shard"));
        }
        let world = shards[0].tp.world;
        if shards.len() != world || shards.iter().enumerate().any(|(i, m)| m.tp.rank != i || m.tp.world != world) {
            return Err(Error::invalid(format!("expected {world} shards in rank order")));
        }
        if shards.iter().any(|m| m.vocab_par != shards[0].vocab_par) {
            return Err(Error::invalid("the ranks of a tensor-parallel group disagree on a vocab-parallel head"));
        }
        let cap = shards.iter().map(|m| m.plan.kv_cap_tokens).min().unwrap();
        for m in shards.iter_mut() {
            if m.plan.kv_cap_tokens > cap {
                m.plan.notes.push(format!("context clamped from {} to {cap} tokens: the smallest rank decides", m.plan.kv_cap_tokens));
                m.plan.kv_cap_tokens = cap;
                m.plan.kv_total_bytes = cap as u64 * m.plan.kv_bytes_per_token;
            }
        }
        let mut it = shards.into_iter();
        let mut primary = it.next().unwrap();
        primary.peers = it.collect();
        for p in primary.peers.iter_mut() {
            p.registry.adopt_status(&primary.registry);
        }
        crate::plan::resolve(&mut primary)?;
        for i in 0..primary.peers.len() {
            let p = &mut primary.peers[i];
            crate::plan::resolve(p)?;
        }
        Ok(primary)
    }

    /// Load a `backends.len()`-way tensor-parallel model: rank `r` on `backends[r]`, talking through `comms[r]`.
    /// Shards load one after the other (they share the admission cache and read the same mmap).
    pub fn load_tp(
        backends: Vec<Arc<dyn Backend>>,
        comms: Vec<Arc<dyn Comm>>,
        recipe: &Recipe,
        dims: &ModelDims,
        src: &dyn WeightSource,
        opts: &LoadOptions,
        telemetry: Arc<Telemetry>,
    ) -> Result<Model> {
        let world = backends.len();
        if world < 2 || comms.len() != world {
            return Err(Error::invalid(format!("tensor parallelism needs >= 2 backends and one comm each (got {world} / {})", comms.len())));
        }
        let mut shards = Vec::new();
        for (rank, (backend, comm)) in backends.into_iter().zip(comms).enumerate() {
            if comm.rank() != rank || comm.world() != world {
                return Err(Error::invalid(format!("comm {rank} is rank {} of {}", comm.rank(), comm.world())));
            }
            // peers keep their own counters: the primary's telemetry is the one the server reports
            let tel = if rank == 0 { telemetry.clone() } else { Arc::new(Telemetry::new()) };
            shards.push(Self::load_shard(backend, recipe, dims, src, opts, tel, TpRank { rank, world, comm: Some(comm) })?);
        }
        Self::from_shards(shards)
    }

    /// Run `ids` through the model starting at `st.len`, appending to the state.
    ///
    /// `on_row` is required for [`HeadMode::AllLogits`]: it receives
    /// `(absolute position, logits row)` in order.
    ///
    /// A failed forward on a model with recurrent state resets the sequence: the recurrent slots are
    /// updated in place chunk by chunk, so a pass that died halfway leaves state that matches no token prefix.
    pub fn forward(
        &self,
        st: &mut SequenceState,
        ids: &[u32],
        mode: HeadMode,
        on_row: Option<RowCallback<'_>>,
    ) -> Result<StepOut> {
        let r = if self.remote.is_some() {
            self.forward_remote(st, ids, mode, on_row)
        } else if self.peers.is_empty() {
            self.forward_inner(st, ids, mode, on_row)
        } else {
            self.forward_tp(st, ids, mode, on_row)
        };
        if r.is_err() && st.has_recurrent() {
            let _ = st.reset(&*self.backend);
        }
        r
    }

    /// Fewest rows a segment of a packed prefill pass may have: below it the kernels a prompt's solo chunk picks (the GEMV-style
    /// MoE path under 16 tokens, the dense GEMV up to 4 rows) are not the ones the packed call picks.
    pub const PACK_MIN_ROWS: usize = 16;

    /// Whether every segment of a packed pass takes the same attention path in every attention block (fused gated prologue or
    /// not: it decides which q buffer holds the block's output, and the output projection reads one buffer for all rows).
    /// `items`: `(state, rows)` per segment. Evaluated before anything launches, so a `false` costs nothing.
    pub fn pack_uniform(&self, items: &[(&SequenceState, usize)]) -> bool {
        let at: Vec<(&SequenceState, usize, usize)> = items.iter().map(|&(st, n)| (st, st.len, n)).collect();
        self.pack_uniform_at(&at)
    }

    /// [`Model::pack_uniform`] with each segment's first position given: `(state, pos, rows)`.
    fn pack_uniform_at(&self, items: &[(&SequenceState, usize, usize)]) -> bool {
        let total: usize = items.iter().map(|&(_, _, n)| n).sum();
        let bufs = self.attn_bufs(total);
        for a in self.layers.iter().flatten().filter_map(|b| if let Block::Attn(a) = b { Some(a) } else { None }) {
            let (mut lo, mut first) = (0usize, None);
            for &(st, pos, len) in items {
                let Ok(slot) = Self::attn_slot(st, a) else { return false };
                let bj = bufs.rows(lo, len);
                let fused = self.gate_qkv_call_at(a, slot, &bj, len, total, lo, pos, None).filter(|c| self.fused_ok(c)).is_some();
                if *first.get_or_insert(fused) != fused {
                    return false;
                }
                lo += len;
            }
        }
        true
    }

    /// Most rows one packed pass holds: the workspace's, and no more than the KV capacity: plan resolution sizes the kernel scratch
    /// for chunks that fit the context, and a packed pass of several prompts must not need more than a solo chunk could.
    pub fn pack_rows_max(&self) -> usize {
        self.ws.max_tokens.min(self.plan.kv_cap_tokens.max(1))
    }

    /// Whether this model can run [`Model::forward_packed`]: a single rank, no Gemma layers, no plan recording.
    pub fn pack_capable(&self) -> bool {
        self.remote.is_none()
            && self.peers.is_empty()
            && self.tp.world == 1
            && self.gemma.is_none()
            && self.recorder.lock().unwrap().is_none()
            && self.rope_table.is_none()
    }

    /// Prefill the next chunks of several prompts in ONE pass: `items` are `(state, ids)`, each `ids` the whole chunk the
    /// prompt's solo prefill would run next (no head: the rows feed no logits). The row-wise kernels run once over the
    /// concatenated rows, attention, KV append and the GatedDeltaNet recurrence once per prompt on its own state, so each
    /// state ends bitwise what `forward(st, ids, HeadMode::None, None)` leaves, when every row-wise kernel is independent of the
    /// other rows of the call (`tests/packed_rows.rs`; each segment has at least [`Model::PACK_MIN_ROWS`] rows and the total
    /// fits one workspace pass). On an error every recurrent state is reset, as after a failed solo forward.
    pub fn forward_packed(&self, items: &mut [(&mut SequenceState, &[u32])]) -> Result<()> {
        let r = self.forward_packed_inner(items);
        if r.is_err() {
            for (st, _) in items.iter_mut() {
                if st.has_recurrent() {
                    let _ = st.reset(&*self.backend);
                }
            }
        }
        r
    }

    fn forward_packed_inner(&self, items: &mut [(&mut SequenceState, &[u32])]) -> Result<()> {
        if !self.pack_capable() {
            return Err(Error::Unsupported("packed prefill needs a single-rank model without Gemma layers".into()));
        }
        let total: usize = items.iter().map(|(_, ids)| ids.len()).sum();
        if items.is_empty() || total > self.pack_rows_max() {
            return Err(Error::invalid(format!("packed prefill of {} prompts, {total} rows (at most {})", items.len(), self.pack_rows_max())));
        }
        let mut all: Vec<u32> = Vec::with_capacity(total);
        for (st, ids) in items.iter() {
            Self::check_forward(st, ids, HeadMode::None, false)?;
            if ids.len() < Self::PACK_MIN_ROWS {
                return Err(Error::invalid(format!("a packed segment has {} rows (at least {})", ids.len(), Self::PACK_MIN_ROWS)));
            }
            if st.mtp.is_some() {
                return Err(Error::Unsupported("packed prefill of a state with an MTP head".into()));
            }
            all.extend_from_slice(ids);
        }
        for (st, _) in items.iter_mut() {
            st.spec_rows.saved = 0;
        }
        let profile = self.profile_every > 0 && self.telemetry.steps_total() % self.profile_every as u64 == 0;
        {
            let mut lo = 0;
            let segs: Vec<Seg<'_>> = items
                .iter()
                .map(|(st, ids)| {
                    let seg = Seg { st: &**st, lo, len: ids.len(), pos: st.len };
                    lo += ids.len();
                    seg
                })
                .collect();
            self.tail_dead.store(false, std::sync::atomic::Ordering::Relaxed);
            self.run_layers(Seqs::Packed(&segs), &all, profile, None, true, None)?;
        }
        for (st, ids) in items.iter_mut() {
            st.commit(ids);
        }
        self.telemetry.step();
        Ok(())
    }

    /// Rejections that must happen before any rank starts (a rank that stops early would leave the others waiting
    /// in a collective).
    fn check_forward(st: &SequenceState, ids: &[u32], mode: HeadMode, has_row_cb: bool) -> Result<()> {
        if ids.is_empty() {
            return Err(Error::BadRequest("empty token list".into()));
        }
        if st.len + ids.len() > st.cap {
            return Err(Error::BadRequest(format!(
                "sequence of {} tokens exceeds the planned context of {}",
                st.len + ids.len(),
                st.cap
            )));
        }
        if matches!(mode, HeadMode::LastTopK(_)) {
            return Err(Error::invalid("LastTopK is a batched-decode head (Model::forward_batch)"));
        }
        if mode == HeadMode::AllLogits && !has_row_cb {
            return Err(Error::invalid("AllLogits needs a row callback"));
        }
        Ok(())
    }

    /// Tensor-parallel forward: ranks 1.. run the layers on their own threads (no head), rank 0 runs the whole
    /// pass here and returns the result. A rank that fails aborts the communicator so no other rank is left
    /// waiting.
    fn forward_tp(
        &self,
        st: &mut SequenceState,
        ids: &[u32],
        mode: HeadMode,
        on_row: Option<RowCallback<'_>>,
    ) -> Result<StepOut> {
        if st.peers.len() != self.peers.len() {
            return Err(Error::invalid("this sequence state does not belong to a tensor-parallel model (use Model::new_state)"));
        }
        Self::check_forward(st, ids, mode, on_row.is_some())?;
        // peers follow the primary's length: a pass that failed halfway can leave them ahead of it
        for p in st.peers.iter_mut() {
            if p.st.len < st.len {
                return Err(Error::backend(format!("tensor-parallel ranks diverged ({} vs {} tokens)", p.st.len, st.len)));
            }
            if p.st.len > st.len {
                p.st.truncate(st.len, &*p.backend)?;
            }
        }
        let comm0 = self.tp.comm.clone().ok_or_else(|| Error::invalid("tensor-parallel model without a communicator"))?;
        let graph = self.decode_graph_now(st, ids, mode);
        let mut peers = std::mem::take(&mut st.peers);
        let out = std::thread::scope(|sc| {
            let handles: Vec<_> = self
                .peers
                .iter()
                .zip(peers.iter_mut())
                .map(|(m, ps)| {
                    sc.spawn(move || {
                        let comm = m.tp.comm.clone();
                        // a panic must release the other ranks too
                        struct Guard(Option<Arc<dyn Comm>>, bool);
                        impl Drop for Guard {
                            fn drop(&mut self) {
                                if !self.1 {
                                    if let Some(c) = &self.0 {
                                        c.abort();
                                    }
                                }
                            }
                        }
                        let mut guard = Guard(comm, false);
                        let r = m.run_peer(&mut ps.st, ids, mode, graph);
                        guard.1 = r.is_ok();
                        r
                    })
                })
                .collect();
            let r0 = self.forward_inner(st, ids, mode, on_row);
            if r0.is_err() {
                comm0.abort();
            }
            let mut res = r0;
            for h in handles {
                match h.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        if res.is_ok() {
                            res = Err(e);
                        }
                    }
                    Err(_) => {
                        if res.is_ok() {
                            res = Err(Error::backend("a tensor-parallel rank panicked"));
                        }
                    }
                }
            }
            res
        });
        st.peers = peers;
        out
    }

    /// Tensor-parallel forward with the other rank in another process: send it the pass, run rank 0's here, then read its
    /// answer. A failure on either side leaves the group unusable (the communicator is aborted).
    fn forward_remote(
        &self,
        st: &mut SequenceState,
        ids: &[u32],
        mode: HeadMode,
        on_row: Option<RowCallback<'_>>,
    ) -> Result<StepOut> {
        Self::check_forward(st, ids, mode, on_row.is_some())?;
        let mut rs = st
            .remote
            .take()
            .ok_or_else(|| Error::invalid("this sequence state does not belong to a tensor-parallel model (use Model::new_state)"))?;
        let r = (|| {
            if rs.len > st.len {
                rs.truncate(st.len)?;
            } else if rs.len < st.len {
                return Err(Error::backend(format!("tensor-parallel ranks diverged ({} vs {} tokens)", rs.len, st.len)));
            }
            rs.start_forward(ids, self.decode_graph_now(st, ids, mode))?;
            let r0 = self.forward_inner(st, ids, mode, on_row);
            if r0.is_err() {
                // the remote rank waits in a collective this rank will not join: the group is gone
                if let Some(c) = &self.tp.comm {
                    c.abort();
                }
                rs.mark_unusable();
                return r0;
            }
            rs.finish()?;
            r0
        })();
        st.remote = Some(rs);
        r
    }

    /// `ids` cut into forward passes of at most `max_tokens` rows, by the engine's prefill rule
    /// ([`crate::scheduler::next_prefill_chunk`]): on a backend that avoids a 1-row tail (Metal) a `max_tokens + 1`
    /// remainder runs as `(max_tokens - 1, 2)`, so accuracy tooling scores the bits serving produces. Recurrent states keep
    /// the plain split, as in the engine. With the prefill tail peel ([`Model::prefill_peel`]) a pass whose rows mod 256
    /// are `1..=peel` runs those rows as their own pass ([`crate::scheduler::peel_tail`]).
    fn forward_chunks<'a>(&self, st: &SequenceState, ids: &'a [u32]) -> Vec<&'a [u32]> {
        let avoid = self.backend.avoid_single_row_prefill() && !st.has_recurrent();
        let mut out = Vec::with_capacity(ids.len().div_ceil(self.ws.max_tokens));
        let mut pos = 0;
        while pos < ids.len() {
            let n = crate::scheduler::next_prefill_chunk_tm(ids.len() - pos, self.ws.max_tokens, avoid, self.tail_merge);
            let n = crate::scheduler::peel_tail(n, self.prefill_peel);
            out.push(&ids[pos..pos + n]);
            pos += n;
        }
        out
    }

    /// A peer rank's share of a forward pass: the layers only (rank 0 owns the head), plus, with a vocab-parallel head, its
    /// slice of the head and the gather of every head group rank 0 runs in `mode` (`Model::head_groups`: the collectives
    /// must be the same sequence on every rank). `graph`: rank 0 replays its decode graph in this step
    /// (`Model::decode_graph_now`), so this rank replays its own.
    /// Debugging (`INTARSIA_TP_SKEW_US=<rank 0 us>,<rank 1 us>`, off by default): delay this rank's launch of every step by that long, to
    /// skew the ranks against each other (a handshake that is only right when the ranks arrive together fails under skew).
    pub(super) fn tp_skew(&self) {
        static SKEW: std::sync::OnceLock<[u64; 2]> = std::sync::OnceLock::new();
        let s = SKEW.get_or_init(|| {
            let v = std::env::var("INTARSIA_TP_SKEW_US").unwrap_or_default();
            let mut it = v.split(',').map(|x| x.trim().parse::<u64>().unwrap_or(0));
            [it.next().unwrap_or(0), it.next().unwrap_or(0)]
        });
        if self.tp.world == 2 {
            let us = s[self.tp.rank.min(1)];
            if us > 0 {
                std::thread::sleep(std::time::Duration::from_micros(us));
            }
        }
    }

    /// `INTARSIA_TP_PEER_SYNC=1` (w27 experiment, off by default): a peer rank's step ends with its stream idle. A peer only enqueues its share
    /// (the next collective or download of rank 0 orders it), so its last replay can still be running when its next step starts.
    pub(super) fn peer_sync(&self) -> Result<()> {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ON.get_or_init(|| std::env::var("INTARSIA_TP_PEER_SYNC").is_ok_and(|v| v == "1")) {
            self.backend.synchronize(self.stream)?;
        }
        Ok(())
    }

    fn run_peer(&self, ps: &mut SequenceState, ids: &[u32], mode: HeadMode, graph: bool) -> Result<()> {
        self.run_peer_inner(ps, ids, mode, graph)?;
        self.peer_sync()
    }

    fn run_peer_inner(&self, ps: &mut SequenceState, ids: &[u32], mode: HeadMode, graph: bool) -> Result<()> {
        self.tp_skew();
        if graph {
            if ids.len() != 1 {
                return Err(Error::invalid(format!("decode graph step with {} tokens", ids.len())));
            }
            self.graph_peer_step(ps, ids[0])?;
            ps.commit(ids);
            return Ok(());
        }
        let chunks = self.forward_chunks(ps, ids);
        let n = chunks.len();
        for (ci, chunk) in chunks.into_iter().enumerate() {
            let hn = self.run_chunk(ps, chunk, false)?;
            if self.vocab_par {
                for (r0, c) in self.head_groups(mode, chunk.len(), ci + 1 == n) {
                    self.head_peer(&hn, r0, c)?;
                }
            }
            ps.commit(chunk);
        }
        Ok(())
    }

    fn forward_inner(
        &self,
        st: &mut SequenceState,
        ids: &[u32],
        mode: HeadMode,
        mut on_row: Option<RowCallback<'_>>,
    ) -> Result<StepOut> {
        Self::check_forward(st, ids, mode, on_row.is_some())?;
        self.tp_skew();
        // a plain pass ends the window in which the last verify pass's rows could be rolled back to
        st.spec_rows.saved = 0;
        // single-token decode: replay the captured step (profiled steps and unsupported backends run eagerly)
        if self.decode_graph_now(st, ids, mode) {
            if let Some(out) = self.graph_decode_step(st, ids[0], mode == HeadMode::LastLogits)? {
                st.commit(ids);
                self.telemetry.step();
                if st.mtp.is_some() {
                    // the captured step leaves its normed hidden row in ws.n, as the eager pass does
                    let hn = self.view(&self.ws.n, self.act, &[1, self.dims.hidden]);
                    self.mtp_after_forward(st, &hn, 1, true, false)?;
                }
                self.dflash_after_forward(st, 1, false)?;
                return Ok(out);
            }
        }
        let chunks = self.forward_chunks(st, ids);
        let nchunks = chunks.len();
        let mut out = StepOut::None;
        // the opt-in macro schedule: runs of consecutive chunks (the same chunks, in the same order) as ONE layer-by-layer pass
        let (macro_rows_n, per) = self.macro_plan(st, &chunks, mode);
        let mut ci = 0usize;
        while ci < nchunks {
            let mut glen = if ci < macro_rows_n { per.min(macro_rows_n - ci) } else { 1 };
            if glen > 1 {
                let group = &chunks[ci..ci + glen];
                let last = ci + glen == nchunks;
                let profile = self.profile_every > 0 && self.telemetry.steps_total() % self.profile_every as u64 == 0;
                if let Some(hn) = self.run_macro_group(st, group, profile)? {
                    let total: usize = group.iter().map(|c| c.len()).sum();
                    match mode {
                        HeadMode::LastLogits if last => out = StepOut::Logits(self.head_logits(&hn, total - 1, 1, profile)?),
                        HeadMode::LastArgmax if last => out = StepOut::Token(self.head_argmax(&hn, total - 1, profile)?),
                        _ => {}
                    }
                    for c in group {
                        st.commit(c);
                        self.telemetry.step();
                    }
                    ci += glen;
                    continue;
                }
                glen = 1; // the segments take different attention paths: chunk by chunk, as without the macro schedule
            }
            debug_assert_eq!(glen, 1);
            let chunk = chunks[ci];
            let last = ci + 1 == nchunks;
            ci += 1;
            let base = st.len;
            let profile = self.profile_every > 0 && self.telemetry.steps_total() % self.profile_every as u64 == 0;
            if ids.len() == 1 && matches!(mode, HeadMode::LastLogits | HeadMode::LastArgmax) {
                self.telemetry.decode_eager_step(profile);
            }
            // a chunk whose rows feed no head: the last layer's work after its cache writes is dead (opt-in)
            let feeds_head = match mode {
                HeadMode::None => false,
                HeadMode::LastLogits | HeadMode::LastArgmax => last,
                HeadMode::AllLogits => true,
                HeadMode::LastTopK(_) => false, // rejected by check_forward
            };
            let dead = !feeds_head && self.tp.world == 1 && self.skip_dead.load(std::sync::atomic::Ordering::Relaxed);
            self.tail_dead.store(dead, std::sync::atomic::Ordering::Relaxed);
            let hn = self.run_chunk(st, chunk, profile);
            self.tail_dead.store(false, std::sync::atomic::Ordering::Relaxed);
            let hn = hn?;
            match mode {
                HeadMode::None => {}
                HeadMode::LastLogits if last => {
                    out = StepOut::Logits(self.head_logits(&hn, chunk.len() - 1, 1, profile)?);
                }
                HeadMode::LastArgmax if last => {
                    out = StepOut::Token(self.head_argmax(&hn, chunk.len() - 1, profile)?);
                }
                HeadMode::AllLogits => {
                    let cb = on_row.as_mut().unwrap();
                    for (r, c) in self.head_groups(mode, chunk.len(), last) {
                        let flat = self.head_logits(&hn, r, c, profile)?;
                        for i in 0..c {
                            cb(base + r + i, &flat[i * self.dims.vocab..(i + 1) * self.dims.vocab])?;
                        }
                    }
                }
                _ => {}
            }
            st.commit(chunk);
            self.telemetry.step();
            if st.mtp.is_some() {
                self.mtp_after_forward(st, &hn, chunk.len(), true, profile)?;
            }
            self.dflash_after_forward(st, chunk.len(), profile)?;
        }
        Ok(out)
    }

    /// The macro schedule of a forward pass over `chunks`: `(k, per)` = the first `k` chunks (all but a trailing chunk too short
    /// to be a segment, [`Model::PACK_MIN_ROWS`]) are cut into passes of `per` chunks. `(0, 1)` = off: the macro knob is unset, the
    /// pass is a single chunk, the head wants every row, the state has no recurrence (the packed kernels are validated on
    /// hybrid models only) or cannot be packed (MTP head, tensor parallelism, Gemma layers).
    fn macro_plan(&self, st: &SequenceState, chunks: &[&[u32]], mode: HeadMode) -> (usize, usize) {
        if self.macro_rows == 0
            || chunks.len() < 2
            || !matches!(mode, HeadMode::None | HeadMode::LastLogits | HeadMode::LastArgmax)
            || !st.has_recurrent()
            || st.mtp.is_some()
            || !self.pack_capable()
        {
            return (0, 1);
        }
        let k = chunks.iter().take_while(|c| c.len() >= Self::PACK_MIN_ROWS).count();
        if k < 2 {
            return (0, 1);
        }
        (k, crate::scheduler::macro_group_len(k, self.macro_rows / self.ws.max_tokens.max(1)))
    }

    /// Run `group` (consecutive chunks of one sequence, `st.len` at the first) as one packed pass: the row-wise ops over all the
    /// rows, attention, KV append and the recurrence once per chunk, in order, on the one state (`Seg::pos` is each chunk's
    /// position). Returns the normed hidden rows of the whole pass, or `None` (nothing launched) when the chunks would take
    /// different attention paths. The caller commits the chunks.
    fn run_macro_group(&self, st: &SequenceState, group: &[&[u32]], profile: bool) -> Result<Option<TensorRef>> {
        let mut lo = 0usize;
        let mut segs: Vec<Seg<'_>> = Vec::with_capacity(group.len());
        for c in group {
            segs.push(Seg { st, lo, len: c.len(), pos: st.len + lo });
            lo += c.len();
        }
        let items: Vec<(&SequenceState, usize, usize)> = segs.iter().map(|s| (s.st, s.pos, s.len)).collect();
        if !self.pack_uniform_at(&items) {
            return Ok(None);
        }
        let all: Vec<u32> = group.iter().flat_map(|c| c.iter().copied()).collect();
        self.tail_dead.store(false, std::sync::atomic::Ordering::Relaxed);
        self.macro_passes.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        MACRO_PASSES_TOTAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        *self.macro_ctx.lock().unwrap() = Some(segs.iter().map(|s| (s.lo, s.len)).collect());
        let r = self.run_layers(Seqs::Packed(&segs), &all, profile, None, true, None);
        *self.macro_ctx.lock().unwrap() = None;
        r.map(Some)
    }

    fn view(&self, b: &Buffer, dtype: DType, shape: &[usize]) -> TensorRef {
        TensorRef::new(b.clone(), dtype, shape)
    }

    fn exec(
        &self,
        op: Op,
        inputs: Vec<TensorRef>,
        outputs: Vec<TensorRef>,
        attrs: OpAttrs,
        profile: bool,
    ) -> Result<()> {
        self.exec_pos(op, inputs, outputs, attrs, None, profile)
    }

    /// `exec` with an optional device-side position (`OpCall::pos`) for position-dependent decode ops.
    fn exec_pos(
        &self,
        op: Op,
        inputs: Vec<TensorRef>,
        outputs: Vec<TensorRef>,
        attrs: OpAttrs,
        pos: Option<&TensorRef>,
        profile: bool,
    ) -> Result<()> {
        let mut call = OpCall::new(op, inputs, outputs, attrs);
        call.pos = pos.cloned();
        self.exec_call(call, profile)
    }

    /// [`Model::exec_call_one`], except that inside a macro pass ([`Model::run_macro_group`]) a call of one of
    /// [`Model::macro_split`] over all the pass's rows runs once per chunk, on that chunk's rows of the activation and the outputs.
    fn exec_call(&self, call: OpCall, profile: bool) -> Result<()> {
        if self.macro_split.contains(&call.op) {
            let segs = self.macro_ctx.lock().unwrap().clone();
            if let Some(segs) = segs {
                let total: usize = segs.iter().map(|s| s.1).sum();
                if call.inputs.first().is_some_and(|x| x.shape.first() == Some(&total)) {
                    for (lo, len) in segs {
                        let mut part = call.clone();
                        part.inputs[0] = call.inputs[0].rows(lo, len);
                        for (o, p) in call.outputs.iter().zip(part.outputs.iter_mut()) {
                            if o.shape.first() == Some(&total) {
                                *p = o.rows(lo, len);
                            }
                        }
                        self.exec_call_one(part, profile)?;
                    }
                    return Ok(());
                }
            }
        }
        self.exec_call_one(call, profile)
    }

    /// Run (or, in a plan dry run, record) a prepared call: `exact_rows` follows the pass, the scratch is the plan's.
    fn exec_call_one(&self, mut call: OpCall, profile: bool) -> Result<()> {
        let op = call.op;
        call.exact_rows = self.verifying();
        call.check()?;
        // a block-end all-reduce held back for this op (`model/ar_norm.rs`): fused with it, or issued before it
        if self.tp_arnorm && self.resolve_ar(&call, profile)? {
            return Ok(());
        }
        {
            let mut rec = self.recorder.lock().unwrap();
            if let Some(r) = rec.as_mut() {
                match self.registry.select(&call) {
                    Ok(k) => r.calls.push(crate::plan::PlannedCall {
                        phase: r.phase.unwrap_or(crate::plan::Phase::Decode),
                        t: r.t,
                        pos0: r.pos0,
                        kernel: k.name.clone(),
                        call,
                        extra: r.extra,
                    }),
                    Err(e) => {
                        let s = ShapeInfo::from_call(&call);
                        r.missing.push((
                            call.op,
                            format!("{e} (heads {}/{} head_dim {} kv_len {})", s.n_heads, s.n_kv_heads, s.head_dim, s.kv_len),
                        ))
                    }
                }
                return Ok(());
            }
        }
        let kernel = self.registry.select(&call)?;
        if kernel.scratch.is_some() {
            let need = kernel.scratch_bytes(&ShapeInfo::from_call(&call));
            if need > 0 {
                call.scratch = Some(self.scratch_view(need)?);
            }
        }
        if profile {
            // drain queued work first, or the previous op's (or an async upload's) time lands on this op
            self.backend.synchronize(self.stream)?;
        }
        let t0 = Instant::now();
        self.backend.launch(&kernel.name, &call, self.stream)?;
        if profile {
            self.backend.synchronize(self.stream)?;
            self.telemetry.record_op(op, Some(t0.elapsed()));
        } else if let Some(ops) = self.capture_ops.lock().unwrap().as_mut() {
            ops.push(op); // being captured: nothing ran, each replay counts these
        } else {
            self.telemetry.record_op(op, None);
        }
        Ok(())
    }

    /// Returns the normed final hidden states `[T, H]`.
    fn run_chunk(&self, st: &SequenceState, ids: &[u32], profile: bool) -> Result<TensorRef> {
        self.run_chunk_at(st, ids, profile, None, true)
    }

    /// `run_chunk` with the decode-graph hooks: `dpos` is the device-side position (attention ops read it when
    /// they run, `attrs.pos0` is then 0) and `upload_ids = false` leaves `ws.ids` as the caller wrote it.
    fn run_chunk_at(
        &self,
        st: &SequenceState,
        ids: &[u32],
        profile: bool,
        dpos: Option<&TensorRef>,
        upload_ids: bool,
    ) -> Result<TensorRef> {
        self.run_layers(Seqs::One(st), ids, profile, dpos, upload_ids, None)
    }

    /// The layer loop behind [`Model::run_chunk_at`]: `ids` are the `t` tokens of one sequence (`Seqs::One`) or the
    /// next token of each of `t` sequences (`Seqs::Batch`, a batched decode step: attention runs per row on the row's
    /// own cache, see `model/batch.rs`). `ids_at` (token feedback, `model/feedback.rs`) makes the embedding read the
    /// ids from that device tensor instead of `ws.ids`; `upload_ids` must then be false.
    fn run_layers(
        &self,
        seqs: Seqs<'_>,
        ids: &[u32],
        profile: bool,
        dpos: Option<&TensorRef>,
        upload_ids: bool,
        ids_at: Option<&TensorRef>,
    ) -> Result<TensorRef> {
        let t = ids.len();
        let _pass = if self.coll_trace.on {
            self.coll_trace.pass(if matches!(seqs, Seqs::Batch(_)) { "layers-batch" } else { "layers" }, t, self.capture_ops.lock().unwrap().is_some())
        } else {
            None
        };
        let d = &self.dims;
        let act = self.act;
        let h = d.hidden;
        let pos0 = match seqs {
            Seqs::One(st) if dpos.is_none() => st.len,
            _ => 0,
        };
        let ws = &self.ws;

        if upload_ids && !self.recorder.lock().unwrap().is_some() {
            let ids_i32: Vec<u8> = ids.iter().flat_map(|&i| (i as i32).to_le_bytes()).collect();
            self.backend.upload(&ws.ids, 0, &ids_i32, self.stream)?;
            for &i in ids {
                if i as usize >= d.vocab {
                    return Err(Error::BadRequest(format!("token id {i} >= vocab {}", d.vocab)));
                }
            }
        }

        let mut cur = 0usize; // which h buffer holds the residual stream
        let hv = |i: usize| self.view(&ws.h[i], act, &[t, h]);
        let nv = self.view(&ws.n, act, &[t, h]);
        let blkv = self.view(&ws.blk, act, &[t, h]);

        self.exec(
            Op::Embedding,
            vec![ids_at.cloned().unwrap_or_else(|| self.view(&ws.ids, DType::I32, &[t])), self.embed.clone()],
            vec![hv(cur)],
            OpAttrs { scale: d.gemma.as_ref().map_or(1.0, |g| g.embed_scale), ..Default::default() },
            profile,
        )?;

        let eps_attrs = OpAttrs { eps: d.rms_eps, w_offset: self.recipe.model.norm_offset, ..Default::default() };
        let mut pending = false;
        let norm_in = |pre: &TensorRef, pending: bool, cur: &mut usize| -> Result<()> {
            if pending {
                let next = 1 - *cur;
                self.exec(
                    Op::RmsNormAdd,
                    vec![blkv.clone(), hv(*cur), pre.clone()],
                    vec![nv.clone(), hv(next)],
                    eps_attrs.clone(),
                    profile,
                )?;
                *cur = next;
            } else {
                self.exec(Op::RmsNorm, vec![hv(*cur), pre.clone()], vec![nv.clone()], eps_attrs.clone(), profile)?;
            }
            Ok(())
        };

        // The GatedDeltaNet and MoE blocks take the norm in front of them (`NormFold`) so that they can fold it into their
        // first GEMV pair; it flips the residual buffers the way `norm_in` does and the block runs the norm exactly once.
        let fold = |pre: &TensorRef, cur: &mut usize| -> NormFold {
            let next = 1 - *cur;
            let f = NormFold { blk: blkv.clone(), res: hv(*cur), w: pre.clone(), sum: hv(next), attrs: eps_attrs.clone() };
            *cur = next;
            f
        };

        // the dead tail (`Model::skip_dead`): one sequence, not a plan recording
        let dead_tail = matches!(seqs, Seqs::One(_))
            && dpos.is_none()
            && self.tail_dead.load(std::sync::atomic::Ordering::Relaxed)
            && !self.recorder.lock().unwrap().is_some();
        let n_layers = self.layers.len();
        for (li, blocks) in self.layers.iter().enumerate() {
            let tail = dead_tail && li + 1 == n_layers;
            for block in blocks {
                if tail && !matches!(block, Block::Attn(_) | Block::Gdn(_)) {
                    // the last layer's FFN / MoE feeds only the final norm and the head
                    continue;
                }
                match block {
                    Block::Gemma(g) => {
                        // a whole layer with its own residual handling: nothing is pending before or after it
                        let Seqs::One(st) = seqs else {
                            return Err(Error::Unsupported("batched decode of a Gemma 4 layer".into()));
                        };
                        if dpos.is_some() {
                            return Err(Error::Unsupported("captured decode steps of a Gemma 4 layer".into()));
                        }
                        debug_assert!(!pending);
                        self.gemma_layer(g, st, &hv(cur), &hv(1 - cur), t, pos0, profile)?;
                        cur = 1 - cur;
                        continue;
                    }
                    Block::Attn(a) => {
                        // the residual add + norm folded into the k / v projections when the backend has the fused op (one
                        // decision for the solo and the batched step: the call is the same, `t` rows of `n`)
                        let kv_done = pending && self.attn_norm_kv(a, &blkv, &hv(cur), &hv(1 - cur), &nv, t, &eps_attrs, profile)?;
                        if kv_done {
                            cur = 1 - cur;
                        } else {
                            norm_in(&a.pre_norm, pending, &mut cur)?;
                        }
                        match seqs {
                            Seqs::One(st) if tail => self.attn_block_kv_only(a, st, &nv, t, pos0, kv_done, profile)?,
                            Seqs::One(st) => self.attn_block(a, st, &nv, &blkv, t, pos0, dpos, kv_done, profile)?,
                            Seqs::Batch(sts) => self.attn_block_batch(a, sts, &nv, &blkv, kv_done, profile)?,
                            Seqs::Packed(segs) => self.attn_block_packed(a, segs, &nv, &blkv, t, kv_done, profile)?,
                        }
                    }
                    Block::Gdn(g) => match seqs {
                        Seqs::Batch(sts) if self.recurrent_batch => {
                            // one decode token of each sequence: the projections run once on all rows, the conv + delta
                            // rule once per row on that row's own recurrent state (`model/gdn_batch.rs`)
                            let f = pending.then(|| fold(&g.pre_norm, &mut cur));
                            if f.is_none() {
                                norm_in(&g.pre_norm, false, &mut cur)?;
                            }
                            self.gdn_block_batch(g, sts, &nv, &blkv, profile, f.as_ref())?;
                        }
                        Seqs::Batch(_) => {
                            norm_in(&g.pre_norm, pending, &mut cur)?;
                            return Err(Error::Unsupported("batched decode of a recurrent block".into()));
                        }
                        Seqs::One(st) => {
                            let f = pending.then(|| fold(&g.pre_norm, &mut cur));
                            if f.is_none() {
                                norm_in(&g.pre_norm, false, &mut cur)?;
                            }
                            self.gdn_block(g, st, &nv, &blkv, t, profile, f.as_ref())?;
                        }
                        Seqs::Packed(segs) => {
                            let f = pending.then(|| fold(&g.pre_norm, &mut cur));
                            if f.is_none() {
                                norm_in(&g.pre_norm, false, &mut cur)?;
                            }
                            self.gdn_block_segs(g, segs, &nv, &blkv, t, profile, f.as_ref())?;
                        }
                    },
                    Block::Ffn(f) => {
                        norm_in(&f.pre_norm, pending, &mut cur)?;
                        self.ffn_block(f, &nv, &blkv, t, profile)?;
                    }
                    Block::Moe(m) => {
                        if matches!(seqs, Seqs::Batch(_)) && !self.recurrent_batch {
                            norm_in(&m.pre_norm, pending, &mut cur)?;
                            return Err(Error::Unsupported("batched decode of a mixture-of-experts block".into()));
                        }
                        let ws = self.ws.moe.as_ref().ok_or_else(|| Error::invalid("moe block without workspace"))?;
                        let f = pending.then(|| fold(&m.pre_norm, &mut cur));
                        if f.is_none() {
                            norm_in(&m.pre_norm, false, &mut cur)?;
                        }
                        m.w.forward_folded(&ExecRunner { m: self, profile }, ws, f.as_ref(), &nv, &blkv)?;
                        // tensor parallelism: the block output is this rank's partial (routed + gated shared expert, both
                        // linear in the rank's slice of the experts' width), summed in place in the activation dtype
                        // (the MoE kernels write it). A no-op without tensor parallelism.
                        self.all_reduce(&blkv, &blkv, profile)?;
                    }
                }
                pending = true;
            }
            // DFlash: this layer's output (residual stream plus the pending block output) is a draft input
            if let Some(head) = &self.dflash {
                if let Some(ti) = head.taps.iter().position(|&x| x == li) {
                    if !matches!(seqs, Seqs::One(_)) || !pending {
                        return Err(Error::Unsupported("--spec dflash feature taps outside a single-sequence pass".into()));
                    }
                    let out = self.view(&head.feat[ti], act, &[t, h]);
                    self.exec(Op::Add, vec![hv(cur), blkv.clone()], vec![out], OpAttrs::default(), profile)?;
                }
            }
        }
        if dead_tail {
            // nothing reads the hidden state of a chunk that feeds no head
            self.flush_ar(profile)?;
            return Ok(nv);
        }
        // final: h = h + block_out ; n = norm(h)
        norm_in(&self.final_norm, pending, &mut cur)?;
        self.flush_ar(profile)?;
        Ok(nv)
    }

    /// Run a fused decode op (`Op::MatmulPair` ... `Op::GatedRmsnorm`) when the registry has a kernel for this exact call,
    /// returning `true`; return `false` without running anything when it has none (another backend, an unsupported shape such
    /// as a prefill chunk, a kernel that failed admission, `INTARSIA_HIP_DISABLE`), and the caller runs the ops the fused one
    /// replaces. The probe goes through `Registry::select`, the lookup `exec` itself makes, so a plan recording sees only
    /// the calls that will run. Fusion is off with `INTARSIA_NO_FUSION=1`, and under tensor parallelism unless row-local
    /// fusion is on (`LoadOptions::tp_fusion`, see [`Model::fused_under_tp`]).
    fn try_fused(
        &self,
        op: Op,
        inputs: Vec<TensorRef>,
        outputs: Vec<TensorRef>,
        attrs: OpAttrs,
        profile: bool,
    ) -> Result<bool> {
        let call = OpCall::new(op, inputs, outputs, attrs);
        if !self.fused_ok(&call) {
            return Ok(false);
        }
        self.exec(call.op, call.inputs, call.outputs, call.attrs, profile)?;
        Ok(true)
    }

    /// Whether `try_fused` would run `call` (fusion enabled, call valid, a kernel selectable), without running it. For a
    /// fused op whose replaced ops are spread over the layer, so the caller must decide before it emits any of them.
    fn fused_ok(&self, call: &OpCall) -> bool {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if (self.tp.world > 1 && !self.fused_under_tp(call.op))
            || *OFF.get_or_init(|| std::env::var_os("INTARSIA_NO_FUSION").is_some_and(|v| v != "0"))
        {
            return false;
        }
        let select = |c: &OpCall| {
            if self.verifying() {
                let mut c = c.clone();
                c.exact_rows = true;
                return self.registry.select(&c).is_ok();
            }
            self.registry.select(c).is_ok()
        };
        if !call.check().is_ok() || !select(call) {
            return false;
        }
        // The batch-consistency rule (`plan::check_batch_ops`): a batched step must run the solo step's op multiset, so with
        // batching on a row-local fused op is taken only if a kernel is selectable at every row count a step can have; a
        // kernel registered for one row only (the HIP decode GEMVs) then falls back to the unfused ops in the solo step too,
        // instead of failing the plan load. Applies to the ops `OpCall::with_rows` knows (the phase-1 folds); the older
        // fused ops keep their behaviour.
        // (A batched pass marks its calls `exact_rows`, so the multi-row probes select the way it does.)
        // A fusion with a registered bitwise-equivalence entry (`crate::fusion`, e.g. HIP's folds, lane hb) needs no such probe:
        // the batched step may run its unfused chain, which the table proves is the same arithmetic, so the solo step keeps
        // the fused kernel at every batch size and a single stream does not pay for batching being on.
        if self.batch_rows >= 2 && crate::fusion::fusion(self.backend.name(), call.op).is_none() {
            for r in 2..=self.batch_rows {
                if let Some(mut c) = call.with_rows(r) {
                    c.exact_rows = true;
                    if !self.registry.select(&c).is_ok() {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Under tensor parallelism, may fused op `op` run on this rank's shard? Only with row-local fusion on, and never
    /// for an op that would replace a row-parallel projection together with what precedes it: its output is a partial
    /// that `proj_reduce` must land in the f32 partial buffer and all-reduce, and the fused kernel writes the block
    /// output directly (`GatedRmsnormMatmulFp8`, the GatedDeltaNet out-projection). `SigmoidMulMatmulFp8` (attention
    /// o_proj) is already tried only with bf16 partials, where the all-reduce follows it.
    fn fused_under_tp(&self, op: Op) -> bool {
        self.tp_fusion && !matches!(op, Op::GatedRmsnormMatmulFp8)
    }

    /// Inside [`Model::forward_verify`].
    fn verifying(&self) -> bool {
        self.exact_rows.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `out = x @ W^T` for a linear layer in any supported weight format.
    fn proj(&self, x: &TensorRef, w: &QTensor, out: &TensorRef, profile: bool) -> Result<()> {
        let call = w.call(x.clone(), out.clone());
        self.exec(call.op, call.inputs, call.outputs, call.attrs, profile)
    }

    /// Two MXFP4 projections of the same activation as one `Op::MatmulMxfp4Pair` launch, when the backend has the kernel.
    /// Only for two MXFP4 weights with the same `act_quant` (the pair carries one `act` input): anything else, or no kernel,
    /// returns `false` without running anything and the caller runs the two `proj`s.
    pub(crate) fn try_mxfp4_pair(
        &self,
        x: &TensorRef,
        a: &QTensor,
        b: &QTensor,
        out_a: &TensorRef,
        out_b: &TensorRef,
        profile: bool,
    ) -> Result<bool> {
        if a.format != crate::quant::MXFP4 || b.format != crate::quant::MXFP4 || a.act_quant != b.act_quant {
            return Ok(false);
        }
        self.try_fused(
            Op::MatmulMxfp4Pair,
            vec![x.clone(), a.parts[0].clone(), a.parts[1].clone(), b.parts[0].clone(), b.parts[1].clone(), a.parts[2].clone()],
            vec![out_a.clone(), out_b.clone()],
            OpAttrs::default(),
            profile,
        )
    }

    /// k and v read the same activation: one launch for both when the backend has the fused fp8 / bf16 / mxfp4 pair. Returns
    /// whether it ran (then `k_out` and `v_out` hold both projections); `false` leaves both to `proj`.
    fn kv_pair(&self, a: &AttnBlock, n: &TensorRef, k_out: &TensorRef, v_out: &TensorRef, profile: bool) -> Result<bool> {
        match (a.wk.format, a.wv.format) {
            (crate::quant::FP8_E4M3, crate::quant::FP8_E4M3) => self.try_fused(
                Op::MatmulFp8Pair,
                vec![n.clone(), a.wk.parts[0].clone(), a.wk.parts[1].clone(), a.wv.parts[0].clone(), a.wv.parts[1].clone()],
                vec![k_out.clone(), v_out.clone()],
                OpAttrs { fp8_static_act: a.wk.act_quant && a.wv.act_quant, ..OpAttrs::default() },
                profile,
            ),
            (crate::quant::BF16, crate::quant::BF16) => self.try_fused(
                Op::MatmulPair,
                vec![n.clone(), a.wk.parts[0].clone(), a.wv.parts[0].clone()],
                vec![k_out.clone(), v_out.clone()],
                OpAttrs::default(),
                profile,
            ),
            (crate::quant::MXFP4, crate::quant::MXFP4) => self.try_mxfp4_pair(n, &a.wk, &a.wv, k_out, v_out, profile),
            _ => Ok(false),
        }
    }

    /// The attention block's input norm (`RmsNormAdd(blk, res, pre_norm) -> (n, sum)`) folded into its fp8 k / v projections
    /// (`Op::RmsNormAddMatmulFp8Pair`, the fp8 twin of `RmsNormAddMatmulPair`): on `true` `n` and `sum` hold the norm's
    /// outputs and `ws.k[0]` / `ws.v` the k and v projections of the `t` rows of `n`, so the caller flips the residual
    /// buffers and skips the norm and the k / v projections. On `false` nothing ran (no fp8 k / v, no kernel for the call,
    /// fusion off) and the caller runs the norm, then the projections, as before.
    #[allow(clippy::too_many_arguments)]
    fn attn_norm_kv(
        &self,
        a: &AttnBlock,
        blk: &TensorRef,
        res: &TensorRef,
        sum: &TensorRef,
        n: &TensorRef,
        t: usize,
        eps: &OpAttrs,
        profile: bool,
    ) -> Result<bool> {
        if (a.wk.format, a.wv.format) != (crate::quant::FP8_E4M3, crate::quant::FP8_E4M3) {
            return Ok(false);
        }
        // the fused op projects k and v inside the norm; a bias has to land between the projection and the rope
        if a.k_bias.is_some() || a.v_bias.is_some() {
            return Ok(false);
        }
        let d = &self.ldims;
        let kvw = d.n_kv_heads * d.head_dim;
        let (k_out, v_out) = (self.view(&self.ws.k[0], self.act, &[t, kvw]), self.view(&self.ws.v, self.act, &[t, kvw]));
        self.try_fused(
            Op::RmsNormAddMatmulFp8Pair,
            vec![
                blk.clone(),
                res.clone(),
                a.pre_norm.clone(),
                a.wk.parts[0].clone(),
                a.wk.parts[1].clone(),
                a.wv.parts[0].clone(),
                a.wv.parts[1].clone(),
            ],
            vec![n.clone(), sum.clone(), k_out, v_out],
            eps.clone(),
            profile,
        )
    }

    /// The attention prologue as one op (`Op::QkNormRopeKv`): q-norm, rope and the k / v cache appends together, on
    /// the projected q / k / v in `bufs`. The projected q is `bufs.q[*qc]`; on `true` the normed and roped q is in
    /// `bufs.q[1 - *qc]` (`qc` flipped once, as the unfused norm + rope pair leaves it) and both caches have the new
    /// rows. On `false` nothing was emitted, so the caller runs the unfused sequence unchanged: the backend has no
    /// kernel for this exact call (CUDA / HIP never register one; a cache that is not bf16 on Metal; admission), the
    /// block has no q or k norm, or fusion is off. The decision depends on the call's shape (`t` rows of `bufs`) only.
    #[allow(clippy::too_many_arguments)]
    fn qkv_fold(
        &self,
        a: &AttnBlock,
        bufs: &AttnBufs,
        qc: &mut usize,
        t: usize,
        pos0: usize,
        dpos: Option<&TensorRef>,
        kcache: &TensorRef,
        vcache: &TensorRef,
        cap: usize,
        profile: bool,
    ) -> Result<bool> {
        let (Some(qw), Some(kw)) = (&a.q_norm, &a.k_norm) else { return Ok(false) };
        // the fused prologue's rope computes plain `theta` frequencies: a scaled embedding runs the unfused ops
        if self.rope_table.is_some() {
            return Ok(false);
        }
        let d = &self.ldims;
        let (nh, nkv, hd) = (d.n_heads, d.n_kv_heads, d.head_dim);
        let attrs = OpAttrs {
            eps: d.rms_eps,
            w_offset: self.recipe.model.norm_offset,
            pos0,
            theta: d.rope_theta,
            head_dim: hd,
            rotary_dim: a.rotary_dim,
            n_heads: nh,
            n_kv_heads: nkv,
            kv_cap: cap,
            ..Default::default()
        };
        debug_assert_eq!(bufs.q[0].shape, [t, nh, hd]);
        let mut call = OpCall::new(
            Op::QkNormRopeKv,
            vec![bufs.q[*qc].clone(), bufs.k[0].clone(), bufs.v.clone(), qw.clone(), kw.clone()],
            vec![bufs.q[1 - *qc].clone(), kcache.clone(), vcache.clone()],
            attrs,
        );
        call.pos = dpos.cloned();
        if !self.fused_ok(&call) {
            return Ok(false);
        }
        self.exec_pos(call.op, call.inputs, call.outputs, call.attrs, dpos, profile)?;
        *qc = 1 - *qc;
        Ok(true)
    }

    /// The gated attention's split folded into the fused prologue (`Op::GateSplitQkNormRopeKv`, P4a): the call that
    /// replaces `GateSplit` + `QkNormRopeKv` for the `t` rows of a gated block, reading the q+gate projection
    /// ([`Model::gated_q_buffers`]) and writing the normed and roped q to `bufs.q[1]` (where the unfused split + prologue
    /// leave it, `qc` = 1), the gate to the gate buffer the output gate reads, and the new k / v rows to the caches.
    /// `None` when the block is not gated or lacks a q or k norm (like [`Model::qkv_fold`]). The caller decides with
    /// [`Model::fused_ok`] before the projections, because the split it replaces runs right after the q projection.
    #[allow(clippy::too_many_arguments)]
    fn gate_qkv_call(
        &self,
        a: &AttnBlock,
        slot: &crate::state::AttnKvState,
        bufs: &AttnBufs,
        t: usize,
        pos0: usize,
        dpos: Option<&TensorRef>,
    ) -> Option<OpCall> {
        self.gate_qkv_call_at(a, slot, bufs, t, t, 0, pos0, dpos)
    }

    /// [`Model::gate_qkv_call`] for rows `[lo, lo + t)` of a `total`-row pass (a packed pass's segment; `bufs` are that
    /// segment's views already). The q+gate projection and the gate buffer are the segment's rows of the whole-pass views.
    #[allow(clippy::too_many_arguments)]
    fn gate_qkv_call_at(
        &self,
        a: &AttnBlock,
        slot: &crate::state::AttnKvState,
        bufs: &AttnBufs,
        t: usize,
        total: usize,
        lo: usize,
        pos0: usize,
        dpos: Option<&TensorRef>,
    ) -> Option<OpCall> {
        let (Some(qw), Some(kw)) = (&a.q_norm, &a.k_norm) else { return None };
        if !a.gated || self.rope_table.is_some() {
            return None;
        }
        let d = &self.ldims;
        let (nh, nkv, hd) = (d.n_heads, d.n_kv_heads, d.head_dim);
        let (qg, gate) = self.gated_q_buffers(total);
        let (qg, gate) = (qg.rows(lo, t), gate.rows(lo, t));
        let cache_shape = [nkv, slot.cap, hd];
        let attrs = OpAttrs {
            eps: d.rms_eps,
            w_offset: self.recipe.model.norm_offset,
            pos0,
            theta: d.rope_theta,
            head_dim: hd,
            rotary_dim: a.rotary_dim,
            n_heads: nh,
            n_kv_heads: nkv,
            kv_cap: slot.cap,
            ..Default::default()
        };
        let mut call = OpCall::new(
            Op::GateSplitQkNormRopeKv,
            vec![qg.reshaped(&[t, nh, 2 * hd]), bufs.k[0].clone(), bufs.v.clone(), qw.clone(), kw.clone()],
            vec![
                bufs.q[1].clone(),
                gate,
                self.view(&slot.k, slot.dtype, &cache_shape),
                self.view(&slot.v, slot.dtype, &cache_shape),
            ],
            attrs,
        );
        call.pos = dpos.cloned();
        Some(call)
    }

    /// The q, k and v projections of an attention block for the `t` rows of `n`: q into `ws.q[0]` (with the output
    /// gate split off when the block has one), k into `ws.k[0]`, v into `ws.v`. Returns the index of the buffer
    /// holding q (always 0) and the gate. `kv_done`: k and v were already projected with the norm ([`Model::attn_norm_kv`]).
    /// `split_folded`: the gated block's split runs in the fused prologue ([`Model::gate_qkv_call`]), so only the q+gate
    /// projection runs here (the gate buffer is returned all the same; the prologue fills it).
    #[allow(clippy::too_many_arguments)]
    fn attn_project(
        &self,
        a: &AttnBlock,
        n: &TensorRef,
        t: usize,
        kv_done: bool,
        split_folded: bool,
        profile: bool,
    ) -> Result<(usize, Option<TensorRef>)> {
        let d = &self.ldims;
        let act = self.act;
        let ws = &self.ws;
        let (nh, nkv, hd) = (d.n_heads, d.n_kv_heads, d.head_dim);
        let qc = 0usize;
        let gate = if a.gated {
            let (qg, gate) = self.gated_q_buffers(t);
            self.proj(n, &a.wq, &qg, profile)?;
            if !split_folded {
                self.exec(
                    Op::GateSplit,
                    vec![qg.reshaped(&[t, nh, 2 * hd])],
                    vec![self.view(&ws.q[qc], act, &[t, nh, hd]), gate.clone()],
                    OpAttrs::default(),
                    profile,
                )?;
            }
            Some(gate)
        } else {
            let qv = self.view(&ws.q[qc], act, &[t, nh * hd]);
            self.proj(n, &a.wq, &qv, profile)?;
            if let Some(bias) = &a.q_bias {
                self.add_bias(&qv, bias, profile)?;
            }
            None
        };
        // k and v read the same activation: one launch for both when the backend has the fused fp8 / bf16 pair
        let (k_out, v_out) = (self.view(&ws.k[0], act, &[t, nkv * hd]), self.view(&ws.v, act, &[t, nkv * hd]));
        if !kv_done && !self.kv_pair(a, n, &k_out, &v_out, profile)? {
            self.proj(n, &a.wk, &k_out, profile)?;
            self.proj(n, &a.wv, &v_out, profile)?;
        }
        if !kv_done {
            if let Some(bias) = &a.k_bias {
                self.add_bias(&k_out, bias, profile)?;
            }
            if let Some(bias) = &a.v_bias {
                self.add_bias(&v_out, bias, profile)?;
            }
        }
        Ok((qc, gate))
    }

    /// `x += bias` in place (`Op::AddBias`): a projection bias, right after the projection that produced `x`.
    fn add_bias(&self, x: &TensorRef, bias: &TensorRef, profile: bool) -> Result<()> {
        self.exec(Op::AddBias, vec![x.clone(), bias.clone()], vec![x.clone()], OpAttrs::default(), profile)
    }

    /// Views of the attention scratch for `t` rows.
    fn attn_bufs(&self, t: usize) -> AttnBufs {
        let (d, act, ws) = (&self.ldims, self.act, &self.ws);
        let q3 = [t, d.n_heads, d.head_dim];
        let k3 = [t, d.n_kv_heads, d.head_dim];
        AttnBufs {
            q: [self.view(&ws.q[0], act, &q3), self.view(&ws.q[1], act, &q3)],
            k: [self.view(&ws.k[0], act, &k3), self.view(&ws.k[1], act, &k3)],
            v: self.view(&ws.v, act, &k3),
        }
    }

    /// Everything between the projections and the output projection, for the `t` rows in `bufs` against one
    /// sequence's cache `slot` (rows at `pos0..pos0 + t`): q-norm, rope, k-norm, rope, KV append (one fused op when
    /// the backend has it), then attention (`AttnDecode` for one row, `AttnPrefill` for more). Returns the attention
    /// output (`bufs.q[1 - qc]`) and the final `qc`. Plain decode, verify and prefill run it on all `t` rows of the
    /// block; a batch step runs it once per row on that row's slice of the scratch and that row's own cache, which is
    /// why a batched row is the solo step's row.
    #[allow(clippy::too_many_arguments)]
    fn attn_rows(
        &self,
        a: &AttnBlock,
        slot: &crate::state::AttnKvState,
        bufs: &AttnBufs,
        qc: usize,
        t: usize,
        pos0: usize,
        dpos: Option<&TensorRef>,
        prologue: Option<OpCall>,
        profile: bool,
    ) -> Result<(TensorRef, usize)> {
        self.attn_rows_core(a, slot, bufs, qc, t, pos0, dpos, prologue, true, profile)
    }

    /// [`Model::attn_rows`]; `core: false` stops after the KV append (the dead tail, [`Model::attn_block_kv_only`]):
    /// the cache is written by exactly the same ops, the attention itself does not run.
    #[allow(clippy::too_many_arguments)]
    fn attn_rows_core(
        &self,
        a: &AttnBlock,
        slot: &crate::state::AttnKvState,
        bufs: &AttnBufs,
        mut qc: usize,
        t: usize,
        pos0: usize,
        dpos: Option<&TensorRef>,
        prologue: Option<OpCall>,
        core: bool,
        profile: bool,
    ) -> Result<(TensorRef, usize)> {
        let d = &self.ldims;
        let (nh, nkv, hd) = (d.n_heads, d.n_kv_heads, d.head_dim);
        let eps = OpAttrs { eps: d.rms_eps, w_offset: self.recipe.model.norm_offset, ..Default::default() };
        let mut rope = OpAttrs { pos0, theta: d.rope_theta, head_dim: hd, rotary_dim: a.rotary_dim, ..Default::default() };
        // scaled embedding: `RopeScaled` reads the frequency table (extra input) and the attention factor
        let (rope_op, rope_in): (Op, Option<&TensorRef>) = match &self.rope_table {
            Some(t) => {
                rope.rope_mscale = t.mscale;
                (Op::RopeScaled, Some(&t.inv_freq))
            }
            None => (Op::Rope, None),
        };
        let rope_inputs = |x: &TensorRef| -> Vec<TensorRef> {
            let mut v = vec![x.clone()];
            v.extend(rope_in.cloned());
            v
        };
        let cache_shape = [nkv, slot.cap, hd];
        let kcache = self.view(&slot.k, slot.dtype, &cache_shape);
        let vcache = self.view(&slot.v, slot.dtype, &cache_shape);
        let append = OpAttrs { pos0, kv_cap: slot.cap, n_kv_heads: nkv, head_dim: hd, ..Default::default() };
        // the gated split + prologue in one op ([`Model::gate_qkv_call`], decided by the caller); else qk-norm + rope +
        // kv-append as one op when the backend has it (needs both norms); else the ops one by one
        let folded = match prologue {
            Some(c) => {
                debug_assert_eq!(qc, 0, "the folded split leaves q in bufs.q[1]");
                self.exec_pos(c.op, c.inputs, c.outputs, c.attrs, dpos, profile)?;
                qc = 1 - qc;
                true
            }
            None => self.qkv_fold(a, bufs, &mut qc, t, pos0, dpos, &kcache, &vcache, slot.cap, profile)?,
        };
        if !folded {
            if let Some(w) = &a.q_norm {
                self.exec(
                    Op::QkNorm,
                    vec![bufs.q[qc].clone(), w.clone()],
                    vec![bufs.q[1 - qc].clone()],
                    eps.clone(),
                    profile,
                )?;
                qc = 1 - qc;
            }
            self.exec_pos(rope_op, rope_inputs(&bufs.q[qc]), vec![bufs.q[1 - qc].clone()], rope.clone(), dpos, profile)?;
            qc = 1 - qc;

            let mut kc = 0usize;
            if let Some(w) = &a.k_norm {
                self.exec(
                    Op::QkNorm,
                    vec![bufs.k[kc].clone(), w.clone()],
                    vec![bufs.k[1 - kc].clone()],
                    eps.clone(),
                    profile,
                )?;
                kc = 1 - kc;
            }
            self.exec_pos(rope_op, rope_inputs(&bufs.k[kc]), vec![bufs.k[1 - kc].clone()], rope.clone(), dpos, profile)?;
            kc = 1 - kc;

            self.exec_pos(Op::KvAppend, vec![bufs.k[kc].clone()], vec![kcache.clone()], append.clone(), dpos, profile)?;
            self.exec_pos(Op::KvAppend, vec![bufs.v.clone()], vec![vcache.clone()], append, dpos, profile)?;
        }

        let attn_out = bufs.q[1 - qc].clone();
        if !core {
            return Ok((attn_out, qc));
        }
        let attrs = OpAttrs {
            pos0,
            scale: 1.0 / (hd as f32).sqrt(),
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
            kv_cap: slot.cap,
            ..Default::default()
        };
        if t > 1 && self.noncausal_rows.load(std::sync::atomic::Ordering::Relaxed) {
            // the DFlash draft block: every row attends to every key up to the block's last position (all of them appended
            // above), one decode call per row at that position; the rows' own rope positions are already in q
            for r in 0..t {
                let mut a = attrs.clone();
                a.pos0 = pos0 + t - 1;
                self.exec_pos(
                    Op::AttnDecode,
                    vec![bufs.q[qc].rows(r, 1), kcache.clone(), vcache.clone()],
                    vec![attn_out.rows(r, 1)],
                    a,
                    None,
                    profile,
                )?;
            }
            return Ok((attn_out, qc));
        }
        if t > 1
            && (self.verifying() || dpos.is_some() || self.head_rows_attn.load(std::sync::atomic::Ordering::Relaxed))
            && (self.backend.verify_attn_per_row() || self.verify_rows_attn)
        {
            // verify pass (eager or captured) on a backend whose multi-row attention is not bitwise the decode step's (HIP: its split-KV
            // partition follows the call's row count, and it takes no device position): one decode call per row
            // (row r at position pos0 + r; the keys it sees are the cache rows already appended). `dpos`, when given, is a
            // table of consecutive device positions, row r reading entry r.
            for r in 0..t {
                let mut a = attrs.clone();
                a.pos0 = pos0 + r;
                let dp = dpos.map(|d| {
                    let mut v = d.clone();
                    v.offset += 4 * r;
                    v
                });
                self.exec_pos(
                    Op::AttnDecode,
                    vec![bufs.q[qc].rows(r, 1), kcache.clone(), vcache.clone()],
                    vec![attn_out.rows(r, 1)],
                    a,
                    dp.as_ref(),
                    profile,
                )?;
            }
            return Ok((attn_out, qc));
        }
        self.exec_pos(
            if t == 1 { Op::AttnDecode } else { Op::AttnPrefill },
            vec![bufs.q[qc].clone(), kcache, vcache],
            vec![attn_out.clone()],
            attrs,
            if t == 1 { dpos } else { None },
            profile,
        )?;
        Ok((attn_out, qc))
    }

    /// The attention block's output projection for `t` rows: the output gate (when the block has one), `wo`, and the
    /// tensor-parallel sum.
    #[allow(clippy::too_many_arguments)]
    fn attn_out_proj(
        &self,
        a: &AttnBlock,
        attn_out: &TensorRef,
        gate: Option<TensorRef>,
        qc: usize,
        out: &TensorRef,
        t: usize,
        profile: bool,
    ) -> Result<()> {
        let d = &self.ldims;
        let qshape = [t, d.n_heads * d.head_dim];
        // the output gate folded into the fp8 o_proj's input (`SigmoidMulMatmulFp8`) when the backend has it; the gated
        // product is then never stored
        if let (Some(g), None) = (&gate, &self.ws.part) {
            if a.wo.format == crate::quant::FP8_E4M3 {
                let fused = self.try_fused(
                    Op::SigmoidMulMatmulFp8,
                    vec![attn_out.reshaped(&qshape), g.reshaped(&qshape), a.wo.parts[0].clone(), a.wo.parts[1].clone()],
                    vec![out.clone()],
                    OpAttrs::default(),
                    profile,
                )?;
                if fused {
                    return self.all_reduce(out, out, profile);
                }
            }
        }
        let o_in = match gate {
            // output gate: attn_out * sigmoid(gate), into the (now free) roped-q buffer
            Some(g) => {
                let gated = self.view(&self.ws.q[qc], self.act, &qshape);
                self.exec(
                    Op::SigmoidMul,
                    vec![attn_out.reshaped(&qshape), g.reshaped(&qshape)],
                    vec![gated.clone()],
                    OpAttrs::default(),
                    profile,
                )?;
                gated
            }
            None => attn_out.reshaped(&qshape),
        };
        match &self.ws.part {
            Some(part) => {
                let part = self.view(part, DType::F32, &[t, d.hidden]);
                self.proj(&o_in, &a.wo, &part, profile)?;
                self.all_reduce(&part, out, profile)
            }
            None => {
                self.proj(&o_in, &a.wo, out, profile)?;
                self.all_reduce(out, out, profile)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_block(
        &self,
        a: &AttnBlock,
        st: &SequenceState,
        n: &TensorRef,
        out: &TensorRef,
        t: usize,
        pos0: usize,
        dpos: Option<&TensorRef>,
        kv_done: bool,
        profile: bool,
    ) -> Result<()> {
        let slot = Self::attn_slot(st, a)?;
        let bufs = self.attn_bufs(t);
        // the gated split folded into the prologue (P4a) when the backend has the fused op for this exact call: decided
        // here, before the q projection the split follows; otherwise the split, then the prologue as before
        let prologue = self.gate_qkv_call(a, slot, &bufs, t, pos0, dpos).filter(|c| self.fused_ok(c));
        let (qc, gate) = self.attn_project(a, n, t, kv_done, prologue.is_some(), profile)?;
        let (attn_out, qc) = self.attn_rows(a, slot, &bufs, qc, t, pos0, dpos, prologue, profile)?;
        self.attn_out_proj(a, &attn_out, gate, qc, out, t, profile)
    }

    /// [`Model::attn_block`] for a packed prefill pass: the q / k / v projections and the output projection run once on all `t`
    /// rows (row-wise kernels: a row's bits do not depend on the other rows, `tests/packed_rows.rs`), the rope, KV append and
    /// attention once per segment on that segment's rows of the scratch and its own cache, as that prompt's solo chunk would.
    #[allow(clippy::too_many_arguments)]
    fn attn_block_packed(
        &self,
        a: &AttnBlock,
        segs: &[Seg<'_>],
        n: &TensorRef,
        out: &TensorRef,
        t: usize,
        kv_done: bool,
        profile: bool,
    ) -> Result<()> {
        let bufs = self.attn_bufs(t);
        // each segment decides on the fused gated prologue exactly as its solo block would (same call shape: its own rows)
        let mut prologues = Vec::with_capacity(segs.len());
        for s in segs {
            let slot = Self::attn_slot(s.st, a)?;
            let bufs_j = bufs.rows(s.lo, s.len);
            prologues.push(self.gate_qkv_call_at(a, slot, &bufs_j, s.len, t, s.lo, s.pos, None).filter(|c| self.fused_ok(c)));
        }
        // the gated split runs per segment below when its prologue is not fused, so the projection leaves it out
        let (qc, gate) = self.attn_project(a, n, t, kv_done, true, profile)?;
        let d = &self.ldims;
        let mut qc_after = qc;
        if prologues.iter().any(|p| p.is_some() != prologues[0].is_some()) {
            return Err(Error::invalid("packed prefill segments take different attention paths (checked by pack_uniform before the pass)"));
        }
        for (s, prologue) in segs.iter().zip(prologues) {
            let slot = Self::attn_slot(s.st, a)?;
            let bufs_j = bufs.rows(s.lo, s.len);
            if a.gated && prologue.is_none() {
                let (qg, gate_all) = self.gated_q_buffers(t);
                self.exec(
                    Op::GateSplit,
                    vec![qg.rows(s.lo, s.len).reshaped(&[s.len, d.n_heads, 2 * d.head_dim])],
                    vec![bufs_j.q[qc].clone(), gate_all.rows(s.lo, s.len)],
                    OpAttrs::default(),
                    profile,
                )?;
            }
            let (_, q_end) = self.attn_rows(a, slot, &bufs_j, qc, s.len, s.pos, None, prologue, profile)?;
            qc_after = q_end;
        }
        let attn_out = bufs.q[1 - qc_after].clone();
        self.attn_out_proj(a, &attn_out, gate, qc_after, out, t, profile)
    }

    /// [`Model::attn_block`] for the last layer of a chunk that feeds no head: the projections and the prologue that
    /// writes the cache (k/v projection, k-norm, rope, KV append, or the fused gated prologue) run exactly as there; the
    /// attention core and the output projection, whose results nothing reads, do not. (The q projection still runs: the
    /// fused prologue takes q+gate as its input, and replacing it by the unfused k path could change the cache bits.)
    fn attn_block_kv_only(
        &self,
        a: &AttnBlock,
        st: &SequenceState,
        n: &TensorRef,
        t: usize,
        pos0: usize,
        kv_done: bool,
        profile: bool,
    ) -> Result<()> {
        let slot = Self::attn_slot(st, a)?;
        let bufs = self.attn_bufs(t);
        let prologue = self.gate_qkv_call(a, slot, &bufs, t, pos0, None).filter(|c| self.fused_ok(c));
        let (qc, _gate) = self.attn_project(a, n, t, kv_done, prologue.is_some(), profile)?;
        let _ = self.attn_rows_core(a, slot, &bufs, qc, t, pos0, None, prologue, false, profile)?;
        Ok(())
    }

    /// The KV cache slot attention block `a` reads and writes in `st`.
    fn attn_slot<'s>(st: &'s SequenceState, a: &AttnBlock) -> Result<&'s crate::state::AttnKvState> {
        match &st.slots[a.slot] {
            StateSlot::AttnKv(s) => Ok(s),
            StateSlot::Recurrent(_) => Err(Error::invalid("attention block bound to a recurrent state slot")),
        }
    }

    fn ffn_block(&self, f: &FfnBlock, n: &TensorRef, out: &TensorRef, t: usize, profile: bool) -> Result<()> {
        let act = self.act;
        let ws = &self.ws;
        let i = self.ldims.intermediate;
        let g = self.view(&ws.g, act, &[t, i]);
        let u = self.view(&ws.u, act, &[t, i]);
        let a = self.view(&ws.a, act, &[t, i]);
        let bf16 = |q: &QTensor| q.format == crate::quant::BF16;
        // gate and up read the same activation: one launch for both when the backend has the fused bf16 / mxfp4 pair
        let gu_fused = (bf16(&f.gate)
            && bf16(&f.up)
            && self.try_fused(
                Op::MatmulPair,
                vec![n.clone(), f.gate.parts[0].clone(), f.up.parts[0].clone()],
                vec![g.clone(), u.clone()],
                OpAttrs::default(),
                profile,
            )?)
            // two MXFP4 projections with the same activation mode: the fused mxfp4 pair (false for any other format)
            || self.try_mxfp4_pair(n, &f.gate, &f.up, &g, &u, profile)?;
        if !gu_fused {
            self.proj(n, &f.gate, &g, profile)?;
            self.proj(n, &f.up, &u, profile)?;
        }
        self.exec(Op::SiluGlu, vec![g, u], vec![a.clone()], OpAttrs::default(), profile)?;
        match &self.ws.part {
            Some(part) => {
                let part = self.view(part, DType::F32, &[t, self.ldims.hidden]);
                if f.down.format == crate::quant::MXFP4 {
                    // the MXFP4 matmul writes the f32 partial itself (Qwen3.8-27B tp=2; the shard holds whole 32-groups of
                    // the intermediate columns, `Mxfp4::load`)
                    self.proj(&a, &f.down, &part, profile)?;
                    return self.all_reduce(&part, out, profile);
                }
                if !bf16(&f.down) {
                    return Err(Error::Unsupported(format!("tensor-parallel dense FFN with {} weights", f.down.format)));
                }
                self.exec(Op::Logits, vec![a, f.down.parts[0].clone()], vec![part.clone()], OpAttrs::default(), profile)?;
                self.all_reduce(&part, out, profile)
            }
            None => {
                self.proj(&a, &f.down, out, profile)?;
                self.all_reduce(out, out, profile)
            }
        }
    }

    /// `out = x @ w^T` for a row-parallel projection (`w` holds this rank's input columns): into the f32 partial and
    /// summed over the group when the partials are f32, else into `out` and summed in place. Without tensor parallelism
    /// this is exactly [`Model::proj`] (the all-reduce is a no-op).
    pub(crate) fn proj_reduce(&self, x: &TensorRef, w: &QTensor, out: &TensorRef, t: usize, profile: bool) -> Result<()> {
        match &self.ws.part {
            Some(part) => {
                let part = self.view(part, DType::F32, &[t, self.ldims.hidden]);
                self.proj(x, w, &part, profile)?;
                self.all_reduce(&part, out, profile)
            }
            None => {
                self.proj(x, w, out, profile)?;
                self.all_reduce(out, out, profile)
            }
        }
    }

    /// Sum a row-parallel projection's partial results over the tensor-parallel group into `out` (in place when
    /// `partial` and `out` are the same tensor). A no-op without tensor parallelism. While the plan is recorded
    /// nothing runs: the call is counted instead (it would otherwise wait for ranks that are not running).
    fn all_reduce(&self, partial: &TensorRef, out: &TensorRef, profile: bool) -> Result<()> {
        let Some(comm) = &self.tp.comm else { return Ok(()) };
        if self.tp_arnorm {
            self.flush_ar(profile)?; // a reduction still pending (nothing between two block-end reductions emits an op, but be exact)
        }
        if let Some(r) = self.recorder.lock().unwrap().as_mut() {
            r.comm.push(crate::plan::PlannedComm {
                phase: r.phase.unwrap_or(crate::plan::Phase::Decode),
                t: r.t,
                pos0: r.pos0,
                bytes: partial.nbytes(),
                extra: r.extra,
                fused_norm: false,
            });
            if self.tp_arnorm {
                self.defer_ar(partial, out, Some(r.comm.len() - 1));
            }
            return Ok(());
        }
        if self.take_dropped_collective() {
            return Ok(()); // test-only protocol bug (`Model::set_tp_batch_mutant`): this rank skips the collective
        }
        if self.no_comm {
            return Ok(()); // INTARSIA_TP_NO_COMM: timing experiment, the activations are garbage
        }
        if let Some((rank, f)) = self.test_bug {
            if rank == self.tp.rank {
                self.scale_partial(partial, f)?;
            }
        }
        if self.tp_arnorm {
            // deferred whether or not this rank profiles the step: rank 1 never does, and the ranks must issue the same collectives
            self.defer_ar(partial, out, None);
            return Ok(());
        }
        self.reduce_now(comm, partial, out, profile)
    }

    /// Issue the all-reduce `partial -> out` on the rank's stream (timed and synchronised when `profile`).
    fn reduce_now(&self, comm: &Arc<dyn Comm>, partial: &TensorRef, out: &TensorRef, profile: bool) -> Result<()> {
        if profile {
            self.backend.synchronize(self.stream)?;
        }
        let t0 = Instant::now();
        comm.all_reduce_sum_into(partial, out, 0)?;
        if self.coll_trace.on && self.capture_ops.lock().unwrap().is_none() {
            self.coll_trace.issue();
        }
        if profile {
            self.backend.synchronize(self.stream)?;
            let d = t0.elapsed();
            comm.note_timed(d);
            self.telemetry.record_comm(Some(d));
        } else if self.capture_ops.lock().unwrap().is_some() {
            // being captured: nothing ran, each replay counts it
            self.capture_comms.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            self.telemetry.record_comm(None);
        }
        Ok(())
    }

    /// All-gather `send` (this rank's segment of `recv`, in place) into `recv` (rank-major) over the tensor-parallel group,
    /// the vocab-parallel head's collective. Bookkeeping as in [`Model::all_reduce`]: while the plan is recorded the call
    /// is counted instead of run, a captured graph counts it per replay.
    fn all_gather(&self, send: &TensorRef, recv: &TensorRef, profile: bool) -> Result<()> {
        let comm = self.tp.comm.as_ref().ok_or_else(|| Error::invalid("vocab-parallel head without a communicator"))?;
        if let Some(r) = self.recorder.lock().unwrap().as_mut() {
            r.comm.push(crate::plan::PlannedComm {
                phase: r.phase.unwrap_or(crate::plan::Phase::Decode),
                t: r.t,
                pos0: r.pos0,
                bytes: send.nbytes(),
                extra: r.extra,
                fused_norm: false,
            });
            return Ok(());
        }
        if self.no_comm {
            return Ok(()); // INTARSIA_TP_NO_COMM: timing experiment, the logits are garbage
        }
        if profile {
            self.backend.synchronize(self.stream)?;
        }
        let t0 = Instant::now();
        comm.all_gather(send, recv, 0)?;
        if self.coll_trace.on && self.capture_ops.lock().unwrap().is_none() {
            self.coll_trace.issue();
        }
        if profile {
            self.backend.synchronize(self.stream)?;
            let d = t0.elapsed();
            comm.note_timed(d);
            self.telemetry.record_comm(Some(d));
        } else if self.capture_ops.lock().unwrap().is_some() {
            self.capture_comms.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            self.telemetry.record_comm(None);
        }
        Ok(())
    }

    /// `INTARSIA_TP_TEST_BUG`: multiply a partial by `f` through the host (slow; a deliberate bug for gate sensitivity runs).
    fn scale_partial(&self, partial: &TensorRef, f: f32) -> Result<()> {
        self.backend.synchronize(self.stream)?;
        let mut bytes = vec![0u8; partial.nbytes()];
        self.backend.download(&partial.buf, partial.offset, &mut bytes, self.stream)?;
        match partial.dtype {
            DType::F32 => {
                for c in bytes.chunks_exact_mut(4) {
                    let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * f;
                    c.copy_from_slice(&v.to_le_bytes());
                }
            }
            DType::Bf16 => {
                for c in bytes.chunks_exact_mut(2) {
                    let v = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) * f;
                    let b = v.to_bits();
                    let r = (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16; // round to nearest even
                    c.copy_from_slice(&r.to_le_bytes());
                }
            }
            d => return Err(Error::invalid(format!("INTARSIA_TP_TEST_BUG: partial dtype {d:?} not supported"))),
        }
        self.backend.upload(&partial.buf, partial.offset, &bytes, self.stream)?;
        self.backend.synchronize(self.stream)
    }

    /// Logits for rows `[r0, r0+c)` of the normed hidden states, row-major f32 `[c, vocab]`.
    fn head_logits(&self, hn: &TensorRef, r0: usize, c: usize, profile: bool) -> Result<Vec<f32>> {
        let v = self.dims.vocab;
        let x = hn.rows(r0, c);
        self.head_proj(&x, c, profile)?;
        if self.recorder.lock().unwrap().is_some() {
            return Ok(Vec::new());
        }
        let mut bytes = vec![0u8; c * v * 4];
        self.backend.download(&self.ws.logits, 0, &mut bytes, self.stream)?;
        let mut out: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        if self.vocab_par && c > 1 {
            // the gathered tile is rank-major `[world][c][V / world]`: put each row's slices side by side
            let (w, vh) = (self.tp.world, v / self.tp.world);
            let flat = std::mem::take(&mut out);
            out = vec![0f32; c * v];
            for r in 0..w {
                for i in 0..c {
                    out[i * v + r * vh..i * v + (r + 1) * vh].copy_from_slice(&flat[(r * c + i) * vh..(r * c + i + 1) * vh]);
                }
            }
        }
        // Gemma 4 final logit soft-capping; monotonic, so the device argmax of `head_argmax` is unaffected
        if let Some(cap) = self.dims.gemma.as_ref().map(|g| g.softcap).filter(|&c| c > 0.0) {
            for x in out.iter_mut() {
                *x = cap * (*x / cap).tanh();
            }
        }
        Ok(out)
    }

    /// lm_head + argmax of one row, left in `ws.logits` (`[1, V]` f32) and `ws.tok`; no host traffic.
    fn head_to_device(&self, hn: &TensorRef, row: usize, profile: bool) -> Result<()> {
        self.head_to_device_into(hn, row, profile, self.view(&self.ws.tok, DType::I32, &[1]))
    }

    /// [`Model::head_to_device`] with the argmax written to `tok` (`i32 [1]`).
    fn head_to_device_into(&self, hn: &TensorRef, row: usize, profile: bool, tok: TensorRef) -> Result<()> {
        let x = hn.rows(row, 1);
        let logits = self.head_proj(&x, 1, profile)?;
        if self.tp.rank != 0 {
            return Ok(()); // a vocab-parallel peer: its half is in the gather, the argmax is rank 0's
        }
        self.exec(Op::Argmax, vec![logits], vec![tok], OpAttrs::default(), profile)
    }

    /// The head projection of `c` rows `x`. Replicated head (rank 0): `[c, V]` f32 logits in `ws.logits`. Vocab-parallel
    /// head: this rank's `V / world` logits of each row land in its segment of `ws.logits`, kept rank-major
    /// (`[world][c][V / world]`), and one all-gather in place fills the other segments, on every rank; for `c == 1` that is
    /// the `[1, V]` row itself, for more rows [`Model::head_logits`] interleaves them on the host. Every logit is the
    /// dot product of one head row with one input row, computed by the same kernel with the same reduction order as in the
    /// full head (the kernel and its tiling depend on the rows and K, never on the number of head rows: pinned by
    /// `kernels.rs` selection tests and the device bitwise test), so the gathered bits are the unsplit head's.
    fn head_proj(&self, x: &TensorRef, c: usize, profile: bool) -> Result<TensorRef> {
        let v = self.dims.vocab;
        if !self.vocab_par {
            let logits = self.view(&self.ws.logits, DType::F32, &[c, v]);
            self.proj(x, self.head()?, &logits, profile)?;
            return Ok(logits);
        }
        let (w, vh) = (self.tp.world, v / self.tp.world);
        let tile = self.view(&self.ws.logits, DType::F32, &[w, c * vh]);
        let mine = tile.rows(self.tp.rank, 1);
        self.proj(x, self.head()?, &mine.reshaped(&[c, vh]), profile)?;
        self.all_gather(&mine, &tile, profile)?;
        Ok(self.view(&self.ws.logits, DType::F32, &[c, v]))
    }

    /// The `(first row, rows)` head groups of a chunk of `len` rows that feeds the head in `mode` (`last`: the final chunk
    /// of the pass), in the order they run. One schedule for every rank: the vocab-parallel head's all-gather is a
    /// collective, so rank 0 (which owns the logits) and the peers must issue the same sequence of groups.
    fn head_groups(&self, mode: HeadMode, len: usize, last: bool) -> Vec<(usize, usize)> {
        match mode {
            HeadMode::LastLogits | HeadMode::LastArgmax if last => vec![(len - 1, 1)],
            HeadMode::AllLogits => {
                // groups of `LOGIT_ROWS` rows by the prefill rule too: a 1-row group runs the decode GEMV, which is not
                // bitwise the same row inside a >= 2-row group, so a `k * LOGIT_ROWS + 1` chunk would score its last
                // row differently from a schedule that ends it in a 2-row chunk (`(.., 15, 2)` instead of `(.., 16, 1)`)
                let avoid = self.backend.avoid_single_row_prefill();
                let (mut r, mut out) = (0, Vec::new());
                while r < len {
                    let c = crate::scheduler::next_prefill_chunk(len - r, LOGIT_ROWS, avoid);
                    out.push((r, c));
                    r += c;
                }
                out
            }
            _ => Vec::new(),
        }
    }

    /// The vocab-parallel head promises the replicated head's bits, which holds only while a logit comes out of the same
    /// kernel (so the same reduction order) whatever the number of head rows. Kernels pick their variant from the row count
    /// and K; this refuses a backend whose selection also looks at the number of head rows: for every row count a head group
    /// can have, the kernel picked for this rank's `V / world` rows must be the one picked for the full `V`. (The full-V call
    /// is the split call with the weight parts' row count and the output width restored: no full weight is needed.)
    pub(crate) fn check_vocab_split(&self) -> Result<()> {
        let head = self.head()?;
        let (v, vh) = (self.dims.vocab, self.dims.vocab / self.tp.world);
        // the head groups of a pass, then the batched decode steps' heads (row-exact kernels, `exact_rows`)
        let plain = (1..=LOGIT_ROWS.min(self.ws.max_tokens)).map(|m| (m, false));
        for (m, exact) in plain.chain((2..=self.batch_rows).map(|m| (m, true))) {
            let mut half = head.call(self.view(&self.ws.n, self.act, &[m, self.dims.hidden]), self.view(&self.ws.logits, DType::F32, &[m, vh]));
            half.exact_rows = exact;
            let mut full = half.clone();
            for t in full.inputs.iter_mut().skip(1).filter(|t| t.shape.first() == Some(&vh)) {
                t.shape[0] = v;
            }
            for t in full.outputs.iter_mut() {
                t.shape = vec![m, v];
            }
            let (a, b) = (self.registry.select(&half)?.name.clone(), self.registry.select(&full)?.name.clone());
            if a != b {
                return Err(Error::Unsupported(format!(
                    "vocab-parallel lm_head: {} rows of the head run on '{a}' but the full {v} rows on '{b}', so the split logits \
                     would not be bitwise the replicated head's (INTARSIA_TP_VOCAB=0)",
                    vh
                )));
            }
        }
        Ok(())
    }

    /// A vocab-parallel peer's share of the head of rows `[r0, r0 + c)`: its slice of the logits and the gather, no argmax.
    fn head_peer(&self, hn: &TensorRef, r0: usize, c: usize) -> Result<()> {
        self.head_proj(&hn.rows(r0, c), c, false).map(|_| ())
    }

    /// [`Model::head_to_device_into`] for the MTP drafter: over the reduced head when `--mtp-draft-vocab` is on.
    fn draft_head_to_device_into(&self, hn: &TensorRef, row: usize, profile: bool, tok: TensorRef) -> Result<()> {
        let Some(h) = self.draft_head.as_ref() else { return self.head_to_device_into(hn, row, profile, tok) };
        let x = hn.rows(row, 1);
        let logits = self.view(&self.ws.logits, DType::F32, &[1, h.n]);
        self.proj(&x, h, &logits, profile)?;
        self.exec(Op::Argmax, vec![logits], vec![tok], OpAttrs::default(), profile)
    }

    /// [`Model::head_argmax`] for the MTP drafter.
    fn draft_head_argmax(&self, hn: &TensorRef, row: usize, profile: bool) -> Result<u32> {
        if self.draft_head.is_none() {
            return self.head_argmax(hn, row, profile);
        }
        self.draft_head_to_device_into(hn, row, profile, self.view(&self.ws.tok, DType::I32, &[1]))?;
        if self.recorder.lock().unwrap().is_some() {
            return Ok(0);
        }
        let mut b = [0u8; 4];
        self.backend.download(&self.ws.tok, 0, &mut b, self.stream)?;
        Ok(i32::from_le_bytes(b) as u32)
    }

    fn head_argmax(&self, hn: &TensorRef, row: usize, profile: bool) -> Result<u32> {
        self.head_to_device(hn, row, profile)?;
        if self.recorder.lock().unwrap().is_some() {
            return Ok(0);
        }
        let mut b = [0u8; 4];
        self.backend.download(&self.ws.tok, 0, &mut b, self.stream)?;
        Ok(i32::from_le_bytes(b) as u32)
    }

    /// The output head (rank 0 only).
    fn head(&self) -> Result<&QTensor> {
        self.lm_head.as_ref().ok_or_else(|| Error::invalid(format!("rank {} of the tensor-parallel group has no lm_head", self.tp.rank)))
    }

    /// KV cache bytes one token occupies across all attention slots.
    pub fn kv_bytes_per_token(&self) -> u64 {
        self.plan.kv_bytes_per_token
    }
}
