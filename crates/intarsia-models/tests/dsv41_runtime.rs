//! The DeepSeek-V4.1 runtime (`intarsia_core::dsv41::runtime`) end to end on a few-layer rank cache against the official
//! code's layer outputs (`golden/dsv41` generator `layers_bf16.safetensors`: 16 prefill rows, then two decode steps, layers
//! 0-2 of the Mia EXL3 checkpoint). Unlike `dsv41_layers.rs` nothing is taken from the fixture except the token ids and the
//! Engram rows (served by a tier keyed by the fixture's hash ids, so the runtime's own hashing must ask for exactly those
//! rows): the routed and shared experts run through the fused `MoeExperts` kernel, every layer reads the previous one's
//! output, the state carries from the prefill into the decode steps.
//!
//! Gated on files outside the repository: `INTARSIA_DSV41_CACHE` = a `--layers 3` rank cache of world 1
//! (`intarsia dsv41-cache --layers 3 --rank 0 --world 1`), `INTARSIA_DSV41_FIXTURES` = directory with `layers_bf16.safetensors`.
//! `INTARSIA_DSV41_BACKEND=cuda` runs it on the CUDA backend (a Spark) instead of the cpu reference.
#[path = "../../intarsia-backend-cpu/tests/dsv41_common/mod.rs"]
mod dsv41_common;

use dsv41_common::*;
use intarsia_core::dsv41::engram::{record_bytes, EngramLayout, EngramShard, EngramTicket, EngramTier};
use intarsia_core::dsv41::graph::Buf;
use intarsia_core::dsv41::runtime::{Dsv41Runtime, EngramInputs};
use intarsia_core::{Backend, DType, Error, Result};
use intarsia_models::dsv41_engine::{load_runtime_with, Dsv41Opts};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Re-encode decoded Engram rows (exact in bf16) as packed records: per 32-block the smallest-exponent E8M0 scale whose
/// quotients are all exact e4m3 values (any such scale decodes to the same row).
fn encode_row(vals: &[f32]) -> Vec<u8> {
    let dh = vals.len();
    let mut table: HashMap<u32, u8> = HashMap::new();
    for b in 0u8..=255 {
        let f = intarsia_core::dtype::fp8_e4m3_to_f32(b);
        if f.is_finite() {
            table.entry(f.to_bits()).or_insert(b);
        }
    }
    let mut payload = vec![0u8; dh];
    let mut scales = vec![0u8; dh / 32];
    for blk in 0..dh / 32 {
        let v = &vals[blk * 32..(blk + 1) * 32];
        let mut done = false;
        for s in 0u8..=254 {
            let m = (2f64).powi(s as i32 - 127);
            let codes: Option<Vec<u8>> = v.iter().map(|&x| table.get(&((x as f64 / m) as f32).to_bits()).copied().filter(|_| ((x as f64 / m) as f32) as f64 * m == x as f64)).collect();
            if let Some(c) = codes {
                payload[blk * 32..(blk + 1) * 32].copy_from_slice(&c);
                scales[blk] = s;
                done = true;
                break;
            }
        }
        assert!(done, "engram row block {blk} has no exact e4m3 + E8M0 encoding");
    }
    payload.extend(scales);
    payload
}

/// Serves the fixture's Engram rows by row id; a row it does not know is an error (the runtime hashed wrong).
struct FixtureTier {
    layout: EngramLayout,
    rows: HashMap<u32, Vec<u8>>,
    /// rows of layer 1 this rank owns (others come back as zero payload with unit scales)
    owned: std::ops::Range<u64>,
    rank: usize,
    world: usize,
    next: AtomicU64,
    pending: Mutex<HashMap<u64, Vec<u32>>>,
}

impl EngramTier for FixtureTier {
    fn layout(&self) -> &EngramLayout {
        &self.layout
    }
    fn shard(&self) -> EngramShard {
        EngramShard { rank: self.rank, world: self.world }
    }
    fn owned_rows(&self, _layer: usize) -> std::ops::Range<u64> {
        self.owned.clone()
    }
    fn issue(&self, layer: usize, rows: &[u32]) -> Result<EngramTicket> {
        if layer != 0 {
            return Err(Error::invalid("fixture tier holds layer 1 only"));
        }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        self.pending.lock().unwrap().insert(id, rows.to_vec());
        Ok(EngramTicket(id))
    }
    fn wait(&self, t: EngramTicket, out: &mut [u8]) -> Result<()> {
        let rows = self.pending.lock().unwrap().remove(&t.0).unwrap();
        let rb = record_bytes(256);
        for (i, r) in rows.iter().enumerate() {
            if !self.owned.contains(&(*r as u64)) {
                out[i * rb..i * rb + 256].fill(0);
                out[i * rb + 256..(i + 1) * rb].fill(127);
                continue;
            }
            let rec = self.rows.get(r).ok_or_else(|| Error::invalid(format!("engram row {r} is not one the official hash asks for")))?;
            out[i * rb..(i + 1) * rb].copy_from_slice(rec);
        }
        Ok(())
    }
}

fn err(got: &[f32], want: &[f32]) -> (f64, f64) {
    let scale = want.iter().fold(0f64, |m, &x| m.max(x.abs() as f64));
    let max_abs = got.iter().zip(want).fold(0f64, |m, (&a, &b)| m.max((a as f64 - b as f64).abs()));
    let rms = (got.iter().zip(want).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    let rrms = (want.iter().map(|&b| (b as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    (max_abs / scale.max(1e-30), rms / rrms.max(1e-30))
}

/// The residual the head reads (the last layer's output), `[t, 4, dim]` as f32.
fn residual(rt: &Dsv41Runtime, t: usize, pos0: usize) -> Vec<f32> {
    let g = rt.graph(t, pos0).unwrap();
    let head = g.stages.last().unwrap();
    let Buf::Ws(name) = &head.steps[0].inputs[0].buf else { panic!("head input is not a workspace") };
    let n = t * 4 * rt.cfg.dim;
    if rt.opts.act == DType::F32 {
        let b = rt.read_ws(name, DType::F32, n).unwrap();
        return b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    }
    let b = rt.read_ws(name, DType::Bf16, n).unwrap();
    b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
}

fn fixture_engram(fx: &St, rank: usize, world: usize) -> EngramInputs {
    let hash = intarsia_engram::HashLayout::dsv41();
    let mut rows = HashMap::new();
    for s in ["s0", "s1", "s2"] {
        let ids = fx.get(&format!("bf16/L1/{s}/engram.hash_ids")).i32();
        let vals = fx.get(&format!("bf16/L1/{s}/engram.rows")).f32();
        for (i, &id) in ids.iter().enumerate() {
            rows.insert(id as u32, encode_row(&vals[i * 256..(i + 1) * 256]));
        }
    }
    let tm: Vec<u32> = fx.get("token_map").i32().into_iter().map(|v| v as u32).collect();
    let owned = hash.rank_rows(0, rank, world);
    let tier = FixtureTier { layout: hash.layout().clone(), rows, owned, rank, world, next: AtomicU64::new(1), pending: Mutex::new(HashMap::new()) };
    EngramInputs { tier: Arc::new(tier), hasher: hash.hasher.clone(), token_map: tm }
}

/// Prefill 16 then decode 2, per truncation depth `layers`; returns (layer, step, max_abs/max|ref|, rel_rms).
fn run_steps(rt: &mut Dsv41Runtime, fx: &St, read: bool) -> Vec<(usize, &'static str, f64, f64)> {
    run_steps_flow(rt, fx, fx, "bf16", read)
}

/// `fxf`: the fixture of the compared flow (`flow`/L../out.h); `fx`: the bf16 fixture (ids, routes).
fn run_steps_flow(rt: &mut Dsv41Runtime, fx: &St, fxf: &St, flow: &str, read: bool) -> Vec<(usize, &'static str, f64, f64)> {
    let ids: Vec<u32> = fx.get("ids").i32().into_iter().map(|v| v as u32).collect();
    let mut out = vec![];
    for l in 0..3 {
        rt.opts.tp.layers = Some(l + 1);
        let mut st = rt.new_state().unwrap();
        for (step, (a, b)) in [("s0", (0, 16)), ("s1", (16, 17)), ("s2", (17, 18))] {
            let pos0 = st.len;
            rt.forward(&mut st, &ids[a..b], true).unwrap();
            if read && b - a == 1 && std::env::var_os("INTARSIA_DSV41_ROUTES").is_some() {
                // the last layer's routing of this decode step against the official one (a near-tie flip explains a jump)
                let r = rt.read_ws("route_ids7", DType::I32, 7).unwrap();
                let mut got: Vec<i32> = r.chunks_exact(4).take(6).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
                got.sort();
                let k = format!("bf16/L{l}/{step}/ffn.gate_idx");
                let mut want = if fx.has(&k) { fx.get(&k).i32() } else { vec![] };
                want.sort();
                println!("ROUTE L{l} {step}: ours {got:?} official {want:?}");
            }
            if read {
                let got = residual(rt, b - a, pos0);
                let want = fxf.get(&format!("{flow}/L{l}/{step}/out.h")).f32();
                let (mx, rr) = err(&got, &want);
                out.push((l, step, mx, rr));
            }
        }
    }
    out
}

/// Two in-process ranks (host communicator) on the two `--layers 3` rank caches of world 2: the sharded weights, the
/// collectives and the column-split Engram rows must reproduce the official layer outputs like one rank does.
/// `INTARSIA_DSV41_CACHE_R0` / `_R1` = the rank caches (`intarsia dsv41-cache --layers 3 --rank {0,1} --world 2`).
#[test]
fn few_layer_runtime_two_ranks_match_the_official_layer_outputs() {
    let (Some(c0), Some(c1), Some(fxd)) = (
        std::env::var_os("INTARSIA_DSV41_CACHE_R0").map(PathBuf::from),
        std::env::var_os("INTARSIA_DSV41_CACHE_R1").map(PathBuf::from),
        std::env::var_os("INTARSIA_DSV41_FIXTURES").map(PathBuf::from),
    ) else {
        println!("skipped: set INTARSIA_DSV41_CACHE_R0, INTARSIA_DSV41_CACHE_R1 and INTARSIA_DSV41_FIXTURES");
        return;
    };
    // bf16 (bound as for one rank) and fp32 (the sharding and the collectives must not change the result beyond rounding)
    for (flow, act, bound) in [("fp32", DType::F32, 1e-4), ("bf16", DType::Bf16, 3e-2)] {
        let backends: Vec<Arc<dyn Backend>> = vec![intarsia_backend_cpu::CpuBackend::arc(), intarsia_backend_cpu::CpuBackend::arc()];
        let comms = intarsia_core::comm::HostCommGroup::new(backends.clone());
        let hs: Vec<_> = [c0.clone(), c1.clone()]
            .into_iter()
            .enumerate()
            .map(|(r, cache)| {
                let (b, c, fxd) = (backends[r].clone(), comms[r].clone(), fxd.clone());
                std::thread::spawn(move || {
                    let fx = St::open(&fxd.join("layers_bf16.safetensors"));
                    let fxf = St::open(&fxd.join(format!("layers_{flow}.safetensors")));
                    let o = Dsv41Opts { cap: 512, chunk: 16, engram_dir: PathBuf::new(), hot_mb: 0, act, spec: false };
                    let (mut rt, m) = load_runtime_with(&cache, b, Some(c), &o, Some(fixture_engram(&fx, r, 2))).unwrap();
                    assert_eq!((m.rank, m.world, m.layers), (r, 2, Some(3)));
                    run_steps_flow(&mut rt, &fx, &fxf, flow, r == 0)
                })
            })
            .collect();
        let res: Vec<_> = hs.into_iter().map(|h| h.join().unwrap()).collect();
        let mut worst = 0f64;
        for (l, step, mx, rr) in &res[0] {
            println!("RUNTIME TP2 {flow} L{l} {step}: max_abs/max|ref| {mx:.3e}  rel_rms {rr:.3e}");
            worst = worst.max(*mx);
        }
        assert!(worst < bound, "{flow}: worst layer output error {worst:.3e}");
    }
}

#[test]
fn few_layer_runtime_fp32_matches_the_official_fp32_layer_outputs() {
    let (Some(cache), Some(fxd)) = (std::env::var_os("INTARSIA_DSV41_CACHE").map(PathBuf::from), std::env::var_os("INTARSIA_DSV41_FIXTURES").map(PathBuf::from)) else {
        println!("skipped: set INTARSIA_DSV41_CACHE and INTARSIA_DSV41_FIXTURES");
        return;
    };
    let fx = St::open(&fxd.join("layers_bf16.safetensors"));
    let fx32 = St::open(&fxd.join("layers_fp32.safetensors"));
    let o = Dsv41Opts { cap: 512, chunk: 16, engram_dir: PathBuf::new(), hot_mb: 0, act: DType::F32, spec: false };
    // cpu only: the CUDA backend's shared kernels (embedding, norms) take bf16 activations only
    if std::env::var("INTARSIA_DSV41_BACKEND").as_deref() == Ok("cuda") {
        println!("skipped: the fp32 flow runs on the cpu reference only (CUDA embedding / norms are bf16-only)");
        return;
    }
    let (backend, bound): (Arc<dyn Backend>, f64) = (intarsia_backend_cpu::CpuBackend::arc(), 1e-4);
    let (mut rt, _) = load_runtime_with(&cache, backend, None, &o, Some(fixture_engram(&fx, 0, 1))).unwrap();
    let mut worst = 0f64;
    for (l, step, mx, rr) in run_steps_flow(&mut rt, &fx, &fx32, "fp32", true) {
        println!("RUNTIME FP32 L{l} {step}: max_abs/max|ref| {mx:.3e}  rel_rms {rr:.3e}");
        worst = worst.max(mx);
    }
    assert!(worst < bound, "worst fp32 layer output error {worst:.3e}");
}

#[test]
fn few_layer_runtime_matches_the_official_layer_outputs() {
    let (Some(cache), Some(fx)) = (std::env::var_os("INTARSIA_DSV41_CACHE").map(PathBuf::from), std::env::var_os("INTARSIA_DSV41_FIXTURES").map(PathBuf::from)) else {
        println!("skipped: set INTARSIA_DSV41_CACHE and INTARSIA_DSV41_FIXTURES (see the file header)");
        return;
    };
    let fx = St::open(&fx.join("layers_bf16.safetensors"));
    let backend: Arc<dyn Backend> = match std::env::var("INTARSIA_DSV41_BACKEND").as_deref() {
        Ok("cuda") => cuda_backend(),
        _ => intarsia_backend_cpu::CpuBackend::arc(),
    };
    let o = Dsv41Opts { cap: 512, chunk: 16, engram_dir: PathBuf::new(), hot_mb: 0, act: DType::Bf16, spec: false };
    let (mut rt, m) = load_runtime_with(&cache, backend, None, &o, Some(fixture_engram(&fx, 0, 1))).unwrap();
    assert_eq!((m.world, m.layers), (1, Some(3)), "a --layers 3 cache of world 1");
    let mut worst = 0f64;
    for (l, step, mx, rr) in run_steps(&mut rt, &fx, true) {
        println!("RUNTIME L{l} {step}: max_abs/max|ref| {mx:.3e}  rel_rms {rr:.3e}");
        worst = worst.max(mx);
    }
    // bf16 flow: the per-layer parity of the ops is 1.4e-3 .. 5.5e-3 (dsv41_layers.rs); here the routed experts are
    // computed too and errors carry through three layers and the decode steps
    assert!(worst < 3e-2, "worst layer output error {worst:.3e}");
}

fn cuda_backend() -> Arc<dyn Backend> {
    intarsia_backend_cuda::create(0).expect("cuda backend")
}

fn to_f32s(b: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        _ => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
    }
}

/// DSpark drafting rounds against the official drafter (`tools/dsv41/fixtures.py dspark`: 16 seed rows absorbed, then 3
/// rounds of one absorbed row + a 5-row block): the head input, the head's logits at the official top-64 ids, the drafts
/// (Markov loop, greedy) and the confidences. `INTARSIA_DSV41_CACHE` = a world-1 rank cache with its DSpark sub-cache
/// (`intarsia dsv41-cache --mtp-only`), `INTARSIA_DSV41_FIXTURES` = the dir with `layers_bf16.safetensors` (for the Engram
/// tier the loader wants) and `dspark_{fp32,bf16}.safetensors` (else the CI cuts in `golden/dsv41` are used).
#[test]
fn dspark_rounds_match_the_official_drafter() {
    let (Some(cache), Some(fxd)) = (std::env::var_os("INTARSIA_DSV41_CACHE").map(PathBuf::from), std::env::var_os("INTARSIA_DSV41_FIXTURES").map(PathBuf::from)) else {
        println!("skipped: set INTARSIA_DSV41_CACHE (with its mtp sub-cache) and INTARSIA_DSV41_FIXTURES");
        return;
    };
    let cuda = std::env::var("INTARSIA_DSV41_BACKEND").as_deref() == Ok("cuda");
    let golden = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/dsv41");
    let fx = St::open(&fxd.join("layers_bf16.safetensors"));
    let mut fails = Vec::new();
    for (flow, act) in [("fp32", DType::F32), ("bf16", DType::Bf16)] {
        if std::env::var("INTARSIA_DSPARK_FLOW").is_ok_and(|f| f != flow) {
            continue;
        }
        if cuda && act == DType::F32 {
            println!("DSPARK {flow}: skipped on cuda (bf16 activations only)");
            continue;
        }
        let full = fxd.join(format!("dspark_{flow}.safetensors"));
        let ds = St::open(&if full.exists() { full } else { golden.join(format!("dspark_{flow}_ci.safetensors")) });
        let backend: Arc<dyn Backend> = if cuda { cuda_backend() } else { intarsia_backend_cpu::CpuBackend::arc() };
        let o = Dsv41Opts { cap: 512, chunk: 16, engram_dir: PathBuf::new(), hot_mb: 0, act, spec: true };
        let (rt, _) = load_runtime_with(&cache, backend, None, &o, Some(fixture_engram(&fx, 0, 1))).unwrap();
        assert_eq!(rt.dspark, 3, "the sub-cache's three stages");
        let (d, v) = (rt.cfg.dim, rt.cfg.vocab);
        let mh = ds.get("in/main_hidden").f32();
        let w = mh.len() / 19;
        let ids: Vec<u32> = ds.get("in/input_ids").i32().into_iter().map(|x| x as u32).collect();
        let mut st = rt.new_state().unwrap();
        rt.absorb(&mut st, &mh[..16 * w]).unwrap();
        let (mut same_ids, mut worst_hx, mut worst_lg, mut worst_conf, mut best_lg) = (0, 0f64, 0f64, 0f64, f64::INFINITY);
        for (j, &first) in ids.iter().enumerate() {
            rt.absorb(&mut st, &mh[(16 + j) * w..(17 + j) * w]).unwrap();
            let dr = rt.draft(&st, first).unwrap().expect("rank 0 drafts");
            let r = format!("r{j}");
            let want_ids: Vec<u32> = ds.get(&format!("{r}/head.output_ids")).i32().into_iter().map(|x| x as u32).collect();
            let hx = to_f32s(&rt.read_ws("dspark.hx", act, 5 * d).unwrap(), act);
            let (_, hx_rr) = err(&hx, &ds.get(&format!("{r}/head.x")).f32());
            // the head's raw logits (before the Markov bias) at the official top-64 ids of every row
            let lg = to_f32s(&rt.read_ws("dspark.logits", DType::F32, 5 * v).unwrap(), DType::F32);
            // CI cut: the official top-64 per row; full fixture: the whole `head.logits_raw [block, V]`, top-64 taken here
            let (ti, tv) = if ds.has(&format!("{r}/head.logits_raw_top64.i")) {
                (ds.get(&format!("{r}/head.logits_raw_top64.i")).i32(), ds.get(&format!("{r}/head.logits_raw_top64.v")).f32())
            } else {
                let raw = ds.get(&format!("{r}/head.logits_raw")).f32();
                let (mut ti, mut tv) = (Vec::new(), Vec::new());
                for row in raw.chunks_exact(v) {
                    let mut ix: Vec<usize> = (0..v).collect();
                    ix.select_nth_unstable_by(63, |&a, &b| row[b].total_cmp(&row[a]));
                    ix.truncate(64);
                    ix.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
                    ti.extend(ix.iter().map(|&i| i as i32));
                    tv.extend(ix.iter().map(|&i| row[i]));
                }
                (ti, tv)
            };
            let got: Vec<f32> = ti.iter().enumerate().map(|(k, &id)| lg[(k / 64) * v + id as usize]).collect();
            let (_, lg_rr) = err(&got, &tv);
            let want_conf = ds.get(&format!("{r}/head.confidence")).f32();
            let conf = dr.confidence.iter().zip(&want_conf).fold(0f64, |m, (a, b)| m.max((a - b).abs() as f64));
            println!("DSPARK {flow} round {j}: ids {:?} official {want_ids:?}; head.x rel_rms {hx_rr:.3e}, top-64 raw logits rel_rms {lg_rr:.3e}, confidence max |diff| {conf:.3e} ({:?} vs {want_conf:?})", dr.ids, dr.confidence);
            same_ids += dr.ids.iter().zip(&want_ids).skip(1).take_while(|(a, b)| a == b).count();
            (worst_hx, worst_lg, worst_conf, best_lg) = (worst_hx.max(hx_rr), worst_lg.max(lg_rr), worst_conf.max(conf), best_lg.min(lg_rr));
        }
        println!("DSPARK {flow}: leading drafts equal {same_ids}/15, worst head.x {worst_hx:.3e}, logits {worst_lg:.3e}, confidence {worst_conf:.3e}");
        // fp32: rounds 1-2 match to ~1e-6; round 0 sits ~1.7e-3 off (likely an FP4 q-quant bucket flip in the drafter's sparse
        // attention moves head.x, the logits stay within 1e-3)
        if act == DType::F32 {
            if !(worst_hx < 5e-3 && worst_lg < 1e-3) || same_ids != 15 || worst_conf >= 1e-2 {
                fails.push(format!("fp32 drafter parity: head.x {worst_hx:.3e}, logits {worst_lg:.3e}, drafts equal {same_ids}/15, confidence {worst_conf:.3e}"));
            }
        } else if cuda {
            // CUDA bf16 differs from the CPU reference by bf16 noise (~1e-2 per stage in the op-by-op dump,
            // INTARSIA_DSV41_DUMP + tools/dsv41/dumpdiff.py), which flips routing near-ties (raw-logit margins 0.03-0.06)
            // inside the 5-row block; the block attends to itself, so one flip moves every row. A round without a flip
            // stays close (logits ~1.4e-2); gross breakage shows as ~1.
            if !(best_lg < 3e-2 && worst_lg < 0.3) {
                fails.push(format!("cuda bf16 drafter parity: best round's logits {best_lg:.3e} (< 3e-2), worst {worst_lg:.3e} (< 0.3)"));
            }
        } else if !(worst_hx < 5e-2 && worst_lg < 5e-2) {
            fails.push(format!("bf16 drafter parity: head.x {worst_hx:.3e}, logits {worst_lg:.3e}"));
        }
    }
    assert!(fails.is_empty(), "{}", fails.join("; "));
}

/// Speculative greedy decoding (DSpark draft, multi-row verify, rollback at the first rejection) gives the same tokens
/// as plain greedy decoding. Same engine, same prompt: a request with a logits trace decodes one token a pass (the
/// engine never speculates with a trace), one without speculates. On the cpu backend with f32 activations the verify
/// rows are row-invariant, so the token ids must be equal; the prompt is long enough that decoding crosses the 128-row
/// window ring wrap, with rollbacks through it and through the ratio-2 compressor groups (layer 2 of a `--layers 3` cache;
/// the ratio-1 layers start at 20). `INTARSIA_DSV41_CACHE` as above, with
/// its DSpark sub-cache; Engram off (zero rows, both requests alike). `INTARSIA_DSV41_SPEC_PROMPT` / `_TOKENS` resize it.
#[test]
fn speculative_greedy_equals_plain_greedy() {
    use intarsia_core::engine::{Engine, GenEvent, GenRequest};
    let Some(cache) = std::env::var_os("INTARSIA_DSV41_CACHE").map(PathBuf::from) else {
        println!("skipped: set INTARSIA_DSV41_CACHE (with its mtp sub-cache)");
        return;
    };
    let cuda = std::env::var("INTARSIA_DSV41_BACKEND").as_deref() == Ok("cuda");
    let n_prompt: usize = std::env::var("INTARSIA_DSV41_SPEC_PROMPT").ok().and_then(|v| v.parse().ok()).unwrap_or(110);
    let n_gen: usize = std::env::var("INTARSIA_DSV41_SPEC_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(48);
    std::env::set_var("INTARSIA_DSV41_SPEC", "1");
    std::env::set_var("INTARSIA_DSV41_ENGRAM", "off");
    std::env::set_var("INTARSIA_DSV41_CAP", "512");
    std::env::set_var("INTARSIA_DSV41_CHUNK", "64");
    if !cuda && std::env::var_os("INTARSIA_DSV41_ACT").is_none() {
        std::env::set_var("INTARSIA_DSV41_ACT", "f32");
    }
    let exact = !cuda && std::env::var("INTARSIA_DSV41_ACT").as_deref() == Ok("f32");
    let backend: Arc<dyn Backend> = if cuda { cuda_backend() } else { intarsia_backend_cpu::CpuBackend::arc() };
    let eng = intarsia_models::dsv41_engine::Dsv41Engine::load(&cache, backend, None, None, None, None).unwrap();
    // a fixed pseudo-text prompt: common token ids, repeated phrases so the drafter gets some right
    let prompt: Vec<u32> = (0..n_prompt).map(|i| [301u32, 270, 8331, 1205, 778, 12345, 1137, 15, 3000, 4321][(i * 7 + i / 10) % 10]).collect();
    let run = |trace: bool| -> Vec<u32> {
        let req = GenRequest {
            prompt: prompt.clone(),
            max_tokens: n_gen,
            temperature: 0.0,
            ignore_eos: true,
            trace_logits: trace.then(|| Arc::new(Mutex::new(Vec::new()))),
            ..Default::default()
        };
        let mut ids = Vec::new();
        eng.generate(req, &mut |e| match e {
            GenEvent::Token { id, .. } => ids.push(id),
            GenEvent::Error { message } => panic!("generate: {message}"),
            _ => {}
        })
        .unwrap();
        ids.truncate(n_gen); // a last Token may repeat the final id to carry held-back text
        ids
    };
    let plain = run(true);
    let spec = run(false);
    let lead = plain.iter().zip(&spec).take_while(|(a, b)| a == b).count();
    println!("SPEC plain {plain:?}\nSPEC spec  {spec:?}\nSPEC leading equal {lead}/{}", plain.len());
    if exact {
        assert_eq!(plain, spec, "speculative greedy differs from plain greedy at token {lead}");
    }
}

/// Teacher-forced check of the verify path: one state decodes a fixed continuation one token a pass; a second runs, at
/// every position, a verify pass `[token, 3 drafts]` and rolls back after it, the drafts alternately wrong (rejected:
/// keep 1 row) and right (accepted: keep all 4 rows, the next pass starts 4 tokens on). Each kept verify row's logits
/// are compared with the decoding state's logits at the same position, so trajectories never part. cpu f32: equal to
/// 1e-5 (and the same argmax); cuda bf16: printed per position (past the 128-row window ring wrap), median < 2e-2, worst
/// < 0.3. Covers the compressor ratios the cache's layers have (`--layers 21` reaches the first ratio-1 layer). `INTARSIA_DSV41_CACHE` as above with its DSpark sub-cache; Engram off.
#[test]
fn verify_rows_track_decode_through_rollback() {
    let Some(cache) = std::env::var_os("INTARSIA_DSV41_CACHE").map(PathBuf::from) else {
        println!("skipped: set INTARSIA_DSV41_CACHE (with its mtp sub-cache)");
        return;
    };
    let cuda = std::env::var("INTARSIA_DSV41_BACKEND").as_deref() == Ok("cuda");
    let act = if cuda || std::env::var("INTARSIA_DSV41_ACT").as_deref() == Ok("bf16") { DType::Bf16 } else { DType::F32 };
    let backend: Arc<dyn Backend> = if cuda { cuda_backend() } else { intarsia_backend_cpu::CpuBackend::arc() };
    let knob = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    // past index_topk (512) compressed keys the indexer really selects: e.g. INTARSIA_DSV41_VT_PROMPT=560 with a cap of 1024
    let (n_prompt, n_cont, cap) = (knob("INTARSIA_DSV41_VT_PROMPT", 110), knob("INTARSIA_DSV41_VT_TOKENS", 64), knob("INTARSIA_DSV41_VT_CAP", 512));
    let o = Dsv41Opts { cap, chunk: 64, engram_dir: PathBuf::from("off"), hot_mb: 0, act, spec: true };
    let (rt, _) = load_runtime_with(&cache, backend, None, &o, None).unwrap();
    let (worst, worst_at, argmax_diff, mut all) = verify_track(&rt, n_prompt, n_cont, true);
    println!("VERIFY worst rel_rms {worst:.3e} at position {worst_at}; argmax differs at {argmax_diff:?}");
    if act == DType::F32 && !cuda {
        assert!(worst < 1e-5 && argmax_diff.is_empty(), "f32 verify rows differ from decode: {worst:.3e} at {worst_at}");
    } else {
        // bf16: ~5e-3 to 1e-2 per row on the 3-layer cache; a routing near-tie that flips one expert gives a spike
        // (cuda, 2026-11-01: 0.18 at one of 64 rows) that decays over the next rows
        all.sort_by(f64::total_cmp);
        let median = all[all.len() / 2];
        println!("VERIFY median rel_rms {median:.3e}");
        assert!(median < 2e-2 && worst < 0.3, "verify rows far from decode: median {median:.3e}, worst {worst:.3e} at {worst_at}");
    }
}

/// The teacher-forced loop of [`verify_rows_track_decode_through_rollback`] on one runtime (every rank of a group runs
/// it in lockstep): `(worst rel_rms, its position, positions whose argmax differs, every kept row's rel_rms)`.
fn verify_track(rt: &intarsia_core::dsv41::runtime::Dsv41Runtime, n_prompt: usize, n_cont: usize, print: bool) -> (f64, usize, Vec<usize>, Vec<f64>) {
    let toks = |n: usize, s: usize| -> Vec<u32> { (0..n).map(|i| [301u32, 270, 8331, 1205, 778, 12345, 1137, 15, 3000, 4321][(i * 7 + i / 10 + s) % 10]).collect() };
    let (prompt, cont) = (toks(n_prompt, 0), toks(n_cont, 3));
    let rel = |a: &[f32], b: &[f32]| -> f64 {
        let (mut d, mut n) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            d += ((x - y) as f64).powi(2);
            n += (*x as f64).powi(2);
        }
        (d / n.max(1e-30)).sqrt()
    };
    // reference: decode one token a pass; logits[i] predicts cont[i + 1] after cont[..=i]
    let mut sa = rt.new_state().unwrap();
    rt.forward(&mut sa, &prompt, false).unwrap();
    let want: Vec<Vec<f32>> = cont.iter().map(|&c| rt.forward(&mut sa, &[c], true).unwrap().unwrap()).collect();
    let mut sb = rt.new_state().unwrap();
    rt.forward(&mut sb, &prompt, false).unwrap();
    let (mut i, mut round, mut worst, mut worst_at) = (0usize, 0usize, 0f64, 0usize);
    let mut argmax_diff = Vec::new();
    let mut all = Vec::new();
    while i + 4 <= cont.len() {
        let accept = round % 2 == 1;
        let ids: Vec<u32> = if accept { cont[i..i + 4].to_vec() } else { vec![cont[i], 99, 98, 97] };
        let len0 = sb.len;
        let rows = rt.forward_rows(&mut sb, &ids).unwrap();
        let keep = if accept { 4 } else { 1 };
        for (k, row) in rows.iter().take(keep).enumerate() {
            let r = rel(&want[i + k], row);
            all.push(r);
            let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(j, _)| j).unwrap();
            if am(row) != am(&want[i + k]) {
                argmax_diff.push(prompt.len() + i + k);
            }
            if r > worst {
                (worst, worst_at) = (r, prompt.len() + i + k);
            }
            if print {
                println!("VERIFY pos {} ({}, row {k}): rel_rms {r:.3e}", prompt.len() + i + k, if accept { "accepted" } else { "rejected" });
            }
        }
        sb.truncate(len0 + keep, &*rt.backend).unwrap();
        i += keep;
        round += 1;
    }
    (worst, worst_at, argmax_diff, all)
}

/// [`verify_rows_track_decode_through_rollback`] at TP=2: two in-process ranks on the cpu backend (`HostCommGroup`),
/// each running the same decode / verify / truncate sequence on its own shard, f32. Rank 0's verify rows must equal its
/// decode logits exactly, so the sharded verify pass and the truncate of every rank's slots are exact. (The tp_remote
/// TCP path that carries `Truncate` to a remote rank is not in this test.) `INTARSIA_DSV41_CACHE_R0` / `_R1`: the two
/// rank caches of one `--layers N` build, each with its DSpark sub-cache; Engram off.
#[test]
fn verify_rows_track_decode_through_rollback_two_ranks() {
    let (Some(c0), Some(c1)) = (std::env::var_os("INTARSIA_DSV41_CACHE_R0").map(PathBuf::from), std::env::var_os("INTARSIA_DSV41_CACHE_R1").map(PathBuf::from)) else {
        println!("skipped: set INTARSIA_DSV41_CACHE_R0 and INTARSIA_DSV41_CACHE_R1 (with their mtp sub-caches)");
        return;
    };
    let knob = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (n_prompt, n_cont, cap) = (knob("INTARSIA_DSV41_VT_PROMPT", 110), knob("INTARSIA_DSV41_VT_TOKENS", 64), knob("INTARSIA_DSV41_VT_CAP", 512));
    let backends: Vec<Arc<dyn Backend>> = vec![intarsia_backend_cpu::CpuBackend::arc(), intarsia_backend_cpu::CpuBackend::arc()];
    let comms = intarsia_core::comm::HostCommGroup::new(backends.clone());
    let hs: Vec<_> = [c0, c1]
        .into_iter()
        .enumerate()
        .map(|(r, cache)| {
            let (b, c) = (backends[r].clone(), comms[r].clone());
            std::thread::spawn(move || {
                let o = Dsv41Opts { cap, chunk: 64, engram_dir: PathBuf::from("off"), hot_mb: 0, act: DType::F32, spec: true };
                let (rt, m) = load_runtime_with(&cache, b, Some(c), &o, None).unwrap();
                assert_eq!((m.rank, m.world), (r, 2));
                verify_track(&rt, n_prompt, n_cont, r == 0)
            })
        })
        .collect();
    let res: Vec<_> = hs.into_iter().map(|h| h.join().unwrap()).collect();
    for (r, (worst, worst_at, argmax_diff, all)) in res.iter().enumerate() {
        println!("VERIFY TP2 rank {r}: {} rows, worst rel_rms {worst:.3e} at position {worst_at}; argmax differs at {argmax_diff:?}", all.len());
        assert!(*worst < 1e-5 && argmax_diff.is_empty(), "rank {r}: f32 TP=2 verify rows differ from decode: {worst:.3e} at {worst_at}");
    }
}

/// Device-position passes (`PassKind::dpos`, the graph CUDA graphs record) against the host-position graph on one runtime:
/// the same prompt, then the same schedule of short passes (1 row, 2-4 row chunks across compressor group boundaries and,
/// with `spec`, 4-row verify passes rolled back to 1 or kept), first with host positions, then with device positions
/// (`INTARSIA_DSV41_DPOS_GRAPHS=1`: replayed CUDA graphs). Every pass's logits must be bitwise equal.
/// `INTARSIA_DSV41_CACHE` (3-layer cache), `INTARSIA_DSV41_DP_PROMPT` / `_TOKENS` / `_CAP`, `INTARSIA_DSV41_DP_SPEC=1`.
#[test]
fn device_position_passes_equal_host_passes() {
    let Some(cache) = std::env::var_os("INTARSIA_DSV41_CACHE").map(PathBuf::from) else {
        println!("skipped: set INTARSIA_DSV41_CACHE");
        return;
    };
    let cuda = std::env::var("INTARSIA_DSV41_BACKEND").as_deref() == Ok("cuda");
    let act = if cuda || std::env::var("INTARSIA_DSV41_ACT").as_deref() == Ok("bf16") { DType::Bf16 } else { DType::F32 };
    let backend: Arc<dyn Backend> = if cuda { cuda_backend() } else { intarsia_backend_cpu::CpuBackend::arc() };
    let knob = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (n_prompt, n_cont, cap) = (knob("INTARSIA_DSV41_DP_PROMPT", 110), knob("INTARSIA_DSV41_DP_TOKENS", 48), knob("INTARSIA_DSV41_DP_CAP", 512));
    let spec = std::env::var("INTARSIA_DSV41_DP_SPEC").as_deref() == Ok("1");
    let graphs = std::env::var("INTARSIA_DSV41_DPOS_GRAPHS").as_deref() == Ok("1");
    let o = Dsv41Opts { cap, chunk: 64, engram_dir: PathBuf::from("off"), hot_mb: 0, act, spec };
    let (rt, _) = load_runtime_with(&cache, backend, None, &o, None).unwrap();
    let tok = |i: usize| ((i * 7919 + 13) % 30000 + 100) as u32;
    let prompt: Vec<u32> = (0..n_prompt).map(tok).collect();
    let cont: Vec<u32> = (0..n_cont + 8).map(|i| tok(n_prompt + i)).collect();
    // the schedule: rows per step (0: a 4-row verify pass, rolled back to 1 row on even rounds, then a drafting round)
    let sched: Vec<usize> = if spec { vec![1, 0, 1, 0, 3, 1, 0, 2, 0, 1, 1] } else { vec![1, 1, 1, 3, 1, 4, 1, 2, 1, 1, 1] };
    let run = |dev: bool| -> Vec<Vec<f32>> {
        rt.set_device_position(dev);
        if dev && graphs {
            assert!(rt.set_graphs(true), "the backend cannot capture");
        }
        let mut st = rt.new_state().unwrap();
        rt.forward(&mut st, &prompt, false).unwrap();
        let (mut out, mut i, mut k, mut round) = (Vec::new(), 0usize, 0usize, 0usize);
        while i < n_cont {
            let n = sched[k % sched.len()];
            k += 1;
            if n == 0 {
                let ids = vec![cont[i], cont[i + 1], 99, 98];
                let len0 = st.len;
                let rows = rt.forward_rows(&mut st, &ids).unwrap();
                let keep = if round % 2 == 1 { 2 } else { 1 };
                round += 1;
                out.extend(rows.into_iter().take(keep));
                st.truncate(len0 + keep, &*rt.backend).unwrap();
                i += keep;
                // a drafting round after the kept rows (replayed from a graph at positions >= the window)
                let d = rt.draft(&st, cont[i]).unwrap().expect("rank 0 drafts");
                out.push(d.ids.iter().map(|&x| x as f32).chain(d.confidence.iter().copied()).collect());
            } else {
                let n = n.min(n_cont - i);
                out.push(rt.forward(&mut st, &cont[i..i + n], true).unwrap().unwrap());
                i += n;
            }
        }
        out
    };
    let host = run(false);
    let dev = run(true);
    let (r_pass, r_verify, r_draft) = rt.graph_replays();
    assert_eq!(host.len(), dev.len());
    let mut first_diff = None;
    let mut worst = 0f32;
    for (j, (a, b)) in host.iter().zip(&dev).enumerate() {
        let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let same = a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        if !same && first_diff.is_none() {
            first_diff = Some(j);
        }
        worst = worst.max(d);
    }
    println!(
        "RUNTIME dpos vs host: {} passes ({} prompt, {} tokens, spec {spec}, graphs {graphs}, replays: {r_pass} decode / {r_verify} verify / {r_draft} draft): first bitwise difference at pass {first_diff:?}, worst |diff| {worst:.3e}",
        host.len(),
        n_prompt,
        n_cont
    );
    assert!(first_diff.is_none(), "device-position passes differ from host-position passes");
    if graphs {
        assert!(r_pass > 0, "no decode pass replayed a CUDA graph");
        if spec {
            assert!(r_verify > 0, "no verify pass replayed a CUDA graph");
            if n_prompt >= 128 {
                assert!(r_draft > 0, "no drafting block replayed a CUDA graph");
            }
        }
    }
}
