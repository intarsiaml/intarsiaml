//! Bitwise gate of the batched decode step (`Model::forward_batch`): for every batch size, a batch of sequences at mixed
//! context lengths, greedy and seeded-sampled rows, taken several steps in a row, must return for every row exactly the
//! bits a solo decode step of that sequence returns (`Model::forward` with one token), and leave the sequence in the same
//! state (the later steps read the KV the earlier ones wrote).
//!
//! Two independent sets of sequence states are prefilled identically: the solo set is stepped one token at a time, the
//! batch set through `forward_batch`; the two are never truncated or replayed, so a batch that forgot to write (or
//! wrote to the wrong place) a KV row shows up as soon as a later step reads it. The batch sizes run one after another
//! on the same, growing states. Compared: the f32 logits of every `LastLogits` row bit for bit (greedy rows also
//! alternate onto the on-device argmax path, `LastArgmax`, compared token for token), and the token a second,
//! identically seeded sampler draws from the batch's logits.

use intarsia_core::model::{HeadMode, Model, StepOut};
use intarsia_core::sampler::{argmax, Sampler, SamplingParams};
use intarsia_core::state::SequenceState;
use std::sync::atomic::{AtomicU64, Ordering};

/// Wall time of the solo and the batched steps so far (progress lines only).
static SOLO_NS: AtomicU64 = AtomicU64::new(0);
static BATCH_NS: AtomicU64 = AtomicU64::new(0);

pub struct Cfg {
    /// Prompt length of each sequence (one state pair per entry).
    pub lens: Vec<usize>,
    /// Batch sizes, in order.
    pub batches: Vec<usize>,
    /// Consecutive batched steps per batch size.
    pub steps: usize,
    /// Token ids of the prompts are drawn from `lo..hi`.
    pub vocab_range: (u32, u32),
}

struct Seq {
    solo: SequenceState,
    batch: SequenceState,
    /// Token to feed next (the same on both sides).
    tok: u32,
    sampled: bool,
    sampler_solo: Sampler,
    sampler_batch: Sampler,
}

fn bits(l: &[f32]) -> Vec<u32> {
    l.iter().map(|x| x.to_bits()).collect()
}

fn prompt(i: usize, len: usize, (lo, hi): (u32, u32)) -> Vec<u32> {
    let mut rng = intarsia_core::rng::SplitMix64::new(0x5eed_0000 + i as u64);
    (0..len).map(|_| lo + (rng.next_u64() % (hi - lo) as u64) as u32).collect()
}

fn make_seqs(model: &Model, lens: &[usize], vocab_range: (u32, u32)) -> Vec<Seq> {
    let params = SamplingParams { temperature: 0.8, top_k: 40, top_p: 0.9 };
    let mut seqs: Vec<Seq> = Vec::new();
    for (i, &len) in lens.iter().enumerate() {
        let p = prompt(i, len, vocab_range);
        let (mut solo, mut batch) = (model.new_state().unwrap(), model.new_state().unwrap());
        // prefill all but the last prompt token on both sides (identical code, identical chunks); the last one is the
        // first decode input
        for st in [&mut solo, &mut batch] {
            model.forward(st, &p[..len - 1], HeadMode::None, None).unwrap();
        }
        let sampled = i % 2 == 1;
        let mk = || Sampler::new(if sampled { params.clone() } else { SamplingParams::default() }, Some(100 + i as u64));
        seqs.push(Seq { solo, batch, tok: p[len - 1], sampled, sampler_solo: mk(), sampler_batch: mk() });
    }
    seqs
}

/// One batched step of `idxs` against the same solo steps; panics at the first bit that differs. Returns rows compared.
fn step(model: &Model, seqs: &mut [Seq], idxs: &[usize], step: usize) -> usize {
    step_on(model, model, seqs, idxs, step)
}

/// [`step`] with the batched pass on `bm` (a decode twin of `model`) and the solo reference on `model`.
fn step_on(model: &Model, bm: &Model, seqs: &mut [Seq], idxs: &[usize], step: usize) -> usize {
    let b = idxs.len();
    let modes: Vec<HeadMode> = idxs
        .iter()
        .map(|&i| if seqs[i].sampled || step % 2 == 0 { HeadMode::LastLogits } else { HeadMode::LastArgmax })
        .collect();
    let ids: Vec<u32> = idxs.iter().map(|&i| seqs[i].tok).collect();
    let ctx: Vec<usize> = idxs.iter().map(|&i| seqs[i].solo.len).collect();

    // INTARSIA_REPRO_CTRS=1 (lane tp2, w26): the device-side collective counters of both ranks around the batched step (HIP P2P comm only)
    let probe = |what: &str| {
        if std::env::var("INTARSIA_REPRO_CTRS").is_ok_and(|v| v == "1") {
            if let Some(st) = model.comm_debug_state() {
                eprintln!("CTRS batch {b} step {step} {what}: {st}");
            }
        }
    };
    probe("before the solo rows");
    // solo steps
    let t = std::time::Instant::now();
    let solo: Vec<StepOut> = idxs
        .iter()
        .zip(&modes)
        .map(|(&i, &m)| {
            let tok = seqs[i].tok;
            let o = model.forward(&mut seqs[i].solo, &[tok], m, None).unwrap();
            probe(&format!("after solo row {i} ({m:?}, context {})", seqs[i].solo.len));
            o
        })
        .collect();
    SOLO_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);

    // INTARSIA_REPRO_SYNC_BETWEEN=1 (w26): every rank's stream idle between the last solo step and the batched step
    if std::env::var("INTARSIA_REPRO_SYNC_BETWEEN").is_ok_and(|v| v == "1") {
        model.sync_all_ranks().unwrap();
    }
    probe("after the solo rows, before the batched step");
    // one batched step
    let t = std::time::Instant::now();
    let mut slots: Vec<Option<&mut SequenceState>> = seqs.iter_mut().map(|s| Some(&mut s.batch)).collect();
    let mut refs: Vec<&mut SequenceState> = idxs.iter().map(|&i| slots[i].take().unwrap()).collect();
    let batched = bm.forward_batch(&mut refs, &ids, &modes).unwrap();
    BATCH_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    drop(refs);
    assert_eq!(batched.len(), b);
    probe("after the batched step");

    for (r, &i) in idxs.iter().enumerate() {
        let what = format!("batch {b} step {step} row {r} (sequence {i}, context {} -> {})", ctx[r], ctx[r] + 1);
        let (next_solo, next_batch) = match (&solo[r], &batched[r]) {
            (StepOut::Token(a), StepOut::Token(c)) => {
                assert_eq!(a, c, "{what}: argmax token");
                (*a, *c)
            }
            (StepOut::Logits(a), StepOut::Logits(c)) => {
                assert_eq!(a.len(), c.len(), "{what}: logits length");
                if let Some(k) = (0..a.len()).find(|&k| a[k].to_bits() != c[k].to_bits()) {
                    let diff = (0..a.len()).filter(|&k| a[k].to_bits() != c[k].to_bits()).count();
                    // which rows of this step differ, and by how much: all rows (a rank-wide fault, e.g. one wrong collective) or one (a row fault)
                    let rows: Vec<String> = solo
                        .iter()
                        .zip(&batched)
                        .enumerate()
                        .filter_map(|(q, (x, y))| match (x, y) {
                            (StepOut::Logits(x), StepOut::Logits(y)) => {
                                let n = (0..x.len()).filter(|&j| x[j].to_bits() != y[j].to_bits()).count();
                                let md = (0..x.len()).map(|j| (x[j] - y[j]).abs()).fold(0f32, f32::max);
                                Some(format!("row {q}: {n} differ, max |d| {md:.3e}, argmax solo {} batch {}", argmax(x), argmax(y)))
                            }
                            (StepOut::Token(x), StepOut::Token(y)) => Some(format!("row {q}: token solo {x} batch {y}")),
                            _ => None,
                        })
                        .collect();
                    panic!("{what}: logits differ at {diff} of {} entries, first at {k}: solo {} batch {}; step rows [{}]", a.len(), a[k], c[k], rows.join("; "));
                }
                assert_eq!(bits(a), bits(c));
                let s = &mut seqs[i];
                if s.sampled {
                    (s.sampler_solo.sample(a) as u32, s.sampler_batch.sample(c) as u32)
                } else {
                    (argmax(a) as u32, argmax(c) as u32)
                }
            }
            (a, c) => panic!("{what}: output kinds differ: solo {a:?} batch {c:?}"),
        };
        assert_eq!(next_solo, next_batch, "{what}: next token");
        assert_eq!(seqs[i].solo.len, seqs[i].batch.len, "{what}: state length");
        assert_eq!(seqs[i].solo.tokens, seqs[i].batch.tokens, "{what}: state tokens");
        seqs[i].tok = next_solo;
    }
    b
}

/// Returns the number of (row, step) pairs compared.
pub fn run(model: &Model, cfg: &Cfg) -> usize {
    let t0 = std::time::Instant::now();
    let mut seqs = make_seqs(model, &cfg.lens, cfg.vocab_range);
    eprintln!("{} sequences prefilled twice in {:.1} s", seqs.len(), t0.elapsed().as_secs_f64());
    let n = seqs.len();
    let mut compared = 0;
    let mut offset = 0;
    for &b in &cfg.batches {
        assert!(b >= 2 && b <= n, "batch {b} of {n} sequences");
        // a rotating window, so every sequence length meets every batch size over the run
        let idxs: Vec<usize> = (0..b).map(|j| (offset + j) % n).collect();
        offset += b + 1;
        let t0 = std::time::Instant::now();
        let (s0, b0) = (SOLO_NS.load(Ordering::Relaxed), BATCH_NS.load(Ordering::Relaxed));
        for s in 0..cfg.steps {
            compared += step(model, &mut seqs, &idxs, s);
        }
        let per = |ns: u64| ns as f64 / 1e6 / cfg.steps as f64;
        eprintln!(
            "batch {b}: {} steps bitwise in {:.1} s (per step: {:.1} ms batched, {:.1} ms for the {b} solo steps)",
            cfg.steps,
            t0.elapsed().as_secs_f64(),
            per(BATCH_NS.load(Ordering::Relaxed) - b0),
            per(SOLO_NS.load(Ordering::Relaxed) - s0)
        );
    }
    compared
}

/// Batched decode steps running beside the prefill of another sequence (same model, shared scratch): between the
/// batched steps of `decode_lens.len()` decoding sequences, the next `chunk` tokens of a `prefill_len` prompt go through
/// `Model::forward` (alternately before and after the batched step). Decode rows must still equal their solo steps
/// ([`step`]); the prefilled state must equal the same prompt prefilled alone chunk by chunk: the logits of the last chunk
/// bit for bit, and three decode steps that read its KV. Returns the number of decode rows compared.
pub fn run_with_prefill(model: &Model, decode_lens: &[usize], prefill_len: usize, chunk: usize, vocab_range: (u32, u32)) -> usize {
    let mut seqs = make_seqs(model, decode_lens, vocab_range);
    let idxs: Vec<usize> = (0..seqs.len()).collect();
    let p = prompt(99, prefill_len, vocab_range);
    let (mut alone, mut mixed) = (model.new_state().unwrap(), model.new_state().unwrap());
    let pieces: Vec<&[u32]> = p.chunks(chunk).collect();
    // the reference: the prompt alone, same chunks, the last one with its logits
    let mut want = None;
    for (k, c) in pieces.iter().enumerate() {
        let last = k + 1 == pieces.len();
        let out = model.forward(&mut alone, c, if last { HeadMode::LastLogits } else { HeadMode::None }, None).unwrap();
        if last {
            want = Some(out);
        }
    }
    let mut got = None;
    let mut compared = 0;
    let mut s = 0;
    for (k, c) in pieces.iter().enumerate() {
        let last = k + 1 == pieces.len();
        let mut chunk_step = |mixed: &mut SequenceState| {
            let out = model.forward(mixed, c, if last { HeadMode::LastLogits } else { HeadMode::None }, None).unwrap();
            if last {
                got = Some(out);
            }
        };
        // two batched steps per chunk, the chunk before the first on even chunks and between the two on odd ones
        if k % 2 == 0 {
            chunk_step(&mut mixed);
            compared += step(model, &mut seqs, &idxs, s);
            compared += step(model, &mut seqs, &idxs, s + 1);
        } else {
            compared += step(model, &mut seqs, &idxs, s);
            chunk_step(&mut mixed);
            compared += step(model, &mut seqs, &idxs, s + 1);
        }
        s += 2;
    }
    let (Some(StepOut::Logits(a)), Some(StepOut::Logits(b))) = (want, got) else { panic!("no prefill logits") };
    assert_eq!(bits(&a), bits(&b), "prefill chunk logits differ from the solo prefill");
    assert_eq!(alone.len, mixed.len);
    // the KV the interleaved prefill wrote is the one the solo prefill wrote: decode three tokens from each
    let mut tok = argmax(&a) as u32;
    for i in 0..3 {
        let (x, y) = (
            model.forward(&mut alone, &[tok], HeadMode::LastLogits, None).unwrap(),
            model.forward(&mut mixed, &[tok], HeadMode::LastLogits, None).unwrap(),
        );
        let (StepOut::Logits(x), StepOut::Logits(y)) = (x, y) else { panic!() };
        assert_eq!(bits(&x), bits(&y), "decode step {i} after the interleaved prefill");
        tok = argmax(&x) as u32;
    }
    compared
}

/// R3d: batched decode steps on the decode twin `twin` (its own stream and workspace) while each prefill chunk of a `prefill_len`
/// prompt runs on ANOTHER THREAD of `model` (stream 0), so the two are really in flight together (there is no ordering between
/// them but the join at the end of each chunk). Decode rows must still equal their solo steps (on `model`, afterwards, so the
/// reference is not concurrent), the prefilled state must equal the same prompt prefilled alone chunk by chunk (last chunk's
/// logits bit for bit, then three decode steps that read its KV). At least `min_steps` batched steps run per chunk. Returns the
/// number of decode rows compared and the number of batched steps that began while a chunk was still running.
pub fn run_overlapped(
    model: &Model,
    twin: &Model,
    decode_lens: &[usize],
    prefill_len: usize,
    chunk: usize,
    vocab_range: (u32, u32),
    min_steps: usize,
) -> (usize, usize) {
    use std::sync::atomic::{AtomicBool, Ordering};
    let mut seqs = make_seqs(model, decode_lens, vocab_range);
    let idxs: Vec<usize> = (0..seqs.len()).collect();
    let p = prompt(99, prefill_len, vocab_range);
    let (mut alone, mut mixed) = (model.new_state().unwrap(), model.new_state().unwrap());
    let pieces: Vec<&[u32]> = p.chunks(chunk).collect();
    let mut want = None;
    for (k, c) in pieces.iter().enumerate() {
        let last = k + 1 == pieces.len();
        let out = model.forward(&mut alone, c, if last { HeadMode::LastLogits } else { HeadMode::None }, None).unwrap();
        if last {
            want = Some(out);
        }
    }
    // phase 1 (concurrent): the batched steps on the twin, recorded. No solo step runs here: a solo step on `model` would share
    // stream 0 (and its graphs, captured on that stream) with the prefill thread.
    struct Rec {
        idxs: Vec<usize>,
        ids: Vec<u32>,
        modes: Vec<HeadMode>,
        out: Vec<StepOut>,
        next: Vec<u32>,
        ctx: Vec<usize>,
    }
    let mut recs: Vec<Rec> = Vec::new();
    let (mut during, mut s) = (0, 0);
    let mut got = None;
    for (k, c) in pieces.iter().enumerate() {
        let last = k + 1 == pieces.len();
        let done = AtomicBool::new(false);
        let out = std::thread::scope(|sc| {
            let h = sc.spawn(|| {
                let r = model.forward(&mut mixed, c, if last { HeadMode::LastLogits } else { HeadMode::None }, None).unwrap();
                done.store(true, Ordering::Release);
                r
            });
            let mut n = 0;
            while n < min_steps || !done.load(Ordering::Acquire) {
                if !done.load(Ordering::Acquire) {
                    during += 1;
                }
                if s < 100 {
                    // the fixture's context bounds the steps; the chunk is joined below
                    let modes: Vec<HeadMode> = idxs
                        .iter()
                        .map(|&i| if seqs[i].sampled || s % 2 == 0 { HeadMode::LastLogits } else { HeadMode::LastArgmax })
                        .collect();
                    let ids: Vec<u32> = idxs.iter().map(|&i| seqs[i].tok).collect();
                    let ctx: Vec<usize> = idxs.iter().map(|&i| seqs[i].batch.len).collect();
                    let mut slots: Vec<Option<&mut SequenceState>> = seqs.iter_mut().map(|q| Some(&mut q.batch)).collect();
                    let mut refs: Vec<&mut SequenceState> = idxs.iter().map(|&i| slots[i].take().unwrap()).collect();
                    let out = twin.forward_batch(&mut refs, &ids, &modes).unwrap();
                    drop(refs);
                    let mut next = Vec::new();
                    for (r, &i) in idxs.iter().enumerate() {
                        let t = match &out[r] {
                            StepOut::Token(t) => *t,
                            StepOut::Logits(l) => {
                                let q = &mut seqs[i];
                                if q.sampled {
                                    q.sampler_batch.sample(l) as u32
                                } else {
                                    argmax(l) as u32
                                }
                            }
                            o => panic!("unexpected output {o:?}"),
                        };
                        seqs[i].tok = t;
                        next.push(t);
                    }
                    recs.push(Rec { idxs: idxs.clone(), ids, modes, out, next, ctx });
                    s += 1;
                } else {
                    std::thread::yield_now();
                }
                n += 1;
                if s >= 100 && n >= min_steps && done.load(Ordering::Acquire) {
                    break;
                }
            }
            h.join().unwrap()
        });
        if last {
            got = Some(out);
        }
    }
    // phase 2 (nothing else in flight): the solo steps with the same inputs, compared bit for bit
    let mut compared = 0;
    for (n, rec) in recs.iter().enumerate() {
        for (r, &i) in rec.idxs.iter().enumerate() {
            let what = format!("overlapped step {n} row {r} (sequence {i}, context {} -> {})", rec.ctx[r], rec.ctx[r] + 1);
            let solo = model.forward(&mut seqs[i].solo, &[rec.ids[r]], rec.modes[r], None).unwrap();
            let next_solo = match (&solo, &rec.out[r]) {
                (StepOut::Token(a), StepOut::Token(c)) => {
                    assert_eq!(a, c, "{what}: argmax token");
                    *a
                }
                (StepOut::Logits(a), StepOut::Logits(c)) => {
                    assert_eq!(bits(a), bits(c), "{what}: logits");
                    if seqs[i].sampled {
                        seqs[i].sampler_solo.sample(a) as u32
                    } else {
                        argmax(a) as u32
                    }
                }
                (a, c) => panic!("{what}: output kinds differ: solo {a:?} batch {c:?}"),
            };
            assert_eq!(next_solo, rec.next[r], "{what}: next token");
            compared += 1;
        }
    }
    for q in &seqs {
        assert_eq!(q.solo.len, q.batch.len, "state length");
        assert_eq!(q.solo.tokens, q.batch.tokens, "state tokens");
    }
    let (Some(StepOut::Logits(a)), Some(StepOut::Logits(b))) = (want, got) else { panic!("no prefill logits") };
    assert_eq!(bits(&a), bits(&b), "prefill chunk logits differ from the solo prefill");
    assert_eq!(alone.len, mixed.len);
    let mut tok = argmax(&a) as u32;
    for i in 0..3 {
        let (x, y) = (
            model.forward(&mut alone, &[tok], HeadMode::LastLogits, None).unwrap(),
            model.forward(&mut mixed, &[tok], HeadMode::LastLogits, None).unwrap(),
        );
        let (StepOut::Logits(x), StepOut::Logits(y)) = (x, y) else { panic!() };
        assert_eq!(bits(&x), bits(&y), "decode step {i} after the overlapped prefill");
        tok = argmax(&x) as u32;
    }
    (compared, during)
}
