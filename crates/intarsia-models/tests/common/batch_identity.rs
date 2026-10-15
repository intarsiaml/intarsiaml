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

/// Returns the number of (row, step) pairs compared.
pub fn run(model: &Model, cfg: &Cfg) -> usize {
    let params = SamplingParams { temperature: 0.8, top_k: 40, top_p: 0.9 };
    let mut seqs: Vec<Seq> = Vec::new();
    for (i, &len) in cfg.lens.iter().enumerate() {
        let p = prompt(i, len, cfg.vocab_range);
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
    let n = seqs.len();
    let mut compared = 0;
    let mut offset = 0;
    for &b in &cfg.batches {
        assert!(b >= 2 && b <= n, "batch {b} of {n} sequences");
        // a rotating window, so every sequence length meets every batch size over the run
        let idxs: Vec<usize> = (0..b).map(|j| (offset + j) % n).collect();
        offset += b + 1;
        for step in 0..cfg.steps {
            let modes: Vec<HeadMode> = idxs
                .iter()
                .map(|&i| if seqs[i].sampled || step % 2 == 0 { HeadMode::LastLogits } else { HeadMode::LastArgmax })
                .collect();
            let ids: Vec<u32> = idxs.iter().map(|&i| seqs[i].tok).collect();
            let ctx: Vec<usize> = idxs.iter().map(|&i| seqs[i].solo.len).collect();

            // solo steps
            let solo: Vec<StepOut> = idxs
                .iter()
                .zip(&modes)
                .map(|(&i, &m)| {
                    let tok = seqs[i].tok;
                    model.forward(&mut seqs[i].solo, &[tok], m, None).unwrap()
                })
                .collect();

            // one batched step
            let mut slots: Vec<Option<&mut SequenceState>> = seqs.iter_mut().map(|s| Some(&mut s.batch)).collect();
            let mut refs: Vec<&mut SequenceState> = idxs.iter().map(|&i| slots[i].take().unwrap()).collect();
            let batched = model.forward_batch(&mut refs, &ids, &modes).unwrap();
            drop(refs);
            assert_eq!(batched.len(), b);

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
                            panic!("{what}: logits differ at {diff} of {} entries, first at {k}: solo {} batch {}", a.len(), a[k], c[k]);
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
                compared += 1;
            }
        }
    }
    compared
}
