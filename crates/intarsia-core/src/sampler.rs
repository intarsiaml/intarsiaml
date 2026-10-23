//! Host-side token sampling (greedy / temperature / top-k / top-p, seeded).
//! The `Sample` op reference on cpu-ref uses [`sample_row`] too, so device and
//! host sampling agree for the same uniform draw.

use crate::rng::SplitMix64;

#[derive(Clone, Debug)]
pub struct SamplingParams {
    /// <= 0 means greedy.
    pub temperature: f32,
    /// 0 = disabled.
    pub top_k: usize,
    /// 1.0 = disabled.
    pub top_p: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams { temperature: 0.0, top_k: 0, top_p: 1.0 }
    }
}

/// Lowest index among maximal values (the tie-break every `argmax` kernel must use).
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

/// Sort key: ascending key order is descending logit, then ascending index (the order `sample_row` walks).
/// -0.0 maps to +0.0 so the two tie on value, as `partial_cmp` has them.
fn order_key(l: f32) -> u32 {
    let b = if l == 0.0 { 0 } else { l.to_bits() };
    let m = if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 };
    !m
}

/// Sort `keys` (key << 32 | index, built in index order) by the high 32 bits. Stable LSD radix, so equal keys
/// keep ascending index: the same total order as sorting the whole u64.
fn radix_sort_hi(keys: &mut Vec<u64>, tmp: &mut Vec<u64>) {
    const BITS: u32 = 11;
    const B: usize = 1 << BITS;
    tmp.clear();
    tmp.resize(keys.len(), 0);
    let mut hist = vec![[0usize; B]; 3];
    for &k in keys.iter() {
        let h = (k >> 32) as u32;
        for (d, row) in hist.iter_mut().enumerate() {
            row[((h >> (BITS * d as u32)) as usize) & (B - 1)] += 1;
        }
    }
    for (d, row) in hist.iter_mut().enumerate() {
        if row.iter().any(|&c| c == keys.len()) {
            continue; // every key has the same digit here
        }
        let mut sum = 0;
        for c in row.iter_mut() {
            let x = *c;
            *c = sum;
            sum += x;
        }
        let shift = 32 + BITS * d as u32;
        for &k in keys.iter() {
            let b = ((k >> shift) as usize) & (B - 1);
            tmp[row[b]] = k;
            row[b] += 1;
        }
        std::mem::swap(keys, tmp);
    }
}

/// Sample one token id from `logits` using the uniform draw `u` in [0, 1).
pub fn sample_row(logits: &[f32], p: &SamplingParams, u: f32) -> usize {
    if p.temperature <= 0.0 {
        return argmax(logits);
    }
    let n = logits.len();
    if n > u32::MAX as usize || logits.iter().any(|x| x.is_nan()) {
        return sample_row_cmp(logits, p, u);
    }
    // Same order as `sample_row_cmp`'s comparator sort, on flat u64 keys (no indirect loads) instead.
    let mut keys: Vec<u64> = logits.iter().enumerate().map(|(i, &l)| (order_key(l) as u64) << 32 | i as u64).collect();
    let keep = if p.top_k > 0 && p.top_k < n { p.top_k } else { n };
    if keep < n {
        keys.select_nth_unstable(keep - 1);
        keys.truncate(keep);
        keys.sort_unstable();
    } else {
        radix_sort_hi(&mut keys, &mut Vec::new());
    }
    let idx: Vec<u32> = keys.iter().map(|&k| k as u32).collect();
    walk(logits, p, u, idx)
}

/// The comparator-sort path (NaN logits: `partial_cmp` has no total order there, so keep the old behaviour).
fn sample_row_cmp(logits: &[f32], p: &SamplingParams, u: f32) -> usize {
    let n = logits.len();
    let mut idx: Vec<u32> = (0..n as u32).collect();
    let cmp = |a: &u32, b: &u32| {
        logits[*b as usize].partial_cmp(&logits[*a as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b))
    };
    let keep = if p.top_k > 0 && p.top_k < n { p.top_k } else { n };
    if keep < n {
        idx.select_nth_unstable_by(keep - 1, cmp);
        idx.truncate(keep);
    }
    idx.sort_unstable_by(cmp);
    walk(logits, p, u, idx)
}

/// Softmax over the sorted candidates `idx`, top-p cut, then the cumulative walk to `u`.
fn walk(logits: &[f32], p: &SamplingParams, u: f32, mut idx: Vec<u32>) -> usize {
    let maxl = logits[idx[0] as usize] as f64;
    let inv_t = 1.0 / p.temperature as f64;
    let mut probs: Vec<f64> = idx.iter().map(|&i| ((logits[i as usize] as f64 - maxl) * inv_t).exp()).collect();
    let total: f64 = probs.iter().sum();
    for q in probs.iter_mut() {
        *q /= total;
    }
    if p.top_p < 1.0 {
        let mut cum = 0.0;
        let mut cut = probs.len();
        for (i, q) in probs.iter().enumerate() {
            cum += q;
            if cum >= p.top_p as f64 {
                cut = i + 1;
                break;
            }
        }
        probs.truncate(cut);
        idx.truncate(cut);
        let s: f64 = probs.iter().sum();
        for q in probs.iter_mut() {
            *q /= s;
        }
    }
    let mut acc = 0.0;
    let target = u as f64;
    for (q, &i) in probs.iter().zip(&idx) {
        acc += q;
        if target < acc {
            return i as usize;
        }
    }
    *idx.last().unwrap() as usize
}

/// Per-request sampler: owns the seeded RNG.
pub struct Sampler {
    pub params: SamplingParams,
    rng: SplitMix64,
}

impl Sampler {
    pub fn new(params: SamplingParams, seed: Option<u64>) -> Sampler {
        let seed = seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x1234_5678)
        });
        Sampler { params, rng: SplitMix64::new(seed) }
    }

    pub fn sample(&mut self, logits: &[f32]) -> usize {
        let u = self.rng.next_f32();
        sample_row(logits, &self.params, u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    #[test]
    fn greedy_and_ties() {
        assert_eq!(argmax(&[0.1, 0.9, 0.9, 0.2]), 1);
        let p = SamplingParams::default();
        assert_eq!(sample_row(&[0.1, 0.9, 0.3], &p, 0.99), 1);
    }

    #[test]
    fn top_k_one_is_argmax() {
        let p = SamplingParams { temperature: 1.0, top_k: 1, top_p: 1.0 };
        for u in [0.0, 0.5, 0.999] {
            assert_eq!(sample_row(&[0.1, 0.9, 0.3, 0.2], &p, u), 1);
        }
    }

    #[test]
    fn top_p_cuts_tail() {
        // probs ~ [0.64, 0.24, 0.09, 0.03]; top_p 0.8 keeps first two only.
        let logits = [2.0f32, 1.0, 0.0, -1.0];
        let p = SamplingParams { temperature: 1.0, top_k: 0, top_p: 0.8 };
        for u in [0.0, 0.3, 0.7, 0.999] {
            assert!(sample_row(&logits, &p, u) <= 1);
        }
    }

    #[test]
    fn seeded_reproducible() {
        let logits: Vec<f32> = (0..50).map(|i| (i as f32 * 0.37).sin()).collect();
        let p = SamplingParams { temperature: 0.8, top_k: 10, top_p: 0.95 };
        let mut a = Sampler::new(p.clone(), Some(42));
        let mut b = Sampler::new(p, Some(42));
        for _ in 0..20 {
            assert_eq!(a.sample(&logits), b.sample(&logits));
        }
    }

    /// Logits rounded to bf16 precision (many exact ties, as a bf16 lm_head produces), skewed like a real row.
    fn bf16_row(rng: &mut SplitMix64, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = rng.normal() * scale + if i % 97 == 0 { 3.0 * scale } else { 0.0 };
                f32::from_bits(x.to_bits() & 0xffff_0000)
            })
            .collect()
    }

    /// The radix-keyed path picks exactly what the comparator sort picks, for every draw.
    #[test]
    fn keyed_sort_matches_comparator_sort() {
        let mut rng = SplitMix64::new(7);
        let us = [0.0f32, 1e-7, 0.25, 0.5, 0.9, 0.999, 1.0 - 1.0 / (1u64 << 24) as f32];
        let mut rows: Vec<Vec<f32>> = Vec::new();
        for &(n, scale) in &[(1usize, 1.0f32), (2, 0.0), (7, 0.5), (300, 2.0), (5000, 3.0), (151936, 3.0)] {
            rows.push(bf16_row(&mut rng, n, scale));
            // signed zeros, -inf (min_tokens stop masking), duplicates of the max
            let mut r = bf16_row(&mut rng, n, scale);
            for (i, x) in r.iter_mut().enumerate() {
                match i % 11 {
                    1 => *x = 0.0,
                    2 => *x = -0.0,
                    3 if i > 0 => *x = f32::NEG_INFINITY,
                    _ => {}
                }
            }
            rows.push(r);
        }
        rows.push(vec![-0.0, 0.0, -0.0, 0.0]);
        rows.push(vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY, 1.0, 5.0]);
        rows.push(vec![2.0; 64]);
        for row in &rows {
            let n = row.len();
            for &t in &[0.3f32, 0.7, 1.0, 1.5] {
                for &k in &[0usize, 1, 40, n.saturating_sub(1), n] {
                    for &tp in &[1.0f32, 0.95, 0.5] {
                        let p = SamplingParams { temperature: t, top_k: k, top_p: tp };
                        for &u in &us {
                            assert_eq!(sample_row(row, &p, u), sample_row_cmp(row, &p, u), "n {n} t {t} k {k} tp {tp} u {u}");
                        }
                        for _ in 0..8 {
                            let u = rng.next_f32();
                            assert_eq!(sample_row(row, &p, u), sample_row_cmp(row, &p, u), "n {n} t {t} k {k} tp {tp} u {u}");
                        }
                    }
                }
            }
        }
        // NaN rows fall back to the comparator path
        let row = vec![1.0, f32::NAN, 0.5];
        let p = SamplingParams { temperature: 1.0, top_k: 0, top_p: 1.0 };
        assert_eq!(sample_row(&row, &p, 0.3), sample_row_cmp(&row, &p, 0.3));
    }

    /// Timing only: `cargo test --release -p intarsia-core sampler_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn sampler_timing() {
        let mut rng = SplitMix64::new(1);
        let row = bf16_row(&mut rng, 151936, 3.0);
        let p = SamplingParams { temperature: 0.7, top_k: 0, top_p: 1.0 };
        let reps = 50;
        let time = |name: &str, f: &dyn Fn() -> usize| {
            let _ = f();
            let t = std::time::Instant::now();
            let mut acc = 0;
            for _ in 0..reps {
                acc += f();
            }
            println!("{name}: {:.3} ms/call ({acc})", t.elapsed().as_secs_f64() * 1e3 / reps as f64);
        };
        time("comparator sample_row", &|| sample_row_cmp(&row, &p, 0.5));
        time("keyed sample_row", &|| sample_row(&row, &p, 0.5));
        time("  comparator sort only", &|| {
            let mut idx: Vec<u32> = (0..row.len() as u32).collect();
            idx.sort_unstable_by(|a, b| {
                row[*b as usize].partial_cmp(&row[*a as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b))
            });
            idx[0] as usize
        });
        time("  key build + radix only", &|| {
            let mut keys: Vec<u64> =
                row.iter().enumerate().map(|(i, &l)| (order_key(l) as u64) << 32 | i as u64).collect();
            radix_sort_hi(&mut keys, &mut Vec::new());
            keys[0] as u32 as usize
        });
        time("  exp + sum pass only", &|| {
            let probs: Vec<f64> = row.iter().map(|&l| ((l as f64 - 9.0) * 1.4).exp()).collect();
            probs.iter().sum::<f64>() as usize
        });
    }
}
