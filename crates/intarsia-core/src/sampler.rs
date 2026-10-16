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

/// Sample one token id from `logits` using the uniform draw `u` in [0, 1).
pub fn sample_row(logits: &[f32], p: &SamplingParams, u: f32) -> usize {
    if p.temperature <= 0.0 {
        return argmax(logits);
    }
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
}
