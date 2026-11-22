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

/// Reusable host buffers of [`sample_row_in`] (lane fp8-decode): a draw over a 248k-token row needs a 2 MB key array (and,
/// without top-k, as much again for the sort and the candidates). Allocated fresh each step they can land on new pages, whose
/// faults cost ~0.7 ms a step on GB10 (docs/results/2026-11-13-fp8-decode.md); [`Sampler`] keeps one across its steps.
#[derive(Default)]
pub struct SampleScratch {
    keys: Vec<u64>,
    tmp: Vec<u64>,
    idx: Vec<u32>,
    probs: Vec<f64>,
    /// per-chunk radix histograms / offsets ([`radix_sort_hi_par`])
    hist: Vec<usize>,
}

/// Rows at least this long take the threaded path ([`sample_row_par`], [`softcap`]) when the host pool is on
/// ([`crate::par::enabled`]; lane ink-hostgap: Inkling-2-Mini's 200058 logits). Shorter rows keep the one-thread loop, whose
/// results are the same.
pub const PAR_MIN: usize = 32768;

/// The logit soft-cap `cap * tanh(x / cap)` of every element, in place: per element the same expression as the one-thread loop,
/// split across the host pool ([`crate::par`]) for a row of at least [`PAR_MIN`].
pub fn softcap(out: &mut [f32], cap: f32) {
    let n = out.len();
    let t = crate::par::threads();
    if n < PAR_MIN || !crate::par::enabled() {
        for x in out.iter_mut() {
            *x = cap * (*x / cap).tanh();
        }
        return;
    }
    let ch = n.div_ceil(2 * t);
    let o = crate::par::SharedMut(out.as_mut_ptr());
    crate::par::run(n.div_ceil(ch), &|c| {
        let (a, b) = (c * ch, ((c + 1) * ch).min(n));
        // SAFETY: chunk c owns out[a..b]; `run` returns after every chunk.
        let xs = unsafe { std::slice::from_raw_parts_mut(o.at(a), b - a) };
        for x in xs {
            *x = cap * (*x / cap).tanh();
        }
    });
}

/// Sample one token id from `logits` using the uniform draw `u` in [0, 1).
pub fn sample_row(logits: &[f32], p: &SamplingParams, u: f32) -> usize {
    sample_row_in(logits, p, u, &mut SampleScratch::default())
}

/// [`sample_row`] with its buffers in `s` (the same token for the same inputs).
pub fn sample_row_in(logits: &[f32], p: &SamplingParams, u: f32, s: &mut SampleScratch) -> usize {
    if p.temperature <= 0.0 {
        return argmax(logits);
    }
    let n = logits.len();
    // the threaded path sorts the whole row: a top-k row's `select_nth` and its few candidates gain nothing from the pool
    let full = p.top_k == 0 || p.top_k >= n;
    if full && n >= PAR_MIN && n <= u32::MAX as usize && crate::par::enabled() {
        return sample_row_par(logits, p, u, s, 2 * crate::par::threads());
    }
    sample_row_seq(logits, p, u, s)
}

/// [`sample_row_in`] on the calling thread alone (the path of rows shorter than [`PAR_MIN`]; temperature > 0).
fn sample_row_seq(logits: &[f32], p: &SamplingParams, u: f32, s: &mut SampleScratch) -> usize {
    let n = logits.len();
    if n > u32::MAX as usize || logits.iter().any(|x| x.is_nan()) {
        return sample_row_cmp(logits, p, u);
    }
    // Same order as `sample_row_cmp`'s comparator sort, on flat u64 keys (no indirect loads) instead.
    let keys = &mut s.keys;
    keys.clear();
    keys.extend(logits.iter().enumerate().map(|(i, &l)| (order_key(l) as u64) << 32 | i as u64));
    let keep = if p.top_k > 0 && p.top_k < n { p.top_k } else { n };
    if keep < n {
        keys.select_nth_unstable(keep - 1);
        keys.truncate(keep);
        keys.sort_unstable();
    } else {
        radix_sort_hi(keys, &mut s.tmp);
    }
    s.idx.clear();
    s.idx.extend(keys.iter().map(|&k| k as u32));
    walk_in(logits, p, u, &mut s.idx, &mut s.probs)
}

/// [`sample_row_in`] with its per-element work in `chunks` tasks on the host pool (lane ink-hostgap): the sort keys and the NaN
/// check, a stable chunked LSD radix sort (all keys are distinct, `key << 32 | index`, so every correct sort gives the same
/// order), the softmax terms and their divides; the f64 sums, the top-p cut and the walk stay sequential in the same order.
/// Bit for bit the token of the one-thread path, for any `chunks`. A top-k row (`0 < top_k < n`) is handed to that path.
/// The scratch keeps its lengths from call to call (every element is overwritten), so a fixed vocabulary zero-fills nothing.
fn sample_row_par(logits: &[f32], p: &SamplingParams, u: f32, s: &mut SampleScratch, chunks: usize) -> usize {
    use crate::par::{run, SharedMut};
    let n = logits.len();
    if p.top_k > 0 && p.top_k < n {
        return sample_row_seq(logits, p, u, s);
    }
    let ch = n.div_ceil(chunks.max(1));
    let nc = n.div_ceil(ch);
    let keys = &mut s.keys;
    keys.resize(n, 0);
    let nan = std::sync::atomic::AtomicBool::new(false);
    let kp = SharedMut(keys.as_mut_ptr());
    run(nc, &|c| {
        let (a, b) = (c * ch, ((c + 1) * ch).min(n));
        let mut seen = false;
        for (i, &l) in logits.iter().enumerate().take(b).skip(a) {
            seen |= l.is_nan();
            // SAFETY: chunk c owns keys[a..b].
            unsafe { *kp.at(i) = (order_key(l) as u64) << 32 | i as u64 };
        }
        if seen {
            nan.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    });
    if nan.load(std::sync::atomic::Ordering::Relaxed) {
        return sample_row_cmp(logits, p, u);
    }
    radix_sort_hi_par(keys, &mut s.tmp, &mut s.hist, chunks);
    walk_par(logits, p, u, keys, &mut s.probs, chunks)
}

/// Stable LSD radix sort of `keys` by their high 32 bits (11-bit digits, as [`radix_sort_hi`]), each pass as per-chunk
/// histograms, offsets bucket-major and chunk-minor, and a per-chunk scatter on the host pool. The same order as
/// [`radix_sort_hi`].
fn radix_sort_hi_par(keys: &mut Vec<u64>, tmp: &mut Vec<u64>, hist: &mut Vec<usize>, chunks: usize) {
    use crate::par::{run, SharedMut};
    const BITS: u32 = 11;
    const B: usize = 1 << BITS;
    let n = keys.len();
    tmp.resize(n, 0);
    let ch = n.div_ceil(chunks.max(1)).max(1);
    let nc = n.div_ceil(ch);
    hist.clear();
    hist.resize(nc * B, 0);
    for d in 0..3u32 {
        let shift = 32 + BITS * d;
        hist.fill(0);
        let (hp, src) = (SharedMut(hist.as_mut_ptr()), &keys[..]);
        run(nc, &|c| {
            // SAFETY: chunk c owns hist[c * B..(c + 1) * B].
            let h = unsafe { std::slice::from_raw_parts_mut(hp.at(c * B), B) };
            for &k in &src[c * ch..((c + 1) * ch).min(n)] {
                h[((k >> shift) as usize) & (B - 1)] += 1;
            }
        });
        let mut sum = 0usize;
        let mut one_digit = false;
        for b in 0..B {
            let mut tot = 0usize;
            for c in 0..nc {
                let x = hist[c * B + b];
                hist[c * B + b] = sum;
                sum += x;
                tot += x;
            }
            one_digit |= tot == n;
        }
        if one_digit {
            continue; // every key has the same digit here: a stable pass leaves the order as it is
        }
        let (hp, dp, src) = (SharedMut(hist.as_mut_ptr()), SharedMut(tmp.as_mut_ptr()), &keys[..]);
        run(nc, &|c| {
            // SAFETY: chunk c owns its offsets hist[c * B..(c + 1) * B]; the offsets give every key its own slot in tmp.
            let off = unsafe { std::slice::from_raw_parts_mut(hp.at(c * B), B) };
            for &k in &src[c * ch..((c + 1) * ch).min(n)] {
                let b = ((k >> shift) as usize) & (B - 1);
                unsafe { *dp.at(off[b]) = k };
                off[b] += 1;
            }
        });
        std::mem::swap(keys, tmp);
    }
}

/// [`walk_in`] over the sorted keys (their low 32 bits are the candidate indices): the same per-element operations as `cut_in`
/// (the exp terms, then each divided by the f64 total; after a top-p cut each divided by the cut's sum), the element-wise ones
/// on the host pool, the sums, the cut and the walk sequential in the same order. The same probabilities, so the same token.
fn walk_par(logits: &[f32], p: &SamplingParams, u: f32, keys: &[u64], probs: &mut Vec<f64>, chunks: usize) -> usize {
    use crate::par::{run, SharedMut};
    let m = keys.len();
    let id = |j: usize| keys[j] as u32 as usize;
    let maxl = logits[id(0)] as f64;
    let inv_t = 1.0 / p.temperature as f64;
    probs.resize(m, 0.0);
    let ch = m.div_ceil(chunks.max(1));
    let nc = m.div_ceil(ch);
    let pp = SharedMut(probs.as_mut_ptr());
    run(nc, &|c| {
        for j in c * ch..((c + 1) * ch).min(m) {
            // SAFETY: chunk c owns probs[c * ch..].
            unsafe { *pp.at(j) = ((logits[id(j)] as f64 - maxl) * inv_t).exp() };
        }
    });
    let total: f64 = probs.iter().sum();
    let div = |len: usize, by: f64| {
        let ch = len.div_ceil(chunks.max(1)).max(1);
        run(len.div_ceil(ch), &|c| {
            for j in c * ch..((c + 1) * ch).min(len) {
                // SAFETY: chunk c owns probs[c * ch..]; `len <= m`.
                unsafe { *pp.at(j) /= by };
            }
        });
    };
    div(m, total);
    let mut len = m;
    if p.top_p < 1.0 {
        let mut cum = 0.0;
        for (j, q) in probs.iter().enumerate() {
            cum += q;
            if cum >= p.top_p as f64 {
                len = j + 1;
                break;
            }
        }
        let s: f64 = probs[..len].iter().sum();
        div(len, s);
    }
    let target = u as f64;
    let mut acc = 0.0;
    for (j, q) in probs[..len].iter().enumerate() {
        acc += q;
        if target < acc {
            return id(j);
        }
    }
    id(len - 1)
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
    walk_in(logits, p, u, &mut idx, &mut Vec::new())
}

/// [`walk`] on the candidates in `idx` (cut in place), the probabilities in `probs`.
fn walk_in(logits: &[f32], p: &SamplingParams, u: f32, idx: &mut Vec<u32>, probs: &mut Vec<f64>) -> usize {
    cut_in(logits, p, idx, probs);
    let mut acc = 0.0;
    let target = u as f64;
    for (q, &i) in probs.iter().zip(idx.iter()) {
        acc += q;
        if target < acc {
            return i as usize;
        }
    }
    *idx.last().unwrap() as usize
}

/// Softmax over the sorted candidates `idx` and the top-p cut: the candidates `walk` can return and their probabilities.
fn cut(logits: &[f32], p: &SamplingParams, mut idx: Vec<u32>) -> (Vec<u32>, Vec<f64>) {
    let mut probs = Vec::new();
    cut_in(logits, p, &mut idx, &mut probs);
    (idx, probs)
}

/// [`cut`] in place: `idx` is truncated to the candidates, `probs` (cleared first) gets their probabilities.
fn cut_in(logits: &[f32], p: &SamplingParams, idx: &mut Vec<u32>, probs: &mut Vec<f64>) {
    let maxl = logits[idx[0] as usize] as f64;
    let inv_t = 1.0 / p.temperature as f64;
    probs.clear();
    probs.extend(idx.iter().map(|&i| ((logits[i as usize] as f64 - maxl) * inv_t).exp()));
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
}

/// A processed next-token distribution: the candidates [`sample_row`] can return for a row and their probabilities (after
/// temperature, top-k and top-p), in its walk order. Tokens not listed have probability 0.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Dist {
    pub ids: Vec<u32>,
    pub probs: Vec<f64>,
}

impl Dist {
    /// A single token with probability 1.
    pub fn point(id: u32) -> Dist {
        Dist { ids: vec![id], probs: vec![1.0] }
    }

    /// The probability of `id` (0 outside the candidates).
    pub fn prob(&self, id: u32) -> f64 {
        self.ids.iter().position(|&i| i == id).map_or(0.0, |j| self.probs[j])
    }

    /// The token at the cumulative position `u` in [0, 1) (the last candidate past the end, as `walk`).
    pub fn pick(&self, u: f64) -> u32 {
        let mut acc = 0.0;
        for (q, &i) in self.probs.iter().zip(&self.ids) {
            acc += q;
            if u < acc {
                return i;
            }
        }
        *self.ids.last().unwrap()
    }
}

/// The distribution [`sample_row`] draws from for `logits` under `p` (greedy: the argmax with probability 1). Exactly the
/// candidates and probabilities `sample_row`'s walk uses, so a token drawn from it has `sample_row`'s law.
pub fn dist(logits: &[f32], p: &SamplingParams) -> Dist {
    if p.temperature <= 0.0 {
        return Dist::point(argmax(logits) as u32);
    }
    let n = logits.len();
    let keep = if p.top_k > 0 && p.top_k < n { p.top_k } else { n };
    let idx: Vec<u32> = if n > u32::MAX as usize || logits.iter().any(|x| x.is_nan()) {
        let mut idx: Vec<u32> = (0..n as u32).collect();
        let cmp = |a: &u32, b: &u32| {
            logits[*b as usize].partial_cmp(&logits[*a as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b))
        };
        if keep < n {
            idx.select_nth_unstable_by(keep - 1, cmp);
            idx.truncate(keep);
        }
        idx.sort_unstable_by(cmp);
        idx
    } else {
        let mut keys: Vec<u64> = logits.iter().enumerate().map(|(i, &l)| (order_key(l) as u64) << 32 | i as u64).collect();
        if keep < n {
            keys.select_nth_unstable(keep - 1);
            keys.truncate(keep);
            keys.sort_unstable();
        } else if p.top_p >= 1.0 {
            // no top-p cut: the order does not change the distribution, so skip the sort
        } else {
            radix_sort_hi(&mut keys, &mut Vec::new());
        }
        if keep == n && p.top_p >= 1.0 {
            // softmax over the whole row, any order; `cut` reads the max from its first candidate, so put it first
            let mut idx: Vec<u32> = keys.iter().map(|&k| k as u32).collect();
            let best = argmax(logits);
            idx.swap(0, best);
            idx
        } else {
            keys.iter().map(|&k| k as u32).collect()
        }
    };
    let (ids, probs) = cut(logits, p, idx);
    Dist { ids, probs }
}

/// Per-request sampler: owns the seeded RNG.
pub struct Sampler {
    pub params: SamplingParams,
    rng: SplitMix64,
    /// [`sample_row_in`]'s buffers, kept across the request's steps
    scratch: SampleScratch,
}

impl Sampler {
    pub fn new(params: SamplingParams, seed: Option<u64>) -> Sampler {
        let seed = seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x1234_5678)
        });
        Sampler { params, rng: SplitMix64::new(seed), scratch: SampleScratch::default() }
    }

    pub fn sample(&mut self, logits: &[f32]) -> usize {
        let u = self.rng.next_f32();
        sample_row_in(logits, &self.params, u, &mut self.scratch)
    }

    /// One uniform draw in [0, 1) with 53 bits (speculative sampling's accept tests and draws).
    pub fn uniform(&mut self) -> f64 {
        (self.rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A token drawn from `d` with one [`Sampler::uniform`] draw.
    pub fn sample_dist(&mut self, d: &Dist) -> u32 {
        let u = self.uniform();
        d.pick(u)
    }

    /// [`Sampler::sample`] from the candidates a device-side top-k selected (`HeadMode::LastTopK`): `ids[j]` / `vals[j]` are
    /// the `j`-th largest logit of the row (value descending, ties by ascending index) and its token id, `k = params.top_k`
    /// of them. Returns the token `sample` would return on the whole row for the same draw: `sample_row` selects exactly these
    /// `k` candidates in exactly this order and everything after that touches only them (`walk`). One draw is consumed.
    pub fn sample_topk(&mut self, ids: &[u32], vals: &[f32]) -> usize {
        debug_assert!(self.params.top_k == vals.len() && ids.len() == vals.len());
        let u = self.rng.next_f32();
        sample_from_topk(ids, vals, &self.params, u)
    }
}

/// The host reference of device-side top-k (`Backend::topk_rows`): the `k` largest values of `row` and their indices in the
/// sampler's order (value descending, ties by ascending index, -0.0 equal to +0.0), i.e. the candidates [`sample_row`] selects
/// for `top_k == k`. `None` for a row holding a NaN.
pub fn topk_row(row: &[f32], k: usize) -> Option<(Vec<u32>, Vec<f32>)> {
    if row.iter().any(|x| x.is_nan()) {
        return None;
    }
    let mut keys: Vec<u64> = row.iter().enumerate().map(|(i, &l)| (order_key(l) as u64) << 32 | i as u64).collect();
    let k = k.min(keys.len());
    if k == 0 {
        return Some((Vec::new(), Vec::new()));
    }
    if k < keys.len() {
        keys.select_nth_unstable(k - 1);
        keys.truncate(k);
    }
    keys.sort_unstable();
    let ids: Vec<u32> = keys.iter().map(|&x| x as u32).collect();
    let vals = ids.iter().map(|&i| row[i as usize]).collect();
    Some((ids, vals))
}

/// [`sample_row`] on a row whose top-`p.top_k` candidates (best first) are `vals` with token ids `ids`.
pub fn sample_from_topk(ids: &[u32], vals: &[f32], p: &SamplingParams, u: f32) -> usize {
    if p.temperature <= 0.0 {
        return ids[0] as usize;
    }
    let idx: Vec<u32> = (0..vals.len() as u32).collect();
    ids[walk(vals, p, u, idx)] as usize
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

    /// One scratch reused across rows of different lengths and every parameter form gives `sample_row`'s token (lane fp8-decode).
    #[test]
    fn scratch_reuse_is_sample_row() {
        let mut rng = SplitMix64::new(7);
        let mut s = SampleScratch::default();
        for n in [1usize, 5, 64, 3000, 40000, 17] {
            for (t, k, tp) in [(1.0f32, 0usize, 1.0f32), (0.7, 20, 0.95), (0.7, 0, 0.95), (1.3, 5, 0.5), (0.0, 0, 1.0)] {
                let p = SamplingParams { temperature: t, top_k: k, top_p: tp };
                for _ in 0..4 {
                    let logits: Vec<f32> = (0..n).map(|_| (rng.next_f32() * 8.0).round() / 2.0 - 2.0).collect();
                    let u = rng.next_f32();
                    assert_eq!(sample_row_in(&logits, &p, u, &mut s), sample_row(&logits, &p, u), "n {n} t {t} k {k} tp {tp} u {u}");
                }
            }
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

    /// Host cost of one sampled decode step on an Inkling-2-Mini-sized row (lane ink-hostgap): the logit soft-cap and
    /// `sample_row_in` with a reused scratch, at the checkpoint's default (temperature 1, no top-k / top-p) and others. Prints
    /// medians; asserts nothing. `cargo test --release -p intarsia-core --lib sampler_host_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn sampler_host_cost() {
        let n = 200058;
        let mut rng = SplitMix64::new(3);
        let raw = bf16_row(&mut rng, n, 3.0);
        let med = |f: &mut dyn FnMut()| {
            let mut v: Vec<f64> = (0..41)
                .map(|_| {
                    let t = std::time::Instant::now();
                    f();
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        // the pool as Inkling-2's load sets it (`INTARSIA_HOST_THREADS=1` for the one-thread rows everywhere)
        crate::par::set_model_default(true);
        let mut row = raw.clone();
        let cap = 30.0f32;
        let ms = med(&mut || {
            row.copy_from_slice(&raw);
            for x in row.iter_mut() {
                *x = cap * (*x / cap).tanh();
            }
            std::hint::black_box(&row);
        });
        println!("softcap (copy + tanh) n={n}: {ms:.3} ms (one thread)");
        let ms = med(&mut || {
            row.copy_from_slice(&raw);
            softcap(&mut row, cap);
            std::hint::black_box(&row);
        });
        println!("softcap (copy + tanh) n={n}: {ms:.3} ms (host pool, {} threads)", crate::par::threads());
        let ms = med(&mut || {
            std::hint::black_box(argmax(std::hint::black_box(&row)));
        });
        println!("argmax n={n}: {ms:.3} ms");
        let mut sc = SampleScratch::default();
        for (t, k, tp) in [(1.0f32, 0usize, 1.0f32), (0.7, 0, 1.0), (1.0, 0, 0.95), (0.7, 20, 0.95), (1.0, 64, 1.0)] {
            let p = SamplingParams { temperature: t, top_k: k, top_p: tp };
            let mut u = 0.0f32;
            let ms = med(&mut || {
                u = (u + 0.37) % 1.0;
                std::hint::black_box(sample_row_seq(&row, &p, u, &mut sc));
            });
            println!("sample_row_in n={n} T={t} top_k={k} top_p={tp}: {ms:.3} ms (one thread)");
            let ms = med(&mut || {
                u = (u + 0.37) % 1.0;
                std::hint::black_box(sample_row_in(&row, &p, u, &mut sc));
            });
            println!("sample_row_in n={n} T={t} top_k={k} top_p={tp}: {ms:.3} ms (host pool, {} threads)", crate::par::threads());
        }
    }

    /// Lane ink-hostgap: the threaded path (`sample_row_par`, any chunk count, forced on short rows too) picks exactly the token
    /// of the one-thread path for every draw: ties, signed zeros, -inf, all-equal rows, top-k / top-p / temperature, NaN rows
    /// (both fall back to the comparator path). And the threaded soft-cap is bitwise the one-thread loop.
    #[test]
    fn threaded_sampler_matches_the_one_thread_path() {
        let mut rng = SplitMix64::new(17);
        let us = [0.0f32, 1e-7, 0.25, 0.5, 0.9, 0.999, 1.0 - 1.0 / (1u64 << 24) as f32];
        let mut rows: Vec<Vec<f32>> = Vec::new();
        for &(n, scale) in &[(1usize, 1.0f32), (2, 0.0), (7, 0.5), (300, 2.0), (5000, 3.0), (40000, 3.0), (200058, 3.0)] {
            rows.push(bf16_row(&mut rng, n, scale));
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
        rows.push(vec![2.0; 70000]);
        rows.push(vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY, 1.0, 5.0]);
        let mut nan = bf16_row(&mut rng, 50000, 2.0);
        nan[777] = f32::NAN;
        rows.push(nan);
        let (mut s1, mut s2) = (SampleScratch::default(), SampleScratch::default());
        for row in &rows {
            let n = row.len();
            for &t in &[0.3f32, 0.7, 1.0, 1.5] {
                // short rows: every setting; long rows (time, also in debug builds): fewer top-k values, chunk counts and draws
                let long = n > 5000;
                let ks: Vec<usize> = if long { vec![0, 64, n] } else { vec![0, 1, 20, 64, n.saturating_sub(1), n] };
                for &k in &ks {
                    for &tp in &[1.0f32, 0.95, 0.5] {
                        let p = SamplingParams { temperature: t, top_k: k, top_p: tp };
                        for (ci, &chunks) in [1usize, 2, 3, 7, 16].iter().enumerate() {
                            if long && ci % 2 == 1 {
                                continue;
                            }
                            let mut draws: Vec<f32> = us.to_vec();
                            if !long {
                                draws.extend((0..4).map(|_| rng.next_f32()));
                            }
                            for &u in &draws {
                                let want = sample_row_seq(row, &p, u, &mut s1);
                                assert_eq!(sample_row_par(row, &p, u, &mut s2, chunks), want, "n {n} t {t} k {k} tp {tp} u {u} chunks {chunks}");
                            }
                        }
                    }
                }
            }
        }
        // the public entry takes the threaded path for a long row (pool on, as Inkling-2's load sets it) and still agrees, also
        // with top-k on (which it hands to the one-thread path)
        crate::par::set_model_default(true);
        let row = &rows[13];
        assert_eq!(row.len(), 200058);
        let p = SamplingParams { temperature: 1.0, top_k: 0, top_p: 1.0 };
        for _ in 0..32 {
            let u = rng.next_f32();
            assert_eq!(sample_row_in(row, &p, u, &mut s2), sample_row_seq(row, &p, u, &mut s1));
            let pk = SamplingParams { temperature: 0.7, top_k: 20, top_p: 0.95 };
            assert_eq!(sample_row_in(row, &pk, u, &mut s2), sample_row_seq(row, &pk, u, &mut s1));
        }
        // soft-cap: threaded == one-thread loop, bit for bit
        for n in [10usize, PAR_MIN - 1, PAR_MIN, 200058] {
            let raw: Vec<f32> = (0..n).map(|_| rng.normal() * 20.0).chain([f32::NEG_INFINITY, f32::INFINITY, -0.0, 0.0]).collect();
            let want: Vec<u32> = raw.iter().map(|&x| (30.0f32 * (x / 30.0).tanh()).to_bits()).collect();
            let mut got = raw.clone();
            softcap(&mut got, 30.0);
            assert!(got.iter().map(|x| x.to_bits()).eq(want.iter().copied()), "softcap n {n}");
        }
    }

    /// Where the threaded sampler's time goes (lane ink-hostgap): each phase of `sample_row_par` timed alone, medians of 41, on the
    /// pool as Inkling-2's load sets it. Prints; asserts nothing.
    /// `cargo test --release -p intarsia-core --lib sampler_par_phases -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn sampler_par_phases() {
        use crate::par::{run, SharedMut};
        crate::par::set_model_default(true);
        let n = 200058;
        let mut rng = SplitMix64::new(3);
        let mut row = bf16_row(&mut rng, n, 3.0);
        softcap(&mut row, 30.0);
        let chunks = 2 * crate::par::threads();
        let med = |f: &mut dyn FnMut()| {
            let mut v: Vec<f64> = (0..41)
                .map(|_| {
                    let t = std::time::Instant::now();
                    f();
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let mut keys = vec![0u64; n];
        let (mut tmp, mut hist) = (Vec::new(), Vec::new());
        let ch = n.div_ceil(chunks);
        let t_keys = med(&mut || {
            let kp = SharedMut(keys.as_mut_ptr());
            run(n.div_ceil(ch), &|c| {
                for i in c * ch..((c + 1) * ch).min(n) {
                    unsafe { *kp.at(i) = (order_key(row[i]) as u64) << 32 | i as u64 };
                }
            });
        });
        let unsorted = keys.clone();
        let t_sort = med(&mut || {
            keys.copy_from_slice(&unsorted);
            radix_sort_hi_par(&mut keys, &mut tmp, &mut hist, chunks);
        });
        let t_copy = med(&mut || keys.copy_from_slice(&unsorted));
        radix_sort_hi_par(&mut keys, &mut tmp, &mut hist, chunks);
        let mut probs = vec![0f64; n];
        let maxl = row[keys[0] as u32 as usize] as f64;
        let t_exp = med(&mut || {
            let pp = SharedMut(probs.as_mut_ptr());
            run(n.div_ceil(ch), &|c| {
                for j in c * ch..((c + 1) * ch).min(n) {
                    unsafe { *pp.at(j) = (row[keys[j] as u32 as usize] as f64 - maxl).exp() };
                }
            });
        });
        let mut total = 0.0;
        let t_sum = med(&mut || total = std::hint::black_box(probs.iter().sum::<f64>()));
        let t_div = med(&mut || {
            let pp = SharedMut(probs.as_mut_ptr());
            run(n.div_ceil(ch), &|c| {
                for j in c * ch..((c + 1) * ch).min(n) {
                    unsafe { *pp.at(j) = *pp.at(j) / total * total };
                }
            });
        });
        let qs: Vec<f64> = probs.iter().map(|e| e / total).collect();
        let mut depth = Vec::new();
        let t_walk = med(&mut || {
            let target = rng.next_f32() as f64;
            let mut acc = 0.0;
            let mut d = qs.len();
            for (j, q) in qs.iter().enumerate() {
                acc += q;
                if target < acc {
                    d = j;
                    break;
                }
            }
            depth.push(d);
        });
        let t_empty = med(&mut || run(chunks, &|_| {}));
        depth.sort();
        println!("threads {} chunks {chunks} n {n}", crate::par::threads());
        println!("keys {t_keys:.3} ms | radix sort {:.3} ms (incl. copy {t_copy:.3}) | exp {t_exp:.3} | f64 total {t_sum:.3} | divide {t_div:.3} | walk {t_walk:.3} (median depth {}) | empty run {t_empty:.4}", t_sort - t_copy, depth[depth.len() / 2]);
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

    /// Sampling from the top-k candidates is the token `sample_row` picks on the whole row, for every draw (R3b).
    #[test]
    fn topk_candidates_sample_like_the_whole_row() {
        let mut rng = SplitMix64::new(9);
        for &(n, scale) in &[(100usize, 1.0f32), (5000, 3.0), (248320, 3.0)] {
            let mut rows = vec![bf16_row(&mut rng, n, scale)];
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
            for row in &rows {
                for &k in &[1usize, 2, 20, 40, 64] {
                    // the device kernel's output: the k best in the sampler's order
                    let mut order: Vec<u32> = (0..n as u32).collect();
                    let cmp = |a: &u32, b: &u32| row[*b as usize].partial_cmp(&row[*a as usize]).unwrap().then(a.cmp(b));
                    order.select_nth_unstable_by(k - 1, cmp);
                    order.truncate(k);
                    order.sort_unstable_by(cmp);
                    let vals: Vec<f32> = order.iter().map(|&i| row[i as usize]).collect();
                    for &t in &[0.3f32, 0.7, 1.0] {
                        for &tp in &[1.0f32, 0.9] {
                            let p = SamplingParams { temperature: t, top_k: k, top_p: tp };
                            for _ in 0..16 {
                                let u = rng.next_f32();
                                assert_eq!(sample_from_topk(&order, &vals, &p, u), sample_row(row, &p, u), "n {n} k {k} t {t} tp {tp} u {u}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// `dist` lists exactly what `sample_row` draws: every candidate's cumulative interval maps to it, the probabilities sum to
    /// 1, and the unsorted fast path (no top-k, no top-p) has the sorted path's probabilities.
    #[test]
    fn dist_is_what_sample_row_draws() {
        let mut rng = SplitMix64::new(21);
        for &(n, scale) in &[(1usize, 1.0f32), (7, 0.5), (300, 2.0), (5000, 3.0)] {
            let mut rows = vec![bf16_row(&mut rng, n, scale)];
            let mut r = bf16_row(&mut rng, n, scale);
            for (i, x) in r.iter_mut().enumerate() {
                if i % 5 == 3 {
                    *x = f32::NEG_INFINITY;
                }
            }
            rows.push(r);
            for row in &rows {
                for &t in &[0.0f32, 0.7, 1.0, 1.5] {
                    for &k in &[0usize, 1, 20, n] {
                        for &tp in &[1.0f32, 0.95, 0.5] {
                            let p = SamplingParams { temperature: t, top_k: k, top_p: tp };
                            let d = dist(row, &p);
                            let s: f64 = d.probs.iter().sum();
                            assert!((s - 1.0).abs() < 1e-9, "sum {s}");
                            if t <= 0.0 {
                                assert_eq!(d.ids, vec![argmax(row) as u32]);
                                continue;
                            }
                            // the fast path (no top-k, no top-p) lists the candidates unsorted: same law, other intervals
                            let sorted = (k > 0 && k < n) || tp < 1.0;
                            let mut cum = 0.0f64;
                            for (&id, &q) in d.ids.iter().zip(&d.probs).filter(|_| sorted) {
                                let mid = cum + q / 2.0;
                                cum += q;
                                if q < 1e-5 {
                                    continue; // below the f32 draw's resolution
                                }
                                assert_eq!(sample_row(row, &p, mid as f32) as u32, id, "n {n} t {t} k {k} tp {tp}");
                                assert_eq!(d.pick(mid), id);
                            }
                            if !sorted {
                                // against the sorted order
                                let mut idx: Vec<u32> = (0..n as u32).collect();
                                idx.sort_by(|a, b| row[*b as usize].partial_cmp(&row[*a as usize]).unwrap().then(a.cmp(b)));
                                let (ids2, probs2) = cut(row, &p, idx);
                                for (&id, &q) in ids2.iter().zip(&probs2) {
                                    assert!((d.prob(id) - q).abs() < 1e-12);
                                }
                            }
                        }
                    }
                }
            }
        }
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
