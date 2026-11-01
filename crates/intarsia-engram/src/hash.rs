//! The Engram n-gram hash: token history -> table row ids.
//!
//! Ported from DeepSeek's official `inference/engram.py` (`EngramLayout`, `compute_hash_multipliers`,
//! `NgramHashState.forward`; MIT, see `LICENSE-DeepSeek-MIT.txt`). Bit-exact against that code on 3,000
//! generated sequences plus real text (`tests/hash_parity.rs`, fixture made by
//! `tests/fixtures/gen_hash_fixture.py`).
//!
//! Per position and Engram layer the hash yields `(max_ngram - 1) * n_heads` row ids (DSv4.1: 3 x 8 = 24),
//! column `c = (ngram - 2) * n_heads + head`. All heads of an n-gram share one XOR key and differ only in their
//! prime modulus; each column owns the disjoint row range `[offsets[c], offsets[c] + primes[c])` of its layer's
//! table. Row addresses depend only on token ids, never on hidden state, so they are known the moment a token id
//! reaches the host.

use std::ops::Range;

/// Most look-back the hasher keeps (n-gram size). DSv4.1 uses 4.
pub const MAX_NGRAM: usize = 4;

/// DeepSeek V4.1 Flash `config.json` (inference/): `engram_*` keys.
pub mod dsv41 {
    pub const LAYER_IDS: [u32; 2] = [1, 14];
    pub const MAX_NGRAM: usize = 4;
    pub const N_HEADS: usize = 8;
    pub const HEAD_DIM: usize = 256;
    pub const ENGRAM_VOCAB_SIZE: u64 = 16_000_000;
    pub const NUM_EMBEDDINGS: [u64; 2] = [384_006_168, 384_016_682];
    pub const COMPRESSED_VOCAB: u32 = 99_092;
    /// `engram_pad_id` (raw token id 2); its compressed id is also 2 under the official token map.
    pub const PAD_RAW: u32 = 2;
    pub const PAD_COMPRESSED: u32 = 2;
    /// `compute_hash_multipliers(layer_ids, 4, 99092)`: `numpy.random.default_rng(10007 * layer_id)
    /// .integers(0, (i64::MAX // 99092) // 2, 4) * 2 + 1`. Taken from the official code's output (not re-derived:
    /// that would mean porting numpy's PCG64 and bounded-integer draw); the parity fixture re-checks them.
    pub const MULTIPLIERS: [[u64; 4]; 2] = [
        [76_632_096_046_245, 4_839_876_093_313, 35_959_672_319_349, 73_987_337_458_391],
        [67_716_810_739_261, 51_510_806_800_915, 30_921_347_202_721, 82_619_226_485_591],
    ];
}

/// Geometry and constants of the hash tables (engram.py `EngramLayout` plus the per-layer multipliers).
#[derive(Clone, Debug)]
pub struct HashLayout {
    pub layer_ids: Vec<u32>,
    pub max_ngram: usize,
    pub n_heads: usize,
    /// Compressed id that fills n-gram slots with no history (`token_map[engram_pad_id]`).
    pub pad: u32,
    /// `[layer][lookback]`, odd, small enough that `id * m` cannot overflow i64.
    pub multipliers: Vec<[u64; MAX_NGRAM]>,
    /// `[layer][column]` bucket modulus.
    pub primes: Vec<Vec<u64>>,
    /// `[layer][column]` first row of the column's range (cumulative sum of the primes before it).
    pub offsets: Vec<Vec<u64>>,
    /// `[layer]` table rows (sum of the layer's primes).
    pub num_embeddings: Vec<u64>,
}

impl HashLayout {
    /// DeepSeek V4.1 Flash. Primes are computed (and checked against `engram_num_embeddings`).
    pub fn dsv41() -> Self {
        let l = Self::new(
            &dsv41::LAYER_IDS,
            dsv41::MAX_NGRAM,
            dsv41::N_HEADS,
            dsv41::ENGRAM_VOCAB_SIZE,
            dsv41::PAD_COMPRESSED,
            dsv41::MULTIPLIERS.to_vec(),
        );
        assert_eq!(l.num_embeddings, dsv41::NUM_EMBEDDINGS, "Engram prime search disagrees with config");
        l
    }

    /// `EngramLayout.from_args`: per layer, per n-gram size, `n_heads` primes searched upward from
    /// `engram_vocab_size - 1`, never reusing one (the `seen` set spans all layers).
    pub fn new(
        layer_ids: &[u32],
        max_ngram: usize,
        n_heads: usize,
        engram_vocab_size: u64,
        pad: u32,
        multipliers: Vec<[u64; MAX_NGRAM]>,
    ) -> Self {
        assert!((2..=MAX_NGRAM).contains(&max_ngram));
        assert_eq!(multipliers.len(), layer_ids.len());
        let mut seen = std::collections::HashSet::new();
        let mut primes = Vec::new();
        let mut offsets = Vec::new();
        let mut num_embeddings = Vec::new();
        for _ in layer_ids {
            let mut layer = Vec::new();
            for _ in 0..max_ngram - 1 {
                let mut current = engram_vocab_size - 1;
                for _ in 0..n_heads {
                    current = next_prime(current, &seen);
                    seen.insert(current);
                    layer.push(current);
                }
            }
            let mut off = Vec::with_capacity(layer.len());
            let mut acc = 0u64;
            for &p in &layer {
                off.push(acc);
                acc += p;
            }
            num_embeddings.push(acc);
            primes.push(layer);
            offsets.push(off);
        }
        Self { layer_ids: layer_ids.to_vec(), max_ngram, n_heads, pad, multipliers, primes, offsets, num_embeddings }
    }

    /// Row ids per position per layer (DSv4.1: 24).
    pub fn cols(&self) -> usize {
        (self.max_ngram - 1) * self.n_heads
    }

    /// Row ids per position over all Engram layers (DSv4.1: 48).
    pub fn rows_per_token(&self) -> usize {
        self.layer_ids.len() * self.cols()
    }

    /// Index of an Engram layer id (1 -> 0, 14 -> 1 for DSv4.1).
    pub fn layer_index(&self, layer_id: u32) -> Option<usize> {
        self.layer_ids.iter().position(|&l| l == layer_id)
    }

    /// Columns a rank owns under the bucket-boundary split: contiguous column groups, so every rank reads exactly
    /// `cols / world` rows per layer per token. At TP=2 rank 0 owns columns 0-11 (2-gram heads 0-7, 3-gram heads
    /// 0-3) and rank 1 columns 12-23. This is the split the existing packed shards (`engram-l*-r*of2.bin`, the
    /// vLLM DSv4.1 kit's "hash-head ranges") use; it differs from model.py's `ceil(rows / world)` split by a
    /// few thousand rows but gives the same output (masked rows contribute zeros to the all-reduce).
    pub fn rank_columns(&self, rank: usize, world: usize) -> Range<usize> {
        assert!(world >= 1 && rank < world);
        let c = self.cols();
        (rank * c / world)..((rank + 1) * c / world)
    }

    /// Table rows a rank owns under the bucket-boundary split.
    pub fn rank_rows(&self, layer: usize, rank: usize, world: usize) -> Range<u64> {
        let cols = self.rank_columns(rank, world);
        let lo = if cols.start == self.cols() { self.num_embeddings[layer] } else { self.offsets[layer][cols.start] };
        let hi = if cols.end == self.cols() { self.num_embeddings[layer] } else { self.offsets[layer][cols.end] };
        lo..hi
    }

    /// Hash one position. `window[s]` is the compressed id `s` positions back (`window[0]` the current token);
    /// `None` is a dead token (an image span) or a position before the sequence start. Writes `rows_per_token()`
    /// row ids into `out`, ordered `[layer][column]`.
    pub fn hash_window(&self, window: &[Option<u32>; MAX_NGRAM], out: &mut [u64]) {
        let cols = self.cols();
        assert!(out.len() >= self.rows_per_token());
        // look-back stops at the first dead / missing slot: it and everything older become pad
        let mut tok = [0u64; MAX_NGRAM];
        let mut blocked = false;
        for s in 0..self.max_ngram {
            blocked |= window[s].is_none();
            tok[s] = if blocked { self.pad as u64 } else { window[s].unwrap() as u64 };
        }
        for (l, m) in self.multipliers.iter().enumerate() {
            let primes = &self.primes[l];
            let offs = &self.offsets[l];
            let dst = &mut out[l * cols..(l + 1) * cols];
            // products stay below 2^63 (multiplier bound), so u64 arithmetic equals the reference's int64
            let mut rolling = tok[0].wrapping_mul(m[0]);
            for i in 1..self.max_ngram {
                rolling ^= tok[i].wrapping_mul(m[i]);
                for h in 0..self.n_heads {
                    let c = (i - 1) * self.n_heads + h;
                    dst[c] = rolling % primes[c] + offs[c];
                }
            }
        }
    }

    /// Hash a whole sequence from its start (prefill). `dead[i]` marks image tokens. Writes
    /// `ids.len() * rows_per_token()` row ids, ordered `[position][layer][column]`.
    pub fn hash_sequence(&self, ids: &[u32], dead: Option<&[bool]>, out: &mut Vec<u64>) {
        let mut st = NgramState::default();
        let n = self.rows_per_token();
        out.clear();
        out.resize(ids.len() * n, 0);
        for (i, &t) in ids.iter().enumerate() {
            let tok = if dead.is_some_and(|d| d[i]) { None } else { Some(t) };
            st.push(self, tok, &mut out[i * n..(i + 1) * n]);
        }
    }
}

/// One sequence's hash history (the last `MAX_NGRAM - 1` compressed ids), carried across prefill and decode.
/// Decode is O(1) per token: `push` the sampled token, get its row ids.
#[derive(Clone, Copy, Debug, Default)]
pub struct NgramState {
    /// `hist[s]` = id `s + 1` positions back; `None` = dead or before the start.
    hist: [Option<u32>; MAX_NGRAM - 1],
}

impl NgramState {
    /// Append a token (compressed id, `None` for a dead/image token) and write its row ids (`[layer][column]`).
    pub fn push(&mut self, layout: &HashLayout, token: Option<u32>, out: &mut [u64]) {
        let w = [token, self.hist[0], self.hist[1], self.hist[2]];
        layout.hash_window(&w, out);
        self.hist = [token, self.hist[0], self.hist[1]];
    }

    /// Forget the history (a new sequence).
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// engram.py `find_next_prime`: the smallest prime above `start` not in `seen`.
fn next_prime(start: u64, seen: &std::collections::HashSet<u64>) -> u64 {
    let mut c = start + 1;
    while !is_prime(c) || seen.contains(&c) {
        c += 1;
    }
    c
}

/// Deterministic Miller-Rabin for u64 (these bases are exact below 2^64).
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n % p == 0 {
            return n == p;
        }
    }
    let mut d = n - 1;
    let mut r = 0;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }
    let mulmod = |a: u64, b: u64| ((a as u128 * b as u128) % n as u128) as u64;
    let powmod = |mut b: u64, mut e: u64| {
        let mut acc = 1u64;
        b %= n;
        while e > 0 {
            if e & 1 == 1 {
                acc = mulmod(acc, b);
            }
            b = mulmod(b, b);
            e >>= 1;
        }
        acc
    };
    'outer: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = powmod(a, d);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mulmod(x, x);
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_config() {
        let l = HashLayout::dsv41();
        assert_eq!(l.cols(), 24);
        assert_eq!(l.rows_per_token(), 48);
        assert_eq!(l.primes[0][0], 16_000_057);
        assert_eq!(l.primes[1][23], 16_000_889);
    }

    #[test]
    fn bucket_split_matches_packed_shards() {
        let l = HashLayout::dsv41();
        // headers of the existing engram-l{1,14}-r{0,1}of2.bin shards (spark1, spark2)
        assert_eq!(l.rank_rows(0, 0, 2), 0..192_001_740);
        assert_eq!(l.rank_rows(0, 1, 2), 192_001_740..384_006_168);
        assert_eq!(l.rank_rows(1, 0, 2), 0..192_007_016);
        assert_eq!(l.rank_rows(1, 1, 2), 192_007_016..384_016_682);
        assert_eq!(l.rank_rows(1, 0, 1), 0..384_016_682);
    }

    #[test]
    fn primes() {
        let small: Vec<u64> = (0..50).filter(|&n| is_prime(n)).collect();
        assert_eq!(small, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]);
        assert!(is_prime(18_446_744_073_709_551_557));
        assert!(!is_prime(3_215_031_751)); // strong pseudoprime to bases 2, 3, 5, 7
    }
}
