//! The Engram n-gram hash, as used by the table tier. The hash itself is `intarsia_core::dsv41::engram::NgramHasher`
//! (the one implementation in the tree; bit-exact with DeepSeek's `engram.py`, checked by `tests/hash_parity.rs` on
//! 3,000 sequences). This module adds what the tier needs around it: the DSv4.1 constants, the per-column row
//! offsets, the bucket-boundary rank split and an O(1) per-token decode state.
//!
//! Per position and Engram layer the hash yields `(max_ngram - 1) * n_heads` row ids (DSv4.1: 3 x 8 = 24), column
//! `c = (ngram - 2) * n_heads + head`; column `c` owns the disjoint rows `[offsets[c], offsets[c] + primes[c])`.

pub use intarsia_core::dsv41::engram::{EngramLayout, NgramHasher, DEAD};
use std::ops::Range;

/// DeepSeek V4.1 Flash `config.json` (inference/): `engram_*` keys.
pub mod dsv41 {
    pub const LAYER_IDS: [usize; 2] = [1, 14];
    pub const MAX_NGRAM: usize = 4;
    pub const N_HEADS: usize = 8;
    pub const HEAD_DIM: usize = 256;
    pub const ENGRAM_VOCAB_SIZE: u64 = 16_000_000;
    pub const NUM_EMBEDDINGS: [u64; 2] = [384_006_168, 384_016_682];
    pub const COMPRESSED_VOCAB: u32 = 99_092;
    /// Compressed id of `engram_pad_id` (raw 2) under the official token map.
    pub const PAD_COMPRESSED: i64 = 2;
    /// `compute_hash_multipliers` output (numpy PCG64 `default_rng(10007 * layer_id)`), as in
    /// `tools/gen_dsv41_engram_params.py`'s fixture; re-checked by `tests/hash_parity.rs`.
    pub const MULTIPLIERS: [[i64; 4]; 2] = [
        [76_632_096_046_245, 4_839_876_093_313, 35_959_672_319_349, 73_987_337_458_391],
        [67_716_810_739_261, 51_510_806_800_915, 30_921_347_202_721, 82_619_226_485_591],
    ];
}

/// The core hasher plus the tier's view of the table geometry.
#[derive(Clone, Debug)]
pub struct HashLayout {
    pub hasher: NgramHasher,
    /// `[layer][column]` first row of each column's range.
    pub offsets: Vec<Vec<u64>>,
}

impl HashLayout {
    /// DeepSeek V4.1 Flash.
    pub fn dsv41() -> Self {
        let layout = EngramLayout::new(
            &dsv41::LAYER_IDS,
            &dsv41::NUM_EMBEDDINGS,
            dsv41::MAX_NGRAM,
            dsv41::N_HEADS,
            dsv41::HEAD_DIM,
            dsv41::ENGRAM_VOCAB_SIZE,
        )
        .expect("DSv4.1 Engram layout");
        Self::new(NgramHasher {
            layout,
            multipliers: dsv41::MULTIPLIERS.iter().map(|m| m.to_vec()).collect(),
            pad_id: dsv41::PAD_COMPRESSED,
        })
    }

    pub fn new(hasher: NgramHasher) -> Self {
        let offsets = hasher
            .layout
            .primes
            .iter()
            .map(|layer| {
                let mut acc = 0u64;
                layer
                    .iter()
                    .flatten()
                    .map(|&p| {
                        let o = acc;
                        acc += p;
                        o
                    })
                    .collect()
            })
            .collect();
        Self { hasher, offsets }
    }

    pub fn layout(&self) -> &EngramLayout {
        &self.hasher.layout
    }
    pub fn n_layers(&self) -> usize {
        self.layout().layer_ids.len()
    }
    pub fn n_heads(&self) -> usize {
        self.layout().n_heads
    }
    pub fn layer_ids(&self) -> &[usize] {
        &self.layout().layer_ids
    }
    /// Row ids per position per layer (DSv4.1: 24).
    pub fn cols(&self) -> usize {
        self.layout().cols()
    }
    /// Row ids per position over all Engram layers (DSv4.1: 48).
    pub fn rows_per_token(&self) -> usize {
        self.n_layers() * self.cols()
    }
    pub fn layer_index(&self, layer_id: usize) -> Option<usize> {
        self.layer_ids().iter().position(|&l| l == layer_id)
    }
    pub fn num_embeddings(&self, layer: usize) -> u64 {
        self.layout().rows[layer]
    }

    /// Columns a rank owns under the bucket-boundary split: contiguous column groups, so every rank reads exactly
    /// `cols / world` rows per layer per token. At TP=2 rank 0 owns columns 0-11 (2-gram heads 0-7, 3-gram heads 0-3)
    /// and rank 1 columns 12-23. The existing packed shards (`engram-l*-r*of2.bin`) use this split; model.py's
    /// `ceil(rows / world)` split differs by a few thousand rows and gives the same output.
    pub fn rank_columns(&self, rank: usize, world: usize) -> Range<usize> {
        assert!(world >= 1 && rank < world);
        let c = self.cols();
        (rank * c / world)..((rank + 1) * c / world)
    }

    /// Table rows a rank owns under the bucket-boundary split.
    pub fn rank_rows(&self, layer: usize, rank: usize, world: usize) -> Range<u64> {
        let cols = self.rank_columns(rank, world);
        let at = |c: usize| if c == self.cols() { self.num_embeddings(layer) } else { self.offsets[layer][c] };
        at(cols.start)..at(cols.end)
    }

    /// Hash a whole sequence from its start (prefill). `dead[i]` marks image tokens. Writes `ids.len() *
    /// rows_per_token()` row ids, ordered `[position][layer][column]`.
    pub fn hash_sequence(&self, ids: &[u32], dead: Option<&[bool]>, out: &mut Vec<u64>) {
        let c: Vec<i64> =
            ids.iter().enumerate().map(|(i, &t)| if dead.is_some_and(|d| d[i]) { DEAD } else { t as i64 }).collect();
        let rows = self.hasher.hash(&c, 0, c.len()).expect("engram hash");
        out.clear();
        out.extend(rows.iter().map(|&r| r as u64));
    }
}

/// One sequence's hash history (the last `max_ngram - 1` compressed ids), carried across prefill and decode, so a
/// decode step hashes in O(1): `push` the sampled token, get its row ids.
#[derive(Clone, Copy, Debug)]
pub struct NgramState {
    /// Oldest first; `DEAD` = dead token or before the sequence start (both stop the look-back).
    hist: [i64; 3],
}

impl Default for NgramState {
    fn default() -> Self {
        Self { hist: [DEAD; 3] }
    }
}

impl NgramState {
    /// Append a token (compressed id, `None` for a dead/image token) and write its row ids (`[layer][column]`).
    pub fn push(&mut self, layout: &HashLayout, token: Option<u32>, out: &mut [u64]) {
        let m = layout.layout().max_ngram;
        assert!((2..=4).contains(&m), "NgramState keeps at most 3 tokens of history");
        let cur = token.map_or(DEAD, |t| t as i64);
        let w = [self.hist[0], self.hist[1], self.hist[2], cur];
        // the last m ids; a position before the start reads as DEAD, which blocks the look-back like `pos < shift`
        let rows = layout.hasher.hash(&w[4 - m..], m - 1, 1).expect("engram hash");
        for (o, r) in out.iter_mut().zip(rows) {
            *o = r as u64;
        }
        self.hist = [self.hist[1], self.hist[2], cur];
    }

    /// Row ids a candidate next token would need, without appending it (for speculative prefetch of the top-k
    /// candidates before sampling).
    pub fn peek(&self, layout: &HashLayout, token: Option<u32>, out: &mut [u64]) {
        let mut s = *self;
        s.push(layout, token, out);
    }

    /// Forget the history (a new sequence).
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_config() {
        let l = HashLayout::dsv41();
        assert_eq!(l.cols(), 24);
        assert_eq!(l.rows_per_token(), 48);
        assert_eq!(l.layout().primes[0][0][0], 16_000_057);
        assert_eq!(l.layout().primes[1][2][7], 16_000_889);
        assert_eq!(l.offsets[0][12], 192_001_740);
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
}
