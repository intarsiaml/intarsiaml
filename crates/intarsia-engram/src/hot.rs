//! The hot tier: Engram rows kept in RAM so they never touch the disk.
//!
//! Two regions in one slab of 264 B records:
//! - **pinned**: rows chosen offline by frequency on a representative corpus ([`HotTier::pin`], e.g. from a
//!   profile via [`crate::EngramStore::preload`]). Never evicted. The scope doc's measurement (and its LRU
//!   comparison) says a static frequency-pinned set beats recency caching for these Zipf-shaped lookups.
//! - **dynamic**: TinyLFU admission (a count-min sketch of recent lookup frequency; a missed row is admitted
//!   only if it is seen more often than a sampled victim), sampled-LFU eviction. Adapts to the current
//!   conversation's n-grams. Its share of the capacity is `dynamic_fraction`.
//!
//! Keys are `(layer index, global row)` packed into a u64 by [`key`].

use crate::layout::RECORD_BYTES;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Packs a layer index and a global row into one key.
#[inline]
pub fn key(layer: usize, row: u64) -> u64 {
    ((layer as u64) << 56) | row
}

/// A fast integer hasher for row keys.
#[derive(Default, Clone, Copy)]
pub struct FxHasher(u64);
impl Hasher for FxHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}
pub type FxBuild = BuildHasherDefault<FxHasher>;

#[derive(Clone, Copy, Debug)]
pub struct HotTierConfig {
    /// Bytes of row data (264 B per row). 0 disables the tier.
    pub capacity_bytes: usize,
    /// Share of the capacity for TinyLFU-admitted rows; the rest is for pinned rows.
    pub dynamic_fraction: f32,
    /// log2 of the count-min sketch width (4 rows of u8 counters).
    pub sketch_log2: u32,
}

impl Default for HotTierConfig {
    fn default() -> Self {
        Self { capacity_bytes: 0, dynamic_fraction: 0.25, sketch_log2: 22 }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HotStats {
    pub lookups: u64,
    pub hits: u64,
    pub pinned_rows: u64,
    pub dynamic_rows: u64,
    pub admitted: u64,
    pub rejected: u64,
    pub evicted: u64,
    pub bytes: u64,
}

impl HotStats {
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.lookups as f64
        }
    }
}

struct Sketch {
    counters: Vec<u8>,
    mask: u64,
    adds: u64,
    reset_at: u64,
}

impl Sketch {
    fn new(log2: u32, reset_at: u64) -> Self {
        let w = 1usize << log2;
        Self { counters: vec![0; 4 * w], mask: w as u64 - 1, adds: 0, reset_at }
    }
    #[inline]
    fn idx(&self, k: u64, i: u64) -> usize {
        let h = (k ^ (i.wrapping_mul(0x9E37_79B9_7F4A_7C15))).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let h = h ^ (h >> 31);
        (i as usize) * (self.mask as usize + 1) + (h & self.mask) as usize
    }
    fn add(&mut self, k: u64) {
        for i in 0..4 {
            let j = self.idx(k, i);
            self.counters[j] = self.counters[j].saturating_add(1);
        }
        self.adds += 1;
        if self.adds >= self.reset_at {
            // TinyLFU aging: halve everything
            for c in &mut self.counters {
                *c >>= 1;
            }
            self.adds /= 2;
        }
    }
    fn estimate(&self, k: u64) -> u8 {
        (0..4).map(|i| self.counters[self.idx(k, i)]).min().unwrap()
    }
}

pub struct HotTier {
    cap_rows: usize,
    pinned_cap: usize,
    data: Vec<u8>,
    /// key -> slot
    index: HashMap<u64, u32, FxBuild>,
    /// slot -> key
    keys: Vec<u64>,
    /// pinned slots are `0..pinned`; dynamic slots live in `pinned_cap..pinned_cap + dynamic`
    pinned: usize,
    dynamic: usize,
    sketch: Option<Sketch>,
    rng: u64,
    pub stats: HotStats,
}

impl HotTier {
    pub fn new(cfg: HotTierConfig) -> Self {
        let cap_rows = cfg.capacity_bytes / RECORD_BYTES;
        let dyn_rows = ((cap_rows as f64) * cfg.dynamic_fraction.clamp(0.0, 1.0) as f64) as usize;
        let pinned_cap = cap_rows - dyn_rows;
        let sketch = (dyn_rows > 0).then(|| Sketch::new(cfg.sketch_log2, (10 * cap_rows as u64).max(1 << 20)));
        // Allocate everything up front: growing a 1 GB slab or a 4M-entry map later costs a 10-20 ms copy/rehash
        // inside a decode step. `vec![0; n]` is a zeroed allocation (lazily committed pages), so RSS still grows
        // only as rows land.
        Self {
            cap_rows,
            pinned_cap,
            data: vec![0u8; cap_rows * RECORD_BYTES],
            index: HashMap::with_capacity_and_hasher(cap_rows, FxBuild::default()),
            keys: vec![u64::MAX; cap_rows],
            pinned: 0,
            dynamic: 0,
            sketch,
            rng: 0x2545_F491_4F6C_DD1D,
            stats: HotStats::default(),
        }
    }

    pub fn capacity_rows(&self) -> usize {
        self.cap_rows
    }
    pub fn pinned_capacity(&self) -> usize {
        self.pinned_cap
    }
    pub fn len(&self) -> usize {
        self.index.len()
    }
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    fn slot_mut(&mut self, slot: usize) -> &mut [u8] {
        &mut self.data[slot * RECORD_BYTES..(slot + 1) * RECORD_BYTES]
    }

    /// Look a row up (counts towards hit rate and frequency).
    #[inline]
    pub fn lookup(&mut self, k: u64) -> Option<&[u8]> {
        self.stats.lookups += 1;
        if let Some(s) = &mut self.sketch {
            s.add(k);
        }
        let slot = *self.index.get(&k)? as usize;
        self.stats.hits += 1;
        Some(&self.data[slot * RECORD_BYTES..(slot + 1) * RECORD_BYTES])
    }

    /// Whether a row is resident (no stats).
    pub fn contains(&self, k: u64) -> bool {
        self.index.contains_key(&k)
    }

    /// Pin a row (static region). Returns false when the pinned region is full or the row is already there.
    pub fn pin(&mut self, k: u64, record: &[u8]) -> bool {
        if self.pinned >= self.pinned_cap || self.index.contains_key(&k) {
            return false;
        }
        let slot = self.pinned;
        self.pinned += 1;
        self.slot_mut(slot).copy_from_slice(&record[..RECORD_BYTES]);
        self.set_key(slot, k);
        self.stats.pinned_rows = self.pinned as u64;
        self.stats.bytes = (self.len() * RECORD_BYTES) as u64;
        true
    }

    fn set_key(&mut self, slot: usize, k: u64) {
        self.keys[slot] = k;
        self.index.insert(k, slot as u32);
    }

    fn next_rand(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    /// Offer a row just read from disk to the dynamic region (TinyLFU admission).
    pub fn offer(&mut self, k: u64, record: &[u8]) {
        let dyn_cap = self.cap_rows - self.pinned_cap;
        if dyn_cap == 0 || self.index.contains_key(&k) {
            return;
        }
        let slot = if self.dynamic < dyn_cap {
            let s = self.pinned_cap + self.dynamic;
            self.dynamic += 1;
            s
        } else {
            let sk = self.sketch.as_ref().unwrap();
            let f_new = sk.estimate(k);
            // sampled LFU victim among 8 random dynamic slots
            let mut victim = usize::MAX;
            let mut f_victim = u8::MAX;
            for _ in 0..8 {
                let s = self.pinned_cap + (self.next_rand() % dyn_cap as u64) as usize;
                let f = self.sketch.as_ref().unwrap().estimate(self.keys[s]);
                if f < f_victim {
                    f_victim = f;
                    victim = s;
                }
            }
            if f_new <= f_victim {
                self.stats.rejected += 1;
                return;
            }
            let old = self.keys[victim];
            self.index.remove(&old);
            self.stats.evicted += 1;
            victim
        };
        self.slot_mut(slot).copy_from_slice(&record[..RECORD_BYTES]);
        self.set_key(slot, k);
        self.stats.admitted += 1;
        self.stats.dynamic_rows = self.dynamic as u64;
        self.stats.bytes = (self.len() * RECORD_BYTES) as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(v: u8) -> Vec<u8> {
        vec![v; RECORD_BYTES]
    }

    #[test]
    fn pinned_and_dynamic() {
        let mut h = HotTier::new(HotTierConfig { capacity_bytes: 10 * RECORD_BYTES, dynamic_fraction: 0.5, sketch_log2: 10 });
        assert_eq!(h.pinned_capacity(), 5);
        for i in 0..7 {
            assert_eq!(h.pin(key(0, i), &rec(i as u8)), i < 5);
        }
        assert_eq!(h.lookup(key(0, 3)).unwrap()[0], 3);
        assert!(h.lookup(key(0, 6)).is_none());
        // fill the dynamic region, then a frequent row displaces an infrequent one
        for i in 100..105 {
            h.lookup(key(1, i));
            h.offer(key(1, i), &rec(1));
        }
        assert_eq!(h.stats.dynamic_rows, 5);
        for _ in 0..20 {
            h.lookup(key(1, 999));
        }
        h.offer(key(1, 999), &rec(9));
        assert_eq!(h.lookup(key(1, 999)).unwrap()[0], 9);
        assert_eq!(h.len(), 10);
        // pinned rows are never evicted
        for i in 0..5 {
            assert!(h.contains(key(0, i)));
        }
        // a one-off row is not admitted over seen-more-often rows
        h.offer(key(1, 5000), &rec(5));
        assert!(!h.contains(key(1, 5000)));
    }
}
