//! DeepSeek V4.1 Engram table tier.
//!
//! Engram (layers 1 and 14 of DSv4.1 Flash) adds an n-gram lookup into the residual stream: each position hashes
//! its last 2-4 (compressed) token ids into 24 rows per layer of two ~384M-row tables (FP8 E4M3 rows of 256 plus
//! E8M0 scales per 32; 101 GB per table). The tables stay on local NVMe, row-sharded across TP ranks; this crate
//! is everything between a token id and the rows in host memory:
//!
//! - [`HashLayout`] / [`NgramState`]: the official hash, bit-exact (`tests/hash_parity.rs`).
//! - [`TokenMap`]: raw -> compressed token ids (feature `tokenizer` builds it from tokenizer.json).
//! - [`TableShard`]: one layer's rows for one rank, in any supported on-disk layout ([`layout`] documents them).
//! - [`EngramStore`]: non-blocking `fetch` -> `poll` / `wait`, O_DIRECT reads through io_uring (Linux) or a pread
//!   pool, a frequency-pinned + TinyLFU [`HotTier`], speculative `prefetch` into a staging cache.
//!
//! Row ids are global (`0..num_embeddings`). Under the bucket-boundary split ([`HashLayout::rank_rows`]) each
//! TP=2 rank owns 12 of a layer's 24 columns, so it reads exactly 12 rows per layer per token (24 per token).

pub mod fp8;
pub mod hash;
pub mod hot;
pub mod io;
pub mod layout;
pub mod store;
pub mod token_map;

pub use hash::{dsv41, HashLayout, NgramState};
pub use hot::{HotStats, HotTier, HotTierConfig};
pub use io::BackendKind;
pub use layout::{LayoutKind, ReadGranule, ScaleMode, TableShard, RECORD_BYTES, SCALE_BYTES, WEIGHT_BYTES};
pub use store::{EngramStore, FetchInfo, StoreConfig, StoreStats, Ticket};
pub use token_map::TokenMap;
