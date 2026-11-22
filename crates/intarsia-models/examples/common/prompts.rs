//! Prompts of the batched-decode benches (batch_step, batch_prof): `--prompts text` (default) or `--prompts random`.
//!
//! `text`: row i is `n` consecutive tokens of the fixed corpus `examples/data/bench_corpus.txt` (a frozen snapshot of
//! repository prose, a license and Rust source, so rows route like real text) tokenized with the model's
//! `tokenizer.json`, starting at token `(i % 16) * len / 16 + (i / 16) * 97` and wrapping at the end. Rows 0..16 start on
//! 16 evenly spaced offsets, so up to 16 rows read disjoint slices at contexts up to len / 16; the same slices every run.
//! `random`: the old uniform-random ids (SplitMix64 seed 0xbe7c_0000 + i, ids 1000..30000), whose rows route to nearly
//! the same experts (results 2026-11-21-inkling2-batch section 19); kept so old numbers stay reproducible.

use std::path::Path;

pub const CORPUS: &str = include_str!("../data/bench_corpus.txt");

pub enum Prompts {
    Random,
    Text(Vec<u32>),
}

impl Prompts {
    /// `mode` = "text" | "random"; `dir` = the checkpoint directory holding `tokenizer.json`.
    pub fn new(mode: &str, dir: &Path) -> Prompts {
        match mode {
            "random" => Prompts::Random,
            "text" => {
                let path = dir.join("tokenizer.json");
                let tk = tokenizers::Tokenizer::from_file(&path)
                    .unwrap_or_else(|e| panic!("--prompts text: loading {}: {e}", path.display()));
                let ids = tk.encode(CORPUS, false).unwrap_or_else(|e| panic!("--prompts text: tokenizing: {e}")).get_ids().to_vec();
                assert!(ids.len() >= 1024, "bench corpus too short: {} tokens", ids.len());
                Prompts::Text(ids)
            }
            m => panic!("--prompts: expected text or random, got {m}"),
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Prompts::Random => "random ids".into(),
            Prompts::Text(ids) => format!("text corpus ({} tokens, 16 slices of up to {})", ids.len(), ids.len() / 16),
        }
    }

    /// The `n` prompt tokens of row `i`.
    pub fn row(&self, i: usize, n: usize) -> Vec<u32> {
        match self {
            Prompts::Random => {
                let mut rng = intarsia_core::rng::SplitMix64::new(0xbe7c_0000 + i as u64);
                (0..n).map(|_| 1000 + (rng.next_u64() % 29000) as u32).collect()
            }
            Prompts::Text(ids) => {
                let len = ids.len();
                let start = ((i % 16) * len / 16 + (i / 16) * 97) % len;
                (0..n).map(|j| ids[(start + j) % len]).collect()
            }
        }
    }
}

/// A per-row digest of the greedy tokens: the first 16 hex digits of SHA-256 over the little-endian ids.
pub fn digest(toks: &[u32]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for t in toks {
        h.update(t.to_le_bytes());
    }
    h.finalize().iter().take(8).map(|b| format!("{b:02x}")).collect()
}
