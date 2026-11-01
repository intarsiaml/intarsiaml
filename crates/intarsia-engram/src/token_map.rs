//! The compressed token map: raw vocab id -> compressed id (tokens that normalize alike share one id).
//!
//! engram.py `build_compressed_token_map` (MIT): decode each token alone with the raw Rust tokenizer, normalize
//! it with NFKC, NFD, StripAccents, Lowercase, whitespace collapse and Strip (a lone space survives via a
//! private-use sentinel), and number the distinct keys in first-seen order. Partial UTF-8 byte tokens (decode
//! contains U+FFFD) are keyed by their raw token string. DSv4.1 yields 99,092 compressed ids over 129,280 tokens.
//!
//! [`build`] (feature `tokenizer`) runs the same HF `tokenizers` normalizers the Python binding wraps; the
//! parity test compares its output with the official Python run and with MiaAI's `dsv41_token_map.json`.

use std::io;
use std::path::Path;

/// A token map as a flat lookup.
#[derive(Clone, Debug)]
pub struct TokenMap {
    pub map: Vec<u32>,
    /// Distinct compressed ids (`engram_compressed_vocab_size`).
    pub compressed_vocab: u32,
}

impl TokenMap {
    /// Load a JSON list `[c0, c1, ...]` (the format of MiaAI's / TensorFold's `dsv41_token_map.json`).
    pub fn load_json(path: &Path) -> io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let v: Vec<u32> = serde_json::from_str(&text).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(Self::from_vec(v))
    }

    pub fn from_vec(map: Vec<u32>) -> Self {
        let compressed_vocab = map.iter().copied().max().map_or(0, |m| m + 1);
        Self { map, compressed_vocab }
    }

    /// Compressed id of a raw token id (`None` if out of range).
    #[inline]
    pub fn get(&self, raw: u32) -> Option<u32> {
        self.map.get(raw as usize).copied()
    }

    /// sha256 of the map as little-endian u32 (the fixture's `token_map_sha256_u32le`).
    pub fn sha256_u32le(&self) -> String {
        let mut bytes = Vec::with_capacity(self.map.len() * 4);
        for v in &self.map {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        use sha2::Digest;
        sha2::Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// Build the map from a tokenizer (engram.py `build_compressed_token_map`).
#[cfg(feature = "tokenizer")]
pub fn build(tok: &tokenizers::Tokenizer) -> io::Result<TokenMap> {
    use std::collections::HashMap;
    use tokenizers::normalizers::{
        replace::Replace, strip::Strip, strip::StripAccents, unicode::NFD, unicode::NFKC, utils::Lowercase,
        utils::Sequence, NormalizerWrapper,
    };
    use tokenizers::{NormalizedString, Normalizer};
    let err = |e: Box<dyn std::error::Error + Send + Sync>| io::Error::other(e.to_string());

    let sentinel = "\u{e000}";
    let chain = Sequence::new(vec![
        NormalizerWrapper::NFKC(NFKC),
        NormalizerWrapper::NFD(NFD),
        NormalizerWrapper::StripAccents(StripAccents),
        NormalizerWrapper::Lowercase(Lowercase),
        NormalizerWrapper::Replace(Replace::new(regex("[ \t\r\n]+"), " ").map_err(err)?),
        NormalizerWrapper::Replace(Replace::new(regex("^ $"), sentinel).map_err(err)?),
        NormalizerWrapper::StripNormalizer(Strip::new(true, true)),
        NormalizerWrapper::Replace(Replace::new(sentinel.to_string(), " ").map_err(err)?),
    ]);
    let n = tok.get_vocab_size(true);
    let mut key_to_new: HashMap<String, u32> = HashMap::new();
    let mut map = vec![0u32; n];
    for (id, slot) in map.iter_mut().enumerate() {
        let text = tok.decode(&[id as u32], false).map_err(err)?;
        let key = if text.contains('\u{fffd}') {
            tok.id_to_token(id as u32).unwrap_or_default()
        } else {
            let mut ns = NormalizedString::from(text.as_str());
            chain.normalize(&mut ns).map_err(err)?;
            let normalized = ns.get().to_string();
            if normalized.is_empty() {
                text
            } else {
                normalized
            }
        };
        let next = key_to_new.len() as u32;
        *slot = *key_to_new.entry(key).or_insert(next);
    }
    Ok(TokenMap { map, compressed_vocab: key_to_new.len() as u32 })
}

#[cfg(feature = "tokenizer")]
fn regex(p: &str) -> tokenizers::normalizers::replace::ReplacePattern {
    tokenizers::normalizers::replace::ReplacePattern::Regex(p.to_string())
}
