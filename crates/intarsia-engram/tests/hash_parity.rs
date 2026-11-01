//! Bit-exact parity of the Rust Engram hash with DeepSeek's official `engram.py`
//! (fixture: `tests/fixtures/hash_dsv41.json`, made by `tests/fixtures/gen_hash_fixture.py`).
//!
//! Gated extras:
//! - `INTARSIA_DSV41_TOKEN_MAP=/path/dsv41_token_map.json` (MiaAI / TensorFold format): the map's sha256 must
//!   equal the official Python run's, and real text hashes through it must match.
//! - `INTARSIA_DSV41_TOKENIZER=/path/tokenizer.json` with `--features tokenizer`: the Rust-built map must equal
//!   the official one (and the JSON map above, when given).

use intarsia_engram::{HashLayout, NgramState, TokenMap};
use serde_json::Value;

const MASK_C: u64 = 0xD1B5_4A32_D192_ED03;

fn fixture() -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/hash_dsv41.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Mirror of gen_hash_fixture.py `gen_sequence`.
fn gen_sequence(seed: u64, k: u64, vocab: u64, pad: u64) -> (Vec<u32>, Option<Vec<bool>>) {
    let mut s = seed ^ k.wrapping_mul(MASK_C);
    let n = 1 + splitmix(&mut s) % 64;
    let with_dead = k % 7 == 3;
    let (mut ids, mut dead) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let r = splitmix(&mut s);
        let t = match r % 16 {
            0 => pad,
            1 => (r >> 8) % 64,
            2 => vocab - 1 - (r >> 8) % 4,
            _ => (r >> 8) % vocab,
        };
        ids.push(t as u32);
        let r = splitmix(&mut s);
        dead.push(with_dead && r % 5 == 0);
    }
    (ids, with_dead.then_some(dead))
}

fn fnv1a(rows: &[u64]) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for v in rows {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01B3);
        }
    }
    h
}

fn u64s(v: &Value) -> Vec<u64> {
    let mut out = Vec::new();
    fn walk(v: &Value, out: &mut Vec<u64>) {
        match v {
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            Value::Number(n) => out.push(n.as_u64().unwrap()),
            _ => panic!("unexpected {v}"),
        }
    }
    walk(v, &mut out);
    out
}

#[test]
fn constants_match_official() {
    let f = fixture();
    let l = HashLayout::dsv41();
    assert_eq!(u64s(&f["primes"]), l.primes.concat());
    assert_eq!(u64s(&f["offsets"]), l.offsets.concat());
    assert_eq!(u64s(&f["multipliers"]), l.multipliers.iter().flatten().copied().collect::<Vec<_>>());
    assert_eq!(u64s(&f["num_embeddings"]), l.num_embeddings);
    assert_eq!(f["pad"].as_u64().unwrap(), l.pad as u64);
}

#[test]
fn hash_matches_official_engram_py() {
    let f = fixture();
    let l = HashLayout::dsv41();
    let seed = f["seed"].as_u64().unwrap();
    let vocab = f["compressed_vocab"].as_u64().unwrap();
    let digests = f["digests"].as_array().unwrap();
    let mut rows = Vec::new();
    let mut bad = 0;
    let mut positions = 0;
    for (k, want) in digests.iter().enumerate() {
        let (ids, dead) = gen_sequence(seed, k as u64, vocab, l.pad as u64);
        positions += ids.len();
        l.hash_sequence(&ids, dead.as_deref(), &mut rows);
        // decode path: one push at a time, after a prefill of the first half
        let mut st = NgramState::default();
        let n = l.rows_per_token();
        let mut inc = vec![0u64; ids.len() * n];
        for (i, &t) in ids.iter().enumerate() {
            let tok = if dead.as_ref().is_some_and(|d| d[i]) { None } else { Some(t) };
            st.push(&l, tok, &mut inc[i * n..(i + 1) * n]);
        }
        assert_eq!(inc, rows, "incremental != sequence for {k}");
        if format!("{:016x}", fnv1a(&rows)) != want.as_str().unwrap() {
            bad += 1;
            if bad < 5 {
                eprintln!("sequence {k} (len {}, dead {:?}) differs", ids.len(), dead.is_some());
            }
        }
    }
    assert_eq!(bad, 0, "{bad} of {} sequences differ from engram.py", digests.len());
    assert!(positions > 90_000);
    // and the stored full rows, element by element
    for case in f["full"].as_array().unwrap() {
        let k = case["k"].as_u64().unwrap();
        let (ids, dead) = gen_sequence(seed, k, vocab, l.pad as u64);
        l.hash_sequence(&ids, dead.as_deref(), &mut rows);
        assert_eq!(rows, u64s(&case["rows"]), "full rows of sequence {k}");
    }
}

fn check_real_text(f: &Value, map: &TokenMap) {
    let l = HashLayout::dsv41();
    let mut rows = Vec::new();
    for case in f["real_text"].as_array().unwrap() {
        let raw = u64s(&case["raw_ids"]);
        let ids: Vec<u32> = raw.iter().map(|&r| map.get(r as u32).unwrap()).collect();
        l.hash_sequence(&ids, None, &mut rows);
        assert_eq!(format!("{:016x}", fnv1a(&rows)), case["digest"].as_str().unwrap(), "{}", case["text"]);
    }
}

#[test]
fn token_map_json_matches_official() {
    let Ok(p) = std::env::var("INTARSIA_DSV41_TOKEN_MAP") else {
        eprintln!("skipped: set INTARSIA_DSV41_TOKEN_MAP=/path/dsv41_token_map.json");
        return;
    };
    let f = fixture();
    let map = TokenMap::load_json(std::path::Path::new(&p)).unwrap();
    assert_eq!(map.map.len() as u64, f["token_map_len"].as_u64().unwrap());
    assert_eq!(map.compressed_vocab as u64, f["compressed_vocab"].as_u64().unwrap());
    assert_eq!(map.sha256_u32le(), f["token_map_sha256_u32le"].as_str().unwrap());
    check_real_text(&f, &map);
}

#[cfg(feature = "tokenizer")]
#[test]
fn token_map_built_in_rust_matches_official() {
    let Ok(p) = std::env::var("INTARSIA_DSV41_TOKENIZER") else {
        eprintln!("skipped: set INTARSIA_DSV41_TOKENIZER=/path/tokenizer.json");
        return;
    };
    let f = fixture();
    let tok = tokenizers::Tokenizer::from_file(&p).unwrap();
    let map = intarsia_engram::token_map::build(&tok).unwrap();
    assert_eq!(map.compressed_vocab, intarsia_engram::dsv41::COMPRESSED_VOCAB);
    assert_eq!(map.sha256_u32le(), f["token_map_sha256_u32le"].as_str().unwrap());
    if let Ok(j) = std::env::var("INTARSIA_DSV41_TOKEN_MAP") {
        let mia = TokenMap::load_json(std::path::Path::new(&j)).unwrap();
        assert_eq!(map.map, mia.map);
    }
    check_real_text(&f, &map);
}
