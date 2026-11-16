//! `--mtp-draft-subset FILE` (lane dsubset, opt-in): the MTP drafter's head over a chosen set of lm_head rows instead of the
//! `--mtp-draft-vocab N` prefix. A BPE vocabulary orders ids by script (the Qwen3.5/3.6/3.8 tokenizer: ASCII and Latin below
//! 95726, Chinese up to ~147k, Cyrillic / Arabic / Hangul / kana above), so a prefix head can never draft the scripts past
//! its end. A subset picks rows by how often the target model emits them, so ~65k rows draft every script.
//!
//! At load the chosen rows are gathered into one contiguous head (`QTensor::gather_rows`), the drafter's GEMV and argmax run on
//! it unchanged, and the argmax (a row of the sub-head) maps back to the token id through a device id table (`Model::draft_map`).
//! The verify pass keeps the full head, so the output is the spec-off output; only acceptance and speed move.
//!
//! File format (text, `#` comments; written by `scripts/dsubset/build_subset.py`):
//!
//! ```text
//! # intarsia mtp draft subset v1
//! # tokenizer <16 hex digits>   (tokenizer_fingerprint of the model's tokenizer.json, its BPE vocabulary; checked at load when present)
//! # rows <N>                    (the number of ids below; a multiple of 8)
//! 0-95723                       (an id or an inclusive id range per line, ascending, no repeats)
//! 95726
//! ```

use crate::error::{Error, Result};

/// A parsed subset file: the token ids the drafter's head keeps, ascending and distinct.
#[derive(Clone, PartialEq)]
pub struct DraftSubset {
    pub ids: Vec<u32>,
    /// The `# tokenizer` fingerprint the file was built for, if it names one.
    pub tokenizer: Option<String>,
    /// Where it came from (the path), for messages.
    pub source: String,
}

impl std::fmt::Debug for DraftSubset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DraftSubset({} rows from {}, tokenizer {:?})", self.ids.len(), self.source, self.tokenizer)
    }
}

impl DraftSubset {
    /// The ids directly (tests): ascending, distinct, a multiple of 8 of them.
    pub fn from_ids(ids: Vec<u32>, source: &str) -> Result<DraftSubset> {
        let s = DraftSubset { ids, tokenizer: None, source: source.into() };
        s.validate()?;
        Ok(s)
    }

    pub fn load(path: &std::path::Path) -> Result<DraftSubset> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::invalid(format!("--mtp-draft-subset {}: {e}", path.display())))?;
        DraftSubset::parse(&text, &path.display().to_string())
    }

    pub fn parse(text: &str, source: &str) -> Result<DraftSubset> {
        let bad = |line: usize, what: &str| Error::invalid(format!("--mtp-draft-subset {source}:{line}: {what}"));
        let (mut ids, mut tokenizer, mut rows, mut magic) = (Vec::new(), None, None, false);
        for (n, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if let Some(c) = line.strip_prefix('#') {
                let mut w = c.split_whitespace();
                match (w.next(), w.next()) {
                    (Some("intarsia"), _) if c.contains("mtp draft subset v1") => magic = true,
                    (Some("tokenizer"), Some(fp)) => tokenizer = Some(fp.to_ascii_lowercase()),
                    (Some("rows"), Some(r)) => rows = Some(r.parse::<usize>().map_err(|_| bad(n + 1, "rows is not a number"))?),
                    _ => {}
                }
                continue;
            }
            if line.is_empty() {
                continue;
            }
            let num = |s: &str| s.trim().parse::<u32>().map_err(|_| bad(n + 1, &format!("'{s}' is not a token id")));
            let (a, b) = match line.split_once('-') {
                Some((a, b)) => (num(a)?, num(b)?),
                None => (num(line)?, num(line)?),
            };
            if b < a {
                return Err(bad(n + 1, "range end below its start"));
            }
            if ids.last().is_some_and(|&l| a <= l) {
                return Err(bad(n + 1, "ids must be ascending and distinct"));
            }
            ids.extend(a..=b);
        }
        if !magic {
            return Err(Error::invalid(format!("--mtp-draft-subset {source}: not a draft subset file (no '# intarsia mtp draft subset v1' line)")));
        }
        if let Some(r) = rows.filter(|&r| r != ids.len()) {
            return Err(Error::invalid(format!("--mtp-draft-subset {source}: the header says {r} rows, the file lists {}", ids.len())));
        }
        let s = DraftSubset { ids, tokenizer, source: source.into() };
        s.validate()?;
        Ok(s)
    }

    fn validate(&self) -> Result<()> {
        let n = self.ids.len();
        if n == 0 || n % 8 != 0 {
            return Err(Error::invalid(format!(
                "--mtp-draft-subset {}: {n} ids; the head's kernels need a positive multiple of 8 rows",
                self.source
            )));
        }
        if self.ids.windows(2).any(|w| w[0] >= w[1]) {
            return Err(Error::invalid(format!("--mtp-draft-subset {}: ids must be ascending and distinct", self.source)));
        }
        Ok(())
    }

    /// Refuse a file built for another tokenizer (when the file names one): the ids would be other tokens.
    pub fn check_tokenizer(&self, tokenizer_json: &str) -> Result<()> {
        let Some(want) = &self.tokenizer else { return Ok(()) };
        let got = tokenizer_fingerprint(tokenizer_json)?;
        if &got != want {
            return Err(Error::invalid(format!(
                "--mtp-draft-subset {}: built for tokenizer {want}, this model's tokenizer.json is {got} (rebuild it with scripts/dsubset/build_subset.py)",
                self.source
            )));
        }
        Ok(())
    }

    /// Ids the head cannot hold (`>= rows`, the lm_head's row count).
    pub fn check_rows(&self, rows: usize) -> Result<()> {
        match self.ids.last() {
            Some(&l) if l as usize >= rows => Err(Error::invalid(format!(
                "--mtp-draft-subset {}: id {l} is past the lm_head's {rows} rows",
                self.source
            ))),
            _ if self.ids.len() >= rows => Err(Error::invalid(format!(
                "--mtp-draft-subset {}: {} ids, the lm_head has only {rows} rows (use the full head)",
                self.source,
                self.ids.len()
            ))),
            _ => Ok(()),
        }
    }
}

/// FNV-1a 64 over the BPE vocabulary in id order (`model.vocab` of an HF tokenizer.json; the added / special tokens are left
/// out, so checkpoints that share the vocabulary but add tokens, e.g. Qwen3.6-35B-A3B and Qwen3.8-27B with its audio tokens,
/// share a fingerprint): each id's token string's UTF-8 bytes, then a 0xff separator (an id with no token adds the separator
/// alone). 16 lowercase hex digits. `scripts/dsubset/build_subset.py` computes the same.
pub fn tokenizer_fingerprint(tokenizer_json: &str) -> Result<String> {
    let v: serde_json::Value =
        serde_json::from_str(tokenizer_json).map_err(|e| Error::invalid(format!("tokenizer.json: {e}")))?;
    let mut toks: Vec<Option<String>> = Vec::new();
    let mut put = |id: u64, s: &str| {
        let id = id as usize;
        if toks.len() <= id {
            toks.resize(id + 1, None);
        }
        toks[id] = Some(s.to_string());
    };
    if let Some(m) = v.pointer("/model/vocab").and_then(|m| m.as_object()) {
        for (s, id) in m {
            if let Some(id) = id.as_u64() {
                put(id, s);
            }
        }
    }
    if toks.is_empty() {
        return Err(Error::invalid("tokenizer.json: no model.vocab"));
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: u8| {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for t in &toks {
        for &b in t.as_deref().unwrap_or("").as_bytes() {
            eat(b);
        }
        eat(0xff);
    }
    Ok(format!("{h:016x}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ranges_and_header() {
        let s = DraftSubset::parse("# intarsia mtp draft subset v1\n# tokenizer ABCDEF\n# rows 16\n0-9\n12\n20-24\n", "t").unwrap();
        assert_eq!(s.ids, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 20, 21, 22, 23, 24]);
        assert_eq!(s.tokenizer.as_deref(), Some("abcdef"));
    }

    #[test]
    fn refuses_bad_files() {
        let p = |t: &str| DraftSubset::parse(t, "t");
        assert!(p("0-15\n").is_err(), "no magic line");
        let h = "# intarsia mtp draft subset v1\n";
        assert!(p(&format!("{h}0-14\n")).is_err(), "15 ids: not a multiple of 8");
        assert!(p(&format!("{h}0-7\n7-14\n")).is_err(), "overlap");
        assert!(p(&format!("{h}8-15\n0-7\n")).is_err(), "descending");
        assert!(p(&format!("{h}# rows 8\n0-15\n")).is_err(), "row count mismatch");
        assert!(p(&format!("{h}0-x\n")).is_err(), "not a number");
        assert!(p(&format!("{h}0-15\n")).is_ok());
        let s = p(&format!("{h}0-7\n100-107\n")).unwrap();
        assert!(s.check_rows(108).is_ok());
        assert!(s.check_rows(107).is_err(), "id past the head");
        assert!(s.check_rows(16).is_err(), "as many rows as the head");
    }

    /// The files shipped in `recipes/data/mtp-draft-subset/` parse, name a tokenizer and have the rows their names say; with
    /// `INTARSIA_TEST_TOKENIZER=<tokenizer.json>` set, their fingerprint must match it (the Qwen3.6-35B-A3B's).
    #[test]
    fn checked_in_subsets_parse() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../recipes/data/mtp-draft-subset");
        let tok = std::env::var("INTARSIA_TEST_TOKENIZER").ok().map(|p| std::fs::read_to_string(p).unwrap());
        let mut n = 0;
        for e in std::fs::read_dir(&dir).unwrap() {
            let p = e.unwrap().path();
            if p.extension().is_some_and(|x| x == "txt") {
                let s = DraftSubset::load(&p).unwrap();
                assert!(s.tokenizer.is_some(), "{}", p.display());
                let rows: usize = p.file_stem().unwrap().to_str().unwrap().rsplit('-').next().unwrap().parse().unwrap();
                assert_eq!(s.ids.len(), rows, "{}", p.display());
                if let Some(t) = &tok {
                    s.check_tokenizer(t).unwrap();
                }
                n += 1;
            }
        }
        assert!(n > 0);
    }

    #[test]
    fn fingerprint_is_order_and_content_sensitive() {
        let a = r#"{"model":{"vocab":{"a":0,"b":1,"c":3}},"added_tokens":[{"id":4,"content":"<s>"}]}"#;
        let b = r#"{"model":{"vocab":{"b":0,"a":1,"c":3}},"added_tokens":[{"id":4,"content":"<s>"}]}"#;
        let a2 = r#"{"model":{"vocab":{"a":0,"b":1,"c":3}},"added_tokens":[{"id":4,"content":"<s>"},{"id":5,"content":"<audio>"}]}"#;
        let fa = tokenizer_fingerprint(a).unwrap();
        assert_eq!(fa.len(), 16);
        assert_ne!(fa, tokenizer_fingerprint(b).unwrap());
        assert_eq!(fa, tokenizer_fingerprint(a2).unwrap(), "added tokens are not part of it");
        // FNV-1a 64 of "a\xff b\xff \xff c\xff" (id 2 empty), the value build_subset.py computes for the same vocabulary
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &x in b"a\xffb\xff\xffc\xff" {
            h ^= x as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        assert_eq!(fa, format!("{h:016x}"));
        let s = DraftSubset { ids: (0..8).collect(), tokenizer: Some(fa.clone()), source: "t".into() };
        assert!(s.check_tokenizer(a).is_ok());
        assert!(s.check_tokenizer(b).is_err());
    }
}
