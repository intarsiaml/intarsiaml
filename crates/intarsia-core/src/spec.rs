//! Speculative decoding, milestone G1 (docs/research/metal-next-levers.md section 7): greedy prompt-lookup (n-gram)
//! drafting and exact-match acceptance. Host side and model free; the verify pass is [`crate::model::Model::forward_verify`]
//! and the engine loop that ties both together is in `engine.rs`.
//!
//! Identity argument: every verify row's logits are bitwise those of the sequential single-token decode step at that
//! position (row-exact kernels, `OpCall::exact_rows`), so its argmax is the token greedy decode would emit there.
//! Accepting the longest prefix of drafts that equal those argmaxes, plus the first argmax after it, emits exactly the
//! greedy token stream, only several tokens per pass.

/// `--spec`: speculative decoding mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpecMode {
    /// Plain one-token decode steps (the default).
    #[default]
    Off,
    /// Prompt-lookup drafting over prompt + generated ids, verified in one multi-row pass.
    Ngram,
}

/// Longest suffix the drafter matches.
pub const NGRAM_MAX: usize = 3;
/// Shortest suffix the drafter matches.
pub const NGRAM_MIN: usize = 1;
/// Most draft tokens per verify pass (`--spec-k`): a pass has `1 + k` rows, and G0 / G0b measured the row-exact kernels
/// cheap up to 7 rows (the 8-row GEMV hits the shard-alignment cliff).
pub const SPEC_K_MAX: usize = 6;
/// Default `--spec-k`.
pub const SPEC_K_DEFAULT: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpecConfig {
    pub mode: SpecMode,
    /// Most draft tokens per verify pass, 1..=[`SPEC_K_MAX`].
    pub k: usize,
}

impl Default for SpecConfig {
    fn default() -> Self {
        SpecConfig { mode: SpecMode::Off, k: SPEC_K_DEFAULT }
    }
}

impl SpecConfig {
    pub fn enabled(&self) -> bool {
        self.mode != SpecMode::Off
    }

    /// Rows of the largest verify pass (`1 + k`).
    pub fn verify_rows(&self) -> usize {
        1 + self.k
    }

    /// `--spec off|ngram` and `--spec-k`.
    pub fn parse(mode: &str, k: usize) -> crate::error::Result<SpecConfig> {
        let mode = match mode {
            "off" | "none" | "" => SpecMode::Off,
            "ngram" => SpecMode::Ngram,
            other => return Err(crate::error::Error::invalid(format!("--spec {other} (off or ngram)"))),
        };
        if !(1..=SPEC_K_MAX).contains(&k) {
            return Err(crate::error::Error::invalid(format!("--spec-k {k} outside 1..={SPEC_K_MAX}")));
        }
        Ok(SpecConfig { mode, k })
    }
}

/// Prompt lookup: take the longest suffix of `ctx` (length [`NGRAM_MAX`] down to [`NGRAM_MIN`]) that occurred
/// earlier, and return up to `k` tokens that followed its most recent earlier occurrence (empty: no match). The
/// suffix itself is not an "earlier occurrence"; an occurrence may overlap it. An occurrence so close to the end that
/// fewer than `k` tokens follow it gives way to the next older one with `k` (repetition runs), else the longest
/// continuation found wins. O(len * n) comparisons per call, a few microseconds at 4k tokens.
pub fn propose(ctx: &[u32], k: usize) -> Vec<u32> {
    let len = ctx.len();
    if k == 0 {
        return Vec::new();
    }
    for n in (NGRAM_MIN..=NGRAM_MAX).rev() {
        if len < n + 1 {
            continue;
        }
        let suffix = &ctx[len - n..];
        let mut best: Option<(usize, usize)> = None; // (start of continuation, its length)
        // start positions i < len - n, newest first; ctx[i + n] exists for every one of them
        for i in (0..len - n).rev() {
            if &ctx[i..i + n] == suffix {
                let got = k.min(len - (i + n));
                if got == k {
                    return ctx[i + n..i + n + k].to_vec();
                }
                if best.is_none_or(|(_, l)| got > l) {
                    best = Some((i + n, got));
                }
            }
        }
        if let Some((s, l)) = best {
            return ctx[s..s + l].to_vec();
        }
    }
    Vec::new()
}

/// Greedy exact-match acceptance. `drafts` = `d_1..d_m`, `out` = the argmax of the `m + 1` verify rows (row 0 was fed
/// the last emitted token, row `i` was fed `d_i`). Returns `j`, the number of accepted drafts: `out[i] == d_{i+1}` for
/// every `i < j`. The tokens this pass emits are `out[..=j]` (`d_1..d_j`, then the first token greedy decode picks
/// after them), and the KV rows to keep are those of rows `0..=j`.
pub fn accept(drafts: &[u32], out: &[u32]) -> usize {
    assert_eq!(out.len(), drafts.len() + 1, "one verify row per draft plus the fed token");
    drafts.iter().zip(out).take_while(|(d, o)| d == o).count()
}

/// Per-step trace (`INTARSIA_SPEC_TRACE=1`, stderr): drafted / accepted counts and where a run was cut.
pub fn trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("INTARSIA_SPEC_TRACE").is_some_and(|v| v != "0"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propose_longest_suffix_most_recent() {
        // "1 2 3 4 | 9 | 1 2 3 5 6 | 7 1 2 3": suffix [1 2 3] last seen at 5, followed by 5 6 7 1
        let ctx = [1, 2, 3, 4, 9, 1, 2, 3, 5, 6, 7, 1, 2, 3];
        assert_eq!(propose(&ctx, 4), vec![5, 6, 7, 1]);
        assert_eq!(propose(&ctx, 2), vec![5, 6]);
        // the newest occurrence has only 6 followers: the older one at 0 has 10
        assert_eq!(propose(&ctx, 10), vec![4, 9, 1, 2, 3, 5, 6, 7, 1, 2]);
        // no occurrence has k followers: the longest continuation (the oldest here) wins
        assert_eq!(propose(&ctx, 12), vec![4, 9, 1, 2, 3, 5, 6, 7, 1, 2, 3]);
        assert_eq!(propose(&ctx, 0), Vec::<u32>::new());
    }

    #[test]
    fn propose_falls_back_to_shorter_suffixes() {
        // no earlier [8 9 3] and no [9 3], but [3] at 2 -> 4 5
        assert_eq!(propose(&[1, 2, 3, 4, 5, 8, 9, 3], 2), vec![4, 5]);
        // [x 7] two-token match preferred over the newer one-token match of [7]
        assert_eq!(propose(&[6, 7, 1, 2, 7, 3, 6, 7], 3), vec![1, 2, 7]);
        // nothing matches
        assert!(propose(&[1, 2, 3, 4], 4).is_empty());
        // too short for any earlier occurrence
        assert!(propose(&[5], 4).is_empty());
        assert!(propose(&[], 4).is_empty());
    }

    #[test]
    fn propose_overlapping_repetition() {
        // a run of one token: the suffix [7 7 7] occurs at 1 (overlapping), followed by one 7
        assert_eq!(propose(&[1, 7, 7, 7, 7], 4), vec![7]);
        // period-2 repetition proposes the continuation of the pattern
        assert_eq!(propose(&[1, 2, 1, 2, 1, 2], 4), vec![1, 2]);
        // the newest [1 2 1] (at 3) has only two followers, the older one at 1 has three
        assert_eq!(propose(&[3, 1, 2, 1, 2, 1, 2, 1], 3), vec![2, 1, 2]);
    }

    #[test]
    fn propose_never_proposes_the_suffix_position_itself() {
        // the only [4 5] is the suffix: fall back to [5], only the suffix too -> nothing
        assert!(propose(&[1, 2, 3, 4, 5], 3).is_empty());
    }

    #[test]
    fn accept_longest_matching_prefix() {
        assert_eq!(accept(&[5, 6, 7], &[5, 6, 7, 8]), 3); // all accepted, bonus 8
        assert_eq!(accept(&[5, 6, 7], &[5, 9, 7, 8]), 1); // d2 rejected, correction 9
        assert_eq!(accept(&[5, 6, 7], &[4, 6, 7, 8]), 0); // nothing accepted, emits 4
        assert_eq!(accept(&[], &[4]), 0);
        // a later match after a mismatch does not count
        assert_eq!(accept(&[1, 2, 3, 4], &[1, 9, 3, 4, 5]), 1);
    }

    #[test]
    #[should_panic]
    fn accept_needs_one_row_per_draft_plus_one() {
        accept(&[1, 2], &[1, 2]);
    }

    #[test]
    fn config_parse() {
        assert_eq!(SpecConfig::parse("off", 4).unwrap(), SpecConfig::default());
        let c = SpecConfig::parse("ngram", 6).unwrap();
        assert!(c.enabled() && c.verify_rows() == 7);
        assert!(SpecConfig::parse("ngram", 7).is_err());
        assert!(SpecConfig::parse("ngram", 0).is_err());
        assert!(SpecConfig::parse("draft", 4).is_err());
    }
}
