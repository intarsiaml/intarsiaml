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
    propose_min(ctx, k, NGRAM_MIN)
}

/// [`propose`] with suffixes shorter than `min_n` (clamped to `NGRAM_MIN..=NGRAM_MAX`) not tried.
pub fn propose_min(ctx: &[u32], k: usize, min_n: usize) -> Vec<u32> {
    let len = ctx.len();
    if k == 0 {
        return Vec::new();
    }
    for n in (min_n.clamp(NGRAM_MIN, NGRAM_MAX)..=NGRAM_MAX).rev() {
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

/// Gate defaults, chosen by replaying the engine loop over 26 greedy streams (`docs/results/2026-10-15-metal-spec-g2g3.md`):
/// suffixes of one token are not drafted from (they are the main source of rejected passes on open-ended text), and six
/// consecutive rejected passes back off for 2, 4, 8, .. 32 plain steps.
pub const GATE_MIN_N: usize = 2;
pub const GATE_FAILS: u32 = 6;
pub const GATE_SKIP0: u32 = 2;
pub const GATE_SKIP_MAX: u32 = 32;

/// Drafter gate parameters (G2). A verify pass that accepts no draft costs more than the plain step it replaces (a
/// 5-row pass is ~15-20 % slower at short context), and on open-ended text nearly every pass is such a pass. So after
/// `fails` consecutive passes without an accepted draft the gate stops drafting for `skip0` plain steps, then probes
/// with one pass; each further failed probe doubles the skip (up to `skip_max`), any accepted draft resets it. On
/// text where nothing is ever accepted it costs `fails` + about `log2(skip_max / skip0) + n / skip_max` wasted passes
/// per request of `n` tokens. Independently, drafts come only from suffixes of at least `min_n` tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateParams {
    /// Consecutive failed passes before backing off (0: the gate is off).
    pub fails: u32,
    /// Plain steps skipped after the first back-off.
    pub skip0: u32,
    /// Longest skip.
    pub skip_max: u32,
    /// Shortest suffix the drafter may match.
    pub min_n: usize,
}

impl Default for GateParams {
    fn default() -> Self {
        GateParams { fails: GATE_FAILS, skip0: GATE_SKIP0, skip_max: GATE_SKIP_MAX, min_n: GATE_MIN_N }
    }
}

impl GateParams {
    /// Defaults, or `INTARSIA_SPEC_GATE=fails,skip0,skip_max,min_n` (experiments; `0,...` turns the gate off).
    pub fn from_env() -> GateParams {
        static P: std::sync::OnceLock<GateParams> = std::sync::OnceLock::new();
        *P.get_or_init(|| {
            let mut p = GateParams::default();
            if let Ok(v) = std::env::var("INTARSIA_SPEC_GATE") {
                let f: Vec<usize> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if f.len() == 4 {
                    p = GateParams { fails: f[0] as u32, skip0: f[1] as u32, skip_max: f[2] as u32, min_n: f[3] };
                }
            }
            p
        })
    }
}

/// Per-request drafter gate: decides whether a step may draft (see [`GateParams`]).
#[derive(Clone, Copy, Debug)]
pub struct SpecGate {
    p: GateParams,
    /// Consecutive passes without an accepted draft.
    streak: u32,
    /// Plain steps still to take before the next probe.
    skip: u32,
    /// Length of the next skip.
    next_skip: u32,
}

impl SpecGate {
    pub fn new(p: GateParams) -> SpecGate {
        SpecGate { p, streak: 0, skip: 0, next_skip: p.skip0 }
    }

    /// Shortest suffix the drafter may match.
    pub fn min_n(&self) -> usize {
        self.p.min_n
    }

    /// Called once per step that could draft. `false`: take a plain step (counts down the back-off).
    pub fn allow(&mut self) -> bool {
        if self.skip > 0 {
            self.skip -= 1;
            return false;
        }
        true
    }

    /// Result of a verify pass: `accepted` of its drafts were accepted.
    pub fn record(&mut self, accepted: usize) {
        if self.p.fails == 0 {
            return;
        }
        if accepted > 0 {
            self.streak = 0;
            self.next_skip = self.p.skip0;
            return;
        }
        self.streak += 1;
        if self.streak >= self.p.fails {
            self.skip = self.next_skip;
            self.next_skip = (self.next_skip * 2).min(self.p.skip_max);
        }
    }
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

    fn gate(fails: u32, skip0: u32, skip_max: u32) -> SpecGate {
        SpecGate::new(GateParams { fails, skip0, skip_max, min_n: NGRAM_MIN })
    }

    /// Steps that draft (true) / stay plain (false) for a stream of pass outcomes: every drafting step is a pass whose
    /// accepted count comes from `accepted(pass_index)`.
    fn run_gate(g: &mut SpecGate, steps: usize, accepted: impl Fn(usize) -> usize) -> Vec<bool> {
        let mut passes = 0;
        (0..steps)
            .map(|_| {
                let go = g.allow();
                if go {
                    g.record(accepted(passes));
                    passes += 1;
                }
                go
            })
            .collect()
    }

    #[test]
    fn gate_starts_open_and_one_miss_does_not_close_it() {
        let mut g = gate(2, 4, 64);
        assert!(g.allow());
        g.record(0);
        assert!(g.allow()); // 1 failed pass < fails
        g.record(3);
        assert!(g.allow());
        g.record(0);
        assert!(g.allow()); // the accepted pass reset the streak
    }

    #[test]
    fn gate_backs_off_exponentially_on_a_run_of_failures() {
        let mut g = gate(2, 4, 16);
        let drafted = run_gate(&mut g, 80, |_| 0);
        // passes at steps 0, 1 (2 fails), then 4 plain, probe, 8 plain, probe, 16 plain, probe, 16 plain, ...
        let at: Vec<usize> = drafted.iter().enumerate().filter(|(_, d)| **d).map(|(i, _)| i).collect();
        assert_eq!(at, vec![0, 1, 6, 15, 32, 49, 66]);
    }

    #[test]
    fn gate_accepted_pass_resets_the_back_off_length() {
        let mut g = gate(2, 4, 64);
        g.record(0);
        g.record(0); // back off: skip 4, next back-off 8
        assert_eq!((g.skip, g.next_skip), (4, 8));
        for _ in 0..4 {
            assert!(!g.allow());
        }
        assert!(g.allow()); // probe
        g.record(2); // accepted: closed streak, back-off length starts over
        assert_eq!((g.streak, g.next_skip), (0, 4));
        g.record(0);
        g.record(0);
        assert_eq!(g.skip, 4);
    }

    #[test]
    fn gate_probe_hit_reopens_immediately() {
        let mut g = gate(2, 4, 64);
        // two misses, 4 plain steps, then a probe that accepts: the following steps all draft
        let drafted = run_gate(&mut g, 12, |pass| if pass >= 2 { 3 } else { 0 });
        assert_eq!(drafted[..6], [true, true, false, false, false, false]);
        assert!(drafted[6..].iter().all(|d| *d));
    }

    #[test]
    fn gate_zero_fails_is_off() {
        let mut g = gate(0, 4, 64);
        let drafted = run_gate(&mut g, 50, |_| 0);
        assert!(drafted.iter().all(|d| *d));
    }

    #[test]
    fn gate_cost_on_open_text_is_logarithmic() {
        // 256 steps without a single accepted draft: a handful of wasted passes, not ~100
        let mut g = SpecGate::new(GateParams::default());
        let passes = run_gate(&mut g, 256, |_| 0).iter().filter(|d| **d).count();
        assert!(passes <= 20, "{passes} passes");
        // with every pass accepted it never closes
        let mut g = SpecGate::new(GateParams::default());
        assert!(run_gate(&mut g, 256, |_| 1).iter().all(|d| *d));
    }

    #[test]
    fn propose_min_skips_short_suffixes() {
        // only a 1-token suffix match ([7]): found at min_n 1, not at 2 or 3
        let ctx = [7, 1, 2, 3, 4, 7];
        assert_eq!(propose_min(&ctx, 3, 1), vec![1, 2, 3]);
        assert!(propose_min(&ctx, 3, 2).is_empty());
        assert!(propose_min(&ctx, 3, 3).is_empty());
        // a 2-token match is kept at min_n 2, dropped at 3
        let ctx = [5, 7, 1, 2, 3, 9, 5, 7];
        assert_eq!(propose_min(&ctx, 3, 2), vec![1, 2, 3]);
        assert!(propose_min(&ctx, 3, 3).is_empty());
        // out-of-range min_n is clamped
        assert_eq!(propose_min(&ctx, 3, 0), propose(&ctx, 3));
        assert_eq!(propose_min(&ctx, 3, 9), propose_min(&ctx, 3, NGRAM_MAX));
    }
}
