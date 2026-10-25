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
    /// The checkpoint's multi-token-prediction head drafts (`crate::model::mtp`, recipe `[mtp]`), verified the same way.
    /// Works on models with recurrent state: the verify pass keeps the state after every row
    /// (`crate::state::SpecRows`).
    Mtp,
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
/// Default `--spec-k` for `--spec mtp` when none is given: the k = 1..3 sweep on spark1
/// (docs/results/2026-10-22-mtp-spec-decode.md section 3).
pub const MTP_K_DEFAULT: usize = 2;

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
            "mtp" => SpecMode::Mtp,
            other => return Err(crate::error::Error::invalid(format!("--spec {other} (off, ngram or mtp)"))),
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
/// Verify passes a request may spend before the busy-batch check (see [`GateParams::busy_warmup`]) applies.
pub const GATE_BUSY_WARMUP: u32 = 1;
/// Mean accepted draft tokens per verify pass a request needs to keep drafting while other sequences decode.
pub const GATE_BUSY_MIN_ACCEPT: u32 = 2;

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
    /// Busy batch (another sequence decodes in the same step): verify passes a request spends before it has to show
    /// `busy_min_accept` (below). A verify pass runs alone on the request's slot, so beside a batched step it is a whole
    /// extra weight pass that the batch cannot absorb.
    pub busy_warmup: u32,
    /// Busy batch: mean accepted draft tokens per verify pass (over the request's passes so far) below which the
    /// request stops drafting until it is alone again (0: no busy check). Open-ended text accepts about 0.2 per pass,
    /// input-grounded text about 3.
    pub busy_min_accept: u32,
}

impl Default for GateParams {
    fn default() -> Self {
        GateParams {
            fails: GATE_FAILS,
            skip0: GATE_SKIP0,
            skip_max: GATE_SKIP_MAX,
            min_n: GATE_MIN_N,
            busy_warmup: GATE_BUSY_WARMUP,
            busy_min_accept: GATE_BUSY_MIN_ACCEPT,
        }
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
                if f.len() == 4 || f.len() == 6 {
                    p = GateParams { fails: f[0] as u32, skip0: f[1] as u32, skip_max: f[2] as u32, min_n: f[3], ..p };
                    if f.len() == 6 {
                        p.busy_warmup = f[4] as u32;
                        p.busy_min_accept = f[5] as u32;
                    }
                    eprintln!("spec gate: INTARSIA_SPEC_GATE={v:?} -> {p:?}");
                } else {
                    eprintln!("spec gate: ignoring INTARSIA_SPEC_GATE={v:?} (want 4 or 6 integers), using {p:?}");
                }
            }
            p
        })
    }
}

/// Parameters of the measured-cost gate. A speculative pass only pays when the tokens it yields cost less than the
/// plain steps they replace: `speedup = tokens_per_pass / (ms_per_pass / ms_per_plain_token)`. On a mixture-of-experts
/// model a 3-row verify reads the union of three tokens' experts, so a pass costs 2.2 to 2.4 plain steps on the R9700
/// and natural text yields 2.3 to 2.5 tokens per pass: no gain, and the fixed gate (accepted drafts only) cannot see it.
/// The request keeps running averages of both costs and falls back to plain decode while the speedup stays under
/// `min_speedup`; plain decode is the baseline it is measured against, so the fallback never loses to it by more than
/// the probes cost. Greedy output is the same on either path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostParams {
    /// Speedup below which speculation is held off (<= 0: the gate is off).
    pub min_speedup: f64,
    /// Consecutive passes (after the warm-up) whose averaged speedup is below `min_speedup` before falling back.
    pub window: u32,
    /// While speculating, one plain step every `probe_every` passes keeps the plain-step cost current.
    pub probe_every: u32,
    /// Tokens decoded plainly before speculation is tried again (acceptance changes with the text).
    pub reprobe_tokens: u32,
    /// Passes after a (re)start that are timed but not averaged (graph capture, the head catching up on skipped rows).
    pub warmup: u32,
    /// Weight of a new sample in the running averages.
    pub alpha: f64,
}

impl CostParams {
    pub const OFF: CostParams =
        CostParams { min_speedup: 0.0, window: 16, probe_every: 32, reprobe_tokens: 256, warmup: 2, alpha: 0.125 };

    /// Defaults for a backend (`Backend::name`). The break-even is the same everywhere (3% over plain decode, about the
    /// run-to-run noise of a request); CUDA and Metal sit far above it on natural text (plain step 8 to 12 ms against a
    /// 15 to 24 ms pass), so only HIP, where the plain step is a captured 6 ms, is ever held off.
    pub fn for_backend(_backend: &str) -> CostParams {
        CostParams { min_speedup: 1.03, ..CostParams::OFF }
    }

    pub fn enabled(&self) -> bool {
        self.min_speedup > 0.0
    }

    /// The recipe's `[mtp]` overrides.
    pub fn with_recipe(self, m: Option<&crate::recipe::MtpSpec>) -> CostParams {
        let Some(m) = m else { return self };
        CostParams {
            min_speedup: m.cost_min_speedup.unwrap_or(self.min_speedup),
            window: m.cost_window.map_or(self.window, |w| w.max(1)),
            probe_every: m.cost_probe_every.map_or(self.probe_every, |w| w.max(1)),
            reprobe_tokens: m.cost_reprobe_tokens.map_or(self.reprobe_tokens, |w| w.max(1)),
            ..self
        }
    }

    /// `INTARSIA_SPEC_COSTGATE`: `0` off, `1` the defaults, or `min_speedup,window,probe_every,reprobe_tokens`.
    pub fn with_env(self, v: Option<&str>) -> CostParams {
        let Some(v) = v.map(str::trim) else { return self };
        match v {
            "" => self,
            "0" | "off" => CostParams { min_speedup: 0.0, ..self },
            "1" | "on" => self,
            _ => {
                let f: Vec<&str> = v.split(',').map(str::trim).collect();
                let num = |i: usize| f.get(i).and_then(|x| x.parse::<f64>().ok());
                match (f.len(), num(0)) {
                    (4, Some(sp)) if sp > 0.0 && f.iter().all(|x| x.parse::<f64>().is_ok()) => CostParams {
                        min_speedup: sp,
                        window: (num(1).unwrap() as u32).max(1),
                        probe_every: (num(2).unwrap() as u32).max(1),
                        reprobe_tokens: (num(3).unwrap() as u32).max(1),
                        ..self
                    },
                    _ => {
                        eprintln!("spec costgate: ignoring INTARSIA_SPEC_COSTGATE={v:?} (want 0, 1 or min_speedup,window,probe_every,reprobe_tokens)");
                        self
                    }
                }
            }
        }
    }
}

/// Counters of the measured-cost gate, drained into telemetry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CostEvents {
    /// Switches from speculating to plain decode.
    pub off: u32,
    /// Switches back (the re-probe after `reprobe_tokens`).
    pub on: u32,
    /// Steps the gate took plainly: held off, or a probe of the plain-step cost.
    pub plain_steps: u32,
    /// Of those, probes.
    pub probes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CostMode {
    Spec,
    /// Decoding plainly; this many tokens until speculation is tried again.
    Plain { left: u32 },
}

/// A running average that starts at its first sample.
#[derive(Clone, Copy, Debug, Default)]
struct Ewma(Option<f64>);

impl Ewma {
    fn add(&mut self, x: f64, alpha: f64) {
        self.0 = Some(match self.0 {
            None => x,
            Some(m) => m + alpha * (x - m),
        });
    }
}

#[derive(Clone, Copy, Debug)]
struct CostGate {
    p: CostParams,
    mode: CostMode,
    tokens_per_pass: Ewma,
    ms_per_pass: Ewma,
    ms_per_plain: Ewma,
    /// Passes since speculation (re)started, and since the last plain step.
    passes: u32,
    since_plain: u32,
    /// Consecutive evaluations under `min_speedup`.
    low: u32,
    /// The next step is a probe of the plain-step cost.
    probe: bool,
    events: CostEvents,
}

impl CostGate {
    fn new(p: CostParams) -> CostGate {
        CostGate {
            p,
            mode: CostMode::Spec,
            tokens_per_pass: Ewma::default(),
            ms_per_pass: Ewma::default(),
            ms_per_plain: Ewma::default(),
            passes: 0,
            since_plain: 0,
            low: 0,
            probe: false,
            events: CostEvents::default(),
        }
    }

    /// May the next step speculate? `false` counts a plain step taken by this gate.
    fn allow(&mut self) -> bool {
        if !self.p.enabled() {
            return true;
        }
        if let CostMode::Plain { .. } = self.mode {
            self.events.plain_steps += 1;
            return false;
        }
        if std::mem::take(&mut self.probe) {
            self.events.plain_steps += 1;
            self.events.probes += 1;
            return false;
        }
        true
    }

    /// Current speedup estimate, once both costs are known.
    fn speedup(&self) -> Option<f64> {
        let (t, m, pl) = (self.tokens_per_pass.0?, self.ms_per_pass.0?, self.ms_per_plain.0?);
        (m > 0.0).then(|| t * pl / m)
    }

    fn spec_sample(&mut self, tokens: usize, ms: f64) {
        if !self.p.enabled() || self.mode != CostMode::Spec || tokens == 0 {
            return;
        }
        self.passes += 1;
        self.since_plain += 1;
        if self.since_plain >= self.p.probe_every || (self.ms_per_plain.0.is_none() && self.passes > self.p.warmup) {
            self.probe = true;
        }
        if self.passes <= self.p.warmup {
            return;
        }
        self.tokens_per_pass.add(tokens as f64, self.p.alpha);
        self.ms_per_pass.add(ms, self.p.alpha);
        match self.speedup() {
            Some(sp) if sp < self.p.min_speedup => {
                self.low += 1;
                if self.low >= self.p.window {
                    self.mode = CostMode::Plain { left: self.p.reprobe_tokens };
                    self.events.off += 1;
                    self.low = 0;
                }
            }
            _ => self.low = 0,
        }
    }

    fn plain_sample(&mut self, tokens: usize, ms: f64) {
        if !self.p.enabled() || tokens == 0 {
            return;
        }
        self.ms_per_plain.add(ms / tokens as f64, self.p.alpha);
        self.since_plain = 0;
        if let CostMode::Plain { left } = self.mode {
            let left = left.saturating_sub(tokens as u32);
            if left == 0 {
                // try speculation again on the text as it is now: the averages of the old stretch are dropped
                self.mode = CostMode::Spec;
                self.events.on += 1;
                self.tokens_per_pass = Ewma::default();
                self.ms_per_pass = Ewma::default();
                self.passes = 0;
                self.low = 0;
            } else {
                self.mode = CostMode::Plain { left };
            }
        }
    }
}

/// Per-request drafter gate: decides whether a step may draft (see [`GateParams`]).
#[derive(Clone, Copy, Debug)]
pub struct SpecGate {
    p: GateParams,
    /// Verify passes so far, and the draft tokens they accepted (the busy-batch check).
    passes: u32,
    accepted: u32,
    /// Consecutive passes without an accepted draft.
    streak: u32,
    /// Plain steps still to take before the next probe.
    skip: u32,
    /// Length of the next skip.
    next_skip: u32,
    /// The measured-cost gate (MTP only, off until [`SpecGate::set_cost`]).
    cost: CostGate,
}

impl SpecGate {
    pub fn new(p: GateParams) -> SpecGate {
        SpecGate { p, passes: 0, accepted: 0, streak: 0, skip: 0, next_skip: p.skip0, cost: CostGate::new(CostParams::OFF) }
    }

    /// Shortest suffix the drafter may match.
    pub fn min_n(&self) -> usize {
        self.p.min_n
    }

    /// Called once per step that could draft. `false`: take a plain step (counts down the back-off). `busy`: other
    /// sequences decode in the same step, where a verify pass is an extra weight pass instead of a replacement for a
    /// step: a request whose passes so far accepted fewer than `busy_min_accept` draft tokens each on average (after
    /// `busy_warmup` passes) takes plain steps and its back-off does not advance, so it neither probes nor loses its
    /// place; once it is alone (`busy == false`) the gate runs as without batching. A request that accepts well keeps
    /// drafting. `busy == false` is exactly the solo gate.
    pub fn allow(&mut self, busy: bool) -> bool {
        if busy && !self.busy_ok() {
            return false;
        }
        // beside other sequences the step time is not this request's own, so the measured-cost gate only runs alone
        if !busy && !self.cost.allow() {
            return false;
        }
        if self.skip > 0 {
            self.skip -= 1;
            return false;
        }
        true
    }

    /// Whether a request with this history may draft beside other decoding sequences.
    fn busy_ok(&self) -> bool {
        self.p.fails == 0
            || self.p.busy_min_accept == 0
            || self.passes < self.p.busy_warmup
            || self.accepted >= self.passes.saturating_mul(self.p.busy_min_accept)
    }

    /// Switch the measured-cost gate on with these parameters (see [`CostParams`]); `CostParams::OFF` switches it off.
    pub fn set_cost(&mut self, p: CostParams) {
        self.cost = CostGate::new(p);
    }

    /// A finished step as the request's wall clock saw it: a speculative pass (`spec`) or a plain step that yielded
    /// `tokens` tokens in `ms` milliseconds, host work included.
    pub fn cost_sample(&mut self, spec: bool, tokens: usize, ms: f64) {
        if spec {
            self.cost.spec_sample(tokens, ms);
        } else {
            self.cost.plain_sample(tokens, ms);
        }
    }

    /// Measured-cost gate counters since the last call (telemetry), and whether it holds speculation off now.
    pub fn take_cost_events(&mut self) -> CostEvents {
        std::mem::take(&mut self.cost.events)
    }

    /// Is the measured-cost gate on for this request?
    pub fn cost_enabled(&self) -> bool {
        self.cost.p.enabled()
    }

    pub fn cost_off(&self) -> bool {
        matches!(self.cost.mode, CostMode::Plain { .. })
    }

    /// Result of a verify pass: `accepted` of its drafts were accepted.
    pub fn record(&mut self, accepted: usize) {
        self.passes = self.passes.saturating_add(1);
        self.accepted = self.accepted.saturating_add(accepted as u32);
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
        assert_eq!(SpecConfig::parse("mtp", 2).unwrap(), SpecConfig { mode: SpecMode::Mtp, k: 2 });
    }

    fn gate(fails: u32, skip0: u32, skip_max: u32) -> SpecGate {
        SpecGate::new(GateParams { fails, skip0, skip_max, min_n: NGRAM_MIN, ..GateParams::default() })
    }

    /// Steps that draft (true) / stay plain (false) for a stream of pass outcomes: every drafting step is a pass whose
    /// accepted count comes from `accepted(pass_index)`.
    fn run_gate(g: &mut SpecGate, steps: usize, accepted: impl Fn(usize) -> usize) -> Vec<bool> {
        let mut passes = 0;
        (0..steps)
            .map(|_| {
                let go = g.allow(false);
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
        assert!(g.allow(false));
        g.record(0);
        assert!(g.allow(false)); // 1 failed pass < fails
        g.record(3);
        assert!(g.allow(false));
        g.record(0);
        assert!(g.allow(false)); // the accepted pass reset the streak
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
            assert!(!g.allow(false));
        }
        assert!(g.allow(false)); // probe
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

    /// Gate with the default busy check (1 warm-up pass, mean 2 accepted tokens per pass).
    fn busy_gate() -> SpecGate {
        SpecGate::new(GateParams { fails: 2, skip0: 4, skip_max: 16, ..GateParams::default() })
    }

    /// Steps that draft for a stream of (busy, accepted-of-this-pass) inputs.
    fn run_busy(g: &mut SpecGate, steps: &[bool], accepted: impl Fn(usize) -> usize) -> Vec<bool> {
        let mut passes = 0;
        steps
            .iter()
            .map(|&busy| {
                let go = g.allow(busy);
                if go {
                    g.record(accepted(passes));
                    passes += 1;
                }
                go
            })
            .collect()
    }

    #[test]
    fn gate_not_busy_ignores_the_busy_check() {
        // the solo schedule is the same whatever the busy parameters are
        let schedule = |p: GateParams| {
            let mut g = SpecGate::new(p);
            run_gate(&mut g, 120, |pass| if pass % 7 == 3 { 1 } else { 0 })
        };
        let base = GateParams { fails: 3, skip0: 2, skip_max: 16, ..GateParams::default() };
        let none = schedule(GateParams { busy_min_accept: 0, ..base });
        assert_eq!(schedule(base), none);
        assert_eq!(schedule(GateParams { busy_warmup: 0, busy_min_accept: 100, ..base }), none);
        // and it is the unchanged back-off: 3 misses, then probes at growing distance
        let mut g = SpecGate::new(GateParams { fails: 2, skip0: 4, skip_max: 16, ..GateParams::default() });
        let at: Vec<usize> =
            run_gate(&mut g, 80, |_| 0).iter().enumerate().filter(|(_, d)| **d).map(|(i, _)| i).collect();
        assert_eq!(at, vec![0, 1, 6, 15, 32, 49, 66]);
    }

    #[test]
    fn gate_busy_stops_probing_a_poor_request_and_keeps_a_good_one() {
        // poor: the warm-up pass is allowed, then no draft at all while busy, however many steps pass
        let mut g = busy_gate();
        let drafted = run_busy(&mut g, &[true; 200], |_| 0);
        assert_eq!(drafted.iter().filter(|d| **d).count(), 1, "one warm-up pass, no re-probe");
        assert!(drafted[1..].iter().all(|d| !*d));
        // the back-off did not advance while it waited
        assert_eq!(g.skip, 0);
        // good: 3 accepted per pass on average, an occasional empty pass does not close it
        let mut g = busy_gate();
        let drafted = run_busy(&mut g, &[true; 200], |pass| if pass % 5 == 4 { 0 } else { 4 });
        assert!(drafted.iter().all(|d| *d));
    }

    #[test]
    fn gate_busy_threshold_is_the_mean_accepted_per_pass() {
        let mut g = busy_gate();
        assert!(g.allow(true)); // warm-up pass
        g.record(3);
        assert!(g.allow(true)); // 3 accepted in 1 pass: mean 3 >= 2
        g.record(1);
        assert!(g.allow(true)); // 4 in 2: mean 2 >= 2
        g.record(0);
        assert!(!g.allow(true)); // 4 in 3: mean 1.33 < 2
    }

    #[test]
    fn gate_resumes_when_alone_again() {
        let mut g = busy_gate();
        assert!(g.allow(true));
        g.record(0); // fails 2: one more miss backs it off
        assert!(!g.allow(true) && !g.allow(true) && !g.allow(true)); // blocked, back-off frozen
        // alone: the normal gate. One more miss closes the streak and starts the back-off
        assert!(g.allow(false));
        g.record(0);
        assert_eq!(g.skip, 4);
        // busy again with the back-off pending: still no countdown, no probe
        for _ in 0..10 {
            assert!(!g.allow(true));
        }
        assert_eq!(g.skip, 4);
        // alone: it counts down and probes as before, and a hit reopens it
        for _ in 0..4 {
            assert!(!g.allow(false));
        }
        assert!(g.allow(false));
        g.record(4);
        // a long accepted run lifts the mean over the threshold: busy drafting is back
        for _ in 0..6 {
            assert!(g.allow(false));
            g.record(4);
        }
        assert!(g.allow(true));
    }

    #[test]
    fn gate_busy_warmup_and_off_switches() {
        // warm-up 2: two passes before the check applies
        let mut g = SpecGate::new(GateParams { busy_warmup: 2, ..GateParams::default() });
        let drafted = run_busy(&mut g, &[true; 20], |_| 0);
        assert_eq!(drafted.iter().filter(|d| **d).count(), 2);
        // busy_min_accept 0: no check
        let mut g = SpecGate::new(GateParams { busy_min_accept: 0, ..GateParams::default() });
        assert!(run_busy(&mut g, &[true; 20], |_| 0).iter().take(6).all(|d| *d));
        // gate off (fails 0): drafts always, busy or not
        let mut g = SpecGate::new(GateParams { fails: 0, ..GateParams::default() });
        assert!(run_busy(&mut g, &[true; 20], |_| 0).iter().all(|d| *d));
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

    // ---- measured-cost gate (scripted timings) ----

    fn cost_gate(p: CostParams) -> SpecGate {
        let mut g = SpecGate::new(GateParams::default());
        g.set_cost(p);
        g
    }

    fn cp() -> CostParams {
        CostParams { min_speedup: 1.03, window: 8, probe_every: 16, reprobe_tokens: 40, warmup: 2, alpha: 0.25 }
    }

    /// Drive `steps` decode steps: a speculative step yields `spec(i)` = (tokens, ms), a plain step 1 token in `plain` ms.
    /// Returns `S` for a speculative step and `p` for a plain one.
    fn drive(g: &mut SpecGate, steps: usize, spec: impl Fn(usize) -> (usize, f64), plain: f64) -> String {
        let mut out = String::new();
        for i in 0..steps {
            if g.allow(false) {
                let (t, ms) = spec(i);
                g.cost_sample(true, t, ms);
                out.push('S');
            } else {
                g.cost_sample(false, 1, plain);
                out.push('p');
            }
        }
        out
    }

    #[test]
    fn cost_gate_holds_speculation_off_where_a_pass_costs_what_its_tokens_would() {
        // the 35B on HIP at 1k: 2.33 tokens per pass in 13.9 ms against a 6.1 ms plain step = speedup 1.02
        let mut g = cost_gate(cp());
        let t = drive(&mut g, 200, |_| (2, 13.9), 6.1);
        // warm-up 2 + the first plain probe + 8 low evaluations, then 40 plain tokens, then speculation again
        let first_off = t.find("pppp").expect("falls back");
        assert!((10..=14).contains(&first_off), "{t}");
        let run = t[first_off..].chars().take_while(|c| *c == 'p').count();
        assert_eq!(run, 40, "reprobe_tokens plain steps: {t}");
        assert!(g.take_cost_events().off >= 2, "falls back again on the same text: {t}");
        // plain steps dominate: speculation only spends the warm-up and window of each probe
        assert!(t.matches('p').count() > t.matches('S').count(), "{t}");
    }

    #[test]
    fn cost_gate_keeps_speculating_where_it_pays() {
        // the 35B on HIP at 128 tokens: 2.8 tokens per pass in 13.7 ms = speedup 1.25
        let mut g = cost_gate(cp());
        let t = drive(&mut g, 200, |_| (3, 13.7), 6.1);
        let ev = g.take_cost_events();
        assert_eq!(ev.off, 0, "{t}");
        // only probes: one plain step per probe_every passes (plus the first, which measures the plain cost)
        assert!(t.matches('p').count() <= 1 + 200 / 16, "{t}");
        assert_eq!(ev.plain_steps as usize, t.matches('p').count());
        assert_eq!(ev.probes, ev.plain_steps);
        assert!(!g.cost_off());
    }

    #[test]
    fn cost_gate_measures_the_plain_step_before_it_decides() {
        // without a plain sample there is no speedup: it speculates until the first probe has run, never off blind
        let mut g = cost_gate(CostParams { probe_every: 1000, ..cp() });
        assert!(g.allow(false));
        for _ in 0..2 {
            g.cost_sample(true, 2, 14.0);
            assert!(g.allow(false), "warm-up passes");
        }
        g.cost_sample(true, 2, 14.0);
        assert!(!g.allow(false), "the first timed pass asks for a probe of the plain step");
        g.cost_sample(false, 1, 6.0);
        assert!(g.allow(false));
        assert!(!g.cost_off());
    }

    #[test]
    fn cost_gate_needs_a_whole_window_under_the_line() {
        // speedup alternates 1.0 and 1.5 per pass: the average sits above 1.03, and a single low evaluation never counts
        let mut g = cost_gate(cp());
        let t = drive(&mut g, 300, |i| if i % 2 == 0 { (2, 12.2) } else { (3, 12.2) }, 6.1);
        assert_eq!(g.take_cost_events().off, 0, "{t}");
        // a text that gets worse after 60 steps is caught: a long low stretch
        let mut g = cost_gate(cp());
        let t = drive(&mut g, 120, |i| if i < 60 { (3, 12.2) } else { (2, 12.2) }, 6.1);
        assert_eq!(g.take_cost_events().off, 1, "{t}");
        assert!(t[60..].contains('p'), "{t}");
    }

    #[test]
    fn cost_gate_retries_on_the_text_as_it_is_now() {
        let mut g = cost_gate(cp());
        // bad text until the gate is off and has taken its 40 plain tokens ...
        let mut steps = String::new();
        while !g.cost_off() {
            steps += &drive(&mut g, 1, |_| (2, 13.9), 6.1);
            assert!(steps.len() < 100, "falls back: {steps}");
        }
        assert_eq!(g.take_cost_events().off, 1);
        let plain = drive(&mut g, 40, |_| (2, 13.9), 6.1);
        assert_eq!(plain, "p".repeat(40));
        // ... then the text is good (3 tokens per pass in 12 ms = 1.5x): it stays on from here on
        assert!(!g.cost_off());
        let t = drive(&mut g, 100, |_| (3, 12.0), 6.1);
        let ev = g.take_cost_events();
        assert_eq!((ev.on, ev.off), (1, 0), "{t}");
        assert!(!g.cost_off() && !t.contains("pppp"), "{t}");
    }

    #[test]
    fn cost_gate_is_off_with_zero_and_beside_other_sequences() {
        let mut off = cost_gate(CostParams::OFF);
        let t = drive(&mut off, 100, |_| (2, 13.9), 6.1);
        assert!(!t.contains('p'), "{t}");
        assert_eq!(off.take_cost_events(), CostEvents::default());
        // busy: the step time is shared with the batch, the gate neither decides nor counts
        let mut g = cost_gate(cp());
        for _ in 0..50 {
            assert!(g.allow(true));
        }
        assert_eq!(g.take_cost_events(), CostEvents::default());
    }

    #[test]
    fn cost_gate_env_syntax() {
        let d = CostParams::for_backend("hip");
        assert_eq!(d.min_speedup, 1.03);
        assert!(d.enabled());
        assert!(!d.with_env(Some("0")).enabled());
        assert!(!d.with_env(Some("off")).enabled());
        assert_eq!(d.with_env(Some("1")), d);
        assert_eq!(d.with_env(None), d);
        let e = d.with_env(Some("1.1, 4, 8, 100"));
        assert_eq!((e.min_speedup, e.window, e.probe_every, e.reprobe_tokens), (1.1, 4, 8, 100));
        assert_eq!(e.warmup, d.warmup);
        assert_eq!(d.with_env(Some("1.1,4")), d, "malformed values are ignored");
        assert_eq!(d.with_env(Some("x,4,8,100")), d);
    }
}
