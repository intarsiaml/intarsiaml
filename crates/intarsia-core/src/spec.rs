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
    /// DFlash parallel block drafter (`crate::model::dflash`, `--spec dflash --spec-draft DIR`, lane dc): a 5-layer draft
    /// reads the target's hidden states at five layers and proposes up to [`DFLASH_K_MAX`] tokens in one pass; verified
    /// by the same multi-row pass (attention one decode call per row where the backend's multi-row call is not bitwise).
    Dflash,
    /// Both model drafters, chosen per pass by measured tokens per ms (`--spec auto --spec-draft DIR`, lane asp,
    /// [`AutoPolicy`]): the MTP head ([`AUTO_MTP_K`] drafts) and the DFlash draft (`--spec-k`, default
    /// [`AUTO_DFLASH_K_DEFAULT`]). Greedy output is the same as `--spec off`: the verify pass decides every token.
    Auto,
}

/// Longest suffix the drafter matches.
pub const NGRAM_MAX: usize = 3;
/// Shortest suffix the drafter matches.
pub const NGRAM_MIN: usize = 1;
/// Most draft tokens per verify pass (`--spec-k`): a pass has `1 + k` rows, and G0 / G0b measured the row-exact kernels
/// cheap up to 7 rows (the 8-row GEMV hits the shard-alignment cliff).
pub const SPEC_K_MAX: usize = 6;
/// Most drafts per verify pass with `--spec dflash`: one block of the draft checkpoint (15) plus the anchor row = 16 rows,
/// `Model::VERIFY_MAX_ROWS`. The row-exact claim for these passes is `docs/results/2026-10-25-dc-dflash-dcoco.md` section 10.1.
pub const DFLASH_K_MAX: usize = 15;
/// `--spec-k` default with `--spec dflash` (the checkpoint's block).
pub const DFLASH_K_DEFAULT: usize = 15;
/// `--spec auto`: drafts per DFlash pass by default (Qwen3.8-27B tp=2 HIP sweet spot, docs/results/2026-10-27-dcp-dflash-cost.md).
pub const AUTO_DFLASH_K_DEFAULT: usize = 7;
/// `--spec auto`: drafts per MTP pass (the recipe's k for the 27B, [`MTP_K_DEFAULT`]).
pub const AUTO_MTP_K: usize = MTP_K_DEFAULT;
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
            "dflash" => SpecMode::Dflash,
            "auto" => SpecMode::Auto,
            other => return Err(crate::error::Error::invalid(format!("--spec {other} (off, ngram, mtp, dflash or auto)"))),
        };
        let kmax = if matches!(mode, SpecMode::Dflash | SpecMode::Auto) { DFLASH_K_MAX } else { SPEC_K_MAX };
        if !(1..=kmax).contains(&k) {
            return Err(crate::error::Error::invalid(format!("--spec-k {k} outside 1..={kmax}")));
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
/// The request keeps running averages of both costs and falls back to plain steps while the speedup stays under
/// `min_speedup`. The plain steps it times are the ones it would take: inside the speculating request, the MTP head
/// still catching up on each row. On HIP such a step cost 9 to 12 ms in window 10 (implied by the served throughput of
/// requests that fell back) where a spec-off step costs 6.3 to 6.9; the extra is most likely the head's catch-up. So the
/// fallback pays only when a pass is worse than that, and leaving the head to catch up later (tried: 6 to 16 % slower
/// than spec off at 1k to 16k) hides the cost from the gate.
/// Greedy output is the same on either path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostParams {
    /// Speedup below which speculation is held off (<= 0: the gate is off).
    pub min_speedup: f64,
    /// Consecutive passes (after the warm-up) whose averaged speedup is below `min_speedup` before falling back.
    pub window: u32,
    /// While speculating, one plain step every `probe_every` passes keeps the plain-step cost current.
    pub probe_every: u32,
    /// Tokens decoded plainly before speculation is tried again (acceptance changes with the text). Each re-probe that
    /// falls back again doubles the next stretch (up to 16 times this); one that holds resets it.
    pub reprobe_tokens: u32,
    /// Passes after a (re)start that are timed but not averaged (graph capture, the head catching up on skipped rows).
    pub warmup: u32,
    /// Weight of a new sample in the running averages.
    pub alpha: f64,
    /// Decide on the smoothed speedup alone: fall back once `warmup + window` passes have run since the (re)start and the
    /// running-mean speedup is under `min_speedup`. Default (false): `window` consecutive passes under it, which a noisy
    /// estimate near the line keeps resetting (DFlash prose: 24+ passes before the fallback, docs/results/2026-10-25-dc-dflash-dcoco.md).
    pub smooth: bool,
}

impl CostParams {
    pub const OFF: CostParams =
        CostParams { min_speedup: 0.0, window: 16, probe_every: 128, reprobe_tokens: 64, warmup: 2, alpha: 0.125, smooth: false };

    /// Built-in starting point of the DFlash gate (the recipe's `[dflash]` table and `INTARSIA_SPEC_COSTGATE` adjust it): OFF by
    /// default. Measured on Qwen3-8B CUDA (docs/results/2026-10-25-dc-dflash-dcoco.md section 12) the gate fixes low-acceptance prose
    /// (0.98x -> 1.00x) but decides on the first passes of an answer and costs recall/code 11-12 %, so ungated DFlash is the default.
    /// `INTARSIA_SPEC_COSTGATE=1` (or `cost_min_speedup = 1.03` in `[dflash]`) turns it on with the window below, a pass is about
    /// 100 ms there so the window is short.
    pub const DFLASH: CostParams =
        CostParams { min_speedup: 0.0, window: 6, probe_every: 128, reprobe_tokens: 128, warmup: 2, alpha: 0.125, smooth: true };

    /// The recipe's `[dflash]` overrides.
    pub fn with_dflash_recipe(self, m: Option<&crate::recipe::DflashSpec>) -> CostParams {
        let Some(m) = m else { return self };
        CostParams {
            min_speedup: m.cost_min_speedup.unwrap_or(self.min_speedup),
            window: m.cost_window.map_or(self.window, |w| w.max(1)),
            probe_every: m.cost_probe_every.map_or(self.probe_every, |w| w.max(1)),
            reprobe_tokens: m.cost_reprobe_tokens.map_or(self.reprobe_tokens, |w| w.max(1)),
            ..self
        }
    }

    /// The speedup the gate holds speculation to when it is on: 3% over plain decode, about the run-to-run noise of a request.
    pub const DEFAULT_MIN_SPEEDUP: f64 = 1.03;

    /// Defaults for a backend (`Backend::name`): off everywhere. The gate has no measured benefit on any backend: on
    /// CUDA and Metal it never switched off in any run (the pass is 1.2 to 1.5x a plain step there), and on HIP window 10
    /// (fleetq #5246) showed it lose, 6 to 18 % under spec off at 1k to 16k, because the plain steps it falls back to
    /// still pay the MTP head's catch-up on every row and cost more than a spec-off step. With it off the decode loop
    /// runs exactly the code it ran before. A recipe (`[mtp]`) or `INTARSIA_SPEC_COSTGATE` turns it on anywhere.
    pub fn for_backend(_backend: &str) -> CostParams {
        CostParams::OFF
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

    /// `INTARSIA_SPEC_COSTGATE`: `0` off, `1` on with the backend's values (the 3% line where it is off by default), or `min_speedup,window,probe_every,reprobe_tokens`.
    pub fn with_env(self, v: Option<&str>) -> CostParams {
        let Some(v) = v.map(str::trim) else { return self };
        match v {
            "" => self,
            "0" | "off" => CostParams { min_speedup: 0.0, ..self },
            "1" | "on" => CostParams {
                min_speedup: if self.enabled() { self.min_speedup } else { Self::DEFAULT_MIN_SPEEDUP },
                ..self
            },
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
    /// Length of the next plain stretch.
    next_reprobe: u32,
    /// The next step is a probe of the plain-step cost.
    probe: bool,
    /// A probe step was taken and its sample has not arrived yet (samples arrive one step late).
    probe_inflight: bool,
    /// Passes since then: a sample that never arrives (a probe step with no yielded token) must not stop probing.
    probe_age: u32,
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
            next_reprobe: p.reprobe_tokens,
            probe: false,
            probe_inflight: false,
            probe_age: 0,
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
            self.probe_inflight = true;
            self.probe_age = 0;
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
        if self.probe_inflight {
            self.probe_age += 1;
            if self.probe_age > 2 {
                self.probe_inflight = false;
            }
        }
        if !self.probe_inflight
            && (self.since_plain >= self.p.probe_every || (self.ms_per_plain.0.is_none() && self.passes > self.p.warmup))
        {
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
                if self.low >= self.p.window || (self.p.smooth && self.passes >= self.p.warmup + self.p.window) {
                    self.mode = CostMode::Plain { left: self.next_reprobe };
                    self.next_reprobe = self.next_reprobe.saturating_mul(2).min(self.p.reprobe_tokens.saturating_mul(16));
                    self.events.off += 1;
                    self.low = 0;
                }
            }
            _ => {
                self.low = 0;
                if self.passes >= self.p.warmup + self.p.window {
                    // a whole window held since the last re-probe: the next fallback starts from the short stretch again
                    self.next_reprobe = self.p.reprobe_tokens;
                }
            }
        }
    }

    fn plain_sample(&mut self, tokens: usize, ms: f64) {
        if !self.p.enabled() || tokens == 0 {
            return;
        }
        self.ms_per_plain.add(ms / tokens as f64, self.p.alpha);
        self.since_plain = 0;
        self.probe_inflight = false;
        self.probe = false;
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

/// Which model drafter proposes a pass of `--spec auto`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drafter {
    Mtp,
    Dflash,
}

impl Drafter {
    fn idx(self) -> usize {
        self as usize
    }
    fn other(self) -> Drafter {
        match self {
            Drafter::Mtp => Drafter::Dflash,
            Drafter::Dflash => Drafter::Mtp,
        }
    }
}

/// `INTARSIA_SPEC_AUTO` override of [`AutoPolicy`]: pin one drafter, or switch every `n` passes (tests, A/B runs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoForce {
    Mtp,
    Dflash,
    /// MTP for `n` passes, then DFlash for `n`, and so on.
    Alternate(u32),
}

/// Parameters of [`AutoPolicy`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoParams {
    /// Passes each drafter gets at the start of a request before the scores decide.
    pub seed_passes: u32,
    /// EWMA weight of a new sample (the first `1 / alpha` samples average evenly).
    pub alpha: f64,
    /// The other drafter must score this much better (ratio) to take over from DFlash.
    pub margin: f64,
    /// Same, to take over from MTP (the cheap, steady default: DFlash has to be clearly ahead).
    pub leave_mtp: f64,
    /// Scored picks a drafter keeps the lead before the other may take over (no flapping on one lucky pass).
    pub dwell: u32,
    /// Weight (at least) of a probe pass's sample: one lucky DFlash pass of 8 tokens must not move the estimate far.
    pub probe_w: f64,
    /// Priors per drafter (tokens per pass, ms per pass) counting as `prior_w` earlier passes; `prior_w` 0 starts at the
    /// first sample. Measured costs on the 27B tp=2 (docs/results/2026-10-28-asa-auto-priors.md).
    pub prior: [(f64, f64); 2],
    pub prior_w: f64,
    /// Passes between one-pass probes of the idle drafter.
    pub probe_every: u32,
    /// When the leader scores this ratio over the idle drafter, probe `probe_far` times as rarely.
    pub far_ratio: f64,
    pub probe_far: u32,
    pub force: Option<AutoForce>,
    /// One stderr line per request: passes per drafter, switches, probes (`INTARSIA_SPEC_AUTO=...,log`).
    pub log: bool,
    /// One stderr line per pass: drafter, emitted tokens, ms, first-after-switch (`INTARSIA_SPEC_AUTO=...,pass`; lane asa replay).
    pub log_pass: bool,
}

impl Default for AutoParams {
    fn default() -> Self {
        AutoParams {
            seed_passes: 2,
            alpha: 0.1,
            margin: 1.03,
            leave_mtp: 1.08,
            dwell: 8,
            probe_w: 0.2,
            prior: [(2.4, 25.5), (3.0, 38.0)],
            prior_w: 3.0,
            probe_every: 48,
            far_ratio: 1.3,
            probe_far: 4,
            force: None,
            log: false,
            log_pass: false,
        }
    }
}

impl AutoParams {
    /// `INTARSIA_SPEC_AUTO`: comma-separated `mtp`, `dflash`, `alt:N` (force a drafter), `log` (a summary line per request); unset
    /// or empty: the defaults.
    pub fn with_env(mut self, v: Option<&str>) -> AutoParams {
        for x in v.unwrap_or("").split(',').map(str::trim) {
            match x {
                "" | "0" | "off" => {}
                "log" => self.log = true,
                "pass" => self.log_pass = true,
                "mtp" => self.force = Some(AutoForce::Mtp),
                "dflash" => self.force = Some(AutoForce::Dflash),
                x if x.contains('=') => {
                    // A/B knobs: alpha, margin, leave, dwell, w (probe weight), priorw, seed, probe, far, probe_far
                    let (k, v) = x.split_once('=').unwrap();
                    match (k, v.parse::<f64>()) {
                        ("alpha", Ok(v)) => self.alpha = v,
                        ("margin", Ok(v)) => self.margin = v,
                        ("leave", Ok(v)) => self.leave_mtp = v,
                        ("dwell", Ok(v)) => self.dwell = v as u32,
                        ("w", Ok(v)) => self.probe_w = v,
                        ("priorw", Ok(v)) => self.prior_w = v,
                        ("seed", Ok(v)) => self.seed_passes = v as u32,
                        ("probe", Ok(v)) if v >= 1.0 => self.probe_every = v as u32,
                        ("far", Ok(v)) => self.far_ratio = v,
                        ("probe_far", Ok(v)) if v >= 1.0 => self.probe_far = v as u32,
                        _ => eprintln!("spec auto: ignoring INTARSIA_SPEC_AUTO part {x:?}"),
                    }
                }
                x => match x.strip_prefix("alt:").and_then(|n| n.parse::<u32>().ok()).filter(|&n| n > 0) {
                    Some(n) => self.force = Some(AutoForce::Alternate(n)),
                    None => eprintln!("spec auto: ignoring INTARSIA_SPEC_AUTO part {x:?} (want mtp, dflash, alt:N, log, pass or key=value)"),
                },
            }
        }
        self
    }

    pub fn from_env() -> AutoParams {
        AutoParams::default().with_env(std::env::var("INTARSIA_SPEC_AUTO").ok().as_deref())
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct AutoStat {
    tokens: Ewma,
    ms: Ewma,
    passes: u32,
}

impl AutoStat {
    /// Tokens per ms; `None` until a pass of the drafter has been timed.
    fn rate(&self) -> Option<f64> {
        match (self.tokens.0, self.ms.0) {
            (Some(t), Some(ms)) if ms > 0.0 => Some(t / ms),
            _ => None,
        }
    }
}

/// `--spec auto`: which drafter proposes the next pass. Per drafter an EWMA of the tokens a pass yields and of the
/// measured milliseconds of a pass (draft + verify + rollback); the better tokens-per-ms wins. Deterministic given the
/// samples. The first pass after a switch pays the new drafter's catch-up and is not timed (its tokens count); the idle
/// drafter is probed for one pass now and then, rarely when the leader is far ahead.
#[derive(Clone, Debug)]
pub struct AutoPolicy {
    p: AutoParams,
    stats: [AutoStat; 2],
    cur: Drafter,
    /// Passes planned so far (the `Alternate` clock).
    passes: u32,
    /// Passes since the last probe (or switch).
    since_probe: u32,
    /// The last pick was a probe of the idle drafter: the next one returns to the leader.
    probing: bool,
    last: Option<Drafter>,
    /// The pass in flight is the first after a switch.
    fresh: bool,
    /// The pass in flight is a probe of the idle drafter (its sample weighs more: the estimate is stale).
    probe_pass: bool,
    /// The probe just taken beat the leader: probe once more before deciding.
    confirm: bool,
    /// Scored picks since the lead last changed hands.
    in_lead: u32,
    pub switches: u32,
    pub probes: u32,
}

impl AutoPolicy {
    pub fn new(p: AutoParams) -> AutoPolicy {
        let mut stats = [AutoStat::default(); 2];
        if p.prior_w > 0.0 {
            for (s, (t, ms)) in stats.iter_mut().zip(p.prior) {
                s.tokens = Ewma(Some(t));
                s.ms = Ewma(Some(ms));
            }
        }
        AutoPolicy {
            p,
            stats,
            cur: Drafter::Mtp,
            passes: 0,
            since_probe: 0,
            probing: false,
            last: None,
            fresh: false,
            probe_pass: false,
            confirm: false,
            in_lead: 0,
            switches: 0,
            probes: 0,
        }
    }

    pub fn params(&self) -> &AutoParams {
        &self.p
    }

    /// Passes taken with `d` so far.
    pub fn passes_of(&self, d: Drafter) -> u32 {
        self.stats[d.idx()].passes
    }

    /// The drafter of the next pass.
    pub fn pick(&mut self) -> Drafter {
        let n = self.passes;
        self.passes += 1;
        let d = match self.p.force {
            Some(AutoForce::Mtp) => Drafter::Mtp,
            Some(AutoForce::Dflash) => Drafter::Dflash,
            Some(AutoForce::Alternate(k)) => {
                if (n / k) % 2 == 0 {
                    Drafter::Mtp
                } else {
                    Drafter::Dflash
                }
            }
            None => self.pick_scored(),
        };
        if self.last.is_some_and(|l| l != d) {
            self.switches += 1;
        }
        self.fresh = self.last != Some(d);
        self.probe_pass = self.probing && d != self.cur;
        self.last = Some(d);
        d
    }

    fn pick_scored(&mut self) -> Drafter {
        let seed = self.p.seed_passes;
        for d in [Drafter::Mtp, Drafter::Dflash] {
            if self.stats[d.idx()].passes < seed {
                return d;
            }
        }
        if self.probing {
            if std::mem::take(&mut self.confirm) {
                return self.cur.other();
            }
            self.probing = false;
            return self.cur;
        }
        let (rc, ro) = (self.stats[self.cur.idx()].rate(), self.stats[self.cur.other().idx()].rate());
        self.in_lead += 1;
        if let (Some(rc), Some(ro)) = (rc, ro) {
            let m = if self.cur == Drafter::Mtp { self.p.leave_mtp } else { self.p.margin };
            if ro > rc * m && self.in_lead >= self.p.dwell {
                self.cur = self.cur.other();
                self.since_probe = 0;
                self.in_lead = 0;
            }
        }
        let (lead, idle) = (self.stats[self.cur.idx()].rate(), self.stats[self.cur.other().idx()].rate());
        let every = match (lead, idle) {
            (Some(l), Some(i)) if l > i * self.p.far_ratio => self.p.probe_every * self.p.probe_far,
            _ => self.p.probe_every,
        };
        self.since_probe += 1;
        if self.since_probe >= every {
            self.since_probe = 0;
            self.probing = true;
            self.probes += 1;
            return self.cur.other();
        }
        self.cur
    }

    /// The pass picked last yielded `tokens` in `ms` (draft + verify + rollback).
    pub fn record(&mut self, d: Drafter, tokens: usize, ms: f64) {
        if self.p.log_pass {
            eprintln!("spec pass {} {tokens} {ms:.3} {}", if d == Drafter::Mtp { 'M' } else { 'D' }, u8::from(self.fresh));
        }
        let a = &mut self.stats[d.idx()];
        a.passes += 1;
        // the first samples average evenly, later ones with weight alpha
        let w = (1.0 / (a.passes as f64 + self.p.prior_w)).max(self.p.alpha);
        let w = if self.probe_pass { w.max(self.p.probe_w) } else { w };
        if self.probe_pass && !self.confirm {
            // a probe that alone beats the leader's average earns a second probe before the decision
            let lead = self.stats[d.other().idx()].rate();
            let ms_est = self.stats[d.idx()].ms.0.unwrap_or(ms);
            self.confirm = lead.is_some_and(|l| tokens as f64 / ms_est > l * self.p.margin);
        } else {
            self.confirm = false;
        }
        let a = &mut self.stats[d.idx()];
        a.tokens.add(tokens as f64, w);
        // the pass after a switch pays the catch-up (and, at the start, the deferred prompt rows): its time is not the drafter's
        if !self.fresh {
            a.ms.add(ms, w);
        }
        self.fresh = false;
        self.probe_pass = false;
    }

    /// Tokens per ms of each drafter (mtp, dflash), for the trace.
    pub fn summary(&self) -> String {
        let r = self.rates();
        format!(
            "passes mtp {} dflash {} switches {} probes {} (tokens/ms mtp {:.3} dflash {:.3})",
            self.passes_of(Drafter::Mtp),
            self.passes_of(Drafter::Dflash),
            self.switches,
            self.probes,
            r.0.unwrap_or(0.0),
            r.1.unwrap_or(0.0)
        )
    }

    pub fn rates(&self) -> (Option<f64>, Option<f64>) {
        (self.stats[0].rate(), self.stats[1].rate())
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
        CostParams { min_speedup: 1.03, window: 8, probe_every: 16, reprobe_tokens: 40, warmup: 2, alpha: 0.25, smooth: false }
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

    /// [`drive`] with the engine's sample delivery: a step's wall time is known when the next step begins, after that
    /// step's own `allow`.
    fn drive_lagged(g: &mut SpecGate, steps: usize, spec: impl Fn(usize) -> (usize, f64), plain: f64) -> String {
        let mut out = String::new();
        let mut held: Option<(bool, usize, f64)> = None;
        for i in 0..steps {
            let spec_step = g.allow(false);
            if let Some((s, t, ms)) = held.take() {
                g.cost_sample(s, t, ms);
            }
            if spec_step {
                let (t, ms) = spec(i);
                held = Some((true, t, ms));
                out.push('S');
            } else {
                held = Some((false, 1, plain));
                out.push('p');
            }
        }
        out
    }

    #[test]
    fn cost_gate_probes_once_when_samples_arrive_a_step_late() {
        let mut g = cost_gate(CostParams { probe_every: 1000, ..cp() });
        let t = drive_lagged(&mut g, 100, |_| (3, 13.7), 6.1);
        assert_eq!(t.matches('p').count(), 1, "one probe to learn the plain cost: {t}");
        assert_eq!(g.take_cost_events().probes, 1);
        // and the periodic probe is one step, not one per pass until its sample lands
        let mut g = cost_gate(cp());
        let t = drive_lagged(&mut g, 200, |_| (3, 13.7), 6.1);
        assert!(t.matches('p').count() <= 1 + 200 / 16, "{t}");
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
    fn cost_gate_smooth_falls_back_on_a_noisy_estimate_near_the_line() {
        // DFlash prose: 1 or 2 tokens per 101 ms pass against a 67.4 ms plain step, mean speedup 1.00 with jitter around the 1.03 line
        let noisy = |i: usize| (if i % 3 == 0 { 2 } else { 1 }, 101.0);
        let first_off = |smooth: bool| {
            let mut g = cost_gate(CostParams { smooth, probe_every: 1000, ..cp() });
            let t = drive(&mut g, 120, noisy, 67.4);
            (t.find("pppp"), t)
        };
        let (smooth, t) = first_off(true);
        assert!(smooth.is_some_and(|i| i <= 2 * (2 + 8 + 4)), "smooth gate falls back late: {t}");
        // a pass that pays stays on with the smooth decision too
        let mut g = cost_gate(CostParams { smooth: true, ..cp() });
        let t = drive(&mut g, 200, |_| (3, 101.0), 67.4);
        assert_eq!(g.take_cost_events().off, 0, "{t}");
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
    fn cost_gate_backs_off_longer_while_retries_keep_failing_and_starts_short_after_one_holds() {
        let mut g = cost_gate(cp());
        let runs = |t: &str| -> Vec<usize> {
            let mut v = vec![];
            let mut n = 0;
            for c in t.chars() {
                if c == 'p' {
                    n += 1;
                } else if n > 0 {
                    v.push(n);
                    n = 0;
                }
            }
            v
        };
        // bad text throughout: stretches of 40, 80, 160, 320, 640, 640 (16x the base would be 640)
        let t = drive(&mut g, 2500, |_| (2, 13.9), 6.1);
        let r: Vec<usize> = runs(&t).into_iter().filter(|n| *n > 10).collect();
        assert_eq!(&r[..5], &[40, 80, 160, 320, 640], "{t}");
        // a retry that holds (good text) resets the stretch: bad again afterwards starts at 40
        let mut g = cost_gate(cp());
        let mut t = drive(&mut g, 14, |_| (2, 13.9), 6.1);
        while g.cost_off() || !t.ends_with('S') {
            t += &drive(&mut g, 1, |_| (2, 13.9), 6.1);
        }
        t += &drive(&mut g, 150, |_| (3, 12.0), 6.1);
        assert!(!g.cost_off());
        let t2 = drive(&mut g, 80, |_| (2, 13.9), 6.1);
        let first = runs(&t2).into_iter().find(|n| *n > 10);
        assert_eq!(first, Some(40), "{t} / {t2}");
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
        let d = CostParams::for_backend("hip").with_env(Some("1"));
        assert_eq!(d.min_speedup, 1.03);
        assert!(d.enabled());
        for b in ["hip", "cuda", "metal", "cpu", ""] {
            assert!(!CostParams::for_backend(b).enabled(), "{b}: off by default");
            assert_eq!(CostParams::for_backend(b).with_env(Some("1")).min_speedup, 1.03, "{b}: forced on");
            assert!(CostParams::for_backend(b).with_env(Some("2.0,4,8,100")).enabled(), "{b}: tuple turns it on");
            assert!(!CostParams::for_backend(b).with_env(Some("0")).enabled());
        }
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

    /// Run the policy against fixed (tokens, ms) per drafter.
    fn drive_auto(p: &mut AutoPolicy, mtp: (usize, f64), dfl: (usize, f64), n: usize) -> (u32, u32) {
        for _ in 0..n {
            let d = p.pick();
            let (t, ms) = if d == Drafter::Mtp { mtp } else { dfl };
            p.record(d, t, ms);
        }
        (p.passes_of(Drafter::Mtp), p.passes_of(Drafter::Dflash))
    }

    #[test]
    fn auto_policy_follows_tokens_per_ms_and_probes_rarely() {
        // code-like: dflash 5.3 t / 39.7 ms beats mtp 2.74 t / 25.7 ms
        let mut p = AutoPolicy::new(AutoParams::default());
        let (m, d) = drive_auto(&mut p, (3, 25.7), (6, 39.7), 200);
        assert!(d >= 140, "dflash leads (after the seed, the dwell and the estimates settling): mtp {m} dflash {d}");
        assert!(m >= 2 && m <= 60, "mtp only for the seed and until the priors give way: {m}");
        // prose-like: mtp 2.04 / 25.7 beats dflash 2.5 / 39.7
        let mut p = AutoPolicy::new(AutoParams::default());
        let (m, d) = drive_auto(&mut p, (2, 25.7), (2, 39.7), 200);
        assert!(m >= 180 && d <= 8, "mtp leads: mtp {m} dflash {d}");
    }

    /// Policy over 600 passes of a drafter-specific cycle of tokens per pass; returns the DFlash passes.
    fn drive_noisy(mtp: &[usize], dfl: &[usize]) -> usize {
        let mut p = AutoPolicy::new(AutoParams::default());
        let mut n = 0;
        for i in 0..600 {
            let d = p.pick();
            let (t, ms) = if d == Drafter::Mtp { (mtp[i % mtp.len()], 25.7) } else { (dfl[i % dfl.len()], 39.7) };
            p.record(d, t, ms);
            n += usize::from(d == Drafter::Dflash);
        }
        n
    }

    #[test]
    fn auto_policy_stays_on_mtp_for_noisy_prose_and_on_dflash_for_noisy_code() {
        // prose: mtp 2.0 tokens / 25.7 ms against dflash 2.5 (0..5) / 39.7 ms: mtp is 20% ahead per ms, dflash passes are lucky now and then
        let dfl = drive_noisy(&[1, 2, 3, 2, 2, 2, 1, 3, 2, 2], &[0, 1, 2, 2, 3, 4, 5, 3, 2, 3]);
        assert!(dfl <= 600 / 12, "prose stays on mtp: {dfl} dflash passes of 600");
        // code: mtp 2.7 against dflash 5.3 (3..8)
        let dfl = drive_noisy(&[2, 3, 3, 2, 4, 2, 3, 3, 2, 3], &[3, 8, 4, 6, 5, 7, 3, 8, 4, 5]);
        assert!(dfl >= 600 * 85 / 100, "code moves to dflash: {dfl} dflash passes of 600");
    }

    #[test]
    fn auto_policy_env_knobs_and_priors() {
        let p = AutoParams::default().with_env(Some("alpha=0.05, dwell=4, leave=1.2, priorw=0, w=0.5, probe_far=2, log"));
        assert_eq!((p.alpha, p.dwell, p.leave_mtp, p.prior_w, p.probe_w, p.probe_far, p.log), (0.05, 4, 1.2, 0.0, 0.5, 2, true));
        assert_eq!(AutoParams::default().with_env(Some("alpha=x,nonsense=1")), AutoParams::default());
        // priors seed both drafters' rates before any pass; without them nothing is known until a pass is timed
        assert!(AutoPolicy::new(AutoParams::default()).rates().0.is_some());
        assert_eq!(AutoPolicy::new(AutoParams { prior_w: 0.0, ..AutoParams::default() }).rates(), (None, None));
    }

    #[test]
    fn auto_policy_switches_when_the_content_changes() {
        let mut p = AutoPolicy::new(AutoParams::default());
        drive_auto(&mut p, (2, 25.7), (2, 39.7), 100);
        // the text turns into code: dflash accepts 6 per pass now, found by a probe
        let (m0, d0) = (p.passes_of(Drafter::Mtp), p.passes_of(Drafter::Dflash));
        drive_auto(&mut p, (3, 25.7), (6, 39.7), 400);
        let (m1, d1) = (p.passes_of(Drafter::Mtp) - m0, p.passes_of(Drafter::Dflash) - d0);
        assert!(d1 > 250, "took over after the change: mtp {m1} dflash {d1}");
    }

    #[test]
    fn auto_policy_is_deterministic_and_forced_modes_hold() {
        let run = || {
            let mut p = AutoPolicy::new(AutoParams::default());
            (0..150)
                .map(|i| {
                    let d = p.pick();
                    p.record(d, 2 + (i % 3), if d == Drafter::Mtp { 25.0 } else { 40.0 });
                    d
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
        let mut p = AutoPolicy::new(AutoParams::default().with_env(Some("alt:3")));
        let seq: Vec<_> = (0..12)
            .map(|_| {
                let d = p.pick();
                p.record(d, 2, 30.0);
                d
            })
            .collect();
        assert_eq!(&seq[..7], &[Drafter::Mtp, Drafter::Mtp, Drafter::Mtp, Drafter::Dflash, Drafter::Dflash, Drafter::Dflash, Drafter::Mtp]);
        let mut p = AutoPolicy::new(AutoParams::default().with_env(Some("dflash")));
        assert!((0..9).all(|_| p.pick() == Drafter::Dflash));
        assert_eq!(SpecConfig::parse("auto", 7).unwrap(), SpecConfig { mode: SpecMode::Auto, k: 7 });
    }
}
