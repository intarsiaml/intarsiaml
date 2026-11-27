//! Lane ink-sampled-adapt (docs/results/2026-11-26-inkling2-sampled-adapt.md): the draft length of a sampled request that drafts with a
//! DFlash-family draft by the seed-exact keyed rule (`INTARSIA_SPEC_SAMPLED_DFLASH`, `engine/keyed.rs`), chosen per draft call from that
//! request's own acceptance. Drafts never change a seeded request's ids (every token is the plain step's draw at its position), so any
//! choice here, including no draft call at all (a plain step), leaves the output exactly plain sampling's; only the speed moves.
//!
//! Model: a draft call with cap `k` costs `d + 1 + r_m` plain steps (the draft, then a verify pass of `1 + m` rows, `m` the drafts the
//! draft's own cut kept, at most `k`; an empty call is the draft and a plain step) and yields `1 + accepted` tokens. Per request,
//! discounted counts estimate, for each row `j`, `q_j` = P(row j proposed) and `s_j` = P(row j proposed and accepted), so
//! `T(k) = 1 + sum_{j<=k} s_j` and `C(k) = d + 1 + sum_{j<=k} q_j (r_j - r_{j-1})`. The next call drafts with the cap that maximises
//! `T / C` if that beats a plain step by `margin`; otherwise the request takes `hold` plain steps (no draft call), then one probe call
//! at the full cap, and decides again (`hold` doubles after each probe that does not resume drafting, up to `hold_max`).
//!
//! The draft's context cache stays in sync through plain steps: every target pass stashes its tap row and the next draft call flushes the
//! stash first (`Model::dflash_after_forward`, `dflash_draft_rank`), the path empty drafts and the back-off gate already take.

use std::time::Instant;

/// Most rows the estimates keep (the DFlash family's largest `--spec-k`, [`super::DFLASH_K_MAX`]).
const ROWS_MAX: usize = super::DFLASH_K_MAX;

/// The policy's parameters (`INTARSIA_SPEC_SAMPLED_ADAPT`, `[tuning.<backend>] sampled_adapt`; [`AdaptParams::apply`] for the syntax).
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptParams {
    /// `d`: a draft call's cost, in plain steps.
    pub draft: f64,
    /// `r`: `rows[m - 1]` is the extra cost of a verify pass with `m` drafts over a plain step, in plain steps; past the end the last
    /// increment repeats.
    pub rows: Vec<f64>,
    /// `g`: discount of the counts per draft call (memory of about `1 / (1 - g)` calls).
    pub g: f64,
    /// `n0`: weight of the prior, in draft calls.
    pub n0: f64,
    /// `a0`: the prior's per-row acceptance (`q0_j = a0^(j-1)`, `s0_j = a0^j`: the draft proposes row `j` while the rows before it are
    /// accepted): a request starts at the full cap, as fixed k does.
    pub a0: f64,
    /// `margin`: draft only when the best cap's predicted tokens per plain-step time is at least `1 + margin`.
    pub margin: f64,
    /// `hold`: plain steps before the first probe call after drafting stopped.
    pub hold0: u32,
    /// `hold_max`: the most plain steps between two probe calls.
    pub hold_max: u32,
    /// `explore`: every `explore`-th draft call uses the full cap (0: never), so the rows above the chosen cap stay current.
    pub explore: u32,
}

impl AdaptParams {
    /// The built-in table of `backend` (`Backend::name`), from measured costs (docs/results/2026-11-26-inkling2-sampled-adapt.md section
    /// 2): HIP (one R9700) the verify probes of lanes ink-hip-spec / ink-hip-vrow and the DSpark profile over a 13.3 ms plain step; Metal
    /// (M5 Max) the keyed draft lines and a fit of lane ink-sampled's slot B. Other backends take HIP's (unmeasured).
    pub fn default_for(backend: &str) -> AdaptParams {
        let (draft, rows) = match backend {
            "metal" => (0.24, vec![0.32, 0.51, 0.70]),
            _ => (0.26, vec![0.26, 0.47, 0.60, 0.75, 0.91]),
        };
        AdaptParams { draft, rows, g: 0.85, n0: 1.0, a0: 0.6, margin: 0.02, hold0: 8, hold_max: 64, explore: 8 }
    }

    /// `r_m`, the extra cost of a verify pass with `m` drafts ([`AdaptParams::rows`]), `r_0 = 0`.
    pub fn row_cost(&self, m: usize) -> f64 {
        let n = self.rows.len();
        match (m, n) {
            (0, _) | (_, 0) => 0.0,
            (m, n) if m <= n => self.rows[m - 1],
            (m, 1) => self.rows[0] * m as f64,
            (m, n) => self.rows[n - 1] + (self.rows[n - 1] - self.rows[n - 2]) * (m - n) as f64,
        }
    }

    /// `self` with the `key=value` pairs of `v` (comma-separated) applied: `d=0.26`, `r=0.26/0.47/0.60` (slash-separated), `g`, `n0`,
    /// `a0`, `margin`, `hold`, `hold_max`, `explore`. Keys not given keep their value.
    pub fn apply(mut self, v: &str) -> Result<AdaptParams, String> {
        for kv in v.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (k, x) = kv.split_once('=').ok_or_else(|| format!("`{kv}`: want key=value"))?;
            let f = || x.trim().parse::<f64>().map_err(|_| format!("`{kv}`: not a number"));
            let u = || x.trim().parse::<u32>().map_err(|_| format!("`{kv}`: not an integer"));
            match k.trim() {
                "d" => self.draft = f()?,
                "r" => {
                    self.rows = x
                        .split('/')
                        .map(|r| r.trim().parse::<f64>().map_err(|_| format!("`{kv}`: not a list of numbers")))
                        .collect::<Result<_, _>>()?
                }
                "g" => self.g = f()?,
                "n0" => self.n0 = f()?,
                "a0" => self.a0 = f()?,
                "margin" => self.margin = f()?,
                "hold" => self.hold0 = u()?,
                "hold_max" => self.hold_max = u()?,
                "explore" => self.explore = u()?,
                other => return Err(format!("unknown key `{other}` (d, r, g, n0, a0, margin, hold, hold_max, explore)")),
            }
        }
        if !(self.draft >= 0.0 && (0.0..1.0).contains(&self.g) && self.n0 > 0.0 && (0.0..=1.0).contains(&self.a0) && self.hold0 >= 1)
            || self.rows.iter().any(|r| !(*r >= 0.0))
        {
            return Err(format!("out of range: {self:?} (d >= 0, r >= 0, 0 <= g < 1, n0 > 0, 0 <= a0 <= 1, hold >= 1)"));
        }
        self.hold_max = self.hold_max.max(self.hold0);
        Ok(self)
    }

    /// One line for the startup log.
    pub fn describe(&self) -> String {
        let r: Vec<String> = self.rows.iter().map(|r| format!("{r}")).collect();
        format!(
            "d={},r={},g={},n0={},a0={},margin={},hold={},hold_max={},explore={}",
            self.draft,
            r.join("/"),
            self.g,
            self.n0,
            self.a0,
            self.margin,
            self.hold0,
            self.hold_max,
            self.explore
        )
    }
}

/// What a sampled DFlash-family request does about its draft length.
#[derive(Clone, Debug, PartialEq)]
pub enum AdaptMode {
    /// Fixed cap (`--spec-k`, lane ink-sampled's behaviour): `INTARSIA_SPEC_SAMPLED_ADAPT=0`.
    Off,
    /// The policy with these parameters.
    On(AdaptParams),
    /// Diagnostic (`=drafts`): every step is a draft call at the full cap followed by a plain step (no verify pass); with
    /// `INTARSIA_SPEC_TRACE=1` each call's drafts are logged (the offline simulator's input).
    Drafts,
}

impl AdaptMode {
    /// `v` over `base`: `0` / `off`: [`AdaptMode::Off`]; empty, `1` / `on`: `base`; `drafts`: [`AdaptMode::Drafts`]; else `key=value`
    /// pairs applied to `base`'s parameters ([`AdaptParams::apply`]; over [`AdaptMode::Off`] they apply to `defaults`).
    pub fn parse_over(v: &str, base: AdaptMode, defaults: &AdaptParams) -> Result<AdaptMode, String> {
        match v.trim() {
            "0" | "off" => Ok(AdaptMode::Off),
            "" | "1" | "on" => Ok(match base {
                AdaptMode::Off => AdaptMode::On(defaults.clone()),
                b => b,
            }),
            "drafts" => Ok(AdaptMode::Drafts),
            kv => {
                let p = match base {
                    AdaptMode::On(p) => p,
                    _ => defaults.clone(),
                };
                p.apply(kv).map(AdaptMode::On)
            }
        }
    }

    /// The mode for `backend`: the built-in table ([`AdaptParams::default_for`]), then the recipe's `[tuning.<backend>] sampled_adapt`,
    /// then `INTARSIA_SPEC_SAMPLED_ADAPT` (an invalid value is reported and skipped).
    pub fn resolve(backend: &str, recipe: Option<&str>, env: Option<&str>) -> AdaptMode {
        let defaults = AdaptParams::default_for(backend);
        let mut m = AdaptMode::On(defaults.clone());
        for (src, v) in [("[tuning] sampled_adapt", recipe), ("INTARSIA_SPEC_SAMPLED_ADAPT", env)] {
            if let Some(v) = v {
                match AdaptMode::parse_over(v, m.clone(), &defaults) {
                    Ok(x) => m = x,
                    Err(e) => eprintln!("{src}={v:?}: {e}; ignored"),
                }
            }
        }
        m
    }

    /// One line for the startup log.
    pub fn describe(&self) -> String {
        match self {
            AdaptMode::Off => "off: fixed draft length (INTARSIA_SPEC_SAMPLED_ADAPT=0)".into(),
            AdaptMode::On(p) => format!("on ({}; INTARSIA_SPEC_SAMPLED_ADAPT=0 turns it off)", p.describe()),
            AdaptMode::Drafts => "drafts (diagnostic: every step drafts and takes a plain step)".into(),
        }
    }
}

/// `INTARSIA_SPEC_SAMPLED_ADAPT` (read once); `None` when unset.
pub fn sampled_adapt_env() -> Option<&'static str> {
    static V: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    V.get_or_init(|| std::env::var("INTARSIA_SPEC_SAMPLED_ADAPT").ok()).as_deref()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Drafting with the best cap.
    Draft,
    /// Plain steps without a draft call, this many left before the probe.
    Plain { left: u32 },
    /// The next draft call is a probe at the full cap.
    Probe,
}

/// One request's policy state ([`AdaptParams`]).
#[derive(Clone, Debug)]
pub struct AdaptK {
    p: AdaptParams,
    kmax: usize,
    n: Vec<f64>,
    q: Vec<f64>,
    s: Vec<f64>,
    phase: Phase,
    hold: u32,
    calls: u32,
}

/// A change of [`AdaptK`]'s phase, for the trace.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AdaptEvent {
    /// Drafting stopped: the best cap `k` predicts `rate` (tokens per plain-step time); `hold` plain steps follow.
    Stop { k: usize, rate: f64, hold: u32 },
    /// A probe call did not resume drafting: `hold` plain steps follow.
    Hold { k: usize, rate: f64, hold: u32 },
    /// Drafting resumed after a probe.
    Resume { k: usize, rate: f64 },
}

impl AdaptK {
    pub fn new(p: AdaptParams, kmax: usize) -> AdaptK {
        let kmax = kmax.clamp(1, ROWS_MAX);
        let hold = p.hold0;
        AdaptK { p, kmax, n: vec![0.0; kmax], q: vec![0.0; kmax], s: vec![0.0; kmax], phase: Phase::Draft, hold, calls: 0 }
    }

    /// `(q_j, s_j)` of row `j` (1-based).
    fn est(&self, j: usize) -> (f64, f64) {
        let i = j - 1;
        let w = self.n[i] + self.p.n0;
        ((self.q[i] + self.p.n0 * self.p.a0.powi(i as i32)) / w, (self.s[i] + self.p.n0 * self.p.a0.powi(j as i32)) / w)
    }

    /// `(T(k), C(k))`: the expected tokens and cost (in plain steps) of a draft call with cap `k`.
    pub fn predict(&self, k: usize) -> (f64, f64) {
        let (mut t, mut c) = (1.0, self.p.draft + 1.0);
        for j in 1..=k.min(self.kmax) {
            let (q, s) = self.est(j);
            t += s;
            c += q * (self.p.row_cost(j) - self.p.row_cost(j - 1));
        }
        (t, c)
    }

    /// The cap that maximises the predicted tokens per plain-step time, and that rate (a plain step's is 1).
    pub fn best(&self) -> (usize, f64) {
        let mut b = (self.kmax, 0.0);
        for k in 1..=self.kmax {
            let (t, c) = self.predict(k);
            let r = t / c;
            if r > b.1 + 1e-12 {
                b = (k, r);
            }
        }
        b
    }

    /// The cap of the next step, given the most it may draft (`k`: `--spec-k` after the context-room and `max_tokens` limits); 0 is a
    /// plain step without a draft call (counted against the hold).
    pub fn next(&mut self, k: usize) -> usize {
        if k == 0 {
            return 0;
        }
        match self.phase {
            Phase::Plain { left } if left > 0 => {
                self.phase = Phase::Plain { left: left - 1 };
                0
            }
            Phase::Plain { .. } => {
                self.phase = Phase::Probe;
                k
            }
            Phase::Probe => k,
            Phase::Draft => {
                if self.p.explore > 0 && (self.calls + 1) % self.p.explore == 0 {
                    k
                } else {
                    self.best().0.min(k)
                }
            }
        }
    }

    /// The draft call with cap `cap` kept `proposed` drafts and `accepted` of them were accepted (an empty call: 0, 0). Updates the
    /// estimates of rows `1..=cap` and decides whether the next steps draft.
    pub fn record(&mut self, cap: usize, proposed: usize, accepted: usize) -> Option<AdaptEvent> {
        self.calls = self.calls.saturating_add(1);
        let g = self.p.g;
        for i in 0..cap.min(self.kmax) {
            self.n[i] = g * self.n[i] + 1.0;
            self.q[i] = g * self.q[i] + f64::from(u8::from(proposed > i));
            self.s[i] = g * self.s[i] + f64::from(u8::from(accepted > i));
        }
        let (k, rate) = self.best();
        let ok = rate >= 1.0 + self.p.margin;
        match (self.phase, ok) {
            (Phase::Draft, true) => None,
            (_, true) => {
                self.phase = Phase::Draft;
                self.hold = self.p.hold0;
                Some(AdaptEvent::Resume { k, rate })
            }
            (Phase::Draft, false) => {
                self.phase = Phase::Plain { left: self.hold };
                Some(AdaptEvent::Stop { k, rate, hold: self.hold })
            }
            (_, false) => {
                self.hold = self.hold.saturating_mul(2).min(self.p.hold_max);
                self.phase = Phase::Plain { left: self.hold };
                Some(AdaptEvent::Hold { k, rate, hold: self.hold })
            }
        }
    }

    /// Whether the request is drafting now (not holding plain steps or waiting for a probe).
    pub fn drafting(&self) -> bool {
        self.phase == Phase::Draft
    }
}

/// The policy as a `Seq` (engine.rs) runs it: the state of [`AdaptMode::On`] (none for [`AdaptMode::Drafts`]), the cap of the call in
/// flight, and the `INTARSIA_SPEC_TRACE=1` step timer (`adapt:` lines: every step's wall time and kind, every change of phase).
#[derive(Clone, Debug)]
pub struct AdaptRun {
    k: Option<AdaptK>,
    cap: usize,
    /// (start of the current step, generated tokens then); the step's kind and drafts.
    mark: Option<(Instant, usize)>,
    kind: (&'static str, usize),
}

impl AdaptRun {
    /// `None` for [`AdaptMode::Off`] (fixed k). `kmax`: the request's `--spec-k`.
    pub fn new(mode: AdaptMode, kmax: usize) -> Option<AdaptRun> {
        let k = match mode {
            AdaptMode::Off => return None,
            AdaptMode::On(p) => Some(AdaptK::new(p, kmax)),
            AdaptMode::Drafts => None,
        };
        Some(AdaptRun { k, cap: 0, mark: None, kind: ("plain", 0) })
    }

    /// [`AdaptMode::Drafts`]: the draft call's proposals are logged and dropped, the step is a plain step.
    pub fn drafts_only(&self) -> bool {
        self.k.is_none()
    }

    /// A step that may draft begins ([`AdaptK::next`]); `generated`: tokens so far. Returns the cap (0: a plain step).
    pub fn next(&mut self, k: usize, generated: usize) -> usize {
        let cap = match self.k.as_mut() {
            Some(a) => a.next(k),
            None => k,
        };
        self.cap = cap;
        if cap > 0 {
            self.kind = ("call", cap);
        }
        let _ = generated;
        cap
    }

    /// The draft call of this step came back empty: the step is a plain step.
    pub fn empty(&mut self, generated: usize) {
        self.kind = ("empty", 0);
        self.record(0, 0, generated);
    }

    /// The draft call of this step kept `proposed` drafts, `accepted` of them accepted by the keyed verify.
    pub fn pass(&mut self, proposed: usize, accepted: usize, generated: usize) {
        self.kind = ("pass", proposed);
        self.record(proposed, accepted, generated);
    }

    /// [`AdaptMode::Drafts`]: the call's drafts, logged (trace); the draws start at generated token `generated + 1`.
    pub fn drafts(&mut self, drafts: &[u32], generated: usize, pos: usize) {
        self.kind = ("drafts", drafts.len());
        if super::trace_enabled() {
            eprintln!("adapt: drafts at token {} pos {pos} cap {} {drafts:?}", generated + 1, self.cap);
        }
    }

    fn record(&mut self, proposed: usize, accepted: usize, generated: usize) {
        let Some(a) = self.k.as_mut() else { return };
        let ev = a.record(self.cap, proposed, accepted);
        if let (Some(ev), true) = (ev, super::trace_enabled()) {
            eprintln!("adapt: token {} {ev:?}", generated + 1);
        }
    }

    /// Trace only: a step begins (`Seq::plan_step` with no verified token queued); logs the step before it (kind, drafts, tokens, ms).
    pub fn tick(&mut self, generated: usize) {
        if !super::trace_enabled() {
            return;
        }
        let now = Instant::now();
        if let Some((t0, g0)) = self.mark {
            eprintln!(
                "adapt: step {} {} {} tokens {:.3} ms",
                self.kind.0,
                self.kind.1,
                generated.saturating_sub(g0),
                now.duration_since(t0).as_secs_f64() * 1e3
            );
        }
        self.mark = Some((now, generated));
        self.kind = ("plain", 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> AdaptParams {
        AdaptParams::default_for("hip")
    }

    #[test]
    fn parse_and_resolve() {
        let d = p();
        assert_eq!(AdaptMode::parse_over("0", AdaptMode::On(d.clone()), &d), Ok(AdaptMode::Off));
        assert_eq!(AdaptMode::parse_over("off", AdaptMode::On(d.clone()), &d), Ok(AdaptMode::Off));
        assert_eq!(AdaptMode::parse_over("1", AdaptMode::Off, &d), Ok(AdaptMode::On(d.clone())));
        assert_eq!(AdaptMode::parse_over("drafts", AdaptMode::Off, &d), Ok(AdaptMode::Drafts));
        let AdaptMode::On(x) = AdaptMode::parse_over("d=0.5, r=0.1/0.2,margin=0.05,hold=4", AdaptMode::On(d.clone()), &d).unwrap() else {
            panic!()
        };
        assert_eq!((x.draft, x.rows.clone(), x.margin, x.hold0, x.g), (0.5, vec![0.1, 0.2], 0.05, 4, d.g));
        assert!(AdaptMode::parse_over("d=x", AdaptMode::Off, &d).is_err());
        assert!(AdaptMode::parse_over("k=3", AdaptMode::Off, &d).is_err());
        assert!(AdaptMode::parse_over("g=1", AdaptMode::Off, &d).is_err());
        // env over recipe over the built-in table; a bad value is skipped
        assert_eq!(AdaptMode::resolve("hip", None, None), AdaptMode::On(d.clone()));
        assert_eq!(AdaptMode::resolve("hip", Some("off"), None), AdaptMode::Off);
        assert_eq!(AdaptMode::resolve("hip", Some("off"), Some("1")), AdaptMode::On(d.clone()));
        let AdaptMode::On(m) = AdaptMode::resolve("metal", Some("margin=0.1"), Some("hold=2")) else { panic!() };
        assert_eq!((m.draft, m.margin, m.hold0), (AdaptParams::default_for("metal").draft, 0.1, 2));
        assert_eq!(AdaptMode::resolve("hip", None, Some("bogus")), AdaptMode::On(d));
    }

    #[test]
    fn row_cost_extends_the_last_increment() {
        let x = p();
        assert_eq!(x.row_cost(0), 0.0);
        assert_eq!(x.row_cost(5), 0.91);
        assert!((x.row_cost(7) - (0.91 + 2.0 * 0.16)).abs() < 1e-9);
        let one = AdaptParams { rows: vec![0.3], ..p() };
        assert!((one.row_cost(3) - 0.9).abs() < 1e-9);
    }

    #[test]
    fn starts_at_the_full_cap_and_keeps_it_when_drafts_are_accepted() {
        let mut a = AdaptK::new(p(), 5);
        assert_eq!(a.next(5), 5, "the prior drafts at k_max");
        for _ in 0..50 {
            let k = a.next(5);
            assert_eq!(k, 5);
            assert_eq!(a.record(k, 5, 5), None);
        }
        // a last row that is proposed and never accepted only costs: the cap drops below it
        for _ in 0..30 {
            let k = a.next(5);
            a.record(k, k, k.min(4));
        }
        assert_eq!(a.next(5), 4);
        assert!(a.drafting());
        // the room / max_tokens limit wins
        assert_eq!(a.next(2), 2);
    }

    #[test]
    fn empty_calls_stop_drafting_then_probe_with_a_doubling_hold() {
        let mut a = AdaptK::new(p(), 5);
        let mut calls = 0;
        while a.drafting() {
            let k = a.next(5);
            assert!(k > 0);
            a.record(k, 0, 0);
            calls += 1;
            assert!(calls < 10, "never stopped");
        }
        let mut holds = vec![];
        for _ in 0..6 {
            let mut plain = 0;
            loop {
                let k = a.next(5);
                if k > 0 {
                    assert_eq!(k, 5, "a probe drafts at the full cap");
                    break;
                }
                plain += 1;
            }
            holds.push(plain);
            a.record(5, 1, 0);
        }
        assert_eq!(holds, vec![8, 16, 32, 64, 64, 64]);
        // a run of good probes resumes drafting, and the hold starts short again
        let mut resumed = false;
        for _ in 0..200 {
            let k = a.next(5);
            if k > 0 {
                if let Some(AdaptEvent::Resume { .. }) = a.record(k, 5, 5) {
                    resumed = true;
                    break;
                }
            }
        }
        assert!(resumed);
        while a.drafting() {
            let k = a.next(5);
            a.record(k, 0, 0);
        }
        let mut plain = 0;
        while a.next(5) == 0 {
            plain += 1;
        }
        assert_eq!(plain, 8);
    }

    #[test]
    fn costly_rows_that_are_rarely_accepted_lower_the_cap() {
        // every call proposes 5 drafts, only the first is often accepted; rows cost a lot: the best cap is 1
        let x = AdaptParams { draft: 0.05, rows: vec![0.1, 0.6, 1.1, 1.6, 2.1], explore: 0, ..p() };
        let mut a = AdaptK::new(x, 5);
        for i in 0..40 {
            let k = a.next(5);
            a.record(k, k, usize::from(i % 4 != 0));
        }
        assert!(a.drafting());
        assert_eq!(a.next(5), 1);
        // explore: every 4th call at the full cap
        let mut a = AdaptK::new(AdaptParams { explore: 4, ..a.p.clone() }, 5);
        let caps: Vec<usize> = (0..12)
            .map(|i| {
                let k = a.next(5);
                a.record(k, k, usize::from(i % 4 != 0));
                k
            })
            .collect();
        assert!(caps[3] == 5 && caps[7] == 5 && caps[11] == 5, "{caps:?}");
        assert!(caps[8..11].iter().all(|&k| k < 5), "{caps:?}");
    }
}
