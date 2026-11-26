//! The direct check of tensor-parallel partial sums, as pure host code: data, the derived fp32 reduction-order bounds,
//! the mutants and the verdicts. Shared by `tp_partial_sum.rs` (the real HIP kernels and the P2P all-reduce, needs two
//! GPUs) and `tp_partial_math.rs` (CPU emulations of a GPU GEMV, runs anywhere, proves the thresholds here are met by
//! different legitimate summation orders and are crossed by the mutants).
//!
//! The setting: a row-parallel projection `y = x W^T`, `x [m, K]`, `W [n, K]`, bf16 inputs, f32 accumulation. tp=1 is
//! one GEMV over the full K. tp=2 gives each rank half of K (`x[:, r*K/2..]`, `W[:, r*K/2..]`, the layout of
//! `ShardAxis::Cols` for out_proj / down_proj), each rank produces an f32 partial `[m, n]`, and the all-reduce
//! (`Comm::all_reduce_sum_into`, f32 partial -> f32 or bf16 out) adds the two.
//!
//! # The bound, per output element
//!
//! A bf16 x bf16 product has at most 16 significant bits, so it is exact in f32. What is left is the f32 summation of K
//! exact terms. For ANY order of K-1 f32 additions (sequential, pairwise, lane-strided then tree, split in halves then
//! one add, ...) with round-to-nearest, `|fl(s) - s| <= gamma_{K-1} * A` where `A = sum_i |x_i w_i|`, `gamma_n = n u /
//! (1 - n u)`, `u = 2^-24` (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., eq. 3.4 and sec. 4.2). tp=1
//! and tp=2 are two such orders over the same terms, so
//!
//!   BOUND_DET   |y2 - y1| <= 2 gamma_K A                      (deterministic; gamma_K is the spec's looser K, not K-1)
//!
//! and each of them also satisfies `|y - y_exact| <= gamma_K A` on its own, which the driver checks first: it is the
//! test that the kernel under test really is an IEEE-style f32 accumulation (a matrix-core path that truncates, or a
//! kernel that accumulates in lower precision, fails here and not as a mysterious TP failure).
//!
//! The deterministic bound is linear in K, so for K = 5120 it is 6.1e-4 of `A`: a 0.1% scale of one half partial (an
//! effect of 5e-4 of `y` on same-sign data) hides under it. The second, tighter bound is the probabilistic one for the
//! same sums (Higham & Mary, "A new approach to probabilistic rounding error analysis", SIAM J. Sci. Comput. 41(5),
//! 2019: rounding errors that are independent with zero mean, as round-to-nearest on this data gives): with
//! probability at least `1 - 2 exp(-lambda^2 (1-u)^2 / 2)` per dot product (lambda = 6: 3e-8)
//!
//!   BOUND_PROB  |y2 - y1| <= 2 gammaT_K(lambda) A,   gammaT_K = exp(lambda sqrt(K) u + K u^2 / (1 - u)) - 1
//!
//! It is sqrt(K) instead of K, i.e. 7.5x / 12x smaller at K = 2048 / 5120 than BOUND_DET at lambda = 6. It rests on an
//! assumption (zero-mean independent errors), not on a theorem about every kernel, so the driver reports both and the
//! tests state which mutant needs which.
//!
//! # bf16 outputs
//!
//! The all-reduce can round the sum once into bf16, tp=1 rounds its f32 result to bf16 (the shipped bf16 path). The two
//! bf16 values differ only when a rounding boundary lies between `y1` and `y2`. With the position of `y1` against the
//! bf16 grid uniform, that happens with probability at most `p_j = min(1, B_j / ulp_j)`, `B_j` the bound above (DET or
//! PROB) and `ulp_j` the bf16 spacing at the smaller of `|y_j|` and `|y_j| - B_j`. The number of differing elements is then
//! stochastically below a sum of Bernoulli(p_j): the driver allows `E + 5 sigma`, `E = sum p_j`, `sigma^2 = sum
//! p_j (1 - p_j)`.
//!
//! # What this does not cover
//!
//! * It bounds the reduction ORDER only, for the sums of one projection. It says nothing about the other TP
//!   collectives (all-gather of the vocab-parallel head), attention, or the norms.
//! * "Uniform position against the grid" is an assumption of the bf16 fraction test; it holds for random data, it is
//!   not a theorem. The f32 bounds do not need it.
//! * With cancellation (random-sign data) `A` is much larger than `|y|`, so both bounds are loose in relative terms
//!   and a small scale mutant can sit under them. The driver asserts only the large mutants (dropped partial, swapped
//!   halves) there and reports the scale mutants.
//! * A partial scale of `1 + e` is only visible when `e |p_1| > 2 gammaT_K A`; on same-sign data (`A = |y|`,
//!   `p_1 = y / 2`) that is `e > 4 gammaT_K` = 6.5e-5 (K = 2048) / 1.0e-4 (K = 5120) under BOUND_PROB, 4.9e-4 / 1.2e-3 under
//!   BOUND_DET. Smaller than that is not killable by a summation-order tolerance, only by a bitwise or exact-input check.
#![allow(dead_code)]

use intarsia_core::dtype::{decode_to_f32, encode_from_f32};
use intarsia_core::rng::SplitMix64;
use intarsia_core::DType;

pub const U: f64 = 1.0 / 16_777_216.0; // 2^-24
pub const LAMBDA: f64 = 6.0;

pub fn gamma(n: f64) -> f64 {
    n * U / (1.0 - n * U)
}

pub fn gamma_tilde(n: f64) -> f64 {
    (LAMBDA * n.sqrt() * U + n * U * U / (1.0 - U)).exp_m1()
}

pub fn round_bf16(v: &[f32]) -> Vec<f32> {
    let mut bytes = vec![0u8; v.len() * 2];
    encode_from_f32(DType::Bf16, v, &mut bytes);
    let mut out = vec![0f32; v.len()];
    decode_to_f32(DType::Bf16, &bytes, &mut out);
    out
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    /// x, w uniform in [0.5, 1.5): every term positive, `A = |y|`, the partials are as large as they get.
    SameSign,
    /// x, w standard normal: cancellation, `A` is ~30x `|y|` at K = 2048.
    RandomSign,
}

pub struct Case {
    pub kind: Kind,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// `[m, k]`, already bf16-representable
    pub x: Vec<f32>,
    /// `[n, k]`, already bf16-representable
    pub w: Vec<f32>,
    /// exact `y = x W^T`, `[m, n]` (f64 accumulation of exact products)
    pub exact: Vec<f64>,
    /// `A = sum_i |x_i w_i|`, `[m, n]`
    pub abs: Vec<f64>,
}

impl Case {
    pub fn new(kind: Kind, m: usize, n: usize, k: usize, seed: u64) -> Case {
        assert!(k % 2 == 0);
        let mut rng = SplitMix64::new(seed);
        let gen = |rng: &mut SplitMix64, len: usize| -> Vec<f32> {
            let v: Vec<f32> = match kind {
                Kind::SameSign => (0..len).map(|_| 0.5 + rng.next_f32()).collect(),
                Kind::RandomSign => rng.normal_vec(len, 1.0),
            };
            round_bf16(&v)
        };
        let x = gen(&mut rng, m * k);
        let w = gen(&mut rng, n * k);
        let (mut exact, mut abs) = (vec![0f64; m * n], vec![0f64; m * n]);
        for i in 0..m {
            let xr = &x[i * k..(i + 1) * k];
            for j in 0..n {
                let wr = &w[j * k..(j + 1) * k];
                let (mut s, mut a) = (0f64, 0f64);
                for (p, q) in xr.iter().zip(wr) {
                    let t = *p as f64 * *q as f64;
                    s += t;
                    a += t.abs();
                }
                exact[i * n + j] = s;
                abs[i * n + j] = a;
            }
        }
        Case { kind, m, n, k, x, w, exact, abs }
    }

    /// `x[:, r*K/2 .. (r+1)*K/2]`, `[m, k/2]`
    pub fn x_half(&self, r: usize) -> Vec<f32> {
        let h = self.k / 2;
        (0..self.m).flat_map(|i| self.x[i * self.k + r * h..i * self.k + (r + 1) * h].iter().copied()).collect()
    }

    /// `W[:, r*K/2 .. (r+1)*K/2]`, `[n, k/2]` (the `ShardAxis::Cols` shard)
    pub fn w_half(&self, r: usize) -> Vec<f32> {
        let h = self.k / 2;
        (0..self.n).flat_map(|j| self.w[j * self.k + r * h..j * self.k + (r + 1) * h].iter().copied()).collect()
    }

    pub fn bound_det(&self, i: usize) -> f64 {
        2.0 * gamma(self.k as f64) * self.abs[i]
    }

    pub fn bound_prob(&self, i: usize) -> f64 {
        2.0 * gamma_tilde(self.k as f64) * self.abs[i]
    }
}

pub struct Reduced {
    /// the f32 output of the f32 -> f32 all-reduce, as rank 0 / rank 1 hold it
    pub f32_out: [Vec<f32>; 2],
    /// the f32 -> bf16 all-reduce output, decoded to f32
    pub bf16_out: [Vec<f32>; 2],
}

/// What the driver needs from an implementation: the GEMV and the all-reduce under test.
pub trait Engine {
    fn name(&self) -> String;
    /// Called once per case (upload the weights).
    fn prepare(&mut self, c: &Case);
    /// tp=1: `x @ W^T` over the full K, f32 `[m, n]`.
    fn tp1(&mut self, c: &Case) -> Vec<f32>;
    /// Each rank's f32 partial `[m, n]` from its `x` half `[m, k/2]` and its cached weight half.
    fn partials(&mut self, c: &Case, x_halves: [Vec<f32>; 2]) -> [Vec<f32>; 2];
    /// The all-reduce of the two partials into f32 and into bf16.
    fn reduce(&mut self, c: &Case, partials: &[Vec<f32>; 2]) -> Reduced;
}

fn bf16_ulp_below(y: f64, b: f64) -> f64 {
    let lo = (y.abs() - b).max(f32::MIN_POSITIVE as f64);
    2f64.powi(lo.log2().floor() as i32 - 7)
}

pub struct Verdict {
    pub label: String,
    /// max over elements of |y2 - y1| / BOUND_DET, / BOUND_PROB
    pub ratio_det: f64,
    pub ratio_prob: f64,
    /// differing bf16 elements, out of `total`, and what the two derived bounds allow
    pub bf16_diff: usize,
    pub total: usize,
    pub allowed_det: f64,
    pub allowed_prob: f64,
    pub max_abs_diff: f64,
}

impl Verdict {
    pub fn line(&self, c: &Case) -> String {
        format!(
            "  {:<22} max|y2-y1| {:.3e}  /BOUND_DET {:>9.3}  /BOUND_PROB {:>9.3}  bf16 differ {:>5}/{} (allowed DET {:.1}, PROB {:.1})  [{:?} m={} n={} k={}]",
            self.label,
            self.max_abs_diff,
            self.ratio_det,
            self.ratio_prob,
            self.bf16_diff,
            self.total,
            self.allowed_det,
            self.allowed_prob,
            c.kind,
            c.m,
            c.n,
            c.k
        )
    }
}

fn verdict(c: &Case, label: &str, y1: &[f32], r: &Reduced) -> Verdict {
    let y2 = &r.f32_out[0];
    let (mut rd, mut rp, mut mx) = (0f64, 0f64, 0f64);
    for i in 0..y2.len() {
        let d = (y2[i] as f64 - y1[i] as f64).abs();
        mx = mx.max(d);
        rd = rd.max(d / c.bound_det(i));
        rp = rp.max(d / c.bound_prob(i));
    }
    let y1b = round_bf16(y1);
    let bf16_diff = r.bf16_out[0].iter().zip(&y1b).filter(|(a, b)| a != b).count();
    let allowed = |bound: &dyn Fn(usize) -> f64| {
        let (mut e, mut v) = (0f64, 0f64);
        for i in 0..y1.len() {
            let p = (bound(i) / bf16_ulp_below(c.exact[i], bound(i))).min(1.0);
            e += p;
            v += p * (1.0 - p);
        }
        e + 5.0 * v.sqrt()
    };
    Verdict {
        label: label.to_string(),
        ratio_det: rd,
        ratio_prob: rp,
        bf16_diff,
        total: y1.len(),
        allowed_det: allowed(&|i| c.bound_det(i)),
        allowed_prob: allowed(&|i| c.bound_prob(i)),
        max_abs_diff: mx,
    }
}

pub struct Report {
    pub lines: Vec<String>,
    pub failures: Vec<String>,
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).fold(0.0, f64::max)
}

/// Run the clean check and the mutants of one case; nothing panics here, the caller prints `lines` and asserts that
/// `failures` is empty (so one GPU window shows every number, not just the first failing one).
pub fn run_case(e: &mut dyn Engine, c: &Case) -> Report {
    let mut rep = Report { lines: vec![], failures: vec![] };
    let tag = format!("{:?} m={} n={} k={}", c.kind, c.m, c.n, c.k);
    let name = e.name();
    let fail = |rep: &mut Report, what: String| rep.failures.push(format!("[{name}] {tag}: {what}"));
    e.prepare(c);
    let y1 = e.tp1(c);
    let clean_p = e.partials(c, [c.x_half(0), c.x_half(1)]);
    let clean = e.reduce(c, &clean_p);

    // non-vacuity: the reference is real data, the partials are real and add up to about the reference
    let y1_max = y1.iter().fold(0f32, |a, v| a.max(v.abs()));
    if !(y1_max > 0.0) {
        fail(&mut rep, "tp1 output is all zero".into());
    }
    if c.kind == Kind::SameSign && y1.iter().any(|v| !(*v > 0.0)) {
        fail(&mut rep, "tp1 output has a non-positive element on same-sign data".into());
    }
    let p1_max = clean_p[1].iter().fold(0f32, |a, v| a.max(v.abs()));
    if !(p1_max > 0.0) || !(clean_p[0].iter().any(|v| *v != 0.0)) {
        fail(&mut rep, "a rank's partial is all zero".into());
    }
    if clean.f32_out[0] != clean.f32_out[1] || clean.bf16_out[0] != clean.bf16_out[1] {
        fail(&mut rep, "the two ranks hold different all-reduce outputs".into());
    }

    // the kernels conform to the f32 bound this is derived from
    let g = gamma(c.k as f64);
    let conf = |y: &[f32]| (0..y.len()).map(|i| (y[i] as f64 - c.exact[i]).abs() / (g * c.abs[i])).fold(0.0, f64::max);
    let (c1, c2) = (conf(&y1), conf(&clean.f32_out[0]));
    rep.lines.push(format!("  {tag}: |y - exact| / (gamma_K A): tp1 {c1:.3e}, tp2 {c2:.3e} (must be <= 1)"));
    if c1 > 1.0 || c2 > 1.0 {
        fail(&mut rep, format!("a kernel is not an f32 accumulation within gamma_K * sum|xw| of the exact sum (tp1 {c1:.3e}, tp2 {c2:.3e})"));
    }

    let v = verdict(c, "clean", &y1, &clean);
    rep.lines.push(v.line(c));
    if v.ratio_det > 1.0 {
        fail(&mut rep, format!("clean tp2 vs tp1 exceeds BOUND_DET ({:.3})", v.ratio_det));
    }
    if v.ratio_prob > 1.0 {
        fail(&mut rep, format!("clean tp2 vs tp1 exceeds BOUND_PROB ({:.3})", v.ratio_prob));
    }
    if v.bf16_diff as f64 > v.allowed_det {
        fail(&mut rep, format!("clean bf16 outputs differ in {} elements, allowed {:.1}", v.bf16_diff, v.allowed_det));
    }

    // mutants
    struct Mutant {
        label: &'static str,
        partials: [Vec<f32>; 2],
        /// scale applied to rank 1 (None: not a scale mutant)
        eps: Option<f64>,
        /// killed by BOUND_DET on same-sign data at every K tested
        det_kills_same_sign: bool,
    }
    let scaled = |f: f32| [clean_p[0].clone(), clean_p[1].iter().map(|v| *v * f).collect()];
    let mutants = [
        Mutant { label: "scale:1.003 rank 1", partials: scaled(1.003), eps: Some(0.003), det_kills_same_sign: true },
        Mutant { label: "scale:1.001 rank 1", partials: scaled(1.001), eps: Some(0.001), det_kills_same_sign: false },
        Mutant { label: "drop rank 1 partial", partials: [clean_p[0].clone(), vec![0.0; clean_p[1].len()]], eps: None, det_kills_same_sign: true },
        Mutant { label: "swap x halves", partials: e.partials(c, [c.x_half(1), c.x_half(0)]), eps: None, det_kills_same_sign: true },
    ];
    for mu in mutants {
        let changed = max_abs_diff(&mu.partials[1], &clean_p[1]).max(max_abs_diff(&mu.partials[0], &clean_p[0]));
        let expect_change = mu.eps.map_or(0.0, |eps| 0.5 * eps * p1_max as f64);
        if !(changed > expect_change) || !(changed > 0.0) {
            fail(&mut rep, format!("{}: the mutant did not change a partial enough (max |dp| {changed:.3e}, expected > {expect_change:.3e})", mu.label));
        }
        let r = e.reduce(c, &mu.partials);
        if r.f32_out[0] == clean.f32_out[0] {
            fail(&mut rep, format!("{}: the all-reduce output is identical to the clean one", mu.label));
        }
        let v = verdict(c, mu.label, &y1, &r);
        rep.lines.push(v.line(c));
        let big = mu.eps.is_none(); // dropped partial / swapped halves: huge effects
        match c.kind {
            Kind::SameSign => {
                if !(v.ratio_prob > 1.0) {
                    fail(&mut rep, format!("{}: not killed by BOUND_PROB ({:.3})", mu.label, v.ratio_prob));
                }
                if mu.det_kills_same_sign && !(v.ratio_det > 1.0) {
                    fail(&mut rep, format!("{}: not killed by BOUND_DET ({:.3})", mu.label, v.ratio_det));
                }
                if !(v.bf16_diff as f64 > v.allowed_prob) {
                    fail(&mut rep, format!("{}: bf16 outputs differ in {} elements, within the PROB allowance {:.1}", mu.label, v.bf16_diff, v.allowed_prob));
                }
            }
            Kind::RandomSign => {
                if big && !(v.ratio_det > 1.0) {
                    fail(&mut rep, format!("{}: not killed by BOUND_DET even with cancellation ({:.3})", mu.label, v.ratio_det));
                }
            }
        }
    }
    rep
}
