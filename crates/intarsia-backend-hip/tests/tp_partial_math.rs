//! The host half of the direct tensor-parallel partial-sum check (`tp_partial_sum.rs` runs the same driver on the real
//! HIP GEMV and the P2P all-reduce). Here the GEMV is emulated in f32 with three very different, all legitimate,
//! summation orders, so this runs on any machine and shows (a) the derived bounds hold for every order, on same-sign and
//! random-sign data, K = 2048 and 5120, and (b) the mutants cross them. See `common/tp_partial.rs` for the derivation
//! and for what the bounds do not cover. No GPU, no `ik_hip`.

mod common;
use common::tp_partial::*;

/// How an emulated kernel adds up K terms in f32 (the products are exact).
#[derive(Clone, Copy, Debug)]
enum Order {
    /// one accumulator, left to right: the worst realistic order, error grows with K
    Sequential,
    /// `lanes` strided accumulators, then a pairwise tree: the shape of a GPU GEMV block
    LanesTree(usize),
    /// 8 interleaved accumulators over a K/8 split, then left to right: a vectorised CPU loop
    Eight,
}

fn dot(order: Order, x: &[f32], w: &[f32]) -> f32 {
    match order {
        Order::Sequential => x.iter().zip(w).fold(0f32, |a, (p, q)| a + p * q),
        Order::LanesTree(l) => {
            let mut acc = vec![0f32; l];
            for (i, (p, q)) in x.iter().zip(w).enumerate() {
                acc[i % l] += p * q;
            }
            let mut n = l;
            while n > 1 {
                n /= 2;
                for i in 0..n {
                    acc[i] += acc[i + n];
                }
            }
            acc[0]
        }
        Order::Eight => {
            let mut acc = [0f32; 8];
            for (i, (p, q)) in x.iter().zip(w).enumerate() {
                acc[i % 8] += p * q;
            }
            acc.iter().fold(0f32, |a, v| a + v)
        }
    }
}

struct Emu {
    order: Order,
    /// the tp=2 kernel uses another order than tp=1 (the kernel selection differs for K/2)
    order_tp2: Order,
    /// a deliberate hidden bug of the "hardware": rank 1's partial is scaled before the reduce (the harness must notice)
    hidden_scale: f32,
    w_half: [Vec<f32>; 2],
}

impl Emu {
    fn new(order: Order, order_tp2: Order) -> Emu {
        Emu { order, order_tp2, hidden_scale: 1.0, w_half: [vec![], vec![]] }
    }
}

fn gemv(order: Order, m: usize, n: usize, k: usize, x: &[f32], w: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            y[i * n + j] = dot(order, &x[i * k..(i + 1) * k], &w[j * k..(j + 1) * k]);
        }
    }
    y
}

impl Engine for Emu {
    fn name(&self) -> String {
        format!("emu {:?}/{:?}", self.order, self.order_tp2)
    }
    fn prepare(&mut self, c: &Case) {
        self.w_half = [c.w_half(0), c.w_half(1)];
    }
    fn tp1(&mut self, c: &Case) -> Vec<f32> {
        gemv(self.order, c.m, c.n, c.k, &c.x, &c.w)
    }
    fn partials(&mut self, c: &Case, xh: [Vec<f32>; 2]) -> [Vec<f32>; 2] {
        let h = c.k / 2;
        let mut p = [0, 1].map(|r| gemv(self.order_tp2, c.m, c.n, h, &xh[r], &self.w_half[r]));
        for v in p[1].iter_mut() {
            *v *= self.hidden_scale;
        }
        p
    }
    fn reduce(&mut self, _c: &Case, p: &[Vec<f32>; 2]) -> Reduced {
        let s: Vec<f32> = p[0].iter().zip(&p[1]).map(|(a, b)| a + b).collect();
        let b = round_bf16(&s);
        Reduced { f32_out: [s.clone(), s], bf16_out: [b.clone(), b] }
    }
}

fn show(rep: &Report) {
    for l in &rep.lines {
        eprintln!("{l}");
    }
}

const ORDERS: [(Order, Order); 3] = [
    (Order::LanesTree(256), Order::LanesTree(256)),
    (Order::Sequential, Order::Eight),
    (Order::Eight, Order::LanesTree(64)),
];

#[test]
fn bounds_hold_for_legitimate_orders_and_mutants_cross_them() {
    eprintln!("gamma_2048 {:.3e}  gamma_5120 {:.3e}  gammaT_2048 {:.3e}  gammaT_5120 {:.3e}", gamma(2048.0), gamma(5120.0), gamma_tilde(2048.0), gamma_tilde(5120.0));
    let mut failures = vec![];
    for kind in [Kind::SameSign, Kind::RandomSign] {
        for (m, n, k) in [(1usize, 512usize, 2048usize), (2, 512, 5120)] {
            let c = Case::new(kind, m, n, k, 17 + k as u64);
            for (o1, o2) in ORDERS {
                let rep = run_case(&mut Emu::new(o1, o2), &c);
                eprintln!("{o1:?} / {o2:?}");
                show(&rep);
                failures.extend(rep.failures);
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The harness must not pass an engine whose "all-reduce" is wrong on its own: a hidden 0.3% / 0.1% scale of rank 1's
/// partial shows up as a failure of the CLEAN row (the clean tp2 output exceeds the bound), on same-sign data.
#[test]
fn a_hidden_partial_scale_in_the_engine_fails_the_clean_row() {
    for (scale, k) in [(1.003f32, 2048usize), (1.003, 5120), (1.001, 2048), (1.001, 5120)] {
        let c = Case::new(Kind::SameSign, 1, 512, k, 99);
        let mut e = Emu::new(Order::LanesTree(256), Order::LanesTree(256));
        e.hidden_scale = scale;
        let rep = run_case(&mut e, &c);
        let clean_failed = rep.failures.iter().any(|f| f.contains("clean tp2 vs tp1 exceeds BOUND_PROB"));
        eprintln!("hidden scale {scale} k={k}: clean row fails BOUND_PROB: {clean_failed}; {} failures", rep.failures.len());
        assert!(clean_failed, "hidden scale {scale} at k={k} slipped through: {:?}", rep.failures);
    }
}
