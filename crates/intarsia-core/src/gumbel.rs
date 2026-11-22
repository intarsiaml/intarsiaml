//! Device sampler (lane ink-hostgap, `INTARSIA_GUMBEL_SAMPLING=1`): Gumbel-max draws for plain-temperature rows, so a sampled
//! step's token stays on the device like a greedy one (token feedback, no `[V]` row download). This file is the reference that
//! the `GumbelArgmax` op (cpu backend) and its kernels (`gumbel_argmax_f32` on Metal) follow.
//!
//! For logits row `l` of the token at absolute position `pos` (the number of tokens before it), seed `s` and temperature `T`:
//!
//! ```text
//! token = argmax_i ( c(l_i) * (1 / T) + g_i )              ties: the smaller id
//! c(x)  = cap * tanh(x / cap)  (cap > 0, exactly `sampler::softcap`), x when cap = 0
//! x_i   = word 0 of Philox4x32-10(counter (i, pos, 0, 0), key (s lo, s hi))
//! w_i   = min((f32(x_i) + 0.5) * 2^-32, 1 - 2^-24)         u_i = 1 - w_i is uniform in (0, 1)
//! E_i   = -ln(1 - w_i)  (its series below w = 1/16)        Exp(1)
//! g_i   = -ln(E_i)                                         Gumbel(0, 1), always finite
//! ```
//!
//! The counter holds the token id and the position, never a batch row, so a row draws the same token solo or batched. A fixed
//! seed gives a fixed token stream. The draw differs from the host sampler's (`Sampler::sample`): a different but exact
//! method for the same distribution, softmax(c(l) / T). Built from w (u = 1 - w) so that the winning tokens, which have u near
//! 1, keep full relative precision: E is small there, and E = w (1 + w/2 + ...) is exact to an ulp or so.

/// Philox4x32-10 (Salmon et al., SC11; Random123's `philox4x32_R(10, ..)`).
pub fn philox4x32_10(mut c: [u32; 4], mut k: [u32; 2]) -> [u32; 4] {
    const M0: u64 = 0xD251_1F53;
    const M1: u64 = 0xCD9E_8D57;
    const W0: u32 = 0x9E37_79B9;
    const W1: u32 = 0xBB67_AE85;
    for r in 0..10 {
        if r > 0 {
            k[0] = k[0].wrapping_add(W0);
            k[1] = k[1].wrapping_add(W1);
        }
        let p0 = M0 * c[0] as u64;
        let p1 = M1 * c[2] as u64;
        let (hi0, lo0) = ((p0 >> 32) as u32, p0 as u32);
        let (hi1, lo1) = ((p1 >> 32) as u32, p1 as u32);
        c = [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0];
    }
    c
}

/// The largest w: 1 - 2^-24, the largest f32 below 1.
pub const W_MAX: f32 = 1.0 - 1.0 / (1u32 << 24) as f32;

/// w of a Philox word: `(f32(x) + 0.5) * 2^-32`, at most [`W_MAX`].
pub fn w_of(x: u32) -> f32 {
    ((x as f32 + 0.5) * (1.0 / 4_294_967_296.0)).min(W_MAX)
}

/// `-ln(1 - w)` for w in (0, 1): the series to w^5 below 1/16 (relative error under w^5 / 6, 1.6e-7), `ln` above.
pub fn neg_ln_1m(w: f32) -> f32 {
    if w < 0.0625 {
        w * (1.0 + w * (0.5 + w * (1.0 / 3.0 + w * (0.25 + w * 0.2))))
    } else {
        -(1.0 - w).ln()
    }
}

/// Gumbel(0, 1) noise of Philox word `x`: `-ln(E)`, E = `neg_ln_1m(w_of(x))`. Finite for every x.
pub fn g_of(x: u32) -> f32 {
    -neg_ln_1m(w_of(x)).ln()
}

/// The Gumbel noise of token id `i` at position `pos` under `seed`.
pub fn noise(seed: u64, pos: u32, i: u32) -> f32 {
    g_of(philox4x32_10([i, pos, 0, 0], [seed as u32, (seed >> 32) as u32])[0])
}

/// One row's key, as the `GumbelArgmax` op's `[R, 4]` i32 input holds it: (seed lo, seed hi, pos, bits of 1 / T).
pub fn key_row(seed: u64, pos: usize, temperature: f32) -> [i32; 4] {
    [seed as u32 as i32, (seed >> 32) as u32 as i32, pos as u32 as i32, (1.0 / temperature).to_bits() as i32]
}

/// The draw of one row (the op's definition): `argmax_i(c(l_i) * inv_t + g_i)`, ties to the smaller id, `cap` 0 = no softcap.
/// A row with no finite candidate (all -inf or NaN) gives 0, as `Argmax` does.
pub fn argmax_row(logits: &[f32], cap: f32, key: [i32; 4]) -> u32 {
    let seed = key[0] as u32 as u64 | ((key[1] as u32 as u64) << 32);
    let pos = key[2] as u32;
    let inv_t = f32::from_bits(key[3] as u32);
    let k = [seed as u32, (seed >> 32) as u32];
    let mut best = f32::NEG_INFINITY;
    let mut bi = u32::MAX;
    for (i, &l) in logits.iter().enumerate() {
        let c = if cap > 0.0 { cap * (l / cap).tanh() } else { l };
        let v = c * inv_t + g_of(philox4x32_10([i as u32, pos, 0, 0], k)[0]);
        if v > best {
            best = v;
            bi = i as u32;
        }
    }
    if bi == u32::MAX { 0 } else { bi }
}

/// The seed of a request's Gumbel draws: the request's own when it sets one, else a fresh one per request (time, mixed with
/// a process counter so two requests in the same nanosecond still differ).
pub fn request_seed(seed: Option<u64>) -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    seed.unwrap_or_else(|| {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
        crate::rng::SplitMix64::new(t ^ N.fetch_add(1, Ordering::Relaxed).rotate_left(32)).next_u64()
    })
}

/// The device sampler for plain-temperature rows: `INTARSIA_GUMBEL_SAMPLING` when set (`1` on, anything else off), else the
/// recipe's `[tuning.<backend>] gumbel_sampling` (`recipe`), else off. Only the Inkling-2 recipe turns it on, on Metal (user
/// decision 2026-11-22): it changes which token a given seed draws, so every other model keeps the host sampler.
pub fn wanted(recipe: Option<bool>) -> bool {
    resolve(std::env::var("INTARSIA_GUMBEL_SAMPLING").ok().as_deref(), recipe)
}

/// [`wanted`] with the variable passed in.
pub fn resolve(env: Option<&str>, recipe: Option<bool>) -> bool {
    match env {
        Some(v) => v.trim() == "1",
        None => recipe.unwrap_or(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    #[test]
    fn the_switch_wins_over_the_recipe_and_unset_follows_it() {
        assert!(!resolve(None, None), "no switch, no recipe key: off");
        assert!(!resolve(None, Some(false)));
        assert!(resolve(None, Some(true)), "the recipe turns it on (Inkling-2 on Metal)");
        for off in ["0", "", "off", "true", "2"] {
            assert!(!resolve(Some(off), Some(true)), "INTARSIA_GUMBEL_SAMPLING={off:?} reverts the recipe");
        }
        assert!(resolve(Some("1"), None), "1 turns it on for any model");
        assert!(resolve(Some(" 1 "), Some(false)));
    }

    #[test]
    fn philox_matches_the_random123_known_answers() {
        assert_eq!(philox4x32_10([0; 4], [0; 2]), [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]);
        assert_eq!(philox4x32_10([u32::MAX; 4], [u32::MAX; 2]), [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]);
        assert_eq!(
            philox4x32_10([0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344], [0xa409_3822, 0x299f_31d0]),
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
        );
    }

    #[test]
    fn edge_words_give_finite_noise() {
        // x = 0: the smallest w, the largest g (about 22.9)
        let w0 = w_of(0);
        assert_eq!(w0, 0.5 / 4_294_967_296.0);
        let g0 = g_of(0);
        assert!(g0.is_finite() && (g0 - 22.88).abs() < 0.01, "{g0}");
        // x = 2^32 - 1 rounds to 2^32 in f32: w clamps to 1 - 2^-24, E = 24 ln 2, g = -ln(16.64) (about -2.81)
        assert_eq!(w_of(u32::MAX), W_MAX);
        let g1 = g_of(u32::MAX);
        assert!(g1.is_finite() && (g1 + (24.0f32 * std::f32::consts::LN_2).ln()).abs() < 1e-5, "{g1}");
        // every word that rounds up to 2^32 clamps the same way
        assert_eq!(w_of(u32::MAX - 127), W_MAX);
    }

    #[test]
    fn noise_is_finite_for_random_words() {
        let mut r = SplitMix64::new(42);
        for _ in 0..1_000_000 {
            let x = r.next_u64() as u32;
            assert!(g_of(x).is_finite(), "x {x}");
        }
        // and every word near the two ends
        for x in (0..4096).chain(u32::MAX - 4096..=u32::MAX) {
            assert!(g_of(x).is_finite(), "x {x}");
        }
    }

    #[test]
    fn the_series_meets_ln_at_one_sixteenth() {
        // both branches agree to a few ulps where they meet, and the series is exact to an ulp or so for small w
        for w in [0.0625f32 - 1e-6, 0.03, 1e-3, 1e-6, 1e-9] {
            let s = neg_ln_1m(w);
            let exact = -(1.0 - w as f64).ln();
            assert!(((s as f64 - exact) / exact).abs() < 4e-7, "w {w}: {s} vs {exact}");
        }
    }

    /// Chi-square of `n` draws of one row against softmax(c(l) / T), the draws taking positions `0..n` (a fresh key each).
    fn chi_square(logits: &[f32], cap: f32, t: f32, n: usize, seed: u64) -> f64 {
        let c: Vec<f64> = logits.iter().map(|&l| if cap > 0.0 { (cap * (l / cap).tanh()) as f64 } else { l as f64 } / t as f64).collect();
        let m = c.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = c.iter().map(|x| (x - m).exp()).sum();
        let mut counts = vec![0usize; logits.len()];
        for pos in 0..n {
            counts[argmax_row(logits, cap, key_row(seed, pos, t)) as usize] += 1;
        }
        c.iter()
            .zip(&counts)
            .map(|(x, &o)| {
                let e = n as f64 * (x - m).exp() / z;
                (o as f64 - e).powi(2) / e
            })
            .sum()
    }

    #[test]
    fn draws_follow_the_softmax_at_t07_and_t1_with_the_softcap() {
        // 8 tokens, 7 degrees of freedom: chi-square above 24.3 has p = 0.001. With the cap (30) the logits reach past it, where
        // tanh bends; without it they are a tenth of that (every expected count stays above 100).
        let capped = [31.0f32, 29.5, 28.0, 30.2, 26.0, 33.0, 29.9, 27.5];
        let plain = capped.map(|l| l / 10.0);
        for t in [0.7f32, 1.0] {
            for (logits, cap, seed) in [(&capped, 30.0f32, 7u64), (&plain, 0.0, 11)] {
                let x2 = chi_square(logits, cap, t, 40_000, seed);
                assert!(x2 < 24.3, "T {t} cap {cap}: chi-square {x2:.2}");
            }
        }
    }

    #[test]
    fn a_fixed_seed_draws_fixed_tokens_and_seeds_differ() {
        let mut r = SplitMix64::new(3);
        let logits: Vec<f32> = (0..1000).map(|_| r.next_f32() * 4.0).collect();
        let a: Vec<u32> = (0..64).map(|p| argmax_row(&logits, 30.0, key_row(5, p, 1.0))).collect();
        let b: Vec<u32> = (0..64).map(|p| argmax_row(&logits, 30.0, key_row(5, p, 1.0))).collect();
        let c: Vec<u32> = (0..64).map(|p| argmax_row(&logits, 30.0, key_row(6, p, 1.0))).collect();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn ties_go_to_the_smaller_id_and_dead_rows_to_zero() {
        assert_eq!(argmax_row(&[f32::NEG_INFINITY; 4], 0.0, key_row(1, 0, 1.0)), 0);
        // a huge gap: only the masked-in token can win
        let mut l = vec![-1e30f32; 16];
        l[9] = 0.0;
        assert_eq!(argmax_row(&l, 0.0, key_row(1, 3, 1.0)), 9);
    }
}
