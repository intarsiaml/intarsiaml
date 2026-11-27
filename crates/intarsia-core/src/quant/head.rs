//! Opt-in quantised `lm_head` (lane lmhead-fp8, docs/results/2026-11-05-lmhead-fp8-27b.md).
//!
//! The dense Qwen3.8-27B reads its 248320 x 5120 bf16 head (2.54 GB) once per decode step. `INTARSIA_LMHEAD_QUANT=fp8|mxfp4` (or a
//! recipe's `[tuning.<backend>] lm_head_quant`) quantises it at load, on the host, from the checkpoint's bf16 bytes, into the formats
//! whose GEMV kernels the dense projections already use:
//!
//! * `fp8`: e4m3 weights with ONE f32 scale per tensor (`amax / 448`), the fp8 ABI's `scales [2] = (ws, 0)` (W8A16: the activations
//!   stay bf16). 1.27 GB. e4m3 is a floating format, so a per-tensor scale costs little against a per-row one; a per-row-scale
//!   variant would need a new scales layout in every backend's kernels and is not built (the results doc has the numbers).
//! * `mxfp4`: e2m1 weights with one e8m0 scale per 32 columns (finer than per-row), W4A16. 0.64 GB + 40 MB of scales.
//!
//! It changes outputs, so it is never a default. The quantised head replaces the bf16 one entirely (the bf16 head is not loaded when
//! the checkpoint has its own `lm_head` tensor), so the MTP drafter's reduced head (`row_prefix`) and the tensor-parallel vocabulary
//! shards are rows of the quantised tensor.

use crate::backend::Backend;
use crate::dtype::{f32_to_fp8_e4m3, fp8_e4m3_to_f32, DType};
use crate::error::{Error, Result};
use crate::quant::{fp8, mxfp4, HostQTensor, Mxfp4Act, QTensor, FP8_E4M3, MXFP4};
use crate::tensor::HostTensor;

/// The environment variable (`fp8`, `mxfp4`; `off`, `0`, `none` or empty: off, and then the recipe key is ignored).
pub const ENV: &str = "INTARSIA_LMHEAD_QUANT";

/// The MTP draft head's environment variable (`fp8`, `mxfp4`, `mix`; the off spellings of [`ENV`]; lane c27). Wins over the recipe's `mtp_quant`.
pub const MTP_ENV: &str = "INTARSIA_MTP_QUANT";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadQuant {
    Fp8,
    Mxfp4,
    /// Lane ink-q8 (accuracy study, cpu reference only): e4m3 with one f32 scale per output row (`amax / 448`).
    Fp8Row,
    /// Lane ink-q8 (accuracy study, cpu reference only): int8 with one f32 scale per output row, the scale from an MSE-optimal clip
    /// search per row (symmetric, round to nearest).
    Int8Row,
    /// Lane ink-q8 controls (cpu reference only): int8 per row with the plain `amax / 127` scale (no clip search), and the identity
    /// (the checkpoint's weights as exact f32: the emulation path with no quantisation).
    Int8AmaxRow,
    IdRow,
    /// Lane ink-q8 noise control (cpu reference only): every weight times `1 + eps * u`, `u` uniform in [-1, 1) from a hash of
    /// (seed, position), `eps = INTARSIA_Q8_NOISE_EPS` (default 2^-10), `seed = INTARSIA_Q8_NOISE_SEED` (default 1). Not a format: it
    /// measures how much KL a weight perturbation of a given size makes against our own bf16 (the numerics floor).
    NoiseRow,
}

impl HeadQuant {
    pub fn name(self) -> &'static str {
        match self {
            HeadQuant::Fp8 => "fp8",
            HeadQuant::Mxfp4 => "mxfp4",
            HeadQuant::Fp8Row => "fp8_row",
            HeadQuant::Int8Row => "int8_row",
            HeadQuant::Int8AmaxRow => "int8a_row",
            HeadQuant::IdRow => "id_row",
            HeadQuant::NoiseRow => "noise_row",
        }
    }

    /// The per-row 8-bit study formats: quantised on the host, then dequantised back to exact f32 weights (the cpu reference's
    /// result for `scale_row * sum(x * q)`, up to the accumulation order), so only the cpu backend takes them (no GPU kernel).
    pub fn is_row_emulated(self) -> bool {
        matches!(self, HeadQuant::Fp8Row | HeadQuant::Int8Row | HeadQuant::Int8AmaxRow | HeadQuant::IdRow | HeadQuant::NoiseRow)
    }

    /// `Ok(None)` for the off spellings.
    pub fn parse(s: &str) -> Result<Option<HeadQuant>> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fp8" | "fp8_e4m3" | "e4m3" => Ok(Some(HeadQuant::Fp8)),
            "mxfp4" => Ok(Some(HeadQuant::Mxfp4)),
            "fp8_row" | "fp8row" | "e4m3_row" => Ok(Some(HeadQuant::Fp8Row)),
            "int8_row" | "int8row" | "int8" => Ok(Some(HeadQuant::Int8Row)),
            "int8a_row" => Ok(Some(HeadQuant::Int8AmaxRow)),
            "id_row" => Ok(Some(HeadQuant::IdRow)),
            "noise_row" => Ok(Some(HeadQuant::NoiseRow)),
            "" | "off" | "0" | "none" | "bf16" => Ok(None),
            other => Err(Error::model(format!("{ENV} / lm_head_quant = '{other}' (fp8 | mxfp4 | fp8_row | int8_row | int8a_row | id_row | noise_row | off)"))),
        }
    }

    /// The env var wins over the recipe's value; unset on both: off.
    pub fn resolve(env: Option<&str>, recipe: Option<&str>) -> Result<Option<HeadQuant>> {
        match env {
            Some(v) => HeadQuant::parse(v),
            None => recipe.map_or(Ok(None), HeadQuant::parse),
        }
    }
}

/// The dense MTP head's projection formats: (fc + attention, feed-forward).
pub type MtpQuant = (HeadQuant, HeadQuant);

/// Parse an `INTARSIA_MTP_QUANT` / recipe `mtp_quant` value: `fp8` (all fp8), `mxfp4` (all mxfp4), `mix` (fc and attention fp8, feed-forward
/// mxfp4); `Ok(None)` for the off spellings.
pub fn parse_mtp_quant(s: &str) -> Result<Option<MtpQuant>> {
    match s.trim().to_ascii_lowercase().as_str() {
        "fp8" | "fp8_e4m3" | "e4m3" => Ok(Some((HeadQuant::Fp8, HeadQuant::Fp8))),
        "mxfp4" => Ok(Some((HeadQuant::Mxfp4, HeadQuant::Mxfp4))),
        "mix" => Ok(Some((HeadQuant::Fp8, HeadQuant::Mxfp4))),
        "" | "off" | "0" | "none" | "bf16" => Ok(None),
        other => Err(Error::model(format!("{MTP_ENV} / mtp_quant = '{other}' (fp8 | mxfp4 | mix | off)"))),
    }
}

/// The env var wins over the recipe's `mtp_quant` (any off spelling of the variable keeps bf16); unset on both: off.
pub fn resolve_mtp_quant(env: Option<&str>, recipe: Option<&str>) -> Result<Option<MtpQuant>> {
    match env {
        Some(v) => parse_mtp_quant(v),
        None => recipe.map_or(Ok(None), parse_mtp_quant),
    }
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get()).min(32)
}

/// Run `f(chunk_index, row_start, rows)` over `n` rows in chunks, `threads()` at a time.
fn par_rows<T: Send>(n: usize, chunk: usize, f: impl Fn(usize, usize) -> T + Sync) -> Vec<T> {
    let nchunks = n.div_ceil(chunk);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut out: Vec<(usize, T)> = Vec::with_capacity(nchunks);
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads().min(nchunks))
            .map(|_| {
                sc.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let c = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if c >= nchunks {
                            break;
                        }
                        let (s, r) = (c * chunk, chunk.min(n - c * chunk));
                        mine.push((c, f(s, r)));
                    }
                    mine
                })
            })
            .collect();
        for h in hs {
            out.extend(h.join().expect("lm_head quantiser thread"));
        }
    });
    out.sort_by_key(|(c, _)| *c);
    out.into_iter().map(|(_, t)| t).collect()
}

const CHUNK_ROWS: usize = 512;

/// The per-row study formats exist only as host emulation (no GPU kernel): refuse them on any backend but the cpu reference.
pub fn emulation_guard(kind: HeadQuant, backend: &dyn Backend) -> Result<()> {
    emulation_guard_for(kind, backend.name())
}

/// [`emulation_guard`] on the backend's name (the part the tests can reach without a backend).
pub fn emulation_guard_for(kind: HeadQuant, backend: &str) -> Result<()> {
    if kind.is_row_emulated() && backend != "cpu" {
        return Err(Error::model(format!("{} is a cpu-reference accuracy study format (no {backend} kernel)", kind.name())));
    }
    Ok(())
}

/// e4m3 with one scale per row: `scale = amax / 448` (f32), `out = e4m3(x / scale) * scale` (the dequantised row, in f32).
pub fn fp8_row_dequant(x: &[f32], out: &mut [f32]) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    if !(amax > 0.0) {
        out.iter_mut().for_each(|o| *o = 0.0);
        return;
    }
    let ws = amax / fp8::MAX;
    for (o, v) in out.iter_mut().zip(x) {
        *o = fp8_e4m3_to_f32(f32_to_fp8_e4m3(v / ws)) * ws;
    }
}

fn int8_err(x: &[f32], clip: f32) -> f64 {
    let s = clip / 127.0;
    let inv = 1.0 / s;
    let mut e = 0f64;
    for v in x {
        let q = (v * inv).round_ties_even().clamp(-127.0, 127.0);
        let d = (v - q * s) as f64;
        e += d * d;
    }
    e
}

/// The MSE-optimal symmetric clip of a row for int8 (round to nearest, levels -127..=127): a coarse grid of clip / amax from 1.0 down to
/// 0.325, then a fine grid around the best coarse point.
pub fn int8_best_clip(x: &[f32], amax: f32) -> f32 {
    let mut best = (f64::INFINITY, 1.0f32);
    for i in 0..28 {
        let r = 1.0 - 0.025 * i as f32;
        let e = int8_err(x, amax * r);
        if e < best.0 {
            best = (e, r);
        }
    }
    let r0 = best.1;
    for j in -8i32..=8 {
        let r = (r0 + 0.003125 * j as f32).min(1.0);
        if r <= 0.0 {
            continue;
        }
        let e = int8_err(x, amax * r);
        if e < best.0 {
            best = (e, r);
        }
    }
    amax * best.1
}

/// int8 with one scale per row (`scale = clip / 127`, f32; clip from [`int8_best_clip`]): the dequantised row, in f32.
pub fn int8_row_dequant(x: &[f32], out: &mut [f32], clip_search: bool) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    if !(amax > 0.0) {
        out.iter_mut().for_each(|o| *o = 0.0);
        return;
    }
    let s = if clip_search { int8_best_clip(x, amax) } else { amax } / 127.0;
    let inv = 1.0 / s;
    for (o, v) in out.iter_mut().zip(x) {
        *o = (v * inv).round_ties_even().clamp(-127.0, 127.0) * s;
    }
}

fn rows_f32(dtype: DType, bytes: &[u8], k: usize, start: usize, rows: usize) -> Vec<f32> {
    let es = dtype.size_bytes();
    let mut v = vec![0f32; rows * k];
    crate::dtype::decode_to_f32(dtype, &bytes[start * k * es..(start + rows) * k * es], &mut v);
    v
}

/// Quantise rows `[row0, row0 + n)` of an `[n_all, k]` float tensor (`all`: its raw row-major bytes, `dtype` bf16/f16/f32) on the host.
/// The fp8 scale is the amax of the WHOLE tensor, whatever rows this rank keeps: a vocabulary shard (tensor parallelism) then holds
/// exactly the rows of the replicated quantised head, so the split logits are bitwise the replicated ones.
pub fn quantize_host(kind: HeadQuant, dtype: DType, all: &[u8], n_all: usize, k: usize, row0: usize, n: usize) -> Result<HostQTensor> {
    if !matches!(dtype, DType::Bf16 | DType::F16 | DType::F32) || all.len() != n_all * k * dtype.size_bytes() || row0 + n > n_all {
        return Err(Error::model(format!(
            "lm_head quantiser: rows {row0}..{} of {n_all} x {k} {dtype} need {} bytes, got {}",
            row0 + n,
            n_all * k * dtype.size_bytes(),
            all.len()
        )));
    }
    match kind {
        HeadQuant::Fp8 => {
            let amax = par_rows(n_all, CHUNK_ROWS, |s, r| rows_f32(dtype, all, k, s, r).iter().fold(0f32, |m, x| m.max(x.abs())))
                .into_iter()
                .fold(0f32, f32::max);
            if !amax.is_finite() {
                return Err(Error::model("lm_head quantiser: the head holds a non-finite value"));
            }
            let ws = if amax > 0.0 { amax / fp8::MAX } else { 1.0 };
            let parts = par_rows(n, CHUNK_ROWS, |s, r| {
                rows_f32(dtype, all, k, row0 + s, r).iter().map(|x| f32_to_fp8_e4m3(x / ws)).collect::<Vec<u8>>()
            });
            let mut data = Vec::with_capacity(n * k);
            for p in parts {
                data.extend_from_slice(&p);
            }
            Ok(HostQTensor {
                format: FP8_E4M3,
                n,
                k,
                parts: vec![HostTensor { dtype: DType::Fp8E4m3, shape: vec![n, k], data }, fp8::scales_tensor(ws, 0.0)],
                act_quant: false,
            })
        }
        HeadQuant::Fp8Row | HeadQuant::Int8Row | HeadQuant::Int8AmaxRow | HeadQuant::IdRow | HeadQuant::NoiseRow => {
            let env_f = |n: &str, d: f64| std::env::var(n).ok().and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(d);
            let (eps, seed) = (env_f("INTARSIA_Q8_NOISE_EPS", 1.0 / 1024.0) as f32, env_f("INTARSIA_Q8_NOISE_SEED", 1.0) as u64);
            let parts = par_rows(n, CHUNK_ROWS, |s, r| {
                let v = rows_f32(dtype, all, k, row0 + s, r);
                let mut out = Vec::with_capacity(r * k * 4);
                let mut row = vec![0f32; k];
                for (ri, x) in v.chunks(k).enumerate() {
                    match kind {
                        HeadQuant::Fp8Row => fp8_row_dequant(x, &mut row),
                        HeadQuant::Int8Row => int8_row_dequant(x, &mut row, true),
                        HeadQuant::Int8AmaxRow => int8_row_dequant(x, &mut row, false),
                        HeadQuant::NoiseRow => {
                            for (ci, (o, v)) in row.iter_mut().zip(x).enumerate() {
                                let mut z = seed ^ (((row0 + s + ri) as u64) << 24 | ci as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                                z ^= z >> 31;
                                let u = (z >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
                                *o = v * (1.0 + eps * u);
                            }
                        }
                        _ => row.copy_from_slice(x),
                    }
                    for y in &row {
                        out.extend_from_slice(&y.to_le_bytes());
                    }
                }
                out
            });
            let mut data = Vec::with_capacity(n * k * 4);
            for p in parts {
                data.extend_from_slice(&p);
            }
            if data.chunks_exact(4).any(|c| !f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_finite()) {
                return Err(Error::model(format!("lm_head quantiser: {} produced a non-finite value (a non-finite weight?)", kind.name())));
            }
            // the dequantised weights as a dense f32 tensor: the cpu `Matmul` / `Logits` / `InkInProj` float path reads them as is
            Ok(HostQTensor { format: crate::quant::BF16, n, k, parts: vec![HostTensor { dtype: DType::F32, shape: vec![n, k], data }], act_quant: false })
        }
        HeadQuant::Mxfp4 => {
            if k % 32 != 0 {
                return Err(Error::model(format!("lm_head quantiser: mxfp4 needs K % 32 == 0, K = {k}")));
            }
            let parts = par_rows(n, CHUNK_ROWS, |s, r| mxfp4::quantize_rows(&rows_f32(dtype, all, k, row0 + s, r), r, k));
            let (mut packed, mut scales) = (Vec::with_capacity(n * k / 2), Vec::with_capacity(n * k / 32));
            for (p, s) in parts {
                packed.extend_from_slice(&p);
                scales.extend_from_slice(&s);
            }
            Ok(HostQTensor {
                format: MXFP4,
                n,
                k,
                parts: vec![
                    HostTensor { dtype: DType::U8, shape: vec![n, k / 2], data: packed },
                    HostTensor { dtype: DType::U8, shape: vec![n, k / 32], data: scales },
                    mxfp4::act_tensor(Mxfp4Act::None),
                ],
                act_quant: false,
            })
        }
    }
}

/// Quantise and upload rows `[row0, row0 + n)` of the `[n_all, k]` float tensor whose raw bytes are `all` (see [`quantize_host`]).
pub fn load_rows(kind: HeadQuant, backend: &dyn Backend, dtype: DType, all: &[u8], n_all: usize, row0: usize, n: usize, k: usize) -> Result<QTensor> {
    emulation_guard(kind, backend)?;
    quantize_host(kind, dtype, all, n_all, k, row0, n)?.upload(backend)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_rows(n: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<u8>) {
        let mut rng = crate::rng::SplitMix64::new(seed);
        let v: Vec<f32> = (0..n * k)
            .map(|i| {
                let u = (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                // per-row magnitudes spread over two binades, with a few outliers
                u * 0.05 * (1.0 + (i / k % 4) as f32) * if i % 977 == 0 { 8.0 } else { 1.0 }
            })
            .collect();
        let bytes: Vec<u8> = v.iter().flat_map(|x| half::bf16::from_f32(*x).to_le_bytes()).collect();
        (v.iter().map(|x| half::bf16::from_f32(*x).to_f32()).collect(), bytes)
    }

    #[test]
    fn parse_and_precedence() {
        assert_eq!(HeadQuant::parse("fp8").unwrap(), Some(HeadQuant::Fp8));
        assert_eq!(HeadQuant::parse("MXFP4").unwrap(), Some(HeadQuant::Mxfp4));
        for off in ["", "off", "0", "none"] {
            assert_eq!(HeadQuant::parse(off).unwrap(), None);
        }
        assert!(HeadQuant::parse("fp4").is_err());
        assert_eq!(HeadQuant::resolve(None, None).unwrap(), None);
        assert_eq!(HeadQuant::resolve(None, Some("fp8")).unwrap(), Some(HeadQuant::Fp8));
        assert_eq!(HeadQuant::resolve(Some("mxfp4"), Some("fp8")).unwrap(), Some(HeadQuant::Mxfp4));
        assert_eq!(HeadQuant::resolve(Some("off"), Some("fp8")).unwrap(), None, "the env var wins, including off");
    }

    #[test]
    fn fp8_matches_the_format_quantiser_and_keeps_its_error_small() {
        let (n, k) = (130, 96);
        let (vals, bytes) = bf16_rows(n, k, 7);
        let q = quantize_host(HeadQuant::Fp8, DType::Bf16, &bytes, n, k, 0, n).unwrap();
        // the same bits as the format's own whole-tensor quantiser
        let r = <crate::quant::fp8::Fp8E4m3 as crate::quant::QuantFormat>::quantize(&crate::quant::fp8::Fp8E4m3, &HostTensor::from_f32(&[n, k], &vals), &Default::default()).unwrap();
        assert_eq!(q.parts[0].data, r.parts[0].data);
        assert_eq!(q.parts[1].data, r.parts[1].data);
        assert!(!q.act_quant && q.parts[1].to_f32()[1] == 0.0, "W8A16: the input scale is 0");
        let d = q.dequant();
        let (mut num, mut den) = (0f64, 0f64);
        for (a, b) in vals.iter().zip(&d) {
            num += ((a - b) as f64).powi(2);
            den += (*a as f64).powi(2);
        }
        let rel = (num / den).sqrt();
        assert!(rel < 0.04, "relative rms error {rel}");
    }

    #[test]
    fn mxfp4_matches_quantize_rows_and_chunks_do_not_matter() {
        let (n, k) = (1100, 64); // more than two chunks of CHUNK_ROWS
        let (vals, bytes) = bf16_rows(n, k, 9);
        let q = quantize_host(HeadQuant::Mxfp4, DType::Bf16, &bytes, n, k, 0, n).unwrap();
        let (p, s) = mxfp4::quantize_rows(&vals, n, k);
        assert_eq!(q.parts[0].data, p);
        assert_eq!(q.parts[1].data, s);
        assert_eq!(q.parts[2].to_f32(), vec![0.0], "W4A16: the act flag is 0 whatever the recipe's mxfp4_act is");
        assert!(!q.act_quant);
        assert_eq!(q.parts[0].shape, vec![n, k / 2]);
    }

    #[test]
    fn a_row_shard_is_the_rows_of_the_whole_quantised_tensor() {
        let (n, k) = (96, 64);
        let (_, bytes) = bf16_rows(n, k, 5);
        for kind in [HeadQuant::Fp8, HeadQuant::Mxfp4] {
            let whole = quantize_host(kind, DType::Bf16, &bytes, n, k, 0, n).unwrap();
            let half = quantize_host(kind, DType::Bf16, &bytes, n, k, n / 2, n / 2).unwrap();
            assert_eq!(half.parts[0].data[..], whole.parts[0].data[whole.parts[0].data.len() / 2..], "{kind:?} weights");
            if kind == HeadQuant::Fp8 {
                assert_eq!(half.parts[1].data, whole.parts[1].data, "one scale for the whole tensor");
            } else {
                assert_eq!(half.parts[1].data[..], whole.parts[1].data[whole.parts[1].data.len() / 2..]);
            }
        }
    }

    #[test]
    fn row_formats_parse_and_are_cpu_only() {
        assert_eq!(HeadQuant::parse("fp8_row").unwrap(), Some(HeadQuant::Fp8Row));
        assert_eq!(HeadQuant::parse("INT8_ROW").unwrap(), Some(HeadQuant::Int8Row));
        assert!(HeadQuant::Fp8Row.is_row_emulated() && HeadQuant::Int8Row.is_row_emulated());
        assert!(!HeadQuant::Fp8.is_row_emulated() && !HeadQuant::Mxfp4.is_row_emulated());
    }

    #[test]
    fn row_study_formats_are_refused_off_the_cpu() {
        for kind in [HeadQuant::Fp8Row, HeadQuant::Int8Row, HeadQuant::Int8AmaxRow, HeadQuant::IdRow, HeadQuant::NoiseRow] {
            assert!(emulation_guard_for(kind, "cpu").is_ok());
            for b in ["hip", "metal", "cuda"] {
                let e = emulation_guard_for(kind, b).unwrap_err().to_string();
                assert!(e.contains("cpu-reference") && e.contains(b), "{e}");
            }
        }
        // the real formats are not touched
        for kind in [HeadQuant::Fp8, HeadQuant::Mxfp4] {
            assert!(emulation_guard_for(kind, "hip").is_ok());
        }
    }

    #[test]
    fn per_row_formats_beat_the_per_tensor_fp8_and_stay_on_the_grid() {
        // rows spread over two binades (bf16_rows): a per-row scale tracks each row's own range
        let (n, k) = (130, 192);
        let (vals, bytes) = bf16_rows(n, k, 11);
        let rel = |d: &[f32]| {
            let (mut num, mut den) = (0f64, 0f64);
            for (a, b) in vals.iter().zip(d) {
                num += ((a - b) as f64).powi(2);
                den += (*a as f64).powi(2);
            }
            (num / den).sqrt()
        };
        let per_tensor = rel(&quantize_host(HeadQuant::Fp8, DType::Bf16, &bytes, n, k, 0, n).unwrap().dequant());
        let row8 = quantize_host(HeadQuant::Fp8Row, DType::Bf16, &bytes, n, k, 0, n).unwrap();
        assert_eq!(row8.parts[0].dtype, DType::F32);
        let e8 = rel(&row8.dequant());
        let i8q = quantize_host(HeadQuant::Int8Row, DType::Bf16, &bytes, n, k, 0, n).unwrap();
        let ei = rel(&i8q.dequant());
        assert!(e8 <= per_tensor, "fp8_row {e8} vs per-tensor {per_tensor}");
        assert!(e8 < 0.04 && ei < 0.02, "fp8_row {e8}, int8_row {ei}");
        // every int8 row has at most 255 distinct values: q * scale with |q| <= 127
        for row in i8q.dequant().chunks(k) {
            let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
            let s = row.iter().filter(|v| **v != 0.0).map(|v| v.abs()).fold(f32::INFINITY, f32::min);
            assert!(amax / s <= 127.5, "|q| <= 127: amax {amax} min nonzero {s}");
        }
        // a zero row stays zero
        let z = vec![0u8; 2 * k * 2];
        assert!(quantize_host(HeadQuant::Int8Row, DType::Bf16, &z, 2, k, 0, 2).unwrap().dequant().iter().all(|v| *v == 0.0));
        // a row shard is the rows of the whole
        let whole = quantize_host(HeadQuant::Int8Row, DType::Bf16, &bytes, n, k, 0, n).unwrap();
        let half = quantize_host(HeadQuant::Int8Row, DType::Bf16, &bytes, n, k, n / 2, n / 2).unwrap();
        assert_eq!(half.parts[0].data[..], whole.parts[0].data[whole.parts[0].data.len() / 2..]);
    }

    #[test]
    fn int8_clip_search_never_loses_to_the_amax_scale() {
        let mut rng = crate::rng::SplitMix64::new(3);
        // heavy-tailed row: a clip below amax wins
        let x: Vec<f32> = (0..2048).map(|i| ((rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * if i % 256 == 0 { 20.0 } else { 1.0 }).collect();
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let c = int8_best_clip(&x, amax);
        assert!(c < amax && int8_err(&x, c) < int8_err(&x, amax));
    }

    #[test]
    fn rejects_a_short_tensor_and_a_bad_k() {
        let (_, bytes) = bf16_rows(8, 32, 1);
        assert!(quantize_host(HeadQuant::Fp8, DType::Bf16, &bytes[..bytes.len() - 2], 8, 32, 0, 8).is_err());
        assert!(quantize_host(HeadQuant::Mxfp4, DType::Bf16, &bytes[..8 * 16 * 2], 8, 16, 0, 8).is_err());
    }
}
