//! P12: lossless 12-bit storage of a bf16 weight (lane ink-pack12, `docs/results/2026-11-21-inkling2-pack12.md`).
//!
//! Not a quantisation: `decode(pack(W)) == W` bit for bit, for every bf16 bit pattern. A weight `[N, K]` (K % 8 == 0) is two
//! tensors, `codes` u8 `[N, 3K/2]` and `aux` u32 `[N + 1 + E]`:
//!
//! * Row `n` has a base exponent `b_n` (0..=240, the 16-exponent window `[b_n, b_n + 15]` holding the most elements of the
//!   row). An element `s | e (8 bits) | m (7 bits)` with `e` in the window is stored in place as the 12 bits `s`, `o = e - b_n`
//!   (4 bits), `m`; every other element (an escape) stores 0 there and its exact bf16 bits go to the side list.
//! * In place: per element the 16-bit half `h = s << 15 | o << 7 | m` (bits 11..14 zero); per pair `j` (elements `2j`, `2j+1`
//!   of an 8-element chunk) the word `Q_j = h(2j) | h(2j + 1) << 16`. Chunk `c` (elements `8c .. 8c + 8`) is three
//!   little-endian u32 words: `W0, W1` at row bytes `[8c, 8c + 8)` (plane A, the first K bytes of the row) and `W2` at row bytes
//!   `[K + 4c, K + 4c + 4)` (plane B, the last K/2 bytes). `W_j = Q_j | holes_j` for `j < 3`, the holes (bits 11..14, 27..30)
//!   carrying `Q3`: `holes_0 = (Q3 & 0x000F000F) << 11`, `holes_1 = (Q3 & 0x00F000F0) << 7`,
//!   `holes_2 = (Q3 & 0x07000700) << 3 | (Q3 & 0x80008000) >> 1`.
//! * Decode: `Q_j = W_j & 0x87FF87FF` (j < 3), `Q3 = (W0 >> 11 & 0x000F000F) | (W1 >> 7 & 0x00F000F0) | (W2 >> 3 & 0x07000700)
//!   | (W2 << 1 & 0x80008000)`; the bf16 pair word (low element in the low half, as a bf16 tensor stores it) is
//!   `Q_j + b_n * 0x00800080` (no carry leaves a half: `o + b_n <= 255`). Then every escape of the row overwrites its element.
//! * `aux[n] = start_n << 8 | b_n` for `n < N`, `aux[N] = E << 8`; the escapes of row `n` are `aux[N + 1 + start_n ..
//!   N + 1 + start_{n+1}]`, each `col << 16 | bf16 bits`, columns ascending. `E < 2^24`, `K <= 65536`.
//!
//! A kernel that decodes each element to its exact bf16 bits in registers and then runs a bf16 kernel's arithmetic in that
//! kernel's order gives that kernel's output bit for bit; the P12 Metal kernels do exactly that (bitwise their bf16 twins).

use crate::backend::{Backend, TensorRef};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// The `QuantFormat` id of a P12 weight (`QTensor::parts = [codes, aux]`, the op `MatmulP12`).
pub const P12: super::QuantId = "p12";

/// `INTARSIA_INK_PACK12`: Inkling-2 stores in_proj, wo and the unembed as P12 too, for the few-row kernels (lossless).
/// Default on on Metal, CUDA and HIP, `0` turns it off, `1` turns it on on any backend with the P12 kernels ([`decide`]).
pub const ENV: &str = "INTARSIA_INK_PACK12";

/// Most activation rows a P12 kernel takes (decode and verify; prefill chunks keep the bf16 weights).
pub const ROWS_MAX: usize = 16;

/// Most rows for which Inkling-2 takes the P12 copy of in_proj and wo; larger calls take the resident bf16 weight (the same
/// output bits, the kernels are bitwise twins). On endor (M5 Max, #10919) P12 is faster at 1 and 4 rows (in_proj 1.24x /
/// 1.07x, wo 1.22x / 1.15x) and slower at 16 (0.64x / 0.70x, both ALU-bound). 4 rows also covers MTP k3 verify. The
/// unembed takes P12 at every row count up to [`ROWS_MAX`] (1.31x at 1 row, still faster at 16).
pub const DENSE_ROWS_MAX: usize = 4;

/// Bytes of one packed row of K elements.
pub const fn row_bytes(k: usize) -> usize {
    3 * k / 2
}

/// A packed weight on the host.
#[derive(Clone, Debug, PartialEq)]
pub struct Packed {
    pub n: usize,
    pub k: usize,
    /// `[n, 3k/2]` bytes.
    pub codes: Vec<u8>,
    /// `[n + 1 + escapes]`.
    pub aux: Vec<u32>,
}

impl Packed {
    pub fn escapes(&self) -> usize {
        self.aux.len() - self.n - 1
    }
    /// Escapes of row `r`.
    pub fn row_escapes(&self, r: usize) -> usize {
        ((self.aux[r + 1] >> 8) - (self.aux[r] >> 8)) as usize
    }
    pub fn bytes(&self) -> usize {
        self.codes.len() + 4 * self.aux.len()
    }
    /// `codes` and `aux` as host tensors (`HostQTensor::parts`).
    pub fn host_parts(&self) -> Vec<crate::tensor::HostTensor> {
        let aux: Vec<u8> = self.aux.iter().flat_map(|v| v.to_le_bytes()).collect();
        vec![
            crate::tensor::HostTensor::new(DType::U8, vec![self.n, row_bytes(self.k)], self.codes.clone()).expect("sized"),
            crate::tensor::HostTensor::new(DType::I32, vec![self.aux.len()], aux).expect("sized"),
        ]
    }
    /// Upload as a [`super::QTensor`] of format [`P12`].
    pub fn upload(&self, b: &dyn Backend) -> Result<super::QTensor> {
        let codes = TensorRef::new(crate::backend::upload_new(b, &self.codes)?, DType::U8, &[self.n, row_bytes(self.k)]);
        let aux_b: Vec<u8> = self.aux.iter().flat_map(|v| v.to_le_bytes()).collect();
        let aux = TensorRef::new(crate::backend::upload_new(b, &aux_b)?, DType::I32, &[self.aux.len()]);
        Ok(super::QTensor { format: P12, n: self.n, k: self.k, parts: vec![codes, aux], act_quant: false, act_fp8: false })
    }
}

/// The window base of one row: the `b` in 0..=240 whose `[b, b + 15]` holds the most exponent fields (ties: the smallest).
fn row_base(row: &[u16]) -> u32 {
    let mut hist = [0u32; 256];
    for &u in row {
        hist[((u >> 7) & 0xFF) as usize] += 1;
    }
    let mut win: u32 = hist[..16].iter().sum();
    let (mut best, mut best_b) = (win, 0u32);
    for b in 1..=240usize {
        win = win - hist[b - 1] + hist[b + 15];
        if win > best {
            best = win;
            best_b = b as u32;
        }
    }
    best_b
}

/// Pack one row into `codes` (`3k/2` bytes); returns (base, escapes as `col << 16 | bits`).
fn pack_row(row: &[u16], codes: &mut [u8]) -> (u32, Vec<u32>) {
    let k = row.len();
    let b = row_base(row);
    let mut esc = Vec::new();
    let mut half = |c: usize| -> u32 {
        let u = row[c] as u32;
        let e = (u >> 7) & 0xFF;
        if e >= b && e <= b + 15 {
            (u & 0x807F) | ((e - b) << 7)
        } else {
            esc.push((c as u32) << 16 | u);
            0
        }
    };
    for c in 0..k / 8 {
        let mut q = [0u32; 4];
        for (j, qj) in q.iter_mut().enumerate() {
            *qj = half(8 * c + 2 * j) | half(8 * c + 2 * j + 1) << 16;
        }
        let w0 = q[0] | (q[3] & 0x000F_000F) << 11;
        let w1 = q[1] | (q[3] & 0x00F0_00F0) << 7;
        let w2 = q[2] | (q[3] & 0x0700_0700) << 3 | (q[3] & 0x8000_8000) >> 1;
        codes[8 * c..8 * c + 4].copy_from_slice(&w0.to_le_bytes());
        codes[8 * c + 4..8 * c + 8].copy_from_slice(&w1.to_le_bytes());
        codes[k + 4 * c..k + 4 * c + 4].copy_from_slice(&w2.to_le_bytes());
    }
    (b, esc)
}

fn check_dims(n: usize, k: usize) -> Result<()> {
    if k == 0 || k % 8 != 0 || k > 65536 {
        return Err(Error::invalid(format!("p12: K = {k} must be a positive multiple of 8, at most 65536")));
    }
    let _ = n;
    Ok(())
}

/// Pack a row-major bf16 weight `[n, k]` (as u16 bit patterns). Rows in parallel on up to `threads` threads (0 = all cores).
pub fn pack(w: &[u16], n: usize, k: usize, threads: usize) -> Result<Packed> {
    check_dims(n, k)?;
    if w.len() != n * k {
        return Err(Error::invalid(format!("p12: {} elements for [{n}, {k}]", w.len())));
    }
    let rb = row_bytes(k);
    let mut codes = vec![0u8; n * rb];
    let threads = if threads == 0 { std::thread::available_parallelism().map_or(4, |v| v.get()) } else { threads };
    let per = n.div_ceil(threads.max(1)).max(1);
    let mut rows: Vec<(u32, Vec<u32>)> = Vec::with_capacity(n);
    std::thread::scope(|sc| {
        let hs: Vec<_> = codes
            .chunks_mut(per * rb)
            .enumerate()
            .map(|(t, cs)| {
                let r0 = t * per;
                sc.spawn(move || {
                    let nr = cs.len() / rb;
                    (0..nr).map(|r| pack_row(&w[(r0 + r) * k..(r0 + r + 1) * k], &mut cs[r * rb..(r + 1) * rb])).collect::<Vec<_>>()
                })
            })
            .collect();
        for h in hs {
            rows.extend(h.join().expect("p12 pack thread"));
        }
    });
    let e: usize = rows.iter().map(|r| r.1.len()).sum();
    if e >= 1 << 24 {
        return Err(Error::invalid(format!("p12: {e} escapes (at most 2^24 - 1)")));
    }
    let mut aux = Vec::with_capacity(n + 1 + e);
    let mut start = 0u32;
    for (b, es) in &rows {
        aux.push(start << 8 | b);
        start += es.len() as u32;
    }
    aux.push(start << 8);
    for (_, es) in rows {
        aux.extend(es);
    }
    Ok(Packed { n, k, codes, aux })
}

/// [`pack`] of little-endian bf16 bytes.
pub fn pack_bytes(w: &[u8], n: usize, k: usize, threads: usize) -> Result<Packed> {
    let v: Vec<u16> = w.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    pack(&v, n, k, threads)
}

/// Validate `aux` for `[n, k]` and return `E`.
fn check_aux(aux: &[u32], n: usize, k: usize) -> Result<usize> {
    if aux.len() < n + 1 {
        return Err(Error::invalid(format!("p12: aux has {} words, needs at least {}", aux.len(), n + 1)));
    }
    let e = (aux[n] >> 8) as usize;
    if aux.len() != n + 1 + e || aux[n] & 0xFF != 0 {
        return Err(Error::invalid(format!("p12: aux has {} words, sentinel says {} escapes for {n} rows", aux.len(), e)));
    }
    let mut prev = 0usize;
    for r in 0..n {
        let (s, b) = ((aux[r] >> 8) as usize, aux[r] & 0xFF);
        let s1 = (aux[r + 1] >> 8) as usize;
        if s != prev || s1 < s || b > 240 {
            return Err(Error::invalid(format!("p12: aux row {r}: start {s}, next {s1}, base {b}")));
        }
        let mut last = None;
        for &ent in &aux[n + 1 + s..n + 1 + s1] {
            let col = (ent >> 16) as usize;
            if col >= k || last.is_some_and(|l| l >= col) {
                return Err(Error::invalid(format!("p12: row {r}: escape column {col} out of order or >= {k}")));
            }
            last = Some(col);
        }
        prev = s1;
    }
    Ok(e)
}

/// Decode to bf16 bit patterns `[n, k]` (validates `aux`).
pub fn decode(codes: &[u8], aux: &[u32], n: usize, k: usize) -> Result<Vec<u16>> {
    check_dims(n, k)?;
    let rb = row_bytes(k);
    if codes.len() != n * rb {
        return Err(Error::invalid(format!("p12: {} code bytes for [{n}, {k}] (needs {})", codes.len(), n * rb)));
    }
    check_aux(aux, n, k)?;
    let mut out = vec![0u16; n * k];
    let word = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    for r in 0..n {
        let row = &codes[r * rb..(r + 1) * rb];
        let bb = (aux[r] & 0xFF) * 0x0080_0080;
        let dst = &mut out[r * k..(r + 1) * k];
        for c in 0..k / 8 {
            let (w0, w1, w2) = (word(row, 8 * c), word(row, 8 * c + 4), word(row, k + 4 * c));
            let q3 = (w0 >> 11 & 0x000F_000F) | (w1 >> 7 & 0x00F0_00F0) | (w2 >> 3 & 0x0700_0700) | (w2 << 1 & 0x8000_8000);
            for (j, q) in [w0 & 0x87FF_87FF, w1 & 0x87FF_87FF, w2 & 0x87FF_87FF, q3].into_iter().enumerate() {
                let p = q.wrapping_add(bb);
                dst[8 * c + 2 * j] = p as u16;
                dst[8 * c + 2 * j + 1] = (p >> 16) as u16;
            }
        }
        let (s, s1) = ((aux[r] >> 8) as usize, (aux[r + 1] >> 8) as usize);
        for &ent in &aux[n + 1 + s..n + 1 + s1] {
            dst[(ent >> 16) as usize] = ent as u16;
        }
    }
    Ok(out)
}

/// [`decode`] of the two parts as host tensors (`codes` u8, `aux` u32 little-endian).
pub fn decode_parts(codes: &[u8], aux_bytes: &[u8], n: usize, k: usize) -> Result<Vec<u16>> {
    let aux: Vec<u32> = aux_bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    decode(codes, &aux, n, k)
}

/// Shape / dtype check of a P12 weight pair for an `[n, k]` weight (the op checks; `aux`'s content is the kernel's trust).
pub fn check_tensors(codes: &TensorRef, aux: &TensorRef, n: usize, k: usize, what: &str) -> std::result::Result<(), String> {
    if k == 0 || k % 8 != 0 || k > 65536 {
        return Err(format!("{what}: K = {k} must be a positive multiple of 8, at most 65536"));
    }
    if codes.dtype != DType::U8 || codes.shape != [n, row_bytes(k)] {
        return Err(format!("{what} codes must be u8 [{n},{}], got {} {:?}", row_bytes(k), codes.dtype, codes.shape));
    }
    if aux.dtype != DType::I32 || aux.shape.len() != 1 || aux.shape[0] < n + 1 {
        return Err(format!("{what} aux must be u32 [>= {}], got {} {:?}", n + 1, aux.dtype, aux.shape));
    }
    Ok(())
}

/// A weight-like random bf16 `[n, k]` (normal values of scale `std`, so each row has its natural escapes: tiny values below
/// the 16-exponent window) for admission cases. `adversarial` adds what the format must survive: +-0, subnormals, a row of
/// only escapes (alternating far exponents), escapes in the first and the last chunk (the K tail of every kernel), a large
/// value far above the window and the smallest normal.
pub fn random_weight(rng: &mut crate::rng::SplitMix64, n: usize, k: usize, std: f32, adversarial: bool) -> Vec<u16> {
    let v = rng.normal_vec(n * k, std);
    let mut w: Vec<u16> = v.iter().map(|x| half::bf16::from_f32(*x).to_bits()).collect();
    if adversarial && n > 0 {
        for r in 0..n {
            let row = &mut w[r * k..(r + 1) * k];
            match r % 5 {
                0 => {
                    row[0] = 0x0000;
                    row[k - 1] = 0x8000;
                }
                1 => {
                    row[1] = 0x0003; // subnormal
                    row[k - 2] = 0x8041;
                }
                2 => {
                    // only escapes: alternating exponents 30 binades apart, so no window holds more than half
                    for (c, x) in row.iter_mut().enumerate() {
                        *x = if c % 2 == 0 { 0x3F80 | (c as u16 & 0x7F) } else { 0x3000 | (c as u16 & 0x7F) | 0x8000 };
                    }
                }
                3 => {
                    row[k / 2] = 0x49D5; // 2^20 x 1.66: far above the window, finite in every product and sum
                    row[k - 8] = 0x0080; // smallest normal
                }
                _ => {}
            }
        }
    }
    w
}

/// Whether the in-projection gets a P12 copy on the backend named `backend` (wo and the unembed always do when P12 is on).
/// Not on CUDA: its P12 in_proj kernel is bitwise the bf16 one but slower there (spark1 #11157, nsys of the decode step at
/// 1.2k: 197 vs 182 us per visit; decode-ALU bound), while wo (48 vs 59 us) and the unembed (2873 vs 3756 us) gain. The
/// kernel stays registered (`INTARSIA_INK_P12_IN_PROJ=1` packs it anyway, for measurement).
pub fn pack_in_proj(backend: &str) -> bool {
    backend != "cuda" || std::env::var("INTARSIA_INK_P12_IN_PROJ").is_ok_and(|v| v.trim() == "1") || in_proj_gemv(backend)
}

/// `INTARSIA_INK_IN_PROJ_GEMV` (lane ink-cuda-attn): `=1` / `=0` force the GEMV form on / off; unset, [`in_proj_gemv`]
/// decides per backend. Read once.
pub fn in_proj_gemv_env() -> Option<bool> {
    static ON: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("INTARSIA_INK_IN_PROJ_GEMV").map(|v| v.trim().to_string()) {
        Ok(v) if v == "1" => Some(true),
        Ok(v) if v == "0" => Some(false),
        _ => None,
    })
}

/// The in-projection at 1..=[`ROWS_MAX`] rows in the GEMV form (one warp per weight row; on CUDA
/// `ink_in_proj_{bf16,p12}_gemv`, bitwise twins) instead of the mma form, and its P12 copy packed and used at every such row
/// count. Default on for CUDA (brief 31, user decision 6, 2026-11-22: +8-9 % spec-off decode; G2 short decode drift 3.73e-3
/// -> 4.43e-3 accepted, no precision loss in the f64 test), off elsewhere; `INTARSIA_INK_IN_PROJ_GEMV=0` reverts. Rows-exact
/// either way; the forms sum in different orders, so the switch moves the model's output bits.
pub fn in_proj_gemv(backend: &str) -> bool {
    in_proj_gemv_env().unwrap_or(backend == "cuda")
}

/// Most rows of an in-projection call that take its P12 copy: [`DENSE_ROWS_MAX`], or [`ROWS_MAX`] in the GEMV form.
pub fn in_proj_rows_max(backend: &str) -> usize {
    if in_proj_gemv(backend) {
        ROWS_MAX
    } else {
        DENSE_ROWS_MAX
    }
}

/// What `INTARSIA_INK_PACK12` (`env`) decides on the backend named `backend`, which registers the P12 kernels
/// (`Op::MatmulP12`: Metal, CUDA, HIP and the cpu reference) or not: whether to pack, and a load-log note when a forced
/// `1` is not honoured. `0` / `off` / `false`: never. `1` / `on` / `true`: wherever the kernels exist; a backend without
/// them keeps the bf16 weights and says so (it never packs and then fails at plan or dispatch). Unset (or anything else):
/// Metal and CUDA (default-on on CUDA since lane ink-cuda-p12, spark1 #11172: decode +4.4 % short / +4.3 % 1.2k / +4.5 % 5k,
/// bitwise at model level) and HIP (lane ink-hip-p12, `docs/results/2026-11-26-inkling2-hip-p12.md`), where they register the kernels.
pub fn decide(env: Option<&str>, backend: &str, has_kernels: bool) -> (bool, Option<String>) {
    match env.map(str::trim) {
        Some("0" | "off" | "false") => (false, None),
        Some("1" | "on" | "true") if has_kernels => (true, None),
        Some("1" | "on" | "true") => {
            (false, Some(format!("{ENV}=1: the {backend} backend has no P12 kernels (MatmulP12); keeping the bf16 weights")))
        }
        _ => (matches!(backend, "metal" | "cuda" | "hip") && has_kernels, None),
    }
}

/// Whether Inkling-2 packs P12 copies at load on `b` ([`decide`] on the environment and `b`'s kernel list). Prints the
/// note of a forced `1` that is not honoured, once per process.
pub fn enabled(b: &dyn Backend) -> bool {
    let has = b.kernels().iter().any(|k| k.op == crate::ops::Op::MatmulP12);
    let (on, note) = decide(std::env::var(ENV).ok().as_deref(), b.name(), has);
    static NOTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if let Some(n) = note {
        if !NOTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!("{n}");
        }
    }
    on
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decide_default_metal_off_switch_and_no_kernel_backends() {
        // unset: Metal and CUDA (and only with the kernels)
        assert_eq!(decide(None, "metal", true), (true, None));
        assert_eq!(decide(None, "cuda", true), (true, None));
        assert_eq!(decide(None, "cpu", true), (false, None));
        assert_eq!(decide(None, "cuda", false), (false, None));
        assert_eq!(decide(None, "hip", false), (false, None));
        assert_eq!(decide(Some("garbage"), "metal", true), (true, None));
        // HIP with its kernels (lane ink-hip-p12): on by default, off with the switch, forced on stays on
        assert_eq!(decide(None, "hip", true), (true, None));
        assert_eq!(decide(Some("1"), "hip", true), (true, None));
        for v in ["0", "off", "false"] {
            assert_eq!(decide(Some(v), "hip", true), (false, None));
        }
        // the in-projection packs on HIP (its bf16 form is the GEMV form, which the P12 kernel twins at 1..=DENSE_ROWS_MAX rows)
        assert!(pack_in_proj("hip"));
        // the off switch, everywhere
        for v in ["0", "off", "false", " 0 "] {
            assert_eq!(decide(Some(v), "metal", true), (false, None));
            assert_eq!(decide(Some(v), "cpu", true), (false, None));
        }
        // forced on: wherever the kernels exist; without them bf16 is kept, with a note naming the backend
        assert_eq!(decide(Some("1"), "cpu", true), (true, None));
        assert_eq!(decide(Some("on"), "metal", true), (true, None));
        for b in ["cuda", "hip"] {
            let (on, note) = decide(Some("1"), b, false);
            assert!(!on, "{b}");
            let n = note.expect("a note");
            assert!(n.contains(b) && n.contains("keeping the bf16 weights"), "{n}");
        }
    }

    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn roundtrip(w: &[u16], n: usize, k: usize) -> Packed {
        let p = pack(w, n, k, 3).unwrap();
        assert_eq!(decode(&p.codes, &p.aux, n, k).unwrap(), w, "round trip");
        p
    }

    #[test]
    fn roundtrip_weight_like() {
        // normal values over ~26 binades, like the checkpoint's dense rows
        let (n, k) = (37, 2304);
        let mut s = 0x1234_5678_9abc_def1u64;
        let w: Vec<u16> = (0..n * k)
            .map(|_| {
                let r = rng(&mut s);
                let e = 98 + (r % 28) as u16;
                ((r >> 8) as u16 & 0x8000) | e << 7 | (r >> 20) as u16 & 0x7F
            })
            .collect();
        let p = roundtrip(&w, n, k);
        assert!(p.escapes() > 0);
        assert_eq!(p.bytes(), n * 3456 + 4 * (n + 1 + p.escapes()));
    }

    #[test]
    fn roundtrip_adversarial() {
        let k = 64;
        let mut rows: Vec<Vec<u16>> = Vec::new();
        // +-0, subnormals, the max exponent / inf / nan, one row of all escapes (alternating far exponents), a K-tail escape
        rows.push((0..k).map(|i| if i % 2 == 0 { 0x0000 } else { 0x8000 }).collect());
        rows.push((0..k).map(|i| (i as u16 % 3).wrapping_mul(0x8000) / 2 | (i as u16 % 127 + 1)).collect()); // wrapping: the values the release-built test used
        rows.push((0..k).map(|i| [0x7F80, 0xFF80, 0x7FC1, 0x7F7F, 0x3F80][i % 5]).collect());
        rows.push((0..k).map(|i| if i % 2 == 0 { 0x0001 } else { 0x7F7F }).collect());
        let mut tail: Vec<u16> = (0..k).map(|i| 0x3C00 | (i as u16 & 0x7F)).collect();
        tail[k - 1] = 0x0080;
        tail[k - 8] = 0xFF00;
        rows.push(tail);
        // every exponent once per row (16 fit, the rest escape)
        rows.push((0..k).map(|i| ((i as u16 * 4) & 0xFF) << 7 | 0x55).collect());
        let w: Vec<u16> = rows.concat();
        let n = rows.len();
        let p = roundtrip(&w, n, k);
        assert!(p.row_escapes(3) >= k / 2, "all-escape-ish row");
        // every bf16 bit pattern
        let all: Vec<u16> = (0..=u16::MAX).collect();
        roundtrip(&all, 65536 / 256, 256);
        // a 1-row weight and a thread count above the rows
        roundtrip(&w[..k], 1, k);
        let p = pack(&w, n, k, 64).unwrap();
        assert_eq!(p, pack(&w, n, k, 1).unwrap(), "same bytes for any thread count");
    }

    #[test]
    fn rejects() {
        assert!(pack(&[0; 12], 1, 12, 1).is_err());
        let p = pack(&[0x3F80; 16], 2, 8, 1).unwrap();
        let mut aux = p.aux.clone();
        aux.push(0);
        assert!(decode(&p.codes, &aux, 2, 8).is_err());
        assert!(decode(&p.codes[1..], &p.aux, 2, 8).is_err());
    }
}
