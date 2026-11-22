//! Lane ink-prefill: second-generation grouped prefill GEMMs of the Inkling-2-Mini experts
//! (kernels/metal/inkling2_moe_nax2.metal). Same bindings, grid and grouping as the first generation in inkmoe.rs; this module
//! only picks the entry points (and the tile rows, which set `moe_group`'s tile size) for a prefill call.
//!
//! `INTARSIA_INK_MOE_NAX2=<form>` picks a form by name suffix (`m64_k32_v3`, ...), `0` keeps the first generation.

/// Entry points of inkling2_moe_nax2.metal as (format 0 NVFP4 / 1 MXFP8, tile rows, suffix); every suffix has a gu and a dn.
pub const NAX2: &[(usize, usize, &str)] = &[
    (0, 64, "m64_k64_v0"),
    (0, 64, "m64_k64_v1"),
    (0, 64, "m64_k32_v1"),
    (0, 64, "m64_k32_v3"),
    (0, 32, "m32_k64_v3"),
    (0, 64, "m64_k64_v4"),
    (0, 64, "m64_k64_v8"),
    (0, 64, "m64_k64_v16"),
    (0, 64, "m64_k64_v28"),
    (1, 64, "m64_k64_v1"),
    (1, 64, "m64_k32_v3"),
    (0, 128, "m128_k64_v1"),
    (0, 128, "m128_k32_v3"),
    (1, 128, "m128_k64_v1"),
    (1, 128, "m128_k32_v3"),
    // V & 64: 256 threads, one decoded weight stage shared by the two 64-row halves of a 128-row tile (brief 5 item 5a)
    (0, 128, "m128_k64_v65"),
    (1, 128, "m128_k64_v65"),
    // format 2: pre-decoded bf16 weights (dequant-once experiment, brief 5 item 5b); selected as `dq1` on NVFP4 calls
    (2, 64, "m64_k64_v1"),
];

const FMT: [&str; 3] = ["nv", "mx", "dq"];

/// The dequant kernel of the dequant-once experiment (NVFP4 blocks -> bf16, inkling2_moe_nax2.metal).
pub const DQ_NV: &str = "ink_moe_dq_nv";

/// Third-generation entry points (kernels/metal/inkling2_moe_nax3.metal, A read from device in slot order, the weight tile
/// double-buffered) as (format, tile rows, tile columns BN, suffix). Selected as `nax3_<suffix>`.
pub const NAX3: &[(usize, usize, usize, &str)] = &[
    (0, 64, 64, "m64_n64_k64"),
    (0, 64, 128, "m64_n128_k32"),
    (0, 32, 128, "m32_n128_k32"),
    (0, 32, 64, "m32_n64_k64"),
    (1, 64, 64, "m64_n64_k64"),
    (1, 64, 128, "m64_n128_k32"),
];

/// x rows into slot order for the third generation's gate+up.
pub const NAX3_GATHER: &str = "ink_moe_nax3_gather";

/// A selected prefill form: gate+up and down entry points, tile rows, and for the third generation its tile columns.
#[derive(Clone, Copy, Debug)]
pub struct Form {
    pub gu: &'static str,
    pub dn: &'static str,
    pub tm: usize,
    pub bn3: Option<usize>,
    /// Threads per threadgroup of the gu / dn kernels.
    pub threads: usize,
    /// Dequant-once: decode the weights to a bf16 scratch first (format-2 kernels).
    pub dq: bool,
}

/// (gu, dn) shader names of each `NAX2` entry, then of each `NAX3` entry.
fn names() -> &'static [(&'static str, &'static str)] {
    static NAMES: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
    NAMES.get_or_init(|| {
        let g2 = NAX2.iter().map(|&(f, _, s)| (format!("ink_moe_nax2_gu_{}_{s}", FMT[f]), format!("ink_moe_nax2_dn_{}_{s}", FMT[f])));
        let g3 = NAX3.iter().map(|&(f, _, _, s)| (format!("ink_moe_nax3_gu_{}_{s}", FMT[f]), format!("ink_moe_nax3_dn_{}_{s}", FMT[f])));
        g2.chain(g3).map(|(g, d)| (leak(g), leak(d))).collect()
    })
}

/// Every shader name of inkling2_moe_nax2.metal and inkling2_moe_nax3.metal (metal4.0 library).
pub fn metal4_shaders() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = names().iter().flat_map(|&(g, d)| [g, d]).collect();
    v.push(NAX3_GATHER);
    v.push(DQ_NV);
    v.extend(BLEND_SHADERS);
    v.extend(PROBE_SHADERS.iter().flat_map(|&(g, d)| [g, d]));
    v
}

/// Timing probes of the first-generation grouped NVFP4 experts at 64-row tiles (inkling2_moe_probe.metal, WRONG RESULTS):
/// (gu, dn) of probe 0 (unchanged copy), 1 (bit copy instead of the decode arithmetic), 2 (no weight loads).
pub const PROBE_SHADERS: [(&str, &str); 3] = [
    ("ink_moe_probe_gu_nv_p0", "ink_moe_probe_dn_nv_p0"),
    ("ink_moe_probe_gu_nv_p1", "ink_moe_probe_dn_nv_p1"),
    ("ink_moe_probe_gu_nv_p2", "ink_moe_probe_dn_nv_p2"),
];

#[cfg(feature = "ink-probe")]
static PROBE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0xFF);

/// Benchmarks built with the `ink-probe` feature only: select probe `v` (0..=2) of [`PROBE_SHADERS`] for NVFP4 prefill
/// calls whose first-generation tile is 64 rows, or None for the normal kernels. No environment variable reaches it.
#[cfg(feature = "ink-probe")]
pub fn set_moe_probe(v: Option<u8>) {
    PROBE.store(v.filter(|&p| p < 3).unwrap_or(0xFF), std::sync::atomic::Ordering::Relaxed);
}

/// The probe (gu, dn) for an NVFP4 call with `tm`-row tiles; always None without the `ink-probe` feature.
pub(crate) fn moe_probe(mx: bool, tm: usize) -> Option<(&'static str, &'static str)> {
    #[cfg(feature = "ink-probe")]
    {
        let v = PROBE.load(std::sync::atomic::Ordering::Relaxed);
        if !mx && tm == 64 && v < 3 {
            return Some(PROBE_SHADERS[v as usize]);
        }
    }
    let _ = (mx, tm);
    None
}

static FORM: std::sync::Mutex<Option<Option<String>>> = std::sync::Mutex::new(None);

fn form() -> Option<String> {
    let mut f = FORM.lock().unwrap();
    f.get_or_insert_with(|| std::env::var("INTARSIA_INK_MOE_NAX2").ok().filter(|v| !v.is_empty() && v != "0")).clone()
}

/// Benchmarks: select a form by suffix (`None` or "0": the first generation), overriding `INTARSIA_INK_MOE_NAX2`.
pub fn set_form(v: Option<&str>) {
    *FORM.lock().unwrap() = Some(v.filter(|v| !v.is_empty() && *v != "0").map(str::to_string));
}

/// The selected form for a prefill call of format `mx` whose first-generation tile is `tm` rows, or None to keep the first
/// generation.
pub fn moe_nax(mx: bool, tm: usize) -> Option<Form> {
    let want = form()?;
    let f = mx as usize;
    if let Some(w3) = want.strip_prefix("nax3_") {
        let _ = tm; // an explicitly selected form runs at any row count (its tile may exceed the mean rows per expert)
        let i = NAX3.iter().position(|&(ff, _, _, s)| ff == f && s == w3)?;
        let (gu, dn) = names()[NAX2.len() + i];
        return Some(Form { gu, dn, tm: NAX3[i].1, bn3: Some(NAX3[i].2), threads: 128, dq: false });
    }
    if want == "dq1" {
        if mx {
            return None;
        }
        let i = NAX2.iter().position(|&(ff, _, _)| ff == 2)?;
        return Some(Form { gu: names()[i].0, dn: names()[i].1, tm: NAX2[i].1, bn3: None, threads: 128, dq: true });
    }
    let i = NAX2.iter().position(|&(ff, _, s)| ff == f && s == want.as_str())?;
    let threads = if NAX2[i].2.rsplit('_').next().and_then(|v| v.strip_prefix('v')).and_then(|v| v.parse::<u32>().ok()).is_some_and(|v| v & 64 != 0) { 256 } else { 128 };
    Some(Form { gu: names()[i].0, dn: names()[i].1, tm: NAX2[i].1, bn3: None, threads, dq: false })
}

/// Timing probe of the grouped prefill path (WRONG RESULTS; bench builds with the `ink-probe` feature only, no environment
/// variable): bit 1 skips gate+up, 2 the down, 4 the gather. `moe_group` always runs. Always 0 without the feature.
pub fn nax_skip() -> u8 {
    #[cfg(feature = "ink-probe")]
    {
        SKIP.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(feature = "ink-probe"))]
    {
        0
    }
}

#[cfg(feature = "ink-probe")]
static SKIP: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Benchmarks built with the `ink-probe` feature: set the timing probe bits (see [`nax_skip`]).
#[cfg(feature = "ink-probe")]
pub fn set_skip(v: u8) {
    SKIP.store(v & 7, std::sync::atomic::Ordering::Relaxed);
}

/// Prefill blend chunks (rows > 16): the h projection on the NAX GEMM (InkInProj's prefill kernel) instead of the 16-row
/// GEMV passes. `INTARSIA_INK_BLEND_NAX=1` takes only this step, `0` neither (default 2: this and the gate below).
pub fn blend_nax(rows: usize) -> bool {
    rows > 16 && blend_level() >= 1 && !EXACT.with(|e| e.get())
}

thread_local! {
    /// The current mix call's `exact_rows` (set by inkmix.rs `launch_inkmix`): row-exact calls keep m-mix's GEMV path.
    static EXACT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Called by inkmix.rs for every mix call: whether it promises row-exact results.
pub fn set_exact(v: bool) {
    EXACT.with(|e| e.set(v));
}

/// Prefill blend chunks: the gate + norm on the accelerators too (kernels/metal/inkling2_blend_nax.metal),
/// `INTARSIA_INK_BLEND_NAX=2`.
pub fn blend_gate_nax(rows: usize) -> bool {
    blend_nax(rows) && blend_level() >= 2
}

fn blend_level() -> u8 {
    static L: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    // default 2 (lane ink-prefill: admitted, ~2.5x faster blend on 2048-row chunks); 0 restores m-mix's GEMV path
    *L.get_or_init(|| std::env::var("INTARSIA_INK_BLEND_NAX").ok().and_then(|v| v.parse().ok()).unwrap_or(2))
}

/// Entry points of inkling2_blend_nax.metal.
pub const BLEND_SHADERS: [&str; 2] = ["ink_blend_gf_nax", "ink_blend_yn"];

/// `InkBlendP` of inkling2_blend_nax.metal (the same 96-byte layout as inkling2_mix.metal's).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct BlendP {
    d: u32,
    rows: u32,
    s: u32,
    sd: u32,
    g: u32,
    nsl: u32,
    cs: u32,
    t0: u32,
    ocount: u32,
    dcount: u32,
    mcount: u32,
    pad0: u32,
    dt0: u32,
    mt0: u32,
    pad1: [u32; 2],
    eps: f32,
    fpad: [f32; 3],
    sk: [u32; 8],
}

fn u32_of(v: usize) -> intarsia_core::Result<u32> {
    u32::try_from(v).map_err(|_| intarsia_core::Error::invalid(format!("metal blend prefill: {v} exceeds u32")))
}

impl crate::MetalBackend {
    /// The gate + norm of a prefill blend chunk on the accelerators: gn / fg (f32, into the dead pair region and the y
    /// region of the blend scratch), then y and the norm per (stream, row). `dims` = [D, rows, S, Sd, G, t0, ocount, dcount,
    /// mcount, dt0, mt0] as inkmix.rs's `InkBlendP`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ink_blend_gate_nax(
        &self,
        dest_b: &crate::dense::Bind,
        msg_b: &crate::dense::Bind,
        h_b: &crate::dense::Bind,
        gain_b: &crate::dense::Bind,
        forget_b: &crate::dense::Bind,
        gn: &intarsia_core::backend::TensorRef,
        fg: &intarsia_core::backend::TensorRef,
        norm: &intarsia_core::backend::TensorRef,
        out: &intarsia_core::backend::TensorRef,
        dims: [usize; 11],
        eps: f32,
    ) -> intarsia_core::Result<()> {
        use intarsia_core::DType;
        let [d, rows, s, sd, g, t0, ocount, dcount, mcount, dt0, mt0] = dims;
        if d > 256 * 12 {
            return Err(intarsia_core::Error::invalid(format!("metal blend prefill: D = {d} > 3072")));
        }
        let gn_t = intarsia_core::backend::TensorRef { dtype: DType::F32, shape: vec![s * rows * d], ..gn.clone() };
        let fg_t = intarsia_core::backend::TensorRef { dtype: DType::F32, shape: vec![s * rows * d], ..fg.clone() };
        let (gn_b, fg_b) = (self.bind(&gn_t, DType::F32, true, "gn")?, self.bind(&fg_t, DType::F32, true, "fg")?);
        let (nrm_b, out_b) = (self.bind(norm, DType::Bf16, false, "norm")?, self.bind(out, DType::Bf16, true, "out")?);
        let p = BlendP {
            d: u32_of(d)?,
            rows: u32_of(rows)?,
            s: u32_of(s)?,
            sd: u32_of(sd)?,
            g: u32_of(g)?,
            t0: u32_of(t0)?,
            ocount: u32_of(ocount)?,
            dcount: u32_of(dcount)?,
            mcount: u32_of(mcount)?,
            dt0: u32_of(dt0)?,
            mt0: u32_of(mt0)?,
            eps,
            sk: [gain_b.skip, forget_b.skip, nrm_b.skip, out_b.skip, 0, 0, 0, 0],
            ..Default::default()
        };
        let td = d.div_ceil(64);
        self.dispatch(
            BLEND_SHADERS[0],
            &[h_b, gain_b, forget_b, &gn_b, &fg_b],
            &p,
            crate::dense::Grid::Groups { x: s * td, y: rows.div_ceil(64), threads: 128 },
        )?;
        self.dispatch(
            BLEND_SHADERS[1],
            &[dest_b, msg_b, &gn_b, &fg_b, &nrm_b, &out_b],
            &p,
            crate::dense::Grid::Groups { x: s * rows, y: 1, threads: 256 },
        )
    }
}

/// `INTARSIA_INK_PREFILL_TRACE=1`: print each prefill kernel this lane's hooks dispatch, once per name, to stderr (proves the
/// selected path ran: admission and benches otherwise cannot tell a silent fallback from the new kernel).
pub fn trace(name: &str) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("INTARSIA_INK_PREFILL_TRACE").is_ok_and(|v| v != "0")) {
        return;
    }
    static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let mut s = SEEN.lock().unwrap();
    if !s.iter().any(|x| x == name) {
        s.push(name.to_string());
        eprintln!("ink-prefill trace: {name}");
    }
}

/// Diagnostic (`INTARSIA_INK_MOE_DUMP=<file>`, slow: a synchronize per call): append the rows per expert of every prefill
/// experts call (T > 16) as one line `<nv|mx> T=<t> <count of expert 0> .. <count of expert E-1>`.
pub fn moe_dump_path() -> Option<&'static str> {
    static P: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    P.get_or_init(|| std::env::var("INTARSIA_INK_MOE_DUMP").ok().filter(|v| !v.is_empty())).as_deref()
}

impl crate::MetalBackend {
    pub(crate) fn moe_dump(&self, ids: &intarsia_core::backend::TensorRef, t: usize, k: usize, e: usize, mx: bool) -> intarsia_core::Result<()> {
        use intarsia_core::backend::Backend;
        use std::io::Write;
        let Some(path) = moe_dump_path() else { return Ok(()) };
        self.synchronize(0)?;
        let mut raw = vec![0u8; t * k * 4];
        self.download(&ids.buf, ids.offset, &mut raw, 0)?;
        let mut c = vec![0usize; e];
        for w in raw.chunks_exact(4) {
            let id = i32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            if (0..e as i32).contains(&id) {
                c[id as usize] += 1;
            }
        }
        let line = format!("{} T={t} {}\n", if mx { "mx" } else { "nv" }, c.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "));
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).map_err(|err| intarsia_core::Error::backend(format!("moe dump {path}: {err}")))?;
        f.write_all(line.as_bytes()).map_err(|err| intarsia_core::Error::backend(format!("moe dump {path}: {err}")))
    }
}

/// Dequant-once scratch (bf16 gate+up and down weights of every expert), grown on demand and kept for the process.
static DQ_SCRATCH: std::sync::Mutex<Option<(intarsia_core::backend::Buffer, usize)>> = std::sync::Mutex::new(None);

impl crate::MetalBackend {
    /// Dequant-once experiment: decode w13 [rows13, H / 2] and w2 [rows2, I / 2] (NVFP4, scales s13 / s2) to bf16 in a
    /// scratch and return the two binds to hand the format-2 kernels as W.
    pub(crate) fn ink_dq_weights(
        &self,
        w13: &crate::dense::Bind,
        s13: &crate::dense::Bind,
        w2: &crate::dense::Bind,
        s2: &crate::dense::Bind,
        rows13: usize,
        h: usize,
        rows2: usize,
        i: usize,
    ) -> intarsia_core::Result<(crate::dense::Bind, crate::dense::Bind)> {
        use intarsia_core::backend::{Backend, MemKind};
        let (b13, b2) = (rows13 * h * 2, rows2 * i * 2);
        let need = b13 + b2;
        let mut g = DQ_SCRATCH.lock().unwrap();
        if g.as_ref().is_none_or(|(_, n)| *n < need) {
            *g = Some((self.alloc(need, MemKind::Device)?, need));
        }
        let buf = g.as_ref().unwrap().0.clone();
        drop(g);
        let t13 = intarsia_core::backend::TensorRef { buf: buf.clone(), offset: 0, dtype: intarsia_core::DType::U8, shape: vec![b13] };
        let t2 = intarsia_core::backend::TensorRef { buf, offset: b13, dtype: intarsia_core::DType::U8, shape: vec![b2] };
        let (d13, d2) = (self.bind(&t13, intarsia_core::DType::U8, true, "dq13")?, self.bind(&t2, intarsia_core::DType::U8, true, "dq2")?);
        for (w, s, d, rows, kd) in [(w13, s13, &d13, rows13, h), (w2, s2, &d2, rows2, i)] {
            let rk = [rows as u32, kd as u32, s.skip, 0u32];
            let n = rows * kd / 16;
            self.dispatch(DQ_NV, &[w, s, d], &rk, crate::dense::Grid::Groups { x: n.div_ceil(256), y: 1, threads: 256 })?;
        }
        Ok((d13, d2))
    }
}
