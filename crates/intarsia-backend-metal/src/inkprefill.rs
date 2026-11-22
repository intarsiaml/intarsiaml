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
];

const FMT: [&str; 2] = ["nv", "mx"];

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
    v
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
        let i = NAX3.iter().position(|&(ff, rows, _, s)| ff == f && s == w3 && rows <= tm.max(32))?;
        let (gu, dn) = names()[NAX2.len() + i];
        return Some(Form { gu, dn, tm: NAX3[i].1, bn3: Some(NAX3[i].2) });
    }
    let i = NAX2.iter().position(|&(ff, rows, s)| ff == f && s == want.as_str() && rows <= tm.max(32))?;
    Some(Form { gu: names()[i].0, dn: names()[i].1, tm: NAX2[i].1, bn3: None })
}

/// Timing probe of the grouped prefill path (`INTARSIA_INK_MOE_NAXSKIP`, a bit set; wrong results): 1 skips gate+up, 2 the
/// down, 4 the gather. `moe_group` always runs.
pub fn nax_skip() -> u8 {
    let v = SKIP.load(std::sync::atomic::Ordering::Relaxed);
    if v != 0xFF {
        return v;
    }
    let e = std::env::var("INTARSIA_INK_MOE_NAXSKIP").ok().and_then(|v| v.parse().ok()).unwrap_or(0u8) & 7;
    SKIP.store(e, std::sync::atomic::Ordering::Relaxed);
    e
}

static SKIP: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0xFF);

/// Benchmarks: set the timing probe bits (see [`nax_skip`]).
pub fn set_skip(v: u8) {
    SKIP.store(v & 7, std::sync::atomic::Ordering::Relaxed);
}
