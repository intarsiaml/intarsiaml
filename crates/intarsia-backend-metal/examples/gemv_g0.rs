//! G0 bench (docs/research/metal-next-levers.md): T(R)/T(1) of the multi-row bit-exact decode GEMVs on the Qwen3-8B
//! shapes, against the serving 1-row kernels, the tiled GEMM that serves M = 2..15 today and the NAX GEMM (M >= 16).
//!
//!   cargo run --release -p intarsia-backend-metal --example gemv_g0 -- [case ...]
//!   cases: qo (4096x4096), down (K=12288, N=4096), kv (1024+1024, K=4096), gu (12288+12288, K=4096), lm (151936x4096, f32 out)
//!   env:   G0_ROUNDS (default 30), G0_POOL_MIB (3072), G0_A8=1 (weights at byte offset 8: the 8-byte-load serving kernels;
//!          no lm), G0_OUT=<csv path> (raw per-chain samples), G0_ROWS=2,4,8 (restrict multi-row counts)
//!
//! Method (as gemv_b3): each timed unit is one command buffer holding a serial chain of dispatches over a rotating pool of
//! distinct weight buffers (3 GiB, far past the SLC, so every dispatch streams its weights from DRAM like a decode step),
//! timed by the command buffer's GPU start/end. Candidates are measured interleaved: a round runs every candidate once, in
//! a rotated order, so a clock swing hits all candidates of a round alike and T(R)/T(1) is taken within a round. Every
//! sample is written with its wall-clock epoch (ms) so rounds can be joined to a macmon trace and dropped when the GPU
//! clock was below 1450 MHz (scripts/analyze_g0.py). Before the timed rounds each multi-row kernel is checked bitwise
//! against R serving 1-row launches on the same buffers.
//!
//! Run inside an exclusive fleetq slot.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gemv_g0 needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_backend_metal::bench;
    use intarsia_backend_metal::gemm_nax::{self, NaxPairParams, NaxParams};
    use intarsia_backend_metal::kernels::{GEMM_O_BF16, GEMM_O_F32};
    use intarsia_backend_metal::multirow::{self as mr, Family, PairParams};
    use intarsia_backend_metal::{encode_gemv, GemvParams, MetalBackend, MtlBuffer};
    use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
    use std::io::Write;

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Kind {
        Matmul,
        Pair,
        Logits,
    }

    struct Case {
        label: &'static str,
        kind: Kind,
        k: usize,
        n0: usize,
        n1: usize,
    }

    const CASES: [Case; 5] = [
        Case { label: "qo", kind: Kind::Matmul, k: 4096, n0: 4096, n1: 0 },
        Case { label: "down", kind: Kind::Matmul, k: 12288, n0: 4096, n1: 0 },
        Case { label: "kv", kind: Kind::Pair, k: 4096, n0: 1024, n1: 1024 },
        Case { label: "gu", kind: Kind::Pair, k: 4096, n0: 12288, n1: 12288 },
        Case { label: "lm", kind: Kind::Logits, k: 4096, n0: 151936, n1: 0 },
    ];

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Cand {
        /// The serving 1-row kernel (what a decode step launches).
        Serving,
        /// Multi-row bit-exact prototype with R rows.
        Mr(usize, usize),
        /// Multi-row split into passes of `rb` rows.
        MrS(usize, usize, usize),
        /// The tiled GEMM that serves M = 2..15 (and logits for any M >= 2) today.
        Tiled(usize),
        /// The NAX GEMM (M >= 16).
        Nax(usize),
    }

    impl Cand {
        fn label(&self) -> String {
            match self {
                Cand::Serving => "serving1".into(),
                Cand::Mr(r, p) => format!("mr{r}x{p}"),
                Cand::MrS(r, p, rb) => format!("mr{r}x{p}s{rb}"),
                Cand::Tiled(m) => format!("tiled{m}"),
                Cand::Nax(m) => format!("nax{m}"),
            }
        }
        /// Rows of activations this candidate processes per dispatch.
        fn rows(&self) -> usize {
            match self {
                Cand::Serving => 1,
                Cand::Mr(r, _) | Cand::MrS(r, _, _) | Cand::Tiled(r) | Cand::Nax(r) => *r,
            }
        }
    }

    struct Pool {
        w0: Vec<MtlBuffer>,
        w1: Vec<MtlBuffer>,
        /// x [16, K]
        x: MtlBuffer,
        /// y buffers [16, N] (bf16 or f32 per kind)
        y0: MtlBuffer,
        y1: MtlBuffer,
        /// reference outputs for the bitwise check
        r0: MtlBuffer,
        r1: MtlBuffer,
        off: usize,
    }

    const MAXR: usize = 16;

    impl Pool {
        fn new(be: &MetalBackend, c: &Case, pool_bytes: usize, off: usize) -> Pool {
            let wb0 = c.n0 * c.k * 2;
            let wb1 = c.n1 * c.k * 2;
            let count = pool_bytes.div_ceil(wb0 + wb1).max(2);
            let (mut w0, mut w1): (Vec<MtlBuffer>, Vec<MtlBuffer>) = (Vec::new(), Vec::new());
            for i in 0..count {
                for (wb, w) in [(wb0, &mut w0), (wb1, &mut w1)] {
                    if wb == 0 {
                        continue;
                    }
                    let b = be.new_mtl_buffer(wb + 64, true).unwrap();
                    // SAFETY: wb bytes inside the allocation (64 spare bytes for the offset).
                    unsafe {
                        let dst = (b.contents().as_ptr() as *mut u8).add(off);
                        if i == 0 {
                            bench::fill_bf16_random(
                                dst as *mut u16,
                                wb / 2,
                                42 + w.len() as u64 + (w.is_empty() as u64),
                            );
                        } else {
                            std::ptr::copy_nonoverlapping((w[0].contents().as_ptr() as *const u8).add(off), dst, wb);
                        }
                    }
                    w.push(b);
                }
            }
            let esz = if c.kind == Kind::Logits { 4 } else { 2 };
            let x = be.new_mtl_buffer(MAXR * c.k * 2, true).unwrap();
            // SAFETY: MAXR*k bf16.
            unsafe { bench::fill_bf16_random(x.contents().as_ptr() as *mut u16, MAXR * c.k, 7) };
            let mk = |n: usize| be.new_mtl_buffer((MAXR * n * esz).max(16) + 16, true).unwrap();
            Pool { w0, w1, x, y0: mk(c.n0), y1: mk(c.n1), r0: mk(c.n0), r1: mk(c.n1), off }
        }
    }

    fn family(kind: Kind, a8: bool) -> Family {
        match (kind, a8) {
            (Kind::Matmul, false) => Family::Matmul16,
            (Kind::Matmul, true) => Family::MatmulA8,
            (Kind::Pair, false) => Family::Pair16,
            (Kind::Pair, true) => Family::PairA8,
            (Kind::Logits, _) => Family::Logits,
        }
    }

    fn mr_name(kind: Kind, a8: bool, r: usize, p: usize) -> &'static str {
        mr::shader(family(kind, a8), r, p).unwrap_or_else(|| panic!("no multi-row shader for {r} rows x {p}"))
    }

    fn mrs_name(kind: Kind, a8: bool, r: usize, p: usize, rb: usize) -> &'static str {
        mr::shader_split(family(kind, a8), r, p, rb).unwrap()
    }

    fn serving_name(kind: Kind, a8: bool) -> &'static str {
        match (kind, a8) {
            (Kind::Matmul, false) => intarsia_backend_metal::kernels::DEFAULT_GEMV.name,
            (Kind::Matmul, true) => intarsia_backend_metal::kernels::DEFAULT_GEMV_A8.name,
            (Kind::Pair, false) => intarsia_backend_metal::kernels::GEMV_PAIR_BF16,
            (Kind::Pair, true) => intarsia_backend_metal::kernels::GEMV_PAIR_BF16_A8,
            (Kind::Logits, _) => intarsia_backend_metal::kernels::GEMV_F32O_V,
        }
    }

    /// Encode candidate `cand` over pool entry `i` into `enc`.
    fn dispatch(
        be: &MetalBackend,
        enc: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        c: &Case,
        a8: bool,
        cand: Cand,
        pool: &Pool,
        i: usize,
        out: (&MtlBuffer, &MtlBuffer),
    ) {
        let off = pool.off;
        let (w0, w1) = (&pool.w0[i], pool.w1.get(i));
        let (x, y0, y1) = (&pool.x, out.0, out.1);
        let gp = GemvParams { k: c.k as u32, n: c.n0 as u32, ..Default::default() };
        let pp = PairParams { k: c.k as u32, n0: c.n0 as u32, n1: c.n1 as u32, ..Default::default() };
        let pso = |name: &str| be.pso(name).unwrap_or_else(|| panic!("no pso {name}"));
        if let Cand::MrS(r, p, rb) = cand {
            let name = mrs_name(c.kind, a8, r, p, rb);
            match c.kind {
                Kind::Matmul => mr::encode_matmul_rows(enc, pso(name), (w0, off), (x, 0), (y0, 0), &gp, p),
                Kind::Pair => mr::encode_pair_rows(
                    enc,
                    pso(name),
                    (w0, off),
                    (w1.unwrap(), off),
                    (x, 0),
                    (y0, 0),
                    (y1, 0),
                    &pp,
                    p,
                ),
                Kind::Logits => mr::encode_logits_rows(enc, pso(name), (w0, off), (x, 0), (y0, 0), &gp, p),
            }
            return;
        }
        match (cand, c.kind) {
            (Cand::Serving, Kind::Matmul) => encode_gemv(
                enc,
                pso(serving_name(c.kind, a8)),
                (w0, off),
                (x, 0),
                (y0, 0),
                &gp,
                c.n0 as u32,
                mr::RT_THREADS,
            ),
            (Cand::Mr(r, p), Kind::Matmul) => {
                mr::encode_matmul_rows(enc, pso(mr_name(c.kind, a8, r, p)), (w0, off), (x, 0), (y0, 0), &gp, p)
            }
            (Cand::Serving, Kind::Pair) => mr::encode_pair_rows(
                enc,
                pso(serving_name(c.kind, a8)),
                (w0, off),
                (w1.unwrap(), off),
                (x, 0),
                (y0, 0),
                (y1, 0),
                &pp,
                1,
            ),
            (Cand::Mr(r, p), Kind::Pair) => mr::encode_pair_rows(
                enc,
                pso(mr_name(c.kind, a8, r, p)),
                (w0, off),
                (w1.unwrap(), off),
                (x, 0),
                (y0, 0),
                (y1, 0),
                &pp,
                p,
            ),
            (Cand::Serving, Kind::Logits) => {
                mr::encode_logits_rows(enc, pso(serving_name(c.kind, a8)), (w0, off), (x, 0), (y0, 0), &gp, 1)
            }
            (Cand::Mr(r, p), Kind::Logits) => {
                mr::encode_logits_rows(enc, pso(mr_name(c.kind, a8, r, p)), (w0, off), (x, 0), (y0, 0), &gp, p)
            }
            (Cand::Tiled(m), Kind::Matmul) => mr::encode_tiled_gemm(
                enc,
                pso(GEMM_O_BF16),
                (x, 0),
                (w0, off),
                (y0, 0),
                m as u32,
                c.n0 as u32,
                c.k as u32,
            ),
            (Cand::Tiled(m), Kind::Pair) => {
                // no pair kernel for M < 16: the model runs the two projections as two ordinary Matmuls
                mr::encode_tiled_gemm(
                    enc,
                    pso(GEMM_O_BF16),
                    (x, 0),
                    (w0, off),
                    (y0, 0),
                    m as u32,
                    c.n0 as u32,
                    c.k as u32,
                );
                mr::encode_tiled_gemm(
                    enc,
                    pso(GEMM_O_BF16),
                    (x, 0),
                    (w1.unwrap(), off),
                    (y1, 0),
                    m as u32,
                    c.n1 as u32,
                    c.k as u32,
                );
            }
            (Cand::Tiled(m), Kind::Logits) => mr::encode_tiled_gemm(
                enc,
                pso(GEMM_O_F32),
                (x, 0),
                (w0, off),
                (y0, 0),
                m as u32,
                c.n0 as u32,
                c.k as u32,
            ),
            (Cand::Nax(m), Kind::Matmul) => {
                let v = gemm_nax::pick_variant(m, c.n0);
                let p = NaxParams { m: m as u32, n: c.n0 as u32, k: c.k as u32, ..Default::default() };
                gemm_nax::encode_nax(enc, pso(v.shader), v, (x, 0), (w0, off), (y0, 0), &p)
            }
            (Cand::Nax(m), Kind::Pair) => {
                let v = gemm_nax::pick_variant(m, c.n0 + c.n1);
                let p = NaxPairParams {
                    m: m as u32,
                    n0: c.n0 as u32,
                    n1: c.n1 as u32,
                    k: c.k as u32,
                    ..Default::default()
                };
                gemm_nax::encode_nax_pair(
                    enc,
                    pso(v.pair_shader),
                    v,
                    (x, 0),
                    (w0, off),
                    (w1.unwrap(), off),
                    (y0, 0),
                    (y1, 0),
                    &p,
                )
            }
            (Cand::MrS(..), _) => unreachable!(),
            (Cand::Nax(_), Kind::Logits) => unreachable!("no NAX logits kernel"),
        }
    }

    fn chain_seconds(be: &MetalBackend, c: &Case, a8: bool, cand: Cand, pool: &Pool, count: usize) -> f64 {
        let cb = be.queue().commandBuffer().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        for d in 0..count {
            dispatch(be, &enc, c, a8, cand, pool, d % pool.w0.len(), (&pool.y0, &pool.y1));
        }
        enc.endEncoding();
        be.commit(&cb);
        MetalBackend::wait(&cb).unwrap();
        bench::gpu_seconds(&cb)
    }

    /// `Device Utilization %` of the GPU now (ioreg).
    fn gpu_util() -> Option<u32> {
        let out = std::process::Command::new("ioreg").args(["-r", "-c", "IOAccelerator", "-d", "1"]).output().ok()?;
        let s = String::from_utf8_lossy(&out.stdout);
        let key = "\"Device Utilization %\"=";
        let at = s.find(key)? + key.len();
        let digits: String = s[at..].chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse().ok()
    }

    fn epoch_ms() -> u128 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()
    }

    /// Bitwise check of the multi-row kernel `r` against `r` serving 1-row launches (entry 0 of the pool, distinct rows of x).
    fn bit_check(be: &MetalBackend, c: &Case, a8: bool, pool: &Pool, r: usize, cand: Cand) -> (bool, usize) {
        let esz = if c.kind == Kind::Logits { 4 } else { 2 };
        let zero = |b: &MtlBuffer| unsafe { std::ptr::write_bytes(b.contents().as_ptr() as *mut u8, 0xEE, b.length()) };
        for b in [&pool.y0, &pool.y1, &pool.r0, &pool.r1] {
            zero(b);
        }
        // serving reference, one row at a time, into r0 / r1 rows
        let cb = be.queue().commandBuffer().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        let off = pool.off;
        let (w0, w1) = (&pool.w0[0], pool.w1.first());
        for row in 0..r {
            let xo = row * c.k * 2;
            let gp = GemvParams { k: c.k as u32, n: c.n0 as u32, ..Default::default() };
            let pp = PairParams { k: c.k as u32, n0: c.n0 as u32, n1: c.n1 as u32, ..Default::default() };
            let pso = |name: &str| be.pso(name).unwrap();
            let (y0o, y1o) = (row * c.n0 * esz, row * c.n1 * esz);
            match c.kind {
                Kind::Matmul => encode_gemv(
                    &enc,
                    pso(serving_name(c.kind, a8)),
                    (w0, off),
                    (&pool.x, xo),
                    (&pool.r0, y0o),
                    &gp,
                    c.n0 as u32,
                    mr::RT_THREADS,
                ),
                Kind::Pair => mr::encode_pair_rows(
                    &enc,
                    pso(serving_name(c.kind, a8)),
                    (w0, off),
                    (w1.unwrap(), off),
                    (&pool.x, xo),
                    (&pool.r0, y0o),
                    (&pool.r1, y1o),
                    &pp,
                    1,
                ),
                Kind::Logits => mr::encode_logits_rows(
                    &enc,
                    pso(serving_name(c.kind, a8)),
                    (w0, off),
                    (&pool.x, xo),
                    (&pool.r0, y0o),
                    &gp,
                    1,
                ),
            }
        }
        enc.endEncoding();
        be.commit(&cb);
        MetalBackend::wait(&cb).unwrap();
        // multi-row
        let cb = be.queue().commandBuffer().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        dispatch(be, &enc, c, a8, cand, pool, 0, (&pool.y0, &pool.y1));
        enc.endEncoding();
        be.commit(&cb);
        MetalBackend::wait(&cb).unwrap();
        let cmp = |a: &MtlBuffer, b: &MtlBuffer| unsafe {
            let n = a.length();
            let (pa, pb) = (a.contents().as_ptr() as *const u8, b.contents().as_ptr() as *const u8);
            std::slice::from_raw_parts(pa, n) == std::slice::from_raw_parts(pb, n)
        };
        let ok = cmp(&pool.y0, &pool.r0) && (c.kind != Kind::Pair || cmp(&pool.y1, &pool.r1));
        (ok, r * c.n0 * esz)
    }

    pub fn main() {
        let be = MetalBackend::new().expect("metal backend");
        let want: Vec<String> = std::env::args().skip(1).collect();
        let rounds: usize = std::env::var("G0_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
        let pool_mib: usize = std::env::var("G0_POOL_MIB").ok().and_then(|v| v.parse().ok()).unwrap_or(3072);
        let a8 = std::env::var("G0_A8").is_ok();
        let off = if a8 { 8 } else { 0 };
        let rows_filter: Option<Vec<usize>> =
            std::env::var("G0_ROWS").ok().map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect());
        let only_split = std::env::var("G0_ONLY_SPLIT").is_ok();
        let per_filter: Option<Vec<usize>> =
            std::env::var("G0_P").ok().map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect());
        let mut csv: Box<dyn Write> = match std::env::var("G0_OUT") {
            Ok(p) => Box::new(std::fs::File::create(p).unwrap()),
            Err(_) => Box::new(std::io::sink()),
        };
        writeln!(csv, "epoch_ms,case,a8,cand,rows,round,sec_per_dispatch,bytes_per_dispatch").unwrap();
        let has_nax = be.has_metal4();
        eprintln!("metal4/NAX: {has_nax}; rounds {rounds}; pool {pool_mib} MiB; weights offset {off}");

        for c in CASES.iter().filter(|c| want.is_empty() || want.iter().any(|w| w == c.label)) {
            if a8 && c.kind == Kind::Logits {
                continue;
            }
            // wait for an idle GPU (ioreg), as gemv_b3
            let mut util = gpu_util();
            for _ in 0..20 {
                if util.is_none_or(|u| u <= 10) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_secs(1));
                util = gpu_util();
            }
            let pool = Pool::new(&be, c, pool_mib << 20, off);
            let wbytes = (c.n0 + c.n1) * c.k * 2;
            let count = pool.w0.len() * 2;
            eprintln!(
                "\n== {} {:?} K={} N={}+{} pool {} x {:.1} MiB, chain {} dispatches, GPU util at start {:?}%",
                c.label,
                c.kind,
                c.k,
                c.n0,
                c.n1,
                pool.w0.len(),
                wbytes as f64 / 1048576.0,
                count,
                util
            );

            let mut cands = vec![Cand::Serving];
            let mr_rows: Vec<usize> = [1usize, 2, 3, 4, 5, 6, 7, 8, 16]
                .into_iter()
                .filter(|r| rows_filter.as_ref().is_none_or(|f| f.contains(r) || *r == 1))
                .collect();
            let pers: Vec<usize> = mr::MR_PER_TG
                .iter()
                .copied()
                .filter(|p| per_filter.as_ref().is_none_or(|f| f.contains(p)))
                .collect();
            for &r in &mr_rows {
                for &p in &pers {
                    // R = 1 is the sanity instance (same kernel as serving); only the P = 1 form is timed
                    if r > 1 || p == 1 {
                        cands.push(Cand::Mr(r, p));
                    }
                }
            }
            for &(r, p, rb) in &mr::MR_SPLITS {
                if rows_filter.as_ref().is_none_or(|f| f.contains(&r)) {
                    cands.push(Cand::MrS(r, p, rb));
                }
            }
            if only_split {
                cands.retain(|c| match c {
                    Cand::Serving | Cand::MrS(..) => true,
                    Cand::Mr(r, p) => *p == 1 && (4..=8).contains(r),
                    _ => false,
                });
            }
            for m in [2usize, 3, 4, 5, 6, 7, 8, 12, 15, 16] {
                if only_split {
                    break;
                }
                if rows_filter.as_ref().is_none_or(|f| f.contains(&m) || m == 15 || m == 16) {
                    cands.push(Cand::Tiled(m));
                }
            }
            if has_nax && c.kind != Kind::Logits && !only_split {
                cands.push(Cand::Nax(16));
            }

            // bitwise check first
            let mut bit_cands: Vec<Cand> = Vec::new();
            for &r in &mr_rows {
                for &p in &mr::MR_PER_TG {
                    bit_cands.push(Cand::Mr(r, p));
                }
            }
            bit_cands.extend(mr::MR_SPLITS.iter().map(|&(r, p, rb)| Cand::MrS(r, p, rb)));
            for bc in bit_cands {
                let r = bc.rows();
                let (ok, n) = bit_check(&be, c, a8, &pool, r, bc);
                eprintln!(
                    "  bitwise {} vs {r} serving 1-row launches: {} ({n} output bytes of y0 compared)",
                    bc.label(),
                    if ok { "IDENTICAL" } else { "DIFFERENT" }
                );
                writeln!(csv, "{},{},{},bitwise,{},{},{},0", epoch_ms(), c.label, a8 as u8, r, bc.label(), ok as u8)
                    .unwrap();
                assert!(ok, "multi-row output differs from the serving 1-row kernel");
            }

            // warm-up: ramp the clock and fault the pool in
            for _ in 0..6 {
                chain_seconds(&be, c, a8, Cand::Serving, &pool, count);
            }
            let mut sums: Vec<Vec<f64>> = vec![Vec::new(); cands.len()];
            for round in 0..rounds {
                for j in 0..cands.len() {
                    let ci = (j + round) % cands.len();
                    let cand = cands[ci];
                    let t = chain_seconds(&be, c, a8, cand, &pool, count);
                    let per = t / count as f64;
                    sums[ci].push(per);
                    writeln!(
                        csv,
                        "{},{},{},{},{},{},{:.9},{}",
                        epoch_ms(),
                        c.label,
                        a8 as u8,
                        cand.label(),
                        cand.rows(),
                        round,
                        per,
                        wbytes
                    )
                    .unwrap();
                }
            }
            csv.flush().unwrap();
            // quick console summary (all rounds; the clock-filtered numbers come from the analysis script)
            let med = |v: &mut Vec<f64>| bench::stats(v).median;
            let t1 = med(&mut sums[0].clone());
            for (ci, cand) in cands.iter().enumerate() {
                let m = med(&mut sums[ci].clone());
                eprintln!(
                    "  {:<9} {:>9.1} us  x{:.3} of T(1)  {:>6.1} GB/s weights",
                    cand.label(),
                    m * 1e6,
                    m / t1,
                    wbytes as f64 / m / 1e9
                );
            }
        }
    }
}
