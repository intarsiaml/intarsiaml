//! Plan resolution at load: missing kernels fail the load with a list, admission on the plan's
//! exact shapes (cached, rejecting bad kernels), `--reference`, scratch workspace, plan JSON,
//! teacher-forced decode.
mod common;
use common::*;
use intarsia_core::backend::{Backend, Buffer, DeviceInfo, MemKind, StreamId};
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::plan::PlanOptions;
use intarsia_core::{DType, Engine, EngineMeta, Error, KernelImpl, LocalEngine, Op, OpCall, Result, Telemetry};
use std::sync::{Arc, Mutex};

/// The cpu backend with surgical changes: drop an op's kernels, declare scratch for an op, or
/// corrupt one op's output.
struct Wrap {
    inner: Arc<dyn Backend>,
    drop_op: Option<Op>,
    scratch_op: Option<Op>,
    corrupt_op: Option<Op>,
    no_ref: bool,
    /// Adds a row-exact (verify-row) `Matmul` kernel taking at most this many rows, as a backend whose multi-row kernel stops there.
    rows_cap: Option<usize>,
    /// Adds two row-exact `Matmul` kernels split as the CUDA MXFP4 table is (2..=4 rows, then 5 and up), both running the cpu matmul.
    split_rows: bool,
    scratch_seen: Mutex<Vec<usize>>,
}

fn wrap(drop_op: Option<Op>, scratch_op: Option<Op>, corrupt_op: Option<Op>) -> Arc<Wrap> {
    Arc::new(Wrap {
        inner: intarsia_backend_cpu::CpuBackend::arc(),
        drop_op,
        scratch_op,
        corrupt_op,
        no_ref: false,
        rows_cap: None,
        split_rows: false,
        scratch_seen: Mutex::new(vec![]),
    })
}

fn scratch_of(s: &intarsia_core::ShapeInfo) -> usize {
    1000 + 16 * s.m
}

impl Backend for Wrap {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn info(&self) -> DeviceInfo {
        self.inner.info()
    }
    fn alloc(&self, bytes: usize, kind: MemKind) -> Result<Buffer> {
        self.inner.alloc(bytes, kind)
    }
    fn upload(&self, dst: &Buffer, o: usize, src: &[u8], s: StreamId) -> Result<()> {
        self.inner.upload(dst, o, src, s)
    }
    fn download(&self, src: &Buffer, o: usize, dst: &mut [u8], s: StreamId) -> Result<()> {
        self.inner.download(src, o, dst, s)
    }
    fn copy(&self, d: &Buffer, d_o: usize, s: &Buffer, s_o: usize, n: usize, st: StreamId) -> Result<()> {
        self.inner.copy(d, d_o, s, s_o, n, st)
    }
    fn kernels(&self) -> Vec<KernelImpl> {
        let mut ks: Vec<KernelImpl> = self
            .inner
            .kernels()
            .into_iter()
            .filter(|k| Some(k.op) != self.drop_op && !(self.no_ref && k.is_reference))
            .map(|k| if Some(k.op) == self.scratch_op { k.scratch(scratch_of) } else { k })
            .collect();
        if let Some(c) = self.rows_cap {
            let mut k = KernelImpl::new(Op::Matmul, "test_mm_rows", "cpu").rows_exact();
            k.when = if c <= 5 { |s| s.m <= 5 } else { |s| s.m <= 6 };
            ks.push(k);
        }
        if self.split_rows {
            let mut lo = KernelImpl::new(Op::Matmul, "test_mm_lo_rows", "cpu").rows_exact();
            lo.when = |s| s.m <= 4;
            let mut hi = KernelImpl::new(Op::Matmul, "test_mm_hi_rows", "cpu").rows_exact();
            hi.when = |s| s.m >= 5;
            ks.extend([lo, hi]);
        }
        ks
    }
    fn launch(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Result<()> {
        if Some(call.op) == self.scratch_op {
            let s = call.scratch.as_ref().expect("kernel declared scratch but the caller passed none");
            assert!(s.nbytes() >= scratch_of(&intarsia_core::ShapeInfo::from_call(call)));
            assert_eq!(s.offset % 16, 0);
            self.scratch_seen.lock().unwrap().push(s.nbytes());
        }
        let kernel = if kernel.starts_with("test_mm_") { "cpu_f32_matmul" } else { kernel };
        self.inner.launch(kernel, call, stream)?;
        if Some(call.op) == self.corrupt_op && !kernel.starts_with("ref_") {
            let out = &call.outputs[0];
            let mut b = vec![0u8; out.nbytes()];
            self.inner.download(&out.buf, out.offset, &mut b, 0)?;
            for x in b.iter_mut().step_by(7) {
                *x ^= 0x40;
            }
            self.inner.upload(&out.buf, out.offset, &b, 0)?;
        }
        Ok(())
    }
    fn synchronize(&self, s: StreamId) -> Result<()> {
        self.inner.synchronize(s)
    }
}

fn load_on(backend: Arc<dyn Backend>, opts: LoadOptions) -> Result<Model> {
    let t = tiny_weights(DType::Bf16, 11);
    Model::load(backend, &recipe(), &dims(), &t.weights, &opts, Arc::new(Telemetry::new()))
}

fn tmp_cache(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("intarsia-plan-test-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

#[test]
fn missing_kernel_fails_the_load_with_a_list() {
    let w = wrap(Some(Op::AttnDecode), None, None);
    let e = load_on(w, LoadOptions { max_seq: 64, ..Default::default() }).err().expect("load must fail");
    let m = e.to_string();
    assert!(matches!(e, Error::NoKernel(_)), "{m}");
    assert!(m.contains("attn_decode") && m.contains("no kernel"), "{m}");
    assert!(m.contains("Decode t=1"), "should say where it is needed: {m}");
}

#[test]
fn plan_json_follows_the_contract() {
    let m =
        load_on(intarsia_backend_cpu::CpuBackend::arc(), LoadOptions { max_seq: 64, ..Default::default() }).unwrap();
    let tel = Arc::new(Telemetry::new());
    let e = LocalEngine::new(
        m,
        Arc::new(ByteCodec),
        EngineMeta { id: "tiny".into(), eos_ids: vec![299], ..Default::default() },
        tel,
    )
    .unwrap();
    let p = e.plan();
    assert_eq!(p["model"], "tiny");
    assert_eq!(p["backend"], "cpu");
    assert!(p["device"]["name"].is_string() && p["device"]["mem_total"].as_u64().unwrap() > 0);
    assert_eq!(p["exactness"], "deterministic");
    let layers = p["layers"].as_array().unwrap();
    assert_eq!(layers.len(), LAYERS);
    assert_eq!(layers[0]["kind"], "attention");
    assert_eq!(layers[0]["blocks"], serde_json::json!(["attention", "dense_ffn"]));
    let ops = p["ops"].as_array().unwrap();
    let find = |name: &str| ops.iter().find(|o| o["op"] == name).unwrap_or_else(|| panic!("no op {name}"));
    // per decode step and layer: q, o and down are single matmuls; gate + up and k + v are fused `MatmulPair`s (the cpu
    // backend registers the optional fused ops too)
    assert_eq!(find("matmul")["calls_per_decode_step"], 3 * LAYERS);
    assert_eq!(find("matmul_pair")["calls_per_decode_step"], 2 * LAYERS);
    assert_eq!(find("attn_decode")["calls_per_decode_step"], LAYERS);
    assert_eq!(find("attn_prefill")["calls_per_decode_step"], 0, "prefill-only kernel");
    assert_eq!(find("matmul")["kernel"], "cpu_f32_matmul");
    for o in ops {
        for k in ["op", "kernel", "backend", "exactness", "admitted", "calls_per_decode_step"] {
            assert!(!o[k].is_null(), "{o}");
        }
    }
    // telemetry op names are the same strings
    assert!(e.telemetry().ops.keys().all(|k| Op::parse(k).is_some()));
}

#[test]
fn admission_on_plan_shapes_is_cached_and_enforced() {
    let dir = tmp_cache("adm");
    let opts = PlanOptions { admit: true, require_admitted: true, cache_dir: Some(dir.clone()), ..Default::default() };
    let oracle = intarsia_backend_cpu::CpuBackend::arc();
    let mut m =
        load_on(intarsia_backend_cpu::CpuBackend::arc(), LoadOptions { max_seq: 64, ..Default::default() }).unwrap();
    let s1 = m.configure_plan(&opts, Some(&*oracle)).unwrap();
    assert!(s1.rejected.is_empty(), "{:?}", s1.rejected);
    assert!(s1.cases_run > 10 && s1.cases_cached == 0, "{s1:?}");
    assert!(m.registry.require_admitted);
    assert!(m.resolved.ops.iter().all(|o| o.admitted), "{:?}", m.resolved.ops);
    // the model still runs with only admitted kernels selectable
    let mut st = m.new_state().unwrap();
    m.forward(&mut st, &[1, 2, 3, 4, 5], intarsia_core::model::HeadMode::LastLogits, None).unwrap();

    // a second load is served from the cache
    let mut m2 =
        load_on(intarsia_backend_cpu::CpuBackend::arc(), LoadOptions { max_seq: 64, ..Default::default() }).unwrap();
    let s2 = m2.configure_plan(&opts, Some(&*oracle)).unwrap();
    assert_eq!(s2.cases_run, 0, "{s2:?}");
    assert!(s2.cases_cached > 10);
    assert!(std::fs::read_dir(&dir).unwrap().count() == 1);

    // require_admitted without admission leaves nothing selectable: a load error, not a runtime one
    let mut m3 =
        load_on(intarsia_backend_cpu::CpuBackend::arc(), LoadOptions { max_seq: 64, ..Default::default() }).unwrap();
    let e = m3
        .configure_plan(&PlanOptions { require_admitted: true, ..Default::default() }, Some(&*oracle))
        .err()
        .expect("must fail");
    assert!(e.to_string().contains("admitted kernels only"), "{e}");
    let _ = std::fs::remove_dir_all(dir);
}

/// Lane mtt: `PlanOptions::lazy_cases` answers cached cases without building their random data (a warm load builds none),
/// and changes nothing else: a cold run writes the same cache file, and after a partial cache loss the cases that run again
/// get the data the eager loop gives them (the skipped draws are replayed), so the file comes back byte for byte.
#[test]
fn lazy_admission_cases_build_nothing_when_cached_and_keep_the_data_of_those_that_run() {
    let oracle = intarsia_backend_cpu::CpuBackend::arc();
    let run = |dir: &std::path::Path, lazy: bool| {
        let opts = PlanOptions {
            admit: true,
            require_admitted: true,
            cache_dir: Some(dir.to_path_buf()),
            lazy_cases: Some(lazy),
            ..Default::default()
        };
        let mut m =
            load_on(intarsia_backend_cpu::CpuBackend::arc(), LoadOptions { max_seq: 64, ..Default::default() }).unwrap();
        let s = m.configure_plan(&opts, Some(&*oracle)).unwrap();
        assert!(s.rejected.is_empty(), "{:?}", s.rejected);
        let kernels: Vec<(String, bool)> = m.resolved.ops.iter().map(|o| (o.kernel.clone(), o.admitted)).collect();
        (s, kernels)
    };
    let file = |dir: &std::path::Path| {
        let mut f: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).collect();
        assert_eq!(f.len(), 1, "{f:?}");
        f.pop().unwrap()
    };
    let (eager, lazy) = (tmp_cache("lazy-e"), tmp_cache("lazy-l"));

    // cold: nothing is cached, so the lazy loop is the eager loop from its first case on
    let (se, ke) = run(&eager, false);
    let (sl, kl) = run(&lazy, true);
    assert!(se.cases_run > 10 && se.cases_cached == 0 && se.cases_built > 10, "{se:?}");
    assert_eq!((sl.cases_run, sl.cases_cached, sl.cases_built), (se.cases_run, se.cases_cached, se.cases_built));
    assert_eq!(kl, ke);
    let full = std::fs::read(file(&eager)).unwrap();
    assert_eq!(std::fs::read(file(&lazy)).unwrap(), full, "a cold lazy run writes the eager cache file");

    // warm: the eager loop builds every case to find its label, the lazy one builds none
    let (se, ke) = run(&eager, false);
    let (sl, kl) = run(&lazy, true);
    assert_eq!((se.cases_run, sl.cases_run), (0, 0));
    assert!(se.cases_built > 10, "{se:?}");
    assert_eq!(sl.cases_built, 0, "{sl:?}");
    assert_eq!(sl.cases_cached, se.cases_cached);
    assert_eq!(kl, ke);

    // some results lost (not the first ones, so cached cases come before a miss): both loops run them again on the same
    // data, which shows in the measured errors the file stores
    let mut v: serde_json::Value = serde_json::from_slice(&full).unwrap();
    let entries = v["entries"].as_object_mut().unwrap();
    let keys: Vec<String> = entries.keys().cloned().collect();
    let lost: Vec<&String> = keys.iter().enumerate().filter(|(i, _)| *i >= 3 && i % 4 == 3).map(|(_, k)| k).collect();
    assert!(lost.len() >= 3, "{}", keys.len());
    for k in &lost {
        entries.remove(*k);
    }
    let partial = serde_json::to_vec_pretty(&v).unwrap();
    std::fs::write(file(&eager), &partial).unwrap();
    std::fs::write(file(&lazy), &partial).unwrap();
    let (se, ke) = run(&eager, false);
    let (sl, kl) = run(&lazy, true);
    assert_eq!(se.cases_run, lost.len(), "{se:?}");
    assert_eq!((sl.cases_run, sl.cases_cached), (se.cases_run, se.cases_cached));
    assert!(sl.cases_built >= sl.cases_run && sl.cases_built <= se.cases_built, "{sl:?} {se:?}");
    assert_eq!(kl, ke);
    let (fe, fl) = (std::fs::read(file(&eager)).unwrap(), std::fs::read(file(&lazy)).unwrap());
    assert_eq!(fl, fe, "the lazy loop re-measured a lost case on other data than the eager loop");
    let (je, jf): (serde_json::Value, serde_json::Value) = (serde_json::from_slice(&fe).unwrap(), serde_json::from_slice(&full).unwrap());
    let diff: Vec<_> = keys.iter().filter(|k| je["entries"][k.as_str()] != jf["entries"][k.as_str()]).map(|k| (k, &je["entries"][k.as_str()], &jf["entries"][k.as_str()])).collect();
    assert!(diff.is_empty(), "re-measured cases differ from the cold run's: {diff:#?}");
    let _ = std::fs::remove_dir_all(eager);
    let _ = std::fs::remove_dir_all(lazy);
}

#[test]
fn bad_kernel_is_rejected_on_plan_shapes_and_replaced() {
    let w = wrap(None, None, Some(Op::SiluGlu));
    let mut m = load_on(w, LoadOptions { max_seq: 64, ..Default::default() }).unwrap();
    let oracle = intarsia_backend_cpu::CpuBackend::arc();
    let sum = m
        .configure_plan(
            &PlanOptions { admit: true, require_admitted: true, no_cache: true, ..Default::default() },
            Some(&*oracle),
        )
        .unwrap();
    assert_eq!(sum.rejected.len(), 1, "{:?}", sum.rejected);
    assert_eq!(sum.rejected[0].0, "cpu_f32_silu_glu");
    let silu = m.resolved.ops.iter().find(|o| o.op == Op::SiluGlu).unwrap();
    assert_eq!(silu.kernel, "ref_silu_glu", "falls back to the next admitted kernel");
}

#[test]
fn reference_flag_forces_ref_kernels() {
    let mut m =
        load_on(intarsia_backend_cpu::CpuBackend::arc(), LoadOptions { max_seq: 64, ..Default::default() }).unwrap();
    assert!(m.resolved.ops.iter().all(|o| o.kernel.starts_with("cpu_f32_")));
    m.configure_plan(&PlanOptions { force_reference: true, ..Default::default() }, None).unwrap();
    assert!(m.resolved.ops.iter().all(|o| o.kernel.starts_with("ref_")), "{:?}", m.resolved.ops);
    let mut st = m.new_state().unwrap();
    m.forward(&mut st, &[5, 6, 7], intarsia_core::model::HeadMode::LastLogits, None).unwrap();

    // a backend without reference kernels cannot honour it: load error, no silent fallback
    let w = Arc::new(Wrap { no_ref: true, ..Arc::try_unwrap(wrap(None, None, None)).ok().unwrap() });
    let mut m = load_on(w, LoadOptions { max_seq: 64, ..Default::default() }).unwrap();
    let e =
        m.configure_plan(&PlanOptions { force_reference: true, ..Default::default() }, None).err().expect("must fail");
    assert!(matches!(e, Error::NoKernel(_)) && e.to_string().contains("reference kernels only"), "{e}");
}

#[test]
fn scratch_workspace_is_reserved_and_passed() {
    let w = wrap(None, Some(Op::AttnDecode), None);
    let m = load_on(w.clone(), LoadOptions { max_seq: 64, prefill_chunk: 8, ..Default::default() }).unwrap();
    assert_eq!(m.resolved.scratch_bytes, 1000 + 16);
    let op = m.resolved.ops.iter().find(|o| o.op == Op::AttnDecode).unwrap();
    assert_eq!(op.scratch_bytes, 1016);
    let mut st = m.new_state().unwrap();
    m.forward(&mut st, &[1, 2, 3, 4], intarsia_core::model::HeadMode::None, None).unwrap();
    assert!(w.scratch_seen.lock().unwrap().is_empty(), "prefill does not use the decode kernel");
    m.forward(&mut st, &[5], intarsia_core::model::HeadMode::LastLogits, None).unwrap();
    let seen = w.scratch_seen.lock().unwrap().clone();
    assert_eq!(seen.len(), LAYERS);
    assert!(seen.iter().all(|&n| n >= 1016));
}

#[test]
fn teacher_forced_decode_matches_full_forward() {
    let t = tiny_weights(DType::Bf16, 21);
    let tel = Arc::new(Telemetry::new());
    let m = Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe(),
        &dims(),
        &t.weights,
        &LoadOptions { max_seq: 64, prefill_chunk: 4, ..Default::default() },
        tel.clone(),
    )
    .unwrap();
    let e = LocalEngine::new(
        m,
        Arc::new(ByteCodec),
        EngineMeta { id: "tiny".into(), eos_ids: vec![299], ..Default::default() },
        tel,
    )
    .unwrap();
    let ids: Vec<u32> = (0..23).map(|i| (i * 17 % 250) as u32).collect();
    let full = e.forward_logits(&ids).unwrap();
    let mut positions = vec![];
    let mut rows = vec![];
    e.forward_logits_decode(&ids, 6, &mut |p, r| {
        positions.push(p);
        rows.push(r.to_vec());
        Ok(())
    })
    .unwrap();
    assert_eq!(positions, vec![17, 18, 19, 20, 21, 22]);
    for (p, r) in positions.iter().zip(&rows) {
        let (mut num, mut den) = (0f64, 0f64);
        for (a, b) in r.iter().zip(&full[*p]) {
            num += ((a - b) as f64).powi(2);
            den += (*b as f64).powi(2);
        }
        assert!((num / den).sqrt() < 2e-2, "position {p}: rel l2 {}", (num / den).sqrt());
    }
    assert!(e.forward_logits_decode(&ids, 0, &mut |_, _| Ok(())).is_err());
    assert!(e.forward_logits_decode(&ids, ids.len(), &mut |_, _| Ok(())).is_err());
}

/// Lane mx6: the plan covers the row counts of batched verify passes (`Model::enable_verify_batch`). A verify run has up to 3 rows and
/// two sequences verify together, so a pass has 4..=6 rows: the projections and the head see row counts a solo verify pass (2..=3) never
/// has. A backend whose row-exact multi-row kernel stops at 5 rows (the server's `no kernel for matmul_mxfp4 ... m=6`) is found at load,
/// and batched verify is switched off (the engine then verifies per sequence) instead of the first request failing; one that reaches 6
/// keeps it on; and a plan without batched verify is what it always was, however few rows the kernel takes.
#[test]
fn batched_verify_row_counts_are_planned_and_a_gap_switches_it_off_at_load() {
    let load_with = |cap: Option<usize>, batched: bool| {
        let w = Arc::new(Wrap { rows_cap: cap, ..Arc::try_unwrap(wrap(None, None, None)).ok().unwrap() });
        let mut m = load_on(w, LoadOptions { max_seq: 64, ..Default::default() }).unwrap();
        m.enable_verify(3).unwrap();
        m.enable_batch(2).unwrap();
        if batched {
            m.enable_verify_batch(2).unwrap();
        }
        m.configure_plan(&PlanOptions::default(), None).unwrap();
        m
    };
    // covered: the kernel takes 6 rows
    let on = load_with(Some(6), true);
    assert_eq!(on.verify_batch_seqs(), 2);
    assert_eq!(on.verify_batch_rows(), 6);
    let rec = on.dry_run(intarsia_core::plan::Phase::VerifyBatch, 6, 0).unwrap();
    assert!(rec.missing.is_empty(), "{:?}", rec.missing);
    assert!(rec.calls.iter().any(|c| c.kernel == "test_mm_rows"), "the 6-row pass takes the row-exact kernel");
    // the plan holds the pass's calls (a solo verify pass has at most 3 rows)
    let probes_on = on.resolved.probes;
    assert!(on.resolved.unique_calls.iter().any(|c| c.phase == intarsia_core::plan::Phase::VerifyBatch));
    // a gap at 6 rows: no load failure, batched verify off, and no batched-verify call left in the plan
    let gap = load_with(Some(5), true);
    assert_eq!(gap.verify_batch_seqs(), 0, "a row count without a kernel switches batched verify off");
    assert!(gap.resolved.unique_calls.iter().all(|c| c.phase != intarsia_core::plan::Phase::VerifyBatch));
    // batched verify not asked for: the plan is what it was (no batched-verify probe, whatever the kernel's reach)
    let off = load_with(Some(5), false);
    assert_eq!(off.verify_batch_seqs(), 0);
    assert!(off.resolved.unique_calls.iter().all(|c| c.phase != intarsia_core::plan::Phase::VerifyBatch));
    assert_eq!(off.resolved.probes, gap.resolved.probes, "the switched-off plan equals the one never asked for batched verify");
    assert!(probes_on > off.resolved.probes, "{probes_on} vs {}", off.resolved.probes);
    // the plan JSON differs from the off plan's only where the pass's shapes add something
    let (a, b) = (intarsia_core::plan::plan_json(&off, "m"), intarsia_core::plan::plan_json(&gap, "m"));
    assert_eq!(a, b);
}

/// Lane mx6, the failure itself: the kernel for a row count EXISTS but was never admitted. The row-exact matmul table is split as
/// CUDA's MXFP4 one is (a kernel for 2..=4 rows, another for 5 and up). Plan admission probes only the solo verify row counts (2..=3), so
/// the second kernel stays untested, `require_admitted` skips it, and a batched verify pass of two sequences x 3 rows (6 rows) fails
/// with `[admitted kernels only] [row-exact verify kernels only]`. With the batched row counts planned, admission tests that kernel
/// at 5 and 6 rows and the pass runs, and its rows are the solo verify passes' bitwise.
#[test]
fn batched_verify_runs_on_a_kernel_only_the_plan_admission_of_its_row_counts_admits() {
    let build = |batched: bool| {
        let w = Arc::new(Wrap { split_rows: true, ..Arc::try_unwrap(wrap(None, None, None)).ok().unwrap() });
        let mut m = load_on(w, LoadOptions { max_seq: 64, ..Default::default() }).unwrap();
        m.enable_verify(4).unwrap();
        if batched {
            m.enable_verify_batch(2).unwrap();
        }
        let oracle = intarsia_backend_cpu::CpuBackend::arc();
        let opts = PlanOptions { admit: true, require_admitted: true, no_cache: true, ..Default::default() };
        let sum = m.configure_plan(&opts, Some(&*oracle)).unwrap();
        assert!(sum.rejected.is_empty(), "{:?}", sum.rejected);
        m
    };
    let prompts: [&[u32]; 2] = [&[3, 9, 27, 81, 50, 7], &[90, 1, 2, 3]];
    let ids: Vec<Vec<u32>> = vec![vec![5, 17, 33, 2], vec![8, 8, 40, 1]]; // 2 sequences x 4 rows = 8 rows
    let states = |m: &Model| -> Vec<_> {
        prompts
            .iter()
            .map(|p| {
                let mut st = m.new_state().unwrap();
                m.forward(&mut st, p, intarsia_core::model::HeadMode::LastArgmax, None).unwrap();
                st
            })
            .collect()
    };
    // not planned: the 5+ row kernel was never admitted, so the pass has no kernel (what the server hit at m=6)
    let m = build(false);
    let mut st = states(&m);
    let mut refs: Vec<&mut _> = st.iter_mut().collect();
    let e = m.forward_verify_batch(&mut refs, &ids).err().expect("an 8-row pass has no admitted kernel");
    let msg = e.to_string();
    assert!(msg.contains("admitted kernels only") && msg.contains("row-exact verify kernels only"), "{msg}");
    // planned: admission tests the 5+ row kernel at the pass's row counts and the pass is each sequence's solo verify, bitwise
    let m = build(true);
    assert_eq!(m.verify_batch_seqs(), 2, "the kernel is admitted at every row count: batched verify stays on");
    let (mut a, mut b) = (states(&m), states(&m));
    let got = {
        let mut refs: Vec<&mut _> = a.iter_mut().collect();
        m.forward_verify_batch(&mut refs, &ids).unwrap()
    };
    for i in 0..2 {
        // a solo pass of 4 rows takes the 1..=4 row kernel: the batched pass's 5+ row kernel gives the same tokens and state
        assert_eq!(got[i], m.forward_verify(&mut b[i], &ids[i]).unwrap(), "sequence {i}");
        assert_eq!(a[i].len, b[i].len);
    }
}
