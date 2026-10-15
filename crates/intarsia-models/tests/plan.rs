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
    scratch_seen: Mutex<Vec<usize>>,
}

fn wrap(drop_op: Option<Op>, scratch_op: Option<Op>, corrupt_op: Option<Op>) -> Arc<Wrap> {
    Arc::new(Wrap {
        inner: intarsia_backend_cpu::CpuBackend::arc(),
        drop_op,
        scratch_op,
        corrupt_op,
        no_ref: false,
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
        self.inner
            .kernels()
            .into_iter()
            .filter(|k| Some(k.op) != self.drop_op && !(self.no_ref && k.is_reference))
            .map(|k| if Some(k.op) == self.scratch_op { k.scratch(scratch_of) } else { k })
            .collect()
    }
    fn launch(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Result<()> {
        if Some(call.op) == self.scratch_op {
            let s = call.scratch.as_ref().expect("kernel declared scratch but the caller passed none");
            assert!(s.nbytes() >= scratch_of(&intarsia_core::ShapeInfo::from_call(call)));
            assert_eq!(s.offset % 16, 0);
            self.scratch_seen.lock().unwrap().push(s.nbytes());
        }
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
    // per decode step: o and down are single matmuls in every layer. The first block has no residual to add, so its q is a
    // plain matmul and its k + v a fused `MatmulPair`; every later attention input is one `RmsNormAddMatmulQkv`, and every
    // FFN (residual add, pre-norm, gate + up, GLU) one `RmsNormAddMatmulGlu` (the cpu backend registers the optional fused
    // ops too)
    assert_eq!(find("matmul")["calls_per_decode_step"], 2 * LAYERS + 1);
    assert_eq!(find("matmul_pair")["calls_per_decode_step"], 1);
    assert_eq!(find("rmsnorm_add_matmul_qkv")["calls_per_decode_step"], LAYERS - 1);
    assert_eq!(find("rmsnorm_add_matmul_glu")["calls_per_decode_step"], LAYERS);
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

#[test]
fn bad_kernel_is_rejected_on_plan_shapes_and_replaced() {
    // (the cpu backend's fused `RmsNormAddMatmulGlu` replaces `SiluGlu` in the FFN; the residual-add norm is still planned)
    let w = wrap(None, None, Some(Op::RmsNormAdd));
    let mut m = load_on(w, LoadOptions { max_seq: 64, ..Default::default() }).unwrap();
    let oracle = intarsia_backend_cpu::CpuBackend::arc();
    let sum = m
        .configure_plan(
            &PlanOptions { admit: true, require_admitted: true, no_cache: true, ..Default::default() },
            Some(&*oracle),
        )
        .unwrap();
    assert_eq!(sum.rejected.len(), 1, "{:?}", sum.rejected);
    assert_eq!(sum.rejected[0].0, "cpu_f32_rmsnorm_add");
    let norm = m.resolved.ops.iter().find(|o| o.op == Op::RmsNormAdd).unwrap();
    assert_eq!(norm.kernel, "ref_rmsnorm_add", "falls back to the next admitted kernel");
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
