//! A panic in the worker marks the engine failed instead of leaving requests hanging.
mod common;
use common::*;
use intarsia_core::model::LoadOptions;
use intarsia_core::{DType, Engine, EngineMeta, GenEvent, GenRequest, LocalEngine, Telemetry};
use std::sync::Arc;

#[test]
fn worker_panic_fails_the_engine() {
    std::env::set_var("INTARSIA_DEBUG_PANIC", "1");
    let t = tiny_weights(DType::Bf16, 1);
    let tel = Arc::new(Telemetry::new());
    let m = intarsia_core::model::Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe(),
        &dims(),
        &t.weights,
        &LoadOptions { max_seq: 64, ..Default::default() },
        tel.clone(),
    )
    .unwrap();
    let e = LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny".into(), ..Default::default() }, tel)
        .unwrap();
    assert!(e.failed().is_none());
    let req = GenRequest { prompt: vec![65, 66, 67], max_tokens: 4, ..Default::default() };
    let rx = e.submit(req.clone()).unwrap();
    match rx.recv().unwrap() {
        GenEvent::Error { message } => assert!(message.contains("panicked"), "{message}"),
        other => panic!("expected an error, got {other:?}"),
    }
    assert!(e.failed().unwrap().contains("INTARSIA_DEBUG_PANIC"));
    assert!(e.submit(req).is_err(), "a failed engine accepts no work");
    assert!(e.forward_logits(&[65]).is_err());
    let t = e.telemetry();
    assert_eq!((t.running, t.waiting), (0, 0), "counters stay consistent");
    std::env::remove_var("INTARSIA_DEBUG_PANIC");
}
