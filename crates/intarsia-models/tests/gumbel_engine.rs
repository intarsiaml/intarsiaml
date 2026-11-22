//! The device sampler (`INTARSIA_GUMBEL_SAMPLING=1`, lane ink-hostgap) end to end on the tiny Qwen3-shaped fixture, cpu
//! backend: plain-temperature requests draw their tokens with `GumbelArgmax` (`HeadMode::LastGumbel`), a fixed seed gives the
//! same stream on every run, a batched engine (B = 4 and 8) gives every request its solo stream, and with the switch off the
//! same requests take the host sampler (other streams). The switch is a process environment variable read when a model or
//! a request is set up, so this binary holds ONE test that sets it in order.
mod common;
use common::*;
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::{BatchedEngine, BatchedOptions, DType, Engine, EngineMeta, GenEvent, GenRequest, LocalEngine, Telemetry};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Duration;

const SEED: u64 = 7;

fn model(rows: usize, tel: &Arc<Telemetry>) -> Model {
    let t = tiny_weights(DType::Bf16, SEED);
    let mut m = Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe(),
        &dims(),
        &t.weights,
        &LoadOptions { max_seq: 160, prefill_chunk: 16, ..Default::default() },
        tel.clone(),
    )
    .unwrap();
    if rows > 1 {
        m.enable_batch(rows).unwrap();
    }
    m.configure_plan(&Default::default(), None).unwrap();
    m
}

fn meta() -> EngineMeta {
    EngineMeta { id: "tiny".into(), eos_ids: vec![299], ..Default::default() }
}

fn solo() -> LocalEngine {
    let tel = Arc::new(Telemetry::new());
    let m = model(1, &tel);
    LocalEngine::new(m, Arc::new(ByteCodec), meta(), tel).unwrap()
}

fn batched(rows: usize) -> BatchedEngine {
    let tel = Arc::new(Telemetry::new());
    let m = model(rows, &tel);
    let mut o = BatchedOptions::new(rows, 64);
    o.prefix_reuse = false;
    BatchedEngine::new(m, Arc::new(ByteCodec), meta(), tel, o).unwrap()
}

fn collect(rx: &Receiver<GenEvent>) -> Vec<u32> {
    let mut ids = Vec::new();
    loop {
        match rx.recv_timeout(Duration::from_secs(120)).expect("engine stalled") {
            GenEvent::Token { id, .. } => ids.push(id),
            GenEvent::Done { .. } => return ids,
            GenEvent::Error { message } => panic!("{message}"),
        }
    }
}

fn run_solo(e: &LocalEngine, req: GenRequest) -> Vec<u32> {
    // a throw-away request first so no KV prefix of the previous case is reused
    let flush = GenRequest { prompt: vec![250], max_tokens: 1, ignore_eos: true, ..Default::default() };
    collect(&e.submit(flush).unwrap());
    collect(&e.submit(req).unwrap())
}

/// Plain-temperature requests (top-k 0, top-p 1: the device sampler's rows), T 0.6..1.3, prompts of 3..90 tokens, 12..40
/// tokens each, some with `min_tokens` (host rows first, device rows after) or EOS on; and two that keep the host path
/// (top-k 40, top-p 0.9) mixed in, so a batch holds both kinds of rows.
fn cases() -> Vec<GenRequest> {
    let mut rng = intarsia_core::rng::SplitMix64::new(0xBEEF);
    (0..14)
        .map(|i| {
            let len = [3, 9, 15, 16, 17, 33, 48, 64, 90, 7, 31][i % 11];
            let mut prompt: Vec<u32> = vec![1 + i as u32];
            while prompt.len() < len {
                prompt.push(1 + (rng.next_u64() % 250) as u32);
            }
            let mut r = GenRequest {
                prompt,
                max_tokens: 12 + (i * 7) % 29,
                ignore_eos: i % 4 != 0,
                temperature: 0.6 + 0.1 * (i % 8) as f32,
                top_k: 0,
                top_p: 1.0,
                seed: Some(500 + 3 * i as u64),
                ..Default::default()
            };
            match i % 7 {
                2 => r.min_tokens = 4,
                5 => r.top_k = 40,
                6 => r.top_p = 0.9,
                _ => {}
            }
            r
        })
        .collect()
}

fn device_rows(r: &GenRequest) -> bool {
    r.top_k == 0 && r.top_p >= 1.0
}

#[test]
fn device_sampler_is_seeded_batch_equals_solo_and_off_by_default() {
    // SAFETY (edition 2021: safe): the only test in this binary, before any engine thread starts
    std::env::set_var("INTARSIA_GUMBEL_SAMPLING", "1");
    let e = solo();
    let on: Vec<Vec<u32>> = cases().into_iter().map(|r| run_solo(&e, r)).collect();
    // a fixed seed draws a fixed stream: the same requests again, on the same engine and on a fresh one
    let again: Vec<Vec<u32>> = cases().into_iter().map(|r| run_solo(&e, r)).collect();
    assert_eq!(on, again, "same engine, second run");
    let fresh = solo();
    let again: Vec<Vec<u32>> = cases().into_iter().map(|r| run_solo(&fresh, r)).collect();
    assert_eq!(on, again, "fresh engine");
    // another seed, another stream (device rows)
    let reseeded: Vec<Vec<u32>> =
        cases().into_iter().map(|r| run_solo(&e, GenRequest { seed: r.seed.map(|s| s + 1_000_000), ..r })).collect();
    let moved = cases().iter().zip(on.iter().zip(&reseeded)).filter(|(r, (a, b))| device_rows(r) && a != b).count();
    assert!(moved >= 8, "only {moved} device-sampled streams changed with the seed");
    // batched, B = 4 and 8 (all at once: more requests than slots, so they queue, overlap and finish at different times):
    // every request gets its solo stream, device rows and host rows alike
    for rows in [4usize, 8] {
        let eng = batched(rows);
        let rxs: Vec<_> = cases().into_iter().map(|r| eng.submit(r).unwrap()).collect();
        for (i, (rx, want)) in rxs.iter().zip(&on).enumerate() {
            assert_eq!(&collect(rx), want, "case {i}, B = {rows}");
        }
        let m = eng.metrics();
        assert!(m.batch_hist[2..].iter().sum::<u64>() > 0, "no batched decode pass ran: {:?}", m.batch_hist);
    }
    // switched off (the default): the device rows take the host sampler, which draws other tokens; host rows are unchanged
    std::env::set_var("INTARSIA_GUMBEL_SAMPLING", "0");
    let off_engine = solo();
    let off: Vec<Vec<u32>> = cases().into_iter().map(|r| run_solo(&off_engine, r)).collect();
    for (i, (r, (a, b))) in cases().iter().zip(on.iter().zip(&off)).enumerate() {
        if !device_rows(r) {
            assert_eq!(a, b, "case {i}: a host row must not change with the switch");
        }
    }
    let differ = cases().iter().zip(on.iter().zip(&off)).filter(|(r, (a, b))| device_rows(r) && a != b).count();
    assert!(differ >= 8, "only {differ} device-sampled streams differ from the host sampler's");
    std::env::remove_var("INTARSIA_GUMBEL_SAMPLING");
}
