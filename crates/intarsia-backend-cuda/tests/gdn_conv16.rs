//! Lane conv16: the GatedDeltaNet conv state stored as bf16 (`Backend::gdn_conv_dtype`) is BITWISE the f32 one. The state only ever
//! holds past bf16 inputs or zeros, so on states built that way `gdn_conv_bf16` (prefill chunks and T = 1) and
//! `gdn_conv_bf16_rows` give the same `y` bytes and leave the same state values with either storage. Chunked runs carry the state
//! from one call to the next (the new state of call n is the history of call n + 1), so a rounding in the store would show.
//!
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::ops::RowTab;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, ShapeInfo};
use std::sync::Arc;

fn backend(test: &str) -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(be) => Some(be),
        Err(e) => {
            common::skip(test, &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn put(be: &Arc<dyn Backend>, t: &HostTensor) -> TensorRef {
    let buf = be.alloc(t.data.len().max(16), MemKind::Device).unwrap();
    be.zero(&buf, 0).unwrap();
    be.upload(&buf, 0, &t.data, 0).unwrap();
    TensorRef::new(buf, t.dtype, &t.shape)
}

fn down(be: &Arc<dyn Backend>, t: &TensorRef) -> Vec<u8> {
    let mut d = vec![0u8; t.nbytes()];
    be.download(&t.buf, t.offset, &mut d, 0).unwrap();
    d
}

fn launch(be: &Arc<dyn Backend>, kernel: &str, call: OpCall) {
    call.check().unwrap_or_else(|e| panic!("{kernel}: check: {e}"));
    let k = be.kernels().into_iter().find(|k| k.name == kernel && k.op == call.op).unwrap_or_else(|| panic!("{kernel} not registered"));
    assert_eq!(k.scratch_bytes(&ShapeInfo::from_call(&call)), 0);
    be.launch(kernel, &call, 0).unwrap_or_else(|e| panic!("{kernel}: {e}"));
    be.synchronize(0).unwrap();
}

/// A state tensor holding `vals` (bf16-representable) as `dtype`.
fn state(be: &Arc<dyn Backend>, dtype: DType, vals: &[f32], shape: &[usize]) -> TensorRef {
    put(be, &HostTensor::from_f32_as(dtype, shape, vals))
}

/// The state's values as f32, whichever way it is stored.
fn values(be: &Arc<dyn Backend>, t: &TensorRef) -> Vec<f32> {
    let bytes = down(be, t);
    match t.dtype {
        DType::F32 => bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        DType::Bf16 => bytes.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect(),
        d => panic!("{d}"),
    }
}

/// `n` bf16-representable values: mostly normal, a few large and tiny, and some exact zeros (a fresh state).
fn bf16_vals(rng: &mut SplitMix64, n: usize, scale: f32) -> Vec<f32> {
    let v: Vec<f32> = (0..n)
        .map(|i| {
            let v = rng.normal_vec(1, 1.0)[0] * scale;
            match i % 17 {
                0 => v * 64.0,
                5 => v * 1e-3,
                9 => 0.0,
                _ => v,
            }
        })
        .collect();
    HostTensor::from_f32_as(DType::Bf16, &[n], &v).to_f32()
}

fn same_bits(what: &str, got: &[u8], want: &[u8]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!("{what}: first differing byte {i} of {} (bf16 state {:#04x} vs f32 state {:#04x})", got.len(), got[i], want[i]);
    }
}

fn same_values(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a.to_bits() != b.to_bits()) {
        panic!("{what}: first differing value {i} of {} (bf16 state {} vs f32 state {})", got.len(), got[i], want[i]);
    }
}

#[test]
fn bf16_conv_state_is_the_f32_conv_state() {
    let Some(be) = backend("gdn_conv16") else { return };
    let mut rng = SplitMix64::new(16);
    let mut calls = 0;
    for (c, ks) in [(8192usize, 4usize), (6144, 4), (130, 3), (48, 2)] {
        let w = put(&be, &HostTensor::from_f32_as(DType::Bf16, &[c, ks], &rng.normal_vec(c * ks, 0.5)));
        // a fresh (all zero) state, then one that holds earlier inputs; chunk lengths cover T = 1, T < K - 1, tiles of 32
        for fresh in [true, false] {
            let init = if fresh { vec![0.0; c * (ks - 1)] } else { bf16_vals(&mut rng, c * (ks - 1), 1.0) };
            let s32 = state(&be, DType::F32, &init, &[c, ks - 1]);
            let s16 = state(&be, DType::Bf16, &init, &[c, ks - 1]);
            for t in [1usize, 2, 1, 7, 33, 1, 100, 64, 1] {
                let x = put(&be, &HostTensor::from_f32_as(DType::Bf16, &[t, c], &bf16_vals(&mut rng, t * c, 1.0)));
                let (y32, y16) = (put(&be, &HostTensor::zeros(DType::Bf16, &[t, c])), put(&be, &HostTensor::zeros(DType::Bf16, &[t, c])));
                launch(&be, "gdn_conv_bf16", OpCall::new(Op::GdnConv, vec![x.clone(), w.clone()], vec![y32.clone(), s32.clone()], OpAttrs::default()));
                launch(&be, "gdn_conv_bf16", OpCall::new(Op::GdnConv, vec![x, w.clone()], vec![y16.clone(), s16.clone()], OpAttrs::default()));
                same_bits(&format!("conv c={c} k={ks} fresh={fresh} t={t} y"), &down(&be, &y16), &down(&be, &y32));
                same_values(&format!("conv c={c} k={ks} fresh={fresh} t={t} state"), &values(&be, &s16), &values(&be, &s32));
                calls += 1;
            }
        }
    }
    println!("conv16 gdn_conv_bf16: bf16 state == f32 state over {calls} chunked calls (y bytes and state values)");
}

#[test]
fn bf16_conv_state_rows_are_the_f32_solo_conv() {
    let Some(be) = backend("gdn_conv16_rows") else { return };
    let mut rng = SplitMix64::new(17);
    let mut calls = 0;
    for (c, ks) in [(8192usize, 4usize), (130, 3)] {
        for m in [1usize, 2, 5, 8, 16] {
            let w = put(&be, &HostTensor::from_f32_as(DType::Bf16, &[c, ks], &rng.normal_vec(c * ks, 0.5)));
            let inits: Vec<Vec<f32>> = (0..m).map(|r| if r == 0 { vec![0.0; c * (ks - 1)] } else { bf16_vals(&mut rng, c * (ks - 1), 1.0) }).collect();
            let s32: Vec<TensorRef> = inits.iter().map(|v| state(&be, DType::F32, v, &[c, ks - 1])).collect();
            let s16: Vec<TensorRef> = inits.iter().map(|v| state(&be, DType::Bf16, v, &[c, ks - 1])).collect();
            let mut bytes = vec![0u8; m * 2 * 8];
            for (r, s) in s16.iter().enumerate() {
                let a = be.device_address(&s.buf).expect("device address");
                for j in 0..2 {
                    bytes[(r * 2 + j) * 8..(r * 2 + j) * 8 + 8].copy_from_slice(&a.to_le_bytes());
                }
            }
            let ptrs = put(&be, &HostTensor::new(DType::U8, vec![m, 2, 8], bytes).unwrap());
            // several steps: each step's new state is the next step's history
            for step in 0..6 {
                let xv = bf16_vals(&mut rng, m * c, 1.0);
                let x = put(&be, &HostTensor::from_f32_as(DType::Bf16, &[m, c], &xv));
                let mut want_y = Vec::new();
                for (r, s) in s32.iter().enumerate() {
                    let y = put(&be, &HostTensor::zeros(DType::Bf16, &[1, c]));
                    launch(&be, "gdn_conv_bf16", OpCall::new(Op::GdnConv, vec![x.rows(r, 1), w.clone()], vec![y.clone(), s.clone()], OpAttrs::default()));
                    want_y.extend(down(&be, &y));
                }
                let y = put(&be, &HostTensor::zeros(DType::Bf16, &[m, c]));
                let mut call = OpCall::new(Op::GdnConv, vec![x, w.clone()], vec![y.clone(), s16[0].clone()], OpAttrs::default());
                call.rows = Some(RowTab { ptrs: ptrs.clone(), rows: m, stride: 2, index: 1, byte_off: 0 });
                launch(&be, "gdn_conv_bf16_rows", call);
                same_bits(&format!("rows c={c} k={ks} m={m} step={step} y"), &down(&be, &y), &want_y);
                for r in 0..m {
                    same_values(&format!("rows c={c} k={ks} m={m} step={step} state {r}"), &values(&be, &s16[r]), &values(&be, &s32[r]));
                }
                calls += 1;
            }
        }
    }
    println!("conv16 gdn_conv_bf16_rows: bf16 state rows == the f32 solo conv over {calls} steps (y bytes and state values)");
}
