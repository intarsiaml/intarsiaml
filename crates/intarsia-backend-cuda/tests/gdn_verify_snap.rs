//! Lane mtp2s: `Backend::gdn_verify_rows` (`ik_gdn_verify_snap_bf16`: the conv in one launch, the delta rule in another, the state saved
//! after every row but the last) is bitwise the row-by-row speculative verify it replaces: T one-row `gdn_conv_bf16` +
//! `gdn_delta_rule_bf16_split` launches (the decode step's), each but the last followed by a copy of the whole recurrent slot into that
//! row's save buffer. Checked: the conv output rows, the delta-rule output rows, the final conv and delta states, and every save, for
//! T in 2..=8, the conv state f32 and bf16, the delta-rule state f32 and bf16 (rounded at the end of every row), four head layouts
//! (one with dv = 36, rows that do not fill a warp's group, and C not a multiple of the conv block), save buffers at a nonzero slot
//! offset. Prints one `MTP2SHASH` line (FNV-64 of everything compared).
//!
//! Needs a GPU: without one the test prints SKIPPED (and fails under CI=1 / INTARSIA_REQUIRE_GPU=1).
//!   cargo test --release -p intarsia-backend-cuda --test gdn_verify_snap -- --nocapture | grep MTP2SHASH
#![cfg(ik_cuda)]

mod common;

use intarsia_core::backend::{Backend, Buffer, MemKind};
use intarsia_core::ops::{Op, OpAttrs, OpCall};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, GdnVerifyRows, HostTensor, TensorRef};
use std::sync::Arc;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("gdn_verify_snap", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

fn rand_t(rng: &mut SplitMix64, dt: DType, shape: &[usize], scale: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(dt, shape, &rng.normal_vec(n, scale))
}

#[test]
fn verify_rows_are_bitwise_the_row_by_row_launches_and_saves() {
    let Some(b) = device() else { return };
    let mut rng = SplitMix64::new(0x4d325);
    let mut hash = 0xcbf29ce484222325u64;
    let (mut saves, mut fused) = (0usize, 0usize);
    let ks = 4usize;
    let dk = 128usize;
    for (hk, hv, dv) in [(8usize, 24usize, 128usize), (16, 32, 128), (4, 8, 128), (2, 2, 36)] {
        for (conv_dt, state_dt) in [(DType::Bf16, DType::Bf16), (DType::Bf16, DType::F32), (DType::F32, DType::Bf16), (DType::F32, DType::F32)] {
            for t in 2..=8usize {
                let c = 2 * hk * dk + hv * dv;
                let conv_bytes = c * (ks - 1) * conv_dt.size_bytes();
                let s_off = conv_bytes.div_ceil(256) * 256;
                let rec_bytes = s_off + hv * dv * dk * state_dt.size_bytes();
                let slot_off = 512usize; // where this layer's slot sits in the save buffers
                let put = |h: &HostTensor| {
                    let buf = b.alloc(h.data.len().max(16), MemKind::Device).unwrap();
                    b.zero(&buf, 0).unwrap();
                    b.upload(&buf, 0, &h.data, 0).unwrap();
                    TensorRef::new(buf, h.dtype, &h.shape)
                };
                let x = put(&rand_t(&mut rng, DType::Bf16, &[t, c], 1.0));
                let taps = put(&rand_t(&mut rng, DType::Bf16, &[c, ks], 0.5));
                let a = put(&rand_t(&mut rng, DType::Bf16, &[t, hv], 1.0));
                let bb = put(&rand_t(&mut rng, DType::Bf16, &[t, hv], 1.0));
                let a_log = put(&rand_t(&mut rng, DType::Bf16, &[hv], 0.5));
                let dt_bias = put(&rand_t(&mut rng, DType::Bf16, &[hv], 0.5));
                // a non-zero start for both states, exactly representable in the slot's dtypes (the conv state only ever holds bf16 values)
                let rnd = |dt: DType, shape: &[usize], rng: &mut SplitMix64, scale: f32| {
                    let v = HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(shape.iter().product(), scale)).to_f32();
                    HostTensor::from_f32_as(dt, shape, &v)
                };
                let cs0 = rnd(conv_dt, &[c, ks - 1], &mut rng, 1.0);
                let ss0 = rnd(state_dt, &[hv, dv, dk], &mut rng, 0.1);
                let mut rec_init = vec![0u8; rec_bytes];
                rec_init[..cs0.data.len()].copy_from_slice(&cs0.data);
                rec_init[s_off..s_off + ss0.data.len()].copy_from_slice(&ss0.data);
                let mk_rec = || {
                    let buf = b.alloc(rec_bytes, MemKind::Device).unwrap();
                    b.upload(&buf, 0, &rec_init, 0).unwrap();
                    let conv = TensorRef::new(buf.clone(), conv_dt, &[c, ks - 1]);
                    let mut st = TensorRef::new(buf.clone(), state_dt, &[hv, dv, dk]);
                    st.offset = s_off;
                    (buf, conv, st)
                };
                let mk_out = |shape: &[usize]| {
                    let n: usize = shape.iter().product();
                    let buf = b.alloc(n * 2, MemKind::Device).unwrap();
                    b.zero(&buf, 0).unwrap();
                    TensorRef::new(buf, DType::Bf16, shape)
                };
                let save_bufs = || -> Vec<Buffer> {
                    (0..t - 1)
                        .map(|_| {
                            let buf = b.alloc(slot_off + rec_bytes, MemKind::Device).unwrap();
                            b.zero(&buf, 0).unwrap();
                            buf
                        })
                        .collect()
                };
                let attrs = OpAttrs { eps: 1e-6, ..Default::default() };
                // reference: row by row, the slot copied after every row but the last
                let (rbuf, rconv, rst) = mk_rec();
                let (ry, ro) = (mk_out(&[t, c]), mk_out(&[t, hv, dv]));
                let rsave = save_bufs();
                for r in 0..t {
                    let conv_call = OpCall::new(Op::GdnConv, vec![x.rows(r, 1), taps.clone()], vec![ry.rows(r, 1), rconv.clone()], OpAttrs::default());
                    conv_call.check().unwrap();
                    b.launch("gdn_conv_bf16", &conv_call, 0).unwrap();
                    let delta_call = OpCall::new(
                        Op::GdnDeltaRule,
                        vec![ry.rows(r, 1), a.rows(r, 1), bb.rows(r, 1), a_log.clone(), dt_bias.clone()],
                        vec![ro.rows(r, 1), rst.clone()],
                        attrs.clone(),
                    );
                    delta_call.check().unwrap();
                    b.launch("gdn_delta_rule_bf16_split", &delta_call, 0).unwrap();
                    if r + 1 < t {
                        b.copy(&rsave[r], slot_off, &rbuf, 0, rec_bytes, 0).unwrap();
                    }
                }
                // the two fused launches
                let (fbuf, fconv, fst) = mk_rec();
                let (fy, fo) = (mk_out(&[t, c]), mk_out(&[t, hv, dv]));
                let fsave = save_bufs();
                let (mut snap_conv, mut snap_s) = (Vec::new(), Vec::new());
                for sb in &fsave {
                    let mut v = TensorRef::new(sb.clone(), conv_dt, &[c, ks - 1]);
                    v.offset = slot_off;
                    snap_conv.push(v);
                    let mut w = TensorRef::new(sb.clone(), state_dt, &[hv, dv, dk]);
                    w.offset = slot_off + s_off;
                    snap_s.push(w);
                }
                let call = GdnVerifyRows {
                    x: &x,
                    taps: &taps,
                    y: &fy,
                    a: &a,
                    b: &bb,
                    a_log: &a_log,
                    dt_bias: &dt_bias,
                    eps: 1e-6,
                    o: &fo,
                    conv_state: &fconv,
                    state: &fst,
                    snap_conv: &snap_conv,
                    snap_s: &snap_s,
                };
                let did = b.gdn_verify_rows(&call, 0).unwrap();
                assert!(did, "hk={hk} hv={hv} dv={dv} {conv_dt}/{state_dt} T={t}: the backend refused the call");
                fused += 1;
                b.synchronize(0).unwrap();
                let dl = |buf: &Buffer, off: usize, n: usize| {
                    let mut v = vec![0u8; n];
                    b.download(buf, off, &mut v, 0).unwrap();
                    v
                };
                let tag = format!("hk={hk} hv={hv} dv={dv} conv={conv_dt} state={state_dt} T={t}");
                let (rcy, fcy) = (dl(&ry.buf, 0, t * c * 2), dl(&fy.buf, 0, t * c * 2));
                assert!(rcy == fcy, "{tag}: conv output differs");
                let (ro_b, fo_b) = (dl(&ro.buf, 0, t * hv * dv * 2), dl(&fo.buf, 0, t * hv * dv * 2));
                assert!(ro_b == fo_b, "{tag}: delta-rule output differs");
                let (rf, ff) = (dl(&rbuf, 0, rec_bytes), dl(&fbuf, 0, rec_bytes));
                assert!(rf[..conv_bytes] == ff[..conv_bytes], "{tag}: final conv state differs");
                assert!(rf[s_off..] == ff[s_off..], "{tag}: final delta-rule state differs");
                fnv(&mut hash, &rcy);
                fnv(&mut hash, &ro_b);
                fnv(&mut hash, &rf);
                for r in 0..t - 1 {
                    let (rs, fs) = (dl(&rsave[r], slot_off, rec_bytes), dl(&fsave[r], slot_off, rec_bytes));
                    assert!(rs[..conv_bytes] == fs[..conv_bytes], "{tag}: conv state of save {r} differs");
                    assert!(rs[s_off..] == fs[s_off..], "{tag}: delta-rule state of save {r} differs");
                    fnv(&mut hash, &rs);
                    saves += 1;
                }
            }
        }
    }
    println!("MTP2SHASH {hash:016x} saves={saves} calls={fused}");
}
