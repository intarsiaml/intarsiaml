//! Lane VRC: `gdn_conv_delta_bf16` with per-row state snapshots (`ik_gdn_conv_delta_snap_bf16`, the speculative-verify launch of
//! `Model::gdn_block`) is bitwise the row-by-row verify it replaces: T single-row launches (the decode step's), each but the last
//! followed by a copy of the whole recurrent slot into that row's save buffer. Checked: the outputs, the final conv and delta
//! states, and every snapshot, for T in 2..=16, three head layouts (value heads per key head 3, 2, 2), save buffers at a nonzero
//! slot offset. Prints one `VRCHASH` line (FNV-64 of everything compared). Seam `INTARSIA_HIP_GDN_SNAP_MUT=1..4` plants a fault in the
//! snapshot stores (1 index + 1, 2 no v history, 3 no q / k history, 4 delta state offset + 16 bytes): the test must then fail.
//!   cargo test --release -p intarsia-backend-hip --test gdn_verify_snap -- --nocapture | grep VRCHASH
#![cfg(ik_hip)]

use intarsia_core::admission::gdn_conv_delta_case;
use intarsia_core::backend::{Backend, MemKind};
use intarsia_core::ops::{Op, OpCall};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, TensorRef};

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

#[test]
fn gdn_verify_snapshots_are_bitwise_the_row_by_row_save() {
    let Ok(b) = intarsia_backend_hip::create(std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0)) else {
        assert!(std::env::var("INTARSIA_REQUIRE_GPU").is_err(), "INTARSIA_REQUIRE_GPU set but no HIP device");
        eprintln!("skipping: no HIP device");
        return;
    };
    let kernel = "gdn_conv_delta_bf16";
    let mut rng = SplitMix64::new(4242);
    let mut hash = 0xcbf29ce484222325u64;
    let mut checked = 0usize;
    for (hk, hv) in [(8usize, 24usize), (16, 32), (4, 8)] {
        for t in 2..=16usize {
            let (dk, dv, ks) = (128usize, 128usize, 4usize);
            let hc = gdn_conv_delta_case(&mut rng, t, hk, hv, dk, dv, ks, DType::Bf16);
            let c = 2 * hk * dk + hv * dv;
            let s_off = (c * (ks - 1) * 4).div_ceil(256) * 256;
            let rec_bytes = s_off + hv * dv * dk * 4;
            let slot_off = 512usize; // where this layer's slot sits in the save buffers
            let put = |bytes: &[u8], shape: &[usize], dt: DType| {
                let buf = b.alloc(bytes.len().max(16), MemKind::Device).unwrap();
                b.zero(&buf, 0).unwrap();
                b.upload(&buf, 0, bytes, 0).unwrap();
                TensorRef::new(buf, dt, shape)
            };
            let inputs: Vec<TensorRef> = hc.inputs.iter().map(|x| put(&x.data, &x.shape, x.dtype)).collect();
            let mut rec_init = vec![0u8; rec_bytes];
            let (cs, ss) = (hc.outputs[2].init.as_ref().unwrap(), hc.outputs[1].init.as_ref().unwrap());
            rec_init[..cs.data.len()].copy_from_slice(&cs.data);
            rec_init[s_off..s_off + ss.data.len()].copy_from_slice(&ss.data);
            let mk_rec = || {
                let buf = b.alloc(rec_bytes, MemKind::Device).unwrap();
                b.upload(&buf, 0, &rec_init, 0).unwrap();
                let conv = TensorRef::new(buf.clone(), DType::F32, &[c, ks - 1]);
                let mut st = TensorRef::new(buf.clone(), DType::F32, &[hv, dv, dk]);
                st.offset = s_off;
                (buf, conv, st)
            };
            let out_t = || {
                let buf = b.alloc(t * hv * dv * 2, MemKind::Device).unwrap();
                b.zero(&buf, 0).unwrap();
                TensorRef::new(buf, DType::Bf16, &[t, hv, dv])
            };
            let save_bufs = || -> Vec<_> {
                (0..t - 1)
                    .map(|_| {
                        let buf = b.alloc(slot_off + rec_bytes, MemKind::Device).unwrap();
                        b.zero(&buf, 0).unwrap();
                        buf
                    })
                    .collect()
            };
            // reference: row by row, the slot copied after every row but the last
            let (rbuf, rconv, rst) = mk_rec();
            let rout = out_t();
            let rsave = save_bufs();
            for r in 0..t {
                let ins: Vec<TensorRef> = vec![
                    inputs[0].rows(r, 1),
                    inputs[1].clone(),
                    inputs[2].rows(r, 1),
                    inputs[3].rows(r, 1),
                    inputs[4].clone(),
                    inputs[5].clone(),
                ];
                let call = OpCall::new(Op::GdnConvDelta, ins, vec![rout.rows(r, 1), rst.clone(), rconv.clone()], hc.attrs.clone());
                call.check().unwrap();
                b.launch(kernel, &call, 0).unwrap();
                if r + 1 < t {
                    b.copy(&rsave[r], slot_off, &rbuf, 0, rec_bytes, 0).unwrap();
                }
            }
            // one launch with the snapshots
            let (fbuf, fconv, fst) = mk_rec();
            let fout = out_t();
            let fsave = save_bufs();
            let mut outs = vec![fout.clone(), fst.clone(), fconv.clone()];
            for sb in &fsave {
                let mut v = TensorRef::new(sb.clone(), DType::F32, &[c, ks - 1]);
                v.offset = slot_off;
                outs.push(v);
            }
            let call = OpCall::new(Op::GdnConvDelta, inputs.clone(), outs, hc.attrs.clone());
            call.check().unwrap();
            b.launch(kernel, &call, 0).unwrap();
            b.synchronize(0).unwrap();
            let dl = |buf: &intarsia_core::backend::Buffer, off: usize, n: usize| {
                let mut v = vec![0u8; n];
                b.download(buf, off, &mut v, 0).unwrap();
                v
            };
            let (ro, fo) = (dl(&rout.buf, 0, t * hv * dv * 2), dl(&fout.buf, 0, t * hv * dv * 2));
            assert!(ro == fo, "hk={hk} hv={hv} T={t}: outputs differ");
            let (rf, ff) = (dl(&rbuf, 0, rec_bytes), dl(&fbuf, 0, rec_bytes));
            assert!(rf == ff, "hk={hk} hv={hv} T={t}: final state differs");
            fnv(&mut hash, &ro);
            fnv(&mut hash, &rf);
            for r in 0..t - 1 {
                let (rs, fs) = (dl(&rsave[r], slot_off, rec_bytes), dl(&fsave[r], slot_off, rec_bytes));
                assert!(rs == fs, "hk={hk} hv={hv} T={t}: snapshot {r} differs");
                fnv(&mut hash, &rs);
                checked += 1;
            }
        }
    }
    println!("VRCMUT {}", std::env::var("INTARSIA_HIP_GDN_SNAP_MUT").unwrap_or_default());
    println!("VRCHASH {hash:016x} snapshots={checked}");
}
