#![cfg(target_os = "macos")]
//! Lane mgdnv: `Backend::gdn_verify_rows` on Metal (`gdn_conv_snap_bf16_k*` + `gdn_delta_snap_bf16_dk*`: the conv in one dispatch, the
//! delta rule in another, the state saved after every row but the last) is bitwise the row-by-row speculative verify it replaces: T
//! one-row `gdn_conv_bf16` + `gdn_delta_rule_bf16` launches (the decode step's), each but the last followed by a copy of the whole
//! recurrent slot into that row's save buffer. Checked: the conv output rows, the delta-rule output rows, the final conv and delta
//! states, and every save, for T in 2..=16 (all of them on the 35B layout, 2 / 3 / 8 / 16 on the others), taps K in {2, 4, 8}, dk in
//! {32, 64, 128, 256}, head layouts with dv rows that do not fill a threadgroup (dv = 36) and C not a multiple of the conv block, save
//! buffers at a nonzero slot offset, operands at a 2-byte offset (the leftover-element path). A negative control feeds the reference a
//! changed input and requires the comparison to see it. Prints one `MGDNVHASH` line (FNV-64 of everything compared).
//!
//!   cargo test --release -p intarsia-backend-metal --test gdn_verify_snap -- --nocapture | grep MGDNVHASH

mod common;
use common::*;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, GdnVerifyRows, HostTensor, Op, OpAttrs, OpCall};

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

/// Everything compared, as byte vectors: conv out, delta out, final slot (conv part, delta part), then the two parts of every save.
type Out = Vec<Vec<u8>>;

struct Shape {
    hk: usize,
    hv: usize,
    dv: usize,
    dk: usize,
    ks: usize,
}

fn run_case(be: &dyn Backend, sh: &Shape, t: usize, seed: u64, slot_off: usize, perturb: bool) -> (Out, Out) {
    let Shape { hk, hv, dv, dk, ks } = *sh;
    let mut rng = SplitMix64::new(seed);
    let c = 2 * hk * dk + hv * dv;
    let conv_bytes = c * (ks - 1) * 4;
    let s_off = conv_bytes.div_ceil(256) * 256;
    let rec_bytes = s_off + hv * dv * dk * 4;
    let lead = 2usize; // bytes before every bf16 operand: the leftover-element path
    let put_bf = |v: &[f32], shape: &[usize]| {
        let h = HostTensor::from_f32_as(DType::Bf16, shape, v);
        let buf = be.alloc(lead + h.data.len() + 16, MemKind::Device).unwrap();
        be.zero(&buf, 0).unwrap();
        be.upload(&buf, lead, &h.data, 0).unwrap();
        let mut t = TensorRef::new(buf, DType::Bf16, shape);
        t.offset = lead;
        t
    };
    let xclean = rng.normal_vec(t * c, 1.0);
    let mut xv = xclean.clone();
    let taps = put_bf(&rng.normal_vec(c * ks, 0.5), &[c, ks]);
    let a = put_bf(&rng.normal_vec(t * hv, 1.0), &[t, hv]);
    let bb = put_bf(&rng.normal_vec(t * hv, 1.0), &[t, hv]);
    let a_log = put_bf(&rng.normal_vec(hv, 0.5), &[hv]);
    let dt_bias = put_bf(&rng.normal_vec(hv, 0.5), &[hv]);
    // the conv state only ever holds bf16 values; the delta state is any f32
    let cs0 = HostTensor::from_f32_as(DType::Bf16, &[c, ks - 1], &rng.normal_vec(c * (ks - 1), 1.0)).to_f32();
    let ss0 = rng.normal_vec(hv * dv * dk, 0.1);
    if perturb {
        // the negative control: one input element changed in the REFERENCE run only (the caller passes the same seed)
        xv[(t - 1) * c + 1] += 0.75;
    }
    let x = put_bf(&xv, &[t, c]);
    let mut rec_init = vec![0u8; rec_bytes];
    for (i, v) in cs0.iter().enumerate() {
        rec_init[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in ss0.iter().enumerate() {
        rec_init[s_off + i * 4..s_off + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let mk_rec = || {
        let buf = be.alloc(rec_bytes, MemKind::Device).unwrap();
        be.upload(&buf, 0, &rec_init, 0).unwrap();
        let conv = TensorRef::new(buf.clone(), DType::F32, &[c, ks - 1]);
        let mut st = TensorRef::new(buf.clone(), DType::F32, &[hv, dv, dk]);
        st.offset = s_off;
        (buf, conv, st)
    };
    let mk_out = |shape: &[usize]| {
        let n: usize = shape.iter().product();
        let buf = be.alloc(lead + n * 2 + 16, MemKind::Device).unwrap();
        be.zero(&buf, 0).unwrap();
        let mut t = TensorRef::new(buf, DType::Bf16, shape);
        t.offset = lead;
        t
    };
    let save_bufs = || -> Vec<Buffer> {
        (0..t - 1)
            .map(|_| {
                let buf = be.alloc(slot_off + rec_bytes, MemKind::Device).unwrap();
                be.zero(&buf, 0).unwrap();
                buf
            })
            .collect()
    };
    let attrs = OpAttrs { eps: 1e-6, ..Default::default() };
    let dl = |buf: &Buffer, off: usize, n: usize| {
        let mut v = vec![0u8; n];
        be.download(buf, off, &mut v, 0).unwrap();
        v
    };

    // reference: row by row, the slot copied after every row but the last
    let (rbuf, rconv, rst) = mk_rec();
    let (ry, ro) = (mk_out(&[t, c]), mk_out(&[t, hv, dv]));
    let rsave = save_bufs();
    for r in 0..t {
        let conv_call = OpCall::new(Op::GdnConv, vec![x.rows(r, 1), taps.clone()], vec![ry.rows(r, 1), rconv.clone()], OpAttrs::default());
        conv_call.check().unwrap();
        be.launch("gdn_conv_bf16", &conv_call, 0).unwrap();
        let delta_call = OpCall::new(
            Op::GdnDeltaRule,
            vec![ry.rows(r, 1), a.rows(r, 1), bb.rows(r, 1), a_log.clone(), dt_bias.clone()],
            vec![ro.rows(r, 1), rst.clone()],
            attrs.clone(),
        );
        delta_call.check().unwrap();
        be.launch("gdn_delta_rule_bf16", &delta_call, 0).unwrap();
        if r + 1 < t {
            be.copy(&rsave[r], slot_off, &rbuf, 0, rec_bytes, 0).unwrap();
        }
    }

    // the fused call (always on the UNPERTURBED input: `x` above is perturbed in the reference run only, so rebuild it clean)
    let xf = if perturb {
        put_bf(&xclean, &[t, c])
    } else {
        x.clone()
    };
    let (fbuf, fconv, fst) = mk_rec();
    let (fy, fo) = (mk_out(&[t, c]), mk_out(&[t, hv, dv]));
    let fsave = save_bufs();
    let (mut snap_conv, mut snap_s) = (Vec::new(), Vec::new());
    for sb in &fsave {
        let mut v = TensorRef::new(sb.clone(), DType::F32, &[c, ks - 1]);
        v.offset = slot_off;
        snap_conv.push(v);
        let mut w = TensorRef::new(sb.clone(), DType::F32, &[hv, dv, dk]);
        w.offset = slot_off + s_off;
        snap_s.push(w);
    }
    let call = GdnVerifyRows {
        x: &xf,
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
    let did = be.gdn_verify_rows(&call, 0).unwrap();
    assert!(did, "hk={hk} hv={hv} dv={dv} dk={dk} K={ks} T={t}: the backend refused the call");
    be.synchronize(0).unwrap();

    let collect = |y: &TensorRef, o: &TensorRef, buf: &Buffer, saves: &[Buffer]| -> Out {
        let mut v = vec![dl(&y.buf, lead, t * c * 2), dl(&o.buf, lead, t * hv * dv * 2)];
        let rec = dl(buf, 0, rec_bytes);
        v.push(rec[..conv_bytes].to_vec());
        v.push(rec[s_off..].to_vec());
        for s in saves {
            let r = dl(s, slot_off, rec_bytes);
            v.push(r[..conv_bytes].to_vec());
            v.push(r[s_off..].to_vec());
        }
        v
    };
    (collect(&ry, &ro, &rbuf, &rsave), collect(&fy, &fo, &fbuf, &fsave))
}

fn diff(r: &Out, f: &Out) -> Option<usize> {
    r.iter().zip(f).position(|(a, b)| a != b)
}

#[test]
fn verify_rows_are_bitwise_the_row_by_row_launches_and_saves() {
    std::env::set_var("INTARSIA_METAL_GDN_VERIFY_FUSE", "1");
    let be = backend();
    let mut hash = 0xcbf29ce484222325u64;
    let (mut saves, mut calls) = (0usize, 0usize);
    // (hk, hv, dv, dk, K): the 35B GatedDeltaNet layout first, then odd ones
    let shapes = [
        Shape { hk: 16, hv: 32, dv: 128, dk: 128, ks: 4 },
        Shape { hk: 8, hv: 24, dv: 128, dk: 128, ks: 4 },
        Shape { hk: 2, hv: 2, dv: 36, dk: 128, ks: 4 },
        Shape { hk: 4, hv: 8, dv: 64, dk: 64, ks: 2 },
        Shape { hk: 2, hv: 4, dv: 40, dk: 32, ks: 8 },
        Shape { hk: 2, hv: 4, dv: 64, dk: 256, ks: 3 },
    ];
    for (si, sh) in shapes.iter().enumerate() {
        let rows: Vec<usize> = if si == 0 { (2..=16).collect() } else { vec![2, 3, 8, 16] };
        for t in rows {
            let slot_off = if t % 2 == 0 { 512 } else { 48 };
            let (r, f) = run_case(&be, sh, t, 0x4d47_4e56 + 31 * si as u64 + t as u64, slot_off, false);
            let tag = format!("hk={} hv={} dv={} dk={} K={} T={t}", sh.hk, sh.hv, sh.dv, sh.dk, sh.ks);
            if let Some(i) = diff(&r, &f) {
                let what = ["conv output", "delta-rule output", "final conv state", "final delta-rule state"].get(i).copied().unwrap_or("a save");
                panic!("{tag}: {what} differs (part {i})");
            }
            for part in &r {
                fnv(&mut hash, part);
            }
            saves += t - 1;
            calls += 1;
        }
    }
    // negative control: the reference fed a changed input must be told apart (outputs, final states and saves), so the compare is live
    for (si, t) in [(0usize, 3usize), (0, 8), (3, 16)] {
        let (r, f) = run_case(&be, &shapes[si], t, 0xbad5eed + t as u64, 512, true);
        let i = diff(&r, &f).expect("negative control: a changed input was not detected");
        // the changed element is in the last row: only the outputs and the final state can see it, never the saves
        assert!(i <= 3, "negative control differs in a save (part {i}), but the saves precede the changed row");
        assert!(r[4..] == f[4..], "negative control: saves taken before the changed row must still agree");
    }
    println!("MGDNVHASH {hash:016x} saves={saves} calls={calls}");
}
