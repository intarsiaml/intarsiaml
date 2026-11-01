//! Parity of the DeepSeek-V4.1 cpu reference ops (`dsv41_ref`, discriminants 44-59, 61, 62) against DeepSeek's official
//! inference code, through the committed fixtures in `golden/dsv41/` (generator: `tools/dsv41/fixtures.py`). No network, no
//! weights: every test here reads only committed files. `dsv41_weights.rs` adds the layer-level checks that need checkpoint
//! tensors (gated behind `INTARSIA_DSV41_WEIGHTS`). Each test prints `PARITY ...` rows (run with `--nocapture`); the
//! tolerances are asserted and explained next to each check.
//!
//! Tolerance policy: fp32 fixtures `<= 1e-5` of the reference's largest magnitude (the ops compute in f64 and round to f32,
//! the reference in f32); bf16 fixtures within bf16 rounding (`<= 4e-3`, one bf16 ulp is `2^-8 = 3.9e-3` of the value, so a
//! single result that rounds the other way is allowed everywhere); ops whose reference computes entirely in fp32 inputs
//! that the fixture stores exactly are held to `<= 1e-6`.
mod dsv41_common;

use dsv41_common::*;
use intarsia_core::dsv41::config::Dsv41Config;
use intarsia_core::dsv41::ops::{QdqFmt, QdqScale, ScoreFunc};
use intarsia_core::exec::OutSpec;
use intarsia_core::{DType, HostTensor, Op, OpAttrs};

fn cfg() -> Dsv41Config {
    Dsv41Config::from_json(&serde_json::from_str(include_str!("../../intarsia-models/tests/fixtures/dsv41/mia-config.json")).unwrap()).unwrap()
}

fn ops() -> St {
    St::golden("ops.safetensors")
}

const BF16_TOL: f64 = 4e-3;

#[test]
fn act_qdq_matches_official_fake_quant_bit_for_bit() {
    let s = ops();
    for (name, fmt, block, scale) in [
        ("act_quant_fp8_b32_ue8m0", QdqFmt::Fp8E4m3, 32, QdqScale::Ue8m0),
        ("fp4_quant_b32_e8m0", QdqFmt::Fp4E2m1, 32, QdqScale::Ue8m0),
        ("fp4_quant_b16_e4m3", QdqFmt::Fp4E2m1, 16, QdqScale::E4m3),
    ] {
        let (x, want) = (s.get(&format!("{name}/in")), s.get(&format!("{name}/out")));
        let mut a = OpAttrs::default();
        (a.dsv.qdq_fmt, a.dsv.qdq_block, a.dsv.qdq_scale) = (fmt, block, scale);
        let hc = call(Op::ActQdq, vec![x.tensor_as(DType::Bf16)], vec![OutSpec::new(DType::Bf16, &x.shape)], a);
        let got = go(&hc)[0].to_f32();
        let e = check(&format!("ActQdq {name}"), &got, &want.f32(), 0.0);
        assert_eq!(e.n_diff, 0, "{name}: the fake-quant is bit-exact");
    }
}

/// Sinkhorn/split of `HcMixes` through the op: the op computes `mixes = fn @ rmsnorm(x)`; with `x = M` and
/// `fn = rms(M) * I` it returns exactly the fixture's `mixes` row `M` (up to the f32 rounding of `rms(M)`, 6e-8 relative).
#[test]
fn hc_mixes_sinkhorn_matches_official_split() {
    let s = ops();
    let c = cfg();
    let (mixes, sc, base) = (s.get("sinkhorn/mixes").f32(), s.get("sinkhorn/scale").f32(), s.get("sinkhorn/base").f32());
    let (want_pre, want_post, want_comb) = (s.get("sinkhorn/pre").f32(), s.get("sinkhorn/post").f32(), s.get("sinkhorn/comb").f32());
    let (st, d) = (4usize, 6usize); // S*D = 24 = the mix width
    let mut a = OpAttrs { eps: 0.0, ..Default::default() };
    (a.dsv.iters, a.dsv.hc_eps) = (c.hc_sinkhorn_iters, c.hc_eps);
    let (mut gp, mut gq, mut gc) = (vec![], vec![], vec![]);
    for r in 0..32 {
        let m = &mixes[r * 24..(r + 1) * 24];
        let rms = (m.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / 24.0).sqrt();
        let mut f = vec![0f32; 24 * 24];
        for j in 0..24 {
            f[j * 24 + j] = rms as f32;
        }
        let hc = call(
            Op::HcMixes,
            vec![f32t(&[1, st, d], m), f32t(&[24, 24], &f), f32t(&[3], &sc), f32t(&[24], &base)],
            vec![OutSpec::new(DType::F32, &[1, st]), OutSpec::new(DType::F32, &[1, st]), OutSpec::new(DType::F32, &[1, st, st])],
            a.clone(),
        );
        let o = go(&hc);
        gp.extend(o[0].to_f32());
        gq.extend(o[1].to_f32());
        gc.extend(o[2].to_f32());
    }
    check("HcMixes pre (sigmoid + eps)", &gp, &want_pre, 1e-6);
    check("HcMixes post (2*sigmoid)", &gq, &want_post, 1e-6);
    check("HcMixes comb (softmax + 20 Sinkhorn iters)", &gc, &want_comb, 1e-6);
    // doubly stochastic after the last column normalisation (columns sum to 1; rows to 1 within the iteration tolerance)
    for r in 0..32 {
        for k in 0..4 {
            let cs: f32 = (0..4).map(|j| gc[r * 16 + j * 4 + k]).sum();
            assert!((cs - 1.0).abs() < 1e-5, "column sum {cs}");
        }
    }
}

#[test]
fn hc_pre_and_post_match_official() {
    let s = ops();
    // HcPre: x [1,4,4,5120] bf16 (T=4 tokens), pre_mix [1,4,4] f32 -> out [1,4,5120] bf16
    let (x, pm) = (s.get("hc_pre/x"), s.get("hc_pre/pre_mix"));
    let hc = call(Op::HcPre, vec![x.tensor_as(DType::Bf16).reshaped(&[4, 4, 5120]), pm.tensor_as(DType::F32).reshaped(&[4, 4])], vec![OutSpec::new(DType::Bf16, &[4, 5120])], OpAttrs::default());
    check("HcPre (bf16 out)", &go(&hc)[0].to_f32(), &s.get("hc_pre/out").f32(), BF16_TOL);
    // HcPost: x [1,4,5120], residual [1,4,4,5120], post [1,4,4], comb [1,4,4,4] -> [1,4,4,5120]
    let hc = call(
        Op::HcPost,
        vec![
            s.get("hc_post/x").tensor_as(DType::Bf16).reshaped(&[4, 5120]),
            s.get("hc_post/residual").tensor_as(DType::Bf16).reshaped(&[4, 4, 5120]),
            s.get("hc_post/post").tensor_as(DType::F32).reshaped(&[4, 4]),
            s.get("hc_post/comb").tensor_as(DType::F32).reshaped(&[4, 4, 4]),
        ],
        vec![OutSpec::new(DType::Bf16, &[4, 4, 5120])],
        OpAttrs::default(),
    );
    check("HcPost (bf16 out)", &go(&hc)[0].to_f32(), &s.get("hc_post/out").f32(), BF16_TOL);
}

/// `select_candidate_blocks`: scores built so that `IndexerBlocks` reproduces the fixture's logits exactly
/// (`Hi = 2, Di = 1, q = [1, -1], w = [1, -1]`: `relu(k) - relu(-k) = k`), one call per row (each row has its own length).
#[test]
fn indexer_blocks_matches_select_candidate_blocks() {
    let s = ops();
    let c = cfg();
    let (logits, lens, keep) = (s.get("candidates/logits"), s.get("candidates/compress_lens").i32(), s.get("candidates/keep").f32());
    let l = logits.f32();
    let width = logits.shape[1];
    for r in 0..4 {
        let len = lens[r] as usize;
        let mut a = OpAttrs { pos0: len - 1, scale: 1.0, top_k: c.candidate_topk_blocks, ..Default::default() };
        (a.dsv.ratio, a.dsv.block) = (1, c.candidate_block_size);
        let nb = len.div_ceil(a.dsv.block);
        let hc = call(
            Op::IndexerBlocks,
            vec![f32t(&[1, 2, 1], &[1.0, -1.0]), f32t(&[len, 1], &l[r * width..r * width + len]), f32t(&[1, 2], &[1.0, -1.0])],
            vec![OutSpec::new(DType::U8, &[1, nb])],
            a.clone(),
        );
        let mask = go(&hc)[0].data.clone();
        // the official mask is per position (block mask repeated); the pinned newest block may extend past `len`
        let mut diff = 0;
        for b in 0..width.div_ceil(a.dsv.block) {
            let want = keep[r * width + b * a.dsv.block] != 0.0;
            let got = b < nb && mask[b] != 0;
            diff += usize::from(want != got);
        }
        println!("PARITY IndexerBlocks row {r} len {len}: {nb} blocks, kept {}, differing blocks {diff}", mask.iter().filter(|m| **m != 0).count());
        assert_eq!(diff, 0, "row {r}");
    }
}

fn rope_case(name: &str, layer: usize) {
    let s = ops();
    let c = cfg();
    let inv = c.inv_freq(layer);
    let rd = 2 * inv.len();
    // 1. the cos / sin tables out to position 1,048,575: angle = pos * inv_freq in f32, as the reference's outer product
    let (pos, cos, sin) = (s.get(&format!("rope_{name}/positions")).i32(), s.get(&format!("rope_{name}/cos")).f32(), s.get(&format!("rope_{name}/sin")).f32());
    let mut worst = (0f64, 0i32);
    for (i, p) in pos.iter().enumerate() {
        for (j, f) in inv.iter().enumerate() {
            let ang = (*p as f32 * f) as f64;
            let e = (ang.cos() - cos[i * inv.len() + j] as f64).abs().max((ang.sin() - sin[i * inv.len() + j] as f64).abs());
            if e > worst.0 {
                worst = (e, *p);
            }
        }
    }
    println!("PARITY RopeTail table {name} (positions up to {}): max |cos/sin - reference| = {:.3e} at position {}", pos.iter().max().unwrap(), worst.0, worst.1);
    assert!(worst.0 < 1e-5, "{name}: angle table off by {:.3e} at position {}", worst.0, worst.1);
    // 2. the op on bf16 rows at positions 5, 6, 7 and 70000, forward then inverse (one call per row: pos0 differs)
    let (x, y, yi, px) = (s.get(&format!("rope_{name}/x")), s.get(&format!("rope_{name}/y")).f32(), s.get(&format!("rope_{name}/y_inverse")).f32(), s.get(&format!("rope_{name}/positions_x")).i32());
    let xf = x.f32(); // [1,4,2,rd]
    let (mut gy, mut gyi) = (vec![], vec![]);
    for (t, p) in px.iter().enumerate() {
        let row = &xf[t * 2 * rd..(t + 1) * 2 * rd];
        let mk = |inverse: bool| {
            let mut a = OpAttrs { pos0: *p as usize, ..Default::default() };
            a.dsv.inverse = inverse;
            a
        };
        let fwd = call(Op::RopeTail, vec![HostTensor::from_f32_as(DType::Bf16, &[1, 2, rd], row), f32t(&[inv.len()], &inv)], vec![OutSpec::new(DType::Bf16, &[1, 2, rd])], mk(false));
        let o = go(&fwd)[0].clone();
        gy.extend(o.to_f32());
        let inv_call = call(Op::RopeTail, vec![o, f32t(&[inv.len()], &inv)], vec![OutSpec::new(DType::Bf16, &[1, 2, rd])], mk(true));
        gyi.extend(go(&inv_call)[0].to_f32());
    }
    check(&format!("RopeTail {name} forward (pos 5,6,7,70000)"), &gy, &y, BF16_TOL);
    check(&format!("RopeTail {name} inverse (round trip)"), &gyi, &yi, BF16_TOL);
}

#[test]
fn rope_plain_matches_official() {
    rope_case("plain", 0);
}

#[test]
fn rope_compressed_yarn_matches_official_far_out() {
    rope_case("compressed_yarn", 2);
}

#[test]
fn moe_route_matches_official_gate() {
    let s = St::golden("router_L2_logits.safetensors");
    let l2 = St::golden("layers_bf16_ci.safetensors");
    let c = cfg();
    let (logits, bias, bias_vl) = (s.get("logits"), s.get("bias").f32(), s.get("bias_vl").f32());
    let mask = s.get("image_mask").f32();
    let (t, e) = (logits.shape[0], logits.shape[1]);
    let a = OpAttrs { top_k: c.top_k, scale: c.route_scale, norm_topk: c.norm_topk, ..Default::default() };
    assert_eq!(c.score_func, ScoreFunc::SqrtSoftplus);
    let route = |b: &[f32], rows: &[usize]| {
        let mut l = vec![];
        for r in rows {
            l.extend_from_slice(&logits.f32()[r * e..(r + 1) * e]);
        }
        let hc = call(Op::MoeRoute, vec![f32t(&[rows.len(), e], &l), f32t(&[e], b)], vec![OutSpec::new(DType::F32, &[rows.len(), c.top_k]), OutSpec::new(DType::I32, &[rows.len(), c.top_k])], a.clone());
        let o = go(&hc);
        (o[0].to_f32(), o[1].to_i32().unwrap())
    };
    // text path (`bias`, the layers fixture's taps): expert set per row and weights
    let all: Vec<usize> = (0..t).collect();
    let (w, ids) = route(&bias, &all);
    let (want_w, want_ids) = (l2.get("bf16/L2/s0/ffn.gate_w").f32(), l2.get("bf16/L2/s0/ffn.gate_idx").i32());
    set_equal_rows("MoeRoute bias ids", &ids, &want_ids, c.top_k);
    check("MoeRoute bias weights", &sorted_by_id(&w, &ids, c.top_k), &sorted_by_id(&want_w, &want_ids, c.top_k), 1e-5);
    // vision-language path: image rows use `bias_vl`
    let (wv, iv) = route(&bias_vl, &all);
    let (want_wv, want_iv) = (s.get("weights_vl").f32(), s.get("indices_vl").i32());
    let img: Vec<usize> = (0..t).filter(|r| mask[*r] != 0.0).collect();
    let pick = |v: &[f32]| img.iter().flat_map(|r| v[r * c.top_k..(r + 1) * c.top_k].to_vec()).collect::<Vec<f32>>();
    let pick_i = |v: &[i32]| img.iter().flat_map(|r| v[r * c.top_k..(r + 1) * c.top_k].to_vec()).collect::<Vec<i32>>();
    set_equal_rows("MoeRoute bias_vl ids (image rows)", &pick_i(&iv), &pick_i(&want_iv), c.top_k);
    check("MoeRoute bias_vl weights (image rows)", &sorted_by_id(&pick(&wv), &pick_i(&iv), c.top_k), &sorted_by_id(&pick(&want_wv), &pick_i(&want_iv), c.top_k), 1e-5);
    // the vl bias genuinely changes the routing of some rows (else the test would not tell the two biases apart)
    let n_diff = img.iter().filter(|r| { let (a, b) = (&ids[**r * c.top_k..(**r + 1) * c.top_k], &iv[**r * c.top_k..(**r + 1) * c.top_k]); let (mut a, mut b) = (a.to_vec(), b.to_vec()); a.sort(); b.sort(); a != b }).count();
    println!("PARITY MoeRoute: {n_diff} of {} image rows route differently under bias_vl", img.len());
    assert!(n_diff > 0);
}

fn sorted_by_id(w: &[f32], ids: &[i32], k: usize) -> Vec<f32> {
    let mut out = vec![];
    for r in 0..ids.len() / k {
        let mut p: Vec<(i32, f32)> = (0..k).map(|j| (ids[r * k + j], w[r * k + j])).collect();
        p.sort_by_key(|x| x.0);
        out.extend(p.into_iter().map(|x| x.1));
    }
    out
}

fn set_equal_rows(what: &str, got: &[i32], want: &[i32], k: usize) {
    let mut bad = 0;
    for r in 0..got.len() / k {
        let (mut a, mut b) = (got[r * k..(r + 1) * k].to_vec(), want[r * k..(r + 1) * k].to_vec());
        a.sort();
        b.sort();
        bad += usize::from(a != b);
    }
    println!("PARITY {what:<46} rows={} differing expert sets={bad}", got.len() / k);
    assert_eq!(bad, 0, "{what}");
}

/// One query row of `sparse_attn_4k_*`: the official kernel's `kv` is the union of the window rows and the selected
/// compressed rows, `idxs` index it (`-1` = none). The op splits them: the window rows go to `win` (row `W - (P - p)` for
/// position `p`, query at `P`), the selected compressed rows to `cmp` (row `c` = compressed row `c`), `idx` lists their `c`.
fn sparse_row(s: &St, r: usize, dt: DType) -> (Vec<f32>, Vec<f32>) {
    let (rows, idx, wp, cp) = (s.get("rows").i32(), s.get("idxs").i32(), s.get("window_positions").i32(), s.get("compressed_positions").i32());
    let (kv, q, sink) = (s.get("kv").f32(), s.get("q"), s.get("sink").f32());
    let (w, d, h) = (128usize, 512usize, 64usize);
    let (kc, tw) = (s.get("compressed_positions").shape[1], s.get("idxs").shape[1]);
    let pos = rows[r] as usize;
    let mut win = vec![0f32; (w + 1) * d];
    for j in 0..w {
        if wp[r * w + j] >= 0 {
            let p = wp[r * w + j] as usize;
            let k = idx[r * tw + j] as usize;
            win[(w - (pos - p)) * d..(w - (pos - p) + 1) * d].copy_from_slice(&kv[k * d..(k + 1) * d]);
        } else {
            assert_eq!(idx[r * tw + j], -1);
        }
    }
    let nc = cp[r * kc..(r + 1) * kc].iter().cloned().max().unwrap().max(-1) + 1;
    let mut cmp = vec![0f32; (nc.max(1) as usize) * d];
    let mut ix = vec![-1i32; kc];
    for j in 0..kc {
        let c = cp[r * kc + j];
        if c >= 0 {
            let k = idx[r * tw + w + j] as usize;
            cmp[c as usize * d..(c as usize + 1) * d].copy_from_slice(&kv[k * d..(k + 1) * d]);
            ix[j] = c;
        }
    }
    let a = OpAttrs { window: w, pos0: pos, scale: (d as f32).powf(-0.5), ..Default::default() };
    let hc = call(
        Op::AttnSparse,
        vec![
            HostTensor::from_f32_as(dt, &[1, h, d], &q.f32()[r * h * d..(r + 1) * h * d]),
            HostTensor::from_f32_as(dt, &[w + 1, d], &win),
            HostTensor::from_f32_as(dt, &[nc.max(1) as usize, d], &cmp),
            HostTensor::from_i32(&[1, kc], &ix),
            f32t(&[h], &sink),
        ],
        vec![OutSpec::new(dt, &[1, h, d])],
        a,
    );
    (go(&hc)[0].to_f32(), s.get("o").f32()[r * h * d..(r + 1) * h * d].to_vec())
}

fn sparse_4k(file: &str, dt: DType, tol: f64) {
    let s = St::golden(file);
    let (mut got, mut want) = (vec![], vec![]);
    for r in 0..s.get("rows").numel() {
        let (g, w) = sparse_row(&s, r, dt);
        check(&format!("AttnSparse {file} row {r} (pos {})", s.get("rows").i32()[r]), &g, &w, tol);
        got.extend(g);
        want.extend(w);
    }
    check(&format!("AttnSparse {file} all rows"), &got, &want, tol);
}

/// Real layer-2 data: window 128 + up to 512 selected compressed rows, 64 heads, head_dim 512, attention sink; rows at
/// positions 0 (a single key), 1, 2 (window not full), 127/128 (window edge), 129, 1023 .. 4094.
/// bf16 tolerance: the official kernel rounds the softmax probabilities to bf16 before `P @ V` (`kernel.py:380`), a
/// relative error of up to 2^-9 per weight that averages out over the keys; the final bf16 rounding adds one ulp.
#[test]
fn attn_sparse_matches_official_on_real_bf16_data() {
    sparse_4k("sparse_attn_4k_bf16.safetensors", DType::Bf16, 8e-3);
}

#[test]
fn attn_sparse_matches_official_on_real_fp32_data() {
    sparse_4k("sparse_attn_4k_fp32.safetensors", DType::F32, 1e-5);
}

/// The synthetic `sparse_attn` fixture: `-1` indices, a 70-wide index list crossing the kernel's 64-key block, and an
/// all-masked row that must be exactly zero. The op always attends the query's own window key; to compare against the
/// official call (`kv` rows chosen only by `idxs`) the window key is made unreachable: `win[r+1] = -1e6 * u_r` with
/// `u_r` pointing against every head's query, so its score is below -1e3 and its softmax weight underflows to 0.
#[test]
fn attn_sparse_matches_official_synthetic_with_masked_rows() {
    let s = ops();
    let (q, kv, sink, idx, want) = (s.get("sparse_attn/q"), s.get("sparse_attn/kv"), s.get("sparse_attn/sink").f32(), s.get("sparse_attn/idxs"), s.get("sparse_attn/out").f32());
    let (t, h, d, n) = (4usize, 64usize, 512usize, 40usize);
    let qf = q.f32();
    let mut win = vec![0f32; (1 + t) * d];
    for r in 0..t {
        let mut u = vec![0f64; d];
        for hh in 0..h {
            for e in 0..d {
                u[e] -= qf[(r * h + hh) * d + e] as f64;
            }
        }
        let nrm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        for e in 0..d {
            win[(r + 1) * d + e] = (1e5 * u[e] / nrm) as f32;
        }
        for hh in 0..h {
            let dot: f64 = (0..d).map(|e| qf[(r * h + hh) * d + e] as f64 * win[(r + 1) * d + e] as f64).sum();
            assert!(dot < -1e4, "window key not suppressed: q.win = {dot}");
        }
    }
    let a = OpAttrs { window: 1, pos0: 0, scale: (d as f32).powf(-0.5), ..Default::default() };
    let hc = call(
        Op::AttnSparse,
        vec![q.tensor_as(DType::Bf16).reshaped(&[t, h, d]), HostTensor::from_f32_as(DType::Bf16, &[1 + t, d], &win), kv.tensor_as(DType::Bf16).reshaped(&[n, d]), idx.tensor_as(DType::I32).reshaped(&[t, 70]), f32t(&[h], &sink)],
        vec![OutSpec::new(DType::Bf16, &[t, h, d])],
        a,
    );
    let got = go(&hc)[0].to_f32();
    for r in 0..t {
        check(&format!("AttnSparse synthetic row {r}"), &got[r * h * d..(r + 1) * h * d], &want[r * h * d..(r + 1) * h * d], 8e-3);
    }
    assert!(want[h * d..2 * h * d].iter().all(|v| *v == 0.0), "fixture row 1 is the all-masked row");
    assert!(got[h * d..2 * h * d].iter().all(|v| v.abs() < 1e-20), "all-masked row -> 0");
}

fn hc_pre(x: &Raw, mix: &Raw, dt: DType) -> Vec<f32> {
    let (t, s, d) = (x.shape[x.shape.len() - 3], x.shape[x.shape.len() - 2], x.shape[x.shape.len() - 1]);
    let hc = call(Op::HcPre, vec![x.tensor_as(dt).reshaped(&[t, s, d]), mix.tensor_as(DType::F32).reshaped(&[t, s])], vec![OutSpec::new(dt, &[t, d])], OpAttrs::default());
    go(&hc)[0].to_f32()
}

fn hc_post(sub: &Raw, res: &Raw, post: &Raw, comb: &Raw, dt: DType) -> Vec<f32> {
    let (t, s, d) = (res.shape[res.shape.len() - 3], res.shape[res.shape.len() - 2], res.shape[res.shape.len() - 1]);
    let hc = call(
        Op::HcPost,
        vec![sub.tensor_as(dt).reshaped(&[t, d]), res.tensor_as(dt).reshaped(&[t, s, d]), post.tensor_as(DType::F32).reshaped(&[t, s]), comb.tensor_as(DType::F32).reshaped(&[t, s, s])],
        vec![OutSpec::new(dt, &[t, s, d])],
        OpAttrs::default(),
    );
    go(&hc)[0].to_f32()
}

/// Layer 2 of the real model, bf16 flow, prefill rows (s0, 16 tokens) and two decode steps (s1, s2): the hyper-connection
/// glue around each sub-block on real activations. Wiring (`Block.forward`): the attention input is `HcPre(in.h, in.pre_mix)`
/// (the previous sub-block's pre-mix), the FFN input is `HcPre(hc_post_attn, hc_attn.pre)`, and this layer's `hc_ffn.pre`
/// leaves as `out.pre_mix`.
#[test]
fn hc_pre_post_glue_matches_layer2_taps_bf16() {
    let s = St::golden("layers_bf16_ci.safetensors");
    for st in ["s0", "s1", "s2"] {
        let g = |n: &str| s.get(&format!("bf16/L2/{st}/{n}"));
        check(&format!("L2 {st} HcPre attn input"), &hc_pre(g("in.h"), g("in.pre_mix"), DType::Bf16), &g("hc_pre_attn").f32(), BF16_TOL);
        check(&format!("L2 {st} HcPost after attention"), &hc_post(g("attn.out"), g("in.h"), g("hc_attn.post"), g("hc_attn.comb"), DType::Bf16), &g("hc_post_attn").f32(), BF16_TOL);
        check(&format!("L2 {st} HcPre ffn input"), &hc_pre(g("hc_post_attn"), g("hc_attn.pre"), DType::Bf16), &g("hc_pre_ffn").f32(), BF16_TOL);
        check(&format!("L2 {st} HcPost after ffn (layer out)"), &hc_post(g("ffn.out"), g("hc_post_attn"), g("hc_ffn.post"), g("hc_ffn.comb"), DType::Bf16), &g("out.h").f32(), BF16_TOL);
        assert_eq!(g("hc_ffn.pre").f32(), g("out.pre_mix").f32());
    }
}

/// fp32 flow (rel 1e-5): layer 2's input is layer 1's output; only the sub-block outputs are tapped, so the chain runs
/// attention-out -> HcPost -> HcPre -> (ffn.out tap) -> HcPost and is compared with the tapped layer output.
#[test]
fn hc_chain_matches_layer2_taps_fp32() {
    let s = St::golden("layers_fp32_ci.safetensors");
    let g = |n: &str| s.get(&format!("fp32/L2/s0/{n}"));
    let in_h = s.get("fp32/L1/s0/out.h");
    let post_attn = hc_post(g("attn.out"), in_h, g("hc_attn.post"), g("hc_attn.comb"), DType::F32);
    let post_attn_t = HostTensor::from_f32(&[1, 16, 4, 5120], &post_attn);
    let post_attn_raw = Raw { dtype: "F32".into(), shape: post_attn_t.shape.clone(), bytes: post_attn_t.data.clone() };
    let _ffn_in = hc_pre(&post_attn_raw, g("hc_attn.pre"), DType::F32);
    let out = hc_post(g("ffn.out"), &post_attn_raw, g("hc_ffn.post"), g("hc_ffn.comb"), DType::F32);
    check("L2 fp32 chain HcPost->HcPre->HcPost vs out.h", &out, &g("out.h").f32(), 1e-5);
}

/// The attention output's de-rotation and the compressed rows' rotation on real layer-2 data.
#[test]
fn rope_on_real_layer2_taps_bf16() {
    let s = St::golden("layers_bf16_ci.safetensors");
    let c = cfg();
    let inv = c.inv_freq(2);
    // de-rotation of the sparse attention output (decode steps at positions 16 and 17), rd = 64 of 512
    for (st, pos) in [("s1", 16usize), ("s2", 17)] {
        let o = s.get(&format!("bf16/L2/{st}/attn.o_sparse"));
        let mut a = OpAttrs { pos0: pos, ..Default::default() };
        a.dsv.inverse = true;
        let hc = call(Op::RopeTail, vec![o.tensor_as(DType::Bf16).reshaped(&[1, 64, 512]), f32t(&[32], &inv)], vec![OutSpec::new(DType::Bf16, &[1, 64, 512])], a);
        check(&format!("RopeTail inverse o_sparse -> o_unrot ({st}, pos {pos})"), &go(&hc)[0].to_f32(), &s.get(&format!("bf16/L2/{st}/attn.o_unrot")).f32(), BF16_TOL);
    }
    // compressed rows: latent_pre_rope -> RopeTail -> ActQdq (e2m1, blocks of 16, e4m3 scales over the whole 512-wide row) ->
    // latent. Row j of the prefill sits at position 2*j (pos_stride 2); the decode step's new row (s2) completes group 8 at
    // position 17 and stands for position 16. The qdq makes the result differ from the plain rotation by up to 0.18 (of 1.5),
    // so matching it bit for bit also pins the rope positions.
    let mut q = OpAttrs::default();
    (q.dsv.qdq_fmt, q.dsv.qdq_block, q.dsv.qdq_scale) = (QdqFmt::Fp4E2m1, 16, QdqScale::E4m3);
    for (st, rows, pos0, stride) in [("s0", 8usize, 0usize, 2usize), ("s2", 1, 16, 2)] {
        let pre = s.get(&format!("bf16/L2/{st}/attn.latent_pre_rope"));
        let mut a = OpAttrs { pos0, ..Default::default() };
        a.dsv.pos_stride = stride;
        let hc = call(Op::RopeTail, vec![pre.tensor_as(DType::Bf16).reshaped(&[rows, 1, 512]), f32t(&[32], &inv)], vec![OutSpec::new(DType::Bf16, &[rows, 1, 512])], a);
        let r = go(&hc)[0].clone().reshaped(&[rows, 512]);
        let hc = call(Op::ActQdq, vec![r], vec![OutSpec::new(DType::Bf16, &[rows, 512])], q.clone());
        let e = check(&format!("RopeTail+ActQdq compressed rows ({st}, stride {stride})"), &go(&hc)[0].to_f32(), &s.get(&format!("bf16/L2/{st}/attn.latent")).f32(), 0.0);
        assert_eq!(e.n_diff, 0);
    }
}

/// Layer 2's real indexer scores (`indexer.score`, bf16, 16 prefill rows x 8 compressed rows) and the compressed rows the
/// official code selected (`attn.idxs` indexes `[window ring 128 | the chunk | compressed]`: compressed row `idx - 144` in the 16-row prefill, `idx - 129` in a decode step): with fewer visible rows than
/// `index_topk` every visible row is selected; the op must reproduce the visibility rule `(pos0 + t + 1) / ratio` per row.
#[test]
fn indexer_topk_matches_layer2_selection() {
    let s = St::golden("layers_bf16_ci.safetensors");
    let (sc, ix) = (s.get("bf16/L2/s0/indexer.score").f32(), s.get("bf16/L2/s0/attn.idxs").i32());
    for t in 1..16usize {
        let nk = (t + 1) / 2;
        let got = topk_via_op(&sc[t * 8..t * 8 + nk], 512, t, 1, 2, None);
        let mut want: Vec<i32> = ix[t * 136..(t + 1) * 136].iter().filter(|v| **v >= 144).map(|v| v - 144).collect();
        want.sort();
        assert_eq!(got, want, "row {t}");
    }
    // decode steps: s1 (position 16, 8 visible rows) and s2 (position 17, 9 visible rows)
    for (st, pos, nk) in [("s1", 16usize, 8usize), ("s2", 17, 9)] {
        let sc = s.get(&format!("bf16/L2/{st}/indexer.score")).f32();
        let ix = s.get(&format!("bf16/L2/{st}/attn.idxs")).i32();
        let got = topk_via_op(&sc[..nk], 512, pos, 1, 2, None);
        let want: Vec<i32> = ix.iter().filter(|v| **v >= 129).map(|v| v - 129).collect();
        assert_eq!(got, want, "{st}");
    }
    println!("PARITY IndexerTopk: layer-2 selections (16 prefill rows + 2 decode steps) equal the official idxs");
}

// ---------------------------------------------------------------------------------------------------------------------
// Ops without a committed official tap on real data: independent naive transcriptions of the official functions
// (written from `model.py`, not from `dsv41_ref.rs`), on random data. The chunked/whole gate is in `dsv41_chunked.rs`.
// ---------------------------------------------------------------------------------------------------------------------

use intarsia_core::rng::SplitMix64;

fn rnd(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    SplitMix64::new(seed).normal_vec(n, scale)
}

/// `Compressor.forward` (ratio 2): `(kv.unflatten(1, (-1, r)) * score.unflatten(..).softmax(dim=2)).sum(dim=2)` over every
/// complete group of the whole stream, against the op run as a prefill chunk of 7 (one token left in the tail) followed by
/// decode steps and a 2-token chunk, so groups complete from the ring and from the chunk.
#[test]
fn compress_pool_matches_official_pooling_with_partial_groups() {
    let (n, d, r, rc) = (11usize, 8usize, 2usize, 6usize);
    let (kv, sc) = (rnd(1, n * d, 1.0), rnd(2, n * d, 2.0));
    let mut want = vec![];
    for g in 0..n / r {
        for c in 0..d {
            let s: Vec<f64> = (0..r).map(|j| sc[(g * r + j) * d + c] as f64).collect();
            let mx = s.iter().cloned().fold(f64::MIN, f64::max);
            let z: f64 = s.iter().map(|v| (v - mx).exp()).sum();
            want.push((0..r).map(|j| kv[(g * r + j) * d + c] as f64 * (s[j] - mx).exp() / z).sum::<f64>() as f32);
        }
    }
    let (mut kr, mut sr) = (HostTensor::zeros(DType::F32, &[rc, d]), HostTensor::zeros(DType::F32, &[rc, d]));
    let (mut got, mut pos) = (vec![], 0usize);
    for t in [7usize, 1, 1, 2] {
        let g = (pos + t) / r - pos / r;
        let mut a = OpAttrs { pos0: pos, ..Default::default() };
        a.dsv.ratio = r;
        let hc = call(
            Op::CompressPool,
            vec![f32t(&[t, d], &kv[pos * d..(pos + t) * d]), f32t(&[t, d], &sc[pos * d..(pos + t) * d])],
            vec![OutSpec::new(DType::F32, &[g, d]), OutSpec { dtype: DType::F32, shape: vec![rc, d], init: Some(kr.clone()) }, OutSpec { dtype: DType::F32, shape: vec![rc, d], init: Some(sr.clone()) }],
            a,
        );
        let o = go(&hc);
        got.extend(o[0].to_f32());
        (kr, sr) = (o[1].clone(), o[2].clone());
        pos += t;
    }
    check("CompressPool ratio 2, chunks 7,1,1,2 (partial groups)", &got, &want, 1e-6);
    // the tail ring holds the rows of position p at slot p % rc
    let last = (n - 1) * d;
    assert_eq!(&kr.to_f32()[(n - 1) % rc * d..(n - 1) % rc * d + d], &kv[last..last + d]);
}

/// `Attention._window_kv`: the window of row `t` is the `W` rows before it plus itself; positions below 0 are not valid
/// (the op leaves them out of attention, `win` rows there are unspecified). Ring rows `p % Wc`.
#[test]
fn window_kv_matches_official_window_and_ring() {
    let (n, d, w, wc) = (40usize, 4usize, 8usize, 12usize);
    let kv = rnd(3, n * d, 1.0);
    let mut ring = HostTensor::zeros(DType::F32, &[wc, d]);
    let mut pos = 0;
    for t in [5usize, 20, 1, 1, 13] {
        let a = OpAttrs { pos0: pos, window: w, ..Default::default() };
        let hc = call(Op::WindowKv, vec![f32t(&[t, d], &kv[pos * d..(pos + t) * d])], vec![OutSpec::new(DType::F32, &[w + t, d]), OutSpec { dtype: DType::F32, shape: vec![wc, d], init: Some(ring.clone()) }], a);
        let o = go(&hc);
        let win = o[0].to_f32();
        for row in 0..w + t {
            // win row `row` is position pos0 + row - W
            if pos + row >= w {
                let p = pos + row - w;
                assert_eq!(&win[row * d..(row + 1) * d], &kv[p * d..(p + 1) * d], "chunk at {pos}: win row {row} (position {p})");
            }
        }
        ring = o[1].clone();
        pos += t;
        let rf = ring.to_f32();
        for p in pos.saturating_sub(wc)..pos {
            assert_eq!(&rf[(p % wc) * d..(p % wc + 1) * d], &kv[p * d..(p + 1) * d], "ring slot of position {p}");
        }
    }
    println!("PARITY WindowKv: win rows and ring slots exact for chunks 5,20,1,1,13 (W 8, ring 12, chunk larger than the ring)");
}

/// Indexer top-k tie-breaking. `score[c] = k[c]` through `Hi = 2, Di = 1, q = [1, -1], w = [1, -1]`. Rule (documented in
/// docs/design/dsv41-ops.md): the `top_k` best rows by score, equal scores resolved to the LOWER row index, the winners
/// returned in ascending row order; the unreachable (`-inf`, `c >= visible`) fill-ins are `-1` at the end.
fn topk_via_op(scores: &[f32], top_k: usize, pos0: usize, t: usize, ratio: usize, cand: Option<(&[u8], usize)>) -> Vec<i32> {
    let nk = scores.len();
    let mut a = OpAttrs { pos0, scale: 1.0, top_k, ..Default::default() };
    a.dsv.ratio = ratio;
    let kk = top_k.min(nk);
    let cand_t = match cand {
        Some((m, b)) => {
            a.dsv.block = b;
            HostTensor::new(DType::U8, vec![t, nk.div_ceil(b)], m.to_vec()).unwrap()
        }
        None => HostTensor::zeros(DType::U8, &[0]),
    };
    let hc = call(Op::IndexerTopk, vec![f32t(&[t, 2, 1], &[1.0, -1.0].repeat(t)), f32t(&[nk, 1], scores), f32t(&[t, 2], &[1.0, -1.0].repeat(t)), cand_t], vec![OutSpec::new(DType::I32, &[t, kk])], a);
    go(&hc)[0].to_i32().unwrap()
}

fn topk_naive(scores: &[f32], top_k: usize, visible: usize) -> Vec<i32> {
    let mut ix: Vec<usize> = (0..scores.len()).collect();
    ix.sort_by(|&a, &b| {
        let (sa, sb) = (if a < visible { scores[a] } else { f32::NEG_INFINITY }, if b < visible { scores[b] } else { f32::NEG_INFINITY });
        sb.partial_cmp(&sa).unwrap().then(a.cmp(&b))
    });
    let mut top: Vec<usize> = ix[..top_k.min(scores.len())].to_vec();
    top.sort();
    top.into_iter().map(|c| if c < visible { c as i32 } else { -1 }).collect()
}

#[test]
fn indexer_topk_tie_break_is_lowest_index_and_deterministic() {
    // 300 rows whose scores take only 5 values: the 64th best is tied with dozens of rows
    let scores: Vec<f32> = rnd(9, 300, 1.0).iter().map(|v| (v * 2.0).round().clamp(-2.0, 2.0)).collect();
    let (pos0, ratio) = (599, 2); // one query at position 599 sees (599+1)/2 = 300 compressed rows
    let got = topk_via_op(&scores, 64, pos0, 1, ratio, None);
    let want = topk_naive(&scores, 64, 300);
    assert_eq!(got, want);
    // the boundary really is tied, and the lower indices of the tied value are the ones kept
    let kept: Vec<usize> = got.iter().map(|c| *c as usize).collect();
    let bound = kept.iter().map(|c| scores[*c]).fold(f32::INFINITY, f32::min);
    let tied: Vec<usize> = (0..300).filter(|c| scores[*c] == bound).collect();
    let tied_kept: Vec<usize> = tied.iter().cloned().filter(|c| kept.contains(c)).collect();
    assert!(tied.len() > tied_kept.len() && !tied_kept.is_empty(), "test data must tie at the cut: {} tied, {} kept", tied.len(), tied_kept.len());
    assert!(tied_kept.iter().zip(&tied).all(|(a, b)| a == b), "kept tied rows must be the lowest-index ones");
    assert!(got.windows(2).all(|w| w[0] < w[1]), "ascending row order");
    // deterministic: same answer on every run, and a larger T with the same row repeated gives the same row
    for _ in 0..3 {
        assert_eq!(topk_via_op(&scores, 64, pos0, 1, ratio, None), got);
    }
    // all-equal scores: the first `top_k` rows
    assert_eq!(topk_via_op(&[0.5; 40], 8, 79, 1, 2, None), (0..8).collect::<Vec<i32>>());
    println!("PARITY IndexerTopk tie-break: lowest index wins, ascending row order, repeatable ({} of 64 tied rows kept)", tied_kept.len());
}

#[test]
fn indexer_topk_unreachable_picks_are_minus_one() {
    // chunk of 8 queries from position 0, ratio 2: 4 compressed rows; query t sees (t+1)/2 of them
    let scores = [0.3f32, -1.0, 2.0, 0.1];
    let got = topk_via_op(&scores, 4, 0, 8, 2, None);
    let vis = |t: usize| (t + 1) / 2;
    for t in 0..8 {
        let want = topk_naive(&scores, 4, vis(t));
        assert_eq!(&got[t * 4..t * 4 + 4], &want[..], "query {t}");
    }
    assert_eq!(&got[0..4], &[-1, -1, -1, -1]);
    assert_eq!(&got[4 * 3..4 * 3 + 4], &[0, 1, -1, -1]);
    // candidates: rows outside the kept blocks (block 2) are -inf first
    let mask = [1u8, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0]; // [8 queries? no: T=2, 8 rows, 4 blocks] below
    let _ = mask;
    let s8: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let cm = [1u8, 0, 0, 1, /* second query */ 0, 1, 1, 0];
    let got = topk_via_op(&s8, 3, 15, 2, 2, Some((&cm, 2)));
    // query 0 (pos 15, sees 8 rows): blocks {0, 3} = rows 0,1,6,7 -> best 3: 7, 6, 1 -> ascending 1, 6, 7
    assert_eq!(&got[0..3], &[1, 6, 7]);
    // query 1 (pos 16 is beyond nk*ratio: sees 8 rows too): blocks {1, 2} = rows 2..6 -> 5, 4, 3 -> 3, 4, 5
    assert_eq!(&got[3..6], &[3, 4, 5]);
}

#[test]
fn silu_glu_clamp_matches_official_expert_activation() {
    // Expert.forward: gate = min(gate, limit), up = clamp(up, -limit, limit), silu(gate) * up (limit 0 = off)
    let (g, u) = (rnd(5, 4096, 12.0), rnd(6, 4096, 12.0));
    for limit in [10.0f32, 0.0] {
        let mut a = OpAttrs::default();
        a.dsv.limit = limit;
        let hc = call(Op::SiluGluClamp, vec![f32t(&[64, 64], &g), f32t(&[64, 64], &u)], vec![OutSpec::new(DType::F32, &[64, 64])], a);
        let want: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&g, &u)| {
                let (g, u) = if limit > 0.0 { (g.min(limit), u.clamp(-limit, limit)) } else { (g, u) };
                (g as f64 / (1.0 + (-(g as f64)).exp()) * u as f64) as f32
            })
            .collect();
        check(&format!("SiluGluClamp limit {limit}"), &go(&hc)[0].to_f32(), &want, 1e-6);
    }
}

#[test]
fn grouped_matmul_and_transpose_match_einsum() {
    // einsum("bsgd,grd->bsgr"): group g of x [T, G*K] times w[g] [N, K]^T
    let (t, g, n, k) = (3usize, 4usize, 5usize, 7usize);
    let (x, w) = (rnd(7, t * g * k, 1.0), rnd(8, g * n * k, 1.0));
    let hc = call(Op::GroupedMatmul, vec![f32t(&[t, g * k], &x), f32t(&[g, n, k], &w)], vec![OutSpec::new(DType::F32, &[t, g * n])], OpAttrs::default());
    let mut want = vec![0f32; t * g * n];
    for r in 0..t {
        for gg in 0..g {
            for nn in 0..n {
                want[r * g * n + gg * n + nn] = (0..k).map(|kk| x[r * g * k + gg * k + kk] as f64 * w[(gg * n + nn) * k + kk] as f64).sum::<f64>() as f32;
            }
        }
    }
    check("GroupedMatmul", &go(&hc)[0].to_f32(), &want, 1e-6);
    // Transpose01
    let (a, b, c) = (3usize, 4usize, 2usize);
    let v = rnd(10, a * b * c, 1.0);
    let hc = call(Op::Transpose01, vec![f32t(&[a, b, c], &v)], vec![OutSpec::new(DType::F32, &[b, a, c])], OpAttrs::default());
    let o = go(&hc)[0].to_f32();
    for i in 0..a {
        for j in 0..b {
            assert_eq!(&o[(j * a + i) * c..(j * a + i + 1) * c], &v[(i * b + j) * c..(i * b + j + 1) * c]);
        }
    }
}
