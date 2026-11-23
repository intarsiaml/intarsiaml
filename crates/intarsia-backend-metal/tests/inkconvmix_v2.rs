#![cfg(target_os = "macos")]
//! Lane ink-prefill: ConvMix v2 (`INTARSIA_INK_CONVMIX_V2`, `ink_mix_rows_r<R>*`: R rows per threadgroup) is bitwise the
//! row kernels (`ink_mix_row_h_*`, main's default) on prefill launches: R 2 and 4, rows not a multiple of R, the in-kernel
//! ring write and the separate `ink_ring_write`, host and device positions, one / several query batches, norm and f32 mix,
//! N 0..7 stored sources, and the blend path (pieces of `set_blend_rows` rows, so t0 > 0).

mod common;
use common::*;
use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
use intarsia_backend_metal::inkctrl::{set_convmix_v2_r, CONVMIX_V2, MIX_HOIST};
use intarsia_backend_metal::inkmix::*;
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

#[derive(Clone)]
struct Ht {
    dt: DType,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl Ht {
    fn rand(dt: DType, shape: &[usize], scale: f32, rng: &mut Rng) -> Ht {
        let n: usize = shape.iter().product();
        let data = match dt {
            DType::Bf16 => (0..n).flat_map(|_| f32_to_bf16(rng.unit() * scale).to_le_bytes()).collect(),
            _ => (0..n).flat_map(|_| (rng.unit() * scale).to_le_bytes()).collect(),
        };
        Ht { dt, shape: shape.to_vec(), data }
    }
    fn pos(shape: &[usize], rng: &mut Rng) -> Ht {
        let n: usize = shape.iter().product();
        Ht { dt: DType::Bf16, shape: shape.to_vec(), data: (0..n).flat_map(|_| f32_to_bf16(1.0 + 0.5 * rng.unit()).to_le_bytes()).collect() }
    }
    fn zeros(dt: DType, shape: &[usize]) -> Ht {
        Ht { dt, shape: shape.to_vec(), data: vec![0; shape.iter().product::<usize>() * dt.size_bytes()] }
    }
    fn none() -> Ht {
        Ht::zeros(DType::Bf16, &[0])
    }
    fn up(&self, be: &MetalBackend) -> TensorRef {
        let b = be.alloc(self.data.len().max(16), MemKind::Device).unwrap();
        if !self.data.is_empty() {
            be.upload(&b, 0, &self.data, 0).unwrap();
        }
        TensorRef { buf: b, offset: 0, dtype: self.dt, shape: self.shape.clone() }
    }
}

fn down(be: &MetalBackend, t: &TensorRef) -> Vec<u8> {
    let mut v = vec![0u8; t.numel() * t.dtype.size_bytes()];
    if !v.is_empty() {
        be.download(&t.buf, t.offset, &mut v, 0).unwrap();
    }
    v
}

/// Run ConvMix on fresh copies of `ins` / `outs` with v2 off (`r` 0) or on with R = `r`; every output's bytes.
fn run(be: &MetalBackend, ins: &[Ht], outs: &[Ht], attrs: &OpAttrs, dev_pos: Option<i32>, r: usize) -> Vec<Vec<u8>> {
    MIX_HOIST.set(Some(true));
    CONVMIX_V2.set(Some(r != 0));
    set_convmix_v2_r(r);
    let it: Vec<TensorRef> = ins.iter().map(|h| h.up(be)).collect();
    let ot: Vec<TensorRef> = outs.iter().map(|h| h.up(be)).collect();
    let mut call = OpCall::new(Op::ConvMix, it, ot.clone(), attrs.clone());
    call.exact_rows = true;
    if let Some(p) = dev_pos {
        let pb = be.alloc(16, MemKind::Device).unwrap();
        be.upload(&pb, 0, &p.to_le_bytes(), 0).unwrap();
        call.pos = Some(TensorRef { buf: pb, offset: 0, dtype: DType::I32, shape: vec![1] });
    }
    call.check().unwrap();
    let n = blend_scratch(&ShapeInfo::from_call(&call));
    let b = be.alloc(n.max(16), MemKind::Device).unwrap();
    call.scratch = Some(TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![n.max(16)] });
    be.launch(CONV_MIX_BF16, &call, 0).unwrap();
    be.synchronize(0).unwrap();
    MIX_HOIST.set(None);
    CONVMIX_V2.set(None);
    set_convmix_v2_r(0);
    ot.iter().map(|t| down(be, t)).collect()
}

/// A ConvMix / SconvRes case: (inputs, outputs) for T rows of width C, K taps, ring Rr, (res, add, sum), (N, Q, norm, S).
fn conv_io(rng: &mut Rng, op: Op, t: usize, c: usize, k: usize, rr: usize, epi: (bool, bool, bool), mix: (usize, usize, bool, usize)) -> (Vec<Ht>, Vec<Ht>) {
    const G: usize = 64;
    let bf = DType::Bf16;
    let (res, add, sum) = epi;
    let (n, q, norm, s) = mix;
    let mut ins = vec![
        Ht::rand(bf, &[t, c], 1.0, rng),
        Ht::rand(bf, &[k, c], 0.5, rng),
        if res { Ht::rand(bf, &[t, c], 1.0, rng) } else { Ht::none() },
        if add { Ht::rand(bf, &[t, c], 1.0, rng) } else { Ht::none() },
    ];
    let mut outs = vec![Ht::zeros(bf, &[t, c]), Ht::rand(bf, &[rr, c], 1.0, rng), if sum { Ht::zeros(bf, &[t, c]) } else { Ht::none() }];
    if op == Op::ConvMix {
        ins.push(Ht::rand(bf, &[n, t, c], 1.5, rng));
        ins.push(Ht::rand(DType::F32, &[q, c], 0.2, rng));
        let nrows = if s > 0 { s } else { q };
        ins.push(if norm || s > 0 { Ht::pos(&[nrows, c], rng) } else { Ht::none() });
        if s > 0 {
            ins.push(Ht::rand(DType::F32, &[s, t, c], 1.0, rng));
            ins.push(Ht::rand(bf, &[s, G, 2 * c], (2.0 * c as f32).powf(-0.5), rng));
            ins.push(Ht::rand(bf, &[s, c, G], 0.25, rng));
            ins.push(Ht::rand(bf, &[s, c, G], 0.25, rng));
        } else {
            ins.extend([Ht::none(), Ht::none(), Ht::none(), Ht::none()]);
        }
        outs.push(if q == 0 {
            Ht::none()
        } else if s > 0 {
            Ht::zeros(bf, &[s, t, c])
        } else if norm {
            Ht::zeros(bf, &[q, t, c])
        } else {
            Ht::zeros(DType::F32, &[q, t, c])
        });
    }
    (ins, outs)
}

/// The v2 instances admit the model's 768 threads (D 2304, NE 3), else the launcher falls back to the row kernel and the
/// bitwise test below compares a kernel with itself.
#[test]
fn convmix_v2_instances_admit_their_rows() {
    use objc2_metal::MTLComputePipelineState;
    let be = backend();
    for r in [2usize, 4] {
        for qb in [1usize, 4] {
            for ns in 1..=8usize {
                let h = format!("ink_mix_rows_r{r}e3n{ns}q{qb}");
                let m = be.pso(&h).unwrap_or_else(|| panic!("no pipeline {h}")).maxTotalThreadsPerThreadgroup();
                eprintln!("{h}: max threads {m}");
                assert!(m >= 768, "{h}: max threads {m}, the model's rows need 768");
            }
        }
    }
}

#[test]
fn convmix_v2_is_bitwise() {
    let be = backend();
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut rng = Rng(0xc0a2);
    let (c, k) = (2304usize, 4usize);
    let mut n = 0;
    for &t in &[64usize, 67, 130, 257] {
        // (ring rows, pos0, device pos): a model-sized ring (separate ring write), one that holds the whole call (in-kernel)
        for &(rr, pos0, dev) in &[(k - 1 + 16, 0usize, false), (k - 1 + 16, 29, true), (t + k + 5, 7, false)] {
            let a = OpAttrs { pos0: if dev { 0 } else { pos0 }, eps: 1e-6, ..OpAttrs::default() };
            let dp = dev.then_some(pos0 as i32);
            for &(epi, mix) in &[
                ((true, true, true), (4usize, 1usize, true, 0usize)),
                ((true, false, false), (7, 3, true, 0)),
                ((false, false, false), (2, 1, false, 0)),
                ((true, false, false), (3, 6, true, 0)),
                ((true, false, false), (1, 6, false, 0)),
                ((true, false, true), (0, 0, false, 0)),
                ((true, true, true), (3, 1, false, 3)),
                ((true, false, false), (5, 3, false, 3)),
                ((true, true, true), (2, 1, false, 1)),
            ] {
                let (ins, outs) = conv_io(&mut rng, Op::ConvMix, t, c, k, rr, epi, mix);
                for blend in [0usize, 96] {
                    if blend > 0 && mix.3 == 0 {
                        continue;
                    }
                    set_blend_rows(blend);
                    let off = run(&be, &ins, &outs, &a, dp, 0);
                    for r in [2usize, 4] {
                        let on = run(&be, &ins, &outs, &a, dp, r);
                        assert_eq!(on, off, "ConvMix v2 R={r} T={t} Rr={rr} pos0={pos0} dev={dev} epi={epi:?} mix={mix:?} blend rows {blend}");
                        n += 1;
                    }
                    set_blend_rows(0);
                }
            }
        }
    }
    eprintln!("{n} cases: ink_mix_rows is bitwise ink_mix_row_h");
}
