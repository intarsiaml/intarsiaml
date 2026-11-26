//! Phase timestamps of the mix row kernel (lane ink-hip-mix, speed round): block 0 / thread 0 records `wall_clock64` (100 MHz ticks) at the phase
//! boundaries of `ink_row_kernel` (kernels/hip/ink_mix.hip, `INK_MARK`). Only an `INK_PROBE` build of that file has the marks (the tatooine copy
//! gets `#define INK_PROBE 1` prepended; the committed file compiles them out): `cargo test --release -p intarsia-backend-hip --test ink_mix_probe
//! -- --ignored --nocapture`. Prints the time between marks for AttnResMix and ConvMix (no blend) at T = 1, D = 2304.
#![cfg(ik_hip)]

mod common;

use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, ShapeInfo};

extern "C" {
    fn ik_ink_probe(out: *mut u64, n: i32) -> i32;
}

fn put(be: &dyn Backend, t: &HostTensor) -> TensorRef {
    let buf = be.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
    be.zero(&buf, 0).unwrap();
    if t.nbytes() > 0 {
        be.upload(&buf, 0, &t.data, 0).unwrap();
    }
    TensorRef::new(buf, t.dtype, &t.shape)
}

fn rnd(rng: &mut SplitMix64, dt: DType, shape: &[usize], s: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(dt, shape, &rng.normal_vec(n, s))
}

#[test]
#[ignore]
fn ink_row_kernel_phases() {
    let be = match intarsia_backend_hip::create(0) {
        Ok(b) => b,
        Err(e) => return common::skip("ink_row_kernel_phases", &format!("no usable HIP device ({e})")),
    };
    let mut rng = SplitMix64::new(0x9);
    let bf = DType::Bf16;
    let (d, t, kt, rr) = (2304usize, 1usize, 4usize, 20usize);
    let cases: Vec<(&str, &str, Op, Vec<HostTensor>, Vec<(DType, Vec<usize>)>)> = vec![
        (
            "attn_res_mix N6 Q1",
            "attn_res_mix_bf16",
            Op::AttnResMix,
            vec![
                rnd(&mut rng, bf, &[6, t, d], 1.5),
                rnd(&mut rng, bf, &[t, d], 1.0),
                HostTensor::zeros(bf, &[0]),
                rnd(&mut rng, DType::F32, &[1, d], 0.2),
                HostTensor::from_f32_as(bf, &[1, d], &vec![1.0; d]),
            ],
            vec![(bf, vec![1, t, d])],
        ),
        (
            "conv_mix N6 Q1 norm",
            "conv_mix_bf16",
            Op::ConvMix,
            vec![
                rnd(&mut rng, bf, &[t, d], 1.0),
                rnd(&mut rng, bf, &[kt, d], 0.5),
                rnd(&mut rng, bf, &[t, d], 1.0),
                rnd(&mut rng, bf, &[t, d], 1.0),
                rnd(&mut rng, bf, &[6, t, d], 1.5),
                rnd(&mut rng, DType::F32, &[1, d], 0.2),
                HostTensor::from_f32_as(bf, &[1, d], &vec![1.0; d]),
                HostTensor::zeros(bf, &[0]),
                HostTensor::zeros(bf, &[0]),
                HostTensor::zeros(bf, &[0]),
                HostTensor::zeros(bf, &[0]),
            ],
            vec![(bf, vec![t, d]), (bf, vec![rr, d]), (bf, vec![t, d]), (bf, vec![1, t, d])],
        ),
    ];
    for (label, kname, op, ins, outs) in cases {
        let inputs: Vec<TensorRef> = ins.iter().map(|x| put(&*be, x)).collect();
        let outputs: Vec<TensorRef> = outs.iter().map(|(dt, sh)| put(&*be, &HostTensor::zeros(*dt, sh))).collect();
        let attrs = OpAttrs { eps: 1e-6, ..OpAttrs::default() };
        let call = OpCall::new(op, inputs, outputs, attrs);
        call.check().unwrap();
        let k = be.kernels().into_iter().find(|k| k.name == kname && k.op == op).unwrap();
        assert!(k.accepts(&ShapeInfo::from_call(&call)));
        let mut all: Vec<[u64; 8]> = Vec::new();
        for _ in 0..60 {
            be.launch(kname, &call, 0).unwrap();
            be.synchronize(0).unwrap();
            let mut p = [0u64; 8];
            // SAFETY: reads 8 u64 of the kernel library's probe array (INK_PROBE builds; else the symbol is missing and this does not link)
            assert_eq!(unsafe { ik_ink_probe(p.as_mut_ptr(), 8) }, 0);
            all.push(p);
        }
        // median of each interval (ticks of 10 ns), marks: 0 entry, 1 conv done, 2 before fin, 3 after fin, 4 mix done, 5 before norm fin, 6 after, 7 stores done
        let names = ["entry->conv", "conv->xs+sums", "fin1", "mix math", "norm sums", "fin2", "stores"];
        let mut line = String::new();
        let mut total = 0.0;
        for (i, name) in names.iter().enumerate() {
            let mut v: Vec<f64> = all.iter().skip(10).filter(|p| p[i + 1] >= p[i] && p[i + 1] > 0 && p[i] > 0).map(|p| (p[i + 1] - p[i]) as f64 * 10.0).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let m = v.get(v.len() / 2).copied().unwrap_or(f64::NAN);
            line += &format!(" {name} {m:.0} ns;");
            if m.is_finite() {
                total += m;
            }
        }
        println!("PROBE {label}:{line} total {total:.0} ns");
    }
}
