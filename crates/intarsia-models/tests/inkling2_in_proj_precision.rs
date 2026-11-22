//! Diagnostic (lane ink-cuda-attn, brief 27): the precision of the InkInProj kernels at the real shape on real decode inputs.
//! Env-gated, no-op without `INTARSIA_INK_PREC_MODEL` (checkpoint dir); `INTARSIA_INK_PREC_GOLDEN` (a top-k golden whose
//! `ids_0` is the prompt), `INTARSIA_INK_PREC_PREFILL` (default 32) and `INTARSIA_INK_PREC_STEPS` (default 20). CUDA only.
//!
//! Runs the real model on CUDA (default kernels: today's mma in_proj), prefill then one-token decode steps, and keeps every
//! visit's in_proj inputs (`attn_in_q / k / v`, bf16 values) for visits 0..=5 (= layers 0..=5, layer 5 full attention). For each
//! step and layer it launches the four in_proj kernels at 1 row on that layer's stacked weight (bf16 and its P12 pack) and
//! compares every output to an f64 reference over the same bf16 inputs: max |err|, RMS err, err in units of the output's bf16
//! ulp, and how many outputs are not the correctly rounded bf16 of the reference.

use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::recipe::ModelDims;
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo, Telemetry};
use intarsia_models::SafeTensors;
use intarsia_core::weights::WeightSource;
use std::path::PathBuf;
use std::sync::Arc;

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

fn golden_ids(path: &str, i: usize) -> Vec<u32> {
    let b = std::fs::read(path).unwrap();
    let hl = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let h: serde_json::Value = serde_json::from_slice(&b[8..8 + hl]).unwrap();
    let t = &h[format!("ids_{i}")];
    let (s, e) = (t["data_offsets"][0].as_u64().unwrap() as usize, t["data_offsets"][1].as_u64().unwrap() as usize);
    let raw = &b[8 + hl + s..8 + hl + e];
    match t["dtype"].as_str().unwrap() {
        "I32" => raw.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect(),
        "I64" => raw.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32).collect(),
        d => panic!("ids dtype {d}"),
    }
}

fn bf(bits: u16) -> f64 {
    f32::from_bits((bits as u32) << 16) as f64
}

/// round-to-nearest-even f64 -> bf16 (via f32)
fn to_bf16(v: f64) -> u16 {
    let x = (v as f32).to_bits();
    let r = 0x7fff + ((x >> 16) & 1);
    ((x.wrapping_add(r)) >> 16) as u16
}

fn f32_to_bf16(v: f32) -> u16 {
    to_bf16(v as f64)
}

#[derive(Default)]
struct Stat {
    n: usize,
    max_abs: f64,
    sq: f64,
    ulp_sq: f64,
    max_ulp: f64,
    not_rounded: usize,
}

impl Stat {
    fn add(&mut self, got: u16, r: f64) {
        let g = bf(got);
        let e = g - r;
        let ulp = {
            let rb = bf(to_bf16(r)).abs().max(1e-30);
            2f64.powi(rb.log2().floor() as i32 - 7)
        };
        self.n += 1;
        self.max_abs = self.max_abs.max(e.abs());
        self.sq += e * e;
        self.ulp_sq += (e / ulp) * (e / ulp);
        self.max_ulp = self.max_ulp.max((e / ulp).abs());
        if got != to_bf16(r) {
            self.not_rounded += 1;
        }
    }
    fn line(&self) -> String {
        let n = self.n.max(1) as f64;
        format!(
            "n {:8} | max|err| {:.3e} | rms err {:.3e} | rms {:.3} ulp, max {:.2} ulp | not correctly rounded {:6} ({:.3} %)",
            self.n,
            self.max_abs,
            (self.sq / n).sqrt(),
            (self.ulp_sq / n).sqrt(),
            self.max_ulp,
            self.not_rounded,
            100.0 * self.not_rounded as f64 / n
        )
    }
}

#[test]
fn inkling2_in_proj_precision() {
    let Some(model) = env("INTARSIA_INK_PREC_MODEL").map(PathBuf::from) else { return };
    let golden = env("INTARSIA_INK_PREC_GOLDEN").expect("INTARSIA_INK_PREC_GOLDEN");
    let pre: usize = env("INTARSIA_INK_PREC_PREFILL").map_or(32, |v| v.parse().unwrap());
    let steps: usize = env("INTARSIA_INK_PREC_STEPS").map_or(20, |v| v.parse().unwrap());
    let ids = golden_ids(&golden, 0);
    assert!(pre + steps <= ids.len());

    let be: Arc<dyn Backend> = intarsia_backend_cuda::create(0).unwrap();
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(model.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let src = SafeTensors::open_dir(&model).unwrap();
    let opts = LoadOptions { max_seq: 1024, prefill_chunk: 256, graphs: false, profile_every: 0, ..Default::default() };
    let m = Model::load(be.clone(), &recipe, &dims, &src, &opts, Arc::new(Telemetry::new())).unwrap();
    let mut st = m.new_state().unwrap();
    m.ink_dump(&mut st, &ids[..pre], None).unwrap();
    let d = dims.hidden;
    const LAYERS: usize = 6;
    // xs[layer][step] = [q, k, v] planes of d bf16 values
    let mut xs: Vec<Vec<Vec<u16>>> = vec![Vec::new(); LAYERS];
    for p in pre..pre + steps {
        let dump = m.ink_dump(&mut st, &ids[p..p + 1], None).unwrap();
        for (li, x) in xs.iter_mut().enumerate() {
            let mut planes = Vec::with_capacity(3 * d);
            for key in ["attn_in_q", "attn_in_k", "attn_in_v"] {
                let (shape, v) = &dump[&format!("v{li:02}.{key}")];
                assert_eq!(shape, &vec![1, d]);
                planes.extend(v.iter().map(|&f| f32_to_bf16(f)));
            }
            x.push(planes);
        }
    }
    drop(m);

    let names = src.names();
    let ks = be.kernels();
    let kern = |n: &str| ks.iter().find(|k| k.name == n).unwrap_or_else(|| panic!("no kernel {n}")).clone();
    let up = |bytes: &[u8]| {
        let b = be.alloc(bytes.len().max(16), MemKind::Device).unwrap();
        be.upload(&b, 0, bytes, 0).unwrap();
        b
    };
    let mut tot: Vec<(String, Stat)> =
        ["ink_in_proj_bf16_mma", "ink_in_proj_p12_mma", "ink_in_proj_bf16_gemv", "ink_in_proj_p12_gemv"].iter().map(|n| (n.to_string(), Stat::default())).collect();
    for li in 0..LAYERS {
        let suffix = format!(".layers.{li}.attn.wq_du.weight");
        let wq = names.iter().find(|n| n.ends_with(&suffix)).expect("layer prefix");
        let p = &wq[..wq.len() - "attn.wq_du.weight".len()];
        let local = src.info(&format!("{p}attn.k_sconv.weight_proj")).is_none();
        let mut groups: Vec<Vec<&str>> = vec![vec!["attn.wq_du.weight", "attn.wr_du.weight", "attn.gate_attn.weight", "attn.attn_sink_proj.weight"]];
        groups.push(if local { vec!["attn.wk_dv.weight"] } else { vec!["attn.wk_dv.weight", "attn.k_sconv.weight_proj"] });
        groups.push(if local { vec!["attn.wv_dv.weight"] } else { vec!["attn.wv_dv.weight", "attn.v_sconv.weight_proj"] });
        let mut stacked: Vec<u8> = Vec::new();
        let mut ns = [0usize; 3];
        for (g, parts) in groups.iter().enumerate() {
            for part in parts {
                let name = format!("{p}{part}");
                let info = src.info(&name).unwrap();
                assert_eq!(info.shape[1], d);
                ns[g] += info.shape[0];
                stacked.extend_from_slice(src.bytes(&name).unwrap());
            }
        }
        let n: usize = ns.iter().sum();
        let w16: Vec<u16> = stacked.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let wbuf = up(&stacked);
        let wt = TensorRef::new(wbuf, DType::Bf16, &[n, d]);
        let pk = intarsia_core::quant::p12::pack_bytes(&stacked, n, d, 0).unwrap().upload(&*be).unwrap();
        let mut lay: Vec<Stat> = (0..4).map(|_| Stat::default()).collect();
        for x in &xs[li] {
            let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            let xt = TensorRef::new(up(&xb), DType::Bf16, &[3, 1, d]);
            // f64 reference: segment j reads plane j
            let mut refs = Vec::with_capacity(n);
            let mut r0 = 0;
            for (j, &nj) in ns.iter().enumerate() {
                let xp = &x[j * d..(j + 1) * d];
                for r in r0..r0 + nj {
                    let wr = &w16[r * d..(r + 1) * d];
                    refs.push(wr.iter().zip(xp).map(|(&a, &b)| bf(a) * bf(b)).sum::<f64>());
                }
                r0 += nj;
            }
            for (ki, (kname, _)) in tot.iter().enumerate() {
                let outs: Vec<TensorRef> =
                    ns.iter().map(|&nj| TensorRef::new(be.alloc((nj * 2).max(16), MemKind::Device).unwrap(), DType::Bf16, &[1, nj])).collect();
                let ins = if kname.contains("p12") { vec![xt.clone(), pk.parts[0].clone(), pk.parts[1].clone()] } else { vec![xt.clone(), wt.clone()] };
                let mut c = OpCall::new(Op::InkInProj, ins, outs.clone(), OpAttrs::default());
                let k = kern(kname);
                let need = k.scratch_bytes(&ShapeInfo::from_call(&c));
                if need > 0 {
                    c.scratch = Some(TensorRef::new(be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need]));
                }
                be.launch(kname, &c, 0).unwrap();
                be.synchronize(0).unwrap();
                let mut got: Vec<u16> = Vec::with_capacity(n);
                for (o, &nj) in outs.iter().zip(&ns) {
                    let mut raw = vec![0u8; nj * 2];
                    be.download(&o.buf, 0, &mut raw, 0).unwrap();
                    got.extend(raw.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])));
                }
                for (g, r) in got.iter().zip(&refs) {
                    lay[ki].add(*g, *r);
                    tot[ki].1.add(*g, *r);
                }
            }
        }
        for (ki, (kname, _)) in tot.iter().enumerate() {
            println!("layer {li} ({}) N={n} {kname:24} {}", if local { "swa" } else { "full" }, lay[ki].line());
        }
    }
    for (kname, s) in &tot {
        println!("ALL {} steps x {LAYERS} layers {kname:24} {}", steps, s.line());
    }
}
