//! Full-checkpoint check of the lossless P12 format (lane ink-pack12): pack and decode every bf16 tensor the Inkling-2
//! decode step reads densely (every in-projection part and wo of every layer, the unembed's unpadded rows), assert the
//! round trip is bit-exact, and print the escape (side-list) statistics per tensor as TSV.
//!
//!   cargo run --release -p intarsia-core --example p12_check -- <checkpoint dir> [--hash-only]
//!
//! `--hash-only` prints only the FNV-1a 64 of each tensor's bytes (to show two copies of the checkpoint hold the same weights).

use intarsia_core::quant::p12;
use std::os::unix::fs::FileExt;

fn fnv(b: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).expect("usage: p12_check <checkpoint dir> [--hash-only]");
    let hash_only = args.iter().any(|a| a == "--hash-only");
    let path = format!("{dir}/model-00001-of-00001.safetensors");
    let f = std::fs::File::open(&path).expect("open checkpoint");
    let mut n8 = [0u8; 8];
    f.read_exact_at(&mut n8, 0).unwrap();
    let n = u64::from_le_bytes(n8) as usize;
    let mut hb = vec![0u8; n];
    f.read_exact_at(&mut hb, 8).unwrap();
    let h: serde_json::Value = serde_json::from_slice(&hb).unwrap();
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{dir}/config.json")).unwrap()).unwrap();
    let vocab = cfg["text_config"]["unpadded_vocab_size"].as_u64().unwrap() as usize;
    let base = 8 + n as u64;
    let re_parts = ["wq_du", "wr_du", "gate_attn", "attn_sink_proj", "wk_dv", "wv_dv", "k_sconv.weight_proj", "v_sconv.weight_proj", "wo_ud"];
    let mut names: Vec<(usize, String)> = Vec::new();
    for k in h.as_object().unwrap().keys() {
        if let Some(rest) = k.strip_prefix("model.llm.layers.") {
            let (li, tail) = rest.split_once('.').unwrap();
            if let Some(p) = tail.strip_prefix("attn.") {
                if re_parts.iter().any(|r| p == format!("{r}.weight") || p == *r) {
                    names.push((li.parse().unwrap(), k.clone()));
                }
            }
        }
    }
    names.sort();
    let mut list: Vec<String> = names.into_iter().map(|(_, k)| k).collect();
    list.push("model.llm.unembed.weight".into());
    if hash_only {
        println!("name\tbytes\tfnv1a64");
    } else {
        println!("name\trows\tcols\tescapes\tfraction\trows_with_escapes\tmax_per_row\tbf16_bytes\tp12_bytes\tpack_ms\tdecode_ms\tfnv1a64");
    }
    let (mut tot_e, mut tot_n, mut tot_b, mut tot_p) = (0usize, 0usize, 0usize, 0usize);
    let mut worst = (0.0f64, String::new());
    for name in &list {
        let t = &h[name];
        assert_eq!(t["dtype"], "BF16", "{name}");
        let shape: Vec<usize> = t["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let rows = if name.contains("unembed") { vocab } else { shape[0] };
        let cols = shape[1];
        let off = t["data_offsets"][0].as_u64().unwrap();
        let mut b = vec![0u8; rows * cols * 2];
        f.read_exact_at(&mut b, base + off).unwrap();
        let hash = fnv(&b);
        if hash_only {
            println!("{name}\t{}\t{hash:016x}", b.len());
            continue;
        }
        let w: Vec<u16> = b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        drop(b);
        let t0 = std::time::Instant::now();
        let p = p12::pack(&w, rows, cols, 0).expect("pack");
        let pack_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        let back = p12::decode(&p.codes, &p.aux, rows, cols).expect("decode");
        let decode_ms = t1.elapsed().as_secs_f64() * 1e3;
        assert!(back == w, "{name}: round trip is not bit-exact");
        let per_row: Vec<usize> = (0..rows).map(|r| p.row_escapes(r)).collect();
        let e = p.escapes();
        let frac = e as f64 / (rows * cols) as f64;
        if frac > worst.0 {
            worst = (frac, name.clone());
        }
        tot_e += e;
        tot_n += rows * cols;
        tot_b += rows * cols * 2;
        tot_p += p.bytes();
        println!(
            "{name}\t{rows}\t{cols}\t{e}\t{frac:.3e}\t{}\t{}\t{}\t{}\t{pack_ms:.1}\t{decode_ms:.1}\t{hash:016x}",
            per_row.iter().filter(|&&c| c > 0).count(),
            per_row.iter().max().unwrap(),
            rows * cols * 2,
            p.bytes()
        );
    }
    if !hash_only {
        println!(
            "TOTAL\t{} tensors\t{tot_n} elements\t{tot_e} escapes\t{:.3e}\tworst {:.3e} ({})\tbf16 {:.4} GB\tp12 {:.4} GB ({:.4} bits/element)\tALL BIT-EXACT",
            list.len(),
            tot_e as f64 / tot_n as f64,
            worst.0,
            worst.1,
            tot_b as f64 / 1e9,
            tot_p as f64 / 1e9,
            8.0 * tot_p as f64 / tot_n as f64
        );
    }
}
