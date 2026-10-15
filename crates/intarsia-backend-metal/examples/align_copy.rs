//! G0b item 4: cost of the 16-byte re-alignment remedy for the 8-mod-16 shard-1 weights. Lists the Qwen3-8B safetensors tensors whose
//! file offset is 8 mod 16 (the zero-copy mmap wrap keeps that alignment, so the GEMV falls to the 8-byte-load kernels) and
//! times copying them into freshly allocated, page-aligned (hence 16-byte aligned) device buffers, as a loader would.
//!
//!   cargo run --release -p intarsia-backend-metal --example align_copy -- [model dir]

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("align_copy needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::backend::{Backend, MemKind};
    use std::time::Instant;
    let dir = std::env::args().nth(1).unwrap_or_else(|| format!("{}/models/Qwen3-8B", std::env::var("HOME").unwrap()));
    let be = MetalBackend::new().unwrap();
    let mut shards: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    shards.sort();
    let (mut total, mut total_nonembed, mut n_t) = (0usize, 0usize, 0usize);
    let mut copy_s = 0.0f64;
    let mut alloc_s = 0.0f64;
    let mut warm_s = 0.0f64;
    for p in &shards {
        let f = std::fs::File::open(p).unwrap();
        // SAFETY: read-only mapping of a file nobody modifies during the run.
        let mm = unsafe { memmap2::Mmap::map(&f).unwrap() };
        let n = u64::from_le_bytes(mm[..8].try_into().unwrap()) as usize;
        let hdr: serde_json::Value = serde_json::from_slice(&mm[8..8 + n]).unwrap();
        let base = 8 + n;
        for (name, v) in hdr.as_object().unwrap() {
            if name == "__metadata__" {
                continue;
            }
            let (a, b) = (v["data_offsets"][0].as_u64().unwrap() as usize, v["data_offsets"][1].as_u64().unwrap() as usize);
            if (base + a) % 16 == 0 {
                continue;
            }
            assert_eq!((base + a) % 16, 8);
            let bytes = &mm[base + a..base + b];
            total += bytes.len();
            n_t += 1;
            if name != "model.embed_tokens.weight" {
                total_nonembed += bytes.len();
            } else {
                continue; // the embedding table is gathered, never streamed by a GEMV: not part of the remedy
            }
            // page the source in first (a real load reads it from disk either way; the remedy adds only the copy)
            let t = Instant::now();
            let mut x = 0u8;
            for i in (0..bytes.len()).step_by(4096) {
                x ^= bytes[i];
            }
            std::hint::black_box(x);
            warm_s += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let buf = be.alloc(bytes.len(), MemKind::Device).unwrap();
            alloc_s += t.elapsed().as_secs_f64();
            let t = Instant::now();
            be.upload(&buf, 0, bytes, 0).unwrap();
            copy_s += t.elapsed().as_secs_f64();
            std::hint::black_box(&buf);
        }
    }
    println!("tensors at 8 mod 16: {n_t}, {:.3} GB total, {:.3} GB without the embedding table (streamed by decode GEMVs)", total as f64 / 1e9, total_nonembed as f64 / 1e9);
    println!(
        "alloc {:.1} ms, copy (upload) {:.1} ms = {:.1} GB/s, source page-in {:.1} ms",
        alloc_s * 1e3,
        copy_s * 1e3,
        total_nonembed as f64 / 1e9 / copy_s,
        warm_s * 1e3
    );
}
