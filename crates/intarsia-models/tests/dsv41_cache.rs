//! Per-rank weight cache (`intarsia_models::dsv41_cache`): a tiny synthetic EXL3 checkpoint is cut for w=1 and w=2 (both ranks)
//! and the cache is compared with `exl3::shard_trellis` / scale slicing of the source, reassembled along the split axis,
//! the renamed `wo_a` slices, the routed experts against a direct `exl3_moe::load`, and loaded back on the CPU backend.
//! `INTARSIA_DSV41_CACHE=<dir>` points the ignored test at a real cache (loader throughput on the CPU backend).

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::quant::{exl3, exl3_moe};
use intarsia_core::rng::SplitMix64;
use intarsia_core::weights::WeightSource;
use intarsia_core::{DType, HostTensor};
use intarsia_models::dsv41::HeaderShapes;
use intarsia_models::dsv41_cache::{build_rank_cache, cache_matches, load_rank_cache, open_manifest, BuildOpts};
use intarsia_models::safetensors::{write_safetensors, SafeTensors};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const H: usize = 256;
const I: usize = 256;
const MUL1: u32 = 0x83DC_D12D;
const MCG: u32 = 0xCBAC_1FED;

fn tmp(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("dsv41-cache-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct Ckpt {
    t: BTreeMap<String, HostTensor>,
    rng: SplitMix64,
}

impl Ckpt {
    fn put(&mut self, n: &str, dtype: DType, shape: &[usize], data: Vec<u8>) {
        assert_eq!(data.len(), shape.iter().product::<usize>() * dtype.size_bytes(), "{n}");
        self.t.insert(n.to_string(), HostTensor { dtype, shape: shape.to_vec(), data });
    }
    fn rand(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.rng.next_u64() as u8).collect()
    }
    /// An EXL3 linear [K, N]; `cb`: 0 3inst, 1 mcg, 2 mul1; `packed`: i16 `su` / `sv` sign words instead of f16 `suh` / `svh`.
    fn lin(&mut self, p: &str, k: usize, n: usize, k2: usize, cb: u32, packed: bool) {
        let tr = self.rand((k / 16) * (n / 16) * 16 * k2);
        self.put(&format!("{p}.trellis"), DType::I16, &[k / 16, n / 16, 8 * k2], tr);
        if packed {
            let (su, sv) = (self.rand(2 * k / 16), self.rand(2 * n / 16));
            self.put(&format!("{p}.su"), DType::I16, &[k / 16], su);
            self.put(&format!("{p}.sv"), DType::I16, &[n / 16], sv);
        } else {
            let (suh, svh) = (self.rand(2 * k), self.rand(2 * n));
            self.put(&format!("{p}.suh"), DType::F16, &[k], suh);
            self.put(&format!("{p}.svh"), DType::F16, &[n], svh);
        }
        match cb {
            1 => self.put(&format!("{p}.mcg"), DType::I32, &[], MCG.to_le_bytes().to_vec()),
            2 => self.put(&format!("{p}.mul1"), DType::I32, &[], MUL1.to_le_bytes().to_vec()),
            _ => {}
        }
    }
}

/// Two layers (layer 0 with two routed experts), head, embed, an mtp tensor and a `bias_vl` to drop; 2 shard files + index.
fn make_source(dir: &Path) {
    make_source_with(dir, false)
}

/// [`make_source`]; `stages`: with a full DSpark stage `mtp.0` (a layer's tensors plus main_proj, the Markov and
/// confidence heads) instead of the lone mtp tensor.
fn make_source_with(dir: &Path, stages: bool) {
    let mut c = Ckpt { t: BTreeMap::new(), rng: SplitMix64::new(7) };
    let e = c.rand(2 * 16 * 256);
    c.put("embed.weight", DType::Bf16, &[16, 256], e);
    let nrm = c.rand(2 * 256);
    c.put("norm.weight", DType::Bf16, &[256], nrm);
    c.lin("head", 256, 512, 8, 0, false);
    let scopes: Vec<String> = if stages {
        c.lin("mtp.0.main_proj", 768, 256, 4, 2, false);
        for (n, shape) in [("mtp.0.main_norm.weight", vec![256]), ("mtp.0.norm.weight", vec![256]), ("mtp.0.markov_head.embed.weight", vec![16, 32]), ("mtp.0.markov_head.head.weight", vec![16, 32]), ("mtp.0.confidence_head.proj.weight", vec![1, 288])] {
            let b = c.rand(2 * shape.iter().product::<usize>());
            c.put(n, DType::Bf16, &shape, b);
        }
        vec!["layers.0".into(), "layers.1".into(), "mtp.0".into()]
    } else {
        c.lin("mtp.0.attn.wq_a", 256, 128, 4, 2, false);
        vec!["layers.0".into(), "layers.1".into()]
    };
    for p in &scopes {
        c.lin(&format!("{p}.attn.wq_a"), 256, 128, 6, 2, false);
        c.lin(&format!("{p}.attn.wq_b"), 128, 512, 6, 2, false);
        c.lin(&format!("{p}.attn.wo_b"), 512, 256, 4, 1, true);
        for g in 0..4 {
            c.lin(&format!("{p}.attn.wo_a.slice.{g}"), 256, 128, 4 + 2 * (g % 2), if g % 2 == 0 { 2 } else { 1 }, g == 3);
        }
        let sink = c.rand(4 * 8);
        c.put(&format!("{p}.attn.attn_sink"), DType::F32, &[8], sink);
        let gate = c.rand(2 * 4 * 256);
        c.put(&format!("{p}.ffn.gate.weight"), DType::F16, &[4, 256], gate);
        let bias = c.rand(4 * 4);
        c.put(&format!("{p}.ffn.gate.bias"), DType::F32, &[4], bias.clone());
        c.put(&format!("{p}.ffn.gate.bias_vl"), DType::F32, &[4], bias);
        let an = c.rand(2 * 256);
        c.put(&format!("{p}.attn_norm.weight"), DType::Bf16, &[256], an);
        c.lin(&format!("{p}.ffn.shared_experts.w1"), 256, 256, 6, 2, false);
        c.lin(&format!("{p}.ffn.shared_experts.w3"), 256, 256, 6, 2, false);
        c.lin(&format!("{p}.ffn.shared_experts.w2"), 256, 256, 6, 2, true);
    }
    for e in if stages { vec!["layers.0", "mtp.0"] } else { vec!["layers.0"] } {
        for x in 0..2 {
            // experts of layer 0, 3 of them so that lexicographic and numeric order agree only for 0..2; widths 2..3 bits (mul1)
            c.lin(&format!("{e}.ffn.experts.{x}.w1"), H, I, 4 + 2 * x, 2, false);
            c.lin(&format!("{e}.ffn.experts.{x}.w3"), H, I, 6, 2, x == 1);
            c.lin(&format!("{e}.ffn.experts.{x}.w2"), I, H, 4, 2, false);
        }
    }
    std::fs::create_dir_all(dir).unwrap();
    let names: Vec<String> = c.t.keys().cloned().collect();
    let half = names.len() / 2;
    let mut wm = serde_json::Map::new();
    for (i, part) in [&names[..half], &names[half..]].iter().enumerate() {
        let f = format!("model-0000{}-of-00002.safetensors", i + 1);
        let ts: Vec<(&str, &HostTensor)> = part.iter().map(|n| (n.as_str(), &c.t[n])).collect();
        write_safetensors(&dir.join(&f), &ts).unwrap();
        for n in part.iter() {
            wm.insert(n.clone(), serde_json::Value::String(f.clone()));
        }
    }
    std::fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec(&serde_json::json!({"metadata": {}, "weight_map": wm})).unwrap()).unwrap();
    let cfg = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv41/mia_headers/config.json");
    std::fs::copy(cfg, dir.join("config.json")).unwrap();
}

fn f16_slice(b: &[u8], off: usize, len: usize) -> Vec<u8> {
    b[2 * off..2 * (off + len)].to_vec()
}

/// Source tensor bytes as the f16 scale vector (packed sign words unpacked).
fn scale(src: &SafeTensors, p: &str, full: &str, packed: &str) -> Vec<u8> {
    match src.bytes(&format!("{p}.{full}")) {
        Ok(b) => b.to_vec(),
        Err(_) => exl3::unpack_signs(src.bytes(&format!("{p}.{packed}")).unwrap()),
    }
}

fn i32s(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn check_rank(src: &SafeTensors, dir: &Path, rank: usize, world: usize, layers: usize) {
    let c = SafeTensors::open_dir(dir).unwrap();
    let names = c.names();
    assert!(!names.iter().any(|n| n.starts_with("mtp.") || n.ends_with("bias_vl")), "dropped tensors present");
    assert!(!names.iter().any(|n| n.contains(".experts.0.") || n.contains(".experts.1.")), "per-expert tensors present");
    let layer_ok = |n: &str| n.strip_prefix("layers.").and_then(|r| r.split('.').next()).and_then(|l| l.parse::<usize>().ok()).is_none_or(|l| l < layers);
    let mut sharded: Vec<(String, exl3::LayerMeta, bool)> = vec![("head".into(), meta(src, "head"), true)];
    for l in 0..layers {
        let p = format!("layers.{l}");
        for (m, rows) in [("attn.wq_b", true), ("ffn.shared_experts.w1", true), ("ffn.shared_experts.w3", true), ("attn.wo_b", false), ("ffn.shared_experts.w2", false)] {
            sharded.push((format!("{p}.{m}"), meta(src, &format!("{p}.{m}")), rows));
        }
    }
    for (p, lm, rows) in &sharded {
        let axis = if *rows { intarsia_core::weights::ShardAxis::Rows } else { intarsia_core::weights::ShardAxis::Cols };
        let sh = intarsia_core::weights::Shard { axis, rank, world };
        let ((k0, kl), (n0, nl)) = exl3::shard_ranges(lm.k, lm.n, sh, p).unwrap();
        let want = exl3::shard_trellis(src.bytes(&format!("{p}.trellis")).unwrap(), lm.k, lm.n, lm.k2, (k0, kl), (n0, nl)).unwrap();
        assert_eq!(c.bytes(&format!("{p}.trellis")).unwrap(), &want[..], "{p} trellis (a)");
        let (suh, svh) = (scale(src, p, "suh", "su"), scale(src, p, "svh", "sv"));
        assert_eq!(c.bytes(&format!("{p}.suh")).unwrap(), &f16_slice(&suh, k0, kl)[..], "{p} suh");
        assert_eq!(c.bytes(&format!("{p}.svh")).unwrap(), &f16_slice(&svh, n0, nl)[..], "{p} svh");
        assert!(c.bytes(&format!("{p}.su")).is_err(), "{p}: packed sign words must be written unpacked");
        let cb = i32::from_le_bytes(c.bytes(&format!("{p}.codebook")).unwrap().try_into().unwrap());
        assert_eq!(cb, lm.cb as i32, "{p} codebook");
        for s in ["mcg", "mul1"] {
            assert_eq!(c.info(&format!("{p}.{s}")).is_some(), src.info(&format!("{p}.{s}")).is_some(), "{p}.{s} marker");
        }
        // (b) the ranks' slices reassemble the source along the split axis: checked rank by rank against the source rows
        let (tb, nt) = (exl3::tile_bytes(lm.k2), lm.n / 16);
        let full = src.bytes(&format!("{p}.trellis")).unwrap();
        let mine = c.bytes(&format!("{p}.trellis")).unwrap();
        if *rows {
            let w = nl / 16 * tb;
            for a in 0..lm.k / 16 {
                assert_eq!(&mine[a * w..(a + 1) * w], &full[(a * nt + n0 / 16) * tb..(a * nt + n0 / 16) * tb + w], "{p} rows reassembly");
            }
        } else {
            assert_eq!(mine, &full[(k0 / 16) * nt * tb..((k0 + kl) / 16) * nt * tb], "{p} cols reassembly");
        }
    }
    // replicated tensors are byte-identical, with codebooks for EXL3 linears
    for n in src.names() {
        if !layer_ok(&n) || n.starts_with("mtp.") || n.ends_with("bias_vl") || n.contains(".experts.") || n.contains("wo_a.slice") || n.ends_with("attn_sink") {
            continue;
        }
        if sharded.iter().any(|(p, _, _)| n.starts_with(&format!("{p}."))) {
            continue;
        }
        assert_eq!(c.bytes(&n).unwrap(), src.bytes(&n).unwrap(), "{n} replicated");
        assert_eq!(c.info(&n).unwrap().shape, src.info(&n).unwrap().shape, "{n} shape");
    }
    for l in 0..layers {
        let p = format!("layers.{l}");
        let cb = i32::from_le_bytes(c.bytes(&format!("{p}.attn.wq_a.codebook")).unwrap().try_into().unwrap());
        assert_eq!(cb, 2);
        // attn_sink halves
        let sink = src.bytes(&format!("{p}.attn.attn_sink")).unwrap();
        let lh = 8 / world;
        assert_eq!(c.bytes(&format!("{p}.attn.attn_sink")).unwrap(), &sink[4 * rank * lh..4 * (rank + 1) * lh], "attn_sink");
        // (c) wo_a slices renamed to the local index
        let per = 4 / world;
        for i in 0..per {
            let (a, b) = (format!("{p}.attn.wo_a.slice.{}", rank * per + i), format!("{p}.attn.wo_a.slice.{i}"));
            for s in ["trellis", "suh", "svh", "su", "sv", "mcg", "mul1"] {
                assert_eq!(c.bytes(&format!("{b}.{s}")).ok(), src.bytes(&format!("{a}.{s}")).ok(), "{b}.{s} <- {a}.{s}");
            }
            let want_cb = if src.info(&format!("{a}.mul1")).is_some() { 2 } else if src.info(&format!("{a}.mcg")).is_some() { 1 } else { 0 };
            assert_eq!(i32s(c.bytes(&format!("{b}.codebook")).unwrap()), vec![want_cb]);
        }
        assert!(c.info(&format!("{p}.attn.wo_a.slice.{per}.trellis")).is_none(), "more wo_a slices than {per}");
    }
    // routed experts of layer 0 against a direct exl3_moe::load of the source
    if layers > 0 {
        let cpu = CpuBackend::new();
        let names: Vec<[String; 3]> = (0..2).map(|e| ["w1", "w3", "w2"].map(|w| format!("layers.0.ffn.experts.{e}.{w}"))).collect();
        let ex = exl3_moe::load(&cpu, src, &names, H, I, rank, world, 10.0).unwrap();
        for (n, t) in [("arena", &ex.gate_up_w), ("scales", &ex.gate_up_s), ("meta", &ex.down_w), ("codebook", &ex.down_s)] {
            let full = format!("layers.0.ffn.experts.{n}");
            assert_eq!(c.bytes(&full).unwrap(), &cpu.download_row_major(t).unwrap()[..], "{full}");
            assert_eq!(c.info(&full).unwrap().shape, t.shape, "{full} shape");
            assert_eq!(c.info(&full).unwrap().dtype, t.dtype, "{full} dtype");
        }
        assert_eq!(c.info("layers.0.ffn.experts.scales").unwrap().shape, vec![2, 3 * H + 3 * (I / world)]);
    }
    // the header shapes the TP loader plans from
    let hs = HeaderShapes::open(dir).unwrap().0;
    assert_eq!(hs.len(), names.len());
    assert_eq!(hs["head.trellis"].1[1], 512 / 16 / world);
    assert_eq!(hs["head.svh"].1, vec![512 / world]);
    assert_eq!(hs["layers.0.attn.wo_b.suh"].1, vec![512 / world]);
}

fn meta(src: &SafeTensors, p: &str) -> exl3::LayerMeta {
    exl3::layer_meta(&intarsia_core::quant::WeightMeta::new(src, p)).unwrap()
}

/// (d) the loader on the CPU backend returns byte-identical tensors.
fn check_load(dir: &Path) {
    let c = SafeTensors::open_dir(dir).unwrap();
    let cpu = CpuBackend::new();
    let lw = load_rank_cache(dir, &cpu).unwrap();
    let m = open_manifest(dir).unwrap();
    assert_eq!(lw.tensors.len(), m.tensors);
    assert_eq!(lw.tensors.len(), c.names().len());
    assert_eq!(lw.bytes, m.bytes);
    for (n, t) in &lw.tensors {
        let info = c.info(n).unwrap();
        assert_eq!((t.dtype, &t.shape), (info.dtype, &info.shape), "{n}");
        assert_eq!(cpu.download_row_major(t).unwrap(), c.bytes(n).unwrap(), "{n} loaded bytes");
    }
}

#[test]
fn cache_split_reassembly_rename_experts_and_load() {
    let root = tmp("split");
    let src_dir = root.join("src");
    make_source(&src_dir);
    let src = SafeTensors::open_dir(&src_dir).unwrap();
    for (world, layers) in [(1usize, None), (2, None), (2, Some(1))] {
        for rank in 0..world {
            let out = root.join(format!("cache-r{rank}of{world}-{layers:?}"));
            let mut o = BuildOpts::new(rank, world, layers);
            o.part_bytes = 48 << 10; // several parts, an expert arena larger than a part
            let m = build_rank_cache(&src_dir, &out, &o).unwrap();
            assert!(m.parts.len() > 1, "expected several parts, got {}", m.parts.len());
            assert!(cache_matches(&out, &src_dir, rank, world, layers).unwrap());
            assert!(!cache_matches(&out, &src_dir, rank, world, Some(5)).unwrap());
            check_rank(&src, &out, rank, world, layers.unwrap_or(2));
            check_load(&out);
            // manifest is the last file; the index maps every tensor to an existing part
            let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(out.join("model.safetensors.index.json")).unwrap()).unwrap();
            assert_eq!(idx["weight_map"].as_object().unwrap().len(), m.tensors);
            assert!(out.join(format!("rank-00001-of-{:05}.safetensors", m.parts.len())).exists());
        }
    }
    // a different rank / world never matches the cache; a changed source config invalidates it
    let out = root.join("cache-r0of2-None");
    assert!(!cache_matches(&out, &src_dir, 1, 2, None).unwrap());
    assert!(!cache_matches(&out, &src_dir, 0, 1, None).unwrap());
    let cfg = std::fs::read_to_string(src_dir.join("config.json")).unwrap();
    std::fs::write(src_dir.join("config.json"), cfg + "\n").unwrap();
    assert!(!cache_matches(&out, &src_dir, 0, 2, None).unwrap());
    // refuses a foreign non-empty directory without --force
    let foreign = root.join("foreign");
    std::fs::create_dir_all(&foreign).unwrap();
    std::fs::write(foreign.join("keep.txt"), "x").unwrap();
    assert!(build_rank_cache(&src_dir, &foreign, &BuildOpts::new(0, 2, Some(1))).is_err());
    assert!(foreign.join("keep.txt").exists());
    let mut o = BuildOpts::new(0, 2, Some(1));
    o.force = true;
    build_rank_cache(&src_dir, &foreign, &o).unwrap();
    assert!(foreign.join("keep.txt").exists(), "--force must not delete foreign files");
    // a split off the 128 grid is refused (the legality check is exl3::shard_ranges): wq_b has N = 512, w = 8 -> 64 per rank
    assert!(build_rank_cache(&src_dir, &root.join("bad"), &BuildOpts::new(0, 8, Some(1))).is_err());
    let _ = std::fs::remove_dir_all(&root);
}

/// Parts larger than one 64 MiB read chunk: tensors that straddle chunk boundaries, tiny and empty ones in between.
#[test]
fn loader_streams_tensors_across_chunk_boundaries() {
    let root = tmp("chunks");
    let src_dir = root.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    let mut c = Ckpt { t: BTreeMap::new(), rng: SplitMix64::new(11) };
    for (i, n) in [3usize << 20, 70 << 20, 1, 5 << 20, 0, 64 << 20, 9, 130 << 20, 3 << 20].iter().enumerate() {
        let d = c.rand(*n);
        c.put(&format!("a{i}.weight"), DType::U8, &[*n], d);
    }
    let ts: Vec<(&str, &HostTensor)> = c.t.iter().map(|(k, v)| (k.as_str(), v)).collect();
    write_safetensors(&src_dir.join("model.safetensors"), &ts).unwrap();
    std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv41/mia_headers/config.json"), src_dir.join("config.json")).unwrap();
    let out = root.join("cache");
    build_rank_cache(&src_dir, &out, &BuildOpts::new(0, 1, None)).unwrap();
    check_load(&out);
    let _ = std::fs::remove_dir_all(&root);
}

/// Loader throughput on a real cache: `INTARSIA_DSV41_CACHE=<dir> cargo test --release -p intarsia-models --test dsv41_cache -- --ignored --nocapture`.
#[test]
#[ignore]
fn real_cache_loads_on_cpu() {
    let Some(dir) = std::env::var_os("INTARSIA_DSV41_CACHE").map(PathBuf::from) else {
        eprintln!("skipped: set INTARSIA_DSV41_CACHE");
        return;
    };
    let cpu = CpuBackend::new();
    let lw = load_rank_cache(&dir, &cpu).unwrap();
    let m = open_manifest(&dir).unwrap();
    assert_eq!(lw.tensors.len(), m.tensors);
    assert_eq!(lw.bytes, m.bytes);
    eprintln!("loaded {} tensors, {:.2} GiB in {:.2} s = {:.2} GB/s", lw.tensors.len(), lw.bytes as f64 / (1u64 << 30) as f64, lw.secs, lw.bytes as f64 / 1e9 / lw.secs);
}

/// The DSpark sub-cache (`--mtp-only`): only `mtp.*`, sharded with the layers' rules (heads, wo_a slices, sink, shared and
/// routed experts), main_proj / norms / Markov / confidence heads replicated; matches only as the mtp scope; the model's
/// cache of the same source still drops every mtp tensor.
#[test]
fn mtp_sub_cache_shards_the_stages_with_the_layer_rules() {
    use intarsia_models::dsv41_cache::{cache_matches_scope, open_manifest, MTP_DIR};
    let root = tmp("mtp");
    let src_dir = root.join("src");
    make_source_with(&src_dir, true);
    for rank in 0..2 {
        let model = root.join(format!("r{rank}of2"));
        build_rank_cache(&src_dir, &model, &BuildOpts::new(rank, 2, None)).unwrap();
        let shapes = intarsia_models::dsv41::HeaderShapes::open(&model).unwrap().0;
        assert!(shapes.keys().all(|n| !n.starts_with("mtp.")), "the model's cache keeps no mtp tensor");
        let out = model.join(MTP_DIR);
        let mut o = BuildOpts::new(rank, 2, None);
        o.mtp_only = true;
        let m = build_rank_cache(&src_dir, &out, &o).unwrap();
        assert_eq!(m.scope.as_deref(), Some("mtp"));
        assert!(cache_matches_scope(&out, &src_dir, rank, 2, None, true).unwrap());
        assert!(!cache_matches_scope(&out, &src_dir, rank, 2, None, false).unwrap(), "a sub-cache is not a model cache");
        assert!(!cache_matches_scope(&model, &src_dir, rank, 2, None, true).unwrap(), "a model cache is not a sub-cache");
        assert_eq!(open_manifest(&model).unwrap().scope, None);
        let sh = intarsia_models::dsv41::HeaderShapes::open(&out).unwrap().0;
        assert!(sh.keys().all(|n| n.starts_with("mtp.0.")), "{:?}", sh.keys().find(|n| !n.starts_with("mtp.0.")));
        let get = |n: &str| sh.get(n).unwrap_or_else(|| panic!("{n} missing")).1.clone();
        assert_eq!(get("mtp.0.attn.wq_b.trellis"), [128 / 16, 512 / 2 / 16, 8 * 6], "heads split over the ranks");
        assert_eq!(get("mtp.0.attn.wo_b.trellis"), [512 / 2 / 16, 256 / 16, 8 * 4]);
        assert_eq!(get("mtp.0.attn.attn_sink"), [4]);
        assert!(sh.contains_key("mtp.0.attn.wo_a.slice.1.trellis") && !sh.contains_key("mtp.0.attn.wo_a.slice.2.trellis"));
        assert_eq!(get("mtp.0.main_proj.trellis"), [768 / 16, 256 / 16, 8 * 4], "main_proj replicated");
        for n in ["mtp.0.main_norm.weight", "mtp.0.norm.weight", "mtp.0.markov_head.embed.weight", "mtp.0.markov_head.head.weight", "mtp.0.confidence_head.proj.weight"] {
            assert!(sh.contains_key(n), "{n}");
        }
        assert!(sh.contains_key("mtp.0.ffn.experts.arena") && !sh.keys().any(|n| n.contains(".experts.0.")));
        check_load(&out);
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// `--first-layer`: a layer-range cache keeps the global tensors and the layers `first_layer .. layers` under their
/// own names, records the range in the manifest, and only matches a request for the same range.
#[test]
fn layer_range_cache_keeps_only_its_layers() {
    let root = tmp("range");
    let src_dir = root.join("src");
    make_source(&src_dir);
    let out = root.join("cache-l1");
    let mut o = BuildOpts::new(0, 1, None);
    o.first_layer = 1;
    let m = build_rank_cache(&src_dir, &out, &o).unwrap();
    assert_eq!(m.first_layer, 1);
    let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(out.join("model.safetensors.index.json")).unwrap()).unwrap();
    let names: Vec<&String> = idx["weight_map"].as_object().unwrap().keys().collect();
    assert!(names.iter().all(|n| !n.starts_with("layers.0.")), "layer 0 kept");
    assert!(names.iter().any(|n| n.starts_with("layers.1.")), "layer 1 dropped");
    assert!(names.iter().any(|n| !n.starts_with("layers.")), "globals dropped");
    assert!(intarsia_models::dsv41_cache::cache_matches_range(&out, &src_dir, 0, 1, None, 1, false).unwrap());
    assert!(!intarsia_models::dsv41_cache::cache_matches_range(&out, &src_dir, 0, 1, None, 0, false).unwrap());
    assert!(!cache_matches(&out, &src_dir, 0, 1, None).unwrap());
    check_load(&out);
    let _ = std::fs::remove_dir_all(&root);
}
