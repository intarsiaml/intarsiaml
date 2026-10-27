//! Facts about the model and the repository that the stages need: a display name, golden discovery, the bytes a decode
//! step reads (for the byte floor), and small fingerprints for provenance.

use anyhow::Result;
use intarsia_core::quant::Mxfp4Act;
use intarsia_core::weights::WeightSource;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// A readable name for a model directory. `~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/<hash>` is
/// "Qwen3-8B"; `/models/Qwen3-8B` is "Qwen3-8B".
pub fn model_name(dir: &Path, override_name: Option<&str>) -> String {
    if let Some(n) = override_name {
        return sanitize(n);
    }
    let comps: Vec<String> = dir
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(s) => Some(s.to_string_lossy().to_string()),
            _ => None,
        })
        .collect();
    for c in comps.iter().rev() {
        if let Some(rest) = c.strip_prefix("models--") {
            if let Some(n) = rest.rsplit("--").next().filter(|n| !n.is_empty()) {
                return sanitize(n);
            }
        }
    }
    let last = comps.last().cloned().unwrap_or_else(|| "model".to_string());
    sanitize(&last)
}

pub fn sanitize(s: &str) -> String {
    let t: String =
        s.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '_' }).collect();
    if t.is_empty() {
        "model".into()
    } else {
        t
    }
}

/// Normalised key for matching a model name against golden file names: lower case, dots and underscores removed
/// ("Qwen3.6-35B-A3B-NVFP4" matches `qwen36-35b-a3b-nvfp4-...`).
fn key(s: &str) -> String {
    s.to_lowercase().chars().filter(|c| *c != '.' && *c != '_').collect()
}

/// Find the golden file of `name` in `dirs`. `long` asks for the longer prompt set (`<name>-long...`, then `...-4k...`);
/// otherwise the short one. A full-vocabulary fp32 reference outranks other sources. For a model whose MXFP4 activation mode is `act`
/// (`None`: not an MXFP4 model, or unknown), of two goldens made from the same weights, the W4A16 one (`-w4a16` in the name) is taken for `Mxfp4Act::None` and never for
/// `Mxfp4Act::Dynamic` (W4A8, `Mxfp4Act::Fp8`, grades against the W4A16 golden too: it is the W4A16 model with fp8 activations, 12x closer to it than W4A4 is; a W4A16 golden sits ~0.14 KL away from a W4A4 run, docs/results/2026-10-25-ac-27b-accuracy-gate.md).
pub fn find_golden_act(name: &str, dirs: &[PathBuf], long: bool, act: Option<Mxfp4Act>) -> Option<PathBuf> {
    find_scored(name, dirs, false, act, |is_long, is_4k| match (long, is_long, is_4k) {
        (false, false, false) => Some(100),
        (true, true, _) => Some(100),
        (true, false, true) => Some(60),
        _ => None,
    })
}

/// Find the ~4k-token golden of `name` (`<name>-4k...`, built from `tools/prompts/golden_4k.jsonl`): the one golden whose
/// prompts are long enough for the prefix-reuse gate at a prefill chunk of 2048 (the long golden's longest is 700 tokens).
pub fn find_golden_4k(name: &str, dirs: &[PathBuf]) -> Option<PathBuf> {
    find_scored(name, dirs, false, None, |_, is_4k| is_4k.then_some(100))
}

/// Find the format-reference golden of `name` (`<name>-formatref...`): the same quantized weights run by the format's own
/// reference implementation (tools/golden_exl3.py for EXL3). The short prompt set; never picked as an ordinary golden.
pub fn find_format_golden(name: &str, dirs: &[PathBuf]) -> Option<PathBuf> {
    find_scored(name, dirs, true, None, |is_long, is_4k| (!is_long && !is_4k).then_some(100))
}

/// One `__metadata__` string of a safetensors file (reads the header only).
pub fn golden_meta(path: &Path, key: &str) -> Option<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut n = [0u8; 8];
    f.read_exact(&mut n).ok()?;
    let len = u64::from_le_bytes(n);
    if len > 100 << 20 {
        return None;
    }
    let mut h = vec![0u8; len as usize];
    f.read_exact(&mut h).ok()?;
    let v: Value = serde_json::from_slice(&h).ok()?;
    v.get("__metadata__")?.get(key)?.as_str().map(String::from)
}

/// The MXFP4 activation mode a golden was made with: `activations dynamic|none` in the `engine_version` metadata that
/// `tools/golden_quark_mxfp4.py` writes, else a `-w4a16` in the file name. `None`: the golden does not say.
pub fn golden_act(path: &Path) -> Option<Mxfp4Act> {
    if let Some(v) = golden_meta(path, "engine_version") {
        if let Some(rest) = v.split("activations ").nth(1) {
            match rest.split(|c: char| !c.is_ascii_alphanumeric()).next() {
                Some("none") => return Some(Mxfp4Act::None),
                Some("dynamic") => return Some(Mxfp4Act::Dynamic),
                _ => {}
            }
        }
    }
    let stem = path.file_stem()?.to_str()?;
    stem.contains("-w4a16").then_some(Mxfp4Act::None)
}

/// `formatref`: look for format-reference goldens (`-formatref` in the name) instead of ordinary ones.
fn find_scored(
    name: &str,
    dirs: &[PathBuf],
    formatref: bool,
    act: Option<Mxfp4Act>,
    base: impl Fn(bool, bool) -> Option<i32>,
) -> Option<PathBuf> {
    let k = key(name);
    let mut best: Option<(i32, PathBuf)> = None;
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("safetensors") {
                continue;
            }
            let stem = key(p.file_stem().and_then(|s| s.to_str()).unwrap_or(""));
            // by file name first; else by identity: the golden's own `model` metadata (the checkpoint it was made from)
            // names this model directory, whatever the file is called
            let (rest, by_identity) = match stem.strip_prefix(&k) {
                Some(r) => (r.to_string(), false),
                None if stem.contains("meta") => continue,
                None if golden_meta(&p, "model").is_some_and(|m| key(&model_name(Path::new(&m), None)) == k) => {
                    (format!("-{stem}"), true)
                }
                None => continue,
            };
            if !(rest.is_empty() || rest.starts_with('-')) || rest.contains("meta") {
                continue;
            }
            // a format-reference golden belongs to these exact weights: `<name>-formatref...`, no other variant of the name
            if rest.contains("-formatref") != formatref || (formatref && !by_identity && !rest.starts_with("-formatref")) {
                continue;
            }
            let is_long = rest.contains("-long");
            let is_4k = rest.contains("-4k");
            let Some(mut score) = base(is_long, is_4k) else { continue };
            match (act, rest.contains("-w4a16")) {
                (Some(Mxfp4Act::None | Mxfp4Act::Fp8), true) => score += 20,
                (Some(Mxfp4Act::Dynamic), true) => continue,
                _ => {}
            }
            if rest.contains("fp32") {
                score += 10;
            } else if rest.contains("vllm") {
                score += 5;
            }
            if best.as_ref().is_none_or(|(s, _)| score > *s) {
                best = Some((score, p));
            }
        }
    }
    best.map(|(_, p)| p)
}

/// Find the f64 CPU-oracle golden of model `name` (`<name>-cpu-oracle...` in the `oracle/` subdirectory of a golden
/// directory: `intarsia accuracy --backend cpu --write-self-golden` on the short prompt set). It lives apart from the
/// goldens so that `find_golden_act` never takes it for the short golden.
pub fn find_oracle_golden(name: &str, dirs: &[PathBuf]) -> Option<PathBuf> {
    let k = key(name);
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(d.join("oracle")) else { continue };
        let mut hits: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("safetensors"))
            .filter(|p| {
                let stem = key(p.file_stem().and_then(|s| s.to_str()).unwrap_or(""));
                stem.strip_prefix(&k).is_some_and(|rest| rest.starts_with("-cpu-oracle"))
            })
            .collect();
        hits.sort();
        if let Some(p) = hits.into_iter().next() {
            return Some(p);
        }
    }
    None
}

/// The recorded baseline (and noise envelope, if there is one) of a golden for a backend: `<golden stem>.<backend>.baseline.json`
/// / `.envelope.json` in the first of `dirs` that has the baseline (`golden/baselines/`, see `docs/accuracy.md`).
pub fn find_baseline(golden: &Path, backend: &str, dirs: &[PathBuf]) -> Option<(PathBuf, Option<PathBuf>)> {
    let stem = golden.file_stem()?.to_str()?;
    for d in dirs {
        let b = d.join(format!("{stem}.{backend}.baseline.json"));
        if b.is_file() {
            let e = d.join(format!("{stem}.{backend}.envelope.json"));
            return Some((b, e.is_file().then_some(e)));
        }
    }
    None
}

/// Root of the repository checkout (the directory with `tools/golden.py`): `INTARSIA_REPO`, the working directory or
/// an ancestor of the executable (`target/release/intarsia`).
pub fn repo_root() -> Option<PathBuf> {
    let ok = |p: &Path| p.join("tools").join("golden.py").exists();
    if let Ok(r) = std::env::var("INTARSIA_REPO") {
        let p = PathBuf::from(r);
        if ok(&p) {
            return Some(p);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        for a in cwd.ancestors() {
            if ok(a) {
                return Some(a.to_path_buf());
            }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        for a in exe.ancestors() {
            if ok(a) {
                return Some(a.to_path_buf());
            }
        }
    }
    None
}

/// The main checkout of the repository when `root` is a linked git worktree (`git worktree add`), whose `.git` is a
/// file `gitdir: <main>/.git/worktrees/<name>` and whose `commondir` leads back to `<main>/.git`. None otherwise.
pub fn main_checkout(root: &Path) -> Option<PathBuf> {
    let t = std::fs::read_to_string(root.join(".git")).ok()?;
    let gitdir = root.join(t.trim().strip_prefix("gitdir:")?.trim());
    let common = std::fs::read_to_string(gitdir.join("commondir")).ok()?;
    let main = std::fs::canonicalize(gitdir.join(common.trim())).ok()?.parent()?.to_path_buf();
    (Some(&main) != std::fs::canonicalize(root).ok().as_ref()).then_some(main)
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// A cheap identity of a checkpoint for provenance: FNV of `config.json` plus the names and sizes of the weight files
/// (the weights themselves are not read).
pub fn model_fingerprint(dir: &Path) -> String {
    let mut buf = std::fs::read(dir.join("config.json")).unwrap_or_default();
    let mut files: Vec<(String, u64)> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("safetensors"))
                .map(|e| (e.file_name().to_string_lossy().to_string(), e.metadata().map(|m| m.len()).unwrap_or(0)))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    for (n, s) in files {
        buf.extend_from_slice(format!("{n}:{s};").as_bytes());
    }
    format!("fnv1a64:{:016x}", fnv1a64(&buf))
}

pub fn file_fingerprint(path: &Path) -> String {
    match std::fs::read(path) {
        Ok(b) => format!("fnv1a64:{:016x} size:{}", fnv1a64(&b), b.len()),
        Err(_) => "unreadable".into(),
    }
}

/// What the checkpoint is, from its tensors and config.
#[derive(Debug, Clone)]
pub struct WeightFacts {
    /// Bytes of all weight tensors in the checkpoint (vision / MTP towers excluded).
    pub total_bytes: u64,
    /// Estimated bytes one decode step reads: all tensors, routed experts scaled by `top_k / experts`, the input
    /// embedding left out when the output head is a separate tensor (a lookup reads one row).
    pub active_bytes: u64,
    pub moe: bool,
    pub hybrid: bool,
    pub quantized: bool,
}

/// Routed experts one token uses: `num_experts_per_tok` (Qwen, Mixtral) or `top_k_experts` (Gemma 4); 0 when neither is set.
fn moe_top_k(tc: &Value) -> u64 {
    ["num_experts_per_tok", "top_k_experts"].iter().find_map(|k| tc.get(*k).and_then(|v| v.as_u64())).unwrap_or(0)
}

pub fn weight_facts(dir: &Path, cfg: &Value) -> Result<WeightFacts> {
    let st = intarsia_models::safetensors::SafeTensors::open_dir(dir).map_err(|e| anyhow::anyhow!("{e}"))?;
    // an MLX affine checkpoint is read as bf16 (dequantised at load): count what the engine reads, not the 4-bit file
    let st: Box<dyn WeightSource> = match intarsia_core::quant::mlx_affine::MlxAffineConfig::from_config(cfg).map_err(|e| anyhow::anyhow!("{e}"))? {
        Some(m) => Box::new(intarsia_core::quant::mlx_affine::MlxAffine::new(st, &m).map_err(|e| anyhow::anyhow!("{e}"))?),
        None => Box::new(st),
    };
    let tc = intarsia_core::recipe::text_config(cfg);
    let (experts, top_k) = (
        tc.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(0) as f64,
        moe_top_k(tc) as f64,
    );
    let moe = experts > 0.0;
    let hybrid = tc
        .get("layer_types")
        .and_then(|v| v.as_array())
        .is_some_and(|a| a.iter().filter_map(|t| t.as_str()).any(|t| t != "full_attention"));
    let names = st.names();
    let has_head = names.iter().any(|n| n == "lm_head.weight");
    let (mut total, mut active) = (0f64, 0f64);
    let mut quantized = cfg.get("quantization_config").is_some()
        || tc.get("quantization_config").is_some()
        || dir.join("hf_quant_config.json").exists();
    for n in &names {
        if n.contains("visual")
            || n.contains("vision")
            || n.starts_with("mtp")
            || n.contains(".mtp.")
            || n.contains("mtp.")
        {
            continue;
        }
        let Some(i) = st.info(n) else { continue };
        let bytes = (i.shape.iter().product::<usize>() * i.dtype.size_bytes()) as f64;
        total += bytes;
        if n.ends_with("_scale") || n.ends_with("weight_scale_2") || n.ends_with("input_scale") {
            quantized = true;
        }
        if has_head && (n.ends_with("embed_tokens.weight")) {
            continue;
        }
        if moe && n.contains(".experts.") && !n.contains("shared_expert") && top_k > 0.0 {
            active += bytes * (top_k / experts);
        } else {
            active += bytes;
        }
    }
    Ok(WeightFacts { total_bytes: total as u64, active_bytes: active as u64, moe, hybrid, quantized })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_top_k_reads_both_config_spellings() {
        assert_eq!(moe_top_k(&serde_json::json!({"num_experts": 128, "num_experts_per_tok": 8})), 8);
        // Gemma 4 spells it `top_k_experts`; unread, the decode byte floor counted every expert (17.6 GB/token for 26B-A4B).
        assert_eq!(moe_top_k(&serde_json::json!({"num_experts": 128, "top_k_experts": 8})), 8);
        assert_eq!(moe_top_k(&serde_json::json!({"hidden_size": 8})), 0);
    }

    #[test]
    fn names_from_paths() {
        assert_eq!(model_name(Path::new("/models/Qwen3-8B"), None), "Qwen3-8B");
        let hf = "/Users/j/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473";
        assert_eq!(model_name(Path::new(hf), None), "Qwen3-Reranker-0.6B");
        assert_eq!(model_name(Path::new("/x"), Some("my model")), "my_model");
    }

    #[test]
    fn golden_discovery_prefers_fp32_and_separates_long() {
        let d = std::env::temp_dir().join(format!("qualify-golden-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        for f in [
            "qwen3-8b.safetensors",
            "qwen3-8b-fp32.safetensors",
            "qwen3-8b-long.safetensors",
            "qwen3-8b-long-fp32.safetensors",
            "qwen3-8b-4k-top20.safetensors",
            "qwen3-8b-4k-fp32.safetensors",
            "qwen36-35b-a3b-nvfp4-vllm-top20.safetensors",
            "qwen36-35b-a3b-nvfp4-vllm-top20-meta.safetensors",
            "qwen3-8b-formatref.safetensors",
        ] {
            std::fs::write(d.join(f), b"x").unwrap();
        }
        let dirs = vec![d.clone()];
        let n = |p: Option<PathBuf>| p.map(|p| p.file_name().unwrap().to_string_lossy().to_string());
        assert_eq!(n(find_golden_act("Qwen3-8B", &dirs, false, None)).as_deref(), Some("qwen3-8b-fp32.safetensors"));
        assert_eq!(n(find_golden_act("Qwen3-8B", &dirs, true, None)).as_deref(), Some("qwen3-8b-long-fp32.safetensors"));
        assert_eq!(
            n(find_golden_act("Qwen3.6-35B-A3B-NVFP4", &dirs, false, None)).as_deref(),
            Some("qwen36-35b-a3b-nvfp4-vllm-top20.safetensors")
        );
        assert_eq!(n(find_golden_act("Qwen3.6-35B-A3B-NVFP4", &dirs, true, None)), None);
        assert_eq!(n(find_golden_act("Llama-3", &dirs, false, None)), None);
        // the 4k golden is found on its own, fp32 first; it is not the short or the long golden
        assert_eq!(n(find_golden_4k("Qwen3-8B", &dirs)).as_deref(), Some("qwen3-8b-4k-fp32.safetensors"));
        assert_eq!(n(find_golden_4k("Qwen3.6-35B-A3B-NVFP4", &dirs)), None);
        assert_eq!(n(find_golden_4k("Llama-3", &dirs)), None);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn mxfp4_golden_follows_the_activation_mode() {
        let d = std::env::temp_dir().join(format!("qualify-mxfp4-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        for f in ["qwen3.8-27b-mxfp4-hfquark-top20.safetensors", "qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors"] {
            std::fs::write(d.join(f), b"x").unwrap();
        }
        let dirs = vec![d.clone()];
        let n = |p: Option<PathBuf>| p.map(|p| p.file_name().unwrap().to_string_lossy().to_string());
        let name = "qwen3.8-27b-mxfp4-hfquark";
        let w4a4 = Some("qwen3.8-27b-mxfp4-hfquark-top20.safetensors");
        let w4a16 = Some("qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors");
        // W4A16 effective (the Qwen3.8-27B recipe default): the W4A16 golden; W4A4 effective (`--mxfp4-act dynamic`): never it
        assert_eq!(n(find_golden_act(name, &dirs, false, Some(Mxfp4Act::None))).as_deref(), w4a16);
        assert_eq!(n(find_golden_act(name, &dirs, false, Some(Mxfp4Act::Dynamic))).as_deref(), w4a4);
        assert_eq!(n(find_golden_act(name, &dirs, false, Some(Mxfp4Act::Fp8))).as_deref(), w4a16, "W4A8 takes the W4A16 golden");
        // not an MXFP4 model: a golden is not picked by its tag
        assert!(n(find_golden_act(name, &dirs, false, None)).is_some());
        // the mode a golden was made with: file name, else the metadata the golden tool writes
        assert_eq!(golden_act(&d.join("qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors")), Some(Mxfp4Act::None));
        assert_eq!(golden_act(&d.join("qwen3.8-27b-mxfp4-hfquark-top20.safetensors")), None);
        let meta = |act: &str| {
            let h = format!(
                r#"{{"__metadata__":{{"engine_version":"transformers 5, layer-streamed, activations {act} (Quark even_round), 496 QDQ'd linears"}}}}"#
            );
            let mut b = (h.len() as u64).to_le_bytes().to_vec();
            b.extend(h.as_bytes());
            b
        };
        std::fs::write(d.join("a4.safetensors"), meta("dynamic")).unwrap();
        std::fs::write(d.join("a16.safetensors"), meta("none")).unwrap();
        assert_eq!(golden_act(&d.join("a4.safetensors")), Some(Mxfp4Act::Dynamic));
        assert_eq!(golden_act(&d.join("a16.safetensors")), Some(Mxfp4Act::None));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn oracle_golden_lives_in_oracle_and_is_never_the_short_golden() {
        let d = std::env::temp_dir().join(format!("qualify-oracle-{}", std::process::id()));
        std::fs::create_dir_all(d.join("oracle")).unwrap();
        std::fs::write(d.join("qwen36-35b-a3b-nvfp4-vllm-top20.safetensors"), b"x").unwrap();
        std::fs::write(d.join("oracle/qwen36-35b-a3b-nvfp4-cpu-oracle-short.safetensors"), b"x").unwrap();
        std::fs::write(d.join("oracle/qwen3-8b-vllm-top20.safetensors"), b"x").unwrap();
        let dirs = vec![d.clone()];
        let n = |p: Option<PathBuf>| p.map(|p| p.file_name().unwrap().to_string_lossy().to_string());
        assert_eq!(
            n(find_oracle_golden("Qwen3.6-35B-A3B-NVFP4", &dirs)).as_deref(),
            Some("qwen36-35b-a3b-nvfp4-cpu-oracle-short.safetensors")
        );
        assert_eq!(n(find_oracle_golden("Qwen3-8B", &dirs)), None, "only -cpu-oracle files are oracles");
        assert_eq!(
            n(find_golden_act("Qwen3.6-35B-A3B-NVFP4", &dirs, false, None)).as_deref(),
            Some("qwen36-35b-a3b-nvfp4-vllm-top20.safetensors")
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn format_golden_is_found_apart_and_never_as_the_ordinary_golden() {
        let d = std::env::temp_dir().join(format!("qualify-formatref-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        for f in ["qwen3-8b-exl3-4.0bpw-formatref-exllamav3.safetensors", "qwen3-8b-exl3-4.0bpw-long-formatref.safetensors"] {
            std::fs::write(d.join(f), b"x").unwrap();
        }
        let dirs = vec![d.clone()];
        let n = |p: Option<PathBuf>| p.map(|p| p.file_name().unwrap().to_string_lossy().to_string());
        assert_eq!(
            n(find_format_golden("qwen3-8b-exl3-4.0bpw", &dirs)).as_deref(),
            Some("qwen3-8b-exl3-4.0bpw-formatref-exllamav3.safetensors")
        );
        assert_eq!(n(find_golden_act("qwen3-8b-exl3-4.0bpw", &dirs, false, None)), None);
        assert_eq!(n(find_golden_act("qwen3-8b-exl3-4.0bpw", &dirs, true, None)), None);
        assert_eq!(n(find_format_golden("Qwen3-8B", &dirs)), None);
        // metadata is read from the header only
        let hdr = br#"{"__metadata__":{"reference_kind":"format","reference_engine":"exllamav3 0.0.x"}}"#;
        let mut bytes = (hdr.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(hdr);
        let p = d.join("meta-only.safetensors");
        std::fs::write(&p, bytes).unwrap();
        assert_eq!(golden_meta(&p, "reference_kind").as_deref(), Some("format"));
        assert_eq!(golden_meta(&p, "nope"), None);
        assert_eq!(golden_meta(&d.join("qwen3-8b-exl3-4.0bpw-long-formatref.safetensors"), "reference_kind"), None);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn baselines_are_found_by_golden_stem_and_backend() {
        let d = std::env::temp_dir().join(format!("qualify-base-{}", std::process::id()));
        let (a, b) = (d.join("a"), d.join("b"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        for f in ["g.metal.baseline.json", "g.metal.envelope.json"] {
            std::fs::write(b.join(f), b"{}").unwrap();
        }
        std::fs::write(a.join("g.cuda.baseline.json"), b"{}").unwrap();
        let dirs = vec![a.clone(), b.clone()];
        let g = Path::new("/x/g.safetensors");
        let (bp, ep) = find_baseline(g, "metal", &dirs).unwrap();
        assert_eq!(bp, b.join("g.metal.baseline.json"));
        assert_eq!(ep, Some(b.join("g.metal.envelope.json")));
        let (bp, ep) = find_baseline(g, "cuda", &dirs).unwrap();
        assert_eq!((bp, ep), (a.join("g.cuda.baseline.json"), None));
        assert!(find_baseline(g, "hip", &dirs).is_none());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn main_checkout_is_found_from_a_worktree() {
        let d = std::env::temp_dir().join(format!("qualify-wt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let (main, wt) = (d.join("main"), d.join("wt"));
        let wtgit = main.join(".git/worktrees/wt");
        std::fs::create_dir_all(&wtgit).unwrap();
        std::fs::create_dir_all(&wt).unwrap();
        std::fs::write(wt.join(".git"), format!("gitdir: {}\n", wtgit.display())).unwrap();
        std::fs::write(wtgit.join("commondir"), "../..\n").unwrap();
        assert_eq!(main_checkout(&wt), Some(std::fs::canonicalize(&main).unwrap()));
        // the main checkout itself (.git is a directory) and a plain directory have none
        assert_eq!(main_checkout(&main), None);
        assert_eq!(main_checkout(&d), None);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Header-only safetensors file with `__metadata__` (what `golden_meta` reads).
    fn write_meta_golden(path: &Path, meta: &str) {
        let h = format!(r#"{{"__metadata__":{{{meta}}}}}"#);
        let mut b = (h.len() as u64).to_le_bytes().to_vec();
        b.extend(h.as_bytes());
        std::fs::write(path, b).unwrap();
    }

    /// A golden named differently from the model directory is found by the checkpoint its metadata names
    /// (drill g4d: `Gemma-4-26B-A4B-NVFP4` vs `gemma4-26b-a4b-nvfp4-fp32.safetensors`).
    #[test]
    fn golden_found_by_model_identity_when_the_name_differs() {
        let d = std::env::temp_dir().join(format!("qualify-golden-id-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        write_meta_golden(&d.join("gemma4-26b-a4b-nvfp4-fp32.safetensors"), r#""model":"/home/nvidia/models/Gemma-4-26B-A4B-NVFP4""#);
        write_meta_golden(&d.join("other-fp32.safetensors"), r#""model":"/models/Llama-3""#);
        let dirs = vec![d.clone()];
        let n = |p: Option<PathBuf>| p.map(|p| p.file_name().unwrap().to_string_lossy().to_string());
        assert_eq!(n(find_golden_act("Gemma-4-26B-A4B-NVFP4", &dirs, false, None)), Some("gemma4-26b-a4b-nvfp4-fp32.safetensors".into()));
        // no identity, no name: still a miss
        assert_eq!(find_golden_act("Qwen3-8B", &dirs, false, None), None);
        std::fs::remove_dir_all(&d).ok();
    }
}
