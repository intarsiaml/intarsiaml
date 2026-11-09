//! `intarsia mtp-dump`: dump what the checkpoint's MTP head sees and says, for offline fine-tuning of the head.
//!
//! ```text
//! intarsia mtp-dump --backend <b> --model <dir> [--recipe ...] [--mtp-draft-vocab N] \
//!     --input <tokens.jsonl> --out <dir> [--head-out]
//! ```
//!
//! `--input` is JSON lines `{"id": "...", "tokens": [u32, ...]}` (already tokenized: prompt + response). Each sequence is
//! prefilled through the normal path on a fresh sequence with the MTP head loaded (`--spec mtp` is implied), teacher
//! forced: nothing is generated, the head catches up over every position exactly as it does in serving (same chunks, same
//! kernels; the only difference is that the K / V-only shortcut of a prompt catch-up runs the full layer so each row's
//! output exists, which leaves the head's cache bitwise the same). Single rank: `--tp` > 1 is refused.
//!
//! For a sequence of `n` tokens `<out>/<id>.safetensors` holds (position `p` = `0..n-1`; the head pairs row `p` with token
//! `p + 1` and predicts token `p + 2`, so the last position, whose next token is unknown, has no row):
//!
//! | tensor | dtype, shape | content |
//! |---|---|---|
//! | `tokens` | I64 `[n]` | the input tokens |
//! | `hidden` | BF16 `[n-1, H]` | row `p`: the main model's final-normed hidden row `mtp_queue` queued for the head at position `p` (read back from the head's pending queue, nothing recomputed; converted to bf16 when the activation dtype is not bf16) |
//! | `head_top1` | I64 `[n-1]` | the stock head's argmax at row `p` over the full vocabulary (the serving 1-row lm_head), whatever `--mtp-draft-vocab` says |
//! | `head_top1_dv` | I64 `[n-1]` | the argmax over the draft vocabulary's rows; only when the head drafts over a prefix (`--mtp-draft-vocab N`, N > 0, or the recipe's default) |
//! | `head_out` | BF16 `[n-1, H]` | with `--head-out`: the head's final normed output row at `p` (the input of a chained draft) |
//!
//! Safetensors metadata: `model` (directory), `backend`, `hidden_size`, `vocab_size`, `draft_vocab` (0 = full head),
//! `act_dtype`, `git_rev`. The lm_head runs for every row here (dev path); the serving catch-up still does it for the last row only.

use crate::common::ModelArgs;
use anyhow::{bail, Context, Result};
use clap::Args;
use intarsia_core::model::MtpRows;
use std::collections::{BTreeMap, HashSet};
use std::io::BufRead;
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct MtpDumpArgs {
    #[command(flatten)]
    pub model: ModelArgs,
    /// JSON lines `{"id": "...", "tokens": [u32, ...]}`: already tokenized sequences (prompt + response).
    #[arg(long)]
    pub input: PathBuf,
    /// Output directory (created); one `<id>.safetensors` per input line.
    #[arg(long)]
    pub out: PathBuf,
    /// Also write `head_out`: the head's final normed output row per position.
    #[arg(long)]
    pub head_out: bool,
}

struct Seq {
    id: String,
    tokens: Vec<u32>,
}

/// Parse the input lines (blank lines skipped). Ids become file names, so they may not contain path separators.
fn parse_input(r: impl BufRead) -> Result<Vec<Seq>> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (i, line) in r.lines().enumerate() {
        let line = line.with_context(|| format!("reading input line {}", i + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line).with_context(|| format!("input line {}: not JSON", i + 1))?;
        let id = v.get("id").and_then(|x| x.as_str()).with_context(|| format!("input line {}: no string \"id\"", i + 1))?.to_string();
        if id.is_empty() || id.contains(['/', '\\']) || id == "." || id == ".." {
            bail!("input line {}: id {id:?} cannot be a file name", i + 1);
        }
        if !seen.insert(id.clone()) {
            bail!("input line {}: duplicate id {id:?}", i + 1);
        }
        let arr = v.get("tokens").and_then(|x| x.as_array()).with_context(|| format!("input line {}: no \"tokens\" array", i + 1))?;
        let tokens = arr
            .iter()
            .map(|t| t.as_u64().and_then(|x| u32::try_from(x).ok()))
            .collect::<Option<Vec<u32>>>()
            .with_context(|| format!("input line {} ({id}): tokens must be non-negative integers below 2^32", i + 1))?;
        if tokens.len() < 2 {
            bail!("input line {} ({id}): needs at least 2 tokens", i + 1);
        }
        out.push(Seq { id, tokens });
    }
    Ok(out)
}

struct Tensor {
    name: &'static str,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn i64s(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u16s(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The safetensors format: an 8-byte little-endian header length, a JSON header (padded to 8 bytes) and the raw data.
fn write_safetensors(path: &Path, tensors: &[Tensor], meta: &BTreeMap<String, String>) -> Result<()> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), serde_json::to_value(meta)?);
    let mut off = 0usize;
    for t in tensors {
        header.insert(t.name.into(), serde_json::json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [off, off + t.bytes.len()]}));
        off += t.bytes.len();
    }
    let mut hdr = serde_json::to_vec(&serde_json::Value::Object(header))?;
    while (8 + hdr.len()) % 8 != 0 {
        hdr.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + hdr.len() + off);
    out.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
    out.extend_from_slice(&hdr);
    for t in tensors {
        out.extend_from_slice(&t.bytes);
    }
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))
}

/// Write one sequence's dump: `tokens` plus the rows ([`MtpRows`]) of the head.
pub fn write_dump(path: &Path, tokens: &[u32], rows: &MtpRows, meta: &BTreeMap<String, String>) -> Result<()> {
    let (n, h) = (rows.rows, rows.hidden);
    if tokens.len() != n + 1 || rows.hidden_rows.len() != n * h || rows.top1.len() != n {
        bail!("dump of {} tokens has {} rows of {} (hidden {}, top1 {})", tokens.len(), n, h, rows.hidden_rows.len(), rows.top1.len());
    }
    let mut ts = vec![
        Tensor { name: "tokens", dtype: "I64", shape: vec![tokens.len()], bytes: i64s(&tokens.iter().map(|&t| t as i64).collect::<Vec<_>>()) },
        Tensor { name: "hidden", dtype: "BF16", shape: vec![n, h], bytes: u16s(&rows.hidden_rows) },
        Tensor { name: "head_top1", dtype: "I64", shape: vec![n], bytes: i64s(&rows.top1) },
    ];
    if let Some(dv) = &rows.top1_dv {
        ts.push(Tensor { name: "head_top1_dv", dtype: "I64", shape: vec![n], bytes: i64s(dv) });
    }
    if let Some(o) = &rows.head_out {
        if o.len() != n * h {
            bail!("head_out has {} values, expected {}", o.len(), n * h);
        }
        ts.push(Tensor { name: "head_out", dtype: "BF16", shape: vec![n, h], bytes: u16s(o) });
    }
    write_safetensors(path, &ts, meta)
}

fn git_rev() -> String {
    let run = |args: &[&str]| std::process::Command::new("git").args(args).output().ok().filter(|o| o.status.success());
    let Some(head) = run(&["rev-parse", "HEAD"]).and_then(|o| String::from_utf8(o.stdout).ok()) else { return "unknown".into() };
    let dirty = run(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|o| !o.stdout.is_empty());
    format!("{}{}", head.trim(), if dirty { "+dirty" } else { "" })
}

pub fn run(mut args: MtpDumpArgs) -> Result<()> {
    if args.model.tp > 1 || args.model.tp_remote.is_some() {
        bail!("mtp-dump runs on a single rank (--tp {}{}): drop --tp / --tp-remote", args.model.tp, if args.model.tp_remote.is_some() { ", --tp-remote" } else { "" });
    }
    match args.model.spec.as_deref() {
        None | Some("mtp") => args.model.spec = Some("mtp".into()),
        Some(other) => bail!("mtp-dump loads the MTP head itself; --spec {other} conflicts (leave --spec unset or `mtp`)"),
    }
    let seqs = parse_input(std::io::BufReader::new(std::fs::File::open(&args.input).with_context(|| format!("opening {}", args.input.display()))?))?;
    if seqs.is_empty() {
        bail!("{}: no sequences", args.input.display());
    }
    std::fs::create_dir_all(&args.out).with_context(|| format!("creating {}", args.out.display()))?;

    let t_load = std::time::Instant::now();
    let loaded = args.model.load()?;
    let engine = loaded.engine;
    let info = engine.info();
    eprintln!("loaded {} on {} in {:.1}s (hidden {}, vocab {}, act {})", info.id, info.device, t_load.elapsed().as_secs_f64(), info.hidden, info.vocab, info.act_dtype);
    if let Some(s) = seqs.iter().find(|s| s.tokens.len() > info.context_len) {
        bail!("{}: {} tokens exceed the context of {} (raise --max-seq)", s.id, s.tokens.len(), info.context_len);
    }

    let rev = git_rev();
    let t_all = std::time::Instant::now();
    let (mut total_rows, mut total_tokens) = (0usize, 0usize);
    for s in &seqs {
        let t0 = std::time::Instant::now();
        let rows = engine.mtp_dump(&s.tokens, args.head_out).map_err(|e| anyhow::anyhow!("{}: {e}", s.id))?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let meta: BTreeMap<String, String> = [
            ("model", args.model.model.display().to_string()),
            ("backend", args.model.backend.clone()),
            ("hidden_size", info.hidden.to_string()),
            ("vocab_size", info.vocab.to_string()),
            ("draft_vocab", rows.draft_vocab.to_string()),
            ("act_dtype", info.act_dtype.to_string()),
            ("git_rev", rev.clone()),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        write_dump(&args.out.join(format!("{}.safetensors", s.id)), &s.tokens, &rows, &meta)?;
        println!("{}: {} rows, {:.0} ms", s.id, rows.rows, ms);
        total_rows += rows.rows;
        total_tokens += s.tokens.len();
    }
    println!("total: {} sequences, {} tokens, {} rows, {:.0} ms", seqs.len(), total_tokens, total_rows, t_all.elapsed().as_secs_f64() * 1e3);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal safetensors reader: (dtype, shape, bytes) per tensor, and the metadata.
    fn read(path: &Path) -> (BTreeMap<String, (String, Vec<usize>, Vec<u8>)>, BTreeMap<String, String>) {
        let b = std::fs::read(path).unwrap();
        let hl = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
        let hdr: serde_json::Value = serde_json::from_slice(&b[8..8 + hl]).unwrap();
        let data = &b[8 + hl..];
        let mut ts = BTreeMap::new();
        let mut meta = BTreeMap::new();
        for (k, v) in hdr.as_object().unwrap() {
            if k == "__metadata__" {
                meta = v.as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string())).collect();
                continue;
            }
            let o = v["data_offsets"].as_array().unwrap();
            let (a, e) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
            let shape = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            ts.insert(k.clone(), (v["dtype"].as_str().unwrap().to_string(), shape, data[a..e].to_vec()));
        }
        (ts, meta)
    }

    #[test]
    fn the_dump_file_round_trips_tokens_and_rows() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target").join(format!("mtp-dump-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tokens = [7u32, 3, 4_000_000_000, 9];
        let rows = MtpRows {
            rows: 3,
            hidden: 2,
            hidden_rows: vec![1, 2, 3, 4, 5, 0xffff],
            head_out: Some(vec![6, 7, 8, 9, 10, 11]),
            top1: vec![10, 20, 30],
            top1_dv: Some(vec![1, 2, 3]),
            draft_vocab: 8,
        };
        let meta: BTreeMap<String, String> = [("hidden_size".to_string(), "2".to_string())].into();
        let p = dir.join("a.safetensors");
        write_dump(&p, &tokens, &rows, &meta).unwrap();
        let (ts, m) = read(&p);
        assert_eq!(m["hidden_size"], "2");
        let (dt, sh, by) = &ts["tokens"];
        assert_eq!((dt.as_str(), sh.clone()), ("I64", vec![4]));
        let got: Vec<i64> = by.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(got, vec![7, 3, 4_000_000_000, 9]);
        let (dt, sh, by) = &ts["hidden"];
        assert_eq!((dt.as_str(), sh.clone()), ("BF16", vec![3, 2]));
        assert_eq!(by.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect::<Vec<_>>(), rows.hidden_rows);
        assert_eq!(ts["head_out"].1, vec![3, 2]);
        assert_eq!(ts["head_top1"].1, vec![3]);
        assert_eq!(ts["head_top1_dv"].1, vec![3]);
        // without a reduced head or --head-out those tensors are absent
        let plain = MtpRows { top1_dv: None, head_out: None, draft_vocab: 0, ..rows };
        write_dump(&p, &tokens, &plain, &meta).unwrap();
        let (ts, _) = read(&p);
        assert!(!ts.contains_key("head_top1_dv") && !ts.contains_key("head_out"));
        assert!(write_dump(&p, &tokens[..3], &plain, &meta).is_err(), "rows must be tokens - 1");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn input_lines_are_validated() {
        let ok = parse_input(std::io::Cursor::new("{\"id\":\"a\",\"tokens\":[1,2,3]}\n\n{\"id\":\"b\",\"tokens\":[4,5]}\n")).unwrap();
        assert_eq!(ok.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(ok[0].tokens, [1, 2, 3]);
        for bad in [
            "{\"id\":\"a\",\"tokens\":[1]}",
            "{\"id\":\"a/b\",\"tokens\":[1,2]}",
            "{\"tokens\":[1,2]}",
            "{\"id\":\"a\",\"tokens\":[1,-2]}",
            "{\"id\":\"a\",\"tokens\":[1,2]}\n{\"id\":\"a\",\"tokens\":[1,2]}",
            "not json",
        ] {
            assert!(parse_input(std::io::Cursor::new(bad)).is_err(), "{bad}");
        }
    }
}
