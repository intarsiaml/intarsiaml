//! Per-row view of the Model::forward path (forward_logits_each, HeadMode::AllLogits) vs the serving path.
//! For each prompt (token prefix of the docs / code corpus, as chunkgolden): FNV-1a hash of EVERY forward row (jsonl),
//! the raw f32 rows of the last --tail positions (dump dir), and the serving path's first sampled row (Engine::generate,
//! 1 token, trace_logits) raw. compare.py diffs two such runs (different --prefill-chunk) row by row.
use clap::Parser;
use intarsia_cli::common::ModelArgs;
use intarsia_core::engine::{Engine, GenEvent, GenRequest};
use std::io::Write;
use std::sync::{Arc, Mutex};

#[derive(Parser)]
struct A {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long)]
    docs: String,
    #[arg(long)]
    code: String,
    #[arg(long, value_delimiter = ',')]
    lens: Vec<usize>,
    /// raw f32 rows kept for the last N positions of each prompt
    #[arg(long, default_value_t = 24)]
    tail: usize,
    #[arg(long)]
    out: String,
    #[arg(long)]
    dump_dir: String,
}

fn row_hash(row: &[f32]) -> String {
    let mut h = 0xcbf29ce484222325u64;
    for x in row {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    format!("{h:016x}")
}
fn raw(row: &[f32]) -> Vec<u8> {
    row.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn serve_row0(e: &dyn Engine, ids: &[u32]) -> anyhow::Result<Vec<f32>> {
    let trace = Arc::new(Mutex::new(Vec::new()));
    let req = GenRequest { prompt: ids.to_vec(), max_tokens: 1, ignore_eos: true, temperature: 0.0, trace_logits: Some(trace.clone()), ..Default::default() };
    let mut err = None;
    e.generate(req, &mut |ev| if let GenEvent::Error { message } = ev { err = Some(message) })?;
    if let Some(m) = err {
        anyhow::bail!("{m}");
    }
    let t = trace.lock().unwrap();
    Ok(t[0].clone())
}

fn main() -> anyhow::Result<()> {
    let a = A::parse();
    let loaded = a.model.load()?;
    let e = loaded.engine;
    eprintln!("chunk {} ({})", loaded.prefill_chunk.value, loaded.prefill_chunk.source);
    let docs = e.tokenize(&std::fs::read_to_string(&a.docs)?, false)?;
    let code = e.tokenize(&std::fs::read_to_string(&a.code)?, false)?;
    std::fs::create_dir_all(&a.dump_dir)?;
    let mut out = std::fs::File::create(&a.out)?;
    for &l in &a.lens {
        let (kind, src) = if l % 2 == 0 { ("docs", &docs) } else { ("code", &code) };
        anyhow::ensure!(src.len() >= l, "corpus too short for {l}");
        let (name, ids) = (format!("{kind}{l}"), &src[..l]);
        // unrelated prompt first so the serving prefill starts from scratch (no prefix hit)
        let head = ((ids[0] as usize + 1) % 1000) as u32;
        serve_row0(&*e, &[head, 5, 17, 99, 3, 8, 21, 40])?;
        let s0 = serve_row0(&*e, ids)?;
        std::fs::write(format!("{}/{name}-serve.f32", a.dump_dir), raw(&s0))?;
        let mut hashes = Vec::with_capacity(l);
        let from = l.saturating_sub(a.tail);
        let mut tail = Vec::new();
        e.forward_logits_each(ids, &mut |p, row| {
            hashes.push(row_hash(row));
            if p >= from {
                tail.extend_from_slice(&raw(row));
            }
            Ok(())
        })?;
        std::fs::write(format!("{}/{name}-fwd-tail{}.f32", a.dump_dir, l - from), tail)?;
        let rec = serde_json::json!({"name": name, "L": l, "chunk": loaded.prefill_chunk.value, "tail_from": from,
            "serve0": row_hash(&s0), "fwd_rows": hashes});
        writeln!(out, "{rec}")?;
        out.flush()?;
        eprintln!("{name} L={l}");
    }
    Ok(())
}
