//! For each prompt: (A) SERVING path = Engine::generate (Seq::prefill_step), greedy 32 tokens, ignore_eos, every
//! sampling-step logits row recorded via GenRequest.trace_logits; (B) Model::forward path = forward_logits_each.
//! Rows are hashed bitwise (FNV-1a 64 over the f32 bits). Row 0 of (A) can be dumped raw (--dump-dir).
use clap::Parser;
use intarsia_cli::common::ModelArgs;
use intarsia_core::engine::{Engine, GenEvent, GenRequest};
use std::sync::{Arc, Mutex};

#[derive(Parser)]
struct A {
    #[command(flatten)]
    model: ModelArgs,
    /// corpus text files (tokenized; prompts are token prefixes of the first one, `code` of the second)
    #[arg(long)]
    docs: String,
    #[arg(long)]
    code: String,
    /// jsonl {"name","ids"} (golden prompts)
    #[arg(long)]
    golden: String,
    #[arg(long, value_delimiter = ',')]
    lens: Vec<usize>,
    #[arg(long, default_value_t = 32)]
    n: usize,
    #[arg(long)]
    out: String,
    #[arg(long)]
    dump_dir: Option<String>,
    /// skip path B (Model::forward)
    #[arg(long)]
    no_forward: bool,
}

fn fnv(h: &mut u64, row: &[f32]) {
    for x in row {
        for b in x.to_bits().to_le_bytes() {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x100000001b3);
        }
    }
}
fn row_hash(row: &[f32]) -> String {
    let mut h = 0xcbf29ce484222325u64;
    fnv(&mut h, row);
    format!("{h:016x}")
}

fn serve(e: &dyn Engine, ids: &[u32], n: usize) -> anyhow::Result<(Vec<u32>, usize, Vec<Vec<f32>>)> {
    let trace = Arc::new(Mutex::new(Vec::new()));
    let req = GenRequest { prompt: ids.to_vec(), max_tokens: n, ignore_eos: true, temperature: 0.0, trace_logits: Some(trace.clone()), ..Default::default() };
    let (mut toks, mut cached, mut done) = (vec![], 0, 0);
    let mut err = None;
    e.generate(req, &mut |ev| match ev {
        GenEvent::Token { id, .. } => toks.push(id),
        GenEvent::Done { cached_tokens, completion_tokens, .. } => { cached = cached_tokens; done = completion_tokens; }
        GenEvent::Error { message } => err = Some(message),
    })?;
    if let Some(m) = err { anyhow::bail!("{m}"); }
    toks.truncate(done);
    let t = trace.lock().unwrap().clone();
    Ok((toks, cached, t))
}

fn main() -> anyhow::Result<()> {
    let a = A::parse();
    let loaded = a.model.load()?;
    let e = loaded.engine;
    eprintln!("chunk {} ({})", loaded.prefill_chunk.value, loaded.prefill_chunk.source);
    let docs = e.tokenize(&std::fs::read_to_string(&a.docs)?, false)?;
    let code = e.tokenize(&std::fs::read_to_string(&a.code)?, false)?;
    let mut prompts: Vec<(String, Vec<u32>)> = vec![];
    for &l in &a.lens {
        let src = if l % 2 == 0 { &docs } else { &code };
        anyhow::ensure!(src.len() >= l, "corpus too short for {l}");
        prompts.push((format!("{}{l}", if l % 2 == 0 { "docs" } else { "code" }), src[..l].to_vec()));
    }
    for line in std::fs::read_to_string(&a.golden)?.lines() {
        let v: serde_json::Value = serde_json::from_str(line)?;
        let ids: Vec<u32> = v["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        prompts.push((v["name"].as_str().unwrap().to_string(), ids));
    }
    let mut out = std::fs::File::create(&a.out)?;
    use std::io::Write;
    for (name, ids) in &prompts {
        // flush the prefix cache with an unrelated prompt so every run prefills from scratch
        let head = ((ids[0] as usize + 1) % 1000) as u32;
        serve(&*e, &[head, 5, 17, 99, 3, 8, 21, 40], 1)?;
        let (toks, cached, rows) = serve(&*e, ids, a.n)?;
        if let Some(d) = &a.dump_dir {
            let mut b = Vec::new();
            for x in &rows[0] { b.extend_from_slice(&x.to_le_bytes()); }
            std::fs::write(format!("{d}/{name}-L{}-row0.f32", ids.len()), b)?;
        }
        let hashes: Vec<String> = rows.iter().map(|r| row_hash(r)).collect();
        let mut rec = serde_json::json!({"name": name, "L": ids.len(), "chunk": loaded.prefill_chunk.value,
            "serve": {"cached": cached, "ids": toks, "row_hashes": hashes}});
        if !a.no_forward {
            let (mut all, mut last, mut cnt) = (0xcbf29ce484222325u64, String::new(), 0usize);
            e.forward_logits_each(ids, &mut |p, row| { fnv(&mut all, row); cnt += 1; if p + 1 == ids.len() { last = row_hash(row); } Ok(()) })?;
            rec["forward"] = serde_json::json!({"rows": cnt, "last_row_hash": last, "all_rows_hash": format!("{all:016x}")});
        }
        writeln!(out, "{rec}")?;
        out.flush()?;
        eprintln!("{name} L={} cached={cached}", ids.len());
    }
    Ok(())
}
