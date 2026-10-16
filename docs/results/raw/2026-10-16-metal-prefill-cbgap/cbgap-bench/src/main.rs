//! Warm prefill timing, serving path. For each length in `--lens` (in order, `--reps` times round-robin) one
//! `Engine::generate` request with `max_tokens` 1 on a corpus slice that shares no prefix with the previous request
//! (no KV prefix reuse). One JSON line per request: len, rep, prefill_ms (engine), wall_ms, and t0/t1 in seconds on the
//! clock of `INTARSIA_METAL_PROFILE=cb:` traces (CLOCK_UPTIME_RAW == CACurrentMediaTime), so a trace can be cut per request.
use clap::Parser;
use intarsia_cli::common::ModelArgs;
use intarsia_core::engine::{GenEvent, GenRequest};

#[derive(Parser)]
struct A {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long)]
    docs: String,
    #[arg(long, value_delimiter = ',')]
    lens: Vec<usize>,
    #[arg(long, default_value_t = 3)]
    reps: usize,
    #[arg(long)]
    out: String,
    /// untimed requests before the first measured one
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    /// A/B inside the process: every measured request alternates between staged uploads on (this run-ahead bound, e.g. 2) and
    /// off (0), order flipped each repetition (ABBA), logged as "mode"
    #[arg(long)]
    ab: Option<usize>,
}

extern "C" {
    fn clock_gettime_nsec_np(clock_id: u32) -> u64;
}
const CLOCK_UPTIME_RAW: u32 = 8;

fn uptime() -> f64 {
    // SAFETY: plain libc call.
    unsafe { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) as f64 * 1e-9 }
}

fn main() -> anyhow::Result<()> {
    let a = A::parse();
    let loaded = a.model.load()?;
    let e = loaded.engine;
    eprintln!("chunk {} ({})", loaded.prefill_chunk.value, loaded.prefill_chunk.source);
    let docs = e.tokenize(&std::fs::read_to_string(&a.docs)?, false)?;
    let mut out = std::fs::File::create(&a.out)?;
    let mut k = 0usize;
    let mut prev_first = u32::MAX;
    let mut run = |len: usize, rep: i64, mode: &str, out: &mut std::fs::File| -> anyhow::Result<()> {
        // slice start: a different first token than the previous request's
        let mut off = (k * 211 + 17) % (docs.len() - len);
        while docs[off] == prev_first {
            off += 1;
        }
        prev_first = docs[off];
        k += 1;
        let req = GenRequest { prompt: docs[off..off + len].to_vec(), max_tokens: 1, ignore_eos: true, temperature: 0.0, ..Default::default() };
        let (mut cached, mut pms, mut err) = (0, 0.0, None);
        let t0 = uptime();
        e.generate(req, &mut |ev| match ev {
            GenEvent::Done { cached_tokens, prefill_ms, .. } => {
                cached = cached_tokens;
                pms = prefill_ms;
            }
            GenEvent::Error { message } => err = Some(message),
            _ => {}
        })?;
        let t1 = uptime();
        if let Some(m) = err {
            anyhow::bail!("{m}");
        }
        anyhow::ensure!(cached == 0, "prefix reuse ({cached} cached) would skew the timing");
        use std::io::Write;
        writeln!(out, "{}", serde_json::json!({"len": len, "rep": rep, "prefill_ms": pms, "wall_ms": (t1 - t0) * 1e3, "t0": t0, "t1": t1, "mode": mode, "chunk": loaded.prefill_chunk.value}))?;
        Ok(())
    };
    for w in 0..a.warmup {
        for &l in &a.lens {
            run(l, -(w as i64) - 1, "warmup", &mut out)?;
        }
    }
    for r in 0..a.reps {
        for &l in &a.lens {
            match a.ab {
                None => run(l, r as i64, "plain", &mut out)?,
                Some(on) => {
                    let order = if r % 2 == 0 { [("on", on), ("off", 0)] } else { [("off", 0), ("on", on)] };
                    for (name, n) in order {
                        intarsia_backend_metal::set_staged_upload_ahead(n);
                        run(l, r as i64, name, &mut out)?;
                    }
                }
            }
        }
    }
    Ok(())
}
