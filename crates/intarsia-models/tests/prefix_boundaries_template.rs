//! Where a multi-turn chat's prompts stop sharing tokens, with the real Qwen3.6 tokenizer and chat template, against the
//! prefix store's cut rule (a cut before every `<|im_start|>` and after the first `\n` that follows it). Turn N+1's
//! prompt contains turn N's answer; the template drops the `<think>` of every turn before the last user query, so the
//! shared prefix must end exactly at a cut (after `<|im_start|>assistant\n` of turn N), or chat reuse would lose the
//! whole turn. No GPU: tokenizer and template only. Skips without `INTARSIA_QWEN36_DIR` (or ~/models/Qwen3.6-35B-A3B-NVFP4).
use intarsia_core::engine::{ChatOpts, TextCodec};
use intarsia_models::{ChatTemplateSource, HfCodec};
use serde_json::json;
use std::path::PathBuf;

fn cuts(ids: &[u32], start: u32, nl: u32) -> Vec<usize> {
    let mut out = Vec::new();
    let mut hdr = false;
    for (i, &t) in ids.iter().enumerate() {
        if t == start {
            if i > 0 {
                out.push(i);
            }
            hdr = true;
        } else if hdr && t == nl {
            out.push(i + 1);
            hdr = false;
        }
    }
    out
}

#[test]
fn qwen36_turns_share_up_to_a_cut() {
    let dir = std::env::var_os("INTARSIA_QWEN36_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("models/Qwen3.6-35B-A3B-NVFP4"));
    if !dir.join("tokenizer.json").exists() {
        eprintln!("SKIP: no Qwen3.6 tokenizer at {}", dir.display());
        return;
    }
    let c = HfCodec::load(&dir, &ChatTemplateSource::default()).unwrap();
    let start = c.encode("<|im_start|>", false).unwrap();
    let nl = c.encode("\n", false).unwrap();
    assert_eq!((start.len(), nl.len()), (1, 1));
    let (start, nl) = (start[0], nl[0]);
    let tools = json!([{"type": "function", "function": {"name": "read_log", "description": "Read a log.", "parameters": {"type": "object", "properties": {"server": {"type": "string"}}}}}]);
    for (thinking, with_tools) in [(true, false), (false, false), (true, true)] {
        let opts = ChatOpts { add_generation_prompt: true, enable_thinking: Some(thinking), tools: with_tools.then(|| tools.clone()), ..Default::default() };
        let mut msgs = vec![json!({"role": "system", "content": "You are terse."}), json!({"role": "user", "content": "What is 2+2? Explain."})];
        let mut prev: Option<Vec<u32>> = None;
        for turn in 0..3 {
            let text = c.apply_chat_template(&msgs, &opts).unwrap();
            let ids = c.encode(&text, false).unwrap();
            if let Some(p) = &prev {
                let common = p.iter().zip(&ids).take_while(|(a, b)| a == b).count();
                let pc = cuts(p, start, nl);
                let best = pc.iter().copied().filter(|&x| x <= common).max().unwrap_or(0);
                println!(
                    "TEMPLATE thinking={thinking} tools={with_tools} turn {turn}: prev {} tok, now {} tok, common {common}, best cut of prev <= common {best}, lost {}; prev tail {:?}",
                    p.len(),
                    ids.len(),
                    common - best,
                    c.decode(&p[best.saturating_sub(4)..], false).unwrap()
                );
                assert_eq!(best, common, "the turns share {common} tokens but the last cut is at {best}");
            }
            // the reply as the client sends it back: thinking in reasoning_content, answer in content
            msgs.push(json!({"role": "assistant", "content": format!("The answer is {}.", 4 + turn), "reasoning_content": "Adding two and two gives four."}));
            msgs.push(json!({"role": "user", "content": format!("And then turn {turn}?")}));
            prev = Some(ids);
        }
    }
}
