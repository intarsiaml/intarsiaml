//! The prompt token ids intarsia's chat template path (minijinja + tokenizer.json) feeds Qwen3.6-35B-A3B-NVFP4 against the ids
//! vLLM's `/tokenize` returned for the 200 pinned GSM8K requests of `localeval quality --bench gsm8k --limit 200 --pin-gsm8k
//! --thinking off` (one user message, `chat_template_kwargs.enable_thinking=false`, generation prompt added; recorded in
//! docs/results/2026-11-20-style7/raw/). Prints SKIPPED unless `INTARSIA_QWEN36_DIR` points at the checkpoint directory
//! (needs `tokenizer_config.json` and `tokenizer.json`).

use intarsia_core::engine::ChatOpts;
use intarsia_models::chat_template::{ChatTemplate, ChatTemplateSource};
use serde_json::{json, Value};
use std::path::PathBuf;

#[test]
fn qwen36_pinned_gsm8k_think_off_prompt_ids_match_vllm() {
    let Some(dir) = std::env::var_os("INTARSIA_QWEN36_DIR").map(PathBuf::from) else {
        eprintln!("SKIPPED: INTARSIA_QWEN36_DIR is not set");
        return;
    };
    let raw = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/results/2026-11-20-style7/raw");
    let prompts: Vec<Value> =
        serde_json::from_slice(&std::fs::read(raw.join("gsm8k-pinned200-prompts.json")).unwrap()).unwrap();
    let vllm: Vec<Value> =
        serde_json::from_slice(&std::fs::read(raw.join("vllm-tokenize-pinned200-think-off.json")).unwrap()).unwrap();
    assert_eq!(prompts.len(), vllm.len());
    let t = ChatTemplate::load(&dir, &ChatTemplateSource::Auto).unwrap().expect("the model has a chat template");
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let opts =
        ChatOpts { add_generation_prompt: true, enable_thinking: Some(false), tools: None, extra: Default::default() };
    let mut differ = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let msgs = vec![json!({"role": "user", "content": p["prompt"].as_str().unwrap()})];
        let text = t.render(&msgs, &opts).unwrap();
        let ids: Vec<u64> = tok.encode(text.as_str(), false).unwrap().get_ids().iter().map(|&x| x as u64).collect();
        let want: Vec<u64> = vllm[i]["tokens"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
        if ids != want {
            differ.push(i);
        }
    }
    assert!(differ.is_empty(), "prompt ids differ from vLLM's /tokenize for questions {differ:?}");
    eprintln!("all {} prompts: identical token ids", vllm.len());
}
