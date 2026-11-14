//! The prompt token ids intarsia's chat template path (minijinja + tokenizer.json) feeds Qwen3.6-35B-A3B-NVFP4 against the ids
//! vLLM's `/tokenize` returned for the same 85 GSM8K requests (`enable_thinking` true and false, one user message, generation
//! prompt added; recorded in docs/results/2026-11-13-think-len/raw/). Prints SKIPPED unless `INTARSIA_QWEN36_DIR` points at the
//! checkpoint directory (needs `tokenizer_config.json` and `tokenizer.json`).

use intarsia_core::engine::ChatOpts;
use intarsia_models::chat_template::{ChatTemplate, ChatTemplateSource};
use serde_json::{json, Value};
use std::path::PathBuf;

#[test]
fn qwen36_thinking_on_prompt_ids_match_vllm() {
    check(true, "vllm-tokenize-think-on.json");
}

#[test]
fn qwen36_thinking_off_prompt_ids_match_vllm() {
    check(false, "vllm-tokenize-think-off.json");
}

fn check(thinking: bool, vllm_file: &str) {
    let Some(dir) = std::env::var_os("INTARSIA_QWEN36_DIR").map(PathBuf::from) else {
        eprintln!("SKIPPED: INTARSIA_QWEN36_DIR is not set");
        return;
    };
    let raw = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/results/2026-11-13-think-len/raw");
    let prompts: Value = serde_json::from_slice(&std::fs::read(raw.join("gsm8k-prompts.json")).unwrap()).unwrap();
    let vllm: Vec<Value> = serde_json::from_slice(&std::fs::read(raw.join(vllm_file)).unwrap()).unwrap();
    let (head, tail) = prompts["template"].as_str().unwrap().split_once("{question}").unwrap();
    let t = ChatTemplate::load(&dir, &ChatTemplateSource::Auto).unwrap().expect("the model has a chat template");
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let opts = ChatOpts { add_generation_prompt: true, enable_thinking: Some(thinking), tools: None, extra: Default::default() };
    let questions = &prompts["questions"].as_array().unwrap()[..vllm.len()];
    let mut differ = Vec::new();
    for (i, q) in questions.iter().enumerate() {
        let msgs = vec![json!({"role": "user", "content": format!("{head}{}{tail}", q.as_str().unwrap())})];
        let text = t.render(&msgs, &opts).unwrap();
        let ids: Vec<u64> = tok.encode(text.as_str(), false).unwrap().get_ids().iter().map(|&x| x as u64).collect();
        let want: Vec<u64> = vllm[i]["tokens"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
        if ids != want {
            differ.push(i);
        }
        if i == 0 {
            eprintln!("generation prompt tail: {:?}", &text[text.len() - 30..]);
        }
    }
    assert!(differ.is_empty(), "prompt ids differ from vLLM's /tokenize for questions {differ:?}");
    eprintln!("all {} prompts: identical token ids", vllm.len());
}
