//! The real Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct chat templates (minijinja) against transformers' rendering of the
//! same conversation (`tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True)`; Llama with a fixed
//! `date_string`). Prints SKIPPED unless `INTARSIA_LLAMA31_DIR` / `INTARSIA_QWEN25_DIR` point at the checkpoint directories.

use intarsia_core::engine::ChatOpts;
use intarsia_models::chat_template::{ChatTemplate, ChatTemplateSource};
use serde_json::json;
use std::path::PathBuf;

fn conversation() -> Vec<serde_json::Value> {
    vec![
        json!({"role": "system", "content": "You are terse."}),
        json!({"role": "user", "content": "Hi there"}),
        json!({"role": "assistant", "content": "Hello."}),
        json!({"role": "user", "content": "Capital of France?"}),
    ]
}

fn render(var: &str, extra: serde_json::Map<String, serde_json::Value>) -> Option<String> {
    let Some(dir) = std::env::var_os(var).map(PathBuf::from) else {
        eprintln!("SKIPPED: {var} is not set");
        return None;
    };
    let t = ChatTemplate::load(&dir, &ChatTemplateSource::Auto).unwrap().expect("the model has a chat template");
    let opts = ChatOpts { add_generation_prompt: true, enable_thinking: None, tools: None, extra };
    Some(t.render(&conversation(), &opts).unwrap())
}

#[test]
fn llama31_template_matches_transformers() {
    let mut extra = serde_json::Map::new();
    extra.insert("date_string".into(), json!("26 Jul 2024"));
    let Some(got) = render("INTARSIA_LLAMA31_DIR", extra) else { return };
    assert_eq!(
        got,
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are terse.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi there<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHello.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nCapital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    );
}

#[test]
fn qwen25_template_matches_transformers() {
    let Some(got) = render("INTARSIA_QWEN25_DIR", Default::default()) else { return };
    assert_eq!(
        got,
        "<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\nHello.<|im_end|>\n<|im_start|>user\nCapital of France?<|im_end|>\n<|im_start|>assistant\n"
    );
}
