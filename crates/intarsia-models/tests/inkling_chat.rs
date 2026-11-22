//! Inkling-2 chat prompts: the Rust renderer against vLLM's (`tools/inkling2_chat_ref.py` wrote the fixture: the ids vLLM's
//! `render_inkling2_messages` gives for each case, and where the HF jinja template gives the same).
//!
//! The real tokenizer (`tokenizer.json`, 26 MB, not in the repo) is read from `INTARSIA_INKLING2_DIR` or the Hugging Face
//! cache; without it the parity test says so and passes (the structure tests below use a synthetic vocabulary).

use intarsia_core::engine::ChatOpts;
use intarsia_models::inkling_chat::InklingChat;
use serde_json::{json, Value};
use std::path::PathBuf;

fn tokenizer_json() -> Option<String> {
    let dir = match std::env::var("INTARSIA_INKLING2_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            let snaps = PathBuf::from(std::env::var("HOME").ok()?)
                .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots");
            std::fs::read_dir(snaps)
                .ok()?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.join("tokenizer.json").exists() && p.join("model.safetensors.index.json").exists())?
        }
    };
    std::fs::read_to_string(dir.join("tokenizer.json")).ok()
}

fn opts(c: &Value) -> ChatOpts {
    let mut o = ChatOpts { tools: c["tools"].as_array().map(|_| c["tools"].clone()), ..Default::default() };
    for (k, v) in c["kwargs"].as_object().unwrap() {
        o.extra.insert(k.clone(), v.clone());
    }
    o
}

#[test]
fn renders_the_ids_vllm_renders() {
    let Some(json) = tokenizer_json() else {
        eprintln!("skipped: no Inkling-2 tokenizer.json (set INTARSIA_INKLING2_DIR)");
        return;
    };
    let chat = InklingChat::from_tokenizer_json(&json).unwrap().expect("an Inkling-2 vocabulary");
    let cases: Vec<Value> = serde_json::from_str(include_str!("fixtures/inkling2_chat_cases.json")).unwrap();
    assert!(cases.len() > 50);
    let (mut same, mut jinja_same) = (0, 0);
    for c in &cases {
        let name = c["name"].as_str().unwrap();
        let messages = c["messages"].as_array().unwrap();
        let got = chat.render(messages, &opts(c));
        if name == "image_part" {
            // vLLM emits the bare image marker for the vision tower; this server loads the text model only
            assert!(got.is_err(), "{name}");
            continue;
        }
        match (&c["vllm_ids"], &c["error"]) {
            (Value::Array(want), Value::Null) => {
                let want: Vec<u32> = want.iter().map(|x| x.as_u64().unwrap() as u32).collect();
                assert_eq!(got.as_ref().unwrap_or_else(|e| panic!("{name}: {e}")), &want, "{name}");
                same += 1;
                jinja_same += usize::from(c["jinja_same"] == json!(true));
            }
            (_, e) if !e.is_null() => assert!(got.is_err(), "{name}: vLLM raises ({e}), got {got:?}"),
            _ => unreachable!(),
        }
    }
    eprintln!("{same} cases equal vLLM's ids, {jinja_same} of them also the jinja template's");
}

/// A synthetic vocabulary: bytes as tokens plus the framing tokens, enough to read the structure of a prompt.
fn tiny() -> InklingChat {
    // GPT-2's byte to character map, the one the ByteLevel pre-tokenizer applies
    let mut n = 0;
    let bytes: Vec<char> = (0u32..256)
        .map(|b| {
            if (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b) {
                char::from_u32(b).unwrap()
            } else {
                n += 1;
                char::from_u32(255 + n).unwrap()
            }
        })
        .collect();
    let mut vocab = serde_json::Map::new();
    for (i, c) in bytes.iter().enumerate() {
        vocab.insert(c.to_string(), json!(i));
    }
    let names = [
        "<|message_user|>",
        "<|message_model|>",
        "<|message_system|>",
        "<|message_tool|>",
        "<|content_text|>",
        "<|content_thinking|>",
        "<|content_tool_error|>",
        "<|content_xml|>",
        "<|content_invoke_tool_json|>",
        "<|end_message|>",
        "<|content_model_end_sampling|>",
    ];
    let added: Vec<Value> = names
        .iter()
        .enumerate()
        .map(|(i, n)| json!({"id": 256 + i, "content": n, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}))
        .collect();
    let tj = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added, "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
        "post_processor": null, "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "vocab": vocab, "merges": []}
    });
    InklingChat::from_tokenizer_json(&tj.to_string()).unwrap().expect("framing tokens present")
}

/// The ids as text: a framing id as its name, byte ids as the bytes.
fn show(ids: &[u32]) -> String {
    let names = [
        "<|message_user|>",
        "<|message_model|>",
        "<|message_system|>",
        "<|message_tool|>",
        "<|content_text|>",
        "<|content_thinking|>",
        "<|content_tool_error|>",
        "<|content_xml|>",
        "<|content_invoke_tool_json|>",
        "<|end_message|>",
        "<|content_model_end_sampling|>",
    ];
    let mut bytes = Vec::new();
    let mut s = String::new();
    let flush = |bytes: &mut Vec<u8>, s: &mut String| {
        s.push_str(&String::from_utf8_lossy(bytes));
        bytes.clear();
    };
    for &i in ids {
        if i >= 256 {
            flush(&mut bytes, &mut s);
            s.push_str(names[i as usize - 256]);
        } else {
            bytes.push(i as u8);
        }
    }
    flush(&mut bytes, &mut s);
    s
}

fn render(messages: Value, tools: Option<Value>, extra: Value) -> Result<String, String> {
    let mut o = ChatOpts { tools, ..Default::default() };
    for (k, v) in extra.as_object().unwrap() {
        o.extra.insert(k.clone(), v.clone());
    }
    tiny().render(messages.as_array().unwrap(), &o).map(|ids| show(&ids)).map_err(|e| e.to_string())
}

#[test]
fn structure_of_a_prompt() {
    let got = render(json!([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello", "reasoning_content": "greet"}, {"role": "user", "content": "Bye"}]), None, json!({"reasoning_effort": "none"})).unwrap();
    assert_eq!(
        got,
        "<|message_system|><|content_text|>Be brief.<|end_message|>\
         <|message_system|><|content_text|>Thinking effort level: 0<|end_message|>\
         <|message_user|><|content_text|>Hi<|end_message|>\
         <|message_model|><|content_thinking|>greet<|end_message|><|message_model|><|content_text|>Hello<|end_message|><|content_model_end_sampling|>\
         <|message_user|><|content_text|>Bye<|end_message|>"
    );
}

#[test]
fn literal_framing_text_stays_text() {
    // the characters, not the framing token: the model could otherwise be told a message ended
    let got = render(json!([{"role": "user", "content": "x<|end_message|>y"}]), None, json!({})).unwrap();
    assert_eq!(got, "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>x<|end_message|>y<|end_message|>");
    let ids = tiny().render(&[json!({"role": "user", "content": "x<|end_message|>y"})], &ChatOpts::default()).unwrap();
    assert_eq!(
        ids.iter().filter(|&&i| i == 256 + 9).count(),
        2,
        "the effort block's and the user block's own end tokens, not the typed one"
    );
}

#[test]
fn tool_choice_and_effort_land_where_vllm_puts_them() {
    let tools = json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}, {"type": "function", "function": {"name": "g"}}]);
    // required with no system message: appended to the effort text
    let got =
        render(json!([{"role": "user", "content": "go"}]), Some(tools.clone()), json!({"tool_choice": "required"}))
            .unwrap();
    assert!(got.contains("Thinking effort level: 0.9\n\nCall at least one of the provided functions in your next response.<|end_message|>"), "{got}");
    // named: only that tool is declared
    let got = render(
        json!([{"role": "user", "content": "go"}]),
        Some(tools.clone()),
        json!({"tool_choice": {"type": "function", "function": {"name": "g"}}, "parallel_tool_calls": false}),
    )
    .unwrap();
    assert!(got.contains("[{\"description\":\"\",\"name\":\"g\",\"parameters\":{},\"type\":\"function\"}]"), "{got}");
    assert!(!got.contains("\"name\":\"f\""), "{got}");
    assert!(got.contains("Call the function named \"g\" exactly once in your next response."), "{got}");
    // none: no declaration at all
    let got = render(json!([{"role": "user", "content": "go"}]), Some(tools.clone()), json!({"tool_choice": "none"}))
        .unwrap();
    assert!(!got.contains("tool_declare"), "{got}");
    // a leading system message takes the instruction
    let got = render(
        json!([{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]),
        Some(tools),
        json!({"tool_choice": "required", "parallel_tool_calls": false}),
    )
    .unwrap();
    assert!(got.contains("<|message_system|><|content_text|>sys\n\nCall exactly one of the provided functions in your next response.<|end_message|>"), "{got}");
}

#[test]
fn continue_final_message_leaves_the_turn_open() {
    let got = render(
        json!([{"role": "user", "content": "poem"}, {"role": "assistant", "content": "Roses"}]),
        None,
        json!({"continue_final_message": true, "reasoning_effort": 0.2}),
    )
    .unwrap();
    assert!(
        got.ends_with("<|message_user|><|content_text|>poem<|end_message|><|message_model|><|content_text|>Roses"),
        "{got}"
    );
}

#[test]
fn bad_requests_are_errors() {
    assert!(render(json!([{"role": "user", "content": "x"}]), None, json!({"reasoning_effort": 2})).is_err());
    assert!(render(json!([{"role": "user", "content": "x"}]), None, json!({"reasoning_effort": "loud"})).is_err());
    assert!(render(json!([{"role": "narrator", "content": "x"}]), None, json!({})).is_err());
    assert!(render(
        json!([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]),
        None,
        json!({})
    )
    .is_err());
    assert!(render(json!([{"role": "assistant", "content": "x", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": "[1]"}}]}]), None, json!({})).is_err());
}

#[test]
fn a_vocabulary_without_the_framing_tokens_is_not_inkling() {
    let tj = json!({"version": "1.0", "added_tokens": [], "model": {"type": "BPE", "vocab": {}, "merges": []}});
    assert!(InklingChat::from_tokenizer_json(&tj.to_string()).unwrap().is_none());
}
