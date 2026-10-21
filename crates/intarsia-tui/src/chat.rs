//! Streaming chat against the server's OpenAI endpoint (`POST /v1/chat/completions`, `stream: true`).

use anyhow::{anyhow, Result};
use intarsia_bench::http;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read};
use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
pub struct ChatMsg {
    pub role: Role,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    /// Local notes (errors, hints); never sent to the server.
    Note,
}

impl Role {
    pub fn api_name(self) -> Option<&'static str> {
        match self {
            Role::User => Some("user"),
            Role::Assistant => Some("assistant"),
            Role::Note => None,
        }
    }
}

/// What the stream thread reports back.
#[derive(Clone, Debug, PartialEq)]
pub enum ChatEvent {
    Token(String),
    /// Stream finished; `tokens` is the server's completion count when it sent usage, else the chunks received.
    Done { tokens: u64, secs: f64 },
    Error(String),
}

/// One parsed line of the SSE body.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamItem {
    Skip,
    Done,
    Text(String),
    Usage { completion_tokens: u64 },
    Error(String),
}

pub fn parse_stream_line(line: &str) -> StreamItem {
    let Some(payload) = line.strip_prefix("data:") else { return StreamItem::Skip };
    let payload = payload.trim();
    if payload == "[DONE]" {
        return StreamItem::Done;
    }
    let Ok(ev) = serde_json::from_str::<Value>(payload) else { return StreamItem::Skip };
    if let Some(err) = ev.get("error") {
        let msg = err["message"].as_str().map(str::to_string).unwrap_or_else(|| err.to_string());
        return StreamItem::Error(msg);
    }
    let mut text = String::new();
    if let Some(choices) = ev["choices"].as_array() {
        for c in choices {
            // reasoning_content is shown too: a thinking model's first tokens are part of what the engine is doing
            for k in ["reasoning_content", "reasoning", "content"] {
                if let Some(s) = c["delta"][k].as_str() {
                    text.push_str(s);
                }
            }
        }
    }
    if !text.is_empty() {
        return StreamItem::Text(text);
    }
    match ev["usage"]["completion_tokens"].as_u64() {
        Some(n) => StreamItem::Usage { completion_tokens: n },
        None => StreamItem::Skip,
    }
}

/// The request body for a chat turn.
pub fn request_body(model: &str, history: &[ChatMsg], max_tokens: usize, thinking: bool) -> Value {
    let messages: Vec<Value> = history
        .iter()
        .filter_map(|m| Some(json!({"role": m.role.api_name()?, "content": m.text})))
        .collect();
    let mut b = json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if !thinking {
        b["chat_template_kwargs"] = json!({"enable_thinking": false});
    }
    b
}

/// Run one streamed completion on the calling thread, sending [`ChatEvent`]s through `emit` until it returns false.
pub fn stream(origin: &str, body: &Value, mut emit: impl FnMut(ChatEvent) -> bool) -> Result<()> {
    let start = Instant::now();
    let agent = http::agent(None);
    let resp = agent
        .post(format!("{origin}/v1/chat/completions"))
        .header("Content-Type", "application/json")
        .send(body.to_string().as_str())
        .map_err(|e| anyhow!("request: {e}"))?;
    let status = resp.status().as_u16();
    let mut reader = resp.into_body().into_reader();
    if !(200..300).contains(&status) {
        let mut s = String::new();
        let _ = reader.by_ref().take(600).read_to_string(&mut s);
        return Err(anyhow!("HTTP {status}: {s}"));
    }
    let (mut chunks, mut usage) = (0u64, None);
    for line in BufReader::new(reader).lines() {
        match parse_stream_line(&line?) {
            StreamItem::Skip => {}
            StreamItem::Done => break,
            StreamItem::Error(e) => return Err(anyhow!("{e}")),
            StreamItem::Usage { completion_tokens } => usage = Some(completion_tokens),
            StreamItem::Text(t) => {
                chunks += 1;
                if !emit(ChatEvent::Token(t)) {
                    return Ok(());
                }
            }
        }
    }
    emit(ChatEvent::Done { tokens: usage.unwrap_or(chunks), secs: start.elapsed().as_secs_f64() });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_delta_usage_done_and_garbage() {
        let d = r#"data: {"choices":[{"index":0,"delta":{"content":"Hel"}}]}"#;
        assert_eq!(parse_stream_line(d), StreamItem::Text("Hel".into()));
        let role = r#"data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}"#;
        assert_eq!(parse_stream_line(role), StreamItem::Skip);
        let think = r#"data: {"choices":[{"delta":{"reasoning_content":"hm"}}]}"#;
        assert_eq!(parse_stream_line(think), StreamItem::Text("hm".into()));
        let u = r#"data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":17}}"#;
        assert_eq!(parse_stream_line(u), StreamItem::Usage { completion_tokens: 17 });
        assert_eq!(parse_stream_line("data: [DONE]"), StreamItem::Done);
        assert_eq!(parse_stream_line(": keep-alive"), StreamItem::Skip);
        assert_eq!(parse_stream_line("data: {not json"), StreamItem::Skip);
        let e = r#"data: {"error":{"message":"queue full","type":"rate_limit_error"}}"#;
        assert_eq!(parse_stream_line(e), StreamItem::Error("queue full".into()));
    }

    #[test]
    fn body_skips_notes_and_disables_thinking_by_default() {
        let h = vec![
            ChatMsg { role: Role::User, text: "hi".into() },
            ChatMsg { role: Role::Note, text: "local".into() },
            ChatMsg { role: Role::Assistant, text: "yo".into() },
            ChatMsg { role: Role::User, text: "again".into() },
        ];
        let b = request_body("m", &h, 64, false);
        let msgs = b["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(b["stream"], true);
        assert_eq!(b["chat_template_kwargs"]["enable_thinking"], false);
        assert!(request_body("m", &h, 64, true).get("chat_template_kwargs").is_none());
    }
}
