//! Inkling-2 output: reasoning, answer and tool calls (`--reasoning-parser inkling` / `--tool-call-parser inkling` of vLLM,
//! `parser/inkling.py`; lane ink-serve, docs/results/2026-11-21-inkling2-serve.md).
//!
//! The model writes typed blocks, every marker a dedicated special token:
//!
//! ```text
//! <|message_model|><|content_thinking|>reasoning<|end_message|>
//! <|message_model|><|content_text|>answer<|end_message|>
//! <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|><|content_model_end_sampling|>
//! ```
//!
//! The server's text stream drops special tokens (`skip_special_tokens`), so unlike the `<think>` / `<tool_call>` parsers of
//! the Qwen models this one is driven by the token ids: [`Stream::push`] takes each token's id and the text it released, and
//! the ids move the state machine of vLLM's `inkling_config()` (message header, reasoning, content, tool arguments). Blocks may
//! come in any order and repeat; the optional function name between the role marker and the content kind is metadata.

use intarsia_core::Engine;
use serde_json::{json, Map, Value};

pub const MESSAGE_MODEL: &str = "<|message_model|>";
pub const CONTENT_TEXT: &str = "<|content_text|>";
pub const CONTENT_THINKING: &str = "<|content_thinking|>";
pub const END_MESSAGE: &str = "<|end_message|>";
pub const CONTENT_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
pub const CONTENT_INVOKE_TOOL_TEXT: &str = "<|content_invoke_tool_text|>";
pub const CONTENT_TOOL_ERROR: &str = "<|content_tool_error|>";
pub const END_SAMPLING: &str = "<|content_model_end_sampling|>";

/// The ids of the block markers of the served vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ids {
    pub model: u32,
    pub text: u32,
    pub thinking: u32,
    pub end_message: u32,
    pub tool_json: u32,
    pub tool_text: u32,
    pub tool_error: u32,
    pub end_sampling: u32,
}

impl Ids {
    /// Each marker is one token of the engine's tokenizer; `None` when any is not.
    pub fn from_engine(e: &dyn Engine) -> Option<Ids> {
        let one = |s: &str| match e.tokenize(s, false) {
            Ok(v) if v.len() == 1 => Some(v[0]),
            _ => None,
        };
        Some(Ids {
            model: one(MESSAGE_MODEL)?,
            text: one(CONTENT_TEXT)?,
            thinking: one(CONTENT_THINKING)?,
            end_message: one(END_MESSAGE)?,
            tool_json: one(CONTENT_INVOKE_TOOL_JSON)?,
            tool_text: one(CONTENT_INVOKE_TOOL_TEXT)?,
            tool_error: one(CONTENT_TOOL_ERROR)?,
            end_sampling: one(END_SAMPLING)?,
        })
    }

    fn is_marker(&self, id: u32) -> bool {
        [
            self.model,
            self.text,
            self.thinking,
            self.end_message,
            self.tool_json,
            self.tool_text,
            self.tool_error,
            self.end_sampling,
        ]
        .contains(&id)
    }
}

/// A piece of the answer, in order.
#[derive(Clone, Debug, PartialEq)]
pub enum Out {
    Reasoning(String),
    Content(String),
    /// A complete call; `arguments` is the `args` object the model wrote (a string when it wrote one).
    Call {
        name: String,
        arguments: Value,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// After `<|message_model|>`: the function name, then the content kind.
    Header,
    Reasoning,
    Content,
    ToolArgs,
}

/// vLLM's parser engine for Inkling, over (token id, released text) events.
pub struct Stream {
    ids: Ids,
    state: State,
    header: String,
    tool: String,
    /// The function name of the open call block (its header), the fallback when the JSON has no `name`.
    tool_name: String,
    /// Whitespace at the end of the reasoning so far: dropped if the block ends there (`strip_trailing_reasoning_whitespace`).
    reason_ws: String,
    /// Whitespace-only content so far: dropped when a call follows it (`drop_whitespace_only_content_before_tools`).
    content_ws: String,
    content_started: bool,
    called: bool,
    last: Option<u32>,
}

fn trailing_ws_start(s: &str) -> usize {
    s.trim_end_matches(char::is_whitespace).len()
}

impl Stream {
    pub fn new(ids: Ids) -> Stream {
        // the prompt ends between messages: the model opens its own `<|message_model|>`
        Stream {
            ids,
            state: State::Header,
            header: String::new(),
            tool: String::new(),
            tool_name: String::new(),
            reason_ws: String::new(),
            content_ws: String::new(),
            content_started: false,
            called: false,
            last: None,
        }
    }

    /// One generated token with the text the engine released for it (empty for a marker).
    pub fn push(&mut self, id: u32, text: &str) -> Vec<Out> {
        let mut out = Vec::new();
        // the extra event that releases held-back text repeats the last id: its text belongs to the stream, its id was seen
        let repeat = !text.is_empty() && self.last == Some(id);
        self.last = Some(id);
        if !text.is_empty() {
            self.text(text, &mut out);
        }
        if self.ids.is_marker(id) && !repeat {
            self.marker(id, &mut out);
        }
        out
    }

    /// Markers and text from a decoded run: `Some(id)` for a marker, else the text.
    pub fn push_text(&mut self, text: &str) -> Vec<Out> {
        let mut out = Vec::new();
        self.text(text, &mut out);
        out
    }

    pub fn push_marker(&mut self, id: u32) -> Vec<Out> {
        let mut out = Vec::new();
        if self.ids.is_marker(id) {
            self.marker(id, &mut out);
        }
        out
    }

    fn text(&mut self, t: &str, out: &mut Vec<Out>) {
        match self.state {
            State::Header => self.header.push_str(t),
            State::ToolArgs => self.tool.push_str(t),
            State::Reasoning => {
                let s = format!("{}{t}", std::mem::take(&mut self.reason_ws));
                let keep = trailing_ws_start(&s);
                if keep > 0 {
                    out.push(Out::Reasoning(s[..keep].to_string()));
                }
                self.reason_ws = s[keep..].to_string();
            }
            State::Content => {
                let s = format!("{}{t}", std::mem::take(&mut self.content_ws));
                if !self.content_started && s.trim().is_empty() {
                    self.content_ws = s;
                } else {
                    self.content_started = true;
                    out.push(Out::Content(s));
                }
            }
        }
    }

    fn marker(&mut self, id: u32, out: &mut Vec<Out>) {
        let ids = self.ids;
        if id == ids.model {
            self.leave_block(out, false);
            self.state = State::Header;
            self.header.clear();
        } else if id == ids.thinking {
            self.leave_block(out, false);
            self.state = State::Reasoning;
            self.header.clear();
        } else if id == ids.text || id == ids.tool_text || id == ids.tool_error {
            // raw / error tool blocks read as visible text
            self.leave_block(out, false);
            self.state = State::Content;
            self.header.clear();
        } else if id == ids.tool_json {
            let name = self.header.trim().to_string();
            self.leave_block(out, false);
            self.tool_name = name;
            self.header.clear();
            self.tool.clear();
            self.state = State::ToolArgs;
        } else if id == ids.end_message || id == ids.end_sampling {
            self.leave_block(out, true);
            self.state = State::Content;
        }
    }

    /// The open block ends (its closing marker, or the next block's opener).
    fn leave_block(&mut self, out: &mut Vec<Out>, flush_header: bool) {
        match self.state {
            State::Reasoning => self.reason_ws.clear(),
            State::ToolArgs => {
                let raw = std::mem::take(&mut self.tool);
                match parse_call(&raw, &self.tool_name) {
                    Some((name, arguments)) => {
                        // whitespace-only content before the call is not content
                        self.content_ws.clear();
                        self.content_started = false;
                        self.called = true;
                        out.push(Out::Call { name, arguments });
                    }
                    // not a call: the block's text is the answer
                    None => out.push(Out::Content(raw)),
                }
            }
            State::Header if flush_header && !self.header.is_empty() => {
                // a header with no block after it: its text is content
                let h = std::mem::take(&mut self.header);
                self.content_started = true;
                out.push(Out::Content(h));
            }
            State::Header => self.header.clear(),
            _ => {}
        }
        self.tool_name.clear();
    }

    /// End of the stream: whatever is held back.
    pub fn finish(&mut self) -> Vec<Out> {
        let mut out = Vec::new();
        self.leave_block(&mut out, true);
        self.state = State::Content;
        let ws = std::mem::take(&mut self.content_ws);
        if !ws.is_empty() && !self.called {
            out.push(Out::Content(ws));
        }
        out
    }

    /// The whole output at once from its ids; `decode` turns a run of non-marker ids into text.
    pub fn parse_ids(ids: Ids, token_ids: &[u32], decode: &mut dyn FnMut(&[u32]) -> String) -> Vec<Out> {
        let mut s = Stream::new(ids);
        let mut out = Vec::new();
        let mut run = 0;
        for (i, &id) in token_ids.iter().enumerate() {
            if ids.is_marker(id) {
                if run < i {
                    out.extend(s.push_text(&decode(&token_ids[run..i])));
                }
                out.extend(s.push_marker(id));
                run = i + 1;
            }
        }
        if run < token_ids.len() {
            out.extend(s.push_text(&decode(&token_ids[run..])));
        }
        out.extend(s.finish());
        out
    }
}

/// `{"name": NAME, "args": {...}}` as a call. The name falls back to the block's header; a payload that is not a JSON object,
/// or names no function, is not a call.
fn parse_call(raw: &str, header_name: &str) -> Option<(String, Value)> {
    let v: Value = serde_json::from_str(raw.trim()).ok()?;
    let o = v.as_object()?;
    let name = o
        .get("name")
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty())
        .map(String::from)
        .or_else(|| (!header_name.is_empty()).then(|| header_name.to_string()))?;
    // `args`, as vLLM reads it (`arguments` and `parameters` too)
    let args = ["args", "arguments", "parameters"]
        .iter()
        .find_map(|k| o.get(*k))
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    Some((name, args))
}

/// The `arguments` string of an OpenAI tool call.
pub fn arguments_string(arguments: &Value) -> String {
    match arguments {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// The whole reply: reasoning, content and calls. Content is `None` when only whitespace is left and calls were made.
pub struct Reply {
    pub reasoning: Option<String>,
    pub content: Option<String>,
    pub calls: Vec<(String, Value)>,
}

pub fn collect(outs: Vec<Out>) -> Reply {
    let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), Vec::new());
    for o in outs {
        match o {
            Out::Reasoning(t) => reasoning.push_str(&t),
            Out::Content(t) => content.push_str(&t),
            Out::Call { name, arguments } => calls.push((name, arguments)),
        }
    }
    let content = if calls.is_empty() { Some(content) } else { Some(content).filter(|c| !c.trim().is_empty()) };
    Reply { reasoning: (!reasoning.is_empty()).then_some(reasoning), content, calls }
}

/// Streaming delta objects for `outs`; `n` counts the calls so far (their `index`), `new_id` makes a call id.
pub fn deltas(outs: Vec<Out>, n: &mut usize, new_id: &dyn Fn() -> String) -> Vec<Value> {
    let mut v = Vec::new();
    for o in outs {
        match o {
            Out::Reasoning(t) if !t.is_empty() => v.push(json!({"reasoning_content": t})),
            Out::Content(t) if !t.is_empty() => v.push(json!({"content": t})),
            Out::Call { name, arguments } => {
                let c = json!({"index": *n, "id": new_id(), "type": "function", "function": {"name": name, "arguments": arguments_string(&arguments)}});
                *n += 1;
                v.push(json!({"tool_calls": [c]}));
            }
            _ => {}
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDS: Ids = Ids {
        model: 1,
        text: 2,
        thinking: 3,
        end_message: 4,
        tool_json: 5,
        tool_text: 6,
        tool_error: 7,
        end_sampling: 8,
    };

    /// Events: a marker id, or text as a run of ids 100+ with the text on the first.
    fn run(events: &[(u32, &str)]) -> Vec<Out> {
        let mut s = Stream::new(IDS);
        let mut out = Vec::new();
        for &(id, t) in events {
            out.extend(s.push(id, t));
        }
        out.extend(s.finish());
        out
    }

    fn joined(outs: &[Out]) -> (String, String, Vec<(String, Value)>) {
        let r = collect(outs.to_vec());
        (r.reasoning.unwrap_or_default(), r.content.unwrap_or_default(), r.calls)
    }

    #[test]
    fn reasoning_then_answer() {
        let outs = run(&[
            (1, ""),
            (3, ""),
            (100, "The user asks"),
            (101, " a question.\n"),
            (4, ""),
            (1, ""),
            (2, ""),
            (102, "Paris"),
            (103, "."),
            (4, ""),
            (8, ""),
        ]);
        assert_eq!(joined(&outs), ("The user asks a question.".to_string(), "Paris.".to_string(), vec![]));
        // the reasoning streams as it comes, without the whitespace it ends in
        assert_eq!(outs[0], Out::Reasoning("The user asks".into()));
        assert!(outs.contains(&Out::Reasoning(" a question.".into())), "{outs:?}");
    }

    #[test]
    fn effort_none_may_skip_the_thinking_block() {
        assert_eq!(
            joined(&run(&[(1, ""), (2, ""), (100, "Hi"), (4, ""), (8, "")])),
            (String::new(), "Hi".to_string(), vec![])
        );
    }

    #[test]
    fn a_call_with_its_header_name() {
        let outs = run(&[
            (1, ""),
            (100, "get_weather"),
            (5, ""),
            (101, "{\"name\":\"get_weather\",\"args\":{\"city\":"),
            (102, "\"Paris\"}}"),
            (4, ""),
            (8, ""),
        ]);
        let (reasoning, content, calls) = joined(&outs);
        assert_eq!((reasoning.as_str(), content.as_str()), ("", ""));
        assert_eq!(calls, vec![("get_weather".to_string(), json!({"city": "Paris"}))]);
    }

    #[test]
    fn text_after_a_call_is_kept_and_whitespace_before_it_is_not() {
        let outs = run(&[
            (1, ""),
            (2, ""),
            (100, "\n\n"),
            (4, ""),
            (1, ""),
            (5, ""),
            (101, "{\"name\":\"f\",\"args\":{}}"),
            (4, ""),
            (1, ""),
            (2, ""),
            (102, "done"),
            (4, ""),
            (8, ""),
        ]);
        let (_, content, calls) = joined(&outs);
        assert_eq!(content, "done");
        assert_eq!(calls, vec![("f".to_string(), json!({}))]);
    }

    #[test]
    fn trailing_whitespace_with_a_call_is_dropped_without_one_kept() {
        let with_call = run(&[
            (1, ""),
            (5, ""),
            (100, "{\"name\":\"f\",\"args\":{}}"),
            (4, ""),
            (1, ""),
            (2, ""),
            (101, "\n"),
            (4, ""),
        ]);
        assert_eq!(joined(&with_call).1, "");
        let without = run(&[(1, ""), (2, ""), (101, "\n"), (4, "")]);
        assert_eq!(joined(&without).1, "\n");
    }

    #[test]
    fn two_thinking_blocks_and_raw_tool_blocks() {
        let outs = run(&[
            (1, ""),
            (3, ""),
            (100, "a "),
            (4, ""),
            (1, ""),
            (3, ""),
            (101, "b"),
            (4, ""),
            (1, ""),
            (6, ""),
            (102, "raw text"),
            (4, ""),
        ]);
        assert_eq!(joined(&outs), ("ab".to_string(), "raw text".to_string(), vec![]));
    }

    #[test]
    fn a_broken_call_is_text_and_a_cut_off_one_too() {
        let outs = run(&[(1, ""), (5, ""), (100, "{\"name\":"), (4, "")]);
        assert_eq!(joined(&outs), (String::new(), "{\"name\":".to_string(), vec![]));
        let cut = run(&[(1, ""), (5, ""), (100, "{\"name\":\"f\",\"args\":{\"a\":")]);
        assert_eq!(joined(&cut).2, vec![]);
    }

    #[test]
    fn the_repeat_of_the_last_id_releases_text_without_a_second_transition() {
        // `<|content_thinking|>` then held text released on a repeat of that id
        let outs = run(&[(1, ""), (3, ""), (3, "held"), (100, " more")]);
        assert_eq!(joined(&outs).0, "held more");
        // a repeated `<|content_invoke_tool_json|>` would reset the call buffer
        let outs = run(&[(1, ""), (5, ""), (100, "{\"name\":\"f\","), (100, "\"args\":{}}"), (4, "")]);
        assert_eq!(joined(&outs).2.len(), 1);
    }

    #[test]
    fn arguments_keep_a_string_the_model_wrote() {
        let outs = run(&[(1, ""), (5, ""), (100, "{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":1}\"}"), (4, "")]);
        assert_eq!(arguments_string(&joined(&outs).2[0].1), "{\"a\":1}");
        let outs = run(&[(1, ""), (5, ""), (100, "{\"name\":\"f\"}"), (4, "")]);
        assert_eq!(arguments_string(&joined(&outs).2[0].1), "{}");
    }

    #[test]
    fn whole_output_from_ids_equals_the_stream() {
        let ids = [1, 3, 100, 101, 4, 1, 2, 102, 4, 1, 5, 103, 4, 8];
        let text = |i: u32| match i {
            100 => "think",
            101 => " hard ",
            102 => "Paris",
            _ => "{\"name\":\"f\",\"args\":{\"x\":1}}",
        };
        let mut s = Stream::new(IDS);
        let mut streamed = Vec::new();
        for &i in &ids {
            streamed.extend(s.push(i, if IDS.is_marker(i) { "" } else { text(i) }));
        }
        streamed.extend(s.finish());
        let whole = Stream::parse_ids(IDS, &ids, &mut |run: &[u32]| run.iter().map(|&i| text(i)).collect::<String>());
        assert_eq!(joined(&whole), joined(&streamed));
        assert_eq!(
            joined(&whole),
            ("think hard".to_string(), "Paris".to_string(), vec![("f".to_string(), json!({"x": 1}))])
        );
    }
}
