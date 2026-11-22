//! Inkling-2 chat prompts as token ids (lane ink-serve, docs/results/2026-11-21-inkling2-serve.md).
//!
//! A port of vLLM's `render_inkling2_messages` (`renderers/inkling_encoding.py`), the renderer that serves this model: every
//! block is `<role token>[author name]<content-kind token>text<|end_message|>`, text is encoded *with the special tokens
//! switched off* (`split_special_tokens=True`: a user typing `<|end_message|>` gets its characters, not the framing token),
//! and the prompt has no generation prefix (the model opens its own `<|message_model|>`). The HF `chat_template.jinja` renders
//! the same stream for plain, system, multi-turn, effort and tool prompts; it cannot do what this adds on top of it: literal
//! special strings staying text, `developer` messages, `tool_choice` (none hides the tools, a named choice declares that one,
//! `required` / `parallel_tool_calls: false` become an instruction text), `continue_final_message`, string tool arguments,
//! and the reference's JSON spelling of floats.

use intarsia_core::engine::ChatOpts;
use intarsia_core::{Error, Result};
use serde_json::{Map, Value};
use std::collections::HashMap;
use tokenizers::Tokenizer;

const MESSAGE_USER: &str = "<|message_user|>";
const MESSAGE_MODEL: &str = "<|message_model|>";
const MESSAGE_SYSTEM: &str = "<|message_system|>";
const MESSAGE_TOOL: &str = "<|message_tool|>";
const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_THINKING: &str = "<|content_thinking|>";
const CONTENT_TOOL_ERROR: &str = "<|content_tool_error|>";
const CONTENT_XML: &str = "<|content_xml|>";
const CONTENT_XML_ALT: &str = "<|unused_200024|>";
const CONTENT_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
const END_MESSAGE: &str = "<|end_message|>";
const END_SAMPLING: &str = "<|content_model_end_sampling|>";

pub const DEFAULT_EFFORT: f64 = 0.9;
const MAX_EFFORT: f64 = 0.99;

fn bad(m: impl Into<String>) -> Error {
    Error::Text(m.into())
}

/// The ids of the framing tokens of an Inkling-2 vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InklingIds {
    pub user: u32,
    pub model: u32,
    pub system: u32,
    pub tool: u32,
    pub text: u32,
    pub thinking: u32,
    pub tool_error: u32,
    pub xml: u32,
    pub invoke_tool_json: u32,
    pub end_message: u32,
    pub end_sampling: u32,
}

impl InklingIds {
    /// Resolve the ids through `id_of` (token string to id); `None` when the vocabulary is not Inkling-2's.
    pub fn resolve(id_of: impl Fn(&str) -> Option<u32>) -> Option<InklingIds> {
        Some(InklingIds {
            user: id_of(MESSAGE_USER)?,
            model: id_of(MESSAGE_MODEL)?,
            system: id_of(MESSAGE_SYSTEM)?,
            tool: id_of(MESSAGE_TOOL)?,
            text: id_of(CONTENT_TEXT)?,
            thinking: id_of(CONTENT_THINKING)?,
            tool_error: id_of(CONTENT_TOOL_ERROR)?,
            xml: id_of(CONTENT_XML).or_else(|| id_of(CONTENT_XML_ALT))?,
            invoke_tool_json: id_of(CONTENT_INVOKE_TOOL_JSON)?,
            end_message: id_of(END_MESSAGE)?,
            end_sampling: id_of(END_SAMPLING)?,
        })
    }
}

/// The renderer: the framing ids and a tokenizer without added tokens (so no text is ever read as a framing token).
pub struct InklingChat {
    pub ids: InklingIds,
    plain: Tokenizer,
}

impl InklingChat {
    /// From the text of `tokenizer.json`: `Some` when it is an Inkling-2 vocabulary (it has every framing token).
    pub fn from_tokenizer_json(json: &str) -> Result<Option<InklingChat>> {
        let mut v: Value = serde_json::from_str(json).map_err(|e| Error::Text(format!("tokenizer.json: {e}")))?;
        let added: HashMap<String, u32> = v
            .get("added_tokens")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|t| {
                        Some((t.get("content")?.as_str()?.to_string(), u32::try_from(t.get("id")?.as_u64()?).ok()?))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let Some(ids) = InklingIds::resolve(|s| added.get(s).copied()) else { return Ok(None) };
        // the same vocabulary with its added tokens removed: the BPE of `<|end_message|>` is plain characters
        v["added_tokens"] = Value::Array(Vec::new());
        let plain =
            Tokenizer::from_bytes(v.to_string().as_bytes()).map_err(|e| Error::Text(format!("tokenizer.json: {e}")))?;
        Ok(Some(InklingChat { ids, plain }))
    }

    fn text(&self, out: &mut Vec<u32>, s: &str) -> Result<()> {
        if s.is_empty() {
            return Ok(());
        }
        let enc = self.plain.encode(s, false).map_err(|e| Error::Text(e.to_string()))?;
        out.extend_from_slice(enc.get_ids());
        Ok(())
    }

    fn role(&self, role: &str) -> u32 {
        match role {
            "user" => self.ids.user,
            "assistant" => self.ids.model,
            "tool" => self.ids.tool,
            _ => self.ids.system, // system, developer
        }
    }

    fn kind(&self, kind: &str) -> u32 {
        match kind {
            "thinking" => self.ids.thinking,
            "tool_error" => self.ids.tool_error,
            "xml" => self.ids.xml,
            "invoke_tool_json" => self.ids.invoke_tool_json,
            _ => self.ids.text,
        }
    }

    fn block(&self, out: &mut Vec<u32>, role: &str, kind: &str, text: &str, author: Option<&str>) -> Result<()> {
        out.push(self.role(role));
        if let Some(a) = author.filter(|a| !a.is_empty()) {
            self.text(out, a)?;
        }
        out.push(self.kind(kind));
        self.text(out, text)?;
        out.push(self.ids.end_message);
        Ok(())
    }

    /// Render `messages` (OpenAI chat messages) to input ids. Variables are read from `opts`: `tools`, and from `opts.extra`
    /// `reasoning_effort`, `tool_choice`, `parallel_tool_calls` and `continue_final_message`.
    pub fn render(&self, messages: &[Value], opts: &ChatOpts) -> Result<Vec<u32>> {
        let extra = &opts.extra;
        let effort = resolve_effort(extra.get("reasoning_effort"))?;
        let parallel = extra.get("parallel_tool_calls").and_then(Value::as_bool).unwrap_or(true);
        let continue_final = extra.get("continue_final_message").and_then(Value::as_bool).unwrap_or(false);
        let choice = Choice::parse(extra.get("tool_choice"))?;

        let mut messages: Vec<Value> = messages.to_vec();
        let mut prefix = None;
        if continue_final {
            if let Some(p) = messages.last().and_then(continuation_prefix) {
                prefix = Some(p);
                messages.pop();
            }
        }

        let mut out = Vec::new();
        let mut declared =
            collect_tools(&messages, if choice == Choice::Mode("none".into()) { None } else { opts.tools.as_ref() })?;
        if let Choice::Named(name) = &choice {
            declared.retain(|t| function_of(t).get("name").and_then(Value::as_str) == Some(name.as_str()));
        }
        if !declared.is_empty() {
            self.block(&mut out, "system", "xml", &tool_declare_json(&declared)?, Some("tool_declare"))?;
        }

        let mut effort_text = Some(effort_text(effort)?);
        let instruction = if declared.is_empty() { None } else { choice_instruction(&choice, parallel) };
        if let Some(instruction) = instruction {
            let first_is_system = messages
                .first()
                .and_then(|m| m.get("role"))
                .and_then(Value::as_str)
                .is_some_and(|r| r == "system" || r == "developer");
            if first_is_system {
                messages[0] = with_appended_text(&messages[0], &instruction);
            } else if let Some(t) = effort_text.as_mut() {
                *t = format!("{t}\n\n{instruction}");
            }
        }

        let mut call_names: HashMap<String, String> = HashMap::new();
        for message in &messages {
            let role = expect_role(message)?;
            if role != "system" && role != "developer" {
                if let Some(t) = effort_text.take() {
                    self.block(&mut out, "system", "text", &t, None)?;
                }
            }
            let author =
                (role == "tool").then(|| match message.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()) {
                    Some(n) => Some(n.to_string()),
                    None => {
                        message.get("tool_call_id").and_then(Value::as_str).and_then(|id| call_names.get(id)).cloned()
                    }
                });
            let author = author.flatten();
            let blocks = blocks_of(message, role, &mut call_names)?;
            for (kind, text, block_author) in &blocks {
                self.block(&mut out, role, kind, text, block_author.as_deref().or(author.as_deref()))?;
            }
            if role == "assistant" && !blocks.is_empty() {
                out.push(self.ids.end_sampling);
            }
        }
        if let Some(t) = effort_text.take() {
            self.block(&mut out, "system", "text", &t, None)?;
        }
        if let Some(p) = prefix {
            out.push(self.ids.model);
            out.push(self.ids.text);
            self.text(&mut out, &p)?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// request variables
// ---------------------------------------------------------------------------------------------------------------------

/// The named efforts of the template (`chat_template.jinja`'s `effort_map`).
fn named_effort(s: &str) -> Option<f64> {
    Some(match s {
        "none" => 0.0,
        "minimal" => 0.1,
        "low" => 0.2,
        "medium" => 0.7,
        "high" => 0.9,
        "xhigh" | "max" => 0.99,
        _ => return None,
    })
}

/// `reasoning_effort` as a number: unset is 0.9, a name or a numeric string or a number is itself, anything else an error.
pub fn resolve_effort(v: Option<&Value>) -> Result<f64> {
    let invalid = |v: &Value| bad(format!("invalid Inkling reasoning_effort: {v}"));
    match v {
        None | Some(Value::Null) => Ok(DEFAULT_EFFORT),
        Some(Value::String(s)) => match named_effort(s) {
            Some(e) => Ok(e),
            None => s.trim().parse::<f64>().map_err(|_| invalid(v.unwrap())),
        },
        Some(Value::Number(n)) => n.as_f64().ok_or_else(|| invalid(v.unwrap())),
        Some(o) => Err(invalid(o)),
    }
}

/// Python's `round()`: half to even.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// `Thinking effort level: E`: the effort goes through integer milli-effort, then two decimals, printed as `:g`.
pub fn effort_text(effort: f64) -> Result<String> {
    if !(0.0..=MAX_EFFORT).contains(&effort) {
        return Err(bad(format!("Inkling reasoning_effort must be in [0.0, 0.99], got {effort}")));
    }
    let milli = round_half_even(effort * 1000.0);
    let hundredths = round_half_even((milli / 1000.0) * 100.0);
    let e = hundredths / 100.0;
    Ok(format!("Thinking effort level: {}", if e == 0.0 { "0".to_string() } else { e.to_string() }))
}

#[derive(Debug, PartialEq, Eq)]
enum Choice {
    Mode(String),
    Named(String),
}

impl Choice {
    fn parse(v: Option<&Value>) -> Result<Choice> {
        match v {
            None | Some(Value::Null) => Ok(Choice::Mode("auto".into())),
            Some(Value::String(s)) => Ok(Choice::Mode(s.clone())),
            Some(Value::Object(o)) if o.get("type").and_then(Value::as_str) == Some("function") => {
                let name = match o.get("function").filter(|f| !f.is_null()) {
                    Some(f) => f.get("name"),
                    None => o.get("name"),
                };
                name.and_then(Value::as_str)
                    .map(|n| Choice::Named(n.to_string()))
                    .ok_or_else(|| bad("tool_choice names no function"))
            }
            Some(o) => Err(bad(format!("unsupported tool_choice: {o}"))),
        }
    }
}

fn choice_instruction(choice: &Choice, parallel: bool) -> Option<String> {
    match choice {
        Choice::Named(name) => {
            let once = if parallel { "" } else { " exactly once" };
            Some(format!("Call the function named {}{once} in your next response.", Value::String(name.clone())))
        }
        Choice::Mode(m) if m == "required" => Some(format!(
            "Call {} of the provided functions in your next response.",
            if parallel { "at least one" } else { "exactly one" }
        )),
        Choice::Mode(m) if m == "auto" && !parallel => {
            Some("If you call a function, call at most one function in your next response.".into())
        }
        _ => None,
    }
}

fn with_appended_text(message: &Value, text: &str) -> Value {
    let mut m = message.clone();
    let content = match message.get("content") {
        Some(Value::String(s)) if !s.is_empty() => Value::String(format!("{s}\n\n{text}")),
        Some(Value::Array(a)) if !a.is_empty() => {
            let mut a = a.clone();
            a.push(serde_json::json!({"type": "text", "text": text}));
            Value::Array(a)
        }
        _ => Value::String(text.to_string()),
    };
    m["content"] = content;
    m
}

fn expect_role(m: &Value) -> Result<&str> {
    match m.get("role").and_then(Value::as_str) {
        Some(r @ ("user" | "assistant" | "system" | "developer" | "tool")) => Ok(r),
        other => Err(bad(format!("unsupported Inkling message role {other:?}; expected one of [\"assistant\", \"developer\", \"system\", \"tool\", \"user\"]"))),
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// messages to blocks
// ---------------------------------------------------------------------------------------------------------------------

type Block = (&'static str, String, Option<String>);

fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
    }
}

fn assistant_reasoning(m: &Value) -> Option<&Value> {
    match m.get("reasoning") {
        None | Some(Value::Null) => m.get("reasoning_content"),
        r => r,
    }
}

/// The text a `continue_final_message` request extends: a plain-string assistant message with no calls and no reasoning.
fn continuation_prefix(m: &Value) -> Option<String> {
    if m.get("role").and_then(Value::as_str) != Some("assistant")
        || truthy(m.get("tool_calls"))
        || truthy(assistant_reasoning(m))
    {
        return None;
    }
    m.get("content").and_then(Value::as_str).map(String::from)
}

fn parts_of(content: Option<&Value>) -> Result<Vec<(&'static str, String)>> {
    let mut out = Vec::new();
    match content {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => out.push(("text", s.clone())),
        Some(Value::Array(parts)) => {
            for p in parts {
                match p {
                    Value::String(s) => out.push(("text", s.clone())),
                    Value::Object(o) => match o.get("type").and_then(Value::as_str) {
                        None | Some("text" | "input_text") => {
                            out.push(("text", o.get("text").and_then(Value::as_str).unwrap_or("").to_string()))
                        }
                        Some("thinking") => {
                            out.push(("thinking", o.get("thinking").and_then(Value::as_str).unwrap_or("").to_string()))
                        }
                        Some("tool_error") => {
                            out.push(("tool_error", o.get("text").and_then(Value::as_str).unwrap_or("").to_string()))
                        }
                        Some("image" | "input_image" | "image_url") => {
                            return Err(bad(
                                "this server runs Inkling-2-Mini's text model only: image parts are not supported",
                            ))
                        }
                        Some(other) => return Err(bad(format!("unsupported Inkling2 content part type: {other:?}"))),
                    },
                    _ => return Err(bad("content part must be an object or a string")),
                }
            }
        }
        Some(_) => return Err(bad("message content must be a string or a sequence of parts")),
    }
    Ok(out)
}

fn blocks_of(m: &Value, role: &str, call_names: &mut HashMap<String, String>) -> Result<Vec<Block>> {
    let content = m.get("content");
    if role != "assistant" {
        return Ok(parts_of(content)?.into_iter().map(|(k, t)| (k, t, None)).collect());
    }
    let reasoning = assistant_reasoning(m);
    if let Some(r) = reasoning.filter(|r| !r.is_null()) {
        if !r.is_string() {
            return Err(bad("assistant reasoning_content must be a string"));
        }
    }
    let calls = m.get("tool_calls").and_then(Value::as_array).cloned().unwrap_or_default();
    // `content == "" and (tool_calls or reasoning)` is no content
    let content = if content.and_then(Value::as_str) == Some("") && (!calls.is_empty() || truthy(reasoning)) {
        None
    } else {
        content
    };
    let mut blocks: Vec<Block> = Vec::new();
    if let Some(r) = reasoning.and_then(Value::as_str).filter(|r| !r.is_empty()) {
        blocks.push(("thinking", r.to_string(), None));
    }
    blocks.extend(parts_of(content)?.into_iter().map(|(k, t)| (k, t, None)));
    for call in &calls {
        let function =
            call.get("function").filter(|f| f.is_object()).ok_or_else(|| bad("tool call has no function"))?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("tool call function name must be a string"))?;
        let args = match function.get("arguments") {
            None | Some(Value::Null) => Value::Object(Map::new()),
            Some(Value::String(s)) if s.trim().is_empty() => Value::Object(Map::new()),
            Some(Value::String(s)) => {
                serde_json::from_str(s).map_err(|e| bad(format!("tool call arguments are not JSON: {e}")))?
            }
            Some(v) => v.clone(),
        };
        let Value::Object(args) = args else {
            return Err(bad("tool call function arguments must decode to an object"));
        };
        if let Some(id) = call.get("id").and_then(Value::as_str).filter(|i| !i.is_empty()) {
            call_names.insert(id.to_string(), name.to_string());
        }
        // the top-level keys sorted, everything below in the order the client sent
        let mut keys: Vec<&String> = args.keys().collect();
        keys.sort();
        let mut s = String::from("{");
        for (i, k) in keys.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&json_str(k));
            s.push(':');
            tml_json(&args[k.as_str()], false, false, &mut s)?;
        }
        s.push('}');
        blocks.push((
            "invoke_tool_json",
            format!("{{\"name\":{},\"args\":{}}}", json_str(name), s),
            Some(name.to_string()),
        ));
    }
    Ok(blocks)
}

fn function_of(tool: &Value) -> &Value {
    static EMPTY: Value = Value::Null;
    tool.get("function").unwrap_or(&EMPTY)
}

/// Request-level tools plus the tools of each `developer` message, declared in one leading system block.
fn collect_tools(messages: &[Value], tools: Option<&Value>) -> Result<Vec<Value>> {
    let mut all: Vec<Value> = match tools {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a.clone(),
        Some(_) => return Err(bad("tools must be an array")),
    };
    for m in messages {
        if m.get("role").and_then(Value::as_str) == Some("developer") {
            if let Some(Value::Array(a)) = m.get("tools") {
                all.extend(a.iter().cloned());
            }
        }
    }
    Ok(all)
}

fn tool_declare_json(tools: &[Value]) -> Result<String> {
    let mut specs = Vec::new();
    for t in tools {
        let Some(function) = t.get("function").and_then(Value::as_object) else {
            return Err(bad("tool has no function"));
        };
        let parameters = match function.get("parameters") {
            None | Some(Value::Null) => Value::Object(Map::new()),
            Some(p @ Value::Object(_)) => p.clone(),
            Some(_) => return Err(bad("tool parameters must be a JSON object")),
        };
        let mut spec = Map::new();
        spec.insert(
            "description".into(),
            function
                .get("description")
                .filter(|d| !d.is_null())
                .cloned()
                .unwrap_or_else(|| Value::String(String::new())),
        );
        spec.insert("name".into(), function.get("name").cloned().ok_or_else(|| bad("tool has no name"))?);
        spec.insert("parameters".into(), parameters);
        spec.insert(
            "type".into(),
            Value::String(
                t.get("type").and_then(Value::as_str).filter(|s| !s.is_empty()).unwrap_or("function").to_string(),
            ),
        );
        specs.push(Value::Object(spec));
    }
    let mut s = String::new();
    tml_json(&Value::Array(specs), true, true, &mut s)?;
    Ok(s)
}

// ---------------------------------------------------------------------------------------------------------------------
// the reference's compact JSON
// ---------------------------------------------------------------------------------------------------------------------

fn json_str(s: &str) -> String {
    serde_json::to_string(s).expect("string")
}

/// A float as the reference spells it (`_tml_json_dumps`): Python's `repr` with exponents -5..=15 written out in full, the
/// exponent form past that (`1e-7`, and `1e+22` when `plus`, the tool declarations' spelling).
fn float_str(x: f64, plus: bool) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("exponent");
    let neg = if x < 0.0 { "-" } else { "" };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if (-5..16).contains(&exp) {
        let decimal = exp + 1;
        let body = if decimal <= 0 {
            format!("0.{}{digits}", "0".repeat((-decimal) as usize))
        } else if decimal as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(decimal as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..decimal as usize], &digits[decimal as usize..])
        };
        format!("{neg}{body}")
    } else {
        format!("{neg}{mantissa}e{}{exp}", if exp >= 0 && plus { "+" } else { "" })
    }
}

/// Compact JSON: object keys sorted at every level when `sort`, the exponent sign of floats printed when `plus`.
fn tml_json(v: &Value, sort: bool, plus: bool, out: &mut String) -> Result<()> {
    match v {
        Value::Object(o) => {
            out.push('{');
            let mut keys: Vec<&String> = o.keys().collect();
            if sort {
                keys.sort();
            }
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&json_str(k));
                out.push(':');
                tml_json(&o[k.as_str()], sort, plus, out)?;
            }
            out.push('}');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                tml_json(x, sort, plus, out)?;
            }
            out.push(']');
        }
        Value::Number(n) if n.is_f64() => {
            let f = n.as_f64().unwrap();
            if !f.is_finite() {
                return Err(bad("JSON numbers must be finite"));
            }
            out.push_str(&float_str(f, plus));
        }
        other => out.push_str(&serde_json::to_string(other).expect("json")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn float_spelling_follows_the_reference() {
        for (x, plus, want) in [
            (1.0, false, "1.0"),
            (0.5, false, "0.5"),
            (100.0, false, "100.0"),
            (1e-5, false, "0.00001"),
            (1.5e-5, false, "0.000015"),
            (1e-6, false, "1e-6"),
            (1e15, false, "1000000000000000.0"),
            (1e16, false, "1e16"),
            (1e16, true, "1e+16"),
            (1.5e22, true, "1.5e+22"),
            (-2.5e-7, true, "-2.5e-7"),
            (123456.789, false, "123456.789"),
            (-0.0, false, "-0.0"),
        ] {
            assert_eq!(float_str(x, plus), want, "{x}");
        }
    }

    #[test]
    fn effort_text_quantizes_like_the_reference() {
        for (e, want) in [
            (0.0, "0"),
            (0.1, "0.1"),
            (0.7, "0.7"),
            (0.9, "0.9"),
            (0.99, "0.99"),
            (0.85, "0.85"),
            (0.125, "0.12"),
            (0.123, "0.12"),
            (0.005, "0"),
            (0.0051, "0"),
            (0.0151, "0.02"),
        ] {
            assert_eq!(effort_text(e).unwrap(), format!("Thinking effort level: {want}"), "{e}");
        }
        assert!(effort_text(1.0).is_err());
        assert!(effort_text(-0.1).is_err());
        assert!(effort_text(f64::NAN).is_err());
    }

    #[test]
    fn effort_names_and_strings() {
        assert_eq!(resolve_effort(None).unwrap(), 0.9);
        assert_eq!(resolve_effort(Some(&json!("none"))).unwrap(), 0.0);
        assert_eq!(resolve_effort(Some(&json!("max"))).unwrap(), 0.99);
        assert_eq!(resolve_effort(Some(&json!("0.35"))).unwrap(), 0.35);
        assert_eq!(resolve_effort(Some(&json!(0))).unwrap(), 0.0);
        assert!(resolve_effort(Some(&json!("loud"))).is_err());
        assert!(resolve_effort(Some(&json!(true))).is_err());
    }

    #[test]
    fn tool_declaration_sorts_every_level_and_args_only_the_first() {
        let tools = vec![
            json!({"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"b": {"type": "number", "maximum": 1e22}, "a": {"type": "string"}}}}}),
        ];
        assert_eq!(
            tool_declare_json(&tools).unwrap(),
            r#"[{"description":"","name":"f","parameters":{"properties":{"a":{"type":"string"},"b":{"maximum":1e+22,"type":"number"}},"type":"object"},"type":"function"}]"#
        );
        let mut names = HashMap::new();
        let m = json!({"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"z\": {\"y\": 1, \"b\": 2.0}, \"a\": [1, 2]}"}}]});
        let blocks = blocks_of(&m, "assistant", &mut names).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].0, "invoke_tool_json");
        assert_eq!(blocks[0].1, r#"{"name":"f","args":{"a":[1,2],"z":{"y":1,"b":2.0}}}"#);
        assert_eq!(blocks[0].2.as_deref(), Some("f"));
        assert_eq!(names["c1"], "f");
    }
}
