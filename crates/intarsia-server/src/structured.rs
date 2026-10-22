//! Structured output (docs/structured-output.md): OpenAI `response_format` and schema-constrained tool calls
//! (OpenAI `tools` / `tool_choice`, Anthropic `tools` / `tool_choice`) compiled with llguidance into per-token masks
//! the engine applies before sampling (`intarsia_core::constrain`).
//!
//! The token table comes from the model's HF `tokenizer.json`, padded to the logit width, with the model's EOS ids as
//! the grammar's end. It is built once, on the first constrained request. Tool calls use the hermes format the Qwen
//! templates use (`<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>`); [`ToolParser`] turns generated text
//! back into calls.

use crate::api::{blocking_as, ApiError, AppState, Flavor};
use intarsia_core::constrain::{ConstraintSpec, TokenConstraint, TokenMask};
use intarsia_core::engine::GenRequest;
use intarsia_core::Engine;
use llguidance::api::TopLevelGrammar;
use llguidance::earley::SlicedBiasComputer;
use llguidance::toktrie::{InferenceCapabilities, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use llguidance::{Matcher, ParserFactory};
use serde_json::{json, Map, Value};
use std::sync::{Arc, Mutex};

/// Tokenize callback of the grammar env (the model's own tokenizer; `None` falls back to the trie's greedy split).
type TokenizeFn = Box<dyn Fn(&str) -> Option<Vec<u32>> + Send + Sync>;

struct ModelTokEnv {
    trie: TokTrie,
    tokenize: TokenizeFn,
}

impl TokenizerEnv for ModelTokEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }
    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        match std::str::from_utf8(s).ok().and_then(|t| (self.tokenize)(t)) {
            Some(ids) if ids.iter().all(|&i| (i as usize) < self.trie.vocab_size()) => ids,
            _ => self.trie.greedy_tokenize(s),
        }
    }
}

/// A hermes tool-call tag (`<tool_call>` / `</tool_call>`) in grammar syntax: the literal, or `<[id]>` when the
/// tokenizer marks the tag token special (special tokens never match literal text).
#[derive(Clone, Debug)]
struct Tag {
    id: Option<u32>,
    lark: String,
}

/// The per-model part of structured output: token table, parser factory, tool-call tags.
pub struct GrammarEnv {
    factory: ParserFactory,
    vocab: usize,
    open: Tag,
    close: Tag,
    /// The single `</think>` token, when the tokenizer has one (needed to gate a constraint behind the think block).
    think_close: Option<u32>,
}

impl GrammarEnv {
    /// Build from the engine's `tokenizer.json`, logit width and EOS ids.
    pub fn from_engine(engine: Arc<dyn Engine>) -> Result<GrammarEnv, String> {
        let info = engine.info();
        let text =
            engine.tokenizer_json().ok_or("structured output is unavailable: the model has no tokenizer.json")?;
        let tj: Value = serde_json::from_str(&text).map_err(|e| format!("tokenizer.json: {e}"))?;
        let bytes = llguidance::token_bytes_from_tokenizer_json(&tj).map_err(|e| format!("tokenizer.json: {e}"))?;
        let e2 = engine.clone();
        GrammarEnv::from_parts(bytes, info.vocab, &info.eos_ids, Box::new(move |s| e2.tokenize(s, false).ok()))
    }

    /// `token_bytes[i]` are token i's bytes (special tokens prefixed with `TokTrie::SPECIAL_TOKEN_MARKER`); the table
    /// is padded (or cut) to `vocab`, the logit width.
    fn from_parts(
        mut token_bytes: Vec<Vec<u8>>,
        vocab: usize,
        eos: &[u32],
        tokenize: TokenizeFn,
    ) -> Result<GrammarEnv, String> {
        token_bytes.resize(vocab, Vec::new());
        let eos: Vec<u32> = eos.iter().copied().filter(|&e| (e as usize) < vocab).collect();
        let Some(&eos0) = eos.first() else {
            return Err("structured output is unavailable: the model has no EOS token".into());
        };
        let trie = TokTrie::from(&TokRxInfo::new(vocab as u32, eos0), &token_bytes).with_eos_tokens(&eos);
        let tag = |s: &str| -> Tag {
            let id = match tokenize(s) {
                Some(ids) if ids.len() == 1 && (ids[0] as usize) < vocab => Some(ids[0]),
                _ => None,
            };
            let lark = match id {
                Some(i) if trie.is_special_token(i) => format!("<[{i}]>"),
                _ => serde_json::to_string(s).expect("string"),
            };
            Tag { id, lark }
        };
        let (open, close) = (tag("<tool_call>"), tag("</tool_call>"));
        let think_close = tag(THINK_CLOSE).id;
        let env: TokEnv = Arc::new(ModelTokEnv { trie, tokenize });
        let mut factory =
            ParserFactory::new(&env, InferenceCapabilities::default(), &SlicedBiasComputer::general_slices())
                .map_err(|e| format!("structured output: {e}"))?;
        factory.quiet();
        Ok(GrammarEnv { factory, vocab, open, close, think_close })
    }

    /// [`GrammarEnv::compile_with`] with thinking off: the grammar starts at the first generated token.
    #[cfg(test)]
    pub fn compile(&self, plan: &Plan) -> Result<Arc<dyn ConstraintSpec>, String> {
        self.compile_with(plan, false)
    }

    /// Compile `plan` into a constraint the engine can run; `Err` is a client error (HTTP 400) with the reason.
    ///
    /// `think`: the request has thinking on, so the constraint applies only after the `</think>` token; the think
    /// section before it is unconstrained (and nothing in it, a `<tool_call>` included, triggers the grammar).
    pub fn compile_with(&self, plan: &Plan, think: bool) -> Result<Arc<dyn ConstraintSpec>, String> {
        let gate = if think {
            Some(self.think_close.ok_or("thinking with structured output needs a model whose tokenizer has a single </think> token")?)
        } else {
            None
        };
        let (grammar, trigger) = match plan {
            Plan::Json(schema) => (TopLevelGrammar::from_json_schema(compact(schema)), None),
            Plan::Tools { tools, required: true, parallel } => {
                // one call (or several, newline-separated), nothing else; the grammar ends where the model must stop
                let call = format!(r#"{} "\n" %json {} "\n" {}"#, self.open.lark, calls_schema(tools), self.close.lark);
                let start = if *parallel { "call (\"\\n\" call)*" } else { "call" };
                (TopLevelGrammar::from_lark(format!("start: {start}\ncall: {call}\n")), None)
            }
            Plan::Tools { tools, required: false, .. } => {
                // free text until the model opens a call; the call's body is constrained, then text is free again
                let trigger = self.open.id.ok_or("tool_choice auto needs a single-token <tool_call> tag")?;
                let body = format!(r#""\n" %json {} "\n" {}"#, calls_schema(tools), self.close.lark);
                (TopLevelGrammar::from_lark(format!("start: {body}\n")), Some(trigger))
            }
        };
        let parser = self.factory.create_parser(grammar).map_err(|e| format!("invalid schema: {e}"))?;
        let m = Matcher::new(Ok(parser));
        if let Some(e) = m.get_error() {
            return Err(format!("invalid schema: {e}"));
        }
        let once = matches!(plan, Plan::Tools { required: false, parallel: false, .. });
        Ok(Arc::new(LlgSpec { proto: Mutex::new(m), trigger, gate, once, vocab: self.vocab }))
    }
}

/// Compile `plan` for the engine `st` serves (building the token table on first use). Every failure is the client's
/// (HTTP 400): an unusable schema, or a model structured output cannot run on.
pub async fn compile_for(
    st: &AppState,
    flavor: Flavor,
    plan: Plan,
    think: bool,
) -> Result<Arc<dyn ConstraintSpec>, ApiError> {
    let (cell, eng) = (st.grammar.clone(), st.engine.clone());
    let r = blocking_as(&st.engine, flavor, move |_| {
        let env = cell.get_or_init(|| GrammarEnv::from_engine(eng).map(Arc::new)).clone();
        Ok(env.and_then(|env| env.compile_with(&plan, think)))
    })
    .await?;
    r.map_err(|m| ApiError::bad_request(flavor, m))
}

/// Attach `plan`'s constraint to `gen` (none for `None`). Rejects the sampling options a constraint cannot honour.
/// `think`: thinking is on, the constraint starts after `</think>` (see [`GrammarEnv::compile`]).
pub async fn attach(
    st: &AppState,
    flavor: Flavor,
    gen: &mut GenRequest,
    plan: Option<Plan>,
    think: bool,
) -> Result<(), ApiError> {
    let Some(plan) = plan else { return Ok(()) };
    let bad = |m: &str| Err(ApiError::bad_request(flavor, m));
    if gen.ignore_eos {
        return bad("ignore_eos cannot be combined with structured output (the grammar ends with EOS)");
    }
    if gen.min_tokens > 0 {
        return bad("min_tokens cannot be combined with structured output");
    }
    gen.constraint = Some(compile_for(st, flavor, plan, think).await?);
    Ok(())
}

/// Split generated text into the content and the hermes tool calls in it. Without calls the content is the text
/// unchanged; with calls it is the text around them, trimmed (`None` when nothing is left).
pub fn split_calls(text: &str) -> (Option<String>, Vec<(String, Value)>) {
    let mut calls = Vec::new();
    let mut rest = String::new();
    for p in ToolParser::parse_all(text) {
        match p {
            Piece::Text(t) => rest.push_str(&t),
            Piece::Call { name, arguments } => calls.push((name, arguments)),
        }
    }
    if calls.is_empty() {
        return (Some(text.to_string()), calls);
    }
    let t = rest.trim();
    (if t.is_empty() { None } else { Some(t.to_string()) }, calls)
}

/// Which plan a request's `response_format` / `tools` / `tool_choice` make, and whether the output is parsed for hermes
/// tool calls. `hermes`: the model's tool format (`tool_parser`) is the one [`ToolParser`] reads.
pub fn choose(
    format: Option<Plan>,
    tools: &[Tool],
    choice: &Choice,
    parallel: bool,
    hermes: bool,
) -> Result<(Option<Plan>, bool), String> {
    if let Some(f) = format {
        if *choice != Choice::None {
            return Err("response_format cannot be combined with tools (unless tool_choice is none)".into());
        }
        return Ok((Some(f), false));
    }
    if *choice == Choice::None {
        return Ok((None, false));
    }
    if !hermes {
        return match choice {
            // as before: the tools reach the template, the output is returned as text
            Choice::Auto => Ok((None, false)),
            _ => Err("tool_choice required / a named function needs a model with hermes-format tool calls".into()),
        };
    }
    Ok((tools_plan(tools, choice, parallel), true))
}

/// What a request asks the output to be.
#[derive(Clone, Debug)]
pub enum Plan {
    /// One JSON value matching the schema.
    Json(Value),
    /// Hermes tool calls with schema-constrained arguments: `required` = calls only (tool_choice `required` or a named
    /// function), else `auto` (free text, each call constrained once its tag is opened). `parallel: false`
    /// (`parallel_tool_calls: false`) allows one call, after which the model must stop.
    Tools { tools: Vec<Tool>, required: bool, parallel: bool },
}

#[derive(Clone, Debug)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
}

impl Tool {
    /// The OpenAI tools-array entry (what the chat templates render).
    pub fn openai(&self) -> Value {
        let mut f = json!({"name": self.name, "parameters": self.parameters});
        if let Some(d) = &self.description {
            f["description"] = json!(d);
        }
        json!({"type": "function", "function": f})
    }
}

/// JSON output style: compact separators as `json.dumps` writes them, no free whitespace (a greedy model can otherwise
/// pad forever and hit `max_tokens` with invalid JSON). A schema's own `x-guidance` wins.
fn compact(schema: &Value) -> Value {
    let mut s = schema.clone();
    if let Value::Object(o) = &mut s {
        o.entry("x-guidance")
            .or_insert_with(|| json!({"whitespace_flexible": false, "item_separator": ", ", "key_separator": ": "}));
    }
    s
}

/// Schema of one hermes call body over `tools`: `{"name": <const>, "arguments": <parameters>}`, any of the tools.
fn calls_schema(tools: &[Tool]) -> String {
    let one = |t: &Tool| {
        json!({"type": "object",
               "properties": {"name": {"const": t.name}, "arguments": t.parameters},
               "required": ["name", "arguments"], "additionalProperties": false})
    };
    let s = if tools.len() == 1 { one(&tools[0]) } else { json!({"anyOf": tools.iter().map(one).collect::<Vec<_>>()}) };
    compact(&s).to_string()
}

// ---------------------------------------------------------------------------
// request parsing (OpenAI / Anthropic shapes)
// ---------------------------------------------------------------------------

fn check_schema(v: &Value, what: &str) -> Result<(), String> {
    match v {
        Value::Object(_) | Value::Bool(true) => Ok(()),
        _ => Err(format!("{what} must be a JSON schema object")),
    }
}

/// OpenAI `response_format`: `None` for free text.
pub fn response_format(v: &Option<Value>) -> Result<Option<Plan>, String> {
    let Some(v) = v else { return Ok(None) };
    if v.is_null() {
        return Ok(None);
    }
    let ty = v.get("type").and_then(Value::as_str).ok_or("response_format.type is required")?;
    match ty {
        "text" => Ok(None),
        "json_object" => Ok(Some(Plan::Json(json!({"type": "object"})))),
        "json_schema" => {
            let js = v.get("json_schema").ok_or("response_format.json_schema is required for type json_schema")?;
            if let Some(n) = js.get("name") {
                if !n.is_string() {
                    return Err("response_format.json_schema.name must be a string".into());
                }
            }
            if let Some(s) = js.get("strict") {
                if !s.is_boolean() && !s.is_null() {
                    return Err("response_format.json_schema.strict must be a boolean".into());
                }
            }
            let schema = js.get("schema").ok_or("response_format.json_schema.schema is required")?;
            check_schema(schema, "response_format.json_schema.schema")?;
            Ok(Some(Plan::Json(if schema == &Value::Bool(true) { json!({}) } else { schema.clone() })))
        }
        other => Err(format!("unsupported response_format type '{other}' (text, json_object, json_schema)")),
    }
}

/// A function without `parameters` takes no arguments.
fn empty_params() -> Value {
    json!({"type": "object", "properties": {}, "additionalProperties": false})
}

fn tool_name(n: Option<&Value>, what: &str) -> Result<String, String> {
    let n = n.and_then(Value::as_str).ok_or(format!("{what} name is required"))?;
    if n.is_empty() || n.len() > 64 || !n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
    {
        return Err(format!("invalid tool name '{n}' (1-64 of a-z A-Z 0-9 _ - .)"));
    }
    Ok(n.to_string())
}

fn unique(tools: &[Tool]) -> Result<(), String> {
    for (i, t) in tools.iter().enumerate() {
        if tools[..i].iter().any(|u| u.name == t.name) {
            return Err(format!("duplicate tool name '{}'", t.name));
        }
    }
    Ok(())
}

/// OpenAI `tools`: `[{type: "function", function: {name, description?, parameters?}}]`.
pub fn openai_tools(v: &Option<Value>) -> Result<Vec<Tool>, String> {
    let Some(v) = v else { return Ok(Vec::new()) };
    let arr = match v {
        Value::Null => return Ok(Vec::new()),
        Value::Array(a) => a,
        _ => return Err("tools must be an array".into()),
    };
    let mut out = Vec::with_capacity(arr.len());
    for t in arr {
        if t.get("type").and_then(Value::as_str) != Some("function") {
            return Err("only tools of type 'function' are supported".into());
        }
        let f = t.get("function").filter(|f| f.is_object()).ok_or("tool.function must be an object")?;
        let parameters = match f.get("parameters") {
            None | Some(Value::Null) => empty_params(),
            Some(p) => {
                check_schema(p, "tool.function.parameters")?;
                p.clone()
            }
        };
        out.push(Tool {
            name: tool_name(f.get("name"), "tool.function")?,
            description: f.get("description").and_then(Value::as_str).map(String::from),
            parameters,
        });
    }
    unique(&out)?;
    Ok(out)
}

/// Anthropic `tools`: `[{name, description?, input_schema}]` (server tools with another `type` are rejected).
pub fn anthropic_tools(v: &Option<Value>) -> Result<Vec<Tool>, String> {
    let Some(v) = v else { return Ok(Vec::new()) };
    let arr = match v {
        Value::Null => return Ok(Vec::new()),
        Value::Array(a) => a,
        _ => return Err("tools must be an array".into()),
    };
    let mut out = Vec::with_capacity(arr.len());
    for t in arr {
        if let Some(ty) = t.get("type").and_then(Value::as_str) {
            if ty != "custom" {
                return Err(format!("unsupported tool type '{ty}' (client tools with input_schema only)"));
            }
        }
        let p = t.get("input_schema").ok_or("tool.input_schema is required")?;
        check_schema(p, "tool.input_schema")?;
        out.push(Tool {
            name: tool_name(t.get("name"), "tool")?,
            description: t.get("description").and_then(Value::as_str).map(String::from),
            parameters: p.clone(),
        });
    }
    unique(&out)?;
    Ok(out)
}

/// What `tool_choice` asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Choice {
    None,
    Auto,
    Required,
    Named(String),
}

fn named(tools: &[Tool], n: &str) -> Result<Choice, String> {
    if tools.iter().any(|t| t.name == n) {
        Ok(Choice::Named(n.to_string()))
    } else {
        Err(format!("tool_choice names '{n}', which is not in tools"))
    }
}

/// OpenAI `tool_choice`: `"none" | "auto" | "required" | {type: "function", function: {name}}`.
pub fn openai_choice(v: &Option<Value>, tools: &[Tool]) -> Result<Choice, String> {
    let c = match v {
        None | Some(Value::Null) => return Ok(if tools.is_empty() { Choice::None } else { Choice::Auto }),
        Some(Value::String(s)) => match s.as_str() {
            "none" => Choice::None,
            "auto" => Choice::Auto,
            "required" => Choice::Required,
            o => return Err(format!("unsupported tool_choice '{o}' (none, auto, required or a function)")),
        },
        Some(o @ Value::Object(_)) => {
            if o.get("type").and_then(Value::as_str) != Some("function") {
                return Err("tool_choice object must have type 'function'".into());
            }
            let n =
                o.pointer("/function/name").and_then(Value::as_str).ok_or("tool_choice.function.name is required")?;
            named(tools, n)?
        }
        _ => return Err("tool_choice must be a string or an object".into()),
    };
    if c != Choice::None && tools.is_empty() {
        return Err("tool_choice needs tools".into());
    }
    Ok(c)
}

/// Anthropic `tool_choice`: `{type: "auto" | "any" | "tool" | "none", name?}`.
pub fn anthropic_choice(v: &Option<Value>, tools: &[Tool]) -> Result<Choice, String> {
    let c = match v {
        None | Some(Value::Null) => return Ok(if tools.is_empty() { Choice::None } else { Choice::Auto }),
        Some(o @ Value::Object(_)) => match o.get("type").and_then(Value::as_str) {
            Some("auto") => Choice::Auto,
            Some("any") => Choice::Required,
            Some("none") => Choice::None,
            Some("tool") => named(tools, o.get("name").and_then(Value::as_str).ok_or("tool_choice.name is required")?)?,
            _ => return Err("tool_choice.type must be auto, any, tool or none".into()),
        },
        _ => return Err("tool_choice must be an object".into()),
    };
    if c != Choice::None && tools.is_empty() {
        return Err("tool_choice needs tools".into());
    }
    Ok(c)
}

/// The constraint plan for a request's tools: `None` for `tool_choice: none` (or no tools).
pub fn tools_plan(tools: &[Tool], choice: &Choice, parallel: bool) -> Option<Plan> {
    let plan = |tools: Vec<Tool>, required| Some(Plan::Tools { tools, required, parallel });
    match choice {
        Choice::None => None,
        Choice::Auto => plan(tools.to_vec(), false),
        Choice::Required => plan(tools.to_vec(), true),
        Choice::Named(n) => plan(tools.iter().filter(|t| &t.name == n).cloned().collect(), true),
    }
}

// ---------------------------------------------------------------------------
// the engine-side constraint
// ---------------------------------------------------------------------------

/// A compiled grammar. Every sequence starts from a clone of the prototype matcher.
struct LlgSpec {
    proto: Mutex<Matcher>,
    trigger: Option<u32>,
    /// Thinking on: the grammar (and its trigger) wait for this `</think>` token.
    gate: Option<u32>,
    /// Lazy constraint that allows one call: after it only EOS.
    once: bool,
    vocab: usize,
}

impl std::fmt::Debug for LlgSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlgSpec").field("trigger", &self.trigger).finish_non_exhaustive()
    }
}

impl ConstraintSpec for LlgSpec {
    fn start(&self) -> intarsia_core::Result<Box<dyn TokenConstraint>> {
        let m = self.proto.lock().unwrap_or_else(|e| e.into_inner()).deep_clone();
        Ok(Box::new(LlgConstraint {
            m,
            trigger: self.trigger,
            gate: self.gate,
            gated: self.gate.is_some(),
            active: self.trigger.is_none() && self.gate.is_none(),
            once: self.once,
            spent: false,
            vocab: self.vocab,
        }))
    }
}

struct LlgConstraint {
    m: Matcher,
    /// Lazy constraint: the grammar starts after this token and re-arms once it is complete.
    trigger: Option<u32>,
    /// Waiting for the `</think>` token (`gate`) before anything else applies.
    gate: Option<u32>,
    gated: bool,
    active: bool,
    once: bool,
    /// `once` and the call is done: only EOS may follow.
    spent: bool,
    vocab: usize,
}

fn llg_err(e: impl std::fmt::Display) -> intarsia_core::Error {
    intarsia_core::Error::Other(e.to_string())
}

impl TokenConstraint for LlgConstraint {
    fn active(&self) -> bool {
        self.active
    }

    fn compute_mask(&mut self) -> intarsia_core::Result<TokenMask> {
        let v = if self.spent {
            self.m.tok_env().map_err(llg_err)?.tok_trie().eos_token_set()
        } else {
            self.m.compute_mask_or_eos().map_err(llg_err)?
        };
        let mut words = v.as_slice().to_vec();
        words.resize(self.vocab.div_ceil(32), 0);
        Ok(TokenMask { words })
    }

    fn commit(&mut self, tok: u32) -> intarsia_core::Result<()> {
        if self.gated {
            if Some(tok) == self.gate {
                self.gated = false;
                self.active = self.trigger.is_none();
            }
            return Ok(());
        }
        if !self.active {
            if Some(tok) == self.trigger {
                self.active = true;
            }
            return Ok(());
        }
        self.m.consume_token(tok).map_err(llg_err)?;
        if self.trigger.is_some() && !self.spent && self.m.is_accepting().map_err(llg_err)? {
            if self.once {
                // the one call allowed is closed: the model must stop
                self.spent = true;
            } else {
                // the call is closed: free text again until the next trigger
                self.m.reset().map_err(llg_err)?;
                self.active = false;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// hermes tool-call output parsing
// ---------------------------------------------------------------------------

/// A piece of generated output.
#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    Text(String),
    /// A complete call: the function name and its arguments object.
    Call {
        name: String,
        arguments: Value,
    },
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// Splits streamed text into plain text and hermes tool calls. Text that may be the start of `<tool_call>` is held
/// back until it is decided; a call is reported once its closing tag arrives. A block that does not parse as a call
/// is passed through as text.
#[derive(Default)]
pub struct ToolParser {
    buf: String,
    in_call: bool,
    /// Inside a call: where to look for the next `</tool_call>` (an earlier one sat inside a JSON string).
    from: usize,
}

impl ToolParser {
    pub fn push(&mut self, text: &str) -> Vec<Piece> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            if self.in_call {
                let Some(i) = self.buf[self.from..].find(CLOSE).map(|i| i + self.from) else { break };
                match parse_call(&self.buf[..i]) {
                    Some(call) => {
                        self.buf.drain(..i + CLOSE.len());
                        self.in_call = false;
                        self.from = 0;
                        out.push(call);
                    }
                    // not a complete call yet: the tag may be part of an argument string
                    None => self.from = i + CLOSE.len(),
                }
            } else if let Some(i) = self.buf.find(OPEN) {
                if i > 0 {
                    out.push(Piece::Text(self.buf[..i].to_string()));
                }
                self.buf.drain(..i + OPEN.len());
                self.in_call = true;
                self.from = 0;
            } else {
                // release all but a possible partial opening tag
                let keep = (1..OPEN.len()).rev().find(|&k| self.buf.ends_with(&OPEN[..k])).unwrap_or(0);
                let cut = self.buf.len() - keep;
                if cut > 0 {
                    out.push(Piece::Text(self.buf[..cut].to_string()));
                    self.buf.drain(..cut);
                }
                break;
            }
        }
        out
    }

    /// End of output: whatever is held back (an unterminated call is returned as text).
    pub fn finish(&mut self) -> Vec<Piece> {
        let rest = std::mem::take(&mut self.buf);
        self.from = 0;
        let text = if std::mem::take(&mut self.in_call) { format!("{OPEN}{rest}") } else { rest };
        if text.is_empty() {
            Vec::new()
        } else {
            vec![Piece::Text(text)]
        }
    }

    /// The whole output at once.
    pub fn parse_all(text: &str) -> Vec<Piece> {
        let mut p = ToolParser::default();
        let mut v = p.push(text);
        v.extend(p.finish());
        v
    }
}

fn parse_call(body: &str) -> Option<Piece> {
    let v: Value = serde_json::from_str(body.trim()).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = match v.get("arguments") {
        None | Some(Value::Null) => Value::Object(Map::new()),
        // some models write the arguments as a JSON string
        Some(Value::String(s)) => serde_json::from_str(s).ok()?,
        Some(a) => a.clone(),
    };
    Some(Piece::Call { name, arguments })
}

// ---------------------------------------------------------------------------
// reasoning (`<think>...</think>`) output parsing
// ---------------------------------------------------------------------------

/// A piece of output split by [`ReasoningParser`].
#[derive(Clone, Debug, PartialEq)]
pub enum Part {
    /// Thinking (`reasoning_content` / a `thinking` block).
    Reasoning(String),
    /// The answer (`content`).
    Content(String),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    /// Before the first byte is decided: an optional `<think>` opener may still arrive.
    Start,
    Reasoning,
    Content,
}

/// Splits generated text at `</think>` as vLLM's `qwen3` reasoning parser does: everything before it is reasoning (an
/// optional leading `<think>` is dropped, since newer templates put it in the prompt), everything after is content.
/// Output without a `</think>` is all reasoning. Whitespace around the split (the `\n` the model writes after
/// `<think>` and `\n\n` around `</think>`) is trimmed. Text that may be the start of `</think>` is held back.
#[derive(Debug)]
pub struct ReasoningParser {
    phase: Phase,
    buf: String,
    /// Nothing but whitespace was emitted in the current section so far: drop leading whitespace.
    lead: bool,
}

impl Default for ReasoningParser {
    fn default() -> Self {
        ReasoningParser { phase: Phase::Start, buf: String::new(), lead: true }
    }
}

/// Length of the longest suffix of `s` that is a proper prefix of `pat`.
fn partial_suffix(s: &str, pat: &str) -> usize {
    (1..pat.len().min(s.len() + 1)).rev().find(|&n| s.is_char_boundary(s.len() - n) && s.ends_with(&pat[..n])).unwrap_or(0)
}

impl ReasoningParser {
    pub fn push(&mut self, text: &str) -> Vec<Part> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            match self.phase {
                Phase::Start => {
                    if let Some(r) = self.buf.strip_prefix(THINK_OPEN) {
                        self.buf = r.to_string();
                        self.phase = Phase::Reasoning;
                    } else if THINK_OPEN.starts_with(self.buf.as_str()) {
                        return out; // empty or a proper prefix of the opener: wait
                    } else {
                        self.phase = Phase::Reasoning;
                    }
                }
                Phase::Reasoning => {
                    if let Some(i) = self.buf.find(THINK_CLOSE) {
                        let rest = self.buf.split_off(i + THINK_CLOSE.len());
                        self.buf.truncate(i);
                        let head = std::mem::take(&mut self.buf);
                        self.emit(&mut out, &head, true);
                        self.buf = rest;
                        self.phase = Phase::Content;
                        self.lead = true;
                    } else {
                        let keep = partial_suffix(&self.buf, THINK_CLOSE);
                        let cut = self.buf.len() - keep;
                        let head = self.buf[..cut].trim_end().len();
                        // trailing whitespace (and a possible partial tag) stays until it is known not to be trailing
                        let emit: String = self.buf.drain(..head).collect();
                        self.emit(&mut out, &emit, false);
                        return out;
                    }
                }
                Phase::Content => {
                    let t = std::mem::take(&mut self.buf);
                    self.emit_content(&mut out, &t);
                    return out;
                }
            }
        }
    }

    fn emit(&mut self, out: &mut Vec<Part>, t: &str, last: bool) {
        let t = if self.lead { t.trim_start() } else { t };
        let t = if last { t.trim_end() } else { t };
        if !t.is_empty() {
            self.lead = false;
            out.push(Part::Reasoning(t.to_string()));
        }
    }

    fn emit_content(&mut self, out: &mut Vec<Part>, t: &str) {
        let t = if self.lead { t.trim_start() } else { t };
        if !t.is_empty() {
            self.lead = false;
            out.push(Part::Content(t.to_string()));
        }
    }

    /// End of output: whatever is held back.
    pub fn finish(&mut self) -> Vec<Part> {
        let rest = std::mem::take(&mut self.buf);
        let mut out = Vec::new();
        match self.phase {
            Phase::Start | Phase::Reasoning => self.emit(&mut out, &rest, true),
            Phase::Content => self.emit_content(&mut out, &rest),
        }
        out
    }

    /// The whole output at once: `(reasoning, content)`, `None` for an empty section.
    pub fn parse_all(text: &str) -> (Option<String>, Option<String>) {
        let mut p = ReasoningParser::default();
        let mut parts = p.push(text);
        parts.extend(p.finish());
        let (mut r, mut c) = (String::new(), String::new());
        for part in parts {
            match part {
                Part::Reasoning(t) => r.push_str(&t),
                Part::Content(t) => c.push_str(&t),
            }
        }
        ((!r.is_empty()).then_some(r), (!c.is_empty()).then_some(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use intarsia_core::rng::SplitMix64;

    /// Single-byte vocabulary (token i = byte i for i < 256), EOS = 256 (special), a `<tool_call>` token 257 and a
    /// `</tool_call>` token 258 (plain text), padded to 300 logits.
    fn env() -> GrammarEnv {
        let mut b: Vec<Vec<u8>> = (0..=255u8).map(|x| vec![x]).collect();
        b.push(b"\xFF<eos>".to_vec());
        b.push(OPEN.as_bytes().to_vec());
        b.push(CLOSE.as_bytes().to_vec());
        b.push(THINK_CLOSE.as_bytes().to_vec());
        GrammarEnv::from_parts(b, 300, &[256], Box::new(|s| Some(tokenize(s)))).unwrap()
    }

    /// The tags are single tokens, the rest is bytes (as a BPE tokenizer with the tags as added tokens splits it).
    fn tokenize(s: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = s;
        while !rest.is_empty() {
            if let Some(r) = rest.strip_prefix(OPEN) {
                out.push(257);
                rest = r;
            } else if let Some(r) = rest.strip_prefix(CLOSE) {
                out.push(258);
                rest = r;
            } else if let Some(r) = rest.strip_prefix(THINK_CLOSE) {
                out.push(259);
                rest = r;
            } else {
                let c = rest.chars().next().unwrap();
                out.extend(c.to_string().bytes().map(u32::from));
                rest = &rest[c.len_utf8()..];
            }
        }
        out
    }

    fn bytes_of(t: u32) -> Vec<u8> {
        match t {
            0..=255 => vec![t as u8],
            257 => OPEN.as_bytes().to_vec(),
            258 => CLOSE.as_bytes().to_vec(),
            259 => THINK_CLOSE.as_bytes().to_vec(),
            _ => Vec::new(),
        }
    }

    /// Generate under the constraint with random logits (a seeded "model"), as the engine does: mask, sample, stop on
    /// EOS, commit. `eos_bias` makes the "model" more eager to stop where the grammar lets it.
    fn run(spec: &Arc<dyn ConstraintSpec>, seed: u64, max: usize, eos_bias: f32) -> (String, bool) {
        let mut c = spec.start().unwrap();
        let mut rng = SplitMix64::new(seed);
        let mut out = Vec::new();
        for step in 0..max {
            let mut l: Vec<f32> = (0..300).map(|_| rng.next_f32() * 4.0).collect();
            l[256] += eos_bias;
            if c.active() {
                assert!(c.compute_mask().unwrap().apply(&mut l), "mask empty at step {step}");
            }
            let t = intarsia_core::sampler::argmax(&l) as u32;
            if t == 256 {
                return (String::from_utf8_lossy(&out).into_owned(), true);
            }
            c.commit(t).unwrap();
            out.extend(bytes_of(t));
        }
        (String::from_utf8_lossy(&out).into_owned(), false)
    }

    fn validates(schema: &Value, v: &Value) -> bool {
        // a small checker for the schemas used here: type, properties, required, enum, min/max, items
        match schema.get("type").and_then(Value::as_str) {
            Some("object") => {
                let Some(o) = v.as_object() else { return false };
                let props = schema.get("properties").and_then(Value::as_object);
                if let Some(req) = schema.get("required").and_then(Value::as_array) {
                    if !req.iter().all(|k| o.contains_key(k.as_str().unwrap())) {
                        return false;
                    }
                }
                if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
                    if !o.keys().all(|k| props.is_some_and(|p| p.contains_key(k))) {
                        return false;
                    }
                }
                o.iter().all(|(k, x)| props.and_then(|p| p.get(k)).is_none_or(|s| validates(s, x)))
            }
            Some("integer") => {
                let Some(i) = v.as_i64() else { return false };
                schema.get("minimum").and_then(Value::as_i64).is_none_or(|m| i >= m)
                    && schema.get("maximum").and_then(Value::as_i64).is_none_or(|m| i <= m)
            }
            Some("string") => {
                v.is_string()
                    && schema.get("enum").and_then(Value::as_array).is_none_or(|e| e.contains(v))
                    && schema
                        .get("maxLength")
                        .and_then(Value::as_u64)
                        .is_none_or(|m| v.as_str().unwrap().chars().count() as u64 <= m)
            }
            Some("boolean") => v.is_boolean(),
            Some("array") => {
                let Some(a) = v.as_array() else { return false };
                schema.get("maxItems").and_then(Value::as_u64).is_none_or(|m| a.len() as u64 <= m)
                    && a.iter().all(|x| schema.get("items").is_none_or(|s| validates(s, x)))
            }
            _ => true,
        }
    }

    fn schema() -> Value {
        json!({"type": "object",
               "properties": {
                   "answer": {"type": "integer", "minimum": -5, "maximum": 999},
                   "unit": {"type": "string", "enum": ["kg", "m"]},
                   "ok": {"type": "boolean"},
                   "tags": {"type": "array", "items": {"type": "string", "maxLength": 4}, "maxItems": 3}},
               "required": ["answer", "unit", "ok", "tags"], "additionalProperties": false})
    }

    const EXPECTED_DIGESTS: [&str; 4] = ["base=0", "nested=0", "enum=0", "array=0"];

    #[test]
    fn json_schema_outputs_validate() {
        let e = env();
        let s = schema();
        let spec = e.compile(&Plan::Json(s.clone())).unwrap();
        for seed in 0..40 {
            let (text, stopped) = run(&spec, seed, 400, 1.0);
            assert!(stopped, "seed {seed}: no EOS after 400 tokens: {text:?}");
            let v: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("seed {seed}: {e}: {text:?}"));
            assert!(validates(&s, &v), "seed {seed}: {text}");
            // compact separators, no free whitespace
            assert!(text.starts_with("{\"answer\": "), "{text}");
        }
    }

    /// FNV-1a over every mask word and every chosen token of a seeded walk: pins the allowed-token set at each step
    /// (and so the greedy output), not just the validity of the final text.
    fn walk_digest(spec: &Arc<dyn ConstraintSpec>, seeds: std::ops::Range<u64>) -> (u64, Vec<String>) {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut eat = |x: u32| {
            for b in x.to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
            }
        };
        let mut texts = Vec::new();
        for seed in seeds {
            let mut c = spec.start().unwrap();
            let mut rng = SplitMix64::new(seed);
            let mut out = Vec::new();
            for _ in 0..600 {
                let mut l: Vec<f32> = (0..300).map(|_| rng.next_f32() * 4.0).collect();
                l[256] += 1.0;
                let m = c.compute_mask().unwrap();
                m.words.iter().for_each(|&w| eat(w));
                assert!(m.apply(&mut l));
                let t = intarsia_core::sampler::argmax(&l) as u32;
                eat(t);
                if t == 256 {
                    break;
                }
                c.commit(t).unwrap();
                out.extend(bytes_of(t));
            }
            texts.push(String::from_utf8_lossy(&out).into_owned());
        }
        (h, texts)
    }

    /// Allowed-token sets and outputs are pinned (digests recorded on main before the 2026-10-22 mask-path changes) for
    /// the schema above and three more shapes: a nested object, an enum, an array of objects.
    #[test]
    fn mask_digests_are_stable() {
        let nested = json!({"type": "object", "properties": {
            "user": {"type": "object", "properties": {
                "name": {"type": "string", "maxLength": 6},
                "address": {"type": "object", "properties": {
                    "zip": {"type": "integer", "minimum": 0, "maximum": 99999}, "city": {"type": "string", "maxLength": 5}},
                    "required": ["zip", "city"], "additionalProperties": false}},
                "required": ["name", "address"], "additionalProperties": false}},
            "required": ["user"], "additionalProperties": false});
        let en = json!({"type": "object", "properties": {
            "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
            "level": {"enum": ["low", "mid", "high", 7, null]}},
            "required": ["answer", "level"], "additionalProperties": false});
        let arr = json!({"type": "object", "properties": {"items": {"type": "array", "minItems": 1, "maxItems": 3,
            "items": {"type": "object", "properties": {
                "id": {"type": "integer", "minimum": 1, "maximum": 99},
                "tag": {"type": "string", "enum": ["x", "yy"]}},
                "required": ["id", "tag"], "additionalProperties": false}}},
            "required": ["items"], "additionalProperties": false});
        let e = env();
        let cases = [("base", schema()), ("nested", nested), ("enum", en), ("array", arr)];
        let mut got = Vec::new();
        for (name, sc) in &cases {
            let spec = e.compile(&Plan::Json(sc.clone())).unwrap();
            let (h, texts) = walk_digest(&spec, 0..8);
            for (i, t) in texts.iter().enumerate() {
                let v: Value = serde_json::from_str(t).unwrap_or_else(|e| panic!("{name} seed {i}: {e}: {t:?}"));
                assert!(validates(sc, &v), "{name} seed {i}: {t}");
            }
            eprintln!("digest {name} {h:016x}");
            got.push((*name, h));
        }
        assert_eq!(got.iter().map(|(n, h)| format!("{n}={h:016x}")).collect::<Vec<_>>(), EXPECTED_DIGESTS);
    }

    #[test]
    fn json_object_and_eos_only_when_complete() {
        let e = env();
        let spec = e.compile(&response_format(&Some(json!({"type": "json_object"}))).unwrap().unwrap()).unwrap();
        let mut c = spec.start().unwrap();
        let m = c.compute_mask().unwrap();
        assert!(m.allows(b'{' as u32) && !m.allows(b'x' as u32) && !m.allows(256), "must open an object, not stop");
        for &b in b"{}" {
            c.commit(b as u32).unwrap();
        }
        let m = c.compute_mask().unwrap();
        assert!(m.allows(256) && !m.allows(b'{' as u32), "complete: EOS only");
        assert!(!m.allows(299), "padding tokens are never allowed");
        // a token the grammar forbids is an error, not silently accepted
        assert!(c.commit(b'x' as u32).is_err());
    }

    #[test]
    fn invalid_schemas_are_errors() {
        let e = env();
        for bad in [
            json!({"type": "nonsense"}),
            json!({"type": "object", "properties": {"a": {"$ref": "#/defs/missing"}}}),
            json!({"type": "string", "pattern": "(unclosed"}),
            json!({"type": "integer", "minimum": 5, "maximum": 1}),
        ] {
            let r = e.compile(&Plan::Json(bad.clone()));
            assert!(r.is_err(), "{bad} compiled");
        }
        for (rf, why) in [
            (json!({"type": "xml"}), "unsupported response_format"),
            (json!({"type": "json_schema"}), "json_schema is required"),
            (json!({"type": "json_schema", "json_schema": {"name": "x"}}), "schema is required"),
            (json!({"type": "json_schema", "json_schema": {"schema": 3}}), "must be a JSON schema"),
            (json!({}), "type is required"),
        ] {
            let e = response_format(&Some(rf)).unwrap_err();
            assert!(e.contains(why), "{e}");
        }
        assert!(response_format(&Some(json!({"type": "text"}))).unwrap().is_none());
    }

    fn weather() -> Vec<Tool> {
        openai_tools(&Some(json!([
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
                "properties": {"city": {"type": "string", "maxLength": 12}, "days": {"type": "integer", "minimum": 1, "maximum": 7}},
                "required": ["city", "days"], "additionalProperties": false}}},
            {"type": "function", "function": {"name": "noop"}}
        ])))
        .unwrap()
    }

    #[test]
    fn required_tool_calls_parse_and_validate() {
        let e = env();
        let tools = weather();
        let spec = e.compile(&tools_plan(&tools, &Choice::Required, true).unwrap()).unwrap();
        let spec1 = e.compile(&tools_plan(&tools, &Choice::Named("get_weather".into()), true).unwrap()).unwrap();
        for seed in 0..30 {
            for (sp, only_weather) in [(&spec, false), (&spec1, true)] {
                let (text, stopped) = run(sp, seed, 2000, 3.0);
                assert!(stopped, "seed {seed}: {text:?}");
                let pieces = ToolParser::parse_all(&text);
                assert!(!pieces.is_empty());
                for p in pieces {
                    match p {
                        Piece::Call { name, arguments } => {
                            let t = tools.iter().find(|t| t.name == name).expect("known tool");
                            assert!(!only_weather || name == "get_weather");
                            assert!(validates(&t.parameters, &arguments), "seed {seed}: {text}");
                        }
                        Piece::Text(t) => assert!(t.trim().is_empty(), "seed {seed}: text {t:?} in {text:?}"),
                    }
                }
            }
        }
    }

    #[test]
    fn auto_tools_are_lazy() {
        let e = env();
        let tools = weather();
        let spec = e.compile(&tools_plan(&tools, &Choice::Auto, true).unwrap()).unwrap();
        let mut c = spec.start().unwrap();
        assert!(!c.active(), "free text until <tool_call>");
        for &b in b"hi " {
            c.commit(b as u32).unwrap();
        }
        assert!(!c.active());
        c.commit(257).unwrap();
        assert!(c.active());
        let m = c.compute_mask().unwrap();
        assert!(m.allows(b'\n' as u32) && !m.allows(b'{' as u32) && !m.allows(256));
        for &b in br#"
{"name": "noop", "arguments": {}}
"# {
            c.commit(b as u32).unwrap();
        }
        assert!(c.active());
        let m = c.compute_mask().unwrap();
        assert!(
            m.allows(258) && !m.allows(256),
            "only the closing tag: 258 {} 256 {} < {}",
            m.allows(258),
            m.allows(256),
            m.allows(b'<' as u32)
        );
        c.commit(258).unwrap();
        assert!(!c.active(), "re-armed after the call");
        c.commit(b'!' as u32).unwrap();
        c.commit(257).unwrap();
        assert!(c.active(), "a second call is constrained too");
        // a random model under auto: every call it opens validates
        for seed in 0..20 {
            let (text, _) = run_auto(&spec, seed);
            for p in ToolParser::parse_all(&text) {
                if let Piece::Call { name, arguments } = p {
                    let t = tools.iter().find(|t| t.name == name).unwrap();
                    assert!(validates(&t.parameters, &arguments), "{text}");
                }
            }
        }
    }

    /// Like [`run`] but opens a call (token 257) at step 2.
    fn run_auto(spec: &Arc<dyn ConstraintSpec>, seed: u64) -> (String, bool) {
        let mut c = spec.start().unwrap();
        let mut rng = SplitMix64::new(seed);
        let mut out = Vec::new();
        for step in 0..400 {
            let mut l: Vec<f32> = (0..300).map(|_| rng.next_f32()).collect();
            if step == 2 {
                l[257] += 10.0;
            }
            if c.active() {
                assert!(c.compute_mask().unwrap().apply(&mut l));
            } else {
                l[256] = f32::NEG_INFINITY;
                if step > 2 {
                    return (String::from_utf8_lossy(&out).into_owned(), true);
                }
            }
            let t = intarsia_core::sampler::argmax(&l) as u32;
            c.commit(t).unwrap();
            out.extend(bytes_of(t));
        }
        (String::from_utf8_lossy(&out).into_owned(), false)
    }

    /// The real Qwen3 token table (skips without the model; `INTARSIA_TEST_TOKENIZER` overrides the path): EOS ids,
    /// tags, padding to the 151936-wide logits, and the GSM8K schema producing parseable JSON under random logits.
    #[test]
    fn qwen3_token_table() {
        let path = std::env::var("INTARSIA_TEST_TOKENIZER").unwrap_or_else(|_| {
            format!("{}/Models/Qwen3-8B/tokenizer.json", std::env::var("HOME").unwrap_or_default())
        });
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("skip: no {path}");
            return;
        };
        let tj: Value = serde_json::from_str(&text).unwrap();
        let bytes = llguidance::token_bytes_from_tokenizer_json(&tj).unwrap();
        let t0 = std::time::Instant::now();
        // the tags tokenize as one token each (what the HF tokenizer does with added tokens)
        let e = GrammarEnv::from_parts(
            bytes,
            151936,
            &[151645, 151643],
            Box::new(|s| match s {
                OPEN => Some(vec![151657]),
                CLOSE => Some(vec![151658]),
                _ => None,
            }),
        )
        .unwrap();
        eprintln!("token table + factory: {:?}", t0.elapsed());
        assert_eq!((e.open.id, e.close.id), (Some(151657), Some(151658)));
        assert_eq!(e.open.lark, "\"<tool_call>\"", "not special in Qwen3: matched as text");
        // (closed and bounded, so a random "model" ends; localeval's open GSM8K schema is the e2e check's)
        let schema = json!({"properties": {"answer": {"type": "integer", "minimum": -999, "maximum": 999}},
                            "required": ["answer"], "type": "object", "additionalProperties": false});
        let t0 = std::time::Instant::now();
        let spec = e.compile(&Plan::Json(schema)).unwrap();
        eprintln!("compile: {:?}", t0.elapsed());
        let trie = e.factory.tok_env().tok_trie().clone();
        for seed in 0..5 {
            let mut c = spec.start().unwrap();
            let mut rng = SplitMix64::new(seed);
            let mut out = Vec::new();
            let mut worst = std::time::Duration::ZERO;
            for _ in 0..64 {
                let mut l: Vec<f32> = (0..151936).map(|_| rng.next_f32()).collect();
                l[151645] += 0.5;
                let t1 = std::time::Instant::now();
                let m = c.compute_mask().unwrap();
                worst = worst.max(t1.elapsed());
                assert!(m.apply(&mut l));
                assert!(l[151900..].iter().all(|x| *x == f32::NEG_INFINITY), "padding is never allowed");
                let t = intarsia_core::sampler::argmax(&l) as u32;
                if t == 151645 || t == 151643 {
                    break;
                }
                c.commit(t).unwrap();
                out.extend_from_slice(trie.token(t));
            }
            let s = String::from_utf8(out).unwrap();
            eprintln!("seed {seed}: {s} (worst mask {worst:?})");
            let v: Value = serde_json::from_str(&s).unwrap_or_else(|e| panic!("{e}: {s:?}"));
            assert!(v["answer"].is_i64() || v["answer"].is_u64(), "{s}");
        }
    }

    fn split_stream(chunks: &[&str]) -> (String, String) {
        let mut p = ReasoningParser::default();
        let (mut r, mut c) = (String::new(), String::new());
        let mut parts: Vec<Part> = chunks.iter().flat_map(|t| p.push(t)).collect();
        parts.extend(p.finish());
        for part in parts {
            match part {
                Part::Reasoning(t) => r.push_str(&t),
                Part::Content(t) => c.push_str(&t),
            }
        }
        (r, c)
    }

    #[test]
    fn reasoning_parser_splits_at_the_close_tag_however_it_is_chunked() {
        let text = "<think>\nlet me see\n</think>\n\n{\"a\": 1}";
        let want = ("let me see".to_string(), "{\"a\": 1}".to_string());
        assert_eq!(split_stream(&[text]), want);
        let bytes: Vec<String> = text.chars().map(String::from).collect();
        let refs: Vec<&str> = bytes.iter().map(String::as_str).collect();
        assert_eq!(split_stream(&refs), want);
        // opener in the prompt (not generated)
        assert_eq!(split_stream(&["\nlet me see\n", "</think>", "\n\n{\"a\": 1}"]), want);
        // a `<` that is not a tag stays in the reasoning
        assert_eq!(split_stream(&["a < b </th", "ink> done"]), ("a < b".to_string(), "done".to_string()));
        // no close tag: all reasoning; empty think section
        assert_eq!(split_stream(&["<think>\nunfinished"]), ("unfinished".to_string(), String::new()));
        assert_eq!(split_stream(&["<think>\n\n</think>\n\nhi"]), (String::new(), "hi".to_string()));
        assert_eq!(ReasoningParser::parse_all(text), (Some("let me see".into()), Some("{\"a\": 1}".into())));
        assert_eq!(ReasoningParser::parse_all(""), (None, None));
    }

    #[test]
    fn think_gate_leaves_the_think_section_unconstrained_then_applies_the_grammar() {
        let e = env();
        let schema = json!({"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"],
                            "additionalProperties": false});
        let spec = e.compile_with(&Plan::Json(schema), true).unwrap();
        let mut c = spec.start().unwrap();
        // thinking: not constrained, even for tokens the grammar would never allow, and a hermes tag changes nothing
        assert!(!c.active());
        for t in tokenize("hmm <tool_call> {x") {
            c.commit(t).unwrap();
            assert!(!c.active());
        }
        c.commit(259).unwrap(); // </think>
        assert!(c.active());
        let m = c.compute_mask().unwrap();
        assert!(m.allows(b'{' as u32) && !m.allows(b'[' as u32) && !m.allows(256));
        for t in tokenize("{\"n\": 1}") {
            assert!(c.compute_mask().unwrap().allows(t));
            c.commit(t).unwrap();
        }
        let m = c.compute_mask().unwrap();
        assert!(m.allows(256) && !m.allows(b'{' as u32));
        // tools: required calls start only after the gate, auto keeps its own trigger after it
        let tools = weather();
        let spec = e.compile_with(&tools_plan(&tools, &Choice::Required, true).unwrap(), true).unwrap();
        let mut c = spec.start().unwrap();
        assert!(!c.active());
        c.commit(259).unwrap();
        assert!(c.active() && c.compute_mask().unwrap().allows(257));
        let spec = e.compile_with(&tools_plan(&tools, &Choice::Auto, true).unwrap(), true).unwrap();
        let mut c = spec.start().unwrap();
        c.commit(257).unwrap(); // a <tool_call> inside the think section does not trigger
        assert!(!c.active());
        c.commit(259).unwrap();
        assert!(!c.active());
        c.commit(257).unwrap();
        assert!(c.active());
    }

    #[test]
    fn parallel_false_allows_one_call() {
        let e = env();
        let tools = weather();
        let call = br#"
{"name": "noop", "arguments": {}}
"#;
        // required: after one call only EOS
        let spec = e.compile(&tools_plan(&tools, &Choice::Required, false).unwrap()).unwrap();
        let mut c = spec.start().unwrap();
        c.commit(257).unwrap();
        for &b in call {
            c.commit(b as u32).unwrap();
        }
        c.commit(258).unwrap();
        let m = c.compute_mask().unwrap();
        assert!(m.allows(256) && !m.allows(b'\n' as u32));
        // auto: likewise after the first call closes
        let spec = e.compile(&tools_plan(&tools, &Choice::Auto, false).unwrap()).unwrap();
        let mut c = spec.start().unwrap();
        c.commit(b'a' as u32).unwrap();
        c.commit(257).unwrap();
        for &b in call {
            c.commit(b as u32).unwrap();
        }
        c.commit(258).unwrap();
        assert!(c.active());
        let m = c.compute_mask().unwrap();
        assert!(m.allows(256) && !m.allows(b'a' as u32) && !m.allows(257));
        // choose(): the request-level rules
        let none = Choice::None;
        assert!(choose(Some(Plan::Json(json!({}))), &tools, &Choice::Auto, true, true).is_err());
        assert!(matches!(
            choose(Some(Plan::Json(json!({}))), &tools, &none, true, true),
            Ok((Some(Plan::Json(_)), false))
        ));
        assert!(matches!(choose(None, &tools, &Choice::Auto, true, false), Ok((None, false))));
        assert!(choose(None, &tools, &Choice::Required, true, false).is_err());
        assert!(matches!(choose(None, &tools, &Choice::Auto, true, true), Ok((Some(Plan::Tools { .. }), true))));
    }

    #[test]
    fn tool_request_errors() {
        let tools = weather();
        for (v, why) in [
            (json!("sometimes"), "unsupported tool_choice"),
            (json!({"type": "function", "function": {"name": "nope"}}), "not in tools"),
            (json!({"type": "function"}), "name is required"),
            (json!(3), "string or an object"),
        ] {
            let e = openai_choice(&Some(v), &tools).unwrap_err();
            assert!(e.contains(why), "{e}");
        }
        assert_eq!(openai_choice(&None, &tools).unwrap(), Choice::Auto);
        assert_eq!(openai_choice(&None, &[]).unwrap(), Choice::None);
        assert!(openai_choice(&Some(json!("required")), &[]).unwrap_err().contains("needs tools"));
        for (v, why) in [
            (json!([{"type": "retrieval"}]), "type 'function'"),
            (json!([{"type": "function", "function": {"name": "a b"}}]), "invalid tool name"),
            (json!([{"type": "function", "function": {"name": "a", "parameters": "x"}}]), "JSON schema"),
            (
                json!([{"type": "function", "function": {"name": "a"}}, {"type": "function", "function": {"name": "a"}}]),
                "duplicate",
            ),
            (json!({}), "must be an array"),
        ] {
            let e = openai_tools(&Some(v)).unwrap_err();
            assert!(e.contains(why), "{e}");
        }
        let at = anthropic_tools(&Some(json!([{"name": "w", "input_schema": {"type": "object"}}]))).unwrap();
        assert_eq!(anthropic_choice(&Some(json!({"type": "any"})), &at).unwrap(), Choice::Required);
        assert_eq!(
            anthropic_choice(&Some(json!({"type": "tool", "name": "w"})), &at).unwrap(),
            Choice::Named("w".into())
        );
        assert!(anthropic_tools(&Some(json!([{"name": "w"}]))).unwrap_err().contains("input_schema"));
        assert!(anthropic_tools(&Some(json!([{"type": "web_search_20250305", "name": "w"}]))).is_err());
        // an argument schema llguidance cannot compile is a 400 at compile time
        let bad = openai_tools(&Some(json!([{"type": "function", "function": {"name": "a",
            "parameters": {"type": "object", "properties": {"x": {"type": "wat"}}}}}])))
        .unwrap();
        assert!(env().compile(&tools_plan(&bad, &Choice::Required, true).unwrap()).is_err());
    }

    #[test]
    fn tool_parser_streams() {
        let text = "Let me check.\n<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call>\n<tool_call>\n{\"name\": \"g\", \"arguments\": \"{\\\"b\\\": 2}\"}\n</tool_call>";
        let want = vec![
            Piece::Text("Let me check.\n".into()),
            Piece::Call { name: "f".into(), arguments: json!({"a": 1}) },
            Piece::Text("\n".into()),
            Piece::Call { name: "g".into(), arguments: json!({"b": 2}) },
        ];
        assert_eq!(ToolParser::parse_all(text), want);
        // byte-by-byte streaming gives the same calls, and never releases a partial tag as text
        let mut p = ToolParser::default();
        let mut got = Vec::new();
        for ch in text.chars() {
            for piece in p.push(&ch.to_string()) {
                if let Piece::Text(t) = &piece {
                    assert!(!t.contains('<'), "partial tag leaked: {t:?}");
                }
                got.push(piece);
            }
        }
        got.extend(p.finish());
        let calls: Vec<_> = got.iter().filter(|p| matches!(p, Piece::Call { .. })).collect();
        assert_eq!(calls, want.iter().filter(|p| matches!(p, Piece::Call { .. })).collect::<Vec<_>>());
        let txt: String =
            got.iter().filter_map(|p| if let Piece::Text(t) = p { Some(t.as_str()) } else { None }).collect();
        assert_eq!(txt, "Let me check.\n\n");
        // a bad block stays text; an unterminated one too; "<" that is not a tag is released at the end
        assert_eq!(
            ToolParser::parse_all("<tool_call>nope</tool_call>"),
            vec![Piece::Text("<tool_call>nope</tool_call>".into())]
        );
        // the closing tag inside an argument string does not end the call
        let t = "<tool_call>{\"name\": \"f\", \"arguments\": {\"s\": \"a</tool_call>b\"}}</tool_call>";
        assert_eq!(
            ToolParser::parse_all(t),
            vec![Piece::Call { name: "f".into(), arguments: json!({"s": "a</tool_call>b"}) }]
        );
        assert_eq!(
            ToolParser::parse_all("a<tool_call>{"),
            vec![Piece::Text("a".into()), Piece::Text("<tool_call>{".into())]
        );
        assert_eq!(ToolParser::parse_all("x <tool"), vec![Piece::Text("x ".into()), Piece::Text("<tool".into())]);
    }
}
