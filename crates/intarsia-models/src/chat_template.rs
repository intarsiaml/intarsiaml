//! Hugging Face chat templates rendered with minijinja.
//!
//! Matches what `transformers` does closely enough for the templates we ship
//! against: `trim_blocks` + `lstrip_blocks`, Python string methods
//! (`startswith`, `split`, `strip`, ...), `raise_exception`, `strftime_now`
//! and a `tojson` that prints like Python's `json.dumps` (", " and ": "
//! separators, no ASCII escaping).

use intarsia_core::engine::ChatOpts;
use intarsia_core::{Error, Result};
use minijinja::{Environment, ErrorKind, Value};
use serde_json::Value as Json;
use std::path::Path;

/// Where the template comes from.
#[derive(Clone, Debug, Default)]
pub enum ChatTemplateSource {
    /// `chat_template.jinja` if present (what transformers prefers), else `tokenizer_config.json`.
    #[default]
    Auto,
    /// Use the `chat_template` entry of `tokenizer_config.json` even if a `.jinja` file exists.
    TokenizerConfig,
    /// An explicit jinja file.
    File(std::path::PathBuf),
}

pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: Json,
    eos_token: Json,
    pub source: String,
}

struct PyJson;

impl serde_json::ser::Formatter for PyJson {
    fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

fn py_json(v: Value) -> std::result::Result<String, minijinja::Error> {
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PyJson);
    serde::Serialize::serialize(&v, &mut ser)
        .map_err(|e| minijinja::Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    String::from_utf8(buf).map_err(|e| minijinja::Error::new(ErrorKind::InvalidOperation, e.to_string()))
}

/// Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn strftime_now(fmt: String) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let secs =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let (y, m, d) = civil(secs.div_euclid(86_400));
    let tod = secs.rem_euclid(86_400);
    let mut out = String::new();
    let mut it = fmt.chars();
    while let Some(c) = it.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('Y') => out.push_str(&y.to_string()),
            Some('y') => out.push_str(&format!("{:02}", y % 100)),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('B') => out.push_str(MONTHS[m as usize - 1]),
            Some('b') => out.push_str(&MONTHS[m as usize - 1][..3]),
            Some('H') => out.push_str(&format!("{:02}", tod / 3600)),
            Some('M') => out.push_str(&format!("{:02}", tod % 3600 / 60)),
            Some('S') => out.push_str(&format!("{:02}", tod % 60)),
            Some('%') => out.push('%'),
            Some(o) => {
                out.push('%');
                out.push(o);
            }
            None => out.push('%'),
        }
    }
    out
}

impl ChatTemplate {
    pub fn from_source(template: &str, source: &str, bos: Json, eos: Json) -> Result<ChatTemplate> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_keep_trailing_newline(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> std::result::Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", strftime_now);
        env.add_filter("tojson", py_json);
        env.add_template_owned("chat", template.to_string())
            .map_err(|e| Error::Text(format!("chat template does not compile: {e}")))?;
        Ok(ChatTemplate { env, bos_token: bos, eos_token: eos, source: source.to_string() })
    }

    /// Load according to `src` from a model directory.
    pub fn load(dir: &Path, src: &ChatTemplateSource) -> Result<Option<ChatTemplate>> {
        let cfg_path = dir.join("tokenizer_config.json");
        let cfg: Json = if cfg_path.exists() {
            serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)
                .map_err(|e| Error::Text(format!("tokenizer_config.json: {e}")))?
        } else {
            Json::Null
        };
        let tok = |k: &str| match cfg.get(k) {
            Some(Json::Object(o)) => o.get("content").cloned().unwrap_or(Json::Null),
            Some(v) => v.clone(),
            None => Json::Null,
        };
        let (bos, eos) = (tok("bos_token"), tok("eos_token"));
        let from_cfg = || -> Option<String> {
            match cfg.get("chat_template")? {
                Json::String(s) => Some(s.clone()),
                Json::Array(a) => a
                    .iter()
                    .find(|t| t.get("name").and_then(|n| n.as_str()) == Some("default"))
                    .or_else(|| a.first())
                    .and_then(|t| t.get("template")?.as_str().map(String::from)),
                _ => None,
            }
        };
        let (text, origin) = match src {
            ChatTemplateSource::File(p) => (Some(std::fs::read_to_string(p)?), format!("file:{}", p.display())),
            ChatTemplateSource::TokenizerConfig => (from_cfg(), "tokenizer_config.json".into()),
            ChatTemplateSource::Auto => {
                let j = dir.join("chat_template.jinja");
                if j.exists() {
                    (Some(std::fs::read_to_string(&j)?), "chat_template.jinja".into())
                } else {
                    (from_cfg(), "tokenizer_config.json".into())
                }
            }
        };
        match text {
            Some(t) => Ok(Some(ChatTemplate::from_source(&t, &origin, bos, eos)?)),
            None => Ok(None),
        }
    }

    pub fn render(&self, messages: &[Json], opts: &ChatOpts) -> Result<String> {
        let mut ctx = serde_json::Map::new();
        ctx.insert("messages".into(), Json::Array(messages.to_vec()));
        ctx.insert("add_generation_prompt".into(), Json::Bool(opts.add_generation_prompt));
        ctx.insert("bos_token".into(), self.bos_token.clone());
        ctx.insert("eos_token".into(), self.eos_token.clone());
        if let Some(t) = opts.enable_thinking {
            ctx.insert("enable_thinking".into(), Json::Bool(t));
        }
        if let Some(t) = &opts.tools {
            ctx.insert("tools".into(), t.clone());
        }
        for (k, v) in &opts.extra {
            ctx.insert(k.clone(), v.clone());
        }
        let tmpl = self.env.get_template("chat").map_err(|e| Error::Text(e.to_string()))?;
        tmpl.render(Value::from_serialize(&Json::Object(ctx))).map_err(|e| Error::Text(format!("chat template: {e:#}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const QWEN3: &str = include_str!("../tests/fixtures/qwen3_chat_template.jinja");

    fn tmpl() -> ChatTemplate {
        ChatTemplate::from_source(QWEN3, "fixture", Json::Null, json!("<|im_end|>")).unwrap()
    }

    #[test]
    fn qwen3_basic_and_thinking_off() {
        let msgs = vec![json!({"role": "system", "content": "Be brief."}), json!({"role": "user", "content": "Hi"})];
        let on = tmpl().render(&msgs, &ChatOpts { add_generation_prompt: true, ..Default::default() }).unwrap();
        assert_eq!(
            on,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
        let off = tmpl()
            .render(
                &msgs,
                &ChatOpts { add_generation_prompt: true, enable_thinking: Some(false), ..Default::default() },
            )
            .unwrap();
        assert!(off.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "{off:?}");
    }

    #[test]
    fn qwen3_assistant_history_strips_reasoning() {
        let msgs = vec![
            json!({"role": "user", "content": "q1"}),
            json!({"role": "assistant", "content": "<think>\nhmm\n</think>\n\nanswer1"}),
            json!({"role": "user", "content": "q2"}),
        ];
        let out = tmpl().render(&msgs, &ChatOpts { add_generation_prompt: true, ..Default::default() }).unwrap();
        assert!(out.contains("<|im_start|>assistant\nanswer1<|im_end|>"), "{out:?}");
    }

    #[test]
    fn tools_use_python_style_json() {
        let msgs = vec![json!({"role": "user", "content": "weather?"})];
        let tools = json!([{"type": "function", "function": {"name": "w", "parameters": {"a": 1}}}]);
        let out = tmpl()
            .render(&msgs, &ChatOpts { add_generation_prompt: true, tools: Some(tools), ..Default::default() })
            .unwrap();
        assert!(out.contains(r#"{"type": "function", "function": {"name": "w", "parameters": {"a": 1}}}"#), "{out}");
    }

    #[test]
    fn raise_exception_surfaces() {
        let t = ChatTemplate::from_source("{{ raise_exception('boom') }}", "t", Json::Null, Json::Null).unwrap();
        let err = t.render(&[], &ChatOpts::default()).unwrap_err().to_string();
        assert!(err.contains("boom"), "{err}");
    }
}
