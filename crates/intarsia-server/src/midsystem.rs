//! A system message that is not the first message (agent frameworks inject reminders mid-conversation).
//!
//! The Qwen3.x chat templates accept a system message only first and otherwise raise `System message must be at the
//! beginning.` (HTTP 400, the same as vLLM, which renders the same template). `INTARSIA_SYSTEM_MIDCONV` (opt-in) rewrites
//! such a message before templating; unset, nothing changes except that the 400 names the variable.
//! A conversation without a non-leading system message is never touched, in any mode (its prompt is byte-identical).
//! Lane sysmsg, docs/results/2026-11-18-sysmsg.md.

use serde_json::{json, Value};

pub const ENV: &str = "INTARSIA_SYSTEM_MIDCONV";
const ERR: &str = "System message must be at the beginning";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Unset: the template's 400.
    Off,
    /// The message stays where it is, as a user-role message: earlier turns render as before (prefix-store stable).
    User,
    /// The text joins the leading system message (one is made if there is none) and the message leaves its place.
    Merge,
}

impl Mode {
    pub fn from_env() -> Mode {
        Mode::parse(std::env::var(ENV).ok().as_deref())
    }

    pub fn parse(v: Option<&str>) -> Mode {
        match v.map(str::trim) {
            None | Some("") | Some("0") | Some("off") => Mode::Off,
            Some("user") | Some("1") => Mode::User,
            Some("merge") => Mode::Merge,
            Some(other) => {
                eprintln!("warning: {ENV}={other:?} is not off|user|merge; using off");
                Mode::Off
            }
        }
    }
}

fn is_system(m: &Value) -> bool {
    m.get("role").and_then(Value::as_str) == Some("system")
}

/// True when some message after the first is a system message.
pub fn has_mid_system(messages: &[Value]) -> bool {
    messages.iter().skip(1).any(is_system)
}

/// `messages` as the template should see them (content already flattened to strings).
pub fn rewrite(messages: Vec<Value>, mode: Mode) -> Vec<Value> {
    if mode == Mode::Off || !has_mid_system(&messages) {
        return messages;
    }
    let text = |m: &Value| m.get("content").and_then(Value::as_str).unwrap_or("").to_string();
    match mode {
        Mode::Off => unreachable!(),
        Mode::User => messages
            .into_iter()
            .enumerate()
            .map(|(i, mut m)| {
                if i > 0 && is_system(&m) {
                    m["role"] = json!("user");
                    m["content"] = json!(format!("[System message]\n{}", text(&m)));
                }
                m
            })
            .collect(),
        Mode::Merge => {
            let mut out: Vec<Value> = Vec::with_capacity(messages.len());
            let mut extra: Vec<String> = Vec::new();
            for (i, m) in messages.into_iter().enumerate() {
                if i > 0 && is_system(&m) {
                    let t = text(&m);
                    if !t.trim().is_empty() {
                        extra.push(t);
                    }
                } else {
                    out.push(m);
                }
            }
            let tail = extra.join("\n\n");
            if tail.is_empty() {
                return out;
            }
            match out.first_mut().filter(|m| is_system(m)) {
                Some(first) => {
                    let head = text(first);
                    first["content"] = json!(if head.trim().is_empty() { tail } else { format!("{head}\n\n{tail}") });
                }
                None => out.insert(0, json!({"role": "system", "content": tail})),
            }
            out
        }
    }
}

/// Name the fix in the template's error (only when the request has the shape the template refuses and the mode is off).
pub fn explain(err: intarsia_core::Error, had_mid_system: bool) -> intarsia_core::Error {
    if had_mid_system && err.to_string().contains(ERR) {
        return intarsia_core::Error::Text(format!(
            "{err} (this chat template accepts a system message only as the first message; \
             set {ENV}=user on the server to render later system messages as user messages, or {ENV}=merge to \
             join them into the leading system message, or send them yourself in one of those forms)"
        ));
    }
    err
}

#[cfg(test)]
mod tests {
    use super::*;
    use intarsia_core::engine::ChatOpts;
    use intarsia_models::chat_template::ChatTemplate;

    const TEMPLATES: &[(&str, &str)] = &[
        ("qwen3.8-27b", include_str!("../tests/fixtures/qwen3_8_27b_chat_template.jinja")),
        ("qwen3.6-35b-a3b", include_str!("../tests/fixtures/qwen3_6_35b_a3b_chat_template.jinja")),
        ("qwen3.5-4b", include_str!("../tests/fixtures/qwen3_5_4b_chat_template.jinja")),
    ];

    fn msg(role: &str, c: &str) -> Value {
        json!({"role": role, "content": c})
    }
    fn tmpl(src: &str, name: &str) -> ChatTemplate {
        ChatTemplate::from_source(src, name, Value::Null, Value::Null).unwrap()
    }
    fn opts() -> ChatOpts {
        ChatOpts { add_generation_prompt: true, ..Default::default() }
    }
    fn mid() -> Vec<Value> {
        vec![msg("system", "S0"), msg("user", "u1"), msg("assistant", "a1"), msg("system", "reminder"), msg("user", "u2")]
    }

    #[test]
    fn parse_modes() {
        assert_eq!(Mode::parse(None), Mode::Off);
        assert_eq!(Mode::parse(Some("")), Mode::Off);
        assert_eq!(Mode::parse(Some("user")), Mode::User);
        assert_eq!(Mode::parse(Some("merge")), Mode::Merge);
        assert_eq!(Mode::parse(Some("bogus")), Mode::Off);
    }

    /// Reproduction: the real templates refuse a mid-conversation system message; the error names the fix.
    #[test]
    fn templates_refuse_mid_system_and_error_names_the_fix() {
        for (name, src) in TEMPLATES {
            let e = tmpl(src, name).render(&mid(), &opts()).unwrap_err();
            assert!(e.to_string().contains(ERR), "{name}: {e}");
            let shown = explain(e, has_mid_system(&mid())).to_string();
            assert!(shown.contains(ENV) && shown.contains(ERR), "{name}: {shown}");
        }
    }

    #[test]
    fn user_mode_keeps_the_prefix_and_renders() {
        for (name, src) in TEMPLATES {
            let t = tmpl(src, name);
            let out = t.render(&rewrite(mid(), Mode::User), &opts()).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(out.contains("<|im_start|>user\n[System message]\nreminder<|im_end|>\n"), "{name}: {out}");
            // the turns before the reminder keep their bytes (what a prefix store holds for the earlier requests)
            let head = t.render(&mid()[..2], &ChatOpts { add_generation_prompt: false, ..opts() }).unwrap();
            assert!(out.starts_with(&head), "{name}");
        }
    }

    #[test]
    fn merge_mode_joins_into_the_leading_system_message() {
        for (name, src) in TEMPLATES {
            let t = tmpl(src, name);
            let out = t.render(&rewrite(mid(), Mode::Merge), &opts()).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(out.contains("S0\n\nreminder<|im_end|>\n"), "{name}: {out}");
            assert_eq!(out.matches("<|im_start|>system").count(), 1, "{name}");
        }
        // no leading system message: one is made
        let m = vec![msg("user", "u1"), msg("system", "r"), msg("user", "u2")];
        let r = rewrite(m, Mode::Merge);
        assert_eq!(r[0], msg("system", "r"));
        assert_eq!(r.len(), 3);
    }

    /// Requests that are already valid render byte-identically in every mode.
    #[test]
    fn valid_requests_are_untouched_in_every_mode() {
        let convs = vec![
            vec![msg("user", "hi")],
            vec![msg("system", "S"), msg("user", "hi")],
            vec![msg("system", "S"), msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2")],
        ];
        for (name, src) in TEMPLATES {
            let t = tmpl(src, name);
            for c in &convs {
                let base = t.render(c, &opts()).unwrap();
                for mode in [Mode::Off, Mode::User, Mode::Merge] {
                    assert_eq!(rewrite(c.clone(), mode), *c, "{name} {mode:?}");
                    assert_eq!(t.render(&rewrite(c.clone(), mode), &opts()).unwrap(), base, "{name} {mode:?}");
                }
            }
        }
    }
}
