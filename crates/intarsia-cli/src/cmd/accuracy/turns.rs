//! Which rows of a chat prompt the model is asked to predict (the rows the KL floor scores).
//!
//! A golden stores one reference row per position of the whole templated prompt: the system and user turns as well as
//! the assistant turns. Serving never asks the model to predict the tokens of a system or user turn, only those of an
//! assistant turn, so the floor scores a row only when its target token lies outside every system and user turn.
//! docs/results/2026-10-21-m3-accuracy-debt.md (section 1, 7.1) measured why this matters: on `chat_reuse_4k` those
//! rows are 12.6 % of the prompt and carry 71-73 % of its KL, and a handful of `<|im_end|>` flips dominate them.
//!
//! * **Row convention.** Row `t` is the distribution over token `t + 1` after `ids[..=t]` (`tools/golden_vllm.py`).
//!   A turn spans the token indices `a` (its turn-open token) to `b` (its turn-close token). The model is not asked to
//!   predict the tokens `a + 1 ..= b` of a hidden turn (its role and text up to and including the close token), so the
//!   rows `a .. b` (that is `a ..= b - 1`) are not scored. The row whose target is the turn-open token `a` stays scored
//!   (it is the row before the turn), and so does the row after the close token. A turn that never closes is hidden
//!   to the end of the prompt. A final open assistant turn is scored.
//! * **Per tokenizer / template.** A [`TurnFormat`] names the open and close markers and the hidden roles as text; they
//!   are turned into token ids with the model's own tokenizer ([`TurnFormat::resolve`]), never hard-coded ids. Each
//!   marker must be a single token and the role must tokenize right after the open marker, or resolving fails loudly.
//!   A format is picked by the golden's `template` metadata, then by the recipe (`[accuracy] chat_turns`).
//! * **Plain text.** A prompt with no turn-open token (and any prompt the golden marks `kind = "raw"`) keeps all rows.

use anyhow::{bail, Result};

/// A chat template's turn markers, as text.
#[derive(Debug)]
pub struct TurnFormat {
    pub name: &'static str,
    /// Starts a turn; the role follows it directly.
    pub open: &'static str,
    /// Ends a turn.
    pub close: &'static str,
    /// Roles whose turns the model is not asked to predict.
    pub hidden_roles: &'static [&'static str],
}

/// ChatML (Qwen): `<|im_start|>role\ncontent<|im_end|>\n`.
pub const CHATML: TurnFormat =
    TurnFormat { name: "chatml", open: "<|im_start|>", close: "<|im_end|>", hidden_roles: &["system", "user"] };

/// Llama 3 / 3.1: `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`. The role follows the open marker directly; the
/// header-close token and the blank line after it are inside the hidden span (open .. close).
pub const LLAMA3: TurnFormat =
    TurnFormat { name: "llama3", open: "<|start_header_id|>", close: "<|eot_id|>", hidden_roles: &["system", "user"] };

/// Turn format by name (the golden's `template` metadata or the recipe's `chat_turns`).
pub fn format_named(name: &str) -> Option<&'static TurnFormat> {
    match name.trim().to_ascii_lowercase().as_str() {
        "chatml" => Some(&CHATML),
        "llama3" => Some(&LLAMA3),
        _ => None,
    }
}

/// A [`TurnFormat`] resolved to token ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Turns {
    pub format: &'static str,
    open: u32,
    close: u32,
    /// Token ids of each hidden role's name.
    hidden: Vec<Vec<u32>>,
}

impl TurnFormat {
    /// Token ids of the markers and hidden roles, through `tokenize` (text -> ids, special tokens parsed, no BOS).
    pub fn resolve(&'static self, tokenize: &dyn Fn(&str) -> Result<Vec<u32>>) -> Result<Turns> {
        let single = |what: &str, text: &str| -> Result<u32> {
            match tokenize(text)?.as_slice() {
                [id] => Ok(*id),
                other => bail!("turn format {}: {what} {text:?} is not a single token (got {other:?})", self.name),
            }
        };
        let open = single("turn-open marker", self.open)?;
        let close = single("turn-close marker", self.close)?;
        let mut hidden = Vec::new();
        for role in self.hidden_roles {
            // opener and role together: the role must tokenize right after the opener, as in a rendered prompt
            let ids = tokenize(&format!("{}{}", self.open, role))?;
            match ids.split_first() {
                Some((&o, rest)) if o == open && !rest.is_empty() => hidden.push(rest.to_vec()),
                _ => bail!("turn format {}: {:?} does not tokenize as the turn-open token plus the role (got {ids:?})", self.name, format!("{}{}", self.open, role)),
            }
        }
        Ok(Turns { format: self.name, open, close, hidden })
    }
}

impl Turns {
    /// `scored[t]`: row `t` of a prompt with token ids `ids` is a row the model is asked to predict.
    pub fn scored_rows(&self, ids: &[u32]) -> Vec<bool> {
        let n = ids.len();
        let mut scored = vec![true; n];
        let mut t = 0;
        while t < n {
            if ids[t] != self.open {
                t += 1;
                continue;
            }
            let after = &ids[t + 1..];
            if !self.hidden.iter().any(|r| after.starts_with(r)) {
                t += 1; // an assistant turn (or an unknown role): scored
                continue;
            }
            // the turn closes at the first close token after the opener; unterminated: hidden to the end
            let b = ids[t + 1..].iter().position(|&x| x == self.close).map_or(n, |p| t + 1 + p);
            for s in scored.iter_mut().take(b.min(n)).skip(t) {
                *s = false;
            }
            t = b + 1;
        }
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPEN: u32 = 100;
    const CLOSE: u32 = 101;
    const NL: u32 = 5;
    const SYSTEM: u32 = 1;
    const USER: u32 = 2;
    const ASSISTANT: u32 = 3;

    /// A synthetic ChatML tokenizer: the markers and roles are single tokens.
    fn tok(text: &str) -> Result<Vec<u32>> {
        Ok(match text {
            "<|im_start|>" => vec![OPEN],
            "<|im_end|>" => vec![CLOSE],
            "<|im_start|>system" => vec![OPEN, SYSTEM],
            "<|im_start|>user" => vec![OPEN, USER],
            other => bail!("unexpected {other:?}"),
        })
    }

    fn turns() -> Turns {
        CHATML.resolve(&tok).unwrap()
    }

    #[test]
    fn llama3_format_is_named_and_resolves_to_its_markers() {
        let f = format_named("Llama3").unwrap();
        assert_eq!((f.name, f.open, f.close), ("llama3", "<|start_header_id|>", "<|eot_id|>"));
        let tok = |text: &str| -> Result<Vec<u32>> {
            Ok(match text {
                "<|start_header_id|>" => vec![OPEN],
                "<|eot_id|>" => vec![CLOSE],
                "<|start_header_id|>system" => vec![OPEN, SYSTEM],
                "<|start_header_id|>user" => vec![OPEN, USER],
                other => bail!("unexpected {other:?}"),
            })
        };
        assert_eq!(f.resolve(&tok).unwrap().format, "llama3");
    }

    /// ChatML render of `(role, body)` turns: `<|im_start|>role \n body <|im_end|> \n`.
    fn render(parts: &[(u32, &[u32])]) -> Vec<u32> {
        let mut v = Vec::new();
        for (role, body) in parts {
            v.extend([OPEN, *role, NL]);
            v.extend_from_slice(body);
            v.extend([CLOSE, NL]);
        }
        v
    }

    /// Rows not scored, as `(first, last)` runs.
    fn hidden_runs(scored: &[bool]) -> Vec<(usize, usize)> {
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for (i, s) in scored.iter().enumerate() {
            if !s {
                match runs.last_mut() {
                    Some(r) if r.1 + 1 == i => r.1 = i,
                    _ => runs.push((i, i)),
                }
            }
        }
        runs
    }

    #[test]
    fn plain_text_keeps_every_row() {
        let ids: Vec<u32> = (200..260).collect();
        assert!(turns().scored_rows(&ids).iter().all(|&s| s));
        assert!(turns().scored_rows(&[]).is_empty());
    }

    #[test]
    fn system_and_user_hidden_assistant_scored() {
        // 0..: OPEN SYSTEM NL 10 11 CLOSE NL | OPEN USER NL 12 CLOSE NL | OPEN ASSISTANT NL 13 14 CLOSE NL
        let ids = render(&[(SYSTEM, &[10, 11]), (USER, &[12]), (ASSISTANT, &[13, 14])]);
        assert_eq!(ids.len(), 7 + 6 + 7);
        let sc = turns().scored_rows(&ids);
        // system turn opens at 0 and closes at 5: rows 0..=4 (targets 1..=5, the role through <|im_end|>) are hidden;
        // the row whose target is the second OPEN (index 7) is row 6, scored; user turn 7..=11: rows 7..=10 hidden
        assert_eq!(hidden_runs(&sc), vec![(0, 4), (7, 10)]);
        assert!(sc[5], "the row predicting the newline after <|im_end|> is scored");
        assert!(sc[6], "the row predicting the next <|im_start|> is scored");
        assert!(sc[11] && sc[12], "the assistant turn is scored, including the row that predicts its <|im_end|>");
        assert!(sc[13..].iter().all(|&s| s));
        assert_eq!(sc.iter().filter(|&&s| !s).count(), 5 + 4);
    }

    #[test]
    fn multi_turn_two_user_turns() {
        let ids = render(&[(SYSTEM, &[10]), (USER, &[11, 12]), (ASSISTANT, &[13]), (USER, &[14]), (ASSISTANT, &[15, 16])]);
        let sc = turns().scored_rows(&ids);
        // lengths: system 6, user 7, assistant 6, user 6, assistant 7
        let starts = [0usize, 6, 13, 19, 25];
        assert_eq!(ids.len(), 32);
        assert_eq!(hidden_runs(&sc), vec![(starts[0], starts[0] + 3), (starts[1], starts[1] + 4), (starts[3], starts[3] + 3)]);
        // every assistant row is scored
        for &a in &[starts[2], starts[4]] {
            assert!(sc[a..a + 6].iter().all(|&s| s));
        }
    }

    #[test]
    fn open_assistant_turn_at_the_end_is_scored() {
        // a generation prompt: ... <|im_start|> assistant \n  with no body and no close
        let mut ids = render(&[(USER, &[11])]);
        ids.extend([OPEN, ASSISTANT, NL]);
        let sc = turns().scored_rows(&ids);
        assert_eq!(hidden_runs(&sc), vec![(0, 3)]);
        assert!(sc[4..].iter().all(|&s| s));
    }

    #[test]
    fn unterminated_user_turn_is_hidden_to_the_end() {
        let ids = vec![OPEN, SYSTEM, NL, 10, CLOSE, NL, OPEN, USER, NL, 11, 12];
        let sc = turns().scored_rows(&ids);
        assert_eq!(hidden_runs(&sc), vec![(0, 3), (6, 10)]);
    }

    #[test]
    fn text_before_the_first_turn_is_scored() {
        let mut ids = vec![200, 201, 202];
        ids.extend(render(&[(USER, &[11])]));
        let sc = turns().scored_rows(&ids);
        assert!(sc[..3].iter().all(|&s| s)); // incl. row 2, whose target is the first OPEN
        assert_eq!(hidden_runs(&sc), vec![(3, 6)]);
    }

    #[test]
    fn resolve_refuses_an_ambiguous_tokenizer() {
        let multi = |t: &str| -> Result<Vec<u32>> { Ok(if t == "<|im_start|>" { vec![7, 8] } else { tok(t)? }) };
        assert!(CHATML.resolve(&multi).is_err());
        let norole = |t: &str| -> Result<Vec<u32>> { Ok(if t.ends_with("user") { vec![OPEN] } else { tok(t)? }) };
        assert!(CHATML.resolve(&norole).is_err());
    }

    #[test]
    fn formats_by_name() {
        assert_eq!(format_named("ChatML").map(|f| f.name), Some("chatml"));
        assert!(format_named("hf").is_none());
    }
}
