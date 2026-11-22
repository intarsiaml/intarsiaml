//! Tokenizer (`tokenizers` crate) + chat template as a [`TextCodec`].

use crate::chat_template::{ChatTemplate, ChatTemplateSource};
use crate::inkling_chat::InklingChat;
use intarsia_core::engine::{ChatOpts, TextCodec};
use intarsia_core::{Error, Result};
use serde_json::Value;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct HfCodec {
    tok: Tokenizer,
    /// `tokenizer.json`, re-read on demand for structured output's token table.
    path: std::path::PathBuf,
    template: Option<ChatTemplate>,
    /// Inkling-2's own renderer (vLLM's, see `inkling_chat.rs`): present when the vocabulary has its framing tokens.
    inkling: Option<InklingChat>,
}

impl HfCodec {
    pub fn load(dir: &Path, src: &ChatTemplateSource) -> Result<HfCodec> {
        let path = dir.join("tokenizer.json");
        let tok = Tokenizer::from_file(&path).map_err(|e| Error::Text(format!("loading {}: {e}", path.display())))?;
        let template = ChatTemplate::load(dir, src)?;
        let inkling = InklingChat::from_tokenizer_json(&std::fs::read_to_string(&path)?)?;
        Ok(HfCodec { tok, path, template, inkling })
    }

    pub fn template_source(&self) -> Option<&str> {
        self.template.as_ref().map(|t| t.source.as_str())
    }
}

impl TextCodec for HfCodec {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self.tok.encode(text, add_special_tokens).map_err(|e| Error::Text(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tok.decode(ids, skip_special_tokens).map_err(|e| Error::Text(e.to_string()))
    }

    fn apply_chat_template(&self, messages: &[Value], opts: &ChatOpts) -> Result<String> {
        match &self.template {
            Some(t) => t.render(messages, opts),
            None => Err(Error::Text("this model has no chat template".into())),
        }
    }

    fn has_chat_template(&self) -> bool {
        self.template.is_some()
    }

    fn encode_chat(&self, messages: &[Value], opts: &ChatOpts) -> Result<Option<Vec<u32>>> {
        match &self.inkling {
            Some(c) => c.render(messages, opts).map(Some),
            None => Ok(None),
        }
    }

    fn tokenizer_json(&self) -> Option<String> {
        std::fs::read_to_string(&self.path).ok()
    }
}
