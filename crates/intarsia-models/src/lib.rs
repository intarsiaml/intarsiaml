//! Model-side plumbing: recipe loader, safetensors (mmap, sharded), tokenizer and
//! Hugging Face chat templates, and the one-call [`load::load_engine`].

pub mod chat_template;
pub mod load;
pub mod recipe;
pub mod safetensors;
pub mod tokenizer;

pub use chat_template::{ChatTemplate, ChatTemplateSource};
pub use load::{load_engine, load_engine_tp, read_model_spec, EngineOptions, Loaded};
pub use recipe::{parse_recipe, select_recipe};
pub use safetensors::SafeTensors;
pub use tokenizer::HfCodec;
