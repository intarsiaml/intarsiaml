//! Review findings 8 and 9: a missing `lm_head` is a load error unless the config ties the embeddings, and the
//! quantisation scheme declared in `hf_quant_config.json` is checked against what the loader finds.

mod common;

use common::{dims, recipe, tiny_weights, V, H};
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::quant::scheme::QuantScheme;
use intarsia_core::weights::{MemWeights, WeightInfo, WeightSource};
use intarsia_core::{DType, HostTensor, Result, Telemetry};
use std::sync::Arc;

/// A `MemWeights` that also declares a scheme.
struct Declared {
    inner: MemWeights,
    scheme: QuantScheme,
}

impl WeightSource for Declared {
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }
    fn info(&self, name: &str) -> Option<WeightInfo> {
        self.inner.info(name)
    }
    fn bytes(&self, name: &str) -> Result<&[u8]> {
        self.inner.bytes(name)
    }
    fn quant_scheme(&self) -> Option<&QuantScheme> {
        Some(&self.scheme)
    }
}

fn load(src: &dyn WeightSource, tie: bool) -> intarsia_core::Result<Model> {
    let mut d = dims();
    d.tie_embeddings = tie;
    Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe(),
        &d,
        src,
        &LoadOptions { max_seq: 64, ..Default::default() },
        Arc::new(Telemetry::new()),
    )
}

fn head() -> HostTensor {
    HostTensor::from_f32_as(DType::Bf16, &[V, H], &vec![0.01; V * H])
}

#[test]
fn a_missing_lm_head_is_an_error_unless_the_config_ties_the_embeddings() {
    let mut t = tiny_weights(DType::Bf16, 1);
    // tied embeddings, no lm_head tensor: the embedding table is the head
    load(&t.weights, true).unwrap();
    // not tied: the head must exist
    let e = load(&t.weights, false).err().expect("untied config without an lm_head must not load").to_string();
    assert!(e.contains("lm_head") && e.contains("missing"), "{e}");
    // and with the tensor it loads
    t.weights.insert("lm_head.weight", head());
    load(&t.weights, false).unwrap();
}

fn with_scheme(json: &str) -> Declared {
    let mut t = tiny_weights(DType::Bf16, 2);
    t.weights.insert("lm_head.weight", head());
    Declared { inner: t.weights, scheme: QuantScheme::parse(&serde_json::from_str(json).unwrap()).unwrap() }
}

#[test]
fn a_declared_scheme_is_checked_against_the_stored_tensors() {
    // nothing declared for these layers: loads as stored
    load(&with_scheme(r#"{"quantization":{"quant_algo":"MIXED_PRECISION","quantized_layers":{}}}"#), false).unwrap();
    // a global FP8 declaration but bf16 tensors: refused, naming the layer
    let e = load(&with_scheme(r#"{"quantization":{"quant_algo":"FP8"}}"#), false).err().unwrap().to_string();
    assert!(e.contains("FP8") && e.contains("plain float"), "{e}"); // the head loads first, so it is the layer named
    // W4A4 NVFP4 declared for one layer: refused (we would run it weight-only)
    let e = load(
        &with_scheme(r#"{"quantization":{"quant_algo":"MIXED_PRECISION","quantized_layers":{"model.layers.1.self_attn.q_proj":{"quant_algo":"NVFP4"}}}}"#),
        false,
    )
    .err()
    .unwrap()
    .to_string();
    assert!(e.contains("4-bit activations") && e.contains("model.layers.1.self_attn.q_proj"), "{e}");
    // the head is checked too; excluded modules are not
    let e = load(&with_scheme(r#"{"quantization":{"quant_algo":"MIXED_PRECISION","quantized_layers":{"lm_head":{"quant_algo":"INT4_AWQ"}}}}"#), false)
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("lm_head") && e.contains("INT4_AWQ"), "{e}");
    load(&with_scheme(r#"{"quantization":{"quant_algo":"FP8","exclude_modules":["*"]}}"#), false).unwrap();
}
