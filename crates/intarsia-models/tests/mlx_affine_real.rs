//! An MLX affine checkpoint read through `SchemedSource` against the bf16 directory `tools/mlx_to_hf_bf16.py` (numpy, an
//! independent implementation) writes from it: every text tensor (the vision tower is left alone) must match bit for bit (names, shapes, dtypes, bytes).
//! Needs both directories, so it is ignored by default:
//!
//!   MLX_CKPT_DIR=<mlx dir> MLX_REF_DIR=<dequantised dir> cargo test --release -p intarsia-models \
//!       --test mlx_affine_real -- --ignored

use intarsia_core::weights::WeightSource;
use intarsia_models::load::SchemedSource;
use intarsia_models::safetensors::SafeTensors;
use std::path::PathBuf;

#[test]
#[ignore]
fn mlx_checkpoint_reads_bit_for_bit_like_the_numpy_dequantiser() {
    let (Some(mlx), Some(rf)) = (std::env::var_os("MLX_CKPT_DIR"), std::env::var_os("MLX_REF_DIR")) else {
        panic!("set MLX_CKPT_DIR and MLX_REF_DIR");
    };
    let (src, want) = (SchemedSource::open(&PathBuf::from(mlx)).unwrap(), SafeTensors::open_dir(&PathBuf::from(rf)).unwrap());
    let (mut n, mut bytes) = (0usize, 0usize);
    for name in want.names() {
        if name.starts_with("model.visual.") {
            continue;
        }
        let (a, b) = (src.info(&name), want.info(&name));
        assert!(a.is_some(), "{name} missing from the engine's view");
        let (a, b) = (a.unwrap(), b.unwrap());
        assert_eq!((a.dtype, &a.shape), (b.dtype, &b.shape), "{name}");
        assert!(src.bytes(&name).unwrap() == want.bytes(&name).unwrap(), "{name}: bytes differ");
        n += 1;
        bytes += want.bytes(&name).unwrap().len();
    }
    let extra: Vec<String> = src.names().into_iter().filter(|x| want.info(x).is_none() && !x.starts_with("vision_tower.")).collect();
    assert!(extra.is_empty(), "engine view has tensors the reference lacks: {:?}", &extra[..extra.len().min(5)]);
    eprintln!("{n} tensors, {:.2} GB identical", bytes as f64 / 1e9);
}

/// A checkpoint with a per-module bit override and stacked 3-D experts (LFM2.5-24B-A2B-MLX-4bit: 4-bit body, 6-bit embedding,
/// `switch_mlp.*` `[64, 1536, 256]` words): every module opens with the shape the bits imply, and the first rows of the embedding
/// and of an expert dequantise to finite values of a sane magnitude.
///
///   MLX_MIXED_DIR=<LFM2.5 mlx dir> cargo test --release -p intarsia-models --test mlx_affine_real -- --ignored
#[test]
#[ignore]
fn mixed_bits_and_stacked_experts_open_with_the_implied_shapes() {
    let dir = std::env::var_os("MLX_MIXED_DIR").expect("set MLX_MIXED_DIR");
    let src = SchemedSource::open(&PathBuf::from(dir)).unwrap();
    let i = src.info("model.embed_tokens.weight").unwrap();
    assert_eq!((i.dtype, i.shape), (intarsia_core::DType::Bf16, vec![128000, 2048]));
    let i = src.info("model.layers.2.feed_forward.switch_mlp.gate_proj.weight").unwrap();
    assert_eq!(i.shape, vec![64, 1536, 2048]);
    assert!(src.info("model.layers.2.feed_forward.switch_mlp.gate_proj.scales").is_none());
    for (name, rows) in [("model.embed_tokens.weight", 4usize), ("model.layers.2.feed_forward.switch_mlp.gate_proj.weight", 4)] {
        let b = src.bytes(name).unwrap();
        let v = intarsia_core::weights::to_f32_vec(intarsia_core::DType::Bf16, &b[..rows * 2048 * 2]).unwrap();
        assert!(v.iter().all(|x| x.is_finite() && x.abs() < 50.0), "{name}");
        assert!(v.iter().any(|x| *x != 0.0), "{name}");
    }
}
