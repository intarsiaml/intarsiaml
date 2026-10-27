//! An MLX affine checkpoint read through `SchemedSource` against the bf16 directory `tools/mlx_to_hf_bf16.py` (numpy, an
//! independent implementation) writes from it: every text tensor (the vision tower is left alone) must match bit for bit (names, shapes, dtypes, bytes).
//! Needs both directories, so it is ignored by default:
//!
//!   INTARSIA_MLX_DIR=<mlx dir> INTARSIA_MLX_REF_DIR=<dequantised dir> cargo test --release -p intarsia-models \
//!       --test mlx_affine_real -- --ignored

use intarsia_core::weights::WeightSource;
use intarsia_models::load::SchemedSource;
use intarsia_models::safetensors::SafeTensors;
use std::path::PathBuf;

#[test]
#[ignore]
fn mlx_checkpoint_reads_bit_for_bit_like_the_numpy_dequantiser() {
    let (Some(mlx), Some(rf)) = (std::env::var_os("INTARSIA_MLX_DIR"), std::env::var_os("INTARSIA_MLX_REF_DIR")) else {
        panic!("set INTARSIA_MLX_DIR and INTARSIA_MLX_REF_DIR");
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
