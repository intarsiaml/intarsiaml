//! `logits_rows_scalar_m<M>` (kernels/metal/logits_rows.metal, lane dm): the M-row twin of `gemv_f32o_scalar`, the 1-row `Logits`
//! kernel for operands that are not 16-byte aligned. The Qwen3.8-27B lm_head sits at byte 8 mod 16 of its file, so every head call
//! takes the scalar kernel, and an M-row verify head (`logits_bf16_gemv_rows`, `multirow.rs`) was M one-row launches (5.8 ms each).
//! Row r of the M-row launch is bitwise the 1-row launch on activation row r (tests/logits_rows.rs).

use crate::dense::Grid;
use crate::MetalBackend;
use intarsia_core::backend::TensorRef;
use intarsia_core::{DType, Error, Result};

/// `logits_rows_scalar_m<M>` (kernels/metal/dflash.metal), M = 2..=16: the M-row twin of `gemv_f32o_scalar`.
pub const LRS_SHADERS: [&str; 15] = [
    "logits_rows_scalar_m2",
    "logits_rows_scalar_m3",
    "logits_rows_scalar_m4",
    "logits_rows_scalar_m5",
    "logits_rows_scalar_m6",
    "logits_rows_scalar_m7",
    "logits_rows_scalar_m8",
    "logits_rows_scalar_m9",
    "logits_rows_scalar_m10",
    "logits_rows_scalar_m11",
    "logits_rows_scalar_m12",
    "logits_rows_scalar_m13",
    "logits_rows_scalar_m14",
    "logits_rows_scalar_m15",
    "logits_rows_scalar_m16",
];
/// Output rows per threadgroup of `logits_rows_scalar_m*` (`LRS_SG * LRS_P`) and its threadgroup width.
const LRS_ROWS_PER_TG: usize = 32;
const LRS_THREADS: usize = 256;

pub fn shader_names() -> Vec<&'static str> {
    LRS_SHADERS.to_vec()
}

/// `INTARSIA_METAL_LOGITS_ROWS`: unset or `1` (default) = a multi-row `Logits` verify call whose operands the 16-byte row kernel cannot
/// take runs `logits_rows_scalar_m<M>` (one weight pass, bitwise the 1-row `gemv_f32o_scalar` per row); `0` = M one-row launches as
/// before. Read at every call (tests flip it).
pub fn logits_rows_on() -> bool {
    std::env::var("INTARSIA_METAL_LOGITS_ROWS").map_or(true, |v| v.trim() != "0")
}

fn dim(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    /// `Logits` of `m` = 2..=16 rows on `logits_rows_scalar_m<m>`: `x [m, k]`, `w [n, k]` bf16 at any 2-byte alignment, `y [m, n]` f32.
    pub(crate) fn launch_logits_rows_scalar(&self, x: &TensorRef, w: &TensorRef, y: &TensorRef) -> Result<()> {
        let (m, k, n) = (x.shape[0], x.shape[1], w.shape[0]);
        let Some(&name) = LRS_SHADERS.get(m.wrapping_sub(2)) else {
            return Err(Error::invalid(format!("metal logits_rows_scalar: {m} rows (2..=16)")));
        };
        let (xb, wb, yb) = (self.bind(x, DType::Bf16, false, "x")?, self.bind(w, DType::Bf16, false, "w")?, self.bind(y, DType::F32, true, "y")?);
        dim(m * k, "M*K")?;
        dim(m * n, "M*N")?;
        let params = crate::GemvParams { k: dim(k, "K")?, n: dim(n, "N")?, w_skip: wb.skip, x_skip: xb.skip, y_skip: yb.skip, pad: [0; 3] };
        self.dispatch(name, &[&wb, &xb, &yb], &params, Grid::Groups { x: n.div_ceil(LRS_ROWS_PER_TG), y: 1, threads: LRS_THREADS })
    }

}
