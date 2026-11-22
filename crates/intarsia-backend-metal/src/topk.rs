//! `Backend::topk_rows` on Metal (lane ink-mtp, kernels/metal/topk.metal): the k largest values of each logits row in the
//! sampler's order (value descending, ties by ascending index), NaN rows flagged. Two dispatches: per-chunk top-k into the
//! caller's scratch (`R * 32 * k` (value, index) pairs = `R * 32 * k * 8` bytes, the trait's contract), then the merge.
//! Logits f32 (the trait's contract) or bf16 (the DSpark draft's base logits). `supports_topk` stays false: no existing caller
//! (device sampling, the DFlash2 draft) switches to it; the DSpark draft calls it directly and falls back to the host on an error.

use crate::dense::{Grid, OpP};
use crate::MetalBackend;
use intarsia_core::{DType, Error, Result, TensorRef};

pub const STAGE1_F32: &str = "topk_f32_stage1";
pub const STAGE1_BF16: &str = "topk_bf16_stage1";
pub const STAGE2: &str = "topk_stage2";
pub const SHADERS: [&str; 3] = [STAGE1_F32, STAGE1_BF16, STAGE2];

/// Chunks per row (the scratch contract) and the kernels' limits (`TOPK_CHUNK_MAX`, `TOPK_CAND_MAX` in the shader).
const CHUNKS: usize = 32;
const CHUNK_MAX: usize = 7168;
const K_MAX: usize = 64;
const TG: usize = 256;

impl MetalBackend {
    pub(crate) fn topk_rows_impl(&self, logits: &TensorRef, k: usize, ids: &TensorRef, vals: &TensorRef, nan: &TensorRef, scratch: &TensorRef) -> Result<()> {
        let rows = logits.shape.first().copied().unwrap_or(0);
        let n: usize = logits.shape.iter().product();
        let v = if rows == 0 { 0 } else { n / rows };
        let chunk = v.div_ceil(CHUNKS);
        if rows == 0 || v == 0 || k == 0 || k > K_MAX || k > v || chunk > CHUNK_MAX {
            return Err(Error::Unsupported(format!("metal topk_rows: {rows} rows of {v}, k {k} (k <= {K_MAX}, V <= {})", CHUNKS * CHUNK_MAX)));
        }
        if ids.numel() < rows * k || vals.numel() < rows * k || nan.numel() < rows || scratch.nbytes() < rows * CHUNKS * k * 8 {
            return Err(Error::invalid(format!("metal topk_rows: outputs / scratch too small for {rows} x {k}")));
        }
        let stage1 = match logits.dtype {
            DType::F32 => STAGE1_F32,
            DType::Bf16 => STAGE1_BF16,
            d => return Err(Error::Unsupported(format!("metal topk_rows: {d} logits"))),
        };
        for t in [logits, ids, vals, nan, scratch] {
            t.check_bounds()?;
        }
        let x = self.bind(logits, logits.dtype, false, "logits")?;
        let s = self.bind(scratch, scratch.dtype, true, "scratch")?;
        if s.skip != 0 {
            return Err(Error::invalid("metal topk_rows: scratch not 4-byte aligned"));
        }
        let (bi, bv, bn) = (self.bind(ids, DType::I32, true, "ids")?, self.bind(vals, DType::F32, true, "vals")?, self.bind(nan, DType::I32, true, "nan")?);
        let u = |x: usize| u32::try_from(x).map_err(|_| Error::invalid("metal topk_rows: dimension over u32"));
        let p = OpP { dims: [u(v)?, u(k)?, u(chunk)?, u(CHUNKS)?, 0, 0, 0, 0], skips: [x.skip, 0, 0, 0, 0], ..Default::default() };
        self.prof_label("topk_stage1", rows);
        self.dispatch(stage1, &[&x, &s], &p, Grid::Groups { x: CHUNKS, y: rows, threads: TG })?;
        self.prof_label("topk_stage2", rows);
        self.dispatch(STAGE2, &[&s, &bi, &bv, &bn], &p, Grid::Groups { x: 1, y: rows, threads: TG })
    }
}
