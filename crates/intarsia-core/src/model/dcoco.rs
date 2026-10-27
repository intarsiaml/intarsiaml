//! DCOCO corrector on top of the DFlash drafter (`--spec dflash --dcoco`), lane dc, written from arXiv 2610.12570 section
//! 4.2-4.3 and the checkpoint's tensor list (docs/results/2026-10-25-dc-dflash-dcoco.md section 3). The upstream repository
//! has no licence and was not read.
//!
//! The draft block gives, per slot `i` (15 of them), a hidden state `h_i` (the draft's normed output, the tensor `lm_head`
//! reads) and logits. The corrector keeps the top-K candidates of each slot and scores each adjacent pair:
//!
//! ```text
//!   b_ik = LN(W_cand E[c_ik])            g_i = LN(W_ctx h_i)            z_ik = LN_gb(W_fuse [b_ik ; g_i])
//!   s_p  = W_pair [h_p ; h_{p+1}]
//!   S_p(a,b) = l_{p+1}(b) + MLP([s_p ; z_pa ; z_{p+1,b} ; l_p(a) ; l_{p+1}(b)])       MLP = Linear(386,256) GELU Linear(256,1)
//!   k_0 = argmax_k l_0(k) ;  k_{p+1} = argmax_b S_p(k_p, b)                           (prefix-first greedy)
//! ```
//!
//! Only the row `a = k_p` of each `S_p` is ever read, so the walk evaluates K transitions per step (14 x 16) instead of
//! the K^2 grid. The first MLP layer is linear in its concatenated input, so it is split by input block: the four big
//! projections (`W_cand E[c]`, `W_ctx h`, both halves of `W_pair`) run on the device; the rest is a few million
//! multiply-adds on the host. The corrector only changes which tokens are proposed: every emitted token is still a verify
//! row's argmax, so greedy output is unchanged by it.

use super::*;

/// LayerNorm epsilon (PyTorch default; the reference run cannot tell it from 1e-6, docs section 10.3).
const LN_EPS: f32 = 1e-5;

/// Host-side corrector weights (f32) and the shape constants.
pub(super) struct Corrector {
    pub(super) k: usize,
    pub(super) r: usize,
    pub(super) mlp: usize,
    fuse_w: Vec<f32>,  // [r, 2r]
    fuse_g: Vec<f32>,  // [r]
    fuse_b: Vec<f32>,  // [r]
    w0: Vec<f32>,      // [mlp, 3r + 2]
    b0: Vec<f32>,      // [mlp]
    w2: Vec<f32>,      // [mlp]
    b2: f32,
    /// Device projections: `W_cand`, `W_ctx`, and `W_pair` split by input halves (`[r, H]` each).
    pub(super) cand: QTensor,
    pub(super) ctx: QTensor,
    pub(super) pair_a: QTensor,
    pub(super) pair_b: QTensor,
    /// Device top-k buffers (`Backend::topk_rows`), when the backend has it.
    pub(super) topk: Option<(Buffer, Buffer, Buffer, Buffer)>,
}

fn host(src: &dyn WeightSource, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let info = src.info(name).ok_or_else(|| Error::model(format!("corrector tensor '{name}' missing from the draft checkpoint")))?;
    if info.shape != shape {
        return Err(Error::model(format!("corrector tensor '{name}' is {:?}, expected {shape:?}", info.shape)));
    }
    crate::weights::to_f32_vec(info.dtype, src.bytes(name)?)
}

/// `x [n, k]` bf16 columns `[c0, c1)` as an `[n, c1 - c0]` device linear.
fn cols(b: &dyn Backend, raw: &[u8], n: usize, k: usize, c0: usize, c1: usize) -> Result<QTensor> {
    let w = c1 - c0;
    let mut part = Vec::with_capacity(n * w * 2);
    for r in 0..n {
        part.extend_from_slice(&raw[(r * k + c0) * 2..(r * k + c1) * 2]);
    }
    let buf = crate::backend::upload_new(b, &part)?;
    Ok(QTensor { format: crate::quant::BF16, n, k: w, parts: vec![TensorRef::new(buf, DType::Bf16, &[n, w])], act_fp8: false, act_quant: false })
}

impl Corrector {
    pub(super) fn load(src: &dyn WeightSource, b: &dyn Backend, h: usize, k: usize, r: usize, mlp: usize, rows: usize) -> Result<Corrector> {
        let dev = |name: &str, n: usize, kk: usize, c0: usize, c1: usize| -> Result<QTensor> {
            let info = src.info(name).ok_or_else(|| Error::model(format!("corrector tensor '{name}' missing from the draft checkpoint")))?;
            if info.dtype != DType::Bf16 || info.shape != [n, kk] {
                return Err(Error::model(format!("corrector '{name}' is {} {:?}, expected bf16 [{n}, {kk}]", info.dtype, info.shape)));
            }
            cols(b, src.bytes(name)?, n, kk, c0, c1)
        };
        let blk = k * 15; // candidate rows of one pass (block <= 15 slots)
        if rows < blk {
            return Err(Error::Unsupported(format!("--dcoco needs prefill chunks of at least {blk} rows (workspace has {rows})")));
        }
        let topk = if b.supports_topk() {
            let n = 15;
            Some((
                b.alloc(n * k * 4, MemKind::Device)?,
                b.alloc(n * k * 4, MemKind::Device)?,
                b.alloc(n * 4, MemKind::Device)?,
                b.alloc(n * 32 * k * 8, MemKind::Device)?,
            ))
        } else {
            None
        };
        Ok(Corrector {
            k,
            r,
            mlp,
            fuse_w: host(src, "dcoco.cand_ctx_proj.weight", &[r, 2 * r])?,
            fuse_g: host(src, "dcoco.ctx_out_norm.weight", &[r])?,
            fuse_b: host(src, "dcoco.ctx_out_norm.bias", &[r])?,
            w0: host(src, "dcoco.trans_mlp.0.weight", &[mlp, 3 * r + 2])?,
            b0: host(src, "dcoco.trans_mlp.0.bias", &[mlp])?,
            w2: host(src, "dcoco.trans_mlp.2.weight", &[1, mlp])?,
            b2: host(src, "dcoco.trans_mlp.2.bias", &[1])?[0],
            cand: dev("dcoco.cand_proj.weight", r, h, 0, h)?,
            ctx: dev("dcoco.ctx_h_proj.weight", r, h, 0, h)?,
            pair_a: dev("dcoco.pair_proj.weight", r, 2 * h, 0, h)?,
            pair_b: dev("dcoco.pair_proj.weight", r, 2 * h, h, 2 * h)?,
            topk,
        })
    }

    /// Prefix-first path. `raw_b [S*K*r]` = `W_cand E[c_ik]`, `raw_g [S*r]` = `W_ctx h_i`, `pa`/`pb [S*r]` = the two halves of
    /// `W_pair` applied to every `h_i` (so `s_p = pa_p + pb_{p+1}`), `logits [S*K]` the candidates' draft logits (best first).
    /// Returns the chosen candidate index per slot, and (when `scores` is set) the full `S_p(a,b)` grid `[(S-1), K, K]`.
    pub(super) fn walk(
        &self,
        slots: usize,
        raw_b: &[f32],
        raw_g: &[f32],
        pa: &[f32],
        pb: &[f32],
        logits: &[f32],
        scores: Option<&mut Vec<f32>>,
    ) -> Vec<usize> {
        let (k, r, m) = (self.k, self.r, self.mlp);
        let w = 3 * r + 2;
        // z_ik and the two per-candidate first-layer projections: left (a) side [m] and right (b) side [m]
        let mut za = vec![0f32; slots * k * m];
        let mut zb = vec![0f32; slots * k * m];
        let mut cat = vec![0f32; 2 * r];
        let mut z = vec![0f32; r];
        for i in 0..slots {
            let mut g = raw_g[i * r..(i + 1) * r].to_vec();
            layer_norm(&mut g, None);
            for c in 0..k {
                cat[..r].copy_from_slice(&raw_b[(i * k + c) * r..(i * k + c + 1) * r]);
                layer_norm(&mut cat[..r], None);
                cat[r..].copy_from_slice(&g);
                for o in 0..r {
                    z[o] = dot(&self.fuse_w[o * 2 * r..(o + 1) * 2 * r], &cat);
                }
                layer_norm(&mut z, Some((&self.fuse_g, &self.fuse_b)));
                for o in 0..m {
                    za[(i * k + c) * m + o] = dot(&self.w0[o * w + r..o * w + 2 * r], &z);
                    zb[(i * k + c) * m + o] = dot(&self.w0[o * w + 2 * r..o * w + 3 * r], &z);
                }
            }
        }
        let mut grid = scores;
        if let Some(g) = grid.as_deref_mut() {
            g.clear();
            g.resize((slots - 1) * k * k, 0.0);
        }
        let argmax = |v: &[f32]| v.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0;
        let mut path = vec![argmax(&logits[..k])];
        let mut s = vec![0f32; r];
        let mut pre = vec![0f32; m];
        for p in 0..slots - 1 {
            for o in 0..r {
                s[o] = pa[p * r + o] + pb[(p + 1) * r + o];
            }
            // the s-dependent part of layer 1 is shared by all transitions of this step
            let mut base = vec![0f32; m];
            for o in 0..m {
                base[o] = self.b0[o] + dot(&self.w0[o * w..o * w + r], &s);
            }
            let a = path[p];
            // the row `a` is the walk's; the full grid (tests) covers every `a`
            let rows: Vec<usize> = if grid.is_some() { (0..k).collect() } else { vec![a] };
            let mut best = (0usize, f32::NEG_INFINITY);
            for &aa in &rows {
                let la = logits[p * k + aa];
                for bb in 0..k {
                    let lb = logits[(p + 1) * k + bb];
                    let (zl, zr) = (&za[(p * k + aa) * m..(p * k + aa + 1) * m], &zb[((p + 1) * k + bb) * m..((p + 1) * k + bb + 1) * m]);
                    let mut out = self.b2;
                    for o in 0..m {
                        pre[o] = base[o] + zl[o] + zr[o] + self.w0[o * w + 3 * r] * la + self.w0[o * w + 3 * r + 1] * lb;
                        out += self.w2[o] * gelu(pre[o]);
                    }
                    let sc = lb + if zero_mlp() { 0.0 } else { out };
                    if let Some(g) = grid.as_deref_mut() {
                        g[(p * k + aa) * k + bb] = sc;
                    }
                    if aa == a && sc > best.1 {
                        best = (bb, sc);
                    }
                }
            }
            path.push(best.0);
        }
        path
    }
}

/// `INTARSIA_DCOCO_ZERO=1` (diagnostic, gate G6): the MLP's contribution is dropped, the walk then is plain argmax drafting.
fn zero_mlp() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_DCOCO_ZERO").is_ok_and(|v| !v.is_empty() && v != "0"))
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// LayerNorm over the whole slice (biased variance, eps 1e-5), optionally affine.
fn layer_norm(x: &mut [f32], affine: Option<(&[f32], &[f32])>) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + LN_EPS).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v = (*v - mean) * inv;
        if let Some((g, b)) = affine {
            *v = *v * g[i] + b[i];
        }
    }
}

/// Exact (erf) GELU.
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// erf in f64: Taylor series below 3.5 (error ~1e-11), the asymptotic erfc tail above (absolute error < 1e-8 there).
fn erf(x: f32) -> f32 {
    let z = (x as f64).abs();
    let e = if z < 3.5 {
        let (mut term, mut sum) = (z, z);
        for n in 1..60 {
            term *= -z * z / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        sum * std::f64::consts::FRAC_2_SQRT_PI
    } else {
        let z2 = z * z;
        let erfc = (-z2).exp() / (z * std::f64::consts::PI.sqrt()) * (1.0 - 0.5 / z2 + 0.75 / (z2 * z2));
        1.0 - erfc
    };
    (if x < 0.0 { -e } else { e }) as f32
}

impl Model {
    /// The corrected block: from the draft's normed block output `n` (`[t, H]`) to `t` drafts (`DflashHead::dcoco`).
    /// logits -> top-K per slot -> `W_cand E[c]`, `W_ctx h`, `W_pair` halves on the device -> host walk.
    pub(super) fn dcoco_draft(&self, c: &Corrector, n: &TensorRef, t: usize, profile: bool) -> Result<Vec<u32>> {
        let (v, h, k, r, act, ws) = (self.dims.vocab, self.dims.hidden, c.k, c.r, self.act, &self.ws);
        let logits = self.view(&ws.logits, DType::F32, &[t, v]);
        self.proj(&n.rows(0, t), self.head()?, &logits, profile)?;
        if self.recorder.lock().unwrap().is_some() {
            // plan dry run: record the device launches of the candidate projections too, nothing to read back
            let cand = self.view(&ws.h[1], act, &[t * k, h]);
            self.exec(Op::Embedding, vec![self.view(&ws.ids, DType::I32, &[t * k]), self.embed.clone()], vec![cand.clone()], OpAttrs::default(), profile)?;
            let out = self.view(&ws.blk, act, &[t * k + 3 * t, r]);
            self.proj(&cand, &c.cand, &out.rows(0, t * k), profile)?;
            self.proj(&n.rows(0, t), &c.ctx, &out.rows(t * k, t), profile)?;
            self.proj(&n.rows(0, t), &c.pair_a, &out.rows(t * k + t, t), profile)?;
            self.proj(&n.rows(0, t), &c.pair_b, &out.rows(t * k + 2 * t, t), profile)?;
            return Ok(Vec::new());
        }
        // top-k of every slot, best first, ties by ascending id (the sampler's order)
        let (ids, vals): (Vec<u32>, Vec<f32>) = if let Some((bi, bv, bn, bs)) = &c.topk {
            let (ti, tv, tn, ts) = (
                self.view(bi, DType::I32, &[t, k]),
                self.view(bv, DType::F32, &[t, k]),
                self.view(bn, DType::I32, &[t]),
                self.view(bs, DType::U8, &[t * 32 * k * 8]),
            );
            self.backend.topk_rows(&logits, k, &ti, &tv, &tn, &ts, self.stream)?;
            let (mut ib, mut vb) = (vec![0u8; t * k * 4], vec![0u8; t * k * 4]);
            self.backend.download(bi, 0, &mut ib, self.stream)?;
            self.backend.download(bv, 0, &mut vb, self.stream)?;
            (
                ib.chunks_exact(4).map(|x| i32::from_le_bytes([x[0], x[1], x[2], x[3]]) as u32).collect(),
                vb.chunks_exact(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect(),
            )
        } else {
            let mut lb = vec![0u8; t * v * 4];
            self.backend.download(&ws.logits, 0, &mut lb, self.stream)?;
            let mut ids = Vec::with_capacity(t * k);
            let mut vals = Vec::with_capacity(t * k);
            for row in lb.chunks_exact(v * 4) {
                let x: Vec<f32> = row.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let mut top: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
                for (i, &val) in x.iter().enumerate() {
                    // value descending, ties by ascending index; a NaN never enters
                    if val.is_nan() || (top.len() == k && val <= top[k - 1].0) {
                        continue;
                    }
                    let at = top.partition_point(|&(tv, _)| tv >= val);
                    top.insert(at, (val, i as u32));
                    top.truncate(k);
                }
                ids.extend(top.iter().map(|&(_, i)| i));
                vals.extend(top.iter().map(|&(v, _)| v));
            }
            (ids, vals)
        };
        // candidate embeddings -> W_cand, and the slot projections
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| (i as i32).to_le_bytes()).collect();
        self.backend.upload(&ws.ids, 0, &bytes, self.stream)?;
        let cand = self.view(&ws.h[1], act, &[t * k, h]);
        self.exec(Op::Embedding, vec![self.view(&ws.ids, DType::I32, &[t * k]), self.embed.clone()], vec![cand.clone()], OpAttrs::default(), profile)?;
        let rows = t * k + 3 * t;
        let out = self.view(&ws.blk, act, &[rows, r]);
        self.proj(&cand, &c.cand, &out.rows(0, t * k), profile)?;
        self.proj(&n.rows(0, t), &c.ctx, &out.rows(t * k, t), profile)?;
        self.proj(&n.rows(0, t), &c.pair_a, &out.rows(t * k + t, t), profile)?;
        self.proj(&n.rows(0, t), &c.pair_b, &out.rows(t * k + 2 * t, t), profile)?;
        let mut ob = vec![0u8; rows * r * act.size_bytes()];
        self.backend.download(&ws.blk, 0, &mut ob, self.stream)?;
        let mut f = vec![0f32; rows * r];
        crate::dtype::decode_to_f32(act, &ob, &mut f);
        let (raw_b, rest) = f.split_at(t * k * r);
        let (raw_g, rest) = rest.split_at(t * r);
        let (pa, pb) = rest.split_at(t * r);
        let path = c.walk(t, raw_b, raw_g, pa, pb, &vals, None);
        Ok(path.iter().enumerate().map(|(i, &j)| ids[i * k + j]).collect())
    }
}

impl Model {
    /// Tests: the corrector's path and score grid from host inputs (`Corrector::walk`).
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn dcoco_walk_host(
        &self,
        slots: usize,
        raw_b: &[f32],
        raw_g: &[f32],
        pa: &[f32],
        pb: &[f32],
        logits: &[f32],
        scores: Option<&mut Vec<f32>>,
    ) -> Result<Vec<usize>> {
        let c = self.dflash.as_ref().and_then(|h| h.dcoco.as_ref()).ok_or_else(|| Error::invalid("--dcoco not loaded"))?;
        Ok(c.walk(slots, raw_b, raw_g, pa, pb, logits, scores))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        for (x, want) in [(0.0f32, 0.0f32), (0.5, 0.5204999), (1.0, 0.8427008), (2.0, 0.9953223), (-1.0, -0.8427008), (4.0, 1.0)] {
            assert!((erf(x) - want).abs() < 2e-7, "erf({x}) = {} want {want}", erf(x));
        }
    }

    #[test]
    fn layer_norm_is_zero_mean_unit_variance() {
        let mut x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.37).sin() * 3.0 + 1.0).collect();
        layer_norm(&mut x, None);
        let mean: f32 = x.iter().sum::<f32>() / 128.0;
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / 128.0;
        assert!(mean.abs() < 1e-5 && (var - 1.0).abs() < 1e-3);
    }
}
