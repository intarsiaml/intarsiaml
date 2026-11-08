# DSpine drafter port: design (2026-11-07)

Goal: run `tencent/Qwen3.8-27B-DSpine-b16` (Apache-2.0) as a `--spec dflash` draft for Qwen3.8-27B, next to DFlash v1
(`model/dflash.rs`) and DFlash2 (`model/dflash2.rs`). Branch `dspine-port` off `831a895fc`. Semantics come from Tencent/DSpine
@6450af7: the MLX package `dspine_mlx/` is the executable reference, and `sglang_dspine/` covers the sliding window and dynamic
verification. We read it and write our own code; nothing is copied. Inputs: `.worktrees/dspine/SPEC-reference.md`
(reference map, cited below as "ref §n"), `.worktrees/dspine/SPEC-engine.md` (engine map, cited as "eng §n") and
`.worktrees/dspine/hdr.json`. Line numbers refer to `831a895fc`.

## 0. Decisions in one table

| question | decision |
|---|---|
| module | New `model/dspine.rs` (bridge weights, tables, block forward, tail). The backbone is a `Draft2` with no conv: `Layer2.{aconv,mconv}` become `Option<Conv2>` and `Draft2` gets `spine: Option<Box<Spine>>`. Every `head.v2` route then works unchanged: backend gate, slot specs, ingest, `cut`, dry-run entry, verify. `dflash.rs` gets only config fields, a dispatch in `dflash_draft_rank` and dry-run loops. A separate `head.spine` beside `head.v2` was rejected: about 8 more call sites to duplicate. |
| detection | `architectures` contains `DFlashDraftModel` (the same string as v1) AND `dflash_config.selected_prefix_code_bridge` is present AND the tensor `selected_prefix_bridge.codebook` exists. If only one of the two markers is present, refuse with an error that names the missing one. v1 refuses any checkpoint that has a `selected_prefix_bridge.*` tensor or a `selected_prefix_*` config key. |
| block vs `--spec-k` | The block runs `t = min(block, k+1)` rows. The block is causal, so the first rows of a shorter block equal the first rows of the 16-row block in exact arithmetic; the MLX reference also serves T=2..16. The dry-run records every t in 2..=block. Metal gates use k ≤ 7, because an MXFP4 verify of more than 8 rows is unverified on Metal (eng §1.5). HIP gates use up to k=15. |
| causal block | One `AttnDecode` per block row r at `pos0 = p0 + r` (sees context plus block rows 0..=r). This is NOT DFlash2's `pos0 = last` with `window + t-1-r`. |
| window | `OpAttrs::window = W` for row r iff `p0 + r + 1 > W`, else 0. `W` is the config's `sliding_window`, not hard-coded 8192. `win_split` as DFlash2. Dense draft KV (as DFlash2); a ring buffer comes later. |
| bridge | 4 new row-wise ops (`BridgeNorm`, `BridgeCode`, `BridgeGate`, `BridgeWrite`) with the reference's rounding points. The next-layer norm is fused into `BridgeWrite` (two-step HF rounding). The 3 bridge projections are plain `Matmul`. |
| row-0 sender | `packed[0, R:3R]` is replaced by `TSV[anchor]` from the host table, uploaded once per pass as an input of `BridgeGate`. Code row 0 and packed row 0 are then don't-care, and the codebook is never resident on the device. This saves 485 MiB, and we need no ×32 Embedding kernel. Cost: row 1's sender can differ from the reference by a matmul accumulation-order ulp (same operands, different M). Chosen deliberately. |
| tail | Greedy serial walk on the host over host tables TG `[V,R]` and TSV `[V,2R]`, built at load on the device in chunks. The `base` dots use host head rows that hold exactly the live head's values. |
| lm_head | Pass 1 runs on the live target head (fp8 e4m3 per-tensor by recipe on Metal/HIP, or bf16). TG and `base` use the same values dequantised (`bf16(e4m3 * scale)`, or the bf16 bits), from a resident host copy: +1.27 GB (fp8) or +2.54 GB (bf16). An mxfp4 head (`INTARSIA_LMHEAD_QUANT=mxfp4`) is refused in the MVP. |
| candidates | Host path on every backend in the MVP. `Op::Logits` f32 rows are downloaded, each logit rounded to bf16, top-K taken with the lower id winning ties, ids sorted ascending. This reproduces the reference's bf16-logit tie rule. HIP device top-k (f32 keys) stays off for DSpine (`d2.topk = None`) until it gets a bf16-key mode. |
| MVP | Greedy (T=0), fixed verify length (`--spec-k`), tp=1, Metal + HIP + CPU. Refuse CUDA, tp>1, `--dcoco`, mxfp4 head, and act ≠ bf16 on GPU backends. |

## 1. Step 0: prerequisites (before code)

1. Fetch `tencent/Qwen3.8-27B-DSpine-b16` (about 4.05 GB, needs the user's OK) to `/Users/josef/models/Qwen3.8-27B-DSpine-b16` on endor and
   `/data/models/intarsia/Qwen3.8-27B-DSpine-b16` on tatooine.
2. Read `config.json` and record `rms_norm_eps`, `selected_prefix_code_bridge.eps`, `rope_theta`/`rope_parameters`, `layer_types`,
   `sliding_window`, `use_sliding_window`, `is_causal`, `block_size` and `head_topk` (`code_bridge_second_pass` and/or
   `selected_prefix_last_write_refine`). The parser reads every one of these and has no defaults for eps.
3. `sha256sum` of the safetensors data matches the size in `hdr.json` (4,051,824,160 B of data).

## 2. Loader (`crates/intarsia-models/src/load.rs` `open_dflash` :330-403, `DflashConfig` dflash.rs:26-60)

New `DflashConfig` field `spine: Option<SpineConfig>`:
`{code: C, rank: R, eps_b, head_topk: K, n_layers, window}`, with `rms_eps`, `rope_theta`, `sliding_window` and `block` reused.
`v2 = true` is set for a spine as well (the backbone path).

The spine branch REPLACES the existing `if config.v2` validation block (load.rs:383-400), not runs after it. That block demands
`is_causal == false` and `window > 0`, so it would refuse every DSpine config. Today `rms_eps` defaults to 1e-6 (load.rs:367) and
`rope_theta` to 10000 (:368-372); for a spine both, and the bridge eps, are required keys.

Validation, each failure a `Error::model` naming the key:

- **Geometry:** `hidden == target hidden`, `vocab == target vocab`, `2 <= block <= 16`, `mask_token_id < vocab`, every tap `< target n_layers`.
- **Bridge config:** `selected_prefix_terminal == false`, `selected_prefix_decode_mode == "serial"`, `selected_prefix_feature_writes == true`,
  `selected_prefix_feature_write_mode == "independent"`, `selected_prefix_feature_layers == [1..=L]`.
- **Refinement width:** `code_bridge_second_pass.mode == "serial_table"` with `first_exact == L-1`. K comes from `code_bridge_second_pass.head_topk`
  and/or `selected_prefix_last_write_refine.head_topk` (`kind == "exact_sender"`, `version == 1`). If both are present they must agree,
  and K must be a power of two in [2, 64].
- **Absent features:** `conv_kernel_size` absent or 0, and no `selector_*`, `projector`, `attention_sink_bias` or `attention_value_scale`.
- **Attention config:** `draft_block_causal == true` and `is_causal == true`. Either every layer is `sliding_attention` with
  `sliding_window = W > 0`, or every layer is `full_attention` with no window (W = 0, the MLX-compatible form). Default RoPE, no scaling,
  `partial_rotary_factor` absent or 1, `tie_word_embeddings == false`.
- **Tensors:** the shapes of ref §0. The 4 I64 version scalars equal 1 and are not loaded. `selected_prefix_bridge.out.weight` is all zero
  (read, checked, dropped). `codebook` is `[V, C]`. `early_read_weights` is `[L-1, C, H]`, `read_weight` `[C, H]`,
  `feature_write_weights` `[L, H, R]`, `gate.weight`/`predecessor_gate.weight`/`prev.weight` `[R, C]` and `gate.bias` `[R]`.
- **Backend gate (load.rs:589-604):** spine runs on cpu|hip|metal, as v2 does. CUDA refuses.
- **Model gate (model.rs:1468-1486):** refuse spine when `tp.world > 1` (it never splits). `load_head2` refuses `--dcoco` (dflash.rs:280).
  Refuse spine when `act != Bf16` on Metal/HIP; the CPU backend accepts f32 act for tests but the parity tests use bf16.

## 3. Weights and resident memory (27B draft)

| what | where | size |
|---|---|---|
| fc `[H, 5H]`, 5 column parts (`load_fc`), hidden_norm, norm | device | 250 MiB |
| 5 Qwen3 layers (q, k, v, o, q/k_norm, ln1, ln2, gate, up, down), bf16 | device | 3,050 MiB |
| `read[l]` `[C,H]` ×5 (early_read 0..3, read_weight for l=4), `write[l]` `[H,R]` ×5, `packed_w` `[3R,C]` = concat(gate, pred, prev), `gate.bias` `[R]` | device | 78 MiB |
| `write4_t` `[R,H]` (transpose of `write[4]`, the TG build weight), `pv_w` `[2R,C]` = concat(pred, prev), uploaded as its own tensor (a QTensor row slice is not assumed; it is 2 MiB) | device | 7 MiB |
| TG `[V,R]` bf16, TSV `[V,2R]` bf16 | host | 727.5 MiB |
| head rows (fp8 bytes + scale / bf16 bits) | host | 1,212.5 MiB (fp8) / 2,425 MiB (bf16) |
| codebook | never resident: read from the draft `WeightSource` during the TSV build only | 0 |
| context K/V | device, per sequence, dense `kv_cap_tokens` | 20 KiB/token: 640 MiB at 32k |
| scratch: `Bufs2` minus conv, plus bridge buffers (per row: 3R packed, R msg, C code, H raw/delta, 4B rms), stash, table-build staging `[chunk,H]`+`[chunk,2R]` (freed after build) | device | < 30 MiB + staging 22 MiB at chunk 2048 |

Device total about 3.30 GiB plus K/V. Host total about 1.9 GiB (fp8 head). On Metal the host and device figures share one pool.

## 4. Per-pass data flow

### 4.1 Ingest
This path is unchanged. Taps are the residual after target layers `[1,16,31,46,61]` (model.rs:3616-3625). Context rows are
`hidden_norm(Σ fc_i(feat_i))` (`dflash_ingest`, dflash.rs:470-499). Per layer, K and V are built with `dflash2_ingest_kv`
(dflash2.rs:424-438): `k = rope(k_norm(wk ctx), pos = kv.len + i)`, `v = wv ctx`. Scheduling, lazy store, verify ingest and
`cut` are as in DFlash2.

Known deviation: `load_fc` splits `fc` into 5 `[H,H]` parts and sums the partial products. The reference does one GEMM over
25600 columns. The difference is only extra roundings, and parity tolerances cover it.

### 4.2 Block forward: `Model::dspine_block(d2, sp, kv, t)` in `dspine.rs`
`p0 = kv.len`, ids `[next, MASK × (t-1)]` (dflash.rs:611-620).
Upload once per pass (skipped under the recorder, like the ids): `sender0 = TSV[next]` (2R bf16) into `sp.bufs.s0`.
```
x   = Embedding(ids, target embed)                        [t,H]
xn  = RmsNorm(x, ln1[0])                                  (existing op; single rounding)
res = x
for l in 0..L:
  q,k,v = proj(xn, wq/wk/wv); q,k = QkNorm+Rope at pos0=p0 (d2_qk); KvAppend k,v at p0 (d2_append; scratch beyond kv.len)
  for r in 0..t: AttnDecode(q[r], kc, vc) with pos0 = p0+r, window = (p0+r+1 > W) ? W : 0, win_split
  h    = Add(res, proj(attn, wo))
  a    = SiluGlu(proj(RmsNorm(h, ln2[l]), gate), proj(.., up));  raw = Add(h, proj(a, down))   -- l==4: raw into stash.raw
  nrm, rms = BridgeNorm(raw; eps_b)                                                            -- l==4: rms into stash.rms
  p    = Matmul(nrm, read[l])                [t,C] bf16
  code = BridgeCode(p; eps_b, scale=sqrt(C))  [t,C] bf16, row 0 don't-care
  pk   = Matmul(code, packed_w)              [t,3R] bf16                                      -- l==4: into stash.packed
  msg  = BridgeGate(pk, gate.bias, sender0)  [t,R] bf16, msg[0]=0
  dlt  = Matmul(msg, write[l])               [t,H] bf16
  res, xn = BridgeWrite(raw, rms, dlt, w_next = ln1[l+1] | norm; eps_n)   -- l==4: xn is written into ws.n rows 0..t
```
- The final norm is fused into the l=4 write. `dspine_draft` must NOT go through the `headed` closure (dflash.rs:640-655), which
  applies `RmsNorm(head.norm)` again. It calls `self.proj(&n.rows(1, t-1), self.head()?, &logits)` itself. `dflash_last_hidden`
  (dflash.rs:924) still reads `ws.n`.
- Copy hazard: `dflash2_block` (dflash2.rs:529-544) runs every row at `pos0 = last` with `window = d2.window + t-1-r`, because its
  block is non-causal. DSpine uses `pos0 = p0+r` and `window ∈ {0, W}`. Copying the DFlash2 loop gives a non-causal block that is wrong
  without any visible failure: no test outside parity catches it, because drafts never change the output.
- Test hook `Model::dspine_trace(bool)`. When on, each layer's `res` (post-write) and `xn` are downloaded after the layer into
  `dspine_layers() -> Vec<[Vec<f32>; 2]>`. It is off by default and costs nothing when off.

### 4.3 Tail (host): `dspine::walk(tabs, rows, stash, cand, anchor) -> Vec<u32>`, a pure function
- Downloads: logits `[t-1,V]` f32, `stash.raw[1..t]` `[t-1,H]` bf16, `stash.rms[1..t]` f32, `stash.packed[1..t, 0..R]`.
  The packed download is the full rows `[t-1,3R]`, because the view is not contiguous per row.
- Candidates: per row, `key = bf16(logit)` as f32. Take the K largest by key; on equal keys the lower id wins. Sort the K ids
  ascending, so lane = position in ascending-id order. Use `row_topk` (dflash2.rs:87) on the rounded row (the rounding is a new
  4-line helper) and rayon over rows.
- Walk (ref §3.2 restricted to the chosen sender; equivalent to the full K×K table plus `_chain`). Start with `s = anchor`, then for
  each row i in 0..t-1 (stash row i+1):
  ```
  m[k]   = sigmoid((q[k] + f32(TSV[s][k])) + bias[k]) * f32(TSV[s][R+k])          f32, k < R
  base_b = Σ_h (f32(raw[h]) * nw[h]) * f32(head[c_b][h])                             nw = f32(norm.weight)
  S_b    = Σ_k m[k] * f32(TG[c_b][k])
  score_b = base_b + rms * S_b
  j = first argmax over lanes b; draft[i] = c_j; s = draft[i]
  ```
  `head[c]` is the dequantised live-head row: bf16 bits, or `bf16(f32(e4m3) * ws)` with `ws` the tensor's one fp8 scale.
  Cost per pass: 240 head rows × 5120 = 1.2 M MAC, plus 15×16×512 for S. Well under 1 ms.
- `drafts.truncate(k)` as today.
- A test-only `dspine::table(...)` computes the full `[n,K,K]` scores plus the chain. A test asserts that it equals `walk`.

### 4.4 Tables, built at load after plan admission: `Model::dspine_build_tables()`
For chunks `[v0, v0+chunk)` with chunk = `ws.max_tokens`:
- **TG:** on the host, `x = bf16(f32(head[v]) * nw)`; upload x to staging `[chunk,H]`; `Matmul(x, write4_t)` gives `[chunk,R]` bf16;
  download into TG.
- **TSV:** on the host, `y = bf16(32 * f32(codebook[v]))` (exact power-of-two scale, read from the draft source); upload; `Matmul(y, pv_w)`
  gives `[chunk,2R]`; download into TSV.

This is about 1.8 TFLOP and 2.5 GB of uploads, expected to take a few seconds. Print `dspine: tables built in X s`. The build must run
where `exec` resolves admitted kernels (after `dry_run`). `dflash_dry_run` records one chunk of each of these two Matmuls, so
`serve` admits them. The host head copy is made in `load_shard` beside `load_head` (model.rs:951-990):
- **fp8:** re-run `quant::head::quantize_host(Fp8, ...)` on the checkpoint bytes. It is deterministic, so the result is the device
  tensor's bytes plus scale. Debug builds assert the scale equals the device QTensor's `parts[1]`.
- **bf16:** copy the bf16 bytes.
- **Other head encodings:** refuse.

## 5. New ops (`ops.rs`: variant, `Op::ALL`, `as_str`, `arity`, shape checks; CPU `ref_*` = admission oracle; `fast.rs`)

All four ops operate on bf16 storage, with f32 math in the stated order. `f(x)` means f32 of the stored value and `bf()` means round to
bf16. `FLT_MIN = 1.1754943508222875e-38`. Each op is one threadgroup or workgroup per row, with the reduction in f32.

| op (as_str) | in → out | semantics (ref §2.4 step) |
|---|---|---|
| `BridgeNorm` (`bridge_norm`), arity (1,2) | x `[T,D]` → y `[T,D]` bf16, rms `[T]` f32; attrs.eps = eps_b | `rms = sqrt(max(Σf²/D, FLT_MIN))`, `y = bf(f / (rms + eps))` (steps 1-2) |
| `BridgeCode` (`bridge_code`), (1,1) | p `[T,C]` → code `[T,C]`; eps = eps_b, scale = sqrt(C) | `n = sqrt(Σf²)` (no clamp), `code = bf((scale * f) / (n + eps))` for every row including row 0, whose value is unused (step 4) |
| `BridgeGate` (`bridge_gate`), (3,1) | packed `[T,3R]`, bias `[R]`, s0 `[2R]` → msg `[T,R]` | `msg[0] = 0`. For t ≥ 1, `(s, v) = t==1 ? (s0[:R], s0[R:]) : (packed[t-1,R:2R], packed[t-1,2R:])`, `msg[t] = bf(sigmoid((f(q[t]) + f(s)) + f(bias)) * f(v))` (steps 5-6, row-0 sender per §0) |
| `BridgeWrite` (`bridge_write`), (4,2) | raw `[T,H]`, rms `[T]` f32, delta `[T,H]`, w `[H]` → out `[T,H]`, normed `[T,H]`; eps = eps_n | `out = bf(f(raw) + (rms * f(delta)) * (row != 0))` (row 0 is bitwise raw), `y = bf(f(out) * rsqrt(Σf(out)²/H + eps))`, `normed = bf(f(y) * f(w))` (steps 8-9) |

| backend | work |
|---|---|
| CPU | `ref_bridge_*` in core (exact formulas above) + `fast.rs` rayon versions |
| Metal | `bridge_{norm,code,gate,write}_bf16` in `kernels.rs` registration + `MAIN_KERNEL_LIST` pins (kernels.rs:897-967); `fast::` exp/rsqrt are fine (ulp) |
| HIP | same 4 kernels in `hip/kernels.rs` `all()` (:241); HIP `AddBias` is absent but not needed |
| CUDA | none (spine refused at load) |
| admission | cases at T ∈ {1, 2, 8, 9, 15, 16}, D ∈ {C=1024, H=5120}, R=512 plus tiny-fixture shapes; `kernels test` rows |

Existing ops used: Embedding, RmsNorm, QkNorm, Rope, Matmul (draft and table shapes), KvAppend, AttnDecode (plain and window, `win_split`),
Add, SiluGlu, Logits (target head, fp8 or bf16).

Per layer this is 7 launches more than a conv-free Qwen3 layer, and one fewer RmsNorm because it is fused. Attention stays at t
launches per layer, so t=16 gives 80 `AttnDecode` launches per pass. Other non-bridge norms (layer-0 ln1, ln2, hidden_norm, q/k
norm) stay single-rounding `RmsNorm`/`QkNorm`. The reference does not pin their rounding (ref §2.5); MLX's fused `rms_norm`
presumably rounds once, and parity decides.

## 6. Integration edits (file:line at 831a895fc)

1. `dflash.rs:26-60`: add `DflashConfig.spine`. `load_head2` (:271) passes the spine config to `load2`. `dflash_slot_specs` (:362):
   unchanged, uses the draft's kv heads and head_dim.
2. `dflash2.rs`: change `Layer2.aconv/mconv` to `Option<Conv2>`. `load2` (:266) skips conv when the checkpoint is a spine, sets
   `selector = None`, `cand = K`, `topk = None`, and calls `dspine::load_spine` to fill `Draft2.spine`. `dflash2_block` unwraps conv,
   because it is never called for a spine.
3. `dflash.rs:620` (the v2 branch of `dflash_draft_rank`): `if let Some(sp) = &d2.spine { return self.dspine_draft(st, d2, sp, next, t) }`
   with `t = min(head.block, k+1)`. That function runs `dspine_block`, then logits, candidates and `walk`. It returns empty drafts when
   `p0 + t > d.kv.cap`, as today, and returns early under the recorder (after the logits) so the dry-run stays device-only.
4. `dflash_dry_run` (dflash.rs:935-990): for a spine, loop `t in 2..=block`. Each t gets one plain pass at context 0 and one windowed
   pass at `cap - 2t` when `W > 0 && cap >= W + 2t`. Also add one table-build chunk of each Matmul. The ingest loop (1..block) is kept.
5. `model.rs:1465-1510`: refuse tp>1 for a spine. Build the host head copy in the `load_head` block. Call `dspine_build_tables` after
   `dry_run`/admission (model.rs:1820 region).
6. `engine.rs:1424` `spec_k` clamp: unchanged (`verify_rows - 1`). `--spec auto` is not touched: its priors are DFlash2's
   (spec.rs ~1065), and DSpine is not defaulted anywhere.
7. Every existing `v2` consumer, and what it does for a spine. All of them are listed, so that no DFlash2-only path runs silently:
   - `load.rs:383` v2 validation: replaced by the spine branch (§2).
   - `load.rs:596-601` backend gate: unchanged. The message says "DSpine" when `spine.is_some()`.
   - `model.rs:1469` split decision (`INTARSIA_DFLASH_TP`): false for a spine.
   - `model.rs:1483` tp check: a spine errors at `tp.world > 1`.
   - `model.rs:1499` load-log label: ", DSpine".
   - `dflash.rs:114` `n_layers`: unchanged.
   - `dflash.rs:142` geometry check: skipped for a spine, as for v2.
   - `dflash.rs:363` slot specs: unchanged.
   - `dflash.rs:489` ingest: unchanged.
   - `dflash.rs:579` trace: unchanged.
   - `dflash.rs:620` draft: spine dispatch first.
   - `dflash.rs:968` dry-run window: the spine branch of edit 4.
   - `CostParams::DFLASH` (spec.rs:627, off by default) and the auto priors (spec.rs ~1065): left alone. The implementer re-greps
     `v2` and `Draft2` and adds any consumer missing here.
8. `docs/tuning.md`: no new env var in the MVP. Optional `INTARSIA_DSPINE_TRACE` only if the test hook needs an env switch;
   otherwise it is API-only.

## 7. Parity and gates

### 7.1 Tiny fixture: `tools/dspine/dspine_mlx_fixture.py` (committed, with output in `crates/intarsia-models/tests/fixtures/tiny_dspine/`)
Run with `/opt/code/TensorFold/.venv/bin/python -I tools/dspine/dspine_mlx_fixture.py --repo <DSpine repo> <out>`. The script does
`sys.path.insert(repo/dspine_mlx)` and sets `mx.set_default_device(mx.cpu)`. Activations are bf16 throughout, because the
reference raises on any other dtype (`bridge.py:105-106`).

- **Geometry:** binds `tests/fixtures/tiny_qwen3_5`: `model.embed_tokens.weight` → bf16 `nn.Embedding`, `lm_head.weight` → bf16
  `nn.Linear` wrapped in `HeadAccess`. H=64, V=96, 4/2 heads × 16, ffn 48, L=3, taps [1,3,5], C=16, R=8, K=4, block 6, W=8,
  MASK=95, eps 1e-6 for both norms (the fixture's own values), theta 1e6. The CPU target loads at `act_dtype: Some(Bf16)`; the
  precedent is `tp_hybrid.rs:409-467`, which runs a GDN hybrid at bf16 act on CPU. The spine's scratch, stash and bridge buffers
  are always bf16, whatever the target act, so the bridge rounds at the reference's points even if a target falls back to f32.
- **Weights:** random, bf16, HF layout, including the 4 version scalars, `out.weight` all zeros and all `selected_prefix_*` keys.
- **Two configs:** the stripped one is handed to MLX (`full_attention`, no sliding fields). The written `config.json` keeps
  `layer_types = sliding_attention`, `sliding_window = 8`, `use_sliding_window = true`, which is what our loader parses.
- **Window oracle:** inside `dspine_mlx.model`, the script replaces `mx.fast.scaled_dot_product_attention` with a wrapper. The
  wrapper passes an explicit boolean mask `kpos <= qpos && qpos - kpos < W`, with `qpos = Sk - L + i` (bottom-right aligned) and
  keys `0..Sk`. This gives the FlashAttention `window_left = W-1` convention SGLang uses. The reference code itself is unmodified.
- **Cases** (taps are random bf16 `[P, 3H]`, anchor random):
  - (a) P=2, T=6: the window is a no-op, because P+T-1 ≤ W-1.
  - (b) P=3, T=6: only row 5 is past the window.
  - (c) P=13 ingested as 9 then 4 (two `append_context` calls), T=6: every row is windowed.
  - (d) case (c) with T=3: a served shorter block.
- **Dumps per case:**
  - `feat`, `ids`;
  - per-layer `(out, normed)`, collected by wrapping `DSpineCodeBridge.__call__`;
  - final hidden;
  - stash `raw/rms/packed`;
  - pass-1 logits (bf16 → f32), `cand`, drafts, and argmax drafts;
  - TG and TSV in full (tiny);
  - margins: the K-boundary logit gap per row and the top-2 score gap along the chain (computed by a small f64 numpy scorer in the
    script, our code), plus the per-row max |logit| and max |score|.
- **Script asserts:**
  - every margin is at least `5 × tol × max|value|` of its row, with `tol = 2e-3`, the CPU hidden-row bound of §7.2. Otherwise it
    reseeds (seed loop). The bitwise cand and draft gates therefore cannot flip on noise the hidden-row gate allows;
  - drafts ≠ argmax in at least one case, so the tail is exercised;
  - case (d)'s rows equal case (c)'s first 3 rows within MLX noise.
- **Size target:** about 150 KB.

### 7.2 `crates/intarsia-models/tests/dspine_cpu.rs` (CPU backend, tiny hybrid target loaded with `act_dtype: Some(Bf16)`)
Modelled on `dflash2_cpu.rs`.
1. `dspine_draft_matches_the_mlx_reference`, for each case: `open_dflash` → `Model::load` → `dflash_ingest_host` (chunked as in the
   case) → `dspine_trace(true)` → `dflash_draft(next=anchor, k=T-1)`. Gates:
   - per-layer `out`/`normed` and the final hidden: rel L2 ≤ 2e-3 per row. Both sides do f32 math with bf16 stores; the only
     known extra rounding is the `fc` partial sums;
   - stash `raw`: ≤ 2e-3; `rms`: rel ≤ 1e-3;
   - `cand`: equal sets;
   - drafts: bitwise.
2. `dspine_tail_is_the_reference_on_its_own_inputs`. `walk` on the golden stash, cand and tables matches the golden drafts bitwise,
   and `table` equals `walk`. Our built TG/TSV against the golden tables: ≥ 99 % of elements bitwise, the rest within 1 bf16 ulp.
3. `bridge_ops_match_f64`: the 4 CPU refs against an f64 restatement, with T ∈ {1, 2, 6}. Row 0 of `BridgeWrite` is bitwise raw,
   `msg[0] == 0`, and row 1 uses `s0`.
4. `shorter_block_is_a_prefix`: the t=3 hidden and stash equal the first 3 rows of t=6, bitwise on CPU. This is the property the
   `min(block, k+1)` rule relies on.
5. `greedy_ids_equal_spec_off` at k ∈ {1, 3, 5} with prompts longer than W (pattern of `dflash2_cpu.rs:430`).
6. Scripted drafts for every acceptance pattern (`DraftScript`, as in `dflash2_cpu.rs:559`).
7. Plan coverage: every t in 2..=block, plain and windowed, then serve with admitted kernels only.
8. Loader refusals:
   - a codebook tensor without the config key, and the config key without the tensor;
   - a version scalar ≠ 1, a nonzero `out.weight`, `terminal=true`, `feature_layers` ≠ 1..=L, K=12;
   - tp=2, `--dcoco`, and v1 loading a spine directory.

### 7.3 Metal
- `crates/intarsia-backend-metal/tests/dspine.rs`. The 4 kernels against the CPU ref at real shapes (H=5120, C=1024, R=512) with
  T ∈ {1, 2, 8, 9, 15, 16}. Gates: outputs within 1 bf16 ulp (≥ 99.9 % bitwise), `rms` rel ≤ 1e-6, row-0 passthrough and `msg[0]`
  bitwise. A plain-call check confirms the window `AttnDecode` is only picked when `window > 0` (the existing test covers this).
  No Metal test loads `tiny_qwen3_5` today (its head_dim of 16 and f32 act are unproven there), so model-level Metal parity uses
  the real geometry below.
- Real-weight fixture, opt-in, not committed, written to `.worktrees/dspine/fixture-real/`. Steps:
  1. A Rust example `examples/dspine_taps.rs` runs the 27B Quark target on Metal (`INTARSIA_LMHEAD_QUANT=off`) over a fixed
     300-token prose prompt and writes `feat [300, 25600]` bf16 plus the token ids.
  2. `tools/dspine/dspine_mlx_real_fixture.py` builds a temp dir under `.worktrees/dspine/fixture-real/mlx-draft/`. It holds a
     stripped `config.json` (`full_attention`, no sliding fields) and a symlink to the real `model.safetensors`, because
     `config.py:232-249` refuses `sliding_window`. It loads that directory with `load_draft` and binds bf16 `embed_tokens` and the
     bf16 `lm_head` read from the Quark checkpoint's own tensors (excluded from quantisation; confirm the embed's dtype). It runs
     MLX on the GPU in bf16 for these cases:
     - P=300, T=8 and T=16, with the patched SDPA at W=8192 (a no-op at P=300, kept so both window cases use one code path);
     - 200+100 chunked;
     - an artificial window W=64 at P=300: our side sets `DflashLoad.config.sliding_window = 64` before load, the MLX side uses
       the patched SDPA with W=64.

     It dumps per-layer rows, hidden, stash, cand, drafts, and the TG/TSV rows for every id it touched plus 1024 random rows.
  3. `crates/intarsia-models/tests/dspine_metal_real.rs` (`#![cfg(target_os = "macos")]`) is gated by
     `INTARSIA_TEST_REAL_MODEL=1`, `INTARSIA_DSPINE_DRAFT`, `INTARSIA_DSPINE_FIXTURE` and an optional `INTARSIA_DSPINE_TARGET`
     (default `~/models/Qwen3.8-27B-Quark-AWQ-MXFP4`). Gates:
     - per-layer, hidden and raw: rel L2 ≤ 2e-2;
     - mean Jaccard of the cand sets ≥ 0.9;
     - draft-token agreement ≥ 0.8;
     - tail on the fixture's own stash, cand and table rows: bitwise drafts;
     - our TG/TSV rows: ≥ 99.9 % bitwise, the rest ≤ 1 ulp.

     Run it in a fleetq slot on endor:
     `fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 30 --label "dspine metal parity" -- cargo test ...`.

### 7.4 End-to-end greedy gate (Metal, then HIP)
- **Script:** `scripts/dspine/ids.sh`, modelled on `scripts/dfh/stages-ids.sh` and `scripts/dm/gen.sh`.
- **Setup:** Quark 27B MXFP4 target with the default recipe (fp8 head), `--spec dflash --spec-draft <DSpine>` at `--spec-k` 3 and 7,
  against `--spec off`. Greedy, 256 new tokens.
- **Prompts:** short (about 20 tokens), about 4k, and about 9,000 tokens. The last puts every row past the window (P > 8176).
- **Gates:**
  - ids identical in all 6 runs;
  - acceptance floor: mean accepted drafts per verify ≥ 2.0 at k=7 on the short and 4k prose prompts, reported next to DFlash2 on
    the same prompts (docs/results/2026-11-06-metal-dflash2.md). Identical ids alone pass even with a broken drafter;
  - `plan show` no-op for 8B/27B/35B without a draft (`scripts/dm/noop.sh`).
- **HIP:** the same gate on tatooine at k 7 and 15 with `/data/models/radiance/hf-cache/.../amd--Qwen3.8-27B-Quark-AWQ-MXFP4`, plus
  `kernels test` for the 4 ops. Run it under `fleetq run --host tatooine --quiet gpu,cpu --expect 45 ...`.
- **Results doc:** `docs/results/2026-11-xx-dspine.md`, with parity numbers, ids, acceptance and decode tok/s at k 3/7 (Metal) and
  7/15 (HIP) against MTP k2 and DFlash2.

## 8. MVP vs later

MVP: everything in §2-§7. That is greedy only (the engine's spec path is greedy-only anyway, `spec_on` engine.rs:1418), fixed `--spec-k`,
tp=1, host candidates and host tail, dense draft KV, Metal, HIP and CPU.

Later, roughly in order of value:
1. Device top-k with bf16 keys (HIP `topk_rows` mode) and device gather of head rows. Together they drop the 15×V logits download
   and the 1.27 GB host head copy.
2. Multi-row causal `AttnPrefill` when the window is inactive (5 launches instead of 80). Needs head_dim 128 prefill admission on
   Metal and HIP.
3. Adaptive depth (#342 `spec::Depth`) for dflash. The causal block makes a shorter depth shrink the draft pass too.
4. A TG/TSV disk cache keyed by (head bytes hash, quantisation).
5. A W+T ring for draft K/V (160 MiB per sequence instead of 20 KiB per token).
6. Fused bridge kernels (norm and code in matmul epilogues).
7. A sampled path: an exact fp32 table and sampled chain (ref §3.6), plus rejection verify with the chain's q (ref §4.2).
8. Dynamic verification (DSpark caps, SPS calibration; ref §4.3). Speed only.
9. tp>1 (the draft on rank 0 under a vocab-parallel head, then a split draft), the CUDA backend, an mxfp4 head, draft quantisation
   (fc and the layer linears only; the bridge stays bf16), and `--spec auto` priors.

## 9. Work order

Implementation agent (done means steps 1-8 green, with step 8 limited to the Metal kernel test):

1. Step 0 (§1).
2. Config and loader with refusals (§2), plus unit tests.
3. The 4 ops: CPU ref and fast, then Metal, then HIP, with admission (§5).
4. Fixture script and committed tiny fixture (§7.1).
5. `dspine.rs` load and block, the `Draft2` conv `Option` change, dispatch, dry-run (§4.2, §6).
6. Host head copy, tables and walk (§4.3-4.4).
7. `dspine_cpu.rs` green (§7.2).
8. The Metal kernel test `crates/intarsia-backend-metal/tests/dspine.rs` (§7.3).

Follow-on validation stage. It needs the 4 GB checkpoint (§1) and fleetq slots on endor and tatooine:

9. Real-weight fixture and `dspine_metal_real.rs` (§7.3).
10. The ids gate and acceptance floor on Metal (§7.4).
11. HIP `kernels test` and ids gate.
12. Results doc.

Existing tests (`dflash2_cpu.rs`, `dflash2.rs` Metal, `kernels test`, no-op gates) must stay green after step 5, because the `Draft2`
change touches DFlash2.

## 10. Open items
- Real `config.json` values (§1). Whether `rope_theta` is 1e7 as stated in the brief.
- FA3 `window_left = W-1` is assumed to be the FlashAttention convention, not read from SGLang source. The fixture and our code both
  use `qpos - kpos < W`. The only oracle past 8176 is SGLang CUDA (not available here). Probing P ∈ {8176, 8177, 8191, 8192, 9000}
  against it comes later.
- Whether MLX `nn.RMSNorm` rounds once (our assumption for the non-bridge norms). The tiny fixture's per-layer gate shows it.
- fp8 head: the reference's quality claims are for a bf16 head. Measure acceptance with the fp8 head against `INTARSIA_LMHEAD_QUANT=off`
  in the results doc.
- Metal verify with more than 8 rows on MXFP4 (eng §1.5). Keep Metal gates at k ≤ 7 until a separate check covers 16-row verify.
- GDN `SpecRows` memory at a 16-row verify on HIP: 15 × 151 MB = 2.3 GB per sequence, plus the draft's 3.3 GiB and host tables.
  Check it fits the R9700's 32 GB before picking a default k.
