# Inkling-2-Mini on HIP: DSpark speculative decoding (lane ink-hip-spec, 2026-11-26)

Branch `ink/hip-spec` (worktree `.worktrees/ink-hip-spec`), base main `87c9745c0`. Goal: DSpark speculative decode (the checkpoint's own
draft, `mtp.safetensors`) for Inkling-2-Mini on one R9700 (tatooine, gfx1201), greedy ids identical to `--spec off`, faster
single-stream greedy decode. Shared rules: `.worktrees/hip-swarm/COMMON-1126.md`.

## Briefs received

1. 2026-11-26 ~11:45 EST (coordinator, lane brief): bring DSpark to HIP (verify, rollback, draft ops, `ink_rows_exact` on hip, device
   argmax, verify graph); tests `inkling2_tiny_rollback_by_length` and `inkling2_tiny_verify_graph_replays_equal_decode_bitwise` pass on
   hip; correctness bar (12 greedy prompts x 128 tokens + the prose texts, admission, G2 at spec off unchanged); speed bar (spec on vs off
   at short / 1k / 4k / 16k, 4 + 4 paired processes with a CI, both rules: cut 0.5 k3 and gate 0.8 k3); default-on for HIP only if on >=
   off at every length and >= +2 % at more than one length with CI lower bound > 1.0 (`[tuning.hip] mtp_default_k` + the rule; the section
   comes from lane ink-hip-pfflip), else opt-in via `--spec dflash`.
2. 2026-11-26 ~12:00 EST (coordinator, COMMON fix): don't pass `--mem` on tatooine jobs (it counts against the card's 32 GB that radiance
   holds, so the job never starts and blocks the queue). This lane's jobs carry no `--mem`.
3. 2026-11-26 ~12:05 EST (coordinator): overlap accepted; rebase on PR #526 (HIP prefill flip) before the gate. #526 makes
   `INTARSIA_HIP_INK_MOE_WMMA` on (process-wide, `kWmDefault` 0 -> 1 in kernels/hip/ink_moe.hip), `INTARSIA_HIP_INK_ATTN_FLASH` on
   (per thread) and prefill chunk 2048 (`[tuning.hip]`) the HIP Inkling defaults; `tests/inkling2.rs` `backend()` hip arm calls
   `inkattn::set_flash(0)` and `inkmoe::set_wmma(0)` before `create(0)` (keep both hunks). Check and write here: (1) with the new
   defaults (no env), are 4-row verify rows still bitwise the one-row decode (does WMMA experts or flash attention ever fire on a
   4-row verify forward; a row threshold below 4 makes `ink_rows_exact("hip")` wrong under the defaults); (2) the identity check
   (spec on vs `--spec off`, greedy md5, short + 4k) on the rebased binary with no env set; (3) if (1) fails, gate the verify forward
   to the split-KV / decode-expert paths for rows <= k+1, or report back first. `ink_rows_exact` line: whichever of this lane and
   ink-hip-batch lands second takes the other's.

## 1. What HIP lacked (checked against the code, main `87c9745c0`)

| Needed | State on HIP before this lane | Done here |
|---|---|---|
| `--spec dflash` with a DSpark draft | refused by the loader (`load.rs`: "cpu, metal and cuda only so far") | hip admitted |
| `ink_rows_exact` (passes of 2..=16 rows under `exact_rows`) | metal / cuda only | `"hip"` added (the one core line only hip sees) |
| Multi-row verify kernels bitwise the one-row decode | already there: `_rows` twins of in_proj / MoE / bf16 GEMV / logits (rows-exact), op 75 at 1..=16 rows, row kernels; `inkling2_tiny_verify_rows_equal_decode_bitwise` passed on hip before this lane | none needed (real-model probe below) |
| Rollback by length (conv rings, SWA rings, attention caches) | core-owned (`SequenceState::truncate`, ring margin 16); the hip failure of `inkling2_tiny_rollback_by_length` was the test's chunk: 64 on hip (one 40-row WMMA pass as the reference) | test helper `on_metal()` takes hip (chunk 16), the reuse assert takes hip's ladder policy like CUDA's |
| Device argmax over the R verify rows | `argmax_f32` over `[R, V]` (the Qwen DFlash2 verify already used it) | none |
| Verify graph capture / replay | `verify_graph_ok` allowed hip, default on only for Inkling on CUDA | default on for Inkling on hip too (`INTARSIA_VERIFY_GRAPH=0` reverts) |
| Draft ops: Matmul / Logits / Rope / QkNorm / windowed AttnDecode / SiluGlu / RmsNorm / Embedding / Add, MXFP4 head | present (DFlash2 / DSpine on HIP) | none |
| Draft op `HcPre` (the four-tap mix) | `UNIMPLEMENTED` | `hc_pre_bf16` (kernels/hip/ink_spec.hip, port of CUDA's `hc_pre_kernel`, same fmaf order) |
| Device top-k of the draft's bf16 base rows (`Backend::topk_rows`) | f32 rows only: the draft would have fallen back to a host top-k (a 1.2 MB download and a host select per draft) | bf16 rows (port of CUDA's `topk_stage1<ik_bf16>`) |

Prefix reuse on hip is unchanged by the `ink_rows_exact` line: `Model::model_min_pass_rows` only applies where the backend returns
`prefix_reuse_min_pass_rows() = Some(n)` (Metal); hip returns `None` and keeps the cold-ladder policy.

## 2. Window 1 (fleetq #12042, 11:57:10-11:59:15 EST, radiance down 2 min 5 s; binary `spec1` = `d54ba1cf9` on main `87c9745c0`)

- Tiny suite on hip (`INTARSIA_INKLING2_BACKEND=hip`): `inkling2` 15 / 17, now including **`inkling2_tiny_rollback_by_length`**,
  **`inkling2_tiny_verify_graph_replays_equal_decode_bitwise`** and `inkling2_tiny_chunked_and_decode_match_prefill_bitwise`; the two
  failures are the screened-head tests (HIP has no `HeadScreen`, refused at load as designed, as before). `inkling2_verify_softcap` 1 / 1,
  `inkling2_dspark` 4 / 4, `inkling2_dspark_cut` 1 / 1.
- `tests/topk.rs` on hip 6 / 6 (new `device_topk_bf16_rows_are_the_host_order`: 1105 pairs at V 200058 / 201024 / 1000 / 70 / 300).
- `intarsia kernels test --backend hip --op hc_pre`: `hc_pre_bf16` 6 cases, 0 failing, ADMITTED.
- Real-model verify probe (`tests/verify_rows_real.rs`, rows 2..=4 at 17 / 1024 / 4000 and 16 greedy windows): **argmax equal on every
  row**, but every row "differed" by up to 37 in some logit. Cause: `Model::verify_check` compared the capped decode logits with the
  verify pass's pre-softcap `ws.logits` (Inkling-2's final softcap 30). Fixed (`verify_check` caps the verify rows first; a no-op for
  models without a softcap); rerun in window 2. The cost lines were cut by my log filter.
- 1k prompt (1021 ids), 128 tokens, `--spec off` vs `--spec dflash --spec-k 3` (gate 0.8): **ids identical**; decode 74.75 -> 79.46
  tok/s (+6.3 %, one pair), 31 verify passes, 62 proposed, 44 accepted (71.0 %, 2.42 tokens per pass); no `device top-k unavailable`
  and no `no row-exact kernel` line; draft load: Markov fold 2.1 s, MXFP4 head 0.2 s.
