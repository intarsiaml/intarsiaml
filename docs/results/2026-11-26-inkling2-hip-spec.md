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
4. 2026-11-26 ~12:25 EST (coordinator): run the tiny Metal suite once, in one exclusive endor slot. This is a one-off exception to
   COMMON's endor rule: `fleetq run --host endor --exclusive --expect 8 --label "ink-hip-spec metal no-op (tiny)"`, `env
   INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2 --test inkling2_dspark --test
   inkling2_dspark_cut --test inkling2_verify_softcap`. Tiny models only. Build it first in a `--load cpu` slot. If anything fails,
   stop and report; no second exclusive slot. Pass counts go next to the cpu runs. Card 1 is fine for identity work if the run held a
   fleetq slot covering it (name the job ids). Speed stays on card 0. Go ahead with the paired speed windows.

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
  failures are the screened-head tests: they force `INTARSIA_INK_HEAD_SCREEN=1`, and HIP has no `HeadScreen` kernel, so the load
  refuses with `NoKernel` (panic at `load_screen`, tests/inkling2.rs:91). Those two failures were already recorded for hip, by design,
  in docs/results/2026-11-25-inkling2-hip.md (test table, line 175), and the chunk the branch's `on_metal()` change sets does not
  matter there, because they fail at load. `inkling2_verify_softcap` 1 / 1,
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

## 3. Window 2 (fleetq #12048, 12:13:34-12:17:24 EST, radiance down 3 min 50 s; both cards: #12048 held tatooine-r9700 and
## tatooine-r9700-2, so card 1's identity runs were inside the slot)

Binaries: `base` = main `87c9745c0` + PR #526 (`da8501b51`, the HIP prefill flip: WMMA experts, flash prefill attention, chunk 2048),
md5 `492211cf` (= pfflip's `intarsia-pf`); `spec2` = this branch with #526 merged (`e49b8a29c`), md5 `f7494043`. No env set unless named.

**Brief 3 checks.** (1) With #526's defaults and no env, 2..4-row verify rows are bitwise the one-row decode steps on the real model:
`verify_rows_real` rows 2..=4 at 17 / 1024 / 4000 and 16 greedy windows, `PROBE_RESULT rows=91 windows_with_diff=0 argmax_flips=0
worst_abs_logit_diff=0e0` (verify graph on: 13 captures, 61 replays), and eager (`INTARSIA_VERIFY_GRAPH=0`, 1024): 73 rows, 0 diff. The
verify forward never reaches the WMMA experts or flash attention: under `exact_rows` the registry only takes `rows_exact` kernels (the
`_rows` expert / GEMV twins and `ink_attn_decode_bf16`), and the startup line names them (`moe_experts=moe_experts_situ_bf16_rows`,
`ink_attn_decode=ink_attn_decode_bf16`). (2) Identity on the rebased binary with no env: below, all equal. (3) not needed.

**Greedy ids, 128 tokens** (`--prompt-ids`, `--show-ids`; md5 over every `ids[...]` line of the process):

| Prompts | base `--spec off` | branch `--spec off` | branch gate 0.8 k3 | branch cut 0.5 k3 |
|---|---|---|---|---|
| 12 greedy golden prompts (`greedy.json`, lane ink-mtp's set) | `24b7ad602ac6` | `24b7ad602ac6` | `24b7ad602ac6` | `24b7ad602ac6` |
| 2 prose texts (`prose.json`: long_prose_300 / 700) | `c849c8651696` | `c849c8651696` | `c849c8651696` | `c849c8651696` |

**Text prompts, 256 tokens `--ignore-eos`** (md5 of the first 64 / all 256 ids):

| Prompt | base off | branch off | gate | cut |
|---|---|---|---|---|
| short-0 | `c5c42d88` / `5f925d56` | same | same | same |
| 4k-0 | `d1fc9761` / `36298f75` | same | same | same |

Acceptance: greedy set gate 70.9 % (2.54 tokens / pass), cut 57.5 % (2.36); prose gate 75.6 % (2.66), cut 59.5 % (2.27); short-0 gate
2.43, cut 2.12; 4k-0 gate 2.66, cut 2.55.

**Verify cost** (probe, 1024 context, graph on / eager): decode 13.30 / 13.24 ms; verify 2 rows 16.48 / 16.75 (1.24x / 1.26x), 3 rows
19.00 / 19.58 (1.43x / 1.48x), 4 rows 20.80 / 21.41 (1.56x / 1.62x). The verify graph saves 0.3-0.6 ms a pass.

**Draft cost** (`INTARSIA_DSPARK_PROFILE=1`, 1k, gate, means over 39 cycles): whole cycle 37.67 ms (= verify + draft + host), draft
block 2.11 ms, its first step 0.70 ms, head 0.71 ms, ingest 0.20 ms. The draft is ~6 % of a cycle; the verify pass is the rest.

**G2 at spec off (branch, no env):** short KL 4.491e-3 (n 569, top-1 99.47 %) PASS, long 4.083e-3 (n 1569, top-1 98.98 %) PASS: equal to
pfflip's G2 at the same defaults (4.491e-3 / 4.083e-3), so spec off is unchanged by this branch.

Single-process speed seen in passing (card 1, not the measurement): greedy set off 77.54 tok/s, gate 100.68 (+29.8 %), cut 102.06
(+31.6 %); prose off 76.13, gate 87.93, cut 92.52.

## 4. cpu and Metal tests unchanged

| Suite (tiny fixtures) | cpu (endor `--load cpu`, fleetq #12040, branch `d54ba1cf9`) | Metal (endor `--exclusive`, fleetq #12055, branch `0b17500ae`) |
|---|---|---|
| `inkling2` | 17 / 17 | 17 / 17 (screened-head tests included: Metal has `HeadScreen`) |
| `inkling2_dspark` | 4 / 4 | 4 / 4 |
| `inkling2_dspark_cut` | 1 / 1 | 1 / 1 |
| `inkling2_verify_softcap` | 1 / 1 | 1 / 1 |
| `intarsia-core` lib (incl. `dspark::rule_tests`) | 266 passed, 4 ignored | n/a |
| `intarsia-models` lib (incl. load / `dspark_default`) | 80 / 80 | n/a |
| `intarsia-cli --test docs_env_vars` | 2 / 2 | n/a |

The Metal run is the brief 4 one-off: build in fleetq #12054 (`--load cpu`), tests alone in #12055 (12:24:30-12:27:42), tiny models only.
Why nothing on Metal or cpu can move: `ink_rows_exact` is `matches!(backend, "metal" | "cuda" | "hip")`, the same value for "metal"; the
verify-graph default adds "hip" beside "cuda"; the loader admits one more backend name; `verify_check` (a diagnostic, not the decode
path) now softcaps the verify rows, which is a no-op on models without a softcap and fixes the Inkling-2 / Gemma-2 comparison on every
backend.
