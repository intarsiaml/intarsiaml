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
