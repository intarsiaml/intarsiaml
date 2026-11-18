# mgdnv-flip: Metal fused GatedDeltaNet verify rows default-on (decision candidate)

Lane `mgdnv-flip`, branch from main `533843814` (PR #443 merged). **Proposed, user decision: not merged.**

## What flips

`INTARSIA_METAL_GDN_VERIFY_FUSE` (`crates/intarsia-backend-metal/src/gdn.rs:verify_fuse_enabled`): unset = on, `0` = off (same semantics as `INTARSIA_CUDA_GDN_VERIFY_FUSE`, `INTARSIA_HIP_GDN_VERIFY_FUSE`). Before: only `1` turned it on. Anything but `0` is now on, including `1`, so existing `=1` users are unaffected. `docs/tuning.md` row updated. Tests: `gdn_verify_snap_off.rs` now sets `=0` explicitly; new `gdn_verify_snap_default.rs` asserts unset takes the fused path (`Ok(true)`); `gdn_verify_snap.rs` (bitwise against the row-by-row path) unchanged.

## Evidence (all measured earlier, not repeated)

- `docs/results/2026-11-18-mgdnv.md` (PR #443): bitwise exact on the 35B, greedy ids equal; +1.5..3 % single sequence, +8..10 % with copy len 15.
- `docs/results/2026-11-18-mgdnv27.md` (PR #444, `git show origin/mgdnv27:docs/results/2026-11-18-mgdnv27.md`): exact on the 27B (7 of 7 cells), batched serving and sampled requests; 27B +1.4..3.2 %, +5 % at mb4.

## Gates run here (endor, fleetq slots, `raw/`)

| gate | result |
|---|---|
| `cargo test --release -p intarsia-backend-metal` (all tests, incl. `gdn_verify_snap`, `_off`, `_default`) | all pass |
| `cargo test -p intarsia-cli --test docs_env_vars` | pass |
| `plan show --json`, 35B, `--spec mtp --spec-k 2`: origin/main binary vs new default vs new `=0` | byte-identical (13188 bytes, sha a3bb42031070215f), the op is not registered |
| greedy ids, 35B default recipe (mtp k2): `--spec off` = new default = `=0` | PASS: chat-short 389 ids, code-gen 500, copy3k 2902 |
| one throughput run each (decode tok/s, default vs `=0`; single runs, noisy) | chat-short 189.5 vs 184.1, code-gen 214.5 vs 207.5, copy3k 178.2 vs 175.0 (+1.8..3.4 %) |

Raw: `raw/gates.log`, `raw/plan`, `raw/ids`; scripts `scripts/mgdnv-flip/`.

## Revert

Set `INTARSIA_METAL_GDN_VERIFY_FUSE=0` (no rebuild), or revert this PR.
