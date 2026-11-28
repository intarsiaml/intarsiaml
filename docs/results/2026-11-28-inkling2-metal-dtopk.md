# Inkling-2-Mini on Metal: DFlash2 candidates by the device top-k (lane ink-metal-dtopk, opt-in, 2026-11-28)

Status: **gates pre-registered, slot D not run yet.**

## Why

Each speculative pass on Metal takes the DFlash2 drafter's 16 candidates per block row on the host. With the Metal default since #557
(k7 + cut 0.3), the host fetches one row's f32 logits at a time (200058 x 4 bytes, a download and a synchronisation each) until
the cut stops the walk, and takes each row's top 16 on the CPU. HIP has taken them on the device since lane dh item 2
(`INTARSIA_DFLASH_DEVICE_TOPK`, on unless `0`), downloading only 16 (id, value) pairs a row. Metal has had a working
`Backend::topk_rows` since lane ink-mtp (`crates/intarsia-backend-metal/src/topk.rs`: the sampler's order, value descending, ties
by ascending index, NaN rows flagged), but `supports_topk()` stays false there, because device sampling (`batch.rs`) and the DCoCo
corrector (`dcoco.rs`) key on it. The DFlash2 draft therefore never used it on Metal.

The candidates do not decide the output (greedy verification keeps the target's tokens), only which drafts are proposed, so ids
cannot change. The candidates must still be bit-identical to the host's, or the acceptance rate (and so the speed) would move.

## The change (branch `ink/metal-dtopk`, on #557's head `9bafcdf02`)

`crates/intarsia-core/src/model/dflash2.rs` `device_topk_resolve`: a backend with `supports_topk()` (HIP) keeps "on unless `0`";
Metal takes the device top-k only with `INTARSIA_DFLASH_DEVICE_TOPK=1` or `=check` (opt-in; the trait is unchanged, so device
sampling and DCoCo are untouched); every other backend stays on the host. A Metal load prints
"dflash: candidates by the Metal device top-k (...)". Unit test in `dflash2.rs` `cut_env_parses`.

## Gates (pre-registered before slot D)

One binary, one endor `--exclusive` fleetq slot, `raw/slotD.sh`. Arms: `dflt` = the Metal default (no flags, no env); `dtopk` =
`INTARSIA_DFLASH_DEVICE_TOPK=1`; `chk` = `=check`; `off` = `--spec off`.

| # | Check | Rule |
|---|---|---|
| D0 | Draft phases (`INTARSIA_SPEC_TRACE=1`, speed.json) | reported, no bar: median `dl + select` a pass for `dflt` (today's baseline, not measured before: the cut lane's traces ran with the conf trace on) and `dtopk` |
| D1 | Candidates bit for bit | `chk` over g3.json (256 tokens, `--ignore-eos`) and speed.json: every `dflash2 topk check` line reports 0 rows differing |
| D2 | Greedy ids | g3.json md5 `off` = `dflt` = `dtopk` = the goldens (`c69d1438...`, `9e85ba11...`, `89994b87...`) |
| D3 | Decode | ABBA x 4 over speed.json (`raw/rule.py dtopk dflt`, `RULE_BAR=1.01`): ids identical in every process. Opt-in PR: no cell with a CI upper bound < 0.99. Default-on recommendation (the user's call): also mean of the six cells >= 1.01 |
| D4 | Serve | `serve --max-batch 1`, `dtopk` vs `--spec off`: 12 / 12 turns' ids and `cached_tokens` equal |

Sampled requests do not draft on Metal by default, so the change cannot reach them (no sampled gate). 45 s GPU idle before every
timed run, macmon logged.

**Decision rule:** D1, D2, D4 pass and D3 shows no slow cell: opt-in PR, merged on green CI. Any candidate difference (D1) or ids
mismatch: no merge, report. D3 mean >= 1.01: ask the user in the morning whether to make it the Metal default.
