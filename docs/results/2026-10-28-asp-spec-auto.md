asp: `--spec auto`, per-request choice between the MTP head and the DFlash2 drafter, Qwen3.8-27B MXFP4 tp=2 (HIP), 2026-10-28

Lane `lane-asp/spec-auto`, base `origin/main` `bb48df640`. Status: design note (this commit); results below as they land.

## 1. Design

Both drafters are lossless under greedy verify: a draft only proposes, the row-exact verify pass decides every emitted token. So switching drafters per pass cannot change the output as long as the target state (KV, GDN rows, rollback) is untouched, which it is: the choice only picks who proposes. What must stay valid for a switch is each drafter's own context:

- MTP head (`MtpState`): queues the hidden row of every pass (verify passes queue without catch-up, rollback keeps the accepted rows); `mtp_draft` catches up over any number of queued rows in one run. Under DFlash-driven passes the queue only grows, and `mtp_queue` overflow would run a collective inside a peer thread; so before a DFlash pass the engine runs a lockstep `Model::mtp_catch_up` when the queue is near `cap_rows`.
- DFlash2 (`DflashState`): ingests the features of every pass. Under MTP passes that would cost a 1.4 ms flush per pass (5% of a 20.8 ms pass), so in auto mode `DflashState::lazy` makes every pass stash its feature rows (a prefix of the stash is exactly the accepted rows after rollback, `DflashState::cut`) and the rows are ingested in one batch before the next DFlash draft.

Policy (`spec::AutoPolicy`, pure host logic, deterministic): per drafter an EWMA of tokens per pass and of measured ms per pass (draft + verify + rollback, excluding the first pass after a switch, which pays the catch-up); score = tokens / ms; the better score wins with a 3% hysteresis; each drafter gets 2 seeding passes at request start; the idle drafter is probed for one pass every 48 passes (192 when the winner leads by 30% or more). `INTARSIA_SPEC_AUTO=mtp|dflash|alt:N` forces a drafter or switches every N passes (tests, A/B).

Memory: DFlash2 bf16 draft (3.85 GB) on rank 0 next to the MTP head.
