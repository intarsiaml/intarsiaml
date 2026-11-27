# Inkling-2 on HIP: a decode graph recorded without drafting is replayed by a drafting request (stale draft taps)

Lane ink-plain-tax, 2026-11-27. It started from the follow-up of `docs/results/2026-11-27-inkling2-metal-sampled-retune.md`: a plain step
of a request in sampled-drafting (policy) mode costs 3.1 % more than an off request's on Metal, and HIP likely pays the same. A code
read (explorer, Sonnet) of the single-sequence engine found that the retune doc's guess at the cause does not hold, and found a bug
on HIP's default path instead.

## What the code read found

- **The graph theory does not hold, on either backend.** Metal has no decode graphs (`Backend::supports_graphs` is false). On HIP,
  the graph capture records `ink_copy`. Only the plan dry-run (`Model::recorder`) skips it. The 3.1 % on Metal is most likely the
  lost token feedback (`Seq::feedback_ok` requires `!spec_on`, so a policy-mode request never pipelines its plain steps) plus 5 copy
  dispatches per step. This is an inference, not measured. Metal sampled drafting is opt-in, so no default pays it. Not pursued here.
- **The bug: the solo decode-graph key ignores the draft taps.** The key is `(st.id, head, full_logits)`
  (`crates/intarsia-core/src/model/graph_step.rs`). A forward writes the draft's tap copies only while the state drafts
  (`d.active && d.synced`, `crates/intarsia-core/src/model/inkling2.rs`). Nothing drops the recordings when that changes. So:
  1. A request that does not draft records the decode graph without tap copies.
  2. The next request that drafts on the same state replays that recording. Its plain steps write no taps, so `dflash_stash`
     copies stale rows from the staging buffer.
  3. The draft then proposes from those stale rows.

  Tokens never change, because drafts only propose. Acceptance (speed) drops.

### How a default HIP server reaches it

The setup is both drafts loaded (#550), greedy requests drafting with DFlash2 and sampled requests with DSpark.

- **Requests that do not draft:**
  - a sampled request with logprobs or top_logprobs (`trace_logits`), or with a constraint;
  - a sampled turn that reuses more than `DFLASH_RESYNC_REUSE_MAX` = 128 rows while the draft is unsynced (`Seq::new`). This
    happens, for example, after a greedy turn, which swaps in DFlash2. Swapping back to DSpark drops the graphs, the draft starts
    unsynced, and the long turn runs without drafting.
- **Requests that replay the stale recording:** the next sampled request that drafts on the state, for example a new conversation.
  `truncate(0)` resyncs the draft, and `select_dflash` returns early because no swap is needed, so nothing drops the recording.
- **The recording survives until** the next draft swap or `configure_plan`.

## Fix

The key gets a byte for the taps a step writes: `decode_taps(st)` = 0 none, 1 the first draft's, 2 the second draft's. This is one
function and the key type. A request that does not draft also stops replaying a recording that writes the taps (wasted copies). A
change of drafting state costs one extra capture.

## Gates (pre-registered 2026-11-27 ~16:15 EST, before the first GPU run)

Test `decode_graph_recorded_without_drafting_is_not_replayed_by_a_drafting_state` (`crates/intarsia-models/tests/inkling2_dflash2.rs`).
It runs on the tiny fixture, for DFlash2 and for DSpark:

1. One state does not draft: prefill, then two plain steps (record, replay).
2. `truncate(0)`, then the same state drafts: prefill, one plain step.
3. Its taps must equal bitwise those of a fresh state that drafts on the same model (its own recording). They must also be within
   1e-2 of an eager model's.
4. It asserts that the decode graph was replayed (>= 2), so the test cannot pass vacuously on a backend with graphs.

| # | Rule |
|---|---|
| T1 | HIP, test-only build (red, `7fd1808e7` = 418c122bd's code plus the test): the test **FAILS** on the tap assertion (not on the vacuous one) |
| T2 | HIP, fix build (green): the test **PASSES**, with graph replays >= 2, for both drafts |
| T3 | HIP, fix build: the whole `inkling2_dflash2` suite passes |
| T4 | cpu (tatooine dev image): the fix build passes `inkling2_dflash2`, `inkling2_dspark`, `inkling2_dspark_sampled`, `inkling2_dspark_sampled_adapt`, `inkling2_sampled_default`, and the intarsia-core lib tests |

Merge rule: T1-T4 hold → PR, merge on green CI. If T1 passes on red, the bug as described is not real. In that case report it and
merge nothing.

Not measured: the real-model speed effect (acceptance of a drafting request after one that did not draft). That needs a longer
radiance window.

## Results

(pending)
