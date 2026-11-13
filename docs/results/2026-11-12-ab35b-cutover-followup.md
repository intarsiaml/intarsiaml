# ab35b: cutover follow-up measurements, Qwen3.6-35B-A3B NVFP4, intarsiaml vs production vLLM (spark2)

## User go-ahead

2026-11-12 ~19:50 EST, verbatim: "use fable and go." This approves taking spark2's vLLM (role `retro`) down for up to 90 minutes in total for the three runs below (at most two fleetq jobs with `--takes-down retro --quiet gpu,cpu`; restore on any exit and wait for `/v1/models` 200 before releasing). Lane `ab35b`, branch `ab35b/cutover-followup` from origin/main `2939d30ac`, fleetq owner `claude-ab35b`.

(Results follow once the runs are done.)
