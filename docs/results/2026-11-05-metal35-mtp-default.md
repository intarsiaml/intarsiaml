# Metal 35B: MTP (k=2) on by default

Lane `lane-fm/metal35-mtp-default`, off main `602a1fcfe`. Qwen3.6-35B-A3B NVFP4 on Metal (endor, M5 Max), single sequence, `--spec mtp --spec-k 2`.

## Approval

User, 2026-11-05 17:06 EST: "flip 1" (relayed by the coordinator): make MTP speculation (k=2) the default for the Qwen3.6-35B-A3B NVFP4 MoE on Metal. The gate set below is the coordinator's brief; a greedy-id difference against `--spec off` stops the merge and goes back to the user.

(Results are filled in below as the windows complete; nothing under "Results" is a measurement until it says so.)
