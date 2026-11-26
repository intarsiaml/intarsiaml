#!/usr/bin/env bash
# endor: build pf2e (final defaults), then the gate window, then the speed window. Do not sync while this runs.
cd /opt/code/intarsiaml/.worktrees/ink-hip-pf2
fleetq run --host tatooine-r9700 --load cpu --expect 15 --label "ink-hip-pf2: build pf2e (final) + ink_attn/ink_moe tests" -- ssh -o BatchMode=yes tatooine 'bash ~/intarsiaml-ink-hip-pf2/.scratch/build.sh pf2e ink_attn ink_moe' < /dev/null > .scratch/build-pf2e.log 2>&1 || exit 5
grep -q "cli build exit 0" .scratch/build-pf2e.log && grep -q "test ink_moe build exit 0" .scratch/build-pf2e.log && grep -q "test ink_attn build exit 0" .scratch/build-pf2e.log || exit 6
grep -qE "(492211cf8ca10b20a8a327f2e0b042f2|96cfcc3f3d6db9a3b9305d85c61902c2)  .scratch/bin/intarsia-pf2e" .scratch/build-pf2e.log && exit 7
fleetq run --host tatooine-r9700 --takes-down radiance --quiet gpu,cpu --expect 8 --label "ink-hip-pf2: window 5 gate (md5, G2, admission, hip tests) pf2e" -- ssh -o BatchMode=yes tatooine 'CAP=285 bash ~/intarsiaml-ink-hip-pf2/.scratch/window.sh ~/intarsiaml-ink-hip-pf2/.scratch/stage_gate.sh' < /dev/null > .scratch/w5.log 2>&1 || exit 8
fleetq run --host tatooine-r9700 --takes-down radiance --quiet gpu,cpu --expect 8 --label "ink-hip-pf2: window 6 speed main2 vs pf2e (prefill 4k/16k 4+4, decode short/4k 2+2)" -- ssh -o BatchMode=yes tatooine 'CAP=285 bash ~/intarsiaml-ink-hip-pf2/.scratch/window.sh ~/intarsiaml-ink-hip-pf2/.scratch/stage_speed.sh' < /dev/null > .scratch/w6.log 2>&1
