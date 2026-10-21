#!/usr/bin/env bash
# A2 end-to-end of the per-prompt noise check on the CPU backend, Qwen3-Reranker-0.6B (decision 2026-10-21 13:58).
# Run inside a fleetq slot. Output: $E/*.log, $E/*.json; prints one summary line per case.
set -u
cd /opt/code/intarsiaml/.claude/worktrees/agent-a8eb89d1ece4a7d80
G=/opt/code/intarsiaml/golden
E=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/e2e3
RR=/Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473
BIN=./target/debug/intarsia
GOLD=$G/qwen3-reranker-0.6b.safetensors
rm -rf $E; mkdir -p $E
SELF="--allow-unstamped --allow-same-code --allow-dirty"
run() { # name, env, args...
  local name=$1 envs=$2; shift 2
  env $envs $BIN accuracy "$@" > $E/$name.log 2>&1; echo "[$name] exit=$?"
  grep -E "KL threshold|^Error" $E/$name.log | cut -c1-900
}

# 1. a record with per-prompt values, measured offline: bf16 golden against the fp32 golden of the same prompts
$BIN ref-noise --golden $GOLD --other $G/qwen3-reranker-0.6b-fp32.safetensors --method precision_spread --how "e2e: bf16 golden against fp32" --out $E/rec.json > $E/00-rec.log 2>&1; echo "[ref-noise] exit=$?"
python3 - <<EOF
import json
d = json.load(open("$E/rec.json"))
print("[record] prompts:", len(d["prompts"]), "mean", d["kl21_mean"], "per-prompt", {k: round(v["kl21_mean"], 5) for k, v in d["prompts"].items()})
d.pop("prompts"); json.dump(d, open("$E/rec-noprompts.json", "w"))
p = json.load(open("$E/rec.json")); p["prompts"] = {k: v for k, v in list(p["prompts"].items())[:2]}; json.dump(p, open("$E/rec-two.json", "w"))
EOF

# 2. a self-golden that carries the record (per-prompt values included)
run write-self "" --model $RR --backend cpu --golden $GOLD --mode prefill --ref-noise $E/rec.json --write-self-golden $E/S.safetensors --allow-unstamped --allow-dirty
python3 - <<EOF
print("[self-golden] metadata carries ref_noise with prompts:", '"prompts"' in open("$E/S.safetensors","rb").read().decode("latin1"))
EOF

SG="--model $RR --backend cpu --mode prefill --self-golden $E/S.safetensors --self-vs-ref-noise $SELF"
# 3. clean build against itself: every prompt on its own noise, PASS
run clean "" $SG --write-verdict $E/v-clean.json
# 4. faulted build, mean over the noise and some prompts over their own limits: FAIL, with the basis named per prompt
run fault-1e-2 "INTARSIA_CPU_FAULT=matmul:1e-2" $SG --write-verdict $E/v-fault2.json
# 5. mild fault, prompt factor tight: per-prompt FAIL naming the own-noise basis
run fault-1e-3-tight "INTARSIA_CPU_FAULT=matmul:1e-3" $SG --ref-noise-prompt-factor 1.0 --write-verdict $E/v-tight.json
# 6. the same run with a record that has no per-prompt values: fallback to the mean noise, basis named "mean noise"
run fault-1e-3-tight-mean "INTARSIA_CPU_FAULT=matmul:1e-3" $SG --ref-noise $E/rec-noprompts.json --ref-noise-prompt-factor 1.0 --write-verdict $E/v-tight-mean.json
# 7. a record with only two prompts' values: the rest fall back to the mean
run two-prompts "INTARSIA_CPU_FAULT=matmul:1e-3" $SG --ref-noise $E/rec-two.json --ref-noise-prompt-factor 1.0 --write-verdict $E/v-two.json

python3 - <<EOF
import json
for n in ["clean", "fault2", "tight", "tight-mean", "two"]:
    v = json.load(open("$E/v-%s.json" % n))["verdicts"][0]
    pc = v["prompt_checks"]
    bases = {}
    for c in pc["checks"]:
        bases[c["basis"]] = bases.get(c["basis"], 0) + 1
    fails = [(c["prompt"], c["basis"], round(c["factor_needed"], 2)) for c in pc["checks"] if not c["pass"]]
    print("[verdict-json %s] pass=%s mean-kl=%.3e prompts=%d bases=%s failing=%s" % (n, v["pass"], v["kl"], len(pc["checks"]), bases, fails))
EOF
