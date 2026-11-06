#!/bin/zsh
# Lane lane-ds, endor, inside a fleetq --exclusive slot: Metal no-op of the device-sampling default. main binary (.scratch/intarsia.main) against this branch's
# (.scratch/intarsia.ds): plan show, kernels test, 256 greedy ids (8B short and 4k prompt, 35B short), goldens (accuracy --gate-class bf16).
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/device-sampling || exit 2
S=$PWD/.scratch; O=$PWD/docs/results/2026-11-05-device-sampling-default/raw/endor; mkdir -p $O
B=$S/intarsia.main; F=$S/intarsia.ds;
[ -x $B -a -x $F ] || { echo "missing $B or $F (the build job did not finish)"; exit 3; }
 W=/Users/josef/models; G=/opt/code/intarsiaml/golden
P4=docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-4096.txt
echo "== plan show"
for mdl in Qwen3.6-35B-A3B-NVFP4 Qwen3-8B; do for fmt in "" "--json"; do
  $B plan show --model $W/$mdl --backend metal $fmt > $O/plan-$mdl.main$fmt.txt 2>&1
  $F plan show --model $W/$mdl --backend metal $fmt > $O/plan-$mdl.ds$fmt.txt 2>&1
  if cmp -s $O/plan-$mdl.main$fmt.txt $O/plan-$mdl.ds$fmt.txt; then echo "plan show $mdl $fmt: identical"; else echo "plan show $mdl $fmt: DIFFERS"; fi
done; done
echo "== kernels test"
$B kernels test --backend metal > $O/kernels-main.log 2>&1; echo "main exit $?"
$F kernels test --backend metal > $O/kernels-ds.log 2>&1; echo "ds exit $?"
tail -2 $O/kernels-ds.log | cut -c1-160; echo "FAIL lines ds: $(grep -c FAIL $O/kernels-ds.log)"
echo "== greedy ids"
gen() { # gen <bin> <tag> <model> <prompt-text> <name>
  $1 generate --model $W/$3 --backend metal -p "$4" --max-tokens 256 --ignore-eos --temperature 0 --show-ids > $O/ids-$5.$2.out 2> $O/ids-$5.$2.err
  grep -a '^ids' $O/ids-$5.$2.err | md5 | cut -c1-12
}
for spec in "Qwen3-8B|short|The history of the printing press begins" "Qwen3-8B|4k|$(cat $P4)" "Qwen3.6-35B-A3B-NVFP4|short|The history of the printing press begins"; do
  mdl=${spec%%|*}; r=${spec#*|}; nm=${r%%|*}; pr=${r#*|}
  a=$(gen $B main $mdl "$pr" $mdl-$nm); b=$(gen $F ds $mdl "$pr" $mdl-$nm)
  n=$(grep -a -c '^ids' $O/ids-$mdl-$nm.ds.err); n2=$(grep -a -c '^ids' $O/ids-$mdl-$nm.main.err)
  [ "$n" -ge 1 -a "$n2" -ge 1 -a "$a" = "$b" ] && echo "ids $mdl $nm: identical ($a)" || echo "ids $mdl $nm: DIFFERS ($a vs $b)"
done
echo "== goldens"
for g in qwen3-8b-fp32 qwen3-8b-long-fp32; do
  for t in main ds; do bin=$B; [ $t = ds ] && bin=$F
    $bin accuracy --model $W/Qwen3-8B --backend metal --golden $G/$g.safetensors --gate-class bf16 --max-seq 8192 > $O/golden-$g.$t.txt 2>&1; echo "golden $g $t exit $?"
  done
  grep -aq PASS $O/golden-$g.main.txt && grep -aq PASS $O/golden-$g.ds.txt && diff <(grep -aE 'KL|top-1|PASS|FAIL' $O/golden-$g.main.txt) <(grep -aE 'KL|top-1|PASS|FAIL' $O/golden-$g.ds.txt) > /dev/null && echo "golden $g: PASS, same figures" || echo "golden $g: NOT PASS or figures DIFFER"
done
