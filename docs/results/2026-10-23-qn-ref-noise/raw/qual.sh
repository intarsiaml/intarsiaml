set -u
export HIP_VISIBLE_DEVICES=0
cd ~/intarsiaml-qn
mkdir -p bench-out/qn
rm -rf bench-out/qn/qual-35b
INTARSIA_REPO=$HOME/intarsiaml-qn INTARSIA_TARGET_VOL=intarsia-cargo-target-qn bash deploy/tatooine/dev.sh bash -c '/target/release/intarsia qualify --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --quick-only --gen 128 --reps 5 --no-unim --out /work/bench-out/qn/qual-35b --mem-budget 29' < /dev/null > bench-out/qn/qual-35b.txt 2>&1
echo "exit $?" >> bench-out/qn/qual-35b.txt
