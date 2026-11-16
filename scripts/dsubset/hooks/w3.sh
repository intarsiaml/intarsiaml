# w3 slot (spark1), before the 27B arms: collection round 2 (prompts.COLLECT2) on the 35B with the lane binary, --max-batch 16
M35_SAVE=$M35
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
if start_serve collect2 "X=1" --max-batch 16; then
  COLLECT_SET=COLLECT2 COLLECT_SEED0=9000 python3 "$S/collect.py" "$URL" guidance "$REPO/docs/results/2026-11-15-dsubset/raw/collect2.jsonl" --conc 16 --max-tokens 1024 2>&1 | tee -a "$LOG"
  finish_serve collect2
fi
M35=$M35_SAVE
