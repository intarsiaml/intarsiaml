#!/bin/sh
# usage: plan_baseline.sh <binary> <outdir>
B=$1; O=$2; mkdir -p $O
$B plan show --model /Users/josef/models/Qwen3-8B --backend metal > $O/plan-qwen3-8b.txt 2>&1
$B plan show --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --backend metal > $O/plan-qwen3_5-4b.txt 2>&1
