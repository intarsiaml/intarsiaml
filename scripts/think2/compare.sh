#!/usr/bin/env bash
# think2: localeval compare for the three pairs of one set. usage: compare.sh <bbh|gsm8k> <A1 run id> <B run id> <A2 run id>
R="$(cd "$(dirname "$0")/../.." && pwd)/docs/results/2026-11-21-think2/raw"; s=$1
cd /opt/code/deepeval || exit 1
uv run localeval compare "$2" "$4" > "$R/compare-$s-A1-vs-A2.txt" 2>&1
uv run localeval compare "$2" "$3" > "$R/compare-$s-A1-vs-B.txt" 2>&1
uv run localeval compare "$4" "$3" > "$R/compare-$s-A2-vs-B.txt" 2>&1
