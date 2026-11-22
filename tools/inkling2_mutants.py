#!/usr/bin/env python3
"""Mutation check of the tiny G1(b) gate (golden/inkling2-tiny): perturb one config key, run the oracle
teacher-forced at bf16 rounding, count gate failures. The control must give 0, every listed mutant > 0
(eps 1e-5 is below bf16 resolution and is expected to pass).

    uv run --project tools python tools/inkling2_mutants.py
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inkling2_ref as R  # noqa: E402

FIX = HERE.parent / "golden" / "inkling2-tiny"
WORK = HERE.parent / ".scratch" / "mutant"

MUTANTS = [
    ("control", {}),
    ("no q/k clip", {"qk_l2_clip_threshold": 1e9}),
    ("qk clip 3.3", {"qk_l2_clip_threshold": 3.3}),
    ("no rel clip", {"rel_l2_clip_threshold": 1e9}),
    ("up softcap off", {"mlp_up_softcap": 1e9}),
    ("gate softcap off", {"mlp_gate_softcap": 1e9}),
    ("final softcap off", {"final_logit_softcapping": None}),
    ("tau off", {"log_scaling_n_floor": None}),
    ("route_scale = topk", {"route_scale": 4.0}),
    ("window 9", {"sliding_window_size": 9}),
    ("eps 1e-5", {"rms_norm_eps": 1e-5}),
]


def main() -> int:
    ids = load_file(str(FIX / "ref_logits.safetensors"))["prompt_ids"]
    f64 = load_file(str(FIX / "ref_dump_f64.safetensors"))
    otf = load_file(str(FIX / "ref_dump_bf16_tf.safetensors"))
    bad = 0
    for name, edit in MUTANTS:
        shutil.rmtree(WORK, ignore_errors=True)
        shutil.copytree(FIX, WORK)
        cfg = json.loads((WORK / "config.json").read_text())
        cfg["text_config"].update(edit)
        (WORK / "config.json").write_text(json.dumps(cfg))
        test = R.build(WORK, torch.float32, "vllm").teacher_forced(ids, f64)
        fails = R.tf_check(test, f64, otf)
        expect_pass = name in ("control", "eps 1e-5")
        ok = (not fails) == expect_pass
        bad += not ok
        print(f"{name:20s} failures {len(fails):4d} {'ok' if ok else 'UNEXPECTED'} {fails[:1]}")
    shutil.rmtree(WORK, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
