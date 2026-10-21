"""Rewrites golden/ref_noise/*.json (the three back-filled reference-noise records of lane A2) from the numbers in docs/results/2026-10-21-cuda-round7/raw/05-kl-table.txt and docs/results/2026-10-21-m1-qwen38-drill.md. Run from anywhere; uv run --no-project python scripts/a2-ref-noise/make_backfill_sidecars.py"""
import json, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.makedirs("golden/ref_noise", exist_ok=True)
doc = "docs/results/2026-10-21-cuda-round7.md"
common = (
    "vLLM `guidance` (0.26.1rc1.dev602+g65b7662d3, the build the golden records, fp8 KV cache, prefix caching on) re-scored the exact ids of the stored golden "
    "(`tools/golden_vllm.py --ids-from`, `scripts/cuda-round7/vllm-self.sh`); KL21(stored golden || re-run) by `scripts/m3-accuracy/setkl.py` "
    "(`scripts/cuda-round7/kl-table.sh`): both sides truncated to top-20, a reference id missing from the re-run's top-20 gets the re-run's 20th probability, "
    "the scored value drops the ChatML system and user-turn rows. "
)
est = "kl21-top20-vs-top20 (scripts/m3-accuracy/split.py rule); on our own rows it agrees with the tool-exact value to about 1e-4 relative (M3: 1.4198e-2 tool, 1.420e-2 setkl)"
backfill = "backfilled by the reference-noise gate lane (decision 2026-10-21 11:23 item 3, docs/decisions/LOG.md)"
short = {
    "format": "intarsia-ref-noise-v1",
    "golden": {"file": "qwen36-35b-a3b-nvfp4-vllm-top20.safetensors", "fnv1a64": "cd6ef94b3dcd4c82", "size": 246804, "positions": 1441},
    "method": "run_to_run", "kl21_mean": 3.161e-3, "kl21_mean_scored": 3.010e-3, "n": 1441, "n_scored": 1279, "runs": 2, "date": "2026-10-21",
    "how": common + "M3 measured two more re-runs of the same golden: 2.5e-3 and 2.9e-3 on all rows (docs/results/2026-10-21-m3-accuracy-debt.md 2.4), so the value moves between 2.5e-3 and 3.2e-3 from one re-run to the next.",
    "estimator": est,
    "source": doc + " 3.1-3.2, raw/05-kl-table.txt (row 'vLLM re-run'); " + backfill,
}
long = {
    "format": "intarsia-ref-noise-v1",
    "golden": {"file": "qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors", "fnv1a64": "1c62c8e49c68fe34", "size": 6061444, "positions": 36887},
    "method": "run_to_run", "kl21_mean": 2.218e-2, "kl21_mean_scored": 8.417e-3, "n": 36887, "n_scored": 32349, "runs": 2, "date": "2026-10-21",
    "how": common + "M3 measured two more re-runs: 2.37e-2 and 2.66e-2 on all rows, 8.6e-3 and 8.7e-3 on the scored rows (docs/results/2026-10-21-m3-accuracy-debt.md 2.4). The all-rows value is dominated by chat_reuse_4k (0.118), the prompt vLLM itself does not reproduce.",
    "estimator": est,
    "source": doc + " 3.1-3.2, raw/05-kl-table.txt (row 'vLLM re-run'); " + backfill,
}
m1 = {
    "format": "intarsia-ref-noise-v1",
    "golden": {"positions": 1441},
    "method": "precision_spread", "kl21_mean": 0.197, "n": 1441, "runs": 2, "date": "2026-10-21",
    "how": (
        "Qwen3.8-27B (AMD Quark MXFP4, declared mode W4A4: MXFP4 weights and dynamically fp4-rounded activations), the HF reference at bf16 against the same HF reference at f32, "
        "teacher-forced over the 10 golden prompts (`golden_kl.py`, the same 21-way KL as the gate). The median is 7.6e-3: the mean is a tail of near-tie positions where rounding "
        "the activations to fp4 turns last-bit differences into whole e2m1 steps. The row set is not recorded and is taken as all rows (the golden has 4 chat prompts, so the "
        "scored rows are a subset). The golden file's name and hash were not available when this record was filled in: it is matched by the sidecar's name and the position count only."
    ),
    "estimator": "kl21 (golden_kl.py, the gate's 21-way KL; 0.176 there against 0.180 from the gate for the same pair)",
    "source": "docs/results/2026-10-21-m1-qwen38-drill.md 5 (row 'HF W4A4 bf16 vs HF W4A4 f32') and 10; backfilled by the reference-noise gate lane (decision 2026-10-21 12:39 item 2, docs/decisions/LOG.md)",
}
for name, d in [
    ("qwen36-35b-a3b-nvfp4-vllm-top20.safetensors", short),
    ("qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors", long),
    ("qwen3.8-27b-mxfp4-hfquark-top20.safetensors", m1),
]:
    with open(f"golden/ref_noise/{name}.json", "w") as f:
        f.write(json.dumps(d, indent=2) + "\n")
print("ok")
