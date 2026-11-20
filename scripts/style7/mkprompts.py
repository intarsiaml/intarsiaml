import json, sys
from localeval import gsm8k, quality
quality._patch_hf_ids()
b = gsm8k.make_benchmark(3, 200, pinned=True)
goldens = b.load_benchmark_dataset()[:200]
tpl = b._template()
print(len(goldens), b.template_sha256, file=sys.stderr)
out = [{"id": f"gsm8k-{i:05d}", "prompt": gsm8k.fill_template(tpl, g.input), "gold": g.expected_output} for i, g in enumerate(goldens)]
json.dump(out, open("prompts200.json", "w"))
