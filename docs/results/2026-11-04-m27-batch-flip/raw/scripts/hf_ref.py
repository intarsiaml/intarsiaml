#!/usr/bin/env python3
"""HF transformers reference on fixed token sequences: tools/golden_quark_mxfp4.py unchanged, with its prompt loading replaced so it takes ids.
usage: hf_ref.py SEQS.json <args of golden_quark_mxfp4.py>"""
import importlib.util, json, pathlib, sys
ROOT = pathlib.Path("/opt/code/intarsiaml/.worktrees/m27-flip/tools")
sys.path.insert(0, str(ROOT))
import goldenlib as gl
seqs = json.load(open(sys.argv[1]))
gl.load_prompts = lambda paths: [{"name": k, "ids": v} for k, v in seqs.items()]
gl.prompt_ids = lambda p, tokenize, template="chatml", apply_template=None: (p["ids"], "")
gl.reuse_split = lambda p, ids, tokenize: None
gl.meta_prompt = lambda p, text, a, b, split: {"name": p["name"], "n": a}
spec = importlib.util.spec_from_file_location("gq", ROOT / "golden_quark_mxfp4.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.argv = ["golden_quark_mxfp4.py"] + sys.argv[2:]
m.main()
