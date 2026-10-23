# Gemma-4 slice: real weights, small memory (lane g4)

`build_slice.py` cuts layers 0 (sliding) and 5 (global) of nvidia/Gemma-4-26B-A4B-NVFP4 into a 2-layer checkpoint in the
engine's format (NVFP4 experts) and an HF-loadable bf16 one (experts dequantised). `fnscale.py` scales the final norm by 2^-5
(the real final norm makes 2 layers' logits saturate at the soft-cap, which makes any comparison vacuous). `verify_hf.py` loads
the bf16 slice with transformers 5.6.2. Then: `tools/golden.py --model <bf16 slice> --topk 20 --template hf`, and
`intarsia accuracy --model <nvfp4 slice> --golden ...` / `intarsia onboard --golden-from ...`. Paths are hard-coded at the top of
each script (they ran on spark2); edit them. See docs/results/2026-10-22-g4-gemma4-onboard.md.
