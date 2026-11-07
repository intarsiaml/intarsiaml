# mtp-ft: fine-tune the MTP draft head on target-generated text

PyTorch half of the pipeline. The engine half is `intarsia mtp-dump`. Data and checkpoints never go in git (scratch: `~/mtp-ft/data` on spark1).

```
prompts.jsonl --regen.py--> tokens.jsonl + meta.json --intarsia mtp-dump--> dumps/<id>.safetensors
   --train.py--> mtp-tuned.safetensors --make_override.sh--> model dir with the tuned head --> engine (accept / tok/s)
```

Files: `head.py` (the head, 15 `mtp.*` tensors + frozen embed / lm_head; conventions with engine file:line in its docstring), `parity.py`,
`accept.py`, `train.py`, `regen.py`, `prompts.py`, `make_override.sh`, `synth.py` (tiny checkpoint + synthetic dumps), `selftest.py` (CPU, f64 loop-by-loop port of the engine's reference ops), `smoke.sh`.

## 0. Build the prompt set (UltraChat first user turns + Magicoder problems, deduped, eval prompts removed)
```
python prompts.py --ultrachat ultrachat_train_sft_0.parquet --magicoder magicoder_oss_instruct.jsonl --eval <dir with the eval category jsonl> --out prompts.jsonl --n 40000 --chat-frac 0.6
```
Needs pyarrow (in the pytorch image). Drops eval overlap (exact, containment, word 5-gram Jaccard >= 0.6), exact duplicates and too short / long prompts.

## 1. Regenerate prompts greedily (the target's own argmax is the training label)
```
python regen.py --ckpt <model dir> --prompts prompts.jsonl --endpoint http://HOST:PORT --out-dir regen/ --max-tokens 1024 --concurrency 16
```
Prompt lines: `{"messages":[...]}` or `{"input":"...","category":"prose"}`. Prompts equal to an eval prompt (`scripts/dfh/cats/*.jsonl`) are dropped. Ids are `<category>__<n>`.
The endpoint must serve this exact model (the script logs `/v1/models`). Outputs: `regen/tokens.jsonl` (`{"id","tokens"}`, what mtp-dump takes) and `regen/meta.json` (`prompt_len` per id; train / accept mask the prompt rows with it).
`--dry-run` renders, tokenises and applies the exclusion without requests; `--selftest` round-trips a stub server.

## 2. Dump (engine)
`intarsia mtp-dump --input regen/tokens.jsonl ...` -> `dumps/<id>.safetensors` (`tokens`, `hidden`, `head_top1`, optional `head_top1_dv`, `head_out`; metadata `hidden_size`, `vocab_size`, `draft_vocab`).

## 3. Parity (gate >= 99 %), then baseline acceptance
```
python parity.py --ckpt <model dir> --dumps dumps/                      # --lm-head must be the head the engine used
python accept.py --ckpt <model dir> --dumps dumps/ --meta regen/meta.json
```
`--lm-head nvfp4` (default) dequantises the checkpoint's NVFP4 `lm_head` (CUDA engine); `bf16:<file>[+fp8]` takes a bf16 head and, with `+fp8`, emulates `lm_head_quant = "fp8"` (HIP / Metal MXFP4 checkpoint).

## 4. Train (head only, held-out split by file id, held-out acceptance logged every `--eval-every` steps; step 0 is the stock head)
```
python train.py --ckpt <model dir> --dumps dumps/ --meta regen/meta.json --out run1/ --epochs 1 --lr 2e-5
```
Writes `run1/mtp-tuned.safetensors` (best held-out acceptance; the stock head if nothing beats it), `mtp-last.safetensors`, `train-log.jsonl`, `split.json`.
Peak GPU memory 15 GB at `--tokens-per-batch 8192` (3.3k rows/s on a GB10). `--loss-vocab draft` trains the softmax over the first `draft_vocab` rows only.

## 5. Serve the tuned head
```
./make_override.sh <model dir> run1/mtp-tuned.safetensors <model dir>-mtp-tuned
python accept.py ... --mtp-file run1/mtp-tuned.safetensors            # offline number on other dumps
```
All files are symlinked, `model.safetensors.index.json` is a copy whose `mtp.*` entries point at `mtp-tuned.safetensors` (it sorts after `model-*`, so the loader's later-file-wins rule picks it).

## Running on spark1 (container has no transformers; regen needs only `tokenizers` + `jinja2`)
Every GPU command goes through fleetq; mount `/mnt/models` (the `~/models` entries are symlinks into it):
```
rsync -a --exclude __pycache__ tools/mtp-ft/ spark1:mtp-ft/tools/
fleetq run --host spark1 --mem 26 --expect 15 --label "mtp-ft smoke" -- ssh spark1 'docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user $(id -u):$(id -g) -e HOME=/tmp -v /mnt/models:/mnt/models:ro -v ~/mtp-ft:/work -w /work/tools nvcr.io/nvidia/pytorch:26.07-py3 bash smoke.sh /mnt/models/Qwen3.8-27B-NVFP4-52d1adc5f38a /work/data/smoke'
```
