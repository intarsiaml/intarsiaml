"""Parity of the PyTorch head against the engine's `intarsia mtp-dump` rows.

For every dump file: run the head over all n-1 rows (teacher forcing) and compare the argmax with the recorded `head_top1`
(full vocab) and, when present, `head_top1_dv` (first `draft_vocab` lm_head rows). Also logit stats (top-1 / top-2 margin, the
share of rows whose margin is below bf16 resolution = ties the engine and PyTorch may order differently) and, when the dump has
`head_out`, the relative error of the head's output rows. Gate: both fractions >= --gate (default 0.99), exit status 1 otherwise.

  python parity.py --ckpt <model dir> --dumps <dir> [--lm-head nvfp4|bf16:<file>[+fp8]] [--mtp-file mtp-tuned.safetensors]
The lm_head must be the one the dump engine used (record its format in the dump's metadata): the NVFP4 head of the NVFP4
checkpoint on CUDA, or a bf16 head with `lm_head_quant = "fp8"` on HIP / Metal (`--lm-head bf16:<file>+fp8`).
"""
import argparse
import json
import sys

import torch

import head as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--lm-head", default="auto")
    ap.add_argument("--mtp-file", default=None, help="tuned mtp.* tensors to use instead of the checkpoint's")
    ap.add_argument("--draft-vocab", type=int, default=None, help="override the dump metadata's draft_vocab")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--gate", type=float, default=0.99)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=None, help="write the summary here")
    a = ap.parse_args()

    head = H.load_head(a.ckpt, a.device, a.lm_head, a.mtp_file)
    files = H.dump_files(a.dumps)
    if a.max_files:
        files = files[: a.max_files]
    if not files:
        sys.exit(f"no *.safetensors in {a.dumps}")
    tot = dict(rows=0, ok=0, rows_dv=0, ok_dv=0, tie=0, out_num=0.0, out_den=0.0)
    margins = []
    per_file = []
    for f in files:
        d = H.read_dump(f)
        dv = H.draft_vocab_of(d, a.draft_vocab)
        g = H.dump_g(head, d, a.device)
        t1, t1dv = head.argmax(g, dv)
        ref = d["head_top1"].to(a.device)
        n = ref.shape[0]
        ok = int((t1 == ref).sum())
        row = dict(file=H.file_id(f), rows=n, match=ok / n)
        tot["rows"] += n
        tot["ok"] += ok
        if "head_top1_dv" in d and dv:
            ok2 = int((t1dv == d["head_top1_dv"].to(a.device)).sum())
            row["match_dv"] = ok2 / n
            tot["rows_dv"] += n
            tot["ok_dv"] += ok2
        # logit stats (first 512 rows per file keeps this cheap)
        lg = head.logits(g[:512])
        top2 = lg.topk(2, dim=-1).values
        mg = (top2[:, 0] - top2[:, 1])
        margins.append(mg.cpu())
        tot["tie"] += int((mg < 0.05).sum())
        if "head_out" in d:
            ho = d["head_out"].to(a.device).float()
            tot["out_num"] += float((g.float() - ho).pow(2).sum())
            tot["out_den"] += float(ho.pow(2).sum())
        per_file.append(row)
    m = torch.cat(margins)
    sample = int(m.numel())
    s = dict(
        files=len(files),
        rows=tot["rows"],
        match_full=tot["ok"] / tot["rows"],
        match_dv=(tot["ok_dv"] / tot["rows_dv"]) if tot["rows_dv"] else None,
        margin_mean=float(m.mean()),
        margin_p05=float(m.quantile(0.05)),
        near_ties_lt_0p05=tot["tie"] / sample,
        head_out_rel_err=(tot["out_num"] / tot["out_den"]) ** 0.5 if tot["out_den"] else None,
        lm_head=a.lm_head,
        mtp_file=a.mtp_file,
    )
    worst = sorted(per_file, key=lambda r: r["match"])[:5]
    print(json.dumps(s, indent=1))
    print("worst files:", json.dumps(worst))
    if a.json:
        json.dump(dict(summary=s, files=per_file), open(a.json, "w"), indent=1)
    gates = [s["match_full"]] + ([s["match_dv"]] if s["match_dv"] is not None else [])
    verdict = all(x >= a.gate for x in gates)
    print(f"PARITY {'PASS' if verdict else 'FAIL'} (gate {a.gate}): full {s['match_full']:.4f}" + (f", draft-vocab {s['match_dv']:.4f}" if s["match_dv"] is not None else ""))
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
