"""Depth-1 fine-tune of the MTP head on dumps of target-generated text.

Frozen: the main model (only its dumped hidden rows are used), embed_tokens, lm_head. Trained: the 15 mtp.* tensors (~425M
parameters), fp32 master weights, bf16 compute (each op casts its weight to bf16, as the engine's bf16 head does), AdamW.
Loss: cross-entropy of lm_head(head(...)) at row p against tokens[p+2], over the generated-response rows only (regen.py's
`--meta` sidecar gives prompt_len; target rows inside the prompt are user text, not the target model's argmax). Logits are
computed and back-propagated in row chunks (`--chunk`), so the 248K-wide logits never exist for a whole batch.
The held-out split is by file id (md5 of the id), and held-out acceptance (argmax == tokens[p+2], full vocab and draft
vocab, response rows) is logged every `--eval-every` steps, starting with step 0 = the stock head.

  python train.py --ckpt <model dir> --dumps <dir> --meta meta.json --out <dir> [--epochs 1 --lr 2e-5 ...]
Writes <out>/mtp-tuned.safetensors (best held-out acceptance, or the last step if there was no eval), <out>/mtp-last.safetensors,
<out>/train-log.jsonl, <out>/split.json. Names, dtypes (bf16) and shapes are asserted against the checkpoint's headers.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

import accept as A
import head as H


def split_ids(files, pct, min_heldout=1):
    key = lambda f: int(hashlib.md5(H.file_id(f).encode()).hexdigest(), 16)
    held = [f for f in files if key(f) % 100 < pct]
    if len(held) < min_heldout and len(files) > min_heldout:
        held = sorted(files, key=key)[:min_heldout]
    hs = set(held)
    return [f for f in files if f not in hs], held


def n_tokens(path):
    with safe_open(path, "pt") as f:
        return f.get_slice("tokens").get_shape()[0]


def make_batches(files, meta, budget, max_len, rng):
    """Token-budget batches of similar-length files (padded batch: B * longest <= budget). -> list of file lists."""
    items = [(f, min(max(n_tokens(f) - 2, 0), max_len)) for f in files]
    items = [(f, L) for f, L in items if L > 0]
    rng.shuffle(items)
    batches = []
    for i in range(0, len(items), 128):
        blk = sorted(items[i : i + 128], key=lambda x: x[1])
        cur, mx = [], 0
        for f, L in blk:
            if cur and max(mx, L) * (len(cur) + 1) > budget:
                batches.append(cur)
                cur, mx = [], 0
            cur.append(f)
            mx = max(mx, L)
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


class BatchSet(torch.utils.data.Dataset):
    def __init__(self, batches, meta, max_len, loss_vocab):
        self.batches, self.meta, self.max_len, self.loss_vocab = batches, meta, max_len, loss_vocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        rows = []
        for f in self.batches[i]:
            d = H.read_dump(f)
            n = d["tokens"].shape[0]
            L = min(n - 2, self.max_len)
            pl = self.meta.get(H.file_id(f), {}).get("prompt_len") or 0
            tgt = d["tokens"][2 : 2 + L].clone()
            tgt[: max(0, pl - 2)] = -100  # row p predicts tokens[p+2]: only response targets train
            if self.loss_vocab:
                tgt[tgt >= self.loss_vocab] = -100  # a token the draft vocab cannot propose
            rows.append((d["hidden"][:L], d["tokens"][1 : 1 + L], tgt))
        T = max(r[0].shape[0] for r in rows)
        B = len(rows)
        hid = torch.zeros(B, T, rows[0][0].shape[1], dtype=torch.bfloat16)
        nxt = torch.zeros(B, T, dtype=torch.long)
        tg = torch.full((B, T), -100, dtype=torch.long)
        for b, (h, t1, t2) in enumerate(rows):
            L = h.shape[0]
            hid[b, :L], nxt[b, :L], tg[b, :L] = h.to(torch.bfloat16), t1, t2
        return hid, nxt, tg


def lr_at(step, total, base, warmup, floor=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def micro_step(head, hid, nxt, tg, chunk, loss_vocab, scale):
    """Forward + backward of one padded micro-batch; returns (sum loss, rows, correct rows)."""
    g = head(hid, nxt)
    sel = tg != -100
    n = int(sel.sum())
    if n == 0:
        return 0.0, 0, 0
    gsel = g[sel]
    tsel = tg[sel]
    gd = gsel.detach().requires_grad_()
    total, correct = 0.0, 0
    for i in range(0, n, chunk):
        lg = head.logits(gd[i : i + chunk], loss_vocab or None)
        t = tsel[i : i + chunk]
        loss = F.cross_entropy(lg, t, reduction="sum")
        (loss * (scale / n)).backward()
        total += float(loss.detach())
        correct += int((lg.argmax(-1) == t).sum())
    gsel.backward(gd.grad)
    return total, n, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lm-head", default="nvfp4")
    ap.add_argument("--init", default=None, help="start from these mtp.* tensors instead of the checkpoint's")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="optimizer steps (overrides --epochs)")
    ap.add_argument("--tokens-per-batch", type=int, default=8192, help="padded rows per micro-batch")
    ap.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step")
    ap.add_argument("--max-len", type=int, default=4096, help="rows per sequence (a prefix; positions stay absolute)")
    ap.add_argument("--chunk", type=int, default=512, help="rows per logits chunk")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--heldout-pct", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-files", type=int, default=64)
    ap.add_argument("--loss-vocab", choices=["full", "draft"], default="full",
                    help="draft: softmax over the first draft_vocab lm_head rows only, rows whose target lies outside are skipped")
    ap.add_argument("--draft-vocab", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    meta = H.load_meta(a.meta)
    files = H.dump_files(a.dumps)
    train_f, held_f = split_ids(files, a.heldout_pct)
    if not train_f:
        raise SystemExit("no training files after the split")
    json.dump(dict(train=[H.file_id(f) for f in train_f], heldout=[H.file_id(f) for f in held_f]), open(os.path.join(a.out, "split.json"), "w"))
    dv = a.draft_vocab if a.draft_vocab is not None else H.draft_vocab_of(H.read_dump(files[0]), None)
    loss_vocab = dv if (a.loss_vocab == "draft" and dv) else 0
    print(f"{len(train_f)} train / {len(held_f)} held-out files, draft_vocab {dv}, loss over {'first %d rows' % loss_vocab if loss_vocab else 'the full vocab'}", flush=True)

    head = H.load_head(a.ckpt, a.device, a.lm_head, a.init, param_dtype=torch.float32)
    params = list(head.named_mtp().values())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)

    def run_eval(step):
        head.eval()
        ov, cats = A.evaluate(head, held_f, meta, a.device, dv or None, max_files=a.eval_files)
        head.train()
        r = ov.row()
        r["step"] = step
        r["categories"] = {c: t.row() for c, t in cats.items()}
        return r

    log = open(os.path.join(a.out, "train-log.jsonl"), "a")
    emit = lambda r: (print(json.dumps(r), flush=True), log.write(json.dumps(r) + "\n"), log.flush())
    best = -1.0
    best_state = None
    metric = lambda r: r.get("draft_vocab", r["full"]) if dv else r["full"]

    def keep_best(r):
        nonlocal best, best_state
        if metric(r) > best:
            best = metric(r)
            best_state = {n: p.detach().to(torch.bfloat16).cpu() for n, p in head.named_mtp().items()}
            return True
        return False

    if held_f and a.eval_every:
        r = run_eval(0)
        r["event"] = "eval"
        emit(r)
        keep_best(r)  # step 0 = the stock head (or --init)

    # one epoch's batches are built once per epoch; the optimizer counts micro-batches / accum
    epochs = max(1, math.ceil(a.epochs))
    all_batches = []
    for e in range(epochs):
        b = make_batches(train_f, meta, a.tokens_per_batch, min(a.max_len, a.tokens_per_batch), rng)
        if a.epochs < e + 1:
            b = b[: int(len(b) * (a.epochs - e))]
        all_batches += b
    total_steps = a.max_steps or max(1, len(all_batches) // a.accum)
    ds = BatchSet(all_batches[: total_steps * a.accum], meta, min(a.max_len, a.tokens_per_batch), loss_vocab)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=a.workers, prefetch_factor=2 if a.workers else None)
    print(f"{len(ds)} micro-batches -> {total_steps} optimizer steps", flush=True)

    head.train()
    t0 = time.time()
    step, micro = 0, 0
    acc = [0.0, 0, 0, 0]  # loss sum, rows, correct, padded tokens
    for hid, nxt, tg in dl:
        hid, nxt, tg = hid.to(a.device), nxt.to(a.device), tg.to(a.device)
        l, n, c = micro_step(head, hid, nxt, tg, a.chunk, loss_vocab, 1.0 / a.accum)
        acc[0] += l; acc[1] += n; acc[2] += c; acc[3] += hid.shape[0] * hid.shape[1]
        micro += 1
        if micro % a.accum:
            continue
        for g_ in opt.param_groups:
            g_["lr"] = lr_at(step, total_steps, a.lr, a.warmup)
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip))
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == total_steps:
            emit(dict(event="train", step=step, loss=acc[0] / max(acc[1], 1), acc=acc[2] / max(acc[1], 1), rows=acc[1], grad_norm=gn,
                      lr=opt.param_groups[0]["lr"], tok_s=acc[3] / (time.time() - t0), mem_gb=torch.cuda.max_memory_allocated() / 2**30 if a.device.startswith("cuda") else 0))
            acc = [0.0, 0, 0, 0]; t0 = time.time()
        if held_f and a.eval_every and (step % a.eval_every == 0 or step == total_steps):
            r = run_eval(step)
            r["event"] = "eval"
            r["best"] = keep_best(r)
            emit(r)
            t0 = time.time()
        if step >= total_steps:
            break

    last = os.path.join(a.out, "mtp-last.safetensors")
    H.save_mtp(head, last, a.ckpt)
    tuned = os.path.join(a.out, "mtp-tuned.safetensors")
    if best_state is not None and best >= 0:
        from safetensors.torch import save_file
        # same assertion as save_mtp, on the stored best state
        ck = H.Checkpoint(a.ckpt)
        for n_, t in best_state.items():
            h = ck.header(n_)
            assert h["dtype"] == "BF16" and list(t.shape) == h["shape"], n_
        save_file({n_: t.contiguous() for n_, t in best_state.items()}, tuned, metadata={"format": "pt"})
    else:
        H.save_mtp(head, tuned, a.ckpt)
    print(f"saved {tuned} (best held-out {'%.4f' % best if best >= 0 else 'n/a'}) and {last}", flush=True)


if __name__ == "__main__":
    main()
