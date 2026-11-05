"""Self-test of replay-recent-gen.py (the step 7 item set) and of the scorer: generates a small set twice, re-measures every item from
its text alone with the DSv4.1 tokenizer, and checks that the fact is recoverable, the measured distance is in its bucket, the
replay arithmetic puts exactly the last 128 rows in the decoder half, and that two generations are byte-identical.
  uv run --no-project --with tokenizers python scripts/dsv41-tp2/replay-recent-selftest.py
Needs the tokenizer (TOKENIZER, default the Mia file the generator names) and a checkout holding the filler source commit."""
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


gen = load("rgen", "replay-recent-gen.py")
lib = load("rlib", "replay-test-lib.py")
FAILS = []


def check(name, ok, why=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {why}" if not ok and why else ""))
    if not ok:
        FAILS.append(name)


ROOT = os.environ.get("SELFTEST_DIR") or os.path.join(HERE, "..", "..", ".scratch")
os.makedirs(ROOT, exist_ok=True)
T = tempfile.mkdtemp(dir=ROOT, prefix="recent-selftest-")

# ---- the replay arithmetic: the decoder half runs on exactly the last 128 rows, whatever the total and the chunk
bad = []
for total in list(range(1, 700)) + [3711, 3712, 4097, 4132, 4224, 4325, 30244, 30336, 30337, 30436, 30721, 30756]:
    for chunk in (1, 7, 64, 128, 512, 2048):
        if total <= 128:
            continue
        try:
            plan = gen.chunk_plan(total, chunk, total - 20, total - 8)
        except AssertionError:
            bad.append((total, chunk))
            continue
        if plan["decoder_rows"] != gen.WINDOW:
            bad.append((total, chunk, plan["decoder_rows"]))
check("replay arithmetic: decoder rows are exactly [total - 128, total) for every total and chunk tried", not bad, str(bad[:5]))
spans = [t for t in range(129, 6000) if gen.chunk_plan(t, 512, t - 20, t - 8)["window_spans_chunks"]]
check("chunk 512: the replayed window spans two chunks exactly when total mod 512 is 1..127", all(1 <= t % 512 <= 127 for t in spans) and
      all(t in spans for t in range(129, 6000) if 1 <= t % 512 <= 127), "")
check("chunk 512: floor on a chunk boundary is an encoder chunk then a whole chunk",
      gen.chunk_plan(4096 + 128, 512, 4100, 4110)["floor_on_boundary"] and gen.chunk_plan(4096 + 128, 512, 4100, 4110)["floor_kind"] == "whole")

# ---- two generations are identical
d1, d2 = os.path.join(T, "g1"), os.path.join(T, "g2")
for d in (d1, d2):
    gen.generate(d, 2, 1)
sums = {}
for d in (d1, d2):
    sums[d] = {f: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest() for f in sorted(os.listdir(d))}
check("determinism: two generations give byte-identical files", sums[d1] == sums[d2], str(sums))

man = json.load(open(os.path.join(d1, "recent.manifest.json")))
items = man["items"]
check("12 items per length pair of 6 buckets: 2 x 6 at 4k, 1 x 6 at 32k", len(items) == 18 and sum(x["length"] == "4k" for x in items) == 12, str(len(items)))
check("every bucket present at both lengths", all(any(x["length"] == ln and x["bucket"] == b for x in items) for ln in ("4k", "32k") for b in range(1, 7)))

# ---- re-measure from the text alone (independent of the generator's bookkeeping)
check("verify: every row re-measures to its manifest entry", gen.verify(d1) == 0)
tk = gen.Tk()
LO = {1: 0, 2: 65, 3: 129, 4: 257, 5: 1025, 6: 0}
HI = {1: 64, 2: 128, 3: 256, 4: 1024, 5: 4096, 6: 10 ** 9}
rows = {}
for ln in ("4k", "32k"):
    for line in open(os.path.join(d1, f"recent.{ln}.jsonl")):
        r = json.loads(line)
        rows[r["name"]] = r
check("rows carry exactly name / input / expected_output", all(set(r) == {"name", "input", "expected_output"} for r in rows.values()))
for m in items:
    r = rows[m["name"]]
    c, exp = r["input"], r["expected_output"]
    ms = gen.measure(tk, c, {"line": (c.index("Vault " + m["vault"] + " code: "), c.index("Vault " + m["vault"] + " code: ") + len(f"Vault {m['vault']} code: {exp}."))})
    d, dl, tot = ms["line"]["d_first"], ms["line"]["d_last"], ms["total"]
    b = m["bucket"]
    check(f"{m['name']}: fact recoverable by an independent solver, code once in the text, not in the question", gen.solve(c) == exp and c.count(exp) == 1 and exp not in m["question"])
    check(f"{m['name']}: measured d_start {d} in bucket {gen.BUCKETS[b]}", LO[b] <= d <= HI[b] and (b != 5 or d < tot // 2) and (b != 6 or d >= tot // 2), f"total {tot}")
    check(f"{m['name']}: inside the replayed rows iff bucket 1/2; wholly below the floor otherwise; never straddling",
          (d <= 128) == (b in (1, 2)) and (b in (1, 2) or dl >= 129) and not (d > 128 and dl <= 128), f"d {d} lineend {dl}")
    check(f"{m['name']}: manifest total = tokens of the rendered prompt (text + 4 template tokens)", tot == m["total_tokens"] == m["text_tokens"] + 4 and tot == tk.n(tk.render(c)) )
    # the rows of the line are replayed (or not) as the bucket says, at both chunk sizes
    fl = tot - 128
    for ch in (512, 2048):
        cp = m["chunks"][str(ch)]
        check(f"{m['name']}: chunk {ch}: decoder half = last 128 rows; line {'in' if b in (1, 2) else 'not in'} them",
              cp["decoder_rows"] == 128 and (m["line_first_token"] >= fl) == (b in (1, 2)) and (m["line_last_token"] < fl) == (b not in (1, 2)))
    check(f"{m['name']}: first line is unique to the item (no prefix-cache hit)", c.startswith("Run id recent-final-") and c.split("\n")[0] == m["run_id"])
    check(f"{m['name']}: scorer: right on the code, wrong on a decoy, None on empty",
          lib.score_code(exp, exp) == 1 and (not m["decoys"] or lib.score_code(m["decoys"][0]["code"], exp) == 0) and lib.score_code("", exp) is None)
names = [x["name"] for x in items]
check("item names are unique", len(set(names)) == len(names))
check("the first lines are unique", len({r["input"].split("\n")[0] for r in rows.values()}) == len(rows))
check("floor residues mod 512 vary across the items of a bucket", len({x["floor"] % 512 for x in items if x["length"] == "4k" and x["bucket"] == 2}) == 2)
shutil.rmtree(T, ignore_errors=True)
print("SELFTEST FAILED: " + ", ".join(FAILS[:8]) if FAILS else "SELFTEST PASSED")
sys.exit(1 if FAILS else 0)
