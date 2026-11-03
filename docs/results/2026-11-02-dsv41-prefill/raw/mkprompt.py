import sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/opt/code/intarsiaml/.worktrees/dsv41-official/mia/tokenizer.json")
text = "".join(open(f).read() + "\n" for f in sys.argv[3:])
target = int(sys.argv[1])
ids = tok.encode(text, add_special_tokens=False).ids
assert len(ids) >= target, len(ids)
# cut at a character boundary giving exactly target tokens when re-encoded
lo, hi = 0, len(text)
while lo < hi:
    mid = (lo + hi) // 2
    n = len(tok.encode(text[:mid], add_special_tokens=False).ids)
    if n < target: lo = mid + 1
    else: hi = mid
t = text[:lo]
print(len(tok.encode(t, add_special_tokens=True).ids), file=sys.stderr)
open(sys.argv[2], "w").write(t)
