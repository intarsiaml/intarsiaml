"""Client-path check: per real-prompt / bitwise client run, the share of inter-token gaps under 5 ms (a decode step takes >= ~10 ms,
so a gap under 5 ms is two tokens coalesced after the scheduler emitted them) and the 'late pairs' (a gap > 0.2 s followed by one
under 10 ms in the same stream). python3 coalesce.py <json>..."""
import json, sys
for f in sys.argv[1:]:
    if "meta" in f: continue
    d = json.load(open(f)); g = [r.get("gaps") or [] for r in d.values()]
    flat = [x for s in g for x in s]
    if not flat: continue
    sub5 = sum(1 for x in flat if x < 0.005)
    pairs = sum(1 for s in g for j in range(len(s) - 1) if s[j] > 0.2 and s[j + 1] < 0.01)
    print(f"{f:50s} gaps {len(flat):5d}  <5 ms {sub5:4d} ({100 * sub5 / len(flat):4.1f} %)  late pairs {pairs:3d}")
