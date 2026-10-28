#!/usr/bin/env python3
"""Verdict of the ids18 stage (scripts/ar/stages-soak.sh): per prompt, the majority of the ref-<prompt>-<r>.ids files (INTARSIA_HIP_AR_DRAIN=3,
main's behaviour) against every def-<prompt>-<r>.ids (the new default). Usage: ids18_verdict.py <dir> [<dir of the def-* files, default the same>]; exit 1 on any mismatch or missing file."""
import collections, glob, os, sys

d = sys.argv[1]
dd = sys.argv[2] if len(sys.argv) > 2 else d
bad = 0
for kp in ("short", "4k", "16k"):
    for i in range(6):
        n = f"{kp}-{i}"
        def rd(f):
            try:
                return open(f).read().strip()
            except OSError:
                return None
        refs = [rd(f) for f in sorted(glob.glob(os.path.join(d, f"ref-{n}-*.ids")))]
        defs = [rd(f) for f in sorted(glob.glob(os.path.join(dd, f"def-{n}-*.ids")))]
        ok_refs = [r for r in refs if r and r.startswith("rc=0 ids")]
        if not ok_refs or not defs:
            print(f"FAIL {n}: no usable runs (refs {len(refs)}, defaults {len(defs)})"); bad += 1; continue
        (maj, cnt), = collections.Counter(ok_refs).most_common(1)
        status = "PASS" if all(x == maj for x in defs) else "FAIL"
        note = f"majority {cnt}/{len(refs)} of main (=3)"
        if len(set(ok_refs)) > 1: note += f"; {len(set(ok_refs))} distinct main outputs"
        print(f"{status} {n}: {len(defs)} default runs {'equal' if status == 'PASS' else 'DIFFER from'} the main majority ({note})")
        bad += status == "FAIL"
print("ALL PASS" if not bad else f"{bad} FAIL")
sys.exit(1 if bad else 0)
