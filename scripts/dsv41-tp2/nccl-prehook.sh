# Lane dsv41/nccl-prefill: sourced by run-full.sh (PRE_HOOK=this file) after PRE="ncclbench", before the AB sessions. Reads the
# ncclbench-r0-<tag>.txt tables and the cold-test lines in $OUT, applies the pass rules written in
# docs/results/2026-11-05-dsv41-nccl.md BEFORE the window, and sets AB / STEPS_ABi / NSYS_ABi for run-full.sh's session loop.
#
# Rules (fixed before the run):
#   baseline arm   p default, 2 channels, 1 MiB buffer (tag pdefault-c2-b1M), the env run-full.sh ships.
#   candidate arm  any other arm whose 16 KiB..128 KiB ar_inplace `sync` time (the closer to a decode collective) is at most
#                  baseline + NB_SMALL_SLACK_US (30) at every one of those sizes; the best is the one with the highest 42 MB GB/s.
#   A passes       best 42 MB GB/s >= 1.3 x baseline's. Then the sessions run the best arm against the baseline (Y).
#   B reproduces   cold test: after-idle median >= warm-at-once median + 2000 us. The warm-up arm (W) uses the smallest lead payload
#                  whose median is within half of the excess of warm-at-once; none -> no W sessions.
#   C              always: hot tier 0 (Z) against the baseline (X, hot tier 1024).
# Sessions (fresh ranks each): round 1 X [Y] [W] Z with the golden ids, round 2 the same without them; then, when W is on, X and W
# once each under nsys (the 4K prompt three times) to read the first all-reduce of each request.
EXPECT_HEAD=${EXPECT_HEAD:?set EXPECT_HEAD (the branch head the Sparks were built from)}
STAMP_NOW=$(r spark1 "cat $SPARK_DIR/bin/STAMP")
[[ "$STAMP_NOW" == "$EXPECT_HEAD dirty=no"* ]] || { log "binary stamp '$STAMP_NOW' is not '$EXPECT_HEAD dirty=no'; not starting"; exit 1; }
log "binary stamp ok: $STAMP_NOW"
DEC=$(python3 - "$OUT" "${NB_SMALL_SLACK_US:-30}" <<'PY'
import glob, os, re, sys
out, slack = sys.argv[1], float(sys.argv[2])
BASE = "pdefault-c2-b1M"
def table(path):
    t = {}
    for l in open(path, errors="replace"):
        m = re.match(r"rank 0\s+(\d+)\s+(.*)", l)
        if not m:
            continue
        nums = [float(x) for x in m.group(2).split()]
        if len(nums) >= 7:
            t[int(m.group(1))] = {"pipe": nums[0], "sync": nums[1], "gbps": nums[-1]}
    return t
arms = {}
for f in glob.glob(os.path.join(out, "ncclbench-r0-*.txt")):
    tag = os.path.basename(f)[len("ncclbench-r0-"):-4]
    t = table(f)
    if 41943040 in t:
        arms[tag] = t
res = []
if BASE not in arms:
    print("A=none base-missing")
else:
    b = arms[BASE]
    small = [s for s in (16384, 32768, 49152, 65536, 98304, 131072) if s in b]
    rows = []
    for tag, t in sorted(arms.items()):
        ok = all(s in t and t[s]["sync"] <= b[s]["sync"] + slack for s in small)
        rows.append((tag, t[41943040]["gbps"], ok))
        res.append(f"# arm {tag}: 42 MB {t[41943040]['gbps']:.2f} GB/s ({t[41943040]['gbps'] / b[41943040]['gbps']:.2f}x), small collectives {'ok' if ok else 'SLOWER'}")
    cands = [r for r in rows if r[0] != BASE and r[2]]
    best = max(cands, key=lambda r: r[1]) if cands else None
    ratio = best[1] / b[41943040]["gbps"] if best else 0
    res.append(f"# baseline {BASE}: {b[41943040]['gbps']:.2f} GB/s ({41943040 / b[41943040]['pipe']:.0f} B/us); best {best}")
    if best and ratio >= 1.3:
        tag = best[0]
        m = re.match(r"p(\w+)-c(\d+)-b(\d+)M", tag)
        env = f"NCCL_MAX_NCHANNELS={m[2]} NCCL_BUFFSIZE={int(m[3]) << 20}" + ("" if m[1] == "default" else f" NCCL_PROTO={m[1]}")
        res.append(f"A=pass {tag} {ratio:.2f}x")
        res.append(f"YENV={env}")
    else:
        res.append(f"A=kill best {best[0] if best else None} {ratio:.2f}x")
# cold test
W = None
for f in sorted(glob.glob(os.path.join(out, "ncclbench-cold*-r0-*.txt"))):
    for l in open(f, errors="replace"):
        m = re.search(r"after-idle ([\d.]+)\s+warm-at-once ([\d.]+)\s+after-idle-with-(\d+)-byte-lead ([\d.]+)", l)
        if m:
            cold, warm, lb, lead = float(m[1]), float(m[2]), int(m[3]), float(m[4])
            res.append(f"# cold {os.path.basename(f)}: after-idle {cold:.0f} us, warm {warm:.0f} us, lead {lb} B {lead:.0f} us")
            if cold >= warm + 2000 and lead <= warm + (cold - warm) / 2 and (W is None or lb < W):
                W = lb
res.append(f"B={'reproduced' if W else 'not-reproduced-or-not-hidden'}")
if W:
    res.append(f"WKIB={max(W // 1024, 1)}")
print("\n".join(res))
PY
)
echo "$DEC" | tee "$OUT/decision.txt" | tee -a "$OUT/steps.log"
YENV=$(sed -n 's/^YENV=//p' <<< "$DEC"); WKIB=$(sed -n 's/^WKIB=//p' <<< "$DEC")
X=""; Z="INTARSIA_DSV41_ENGRAM_HOT_MB=0"; Y="${YENV:+$YENV}"; W="${WKIB:+INTARSIA_DSV41_COMM_WARM=$WKIB}"
CFG=(); NAME=()
for round in 1 2; do
  CFG+=("$X"); NAME+=("X$round")
  [[ -n $Y ]] && { CFG+=("$Y"); NAME+=("Y$round"); }
  [[ -n $W ]] && { CFG+=("$W"); NAME+=("W$round"); }
  CFG+=("$Z"); NAME+=("Z$round")
done
# nsys sessions for the first all-reduce (B), only when the warm-up arm is on
if [[ -n $W ]]; then CFG+=("$X" "$W"); NAME+=("Xn" "Wn"); fi
AB=""; for i in "${!CFG[@]}"; do AB+="${CFG[$i]}|"; done; AB=${AB%|}
for i in "${!NAME[@]}"; do
  case ${NAME[$i]} in
    *1) printf -v "STEPS_AB$i" '%s' "golden lspeed lcat" ;;
    *2) printf -v "STEPS_AB$i" '%s' "lspeed lcat" ;;
    *n) printf -v "STEPS_AB$i" '%s' "pfnsys"; printf -v "NSYS_AB$i" '%s' 1 ;;
  esac
done
log "sessions: $(for i in "${!NAME[@]}"; do echo -n "ab$i=${NAME[$i]}[${CFG[$i]:-current}] "; done)"
echo "$(for i in "${!NAME[@]}"; do echo "ab$i ${NAME[$i]} ${CFG[$i]:-current}"; done)" > "$OUT/sessions.txt"
