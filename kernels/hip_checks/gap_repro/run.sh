#!/usr/bin/env bash
# Build and run the gap reproducer: plain (device timestamps) and under rocprofv3 --kernel-trace, for the base config and the env
# knobs that were tried in the engine. Needs hipcc, rocprofv3, python3 on PATH (ROCm container); HIP_VISIBLE_DEVICES picks the GPU.
# Everything lands in results/ (~2 min).
set -u
cd "$(dirname "$0")"; R=results; mkdir -p $R
hipcc -O3 --offload-arch=gfx1201 gap_repro.hip -o gap_repro || exit 1
{ echo "kernel: $(uname -r)"; echo "amdgpu: $(cat /sys/module/amdgpu/version 2>/dev/null || echo 'in-tree (no version file)')"
  for f in /sys/class/drm/card*/device/vbios_version; do echo "vbios $f: $(cat "$f" 2>/dev/null)"; done
  echo "rocm: $(cat /opt/rocm/.info/version 2>/dev/null)"; hipcc --version | head -3
  rocminfo 2>/dev/null | grep -A2 gfx1201 | head -6; (amd-smi firmware 2>&1 || rocm-smi --showfwinfo 2>&1) | head -60; } > $R/env.txt 2>&1

analyze() {  # rocprofv3 csv + launch plan -> gap table; gap = Start[i+1] - End[i] of consecutive dispatches of one experiment
python3 - "$@" <<'EOF'
import csv, glob, sys, statistics as st
d = sys.argv[1]
rows = [r for f in glob.glob(d + "/**/run_kernel_trace.csv", recursive=True) for r in csv.DictReader(open(f)) if "spin_k" in r["Kernel_Name"]]
rows.sort(key=lambda r: int(r["Start_Timestamp"]))
plan = [l.split() for l in open(sys.argv[2])]
assert len(rows) == len(plan), (len(rows), len(plan))
g = {}
for i in range(len(rows) - 1):
    e, mode, grid, lab = plan[i]
    if plan[i + 1][0] == e: g.setdefault((mode, int(grid), lab), []).append((int(rows[i + 1]["Start_Timestamp"]) - int(rows[i]["End_Timestamp"])) / 1e3)
print("# rocprofv3 --kernel-trace; gap = Start[i+1] - End[i] in us")
print("%-5s %5s %-8s %5s %9s %9s %9s %9s %7s" % ("mode", "grid", "dur_ms", "n", "gap_mean", "gap_p50", "gap_p90", "gap_max", ">100us"))
for (m, gr, lab), v in g.items():
    v.sort(); print("%-5s %5d %-8s %5d %9.1f %9.1f %9.1f %9.1f %6.0f%%" % (m, gr, lab, len(v), st.mean(v), v[len(v)//2], v[int(.9*(len(v)-1))], v[-1], 100*sum(x > 100 for x in v)/len(v)))
EOF
}

# name:ENV=VAL  (the three knobs had no effect in the engine's rocprofv3 traces)
for v in base: hsa_interrupt0:HSA_ENABLE_INTERRUPT=0 active_wait:ROC_ACTIVE_WAIT_TIMEOUT=5000 hwqueues4:GPU_MAX_HW_QUEUES=4; do
  n=${v%%:*}; e=${v#*:}
  env $e ./gap_repro > $R/dev-$n.txt                                  # no profiler: device timestamps only
  env $e GAP_PLAN=$R/plan.txt rocprofv3 --kernel-trace --output-format csv -d $R/prof-$n -o run -- ./gap_repro > $R/dev-$n-under-rocprofv3.txt 2> $R/rocprofv3-$n.err
  analyze $R/prof-$n $R/plan.txt > $R/rocprofv3-$n.txt; gzip -f $(find $R/prof-$n -name run_kernel_trace.csv)
done
