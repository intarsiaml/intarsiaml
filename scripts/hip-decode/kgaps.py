# kgaps.py <run_kernel_trace.csv> [min_us] [max_us]: GPU gaps between consecutive kernels in [min_us, max_us], with the
# kernels on both sides and the 3 kernels before / after
import csv, sys
K = sorted((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"][:50]) for r in csv.DictReader(open(sys.argv[1])))
lo = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
hi = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
t0 = K[0][0]
for i in range(1, len(K)):
    g = (K[i][0] - K[i - 1][1]) / 1e3
    if lo <= g <= hi:
        print(f"[{i}] t={(K[i-1][1]-t0)/1e6:.3f} s gap {g:.1f} us")
        for j in range(max(0, i - 3), min(len(K), i + 3)):
            print(f"     {j} {K[j][2]:50s} {(K[j][1]-K[j][0])/1e3:8.1f} us")
