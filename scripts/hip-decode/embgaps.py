# embgaps.py <run_kernel_trace.csv>: every embedding launch, its duration, the gap before it and the 2 kernels before it,
# plus the gap after the last kernel before it that was not a runtime copy / fill
import csv, sys
K = sorted((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"][:44]) for r in csv.DictReader(open(sys.argv[1])))
t0 = K[0][0]
for i in range(1, len(K)):
    if "embedding" not in K[i][2]: continue
    j = i - 1
    while j > 0 and "rocclr" in K[j][2]: j -= 1
    print(f"[{i}] t={(K[i][0]-t0)/1e6:9.3f} ms emb {(K[i][1]-K[i][0])/1e3:7.1f} us; gap {(K[i][0]-K[i-1][1])/1e3:8.1f} us after {K[i-1][2]}; "
          f"{(K[i][0]-K[j][1])/1e3:8.1f} us after last compute kernel [{j}] {K[j][2]} ({i-1-j} runtime ops between)")
