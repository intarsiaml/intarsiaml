import re,sys,glob,os
# one line per probe run: dir/mode-gGAP-kKEEP-WHAT-REP.txt -> first use of each shard (ms), sum
for d in sys.argv[1:]:
    for f in sorted(glob.glob(f"{d}/*.txt")):
        u=[float(m.group(1)) for m in re.finditer(r"use 0 .*?: ([\d.]+) ms",open(f).read())]
        if len(u)==5: print(f"{os.path.basename(f)[:-4]:34s} first use ms: "+" / ".join(f"{x:7.2f}" for x in u)+f"   sum {sum(u):7.1f}")
