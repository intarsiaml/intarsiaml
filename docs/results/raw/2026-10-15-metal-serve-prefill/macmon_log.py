#!/usr/bin/env python3
# reads `macmon pipe` JSON lines on stdin; tab-separated, one line per sample:
# epoch gpuMHz gpuutil% gputemp sys_power gpu_power pcpu_util% ecpu_util% cpu_power
import sys, json, time
for line in sys.stdin:
    try:
        d = json.loads(line)
    except Exception:
        continue
    print(f"{time.time():.3f}\t{d['gpu_usage'][0]}\t{d['gpu_usage'][1]*100:.0f}\t{d['temp']['gpu_temp_avg']:.1f}\t{d['sys_power']:.1f}\t{d.get('gpu_power',0):.1f}"
          f"\t{d['pcpu_usage'][1]*100:.0f}\t{d['ecpu_usage'][1]*100:.0f}\t{d.get('cpu_power',0):.1f}", flush=True)
