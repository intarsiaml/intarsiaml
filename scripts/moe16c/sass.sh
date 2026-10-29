#!/usr/bin/env bash
# Lane moe16c: per-kernel SASS instruction counts (whole function, all instructions, and the opcodes of the inner loop's work)
# of the M = 16 row kernels the 35B's batched step takes, for each binary given.
for b in "$@"; do
  echo "== $b"
  /usr/local/cuda/bin/cuobjdump -sass "$b" | awk '
    /Function :/ { fn = $3; keep = (fn ~ /fp8_gemv_rows_q_kernelILi16ELi2ELi8ELi2E/ || fn ~ /fp8_gemv_rows_q_kernelILi16ELi1ELi16ELi1E/ || fn ~ /nvfp4_gemv_full_kernelILi16ELi2ELi16E/ || fn ~ /nvfp4_gemv_tile_kernelILi16ELi4E/ || fn ~ /nvfp4_gemv_split_kernelILi8ELi1E/); if (keep) order[++n] = fn; next }
    keep && /\/\*[0-9a-f]+\*\/ / { op = $2; if (op ~ /^@/) op = $3; sub(/\..*/, "", op); sub(/;$/, "", op); c[fn, op]++; tot[fn]++ }
    END { split("FFMA FHFMA HADD2 F2FP PRMT LDS LDG IMAD SHF LOP3 FADD SHFL", ops, " ");
          for (i = 1; i <= n; i++) { f = order[i]; nm = f; sub(/^.*_cu_[0-9a-f]*[0-9][0-9]/, "", nm); line = sprintf("%-58.58s total %5d", nm, tot[f]);
            for (k = 1; k <= 12; k++) line = line sprintf(" %s %d", ops[k], c[f, ops[k]] + 0); print line } }'
done
