# Summarise the rope_tail kernels of a cuobjdump -sass listing: instruction counts that show how nvcc contracted `re * cs - im * sn` (FFMA vs FMUL/FADD),
# and keep their SASS for diagnosing a hash mismatch without another window.
# usage: cuobjdump -sass obj | awk -v tag=<variant> -v out=<file for the rope kernels' SASS> -f rope-sass.awk
/Function :/ { fn = $3; keep = (fn ~ /rope_tail/); if (keep) seen[fn] = 1 }
keep { print > out
       if ($0 ~ /FFMA/) ffma[fn]++
       if ($0 ~ /FMUL/) fmul[fn]++
       if ($0 ~ /FADD/) fadd[fn]++
       if ($0 ~ /MUFU/) mufu[fn]++ }
END { for (f in seen) printf "rope sass %s %s: FFMA %d FMUL %d FADD %d MUFU %d\n", tag, f, ffma[f], fmul[f], fadd[f], mufu[f] }
