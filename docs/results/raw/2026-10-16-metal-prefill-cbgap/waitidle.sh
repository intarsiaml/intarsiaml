# source me: wait_idle waits (up to 120 s) until the GPU device utilisation is <= 12% in 3 consecutive 1 s samples (the idle box
# shows 0 to 10% of window server / macmon noise); returns 1 if it never is
wait_idle() {
  local n=0 u i
  for i in $(seq 1 120); do
    u=$(ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*' | grep -o '[0-9]*$' | head -1)
    if [ "${u:-100}" -le 12 ]; then n=$((n+1)); [ $n -ge 3 ] && return 0; else n=0; fi
    sleep 1
  done
  echo "GPU not idle (util ${u}%)" >&2
  return 1
}
