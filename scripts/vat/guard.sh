#!/usr/bin/env bash
# Lane vat (copy of scripts/tp2-hip/guard.sh) (ON tatooine, run over ssh by speed-window.sh with its stdin on a fifo): stop radiance, and restore it when stdin
# closes (the Mac side ended or died); prints when radiance answers.
restore() {
  docker rm -f vat-serve >/dev/null 2>&1
  echo "[guard] starting radiance $(date '+%F %T')"
  systemctl --user start radiance < /dev/null
  for i in $(seq 1 48); do
    if curl -s -m 3 http://localhost:8080/v1/models < /dev/null | grep -q '"data":\['; then echo "[guard] radiance answers $(date '+%F %T') (after $((i * 5)) s)"; return 0; fi
    sleep 5
  done
  echo "[guard] RADIANCE DID NOT ANSWER within 240 s $(date '+%F %T')"
}
trap restore EXIT
echo "[guard] stopping radiance $(date '+%F %T')"
systemctl --user stop radiance < /dev/null
sleep 2
echo "[guard] radiance stopped $(date '+%F %T')"
read -r _ || true
