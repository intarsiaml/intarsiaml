#!/usr/bin/env bash
# Stage of window.sh (lane ink-hip-bpf, ON tatooine): hold the radiance-down window while the Mac drives server visits
# (serve_ctl.sh); ends when logs/hold-stop exists (or window.sh's cap). Samples both cards' VRAM (sysfs) every 0.5 s meanwhile.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-bpf
rm -f "$REPO/logs/hold-stop"
echo "[hold] holding $(date '+%F %T')"
while [ ! -e "$REPO/logs/hold-stop" ]; do
  echo "$(date +%s.%N) $(cat /sys/class/drm/card0/device/mem_info_vram_used) $(cat /sys/class/drm/card1/device/mem_info_vram_used)" >> "$REPO/logs/vram.log"
  sleep 0.5
done
echo "[hold] stop file seen $(date '+%F %T')"
