#!/usr/bin/env bash
# one-line GPU state for every card (which card is Vulkan/HIP device 0 is not guaranteed): sclk, temp, vram used, busy
for c in /sys/class/drm/card[0-9]/device; do
  [ -f $c/pp_dpm_sclk ] || continue
  echo -n "$(basename $(dirname $c)): sclk=$(grep '\*' $c/pp_dpm_sclk | tr -s ' ' | tr '\n' ' ') temp_mC=$(cat $c/hwmon/hwmon*/temp1_input 2>/dev/null | head -1) vram_used=$(cat $c/mem_info_vram_used) busy=$(cat $c/gpu_busy_percent) | "
done
echo
