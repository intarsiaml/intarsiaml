#!/usr/bin/env bash
docker rm -f fx6-intarsia fx6-llamacpp-hip fx6-hipfire intarsia-serve llamacpp-hip-serve >/dev/null 2>&1
for f in /tmp/fx6-llama.pid; do
  [ -f $f ] && { kill $(cat $f) 2>/dev/null; sleep 2; kill -9 $(cat $f) 2>/dev/null; rm -f $f; }
done
true
