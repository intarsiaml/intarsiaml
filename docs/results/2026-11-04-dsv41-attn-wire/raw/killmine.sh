#!/bin/bash
# kill every process whose cwd is under ~/intarsia-dsv41-attnwire or whose binary lives there (never another lane's processes)
d=$HOME/intarsia-dsv41-attnwire
for p in /proc/[0-9]*; do
  pid=${p#/proc/}
  [ "$pid" = "$$" ] && continue
  c=$(readlink "$p/cwd" 2>/dev/null); e=$(readlink "$p/exe" 2>/dev/null)
  case "$c" in "$d"|"$d"/*) kill -9 "$pid" 2>/dev/null;; esac
  case "$e" in "$d"/*) kill -9 "$pid" 2>/dev/null;; esac
done
