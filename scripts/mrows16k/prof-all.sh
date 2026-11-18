#!/bin/zsh
# mrows16k: the profile runs of prof.sh for every mode in one slot (none = plain pass times from INTARSIA_SPEC_TRACE, cb, ce1, kernel).
for m in none cb ce1 kernel; do MODE=$m ${0:A:h}/prof.sh; done
