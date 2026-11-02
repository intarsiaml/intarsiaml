#!/usr/bin/env bash
# Lane r9t (run ON tatooine, e.g. `ssh tatooine bash -s < scripts/r9t/preflight.sh` inside a --load none fleetq job): the built binary's
# version and the serve flags the window uses, before radiance goes down.
D="docker run --rm -v $HOME/intarsiaml-tp2-${TREE:-r9s}:/work -w /work intarsia-hip-dev:rocm7.2.4-rust1.90.0 /work/tp2-intarsia"
$D --version
$D serve --help | grep -aE -B1 -A3 -e '--spec|--max-batch|--tp|--devices'
sha256sum "$HOME/intarsiaml-tp2-${TREE:-r9s}/tp2-intarsia"; ls -l --time-style=full-iso "$HOME/intarsiaml-tp2-${TREE:-r9s}/tp2-intarsia"
