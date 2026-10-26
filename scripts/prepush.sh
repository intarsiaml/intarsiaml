#!/usr/bin/env bash
# Fast pre-push check: the two docs-sync tests behind most lane CI failures (2026-10-25/26: 15 of 21 failed runs).
#   every_env_var_in_the_code_has_a_row_in_docs_tuning_md  (crates/intarsia-cli/tests/docs_env_vars.rs)
#   every_schema_key_is_documented                         (crates/intarsia-models/tests/recipes_doc.rs)
# About 45 s from a cold build on endor, a few seconds warm. Run by .githooks/pre-push; enable once per clone with
#   git config core.hooksPath .githooks
# Skip for one push with `git push --no-verify`.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
cargo test -q --locked -p intarsia-cli --test docs_env_vars
cargo test -q --locked -p intarsia-models --test recipes_doc
