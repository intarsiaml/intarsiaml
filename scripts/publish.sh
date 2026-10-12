#!/usr/bin/env bash
# Build and publish to PyPI with the project-scoped token from ~/.secrets.env.
# Bump the version in pyproject.toml and src/intarsiaml/__init__.py first.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/.secrets.env
export UV_PUBLISH_TOKEN="$PYPI_INTARSIAML"
rm -rf dist
uv build
uv publish "$@"
