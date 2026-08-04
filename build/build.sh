#!/usr/bin/env bash
set -euo pipefail

mkdir -p dist
mojo build --emit shared-lib src/capi.mojo -o dist/libmojo-pybigwig.so
