#!/bin/sh
# Builds libpattern_lu.so next to this script. The binary is committed; this script is only
# needed after a change of pattern_lu.c. Baseline x86-64 code, AVX2 kernels are selected at run
# time (see plu_select). No fused multiply-add: results are the same on every x86-64 CPU.
set -e
cd "$(dirname "$0")"
${CC:-gcc} -O2 -fPIC -shared -std=gnu11 -ffp-contract=off -fno-math-errno -Wall -Wextra \
    -o libpattern_lu.so pattern_lu.c
echo "built $(pwd)/libpattern_lu.so"
