#!/bin/sh
# Build the standalone Q4_0 repack GEMM test harness.
#
#   ./repack_q40_test/build.sh        # build
#   ./repack_q40_test/test_gemm_q40   # run
#
# Uses the same compiler, defines and include paths as the ggml-hip build
# (see build/compile_commands.json), plus the HIP runtime for linking.
set -e
cd "$(dirname "$0")/.."   # repo root

CLANG="${CLANG:-/opt/rocm-7.2.4/lib/llvm/bin/clang}"
[ -x "$CLANG" ] || CLANG="$(hipconfig -l)/clang"

HIP_LIBS="$(hipconfig -R)/lib"

"$CLANG" \
    -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_HIP_GRAPHS -DGGML_HIP_NO_VMM \
    -DGGML_SCHED_MAX_COPIES=16 -DGGML_SCHED_MAX_SPLIT_INPUTS=64 -DGGML_SHARED \
    -DGGML_USE_HIP -DUSE_PROF_API=1 -D_GNU_SOURCE -D_XOPEN_SOURCE=600 \
    -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
    -I ggml/src/ggml-cuda \
    -I ggml/src \
    -I ggml/include \
    -O3 -DNDEBUG -std=gnu++17 --offload-arch=gfx906 -fPIC \
    -x hip \
    repack_q40_test/test_gemm_q40.cu \
    ggml/src/ggml-cuda/q8_repack/repack-common.cu \
    -o repack_q40_test/test_gemm_q40 \
    -L "$HIP_LIBS" -lamdhip64 -lstdc++ -lm -lpthread

echo "built: repack_q40_test/test_gemm_q40"
echo "run:   ./repack_q40_test/test_gemm_q40"
