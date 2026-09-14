# Q4_0 dot8 port to NAVI21 / gfx1030

Date: 2026-09-13

Ports the gfx906 Q4_0 `v_dot8_i32_i4` MMQ path (see `q40_dot8_experiment_progress.md`
and `dot8_optimization_sept_12.md`) to RDNA2 / gfx1030 (Radeon PRO V620).
Hardware facts for gfx1030 live in `PROFILE-NAVI21.md` (dot8 availability, LDS cap,
VGPR cap, profiler counter limits); this file only records the port itself.

## Why it works with almost no kernel changes

- `v_dot8_i32_i4` exists on gfx1030 and runs at full issue rate (PROFILE-NAVI21.md section 2/3).
- `ds_read_b128` (16B LDS reads, tuning item c) exists on RDNA2.
- The dp8 kernel, X-tile repack, activation quantizer, and workspace sizing are all
  generic over `ggml_cuda_get_physical_warp_size()` (32 on RDNA2, 64 on gfx906) and over
  the tile config (`nwarps`, `I`, `J`). No kernel logic was changed.
- The rdna2 config already runs Q4_0 at 256 threads = 8 wave32 waves, which equals the
  8-wave setup that was tuning item (a) on gfx906. Q4_0 is capped at J=64 by the rdna2
  config table.

## Code changes (8 sites, guard/check widening only)

| File | Change |
|---|---|
| `ggml/src/ggml-cuda/common.cuh` | `ggml_cuda_dp8_i4` guard: `__gfx906__` -> `__gfx906__ \|\| RDNA2` |
| `ggml/src/ggml-cuda/mmq.cuh` | `mmq_use_q4_0_dp8`: cc check now `GGML_CUDA_CC_VEGA20 \|\| GGML_CUDA_CC_IS_RDNA2(cc)`; same guard widening on `mmq_get_y_block_size` and the Q4_0 util-funcs dispatch |
| `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | Q4_0 X-tile repack branch guard widened to `__gfx906__ \|\| RDNA2` |
| `ggml/src/ggml-cuda/mmq-vec-dot.cuh` | dp8 vec-dot kernel guard widened |
| `ggml/src/ggml-cuda/quantize.cu` | `quantize_mmq_q8_1_cuda` and `quantize_scatter_mmq_q8_1_cuda` select the int4 activation quantizer for RDNA2 as well |

The `RDNA2` macro is defined in `ggml/src/ggml-cuda/vendors/hip.h` for gfx1030-1037.

## Build

Environment setup required before building:

```sh
export HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)"
export CCC_OVERRIDE_OPTIONS="^--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/12"
```

### dp8 build for gfx1030

Built with the portability flags (see below), since the binaries run on the V620 machine:

```sh
cmake -S . -B build-dp8-gfx1030 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1030 \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_HIP_FLAGS=-DGGML_CUDA_Q4_0_INT4_ACTIVATIONS
cmake --build build-dp8-gfx1030 --target llama-cli llama-bench llama-perplexity -j18
```

### dp4 baseline build for gfx1030 (macro disabled)

`build-dp4-gfx1030` is already built with these flags.

```sh
cmake -S . -B build-dp4-gfx1030 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1030 \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build-dp4-gfx1030 --target llama-cli llama-bench llama-perplexity -j18
```

### gfx906 (regression check)

The existing `build-dp8` directory (gfx906, macro enabled) was rebuilt incrementally and
`llama-cli`, `llama-bench`, `llama-perplexity` all build clean.

### Building on one PC, running on another

llama.cpp needs to be built with the following to work on a different PC:

```sh
  -DBUILD_SHARED_LIBS=OFF
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
```

Static linking avoids carrying this machine's shared library set across machines;
position-independent code keeps the statically linked executables loadable on the
target system. `build-dp8-gfx1030` was configured with both flags.

## Verification done here (no gfx1030 HW on this machine)

- Both gfx906 and gfx1030 builds compile clean.
- ISA check: extracted the `.hip_fatbin` from `build-dp8-gfx1030/bin/libggml-hip.so`
  (`llvm-objcopy --dump-section .hip_fatbin=...`, split at `\x7fELF`, disassemble with
  `llvm-objdump -d --disassemble-all`) and confirmed the Q4_0 `mul_mat_q` kernels
  (ggml_type 2, J=8..64) contain `v_dot8_i32_i4` and `ds_read_b128`. The J>64 Q4_0
  instances are empty stubs, as expected (rdna2 config caps Q4_0 at J=64).

## Testing plan on the V620 machine (gfx1030 HW)

1. Baseline: run `llama-bench -m <Q4_0 model> -p 512` with the dp4 build to get the
   dp4a reference number.
2. Correctness: `llama-perplexity` with the dp8 build, same smoke test as the MI50 work
   (`tools/server/README.md`, 8 chunks, n_ctx=512, batch 512). Expect PPL near the MI50
   dp8 result (~4.0 vs 3.69 CPU control for the 9B model), not the catastrophic ~110748
   of the old packing bug.
3. Optional: scalar-reference build (`-DGGML_CUDA_Q4_0_INT4_SCALAR_REFERENCE` in
   CMAKE_HIP_FLAGS) + `tools/q40-dot8-compare.py` KL check, same as the MI50 A/B.
4. Perf: compare pp512 dp8 vs dp4a. The MI50 "+7% over baseline" does not transfer as an
   absolute number.
5. Profiling: gfx1030 rocprof has no cache/VALU counters (PROFILE-NAVI21.md section 10),
   so tune on llama-bench deltas plus
   `hipcc --offload-arch=gfx1030 -O3 -Rpass-analysis=kernel-resource-usage` for
   VGPR/occupancy/spill checks.

## Results on V620 / gfx1030 (measured 2026-09-13)

`llama-bench`, pp4096 / ub2048, single V620. dp4 build: d9c6fc44d, dp8 build: 70cba9f29.

| model | build | pp4096 t/s | tg128 t/s |
|---|---|---:|---:|
| Qwen3.5-9B-Q4_0 | dp4 | 1903.10 +/- 5.39 | 63.85 +/- 0.13 |
| Qwen3.5-9B-Q4_0 | dp8 | 2368.02 (single run, -r 1) | 63.54 +/- 0.00 |
| Qwen3.8-27B-Q4_0 | dp4 | 556.87 +/- 3.58 | 22.35 +/- 0.03 |
| Qwen3.8-27B-Q4_0 | dp8 | 704.55 +/- 4.19 | 22.10 +/- 0.08 |
| Qwen3.8-27B-IQ4_NL | dp4 | 529.49 +/- 3.81 | 22.07 +/- 0.02 |

Takeaways:

- 9B Q4_0: pp4096 +24.4% (1903 -> 2368 t/s), tg128 unchanged (63.85 -> 63.54, decode is
  memory-bound and does not use the dp8 prefill path). The 9B dp8 numbers are from a
  single run (-r 1), so the +/- is 0.00 by construction; re-run with -r 5 for a stable
  number if it is ever cited.
- 27B Q4_0: pp4096 +26.5% (556.87 -> 704.55 t/s), tg128 unchanged (22.35 -> 22.10). The
  dp8 gain holds on the larger model, where prefill is more weight-bandwidth bound.
- The IQ4_NL dp4 row is a different quant (4.5 bpw, not plain Q4_0) and the dp8 path does
  not apply to it; listed for reference only, not as an A/B.
- Correctness: coherent output and expected perplexity on the V620, no packing-bug
  symptoms.

## Next tuning candidates (not done)

- J=128 for Q4_0 on RDNA2: add a J=128 row to the config (rdna2 table or a new
  `mmq-config-gfx1030.cuh`, mirroring `mmq-config-gfx906.cuh`), and extend the `J > 64`
  occupancy gate in `mul_mat_q_switch_J` (`mmq.cuh`, currently `cc == GGML_CUDA_CC_VEGA20`)
  to RDNA2 as well, or row-sharded/MoE shapes get over-wide tiles.
- k01 unroll: the dp8 kernel already has `#pragma unroll 2` unconditionally. wave32 has a
  256 VGPR cap (vs 128/lane on wave64 gfx906), so unroll 4 is worth re-testing on
  gfx1030; it spilled on gfx906 for a wave64-specific reason.
  -> very poor performance apparently spills
