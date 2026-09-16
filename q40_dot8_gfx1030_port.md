# Q4_0 dot8 port to NAVI21 / gfx1030

Date: 2026-09-13

Ports the gfx906 Q4_0 `v_dot8_i32_i4` MMQ path (see `q40_dot8_experiment_progress.md`
and `dot8_optimization_sept_12.md`) to RDNA2 / gfx1030 (Radeon PRO V620).
Hardware facts for gfx1030 live in `PROFILE-NAVI21.md`
(`~/sambashare/vllm-rdna2-qwen/docs/rdna2/`, the single source of truth: bandwidths,
latencies, cache sizes, instruction rates, occupancy); this file only records the port
itself and links there instead of restating hardware figures where possible.

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

Environment setup required before building, depending on the machine:

Xeon E5 2699v3 machine (Rocm 7.2.4):

```sh
export HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)"
export CCC_OVERRIDE_OPTIONS="^--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/12"
```

Epyc 7302 machine (Rocm 10.0):

```sh
export LD_LIBRARY_PATH=/opt/rocm/core/lib:$LD_LIBRARY_PATH
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

### Building for the same machine

If the build runs on the same machine (no cross-machine transfer), invert both flags:

```sh
  -DBUILD_SHARED_LIBS=ON
  -DCMAKE_POSITION_INDEPENDENT_CODE=OFF
```

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

## Results on V620 / gfx1030 (measured 2026-09-15)

Qwen3.5-2B-Q4_0 (1.12 GiB, 1.88 B params) is the only model that fits in the free
VRAM currently available on the V620, but it is a good progress tracker since it
exercises the same dp8 prefill path.

`llama-bench -m ~/sambashare/Qwen3.5-2B-Q4_0.gguf -ngl 99 -dev ROCM0 -p 512 -ub 512 -r 10 -n 0`,
single V620. Both builds: d2f2c0e49 (10937).
GPU tuned: power limit 232W, voltage offset -0.1V, max clock 2100MHz.

| model | build | pp512 t/s |
|---|---|---:|
| Qwen3.5-2B-Q4_0 | dp4 | 6189.27 +/- 159.42 |
| Qwen3.5-2B-Q4_0 | dp8 | 7487.29 +/- 218.44 |

Takeaway: pp512 +21.0% (6189 -> 7487 t/s), consistent with the earlier 9B/27B pp4096
results (2026-09-13, +24.4% / +26.5%).

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

## J=128 for Q4_0 on RDNA2 (tested 2026-09-15, rejected)

Implemented as `mmq-config-gfx1030.cuh` (wraps the rdna2 table, offers Q4_0 J=8..128,
256 threads = 8 wave32 unchanged), host dispatch on `cc == GGML_CUDA_CC_RDNA2`, device
dispatch on `defined(RDNA2)`, and the `J > 64` occupancy gate in `mul_mat_q_switch_J`
extended to `VEGA20 || RDNA2`. A/B on the V620 (36 WGPs / 72 CUs), Qwen3.5-2B-Q4_0, same build
otherwise, -r 10:

| test | dp8, J<=64 | dp8, J<=128 | delta |
|---|---:|---:|---:|
| pp512 (ub512) | 7491.52 +/- 236.54 | 7339.78 +/- 225.57 | -2.0% |
| pp2048 (ub512) | 7408.62 +/- 8.02 | 7275.80 +/- 8.01 | -1.8% |

J=128 was confirmed selected for the large shapes (debug print in `mul_mat_q_switch_J`;
the gate still holds small shapes at J=64) and output was correct, so the delta is real,
not a selection artifact. Unlike gfx906, where the wide-tile config also raised the
kernel to 8 wave64 (512 threads), the rdna2 config already runs 8 wave32 - J=128 only
halves the tile count over an already well-filled 36-WGP grid and loses ~2%. Reverted.

## Splitting the dp8 dot8 accumulator (analyzed 2026-09-15, not done)

Idea: in `ggml_cuda_mmq_vec_dot_q4_0_q4_0_dp8`, the 4 `v_dot8_i32_i4` per (i, k01) form a
serial chain (the asm uses the destination VGPR as the accumulator input, so each dot8
depends on the previous one). Splitting `sumi` into two independent accumulators (2+2)
would let the scoreboard overlap them.

Not worth doing, for two reasons found in the disassembly of the built J=64 kernel:

- The compiler already interleaves independent chains across the unrolled `i0` loop -
  4 chains of depth 4 (one VGPR per unrolled i, e.g. v93-v96), issued back-to-back.
  8 chains of depth 2 would shorten an already-hidden stall marginally.
- VGPRs: the J=64 fallback=0 kernel already uses 254 of the 256 wave32 VGPRs. The split
  adds one live accumulator per (j, i) pair, +32 for J=64, which would spill to scratch -
  the same failure as unroll 4.

VGPR count method: `tools/vgpr-check.py build-dp8-gfx1030/bin/libggml-hip.so
--filter 'mul_mat_qIL9ggml_type2E'` (dumps `.hip_fatbin`, splits at the ELF magic,
disassembles, reports max vreg per kernel; Q4_0 is `ggml_type 2`).

## SW-pipelining the MMQ tile loads (investigated 2026-09-15, probe not yet built)

Question: can the gfx906 `q8_repack` idea (write-once permuted weights + SW-pipelined
K loop) help the dp8 MMQ on gfx1030? Investigation of the actual kernel structure first,
because it changes where any pipelining would have to go.

### Kernel structure (what the dp8 MMQ actually does per K-iteration)

Both operands are already LDS-staged; the global loads happen once per kb0
iteration, not per k01. The dp8 vec-dot's k01 loop is 4 iterations (`k01 < 32,
step QR4_0*VDR = 8`, `#pragma unroll 2`) reading LDS only. So "prefetch the next
k01 weight fragment" (the original probe idea) is moot - the exposure is at the
kb0 level.

Constants (rdna2 Q4_0 config, J=64): 256 threads = 8 wave32, I=128, J=64,
`MMQ_ITER_K = 256`, `blocks_per_iter = 256/32 = 8` blocks = 256 K values per kb0
iteration. Activation layout `block_q4_0_mmq_dp8` = 80B per 128 values (4x half2
scales = 16B + 16 packed ints = 64B), written k-block-major by the quantizer so a
tile's rows are contiguous. 80B is 16B-aligned, which is why the dp8 kernel can read
the 4 y operands of a (j, k01) as one `ds_read_b128` (confirmed in the J=64
disassembly: 176 b128, 0 b64).

Per kb0 iteration (256 K values):

- x tile (weights): `load_tiles_q4_0` repacks 128 rows x 8 blocks x 20B = ~20KB of
  canonical Q4_0 from VRAM (strided by the full row stride) into 128 x 132B = 16.9KB
  of LDS (32 code ints + 1 float per row).
- y tile (activations): two flat VRAM->LDS copies of 64 rows x 80B = 5.12KB each
  (two 128-value halves), contiguous in the k-block-major buffer.
- compute: two `vec_dot` calls (k00 = 0 and 32). Per call per wave: 4 k01 x 8 j
  (J/nwarps) x 4 i (I/warp) x 4 dot8 = 512 dot8 wave-instructions. dot8 is full-rate
  hardware (PROFILE-NAVI21.md section 2: 0.757 issue/clk/SIMD measured in a tight
  microbenchmark, 76% of the 1/clk peak, shortfall attributed to that benchmark's loop
  overhead) -> ~512-670 cycles per call. Per iteration: ~1024-1350 cycles per wave;
  the block's 8 waves sit on the WGP's 4 SIMDs (2 per SIMD) -> ~2050-2700 cycles per
  block-iteration (~1.0-1.3 us at our 2.1 GHz tuning).

The iteration pattern is strictly alternating, every load fully exposed:

```
L x tile (~20KB, strided)  -> sync -> C0 (~512-670 cyc) -> sync
L y half1 (5KB)            -> sync -> C1 (~512-670 cyc) -> sync -> next iteration
```

### Is it bandwidth-bound, compute-bound, or latency-bound?

Hardware figures from PROFILE-NAVI21.md (measured on this card): DRAM 506 GB/s
(98.8% of the 512 theoretical); latency ladder L2 116.6 ns / Infinity Cache 150.1 ns /
DRAM 180.3 ns at ~2.4 GHz, i.e. ~245 / 315 / 379 cycles at our 2.1 GHz tuning; L2 is
4 MiB, Infinity Cache 128 MB at ~1890 GB/s.

- Aggregate request rate if every block-iteration hit DRAM: 36 WGPs x ~31KB per
  ~2050 cycles = ~1.1 TB/s, ~2.2x DRAM. But the unique data per iteration is ~30KB
  (weights 128x256 x 0.625B + acts 64x256 x 0.625B); the rest is reuse (each weight
  tile is read by all 8 j-tile blocks, each act tile by all 16 i-tile blocks). The
  per-k-slice working set (~410KB) fits the 4 MiB L2 with 10x margin, and a whole
  layer's weights (~30MB) fit the 128 MB Infinity Cache - so tile loads should be
  L2/IC hits, not DRAM, and the request rate is absorbed by caches whose bandwidth
  is far above 506 GB/s.
- Whole-batch roofline, per 512-token prefill batch (1.88B params, ~1.3B
  non-embedding): dot8 peak = 144 SIMD32 (36 WGPs x 4) x 32 lanes x 8 = 36,864
  MAC/cycle = ~77 GMAC/s (~155 GFLOPS) at 2.1 GHz; the profile's microbenchmark
  measures 76% of that. ~1.3e12 MACs per batch -> ~17 ms at peak, ~22 ms at the
  measured 76%. Minimum DRAM = ~1.12GB weights + ~0.1GB acts = ~1.2GB -> ~2.4 ms at
  506 GB/s. Measured: 512/7487 = 68.4 ms.
- So the measured time is ~3-4x the compute roof and ~28x the DRAM roof. Neither
  peak FLOPs nor peak bandwidth is the direct limiter. But the exposed-latency
  estimate below (~25-35% of iteration time) does not by itself explain a 3-4x gap
  to the compute roof, so additional factors must also be at play (4B-wide loads
  limiting in-flight bytes, barrier overhead, non-dp8 ops, launch overhead across
  the hundreds of kernel launches per batch). The gfx1030 profiler has no cache
  counters
  (PROFILE-NAVI21.md section 10), so this decomposition cannot be confirmed
  directly.

Each exposed load phase costs ~one memory latency (the tile's loads are issued in
parallel, the sync waits for the last): ~245-379 cycles for L2/IC/DRAM. Two phases
per iteration -> ~500-760 cycles against ~2050-2700 of compute, ~25-35% of
iteration time. This is the hypothesis the probe tests.

PROFILE-NAVI21.md section 4a rule 2 / T-O1 names exactly this pattern as a failure
mode: lanes sitting on a synchronised LDS tile with one outstanding load per lane,
then a barrier. Its T-O1 table shows 2 waves/SIMD with one load per lane still
reaches ~96% of DRAM peak - but that assumes loads are independent and re-issuable
every cycle, which a barrier-serialised load->store->compute loop is not: the
memory pipe sits idle through the whole compute phase. The measured sweet spot is
2-4 independent in-flight loads per lane at low occupancy - which is exactly what
the y-pipeline probe adds.

### Probe design (corrected) and its limits

AMD has no cp.async: global->LDS must round-trip through VGPRs, and the LDS store
instruction stalls until the data arrives. Hiding latency therefore means holding
the prefetched data in VGPRs across the compute phase (the repack-GEMM pattern),
which costs VGPRs - and the J=64 kernel is at 254 of 256.

- y tile (activations) is pipelinable cheaply: double-buffer the 5.12KB y tile in
  LDS (plenty of headroom under the 64KB cap), prefetch each half one iteration
  ahead into 5 VGPRs per thread (64 x 20 ints / 256 threads = exactly 5), issue at
  the top of the compute phase, store after it. +10 VGPR total -> 264, over the
  cap, so the tile copy code would also move to a separate function to let the
  allocator free the transient copy VGPRs first. ~40-50 lines in
  `mul_mat_q_process_tile`, dp8-guarded.
- x tile (weights) is NOT pipelinable this way: 128 x 33 ints / 256 threads = 17
  VGPR per thread, which would spill. Hiding the ~20KB weight load is what the full
  repack redesign (dedicated GEMM, repacked write-once buffer) is for - the probe
  cannot test it.
- Consequence: the y-only probe is ambiguous. A win proves latency matters. A
  neutral result does not rule it out, because the weight load (the larger, more
  likely-DRAM-missy one) stays exposed. Only the repack - or an x-pipeline that
  frees ~25 VGPR elsewhere - settles it.

### Straight-copy probe (built 2026-09-16): the repack is register-free

Idea: disable the in-kernel repack (straight copy of the canonical code words,
values come out wrong, timing is the point) and see if that frees enough VGPRs
for the double-buffer prefetch. `GGML_CUDA_Q4_0_DP8_PROBE_STRAIGHT` in
`load_tiles_q4_0`: 4B straight copies of the 16B code words (20B blocks and 33-int
LDS rows are not 16B aligned, so no int4), no nibble shuffle / sign fix. Separate
build dir `build-dp8-probe` (baseline build left intact).

Result: it frees nothing.

| kernel (Q4_0) | baseline | probe | |kernel (Q4_0) | baseline | probe |
|---|---|---|---|---|---|---|
| J=64 fb0 | 254 | 254 | | J=32 fb0 | 167 | 167 |
| J=64 fb1 | 184 | 183 | | J=32 fb1 | 186 | 185 |

The probe is verifiably active in the disassembly (nibble-shuffle ALU gone:
v_and 1652->237, v_lshrrev 749->57, v_xor 353->1, v_or3 528->0; loads now
dwordx4 9->185, stores ds_write2 0->352) yet maxvgpr is unchanged. The repack
conversion was already register-free; the 254 VGPR wall is the fully unrolled
dot8 compute itself (fb0 254 vs fb1 184 = the unrolling cost), which is also
what makes the dot8 issue rate good. Any prefetch (y: +10, x: +17) still spills
(264/271 > 256).

Timing A/B pp512 -r 10 (garbage output, timing only): contaminated - a
llama-server with a 27B MTP draft on the same GPU was running during the bench.
A-B-A = 7086 / 6429 / 6527 t/s; the probe sits inside the baseline's own run-to-
run spread, so the repack appears timing-neutral but this is not conclusive.

Consequence: the double-buffer probe is dead in its current form. What remains:

1. Full `q8_repack` port: a different kernel structure with its own VGPR budget
   (the tiled GEMM pipelines the K loop by design). Big job, benefit unproven on
   RDNA2/GDDR6.
2. J=32-based pipeline: J=32 fb0 is 167 VGPR, so +10 (y) and even +17 (x)
   prefetches fit. But changing J=64->32 doubles the block count, so an A/B
   would conflate tile shape with pipelining.
3. Shelve pipelining; keep the current dp8 (pp512 ~7500 t/s, +21% over dp4).

### SW pipeline implemented and shelved (2026-09-16)

Option 2 above was implemented and A/B'd. Plan: `q40_dp8_pipeline_plan.md`.

Implementation (all behind `GGML_CUDA_Q4_0_DP8_PIPELINE`, off by default):

- `mmq-load-tiles.cuh`: `prefetch_x_q4_0_dp8` (canonical 4B word-pair loads into
  registers, no conversion) + `store_x_q4_0_dp8` (repack conversion + LDS store).
  The existing fused `load_tiles_q4_0` is untouched; the non-pipelined path is
  byte-identical to before.
- `mmq.cuh` `mul_mat_q_process_tile`: pipelined kb0 loop for `type==Q4_0 && J==32`
  only (J=64 + prefetch would spill: 254 + ~30 > 256). The y tile is prefetched
  as two register arrays (the two 80B halves are separate VRAM locations, stride
  `ncols_y*sz`).
- `mmq.cuh` `mul_mat_q_switch_J`: `GGML_CUDA_MMQ_J` env override (static-cached
  getenv, printed once) to force the tile width for A/B runs.

Loop order (the subtle part): the prefetch must come AFTER both y half-stores,
not after the first. The plan doc's original order (prefetch mid-iteration) had
a clobber hazard: the prefetch overwrites `y_r2`, which `store_y_half(1)` of the
SAME iteration still needs - the second vec_dot would compute on the next slab's
y half. q8_repack never hits this because its y tile is stored once; the dp8
kernel stores y twice per iteration. Corrected order:

```
store x -> LDS; store y half0 -> LDS; sync;
vec_dot(0); sync;
store y half1 -> LDS; sync;          // all register consumers done
if (next slab) prefetch(next);       // loads in flight...
vec_dot(32); sync;                   // ...hidden under this compute
```

This bug was caught by a perplexity bisection (PPL 849,937 vs 2.7970) - the
deterministic-token check would have flagged it too, but PPL is the sharper
tool. Lesson: verify any kernel restructure with perplexity before benching.

Verification:

- VGPR (vgpr-check.py): J=32 fb0 167->193, fb1 186->199. No spills. J=64
  kernels unchanged (254/184).
- PPL (256-ctx, 800-token ref): 2.7970 for baseline J=64, J=32 no-pipe, and
  J=32 pipe - identical.
- Deterministic tokens (--temp 0 --seed 42, 100 tokens): J=32 no-pipe and
  J=32 pipe produce identical token streams.

3-way A/B (pp512 -r 10, interleaved 1,3,2,3,1,2; run-to-run variance <0.3%):

| run | t/s | t/s |
|---|---|---|
| 1: J=64 baseline | 7493 | 7500 |
| 2: J=32 no-pipe | 6737 | 6732 |
| 3: J=32 pipe | 6750 | 6752 |

pp2048 -r 5 (confirm): 7434 / 6670 / 6688.

- shape effect (2 vs 1): -10.2% (J=32: 2x blocks, half the compute per
  iteration, more exposed load per FLOP)
- pipeline effect (3 vs 2): +0.25% / +0.27% - within noise, no benefit
- net (3 vs 1): -10%

Conclusion: shelved. The exposed tile loads were already hidden on gfx1030
(L2/IC hits + inter-wave scheduling); the pipeline recovers nothing, and the
J=64->32 shape change it requires costs 10%. This also closes out option 1
(q8_repack port): its benefit on this hardware is now directly measured as
~zero, so the port is not worth it. The code stays in the tree, macro-gated
and off by default, in case a larger model or different GPU state changes the
picture; the `GGML_CUDA_MMQ_J` override is a generally useful debug hook.

## Next tuning candidates (not done)

- k01 unroll: the dp8 kernel already has `#pragma unroll 2` unconditionally. wave32 has a
  256 VGPR cap (vs 128/lane on wave64 gfx906), so unroll 4 is worth re-testing on
  gfx1030; it spilled on gfx906 for a wave64-specific reason.
  -> very poor performance apparently spills
- SW-pipelining: implemented and shelved (see above) - no measurable benefit on
  gfx1030, the q8_repack port is closed out as not worth it.
