# Q4_0 repack GEMM port - implementation status

Target: gfx906 (Vega 20 / MI50). Port the dedicated Q8_0/MXFP4 repack GEMM
(two-plane layout, software pipelining, double buffering, swizzled LDS) to Q4_0,
reusing the MXFP4 RAW_REG pattern. Projected +20-35% prefill over the generic
`mul_mat_q<Q4_0>` path.

Design doc: `port_q40_repack.md`. This file tracks what is implemented and the
current state.

## Status: bug ISOLATED to the Q4_0 repack path; every Q4_0 GEMM output verifies correct

Builds clean. Bugs found and fixed so far:
1. **D4/DS4 activation scale layout** (fixed in `mul-mat.cu`).
2. **Repack nibble re-order was a WRONG permutation** (Session 2) - must be a NO-OP.

The standalone harness PASSES at all dims (Q4_0 max_rel ~2e-4). But the real model
(repack ON) is still broken: **PPL = 2.34M**, vs `--no-repack` (generic) **PPL = 1.27**.

Session 3 isolated the fault to the **Q4_0 repack** path specifically (see below):
forcing Q4_0 to generic fixes PPL, forcing Q8_0 to generic does not. Yet a CPU-reference
check of ALL five Q4_0 GEMM shapes shows each GEMM output is correct, the Q4_0 weights are
byte-exact on device, and the ffn_down float activation is finite/sane. So the fault is a
**wrong float hidden state produced by a non-GEMM op** (most likely the SSM
gated_delta_net recurrent scan) that a later GEMM then "correctly" consumes. See "Session 3".

Do NOT bench or ship until repack-ON PPL matches `--no-repack` (~1.27).

## What was implemented (5 files, all in `ggml/src/ggml-cuda/q8_repack/`)

### 1. `repack-common.cuh` (core abstraction)
- `repack_qs_row_stride`: Q4_0 rows carry packed nibbles (`ne0/2` B), same as MXFP4.
- `repack_gcn_nbytes`: Q4_0 = `ne1 * (qs_stride(ne0/2) + (ne0/32) * 2)`
  (nibble plane + 2-byte f16 scale plane per row).
- `repack_qs_bytes`: Q4_0 = 16 (packed nibbles, 1 uint4 per sub-block).
- `kvalues_q4_0[16] = {0..7, -8..-1}` (sign-extended 4-bit table).
- `rp_q4_0_expand(nib, lo, hi)`: widen 16 packed nibbles to two uint4 of int8 via
  the same `get_int_from_table_16` perm-LUT as MXFP4, with the Q4_0 table.
- `rp_traits<GGML_TYPE_Q4_0>`: `raw_lds = true`, `geom` (rs, rs_u4, n_sub,
  dplane_off), `load_w` / `load_w_raw` (nibble uint4 + f16 scale uint16_t),
  `expand` (rp_q4_0_expand), `scale` (`__half2float`).
- Added `expand` to the MXFP4 trait (so the GEMM can call it generically).
- `RP_FOREACH_TYPE` now includes Q4_0.
- Declared `repack_q4_0_host`.

### 2. `repack-kernels.cuh` (GEMM)
- `store_stage`: changed the hardcoded `rp_mxfp4_expand` to the type-generic
  `rp_traits<WT>::expand`. This is the only kernel change; the rest of the GEMM
  (tiled K-loop, double buffering, swizzled LDS, dp4a) is shared.

### 3. `repack-common.cu` (host repack)
- `tensor_supported`: Q4_0 accepted (ne0 % 32 == 0).
- `repack_q4_0_host`: canonical `block_q4_0` (18 B: f16 d + 16 nibble bytes) to the
  two-plane layout. **NO-OP** (copy nibbles verbatim) - the original byte order
  already matches the GEMM pairing (see "Session 2" below). NOTE: the top-of-function
  comment still describes the old re-order (stale).
- `repack_host`: Q4_0 case.

### 4. `buffer.cu` (device-side async repack)
- `repack_q4_0_kernel`: NO-OP copy (18-byte canonical blocks; d@0, qs@2).
- `set_tensor_async`: Q4_0 case.

### 5. `mul-mat.cu` (dispatch)
- Q4_0 added to all four dispatch sites:
  - `mul_mat_vec_rp<Q4_0, 16, 16, false>` (single-token decode).
  - `mmq_gemm_repacked<..., Q4_0>` (ne11 >= 128, GEMM 64-wide).
  - `mmq_gemm_repacked_w32<..., Q4_0>` (9 <= ne11 < 128, GEMM 32-wide).
  - `ggml_cuda_mul_mat_repacked_nc_t<Q4_0>` (narrow 2-8 tokens).

## Verified correct (Session 2 probes, all on-GPU)
- **Widen (`rp_q4_0_expand`)**: probe PASSED (bad=0).
- **Weight load (`load_w_raw`)**: probe PASSED (bad=0), tested at non-origin wrow/sb.
- **GEMM store_stage/compute_stage LDS tiling**: verified (W_ELM = full tile, no
  uninitialized reads).
- **Full GEMM (harness)**: Q8_0 PASS (max_rel 5e-5), Q4_0 PASS (max_rel 2e-4) at
  ne0=256/ne1=64 AND at the real model's ne0=4096/ne1=8192.
- **Q8_0 path unaffected**: Q8_0 model still produces correct output.

## Bug found and fixed: D4 vs DS4 activation scale layout
- `quantize_mmq_q8_1_cuda` picks the activation scale layout via
  `mmq_get_q8_1_ds_layout(weight_type)`. Q8_0/MXFP4 map to **D4** (`float d4[4]`,
  scales at offsets 0/4/8/12), but Q4_0 maps to **DS4** (`half2 ds4[4]`,
  scale + partial-sum interleaved).
- The repack GEMM reads the activation scale via `block_q8_1_mmq::d4` (the D4
  layout) in `rp_x_sub_from_mmq_group`. With DS4 the scale was read from the wrong
  offset -> every sub-block scale wrong -> all `????`.
- Fix (in `mul-mat.cu`): the repack GEMM always reads the D4 layout, so force the
  D4-layout type (Q8_0) when quantizing src1, regardless of the actual weight type:
  ```c
  const ggml_type ds_type = GGML_TYPE_Q8_0;   // maps to MMQ_Q8_1_DS_LAYOUT_D4
  quantize_mmq_q8_1_cuda(..., ds_type, ...);
  ```
  Only the scale layout differs, not the q data. This moved the output from all
  `????` to "real words but garbled".

## PINNED: the bug is in the PREFILL (GEMM), not decode
Verified with `llama-perplexity` (processes the prompt = pure prefill/GEMM, never
feeds a predicted token back, so it isolates prefill from decode):

  repack ON  (GEMM) : PPL = 1,903,537   <-- garbage
  repack OFF (generic) : PPL = 1.0053    <-- near-perfect (repeated text)

So the GEMM is definitively broken. The decode/mat-vec is NOT the problem (the
earlier "suspect #1" was wrong). All the pieces I verified in isolation look
right, so the fault is in something the GEMM does that I have not yet checked end
to end:

- re-order + widen: verified (CPU test + manual check of a real tensor).
- D4 activation scale: fixed (this alone moved output from all `????` to words).
- weight scale: `sWdh` is `uint16_t` for every type, decoded by `rp_traits<WT>::scale`
  (= `__half2float` for Q4_0). Looks right.
- compute_stage dp4a pairing: wlo_c.x*act[0..3], whi_c.x*act[16..19], ... i.e.
  weight[i]*act[i]; matches lo=0..15 / hi=16..31. Looks right.

Remaining GEMM suspects (to check next):
1. **Activation LDS prefetch** - how `sXq`/`sXd` are filled from the D4-layout src1.
   Shared with Q8_0, but confirm the Q4_0 tile actually reads the D4 scales (not a
   stale DS4 assumption baked into the prefetch).
2. **Weight LDS store / swizzle** - `sW_lo`/`sW_hi` indexing in `store_stage` for the
   RAW_REG path; confirm the widened uint4 lands where `compute_stage` reads it.
3. **W_EPT / W_ELM element mapping** - the `e -> (lr, lk)` split in `prefetch_w_*`
   must cover exactly the tile; a mismatch reads the wrong rows/sub-blocks.
4. **Tile boundary / ne0 padding** - confirm ne0=4096 tiles cleanly (BM, BN, BK).

Debug approach: write a small standalone HIP test that (a) repacks a known Q4_0
matrix, (b) runs `mmq_gemm_repacked<...,Q4_0>` against a known activation, and
(c) diffs vs a CPU reference. That isolates the GEMM from the rest of the stack.

## DONE: standalone harness built (`repack_q40_test/`)
- `test_gemm_q40.cu` + `build.sh` (uses the same compiler/defines/includes as the
  ggml-hip build; links libamdhip64). Dimensions are now CLI-configurable
  (`ne0=`/`ne1=`/`ntok=`); also has `widen` and `loadw` probe subcommands, and a
  `ggml_abort` stub.
- Builds a known W [ne1 x ne0] and X [ne0 x n_tok], repacks W (host), quantizes X
  to the q8_1 MMQ D4 layout, runs the GEMM, diffs vs a CPU reference that uses the
  SAME dequantized W and X. Templated on the weight type (Q8_0 and Q4_0).
- Result BEFORE the nibble fix (ne0=256 ne1=64 n_tok=128):
    [Q8_0] max_rel=5.0e-05  -> PASS   (harness is correct)
    [Q4_0] max_rel=61.1     -> FAIL   (Q4_0 GEMM broken)
- Result AFTER the nibble fix (no-op repack), same + real dims:
    [Q4_0] ne0=256/ne1=64    max_rel=1.27e-05  bad=0  -> PASS
    [Q4_0] ne0=4096/ne1=8192 max_rel=2.19e-04  bad=0  -> PASS
  => the GEMM + widen + weight load + D4 activation are ALL correct in isolation.

## Session 2: the nibble re-order bug (found + fixed)

**Root cause of the Q4_0 GEMM failure**: the original repack's nibble re-order was a
WRONG permutation. Ground-truth ggml Q4_0 layout: byte `j` holds `val[j]` (low) +
`val[j+16]` (high). The GEMM pairs `lo[j]` with `act[j]`, `hi[j]` with `act[16+j]`
- which is EXACTLY the original byte order. So the repack must be a **NO-OP**.

- Why MXFP4's re-order is NOT a no-op but Q4_0's is: MXFP4's original byte order is
  `val[2j]`/`val[2j+1]` (identity extraction), so its `(idx[16+j]<<4)|idx[j]`
  formula produces the correct GEMM layout. Q4_0's byte order is `val[j]`/`val[j+16]`
  (NOT identity), so applying the same formula produces a scramble.
- Fix: `repack_q4_0_host` (repack-common.cu) now `d_qs[j] = b->qs[j]` (copy verbatim);
  `repack_q4_0_kernel` (buffer.cu) now `d_qs[j] = sb[2 + j]` (copy verbatim).
- Caveat: the top-of-function comment in `repack_q4_0_host` still describes the old
  re-order (STALE) - the code is the no-op, only the comment is outdated.

## Session 2: real-model isolation
- Recreated `repack_q40_test/perp.txt` (repeated CPU-cache paragraph, ~35 KB) in the
  repo for reusable PPL runs (the /tmp copy was lost).
- ROCM1 was full (27B server holding VRAM), so PPL runs used **ROCM3**.
- `llama-perplexity -m $M -ngl 99 -dev ROCM3 -f repack_q40_test/perp.txt -b 512 -c 512`:
    repack ON  (GEMM)    : PPL = 2,427,154   <-- still garbage
    --no-repack (generic): PPL = 1.2674       <-- correct
- Audited the device repack kernel (`repack_q4_0_kernel`) and its async launch
  (`set_tensor_async`): both CORRECT. `block_q4_0` is 18 B (d@0 + qs@2, no `m` field;
  confirmed via `static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0/2)` in
  ggml-common.h), matching the kernel's raw offsets and 18-byte stride.
- So the remaining bug is NOT the repack kernel, NOT the GEMM. It is in the async
  upload path (sync/scratch lifetime) or the real activation quantization
  (`quantize_mmq_q8_1_cuda`), which the harness does not exercise (the harness uses a
  simplified D4 quant + the synchronous host repack).

## Session 3: isolation + per-GEMM verification (bug narrowed, not yet found)

Isolation experiments (env-gated in `ggml_cuda_repack_tensor_supported`, `repack-common.cu`):
- `RP_NO_Q40=1` (Q4_0 -> generic, Q8_0 stays repack): PPL = 1.298 (CORRECT).
- `RP_NO_Q80=1` (Q4_0 stays repack, Q8_0 -> generic): PPL = 2,339,638 (BROKEN).
=> the fault is in the **Q4_0 repack** path, not Q8_0.

Per-GEMM verification: dumped w/xq/y for the first occurrence of each (ne0,ne1) shape
(`RP_DUMPG` in `mul-mat.cu`), then computed Yref = W_dequant @ X_dequant in Python and
diffed vs the GEMM output (dequant logic mirrors the C kernel):
| shape (ne0 x ne1) | tensor | max_abs | corr |
|-------------------|--------|---------|------|
| 4096 x 8192 | attn_qkv | 2e-5 | 1.0 |
| 4096 x 4096 | attn_gate / attn_output | 1e-5 | 1.0 |
| 4096 x 12288 | ffn_gate / ffn_up | 1e-5 | 1.0 |
| 4096 x 1024 | attn_v / attn_k | 1e-5 | 1.0 |
| 12288 x 4096 | ffn_down | 3e-5 | 1.0 |
=> every Q4_0 GEMM is a correct linear function of its (w, xq).

Also verified:
- Q4_0 weights are byte-exact on device (dumped the repacked buffer; matched the file's
  canonical repack computed from the GGUF).
- token_embd (Q4_0) is NOT in the repack buffer (its set_tensor does not route through the
  repack path) -> canonical, so not the bug.
- ffn_down float activation (src1) is finite and sane (std 0.84, absmax ~12.5, no NaN).

Key gap in the verification: the per-GEMM check proves the GEMM correctly computes W @ X
from the DUMPED xq. It does NOT prove the float activation (src1) is the correct hidden
state. If an earlier non-GEMM op emits a wrong hidden state, that becomes a wrong
activation for a later GEMM, which then "correctly" computes from the wrong input and
still passes the check. The ffn_down src1 being sane means the divergence is subtle, not a
wild memory fault.

Next suspects (non-GEMM ops not yet covered):
1. **SSM gated_delta_net recurrent scan** - the SSM blocks keep a recurrent state across
tokens. If the scan reads/writes state incorrectly (or a Q4_0/Q8_0 tensor is used in the
scan as canonical while stored two-plane), the hidden state diverges from some token on.
2. A Q4_0/Q8_0 weight used in a non-mul_mat op (read as canonical while stored two-plane).

## Cleanup (temporary debug code to REMOVE before any commit)
- `repack-common.cu`: `RP_NO_Q40` / `RP_NO_Q80` env gates in
  `ggml_cuda_repack_tensor_supported`.
- `mul-mat.cu`: `RP_TRACE` / `RP_DUMPG` / `RP_DUMPS1` debug blocks, plus the added
  `#include <cstdio>` / `<vector>` / `<set>`.
- `buffer.cu`: `RP_DUMP_EMBD` debug block (+ includes).
- KEEP: the f16 reference fixes in `repack_q40_test/test_gemm_q40.cu` (legitimate).

## How to verify (when resumed)

**Step 1 - PPL isolation (the key diagnostic).** repack-ON must match `--no-repack`
(~1.27 on the repeated-text `perp.txt`). Use ROCM3 (ROCM1 is often full from the 27B
server):
```sh
export HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  CCC_OVERRIDE_OPTIONS="^--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/12"
cmake --build build --config Release -j 36

M=/media/muselko/55734f1e-94a9-48f5-811a-ccdcf6d36011/llm/Qwen3.5-9B-Q4_0.gguf
timeout 400 ./build/bin/llama-perplexity -m $M -ngl 99 -dev ROCM3 -f repack_q40_test/perp.txt -b 512 -c 512
timeout 400 ./build/bin/llama-perplexity -m $M -ngl 99 -dev ROCM3 --no-repack -f repack_q40_test/perp.txt -b 512 -c 512
```
Current: ON = 2.4M (broken), OFF = 1.2674 (correct).

**Step 2 - once PPL matches, bit-exact token check, then bench:**
```sh
P="Explain how a CPU cache works in three sentences."
timeout 90 ./build/bin/llama-cli -m $M -ngl 99 -dev ROCM3 -p "$P" -n 60 -s 42 --temp 0 -st
timeout 90 ./build/bin/llama-cli -m $M -ngl 99 -dev ROCM3 -p "$P" -n 60 -s 42 --temp 0 -st --no-repack

./build/bin/llama-bench -m $M -ngl 99 -dev ROCM1 -p 4096 -n 0 -ub 2048
./build/bin/llama-bench -m $M -ngl 99 -dev ROCM1 -p 0 -n 128
./build/bin/llama-bench -m /media/.../Qwen3.5-9B-Q8_0.gguf -ngl 99 -dev ROCM1 -p 4096 -n 0 -ub 2048
```

## Next step (the one open thread)
The Q4_0 GEMMs are proven correct and the weights are byte-exact, so the fault is a wrong
float hidden state from a non-GEMM op. Next:
1. Compare the ffn_down float activation (src1) between the repack config and a CPU (or
   generic) reference to find the first token/layer where they diverge.
2. Inspect the SSM gated_delta_net recurrent scan ops (the non-GEMM ops not yet covered):
   confirm no repacked (two-plane) Q4_0/Q8_0 tensor is read as canonical in the scan, and
   that the recurrent state buffer is sized/synchronized correctly.

## Reference numbers (generic path, before this work)
| Model   | pp4096 (t/s) |
|---------|--------------|
| Q4_0    | 994.87       |
| Q8_0 (repack) | 1233.79 |
| Q8_0 (generic) | 948.48 |

Expected if the port works: Q4_0 ~1200-1350 t/s (+20-35% over 994).

## Notes
- Do NOT run bench/CLI without `timeout`.
- A lingering `llama-cli` can hold GPU1 VRAM and cause OOM; kill strays
  (`ps aux | grep llama-cli`) before re-running.
