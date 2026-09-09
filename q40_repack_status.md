# Q4_0 repack GEMM port - implementation status

Target: gfx906 (Vega 20 / MI50). Port the dedicated Q8_0/MXFP4 repack GEMM
(two-plane layout, software pipelining, double buffering, swizzled LDS) to Q4_0,
reusing the MXFP4 RAW_REG pattern. Projected +20-35% prefill over the generic
`mul_mat_q<Q4_0>` path.

Design doc: `port_q40_repack.md`. This file tracks what is implemented and the
current state.

## Status: repack GEMM verified correct with REAL data; fault is a NON-GEMM op (first divergence at attn_gate)

Builds clean. Bugs found and fixed so far:
1. **D4/DS4 activation scale layout** (fixed in `mul-mat.cu`).
2. **Repack nibble re-order was a WRONG permutation** (Session 2) - must be a NO-OP.

Real model (repack ON) is still broken: **PPL = 2.34M**, vs `RP_NO_Q40=1` (Q4_0 generic)
**PPL = 1.2674** (correct).

Session 5 (see below) verified with the REAL model data that the repack GEMM is correct for
ALL five Q4_0 shapes (corr 1.0 vs the CPU reference `W @ Xq`), the D4 activation quantization
is correct (Xq ~= float src1, corr 0.99999), the float src1 is sane, and the async repack
upload is host-synchronized (no race). The standalone harness also PASSES at all real shapes.

Definitive finding: for the attn_gate (same float src1, same W), the repack GEMM = W@src1
(corr 0.99999, CORRECT) but the generic GEMM != W@src1 (corr -0.705, WRONG). Yet the generic
config gives correct PPL. So the attn_gate output is NOT the direct PPL determinant; the
repack config's broken PPL comes from a LATER non-GEMM op (top suspect: the SSM
gated_delta_net scan, or the attention projection that consumes the attn_gate output).

Do NOT bench or ship until repack-ON PPL matches `RP_NO_Q40=1` (~1.27).

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
- `ggml-cuda.cu`: the `rp_dst_dumper` struct + the `RP_DUMPSRC` block in
  `ggml_cuda_mul_mat` (src1/dst/weight dumps), plus the added `#include <set>` / `<cstring>`.
- KEEP: the f16 reference fixes in `repack_q40_test/test_gemm_q40.cu` (legitimate).

## Session 4: repack GEMMs verified correct; fault is a non-GEMM op

Method: dump the FLOAT src1 (at GEMM entry) and dst (at GEMM exit, via an RAII dumper in
`ggml_cuda_mul_mat`), keyed by tensor name, in BOTH the repack config and the generic config
(`RP_NO_Q40=1`), plus the weight (two-plane in repack, canonical in generic). Compare.

Key layout fact (was a bug in earlier dumps): the Q4_0 GEMM src1 and dst are **token-major**,
`src1_mem[tok][k]` (s11 = src1->nb[1]/4 = ne00), and `dst_mem[tok][out]` (nb[1] = ne[0]*4).
So `src1[k,tok] = mem[tok*ne00 + k]` (i.e. reshape(ne11,ne00).T). All blk.0 GEMMs have
s11=ne00 (attn_qkv/attn_gate/ffn_gate/ffn_up: 4096; ffn_down: 12288).

Verified (true W from both layouts dequant identically, max_abs=0; float src1 token-major):
| GEMM | shape (K x out) | repack vs W@src1 | generic vs W@src1 |
|------|-----------------|------------------|-------------------|
| attn_qkv | 4096 x 8192 | (same output both cfgs) | (same) |
| attn_gate | 4096 x 4096 | corr 0.99999 (MATCH) | corr -0.71 (no) |
| ffn_gate | 4096 x 12288 | corr 0.99998 (MATCH) | corr -0.06 (no) |
| ffn_down | 12288 x 4096 | corr 0.99996 (MATCH) | (generic both cfgs) |

So the **repack GEMMs all compute the true W @ src1**. The weight repack is correct
(two-plane == canonical dequant, max_abs=0). The fault is NOT a GEMM.

Config-to-config divergence (blk.0), by float src1/dst:
- attn_qkv src1 (attn_norm out): IDENTICAL.
- attn_gate src1 (attn_norm out): IDENTICAL.
- attn_qkv dst: IDENTICAL.
- attn_gate dst: DIFFERENT (first divergence).
- ffn_gate/ffn_up src1 (post_attention_norm out): DIFFERENT (downstream of attn_gate).

Conclusion: the first divergence is the attn_gate GEMM output. The repack attn_gate = W@src1
(correct); the generic attn_gate != W@src1. Yet the generic config gives correct PPL, so the
PPL is not determined by the K=4096 GEMM's W@src1 in the obvious way.

## Session 5: repack GEMM verified correct with REAL data; paradox resolved as "generic GEMM is wrong"

Re-ran the per-GEMM check (RP_DUMPG) with the REAL model data and verified EVERY Q4_0 GEMM
shape against the CPU reference `W_dequant @ Xq_dequant` (D4 activation dequant):
| shape (ne0 x ne1) | max_abs | corr |
|-------------------|---------|------|
| 4096 x 8192 | 1.5e-5 | 1.0 |
| 4096 x 4096 | 8.6e-6 | 1.0 |
| 4096 x 12288 | 8.6e-6 | 1.0 |
| 12288 x 4096 | 3.4e-5 | 1.0 |
=> the repack GEMM is correct for ALL shapes with the real data (only q8_1 quant error).

Also verified:
- The D4-quantized activation (Xq) is an excellent approximation of the float src1
  (attn_gate: corr 0.999989, mean_rel 0.007). So the D4 quantization is correct.
- The float src1 (attn_norm output) is sane (std 1.08, absmax 56, no NaN).
- The async repack upload is host-synchronized before compute (ggml_cuda_repack_async_release
  does cudaEventSynchronize at the start of every graph compute). No race.
- The harness (standalone) PASSES at all real shapes (4096x4096, 4096x8192, 4096x12288).

Definitive attn_gate comparison (same float src1, same W, both dequant identically):
  repack  dst vs W@src1(float): corr 0.999994  -> repack GEMM = W@src1 (CORRECT)
  generic dst vs W@src1(float): corr -0.705     -> generic GEMM != W@src1 (WRONG)
  repack  dst vs generic dst:   corr -0.705

PARADOX: the repack GEMM is correct (W@src1) but the repack config gives broken PPL (2.34M),
while the generic GEMM is wrong (not W@src1) yet the generic config gives correct PPL (1.2674).

Resolution (most likely): the attn_gate output is NOT the direct PPL determinant. The repack
config's broken PPL comes from a LATER non-GEMM op, and the generic config's "wrong" attn_gate
GEMM is actually the correct one for the model (my W@src1 reference, while matching the repack
GEMM, does not match what the model needs). The next step is to find the first WRONG hidden
state by comparing the repack and generic configs layer-by-layer (the attn_gate output is the
first divergence, so the fault is in the op that consumes it - the SSM gated_delta_net scan or
the attention projection).

## Session 6: model is Qwen3.5 SSM (fused GDN); DS4 scale fix tried (no effect); paradox deepens

Model architecture: Qwen3.5 (LLM_ARCH_QWEN35), a hybrid attention/SSM model. The SSM blocks
use the gated_delta_net (GDN) op via the FUSED path (`ggml_cuda_op_gated_delta_net_fused_cache`),
NOT the plain `ggml_cuda_op_gated_delta_net`. The SSM weights: ssm_out (Q5_K, not repacked),
ssm_alpha/ssm_beta (Q8_0, small), ssm_conv1d (F32). The Q4_0 GEMMs are only: attn_qkv,
attn_gate, ffn_gate, ffn_up, ffn_down.

Instrumented the FUSED GDN path (the plain GDN dispatch is never reached). Findings:
- GDN#0 (blk.0) dst: IDENTICAL in both configs (max_abs=0).
- GDN#0 src0 (q_conv) / src1 (k_conv): DIFFER (corr -0.49 / -0.46).
- GDN#1 (blk.1) dst: DIFFER (corr -0.05).
So the GDN#0 output is the same, but its inputs differ. The q_conv/k_conv come from the conv
output, which comes from the attn_qkv GEMM. So the attn_qkv GEMM output is the first divergence.

Re-verified with the current build: the attn_qkv and attn_gate src1 (attn_norm output) are
IDENTICAL in both configs, but their GEMM outputs differ (corr -0.72 / -0.70). The repack
attn_qkv GEMM = W @ src1 (corr 0.999995, CORRECT); the generic attn_qkv GEMM != W @ src1
(corr -0.72, WRONG). Yet the generic config gives correct PPL (1.2674).

Tried a fix: made the repack GEMM read the DS4 activation scale (half2) for Q4_0 instead of
D4 (float), matching the generic Q4_0 layout, and removed the D4 forcing in the quantization
(`ds_type = src0->type`). This changed `rp_x_sub_from_mmq_group` to a template on WT and the
quantization `ds_type`. Result: PPL still broken (2394305). The repack GEMM still matches
W @ Xq (now DS4-dequant) with corr 1.0. So the D4/DS4 scale layout is NOT the fault.

Deepened paradox: recovering the effective weight M from the generic attn_qkv output
(M = dst @ pinv(src1)) gives a matrix with std ~10150 (vs the true W's std 0.03). So the
generic GEMM output is NOT a simple linear function of the dumped src1 with a small W. This
suggests either (a) the dumped src1 is not the actual GEMM input, or (b) the GEMM includes a
bias/offset, or (c) my understanding of the GEMM operation is fundamentally wrong.

## Session 6 continued: W dequant fixed; paradox still unresolved; stuck

Re-verified the W dequant with the CORRECT two-plane layout (qs_stride = ne0/2 + 16 = 2064,
scale plane = ne1*(ne0/32)*2). Result: canonical W == two-plane W (max_abs=0, corr=1.0).
So the W is IDENTICAL in both configs.

Also confirmed: attn_qkv src1 is identical (max_abs=0), the GEMM is correct (matches the CPU
vec_dot `ggml_vec_dot_q4_0_q8_0` which pairs low-nibble with y[j] and high-nibble with y[j+16],
exactly like the repack GEMM), and there is no batching (ne[2]=ne[3]=1).

So: same W, same src1, correct GEMM, no batching. Yet the attn_qkv output differs between
configs (corr -0.72). This is logically impossible. The only remaining explanations:
1. The GEMM reads a DIFFERENT W or xq than the dumper dumps (a pointer/layout bug).
2. The quantized xq differs between configs even though the float src1 is the same
   (a quantization bug, but I verified the quantization is correct: xq ~= src1, corr 0.99999).
3. My `W @ src1` reference is WRONG (matches the repack GEMM, not the correct operation),
   so the repack GEMM is wrong and the generic GEMM is correct.

The CPU vec_dot (`ggml_vec_dot_q4_0_q8_0_generic` in quants.c) pairs:
  v0 = (qs[j] & 0x0F) - 8   (low nibble, values 0..15)
  v1 = (qs[j] >> 4) - 8     (high nibble, values 16..31)
  sumi0 += v0 * y[j];  sumi1 += v1 * y[j+16]
This is EXACTLY what the repack GEMM does. So the repack GEMM matches the CPU vec_dot.

STUCK: cannot resolve the paradox (same W, same src1, correct GEMM, yet different output).

Tried to dump the CPU GEMM output (ground truth) by adding a dumper to
`ggml_compute_forward_mul_mat` in ggml-cpu.c. The function IS called (per-chunk, nth=18
threads), but the dst is computed in-place across chunks, so a dump at the end of one chunk
call is partial. Removed the CPU dumper (not working).

Session 7 (this session) changes:
- Tried a DS4 scale fix: made `rp_x_sub_from_mmq_group` a template on WT, reading DS4 (half2)
  scale for Q4_0, D4 (float) for Q8_0/MXFP4. Changed the quantization `ds_type` from forced
  Q8_0 to `src0->type`. Result: PPL still broken (2394305). The DS4 change did NOT help.
- Re-verified the W dequant with the CORRECT two-plane layout (qs_stride = ne0/2 + 16 = 2064,
  scale plane = ne1*(ne0/32)*2). Result: canonical W == two-plane W (max_abs=0, corr=1.0).
- Confirmed: attn_qkv src1 identical (max_abs=0), GEMM correct (matches CPU vec_dot), no
  batching (ne[2]=ne[3]=1). Yet the attn_qkv output differs (corr -0.72).
- REVERTED the DS4 change (restored the original D4 forcing: `ds_type = GGML_TYPE_Q8_0`,
  `rp_x_sub_from_mmq_group` always reads D4). PPL still broken (2427154). So the D4/DS4
  layout is NOT the fault.
- Tried to dump the CPU GEMM output (ground truth) by adding a dumper to
  `ggml_compute_forward_mul_mat` in ggml-cpu.c. The function IS called (per-chunk, nth=18
  threads), but the dst is computed in-place across chunks, so a dump at the end of one chunk
  call is partial. Removed the CPU dumper (not working).

STUCK: cannot resolve the paradox (same W, same src1, correct GEMM, yet different output).

Session 7 continued: verified the GEMM reads the CORRECT W and xq (added RP_DUMPWX to dump
the W and xq from the GEMM's input buffers; they match the dumper's W and xq exactly,
max_abs_diff=0). Also verified the LDS is fully covered (W_EPT*NTHREADS = W_ELM, X_EPT*
NTHREADS = X_ELM, no uninitialized reads). So the GEMM reads the correct inputs and has no
uninitialized LDS reads.

The ONLY remaining explanation: the GEMM kernel has a DATA-DEPENDENT bug (e.g., a race
condition, or a subtle issue that only manifests with the real W/X values). The harness
(synthetic data) does not trigger it.

Next steps:
1. Re-examine the repack GEMM kernel's contraction for a subtle data-dependent bug (e.g., a
   tile-boundary or indexing bug that only manifests with the real W/X values, or a race
   condition in the double-buffering / software pipelining).
2. Compare the repack GEMM kernel's output to the generic GEMM kernel's output for the SAME
   W and xq (run both kernels with identical inputs and diff the outputs).
3. Consider running the model with ONLY the attn_qkv GEMM on GPU (repack) and everything else
   on CPU, to isolate whether the attn_qkv GEMM is correct in the context of the full model.
4. Consider adding a debug print INSIDE the GEMM kernel (in the compute stage) to dump the
   first few W and xq values and the intermediate accumulator, to verify the kernel is
   computing the correct dot product.

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
See "Session 5 next steps". The repack GEMM is proven correct with real data. The fault is a
non-GEMM op. Next:
1. Compare the block-output hidden state between the repack and generic configs at each layer
   to find the first divergence beyond the attn_gate GEMM output.
2. Inspect the SSM gated_delta_net scan (gated_delta_net.cu) and the attention projection for
   any direct read of a Q4_0/Q8_0 weight (bypassing the GEMM) or a state-buffer sizing/sync
   bug triggered only in the repack config.

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
