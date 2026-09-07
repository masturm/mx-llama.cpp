# Port the Q8_0 repack GEMM to Q4_0 (gfx906)

## Goal

Close the Q4_0 prefill gap to Q8_0. Today:

| model | pp4096 | kernel |
|---|---|---|
| Q8_0 (repack GEMM) | 1234 | `mmq_gemm_repacked` |
| Q4_0 (generic mmq) | 994 | `mul_mat_q<Q4_0>` |

The Q8_0 win is a **dedicated, hand-tuned tiled GEMM** (two-plane weight layout,
software-pipelined K-loop, double buffering, swizzled LDS) that no 4-bit type gets.

## Why it should work (measured)

- The generic `mul_mat_q<Q4_0>` runs at **14.6% VALU**; the Q8_0 repack GEMM runs at
  **16.2% VALU**. Both are the *same* latency/scheduling-bound regime, not ALU-bound.
  So the +30% is from **latency hiding**, which transfers to any weight type.
- Q4_0 is a **4-bit type** -> it maps onto the existing **MXFP4 RAW_REG pattern**
  (packed nibbles in VRAM, widen to int8 in LDS, int8 `dp4a`). Only the widen table
  and the scale format differ.
- Q4_0 model (Qwen3.5-9B-Q4_0) tensor mix: **173 x Q4_0**, 48 x Q8_0, 4 x Q4_1,
  1 x Q6_K, 177 x F32. The 173 Q4_0 matmuls are the target.

Not using `v_dot8` (4-bit dot): the ALU is 84% idle, so 2x dot throughput buys
nothing, and it would force 4-bit *activations* (accuracy loss). Keep int8 `dp4a`.

## The Q4_0 repack layout

Canonical `block_q4_0` (34 B): `f16 d` + `qs[16]` (32 nibbles, byte b = values 2b,2b+1).

Repacked (mirrors MXFP4): a de-aliased **nibble plane** [ne1 x ne0/2 bytes] + an
**f16 scale plane** [ne1 x ne0/32 x 2 bytes]. The nibbles are re-ordered at upload so
the device widen (shared with MXFP4) yields the lo/hi split the GEMM's `dp4a` wants:

    target nibble byte b (0..15) = ( idx[16+b] << 4 ) | idx[b]

where `idx[p] = (qs[p/2] >> (4*(p%2))) & 0xf`. This makes `lo` = values 0..15 and
`hi` = values 16..31, exactly the Q8_0 convention.

Device widen (same shape as `rp_mxfp4_expand`, different table):

    kvalues_q4_0[16] = {0,1,2,3,4,5,6,7,-8,-7,-6,-5,-4,-3,-2,-1}   // sign-extend
    rp_q4_0_expand(nib, lo, hi): get_int_from_table_16 per uint32, lo = evens, hi = odds

The GEMM stages raw nibbles in registers and widens **once at the LDS store** (same as
MXFP4; widening at consume measured -40%). LDS stays int8-sized (30 KB, 2 blocks/CU),
so occupancy is unchanged from Q8_0; VRAM is half (packed 4-bit).

## Changes (one small edit per existing file)

1. `q8_repack/repack-common.cuh`
   - `repack_qs_row_stride`: Q4_0 uses `ne0/2` (nibble rows) like MXFP4.
   - `repack_qs_bytes`: Q4_0 -> 16.
   - `repack_gcn_nbytes`: add Q4_0 case = `ne1*(qs_stride(ne0/2) + (ne0/32)*2)`.
     (`repack_scale_row_bytes` already yields 2 B f16 for Q4_0 - no change.)
   - `rp_traits<GGML_TYPE_Q4_0>`: `raw_lds=true`, geom (ne0/2 rows), `load_w` (widen),
     `load_w_raw` (nibble + f16 scale), `scale` = `__half2float`, `expand` = `rp_q4_0_expand`.
   - `rp_mxfp4_expand` gains a sibling `rp_q4_0_expand` + `kvalues_q4_0`.
   - add an `expand` entry to both `rp_traits` specializations (see #2).
   - `RP_FOREACH_TYPE`: add `X(GGML_TYPE_Q4_0)`.

2. `q8_repack/repack-kernels.cuh`
   - `store_stage`: replace the hardcoded `rp_mxfp4_expand(pw_lo[i], lo, hi)` with
     `rp_traits<WT>::expand(pw_lo[i], lo, hi)` (type-generic widen).

3. `q8_repack/repack-common.cu`
   - `ggml_cuda_repack_tensor_supported`: add `case GGML_TYPE_Q4_0` (ne0%32==0).
   - `repack_q4_0_host`: canonical -> nibble plane (re-ordered) + f16 scale plane.
   - `repack_host`: add Q4_0 case.
   - (optional) `ggml_cuda_repack_mmv_fusion_supported`: allow Q4_0 for fused decode FFN.

4. `q8_repack/buffer.cu`
   - `repack_q4_0_kernel`: device repack for async upload (same re-order as host).
   - `ggml_cuda_repack_set_tensor_async`: add Q4_0 case.

5. `q8_repack/mul-mat.cu` (dense dispatch)
   - slice, `ne11==1`: `mul_mat_vec_rp<Q4_0,16,16,false>` (like MXFP4).
   - slice, `ne11>=128`: `mmq_gemm_repacked<false,TN,nrl,Q4_0>`.
   - slice, `9<=ne11<128`: `mmq_gemm_repacked_w32<false,1,nrl*2,Q4_0>`.
   - narrow (2..8): `ggml_cuda_mul_mat_repacked_nc_t<Q4_0>`.
   - fused MMV: `mul_mat_vec_rp<Q4_0,...,true>` (optional, Phase 2).

(Phase 2, optional: `mul-mat-id.cu` MoE dispatch, and the fused decode FFN.
Qwen3.5-9B is dense, so the benchmark does not need MoE.)

## Expected result

Q4_0 repack GEMM should reach ~1200-1350 t/s (Q8_0 repack is 1234; Q4_0 has half the
weight bytes but a small widen cost, and the ALU is idle). That is **+20-35%** over the
current 994 t/s. Decode should be ~parity-or-better (two-plane scale locality helps the
mat-vec; it is memory-bound at 4-bit either way).

## Verification

1. Build (gfx906). No new warnings.
2. **Bit-exact**: run the Q4_0 model with the repack on and with `--no-repack 1`,
   same prompt + seed; generated text must be identical. (Catches a re-order or widen bug.)
3. `llama-bench` pp256/512/1024/2048/4096 + tg128, Q4_0, before/after.
4. Confirm no regression on Q8_0 (shared `store_stage` change) and on a K-quant model.
5. rocprofv3: confirm the Q4_0 GEMM now shows as `mmq_gemm_repacked` and check its VALU/LDS.

## Risks

- **Re-order / widen correctness**: the whole thing hinges on the nibble re-order +
  widen matching the GEMM's lo/hi convention. Mitigated by the bit-exact check (#2).
- **Decode regression**: if `mul_mat_vec_rp<Q4_0>` is slower than the generic mmvq,
  carve Q4_0 decode out to the canonical path (dispatch is per-matmul, so feasible).
- **Scope**: ~100-150 lines across 5 files, all following the existing MXFP4 pattern.
  Self-contained to `q8_repack/` + the dispatch; no generic-kernel changes.
