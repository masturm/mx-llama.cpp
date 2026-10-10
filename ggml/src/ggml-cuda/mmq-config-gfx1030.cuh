// gfx1030 (Navi 21 / RDNA2, wave32) MMQ config.
//
// Perf-only knobs (tile width J) - results are bit-exact vs the stock rdna2
// config, which is what upstream routes gfx1030 through. Include after
// mmq-config-rdna2.cuh.
//
// Widening J past 64 halves the redundant weight reads when a batch spans more
// than one J-tile (ntiles_x = ceil(batch/J)). On RDNA2 the win is type-dependent
// (measured on Qwen3.5-9B pp512): the affine K-quants (Q4_K/Q5_K, Q8_1 layout)
// gain ~10%, but the LUT-based IQ types regress because their tile_x is ~2x and
// the wide tile drops occupancy. So J is widened only for the types that gain.
//
// The wide-J occupancy gate in mul_mat_q_switch_J holds J<=64 whenever the tile
// grid can no longer fill the CUs (row-sharded / MoE shapes).
//
// RDNA2 is always wave32, so nwarps = nthreads/32 = 8 here. Two constraints on J,
// both stricter than the wave64 (gfx906) path where nwarps is half:
//   - J >= nwarps, else sum[j0/nwarps*...] and float2 y_df[J/nwarps] go to zero.
//   - J % nwarps == 0, else the sum array (sized J*I/(nwarps*warp_size)) is
//     indexed past its end by the last partial j-group.
// With nthreads=256 (nwarps=8) the stock J%8 rule already covers both.

#ifndef GFX1030_NTHREADS
#define GFX1030_NTHREADS 256
#endif

// Per-type wide-J ceiling. 64 == stock rdna2 (no change); 128 == wide tile.
#ifndef GFX1030_JMAX_KQ   // Q4_K, Q5_K (Q8_1 layout)
#define GFX1030_JMAX_KQ   128
#endif
#ifndef GFX1030_JMAX_Q6K
#define GFX1030_JMAX_Q6K  128
#endif
#ifndef GFX1030_JMAX_Q8_0
#define GFX1030_JMAX_Q8_0 64
#endif
#ifndef GFX1030_JMAX_IQ   // IQ*, Q2_K, Q3_K, MXFP4
#define GFX1030_JMAX_IQ   64
#endif

#define GFX1030_NWARPS (GFX1030_NTHREADS / 32)
#define GFX1030_J_MIN  (GFX1030_NWARPS < 8 ? 8 : GFX1030_NWARPS)
// J must be a multiple of both 8 (upstream tiling) and nwarps (sum-array index).
#define GFX1030_J_STEP (GFX1030_NWARPS > 8 ? GFX1030_NWARPS : 8)

static constexpr __host__ __device__ bool ggml_cuda_mmq_gfx1030_j_ok(int J, int jmax) {
    return J >= GFX1030_J_MIN && J <= jmax && (J % GFX1030_J_STEP) == 0;
}

static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_gfx1030(ggml_type type, int J, bool fallback) {
    if (type == GGML_TYPE_Q8_0 && ggml_cuda_mmq_gfx1030_j_ok(J, GFX1030_JMAX_Q8_0)) {
        return ggml_cuda_mmq_config(
            GGML_TYPE_Q8_0, GFX1030_NTHREADS, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, fallback);
    }
    if (ggml_cuda_mmq_gfx1030_j_ok(J, GFX1030_JMAX_KQ)) {
        switch (type) {
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ggml_cuda_mmq_config(
                    type, GFX1030_NTHREADS, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, fallback);
            default:
                break;
        }
    }
    if (ggml_cuda_mmq_gfx1030_j_ok(J, GFX1030_JMAX_Q6K) && type == GGML_TYPE_Q6_K) {
        return ggml_cuda_mmq_config(
            type, GFX1030_NTHREADS, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, fallback);
    }
    if (ggml_cuda_mmq_gfx1030_j_ok(J, GFX1030_JMAX_IQ)) {
        switch (type) {
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ4_XS:
            case GGML_TYPE_IQ4_NL:
                return ggml_cuda_mmq_config(
                    type, GFX1030_NTHREADS, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, fallback);
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
                return ggml_cuda_mmq_config(
                    type, GFX1030_NTHREADS, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, fallback);
            case GGML_TYPE_Q2_K:
                return ggml_cuda_mmq_config(
                    type, GFX1030_NTHREADS, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K, MMQ_ITER_K, false, fallback);
            case GGML_TYPE_Q3_K:
                return ggml_cuda_mmq_config(
                    type, GFX1030_NTHREADS, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, fallback);
            case GGML_TYPE_MXFP4:
                return ggml_cuda_mmq_config(
                    type, GFX1030_NTHREADS, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, fallback);
            default:
                break;
        }
    }
    return ggml_cuda_mmq_get_config_rdna2(type, J, fallback);
}
