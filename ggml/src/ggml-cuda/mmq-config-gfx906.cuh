// gfx906 (Vega20 / MI50, wave64) MMQ config.
//
// Perf-only knobs (nthreads / tile widths) - results are bit-exact vs rdna2,
// which is what upstream routes gfx906 through. Include after mmq-config-rdna2.cuh.
//
// Q8_0: 8 warps (nthreads 512, vs rdna2's 4 = 256), and offer tile widths up to
// J=128 (rdna2 caps its Q8_0 table at 64). The wide tiles are only SELECTED when
// they keep the CUs busy - see the occupancy gate in mul_mat_q_switch_J
// (mmq.cuh), which holds J<=64 for row-sharded (-sm tensor) and MoE shapes and
// lets full-row shapes (1-GPU, -sm layer) take the wider tile.
//
// IQ* and K-quant types: same treatment as Q8_0 - 8 warps and J up to 128. On
// gfx906 the dequant (grid LUT + signs, or the K-quant scales/mins) happens in
// the tile load, so wider tiles and more warps amortize it the same way as for
// Q8_0. Each type keeps its rdna2 SRAM layout; the sram size check in
// mul_mat_q_switch_J caps J where needed (e.g. the wider Q3_K / Q6_K layouts do
// not fit J=128).
//
// MXFP4: 8 warps as well - rdna2's table with nthreads overridden, so tile
// widths and layout stay in sync with upstream.
static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_gfx906(ggml_type type, int J, bool fallback) {
    if (type == GGML_TYPE_Q8_0 && J >= 8 && J <= 128 && (J % 8) == 0) {
        return ggml_cuda_mmq_config(
            GGML_TYPE_Q8_0, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, fallback);
    }
    if (J >= 8 && J <= 128 && (J % 8) == 0) {
        switch (type) {
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ4_XS:
            case GGML_TYPE_IQ4_NL:
                return ggml_cuda_mmq_config(
                    type, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, fallback);
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
                return ggml_cuda_mmq_config(
                    type, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, fallback);
            // K-quants: same treatment, each keeps its rdna2 SRAM layout. Q6_K is the
            // widest (6.5625 bpw) so the sram check in mul_mat_q_switch_J caps its J.
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ggml_cuda_mmq_config(
                    type, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, fallback);
            case GGML_TYPE_Q2_K:
                return ggml_cuda_mmq_config(
                    type, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K, MMQ_ITER_K, false, fallback);
            case GGML_TYPE_Q3_K:
                return ggml_cuda_mmq_config(
                    type, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, fallback);
            case GGML_TYPE_Q6_K:
                return ggml_cuda_mmq_config(
                    type, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, fallback);
            default:
                break;
        }
    }
    if (type == GGML_TYPE_MXFP4) {
        const ggml_cuda_mmq_config rdna2 = ggml_cuda_mmq_get_config_rdna2(type, J, fallback);
        if (rdna2.type == GGML_TYPE_COUNT) {
            return rdna2;
        }
        return ggml_cuda_mmq_config(
            rdna2.type, 512, rdna2.occupancy, rdna2.I, rdna2.J, rdna2.sram_layout, rdna2.K_vram, rdna2.stream_k, rdna2.fallback);
    }
    return ggml_cuda_mmq_get_config_rdna2(type, J, fallback);
}
