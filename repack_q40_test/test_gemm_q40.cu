// Standalone test harness for the repack GEMM (Q8_0 and Q4_0).
//
// Isolates mmq_gemm_repacked<...,WT> from the rest of llama.cpp:
//   1. build a known weight W [ne1 x ne0] (canonical blocks, type WT),
//   2. repack it (host) to the repacked layout,
//   3. build a known float activation X [ne0 x n_tok] and quantize it to the
//      q8_1 MMQ D4 layout the GEMM consumes,
//   4. run the GEMM on GPU,
//   5. diff vs a CPU reference that uses the SAME dequantized W and X.
//
// Q8_0 is known-correct in the real model, so it validates the harness itself.
// If Q8_0 PASSes and Q4_0 FAILs, the bug is in the Q4_0 GEMM path (weight load
// / widen / scale / dp4a), not in the harness.
//
// Build:  ./repack_q40_test/build.sh
// Run:    ./repack_q40_test/test_gemm_q40 [q8|q40]   (default: both)

#include "q8_repack/repack-kernels.cuh"
#include "q8_repack/repack-common.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <functional>

// Stand-in for the common.cu helper that CUDA_CHECK calls on failure.
void ggml_cuda_error(const char * stmt, const char * func, const char * file,
        int line, const char * msg) {
    fprintf(stderr, "CUDA error: %s\n  stmt: %s\n  func: %s\n  file: %s:%d\n", msg, stmt, func, file, line);
    exit(1);
}

// Stubs for the ggml.c f16 conversions (avoid linking the whole ggml lib).
// ggml_fp16_t is uint16_t holding f16 BITS. __half2float(uint16_t) would value-
// convert (e.g. 0x9d80 -> 40320), so bit-cast into a __half first.
float ggml_fp16_to_fp32(ggml_fp16_t x) {
    __half h; memcpy(&h, &x, sizeof(h)); return __half2float(h);
}
ggml_fp16_t ggml_fp32_to_fp16(float x) {
    __half h = __float2half(x); ggml_fp16_t r; memcpy(&r, &h, sizeof(r)); return r;
}
// Local, unambiguously-correct f16-bit <-> float for the CPU reference. Used
// directly (not via the ggml_fp16_to_fp32 symbol) so the reference can never be
// corrupted by a value-conversion stub.
static inline float f16bits_to_f32(uint16_t bits) {
    __half h; memcpy(&h, &bits, 2); return __half2float(h);
}
static inline uint16_t f32_to_f16bits(float x) {
    __half h = __float2half(x); uint16_t r; memcpy(&r, &h, 2); return r;
}
[[noreturn]] void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "abort %s:%d\n", file, line); exit(1);
}

// ---------------------------------------------------------------------------
// Per-weight-type helpers: block type, random generation, host repack, and
// CPU dequant. The GEMM and activation path are shared.
// ---------------------------------------------------------------------------
template <ggml_type WT>
struct wt_op;

template <>
struct wt_op<GGML_TYPE_Q8_0> {
    using block_t = block_q8_0;
    static void make(std::mt19937 & rng, std::vector<block_t> & W, int ne0, int ne1) {
        W.resize((size_t) ne1 * (ne0 / 32));
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto & b : W) {
            b.d = __float2half(dist(rng) + 0.5f);
            for (int i = 0; i < 32; i++) b.qs[i] = (int8_t) ((int) (rng() % 255) - 127);
        }
    }
    static void repack(const block_t * blocks, uint8_t * dst, int64_t ne0, int64_t ne1) {
        const int64_t n_blocks = ne0 / 32;
        const int64_t qs_str   = repack_qs_row_stride(GGML_TYPE_Q8_0, ne0);
        const size_t  qs_len   = (size_t) ne1 * qs_str;
        memset(dst, 0, qs_len + (size_t) ne1 * n_blocks * 2);
        for (int64_t row = 0; row < ne1; row++)
            for (int64_t blk = 0; blk < n_blocks; blk++) {
                const block_t * b = &blocks[row * n_blocks + blk];
                memcpy(dst + (size_t) row * qs_str + (size_t) blk * 32, b->qs, 32);
                memcpy(dst + qs_len + (size_t) (row * n_blocks + blk) * 2, &b->d, 2);
            }
    }
    static void dequant(const block_t * W, float * Wq, int ne0, int ne1) {
        for (int i = 0; i < ne1; i++)
            for (int k = 0; k < ne0; k++) {
                const block_t * b = &W[i * (ne0 / 32) + (k / 32)];
                Wq[k + i * ne0] = __half2float(b->d) * (float) b->qs[k % 32];
            }
    }
};

template <>
struct wt_op<GGML_TYPE_Q4_0> {
    using block_t = block_q4_0;
    static void make(std::mt19937 & rng, std::vector<block_t> & W, int ne0, int ne1) {
        W.resize((size_t) ne1 * (ne0 / 32));
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto & b : W) {
            b.d = __float2half(dist(rng) + 0.5f);
            for (int i = 0; i < 16; i++) b.qs[i] = (uint8_t) (rng() & 0xff);
        }
    }
    // Nibble re-order so byte j holds value j (low) and value 16+j (high); the
    // shared rp_q4_0_expand then yields lo = 0..15, hi = 16..31.
    static void repack(const block_t * blocks, uint8_t * dst, int64_t ne0, int64_t ne1) {
        const int64_t n_blocks = ne0 / 32;
        const int64_t qs_str   = repack_qs_row_stride(GGML_TYPE_Q4_0, ne0);
        const size_t  qs_len   = (size_t) ne1 * qs_str;
        memset(dst, 0, qs_len + (size_t) ne1 * n_blocks * 2);
        for (int64_t row = 0; row < ne1; row++)
            for (int64_t blk = 0; blk < n_blocks; blk++) {
                const block_t * b  = &blocks[row * n_blocks + blk];
                uint8_t * d_qs     = dst + (size_t) row * qs_str + (size_t) blk * 16;
                // Q4_0 original byte j already holds val[j] (low) + val[j+16] (high),
                // which is exactly the GEMM pairing (lo[j]=val[j], hi[j]=val[j+16]).
                for (int j = 0; j < 16; j++) d_qs[j] = b->qs[j];
                memcpy(dst + qs_len + (size_t) (row * n_blocks + blk) * 2, &b->d, 2);
            }
    }
    static void dequant(const block_t * W, float * Wq, int ne0, int ne1) {
        for (int i = 0; i < ne1; i++)
            for (int k = 0; k < ne0; k++) {
                const block_t * b = &W[i * (ne0 / 32) + (k / 32)];
                const int   v   = k % 32;
                const int   nib = (v < 16) ? (b->qs[v] & 0xf) : ((b->qs[v - 16] >> 4) & 0xf);
                const int8_t q  = (int8_t) ((nib < 8) ? nib : (nib - 16));
                Wq[k + i * ne0] = __half2float(b->d) * (float) q;
            }
    }
};

// ---------------------------------------------------------------------------
// Activation: quantize X [ne0 x n_tok] (K contiguous) to the q8_1 MMQ D4 layout
// the GEMM consumes. y[(k/128)*n_tok + tok].qs[k%128], scale .d4[(k%128)/32].
// Mirrors quantize_mmq_q8_1 (D4 layout).
// ---------------------------------------------------------------------------
static void quantize_x_mmq_d4(const float * X, block_q8_1_mmq * y, int ne0, int n_tok) {
    memset(y, 0, sizeof(block_q8_1_mmq) * (size_t)(ne0 / 128) * n_tok);
    for (int tok = 0; tok < n_tok; tok++)
        for (int sb = 0; sb < ne0 / 32; sb++) {
            const int k_block = sb / 4;
            const int s       = sb % 4;
            float amax = 0.0f;
            for (int v = 0; v < 32; v++) amax = fmaxf(amax, fabsf(X[sb * 32 + v + tok * ne0]));
            if (amax == 0.0f) amax = 1.0f;
            const float d_inv = 127.0f / amax;
            const float d     = 1.0f / d_inv;
            for (int v = 0; v < 32; v++)
                y[k_block * n_tok + tok].qs[s * 32 + v] = (int8_t) roundf(X[sb * 32 + v + tok * ne0] * d_inv);
            y[k_block * n_tok + tok].d4[s] = d;
        }
}

static void dequant_x_mmq_d4(const block_q8_1_mmq * y, float * Xq, int ne0, int n_tok) {
    for (int tok = 0; tok < n_tok; tok++)
        for (int k = 0; k < ne0; k++) {
            const int k_block = k / 128;
            const int iqs     = k % 128;
            const int s       = iqs / 32;
            Xq[k + tok * ne0] = (float) y[k_block * n_tok + tok].qs[iqs] *
                                y[k_block * n_tok + tok].d4[s];
        }
}

// CPU reference: Y[tok][i] = sum_k Wq[i][k] * Xq[k][tok]. Output is token
// contiguous with token stride ne1 (matches the GEMM epilogue, dst_s1 = ne1).
static void gemm_cpu_ref(const float * Wq, const float * Xq, float * Y,
        int ne0, int ne1, int n_tok) {
    for (int tok = 0; tok < n_tok; tok++)
        for (int i = 0; i < ne1; i++) {
            float sum = 0.0f;
            for (int k = 0; k < ne0; k++) sum += Wq[k + i * ne0] * Xq[k + tok * ne0];
            Y[tok * ne1 + i] = sum;
        }
}

// ---------------------------------------------------------------------------
// Widen-in-isolation probe: run rp_q4_0_expand on the GPU against a known
// nibble and check the int8 output. Byte j of the nibble is (high_j << 4) | low_j;
// the widen must yield lo[j] = signext(low_j) and hi[j] = signext(high_j).
// ---------------------------------------------------------------------------
static __global__ void widen_probe_kernel(const uint4 * nib, uint4 * lo, uint4 * hi) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    rp_traits<GGML_TYPE_Q4_0>::expand(nib[0], lo[0], hi[0]);
#else
    (void) nib; (void) lo; (void) hi; NO_DEVICE_CODE;
#endif
}

static int run_widen_test() {
    const int8_t kvalues[16] = { 0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1 };
    uint4 nib; uint8_t * nb = (uint8_t *) &nib;
    int8_t exp_lo[16], exp_hi[16];
    for (int j = 0; j < 16; j++) {
        const int low  = j % 16;
        const int high = (j + 5) % 16;
        nb[j] = (uint8_t) ((high << 4) | low);
        exp_lo[j] = kvalues[low];
        exp_hi[j] = kvalues[high];
    }
    uint4 * nib_dev, * lo_dev, * hi_dev;
    CUDA_CHECK(cudaMalloc(&nib_dev, sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&lo_dev, sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&hi_dev, sizeof(uint4)));
    CUDA_CHECK(cudaMemcpy(nib_dev, &nib, sizeof(uint4), cudaMemcpyHostToDevice));
    widen_probe_kernel<<<1, 1>>>(nib_dev, lo_dev, hi_dev);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    uint4 lo, hi;
    CUDA_CHECK(cudaMemcpy(&lo, lo_dev, sizeof(uint4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hi, hi_dev, sizeof(uint4), cudaMemcpyDeviceToHost));
    const int8_t * lo8 = (const int8_t *) &lo;
    const int8_t * hi8 = (const int8_t *) &hi;
    int bad = 0;
    for (int j = 0; j < 16; j++) {
        if (lo8[j] != exp_lo[j]) { printf("  lo[%2d] got=%3d exp=%3d\n", j, lo8[j], exp_lo[j]); bad++; }
        if (hi8[j] != exp_hi[j]) { printf("  hi[%2d] got=%3d exp=%3d\n", j, hi8[j], exp_hi[j]); bad++; }
    }
    printf("[widen] rp_q4_0_expand: bad=%d -> %s\n", bad, bad == 0 ? "PASS" : "FAIL");
    CUDA_CHECK(cudaFree(nib_dev));
    CUDA_CHECK(cudaFree(lo_dev));
    CUDA_CHECK(cudaFree(hi_dev));
    return bad == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// load_w_raw-in-isolation probe: set a known repacked buffer, call
// rp_traits<Q4_0>::load_w_raw for a (wrow, sb), and check the nibble + scale.
// ---------------------------------------------------------------------------
static __global__ void loadw_probe_kernel(const uint8_t * wbase, uint32_t ne0,
        uint32_t ne1, uint32_t wrow, uint32_t sb, uint4 * raw, uint16_t * d) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    rp_traits<GGML_TYPE_Q4_0>::geom wg(ne0, ne1);
    rp_traits<GGML_TYPE_Q4_0>::load_w_raw(wbase, wg, wrow, sb, *raw, *d);
#else
    (void) wbase; (void) ne0; (void) ne1; (void) wrow; (void) sb; (void) raw; (void) d;
    NO_DEVICE_CODE;
#endif
}

static int run_loadw_test() {
    const int ne0 = 256, ne1 = 64;
    const int64_t qs_str = repack_qs_row_stride(GGML_TYPE_Q4_0, ne0);
    const size_t  w_bytes = repack_gcn_nbytes(GGML_TYPE_Q4_0, ne0, ne1);
    std::vector<uint8_t> w_host(w_bytes, 0);
    const uint32_t wrow = 5, sb = 3;   // not the origin, to catch stride bugs
    // nibble for (wrow, sb) is at byte offset wrow*qs_str + sb*16
    uint4 * nib = (uint4 *) (w_host.data() + (size_t) wrow * qs_str + (size_t) sb * 16);
    nib->x = 0x0f0e0d0c; nib->y = 0x0b0a0908; nib->z = 0x07060504; nib->w = 0x03020100;
    // scale for (wrow, sb) is at byte offset ne1*qs_str + (wrow*n_sub + sb)*2
    const uint16_t scale = 0x3C00;
    ((uint16_t *) (w_host.data() + (size_t) ne1 * qs_str))[(size_t) wrow * (ne0 / 32) + sb] = scale;

    uint8_t * w_dev; uint4 * raw_dev; uint16_t * d_dev;
    CUDA_CHECK(cudaMalloc(&w_dev, w_bytes));
    CUDA_CHECK(cudaMemcpy(w_dev, w_host.data(), w_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&raw_dev, sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&d_dev, sizeof(uint16_t)));
    loadw_probe_kernel<<<1, 1>>>(w_dev, ne0, ne1, wrow, sb, raw_dev, d_dev);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    uint4 raw; uint16_t d;
    CUDA_CHECK(cudaMemcpy(&raw, raw_dev, sizeof(uint4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&d, d_dev, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    int bad = 0;
    if (raw.x != 0x0f0e0d0c) { printf("  raw.x=%08x exp 0f0e0d0c\n", raw.x); bad++; }
    if (raw.y != 0x0b0a0908) { printf("  raw.y=%08x exp 0b0a0908\n", raw.y); bad++; }
    if (raw.z != 0x07060504) { printf("  raw.z=%08x exp 07060504\n", raw.z); bad++; }
    if (raw.w != 0x03020100) { printf("  raw.w=%08x exp 03020100\n", raw.w); bad++; }
    if (d != scale)         { printf("  d=%04x exp %04x\n", d, scale); bad++; }
    printf("[load_w_raw] (wrow=%u sb=%u) bad=%d -> %s\n", wrow, sb, bad, bad == 0 ? "PASS" : "FAIL");
    CUDA_CHECK(cudaFree(w_dev)); CUDA_CHECK(cudaFree(raw_dev)); CUDA_CHECK(cudaFree(d_dev));
    return bad == 0 ? 0 : 1;
}

template <ggml_type WT>
static int run_gemm_test(const char * name, int ne0, int ne1, int n_tok) {
    using wt = wt_op<WT>;
    using block_t = typename wt::block_t;
    const int nrl = MMQ_RP_Q8_NROW_LANES;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 1. weight + repack + device copy
    std::vector<block_t> W;
    wt::make(rng, W, ne0, ne1);
    const size_t w_bytes = repack_gcn_nbytes(WT, ne0, ne1);
    std::vector<uint8_t> w_host(w_bytes);
    wt::repack(W.data(), w_host.data(), ne0, ne1);
    uint8_t * w_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&w_dev, w_bytes));
    CUDA_CHECK(cudaMemcpy(w_dev, w_host.data(), w_bytes, cudaMemcpyHostToDevice));

    // 2. activation + quantize + device copy
    std::vector<float> X((size_t) ne0 * n_tok);
    for (auto & x : X) x = dist(rng);
    std::vector<block_q8_1_mmq> xq_host((size_t)(ne0 / 128) * n_tok);
    quantize_x_mmq_d4(X.data(), xq_host.data(), ne0, n_tok);
    block_q8_1_mmq * xq_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&xq_dev, xq_host.size() * sizeof(block_q8_1_mmq)));
    CUDA_CHECK(cudaMemcpy(xq_dev, xq_host.data(), xq_host.size() * sizeof(block_q8_1_mmq),
        cudaMemcpyHostToDevice));

    // 3. CPU reference (dequant W and X)
    std::vector<float> Wq((size_t) ne0 * ne1), Xq((size_t) ne0 * n_tok);
    wt::dequant(W.data(), Wq.data(), ne0, ne1);
    dequant_x_mmq_d4(xq_host.data(), Xq.data(), ne0, n_tok);
    std::vector<float> Y_ref((size_t) ne1 * n_tok);
    gemm_cpu_ref(Wq.data(), Xq.data(), Y_ref.data(), ne0, ne1, n_tok);

    // 4. run the GEMM
    float * y_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&y_dev, (size_t) ne1 * n_tok * sizeof(float)));
    CUDA_CHECK(cudaMemset(y_dev, 0, (size_t) ne1 * n_tok * sizeof(float)));
    const int  bn   = 64 * MMQ_RP_Q8_TN;
    const dim3 grid((ne1 + 64 - 1) / 64, (n_tok + bn - 1) / bn, 1);
    mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, WT>
        <<<grid, dim3(64, nrl), 0, 0>>>(
            w_dev, (const block_q8_1 *) xq_dev, y_dev,
            (uint32_t) ne0, (uint32_t) ne1, (uint32_t) n_tok,
            nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> Y_gpu((size_t) ne1 * n_tok);
    CUDA_CHECK(cudaMemcpy(Y_gpu.data(), y_dev, (size_t) ne1 * n_tok * sizeof(float),
        cudaMemcpyDeviceToHost));

    // 5. compare
    double max_abs = 0.0, max_rel = 0.0;
    int bad = 0;
    for (size_t i = 0; i < Y_gpu.size(); i++) {
        const double diff  = std::abs((double) Y_gpu[i] - (double) Y_ref[i]);
        const double denom = std::max(1.0, std::abs((double) Y_ref[i]));
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, diff / denom);
        if (diff / denom > 0.01) bad++;
    }
    printf("[%s] ne0=%d ne1=%d n_tok=%d  max_abs=%.4g  max_rel=%.4g  bad(>1%%)=%d/%zu  -> %s\n",
        name, ne0, ne1, n_tok, max_abs, max_rel, bad, Y_gpu.size(), (max_rel < 0.01) ? "PASS" : "FAIL");
    if (max_rel >= 0.01) {
        printf("    first 8 gpu vs ref:\n");
        for (int i = 0; i < 8; i++) printf("      [%d] gpu=%.5f ref=%.5f\n", i, Y_gpu[i], Y_ref[i]);
    }

    CUDA_CHECK(cudaFree(w_dev));
    CUDA_CHECK(cudaFree(xq_dev));
    CUDA_CHECK(cudaFree(y_dev));
    return (max_rel < 0.01) ? 0 : 1;
}

static int run_real_test(const char * fname, const char * tname, uint64_t data_off, int n_tok);

int main(int argc, char ** argv) {
    int ne0 = 256, ne1 = 64, n_tok = 128;
    const char * sel = "both";
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "widen") == 0) return run_widen_test();
        if (strcmp(argv[i], "loadw") == 0) return run_loadw_test();
        if (strcmp(argv[i], "real") == 0) {
            if (i + 3 > argc) { printf("usage: real <gguf> <tensor> <data_off> [ntok]\n"); return 2; }
            const int nt = (i + 4 < argc) ? atoi(argv[i + 4]) : 512;
            return run_real_test(argv[i + 1], argv[i + 2], strtoull(argv[i + 3], nullptr, 10), nt);
        }
        else if (strcmp(argv[i], "q8") == 0) sel = "q8";
        else if (strcmp(argv[i], "q40") == 0) sel = "q40";
        else if (strncmp(argv[i], "ne0=", 4) == 0) ne0 = atoi(argv[i] + 4);
        else if (strncmp(argv[i], "ne1=", 4) == 0) ne1 = atoi(argv[i] + 4);
        else if (strncmp(argv[i], "ntok=", 5) == 0) n_tok = atoi(argv[i] + 5);
    }
    int rc = 0;
    if (strcmp(sel, "q8") == 0 || strcmp(sel, "both") == 0)
        rc |= run_gemm_test<GGML_TYPE_Q8_0>("Q8_0 ", ne0, ne1, n_tok);
    if (strcmp(sel, "q40") == 0 || strcmp(sel, "both") == 0)
        rc |= run_gemm_test<GGML_TYPE_Q4_0>("Q4_0 ", ne0, ne1, n_tok);
    return rc;
}

// ---------------------------------------------------------------------------
// Real-weight mode: load a canonical Q4_0 tensor from a GGUF file, repack it
// with the production repack_q4_0_host, run the GEMM, diff vs CPU reference.
// Usage: test_gemm_q40 real <gguf> <tensor-name> <data_offset> [ntok]
// ---------------------------------------------------------------------------
static int run_real_test(const char * fname, const char * tname, uint64_t data_off, int n_tok) {
    FILE * f = fopen(fname, "rb");
    if (!f) { printf("cannot open %s\n", fname); return 1; }
    // find tensor offset in the GGUF header
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char hdr[4096 * 1024];
    // read the whole header region up to data_off
    long hlen = (long) std::min<long>(fsize, data_off);
    std::vector<unsigned char> hdrb(hlen);
    fread(hdrb.data(), 1, hlen, f);
    // parse: magic, version, n_tensors, n_kv, kvs, tensor entries
    size_t p = 0;
    auto rd = [&](size_t n) -> std::vector<unsigned char> {
        std::vector<unsigned char> r(hdrb.begin() + p, hdrb.begin() + p + n); p += n; return r; };
    auto rdstr = [&]() -> std::string {
        uint64_t n; memcpy(&n, rd(8).data(), 8);
        std::string s((const char *) rd(n).data(), n); return s; };
    auto rdval = [&](uint32_t t) {
        std::function<void(uint32_t)> go;
        go = [&](uint32_t t) {
            if (t == 8) { rdstr(); }
            else if (t == 9) { uint32_t et; uint64_t cnt; memcpy(&et, rd(4).data(), 4); memcpy(&cnt, rd(8).data(), 8); for (uint64_t i = 0; i < cnt; i++) go(et); }
            else { static const size_t sz[] = {1,1,2,2,4,4,4,1,0,0,8,8,8}; rd(sz[t]); }
        };
        go(t);
    };
    p = 8;
    uint64_t n_tensors, n_kv;
    memcpy(&n_tensors, rd(8).data(), 8);
    memcpy(&n_kv, rd(8).data(), 8);
    for (uint64_t i = 0; i < n_kv; i++) { rdstr(); uint32_t t; memcpy(&t, rd(4).data(), 4); rdval(t); }
    int64_t ne0 = 0, ne1 = 0; uint64_t off = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        std::string name = rdstr();
        uint32_t ndim; memcpy(&ndim, rd(4).data(), 4);
        std::vector<int64_t> ne(ndim);
        for (uint32_t d = 0; d < ndim; d++) { uint64_t v; memcpy(&v, rd(8).data(), 8); ne[d] = (int64_t) v; }
        uint32_t tt; memcpy(&tt, rd(4).data(), 4);
        uint64_t toff; memcpy(&toff, rd(8).data(), 8);
        if (name == tname) { ne0 = ne[0]; ne1 = ne[1]; off = toff; }
    }
    if (ne0 == 0) { printf("tensor %s not found\n", tname); return 1; }
    printf("[real] %s ne0=%ld ne1=%ld off=%llu\n", tname, (long) ne0, (long) ne1, (unsigned long long) off);

    const int64_t n_blocks = ne0 / 32;
    std::vector<block_q4_0> W((size_t) ne1 * n_blocks);
    fseek(f, (long) (data_off + off), SEEK_SET);
    size_t got = fread(W.data(), 1, W.size() * sizeof(block_q4_0), f);
    fclose(f);
    if (got != W.size() * sizeof(block_q4_0)) { printf("short read\n"); return 1; }

    // repack with the production host function
    const size_t w_bytes = repack_gcn_nbytes(GGML_TYPE_Q4_0, ne0, ne1);
    std::vector<uint8_t> w_host(w_bytes);
    repack_q4_0_host(W.data(), w_host.data(), ne0, ne1);

    // compare against the dumped device buffer, if present
    char dpath[256]; snprintf(dpath, sizeof(dpath), "/tmp/rp_dev_%ldx%ld.bin", (long) ne0, (long) ne1);
    FILE * df = fopen(dpath, "rb");
    if (df) {
        std::vector<uint8_t> dev(w_bytes);
        size_t g2 = fread(dev.data(), 1, w_bytes, df); fclose(df);
        if (g2 == w_bytes) {
            size_t nd = 0; for (size_t i = 0; i < w_bytes; i++) if (dev[i] != w_host[i]) nd++;
            printf("[real] host-repack vs device dump: %s (%zu diff bytes)\n",
                nd == 0 ? "MATCH" : "DIFFER", nd);
        }
    }

    uint8_t * w_dev;
    CUDA_CHECK(cudaMalloc(&w_dev, w_bytes));
    CUDA_CHECK(cudaMemcpy(w_dev, w_host.data(), w_bytes, cudaMemcpyHostToDevice));

    // random X, D4 quant, CPU reference
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X((size_t) ne0 * n_tok);
    for (auto & x : X) x = dist(rng);
    std::vector<block_q8_1_mmq> xq_host((size_t)(ne0 / 128) * n_tok);
    quantize_x_mmq_d4(X.data(), xq_host.data(), ne0, n_tok);
    block_q8_1_mmq * xq_dev;
    CUDA_CHECK(cudaMalloc(&xq_dev, xq_host.size() * sizeof(block_q8_1_mmq)));
    CUDA_CHECK(cudaMemcpy(xq_dev, xq_host.data(), xq_host.size() * sizeof(block_q8_1_mmq), cudaMemcpyHostToDevice));

    std::vector<float> Wq((size_t) ne0 * ne1);
    wt_op<GGML_TYPE_Q4_0>::dequant(W.data(), Wq.data(), ne0, ne1);
    std::vector<float> Xq((size_t) ne0 * n_tok);
    dequant_x_mmq_d4(xq_host.data(), Xq.data(), ne0, n_tok);
    std::vector<float> Y_ref((size_t) ne1 * n_tok);
    gemm_cpu_ref(Wq.data(), Xq.data(), Y_ref.data(), ne0, ne1, n_tok);

    printf("[dbg] sizeof(block_q4_0)=%zu\n", sizeof(block_q4_0));
    const uint8_t * wb = (const uint8_t *) W.data();
    printf("[dbg] W[0] raw = %02x %02x %02x %02x %02x %02x %02x %02x\n",
        wb[0], wb[1], wb[2], wb[3], wb[4], wb[5], wb[6], wb[7]);
    printf("[dbg] d as f16 bits = %04x -> f32 = %g\n",
        (uint16_t) (wb[1] << 8 | wb[0]), __half2float(W[0].d));
    printf("[dbg] Wq[0..3] = %g %g %g %g\n", Wq[0], Wq[1], Wq[2], Wq[3]);
    printf("[dbg] Xq[0..3] = %g %g %g %g\n", Xq[0], Xq[1], Xq[2], Xq[3]);
    printf("[dbg] Y_ref[0..7] = %g %g %g %g %g %g %g %g\n",
        Y_ref[0], Y_ref[1], Y_ref[2], Y_ref[3], Y_ref[4], Y_ref[5], Y_ref[6], Y_ref[7]);

    float * y_dev;
    CUDA_CHECK(cudaMalloc(&y_dev, (size_t) ne1 * n_tok * sizeof(float)));
    const int nrl = MMQ_RP_Q8_NROW_LANES;
    const int bn = 64 * MMQ_RP_Q8_TN;
    const dim3 grid((ne1 + 63) / 64, (n_tok + bn - 1) / bn, 1);
    mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q4_0>
        <<<grid, dim3(64, nrl), 0, 0>>>(
            w_dev, (const block_q8_1 *) xq_dev, y_dev,
            (uint32_t) ne0, (uint32_t) ne1, (uint32_t) n_tok,
            nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> Y_gpu((size_t) ne1 * n_tok);
    CUDA_CHECK(cudaMemcpy(Y_gpu.data(), y_dev, (size_t) ne1 * n_tok * sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs = 0.0, max_rel = 0.0; int bad = 0;
    for (size_t i = 0; i < Y_gpu.size(); i++) {
        const double diff = std::abs((double) Y_gpu[i] - (double) Y_ref[i]);
        const double denom = std::max(1.0, std::abs((double) Y_ref[i]));
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, diff / denom);
        if (diff / denom > 0.01) bad++;
    }
    printf("[real] max_abs=%.4g max_rel=%.4g bad(>1%%)=%d/%zu -> %s\n",
        max_abs, max_rel, bad, Y_gpu.size(), max_rel < 0.01 ? "PASS" : "FAIL");
    if (max_rel >= 0.01) {
        for (int i = 0; i < 8; i++)
            printf("    [%d] gpu=%.5f ref=%.5f\n", i, Y_gpu[i], Y_ref[i]);
    }
    CUDA_CHECK(cudaFree(w_dev)); CUDA_CHECK(cudaFree(xq_dev)); CUDA_CHECK(cudaFree(y_dev));
    return max_rel < 0.01 ? 0 : 1;
}

// Stubs for symbols referenced by repack-common.cu paths the harness never calls.
int ggml_n_dims(const ggml_tensor * t) { return 2; }
bool ggml_is_contiguous(const ggml_tensor * t) { return true; }
ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t) { return nullptr; }
bool ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_type_t) { return false; }
void ggml_cuda_set_device(int) {}
