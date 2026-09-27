// Direct test: Q4_0_64 x Q8_0 matmul via vec_dot, compared to float reference.
// Mimics the CPU GEMM path: quantize weights to Q4_0_64, activations to Q8_0,
// then compute each output row via ggml_vec_dot_q4_0_64_q8_0.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ggml.h"
#include "ggml-quants.h"
#include "../../ggml/src/ggml-cpu/quants.h"

float ggml_table_f32_f16[1 << 16] = {0};
void ggml_abort(const char * file, int line, const char * fmt, ...) { (void)file; (void)line; fprintf(stderr, "abort: %s\n", fmt); exit(1); }
size_t ggml_row_size(enum ggml_type type, int64_t ne) { (void)type; (void)ne; return 0; }
size_t ggml_type_size(enum ggml_type type) { (void)type; return 0; }
const char * ggml_type_name(enum ggml_type type) { (void)type; return "?"; }
void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) { (void)tensor; (void)data; (void)offset; (void)size; }
void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) { (void)tensor; (void)value; (void)offset; (void)size; }

static float fp16_to_fp32_local(ggml_fp16_t x) {
    union { uint16_t u16; ggml_fp16_t fp16; } u = {x};
    const uint32_t sign = (uint32_t)(u.u16 >> 15) & 1;
    const uint32_t exp  = (uint32_t)(u.u16 >> 10) & 0x1F;
    uint32_t mant = (uint32_t)(u.u16 & 0x3FF);
    uint32_t f32;
    if (exp == 0) {
        f32 = sign << 31;
        if (mant) {
            uint32_t e = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; e--; }
            mant &= 0x3FF;
            f32 |= (e << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f32 = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f32 = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    union { uint32_t u32; float f; } r = {f32};
    return r.f;
}

int main(int argc, char ** argv) {
    for (int i = 0; i < (1 << 16); ++i) {
        union { uint16_t u16; ggml_fp16_t fp16; } u = {i};
        ggml_table_f32_f16[i] = fp16_to_fp32_local(u.fp16);
    }

    // W: [M x K] weights, A: [K x N] activations
    // Use positive-ish values so the dot product is large and relative error is meaningful.
    const int M = 128;   // output features (rows of W)
    const int K = 2048;  // hidden dim (cols of W, rows of A)
    const int N = 16;    // batch size (cols of A)

    float * W = malloc(M * K * sizeof(float));
    float * A = malloc(K * N * sizeof(float));
    for (int i = 0; i < M * K; i++) W[i] = 0.1f * (1.0f + sinf(i * 0.01f));  // positive, ~[0, 0.2]
    for (int i = 0; i < K * N; i++) A[i] = 0.1f * (1.0f + cosf(i * 0.01f));  // positive, ~[0, 0.2]

    // Reference: float matmul C = W * A  (C is [M x N])
    float * C_ref = malloc(M * N * sizeof(float));
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double sum = 0;
            for (int k = 0; k < K; k++) sum += (double) W[m * K + k] * A[k * N + n];
            C_ref[m * N + n] = (float) sum;
        }
    }

    // Quantize W to Q4_0_64 (row-major: each row is K elements)
    block_q4_0_64 * Wq = malloc(M * (K / QK4_0_64) * sizeof(block_q4_0_64));
    for (int m = 0; m < M; m++) {
        quantize_row_q4_0_64_ref(W + m * K, Wq + m * (K / QK4_0_64), K);
    }

    // Quantize A to Q8_0 (column-major: each column is K elements)
    // In the GEMM path, src1 is [K x N] with ne10=K, ne11=N.
    // Each column n is a Q8_0 row of K elements.
    block_q8_0 * Aq = malloc(N * (K / QK8_0) * sizeof(block_q8_0));
    for (int n = 0; n < N; n++) {
        float * col = malloc(K * sizeof(float));
        for (int k = 0; k < K; k++) col[k] = A[k * N + n];
        quantize_row_q8_0(col, Aq + n * (K / QK8_0), K);
        free(col);
    }

    // Compute C = Wq * Aq via vec_dot (mimics the GEMM path)
    float * C = malloc(M * N * sizeof(float));
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float s = 0;
            ggml_vec_dot_q4_0_64_q8_0(K, &s, 0, Wq + m * (K / QK4_0_64), 0, Aq + n * (K / QK8_0), 0, 1);
            C[m * N + n] = s;
        }
    }

    // Compare
    double max_rel = 0;
    int worst_m = 0, worst_n = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double rel = fabs(C[m * N + n] - C_ref[m * N + n]) / (fabs(C_ref[m * N + n]) + 1e-6);
            if (rel > max_rel) { max_rel = rel; worst_m = m; worst_n = n; }
        }
    }
    printf("M=%d K=%d N=%d  max_rel_err=%.4f%%  (at m=%d n=%d: ref=%.4f quant=%.4f)\n",
           M, K, N, 100.0 * max_rel, worst_m, worst_n, C_ref[worst_m * N + worst_n], C[worst_m * N + worst_n]);

    // Also test with N=1 (GEMV) for comparison
    {
        float s0 = 0;
        ggml_vec_dot_q4_0_64_q8_0(K, &s0, 0, Wq, 0, Aq, 0, 1);
        double rel0 = fabs(s0 - C_ref[0]) / (fabs(C_ref[0]) + 1e-6);
        printf("N=1 (GEMV): ref=%.4f quant=%.4f rel_err=%.4f%%\n", C_ref[0], s0, 100.0 * rel0);
    }

    free(W); free(A); free(C_ref); free(Wq); free(Aq); free(C);
    return 0;
}
