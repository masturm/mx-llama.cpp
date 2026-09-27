// Direct ggml-API test: Q4_0_64 x F32 mul_mat on the CPU backend.
// Exercises the real GEMM path (src1 quantization, chunking, vec_dot).
// Compares against a float reference. Varies N (batch) to find the threshold.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-quants.h"

int main(int argc, char ** argv) {
    const int K = 2048;  // hidden dim (ne0)
    const int M = 256;   // output features (ne1)

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!cpu) { fprintf(stderr, "no cpu backend\n"); return 1; }

    // Build a float reference weight matrix W [K x M] (column-major per ggml: ne0=K, ne1=M)
    float * Wf = malloc(K * M * sizeof(float));
    for (int i = 0; i < K * M; i++) Wf[i] = 0.1f * (1.0f + sinf(i * 0.01f));

    // Quantize W to Q4_0_64. ggml stores rows of length ne0=K.
    // block layout: for each of M rows, quantize K elements.
    size_t row_blocks = K / QK4_0_64;
    block_q4_0_64 * Wq = malloc(M * row_blocks * sizeof(block_q4_0_64));
    for (int m = 0; m < M; m++) {
        quantize_row_q4_0_64_ref(Wf + m * K, Wq + m * row_blocks, K);
    }

    // Test several batch sizes N
    const int Ns[] = {1, 2, 8, 16, 32, 64};
    for (int t = 0; t < (int)(sizeof(Ns)/sizeof(Ns[0])); ++t) {
        const int N = Ns[t];

        // Activation A [K x N] F32
        float * Af = malloc(K * N * sizeof(float));
        for (int i = 0; i < K * N; i++) Af[i] = 0.1f * (1.0f + cosf(i * 0.01f));

        // Float reference: C[m][n] = sum_k W[m][k] * A[n][k]
        // W is row-major [M x K] (row m = Wf + m*K), A is row-major [N x K] (row n = Af + n*K)
        float * Cref = malloc(M * N * sizeof(float));
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                double sum = 0;
                for (int k = 0; k < K; k++) sum += (double) Wf[m * K + k] * Af[n * K + k];
                Cref[m * N + n] = (float) sum;
            }
        }

        // ggml graph
        struct ggml_init_params params = { 16 * 1024 * 1024, NULL, true };
        struct ggml_context * ctx = ggml_init(params);
        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0_64, K, M);
        struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        struct ggml_tensor * c = ggml_mul_mat(ctx, w, a);  // [M x N]

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }

        memcpy(w->data, Wq, M * row_blocks * sizeof(block_q4_0_64));
        memcpy(a->data, Af, K * N * sizeof(float));

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(gf, c);
        enum ggml_status st = ggml_backend_graph_compute(cpu, gf);
        if (st != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed %d\n", st); return 1; }

        // Compare. c has ne0=M, ne1=N, so c[m][n] is at c->data[n*M + m].
        double max_rel = 0;
        int worst_m = 0, worst_n = 0;
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float got = ((float *) c->data)[n * M + m];
                float ref = Cref[m * N + n];
                double rel = fabs(got - ref) / (fabs(ref) + 1e-6);
                if (rel > max_rel) { max_rel = rel; worst_m = m; worst_n = n; }
            }
        }
        printf("N=%-4d max_rel_err=%.4f%%  (worst m=%d n=%d: ref=%.4f got=%.4f)\n",
               N, 100.0 * max_rel, worst_m, worst_n, Cref[worst_m * N + worst_n],
               ((float *) c->data)[worst_n * M + worst_m]);
        if (N <= 4) {
            for (int m = 0; m < 4; m++) {
                printf("  m=%d: ", m);
                for (int n = 0; n < N; n++) {
                    printf("got=%.2f(ref=%.2f) ", ((float *) c->data)[n * M + m], Cref[m * N + n]);
                }
                printf("\n");
            }
        }

        ggml_free(ctx);
        free(Af); free(Cref);
    }

    free(Wf); free(Wq);
    ggml_backend_free(cpu);
    return 0;
}
