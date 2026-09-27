// Verify vec_dot_q4_0_64_q8_0 against a plain float dot product.
// Self-contained: stubs out the few external symbols needed to link quants.c.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ggml.h"
#include "ggml-quants.h"
#include "../../ggml/src/ggml-cpu/quants.h"

// stubs for symbols referenced by quants.c / ggml-quants.c
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
            // subnormal
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

int main(void) {
    // init f16->f32 lookup table (same as ggml-cpu.c)
    for (int i = 0; i < (1 << 16); ++i) {
        union { uint16_t u16; ggml_fp16_t fp16; } u = {i};
        ggml_table_f32_f16[i] = fp16_to_fp32_local(u.fp16);
    }

    const int qk = QK4_0_64;
    // test several row lengths, including the real model dims
    const int ks[] = {512, 1024, 2048, 4096, 6144};
    for (int t = 0; t < (int)(sizeof(ks)/sizeof(ks[0])); ++t) {
        const int k = ks[t];
        float * x = malloc(k * sizeof(float));
        float * yf = malloc(k * sizeof(float));
        for (int i = 0; i < k; i++) {
            x[i]  = (float)(10.0 * sin(i * 0.7) + 0.5 * cos(i * 1.3));
            yf[i] = (float)(3.0  * cos(i * 0.4) + 0.2 * sin(i * 2.1));
        }

        double dot_ref = 0;
        for (int i = 0; i < k; i++) dot_ref += (double) x[i] * yf[i];

        block_q4_0_64 * qx = malloc((k / qk) * sizeof(block_q4_0_64));
        quantize_row_q4_0_64_ref(x, qx, k);

        block_q8_0 * qy = malloc((k / QK8_0) * sizeof(block_q8_0));
        quantize_row_q8_0(yf, qy, k);

        float dot_quant = 0;
        ggml_vec_dot_q4_0_64_q8_0(k, &dot_quant, 0, qx, 0, qy, 0, 1);

        printf("k=%-5d dot_ref=%.4f dot_quant=%.4f rel_err=%.4f%%\n",
               k, dot_ref, dot_quant, 100.0 * fabs(dot_ref - dot_quant) / fabs(dot_ref));

        free(x); free(yf); free(qx); free(qy);
    }
    return 0;
}
