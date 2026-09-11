# Q4_0 gfx906 V_DOT8 Experiment

Date: 2026-09-10

## Goal

Test Q4_0 weight matmul on gfx906 with:

- 4-bit weights
- 4-bit activation quantization
- `V_DOT8_I32_I4` instead of `V_DOT4_I32_I8`

The experiment is opt-in through:

```text
-DGGML_CUDA_Q4_0_INT4_ACTIVATIONS
```

## Build

The MI50 build requires the GCC override used in `llamacpp_build_MI50.md`:

```bash
export HIPCXX="$(hipconfig -l)/clang"
export HIP_PATH="$(hipconfig -R)"
export CCC_OVERRIDE_OPTIONS="^--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/12"
```

The fast iteration targets are:

```bash
cmake --build build --target llama-cli -j18
cmake --build build --target llama-bench -j18
cmake --build build --target llama-perplexity -j18
```

The active build is configured for `gfx906` with `GGML_CUDA_Q4_0_INT4_ACTIVATIONS` enabled. Both targets build successfully.

## Code Changes

Modified files:

- `ggml/src/ggml-cuda/common.cuh`
  - Adds the gfx906 `V_DOT8_I32_I4` helper.
  - Supports `GGML_CUDA_Q4_0_INT4_SCALAR_REFERENCE`, which replaces the hardware instruction with an independent scalar signed-int4 dot for validation.
- `ggml/src/ggml-cuda/mmq-vec-dot.cuh`
  - Adds the Q4_0/int4 activation dot8 path.
  - Packs eight signed int4 values for the dot8 instruction. The original bug was in `ggml_cuda_pack_i4x8`: the nibbles from the second input word were shifted in the wrong direction. The corrected implementation uses left shifts from the original nibble positions.
  - Converts Q4_0 weight nibbles from unsigned `0..15` to signed `-8..7`.
- `ggml/src/ggml-cuda/mmq.cuh`
  - Selects the dot8 Q4_0 helper when the experiment macro is enabled.
- `ggml/src/ggml-cuda/quantize.cu`
  - Adds Q4_0-specific symmetric int4 activation quantization.
  - Uses the Q4_0 scale convention with `d = amax / 8`.

The normal Q8 activation path remains available when the macro is disabled.

## Performance

Original Q4_0 GPU baseline:

```text
917.24 t/s, pp512
```

Experimental Q4_0 GPU path using int4 activations and `V_DOT8_I32_I4`:

```text
540.82 t/s, pp512
```

Q4_1 control:

```text
907.74 t/s, pp512
```

Corrected Q4_0 DP8 path:

```text
482.59 t/s, pp4096
```

The first implementation is approximately 41% slower than the original Q4_0 path. The current dot8 helper still performs packing and nibble rearrangement inside the matmul loop, so this is not yet an optimized implementation.

## Correctness and Quality

The initial implementation produced constant or garbled output. The following fixes removed the constant-output failure:

- Corrected the signed int4 conversion mask from `0x88` to `0x08` per nibble byte.
- Transposed Q4_0 low and high nibble groups before dot8 execution.
- Restored hardware `V_DOT8_I32_I4` after using a scalar int4 reference for isolation.

The latest corrected implementation produces coherent output. The previous quality failure was caused by incorrect int4 nibble packing, not by the `V_DOT8_I32_I4` instruction.

Test command:

```bash
./build/bin/llama-cli \
  -m /media/muselko/55734f1e-94a9-48f5-811a-ccdcf6d36011/llm/Qwen3.5-9B-Q4_0.gguf \
  -ngl 99 -dev ROCM2 \
  -p 'The capital of France is?' \
  -n 8 -s 42 --no-display-prompt -rea off -st
```

## Perplexity

The matched smoke test used `tools/server/README.md`, 8 chunks, `n_ctx=512`, and batch size `512`.

Correct Q4_0 CPU control, forced with `--no-op-offload`:

```text
PPL = 3.6947 +/- 0.19395
```

Q4_1 GPU control:

```text
PPL = 3.6124 +/- 0.18639
```

The earlier experimental Q4_0 GPU run reported:

```text
PPL = 110748.1269 +/- 13526.84042
```

That result was collected before the final activation-scale correction from `amax / 7` to `amax / 8`, so it is provisional and must not be treated as the final quality number.

After fixing `ggml_cuda_pack_i4x8`, the corrected DP8 run produced:

```text
[1] 7.4700
[2] 4.9565
[3] 3.9743
[4] 3.6901
[5] 4.0684
[6] 4.0857
[7] 4.1350
[8] 4.0132

PPL = 4.0132 +/- 0.21808
```

Compared with the corrected CPU Q4_0 control (`PPL = 3.6947`), the remaining
degradation is consistent with the intended lower precision of the activation
quantizer. The catastrophic `110748` PPL was caused by the packing bug and is
superseded by this result.

For reference "Qwen3.5-9B-UD-IQ2_XXS.gguf" gives below 
Final estimate: PPL = 4.6741 +/- 0.24987

## Next Steps

1. Build a second DP4 directory with the macro disabled.
2. Run the numerical A/B comparison below.
3. Compare both GPU results with the corrected CPU Q4_0 control.
4. Optimize packing, preferably by storing activations in a dot8-ready layout during quantization rather than repacking inside `mmq-vec-dot.cuh`.
5. Investigate finer activation scale granularity if the corrected perplexity remains poor.
6. Profile the final version with rocprof kernel traces and compare Q4_0 matmul time against the original Q8 activation path.

## Numerical A/B Test

`tools/q40-dot8-compare.py` runs the same GPU perplexity evaluation against two
builds and reports the DP4-to-DP8 absolute and relative PPL difference. It does
not force CPU execution; both paths must use the same ROCm device.

The DP4 build must be configured without `GGML_CUDA_Q4_0_INT4_ACTIVATIONS` and
the DP8 build must be configured with it. Keep the build directories separate:

```bash
cmake -S . -B build-dp4 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx906

cmake -S . -B build-dp8 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx906 \
  -DCMAKE_HIP_FLAGS=-DGGML_CUDA_Q4_0_INT4_ACTIVATIONS

export HIPCXX="$(hipconfig -l)/clang"
export HIP_PATH="$(hipconfig -R)"
export CCC_OVERRIDE_OPTIONS="^--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/12"

cmake --build build-dp4 --target llama-perplexity -j18
cmake --build build-dp8 --target llama-perplexity -j18
```

Run the comparison with the same model and corpus:

```bash
/usr/bin/python tools/q40-dot8-compare.py \
  --dp4-build build-dp4 \
  --dp8-build build-dp8 \
  --model /absolute/path/to/Qwen3.5-9B-Q4_0.gguf \
  --corpus tools/server/README.md \
  --device ROCM2 \
  --context 512 \
  --batch 512 \
  --chunks 8
```

The optional `--max-relative-delta 0.05` makes the test fail when DP8 PPL is
more than 5% above DP4 PPL. The tolerance should be chosen from the intended
activation-quantization quality target, not assumed to be zero: DP4 and DP8
intentionally use different activation quantizers.

For a lower-level instruction check, build a third directory with both Q4_0
macros enabled:

```bash
cmake -S . -B build-dp8-scalar \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx906 \
  -DCMAKE_HIP_FLAGS="-DGGML_CUDA_Q4_0_INT4_ACTIVATIONS -DGGML_CUDA_Q4_0_INT4_SCALAR_REFERENCE"

cmake --build build-dp8-scalar --target llama-perplexity -j18
```

Then add the reference build to the comparison:

```bash
/usr/bin/python tools/q40-dot8-compare.py \
  --dp4-build build-dp4 \
  --dp8-build build-dp8 \
  --dp8-reference-build build-dp8-scalar \
  --model /absolute/path/to/Qwen3.5-9B-Q4_0.gguf \
  --corpus tools/server/README.md \
  --device ROCM2 \
  --context 512 \
  --batch 512 \
  --chunks 8
```

The scalar-reference comparison uses `--save-all-logits`, `--kl-divergence`,
and `--kl-divergence-base`. The explicit `--kl-divergence` flag is required to
activate the comparison mode. It reports:

- DP8 hardware versus scalar-reference KL divergence
- RMS token-probability difference
- Same top-token percentage

These values should be near numerical noise. A large difference means the
packing, signed-nibble conversion, lane order, or hardware instruction path is
wrong. The DP4 versus DP8 PPL difference remains the separate measure of
activation-quantization quality loss.

Raw earlier traces are in:

```text
/tmp/llama-prefill-profile/results_prefill.csv
/tmp/llama-dot8-profile/results_q40-dot8.csv
```
