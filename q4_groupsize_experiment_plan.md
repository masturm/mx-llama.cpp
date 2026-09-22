# Q4 group-size experiment (32/64/128) for attention vs FFN

Plan to explore larger Q4 quantization group sizes as the next big speedup lever for
the dp8 Q4_0 kernel, and to test the hypothesis that attention layers are more
sensitive to large groups than FFN layers.

Status: Phase 1 (accuracy) and Phase 3 (real Q4_0_64 kernel) done. The Q4_0_64
prefill regression was root-caused (unaligned LDS reads) and fixed - Q4_0_64 is
now ~9% FASTER than Q4_0 on pp1024 (8446 vs 7731 t/s). Remaining: a perplexity
number (decode/mmvq not yet implemented for this type, see
q40_64_decode_fix_plan.md), then decide ship 64 vs 128.

---

## Why this is the next lever

`dot8_kernel_perf_findings.md` established that the dp8 Q4_0 kernel is at its optimum
for the 32-block format: all source-level and LDS-layout levers are dead ends. The
dominant remaining gap is the **FPU scaling epilogue** (cvt + 2 mul + fma per 32-element
block), which shares the vector pipe with the dot8 MACs and eats ~58.5% of the raw dot8
peak (C/D = 0.415).

The epilogue fires **once per 32-element block**. A larger block amortizes it over more
dot8:

| block | dot8/scale | FPU/scale | FPU/dot8 | dot8 share of VOP pipe |
|-------|-----------|-----------|----------|------------------------|
| 32    | 4         | 4         | 1.00     | 50%                    |
| 64    | 8         | 4         | 0.50     | 67%                    |
| 128   | 16        | 4         | 0.25     | 80%                    |

(The FPU/scale count is constant - cvt + 2 mul + fma - but is spread over 2x/4x more
dot8.)

### Projected compute-phase speedup (range)

The dot8 share of the *vector pipe* goes 50% -> 67% -> 80%. But the LDS reads and
scalar addressing also scale with the dot8 count and may partially contend for issue
slots (the measured C/D = 0.415 is below the 50% vector-pipe prediction, implying
nonzero LDS/scalar contention). So the real speedup is a range:

| block | optimistic (LDS/scalar overlaps) | conservative (LDS/scalar contends) |
|-------|----------------------------------|------------------------------------|
| 64    | ~1.33x                           | ~1.15x                             |
| 128   | ~1.6x                            | ~1.25x                             |

Phase 2 (benchmark) pins down which end of the range we're at. Even the conservative
end is a meaningful win.

### FLOP weighting (Qwen3.5-2B)

Per layer, MMQ (projection) MACs: attention (QKV + O) = 16.8M, FFN (gate+up+down) =
37.7M. **FFN is ~69% of the MMQ FLOPs, attention ~31%.** So the FFN group size dominates
the end-to-end speedup, while the attention group size dominates the accuracy risk
(see below). (The O(n^2) attention *scores* use a different kernel, not the MMQ dp8
path, so they're excluded here.)

---

## Key insight: the group size affects BOTH activations and weights

The dp8 path is **W4A4**: both the activation (x) and the weight (y) are Q4_0 with
per-32-block scales. To get the amortization, **both must use the larger group** - if
the activation stayed at 32-block while the weight went to 128-block, the activation
scale (x_df) would still change every 32 elements, forcing the FPU epilogue every 32
anyway. So the group size is a single knob applied to both operands.

This makes the **activation quantization** the main accuracy risk, not the weight:
- Activations (the residual stream) have **outliers** - a few channels with large
  magnitude. A single scale over 128 elements is dominated by the outlier, so the
  remaining 127 elements quantize poorly.
- **Attention layers likely have more outlier-prone activations** (the residual stream
  feeding the projections), so they are more sensitive to a coarse group size.

This refines the hypothesis: "attention hurts more" is plausibly driven by the
**activation (x) quantization** in attention layers, not (only) the weight. The
experiments below test this directly.

Note: GGML already has larger-block formats, but none is a *flat* large-group Q4:
- `Q4_K`: 256-element super-blocks with 8x32 sub-blocks, two-level (6-bit sub + 6-bit
  super) scaling. More accurate than flat, but a complex format and a different kernel.
- `NVFP4`: 64-element blocks with 4x16 sub-blocks (UE4M3 sub-scales), a different
  4-bit code (E2M1) than Q4_0's symmetric code.

A **flat** 64/128-element single-scale Q4 (the simplest "larger group") does not exist
and would be a new type. (Two-level scaling, like Q4_K, is a possible accuracy
compromise for the 128-block if flat is too lossy - a Phase 3 option.)

---

## Hypotheses

- H1: Larger group sizes give a real compute speedup (confirm the model, Phase 2).
- H2: Attention is more sensitive to large groups than FFN (the user's hypothesis).
- H3: The activation (x) quantization is the dominant accuracy risk (vs the weight y).
- H4: Some (group_attn, group_ffn) combination gives a good speedup at acceptable
  accuracy (a Pareto point worth shipping).

---

## Phase 1 - Cheap accuracy estimate (NO GGML changes) - GATE

The goal: find out whether large groups are viable at all, and which layers/operands
are sensitive, before investing in any kernel or type work.

### 1a. Weight re-quantization error (cheapest, no model run)
- Dequantize the Q4_0 weights of the existing Qwen3.5-2B GGUF to fp32.
- Re-quantize to 32 / 64 / 128-element flat groups (nearest, per-group absmax scale).
- Measure per-tensor error: MSE, max abs error, and relative error (||e||/||w||).
- Group the results by layer type: attention (QKV, O) vs FFN (gate, up, down).
- Tests H2 (weight side). A script over the GGUF; ~half a day.

### 1b. Activation quantization error (needs a model run)
- Run the model (existing Q4_0 build) and dump per-layer activations (the x input to
  each MMQ) for a fixed calibration prompt.
- Quantize those activations to 32 / 64 / 128-element groups; measure error per layer.
- Compare attention vs FFN, and compare the activation error to the weight error from
  1a.
- Tests H2 + H3 (activation side). ~1 day.

### Decision gate
- If 128-block error is unacceptable for **all** layer types -> the flat large-group
  format is not viable; STOP (or pivot to two-level scaling, Phase 3 option).
- If attention is clearly more sensitive -> confirms H2; use it to pick the
  (group_attn, group_ffn) combinations for Phase 3.
- If the activation (x) error dominates the weight (y) error -> confirms H3; the
  activation quantization is what to protect (e.g., keep x at 32-block, accept that the
  FPU epilogue then fires every 32 - see the caveat below).

**Caveat:** if the activation must stay at 32-block for accuracy (H3 strongly
confirmed), the full 128-block amortization is NOT available (the x scale forces the
epilogue every 32). In that case the win shrinks to the weight-side benefit only, which
is smaller. Phase 1b tells us whether this caveat bites.

**Phase 2 confirmation (done):** the speedup is real. First-principles VOP model gives
1.43x (64-block) / 1.82x (128-block) as the upper bound; the kernel is LDS/VOP co-bound,
so the real speedup is ~1.2-1.4x (64) / ~1.3-1.8x (128). This is a solid, meaningful win
that justifies the moderate activation-accuracy cost of 64-block (and the larger cost of
128-block). **Gate passed - proceed to Phase 3** for the definitive real-model perplexity
+ pp512 numbers.

---

## Phase 2 - Speed confirmation (medium) - GATE

Extend `tools/mmq-bench` (the standalone benchmark from the findings doc):
- Add 64-block and 128-block kernel variants of the compute-only test: accumulate 8 /
  16 dot8 into the integer sum before the single FPU epilogue, instead of 4.
- Measure MAC/s for each; compare to the 32-block C test and to the model's prediction.
- Confirms H1 and pins down the optimistic-vs-conservative speedup range.
- ~1-2 days. No GGML changes (the benchmark is standalone).

### Decision gate
- If the measured speedup is near the conservative end (or below) -> the LDS/scalar
  contention is higher than hoped; reassess whether Phase 3 is worth it.
- If near the optimistic end -> strong case for Phase 3.

### Phase 2 RESULTS (measured)

Extended `tools/mmq-bench` with a block-size x ILP sweep: a kernel doing G/8 dot8 +
G/32 y-LDS-reads + 1 FPU epilogue per G-element group, with CH independent accumulators
(ILP). Ran G=32/64/128 x ILP=4/8/16.

```
ILP= 4:  64-block 1.401x, 128-block 1.958x
ILP= 8:  64-block 1.752x, 128-block 2.850x
ILP=16:  64-block 1.863x, 128-block 3.289x
```

**Key observation:** the speedup INCREASES with ILP (opposite of the naive expectation).
Reason: the G=32 case has the most epilogues, so at high ILP its FPU epilogues contend
harder (its throughput DROPS: 6.35->5.26e12), while G=64/128 (fewer epilogues) become
dot8-bound and benefit from the higher ILP. So the benchmark OVERESTIMATES the real
speedup: (a) it lacks the full LDS-bandwidth overhead of the real kernel (the y-reads /
scale-reads that do NOT amortize), and (b) its simplified structure makes G=32 more
FPU-bound than the real kernel's 32-accumulator (i,j) grid.

**First-principles VOP model (reliable upper bound).** Per G-element block, count VOP ops
(dot8 + addressing + epilogue; on RDNA2 the epilogue is 2 cvt + 2 mul + 1 add = 5 VOP,
no packed half2->float2):
```
G= 32: 15.0 VOP /  32 MAC  -> 0.469 VOP/MAC
G= 64: 21.0 VOP /  64 MAC  -> 0.328 VOP/MAC
G=128: 33.0 VOP / 128 MAC  -> 0.258 VOP/MAC
=> VOP-bound speedup: 64/32 = 1.43x, 128/32 = 1.82x
```

**LDS/VOP co-bound.** y-reads are 16B per 32 elem = 0.5 B/elem; at 128 B/cyc/CU that is
256 elem/cyc. The VOP dot8 is 256 MAC/cyc = 256 elem/cyc. They are BALANCED, so the
kernel is co-bound: the real speedup sits between the conservative (LDS-bound) 1.15/1.25
and the VOP-bound 1.43/1.82. The y-reads and dot8 are proportional to G (do NOT
amortize); only the FPU epilogue (and scale reads) amortize.

**Conclusion / gate: PASS.** The speedup is real and meaningful. Best estimate (central):
**64-block ~1.2-1.4x, 128-block ~1.3-1.8x** (optimistic model 1.33/1.60 is a good central
estimate; the benchmark's raw 1.4-1.9 / 2.0-3.3 is an overestimate). Combined with Phase
1b (64-block activation error ~15%, 128-block ~20%), the 64-block option offers a solid
speedup for a moderate accuracy cost. **Proceed to Phase 3** to get the definitive
real-model perplexity + pp512 numbers.

---

## Phase 3 - Full implementation (EXPENSIVE) - only if Phases 1+2 justify

This is a large, invasive change (a new GGML type touches many files: the type enum,
quantize/dequantize, the allocator, the graph, every backend's type switch, the GGUF
loader). Scope it carefully and keep it optional.

- Define new GGML types `Q4_0_64` and `Q4_0_128` (flat, one fp16 scale per block):
  block struct, `ggml_quantize_row_*`, `ggml_dequantize_row_*`.
- Add the CUDA dp8 kernel for the new types (the vec_dot accumulates 8/16 dot8 before
  the epilogue; the load-tiles repack uses the larger block).
- Add dispatch in the CUDA backend (mirror the existing Q4_0 dp8 path).
- Quantize-tool support to produce the new GGUF.
- (Option) two-level scaling for the 128-block if flat is too lossy (Q4_K-style).

### Combinations to measure (perplexity + pp512 speed)
| group_attn | group_ffn | rationale |
|-----------|-----------|-----------|
| 32 | 32 | baseline (current) |
| 64 | 64 | uniform, mild |
| 128 | 128 | uniform, max speed |
| 32 | 128 | attn-accurate / ffn-fast (H2 prediction) |
| 64 | 128 | balanced |
| 128 | 32 | attn-fast / ffn-accurate (opposite of H2) |

Find the Pareto-optimal (accuracy vs speed) point. The FLOP weighting (FFN ~69%) means
the ffn group size drives the speed, the attn group size drives the accuracy.

### Effort
~1-2 weeks (the new GGML type is the bulk of it). This is the part that would be hard
to keep upstream-acceptable; on a private fork it's freer.

---

## Phase 3 kernel RESULTS (2026-09-21) - real Q4_0_64, opposite of the prediction

`GGML_TYPE_Q4_0_64` (64-elem flat groups, one fp16 scale per block) was implemented:
type enum, quantize/dequantize, CPU vec_dot, and the RDNA2/gfx906 dp8 CUDA kernel
(mirrors the Q4_0 dp8 path: `mmq-vec-dot.cuh`, `mmq-load-tiles.cuh`, `mmq.cuh`,
`mmq-config-rdna2.cuh`, `quantize.cu`). Prefill (mmq) runs; decode (mmvq) is not
wired up yet (`GGML_ABORT`, tracked separately in `q40_64_decode_fix_plan.md`) - so
**no perplexity number exists yet for this type**, only pp1024 (Qwen3.5-2B, V620,
gfx1030).

Measured: pp1024 **5626-5666 t/s** vs the Q4_0 (32-block) baseline **7718-7807 t/s**
on the same build - **~27% SLOWER**, the opposite of Phase 2's predicted 1.2-1.4x
(64-block) speedup.

Investigated and fixed one real bug: the vec_dot's k01 loop has a 2-iteration trip
count, so `#pragma unroll 2` fully flattened it (unlike Q4_0's dp8 kernel, whose
4-iteration loop keeps a real 2-trip runtime loop under the same pragma) - doubling
the static `v_dot8`/`ds_read_b128` count for identical dynamic work. Fixed to
`#pragma unroll 1`; confirmed via `tools/vgpr-check.py` that dot8/VGPR/epilogue
counts now match Q4_0's shape (dot8 512=512, VGPR 250->207, FPU epilogue instructions
halved 132/128 -> 66/64, i.e. the amortization the design intended now shows up in
the compiled code). **Effect: only +2.9%** (5666 -> 5824-5832 t/s). ~25% gap to
Q4_0 remains.

Ruled out, with measurements (see `q40_64_decode_fix_plan.md` for the full log):
- **Tile width.** `GGML_CUDA_MMQ_J` swept 32/40/48/56/64 - flat at 5629-5836 t/s,
  not the cause.
- **Registers/occupancy/shared memory.** `rocprofv3 --kernel-trace --stats` shows
  the Q4_0_64 `mul_mat_q` kernel at 399 us/dispatch vs Q4_0's 248 us/dispatch
  (1.61x slower, identical grid/workgroup dims and dispatch counts on both), despite
  Q4_0_64 having *fewer* static instructions, *fewer* VGPRs, and *less* shared
  memory than Q4_0's kernel. All three point the wrong direction for a
  register/occupancy explanation.
- **dot8 accumulator chain depth.** The u4a+u4b combined read makes each `sumi`
  chain 8 `v_dot8` deep (vs Q4_0's 4-deep, twice as often). Splitting it into two
  independent 4-deep chains (combined only at the epilogue, same math) made it
  *worse* (5824 -> 5688 t/s), because the second live accumulator across all 32
  unrolled `(i,j)` slots raised VGPR 207 -> 239. Reverted.

**RESOLVED (2026-09-22): the gap was unaligned LDS reads, now fixed.**
Disassembling both `mul_mat_q<J=64>` kernels (`tools/kernel-disasm.py`) showed the
Q4_0_64 y-tile row is `sizeof(block_q4_0_64_mmq_dp8)` = 72B (2 half2 scales + 16
int qs) vs Q4_0's 80B (4 half2 scales + 16 int qs). The vec_dot reads the 128-bit
qs chunks as `int4` = `ds_read_b128`, which needs 16B alignment. With a 72B row
(qs at +8B) every y b128 read lands on an 8B boundary (offsets 264, 280, 840, ...
= 8 mod 16); Q4_0's are all 16B-aligned (272, 288, 912, ... = 0 mod 16). On RDNA
an unaligned b128 splits into multiple LDS transactions, so the y-operand fetch
(32 b128 per kb0 iteration) paid up to 4x Q4_0's LDS cost for the same data -
swamping the halved FPU epilogue. That is the whole regression; compute (512
dot8) and the epilogue halving were already correct. (Full log in
`q40_64_decode_fix_plan.md`.)

Fix: pad `block_q4_0_64_mmq_dp8` to 80B (add `int pad[2]` so qs sits at +16B,
matching Q4_0's geometry). Only the struct, its static_assert, and the vec_dot
`y_qs` offset (+2 -> +4) needed manual changes; the quantize writer, the y-tile
copy, `mmq_get_nbytes_shared`, and the workspace size all derive from `sizeof`
and update automatically. After the fix the b128 offsets are 272, 288, 912, ...
(all 16B-aligned) and the kernel matches Q4_0's LDS pattern.

**Measured (2026-09-22, same build, Qwen3.5-2B, V620 gfx1030, pp1024 ub512):**
Q4_0_64 **8446 t/s** vs Q4_0 baseline **7731 t/s** - now **~9% FASTER**. The sign
flipped from -27% to +9%. The end-to-end gain is below Phase 2's 1.15-1.33x
pure-MMQ-compute prediction, as expected: pp1024 also includes non-MMQ work
(attention scores, embeddings, norms) that does not scale with the MMQ block size.

**Consequence for this plan:** the Q4_0_64 prefill regression is root-caused and
fixed, so the Q4_0_128 kernel and the attn/ffn group-size sweep are unblocked on
the speed side. What is still missing is a **perplexity number**: decode (mmvq)
is not implemented for this type (see `q40_64_decode_fix_plan.md`), so the
accuracy side of the tradeoff is still unpriced.

---

## Risks

- **Activation outliers** make the 128-block activation quantization too lossy -> the
  whole approach fails. Phase 1b gates this.
- The **new GGML type is invasive** (many files) -> scope carefully, keep it
  macro-gated/optional, don't disturb the existing Q4_0 path.
- The **speedup may be smaller than modeled** (LDS/scalar contention) -> Phase 2 gates
  this.
- **Scope creep**: this is a new quant format, not a tuning tweak. It deserves its own
  branch and careful review, per the project's contribution guidelines.

---

## Phase 1a RESULTS (done, 2026-09-19)

Model is a **hybrid** (Qwen3-Next style): 18 linear-attention (gated-delta-net, `ssm_*`)
blocks + 7 full-attention blocks (`attn_q/k/v/output`), all 25 with SwiGLU FFN. Source: the
original BF16 GGUF (3.9 GB, 335 tensors). Tool: `tools/quant-err/quant-err.cc` (reuses
ggml's GGUF loader, exact Q4_0 convention for all group sizes, fp16 scale).

Relative L2 error sqrt(sum e^2 / sum x^2), per category:

| category            | G=32 (Q4_0) | G=64   | G=128  | 32->128 ratio | elems  |
|---------------------|-------------|--------|--------|---------------|--------|
| attn (full+linear)  | 9.07%       | 10.22% | 11.35% | 1.251x        | 3.75e8 |
| ssm (delta-net)     | 9.10%       | 10.24% | 11.34% | 1.246x        | 7.67e7 |
| ffn (SwiGLU)        | 8.90%       | 9.96%  | 10.97% | 1.232x        | 9.44e8 |
| embd (token embd)   | 9.16%       | 10.35% | 11.53% | 1.258x        | 5.38e8 |
| other (nextn/spec)  | 9.90%       | 12.59% | 16.15% | 1.63x         | 8.4e6  |

Findings:
- **~9% Q4_0 error validates the tool** (4-bit -> ~9% relative L2 is expected).
- **32->128 adds ~25% relative error (9%->11.3%), i.e. ~56% more MSE** - a real, uniform cost.
- **The "attention hurts more" hypothesis is NOT supported on the weight side.** The 32->128
  ratio is nearly identical across attn/ssm/ffn/embd (1.23-1.26x). FFN is marginally the LEAST
  sensitive globally (1.232x): it has the largest individual outliers (max|e| 0.055) but the
  smallest global error (its signal is also larger, so relative error is smaller).
- **Caveat:** global L2 dilutes outliers, so this cannot see the outlier-sensitivity that
  actually drives perplexity. That lives in the ACTIVATION (x) quantization - untested here.

Implication: on the weight side, a **uniform** 128-block (attn=ffn=128) degrades all layers
roughly equally - no per-layer group-size scheme is needed for the weights. The open question
is the activation side (Phase 1b) and the real perplexity (Phase 3).

---

## Phase 1b RESULTS (done, 2026-09-19)

Method: a temporary env-gated hook in `llama_context::graph_compute` (`LLAMA_DUMP_ACT_DIR`) dumps
each MUL_MAT's **x activation** (`src[1]`, the reduction-dim input the dp8 kernel quantizes) as
fp32 during the prefill. Ran the **BF16** model on CPU (`-ngl 0`) with a 236-token calibration
prompt -> 373 activation files. Re-quantized to 32/64/128 with the exact Q4_0 convention
(`tools/quant-err/act-err.cc`), groups along the contiguous reduction dim per token. Categories
are the GEMM the activation feeds.

Quantization error (relative L2) per GEMM category:

| cat       | G=32    | G=64    | G=128   | 32->128 | max\|e\|@128 |
|-----------|---------|---------|---------|---------|--------------|
| attn_in   | 12.12%  | 15.46%  | 19.80%  | 1.633x  | 4.468        |
| attn_out  | 12.46%  | 15.88%  | 20.32%  | 1.630x  | 4.468        |
| ffn_in    | 12.30%  | 15.67%  | 20.05%  | 1.630x  | 4.468        |
| ffn_mid   | 11.23%  | 14.13%  | 17.47%  | 1.557x  | 4.468        |
| ssm       | 11.54%  | 14.62%  | 18.52%  | 1.605x  | 4.468        |
| embd      | 12.13%  | 14.60%  | 17.54%  | 1.446x  | 1.600        |

(attn_in = qkv/gate/q/k/v inputs; ffn_in = gate/up inputs; ffn_mid = post-SwiGLU feeding down;
embd = final hidden feeding lm_head.)

Outlier structure of the dumped activations:

| cat       | max\|x\| | RMS(x) | max/RMS |
|-----------|----------|--------|---------|
| attn_in   | 73.27    | 1.398  | 52.4x   |
| attn_out  | 73.27    | 1.992  | 36.8x   |
| ffn_in    | 73.27    | 1.819  | 40.3x   |
| ffn_mid   | 73.27    | 1.066  | 68.8x   |
| ssm       | 73.27    | 1.233  | 59.4x   |
| embd      | 25.79    | 3.058  | 8.4x    |

Findings:
- **The "attention hurts more" hypothesis is REFUTED.** attn_in (1.633x) ~ ffn_in (1.630x) ~
  attn_out (1.630x): attention and FFN inputs have essentially identical group-size sensitivity.
  A uniform group size is correct - no per-layer scheme is needed on the activation side either.
- **The outliers are a shared property of the residual stream.** max\|x\| = 73.27 is *identical*
  across attn_in/attn_out/ffn_in/ffn_mid/ssm: one dominant outlier channel propagates through the
  residual stream and shows up in every downstream activation. max/RMS is 37-69x (very
  outlier-prone) - this is the classic LLM.int8() activation-outlier phenomenon.
- **ffn_mid (post-SwiGLU) is the LEAST sensitive (1.557x)** despite the highest max/RMS (68.8x):
  SiLU produces *isolated* outliers (one per group), so a bigger group's scale is set by the
  outlier but the small neighbours still quantize fine - the error grows slowly.
- **embd (final hidden, pre-lm_head) is the tamest** (max/RMS 8.4x, 1.446x): the final norm
  tames the outlier (73 -> 26). The lm_head GEMM input is easy to quantize.
- **The activation is the accuracy bottleneck, not the weight.** Activation error is higher than
  weight error at every group size (12.1% vs 9.1% at 32) and grows faster (32->128 = 1.63x vs
  1.25x). At 128-block the residual activations hit ~20% relative error with a max single-value
  error of 4.47 (huge vs RMS ~1.4) - expect a real perplexity hit.

Implication: no per-layer group-size scheme is needed (uniform is fine on both sides). The open
decision is **64 vs 128**, driven by the activation cost: 64 ~15% act error (moderate) vs 128
~20% (large). Only the real perplexity (Phase 3) decides whether the extra speedup of 128 is
worth that cost.

---

## Recommendation (updated after Phase 1a + 1b)

Both cheap accuracy gates are done. The "attention is more sensitive" hypothesis is **refuted on
both sides**: weights (1a) and activations (1b) degrade uniformly across attn/ffn/ssm. A **uniform
group size** is correct - no per-layer scheme is needed. The activation is the accuracy bottleneck
(outliers, ~20% error at 128 vs ~9% weight error), so the decision is **64 vs 128**, and only the
real perplexity can price it.

Remaining, in order of cost:
- **Phase 2** (1-2 days, no model changes): confirm the compute speedup in `mmq-bench` for
  64/128-block kernels (pin down the 1.15-1.6x range; the conservative end is the risk).
- **Phase 3** (1-2 weeks): build a real 64-block AND 128-block model, measure perplexity + pp512.
  This is the definitive ship/don't-ship answer. Given 1b, 128-block activations are ~20% error
  (likely a real ppl hit); 64-block ~15% (moderate). If 128's ppl degrades too much but 64's
  speedup is enough, ship 64.

Tradeoff so far: uniform group size; activation error 32->64 ~12->15%, 32->128 ~12->20%; for
~1.15-1.6x MMQ compute speedup. Whether it's worth it is a perplexity question (Phase 3).

**Update (2026-09-22): the Q4_0_64 prefill regression is fixed.** The ~25% pp1024
regression was root-caused to unaligned LDS reads from the 72B y-tile row and
fixed by padding `block_q4_0_64_mmq_dp8` to 80B (see "Phase 3 kernel RESULTS"
above). Q4_0_64 is now **~9% FASTER** than Q4_0 on pp1024 (8446 vs 7731 t/s).
There is still no perplexity number (decode/mmvq is unimplemented for this type,
tracked in `q40_64_decode_fix_plan.md`), so the accuracy side of the tradeoff is
unpriced. The accuracy-side conclusions above (uniform group size, activation is
the accuracy bottleneck) still stand. Q4_0_128 and the attn/ffn group-size sweep
are now unblocked on the speed side; the remaining gate is a perplexity number.
