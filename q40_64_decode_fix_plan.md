# Q4_0_64 decode (mmvq) fix plan

Status: PLANNED, not implemented.

## Why

`llama-perplexity` runs prefill (mmq) + decode (mmvq). The Q4_0_64 prefill (mmq)
path is implemented and runs (5626 t/s pp1024), but the decode (mmvq) path aborts
at `mmvq.cu:1347` (`default: GGML_ABORT("fatal error")`) because `GGML_TYPE_Q4_0_64`
was never added to the decode kernel. This blocks the perplexity / correctness
check needed for the ship/no-ship decision.

The decode-path gap does NOT affect the prefill perf number (llama-bench pp1024
is prefill-only and never touches mmvq).

## The core problem: 64-vs-32 block mismatch

- Q4_0_64 weight block = 64 elements (nibbles) = 8 x int32. (`QK4_0_64=64`, `QI4_0_64=8`)
- Q8_1 activation block = 32 elements (bytes) = 8 x int32. (`QK8_1=32`)

So one Q4_0_64 block spans TWO Q8_1 blocks. The existing `vec_dot_q_cuda_t`
signature takes a single `bq8_1` pointer:

    typedef float (*vec_dot_q_cuda_t)(const void* vbq, const block_q8_1* bq8_1,
                                      const int& kbx, const int& iqs);

The call site (mmvq.cu ~666/849) drives:

    kby = kbx * (qk/QK8_1);   // = kbx*2 for Q4_0_64 -> points at FIRST Q8_1 block
    kqs = vdr * (tid % (qi/vdr));  // = {0,2,4,6} for Q4_0_64 (vdr=2, qi=8)

So `bq8_1` already points at the first of the two Q8_1 blocks. The vec_dot just
needs to pick the right half from `iqs`:

    half    = iqs / 4;   // 0 for iqs in {0,2}, 1 for iqs in {4,6}
    iqs_in  = iqs % 4;   // 0 or 2
    bq8_1h  = bq8_1 + half;
    // weight nibbles:  get_int_b2(bq4_0_64->qs, iqs + i)        (full 8-int block)
    // activation:      get_int_b4(bq8_1h->qs, iqs_in + i)
    //                  get_int_b4(bq8_1h->qs, iqs_in + i + QI4_0)
    // weight scale:    bq4_0_64->d      (one per 64, same for both halves)
    // activation scale:bq8_1h->ds       (one per 32, differs per half)

No call-site change is required: `kby` already lands on the first Q8_1 block and
`bq8_1 + half` reaches the second.

## Changes

### 1. ggml/src/ggml-cuda/vecdotq.cuh
- Add `#define VDR_Q4_0_64_Q8_1_MMVQ 2` (match Q4_0; block processed in 4 calls).
- Add `vec_dot_q4_0_64_q8_1_impl<vdr>` - identical body to
  `vec_dot_q4_0_q8_1_impl` (dp4a loop + `d4 * (sumi*ds8f.x - (8*vdr/QI4_0_64)*ds8f.y)`).
  Note the constant term uses `QI4_0_64` (8), not `QI4_0` (4).
- Add `vec_dot_q4_0_64_q8_1(...)` wrapper implementing the half-select logic above.

### 2. ggml/src/ggml-cuda/mmvq.cu (required dispatch)
- `get_vec_dot_q_cuda` (~line 15): `case GGML_TYPE_Q4_0_64: return vec_dot_q4_0_64_q8_1;`
- `get_vdr_mmvq` (~line 44): `case GGML_TYPE_Q4_0_64: return VDR_Q4_0_64_Q8_1_MMVQ;`
- `mul_mat_vec_q_switch_type` (~line 1220):
  `case GGML_TYPE_Q4_0_64: mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0_64>(...);`

### 3. ggml/src/ggml-cuda/mmvq.cu (optional tuning, default is safe)
These are per-arch tuning tables with a `default` fallback; Q4_0_64 works without
them but will use the generic (conservative) batch/nwarps. Add only if tuning
decode perf matters:
- max-batch tables: `get_mmvq_mmid_max_batch_pascal_older` (~136), `_gcn` (~174),
  `_turing_plus` (~243), CDNA1 block (~332). Mirror Q4_0's value.
- nwarps whitelists: `calc_nwarps` RDNA4 (~440), RDNA3_0 (~463), GB10 (~511).
  Only relevant on those archs; gfx1030 (RDNA2) is not in these lists, so skip.

## Already in place (no action)
- `ggml_cuda_type_traits<GGML_TYPE_Q4_0_64>` (qk=64, qr=2, qi=8) in common.cuh.
- `block_q4_0_64` (ggml-common.h), CPU `vec_dot_q4_0_64_q8_0_generic`, quantize.

## Verification
1. Build ggml-hip.
2. `llama-perplexity -m Q4_0_64.gguf -f wikitext-2-raw/wiki.test.raw -c 512 -ngl 99 -dev ROCM0`
   - Must run to completion (no abort).
   - PPL should be near the Q4_0 baseline (15.53). A large jump indicates a
     quantization or vec_dot bug.
3. `llama-cli -m Q4_0_64.gguf -ngl 99 -c 512 -n 30 -st -p "..." -dev ROCM0`
   - Output should be coherent (sanity check).
4. Re-run `llama-bench` pp1024 to confirm prefill perf is unchanged by the
   decode fix (it should be - separate path).

## Prefill (mmq) regression (separate from this fix) - RESOLVED
The prefill (mmq) Q4_0_64 kernel was 27% SLOWER than Q4_0 (5626 vs 7718 t/s), the
opposite of the expected speedup. This is a distinct issue from the decode gap and
is now root-caused and fixed: unaligned LDS reads from the 72B y-tile row (see
"Prefill (mmq) regression: root cause" below). After the fix Q4_0_64 is ~9%
FASTER than Q4_0 (8446 vs 7731 t/s). The decode fix does not address it, and this
fix does not address decode.

### Investigation (2026-09-21, on the V620)

Found and fixed one real bug in `ggml_cuda_mmq_vec_dot_q4_0_64_q4_0_64_dp8`
(`mmq-vec-dot.cuh`): the k01 loop has a 2-iteration trip count (64-elem group /
2*QR4_0*VDR = 2), so `#pragma unroll 2` fully flattens it - unlike the Q4_0 dp8
kernel's 4-iteration loop, where `unroll 2` leaves a real 2-trip runtime loop.
Full unroll doubled the static `v_dot8`/`ds_read_b128` count (1024/62 vs 512/32,
via `tools/vgpr-check.py`) for the same dynamic work, since the compiler could no
longer share address/wait bookkeeping across iterations. Changed to
`#pragma unroll 1` to force a real loop, matching Q4_0's shape. Verified: dot8
count now matches Q4_0 exactly (512), VGPR dropped 250->207, FPU-epilogue
instructions (`v_cvt`/`v_fma`/`v_mul`/`v_add`) correctly halved (132/128 ->
66/64, confirming the intended amortization now shows up in the compiled code).

Effect: pp1024 (ub512, Qwen3.5-2B) 5666 -> 5824-5832 t/s, only +2.9%. Gap to the
Q4_0 baseline (7777-7807 t/s on the same build) is still ~25%.

**The remaining gap does not trace to compute, registers, or occupancy** -
`rocprofv3 --kernel-trace --stats` on both runs (same grid/workgroup dims,
128/512/1024/1536 x 64, 256 threads, same dispatch counts) shows the Q4_0_64
`mul_mat_q<..., J=64, fallback=0>` kernel spends **399 us/dispatch vs Q4_0's
248 us/dispatch (1.61x slower)**, despite Q4_0_64 having *fewer* static
instructions (1790 vs 2071), *fewer* VGPRs (207 vs 256, i.e. equal-or-better
occupancy headroom), and *less* shared memory (smaller x_df/y-tile). Every
static lever checked (`GGML_CUDA_MMQ_J` sweep 32/40/48/56/64 - flat around
5630-5836; `mmq_get_config` row - byte-identical to Q4_0's) points away from
the usual suspects.

Global memory access pattern for the weight load (`load_tiles`) was checked
too, and also does not explain it - if anything it favors Q4_0_64: `block_q4_0`
is `{fp16 d; qs[16B]}`=18B/32 elems, `block_q4_0_64` is `{fp16 d; qs[32B]}`=
34B/64 elems. Per warp, Q4_0's `qs` load touches 8 blocks (`kbx=txi/QI4_0`,
QI4_0=4) spanning 8*18=144B, of which 8*16=128B is real qs data and 8*2=16B is
skipped `d`-scale gaps (8 separate 2B gaps). Q4_0_64 touches 4 blocks
(`kbx=txi/QI4_0_64`, QI4_0_64=8) spanning 4*34=136B, 4*32=128B real qs data,
4*2=8B gaps (4 gaps). Both warps read the same 128B of useful qs data (fixed
by `warp_size*4B`), but Q4_0_64 has *half* the gap bytes and a better
useful/total ratio (94.1% vs 88.9%) - the opposite of what would be needed to
explain a slowdown.

Leading hypothesis, not yet confirmed: the u4a+u4b combined read makes each
`sumi` accumulator chain 8 `v_dot8` deep before its one epilogue, vs Q4_0's
4-deep chain twice as often. gfx1030 has no cache/VALU counters (see
PROFILE-NAVI21.md section 10), so this can't be confirmed with hardware
counters, only by further A/B kernel-structure experiments (e.g. forcing 4
loop trips of one int4 read each, still with a shared/2x-amortized epilogue,
to isolate chain depth from trip count).

### Chain-depth hypothesis tested and rejected (2026-09-21)

Split the 8-deep `sumi` chain into two independent 4-deep chains (`sumi_a` for
u4a, `sumi_b` for u4b), combined only at the epilogue (`sumi_a + sumi_b`,
same math). Result: pp1024 5824-5832 -> 5688 t/s (-2.3%, worse), VGPR
207 -> 239 (`tools/vgpr-check.py`). The second live accumulator across all 32
unrolled `(i,j)` slots (32 -> 64 live `sumi` registers) cost more in register
pressure than the shorter dependency chain saved. Reverted. Chain depth is not
the (or at least not a cheaply-fixable) lever - do not retry this shape without
also finding a way to avoid the extra live accumulators (e.g. reusing one of
the two i0/j0-unrolled slots' registers, which would need a real restructure,
not a local one).

## Prefill (mmq) regression: root cause found - unaligned y reads (2026-09-22)

Disassembled both `mul_mat_q<..., J=64, fallback=false>` kernels (extracted
from the `.hip_fatbin` section of `build-dp8-gfx1030/bin/libggml-hip.so` with
`llvm-objdump`) and diffed the LDS read patterns.

### High-level kernel overview

Both dp8 kernels have the same structure: 256 threads (8 warps), each kb0
iteration loads one x tile (I=128 rows x 256 elems, the weights = src0) and one
y tile (J=64 rows x 128 elems, the quantized activations = src1) into LDS, then
each warp walks its 8 y rows x 4 x rows in a fully unrolled j0/i0 grid with a
small k01 inner loop. Per (i,j)
pair and k01 trip each lane does 8 `v_dot8_i32_i4` (8 signed-nibble dot8s =
64 elements), converts the accumulator to float, and does one FMA with
`x_scale * y_scale`. Q4_0_64's only intended difference: one weight block
covers 64 elements, so the k01 loop has 2 trips instead of 4 and the
scale-convert/FMA epilogue runs half as often (amortization win).

The y tile row in LDS is the quantized block struct itself:

- Q4_0: `block_q4_0_mmq_dp8` = 4 half2 scales + 16 int qs = 80B (20 ints)
- Q4_0_64: `block_q4_0_64_mmq_dp8` = 2 half2 scales + 16 int qs = 72B (18 ints)

The vec_dot core reads the 128-bit qs chunks as `int4` = `ds_read_b128`, which
requires 16B alignment to be a single LDS transaction.

### The problem: Q4_0_64's y reads are 8B-unaligned

- Q4_0: qs starts at +16B in the row, row stride 80B (multiple of 16B) ->
  every b128 read is 16B-aligned. Disassembly offsets: 272, 288, 912, 928,
  1552, ... (all = 0 mod 16).
- Q4_0_64: qs starts at +8B in the row, row stride 72B (NOT a multiple of
  16B) -> every b128 read lands on an 8B boundary. Disassembly offsets:
  264, 280, 840, 856, 1416, ... (all = 8 mod 16).

On RDNA2 an unaligned `ds_read_b128` is split into multiple smaller LDS
transactions (2x b64 / 4x b32 worth of issue+LDS work). The y-operand fetch
is 32 b128s per kb0 iteration, so Q4_0_64 pays up to 4x the LDS cost of the
baseline for the same 4KB of y data - which swamps the halved FPU epilogue
and explains the 1.61x per-dispatch time (399 vs 248 us) and the ~27% pp1024
regression.

### Instruction mix (static, per kb0 iteration, J=64)

| op                 | Q4_0   | Q4_0_64 |
|--------------------|--------|---------|
| v_dot8_i32_i4      | 512    | 512     |
| ds_read_b128 (y)   | 32     | 32      |
| ds_read2_b32 (x)   | 56     | 40      |
| ds_read_b32        | 8      | 16      |
| v_cvt_f32_i32 (ep) | 128    | 64      |

Compute (512 dot8s) is identical; the epilogue halving (128 -> 64 cvts) is
present as designed. The x-side reads are the same shape in both. The one
structural difference that matters is the alignment of the 32 y b128 reads
(aligned in Q4_0, 8B-unaligned in Q4_0_64).

### Fix (applied 2026-09-22)

Pad `block_q4_0_64_mmq_dp8` to 80B (20 ints) so the LDS y row matches the Q4_0
geometry: add 2 pad ints between the 2 scales and qs. Then qs starts at +16B,
row stride is 80B, and all 32 y b128 reads are 16B-aligned again (disassembly
offsets now 272, 288, 912, ... = 0 mod 16, identical to Q4_0). Cost: +8B per
128 activations in the quantized workspace and the y tile (y tile at J=64:
5120B vs 4608B, still far below the LDS budget).

Only two manual edits were needed; everything else derives from `sizeof` or
struct-member access and updates automatically:
- `mmq.cuh`: add `int pad[2]` to `block_q4_0_64_mmq_dp8`, update its
  static_assert to `sizeof(...) == sizeof(block_q4_0_mmq_dp8)`.
- `mmq-vec-dot.cuh`: `y_qs` offset +2 -> +4 (qs now at int 4). The scale index
  `k01/(2*QI8_1)` = k01/16 is already correct (QI8_1=8 -> half2 indices {0,1}
  = ds2[0], ds2[1]) and is unchanged.
- Nothing else: the `quantize_mmq_q8_1` writer uses struct-member access
  (`.ds2`, `.qs`), the y-tile copy and `mmq_get_nbytes_shared` use `sizeof`,
  and the workspace size in `mmq.cu` uses `sizeof` - all auto-update.

**Result (2026-09-22, Qwen3.5-2B, V620 gfx1030, pp1024 ub512, same build):**
Q4_0_64 **8446 t/s** vs Q4_0 baseline **7731 t/s** - now **~9% FASTER**, the
regression is gone and the sign flipped from -27% to +9%. PPL still cannot be
measured: decode (mmvq) is not implemented for this type (see the sections
above), so the accuracy side of the tradeoff remains unpriced.
