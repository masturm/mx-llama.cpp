# gfx906 (MI50/MI60) architecture notes for kernel tuning

Reference for tuning the Q4_0 dot8 MMQ kernel on gfx906 (AMD Instinct MI50/MI60,
Vega 20). Values verified against `rocminfo` on this machine (4x MI50, 32 GB HBM2
each) and the disassembled kernels in `rocprof_q40_dp8/`. Focus is on what actually
matters for the MMQ prefill kernel; general background is kept brief.

## Key specs

| Parameter | Value | Why it matters for tuning |
|---|---|---|
| Compute Units (CU) | 60 | Total parallelism; `nsm` in the code. |
| SIMDs per CU | 4 | 240 SIMD total. A wave is scheduled to one SIMD. |
| Wavefront size | 64 | wave64, not 32. Thread counts and warp-level ops must be multiples of 64. |
| LDS (local data) per CU | 64 KB | Caps blocks/CU. A 32 KB block -> 2 blocks/CU. |
| Max waves per CU | 40 | Hardware wave slots (10 per SIMD). |
| Max work-items per CU | 2560 | = 40 waves x 64. |
| Workgroup max size | 1024 | 512-thread blocks are well within this. |
| VGPR per wave (max) | 512 | Full register file at 1 wave/SIMD. Occupancy divides this down. |
| L1 (per CU) | 16 KB | Instruction + data cache. |
| L2 | 8 MB | Shared across the GPU. |
| Memory | HBM2, 32 GB | High bandwidth; the X (weight) tile is the VRAM traffic. |
| Peak clock | ~1725 MHz | Boost; real clocks run lower under load. |

## Compute structure and occupancy

- A **wave** is 64 threads (one SIMD lane per thread). A **workgroup** is a multiple
  of waves scheduled to a CU; its waves are spread across the CU's 4 SIMDs.
- **Occupancy** = resident waves per SIMD. It is limited by three resources, and the
  tightest one wins:
  - LDS: 64 KB/CU. A block using 32 KB allows 2 blocks/CU.
  - VGPR: 512/wave max, so N waves/SIMD caps VGPR at 512/N.
  - Wave slots: 40/CU = 10/SIMD.
- The 512-thread (8-wave) config used for Q4_0 at J=128: 2 blocks/CU (LDS) x 8
  waves = 16 waves/CU = **4 waves/SIMD**. That caps VGPR at 512/4 = **128/wave**.
  This is the "128 VGPR cap" that a full unroll of the dot loop hits (it spills to
  scratch past 128). The measured sweet spot sits at 112-120 VGPR, just under the cap.

Tuning implication: raising the thread count (256 -> 512) raised occupancy (4 -> 8
waves/block, 2 -> 4 waves/SIMD here because the block also got wider) and let a wider
J tile amortize the X-tile VRAM load. That was item (a), +8.3%.

## Register file (VGPR)

- 512 VGPR per wave at 1 wave/SIMD. At 4 waves/SIMD the usable cap is 128/wave.
- The accumulator `sum[J*I/(nwarps*warp_size)]` dominates VGPR. For J=128, I=128,
  nwarps=8: 128*128/(8*64) = 32 floats per lane, same as J=64/nwarps=4 (64*128/(4*64)
  = 32). So widening J without adding waves does not grow the accumulator.
- Spilling to scratch (local memory) is very expensive on this ISA: the full-unroll
  experiment that hit 128 VGPR + scratch collapsed prefill throughput by ~64%.
  Always check `arch_vgpr` and `scr` (scratch) in the rocprof CSV before keeping a
  change that adds registers.

## Local Data Share (LDS)

- 64 KB per CU, 32 banks, 4 bytes per bank (128 bytes/cycle aggregate).
- Read widths available: `ds_read_b16`, `ds_read_b32`, `ds_read_b64` (as
  `ds_read2_b32`, two 32-bit), and `ds_read_b128` (16 bytes). The compiler picks the
  width, but only up to what it can prove aligned and contiguous.
- **Wider reads cut the instruction count, not the byte throughput.** A 64-lane
  wave reading 16 B moves the same bytes as four 4-B reads, but issues 1 instruction
  instead of 4 and shortens the LDS->dot dependency chain. On a latency-bound kernel
  that is the win.
- **Bank conflicts** occur when multiple lanes in a wave hit the same bank in one
  cycle. The MMQ tiles are padded so the Q4_0 dot8 kernel runs at 0 conflicts
  (verified via the `LDSBankConflict` counter).
- The y tile row is 80 B (20 ints): 4 half2 scales + 16 packed ints. It is 16 B
  aligned, so the 4 y operands load as one `ds_read_b128`. The x tile row stride is
  33 ints (odd), so x operands are not uniformly 16 B aligned and stay `ds_read_b32`
  / `ds_read2_b32`.

Tuning implication: item (c) forced the 4 y operands into one `int4` (16 B) load.
The compiler had only gone as far as two 8-B (`ds_read2_b32`) reads on its own. The
per-wavefront LDS instruction count dropped ~34% (6033 -> 3985) and throughput rose
+4.6%.

## The v_dot8_i32_i4 instruction

- The core primitive of the dp8 path: one instruction computes 8 int4 x int4 products
  (with an accumulate) per lane. It replaces a scalar 8x multiply-add sequence, which
  is why the Q4_0 dot8 path exists at all.
- It is a VALU-class op but does more work per issue than a plain VALU, so the kernel
  is not VALU-throughput saturated (measured VALUBusy ~50-55%). The kernel is
  latency/occupancy bound, which is why more waves (item a) and more ILP (items b, c)
  help rather than a raw ALU count.
- The Q4_0 nibbles are sign-flipped and packed into the int4 operand layout at tile
  load time (see `mmq-load-tiles.cuh`); the activations are pre-packed at quantize
  time. That in-tile weight repack is ALU work on the load path - a candidate to move
  to a one-time model-load repack if a profile shows ALU pressure there.

## Memory hierarchy

- L1 16 KB/CU, L2 8 MB, then HBM2 (32 GB). The X (weight) tile is streamed from HBM
  into LDS once per block and reused across the block's J output columns. A wider J
  tile (item a) reuses that HBM fetch over 2x the columns, which is why per-output
  weight bandwidth (FlatVMem) dropped ~31% when J went 64 -> 128.
- The activations (Y) are small per tile and stay mostly in LDS/L1.

## Mapping to the Q4_0 dot8 tuning items

| Item | Change | Hardware reason | Measured (pp512) |
|---|---|---|---|
| (a) | 256 -> 512 threads, J 64 -> 128 | more waves/SIMD (occupancy) + wider tile amortizes X HBM load | 984 -> 1065 (+8.3%) |
| (b) | `#pragma unroll 2` on the k01 loop | more ILP to hide LDS/dot latency, under the 128 VGPR cap | 1065 -> 1128 (+6.0%) |
| (c) | 16 B (`int4`) y load | one `ds_read_b128` instead of two `ds_read2_b32`, fewer LDS insts | 1128 -> ~1182 (+4.6%) |
| (d) | store x scale as `half` not `float` | (dropped, see below) | - |

Baseline before (a): 984.52 t/s. After (a)+(b)+(c): ~1182 t/s, +20% total.

### Why (d) was dropped

`x_df` is the Q4_0 per-block scale `d`. Storing it as `half` rounds each scale to
~0.05-0.1% relative error - a small but real accuracy cost (fp16 rounding of a
value the original quantization kept as float32). The benefit is also marginal:
the disassembled J=128 kernel has 128 epilogue scale multiplies but only 16
`ds_read_b32` float reads, so the compiler already hoists `x_df` out of the
`j0` output-column loop (it is invariant across `j0`, 2 values/lane for
I=128/warp64) and reuses it across all 16 output columns. Little LDS traffic is
left to save. Dropped: small accuracy cost + marginal benefit.

## Counters that matter (rocprof v1)

Collected via `rocprof_q40_dp8/rocprof_input.txt`. Per-wavefront, on the Q4_0 kernel:

- `VALUBusy` / `MemUnitBusy`: which side is hotter. Neither saturated here -> latency
  bound.
- `LDSInsts` / `FlatVMemInsts`: instruction counts; dropping these (via wider reads,
  wider tiles) is the lever.
- `LDSBankConflict`: must stay 0; the tiles are padded for this.
- `arch_vgpr` / `scr`: register pressure and spills. Keep `scr` at 0.
