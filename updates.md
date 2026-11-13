# K3 updates (branch `k3-r`)

Work on the SpacemiT K3 (8 X100 cores + 8 A100 cores) with XPU-RT and ModelBlaster. Newest first.

## 2026-11-13: convolution algorithms and layouts, compared fairly on both K3 cores

### Why this was redone

The comparison shown on 2026-11-05 put numbers from different experiments side by side. The indirect GEMM rows were early prototypes: built before the A100 fixes, with weights rearranged on every call. The implicit GEMM row was the installed kernel with every fix. So "implicit beats indirect" was not a valid conclusion. This time every variant ran under the same conditions.

### What was compared

Three ways to compute a convolution:

- **Direct**: the RVV vector kernel. It loops over the convolution itself and does not use the matrix engine.
- **Implicit GEMM**: matrix engine. Each input tile (a few output pixels × 16 or 8 values of their patches) is built from the image on the fly with address arithmetic. This is what the installed kernels do.
- **Indirect GEMM**: matrix engine. A table of pointers, built once per layer, says where each patch's data sits. Padding points at a zero buffer, so the input is never copied.

Each matrix version comes in two memory layouts, **NCHW** and **NHWC**, on both core types. A "patch" is the block of input values one output value is computed from (3×3 × input channels for a 3×3 conv).

### How (the same conditions for every variant)

- **One program per core type and kernel type**, holding every variant. All variants run on the same layers, with the same random data, against the same scalar reference.
- **Within a core, every matrix variant uses the installed kernel's own matrix loop, weight tile format and output step.** They differ only in how the input tile is built, and where NHWC output is written.
- **Common setup for every matrix variant:**
  - Weights pre-arranged in the matrix engine's order, not timed.
  - Implicit variants read a zero-padded input buffer kept per layer; only the interior is copied each call.
  - Indirect tables are built on the first call for a layer and kept.
  - The A100 fixes are in: `vl`/`vtype` clobbers, no GCC auto-vectorization in the A100 code, 8-byte moves, and BN+SiLU tables kept across calls.
- **Correctness:** every variant is bit-exact against the reference on every layer.
- **Timing:** the median of 21 calls after one warm-up call. Each bench ran twice, and totals agree within 3% (most within 1%).
- **Units and cores:** times are ticks of the 24 MHz clock (1 tick = 41.7 ns). X100 runs on cpu 0, A100 on cpu 8.
- **Conversion columns:** NHWC variants get NHWC input and write NHWC output. The cost of converting between the layouts is timed separately (see below).

"Without the first layer" leaves out each network's first conv, which has only 3 input channels (RGB). NHWC cannot use its fast copies there, so that one layer dominates some totals.

### Results

#### A100, plain conv (13 DroNet + YOLOv8-nano layers)

| Algorithm | Layout | All layers (ticks) | vs installed | Without the first layer | vs installed | Layers won |
|---|---|---:|---:|---:|---:|---:|
| Direct (RVV vector, no matrix engine) | NCHW | 162,796 | 4.30x | 110,760 | 4.19x | 0 |
| Direct (RVV, channel-blocked) | NCHW | 161,474 | 4.26x | 107,184 | 4.05x | 0 |
| Implicit GEMM, installed today | NCHW | 37,886 | 1.00x | 26,462 | 1.00x | 0 |
| Implicit GEMM | NCHW | 36,425 | 0.96x | 25,126 | 0.95x | 1 |
| Indirect GEMM | NCHW | 37,606 | 0.99x | 25,668 | 0.97x | 0 |
| Implicit GEMM | NHWC | 39,544 | 1.04x | 18,660 | 0.71x | 0 |
| Indirect GEMM | NHWC | 43,088 | 1.14x | **18,052** | 0.68x | 12 |
| Best variant per layer, NCHW only | | 34,630 | 0.91x | 23,330 | 0.88x | |
| Best variant per layer, NHWC allowed | | 29,352 | 0.77x | 18,052 | 0.68x | |
| Converting NCHW->NHWC before and back after every layer | | 16,432 | | | | |

#### A100, fused conv+BN+SiLU (35 YOLOv8-nano layers, 160x160)

| Algorithm | Layout | All layers (ticks) | vs installed | Without the first layer | vs installed | Layers won |
|---|---|---:|---:|---:|---:|---:|
| Direct (RVV vector, no matrix engine) | NCHW | 2,073,126 | 4.48x | 1,868,237 | 4.44x | 0 |
| Implicit GEMM, installed today | NCHW | 462,343 | 1.00x | 421,084 | 1.00x | 0 |
| Implicit GEMM | NCHW | 438,524 | 0.95x | 397,998 | 0.95x | 1 |
| Indirect GEMM | NCHW | 502,967 | 1.09x | 453,998 | 1.08x | 0 |
| Implicit GEMM | NHWC | 333,408 | 0.72x | 277,540 | 0.66x | 10 |
| Indirect GEMM | NHWC | 340,738 | 0.74x | **275,492** | 0.65x | 24 |
| Best variant per layer, NCHW only | | 428,060 | 0.93x | 387,534 | 0.92x | |
| Best variant per layer, NHWC allowed | | 314,304 | 0.68x | 273,777 | 0.65x | |
| Converting NCHW->NHWC before and back after every layer | | 74,800 | | | | |

#### A100, fused conv+BN+SiLU (35 YOLOv8-nano layers, 64x96)

| Algorithm | Layout | All layers (ticks) | vs installed | Without the first layer | vs installed | Layers won |
|---|---|---:|---:|---:|---:|---:|
| Direct (RVV vector, no matrix engine) | NCHW | 518,122 | 3.77x | 469,416 | 3.68x | 0 |
| Implicit GEMM, installed today | NCHW | 137,327 | 1.00x | 127,408 | 1.00x | 0 |
| Implicit GEMM | NCHW | 128,126 | 0.93x | 118,424 | 0.93x | 2 |
| Indirect GEMM | NCHW | 124,272 | 0.90x | 113,740 | 0.89x | 1 |
| Implicit GEMM | NHWC | 101,644 | 0.74x | 88,206 | 0.69x | 2 |
| Indirect GEMM | NHWC | 99,262 | 0.72x | **83,642** | 0.66x | 30 |
| Best variant per layer, NCHW only | | 116,742 | 0.85x | 107,040 | 0.84x | |
| Best variant per layer, NHWC allowed | | 93,225 | 0.68x | 83,523 | 0.66x | |
| Converting NCHW->NHWC before and back after every layer | | 17,937 | | | | |

#### X100, plain conv (13 DroNet + YOLOv8-nano layers)

| Algorithm | Layout | All layers (ticks) | vs RVV, what runs today | Without the first layer | vs RVV, what runs today | Layers won |
|---|---|---:|---:|---:|---:|---:|
| Direct (RVV vector, no matrix engine) | NCHW | 71,164 | 1.00x | 56,928 | 1.00x | 0 |
| Direct (RVV, channel-blocked) | NCHW | 73,692 | 1.04x | 58,033 | 1.02x | 0 |
| Implicit GEMM, K1 team kernel | NCHW | 113,476 | 1.59x | 78,258 | 1.37x | 0 |
| Implicit GEMM | NCHW | 26,431 | 0.37x | 16,875 | 0.30x | 4 |
| Indirect GEMM | NCHW | 30,250 | 0.43x | 17,896 | 0.31x | 0 |
| Implicit GEMM | NHWC | 26,037 | 0.37x | 12,080 | 0.21x | 3 |
| Implicit GEMM (scalar packing) | NHWC | 26,069 | 0.37x | **11,908** | 0.21x | 5 |
| Indirect GEMM | NHWC | 41,742 | 0.59x | 13,126 | 0.23x | 1 |
| Best variant per layer, NCHW only | | 24,358 | 0.34x | 14,802 | 0.26x | |
| Best variant per layer, NHWC allowed | | 21,206 | 0.30x | 11,650 | 0.20x | |
| Converting NCHW->NHWC before and back after every layer | | 6,326 | | | | |

#### X100, fused conv+BN+SiLU (35 YOLOv8-nano layers, 160x160)

| Algorithm | Layout | All layers (ticks) | vs installed | Without the first layer | vs installed | Layers won |
|---|---|---:|---:|---:|---:|---:|
| Direct (RVV vector, no matrix engine) | NCHW | 1,121,788 | 4.12x | 1,090,366 | 4.27x | 0 |
| Implicit GEMM, K1 team kernel | NCHW | 796,000 | 2.92x | 766,296 | 3.00x | 0 |
| Implicit GEMM, installed today | NCHW | 272,362 | 1.00x | 255,142 | 1.00x | 0 |
| Implicit GEMM | NCHW | 231,215 | 0.85x | 214,490 | 0.84x | 1 |
| Indirect GEMM | NCHW | 227,854 | 0.84x | 208,792 | 0.82x | 0 |
| Implicit GEMM | NHWC | 165,994 | 0.61x | **143,172** | 0.56x | 15 |
| Implicit GEMM (scalar packing) | NHWC | 167,706 | 0.62x | 143,584 | 0.56x | 14 |
| Indirect GEMM | NHWC | 203,888 | 0.75x | 152,868 | 0.60x | 5 |
| Best variant per layer, NCHW only | | 205,386 | 0.75x | 188,661 | 0.74x | |
| Best variant per layer, NHWC allowed | | 157,622 | 0.58x | 140,898 | 0.55x | |
| Converting NCHW->NHWC before and back after every layer | | 29,342 | | | | |

#### X100, fused conv+BN+SiLU (35 YOLOv8-nano layers, 64x96)

| Algorithm | Layout | All layers (ticks) | vs installed | Without the first layer | vs installed | Layers won |
|---|---|---:|---:|---:|---:|---:|
| Direct (RVV vector, no matrix engine) | NCHW | 310,808 | 3.13x | 300,204 | 3.16x | 0 |
| Implicit GEMM, K1 team kernel | NCHW | 271,292 | 2.73x | 261,052 | 2.75x | 0 |
| Implicit GEMM, installed today | NCHW | 99,228 | 1.00x | 94,898 | 1.00x | 0 |
| Implicit GEMM | NCHW | 66,402 | 0.67x | 62,074 | 0.65x | 1 |
| Indirect GEMM | NCHW | 52,704 | 0.53x | 48,097 | 0.51x | 1 |
| Implicit GEMM | NHWC | 41,034 | 0.41x | 35,589 | 0.38x | 12 |
| Implicit GEMM (scalar packing) | NHWC | 40,916 | 0.41x | **35,000** | 0.37x | 6 |
| Indirect GEMM | NHWC | 48,666 | 0.49x | 36,002 | 0.38x | 15 |
| Best variant per layer, NCHW only | | 49,590 | 0.50x | 45,262 | 0.48x | |
| Best variant per layer, NHWC allowed | | 37,842 | 0.38x | 33,514 | 0.35x | |
| Converting NCHW->NHWC before and back after every layer | | 7,476 | | | | |

### Is there one overall winner?

**No.** No single variant is fastest on every layer, but the winner follows the layer's shape closely enough to write down as a rule. A kernel can apply that rule itself, so nothing has to be chosen by hand per model.

1. **First layer (3 input channels): implicit GEMM, NCHW**, on both cores.
2. **Every other layer: NHWC wins on both cores.** It is 1.6-3× faster on 3×3 layers. On 1×1 layers the two layouts are within ±5% of each other. In NHWC, 16 (A100) or 8 (X100) consecutive values of a patch are consecutive channels of one pixel, so building a tile is one 16- or 8-byte copy per pixel. In NCHW they are scattered across channel planes.
3. **Implicit vs indirect depends on the output size**, on both cores:
   - Implicit wins large outputs (about 20×20 and up).
   - Indirect wins small ones (10×10 and below), because it never copies the input and its tiles don't care where output rows end.
   - The A100 switches to indirect at about 10×10. The X100 stays with implicit NHWC down to about 5×5.
4. **If the data stays NCHW**, which is what can be installed without other changes, the same split holds:
   - **A100:** indirect when the output is narrower than 8, or stride 2 and at most 12 wide; implicit otherwise. 1×1 stride-1 layers stay implicit unless the output is tiny.
   - **X100:** indirect when the output is at most 12 wide; implicit otherwise.
5. **Direct (RVV) never wins here.** It is 3-4.5× slower than the best matrix variant on both cores. The X100's only matrix-engine plain conv was the K1 team kernel, which is slower than RVV, so RVV is what runs today. The new X100 matrix conv is 2.7× faster than RVV alone, and 2.9× with the per-layer choice.

### What the conversion columns mean

NCHW and NHWC hold the same numbers in two different orders. If one layer writes NCHW and the next wants NHWC, the whole tensor has to be reordered first; that reorder is the conversion. Converting before and after every single layer costs about as much as a small 1×1 layer, which eats most of NHWC's gain on 1×1 layers.

The way to use NHWC is to keep a whole run of layers in NHWC: convert once at the start of the run (the first layer can even write NHWC directly) and once at the end.

### What changes next

- **Installing now:** the NCHW winners.
  - In `ime2_a100`, the conv and fused conv will pick implicit or indirect per layer shape using the rule above.
  - In `ime_x100`, the same for the fused conv, plus a new matrix-engine plain conv.
  - The networks are then re-profiled so XPU-RT plans with the new times.
- **NHWC needs ModelBlaster to carry a layout per tensor.** Today every kernel between layers (maxpool, add, concat, upsample, …) reads and writes NCHW. Handing a maxpool NHWC data would give wrong results. Supporting NHWC means:
  - choosing a layout per tensor
  - NHWC versions of those ops, or conversions where layouts change
  - conv weights emitted in NHWC tile order

### Caveats

- **These are single-layer timings** (the same layer called repeatedly, so its data stays in cache). The whole-model effect comes from the re-profile after the install.
- **On the X100, the matrix variants got pre-arranged weights; the installed X100 kernel rearranges them on every call.** ModelBlaster pre-arranges weights only for the A100 today. Part of the X100 gain may depend on adding that for the X100, which will be measured with the install.
- **The bench and variant sources** (`t014_bench.c`, `variants/t014/`) live in the author's K3 workspace and go into ModelBlaster with the install.
