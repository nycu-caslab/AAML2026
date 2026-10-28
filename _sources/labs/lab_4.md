# Lab 4 : Elementwise Unit

## Introduction
---
Modern models frequently utilize specialized activation functions that involve complex mathematical computations, such as exponentials, square roots, and reciprocals. Unlike simpler functions like ReLU, these sophisticated operations can become bottlenecks during model inference. In this lab, we will design an element-wise unit specifically for enhancing the inference speed of the **Logistic** functions in the **MobileViT** model.
Logistic applies the sigmoid function to each element:
$$ \sigma(x)=\frac{1}{1+e^{-x}} $$
Additionally, MobileViT combines convolution, nonlinear operations, and multidimensional tensor rearrangement. This lab focuses on two different costs: fixed-point numerical computation in Logistic, and index calculation and memory transfers in **Transpose**.


## Goal of this lab
---
1. Integrate MobileViT xxs and its tests into the platform and establish a correct software baseline.
2. Trace the quantization and fixed-point arithmetic in int8 Logistic and design a hardware accelerator.
3. Understand multidimensional Transpose strides and address generation, and rearrange data through the NPU AXI master.
4. Handle multicycle custom instructions and memory consistency between the CPU cache and NPU.

## 1. Porting the Model and Tests
---
### 1.1 Project Structure & Extracting
```text
YOUR_PROJECT/
    ├── tflite-micro/
    └── Platform/
        ├── Makefile
        ├── hw/
        │   └── srcs/NPU.v
        └── sw/
            ├── app/
            ├── models/
            ├── project/
            ├── tflm_patches/
            └── project.mk
```
**Merge folders, overwrite files with the same path, and add new files.** Keep existing files that are not included in the package; the package contains updates, not a complete replacement for the platform or TFLM tree.

| Source in the lab4 | Destination under `TinyRISC-V-SoC-Platform/` |
| --- | --- |
| `lab4/Platform/sw/app/` | `Platform/sw/app/` |
| `lab4/Platform/sw/models/` | `Platform/sw/models/` |
| `lab4/Platform/sw/project/` | `Platform/sw/project/` |
| `lab4/Platform/sw/tflm_patches/` | `Platform/sw/tflm_patches/` |
| `lab4/Platform/sw/project.mk` | Overwrite `Platform/sw/project.mk` |
| `lab4/tflite-micro/` | **`tflite-micro/` at the project root** |

### 1.2 Model Configuration and TFLM Updates

The package supplies `Platform/sw/project.mk`, which is read automatically by the default build commands. Its Lab4 settings are:

```makefile
MODEL_DIR := models
MODEL_FILE := mobileViT_xxs.tflite
MODEL_PROFILE := mobileViT_xxs
TENSOR_ARENA_SIZE := 8*1024*1024
TARGET_PREFIX := riscv64-unknown-elf
USE_SOFTWARE_CFU := 0
```

The model and profile are located at:

```text
Platform/sw/models/mobileViT_xxs.tflite
Platform/sw/models/mobileViT_xxs_profile.cc
```

This model requires support for 6D tensors. Verify that the sources actually being compiled contain the following updates:

- `tflite-micro/tensorflow/lite/kernels/internal/runtime_shape.h`: `kMaxSmallSize` is 6.
- Both the micro Transpose kernel and reference Transpose support up to six dimensions.
- The BatchMatMul kernel and registration updates supplied in the ZIP are installed.
- `Platform/sw/project/tflm_ops.cc` registers Logistic, Transpose, BatchMatMul, and the other operators required by the model. The current resolver capacity is 27.

The insufficient shape capacity in the old version can corrupt memory. Increasing the tensor arena size does not replace this fix. Do not force every tensor's actual dimension count to 6 either.

### 1.3 Clearing the Previous Software Build

`Platform/sw/Makefile` copies the root TFLM sources into `Platform/build/sw/src/`, then applies `tflm_patches/`. It does not automatically synchronize subsequent changes to the root TFLM tree. An existing build may therefore still contain the old TFLM sources after the package is installed.

After installing the package, run this once from `Platform/`, before `make prog`:

```bash
cd /Platform
make sw_clean
```

This removes generated software build files, including the cached TFLM copy. It completes the Lab4 environment update. You do not need to create `build/sw/` yourself: `make run` recreates it and compiles the updated sources. If the directory does not exist, the command has no effect.

## 2. Establishing a Software Baseline
---

Using the same model, input, and compilation settings, first record results with both Logistic and Transpose running on the CPU. Then enable acceleration separately.

| Experiment mode | Logistic | Transpose |
| --- | --- | --- |
| SW baseline | CPU | CPU |
| Logistic acceleration only | NPU | CPU |
| Transpose acceleration only | CPU | NPU |
| Both accelerated | NPU | NPU |

Here, SW refers only to these two operators. The existing Conv patch still uses Lab2's `hw_simd_mac`. Keep the Conv mode fixed and record it explicitly when comparing results. To claim that the entire model runs purely on the CPU, you must also switch and validate Conv and any other existing accelerated paths.

The current profile fills every byte of the input buffer with 0. This is **raw int8 zero**; the represented numerical value still depends on the input scale and zero point.

## 3. Accelerating the Logistic Function
---

Before modifying the operator, review the measurement requirements in Section 5.2. All per-call work must remain within the provided timing boundaries.

### 3.1 Tracing the Original Computation

The entry point is the topmost int8 `Logistic(...)` overload in:

```text
Platform/sw/tflm_patches/tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h
```

This lab targets the int8 overload; the int16 overload later in the file is a separate implementation. The original CPU computation follows this flow:

```text
int8 input
  → subtract input_zero_point
  → check input_range_radius for saturation
  → MultiplyByQuantizedMultiplier
  → FixedPoint<int32_t, 4>
  → gemmlowp::logistic
  → FixedPoint<int32_t, 0>
  → rounding, output zero point, and clamp
  → int8 output
```

Inputs outside the permitted radius take the saturation branch directly. For other inputs, accelerate the sigmoid computation as described in Section 3.3.

`logistic.h` obtains the gemmlowp definitions indirectly through `common.h`. The implementation is located in:

```text
tflite-micro/third_party/gemmlowp/fixedpoint/fixedpoint.h
```

Follow `logistic` and `logistic_on_positive_values` to find `exp_on_negative_values` and `one_over_one_plus_x_for_x_in_0_1`, which implement the exponential and reciprocal portions, respectively.

The mathematical relationships behind these functions are:

```text
sigmoid(x) = 1 / (1 + exp(-x))
sigmoid(-x) = 1 - sigmoid(x)
sigmoid(0) = 0.5
```

For a positive magnitude, the original implementation evaluates the exponential of its negative value, then the reciprocal expression. It uses symmetry to handle negative inputs and handles zero explicitly. These equations explain the computation; the fixed-point implementation also determines approximation, rounding, and saturation behavior.

### 3.2 Fixed-Point Formats

In the gemmlowp formats described here, the integer-bit count excludes the sign bit:

| Type | Sign bits | Integer bits | Fractional bits | Real value represented by raw |
| --- | ---: | ---: | ---: | --- |
| `FixedPoint<int32_t, 4>` | 1 | 4 | 27 | `raw / 2^27` |
| `FixedPoint<int32_t, 0>` | 1 | 0 | 31 | `raw / 2^31` |

`FromRaw()` interprets an integer bit pattern in the specified fixed-point format; `.raw()` returns the stored integer. Neither automatically quantizes an arbitrary floating-point value.

The Q0.31 sigmoid result is an intermediate value, not the final int8 output. The original code applies `RoundingDivideByPOT(..., 23)`, adds the output zero point of `-128`, and clamps the result to `[-128, 127]`. For example, the Q0.31 representation of `0.5` becomes `128` after rounding and scaling, then `0` after the zero-point adjustment.

Handle positive and negative inputs, zero, saturation, multiplication widths, truncation, and rounding. Mathematically, `sigmoid(0)=0.5`, whose Q0.31 raw representation is `1 << 30`. Do not reuse the pseudocode in the reference notes that returns 1 at zero.

### 3.3 Connecting Software and Hardware

Implement your changes inside the `DESIGN REGION BEGIN` / `DESIGN REGION END` markers in the int8 Logistic template, following Section 5.2. Preserve Counter 0 and its timing boundaries around the complete operator, including the input loop, quantization, saturation handling, hardware calls, and output processing.

A useful starting point is to retain the original input quantization and output rounding, replacing the following CPU sigmoid computation with a call to your hardware implementation:

```cpp
const int32_t output_in_q0 =
    gemmlowp::logistic(FixedPoint4::FromRaw(input_in_q4)).raw();
```

This computation is inside the non-saturated branch of the input loop. Record the original CPU implementation's results for the baseline comparisons described in Section 2.

After changing the header's implementation, run `make sw_clean` from `Platform/` before rebuilding with `make run`. If the RTL changes, also rebuild and program the hardware with `make prog` before running the software.

If you replace only this expression, the replacement must accept the signed Q4.27 raw input and produce a compatible Q0.31 raw result. Preserve the subsequent rounding, output zero-point adjustment, and clamping. You may organize the hardware internally as you choose while preserving the operator's required behavior.

Connect the software call to your hardware through the platform's custom-instruction interface. A typical call path is:

```text
logistic.h: accelerator call inside the design region
  → your software wrapper
  → cfu_raw_op(funct3, funct7, rs1, rs2)
  → cfu.h: .insn r 0x0B, ...
  → CPU decodes the CUSTOM-0 instruction
  → NPU_start, followed by a wait for NPU_done
  → NPU_out is written back to a CPU register
```

CFU stands for Custom Function Unit; this platform reuses the corresponding software helpers. `.insn` is assembler syntax for encoding an instruction, and `0x0B` is the CUSTOM-0 opcode. The actual hardware entry point is `Platform/hw/srcs/NPU.v`. Follow its `NPU_start` / `NPU_done` protocol for multicycle operations.

Define the instruction function fields, operand meanings, and result format consistently in your software and RTL. Preserve the instruction functions already used by Lab1 and Lab2. For a multicycle computation, assert completion only when the result is valid; the CPU must wait for completion before consuming it. Include this execution and waiting time in the operator measurement specified in Section 5.2.

## 4. Accelerating the Transpose Function
---

Before modifying the operator, review the measurement requirements in Section 5.2. All per-call work must remain within the provided timing boundaries.

### 4.1 From Matrix Transpose to Multidimensional Indexing

The entry point is `Transpose(...)` in:

```text
Platform/sw/tflm_patches/tensorflow/lite/kernels/internal/reference/transpose.h
```

Transpose changes the order of tensor dimensions and rearranges the corresponding elements. It preserves every element's value and the total element count; it does not perform arithmetic or requantization on tensor values.

The original implementation uses contiguous row-major storage, where the last dimension varies fastest. The following metadata describes the mapping:

| Metadata | Meaning |
| --- | --- |
| `input_shape[d]`, `output_shape[d]` | Number of elements along dimension `d` of the respective tensor |
| `perm[d]` | Input dimension corresponding to output dimension `d` |
| `input_stride[d]`, `output_stride[d]` | Offset in elements for advancing one position along dimension `d` of the respective tensor |

For a rank-`D` tensor, `perm` contains each dimension index from `0` to `D - 1` exactly once. The shape and stride relationships are:

```text
output_shape[d] = input_shape[perm[d]]

// Compute strides separately for the input and output shapes.
stride[D - 1] = 1
stride[d] = stride[d + 1] * shape[d + 1]   // d = D - 2, ..., 0
```

Strides and offsets here are measured in elements. Multiply an element offset by `sizeof(T)` when calculating a byte address.

For a two-dimensional matrix, `output[i][j] = input[j][i]` can be expressed as strided reads from contiguous memory:

```text
input shape = [2, 3], perm = [1, 0]

input:              output:
1 2 3               1 4
4 5 6               2 5
                    3 6

input buffer:  1, 2, 3, 4, 5, 6
output buffer: 1, 4, 2, 5, 3, 6
```

The input strides in this example are `[3, 1]`, and the output strides are `[2, 1]`. Since `perm[1] = 0`, advancing along the innermost output dimension advances the input position by `input_stride[0]`, or three elements.

The same mapping applies to higher-dimensional tensors. For example:

```text
input_shape   = [2, 3, 4]
input_stride  = [12, 4, 1]
perm          = [1, 0, 2]
output_shape  = [3, 2, 4]
output_stride = [8, 4, 1]

output[a, b, c] = input[b, a, c]
input_offset   = b * 12 + a * 4 + c
output_offset  = a * 8 + b * 4 + c
```

For a general tensor with output coordinates `o[d]`, sum over all dimensions:

```text
input_offset = Σ o[d] * input_stride[perm[d]]
output_offset = Σ o[d] * output_stride[d]
input_byte_address = input_base + input_offset * sizeof(T)
output_byte_address = output_base + output_offset * sizeof(T)
```

Trace the original CPU implementation through these functions:

```text
Transpose(...)
  → SetupTransposeStrides(...) for the input and output shapes
  → TransposeImpl(...) starting at depth 0
      → recursively traverse the output dimensions
      → copy elements at the innermost dimension
```

Each recursion level corresponds to one output dimension. At an outer level, the function advances the input pointer by `input_stride[perm[depth]]` and the output pointer by `output_stride[depth]` between recursive calls. At the innermost level, output positions are contiguous, so the original CPU loop is:

```cpp
const int loop_stride = input_stride[perm[depth]];
for (int i = 0; i < output_shape[depth]; ++i) {
  output_data[i] = *input_data;
  input_data += loop_stride;
}
```

This loop performs the transpose because the recursion has already selected the starting positions, and `perm` determines the input step for each output dimension. These details explain the original algorithm; your accelerator does not need to reproduce its recursive structure.

### 4.2 Implementation Constraints

Implement your changes within the Transpose template's `DESIGN REGION BEGIN` / `DESIGN REGION END` markers. Preserve Counter 1 and include all per-call preparation, data movement, completion waits, and required cache maintenance within the timing boundaries specified in Section 5.2.

In hardware-accelerated mode, Transpose must move tensor elements through the NPU's AXI master interface. The data-copy operation corresponding to `*output_data = *input_data` must be implemented in the NPU.

- Input tensor elements must not be loaded into CPU general-purpose registers as part of the Transpose operation.
- Output tensor elements must not be written from CPU general-purpose registers as part of the Transpose operation.
- The CPU must not use ordinary load/store instructions through its load/store unit (LSU) to perform the tensor rearrangement. Reading an element on the CPU and passing its value to the NPU is also prohibited.
- The CPU may compute shapes, strides, and addresses; pass configuration parameters and buffer addresses to the NPU; issue commands; and wait for completion.
- Necessary cache maintenance operations, including clean and invalidate, are allowed to maintain consistency between CPU and NPU memory accesses.

### 4.3 Optimization Hints

These hints are provided for reference only. You are not required to follow them or implement any of the listed techniques. You may develop your own design, provided it satisfies the implementation constraints in Section 4.2 and the timing requirements in Section 5.2.

The effectiveness of each optimization depends on tensor shapes, permutations, and platform behavior. Evaluate each change using the complete operator's Total cycles.

- **Reduce cache-maintenance overhead.** Consider the cache-line coverage of each buffer and avoid redundant operations on the same line when correctness permits. Preserve required clean/invalidate operations and unrelated data that shares a cache line.

- **Overlap CPU and NPU work.** If your interface supports overlap, consider preparing metadata or addresses for upcoming work while the NPU performs the current transfer. The CPU must not read or write tensor elements, and all work must finish within the measured operator interval.

- **Explore CPU/NPU work partitioning.** Compare computing address information on the CPU with generating it in the NPU. Consider the tradeoff between CPU computation, command traffic, and hardware complexity. Tensor data movement must remain in the NPU.

- **Reduce command overhead.** Consider whether one command can describe more work, reducing repeated configuration and synchronization. Account for any additional metadata preparation or hardware control cost.

- **Simplify address calculations.** Look for repeated arithmetic that can be reused within an invocation. Consider incremental address updates instead of repeatedly calculating full multidimensional offsets.

- **Exploit contiguous dimensions.** Some permutations preserve contiguous groups of elements. Consider whether these groups can be handled together and whether adjacent dimensions can be combined without changing the required mapping.

- **Explore traversal order and tiling.** Different traversal orders or tile sizes may improve read or write locality. Consider the tradeoff between contiguous input reads, contiguous output writes, and temporary storage inside the NPU.

- **Use AXI bandwidth efficiently.** Consider packing multiple tensor elements into each bus-width transfer and using bursts for contiguous addresses where supported. Handle alignment, partial transfers, and write strobes correctly.

- **Use internal buffering.** Consider FIFOs or double buffering inside the NPU to decouple reading, rearrangement, and writing. Additional buffering may help hide latency, but its control and resource costs may outweigh the benefit.

- **Specialize for general shape patterns.** Consider optimized paths for classes of permutations or shapes, while retaining correct handling of all required cases. Select paths using metadata; do not recognize public test inputs or return precomputed answers.

## 5. Performance Evaluation and Timing Requirements
---

### 5.1 Counter Definitions and Measurement Scope

| Field | Meaning |
| --- | --- |
| Counter 0 | Accumulated cycles for the complete int8 Logistic function |
| Counter 1 | Accumulated cycles for the complete Transpose function |
| Starts | Accumulated function call count |
| Total | Sum of measured cycles across all calls to that operator during one inference |
| Average | Total / Starts for that counter |
| Inference `Cycles` | Cycles measured by the runner for the complete `Invoke()` call |

### 5.2 Implementation and Timing Requirements

Implement each operator within the marked design region below. Use counter ID `0` for Logistic and `1` for Transpose. Helper functions may be defined elsewhere, but all per-call work must execute within the measured interval.

```cpp
// TIMING START - DO NOT MODIFY OR MOVE.
uint64_t start = perf_get_mcycle64();

// ================= DESIGN REGION BEGIN =================
// Implement the complete operator here.
// You may call helper functions and hardware accelerators.
//
// Include all per-call work:
// - Parameter and stride calculations and required input processing.
// - Required cache maintenance and data transfers.
// - CPU or NPU computation.
// - Waiting for hardware completion.
// - Final output processing and required cache maintenance.
//
// All output data must be ready before leaving this region.
// Do not move work outside this region to reduce measured cycles.
// Do not return early or bypass the timing and statistics below.
// ================== DESIGN REGION END ==================

// TIMING END AND STATISTICS - DO NOT MODIFY OR MOVE.
perf_result.add_start(counter_id);
perf_result.add_result(counter_id, perf_get_mcycle64() - start);
```

```{warning}
Preserve the provided timing boundaries, counter assignments, cycle-counter implementation, and statistics collection and reporting logic. These requirements apply to both the CPU baseline and the accelerated implementation.

Each operator invocation must be measured exactly once, from before its per-call preparation begins until computation and required memory/cache operations have completed and the output is ready for use. Moving work outside this interval, bypassing measurement, or altering recorded cycles to improve the reported speedup is prohibited.
```

### 5.3 Reporting Cycles and Speedup

Preserve the existing measurement sequence: clear statistics when preparing each model input, accumulate results across all operator calls during inference, and print statistics after inference. Keep UART printing outside the operator measurement intervals.

Use **Total**, not Average, to calculate operator speedup. Calculate operator and model speedups separately:

```text
Operator speedup = operator Total cycles in SW mode / operator Total cycles in HW mode
Model speedup = baseline full Invoke cycles / accelerated full Invoke cycles
Time (seconds) = cycles / CPU clock frequency (Hz)
```

Keep the model, input, clock frequency, Conv mode, tool versions, and compiler options fixed between baseline and accelerated runs. Report the operator Total cycles and full inference cycles for each experiment mode, together with the corresponding speedups.

Counter 2 does not currently measure Conv automatically. Any additional profiling must preserve the required measurements above. Measurements of only Logistic and Transpose do not constitute a complete breakdown of all operators.

### 5.4 FPGA Timing Constraint

```{important}
The complete FPGA design must achieve **WNS >= 0 ns** for setup timing in the final post-route Design Timing Summary generated by Vivado. This requirement applies to the integrated design containing both the Logistic and Transpose accelerators.

Use the provided clock and timing constraints. Do not reduce the required clock frequency or relax timing constraints to satisfy this requirement. A synthesis timing estimate or a timing report for an isolated accelerator module is not sufficient.

Passing functional tests on the FPGA does not waive this timing requirement.
```

## 6. Check Results
---

```{warning}
Your implementation must compute results from the provided inputs. Hardcoding expected answers in hardware or software, or detecting specific test inputs to return precomputed answers, is not allowed. This requirement applies to both the Logistic and Transpose tests and the final TFLM model outputs.

Passing the public tests and matching the example model output are necessary but do not guarantee correctness. We will also run hidden test cases with different inputs to verify your implementation. The ±2 tolerance for final model outputs also applies to hidden model inference tests.
```

### 6.1 Basic Test Verification

Losigtic function acceleration:
Run `l: run logistic test` in `l: Lab menu`

```
Lab : Self Test
===============
 l: run logistic test
 t: run transpose test
 x: eXit to previous menu
lab> l

Running run logistic test
[PASS] LogisticQuantizedInt8AroundZeroShouldMatchGolden
[PASS] LogisticQuantizedInt8NarrowRangeShouldMatchGolden
[PASS] LogisticQuantizedInt8BasicShouldMatchGolden
[PASS] LogisticQuantizedInt8WideRangeShouldMatchGolden
Logistic tests: 4/4 passed
---
```

Transpose function acceleration:
Run `t: run transpose test` in `l: Lab menu`

```
Lab : Self Test
===============
 l: run logistic test
 t: run transpose test
 x: eXit to previous menu
lab> t

Running run transpose test
[PASS] 2D_2x3_perm10
[PASS] 2D_4x5_perm10
[PASS] 2D_1x8_perm10
[PASS] 2D_identity_3x4_perm01
[PASS] 3D_2x2x2_perm210
[PASS] 3D_2x3x4_perm201
[PASS] 3D_2x3x4_perm021
[PASS] 3D_3x2x5_perm102
[PASS] 3D_identity_2x3x4_perm012
[PASS] 4D_NHWC_to_NCHW_1x3x4x2
[PASS] 4D_NCHW_to_NHWC_2x2x3x4
[PASS] 4D_reverse_2x3x2x3_perm3210
[PASS] 5D_2x2x2x2x3_perm43210
[PASS] 6D_2x2x2x2x2x2_perm543210
Transpose tests: 14/14 passed
---
```

### 6.2 Running TFLM Model Inference
In this part, We use `mobileViT_xxs` as our benchmark model. Then, compare runtime cycles to show performance improve on accelerated version logistic & tranpose function.
The model and profile are located at:

```text
Platform/sw/models/mobileViT_xxs.tflite
Platform/sw/models/mobileViT_xxs_profile.cc
```
Run `t: TFLM inferencr, verificatoin, and cycles` at the default menu.
```
TinyRISC-V SoC Platform
=======================
 l: Lab menu
 t: TFLM inference, verification, and cycles
 p: Performance counter tests
 i: System info
main> t

Running TFLM inference, verification, and cycles

=== TFLM Functional Verification ===
Model: mobileViT_xxs.tflite (1212672 bytes)
Tensor arena: 8388608 bytes
Profile: mobileVit_xxs
Input profile: zero input
Output profile: Print Output Value, tolerance=0
Samples: 1

```
```
--- Sample 1/1 ---
Running inference...
Inference complete.
Output (10 bytes):
00: -32
01: 114
02: -38
03: 52
04: 50
05: 127
06: -120
07: -128
08: -71
09: 80

==============================================
 Counter |       Total | Starts |   Average
---------+-------------+--------+-------------
    0    |         xxx |    xxx |         xxx
    1    |         xxx |    xxx |         xxx
    2    |           0 |      0 |         n/a
    3    |           0 |      0 |         n/a
    4    |           0 |      0 |         n/a
    5    |           0 |      0 |         n/a
    6    |           0 |      0 |         n/a
    7    |           0 |      0 |         n/a
    8    |           0 |      0 |         n/a
    9    |           0 |      0 |         n/a
==============================================
Cycles: xxx
Time (50000000 Hz): xxx ms
---
```
For the same model and zero input shown above, compare each final int8 output element against the corresponding reference value in the example. An absolute error of up to **2 quantization steps (±2)** is allowed:

`abs(actual[i] - reference[i]) <= 2`

All 10 output elements must satisfy this condition to pass. This tolerance applies to the raw quantized integer values, not percentages or dequantized floating-point values. For example, a reference value of `-32` accepts results from `-34` to `-30`, inclusive.

> xxx represents values measured during execution. Counter 0 measures Logistic, and Counter 1 measures Transpose. Total is the accumulated cycle count, Starts is the number of calls, and Average is Total / Starts. The final Cycles and Time report the complete inference measurement.

## 7. Grading Policy
---

| Category | Requirement | Weight |
| --- | --- | ---: |
| Basic | Logistic passes all correctness tests and runs faster than the CPU baseline. | 25% |
| Basic | Transpose passes all correctness tests. | 25% |
| Advanced | Transpose passes all correctness tests and runs faster than the CPU baseline. | 10% |
| Advanced | Transpose speedup ranking among correct implementations. | 10% |
| Demo | Explain your implementation and answer questions during the demo. | 30% |
| **Total** | | **100%** |

The Basic component accounts for 50%, the Advanced component for 20%, and the Demo for 30% of the lab score. The 25% Basic score for Logistic requires both correctness and a speedup greater than 1.

Speedup is calculated as the CPU baseline's accumulated operator cycles divided by the accelerated implementation's accumulated operator cycles. Use Counter 0 for Logistic and Counter 1 for Transpose, following the measurement requirements in Section 5. Compare runs using the same model, input, clock frequency, and compilation settings. Only implementations that pass all Transpose correctness tests are eligible for the Transpose speedup ranking.

| Performance ranking | Score |
| --- | ---: |
| Top 1–25% | 10 |
| Top 26–50% | 7 |
| Top 51–75% | 4 |
| Top 76–100% | 1 |

## 8. Questions in the Demo
---

You will be asked several questions about the concepts covered in this lab and your implementation. This section accounts for 30% of the total lab score.


## 9. Submission
--- 

Please organize your submission files into a zip archive structured as follows:
```
[YourID]-lab4.zip
└── [YourID]-lab4/
    ├── hw/
    │   ├── NPU.v
    │   └── {any other required modules}
    └── sw/
        ├── logistic.h
        ├── transpose.h
        └── {any other required headers or source files}
```

```{important}
TAs should be able to run your project without any modification. If TAs cannot compile or run your code, **you won't receive any points, even if you passed the DEMO**. Also, **PLAGIARISM is not allowed**.
```
