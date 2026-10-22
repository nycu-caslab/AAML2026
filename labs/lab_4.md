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

`lab4.zip` provides the model, tests, and TFLM updates. The current project also includes accelerators, wrappers, simulation tools, and upload-flow fixes. The ZIP is not a backup of the complete current implementation.

Omit the ZIP's outermost `lab4/` directory when mapping files into the project. **Merge folders, overwrite files with the same path, and add new files.** Keep existing files that are not included in the package; the package contains updates, not a complete replacement for the platform or TFLM tree.

| Source in the ZIP | Destination under `AAML_CPU_Backup/` |
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
rm -rf build/sw
```

This removes generated software build files, including the cached TFLM copy. It completes the Lab4 environment update. You do not need to create `build/sw/` yourself: `make run` recreates it and compiles the updated sources. If the directory does not exist, the command has no effect.

## 2. Profiling the MobileViT
---

### 2.1 Establishing a Comparable Baseline

Using the same model, input, and compilation settings, first record results with both Logistic and Transpose running on the CPU. Then enable acceleration separately.

| Experiment mode | Logistic | Transpose |
| --- | --- | --- |
| SW baseline | CPU | CPU |
| Logistic acceleration only | NPU | CPU |
| Transpose acceleration only | CPU | NPU |
| Both accelerated | NPU | NPU |

Here, SW refers only to these two operators. The existing Conv patch still uses Lab2's `hw_simd_mac`. Keep the Conv mode fixed and record it explicitly when comparing results. To claim that the entire model runs purely on the CPU, you must also switch and validate Conv and any other existing accelerated paths.

The current profile fills every byte of the input buffer with 0. This is **raw int8 zero**; the represented numerical value still depends on the input scale and zero point.

### 2.2 Counter Definitions and Measurement Scope

| Field | Meaning |
| --- | --- |
| Counter 0 | Accumulated cycles for the complete int8 Logistic function |
| Counter 1 | Accumulated cycles for the complete Transpose function |
| Starts | Accumulated function call count |
| Average | Total / Starts for that counter |
| Inference `Cycles` | Cycles measured by the runner for the complete `Invoke()` call |

The current template reads the 64-bit cycle counter with `perf_get_mcycle64()` and accumulates results through `perf_result`:

```cpp
uint64_t start = perf_get_mcycle64();

// Complete operator: CPU computation or hardware calls, including
// required waits and cache maintenance.

perf_result.add_start(counter_id);
perf_result.add_result(counter_id, perf_get_mcycle64() - start);
```

Use `counter_id` 0 for Logistic and 1 for Transpose. Clear statistics when preparing each model input and print them after inference. Keep UART printing outside the measurement interval.

Transpose measurements must include cache clean/invalidate operations, parameter setup, transfers, and completion waits. Measuring only the NPU core execution time understates the actual cost. Counter 2 does not currently measure Conv automatically; add timing or a profiler if you need statistics for other operators.

Record Total when comparing results, rather than relying only on Average. Calculate operator and model speedups separately:

```text
Operator speedup = operator Total cycles in SW mode / operator Total cycles in HW mode
Model speedup = baseline full Invoke cycles / accelerated full Invoke cycles
Time (seconds) = cycles / CPU clock frequency (Hz)
```

Keep the model, input, clock frequency, Conv mode, tool versions, and compiler options fixed. Tables or charts can help explain bottlenecks, but measurements of only Logistic and Transpose do not constitute a complete breakdown of all operators.

## 3. Accelerating the Logistic Function
---

### 3.1 Tracing the Original Computation

The entry point is the topmost int8 `Logistic(...)` overload in:

```text
Platform/sw/tflm_patches/tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h
```

Its main computation flow is:

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

`logistic.h` obtains the gemmlowp definitions indirectly through `common.h`. The implementation is located in:

```text
tflite-micro/third_party/gemmlowp/fixedpoint/fixedpoint.h
```

Follow `logistic` and `logistic_on_positive_values` to find `exp_on_negative_values` and `one_over_one_plus_x_for_x_in_0_1`, which implement the exponential and reciprocal portions, respectively.

### 3.2 Fixed-Point Formats

In the gemmlowp formats described here, the integer-bit count excludes the sign bit:

| Type | Sign bits | Integer bits | Fractional bits | Real value represented by raw |
| --- | ---: | ---: | ---: | --- |
| `FixedPoint<int32_t, 4>` | 1 | 4 | 27 | `raw / 2^27` |
| `FixedPoint<int32_t, 0>` | 1 | 0 | 31 | `raw / 2^31` |

`FromRaw()` interprets an integer bit pattern in the specified fixed-point format; `.raw()` returns the stored integer. Neither automatically quantizes an arbitrary floating-point value.

Handle positive and negative inputs, zero, saturation, multiplication widths, truncation, and rounding. Mathematically, `sigmoid(0)=0.5`, whose Q0.31 raw representation is `1 << 30`. Do not reuse the pseudocode in the reference notes that returns 1 at zero.

### 3.3 Connecting Software and Hardware

A useful starting point is to retain the original input quantization and output rounding, replacing only the sigmoid computation in between:

```cpp
// Original CPU computation, preserved for baseline comparisons:
// gemmlowp::logistic(FixedPoint4::FromRaw(input_in_q4)).raw();

```

A function name alone does not move work to the NPU. The actual call path in the current interface is:

```text
logistic.h
  → sw/app/lab4_accel.h
  → cfu_raw_op(funct3, funct7, rs1, rs2)
  → cfu.h: .insn r 0x0B, ...
  → CPU decodes the CUSTOM-0 instruction
  → NPU_start, followed by a wait for NPU_done
  → NPU_out is written back to a CPU register
```

CFU stands for Custom Function Unit; this platform reuses the corresponding software helpers. `.insn` is assembler syntax for encoding an instruction, and `0x0B` is the CUSTOM-0 opcode. The actual hardware entry point is `Platform/hw/srcs/NPU.v`. Follow its `NPU_start` / `NPU_done` protocol for multicycle operations.

Preserve the instruction functions already used by Lab1 and Lab2. The new RTL and software calls must agree on `funct3`, `funct7`, input formats, and return values.

## 4. Accelerating the Transpose Function
---

### 4.1 From Matrix Transpose to Multidimensional Indexing

The entry point is located in:

```text
Platform/sw/tflm_patches/tensorflow/lite/kernels/internal/reference/transpose.h
```

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

The input strides in this example are `[3, 1]`. Advancing along the innermost output dimension moves the input pointer by three elements, so the innermost loop can use:

```cpp
output_data[i] = *input_data;
input_data += loop_stride;
```

This is part of the transpose because the outer recursion, `perm`, and `loop_stride` have already determined which input positions to read.

For a multidimensional tensor with output coordinates `o[d]`:

```text
output_shape[d] = input_shape[perm[d]]
input_offset = Σ o[d] * input_stride[perm[d]]
input_byte_address = input_base + input_offset * sizeof(T)
```
### 4.2 Implementation Constraints

In hardware-accelerated mode, Transpose must move tensor elements through the NPU's AXI master interface. The data-copy operation corresponding to `*output_data = *input_data` must be implemented in the NPU.

- Input tensor elements must not be loaded into CPU general-purpose registers as part of the Transpose operation.
- Output tensor elements must not be written from CPU general-purpose registers as part of the Transpose operation.
- The CPU must not use ordinary load/store instructions through its load/store unit (LSU) to perform the tensor rearrangement. Reading an element on the CPU and passing its value to the NPU is also prohibited.
- The CPU may compute shapes, strides, and addresses; pass configuration parameters and buffer addresses to the NPU; issue commands; and wait for completion.
- Necessary cache maintenance operations, including clean and invalidate, are allowed to maintain consistency between CPU and NPU memory accesses.

## 5. Check Results
---
### 5.1 Basic Test Verification

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

### 5.2 Running TFLM Model Inference
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
> xxx represents values measured during execution. Counter 0 measures Logistic, and Counter 1 measures Transpose. Total is the accumulated cycle count, Starts is the number of calls, and Average is Total / Starts. The final Cycles and Time report the complete inference measurement.

## 6. Questions in the Demo
---

You will be asked several questions about the concepts covered in this lab and your implementation. This section accounts for X% of the total lab score.


## 7. Submission
--- 

Please organize your submission files into a zip archive structured as follows:
```
[YourID]-lab4.zip
└── [YourID]-lab4/
    ├── hw/
    │   ├── NPU.v
    │   └── {any other required modules}
    ├── sw/
    │   ├── logistic.h
    │   ├── transpose.h
    │   └── {any other required headers or source files}
    └── report/
        └── lab4_report.pdf
```

```{important}
TAs should be able to run your project without any modification. If TAs cannot compile or run your code, **you won't receive any points, even if you passed the DEMO**. Also, **PLAGIARISM is not allowed**.
```