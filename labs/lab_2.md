# Lab 2: AXI4 Burst Reads for a SIMD CFU

## Introduction

In Lab 1, the NPU fetched one 32-bit word per custom instruction. In this lab, one software command must start an AXI4 INCR read burst and retain multiple beats in accelerator-local storage. You will reuse those burst-loaded operands in a self-designed compute datapath and accelerate the int8 convolution used by `ds_cnn_stream_fe.tflite` in Lab 1.

The compute instruction encoding and internal microarchitecture are your choice while the external visible behavior is fixed.

## Learning goals

- implement an AXI4 read burst using the VALID/READY protocol
- interpret `ARLEN` and `RLAST` correctly and tolerate bus stalls
- buffer and reuse data instead of issuing repeated single-word reads
- define a stable hardware/software interface for custom instructions
- compare latency and FPGA resource cost under a reproducible flow

## Grading

| Component | Weight |
| --- | ---: |
| AXI4 burst engine and dot-product service | 20% |
| TFLM convolution and model correctness | 20% |
| Latency and accelerator-resource efficiency | 40% |
| Demo and questions | 20% |

## Prerequisites and setup

- Vivado 2024.1
- RISC-V toolchain
- Python 3 with `pyserial`
- Verilator and a host C++11 compiler
- Model file `ds_cnn_stream_fe.tflite` and its profile from Lab1

From the repository root, run this preflight before editing RTL:

```sh
vivado -version
verilator --version
c++ --version
riscv64-unknown-elf-g++ --version
python3 -c "import serial"
make -C Platform/sw validate
make -C Platform/sw check-env \
  MODEL_FILE=ds_cnn_stream_fe.tflite MODEL_PROFILE=ds_cnn_stream_fe
```

Use the course setup instructions if one of these commands fails.

## Files and submission boundary

You may modify:

- `Platform/hw/srcs/NPU.v`
- `Platform/sw/project/lab2_api.h`
- `Platform/sw/project/lab2_api.cc`
- `Platform/sw/tflm_patches/tensorflow/lite/kernels/internal/reference/integer_ops/conv.h`

You may add helper Verilog modules under `Platform/hw/srcs/` and helper C/C++ files under `Platform/sw/project/`. Do not replace or rename the `NPU` module. Preserve its complete port list, active-low `rst_n` behavior, and CPU/AXI signal directions so the existing block design still elaborates.

`NPU.v` is starter code, not a golden implementation. Write your own accelerator after all the environmental check passed.

> [!IMPORTANT]
> Your submitted project must build and run without source edits. Keep the required API signatures below even if you change every underlying `funct3`/`funct7` value.

## Required software contract

The public and grading software calls the following function declared in `Platform/sw/project/lab2_api.h`:

```cpp
int32_t hw_simd_mac(int8_t* ptr1, int8_t* ptr2, int8_t input_offset, uint32_t N);
```

Keep this signature and its C linkage unchanged, and implement it in `Platform/sw/project/lab2_api.cc`. The buffers are read-only from the accelerator's point of view.

For every valid call, return the exact signed 32-bit result:

```python
result = sum((ptr1[i] + input_offset)*ptr2[i] for i in range(N))
```

You can check function `ConvPerChannel` in file `tflite-micro/tensorflow/lite/kernels/internal/reference/integer_ops/conv.h` for more detail.

The required input domain is:

| Property | Required domain |
| --- | --- |
| `ptr1[i]`, `ptr2[i]` | Signed two's-complement int8 values |
| `input_offset` | Signed int8 value from `-128` through `127` |
| `N` | Every integer from `1` through `256` |
| Pointer alignment | Each pointer may independently have byte offset `0`, `1`, `2`, or `3` from a 4-byte boundary |
| Buffer access | Each pointer supplies at least `N` readable bytes; the function must not modify them |
| Return value | Exact signed 32-bit result for the formula above |

Accumulate and return the result as signed 32-bit data.

The CUSTOM-0 encoding behind `hw_simd_mac` is your design. You may choose the `funct3` and `funct7` values, command sequence, local-buffer organization, state machine, and SIMD width. Use the helpers in `Platform/sw/app/cfu.h`.

The multiplication and main accumulation must be performed by custom hardware using operand data read through AXI4 bursts. A CPU loop that calculates the dot product, including one hidden inside `hw_simd_mac`, does not satisfy the assignment. Software may copy or pad operands into legal staging buffers, clean cache lines, issue custom instructions, and combine accelerator partial results.

## Part 1: AXI4 read burst engine and dot service (20%)

Implement the read-burst engine and SIMD multiply-accumulate datapath in `Platform/hw/srcs/NPU.v`, then implement `hw_simd_mac` in `Platform/sw/project/lab2_api.cc` using your custom instructions. The NPU must fetch both operand vectors through its AXI master and use the fetched bytes in the returned dot product. AXI writes are not required for Lab 2.

### Required transfer domain

`N` in the software ABI is an element count measured in bytes, not an AXI word count. Use `B` below for the number of 32-bit AXI beats in one load. Every burst issued for Lab 2 must satisfy:

| Property | Requirement |
| --- | --- |
| Direction | Read only |
| AXI data width | 32 bits |
| Burst type | INCR (`ARBURST = 2'b01`) |
| Beat size | 4 bytes (`ARSIZE = 3'b010`) |
| Length | 2 through 64 beats |
| AXI start address | 4-byte aligned address in `0x6000_0000..0x67ff_ffff` |
| Local capacity | Enough local storage or streaming state for 256 bytes from each operand |
| Outstanding requests | One is sufficient |
| 4-KiB boundary | An individual burst must not cross it |

AXI encodes a burst of `B` beats as `ARLEN = B - 1`. Each AXI burst issues one address handshake with:

```text
ARADDR  = legal aligned source or staging address
ARLEN   = B - 1
ARSIZE  = 3'b010
ARBURST = 2'b01
```

A legal request satisfies:

```text
address[1:0] == 0
address[11:0] + 4 * B <= 4096
0x6000_0000 <= address
address + 4 * B <= 0x6800_0000
2 <= B <= 64
```

Your hardware design must handle independently unaligned pointers, arbitrary `N`, the final partial word, the `N=1..4` cases that still require a real multi-beat burst, and the misaligned `N=256` case without issuing an illegal 65-beat request. Replacing a multi-beat burst with repeated single-beat requests does not satisfy this lab. Write bursts, unaligned AXI starts, multiple outstanding requests, and hardware that splits a crossing request at a 4-KiB boundary may be used to avoid those cases.

### AXI protocol requirements

Your burst engine must satisfy all of the following:

- Assert `ARVALID` with a legal address and the required control values, and keep them stable until `ARVALID && ARREADY`.
- Accept a read beat only when `RVALID && RREADY`.
- Accept exactly `B` beats in address order and interpret each 32-bit beat in little-endian byte order.
- Do not assume that `ARREADY` or `RVALID` remains high continuously.
- Complete a successful load only after accepting beat `B` with `RLAST`; an early `RLAST`, a missing `RLAST` on beat `B`, an incorrect beat count, or a non-OKAY `RRESP` is an error and must assert `NPU_exception`.
- An early `RLAST` may finish the command immediately with `NPU_done` and `NPU_exception` asserted. If beat `B` arrives without `RLAST`, stop writing local operand storage, accept and discard later beats until `RLAST`, then finish with `NPU_done` and `NPU_exception` asserted.
- Keep `NPU_exception` low for a valid command and clear the previous command's status before accepting a new command.
- Pulse `NPU_done` exactly once for each completed custom command and do not leave it asserted while idle.
- Return to a clean idle state after reset and accept back-to-back commands.
- While reset is asserted, drive `M_AXI_ARVALID`, `M_AXI_AWVALID`, `M_AXI_WVALID`, `NPU_done`, and `NPU_exception` low.
- Do not write beyond the storage reserved for the current operand or corrupt operand data that is not the destination of the current load.

### Software verification

After changing RTL, program the FPGA and upload the model-free firmware from the repository root:

```sh
make -C Platform prog
make -C Platform run
```

Press `CPU RESET` after the upload, press `2` to enter the Lab 2 submenu, and run both entries:

```text
Main menu -> Lab 2: AXI burst accelerator -> Lab2 set tests (key t)
Main menu -> Lab 2: AXI burst accelerator -> Lab2 random tests (key r)
```

The test code writes the input arrays and then calls `hw_simd_mac`; it does not clean them first, so cache cleaning is part of the service implementation.

## Part 2: TFLM convolution integration (20%)

Modify the int8 `ConvPerChannel` implementation in:

```text
Platform/sw/tflm_patches/tensorflow/lite/kernels/internal/reference/integer_ops/conv.h
```

Include `lab2_api.h` and replace existing software implementations that can be accelerated with calls to `hw_simd_mac`.

CPU code may still perform loop control, address calculation, padding checks, bias addition, requantization, output offset, and clamping, but the input/filter channel products must contribute through `hw_simd_mac`.

Your accelerated kernel must preserve the reference behavior for:

- signed int8 input and filter values
- grouped-convolution input-channel offsets
- `input_offset` within the published int8 service domain and arbitrary input-channel counts
- stride, dilation, and omitted out-of-image padding positions
- bias addition
- per-channel multiplier and shift
- output offset and activation clamping
- output indexing and all tensor shapes

### Software verification

First upload the model-free firmware and run the existing golden convolution cases:

```sh
make -C Platform run
```

After pressing `CPU RESET`, select:

```text
Main menu -> Lab 1: single-transaction accelerator -> Basic convolution tests (key 4)
```
All three cases must report `[PASS]`.

### Run the `ds_cnn_stream_fe` model

After the test passed, recompile the software with following command:
```bash
make run -C Platform/sw MODEL_FILE=ds_cnn_stream_fe.tflite MODEL_PROFILE=ds_cnn_stream_fe 
```
It usually takes 20 min to inference. You can print out model output by yourself to check the correctness of model.

## Latency and accelerator-resource efficiency (40%)

Only submissions that complete every preceding requirement are eligible for ranking. An eligible submission must:
- receives `20/20` in Part 1
- receives `20/20` in Part 2
- compliance with the mandatory accelerator-command, AXI-trace, and convolution-coverage requirements in Part 2
- successful post-route implementation for `xc7a100tcsg324-1` with exactly one recursive `NPU` instance identified in the utilization report
- non-negative overall setup `WNS (ns)` in the final Design Timing Summary

### Available hardware and maximum usage

Resource counts are taken from the recursive NPU-only final post-route utilization report; the fixed CPU, caches, DDR controller, and interconnect are excluded. Each RAMB18 counts as one-half of a BRAM36 equivalent.

| Resource | Available on FPGA | Maximum NPU usage |
| --- | ---: | ---: |
| Slice LUT | 63,400 | 63,400 |
| Flip-flop | 126,800 | 126,800 |
| DSP48E1 | 240 | 10 |
| BRAM36 equivalent | 135 | 135 |

### Latency efficiency

Let $\mathcal{T}$ be the official test set and let $C_{i,t}$ be the `interpreter.Invoke()` cycle count for student $i$ on test $t$. Every test has equal weight, and the latency used for ranking is the average:

$$
\overline{C}_i=\frac{1}{|\mathcal{T}|}\sum_{t\in\mathcal{T}}C_{i,t}.
$$

Lower $\overline{C}_i$ is better. Cached final outputs are not allowed.

### Ranking

Eligible submissions are ordered from the lowest to the highest average cycle count and divided into five groups as evenly as possible.

| Level | Nominal position among eligible submissions | Efficiency points |
| ---: | --- | ---: |
| 1 | Top 20% | 40 |
| 2 | 20%-40% | 30 |
| 3 | 40%-60% | 20 |
| 4 | 60%-80% | 10 |
| 5 | Bottom 20% | 0 |

## Demo and questions (20%)

Prepare a short explanation of your implementation, including how you use aligned staging when a source address is not directly legal for the burst interface. Be ready to reproduce your results and answer questions about the following areas:

- AXI4 handshakes, burst state machine, stalls, `RLAST`, and error handling
- TFLM convolution mapping, aligned staging, and cache coherence

## Submission

Submit the source repository (or source-only archive requested on the course page) and a PDF report named `[id]-aaml-lab2.pdf` of at most two pages. Do not include `Platform/build/`, a generated Vivado project, or other reproducible build output. The report must contain your instruction mapping, accelerator/FSM diagram, convolution mapping, cache-coherence strategy, public-test results, inference cycles, post-route resources, and WNS.

> [!IMPORTANT]
> If the project cannot be compiled or run using the documented commands, Parts 1 and 2 and the efficiency section receive zero; only the 20-point demo can be assessed. Follow the course collaboration policy; plagiarism is not allowed.

