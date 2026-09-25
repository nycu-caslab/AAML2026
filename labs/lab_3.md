# Lab 3 : Systolic Array
---
<!-- ## Goal of this lab
---
- [Convolution Exercise #1 - 10%](#convolution-exercise-1---10)
- [Convolution Exercise #2 - 10%](#convolution-exercise-2---10)
- [Convolution Exercise #3 - 20%](#convolution-exercise-3---20)
- [Convolution Exercise #4 - 20%](#convolution-exercise-4---20)
- [GEMV Basic Exercise - 20%](#gemv-basic-exercise---20)
- [GEMV Performance Challenge - 20%](#gemv-performance-challenge---20)
 -->


## Introduction
---
The systolic array used by Google Tensor Processing Unit (TPU) accelerates the matrix computation by using the dataflow operation. The systolic array contains multiple processing elements (PEs), each of them is responsible for the multiply–and-accumulate (MAC) operation. It can performs multiple elements in a matrix simultaneously and achieves high computational throughput.

In this lab, we will use Verilog to implement PEs and a small systolic array composed of multiple PEs to accelerate convolution and general matrix-vector multiplication (GEMV). Different dataflows, such as Weight Stationary, Output Stationary, Input Stationary, and Row Stationary, can be considered when designing the PE array. 

(Hint: The weight stationary is more complicated than output stationary.)
```{note}
1. This lab does not require demonstration, submit your code to E3 only, details are in [Submission](#submission).


2. You will be asked to put your systolic array on FPGA in ***lab 5***, so please make sure that your systolic array is a synthesizable circuit.
```

## Learning goals
- Implement 2D convolution on a systolic array with processing element(s) (PEs), using a dataflow of your choice.
- Implement GEMV on a systolic array using a dataflow of your choice.
- Optimize the GEMV architecture and dataflow to reduce the total execution cycle count.

## Grading

| Component | Description | Weight |
| --- | --- | ---: |
| [Convolution Exercise #1](#convolution-exercise-1---10)  | 2D convolution with fixed `M = 7`, `N = 4` | 10% |
| [Convolution Exercise #2](#convolution-exercise-2---10) | 2D convolution with fixed `M = 10`, `N = 5` | 10% |
| [Convolution Exercise #3](#convolution-exercise-3---20) | 2D convolution with 4 fixed-dimension cases and 6 random-dimension cases, all satisfying `1 <= N <= M <= 254` | 20% |
| [Convolution Exercise #4](#convolution-exercise-4---20) | 2D convolution with advanced test cases | 20% |
| [GEMV Basic Exercise](#gemv-basic-exercise---20)  | Functional correctness in hardware and hidden test cases | 20% |
| [GEMV Performance Challenge](#gemv-performance-challenge---20) | Ranking by execution cycles, with hardware resources used as the tie-breaker | 20% |
| **Total** |  | **100%** |

## Background
---
### Weight Stationary
In the weight stationary dataflow, each PE's register stores the weight values and remains stationary during MACs. Input activations are propagated through the PE array and multiplied by the locally stored weights. The generated partial sums are then forwarded to other PEs or accumulated outside the PE.

![Weight Stationary](images/lab3/WS.png)


### Output Stationary
In the output stationary dataflow, each PE's register stores the partial sum of an output value and keeps it stationary during MACs. Input activations and weights are propagated through the PE array and multiplied inside the PE. The partial sum is accumulated locally until the output value is completed.


![Output Stationary](images/lab3/OS.png)

### Input Stationary
In the input stationary dataflow, each PE's register stores an input activation and keeps it stationary during MACs. Different weights are propagated through the PE array and multiplied by the locally stored input value. The generated partial sums are then forwarded to other PEs or accumulated for the corresponding outputs.

![Input Stationary](images/lab3/IS.png)

### Row Stationary
In the row stationary dataflow, each PE performs part of a row convolution while reusing kernel values and overlapping input activations locally. Input activations are shifted or propagated between neighboring PEs to reuse the overlapping data of adjacent convolution windows. The generated partial sums are accumulated across PEs to produce the final convolution output.

![Row Stationary](images/lab3/RS.png)



## Systolic Array Implementation
---
The goals of this lab are to familiarize you with the concepts of dataflows in systolic array architectures. This will get you hands-on experience with dataflow routing and processing element implementations. In this lab, you only need to construct the TPU module.

### Prerequisite

- Python3 with numpy library installed
- `iverilog` or `VCS` or `irun`
- `nWave` or `Verdi` or `GTKWave` or anything that can read `.vcd` or `.fsdb`
- Makefile
- Vivado 2024.1

### Requirements and Rules
#### General Rules

* Use an asynchronous active-low reset architecture.
* `in_valid` is asserted for one clock cycle. The dimension inputs are valid only during that cycle.
* Assert `busy` in response to `in_valid` and keep it high until all output data is ready. The next test case will begin only after `busy` returns low.
* For Lab 3-1, the execution cycle limit depends on the test case. Refer to `wait_finished` in `TESTBENCH/PATTERN.v` for the limit calculation.
* For Lab 3-2, there is no execution cycle limit for functional correctness grading. Execution cycles are used to rank designs in the GEMV Performance Challenge.
* Before each computation, the testbench initializes the output region of Global Buffer C to zero in Lab 3-1, and the board controller clears all of BRAM C to zero in Lab 3-2.

#### Lab 3-1

You need to perform 2D convolution between an `M × M` unsigned 8-bit input feature map and an `N × N` unsigned 8-bit kernel. The design must use a systolic array with (PEs) to perform the computation, using a dataflow of your choice. Accumulated results are unsigned 32-bit values, so pay careful attention to intermediate bit widths.

The following detailed requirements apply:

Row Stationary is recommended for Lab 3-1, but other dataflows are allowed.

* Implement your TPU design in `TPU.v`. You may add supporting Verilog files in the `RTL` directory.
* You may not modify `global_buffer.v` or the interface of `TPU.v`.
* `1 <= N <= M <= 254`
* Stride = 1
* Padding = 0
* Use the kernel in its original orientation; do not flip it horizontally or vertically.
* The output feature map has dimensions `(M-N+1) × (M-N+1)`.
* Implement your own data loader, PEs, and controller to schedule data from Global Buffer A and Global Buffer B into the PE array.
* The PE array may contain at most 16 PEs.
* Multiplication and accumulation for the convolution must be performed inside the PEs.
* The provided `global_buffer.v` can be treated as a behavioral model of BRAM.
* Before each simulation test case, the testbench loads Global Buffer A and Global Buffer B. The testbench keeps `in_valid` high for one cycle, and the `M` and `N` inputs are valid during that cycle.
* The TPU may only read from Global Buffer A and Global Buffer B; it cannot write to them.
* While `busy = 0`, keep all BRAM access enables (`A_ram_en`, `B_ram_en`, `C_ram_en`, and `C_wr_en`) at 0.
* When `busy` returns low, the testbench compares Global Buffer C against the golden result.

#### Lab 3-2

You need to perform GEMV between an `M × K` signed 8-bit matrix and a `K × 1` signed 8-bit vector using a PE array. The accumulated results are signed 32-bit values. The dataflow is not restricted; you may use Weight Stationary, Input Stationary, Output Stationary, or another dataflow of your choice.

BRAM A and BRAM B each use a 32-bit interface with four signed 8-bit positions. BRAM C uses a 128-bit interface with four signed 32-bit positions.

The following detailed requirements apply:

* Implement your design inside `TPU.v` without modifying its provided module interface. You may add supporting RTL files such as `PE.v`. The provided `PE.v` interface is a recommendation and can be changed.
* You may not modify the provided UART, BRAM, or `TPU_top.v` designs. The TAs will integrate and test your submitted `TPU.v` and supporting RTL files with these provided modules and interfaces.
* `1 <= M, K <= 254` and `N = 1`.
* For the 1024-word BRAM configuration used in hardware verification, the matrix dimensions must also satisfy `ceil(M / 4) × K <= 1024`.
* During hardware verification, the provided UART flow loads input data into BRAM A and BRAM B. The UART flow keeps `in_valid` high for one cycle, and the `K`, `M`, and `N` inputs are valid during that cycle.
* The TPU may only read from BRAM A and BRAM B; it cannot write to them. The TPU outputs `A_wr_en`, `A_data_in`, `B_wr_en`, and `B_data_in` are unused in the provided board design.
* Complete all computation and write the final results to BRAM C before setting `busy` to 0.
* While `busy = 0`, keep all BRAM access enables (`A_ram_en`, `B_ram_en`, `C_ram_en`, and `C_wr_en`) at 0.
* When `busy` returns low, the hardware verification flow reads BRAM C and compares the returned values against the golden result.
* After implementation with Vivado 2024.1, the design must satisfy `LUT <= 2000`, `FF <= 2000`, `DSP <= 16`, and `WNS >= 0`. Designs that violate any of these limits receive no credit for this part.


```{note}
For the details of the data mapping into the global buffers or BRAM, please refer to the ***[Appendix](#appendix)***. The Appendix describes the memory layouts used for the 2D convolution in Lab 3-1 and the GEMV in Lab 3-2.
```

### Getting Started
::: danger
This lab will require a beginner’s level of verilog.
```bash
$ git clone https://github.com/nycu-caslab/AAML2026-Lab3.git
```
:::

```{note}
The testbench generates waveform to `dump.vcd` or `dump.fsdb` (change the output file in the `TESTBENCH.v`).
After running the simulation each time, you may use your waveform viewer to check it out. 


    nWave (dump.fsdb | dump.vcd)
    or
    gtkwave dump.vcd
```
```{note}
1. The default verilog compiler in the makefile is `iverilog`, and using `vvp` as simulator.
If you have the licence of `VCS` and want faster simulation, you may use the `Makefile_vcs`.
2. The `Makefile_ncverilog` is for reference only, our Cadence licence has been down for a while :(
```

### Interface and Block Diagram
### lab 3-1
**Block Diagram**

<img src="images/lab3/block_diagram-2.png" width="560px">

**Table 1: The Control Signals**
| I/O    | Signal name | Bit width | Description                                                                                                               |
| ------ | ----------- | --------: | ------------------------------------------------------------------------------------------------------------------------- |
| Input  | `clk`       |         1 | The clock signal                                                                                                          |
| Input  | `rst_n`     |         1 | The reset signal, which is active low                                                                                     |
| Input  | `in_valid`  |         1 | High for one cycle when the input and dimension signals are valid. |
| Input  | `M`         |         8 | Dimension `M` of the `M × M` input feature map                                                                            |
| Input  | `N`         |         8 | Dimension `N` of the `N × N` convolution kernel                                                                           |
| Output | `busy`      |         1 | High when the design is busy. We will check your answer when `busy` returns low after each `in_valid`.             |



**Table 2: The BRAM Interface of BRAM A and BRAM B**

The I/O directions in Tables 2 and 3 are relative to the BRAM module. The TPU outputs `A_ram_en`, `B_ram_en`, and `C_ram_en` drive the corresponding BRAM's `ram_en` input during computation.

|  I/O   | Signal name  | Bit width | Description                                  |
|  ----  | ----         | ----      |  ----                                        |     
| Input  | `ram_en`      | 1         | Enables BRAM reads and writes when high. |
| Input  | `wr_en`        | 1         | Selects write when high and read when low, provided `ram_en` is high. |
| Input  | `index`        | 16        | The address of the BRAM to be read or written. |
| Input  | `data_in`      | 32        | The data input to write to the BRAM.         |
| Output | `data_out`     | 32        | The data output from the BRAM.               |

**Table 3: The BRAM Interface of BRAM C**
|  I/O   | Signal name  | Bit width | Description                                  |
|  ----  | ----         | ----      |  ----                                        |     
| Input  | `ram_en`      | 1         | Enables BRAM reads and writes when high. |
| Input  | `wr_en`        | 1         | Selects write when high and read when low, provided `ram_en` is high. |
| Input  | `index`        | 16        | The address of the BRAM to be read or written. |
| Input  | `data_in`      | 128       | The data input to write to the BRAM.         |
| Output | `data_out`     | 128       | The data output from the BRAM.               |
``` {note}
The 128-bit data input stands for 4 * 32-bit values, allowing 4 elements to be written simultaneously to BRAM C.

Global Buffer reads and writes occur on the falling edge of `clk`; read data is available after that edge.
```

### lab 3-2

**Table 1: The Control Signals**
| I/O    | Signal name | Bit width | Description                                                                                           |
| ------ | ----------- | --------: | ----------------------------------------------------------------------------------------------------- |
| Input  | `clk`       |         1 | The clock signal                                                                                      |
| Input  | `rst_n`     |         1 | The reset signal, which is active low                                                                 |
| Input  | `in_valid`  |         1 | High for one cycle when the input and dimension signals are valid.                                   |
| Input  | `K`         |         8 | Dimension `K` of the input matrix and vector                                                          |
| Input  | `M`         |         8 | Dimension `M` of the `M × K` input matrix                                                             |
| Input  | `N`         |         8 | Dimension `N` of the `K × N` input vector. `N` is fixed to 1 for GEMV                                 |
| Output | `busy`      |         1 | High when the design is busy. We will check your answer when `busy` returns low after each `in_valid`. |

**Table 2: The BRAM Interface of A BRAM**

The I/O directions in Tables 2–4 are relative to the BRAM interface. The TPU outputs `A_ram_en`, `B_ram_en`, and `C_ram_en` enable access to the corresponding BRAM.

| I/O    | Signal name | Bit width | Description                                  |
| ------ | ----------- | --------- | -------------------------------------------- |
| Input  | `ram_en`    | 1         | Enables BRAM access when high. |
| Input  | `wr_en`     | 1         | The write enable signal.                     |
| Input  | `index`     | 16        | The address of the BRAM to be read or written. |
| Input  | `data_in`   | 32        | The data input to write to the BRAM.         |
| Output | `data_out`  | 32        | The data output from the BRAM.               |

**Table 3: The BRAM Interface of B BRAM**
| I/O    | Signal name | Bit width | Description                                  |
| ------ | ----------- | --------- | -------------------------------------------- |
| Input  | `ram_en`    | 1         | Enables BRAM access when high. |
| Input  | `wr_en`     | 1         | The write enable signal.                     |
| Input  | `index`     | 16        | The address of the BRAM to be read or written. |
| Input  | `data_in`   | 32        | The data input to write to the BRAM.         |
| Output | `data_out`  | 32        | The data output from the BRAM.               |

**Table 4: The BRAM Interface of C BRAM**
|  I/O   | Signal name  | Bit width | Description                                  |
|  ----  | ----         | ----      |  ----                                        |     
| Input  | `ram_en`      | 1         | Enables BRAM access when high. |
| Input  | `wr_en`        | 1         | Selects write when high and read when low, provided `ram_en` is high.                     |
| Input  | `index`        | 16        | The address of the BRAM to be read or written. |
| Input  | `data_in`      | 128       | The data input to write to the BRAM.         |
| Output | `data_out`     | 128       | The data output from the BRAM.               |

```{note}
BRAM reads and writes occur on the rising edge of `clk`; read data is available after that edge.
```

## Convolution Exercise #1 - 10%
---
- Input data:
    - Input feature map (M * M) and Kernel (N * N) where M = 7, N = 4
    - control signal (refer to details in table 1, 2, 3)

- Required Output:
    - the Output feature map (UINT32) of the 2D convolution
- Steps:
    1. Take data from global buffer
    2. Use the data from global buffer to calculate with PEs
    3. Output the result to C global buffer
    4. `\lab3-1$ make verif1`
        - 10 test cases with fixed input dimensions and random data values
    5. The bench will tell if you did it correctly

## Convolution Exercise #2 - 10%
---
- Input data:
    - Input feature map (M * M) and Kernel (N * N) where M = 10, N = 5
    - control signal (refer to details in table 1, 2, 3)

- Required Output:
    - the Output feature map (UINT32) of the 2D convolution
- Steps:
    1. refer to Convolution Exercise #1
    2. `\lab3-1$ make verif2`
        - 10 test cases with fixed input dimensions and random data values

## Convolution Exercise #3 - 20%
---
- Input data:
    - Input feature map (M * M) and Kernel (N * N) where 1 <= N <= M <= 254
    - control signal (refer to details in table 1, 2, 3)

- Required Output:
    - the Output feature map (UINT32) of the 2D convolution

- Steps:
    1. refer to Convolution Exercise #1
    2. `\lab3-1$ make verif3`
        - 10 test cases: the first 4 use fixed dimensions, and the remaining 6 use random dimensions. All cases use random data values.

## Convolution Exercise #4 - 20%
---
- Input data:
    - Input feature map (M * M) and Kernel (N * N) where 1 <= N <= M <= 254
    - control signal (refer to details in table 1, 2, 3)

- Required Output:
    - the Output feature map (UINT32) of the 2D convolution

- Steps:
    1. refer to Convolution Exercise #1
    2. `\lab3-1$ make verif4`
        - Advanced test cases
        
## GEMV Basic Exercise - 20%
---
- Input data:
    - Input Matrix (INT8) and Input Vector (INT8)
- Required Output:
    - the Output Vector (INT32) of the GEMV operation
- Steps:
    1. Take data from BRAM A and BRAM B
    2. Process the BRAM data using your TPU architecture and PE array
    3. Output the result to BRAM C
    4. From the `lab3-2` directory, build the FPGA bitstream, program the connected Arty A7-100T, and run UART verification in order:

        ```bash
        make generate_gemv_bitstream
        make program
        make hardware_verify
        ```

        - `make generate_gemv_bitstream` creates the Vivado project, runs synthesis and implementation, and generates the bitstream.
        - `make program` programs the FPGA with the generated bitstream.
        - `make hardware_verify` runs the UART tests against the programmed FPGA. Follow the terminal prompts. Pass all hardware verification test cases to earn 10%.
        - After changing the RTL, repeat all three commands to test the updated design.
    5. Pass the hidden test cases (10%)

## GEMV Performance Challenge - 20%
---

Only designs that pass all GEMV Basic Exercise test cases are eligible for the performance challenge. Performance will be evaluated using the TA's hidden test cases. Designs with fewer total execution cycles will receive a higher ranking. If two or more designs have the same cycle count, the design with the smaller hardware resource score will rank higher:

`LUT + FF + DSP × 100`

The performance challenge score is assigned according to the final ranking percentile among all eligible designs:

| Performance ranking | Score |
| --- | ---: |
| Top 1–20% | 20 |
| Top 21–40% | 16 |
| Top 41–60% | 12 |
| Top 61–80% | 8 |
| Top 81–100% | 4 |




## Appendix
---

### Lab 3-1 Memory Mapping - 2D Convolution

The following sections describe the memory layout for 2D convolution in Lab 3-1.

#### Global Buffer A - Input Feature Map

The input feature map is stored column by column. Three zero-padded elements are appended after each column so that the TPU can safely access four vertically adjacent 8-bit input elements at a time.

![Global Buffer A mapping](images/lab3/lab3-1_A.png)


<!-- Figure for Lab 3-1 Global Buffer A -->

#### Global Buffer B - Kernel

The kernel is divided into row tiles. Each word in Global Buffer B contains four vertically adjacent 8-bit kernel elements from the same column. If fewer than four elements remain in the last row tile, the remaining positions are zero-padded.

![Global Buffer B mapping](images/lab3/lab3-1_B.png)



<!-- Figure for Lab 3-1 Global Buffer B -->

#### Global Buffer C - Output

The output feature map is stored row by row. Each 128-bit word contains four horizontally adjacent 32-bit accumulated output values. If fewer than four output values remain in a row, the remaining positions are zero-padded.

![Global Buffer C mapping](images/lab3/lab3-1_C.png)


<!-- Figure for Lab 3-1 Global Buffer C -->

### Lab 3-2 Memory Mapping - GEMV

#### BRAM A Mapping

Matrix A is an `M × K` matrix. Each 32-bit word in BRAM A stores four signed 8-bit elements from four adjacent rows at the same column. When fewer than four rows remain, the unused positions are zero-padded. The following figure shows how matrix A is stored in BRAM A.

![Matrix A mapping in BRAM A](images/lab3/A_Bram.png)

#### BRAM B Mapping

Because `N` is fixed to 1, matrix B is a `K × 1` vector. Each 32-bit word in BRAM B stores one signed 8-bit vector element, and the other three positions are zero-padded. The following figure shows how vector B is stored in BRAM B.

![Vector B mapping in BRAM B](images/lab3/B_Bram.png)

#### BRAM C Mapping

The GEMV result C is an `M × 1` vector of signed 32-bit accumulated values. Each 128-bit word in BRAM C stores one result element, and the other three 32-bit positions are zero-padded. The following figure shows how vector C is stored in BRAM C.

![Vector C mapping in BRAM C](images/lab3/C_Bram.png)

## Submission
---

Please organize your submission files into a zip archive structured as follows:
```
YourID.zip
    └── YourID/
        ├── lab3-1/
        │   ├── TPU.v
        │   └── other supporting RTL files...
        └── lab3-2/
            ├── TPU.v
            ├── PE.v (if used or modified)
            └── other supporting RTL files...
```

For Lab 3-2, obtain `TPU.v`, `PE.v`, and any supporting RTL files from `lab3-2/BOARD/reference/rtl`. Keep the flat submission structure shown above; you do not need to include the `BOARD/reference/rtl` directory hierarchy in the zip archive.

```{important}
1. Make sure all RTL files required by your design are included.
2. For Lab 3-1, you **DO NOT** have to submit `global_buffer.v`.
3. For Lab 3-2, submit only `TPU.v` and the supporting RTL files required by your design. Do not submit the provided UART, BRAM, `TPU_top.v`, constraint, or IP files.

The TAs will integrate your submitted files with the provided interfaces and board design. Your submission must compile and run without modifications. **PLAGIARISM is not allowed**.
```
