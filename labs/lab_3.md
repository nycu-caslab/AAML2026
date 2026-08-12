# Lab 3 : Systolic Array
---
## Goal of this lab
---
- [Row Stationary Exercise #1 - 10%](#row-stationary-exercise-1---10) 
- [Row Stationary Exercise #2 - 10%](#row-stationary-exercise-2---10) 
- [Row Stationary Exercise #3 - 20%](#row-stationary-exercise-3---20) 
- [Row Stationary Exercise #4 - 20%](#row-stationary-exercise-4---20)
- [GEMV Basic Exercise - 30%](#gemv-basic-exercise---30) 
- [GEMV Performance Challenge - 10%](#gemv-performance-challenge---10)



## Introduction
---
The systolic array used by Google Tensor Processing Unit (TPU) accelerates the matrix computation by using the dataflow operation. The systolic array contains multiple processing elements (PEs), each of them is responsible for the multiply–and-accumulate (MAC) operation. It can performs multiple elements in a matrix simultaneously and achieves high computational throughput.

In this lab, we will use Verilog to implement PEs and a small systolic array composed of multiple PEs to accelerate convolution and general matrix vector multiplication (GEMV). Different stationary dataflows, such as Weight Stationary , Output Stationary , and Row Stationary, can be considered when designing the PE array. 

(Hint: The weight stationary is more complicated than output stationary.)
```{note}
1. This lab does not require demonstration, submit your code to E3 only, details are in [Submission](#submission).


2. You will be asked to put your systolic array on FPGA in ***lab 5***, so please make sure that your systolic array is a synthesizable circuit.
```

## Systolic Array Implementation
---
The goals of this lab are to familiarize you with the concepts of dataflows in systolic array architectures. This will get you hand-on experience with dataflow routing and processing elements implementations. In this lab, you only need to construct the TPU module.

### Prerequisite

- Python3 with numpy library installed
- `iverilog` or `VCS` or `irun`
- `nWave` or `Verdi` or `GTKWave` or anything that can read `.vcd` or `.fsdb`
- Makefile

### Requirements
#### lab 3-1

You need to perform Convolution using the row stationary method with correct functional simulation using Processing Elements (PEs). That is, this design should perform 2D convolution between an M × M 8-bit matrix and an N × N 8-bit kernel.

Your design should be written in Verilog. There is no limitation on how you implement your design.
Your PEs shouldn’t exceed 4 × 4, and a 2D systolic array architecture is recommended.
The design uses 8-bit input data and 32-bit accumulated data. Please be careful with bit-width issues.
The total global buffer size is (1024 + 256 × 2) KiB.
#### lab 3-2
You need to perform GEMV using any one of the stationary methods with correct functional simulation using PEs. This design should perform multiplication between an M × K 4-bit matrix and a K × 1 8-bit vector.

Your design should be written in Verilog. There is no limitation on how you implement your design.
Your PEs shouldn’t exceed 4 × 4. The design uses 4-bit matrix data, 8-bit vector data, and 32-bit accumulated data. Please be careful with bit-width issues.
The total global buffer size is (1024 + 256 + 128) KiB.

### Getting Started
::: danger
This lab will require a beginner’s level of verilog.
```bash
$ git clone https://github.com/nycu-caslab/AAML2025-Lab3.git
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

**Tabel 1: The Control Signals**
| I/O    | Signal name | Bit width | Description                                                                                                               |
| ------ | ----------- | --------: | ------------------------------------------------------------------------------------------------------------------------- |
| Input  | `clk`       |         1 | The clock signal                                                                                                          |
| Input  | `rst_n`     |         1 | The reset signal, which is active low                                                                                     |
| Input  | `in_valid`  |         1 | The input is valid when in_valid is high and will only high for one cycle |
| Input  | `M`         |         8 | Dimension `M` of the `M × M` input feature map                                                                            |
| Input  | `N`         |         8 | Dimension `N` of the `N × N` convolution kernel                                                                           |
| Output | `busy`      |         1 | High when the design is busy. Pattern will check your answer when busy is low after every in_valid.                |



**Tabel 2: The SRAM interface of A and B SRAM**
|  I/O   | Signal name  | Bit width | Description                                  |
|  ----  | ----         | ----      |  ----                                        |     
| Input  | wr_en        | 1         | The write enable signal.                     |
| Input  | index        | 16        | The address of the sram to be read or write. |
| Input  | data_in      | 32        | The data input to write to the SRAM          |
| Output | data_out     | 32        | The data output from the SRAM                |

**Tabel 3: The SRAM interface of C SRAM**
|  I/O   | Signal name  | Bit width | Description                                  |
|  ----  | ----         | ----      |  ----                                        |     
| Input  | wr_en        | 1         | The write enable signal.                     |
| Input  | index        | 16        | The address of the sram to be read or write. |
| Input  | data_in      | 128       | The data input to write to the SRAM          |
| Output | data_out     | 128       | The data output from the SRAM                |
``` {note}
The 128-bit data input stands for 4 * 32-bit values, allowing 4 elements to be written simultaneously to the C SRAM.
```

### lab 3-2

**Tabel 1: The Control Signals**
| I/O    | Signal name | Bit width | Description                                                                                           |
| ------ | ----------- | --------: | ----------------------------------------------------------------------------------------------------- |
| Input  | `clk`       |         1 | The clock signal                                                                                      |
| Input  | `rst_n`     |         1 | The reset signal, which is active low                                                                 |
| Input  | `in_valid`  |         1 | The input is valid when `in_valid` is high and will only high for one cycle                           |
| Input  | `K`         |         8 | Dimension `K` of the input matrix and vector                                                          |
| Input  | `M`         |         8 | Dimension `M` of the `M × K` input matrix                                                             |
| Input  | `N`         |         8 | Dimension `N` of the `K × N` input vector. `N` is fixed to 1 for GEMV                                 |
| Output | `busy`      |         1 | High when the design is busy. Pattern will check your answer when busy is low after every `in_valid`. |

**Table 2: The SRAM Interface of A SRAM**
| I/O    | Signal name | Bit width | Description                                  |
| ------ | ----------- | --------- | -------------------------------------------- |
| Input  | `wr_en`     | 1         | The write enable signal.                     |
| Input  | `index`     | 16        | The address of the SRAM to be read or write. |
| Input  | `data_in`   | 16        | The data input to write to the SRAM          |
| Output | `data_out`  | 16        | The data output from the SRAM                |

**Table 3: The SRAM Interface of B SRAM**
| I/O    | Signal name | Bit width | Description                                  |
| ------ | ----------- | --------- | -------------------------------------------- |
| Input  | `wr_en`     | 1         | The write enable signal.                     |
| Input  | `index`     | 16        | The address of the SRAM to be read or write. |
| Input  | `data_in`   | 32        | The data input to write to the SRAM          |
| Output | `data_out`  | 32        | The data output from the SRAM                |

**Table 4: The SRAM Interface of C SRAM**
|  I/O   | Signal name  | Bit width | Description                                  |
|  ----  | ----         | ----      |  ----                                        |     
| Input  | wr_en        | 1         | The write enable signal.                     |
| Input  | index        | 16        | The address of the sram to be read or write. |
| Input  | data_in      | 128       | The data input to write to the SRAM          |
| Output | data_out     | 128       | The data output from the SRAM  

### Rules

* Your TPU design (`TPU.v`) should be under the top module provided by TA. It's fine to add various new files in the `RTL` directory.

* You may not modify `global_buffer.v`.

* At the start of the simulation, the testbench will load global buffer A & B, assuming that the CPU or DMA has already prepared the data for the TPU in the global buffer. When signal `in_valid == 1`, the input size parameters (`m`, `n`, `k`) will be available for the TPU for **only one cycle**.

```{note}
For the details of the data mapping into the global buffers, please refer to the ***[Appendix](#appendix)***. The Appendix describes the memory layouts used for the Row Stationary convolution in Lab 3-1 and the GEMV in Lab 3-2.
```

* The testbench will compare your output global buffer with the golden result when you finish the calculation, that is, when `busy == 0`. Then, you need to wait for the next `in_valid` for the next test case.

* You should implement your own data loader, PEs, and controller to schedule the data in global buffer A & B to be calculated in the PE array.

* The number of PEs must not exceed **16** (maximum `4 × 4` PEs).

* For **Lab 3-1**, you must perform 2D convolution using the **Row Stationary** method and a PE array.

  * You may not modify the interface of `TPU.v`.
  * `1 <= N <= M <= 254`
  * Stride = 1
  * Padding = 0
  * Multiplication and accumulation must be performed inside the PEs.

* For **Lab 3-2**, you must perform GEMV using one of the **Output Stationary, Weight Stationary, or Row Stationary** methods and a PE array.

  * You may not modify the interface of `TPU.v`.
  * Multiplication and accumulation must be performed inside the PEs. 
  * Performance is evaluated by cycle count, and a lower cycle count results in a higher ranking.

* You need to set `busy` to high immediately after `in_valid` falls from high to low.

* Use an asynchronous active-low reset architecture.

* The execution latency per test case is limited to 1,500,000 cycles.



## Row Stationary Exercise #1 - 10%
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
        - 10 test cases of fixed input
    5. The bench will tell if you did it correctly

## Row Stationary Exercise #2 - 10%
---
- Input data:
    - Input feature map (M * M) and Kernel (N * N) where M = 10, N = 5
    - control signal (refer to details in table 1, 2, 3)

- Required Output:
    - the Output feature map (UINT32) of the 2D convolution
- Steps:
    1. refer to Row Stationary Exercise #1
    2. `\lab3-1$ make verif2`
        - 10 test cases of fixed input

## Row Stationary Exercise #3 - 20%
---
- Input data:
    - Input feature map (M * M) and Kernel (N * N) where 1 <= N <= M <= 254
    - control signal (refer to details in table 1, 2, 3)

- Required Output:
    - the Output feature map (UINT32) of the 2D convolution

- Steps:
    1. refer to Row Stationary Exercise #1
    2. `\lab3-1$ make verif3`
        - 10 test cases of large random M, N

## Row Stationary Exercise #4 - 20%
---
- Input data:
    - Input feature map (M * M) and Kernel (N * N)
    - control signal (refer to details in table 1, 2, 3)

- Required Output:
    - the Output feature map (UINT32) of the 2D convolution

- Steps:
    1. refer to Row Stationary Exercise #1
    2. `\lab3-1$ make verif4`
        - Extreme test cases
        
## GEMV Basic Exercise - 30%
---
- Input data:
    - Input Matrix (UINT4) and Input Vector (UINT8)
- Required Output:
    - the Output Vector (UINT32) of the GEMV operation
- Steps:
    1. Take data from global buffer
    2. Use the data from global buffer to calculate with PEs
    3. Output the result to C global buffer
    4. `\lab3-2$ make verif1`
        - 10 explicit testcases (10%)
    5. `\lab3-2$ make verif2`
        - 10 explicit testcases (10%)
    6. `\lab3-2$ make verif3`
        - 25 hidden testcases (10%)
    7. The bench will tell if you did it correctly

## GEMV Performance Challenge - 10%
---
-   Input data: 
    -   refer to GEMV Basic Exercise
-   Required Output:
    -   refer to GEMV Basic Exercise
-   Steps:
    1.  refer to GEMV Basic Exercise 
    2.  Run `\lab3-2$ make verif4` for performance evaluation
        -   The lower the total cycle count, the higher the ranking and score




## Appendix
---

### Lab 3-1 Memory Mapping - Row Stationary Convolution

Lab 3-1 uses a memory layout suitable for Row Stationary execution.

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

#### Memory Mapping - Type A (with transpose)

The matrix A in global buffer A is placed with a transposed style, and other spaces are all 0-padded
<img src="https://hackmd.io/_uploads/S18IElrj2.png" width="660px">


Example of a 10 * 7 matrix

<img src="images/lab3/A_3.png" width="660px">

the memory layout of this matrix looks like (note the transpose in the layout)

<img src="images/lab3/A_4.png" width="150px">

#### Memory Mapping - Type B (without transpose)

<img src="https://hackmd.io/_uploads/BJYR4gSo3.png" width="660px">

The matrix B looks more forward in memory layout, for example, a 7 * 9 matrix

<img src="images/lab3/B_1.png" width="660px">

looks like this in global buffer B, with 0-padded also

<img src="images/lab3/B_2.png" width="150px">

## Submission
---

Please organize your submission files into a zip archive structured as follows:
```
YourID.zip
    └── YourID/
        ├── lab3-1/ 
        |   ├──TPU.v
        |   └──other files you added inside the RTL directory...
        └── lab3-2/ 
            ├──TPU.v
            └──other files you added inside the RTL directory...
```

```{important}
1. Make sure your files are well included! 
2. You **DO NOT** have to submit the `global_buffer.v`.

TAs should be able to run your project without any modification. If TAs cannot compile or run your code, **you can't get any scores**. Also, **PLAGIARISM is not allowed**.
```
