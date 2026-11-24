# Final Project: BERT-Medium Accelerator

## Introduction
----
In this final project, you are required to design a NPU to accelerate the BERT-Medium model for natural language processing (NLP) sequence classification tasks. The primary goal is to minimize inference latency—targeting well under the software-only execution time—while maintaining the model's accuracy.

### Selected Model
We use an INT8 quantized version of the BERT-Medium architecture. The default implementation provided in the template is configured with a fixed sequence length of 125 to optimize the matrix dimensions for hardware acceleration while preserving sequence classification performance. BERT is an encoder-based model used to classify sequences into positive or negative. We also introduced specialized modifications to BERT to support INT8-based QKV computations.

You don’t need to integrate the model yourself; the bare-metal C headers are already included in the CFU template. You can inspect the .tflite model architecture using [Netron](https://netron.app/).


```{note}
The model takes a long time (**>60 minutes**) to inference the full evaluation dataset using the pure software implementation. Please start your project as soon as possible.
```

## Setup
----
### Clone the Final Project Template
We have integrated all the necessary files into the repo. Please clone it into a clean environment.
```bash
git clone git@github.com:nycu-caslab/AAML-2026-Project.git
```
After you clone the repository, change the remote origin to your own repo. Then, make your repo private.

## Requirements
----
### Files
- **Required Modifications**
    1. `tensorflow/lite/kernels/internal/reference/batch_matmul.h`
    2. `tensorflow/lite/kernels/internal/reference/fully_connected.h`
    3. `NPU.v`

- **Selected Modifications**
    - You can adjust the clock frequency by modifying **PLATFORM_CLOCK_HZ** in **sw/app/platform_config.h**. This parameter is connected to the hardware configuration, so any changes will be reflected in the hardware settings.
    - Remember to run **make prog** again after modifying PLATFORM_CLOCK_HZ.
    - You are only allowed to adjust the clock frequency in increments of **5 Hz**.

- **Other Modifiable files**
    - Other tflm operation files related to this model such as transpose.
- **Custom files**
    - You can add any files you need to improve your design.
    - You can even adjust the memory layout if you know how to do. 

```{important}
**DO NOT MODIFY ** any other files under tflite-micro
```
```{important}
**You are not permitted to modify the AXI interface, but you can add ports for hardware coherence.
```
```{important}
If you want to modify tflm related files, you must copy those files to the tflm_patches
```

### Golden Test
There are five test cases for you to validate the output results. The software will automatically compare your results against the golden outputs.

To prevent someone from designing based on input data, we might use some hidden case to test your design.

The permitted error tolerance is 0.01%.

- You can refer the result below:
    ```language = c
    TinyRISC-V SoC Platform
    =======================
     l: Lab menu
     t: TFLM inference, verification, and cycles
     p: Performance counter tests
     i: System info
    main> t

	Running TFLM inference, verification, and cycles
	
	=== TFLM Functional Verification ===
	Model: bert_medium_int8.tflite (42311176 bytes)
	Tensor arena: 8000000 bytes
	Profile: bert_medium_classifier
	Input profile: public text fixtures
	Output profile: golden outputs check, tolerance=0.000100
	Samples: 5
	
	========== Running Test Case 0 ==========
	Loading attention mask for public_test_0...
	Loading input IDs for public_test_0...
	Running inference...
	Inference complete.
	Output type: 1
	Output bytes: 8
	Output[0]: Actual=0.076653, Golden=0.076653 [PASS]
	Output[1]: Actual=0.204408, Golden=0.204408 [PASS]
	[Status] public_test_0: ALL OUTPUTS PASS.
	
	Cycles: 52635942199
	Time (65000000 Hz): 809783 ms
	
	========== Running Test Case 1 ==========
	Loading attention mask for public_test_1...
	Loading input IDs for public_test_1...
	Running inference...
	Inference complete.
	Output type: 1
	Output bytes: 8
	Output[0]: Actual=0.101352, Golden=0.101352 [PASS]
	Output[1]: Actual=0.186522, Golden=0.186522 [PASS]
	[Status] public_test_1: ALL OUTPUTS PASS.
	
	Cycles: 52636854654
	Time (65000000 Hz): 809797 ms
	
	========== Running Test Case 2 ==========
	Loading attention mask for public_test_2...
	Loading input IDs for public_test_2...
	Running inference...
	Inference complete.
	Output type: 1
	Output bytes: 8
	Output[0]: Actual=0.017034, Golden=0.017033 [PASS]
	Output[1]: Actual=0.096242, Golden=0.096242 [PASS]
	[Status] public_test_2: ALL OUTPUTS PASS.
	
	Cycles: 52635456300
	Time (65000000 Hz): 809776 ms
	
	========== Running Test Case 3 ==========
	Loading attention mask for public_test_3...
	Loading input IDs for public_test_3...
	Running inference...
	Inference complete.
	Output type: 1
	Output bytes: 8
	Output[0]: Actual=0.094539, Golden=0.094539 [PASS]
	Output[1]: Actual=0.215480, Golden=0.215480 [PASS]
	[Status] public_test_3: ALL OUTPUTS PASS.
	
	Cycles: 52633864419
	Time (65000000 Hz): 809751 ms
	
	========== Running Test Case 4 ==========
	Loading attention mask for public_test_4...
	Loading input IDs for public_test_4...
	Running inference...
	Inference complete.
	Output type: 1
	Output bytes: 8
	Output[0]: Actual=0.030661, Golden=0.030661 [PASS]
	Output[1]: Actual=0.160120, Golden=0.160120 [PASS]
	[Status] public_test_4: ALL OUTPUTS PASS.
	
	Cycles: 52628198640
	Time (65000000 Hz): 809664 ms
	
	========== Summary ==========
	Total PASS: 5
	Total FAIL: 0
	
	Total for 5 samples:
	Cycles: 263170316212
	Time (65000000 Hz): 4048774 ms
	---
    ```

### Performance
There are no restrictions on hardware resources, as long as the design can run on the Arty A7 board. We strongly encourage students to focus on accelerator design as much as possible.

You need to minimize the latency as low as it could be.

```{Tip}
If you just want to check the latency of your design, it would be easier to run a test input instead of whole process of evaluation.
```

## Presentation
----
- You should give a presentation in the last class of this semester.
    - **2026/12/24(Thu.)**, starting at **10:10**.
- Time Limit: At most **5 minutes** per team.
- Content:
    - Task Assignment Table
    - Accelerated Items
    - Introduction: Strategy for SW profiling & HW architecture.
    - Implementation: Details of your NPU & kernel optimizations.
    - Evaluation: Final accuracy & latency & hardware utilization.

```{important}
Each group member must present the part they claimed responsibility for in the task assignment table.
```

## Grading Policy
----
- Basic Requirements
    - You must to design the hardware accelerator for the batch_matmul and fully_connected.
    - Your design must achieve performance equal to or better than the baseline implementation using SIMD MAC and AXI burst transfers.
    - Your design must pass all output validation tests.
    - Your clock frequency must be in increments of **5 Hz**
    - Your slacks must be **non-negative**.

- **Ranking:** A leaderboard will be released after the deadline.

## Grading Formula
----
### Basic Requirements
$$
\begin{gathered}
\text{BASIC} =
\begin{cases}
    1 & \text{if the basic requirements are met} \\
    0 & \text{if the basic requirements are not met}
\end{cases}
\end{gathered}
$$

### Latency
$$
\begin{gathered}
\text{LAT}_{\text{rank}} = \min \left( 20 \times \frac{N - \text{Rank}_{\text{student}}}{N}, 20 \right) \\
\text{where } N = \text{number of students}, \quad \text{Rank}_{\text{student}} \in [0, N - 1]
\end{gathered}
$$

### Presentation

$$
\begin{gathered}
\text{Present} = 
\begin{cases} 
    -30 & \text{if you submit a plain impl of Lab 2 OR do not present} \\
    -0 & \text{otherwise} 
\end{cases}
\end{gathered}
$$

### Final score

$$
\begin{gathered}
\text{Score} = 
\text{BASIC} \times (\text{80} + \text{LAT}_{\text{rank}} + \text{Present}) \\
(\text{Highest score} = 1 \times (80 + 20 - 0) = 100)
\end{gathered}
$$

## Submission
----
- **Deadline: 2026/12/24 10:00** 
    - Your last commit will be considered as your submission time. (We only accept timestamps **from the main branch**.)
    - Submit your repo link and slides **to E3** before the deadline.
    - You should keep your repo private until 2026/12/24 10:00.
    - You **must** change your repo to public after 2026/12/24 10:00, **otherwise you will receive 0 points**.
- Please fork the repo and push your work to it.
    - If you use a custom model architecture, you **MUST** follow these steps:
        1. Explicitly state in README.md that you are using a custom memory layout.
        2. Upload your modified `.tflite` file and the converted `.h` file under the corresponding path.
        
- Grading workflow will be:
    1. Clone your repo.
    2. `make prog && make run` 
    3. Run the evaluation script.
    4. Record the metrics.
