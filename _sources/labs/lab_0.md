# Lab 0 : Environment Setup

## Introduction
Throughout AAML 2026, we will use the AAML RISC-V SoC platform to explore hardware/software co-design for machine-learning acceleration. The platform integrates a RISC-V CPU, a customizable ML accelerator connected through custom instructions and AXI, and TensorFlow Lite for Microcontrollers (TFLM). In Lab 0, you will set up the required tools, program the reference SoC onto the FPGA, and run a reference TFLM application to verify the development environment for the following labs.
## Goal of this lab
- [Identify the roles of the CPU, custom accelerator, AXI interface, custom instructions, and TFLM in the AAML RISC-V SoC platform.](#aaml-risc-v-soc-platform)
- [Install and verify Vivado and the RISC-V cross-compilation toolchain required by the platform.](#toolchain-setup)
- [Clone and prepare the `AAML-sw-dev` version of `CUSTOM_SoC_Platform`.](#platform-setup)
- [Build the reference SoC and program it onto a Nexys A7-100T FPGA.](#running-the-platform)
- [Build and upload a reference TFLM application, and verify its operation through the UART console.](#running-the-platform)
## AAML RISC-V SoC Platform

### RISC-V CPU
For a CPU to function, it must rely on the Instruction Set Architecture (ISA). An instruction set is a predefined list of commands that determines which operations the hardware can understand and execute. In this Lab, we use **RISC-V**, an open source ISA.  

Our CPU supports RV32IM, with M-mode only supporting `mcycle` and `mcycleh`, and it also supports the `fence.i` instructions.  

Moreover, the hardware supports `CBO` (Cache-Block Management Operations). We implement the `zicbom` extension, which allows software to manually manage cache coherence by cleaning or invalidating cache blocks.

### Extended ISAs
ISA is divided into two levels: the Base ISA and the Extended ISAs.  

- Base ISA: For RV32I as an example. it only defines the basic integer instructions that a CPU must support, such as `add`, `lw`, `sw`, and `beq`.
- Extended ISA: Depending on specific computational needs — such as floating-point arithmetic, vector and matrix processing, or even AI tensor operations — we can selectively incorporate these extensions into CPU designs.
  
The official specification reserves blank opcodes (such as custom-0), allowing developers like us to extend R-type formatted custom instructions.  

![R-type](images/lab0/R-type.png)
- We referenced custom-0 and R-type
- Custom-0 is an opcode retained by the official
- NPU can be controlled by function3 and function7

### NPU (block)

### AXI
### TensorFlow Lite For Micro (TFLM)

## Porting AAML RISC-V SoC to FPGA
This section will guide you through the core three steps from installing the toolchain, cloning the platform, to running the platform.

### Toolchain Setup
#### Step 0: Get Your FPGA Board
[Nexys A7-100T](https://digilent.com/reference/programmable-logic/nexys-a7/start) is used in this course, contact the TAs if you haven't get one.
#### Step 1: Install Vivado
- Install Vivado
We recommend you use version 2023.2 or version 2024.1
> [Vivado Download page](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools/2023-2.html)

- Make sure vivado is installed correctly on your machine before you start by running:
``` bash
$ vivado -version
```
- Since the full package of Vivado is pretty big, you may check only the `Artix-7` option to save disk space and download time, and please note that the Vivado can take up to 8 hours to download, so plan to do that ahead of time!
![vivado_option](images/lab0/vivado_option.png)
#### Step 2: Install RISC-V toolchain ([Download linux-ubuntu](https://github.com/sifive/freedom-tools/releases/tag/v2020.08.0))
Open your terminal and execute the following commands to download and extract the toolchain:
![ToolChain](images/lab0/riscv_toolchain.png)

Download the August 2020 toolchain from freedom-tools and unpack the binaries to your home directory:
``` bash
$ tar xvfz ~/Downloads/riscv64-unknown-elf-gcc-10.1.0-2020.08.2-x86_64-linux-ubuntu14.tar.gz
```

Add the toolchain to your PATH in your `.bashrc` or `.zshrc`:
``` bash
export PATH=$PATH:$HOME/riscv64-unknown-elf-gcc-10.1.0-2020.08.2-x86_64-linux-ubuntu14/bin
```

### Platform Setup
#### Step 0: Initial Preparation
Prepare a clean directory in your Linux environment and navigate into it.
#### Step 1: Clone the Repository
```bash
touch {your folder name}
cd {your folder name}
git clone -b AAML-sw-dev git@github.com:TTY-RISCV/CUSTOM_SoC_Platform.git
```
#### Step 2: Set Permissions
```bash
cd CUSTOM_SoC_Platform/Platform
chmod +x hw/run_hw.sh
```

### Running the Platform
#### Step 1: Hardware Synthesis and FPGA Bitstream Programming
- Connect the FPGA board to your machine by the USB port
- Run the following command to synthesize the hardware and program the bitstream into the FPGA board
```bash
make prog
```
- Make sure this step is done correctly before moving to the next step

#### Step 2: Running the Software

- Run the following command to compile the software application and initiate interaction with the FPGA
```
make run MODEL_FILE=ad01_int8.tflite MODEL_PROFILE=ad01
```
- When you see the line "Listening on /dev/ttyUSB1 for BIOS/App" on your terminal, press the CPU Reset button.
- After that, you should see the menu on your terminal that you can play around with.
- (More details on software to be added)

### Detail Reference on How to Use the Platform

See the documentation for [hardware](to be placed) and [software](to be placed)