# Custom ML Accelerator Hardware

This directory contains the hardware side of the custom accelerator slot. The
current drop-in example is `srcs/NPU.v`, but the interface is intended for any
custom ML accelerator that uses the same CPU CUSTOM-0 and optional AXI master
ports.

## Interface Contract

The CPU issues R-type-like CUSTOM-0 instructions from software. The fields arrive
at the accelerator as:

```text
rs1_i      software operand rs1
rs2_i      software operand rs2
funct3_i   operation group, selected by cfu_op0..cfu_op7
funct7_i   operation subfunction
NPU_start  one-cycle request valid from the CPU pipeline
NPU_out    32-bit result returned to rd
NPU_done   accelerator response valid
NPU_exception  raise for unsupported or failed operations
```

The current SW mapping is:

```text
cfu_op0(funct7, rs1, rs2) -> funct3=0, compute/control group
cfu_op1(funct7, rs1, rs2) -> funct3=1, AXI read group in current example
cfu_op2(funct7, rs1, rs2) -> funct3=2, AXI write group in current example
cfu_op3..cfu_op7          -> reserved for user accelerators
```

`funct7_i` is the second selector. `srcs/NPU.v` currently implements
`funct3=0` as `rs1 + rs2 + funct7`, so the software example
`accel_scalar_compute(100, 200)` uses `funct7=5` and expects `305`.

## Add A Custom Instruction

1. Choose a `funct3/funct7` pair. Prefer putting the semantic wrapper in
   `Platform/sw/project/accel_ops.h`.
2. Implement the hardware behavior in `srcs/NPU.v`, or copy
   `templates/custom_accelerator_template.v` into `srcs/` and wire the Vivado
   block design to that module.
3. Add or update the software fallback in `Platform/sw/app/software_cfu.cc`.
   This lets `USE_SOFTWARE_CFU=1` and host syntax checks exercise the same API
   before hardware is ready.
4. Add a functional or performance test in `Platform/sw/project/user_menu.cc`
   or `Platform/sw/project/accel_tests.cc`. Use `perf_get_mcycle64()` for cycle
   measurements.
5. Run the host-side checks:

```sh
cd ../sw
make validate
make host-check APP_EXTRA_SRCS=templates/custom_instruction_template.cc
```

Then rebuild hardware in Vivado and rebuild firmware when the RISC-V toolchain is
available.

## AXI And Cache Notes

The AXI master ports are optional. A compute-only accelerator can keep all AXI
valid signals low, as shown in the template.

When the accelerator reads data that the CPU recently wrote to DRAM, call
`cbo_clean` or `cbo_clean_range` in software before starting the accelerator.
When the accelerator writes data that the CPU will read, call `cbo_invalidate` or
`cbo_invalidate_range` after the accelerator finishes.
