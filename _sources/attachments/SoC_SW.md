# CUSTOM SoC Software

## Quick Start

```sh
make help
make config
make models
make profiles
make templates
make validate
make check-env
make MODEL_FILE=ad01_int8.tflite
make run MODEL_FILE=ad01_int8.tflite
```

Useful build overrides:

```sh
MODEL_DIR=models
MODEL_FILE=ad01_int8.tflite
MODEL_PROFILE=ad01
TENSOR_ARENA_SIZE=262144
USE_SOFTWARE_CFU=1
TARGET_PREFIX=riscv64-unknown-elf
APP_DEFINES="-DMY_FLAG=1"
APP_EXTRA_SRCS="project/my_extra_file.cc"
```

For repeated local settings, copy `project.mk.example` to `project.mk` and edit
that file. `project.mk` is ignored by git, and command-line variables still
override it:

```sh
cp project.mk.example project.mk
make config
make
make MODEL_FILE=ad01_int8.tflite MODEL_PROFILE=ad01
```

`make config` prints the resolved build settings, including the selected model
profile source.

`make validate` prints the resolved build settings and lists available models,
profiles, and templates. It does not build or require the RISC-V embedded
toolchain.

## File Layout

`project/proj_menu.*` owns the built-in project menu entries for accelerator
tests, TFLM inference, and the user extension menu.

`project/user_menu.*` is the intended first place to add a new project-specific
test, demo, or experiment.

`app/cfu.*`, `app/software_cfu.*`, and `project/accel_ops.h` wrap CUSTOM-0
access. Use `cfu_op0..cfu_op7(funct7, rs1, rs2)` for raw CUSTOM-0 fields, or
add semantic helpers in `accel_ops.h`.

## Adding A Menu Item

Edit `project/user_menu.cc`:

```c++
void do_my_test(void) {
  // Project-specific test code.
}

const MenuItem kUserItems[] = {
    MENU_ITEM('m', "my test", do_my_test),
    MENU_END,
};
```

Rebuild and open the firmware menu:

1. Select `1` for Project menu.
2. Select `u` for User extension menu.
3. Select your new entry.

For a larger test, start from `templates/app_extension_template.cc`, add the new
file through `APP_EXTRA_SRCS`, and register its function in
`project/user_menu.cc`.

## Adding A Custom Instruction Helper

1. Choose a `funct3/funct7` pair. `cfu_op0..cfu_op7` select `funct3=0..7`;
   their first argument is `funct7`.
2. Implement the hardware behavior in `../hw/srcs/NPU.v`, or start from
   `../hw/templates/custom_accelerator_template.v`.
3. Implement the software fallback in `app/software_cfu.cc`.
4. Add a readable wrapper in `project/accel_ops.h`.
5. Add a functional or cycle-counting test entry in `project/user_menu.cc` or
   `project/accel_tests.cc`.

Build with `USE_SOFTWARE_CFU=1` when you want to test the software fallback
without issuing CUSTOM-0 instructions.

For raw hardware calls, keep `funct3` and `funct7` as compile-time numeric
literals or preprocessor macros, for example:

```c++
#define MY_ACCEL_FUNCT7 6

static inline uint32_t accel_my_op(uint32_t a, uint32_t b) {
  return cfu_op0(MY_ACCEL_FUNCT7, a, b);
}
```

Use `templates/custom_instruction_template.cc` for a standalone performance
test skeleton. After adding template-based code, run `make validate`; if the
extension adds a new menu item, register its function in `project/user_menu.cc`.

## Running Performance Tests

The firmware menu has three built-in performance paths:

1. Main menu `2` runs low-level performance counter checks.
2. Main menu `1`, then project menu `b`, runs the scalar custom-instruction
   benchmark through `project/accel_ops.h`.
3. Main menu `1`, then project menu `3`, runs the selected TFLM model and prints
   inference cycles and time.

## Adding Or Switching Models

List bundled models:

```sh
make models
make profiles
make templates
```

Bare `MODEL_FILE` values are resolved under `MODEL_DIR`, which defaults to
`models`.

Bundled profiles:

```text
ad01          ad01_int8.tflite fixture input and golden output
vww_96        vww_96_int8.tflite zero input, output verification skipped
generic_zero  fallback for new models while bring-up data is not ready
```

Build a specific model:

```sh
make MODEL_FILE=ad01_int8.tflite
```

Select a profile explicitly:

```sh
make MODEL_FILE=my_model.tflite MODEL_PROFILE=generic_zero
make MODEL_FILE=ad01_int8.tflite MODEL_PROFILE=ad01
```

When adding a new `.tflite` file:

1. Copy it into `Platform/sw/models`.
2. Register any missing kernels in `project/tflm_ops.cc`.
3. Start with `MODEL_PROFILE=generic_zero` if fixture data is not ready.
4. Create `models/<profile>_profile.cc` when the model needs fixture input,
   generated input, custom tensor handling, or golden-output verification. Use
   `templates/model_profile_template.cc` as the starting pattern.
5. Implement `const ModelProfile* model_profile_get(void)`.
6. For simple int8 models, fill `input_data` and `expected_output`.
7. For custom tensor types, generated inputs, or multi-output checks, set the
   `prepare_input` or `verify_output` function pointers.
8. Set `TENSOR_ARENA_SIZE` if the model needs a larger arena.
9. Run `make validate`, then build with `make MODEL_FILE=<name>.tflite`.

To make a new model auto-select its profile, add a `MODEL_BASENAME` mapping in
the Makefile.

## Generated Files

The Makefile generates:

```text
../build/sw/generated/model_data.h
../build/sw/model_data.c
../build/sw/main.elf
../build/sw/main.bin
```

Do not edit generated files directly. Change `MODEL_FILE`, source files, or
Makefile variables instead.
