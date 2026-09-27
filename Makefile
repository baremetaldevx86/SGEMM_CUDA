.PHONY: all build debug clean profile bench cuobjdump

CMAKE ?= cmake
CUOBJDUMP ?= cuobjdump
NCU ?= ncu

BUILD_DIR ?= build
BENCHMARK_DIR ?= benchmark_results
CMAKE_ARGS ?=
# Example: make CUDA_ARCHITECTURES='80;86'. Leave unset to use CMake's default
# (or the architecture already selected in the build directory's cache).
CUDA_ARCHITECTURES ?=
CMAKE_ARCH_ARGS = $(if $(strip $(CUDA_ARCHITECTURES)),-DCMAKE_CUDA_ARCHITECTURES="$(CUDA_ARCHITECTURES)")

all: build

build:
	@$(CMAKE) -S . -B "$(BUILD_DIR)" -DCMAKE_BUILD_TYPE=Release $(CMAKE_ARCH_ARGS) $(CMAKE_ARGS)
	@$(CMAKE) --build "$(BUILD_DIR)" --config Release

debug:
	@$(CMAKE) -S . -B "$(BUILD_DIR)" -DCMAKE_BUILD_TYPE=Debug $(CMAKE_ARCH_ARGS) $(CMAKE_ARGS)
	@$(CMAKE) --build "$(BUILD_DIR)" --config Debug

clean:
	@rm -rf "$(BUILD_DIR)"

# Dump all compiled architectures by default; optionally select one with
# make cuobjdump CUOBJDUMP_ARCH=sm_80 CUDA_ARCHITECTURES=80.
CUOBJDUMP_ARCH ?=
FUNCTION ?= $$($(CUOBJDUMP) --dump-elf-symbols "$(BUILD_DIR)/sgemm" | awk 'tolower($$0) ~ /warptiling/ {print $$NF}' | sort -u | paste -sd, -)

cuobjdump: build
	@$(CUOBJDUMP) $(if $(CUOBJDUMP_ARCH),--gpu-architecture $(CUOBJDUMP_ARCH)) --dump-sass --function "$(FUNCTION)" "$(BUILD_DIR)/sgemm" | c++filt > "$(BUILD_DIR)/cuobjdump.sass"
	@$(CUOBJDUMP) $(if $(CUOBJDUMP_ARCH),--gpu-architecture $(CUOBJDUMP_ARCH)) --dump-ptx --function "$(FUNCTION)" "$(BUILD_DIR)/sgemm" | c++filt > "$(BUILD_DIR)/cuobjdump.ptx"

# Usage: make profile KERNEL=<integer> PREFIX=<optional string>
profile: build
	@mkdir -p "$(BENCHMARK_DIR)"
	@$(NCU) --set full --export "$(BENCHMARK_DIR)/$(PREFIX)kernel_$(KERNEL)" --force-overwrite "$(BUILD_DIR)/sgemm" $(KERNEL)

bench: build
	@SGEMM_BIN="$(BUILD_DIR)/sgemm" bash gen_benchmark_results.sh
