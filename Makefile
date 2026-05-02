# =============================================================================
# Makefile – CUDA Image Processing Project
# =============================================================================

# Compiler
NVCC     := nvcc

# Target binary
TARGET   := image_processing

# Source files
SRCS     := src/image_processing.cu

# Compiler flags
NVCC_FLAGS := -O2 -std=c++14 -arch=sm_60 \
              -Xcompiler "-Wall -Wextra" \
              -lm

# Directories
BUILD_DIR  := build
OUTPUT_DIR := output
INPUT_DIR  := input

# Default target
.PHONY: all
all: $(BUILD_DIR)/$(TARGET)

# Build binary
$(BUILD_DIR)/$(TARGET): $(SRCS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^
	@echo "Build successful: $@"

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Create output directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Run on a single image (usage: make run INPUT=input/sample.ppm)
.PHONY: run
run: $(BUILD_DIR)/$(TARGET) $(OUTPUT_DIR)
	@if [ -z "$(INPUT)" ]; then \
	  echo "Usage: make run INPUT=<path_to_ppm>"; exit 1; \
	fi
	./$(BUILD_DIR)/$(TARGET) $(INPUT) $(OUTPUT_DIR)

# Run on all PPM images in the input directory
.PHONY: run-dir
run-dir: $(BUILD_DIR)/$(TARGET) $(OUTPUT_DIR)
	./$(BUILD_DIR)/$(TARGET) --dir $(INPUT_DIR) $(OUTPUT_DIR)

# Generate a synthetic test image and run the full pipeline
.PHONY: test
test: $(BUILD_DIR)/$(TARGET) $(OUTPUT_DIR)
	python3 scripts/gen_test_image.py input/test_512.ppm 512 512
	./$(BUILD_DIR)/$(TARGET) input/test_512.ppm $(OUTPUT_DIR)
	@echo "Test complete. Check $(OUTPUT_DIR)/ for results."

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

# Clean everything including output
.PHONY: distclean
distclean: clean
	rm -rf $(OUTPUT_DIR)

# Print help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all       – Build the binary (default)"
	@echo "  run       – Run on a single image:  make run INPUT=input/sample.ppm"
	@echo "  run-dir   – Run on all PPMs in input/"
	@echo "  test      – Generate a test image and run the pipeline"
	@echo "  clean     – Remove build artifacts"
	@echo "  distclean – Remove build artifacts and output"
