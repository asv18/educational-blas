# AI Citation: Anthropic's Claude AI (Claude Sonnet 4.6, free tier, OTD 5/20/2026) \
 generated this makefile and made tweaks.
# Original Prompt: Generate a C++ Makefile with three targets: \
	1. all: compile nested C++ files in src/. \
	2. bench: compile and run C++ files in bench/ using Google's benchmarking suite \
	3. test: compile and run C++ files in tests/ using Google's testing suite
# Adjustment Prompt(s): \
	1. I brew-installed gtest and benchmark, so adjust my makefile to recognize that \
	2. I brew-installed libomp, so adjust my targets to accept libomp \
	3. I'm getting this error when running make bench: <insert error here>\n\n How do I fix this?
# Notes: \
	1. I don't generally like designing and writing Makefiles, and since the scope of this project wasn't \
	Makefile design, I saw it apt to generate one (with guidance).  \
	2. The prompt I put in "Original Prompt" is paraphrasing; I don't remember the exact prompt, \
	and I can't recover the original chat where I sent it unfortunately. \
	3. I don't usually cite AI, so I decided to give it a shot here. Any further suggestions are greatly appreciated!

# =========================================================
# Compiler
# =========================================================

CXX := /opt/homebrew/opt/llvm/bin/clang++

CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -fopenmp
CXXFLAGS += -isysroot $(shell xcrun --show-sdk-path)

CPPFLAGS := -Iinclude \
            -I$(shell brew --prefix libomp)/include

LDFLAGS := -L$(shell brew --prefix libomp)/lib
LDLIBS := -lomp

AR := ar
ARFLAGS := rcs

# =========================================================
# Directories
# =========================================================

SRC_DIR   := src
TEST_DIR  := tests
BENCH_DIR := bench
BUILD_DIR := build

# =========================================================
# Recursive source discovery
# =========================================================

SRC_FILES   := $(shell find $(SRC_DIR)   -name '*.cpp')
TEST_FILES  := $(shell find $(TEST_DIR)  -name '*.cpp')
BENCH_FILES := $(shell find $(BENCH_DIR) -name '*.cpp')

# =========================================================
# Object files
# =========================================================

SRC_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/src/%.o,$(SRC_FILES))

TEST_OBJS := $(patsubst $(TEST_DIR)/%.cpp,$(BUILD_DIR)/test/%.o,$(TEST_FILES))

BENCH_OBJS := $(patsubst $(BENCH_DIR)/%.cpp,$(BUILD_DIR)/bench/%.o,$(BENCH_FILES))

# =========================================================
# External libraries
# =========================================================

GTEST_CFLAGS := $(shell pkg-config --cflags gtest_main)
GTEST_LIBS   := $(shell pkg-config --libs gtest_main)

GBENCH_CFLAGS := $(shell pkg-config --cflags benchmark_main)
GBENCH_LIBS   := $(shell pkg-config --libs benchmark_main)

# =========================================================
# Targets
# =========================================================

LIB       := $(BUILD_DIR)/libblas.a
TEST_BIN  := $(BUILD_DIR)/run_tests
BENCH_BIN := $(BUILD_DIR)/run_benchmarks

# =========================================================
# Default target
# =========================================================

.PHONY: all
all: $(LIB)

# =========================================================
# Static library
# =========================================================

$(LIB): $(SRC_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $@ $^

# =========================================================
# Tests
# =========================================================

.PHONY: test
test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_OBJS) $(LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) $(LDLIBS) $(GTEST_LIBS) -o $@

# =========================================================
# Benchmarks
# =========================================================

.PHONY: bench
bench: $(BENCH_BIN)
	./$(BENCH_BIN)

.PHONY: bench_to_csv
bench_to_csv: $(BENCH_BIN)
	./$(BENCH_BIN) --benchmark_out_format=csv --benchmark_out=bench_results.csv

$(BENCH_BIN): $(BENCH_OBJS) $(LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) $(LDLIBS) $(GBENCH_LIBS) -o $@

# =========================================================
# Compile rules
# =========================================================

$(BUILD_DIR)/src/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(GTEST_CFLAGS) -c $< -o $@

$(BUILD_DIR)/bench/%.o: $(BENCH_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(GBENCH_CFLAGS) -c $< -o $@

# =========================================================
# Cleanup
# =========================================================

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

# =========================================================
# Convenience
# =========================================================

.PHONY: rebuild
rebuild: clean all