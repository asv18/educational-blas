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