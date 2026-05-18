# Compiler & flags
CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2
INCLUDES := -Iinclude

# Directories
SRC_DIR   := src
TEST_DIR  := tests
BENCH_DIR := bench
BUILD_DIR := build

# Sources
SRC_FILES   := $(wildcard $(SRC_DIR)/level_*.cpp)
TEST_FILES  := $(wildcard $(TEST_DIR)/test_*.cpp)
BENCH_FILES := $(wildcard $(BENCH_DIR)/benchmark_*.cpp)

GTEST_CFLAGS := $(shell pkg-config --cflags gtest_main)
GTEST_LIBS   := $(shell pkg-config --libs   gtest_main)

# Objects
SRC_OBJS  := $(patsubst $(SRC_DIR)/%.cpp,   $(BUILD_DIR)/src/%.o,   $(SRC_FILES))
TEST_OBJS := $(patsubst $(TEST_DIR)/%.cpp,  $(BUILD_DIR)/test/%.o,  $(TEST_FILES))
BENCH_OBJS:= $(patsubst $(BENCH_DIR)/%.cpp, $(BUILD_DIR)/bench/%.o, $(BENCH_FILES))

# Targets
LIB        := $(BUILD_DIR)/libblas.a
TEST_BIN   := $(BUILD_DIR)/run_tests
BENCH_BIN  := $(BUILD_DIR)/run_benchmarks

# Default
.PHONY: all
all: clean $(LIB)

# Static library (core)
$(LIB): $(SRC_OBJS) | $(BUILD_DIR)
	ar rcs $@ $^

# Tests
.PHONY: test
test: clean $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_OBJS) $(LIB) | $(BUILD_DIR)/test
	$(CXX) $(CXXFLAGS) $^ $(GTEST_LIBS) -o $@

# Benchmarks
.PHONY: bench
bench: clean $(BENCH_BIN)
	./$(BENCH_BIN)

$(BENCH_BIN): $(BENCH_OBJS) $(LIB) | $(BUILD_DIR)/bench
	$(CXX) $(CXXFLAGS) $^ -o $@

# Compile rules
$(BUILD_DIR)/src/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)/src
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/test/%.o: $(TEST_DIR)/%.cpp | $(BUILD_DIR)/test
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GTEST_CFLAGS) -c $< -o $@

$(BUILD_DIR)/bench/%.o: $(BENCH_DIR)/%.cpp | $(BUILD_DIR)/bench
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Directory creation
$(BUILD_DIR) $(BUILD_DIR)/src $(BUILD_DIR)/test $(BUILD_DIR)/bench:
	mkdir -p $@

# Cleanup
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)