CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
LDFLAGS = 

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Main LSM tree sources (excluding generate_test_data.cpp)
MAIN_SRCS = $(filter-out $(SRC_DIR)/generate_test_data.cpp, $(wildcard $(SRC_DIR)/*.cpp))
MAIN_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(MAIN_SRCS))
MAIN_TARGET = $(BIN_DIR)/lsm_tree

# Test data generator sources
GEN_SRCS = $(SRC_DIR)/generate_test_data.cpp $(SRC_DIR)/utils.cpp
GEN_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(GEN_SRCS))
GEN_TARGET = $(BIN_DIR)/generate_test_data

# New data generator 
DATA_GEN_OBJS = $(OBJ_DIR)/utils.o 
DATA_GEN_TARGET = $(BIN_DIR)/data_generator

.PHONY: all clean

all: $(MAIN_TARGET) $(GEN_TARGET) $(DATA_GEN_TARGET)

$(MAIN_TARGET): $(MAIN_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(GEN_TARGET): $(GEN_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(DATA_GEN_TARGET): $(DATA_GEN_OBJS) data_generator.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) data_generator.cpp $(DATA_GEN_OBJS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

run: $(MAIN_TARGET)
	./$(MAIN_TARGET)

generate: $(GEN_TARGET)
	mkdir -p data
	./$(GEN_TARGET) data/test_data.bin 1000

generate-new: $(DATA_GEN_TARGET)
	mkdir -p data
	./$(DATA_GEN_TARGET) --count 2000000 --key-range 20000000 --output data/test_data.bin

test: all generate run 

test-new: all generate-new run 