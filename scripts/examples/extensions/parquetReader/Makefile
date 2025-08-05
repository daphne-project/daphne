# Paths
DAPHNE_ROOT   := /home/yazan/daphneFork/daphne
TP_INC_DIR    := $(DAPHNE_ROOT)/thirdparty/installed/include
TP_LIB_DIR    := $(DAPHNE_ROOT)/thirdparty/installed/lib
DS_LIB_DIR    := $(DAPHNE_ROOT)/build/src/runtime/local/datastructures

# Compiler & flags
CXX           := g++
CXXFLAGS      := -std=c++17 -O3 -mavx2 -fPIC  \
                  -I$(DAPHNE_ROOT)/src \
                  -I$(TP_INC_DIR) \
                  -I$(DAPHNE_ROOT)/src/runtime/local/io \
                  -DSPDLOG_FMT_EXTERNAL 

# Linker & libraries
LDFLAGS       := -shared -Wl,-rpath=$(TP_LIB_DIR)
LDLIBS        := \
    -L$(DS_LIB_DIR)    -lDataStructures \
    -L$(TP_LIB_DIR)    -larrow -lparquet -lMLIRIR -lMLIRSupport -lLLVMOption -lLLVMSupport -lLLVMDemangle -lLLVMCore -lspdlog -lfmt      

# Source files
SRC           := parquetPlugIn.cpp
OBJ           := $(SRC:.cpp=.o)
LIB_NAME      := libparquetPlugIn.so

.PHONY: all clean

all: $(LIB_NAME)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LIB_NAME): $(OBJ)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS)

clean:
	rm -f $(OBJ) $(LIB_NAME)
