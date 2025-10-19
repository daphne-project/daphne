# Makefile for gen_data and gen_meta (Arrow/Parquet C++)
# Adjust DAPHNE_ROOT if needed

DAPHNE_ROOT   := /home/yazan/daphneFork/daphne
TP_INC_DIR    := $(DAPHNE_ROOT)/thirdparty/installed/include
TP_LIB_DIR    := $(DAPHNE_ROOT)/thirdparty/installed/lib

CXX           := g++
CXXFLAGS      := -std=c++17 -O3 -fPIC -Wall -Wextra \
                  -I$(TP_INC_DIR)
LDFLAGS       := -L$(TP_LIB_DIR) -Wl,-rpath,$(TP_LIB_DIR)
# Common Arrow/Parquet dependencies; add -lzstd/-llz4 here if your Arrow build needs them
LDLIBS        := -lparquet -larrow \
                 -lz -lbz2 -pthread

# Binaries
BINS := gen_data gen_meta

all: $(BINS)

# ----------------------------------------------------------------------
# gen_data
# ----------------------------------------------------------------------
gen_data: gen_data.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

gen_data.o: gen_data.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ----------------------------------------------------------------------
# gen_meta
# ----------------------------------------------------------------------
gen_meta: gen_meta.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

gen_meta.o: gen_meta.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ----------------------------------------------------------------------
# housekeeping
# ----------------------------------------------------------------------
.PHONY: clean
clean:
	rm -f $(BINS) gen_data.o gen_meta.o
