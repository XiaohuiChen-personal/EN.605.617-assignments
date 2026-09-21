# Module 3 assignment. `make` with no arguments builds assignment.exe.
# The course starter compiled assignment.c with gcc; this assignment is
# CUDA, so the compiler is nvcc. Override ARCH if you are not on the
# machine that will run the binary (DGX Spark: -arch=native is sm_121).

NVCC     ?= nvcc
ARCH     ?= -arch=native
CXXFLAGS ?= -O2 -std=c++17
TARGET    = assignment.exe

.PHONY: all clean

all: $(TARGET)

$(TARGET): assignment.cu cuda_check.h
	$(NVCC) $(ARCH) $(CXXFLAGS) -o $(TARGET) assignment.cu

clean:
	rm -f $(TARGET) results.csv
