# Module 5 assignment. `make` with no arguments builds assignment.exe.

NVCC     ?= nvcc
ARCH     ?= -arch=native
CXXFLAGS ?= -O2 -std=c++17
TARGET    = assignment.exe

.PHONY: all clean

all: $(TARGET)

$(TARGET): assignment.cu cuda_check.h
	$(NVCC) $(ARCH) $(CXXFLAGS) -o $(TARGET) assignment.cu

clean:
	rm -f $(TARGET)
