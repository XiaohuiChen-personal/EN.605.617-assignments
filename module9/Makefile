# Module 9 assignment. `./build.sh` builds assignment.exe.
# `make` needs CUDNN_ROOT (the directory with include/cudnn.h and lib/libcudnn.so.9).
# CUDNN_ROOT is the directory that contains include/cudnn.h and lib/libcudnn.so.9.
# build.sh fills it in when the wheel from Module 9 is on this machine.

NVCC       ?= nvcc
ARCH       ?= -arch=native
CXXFLAGS   ?= -O2 -std=c++17
CUDNN_ROOT ?=
TARGET      = assignment.exe

ifeq ($(CUDNN_ROOT),)
$(error CUDNN_ROOT is empty; run ./build.sh or pass CUDNN_ROOT=...)
endif

.PHONY: all clean

all: $(TARGET)

$(TARGET): assignment.cu cuda_check.h
	$(NVCC) $(ARCH) $(CXXFLAGS) -I$(CUDNN_ROOT)/include -o $(TARGET) assignment.cu \
	    -L$(CUDNN_ROOT)/lib -Xlinker -rpath=$(CUDNN_ROOT)/lib -l:libcudnn.so.9 -lcufft

clean:
	rm -f $(TARGET)
