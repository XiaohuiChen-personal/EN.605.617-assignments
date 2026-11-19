# Prefer ./build.sh. It checks that an OpenCL header is present.
CC = gcc
CFLAGS = -std=c11 -O2 -Wall -Wextra -DCL_TARGET_OPENCL_VERSION=300
LIBS = -lOpenCL -lm
INC =

ifeq ($(wildcard /usr/include/CL/cl.h),)
  ifneq ($(wildcard $(HOME)/.local/include/CL/cl.h),)
    INC = -I$(HOME)/.local/include
    LIBS = -L$(HOME)/.local/lib -lOpenCL -lm
  endif
endif

assignment.exe: assignment.c cl_check.h words.cl
	$(CC) $(CFLAGS) $(INC) -o $@ assignment.c $(LIBS)

clean:
	rm -f assignment.exe

.PHONY: clean
