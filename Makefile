# Paths
# MPI_HOME can also be set from: $(shell mpicc --showme:prefix)
# NCCL_HOME can also be provided via the environment.
CUDA_HOME ?= /usr/local/cuda
MPI_HOME ?= /usr/lib/x86_64-linux-gnu/openmpi
NCCL_HOME ?= /usr/local/nccl

NVCC := $(CUDA_HOME)/bin/nvcc
CUDA_ARCH := -gencode arch=compute_90,code=sm_90

INC := -I$(CUDA_HOME)/include \
       -I$(MPI_HOME)/include \
       -I$(NCCL_HOME)/include

LIBS := -L$(CUDA_HOME)/lib64 -lcudart \
        -L$(MPI_HOME)/lib -lmpi \
        -L$(NCCL_HOME)/lib -lnccl

RPATH := -Xlinker -rpath -Xlinker $(NCCL_HOME)/lib

TARGET := bench

all: $(TARGET)

$(TARGET): bench.cu
	$(NVCC) $(CUDA_ARCH) -O3 -std=c++17 \
	    $(INC) \
	    -o $@ $< \
	    $(LIBS) $(RPATH)

clean:
	rm -f $(TARGET)

.PHONY: all clean
