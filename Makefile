# Paths
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr/local/nccl

NVCC := $(CUDA_HOME)/bin/nvcc
CUDA_ARCH := -gencode arch=compute_90,code=sm_90

INC := -I$(CUDA_HOME)/include \
       -I$(NCCL_HOME)/include

LIBS := -L$(CUDA_HOME)/lib64 -lcudart \
        -L$(NCCL_HOME)/lib -lnccl

RPATH := -Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64 \
         -Xlinker -rpath -Xlinker $(NCCL_HOME)/lib

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
