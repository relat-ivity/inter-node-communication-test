# ── paths (adjust if your environment differs) ───────────────────────────────
CUDA_HOME  ?= /usr/local/cuda
MPI_HOME   ?= /usr/lib/x86_64-linux-gnu/openmpi   # or $(shell mpicc --showme:prefix)
NCCL_HOME  ?= /usr/local/nccl                      # or set via $NCCL_HOME

NVCC       := $(CUDA_HOME)/bin/nvcc
CUDA_ARCH  := -gencode arch=compute_90,code=sm_90  # H20 = Hopper sm_90

# ── includes & libs ──────────────────────────────────────────────────────────
INC := -I$(CUDA_HOME)/include \
       -I$(MPI_HOME)/include  \
       -I$(NCCL_HOME)/include

LDFLAGS := -L$(CUDA_HOME)/lib64   -lcudart          \
           -L$(MPI_HOME)/lib      -lmpi              \
           -L$(NCCL_HOME)/lib     -lnccl             \
           -Wl,-rpath,$(NCCL_HOME)/lib

# ── build ────────────────────────────────────────────────────────────────────
TARGET := bench

all: $(TARGET)

$(TARGET): bench.cu
	$(NVCC) $(CUDA_ARCH) -O3 -std=c++17 \
	    $(INC) $(LDFLAGS) \
	    -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
