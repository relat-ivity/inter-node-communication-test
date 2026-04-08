# Paths
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr/local/nccl

NVCC := $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= -gencode arch=compute_90,code=sm_90
BUILD_DIR := build

COMMON_INC := -I$(CUDA_HOME)/include \
              -I$(NCCL_HOME)/include

COMMON_LIBS := -L$(CUDA_HOME)/lib64 -lcudart \
               -L$(NCCL_HOME)/lib -lnccl

COMMON_RPATH := -Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64 \
                -Xlinker -rpath -Xlinker $(NCCL_HOME)/lib

CUDA_BENCH := $(BUILD_DIR)/cuda_bench
GDR_BENCH := $(BUILD_DIR)/gdr_bench
TARGETS := $(CUDA_BENCH) $(GDR_BENCH)

all: $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

cuda_bench: $(CUDA_BENCH)

gdr_bench: $(GDR_BENCH)

$(CUDA_BENCH): cuda_bench.cu | $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH) -O3 -std=c++17 \
	    $(COMMON_INC) \
	    -o $@ $< \
	    $(COMMON_LIBS) $(COMMON_RPATH)

$(GDR_BENCH): gdr_bench.cu gdr/gdr_copy.cpp gdr/gdr_copy.h gdr/mr_cache.h gdr/pinned_pool.h | $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH) -O3 -std=c++17 \
	    $(COMMON_INC) -Igdr \
	    -o $@ gdr_bench.cu gdr/gdr_copy.cpp \
	    $(COMMON_LIBS) -libverbs -lpthread $(COMMON_RPATH)

clean:
	rm -rf $(BUILD_DIR) bench cuda_bench gdr_bench

.PHONY: all clean cuda_bench gdr_bench
