# Paths
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr/local/nccl

NVCC := $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= -gencode arch=compute_90,code=sm_90
BUILD_DIR := build
BENCH_SRC_DIR := bench

COMMON_INC := -I. \
              -I$(CUDA_HOME)/include \
              -I$(NCCL_HOME)/include

COMMON_LIBS := -L$(CUDA_HOME)/lib64 -lcudart \
               -L$(NCCL_HOME)/lib -lnccl

COMMON_RPATH := -Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64 \
                -Xlinker -rpath -Xlinker $(NCCL_HOME)/lib

CUDA_BENCH := $(BUILD_DIR)/cuda_bench
CUDA_NCCL_BENCH := $(BUILD_DIR)/cuda_nccl_bench
GDR_BENCH := $(BUILD_DIR)/gdr_bench
GDR_QOS_BENCH := $(BUILD_DIR)/gdr_qos_bench
NCCL_LATENCY_BENCH := $(BUILD_DIR)/nccl_latency_bench
TARGETS := $(CUDA_BENCH) $(CUDA_NCCL_BENCH) $(GDR_BENCH) $(GDR_QOS_BENCH) $(NCCL_LATENCY_BENCH)

CUDA_BENCH_SRC := $(BENCH_SRC_DIR)/cuda_bench.cu
GDR_BENCH_SRC := $(BENCH_SRC_DIR)/gdr_bench.cu
GDR_QOS_BENCH_SRC := $(BENCH_SRC_DIR)/gdr_qos_bench.cu
NCCL_LATENCY_BENCH_SRC := $(BENCH_SRC_DIR)/nccl_latency_bench.cu

all: $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

cuda_bench: $(CUDA_BENCH)

cuda_nccl_bench: $(CUDA_NCCL_BENCH)

gdr_bench: $(GDR_BENCH)

gdr_qos_bench: $(GDR_QOS_BENCH)

nccl_latency_bench: $(NCCL_LATENCY_BENCH)

$(CUDA_BENCH): $(CUDA_BENCH_SRC) | $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH) -O3 -std=c++17 \
	    $(COMMON_INC) \
	    -o $@ $< \
	    $(COMMON_LIBS) $(COMMON_RPATH)

$(CUDA_NCCL_BENCH): $(CUDA_BENCH_SRC) | $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH) -O3 -std=c++17 \
	    $(COMMON_INC) \
	    -o $@ $< \
	    $(COMMON_LIBS) $(COMMON_RPATH)

$(GDR_BENCH): $(GDR_BENCH_SRC) gdr/gdr_copy.cpp gdr/gdr_copy.h gdr/mr_cache.h | $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH) -O3 -std=c++17 \
	    $(COMMON_INC) -Igdr \
	    -o $@ $(GDR_BENCH_SRC) gdr/gdr_copy.cpp \
	    $(COMMON_LIBS) -libverbs -lpthread $(COMMON_RPATH)

$(GDR_QOS_BENCH): $(GDR_QOS_BENCH_SRC) $(GDR_BENCH_SRC) gdr/gdr_copy.cpp gdr/gdr_copy.h gdr/mr_cache.h | $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH) -O3 -std=c++17 \
	    $(COMMON_INC) -Igdr \
	    -o $@ $(GDR_QOS_BENCH_SRC) gdr/gdr_copy.cpp \
	    $(COMMON_LIBS) -libverbs -lpthread $(COMMON_RPATH)

$(NCCL_LATENCY_BENCH): $(NCCL_LATENCY_BENCH_SRC) | $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH) -O3 -std=c++17 \
	    $(COMMON_INC) \
	    -o $@ $< \
	    $(COMMON_LIBS) $(COMMON_RPATH)

clean:
	rm -rf $(BUILD_DIR) cuda_bench cuda_nccl_bench gdr_bench gdr_qos_bench nccl_latency_bench

.PHONY: all clean cuda_bench cuda_nccl_bench gdr_bench gdr_qos_bench nccl_latency_bench
