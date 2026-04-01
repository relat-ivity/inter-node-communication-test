/*
 * Cross-node H2D / D2H / D2D (ReduceScatter) Interference Benchmark
 * ==================================================================
 * Measures whether cudaMemcpy (H2D or D2H) and NCCL ReduceScatter
 * degrade each other's bandwidth when run concurrently on two H20 nodes.
 *
 * Build:  see Makefile
 * Run:    mpirun -np 2 -H node0,node1 ./bench  (one process per node)
 *
 * Env vars:
 *   BENCH_BUF_GB       D2H/H2D buffer size   (default 4 GB)
 *   BENCH_RS_BUF_GB    ReduceScatter buffer   (default 2 GB)
 *   BENCH_ITERS        iterations per phase   (default 20)
 *   BENCH_WARMUP       warmup iterations      (default 5)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

// ── error checking ───────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA] %s:%d  %s\n",                             \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

#define NCCL_CHECK(call)                                                       \
    do {                                                                       \
        ncclResult_t _r = (call);                                              \
        if (_r != ncclSuccess) {                                               \
            fprintf(stderr, "[NCCL] %s:%d  %s\n",                             \
                    __FILE__, __LINE__, ncclGetErrorString(_r));               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

#define MPI_CHECK(call)                                                        \
    do {                                                                       \
        int _r = (call);                                                       \
        if (_r != MPI_SUCCESS) {                                               \
            char buf[256]; int len;                                            \
            MPI_Error_string(_r, buf, &len);                                   \
            fprintf(stderr, "[MPI] %s:%d  %s\n", __FILE__, __LINE__, buf);    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// ── config ───────────────────────────────────────────────────────────────────

static double env_double(const char *name, double def) {
    const char *v = getenv(name);
    return v ? atof(v) : def;
}
static int env_int(const char *name, int def) {
    const char *v = getenv(name);
    return v ? atoi(v) : def;
}

// ── timing helpers ───────────────────────────────────────────────────────────

struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin(cudaStream_t s) { CUDA_CHECK(cudaEventRecord(start, s)); }
    void end(cudaStream_t s)   { CUDA_CHECK(cudaEventRecord(stop,  s)); }
    // blocks until stop is recorded
    float elapsed_ms() {
        float ms = 0;
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ── statistics ───────────────────────────────────────────────────────────────

struct Stats {
    std::string label;
    double bw_gb_s;
    double avg_ms;
    double p99_ms;
    double std_ms;
};

static Stats compute_stats(const std::string &label,
                            std::vector<float> &lats_ms,
                            size_t bytes_per_iter)
{
    std::sort(lats_ms.begin(), lats_ms.end());
    double sum = 0;
    for (float v : lats_ms) sum += v;
    double avg = sum / lats_ms.size();

    double var = 0;
    for (float v : lats_ms) var += (v - avg) * (v - avg);
    double std = sqrt(var / lats_ms.size());

    size_t p99_idx = (size_t)(0.99 * lats_ms.size());
    double p99 = lats_ms[p99_idx];

    double total_gb = (double)bytes_per_iter * lats_ms.size() / 1e9;
    double total_s  = sum / 1e3;
    double bw = total_gb / total_s;

    return {label, bw, avg, p99, std};
}

static void print_stats(const Stats &s) {
    printf("  %-38s  BW=%7.2f GB/s  avg=%8.3f ms  p99=%8.3f ms  std=%7.3f ms\n",
           s.label.c_str(), s.bw_gb_s, s.avg_ms, s.p99_ms, s.std_ms);
}

static Stats summarize_stats_across_ranks(const Stats &local, int rank, int nranks)
{
    double local_vals[4] = {
        local.bw_gb_s,
        local.avg_ms,
        local.p99_ms,
        local.std_ms,
    };

    std::vector<double> gathered;
    if (rank == 0) gathered.resize(nranks * 4);

    MPI_CHECK(MPI_Gather(local_vals, 4, MPI_DOUBLE,
                         rank == 0 ? gathered.data() : nullptr,
                         4, MPI_DOUBLE, 0, MPI_COMM_WORLD));

    if (rank != 0) return local;

    double min_bw  = gathered[0];
    double max_avg = gathered[1];
    double max_p99 = gathered[2];
    double max_std = gathered[3];

    for (int r = 0; r < nranks; r++) {
        const double *vals = &gathered[r * 4];
        printf("    rank %-2d %-28s  BW=%7.2f GB/s  avg=%8.3f ms  p99=%8.3f ms  std=%7.3f ms\n",
               r, local.label.c_str(), vals[0], vals[1], vals[2], vals[3]);
        min_bw  = std::min(min_bw,  vals[0]);
        max_avg = std::max(max_avg, vals[1]);
        max_p99 = std::max(max_p99, vals[2]);
        max_std = std::max(max_std, vals[3]);
    }

    return {local.label + " [worst rank]", min_bw, max_avg, max_p99, max_std};
}

static void print_delta(const char *name,
                        const Stats &solo, const Stats &conc)
{
    double bw_delta  = (conc.bw_gb_s - solo.bw_gb_s) / solo.bw_gb_s * 100.0;
    double lat_delta = (conc.avg_ms  - solo.avg_ms)   / solo.avg_ms  * 100.0;
    printf("  %-10s  BW: %+6.1f%%   avg-lat: %+6.1f%%", name, bw_delta, lat_delta);
    bool bad = (bw_delta < -5.0) || (lat_delta > 5.0);
    printf("  %s\n", bad ? "<-- INTERFERENCE" : "OK");
}

// ── benchmark primitives ─────────────────────────────────────────────────────

// Solo D2H
static Stats bench_d2h(float *d_buf, float *h_buf, size_t n_bytes,
                        cudaStream_t stream, int iters, int warmup)
{
    CudaTimer timer;
    for (int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaMemcpyAsync(h_buf, d_buf, n_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::vector<float> lats;
    lats.reserve(iters);
    for (int i = 0; i < iters; i++) {
        timer.begin(stream);
        CUDA_CHECK(cudaMemcpyAsync(h_buf, d_buf, n_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        timer.end(stream);
        lats.push_back(timer.elapsed_ms());   // syncs internally
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    return compute_stats("D2H (solo)", lats, n_bytes);
}

// Solo H2D
static Stats bench_h2d(float *d_buf, float *h_buf, size_t n_bytes,
                        cudaStream_t stream, int iters, int warmup)
{
    CudaTimer timer;
    for (int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d_buf, h_buf, n_bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::vector<float> lats;
    lats.reserve(iters);
    for (int i = 0; i < iters; i++) {
        timer.begin(stream);
        CUDA_CHECK(cudaMemcpyAsync(d_buf, h_buf, n_bytes,
                                   cudaMemcpyHostToDevice, stream));
        timer.end(stream);
        lats.push_back(timer.elapsed_ms());
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    return compute_stats("H2D (solo)", lats, n_bytes);
}

// Solo ReduceScatter
// send_buf size = rs_bytes, recv_buf size = rs_bytes / nranks
static Stats bench_rs(float *send_buf, float *recv_buf,
                       size_t rs_bytes, int nranks,
                       ncclComm_t comm, cudaStream_t stream,
                       int iters, int warmup)
{
    size_t recv_count = rs_bytes / sizeof(float) / nranks;
    CudaTimer timer;

    for (int i = 0; i < warmup; i++) {
        NCCL_CHECK(ncclReduceScatter(send_buf, recv_buf, recv_count,
                                      ncclFloat, ncclSum, comm, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
    std::vector<float> lats;
    lats.reserve(iters);
    for (int i = 0; i < iters; i++) {
        timer.begin(stream);
        NCCL_CHECK(ncclReduceScatter(send_buf, recv_buf, recv_count,
                                      ncclFloat, ncclSum, comm, stream));
        timer.end(stream);
        float ms = timer.elapsed_ms();  // syncs stream
        // Barrier so both sides finish before next iter
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        lats.push_back(ms);
    }
    // ReduceScatter bus BW = (N-1)/N * size / time  (N=2 → 0.5x)
    // We report the raw byte count moved per rank for fair comparison
    return compute_stats("ReduceScatter (solo)", lats, rs_bytes);
}

// ── concurrent benchmark ─────────────────────────────────────────────────────
/*
 * We kick off ReduceScatter on stream_rs and cudaMemcpy on stream_mem
 * WITHOUT synchronising between them, then collect both timings via
 * two independent CUDA event pairs.  This is the only correct way to
 * measure overlap on a single thread — no OS threads needed and no GIL.
 */
struct ConcurrentResult {
    Stats mem_stats;
    Stats rs_stats;
};

static ConcurrentResult bench_concurrent_d2h_rs(
        float *d_buf, float *h_buf, size_t mem_bytes,
        float *send_buf, float *recv_buf, size_t rs_bytes, int nranks,
        ncclComm_t comm,
        cudaStream_t stream_mem, cudaStream_t stream_rs,
        int iters, int warmup,
        const char *mem_label)
{
    size_t recv_count = rs_bytes / sizeof(float) / nranks;
    CudaTimer timer_mem, timer_rs;

    auto do_memcpy = [&](cudaStream_t s) {
        CUDA_CHECK(cudaMemcpyAsync(h_buf, d_buf, mem_bytes,
                                   cudaMemcpyDeviceToHost, s));
    };

    // warmup
    for (int i = 0; i < warmup; i++) {
        do_memcpy(stream_mem);
        NCCL_CHECK(ncclReduceScatter(send_buf, recv_buf, recv_count,
                                      ncclFloat, ncclSum, comm, stream_rs));
        CUDA_CHECK(cudaStreamSynchronize(stream_mem));
        CUDA_CHECK(cudaStreamSynchronize(stream_rs));
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    std::vector<float> lats_mem, lats_rs;
    lats_mem.reserve(iters);
    lats_rs.reserve(iters);

    for (int i = 0; i < iters; i++) {
        // launch both operations without stalling between them
        timer_mem.begin(stream_mem);
        do_memcpy(stream_mem);
        timer_mem.end(stream_mem);

        timer_rs.begin(stream_rs);
        NCCL_CHECK(ncclReduceScatter(send_buf, recv_buf, recv_count,
                                      ncclFloat, ncclSum, comm, stream_rs));
        timer_rs.end(stream_rs);

        // collect timings (each call syncs its own stream)
        lats_mem.push_back(timer_mem.elapsed_ms());
        lats_rs.push_back(timer_rs.elapsed_ms());

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    std::string mem_lbl = std::string(mem_label) + " (concurrent)";
    return {
        compute_stats(mem_lbl,                    lats_mem, mem_bytes),
        compute_stats("ReduceScatter (concurrent)", lats_rs,  rs_bytes),
    };
}

// H2D variant
static ConcurrentResult bench_concurrent_h2d_rs(
        float *d_buf, float *h_buf, size_t mem_bytes,
        float *send_buf, float *recv_buf, size_t rs_bytes, int nranks,
        ncclComm_t comm,
        cudaStream_t stream_mem, cudaStream_t stream_rs,
        int iters, int warmup)
{
    size_t recv_count = rs_bytes / sizeof(float) / nranks;
    CudaTimer timer_mem, timer_rs;

    for (int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d_buf, h_buf, mem_bytes,
                                   cudaMemcpyHostToDevice, stream_mem));
        NCCL_CHECK(ncclReduceScatter(send_buf, recv_buf, recv_count,
                                      ncclFloat, ncclSum, comm, stream_rs));
        CUDA_CHECK(cudaStreamSynchronize(stream_mem));
        CUDA_CHECK(cudaStreamSynchronize(stream_rs));
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    std::vector<float> lats_mem, lats_rs;
    lats_mem.reserve(iters);
    lats_rs.reserve(iters);

    for (int i = 0; i < iters; i++) {
        timer_mem.begin(stream_mem);
        CUDA_CHECK(cudaMemcpyAsync(d_buf, h_buf, mem_bytes,
                                   cudaMemcpyHostToDevice, stream_mem));
        timer_mem.end(stream_mem);

        timer_rs.begin(stream_rs);
        NCCL_CHECK(ncclReduceScatter(send_buf, recv_buf, recv_count,
                                      ncclFloat, ncclSum, comm, stream_rs));
        timer_rs.end(stream_rs);

        lats_mem.push_back(timer_mem.elapsed_ms());
        lats_rs.push_back(timer_rs.elapsed_ms());
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    return {
        compute_stats("H2D (concurrent)",           lats_mem, mem_bytes),
        compute_stats("ReduceScatter (concurrent)", lats_rs,  rs_bytes),
    };
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank, nranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    if (nranks != 2) {
        if (rank == 0) fprintf(stderr, "This benchmark requires exactly 2 MPI ranks.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Config
    double buf_gb    = env_double("BENCH_BUF_GB",    4.0);
    double rs_buf_gb = env_double("BENCH_RS_BUF_GB", 2.0);
    int    iters     = env_int   ("BENCH_ITERS",     20);
    int    warmup    = env_int   ("BENCH_WARMUP",    5);
    int    gpu_id    = env_int   ("BENCH_GPU_ID",    0);

    size_t mem_bytes = (size_t)(buf_gb    * 1e9);
    size_t rs_bytes  = (size_t)(rs_buf_gb * 1e9);
    // round down to float and nranks alignment
    mem_bytes = (mem_bytes / sizeof(float))             * sizeof(float);
    rs_bytes  = (rs_bytes  / sizeof(float) / nranks)   * sizeof(float) * nranks;

    // GPU setup
    int n_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
    if (gpu_id < 0 || gpu_id >= n_gpus) {
        fprintf(stderr, "[rank %d] BENCH_GPU_ID=%d out of range (0..%d)\n",
                rank, gpu_id, n_gpus - 1);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(gpu_id));

    char gpu_name[256];
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_id));
    strncpy(gpu_name, prop.name, sizeof(gpu_name));

    // NIC: read NCCL_IB_HCA for display (NCCL picks it up automatically from env)
    const char *ib_hca = getenv("NCCL_IB_HCA");
    if (!ib_hca) ib_hca = "(auto)";

    // NCCL communicator
    ncclUniqueId nccl_id;
    if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    MPI_CHECK(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, nccl_id, rank));

    // CUDA streams (separate for mem and collective)
    cudaStream_t stream_mem, stream_rs;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_mem, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_rs,  cudaStreamNonBlocking));

    // Allocate device buffers
    float *d_mem_buf = nullptr;   // for H2D / D2H
    float *d_rs_send = nullptr;   // ReduceScatter input
    float *d_rs_recv = nullptr;   // ReduceScatter output

    CUDA_CHECK(cudaMalloc(&d_mem_buf, mem_bytes));
    CUDA_CHECK(cudaMalloc(&d_rs_send, rs_bytes));
    CUDA_CHECK(cudaMalloc(&d_rs_recv, rs_bytes / nranks));
    CUDA_CHECK(cudaMemset(d_mem_buf, 1, mem_bytes));
    CUDA_CHECK(cudaMemset(d_rs_send, 1, rs_bytes));
    CUDA_CHECK(cudaMemset(d_rs_recv, 0, rs_bytes / nranks));

    // Allocate pinned host buffer
    float *h_buf = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_buf, mem_bytes));
    memset(h_buf, 1, mem_bytes);

    if (rank == 0) {
        printf("\n");
        printf("============================================================\n");
        printf("  Cross-node H2D/D2H vs ReduceScatter Interference Benchmark\n");
        printf("============================================================\n");
        printf("  GPU          : %s  (device %d)\n", gpu_name, gpu_id);
        printf("  NIC (IB HCA) : %s\n", ib_hca);
        printf("  Ranks        : %d  (1 GPU per node)\n", nranks);
        printf("  Mem buf      : %.1f GB  (H2D / D2H)\n", buf_gb);
        printf("  RS buf       : %.1f GB  (ReduceScatter input per rank)\n", rs_buf_gb);
        printf("  Iterations   : %d  (warmup=%d)\n", iters, warmup);
        printf("============================================================\n\n");
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // ── Phase 1: Solo D2H ────────────────────────────────────────────────────
    if (rank == 0) {
        printf("Phase 1: Solo D2H\n");
        printf("------------------------------------------------------------\n");
    }
    Stats solo_d2h_local = bench_d2h(d_mem_buf, h_buf, mem_bytes,
                                     stream_mem, iters, warmup);
    Stats solo_d2h = summarize_stats_across_ranks(solo_d2h_local, rank, nranks);
    if (rank == 0) { print_stats(solo_d2h); printf("\n"); }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // ── Phase 2: Solo H2D ────────────────────────────────────────────────────
    if (rank == 0) {
        printf("Phase 2: Solo H2D\n");
        printf("------------------------------------------------------------\n");
    }
    Stats solo_h2d_local = bench_h2d(d_mem_buf, h_buf, mem_bytes,
                                     stream_mem, iters, warmup);
    Stats solo_h2d = summarize_stats_across_ranks(solo_h2d_local, rank, nranks);
    if (rank == 0) { print_stats(solo_h2d); printf("\n"); }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // ── Phase 3: Solo ReduceScatter ──────────────────────────────────────────
    if (rank == 0) {
        printf("Phase 3: Solo ReduceScatter (cross-node NCCL)\n");
        printf("------------------------------------------------------------\n");
    }
    Stats solo_rs_local = bench_rs(d_rs_send, d_rs_recv, rs_bytes, nranks,
                                   comm, stream_rs, iters, warmup);
    Stats solo_rs = summarize_stats_across_ranks(solo_rs_local, rank, nranks);
    if (rank == 0) { print_stats(solo_rs); printf("\n"); }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // ── Phase 4: Concurrent D2H + ReduceScatter ──────────────────────────────
    if (rank == 0) {
        printf("Phase 4: Concurrent D2H + ReduceScatter\n");
        printf("------------------------------------------------------------\n");
    }
    ConcurrentResult cr_d2h_local = bench_concurrent_d2h_rs(
        d_mem_buf, h_buf, mem_bytes,
        d_rs_send, d_rs_recv, rs_bytes, nranks,
        comm, stream_mem, stream_rs, iters, warmup, "D2H");
    ConcurrentResult cr_d2h {
        summarize_stats_across_ranks(cr_d2h_local.mem_stats, rank, nranks),
        summarize_stats_across_ranks(cr_d2h_local.rs_stats,  rank, nranks),
    };
    if (rank == 0) {
        print_stats(cr_d2h.mem_stats);
        print_stats(cr_d2h.rs_stats);
        printf("\n");
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // ── Phase 5: Concurrent H2D + ReduceScatter ──────────────────────────────
    if (rank == 0) {
        printf("Phase 5: Concurrent H2D + ReduceScatter\n");
        printf("------------------------------------------------------------\n");
    }
    ConcurrentResult cr_h2d_local = bench_concurrent_h2d_rs(
        d_mem_buf, h_buf, mem_bytes,
        d_rs_send, d_rs_recv, rs_bytes, nranks,
        comm, stream_mem, stream_rs, iters, warmup);
    ConcurrentResult cr_h2d {
        summarize_stats_across_ranks(cr_h2d_local.mem_stats, rank, nranks),
        summarize_stats_across_ranks(cr_h2d_local.rs_stats,  rank, nranks),
    };
    if (rank == 0) {
        print_stats(cr_h2d.mem_stats);
        print_stats(cr_h2d.rs_stats);
        printf("\n");
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // ── Summary ───────────────────────────────────────────────────────────────
    if (rank == 0) {
        printf("============================================================\n");
        printf("  Interference Summary (worst rank, >5%% BW drop or latency increase)\n");
        printf("============================================================\n");
        printf("  [D2H + RS concurrently vs solo]\n");
        print_delta("  D2H",          solo_d2h, cr_d2h.mem_stats);
        print_delta("  RS (vs D2H)",  solo_rs,  cr_d2h.rs_stats);
        printf("\n");
        printf("  [H2D + RS concurrently vs solo]\n");
        print_delta("  H2D",          solo_h2d, cr_h2d.mem_stats);
        print_delta("  RS (vs H2D)",  solo_rs,  cr_h2d.rs_stats);
        printf("============================================================\n\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_mem_buf));
    CUDA_CHECK(cudaFree(d_rs_send));
    CUDA_CHECK(cudaFree(d_rs_recv));
    CUDA_CHECK(cudaFreeHost(h_buf));
    CUDA_CHECK(cudaStreamDestroy(stream_mem));
    CUDA_CHECK(cudaStreamDestroy(stream_rs));
    NCCL_CHECK(ncclCommDestroy(comm));
    MPI_CHECK(MPI_Finalize());
    return 0;
}
