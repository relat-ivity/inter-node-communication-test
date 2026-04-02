/*
 * Cross-node H2D / D2H / D2D (ReduceScatter) interference benchmark.
 *
 * This version does not depend on MPI. Start one process per node and use:
 *   BENCH_RANK=0 / 1
 *   BENCH_WORLD_SIZE=2
 *   BENCH_MASTER_ADDR=<rank0_ip>
 *   BENCH_MASTER_PORT=<tcp_port>
 *
 * NCCL is still responsible for the data path. A tiny TCP control channel is
 * only used for bootstrap, barriers, and result aggregation.
 */

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <netdb.h>
#include <netinet/in.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

struct ControlPlane {
    int rank = -1;
    int nranks = -1;
    int sock_fd = -1;
};

static int g_rank_for_log = -1;
static ControlPlane g_control;

[[noreturn]] static void fatalf(const char *scope, const char *file, int line,
                                const char *fmt, ...)
{
    fprintf(stderr, "[%s][rank %d] %s:%d ", scope, g_rank_for_log, file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    fflush(stderr);
    std::exit(1);
}

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fatalf("CUDA", __FILE__, __LINE__, "%s", cudaGetErrorString(_e)); \
        }                                                                      \
    } while (0)

#define NCCL_CHECK(call)                                                       \
    do {                                                                       \
        ncclResult_t _r = (call);                                              \
        if (_r != ncclSuccess) {                                               \
            fatalf("NCCL", __FILE__, __LINE__, "%s", ncclGetErrorString(_r)); \
        }                                                                      \
    } while (0)

static double env_double(const char *name, double def) {
    const char *v = getenv(name);
    return v ? atof(v) : def;
}

static int env_int(const char *name, int def) {
    const char *v = getenv(name);
    return v ? atoi(v) : def;
}

static const char *env_str(const char *name, const char *def = nullptr) {
    const char *v = getenv(name);
    return (v && v[0] != '\0') ? v : def;
}

static void close_fd(int &fd) {
    if (fd >= 0) close(fd);
    fd = -1;
}

static void send_all(int fd, const void *buf, size_t len)
{
    const char *ptr = static_cast<const char *>(buf);
    while (len > 0) {
        ssize_t sent = send(fd, ptr, len, 0);
        if (sent <= 0) {
            fatalf("SOCKET", __FILE__, __LINE__, "send failed: %s", strerror(errno));
        }
        ptr += sent;
        len -= static_cast<size_t>(sent);
    }
}

static void recv_all(int fd, void *buf, size_t len)
{
    char *ptr = static_cast<char *>(buf);
    while (len > 0) {
        ssize_t recvd = recv(fd, ptr, len, 0);
        if (recvd <= 0) {
            fatalf("SOCKET", __FILE__, __LINE__, "recv failed: %s", strerror(errno));
        }
        ptr += recvd;
        len -= static_cast<size_t>(recvd);
    }
}

static int create_listener_and_accept(int port)
{
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        fatalf("SOCKET", __FILE__, __LINE__, "socket failed: %s", strerror(errno));
    }

    int reuse = 1;
    if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) {
        close_fd(listen_fd);
        fatalf("SOCKET", __FILE__, __LINE__, "setsockopt failed: %s", strerror(errno));
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        close_fd(listen_fd);
        fatalf("SOCKET", __FILE__, __LINE__, "bind failed on port %d: %s",
               port, strerror(errno));
    }
    if (listen(listen_fd, 1) != 0) {
        close_fd(listen_fd);
        fatalf("SOCKET", __FILE__, __LINE__, "listen failed: %s", strerror(errno));
    }

    int peer_fd = accept(listen_fd, nullptr, nullptr);
    close_fd(listen_fd);
    if (peer_fd < 0) {
        fatalf("SOCKET", __FILE__, __LINE__, "accept failed: %s", strerror(errno));
    }
    return peer_fd;
}

static int connect_with_retry(const char *host, int port)
{
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo *result = nullptr;
    int gai_rc = getaddrinfo(host, port_str, &hints, &result);
    if (gai_rc != 0) {
        fatalf("SOCKET", __FILE__, __LINE__, "getaddrinfo(%s:%d) failed: %s",
               host, port, gai_strerror(gai_rc));
    }

    for (int attempt = 0; attempt < 60; attempt++) {
        for (addrinfo *rp = result; rp != nullptr; rp = rp->ai_next) {
            int fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
            if (fd < 0) continue;
            if (connect(fd, rp->ai_addr, static_cast<socklen_t>(rp->ai_addrlen)) == 0) {
                freeaddrinfo(result);
                return fd;
            }
            close_fd(fd);
        }
        sleep(1);
    }

    freeaddrinfo(result);
    fatalf("SOCKET", __FILE__, __LINE__, "unable to connect to %s:%d after retries",
           host, port);
}

static ControlPlane init_control_plane(int rank, int nranks,
                                       const char *master_addr, int master_port)
{
    if (nranks != 2) {
        fatalf("BOOTSTRAP", __FILE__, __LINE__, "this benchmark requires exactly 2 ranks");
    }
    if (rank < 0 || rank >= nranks) {
        fatalf("BOOTSTRAP", __FILE__, __LINE__, "invalid BENCH_RANK=%d for BENCH_WORLD_SIZE=%d",
               rank, nranks);
    }

    ControlPlane cp;
    cp.rank = rank;
    cp.nranks = nranks;
    cp.sock_fd = (rank == 0)
        ? create_listener_and_accept(master_port)
        : connect_with_retry(master_addr, master_port);
    return cp;
}

static void finalize_control_plane()
{
    close_fd(g_control.sock_fd);
}

static void control_broadcast(void *buf, size_t len)
{
    if (g_control.rank == 0) {
        send_all(g_control.sock_fd, buf, len);
    } else {
        recv_all(g_control.sock_fd, buf, len);
    }
}

static void control_barrier()
{
    uint8_t token = 0;
    if (g_control.rank == 0) {
        recv_all(g_control.sock_fd, &token, sizeof(token));
        send_all(g_control.sock_fd, &token, sizeof(token));
    } else {
        send_all(g_control.sock_fd, &token, sizeof(token));
        recv_all(g_control.sock_fd, &token, sizeof(token));
    }
}

static std::vector<double> control_gather_doubles(const double *local, size_t count)
{
    std::vector<double> gathered;
    if (g_control.rank == 0) {
        gathered.resize(static_cast<size_t>(g_control.nranks) * count);
        memcpy(gathered.data(), local, count * sizeof(double));
        recv_all(g_control.sock_fd, gathered.data() + count, count * sizeof(double));
    } else {
        send_all(g_control.sock_fd, local, count * sizeof(double));
    }
    return gathered;
}

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
    void end(cudaStream_t s) { CUDA_CHECK(cudaEventRecord(stop, s)); }
    float elapsed_ms() {
        float ms = 0;
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

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

    size_t p99_idx = static_cast<size_t>(0.99 * lats_ms.size());
    double p99 = lats_ms[p99_idx];

    double total_gb = static_cast<double>(bytes_per_iter) * lats_ms.size() / 1e9;
    double total_s = sum / 1e3;
    double bw = total_gb / total_s;

    return {label, bw, avg, p99, std};
}

static void print_stats(const Stats &s)
{
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

    std::vector<double> gathered = control_gather_doubles(local_vals, 4);
    if (rank != 0) return local;

    double min_bw = gathered[0];
    double max_avg = gathered[1];
    double max_p99 = gathered[2];
    double max_std = gathered[3];

    for (int r = 0; r < nranks; r++) {
        const double *vals = &gathered[static_cast<size_t>(r) * 4];
        printf("    rank %-2d %-28s  BW=%7.2f GB/s  avg=%8.3f ms  p99=%8.3f ms  std=%7.3f ms\n",
               r, local.label.c_str(), vals[0], vals[1], vals[2], vals[3]);
        min_bw = std::min(min_bw, vals[0]);
        max_avg = std::max(max_avg, vals[1]);
        max_p99 = std::max(max_p99, vals[2]);
        max_std = std::max(max_std, vals[3]);
    }

    return {local.label + " [worst rank]", min_bw, max_avg, max_p99, max_std};
}

static void print_delta(const char *name, const Stats &solo, const Stats &conc)
{
    double bw_delta = (conc.bw_gb_s - solo.bw_gb_s) / solo.bw_gb_s * 100.0;
    double lat_delta = (conc.avg_ms - solo.avg_ms) / solo.avg_ms * 100.0;
    printf("  %-10s  BW: %+6.1f%%   avg-lat: %+6.1f%%", name, bw_delta, lat_delta);
    bool bad = (bw_delta < -5.0) || (lat_delta > 5.0);
    printf("  %s\n", bad ? "<-- INTERFERENCE" : "OK");
}

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
        lats.push_back(timer.elapsed_ms());
    }
    control_barrier();
    return compute_stats("D2H (solo)", lats, n_bytes);
}

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
    control_barrier();
    return compute_stats("H2D (solo)", lats, n_bytes);
}

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
        control_barrier();
    }
    std::vector<float> lats;
    lats.reserve(iters);
    for (int i = 0; i < iters; i++) {
        timer.begin(stream);
        NCCL_CHECK(ncclReduceScatter(send_buf, recv_buf, recv_count,
                                     ncclFloat, ncclSum, comm, stream));
        timer.end(stream);
        float ms = timer.elapsed_ms();
        control_barrier();
        lats.push_back(ms);
    }
    return compute_stats("ReduceScatter (solo)", lats, rs_bytes);
}

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

    for (int i = 0; i < warmup; i++) {
        do_memcpy(stream_mem);
        NCCL_CHECK(ncclReduceScatter(send_buf, recv_buf, recv_count,
                                     ncclFloat, ncclSum, comm, stream_rs));
        CUDA_CHECK(cudaStreamSynchronize(stream_mem));
        CUDA_CHECK(cudaStreamSynchronize(stream_rs));
        control_barrier();
    }

    std::vector<float> lats_mem, lats_rs;
    lats_mem.reserve(iters);
    lats_rs.reserve(iters);

    for (int i = 0; i < iters; i++) {
        timer_mem.begin(stream_mem);
        do_memcpy(stream_mem);
        timer_mem.end(stream_mem);

        timer_rs.begin(stream_rs);
        NCCL_CHECK(ncclReduceScatter(send_buf, recv_buf, recv_count,
                                     ncclFloat, ncclSum, comm, stream_rs));
        timer_rs.end(stream_rs);

        lats_mem.push_back(timer_mem.elapsed_ms());
        lats_rs.push_back(timer_rs.elapsed_ms());
        control_barrier();
    }

    std::string mem_lbl = std::string(mem_label) + " (concurrent)";
    return {
        compute_stats(mem_lbl, lats_mem, mem_bytes),
        compute_stats("ReduceScatter (concurrent)", lats_rs, rs_bytes),
    };
}

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
        control_barrier();
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
        control_barrier();
    }

    return {
        compute_stats("H2D (concurrent)", lats_mem, mem_bytes),
        compute_stats("ReduceScatter (concurrent)", lats_rs, rs_bytes),
    };
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    int rank = env_int("BENCH_RANK", -1);
    int nranks = env_int("BENCH_WORLD_SIZE", 2);
    const char *master_addr = env_str("BENCH_MASTER_ADDR", nullptr);
    int master_port = env_int("BENCH_MASTER_PORT", 29500);

    if (rank < 0) {
        fatalf("CONFIG", __FILE__, __LINE__,
               "set BENCH_RANK to 0 or 1 before starting the process");
    }
    if (rank != 0 && master_addr == nullptr) {
        fatalf("CONFIG", __FILE__, __LINE__,
               "rank 1 requires BENCH_MASTER_ADDR=<rank0_ip>");
    }
    if (rank == 0 && master_addr == nullptr) {
        master_addr = "127.0.0.1";
    }

    g_rank_for_log = rank;
    g_control = init_control_plane(rank, nranks, master_addr, master_port);

    double buf_gb = env_double("BENCH_BUF_GB", 4.0);
    double rs_buf_gb = env_double("BENCH_RS_BUF_GB", 2.0);
    int iters = env_int("BENCH_ITERS", 20);
    int warmup = env_int("BENCH_WARMUP", 5);
    int gpu_id = env_int("BENCH_GPU_ID", 0);

    size_t mem_bytes = static_cast<size_t>(buf_gb * 1e9);
    size_t rs_bytes = static_cast<size_t>(rs_buf_gb * 1e9);
    mem_bytes = (mem_bytes / sizeof(float)) * sizeof(float);
    rs_bytes = (rs_bytes / sizeof(float) / nranks) * sizeof(float) * nranks;

    int n_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
    if (gpu_id < 0 || gpu_id >= n_gpus) {
        fatalf("CONFIG", __FILE__, __LINE__,
               "BENCH_GPU_ID=%d out of range (0..%d)", gpu_id, n_gpus - 1);
    }
    CUDA_CHECK(cudaSetDevice(gpu_id));

    char gpu_name[256];
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_id));
    strncpy(gpu_name, prop.name, sizeof(gpu_name) - 1);
    gpu_name[sizeof(gpu_name) - 1] = '\0';

    const char *ib_hca = env_str("NCCL_IB_HCA", "(auto)");

    ncclUniqueId nccl_id;
    if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    control_broadcast(&nccl_id, sizeof(nccl_id));

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, nccl_id, rank));

    cudaStream_t stream_mem, stream_rs;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_mem, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_rs, cudaStreamNonBlocking));

    float *d_mem_buf = nullptr;
    float *d_rs_send = nullptr;
    float *d_rs_recv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mem_buf, mem_bytes));
    CUDA_CHECK(cudaMalloc(&d_rs_send, rs_bytes));
    CUDA_CHECK(cudaMalloc(&d_rs_recv, rs_bytes / nranks));
    CUDA_CHECK(cudaMemset(d_mem_buf, 1, mem_bytes));
    CUDA_CHECK(cudaMemset(d_rs_send, 1, rs_bytes));
    CUDA_CHECK(cudaMemset(d_rs_recv, 0, rs_bytes / nranks));

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
        printf("  Master       : %s:%d\n", master_addr, master_port);
        printf("  Mem buf      : %.1f GB  (H2D / D2H)\n", buf_gb);
        printf("  RS buf       : %.1f GB  (ReduceScatter input per rank)\n", rs_buf_gb);
        printf("  Iterations   : %d  (warmup=%d)\n", iters, warmup);
        printf("============================================================\n\n");
    }
    control_barrier();

    if (rank == 0) {
        printf("Phase 1: Solo D2H\n");
        printf("------------------------------------------------------------\n");
    }
    Stats solo_d2h_local = bench_d2h(d_mem_buf, h_buf, mem_bytes, stream_mem, iters, warmup);
    Stats solo_d2h = summarize_stats_across_ranks(solo_d2h_local, rank, nranks);
    if (rank == 0) { print_stats(solo_d2h); printf("\n"); }
    control_barrier();

    if (rank == 0) {
        printf("Phase 2: Solo H2D\n");
        printf("------------------------------------------------------------\n");
    }
    Stats solo_h2d_local = bench_h2d(d_mem_buf, h_buf, mem_bytes, stream_mem, iters, warmup);
    Stats solo_h2d = summarize_stats_across_ranks(solo_h2d_local, rank, nranks);
    if (rank == 0) { print_stats(solo_h2d); printf("\n"); }
    control_barrier();

    if (rank == 0) {
        printf("Phase 3: Solo ReduceScatter (cross-node NCCL)\n");
        printf("------------------------------------------------------------\n");
    }
    Stats solo_rs_local = bench_rs(d_rs_send, d_rs_recv, rs_bytes, nranks,
                                   comm, stream_rs, iters, warmup);
    Stats solo_rs = summarize_stats_across_ranks(solo_rs_local, rank, nranks);
    if (rank == 0) { print_stats(solo_rs); printf("\n"); }
    control_barrier();

    if (rank == 0) {
        printf("Phase 4: Concurrent D2H + ReduceScatter\n");
        printf("------------------------------------------------------------\n");
    }
    ConcurrentResult cr_d2h_local = bench_concurrent_d2h_rs(
        d_mem_buf, h_buf, mem_bytes, d_rs_send, d_rs_recv, rs_bytes, nranks,
        comm, stream_mem, stream_rs, iters, warmup, "D2H");
    ConcurrentResult cr_d2h{
        summarize_stats_across_ranks(cr_d2h_local.mem_stats, rank, nranks),
        summarize_stats_across_ranks(cr_d2h_local.rs_stats, rank, nranks),
    };
    if (rank == 0) {
        print_stats(cr_d2h.mem_stats);
        print_stats(cr_d2h.rs_stats);
        printf("\n");
    }
    control_barrier();

    if (rank == 0) {
        printf("Phase 5: Concurrent H2D + ReduceScatter\n");
        printf("------------------------------------------------------------\n");
    }
    ConcurrentResult cr_h2d_local = bench_concurrent_h2d_rs(
        d_mem_buf, h_buf, mem_bytes, d_rs_send, d_rs_recv, rs_bytes, nranks,
        comm, stream_mem, stream_rs, iters, warmup);
    ConcurrentResult cr_h2d{
        summarize_stats_across_ranks(cr_h2d_local.mem_stats, rank, nranks),
        summarize_stats_across_ranks(cr_h2d_local.rs_stats, rank, nranks),
    };
    if (rank == 0) {
        print_stats(cr_h2d.mem_stats);
        print_stats(cr_h2d.rs_stats);
        printf("\n");
    }
    control_barrier();

    if (rank == 0) {
        printf("============================================================\n");
        printf("  Interference Summary (worst rank, >5%% BW drop or latency increase)\n");
        printf("============================================================\n");
        printf("  [D2H + RS concurrently vs solo]\n");
        print_delta("  D2H", solo_d2h, cr_d2h.mem_stats);
        print_delta("  RS (vs D2H)", solo_rs, cr_d2h.rs_stats);
        printf("\n");
        printf("  [H2D + RS concurrently vs solo]\n");
        print_delta("  H2D", solo_h2d, cr_h2d.mem_stats);
        print_delta("  RS (vs H2D)", solo_rs, cr_h2d.rs_stats);
        printf("============================================================\n\n");
    }

    CUDA_CHECK(cudaFree(d_mem_buf));
    CUDA_CHECK(cudaFree(d_rs_send));
    CUDA_CHECK(cudaFree(d_rs_recv));
    CUDA_CHECK(cudaFreeHost(h_buf));
    CUDA_CHECK(cudaStreamDestroy(stream_mem));
    CUDA_CHECK(cudaStreamDestroy(stream_rs));
    NCCL_CHECK(ncclCommDestroy(comm));
    finalize_control_plane();
    return 0;
}
