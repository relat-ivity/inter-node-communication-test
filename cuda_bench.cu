/*
 * Cross-node H2D / D2H / D2D (node0 ncclSend -> node1 ncclRecv) interference benchmark.
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
#include <atomic>
#include <cerrno>
#include <chrono>
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
#include <thread>
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

static Stats make_idle_stats(const std::string &label)
{
    return {label, 0.0, 0.0, 0.0, 0.0};
}

struct Measurement {
    std::vector<float> lats_ms;
    double total_ms = 0.0;
};

static Stats compute_stats(const std::string &label,
                           std::vector<float> &lats_ms,
                           size_t bytes_per_iter,
                           double total_ms)
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
    double total_s = total_ms / 1e3;
    double bw = total_gb / total_s;

    return {label, bw, avg, p99, std};
}

static void print_stats(const Stats &s)
{
    printf("  %-28s | BW=%8.2f GB/s | avg=%8.3f ms | p99=%8.3f ms | std=%7.3f ms\n",
           s.label.c_str(), s.bw_gb_s, s.avg_ms, s.p99_ms, s.std_ms);
}

static void print_delta(const char *name, const Stats &solo, const Stats &conc)
{
    double bw_delta = (conc.bw_gb_s - solo.bw_gb_s) / solo.bw_gb_s * 100.0;
    double lat_delta = (conc.avg_ms - solo.avg_ms) / solo.avg_ms * 100.0;
    printf("  %-16s | BW=%+6.1f%% | avg-lat=%+6.1f%%", name, bw_delta, lat_delta);
    bool bad = (bw_delta < -5.0) || (lat_delta > 5.0);
    printf(" | %s\n", bad ? "<-- INTERFERENCE" : "OK");
}

static Measurement measure_d2h(float *d_buf, float *h_buf, size_t n_bytes,
                               cudaStream_t stream, int iters, int warmup)
{
    CudaTimer timer;
    for (int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaMemcpyAsync(h_buf, d_buf, n_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    Measurement result;
    result.lats_ms.reserve(iters);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
        timer.begin(stream);
        CUDA_CHECK(cudaMemcpyAsync(h_buf, d_buf, n_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        timer.end(stream);
        result.lats_ms.push_back(timer.elapsed_ms());
    }
    auto end = std::chrono::steady_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

static Measurement measure_h2d(float *d_buf, float *h_buf, size_t n_bytes,
                               cudaStream_t stream, int iters, int warmup)
{
    CudaTimer timer;
    for (int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d_buf, h_buf, n_bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    Measurement result;
    result.lats_ms.reserve(iters);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
        timer.begin(stream);
        CUDA_CHECK(cudaMemcpyAsync(d_buf, h_buf, n_bytes,
                                   cudaMemcpyHostToDevice, stream));
        timer.end(stream);
        result.lats_ms.push_back(timer.elapsed_ms());
    }
    auto end = std::chrono::steady_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

static Measurement measure_p2p(float *send_buf, float *recv_buf,
                               size_t p2p_bytes, int nranks,
                               ncclComm_t comm, cudaStream_t stream,
                               int iters, int warmup)
{
    if (nranks != 2) {
        fatalf("CONFIG", __FILE__, __LINE__, "ncclSend/ncclRecv mode requires 2 ranks");
    }
    size_t send_count = p2p_bytes / sizeof(float);
    CudaTimer timer;

    for (int i = 0; i < warmup; i++) {
        if (g_control.rank == 0) {
            NCCL_CHECK(ncclSend(send_buf, send_count, ncclFloat, 1, comm, stream));
        } else {
            NCCL_CHECK(ncclRecv(recv_buf, send_count, ncclFloat, 0, comm, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    Measurement result;
    result.lats_ms.reserve(iters);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
        timer.begin(stream);
        if (g_control.rank == 0) {
            NCCL_CHECK(ncclSend(send_buf, send_count, ncclFloat, 1, comm, stream));
        } else {
            NCCL_CHECK(ncclRecv(recv_buf, send_count, ncclFloat, 0, comm, stream));
        }
        timer.end(stream);
        result.lats_ms.push_back(timer.elapsed_ms());
    }
    auto end = std::chrono::steady_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

struct ConcurrentResult {
    Stats mem_stats;
    Stats rs_stats;
};

static ConcurrentResult bench_concurrent_d2h_rs(
        float *d_buf, float *h_buf, size_t mem_bytes,
        float *send_buf, float *recv_buf, size_t p2p_bytes, int nranks,
        ncclComm_t comm,
        cudaStream_t stream_mem, cudaStream_t stream_rs,
        int gpu_id, int iters, int warmup,
        const char *mem_label)
{
    Measurement mem_measurement;
    Measurement rs_measurement;
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    const bool do_memcpy = (g_control.rank == 0);

    std::thread mem_thread;
    if (do_memcpy) {
        mem_thread = std::thread([&]() {
            CUDA_CHECK(cudaSetDevice(gpu_id));
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            mem_measurement = measure_d2h(
                d_buf, h_buf, mem_bytes, stream_mem, iters, warmup);
        });
    }

    std::thread rs_thread([&]() {
        CUDA_CHECK(cudaSetDevice(gpu_id));
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        rs_measurement = measure_p2p(
            send_buf, recv_buf, p2p_bytes, nranks, comm, stream_rs, iters, warmup);
    });

    const int expected_ready = do_memcpy ? 2 : 1;
    while (ready.load(std::memory_order_acquire) < expected_ready) {
        std::this_thread::yield();
    }
    control_barrier();
    start.store(true, std::memory_order_release);

    if (do_memcpy) {
        mem_thread.join();
    }
    rs_thread.join();

    std::string mem_lbl = std::string(mem_label) + " (concurrent)";
    return {
        do_memcpy
            ? compute_stats(mem_lbl, mem_measurement.lats_ms, mem_bytes, mem_measurement.total_ms)
            : make_idle_stats(mem_lbl + ", rank1 idle"),
        compute_stats("ncclSend (concurrent)", rs_measurement.lats_ms,
                      p2p_bytes, rs_measurement.total_ms),
    };
}

static ConcurrentResult bench_concurrent_h2d_rs(
        float *d_buf, float *h_buf, size_t mem_bytes,
        float *send_buf, float *recv_buf, size_t p2p_bytes, int nranks,
        ncclComm_t comm,
        cudaStream_t stream_mem, cudaStream_t stream_rs,
        int gpu_id, int iters, int warmup)
{
    Measurement mem_measurement;
    Measurement rs_measurement;
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    const bool do_memcpy = (g_control.rank == 0);

    std::thread mem_thread;
    if (do_memcpy) {
        mem_thread = std::thread([&]() {
            CUDA_CHECK(cudaSetDevice(gpu_id));
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            mem_measurement = measure_h2d(
                d_buf, h_buf, mem_bytes, stream_mem, iters, warmup);
        });
    }

    std::thread rs_thread([&]() {
        CUDA_CHECK(cudaSetDevice(gpu_id));
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        rs_measurement = measure_p2p(
            send_buf, recv_buf, p2p_bytes, nranks, comm, stream_rs, iters, warmup);
    });

    const int expected_ready = do_memcpy ? 2 : 1;
    while (ready.load(std::memory_order_acquire) < expected_ready) {
        std::this_thread::yield();
    }
    control_barrier();
    start.store(true, std::memory_order_release);

    if (do_memcpy) {
        mem_thread.join();
    }
    rs_thread.join();

    return {
        do_memcpy
            ? compute_stats("H2D (concurrent)", mem_measurement.lats_ms,
                            mem_bytes, mem_measurement.total_ms)
            : make_idle_stats("H2D (concurrent, rank1 idle)"),
        compute_stats("ncclSend (concurrent)", rs_measurement.lats_ms,
                      p2p_bytes, rs_measurement.total_ms),
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
    double p2p_buf_gb = env_double("BENCH_P2P_BUF_GB",
                                   env_double("BENCH_AG_BUF_GB",
                                              env_double("BENCH_RS_BUF_GB", 2.0)));
    int iters = env_int("BENCH_ITERS", 20);
    int warmup = env_int("BENCH_WARMUP", 5);
    int gpu_id = env_int("BENCH_GPU_ID", 0);

    size_t mem_bytes = static_cast<size_t>(buf_gb * 1e9);
    size_t p2p_bytes = static_cast<size_t>(p2p_buf_gb * 1e9);
    mem_bytes = (mem_bytes / sizeof(float)) * sizeof(float);
    p2p_bytes = (p2p_bytes / sizeof(float)) * sizeof(float);

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
    float *d_p2p_send = nullptr;
    float *d_p2p_recv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mem_buf, mem_bytes));
    CUDA_CHECK(cudaMalloc(&d_p2p_send, p2p_bytes));
    CUDA_CHECK(cudaMalloc(&d_p2p_recv, p2p_bytes));
    CUDA_CHECK(cudaMemset(d_mem_buf, 1, mem_bytes));
    CUDA_CHECK(cudaMemset(d_p2p_send, 1, p2p_bytes));
    CUDA_CHECK(cudaMemset(d_p2p_recv, 0, p2p_bytes));

    float *h_buf = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_buf, mem_bytes));
    memset(h_buf, 1, mem_bytes);

    if (rank == 0) {
        printf("\n");
        printf("============================================================\n");
        printf("  Cross-node H2D/D2H vs ncclSend/ncclRecv Interference Benchmark\n");
        printf("============================================================\n");
        printf("  GPU          : %s  (device %d)\n", gpu_name, gpu_id);
        printf("  NIC (IB HCA) : %s\n", ib_hca);
        printf("  Ranks        : %d  (1 GPU per node)\n", nranks);
        printf("  Mem copy     : rank0 only\n");
        printf("  Master       : %s:%d\n", master_addr, master_port);
        printf("  Mem buf      : %.1f GB  (H2D / D2H)\n", buf_gb);
        printf("  P2P buf      : %.1f GB  (node0 send -> node1 recv)\n", p2p_buf_gb);
        printf("  Iterations   : %d  (warmup=%d)\n", iters, warmup);
        printf("============================================================\n\n");
    }
    control_barrier();

    if (rank == 0) {
        printf("Phase 1: Solo D2H\n");
        printf("------------------------------------------------------------\n");
    }
    control_barrier();
    Stats solo_d2h = make_idle_stats("D2H (solo)");
    if (rank == 0) {
        Measurement solo_d2h_measurement =
            measure_d2h(d_mem_buf, h_buf, mem_bytes, stream_mem, iters, warmup);
        solo_d2h = compute_stats(
            "D2H (solo)", solo_d2h_measurement.lats_ms, mem_bytes, solo_d2h_measurement.total_ms);
    }
    if (rank == 0) { print_stats(solo_d2h); printf("\n"); }
    control_barrier();

    if (rank == 0) {
        printf("Phase 2: Solo H2D\n");
        printf("------------------------------------------------------------\n");
    }
    control_barrier();
    Stats solo_h2d = make_idle_stats("H2D (solo)");
    if (rank == 0) {
        Measurement solo_h2d_measurement =
            measure_h2d(d_mem_buf, h_buf, mem_bytes, stream_mem, iters, warmup);
        solo_h2d = compute_stats(
            "H2D (solo)", solo_h2d_measurement.lats_ms, mem_bytes, solo_h2d_measurement.total_ms);
    }
    if (rank == 0) { print_stats(solo_h2d); printf("\n"); }
    control_barrier();

    if (rank == 0) {
        printf("Phase 3: Solo ncclSend/ncclRecv (node0 -> node1)\n");
        printf("------------------------------------------------------------\n");
    }
    control_barrier();
    Measurement solo_rs_measurement =
        measure_p2p(d_p2p_send, d_p2p_recv, p2p_bytes, nranks, comm, stream_rs, iters, warmup);
    Stats solo_rs = compute_stats(
        "ncclSend (solo)", solo_rs_measurement.lats_ms, p2p_bytes, solo_rs_measurement.total_ms);
    if (rank == 0) { print_stats(solo_rs); printf("\n"); }
    control_barrier();

    if (rank == 0) {
        printf("Phase 4: Concurrent D2H + ncclSend\n");
        printf("------------------------------------------------------------\n");
    }
    control_barrier();
    ConcurrentResult cr_d2h = bench_concurrent_d2h_rs(
        d_mem_buf, h_buf, mem_bytes, d_p2p_send, d_p2p_recv, p2p_bytes, nranks,
        comm, stream_mem, stream_rs, gpu_id, iters, warmup, "D2H");
    if (rank == 0) {
        print_stats(cr_d2h.mem_stats);
        print_stats(cr_d2h.rs_stats);
        printf("\n");
    }
    control_barrier();

    if (rank == 0) {
        printf("Phase 5: Concurrent H2D + ncclSend\n");
        printf("------------------------------------------------------------\n");
    }
    control_barrier();
    ConcurrentResult cr_h2d = bench_concurrent_h2d_rs(
        d_mem_buf, h_buf, mem_bytes, d_p2p_send, d_p2p_recv, p2p_bytes, nranks,
        comm, stream_mem, stream_rs, gpu_id, iters, warmup);
    if (rank == 0) {
        print_stats(cr_h2d.mem_stats);
        print_stats(cr_h2d.rs_stats);
        printf("\n");
    }
    control_barrier();

    if (rank == 0) {
        printf("============================================================\n");
        printf("  Interference Summary (rank 0 local view, >5%% BW drop or latency increase)\n");
        printf("============================================================\n");
        printf("  [D2H + ncclSend concurrently vs solo]\n");
        print_delta("D2H", solo_d2h, cr_d2h.mem_stats);
        print_delta("Send (vs D2H)", solo_rs, cr_d2h.rs_stats);
        printf("\n");
        printf("  [H2D + ncclSend concurrently vs solo]\n");
        print_delta("H2D", solo_h2d, cr_h2d.mem_stats);
        print_delta("Send (vs H2D)", solo_rs, cr_h2d.rs_stats);
        printf("============================================================\n\n");
    }

    CUDA_CHECK(cudaFree(d_mem_buf));
    CUDA_CHECK(cudaFree(d_p2p_send));
    CUDA_CHECK(cudaFree(d_p2p_recv));
    CUDA_CHECK(cudaFreeHost(h_buf));
    CUDA_CHECK(cudaStreamDestroy(stream_mem));
    CUDA_CHECK(cudaStreamDestroy(stream_rs));
    NCCL_CHECK(ncclCommDestroy(comm));
    finalize_control_plane();
    return 0;
}
