/*
 * NCCL send/recv issue-latency and batch-bandwidth benchmark.
 *
 * Start one process per node and use:
 *   BENCH_RANK=0 / 1
 *   BENCH_WORLD_SIZE=2
 *   BENCH_MASTER_ADDR=<rank0_ip>
 *   BENCH_MASTER_PORT=<tcp_port>
 *
 * The measurement mirrors the async-submit style used by the GDR bench:
 *   - warm up with a batch of async ncclSend/ncclRecv operations
 *   - measure per-operation issue latency around NCCL enqueue
 *   - synchronize once at the end of the measured batch
 *   - compute bandwidth from bytes * iterations / total batch time
 */

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <netdb.h>
#include <netinet/in.h>
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

static double now_us()
{
    using namespace std::chrono;
    return duration_cast<nanoseconds>(
               steady_clock::now().time_since_epoch()).count() / 1e3;
}

static int env_int(const char *name, int def)
{
    const char *v = getenv(name);
    return (v && v[0] != '\0') ? atoi(v) : def;
}

static const char *env_str(const char *name, const char *def = nullptr)
{
    const char *v = getenv(name);
    return (v && v[0] != '\0') ? v : def;
}

static void close_fd(int &fd)
{
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

template <typename T>
static T exchange_with_peer(const T &local)
{
    T remote{};
    if (g_control.rank == 0) {
        recv_all(g_control.sock_fd, &remote, sizeof(remote));
        send_all(g_control.sock_fd, &local, sizeof(local));
    } else {
        send_all(g_control.sock_fd, &local, sizeof(local));
        recv_all(g_control.sock_fd, &remote, sizeof(remote));
    }
    return remote;
}

struct RankResult {
    double issue_median_us = 0.0;
    double issue_p99_us = 0.0;
    double total_us = 0.0;
    double bw_GBs = 0.0;
};

struct SweepRow {
    size_t bytes = 0;
    RankResult send;
    RankResult recv;
    double overall_total_us = 0.0;
    double overall_bw_GBs = 0.0;
};

static RankResult analyse(std::vector<double> &issue_samples,
                          size_t bytes, int iters, double total_us)
{
    if (issue_samples.empty()) {
        fatalf("STATS", __FILE__, __LINE__, "empty issue sample set");
    }
    std::sort(issue_samples.begin(), issue_samples.end());

    size_t n = issue_samples.size();
    size_t p99_idx = static_cast<size_t>(n * 0.99);
    if (p99_idx >= n) p99_idx = n - 1;

    RankResult r{};
    r.issue_median_us = issue_samples[n / 2];
    r.issue_p99_us = issue_samples[p99_idx];
    r.total_us = total_us;
    r.bw_GBs = (iters > 0 && total_us > 0.0)
                 ? ((double)bytes * (double)iters / 1e9) / (total_us / 1e6)
                 : 0.0;
    return r;
}

static void check_nccl_async(ncclComm_t comm)
{
    ncclResult_t async_err = ncclSuccess;
    NCCL_CHECK(ncclCommGetAsyncError(comm, &async_err));
    if (async_err != ncclSuccess) {
        fatalf("NCCL", __FILE__, __LINE__, "async error: %s",
               ncclGetErrorString(async_err));
    }
}

static void issue_one_nccl_p2p(float *send_buf, float *recv_buf,
                               size_t count, ncclComm_t comm, cudaStream_t stream)
{
    NCCL_CHECK(ncclGroupStart());
    if (g_control.rank == 0) {
        NCCL_CHECK(ncclSend(send_buf, count, ncclFloat, 1, comm, stream));
    } else {
        NCCL_CHECK(ncclRecv(recv_buf, count, ncclFloat, 0, comm, stream));
    }
    NCCL_CHECK(ncclGroupEnd());
}

static RankResult run_nccl_timings(float *send_buf, float *recv_buf,
                                   size_t bytes, ncclComm_t comm,
                                   cudaStream_t stream, int warmup, int iters)
{
    if (iters <= 0) {
        fatalf("CONFIG", __FILE__, __LINE__, "BENCH_NCCL_ITERS must be positive");
    }
    if (warmup < 0) {
        fatalf("CONFIG", __FILE__, __LINE__, "BENCH_NCCL_WARMUP must be >= 0");
    }
    if (bytes % sizeof(float) != 0) {
        fatalf("CONFIG", __FILE__, __LINE__, "bytes=%zu is not divisible by sizeof(float)",
               bytes);
    }

    size_t count = bytes / sizeof(float);

    control_barrier();
    for (int i = 0; i < warmup; i++) {
        issue_one_nccl_p2p(send_buf, recv_buf, count, comm, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    check_nccl_async(comm);

    control_barrier();
    std::vector<double> issue_samples;
    issue_samples.reserve(static_cast<size_t>(iters));

    double t0 = now_us();
    for (int i = 0; i < iters; i++) {
        double issue0 = now_us();
        issue_one_nccl_p2p(send_buf, recv_buf, count, comm, stream);
        issue_samples.push_back(now_us() - issue0);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    check_nccl_async(comm);
    double total_us = now_us() - t0;

    return analyse(issue_samples, bytes, iters, total_us);
}

static void format_size(size_t bytes, char *out, size_t out_len)
{
    if (bytes < (1ULL << 10)) {
        snprintf(out, out_len, "%zuB", bytes);
    } else if (bytes < (1ULL << 20)) {
        snprintf(out, out_len, "%zuKiB", bytes >> 10);
    } else {
        snprintf(out, out_len, "%zuMiB", bytes >> 20);
    }
}

static void print_issue_table(const std::vector<SweepRow> &rows)
{
    printf("\n--- NCCL Send/Recv Issue Latency ---\n");
    printf("%-12s | %-23s | %-23s\n",
           "Size", "Send median / p99", "Recv median / p99");
    printf("%-12s-+-%-23s-+-%-23s\n",
           "------------", "-----------------------", "-----------------------");

    for (const SweepRow &row : rows) {
        char size_str[32];
        format_size(row.bytes, size_str, sizeof(size_str));
        printf("%-12s | %8.2f us / %8.2f us | %8.2f us / %8.2f us\n",
               size_str,
               row.send.issue_median_us, row.send.issue_p99_us,
               row.recv.issue_median_us, row.recv.issue_p99_us);
    }
}

static void print_bandwidth_table(const std::vector<SweepRow> &rows)
{
    printf("\n--- NCCL Send/Recv Batch Bandwidth ---\n");
    printf("%-12s | %-14s | %-14s | %-14s | %-14s\n",
           "Size", "Overall BW", "Send BW", "Recv BW", "Batch total");
    printf("%-12s-+-%-14s-+-%-14s-+-%-14s-+-%-14s\n",
           "------------", "--------------", "--------------", "--------------", "--------------");

    for (const SweepRow &row : rows) {
        char size_str[32];
        format_size(row.bytes, size_str, sizeof(size_str));
        printf("%-12s | %8.2f GB/s | %8.2f GB/s | %8.2f GB/s | %8.2f ms\n",
               size_str,
               row.overall_bw_GBs,
               row.send.bw_GBs,
               row.recv.bw_GBs,
               row.overall_total_us / 1000.0);
    }
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

    int gpu_id = env_int("BENCH_GPU_ID", 0);
    int iters = env_int("BENCH_NCCL_ITERS", env_int("BENCH_ITERS", 1000));
    int warmup = env_int("BENCH_NCCL_WARMUP", env_int("BENCH_WARMUP", 100));

    int n_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
    if (gpu_id < 0 || gpu_id >= n_gpus) {
        fatalf("CONFIG", __FILE__, __LINE__,
               "BENCH_GPU_ID=%d out of range (0..%d)", gpu_id, n_gpus - 1);
    }
    CUDA_CHECK(cudaSetDevice(gpu_id));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_id));

    const char *ib_hca = env_str("NCCL_IB_HCA", "(auto)");

    ncclUniqueId nccl_id;
    if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    control_broadcast(&nccl_id, sizeof(nccl_id));

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, nccl_id, rank));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    constexpr size_t kMinBytes = 4ULL << 10;
    constexpr size_t kMaxBytes = 4ULL << 20;
    std::vector<size_t> sizes;
    for (size_t s = kMinBytes; s <= kMaxBytes; s *= 4) {
        sizes.push_back(s);
    }

    const size_t max_bytes = sizes.back();
    float *send_buf = nullptr;
    float *recv_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&send_buf, max_bytes));
    CUDA_CHECK(cudaMalloc(&recv_buf, max_bytes));
    CUDA_CHECK(cudaMemset(send_buf, 0xA5, max_bytes));
    CUDA_CHECK(cudaMemset(recv_buf, 0, max_bytes));

    if (rank == 0) {
        printf("\n");
        printf("=================================================================\n");
        printf("  NCCL Send/Recv Issue Latency + Batch Bandwidth Benchmark\n");
        printf("=================================================================\n");
        printf("  GPU          : %s  (device %d)\n", prop.name, gpu_id);
        printf("  NIC (IB HCA) : %s\n", ib_hca);
        printf("  Ranks        : %d  (rank0 send -> rank1 recv)\n", nranks);
        printf("  Master       : %s:%d\n", master_addr, master_port);
        printf("  Sizes        : 4KiB .. 4MiB  (x4)\n");
        printf("  Iterations   : %d  (warmup=%d)\n", iters, warmup);
        printf("=================================================================\n");
        fflush(stdout);
    }
    control_barrier();

    std::vector<SweepRow> rows;
    rows.reserve(sizes.size());

    for (size_t bytes : sizes) {
        char size_str[32];
        format_size(bytes, size_str, sizeof(size_str));
        if (rank == 0) {
            printf("\nRunning %s ...\n", size_str);
            fflush(stdout);
        }

        RankResult local = run_nccl_timings(send_buf, recv_buf, bytes, comm,
                                            stream, warmup, iters);
        RankResult peer = exchange_with_peer(local);

        if (rank == 0) {
            SweepRow row{};
            row.bytes = bytes;
            row.send = local;
            row.recv = peer;
            row.overall_total_us = std::max(local.total_us, peer.total_us);
            row.overall_bw_GBs = (iters > 0 && row.overall_total_us > 0.0)
                                   ? ((double)bytes * (double)iters / 1e9) /
                                         (row.overall_total_us / 1e6)
                                   : 0.0;
            rows.push_back(row);
        }

        control_barrier();
    }

    if (rank == 0) {
        print_issue_table(rows);
        print_bandwidth_table(rows);
        printf("\n=================================================================\n");
        printf("Done.\n");
        printf("=================================================================\n");
    }

    CUDA_CHECK(cudaFree(send_buf));
    CUDA_CHECK(cudaFree(recv_buf));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommDestroy(comm));
    finalize_control_plane();
    return 0;
}
