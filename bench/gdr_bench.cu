/*
 * Cross-node GPUDirect RDMA H2D / D2H and ncclSend/ncclRecv interference benchmark.
 *
 * This version mirrors cuda_bench.cu, but replaces rank0's local memcpy path
 * with GDRCopy-based H2D/D2H transfers and measures their bandwidth while
 * running concurrently with node0 -> node1 NCCL send/recv traffic.
 *
 * Start one process per node and use:
 *   BENCH_RANK=0 / 1
 *   BENCH_WORLD_SIZE=2
 *   BENCH_MASTER_ADDR=<rank0_ip>
 *   BENCH_MASTER_PORT=<tcp_port>
 *
 * Rank0:
 *   - opens the GDR channel on BENCH_GDR_NIC (defaults to NCCL_IB_HCA sans port)
 *   - runs solo/concurrent GDR H2D and D2H
 * Rank1:
 *   - only participates in ncclRecv to provide the remote communication peer
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
#include <memory>
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

#include "gdr/gdr_copy.h"

#ifndef GDR_BENCH_ENABLE_QOS
#define GDR_BENCH_ENABLE_QOS 0
#endif

struct ControlPlane {
    int rank = -1;
    int nranks = -1;
    int sock_fd = -1;
};

static int g_rank_for_log = -1;
static ControlPlane g_control;
static constexpr size_t kBytesPerKiB = 1ULL << 10;
static constexpr size_t kKiBPerGiB = 1ULL << 20;
static constexpr size_t kBytesPerGiB = 1ULL << 30;

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

static double env_buffer_kb(const char *kb_name, const char *gb_name, double def_kb) {
    const char *kb = getenv(kb_name);
    if (kb && kb[0] != '\0') return atof(kb);

    const char *gb = getenv(gb_name);
    if (gb && gb[0] != '\0') return atof(gb) * static_cast<double>(kKiBPerGiB);

    return def_kb;
}

static int env_int(const char *name, int def) {
    const char *v = getenv(name);
    return v ? atoi(v) : def;
}

static const char *env_str(const char *name, const char *def = nullptr) {
    const char *v = getenv(name);
    return (v && v[0] != '\0') ? v : def;
}

static int env_int_strict(const char *name, int def)
{
    const char *v = getenv(name);
    if (!v || v[0] == '\0') return def;

    char *end = nullptr;
    errno = 0;
    long parsed = strtol(v, &end, 10);
    if (errno != 0 || !end || end == v || *end != '\0') {
        fatalf("CONFIG", __FILE__, __LINE__,
               "invalid integer for %s: '%s'", name, v);
    }
    return static_cast<int>(parsed);
}

static double env_double_strict(const char *name, double def)
{
    const char *v = getenv(name);
    if (!v || v[0] == '\0') return def;

    char *end = nullptr;
    errno = 0;
    double parsed = strtod(v, &end);
    if (errno != 0 || !end || end == v || *end != '\0') {
        fatalf("CONFIG", __FILE__, __LINE__,
               "invalid floating-point value for %s: '%s'", name, v);
    }
    return parsed;
}

static int env_traffic_class(const char *name, int def)
{
    int tc = env_int_strict(name, def);
    if (tc < 0 || tc > 255) {
        fatalf("CONFIG", __FILE__, __LINE__,
               "%s=%d out of range (expected 0..255)", name, tc);
    }
    return tc;
}

static int traffic_class_to_dscp(int tc)
{
    return (tc >> 2) & 0x3f;
}

static bool env_bool(const char *name, bool def)
{
    const char *v = getenv(name);
    if (!v || v[0] == '\0') return def;
    if (strcmp(v, "1") == 0 || strcmp(v, "true") == 0 ||
        strcmp(v, "TRUE") == 0 || strcmp(v, "on") == 0 ||
        strcmp(v, "ON") == 0 || strcmp(v, "yes") == 0 ||
        strcmp(v, "YES") == 0) {
        return true;
    }
    if (strcmp(v, "0") == 0 || strcmp(v, "false") == 0 ||
        strcmp(v, "FALSE") == 0 || strcmp(v, "off") == 0 ||
        strcmp(v, "OFF") == 0 || strcmp(v, "no") == 0 ||
        strcmp(v, "NO") == 0) {
        return false;
    }
    return atoi(v) != 0;
}

static std::string normalize_nic_name(const char *value)
{
    if (!value || value[0] == '\0') return {};
    std::string nic(value);
    size_t split = nic.find_first_of(",:");
    if (split != std::string::npos) nic.erase(split);
    return nic;
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

struct Stats {
    std::string label;
    double bw_gib_s;
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

static Measurement make_batch_measurement(int iters, double total_ms)
{
    if (iters <= 0) {
        fatalf("CONFIG", __FILE__, __LINE__, "BENCH_ITERS must be positive");
    }

    Measurement result;
    result.total_ms = total_ms;
    result.lats_ms.assign(static_cast<size_t>(iters),
                          static_cast<float>(total_ms / iters));
    return result;
}

struct GDRMeasurement {
    Measurement timing;
    GDRStats transport;
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

    double total_gib = static_cast<double>(bytes_per_iter) * lats_ms.size() /
                       static_cast<double>(kBytesPerGiB);
    double total_s = total_ms / 1e3;
    double bw = total_gib / total_s;

    return {label, bw, avg, p99, std};
}

static void print_stats(const Stats &s)
{
    printf("  %-28s | BW=%8.2f GiB/s | avg=%8.3f ms | p99=%8.3f ms | std=%7.3f ms\n",
           s.label.c_str(), s.bw_gib_s, s.avg_ms, s.p99_ms, s.std_ms);
}

static const char *gdr_path_name(const GDRStats &s)
{
    if (s.total_ops == 0) return "idle";
    if (s.rdma_ops > 0 && s.fallback_ops == 0) return "RDMA";
    if (s.rdma_ops == 0 && s.fallback_ops > 0) return "cudaMemcpy fallback";
    return "mixed";
}

static void print_gdr_transport_stats(const char *label, const GDRStats &s)
{
    printf("    %-28s | path=%-17s | ops=%6llu | rdma=%6llu | fallback=%6llu | avg-op=%9.3f us\n",
           label, gdr_path_name(s),
           static_cast<unsigned long long>(s.total_ops),
           static_cast<unsigned long long>(s.rdma_ops),
           static_cast<unsigned long long>(s.fallback_ops),
           s.avg_latency_us);
}

static void print_delta(const char *name, const Stats &solo, const Stats &conc)
{
    double bw_delta = (conc.bw_gib_s - solo.bw_gib_s) / solo.bw_gib_s * 100.0;
    double lat_delta = (conc.avg_ms - solo.avg_ms) / solo.avg_ms * 100.0;
    printf("  %-16s | BW=%+6.1f%% | avg-lat=%+6.1f%%", name, bw_delta, lat_delta);
    bool bad = (bw_delta < -5.0) || (lat_delta > 5.0);
    printf(" | %s\n", bad ? "<-- INTERFERENCE" : "OK");
}

static void print_qos_check(const char *label, const Stats &solo_nccl,
                            const Stats &conc_nccl, double target_ratio)
{
    double retain_ratio = 0.0;
    if (solo_nccl.bw_gib_s > 0.0) {
        retain_ratio = conc_nccl.bw_gib_s / solo_nccl.bw_gib_s;
    }
    printf("    %-24s | solo=%8.2f GiB/s | concurrent=%8.2f GiB/s | retain=%6.2f%% | target=%6.2f%% | %s\n",
           label, solo_nccl.bw_gib_s, conc_nccl.bw_gib_s,
           retain_ratio * 100.0, target_ratio * 100.0,
           retain_ratio >= target_ratio ? "PASS" : "FAIL");
}

static Measurement measure_p2p(float *send_buf, float *recv_buf,
                               size_t p2p_bytes, int nranks,
                               ncclComm_t comm, cudaStream_t stream,
                               int iters, int warmup)
{
    if (nranks != 2) {
        fatalf("CONFIG", __FILE__, __LINE__, "ncclSend/ncclRecv mode requires 2 ranks");
    }
    if (iters <= 0) {
        fatalf("CONFIG", __FILE__, __LINE__, "BENCH_ITERS must be positive");
    }
    size_t send_count = p2p_bytes / sizeof(float);

    if (warmup > 0) {
        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < warmup; i++) {
            if (g_control.rank == 0) {
                NCCL_CHECK(ncclSend(send_buf, send_count, ncclFloat, 1, comm, stream));
            } else {
                NCCL_CHECK(ncclRecv(recv_buf, send_count, ncclFloat, 0, comm, stream));
            }
        }
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    auto start = std::chrono::steady_clock::now();
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < iters; i++) {
        if (g_control.rank == 0) {
            NCCL_CHECK(ncclSend(send_buf, send_count, ncclFloat, 1, comm, stream));
        } else {
            NCCL_CHECK(ncclRecv(recv_buf, send_count, ncclFloat, 0, comm, stream));
        }
    }
    NCCL_CHECK(ncclGroupEnd());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::steady_clock::now();
    return make_batch_measurement(
        iters, std::chrono::duration<double, std::milli>(end - start).count());
}

static std::string gdr_error_string(int rc)
{
    if (rc == 0) return "success";
    int err = (rc < 0) ? -rc : rc;
    const char *msg = strerror(err);
    if (msg && msg[0] != '\0') {
        return std::string(msg) + " (rc=" + std::to_string(rc) + ")";
    }
    return "rc=" + std::to_string(rc);
}

static const char *gdr_kind_name(GDRCopyKind kind)
{
    switch (kind) {
        case GDR_H2D: return "GDR_H2D";
        case GDR_D2H: return "GDR_D2H";
        case GDR_D2D: return "GDR_D2D";
        default: return "GDR_UNKNOWN";
    }
}

static void gdr_wait_for_one_completion(const std::shared_ptr<GDRCopyChannel> &channel,
                                        GDRCopyKind kind, int &outstanding)
{
    while (true) {
        uint64_t done_id = 0;
        int rc = channel->poll_wc(&done_id);
        if (rc == 0) {
            (void)done_id;
            if (outstanding > 0) outstanding--;
            return;
        }
        if (rc == -EAGAIN) {
            std::this_thread::yield();
            continue;
        }
        fatalf("GDR", __FILE__, __LINE__, "poll %s failed: %s",
               gdr_kind_name(kind), gdr_error_string(rc).c_str());
    }
}

static void gdr_submit_batch_and_wait(const std::shared_ptr<GDRCopyChannel> &channel,
                                      void *dst, const void *src, size_t bytes,
                                      GDRCopyKind kind, int count)
{
    int outstanding = 0;
    for (int i = 0; i < count; i++) {
        while (true) {
            uint64_t req_id = 0;
            int expected_wcs = 0;
            int rc = channel->memcpy_async_tagged(
                dst, src, bytes, kind, &req_id, &expected_wcs);
            if (rc == 0) {
                (void)req_id;
                (void)expected_wcs;
                outstanding++;
                break;
            }
            if (rc == -EBUSY) {
                if (outstanding <= 0) {
                    fatalf("GDR", __FILE__, __LINE__,
                           "submit %s reported busy with no outstanding requests",
                           gdr_kind_name(kind));
                }
                gdr_wait_for_one_completion(channel, kind, outstanding);
                continue;
            }
            fatalf("GDR", __FILE__, __LINE__, "submit %s failed: %s",
                   gdr_kind_name(kind), gdr_error_string(rc).c_str());
        }
    }

    while (outstanding > 0) {
        gdr_wait_for_one_completion(channel, kind, outstanding);
    }
}

static GDRMeasurement measure_gdr_copy(const std::shared_ptr<GDRCopyChannel> &channel,
                                       void *dst, const void *src, size_t n_bytes,
                                       GDRCopyKind kind, int iters, int warmup)
{
    gdr_submit_batch_and_wait(channel, dst, src, n_bytes, kind, warmup);

    channel->reset_stats();

    GDRMeasurement result;
    auto start = std::chrono::steady_clock::now();
    gdr_submit_batch_and_wait(channel, dst, src, n_bytes, kind, iters);
    auto end = std::chrono::steady_clock::now();
    result.timing = make_batch_measurement(
        iters, std::chrono::duration<double, std::milli>(end - start).count());
    result.transport = channel->stats();
    return result;
}

static GDRMeasurement measure_gdr_d2h(const std::shared_ptr<GDRCopyChannel> &channel,
                                      float *d_buf, float *h_buf, size_t n_bytes,
                                      int iters, int warmup)
{
    return measure_gdr_copy(channel, h_buf, d_buf, n_bytes, GDR_D2H, iters, warmup);
}

static GDRMeasurement measure_gdr_h2d(const std::shared_ptr<GDRCopyChannel> &channel,
                                      float *d_buf, float *h_buf, size_t n_bytes,
                                      int iters, int warmup)
{
    return measure_gdr_copy(channel, d_buf, h_buf, n_bytes, GDR_H2D, iters, warmup);
}

struct ConcurrentResult {
    Stats mem_stats;
    Stats rs_stats;
    GDRStats gdr_stats;
};

static ConcurrentResult bench_concurrent_gdr_d2h_rs(
        const std::shared_ptr<GDRCopyChannel> &channel,
        float *d_buf, float *h_buf, size_t mem_bytes,
        float *send_buf, float *recv_buf, size_t p2p_bytes, int nranks,
        ncclComm_t comm, cudaStream_t stream_rs,
        int gpu_id, int iters, int warmup)
{
    GDRMeasurement mem_measurement;
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
            mem_measurement = measure_gdr_d2h(channel, d_buf, h_buf, mem_bytes,
                                              iters, warmup);
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
            ? compute_stats("GDR D2H (concurrent)", mem_measurement.timing.lats_ms,
                            mem_bytes, mem_measurement.timing.total_ms)
            : make_idle_stats("GDR D2H (concurrent, rank1 idle)"),
        compute_stats("ncclSend (concurrent)", rs_measurement.lats_ms,
                      p2p_bytes, rs_measurement.total_ms),
        do_memcpy ? mem_measurement.transport : GDRStats{},
    };
}

static ConcurrentResult bench_concurrent_gdr_h2d_rs(
        const std::shared_ptr<GDRCopyChannel> &channel,
        float *d_buf, float *h_buf, size_t mem_bytes,
        float *send_buf, float *recv_buf, size_t p2p_bytes, int nranks,
        ncclComm_t comm, cudaStream_t stream_rs,
        int gpu_id, int iters, int warmup)
{
    GDRMeasurement mem_measurement;
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
            mem_measurement = measure_gdr_h2d(channel, d_buf, h_buf, mem_bytes,
                                              iters, warmup);
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
            ? compute_stats("GDR H2D (concurrent)", mem_measurement.timing.lats_ms,
                            mem_bytes, mem_measurement.timing.total_ms)
            : make_idle_stats("GDR H2D (concurrent, rank1 idle)"),
        compute_stats("ncclSend (concurrent)", rs_measurement.lats_ms,
                      p2p_bytes, rs_measurement.total_ms),
        do_memcpy ? mem_measurement.transport : GDRStats{},
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

    double buf_kb = env_buffer_kb("BENCH_BUF_KB", "BENCH_BUF_GB", 4.0);
    double p2p_buf_kb = env_buffer_kb(
        "BENCH_P2P_BUF_KB", "BENCH_P2P_BUF_GB",
        env_buffer_kb("BENCH_AG_BUF_KB", "BENCH_AG_BUF_GB",
                      env_buffer_kb("BENCH_RS_BUF_KB", "BENCH_RS_BUF_GB", 4.0)));
    if (buf_kb <= 0.0 || p2p_buf_kb <= 0.0) {
        fatalf("CONFIG", __FILE__, __LINE__,
               "BENCH_BUF_KB and BENCH_P2P_BUF_KB must be positive");
    }
    int iters = env_int("BENCH_ITERS", 20);
    int warmup = env_int("BENCH_WARMUP", 5);
    int gpu_id = env_int("BENCH_GPU_ID", 0);
    bool gdr_use_odp = env_bool("BENCH_GDR_USE_ODP", false);
#if GDR_BENCH_ENABLE_QOS
    int gdr_ib_tc = env_traffic_class("BENCH_GDR_IB_TC", 0);
    int nccl_ib_tc = env_traffic_class("NCCL_IB_TC", 104);
    double qos_min_nccl_ratio = env_double_strict("BENCH_QOS_MIN_NCCL_RATIO", 0.99);
    if (qos_min_nccl_ratio <= 0.0 || qos_min_nccl_ratio > 1.0) {
        fatalf("CONFIG", __FILE__, __LINE__,
               "BENCH_QOS_MIN_NCCL_RATIO=%.6f out of range (expected 0 < ratio <= 1)",
               qos_min_nccl_ratio);
    }
    if (gdr_ib_tc == nccl_ib_tc) {
        fatalf("CONFIG", __FILE__, __LINE__,
               "BENCH_GDR_IB_TC=%d must differ from NCCL_IB_TC=%d for RoCE QoS validation",
               gdr_ib_tc, nccl_ib_tc);
    }
#else
    int gdr_ib_tc = 0;
#endif

    size_t mem_bytes = static_cast<size_t>(buf_kb * static_cast<double>(kBytesPerKiB));
    size_t p2p_bytes = static_cast<size_t>(p2p_buf_kb * static_cast<double>(kBytesPerKiB));
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
    std::string gdr_nic_name = normalize_nic_name(
        env_str("BENCH_GDR_NIC", env_str("NCCL_IB_HCA", nullptr)));
    if (rank == 0 && gdr_nic_name.empty()) {
        fatalf("CONFIG", __FILE__, __LINE__,
               "rank 0 requires BENCH_GDR_NIC or NCCL_IB_HCA to locate the GDR NIC");
    }

    ncclUniqueId nccl_id;
    if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    control_broadcast(&nccl_id, sizeof(nccl_id));

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, nccl_id, rank));

    cudaStream_t stream_rs;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_rs, cudaStreamNonBlocking));

    float *d_mem_buf = nullptr;
    float *h_buf = nullptr;
    std::shared_ptr<GDRCopyChannel> gdr_channel;
    if (rank == 0) {
        CUDA_CHECK(cudaMalloc(&d_mem_buf, mem_bytes));
        CUDA_CHECK(cudaMemset(d_mem_buf, 1, mem_bytes));
        CUDA_CHECK(cudaMallocHost(&h_buf, mem_bytes));
        memset(h_buf, 1, mem_bytes);

        try {
            gdr_channel = GDRCopyLib::open(
                gpu_id, gdr_nic_name, gdr_use_odp, static_cast<uint8_t>(gdr_ib_tc));
        } catch (const std::exception &ex) {
            fatalf("GDR", __FILE__, __LINE__,
                   "failed to open GDR channel on NIC %s: %s",
                   gdr_nic_name.c_str(), ex.what());
        }
#if GDR_BENCH_ENABLE_QOS
        if (!gdr_channel->is_roce()) {
            fatalf("CONFIG", __FILE__, __LINE__,
                   "gdr_qos_bench requires a RoCE/Ethernet link layer because it validates NCCL_IB_TC and GDR traffic_class");
        }
#endif
    }

    float *d_p2p_send = nullptr;
    float *d_p2p_recv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_p2p_send, p2p_bytes));
    CUDA_CHECK(cudaMalloc(&d_p2p_recv, p2p_bytes));
    CUDA_CHECK(cudaMemset(d_p2p_send, 1, p2p_bytes));
    CUDA_CHECK(cudaMemset(d_p2p_recv, 0, p2p_bytes));

    if (rank == 0) {
        printf("\n");
        printf("============================================================\n");
#if GDR_BENCH_ENABLE_QOS
        printf("  Cross-node GDR QoS H2D/D2H vs ncclSend/ncclRecv RoCE Benchmark\n");
#else
        printf("  Cross-node GDR H2D/D2H vs ncclSend/ncclRecv Benchmark\n");
#endif
        printf("============================================================\n");
        printf("  GPU          : %s  (device %d)\n", gpu_name, gpu_id);
        printf("  NCCL HCA     : %s\n", ib_hca);
        printf("  GDR NIC      : %s  (ODP=%s)\n",
               gdr_nic_name.c_str(), gdr_use_odp ? "on" : "off");
#if GDR_BENCH_ENABLE_QOS
        printf("  NCCL TC      : %d  (DSCP=%d)\n",
               nccl_ib_tc, traffic_class_to_dscp(nccl_ib_tc));
        printf("  GDR TC       : %u  (DSCP=%d)\n",
               static_cast<unsigned>(gdr_channel->traffic_class()),
               traffic_class_to_dscp(static_cast<int>(gdr_channel->traffic_class())));
        printf("  NCCL target  : %.2f%% of solo BW\n", qos_min_nccl_ratio * 100.0);
#endif
        printf("  Ranks        : %d  (1 GPU per node)\n", nranks);
        printf("  Mem copy     : rank0 only, GPUDirect RDMA H2D / D2H\n");
        printf("  Master       : %s:%d\n", master_addr, master_port);
        printf("  Mem buf      : %.1f KiB  (GDR H2D / D2H)\n", buf_kb);
        printf("  P2P buf      : %.1f KiB  (node0 send -> node1 recv)\n", p2p_buf_kb);
        printf("  Iterations   : %d  (warmup=%d)\n", iters, warmup);
        printf("============================================================\n\n");
    }
    control_barrier();

    if (rank == 0) {
        printf("Phase 1: Solo GDR D2H\n");
        printf("------------------------------------------------------------\n");
    }
    control_barrier();
    Stats solo_d2h = make_idle_stats("GDR D2H (solo)");
    GDRStats solo_d2h_gdr{};
    if (rank == 0) {
        GDRMeasurement solo_d2h_measurement =
            measure_gdr_d2h(gdr_channel, d_mem_buf, h_buf, mem_bytes, iters, warmup);
        solo_d2h = compute_stats("GDR D2H (solo)",
                                 solo_d2h_measurement.timing.lats_ms,
                                 mem_bytes, solo_d2h_measurement.timing.total_ms);
        solo_d2h_gdr = solo_d2h_measurement.transport;
    }
    if (rank == 0) {
        print_stats(solo_d2h);
        print_gdr_transport_stats("GDR D2H transport", solo_d2h_gdr);
        printf("\n");
    }
    control_barrier();

    if (rank == 0) {
        printf("Phase 2: Solo GDR H2D\n");
        printf("------------------------------------------------------------\n");
    }
    control_barrier();
    Stats solo_h2d = make_idle_stats("GDR H2D (solo)");
    GDRStats solo_h2d_gdr{};
    if (rank == 0) {
        GDRMeasurement solo_h2d_measurement =
            measure_gdr_h2d(gdr_channel, d_mem_buf, h_buf, mem_bytes, iters, warmup);
        solo_h2d = compute_stats("GDR H2D (solo)",
                                 solo_h2d_measurement.timing.lats_ms,
                                 mem_bytes, solo_h2d_measurement.timing.total_ms);
        solo_h2d_gdr = solo_h2d_measurement.transport;
    }
    if (rank == 0) {
        print_stats(solo_h2d);
        print_gdr_transport_stats("GDR H2D transport", solo_h2d_gdr);
        printf("\n");
    }
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
    if (rank == 0) {
        print_stats(solo_rs);
        printf("\n");
    }
    control_barrier();

    if (rank == 0) {
        printf("Phase 4: Concurrent GDR D2H + ncclSend\n");
        printf("------------------------------------------------------------\n");
    }
    control_barrier();
    ConcurrentResult cr_d2h = bench_concurrent_gdr_d2h_rs(
        gdr_channel, d_mem_buf, h_buf, mem_bytes,
        d_p2p_send, d_p2p_recv, p2p_bytes, nranks,
        comm, stream_rs, gpu_id, iters, warmup);
    if (rank == 0) {
        print_stats(cr_d2h.mem_stats);
        print_gdr_transport_stats("GDR D2H transport", cr_d2h.gdr_stats);
        print_stats(cr_d2h.rs_stats);
#if GDR_BENCH_ENABLE_QOS
        print_qos_check("D2H + ncclSend", solo_rs, cr_d2h.rs_stats, qos_min_nccl_ratio);
#endif
        printf("\n");
    }
    control_barrier();

    if (rank == 0) {
        printf("Phase 5: Concurrent GDR H2D + ncclSend\n");
        printf("------------------------------------------------------------\n");
    }
    control_barrier();
    ConcurrentResult cr_h2d = bench_concurrent_gdr_h2d_rs(
        gdr_channel, d_mem_buf, h_buf, mem_bytes,
        d_p2p_send, d_p2p_recv, p2p_bytes, nranks,
        comm, stream_rs, gpu_id, iters, warmup);
    if (rank == 0) {
        print_stats(cr_h2d.mem_stats);
        print_gdr_transport_stats("GDR H2D transport", cr_h2d.gdr_stats);
        print_stats(cr_h2d.rs_stats);
#if GDR_BENCH_ENABLE_QOS
        print_qos_check("H2D + ncclSend", solo_rs, cr_h2d.rs_stats, qos_min_nccl_ratio);
#endif
        printf("\n");
    }
    control_barrier();

    if (rank == 0) {
        printf("============================================================\n");
        printf("  Interference Summary (rank 0 local view, >5%% BW drop or latency increase)\n");
        printf("============================================================\n");
        printf("  [GDR D2H + ncclSend concurrently vs solo]\n");
        print_delta("GDR D2H", solo_d2h, cr_d2h.mem_stats);
        print_delta("Send (vs D2H)", solo_rs, cr_d2h.rs_stats);
        printf("\n");
        printf("  [GDR H2D + ncclSend concurrently vs solo]\n");
        print_delta("GDR H2D", solo_h2d, cr_h2d.mem_stats);
        print_delta("Send (vs H2D)", solo_rs, cr_h2d.rs_stats);
#if GDR_BENCH_ENABLE_QOS
        printf("\n");
        printf("  [QoS Validation]\n");
        print_qos_check("D2H + ncclSend", solo_rs, cr_d2h.rs_stats, qos_min_nccl_ratio);
        print_qos_check("H2D + ncclSend", solo_rs, cr_h2d.rs_stats, qos_min_nccl_ratio);
#endif
        printf("============================================================\n\n");
    }

    if (rank == 0) {
        gdr_channel.reset();
        GDRCopyLib::shutdown();
    }
    if (d_mem_buf) CUDA_CHECK(cudaFree(d_mem_buf));
    if (h_buf) CUDA_CHECK(cudaFreeHost(h_buf));
    if (d_p2p_send) CUDA_CHECK(cudaFree(d_p2p_send));
    if (d_p2p_recv) CUDA_CHECK(cudaFree(d_p2p_recv));
    CUDA_CHECK(cudaStreamDestroy(stream_rs));
    NCCL_CHECK(ncclCommDestroy(comm));
    finalize_control_plane();
    return 0;
}
