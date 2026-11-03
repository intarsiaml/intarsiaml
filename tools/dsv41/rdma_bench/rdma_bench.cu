// Lane dsv41/rdma: one-shot exchange latency of a direct RDMA all-reduce against NCCL's current path, two ranks (the two
// DGX Sparks over the CX7 link), both inside a captured CUDA graph as the decode pass runs them.
//
// What is measured. The engine's collective is `all_reduce_sum_into`: every rank has an f32 partial `[n]`, the result is
// bf16(partial0 + partial1) (a + b is commutative, so both ranks and both transports give the same bits). Two paths:
//   nccl  ncclAllReduce f32 (out of place, scratch) + a round kernel   = Comm::all_reduce_sum_into, default settings
//   rdma  stage kernel (copy the partial into a pinned host ring slot, ring a doorbell) -> a host proxy thread posts one
//         RDMA write-with-immediate to the peer's receive ring -> the peer's proxy sees the completion and sets a flag in
//         pinned host memory -> the collect kernel waits for the flag (ld.acquire.sys), sums own + peer, rounds to bf16.
//         The design is TensorFold's (zig/src/families/dsv41/rdma.zig, kernels/cuda/dsv41/rdma_gather.cu), written afresh
//         here with an all-reduce collect. No GPUDirect, no peermem, no driver setting: libibverbs on RoCE + cuMemHostRegister.
//   null  two empty kernels in place of the exchange: the floor of the harness (spin kernel, 2 launches, stamp).
// Every call is `[spin kernel: GAP us of device time, stamps t_start] [exchange] [stamp t_done]`; latency = t_done - t_start
// on the same GPU's globaltimer. K calls per graph, R replays. A call's latency on a rank includes waiting for the peer
// when the peer started later (skew); the per-iteration minimum over the two ranks is the skew-free exchange cost (the
// later rank never waits), the maximum includes the skew. Rank 1 sends its stamps to rank 0, which prints the table.
//
//   rank 1:  rdma_bench 1 LISTEN_IP PORT IBDEV GID [opts]
//   rank 0:  rdma_bench 0 PEER_IP   PORT IBDEV GID [opts]       (NCCL_* env as for the engine)
// opts: --modes nccl,rdma,null  --sizes 20480,40960,81920  --gaps 0,500  --reps N (replays)  --k N (calls a graph)
//       --pdl 0|1  --stage-blocks N  --collect-blocks N  --pin CPU  --slots N  --check 0|1
#include <arpa/inet.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <nccl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(3); } } while (0)
#define NK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, ncclGetErrorString(r_)); exit(3); } } while (0)
#define DIE(...) do { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); exit(3); } while (0)

typedef unsigned long long u64;

// ------------------------------------------------------------------------------------------------------------------
// device side
// ------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ u64 ld_acq(const u64* p) { u64 v; asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rlx(u64* p, u64 v) { asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory"); }
__device__ __forceinline__ u64 now_ns() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void pdl_enter(int pdl) {
#if __CUDA_ARCH__ >= 900
    if (pdl) { asm volatile("griddepcontrol.wait;" ::: "memory"); asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
#endif
}

struct Stamps { u64* t_start; u64* t_done; u64* ctr; unsigned* done; };  // device arrays; ctr = index of the call in flight

// 1 thread: wait GAP ns (stands for the compute between two collectives), stamp the start of call *ctr.
__global__ void spin_kernel(u64 gap_ns, Stamps st, int pdl) {
#if __CUDA_ARCH__ >= 900
    if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // lets the next kernel launch while this one still computes (as a producer kernel would)
#endif
    u64 t0 = now_ns();
    if (gap_ns) while (now_ns() - t0 < gap_ns) {}
    st.t_start[*st.ctr] = now_ns();
}

// NCCL path's last kernel / null path: f32 -> bf16 (round to nearest even), stamp the end of call *ctr.
__global__ void round_stamp_kernel(const float* src, __nv_bfloat16* dst, int n, Stamps st) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = __float2bfloat16_rn(src[i]);
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(st.done, 1u) == gridDim.x - 1) { *st.done = 0; st.t_done[*st.ctr] = now_ns(); *st.ctr += 1; }
    }
}
__global__ void null_kernel() {}

// DRAM load beside the exchange (the decode pass streams ~250 GB/s of weights through the same LPDDR): read a big buffer
// in a loop until the host sets the stop word (pinned host memory).
__global__ void hog_kernel(const float4* buf, size_t n4, const volatile u64* stop, float* sink) {
    float acc = 0; size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; const size_t stride = (size_t)gridDim.x * blockDim.x;
    while (!*stop) { for (int k = 0; k < 64; ++k) { float4 v = __ldcg(buf + i); acc += v.x + v.y + v.z + v.w; i += stride; if (i >= n4) i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; } }
    if (acc == 12345.f) *sink = acc;
}

struct RingDev {
    char* send; char* recv;          // device pointers to the pinned host rings
    u64* flags; u64* doorbell; u64* sizes; u64* sent; u64* abort_word;   // pinned host metadata
    u64* seq; unsigned* staged;      // device scalars
    int slots; int rank; int world; u64 maxb;
    u64* t_bell; u64* t_flag;        // per-call probes (rank-local globaltimer), indexed by seq
    u64 deadline_ns;                 // a spin gives up (sets nothing, just leaves) after this long: a dead peer never wedges the GPU
};

__global__ void __launch_bounds__(512) rdma_stage(const float4* __restrict__ src, int n4, RingDev r, int pdl) {
    __shared__ u64 s_seq; __shared__ int s_last;
    pdl_enter(pdl);
    if (threadIdx.x == 0) {
        const u64 q = *r.seq + 1; const int slot = (int)(q % r.slots);
        if (q > (u64)r.slots) { u64 t0 = now_ns(); for (unsigned i = 1; ld_acq(r.sent + slot) + r.slots < q; ++i) if ((i & 1023) == 0 && (ld_acq(r.abort_word) || now_ns() - t0 > r.deadline_ns)) break; }
        s_seq = q;
        if (blockIdx.x == 0) st_rlx(r.sizes + slot, (u64)n4 * 16);
    }
    __syncthreads();
    const u64 q = s_seq; const int slot = (int)(q % r.slots);
    float4* dst = reinterpret_cast<float4*>(r.send + (size_t)slot * r.maxb);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) dst[i] = src[i];
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence_system(); s_last = atomicAdd(r.staged, 1u) == gridDim.x - 1; }
    __syncthreads();
    if (threadIdx.x == 0 && s_last) {
        __threadfence_system();
        *r.staged = 0; *r.seq = q;
        st_rlx(r.doorbell, q);
        r.t_bell[q & 0xffff] = now_ns();
    }
}

// wait for the peer's flag at this sequence, then out = bf16(own + peer) (own + peer == peer + own bitwise)
__global__ void __launch_bounds__(256) rdma_collect_sum(const float4* __restrict__ own, __nv_bfloat162* __restrict__ out, int n4, RingDev r, int pdl, Stamps st, int ahead) {
    __shared__ int s_last;
    pdl_enter(pdl);
    const u64 q = *r.seq + ahead; const int slot = (int)(q % r.slots);   // ahead = 1: this rank collects before it stages (ping-pong, rank 1)
    const int peer = 1 - r.rank;
    if (threadIdx.x == 0) { u64 t0 = now_ns(); for (unsigned i = 1; ld_acq(r.flags + (size_t)slot * r.world + peer) < q; ++i) if ((i & 1023) == 0 && (ld_acq(r.abort_word) || now_ns() - t0 > r.deadline_ns)) break; }
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) r.t_flag[q & 0xffff] = now_ns();
    const float4* in = reinterpret_cast<const float4*>(r.recv + ((size_t)slot * r.world + peer) * r.maxb);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        const float4 a = own[i], b = __ldcv(in + i);
        out[2 * i] = __floats2bfloat162_rn(a.x + b.x, a.y + b.y);
        out[2 * i + 1] = __floats2bfloat162_rn(a.z + b.z, a.w + b.w);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        s_last = atomicAdd(st.done, 1u) == gridDim.x - 1;
        if (s_last) { *st.done = 0; st.t_done[*st.ctr] = now_ns(); *st.ctr += 1; }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// host side: sockets
// ------------------------------------------------------------------------------------------------------------------
static int g_sock = -1;
static void sock_all(const void* p, size_t n, bool wr) {
    size_t d = 0;
    while (d < n) { ssize_t k = wr ? write(g_sock, (const char*)p + d, n - d) : read(g_sock, (char*)p + d, n - d); if (k <= 0) DIE("socket %s failed", wr ? "write" : "read"); d += (size_t)k; }
}
static void sock_open(int rank, const char* ip, int port) {
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port); inet_pton(AF_INET, ip, &a.sin_addr);
    if (rank == 1) {
        int l = socket(AF_INET, SOCK_STREAM, 0), yes = 1; setsockopt(l, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);
        if (bind(l, (sockaddr*)&a, sizeof a) || listen(l, 1)) DIE("bind/listen %s:%d", ip, port);
        g_sock = accept(l, nullptr, nullptr); close(l);
    } else {
        for (int i = 0; i < 600; ++i) { g_sock = socket(AF_INET, SOCK_STREAM, 0); if (!connect(g_sock, (sockaddr*)&a, sizeof a)) break; close(g_sock); g_sock = -1; usleep(100000); }
        if (g_sock < 0) DIE("connect %s:%d", ip, port);
    }
    int one = 1; setsockopt(g_sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
}
static void barrier() { char b = 1, c = 0; sock_all(&b, 1, true); sock_all(&c, 1, false); }
template <class T> static void swap_with_peer(const T& mine, T& theirs, int rank) {
    if (rank == 0) { sock_all(&mine, sizeof(T), true); sock_all(&theirs, sizeof(T), false); } else { sock_all(&theirs, sizeof(T), false); sock_all(&mine, sizeof(T), true); }
}

// ------------------------------------------------------------------------------------------------------------------
// host side: the ring (verbs + pinned memory + proxy thread)
// ------------------------------------------------------------------------------------------------------------------
struct Info { uint32_t qpn, psn, rkey; uint8_t gid[16]; int mtu; uint64_t recv; };
struct InfoSet { int nd; Info i[2]; };

struct Pinned {
    char* host = nullptr; char* dev = nullptr; size_t len = 0;
    void alloc(size_t n) {
        len = (n + 65535) & ~(size_t)65535;
        host = (char*)mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (host == MAP_FAILED) DIE("mmap");
        memset(host, 0, len); mlock(host, len);
        CK(cudaHostRegister(host, len, cudaHostRegisterPortable | cudaHostRegisterMapped));
        CK(cudaHostGetDevicePointer((void**)&dev, host, 0));
    }
};

struct Nic { ibv_context* ctx; ibv_pd* pd; ibv_cq* cq; ibv_mr *mr_send, *mr_recv; ibv_qp* qp; int mtu; uint8_t gid[16]; };

struct Ring {
    int rank = 0, world = 2, slots = 4; size_t maxb = 0;
    Pinned send, recv, meta; u64* d_scalars = nullptr;
    u64 *flags, *doorbell, *sizes, *sent, *abort_word;
    Nic nic[2]; int nd = 0; int gid_index = 3;
    InfoSet peer{}; std::thread proxy; std::atomic<bool> running{false}; std::atomic<bool> failed{false}; int pin_cpu = -1;

    static constexpr size_t LINE = 128;
    void layout() {   // every hot word on its own 128-byte line: flags (host writes, GPU polls), doorbell (GPU writes, host polls)
        char* m = meta.host; auto take = [&](size_t bytes) { u64* p = (u64*)m; m += (bytes + LINE - 1) / LINE * LINE; return p; };
        flags = take(slots * world * 8); doorbell = take(8); sizes = take(slots * 8); sent = take(slots * 8); abort_word = take(8);
    }
    void open(const std::vector<std::string>& devs, size_t max_bytes, int nslots, int gid_idx) {
        slots = nslots; maxb = max_bytes; gid_index = gid_idx; nd = (int)devs.size();
        send.alloc(slots * maxb); recv.alloc((size_t)slots * world * maxb); meta.alloc(16 * LINE + slots * world * 8 + 4 * slots * 8);
        layout();
        CK(cudaMalloc(&d_scalars, 64)); CK(cudaMemset(d_scalars, 0, 64));
        int n = 0; ibv_device** l = ibv_get_device_list(&n);
        for (int k = 0; k < nd; ++k) {
            ibv_device* d = nullptr;
            for (int i = 0; i < n; ++i) if (devs[k] == ibv_get_device_name(l[i])) d = l[i];
            if (!d) DIE("no RDMA device %s", devs[k].c_str());
            Nic& c = nic[k];
            c.ctx = ibv_open_device(d); if (!c.ctx) DIE("ibv_open_device");
            ibv_port_attr pa{}; if (ibv_query_port(c.ctx, 1, &pa)) DIE("query_port"); c.mtu = pa.active_mtu;
            c.pd = ibv_alloc_pd(c.ctx); c.cq = ibv_create_cq(c.ctx, 4096, nullptr, nullptr, 0);
            int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
            c.mr_send = ibv_reg_mr(c.pd, send.host, send.len, acc); c.mr_recv = ibv_reg_mr(c.pd, recv.host, recv.len, acc);
            if (!c.mr_send || !c.mr_recv) DIE("ibv_reg_mr (memlock?)");
            ibv_gid g; if (ibv_query_gid(c.ctx, 1, gid_index, &g)) DIE("query_gid"); memcpy(c.gid, g.raw, 16);
            ibv_qp_init_attr qa{}; qa.send_cq = c.cq; qa.recv_cq = c.cq; qa.qp_type = IBV_QPT_RC; qa.cap = {512, 256, 1, 1, 0};
            c.qp = ibv_create_qp(c.pd, &qa); if (!c.qp) DIE("ibv_create_qp");
            ibv_qp_attr a{}; a.qp_state = IBV_QPS_INIT; a.pkey_index = 0; a.port_num = 1; a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
            if (ibv_modify_qp(c.qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) DIE("qp INIT");
            for (int i = 0; i < 256; ++i) post_recv(c);
        }
    }
    void post_recv(Nic& c) { ibv_recv_wr wr{}, *bad; if (ibv_post_recv(c.qp, &wr, &bad)) DIE("post_recv"); }
    InfoSet mine() const {
        InfoSet o{}; o.nd = nd;
        for (int k = 0; k < nd; ++k) { const Nic& c = nic[k]; Info& i = o.i[k]; i.qpn = c.qp->qp_num; i.psn = (c.qp->qp_num * 2654435761u) & 0xFFFFFF; i.rkey = c.mr_recv->rkey; memcpy(i.gid, c.gid, 16); i.mtu = c.mtu; i.recv = (uint64_t)recv.host; }
        return o;
    }
    void connect(const InfoSet& p) {
        peer = p; if (p.nd != nd) DIE("rail count differs");
        for (int k = 0; k < nd; ++k) {
            Nic& c = nic[k]; const Info& pi = p.i[k]; int m = std::min(c.mtu, pi.mtu);
            ibv_qp_attr a{}; a.qp_state = IBV_QPS_RTR; a.path_mtu = (ibv_mtu)m; a.dest_qp_num = pi.qpn; a.rq_psn = pi.psn; a.max_dest_rd_atomic = 1; a.min_rnr_timer = 12;
            a.ah_attr.is_global = 1; memcpy(a.ah_attr.grh.dgid.raw, pi.gid, 16); a.ah_attr.grh.sgid_index = gid_index; a.ah_attr.grh.hop_limit = 64; a.ah_attr.port_num = 1;
            if (ibv_modify_qp(c.qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) DIE("qp RTR");
            memset(&a, 0, sizeof a); a.qp_state = IBV_QPS_RTS; a.timeout = 14; a.retry_cnt = 7; a.rnr_retry = 7; a.sq_psn = mine().i[k].psn; a.max_rd_atomic = 1;
            if (ibv_modify_qp(c.qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) DIE("qp RTS");
        }
    }
    void start() { running = true; proxy = std::thread([this] { loop(); }); }
    void stop() { running = false; if (proxy.joinable()) proxy.join(); }
    void fail(const char* why) { fprintf(stderr, "rank %d: RDMA failure: %s\n", rank, why); failed = true; __atomic_store_n(abort_word, 1ull, __ATOMIC_RELEASE); }
    void loop() {
        if (pin_cpu >= 0) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(pin_cpu, &s); sched_setaffinity(0, sizeof s, &s); }
        u64 posted = 0, published = 0; u64 got[2] = {0, 0}; unsigned sent_cnt[16] = {0}; u64 sent_q[16] = {0}; ibv_wc wc[16];
        while (running.load(std::memory_order_relaxed)) {
            const u64 ring = __atomic_load_n(doorbell, __ATOMIC_ACQUIRE);
            while (posted < ring) {
                const u64 q = posted + 1; const int slot = (int)(q % slots);
                const u64 n = __atomic_load_n(sizes + slot, __ATOMIC_ACQUIRE);
                const u64 unit = n / 16 / nd * 16;   // rail d writes [unit * d, unit * (d + 1)); the last rail takes the rest
                for (int d = 0; d < nd; ++d) {
                    const u64 off = unit * d, len = d == nd - 1 ? n - off : unit; Nic& c = nic[d];
                    ibv_sge sge{(uint64_t)(send.host + (size_t)slot * maxb + off), (uint32_t)len, c.mr_send->lkey};
                    ibv_send_wr wr{}, *bad; wr.wr_id = q; wr.sg_list = &sge; wr.num_sge = len ? 1 : 0; wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM; wr.send_flags = IBV_SEND_SIGNALED;
                    wr.imm_data = htonl((uint32_t)q); wr.wr.rdma.remote_addr = peer.i[d].recv + ((size_t)slot * world + rank) * maxb + off; wr.wr.rdma.rkey = peer.i[d].rkey;
                    if (ibv_post_send(c.qp, &wr, &bad)) { fail("ibv_post_send"); return; }
                }
                posted = q;
            }
            for (int d = 0; d < nd; ++d) {
                Nic& c = nic[d];
                const int k = ibv_poll_cq(c.cq, 16, wc);
                if (k < 0) { fail("ibv_poll_cq"); return; }
                for (int i = 0; i < k; ++i) {
                    if (wc[i].status != IBV_WC_SUCCESS) { fail(ibv_wc_status_str(wc[i].status)); return; }
                    if (wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
                        u64 q = (got[d] & ~0xffffffffull) | ntohl(wc[i].imm_data); if (q <= got[d]) q += 1ull << 32; got[d] = q;
                        u64 in = got[0]; for (int e = 1; e < nd; ++e) in = std::min(in, got[e]);   // every rail has delivered up to `in`
                        for (u64 x = published + 1; x <= in; ++x) __atomic_store_n(flags + (size_t)(x % slots) * world + (1 - rank), x, __ATOMIC_RELEASE);
                        if (in > published) published = in;
                        post_recv(c);
                    } else {
                        const u64 q = wc[i].wr_id; const int sl = (int)(q % slots);
                        if (sent_q[sl] != q) { sent_q[sl] = q; sent_cnt[sl] = 0; }
                        if (++sent_cnt[sl] == (unsigned)nd) __atomic_store_n(sent + sl, q, __ATOMIC_RELEASE);
                    }
                }
            }
        }
    }
    RingDev dev(u64* t_bell, u64* t_flag, u64 deadline_ns) const {
        RingDev r{}; r.send = send.dev; r.recv = recv.dev; auto m = [&](u64* p) { return (u64*)(meta.dev + ((char*)p - meta.host)); };
        r.flags = m(flags); r.doorbell = m(doorbell); r.sizes = m(sizes); r.sent = m(sent); r.abort_word = m(abort_word);
        r.seq = d_scalars; r.staged = (unsigned*)(d_scalars + 1); r.slots = slots; r.rank = rank; r.world = world; r.maxb = maxb;
        r.t_bell = t_bell; r.t_flag = t_flag; r.deadline_ns = deadline_ns; return r;
    }
};

// ------------------------------------------------------------------------------------------------------------------
// the bench
// ------------------------------------------------------------------------------------------------------------------
template <class... KArgs, class... Args>
static void launch(void (*k)(KArgs...), dim3 g, dim3 b, cudaStream_t s, int pdl, Args... args) {
    cudaLaunchAttribute a[1]; a[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; a[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t c{}; c.gridDim = g; c.blockDim = b; c.stream = s; c.attrs = a; c.numAttrs = pdl ? 1 : 0;
    CK(cudaLaunchKernelEx(&c, k, (KArgs)args...));
}

static std::vector<std::string> split(const std::string& s) { std::vector<std::string> o; size_t i = 0; while (i <= s.size()) { size_t j = s.find(',', i); if (j == std::string::npos) j = s.size(); if (j > i) o.push_back(s.substr(i, j - i)); i = j + 1; } return o; }
static double pct(std::vector<double>& v, double f) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(v.size() * f))]; }

int main(int argc, char** argv) {
    if (argc < 6) DIE("usage: rdma_bench RANK IP PORT IBDEV GID [opts]");
    const int rank = atoi(argv[1]); const char* ip = argv[2]; const int port = atoi(argv[3]); const char* ibdev = argv[4]; const int gid = atoi(argv[5]);
    std::string modes = "nccl,rdma,null", sizes = "20480,40960,81920"; std::string gaps = "0,500";
    int hog = 0, reps = 40, K = 64, pdl = 0, sblocks = 16, cblocks = 0, pin = -1, slots = 4, check = 1, skip = 2;
    for (int i = 6; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--modes") modes = v; else if (k == "--sizes") sizes = v; else if (k == "--gaps") gaps = v; else if (k == "--reps") reps = atoi(v.c_str());
        else if (k == "--k") K = atoi(v.c_str()); else if (k == "--pdl") pdl = atoi(v.c_str()); else if (k == "--stage-blocks") sblocks = atoi(v.c_str());
        else if (k == "--collect-blocks") cblocks = atoi(v.c_str()); else if (k == "--pin") pin = atoi(v.c_str()); else if (k == "--slots") slots = atoi(v.c_str());
        else if (k == "--check") check = atoi(v.c_str()); else if (k == "--hog") hog = atoi(v.c_str()); else DIE("unknown option %s", k.c_str());
    }
    CK(cudaSetDevice(0));
    sock_open(rank, ip, port);
    size_t maxn = 0; for (auto& s : split(sizes)) maxn = std::max(maxn, (size_t)atol(s.c_str()));
    // NCCL
    ncclUniqueId id; if (rank == 0) NK(ncclGetUniqueId(&id)); if (rank == 0) sock_all(&id, sizeof id, true); else sock_all(&id, sizeof id, false);
    ncclComm_t comm; NK(ncclCommInitRank(&comm, 2, id, rank));
    int ver; ncclGetVersion(&ver);
    // RDMA
    Ring ring; ring.rank = rank; ring.pin_cpu = pin;
    ring.open(split(ibdev), (maxn + 63) & ~(size_t)63, slots, gid);
    InfoSet pi; InfoSet mi = ring.mine(); swap_with_peer(mi, pi, rank);
    ring.connect(pi); barrier(); ring.start();
    printf("rank %d: NCCL %d, RDMA %s gid %d mtu %d, proxy pin %d, pdl %d, stage blocks %d, slots %d, %zu max bytes\n", rank, ver, ibdev, gid, ring.nic[0].mtu, pin, pdl, sblocks, slots, maxn);

    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const size_t maxcalls = (size_t)K * reps * 4 + 64;
    float *d_send, *d_tmp; __nv_bfloat16* d_out; CK(cudaMalloc(&d_send, maxn)); CK(cudaMalloc(&d_tmp, maxn)); CK(cudaMalloc(&d_out, maxn));
    u64 *d_ts, *d_td, *d_ctr, *d_bell, *d_flag; unsigned* d_done;
    CK(cudaMalloc(&d_ts, maxcalls * 8)); CK(cudaMalloc(&d_td, maxcalls * 8)); CK(cudaMalloc(&d_bell, 65536 * 8)); CK(cudaMalloc(&d_flag, 65536 * 8));
    CK(cudaMalloc(&d_ctr, 8)); CK(cudaMalloc(&d_done, 4)); CK(cudaMemset(d_done, 0, 4));
    Stamps st{d_ts, d_td, d_ctr, d_done};
    // payload: rank-specific values so a wrong sum cannot pass; values small enough that a + b is exact in f32 but not in bf16
    {
        std::vector<float> h(maxn / 4); for (size_t i = 0; i < h.size(); ++i) h[i] = (float)((i * 2654435761u + rank * 97u) % 10007) * 0.0123f + (rank ? 0.001f : -0.5f);
        CK(cudaMemcpy(d_send, h.data(), maxn, cudaMemcpyHostToDevice));
    }
    cudaStream_t hs_ = nullptr; u64 *h_stop = nullptr, *hog_stop = nullptr; float4* hog_buf = nullptr; float* hog_sink = nullptr; size_t hog_n4 = 0; float4* d_hog = nullptr; float* d_sink = nullptr;
    if (hog) {   // hog blocks of 1024 threads on a second stream over a 3 GB buffer (a pass reads ~3.5 GB of weights)
        hog_n4 = (3ull << 30) / 16; CK(cudaMalloc(&d_hog, hog_n4 * 16)); CK(cudaMemset(d_hog, 1, hog_n4 * 16)); CK(cudaMalloc(&d_sink, 4));
        CK(cudaHostAlloc((void**)&h_stop, 8, cudaHostAllocMapped)); *h_stop = 0; u64* d_stop; CK(cudaHostGetDevicePointer((void**)&d_stop, h_stop, 0));
        CK(cudaStreamCreateWithFlags(&hs_, cudaStreamNonBlocking));
        hog_stop = d_stop; hog_buf = d_hog; hog_sink = d_sink;
        printf("rank %d: DRAM hog on %d blocks during the replays\n", rank, hog);
    }
    const u64 deadline_ns = 2000000000ull;   // every device spin gives up after 2 s
    RingDev rd = ring.dev(d_bell, d_flag, deadline_ns);
    u64 calls_total = 0;   // calls issued so far on this rank (the same on both)

    u64 qcount = 0;   // rdma sequences consumed so far (the same on both ranks)
    printf("%-5s %-8s %-7s %-9s %9s %9s %9s %9s %9s %9s %s\n", "path", "bytes", "gap_us", "who", "p50_us", "p90_us", "p99_us", "mean_us", "min_us", "max_us", "n");
    for (auto& gs : split(gaps)) for (auto& ss : split(sizes)) for (auto& mode : split(modes)) {
        const u64 gap_ns = (u64)atol(gs.c_str()) * 1000; const size_t bytes = (size_t)atol(ss.c_str()); const int n = (int)(bytes / 4), n4 = n / 4;
        const int sb = sblocks, cb = cblocks ? cblocks : std::max(1, std::min(48, (n4 + 255) / 256));
        auto issue = [&](int m) {   // one call
            spin_kernel<<<1, 1, 0, s>>>(gap_ns, st, pdl);
            if (m == 0) {   // nccl
                NK(ncclAllReduce(d_send, d_tmp, n, ncclFloat, ncclSum, comm, s));
                round_stamp_kernel<<<std::max(1, std::min(8, n / 1024)), 256, 0, s>>>(d_tmp, d_out, n, st);
            } else if (m == 1) {   // rdma
                launch(rdma_stage, dim3(std::max(1, std::min(sb, (n4 + 1023) / 1024))), dim3(512), s, pdl, (const float4*)d_send, n4, rd, pdl);
                launch(rdma_collect_sum, dim3(cb), dim3(256), s, pdl, (const float4*)d_send, (__nv_bfloat162*)d_out, n4, rd, pdl, st, 0);
            } else if (m == 3) {   // ping-pong: rank 0 stages then collects, rank 1 collects then stages; RTT / 2 = the one-way latency
                auto stg = [&] { launch(rdma_stage, dim3(std::max(1, std::min(sb, (n4 + 1023) / 1024))), dim3(512), s, pdl, (const float4*)d_send, n4, rd, pdl); };
                auto col = [&] { launch(rdma_collect_sum, dim3(cb), dim3(256), s, pdl, (const float4*)d_send, (__nv_bfloat162*)d_out, n4, rd, pdl, st, rank == 0 ? 0 : 1); };
                if (rank == 0) { stg(); col(); } else { col(); stg(); }
            } else {        // null
                null_kernel<<<1, 32, 0, s>>>();
                round_stamp_kernel<<<std::max(1, std::min(8, n / 1024)), 256, 0, s>>>(d_send, d_out, 0, st);
            }
        };
        const int m = mode == "nccl" ? 0 : mode == "rdma" ? 1 : mode == "pp" ? 3 : 2;
        if ((m == 1 || m == 3) && n4 * 16 > (int)ring.maxb) DIE("size too large");
        // warm-up (eager) then capture
        barrier(); CK(cudaMemset(d_ctr, 0, 8));
        for (int i = 0; i < 20; ++i) issue(m);
        CK(cudaStreamSynchronize(s)); barrier();
        if (m == 1 || m == 3) qcount += 20;
        // the stage kernel probes are indexed by seq; the stamps by *ctr. Reset the stamp counter, keep seq.
        CK(cudaMemset(d_ctr, 0, 8));
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < K; ++i) issue(m);
        CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
        CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s)); CK(cudaMemset(d_ctr, 0, 8)); barrier();
        if (m == 1 || m == 3) qcount += K;
        const u64 q0 = qcount;   // the replays use sequences q0 + 1 ...
        std::vector<double> sync_us;
        if (hog) { *(volatile u64*)h_stop = 0; hog_kernel<<<hog, 1024, 0, hs_>>>(hog_buf, hog_n4, hog_stop, hog_sink); usleep(20000); }
        for (int r = 0; r < reps; ++r) {
            barrier();
            CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
        }
        if (hog) { *(volatile u64*)h_stop = 1; CK(cudaStreamSynchronize(hs_)); }
        if (m == 1 || m == 3) qcount += (u64)reps * K;
        // stamps: reps * K entries (the pre-capture replay was reset)
        const size_t N = (size_t)reps * K;
        std::vector<u64> hs(N), hd(N);
        CK(cudaMemcpy(hs.data(), d_ts, N * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hd.data(), d_td, N * 8, cudaMemcpyDeviceToHost));
        std::vector<double> lat; lat.reserve(N);
        for (size_t i = 0; i < N; ++i) lat.push_back(((double)hd[i] - (double)hs[i]) / 1e3);
        if (m == 1) {   // where the time goes on this rank: stage kernel start -> doorbell, doorbell -> peer's flag seen, flag -> result written
            std::vector<u64> hb(65536), hf(65536); CK(cudaMemcpy(hb.data(), d_bell, 65536 * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hf.data(), d_flag, 65536 * 8, cudaMemcpyDeviceToHost));
            std::vector<double> a, b, c;
            for (size_t i = 0; i < N; ++i) { if ((int)(i % K) < skip) continue; const u64 q = (q0 + 1 + i) & 0xffff; a.push_back(((double)hb[q] - (double)hs[i]) / 1e3); b.push_back(((double)hf[q] - (double)hb[q]) / 1e3); c.push_back(((double)hd[i] - (double)hf[q]) / 1e3); }
            printf("rank %d: %zu B gap %s: stage start->doorbell p50 %.1f us, doorbell->flag p50 %.1f (p10 %.1f), flag->done p50 %.1f\n", rank, bytes, gs.c_str(), pct(a, .5), pct(b, .5), pct(b, .1), pct(c, .5));
        }
        // pair with the peer: min / max over the two ranks per call (skip the first `skip` calls of every replay: start skew)
        std::vector<double> peer_lat(N);
        if (rank == 1) sock_all(lat.data(), N * 8, true); else sock_all(peer_lat.data(), N * 8, false);
        if (rank == 0) {
            std::vector<double> a, b, mn, mx;
            for (size_t i = 0; i < N; ++i) {
                if ((int)(i % K) < skip) continue;
                a.push_back(lat[i]); b.push_back(peer_lat[i]); mn.push_back(std::min(lat[i], peer_lat[i])); mx.push_back(std::max(lat[i], peer_lat[i]));
            }
            auto row = [&](const char* who, std::vector<double>& v) {
                double sum = 0, lo = 1e18, hi = 0; for (double x : v) { sum += x; lo = std::min(lo, x); hi = std::max(hi, x); }
                double p50 = pct(v, .5), p90 = pct(v, .9), p99 = pct(v, .99);
                printf("%-5s %-8zu %-7s %-9s %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %zu\n", mode.c_str(), bytes, gs.c_str(), who, p50, p90, p99, sum / v.size(), lo, hi, v.size());
            };
            if (m == 3) { std::vector<double> h; for (size_t i = 0; i < N; ++i) if ((int)(i % K) >= skip) h.push_back(lat[i] / 2); row("pp_rtt/2", h); }
            else {
                std::vector<double> avg; for (size_t i = 0; i < a.size(); ++i) avg.push_back((a[i] + b[i]) / 2);
                row("mean(r0,r1)", avg); row("min(r0,r1)", mn); row("rank0", a); row("rank1", b); row("max(r0,r1)", mx);
            }
            fflush(stdout);
        }
        // correctness: the bf16 result of this path against the NCCL path's, bitwise (both ranks, own data)
        if (check && m == 1 && bytes == maxn) {
            // run each path once on the same inputs, copy the outputs
            std::vector<uint16_t> o_nccl(n), o_rdma(n);
            CK(cudaMemset(d_ctr, 0, 8)); CK(cudaMemset(d_out, 0, maxn));
            barrier(); issue(0); CK(cudaStreamSynchronize(s)); CK(cudaMemcpy(o_nccl.data(), d_out, n * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemset(d_out, 0, maxn)); barrier(); issue(1); CK(cudaStreamSynchronize(s)); CK(cudaMemcpy(o_rdma.data(), d_out, n * 2, cudaMemcpyDeviceToHost));
            size_t bad = 0; for (int i = 0; i < n; ++i) bad += o_nccl[i] != o_rdma[i];
            printf("rank %d: bitwise check %d bf16 values nccl vs rdma: %zu differ%s\n", rank, n, bad, ring.failed ? " (RDMA FAILED)" : "");
        }
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
        barrier();
    }
    ring.stop(); NK(ncclCommDestroy(comm));
    printf("rank %d: done, rdma failed=%d\n", rank, (int)ring.failed.load());
    return 0;
}
