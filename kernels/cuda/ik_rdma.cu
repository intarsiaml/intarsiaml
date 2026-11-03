// Direct RDMA all-reduce of f32 partials between the two TP ranks (lane dsv41/rdma, INTARSIA_DSV41_RDMA;
// docs/results/2026-11-02-dsv41-rdma.md). The collective of `Comm::all_reduce_sum_into` at world 2 for an f32 partial and a
// bf16 output: out = bf16(own + peer), bitwise what ncclAllReduce + ik_rt_f32_round give (a + b commutes).
//
// Design (TensorFold's one-hop RDMA ring, zig/src/families/dsv41/rdma.zig, written afresh with an all-reduce collect):
//  * pinned host memory (mmap + mlock + cudaHostRegister, mapped): a send ring, a receive ring and a metadata block (flags,
//    doorbell, sizes, sent, abort word); the rings are ibv_reg_mr'ed on every rail. No GPUDirect, no peermem, no driver setting.
//  * ik_rdma_stage (kernel): copies this rank's partial into its send slot, fences, and rings the doorbell. The sequence
//    number is device state (`*seq`), so a recorded CUDA graph replays it unchanged.
//  * a host proxy thread polls the doorbell, posts one RDMA_WRITE_WITH_IMM per rail (the immediate is the sequence) into
//    the peer's receive slot, polls the completion queues and publishes `flags[slot][peer] = seq` once every rail has
//    delivered.
//  * ik_rdma_collect (kernel): waits for the flag (ld.acquire.sys), then writes bf16(own + peer).
// A wait ends with __trap() when the abort word is set (a failed work completion, a lost peer, `ik_rdma_abort`) or after the
// deadline (`deadline_s` of ik_rdma_open): the pass fails loudly at the next synchronise instead of summing stale bytes.
// Link errors: the queue pairs use NCCL's own settings (timeout 20 = 4.1 s a try, 7 retries: about 34 s before a dead link
// completes with an error), so a flap that NCCL rides out is ridden out here too.
// libibverbs is loaded with dlopen when the first ring is opened, so a binary built with this file starts on a host without it
// (ik_rdma_open then reports the missing library and the caller keeps NCCL).
// Built with the verbs header only when infiniband/verbs.h was found (build.rs defines IK_HAVE_VERBS); otherwise every entry
// point reports it.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#ifdef IK_HAVE_VERBS
#include <arpa/inet.h>
#include <dlfcn.h>
#include <infiniband/verbs.h>
#include <sched.h>
#include <sys/mman.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

typedef unsigned long long ik_u64;

namespace {

__device__ __forceinline__ ik_u64 ik_ld_acq(const ik_u64* p) { ik_u64 v; asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void ik_st_rlx(ik_u64* p, ik_u64 v) { asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory"); }
__device__ __forceinline__ ik_u64 ik_now_ns() { ik_u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

struct RingDev {
    char* send; char* recv;                                   // device views of the pinned host rings
    ik_u64* flags; ik_u64* doorbell; ik_u64* sizes; ik_u64* sent; ik_u64* abort_word;   // pinned host metadata
    ik_u64* seq; unsigned* staged;                            // device scalars
    int slots; int rank; int world; ik_u64 maxb;
    ik_u64 deadline_ns;                                       // a wait this long means the peer is gone
};

// spin until `*p >= want`; the abort word or the deadline traps
__device__ __forceinline__ void ik_wait_ge(const ik_u64* p, ik_u64 want, const RingDev& r) {
    ik_u64 t0 = ik_now_ns();
    for (unsigned i = 1; ik_ld_acq(p) < want; ++i)
        if ((i & 1023) == 0 && (ik_ld_acq(r.abort_word) || ik_now_ns() - t0 > r.deadline_ns)) __trap();
}

__global__ void __launch_bounds__(512) ik_rdma_stage(const float4* __restrict__ src, int n4, RingDev r) {
    __shared__ ik_u64 s_seq; __shared__ int s_last;
    if (threadIdx.x == 0) {
        const ik_u64 q = *r.seq + 1; const int slot = (int)(q % r.slots);
        if (q > (ik_u64)r.slots) {   // the slot's previous send must have completed
            ik_u64 t0 = ik_now_ns();
            for (unsigned i = 1; ik_ld_acq(r.sent + slot) + r.slots < q; ++i)
                if ((i & 1023) == 0 && (ik_ld_acq(r.abort_word) || ik_now_ns() - t0 > r.deadline_ns)) __trap();
        }
        s_seq = q;
        if (blockIdx.x == 0) ik_st_rlx(r.sizes + slot, (ik_u64)n4 * 16);
    }
    __syncthreads();
    const ik_u64 q = s_seq; const int slot = (int)(q % r.slots);
    float4* dst = reinterpret_cast<float4*>(r.send + (size_t)slot * r.maxb);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) dst[i] = src[i];
    __threadfence_system();   // every thread orders its own stores to the pinned ring before the barrier and the doorbell
    __syncthreads();
    if (threadIdx.x == 0) { s_last = atomicAdd(r.staged, 1u) == gridDim.x - 1; }
    __syncthreads();
    if (threadIdx.x == 0 && s_last) {
        __threadfence_system();
        *r.staged = 0; *r.seq = q;
        ik_st_rlx(r.doorbell, q);
    }
}

// out = bf16(own + peer), a + b == b + a so both ranks write the same bits
__global__ void __launch_bounds__(256) ik_rdma_collect(const float4* __restrict__ own, __nv_bfloat162* __restrict__ out, int n4, RingDev r) {
    const ik_u64 q = *r.seq; const int slot = (int)(q % r.slots); const int peer = 1 - r.rank;
    if (threadIdx.x == 0) ik_wait_ge(r.flags + (size_t)slot * r.world + peer, q, r);
    __syncthreads();
    const float4* in = reinterpret_cast<const float4*>(r.recv + ((size_t)slot * r.world + peer) * r.maxb);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        const float4 a = own[i], b = __ldcv(in + i);
        out[2 * i] = __floats2bfloat162_rn(a.x + b.x, a.y + b.y);
        out[2 * i + 1] = __floats2bfloat162_rn(a.z + b.z, a.w + b.w);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// libibverbs through dlopen. The post / poll calls are inline in verbs.h (they go through the context's ops table), so only
// these exported functions are looked up; the names carry no ibv_ prefix because verbs.h defines macros for some of them.
// ---------------------------------------------------------------------------------------------------------------------
struct VerbsApi {
    void* lib = nullptr;
    ibv_device** (*get_device_list)(int*) = nullptr;
    void (*free_device_list)(ibv_device**) = nullptr;
    const char* (*get_device_name)(ibv_device*) = nullptr;
    ibv_context* (*open_device)(ibv_device*) = nullptr;
    int (*close_device)(ibv_context*) = nullptr;
    int (*query_port)(ibv_context*, uint8_t, ibv_port_attr*) = nullptr;
    int (*query_gid)(ibv_context*, uint8_t, int, ibv_gid*) = nullptr;
    ibv_pd* (*alloc_pd)(ibv_context*) = nullptr;
    int (*dealloc_pd)(ibv_pd*) = nullptr;
    ibv_cq* (*create_cq)(ibv_context*, int, void*, ibv_comp_channel*, int) = nullptr;
    int (*destroy_cq)(ibv_cq*) = nullptr;
    ibv_mr* (*reg_mr)(ibv_pd*, void*, size_t, int) = nullptr;
    int (*dereg_mr)(ibv_mr*) = nullptr;
    ibv_qp* (*create_qp)(ibv_pd*, ibv_qp_init_attr*) = nullptr;
    int (*modify_qp)(ibv_qp*, ibv_qp_attr*, int) = nullptr;
    int (*destroy_qp)(ibv_qp*) = nullptr;
    const char* (*wc_status_str)(ibv_wc_status) = nullptr;
};
VerbsApi V;
std::mutex v_mu;
std::string v_err;

bool load_verbs(bool pretend_missing, std::string& err) {
    std::lock_guard<std::mutex> g(v_mu);
    if (V.lib) return true;
    if (pretend_missing) { err = "libibverbs is not available (fault hook 'nolib')"; return false; }
    if (!v_err.empty()) { err = v_err; return false; }
    void* h = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libibverbs.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) { const char* d = dlerror(); err = std::string("libibverbs not found: ") + (d ? d : "dlopen failed"); return false; }
    VerbsApi a; a.lib = h;
    bool ok = true;
    auto sym = [&](auto& fn, const char* name) { fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(h, name)); if (!fn) { ok = false; err = std::string("libibverbs lacks ") + name; } };
    sym(a.get_device_list, "ibv_get_device_list"); sym(a.free_device_list, "ibv_free_device_list"); sym(a.get_device_name, "ibv_get_device_name");
    sym(a.open_device, "ibv_open_device"); sym(a.close_device, "ibv_close_device"); sym(a.query_port, "ibv_query_port"); sym(a.query_gid, "ibv_query_gid");
    sym(a.alloc_pd, "ibv_alloc_pd"); sym(a.dealloc_pd, "ibv_dealloc_pd"); sym(a.create_cq, "ibv_create_cq"); sym(a.destroy_cq, "ibv_destroy_cq");
    sym(a.reg_mr, "ibv_reg_mr"); sym(a.dereg_mr, "ibv_dereg_mr"); sym(a.create_qp, "ibv_create_qp"); sym(a.modify_qp, "ibv_modify_qp");
    sym(a.destroy_qp, "ibv_destroy_qp"); sym(a.wc_status_str, "ibv_wc_status_str");
    if (!ok) { v_err = err; dlclose(h); return false; }
    V = a;
    return true;
}

struct Info { uint32_t qpn, psn, rkey; uint8_t gid[16]; int mtu; uint64_t recv; };
struct InfoSet { int nd; Info i[2]; };

struct Pinned {
    char* host = nullptr; char* dev = nullptr; size_t len = 0; bool registered = false;
    bool alloc(size_t n, std::string& err) {
        len = (n + 65535) & ~(size_t)65535;
        host = (char*)mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (host == MAP_FAILED) { err = "mmap failed"; host = nullptr; return false; }
        memset(host, 0, len); mlock(host, len);
        cudaError_t e = cudaHostRegister(host, len, cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (e != cudaSuccess) { cudaGetLastError(); err = std::string("cudaHostRegister: ") + cudaGetErrorString(e); return false; }
        registered = true;
        e = cudaHostGetDevicePointer((void**)&dev, host, 0);
        if (e != cudaSuccess) { cudaGetLastError(); err = std::string("cudaHostGetDevicePointer: ") + cudaGetErrorString(e); return false; }
        return true;
    }
    void release() { if (host) { if (registered) cudaHostUnregister(host); munmap(host, len); host = nullptr; registered = false; } }
};

struct Nic { ibv_context* ctx = nullptr; ibv_pd* pd = nullptr; ibv_cq* cq = nullptr; ibv_mr *mr_send = nullptr, *mr_recv = nullptr; ibv_qp* qp = nullptr; int mtu = 0; uint8_t gid[16]{}; };

struct Ring {
    int rank = 0, world = 2, slots = 4, gid_index = 3, nd = 0; size_t maxb = 0; ik_u64 deadline_ns = 0;
    Pinned send, recv, meta; ik_u64* d_scalars = nullptr;
    ik_u64 *flags = nullptr, *doorbell = nullptr, *sizes = nullptr, *sent = nullptr, *abort_word = nullptr;
    Nic nic[2]; InfoSet peer{};
    std::thread proxy; std::atomic<bool> running{false}; std::atomic<bool> failed{false}; bool ever_started = false; char why[200] = {0};
    cpu_set_t pin; bool pinned = false;
    // test-only fault hook (INTARSIA_DSV41_RDMA_FAULT): move a rail's queue pair to the error state after the n-th send
    ik_u64 fault_after = 0; int fault_rail = 0; bool fault_ping = false;

    static constexpr size_t LINE = 128;
    ~Ring() { if (!ever_started) teardown(); }

    // Before the proxy ran and before any graph saw the addresses everything can be freed; afterwards (runtime teardown) the
    // rings stay mapped, a recorded graph may still hold them.
    void teardown() {
        for (int k = 0; k < 2; ++k) {
            Nic& c = nic[k];
            if (c.qp) V.destroy_qp(c.qp);
            if (c.mr_send) V.dereg_mr(c.mr_send);
            if (c.mr_recv) V.dereg_mr(c.mr_recv);
            if (c.cq) V.destroy_cq(c.cq);
            if (c.pd) V.dealloc_pd(c.pd);
            if (c.ctx) V.close_device(c.ctx);
            c = Nic{};
        }
        send.release(); recv.release(); meta.release();
        if (d_scalars) { cudaFree(d_scalars); d_scalars = nullptr; }
    }
    bool fail(const char* w) {
        if (!failed.exchange(true)) { snprintf(why, sizeof why, "%s", w); fprintf(stderr, "rank %d: RDMA ring failed: %s\n", rank, w); }
        if (abort_word) __atomic_store_n(abort_word, 1ull, __ATOMIC_RELEASE);
        return false;
    }
    static std::vector<std::string> split(const std::string& s) {
        std::vector<std::string> o; size_t i = 0;
        while (i <= s.size()) { size_t j = s.find(',', i); if (j == std::string::npos) j = s.size(); if (j > i) o.push_back(s.substr(i, j - i)); i = j + 1; }
        return o;
    }
    bool open(const std::string& devs, int gid_idx, int rk, size_t max_bytes, int nslots, int min_rails, double deadline_s, const std::string& fault, std::string& err) {
        rank = rk; maxb = max_bytes; slots = nslots; gid_index = gid_idx; deadline_ns = (ik_u64)(deadline_s * 1e9);
        if (!fault.empty()) {
            if (fault.rfind("qp-err:", 0) == 0) {
                fault_after = strtoull(fault.c_str() + 7, nullptr, 10);
                const size_t c = fault.find(':', 7); if (c != std::string::npos) fault_rail = atoi(fault.c_str() + c + 1);
            } else if (fault == "ping") fault_ping = true;
            else if (fault != "nolib") { err = "INTARSIA_DSV41_RDMA_FAULT: unknown value '" + fault + "' (qp-err:N[:RAIL], ping, nolib)"; return false; }
        }
        if (!load_verbs(fault == "nolib", err)) return false;
        const std::vector<std::string> names = split(devs);
        nd = (int)names.size();
        if (nd < 1 || nd > 2) { err = "1 or 2 RDMA devices wanted, NCCL_IB_HCA names " + std::to_string(nd); return false; }
        if (nd < min_rails) { err = "needs " + std::to_string(min_rails) + " RDMA rails, NCCL_IB_HCA names " + std::to_string(nd); return false; }
        if (nd == 2 && names[0] == names[1]) { err = "NCCL_IB_HCA names " + names[0] + " twice"; return false; }
        if (!send.alloc((size_t)slots * maxb, err) || !recv.alloc((size_t)slots * world * maxb, err) || !meta.alloc(16 * LINE + (size_t)slots * world * 8 + 4 * slots * 8, err)) return false;
        char* m = meta.host; auto take = [&](size_t bytes) { ik_u64* p = (ik_u64*)m; m += (bytes + LINE - 1) / LINE * LINE; return p; };
        flags = take((size_t)slots * world * 8); doorbell = take(8); sizes = take((size_t)slots * 8); sent = take((size_t)slots * 8); abort_word = take(8);
        if (cudaMalloc(&d_scalars, 64) != cudaSuccess) { cudaGetLastError(); err = "cudaMalloc"; return false; }
        cudaMemset(d_scalars, 0, 64);
        int n = 0; ibv_device** l = V.get_device_list(&n);
        if (!l) { err = "ibv_get_device_list failed (no RDMA devices)"; return false; }
        struct ListGuard { ibv_device** l; ~ListGuard() { V.free_device_list(l); } } guard{l};
        for (int k = 0; k < nd; ++k) {
            ibv_device* d = nullptr;
            for (int i = 0; i < n; ++i) if (names[k] == V.get_device_name(l[i])) d = l[i];
            if (!d) { err = "no RDMA device " + names[k]; return false; }
            Nic& c = nic[k];
            c.ctx = V.open_device(d); if (!c.ctx) { err = "ibv_open_device " + names[k]; return false; }
            ibv_port_attr pa{}; if (V.query_port(c.ctx, 1, &pa)) { err = "ibv_query_port " + names[k]; return false; }
            if (pa.state != IBV_PORT_ACTIVE) { err = "RDMA port of " + names[k] + " is not active"; return false; }
            c.mtu = pa.active_mtu;
            c.pd = V.alloc_pd(c.ctx); c.cq = V.create_cq(c.ctx, 4096, nullptr, nullptr, 0);
            if (!c.pd || !c.cq) { err = "ibv_alloc_pd / ibv_create_cq on " + names[k]; return false; }
            const int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
            c.mr_send = V.reg_mr(c.pd, send.host, send.len, acc); c.mr_recv = V.reg_mr(c.pd, recv.host, recv.len, acc);
            if (!c.mr_send || !c.mr_recv) { err = "ibv_reg_mr failed on " + names[k] + " (RLIMIT_MEMLOCK?)"; return false; }
            ibv_gid g; if (V.query_gid(c.ctx, 1, gid_index, &g)) { err = "ibv_query_gid " + std::to_string(gid_index) + " on " + names[k]; return false; }
            bool zero = true; for (int b = 0; b < 16; ++b) zero = zero && g.raw[b] == 0;
            if (zero) { err = "GID index " + std::to_string(gid_index) + " of " + names[k] + " is empty (NCCL_IB_GID_INDEX)"; return false; }
            memcpy(c.gid, g.raw, 16);
            ibv_qp_init_attr qa{}; qa.send_cq = c.cq; qa.recv_cq = c.cq; qa.qp_type = IBV_QPT_RC; qa.cap = {512, 256, 1, 1, 0};
            c.qp = V.create_qp(c.pd, &qa); if (!c.qp) { err = "ibv_create_qp on " + names[k]; return false; }
            ibv_qp_attr a{}; a.qp_state = IBV_QPS_INIT; a.pkey_index = 0; a.port_num = 1; a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
            if (V.modify_qp(c.qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) { err = "qp INIT on " + names[k]; return false; }
            for (int i = 0; i < 256; ++i) if (!post_recv(c)) { err = "ibv_post_recv"; return false; }
        }
        return true;
    }
    bool post_recv(Nic& c) { ibv_recv_wr wr{}, *bad; return ibv_post_recv(c.qp, &wr, &bad) == 0; }
    InfoSet mine() const {
        InfoSet o{}; o.nd = nd;
        for (int k = 0; k < nd; ++k) { const Nic& c = nic[k]; Info& i = o.i[k]; i.qpn = c.qp->qp_num; i.psn = (c.qp->qp_num * 2654435761u) & 0xFFFFFF; i.rkey = c.mr_recv->rkey; memcpy(i.gid, c.gid, 16); i.mtu = c.mtu; i.recv = (uint64_t)recv.host; }
        return o;
    }
    bool connect(const InfoSet& p, std::string& err) {
        peer = p; if (p.nd != nd) { err = "the peer opened " + std::to_string(p.nd) + " RDMA rails, this rank " + std::to_string(nd); return false; }
        const InfoSet me = mine();
        for (int k = 0; k < nd; ++k) {
            Nic& c = nic[k]; const Info& pi = p.i[k]; const int m = c.mtu < pi.mtu ? c.mtu : pi.mtu;
            ibv_qp_attr a{}; a.qp_state = IBV_QPS_RTR; a.path_mtu = (ibv_mtu)m; a.dest_qp_num = pi.qpn; a.rq_psn = pi.psn; a.max_dest_rd_atomic = 1; a.min_rnr_timer = 12;
            a.ah_attr.is_global = 1; memcpy(a.ah_attr.grh.dgid.raw, pi.gid, 16); a.ah_attr.grh.sgid_index = gid_index; a.ah_attr.grh.hop_limit = 64; a.ah_attr.port_num = 1;
            if (V.modify_qp(c.qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) { err = "qp RTR"; return false; }
            // NCCL's own settings (src/transport/net_ib: NCCL_IB_TIMEOUT 20, NCCL_IB_RETRY_CNT 7, rnr_retry 7)
            memset(&a, 0, sizeof a); a.qp_state = IBV_QPS_RTS; a.timeout = 20; a.retry_cnt = 7; a.rnr_retry = 7; a.sq_psn = me.i[k].psn; a.max_rd_atomic = 1;
            if (V.modify_qp(c.qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) { err = "qp RTS"; return false; }
        }
        return true;
    }
    // Both ranks, queue pairs connected, proxy not running: one 16-byte write-with-immediate each way on every rail, the
    // payload checked. Proves the path (route, GIDs, rkeys, memlock) before a graph depends on it. Receives it consumes are
    // reposted; the proxy's sequence numbers are not touched.
    bool ping(int timeout_ms, std::string& err) {
        if (fault_ping) { err = "fault hook 'ping'"; return false; }
        auto magic = [&](int r, int d, int w) { return 0x1badcafe00000000ull | ((ik_u64)r << 16) | ((ik_u64)d << 8) | (ik_u64)w; };
        for (int d = 0; d < nd; ++d) {
            ik_u64* p = (ik_u64*)(send.host + 16 * d); p[0] = magic(rank, d, 0); p[1] = magic(rank, d, 1);
            Nic& c = nic[d];
            ibv_sge sge{(uint64_t)(send.host + 16 * d), 16, c.mr_send->lkey};
            ibv_send_wr wr{}, *bad; wr.wr_id = ~0ull - d; wr.sg_list = &sge; wr.num_sge = 1; wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM; wr.send_flags = IBV_SEND_SIGNALED;
            wr.imm_data = htonl(0x7e570000u + d); wr.wr.rdma.remote_addr = peer.i[d].recv + ((size_t)0 * world + rank) * maxb + 16 * d; wr.wr.rdma.rkey = peer.i[d].rkey;
            if (ibv_post_send(c.qp, &wr, &bad)) { err = "ping: ibv_post_send failed"; return false; }
        }
        int sent_n = 0, got_n = 0; ibv_wc wc[8];
        const auto t0 = std::chrono::steady_clock::now();
        while (sent_n < nd || got_n < nd) {
            for (int d = 0; d < nd; ++d) {
                const int k = ibv_poll_cq(nic[d].cq, 8, wc);
                if (k < 0) { err = "ping: ibv_poll_cq failed"; return false; }
                for (int i = 0; i < k; ++i) {
                    if (wc[i].status != IBV_WC_SUCCESS) { err = std::string("ping: work completion on rail ") + std::to_string(d) + ": " + V.wc_status_str(wc[i].status); return false; }
                    if (wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) { ++got_n; if (!post_recv(nic[d])) { err = "ping: ibv_post_recv failed"; return false; } }
                    else ++sent_n;
                }
            }
            if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) {
                err = "ping: no answer from the peer within " + std::to_string(timeout_ms) + " ms (sent " + std::to_string(sent_n) + "/" + std::to_string(nd) + ", received " + std::to_string(got_n) + "/" + std::to_string(nd) + ")";
                return false;
            }
        }
        for (int d = 0; d < nd; ++d) {
            const ik_u64* p = (const ik_u64*)(recv.host + ((size_t)0 * world + (1 - rank)) * maxb + 16 * d);
            if (__atomic_load_n(p, __ATOMIC_ACQUIRE) != magic(1 - rank, d, 0) || p[1] != magic(1 - rank, d, 1)) { err = "ping: wrong payload on rail " + std::to_string(d); return false; }
        }
        return true;
    }
    void start(int pin_cpu) {
        if (pin_cpu >= 0) { CPU_ZERO(&pin); CPU_SET(pin_cpu, &pin); pinned = true; }
        ever_started = true; running = true; proxy = std::thread([this] { loop(); });
    }
    void stop() { running = false; if (proxy.joinable()) proxy.join(); }
    void loop() {
        if (pinned) sched_setaffinity(0, sizeof pin, &pin);
        ik_u64 posted = 0, published = 0; ik_u64 got[2] = {0, 0}; unsigned sent_cnt[16] = {0}; ik_u64 sent_q[16] = {0}; ibv_wc wc[16];
        auto last = std::chrono::steady_clock::now(); unsigned spins = 0; bool napping = false;
        while (running.load(std::memory_order_relaxed)) {
            bool busy = false;
            const ik_u64 ring = __atomic_load_n(doorbell, __ATOMIC_ACQUIRE);
            while (posted < ring) {
                busy = true;
                const ik_u64 q = posted + 1; const int slot = (int)(q % slots);
                const ik_u64 n = __atomic_load_n(sizes + slot, __ATOMIC_ACQUIRE);
                const ik_u64 unit = n / 16 / nd * 16;   // rail d writes [unit * d, unit * (d + 1)); the last rail takes the rest
                for (int d = 0; d < nd; ++d) {
                    const ik_u64 off = unit * d, len = d == nd - 1 ? n - off : unit; Nic& c = nic[d];
                    ibv_sge sge{(uint64_t)(send.host + (size_t)slot * maxb + off), (uint32_t)len, c.mr_send->lkey};
                    ibv_send_wr wr{}, *bad; wr.wr_id = q; wr.sg_list = &sge; wr.num_sge = len ? 1 : 0; wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM; wr.send_flags = IBV_SEND_SIGNALED;
                    wr.imm_data = htonl((uint32_t)q); wr.wr.rdma.remote_addr = peer.i[d].recv + ((size_t)slot * world + rank) * maxb + off; wr.wr.rdma.rkey = peer.i[d].rkey;
                    if (ibv_post_send(c.qp, &wr, &bad)) { fail("ibv_post_send failed"); return; }
                }
                posted = q;
                if (fault_after && q == fault_after) {   // test hook: a rail's queue pair goes to the error state, as after a fatal link error
                    ibv_qp_attr a{}; a.qp_state = IBV_QPS_ERR;
                    const int r = fault_rail < nd ? fault_rail : 0;
                    fprintf(stderr, "rank %d: RDMA fault hook: queue pair of rail %d moved to ERR after send %llu\n", rank, r, (unsigned long long)q);
                    V.modify_qp(nic[r].qp, &a, IBV_QP_STATE);
                }
            }
            for (int d = 0; d < nd; ++d) {
                Nic& c = nic[d];
                const int k = ibv_poll_cq(c.cq, 16, wc);
                if (k < 0) { fail("ibv_poll_cq failed"); return; }
                if (k > 0) busy = true;
                for (int i = 0; i < k; ++i) {
                    if (wc[i].status != IBV_WC_SUCCESS) { char b[200]; snprintf(b, sizeof b, "work completion on rail %d: %s", d, V.wc_status_str(wc[i].status)); fail(b); return; }
                    if (wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
                        ik_u64 q = (got[d] & ~0xffffffffull) | ntohl(wc[i].imm_data); if (q <= got[d]) q += 1ull << 32; got[d] = q;
                        ik_u64 in = got[0]; for (int e = 1; e < nd; ++e) in = got[e] < in ? got[e] : in;   // every rail has delivered up to `in`
                        for (ik_u64 x = published + 1; x <= in; ++x) __atomic_store_n(flags + (size_t)(x % slots) * world + (1 - rank), x, __ATOMIC_RELEASE);
                        if (in > published) published = in;
                        if (!post_recv(c)) { fail("ibv_post_recv failed"); return; }
                    } else {
                        const ik_u64 q = wc[i].wr_id; const int sl = (int)(q % slots);
                        if (sent_q[sl] != q) { sent_q[sl] = q; sent_cnt[sl] = 0; }
                        if (++sent_cnt[sl] == (unsigned)nd) __atomic_store_n(sent + sl, q, __ATOMIC_RELEASE);
                    }
                }
            }
            // spin while there is traffic; after 20 ms without any, nap 20 us a turn (the first exchange after an idle spell pays at most that)
            if (busy) { last = std::chrono::steady_clock::now(); napping = false; spins = 0; }
            else if ((++spins & 255) == 0) napping = std::chrono::steady_clock::now() - last > std::chrono::milliseconds(20);
            if (napping) std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
    }
    RingDev dev() const {
        RingDev r{}; r.send = send.dev; r.recv = recv.dev; auto m = [&](ik_u64* p) { return (ik_u64*)(meta.dev + ((char*)p - meta.host)); };
        r.flags = m(flags); r.doorbell = m(doorbell); r.sizes = m(sizes); r.sent = m(sent); r.abort_word = m(abort_word);
        r.seq = d_scalars; r.staged = (unsigned*)(d_scalars + 1); r.slots = slots; r.rank = rank; r.world = world; r.maxb = maxb; r.deadline_ns = deadline_ns;
        return r;
    }
};

}  // namespace

static void ik_set_err(char* err, int errlen, const std::string& s) { if (err && errlen > 0) snprintf(err, (size_t)errlen, "%s", s.c_str()); }

extern "C" int ik_rdma_available() { return 1; }
extern "C" int ik_rdma_info_size() { return (int)sizeof(InfoSet); }

// min_rails: 2 under auto (one rail missed the latency gate), 1 when the user asked for RDMA outright. deadline_s: how long a
// device wait may last before it traps. fault: the test hook's value or "".
extern "C" void* ik_rdma_open(const char* devs, int gid, int rank, size_t max_bytes, int slots, int min_rails, double deadline_s, const char* fault, char* err, int errlen) {
    Ring* r = new Ring();
    std::string e;
    if (!r->open(devs ? devs : "", gid, rank, max_bytes, slots, min_rails, deadline_s, fault ? fault : "", e)) { ik_set_err(err, errlen, e); delete r; return nullptr; }
    return r;
}
extern "C" int ik_rdma_info(void* h, void* out) { *(InfoSet*)out = ((Ring*)h)->mine(); return 0; }
extern "C" int ik_rdma_connect(void* h, const void* peer, char* err, int errlen) {
    std::string e;
    if (!((Ring*)h)->connect(*(const InfoSet*)peer, e)) { ik_set_err(err, errlen, e); return 1; }
    return 0;
}
extern "C" int ik_rdma_ping(void* h, int timeout_ms, char* err, int errlen) {
    std::string e;
    if (!((Ring*)h)->ping(timeout_ms, e)) { ik_set_err(err, errlen, e); return 1; }
    return 0;
}
extern "C" int ik_rdma_start(void* h, int pin_cpu) { ((Ring*)h)->start(pin_cpu); return 0; }

// out (bf16, n elements) = bf16(partial (f32, n elements) + the peer's), on `stream`. n must be a multiple of 4 and n * 4 <= max_bytes.
extern "C" int ik_rdma_exchange(void* h, const void* partial, void* out, size_t n, void* stream) {
    Ring* r = (Ring*)h;
    if (r->failed.load()) return (int)cudaErrorUnknown;
    if (n == 0 || n % 4 != 0 || n * 4 > r->maxb) return (int)cudaErrorInvalidValue;
    const int n4 = (int)(n / 4);
    cudaStream_t s = (cudaStream_t)stream;
    const RingDev d = r->dev();
    int sb = (n4 + 1023) / 1024; sb = sb < 1 ? 1 : sb > 16 ? 16 : sb;   // the bench's settings (docs/results/2026-11-02-dsv41-rdma.md)
    int cb = (n4 + 255) / 256; cb = cb < 1 ? 1 : cb > 48 ? 48 : cb;
    ik_rdma_stage<<<sb, 512, 0, s>>>((const float4*)partial, n4, d);
    ik_rdma_collect<<<cb, 256, 0, s>>>((const float4*)partial, (__nv_bfloat162*)out, n4, d);
    return (int)cudaGetLastError();
}
extern "C" int ik_rdma_failed(void* h, char* why, int len) {
    Ring* r = (Ring*)h;
    if (!r->failed.load()) return 0;
    ik_set_err(why, len, r->why);
    return 1;
}
extern "C" void ik_rdma_abort(void* h) { ((Ring*)h)->fail("aborted (the group failed)"); }
// A ring that was opened but never started (setup fell back to NCCL): everything is freed. A started ring only stops its
// proxy: the rings stay registered, a graph may still hold their addresses until the process ends.
extern "C" void ik_rdma_close(void* h) {
    Ring* r = (Ring*)h;
    r->stop();
    if (!r->ever_started) delete r;
}

#else  // IK_HAVE_VERBS

extern "C" int ik_rdma_available() { return 0; }
extern "C" int ik_rdma_info_size() { return 0; }
extern "C" void* ik_rdma_open(const char*, int, int, size_t, int, int, double, const char*, char* err, int errlen) {
    if (err && errlen > 0) snprintf(err, (size_t)errlen, "built without libibverbs (no infiniband/verbs.h at build time)");
    return nullptr;
}
extern "C" int ik_rdma_info(void*, void*) { return 1; }
extern "C" int ik_rdma_connect(void*, const void*, char*, int) { return 1; }
extern "C" int ik_rdma_ping(void*, int, char*, int) { return 1; }
extern "C" int ik_rdma_start(void*, int) { return 1; }
extern "C" int ik_rdma_exchange(void*, const void*, void*, size_t, void*) { return (int)cudaErrorNotSupported; }
extern "C" int ik_rdma_failed(void*, char*, int) { return 1; }
extern "C" void ik_rdma_abort(void*) {}
extern "C" void ik_rdma_close(void*) {}

#endif
