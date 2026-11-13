// gb10-d2h: does a device-to-host copy into PAGEABLE host memory ever return success with stale or misplaced bytes on GB10?
// (TensorFold issue #2102: driver 580.178.04, ATS mode 1, cuMemcpyDtoH_v2 into pageable memory, no small public reproducer.)
//
// Methods (all into pageable host memory unless noted):
//   ours       ik_rt_memcpy_d2h from kernels/cuda/ik_runtime.cu (cudaMemcpyAsync + cudaStreamSynchronize), the engine's exact call
//   ours_hp    the same on a high-priority stream (the decode lane, device.rs stream ids >= 1)
//   sync_rt    cudaMemcpy(D2H)                       TensorFold's condition through the runtime API
//   sync_drv   cuMemcpyDtoH_v2                       TensorFold's exact call
//   bounce     negative control: ik_rt_memcpy_d2h into a pinned buffer, then memcpy (the TensorFold / candidate fix)
// Destination kinds: thp = fresh mmap, 2 MiB aligned, MADV_HUGEPAGE, prefilled; pageout = fresh, prefilled, then MADV_PAGEOUT (pages in swap
// when the copy starts); fresh = malloc/mmap'd per copy and prefilled 0xAA (TensorFold's compare buffers), cold = fresh and untouched
// (pages not yet faulted in), reuse = one long-lived buffer, rand_off = fresh, prefilled, copy lands at an unaligned offset.
// Checks after every copy: every byte of the destination equals the device pattern, guard pages around it keep their canary, and
// a pool of live heap buffers (allocator churn, like the GPA in the TensorFold report) still holds its fingerprints.
#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <string>
#include <vector>

extern "C" {
int ik_rt_malloc(void** p, size_t bytes);
int ik_rt_memcpy_d2h(void* dst, const void* src, size_t n, void* stream);
int ik_rt_stream_create(void** stream);
int ik_rt_stream_create_high_priority(void** stream);
int ik_rt_init_device(int ordinal, void* info);
}
// ik_runtime.cu references these two from other translation units; the repro does not need them.
extern "C" int ik_cublas_init(int) { return 0; }
extern "C" void ik_fp8_bf16_cache_forget(const void*) {}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)

static const size_t PAGE = 4096;
static inline uint64_t mix(uint64_t x) { x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31); }
static double now_us() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }

// ---- device pattern: byte i of buffer `seed` is byte (i & 7) of mix(seed * 2^32 + i / 8)
__global__ void fill_pat(uint64_t* p, size_t words, uint64_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x) {
        uint64_t x = (seed << 32) + i;
        x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        p[i] = x ^ (x >> 31);
    }
}
static inline uint8_t pat_byte(uint64_t seed, size_t i) { return (uint8_t)(mix((seed << 32) + i / 8) >> (8 * (i & 7))); }

struct Dev { void* p; size_t bytes; uint64_t seed; };

// bytes of p[0..n) that differ from the pattern of `seed`; whole words compared first
static size_t count_bad(const uint8_t* p, size_t n, uint64_t seed, size_t* first = nullptr, size_t* aa = nullptr) {
    size_t bad = 0;
    for (size_t w = 0; w < n / 8; w++) {
        uint64_t e = mix((seed << 32) + w), g;
        memcpy(&g, p + w * 8, 8);
        if (g == e) continue;
        for (size_t i = w * 8; i < w * 8 + 8; i++) if (p[i] != pat_byte(seed, i)) { bad++; if (first && *first == (size_t)-1) *first = i; if (aa && p[i] == 0xAA) (*aa)++; }
    }
    for (size_t i = n / 8 * 8; i < n; i++) if (p[i] != pat_byte(seed, i)) { bad++; if (first && *first == (size_t)-1) *first = i; if (aa && p[i] == 0xAA) (*aa)++; }
    return bad;
}

// ---- allocator churn pool: live buffers with fingerprints
struct Pool {
    struct Item { uint8_t* p; size_t n; uint64_t seed; };
    std::vector<Item> items;
    uint64_t rng;
    explicit Pool(uint64_t s, int count, size_t total) : rng(s) {
        size_t per = total / count;
        for (int i = 0; i < count; i++) items.push_back(make(per));
    }
    uint64_t next() { return rng = mix(rng); }
    Item make(size_t maxn) {
        size_t n = 4096 + next() % (maxn - 4096);
        Item it{(uint8_t*)malloc(n), n, next()};
        for (size_t i = 0; i < n; i++) it.p[i] = pat_byte(it.seed, i);
        return it;
    }
    void churn(size_t maxn) {  // replace a few items
        for (int k = 0; k < 3; k++) {
            size_t j = next() % items.size();
            free(items[j].p);
            items[j] = make(maxn);
        }
    }
    size_t verify() {  // bad bytes
        size_t bad = 0;
        for (auto& it : items) bad += count_bad(it.p, it.n, it.seed);
        return bad;
    }
};

enum Method { OURS, OURS_HP, SYNC_RT, SYNC_DRV, BOUNCE, MUT_NOOP, MUT_SKIPPAGE };
static const char* mname[] = {"ours", "ours_hp", "sync_rt", "sync_drv", "bounce", "mut_noop", "mut_skippage"};
enum Kind { FRESH, COLD, REUSE, RANDOFF, THP, PAGEOUT };
static const char* kname[] = {"fresh", "cold", "reuse", "rand_off", "thp", "pageout"};

struct Cfg { Method m; Kind k; size_t bytes; int iters; };
struct Result { size_t po_pages = 0, po_resident = 0, runs = 0, bad_runs = 0, stale_bytes = 0, canary_bad = 0, pool_bad = 0, cuda_err = 0; std::vector<double> us; std::string first; };

struct Worker {
    int tid; Dev dev; void* stream; void* stream_hp; uint8_t* pinned = nullptr; size_t pinned_n = 0;
};

static uint8_t* map_fresh(size_t n, bool use_mmap, uint8_t** base, size_t* total) {
    // guard pages before and after (canary-filled), region of n bytes between; either mmap or malloc
    size_t body = (n + PAGE - 1) / PAGE * PAGE;
    *total = body + 2 * PAGE;
    uint8_t* b = use_mmap ? (uint8_t*)mmap(nullptr, *total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) : (uint8_t*)malloc(*total);
    if (b == (uint8_t*)MAP_FAILED || !b) { perror("alloc"); exit(3); }
    *base = b;
    return b + PAGE;
}

static std::atomic<int> g_bad_printed{0};

static void run_cfg(Worker& w, const Cfg& c, Result& r, bool pool_on) {
    Pool pool(0x1234 + w.tid, 48, 8u << 20);
    uint8_t* reuse = nullptr; uint8_t* reuse_base = nullptr; size_t reuse_total = 0;
    uint64_t rng = mix(w.tid * 77 + c.bytes);
    for (int it = 0; it < c.iters; it++) {
        rng = mix(rng);
        bool use_mmap = c.bytes >= (128 << 10) ? (rng & 1) : false;  // large mallocs are mmap'd by glibc anyway; vary
        size_t off = c.k == RANDOFF ? (rng >> 8) % 4093 + 1 : 0;
        size_t need = c.bytes + off;
        uint8_t *base, *dst, *raw = nullptr; size_t total, rawlen = 0;
        if (c.k == REUSE) {
            if (!reuse) { reuse = map_fresh(need, false, &reuse_base, &reuse_total); }
            dst = reuse; base = reuse_base; total = reuse_total;
        } else {
            if (c.k == THP) {
                size_t body0 = (need + PAGE - 1) / PAGE * PAGE, two = 2u << 20;
                rawlen = body0 + 2 * PAGE + 2 * two;
                raw = (uint8_t*)mmap(nullptr, rawlen, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (raw == (uint8_t*)MAP_FAILED) { perror("mmap"); exit(3); }
                dst = (uint8_t*)(((uintptr_t)raw + PAGE + two - 1) / two * two);
                base = dst - PAGE; total = body0 + 2 * PAGE;
                madvise(dst, (body0 + two - 1) / two * two, MADV_HUGEPAGE);
                use_mmap = true;
            } else {
                if (c.k == PAGEOUT) use_mmap = true;
                dst = map_fresh(need, use_mmap, &base, &total);
            }
        }
        uint8_t* d = dst + off;
        size_t body = total - 2 * PAGE;
        // canaries in both guard pages; tail slack after c.bytes inside the body gets a canary too
        memset(base, 0xC5, PAGE); memset(base + PAGE + body, 0xC5, PAGE);
        if (c.k != COLD) memset(dst, 0xAA, body);
        else if (off + c.bytes < body) memset(dst + off + c.bytes, 0xAA, body - off - c.bytes);  // tail only; the copy's own pages stay cold
        if (off && c.k != COLD) {}  // head (off bytes) is 0xAA already
        if (c.k == COLD && off) memset(dst, 0xAA, off);
        if (c.k == PAGEOUT) {
            madvise(dst, body, 21 /* MADV_PAGEOUT */);
            // /proc/self/pagemap bit 63 = page present in this process's page table (mincore also counts the swap cache as resident)
            static thread_local int pm = open("/proc/self/pagemap", O_RDONLY);
            std::vector<uint64_t> vec(body / PAGE);
            if (pm >= 0 && pread(pm, vec.data(), vec.size() * 8, (uintptr_t)dst / PAGE * 8) == (ssize_t)(vec.size() * 8)) {
                for (uint64_t v : vec) r.po_resident += v >> 63;
                r.po_pages += vec.size();
            }
        }
        if (pool_on && it % 4 == 0) pool.churn(1u << 20);

        double t0 = now_us();
        int rc = 0;
        switch (c.m) {
            case OURS: rc = ik_rt_memcpy_d2h(d, w.dev.p, c.bytes, w.stream); break;
            case OURS_HP: rc = ik_rt_memcpy_d2h(d, w.dev.p, c.bytes, w.stream_hp); break;
            case SYNC_RT: rc = (int)cudaMemcpy(d, w.dev.p, c.bytes, cudaMemcpyDeviceToHost); break;
            case SYNC_DRV: rc = (int)cuMemcpyDtoH((void*)d, (CUdeviceptr)w.dev.p, c.bytes); break;
            case MUT_NOOP: rc = ik_rt_memcpy_d2h(w.pinned, w.dev.p, 8, w.stream); break;  // harness self-test: nothing lands in d
            case MUT_SKIPPAGE: {  // harness self-test: one page in the middle is missed, as in the TensorFold report
                size_t mid = c.bytes / 2 / PAGE * PAGE;
                rc = ik_rt_memcpy_d2h(d, w.dev.p, mid, w.stream);
                if (!rc && c.bytes > mid + PAGE) rc = ik_rt_memcpy_d2h(d + mid + PAGE, (const char*)w.dev.p + mid + PAGE, c.bytes - mid - PAGE, w.stream);
            } break;
            case BOUNCE: {
                size_t done = 0;
                while (done < c.bytes) {
                    size_t n = std::min(c.bytes - done, w.pinned_n);
                    rc = ik_rt_memcpy_d2h(w.pinned, (const char*)w.dev.p + done, n, w.stream);
                    if (rc) break;
                    memcpy(d + done, w.pinned, n);
                    done += n;
                }
            } break;
        }
        double t1 = now_us();
        r.runs++;
        if (rc) { r.cuda_err++; r.bad_runs++; continue; }
        r.us.push_back(t1 - t0);

        size_t stale = 0, first_bad = (size_t)-1, stale_aa = 0;
        stale = count_bad(d, c.bytes, w.dev.seed, &first_bad, &stale_aa);
        size_t canary = 0;
        for (size_t i = 0; i < PAGE; i++) canary += base[i] != 0xC5 || base[PAGE + body + i] != 0xC5;
        // head slack and tail slack inside the body
        for (size_t i = 0; i < off; i++) canary += dst[i] != 0xAA;
        for (size_t i = off + c.bytes; i < body; i++) canary += dst[i] != 0xAA;
        size_t pbad = pool_on ? pool.verify() : 0;
        if (stale || canary || pbad) {
            r.bad_runs++; r.stale_bytes += stale; r.canary_bad += canary; r.pool_bad += pbad;
            if (r.first.empty() || g_bad_printed++ < 20) {
                char buf[256];
                snprintf(buf, sizeof buf, "iter %d: stale %zu of %zu (still 0xAA: %zu, first at %zu), canary %zu, pool %zu",
                         it, stale, c.bytes, stale_aa, first_bad, canary, pbad);
                if (r.first.empty()) r.first = buf;
                fprintf(stderr, "  !! %s %s %zu %s\n", mname[c.m], kname[c.k], c.bytes, buf);
            }
        }
        if (c.k == THP) munmap(raw, rawlen);
        else if (c.k != REUSE) { if (use_mmap) munmap(base, total); else free(base); }
    }
    if (reuse) free(reuse_base);
}

// ---- background load: host memory churn + a GPU kernel stream
static std::atomic<bool> g_stop{false};
static void* pressure_thread(void* arg) {
    size_t hold = (size_t)(intptr_t)arg;  // bytes held and re-touched
    uint8_t* h = hold ? (uint8_t*)mmap(nullptr, hold, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) : nullptr;
    uint64_t r = 1;
    while (!g_stop.load()) {
        if (h) for (size_t i = 0; i < hold; i += PAGE * 64) h[i] = (uint8_t)r;  // keep pages hot
        size_t n = 256u << 20;  // fault in and drop 256 MiB repeatedly
        uint8_t* t = (uint8_t*)mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        for (size_t i = 0; i < n; i += PAGE) t[i] = (uint8_t)(r += 3);
        madvise(t, n / 2, MADV_DONTNEED);
        munmap(t, n);
    }
    if (h) munmap(h, hold);
    return nullptr;
}
__global__ void spin_kernel(float* x, size_t n, int reps) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    for (int k = 0; k < reps; k++) v = v * 1.0000001f + 0.5f;
    x[i] = v;
}
static void* gpu_load_thread(void*) {
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    size_t n = 64u << 20; float* x; CK(cudaMalloc(&x, n * 4)); CK(cudaMemset(x, 0, n * 4));
    while (!g_stop.load()) { spin_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, n, 400); CK(cudaStreamSynchronize(s)); }
    cudaFree(x); cudaStreamDestroy(s);
    return nullptr;
}

struct TArg { int tid; std::vector<Cfg> cfgs; std::vector<Result>* res; bool pool_on; size_t maxbytes; };
static void* thread_main(void* a) {
    TArg* t = (TArg*)a;
    CK(cudaSetDevice(0));
    Worker w; w.tid = t->tid;
    w.dev.bytes = t->maxbytes; w.dev.seed = 100 + t->tid;
    if (ik_rt_malloc(&w.dev.p, w.dev.bytes)) { fprintf(stderr, "ik_rt_malloc failed\n"); exit(2); }
    fill_pat<<<1024, 256>>>((uint64_t*)w.dev.p, w.dev.bytes / 8, w.dev.seed);
    CK(cudaDeviceSynchronize());
    if (ik_rt_stream_create(&w.stream) || ik_rt_stream_create_high_priority(&w.stream_hp)) exit(2);
    w.pinned_n = 4u << 20; CK(cudaHostAlloc((void**)&w.pinned, w.pinned_n, cudaHostAllocPortable));
    for (size_t i = 0; i < t->cfgs.size(); i++) run_cfg(w, t->cfgs[i], (*t->res)[i], t->pool_on);
    return nullptr;
}

int main(int argc, char** argv) {
    // usage: d2h_repro [--iters N] [--threads T] [--pressure GB] [--gpuload] [--no-pool] [--sizes a,b,c (bytes)] [--methods ours,sync_rt,...] [--kinds fresh,cold,...] [--maxbytes B]
    int iters = 200, threads = 1; double pressure_gb = 0; bool gpuload = false, pool_on = true;
    std::vector<size_t> sizes = {4096, 65536, 303104, 1u << 20, 4u << 20, 16u << 20, 64u << 20};
    std::vector<Method> methods = {OURS, OURS_HP, SYNC_RT, SYNC_DRV, BOUNCE};
    std::vector<Kind> kinds = {FRESH, COLD, REUSE, RANDOFF};
    auto split = [](std::string s) { std::vector<std::string> o; size_t p; while ((p = s.find(',')) != std::string::npos) { o.push_back(s.substr(0, p)); s = s.substr(p + 1); } o.push_back(s); return o; };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) exit(1); return std::string(argv[++i]); };
        if (a == "--iters") iters = atoi(nx().c_str());
        else if (a == "--threads") threads = atoi(nx().c_str());
        else if (a == "--pressure") pressure_gb = atof(nx().c_str());
        else if (a == "--gpuload") gpuload = true;
        else if (a == "--no-pool") pool_on = false;
        else if (a == "--sizes") { sizes.clear(); for (auto& s : split(nx())) sizes.push_back(strtoull(s.c_str(), 0, 10)); }
        else if (a == "--methods") { methods.clear(); for (auto& s : split(nx())) for (int m = 0; m < 7; m++) if (s == mname[m]) methods.push_back((Method)m); }
        else if (a == "--kinds") { kinds.clear(); for (auto& s : split(nx())) for (int k = 0; k < 6; k++) if (s == kname[k]) kinds.push_back((Kind)k); }
    }
    CK(cudaSetDevice(0));
    char info[1024]; memset(info, 0, sizeof info);
    if (ik_rt_init_device(0, info)) { fprintf(stderr, "init_device failed\n"); return 2; }
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int drv = 0, rt = 0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
    printf("# device %s integrated=%d pageableMemoryAccess=%d hostNativeAtomic=%d driver_api=%d runtime=%d\n", p.name, p.integrated, p.pageableMemoryAccess, p.hostNativeAtomicSupported, drv, rt);
    printf("# iters=%d threads=%d pressure_gb=%.1f gpuload=%d pool=%d\n", iters, threads, pressure_gb, (int)gpuload, (int)pool_on);
    size_t maxbytes = *std::max_element(sizes.begin(), sizes.end());
    maxbytes = (maxbytes + 7) / 8 * 8;

    pthread_t pt, gt; bool pr = pressure_gb > 0, gl = gpuload;
    if (pr) pthread_create(&pt, 0, pressure_thread, (void*)(intptr_t)(size_t)(pressure_gb * (1ull << 30)));
    if (gl) pthread_create(&gt, 0, gpu_load_thread, 0);

    std::vector<Cfg> cfgs;
    for (size_t s : sizes) for (Kind k : kinds) for (Method m : methods) cfgs.push_back({m, k, s, iters});
    std::vector<std::vector<Result>> res(threads, std::vector<Result>(cfgs.size()));
    std::vector<TArg> targs(threads);
    std::vector<pthread_t> th(threads);
    double t0 = now_us();
    for (int t = 0; t < threads; t++) { targs[t] = {t, cfgs, &res[t], pool_on, maxbytes}; pthread_create(&th[t], 0, thread_main, &targs[t]); }
    for (int t = 0; t < threads; t++) pthread_join(th[t], 0);
    g_stop = true;
    if (pr) pthread_join(pt, 0);
    if (gl) pthread_join(gt, 0);

    size_t total_runs = 0, total_bad = 0, ctl_bad = 0, ours_bad = 0, sync_bad = 0;
    printf("%-9s %-9s %12s %8s %8s %12s %7s %7s %9s\n", "method", "kind", "bytes", "runs", "bad", "stale_bytes", "canary", "pool", "med_us");
    for (size_t i = 0; i < cfgs.size(); i++) {
        Result a;
        for (int t = 0; t < threads; t++) { Result& b = res[t][i]; a.po_pages += b.po_pages; a.po_resident += b.po_resident; a.runs += b.runs; a.bad_runs += b.bad_runs; a.stale_bytes += b.stale_bytes; a.canary_bad += b.canary_bad; a.pool_bad += b.pool_bad; a.cuda_err += b.cuda_err; a.us.insert(a.us.end(), b.us.begin(), b.us.end()); if (a.first.empty()) a.first = b.first; }
        std::sort(a.us.begin(), a.us.end());
        double med = a.us.empty() ? 0 : a.us[a.us.size() / 2];
        printf("%-9s %-9s %12zu %8zu %8zu %12zu %7zu %7zu %9.1f%s%s\n", mname[cfgs[i].m], kname[cfgs[i].k], cfgs[i].bytes, a.runs, a.bad_runs, a.stale_bytes, a.canary_bad, a.pool_bad, med,
               a.cuda_err ? "  CUDA_ERR" : "", a.first.empty() ? "" : ("  " + a.first).c_str());
        if (cfgs[i].k == PAGEOUT && a.po_pages) printf("          (pageout: %.1f %% of the destination's pages present in the page table right before the copy)\n", 100.0 * a.po_resident / a.po_pages);
        total_runs += a.runs; if (cfgs[i].m < MUT_NOOP) total_bad += a.bad_runs;
        if (cfgs[i].m >= MUT_NOOP) {}  // mutants are expected to fail
        else if (cfgs[i].m == OURS || cfgs[i].m == OURS_HP) ours_bad += a.bad_runs;
        else if (cfgs[i].m == SYNC_RT || cfgs[i].m == SYNC_DRV) sync_bad += a.bad_runs;
        else ctl_bad += a.bad_runs;
    }
    printf("TOTAL runs=%zu bad=%zu | ours_bad=%zu sync(positive control)_bad=%zu bounce_bad=%zu | wall %.1f s\n", total_runs, total_bad, ours_bad, sync_bad, ctl_bad, (now_us() - t0) / 1e6);
    return total_bad ? 1 : 0;
}
