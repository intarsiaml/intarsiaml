// R9700 (gfx1201) smoke: vector add, copy + read-only-reduce bandwidth, hipMemGetInfo, P2P.
// hipcc --offload-arch=gfx1201 -O3 -o smoke smoke.hip && HIP_VISIBLE_DEVICES=0 ./smoke
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); exit(1); } } while (0)

__global__ void vadd(const float* a, const float* b, float* c, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
__global__ void copy4(const float4* __restrict__ s, float4* __restrict__ d, size_t n) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += stride) d[i] = s[i];
}
__global__ void reduce4(const float4* __restrict__ s, float* out, size_t n) {
  float acc = 0.f;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += stride) {
    float4 v = s[i]; acc += v.x + v.y + v.z + v.w;
  }
  for (int o = 32; o > 0; o >>= 1) acc += __shfl_down(acc, o);
  __shared__ float sh[32];
  int lane = threadIdx.x & 63, w = threadIdx.x >> 6;
  if (lane == 0) sh[w] = acc;
  __syncthreads();
  if (threadIdx.x == 0) { float t = 0; for (int i = 0; i < (int)(blockDim.x >> 6); i++) t += sh[i]; atomicAdd(out, t); }
}

template <class F> static float best_ms(F f, int iters = 10) {
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  f(); CK(hipDeviceSynchronize());  // warmup
  float best = 1e30f;
  for (int i = 0; i < iters; i++) {
    CK(hipEventRecord(a)); f(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms; CK(hipEventElapsedTime(&ms, a, b)); best = std::min(best, ms);
  }
  return best;
}

int main() {
  const size_t GiB = 1ull << 30;
  int ndev = 0; CK(hipGetDeviceCount(&ndev));
  printf("visible devices: %d\n", ndev);
  for (int d = 0; d < ndev; d++) {
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, d));
    size_t fr, tot; CK(hipSetDevice(d)); CK(hipMemGetInfo(&fr, &tot));
    printf("dev %d: %s arch=%s CUs=%d clock=%d MHz memclk=%d MHz busw=%d L2=%d KB mem total=%.2f GiB free=%.2f GiB\n",
           d, p.name, p.gcnArchName, p.multiProcessorCount, p.clockRate / 1000, p.memoryClockRate / 1000,
           p.memoryBusWidth, p.l2CacheSize / 1024, tot / (double)GiB, fr / (double)GiB);
  }
  CK(hipSetDevice(0));
  // vector add correctness
  {
    size_t n = 1 << 24; std::vector<float> ha(n), hb(n), hc(n);
    for (size_t i = 0; i < n; i++) { ha[i] = i * 0.5f; hb[i] = 1.0f; }
    float *a, *b, *c; CK(hipMalloc(&a, n * 4)); CK(hipMalloc(&b, n * 4)); CK(hipMalloc(&c, n * 4));
    CK(hipMemcpy(a, ha.data(), n * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(b, hb.data(), n * 4, hipMemcpyHostToDevice));
    vadd<<<(n + 255) / 256, 256>>>(a, b, c, n); CK(hipGetLastError());
    CK(hipMemcpy(hc.data(), c, n * 4, hipMemcpyDeviceToHost));
    size_t bad = 0; for (size_t i = 0; i < n; i++) if (hc[i] != ha[i] + 1.0f) bad++;
    printf("vadd n=%zu mismatches=%zu %s\n", n, bad, bad ? "FAIL" : "OK");
    CK(hipFree(a)); CK(hipFree(b)); CK(hipFree(c));
  }
  // bandwidth, 1 GiB buffers
  {
    size_t n4 = GiB / 16; float4 *s, *d; float* o;
    CK(hipMalloc(&s, GiB)); CK(hipMalloc(&d, GiB)); CK(hipMalloc(&o, 4));
    CK(hipMemset(s, 0, GiB)); CK(hipMemset(d, 0, GiB));
    int cus; CK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0));
    int grid = cus * 16, blk = 256;
    float ms = best_ms([&] { copy4<<<grid, blk>>>(s, d, n4); });
    printf("copy kernel   1 GiB: best %.3f ms  %.1f GB/s (read+write counted, %.1f GB/s each way)\n", ms, 2.0 * GiB / ms / 1e6, GiB / ms / 1e6);
    ms = best_ms([&] { CK(hipMemsetAsync(o, 0, 4)); reduce4<<<grid, blk>>>(s, o, n4); });
    printf("reduce kernel 1 GiB: best %.3f ms  %.1f GB/s (read-only)\n", ms, GiB / ms / 1e6);
    ms = best_ms([&] { CK(hipMemcpyAsync(d, s, GiB, hipMemcpyDeviceToDevice)); });
    printf("hipMemcpy D2D 1 GiB: best %.3f ms  %.1f GB/s (read+write counted)\n", ms, 2.0 * GiB / ms / 1e6);
    CK(hipFree(s)); CK(hipFree(d)); CK(hipFree(o));
  }
  // P2P
  if (ndev >= 2) {
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) if (i != j) {
      int can = 0; CK(hipDeviceCanAccessPeer(&can, i, j));
      int rank = -1; hipDeviceGetP2PAttribute(&rank, hipDevP2PAttrPerformanceRank, i, j);
      int atomics = -1; hipDeviceGetP2PAttribute(&atomics, hipDevP2PAttrNativeAtomicSupported, i, j);
      printf("canAccessPeer(%d -> %d) = %d  perfRank=%d nativeAtomics=%d\n", i, j, can, rank, atomics);
    }
    void *b0, *b1;
    CK(hipSetDevice(0)); CK(hipMalloc(&b0, GiB)); CK(hipMemset(b0, 1, GiB));
    CK(hipSetDevice(1)); CK(hipMalloc(&b1, GiB)); CK(hipMemset(b1, 2, GiB));
    int can01 = 0, can10 = 0; CK(hipDeviceCanAccessPeer(&can01, 0, 1)); CK(hipDeviceCanAccessPeer(&can10, 1, 0));
    if (can01) { CK(hipSetDevice(0)); hipError_t e = hipDeviceEnablePeerAccess(1, 0); if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) printf("enable 0->1: %s\n", hipGetErrorString(e)); }
    if (can10) { CK(hipSetDevice(1)); hipError_t e = hipDeviceEnablePeerAccess(0, 0); if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) printf("enable 1->0: %s\n", hipGetErrorString(e)); }
    CK(hipSetDevice(0));
    float ms = best_ms([&] { CK(hipMemcpyPeerAsync(b1, 1, b0, 0, GiB, 0)); });
    printf("P2P hipMemcpyPeer 0->1 1 GiB: best %.3f ms  %.1f GB/s\n", ms, GiB / ms / 1e6);
    CK(hipSetDevice(1));
    ms = best_ms([&] { CK(hipMemcpyPeerAsync(b0, 0, b1, 1, GiB, 0)); });
    printf("P2P hipMemcpyPeer 1->0 1 GiB: best %.3f ms  %.1f GB/s\n", ms, GiB / ms / 1e6);
    // direct kernel P2P: dev0 kernel reads dev1 memory (needs peer access) -> remote read bandwidth
    if (can01) {
      CK(hipSetDevice(0));
      float* o; CK(hipMalloc(&o, 4));
      int cus; CK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0));
      size_t n4 = GiB / 16;
      ms = best_ms([&] { CK(hipMemsetAsync(o, 0, 4)); reduce4<<<cus * 16, 256>>>((const float4*)b1, o, n4); });
      printf("P2P kernel remote read (dev0 reads dev1) 1 GiB: best %.3f ms  %.1f GB/s\n", ms, GiB / ms / 1e6);
      // small-message latency of peer copy (one-shot all-reduce relevant): 64 KiB
      ms = best_ms([&] { CK(hipMemcpyPeerAsync(b1, 1, b0, 0, 64 << 10, 0)); }, 50);
      printf("P2P hipMemcpyPeer 64 KiB: best %.1f us\n", ms * 1000);
    }
  } else {
    printf("only %d device visible; P2P skipped\n", ndev);
  }
  return 0;
}
