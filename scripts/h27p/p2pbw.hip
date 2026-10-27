// Lane H27P: peer-to-peer bandwidth between the two R9700s: copy engine vs kernel stores, coarse vs fine-grained destination,
// one direction vs both at once, by block count. Build: hipcc -O2 --offload-arch=gfx1201 p2pbw.hip -o p2pbw
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); return 1; } } while (0)
__global__ void push(const int4* __restrict__ src, int4* __restrict__ dst, size_t n16) {
  size_t nt = (size_t)gridDim.x * blockDim.x;
  size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; k + 3 * nt < n16; k += 4 * nt) { int4 a = src[k], b = src[k + nt], c = src[k + 2 * nt], d = src[k + 3 * nt]; dst[k] = a; dst[k + nt] = b; dst[k + 2 * nt] = c; dst[k + 3 * nt] = d; }
  for (; k < n16; k += nt) dst[k] = src[k];
}
static double ms(hipEvent_t a, hipEvent_t b) { float m; hipEventElapsedTime(&m, a, b); return m; }
int main() {
  const size_t bytes = 32u << 20; // one prefill-sized partial (bf16 3858x5120 is 39.5 MB f32 / 19.7 MB bf16)
  void *src[2], *dstc[2], *dstf[2];
  hipStream_t st[2]; hipEvent_t e0[2], e1[2];
  for (int d = 0; d < 2; ++d) {
    CK(hipSetDevice(d)); CK(hipDeviceEnablePeerAccess(1 - d, 0));
    CK(hipMalloc(&src[d], bytes)); CK(hipMalloc(&dstc[d], bytes)); CK(hipExtMallocWithFlags(&dstf[d], bytes, hipDeviceMallocFinegrained));
    CK(hipMemset(src[d], 1, bytes)); CK(hipStreamCreate(&st[d])); CK(hipEventCreate(&e0[d])); CK(hipEventCreate(&e1[d]));
  }
  for (int mode = 0; mode < 6; ++mode) {
    const char* names[] = {"memcpyPeer coarse 1dir", "memcpyPeer coarse 2dir", "kernel->coarse 1dir", "kernel->coarse 2dir", "kernel->finegrained 1dir", "kernel->finegrained 2dir"};
    for (int nb : {16, 64, 256}) {
      if (mode < 2 && nb != 16) continue;
      int ndir = (mode % 2) ? 2 : 1;
      double best = 1e9;
      for (int rep = 0; rep < 6; ++rep) {
        for (int d = 0; d < ndir; ++d) { CK(hipSetDevice(d)); CK(hipDeviceSynchronize()); }
        for (int d = 0; d < ndir; ++d) {
          CK(hipSetDevice(d)); CK(hipEventRecord(e0[d], st[d]));
          if (mode < 2) CK(hipMemcpyPeerAsync(dstc[1 - d], 1 - d, src[d], d, bytes, st[d]));
          else push<<<nb, 256, 0, st[d]>>>((const int4*)src[d], (int4*)(mode < 4 ? dstc[1 - d] : dstf[1 - d]), bytes / 16);
          CK(hipEventRecord(e1[d], st[d]));
        }
        double worst = 0;
        for (int d = 0; d < ndir; ++d) { CK(hipSetDevice(d)); CK(hipEventSynchronize(e1[d])); worst = std::max(worst, ms(e0[d], e1[d])); }
        if (rep > 0) best = std::min(best, worst);
      }
      printf("%-28s blocks %3d: %7.3f ms  %6.1f GB/s per direction\n", names[mode], mode < 2 ? 0 : nb, best, bytes / best / 1e6);
    }
  }
  return 0;
}
