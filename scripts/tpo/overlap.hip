// Lane TPO: do kernels of two streams run side by side on one R9700 (gfx1201) in the engine's pattern, under GPU_MAX_HW_QUEUES = 1 / 2 / 4,
// with a plain comm stream and with a CU-masked comm stream (hipExtStreamCreateWithCUMask)?
// Pattern per iteration (the engine's): s1: G1 ; record e1 ; s2: wait e1 ; B (all-reduce-like, 16 blocks) ; record e2 ; s1: G2 ; (next iteration:) s1 waits e2.
// G = a GEMM-like kernel (512 threads, 60 KB LDS, 2 x CUs x 2 blocks, each block spins `gus` us), B = 16 blocks of 512 threads spinning `bus` us.
// Arms: nocomm (no B), serial (B on s1 in line), overlap (B on s2 as above). Time of NIT iterations in ms.
// Build: hipcc -O2 --offload-arch=gfx1201 overlap.hip -o overlap   Run: GPU_MAX_HW_QUEUES=2 HIP_VISIBLE_DEVICES=0 ./overlap
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(1); } } while (0)
__global__ void spinG(long long ticks, float* sink) {
  __shared__ float lds[15 * 1024];
  lds[threadIdx.x] = threadIdx.x;
  long long t0 = wall_clock64();
  float x = lds[(threadIdx.x * 7) & 1023];
  while (wall_clock64() - t0 < ticks) x = x * 1.0000001f + 1.0f;
  if (x == 123.f) sink[0] = x;
}
__global__ void spinB(long long ticks, float* sink) {
  long long t0 = wall_clock64();
  float x = threadIdx.x;
  while (wall_clock64() - t0 < ticks) x = x * 1.0000001f + 1.0f;
  if (x == 123.f) sink[1] = x;
}
int main(int argc, char** argv) {
  CK(hipSetDevice(0));
  hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
  int clk = 0; CK(hipDeviceGetAttribute(&clk, hipDeviceAttributeWallClockRate, 0)); // kHz = ticks per ms
  const char* q = getenv("GPU_MAX_HW_QUEUES");
  printf("%s CUs %d wall clock %d kHz GPU_MAX_HW_QUEUES=%s\n", p.name, p.multiProcessorCount, clk, q ? q : "(unset)");
  float* sink; CK(hipMalloc(&sink, 64));
  const int gblocks = p.multiProcessorCount * 2; // 2 blocks of 512 threads + 60 KB LDS per WGP: one full wave of blocks
  const int NIT = 40;
  const long long gticks = (long long)clk * 400 / 1000, bticks = (long long)clk * 250 / 1000; // 0.4 ms (x2 in this clock) and 0.25 ms
  hipStream_t s1; CK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
  for (int masked = 0; masked < 2; ++masked) {
    hipStream_t s2;
    if (!masked) CK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
    else {
      // all CUs except the first quarter's worth would still leave room for G; here: the comm stream may use CUs 0..(n/2-1) only
      uint32_t mask[4] = {0, 0, 0, 0};
      for (int i = 0; i < p.multiProcessorCount && i < 128; ++i) if (i < (int)p.multiProcessorCount / 2) mask[i / 32] |= 1u << (i % 32);
      hipError_t e = hipExtStreamCreateWithCUMask(&s2, 4, mask);
      if (e != hipSuccess) { printf("CU-masked stream: %s\n", hipGetErrorString(e)); continue; }
    }
    hipEvent_t e1, e2, t0, t1; CK(hipEventCreateWithFlags(&e1, hipEventDisableTiming)); CK(hipEventCreateWithFlags(&e2, hipEventDisableTiming));
    CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
    for (int arm = 0; arm < 3; ++arm) {
      float best = 1e9;
      for (int rep = 0; rep < 5; ++rep) {
        CK(hipDeviceSynchronize());
        CK(hipEventRecord(t0, s1));
        for (int it = 0; it < NIT; ++it) {
          hipLaunchKernelGGL(spinG, dim3(gblocks), dim3(512), 0, s1, gticks, sink);  // G1
          if (arm == 1) hipLaunchKernelGGL(spinB, dim3(16), dim3(512), 0, s1, bticks, sink);  // serial
          if (arm == 2) {
            CK(hipEventRecord(e1, s1)); CK(hipStreamWaitEvent(s2, e1, 0));
            hipLaunchKernelGGL(spinB, dim3(16), dim3(512), 0, s2, bticks, sink);
            CK(hipEventRecord(e2, s2));
          }
          hipLaunchKernelGGL(spinG, dim3(gblocks), dim3(512), 0, s1, gticks, sink);  // G2 (independent of B)
          if (arm == 2) CK(hipStreamWaitEvent(s1, e2, 0)); // join before the next iteration
        }
        CK(hipEventRecord(t1, s1)); CK(hipEventSynchronize(t1));
        float ms; CK(hipEventElapsedTime(&ms, t0, t1)); if (ms < best) best = ms;
      }
      const char* names[] = {"no comm (G G only)", "comm in line on s1 (serial)", "comm on s2 (overlap)"};
      printf("  %s comm stream, %-30s %.3f ms for %d iterations, %.3f ms per iteration\n", masked ? "CU-masked" : "plain    ", names[arm], best, NIT, best / NIT);
    }
  }
  return 0;
}
