// storettft: device -> host copy rate on the GB10 for the sizes the prefix store's settle moves.
// Modes: reuse = one pageable pre-touched destination reused; fresh = a new pageable destination per copy, page-touched like the store's
// host_buf (calloc + one volatile write per 4 KiB page), only the copy timed; cycle64 = 64 distinct touched destinations in turn;
// pinned = cudaHostAlloc destination reused; freshreg = fresh destination, cudaHostRegister before and Unregister after (copy timed
// only); src_dirty = reuse, but the source is rewritten by a D2D copy right before each D2H (as a capture is).
// nvcc -O2 -o d2h d2h.cu && ./d2h        (run under fleetq, --quiet gpu)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
static unsigned char* touched(size_t n) {
  unsigned char* p = (unsigned char*)calloc(n, 1);
  for (size_t i = 0; i < n; i += 4096) ((volatile unsigned char*)p)[i] = 0;
  return p;
}
int main() {
  const size_t total = 1ull << 30;
  void* d; CK(cudaMalloc(&d, total)); CK(cudaMemset(d, 1, total));
  void* d2; CK(cudaMalloc(&d2, total)); CK(cudaMemset(d2, 2, total));
  cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  for (size_t mb : {1ul, 5ul, 32ul}) {
    size_t n = mb << 20; int reps = (int)(total / n);
    double t, g = (double)total / (1 << 30), reuse, fresh = 0, cyc, pin, freg = 0, dirty;
    unsigned char* p = touched(n);
    t = now(); for (int i = 0; i < reps; i++) { CK(cudaMemcpyAsync(p, (char*)d + (size_t)i * n, n, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); } reuse = now() - t;
    for (int i = 0; i < reps; i++) {
      unsigned char* q = touched(n);
      t = now(); CK(cudaMemcpyAsync(q, (char*)d + (size_t)i * n, n, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); fresh += now() - t;
      free(q);
    }
    std::vector<unsigned char*> bufs; for (int i = 0; i < 64; i++) bufs.push_back(touched(n));
    t = now(); for (int i = 0; i < reps; i++) { CK(cudaMemcpyAsync(bufs[i % 64], (char*)d + (size_t)i * n, n, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); } cyc = now() - t;
    for (auto b : bufs) free(b);
    void* h; CK(cudaHostAlloc(&h, n, cudaHostAllocDefault));
    t = now(); for (int i = 0; i < reps; i++) { CK(cudaMemcpyAsync(h, (char*)d + (size_t)i * n, n, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); } pin = now() - t;
    CK(cudaFreeHost(h));
    for (int i = 0; i < reps; i++) {
      unsigned char* q = touched(n); CK(cudaHostRegister(q, n, 0));
      t = now(); CK(cudaMemcpyAsync(q, (char*)d + (size_t)i * n, n, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); freg += now() - t;
      CK(cudaHostUnregister(q)); free(q);
    }
    dirty = 0;
    for (int i = 0; i < reps; i++) {
      CK(cudaMemcpyAsync((char*)d2 + (size_t)i * n, (char*)d + (size_t)i * n, n, cudaMemcpyDeviceToDevice, s));
      t = now(); CK(cudaMemcpyAsync(p, (char*)d2 + (size_t)i * n, n, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); dirty += now() - t;
    }
    free(p);
    printf("%3zu MiB x%-4d GB/s: reuse %6.2f | fresh-touched %6.2f | cycle64 %6.2f | pinned %6.2f | fresh+register %6.2f | reuse after D2D write %6.2f   (fresh copy %.2f ms each)\n",
           mb, reps, g / reuse, g / fresh, g / cyc, g / pin, g / freg, g / dirty, fresh / reps * 1e3);
  }
  return 0;
}
