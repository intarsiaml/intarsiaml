// storettft: all-in cost (allocate + page-touch + register/pin + copy + unregister) of landing a device buffer in a NEW host buffer, per size.
// A: calloc + touch + copy (the store's host_buf + download today). B: calloc + touch + cudaHostRegister + copy + cudaHostUnregister.
// E: mmap MAP_POPULATE + copy; F: E + register + unregister.
// C: cudaHostAlloc + copy. D: cudaHostAlloc once and reuse (the floor). Times in ms per buffer, mean over reps; the buffers are kept alive until
// the end of each size (so pages are really fresh) and freed afterwards.
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <sys/mman.h>
#include <cuda_runtime.h>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() * 1e3; }
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
static unsigned char* populated(size_t n) { void* p = mmap(0, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0); if (p == MAP_FAILED) { puts("mmap"); exit(1); } return (unsigned char*)p; }
static unsigned char* touched(size_t n) { unsigned char* p = (unsigned char*)malloc(n); for (size_t i = 0; i < n; i += 4096) ((volatile unsigned char*)p)[i] = 0; return p; }
int main() {
  const size_t total = 1ull << 30;
  void* d; CK(cudaMalloc(&d, total)); CK(cudaMemset(d, 1, total));
  cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  for (int round = 0; round < 2; round++)
  for (size_t kb : {512ul, 5120ul, 32160ul, 102400ul}) {
    size_t n = kb << 10; int reps = (int)(total / n); if (reps > 40) reps = 40;
    double a = 0, b = 0, c = 0, dd = 0, e = 0, f = 0, t;
    std::vector<unsigned char*> keep;
    for (int i = 0; i < reps; i++) { t = now(); unsigned char* p = touched(n); CK(cudaMemcpyAsync(p, (char*)d + (size_t)i * n, n, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); a += now() - t; keep.push_back(p); }
    for (auto p : keep) free(p); keep.clear();
    for (int i = 0; i < reps; i++) { t = now(); unsigned char* p = touched(n); CK(cudaHostRegister(p, n, 0)); CK(cudaMemcpyAsync(p, (char*)d + (size_t)i * n, n, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); CK(cudaHostUnregister(p)); b += now() - t; keep.push_back(p); }
    for (auto p : keep) free(p); keep.clear();
    for (int i = 0; i < reps; i++) { t = now(); unsigned char* p = populated(n); CK(cudaMemcpyAsync(p, (char*)d + (size_t)i * n, n, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); e += now() - t; keep.push_back(p); }
    for (auto p : keep) munmap(p, n); keep.clear();
    for (int i = 0; i < reps; i++) { t = now(); unsigned char* p = populated(n); CK(cudaHostRegister(p, n, 0)); CK(cudaMemcpyAsync(p, (char*)d + (size_t)i * n, n, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); CK(cudaHostUnregister(p)); f += now() - t; keep.push_back(p); }
    for (auto p : keep) munmap(p, n); keep.clear();
    std::vector<void*> hk;
    for (int i = 0; i < reps; i++) { t = now(); void* p; CK(cudaHostAlloc(&p, n, cudaHostAllocDefault)); CK(cudaMemcpyAsync(p, (char*)d + (size_t)i * n, n, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); c += now() - t; hk.push_back(p); }
    for (auto p : hk) cudaFreeHost(p);
    void* p; CK(cudaHostAlloc(&p, n, cudaHostAllocDefault));
    for (int i = 0; i < reps; i++) { t = now(); CK(cudaMemcpyAsync(p, (char*)d + (size_t)i * n, n, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); dd += now() - t; }
    cudaFreeHost(p);
    printf("round %d %6zu KiB x%-3d ms each: A touch+copy %7.3f | B touch+reg+copy+unreg %7.3f | C hostalloc+copy %7.3f | D reused pinned copy %7.3f | E populate+copy %7.3f | F populate+reg+copy+unreg %7.3f\n", round, kb, reps, a / reps, b / reps, c / reps, dd / reps, e / reps, f / reps);
  }
  return 0;
}
