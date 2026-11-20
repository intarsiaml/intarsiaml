// storeprefault: which preparation of a NEW host buffer makes the prefix store's device-to-host download fast, and can it be
// done on a helper thread ahead of time. Sizes are the store's object sizes on Qwen3.6-35B-A3B (512 KiB head page, 5120 KiB KV
// page, 32160 KiB recurrent blob). Every buffer stays alive until the end of its size (pages are really new).
// Per mode: prep = ms per buffer on the helper thread (off the critical path), copy1 = the store's download into it (main
// thread, stream A), copy2 = the same download again (the reused floor).
//   (buffers come from mmap and go back with munmap, so every one is new pages; d2h3-calloc.txt was a first run with calloc/free,
//   whose later modes reused freed pages)
//   A  calloc + one write per 4 KiB page (today's host_buf)
//   H  as A, plus madvise(MADV_HUGEPAGE) on the 2 MiB-aligned interior before the touch (THP is `madvise` on spark2)
//   P  as A, plus one priming D2H into it from a scratch device buffer on the helper's own stream B (then synchronised)
//   R  as A, plus one H2D FROM it on stream B (the GPU reads the pages instead of writing them)
//   G  as A, plus cudaHostRegister (kept registered; unregistered at the end)
//   HP H + P
// Then a contention check: 40 D2D copies of 512 MiB on stream A (memory bound, like a prefill) alone, and while the helper
// prepares 2 GiB of P buffers.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <sys/mman.h>
#include <cuda_runtime.h>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() * 1e3; }
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
static void touch(unsigned char* p, size_t n) { for (size_t i = 0; i < n; i += 4096) ((volatile unsigned char*)p)[i] = 0; }
static void huge(unsigned char* p, size_t n) {
  uintptr_t a = ((uintptr_t)p + (2u << 20) - 1) & ~(uintptr_t)((2u << 20) - 1), e = ((uintptr_t)p + n) & ~(uintptr_t)((2u << 20) - 1);
  if (e > a) madvise((void*)a, e - a, MADV_HUGEPAGE);
}
static void* dscratch; static cudaStream_t sA, sB;
static unsigned char* prep(char mode, size_t n) {
  unsigned char* p = (unsigned char*)mmap(0, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);  // always new pages (calloc/free would reuse them)
  if (mode == 'H' || mode == 'h') huge(p, n);
  touch(p, n);
  if (mode == 'P' || mode == 'h') { CK(cudaMemcpyAsync(p, dscratch, n, cudaMemcpyDeviceToHost, sB)); CK(cudaStreamSynchronize(sB)); }
  if (mode == 'R') { CK(cudaMemcpyAsync(dscratch, p, n, cudaMemcpyHostToDevice, sB)); CK(cudaStreamSynchronize(sB)); }
  if (mode == 'G') CK(cudaHostRegister(p, n, 0));
  return p;
}
int main() {
  CK(cudaSetDevice(0));
  const size_t total = 1ull << 30;
  void* d; CK(cudaMalloc(&d, total)); CK(cudaMemset(d, 1, total));
  CK(cudaMalloc(&dscratch, 64ull << 20)); CK(cudaMemset(dscratch, 2, 64ull << 20));
  CK(cudaStreamCreateWithFlags(&sA, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));
  for (int round = 0; round < 2; round++)
  for (size_t kb : {512ul, 5120ul, 32160ul}) {
    size_t n = kb << 10; int reps = (int)(total / n); if (reps > 64) reps = 64;
    printf("round %d %6zu KiB x%-3d ms/buf (prep on helper thread | copy1 | copy2):", round, kb, reps);
    for (char mode : {'A', 'H', 'P', 'R', 'G', 'h'}) {
      std::vector<unsigned char*> bufs(reps);
      double tp = 0, c1 = 0, c2 = 0;
      std::thread th([&] { CK(cudaSetDevice(0)); double t = now(); for (int i = 0; i < reps; i++) bufs[i] = prep(mode, n); tp = now() - t; });
      th.join();
      for (int i = 0; i < reps; i++) { double t = now(); CK(cudaMemcpyAsync(bufs[i], (char*)d + (size_t)i * n, n, cudaMemcpyDeviceToHost, sA)); CK(cudaStreamSynchronize(sA)); c1 += now() - t; }
      for (int i = 0; i < reps; i++) { double t = now(); CK(cudaMemcpyAsync(bufs[i], (char*)d + (size_t)i * n, n, cudaMemcpyDeviceToHost, sA)); CK(cudaStreamSynchronize(sA)); c2 += now() - t; }
      for (auto p : bufs) { if (mode == 'G') CK(cudaHostUnregister(p)); munmap(p, n); }
      printf("  %c %6.3f|%6.3f|%6.3f", mode == 'h' ? 'Q' : mode, tp / reps, c1 / reps, c2 / reps);
    }
    printf("\n"); fflush(stdout);
  }
  // contention: main-stream memory-bound work alone vs with the helper preparing P buffers of 5120 KiB
  void* d2; CK(cudaMalloc(&d2, 512ull << 20));
  for (int rep = 0; rep < 2; rep++) {
    double t = now(); for (int i = 0; i < 40; i++) CK(cudaMemcpyAsync(d2, d, 512ull << 20, cudaMemcpyDeviceToDevice, sA)); CK(cudaStreamSynchronize(sA)); double alone = now() - t;
    std::atomic<bool> stop{false}; std::vector<unsigned char*> keep; double hb = 0; int hn = 0;
    std::thread th([&] { CK(cudaSetDevice(0)); double t0 = now(); while (!stop && hn < 400) { keep.push_back(prep('P', 5120ul << 10)); hn++; } hb = now() - t0; });
    t = now(); for (int i = 0; i < 40; i++) CK(cudaMemcpyAsync(d2, d, 512ull << 20, cudaMemcpyDeviceToDevice, sA)); CK(cudaStreamSynchronize(sA)); double with = now() - t;
    stop = true; th.join();
    for (auto p : keep) munmap(p, 5120ul << 10);
    printf("contention rep %d: 40 x 512 MiB D2D on stream A alone %.1f ms, with helper priming %.1f ms (helper made %d x 5 MiB = %.0f MiB in %.0f ms)\n", rep, alone, with, hn, hn * 5.0, hb);
  }
  return 0;
}
