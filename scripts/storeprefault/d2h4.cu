// storeprefault: is the slow first download into a new host buffer about where the buffer starts (calloc/malloc put a 16-byte header
// before a large block, so the data starts 16 bytes into a page) or about when the pages were touched?
// Every buffer is new pages. Per mode, ms per buffer: prep (touch etc., same thread) | copy1 (first D2H) | copy2 (again).
//   M  mmap + touch, copy to the page-aligned start (all buffers prepared first, then copied)
//   O  as M, but the copy starts 16 bytes into the mapping (calloc's layout)
//   C  calloc + touch (what Rust's vec![0u8; n] + host_buf does)
//   I  mmap + touch + copy immediately, buffer by buffer (the store's demote: alloc then download)
//   J  as I, with the 16-byte offset
//   A  posix_memalign(4096) + touch, copied after all are prepared
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <sys/mman.h>
#include <cuda_runtime.h>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() * 1e3; }
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
static void touch(unsigned char* p, size_t n) { for (size_t i = 0; i < n; i += 4096) ((volatile unsigned char*)p)[i] = 0; }
int main() {
  const size_t total = 1ull << 30;
  void* d; CK(cudaMalloc(&d, total)); CK(cudaMemset(d, 1, total));
  cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  for (int round = 0; round < 2; round++)
  for (size_t kb : {5120ul, 32160ul}) {
    size_t n = kb << 10; int reps = (int)(total / n); if (reps > 48) reps = 48;
    printf("round %d %6zu KiB x%-3d ms/buf prep|copy1|copy2:", round, kb, reps);
    for (char mode : {'C', 'M', 'O', 'I', 'J', 'A'}) {
      std::vector<unsigned char*> base(reps), dst(reps);
      double tp = 0, c1 = 0, c2 = 0, t;
      bool imm = mode == 'I' || mode == 'J';
      for (int i = 0; i < reps; i++) {
        t = now();
        if (mode == 'C') base[i] = (unsigned char*)calloc(n, 1);
        else if (mode == 'A') { void* p; if (posix_memalign(&p, 4096, n)) exit(1); base[i] = (unsigned char*)p; }
        else base[i] = (unsigned char*)mmap(0, n + 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        dst[i] = base[i] + ((mode == 'O' || mode == 'J') ? 16 : 0);
        touch(base[i], n);
        tp += now() - t;
        if (imm) { t = now(); CK(cudaMemcpyAsync(dst[i], (char*)d + (size_t)i * n, n, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); c1 += now() - t; }
      }
      if (!imm) for (int i = 0; i < reps; i++) { t = now(); CK(cudaMemcpyAsync(dst[i], (char*)d + (size_t)i * n, n, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); c1 += now() - t; }
      for (int i = 0; i < reps; i++) { t = now(); CK(cudaMemcpyAsync(dst[i], (char*)d + (size_t)i * n, n, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); c2 += now() - t; }
      for (int i = 0; i < reps; i++) { if (mode == 'C' || mode == 'A') free(base[i]); else munmap(base[i], n + 4096); }
      printf("  %c %6.3f|%6.3f|%6.3f", mode, tp / reps, c1 / reps, c2 / reps);
    }
    printf("\n"); fflush(stdout);
  }
  return 0;
}
