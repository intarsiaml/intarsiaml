
// ---- bench-only occupancy probe, appended to the unmodified and to the variant copy of dsv41_attn.cu ----
// As ../../../2026-11-03-dsv41-attn-fusion/raw/bench/occ_probe.inc, plus: the probed kernel follows the build (attn_sparse_mma_tfb under
// -DATTN_TFBLOCK, whose block is kTfThreads threads), and the line also prints the kernel's local (spill) bytes.
#include <stdio.h>
extern "C" int ik_attn_mma_probe(int64_t D) {
#ifdef ATTN_TFBLOCK
#define PROBE_KERNEL attn_sparse_mma_tfb
#define PROBE_THREADS kTfThreads
#define PROBE_SMEM attn_mma_smem_tfb()
#else
#define PROBE_KERNEL attn_sparse_mma
#define PROBE_THREADS kThreads
#define PROBE_SMEM attn_mma_smem(D)
#endif
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, PROBE_KERNEL);
    int dev = 0, smsm = 0, resv = 0, optin = 0, maxthr = 0, nsm = 0, perblk = 0, regs = 0, blocks = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&smsm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    cudaDeviceGetAttribute(&resv, cudaDevAttrReservedSharedMemoryPerBlock, dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&perblk, cudaDevAttrMaxSharedMemoryPerBlock, dev);
    cudaDeviceGetAttribute(&maxthr, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&regs, cudaDevAttrMaxRegistersPerMultiprocessor, dev);
    const size_t dyn = PROBE_SMEM;
    cudaFuncSetAttribute(PROBE_KERNEL, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, PROBE_KERNEL, PROBE_THREADS, dyn);
    printf("probe: SMs %d | smem/SM %d B, reserved/block %d B, per-block max %d B, optin %d B | threads/SM %d, regs/SM %d | kernel: regs/thread %d, static smem %zu B, "
           "dynamic smem %zu B, total %zu B, local %zu B, block %d threads | active blocks per SM %d\n",
           nsm, smsm, resv, perblk, optin, maxthr, regs, fa.numRegs, (size_t)fa.sharedSizeBytes, dyn, (size_t)fa.sharedSizeBytes + dyn, (size_t)fa.localSizeBytes,
           (int)PROBE_THREADS, blocks);
    return blocks;
}
