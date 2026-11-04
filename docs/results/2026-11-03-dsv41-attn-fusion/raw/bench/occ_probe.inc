
// ---- bench-only occupancy probe, appended to both the unmodified and the variant copy of dsv41_attn.cu ----
#include <stdio.h>
extern "C" int ik_attn_mma_probe(int64_t D) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, attn_sparse_mma);
    int dev = 0, smsm = 0, resv = 0, optin = 0, maxthr = 0, nsm = 0, perblk = 0, regs = 0, blocks = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&smsm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    cudaDeviceGetAttribute(&resv, cudaDevAttrReservedSharedMemoryPerBlock, dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&perblk, cudaDevAttrMaxSharedMemoryPerBlock, dev);
    cudaDeviceGetAttribute(&maxthr, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&regs, cudaDevAttrMaxRegistersPerMultiprocessor, dev);
    cudaFuncSetAttribute(attn_sparse_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)attn_mma_smem(kMaxD));
    const size_t dyn = attn_mma_smem(D);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, attn_sparse_mma, kThreads, dyn);
    printf("probe: SMs %d | smem/SM %d B, reserved/block %d B, per-block max %d B, optin %d B | threads/SM %d, regs/SM %d | kernel: regs/thread %d, static smem %zu B, dynamic smem %zu B, "
           "total %zu B | active blocks per SM %d\n",
           nsm, smsm, resv, perblk, optin, maxthr, regs, fa.numRegs, (size_t)fa.sharedSizeBytes, dyn, (size_t)fa.sharedSizeBytes + dyn, blocks);
    return blocks;
}
