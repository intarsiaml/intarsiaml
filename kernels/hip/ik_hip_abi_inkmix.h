/* Launchers of the Inkling-2-Mini in-projection and sublayer-boundary ops: InkInProj (73), ConvMix (74), AttnResMix (67), SconvRes (69), LoopBlend (68) on HIP
 * (lane ink-hip-mix; crates/intarsia-backend-hip/src/inkmix.rs holds the table, kernels.rs checks that every launcher of that table is
 * declared here). Contract: crates/intarsia-core/src/inkling2/ops.rs. Every launcher returns an ik_status, never synchronises
 * and never allocates. */
#ifndef IK_HIP_ABI_INKMIX_H
#define IK_HIP_ABI_INKMIX_H

#include "../abi/ik_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif
