/* Launchers of the Inkling-2-Mini attention ops: InkKvPrologue (70), RelSinkAttn (71), InkAttnDecode (75) on HIP
 * (lane ink-hip-attn; crates/intarsia-backend-hip/src/inkattn.rs holds the table, kernels.rs checks that every launcher of that table is
 * declared here). Contract: crates/intarsia-core/src/inkling2/ops.rs. Every launcher returns an ik_status, never synchronises
 * and never allocates. */
#ifndef IK_HIP_ABI_INKATTN_H
#define IK_HIP_ABI_INKATTN_H

#include "../abi/ik_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif
