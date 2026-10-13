/* Declarations shared between the launcher files and ik_runtime.cu (not part of the public ABI). */
#ifndef IK_INTERNAL_H
#define IK_INTERNAL_H
#ifdef __cplusplus
extern "C" {
#endif
/* cuBLAS handle + workspace for the current device; created once, outside any launcher. */
int ik_cublas_init(int device);
void ik_cublas_shutdown(int device);
#ifdef __cplusplus
}
#endif
#endif
