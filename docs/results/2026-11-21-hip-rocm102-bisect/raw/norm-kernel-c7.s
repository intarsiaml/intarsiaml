_ZN12_GLOBAL__N_121gdn_chunk_norm_kernelEPK7ik_bf16ffPfS3_ii: ; @_ZN12_GLOBAL__N_121gdn_chunk_norm_kernelEPK7ik_bf16ffPfS3_ii
; %bb.0:
	s_clause 0x1
	s_load_b64 s[8:9], s[0:1], 0x20
	s_load_b64 s[4:5], s[0:1], 0x0
	s_mov_b32 s2, ttmp9
	s_mov_b32 s3, 0
	s_lshl_b32 s6, ttmp7, 7
	v_lshlrev_b32_e32 v3, 3, v0
	s_ashr_i32 s7, s6, 31
	v_mbcnt_lo_u32_b32 v5, -1, 0
	s_lshl_b64 s[6:7], s[6:7], 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v6, 16, v5
	v_cmp_gt_i32_e32 vcc_lo, 32, v6
	s_wait_kmcnt 0x0
	s_ashr_i32 s11, s8, 31
	s_mov_b32 s10, s8
	s_add_co_i32 s8, s9, ttmp7
	s_mul_u64 s[10:11], s[10:11], s[2:3]
	s_lshl_b32 s12, s8, 7
	s_lshl_b64 s[10:11], s[10:11], 1
	s_ashr_i32 s13, s12, 31
	s_add_nc_u64 s[4:5], s[4:5], s[10:11]
	s_lshl_b64 s[10:11], s[12:13], 1
	s_add_nc_u64 s[6:7], s[4:5], s[6:7]
	s_add_nc_u64 s[4:5], s[4:5], s[10:11]
	global_load_b64 v[1:2], v3, s[6:7]
	v_cndmask_b32_e32 v6, v5, v6, vcc_lo
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v8, 16, v2
	v_and_b32_e32 v2, 0xffff0000, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v8, v8, v8 :: v_dual_lshlrev_b32 v7, 16, v1
	v_mul_f32_e32 v2, v2, v2
	global_load_b64 v[3:4], v3, s[4:5]
	v_mul_f32_e32 v7, v7, v7
	s_mov_b32 s4, exec_lo
	v_add_f32_e32 v2, v8, v2
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v10, 16, v4
	v_and_b32_e32 v4, 0xffff0000, v4
	v_lshlrev_b32_e32 v9, 16, v3
	v_and_b32_e32 v3, 0xffff0000, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v3, v3, v3 :: v_dual_mul_f32 v4, v4, v4
	v_mul_f32_e32 v9, v9, v9
	v_dual_mul_f32 v10, v10, v10 :: v_dual_and_b32 v1, 0xffff0000, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v9, v3
	v_dual_mul_f32 v1, v1, v1 :: v_dual_add_f32 v4, v10, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v7, v1
	v_dual_add_f32 v1, v1, v2 :: v_dual_add_f32 v2, v3, v4
	v_lshlrev_b32_e32 v6, 2, v6
	ds_bpermute_b32 v3, v6, v1
	s_wait_dscnt 0x0
	v_add_f32_e32 v1, v1, v3
	ds_bpermute_b32 v4, v6, v2
	v_xor_b32_e32 v6, 8, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v6
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v6, v5, v6, vcc_lo
	v_lshlrev_b32_e32 v6, 2, v6
	ds_bpermute_b32 v3, v6, v1
	s_wait_dscnt 0x0
	v_dual_add_f32 v1, v1, v3 :: v_dual_add_f32 v2, v2, v4
	ds_bpermute_b32 v4, v6, v2
	v_xor_b32_e32 v6, 4, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v6
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v6, v5, v6, vcc_lo
	v_lshlrev_b32_e32 v6, 2, v6
	ds_bpermute_b32 v3, v6, v1
	s_wait_dscnt 0x0
	v_dual_add_f32 v1, v1, v3 :: v_dual_add_f32 v2, v2, v4
	ds_bpermute_b32 v4, v6, v2
	v_xor_b32_e32 v6, 2, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v6
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v6, v5, v6, vcc_lo
	v_lshlrev_b32_e32 v6, 2, v6
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v2, v4
	ds_bpermute_b32 v3, v6, v1
	ds_bpermute_b32 v4, v6, v2
	v_xor_b32_e32 v6, 1, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v6
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v5, v5, v6, vcc_lo
	v_lshlrev_b32_e32 v5, 2, v5
	s_wait_dscnt 0x0
	v_dual_add_f32 v1, v1, v3 :: v_dual_add_f32 v2, v2, v4
	ds_bpermute_b32 v3, v5, v1
	ds_bpermute_b32 v4, v5, v2
	v_cmpx_eq_u32_e32 0, v0
	s_cbranch_execz .LBB1_2
; %bb.1:
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x8
	s_load_b64 s[10:11], s[0:1], 0x18
	s_wait_dscnt 0x0
	v_dual_add_f32 v0, v1, v3 :: v_dual_add_f32 v1, v2, v4
	s_mov_b32 s8, ttmp7
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_add_f32 v0, s4, v0 :: v_dual_add_f32 v1, s4, v1
	s_mov_b32 s4, s9
	v_dual_mul_f32 v2, 0x4b800000, v0 :: v_dual_mul_f32 v3, 0x4b800000, v1
	v_cmp_gt_f32_e64 s0, 0x800000, v1
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v1, v3, s0
	v_rsq_f32_e32 v1, v1
	s_wait_alu 0xfffd
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v0, v0, v2 :: v_dual_mul_f32 v3, 0x45800000, v1
	v_rsq_f32_e32 v0, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e64 v1, v1, v3, s0
	v_mul_f32_e32 v2, 0x45800000, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v0, v0, v2, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v1
	s_wait_alu 0xfffd
	v_cndmask_b32_e64 v2, 0, 32, vcc_lo
	v_cndmask_b32_e64 v4, 0, 0x41b17218, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ldexp_f32 v1, v1, v2
	v_log_f32_e32 v1, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v2, 0x3f317217, v1
	v_cmp_gt_f32_e64 vcc_lo, 0x7f800000, |v1|
	v_fma_f32 v2, 0x3f317217, v1, -v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, 0x3377d1cf, v1
	v_fmac_f32_e32 v2, 0x3f317217, v1
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v1, v1, v2 :: v_dual_mov_b32 v2, 0
	v_sub_f32_e32 v1, v1, v4
	v_mul_f32_e32 v0, s5, v0
	s_ashr_i32 s5, s9, 31
	s_ashr_i32 s9, ttmp7, 31
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_f32_e64 s0, 0x800000, v0
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v3, 0, 32, s0
	v_cndmask_b32_e64 v5, 0, 0x41b17218, s0
	s_wait_alu 0xfffe
	s_mul_u64 s[0:1], s[4:5], s[2:3]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[0:1], s[0:1], s[8:9]
	v_ldexp_f32 v0, v0, v3
	s_wait_alu 0xfffe
	s_lshl_b64 s[0:1], s[0:1], 2
	s_wait_alu 0xfffe
	s_add_nc_u64 s[2:3], s[6:7], s[0:1]
	s_add_nc_u64 s[0:1], s[10:11], s[0:1]
	v_log_f32_e32 v0, v0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v3, 0x3f317217, v0
	v_cmp_gt_f32_e64 vcc_lo, 0x7f800000, |v0|
	v_fma_f32 v3, 0x3f317217, v0, -v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v3, 0x3377d1cf, v0
	v_fmac_f32_e32 v3, 0x3f317217, v0
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v0, v0, v3, vcc_lo
	v_sub_f32_e32 v0, v0, v5
	s_clause 0x1
	global_store_b32 v2, v0, s[2:3]
	global_store_b32 v2, v1, s[0:1]
.LBB1_2:
	s_endpgm
