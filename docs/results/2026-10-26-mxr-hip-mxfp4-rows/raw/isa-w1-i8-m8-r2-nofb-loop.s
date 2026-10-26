_Z18mxr_i8_rows_kernelILi8ELi2ELb1EfLi0EEvPK7ik_bf16PK15HIP_vector_typeIjLj4EEPKfPKhSA_PT2_li: ; @_Z18mxr_i8_rows_kernelILi8ELi2ELb1EfLi0EEvPK7ik_bf16PK15HIP_vector_typeIjLj4EEPKfPKhSA_PT2_li
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e32 16, v0
	s_cbranch_execz .LBB234_2
	v_and_b32_e32 v1, 7, v0
	v_lshlrev_b32_e32 v3, 28, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v2, v1, 22, 0.5
	v_cmp_ne_u32_e32 vcc_lo, 1, v1
	v_cndmask_b32_e32 v2, 0.5, v2, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0, v1
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v1, 0, v2 :: v_dual_lshlrev_b32 v2, 2, v0
	v_and_or_b32 v1, 0x80000000, v3, v1
	ds_store_b32 v2, v1
.LBB234_2:
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b64 s[2:3], s[0:1], 0x30
	v_lshrrev_b32_e32 v1, 4, v0
	s_mov_b32 s4, ttmp9
	s_mov_b32 s5, 0
	s_wait_dscnt 0x0
	s_lshl_b64 s[4:5], s[4:5], 4
	s_barrier_signal -1
	v_and_or_b32 v10, v1, 14, s4
	v_mov_b32_e32 v11, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_mov_b32 s4, exec_lo
	s_wait_kmcnt 0x0
	v_cmpx_gt_i64_e64 s[2:3], v[10:11]
	s_cbranch_execz .LBB234_51
	s_clause 0x1
	s_load_b32 s17, s[0:1], 0x38
	s_load_b64 s[12:13], s[0:1], 0x28
	v_dual_mov_b32 v39, 0 :: v_dual_and_b32 v8, 31, v0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v31, 0
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v9, 0
	v_mov_b32_e32 v26, 0
	s_mov_b32 s16, exec_lo
	s_wait_kmcnt 0x0
	s_ashr_i32 s4, s17, 31
	s_wait_alu 0xfffe
	s_lshr_b32 s4, s4, 27
	s_wait_alu 0xfffe
	s_add_co_i32 s4, s17, s4
	s_wait_alu 0xfffe
	s_ashr_i32 s14, s4, 5
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmpx_gt_i32_e64 s14, v8
	s_cbranch_execz .LBB234_19
	s_load_b256 s[4:11], s[0:1], 0x8
	s_add_nc_u64 s[18:19], s[2:3], -1
	v_add_co_u32 v2, s0, v10, 1
	v_cmp_gt_u64_e32 vcc_lo, s[18:19], v[10:11]
	v_add_co_ci_u32_e64 v3, null, 0, v11, s0
	v_dual_mov_b32 v9, 0 :: v_dual_lshlrev_b32 v0, 4, v8
	s_lshr_b32 s1, s17, 31
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v4, s18, v10, vcc_lo
	s_wait_alu 0xfffe
	s_add_co_i32 s17, s17, s1
	v_cndmask_b32_e32 v5, s19, v11, vcc_lo
	v_cmp_gt_u64_e32 vcc_lo, s[18:19], v[2:3]
	v_mov_b32_e32 v1, v9
	s_ashr_i32 s0, s17, 1
	s_ashr_i32 s15, s14, 31
	s_wait_alu 0xfffe
	s_ashr_i32 s1, s0, 31
	v_mul_lo_u32 v6, v4, s15
	v_mul_lo_u32 v7, v5, s14
	s_wait_kmcnt 0x0
	v_mad_co_u64_u32 v[12:13], null, v4, s14, s[10:11]
	s_wait_alu 0xfffe
	v_mul_lo_u32 v14, v4, s1
	v_mul_lo_u32 v15, v5, s0
	v_mad_co_u64_u32 v[4:5], null, v4, s0, v[0:1]
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v2, s18, v2 :: v_dual_mov_b32 v25, v9
	v_cndmask_b32_e32 v3, s19, v3, vcc_lo
	v_add3_u32 v13, v7, v13, v6
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v27, v9
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_u64_u32 v[0:1], null, v2, s0, v[0:1]
	v_add3_u32 v5, v15, v5, v14
	v_add_co_u32 v16, vcc_lo, s8, v4
	v_mul_lo_u32 v6, v2, s1
	v_mul_lo_u32 v20, v2, s15
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v17, null, s9, v5, vcc_lo
	v_add_co_u32 v18, vcc_lo, s8, v0
	v_lshlrev_b32_e32 v0, 2, v8
	v_mul_lo_u32 v7, v3, s0
	v_mul_lo_u32 v3, v3, s14
	v_mad_co_u64_u32 v[14:15], null, v2, s14, s[10:11]
	v_lshlrev_b32_e32 v2, 5, v8
	v_dual_mov_b32 v26, v9 :: v_dual_mov_b32 v29, v9
	v_dual_mov_b32 v28, v9 :: v_dual_mov_b32 v31, v9
	v_add3_u32 v1, v7, v1, v6
	v_add3_u32 v15, v3, v15, v20
	v_add_co_u32 v20, s0, s4, v2
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v21, null, s5, 0, s0
	v_add_co_u32 v22, s0, s6, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v19, null, s9, v1, vcc_lo
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v23, null, s7, 0, s0
	v_dual_mov_b32 v30, v9 :: v_dual_mov_b32 v33, v9
	v_dual_mov_b32 v32, v9 :: v_dual_mov_b32 v35, v9
	v_dual_mov_b32 v34, v9 :: v_dual_mov_b32 v37, v9
	v_dual_mov_b32 v36, v9 :: v_dual_mov_b32 v39, v9
	v_mov_b32_e32 v38, v9
	v_mov_b32_e32 v40, v9
	s_lshl_b64 s[4:5], s[14:15], 5
	s_lshl_b64 s[6:7], s[14:15], 2
	s_mov_b32 s1, 0
	s_mov_b32 s8, 0x1010101
	s_mov_b32 s9, 0xf4f8fafc
	s_mov_b32 s10, 0xc080604
	s_branch .LBB234_7
.LBB234_5:                              ;   in Loop: Header=BB234_7 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s11
.LBB234_6:                              ;   in Loop: Header=BB234_7 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v43, 3, v4
	v_lshrrev_b32_e32 v44, 4, v4
	v_lshrrev_b32_e32 v45, 7, v4
	v_and_b32_e32 v4, 0x7070707, v4
	v_lshrrev_b32_e32 v46, 3, v5
	v_and_or_b32 v43, v43, s8, 0xc0c0c0c
	v_and_b32_e32 v44, 0x7070707, v44
	v_and_or_b32 v45, v45, s8, 0xc0c0c0c
	v_perm_b32 v47, s9, 0xfdfeff00, v4
	v_perm_b32 v4, s10, 0x3020100, v4
	v_perm_b32 v43, 0, 0, v43
	v_lshrrev_b32_e32 v48, 7, v5
	v_perm_b32 v45, 0, 0, v45
	v_perm_b32 v49, s9, 0xfdfeff00, v44
	v_perm_b32 v44, s10, 0x3020100, v44
	v_bfi_b32 v47, v43, v47, v4
	v_lshrrev_b32_e32 v4, 4, v5
	v_and_or_b32 v43, v46, s8, 0xc0c0c0c
	v_and_b32_e32 v5, 0x7070707, v5
	v_and_or_b32 v46, v48, s8, 0xc0c0c0c
	v_bfi_b32 v48, v45, v49, v44
	v_and_b32_e32 v4, 0x7070707, v4
	v_perm_b32 v43, 0, 0, v43
	v_perm_b32 v45, s9, 0xfdfeff00, v5
	v_perm_b32 v5, s10, 0x3020100, v5
	v_perm_b32 v44, 0, 0, v46
	v_perm_b32 v46, s9, 0xfdfeff00, v4
	v_lshrrev_b32_e32 v49, 3, v6
	v_perm_b32 v4, s10, 0x3020100, v4
	v_lshrrev_b32_e32 v50, 7, v6
	v_bfi_b32 v51, v43, v45, v5
	v_lshrrev_b32_e32 v5, 4, v6
	v_and_or_b32 v43, v49, s8, 0xc0c0c0c
	v_bfi_b32 v49, v44, v46, v4
	v_and_or_b32 v4, v50, s8, 0xc0c0c0c
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v5, 0x7070707, v5
	v_lshrrev_b32_e32 v44, 3, v7
	v_perm_b32 v43, 0, 0, v43
	v_perm_b32 v4, 0, 0, v4
	v_perm_b32 v45, s9, 0xfdfeff00, v6
	v_perm_b32 v6, s10, 0x3020100, v6
	v_perm_b32 v46, s9, 0xfdfeff00, v5
	v_and_or_b32 v44, v44, s8, 0xc0c0c0c
	v_lshrrev_b32_e32 v50, 7, v7
	v_and_b32_e32 v52, 0x7070707, v7
	v_perm_b32 v5, s10, 0x3020100, v5
	v_lshrrev_b32_e32 v7, 4, v7
	v_perm_b32 v44, 0, 0, v44
	v_and_or_b32 v50, v50, s8, 0xc0c0c0c
	v_perm_b32 v53, s9, 0xfdfeff00, v52
	v_perm_b32 v52, s10, 0x3020100, v52
	v_bfi_b32 v54, v43, v45, v6
	v_and_b32_e32 v6, 0x7070707, v7
	v_bfi_b32 v55, v4, v46, v5
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v5, 3, v0
	v_perm_b32 v4, 0, 0, v50
	v_bfi_b32 v50, v44, v53, v52
	v_perm_b32 v7, s9, 0xfdfeff00, v6
	v_perm_b32 v6, s10, 0x3020100, v6
	v_lshrrev_b32_e32 v43, 7, v0
	v_lshrrev_b32_e32 v44, 4, v0
	v_and_or_b32 v5, v5, s8, 0xc0c0c0c
	v_and_b32_e32 v0, 0x7070707, v0
	v_bfi_b32 v52, v4, v7, v6
	v_and_or_b32 v4, v43, s8, 0xc0c0c0c
	v_and_b32_e32 v6, 0x7070707, v44
	v_perm_b32 v5, 0, 0, v5
	v_perm_b32 v7, s9, 0xfdfeff00, v0
	v_perm_b32 v0, s10, 0x3020100, v0
	v_lshrrev_b32_e32 v43, 3, v1
	v_perm_b32 v4, 0, 0, v4
	v_perm_b32 v44, s9, 0xfdfeff00, v6
	v_perm_b32 v6, s10, 0x3020100, v6
	v_bfi_b32 v53, v5, v7, v0
	v_and_or_b32 v5, v43, s8, 0xc0c0c0c
	v_and_b32_e32 v7, 0x7070707, v1
	v_lshrrev_b32_e32 v0, 7, v1
	v_bfi_b32 v56, v4, v44, v6
	v_lshrrev_b32_e32 v1, 4, v1
	v_perm_b32 v4, 0, 0, v5
	v_perm_b32 v5, s9, 0xfdfeff00, v7
	v_perm_b32 v6, s10, 0x3020100, v7
	v_lshrrev_b32_e32 v7, 3, v2
	v_lshrrev_b32_e32 v43, 7, v2
	v_lshrrev_b32_e32 v44, 3, v3
	v_lshrrev_b32_e32 v45, 7, v3
	v_bfi_b32 v57, v4, v5, v6
	v_lshrrev_b32_e32 v5, 4, v2
	v_lshrrev_b32_e32 v58, 4, v3
	v_and_or_b32 v0, v0, s8, 0xc0c0c0c
	v_and_b32_e32 v1, 0x7070707, v1
	v_and_or_b32 v6, v7, s8, 0xc0c0c0c
	v_and_or_b32 v7, v43, s8, 0xc0c0c0c
	v_and_b32_e32 v2, 0x7070707, v2
	v_and_b32_e32 v5, 0x7070707, v5
	v_and_or_b32 v44, v44, s8, 0xc0c0c0c
	v_and_or_b32 v45, v45, s8, 0xc0c0c0c
	v_and_b32_e32 v3, 0x7070707, v3
	v_and_b32_e32 v58, 0x7070707, v58
	v_perm_b32 v0, 0, 0, v0
	v_perm_b32 v4, s9, 0xfdfeff00, v1
	v_perm_b32 v1, s10, 0x3020100, v1
	v_perm_b32 v6, 0, 0, v6
	v_perm_b32 v7, 0, 0, v7
	v_perm_b32 v43, s9, 0xfdfeff00, v2
	v_perm_b32 v2, s10, 0x3020100, v2
	v_perm_b32 v46, s9, 0xfdfeff00, v5
	v_perm_b32 v5, s10, 0x3020100, v5
	v_perm_b32 v44, 0, 0, v44
	v_perm_b32 v45, 0, 0, v45
	v_perm_b32 v59, s9, 0xfdfeff00, v3
	v_perm_b32 v3, s10, 0x3020100, v3
	v_perm_b32 v60, s9, 0xfdfeff00, v58
	v_perm_b32 v58, s10, 0x3020100, v58
	v_bfi_b32 v61, v0, v4, v1
	v_bfi_b32 v62, v6, v43, v2
	v_bfi_b32 v63, v7, v46, v5
	v_bfi_b32 v59, v44, v59, v3
	v_bfi_b32 v58, v45, v60, v58
	s_clause 0x1
	global_load_b128 v[0:3], v[20:21], off
	global_load_b128 v[4:7], v[20:21], off offset:16
	global_load_b32 v43, v[22:23], off
	s_wait_loadcnt 0x2
	v_dot4_i32_iu8 v44, v47, v0, 0 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v53, v0, 0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v44, v48, v1, v44 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v56, v1, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v51, v2, v44 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v57, v2, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v49, v3, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v61, v3, v0 neg_lo:[1,1,0]
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v54, v4, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v62, v4, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v55, v5, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v63, v5, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v50, v6, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v59, v6, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v52, v7, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v58, v7, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v0, v0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v0, v43, v0
	v_dual_mul_f32 v1, v43, v1 :: v_dual_fmac_f32 v32, v0, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v1, v41
	v_add_co_u32 v43, vcc_lo, v20, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v44, null, s5, v21, vcc_lo
	v_add_co_u32 v45, vcc_lo, v22, s6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v46, null, s7, v23, vcc_lo
	s_clause 0x1
	global_load_b128 v[0:3], v[43:44], off
	global_load_b128 v[4:7], v[43:44], off offset:16
	global_load_b32 v60, v[45:46], off
	s_wait_loadcnt 0x2
	v_dot4_i32_iu8 v64, v47, v0, 0 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v53, v0, 0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v64, v48, v1, v64 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v56, v1, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v51, v2, v64 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v57, v2, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v49, v3, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v61, v3, v0 neg_lo:[1,1,0]
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v54, v4, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v62, v4, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v55, v5, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v63, v5, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v50, v6, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v59, v6, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v52, v7, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v58, v7, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v0, v0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v1, v60, v1
	v_dual_mul_f32 v0, v60, v0 :: v_dual_fmac_f32 v39, v1, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v0, v42
	v_add_co_u32 v43, vcc_lo, v43, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v44, null, s5, v44, vcc_lo
	v_add_co_u32 v45, vcc_lo, v45, s6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v46, null, s7, v46, vcc_lo
	s_clause 0x1
	global_load_b128 v[0:3], v[43:44], off
	global_load_b128 v[4:7], v[43:44], off offset:16
	global_load_b32 v60, v[45:46], off
	s_wait_loadcnt 0x2
	v_dot4_i32_iu8 v64, v47, v0, 0 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v53, v0, 0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v64, v48, v1, v64 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v56, v1, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v51, v2, v64 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v57, v2, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v49, v3, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v61, v3, v0 neg_lo:[1,1,0]
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v54, v4, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v62, v4, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v55, v5, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v63, v5, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v50, v6, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v59, v6, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v52, v7, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v58, v7, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v0, v0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v1, v60, v1
	v_mul_f32_e32 v0, v60, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v38, v1, v41
	v_fmac_f32_e32 v30, v0, v42
	v_add_co_u32 v43, vcc_lo, v43, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v44, null, s5, v44, vcc_lo
	v_add_co_u32 v45, vcc_lo, v45, s6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v46, null, s7, v46, vcc_lo
	s_clause 0x1
	global_load_b128 v[0:3], v[43:44], off
	global_load_b128 v[4:7], v[43:44], off offset:16
	global_load_b32 v60, v[45:46], off
	s_wait_loadcnt 0x2
	v_dot4_i32_iu8 v64, v47, v0, 0 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v53, v0, 0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v64, v48, v1, v64 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v56, v1, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v51, v2, v64 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v57, v2, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v49, v3, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v61, v3, v0 neg_lo:[1,1,0]
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v54, v4, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v62, v4, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v55, v5, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v63, v5, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v50, v6, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v59, v6, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v52, v7, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v58, v7, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v0, v0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v1, v60, v1
	v_dual_mul_f32 v0, v60, v0 :: v_dual_fmac_f32 v37, v1, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v29, v0, v42
	v_add_co_u32 v43, vcc_lo, v43, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v44, null, s5, v44, vcc_lo
	v_add_co_u32 v45, vcc_lo, v45, s6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v46, null, s7, v46, vcc_lo
	s_clause 0x1
	global_load_b128 v[0:3], v[43:44], off
	global_load_b128 v[4:7], v[43:44], off offset:16
	global_load_b32 v60, v[45:46], off
	s_wait_loadcnt 0x2
	v_dot4_i32_iu8 v64, v47, v0, 0 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v53, v0, 0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v64, v48, v1, v64 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v56, v1, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v51, v2, v64 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v57, v2, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v49, v3, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v61, v3, v0 neg_lo:[1,1,0]
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v54, v4, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v62, v4, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v55, v5, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v63, v5, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v50, v6, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v59, v6, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v52, v7, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v58, v7, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v0, v0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v1, v60, v1
	v_mul_f32_e32 v0, v60, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v36, v1, v41
	v_fmac_f32_e32 v28, v0, v42
	v_add_co_u32 v43, vcc_lo, v43, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v44, null, s5, v44, vcc_lo
	v_add_co_u32 v45, vcc_lo, v45, s6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v46, null, s7, v46, vcc_lo
	s_clause 0x1
	global_load_b128 v[0:3], v[43:44], off
	global_load_b128 v[4:7], v[43:44], off offset:16
	global_load_b32 v60, v[45:46], off
	s_wait_loadcnt 0x2
	v_dot4_i32_iu8 v64, v47, v0, 0 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v53, v0, 0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v64, v48, v1, v64 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v56, v1, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v51, v2, v64 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v57, v2, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v49, v3, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v61, v3, v0 neg_lo:[1,1,0]
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v54, v4, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v62, v4, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v55, v5, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v63, v5, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v50, v6, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v59, v6, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v52, v7, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v58, v7, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v0, v0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v1, v60, v1
	v_dual_mul_f32 v0, v60, v0 :: v_dual_fmac_f32 v35, v1, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v0, v42
	v_add_co_u32 v43, vcc_lo, v43, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v44, null, s5, v44, vcc_lo
	v_add_co_u32 v45, vcc_lo, v45, s6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v46, null, s7, v46, vcc_lo
	s_clause 0x1
	global_load_b128 v[0:3], v[43:44], off
	global_load_b128 v[4:7], v[43:44], off offset:16
	global_load_b32 v60, v[45:46], off
	s_wait_loadcnt 0x2
	v_dot4_i32_iu8 v64, v47, v0, 0 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v53, v0, 0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v64, v48, v1, v64 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v56, v1, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v51, v2, v64 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v57, v2, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v49, v3, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v61, v3, v0 neg_lo:[1,1,0]
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v54, v4, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v62, v4, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v55, v5, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v63, v5, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v50, v6, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v59, v6, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v52, v7, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v58, v7, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v0, v0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v1, v60, v1
	v_mul_f32_e32 v0, v60, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v34, v1, v41
	v_fmac_f32_e32 v26, v0, v42
	v_add_co_u32 v4, vcc_lo, v43, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, s5, v44, vcc_lo
	v_add_co_u32 v43, vcc_lo, v45, s6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v44, null, s7, v46, vcc_lo
	s_clause 0x1
	global_load_b128 v[0:3], v[4:5], off
	global_load_b128 v[4:7], v[4:5], off offset:16
	v_add_co_u32 v24, vcc_lo, v24, 32
	global_load_b32 v43, v[43:44], off
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v25, null, 0, v25, vcc_lo
	v_add_co_u32 v16, vcc_lo, 0x200, v16
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v17, null, 0, v17, vcc_lo
	v_add_co_u32 v18, vcc_lo, 0x200, v18
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v19, null, 0, v19, vcc_lo
	v_add_co_u32 v20, vcc_lo, 0x400, v20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v21, null, 0, v21, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, s14, v24
	v_add_co_u32 v22, s0, 0x80, v22
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v23, null, 0, v23, s0
	s_or_b32 s1, vcc_lo, s1
	s_wait_loadcnt 0x2
	v_dot4_i32_iu8 v44, v47, v0, 0 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v53, v0, 0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v44, v48, v1, v44 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v56, v1, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v51, v2, v44 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v57, v2, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v49, v3, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v61, v3, v0 neg_lo:[1,1,0]
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v54, v4, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v62, v4, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v55, v5, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v63, v5, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v50, v6, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v59, v6, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_i32_iu8 v1, v52, v7, v1 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v0, v58, v7, v0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v0, v0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v1, v43, v1
	v_dual_mul_f32 v0, v43, v0 :: v_dual_fmac_f32 v33, v1, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v0, v42
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB234_18
.LBB234_7:                              ; =>This Inner Loop Header: Depth=1
	v_add_co_u32 v0, vcc_lo, v12, v24
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v13, v25, vcc_lo
	s_mov_b32 s0, exec_lo
	global_load_u8 v0, v[0:1], off
	global_load_b128 v[4:7], v[16:17], off th:TH_LOAD_NT
	s_wait_loadcnt 0x1
	v_cmpx_lt_i16_e32 0xfe, v0
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
	s_wait_alu 0xfffe
	s_or_saveexec_b32 s0, s0
	v_mov_b32_e32 v41, 0x7fc00000
	s_wait_alu 0xfffe
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB234_13
	v_mov_b32_e32 v41, 0x400000
	s_mov_b32 s11, exec_lo
	v_cmpx_ne_u16_e32 0, v0
	v_lshlrev_b32_e32 v41, 23, v0
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s11
.LBB234_13:                             ;   in Loop: Header=BB234_7 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s0
	v_add_co_u32 v0, vcc_lo, v14, v24
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v15, v25, vcc_lo
	s_mov_b32 s0, exec_lo
	global_load_u8 v43, v[0:1], off
	global_load_b128 v[0:3], v[18:19], off th:TH_LOAD_NT
	s_wait_loadcnt 0x1
	v_cmpx_lt_i16_e32 0xfe, v43
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
	s_wait_alu 0xfffe
	s_or_saveexec_b32 s0, s0
	v_mov_b32_e32 v42, 0x7fc00000
	s_wait_alu 0xfffe
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB234_6
	v_mov_b32_e32 v42, 0x400000
	s_mov_b32 s11, exec_lo
	v_cmpx_ne_u16_e32 0, v43
	s_cbranch_execz .LBB234_5
	v_lshlrev_b32_e32 v42, 23, v43
	s_branch .LBB234_5
.LBB234_18:
	s_or_b32 exec_lo, exec_lo, s1
.LBB234_19:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	s_or_b32 exec_lo, exec_lo, s16
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_xor_b32_e32 v1, 16, v0
	v_xor_b32_e32 v3, 8, v0
	v_xor_b32_e32 v4, 4, v0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_i32_e32 vcc_lo, 32, v1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v1, v0, v1, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v3
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v0, v3, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v4
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v4, v0, v4 :: v_dual_lshlrev_b32 v3, 2, v3
	v_lshlrev_b32_e32 v4, 2, v4
	v_lshlrev_b32_e32 v2, 2, v1
	ds_bpermute_b32 v1, v2, v40
	s_wait_dscnt 0x0
	v_add_f32_e32 v1, v40, v1
	ds_bpermute_b32 v5, v3, v1
	s_wait_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	v_xor_b32_e32 v5, 2, v0
	ds_bpermute_b32 v6, v4, v1
	v_cmp_gt_i32_e32 vcc_lo, 32, v5
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v5, v0, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v5, 2, v5
	s_wait_dscnt 0x0
	v_add_f32_e32 v1, v1, v6
	v_xor_b32_e32 v6, 1, v0
	ds_bpermute_b32 v7, v5, v1
	v_cmp_gt_i32_e32 vcc_lo, 32, v6
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v0, v0, v6, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v8
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v7, v1, v7 :: v_dual_lshlrev_b32 v6, 2, v0
	v_lshlrev_b64_e32 v[0:1], 2, v[10:11]
	ds_bpermute_b32 v12, v6, v7
	v_add_co_u32 v0, s0, s12, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v1, null, s13, v1, s0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB234_21
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v12
	global_store_b32 v[0:1], v7, off
.LBB234_21:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	ds_bpermute_b32 v7, v2, v39
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v39, v7
	ds_bpermute_b32 v8, v3, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v4, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v5, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v6, v7
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB234_23
	s_lshl_b64 s[4:5], s[2:3], 2
	s_wait_dscnt 0x0
	v_add_f32_e32 v12, v7, v8
	s_wait_alu 0xfffe
	v_add_co_u32 v7, s0, v0, s4
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v8, null, s5, v1, s0
	global_store_b32 v[7:8], v12, off
.LBB234_23:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	ds_bpermute_b32 v7, v2, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v38, v7
	ds_bpermute_b32 v8, v3, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v4, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v5, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v6, v7
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB234_25
	s_lshl_b64 s[4:5], s[2:3], 3
	s_wait_dscnt 0x0
	v_add_f32_e32 v12, v7, v8
	s_wait_alu 0xfffe
	v_add_co_u32 v7, s0, v0, s4
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v8, null, s5, v1, s0
	global_store_b32 v[7:8], v12, off
.LBB234_25:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	ds_bpermute_b32 v7, v2, v37
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v37, v7
	ds_bpermute_b32 v8, v3, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v4, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v5, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v6, v7
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB234_27
	v_mad_co_u64_u32 v[12:13], null, s2, 12, v[0:1]
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_u64_u32 v[13:14], null, s3, 12, v[13:14]
	global_store_b32 v[12:13], v7, off
.LBB234_27:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	ds_bpermute_b32 v7, v2, v36
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v36, v7
	ds_bpermute_b32 v8, v3, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v4, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v5, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v6, v7
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB234_29
	s_lshl_b64 s[4:5], s[2:3], 4
	s_wait_dscnt 0x0
	v_add_f32_e32 v12, v7, v8
	s_wait_alu 0xfffe
	v_add_co_u32 v7, s0, v0, s4
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v8, null, s5, v1, s0
	global_store_b32 v[7:8], v12, off
.LBB234_29:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	ds_bpermute_b32 v7, v2, v35
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v35, v7
	ds_bpermute_b32 v8, v3, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v4, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v5, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v6, v7
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB234_31
	v_mad_co_u64_u32 v[12:13], null, s2, 20, v[0:1]
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_u64_u32 v[13:14], null, s3, 20, v[13:14]
	global_store_b32 v[12:13], v7, off
.LBB234_31:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	ds_bpermute_b32 v7, v2, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v34, v7
	ds_bpermute_b32 v8, v3, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v4, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v5, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v6, v7
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB234_33
	v_mad_co_u64_u32 v[12:13], null, s2, 24, v[0:1]
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_u64_u32 v[13:14], null, s3, 24, v[13:14]
	global_store_b32 v[12:13], v7, off
.LBB234_33:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	ds_bpermute_b32 v7, v2, v33
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v33, v7
	ds_bpermute_b32 v8, v3, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v4, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v5, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v6, v7
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB234_35
	v_mad_co_u64_u32 v[12:13], null, s2, 28, v[0:1]
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_u64_u32 v[13:14], null, s3, 28, v[13:14]
	global_store_b32 v[12:13], v7, off
.LBB234_35:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	ds_bpermute_b32 v7, v2, v32
	v_or_b32_e32 v10, 1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i64_e64 s0, s[2:3], v[10:11]
	s_and_b32 s0, vcc_lo, s0
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v32, v7
	ds_bpermute_b32 v8, v3, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v4, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v5, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v6, v7
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB234_37
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	global_store_b32 v[0:1], v7, off offset:4
.LBB234_37:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	ds_bpermute_b32 v7, v2, v31
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v31, v7
	ds_bpermute_b32 v8, v3, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v4, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v5, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v6, v7
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB234_39
	s_lshl_b64 s[4:5], s[2:3], 2
	s_wait_dscnt 0x0
	v_add_f32_e32 v10, v7, v8
	s_wait_alu 0xfffe
	v_add_co_u32 v7, vcc_lo, v0, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, s5, v1, vcc_lo
	global_store_b32 v[7:8], v10, off offset:4
.LBB234_39:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	ds_bpermute_b32 v7, v2, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v30, v7
	ds_bpermute_b32 v8, v3, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v4, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v5, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v6, v7
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB234_41
	s_lshl_b64 s[4:5], s[2:3], 3
	s_wait_dscnt 0x0
	v_add_f32_e32 v10, v7, v8
	s_wait_alu 0xfffe
	v_add_co_u32 v7, vcc_lo, v0, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, s5, v1, vcc_lo
	global_store_b32 v[7:8], v10, off offset:4
.LBB234_41:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	ds_bpermute_b32 v7, v2, v29
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v29, v7
	ds_bpermute_b32 v8, v3, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v4, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v5, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v6, v7
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB234_43
	v_mad_co_u64_u32 v[10:11], null, s2, 12, v[0:1]
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_u64_u32 v[11:12], null, s3, 12, v[11:12]
	global_store_b32 v[10:11], v7, off offset:4
.LBB234_43:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	ds_bpermute_b32 v7, v2, v28
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v28, v7
	ds_bpermute_b32 v8, v3, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v4, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v5, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v6, v7
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB234_45
	s_lshl_b64 s[4:5], s[2:3], 4
	s_wait_dscnt 0x0
	v_add_f32_e32 v10, v7, v8
	s_wait_alu 0xfffe
	v_add_co_u32 v7, vcc_lo, v0, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, s5, v1, vcc_lo
	global_store_b32 v[7:8], v10, off offset:4
.LBB234_45:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	ds_bpermute_b32 v7, v2, v27
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v27, v7
	ds_bpermute_b32 v8, v3, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v4, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v5, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v6, v7
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB234_47
	v_mad_co_u64_u32 v[10:11], null, s2, 20, v[0:1]
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_u64_u32 v[11:12], null, s3, 20, v[11:12]
	global_store_b32 v[10:11], v7, off offset:4
.LBB234_47:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	ds_bpermute_b32 v7, v2, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v26, v7
	ds_bpermute_b32 v8, v3, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v4, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v5, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v6, v7
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB234_49
	v_mad_co_u64_u32 v[10:11], null, s2, 24, v[0:1]
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_u64_u32 v[11:12], null, s3, 24, v[11:12]
	global_store_b32 v[10:11], v7, off offset:4
.LBB234_49:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	ds_bpermute_b32 v2, v2, v9
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v9, v2
	ds_bpermute_b32 v3, v3, v2
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v2, v3
	ds_bpermute_b32 v3, v4, v2
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v2, v3
	ds_bpermute_b32 v3, v5, v2
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v2, v3
	ds_bpermute_b32 v3, v6, v2
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB234_51
	v_mad_co_u64_u32 v[0:1], null, s2, 28, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[4:5], null, s3, 28, v[1:2]
	s_wait_dscnt 0x0
	v_dual_add_f32 v2, v2, v3 :: v_dual_mov_b32 v1, v4
	global_store_b32 v[0:1], v2, off offset:4
.LBB234_51:
	s_endpgm
