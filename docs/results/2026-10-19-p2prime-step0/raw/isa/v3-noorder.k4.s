_ZN12_GLOBAL__N_124moe_experts_p1_v3_kernelILi4EEEvPK7ik_bf16PKhS5_S5_S5_PKfPKiS7_PS1_SA_iiiiiiii: ; @_ZN12_GLOBAL__N_124moe_experts_p1_v3_kernelILi4EEEvPK7ik_bf16PKhS5_S5_S5_PKfPKiS7_PS1_SA_iiiiiiii
; %bb.0:
	s_clause 0x1
	s_load_b256 s[16:23], s[0:1], 0x50
	s_load_b128 s[24:27], s[0:1], 0x8
	v_cmp_eq_u32_e64 s2, 0, v0
	s_wait_kmcnt 0x0
	s_lshl_b32 s4, s22, 4
	s_getpc_b64 s[6:7]
	s_sext_i32_i16 s7, s7
	s_add_co_u32 s6, s6, _ZN12_GLOBAL__N_110g_moe_syncE@rel32@lo+8
	s_add_co_ci_u32 s7, s7, _ZN12_GLOBAL__N_110g_moe_syncE@rel32@hi+16
	s_ashr_i32 s5, s4, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_nc_u64 s[28:29], s[6:7], s[4:5]
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB_4
; %bb.1:
	s_mov_b32 s5, exec_lo
	s_mov_b32 s4, exec_lo
	v_mbcnt_lo_u32_b32 v1, s5, 0
                                        ; implicit-def: $vgpr2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v1
	s_cbranch_execz .LBB_3
; %bb.2:
	s_bcnt1_i32_b32 s5, s5
	s_wait_alu 0xfffe
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, s5
	global_atomic_add_u32 v2, v2, v3, s[28:29] offset:8 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
.LBB_3:
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s4, v2
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v1, s4, v1
	ds_store_b32 v2, v1 offset:64
.LBB_4:
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v1, 0 :: v_dual_and_b32 v58, 31, v0
	s_clause 0x3
	s_load_b64 s[34:35], s[0:1], 0x0
	s_load_b256 s[8:15], s[0:1], 0x18
	s_load_b64 s[38:39], s[0:1], 0x48
	s_load_b64 s[42:43], s[0:1], 0x38
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v1, v1 offset:64
	s_load_b64 s[30:31], s[0:1], 0x40
	v_lshrrev_b32_e32 v52, 5, v0
	s_mov_b32 s36, s17
	s_mov_b32 s0, -1
	s_wait_dscnt 0x0
	v_cmp_le_i32_e32 vcc_lo, s20, v1
	v_readfirstlane_b32 s22, v1
	s_cbranch_vccz .LBB_286
; %bb.5:
	s_add_co_i32 s0, s17, 7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 29
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s0, s0, s1
	s_ashr_i32 s0, s0, 3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_abs_i32 s3, s0
	s_cvt_f32_u32 s1, s3
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s1
	v_readfirstlane_b32 s1, v1
	s_mul_f32 s1, s1, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_cvt_u32_f32 s4, s1
	s_sub_co_i32 s1, 0, s3
	s_wait_alu 0xfffe
	s_mul_i32 s5, s1, s4
	s_sub_co_i32 s1, s22, s20
	s_wait_alu 0xfffe
	s_mul_hi_u32 s5, s4, s5
	s_abs_i32 s6, s1
	s_wait_alu 0xfffe
	s_add_co_i32 s4, s4, s5
	s_xor_b32 s5, s1, s0
	s_wait_alu 0xfffe
	s_mul_hi_u32 s4, s6, s4
	s_ashr_i32 s5, s5, 31
	s_wait_alu 0xfffe
	s_mul_i32 s7, s4, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s6, s6, s7
	s_add_co_i32 s7, s4, 1
	s_sub_co_i32 s33, s6, s3
	s_cmp_ge_u32 s6, s3
	s_cselect_b32 s4, s7, s4
	s_cselect_b32 s6, s33, s6
	s_wait_alu 0xfffe
	s_add_co_i32 s7, s4, 1
	s_cmp_ge_u32 s6, s3
	s_mov_b32 s6, s19
	s_cselect_b32 s3, s7, s4
	s_ashr_i32 s7, s19, 31
	s_xor_b32 s3, s3, s5
	s_mov_b32 s33, -1
	s_sub_co_i32 s40, s3, s5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s41, s40, 31
	s_mul_u64 s[46:47], s[40:41], s[6:7]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[44:45], s[46:47], 2
	s_cmp_lt_i32 s19, 1
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[4:5], s[14:15], s[44:45]
	s_cbranch_scc1 .LBB_7
; %bb.6:
	s_load_b32 s33, s[4:5], 0x0
.LBB_7:
	s_mul_i32 s0, s40, s0
	s_ashr_i32 s37, s17, 31
	s_wait_alu 0xfffe
	s_sub_co_i32 s0, s1, s0
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s33, -1
	s_wait_alu 0xfffe
	v_lshl_or_b32 v1, s0, 3, v52
	s_cselect_b32 s1, -1, 0
	s_cmp_lt_i32 s33, s16
	v_mov_b32_e32 v16, 0
	s_cselect_b32 s3, -1, 0
	v_cmp_gt_i32_e64 s0, s17, v1
	s_wait_alu 0xfffe
	s_and_b32 s1, s1, s3
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v2, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e64 v4, 0, 1, s1
	v_cndmask_b32_e64 v32, 0, v1, s0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v33, 0
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB_9
; %bb.8:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[2:3], null, s33, s36, v[32:33]
	v_mad_co_u64_u32 v[5:6], null, s33, s37, v[3:4]
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v3, v5
.LBB_9:
	s_lshr_b32 s1, s18, 31
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_alu 0xfffe
	s_add_co_i32 s1, s18, s1
	s_wait_alu 0xfffe
	s_ashr_i32 s3, s1, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_ashr_i32 s48, s3, 31
	s_cbranch_vccnz .LBB_11
; %bb.10:
	v_mad_co_u64_u32 v[16:17], null, v2, s3, s[8:9]
	v_mul_lo_u32 v5, v2, s48
	v_mul_lo_u32 v6, v3, s3
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v17, v6, v17, v5
.LBB_11:
	s_ashr_i32 s49, s18, 31
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_alu 0xfffe
	s_lshr_b32 s1, s49, 28
	s_wait_alu 0xfffe
	s_add_co_i32 s1, s18, s1
	s_wait_alu 0xfffe
	s_ashr_i32 s68, s1, 4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_ashr_i32 s1, s68, 31
	s_cbranch_vccnz .LBB_13
; %bb.12:
	s_mul_i32 s50, s33, 3
	s_mov_b32 s51, 0
	v_mad_co_u64_u32 v[18:19], null, v2, s68, s[10:11]
	s_lshl_b64 s[50:51], s[50:51], 2
	s_wait_alu 0xfffe
	v_mul_lo_u32 v2, v2, s1
	s_add_nc_u64 s[50:51], s[12:13], s[50:51]
	v_mul_lo_u32 v3, v3, s68
	s_load_b32 s67, s[50:51], 0x8
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v19, v3, v19, v2
	s_branch .LBB_14
.LBB_13:
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	s_mov_b32 s67, 0
.LBB_14:
	s_cmp_gt_i32 s19, 1
	s_mov_b32 s33, -1
	s_cselect_b32 s56, -1, 0
	s_cmp_lt_i32 s19, 2
	s_cbranch_scc1 .LBB_16
; %bb.15:
	s_load_b32 s33, s[4:5], 0x4
.LBB_16:
	s_wait_kmcnt 0x0
	s_wait_alu 0xfffe
	s_cmp_gt_i32 s33, -1
	v_mov_b32_e32 v14, 0
	s_cselect_b32 s50, -1, 0
	s_cmp_lt_i32 s33, s16
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v2, 0
	s_cselect_b32 s51, -1, 0
	v_mov_b32_e32 v3, 0
	s_and_b32 s51, s50, s51
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s50, s51, -1
	s_and_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB_20
; %bb.17:
	v_mad_co_u64_u32 v[2:3], null, s33, s36, v[32:33]
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[3:4], null, s33, s37, v[3:4]
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccz .LBB_21
.LBB_18:
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB_22
.LBB_19:
	s_mov_b32 s50, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v20, s50 :: v_dual_mov_b32 v21, s50
	s_mov_b32 s66, 0
	s_cbranch_execz .LBB_23
	s_branch .LBB_24
.LBB_20:
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccnz .LBB_18
.LBB_21:
	v_mad_co_u64_u32 v[14:15], null, v2, s3, s[8:9]
	v_mul_lo_u32 v4, v2, s48
	v_mul_lo_u32 v5, v3, s3
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v5, v15, v4
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccz .LBB_19
.LBB_22:
                                        ; implicit-def: $vgpr20_vgpr21
	s_mov_b32 s66, 0
.LBB_23:
	s_mul_i32 s50, s33, 3
	s_mov_b32 s51, 0
	v_mad_co_u64_u32 v[20:21], null, v2, s68, s[10:11]
	s_wait_alu 0xfffe
	s_lshl_b64 s[50:51], s[50:51], 2
	v_mul_lo_u32 v2, v2, s1
	s_wait_alu 0xfffe
	s_add_nc_u64 s[50:51], s[12:13], s[50:51]
	v_mul_lo_u32 v3, v3, s68
	s_load_b32 s66, s[50:51], 0x8
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v21, v3, v21, v2
.LBB_24:
	s_cmp_gt_i32 s19, 2
	s_mov_b32 s33, -1
	s_cselect_b32 s55, -1, 0
	s_cmp_lt_i32 s19, 3
	s_cbranch_scc1 .LBB_26
; %bb.25:
	s_load_b32 s33, s[4:5], 0x8
.LBB_26:
	s_wait_kmcnt 0x0
	s_wait_alu 0xfffe
	s_cmp_gt_i32 s33, -1
	v_mov_b32_e32 v12, 0
	s_cselect_b32 s50, -1, 0
	s_cmp_lt_i32 s33, s16
	v_dual_mov_b32 v13, 0 :: v_dual_mov_b32 v2, 0
	s_cselect_b32 s51, -1, 0
	v_mov_b32_e32 v3, 0
	s_wait_alu 0xfffe
	s_and_b32 s51, s50, s51
	s_wait_alu 0xfffe
	s_xor_b32 s50, s51, -1
	s_wait_alu 0xfffe
	s_and_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB_30
; %bb.27:
	v_mad_co_u64_u32 v[2:3], null, s33, s36, v[32:33]
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[3:4], null, s33, s37, v[3:4]
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccz .LBB_31
.LBB_28:
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB_32
.LBB_29:
	s_mov_b32 s50, 0
	s_wait_alu 0xfffe
	v_dual_mov_b32 v22, s50 :: v_dual_mov_b32 v23, s50
	s_mov_b32 s65, 0
	s_cbranch_execz .LBB_33
	s_branch .LBB_34
.LBB_30:
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccnz .LBB_28
.LBB_31:
	v_mad_co_u64_u32 v[12:13], null, v2, s3, s[8:9]
	v_mul_lo_u32 v4, v2, s48
	v_mul_lo_u32 v5, v3, s3
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v13, v5, v13, v4
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccz .LBB_29
.LBB_32:
                                        ; implicit-def: $vgpr22_vgpr23
	s_mov_b32 s65, 0
.LBB_33:
	s_mul_i32 s50, s33, 3
	s_mov_b32 s51, 0
	v_mad_co_u64_u32 v[22:23], null, v2, s68, s[10:11]
	s_wait_alu 0xfffe
	s_lshl_b64 s[50:51], s[50:51], 2
	v_mul_lo_u32 v2, v2, s1
	s_wait_alu 0xfffe
	s_add_nc_u64 s[50:51], s[12:13], s[50:51]
	v_mul_lo_u32 v3, v3, s68
	s_load_b32 s65, s[50:51], 0x8
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v23, v3, v23, v2
.LBB_34:
	s_cmp_gt_i32 s19, 3
	s_mov_b32 s33, -1
	s_cselect_b32 s54, -1, 0
	s_cmp_lt_i32 s19, 4
	s_cbranch_scc1 .LBB_36
; %bb.35:
	s_load_b32 s33, s[4:5], 0xc
.LBB_36:
	s_wait_kmcnt 0x0
	s_wait_alu 0xfffe
	s_cmp_gt_i32 s33, -1
	v_mov_b32_e32 v10, 0
	s_cselect_b32 s50, -1, 0
	s_cmp_lt_i32 s33, s16
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v2, 0
	s_cselect_b32 s51, -1, 0
	v_mov_b32_e32 v3, 0
	s_wait_alu 0xfffe
	s_and_b32 s51, s50, s51
	s_wait_alu 0xfffe
	s_xor_b32 s50, s51, -1
	s_wait_alu 0xfffe
	s_and_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB_40
; %bb.37:
	v_mad_co_u64_u32 v[2:3], null, s33, s36, v[32:33]
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[3:4], null, s33, s37, v[3:4]
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccz .LBB_41
.LBB_38:
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB_42
.LBB_39:
	s_mov_b32 s50, 0
	s_wait_alu 0xfffe
	v_dual_mov_b32 v24, s50 :: v_dual_mov_b32 v25, s50
	s_mov_b32 s64, 0
	s_cbranch_execz .LBB_43
	s_branch .LBB_44
.LBB_40:
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccnz .LBB_38
.LBB_41:
	v_mad_co_u64_u32 v[10:11], null, v2, s3, s[8:9]
	v_mul_lo_u32 v4, v2, s48
	v_mul_lo_u32 v5, v3, s3
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v11, v5, v11, v4
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccz .LBB_39
.LBB_42:
                                        ; implicit-def: $vgpr24_vgpr25
	s_mov_b32 s64, 0
.LBB_43:
	s_mul_i32 s50, s33, 3
	s_mov_b32 s51, 0
	v_mad_co_u64_u32 v[24:25], null, v2, s68, s[10:11]
	s_wait_alu 0xfffe
	s_lshl_b64 s[50:51], s[50:51], 2
	v_mul_lo_u32 v2, v2, s1
	s_wait_alu 0xfffe
	s_add_nc_u64 s[50:51], s[12:13], s[50:51]
	v_mul_lo_u32 v3, v3, s68
	s_load_b32 s64, s[50:51], 0x8
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v25, v3, v25, v2
.LBB_44:
	s_cmp_gt_i32 s19, 4
	s_mov_b32 s33, -1
	s_cselect_b32 s53, -1, 0
	s_cmp_lt_i32 s19, 5
	s_cbranch_scc1 .LBB_46
; %bb.45:
	s_load_b32 s33, s[4:5], 0x10
.LBB_46:
	s_wait_kmcnt 0x0
	s_wait_alu 0xfffe
	s_cmp_gt_i32 s33, -1
	v_mov_b32_e32 v8, 0
	s_cselect_b32 s50, -1, 0
	s_cmp_lt_i32 s33, s16
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v2, 0
	s_cselect_b32 s51, -1, 0
	v_mov_b32_e32 v3, 0
	s_wait_alu 0xfffe
	s_and_b32 s51, s50, s51
	s_wait_alu 0xfffe
	s_xor_b32 s50, s51, -1
	s_wait_alu 0xfffe
	s_and_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB_50
; %bb.47:
	v_mad_co_u64_u32 v[2:3], null, s33, s36, v[32:33]
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[3:4], null, s33, s37, v[3:4]
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccz .LBB_51
.LBB_48:
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB_52
.LBB_49:
	s_mov_b32 s50, 0
	s_wait_alu 0xfffe
	v_dual_mov_b32 v26, s50 :: v_dual_mov_b32 v27, s50
	s_mov_b32 s63, 0
	s_cbranch_execz .LBB_53
	s_branch .LBB_54
.LBB_50:
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccnz .LBB_48
.LBB_51:
	v_mad_co_u64_u32 v[8:9], null, v2, s3, s[8:9]
	v_mul_lo_u32 v4, v2, s48
	v_mul_lo_u32 v5, v3, s3
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v9, v5, v9, v4
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccz .LBB_49
.LBB_52:
                                        ; implicit-def: $vgpr26_vgpr27
	s_mov_b32 s63, 0
.LBB_53:
	s_mul_i32 s50, s33, 3
	s_mov_b32 s51, 0
	v_mad_co_u64_u32 v[26:27], null, v2, s68, s[10:11]
	s_wait_alu 0xfffe
	s_lshl_b64 s[50:51], s[50:51], 2
	v_mul_lo_u32 v2, v2, s1
	s_wait_alu 0xfffe
	s_add_nc_u64 s[50:51], s[12:13], s[50:51]
	v_mul_lo_u32 v3, v3, s68
	s_load_b32 s63, s[50:51], 0x8
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v27, v3, v27, v2
.LBB_54:
	s_cmp_gt_i32 s19, 5
	s_mov_b32 s33, -1
	s_cselect_b32 s52, -1, 0
	s_cmp_lt_i32 s19, 6
	s_cbranch_scc1 .LBB_56
; %bb.55:
	s_load_b32 s33, s[4:5], 0x14
.LBB_56:
	s_wait_kmcnt 0x0
	s_wait_alu 0xfffe
	s_cmp_gt_i32 s33, -1
	v_mov_b32_e32 v6, 0
	s_cselect_b32 s50, -1, 0
	s_cmp_lt_i32 s33, s16
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v2, 0
	s_cselect_b32 s51, -1, 0
	v_mov_b32_e32 v3, 0
	s_wait_alu 0xfffe
	s_and_b32 s51, s50, s51
	s_wait_alu 0xfffe
	s_xor_b32 s50, s51, -1
	s_wait_alu 0xfffe
	s_and_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB_60
; %bb.57:
	v_mad_co_u64_u32 v[2:3], null, s33, s36, v[32:33]
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[3:4], null, s33, s37, v[3:4]
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccz .LBB_61
.LBB_58:
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB_62
.LBB_59:
	s_mov_b32 s50, 0
	s_wait_alu 0xfffe
	v_dual_mov_b32 v28, s50 :: v_dual_mov_b32 v29, s50
	s_mov_b32 s62, 0
	s_cbranch_execz .LBB_63
	s_branch .LBB_64
.LBB_60:
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccnz .LBB_58
.LBB_61:
	v_mad_co_u64_u32 v[6:7], null, v2, s3, s[8:9]
	v_mul_lo_u32 v4, v2, s48
	v_mul_lo_u32 v5, v3, s3
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v7, v5, v7, v4
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccz .LBB_59
.LBB_62:
                                        ; implicit-def: $vgpr28_vgpr29
	s_mov_b32 s62, 0
.LBB_63:
	s_mul_i32 s50, s33, 3
	s_mov_b32 s51, 0
	v_mad_co_u64_u32 v[28:29], null, v2, s68, s[10:11]
	s_wait_alu 0xfffe
	s_lshl_b64 s[50:51], s[50:51], 2
	v_mul_lo_u32 v2, v2, s1
	s_wait_alu 0xfffe
	s_add_nc_u64 s[50:51], s[12:13], s[50:51]
	v_mul_lo_u32 v3, v3, s68
	s_load_b32 s62, s[50:51], 0x8
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v29, v3, v29, v2
.LBB_64:
	s_cmp_gt_i32 s19, 6
	s_mov_b32 s50, -1
	s_cselect_b32 s51, -1, 0
	s_cmp_lt_i32 s19, 7
	s_cbranch_scc1 .LBB_66
; %bb.65:
	s_load_b32 s50, s[4:5], 0x18
.LBB_66:
	s_wait_kmcnt 0x0
	s_wait_alu 0xfffe
	s_cmp_gt_i32 s50, -1
	v_mov_b32_e32 v4, 0
	s_cselect_b32 s33, -1, 0
	s_cmp_lt_i32 s50, s16
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v2, 0
	s_cselect_b32 s57, -1, 0
	v_mov_b32_e32 v3, 0
	s_wait_alu 0xfffe
	s_and_b32 s57, s33, s57
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s33, s57, -1
	s_wait_alu 0xfffe
	s_and_b32 vcc_lo, exec_lo, s33
	s_cbranch_vccnz .LBB_70
; %bb.67:
	v_mad_co_u64_u32 v[2:3], null, s50, s36, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, s50, s37, v[3:4]
	v_mov_b32_e32 v3, v30
	s_and_not1_b32 vcc_lo, exec_lo, s57
	s_cbranch_vccz .LBB_71
.LBB_68:
	s_and_not1_b32 vcc_lo, exec_lo, s33
	s_cbranch_vccnz .LBB_72
.LBB_69:
	s_mov_b32 s33, 0
	s_wait_alu 0xfffe
	v_dual_mov_b32 v30, s33 :: v_dual_mov_b32 v31, s33
	s_mov_b32 s33, 0
	s_cbranch_execz .LBB_73
	s_branch .LBB_74
.LBB_70:
	s_and_not1_b32 vcc_lo, exec_lo, s57
	s_cbranch_vccnz .LBB_68
.LBB_71:
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[4:5], null, v2, s3, s[8:9]
	v_mul_lo_u32 v30, v2, s48
	v_mul_lo_u32 v31, v3, s3
	v_add3_u32 v5, v31, v5, v30
	s_and_not1_b32 vcc_lo, exec_lo, s33
	s_cbranch_vccz .LBB_69
.LBB_72:
                                        ; implicit-def: $vgpr30_vgpr31
	s_mov_b32 s33, 0
.LBB_73:
	s_mul_i32 s58, s50, 3
	s_mov_b32 s59, 0
	v_mad_co_u64_u32 v[30:31], null, v2, s68, s[10:11]
	s_lshl_b64 s[58:59], s[58:59], 2
	v_mul_lo_u32 v2, v2, s1
	s_add_nc_u64 s[58:59], s[12:13], s[58:59]
	v_mul_lo_u32 v3, v3, s68
	s_load_b32 s33, s[58:59], 0x8
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v31, v3, v31, v2
.LBB_74:
	s_cmp_gt_i32 s19, 7
	s_mov_b32 s57, -1
	s_cselect_b32 s50, -1, 0
	s_cmp_lt_i32 s19, 8
	s_cbranch_scc1 .LBB_76
; %bb.75:
	s_load_b32 s57, s[4:5], 0x1c
.LBB_76:
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s57, -1
	v_mov_b32_e32 v2, 0
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s57, s16
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v34, 0
	s_cselect_b32 s5, -1, 0
	v_mov_b32_e32 v35, 0
	s_wait_alu 0xfffe
	s_and_b32 s5, s4, s5
	s_wait_alu 0xfffe
	s_xor_b32 s4, s5, -1
	s_wait_alu 0xfffe
	s_and_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB_80
; %bb.77:
	v_mad_co_u64_u32 v[34:35], null, s57, s36, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_e32 v32, v35
	v_mad_co_u64_u32 v[32:33], null, s57, s37, v[32:33]
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v35, v32
	s_and_not1_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccz .LBB_81
.LBB_78:
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB_82
.LBB_79:
	s_mov_b32 s3, 0
	s_wait_alu 0xfffe
	v_dual_mov_b32 v32, s3 :: v_dual_mov_b32 v33, s3
	s_mov_b32 s9, 0
	s_cbranch_execz .LBB_83
	s_branch .LBB_84
.LBB_80:
	s_and_not1_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccnz .LBB_78
.LBB_81:
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[2:3], null, v34, s3, s[8:9]
	v_mul_lo_u32 v32, v34, s48
	v_mul_lo_u32 v33, v35, s3
	v_add3_u32 v3, v33, v3, v32
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccz .LBB_79
.LBB_82:
                                        ; implicit-def: $vgpr32_vgpr33
	s_mov_b32 s9, 0
.LBB_83:
	s_mul_i32 s4, s57, 3
	s_mov_b32 s5, 0
	v_mad_co_u64_u32 v[32:33], null, v34, s68, s[10:11]
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 2
	v_mul_lo_u32 v34, v34, s1
	s_wait_alu 0xfffe
	s_add_nc_u64 s[4:5], s[12:13], s[4:5]
	v_mul_lo_u32 v35, v35, s68
	s_load_b32 s9, s[4:5], 0x8
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v33, v35, v33, v34
.LBB_84:
	v_cmp_ne_u64_e32 vcc_lo, 0, v[16:17]
	v_cmp_gt_i32_e64 s1, s68, v58
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v69, 0
	s_and_b32 s4, s1, vcc_lo
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB_86
; %bb.85:
	v_lshlrev_b32_e32 v34, 3, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v34, vcc_lo, v16, v34
	v_add_co_ci_u32_e64 v35, null, 0, v17, vcc_lo
	v_add_co_u32 v36, vcc_lo, v18, v58
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v37, null, 0, v19, vcc_lo
	global_load_b64 v[50:51], v[34:35], off th:TH_LOAD_NT
	flat_load_u8 v69, v[36:37]
.LBB_86:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	v_cmp_ne_u64_e32 vcc_lo, 0, v[14:15]
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v68, 0
	s_and_b32 s4, s1, vcc_lo
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB_88
; %bb.87:
	v_lshlrev_b32_e32 v34, 3, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v34, vcc_lo, v14, v34
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v35, null, 0, v15, vcc_lo
	v_add_co_u32 v36, vcc_lo, v20, v58
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v37, null, 0, v21, vcc_lo
	global_load_b64 v[48:49], v[34:35], off th:TH_LOAD_NT
	flat_load_u8 v68, v[36:37]
.LBB_88:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	v_cmp_ne_u64_e32 vcc_lo, 0, v[12:13]
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v67, 0
	s_and_b32 s4, s1, vcc_lo
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB_90
; %bb.89:
	v_lshlrev_b32_e32 v34, 3, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v34, vcc_lo, v12, v34
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v35, null, 0, v13, vcc_lo
	v_add_co_u32 v36, vcc_lo, v22, v58
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v37, null, 0, v23, vcc_lo
	global_load_b64 v[46:47], v[34:35], off th:TH_LOAD_NT
	flat_load_u8 v67, v[36:37]
.LBB_90:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	v_cmp_ne_u64_e32 vcc_lo, 0, v[10:11]
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v66, 0
	s_and_b32 s4, s1, vcc_lo
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB_92
; %bb.91:
	v_lshlrev_b32_e32 v34, 3, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v34, vcc_lo, v10, v34
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v35, null, 0, v11, vcc_lo
	v_add_co_u32 v36, vcc_lo, v24, v58
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v37, null, 0, v25, vcc_lo
	global_load_b64 v[42:43], v[34:35], off th:TH_LOAD_NT
	flat_load_u8 v66, v[36:37]
.LBB_92:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	v_cmp_ne_u64_e32 vcc_lo, 0, v[8:9]
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v65, 0
	s_and_b32 s4, s1, vcc_lo
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB_94
; %bb.93:
	v_lshlrev_b32_e32 v34, 3, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v34, vcc_lo, v8, v34
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v35, null, 0, v9, vcc_lo
	v_add_co_u32 v36, vcc_lo, v26, v58
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v37, null, 0, v27, vcc_lo
	global_load_b64 v[40:41], v[34:35], off th:TH_LOAD_NT
	flat_load_u8 v65, v[36:37]
.LBB_94:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	v_cmp_ne_u64_e32 vcc_lo, 0, v[6:7]
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v64, 0
	s_and_b32 s4, s1, vcc_lo
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB_96
; %bb.95:
	v_lshlrev_b32_e32 v34, 3, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v34, vcc_lo, v6, v34
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v35, null, 0, v7, vcc_lo
	v_add_co_u32 v36, vcc_lo, v28, v58
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v37, null, 0, v29, vcc_lo
	global_load_b64 v[38:39], v[34:35], off th:TH_LOAD_NT
	flat_load_u8 v64, v[36:37]
.LBB_96:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	v_cmp_ne_u64_e32 vcc_lo, 0, v[4:5]
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v63, 0
	s_and_b32 s4, s1, vcc_lo
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB_98
; %bb.97:
	v_lshlrev_b32_e32 v34, 3, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v36, vcc_lo, v4, v34
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v37, null, 0, v5, vcc_lo
	v_add_co_u32 v44, vcc_lo, v30, v58
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v45, null, 0, v31, vcc_lo
	global_load_b64 v[36:37], v[36:37], off th:TH_LOAD_NT
	flat_load_u8 v63, v[44:45]
.LBB_98:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	v_cmp_ne_u64_e32 vcc_lo, 0, v[2:3]
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v62, 0
	s_and_b32 s4, s1, vcc_lo
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB_100
; %bb.99:
	v_lshlrev_b32_e32 v34, 3, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v34, vcc_lo, v2, v34
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v35, null, 0, v3, vcc_lo
	v_add_co_u32 v44, vcc_lo, v32, v58
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v45, null, 0, v33, vcc_lo
	global_load_b64 v[34:35], v[34:35], off th:TH_LOAD_NT
	flat_load_u8 v62, v[44:45]
.LBB_100:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_u32_e32 16, v0
	s_cbranch_execz .LBB_102
; %bb.101:
	v_and_b32_e32 v44, 7, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v45, v44, 22, 0.5
	v_cmp_ne_u32_e32 vcc_lo, 1, v44
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v45, 0.5, v45, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0, v44
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_cndmask_b32 v44, 0, v45 :: v_dual_lshlrev_b32 v53, 28, v0
	v_lshlrev_b32_e32 v45, 2, v0
	v_and_or_b32 v44, 0x80000000, v53, v44
	ds_store_b32 v45, v44
.LBB_102:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_bitcmp1_b32 s23, 17
	s_mov_b32 s48, s18
	s_cselect_b32 s3, -1, 0
	s_wait_alu 0xfffe
	s_and_b32 s3, s3, s0
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s5, s3
	s_cbranch_execz .LBB_136
; %bb.103:
	v_cmp_ne_u64_e64 s3, 0, v[16:17]
	v_cmp_gt_i32_e64 s4, s68, v58
	v_mov_b32_e32 v53, 0
	v_cmp_le_i32_e32 vcc_lo, s68, v58
	s_and_b32 s3, s3, s4
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s8, s3
	s_cbranch_execz .LBB_107
; %bb.104:
	v_dual_mov_b32 v53, 0 :: v_dual_and_b32 v44, 31, v0
	s_mul_u64 s[6:7], s[6:7], s[48:49]
	v_mov_b32_e32 v54, v58
	s_mul_u64 s[6:7], s[6:7], s[40:41]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_lshlrev_b32_e32 v44, 5, v44
	s_lshl_b64 s[6:7], s[6:7], 1
	s_add_nc_u64 s[6:7], s[30:31], s[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_add_co_u32 v44, s3, s6, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v45, null, s7, 0, s3
	s_mov_b32 s6, 0
.LBB_105:                              ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[70:73], v[44:45], off
	global_load_b128 v[74:77], v[44:45], off offset:16
	v_add_nc_u32_e32 v54, 32, v54
	v_add_co_u32 v44, s3, 0x400, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v45, null, 0, v45, s3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s4, s68, v54
	s_wait_alu 0xfffe
	s_or_b32 s6, s4, s6
	s_wait_loadcnt 0x1
	v_xor3_b32 v53, v70, v53, v71
	v_xor3_b32 v53, v53, v72, v73
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor3_b32 v53, v53, v74, v75
	v_xor3_b32 v53, v53, v76, v77
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB_105
; %bb.106:
	s_or_b32 exec_lo, exec_lo, s6
.LBB_107:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s8
	v_cmp_ne_u64_e64 s3, 0, v[14:15]
	s_xor_b32 s4, vcc_lo, -1
	s_wait_alu 0xfffe
	s_and_b32 s3, s3, s4
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s6, s3
	s_cbranch_execz .LBB_111
; %bb.108:
	v_and_b32_e32 v44, 31, v0
	s_lshl_b64 s[10:11], s[46:47], 1
	v_mov_b32_e32 v54, v58
	s_wait_alu 0xfffe
	s_add_nc_u64 s[10:11], s[10:11], 2
	s_mov_b32 s7, 0
	v_lshlrev_b32_e32 v44, 5, v44
	s_wait_alu 0xfffe
	s_mul_u64 s[10:11], s[10:11], s[48:49]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[10:11], s[30:31], s[10:11]
	s_wait_alu 0xfffe
	v_add_co_u32 v44, s3, s10, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v45, null, s11, 0, s3
.LBB_109:                              ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[70:73], v[44:45], off
	global_load_b128 v[74:77], v[44:45], off offset:16
	v_add_nc_u32_e32 v54, 32, v54
	v_add_co_u32 v44, vcc_lo, 0x400, v44
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v45, null, 0, v45, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s3, s68, v54
	s_or_b32 s7, s3, s7
	s_wait_loadcnt 0x1
	v_xor3_b32 v53, v70, v53, v71
	v_xor3_b32 v53, v53, v72, v73
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor3_b32 v53, v53, v74, v75
	v_xor3_b32 v53, v53, v76, v77
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s7
	s_cbranch_execnz .LBB_109
; %bb.110:
	s_or_b32 exec_lo, exec_lo, s7
.LBB_111:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s6
	v_cmp_ne_u64_e32 vcc_lo, 0, v[12:13]
	s_and_b32 s3, vcc_lo, s4
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s6, s3
	s_cbranch_execz .LBB_115
; %bb.112:
	v_and_b32_e32 v44, 31, v0
	s_lshl_b64 s[10:11], s[46:47], 1
	v_mov_b32_e32 v54, v58
	s_wait_alu 0xfffe
	s_add_nc_u64 s[10:11], s[10:11], 4
	s_mov_b32 s7, 0
	v_lshlrev_b32_e32 v44, 5, v44
	s_wait_alu 0xfffe
	s_mul_u64 s[10:11], s[10:11], s[48:49]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[10:11], s[30:31], s[10:11]
	s_wait_alu 0xfffe
	v_add_co_u32 v44, s3, s10, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v45, null, s11, 0, s3
.LBB_113:                              ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[70:73], v[44:45], off
	global_load_b128 v[74:77], v[44:45], off offset:16
	v_add_nc_u32_e32 v54, 32, v54
	v_add_co_u32 v44, vcc_lo, 0x400, v44
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v45, null, 0, v45, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s3, s68, v54
	s_or_b32 s7, s3, s7
	s_wait_loadcnt 0x1
	v_xor3_b32 v53, v70, v53, v71
	v_xor3_b32 v53, v53, v72, v73
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor3_b32 v53, v53, v74, v75
	v_xor3_b32 v53, v53, v76, v77
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s7
	s_cbranch_execnz .LBB_113
; %bb.114:
	s_or_b32 exec_lo, exec_lo, s7
.LBB_115:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s6
	v_cmp_ne_u64_e32 vcc_lo, 0, v[10:11]
	s_and_b32 s3, vcc_lo, s4
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s6, s3
	s_cbranch_execz .LBB_119
; %bb.116:
	v_and_b32_e32 v44, 31, v0
	s_lshl_b64 s[10:11], s[46:47], 1
	v_mov_b32_e32 v54, v58
	s_wait_alu 0xfffe
	s_add_nc_u64 s[10:11], s[10:11], 6
	s_mov_b32 s7, 0
	v_lshlrev_b32_e32 v44, 5, v44
	s_wait_alu 0xfffe
	s_mul_u64 s[10:11], s[10:11], s[48:49]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[10:11], s[30:31], s[10:11]
	s_wait_alu 0xfffe
	v_add_co_u32 v44, s3, s10, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v45, null, s11, 0, s3
.LBB_117:                              ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[70:73], v[44:45], off
	global_load_b128 v[74:77], v[44:45], off offset:16
	v_add_nc_u32_e32 v54, 32, v54
	v_add_co_u32 v44, vcc_lo, 0x400, v44
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v45, null, 0, v45, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s3, s68, v54
	s_or_b32 s7, s3, s7
	s_wait_loadcnt 0x1
	v_xor3_b32 v53, v70, v53, v71
	v_xor3_b32 v53, v53, v72, v73
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor3_b32 v53, v53, v74, v75
	v_xor3_b32 v53, v53, v76, v77
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s7
	s_cbranch_execnz .LBB_117
; %bb.118:
	s_or_b32 exec_lo, exec_lo, s7
.LBB_119:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s6
	v_cmp_ne_u64_e32 vcc_lo, 0, v[8:9]
	s_and_b32 s3, vcc_lo, s4
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s6, s3
	s_cbranch_execz .LBB_123
; %bb.120:
	v_and_b32_e32 v44, 31, v0
	s_lshl_b64 s[10:11], s[46:47], 1
	v_mov_b32_e32 v54, v58
	s_wait_alu 0xfffe
	s_add_nc_u64 s[10:11], s[10:11], 8
	s_mov_b32 s7, 0
	v_lshlrev_b32_e32 v44, 5, v44
	s_wait_alu 0xfffe
	s_mul_u64 s[10:11], s[10:11], s[48:49]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[10:11], s[30:31], s[10:11]
	s_wait_alu 0xfffe
	v_add_co_u32 v44, s3, s10, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v45, null, s11, 0, s3
.LBB_121:                              ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[70:73], v[44:45], off
	global_load_b128 v[74:77], v[44:45], off offset:16
	v_add_nc_u32_e32 v54, 32, v54
	v_add_co_u32 v44, vcc_lo, 0x400, v44
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v45, null, 0, v45, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s3, s68, v54
	s_or_b32 s7, s3, s7
	s_wait_loadcnt 0x1
	v_xor3_b32 v53, v70, v53, v71
	v_xor3_b32 v53, v53, v72, v73
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor3_b32 v53, v53, v74, v75
	v_xor3_b32 v53, v53, v76, v77
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s7
	s_cbranch_execnz .LBB_121
; %bb.122:
	s_or_b32 exec_lo, exec_lo, s7
.LBB_123:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s6
	v_cmp_ne_u64_e32 vcc_lo, 0, v[6:7]
	s_and_b32 s3, vcc_lo, s4
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s6, s3
	s_cbranch_execz .LBB_127
; %bb.124:
	v_and_b32_e32 v44, 31, v0
	s_lshl_b64 s[10:11], s[46:47], 1
	v_mov_b32_e32 v54, v58
	s_wait_alu 0xfffe
	s_add_nc_u64 s[10:11], s[10:11], 10
	s_mov_b32 s7, 0
	v_lshlrev_b32_e32 v44, 5, v44
	s_wait_alu 0xfffe
	s_mul_u64 s[10:11], s[10:11], s[48:49]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[10:11], s[30:31], s[10:11]
	s_wait_alu 0xfffe
	v_add_co_u32 v44, s3, s10, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v45, null, s11, 0, s3
.LBB_125:                              ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[70:73], v[44:45], off
	global_load_b128 v[74:77], v[44:45], off offset:16
	v_add_nc_u32_e32 v54, 32, v54
	v_add_co_u32 v44, vcc_lo, 0x400, v44
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v45, null, 0, v45, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s3, s68, v54
	s_or_b32 s7, s3, s7
	s_wait_loadcnt 0x1
	v_xor3_b32 v53, v70, v53, v71
	v_xor3_b32 v53, v53, v72, v73
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor3_b32 v53, v53, v74, v75
	v_xor3_b32 v53, v53, v76, v77
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s7
	s_cbranch_execnz .LBB_125
; %bb.126:
	s_or_b32 exec_lo, exec_lo, s7
.LBB_127:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s6
	v_cmp_ne_u64_e32 vcc_lo, 0, v[4:5]
	s_and_b32 s3, vcc_lo, s4
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s6, s3
	s_cbranch_execz .LBB_131
; %bb.128:
	v_and_b32_e32 v44, 31, v0
	s_lshl_b64 s[10:11], s[46:47], 1
	v_mov_b32_e32 v54, v58
	s_wait_alu 0xfffe
	s_add_nc_u64 s[10:11], s[10:11], 12
	s_mov_b32 s7, 0
	v_lshlrev_b32_e32 v44, 5, v44
	s_wait_alu 0xfffe
	s_mul_u64 s[10:11], s[10:11], s[48:49]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[10:11], s[30:31], s[10:11]
	s_wait_alu 0xfffe
	v_add_co_u32 v44, s3, s10, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v45, null, s11, 0, s3
.LBB_129:                              ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[70:73], v[44:45], off
	global_load_b128 v[74:77], v[44:45], off offset:16
	v_add_nc_u32_e32 v54, 32, v54
	v_add_co_u32 v44, vcc_lo, 0x400, v44
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v45, null, 0, v45, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s3, s68, v54
	s_or_b32 s7, s3, s7
	s_wait_loadcnt 0x1
	v_xor3_b32 v53, v70, v53, v71
	v_xor3_b32 v53, v53, v72, v73
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor3_b32 v53, v53, v74, v75
	v_xor3_b32 v53, v53, v76, v77
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s7
	s_cbranch_execnz .LBB_129
; %bb.130:
	s_or_b32 exec_lo, exec_lo, s7
.LBB_131:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s6
	v_cmp_ne_u64_e32 vcc_lo, 0, v[2:3]
	s_and_b32 s3, vcc_lo, s4
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB_135
; %bb.132:
	v_and_b32_e32 v44, 31, v0
	s_lshl_b64 s[6:7], s[46:47], 1
	v_mov_b32_e32 v54, v58
	s_wait_alu 0xfffe
	s_add_nc_u64 s[6:7], s[6:7], 14
	s_wait_alu 0xfffe
	s_mul_u64 s[6:7], s[6:7], s[48:49]
	v_lshlrev_b32_e32 v44, 5, v44
	s_wait_alu 0xfffe
	s_add_nc_u64 s[6:7], s[30:31], s[6:7]
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v44, s3, s6, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v45, null, s7, 0, s3
	s_mov_b32 s6, 0
.LBB_133:                              ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[70:73], v[44:45], off
	global_load_b128 v[74:77], v[44:45], off offset:16
	v_add_nc_u32_e32 v54, 32, v54
	v_add_co_u32 v44, vcc_lo, 0x400, v44
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v45, null, 0, v45, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s3, s68, v54
	s_wait_alu 0xfffe
	s_or_b32 s6, s3, s6
	s_wait_loadcnt 0x1
	v_xor3_b32 v53, v70, v53, v71
	v_xor3_b32 v53, v53, v72, v73
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor3_b32 v53, v53, v74, v75
	v_xor3_b32 v53, v53, v76, v77
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB_133
; %bb.134:
	s_or_b32 exec_lo, exec_lo, s6
.LBB_135:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	;;#ASMSTART
	;;#ASMEND
.LBB_136:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s5
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB_139
; %bb.137:
	v_mov_b32_e32 v44, 0
	global_load_b32 v45, v44, s[28:29] scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	v_cmp_le_u32_e32 vcc_lo, s20, v45
	s_cbranch_vccnz .LBB_139
.LBB_138:                              ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_load_b32 v45, v44, s[28:29] scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	v_cmp_gt_u32_e32 vcc_lo, s20, v45
	s_cbranch_vccnz .LBB_138
.LBB_139:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s69, s0
	s_cbranch_execz .LBB_280
; %bb.140:
	s_mul_u64 s[4:5], s[46:47], s[48:49]
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v60, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v61, 0
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 1
	s_wait_alu 0xfffe
	s_add_nc_u64 s[10:11], s[30:31], s[4:5]
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB_158
; %bb.141:
	v_dual_mov_b32 v53, 0 :: v_dual_lshlrev_b32 v44, 5, v58
	v_mov_b32_e32 v54, 0
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_u32 v44, s1, s10, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v45, null, s11, 0, s1
	s_mov_b32 s1, exec_lo
	v_cmpx_ne_u64_e32 0, v[16:17]
	s_cbranch_execz .LBB_143
; %bb.142:
	s_clause 0x1
	global_load_b128 v[54:57], v[44:45], off
	global_load_b128 v[70:73], v[44:45], off offset:16
	v_and_b32_e32 v59, 15, v50
	v_lshrrev_b32_e32 v60, 2, v50
	v_lshrrev_b32_e32 v61, 6, v50
	v_lshrrev_b32_e32 v76, 18, v50
	v_lshrrev_b32_e32 v74, 10, v50
	v_lshlrev_b32_e32 v59, 2, v59
	v_lshrrev_b32_e32 v75, 14, v50
	v_and_b32_e32 v61, 60, v61
	v_lshrrev_b32_e32 v77, 22, v50
	v_lshrrev_b32_e32 v50, 26, v50
	ds_load_b32 v59, v59
	v_and_b32_e32 v75, 60, v75
	v_lshrrev_b32_e32 v80, 10, v51
	v_and_b32_e32 v77, 60, v77
	v_lshrrev_b32_e32 v81, 22, v51
	ds_load_b32 v61, v61
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v79, 16, v55
	v_and_b32_e32 v60, 60, v60
	v_lshlrev_b32_e32 v78, 16, v54
	v_and_b32_e32 v54, 0xffff0000, v54
	v_lshlrev_b32_e32 v82, 16, v56
	v_and_b32_e32 v56, 0xffff0000, v56
	ds_load_b32 v60, v60
	s_wait_dscnt 0x2
	v_fma_f32 v59, v59, v78, 0
	v_and_b32_e32 v76, 60, v76
	v_and_b32_e32 v74, 60, v74
	v_and_b32_e32 v55, 0xffff0000, v55
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v59, v60, v54
	v_lshrrev_b32_e32 v60, 6, v51
	v_lshrrev_b32_e32 v54, 2, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v59, v61, v79 :: v_dual_and_b32 v50, 60, v50
	ds_load_b32 v74, v74
	ds_load_b32 v75, v75
	ds_load_b32 v76, v76
	ds_load_b32 v77, v77
	ds_load_b32 v50, v50
	v_and_b32_e32 v60, 60, v60
	v_lshrrev_b32_e32 v61, 14, v51
	v_lshrrev_b32_e32 v79, 18, v51
	ds_load_b32 v60, v60
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v59, v74, v55
	v_and_b32_e32 v78, 15, v51
	v_lshrrev_b32_e32 v51, 26, v51
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v59, v75, v82 :: v_dual_and_b32 v74, 60, v80
	v_lshlrev_b32_e32 v75, 16, v57
	v_lshlrev_b32_e32 v55, 2, v78
	v_and_b32_e32 v57, 0xffff0000, v57
	v_and_b32_e32 v51, 60, v51
	s_wait_loadcnt_dscnt 0x3
	v_dual_fmac_f32 v59, v76, v56 :: v_dual_lshlrev_b32 v76, 16, v70
	ds_load_b32 v55, v55
	v_and_b32_e32 v56, 60, v61
	v_and_b32_e32 v61, 60, v79
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v59, v77, v75
	v_and_b32_e32 v75, 60, v81
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v59, v50, v57 :: v_dual_and_b32 v54, 60, v54
	ds_load_b32 v54, v54
	ds_load_b32 v50, v74
	ds_load_b32 v56, v56
	ds_load_b32 v57, v61
	ds_load_b32 v61, v75
	ds_load_b32 v51, v51
	s_wait_dscnt 0x6
	v_dual_fmac_f32 v59, v55, v76 :: v_dual_and_b32 v70, 0xffff0000, v70
	v_lshlrev_b32_e32 v55, 16, v71
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v59, v54, v70 :: v_dual_and_b32 v54, 0xffff0000, v71
	v_fmac_f32_e32 v59, v60, v55
	v_and_b32_e32 v55, 7, v69
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v59, v50, v54 :: v_dual_lshlrev_b32 v60, 16, v72
	v_and_b32_e32 v50, 0x7f, v69
	v_cvt_f32_ubyte0_e32 v54, v55
	v_and_b32_e32 v55, 0xffff0000, v72
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v59, v56, v60
	v_lshl_add_u32 v56, v50, 20, 0x3c000000
	v_mul_f32_e32 v54, 0x3b000000, v54
	v_cmp_lt_u32_e32 vcc_lo, 7, v50
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v59, v57, v55 :: v_dual_lshlrev_b32 v60, 16, v73
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v54, v54, v56 :: v_dual_lshlrev_b32 v55, 24, v69
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v50
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v59, v61, v60 :: v_dual_and_b32 v56, 0xffff0000, v73
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v50, 0x7fc00000, v54 :: v_dual_fmac_f32 v59, v51, v56
	v_and_or_b32 v50, 0x80000000, v55, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v54, v59, v50, 0
.LBB_143:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s1, exec_lo
	v_cmpx_ne_u64_e32 0, v[14:15]
	s_cbranch_execz .LBB_145
; %bb.144:
	s_and_b32 s3, s56, exec_lo
	s_cselect_b32 s5, s49, 0
	s_cselect_b32 s4, s48, 0
	v_lshrrev_b32_e32 v53, 6, v48
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 1
	v_lshrrev_b32_e32 v55, 10, v48
	s_wait_alu 0xfffe
	v_add_co_u32 v50, vcc_lo, v44, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v51, null, s5, v45, vcc_lo
	v_lshrrev_b32_e32 v56, 14, v48
	v_lshrrev_b32_e32 v57, 18, v48
	s_clause 0x1
	global_load_b128 v[69:72], v[50:51], off
	global_load_b128 v[73:76], v[50:51], off offset:16
	v_and_b32_e32 v50, 15, v48
	v_lshrrev_b32_e32 v51, 2, v48
	v_lshrrev_b32_e32 v59, 22, v48
	v_lshrrev_b32_e32 v48, 26, v48
	v_and_b32_e32 v56, 60, v56
	v_lshlrev_b32_e32 v50, 2, v50
	v_lshrrev_b32_e32 v77, 10, v49
	v_and_b32_e32 v59, 60, v59
	v_and_b32_e32 v48, 60, v48
	v_lshrrev_b32_e32 v78, 22, v49
	ds_load_b32 v50, v50
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v60, 16, v69
	v_and_b32_e32 v61, 0xffff0000, v69
	v_lshlrev_b32_e32 v69, 16, v70
	v_and_b32_e32 v51, 60, v51
	v_and_b32_e32 v70, 0xffff0000, v70
	v_and_b32_e32 v53, 60, v53
	v_lshlrev_b32_e32 v79, 16, v71
	v_and_b32_e32 v57, 60, v57
	ds_load_b32 v51, v51
	ds_load_b32 v53, v53
	s_wait_dscnt 0x2
	v_fma_f32 v50, v50, v60, 0
	v_and_b32_e32 v55, 60, v55
	v_and_b32_e32 v60, 15, v49
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v50, v51, v61
	ds_load_b32 v55, v55
	ds_load_b32 v56, v56
	ds_load_b32 v57, v57
	ds_load_b32 v59, v59
	ds_load_b32 v48, v48
	v_lshrrev_b32_e32 v51, 2, v49
	v_lshrrev_b32_e32 v61, 6, v49
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v50, v53, v69
	v_lshrrev_b32_e32 v69, 18, v49
	v_and_b32_e32 v51, 60, v51
	v_lshrrev_b32_e32 v53, 14, v49
	v_lshrrev_b32_e32 v49, 26, v49
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v50, v55, v70 :: v_dual_and_b32 v49, 60, v49
	v_lshlrev_b32_e32 v55, 2, v60
	v_and_b32_e32 v60, 0xffff0000, v71
	v_lshlrev_b32_e32 v70, 16, v72
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v50, v56, v79
	v_and_b32_e32 v56, 60, v61
	ds_load_b32 v55, v55
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v50, v57, v60
	ds_load_b32 v56, v56
	ds_load_b32 v51, v51
	v_and_b32_e32 v60, 0xffff0000, v72
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v50, v59, v70 :: v_dual_and_b32 v57, 60, v69
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v69, 16, v73
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v50, v48, v60 :: v_dual_and_b32 v59, 60, v78
	v_and_b32_e32 v60, 0xffff0000, v73
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v50, v55, v69
	v_and_b32_e32 v61, 60, v77
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v50, v51, v60 :: v_dual_lshlrev_b32 v55, 16, v74
	v_and_b32_e32 v53, 60, v53
	v_dual_fmac_f32 v50, v56, v55 :: v_dual_and_b32 v51, 0xffff0000, v74
	ds_load_b32 v48, v61
	ds_load_b32 v53, v53
	ds_load_b32 v57, v57
	ds_load_b32 v59, v59
	ds_load_b32 v49, v49
	v_and_b32_e32 v55, 7, v68
	v_lshlrev_b32_e32 v56, 16, v75
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v50, v48, v51
	v_and_b32_e32 v48, 0x7f, v68
	v_cvt_f32_ubyte0_e32 v51, v55
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v50, v53, v56 :: v_dual_and_b32 v55, 0xffff0000, v75
	v_lshl_add_u32 v53, v48, 20, 0x3c000000
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v51, 0x3b000000, v51
	v_cmp_lt_u32_e32 vcc_lo, 7, v48
	v_lshlrev_b32_e32 v56, 16, v76
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v50, v57, v55 :: v_dual_and_b32 v55, 0xffff0000, v76
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v51, v51, v53, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v48
	v_lshlrev_b32_e32 v53, 24, v68
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v50, v59, v56
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v48, 0x7fc00000, v51, vcc_lo
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v50, v49, v55
	v_and_or_b32 v48, 0x80000000, v53, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v53, v50, v48, 0
.LBB_145:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v56, 0
	s_mov_b32 s1, exec_lo
	v_cmpx_ne_u64_e32 0, v[12:13]
	s_cbranch_execz .LBB_147
; %bb.146:
	s_and_b32 s3, s55, exec_lo
	s_mov_b32 s5, 0
	s_cselect_b32 s4, 2, 0
	v_lshrrev_b32_e32 v59, 6, v46
	s_wait_alu 0xfffe
	s_mul_u64 s[4:5], s[4:5], s[48:49]
	v_lshrrev_b32_e32 v60, 10, v46
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 1
	v_lshrrev_b32_e32 v61, 14, v46
	s_wait_alu 0xfffe
	v_add_co_u32 v56, vcc_lo, v44, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v57, null, s5, v45, vcc_lo
	v_lshrrev_b32_e32 v72, 18, v46
	v_lshrrev_b32_e32 v73, 22, v46
	s_clause 0x1
	global_load_b128 v[48:51], v[56:57], off
	global_load_b128 v[68:71], v[56:57], off offset:16
	v_and_b32_e32 v56, 15, v46
	v_lshrrev_b32_e32 v57, 2, v46
	v_lshrrev_b32_e32 v46, 26, v46
	v_and_b32_e32 v60, 60, v60
	v_and_b32_e32 v72, 60, v72
	v_lshlrev_b32_e32 v56, 2, v56
	v_lshrrev_b32_e32 v76, 10, v47
	v_and_b32_e32 v46, 60, v46
	v_lshrrev_b32_e32 v77, 22, v47
	ds_load_b32 v56, v56
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v75, 16, v49
	v_and_b32_e32 v57, 60, v57
	v_lshlrev_b32_e32 v74, 16, v48
	v_and_b32_e32 v48, 0xffff0000, v48
	v_lshlrev_b32_e32 v78, 16, v50
	v_and_b32_e32 v50, 0xffff0000, v50
	ds_load_b32 v57, v57
	v_and_b32_e32 v59, 60, v59
	v_and_b32_e32 v61, 60, v61
	v_and_b32_e32 v73, 60, v73
	v_and_b32_e32 v49, 0xffff0000, v49
	ds_load_b32 v59, v59
	ds_load_b32 v60, v60
	ds_load_b32 v61, v61
	ds_load_b32 v72, v72
	ds_load_b32 v73, v73
	ds_load_b32 v46, v46
	s_wait_dscnt 0x7
	v_fma_f32 v56, v56, v74, 0
	v_and_b32_e32 v74, 15, v47
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v56, v57, v48
	v_lshrrev_b32_e32 v48, 2, v47
	v_lshrrev_b32_e32 v57, 6, v47
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v56, v59, v75
	v_lshrrev_b32_e32 v59, 14, v47
	v_lshrrev_b32_e32 v75, 18, v47
	v_and_b32_e32 v48, 60, v48
	v_lshrrev_b32_e32 v47, 26, v47
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v56, v60, v49 :: v_dual_lshlrev_b32 v49, 2, v74
	v_and_b32_e32 v60, 60, v76
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v56, v61, v78 :: v_dual_and_b32 v47, 60, v47
	v_lshlrev_b32_e32 v61, 16, v51
	v_and_b32_e32 v51, 0xffff0000, v51
	ds_load_b32 v49, v49
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v56, v72, v50
	v_and_b32_e32 v50, 60, v59
	v_and_b32_e32 v59, 60, v75
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v72, 16, v68
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v56, v73, v61
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v56, v46, v51 :: v_dual_and_b32 v61, 60, v77
	v_and_b32_e32 v57, 60, v57
	ds_load_b32 v57, v57
	ds_load_b32 v48, v48
	ds_load_b32 v46, v60
	ds_load_b32 v50, v50
	ds_load_b32 v51, v59
	ds_load_b32 v59, v61
	ds_load_b32 v47, v47
	v_and_b32_e32 v60, 0xffff0000, v68
	s_wait_dscnt 0x7
	v_dual_fmac_f32 v56, v49, v72 :: v_dual_lshlrev_b32 v49, 16, v69
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v56, v48, v60
	v_and_b32_e32 v48, 0xffff0000, v69
	v_dual_fmac_f32 v56, v57, v49 :: v_dual_lshlrev_b32 v57, 16, v70
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v56, v46, v48
	v_and_b32_e32 v46, 0x7f, v67
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v56, v50, v57 :: v_dual_lshlrev_b32 v57, 16, v71
	v_and_b32_e32 v49, 7, v67
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v50, v46, 20, 0x3c000000
	v_cmp_lt_u32_e32 vcc_lo, 7, v46
	v_cvt_f32_ubyte0_e32 v48, v49
	v_and_b32_e32 v49, 0xffff0000, v70
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v48, 0x3b000000, v48
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v56, v51, v49
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v48, v48, v50, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v46
	v_and_b32_e32 v50, 0xffff0000, v71
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v46, 0x7fc00000, v48, vcc_lo
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v56, v59, v57 :: v_dual_lshlrev_b32 v49, 24, v67
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v56, v47, v50
	v_and_or_b32 v46, 0x80000000, v49, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v56, v56, v46, 0
.LBB_147:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s1, exec_lo
	v_cmpx_ne_u64_e32 0, v[10:11]
	s_cbranch_execz .LBB_149
; %bb.148:
	s_and_b32 s3, s54, exec_lo
	s_mov_b32 s5, 0
	s_cselect_b32 s4, 3, 0
	v_lshrrev_b32_e32 v55, 6, v42
	s_wait_alu 0xfffe
	s_mul_u64 s[4:5], s[4:5], s[48:49]
	v_lshrrev_b32_e32 v57, 10, v42
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 1
	v_lshrrev_b32_e32 v59, 14, v42
	s_wait_alu 0xfffe
	v_add_co_u32 v50, vcc_lo, v44, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v51, null, s5, v45, vcc_lo
	v_lshrrev_b32_e32 v60, 18, v42
	v_lshrrev_b32_e32 v61, 22, v42
	s_clause 0x1
	global_load_b128 v[46:49], v[50:51], off
	global_load_b128 v[67:70], v[50:51], off offset:16
	v_and_b32_e32 v50, 15, v42
	v_lshrrev_b32_e32 v51, 2, v42
	v_lshrrev_b32_e32 v42, 26, v42
	v_and_b32_e32 v60, 60, v60
	v_lshrrev_b32_e32 v73, 10, v43
	v_lshlrev_b32_e32 v50, 2, v50
	v_lshrrev_b32_e32 v74, 22, v43
	v_and_b32_e32 v42, 60, v42
	ds_load_b32 v50, v50
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v72, 16, v47
	v_and_b32_e32 v51, 60, v51
	v_lshlrev_b32_e32 v71, 16, v46
	v_and_b32_e32 v46, 0xffff0000, v46
	v_lshlrev_b32_e32 v75, 16, v48
	v_and_b32_e32 v57, 60, v57
	ds_load_b32 v51, v51
	v_and_b32_e32 v55, 60, v55
	v_and_b32_e32 v48, 0xffff0000, v48
	v_and_b32_e32 v59, 60, v59
	v_and_b32_e32 v47, 0xffff0000, v47
	ds_load_b32 v55, v55
	s_wait_dscnt 0x2
	v_fma_f32 v50, v50, v71, 0
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v50, v51, v46
	v_lshrrev_b32_e32 v46, 2, v43
	v_lshrrev_b32_e32 v51, 6, v43
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v50, v55, v72 :: v_dual_and_b32 v61, 60, v61
	ds_load_b32 v57, v57
	ds_load_b32 v59, v59
	ds_load_b32 v60, v60
	ds_load_b32 v61, v61
	ds_load_b32 v42, v42
	v_and_b32_e32 v46, 60, v46
	v_lshrrev_b32_e32 v55, 14, v43
	v_lshrrev_b32_e32 v72, 18, v43
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v50, v57, v47 :: v_dual_and_b32 v57, 60, v73
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v50, v59, v75 :: v_dual_lshlrev_b32 v59, 16, v49
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v50, v60, v48 :: v_dual_and_b32 v49, 0xffff0000, v49
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v60, 16, v67
	v_and_b32_e32 v48, 60, v55
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v50, v61, v59 :: v_dual_and_b32 v55, 60, v72
	v_and_b32_e32 v71, 15, v43
	v_lshrrev_b32_e32 v43, 26, v43
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v50, v42, v49 :: v_dual_and_b32 v59, 60, v74
	v_lshlrev_b32_e32 v47, 2, v71
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v43, 60, v43
	ds_load_b32 v47, v47
	v_and_b32_e32 v51, 60, v51
	ds_load_b32 v51, v51
	ds_load_b32 v46, v46
	ds_load_b32 v42, v57
	ds_load_b32 v48, v48
	ds_load_b32 v49, v55
	ds_load_b32 v55, v59
	ds_load_b32 v43, v43
	s_wait_dscnt 0x7
	v_dual_fmac_f32 v50, v47, v60 :: v_dual_and_b32 v57, 0xffff0000, v67
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v50, v46, v57 :: v_dual_lshlrev_b32 v47, 16, v68
	v_and_b32_e32 v46, 0xffff0000, v68
	v_dual_fmac_f32 v50, v51, v47 :: v_dual_and_b32 v47, 7, v66
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v50, v42, v46 :: v_dual_lshlrev_b32 v51, 16, v69
	v_and_b32_e32 v42, 0x7f, v66
	v_cvt_f32_ubyte0_e32 v46, v47
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v50, v48, v51 :: v_dual_and_b32 v47, 0xffff0000, v69
	v_lshl_add_u32 v48, v42, 20, 0x3c000000
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v46, 0x3b000000, v46
	v_cmp_lt_u32_e32 vcc_lo, 7, v42
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v50, v49, v47 :: v_dual_lshlrev_b32 v51, 16, v70
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v46, v46, v48 :: v_dual_lshlrev_b32 v47, 24, v66
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v42
	v_and_b32_e32 v48, 0xffff0000, v70
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v50, v55, v51
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v42, 0x7fc00000, v46, vcc_lo
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v50, v43, v48
	v_and_or_b32 v42, 0x80000000, v47, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v55, v50, v42, 0
.LBB_149:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v59, 0
	s_mov_b32 s1, exec_lo
	v_cmpx_ne_u64_e32 0, v[8:9]
	s_cbranch_execz .LBB_151
; %bb.150:
	s_and_b32 s3, s53, exec_lo
	s_mov_b32 s5, 0
	s_cselect_b32 s4, 4, 0
	v_lshrrev_b32_e32 v50, 6, v40
	s_wait_alu 0xfffe
	s_mul_u64 s[4:5], s[4:5], s[48:49]
	v_lshrrev_b32_e32 v51, 10, v40
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 1
	v_lshrrev_b32_e32 v59, 14, v40
	s_wait_alu 0xfffe
	v_add_co_u32 v42, vcc_lo, v44, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v43, null, s5, v45, vcc_lo
	v_and_b32_e32 v50, 60, v50
	v_lshrrev_b32_e32 v60, 18, v40
	s_clause 0x1
	global_load_b128 v[46:49], v[42:43], off
	global_load_b128 v[66:69], v[42:43], off offset:16
	v_and_b32_e32 v42, 15, v40
	v_lshrrev_b32_e32 v43, 2, v40
	v_lshrrev_b32_e32 v61, 22, v40
	v_lshrrev_b32_e32 v40, 26, v40
	v_and_b32_e32 v60, 60, v60
	v_lshlrev_b32_e32 v42, 2, v42
	v_lshrrev_b32_e32 v72, 10, v41
	v_lshrrev_b32_e32 v73, 22, v41
	v_and_b32_e32 v40, 60, v40
	ds_load_b32 v50, v50
	ds_load_b32 v42, v42
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v74, 16, v48
	v_and_b32_e32 v43, 60, v43
	v_lshlrev_b32_e32 v70, 16, v46
	v_and_b32_e32 v46, 0xffff0000, v46
	v_lshlrev_b32_e32 v71, 16, v47
	v_and_b32_e32 v48, 0xffff0000, v48
	ds_load_b32 v43, v43
	s_wait_dscnt 0x1
	v_fma_f32 v42, v42, v70, 0
	v_and_b32_e32 v51, 60, v51
	v_and_b32_e32 v59, 60, v59
	v_and_b32_e32 v47, 0xffff0000, v47
	v_and_b32_e32 v70, 15, v41
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v42, v43, v46 :: v_dual_and_b32 v61, 60, v61
	v_lshrrev_b32_e32 v43, 2, v41
	v_lshrrev_b32_e32 v46, 6, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v42, v50, v71
	ds_load_b32 v51, v51
	ds_load_b32 v59, v59
	ds_load_b32 v60, v60
	ds_load_b32 v61, v61
	ds_load_b32 v40, v40
	v_lshrrev_b32_e32 v50, 14, v41
	v_and_b32_e32 v46, 60, v46
	v_lshrrev_b32_e32 v71, 18, v41
	v_lshrrev_b32_e32 v41, 26, v41
	ds_load_b32 v46, v46
	s_wait_dscnt 0x5
	v_dual_fmac_f32 v42, v51, v47 :: v_dual_and_b32 v41, 60, v41
	v_lshlrev_b32_e32 v47, 2, v70
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v59, v74 :: v_dual_lshlrev_b32 v59, 16, v49
	v_and_b32_e32 v49, 0xffff0000, v49
	ds_load_b32 v47, v47
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v42, v60, v48
	v_and_b32_e32 v51, 60, v72
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v60, 16, v66
	v_and_b32_e32 v48, 60, v50
	v_and_b32_e32 v50, 60, v71
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v42, v61, v59 :: v_dual_and_b32 v59, 60, v73
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v42, v40, v49 :: v_dual_and_b32 v43, 60, v43
	ds_load_b32 v43, v43
	ds_load_b32 v40, v51
	ds_load_b32 v48, v48
	ds_load_b32 v49, v50
	ds_load_b32 v50, v59
	ds_load_b32 v41, v41
	s_wait_dscnt 0x6
	v_dual_fmac_f32 v42, v47, v60 :: v_dual_and_b32 v51, 0xffff0000, v66
	v_lshlrev_b32_e32 v47, 16, v67
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v42, v43, v51
	v_and_b32_e32 v43, 0xffff0000, v67
	v_fmac_f32_e32 v42, v46, v47
	v_and_b32_e32 v46, 7, v65
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v42, v40, v43 :: v_dual_lshlrev_b32 v47, 16, v68
	v_and_b32_e32 v40, 0x7f, v65
	v_cvt_f32_ubyte0_e32 v43, v46
	v_and_b32_e32 v46, 0xffff0000, v68
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v42, v48, v47
	v_lshl_add_u32 v47, v40, 20, 0x3c000000
	v_mul_f32_e32 v43, 0x3b000000, v43
	v_cmp_lt_u32_e32 vcc_lo, 7, v40
	v_lshlrev_b32_e32 v48, 16, v69
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v42, v49, v46
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v43, v43, v47 :: v_dual_lshlrev_b32 v46, 24, v65
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v40
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v42, v50, v48 :: v_dual_and_b32 v47, 0xffff0000, v69
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v40, 0x7fc00000, v43, vcc_lo
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v42, v41, v47
	v_and_or_b32 v40, 0x80000000, v46, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v59, v42, v40, 0
.LBB_151:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s1, exec_lo
	v_cmpx_ne_u64_e32 0, v[6:7]
	s_cbranch_execz .LBB_153
; %bb.152:
	s_and_b32 s3, s52, exec_lo
	s_mov_b32 s5, 0
	s_cselect_b32 s4, 5, 0
	v_and_b32_e32 v50, 15, v38
	s_wait_alu 0xfffe
	s_mul_u64 s[4:5], s[4:5], s[48:49]
	v_lshrrev_b32_e32 v51, 2, v38
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 1
	v_lshrrev_b32_e32 v57, 6, v38
	s_wait_alu 0xfffe
	v_add_co_u32 v46, vcc_lo, v44, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v47, null, s5, v45, vcc_lo
	v_lshlrev_b32_e32 v50, 2, v50
	v_lshrrev_b32_e32 v60, 10, v38
	s_clause 0x1
	global_load_b128 v[40:43], v[46:47], off
	global_load_b128 v[46:49], v[46:47], off offset:16
	v_lshrrev_b32_e32 v61, 14, v38
	ds_load_b32 v50, v50
	v_lshrrev_b32_e32 v65, 18, v38
	v_lshrrev_b32_e32 v66, 22, v38
	v_lshrrev_b32_e32 v38, 26, v38
	v_and_b32_e32 v60, 60, v60
	v_lshrrev_b32_e32 v69, 10, v39
	v_lshrrev_b32_e32 v70, 22, v39
	v_and_b32_e32 v66, 60, v66
	v_and_b32_e32 v38, 60, v38
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v68, 16, v41
	v_and_b32_e32 v51, 60, v51
	v_and_b32_e32 v41, 0xffff0000, v41
	v_lshlrev_b32_e32 v67, 16, v40
	v_and_b32_e32 v40, 0xffff0000, v40
	v_lshlrev_b32_e32 v71, 16, v42
	ds_load_b32 v51, v51
	v_and_b32_e32 v57, 60, v57
	v_and_b32_e32 v42, 0xffff0000, v42
	v_and_b32_e32 v61, 60, v61
	v_and_b32_e32 v65, 60, v65
	ds_load_b32 v57, v57
	ds_load_b32 v60, v60
	ds_load_b32 v61, v61
	ds_load_b32 v65, v65
	ds_load_b32 v66, v66
	ds_load_b32 v38, v38
	s_wait_dscnt 0x7
	v_fma_f32 v50, v50, v67, 0
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v50, v51, v40
	v_lshrrev_b32_e32 v40, 2, v39
	v_lshrrev_b32_e32 v51, 6, v39
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v50, v57, v68
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v40, 60, v40
	v_and_b32_e32 v51, 60, v51
	v_lshrrev_b32_e32 v57, 14, v39
	v_lshrrev_b32_e32 v68, 18, v39
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v50, v60, v41
	v_and_b32_e32 v60, 60, v69
	ds_load_b32 v51, v51
	ds_load_b32 v40, v40
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v50, v61, v71
	v_lshlrev_b32_e32 v61, 16, v43
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v50, v65, v42 :: v_dual_and_b32 v43, 0xffff0000, v43
	v_and_b32_e32 v67, 15, v39
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v65, 16, v46
	v_and_b32_e32 v46, 0xffff0000, v46
	v_lshrrev_b32_e32 v39, 26, v39
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v50, v66, v61 :: v_dual_lshlrev_b32 v41, 2, v67
	v_and_b32_e32 v42, 60, v57
	v_and_b32_e32 v61, 60, v70
	v_and_b32_e32 v39, 60, v39
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v50, v38, v43
	ds_load_b32 v41, v41
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v50, v41, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v50, v40, v46
	v_lshlrev_b32_e32 v46, 16, v48
	v_and_b32_e32 v57, 60, v68
	ds_load_b32 v38, v60
	ds_load_b32 v42, v42
	ds_load_b32 v43, v57
	ds_load_b32 v57, v61
	ds_load_b32 v39, v39
	v_lshlrev_b32_e32 v41, 16, v47
	v_and_b32_e32 v40, 0xffff0000, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v50, v51, v41 :: v_dual_and_b32 v41, 7, v64
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v50, v38, v40
	v_and_b32_e32 v38, 0x7f, v64
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v40, v41
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v50, v42, v46 :: v_dual_and_b32 v41, 0xffff0000, v48
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v42, v38, 20, 0x3c000000
	v_mul_f32_e32 v40, 0x3b000000, v40
	v_cmp_lt_u32_e32 vcc_lo, 7, v38
	v_lshlrev_b32_e32 v46, 16, v49
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v50, v43, v41 :: v_dual_lshlrev_b32 v41, 24, v64
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v40, v40, v42, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v38
	v_and_b32_e32 v42, 0xffff0000, v49
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v50, v57, v46
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v38, 0x7fc00000, v40, vcc_lo
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v50, v39, v42
	v_and_or_b32 v38, 0x80000000, v41, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v57, v50, v38, 0
.LBB_153:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v61, 0
	s_mov_b32 s1, exec_lo
	v_cmpx_ne_u64_e32 0, v[4:5]
	s_cbranch_execz .LBB_155
; %bb.154:
	s_and_b32 s3, s51, exec_lo
	s_mov_b32 s5, 0
	s_cselect_b32 s4, 6, 0
	v_lshrrev_b32_e32 v50, 6, v36
	s_wait_alu 0xfffe
	s_mul_u64 s[4:5], s[4:5], s[48:49]
	v_lshrrev_b32_e32 v51, 10, v36
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 1
	v_lshrrev_b32_e32 v61, 14, v36
	s_wait_alu 0xfffe
	v_add_co_u32 v42, vcc_lo, v44, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v43, null, s5, v45, vcc_lo
	v_and_b32_e32 v50, 60, v50
	v_lshrrev_b32_e32 v64, 18, v36
	s_clause 0x1
	global_load_b128 v[38:41], v[42:43], off
	global_load_b128 v[46:49], v[42:43], off offset:16
	v_and_b32_e32 v42, 15, v36
	v_lshrrev_b32_e32 v43, 2, v36
	v_lshrrev_b32_e32 v65, 22, v36
	v_lshrrev_b32_e32 v36, 26, v36
	v_and_b32_e32 v64, 60, v64
	v_lshlrev_b32_e32 v42, 2, v42
	v_lshrrev_b32_e32 v68, 10, v37
	v_lshrrev_b32_e32 v69, 22, v37
	v_and_b32_e32 v36, 60, v36
	ds_load_b32 v50, v50
	ds_load_b32 v42, v42
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v70, 16, v40
	v_and_b32_e32 v43, 60, v43
	v_lshlrev_b32_e32 v66, 16, v38
	v_and_b32_e32 v38, 0xffff0000, v38
	v_and_b32_e32 v40, 0xffff0000, v40
	v_lshlrev_b32_e32 v67, 16, v39
	ds_load_b32 v43, v43
	s_wait_dscnt 0x1
	v_fma_f32 v42, v42, v66, 0
	v_and_b32_e32 v51, 60, v51
	v_and_b32_e32 v39, 0xffff0000, v39
	v_and_b32_e32 v66, 15, v37
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v42, v43, v38 :: v_dual_and_b32 v61, 60, v61
	v_lshrrev_b32_e32 v38, 2, v37
	v_lshrrev_b32_e32 v43, 6, v37
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v42, v50, v67 :: v_dual_and_b32 v65, 60, v65
	ds_load_b32 v51, v51
	ds_load_b32 v61, v61
	ds_load_b32 v64, v64
	ds_load_b32 v65, v65
	ds_load_b32 v36, v36
	v_and_b32_e32 v38, 60, v38
	v_lshrrev_b32_e32 v50, 14, v37
	v_lshrrev_b32_e32 v67, 18, v37
	v_lshrrev_b32_e32 v37, 26, v37
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v42, v51, v39 :: v_dual_and_b32 v37, 60, v37
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v42, v61, v70 :: v_dual_lshlrev_b32 v61, 16, v41
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v42, v64, v40 :: v_dual_and_b32 v41, 0xffff0000, v41
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v64, 16, v46
	v_and_b32_e32 v46, 0xffff0000, v46
	v_and_b32_e32 v40, 60, v50
	v_and_b32_e32 v50, 60, v67
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v42, v65, v61 :: v_dual_lshlrev_b32 v39, 2, v66
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v36, v41
	ds_load_b32 v39, v39
	v_and_b32_e32 v43, 60, v43
	ds_load_b32 v43, v43
	ds_load_b32 v38, v38
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v42, v39, v64
	v_and_b32_e32 v51, 60, v68
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v42, v38, v46 :: v_dual_lshlrev_b32 v39, 16, v47
	v_and_b32_e32 v38, 0xffff0000, v47
	v_dual_fmac_f32 v42, v43, v39 :: v_dual_lshlrev_b32 v43, 16, v48
	v_and_b32_e32 v61, 60, v69
	ds_load_b32 v36, v51
	ds_load_b32 v40, v40
	ds_load_b32 v41, v50
	ds_load_b32 v50, v61
	ds_load_b32 v37, v37
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v42, v36, v38 :: v_dual_and_b32 v39, 7, v63
	v_and_b32_e32 v36, 0x7f, v63
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v38, v39
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v42, v40, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v40, v36, 20, 0x3c000000
	v_mul_f32_e32 v38, 0x3b000000, v38
	v_cmp_lt_u32_e32 vcc_lo, 7, v36
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v38, v38, v40, vcc_lo
	v_and_b32_e32 v40, 0xffff0000, v49
	v_lshlrev_b32_e32 v43, 16, v49
	v_and_b32_e32 v39, 0xffff0000, v48
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v36
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v42, v41, v39
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v36, 0x7fc00000, v38 :: v_dual_lshlrev_b32 v39, 24, v63
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v42, v50, v43
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_or_b32 v36, 0x80000000, v39, v36
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v42, v37, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v61, v42, v36, 0
.LBB_155:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s1, exec_lo
	v_cmpx_ne_u64_e32 0, v[2:3]
	s_cbranch_execz .LBB_157
; %bb.156:
	s_and_b32 s3, s50, exec_lo
	s_mov_b32 s5, 0
	s_cselect_b32 s4, 7, 0
	v_lshrrev_b32_e32 v46, 6, v34
	s_wait_alu 0xfffe
	s_mul_u64 s[4:5], s[4:5], s[48:49]
	v_lshrrev_b32_e32 v47, 10, v34
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 1
	v_lshrrev_b32_e32 v48, 14, v34
	s_wait_alu 0xfffe
	v_add_co_u32 v40, vcc_lo, v44, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v41, null, s5, v45, vcc_lo
	v_and_b32_e32 v44, 15, v34
	v_lshrrev_b32_e32 v45, 2, v34
	s_clause 0x1
	global_load_b128 v[36:39], v[40:41], off
	global_load_b128 v[40:43], v[40:41], off offset:16
	v_and_b32_e32 v46, 60, v46
	v_lshlrev_b32_e32 v44, 2, v44
	v_lshrrev_b32_e32 v49, 18, v34
	v_lshrrev_b32_e32 v50, 22, v34
	v_lshrrev_b32_e32 v34, 26, v34
	v_and_b32_e32 v48, 60, v48
	ds_load_b32 v44, v44
	ds_load_b32 v46, v46
	v_and_b32_e32 v49, 60, v49
	v_and_b32_e32 v50, 60, v50
	v_and_b32_e32 v34, 60, v34
	v_lshrrev_b32_e32 v63, 10, v35
	v_lshrrev_b32_e32 v64, 22, v35
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v60, 16, v37
	v_and_b32_e32 v45, 60, v45
	v_lshlrev_b32_e32 v65, 16, v38
	v_and_b32_e32 v38, 0xffff0000, v38
	v_and_b32_e32 v37, 0xffff0000, v37
	v_lshlrev_b32_e32 v51, 16, v36
	ds_load_b32 v45, v45
	v_and_b32_e32 v36, 0xffff0000, v36
	s_wait_dscnt 0x2
	v_fma_f32 v44, v44, v51, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v44, v45, v36 :: v_dual_and_b32 v47, 60, v47
	ds_load_b32 v47, v47
	ds_load_b32 v48, v48
	ds_load_b32 v49, v49
	ds_load_b32 v50, v50
	ds_load_b32 v34, v34
	v_lshrrev_b32_e32 v36, 2, v35
	v_lshrrev_b32_e32 v45, 6, v35
	v_fmac_f32_e32 v44, v46, v60
	v_lshrrev_b32_e32 v46, 14, v35
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v36, 60, v36
	v_and_b32_e32 v45, 60, v45
	v_lshrrev_b32_e32 v60, 18, v35
	ds_load_b32 v45, v45
	ds_load_b32 v36, v36
	s_wait_dscnt 0x6
	v_dual_fmac_f32 v44, v47, v37 :: v_dual_and_b32 v51, 15, v35
	v_lshrrev_b32_e32 v35, 26, v35
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v44, v48, v65
	v_lshlrev_b32_e32 v48, 16, v39
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v44, v49, v38 :: v_dual_and_b32 v39, 0xffff0000, v39
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v49, 16, v40
	v_and_b32_e32 v40, 0xffff0000, v40
	v_lshlrev_b32_e32 v37, 2, v51
	v_and_b32_e32 v38, 60, v46
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v44, v50, v48
	v_and_b32_e32 v46, 60, v60
	v_and_b32_e32 v48, 60, v64
	ds_load_b32 v37, v37
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v44, v34, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v44, v37, v49 :: v_dual_and_b32 v47, 60, v63
	v_dual_fmac_f32 v44, v36, v40 :: v_dual_lshlrev_b32 v37, 16, v41
	v_lshlrev_b32_e32 v40, 16, v42
	v_and_b32_e32 v35, 60, v35
	ds_load_b32 v34, v47
	ds_load_b32 v38, v38
	ds_load_b32 v39, v46
	ds_load_b32 v46, v48
	ds_load_b32 v35, v35
	v_and_b32_e32 v36, 0xffff0000, v41
	v_fmac_f32_e32 v44, v45, v37
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v44, v34, v36
	v_and_b32_e32 v34, 0x7f, v62
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v44, v38, v40
	v_lshlrev_b32_e32 v40, 16, v43
	v_and_b32_e32 v37, 7, v62
	v_lshl_add_u32 v38, v34, 20, 0x3c000000
	v_cmp_lt_u32_e32 vcc_lo, 7, v34
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v36, v37
	v_and_b32_e32 v37, 0xffff0000, v42
	v_mul_f32_e32 v36, 0x3b000000, v36
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v44, v39, v37
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v36, v36, v38, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v34
	v_lshlrev_b32_e32 v37, 24, v62
	v_and_b32_e32 v38, 0xffff0000, v43
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v44, v46, v40
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v34, 0x7fc00000, v36, vcc_lo
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v44, v35, v38
	v_and_or_b32 v34, 0x80000000, v37, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v60, v44, v34, 0
.LBB_157:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
.LBB_158:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v34, 32, v58
	s_mov_b32 s70, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s68, v34
	s_cbranch_execz .LBB_194
; %bb.159:
	v_add_co_u32 v36, s8, 0x100, v16
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v37, null, 0, v17, s8
	v_add_co_u32 v38, s8, 0x100, v14
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v39, null, 0, v15, s8
	v_add_co_u32 v40, s8, 0x100, v12
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v41, null, 0, v13, s8
	v_add_co_u32 v42, s8, 0x100, v10
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v43, null, 0, v11, s8
	v_add_co_u32 v44, s8, 0x100, v8
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v45, null, 0, v9, s8
	v_add_co_u32 v46, s8, 0x100, v6
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v47, null, 0, v7, s8
	v_add_co_u32 v48, s8, 0x100, v4
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v49, null, 0, v5, s8
	v_add_co_u32 v50, s8, 0x100, v2
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v51, null, 0, v3, s8
	v_add_co_u32 v62, s8, v18, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v63, null, 0, v19, s8
	v_add_co_u32 v64, s8, v20, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v65, null, 0, v21, s8
	v_add_co_u32 v66, s8, v22, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v67, null, 0, v23, s8
	v_add_co_u32 v68, s8, v24, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v69, null, 0, v25, s8
	v_add_co_u32 v70, s8, v26, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v71, null, 0, v27, s8
	v_add_co_u32 v72, s8, v28, 32
	v_and_b32_e32 v34, 31, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v73, null, 0, v29, s8
	v_add_co_u32 v74, s8, v30, 32
	s_lshl_b64 s[60:61], s[46:47], 1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v75, null, 0, v31, s8
	v_add_co_u32 v76, s8, v32, 32
	v_cmp_ne_u64_e64 s0, 0, v[14:15]
	v_cmp_ne_u64_e64 s1, 0, v[12:13]
	v_cmp_ne_u64_e64 s3, 0, v[10:11]
	v_cmp_ne_u64_e64 s4, 0, v[8:9]
	v_cmp_ne_u64_e64 s5, 0, v[6:7]
	v_cmp_ne_u64_e64 s6, 0, v[4:5]
	v_cmp_ne_u64_e64 s7, 0, v[2:3]
	s_add_nc_u64 s[46:47], s[60:61], 2
	s_add_nc_u64 s[52:53], s[60:61], 4
	s_add_nc_u64 s[54:55], s[60:61], 6
	s_add_nc_u64 s[56:57], s[60:61], 8
	s_add_nc_u64 s[58:59], s[60:61], 10
	s_add_nc_u64 s[72:73], s[60:61], 12
	s_add_nc_u64 s[60:61], s[60:61], 14
	v_lshlrev_b32_e32 v35, 3, v34
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v77, null, 0, v33, s8
	v_lshlrev_b32_e32 v78, 5, v34
	s_mul_u64 s[46:47], s[46:47], s[48:49]
	s_mul_u64 s[52:53], s[52:53], s[48:49]
	s_mul_u64 s[54:55], s[54:55], s[48:49]
	s_wait_alu 0xfffe
	s_mul_u64 s[56:57], s[56:57], s[48:49]
	s_mul_u64 s[58:59], s[58:59], s[48:49]
	s_mul_u64 s[72:73], s[72:73], s[48:49]
	s_mul_u64 s[60:61], s[60:61], s[48:49]
	s_mov_b64 s[50:51], 32
	s_add_nc_u64 s[46:47], s[30:31], s[46:47]
	s_add_nc_u64 s[52:53], s[30:31], s[52:53]
	s_add_nc_u64 s[54:55], s[30:31], s[54:55]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[56:57], s[30:31], s[56:57]
	s_add_nc_u64 s[58:59], s[30:31], s[58:59]
	s_add_nc_u64 s[48:49], s[30:31], s[72:73]
	s_add_nc_u64 s[60:61], s[30:31], s[60:61]
	s_mov_b32 s71, 0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[16:17]
	s_branch .LBB_161
.LBB_160:                              ;   in Loop: Header=BB8_161 Depth=1
	s_or_b32 exec_lo, exec_lo, s72
	v_add_co_u32 v36, s8, 0x100, v36
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v37, null, 0, v37, s8
	v_add_co_u32 v38, s8, 0x100, v38
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v39, null, 0, v39, s8
	v_add_co_u32 v40, s8, 0x100, v40
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v41, null, 0, v41, s8
	v_add_co_u32 v42, s8, 0x100, v42
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v43, null, 0, v43, s8
	v_add_co_u32 v44, s8, 0x100, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v45, null, 0, v45, s8
	v_add_co_u32 v46, s8, 0x100, v46
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v47, null, 0, v47, s8
	v_add_co_u32 v48, s8, 0x100, v48
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v49, null, 0, v49, s8
	v_add_co_u32 v50, s8, 0x100, v50
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v51, null, 0, v51, s8
	v_add_co_u32 v62, s8, v62, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v63, null, 0, v63, s8
	v_add_co_u32 v64, s8, v64, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v65, null, 0, v65, s8
	v_add_co_u32 v66, s8, v66, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v67, null, 0, v67, s8
	v_add_co_u32 v68, s8, v68, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v69, null, 0, v69, s8
	v_add_co_u32 v70, s8, v70, 32
	s_add_nc_u64 s[50:51], s[50:51], 32
	v_add_co_ci_u32_e64 v71, null, 0, v71, s8
	v_add_co_u32 v72, s8, v72, 32
	s_wait_loadcnt 0x1
	s_wait_alu 0xfffe
	v_add_nc_u32_e32 v18, s50, v34
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v73, null, 0, v73, s8
	v_add_co_u32 v74, s8, v74, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v75, null, 0, v75, s8
	v_add_co_u32 v76, s8, v76, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v77, null, 0, v77, s8
	v_cmp_le_i32_e64 s8, s68, v18
	s_add_nc_u64 s[10:11], s[10:11], 0x400
	s_add_nc_u64 s[46:47], s[46:47], 0x400
	s_add_nc_u64 s[52:53], s[52:53], 0x400
	s_add_nc_u64 s[54:55], s[54:55], 0x400
	s_add_nc_u64 s[56:57], s[56:57], 0x400
	s_add_nc_u64 s[58:59], s[58:59], 0x400
	s_add_nc_u64 s[48:49], s[48:49], 0x400
	s_or_b32 s71, s8, s71
	s_add_nc_u64 s[60:61], s[60:61], 0x400
	s_and_not1_b32 exec_lo, exec_lo, s71
	s_cbranch_execz .LBB_193
.LBB_161:                              ; =>This Inner Loop Header: Depth=1
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0
	s_wait_loadcnt_dscnt 0x0
	v_mov_b32_e32 v86, 0
	s_and_saveexec_b32 s72, vcc_lo
	s_cbranch_execz .LBB_163
; %bb.162:                              ;   in Loop: Header=BB8_161 Depth=1
	v_add_co_u32 v18, s8, v36, v35
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v19, null, 0, v37, s8
	v_add_co_u32 v20, s8, v62, v34
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v21, null, 0, v63, s8
	global_load_b64 v[32:33], v[18:19], off th:TH_LOAD_NT
	flat_load_u8 v86, v[20:21]
.LBB_163:                              ;   in Loop: Header=BB8_161 Depth=1
	s_or_b32 exec_lo, exec_lo, s72
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v85, 0
	s_and_saveexec_b32 s72, s0
	s_cbranch_execz .LBB_165
; %bb.164:                              ;   in Loop: Header=BB8_161 Depth=1
	v_add_co_u32 v18, s8, v38, v35
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v19, null, 0, v39, s8
	v_add_co_u32 v20, s8, v64, v34
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v21, null, 0, v65, s8
	global_load_b64 v[30:31], v[18:19], off th:TH_LOAD_NT
	flat_load_u8 v85, v[20:21]
.LBB_165:                              ;   in Loop: Header=BB8_161 Depth=1
	s_or_b32 exec_lo, exec_lo, s72
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v84, 0
	s_and_saveexec_b32 s72, s1
	s_cbranch_execz .LBB_167
; %bb.166:                              ;   in Loop: Header=BB8_161 Depth=1
	v_add_co_u32 v18, s8, v40, v35
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v19, null, 0, v41, s8
	v_add_co_u32 v20, s8, v66, v34
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v21, null, 0, v67, s8
	global_load_b64 v[28:29], v[18:19], off th:TH_LOAD_NT
	flat_load_u8 v84, v[20:21]
.LBB_167:                              ;   in Loop: Header=BB8_161 Depth=1
	s_or_b32 exec_lo, exec_lo, s72
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v83, 0
	s_and_saveexec_b32 s72, s3
	s_cbranch_execz .LBB_169
; %bb.168:                              ;   in Loop: Header=BB8_161 Depth=1
	v_add_co_u32 v18, s8, v42, v35
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v19, null, 0, v43, s8
	v_add_co_u32 v20, s8, v68, v34
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v21, null, 0, v69, s8
	global_load_b64 v[26:27], v[18:19], off th:TH_LOAD_NT
	flat_load_u8 v83, v[20:21]
.LBB_169:                              ;   in Loop: Header=BB8_161 Depth=1
	s_or_b32 exec_lo, exec_lo, s72
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v82, 0
	s_and_saveexec_b32 s72, s4
	s_cbranch_execz .LBB_171
; %bb.170:                              ;   in Loop: Header=BB8_161 Depth=1
	v_add_co_u32 v18, s8, v44, v35
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v19, null, 0, v45, s8
	v_add_co_u32 v20, s8, v70, v34
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v21, null, 0, v71, s8
	global_load_b64 v[24:25], v[18:19], off th:TH_LOAD_NT
	flat_load_u8 v82, v[20:21]
.LBB_171:                              ;   in Loop: Header=BB8_161 Depth=1
	s_or_b32 exec_lo, exec_lo, s72
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v81, 0
	s_and_saveexec_b32 s72, s5
	s_cbranch_execz .LBB_173
; %bb.172:                              ;   in Loop: Header=BB8_161 Depth=1
	v_add_co_u32 v18, s8, v46, v35
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v19, null, 0, v47, s8
	v_add_co_u32 v79, s8, v72, v34
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v80, null, 0, v73, s8
	global_load_b64 v[22:23], v[18:19], off th:TH_LOAD_NT
	flat_load_u8 v81, v[79:80]
.LBB_173:                              ;   in Loop: Header=BB8_161 Depth=1
	s_or_b32 exec_lo, exec_lo, s72
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v80, 0
	s_and_saveexec_b32 s72, s6
	s_cbranch_execz .LBB_175
; %bb.174:                              ;   in Loop: Header=BB8_161 Depth=1
	v_add_co_u32 v18, s8, v48, v35
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v19, null, 0, v49, s8
	v_add_co_u32 v79, s8, v74, v34
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v80, null, 0, v75, s8
	global_load_b64 v[20:21], v[18:19], off th:TH_LOAD_NT
	flat_load_u8 v80, v[79:80]
.LBB_175:                              ;   in Loop: Header=BB8_161 Depth=1
	s_or_b32 exec_lo, exec_lo, s72
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v18, 0
	v_mov_b32_e32 v79, 0
	s_and_saveexec_b32 s72, s7
	s_cbranch_execz .LBB_184
; %bb.176:                              ;   in Loop: Header=BB8_161 Depth=1
	v_add_co_u32 v18, s8, v50, v35
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v19, null, 0, v51, s8
	v_add_co_u32 v87, s8, v76, v34
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v88, null, 0, v77, s8
	global_load_b64 v[18:19], v[18:19], off th:TH_LOAD_NT
	flat_load_u8 v79, v[87:88]
	s_or_b32 exec_lo, exec_lo, s72
	s_and_saveexec_b32 s72, vcc_lo
	s_cbranch_execnz .LBB_185
.LBB_177:                              ;   in Loop: Header=BB8_161 Depth=1
	s_or_b32 exec_lo, exec_lo, s72
	s_and_saveexec_b32 s72, s0
	s_cbranch_execz .LBB_186
.LBB_178:                              ;   in Loop: Header=BB8_161 Depth=1
	s_wait_loadcnt 0x1
	s_wait_alu 0xfffe
	v_add_co_u32 v32, s8, s46, v78
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v33, null, s47, 0, s8
	v_lshrrev_b32_e32 v94, 6, v30
	v_lshrrev_b32_e32 v95, 10, v30
	v_lshrrev_b32_e32 v96, 14, v30
	s_wait_loadcnt_dscnt 0x0
	s_clause 0x1
	global_load_b128 v[86:89], v[32:33], off offset:1024
	global_load_b128 v[90:93], v[32:33], off offset:1040
	v_and_b32_e32 v32, 15, v30
	v_lshrrev_b32_e32 v33, 2, v30
	v_and_b32_e32 v94, 60, v94
	v_lshrrev_b32_e32 v97, 18, v30
	v_lshrrev_b32_e32 v98, 22, v30
	v_lshlrev_b32_e32 v32, 2, v32
	v_lshrrev_b32_e32 v30, 26, v30
	v_and_b32_e32 v96, 60, v96
	v_lshrrev_b32_e32 v101, 10, v31
	v_and_b32_e32 v98, 60, v98
	ds_load_b32 v32, v32
	ds_load_b32 v94, v94
	v_and_b32_e32 v30, 60, v30
	v_lshrrev_b32_e32 v102, 22, v31
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v100, 16, v87
	v_and_b32_e32 v33, 60, v33
	v_lshlrev_b32_e32 v103, 16, v88
	v_lshlrev_b32_e32 v99, 16, v86
	v_and_b32_e32 v86, 0xffff0000, v86
	v_and_b32_e32 v87, 0xffff0000, v87
	ds_load_b32 v33, v33
	v_and_b32_e32 v88, 0xffff0000, v88
	s_wait_dscnt 0x2
	v_fma_f32 v32, v32, v99, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v32, v33, v86 :: v_dual_and_b32 v95, 60, v95
	v_lshrrev_b32_e32 v33, 2, v31
	v_lshrrev_b32_e32 v86, 6, v31
	v_dual_fmac_f32 v32, v94, v100 :: v_dual_and_b32 v97, 60, v97
	ds_load_b32 v95, v95
	ds_load_b32 v96, v96
	ds_load_b32 v97, v97
	ds_load_b32 v98, v98
	ds_load_b32 v30, v30
	v_and_b32_e32 v33, 60, v33
	v_and_b32_e32 v86, 60, v86
	v_lshrrev_b32_e32 v94, 14, v31
	v_lshrrev_b32_e32 v100, 18, v31
	ds_load_b32 v33, v33
	ds_load_b32 v86, v86
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v32, v95, v87
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v32, v96, v103
	v_and_b32_e32 v99, 15, v31
	v_lshlrev_b32_e32 v96, 16, v89
	v_and_b32_e32 v89, 0xffff0000, v89
	v_lshrrev_b32_e32 v31, 26, v31
	s_wait_loadcnt_dscnt 0x4
	v_dual_fmac_f32 v32, v97, v88 :: v_dual_lshlrev_b32 v97, 16, v90
	v_and_b32_e32 v90, 0xffff0000, v90
	v_lshlrev_b32_e32 v87, 2, v99
	v_and_b32_e32 v88, 60, v94
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v32, v98, v96
	v_and_b32_e32 v94, 60, v100
	v_and_b32_e32 v96, 60, v102
	ds_load_b32 v87, v87
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v32, v30, v89 :: v_dual_and_b32 v31, 60, v31
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v32, v87, v97 :: v_dual_lshlrev_b32 v87, 16, v91
	v_fmac_f32_e32 v32, v33, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v32, v86, v87 :: v_dual_lshlrev_b32 v87, 16, v92
	v_and_b32_e32 v95, 60, v101
	ds_load_b32 v30, v95
	ds_load_b32 v88, v88
	ds_load_b32 v89, v94
	ds_load_b32 v94, v96
	ds_load_b32 v31, v31
	v_and_b32_e32 v33, 0xffff0000, v91
	v_and_b32_e32 v86, 7, v85
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v32, v30, v33
	v_and_b32_e32 v30, 0x7f, v85
	v_cvt_f32_ubyte0_e32 v33, v86
	v_and_b32_e32 v86, 0xffff0000, v92
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v32, v88, v87 :: v_dual_lshlrev_b32 v85, 24, v85
	v_lshl_add_u32 v87, v30, 20, 0x3c000000
	v_mul_f32_e32 v33, 0x3b000000, v33
	v_cmp_lt_u32_e64 s8, 7, v30
	v_lshlrev_b32_e32 v88, 16, v93
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v32, v89, v86
	v_and_b32_e32 v86, 0xffff0000, v93
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v33, v33, v87, s8
	v_cmp_ne_u32_e64 s8, 0x7f, v30
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v32, v94, v88
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v30, 0x7fc00000, v33, s8
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v32, v31, v86
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v30, 0x80000000, v85, v30
	v_fmac_f32_e32 v53, v32, v30
	s_or_b32 exec_lo, exec_lo, s72
	s_and_saveexec_b32 s72, s1
	s_cbranch_execnz .LBB_187
.LBB_179:                              ;   in Loop: Header=BB8_161 Depth=1
	s_or_b32 exec_lo, exec_lo, s72
	s_and_saveexec_b32 s72, s3
	s_cbranch_execz .LBB_188
.LBB_180:                              ;   in Loop: Header=BB8_161 Depth=1
	s_wait_loadcnt 0x1
	s_wait_alu 0xfffe
	v_add_co_u32 v32, s8, s54, v78
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v33, null, s55, 0, s8
	v_lshrrev_b32_e32 v88, 6, v26
	v_lshrrev_b32_e32 v89, 10, v26
	v_lshrrev_b32_e32 v90, 14, v26
	global_load_b128 v[28:31], v[32:33], off offset:1024
	s_wait_loadcnt_dscnt 0x100
	global_load_b128 v[84:87], v[32:33], off offset:1040
	v_and_b32_e32 v32, 15, v26
	v_lshrrev_b32_e32 v33, 2, v26
	v_and_b32_e32 v88, 60, v88
	v_lshrrev_b32_e32 v91, 18, v26
	v_lshrrev_b32_e32 v92, 22, v26
	v_lshlrev_b32_e32 v32, 2, v32
	v_lshrrev_b32_e32 v26, 26, v26
	v_and_b32_e32 v90, 60, v90
	v_lshrrev_b32_e32 v95, 10, v27
	v_and_b32_e32 v92, 60, v92
	ds_load_b32 v32, v32
	ds_load_b32 v88, v88
	v_and_b32_e32 v26, 60, v26
	v_lshrrev_b32_e32 v96, 22, v27
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v94, 16, v29
	v_and_b32_e32 v33, 60, v33
	v_lshlrev_b32_e32 v97, 16, v30
	v_lshlrev_b32_e32 v93, 16, v28
	v_and_b32_e32 v28, 0xffff0000, v28
	v_and_b32_e32 v29, 0xffff0000, v29
	ds_load_b32 v33, v33
	v_and_b32_e32 v30, 0xffff0000, v30
	s_wait_dscnt 0x2
	v_fma_f32 v32, v32, v93, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v32, v33, v28 :: v_dual_and_b32 v89, 60, v89
	v_lshrrev_b32_e32 v28, 2, v27
	v_lshrrev_b32_e32 v33, 6, v27
	v_dual_fmac_f32 v32, v88, v94 :: v_dual_and_b32 v91, 60, v91
	ds_load_b32 v89, v89
	ds_load_b32 v90, v90
	ds_load_b32 v91, v91
	ds_load_b32 v92, v92
	ds_load_b32 v26, v26
	v_and_b32_e32 v28, 60, v28
	v_lshrrev_b32_e32 v88, 14, v27
	v_lshrrev_b32_e32 v94, 18, v27
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v32, v89, v29
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v32, v90, v97 :: v_dual_and_b32 v93, 15, v27
	v_lshlrev_b32_e32 v90, 16, v31
	v_and_b32_e32 v31, 0xffff0000, v31
	v_lshrrev_b32_e32 v27, 26, v27
	s_wait_loadcnt_dscnt 0x2
	v_dual_fmac_f32 v32, v91, v30 :: v_dual_lshlrev_b32 v91, 16, v84
	v_and_b32_e32 v84, 0xffff0000, v84
	v_lshlrev_b32_e32 v29, 2, v93
	v_and_b32_e32 v30, 60, v88
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v32, v92, v90
	v_and_b32_e32 v88, 60, v94
	v_and_b32_e32 v90, 60, v96
	ds_load_b32 v29, v29
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v32, v26, v31 :: v_dual_and_b32 v33, 60, v33
	ds_load_b32 v33, v33
	ds_load_b32 v28, v28
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v32, v29, v91 :: v_dual_lshlrev_b32 v29, 16, v85
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v32, v28, v84 :: v_dual_and_b32 v89, 60, v95
	v_and_b32_e32 v28, 0xffff0000, v85
	v_dual_fmac_f32 v32, v33, v29 :: v_dual_lshlrev_b32 v33, 16, v86
	v_and_b32_e32 v27, 60, v27
	ds_load_b32 v26, v89
	ds_load_b32 v30, v30
	ds_load_b32 v31, v88
	ds_load_b32 v88, v90
	ds_load_b32 v27, v27
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v32, v26, v28
	v_and_b32_e32 v26, 0x7f, v83
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v32, v30, v33 :: v_dual_lshlrev_b32 v33, 16, v87
	v_and_b32_e32 v29, 7, v83
	v_lshl_add_u32 v30, v26, 20, 0x3c000000
	v_cmp_lt_u32_e64 s8, 7, v26
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v28, v29
	v_and_b32_e32 v29, 0xffff0000, v86
	v_mul_f32_e32 v28, 0x3b000000, v28
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v32, v31, v29 :: v_dual_lshlrev_b32 v29, 24, v83
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v28, v28, v30, s8
	v_cmp_ne_u32_e64 s8, 0x7f, v26
	v_and_b32_e32 v30, 0xffff0000, v87
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v32, v88, v33
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v26, 0x7fc00000, v28, s8
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v32, v27, v30
	v_and_or_b32 v26, 0x80000000, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v55, v32, v26
	s_or_b32 exec_lo, exec_lo, s72
	s_and_saveexec_b32 s72, s4
	s_cbranch_execnz .LBB_189
.LBB_181:                              ;   in Loop: Header=BB8_161 Depth=1
	s_or_b32 exec_lo, exec_lo, s72
	s_and_saveexec_b32 s72, s5
	s_cbranch_execz .LBB_190
.LBB_182:                              ;   in Loop: Header=BB8_161 Depth=1
	s_wait_loadcnt 0x1
	s_wait_alu 0xfffe
	v_add_co_u32 v28, s8, s58, v78
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v29, null, s59, 0, s8
	v_and_b32_e32 v32, 15, v22
	v_lshrrev_b32_e32 v33, 2, v22
	s_wait_loadcnt_dscnt 0x0
	v_lshrrev_b32_e32 v82, 6, v22
	s_clause 0x1
	global_load_b128 v[24:27], v[28:29], off offset:1024
	global_load_b128 v[28:31], v[28:29], off offset:1040
	v_lshrrev_b32_e32 v83, 10, v22
	v_lshlrev_b32_e32 v32, 2, v32
	v_lshrrev_b32_e32 v84, 14, v22
	v_and_b32_e32 v82, 60, v82
	v_lshrrev_b32_e32 v85, 18, v22
	v_lshrrev_b32_e32 v86, 22, v22
	ds_load_b32 v32, v32
	v_lshrrev_b32_e32 v22, 26, v22
	v_and_b32_e32 v84, 60, v84
	v_lshrrev_b32_e32 v89, 10, v23
	v_and_b32_e32 v86, 60, v86
	v_lshrrev_b32_e32 v90, 22, v23
	v_and_b32_e32 v22, 60, v22
	ds_load_b32 v82, v82
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v88, 16, v25
	v_and_b32_e32 v33, 60, v33
	v_and_b32_e32 v25, 0xffff0000, v25
	v_lshlrev_b32_e32 v87, 16, v24
	v_and_b32_e32 v24, 0xffff0000, v24
	v_lshlrev_b32_e32 v91, 16, v26
	ds_load_b32 v33, v33
	v_and_b32_e32 v85, 60, v85
	s_wait_dscnt 0x2
	v_fma_f32 v32, v32, v87, 0
	v_and_b32_e32 v83, 60, v83
	v_and_b32_e32 v26, 0xffff0000, v26
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v32, v33, v24
	ds_load_b32 v83, v83
	ds_load_b32 v84, v84
	ds_load_b32 v85, v85
	ds_load_b32 v86, v86
	ds_load_b32 v22, v22
	v_lshrrev_b32_e32 v24, 2, v23
	v_lshrrev_b32_e32 v33, 6, v23
	v_fmac_f32_e32 v32, v82, v88
	v_lshrrev_b32_e32 v82, 14, v23
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v24, 60, v24
	v_and_b32_e32 v33, 60, v33
	v_lshrrev_b32_e32 v88, 18, v23
	ds_load_b32 v33, v33
	ds_load_b32 v24, v24
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v32, v83, v25
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v32, v84, v91
	v_and_b32_e32 v87, 15, v23
	v_lshlrev_b32_e32 v84, 16, v27
	v_lshrrev_b32_e32 v23, 26, v23
	s_wait_loadcnt_dscnt 0x4
	v_dual_fmac_f32 v32, v85, v26 :: v_dual_lshlrev_b32 v85, 16, v28
	v_and_b32_e32 v28, 0xffff0000, v28
	v_lshlrev_b32_e32 v25, 2, v87
	v_and_b32_e32 v26, 60, v82
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v32, v86, v84
	v_and_b32_e32 v82, 60, v88
	v_and_b32_e32 v84, 60, v90
	ds_load_b32 v25, v25
	v_and_b32_e32 v27, 0xffff0000, v27
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v32, v22, v27
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v32, v25, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v32, v24, v28 :: v_dual_lshlrev_b32 v25, 16, v29
	v_lshlrev_b32_e32 v28, 16, v30
	v_and_b32_e32 v83, 60, v89
	v_and_b32_e32 v23, 60, v23
	v_and_b32_e32 v24, 0xffff0000, v29
	v_fmac_f32_e32 v32, v33, v25
	ds_load_b32 v22, v83
	ds_load_b32 v26, v26
	ds_load_b32 v27, v82
	ds_load_b32 v82, v84
	ds_load_b32 v23, v23
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v32, v22, v24
	v_and_b32_e32 v22, 0x7f, v81
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v32, v26, v28
	v_lshlrev_b32_e32 v28, 16, v31
	v_and_b32_e32 v25, 7, v81
	v_lshl_add_u32 v26, v22, 20, 0x3c000000
	v_cmp_lt_u32_e64 s8, 7, v22
	v_cvt_f32_ubyte0_e32 v24, v25
	v_and_b32_e32 v25, 0xffff0000, v30
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v24, 0x3b000000, v24
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v32, v27, v25
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v24, v24, v26, s8
	v_cmp_ne_u32_e64 s8, 0x7f, v22
	v_and_b32_e32 v26, 0xffff0000, v31
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v32, v82, v28 :: v_dual_lshlrev_b32 v25, 24, v81
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v22, 0x7fc00000, v24, s8
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v32, v23, v26
	v_and_or_b32 v22, 0x80000000, v25, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v57, v32, v22
	s_or_b32 exec_lo, exec_lo, s72
	s_and_saveexec_b32 s72, s6
	s_cbranch_execnz .LBB_191
.LBB_183:                              ;   in Loop: Header=BB8_161 Depth=1
	s_or_b32 exec_lo, exec_lo, s72
	s_and_saveexec_b32 s72, s7
	s_cbranch_execz .LBB_160
	s_branch .LBB_192
.LBB_184:                              ;   in Loop: Header=BB8_161 Depth=1
	s_or_b32 exec_lo, exec_lo, s72
	s_and_saveexec_b32 s72, vcc_lo
	s_cbranch_execz .LBB_177
.LBB_185:                              ;   in Loop: Header=BB8_161 Depth=1
	s_wait_alu 0xfffe
	v_add_co_u32 v91, s8, s10, v78
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v92, null, s11, 0, s8
	s_wait_loadcnt 0x1
	v_and_b32_e32 v95, 15, v32
	v_lshrrev_b32_e32 v96, 2, v32
	v_lshrrev_b32_e32 v97, 6, v32
	s_clause 0x1
	global_load_b128 v[87:90], v[91:92], off offset:1024
	global_load_b128 v[91:94], v[91:92], off offset:1040
	v_lshrrev_b32_e32 v98, 10, v32
	v_lshlrev_b32_e32 v95, 2, v95
	v_lshrrev_b32_e32 v99, 14, v32
	v_and_b32_e32 v97, 60, v97
	v_lshrrev_b32_e32 v100, 18, v32
	v_lshrrev_b32_e32 v101, 22, v32
	ds_load_b32 v95, v95
	v_lshrrev_b32_e32 v32, 26, v32
	v_and_b32_e32 v99, 60, v99
	v_lshrrev_b32_e32 v104, 10, v33
	v_and_b32_e32 v101, 60, v101
	v_lshrrev_b32_e32 v105, 22, v33
	ds_load_b32 v97, v97
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v103, 16, v88
	v_and_b32_e32 v96, 60, v96
	v_and_b32_e32 v88, 0xffff0000, v88
	v_lshlrev_b32_e32 v102, 16, v87
	v_and_b32_e32 v87, 0xffff0000, v87
	v_lshlrev_b32_e32 v106, 16, v89
	ds_load_b32 v96, v96
	v_and_b32_e32 v100, 60, v100
	s_wait_dscnt 0x2
	v_fma_f32 v95, v95, v102, 0
	v_and_b32_e32 v98, 60, v98
	v_and_b32_e32 v89, 0xffff0000, v89
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v95, v96, v87 :: v_dual_and_b32 v32, 60, v32
	ds_load_b32 v98, v98
	ds_load_b32 v99, v99
	ds_load_b32 v100, v100
	ds_load_b32 v101, v101
	ds_load_b32 v32, v32
	v_lshrrev_b32_e32 v87, 2, v33
	v_lshrrev_b32_e32 v96, 6, v33
	v_fmac_f32_e32 v95, v97, v103
	v_lshrrev_b32_e32 v97, 14, v33
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v87, 60, v87
	v_and_b32_e32 v96, 60, v96
	v_lshrrev_b32_e32 v103, 18, v33
	ds_load_b32 v96, v96
	ds_load_b32 v87, v87
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v95, v98, v88
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v95, v99, v106
	v_lshlrev_b32_e32 v99, 16, v90
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v95, v100, v89 :: v_dual_and_b32 v90, 0xffff0000, v90
	v_and_b32_e32 v102, 15, v33
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v100, 16, v91
	v_and_b32_e32 v91, 0xffff0000, v91
	v_lshrrev_b32_e32 v33, 26, v33
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v95, v101, v99 :: v_dual_lshlrev_b32 v88, 2, v102
	v_and_b32_e32 v89, 60, v97
	v_and_b32_e32 v97, 60, v103
	v_and_b32_e32 v99, 60, v105
	v_and_b32_e32 v33, 60, v33
	ds_load_b32 v88, v88
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v95, v32, v90
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v95, v88, v100
	v_dual_fmac_f32 v95, v87, v91 :: v_dual_lshlrev_b32 v88, 16, v92
	v_lshlrev_b32_e32 v91, 16, v93
	v_and_b32_e32 v98, 60, v104
	ds_load_b32 v32, v98
	ds_load_b32 v89, v89
	ds_load_b32 v90, v97
	ds_load_b32 v97, v99
	ds_load_b32 v33, v33
	v_and_b32_e32 v87, 0xffff0000, v92
	v_fmac_f32_e32 v95, v96, v88
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v95, v32, v87
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v95, v89, v91
	v_lshlrev_b32_e32 v91, 16, v94
	v_and_b32_e32 v88, 7, v86
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v87, v88
	v_and_b32_e32 v88, 0xffff0000, v93
	v_mul_f32_e32 v87, 0x3b000000, v87
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v95, v90, v88 :: v_dual_and_b32 v88, 0xffff0000, v94
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v95, v97, v91 :: v_dual_and_b32 v32, 0x7f, v86
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v95, v33, v88 :: v_dual_lshlrev_b32 v86, 24, v86
	v_lshl_add_u32 v89, v32, 20, 0x3c000000
	v_cmp_lt_u32_e64 s8, 7, v32
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v87, v87, v89, s8
	v_cmp_ne_u32_e64 s8, 0x7f, v32
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v32, 0x7fc00000, v87, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v32, 0x80000000, v86, v32
	v_fmac_f32_e32 v54, v95, v32
	s_or_b32 exec_lo, exec_lo, s72
	s_and_saveexec_b32 s72, s0
	s_cbranch_execnz .LBB_178
.LBB_186:                              ;   in Loop: Header=BB8_161 Depth=1
	s_or_b32 exec_lo, exec_lo, s72
	s_and_saveexec_b32 s72, s1
	s_cbranch_execz .LBB_179
.LBB_187:                              ;   in Loop: Header=BB8_161 Depth=1
	s_wait_loadcnt_dscnt 0x0
	s_wait_alu 0xfffe
	v_add_co_u32 v85, s8, s52, v78
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v86, null, s53, 0, s8
	v_and_b32_e32 v89, 15, v28
	v_lshrrev_b32_e32 v90, 2, v28
	v_lshrrev_b32_e32 v91, 6, v28
	s_clause 0x1
	global_load_b128 v[30:33], v[85:86], off offset:1024
	global_load_b128 v[85:88], v[85:86], off offset:1040
	v_lshrrev_b32_e32 v92, 10, v28
	v_lshlrev_b32_e32 v89, 2, v89
	v_lshrrev_b32_e32 v93, 14, v28
	v_and_b32_e32 v91, 60, v91
	v_lshrrev_b32_e32 v94, 18, v28
	v_lshrrev_b32_e32 v95, 22, v28
	ds_load_b32 v89, v89
	v_lshrrev_b32_e32 v28, 26, v28
	v_and_b32_e32 v93, 60, v93
	v_lshrrev_b32_e32 v98, 10, v29
	v_and_b32_e32 v95, 60, v95
	v_lshrrev_b32_e32 v99, 22, v29
	v_and_b32_e32 v28, 60, v28
	ds_load_b32 v91, v91
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v97, 16, v31
	v_and_b32_e32 v90, 60, v90
	v_lshlrev_b32_e32 v100, 16, v32
	v_lshlrev_b32_e32 v96, 16, v30
	v_and_b32_e32 v32, 0xffff0000, v32
	v_and_b32_e32 v30, 0xffff0000, v30
	ds_load_b32 v90, v90
	v_and_b32_e32 v31, 0xffff0000, v31
	s_wait_dscnt 0x2
	v_fma_f32 v89, v89, v96, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v89, v90, v30 :: v_dual_and_b32 v92, 60, v92
	v_lshrrev_b32_e32 v90, 6, v29
	v_lshrrev_b32_e32 v30, 2, v29
	v_dual_fmac_f32 v89, v91, v97 :: v_dual_and_b32 v94, 60, v94
	ds_load_b32 v92, v92
	ds_load_b32 v93, v93
	ds_load_b32 v94, v94
	ds_load_b32 v95, v95
	ds_load_b32 v28, v28
	v_and_b32_e32 v90, 60, v90
	v_lshrrev_b32_e32 v91, 14, v29
	v_lshrrev_b32_e32 v97, 18, v29
	ds_load_b32 v90, v90
	v_and_b32_e32 v30, 60, v30
	ds_load_b32 v30, v30
	s_wait_dscnt 0x6
	v_dual_fmac_f32 v89, v92, v31 :: v_dual_and_b32 v92, 60, v98
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v89, v93, v100
	v_lshlrev_b32_e32 v93, 16, v33
	v_and_b32_e32 v33, 0xffff0000, v33
	s_wait_loadcnt_dscnt 0x4
	v_dual_fmac_f32 v89, v94, v32 :: v_dual_lshlrev_b32 v94, 16, v85
	v_and_b32_e32 v85, 0xffff0000, v85
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v89, v95, v93
	v_and_b32_e32 v93, 60, v99
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v89, v28, v33
	v_and_b32_e32 v96, 15, v29
	v_lshrrev_b32_e32 v29, 26, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v31, 2, v96
	v_and_b32_e32 v29, 60, v29
	ds_load_b32 v31, v31
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v89, v31, v94
	v_lshlrev_b32_e32 v31, 16, v86
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v89, v30, v85 :: v_dual_and_b32 v30, 0xffff0000, v86
	v_lshlrev_b32_e32 v85, 16, v87
	v_fmac_f32_e32 v89, v90, v31
	v_and_b32_e32 v32, 60, v91
	v_and_b32_e32 v91, 60, v97
	ds_load_b32 v28, v92
	ds_load_b32 v32, v32
	ds_load_b32 v33, v91
	ds_load_b32 v91, v93
	ds_load_b32 v29, v29
	v_and_b32_e32 v31, 7, v84
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v89, v28, v30 :: v_dual_and_b32 v28, 0x7f, v84
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v30, v31
	v_and_b32_e32 v31, 0xffff0000, v87
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v89, v32, v85
	v_lshl_add_u32 v32, v28, 20, 0x3c000000
	v_mul_f32_e32 v30, 0x3b000000, v30
	v_cmp_lt_u32_e64 s8, 7, v28
	v_lshlrev_b32_e32 v85, 16, v88
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v89, v33, v31
	v_lshlrev_b32_e32 v31, 24, v84
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v30, v30, v32, s8
	v_cmp_ne_u32_e64 s8, 0x7f, v28
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v89, v91, v85 :: v_dual_and_b32 v32, 0xffff0000, v88
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v28, 0x7fc00000, v30, s8
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v89, v29, v32
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v28, 0x80000000, v31, v28
	v_fmac_f32_e32 v56, v89, v28
	s_or_b32 exec_lo, exec_lo, s72
	s_and_saveexec_b32 s72, s3
	s_cbranch_execnz .LBB_180
.LBB_188:                              ;   in Loop: Header=BB8_161 Depth=1
	s_or_b32 exec_lo, exec_lo, s72
	s_and_saveexec_b32 s72, s4
	s_cbranch_execz .LBB_181
.LBB_189:                              ;   in Loop: Header=BB8_161 Depth=1
	s_wait_loadcnt 0x1
	s_wait_alu 0xfffe
	v_add_co_u32 v30, s8, s56, v78
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v31, null, s57, 0, s8
	s_wait_loadcnt_dscnt 0x0
	v_and_b32_e32 v83, 15, v24
	v_lshrrev_b32_e32 v84, 2, v24
	v_lshrrev_b32_e32 v85, 6, v24
	s_clause 0x1
	global_load_b128 v[26:29], v[30:31], off offset:1024
	global_load_b128 v[30:33], v[30:31], off offset:1040
	v_lshrrev_b32_e32 v88, 18, v24
	v_lshlrev_b32_e32 v83, 2, v83
	v_lshrrev_b32_e32 v86, 10, v24
	v_and_b32_e32 v85, 60, v85
	v_lshrrev_b32_e32 v87, 14, v24
	v_lshrrev_b32_e32 v89, 22, v24
	ds_load_b32 v83, v83
	v_lshrrev_b32_e32 v24, 26, v24
	v_lshrrev_b32_e32 v92, 10, v25
	v_and_b32_e32 v87, 60, v87
	v_and_b32_e32 v89, 60, v89
	v_lshrrev_b32_e32 v93, 22, v25
	ds_load_b32 v85, v85
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v91, 16, v27
	v_and_b32_e32 v84, 60, v84
	v_lshlrev_b32_e32 v90, 16, v26
	v_and_b32_e32 v26, 0xffff0000, v26
	v_lshlrev_b32_e32 v94, 16, v28
	v_and_b32_e32 v28, 0xffff0000, v28
	ds_load_b32 v84, v84
	s_wait_dscnt 0x2
	v_fma_f32 v83, v83, v90, 0
	v_and_b32_e32 v88, 60, v88
	v_and_b32_e32 v86, 60, v86
	v_and_b32_e32 v27, 0xffff0000, v27
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v83, v84, v26
	v_lshrrev_b32_e32 v84, 6, v25
	v_lshrrev_b32_e32 v26, 2, v25
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v83, v85, v91 :: v_dual_and_b32 v24, 60, v24
	ds_load_b32 v86, v86
	ds_load_b32 v87, v87
	ds_load_b32 v88, v88
	ds_load_b32 v89, v89
	ds_load_b32 v24, v24
	v_and_b32_e32 v84, 60, v84
	v_lshrrev_b32_e32 v85, 14, v25
	v_lshrrev_b32_e32 v91, 18, v25
	ds_load_b32 v84, v84
	s_wait_dscnt 0x5
	v_dual_fmac_f32 v83, v86, v27 :: v_dual_and_b32 v90, 15, v25
	v_lshrrev_b32_e32 v25, 26, v25
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v83, v87, v94
	v_lshlrev_b32_e32 v87, 16, v29
	v_lshlrev_b32_e32 v27, 2, v90
	v_and_b32_e32 v29, 0xffff0000, v29
	v_and_b32_e32 v25, 60, v25
	s_wait_loadcnt_dscnt 0x3
	v_dual_fmac_f32 v83, v88, v28 :: v_dual_lshlrev_b32 v88, 16, v30
	ds_load_b32 v27, v27
	v_and_b32_e32 v30, 0xffff0000, v30
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v83, v89, v87 :: v_dual_and_b32 v28, 60, v85
	v_and_b32_e32 v85, 60, v91
	v_and_b32_e32 v87, 60, v93
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v83, v24, v29 :: v_dual_and_b32 v26, 60, v26
	ds_load_b32 v26, v26
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v83, v27, v88
	v_lshlrev_b32_e32 v27, 16, v31
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v83, v26, v30 :: v_dual_lshlrev_b32 v30, 16, v32
	v_and_b32_e32 v86, 60, v92
	ds_load_b32 v24, v86
	ds_load_b32 v28, v28
	ds_load_b32 v29, v85
	ds_load_b32 v85, v87
	ds_load_b32 v25, v25
	v_and_b32_e32 v26, 0xffff0000, v31
	v_fmac_f32_e32 v83, v84, v27
	v_and_b32_e32 v27, 7, v82
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v83, v24, v26
	v_and_b32_e32 v24, 0x7f, v82
	v_cvt_f32_ubyte0_e32 v26, v27
	v_and_b32_e32 v27, 0xffff0000, v32
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v83, v28, v30
	v_lshl_add_u32 v28, v24, 20, 0x3c000000
	v_mul_f32_e32 v26, 0x3b000000, v26
	v_cmp_lt_u32_e64 s8, 7, v24
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v83, v29, v27 :: v_dual_lshlrev_b32 v30, 16, v33
	v_lshlrev_b32_e32 v27, 24, v82
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v26, v26, v28, s8
	v_cmp_ne_u32_e64 s8, 0x7f, v24
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v83, v85, v30 :: v_dual_and_b32 v28, 0xffff0000, v33
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v24, 0x7fc00000, v26, s8
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v83, v25, v28
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v24, 0x80000000, v27, v24
	v_fmac_f32_e32 v59, v83, v24
	s_or_b32 exec_lo, exec_lo, s72
	s_and_saveexec_b32 s72, s5
	s_cbranch_execnz .LBB_182
.LBB_190:                              ;   in Loop: Header=BB8_161 Depth=1
	s_or_b32 exec_lo, exec_lo, s72
	s_and_saveexec_b32 s72, s6
	s_cbranch_execz .LBB_183
.LBB_191:                              ;   in Loop: Header=BB8_161 Depth=1
	s_wait_loadcnt 0x1
	s_wait_alu 0xfffe
	v_add_co_u32 v26, s8, s48, v78
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v27, null, s49, 0, s8
	v_and_b32_e32 v30, 15, v20
	v_lshrrev_b32_e32 v31, 2, v20
	v_lshrrev_b32_e32 v32, 6, v20
	s_clause 0x1
	global_load_b128 v[22:25], v[26:27], off offset:1024
	global_load_b128 v[26:29], v[26:27], off offset:1040
	v_lshrrev_b32_e32 v33, 10, v20
	v_lshlrev_b32_e32 v30, 2, v30
	s_wait_loadcnt_dscnt 0x200
	v_lshrrev_b32_e32 v81, 14, v20
	v_and_b32_e32 v32, 60, v32
	v_lshrrev_b32_e32 v82, 18, v20
	v_lshrrev_b32_e32 v83, 22, v20
	ds_load_b32 v30, v30
	v_lshrrev_b32_e32 v20, 26, v20
	v_lshrrev_b32_e32 v86, 10, v21
	v_and_b32_e32 v82, 60, v82
	v_lshrrev_b32_e32 v87, 22, v21
	ds_load_b32 v32, v32
	v_and_b32_e32 v20, 60, v20
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v88, 16, v24
	v_and_b32_e32 v24, 0xffff0000, v24
	v_lshlrev_b32_e32 v85, 16, v23
	v_and_b32_e32 v31, 60, v31
	v_lshlrev_b32_e32 v84, 16, v22
	v_and_b32_e32 v22, 0xffff0000, v22
	v_and_b32_e32 v23, 0xffff0000, v23
	v_and_b32_e32 v83, 60, v83
	ds_load_b32 v31, v31
	s_wait_dscnt 0x2
	v_fma_f32 v30, v30, v84, 0
	v_and_b32_e32 v33, 60, v33
	v_and_b32_e32 v84, 15, v21
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v30, v31, v22
	v_lshrrev_b32_e32 v22, 2, v21
	v_lshrrev_b32_e32 v31, 6, v21
	v_fmac_f32_e32 v30, v32, v85
	v_and_b32_e32 v81, 60, v81
	ds_load_b32 v33, v33
	ds_load_b32 v81, v81
	ds_load_b32 v82, v82
	ds_load_b32 v83, v83
	ds_load_b32 v20, v20
	v_and_b32_e32 v22, 60, v22
	v_lshrrev_b32_e32 v32, 14, v21
	v_lshrrev_b32_e32 v85, 18, v21
	v_lshrrev_b32_e32 v21, 26, v21
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v30, v33, v23 :: v_dual_and_b32 v21, 60, v21
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v30, v81, v88
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v82, v24
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v82, 16, v26
	v_and_b32_e32 v26, 0xffff0000, v26
	v_lshlrev_b32_e32 v81, 16, v25
	v_and_b32_e32 v25, 0xffff0000, v25
	v_and_b32_e32 v24, 60, v32
	v_and_b32_e32 v32, 60, v85
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v30, v83, v81 :: v_dual_lshlrev_b32 v23, 2, v84
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v20, v25
	ds_load_b32 v23, v23
	v_and_b32_e32 v31, 60, v31
	ds_load_b32 v31, v31
	ds_load_b32 v22, v22
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v30, v23, v82 :: v_dual_lshlrev_b32 v23, 16, v27
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v22, v26
	v_and_b32_e32 v33, 60, v86
	v_lshlrev_b32_e32 v26, 16, v28
	v_and_b32_e32 v81, 60, v87
	v_and_b32_e32 v22, 0xffff0000, v27
	v_fmac_f32_e32 v30, v31, v23
	ds_load_b32 v20, v33
	ds_load_b32 v24, v24
	ds_load_b32 v25, v32
	ds_load_b32 v32, v81
	ds_load_b32 v21, v21
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v30, v20, v22 :: v_dual_and_b32 v23, 7, v80
	v_and_b32_e32 v20, 0x7f, v80
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v22, v23
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v30, v24, v26 :: v_dual_and_b32 v23, 0xffff0000, v28
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v24, v20, 20, 0x3c000000
	v_mul_f32_e32 v22, 0x3b000000, v22
	v_cmp_lt_u32_e64 s8, 7, v20
	v_lshlrev_b32_e32 v26, 16, v29
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v30, v25, v23 :: v_dual_lshlrev_b32 v23, 24, v80
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v22, v22, v24, s8
	v_cmp_ne_u32_e64 s8, 0x7f, v20
	v_and_b32_e32 v24, 0xffff0000, v29
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v30, v32, v26
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v20, 0x7fc00000, v22, s8
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v30, v21, v24
	v_and_or_b32 v20, 0x80000000, v23, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v61, v30, v20
	s_or_b32 exec_lo, exec_lo, s72
	s_and_saveexec_b32 s72, s7
	s_cbranch_execz .LBB_160
.LBB_192:                              ;   in Loop: Header=BB8_161 Depth=1
	s_wait_loadcnt 0x1
	s_wait_alu 0xfffe
	v_add_co_u32 v24, s8, s60, v78
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v25, null, s61, 0, s8
	v_and_b32_e32 v28, 15, v18
	v_lshrrev_b32_e32 v29, 2, v18
	v_lshrrev_b32_e32 v30, 6, v18
	s_clause 0x1
	global_load_b128 v[20:23], v[24:25], off offset:1024
	global_load_b128 v[24:27], v[24:25], off offset:1040
	v_lshrrev_b32_e32 v31, 10, v18
	v_lshlrev_b32_e32 v28, 2, v28
	v_lshrrev_b32_e32 v32, 14, v18
	v_and_b32_e32 v30, 60, v30
	v_lshrrev_b32_e32 v33, 18, v18
	s_wait_loadcnt_dscnt 0x200
	v_lshrrev_b32_e32 v80, 22, v18
	ds_load_b32 v28, v28
	v_lshrrev_b32_e32 v18, 26, v18
	v_and_b32_e32 v32, 60, v32
	v_and_b32_e32 v33, 60, v33
	v_and_b32_e32 v80, 60, v80
	v_lshrrev_b32_e32 v83, 10, v19
	v_and_b32_e32 v18, 60, v18
	v_lshrrev_b32_e32 v84, 22, v19
	ds_load_b32 v30, v30
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v82, 16, v21
	v_and_b32_e32 v29, 60, v29
	v_lshlrev_b32_e32 v85, 16, v22
	v_and_b32_e32 v22, 0xffff0000, v22
	v_and_b32_e32 v21, 0xffff0000, v21
	v_lshlrev_b32_e32 v81, 16, v20
	ds_load_b32 v29, v29
	v_and_b32_e32 v20, 0xffff0000, v20
	s_wait_dscnt 0x2
	v_fma_f32 v28, v28, v81, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v28, v29, v20 :: v_dual_and_b32 v31, 60, v31
	ds_load_b32 v31, v31
	ds_load_b32 v32, v32
	ds_load_b32 v33, v33
	ds_load_b32 v80, v80
	ds_load_b32 v18, v18
	v_lshrrev_b32_e32 v20, 2, v19
	v_lshrrev_b32_e32 v29, 6, v19
	v_fmac_f32_e32 v28, v30, v82
	v_lshrrev_b32_e32 v30, 14, v19
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v20, 60, v20
	v_and_b32_e32 v29, 60, v29
	v_lshrrev_b32_e32 v82, 18, v19
	ds_load_b32 v29, v29
	ds_load_b32 v20, v20
	s_wait_dscnt 0x6
	v_dual_fmac_f32 v28, v31, v21 :: v_dual_and_b32 v81, 15, v19
	v_lshrrev_b32_e32 v19, 26, v19
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v28, v32, v85
	v_lshlrev_b32_e32 v32, 16, v23
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v28, v33, v22 :: v_dual_and_b32 v23, 0xffff0000, v23
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v33, 16, v24
	v_and_b32_e32 v24, 0xffff0000, v24
	v_lshlrev_b32_e32 v21, 2, v81
	v_and_b32_e32 v22, 60, v30
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v28, v80, v32
	v_and_b32_e32 v30, 60, v82
	v_and_b32_e32 v32, 60, v84
	ds_load_b32 v21, v21
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v28, v18, v23
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v28, v21, v33 :: v_dual_and_b32 v31, 60, v83
	v_dual_fmac_f32 v28, v20, v24 :: v_dual_lshlrev_b32 v21, 16, v25
	v_lshlrev_b32_e32 v24, 16, v26
	v_and_b32_e32 v19, 60, v19
	ds_load_b32 v18, v31
	ds_load_b32 v22, v22
	ds_load_b32 v23, v30
	ds_load_b32 v30, v32
	ds_load_b32 v19, v19
	v_and_b32_e32 v20, 0xffff0000, v25
	v_fmac_f32_e32 v28, v29, v21
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v28, v18, v20
	v_and_b32_e32 v18, 0x7f, v79
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v28, v22, v24
	v_lshlrev_b32_e32 v24, 16, v27
	v_and_b32_e32 v21, 7, v79
	v_lshl_add_u32 v22, v18, 20, 0x3c000000
	v_cmp_lt_u32_e64 s8, 7, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v20, v21
	v_and_b32_e32 v21, 0xffff0000, v26
	v_mul_f32_e32 v20, 0x3b000000, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v28, v23, v21
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v20, v20, v22, s8
	v_cmp_ne_u32_e64 s8, 0x7f, v18
	v_and_b32_e32 v22, 0xffff0000, v27
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v28, v30, v24 :: v_dual_lshlrev_b32 v21, 24, v79
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v18, 0x7fc00000, v20, s8
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v28, v19, v22
	v_and_or_b32 v18, 0x80000000, v21, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v60, v28, v18
	s_branch .LBB_160
.LBB_193:
	s_or_b32 exec_lo, exec_lo, s71
.LBB_194:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s70
	v_cmp_ne_u64_e32 vcc_lo, 0, v[16:17]
	v_mov_b32_e32 v16, 0
	s_add_nc_u64 s[0:1], s[42:43], s[44:45]
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB_204
; %bb.195:
	v_mbcnt_lo_u32_b32 v16, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v17, 16, v16
	v_xor_b32_e32 v18, 8, v16
	v_xor_b32_e32 v19, 4, v16
	v_cmp_gt_i32_e32 vcc_lo, 32, v17
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v17, v16, v17, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v18
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v18, v16, v18, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v19
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v18, 2, v18
	v_lshlrev_b32_e32 v17, 2, v17
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v19, v16, v19, vcc_lo
	ds_bpermute_b32 v17, v17, v54
	v_lshlrev_b32_e32 v19, 2, v19
	s_wait_dscnt 0x0
	v_add_f32_e32 v17, v54, v17
	ds_bpermute_b32 v18, v18, v17
	s_wait_dscnt 0x0
	v_add_f32_e32 v17, v17, v18
	ds_bpermute_b32 v18, v19, v17
	v_xor_b32_e32 v19, 2, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v19
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v19, v16, v19, vcc_lo
	v_lshlrev_b32_e32 v19, 2, v19
	s_wait_dscnt 0x0
	v_add_f32_e32 v17, v17, v18
	ds_bpermute_b32 v18, v19, v17
	v_xor_b32_e32 v19, 1, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v19
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v16, v16, v19, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v17, v17, v18 :: v_dual_lshlrev_b32 v16, 2, v16
	ds_bpermute_b32 v16, v16, v17
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v17, s67, v16
	v_and_b32_e32 v16, 0x7fffffff, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v16
                                        ; implicit-def: $vgpr16
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s4, exec_lo, s4
; %bb.196:
	v_bfe_u32 v16, v17, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v16, v17, v16, 0x7fff
                                        ; implicit-def: $vgpr17
	v_lshrrev_b32_e32 v16, 16, v16
; %bb.197:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s4, s4
; %bb.198:
	v_lshrrev_b32_e32 v16, 16, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v16, 64, v16
; %bb.199:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	s_load_b32 s4, s[0:1], 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v16, 16, v16
	s_wait_kmcnt 0x0
	v_mul_f32_e32 v17, s4, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v16, 0x7fffffff, v17
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v16
                                        ; implicit-def: $vgpr16
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s4, exec_lo, s4
; %bb.200:
	v_bfe_u32 v16, v17, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v16, v17, v16, 0x7fff
                                        ; implicit-def: $vgpr17
	v_lshrrev_b32_e32 v16, 16, v16
; %bb.201:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s4, s4
; %bb.202:
	v_lshrrev_b32_e32 v16, 16, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v16, 64, v16
; %bb.203:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v16, 16, v16
	v_add_f32_e32 v16, 0, v16
.LBB_204:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	v_cmpx_ne_u64_e32 0, v[14:15]
	s_cbranch_execz .LBB_214
; %bb.205:
	v_mbcnt_lo_u32_b32 v14, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v15, 16, v14
	v_xor_b32_e32 v17, 8, v14
	v_xor_b32_e32 v18, 4, v14
	v_cmp_gt_i32_e32 vcc_lo, 32, v15
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v15, v14, v15, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v17
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v17, v14, v17, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v18
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v18, v14, v18 :: v_dual_lshlrev_b32 v17, 2, v17
	v_lshlrev_b32_e32 v18, 2, v18
	v_lshlrev_b32_e32 v15, 2, v15
	ds_bpermute_b32 v15, v15, v53
	s_wait_dscnt 0x0
	v_add_f32_e32 v15, v53, v15
	ds_bpermute_b32 v17, v17, v15
	s_wait_dscnt 0x0
	v_add_f32_e32 v15, v15, v17
	ds_bpermute_b32 v17, v18, v15
	v_xor_b32_e32 v18, 2, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v18
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v18, v14, v18, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v15, v15, v17 :: v_dual_lshlrev_b32 v18, 2, v18
	ds_bpermute_b32 v17, v18, v15
	v_xor_b32_e32 v18, 1, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v18
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v14, v18, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v15, v15, v17 :: v_dual_lshlrev_b32 v14, 2, v14
	ds_bpermute_b32 v14, v14, v15
	s_wait_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, s66, v14
	v_and_b32_e32 v14, 0x7fffffff, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s4, exec_lo, s4
; %bb.206:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
	v_lshrrev_b32_e32 v14, 16, v14
; %bb.207:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s4, s4
; %bb.208:
	v_lshrrev_b32_e32 v14, 16, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v14, 64, v14
; %bb.209:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	s_load_b32 s4, s[0:1], 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v14, 16, v14
	s_wait_kmcnt 0x0
	v_mul_f32_e32 v15, s4, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v14, 0x7fffffff, v15
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s4, exec_lo, s4
; %bb.210:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
	v_lshrrev_b32_e32 v14, 16, v14
; %bb.211:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s4, s4
; %bb.212:
	v_lshrrev_b32_e32 v14, 16, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v14, 64, v14
; %bb.213:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v14, 16, v14
	v_add_f32_e32 v16, v16, v14
.LBB_214:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	v_cmpx_ne_u64_e32 0, v[12:13]
	s_cbranch_execz .LBB_224
; %bb.215:
	v_mbcnt_lo_u32_b32 v12, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v13, 16, v12
	v_xor_b32_e32 v14, 8, v12
	v_xor_b32_e32 v15, 4, v12
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v12, v13, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v12, v14, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v15
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v15, v12, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v15, 2, v15
	v_lshlrev_b32_e32 v14, 2, v14
	v_lshlrev_b32_e32 v13, 2, v13
	ds_bpermute_b32 v13, v13, v56
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v56, v13
	ds_bpermute_b32 v14, v14, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v13, v14
	ds_bpermute_b32 v14, v15, v13
	v_xor_b32_e32 v15, 2, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v15
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v15, v12, v15, vcc_lo
	v_lshlrev_b32_e32 v15, 2, v15
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v13, v14
	ds_bpermute_b32 v14, v15, v13
	v_xor_b32_e32 v15, 1, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v15
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v12, v12, v15, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v13, v13, v14 :: v_dual_lshlrev_b32 v12, 2, v12
	ds_bpermute_b32 v12, v12, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v13, s65, v12
	v_and_b32_e32 v12, 0x7fffffff, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v12
                                        ; implicit-def: $vgpr12
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s4, exec_lo, s4
; %bb.216:
	v_bfe_u32 v12, v13, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v12, v13, v12, 0x7fff
                                        ; implicit-def: $vgpr13
	v_lshrrev_b32_e32 v12, 16, v12
; %bb.217:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s4, s4
; %bb.218:
	v_lshrrev_b32_e32 v12, 16, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v12, 64, v12
; %bb.219:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	s_load_b32 s4, s[0:1], 0x8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v12, 16, v12
	s_wait_kmcnt 0x0
	v_mul_f32_e32 v13, s4, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v12, 0x7fffffff, v13
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v12
                                        ; implicit-def: $vgpr12
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s4, exec_lo, s4
; %bb.220:
	v_bfe_u32 v12, v13, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v12, v13, v12, 0x7fff
                                        ; implicit-def: $vgpr13
	v_lshrrev_b32_e32 v12, 16, v12
; %bb.221:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s4, s4
; %bb.222:
	v_lshrrev_b32_e32 v12, 16, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v12, 64, v12
; %bb.223:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v12, 16, v12
	v_add_f32_e32 v16, v16, v12
.LBB_224:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	v_cmpx_ne_u64_e32 0, v[10:11]
	s_cbranch_execz .LBB_234
; %bb.225:
	v_mbcnt_lo_u32_b32 v10, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v11, 16, v10
	v_xor_b32_e32 v12, 8, v10
	v_xor_b32_e32 v13, 4, v10
	v_cmp_gt_i32_e32 vcc_lo, 32, v11
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v11, v10, v11, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v12
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v12, v10, v12, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v12, 2, v12
	v_lshlrev_b32_e32 v11, 2, v11
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v10, v13, vcc_lo
	ds_bpermute_b32 v11, v11, v55
	v_lshlrev_b32_e32 v13, 2, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v11, v55, v11
	ds_bpermute_b32 v12, v12, v11
	s_wait_dscnt 0x0
	v_add_f32_e32 v11, v11, v12
	ds_bpermute_b32 v12, v13, v11
	v_xor_b32_e32 v13, 2, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v10, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v11, v11, v12
	ds_bpermute_b32 v12, v13, v11
	v_xor_b32_e32 v13, 1, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v10, v10, v13, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v11, v11, v12 :: v_dual_lshlrev_b32 v10, 2, v10
	ds_bpermute_b32 v10, v10, v11
	s_wait_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v11, s64, v10
	v_and_b32_e32 v10, 0x7fffffff, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s4, exec_lo, s4
; %bb.226:
	v_bfe_u32 v10, v11, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v10, v11, v10, 0x7fff
                                        ; implicit-def: $vgpr11
	v_lshrrev_b32_e32 v10, 16, v10
; %bb.227:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s4, s4
; %bb.228:
	v_lshrrev_b32_e32 v10, 16, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v10, 64, v10
; %bb.229:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	s_load_b32 s4, s[0:1], 0xc
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v10, 16, v10
	s_wait_kmcnt 0x0
	v_mul_f32_e32 v11, s4, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v10, 0x7fffffff, v11
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s4, exec_lo, s4
; %bb.230:
	v_bfe_u32 v10, v11, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v10, v11, v10, 0x7fff
                                        ; implicit-def: $vgpr11
	v_lshrrev_b32_e32 v10, 16, v10
; %bb.231:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s4, s4
; %bb.232:
	v_lshrrev_b32_e32 v10, 16, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v10, 64, v10
; %bb.233:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v10, 16, v10
	v_add_f32_e32 v16, v16, v10
.LBB_234:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	v_cmpx_ne_u64_e32 0, v[8:9]
	s_cbranch_execz .LBB_244
; %bb.235:
	v_mbcnt_lo_u32_b32 v8, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v9, 16, v8
	v_xor_b32_e32 v10, 8, v8
	v_xor_b32_e32 v11, 4, v8
	v_cmp_gt_i32_e32 vcc_lo, 32, v9
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v9, v8, v9, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v10
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v10, v8, v10, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v10, 2, v10
	v_lshlrev_b32_e32 v9, 2, v9
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v11, v8, v11, vcc_lo
	ds_bpermute_b32 v9, v9, v59
	v_lshlrev_b32_e32 v11, 2, v11
	s_wait_dscnt 0x0
	v_add_f32_e32 v9, v59, v9
	ds_bpermute_b32 v10, v10, v9
	s_wait_dscnt 0x0
	v_add_f32_e32 v9, v9, v10
	ds_bpermute_b32 v10, v11, v9
	v_xor_b32_e32 v11, 2, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v11
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v11, v8, v11, vcc_lo
	v_lshlrev_b32_e32 v11, 2, v11
	s_wait_dscnt 0x0
	v_add_f32_e32 v9, v9, v10
	ds_bpermute_b32 v10, v11, v9
	v_xor_b32_e32 v11, 1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v11
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v8, v8, v11, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v9, v9, v10 :: v_dual_lshlrev_b32 v8, 2, v8
	ds_bpermute_b32 v8, v8, v9
	s_wait_dscnt 0x0
	v_add_f32_e32 v8, v9, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v9, s63, v8
	v_and_b32_e32 v8, 0x7fffffff, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v8
                                        ; implicit-def: $vgpr8
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s4, exec_lo, s4
; %bb.236:
	v_bfe_u32 v8, v9, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v8, v9, v8, 0x7fff
                                        ; implicit-def: $vgpr9
	v_lshrrev_b32_e32 v8, 16, v8
; %bb.237:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s4, s4
; %bb.238:
	v_lshrrev_b32_e32 v8, 16, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v8, 64, v8
; %bb.239:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	s_load_b32 s4, s[0:1], 0x10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v8, 16, v8
	s_wait_kmcnt 0x0
	v_mul_f32_e32 v9, s4, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v8, 0x7fffffff, v9
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v8
                                        ; implicit-def: $vgpr8
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s4, exec_lo, s4
; %bb.240:
	v_bfe_u32 v8, v9, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v8, v9, v8, 0x7fff
                                        ; implicit-def: $vgpr9
	v_lshrrev_b32_e32 v8, 16, v8
; %bb.241:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s4, s4
; %bb.242:
	v_lshrrev_b32_e32 v8, 16, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v8, 64, v8
; %bb.243:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v8, 16, v8
	v_add_f32_e32 v16, v16, v8
.LBB_244:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	v_cmpx_ne_u64_e32 0, v[6:7]
	s_cbranch_execz .LBB_254
; %bb.245:
	v_mbcnt_lo_u32_b32 v6, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v7, 16, v6
	v_xor_b32_e32 v8, 8, v6
	v_xor_b32_e32 v9, 4, v6
	v_cmp_gt_i32_e32 vcc_lo, 32, v7
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v7, v6, v7, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v8
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v8, v6, v8, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v9
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v8, 2, v8
	v_lshlrev_b32_e32 v7, 2, v7
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v9, v6, v9, vcc_lo
	ds_bpermute_b32 v7, v7, v57
	v_lshlrev_b32_e32 v9, 2, v9
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v57, v7
	ds_bpermute_b32 v8, v8, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v9, v7
	v_xor_b32_e32 v9, 2, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v9
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v9, v6, v9, vcc_lo
	v_lshlrev_b32_e32 v9, 2, v9
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v9, v7
	v_xor_b32_e32 v9, 1, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v9
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v6, v6, v9, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v7, v7, v8 :: v_dual_lshlrev_b32 v6, 2, v6
	ds_bpermute_b32 v6, v6, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v7, s62, v6
	v_and_b32_e32 v6, 0x7fffffff, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v6
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s4, exec_lo, s4
; %bb.246:
	v_bfe_u32 v6, v7, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v6, v7, v6, 0x7fff
                                        ; implicit-def: $vgpr7
	v_lshrrev_b32_e32 v6, 16, v6
; %bb.247:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s4, s4
; %bb.248:
	v_lshrrev_b32_e32 v6, 16, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v6, 64, v6
; %bb.249:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	s_load_b32 s4, s[0:1], 0x14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v6, 16, v6
	s_wait_kmcnt 0x0
	v_mul_f32_e32 v7, s4, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v6, 0x7fffffff, v7
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v6
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s4, exec_lo, s4
; %bb.250:
	v_bfe_u32 v6, v7, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v6, v7, v6, 0x7fff
                                        ; implicit-def: $vgpr7
	v_lshrrev_b32_e32 v6, 16, v6
; %bb.251:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s4, s4
; %bb.252:
	v_lshrrev_b32_e32 v6, 16, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v6, 64, v6
; %bb.253:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v6, 16, v6
	v_add_f32_e32 v16, v16, v6
.LBB_254:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	v_cmpx_ne_u64_e32 0, v[4:5]
	s_cbranch_execz .LBB_264
; %bb.255:
	v_mbcnt_lo_u32_b32 v4, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v5, 16, v4
	v_xor_b32_e32 v6, 8, v4
	v_xor_b32_e32 v7, 4, v4
	v_cmp_gt_i32_e32 vcc_lo, 32, v5
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v5, v4, v5, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v6
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v6, v4, v6, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v7
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v6, 2, v6
	v_lshlrev_b32_e32 v5, 2, v5
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v7, v4, v7, vcc_lo
	ds_bpermute_b32 v5, v5, v61
	v_lshlrev_b32_e32 v7, 2, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v61, v5
	ds_bpermute_b32 v6, v6, v5
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	ds_bpermute_b32 v6, v7, v5
	v_xor_b32_e32 v7, 2, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v7
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v7, v4, v7, vcc_lo
	v_lshlrev_b32_e32 v7, 2, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	ds_bpermute_b32 v6, v7, v5
	v_xor_b32_e32 v7, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v7
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v4, v4, v7, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v5, v5, v6 :: v_dual_lshlrev_b32 v4, 2, v4
	ds_bpermute_b32 v4, v4, v5
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v5, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v5, s33, v4
	v_and_b32_e32 v4, 0x7fffffff, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v4
                                        ; implicit-def: $vgpr4
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s4, exec_lo, s4
; %bb.256:
	v_bfe_u32 v4, v5, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v4, v5, v4, 0x7fff
                                        ; implicit-def: $vgpr5
	v_lshrrev_b32_e32 v4, 16, v4
; %bb.257:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s4, s4
; %bb.258:
	v_lshrrev_b32_e32 v4, 16, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v4, 64, v4
; %bb.259:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	s_load_b32 s4, s[0:1], 0x18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v4, 16, v4
	s_wait_kmcnt 0x0
	v_mul_f32_e32 v5, s4, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v4, 0x7fffffff, v5
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v4
                                        ; implicit-def: $vgpr4
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s4, exec_lo, s4
; %bb.260:
	v_bfe_u32 v4, v5, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v4, v5, v4, 0x7fff
                                        ; implicit-def: $vgpr5
	v_lshrrev_b32_e32 v4, 16, v4
; %bb.261:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s4, s4
; %bb.262:
	v_lshrrev_b32_e32 v4, 16, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v4, 64, v4
; %bb.263:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v4, 16, v4
	v_add_f32_e32 v16, v16, v4
.LBB_264:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	v_cmpx_ne_u64_e32 0, v[2:3]
	s_cbranch_execz .LBB_274
; %bb.265:
	v_mbcnt_lo_u32_b32 v2, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v3, 16, v2
	v_xor_b32_e32 v4, 8, v2
	v_xor_b32_e32 v5, 4, v2
	v_cmp_gt_i32_e32 vcc_lo, 32, v3
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v2, v3, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v4, v2, v4, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v5
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v4, 2, v4
	v_lshlrev_b32_e32 v3, 2, v3
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v5, v2, v5, vcc_lo
	ds_bpermute_b32 v3, v3, v60
	v_lshlrev_b32_e32 v5, 2, v5
	s_wait_dscnt 0x0
	v_add_f32_e32 v3, v60, v3
	ds_bpermute_b32 v4, v4, v3
	s_wait_dscnt 0x0
	v_add_f32_e32 v3, v3, v4
	ds_bpermute_b32 v4, v5, v3
	v_xor_b32_e32 v5, 2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v5
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v5, v2, v5, vcc_lo
	v_lshlrev_b32_e32 v5, 2, v5
	s_wait_dscnt 0x0
	v_add_f32_e32 v3, v3, v4
	ds_bpermute_b32 v4, v5, v3
	v_xor_b32_e32 v5, 1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v5
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v2, v2, v5, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v3, v4 :: v_dual_lshlrev_b32 v2, 2, v2
	ds_bpermute_b32 v2, v2, v3
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v3, v2
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v3, s9, v2
	v_and_b32_e32 v2, 0x7fffffff, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s4, exec_lo, s4
; %bb.266:
	v_bfe_u32 v2, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v2, v3, v2, 0x7fff
                                        ; implicit-def: $vgpr3
	v_lshrrev_b32_e32 v2, 16, v2
; %bb.267:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s4, s4
; %bb.268:
	v_lshrrev_b32_e32 v2, 16, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, 64, v2
; %bb.269:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	s_load_b32 s0, s[0:1], 0x1c
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 16, v2
	s_wait_kmcnt 0x0
	v_mul_f32_e32 v3, s0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0x7fffffff, v3
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.270:
	v_bfe_u32 v2, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v2, v3, v2, 0x7fff
                                        ; implicit-def: $vgpr3
	v_lshrrev_b32_e32 v2, 16, v2
; %bb.271:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.272:
	v_lshrrev_b32_e32 v2, 16, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, 64, v2
; %bb.273:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 16, v2
	v_add_f32_e32 v16, v16, v2
.LBB_274:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	v_cmp_eq_u32_e32 vcc_lo, 0, v58
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB_280
; %bb.275:
	v_and_b32_e32 v2, 0x7fffffff, v16
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e32 0x7f800001, v2
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.276:
	v_bfe_u32 v2, v16, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v2, v16, v2, 0x7fff
                                        ; implicit-def: $vgpr16
	v_lshrrev_b32_e32 v3, 16, v2
; %bb.277:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.278:
	v_lshrrev_b32_e32 v2, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v3, 64, v2
; %bb.279:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_ashrrev_i32_e32 v2, 31, v1
	s_mul_u64 s[0:1], s[40:41], s[36:37]
	s_wait_alu 0xfffe
	s_lshl_b64 s[0:1], s[0:1], 1
	s_wait_alu 0xfffe
	s_add_nc_u64 s[0:1], s[38:39], s[0:1]
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s0, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s1, v2, vcc_lo
	global_store_b16 v[1:2], v3, off
.LBB_280:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s69
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB_285
; %bb.281:
	s_mov_b32 s3, exec_lo
	s_mov_b32 s1, exec_lo
	s_wait_alu 0xfffe
	v_mbcnt_lo_u32_b32 v1, s3, 0
                                        ; implicit-def: $vgpr2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v1
	s_cbranch_execz .LBB_283
; %bb.282:
	s_bcnt1_i32_b32 s3, s3
	s_wait_alu 0xfffe
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, s3
	s_wait_loadcnt 0x0
	global_atomic_add_u32 v2, v2, v3, s[28:29] offset:4 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	global_inv scope:SCOPE_DEV
.LBB_283:
	s_or_b32 exec_lo, exec_lo, s1
	v_readfirstlane_b32 s1, v2
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s1, v1
	s_add_co_i32 s1, s21, -1
	s_wait_alu 0xfffe
	v_cmp_eq_u32_e32 vcc_lo, s1, v1
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB_285
; %bb.284:
	v_mov_b32_e32 v1, 0
	s_clause 0x2
	global_store_b32 v1, v1, s[28:29] scope:SCOPE_DEV
	global_store_b32 v1, v1, s[28:29] offset:4 scope:SCOPE_DEV
	global_store_b32 v1, v1, s[28:29] offset:8 scope:SCOPE_DEV
.LBB_285:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB_286:
	s_wait_alu 0xfffe
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB_351
; %bb.287:
	s_and_b32 s1, s23, 0xffff
	s_bitcmp0_b32 s23, 16
	s_mov_b32 s3, 0
	s_cbranch_scc1 .LBB_289
; %bb.288:
	s_ashr_i32 s0, s22, 31
	s_wait_alu 0xfffe
	s_lshr_b32 s0, s0, 30
	s_wait_alu 0xfffe
	s_add_co_i32 s0, s22, s0
	s_wait_alu 0xfffe
	s_and_b32 s0, s0, -4
	s_wait_alu 0xfffe
	s_sub_co_i32 s0, s22, s0
	s_wait_alu 0xfffe
	s_add_co_i32 s0, s0, 1
	s_wait_alu 0xfffe
	s_mul_i32 s0, s0, s1
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB_290
	s_branch .LBB_291
.LBB_289:
                                        ; implicit-def: $sgpr0
.LBB_290:
	s_add_co_i32 s0, s20, -1
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s22, s0
	s_cselect_b32 s0, s1, 0
.LBB_291:
	s_wait_alu 0xfffe
	s_cmp_lt_i32 s0, 1
	s_cbranch_scc1 .LBB_293
.LBB_292:                              ; =>This Inner Loop Header: Depth=1
	s_add_co_i32 s0, s0, -1
	s_sleep 0x7f
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s0, 0
	s_cbranch_scc0 .LBB_292
.LBB_293:
	s_add_co_i32 s0, s18, 7
	s_abs_i32 s4, s22
	s_wait_alu 0xfffe
	s_ashr_i32 s1, s0, 31
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v56, 0
	s_wait_alu 0xfffe
	s_lshr_b32 s1, s1, 29
	s_wait_kmcnt 0x0
	s_mov_b32 s9, 0
	s_wait_alu 0xfffe
	s_add_co_i32 s0, s0, s1
	v_mov_b32_e32 v50, 0
	s_wait_alu 0xfffe
	s_ashr_i32 s7, s0, 3
	s_wait_alu 0xfffe
	s_abs_i32 s0, s7
	s_wait_alu 0xfffe
	s_cvt_f32_u32 s1, s0
	s_sub_co_i32 s3, 0, s0
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s1
	v_readfirstlane_b32 s1, v1
	s_mul_f32 s1, s1, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s1, s1
	s_wait_alu 0xfffe
	s_mul_i32 s3, s3, s1
	s_wait_alu 0xfffe
	s_mul_hi_u32 s3, s1, s3
	s_wait_alu 0xfffe
	s_add_co_i32 s1, s1, s3
	s_xor_b32 s3, s22, s7
	s_wait_alu 0xfffe
	s_mul_hi_u32 s1, s4, s1
	s_ashr_i32 s3, s3, 31
	s_wait_alu 0xfffe
	s_mul_i32 s5, s1, s0
	s_wait_alu 0xfffe
	s_sub_co_i32 s4, s4, s5
	s_add_co_i32 s5, s1, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s6, s4, s0
	s_cmp_ge_u32 s4, s0
	s_cselect_b32 s1, s5, s1
	s_wait_alu 0xfffe
	s_cselect_b32 s4, s6, s4
	s_add_co_i32 s5, s1, 1
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s4, s0
	s_mov_b32 s6, s18
	s_cselect_b32 s0, s5, s1
	s_wait_alu 0xfffe
	s_xor_b32 s0, s0, s3
	s_wait_alu 0xfffe
	s_sub_co_i32 s4, s0, s3
	s_wait_alu 0xfffe
	s_ashr_i32 s5, s4, 31
	s_wait_alu 0xfffe
	s_lshl_b64 s[0:1], s[4:5], 2
	s_wait_alu 0xfffe
	s_add_nc_u64 s[0:1], s[14:15], s[0:1]
	s_load_b32 s3, s[0:1], 0x0
	s_mul_i32 s1, s4, s7
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s3, -1
	s_cselect_b32 s0, -1, 0
	s_cmp_lt_i32 s3, s16
	s_cselect_b32 s7, -1, 0
	s_wait_alu 0xfffe
	s_sub_co_i32 s1, s22, s1
	s_and_b32 s15, s0, s7
	s_wait_alu 0xfffe
	v_lshl_or_b32 v33, s1, 3, v52
	s_ashr_i32 s8, s3, 31
	s_lshl_b32 s0, s3, 1
	s_and_b32 s1, s15, exec_lo
	s_wait_alu 0xfffe
	s_cselect_b32 s1, s8, 0
	v_cmp_gt_i32_e32 vcc_lo, s18, v33
	s_cselect_b32 s0, s0, 0
	s_abs_i32 s16, s19
	s_ashr_i32 s7, s18, 31
	s_wait_alu 0xfffe
	s_cvt_f32_u32 s8, s16
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v1, 0, v33, vcc_lo
	s_lshr_b32 s10, s17, 31
	s_mul_i32 s1, s1, s18
	s_wait_alu 0xfffe
	v_rcp_iflag_f32_e32 v3, s8
	s_add_co_i32 s8, s17, s10
	v_ashrrev_i32_e32 v2, 31, v1
	s_ashr_i32 s37, s17, 31
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(TRANS32_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, s0, s18, v[1:2]
	s_mul_i32 s0, s0, s7
	s_wait_alu 0xfffe
	s_ashr_i32 s18, s8, 1
	v_readfirstlane_b32 s10, v3
	s_wait_alu 0xfffe
	s_ashr_i32 s20, s18, 31
	s_lshr_b32 s8, s37, 28
	v_add3_u32 v2, s0, s1, v2
	s_wait_alu 0xfffe
	v_mul_lo_u32 v3, v1, s20
	v_mad_co_u64_u32 v[34:35], null, v1, s18, 0
	s_mul_f32 s0, s10, 0x4f7ffffe
	v_mul_lo_u32 v4, v2, s18
	s_add_co_i32 s1, s17, s8
	s_abs_i32 s8, s4
	s_wait_alu 0xfffe
	s_cvt_u32_f32 s10, s0
	s_ashr_i32 s14, s1, 4
	s_sub_co_i32 s1, 0, s16
	v_cmp_gt_i32_e64 s0, s14, v58
	s_wait_alu 0xfffe
	s_mul_i32 s1, s1, s10
	v_add3_u32 v35, v35, v3, v4
	s_wait_alu 0xfffe
	s_mul_hi_u32 s11, s10, s1
	v_add_co_u32 v42, s1, s24, v34
	s_wait_alu 0xfffe
	s_add_co_i32 s10, s10, s11
	v_add_co_ci_u32_e64 v43, null, s25, v35, s1
	s_mov_b32 s11, s9
	s_and_saveexec_b32 s17, s0
	s_cbranch_execz .LBB_295
; %bb.294:
	v_lshlrev_b32_e32 v3, 3, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v3, s1, v42, v3
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v4, null, 0, v43, s1
	global_load_b64 v[56:57], v[3:4], off th:TH_LOAD_NT
.LBB_295:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s17
	v_add_co_u32 v3, s1, v1, s6
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v4, null, s7, v2, s1
	s_ashr_i32 s1, s14, 31
	v_mul_lo_u32 v6, v3, s20
	v_mad_co_u64_u32 v[40:41], null, v3, s18, 0
	v_mul_lo_u32 v5, v4, s18
	v_mul_lo_u32 v2, v2, s14
	s_wait_alu 0xfffe
	v_mul_lo_u32 v7, v1, s1
	v_mad_co_u64_u32 v[36:37], null, v1, s14, 0
	v_mul_lo_u32 v1, v3, s1
	v_mul_lo_u32 v4, v4, s14
	v_mad_co_u64_u32 v[38:39], null, v3, s14, 0
	v_add3_u32 v41, v41, v6, v5
	v_add_co_u32 v5, s1, s24, v40
	v_add3_u32 v37, v37, v7, v2
	v_mov_b32_e32 v54, 0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v8, null, s25, v41, s1
	v_add3_u32 v39, v39, v1, v4
	v_add_co_u32 v3, s1, s26, v36
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v6, null, s27, v37, s1
	v_add_co_u32 v4, s1, s26, v38
	v_mov_b32_e32 v1, 0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v7, null, s27, v39, s1
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v68, 0
	s_mul_u64 s[10:11], s[8:9], s[10:11]
	s_ashr_i32 s1, s19, 31
	s_and_saveexec_b32 s9, s0
	s_cbranch_execz .LBB_297
; %bb.296:
	v_lshlrev_b32_e32 v1, 3, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v1, s0, v5, v1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v2, null, 0, v8, s0
	v_add_co_u32 v9, s0, v3, v58
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v10, null, 0, v6, s0
	v_add_co_u32 v11, s0, v4, v58
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v12, null, 0, v7, s0
	global_load_b64 v[54:55], v[1:2], off th:TH_LOAD_NT
	s_clause 0x1
	global_load_u8 v68, v[9:10], off
	global_load_u8 v69, v[11:12], off
	v_lshlrev_b32_e32 v1, 4, v58
.LBB_297:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s9
	s_mul_i32 s0, s11, s16
	s_xor_b32 s1, s5, s1
	s_wait_alu 0xfffe
	s_sub_co_i32 s0, s8, s0
	s_add_co_i32 s8, s11, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s9, s0, s16
	s_cmp_ge_u32 s0, s16
	v_dual_mov_b32 v52, 0 :: v_dual_lshlrev_b32 v1, 1, v1
	s_cselect_b32 s8, s8, s11
	s_wait_alu 0xfffe
	s_cselect_b32 s0, s9, s0
	s_add_co_i32 s9, s8, 1
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s0, s16
	v_or_b32_e32 v9, 32, v58
	s_cselect_b32 s0, s9, s8
	v_mov_b32_e32 v51, 0
	s_wait_alu 0xfffe
	s_xor_b32 s0, s0, s1
	s_mov_b32 s10, exec_lo
	s_wait_alu 0xfffe
	s_sub_co_i32 s0, s0, s1
	s_wait_alu 0xfffe
	s_ashr_i32 s1, s0, 31
	s_wait_alu 0xfffe
	s_mul_u64 s[0:1], s[0:1], s[36:37]
	s_wait_alu 0xfffe
	s_lshl_b64 s[0:1], s[0:1], 1
	s_wait_alu 0xfffe
	s_add_nc_u64 s[8:9], s[34:35], s[0:1]
	s_clause 0x1
	global_load_b128 v[25:28], v1, s[8:9] offset:16
	global_load_b128 v[29:32], v1, s[8:9]
	v_cmp_le_i32_e64 s0, s14, v9
	v_cmpx_gt_i32_e64 s14, v9
	s_cbranch_execz .LBB_299
; %bb.298:
	v_lshlrev_b32_e32 v1, 3, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v1, s1, v42, v1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v2, null, 0, v43, s1
	global_load_b64 v[50:51], v[1:2], off offset:256 th:TH_LOAD_NT
.LBB_299:
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s1, s0
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s1
                                        ; implicit-def: $vgpr63
                                        ; implicit-def: $vgpr66
                                        ; implicit-def: $vgpr60
; %bb.300:
	s_mov_b32 s1, 0
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr63
                                        ; implicit-def: $vgpr60
	s_wait_alu 0xfffe
	v_mov_b32_e32 v66, s1
; %bb.301:
	s_or_saveexec_b32 s1, s0
	v_mov_b32_e32 v1, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v53, 0
	v_mov_b32_e32 v67, 0
	s_wait_alu 0xfffe
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB_303
; %bb.302:
	v_lshlrev_b32_e32 v1, 3, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v1, s0, v5, v1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v2, null, 0, v8, s0
	v_add_co_u32 v10, s0, v3, v58
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v11, null, 0, v6, s0
	v_add_co_u32 v12, s0, v4, v58
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v13, null, 0, v7, s0
	global_load_b64 v[52:53], v[1:2], off offset:256 th:TH_LOAD_NT
	s_clause 0x1
	global_load_u8 v66, v[10:11], off offset:32
	global_load_u8 v67, v[12:13], off offset:32
	v_lshlrev_b32_e32 v1, 4, v9
.LBB_303:
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v48, 0 :: v_dual_lshlrev_b32 v1, 1, v1
	v_or_b32_e32 v9, 64, v58
	v_mov_b32_e32 v49, 0
	s_mov_b32 s10, exec_lo
	s_clause 0x1
	global_load_b128 v[17:20], v1, s[8:9] offset:16
	global_load_b128 v[21:24], v1, s[8:9]
	v_cmp_le_i32_e64 s0, s14, v9
	v_cmpx_gt_i32_e64 s14, v9
	s_cbranch_execz .LBB_305
; %bb.304:
	v_lshlrev_b32_e32 v1, 3, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v1, s1, v42, v1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v2, null, 0, v43, s1
	global_load_b64 v[48:49], v[1:2], off offset:512 th:TH_LOAD_NT
.LBB_305:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s1, s0
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s1
; %bb.306:
	s_mov_b32 s1, 0
                                        ; implicit-def: $vgpr9
	s_wait_alu 0xfffe
	v_mov_b32_e32 v63, s1
; %bb.307:
	s_or_saveexec_b32 s1, s0
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v64, 0
	s_wait_alu 0xfffe
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB_309
; %bb.308:
	v_lshlrev_b32_e32 v1, 3, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v1, s0, v5, v1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v2, null, 0, v8, s0
	v_add_co_u32 v10, s0, v3, v58
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v11, null, 0, v6, s0
	v_add_co_u32 v12, s0, v4, v58
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v13, null, 0, v7, s0
	global_load_b64 v[46:47], v[1:2], off offset:512 th:TH_LOAD_NT
	s_clause 0x1
	global_load_u8 v63, v[10:11], off offset:64
	global_load_u8 v64, v[12:13], off offset:64
	v_lshlrev_b32_e32 v1, 4, v9
.LBB_309:
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 1, v1
	v_or_b32_e32 v59, 0x60, v58
	v_mov_b32_e32 v45, 0
	s_mov_b32 s10, exec_lo
	s_clause 0x1
	global_load_b128 v[9:12], v1, s[8:9] offset:16
	global_load_b128 v[13:16], v1, s[8:9]
	v_cmp_le_i32_e64 s0, s14, v59
	v_cmpx_gt_i32_e64 s14, v59
	s_cbranch_execz .LBB_311
; %bb.310:
	v_lshlrev_b32_e32 v1, 3, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v1, s1, v42, v1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v2, null, 0, v43, s1
	global_load_b64 v[44:45], v[1:2], off offset:768 th:TH_LOAD_NT
.LBB_311:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s1, s0
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s1
; %bb.312:
	s_mov_b32 s1, 0
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr3
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr4
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr59
	s_wait_alu 0xfffe
	v_mov_b32_e32 v60, s1
; %bb.313:
	s_or_saveexec_b32 s1, s0
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v43, 0
	v_mov_b32_e32 v62, 0
	s_wait_alu 0xfffe
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB_315
; %bb.314:
	v_lshlrev_b32_e32 v1, 3, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v1, s0, v5, v1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v2, null, 0, v8, s0
	v_add_co_u32 v5, s0, v3, v58
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v6, null, 0, v6, s0
	v_add_co_u32 v3, s0, v4, v58
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v4, null, 0, v7, s0
	global_load_b64 v[42:43], v[1:2], off offset:768 th:TH_LOAD_NT
	s_clause 0x1
	global_load_u8 v60, v[5:6], off offset:96
	global_load_u8 v62, v[3:4], off offset:96
	v_lshlrev_b32_e32 v1, 4, v59
.LBB_315:
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v5, 1, v1
	s_mov_b32 s1, exec_lo
	s_clause 0x1
	global_load_b128 v[1:4], v5, s[8:9] offset:16
	global_load_b128 v[5:8], v5, s[8:9]
	v_cmpx_gt_u32_e32 16, v0
	s_cbranch_execz .LBB_317
; %bb.316:
	v_and_b32_e32 v59, 7, v0
	v_lshlrev_b32_e32 v65, 28, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v61, v59, 22, 0.5
	v_cmp_ne_u32_e64 s0, 1, v59
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v61, 0.5, v61, s0
	v_cmp_ne_u32_e64 s0, 0, v59
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v59, 0, v61, s0
	v_lshlrev_b32_e32 v61, 2, v0
	v_and_or_b32 v59, 0x80000000, v65, v59
	ds_store_b32 v61, v59
.LBB_317:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_and_b32 s0, s15, vcc_lo
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s10, s0
	s_cbranch_execz .LBB_348
; %bb.318:
	v_sub_nc_u32_e32 v65, s14, v58
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v59, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_lt_i32_e32 0, v65
	s_cbranch_execz .LBB_320
; %bb.319:
	v_lshlrev_b32_e32 v90, 16, v29
	v_and_b32_e32 v29, 0xffff0000, v29
	v_and_b32_e32 v59, 15, v56
	v_lshrrev_b32_e32 v61, 2, v56
	v_lshrrev_b32_e32 v70, 6, v56
	v_lshrrev_b32_e32 v71, 10, v56
	v_lshrrev_b32_e32 v72, 14, v56
	v_lshrrev_b32_e32 v73, 18, v56
	v_lshrrev_b32_e32 v74, 22, v56
	v_lshrrev_b32_e32 v56, 26, v56
	v_lshlrev_b32_e32 v92, 16, v30
	v_lshlrev_b32_e32 v59, 2, v59
	v_and_b32_e32 v30, 0xffff0000, v30
	v_and_b32_e32 v61, 60, v61
	v_and_b32_e32 v70, 60, v70
	v_and_b32_e32 v71, 60, v71
	v_and_b32_e32 v72, 60, v72
	v_and_b32_e32 v73, 60, v73
	v_and_b32_e32 v74, 60, v74
	v_and_b32_e32 v56, 60, v56
	ds_load_b32 v59, v59
	ds_load_b32 v61, v61
	ds_load_b32 v70, v70
	ds_load_b32 v71, v71
	ds_load_b32 v72, v72
	ds_load_b32 v73, v73
	ds_load_b32 v74, v74
	ds_load_b32 v56, v56
	v_and_b32_e32 v75, 15, v57
	v_lshrrev_b32_e32 v76, 2, v57
	v_lshrrev_b32_e32 v77, 6, v57
	v_lshrrev_b32_e32 v78, 10, v57
	v_lshrrev_b32_e32 v79, 14, v57
	v_lshrrev_b32_e32 v80, 18, v57
	v_lshrrev_b32_e32 v81, 22, v57
	v_lshrrev_b32_e32 v57, 26, v57
	v_and_b32_e32 v82, 15, v54
	v_lshrrev_b32_e32 v83, 2, v54
	v_lshlrev_b32_e32 v75, 2, v75
	v_and_b32_e32 v77, 60, v77
	s_wait_dscnt 0x7
	v_fma_f32 v59, v59, v90, 0
	v_and_b32_e32 v79, 60, v79
	v_and_b32_e32 v80, 60, v80
	v_and_b32_e32 v81, 60, v81
	v_and_b32_e32 v57, 60, v57
	s_wait_dscnt 0x6
	v_dual_fmac_f32 v59, v61, v29 :: v_dual_lshlrev_b32 v82, 2, v82
	v_and_b32_e32 v83, 60, v83
	v_lshrrev_b32_e32 v84, 6, v54
	v_lshrrev_b32_e32 v85, 10, v54
	s_wait_dscnt 0x5
	v_dual_fmac_f32 v59, v70, v92 :: v_dual_lshlrev_b32 v70, 16, v31
	v_and_b32_e32 v31, 0xffff0000, v31
	v_lshrrev_b32_e32 v86, 14, v54
	v_lshrrev_b32_e32 v87, 18, v54
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v59, v71, v30
	v_lshrrev_b32_e32 v88, 22, v54
	v_lshrrev_b32_e32 v54, 26, v54
	v_and_b32_e32 v85, 60, v85
	v_and_b32_e32 v87, 60, v87
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v59, v72, v70 :: v_dual_and_b32 v76, 60, v76
	v_and_b32_e32 v89, 15, v55
	v_lshrrev_b32_e32 v91, 2, v55
	v_lshrrev_b32_e32 v61, 6, v55
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v59, v73, v31 :: v_dual_and_b32 v78, 60, v78
	ds_load_b32 v75, v75
	ds_load_b32 v76, v76
	ds_load_b32 v77, v77
	ds_load_b32 v78, v78
	ds_load_b32 v79, v79
	ds_load_b32 v80, v80
	ds_load_b32 v81, v81
	ds_load_b32 v57, v57
	ds_load_b32 v82, v82
	ds_load_b32 v83, v83
	v_and_b32_e32 v84, 60, v84
	v_and_b32_e32 v86, 60, v86
	v_and_b32_e32 v88, 60, v88
	v_and_b32_e32 v54, 60, v54
	ds_load_b32 v84, v84
	ds_load_b32 v85, v85
	ds_load_b32 v86, v86
	ds_load_b32 v87, v87
	ds_load_b32 v88, v88
	ds_load_b32 v54, v54
	v_lshlrev_b32_e32 v89, 2, v89
	v_lshrrev_b32_e32 v72, 18, v55
	s_wait_dscnt 0x7
	v_fma_f32 v82, v82, v90, 0
	v_and_b32_e32 v90, 60, v91
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v82, v83, v29
	v_and_b32_e32 v29, 60, v61
	ds_load_b32 v71, v89
	ds_load_b32 v83, v90
	ds_load_b32 v29, v29
	s_wait_dscnt 0x8
	v_fmac_f32_e32 v82, v84, v92
	v_lshlrev_b32_e32 v84, 16, v32
	v_lshrrev_b32_e32 v61, 10, v55
	v_and_b32_e32 v32, 0xffff0000, v32
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v82, v85, v30
	v_lshrrev_b32_e32 v30, 14, v55
	v_fmac_f32_e32 v59, v74, v84
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v82, v86, v70 :: v_dual_and_b32 v61, 60, v61
	v_and_b32_e32 v30, 60, v30
	v_and_b32_e32 v70, 60, v72
	v_dual_fmac_f32 v59, v56, v32 :: v_dual_lshlrev_b32 v72, 16, v25
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v82, v87, v31
	v_lshrrev_b32_e32 v31, 22, v55
	v_lshrrev_b32_e32 v55, 26, v55
	v_and_b32_e32 v25, 0xffff0000, v25
	v_fmac_f32_e32 v59, v75, v72
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v82, v88, v84 :: v_dual_and_b32 v31, 60, v31
	v_and_b32_e32 v55, 60, v55
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v59, v76, v25 :: v_dual_lshlrev_b32 v56, 16, v26
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v82, v54, v32
	ds_load_b32 v32, v61
	ds_load_b32 v30, v30
	ds_load_b32 v54, v70
	ds_load_b32 v31, v31
	ds_load_b32 v55, v55
	v_fmac_f32_e32 v59, v77, v56
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v82, v71, v72
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v82, v83, v25 :: v_dual_and_b32 v25, 0xffff0000, v26
	v_lshlrev_b32_e32 v26, 16, v27
	s_wait_dscnt 0x5
	v_dual_fmac_f32 v82, v29, v56 :: v_dual_and_b32 v27, 0xffff0000, v27
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v59, v78, v25
	v_and_b32_e32 v29, 7, v69
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v82, v32, v25 :: v_dual_fmac_f32 v59, v79, v26
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_and_b32_e32 v32, 0x7f, v68
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v82, v30, v26 :: v_dual_lshlrev_b32 v25, 16, v28
	v_and_b32_e32 v26, 7, v68
	v_dual_fmac_f32 v59, v80, v27 :: v_dual_and_b32 v30, 0x7f, v69
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v29, 0x3b000000, v29 :: v_dual_fmac_f32 v82, v54, v27
	v_cvt_f32_ubyte0_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v27, v30, 20, 0x3c000000
	v_cmp_lt_u32_e32 vcc_lo, 7, v30
	v_lshl_add_u32 v54, v32, 20, 0x3c000000
	v_dual_fmac_f32 v59, v81, v25 :: v_dual_mul_f32 v26, 0x3b000000, v26
	s_wait_dscnt 0x1
	s_wait_alu 0xfffd
	v_dual_fmac_f32 v82, v31, v25 :: v_dual_cndmask_b32 v27, v29, v27
	v_cmp_lt_u32_e32 vcc_lo, 7, v32
	v_and_b32_e32 v28, 0xffff0000, v28
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v26, v26, v54 :: v_dual_lshlrev_b32 v29, 24, v69
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v30
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v59, v57, v28
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_dual_fmac_f32 v82, v55, v28 :: v_dual_cndmask_b32 v25, 0x7fc00000, v27
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v32
	v_lshlrev_b32_e32 v27, 24, v68
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_and_or_b32 v25, 0x80000000, v29, v25
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v26, 0x7fc00000, v26, vcc_lo
	v_and_or_b32 v26, 0x80000000, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v61, v59, v26, 0
	v_fma_f32 v59, v82, v25, 0
.LBB_320:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_lt_i32_e32 32, v65
	s_cbranch_execz .LBB_322
; %bb.321:
	v_lshlrev_b32_e32 v78, 16, v21
	v_and_b32_e32 v21, 0xffff0000, v21
	v_and_b32_e32 v25, 15, v50
	v_lshrrev_b32_e32 v26, 2, v50
	v_lshrrev_b32_e32 v27, 6, v50
	v_lshrrev_b32_e32 v28, 10, v50
	v_lshrrev_b32_e32 v29, 14, v50
	v_lshrrev_b32_e32 v30, 18, v50
	v_lshrrev_b32_e32 v31, 22, v50
	v_lshrrev_b32_e32 v32, 26, v50
	v_lshlrev_b32_e32 v80, 16, v22
	v_lshlrev_b32_e32 v25, 2, v25
	v_and_b32_e32 v26, 60, v26
	v_and_b32_e32 v22, 0xffff0000, v22
	v_and_b32_e32 v27, 60, v27
	v_and_b32_e32 v28, 60, v28
	v_and_b32_e32 v29, 60, v29
	v_and_b32_e32 v30, 60, v30
	v_and_b32_e32 v31, 60, v31
	v_and_b32_e32 v32, 60, v32
	ds_load_b32 v25, v25
	ds_load_b32 v26, v26
	ds_load_b32 v27, v27
	ds_load_b32 v28, v28
	ds_load_b32 v29, v29
	ds_load_b32 v30, v30
	ds_load_b32 v31, v31
	ds_load_b32 v32, v32
	v_lshrrev_b32_e32 v54, 2, v51
	v_lshrrev_b32_e32 v55, 6, v51
	v_lshrrev_b32_e32 v56, 10, v51
	v_lshrrev_b32_e32 v57, 14, v51
	v_lshrrev_b32_e32 v68, 18, v51
	v_lshrrev_b32_e32 v69, 22, v51
	v_lshrrev_b32_e32 v71, 2, v52
	v_and_b32_e32 v55, 60, v55
	v_lshrrev_b32_e32 v72, 6, v52
	v_and_b32_e32 v68, 60, v68
	v_lshrrev_b32_e32 v73, 10, v52
	v_and_b32_e32 v71, 60, v71
	s_wait_dscnt 0x7
	v_fma_f32 v25, v25, v78, 0
	v_lshrrev_b32_e32 v74, 14, v52
	v_lshrrev_b32_e32 v75, 18, v52
	v_lshrrev_b32_e32 v76, 22, v52
	v_and_b32_e32 v73, 60, v73
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v25, v26, v21
	v_and_b32_e32 v77, 15, v53
	v_and_b32_e32 v75, 60, v75
	v_and_b32_e32 v76, 60, v76
	v_lshrrev_b32_e32 v79, 2, v53
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v25, v27, v80
	v_lshlrev_b32_e32 v27, 16, v23
	v_and_b32_e32 v23, 0xffff0000, v23
	v_and_b32_e32 v57, 60, v57
	v_lshrrev_b32_e32 v26, 6, v53
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v25, v28, v22
	v_lshlrev_b32_e32 v77, 2, v77
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v25, v29, v27 :: v_dual_and_b32 v54, 60, v54
	v_lshrrev_b32_e32 v29, 18, v53
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v25, v30, v23
	v_and_b32_e32 v50, 15, v51
	v_lshrrev_b32_e32 v51, 26, v51
	v_lshlrev_b32_e32 v30, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v17
	v_and_b32_e32 v70, 15, v52
	v_lshlrev_b32_e32 v50, 2, v50
	v_and_b32_e32 v56, 60, v56
	v_and_b32_e32 v69, 60, v69
	v_and_b32_e32 v51, 60, v51
	v_lshlrev_b32_e32 v70, 2, v70
	ds_load_b32 v50, v50
	ds_load_b32 v54, v54
	ds_load_b32 v55, v55
	ds_load_b32 v56, v56
	ds_load_b32 v57, v57
	ds_load_b32 v68, v68
	ds_load_b32 v69, v69
	ds_load_b32 v51, v51
	ds_load_b32 v70, v70
	ds_load_b32 v71, v71
	v_lshrrev_b32_e32 v52, 26, v52
	v_and_b32_e32 v72, 60, v72
	v_and_b32_e32 v74, 60, v74
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v52, 60, v52
	ds_load_b32 v72, v72
	ds_load_b32 v73, v73
	ds_load_b32 v74, v74
	ds_load_b32 v75, v75
	ds_load_b32 v76, v76
	ds_load_b32 v52, v52
	s_wait_dscnt 0x7
	v_fma_f32 v70, v70, v78, 0
	v_and_b32_e32 v78, 60, v79
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v70, v71, v21 :: v_dual_and_b32 v21, 60, v26
	ds_load_b32 v28, v77
	ds_load_b32 v71, v78
	ds_load_b32 v21, v21
	s_wait_dscnt 0x8
	v_fmac_f32_e32 v70, v72, v80
	v_lshlrev_b32_e32 v72, 16, v24
	v_lshrrev_b32_e32 v26, 10, v53
	v_and_b32_e32 v24, 0xffff0000, v24
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v70, v73, v22
	v_lshrrev_b32_e32 v22, 14, v53
	v_dual_fmac_f32 v25, v31, v72 :: v_dual_and_b32 v26, 60, v26
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v70, v74, v27 :: v_dual_and_b32 v27, 60, v29
	v_lshrrev_b32_e32 v29, 26, v53
	v_dual_fmac_f32 v25, v32, v24 :: v_dual_and_b32 v22, 60, v22
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v70, v75, v23
	v_lshrrev_b32_e32 v23, 22, v53
	v_and_b32_e32 v29, 60, v29
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v25, v50, v30 :: v_dual_fmac_f32 v70, v76, v72
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_b32_e32 v23, 60, v23
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v25, v54, v17 :: v_dual_fmac_f32 v70, v52, v24
	ds_load_b32 v24, v26
	ds_load_b32 v22, v22
	ds_load_b32 v26, v27
	ds_load_b32 v23, v23
	ds_load_b32 v27, v29
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v70, v28, v30
	v_lshlrev_b32_e32 v28, 16, v18
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v70, v71, v17 :: v_dual_and_b32 v17, 0xffff0000, v18
	v_dual_fmac_f32 v25, v55, v28 :: v_dual_lshlrev_b32 v18, 16, v19
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v70, v21, v28 :: v_dual_and_b32 v19, 0xffff0000, v19
	v_and_b32_e32 v21, 7, v67
	v_fmac_f32_e32 v25, v56, v17
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v70, v24, v17
	v_cvt_f32_ubyte0_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v25, v57, v18
	v_and_b32_e32 v24, 0x7f, v66
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v70, v22, v18 :: v_dual_lshlrev_b32 v17, 16, v20
	v_and_b32_e32 v18, 7, v66
	v_and_b32_e32 v22, 0x7f, v67
	v_fmac_f32_e32 v25, v68, v19
	s_wait_dscnt 0x2
	v_dual_mul_f32 v21, 0x3b000000, v21 :: v_dual_fmac_f32 v70, v26, v19
	v_cvt_f32_ubyte0_e32 v18, v18
	v_lshl_add_u32 v19, v22, 20, 0x3c000000
	v_cmp_lt_u32_e32 vcc_lo, 7, v22
	v_lshl_add_u32 v26, v24, 20, 0x3c000000
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v25, v69, v17 :: v_dual_mul_f32 v18, 0x3b000000, v18
	s_wait_dscnt 0x1
	s_wait_alu 0xfffd
	v_dual_fmac_f32 v70, v23, v17 :: v_dual_cndmask_b32 v19, v21, v19
	v_cmp_lt_u32_e32 vcc_lo, 7, v24
	v_and_b32_e32 v20, 0xffff0000, v20
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v18, v18, v26 :: v_dual_lshlrev_b32 v21, 24, v67
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v22
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v25, v51, v20
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_dual_fmac_f32 v70, v27, v20 :: v_dual_cndmask_b32 v17, 0x7fc00000, v19
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v24
	v_lshlrev_b32_e32 v19, 24, v66
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_or_b32 v17, 0x80000000, v21, v17
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v18, 0x7fc00000, v18 :: v_dual_fmac_f32 v59, v70, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v18, 0x80000000, v19, v18
	v_fmac_f32_e32 v61, v25, v18
.LBB_322:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_lt_i32_e32 64, v65
	s_cbranch_execz .LBB_324
; %bb.323:
	v_lshlrev_b32_e32 v56, 16, v13
	v_and_b32_e32 v13, 0xffff0000, v13
	v_and_b32_e32 v17, 15, v48
	v_lshrrev_b32_e32 v18, 2, v48
	v_lshrrev_b32_e32 v19, 6, v48
	v_lshrrev_b32_e32 v20, 10, v48
	v_lshrrev_b32_e32 v21, 14, v48
	v_lshrrev_b32_e32 v22, 18, v48
	v_lshrrev_b32_e32 v23, 22, v48
	v_lshrrev_b32_e32 v24, 26, v48
	v_lshlrev_b32_e32 v66, 16, v14
	v_lshlrev_b32_e32 v17, 2, v17
	v_and_b32_e32 v18, 60, v18
	v_and_b32_e32 v14, 0xffff0000, v14
	v_and_b32_e32 v19, 60, v19
	v_and_b32_e32 v20, 60, v20
	v_and_b32_e32 v21, 60, v21
	v_and_b32_e32 v22, 60, v22
	v_and_b32_e32 v23, 60, v23
	v_and_b32_e32 v24, 60, v24
	ds_load_b32 v17, v17
	ds_load_b32 v18, v18
	ds_load_b32 v19, v19
	ds_load_b32 v20, v20
	ds_load_b32 v21, v21
	ds_load_b32 v22, v22
	ds_load_b32 v23, v23
	ds_load_b32 v24, v24
	v_lshrrev_b32_e32 v28, 10, v49
	v_lshrrev_b32_e32 v26, 2, v49
	v_lshrrev_b32_e32 v27, 6, v49
	v_lshrrev_b32_e32 v29, 14, v49
	v_lshrrev_b32_e32 v30, 18, v49
	v_lshrrev_b32_e32 v31, 22, v49
	v_lshrrev_b32_e32 v32, 26, v49
	v_lshrrev_b32_e32 v50, 6, v46
	v_and_b32_e32 v29, 60, v29
	v_and_b32_e32 v30, 60, v30
	v_lshrrev_b32_e32 v51, 10, v46
	v_lshrrev_b32_e32 v52, 14, v46
	s_wait_dscnt 0x7
	v_fma_f32 v17, v17, v56, 0
	v_lshrrev_b32_e32 v53, 18, v46
	v_lshrrev_b32_e32 v54, 22, v46
	v_and_b32_e32 v51, 60, v51
	v_and_b32_e32 v55, 15, v47
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v17, v18, v13
	v_and_b32_e32 v53, 60, v53
	v_lshrrev_b32_e32 v57, 2, v47
	v_lshrrev_b32_e32 v18, 6, v47
	v_lshlrev_b32_e32 v55, 2, v55
	s_wait_dscnt 0x5
	v_dual_fmac_f32 v17, v19, v66 :: v_dual_and_b32 v28, 60, v28
	v_lshlrev_b32_e32 v19, 16, v15
	v_and_b32_e32 v15, 0xffff0000, v15
	v_and_b32_e32 v25, 15, v49
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v17, v20, v14
	v_lshrrev_b32_e32 v49, 2, v46
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v25, 2, v25
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v17, v21, v19 :: v_dual_and_b32 v26, 60, v26
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_and_b32_e32 v49, 60, v49
	v_lshrrev_b32_e32 v21, 18, v47
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v17, v22, v15 :: v_dual_lshlrev_b32 v22, 16, v9
	v_and_b32_e32 v9, 0xffff0000, v9
	v_and_b32_e32 v48, 15, v46
	v_and_b32_e32 v27, 60, v27
	v_and_b32_e32 v31, 60, v31
	v_and_b32_e32 v32, 60, v32
	v_lshrrev_b32_e32 v46, 26, v46
	v_lshlrev_b32_e32 v48, 2, v48
	ds_load_b32 v25, v25
	ds_load_b32 v26, v26
	ds_load_b32 v27, v27
	ds_load_b32 v28, v28
	ds_load_b32 v29, v29
	ds_load_b32 v30, v30
	ds_load_b32 v31, v31
	ds_load_b32 v32, v32
	ds_load_b32 v48, v48
	ds_load_b32 v49, v49
	v_and_b32_e32 v50, 60, v50
	v_and_b32_e32 v52, 60, v52
	v_and_b32_e32 v54, 60, v54
	v_and_b32_e32 v46, 60, v46
	ds_load_b32 v50, v50
	ds_load_b32 v51, v51
	ds_load_b32 v52, v52
	ds_load_b32 v53, v53
	ds_load_b32 v54, v54
	ds_load_b32 v46, v46
	s_wait_dscnt 0x7
	v_fma_f32 v48, v48, v56, 0
	v_and_b32_e32 v56, 60, v57
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v48, v49, v13 :: v_dual_and_b32 v13, 60, v18
	ds_load_b32 v20, v55
	ds_load_b32 v49, v56
	ds_load_b32 v13, v13
	s_wait_dscnt 0x8
	v_fmac_f32_e32 v48, v50, v66
	v_lshlrev_b32_e32 v50, 16, v16
	v_lshrrev_b32_e32 v18, 10, v47
	v_and_b32_e32 v16, 0xffff0000, v16
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v48, v51, v14
	v_lshrrev_b32_e32 v14, 14, v47
	v_fmac_f32_e32 v17, v23, v50
	v_and_b32_e32 v18, 60, v18
	s_wait_dscnt 0x6
	v_dual_fmac_f32 v48, v52, v19 :: v_dual_and_b32 v19, 60, v21
	v_lshrrev_b32_e32 v21, 26, v47
	v_dual_fmac_f32 v17, v24, v16 :: v_dual_and_b32 v14, 60, v14
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v48, v53, v15
	v_lshrrev_b32_e32 v15, 22, v47
	v_and_b32_e32 v21, 60, v21
	v_fmac_f32_e32 v17, v25, v22
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v48, v54, v50 :: v_dual_and_b32 v15, 60, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v17, v26, v9
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v48, v46, v16
	ds_load_b32 v16, v18
	ds_load_b32 v14, v14
	ds_load_b32 v18, v19
	ds_load_b32 v15, v15
	ds_load_b32 v19, v21
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v48, v20, v22
	v_lshlrev_b32_e32 v20, 16, v10
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v48, v49, v9 :: v_dual_and_b32 v9, 0xffff0000, v10
	v_dual_fmac_f32 v17, v27, v20 :: v_dual_lshlrev_b32 v10, 16, v11
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v48, v13, v20 :: v_dual_and_b32 v11, 0xffff0000, v11
	v_and_b32_e32 v13, 7, v64
	v_fmac_f32_e32 v17, v28, v9
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v48, v16, v9
	v_cvt_f32_ubyte0_e32 v13, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v17, v29, v10 :: v_dual_and_b32 v16, 0x7f, v63
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v48, v14, v10 :: v_dual_lshlrev_b32 v9, 16, v12
	v_and_b32_e32 v10, 7, v63
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v17, v30, v11 :: v_dual_and_b32 v14, 0x7f, v64
	s_wait_dscnt 0x2
	v_dual_mul_f32 v13, 0x3b000000, v13 :: v_dual_fmac_f32 v48, v18, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v10, v10
	v_lshl_add_u32 v11, v14, 20, 0x3c000000
	v_cmp_lt_u32_e32 vcc_lo, 7, v14
	v_lshl_add_u32 v18, v16, 20, 0x3c000000
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v17, v31, v9 :: v_dual_mul_f32 v10, 0x3b000000, v10
	s_wait_dscnt 0x1
	s_wait_alu 0xfffd
	v_dual_fmac_f32 v48, v15, v9 :: v_dual_cndmask_b32 v11, v13, v11
	v_cmp_lt_u32_e32 vcc_lo, 7, v16
	v_and_b32_e32 v12, 0xffff0000, v12
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v10, v10, v18 :: v_dual_lshlrev_b32 v13, 24, v64
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v14
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v17, v32, v12
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_dual_fmac_f32 v48, v19, v12 :: v_dual_cndmask_b32 v9, 0x7fc00000, v11
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v16
	v_lshlrev_b32_e32 v11, 24, v63
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_or_b32 v9, 0x80000000, v13, v9
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v10, 0x7fc00000, v10 :: v_dual_fmac_f32 v59, v48, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v10, 0x80000000, v11, v10
	v_fmac_f32_e32 v61, v17, v10
.LBB_324:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_lt_i32_e32 0x60, v65
	s_cbranch_execz .LBB_326
; %bb.325:
	v_and_b32_e32 v9, 15, v44
	v_lshrrev_b32_e32 v10, 2, v44
	v_lshrrev_b32_e32 v11, 6, v44
	v_lshrrev_b32_e32 v12, 10, v44
	v_lshrrev_b32_e32 v13, 14, v44
	v_lshrrev_b32_e32 v14, 18, v44
	v_lshrrev_b32_e32 v15, 22, v44
	v_lshrrev_b32_e32 v16, 26, v44
	v_lshlrev_b32_e32 v9, 2, v9
	v_lshlrev_b32_e32 v44, 16, v5
	v_and_b32_e32 v5, 0xffff0000, v5
	v_and_b32_e32 v10, 60, v10
	v_and_b32_e32 v11, 60, v11
	v_and_b32_e32 v12, 60, v12
	v_and_b32_e32 v13, 60, v13
	v_and_b32_e32 v14, 60, v14
	v_and_b32_e32 v15, 60, v15
	v_lshlrev_b32_e32 v46, 16, v6
	v_and_b32_e32 v16, 60, v16
	ds_load_b32 v9, v9
	ds_load_b32 v10, v10
	ds_load_b32 v11, v11
	ds_load_b32 v12, v12
	ds_load_b32 v13, v13
	ds_load_b32 v14, v14
	ds_load_b32 v15, v15
	ds_load_b32 v16, v16
	v_lshrrev_b32_e32 v24, 26, v45
	v_and_b32_e32 v6, 0xffff0000, v6
	v_and_b32_e32 v17, 15, v45
	v_lshrrev_b32_e32 v18, 2, v45
	v_lshrrev_b32_e32 v19, 6, v45
	v_lshrrev_b32_e32 v20, 10, v45
	v_lshrrev_b32_e32 v21, 14, v45
	v_lshrrev_b32_e32 v22, 18, v45
	v_lshrrev_b32_e32 v23, 22, v45
	v_and_b32_e32 v25, 15, v42
	v_lshrrev_b32_e32 v26, 2, v42
	v_lshlrev_b32_e32 v17, 2, v17
	s_wait_dscnt 0x7
	v_fma_f32 v9, v9, v44, 0
	v_and_b32_e32 v19, 60, v19
	v_and_b32_e32 v20, 60, v20
	v_and_b32_e32 v21, 60, v21
	v_and_b32_e32 v23, 60, v23
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v9, v10, v5
	v_lshlrev_b32_e32 v25, 2, v25
	v_lshrrev_b32_e32 v27, 6, v42
	v_lshrrev_b32_e32 v28, 10, v42
	v_lshrrev_b32_e32 v29, 14, v42
	s_wait_dscnt 0x5
	v_dual_fmac_f32 v9, v11, v46 :: v_dual_and_b32 v24, 60, v24
	v_lshlrev_b32_e32 v11, 16, v7
	v_and_b32_e32 v7, 0xffff0000, v7
	v_and_b32_e32 v18, 60, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v9, v12, v6
	v_and_b32_e32 v26, 60, v26
	v_lshrrev_b32_e32 v30, 18, v42
	v_lshrrev_b32_e32 v31, 22, v42
	v_lshrrev_b32_e32 v32, 26, v42
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v9, v13, v11 :: v_dual_and_b32 v22, 60, v22
	v_and_b32_e32 v27, 60, v27
	v_and_b32_e32 v29, 60, v29
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v9, v14, v7 :: v_dual_and_b32 v30, 60, v30
	ds_load_b32 v17, v17
	ds_load_b32 v18, v18
	ds_load_b32 v19, v19
	ds_load_b32 v20, v20
	ds_load_b32 v21, v21
	ds_load_b32 v22, v22
	ds_load_b32 v23, v23
	ds_load_b32 v24, v24
	ds_load_b32 v25, v25
	ds_load_b32 v26, v26
	v_and_b32_e32 v28, 60, v28
	v_and_b32_e32 v31, 60, v31
	v_and_b32_e32 v32, 60, v32
	ds_load_b32 v27, v27
	ds_load_b32 v28, v28
	ds_load_b32 v29, v29
	ds_load_b32 v30, v30
	ds_load_b32 v31, v31
	ds_load_b32 v32, v32
	v_lshlrev_b32_e32 v14, 16, v1
	v_lshrrev_b32_e32 v45, 2, v43
	v_lshrrev_b32_e32 v10, 6, v43
	v_and_b32_e32 v1, 0xffff0000, v1
	v_lshrrev_b32_e32 v13, 18, v43
	s_wait_dscnt 0x7
	v_fma_f32 v25, v25, v44, 0
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v25, v26, v5
	v_and_b32_e32 v5, 60, v10
	v_lshrrev_b32_e32 v10, 10, v43
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v25, v27, v46
	v_lshlrev_b32_e32 v27, 16, v8
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v25, v28, v6 :: v_dual_and_b32 v8, 0xffff0000, v8
	v_fmac_f32_e32 v9, v15, v27
	v_and_b32_e32 v42, 15, v43
	v_lshrrev_b32_e32 v6, 14, v43
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v25, v29, v11
	v_dual_fmac_f32 v9, v16, v8 :: v_dual_and_b32 v44, 60, v45
	v_and_b32_e32 v11, 60, v13
	v_lshrrev_b32_e32 v13, 26, v43
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v25, v30, v7
	v_fmac_f32_e32 v9, v17, v14
	v_lshlrev_b32_e32 v42, 2, v42
	v_lshrrev_b32_e32 v7, 22, v43
	v_and_b32_e32 v13, 60, v13
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v25, v31, v27
	v_fmac_f32_e32 v9, v18, v1
	ds_load_b32 v12, v42
	ds_load_b32 v26, v44
	ds_load_b32 v5, v5
	v_and_b32_e32 v7, 60, v7
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v25, v32, v8
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v25, v12, v14
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v25, v26, v1
	v_and_b32_e32 v1, 0xffff0000, v2
	v_lshlrev_b32_e32 v12, 16, v2
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v25, v5, v12 :: v_dual_and_b32 v10, 60, v10
	v_dual_fmac_f32 v9, v19, v12 :: v_dual_and_b32 v6, 60, v6
	v_and_b32_e32 v5, 7, v62
	v_fmac_f32_e32 v9, v20, v1
	ds_load_b32 v8, v10
	ds_load_b32 v6, v6
	ds_load_b32 v10, v11
	ds_load_b32 v7, v7
	ds_load_b32 v11, v13
	v_cvt_f32_ubyte0_e32 v5, v5
	v_lshlrev_b32_e32 v2, 16, v3
	v_and_b32_e32 v3, 0xffff0000, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v5, 0x3b000000, v5
	v_fmac_f32_e32 v9, v21, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v22, v3
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v25, v8, v1
	v_lshlrev_b32_e32 v1, 16, v4
	v_and_b32_e32 v4, 0xffff0000, v4
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v25, v6, v2 :: v_dual_and_b32 v8, 0x7f, v60
	v_and_b32_e32 v2, 7, v60
	v_dual_fmac_f32 v9, v23, v1 :: v_dual_and_b32 v6, 0x7f, v62
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v25, v10, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v2, v2
	v_lshl_add_u32 v3, v6, 20, 0x3c000000
	v_cmp_lt_u32_e32 vcc_lo, 7, v6
	v_lshl_add_u32 v10, v8, 20, 0x3c000000
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v25, v7, v1 :: v_dual_mul_f32 v2, 0x3b000000, v2
	v_fmac_f32_e32 v9, v24, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v5, v3, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, 7, v8
	v_lshlrev_b32_e32 v5, 24, v62
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_dual_fmac_f32 v25, v11, v4 :: v_dual_cndmask_b32 v2, v2, v10
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v6
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v1, 0x7fc00000, v3, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v8
	v_lshlrev_b32_e32 v3, 24, v60
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_or_b32 v1, 0x80000000, v5, v1
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v2, 0x7fc00000, v2 :: v_dual_fmac_f32 v59, v25, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v2, 0x80000000, v3, v2
	v_fmac_f32_e32 v61, v9, v2
.LBB_326:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v1, 0x80, v58
	s_mov_b32 s11, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s14, v1
	s_cbranch_execz .LBB_330
; %bb.327:
	v_and_b32_e32 v6, 31, v0
	s_mov_b32 s15, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v0, 5, v6
	v_lshlrev_b32_e32 v2, 3, v6
	v_add_co_u32 v0, s0, s8, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v1, null, s9, 0, s0
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v3, vcc_lo, v34, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, 0, v35, vcc_lo
	v_add_co_u32 v0, vcc_lo, 0x101c, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	v_add_co_u32 v2, vcc_lo, v40, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, 0, v41, vcc_lo
	v_add_co_u32 v3, vcc_lo, s24, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s25, v4, vcc_lo
	v_add_co_u32 v7, vcc_lo, s24, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, s25, v5, vcc_lo
	v_add_co_u32 v2, vcc_lo, 0x400, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, 0, v4, vcc_lo
	v_add_co_u32 v4, vcc_lo, 0x400, v7
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	v_add_co_u32 v7, vcc_lo, s26, v36
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, s27, v37, vcc_lo
	v_add_co_u32 v9, vcc_lo, s26, v38
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v10, null, s27, v39, vcc_lo
	v_add_co_u32 v7, vcc_lo, 0x80, v7
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, 0, v8, vcc_lo
	v_add_co_u32 v9, vcc_lo, 0x80, v9
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v10, null, 0, v10, vcc_lo
	s_mov_b64 s[8:9], 0x80
.LBB_328:                              ; =>This Inner Loop Header: Depth=1
	global_load_b64 v[19:20], v[2:3], off th:TH_LOAD_NT
	global_load_b64 v[21:22], v[4:5], off th:TH_LOAD_NT
	s_clause 0x1
	global_load_b128 v[11:14], v[0:1], off offset:-28
	global_load_b128 v[15:18], v[0:1], off offset:-12
	v_add_co_u32 v23, vcc_lo, v9, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v24, null, 0, v10, vcc_lo
	v_add_co_u32 v25, vcc_lo, v7, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v26, null, 0, v8, vcc_lo
	global_load_i8 v23, v[23:24], off
	global_load_i8 v24, v[25:26], off
	v_add_co_u32 v0, vcc_lo, 0x400, v0
	s_wait_alu 0xfffe
	s_add_nc_u64 s[8:9], s[8:9], 32
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	v_add_co_u32 v2, vcc_lo, 0x100, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	v_add_co_u32 v4, vcc_lo, 0x100, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, 32
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, 0, v8, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, 32
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v10, null, 0, v10, vcc_lo
	s_wait_loadcnt 0x5
	v_lshrrev_b32_e32 v26, 2, v19
	s_wait_loadcnt 0x4
	v_lshrrev_b32_e32 v41, 2, v21
	v_lshrrev_b32_e32 v42, 6, v21
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v60, 16, v15
	v_lshlrev_b32_e32 v57, 16, v14
	v_and_b32_e32 v14, 0xffff0000, v14
	v_lshlrev_b32_e32 v56, 16, v13
	v_and_b32_e32 v13, 0xffff0000, v13
	v_lshlrev_b32_e32 v55, 16, v12
	v_and_b32_e32 v12, 0xffff0000, v12
	v_and_b32_e32 v25, 15, v19
	v_and_b32_e32 v40, 15, v21
	v_and_b32_e32 v26, 60, v26
	v_and_b32_e32 v41, 60, v41
	v_lshrrev_b32_e32 v27, 6, v19
	v_lshlrev_b32_e32 v25, 2, v25
	v_lshlrev_b32_e32 v40, 2, v40
	ds_load_b32 v25, v25
	ds_load_b32 v40, v40
	ds_load_b32 v26, v26
	ds_load_b32 v41, v41
	v_lshrrev_b32_e32 v28, 10, v19
	v_lshrrev_b32_e32 v29, 14, v19
	v_lshrrev_b32_e32 v30, 18, v19
	v_lshrrev_b32_e32 v31, 22, v19
	v_lshrrev_b32_e32 v19, 26, v19
	v_lshrrev_b32_e32 v43, 10, v21
	v_lshrrev_b32_e32 v44, 14, v21
	v_lshrrev_b32_e32 v45, 18, v21
	v_lshrrev_b32_e32 v46, 22, v21
	v_lshrrev_b32_e32 v21, 26, v21
	v_and_b32_e32 v42, 60, v42
	v_and_b32_e32 v27, 60, v27
	v_and_b32_e32 v28, 60, v28
	v_and_b32_e32 v29, 60, v29
	v_and_b32_e32 v30, 60, v30
	v_and_b32_e32 v31, 60, v31
	v_and_b32_e32 v19, 60, v19
	v_and_b32_e32 v43, 60, v43
	v_and_b32_e32 v44, 60, v44
	v_and_b32_e32 v45, 60, v45
	v_and_b32_e32 v46, 60, v46
	v_and_b32_e32 v21, 60, v21
	ds_load_b32 v42, v42
	ds_load_b32 v27, v27
	ds_load_b32 v28, v28
	ds_load_b32 v29, v29
	ds_load_b32 v30, v30
	ds_load_b32 v31, v31
	ds_load_b32 v19, v19
	ds_load_b32 v43, v43
	ds_load_b32 v44, v44
	ds_load_b32 v45, v45
	ds_load_b32 v46, v46
	ds_load_b32 v21, v21
	v_lshlrev_b32_e32 v54, 16, v11
	v_and_b32_e32 v11, 0xffff0000, v11
	v_lshrrev_b32_e32 v34, 2, v20
	v_lshrrev_b32_e32 v35, 6, v20
	v_lshrrev_b32_e32 v36, 10, v20
	s_wait_dscnt 0xf
	v_fma_f32 v25, v25, v54, 0
	v_and_b32_e32 v32, 15, v20
	v_lshrrev_b32_e32 v37, 14, v20
	v_lshrrev_b32_e32 v48, 2, v22
	v_lshrrev_b32_e32 v38, 18, v20
	s_wait_dscnt 0xd
	v_dual_fmac_f32 v25, v26, v11 :: v_dual_lshlrev_b32 v32, 2, v32
	v_and_b32_e32 v36, 60, v36
	v_lshrrev_b32_e32 v39, 22, v20
	v_and_b32_e32 v37, 60, v37
	s_wait_dscnt 0xa
	v_fmac_f32_e32 v25, v27, v55
	ds_load_b32 v32, v32
	v_lshrrev_b32_e32 v49, 6, v22
	v_and_b32_e32 v48, 60, v48
	s_wait_dscnt 0xa
	v_dual_fmac_f32 v25, v28, v12 :: v_dual_and_b32 v38, 60, v38
	v_fma_f32 v40, v40, v54, 0
	v_and_b32_e32 v47, 15, v22
	v_lshrrev_b32_e32 v20, 26, v20
	v_and_b32_e32 v39, 60, v39
	s_wait_dscnt 0x9
	v_fmac_f32_e32 v25, v29, v56
	v_fmac_f32_e32 v40, v41, v11
	v_lshlrev_b32_e32 v47, 2, v47
	v_lshrrev_b32_e32 v50, 10, v22
	v_and_b32_e32 v49, 60, v49
	v_lshrrev_b32_e32 v51, 14, v22
	v_fmac_f32_e32 v40, v42, v55
	s_wait_dscnt 0x8
	v_dual_fmac_f32 v25, v30, v13 :: v_dual_and_b32 v20, 60, v20
	v_and_b32_e32 v11, 60, v50
	v_and_b32_e32 v26, 60, v51
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v40, v43, v12
	v_fmac_f32_e32 v25, v31, v57
	v_lshrrev_b32_e32 v52, 18, v22
	v_lshrrev_b32_e32 v53, 22, v22
	v_lshrrev_b32_e32 v22, 26, v22
	s_wait_loadcnt 0x0
	v_and_b32_e32 v31, 0x7f, v24
	v_fmac_f32_e32 v25, v19, v14
	v_lshlrev_b32_e32 v64, 16, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v40, v44, v56
	v_and_b32_e32 v41, 60, v52
	v_and_b32_e32 v27, 60, v53
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v25, v32, v60 :: v_dual_and_b32 v34, 60, v34
	v_and_b32_e32 v35, 60, v35
	ds_load_b32 v12, v34
	ds_load_b32 v28, v35
	ds_load_b32 v34, v47
	ds_load_b32 v29, v36
	ds_load_b32 v35, v37
	ds_load_b32 v36, v48
	v_dual_fmac_f32 v40, v45, v13 :: v_dual_and_b32 v15, 0xffff0000, v15
	ds_load_b32 v13, v38
	ds_load_b32 v30, v39
	ds_load_b32 v37, v49
	ds_load_b32 v20, v20
	ds_load_b32 v11, v11
	ds_load_b32 v26, v26
	v_and_b32_e32 v22, 60, v22
	v_fmac_f32_e32 v40, v46, v57
	v_lshlrev_b32_e32 v63, 16, v17
	v_lshlrev_b32_e32 v62, 16, v16
	v_cmp_lt_u32_e64 s1, 7, v31
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v21, v14
	ds_load_b32 v14, v41
	ds_load_b32 v19, v27
	ds_load_b32 v21, v22
	v_and_b32_e32 v27, 7, v23
	v_and_b32_e32 v22, 0x7f, v23
	s_wait_dscnt 0xe
	v_dual_fmac_f32 v25, v12, v15 :: v_dual_and_b32 v16, 0xffff0000, v16
	s_wait_dscnt 0xc
	v_dual_fmac_f32 v40, v34, v60 :: v_dual_and_b32 v17, 0xffff0000, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v25, v28, v62 :: v_dual_and_b32 v12, 7, v24
	v_and_b32_e32 v18, 0xffff0000, v18
	s_wait_dscnt 0x9
	v_fmac_f32_e32 v40, v36, v15
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f32_ubyte0_e32 v12, v12
	v_fmac_f32_e32 v25, v29, v16
	v_lshl_add_u32 v15, v22, 20, 0x3c000000
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v40, v37, v62
	v_lshl_add_u32 v28, v31, 20, 0x3c000000
	v_mul_f32_e32 v27, 0x3b000000, v27
	v_cmp_lt_u32_e64 s0, 7, v22
	v_fmac_f32_e32 v25, v35, v63
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v40, v11, v16
	v_mul_f32_e32 v11, 0x3b000000, v12
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v12, v27, v15, s0
	v_cmp_ne_u32_e64 s0, 0x7f, v31
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v40, v26, v63
	v_cndmask_b32_e64 v11, v11, v28, s1
	v_fmac_f32_e32 v25, v13, v17
	v_cmp_ne_u32_e64 s1, 0x7f, v22
	s_wait_dscnt 0x2
	s_wait_alu 0xfffe
	v_dual_fmac_f32 v40, v14, v17 :: v_dual_add_nc_u32 v13, s8, v6
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v11, 0x7fc00000, v11, s0
	v_fmac_f32_e32 v25, v30, v64
	v_cndmask_b32_e64 v12, 0x7fc00000, v12, s1
	v_cmp_le_i32_e64 s0, s14, v13
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v40, v19, v64
	v_and_or_b32 v11, 0x80000000, v24, v11
	v_fmac_f32_e32 v25, v20, v18
	v_and_or_b32 v12, 0x80000000, v23, v12
	s_or_b32 s15, s0, s15
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v40, v21, v18
	v_fmac_f32_e32 v61, v25, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v59, v40, v12
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s15
	s_cbranch_execnz .LBB_328
; %bb.329:
	s_or_b32 exec_lo, exec_lo, s15
.LBB_330:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s11
	v_mbcnt_lo_u32_b32 v1, -1, 0
	s_mov_b32 s1, 0
	s_mul_i32 s0, s3, 3
	s_wait_alu 0xfffe
	s_lshl_b64 s[0:1], s[0:1], 2
	v_xor_b32_e32 v0, 16, v1
	v_xor_b32_e32 v2, 8, v1
	v_xor_b32_e32 v7, 1, v1
	s_wait_alu 0xfffe
	s_add_nc_u64 s[0:1], s[12:13], s[0:1]
	s_load_b32 s3, s[0:1], 0x0
	v_cmp_gt_i32_e32 vcc_lo, 32, v0
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v0, v1, v0, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v0, 2, v0
	ds_bpermute_b32 v3, v0, v61
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v61, v3
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v2, v1, v2, vcc_lo
	v_xor_b32_e32 v3, 4, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v2, 2, v2
	v_cmp_gt_i32_e32 vcc_lo, 32, v3
	ds_bpermute_b32 v5, v2, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v1, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v3, 2, v3
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v4, v5
	v_xor_b32_e32 v4, 2, v1
	ds_bpermute_b32 v6, v3, v5
	v_cmp_gt_i32_e32 vcc_lo, 32, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v4, v1, v4, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v7
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_cndmask_b32 v1, v1, v7 :: v_dual_lshlrev_b32 v4, 2, v4
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	ds_bpermute_b32 v6, v4, v5
	s_wait_dscnt 0x0
	v_dual_add_f32 v6, v5, v6 :: v_dual_lshlrev_b32 v5, 2, v1
	ds_bpermute_b32 v1, v5, v6
	s_wait_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v6, s3, v1
	v_and_b32_e32 v1, 0x7fffffff, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.331:
	v_bfe_u32 v1, v6, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v1, v6, v1, 0x7fff
                                        ; implicit-def: $vgpr6
	v_lshrrev_b32_e32 v1, 16, v1
; %bb.332:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.333:
	v_lshrrev_b32_e32 v1, 16, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v1, 64, v1
; %bb.334:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	ds_bpermute_b32 v0, v0, v59
	s_load_b32 s0, s[0:1], 0x4
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v59, v0
	ds_bpermute_b32 v2, v2, v0
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v0, v2
	ds_bpermute_b32 v2, v3, v0
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v0, v2
	ds_bpermute_b32 v2, v4, v0
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v0, v2
	ds_bpermute_b32 v2, v5, v0
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v0, v2
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v2, s0, v0
	v_and_b32_e32 v0, 0x7fffffff, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.335:
	v_bfe_u32 v0, v2, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v0, v2, v0, 0x7fff
                                        ; implicit-def: $vgpr2
	v_lshrrev_b32_e32 v0, 16, v0
; %bb.336:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.337:
	v_lshrrev_b32_e32 v0, 16, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v0, 64, v0
; %bb.338:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_lshlrev_b32_e32 v1, 16, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v2, 0xbfb8aa3b, v1
	v_fma_f32 v3, 0xbfb8aa3b, v1, -v2
	v_rndne_f32_e32 v4, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v2, v2, v4
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v1
	v_fmamk_f32 v3, v1, 0xb2a5705f, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v2, v2, v3
	v_cvt_i32_f32_e32 v3, v4
	v_exp_f32_e32 v2, v2
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v2, v2, v3
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v1
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0x7f800000, v2, vcc_lo
	v_add_f32_e32 v2, 1.0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v3, null, v2, v2, v1
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v5, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v5, v4
	v_div_scale_f32 v5, vcc_lo, v1, v2, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v6, v5, v4
	v_fma_f32 v7, -v3, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v7, v4
	v_fma_f32 v3, -v3, v6, v5
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v6
	v_div_fixup_f32 v2, v3, v2, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v1, 0x7fffffff, v2
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
	s_cbranch_execz .LBB_341
; %bb.339:
	v_bfe_u32 v1, v2, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v1, v2, v1, 0x7fff
                                        ; implicit-def: $vgpr2
	v_lshrrev_b32_e32 v1, 16, v1
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
	s_cbranch_execnz .LBB_342
.LBB_340:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_eq_u32_e32 vcc_lo, 0, v58
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execnz .LBB_343
	s_branch .LBB_348
.LBB_341:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
	s_cbranch_execz .LBB_340
.LBB_342:
	v_lshrrev_b32_e32 v1, 16, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v1, 64, v1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_eq_u32_e32 vcc_lo, 0, v58
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB_348
.LBB_343:
	v_lshlrev_b32_e32 v0, 16, v0
	v_lshlrev_b32_e32 v1, 16, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v1, v0, v1
	v_and_b32_e32 v0, 0x7fffffff, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.344:
	v_bfe_u32 v0, v1, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v0, v1, v0, 0x7fff
                                        ; implicit-def: $vgpr1
	v_lshrrev_b32_e32 v0, 16, v0
; %bb.345:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.346:
	v_lshrrev_b32_e32 v0, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v0, 64, v0
; %bb.347:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_ashrrev_i32_e32 v34, 31, v33
	s_mul_u64 s[0:1], s[4:5], s[6:7]
	s_wait_alu 0xfffe
	s_lshl_b64 s[0:1], s[0:1], 1
	s_wait_alu 0xfffe
	s_add_nc_u64 s[0:1], s[30:31], s[0:1]
	v_lshlrev_b64_e32 v[1:2], 1, v[33:34]
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s0, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s1, v2, vcc_lo
	global_store_b16 v[1:2], v0, off
.LBB_348:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	global_inv scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB_351
; %bb.349:
	s_mov_b32 s0, exec_lo
	s_wait_alu 0xfffe
	v_mbcnt_lo_u32_b32 v0, s0, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_and_b32 s1, exec_lo, vcc_lo
	s_wait_alu 0xfffe
	s_mov_b32 exec_lo, s1
	s_cbranch_execz .LBB_351
; %bb.350:
	s_bcnt1_i32_b32 s0, s0
	s_wait_alu 0xfffe
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s0
	s_wait_loadcnt 0x0
	global_atomic_add_u32 v0, v1, s[28:29] scope:SCOPE_DEV
.LBB_351:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN12_GLOBAL__N_124moe_experts_p1_v3_kernelILi4EEEvPK7ik_bf16PKhS5_S5_S5_PKfPKiS7_PS1_SA_iiiiiiii
		.amdhsa_group_segment_fixed_size 68
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 112
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 107
		.amdhsa_next_free_sgpr 74
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 254
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN12_GLOBAL__N_124moe_experts_p1_v3_kernelILi4EEEvPK7ik_bf16PKhS5_S5_S5_PKfPKiS7_PS1_SA_iiiiiiii,"axG",@progbits,_ZN12_GLOBAL__N_124moe_experts_p1_v3_kernelILi4EEEvPK7ik_bf16PKhS5_S5_S5_PKfPKiS7_PS1_SA_iiiiiiii,comdat
.Lfunc_end8:
