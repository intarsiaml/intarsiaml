_ZN12_GLOBAL__N_116moe_block_kernelILi4ELi8EEEvNS_9BlockArgsE: ; @_ZN12_GLOBAL__N_116moe_block_kernelILi4ELi8EEEvNS_9BlockArgsE
; %bb.0:
	s_clause 0x1
	s_load_b96 s[28:30], s[0:1], 0xc0
	s_load_b256 s[52:59], s[0:1], 0xa0
	s_wait_kmcnt 0x0
	s_bitcmp0_b32 s30, 21
	s_cbranch_scc1 .LBB_3
; %bb.1:
	s_mul_hi_u32 s2, ttmp9, 0x24924925
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s3, ttmp9, s2
	s_lshr_b32 s3, s3, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s3, s3, s2
	s_lshr_b32 s2, s3, 2
	s_and_b32 s3, s30, 0xffff
	s_mul_i32 s2, s2, 7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s2, ttmp9, s2
	s_mul_i32 s2, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .LBB_3
.LBB_2:                                ; =>This Inner Loop Header: Depth=1
	s_add_co_i32 s2, s2, -1
	s_sleep 8
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .LBB_2
.LBB_3:
	s_lshl_b32 s4, s29, 4
	s_getpc_b64 s[6:7]
	s_sext_i32_i16 s7, s7
	s_add_co_u32 s6, s6, _ZN12_GLOBAL__N_116g_moe_block_syncE@rel32@lo+8
	s_add_co_ci_u32 s7, s7, _ZN12_GLOBAL__N_116g_moe_block_syncE@rel32@hi+16
	s_ashr_i32 s5, s4, 31
	v_cmp_eq_u32_e64 s2, 0, v0
	s_lshl_b64 s[4:5], s[4:5], 2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[34:35], s[6:7], s[4:5]
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB_7
; %bb.4:
	s_mov_b32 s5, exec_lo
	s_mov_b32 s4, exec_lo
	v_mbcnt_lo_u32_b32 v1, s5, 0
                                        ; implicit-def: $vgpr2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v1
	s_cbranch_execz .LBB_6
; %bb.5:
	s_bcnt1_i32_b32 s5, s5
	s_wait_alu 0xfffe
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, s5
	global_atomic_add_u32 v2, v2, v3, s[34:35] offset:8 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
.LBB_6:
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s4, v2
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v1, s4, v1
	ds_store_b32 v2, v1 offset:64
.LBB_7:
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v2, 0 :: v_dual_and_b32 v33, 31, v0
	s_clause 0x2
	s_load_b512 s[12:27], s[0:1], 0x0
	s_load_b512 s[36:51], s[0:1], 0x40
	s_load_b256 s[60:67], s[0:1], 0x80
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v1, v2 offset:64
	s_add_co_i32 s31, s59, s58
	v_lshrrev_b32_e32 v61, 5, v0
	s_mov_b32 s0, -1
	s_wait_dscnt 0x0
	v_cmp_le_i32_e32 vcc_lo, s31, v1
	v_readfirstlane_b32 s29, v1
	s_cbranch_vccz .LBB_369
; %bb.8:
	s_add_co_i32 s0, s53, 7
	s_sub_co_i32 s71, s29, s31
	s_ashr_i32 s1, s0, 31
	s_abs_i32 s4, s71
	s_lshr_b32 s1, s1, 29
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s0, s0, s1
	s_ashr_i32 s70, s0, 3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_abs_i32 s0, s70
	s_cvt_f32_u32 s1, s0
	s_sub_co_i32 s3, 0, s0
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s1
	v_readfirstlane_b32 s1, v1
	s_mul_f32 s1, s1, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s1, s1
	s_wait_alu 0xfffe
	s_mul_i32 s3, s3, s1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s3, s1, s3
	s_add_co_i32 s1, s1, s3
	s_xor_b32 s3, s71, s70
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
	s_cselect_b32 s4, s6, s4
	s_wait_alu 0xfffe
	s_add_co_i32 s5, s1, 1
	s_cmp_ge_u32 s4, s0
	s_wait_alu 0xfffe
	s_cselect_b32 s0, s5, s1
	s_ashr_i32 s1, s52, 31
	s_wait_alu 0xfffe
	s_xor_b32 s0, s0, s3
	s_wait_alu 0xfffe
	s_sub_co_i32 s68, s0, s3
	s_mov_b32 s0, s52
	s_ashr_i32 s69, s68, 31
	s_wait_alu 0xfffe
	s_mul_u64 s[4:5], s[68:69], s[0:1]
	v_cmp_gt_i32_e64 s0, s52, v33
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 1
	s_wait_kmcnt 0x0
	s_wait_alu 0xfffe
	s_add_nc_u64 s[10:11], s[14:15], s[4:5]
	s_and_saveexec_b32 s3, s0
	s_cbranch_execz .LBB_10
; %bb.9:
	v_lshlrev_b32_e32 v1, 1, v33
	global_load_u16 v1, v1, s[10:11]
	s_wait_loadcnt 0x0
	v_cmp_gt_i16_e32 vcc_lo, 0, v1
	v_and_b32_e32 v2, 0xffff, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_not_b32_e32 v3, v2
	v_or_b32_e32 v4, 0x8000, v2
	v_cndmask_b32_e32 v1, v4, v3, vcc_lo
	v_and_b32_e32 v5, 0x7f80, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v1, 16, v1
	v_and_b32_e32 v2, 0x7f, v2
	v_cmp_ne_u32_e32 vcc_lo, 0x7f80, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_eq_u32_e64 s1, 0, v2
	s_or_b32 vcc_lo, vcc_lo, s1
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	v_or_b32_e32 v1, v33, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v2, 0xffff, v1
.LBB_10:
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v32, 32, v33
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s52, v32
	s_and_saveexec_b32 s4, s1
	s_cbranch_execz .LBB_12
; %bb.11:
	v_lshlrev_b32_e32 v1, 1, v33
	global_load_u16 v1, v1, s[10:11] offset:64
	s_wait_loadcnt 0x0
	v_and_b32_e32 v4, 0xffff, v1
	v_cmp_gt_i16_e32 vcc_lo, 0, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_not_b32_e32 v5, v4
	v_or_b32_e32 v6, 0x8000, v4
	v_and_b32_e32 v7, 0x7f80, v4
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v1, v6, v5 :: v_dual_and_b32 v4, 0x7f, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f80, v7
	v_cmp_eq_u32_e64 s3, 0, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 16, v1
	s_or_b32 vcc_lo, vcc_lo, s3
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v1, v33, v1
	v_xor_b32_e32 v4, 0xffdf, v1
.LBB_12:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v1, 64, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s3, s52, v1
	s_and_saveexec_b32 s5, s3
	s_cbranch_execz .LBB_14
; %bb.13:
	v_lshlrev_b32_e32 v1, 1, v33
	global_load_u16 v1, v1, s[10:11] offset:128
	s_wait_loadcnt 0x0
	v_and_b32_e32 v3, 0xffff, v1
	v_cmp_gt_i16_e32 vcc_lo, 0, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_not_b32_e32 v5, v3
	v_or_b32_e32 v6, 0x8000, v3
	v_and_b32_e32 v7, 0x7f80, v3
	v_and_b32_e32 v3, 0x7f, v3
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v1, v6, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f80, v7
	v_cmp_eq_u32_e64 s4, 0, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 16, v1
	s_or_b32 vcc_lo, vcc_lo, s4
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v1, v33, v1
	v_xor_b32_e32 v3, 0xffbf, v1
.LBB_14:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 0x60, v33
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s4, s52, v1
	s_and_saveexec_b32 s6, s4
	s_cbranch_execz .LBB_16
; %bb.15:
	v_lshlrev_b32_e32 v1, 1, v33
	global_load_u16 v1, v1, s[10:11] offset:192
	s_wait_loadcnt 0x0
	v_cmp_gt_i16_e32 vcc_lo, 0, v1
	v_and_b32_e32 v6, 0xffff, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_not_b32_e32 v7, v6
	v_or_b32_e32 v8, 0x8000, v6
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v1, v8, v7, vcc_lo
	v_and_b32_e32 v9, 0x7f80, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v1, 16, v1
	v_and_b32_e32 v6, 0x7f, v6
	v_cmp_ne_u32_e32 vcc_lo, 0x7f80, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_eq_u32_e64 s5, 0, v6
	s_or_b32 vcc_lo, vcc_lo, s5
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	v_or_b32_e32 v1, v33, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v6, 0xff9f, v1
.LBB_16:
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v1, 0x80, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s52, v1
	s_and_saveexec_b32 s7, s5
	s_cbranch_execz .LBB_18
; %bb.17:
	v_lshlrev_b32_e32 v1, 1, v33
	global_load_u16 v1, v1, s[10:11] offset:256
	s_wait_loadcnt 0x0
	v_and_b32_e32 v5, 0xffff, v1
	v_cmp_gt_i16_e32 vcc_lo, 0, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_not_b32_e32 v7, v5
	v_or_b32_e32 v8, 0x8000, v5
	v_and_b32_e32 v9, 0x7f80, v5
	v_and_b32_e32 v5, 0x7f, v5
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v1, v8, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f80, v9
	v_cmp_eq_u32_e64 s6, 0, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 16, v1
	s_or_b32 vcc_lo, vcc_lo, s6
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v1, v33, v1
	v_xor_b32_e32 v5, 0xff7f, v1
.LBB_18:
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v1, 0xa0, v33
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v8, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s6, s52, v1
	s_and_saveexec_b32 s8, s6
	s_cbranch_execz .LBB_20
; %bb.19:
	v_lshlrev_b32_e32 v1, 1, v33
	global_load_u16 v1, v1, s[10:11] offset:320
	s_wait_loadcnt 0x0
	v_and_b32_e32 v8, 0xffff, v1
	v_cmp_gt_i16_e32 vcc_lo, 0, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_not_b32_e32 v9, v8
	v_or_b32_e32 v10, 0x8000, v8
	v_and_b32_e32 v11, 0x7f80, v8
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v1, v10, v9 :: v_dual_and_b32 v8, 0x7f, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f80, v11
	v_cmp_eq_u32_e64 s7, 0, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 16, v1
	s_or_b32 vcc_lo, vcc_lo, s7
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v1, v33, v1
	v_xor_b32_e32 v8, 0xff5f, v1
.LBB_20:
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v1, 0xc0, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s52, v1
	s_and_saveexec_b32 s9, s7
	s_cbranch_execz .LBB_22
; %bb.21:
	v_lshlrev_b32_e32 v1, 1, v33
	global_load_u16 v1, v1, s[10:11] offset:384
	s_wait_loadcnt 0x0
	v_and_b32_e32 v7, 0xffff, v1
	v_cmp_gt_i16_e32 vcc_lo, 0, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_not_b32_e32 v9, v7
	v_or_b32_e32 v10, 0x8000, v7
	v_and_b32_e32 v11, 0x7f80, v7
	v_and_b32_e32 v7, 0x7f, v7
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v1, v10, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f80, v11
	v_cmp_eq_u32_e64 s8, 0, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 16, v1
	s_or_b32 vcc_lo, vcc_lo, s8
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v1, v33, v1
	v_xor_b32_e32 v7, 0xff3f, v1
.LBB_22:
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v9, 0xe0, v33
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s8, s52, v9
	v_mov_b32_e32 v9, 0
	s_and_saveexec_b32 s33, s8
	s_cbranch_execz .LBB_24
; %bb.23:
	v_lshlrev_b32_e32 v9, 1, v33
	global_load_u16 v9, v9, s[10:11] offset:448
	s_wait_loadcnt 0x0
	v_cmp_gt_i16_e32 vcc_lo, 0, v9
	v_and_b32_e32 v10, 0xffff, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_not_b32_e32 v11, v10
	v_or_b32_e32 v12, 0x8000, v10
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v9, v12, v11, vcc_lo
	v_and_b32_e32 v13, 0x7f80, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v9, 16, v9
	v_and_b32_e32 v10, 0x7f, v10
	v_cmp_ne_u32_e32 vcc_lo, 0x7f80, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_eq_u32_e64 s9, 0, v10
	s_or_b32 vcc_lo, vcc_lo, s9
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v9, 0, v9, vcc_lo
	v_or_b32_e32 v9, v33, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v9, 0xff1f, v9
.LBB_24:
	s_or_b32 exec_lo, exec_lo, s33
	s_cmp_gt_i32 s56, 0
	s_cselect_b32 s33, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s33
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB_36
; %bb.25:
	v_mbcnt_lo_u32_b32 v10, -1, 0
	v_max_u32_e32 v1, v2, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v11, 16, v10
	v_xor_b32_e32 v12, 8, v10
	v_max3_u32 v1, v1, v3, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v11
	v_max3_u32 v1, v1, v5, v8
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v11, v10, v11, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_max3_u32 v1, v1, v7, v9
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v12, v10, v12, vcc_lo
	v_lshlrev_b32_e32 v12, 2, v12
	v_lshlrev_b32_e32 v11, 2, v11
	ds_bpermute_b32 v11, v11, v1
	s_wait_dscnt 0x0
	v_max_u32_e32 v1, v1, v11
	ds_bpermute_b32 v11, v12, v1
	v_xor_b32_e32 v12, 4, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v12
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v12, v10, v12, vcc_lo
	v_lshlrev_b32_e32 v12, 2, v12
	s_wait_dscnt 0x0
	v_max_u32_e32 v1, v1, v11
	ds_bpermute_b32 v11, v12, v1
	v_xor_b32_e32 v12, 2, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v12
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v12, v10, v12, vcc_lo
	v_lshlrev_b32_e32 v12, 2, v12
	s_wait_dscnt 0x0
	v_max_u32_e32 v1, v1, v11
	ds_bpermute_b32 v11, v12, v1
	v_xor_b32_e32 v12, 1, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v12
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v10, v10, v12, vcc_lo
	v_lshlrev_b32_e32 v10, 2, v10
	s_wait_dscnt 0x0
	v_max_u32_e32 v1, v1, v11
	ds_bpermute_b32 v10, v10, v1
	s_wait_dscnt 0x0
	v_max_u32_e32 v1, v1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s9, v1
	s_and_not1_b32 s59, 31, s9
	s_and_not1_b32 s78, 0xffff, s9
	v_cmp_eq_u32_e32 vcc_lo, s59, v33
	s_and_saveexec_b32 s59, vcc_lo
	s_cbranch_execz .LBB_27
; %bb.26:
	s_lshr_b32 s72, s78, 5
	s_cmp_gt_u32 s78, 31
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s72, 1
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s72, 2
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s72, 3
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s72, 4
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s72, 5
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v5, 0, v5, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s72, 6
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s72, 7
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v9, 0, v9, vcc_lo
.LBB_27:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s59
	s_and_b32 s59, s9, 0xffff0000
	s_brev_b32 s72, 1
	s_cmp_gt_i32 s9, -1
	s_cselect_b32 s9, 0xffff0000, s72
	s_wait_alu 0xfffe
	s_xor_b32 s81, s9, s59
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_sub_f32 s9, s81, s81
	s_mul_f32 s59, s9, 0x3fb8aa3b
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_xor_b32 s72, s59, 0x80000000
	s_rndne_f32 s73, s59
	s_fmamk_f32 s72, s9, 0x3fb8aa3b, s72
	s_cmp_nlt_f32 s9, 0xc2ce8ed0
	s_sub_f32 s59, s59, s73
	s_delay_alu instid0(SALU_CYCLE_1)
	s_fmamk_f32 s72, s9, 0x32a5705f, s72
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s9, 0x42b17218
	s_wait_alu 0xfffe
	s_add_f32 s59, s59, s72
	s_cvt_i32_f32 s72, s73
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s59, s59
	s_wait_alu 0xf1ff
	v_ldexp_f32 v1, s59, s72
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v1, 0x7f800000, v1, vcc_lo
	s_cmp_gt_i32 s56, 1
	s_cselect_b32 s77, -1, 0
	s_cmp_lt_i32 s56, 2
	s_cbranch_scc1 .LBB_37
.LBB_28:
	v_mbcnt_lo_u32_b32 v11, -1, 0
	v_max_u32_e32 v10, v2, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v12, 16, v11
	v_xor_b32_e32 v13, 8, v11
	v_max3_u32 v10, v10, v3, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v12
	v_max3_u32 v10, v10, v5, v8
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v12, v11, v12, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_max3_u32 v10, v10, v7, v9
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v12, 2, v12
	ds_bpermute_b32 v12, v12, v10
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 4, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 2, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 1, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v11, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v11, 2, v11
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v11, v11, v10
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s9, v10
	s_and_not1_b32 s59, 31, s9
	s_and_not1_b32 s79, 0xffff, s9
	s_wait_alu 0xfffe
	v_cmp_eq_u32_e32 vcc_lo, s59, v33
	s_and_saveexec_b32 s59, vcc_lo
	s_cbranch_execz .LBB_30
; %bb.29:
	s_lshr_b32 s72, s79, 5
	s_cmp_gt_u32 s79, 31
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	s_cmp_lg_u32 s72, 1
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s72, 2
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s72, 3
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s72, 4
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s72, 5
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v5, 0, v5, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s72, 6
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s72, 7
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v9, 0, v9, vcc_lo
.LBB_30:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s59
	s_and_b32 s59, s9, 0xffff0000
	s_brev_b32 s72, 1
	s_cmp_gt_i32 s9, -1
	s_wait_alu 0xfffe
	s_cselect_b32 s9, 0xffff0000, s72
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_3)
	s_xor_b32 s59, s9, s59
	s_wait_alu 0xfffe
	s_sub_f32 s9, s59, s81
	s_mul_f32 s72, s9, 0x3fb8aa3b
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2)
	s_xor_b32 s73, s72, 0x80000000
	s_rndne_f32 s74, s72
	s_wait_alu 0xfffe
	s_fmamk_f32 s73, s9, 0x3fb8aa3b, s73
	s_cmp_nlt_f32 s9, 0xc2ce8ed0
	s_sub_f32 s72, s72, s74
	s_wait_alu 0xfffe
	s_fmamk_f32 s73, s9, 0x32a5705f, s73
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s9, 0x42b17218
	s_wait_alu 0xfffe
	s_add_f32 s72, s72, s73
	s_cvt_i32_f32 s73, s74
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s72, s72
	s_wait_alu 0xf1ff
	v_ldexp_f32 v10, s72, s73
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v10, 0, v10, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v10, 0x7f800000, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v10
	s_cmp_gt_i32 s56, 2
	s_cselect_b32 s76, -1, 0
	s_cmp_lt_i32 s56, 3
	s_cbranch_scc0 .LBB_38
.LBB_31:
	s_mov_b32 s80, -1
	s_mov_b32 s84, 0
	s_cmp_gt_i32 s56, 3
	s_cselect_b32 s75, -1, 0
	s_cmp_lt_i32 s56, 4
	s_cbranch_scc0 .LBB_41
.LBB_32:
	s_mov_b32 s82, -1
	s_mov_b32 s85, 0
	s_cmp_gt_i32 s56, 4
	s_cselect_b32 s74, -1, 0
	s_cmp_lt_i32 s56, 5
	s_cbranch_scc0 .LBB_44
.LBB_33:
	s_mov_b32 s83, -1
	s_mov_b32 s93, 0
	s_cmp_gt_i32 s56, 5
	s_cselect_b32 s73, -1, 0
	s_cmp_lt_i32 s56, 6
	s_cbranch_scc0 .LBB_47
.LBB_34:
	s_mov_b32 s89, -1
	s_mov_b32 s94, 0
	s_cmp_gt_i32 s56, 6
	s_cselect_b32 s72, -1, 0
	s_cmp_lt_i32 s56, 7
	s_cbranch_scc0 .LBB_50
.LBB_35:
	s_mov_b32 s91, -1
	s_mov_b32 s95, 0
	s_branch .LBB_53
.LBB_36:
	s_mov_b32 s78, -1
	s_mov_b32 s81, 0
	s_cmp_gt_i32 s56, 1
	s_cselect_b32 s77, -1, 0
	s_cmp_lt_i32 s56, 2
	s_cbranch_scc0 .LBB_28
.LBB_37:
	s_mov_b32 s59, 0
	s_mov_b32 s79, -1
	s_cmp_gt_i32 s56, 2
	s_cselect_b32 s76, -1, 0
	s_cmp_lt_i32 s56, 3
	s_cbranch_scc1 .LBB_31
.LBB_38:
	v_mbcnt_lo_u32_b32 v11, -1, 0
	v_max_u32_e32 v10, v2, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v12, 16, v11
	v_xor_b32_e32 v13, 8, v11
	v_max3_u32 v10, v10, v3, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v12
	v_max3_u32 v10, v10, v5, v8
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v12, v11, v12, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_max3_u32 v10, v10, v7, v9
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v12, 2, v12
	ds_bpermute_b32 v12, v12, v10
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 4, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 2, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 1, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v11, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v11, 2, v11
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v11, v11, v10
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s9, v10
	s_and_not1_b32 s72, 31, s9
	s_and_not1_b32 s80, 0xffff, s9
	s_wait_alu 0xfffe
	v_cmp_eq_u32_e32 vcc_lo, s72, v33
	s_and_saveexec_b32 s72, vcc_lo
	s_cbranch_execz .LBB_40
; %bb.39:
	s_lshr_b32 s73, s80, 5
	s_cmp_gt_u32 s80, 31
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	s_cmp_lg_u32 s73, 1
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 2
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 3
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 4
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 5
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v5, 0, v5, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 6
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 7
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v9, 0, v9, vcc_lo
.LBB_40:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s72
	s_and_b32 s72, s9, 0xffff0000
	s_brev_b32 s73, 1
	s_cmp_gt_i32 s9, -1
	s_wait_alu 0xfffe
	s_cselect_b32 s9, 0xffff0000, s73
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s84, s9, s72
	s_sub_f32 s9, s84, s81
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s72, s9, 0x3fb8aa3b
	s_wait_alu 0xfffe
	s_xor_b32 s73, s72, 0x80000000
	s_rndne_f32 s74, s72
	s_wait_alu 0xfffe
	s_fmamk_f32 s73, s9, 0x3fb8aa3b, s73
	s_cmp_nlt_f32 s9, 0xc2ce8ed0
	s_sub_f32 s72, s72, s74
	s_wait_alu 0xfffe
	s_fmamk_f32 s73, s9, 0x32a5705f, s73
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s9, 0x42b17218
	s_wait_alu 0xfffe
	s_add_f32 s72, s72, s73
	s_cvt_i32_f32 s73, s74
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s72, s72
	s_wait_alu 0xf1ff
	v_ldexp_f32 v10, s72, s73
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v10, 0, v10, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v10, 0x7f800000, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v10
	s_cmp_gt_i32 s56, 3
	s_cselect_b32 s75, -1, 0
	s_cmp_lt_i32 s56, 4
	s_cbranch_scc1 .LBB_32
.LBB_41:
	v_mbcnt_lo_u32_b32 v11, -1, 0
	v_max_u32_e32 v10, v2, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v12, 16, v11
	v_xor_b32_e32 v13, 8, v11
	v_max3_u32 v10, v10, v3, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v12
	v_max3_u32 v10, v10, v5, v8
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v12, v11, v12, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_max3_u32 v10, v10, v7, v9
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v12, 2, v12
	ds_bpermute_b32 v12, v12, v10
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 4, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 2, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 1, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v11, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v11, 2, v11
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v11, v11, v10
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s9, v10
	s_and_not1_b32 s72, 31, s9
	s_and_not1_b32 s82, 0xffff, s9
	s_wait_alu 0xfffe
	v_cmp_eq_u32_e32 vcc_lo, s72, v33
	s_and_saveexec_b32 s72, vcc_lo
	s_cbranch_execz .LBB_43
; %bb.42:
	s_lshr_b32 s73, s82, 5
	s_cmp_gt_u32 s82, 31
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	s_cmp_lg_u32 s73, 1
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 2
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 3
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 4
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 5
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v5, 0, v5, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 6
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 7
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v9, 0, v9, vcc_lo
.LBB_43:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s72
	s_and_b32 s72, s9, 0xffff0000
	s_brev_b32 s73, 1
	s_cmp_gt_i32 s9, -1
	s_wait_alu 0xfffe
	s_cselect_b32 s9, 0xffff0000, s73
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s85, s9, s72
	s_sub_f32 s9, s85, s81
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s72, s9, 0x3fb8aa3b
	s_wait_alu 0xfffe
	s_xor_b32 s73, s72, 0x80000000
	s_rndne_f32 s74, s72
	s_wait_alu 0xfffe
	s_fmamk_f32 s73, s9, 0x3fb8aa3b, s73
	s_cmp_nlt_f32 s9, 0xc2ce8ed0
	s_sub_f32 s72, s72, s74
	s_wait_alu 0xfffe
	s_fmamk_f32 s73, s9, 0x32a5705f, s73
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s9, 0x42b17218
	s_wait_alu 0xfffe
	s_add_f32 s72, s72, s73
	s_cvt_i32_f32 s73, s74
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s72, s72
	s_wait_alu 0xf1ff
	v_ldexp_f32 v10, s72, s73
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v10, 0, v10, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v10, 0x7f800000, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v10
	s_cmp_gt_i32 s56, 4
	s_cselect_b32 s74, -1, 0
	s_cmp_lt_i32 s56, 5
	s_cbranch_scc1 .LBB_33
.LBB_44:
	v_mbcnt_lo_u32_b32 v11, -1, 0
	v_max_u32_e32 v10, v2, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v12, 16, v11
	v_xor_b32_e32 v13, 8, v11
	v_max3_u32 v10, v10, v3, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v12
	v_max3_u32 v10, v10, v5, v8
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v12, v11, v12, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_max3_u32 v10, v10, v7, v9
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v12, 2, v12
	ds_bpermute_b32 v12, v12, v10
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 4, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 2, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 1, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v11, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v11, 2, v11
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v11, v11, v10
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s9, v10
	s_and_not1_b32 s72, 31, s9
	s_and_not1_b32 s83, 0xffff, s9
	s_wait_alu 0xfffe
	v_cmp_eq_u32_e32 vcc_lo, s72, v33
	s_and_saveexec_b32 s72, vcc_lo
	s_cbranch_execz .LBB_46
; %bb.45:
	s_lshr_b32 s73, s83, 5
	s_cmp_gt_u32 s83, 31
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	s_cmp_lg_u32 s73, 1
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 2
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 3
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 4
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 5
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v5, 0, v5, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 6
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s73, 7
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v9, 0, v9, vcc_lo
.LBB_46:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s72
	s_and_b32 s72, s9, 0xffff0000
	s_brev_b32 s73, 1
	s_cmp_gt_i32 s9, -1
	s_wait_alu 0xfffe
	s_cselect_b32 s9, 0xffff0000, s73
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s93, s9, s72
	s_sub_f32 s9, s93, s81
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s72, s9, 0x3fb8aa3b
	s_wait_alu 0xfffe
	s_xor_b32 s73, s72, 0x80000000
	s_rndne_f32 s86, s72
	s_wait_alu 0xfffe
	s_fmamk_f32 s73, s9, 0x3fb8aa3b, s73
	s_cmp_nlt_f32 s9, 0xc2ce8ed0
	s_sub_f32 s72, s72, s86
	s_wait_alu 0xfffe
	s_fmamk_f32 s73, s9, 0x32a5705f, s73
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s9, 0x42b17218
	s_wait_alu 0xfffe
	s_add_f32 s72, s72, s73
	s_cvt_i32_f32 s73, s86
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s72, s72
	s_wait_alu 0xf1ff
	v_ldexp_f32 v10, s72, s73
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v10, 0, v10, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v10, 0x7f800000, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v10
	s_cmp_gt_i32 s56, 5
	s_cselect_b32 s73, -1, 0
	s_cmp_lt_i32 s56, 6
	s_cbranch_scc1 .LBB_34
.LBB_47:
	v_mbcnt_lo_u32_b32 v11, -1, 0
	v_max_u32_e32 v10, v2, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v12, 16, v11
	v_xor_b32_e32 v13, 8, v11
	v_max3_u32 v10, v10, v3, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v12
	v_max3_u32 v10, v10, v5, v8
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v12, v11, v12, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_max3_u32 v10, v10, v7, v9
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v12, 2, v12
	ds_bpermute_b32 v12, v12, v10
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 4, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 2, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 1, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v11, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v11, 2, v11
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v11, v11, v10
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s9, v10
	s_and_not1_b32 s72, 31, s9
	s_and_not1_b32 s89, 0xffff, s9
	s_wait_alu 0xfffe
	v_cmp_eq_u32_e32 vcc_lo, s72, v33
	s_and_saveexec_b32 s72, vcc_lo
	s_cbranch_execz .LBB_49
; %bb.48:
	s_lshr_b32 s86, s89, 5
	s_cmp_gt_u32 s89, 31
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s86, 1
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s86, 2
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s86, 3
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s86, 4
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s86, 5
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v5, 0, v5, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s86, 6
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s86, 7
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v9, 0, v9, vcc_lo
.LBB_49:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s72
	s_and_b32 s72, s9, 0xffff0000
	s_brev_b32 s86, 1
	s_cmp_gt_i32 s9, -1
	s_cselect_b32 s9, 0xffff0000, s86
	s_wait_alu 0xfffe
	s_xor_b32 s94, s9, s72
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_sub_f32 s9, s94, s81
	s_mul_f32 s72, s9, 0x3fb8aa3b
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_xor_b32 s86, s72, 0x80000000
	s_rndne_f32 s87, s72
	s_fmamk_f32 s86, s9, 0x3fb8aa3b, s86
	s_cmp_nlt_f32 s9, 0xc2ce8ed0
	s_sub_f32 s72, s72, s87
	s_delay_alu instid0(SALU_CYCLE_1)
	s_fmamk_f32 s86, s9, 0x32a5705f, s86
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s9, 0x42b17218
	s_wait_alu 0xfffe
	s_add_f32 s72, s72, s86
	s_cvt_i32_f32 s86, s87
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s72, s72
	s_wait_alu 0xf1ff
	v_ldexp_f32 v10, s72, s86
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v10, 0, v10, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v10, 0x7f800000, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v10
	s_cmp_gt_i32 s56, 6
	s_cselect_b32 s72, -1, 0
	s_cmp_lt_i32 s56, 7
	s_cbranch_scc1 .LBB_35
.LBB_50:
	v_mbcnt_lo_u32_b32 v11, -1, 0
	v_max_u32_e32 v10, v2, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v12, 16, v11
	v_xor_b32_e32 v13, 8, v11
	v_max3_u32 v10, v10, v3, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v12
	v_max3_u32 v10, v10, v5, v8
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v12, v11, v12, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_max3_u32 v10, v10, v7, v9
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v12, 2, v12
	ds_bpermute_b32 v12, v12, v10
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 4, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 2, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v13, 2, v13
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v12, v13, v10
	v_xor_b32_e32 v13, 1, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v11, v11, v13, vcc_lo
	v_lshlrev_b32_e32 v11, 2, v11
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v12
	ds_bpermute_b32 v11, v11, v10
	s_wait_dscnt 0x0
	v_max_u32_e32 v10, v10, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s9, v10
	s_and_not1_b32 s86, 31, s9
	s_and_not1_b32 s91, 0xffff, s9
	s_wait_alu 0xfffe
	v_cmp_eq_u32_e32 vcc_lo, s86, v33
	s_and_saveexec_b32 s86, vcc_lo
	s_cbranch_execz .LBB_52
; %bb.51:
	s_lshr_b32 s87, s91, 5
	s_cmp_gt_u32 s91, 31
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	s_cmp_lg_u32 s87, 1
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s87, 2
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s87, 3
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s87, 4
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s87, 5
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v5, 0, v5, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s87, 6
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s87, 7
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v9, 0, v9, vcc_lo
.LBB_52:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s86
	s_and_b32 s86, s9, 0xffff0000
	s_brev_b32 s87, 1
	s_cmp_gt_i32 s9, -1
	s_wait_alu 0xfffe
	s_cselect_b32 s9, 0xffff0000, s87
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s95, s9, s86
	s_sub_f32 s9, s95, s81
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s86, s9, 0x3fb8aa3b
	s_wait_alu 0xfffe
	s_xor_b32 s87, s86, 0x80000000
	s_rndne_f32 s88, s86
	s_wait_alu 0xfffe
	s_fmamk_f32 s87, s9, 0x3fb8aa3b, s87
	s_cmp_nlt_f32 s9, 0xc2ce8ed0
	s_sub_f32 s86, s86, s88
	s_wait_alu 0xfffe
	s_fmamk_f32 s87, s9, 0x32a5705f, s87
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s9, 0x42b17218
	s_wait_alu 0xfffe
	s_add_f32 s86, s86, s87
	s_cvt_i32_f32 s87, s88
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s86, s86
	s_wait_alu 0xf1ff
	v_ldexp_f32 v10, s86, s87
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v10, 0, v10, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v10, 0x7f800000, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v10
.LBB_53:
	s_cmp_gt_i32 s56, 7
	s_mov_b32 s96, 0
	s_cselect_b32 s9, -1, 0
	s_cmp_lt_i32 s56, 8
	s_mov_b32 s92, -1
	s_cbranch_scc1 .LBB_63
; %bb.54:
	v_max_u32_e32 v2, v2, v4
	v_mbcnt_lo_u32_b32 v4, -1, 0
	s_brev_b32 s87, 1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_u32 v2, v2, v3, v6
	v_xor_b32_e32 v3, 16, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_u32 v2, v2, v5, v8
	v_cmp_gt_i32_e32 vcc_lo, 32, v3
	v_xor_b32_e32 v5, 8, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_max3_u32 v2, v2, v7, v9
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v4, v3, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v5
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v3, 2, v3
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v5, v4, v5, vcc_lo
	ds_bpermute_b32 v3, v3, v2
	v_lshlrev_b32_e32 v5, 2, v5
	s_wait_dscnt 0x0
	v_max_u32_e32 v2, v2, v3
	ds_bpermute_b32 v3, v5, v2
	v_xor_b32_e32 v5, 4, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v5
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v5, v4, v5, vcc_lo
	v_lshlrev_b32_e32 v5, 2, v5
	s_wait_dscnt 0x0
	v_max_u32_e32 v2, v2, v3
	ds_bpermute_b32 v3, v5, v2
	v_xor_b32_e32 v5, 2, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v5
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v5, v4, v5, vcc_lo
	v_lshlrev_b32_e32 v5, 2, v5
	s_wait_dscnt 0x0
	v_max_u32_e32 v2, v2, v3
	ds_bpermute_b32 v3, v5, v2
	v_xor_b32_e32 v5, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v5
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo
	v_lshlrev_b32_e32 v4, 2, v4
	s_wait_dscnt 0x0
	v_max_u32_e32 v2, v2, v3
	ds_bpermute_b32 v3, v4, v2
	s_wait_dscnt 0x0
	v_max_u32_e32 v2, v2, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s86, v2
	s_and_not1_b32 s92, 0xffff, s86
	s_and_b32 s88, s86, 0xffff0000
	s_cmp_gt_i32 s86, -1
	s_wait_alu 0xfffe
	s_cselect_b32 s86, 0xffff0000, s87
	s_wait_alu 0xfffe
	s_xor_b32 s96, s86, s88
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_sub_f32 s86, s96, s81
	s_wait_alu 0xfffe
	s_mul_f32 s87, s86, 0x3fb8aa3b
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_xor_b32 s88, s87, 0x80000000
	s_rndne_f32 s90, s87
	s_fmamk_f32 s88, s86, 0x3fb8aa3b, s88
	s_cmp_nlt_f32 s86, 0xc2ce8ed0
	s_sub_f32 s87, s87, s90
	s_delay_alu instid0(SALU_CYCLE_1)
	s_fmamk_f32 s88, s86, 0x32a5705f, s88
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s86, 0x42b17218
	s_wait_alu 0xfffe
	s_add_f32 s87, s87, s88
	s_cvt_i32_f32 s88, s90
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s87, s87
	s_wait_alu 0xf1ff
	v_ldexp_f32 v2, s87, s88
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v2, 0x7f800000, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v2
	s_cmp_eq_u32 s57, 0
	s_cbranch_scc0 .LBB_64
.LBB_55:
	v_mov_b32_e32 v1, 0
	s_and_saveexec_b32 s57, s0
	s_cbranch_execz .LBB_88
; %bb.56:
	v_lshlrev_b32_e32 v1, 1, v33
	global_load_u16 v1, v1, s[10:11]
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v1, 16, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_subrev_f32_e32 v1, s81, v1
	v_mul_f32_e32 v2, 0x3fb8aa3b, v1
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v3, 0x3fb8aa3b, v1, -v2
	v_rndne_f32_e32 v4, v2
	v_dual_fmamk_f32 v3, v1, 0x32a5705f, v3 :: v_dual_sub_f32 v2, v2, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v2, v2, v3
	v_cvt_i32_f32_e32 v3, v4
	v_exp_f32_e32 v2, v2
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v2, v2, v3
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v1
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v1, 0x7f800000, v2, vcc_lo
	s_or_b32 exec_lo, exec_lo, s57
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB_89
.LBB_57:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s3
	s_cbranch_execz .LBB_90
.LBB_58:
	v_lshlrev_b32_e32 v2, 1, v33
	global_load_u16 v2, v2, s[10:11] offset:128
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v2, 16, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_subrev_f32_e32 v2, s81, v2
	v_mul_f32_e32 v3, 0x3fb8aa3b, v2
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v4, 0x3fb8aa3b, v2, -v3
	v_rndne_f32_e32 v5, v3
	v_dual_fmamk_f32 v4, v2, 0x32a5705f, v4 :: v_dual_sub_f32 v3, v3, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v3, v4
	v_cvt_i32_f32_e32 v4, v5
	v_exp_f32_e32 v3, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v3, v3, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0x7f800000, v3, vcc_lo
	v_add_f32_e32 v1, v1, v2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s4
	s_cbranch_execnz .LBB_91
.LBB_59:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s5
	s_cbranch_execz .LBB_92
.LBB_60:
	v_lshlrev_b32_e32 v2, 1, v33
	global_load_u16 v2, v2, s[10:11] offset:256
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v2, 16, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_subrev_f32_e32 v2, s81, v2
	v_mul_f32_e32 v3, 0x3fb8aa3b, v2
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v4, 0x3fb8aa3b, v2, -v3
	v_rndne_f32_e32 v5, v3
	v_dual_fmamk_f32 v4, v2, 0x32a5705f, v4 :: v_dual_sub_f32 v3, v3, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v3, v4
	v_cvt_i32_f32_e32 v4, v5
	v_exp_f32_e32 v3, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v3, v3, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0x7f800000, v3, vcc_lo
	v_add_f32_e32 v1, v1, v2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s6
	s_cbranch_execnz .LBB_93
.LBB_61:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s7
	s_cbranch_execz .LBB_94
.LBB_62:
	v_lshlrev_b32_e32 v2, 1, v33
	global_load_u16 v2, v2, s[10:11] offset:384
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v2, 16, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_subrev_f32_e32 v2, s81, v2
	v_mul_f32_e32 v3, 0x3fb8aa3b, v2
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v4, 0x3fb8aa3b, v2, -v3
	v_rndne_f32_e32 v5, v3
	v_dual_fmamk_f32 v4, v2, 0x32a5705f, v4 :: v_dual_sub_f32 v3, v3, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v3, v4
	v_cvt_i32_f32_e32 v4, v5
	v_exp_f32_e32 v3, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v3, v3, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0x7f800000, v3, vcc_lo
	v_add_f32_e32 v1, v1, v2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s8
	s_cbranch_execnz .LBB_95
	s_branch .LBB_96
.LBB_63:
	s_cmp_eq_u32 s57, 0
	s_cbranch_scc1 .LBB_55
.LBB_64:
	s_and_not1_b32 vcc_lo, exec_lo, s33
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_97
.LBB_65:
	s_sub_f32 s0, s81, s81
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s1, s0, 0x3fb8aa3b
	s_wait_alu 0xfffe
	s_xor_b32 s3, s1, 0x80000000
	s_rndne_f32 s4, s1
	s_fmamk_f32 s3, s0, 0x3fb8aa3b, s3
	s_cmp_nlt_f32 s0, 0xc2ce8ed0
	s_wait_alu 0xfffe
	s_sub_f32 s1, s1, s4
	s_fmamk_f32 s3, s0, 0x32a5705f, s3
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s0, 0x42b17218
	s_wait_alu 0xfffe
	s_add_f32 s1, s1, s3
	s_cvt_i32_f32 s3, s4
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s1, s1
	s_wait_alu 0xf1ff
	v_ldexp_f32 v2, s1, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v2, 0x7f800000, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v3, null, v1, v1, v2
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v5, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v5, v4
	v_div_scale_f32 v5, vcc_lo, v2, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v6, v5, v4
	v_fma_f32 v7, -v3, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v7, v4
	v_fma_f32 v3, -v3, v6, v5
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v6
	v_div_fixup_f32 v3, v3, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0x7fffffff, v3
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.66:
	v_bfe_u32 v2, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v2, v3, v2, 0x7fff
                                        ; implicit-def: $vgpr3
	v_lshrrev_b32_e32 v2, 16, v2
; %bb.67:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.68:
	v_lshrrev_b32_e32 v2, 16, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, 64, v2
; %bb.69:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s0, v2
	s_lshl_b32 s90, s0, 16
	s_and_not1_b32 vcc_lo, exec_lo, s77
	s_mov_b32 s88, 0
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_98
.LBB_70:
	s_sub_f32 s0, s59, s81
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s1, s0, 0x3fb8aa3b
	s_wait_alu 0xfffe
	s_xor_b32 s3, s1, 0x80000000
	s_rndne_f32 s4, s1
	s_wait_alu 0xfffe
	s_fmamk_f32 s3, s0, 0x3fb8aa3b, s3
	s_cmp_nlt_f32 s0, 0xc2ce8ed0
	s_sub_f32 s1, s1, s4
	s_wait_alu 0xfffe
	s_fmamk_f32 s3, s0, 0x32a5705f, s3
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s0, 0x42b17218
	s_wait_alu 0xfffe
	s_add_f32 s1, s1, s3
	s_cvt_i32_f32 s3, s4
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s1, s1
	s_wait_alu 0xf1ff
	v_ldexp_f32 v2, s1, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v2, 0x7f800000, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v3, null, v1, v1, v2
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v5, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v5, v4
	v_div_scale_f32 v5, vcc_lo, v2, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v6, v5, v4
	v_fma_f32 v7, -v3, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v7, v4
	v_fma_f32 v3, -v3, v6, v5
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v6
	v_div_fixup_f32 v3, v3, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0x7fffffff, v3
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.71:
	v_bfe_u32 v2, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v2, v3, v2, 0x7fff
                                        ; implicit-def: $vgpr3
	v_lshrrev_b32_e32 v2, 16, v2
; %bb.72:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.73:
	v_lshrrev_b32_e32 v2, 16, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, 64, v2
; %bb.74:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s0, v2
	s_lshl_b32 s88, s0, 16
	s_and_not1_b32 vcc_lo, exec_lo, s76
	s_mov_b32 s87, 0
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB_99
.LBB_75:
	s_and_not1_b32 vcc_lo, exec_lo, s75
	s_mov_b32 s86, 0
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_104
.LBB_76:
	s_sub_f32 s0, s85, s81
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s1, s0, 0x3fb8aa3b
	s_wait_alu 0xfffe
	s_xor_b32 s3, s1, 0x80000000
	s_rndne_f32 s4, s1
	s_wait_alu 0xfffe
	s_fmamk_f32 s3, s0, 0x3fb8aa3b, s3
	s_cmp_nlt_f32 s0, 0xc2ce8ed0
	s_sub_f32 s1, s1, s4
	s_wait_alu 0xfffe
	s_fmamk_f32 s3, s0, 0x32a5705f, s3
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s0, 0x42b17218
	s_wait_alu 0xfffe
	s_add_f32 s1, s1, s3
	s_cvt_i32_f32 s3, s4
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s1, s1
	s_wait_alu 0xf1ff
	v_ldexp_f32 v2, s1, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v2, 0x7f800000, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v3, null, v1, v1, v2
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v5, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v5, v4
	v_div_scale_f32 v5, vcc_lo, v2, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v6, v5, v4
	v_fma_f32 v7, -v3, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v7, v4
	v_fma_f32 v3, -v3, v6, v5
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v6
	v_div_fixup_f32 v3, v3, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0x7fffffff, v3
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.77:
	v_bfe_u32 v2, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v2, v3, v2, 0x7fff
                                        ; implicit-def: $vgpr3
	v_lshrrev_b32_e32 v2, 16, v2
; %bb.78:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.79:
	v_lshrrev_b32_e32 v2, 16, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, 64, v2
; %bb.80:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s0, v2
	s_lshl_b32 s86, s0, 16
	s_and_not1_b32 vcc_lo, exec_lo, s74
	s_mov_b32 s85, 0
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB_105
.LBB_81:
	s_and_not1_b32 vcc_lo, exec_lo, s73
	s_mov_b32 s84, 0
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_110
.LBB_82:
	s_sub_f32 s0, s94, s81
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s1, s0, 0x3fb8aa3b
	s_wait_alu 0xfffe
	s_xor_b32 s3, s1, 0x80000000
	s_rndne_f32 s4, s1
	s_wait_alu 0xfffe
	s_fmamk_f32 s3, s0, 0x3fb8aa3b, s3
	s_cmp_nlt_f32 s0, 0xc2ce8ed0
	s_sub_f32 s1, s1, s4
	s_wait_alu 0xfffe
	s_fmamk_f32 s3, s0, 0x32a5705f, s3
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s0, 0x42b17218
	s_wait_alu 0xfffe
	s_add_f32 s1, s1, s3
	s_cvt_i32_f32 s3, s4
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s1, s1
	s_wait_alu 0xf1ff
	v_ldexp_f32 v2, s1, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v2, 0x7f800000, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v3, null, v1, v1, v2
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v5, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v5, v4
	v_div_scale_f32 v5, vcc_lo, v2, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v6, v5, v4
	v_fma_f32 v7, -v3, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v7, v4
	v_fma_f32 v3, -v3, v6, v5
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v6
	v_div_fixup_f32 v3, v3, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0x7fffffff, v3
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.83:
	v_bfe_u32 v2, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v2, v3, v2, 0x7fff
                                        ; implicit-def: $vgpr3
	v_lshrrev_b32_e32 v2, 16, v2
; %bb.84:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.85:
	v_lshrrev_b32_e32 v2, 16, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, 64, v2
; %bb.86:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s0, v2
	s_lshl_b32 s84, s0, 16
	s_and_not1_b32 vcc_lo, exec_lo, s72
	s_mov_b32 s59, 0
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB_111
.LBB_87:
	s_and_not1_b32 vcc_lo, exec_lo, s9
	s_mov_b32 s33, 0
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB_116
	s_branch .LBB_121
.LBB_88:
	s_or_b32 exec_lo, exec_lo, s57
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB_57
.LBB_89:
	v_lshlrev_b32_e32 v2, 1, v33
	global_load_u16 v2, v2, s[10:11] offset:64
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v2, 16, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_subrev_f32_e32 v2, s81, v2
	v_mul_f32_e32 v3, 0x3fb8aa3b, v2
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v4, 0x3fb8aa3b, v2, -v3
	v_rndne_f32_e32 v5, v3
	v_dual_fmamk_f32 v4, v2, 0x32a5705f, v4 :: v_dual_sub_f32 v3, v3, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v3, v4
	v_cvt_i32_f32_e32 v4, v5
	v_exp_f32_e32 v3, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v3, v3, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0x7f800000, v3, vcc_lo
	v_add_f32_e32 v1, v1, v2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s3
	s_cbranch_execnz .LBB_58
.LBB_90:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB_59
.LBB_91:
	v_lshlrev_b32_e32 v2, 1, v33
	global_load_u16 v2, v2, s[10:11] offset:192
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v2, 16, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_subrev_f32_e32 v2, s81, v2
	v_mul_f32_e32 v3, 0x3fb8aa3b, v2
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v4, 0x3fb8aa3b, v2, -v3
	v_rndne_f32_e32 v5, v3
	v_dual_fmamk_f32 v4, v2, 0x32a5705f, v4 :: v_dual_sub_f32 v3, v3, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v3, v4
	v_cvt_i32_f32_e32 v4, v5
	v_exp_f32_e32 v3, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v3, v3, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0x7f800000, v3, vcc_lo
	v_add_f32_e32 v1, v1, v2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s5
	s_cbranch_execnz .LBB_60
.LBB_92:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s6
	s_cbranch_execz .LBB_61
.LBB_93:
	v_lshlrev_b32_e32 v2, 1, v33
	global_load_u16 v2, v2, s[10:11] offset:320
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v2, 16, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_subrev_f32_e32 v2, s81, v2
	v_mul_f32_e32 v3, 0x3fb8aa3b, v2
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v4, 0x3fb8aa3b, v2, -v3
	v_rndne_f32_e32 v5, v3
	v_dual_fmamk_f32 v4, v2, 0x32a5705f, v4 :: v_dual_sub_f32 v3, v3, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v3, v4
	v_cvt_i32_f32_e32 v4, v5
	v_exp_f32_e32 v3, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v3, v3, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0x7f800000, v3, vcc_lo
	v_add_f32_e32 v1, v1, v2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s7
	s_cbranch_execnz .LBB_62
.LBB_94:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s8
	s_cbranch_execz .LBB_96
.LBB_95:
	v_lshlrev_b32_e32 v2, 1, v33
	global_load_u16 v2, v2, s[10:11] offset:448
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v2, 16, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_subrev_f32_e32 v2, s81, v2
	v_mul_f32_e32 v3, 0x3fb8aa3b, v2
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v4, 0x3fb8aa3b, v2, -v3
	v_rndne_f32_e32 v5, v3
	v_dual_fmamk_f32 v4, v2, 0x32a5705f, v4 :: v_dual_sub_f32 v3, v3, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v3, v4
	v_cvt_i32_f32_e32 v4, v5
	v_exp_f32_e32 v3, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v3, v3, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0x7f800000, v3, vcc_lo
	v_add_f32_e32 v1, v1, v2
.LBB_96:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_mbcnt_lo_u32_b32 v2, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v3, 16, v2
	v_xor_b32_e32 v4, 8, v2
	v_cmp_gt_i32_e32 vcc_lo, 32, v3
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v2, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v4, v2, v4, vcc_lo
	v_lshlrev_b32_e32 v4, 2, v4
	v_lshlrev_b32_e32 v3, 2, v3
	ds_bpermute_b32 v3, v3, v1
	s_wait_dscnt 0x0
	v_add_f32_e32 v1, v1, v3
	ds_bpermute_b32 v3, v4, v1
	v_xor_b32_e32 v4, 4, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v4, v2, v4, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v1, v1, v3 :: v_dual_lshlrev_b32 v4, 2, v4
	ds_bpermute_b32 v3, v4, v1
	v_xor_b32_e32 v4, 2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v4
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v4, v2, v4 :: v_dual_add_f32 v1, v1, v3
	v_lshlrev_b32_e32 v4, 2, v4
	ds_bpermute_b32 v3, v4, v1
	v_xor_b32_e32 v4, 1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v2, v2, v4, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v1, v1, v3 :: v_dual_lshlrev_b32 v2, 2, v2
	ds_bpermute_b32 v2, v2, v1
	s_wait_dscnt 0x0
	v_add_f32_e32 v1, v1, v2
	s_and_not1_b32 vcc_lo, exec_lo, s33
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB_65
.LBB_97:
	s_mov_b32 s90, 0
	s_and_not1_b32 vcc_lo, exec_lo, s77
	s_mov_b32 s88, 0
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB_70
.LBB_98:
	s_and_not1_b32 vcc_lo, exec_lo, s76
	s_mov_b32 s87, 0
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_75
.LBB_99:
	s_sub_f32 s0, s84, s81
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s1, s0, 0x3fb8aa3b
	s_wait_alu 0xfffe
	s_xor_b32 s3, s1, 0x80000000
	s_rndne_f32 s4, s1
	s_wait_alu 0xfffe
	s_fmamk_f32 s3, s0, 0x3fb8aa3b, s3
	s_cmp_nlt_f32 s0, 0xc2ce8ed0
	s_sub_f32 s1, s1, s4
	s_wait_alu 0xfffe
	s_fmamk_f32 s3, s0, 0x32a5705f, s3
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s0, 0x42b17218
	s_wait_alu 0xfffe
	s_add_f32 s1, s1, s3
	s_cvt_i32_f32 s3, s4
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s1, s1
	s_wait_alu 0xf1ff
	v_ldexp_f32 v2, s1, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v2, 0x7f800000, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v3, null, v1, v1, v2
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v5, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v5, v4
	v_div_scale_f32 v5, vcc_lo, v2, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v6, v5, v4
	v_fma_f32 v7, -v3, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v7, v4
	v_fma_f32 v3, -v3, v6, v5
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v6
	v_div_fixup_f32 v3, v3, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0x7fffffff, v3
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.100:
	v_bfe_u32 v2, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v2, v3, v2, 0x7fff
                                        ; implicit-def: $vgpr3
	v_lshrrev_b32_e32 v2, 16, v2
; %bb.101:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.102:
	v_lshrrev_b32_e32 v2, 16, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, 64, v2
; %bb.103:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s0, v2
	s_lshl_b32 s87, s0, 16
	s_and_not1_b32 vcc_lo, exec_lo, s75
	s_mov_b32 s86, 0
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB_76
.LBB_104:
	s_and_not1_b32 vcc_lo, exec_lo, s74
	s_mov_b32 s85, 0
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_81
.LBB_105:
	s_sub_f32 s0, s93, s81
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s1, s0, 0x3fb8aa3b
	s_wait_alu 0xfffe
	s_xor_b32 s3, s1, 0x80000000
	s_rndne_f32 s4, s1
	s_wait_alu 0xfffe
	s_fmamk_f32 s3, s0, 0x3fb8aa3b, s3
	s_cmp_nlt_f32 s0, 0xc2ce8ed0
	s_sub_f32 s1, s1, s4
	s_wait_alu 0xfffe
	s_fmamk_f32 s3, s0, 0x32a5705f, s3
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s0, 0x42b17218
	s_wait_alu 0xfffe
	s_add_f32 s1, s1, s3
	s_cvt_i32_f32 s3, s4
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s1, s1
	s_wait_alu 0xf1ff
	v_ldexp_f32 v2, s1, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v2, 0x7f800000, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v3, null, v1, v1, v2
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v5, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v5, v4
	v_div_scale_f32 v5, vcc_lo, v2, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v6, v5, v4
	v_fma_f32 v7, -v3, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v7, v4
	v_fma_f32 v3, -v3, v6, v5
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v6
	v_div_fixup_f32 v3, v3, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0x7fffffff, v3
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.106:
	v_bfe_u32 v2, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v2, v3, v2, 0x7fff
                                        ; implicit-def: $vgpr3
	v_lshrrev_b32_e32 v2, 16, v2
; %bb.107:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.108:
	v_lshrrev_b32_e32 v2, 16, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, 64, v2
; %bb.109:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s0, v2
	s_lshl_b32 s85, s0, 16
	s_and_not1_b32 vcc_lo, exec_lo, s73
	s_mov_b32 s84, 0
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB_82
.LBB_110:
	s_and_not1_b32 vcc_lo, exec_lo, s72
	s_mov_b32 s59, 0
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_87
.LBB_111:
	s_sub_f32 s0, s95, s81
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s1, s0, 0x3fb8aa3b
	s_wait_alu 0xfffe
	s_xor_b32 s3, s1, 0x80000000
	s_rndne_f32 s4, s1
	s_wait_alu 0xfffe
	s_fmamk_f32 s3, s0, 0x3fb8aa3b, s3
	s_cmp_nlt_f32 s0, 0xc2ce8ed0
	s_sub_f32 s1, s1, s4
	s_wait_alu 0xfffe
	s_fmamk_f32 s3, s0, 0x32a5705f, s3
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s0, 0x42b17218
	s_wait_alu 0xfffe
	s_add_f32 s1, s1, s3
	s_cvt_i32_f32 s3, s4
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s1, s1
	s_wait_alu 0xf1ff
	v_ldexp_f32 v2, s1, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v2, 0x7f800000, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v3, null, v1, v1, v2
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v5, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v5, v4
	v_div_scale_f32 v5, vcc_lo, v2, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v6, v5, v4
	v_fma_f32 v7, -v3, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v7, v4
	v_fma_f32 v3, -v3, v6, v5
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v6
	v_div_fixup_f32 v3, v3, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0x7fffffff, v3
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.112:
	v_bfe_u32 v2, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v2, v3, v2, 0x7fff
                                        ; implicit-def: $vgpr3
	v_lshrrev_b32_e32 v2, 16, v2
; %bb.113:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.114:
	v_lshrrev_b32_e32 v2, 16, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, 64, v2
; %bb.115:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s0, v2
	s_lshl_b32 s59, s0, 16
	s_and_not1_b32 vcc_lo, exec_lo, s9
	s_mov_b32 s33, 0
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_121
.LBB_116:
	s_sub_f32 s0, s96, s81
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s1, s0, 0x3fb8aa3b
	s_wait_alu 0xfffe
	s_xor_b32 s3, s1, 0x80000000
	s_rndne_f32 s4, s1
	s_wait_alu 0xfffe
	s_fmamk_f32 s3, s0, 0x3fb8aa3b, s3
	s_cmp_nlt_f32 s0, 0xc2ce8ed0
	s_sub_f32 s1, s1, s4
	s_wait_alu 0xfffe
	s_fmamk_f32 s3, s0, 0x32a5705f, s3
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s0, 0x42b17218
	s_wait_alu 0xfffe
	s_add_f32 s1, s1, s3
	s_cvt_i32_f32 s3, s4
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s1, s1
	s_wait_alu 0xf1ff
	v_ldexp_f32 v2, s1, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v2, 0x7f800000, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v3, null, v1, v1, v2
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v5, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v5, v4
	v_div_scale_f32 v5, vcc_lo, v2, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v6, v5, v4
	v_fma_f32 v7, -v3, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v7, v4
	v_fma_f32 v3, -v3, v6, v5
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v6
	v_div_fixup_f32 v2, v3, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v1, 0x7fffffff, v2
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.117:
	v_bfe_u32 v1, v2, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v1, v2, v1, 0x7fff
                                        ; implicit-def: $vgpr2
	v_lshrrev_b32_e32 v1, 16, v1
; %bb.118:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.119:
	v_lshrrev_b32_e32 v1, 16, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v1, 64, v1
; %bb.120:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s0, v1
	s_lshl_b32 s33, s0, 16
.LBB_121:
	s_mul_i32 s0, s68, s70
	s_ashr_i32 s11, s54, 31
	s_wait_alu 0xfffe
	s_sub_co_i32 s1, s71, s0
	s_lshr_b32 s0, s11, 28
	s_wait_alu 0xfffe
	v_lshl_or_b32 v1, s1, 3, v61
	s_lshr_b32 s3, s54, 31
	s_add_co_i32 s0, s54, s0
	s_wait_alu 0xfffe
	s_add_co_i32 s3, s54, s3
	s_ashr_i32 s93, s0, 4
	v_cmp_gt_i32_e64 s1, s53, v1
	s_wait_alu 0xfffe
	s_ashr_i32 s0, s3, 1
	s_ashr_i32 s71, s53, 31
	s_wait_alu 0xfffe
	s_ashr_i32 s6, s0, 31
	s_ashr_i32 s3, s93, 31
	s_cmp_gt_i32 s78, -1
	v_cndmask_b32_e64 v2, 0, v1, s1
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s78, s52
	v_mov_b32_e32 v18, 0
	s_cselect_b32 s5, -1, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v24, 0
	v_ashrrev_i32_e32 v3, 31, v2
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v68, 0
	v_mov_b32_e32 v23, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v70, 0
	s_wait_alu 0xfffe
	s_and_b32 s4, s4, s5
	s_mov_b32 s5, 0
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_mov_b32 s70, s53
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_123
; %bb.122:
	s_mul_i32 s4, s78, 3
	v_mov_b32_e32 v4, 0
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 2
	s_wait_alu 0xfffe
	s_add_nc_u64 s[4:5], s[26:27], s[4:5]
	global_load_b32 v70, v4, s[4:5] offset:8
	v_mad_co_u64_u32 v[4:5], null, s78, s70, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[5:6], null, s78, s71, v[5:6]
	v_mul_lo_u32 v6, v4, s6
	v_mad_co_u64_u32 v[22:23], null, v4, s0, s[22:23]
	v_mul_lo_u32 v8, v4, s3
	v_mad_co_u64_u32 v[24:25], null, v4, s93, s[24:25]
	v_mul_lo_u32 v7, v5, s0
	v_mul_lo_u32 v5, v5, s93
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add3_u32 v23, v7, v23, v6
	v_add3_u32 v25, v5, v25, v8
.LBB_123:
	s_cmp_gt_i32 s79, -1
	v_mov_b32_e32 v26, 0
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s79, s52
	v_mov_b32_e32 v27, 0
	s_cselect_b32 s5, -1, 0
	s_wait_alu 0xfffe
	s_and_b32 s4, s4, s5
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_125
; %bb.124:
	s_mul_i32 s4, s79, 3
	s_mov_b32 s5, 0
	v_mov_b32_e32 v4, 0
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 2
	s_wait_alu 0xfffe
	s_add_nc_u64 s[4:5], s[26:27], s[4:5]
	global_load_b32 v68, v4, s[4:5] offset:8
	v_mad_co_u64_u32 v[4:5], null, s79, s70, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[5:6], null, s79, s71, v[5:6]
	v_mul_lo_u32 v6, v4, s6
	v_mad_co_u64_u32 v[18:19], null, v4, s0, s[22:23]
	v_mul_lo_u32 v8, v4, s3
	v_mad_co_u64_u32 v[26:27], null, v4, s93, s[24:25]
	v_mul_lo_u32 v7, v5, s0
	v_mul_lo_u32 v5, v5, s93
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add3_u32 v19, v7, v19, v6
	v_add3_u32 v27, v5, v27, v8
.LBB_125:
	s_cmp_gt_i32 s80, -1
	v_mov_b32_e32 v14, 0
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s80, s52
	v_mov_b32_e32 v20, 0
	s_cselect_b32 s5, -1, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v69, 0
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v66, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v29, 0
	s_wait_alu 0xfffe
	s_and_b32 s4, s4, s5
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_127
; %bb.126:
	s_mul_i32 s4, s80, 3
	s_mov_b32 s5, 0
	v_mov_b32_e32 v4, 0
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 2
	s_wait_alu 0xfffe
	s_add_nc_u64 s[4:5], s[26:27], s[4:5]
	global_load_b32 v69, v4, s[4:5] offset:8
	v_mad_co_u64_u32 v[4:5], null, s80, s70, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[5:6], null, s80, s71, v[5:6]
	v_mul_lo_u32 v6, v4, s6
	v_mad_co_u64_u32 v[20:21], null, v4, s0, s[22:23]
	v_mul_lo_u32 v8, v4, s3
	v_mad_co_u64_u32 v[28:29], null, v4, s93, s[24:25]
	v_mul_lo_u32 v7, v5, s0
	v_mul_lo_u32 v5, v5, s93
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add3_u32 v21, v7, v21, v6
	v_add3_u32 v29, v5, v29, v8
.LBB_127:
	s_cmp_gt_i32 s82, -1
	v_mov_b32_e32 v30, 0
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s82, s52
	v_mov_b32_e32 v31, 0
	s_cselect_b32 s5, -1, 0
	s_wait_alu 0xfffe
	s_and_b32 s4, s4, s5
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_129
; %bb.128:
	s_mul_i32 s4, s82, 3
	s_mov_b32 s5, 0
	v_mov_b32_e32 v4, 0
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 2
	s_wait_alu 0xfffe
	s_add_nc_u64 s[4:5], s[26:27], s[4:5]
	global_load_b32 v66, v4, s[4:5] offset:8
	v_mad_co_u64_u32 v[4:5], null, s82, s70, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[5:6], null, s82, s71, v[5:6]
	v_mul_lo_u32 v6, v4, s6
	v_mad_co_u64_u32 v[14:15], null, v4, s0, s[22:23]
	v_mul_lo_u32 v8, v4, s3
	v_mad_co_u64_u32 v[30:31], null, v4, s93, s[24:25]
	v_mul_lo_u32 v7, v5, s0
	v_mul_lo_u32 v5, v5, s93
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add3_u32 v15, v7, v15, v6
	v_add3_u32 v31, v5, v31, v8
.LBB_129:
	s_cmp_gt_i32 s83, -1
	v_mov_b32_e32 v10, 0
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s83, s52
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v35, 0
	s_cselect_b32 s5, -1, 0
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v64, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v36, 0
	v_mov_b32_e32 v67, 0
	s_wait_alu 0xfffe
	s_and_b32 s4, s4, s5
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_131
; %bb.130:
	s_mul_i32 s4, s83, 3
	s_mov_b32 s5, 0
	v_mov_b32_e32 v4, 0
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 2
	s_wait_alu 0xfffe
	s_add_nc_u64 s[4:5], s[26:27], s[4:5]
	global_load_b32 v67, v4, s[4:5] offset:8
	v_mad_co_u64_u32 v[4:5], null, s83, s70, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[5:6], null, s83, s71, v[5:6]
	v_mul_lo_u32 v6, v4, s6
	v_mad_co_u64_u32 v[16:17], null, v4, s0, s[22:23]
	v_mul_lo_u32 v8, v4, s3
	v_mad_co_u64_u32 v[35:36], null, v4, s93, s[24:25]
	v_mul_lo_u32 v7, v5, s0
	v_mul_lo_u32 v5, v5, s93
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add3_u32 v17, v7, v17, v6
	v_add3_u32 v36, v5, v36, v8
.LBB_131:
	s_cmp_gt_i32 s89, -1
	v_mov_b32_e32 v37, 0
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s89, s52
	v_mov_b32_e32 v38, 0
	s_cselect_b32 s5, -1, 0
	s_wait_alu 0xfffe
	s_and_b32 s4, s4, s5
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_133
; %bb.132:
	s_mul_i32 s4, s89, 3
	s_mov_b32 s5, 0
	v_mov_b32_e32 v4, 0
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 2
	s_wait_alu 0xfffe
	s_add_nc_u64 s[4:5], s[26:27], s[4:5]
	global_load_b32 v64, v4, s[4:5] offset:8
	v_mad_co_u64_u32 v[4:5], null, s89, s70, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[5:6], null, s89, s71, v[5:6]
	v_mul_lo_u32 v6, v4, s6
	v_mad_co_u64_u32 v[10:11], null, v4, s0, s[22:23]
	v_mul_lo_u32 v8, v4, s3
	v_mad_co_u64_u32 v[37:38], null, v4, s93, s[24:25]
	v_mul_lo_u32 v7, v5, s0
	v_mul_lo_u32 v5, v5, s93
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add3_u32 v11, v7, v11, v6
	v_add3_u32 v38, v5, v38, v8
.LBB_133:
	s_cmp_gt_i32 s91, -1
	v_mov_b32_e32 v6, 0
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s91, s52
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v39, 0
	s_cselect_b32 s5, -1, 0
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v13, 0 :: v_dual_mov_b32 v40, 0
	v_mov_b32_e32 v65, 0
	s_wait_alu 0xfffe
	s_and_b32 s4, s4, s5
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_135
; %bb.134:
	s_mul_i32 s4, s91, 3
	s_mov_b32 s5, 0
	v_mov_b32_e32 v4, 0
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 2
	s_wait_alu 0xfffe
	s_add_nc_u64 s[4:5], s[26:27], s[4:5]
	global_load_b32 v65, v4, s[4:5] offset:8
	v_mad_co_u64_u32 v[4:5], null, s91, s70, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[8:9], null, s91, s71, v[5:6]
	v_mul_lo_u32 v5, v4, s6
	v_mad_co_u64_u32 v[12:13], null, v4, s0, s[22:23]
	v_mul_lo_u32 v34, v4, s3
	v_mad_co_u64_u32 v[39:40], null, v4, s93, s[24:25]
	v_mul_lo_u32 v9, v8, s0
	v_mul_lo_u32 v8, v8, s93
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add3_u32 v13, v9, v13, v5
	v_add3_u32 v40, v8, v40, v34
.LBB_135:
	s_cmp_gt_i32 s92, -1
	v_mov_b32_e32 v41, 0
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s92, s52
	v_mov_b32_e32 v42, 0
	s_cselect_b32 s5, -1, 0
	s_wait_alu 0xfffe
	s_and_b32 s4, s4, s5
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_137
; %bb.136:
	s_mul_i32 s4, s92, 3
	s_mov_b32 s5, 0
	v_mov_b32_e32 v4, 0
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 2
	s_wait_alu 0xfffe
	s_add_nc_u64 s[4:5], s[26:27], s[4:5]
	global_load_b32 v62, v4, s[4:5] offset:8
	v_mad_co_u64_u32 v[3:4], null, s92, s70, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[4:5], null, s92, s71, v[4:5]
	v_mul_lo_u32 v5, v3, s6
	v_mad_co_u64_u32 v[6:7], null, v3, s0, s[22:23]
	v_mul_lo_u32 v9, v3, s3
	v_mad_co_u64_u32 v[41:42], null, v3, s93, s[24:25]
	v_mul_lo_u32 v8, v4, s0
	v_mul_lo_u32 v4, v4, s93
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add3_u32 v7, v8, v7, v5
	v_add3_u32 v42, v4, v42, v9
.LBB_137:
	v_mov_b32_e32 v34, 0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[22:23]
	v_cmp_gt_i32_e64 s3, s93, v33
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v60, v34 :: v_dual_mov_b32 v59, v34
	v_mov_b32_e32 v86, v34
	s_and_b32 s4, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB_139
; %bb.138:
	v_lshlrev_b64_e32 v[3:4], 3, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, v22, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, v23, v4, vcc_lo
	v_add_co_u32 v8, vcc_lo, v24, v33
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, 0, v25, vcc_lo
	global_load_b64 v[59:60], v[3:4], off th:TH_LOAD_NT
	global_load_u8 v86, v[8:9], off
.LBB_139:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[18:19]
	v_dual_mov_b32 v58, v34 :: v_dual_mov_b32 v57, v34
	v_mov_b32_e32 v85, v34
	s_and_b32 s4, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB_141
; %bb.140:
	v_lshlrev_b64_e32 v[3:4], 3, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, v18, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, v19, v4, vcc_lo
	v_add_co_u32 v8, vcc_lo, v26, v33
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, 0, v27, vcc_lo
	global_load_b64 v[57:58], v[3:4], off th:TH_LOAD_NT
	global_load_u8 v85, v[8:9], off
.LBB_141:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[20:21]
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v84, 0
	s_and_b32 s4, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB_143
; %bb.142:
	v_lshlrev_b64_e32 v[3:4], 3, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, v20, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, v21, v4, vcc_lo
	v_add_co_u32 v8, vcc_lo, v28, v33
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, 0, v29, vcc_lo
	global_load_b64 v[53:54], v[3:4], off th:TH_LOAD_NT
	global_load_u8 v84, v[8:9], off
.LBB_143:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[14:15]
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v83, 0
	s_and_b32 s4, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB_145
; %bb.144:
	v_lshlrev_b64_e32 v[3:4], 3, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, v14, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, v15, v4, vcc_lo
	v_add_co_u32 v8, vcc_lo, v30, v33
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, 0, v31, vcc_lo
	global_load_b64 v[51:52], v[3:4], off th:TH_LOAD_NT
	global_load_u8 v83, v[8:9], off
.LBB_145:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[16:17]
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v82, 0
	s_and_b32 s4, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB_147
; %bb.146:
	v_lshlrev_b64_e32 v[3:4], 3, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, v16, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, v17, v4, vcc_lo
	v_add_co_u32 v8, vcc_lo, v35, v33
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, 0, v36, vcc_lo
	global_load_b64 v[49:50], v[3:4], off th:TH_LOAD_NT
	global_load_u8 v82, v[8:9], off
.LBB_147:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[10:11]
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v81, 0
	s_and_b32 s4, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB_149
; %bb.148:
	v_lshlrev_b64_e32 v[3:4], 3, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, v10, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, v11, v4, vcc_lo
	v_add_co_u32 v8, vcc_lo, v37, v33
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, 0, v38, vcc_lo
	global_load_b64 v[47:48], v[3:4], off th:TH_LOAD_NT
	global_load_u8 v81, v[8:9], off
.LBB_149:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[12:13]
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v80, 0
	s_and_b32 s4, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB_151
; %bb.150:
	v_lshlrev_b64_e32 v[3:4], 3, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, v12, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, v13, v4, vcc_lo
	v_add_co_u32 v8, vcc_lo, v39, v33
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, 0, v40, vcc_lo
	global_load_b64 v[45:46], v[3:4], off th:TH_LOAD_NT
	global_load_u8 v80, v[8:9], off
.LBB_151:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[6:7]
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v79, 0
	s_and_b32 s4, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB_153
; %bb.152:
	v_lshlrev_b64_e32 v[3:4], 3, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, v6, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, v7, v4, vcc_lo
	v_add_co_u32 v8, vcc_lo, v41, v33
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, 0, v42, vcc_lo
	global_load_b64 v[43:44], v[3:4], off th:TH_LOAD_NT
	global_load_u8 v79, v[8:9], off
.LBB_153:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_lshr_b32 s0, s55, 31
	s_ashr_i32 s5, s55, 31
	s_wait_alu 0xfffe
	s_add_co_i32 s0, s55, s0
	s_lshr_b32 s4, s5, 28
	s_wait_alu 0xfffe
	s_ashr_i32 s0, s0, 1
	s_add_co_i32 s4, s55, s4
	s_wait_alu 0xfffe
	v_mad_co_i64_i32 v[4:5], null, v2, s0, 0
	s_ashr_i32 s91, s4, 4
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v8, 0
	s_wait_alu 0xfffe
	v_cmp_gt_i32_e64 s0, s91, v33
	v_mov_b32_e32 v9, 0
	s_and_saveexec_b32 s4, s0
	s_cbranch_execz .LBB_155
; %bb.154:
	v_lshlrev_b64_e32 v[8:9], 3, v[33:34]
	v_add_co_u32 v3, vcc_lo, s48, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v34, null, s49, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v3, v8
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, v34, v9, vcc_lo
	global_load_b64 v[8:9], v[8:9], off th:TH_LOAD_NT
.LBB_155:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	v_mad_co_i64_i32 v[2:3], null, v2, s91, s[50:51]
	s_and_saveexec_b32 s4, s0
	s_cbranch_execz .LBB_157
; %bb.156:
	v_add_co_u32 v55, vcc_lo, v2, v33
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v56, null, 0, v3, vcc_lo
	global_load_u8 v63, v[55:56], off
.LBB_157:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	v_mov_b32_e32 v34, 0
	s_lshl_b64 s[6:7], s[68:69], 1
	s_mov_b32 s4, exec_lo
	s_wait_alu 0xfffe
	s_add_nc_u64 s[6:7], s[16:17], s[6:7]
	global_load_u16 v34, v34, s[6:7]
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s89, v34
	v_cmpx_gt_u32_e32 16, v0
	s_cbranch_execz .LBB_159
; %bb.158:
	v_and_b32_e32 v34, 7, v0
	v_lshlrev_b32_e32 v56, 28, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v55, v34, 22, 0.5
	v_cmp_ne_u32_e32 vcc_lo, 1, v34
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v55, 0.5, v55, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0, v34
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v34, 0, v55 :: v_dual_lshlrev_b32 v55, 2, v0
	v_and_or_b32 v34, 0x80000000, v56, v34
	ds_store_b32 v55, v34
.LBB_159:
	s_or_b32 exec_lo, exec_lo, s4
	s_ashr_i32 s57, s56, 31
	s_mov_b32 s10, s54
	s_mov_b32 s4, s55
	s_bitcmp1_b32 s30, 17
	s_mul_u64 s[6:7], s[56:57], s[10:11]
	s_wait_alu 0xfffe
	s_mul_u64 s[16:17], s[68:69], s[4:5]
	s_cselect_b32 s4, -1, 0
	s_mul_u64 s[6:7], s[6:7], s[68:69]
	s_wait_alu 0xfffe
	s_and_b32 s4, s4, s1
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s8, s4
	s_cbranch_execz .LBB_197
; %bb.160:
	v_cmp_ne_u64_e64 s4, 0, v[22:23]
	v_cmp_gt_i32_e64 s5, s93, v33
	v_mov_b32_e32 v34, 0
	v_cmp_le_i32_e32 vcc_lo, s93, v33
	s_and_b32 s4, s4, s5
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s22, s4
	s_cbranch_execz .LBB_164
; %bb.161:
	v_dual_mov_b32 v71, v33 :: v_dual_and_b32 v34, 31, v0
	s_lshl_b64 s[4:5], s[6:7], 1
	s_mov_b32 s23, 0
	s_wait_alu 0xfffe
	s_add_nc_u64 s[4:5], s[62:63], s[4:5]
	v_dual_mov_b32 v34, 0 :: v_dual_lshlrev_b32 v55, 5, v34
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v55, s4, s4, v55
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v56, null, s5, 0, s4
.LBB_162:                              ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[72:75], v[55:56], off
	global_load_b128 v[87:90], v[55:56], off offset:16
	v_add_nc_u32_e32 v71, 32, v71
	v_add_co_u32 v55, s4, 0x400, v55
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v56, null, 0, v56, s4
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s5, s93, v71
	s_or_b32 s23, s5, s23
	s_wait_loadcnt 0x1
	v_xor3_b32 v34, v72, v34, v73
	v_xor3_b32 v34, v34, v74, v75
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor3_b32 v34, v34, v87, v88
	v_xor3_b32 v34, v34, v89, v90
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s23
	s_cbranch_execnz .LBB_162
; %bb.163:
	s_or_b32 exec_lo, exec_lo, s23
.LBB_164:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	v_cmp_ne_u64_e64 s4, 0, v[18:19]
	s_xor_b32 s5, vcc_lo, -1
	s_wait_alu 0xfffe
	s_and_b32 s4, s4, s5
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s22, s4
	s_cbranch_execz .LBB_168
; %bb.165:
	s_mul_u64 s[24:25], s[56:57], s[68:69]
	v_and_b32_e32 v55, 31, v0
	s_wait_alu 0xfffe
	s_lshl_b64 s[24:25], s[24:25], 1
	v_mov_b32_e32 v71, v33
	s_wait_alu 0xfffe
	s_add_nc_u64 s[24:25], s[24:25], 2
	s_mov_b32 s23, 0
	v_lshlrev_b32_e32 v55, 5, v55
	s_wait_alu 0xfffe
	s_mul_u64 s[24:25], s[24:25], s[10:11]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[24:25], s[62:63], s[24:25]
	s_wait_alu 0xfffe
	v_add_co_u32 v55, s4, s24, v55
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v56, null, s25, 0, s4
.LBB_166:                              ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[72:75], v[55:56], off
	global_load_b128 v[87:90], v[55:56], off offset:16
	v_add_nc_u32_e32 v71, 32, v71
	v_add_co_u32 v55, vcc_lo, 0x400, v55
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v56, null, 0, v56, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s4, s93, v71
	s_or_b32 s23, s4, s23
	s_wait_loadcnt 0x1
	v_xor3_b32 v34, v72, v34, v73
	v_xor3_b32 v34, v34, v74, v75
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor3_b32 v34, v34, v87, v88
	v_xor3_b32 v34, v34, v89, v90
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s23
	s_cbranch_execnz .LBB_166
; %bb.167:
	s_or_b32 exec_lo, exec_lo, s23
.LBB_168:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	v_cmp_ne_u64_e32 vcc_lo, 0, v[20:21]
	s_and_b32 s4, vcc_lo, s5
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s22, s4
	s_cbranch_execz .LBB_172
; %bb.169:
	s_mul_u64 s[24:25], s[56:57], s[68:69]
	v_and_b32_e32 v55, 31, v0
	s_wait_alu 0xfffe
	s_lshl_b64 s[24:25], s[24:25], 1
	v_mov_b32_e32 v71, v33
	s_wait_alu 0xfffe
	s_add_nc_u64 s[24:25], s[24:25], 4
	s_mov_b32 s23, 0
	v_lshlrev_b32_e32 v55, 5, v55
	s_wait_alu 0xfffe
	s_mul_u64 s[24:25], s[24:25], s[10:11]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[24:25], s[62:63], s[24:25]
	s_wait_alu 0xfffe
	v_add_co_u32 v55, s4, s24, v55
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v56, null, s25, 0, s4
.LBB_170:                              ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[72:75], v[55:56], off
	global_load_b128 v[87:90], v[55:56], off offset:16
	v_add_nc_u32_e32 v71, 32, v71
	v_add_co_u32 v55, vcc_lo, 0x400, v55
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v56, null, 0, v56, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s4, s93, v71
	s_or_b32 s23, s4, s23
	s_wait_loadcnt 0x1
	v_xor3_b32 v34, v72, v34, v73
	v_xor3_b32 v34, v34, v74, v75
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor3_b32 v34, v34, v87, v88
	v_xor3_b32 v34, v34, v89, v90
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s23
	s_cbranch_execnz .LBB_170
; %bb.171:
	s_or_b32 exec_lo, exec_lo, s23
.LBB_172:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	v_cmp_ne_u64_e32 vcc_lo, 0, v[14:15]
	s_and_b32 s4, vcc_lo, s5
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s22, s4
	s_cbranch_execz .LBB_176
; %bb.173:
	s_mul_u64 s[24:25], s[56:57], s[68:69]
	v_and_b32_e32 v55, 31, v0
	s_wait_alu 0xfffe
	s_lshl_b64 s[24:25], s[24:25], 1
	v_mov_b32_e32 v71, v33
	s_wait_alu 0xfffe
	s_add_nc_u64 s[24:25], s[24:25], 6
	s_mov_b32 s23, 0
	v_lshlrev_b32_e32 v55, 5, v55
	s_wait_alu 0xfffe
	s_mul_u64 s[24:25], s[24:25], s[10:11]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[24:25], s[62:63], s[24:25]
	s_wait_alu 0xfffe
	v_add_co_u32 v55, s4, s24, v55
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v56, null, s25, 0, s4
.LBB_174:                              ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[72:75], v[55:56], off
	global_load_b128 v[87:90], v[55:56], off offset:16
	v_add_nc_u32_e32 v71, 32, v71
	v_add_co_u32 v55, vcc_lo, 0x400, v55
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v56, null, 0, v56, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s4, s93, v71
	s_or_b32 s23, s4, s23
	s_wait_loadcnt 0x1
	v_xor3_b32 v34, v72, v34, v73
	v_xor3_b32 v34, v34, v74, v75
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor3_b32 v34, v34, v87, v88
	v_xor3_b32 v34, v34, v89, v90
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s23
	s_cbranch_execnz .LBB_174
; %bb.175:
	s_or_b32 exec_lo, exec_lo, s23
.LBB_176:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	v_cmp_ne_u64_e32 vcc_lo, 0, v[16:17]
	s_and_b32 s4, vcc_lo, s5
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s22, s4
	s_cbranch_execz .LBB_180
; %bb.177:
	s_mul_u64 s[24:25], s[56:57], s[68:69]
	v_and_b32_e32 v55, 31, v0
	s_wait_alu 0xfffe
	s_lshl_b64 s[24:25], s[24:25], 1
	v_mov_b32_e32 v71, v33
	s_wait_alu 0xfffe
	s_add_nc_u64 s[24:25], s[24:25], 8
	s_mov_b32 s23, 0
	v_lshlrev_b32_e32 v55, 5, v55
	s_wait_alu 0xfffe
	s_mul_u64 s[24:25], s[24:25], s[10:11]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[24:25], s[62:63], s[24:25]
	s_wait_alu 0xfffe
	v_add_co_u32 v55, s4, s24, v55
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v56, null, s25, 0, s4
.LBB_178:                              ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[72:75], v[55:56], off
	global_load_b128 v[87:90], v[55:56], off offset:16
	v_add_nc_u32_e32 v71, 32, v71
	v_add_co_u32 v55, vcc_lo, 0x400, v55
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v56, null, 0, v56, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s4, s93, v71
	s_or_b32 s23, s4, s23
	s_wait_loadcnt 0x1
	v_xor3_b32 v34, v72, v34, v73
	v_xor3_b32 v34, v34, v74, v75
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor3_b32 v34, v34, v87, v88
	v_xor3_b32 v34, v34, v89, v90
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s23
	s_cbranch_execnz .LBB_178
; %bb.179:
	s_or_b32 exec_lo, exec_lo, s23
.LBB_180:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	v_cmp_ne_u64_e32 vcc_lo, 0, v[10:11]
	s_and_b32 s4, vcc_lo, s5
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s22, s4
	s_cbranch_execz .LBB_184
; %bb.181:
	s_mul_u64 s[24:25], s[56:57], s[68:69]
	v_and_b32_e32 v55, 31, v0
	s_wait_alu 0xfffe
	s_lshl_b64 s[24:25], s[24:25], 1
	v_mov_b32_e32 v71, v33
	s_wait_alu 0xfffe
	s_add_nc_u64 s[24:25], s[24:25], 10
	s_mov_b32 s23, 0
	v_lshlrev_b32_e32 v55, 5, v55
	s_wait_alu 0xfffe
	s_mul_u64 s[24:25], s[24:25], s[10:11]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[24:25], s[62:63], s[24:25]
	s_wait_alu 0xfffe
	v_add_co_u32 v55, s4, s24, v55
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v56, null, s25, 0, s4
.LBB_182:                              ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[72:75], v[55:56], off
	global_load_b128 v[87:90], v[55:56], off offset:16
	v_add_nc_u32_e32 v71, 32, v71
	v_add_co_u32 v55, vcc_lo, 0x400, v55
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v56, null, 0, v56, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s4, s93, v71
	s_or_b32 s23, s4, s23
	s_wait_loadcnt 0x1
	v_xor3_b32 v34, v72, v34, v73
	v_xor3_b32 v34, v34, v74, v75
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor3_b32 v34, v34, v87, v88
	v_xor3_b32 v34, v34, v89, v90
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s23
	s_cbranch_execnz .LBB_182
; %bb.183:
	s_or_b32 exec_lo, exec_lo, s23
.LBB_184:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	v_cmp_ne_u64_e32 vcc_lo, 0, v[12:13]
	s_and_b32 s4, vcc_lo, s5
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s22, s4
	s_cbranch_execz .LBB_188
; %bb.185:
	s_mul_u64 s[24:25], s[56:57], s[68:69]
	v_and_b32_e32 v55, 31, v0
	s_wait_alu 0xfffe
	s_lshl_b64 s[24:25], s[24:25], 1
	v_mov_b32_e32 v71, v33
	s_wait_alu 0xfffe
	s_add_nc_u64 s[24:25], s[24:25], 12
	s_mov_b32 s23, 0
	v_lshlrev_b32_e32 v55, 5, v55
	s_wait_alu 0xfffe
	s_mul_u64 s[24:25], s[24:25], s[10:11]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[24:25], s[62:63], s[24:25]
	s_wait_alu 0xfffe
	v_add_co_u32 v55, s4, s24, v55
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v56, null, s25, 0, s4
.LBB_186:                              ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[72:75], v[55:56], off
	global_load_b128 v[87:90], v[55:56], off offset:16
	v_add_nc_u32_e32 v71, 32, v71
	v_add_co_u32 v55, vcc_lo, 0x400, v55
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v56, null, 0, v56, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s4, s93, v71
	s_or_b32 s23, s4, s23
	s_wait_loadcnt 0x1
	v_xor3_b32 v34, v72, v34, v73
	v_xor3_b32 v34, v34, v74, v75
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor3_b32 v34, v34, v87, v88
	v_xor3_b32 v34, v34, v89, v90
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s23
	s_cbranch_execnz .LBB_186
; %bb.187:
	s_or_b32 exec_lo, exec_lo, s23
.LBB_188:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	v_cmp_ne_u64_e32 vcc_lo, 0, v[6:7]
	s_and_b32 s4, vcc_lo, s5
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB_192
; %bb.189:
	s_mul_u64 s[22:23], s[56:57], s[68:69]
	v_and_b32_e32 v55, 31, v0
	s_wait_alu 0xfffe
	s_lshl_b64 s[22:23], s[22:23], 1
	v_mov_b32_e32 v71, v33
	s_wait_alu 0xfffe
	s_add_nc_u64 s[22:23], s[22:23], 14
	v_lshlrev_b32_e32 v55, 5, v55
	s_wait_alu 0xfffe
	s_mul_u64 s[22:23], s[22:23], s[10:11]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[22:23], s[62:63], s[22:23]
	s_wait_alu 0xfffe
	v_add_co_u32 v55, s4, s22, v55
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v56, null, s23, 0, s4
	s_mov_b32 s22, 0
.LBB_190:                              ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[72:75], v[55:56], off
	global_load_b128 v[87:90], v[55:56], off offset:16
	v_add_nc_u32_e32 v71, 32, v71
	v_add_co_u32 v55, vcc_lo, 0x400, v55
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v56, null, 0, v56, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s4, s93, v71
	s_wait_alu 0xfffe
	s_or_b32 s22, s4, s22
	s_wait_loadcnt 0x1
	v_xor3_b32 v34, v72, v34, v73
	v_xor3_b32 v34, v34, v74, v75
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor3_b32 v34, v34, v87, v88
	v_xor3_b32 v34, v34, v89, v90
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s22
	s_cbranch_execnz .LBB_190
; %bb.191:
	s_or_b32 exec_lo, exec_lo, s22
.LBB_192:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s5
	s_and_saveexec_b32 s5, s0
	s_cbranch_execz .LBB_196
; %bb.193:
	v_and_b32_e32 v55, 31, v0
	s_lshl_b64 s[22:23], s[16:17], 1
	v_mov_b32_e32 v71, v33
	s_wait_alu 0xfffe
	s_add_nc_u64 s[22:23], s[64:65], s[22:23]
	v_lshlrev_b32_e32 v55, 5, v55
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v55, s4, s22, v55
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v56, null, s23, 0, s4
	s_mov_b32 s22, 0
.LBB_194:                              ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[72:75], v[55:56], off
	global_load_b128 v[87:90], v[55:56], off offset:16
	v_add_nc_u32_e32 v71, 32, v71
	v_add_co_u32 v55, vcc_lo, 0x400, v55
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v56, null, 0, v56, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s4, s91, v71
	s_wait_alu 0xfffe
	s_or_b32 s22, s4, s22
	s_wait_loadcnt 0x1
	v_xor3_b32 v34, v72, v34, v73
	v_xor3_b32 v34, v34, v74, v75
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor3_b32 v34, v34, v87, v88
	v_xor3_b32 v34, v34, v89, v90
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s22
	s_cbranch_execnz .LBB_194
; %bb.195:
	s_or_b32 exec_lo, exec_lo, s22
.LBB_196:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s5
	;;#ASMSTART
	;;#ASMEND
.LBB_197:
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB_200
; %bb.198:
	v_mov_b32_e32 v34, 0
	global_load_b32 v55, v34, s[34:35] scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	global_inv scope:SCOPE_DEV
	v_cmp_le_u32_e32 vcc_lo, s31, v55
	s_cbranch_vccnz .LBB_200
.LBB_199:                              ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_load_b32 v55, v34, s[34:35] scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	global_inv scope:SCOPE_DEV
	v_cmp_gt_u32_e32 vcc_lo, s31, v55
	s_cbranch_vccnz .LBB_199
.LBB_200:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_loadcnt 0x0
	global_inv scope:SCOPE_DEV
	s_and_saveexec_b32 s92, s1
	s_cbranch_execz .LBB_363
; %bb.201:
	v_dual_mov_b32 v72, 0 :: v_dual_mov_b32 v71, 0
	v_dual_mov_b32 v74, 0 :: v_dual_mov_b32 v73, 0
	v_dual_mov_b32 v76, 0 :: v_dual_mov_b32 v75, 0
	v_dual_mov_b32 v78, 0 :: v_dual_mov_b32 v77, 0
	s_lshl_b64 s[4:5], s[6:7], 1
	s_wait_alu 0xfffe
	s_add_nc_u64 s[22:23], s[62:63], s[4:5]
	s_and_saveexec_b32 s1, s3
	s_cbranch_execz .LBB_219
; %bb.202:
	v_dual_mov_b32 v71, 0 :: v_dual_lshlrev_b32 v34, 5, v33
	v_mov_b32_e32 v72, 0
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_u32 v55, s3, s22, v34
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v56, null, s23, 0, s3
	s_mov_b32 s3, exec_lo
	v_cmpx_ne_u64_e32 0, v[22:23]
	s_cbranch_execz .LBB_204
; %bb.203:
	s_clause 0x1
	global_load_b128 v[72:75], v[55:56], off
	global_load_b128 v[87:90], v[55:56], off offset:16
	v_and_b32_e32 v34, 15, v59
	v_lshrrev_b32_e32 v76, 2, v59
	v_lshrrev_b32_e32 v77, 6, v59
	v_lshrrev_b32_e32 v78, 10, v59
	v_lshrrev_b32_e32 v91, 14, v59
	v_lshlrev_b32_e32 v34, 2, v34
	v_and_b32_e32 v76, 60, v76
	v_lshrrev_b32_e32 v93, 22, v59
	v_lshrrev_b32_e32 v92, 18, v59
	v_lshrrev_b32_e32 v59, 26, v59
	ds_load_b32 v34, v34
	ds_load_b32 v76, v76
	v_and_b32_e32 v78, 60, v78
	v_and_b32_e32 v92, 60, v92
	v_lshrrev_b32_e32 v96, 10, v60
	v_lshrrev_b32_e32 v97, 22, v60
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v94, 16, v72
	v_and_b32_e32 v72, 0xffff0000, v72
	v_and_b32_e32 v77, 60, v77
	v_lshlrev_b32_e32 v98, 16, v74
	v_and_b32_e32 v74, 0xffff0000, v74
	v_lshlrev_b32_e32 v95, 16, v73
	v_and_b32_e32 v73, 0xffff0000, v73
	ds_load_b32 v77, v77
	s_wait_dscnt 0x2
	v_fma_f32 v34, v34, v94, 0
	v_and_b32_e32 v59, 60, v59
	v_and_b32_e32 v94, 15, v60
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v34, v76, v72 :: v_dual_and_b32 v93, 60, v93
	v_lshrrev_b32_e32 v72, 2, v60
	v_lshrrev_b32_e32 v76, 6, v60
	v_and_b32_e32 v72, 60, v72
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v76, 60, v76
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v34, v77, v95
	v_and_b32_e32 v91, 60, v91
	ds_load_b32 v78, v78
	ds_load_b32 v91, v91
	ds_load_b32 v92, v92
	ds_load_b32 v93, v93
	ds_load_b32 v59, v59
	v_lshrrev_b32_e32 v77, 14, v60
	v_lshrrev_b32_e32 v95, 18, v60
	v_lshrrev_b32_e32 v60, 26, v60
	ds_load_b32 v76, v76
	ds_load_b32 v72, v72
	v_and_b32_e32 v60, 60, v60
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v34, v78, v73
	v_and_b32_e32 v78, 60, v96
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v34, v91, v98
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v34, v92, v74
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v92, 16, v87
	v_lshlrev_b32_e32 v91, 16, v75
	v_and_b32_e32 v75, 0xffff0000, v75
	v_and_b32_e32 v74, 60, v77
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v34, v93, v91 :: v_dual_lshlrev_b32 v73, 2, v94
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v34, v59, v75 :: v_dual_and_b32 v91, 60, v97
	ds_load_b32 v73, v73
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v34, v73, v92 :: v_dual_and_b32 v77, 60, v95
	ds_load_b32 v59, v78
	ds_load_b32 v74, v74
	ds_load_b32 v75, v77
	ds_load_b32 v77, v91
	ds_load_b32 v60, v60
	v_and_b32_e32 v78, 0xffff0000, v87
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v34, v72, v78 :: v_dual_lshlrev_b32 v73, 16, v88
	v_and_b32_e32 v72, 0xffff0000, v88
	v_fmac_f32_e32 v34, v76, v73
	v_lshlrev_b32_e32 v76, 16, v89
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v34, v59, v72 :: v_dual_and_b32 v73, 7, v86
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v34, v74, v76
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v72, v73
	v_lshlrev_b32_e32 v76, 16, v90
	v_and_b32_e32 v73, 0xffff0000, v89
	v_and_b32_e32 v59, 0x7f, v86
	v_mul_f32_e32 v72, 0x3b000000, v72
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v34, v75, v73
	v_lshl_add_u32 v74, v59, 20, 0x3c000000
	v_cmp_lt_u32_e32 vcc_lo, 7, v59
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v34, v77, v76 :: v_dual_lshlrev_b32 v73, 24, v86
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v72, v72, v74, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v59
	v_and_b32_e32 v74, 0xffff0000, v90
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v59, 0x7fc00000, v72 :: v_dual_fmac_f32 v34, v60, v74
	v_and_or_b32 v59, 0x80000000, v73, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v72, v34, v59, 0
.LBB_204:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	v_cmpx_ne_u64_e32 0, v[18:19]
	s_cbranch_execz .LBB_206
; %bb.205:
	s_and_b32 s4, s77, exec_lo
	s_cselect_b32 s5, s11, 0
	s_cselect_b32 s4, s10, 0
	v_and_b32_e32 v34, 15, v57
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 1
	v_lshrrev_b32_e32 v71, 10, v57
	s_wait_alu 0xfffe
	v_add_co_u32 v59, vcc_lo, v55, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v60, null, s5, v56, vcc_lo
	v_lshlrev_b32_e32 v34, 2, v34
	v_lshrrev_b32_e32 v77, 14, v57
	s_clause 0x1
	global_load_b128 v[73:76], v[59:60], off
	global_load_b128 v[86:89], v[59:60], off offset:16
	v_lshrrev_b32_e32 v59, 2, v57
	v_lshrrev_b32_e32 v60, 6, v57
	ds_load_b32 v34, v34
	v_lshrrev_b32_e32 v78, 18, v57
	v_lshrrev_b32_e32 v90, 22, v57
	v_lshrrev_b32_e32 v57, 26, v57
	v_and_b32_e32 v60, 60, v60
	v_lshrrev_b32_e32 v93, 10, v58
	v_and_b32_e32 v78, 60, v78
	v_and_b32_e32 v90, 60, v90
	v_lshrrev_b32_e32 v94, 22, v58
	ds_load_b32 v60, v60
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v92, 16, v74
	v_and_b32_e32 v59, 60, v59
	v_lshlrev_b32_e32 v91, 16, v73
	v_and_b32_e32 v73, 0xffff0000, v73
	v_and_b32_e32 v74, 0xffff0000, v74
	v_lshlrev_b32_e32 v95, 16, v75
	ds_load_b32 v59, v59
	s_wait_dscnt 0x2
	v_fma_f32 v34, v34, v91, 0
	v_and_b32_e32 v71, 60, v71
	v_and_b32_e32 v57, 60, v57
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v34, v59, v73
	v_lshrrev_b32_e32 v59, 2, v58
	v_lshrrev_b32_e32 v73, 6, v58
	v_dual_fmac_f32 v34, v60, v92 :: v_dual_and_b32 v77, 60, v77
	ds_load_b32 v71, v71
	ds_load_b32 v77, v77
	ds_load_b32 v78, v78
	ds_load_b32 v90, v90
	ds_load_b32 v57, v57
	v_and_b32_e32 v59, 60, v59
	v_and_b32_e32 v73, 60, v73
	v_lshrrev_b32_e32 v60, 14, v58
	v_lshrrev_b32_e32 v92, 18, v58
	ds_load_b32 v73, v73
	ds_load_b32 v59, v59
	v_and_b32_e32 v60, 60, v60
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v34, v71, v74
	v_and_b32_e32 v74, 0xffff0000, v75
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v34, v77, v95 :: v_dual_lshlrev_b32 v77, 16, v76
	v_and_b32_e32 v76, 0xffff0000, v76
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v34, v78, v74
	v_and_b32_e32 v91, 15, v58
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v78, 16, v86
	v_lshrrev_b32_e32 v58, 26, v58
	v_and_b32_e32 v74, 60, v92
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v34, v90, v77 :: v_dual_and_b32 v77, 60, v94
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_b32_e32 v58, 60, v58
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v34, v57, v76
	v_and_b32_e32 v76, 0xffff0000, v86
	v_lshlrev_b32_e32 v71, 2, v91
	ds_load_b32 v71, v71
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v34, v71, v78 :: v_dual_lshlrev_b32 v71, 16, v87
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v34, v59, v76 :: v_dual_and_b32 v75, 60, v93
	v_and_b32_e32 v59, 0xffff0000, v87
	v_fmac_f32_e32 v34, v73, v71
	ds_load_b32 v57, v75
	ds_load_b32 v60, v60
	ds_load_b32 v74, v74
	ds_load_b32 v75, v77
	ds_load_b32 v58, v58
	v_and_b32_e32 v71, 7, v85
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v34, v57, v59 :: v_dual_lshlrev_b32 v73, 16, v88
	v_and_b32_e32 v57, 0x7f, v85
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v59, v71
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v34, v60, v73 :: v_dual_and_b32 v71, 0xffff0000, v88
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v60, v57, 20, 0x3c000000
	v_mul_f32_e32 v59, 0x3b000000, v59
	v_cmp_lt_u32_e32 vcc_lo, 7, v57
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v34, v74, v71 :: v_dual_lshlrev_b32 v73, 16, v89
	v_and_b32_e32 v71, 0xffff0000, v89
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v59, v59, v60, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v57
	v_lshlrev_b32_e32 v60, 24, v85
	s_wait_dscnt 0x1
	s_wait_alu 0xfffd
	v_dual_fmac_f32 v34, v75, v73 :: v_dual_cndmask_b32 v57, 0x7fc00000, v59
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v34, v58, v71
	v_and_or_b32 v57, 0x80000000, v60, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v71, v34, v57, 0
.LBB_206:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v74, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_ne_u64_e32 0, v[20:21]
	s_cbranch_execz .LBB_208
; %bb.207:
	s_and_b32 s4, s76, exec_lo
	s_mov_b32 s5, 0
	s_cselect_b32 s4, 2, 0
	v_and_b32_e32 v34, 15, v53
	s_wait_alu 0xfffe
	s_mul_u64 s[4:5], s[4:5], s[10:11]
	v_lshrrev_b32_e32 v78, 2, v53
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 1
	v_lshrrev_b32_e32 v85, 6, v53
	s_wait_alu 0xfffe
	v_add_co_u32 v74, vcc_lo, v55, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v75, null, s5, v56, vcc_lo
	v_lshlrev_b32_e32 v34, 2, v34
	v_and_b32_e32 v78, 60, v78
	s_clause 0x1
	global_load_b128 v[57:60], v[74:75], off
	global_load_b128 v[74:77], v[74:75], off offset:16
	v_lshrrev_b32_e32 v86, 10, v53
	ds_load_b32 v34, v34
	ds_load_b32 v78, v78
	v_lshrrev_b32_e32 v87, 14, v53
	v_lshrrev_b32_e32 v88, 18, v53
	v_lshrrev_b32_e32 v89, 22, v53
	v_lshrrev_b32_e32 v53, 26, v53
	v_and_b32_e32 v86, 60, v86
	v_lshrrev_b32_e32 v92, 10, v54
	v_and_b32_e32 v88, 60, v88
	v_lshrrev_b32_e32 v93, 22, v54
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v91, 16, v58
	v_and_b32_e32 v58, 0xffff0000, v58
	v_and_b32_e32 v85, 60, v85
	v_lshlrev_b32_e32 v90, 16, v57
	v_lshlrev_b32_e32 v94, 16, v59
	v_and_b32_e32 v59, 0xffff0000, v59
	v_and_b32_e32 v57, 0xffff0000, v57
	ds_load_b32 v85, v85
	s_wait_dscnt 0x2
	v_fma_f32 v34, v34, v90, 0
	v_and_b32_e32 v87, 60, v87
	v_and_b32_e32 v53, 60, v53
	v_and_b32_e32 v90, 15, v54
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v34, v78, v57
	v_and_b32_e32 v89, 60, v89
	v_lshrrev_b32_e32 v57, 2, v54
	v_lshrrev_b32_e32 v78, 6, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v78, 60, v78
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v34, v85, v91
	ds_load_b32 v86, v86
	ds_load_b32 v87, v87
	ds_load_b32 v88, v88
	ds_load_b32 v89, v89
	ds_load_b32 v53, v53
	v_lshrrev_b32_e32 v85, 14, v54
	v_lshrrev_b32_e32 v91, 18, v54
	v_lshrrev_b32_e32 v54, 26, v54
	ds_load_b32 v78, v78
	v_and_b32_e32 v54, 60, v54
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v34, v86, v58
	v_lshlrev_b32_e32 v58, 2, v90
	v_and_b32_e32 v86, 60, v92
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v34, v87, v94
	ds_load_b32 v58, v58
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v34, v88, v59
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v88, 16, v74
	v_and_b32_e32 v74, 0xffff0000, v74
	v_lshlrev_b32_e32 v87, 16, v60
	v_and_b32_e32 v60, 0xffff0000, v60
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v34, v89, v87 :: v_dual_and_b32 v57, 60, v57
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v34, v53, v60
	ds_load_b32 v57, v57
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v34, v58, v88
	v_lshlrev_b32_e32 v58, 16, v75
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v34, v57, v74 :: v_dual_and_b32 v59, 60, v85
	v_and_b32_e32 v57, 0xffff0000, v75
	v_lshlrev_b32_e32 v74, 16, v76
	v_dual_fmac_f32 v34, v78, v58 :: v_dual_and_b32 v87, 60, v93
	v_and_b32_e32 v85, 60, v91
	ds_load_b32 v53, v86
	ds_load_b32 v59, v59
	ds_load_b32 v60, v85
	ds_load_b32 v85, v87
	ds_load_b32 v54, v54
	v_and_b32_e32 v58, 7, v84
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v34, v53, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v57, v58
	v_and_b32_e32 v58, 0xffff0000, v76
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v34, v59, v74
	v_lshlrev_b32_e32 v74, 16, v77
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v57, 0x3b000000, v57 :: v_dual_fmac_f32 v34, v60, v58
	v_and_b32_e32 v53, 0x7f, v84
	v_lshlrev_b32_e32 v58, 24, v84
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v34, v85, v74
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v59, v53, 20, 0x3c000000
	v_cmp_lt_u32_e32 vcc_lo, 7, v53
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v57, v57, v59, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v53
	v_and_b32_e32 v59, 0xffff0000, v77
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v53, 0x7fc00000, v57 :: v_dual_fmac_f32 v34, v54, v59
	v_and_or_b32 v53, 0x80000000, v58, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v74, v34, v53, 0
.LBB_208:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	v_cmpx_ne_u64_e32 0, v[14:15]
	s_cbranch_execz .LBB_210
; %bb.209:
	s_and_b32 s4, s75, exec_lo
	s_mov_b32 s5, 0
	s_cselect_b32 s4, 3, 0
	v_and_b32_e32 v34, 15, v51
	s_wait_alu 0xfffe
	s_mul_u64 s[4:5], s[4:5], s[10:11]
	v_lshrrev_b32_e32 v73, 10, v51
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 1
	v_lshrrev_b32_e32 v84, 14, v51
	s_wait_alu 0xfffe
	v_add_co_u32 v53, vcc_lo, v55, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v54, null, s5, v56, vcc_lo
	v_lshlrev_b32_e32 v34, 2, v34
	v_lshrrev_b32_e32 v85, 18, v51
	s_clause 0x1
	global_load_b128 v[57:60], v[53:54], off
	global_load_b128 v[75:78], v[53:54], off offset:16
	v_lshrrev_b32_e32 v53, 2, v51
	v_lshrrev_b32_e32 v54, 6, v51
	ds_load_b32 v34, v34
	v_lshrrev_b32_e32 v86, 22, v51
	v_lshrrev_b32_e32 v51, 26, v51
	v_and_b32_e32 v84, 60, v84
	v_and_b32_e32 v54, 60, v54
	v_lshrrev_b32_e32 v89, 10, v52
	v_and_b32_e32 v86, 60, v86
	v_lshrrev_b32_e32 v90, 22, v52
	ds_load_b32 v54, v54
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v88, 16, v58
	v_and_b32_e32 v53, 60, v53
	v_lshlrev_b32_e32 v91, 16, v59
	v_and_b32_e32 v59, 0xffff0000, v59
	v_lshlrev_b32_e32 v87, 16, v57
	v_and_b32_e32 v57, 0xffff0000, v57
	ds_load_b32 v53, v53
	v_and_b32_e32 v73, 60, v73
	v_and_b32_e32 v58, 0xffff0000, v58
	s_wait_dscnt 0x2
	v_fma_f32 v34, v34, v87, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v34, v53, v57 :: v_dual_and_b32 v51, 60, v51
	v_lshrrev_b32_e32 v53, 2, v52
	v_lshrrev_b32_e32 v57, 6, v52
	v_dual_fmac_f32 v34, v54, v88 :: v_dual_and_b32 v85, 60, v85
	ds_load_b32 v73, v73
	ds_load_b32 v84, v84
	ds_load_b32 v85, v85
	ds_load_b32 v86, v86
	ds_load_b32 v51, v51
	v_and_b32_e32 v53, 60, v53
	v_and_b32_e32 v57, 60, v57
	v_lshrrev_b32_e32 v54, 14, v52
	v_lshrrev_b32_e32 v88, 18, v52
	ds_load_b32 v53, v53
	ds_load_b32 v57, v57
	v_and_b32_e32 v54, 60, v54
	s_wait_dscnt 0x6
	v_dual_fmac_f32 v34, v73, v58 :: v_dual_and_b32 v73, 60, v89
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v34, v84, v91
	v_lshlrev_b32_e32 v84, 16, v60
	v_and_b32_e32 v60, 0xffff0000, v60
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v34, v85, v59
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v85, 16, v75
	v_and_b32_e32 v59, 60, v88
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v34, v86, v84
	v_and_b32_e32 v84, 60, v90
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v34, v51, v60
	v_and_b32_e32 v87, 15, v52
	v_lshrrev_b32_e32 v52, 26, v52
	v_lshlrev_b32_e32 v58, 2, v87
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v52, 60, v52
	ds_load_b32 v58, v58
	ds_load_b32 v51, v73
	ds_load_b32 v54, v54
	ds_load_b32 v59, v59
	ds_load_b32 v60, v84
	ds_load_b32 v52, v52
	s_wait_dscnt 0x5
	v_dual_fmac_f32 v34, v58, v85 :: v_dual_and_b32 v73, 0xffff0000, v75
	v_lshlrev_b32_e32 v58, 16, v76
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v34, v53, v73 :: v_dual_and_b32 v53, 0xffff0000, v76
	v_dual_fmac_f32 v34, v57, v58 :: v_dual_and_b32 v57, 7, v83
	v_lshlrev_b32_e32 v58, 16, v77
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v34, v51, v53 :: v_dual_and_b32 v51, 0x7f, v83
	v_cvt_f32_ubyte0_e32 v53, v57
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v34, v54, v58 :: v_dual_and_b32 v57, 0xffff0000, v77
	v_lshl_add_u32 v54, v51, 20, 0x3c000000
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v53, 0x3b000000, v53
	v_cmp_lt_u32_e32 vcc_lo, 7, v51
	v_lshlrev_b32_e32 v58, 16, v78
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v34, v59, v57 :: v_dual_and_b32 v57, 0xffff0000, v78
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v53, v53, v54, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v51
	v_lshlrev_b32_e32 v54, 24, v83
	s_wait_dscnt 0x1
	s_wait_alu 0xfffd
	v_dual_fmac_f32 v34, v60, v58 :: v_dual_cndmask_b32 v51, 0x7fc00000, v53
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v34, v52, v57
	v_and_or_b32 v51, 0x80000000, v54, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v73, v34, v51, 0
.LBB_210:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v76, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_ne_u64_e32 0, v[16:17]
	s_cbranch_execz .LBB_212
; %bb.211:
	s_and_b32 s4, s74, exec_lo
	s_mov_b32 s5, 0
	s_cselect_b32 s4, 4, 0
	v_and_b32_e32 v34, 15, v49
	s_wait_alu 0xfffe
	s_mul_u64 s[4:5], s[4:5], s[10:11]
	v_lshrrev_b32_e32 v76, 2, v49
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 1
	v_lshrrev_b32_e32 v77, 6, v49
	s_wait_alu 0xfffe
	v_add_co_u32 v57, vcc_lo, v55, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v58, null, s5, v56, vcc_lo
	v_lshlrev_b32_e32 v34, 2, v34
	v_and_b32_e32 v76, 60, v76
	s_clause 0x1
	global_load_b128 v[51:54], v[57:58], off
	global_load_b128 v[57:60], v[57:58], off offset:16
	v_lshrrev_b32_e32 v78, 10, v49
	ds_load_b32 v34, v34
	ds_load_b32 v76, v76
	v_lshrrev_b32_e32 v83, 14, v49
	v_lshrrev_b32_e32 v85, 22, v49
	v_lshrrev_b32_e32 v84, 18, v49
	v_lshrrev_b32_e32 v49, 26, v49
	v_and_b32_e32 v78, 60, v78
	v_lshrrev_b32_e32 v88, 10, v50
	v_lshrrev_b32_e32 v89, 22, v50
	v_and_b32_e32 v84, 60, v84
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v86, 16, v51
	v_lshlrev_b32_e32 v90, 16, v53
	v_and_b32_e32 v53, 0xffff0000, v53
	v_and_b32_e32 v51, 0xffff0000, v51
	v_and_b32_e32 v77, 60, v77
	v_lshlrev_b32_e32 v87, 16, v52
	v_and_b32_e32 v52, 0xffff0000, v52
	v_and_b32_e32 v49, 60, v49
	ds_load_b32 v77, v77
	s_wait_dscnt 0x2
	v_fma_f32 v34, v34, v86, 0
	v_and_b32_e32 v86, 15, v50
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v34, v76, v51 :: v_dual_and_b32 v85, 60, v85
	v_lshrrev_b32_e32 v51, 2, v50
	v_lshrrev_b32_e32 v76, 6, v50
	v_and_b32_e32 v76, 60, v76
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v34, v77, v87
	v_and_b32_e32 v83, 60, v83
	ds_load_b32 v78, v78
	ds_load_b32 v83, v83
	ds_load_b32 v84, v84
	ds_load_b32 v85, v85
	ds_load_b32 v49, v49
	v_lshrrev_b32_e32 v77, 14, v50
	v_lshrrev_b32_e32 v87, 18, v50
	v_lshrrev_b32_e32 v50, 26, v50
	ds_load_b32 v76, v76
	v_and_b32_e32 v50, 60, v50
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v34, v78, v52
	v_lshlrev_b32_e32 v52, 2, v86
	v_and_b32_e32 v78, 60, v88
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v34, v83, v90
	ds_load_b32 v52, v52
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v34, v84, v53
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v84, 16, v57
	v_and_b32_e32 v57, 0xffff0000, v57
	v_lshlrev_b32_e32 v83, 16, v54
	v_and_b32_e32 v54, 0xffff0000, v54
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v34, v85, v83
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v34, v49, v54 :: v_dual_and_b32 v51, 60, v51
	ds_load_b32 v51, v51
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v34, v52, v84
	v_lshlrev_b32_e32 v52, 16, v58
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v34, v51, v57
	v_and_b32_e32 v53, 60, v77
	v_lshlrev_b32_e32 v57, 16, v59
	v_and_b32_e32 v77, 60, v87
	v_dual_fmac_f32 v34, v76, v52 :: v_dual_and_b32 v51, 0xffff0000, v58
	v_and_b32_e32 v83, 60, v89
	ds_load_b32 v49, v78
	ds_load_b32 v53, v53
	ds_load_b32 v54, v77
	ds_load_b32 v77, v83
	ds_load_b32 v50, v50
	v_and_b32_e32 v52, 7, v82
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v34, v49, v51 :: v_dual_and_b32 v49, 0x7f, v82
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v51, v52
	v_and_b32_e32 v52, 0xffff0000, v59
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v34, v53, v57
	v_lshl_add_u32 v53, v49, 20, 0x3c000000
	v_mul_f32_e32 v51, 0x3b000000, v51
	v_cmp_lt_u32_e32 vcc_lo, 7, v49
	v_lshlrev_b32_e32 v57, 16, v60
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v34, v54, v52
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v51, v51, v53 :: v_dual_lshlrev_b32 v52, 24, v82
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v49
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v34, v77, v57 :: v_dual_and_b32 v53, 0xffff0000, v60
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v49, 0x7fc00000, v51 :: v_dual_fmac_f32 v34, v50, v53
	v_and_or_b32 v49, 0x80000000, v52, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v76, v34, v49, 0
.LBB_212:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	v_cmpx_ne_u64_e32 0, v[10:11]
	s_cbranch_execz .LBB_214
; %bb.213:
	s_and_b32 s4, s73, exec_lo
	s_mov_b32 s5, 0
	s_cselect_b32 s4, 5, 0
	v_and_b32_e32 v34, 15, v47
	s_wait_alu 0xfffe
	s_mul_u64 s[4:5], s[4:5], s[10:11]
	v_lshrrev_b32_e32 v75, 10, v47
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 1
	v_lshrrev_b32_e32 v77, 14, v47
	s_wait_alu 0xfffe
	v_add_co_u32 v53, vcc_lo, v55, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v54, null, s5, v56, vcc_lo
	v_lshlrev_b32_e32 v34, 2, v34
	v_lshrrev_b32_e32 v78, 18, v47
	s_clause 0x1
	global_load_b128 v[49:52], v[53:54], off
	global_load_b128 v[57:60], v[53:54], off offset:16
	v_lshrrev_b32_e32 v53, 2, v47
	v_lshrrev_b32_e32 v54, 6, v47
	ds_load_b32 v34, v34
	v_lshrrev_b32_e32 v82, 22, v47
	v_lshrrev_b32_e32 v47, 26, v47
	v_and_b32_e32 v78, 60, v78
	v_and_b32_e32 v54, 60, v54
	v_lshrrev_b32_e32 v85, 10, v48
	v_and_b32_e32 v82, 60, v82
	v_and_b32_e32 v47, 60, v47
	v_lshrrev_b32_e32 v86, 22, v48
	ds_load_b32 v54, v54
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v84, 16, v50
	v_and_b32_e32 v53, 60, v53
	v_lshlrev_b32_e32 v87, 16, v51
	v_lshlrev_b32_e32 v83, 16, v49
	v_and_b32_e32 v51, 0xffff0000, v51
	v_and_b32_e32 v49, 0xffff0000, v49
	ds_load_b32 v53, v53
	v_and_b32_e32 v50, 0xffff0000, v50
	s_wait_dscnt 0x2
	v_fma_f32 v34, v34, v83, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v34, v53, v49 :: v_dual_and_b32 v75, 60, v75
	v_lshrrev_b32_e32 v53, 6, v48
	v_lshrrev_b32_e32 v49, 2, v48
	v_dual_fmac_f32 v34, v54, v84 :: v_dual_and_b32 v77, 60, v77
	ds_load_b32 v75, v75
	ds_load_b32 v77, v77
	ds_load_b32 v78, v78
	ds_load_b32 v82, v82
	ds_load_b32 v47, v47
	v_and_b32_e32 v53, 60, v53
	v_lshrrev_b32_e32 v54, 14, v48
	v_lshrrev_b32_e32 v84, 18, v48
	ds_load_b32 v53, v53
	v_and_b32_e32 v49, 60, v49
	ds_load_b32 v49, v49
	s_wait_dscnt 0x6
	v_dual_fmac_f32 v34, v75, v50 :: v_dual_and_b32 v75, 60, v85
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v34, v77, v87 :: v_dual_lshlrev_b32 v77, 16, v52
	v_and_b32_e32 v52, 0xffff0000, v52
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v34, v78, v51
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v78, 16, v57
	v_and_b32_e32 v57, 0xffff0000, v57
	v_and_b32_e32 v51, 60, v54
	v_and_b32_e32 v54, 60, v84
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v34, v82, v77
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v34, v47, v52
	v_and_b32_e32 v83, 15, v48
	v_lshrrev_b32_e32 v48, 26, v48
	v_lshlrev_b32_e32 v50, 2, v83
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_and_b32_e32 v48, 60, v48
	ds_load_b32 v50, v50
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v34, v50, v78
	v_lshlrev_b32_e32 v50, 16, v58
	v_dual_fmac_f32 v34, v49, v57 :: v_dual_and_b32 v49, 0xffff0000, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v34, v53, v50 :: v_dual_lshlrev_b32 v53, 16, v59
	v_and_b32_e32 v77, 60, v86
	ds_load_b32 v47, v75
	ds_load_b32 v51, v51
	ds_load_b32 v52, v54
	ds_load_b32 v54, v77
	ds_load_b32 v48, v48
	v_and_b32_e32 v50, 7, v81
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v34, v47, v49
	v_and_b32_e32 v47, 0x7f, v81
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v49, v50
	v_and_b32_e32 v50, 0xffff0000, v59
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v34, v51, v53
	v_lshl_add_u32 v51, v47, 20, 0x3c000000
	v_mul_f32_e32 v49, 0x3b000000, v49
	v_cmp_lt_u32_e32 vcc_lo, 7, v47
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v34, v52, v50 :: v_dual_lshlrev_b32 v53, 16, v60
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v49, v49, v51 :: v_dual_lshlrev_b32 v50, 24, v81
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v47
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v34, v54, v53 :: v_dual_and_b32 v51, 0xffff0000, v60
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v47, 0x7fc00000, v49 :: v_dual_fmac_f32 v34, v48, v51
	v_and_or_b32 v47, 0x80000000, v50, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v75, v34, v47, 0
.LBB_214:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v78, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_ne_u64_e32 0, v[12:13]
	s_cbranch_execz .LBB_216
; %bb.215:
	s_and_b32 s4, s72, exec_lo
	s_mov_b32 s5, 0
	s_cselect_b32 s4, 6, 0
	v_and_b32_e32 v34, 15, v45
	s_wait_alu 0xfffe
	s_mul_u64 s[4:5], s[4:5], s[10:11]
	v_lshrrev_b32_e32 v57, 2, v45
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 1
	v_lshrrev_b32_e32 v58, 6, v45
	s_wait_alu 0xfffe
	v_add_co_u32 v51, vcc_lo, v55, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v52, null, s5, v56, vcc_lo
	v_lshlrev_b32_e32 v34, 2, v34
	v_and_b32_e32 v58, 60, v58
	s_clause 0x1
	global_load_b128 v[47:50], v[51:52], off
	global_load_b128 v[51:54], v[51:52], off offset:16
	v_lshrrev_b32_e32 v81, 22, v45
	ds_load_b32 v34, v34
	ds_load_b32 v58, v58
	v_lshrrev_b32_e32 v59, 10, v45
	v_lshrrev_b32_e32 v60, 14, v45
	v_lshrrev_b32_e32 v78, 18, v45
	v_lshrrev_b32_e32 v45, 26, v45
	v_lshrrev_b32_e32 v84, 10, v46
	v_lshrrev_b32_e32 v85, 22, v46
	v_and_b32_e32 v60, 60, v60
	v_and_b32_e32 v78, 60, v78
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v83, 16, v48
	v_and_b32_e32 v57, 60, v57
	v_lshlrev_b32_e32 v82, 16, v47
	v_lshlrev_b32_e32 v86, 16, v49
	v_and_b32_e32 v49, 0xffff0000, v49
	v_and_b32_e32 v47, 0xffff0000, v47
	ds_load_b32 v57, v57
	s_wait_dscnt 0x2
	v_fma_f32 v34, v34, v82, 0
	v_and_b32_e32 v81, 60, v81
	v_and_b32_e32 v59, 60, v59
	v_and_b32_e32 v48, 0xffff0000, v48
	v_and_b32_e32 v82, 15, v46
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v34, v57, v47
	v_lshrrev_b32_e32 v47, 2, v46
	v_lshrrev_b32_e32 v57, 6, v46
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v34, v58, v83 :: v_dual_and_b32 v45, 60, v45
	ds_load_b32 v59, v59
	ds_load_b32 v60, v60
	ds_load_b32 v78, v78
	ds_load_b32 v81, v81
	ds_load_b32 v45, v45
	v_lshrrev_b32_e32 v58, 14, v46
	v_lshrrev_b32_e32 v83, 18, v46
	v_lshrrev_b32_e32 v46, 26, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_and_b32_e32 v46, 60, v46
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v34, v59, v48
	v_lshlrev_b32_e32 v48, 2, v82
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v34, v60, v86
	ds_load_b32 v48, v48
	v_lshlrev_b32_e32 v60, 16, v50
	v_and_b32_e32 v50, 0xffff0000, v50
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v34, v78, v49 :: v_dual_and_b32 v47, 60, v47
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v78, 16, v51
	v_and_b32_e32 v51, 0xffff0000, v51
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v34, v81, v60 :: v_dual_and_b32 v57, 60, v57
	v_and_b32_e32 v60, 60, v85
	ds_load_b32 v57, v57
	ds_load_b32 v47, v47
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v34, v45, v50
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v34, v48, v78
	v_lshlrev_b32_e32 v48, 16, v52
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v34, v47, v51 :: v_dual_and_b32 v49, 60, v58
	v_lshlrev_b32_e32 v51, 16, v53
	v_and_b32_e32 v47, 0xffff0000, v52
	v_and_b32_e32 v58, 60, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v34, v57, v48
	v_and_b32_e32 v59, 60, v84
	ds_load_b32 v45, v59
	ds_load_b32 v49, v49
	ds_load_b32 v50, v58
	ds_load_b32 v58, v60
	ds_load_b32 v46, v46
	v_and_b32_e32 v48, 7, v80
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v34, v45, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v47, v48
	v_and_b32_e32 v48, 0xffff0000, v53
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v34, v49, v51 :: v_dual_lshlrev_b32 v51, 16, v54
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v47, 0x3b000000, v47 :: v_dual_fmac_f32 v34, v50, v48
	v_and_b32_e32 v45, 0x7f, v80
	v_lshlrev_b32_e32 v48, 24, v80
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v34, v58, v51
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v49, v45, 20, 0x3c000000
	v_cmp_lt_u32_e32 vcc_lo, 7, v45
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v47, v47, v49, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v45
	v_and_b32_e32 v49, 0xffff0000, v54
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v45, 0x7fc00000, v47 :: v_dual_fmac_f32 v34, v46, v49
	v_and_or_b32 v45, 0x80000000, v48, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v78, v34, v45, 0
.LBB_216:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	v_cmpx_ne_u64_e32 0, v[6:7]
	s_cbranch_execz .LBB_218
; %bb.217:
	s_and_b32 s4, s9, exec_lo
	s_mov_b32 s5, 0
	s_cselect_b32 s4, 7, 0
	v_and_b32_e32 v34, 15, v43
	s_wait_alu 0xfffe
	s_mul_u64 s[4:5], s[4:5], s[10:11]
	v_lshrrev_b32_e32 v53, 2, v43
	s_wait_alu 0xfffe
	s_lshl_b64 s[4:5], s[4:5], 1
	v_lshrrev_b32_e32 v54, 6, v43
	s_wait_alu 0xfffe
	v_add_co_u32 v49, vcc_lo, v55, s4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v50, null, s5, v56, vcc_lo
	v_lshlrev_b32_e32 v34, 2, v34
	v_and_b32_e32 v54, 60, v54
	s_clause 0x1
	global_load_b128 v[45:48], v[49:50], off
	global_load_b128 v[49:52], v[49:50], off offset:16
	v_lshrrev_b32_e32 v55, 10, v43
	ds_load_b32 v34, v34
	ds_load_b32 v54, v54
	v_lshrrev_b32_e32 v56, 14, v43
	v_lshrrev_b32_e32 v57, 18, v43
	v_lshrrev_b32_e32 v58, 22, v43
	v_lshrrev_b32_e32 v43, 26, v43
	v_lshrrev_b32_e32 v77, 10, v44
	v_and_b32_e32 v56, 60, v56
	v_lshrrev_b32_e32 v80, 22, v44
	v_and_b32_e32 v58, 60, v58
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v60, 16, v46
	v_and_b32_e32 v53, 60, v53
	v_lshlrev_b32_e32 v81, 16, v47
	v_and_b32_e32 v47, 0xffff0000, v47
	v_lshlrev_b32_e32 v59, 16, v45
	v_and_b32_e32 v45, 0xffff0000, v45
	ds_load_b32 v53, v53
	v_and_b32_e32 v46, 0xffff0000, v46
	v_and_b32_e32 v43, 60, v43
	s_wait_dscnt 0x2
	v_fma_f32 v34, v34, v59, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v34, v53, v45 :: v_dual_and_b32 v55, 60, v55
	v_lshrrev_b32_e32 v53, 6, v44
	v_lshrrev_b32_e32 v45, 2, v44
	v_dual_fmac_f32 v34, v54, v60 :: v_dual_and_b32 v57, 60, v57
	ds_load_b32 v55, v55
	ds_load_b32 v56, v56
	ds_load_b32 v57, v57
	ds_load_b32 v58, v58
	ds_load_b32 v43, v43
	v_and_b32_e32 v53, 60, v53
	v_lshrrev_b32_e32 v54, 14, v44
	v_lshrrev_b32_e32 v60, 18, v44
	ds_load_b32 v53, v53
	v_and_b32_e32 v45, 60, v45
	ds_load_b32 v45, v45
	s_wait_dscnt 0x6
	v_dual_fmac_f32 v34, v55, v46 :: v_dual_and_b32 v55, 60, v77
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v34, v56, v81
	v_lshlrev_b32_e32 v56, 16, v48
	v_and_b32_e32 v48, 0xffff0000, v48
	s_wait_loadcnt_dscnt 0x4
	v_dual_fmac_f32 v34, v57, v47 :: v_dual_lshlrev_b32 v57, 16, v49
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v34, v58, v56 :: v_dual_and_b32 v49, 0xffff0000, v49
	v_and_b32_e32 v56, 60, v80
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v34, v43, v48
	v_and_b32_e32 v59, 15, v44
	v_lshrrev_b32_e32 v44, 26, v44
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v46, 2, v59
	v_and_b32_e32 v44, 60, v44
	ds_load_b32 v46, v46
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v34, v46, v57
	v_lshlrev_b32_e32 v46, 16, v50
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v34, v45, v49 :: v_dual_lshlrev_b32 v49, 16, v51
	v_and_b32_e32 v45, 0xffff0000, v50
	v_fmac_f32_e32 v34, v53, v46
	v_and_b32_e32 v47, 60, v54
	v_and_b32_e32 v54, 60, v60
	ds_load_b32 v43, v55
	ds_load_b32 v47, v47
	ds_load_b32 v48, v54
	ds_load_b32 v54, v56
	ds_load_b32 v44, v44
	v_and_b32_e32 v46, 7, v79
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v34, v43, v45
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v45, v46
	v_and_b32_e32 v46, 0xffff0000, v51
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v34, v47, v49 :: v_dual_lshlrev_b32 v49, 16, v52
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v45, 0x3b000000, v45 :: v_dual_fmac_f32 v34, v48, v46
	v_and_b32_e32 v43, 0x7f, v79
	v_lshlrev_b32_e32 v46, 24, v79
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v34, v54, v49
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v47, v43, 20, 0x3c000000
	v_cmp_lt_u32_e32 vcc_lo, 7, v43
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v45, v45, v47, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v43
	v_and_b32_e32 v47, 0xffff0000, v52
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v43, 0x7fc00000, v45 :: v_dual_fmac_f32 v34, v44, v47
	v_and_or_b32 v43, 0x80000000, v46, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v77, v34, v43, 0
.LBB_218:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
.LBB_219:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s94, exec_lo
	v_cmpx_gt_i32_e64 s93, v32
	s_cbranch_execz .LBB_255
; %bb.220:
	v_add_co_u32 v45, s9, 0x100, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v46, null, 0, v23, s9
	v_add_co_u32 v47, s9, v24, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v48, null, 0, v25, s9
	v_add_co_u32 v49, s9, 0x100, v18
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v50, null, 0, v19, s9
	v_add_co_u32 v51, s9, v26, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v52, null, 0, v27, s9
	v_add_co_u32 v53, s9, 0x100, v20
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v54, null, 0, v21, s9
	v_add_co_u32 v55, s9, v28, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v56, null, 0, v29, s9
	v_add_co_u32 v57, s9, 0x100, v14
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v58, null, 0, v15, s9
	v_add_co_u32 v59, s9, v30, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v60, null, 0, v31, s9
	v_add_co_u32 v79, s9, 0x100, v16
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v80, null, 0, v17, s9
	v_add_co_u32 v81, s9, v35, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v82, null, 0, v36, s9
	v_add_co_u32 v83, s9, 0x100, v10
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v84, null, 0, v11, s9
	v_add_co_u32 v85, s9, v37, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v86, null, 0, v38, s9
	v_add_co_u32 v87, s9, 0x100, v12
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v88, null, 0, v13, s9
	v_add_co_u32 v89, s9, v39, 32
	v_and_b32_e32 v43, 31, v0
	s_mul_u64 s[50:51], s[56:57], s[68:69]
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v90, null, 0, v40, s9
	v_add_co_u32 v91, s9, 0x100, v6
	s_wait_alu 0xfffe
	s_lshl_b64 s[80:81], s[50:51], 1
	v_add_co_ci_u32_e64 v92, null, 0, v7, s9
	v_add_co_u32 v93, s9, v41, 32
	v_cmp_ne_u64_e64 s1, 0, v[18:19]
	v_cmp_ne_u64_e64 s3, 0, v[20:21]
	v_cmp_ne_u64_e64 s4, 0, v[14:15]
	v_cmp_ne_u64_e64 s5, 0, v[16:17]
	v_cmp_ne_u64_e64 s6, 0, v[10:11]
	v_cmp_ne_u64_e64 s7, 0, v[12:13]
	v_cmp_ne_u64_e64 s8, 0, v[6:7]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[50:51], s[80:81], 2
	s_add_nc_u64 s[72:73], s[80:81], 4
	s_add_nc_u64 s[74:75], s[80:81], 6
	s_add_nc_u64 s[76:77], s[80:81], 8
	s_add_nc_u64 s[78:79], s[80:81], 10
	s_add_nc_u64 s[82:83], s[80:81], 12
	s_add_nc_u64 s[80:81], s[80:81], 14
	v_lshlrev_b32_e32 v44, 3, v43
	v_add_co_ci_u32_e64 v42, null, 0, v42, s9
	v_lshlrev_b32_e32 v94, 5, v43
	s_wait_alu 0xfffe
	s_mul_u64 s[50:51], s[50:51], s[10:11]
	s_mul_u64 s[72:73], s[72:73], s[10:11]
	s_mul_u64 s[74:75], s[74:75], s[10:11]
	s_mul_u64 s[76:77], s[76:77], s[10:11]
	s_mul_u64 s[78:79], s[78:79], s[10:11]
	s_mul_u64 s[82:83], s[82:83], s[10:11]
	s_mul_u64 s[10:11], s[80:81], s[10:11]
	s_mov_b64 s[24:25], 32
	s_wait_alu 0xfffe
	s_add_nc_u64 s[50:51], s[62:63], s[50:51]
	s_add_nc_u64 s[72:73], s[62:63], s[72:73]
	s_add_nc_u64 s[74:75], s[62:63], s[74:75]
	s_add_nc_u64 s[76:77], s[62:63], s[76:77]
	s_add_nc_u64 s[78:79], s[62:63], s[78:79]
	s_add_nc_u64 s[80:81], s[62:63], s[82:83]
	s_add_nc_u64 s[82:83], s[62:63], s[10:11]
	s_mov_b32 s11, 0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[22:23]
	s_branch .LBB_222
.LBB_221:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	v_add_co_u32 v45, s9, 0x100, v45
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v46, null, 0, v46, s9
	v_add_co_u32 v47, s9, v47, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v48, null, 0, v48, s9
	v_add_co_u32 v49, s9, 0x100, v49
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v50, null, 0, v50, s9
	v_add_co_u32 v51, s9, v51, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v52, null, 0, v52, s9
	v_add_co_u32 v53, s9, 0x100, v53
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v54, null, 0, v54, s9
	v_add_co_u32 v55, s9, v55, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v56, null, 0, v56, s9
	v_add_co_u32 v57, s9, 0x100, v57
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v58, null, 0, v58, s9
	v_add_co_u32 v59, s9, v59, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v60, null, 0, v60, s9
	v_add_co_u32 v79, s9, 0x100, v79
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v80, null, 0, v80, s9
	v_add_co_u32 v81, s9, v81, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v82, null, 0, v82, s9
	v_add_co_u32 v83, s9, 0x100, v83
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v84, null, 0, v84, s9
	v_add_co_u32 v85, s9, v85, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v86, null, 0, v86, s9
	v_add_co_u32 v87, s9, 0x100, v87
	s_add_nc_u64 s[24:25], s[24:25], 32
	v_add_co_ci_u32_e64 v88, null, 0, v88, s9
	v_add_co_u32 v89, s9, v89, 32
	s_wait_loadcnt 0x1
	s_wait_alu 0xfffe
	v_add_nc_u32_e32 v24, s24, v43
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v90, null, 0, v90, s9
	v_add_co_u32 v91, s9, 0x100, v91
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v92, null, 0, v92, s9
	v_add_co_u32 v93, s9, v93, 32
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v42, null, 0, v42, s9
	v_cmp_le_i32_e64 s9, s93, v24
	s_add_nc_u64 s[22:23], s[22:23], 0x400
	s_add_nc_u64 s[50:51], s[50:51], 0x400
	s_add_nc_u64 s[72:73], s[72:73], 0x400
	s_add_nc_u64 s[74:75], s[74:75], 0x400
	s_add_nc_u64 s[76:77], s[76:77], 0x400
	s_add_nc_u64 s[78:79], s[78:79], 0x400
	s_add_nc_u64 s[80:81], s[80:81], 0x400
	s_or_b32 s11, s9, s11
	s_add_nc_u64 s[82:83], s[82:83], 0x400
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s11
	s_cbranch_execz .LBB_254
.LBB_222:                              ; =>This Inner Loop Header: Depth=1
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v40, 0
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v102, 0
	s_and_saveexec_b32 s10, vcc_lo
	s_cbranch_execz .LBB_224
; %bb.223:                              ;   in Loop: Header=BB4_222 Depth=1
	v_add_co_u32 v24, s9, v45, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v25, null, 0, v46, s9
	v_add_co_u32 v26, s9, v47, v43
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v27, null, 0, v48, s9
	global_load_b64 v[40:41], v[24:25], off th:TH_LOAD_NT
	global_load_u8 v102, v[26:27], off
.LBB_224:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v101, 0
	s_and_saveexec_b32 s10, s1
	s_cbranch_execz .LBB_226
; %bb.225:                              ;   in Loop: Header=BB4_222 Depth=1
	v_add_co_u32 v24, s9, v49, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v25, null, 0, v50, s9
	v_add_co_u32 v26, s9, v51, v43
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v27, null, 0, v52, s9
	global_load_b64 v[38:39], v[24:25], off th:TH_LOAD_NT
	global_load_u8 v101, v[26:27], off
.LBB_226:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v100, 0
	s_and_saveexec_b32 s10, s3
	s_cbranch_execz .LBB_228
; %bb.227:                              ;   in Loop: Header=BB4_222 Depth=1
	v_add_co_u32 v24, s9, v53, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v25, null, 0, v54, s9
	v_add_co_u32 v26, s9, v55, v43
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v27, null, 0, v56, s9
	global_load_b64 v[36:37], v[24:25], off th:TH_LOAD_NT
	global_load_u8 v100, v[26:27], off
.LBB_228:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v99, 0
	s_and_saveexec_b32 s10, s4
	s_cbranch_execz .LBB_230
; %bb.229:                              ;   in Loop: Header=BB4_222 Depth=1
	v_add_co_u32 v24, s9, v57, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v25, null, 0, v58, s9
	v_add_co_u32 v26, s9, v59, v43
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v27, null, 0, v60, s9
	global_load_b64 v[34:35], v[24:25], off th:TH_LOAD_NT
	global_load_u8 v99, v[26:27], off
.LBB_230:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v98, 0
	s_and_saveexec_b32 s10, s5
	s_cbranch_execz .LBB_232
; %bb.231:                              ;   in Loop: Header=BB4_222 Depth=1
	v_add_co_u32 v24, s9, v79, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v25, null, 0, v80, s9
	v_add_co_u32 v26, s9, v81, v43
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v27, null, 0, v82, s9
	global_load_b64 v[30:31], v[24:25], off th:TH_LOAD_NT
	global_load_u8 v98, v[26:27], off
.LBB_232:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v97, 0
	s_and_saveexec_b32 s10, s6
	s_cbranch_execz .LBB_234
; %bb.233:                              ;   in Loop: Header=BB4_222 Depth=1
	v_add_co_u32 v24, s9, v83, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v25, null, 0, v84, s9
	v_add_co_u32 v95, s9, v85, v43
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v96, null, 0, v86, s9
	global_load_b64 v[28:29], v[24:25], off th:TH_LOAD_NT
	global_load_u8 v97, v[95:96], off
.LBB_234:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v96, 0
	s_and_saveexec_b32 s10, s7
	s_cbranch_execz .LBB_236
; %bb.235:                              ;   in Loop: Header=BB4_222 Depth=1
	v_add_co_u32 v24, s9, v87, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v25, null, 0, v88, s9
	v_add_co_u32 v95, s9, v89, v43
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v96, null, 0, v90, s9
	global_load_b64 v[26:27], v[24:25], off th:TH_LOAD_NT
	global_load_u8 v96, v[95:96], off
.LBB_236:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v24, 0
	v_mov_b32_e32 v95, 0
	s_and_saveexec_b32 s10, s8
	s_cbranch_execz .LBB_245
; %bb.237:                              ;   in Loop: Header=BB4_222 Depth=1
	v_add_co_u32 v24, s9, v91, v44
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v25, null, 0, v92, s9
	v_add_co_u32 v103, s9, v93, v43
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v104, null, 0, v42, s9
	global_load_b64 v[24:25], v[24:25], off th:TH_LOAD_NT
	global_load_u8 v95, v[103:104], off
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s57, vcc_lo
	s_cbranch_execnz .LBB_246
.LBB_238:                              ;   in Loop: Header=BB4_222 Depth=1
	s_or_b32 exec_lo, exec_lo, s57
	s_and_saveexec_b32 s10, s1
	s_cbranch_execz .LBB_247
.LBB_239:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_loadcnt 0x1
	s_wait_alu 0xfffe
	v_add_co_u32 v40, s9, s50, v94
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v41, null, s51, 0, s9
	v_lshrrev_b32_e32 v110, 6, v38
	v_lshrrev_b32_e32 v111, 10, v38
	v_lshrrev_b32_e32 v112, 14, v38
	s_wait_loadcnt 0x0
	s_clause 0x1
	global_load_b128 v[102:105], v[40:41], off offset:1024
	global_load_b128 v[106:109], v[40:41], off offset:1040
	v_and_b32_e32 v40, 15, v38
	v_lshrrev_b32_e32 v41, 2, v38
	v_and_b32_e32 v110, 60, v110
	v_lshrrev_b32_e32 v113, 18, v38
	v_lshrrev_b32_e32 v114, 22, v38
	v_lshlrev_b32_e32 v40, 2, v40
	v_lshrrev_b32_e32 v38, 26, v38
	v_and_b32_e32 v112, 60, v112
	v_lshrrev_b32_e32 v117, 10, v39
	v_and_b32_e32 v114, 60, v114
	ds_load_b32 v40, v40
	ds_load_b32 v110, v110
	v_and_b32_e32 v38, 60, v38
	v_lshrrev_b32_e32 v118, 22, v39
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v116, 16, v103
	v_and_b32_e32 v41, 60, v41
	v_lshlrev_b32_e32 v119, 16, v104
	v_lshlrev_b32_e32 v115, 16, v102
	v_and_b32_e32 v102, 0xffff0000, v102
	v_and_b32_e32 v103, 0xffff0000, v103
	ds_load_b32 v41, v41
	v_and_b32_e32 v104, 0xffff0000, v104
	s_wait_dscnt 0x2
	v_fma_f32 v40, v40, v115, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v40, v41, v102 :: v_dual_and_b32 v111, 60, v111
	v_lshrrev_b32_e32 v41, 2, v39
	v_lshrrev_b32_e32 v102, 6, v39
	v_dual_fmac_f32 v40, v110, v116 :: v_dual_and_b32 v113, 60, v113
	ds_load_b32 v111, v111
	ds_load_b32 v112, v112
	ds_load_b32 v113, v113
	ds_load_b32 v114, v114
	ds_load_b32 v38, v38
	v_and_b32_e32 v41, 60, v41
	v_and_b32_e32 v102, 60, v102
	v_lshrrev_b32_e32 v110, 14, v39
	v_lshrrev_b32_e32 v116, 18, v39
	ds_load_b32 v41, v41
	ds_load_b32 v102, v102
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v40, v111, v103
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v112, v119
	v_and_b32_e32 v115, 15, v39
	v_lshlrev_b32_e32 v112, 16, v105
	v_and_b32_e32 v105, 0xffff0000, v105
	v_lshrrev_b32_e32 v39, 26, v39
	s_wait_loadcnt_dscnt 0x4
	v_dual_fmac_f32 v40, v113, v104 :: v_dual_lshlrev_b32 v113, 16, v106
	v_and_b32_e32 v106, 0xffff0000, v106
	v_lshlrev_b32_e32 v103, 2, v115
	v_and_b32_e32 v104, 60, v110
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v40, v114, v112
	v_and_b32_e32 v110, 60, v116
	v_and_b32_e32 v112, 60, v118
	ds_load_b32 v103, v103
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v40, v38, v105 :: v_dual_and_b32 v39, 60, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v40, v103, v113 :: v_dual_lshlrev_b32 v103, 16, v107
	v_fmac_f32_e32 v40, v41, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v40, v102, v103 :: v_dual_lshlrev_b32 v103, 16, v108
	v_and_b32_e32 v111, 60, v117
	ds_load_b32 v38, v111
	ds_load_b32 v104, v104
	ds_load_b32 v105, v110
	ds_load_b32 v110, v112
	ds_load_b32 v39, v39
	v_and_b32_e32 v41, 0xffff0000, v107
	v_and_b32_e32 v102, 7, v101
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v40, v38, v41
	v_and_b32_e32 v38, 0x7f, v101
	v_cvt_f32_ubyte0_e32 v41, v102
	v_and_b32_e32 v102, 0xffff0000, v108
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v40, v104, v103 :: v_dual_lshlrev_b32 v101, 24, v101
	v_lshl_add_u32 v103, v38, 20, 0x3c000000
	v_mul_f32_e32 v41, 0x3b000000, v41
	v_cmp_lt_u32_e64 s9, 7, v38
	v_lshlrev_b32_e32 v104, 16, v109
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v40, v105, v102
	v_and_b32_e32 v102, 0xffff0000, v109
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v41, v41, v103, s9
	v_cmp_ne_u32_e64 s9, 0x7f, v38
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v40, v110, v104
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v38, 0x7fc00000, v41, s9
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v40, v39, v102
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v38, 0x80000000, v101, v38
	v_fmac_f32_e32 v71, v40, v38
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s3
	s_cbranch_execnz .LBB_248
.LBB_240:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s4
	s_cbranch_execz .LBB_249
.LBB_241:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_loadcnt 0x1
	v_add_co_u32 v40, s9, s74, v94
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v41, null, s75, 0, s9
	v_lshrrev_b32_e32 v104, 6, v34
	v_lshrrev_b32_e32 v105, 10, v34
	v_lshrrev_b32_e32 v106, 14, v34
	global_load_b128 v[36:39], v[40:41], off offset:1024
	s_wait_loadcnt 0x1
	global_load_b128 v[100:103], v[40:41], off offset:1040
	v_and_b32_e32 v40, 15, v34
	v_lshrrev_b32_e32 v41, 2, v34
	v_and_b32_e32 v104, 60, v104
	v_lshrrev_b32_e32 v107, 18, v34
	v_lshrrev_b32_e32 v108, 22, v34
	v_lshlrev_b32_e32 v40, 2, v40
	v_lshrrev_b32_e32 v34, 26, v34
	v_and_b32_e32 v106, 60, v106
	v_lshrrev_b32_e32 v111, 10, v35
	v_and_b32_e32 v108, 60, v108
	ds_load_b32 v40, v40
	ds_load_b32 v104, v104
	v_and_b32_e32 v34, 60, v34
	v_lshrrev_b32_e32 v112, 22, v35
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v110, 16, v37
	v_and_b32_e32 v41, 60, v41
	v_lshlrev_b32_e32 v113, 16, v38
	v_lshlrev_b32_e32 v109, 16, v36
	v_and_b32_e32 v36, 0xffff0000, v36
	v_and_b32_e32 v37, 0xffff0000, v37
	ds_load_b32 v41, v41
	v_and_b32_e32 v38, 0xffff0000, v38
	s_wait_dscnt 0x2
	v_fma_f32 v40, v40, v109, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v40, v41, v36 :: v_dual_and_b32 v105, 60, v105
	v_lshrrev_b32_e32 v36, 2, v35
	v_lshrrev_b32_e32 v41, 6, v35
	v_dual_fmac_f32 v40, v104, v110 :: v_dual_and_b32 v107, 60, v107
	ds_load_b32 v105, v105
	ds_load_b32 v106, v106
	ds_load_b32 v107, v107
	ds_load_b32 v108, v108
	ds_load_b32 v34, v34
	v_and_b32_e32 v36, 60, v36
	v_lshrrev_b32_e32 v104, 14, v35
	v_lshrrev_b32_e32 v110, 18, v35
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v40, v105, v37
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v40, v106, v113 :: v_dual_and_b32 v109, 15, v35
	v_lshlrev_b32_e32 v106, 16, v39
	v_and_b32_e32 v39, 0xffff0000, v39
	v_lshrrev_b32_e32 v35, 26, v35
	s_wait_loadcnt_dscnt 0x2
	v_dual_fmac_f32 v40, v107, v38 :: v_dual_lshlrev_b32 v107, 16, v100
	v_and_b32_e32 v100, 0xffff0000, v100
	v_lshlrev_b32_e32 v37, 2, v109
	v_and_b32_e32 v38, 60, v104
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v40, v108, v106
	v_and_b32_e32 v104, 60, v110
	v_and_b32_e32 v106, 60, v112
	ds_load_b32 v37, v37
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v40, v34, v39 :: v_dual_and_b32 v41, 60, v41
	ds_load_b32 v41, v41
	ds_load_b32 v36, v36
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v40, v37, v107 :: v_dual_lshlrev_b32 v37, 16, v101
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v40, v36, v100 :: v_dual_and_b32 v105, 60, v111
	v_and_b32_e32 v36, 0xffff0000, v101
	v_dual_fmac_f32 v40, v41, v37 :: v_dual_lshlrev_b32 v41, 16, v102
	v_and_b32_e32 v35, 60, v35
	ds_load_b32 v34, v105
	ds_load_b32 v38, v38
	ds_load_b32 v39, v104
	ds_load_b32 v104, v106
	ds_load_b32 v35, v35
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v40, v34, v36
	v_and_b32_e32 v34, 0x7f, v99
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v40, v38, v41 :: v_dual_lshlrev_b32 v41, 16, v103
	v_and_b32_e32 v37, 7, v99
	v_lshl_add_u32 v38, v34, 20, 0x3c000000
	v_cmp_lt_u32_e64 s9, 7, v34
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v36, v37
	v_and_b32_e32 v37, 0xffff0000, v102
	v_mul_f32_e32 v36, 0x3b000000, v36
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v40, v39, v37 :: v_dual_lshlrev_b32 v37, 24, v99
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v36, v36, v38, s9
	v_cmp_ne_u32_e64 s9, 0x7f, v34
	v_and_b32_e32 v38, 0xffff0000, v103
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v40, v104, v41
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v34, 0x7fc00000, v36, s9
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v40, v35, v38
	v_and_or_b32 v34, 0x80000000, v37, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v73, v40, v34
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s5
	s_cbranch_execnz .LBB_250
.LBB_242:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s6
	s_cbranch_execz .LBB_251
.LBB_243:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_loadcnt 0x1
	v_add_co_u32 v30, s9, s78, v94
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v31, null, s79, 0, s9
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v98, 6, v28
	v_lshrrev_b32_e32 v99, 10, v28
	v_lshrrev_b32_e32 v100, 14, v28
	s_clause 0x1
	global_load_b128 v[34:37], v[30:31], off offset:1024
	global_load_b128 v[38:41], v[30:31], off offset:1040
	v_and_b32_e32 v30, 15, v28
	v_lshrrev_b32_e32 v31, 2, v28
	v_and_b32_e32 v98, 60, v98
	v_lshrrev_b32_e32 v101, 18, v28
	v_lshrrev_b32_e32 v102, 22, v28
	v_lshlrev_b32_e32 v30, 2, v30
	v_lshrrev_b32_e32 v28, 26, v28
	v_and_b32_e32 v100, 60, v100
	v_lshrrev_b32_e32 v105, 10, v29
	v_and_b32_e32 v102, 60, v102
	ds_load_b32 v30, v30
	ds_load_b32 v98, v98
	v_and_b32_e32 v28, 60, v28
	v_lshrrev_b32_e32 v106, 22, v29
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v104, 16, v35
	v_and_b32_e32 v31, 60, v31
	v_lshlrev_b32_e32 v107, 16, v36
	v_lshlrev_b32_e32 v103, 16, v34
	v_and_b32_e32 v34, 0xffff0000, v34
	v_and_b32_e32 v35, 0xffff0000, v35
	ds_load_b32 v31, v31
	v_and_b32_e32 v36, 0xffff0000, v36
	s_wait_dscnt 0x2
	v_fma_f32 v30, v30, v103, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v30, v31, v34 :: v_dual_and_b32 v99, 60, v99
	v_lshrrev_b32_e32 v31, 2, v29
	v_lshrrev_b32_e32 v34, 6, v29
	v_dual_fmac_f32 v30, v98, v104 :: v_dual_and_b32 v101, 60, v101
	ds_load_b32 v99, v99
	ds_load_b32 v100, v100
	ds_load_b32 v101, v101
	ds_load_b32 v102, v102
	ds_load_b32 v28, v28
	v_and_b32_e32 v34, 60, v34
	v_lshrrev_b32_e32 v98, 14, v29
	v_lshrrev_b32_e32 v104, 18, v29
	ds_load_b32 v34, v34
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v30, v99, v35
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v30, v100, v107 :: v_dual_and_b32 v103, 15, v29
	v_lshlrev_b32_e32 v100, 16, v37
	v_and_b32_e32 v37, 0xffff0000, v37
	v_lshrrev_b32_e32 v29, 26, v29
	s_wait_loadcnt_dscnt 0x3
	v_dual_fmac_f32 v30, v101, v36 :: v_dual_lshlrev_b32 v101, 16, v38
	v_and_b32_e32 v38, 0xffff0000, v38
	v_and_b32_e32 v36, 60, v98
	v_and_b32_e32 v98, 60, v104
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v30, v102, v100
	v_and_b32_e32 v100, 60, v106
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v30, v28, v37 :: v_dual_and_b32 v31, 60, v31
	ds_load_b32 v31, v31
	v_lshlrev_b32_e32 v35, 2, v103
	ds_load_b32 v35, v35
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v30, v35, v101 :: v_dual_lshlrev_b32 v35, 16, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v30, v31, v38 :: v_dual_and_b32 v99, 60, v105
	v_and_b32_e32 v31, 0xffff0000, v39
	v_dual_fmac_f32 v30, v34, v35 :: v_dual_lshlrev_b32 v35, 16, v40
	v_and_b32_e32 v29, 60, v29
	ds_load_b32 v28, v99
	ds_load_b32 v36, v36
	ds_load_b32 v37, v98
	ds_load_b32 v98, v100
	ds_load_b32 v29, v29
	v_and_b32_e32 v34, 7, v97
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v30, v28, v31
	v_and_b32_e32 v28, 0x7f, v97
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v31, v34
	v_and_b32_e32 v34, 0xffff0000, v40
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v30, v36, v35
	v_lshl_add_u32 v35, v28, 20, 0x3c000000
	v_mul_f32_e32 v31, 0x3b000000, v31
	v_cmp_lt_u32_e64 s9, 7, v28
	v_lshlrev_b32_e32 v36, 16, v41
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v30, v37, v34
	v_lshlrev_b32_e32 v34, 24, v97
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v31, v31, v35, s9
	v_cmp_ne_u32_e64 s9, 0x7f, v28
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v30, v98, v36 :: v_dual_and_b32 v35, 0xffff0000, v41
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v28, 0x7fc00000, v31, s9
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v30, v29, v35
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v28, 0x80000000, v34, v28
	v_fmac_f32_e32 v75, v30, v28
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s7
	s_cbranch_execnz .LBB_252
.LBB_244:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s8
	s_cbranch_execz .LBB_221
	s_branch .LBB_253
.LBB_245:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s57, vcc_lo
	s_cbranch_execz .LBB_238
.LBB_246:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_alu 0xfffe
	v_add_co_u32 v107, s9, s22, v94
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v108, null, s23, 0, s9
	global_load_b128 v[103:106], v[107:108], off offset:1024
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v111, 16, v105
	v_lshlrev_b32_e32 v112, 16, v106
	v_and_b32_e32 v113, 0xffff0000, v106
	v_and_b32_e32 v106, 15, v40
	v_and_b32_e32 v105, 0xffff0000, v105
	v_lshlrev_b32_e32 v109, 16, v103
	v_and_b32_e32 v103, 0xffff0000, v103
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v106, 2, v106
	ds_load_b32 v106, v106
	s_wait_dscnt 0x0
	v_fma_f32 v109, v106, v109, 0
	v_lshrrev_b32_e32 v106, 2, v40
	v_and_b32_e32 v106, 60, v106
	ds_load_b32 v106, v106
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v109, v106, v103 :: v_dual_lshlrev_b32 v110, 16, v104
	v_lshrrev_b32_e32 v103, 6, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_and_b32_e32 v103, 60, v103
	ds_load_b32 v103, v103
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v109, v103, v110 :: v_dual_and_b32 v104, 0xffff0000, v104
	v_lshrrev_b32_e32 v103, 10, v40
	v_and_b32_e32 v103, 60, v103
	ds_load_b32 v103, v103
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v109, v103, v104
	v_lshrrev_b32_e32 v103, 14, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_and_b32_e32 v103, 60, v103
	ds_load_b32 v103, v103
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v109, v103, v111
	v_lshrrev_b32_e32 v103, 18, v40
	v_and_b32_e32 v103, 60, v103
	ds_load_b32 v103, v103
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v109, v103, v105
	global_load_b128 v[103:106], v[107:108], off offset:1040
	v_lshrrev_b32_e32 v107, 22, v40
	v_lshrrev_b32_e32 v40, 26, v40
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v107, 60, v107
	v_and_b32_e32 v40, 60, v40
	ds_load_b32 v107, v107
	ds_load_b32 v40, v40
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v109, v107, v112
	v_and_b32_e32 v107, 15, v41
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v109, v40, v113
	v_lshlrev_b32_e32 v107, 2, v107
	v_lshrrev_b32_e32 v40, 2, v41
	ds_load_b32 v107, v107
	v_and_b32_e32 v40, 60, v40
	ds_load_b32 v40, v40
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v108, 16, v103
	v_and_b32_e32 v103, 0xffff0000, v103
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v109, v107, v108
	v_lshrrev_b32_e32 v107, 6, v41
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v109, v40, v103
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v107, 60, v107
	v_lshrrev_b32_e32 v40, 10, v41
	ds_load_b32 v103, v107
	v_and_b32_e32 v40, 60, v40
	v_lshlrev_b32_e32 v107, 16, v104
	ds_load_b32 v40, v40
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v109, v103, v107
	v_lshrrev_b32_e32 v103, 14, v41
	v_lshlrev_b32_e32 v107, 16, v105
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_and_b32_e32 v103, 60, v103
	ds_load_b32 v103, v103
	v_and_b32_e32 v104, 0xffff0000, v104
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v109, v40, v104
	v_lshrrev_b32_e32 v40, 18, v41
	v_lshrrev_b32_e32 v104, 22, v41
	v_lshrrev_b32_e32 v41, 26, v41
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v40, 60, v40
	v_and_b32_e32 v41, 60, v41
	ds_load_b32 v40, v40
	ds_load_b32 v41, v41
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v109, v103, v107
	v_and_b32_e32 v103, 0xffff0000, v105
	v_and_b32_e32 v104, 60, v104
	v_and_b32_e32 v105, 7, v102
	ds_load_b32 v104, v104
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v109, v40, v103 :: v_dual_and_b32 v40, 0x7f, v102
	v_cvt_f32_ubyte0_e32 v103, v105
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v105, v40, 20, 0x3c000000
	v_cmp_lt_u32_e64 s9, 7, v40
	v_mul_f32_e32 v103, 0x3b000000, v103
	v_cmp_ne_u32_e64 s10, 0x7f, v40
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v103, v103, v105, s9
	v_and_b32_e32 v105, 0xffff0000, v106
	v_lshlrev_b32_e32 v40, 16, v106
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v109, v104, v40 :: v_dual_lshlrev_b32 v102, 24, v102
	v_cndmask_b32_e64 v40, 0x7fc00000, v103, s10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v109, v41, v105
	v_and_or_b32 v40, 0x80000000, v102, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v72, v109, v40
	s_or_b32 exec_lo, exec_lo, s57
	s_and_saveexec_b32 s10, s1
	s_cbranch_execnz .LBB_239
.LBB_247:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s3
	s_cbranch_execz .LBB_240
.LBB_248:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_loadcnt 0x0
	v_add_co_u32 v101, s9, s72, v94
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v102, null, s73, 0, s9
	v_and_b32_e32 v105, 15, v36
	v_lshrrev_b32_e32 v106, 2, v36
	v_lshrrev_b32_e32 v107, 6, v36
	s_clause 0x1
	global_load_b128 v[38:41], v[101:102], off offset:1024
	global_load_b128 v[101:104], v[101:102], off offset:1040
	v_lshrrev_b32_e32 v108, 10, v36
	v_lshlrev_b32_e32 v105, 2, v105
	v_lshrrev_b32_e32 v109, 14, v36
	v_and_b32_e32 v107, 60, v107
	v_lshrrev_b32_e32 v110, 18, v36
	v_lshrrev_b32_e32 v111, 22, v36
	ds_load_b32 v105, v105
	v_lshrrev_b32_e32 v36, 26, v36
	v_and_b32_e32 v109, 60, v109
	v_lshrrev_b32_e32 v114, 10, v37
	v_and_b32_e32 v111, 60, v111
	v_lshrrev_b32_e32 v115, 22, v37
	v_and_b32_e32 v36, 60, v36
	ds_load_b32 v107, v107
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v113, 16, v39
	v_and_b32_e32 v106, 60, v106
	v_lshlrev_b32_e32 v116, 16, v40
	v_lshlrev_b32_e32 v112, 16, v38
	v_and_b32_e32 v40, 0xffff0000, v40
	v_and_b32_e32 v38, 0xffff0000, v38
	ds_load_b32 v106, v106
	v_and_b32_e32 v39, 0xffff0000, v39
	s_wait_dscnt 0x2
	v_fma_f32 v105, v105, v112, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v105, v106, v38 :: v_dual_and_b32 v108, 60, v108
	v_lshrrev_b32_e32 v106, 6, v37
	v_lshrrev_b32_e32 v38, 2, v37
	v_dual_fmac_f32 v105, v107, v113 :: v_dual_and_b32 v110, 60, v110
	ds_load_b32 v108, v108
	ds_load_b32 v109, v109
	ds_load_b32 v110, v110
	ds_load_b32 v111, v111
	ds_load_b32 v36, v36
	v_and_b32_e32 v106, 60, v106
	v_lshrrev_b32_e32 v107, 14, v37
	v_lshrrev_b32_e32 v113, 18, v37
	ds_load_b32 v106, v106
	v_and_b32_e32 v38, 60, v38
	ds_load_b32 v38, v38
	s_wait_dscnt 0x6
	v_dual_fmac_f32 v105, v108, v39 :: v_dual_and_b32 v108, 60, v114
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v105, v109, v116
	v_lshlrev_b32_e32 v109, 16, v41
	v_and_b32_e32 v41, 0xffff0000, v41
	s_wait_loadcnt_dscnt 0x4
	v_dual_fmac_f32 v105, v110, v40 :: v_dual_lshlrev_b32 v110, 16, v101
	v_and_b32_e32 v101, 0xffff0000, v101
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v105, v111, v109
	v_and_b32_e32 v109, 60, v115
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v105, v36, v41
	v_and_b32_e32 v112, 15, v37
	v_lshrrev_b32_e32 v37, 26, v37
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v39, 2, v112
	v_and_b32_e32 v37, 60, v37
	ds_load_b32 v39, v39
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v105, v39, v110
	v_lshlrev_b32_e32 v39, 16, v102
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v105, v38, v101 :: v_dual_and_b32 v38, 0xffff0000, v102
	v_lshlrev_b32_e32 v101, 16, v103
	v_fmac_f32_e32 v105, v106, v39
	v_and_b32_e32 v40, 60, v107
	v_and_b32_e32 v107, 60, v113
	ds_load_b32 v36, v108
	ds_load_b32 v40, v40
	ds_load_b32 v41, v107
	ds_load_b32 v107, v109
	ds_load_b32 v37, v37
	v_and_b32_e32 v39, 7, v100
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v105, v36, v38 :: v_dual_and_b32 v36, 0x7f, v100
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v38, v39
	v_and_b32_e32 v39, 0xffff0000, v103
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v105, v40, v101
	v_lshl_add_u32 v40, v36, 20, 0x3c000000
	v_mul_f32_e32 v38, 0x3b000000, v38
	v_cmp_lt_u32_e64 s9, 7, v36
	v_lshlrev_b32_e32 v101, 16, v104
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v105, v41, v39
	v_lshlrev_b32_e32 v39, 24, v100
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v38, v38, v40, s9
	v_cmp_ne_u32_e64 s9, 0x7f, v36
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v105, v107, v101 :: v_dual_and_b32 v40, 0xffff0000, v104
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v36, 0x7fc00000, v38, s9
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v105, v37, v40
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v36, 0x80000000, v39, v36
	v_fmac_f32_e32 v74, v105, v36
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s4
	s_cbranch_execnz .LBB_241
.LBB_249:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s5
	s_cbranch_execz .LBB_242
.LBB_250:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_loadcnt 0x1
	v_add_co_u32 v38, s9, s76, v94
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v39, null, s77, 0, s9
	s_wait_loadcnt 0x0
	v_and_b32_e32 v99, 15, v30
	v_lshrrev_b32_e32 v100, 2, v30
	v_lshrrev_b32_e32 v101, 6, v30
	s_clause 0x1
	global_load_b128 v[34:37], v[38:39], off offset:1024
	global_load_b128 v[38:41], v[38:39], off offset:1040
	v_lshrrev_b32_e32 v104, 18, v30
	v_lshlrev_b32_e32 v99, 2, v99
	v_lshrrev_b32_e32 v102, 10, v30
	v_and_b32_e32 v101, 60, v101
	v_lshrrev_b32_e32 v103, 14, v30
	v_lshrrev_b32_e32 v105, 22, v30
	ds_load_b32 v99, v99
	v_lshrrev_b32_e32 v30, 26, v30
	v_lshrrev_b32_e32 v108, 10, v31
	v_and_b32_e32 v103, 60, v103
	v_and_b32_e32 v105, 60, v105
	v_lshrrev_b32_e32 v109, 22, v31
	ds_load_b32 v101, v101
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v107, 16, v35
	v_and_b32_e32 v100, 60, v100
	v_lshlrev_b32_e32 v106, 16, v34
	v_and_b32_e32 v34, 0xffff0000, v34
	v_lshlrev_b32_e32 v110, 16, v36
	v_and_b32_e32 v36, 0xffff0000, v36
	ds_load_b32 v100, v100
	s_wait_dscnt 0x2
	v_fma_f32 v99, v99, v106, 0
	v_and_b32_e32 v104, 60, v104
	v_and_b32_e32 v102, 60, v102
	v_and_b32_e32 v35, 0xffff0000, v35
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v99, v100, v34
	v_lshrrev_b32_e32 v100, 6, v31
	v_lshrrev_b32_e32 v34, 2, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v99, v101, v107 :: v_dual_and_b32 v30, 60, v30
	ds_load_b32 v102, v102
	ds_load_b32 v103, v103
	ds_load_b32 v104, v104
	ds_load_b32 v105, v105
	ds_load_b32 v30, v30
	v_and_b32_e32 v100, 60, v100
	v_lshrrev_b32_e32 v101, 14, v31
	v_lshrrev_b32_e32 v107, 18, v31
	ds_load_b32 v100, v100
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v99, v102, v35
	v_and_b32_e32 v106, 15, v31
	v_lshrrev_b32_e32 v31, 26, v31
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v99, v103, v110
	v_lshlrev_b32_e32 v103, 16, v37
	v_lshlrev_b32_e32 v35, 2, v106
	v_and_b32_e32 v37, 0xffff0000, v37
	v_and_b32_e32 v31, 60, v31
	s_wait_loadcnt_dscnt 0x3
	v_dual_fmac_f32 v99, v104, v36 :: v_dual_lshlrev_b32 v104, 16, v38
	ds_load_b32 v35, v35
	v_and_b32_e32 v38, 0xffff0000, v38
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v99, v105, v103 :: v_dual_and_b32 v36, 60, v101
	v_and_b32_e32 v101, 60, v107
	v_and_b32_e32 v103, 60, v109
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v99, v30, v37 :: v_dual_and_b32 v34, 60, v34
	ds_load_b32 v34, v34
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v99, v35, v104
	v_lshlrev_b32_e32 v35, 16, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v99, v34, v38 :: v_dual_lshlrev_b32 v38, 16, v40
	v_and_b32_e32 v102, 60, v108
	ds_load_b32 v30, v102
	ds_load_b32 v36, v36
	ds_load_b32 v37, v101
	ds_load_b32 v101, v103
	ds_load_b32 v31, v31
	v_and_b32_e32 v34, 0xffff0000, v39
	v_fmac_f32_e32 v99, v100, v35
	v_and_b32_e32 v35, 7, v98
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v99, v30, v34
	v_and_b32_e32 v30, 0x7f, v98
	v_cvt_f32_ubyte0_e32 v34, v35
	v_and_b32_e32 v35, 0xffff0000, v40
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v99, v36, v38
	v_lshl_add_u32 v36, v30, 20, 0x3c000000
	v_mul_f32_e32 v34, 0x3b000000, v34
	v_cmp_lt_u32_e64 s9, 7, v30
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v99, v37, v35 :: v_dual_lshlrev_b32 v38, 16, v41
	v_lshlrev_b32_e32 v35, 24, v98
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v34, v34, v36, s9
	v_cmp_ne_u32_e64 s9, 0x7f, v30
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v99, v101, v38 :: v_dual_and_b32 v36, 0xffff0000, v41
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v30, 0x7fc00000, v34, s9
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v99, v31, v36
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v30, 0x80000000, v35, v30
	v_fmac_f32_e32 v76, v99, v30
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s6
	s_cbranch_execnz .LBB_243
.LBB_251:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s7
	s_cbranch_execz .LBB_244
.LBB_252:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_loadcnt 0x1
	v_add_co_u32 v34, s9, s80, v94
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v35, null, s81, 0, s9
	v_and_b32_e32 v38, 15, v26
	v_lshrrev_b32_e32 v39, 2, v26
	v_lshrrev_b32_e32 v40, 6, v26
	s_clause 0x1
	global_load_b128 v[28:31], v[34:35], off offset:1024
	global_load_b128 v[34:37], v[34:35], off offset:1040
	v_lshrrev_b32_e32 v41, 10, v26
	v_lshlrev_b32_e32 v38, 2, v38
	s_wait_loadcnt 0x2
	v_lshrrev_b32_e32 v97, 14, v26
	v_and_b32_e32 v40, 60, v40
	v_lshrrev_b32_e32 v98, 18, v26
	v_lshrrev_b32_e32 v99, 22, v26
	ds_load_b32 v38, v38
	v_lshrrev_b32_e32 v26, 26, v26
	v_lshrrev_b32_e32 v102, 10, v27
	v_and_b32_e32 v98, 60, v98
	v_lshrrev_b32_e32 v103, 22, v27
	ds_load_b32 v40, v40
	v_and_b32_e32 v26, 60, v26
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v104, 16, v30
	v_lshlrev_b32_e32 v101, 16, v29
	v_and_b32_e32 v39, 60, v39
	v_lshlrev_b32_e32 v100, 16, v28
	v_and_b32_e32 v28, 0xffff0000, v28
	v_and_b32_e32 v30, 0xffff0000, v30
	v_and_b32_e32 v97, 60, v97
	ds_load_b32 v39, v39
	s_wait_dscnt 0x2
	v_fma_f32 v38, v38, v100, 0
	v_and_b32_e32 v41, 60, v41
	v_and_b32_e32 v29, 0xffff0000, v29
	v_and_b32_e32 v100, 15, v27
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v38, v39, v28
	v_lshrrev_b32_e32 v28, 2, v27
	v_lshrrev_b32_e32 v39, 6, v27
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v38, v40, v101 :: v_dual_and_b32 v99, 60, v99
	ds_load_b32 v41, v41
	ds_load_b32 v97, v97
	ds_load_b32 v98, v98
	ds_load_b32 v99, v99
	ds_load_b32 v26, v26
	v_and_b32_e32 v28, 60, v28
	v_lshrrev_b32_e32 v40, 14, v27
	v_lshrrev_b32_e32 v101, 18, v27
	v_lshrrev_b32_e32 v27, 26, v27
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v38, v41, v29 :: v_dual_and_b32 v27, 60, v27
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v38, v97, v104
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v38, v98, v30
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v98, 16, v34
	v_and_b32_e32 v34, 0xffff0000, v34
	v_lshlrev_b32_e32 v97, 16, v31
	v_lshlrev_b32_e32 v29, 2, v100
	v_and_b32_e32 v31, 0xffff0000, v31
	v_and_b32_e32 v30, 60, v40
	v_and_b32_e32 v40, 60, v101
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v38, v99, v97
	ds_load_b32 v29, v29
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v38, v26, v31
	v_and_b32_e32 v39, 60, v39
	ds_load_b32 v39, v39
	ds_load_b32 v28, v28
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v38, v29, v98 :: v_dual_lshlrev_b32 v29, 16, v35
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v38, v28, v34
	v_and_b32_e32 v41, 60, v102
	v_lshlrev_b32_e32 v34, 16, v36
	v_and_b32_e32 v97, 60, v103
	v_and_b32_e32 v28, 0xffff0000, v35
	v_fmac_f32_e32 v38, v39, v29
	ds_load_b32 v26, v41
	ds_load_b32 v30, v30
	ds_load_b32 v31, v40
	ds_load_b32 v40, v97
	ds_load_b32 v27, v27
	v_and_b32_e32 v29, 7, v96
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v38, v26, v28
	v_and_b32_e32 v26, 0x7f, v96
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v28, v29
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v38, v30, v34 :: v_dual_and_b32 v29, 0xffff0000, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v30, v26, 20, 0x3c000000
	v_mul_f32_e32 v28, 0x3b000000, v28
	v_cmp_lt_u32_e64 s9, 7, v26
	v_lshlrev_b32_e32 v34, 16, v37
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v38, v31, v29 :: v_dual_lshlrev_b32 v29, 24, v96
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v28, v28, v30, s9
	v_cmp_ne_u32_e64 s9, 0x7f, v26
	v_and_b32_e32 v30, 0xffff0000, v37
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v38, v40, v34
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v26, 0x7fc00000, v28, s9
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v38, v27, v30
	v_and_or_b32 v26, 0x80000000, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v78, v38, v26
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s8
	s_cbranch_execz .LBB_221
.LBB_253:                              ;   in Loop: Header=BB4_222 Depth=1
	s_wait_loadcnt 0x1
	v_add_co_u32 v30, s9, s82, v94
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v31, null, s83, 0, s9
	v_lshrrev_b32_e32 v38, 6, v24
	v_lshrrev_b32_e32 v39, 10, v24
	v_lshrrev_b32_e32 v40, 14, v24
	s_clause 0x1
	global_load_b128 v[26:29], v[30:31], off offset:1024
	global_load_b128 v[34:37], v[30:31], off offset:1040
	v_and_b32_e32 v30, 15, v24
	v_lshrrev_b32_e32 v31, 2, v24
	v_and_b32_e32 v38, 60, v38
	v_lshrrev_b32_e32 v41, 18, v24
	s_wait_loadcnt 0x2
	v_lshrrev_b32_e32 v96, 22, v24
	v_lshlrev_b32_e32 v30, 2, v30
	v_lshrrev_b32_e32 v24, 26, v24
	v_and_b32_e32 v40, 60, v40
	v_lshrrev_b32_e32 v99, 10, v25
	v_and_b32_e32 v96, 60, v96
	ds_load_b32 v30, v30
	ds_load_b32 v38, v38
	v_and_b32_e32 v24, 60, v24
	v_lshrrev_b32_e32 v100, 22, v25
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v98, 16, v27
	v_and_b32_e32 v31, 60, v31
	v_lshlrev_b32_e32 v101, 16, v28
	v_lshlrev_b32_e32 v97, 16, v26
	v_and_b32_e32 v26, 0xffff0000, v26
	v_and_b32_e32 v27, 0xffff0000, v27
	ds_load_b32 v31, v31
	v_and_b32_e32 v28, 0xffff0000, v28
	s_wait_dscnt 0x2
	v_fma_f32 v30, v30, v97, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v30, v31, v26 :: v_dual_and_b32 v39, 60, v39
	v_lshrrev_b32_e32 v26, 2, v25
	v_lshrrev_b32_e32 v31, 6, v25
	v_dual_fmac_f32 v30, v38, v98 :: v_dual_and_b32 v41, 60, v41
	ds_load_b32 v39, v39
	ds_load_b32 v40, v40
	ds_load_b32 v41, v41
	ds_load_b32 v96, v96
	ds_load_b32 v24, v24
	v_and_b32_e32 v26, 60, v26
	v_lshrrev_b32_e32 v38, 14, v25
	v_lshrrev_b32_e32 v98, 18, v25
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v30, v39, v27
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v40, v101
	v_and_b32_e32 v97, 15, v25
	v_lshlrev_b32_e32 v40, 16, v29
	v_and_b32_e32 v29, 0xffff0000, v29
	v_lshrrev_b32_e32 v25, 26, v25
	s_wait_loadcnt_dscnt 0x2
	v_dual_fmac_f32 v30, v41, v28 :: v_dual_lshlrev_b32 v41, 16, v34
	v_and_b32_e32 v34, 0xffff0000, v34
	v_lshlrev_b32_e32 v27, 2, v97
	v_and_b32_e32 v28, 60, v38
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v30, v96, v40
	v_and_b32_e32 v38, 60, v98
	v_and_b32_e32 v40, 60, v100
	ds_load_b32 v27, v27
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v30, v24, v29 :: v_dual_and_b32 v31, 60, v31
	ds_load_b32 v31, v31
	ds_load_b32 v26, v26
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v30, v27, v41 :: v_dual_lshlrev_b32 v27, 16, v35
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v30, v26, v34
	v_and_b32_e32 v26, 0xffff0000, v35
	v_fmac_f32_e32 v30, v31, v27
	v_and_b32_e32 v39, 60, v99
	v_lshlrev_b32_e32 v31, 16, v36
	v_and_b32_e32 v25, 60, v25
	ds_load_b32 v24, v39
	ds_load_b32 v28, v28
	ds_load_b32 v29, v38
	ds_load_b32 v38, v40
	ds_load_b32 v25, v25
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v30, v24, v26 :: v_dual_and_b32 v27, 7, v95
	v_and_b32_e32 v24, 0x7f, v95
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v26, v27
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v30, v28, v31
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v28, v24, 20, 0x3c000000
	v_dual_mul_f32 v26, 0x3b000000, v26 :: v_dual_lshlrev_b32 v31, 16, v37
	v_and_b32_e32 v27, 0xffff0000, v36
	v_cmp_lt_u32_e64 s9, 7, v24
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v30, v29, v27
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v26, v26, v28, s9
	v_cmp_ne_u32_e64 s9, 0x7f, v24
	v_lshlrev_b32_e32 v27, 24, v95
	v_and_b32_e32 v28, 0xffff0000, v37
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v30, v38, v31
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v24, 0x7fc00000, v26, s9
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v30, v25, v28
	v_and_or_b32 v24, 0x80000000, v27, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v77, v30, v24
	s_branch .LBB_221
.LBB_254:
	s_or_b32 exec_lo, exec_lo, s11
.LBB_255:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s94
	v_cmp_ne_u64_e32 vcc_lo, 0, v[22:23]
	v_mov_b32_e32 v22, 0
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB_265
; %bb.256:
	v_mbcnt_lo_u32_b32 v22, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v23, 16, v22
	v_xor_b32_e32 v24, 8, v22
	v_xor_b32_e32 v25, 4, v22
	v_cmp_gt_i32_e32 vcc_lo, 32, v23
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v23, v22, v23, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v24
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v24, v22, v24, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v24, 2, v24
	v_lshlrev_b32_e32 v23, 2, v23
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v25, v22, v25, vcc_lo
	ds_bpermute_b32 v23, v23, v72
	v_lshlrev_b32_e32 v25, 2, v25
	s_wait_dscnt 0x0
	v_add_f32_e32 v23, v72, v23
	ds_bpermute_b32 v24, v24, v23
	s_wait_dscnt 0x0
	v_add_f32_e32 v23, v23, v24
	ds_bpermute_b32 v24, v25, v23
	v_xor_b32_e32 v25, 2, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v25
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v25, v22, v25, vcc_lo
	v_lshlrev_b32_e32 v25, 2, v25
	s_wait_dscnt 0x0
	v_add_f32_e32 v23, v23, v24
	ds_bpermute_b32 v24, v25, v23
	v_xor_b32_e32 v25, 1, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v25
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v22, v22, v25, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v23, v23, v24 :: v_dual_lshlrev_b32 v22, 2, v22
	ds_bpermute_b32 v22, v22, v23
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v23, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v23, v70, v22
	v_and_b32_e32 v22, 0x7fffffff, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v22
                                        ; implicit-def: $vgpr22
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.257:
	v_bfe_u32 v22, v23, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v22, v23, v22, 0x7fff
                                        ; implicit-def: $vgpr23
	v_lshrrev_b32_e32 v22, 16, v22
; %bb.258:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.259:
	v_lshrrev_b32_e32 v22, 16, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v22, 64, v22
; %bb.260:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v22, 16, v22
	v_mul_f32_e32 v23, s90, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v22, 0x7fffffff, v23
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v22
                                        ; implicit-def: $vgpr22
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.261:
	v_bfe_u32 v22, v23, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v22, v23, v22, 0x7fff
                                        ; implicit-def: $vgpr23
	v_lshrrev_b32_e32 v22, 16, v22
; %bb.262:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.263:
	v_lshrrev_b32_e32 v22, 16, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v22, 64, v22
; %bb.264:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v22, 16, v22
	v_add_f32_e32 v22, 0, v22
.LBB_265:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s1, exec_lo
	v_cmpx_ne_u64_e32 0, v[18:19]
	s_cbranch_execz .LBB_275
; %bb.266:
	v_mbcnt_lo_u32_b32 v18, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v19, 16, v18
	v_xor_b32_e32 v23, 8, v18
	v_xor_b32_e32 v24, 4, v18
	v_cmp_gt_i32_e32 vcc_lo, 32, v19
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v19, v18, v19, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v23
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v23, v18, v23, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v24
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v24, v18, v24 :: v_dual_lshlrev_b32 v23, 2, v23
	v_lshlrev_b32_e32 v24, 2, v24
	v_lshlrev_b32_e32 v19, 2, v19
	ds_bpermute_b32 v19, v19, v71
	s_wait_dscnt 0x0
	v_add_f32_e32 v19, v71, v19
	ds_bpermute_b32 v23, v23, v19
	s_wait_dscnt 0x0
	v_add_f32_e32 v19, v19, v23
	ds_bpermute_b32 v23, v24, v19
	v_xor_b32_e32 v24, 2, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v24
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v24, v18, v24, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v19, v19, v23 :: v_dual_lshlrev_b32 v24, 2, v24
	ds_bpermute_b32 v23, v24, v19
	v_xor_b32_e32 v24, 1, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v24
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v18, v18, v24, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v19, v19, v23 :: v_dual_lshlrev_b32 v18, 2, v18
	ds_bpermute_b32 v18, v18, v19
	s_wait_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v19, v68, v18
	v_and_b32_e32 v18, 0x7fffffff, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v18
                                        ; implicit-def: $vgpr18
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.267:
	v_bfe_u32 v18, v19, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v18, v19, v18, 0x7fff
                                        ; implicit-def: $vgpr19
	v_lshrrev_b32_e32 v18, 16, v18
; %bb.268:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.269:
	v_lshrrev_b32_e32 v18, 16, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v18, 64, v18
; %bb.270:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v18, 16, v18
	v_mul_f32_e32 v19, s88, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v18, 0x7fffffff, v19
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v18
                                        ; implicit-def: $vgpr18
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.271:
	v_bfe_u32 v18, v19, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v18, v19, v18, 0x7fff
                                        ; implicit-def: $vgpr19
	v_lshrrev_b32_e32 v18, 16, v18
; %bb.272:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.273:
	v_lshrrev_b32_e32 v18, 16, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v18, 64, v18
; %bb.274:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v18, 16, v18
	v_add_f32_e32 v22, v22, v18
.LBB_275:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s1, exec_lo
	v_cmpx_ne_u64_e32 0, v[20:21]
	s_cbranch_execz .LBB_285
; %bb.276:
	v_mbcnt_lo_u32_b32 v18, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v19, 16, v18
	v_xor_b32_e32 v20, 8, v18
	v_xor_b32_e32 v21, 4, v18
	v_cmp_gt_i32_e32 vcc_lo, 32, v19
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v19, v18, v19, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v20
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v20, v18, v20, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v21
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v21, v18, v21, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v21, 2, v21
	v_lshlrev_b32_e32 v20, 2, v20
	v_lshlrev_b32_e32 v19, 2, v19
	ds_bpermute_b32 v19, v19, v74
	s_wait_dscnt 0x0
	v_add_f32_e32 v19, v74, v19
	ds_bpermute_b32 v20, v20, v19
	s_wait_dscnt 0x0
	v_add_f32_e32 v19, v19, v20
	ds_bpermute_b32 v20, v21, v19
	v_xor_b32_e32 v21, 2, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v21
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v21, v18, v21, vcc_lo
	v_lshlrev_b32_e32 v21, 2, v21
	s_wait_dscnt 0x0
	v_add_f32_e32 v19, v19, v20
	ds_bpermute_b32 v20, v21, v19
	v_xor_b32_e32 v21, 1, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v21
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v18, v18, v21, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v19, v19, v20 :: v_dual_lshlrev_b32 v18, 2, v18
	ds_bpermute_b32 v18, v18, v19
	s_wait_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v19, v69, v18
	v_and_b32_e32 v18, 0x7fffffff, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v18
                                        ; implicit-def: $vgpr18
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.277:
	v_bfe_u32 v18, v19, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v18, v19, v18, 0x7fff
                                        ; implicit-def: $vgpr19
	v_lshrrev_b32_e32 v18, 16, v18
; %bb.278:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.279:
	v_lshrrev_b32_e32 v18, 16, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v18, 64, v18
; %bb.280:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v18, 16, v18
	v_mul_f32_e32 v19, s87, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v18, 0x7fffffff, v19
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v18
                                        ; implicit-def: $vgpr18
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.281:
	v_bfe_u32 v18, v19, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v18, v19, v18, 0x7fff
                                        ; implicit-def: $vgpr19
	v_lshrrev_b32_e32 v18, 16, v18
; %bb.282:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.283:
	v_lshrrev_b32_e32 v18, 16, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v18, 64, v18
; %bb.284:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v18, 16, v18
	v_add_f32_e32 v22, v22, v18
.LBB_285:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s1, exec_lo
	v_cmpx_ne_u64_e32 0, v[14:15]
	s_cbranch_execz .LBB_295
; %bb.286:
	v_mbcnt_lo_u32_b32 v14, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v15, 16, v14
	v_xor_b32_e32 v18, 8, v14
	v_xor_b32_e32 v19, 4, v14
	v_cmp_gt_i32_e32 vcc_lo, 32, v15
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v15, v14, v15, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v18
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v18, v14, v18, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v19
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v18, 2, v18
	v_lshlrev_b32_e32 v15, 2, v15
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v19, v14, v19, vcc_lo
	ds_bpermute_b32 v15, v15, v73
	v_lshlrev_b32_e32 v19, 2, v19
	s_wait_dscnt 0x0
	v_add_f32_e32 v15, v73, v15
	ds_bpermute_b32 v18, v18, v15
	s_wait_dscnt 0x0
	v_add_f32_e32 v15, v15, v18
	ds_bpermute_b32 v18, v19, v15
	v_xor_b32_e32 v19, 2, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v19
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v19, v14, v19, vcc_lo
	v_lshlrev_b32_e32 v19, 2, v19
	s_wait_dscnt 0x0
	v_add_f32_e32 v15, v15, v18
	ds_bpermute_b32 v18, v19, v15
	v_xor_b32_e32 v19, 1, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v19
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v14, v19, vcc_lo
	v_lshlrev_b32_e32 v14, 2, v14
	s_wait_dscnt 0x0
	v_add_f32_e32 v15, v15, v18
	ds_bpermute_b32 v14, v14, v15
	s_wait_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v66, v14
	v_and_b32_e32 v14, 0x7fffffff, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.287:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
	v_lshrrev_b32_e32 v14, 16, v14
; %bb.288:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.289:
	v_lshrrev_b32_e32 v14, 16, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v14, 64, v14
; %bb.290:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v14, 16, v14
	v_mul_f32_e32 v15, s86, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v14, 0x7fffffff, v15
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.291:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
	v_lshrrev_b32_e32 v14, 16, v14
; %bb.292:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.293:
	v_lshrrev_b32_e32 v14, 16, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v14, 64, v14
; %bb.294:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v14, 16, v14
	v_add_f32_e32 v22, v22, v14
.LBB_295:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s1, exec_lo
	v_cmpx_ne_u64_e32 0, v[16:17]
	s_cbranch_execz .LBB_305
; %bb.296:
	v_mbcnt_lo_u32_b32 v14, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v15, 16, v14
	v_xor_b32_e32 v16, 8, v14
	v_xor_b32_e32 v17, 4, v14
	v_cmp_gt_i32_e32 vcc_lo, 32, v15
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v15, v14, v15, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v16
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v16, v14, v16, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v17
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v16, 2, v16
	v_lshlrev_b32_e32 v15, 2, v15
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v17, v14, v17, vcc_lo
	ds_bpermute_b32 v15, v15, v76
	v_lshlrev_b32_e32 v17, 2, v17
	s_wait_dscnt 0x0
	v_add_f32_e32 v15, v76, v15
	ds_bpermute_b32 v16, v16, v15
	s_wait_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	ds_bpermute_b32 v16, v17, v15
	v_xor_b32_e32 v17, 2, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v17
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v17, v14, v17, vcc_lo
	v_lshlrev_b32_e32 v17, 2, v17
	s_wait_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	ds_bpermute_b32 v16, v17, v15
	v_xor_b32_e32 v17, 1, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v17
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v14, v17, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v15, v15, v16 :: v_dual_lshlrev_b32 v14, 2, v14
	ds_bpermute_b32 v14, v14, v15
	s_wait_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v67, v14
	v_and_b32_e32 v14, 0x7fffffff, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.297:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
	v_lshrrev_b32_e32 v14, 16, v14
; %bb.298:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.299:
	v_lshrrev_b32_e32 v14, 16, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v14, 64, v14
; %bb.300:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v14, 16, v14
	v_mul_f32_e32 v15, s85, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v14, 0x7fffffff, v15
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.301:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
	v_lshrrev_b32_e32 v14, 16, v14
; %bb.302:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.303:
	v_lshrrev_b32_e32 v14, 16, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v14, 64, v14
; %bb.304:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v14, 16, v14
	v_add_f32_e32 v22, v22, v14
.LBB_305:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s1, exec_lo
	v_cmpx_ne_u64_e32 0, v[10:11]
	s_cbranch_execz .LBB_315
; %bb.306:
	v_mbcnt_lo_u32_b32 v10, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v11, 16, v10
	v_xor_b32_e32 v14, 8, v10
	v_xor_b32_e32 v15, 4, v10
	v_cmp_gt_i32_e32 vcc_lo, 32, v11
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v11, v10, v11, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v10, v14, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v15
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v14, 2, v14
	v_lshlrev_b32_e32 v11, 2, v11
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v15, v10, v15, vcc_lo
	ds_bpermute_b32 v11, v11, v75
	v_lshlrev_b32_e32 v15, 2, v15
	s_wait_dscnt 0x0
	v_add_f32_e32 v11, v75, v11
	ds_bpermute_b32 v14, v14, v11
	s_wait_dscnt 0x0
	v_add_f32_e32 v11, v11, v14
	ds_bpermute_b32 v14, v15, v11
	v_xor_b32_e32 v15, 2, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v15
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v15, v10, v15, vcc_lo
	v_lshlrev_b32_e32 v15, 2, v15
	s_wait_dscnt 0x0
	v_add_f32_e32 v11, v11, v14
	ds_bpermute_b32 v14, v15, v11
	v_xor_b32_e32 v15, 1, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v15
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v10, v10, v15, vcc_lo
	v_lshlrev_b32_e32 v10, 2, v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v11, v11, v14
	ds_bpermute_b32 v10, v10, v11
	s_wait_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v11, v64, v10
	v_and_b32_e32 v10, 0x7fffffff, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.307:
	v_bfe_u32 v10, v11, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v10, v11, v10, 0x7fff
                                        ; implicit-def: $vgpr11
	v_lshrrev_b32_e32 v10, 16, v10
; %bb.308:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.309:
	v_lshrrev_b32_e32 v10, 16, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v10, 64, v10
; %bb.310:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v10, 16, v10
	v_mul_f32_e32 v11, s84, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v10, 0x7fffffff, v11
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.311:
	v_bfe_u32 v10, v11, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v10, v11, v10, 0x7fff
                                        ; implicit-def: $vgpr11
	v_lshrrev_b32_e32 v10, 16, v10
; %bb.312:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.313:
	v_lshrrev_b32_e32 v10, 16, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v10, 64, v10
; %bb.314:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v10, 16, v10
	v_add_f32_e32 v22, v22, v10
.LBB_315:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s1, exec_lo
	v_cmpx_ne_u64_e32 0, v[12:13]
	s_cbranch_execz .LBB_325
; %bb.316:
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
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v10, v13, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v12, 2, v12
	v_lshlrev_b32_e32 v11, 2, v11
	ds_bpermute_b32 v11, v11, v78
	s_wait_dscnt 0x0
	v_add_f32_e32 v11, v78, v11
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
	v_mul_f32_e32 v11, v65, v10
	v_and_b32_e32 v10, 0x7fffffff, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.317:
	v_bfe_u32 v10, v11, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v10, v11, v10, 0x7fff
                                        ; implicit-def: $vgpr11
	v_lshrrev_b32_e32 v10, 16, v10
; %bb.318:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.319:
	v_lshrrev_b32_e32 v10, 16, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v10, 64, v10
; %bb.320:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v10, 16, v10
	v_mul_f32_e32 v11, s59, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v10, 0x7fffffff, v11
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.321:
	v_bfe_u32 v10, v11, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v10, v11, v10, 0x7fff
                                        ; implicit-def: $vgpr11
	v_lshrrev_b32_e32 v10, 16, v10
; %bb.322:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.323:
	v_lshrrev_b32_e32 v10, 16, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v10, 64, v10
; %bb.324:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v10, 16, v10
	v_add_f32_e32 v22, v22, v10
.LBB_325:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s1, exec_lo
	v_cmpx_ne_u64_e32 0, v[6:7]
	s_cbranch_execz .LBB_335
; %bb.326:
	v_mbcnt_lo_u32_b32 v6, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v7, 16, v6
	v_xor_b32_e32 v10, 8, v6
	v_xor_b32_e32 v11, 4, v6
	v_cmp_gt_i32_e32 vcc_lo, 32, v7
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v7, v6, v7, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v10
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v10, v6, v10, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v10, 2, v10
	v_lshlrev_b32_e32 v7, 2, v7
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v11, v6, v11, vcc_lo
	ds_bpermute_b32 v7, v7, v77
	v_lshlrev_b32_e32 v11, 2, v11
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v77, v7
	ds_bpermute_b32 v10, v10, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_bpermute_b32 v10, v11, v7
	v_xor_b32_e32 v11, 2, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v11
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v11, v6, v11, vcc_lo
	v_lshlrev_b32_e32 v11, 2, v11
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_bpermute_b32 v10, v11, v7
	v_xor_b32_e32 v11, 1, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v11
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v6, v6, v11, vcc_lo
	v_lshlrev_b32_e32 v6, 2, v6
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_bpermute_b32 v6, v6, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v7, v62, v6
	v_and_b32_e32 v6, 0x7fffffff, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v6
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.327:
	v_bfe_u32 v6, v7, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v6, v7, v6, 0x7fff
                                        ; implicit-def: $vgpr7
	v_lshrrev_b32_e32 v6, 16, v6
; %bb.328:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.329:
	v_lshrrev_b32_e32 v6, 16, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v6, 64, v6
; %bb.330:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v6, 16, v6
	v_mul_f32_e32 v7, s33, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v6, 0x7fffffff, v7
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v6
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.331:
	v_bfe_u32 v6, v7, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v6, v7, v6, 0x7fff
                                        ; implicit-def: $vgpr7
	v_lshrrev_b32_e32 v6, 16, v6
; %bb.332:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.333:
	v_lshrrev_b32_e32 v6, 16, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v6, 64, v6
; %bb.334:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v6, 16, v6
	v_add_f32_e32 v22, v22, v6
.LBB_335:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v6, 0x7fffffff, v22
	s_mov_b32 s1, exec_lo
                                        ; implicit-def: $vgpr10
	v_cmpx_gt_u32_e32 0x7f800001, v6
	s_wait_alu 0xfffe
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB_339
; %bb.336:
	v_bfe_u32 v6, v22, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v6, v22, v6, 0x7fff
                                        ; implicit-def: $vgpr22
	v_lshrrev_b32_e32 v10, 16, v6
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execnz .LBB_340
.LBB_337:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v11, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB_341
.LBB_338:
	v_lshlrev_b32_e32 v6, 5, v33
	s_lshl_b64 s[4:5], s[16:17], 1
	v_lshrrev_b32_e32 v19, 6, v8
	s_wait_alu 0xfffe
	s_add_nc_u64 s[4:5], s[64:65], s[4:5]
	v_lshrrev_b32_e32 v20, 10, v8
	s_clause 0x1
	global_load_b128 v[11:14], v6, s[4:5]
	global_load_b128 v[15:18], v6, s[4:5] offset:16
	v_and_b32_e32 v6, 15, v8
	v_lshrrev_b32_e32 v22, 18, v8
	v_lshrrev_b32_e32 v7, 2, v8
	v_and_b32_e32 v19, 60, v19
	v_and_b32_e32 v20, 60, v20
	v_lshlrev_b32_e32 v6, 2, v6
	v_and_b32_e32 v22, 60, v22
	v_lshrrev_b32_e32 v21, 14, v8
	ds_load_b32 v19, v19
	ds_load_b32 v20, v20
	ds_load_b32 v6, v6
	ds_load_b32 v22, v22
	v_and_b32_e32 v7, 60, v7
	v_lshrrev_b32_e32 v24, 22, v8
	v_lshrrev_b32_e32 v8, 26, v8
	ds_load_b32 v7, v7
	v_and_b32_e32 v8, 60, v8
	ds_load_b32 v8, v8
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v23, 16, v11
	v_and_b32_e32 v11, 0xffff0000, v11
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v6, v6, v23, 0
	v_and_b32_e32 v21, 60, v21
	v_and_b32_e32 v23, 60, v24
	v_lshlrev_b32_e32 v24, 16, v12
	v_and_b32_e32 v12, 0xffff0000, v12
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v6, v7, v11
	ds_load_b32 v21, v21
	ds_load_b32 v11, v23
	v_fmac_f32_e32 v6, v19, v24
	v_lshrrev_b32_e32 v19, 2, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v6, v20, v12
	v_lshrrev_b32_e32 v12, 6, v9
	v_lshrrev_b32_e32 v20, 10, v9
	v_and_b32_e32 v19, 60, v19
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v12, 60, v12
	v_and_b32_e32 v20, 60, v20
	ds_load_b32 v19, v19
	ds_load_b32 v12, v12
	ds_load_b32 v20, v20
	v_lshlrev_b32_e32 v23, 16, v13
	v_and_b32_e32 v7, 15, v9
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v6, v21, v23 :: v_dual_and_b32 v13, 0xffff0000, v13
	v_lshlrev_b32_e32 v21, 16, v14
	v_and_b32_e32 v14, 0xffff0000, v14
	v_fmac_f32_e32 v6, v22, v13
	v_lshrrev_b32_e32 v13, 14, v9
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v6, v11, v21
	v_and_b32_e32 v13, 60, v13
	v_lshrrev_b32_e32 v11, 18, v9
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v6, v8, v14
	s_wait_loadcnt 0x0
	v_and_b32_e32 v14, 0xffff0000, v15
	v_lshlrev_b32_e32 v7, 2, v7
	v_lshrrev_b32_e32 v8, 22, v9
	ds_load_b32 v13, v13
	v_lshlrev_b32_e32 v21, 16, v15
	v_and_b32_e32 v11, 60, v11
	ds_load_b32 v7, v7
	v_and_b32_e32 v8, 60, v8
	v_and_b32_e32 v15, 0xffff0000, v16
	ds_load_b32 v8, v8
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v6, v7, v21
	v_lshrrev_b32_e32 v7, 26, v9
	ds_load_b32 v9, v11
	v_dual_fmac_f32 v6, v19, v14 :: v_dual_and_b32 v7, 60, v7
	v_and_b32_e32 v14, 7, v63
	ds_load_b32 v7, v7
	v_lshlrev_b32_e32 v11, 16, v16
	v_and_b32_e32 v16, 0xffff0000, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v6, v12, v11
	v_cvt_f32_ubyte0_e32 v12, v14
	v_lshlrev_b32_e32 v14, 16, v17
	v_fmac_f32_e32 v6, v20, v15
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v12, 0x3b000000, v12
	v_fmac_f32_e32 v6, v13, v14
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v6, v9, v16 :: v_dual_lshlrev_b32 v13, 16, v18
	v_dual_fmac_f32 v6, v8, v13 :: v_dual_and_b32 v11, 0x7f, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v15, v11, 20, 0x3c000000
	v_cmp_lt_u32_e32 vcc_lo, 7, v11
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v12, v12, v15, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v11
	v_lshlrev_b32_e32 v11, 24, v63
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v9, 0x7fc00000, v12, vcc_lo
	v_and_b32_e32 v12, 0xffff0000, v18
	v_and_or_b32 v8, 0x80000000, v11, v9
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v7, v12
	v_fma_f32 v11, v6, v8, 0
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_i32_e64 s91, v32
	s_cbranch_execz .LBB_345
	s_branch .LBB_342
.LBB_339:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB_337
.LBB_340:
	v_lshrrev_b32_e32 v6, 16, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v10, 64, v6
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v11, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execnz .LBB_338
.LBB_341:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_i32_e64 s91, v32
	s_cbranch_execz .LBB_345
.LBB_342:
	v_and_b32_e32 v6, 31, v0
	s_lshl_b64 s[4:5], s[16:17], 1
	s_mov_b32 s3, 0
	s_wait_alu 0xfffe
	s_add_nc_u64 s[4:5], s[64:65], s[4:5]
	v_dual_mov_b32 v8, 0 :: v_dual_lshlrev_b32 v7, 3, v6
	v_or_b32_e32 v9, 32, v6
	v_lshlrev_b32_e32 v6, 5, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v4, vcc_lo, v4, v7
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	s_wait_alu 0xfffe
	v_add_co_u32 v6, s0, s4, v6
	v_add_co_u32 v4, vcc_lo, s48, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, s49, v5, vcc_lo
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v7, null, s5, 0, s0
	v_add_co_u32 v4, vcc_lo, 0x100, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	v_add_co_u32 v6, vcc_lo, 0x41c, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
.LBB_343:                              ; =>This Inner Loop Header: Depth=1
	global_load_b64 v[20:21], v[4:5], off th:TH_LOAD_NT
	s_clause 0x1
	global_load_b128 v[12:15], v[6:7], off offset:-28
	global_load_b128 v[16:19], v[6:7], off offset:-12
	v_add_co_u32 v22, vcc_lo, v2, v9
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v23, null, v3, v8, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, 32
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, 0, v8, vcc_lo
	global_load_i8 v22, v[22:23], off
	v_add_co_u32 v4, vcc_lo, 0x100, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	v_add_co_u32 v6, vcc_lo, 0x400, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	s_wait_loadcnt 0x3
	v_and_b32_e32 v23, 15, v20
	v_lshrrev_b32_e32 v24, 2, v20
	v_lshrrev_b32_e32 v25, 6, v20
	v_lshrrev_b32_e32 v26, 10, v20
	v_lshrrev_b32_e32 v27, 14, v20
	v_lshlrev_b32_e32 v23, 2, v23
	v_and_b32_e32 v24, 60, v24
	v_and_b32_e32 v25, 60, v25
	v_lshrrev_b32_e32 v28, 18, v20
	v_lshrrev_b32_e32 v29, 22, v20
	ds_load_b32 v23, v23
	ds_load_b32 v24, v24
	ds_load_b32 v25, v25
	v_lshrrev_b32_e32 v20, 26, v20
	v_and_b32_e32 v26, 60, v26
	v_and_b32_e32 v27, 60, v27
	v_and_b32_e32 v28, 60, v28
	v_and_b32_e32 v29, 60, v29
	v_and_b32_e32 v20, 60, v20
	ds_load_b32 v26, v26
	ds_load_b32 v27, v27
	ds_load_b32 v28, v28
	ds_load_b32 v29, v29
	ds_load_b32 v20, v20
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v38, 16, v12
	v_lshlrev_b32_e32 v39, 16, v14
	v_and_b32_e32 v12, 0xffff0000, v12
	v_lshrrev_b32_e32 v31, 2, v21
	v_lshrrev_b32_e32 v32, 6, v21
	v_lshrrev_b32_e32 v34, 10, v21
	v_lshrrev_b32_e32 v35, 14, v21
	v_lshrrev_b32_e32 v36, 18, v21
	v_lshrrev_b32_e32 v37, 22, v21
	s_wait_dscnt 0x7
	v_fma_f32 v23, v23, v38, 0
	v_and_b32_e32 v30, 15, v21
	v_lshrrev_b32_e32 v21, 26, v21
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v23, v24, v12
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v24, 16, v16
	v_and_b32_e32 v12, 0xffff0000, v14
	v_and_b32_e32 v16, 0xffff0000, v16
	v_lshlrev_b32_e32 v14, 16, v15
	v_and_b32_e32 v15, 0xffff0000, v15
	v_lshlrev_b32_e32 v38, 16, v13
	v_and_b32_e32 v13, 0xffff0000, v13
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v23, v25, v38
	v_lshlrev_b32_e32 v25, 2, v30
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v23, v26, v13
	v_and_b32_e32 v13, 60, v31
	v_and_b32_e32 v26, 60, v32
	ds_load_b32 v25, v25
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v23, v27, v39
	v_and_b32_e32 v27, 60, v34
	ds_load_b32 v26, v26
	ds_load_b32 v13, v13
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v23, v28, v12
	v_and_b32_e32 v12, 60, v36
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v23, v29, v14 :: v_dual_and_b32 v28, 60, v37
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v23, v20, v15 :: v_dual_and_b32 v14, 60, v21
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v23, v25, v24 :: v_dual_lshlrev_b32 v20, 16, v17
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v23, v13, v16 :: v_dual_and_b32 v30, 60, v35
	v_and_b32_e32 v13, 0xffff0000, v17
	s_wait_loadcnt 0x0
	v_and_b32_e32 v16, 7, v22
	v_lshlrev_b32_e32 v17, 16, v18
	v_fmac_f32_e32 v23, v26, v20
	ds_load_b32 v21, v27
	ds_load_b32 v27, v30
	ds_load_b32 v12, v12
	ds_load_b32 v15, v28
	ds_load_b32 v14, v14
	v_and_b32_e32 v20, 0x7f, v22
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_lt_u32_e64 s0, 7, v20
	s_wait_dscnt 0x4
	v_dual_mul_f32 v16, 0x3b000000, v16 :: v_dual_fmac_f32 v23, v21, v13
	v_and_b32_e32 v13, 0xffff0000, v18
	v_lshl_add_u32 v18, v20, 20, 0x3c000000
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v23, v27, v17
	v_lshlrev_b32_e32 v17, 16, v19
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v16, v16, v18, s0
	v_cmp_ne_u32_e64 s0, 0x7f, v20
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v23, v12, v13 :: v_dual_and_b32 v12, 0xffff0000, v19
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v13, 0x7fc00000, v16, s0
	v_cmp_le_i32_e64 s0, s91, v9
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v23, v15, v17
	v_and_or_b32 v13, 0x80000000, v22, v13
	s_or_b32 s3, s0, s3
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v23, v14, v12
	v_fmac_f32_e32 v11, v23, v13
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s3
	s_cbranch_execnz .LBB_343
; %bb.344:
	s_or_b32 exec_lo, exec_lo, s3
.LBB_345:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
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
	ds_bpermute_b32 v3, v3, v11
	v_lshlrev_b32_e32 v5, 2, v5
	s_wait_dscnt 0x0
	v_add_f32_e32 v3, v11, v3
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
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v5
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v5, v2, v5, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v33
	s_wait_dscnt 0x0
	v_dual_add_f32 v2, v3, v4 :: v_dual_lshlrev_b32 v3, 2, v5
	ds_bpermute_b32 v3, v3, v2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB_363
; %bb.346:
	v_mov_b32_e32 v4, 0
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v2, v3
	global_load_b32 v4, v4, s[60:61]
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v3, v2, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0x7fffffff, v3
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.347:
	v_bfe_u32 v2, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v2, v3, v2, 0x7fff
                                        ; implicit-def: $vgpr3
	v_lshrrev_b32_e32 v2, 16, v2
; %bb.348:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.349:
	v_lshrrev_b32_e32 v2, 16, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, 64, v2
; %bb.350:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_lshl_b32 s0, s89, 16
	s_wait_alu 0xfffe
	s_mul_f32 s1, s0, 0xbfb8aa3b
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2)
	s_xor_b32 s3, s1, 0x80000000
	s_rndne_f32 s4, s1
	s_wait_alu 0xfffe
	s_fmamk_f32 s3, s0, 0xbfb8aa3b, s3
	s_cmp_ngt_f32 s0, 0x42ce8ed0
	s_sub_f32 s1, s1, s4
	s_wait_alu 0xfffe
	s_fmamk_f32 s3, s0, 0xb2a5705f, s3
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_nlt_f32 s0, 0xc2b17218
	s_wait_alu 0xfffe
	s_add_f32 s1, s1, s3
	s_cvt_i32_f32 s3, s4
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s1, s1
	s_wait_alu 0xf1ff
	v_ldexp_f32 v3, s1, s3
	s_mov_b32 s3, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v3, 0x7f800000, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v3, 1.0, v3
	v_div_scale_f32 v4, null, v3, v3, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v5, v4
	v_xor_b32_e32 v4, 0x80000000, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v6, v4, v5, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v5, v6, v5
	v_div_scale_f32 v6, vcc_lo, 1.0, v3, 1.0
	v_mul_f32_e32 v7, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v8, v4, v7, v6
	v_fmac_f32_e32 v7, v8, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v4, v7
	s_wait_alu 0xfffd
	v_div_fmas_f32 v4, v6, v5, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v3, v4, v3, 1.0
	v_and_b32_e32 v4, 0x7fffffff, v3
	v_readfirstlane_b32 s0, v3
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v4
	s_cbranch_vccz .LBB_352
; %bb.351:
	s_bfe_u32 s1, s0, 0x10010
	s_wait_alu 0xfffe
	s_add_co_i32 s1, s0, s1
	s_wait_alu 0xfffe
	s_addk_co_i32 s1, 0x7fff
	s_wait_alu 0xfffe
	s_lshr_b32 s1, s1, 16
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB_353
	s_branch .LBB_354
.LBB_352:
                                        ; implicit-def: $sgpr1
.LBB_353:
	s_lshr_b32 s0, s0, 16
	s_wait_alu 0xfffe
	s_or_b32 s1, s0, 64
.LBB_354:
	v_lshlrev_b32_e32 v2, 16, v2
	s_wait_alu 0xfffe
	s_lshl_b32 s0, s1, 16
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v3, s0, v2
	v_and_b32_e32 v2, 0x7fffffff, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.355:
	v_bfe_u32 v2, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v2, v3, v2, 0x7fff
                                        ; implicit-def: $vgpr3
	v_lshrrev_b32_e32 v2, 16, v2
; %bb.356:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.357:
	v_lshrrev_b32_e32 v2, 16, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, 64, v2
; %bb.358:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_lshlrev_b32_e32 v3, 16, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 16, v2
	v_add_f32_e32 v2, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v3, 0x7fffffff, v2
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v3
                                        ; implicit-def: $vgpr3
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.359:
	v_bfe_u32 v3, v2, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v2, v2, v3, 0x7fff
	v_lshrrev_b32_e32 v3, 16, v2
                                        ; implicit-def: $vgpr2
; %bb.360:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.361:
	v_lshrrev_b32_e32 v2, 16, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v3, 64, v2
; %bb.362:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v5, 64, v3
	v_lshlrev_b32_e32 v4, 16, v3
	v_ashrrev_i32_e32 v2, 31, v1
	s_mul_u64 s[0:1], s[68:69], s[70:71]
	s_wait_alu 0xfffe
	s_lshl_b64 s[0:1], s[0:1], 1
	v_and_b32_e32 v4, 0x7fffffff, v4
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[0:1], s[66:67], s[0:1]
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_lt_u32_e32 vcc_lo, 0x7f800000, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v3, v5, vcc_lo
	s_wait_alu 0xfffe
	v_add_co_u32 v1, vcc_lo, s0, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s1, v2, vcc_lo
	global_store_b16 v[1:2], v3, off
.LBB_363:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s92
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB_368
; %bb.364:
	s_mov_b32 s3, exec_lo
	s_mov_b32 s1, exec_lo
	s_wait_alu 0xfffe
	v_mbcnt_lo_u32_b32 v1, s3, 0
                                        ; implicit-def: $vgpr2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v1
	s_cbranch_execz .LBB_366
; %bb.365:
	s_bcnt1_i32_b32 s3, s3
	s_wait_alu 0xfffe
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, s3
	s_wait_loadcnt 0x0
	global_atomic_add_u32 v2, v2, v3, s[34:35] offset:4 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	global_inv scope:SCOPE_DEV
.LBB_366:
	s_or_b32 exec_lo, exec_lo, s1
	v_readfirstlane_b32 s1, v2
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s1, v1
	s_add_co_i32 s1, s28, -1
	s_wait_alu 0xfffe
	v_cmp_eq_u32_e32 vcc_lo, s1, v1
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB_368
; %bb.367:
	v_mov_b32_e32 v1, 0
	s_clause 0x2
	global_store_b32 v1, v1, s[34:35] scope:SCOPE_DEV
	global_store_b32 v1, v1, s[34:35] offset:4 scope:SCOPE_DEV
	global_store_b32 v1, v1, s[34:35] offset:8 scope:SCOPE_DEV
.LBB_368:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB_369:
	s_wait_alu 0xfffe
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB_525
; %bb.370:
	s_add_co_i32 s0, s55, 7
	s_add_co_i32 s1, s54, 7
	s_wait_alu 0xfffe
	s_ashr_i32 s3, s0, 31
	s_ashr_i32 s5, s1, 31
	s_wait_alu 0xfffe
	s_lshr_b32 s3, s3, 29
	s_lshr_b32 s5, s5, 29
	s_wait_alu 0xfffe
	s_add_co_i32 s0, s0, s3
	s_add_co_i32 s1, s1, s5
	s_bfe_u32 s4, s30, 0x30012
	s_wait_alu 0xfffe
	s_ashr_i32 s0, s0, 3
	s_ashr_i32 s3, s1, 3
	s_cmp_ge_i32 s29, s58
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s29, s58
	s_cselect_b32 s1, -1, 0
	s_sub_co_i32 s10, s29, s58
	s_cmp_lt_i32 s4, 2
	s_cbranch_scc1 .LBB_375
; %bb.371:
	s_cmp_lt_i32 s4, 3
	s_wait_alu 0xfffe
	s_mul_i32 s7, s56, s3
	s_cbranch_scc1 .LBB_376
; %bb.372:
	s_cmp_lt_i32 s4, 4
	s_cbranch_scc1 .LBB_377
; %bb.373:
	s_cmp_lg_u32 s4, 4
	s_cbranch_scc0 .LBB_378
; %bb.374:
	s_wait_alu 0xfffe
	s_add_co_i32 s5, s7, -1
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s10, s5
	s_cselect_b32 s5, -1, 0
	s_add_co_i32 s8, s0, -1
	s_wait_alu 0xfffe
	s_and_b32 s5, s6, s5
	s_cmp_eq_u32 s29, s8
	s_cselect_b32 s8, -1, 0
	s_wait_alu 0xfffe
	s_or_b32 s5, s5, s8
	s_mov_b32 s8, 0
	s_branch .LBB_379
.LBB_375:
                                        ; implicit-def: $sgpr5
	s_branch .LBB_387
.LBB_376:
                                        ; implicit-def: $sgpr5
	s_branch .LBB_385
.LBB_377:
                                        ; implicit-def: $sgpr5
	s_branch .LBB_383
.LBB_378:
	s_mov_b32 s8, -1
                                        ; implicit-def: $sgpr5
.LBB_379:
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_382
; %bb.380:
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_mov_b32 s5, 0
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB_382
; %bb.381:
	s_abs_i32 s5, s7
	s_add_co_i32 s11, s10, 1
	s_wait_alu 0xfffe
	s_cvt_f32_u32 s8, s5
	s_sub_co_i32 s9, 0, s5
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s8
	v_readfirstlane_b32 s8, v1
	s_mul_f32 s8, s8, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s8, s8
	s_wait_alu 0xfffe
	s_mul_i32 s9, s9, s8
	s_wait_alu 0xfffe
	s_mul_hi_u32 s9, s8, s9
	s_wait_alu 0xfffe
	s_add_co_i32 s8, s8, s9
	s_wait_alu 0xfffe
	s_mul_hi_u32 s8, s11, s8
	s_wait_alu 0xfffe
	s_mul_i32 s8, s8, s5
	s_wait_alu 0xfffe
	s_sub_co_i32 s8, s11, s8
	s_wait_alu 0xfffe
	s_sub_co_i32 s9, s8, s5
	s_cmp_ge_u32 s8, s5
	s_wait_alu 0xfffe
	s_cselect_b32 s8, s9, s8
	s_wait_alu 0xfffe
	s_sub_co_i32 s9, s8, s5
	s_cmp_ge_u32 s8, s5
	s_wait_alu 0xfffe
	s_cselect_b32 s5, s9, s8
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s5, -1, 0
.LBB_382:
	s_cbranch_execnz .LBB_384
.LBB_383:
	s_add_co_i32 s5, s0, -1
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s29, s5
	s_cselect_b32 s5, -1, 0
.LBB_384:
	s_cbranch_execnz .LBB_386
.LBB_385:
	s_wait_alu 0xfffe
	s_add_co_i32 s7, s7, -1
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s10, s7
	s_cselect_b32 s5, -1, 0
	s_wait_alu 0xfffe
	s_and_b32 s5, s6, s5
.LBB_386:
	s_cbranch_execnz .LBB_391
.LBB_387:
	s_cmp_gt_i32 s4, 0
	s_mov_b32 s4, 0
	s_cbranch_scc0 .LBB_389
; %bb.388:
	s_add_co_i32 s5, s58, -1
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s29, s5
	s_cselect_b32 s5, -1, 0
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB_390
	s_branch .LBB_391
.LBB_389:
                                        ; implicit-def: $sgpr5
.LBB_390:
	s_add_co_i32 s31, s31, -1
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s29, s31
	s_cselect_b32 s5, -1, 0
.LBB_391:
	s_and_b32 s6, s30, 0xffff
	s_bitcmp0_b32 s30, 16
	s_mov_b32 s7, 0
	s_cbranch_scc1 .LBB_393
; %bb.392:
	s_ashr_i32 s4, s29, 31
	s_wait_alu 0xfffe
	s_lshr_b32 s4, s4, 30
	s_wait_alu 0xfffe
	s_add_co_i32 s4, s29, s4
	s_wait_alu 0xfffe
	s_and_b32 s4, s4, -4
	s_wait_alu 0xfffe
	s_sub_co_i32 s4, s29, s4
	s_wait_alu 0xfffe
	s_add_co_i32 s4, s4, 1
	s_wait_alu 0xfffe
	s_mul_i32 s4, s4, s6
	s_and_not1_b32 vcc_lo, exec_lo, s7
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB_394
	s_branch .LBB_395
.LBB_393:
                                        ; implicit-def: $sgpr4
.LBB_394:
	s_wait_alu 0xfffe
	s_and_b32 s4, s5, exec_lo
	s_cselect_b32 s4, s6, 0
.LBB_395:
	s_wait_alu 0xfffe
	s_cmp_lt_i32 s4, 1
	s_cbranch_scc1 .LBB_397
.LBB_396:                              ; =>This Inner Loop Header: Depth=1
	s_add_co_i32 s4, s4, -1
	s_sleep 0x7f
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s4, 0
	s_cbranch_scc0 .LBB_396
.LBB_397:
	s_and_b32 vcc_lo, exec_lo, s1
	s_mov_b32 s1, -1
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB_447
; %bb.398:
	s_abs_i32 s1, s0
	s_abs_i32 s6, s29
	s_wait_alu 0xfffe
	s_cvt_f32_u32 s4, s1
	s_sub_co_i32 s5, 0, s1
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v54, 0
	s_wait_alu 0xfffe
	v_rcp_iflag_f32_e32 v1, s4
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v65, 0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_2)
	v_readfirstlane_b32 s4, v1
	s_mul_f32 s4, s4, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_cvt_u32_f32 s4, s4
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2)
	s_mul_i32 s5, s5, s4
	s_wait_alu 0xfffe
	s_mul_hi_u32 s5, s4, s5
	s_wait_alu 0xfffe
	s_add_co_i32 s4, s4, s5
	s_xor_b32 s5, s29, s0
	s_wait_alu 0xfffe
	s_mul_hi_u32 s4, s6, s4
	s_ashr_i32 s5, s5, 31
	s_wait_alu 0xfffe
	s_mul_i32 s7, s4, s1
	s_wait_alu 0xfffe
	s_sub_co_i32 s6, s6, s7
	s_add_co_i32 s7, s4, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s8, s6, s1
	s_cmp_ge_u32 s6, s1
	s_cselect_b32 s4, s7, s4
	s_wait_alu 0xfffe
	s_cselect_b32 s6, s8, s6
	s_add_co_i32 s7, s4, 1
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s6, s1
	s_cselect_b32 s1, s7, s4
	s_ashr_i32 s7, s53, 31
	s_wait_alu 0xfffe
	s_xor_b32 s1, s1, s5
	s_wait_alu 0xfffe
	s_sub_co_i32 s4, s1, s5
	s_lshr_b32 s1, s7, 28
	s_wait_alu 0xfffe
	s_mul_i32 s0, s4, s0
	s_add_co_i32 s1, s53, s1
	s_wait_alu 0xfffe
	s_sub_co_i32 s0, s29, s0
	s_ashr_i32 s11, s1, 4
	s_wait_alu 0xfffe
	v_lshl_or_b32 v34, s0, 3, v61
	s_lshr_b32 s0, s53, 31
	s_wait_alu 0xfffe
	s_add_co_i32 s0, s53, s0
	s_wait_alu 0xfffe
	s_ashr_i32 s0, s0, 1
	v_cmp_gt_i32_e32 vcc_lo, s55, v34
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v1, 0, v34, vcc_lo
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[37:38], null, v1, s0, 0
	v_mad_co_i64_i32 v[35:36], null, v1, s11, 0
	v_cmp_gt_i32_e64 s0, s11, v33
	s_wait_kmcnt 0x0
	v_add_co_u32 v3, s1, s36, v37
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_co_ci_u32_e64 v4, null, s37, v38, s1
	v_add_co_u32 v1, s1, s38, v35
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v2, null, s39, v36, s1
	s_and_saveexec_b32 s5, s0
	s_cbranch_execz .LBB_400
; %bb.399:
	v_lshlrev_b32_e32 v5, 3, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v5, s1, v3, v5
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v6, null, 0, v4, s1
	v_add_co_u32 v7, s1, v1, v33
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v8, null, 0, v2, s1
	global_load_b64 v[53:54], v[5:6], off th:TH_LOAD_NT
	global_load_u8 v65, v[7:8], off
.LBB_400:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s5
	v_add_co_u32 v7, s1, s42, v37
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v8, null, s43, v38, s1
	v_add_co_u32 v5, s1, s44, v35
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v6, null, s45, v36, s1
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v66, 0
	s_and_saveexec_b32 s5, s0
	s_cbranch_execz .LBB_402
; %bb.401:
	v_lshlrev_b32_e32 v9, 3, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v9, s1, v7, v9
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v10, null, 0, v8, s1
	v_add_co_u32 v11, s1, v5, v33
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v12, null, 0, v6, s1
	global_load_b64 v[51:52], v[9:10], off th:TH_LOAD_NT
	global_load_u8 v66, v[11:12], off
.LBB_402:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s5
	v_dual_mov_b32 v50, 0 :: v_dual_lshlrev_b32 v9, 4, v33
	s_ashr_i32 s5, s4, 31
	s_mov_b32 s6, s53
	v_mov_b32_e32 v47, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v9, 0, v9, s0
	s_wait_alu 0xfffe
	s_mul_u64 s[0:1], s[4:5], s[6:7]
	v_mov_b32_e32 v49, 0
	s_wait_alu 0xfffe
	s_lshl_b64 s[0:1], s[0:1], 1
	v_mov_b32_e32 v63, 0
	v_lshlrev_b32_e32 v9, 1, v9
	s_wait_alu 0xfffe
	s_add_nc_u64 s[6:7], s[12:13], s[0:1]
	s_clause 0x1
	global_load_b128 v[25:28], v9, s[6:7] offset:16
	global_load_b128 v[29:32], v9, s[6:7]
	v_or_b32_e32 v9, 32, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s11, v9
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB_404
; %bb.403:
	v_lshlrev_b32_e32 v10, 3, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v10, s1, v3, v10
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v11, null, 0, v4, s1
	v_add_co_u32 v12, s1, v1, v33
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v13, null, 0, v2, s1
	global_load_b64 v[49:50], v[10:11], off offset:256 th:TH_LOAD_NT
	global_load_u8 v63, v[12:13], off offset:32
.LBB_404:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s8
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v64, 0
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB_406
; %bb.405:
	v_lshlrev_b32_e32 v10, 3, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v10, s1, v7, v10
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v11, null, 0, v8, s1
	v_add_co_u32 v12, s1, v5, v33
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v13, null, 0, v6, s1
	global_load_b64 v[47:48], v[10:11], off offset:256 th:TH_LOAD_NT
	global_load_u8 v64, v[12:13], off offset:32
.LBB_406:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v46, 0 :: v_dual_lshlrev_b32 v9, 4, v9
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v59, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v9, 0, v9, s0
	v_lshlrev_b32_e32 v9, 1, v9
	s_clause 0x1
	global_load_b128 v[17:20], v9, s[6:7] offset:16
	global_load_b128 v[21:24], v9, s[6:7]
	v_or_b32_e32 v9, 64, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s11, v9
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB_408
; %bb.407:
	v_lshlrev_b32_e32 v10, 3, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v10, s1, v3, v10
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v11, null, 0, v4, s1
	v_add_co_u32 v12, s1, v1, v33
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v13, null, 0, v2, s1
	global_load_b64 v[45:46], v[10:11], off offset:512 th:TH_LOAD_NT
	global_load_u8 v59, v[12:13], off offset:64
.LBB_408:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s8
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v60, 0
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB_410
; %bb.409:
	v_lshlrev_b32_e32 v10, 3, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v10, s1, v7, v10
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v11, null, 0, v8, s1
	v_add_co_u32 v12, s1, v5, v33
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v13, null, 0, v6, s1
	global_load_b64 v[43:44], v[10:11], off offset:512 th:TH_LOAD_NT
	global_load_u8 v60, v[12:13], off offset:64
.LBB_410:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v42, 0 :: v_dual_lshlrev_b32 v9, 4, v9
	v_or_b32_e32 v55, 0x60, v33
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v41, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v9, 0, v9, s0
	v_mov_b32_e32 v57, 0
	v_cmp_gt_i32_e64 s0, s11, v55
	v_lshlrev_b32_e32 v13, 1, v9
	s_clause 0x1
	global_load_b128 v[9:12], v13, s[6:7] offset:16
	global_load_b128 v[13:16], v13, s[6:7]
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB_412
; %bb.411:
	v_lshlrev_b32_e32 v40, 3, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v3, s1, v3, v40
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v4, null, 0, v4, s1
	v_add_co_u32 v1, s1, v1, v33
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v2, null, 0, v2, s1
	global_load_b64 v[41:42], v[3:4], off offset:768 th:TH_LOAD_NT
	global_load_u8 v57, v[1:2], off offset:96
.LBB_412:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s8
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v58, 0
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB_414
; %bb.413:
	v_lshlrev_b32_e32 v1, 3, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v1, s1, v7, v1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v2, null, 0, v8, s1
	v_add_co_u32 v3, s1, v5, v33
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v4, null, 0, v6, s1
	global_load_b64 v[39:40], v[1:2], off offset:768 th:TH_LOAD_NT
	global_load_u8 v58, v[3:4], off offset:96
.LBB_414:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s8
	v_lshlrev_b32_e32 v1, 4, v55
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, 0, v1, s0
	v_lshlrev_b32_e32 v5, 1, v1
	s_clause 0x1
	global_load_b128 v[1:4], v5, s[6:7] offset:16
	global_load_b128 v[5:8], v5, s[6:7]
	v_cmpx_gt_u32_e32 16, v0
	s_cbranch_execz .LBB_416
; %bb.415:
	v_and_b32_e32 v55, 7, v0
	v_lshlrev_b32_e32 v62, 28, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v56, v55, 22, 0.5
	v_cmp_ne_u32_e64 s0, 1, v55
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v56, 0.5, v56, s0
	v_cmp_ne_u32_e64 s0, 0, v55
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v55, 0, v56, s0
	v_lshlrev_b32_e32 v56, 2, v0
	v_and_or_b32 v55, 0x80000000, v62, v55
	ds_store_b32 v56, v55
.LBB_416:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s16, vcc_lo
	s_cbranch_execz .LBB_443
; %bb.417:
	v_sub_nc_u32_e32 v62, s11, v33
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v56, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 0, v62
	s_cbranch_execz .LBB_419
; %bb.418:
	v_and_b32_e32 v55, 15, v53
	v_lshrrev_b32_e32 v56, 2, v53
	v_lshrrev_b32_e32 v67, 6, v53
	v_lshlrev_b32_e32 v77, 16, v30
	v_lshlrev_b32_e32 v73, 16, v29
	v_lshlrev_b32_e32 v55, 2, v55
	v_and_b32_e32 v56, 60, v56
	v_and_b32_e32 v67, 60, v67
	v_and_b32_e32 v29, 0xffff0000, v29
	v_lshrrev_b32_e32 v68, 10, v53
	ds_load_b32 v55, v55
	ds_load_b32 v56, v56
	ds_load_b32 v67, v67
	v_lshrrev_b32_e32 v69, 14, v53
	v_lshrrev_b32_e32 v70, 18, v53
	v_lshrrev_b32_e32 v71, 22, v53
	v_lshrrev_b32_e32 v53, 26, v53
	v_lshrrev_b32_e32 v74, 2, v54
	v_and_b32_e32 v69, 60, v69
	v_lshrrev_b32_e32 v75, 6, v54
	v_and_b32_e32 v71, 60, v71
	v_and_b32_e32 v53, 60, v53
	v_lshrrev_b32_e32 v76, 10, v54
	v_lshrrev_b32_e32 v78, 18, v54
	v_lshrrev_b32_e32 v79, 22, v54
	v_lshrrev_b32_e32 v81, 18, v51
	v_lshrrev_b32_e32 v82, 22, v51
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_and_b32_e32 v81, 60, v81
	s_wait_dscnt 0x2
	v_fma_f32 v55, v55, v73, 0
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v55, v56, v29 :: v_dual_and_b32 v72, 15, v54
	v_lshrrev_b32_e32 v56, 14, v54
	v_lshrrev_b32_e32 v54, 26, v54
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v55, v67, v77
	v_lshlrev_b32_e32 v67, 16, v31
	v_and_b32_e32 v68, 60, v68
	v_lshlrev_b32_e32 v80, 16, v32
	v_and_b32_e32 v70, 60, v70
	ds_load_b32 v68, v68
	ds_load_b32 v69, v69
	ds_load_b32 v70, v70
	ds_load_b32 v71, v71
	ds_load_b32 v53, v53
	v_and_b32_e32 v31, 0xffff0000, v31
	v_and_b32_e32 v30, 0xffff0000, v30
	v_and_b32_e32 v54, 60, v54
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v55, v68, v30 :: v_dual_lshlrev_b32 v68, 2, v72
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v55, v69, v67
	v_and_b32_e32 v69, 60, v78
	v_lshrrev_b32_e32 v78, 6, v51
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v55, v70, v31 :: v_dual_and_b32 v32, 0xffff0000, v32
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v55, v71, v80 :: v_dual_and_b32 v82, 60, v82
	v_and_b32_e32 v72, 60, v74
	v_and_b32_e32 v74, 60, v75
	v_and_b32_e32 v75, 60, v76
	v_and_b32_e32 v76, 60, v79
	v_lshlrev_b32_e32 v79, 16, v25
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v55, v53, v32
	v_and_b32_e32 v56, 60, v56
	ds_load_b32 v68, v68
	ds_load_b32 v70, v72
	ds_load_b32 v72, v74
	ds_load_b32 v74, v75
	ds_load_b32 v56, v56
	ds_load_b32 v69, v69
	ds_load_b32 v75, v76
	ds_load_b32 v54, v54
	v_and_b32_e32 v76, 15, v51
	v_lshrrev_b32_e32 v71, 2, v51
	v_and_b32_e32 v25, 0xffff0000, v25
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v76, 2, v76
	v_and_b32_e32 v53, 60, v71
	v_and_b32_e32 v71, 60, v78
	v_lshrrev_b32_e32 v78, 14, v51
	ds_load_b32 v76, v76
	s_wait_dscnt 0x8
	v_fmac_f32_e32 v55, v68, v79
	ds_load_b32 v53, v53
	ds_load_b32 v68, v71
	v_lshrrev_b32_e32 v71, 10, v51
	v_lshrrev_b32_e32 v51, 26, v51
	s_wait_dscnt 0x9
	v_dual_fmac_f32 v55, v70, v25 :: v_dual_and_b32 v78, 60, v78
	v_lshrrev_b32_e32 v70, 6, v52
	v_and_b32_e32 v71, 60, v71
	v_and_b32_e32 v51, 60, v51
	ds_load_b32 v71, v71
	ds_load_b32 v78, v78
	ds_load_b32 v81, v81
	ds_load_b32 v82, v82
	ds_load_b32 v51, v51
	s_wait_dscnt 0x7
	v_fma_f32 v73, v76, v73, 0
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v73, v53, v29
	v_and_b32_e32 v53, 15, v52
	v_lshlrev_b32_e32 v29, 16, v27
	v_and_b32_e32 v27, 0xffff0000, v27
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v73, v68, v77
	v_lshrrev_b32_e32 v68, 2, v52
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v73, v71, v30 :: v_dual_lshlrev_b32 v30, 2, v53
	v_and_b32_e32 v53, 60, v68
	v_and_b32_e32 v68, 60, v70
	v_lshrrev_b32_e32 v70, 10, v52
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v73, v78, v67
	ds_load_b32 v30, v30
	ds_load_b32 v53, v53
	ds_load_b32 v67, v68
	v_lshrrev_b32_e32 v71, 22, v52
	s_wait_dscnt 0x5
	v_dual_fmac_f32 v73, v81, v31 :: v_dual_lshlrev_b32 v76, 16, v26
	v_lshrrev_b32_e32 v31, 14, v52
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_and_b32_e32 v71, 60, v71
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v73, v82, v80
	v_fmac_f32_e32 v55, v72, v76
	v_and_b32_e32 v31, 60, v31
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v73, v51, v32 :: v_dual_and_b32 v26, 0xffff0000, v26
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v73, v30, v79
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v55, v74, v26
	v_and_b32_e32 v68, 60, v70
	v_lshrrev_b32_e32 v70, 18, v52
	v_lshrrev_b32_e32 v52, 26, v52
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v73, v53, v25
	v_fmac_f32_e32 v55, v56, v29
	v_and_b32_e32 v53, 7, v66
	v_and_b32_e32 v70, 60, v70
	v_and_b32_e32 v25, 0xffff0000, v28
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v73, v67, v76
	v_dual_fmac_f32 v55, v69, v27 :: v_dual_lshlrev_b32 v30, 16, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v55, v75, v30 :: v_dual_and_b32 v52, 60, v52
	ds_load_b32 v32, v68
	ds_load_b32 v31, v31
	ds_load_b32 v51, v70
	ds_load_b32 v68, v71
	ds_load_b32 v52, v52
	v_fmac_f32_e32 v55, v54, v25
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v73, v32, v26
	v_cvt_f32_ubyte0_e32 v32, v53
	v_and_b32_e32 v28, 7, v65
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v73, v31, v29 :: v_dual_and_b32 v26, 0x7f, v66
	v_and_b32_e32 v56, 0x7f, v65
	v_mul_f32_e32 v31, 0x3b000000, v32
	v_cvt_f32_ubyte0_e32 v28, v28
	v_lshl_add_u32 v29, v26, 20, 0x3c000000
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v73, v51, v27
	v_lshl_add_u32 v53, v56, 20, 0x3c000000
	v_cmp_lt_u32_e32 vcc_lo, 7, v56
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v28, 0x3b000000, v28 :: v_dual_fmac_f32 v73, v68, v30
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v28, v28, v53, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, 7, v26
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v73, v52, v25
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v27, v31, v29, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v56
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v28, 0x7fc00000, v28 :: v_dual_lshlrev_b32 v29, 24, v65
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v26
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_and_or_b32 v28, 0x80000000, v29, v28
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v26, 0x7fc00000, v27 :: v_dual_lshlrev_b32 v27, 24, v66
	v_fma_f32 v56, v55, v28, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v25, 0x80000000, v27, v26
	v_fma_f32 v55, v73, v25, 0
.LBB_419:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_lt_i32_e32 32, v62
	s_cbranch_execz .LBB_421
; %bb.420:
	v_and_b32_e32 v25, 15, v49
	v_lshrrev_b32_e32 v26, 2, v49
	v_lshrrev_b32_e32 v27, 6, v49
	v_lshlrev_b32_e32 v65, 16, v22
	v_lshlrev_b32_e32 v51, 16, v21
	v_lshlrev_b32_e32 v25, 2, v25
	v_and_b32_e32 v26, 60, v26
	v_and_b32_e32 v27, 60, v27
	v_lshrrev_b32_e32 v28, 10, v49
	v_and_b32_e32 v21, 0xffff0000, v21
	ds_load_b32 v25, v25
	ds_load_b32 v26, v26
	ds_load_b32 v27, v27
	v_lshrrev_b32_e32 v29, 14, v49
	v_lshrrev_b32_e32 v30, 18, v49
	v_lshrrev_b32_e32 v31, 22, v49
	v_lshrrev_b32_e32 v32, 26, v49
	v_lshlrev_b32_e32 v68, 16, v24
	v_and_b32_e32 v29, 60, v29
	v_and_b32_e32 v30, 60, v30
	v_and_b32_e32 v31, 60, v31
	v_and_b32_e32 v49, 15, v50
	v_lshrrev_b32_e32 v52, 2, v50
	v_lshrrev_b32_e32 v53, 6, v50
	v_lshrrev_b32_e32 v54, 10, v50
	v_lshrrev_b32_e32 v66, 18, v50
	v_lshrrev_b32_e32 v67, 22, v50
	v_and_b32_e32 v24, 0xffff0000, v24
	v_lshrrev_b32_e32 v69, 18, v47
	v_lshrrev_b32_e32 v70, 22, v47
	s_wait_dscnt 0x2
	v_fma_f32 v25, v25, v51, 0
	v_and_b32_e32 v28, 60, v28
	v_and_b32_e32 v69, 60, v69
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v25, v26, v21
	v_lshrrev_b32_e32 v26, 14, v50
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v25, v27, v65
	v_lshrrev_b32_e32 v27, 26, v50
	v_lshlrev_b32_e32 v50, 16, v23
	v_and_b32_e32 v23, 0xffff0000, v23
	v_and_b32_e32 v22, 0xffff0000, v22
	v_and_b32_e32 v32, 60, v32
	ds_load_b32 v28, v28
	ds_load_b32 v29, v29
	ds_load_b32 v30, v30
	ds_load_b32 v31, v31
	ds_load_b32 v32, v32
	v_and_b32_e32 v27, 60, v27
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v25, v28, v22 :: v_dual_lshlrev_b32 v28, 2, v49
	v_and_b32_e32 v49, 60, v52
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v25, v29, v50
	v_and_b32_e32 v29, 60, v66
	v_lshrrev_b32_e32 v66, 6, v47
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v25, v30, v23
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v25, v31, v68
	v_lshrrev_b32_e32 v31, 2, v47
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v25, v32, v24 :: v_dual_and_b32 v52, 60, v53
	v_and_b32_e32 v53, 60, v54
	v_and_b32_e32 v54, 60, v67
	v_lshlrev_b32_e32 v67, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v17
	v_and_b32_e32 v26, 60, v26
	ds_load_b32 v28, v28
	ds_load_b32 v30, v49
	ds_load_b32 v49, v52
	ds_load_b32 v52, v53
	ds_load_b32 v26, v26
	ds_load_b32 v29, v29
	ds_load_b32 v53, v54
	ds_load_b32 v27, v27
	v_and_b32_e32 v54, 15, v47
	v_and_b32_e32 v31, 60, v31
	v_and_b32_e32 v32, 60, v66
	v_lshrrev_b32_e32 v66, 14, v47
	v_and_b32_e32 v70, 60, v70
	v_lshlrev_b32_e32 v54, 2, v54
	ds_load_b32 v54, v54
	s_wait_dscnt 0x8
	v_fmac_f32_e32 v25, v28, v67
	ds_load_b32 v28, v31
	ds_load_b32 v31, v32
	v_lshrrev_b32_e32 v32, 10, v47
	v_lshrrev_b32_e32 v47, 26, v47
	s_wait_dscnt 0x9
	v_fmac_f32_e32 v25, v30, v17
	v_lshrrev_b32_e32 v30, 2, v48
	v_and_b32_e32 v32, 60, v32
	v_and_b32_e32 v47, 60, v47
	s_wait_dscnt 0x2
	v_fma_f32 v51, v54, v51, 0
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v51, v28, v21
	v_lshlrev_b32_e32 v21, 16, v19
	v_lshlrev_b32_e32 v54, 16, v18
	v_and_b32_e32 v18, 0xffff0000, v18
	v_and_b32_e32 v19, 0xffff0000, v19
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v51, v31, v65
	v_lshrrev_b32_e32 v31, 6, v48
	v_fmac_f32_e32 v25, v49, v54
	v_lshrrev_b32_e32 v49, 22, v48
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v25, v52, v18
	v_and_b32_e32 v66, 60, v66
	ds_load_b32 v32, v32
	ds_load_b32 v66, v66
	ds_load_b32 v69, v69
	ds_load_b32 v70, v70
	ds_load_b32 v47, v47
	v_dual_fmac_f32 v25, v26, v21 :: v_dual_and_b32 v28, 15, v48
	v_and_b32_e32 v49, 60, v49
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v25, v29, v19
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v51, v32, v22 :: v_dual_lshlrev_b32 v22, 2, v28
	v_and_b32_e32 v28, 60, v30
	v_and_b32_e32 v30, 60, v31
	ds_load_b32 v22, v22
	ds_load_b32 v28, v28
	ds_load_b32 v30, v30
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v51, v66, v50
	v_lshrrev_b32_e32 v31, 10, v48
	v_lshrrev_b32_e32 v32, 18, v48
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v51, v69, v23
	v_lshrrev_b32_e32 v23, 14, v48
	v_lshrrev_b32_e32 v48, 26, v48
	v_and_b32_e32 v31, 60, v31
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v51, v70, v68
	v_and_b32_e32 v23, 60, v23
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v51, v47, v24
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v51, v22, v67
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v51, v28, v17 :: v_dual_and_b32 v32, 60, v32
	v_and_b32_e32 v17, 0xffff0000, v20
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v51, v30, v54 :: v_dual_and_b32 v48, 60, v48
	ds_load_b32 v24, v31
	ds_load_b32 v23, v23
	ds_load_b32 v31, v32
	ds_load_b32 v32, v49
	ds_load_b32 v47, v48
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v51, v24, v18
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v51, v23, v21 :: v_dual_lshlrev_b32 v22, 16, v20
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v51, v31, v19
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v25, v53, v22 :: v_dual_and_b32 v18, 0x7f, v64
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v51, v32, v22 :: v_dual_and_b32 v28, 0x7f, v63
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v25, v27, v17
	v_lshl_add_u32 v21, v18, 20, 0x3c000000
	v_and_b32_e32 v26, 7, v64
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cmp_lt_u32_e32 vcc_lo, 7, v28
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v51, v47, v17
	v_cvt_f32_ubyte0_e32 v24, v26
	v_and_b32_e32 v20, 7, v63
	v_lshl_add_u32 v26, v28, 20, 0x3c000000
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v23, 0x3b000000, v24
	v_cvt_f32_ubyte0_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v20, 0x3b000000, v20
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v20, v20, v26, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, 7, v18
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v19, v23, v21, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v28
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v20, 0x7fc00000, v20 :: v_dual_lshlrev_b32 v21, 24, v63
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_and_or_b32 v20, 0x80000000, v21, v20
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v18, 0x7fc00000, v19 :: v_dual_lshlrev_b32 v19, 24, v64
	v_fmac_f32_e32 v56, v25, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v17, 0x80000000, v19, v18
	v_fmac_f32_e32 v55, v51, v17
.LBB_421:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_lt_i32_e32 64, v62
	s_cbranch_execz .LBB_423
; %bb.422:
	v_lshlrev_b32_e32 v30, 16, v14
	v_and_b32_e32 v17, 15, v45
	v_lshrrev_b32_e32 v18, 2, v45
	v_and_b32_e32 v14, 0xffff0000, v14
	v_lshrrev_b32_e32 v19, 6, v45
	v_lshrrev_b32_e32 v20, 10, v45
	v_lshlrev_b32_e32 v17, 2, v17
	v_lshrrev_b32_e32 v21, 14, v45
	v_lshrrev_b32_e32 v22, 18, v45
	v_lshrrev_b32_e32 v23, 22, v45
	v_lshrrev_b32_e32 v24, 26, v45
	ds_load_b32 v17, v17
	v_lshlrev_b32_e32 v45, 16, v15
	v_and_b32_e32 v18, 60, v18
	v_lshlrev_b32_e32 v26, 16, v13
	v_and_b32_e32 v15, 0xffff0000, v15
	v_and_b32_e32 v13, 0xffff0000, v13
	v_and_b32_e32 v19, 60, v19
	ds_load_b32 v18, v18
	v_and_b32_e32 v25, 15, v46
	v_lshrrev_b32_e32 v27, 2, v46
	v_lshrrev_b32_e32 v28, 6, v46
	ds_load_b32 v19, v19
	v_lshrrev_b32_e32 v29, 10, v46
	v_lshrrev_b32_e32 v31, 18, v46
	v_lshrrev_b32_e32 v32, 22, v46
	v_lshrrev_b32_e32 v47, 18, v43
	v_lshrrev_b32_e32 v48, 22, v43
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_and_b32_e32 v47, 60, v47
	s_wait_dscnt 0x2
	v_fma_f32 v17, v17, v26, 0
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v17, v18, v13 :: v_dual_and_b32 v20, 60, v20
	v_lshrrev_b32_e32 v18, 14, v46
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v17, v19, v30
	v_lshrrev_b32_e32 v19, 26, v46
	v_lshlrev_b32_e32 v46, 16, v16
	v_and_b32_e32 v21, 60, v21
	v_and_b32_e32 v16, 0xffff0000, v16
	v_and_b32_e32 v22, 60, v22
	v_and_b32_e32 v23, 60, v23
	v_and_b32_e32 v24, 60, v24
	ds_load_b32 v20, v20
	ds_load_b32 v21, v21
	ds_load_b32 v22, v22
	ds_load_b32 v23, v23
	ds_load_b32 v24, v24
	v_and_b32_e32 v18, 60, v18
	v_and_b32_e32 v19, 60, v19
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v17, v20, v14
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v17, v21, v45
	v_and_b32_e32 v21, 60, v31
	v_lshrrev_b32_e32 v31, 6, v43
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v17, v22, v15 :: v_dual_lshlrev_b32 v20, 2, v25
	v_and_b32_e32 v25, 60, v27
	v_and_b32_e32 v27, 60, v28
	v_and_b32_e32 v28, 60, v29
	v_and_b32_e32 v29, 60, v32
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v17, v23, v46
	ds_load_b32 v20, v20
	ds_load_b32 v22, v25
	ds_load_b32 v25, v27
	ds_load_b32 v27, v28
	ds_load_b32 v18, v18
	ds_load_b32 v21, v21
	ds_load_b32 v28, v29
	ds_load_b32 v19, v19
	v_and_b32_e32 v29, 15, v43
	v_lshrrev_b32_e32 v23, 2, v43
	s_wait_dscnt 0x8
	v_dual_fmac_f32 v17, v24, v16 :: v_dual_lshlrev_b32 v32, 16, v9
	v_and_b32_e32 v24, 60, v31
	v_lshlrev_b32_e32 v29, 2, v29
	v_and_b32_e32 v23, 60, v23
	v_lshrrev_b32_e32 v31, 14, v43
	v_and_b32_e32 v48, 60, v48
	v_and_b32_e32 v9, 0xffff0000, v9
	ds_load_b32 v29, v29
	s_wait_dscnt 0x8
	v_fmac_f32_e32 v17, v20, v32
	ds_load_b32 v20, v23
	ds_load_b32 v23, v24
	v_lshrrev_b32_e32 v24, 10, v43
	v_lshrrev_b32_e32 v43, 26, v43
	v_and_b32_e32 v31, 60, v31
	s_wait_dscnt 0x9
	v_fmac_f32_e32 v17, v22, v9
	v_lshrrev_b32_e32 v22, 2, v44
	v_and_b32_e32 v24, 60, v24
	v_and_b32_e32 v43, 60, v43
	ds_load_b32 v24, v24
	ds_load_b32 v31, v31
	ds_load_b32 v47, v47
	ds_load_b32 v48, v48
	ds_load_b32 v43, v43
	s_wait_dscnt 0x7
	v_fma_f32 v26, v29, v26, 0
	v_lshlrev_b32_e32 v29, 16, v10
	v_and_b32_e32 v10, 0xffff0000, v10
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v26, v20, v13
	v_dual_fmac_f32 v17, v25, v29 :: v_dual_and_b32 v20, 15, v44
	v_lshlrev_b32_e32 v13, 16, v11
	v_lshrrev_b32_e32 v25, 22, v44
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v26, v23, v30
	v_lshrrev_b32_e32 v23, 6, v44
	v_fmac_f32_e32 v17, v27, v10
	v_lshrrev_b32_e32 v27, 26, v44
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v26, v24, v14 :: v_dual_and_b32 v11, 0xffff0000, v11
	v_lshlrev_b32_e32 v14, 2, v20
	v_and_b32_e32 v20, 60, v22
	v_and_b32_e32 v22, 60, v23
	ds_load_b32 v14, v14
	ds_load_b32 v20, v20
	ds_load_b32 v22, v22
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v26, v31, v45
	v_lshrrev_b32_e32 v23, 10, v44
	v_lshrrev_b32_e32 v24, 18, v44
	v_fmac_f32_e32 v17, v18, v13
	s_wait_dscnt 0x5
	v_dual_fmac_f32 v26, v47, v15 :: v_dual_and_b32 v25, 60, v25
	v_lshrrev_b32_e32 v15, 14, v44
	v_and_b32_e32 v23, 60, v23
	v_and_b32_e32 v24, 60, v24
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v26, v48, v46 :: v_dual_and_b32 v27, 60, v27
	v_and_b32_e32 v15, 60, v15
	v_dual_fmac_f32 v17, v21, v11 :: v_dual_and_b32 v18, 7, v60
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v26, v43, v16
	ds_load_b32 v16, v23
	ds_load_b32 v15, v15
	ds_load_b32 v23, v24
	ds_load_b32 v24, v25
	ds_load_b32 v25, v27
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v26, v14, v32
	v_lshlrev_b32_e32 v14, 16, v12
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v26, v20, v9 :: v_dual_and_b32 v9, 0xffff0000, v12
	v_and_b32_e32 v12, 7, v59
	v_dual_fmac_f32 v17, v28, v14 :: v_dual_and_b32 v20, 0x7f, v59
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v26, v22, v29
	v_cvt_f32_ubyte0_e32 v12, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cmp_lt_u32_e32 vcc_lo, 7, v20
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v17, v19, v9 :: v_dual_fmac_f32 v26, v16, v10
	v_and_b32_e32 v10, 0x7f, v60
	v_cvt_f32_ubyte0_e32 v16, v18
	v_lshl_add_u32 v18, v20, 20, 0x3c000000
	v_mul_f32_e32 v12, 0x3b000000, v12
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v26, v15, v13
	v_lshl_add_u32 v13, v10, 20, 0x3c000000
	s_wait_alu 0xfffd
	v_dual_mul_f32 v15, 0x3b000000, v16 :: v_dual_cndmask_b32 v12, v12, v18
	v_cmp_lt_u32_e32 vcc_lo, 7, v10
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v26, v23, v11
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v11, v15, v13, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v20
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v26, v24, v14 :: v_dual_lshlrev_b32 v13, 24, v59
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v12, 0x7fc00000, v12, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v10
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v26, v25, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_and_or_b32 v12, 0x80000000, v13, v12
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v10, 0x7fc00000, v11 :: v_dual_lshlrev_b32 v11, 24, v60
	v_fmac_f32_e32 v56, v17, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v9, 0x80000000, v11, v10
	v_fmac_f32_e32 v55, v26, v9
.LBB_423:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_lt_i32_e32 0x60, v62
	s_cbranch_execz .LBB_425
; %bb.424:
	v_lshlrev_b32_e32 v25, 16, v7
	v_and_b32_e32 v7, 0xffff0000, v7
	v_lshlrev_b32_e32 v22, 16, v6
	v_and_b32_e32 v9, 15, v41
	v_lshrrev_b32_e32 v10, 2, v41
	v_and_b32_e32 v6, 0xffff0000, v6
	v_lshrrev_b32_e32 v11, 6, v41
	v_lshlrev_b32_e32 v18, 16, v5
	v_lshlrev_b32_e32 v9, 2, v9
	v_and_b32_e32 v10, 60, v10
	v_and_b32_e32 v5, 0xffff0000, v5
	v_and_b32_e32 v11, 60, v11
	v_lshrrev_b32_e32 v12, 10, v41
	ds_load_b32 v9, v9
	ds_load_b32 v10, v10
	v_lshrrev_b32_e32 v13, 14, v41
	ds_load_b32 v11, v11
	v_lshrrev_b32_e32 v14, 18, v41
	v_lshrrev_b32_e32 v15, 22, v41
	v_lshrrev_b32_e32 v16, 26, v41
	v_lshlrev_b32_e32 v26, 16, v8
	v_and_b32_e32 v13, 60, v13
	v_and_b32_e32 v8, 0xffff0000, v8
	v_and_b32_e32 v14, 60, v14
	v_and_b32_e32 v15, 60, v15
	v_and_b32_e32 v17, 15, v42
	v_lshrrev_b32_e32 v19, 2, v42
	v_lshrrev_b32_e32 v20, 6, v42
	v_lshrrev_b32_e32 v21, 10, v42
	v_lshrrev_b32_e32 v23, 18, v42
	v_lshrrev_b32_e32 v24, 22, v42
	v_lshrrev_b32_e32 v27, 18, v39
	v_lshrrev_b32_e32 v28, 22, v39
	s_wait_dscnt 0x2
	v_fma_f32 v9, v9, v18, 0
	v_and_b32_e32 v12, 60, v12
	v_lshrrev_b32_e32 v29, 26, v39
	v_and_b32_e32 v27, 60, v27
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v9, v10, v5
	v_lshrrev_b32_e32 v10, 14, v42
	v_and_b32_e32 v29, 60, v29
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v9, v11, v22 :: v_dual_and_b32 v16, 60, v16
	ds_load_b32 v12, v12
	ds_load_b32 v13, v13
	ds_load_b32 v14, v14
	ds_load_b32 v15, v15
	ds_load_b32 v16, v16
	v_lshrrev_b32_e32 v11, 26, v42
	v_and_b32_e32 v10, 60, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_and_b32_e32 v11, 60, v11
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v9, v12, v6
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v9, v13, v25
	v_and_b32_e32 v13, 60, v23
	v_lshrrev_b32_e32 v23, 6, v39
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v9, v14, v7 :: v_dual_lshlrev_b32 v12, 2, v17
	v_and_b32_e32 v17, 60, v19
	v_and_b32_e32 v19, 60, v20
	v_and_b32_e32 v20, 60, v21
	v_and_b32_e32 v21, 60, v24
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v9, v15, v26
	ds_load_b32 v12, v12
	ds_load_b32 v14, v17
	ds_load_b32 v17, v19
	ds_load_b32 v19, v20
	ds_load_b32 v10, v10
	ds_load_b32 v13, v13
	ds_load_b32 v20, v21
	ds_load_b32 v11, v11
	v_and_b32_e32 v21, 15, v39
	v_lshrrev_b32_e32 v15, 2, v39
	s_wait_dscnt 0x8
	v_dual_fmac_f32 v9, v16, v8 :: v_dual_lshlrev_b32 v24, 16, v1
	v_and_b32_e32 v16, 60, v23
	v_lshlrev_b32_e32 v21, 2, v21
	v_and_b32_e32 v15, 60, v15
	v_lshrrev_b32_e32 v23, 14, v39
	v_and_b32_e32 v28, 60, v28
	v_and_b32_e32 v1, 0xffff0000, v1
	ds_load_b32 v21, v21
	s_wait_dscnt 0x8
	v_fmac_f32_e32 v9, v12, v24
	ds_load_b32 v12, v15
	ds_load_b32 v15, v16
	v_lshrrev_b32_e32 v16, 10, v39
	v_and_b32_e32 v23, 60, v23
	s_wait_dscnt 0x9
	v_fmac_f32_e32 v9, v14, v1
	v_lshrrev_b32_e32 v14, 2, v40
	v_and_b32_e32 v16, 60, v16
	ds_load_b32 v16, v16
	ds_load_b32 v23, v23
	ds_load_b32 v27, v27
	ds_load_b32 v28, v28
	ds_load_b32 v29, v29
	s_wait_dscnt 0x7
	v_fma_f32 v18, v21, v18, 0
	v_lshlrev_b32_e32 v21, 16, v2
	v_and_b32_e32 v2, 0xffff0000, v2
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v18, v12, v5
	v_dual_fmac_f32 v9, v17, v21 :: v_dual_and_b32 v12, 15, v40
	v_lshlrev_b32_e32 v5, 16, v3
	v_lshrrev_b32_e32 v17, 22, v40
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v18, v15, v22
	v_lshrrev_b32_e32 v15, 6, v40
	v_fmac_f32_e32 v9, v19, v2
	v_lshrrev_b32_e32 v19, 26, v40
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v18, v16, v6 :: v_dual_and_b32 v3, 0xffff0000, v3
	v_lshlrev_b32_e32 v6, 2, v12
	v_and_b32_e32 v12, 60, v14
	v_and_b32_e32 v14, 60, v15
	ds_load_b32 v6, v6
	ds_load_b32 v12, v12
	ds_load_b32 v14, v14
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v18, v23, v25
	v_lshrrev_b32_e32 v15, 10, v40
	v_lshrrev_b32_e32 v16, 18, v40
	v_fmac_f32_e32 v9, v10, v5
	s_wait_dscnt 0x5
	v_dual_fmac_f32 v18, v27, v7 :: v_dual_and_b32 v17, 60, v17
	v_lshrrev_b32_e32 v7, 14, v40
	v_and_b32_e32 v15, 60, v15
	v_and_b32_e32 v16, 60, v16
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v18, v28, v26 :: v_dual_and_b32 v19, 60, v19
	v_and_b32_e32 v7, 60, v7
	v_dual_fmac_f32 v9, v13, v3 :: v_dual_and_b32 v10, 7, v58
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v18, v29, v8
	ds_load_b32 v8, v15
	ds_load_b32 v7, v7
	ds_load_b32 v15, v16
	ds_load_b32 v16, v17
	ds_load_b32 v17, v19
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v18, v6, v24
	v_lshlrev_b32_e32 v6, 16, v4
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v18, v12, v1 :: v_dual_and_b32 v1, 0xffff0000, v4
	v_and_b32_e32 v4, 7, v57
	v_dual_fmac_f32 v9, v20, v6 :: v_dual_and_b32 v12, 0x7f, v57
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v18, v14, v21
	v_cvt_f32_ubyte0_e32 v4, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cmp_lt_u32_e32 vcc_lo, 7, v12
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v9, v11, v1 :: v_dual_fmac_f32 v18, v8, v2
	v_and_b32_e32 v2, 0x7f, v58
	v_cvt_f32_ubyte0_e32 v8, v10
	v_lshl_add_u32 v10, v12, 20, 0x3c000000
	v_mul_f32_e32 v4, 0x3b000000, v4
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v18, v7, v5
	v_lshl_add_u32 v5, v2, 20, 0x3c000000
	s_wait_alu 0xfffd
	v_dual_mul_f32 v7, 0x3b000000, v8 :: v_dual_cndmask_b32 v4, v4, v10
	v_cmp_lt_u32_e32 vcc_lo, 7, v2
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v18, v15, v3
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v7, v5, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v12
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v18, v16, v6 :: v_dual_lshlrev_b32 v5, 24, v57
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v4, 0x7fc00000, v4, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v2
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v18, v17, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_and_or_b32 v4, 0x80000000, v5, v4
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v2, 0x7fc00000, v3 :: v_dual_lshlrev_b32 v3, 24, v58
	v_fmac_f32_e32 v56, v9, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v1, 0x80000000, v3, v2
	v_fmac_f32_e32 v55, v18, v1
.LBB_425:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v1, 0x80, v33
	s_mov_b32 s17, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s11, v1
	s_cbranch_execz .LBB_429
; %bb.426:
	v_and_b32_e32 v13, 31, v0
	s_mov_b64 s[8:9], 0x80
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v1, 3, v13
	v_lshlrev_b32_e32 v9, 5, v13
	v_add_co_u32 v1, vcc_lo, v37, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, 0, v38, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, 0x400, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	v_add_co_u32 v3, vcc_lo, 0x80, v35
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, 0, v36, vcc_lo
	v_add_co_u32 v5, vcc_lo, s36, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, s37, v2, vcc_lo
	v_add_co_u32 v14, vcc_lo, s38, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v15, null, s39, v4, vcc_lo
	v_add_co_u32 v7, vcc_lo, s42, v1
	v_add_co_u32 v1, s0, s6, v9
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, s43, v2, vcc_lo
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v2, null, s7, 0, s0
	v_add_co_u32 v16, vcc_lo, s44, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v17, null, s45, v4, vcc_lo
	v_add_co_u32 v9, vcc_lo, 0x101c, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v10, null, 0, v2, vcc_lo
	s_mov_b32 s6, 0
.LBB_427:                              ; =>This Inner Loop Header: Depth=1
	global_load_b64 v[11:12], v[5:6], off th:TH_LOAD_NT
	global_load_b128 v[1:4], v[9:10], off offset:-28
	global_load_b64 v[23:24], v[7:8], off th:TH_LOAD_NT
	global_load_b128 v[19:22], v[9:10], off offset:-12
	v_add_co_u32 v25, vcc_lo, v14, v13
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v26, null, 0, v15, vcc_lo
	v_add_co_u32 v27, vcc_lo, v16, v13
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v28, null, 0, v17, vcc_lo
	global_load_i8 v18, v[25:26], off
	global_load_i8 v25, v[27:28], off
	v_add_co_u32 v5, vcc_lo, 0x100, v5
	s_wait_alu 0xfffe
	s_add_nc_u64 s[8:9], s[8:9], 32
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
	v_add_co_u32 v14, vcc_lo, v14, 32
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v15, null, 0, v15, vcc_lo
	v_add_co_u32 v7, vcc_lo, 0x100, v7
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, 0, v8, vcc_lo
	v_add_co_u32 v16, vcc_lo, v16, 32
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v17, null, 0, v17, vcc_lo
	v_add_co_u32 v9, vcc_lo, 0x400, v9
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v10, null, 0, v10, vcc_lo
	s_wait_loadcnt 0x5
	v_and_b32_e32 v26, 15, v11
	v_lshrrev_b32_e32 v27, 2, v11
	s_wait_loadcnt 0x3
	v_and_b32_e32 v50, 15, v23
	v_lshrrev_b32_e32 v51, 2, v23
	v_lshrrev_b32_e32 v52, 6, v23
	v_lshlrev_b32_e32 v26, 2, v26
	v_and_b32_e32 v27, 60, v27
	v_lshlrev_b32_e32 v50, 2, v50
	v_and_b32_e32 v51, 60, v51
	v_lshrrev_b32_e32 v28, 6, v11
	v_lshrrev_b32_e32 v29, 10, v11
	ds_load_b32 v26, v26
	ds_load_b32 v50, v50
	ds_load_b32 v27, v27
	ds_load_b32 v51, v51
	v_lshrrev_b32_e32 v30, 14, v11
	v_lshrrev_b32_e32 v31, 18, v11
	v_lshrrev_b32_e32 v32, 22, v11
	v_lshrrev_b32_e32 v11, 26, v11
	v_lshrrev_b32_e32 v53, 10, v23
	v_lshrrev_b32_e32 v54, 14, v23
	v_lshrrev_b32_e32 v57, 18, v23
	v_lshrrev_b32_e32 v58, 22, v23
	v_lshrrev_b32_e32 v23, 26, v23
	v_lshlrev_b32_e32 v42, 16, v1
	v_and_b32_e32 v52, 60, v52
	v_and_b32_e32 v28, 60, v28
	v_and_b32_e32 v29, 60, v29
	v_and_b32_e32 v30, 60, v30
	v_and_b32_e32 v31, 60, v31
	v_and_b32_e32 v32, 60, v32
	v_and_b32_e32 v11, 60, v11
	v_and_b32_e32 v53, 60, v53
	v_and_b32_e32 v54, 60, v54
	v_and_b32_e32 v57, 60, v57
	v_and_b32_e32 v58, 60, v58
	v_and_b32_e32 v23, 60, v23
	ds_load_b32 v52, v52
	ds_load_b32 v28, v28
	ds_load_b32 v29, v29
	ds_load_b32 v30, v30
	ds_load_b32 v31, v31
	ds_load_b32 v32, v32
	ds_load_b32 v11, v11
	ds_load_b32 v53, v53
	ds_load_b32 v54, v54
	ds_load_b32 v57, v57
	ds_load_b32 v58, v58
	ds_load_b32 v23, v23
	s_wait_dscnt 0xf
	v_fma_f32 v26, v26, v42, 0
	s_wait_dscnt 0xe
	v_fma_f32 v42, v50, v42, 0
	v_lshlrev_b32_e32 v43, 16, v2
	v_and_b32_e32 v1, 0xffff0000, v1
	v_and_b32_e32 v35, 15, v12
	v_and_b32_e32 v2, 0xffff0000, v2
	v_lshlrev_b32_e32 v44, 16, v3
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v46, 16, v19
	s_wait_dscnt 0xd
	v_fmac_f32_e32 v26, v27, v1
	v_lshlrev_b32_e32 v48, 16, v21
	v_lshrrev_b32_e32 v36, 2, v12
	v_lshrrev_b32_e32 v37, 6, v12
	v_lshrrev_b32_e32 v38, 10, v12
	s_wait_dscnt 0xa
	v_fmac_f32_e32 v26, v28, v43
	v_dual_fmac_f32 v42, v51, v1 :: v_dual_lshlrev_b32 v45, 16, v4
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v4, 0xffff0000, v4
	s_wait_dscnt 0x9
	v_fmac_f32_e32 v26, v29, v2
	v_fmac_f32_e32 v42, v52, v43
	v_and_b32_e32 v19, 0xffff0000, v19
	v_and_b32_e32 v21, 0xffff0000, v21
	v_and_b32_e32 v36, 60, v36
	s_wait_dscnt 0x8
	v_fmac_f32_e32 v26, v30, v44
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v42, v53, v2
	v_lshlrev_b32_e32 v49, 16, v22
	v_lshlrev_b32_e32 v47, 16, v20
	v_lshrrev_b32_e32 v39, 14, v12
	v_fmac_f32_e32 v26, v31, v3
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v42, v54, v44
	v_and_b32_e32 v59, 15, v24
	v_lshrrev_b32_e32 v60, 2, v24
	v_and_b32_e32 v38, 60, v38
	v_fmac_f32_e32 v26, v32, v45
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v42, v57, v3
	v_lshlrev_b32_e32 v50, 2, v59
	v_and_b32_e32 v39, 60, v39
	v_and_b32_e32 v59, 60, v60
	v_fmac_f32_e32 v26, v11, v4
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v42, v58, v45
	v_lshrrev_b32_e32 v40, 18, v12
	v_lshrrev_b32_e32 v41, 22, v12
	v_lshrrev_b32_e32 v62, 6, v24
	v_and_b32_e32 v20, 0xffff0000, v20
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v42, v23, v4 :: v_dual_lshlrev_b32 v35, 2, v35
	v_and_b32_e32 v40, 60, v40
	v_and_b32_e32 v60, 60, v62
	v_lshrrev_b32_e32 v12, 26, v12
	v_lshrrev_b32_e32 v63, 10, v24
	ds_load_b32 v35, v35
	v_and_b32_e32 v37, 60, v37
	v_lshrrev_b32_e32 v64, 14, v24
	v_and_b32_e32 v12, 60, v12
	v_lshrrev_b32_e32 v65, 18, v24
	v_lshrrev_b32_e32 v66, 22, v24
	v_lshrrev_b32_e32 v24, 26, v24
	s_wait_loadcnt 0x0
	v_and_b32_e32 v32, 0x7f, v25
	v_and_b32_e32 v22, 0xffff0000, v22
	v_and_b32_e32 v28, 60, v66
	v_and_b32_e32 v24, 60, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_lt_u32_e64 s1, 7, v32
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v26, v35, v46
	ds_load_b32 v2, v36
	ds_load_b32 v29, v37
	ds_load_b32 v36, v50
	ds_load_b32 v30, v38
	ds_load_b32 v37, v39
	ds_load_b32 v38, v59
	s_wait_dscnt 0x5
	v_dual_fmac_f32 v26, v2, v19 :: v_dual_and_b32 v41, 60, v41
	ds_load_b32 v3, v40
	ds_load_b32 v31, v41
	ds_load_b32 v39, v60
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v42, v36, v46
	v_and_b32_e32 v2, 7, v25
	v_fmac_f32_e32 v26, v29, v47
	v_lshl_add_u32 v29, v32, 20, 0x3c000000
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v42, v38, v19
	v_cvt_f32_ubyte0_e32 v2, v2
	v_dual_fmac_f32 v26, v30, v20 :: v_dual_and_b32 v1, 60, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v26, v37, v48
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v26, v3, v21 :: v_dual_and_b32 v27, 60, v64
	ds_load_b32 v12, v12
	ds_load_b32 v1, v1
	ds_load_b32 v27, v27
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v42, v39, v47
	v_fmac_f32_e32 v26, v31, v49
	s_wait_dscnt 0x2
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v26, v12, v22 :: v_dual_add_nc_u32 v3, s8, v13
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v42, v1, v20 :: v_dual_and_b32 v51, 60, v65
	s_wait_dscnt 0x0
	v_dual_mul_f32 v1, 0x3b000000, v2 :: v_dual_fmac_f32 v42, v27, v48
	ds_load_b32 v4, v51
	ds_load_b32 v11, v28
	ds_load_b32 v23, v24
	v_and_b32_e32 v28, 7, v18
	v_and_b32_e32 v24, 0x7f, v18
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v1, v1, v29, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v28, v28
	v_lshl_add_u32 v19, v24, 20, 0x3c000000
	v_cmp_lt_u32_e64 s0, 7, v24
	v_cmp_ne_u32_e64 s1, 0x7f, v24
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v28, 0x3b000000, v28
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v2, v28, v19, s0
	v_cmp_ne_u32_e64 s0, 0x7f, v32
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v42, v4, v21
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v2, 0x7fc00000, v2, s1
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v1, 0x7fc00000, v1, s0
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v42, v11, v49
	v_cmp_le_i32_e64 s0, s11, v3
	v_and_or_b32 v2, 0x80000000, v18, v2
	v_and_or_b32 v1, 0x80000000, v25, v1
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v42, v23, v22
	s_or_b32 s6, s0, s6
	v_fmac_f32_e32 v56, v26, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v55, v42, v1
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB_427
; %bb.428:
	s_or_b32 exec_lo, exec_lo, s6
.LBB_429:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	s_or_b32 exec_lo, exec_lo, s17
	v_mbcnt_lo_u32_b32 v6, -1, 0
	v_xor_b32_e32 v2, 16, v6
	v_xor_b32_e32 v4, 8, v6
	v_xor_b32_e32 v7, 4, v6
	v_xor_b32_e32 v8, 2, v6
	v_xor_b32_e32 v9, 1, v6
	v_cmp_gt_i32_e32 vcc_lo, 32, v2
	s_wait_alu 0xfffd
	v_dual_mov_b32 v1, 0 :: v_dual_cndmask_b32 v2, v6, v2
	global_load_b32 v1, v1, s[40:41]
	v_cmp_gt_i32_e32 vcc_lo, 32, v4
	v_lshlrev_b32_e32 v2, 2, v2
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v4, v6, v4, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v7
	ds_bpermute_b32 v3, v2, v56
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v7, v6, v7, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v8
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v8, v6, v8, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v9
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v6, v6, v9 :: v_dual_add_f32 v5, v56, v3
	v_lshlrev_b32_e32 v3, 2, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v6, 2, v6
	ds_bpermute_b32 v4, v3, v5
	s_wait_dscnt 0x0
	v_dual_add_f32 v5, v5, v4 :: v_dual_lshlrev_b32 v4, 2, v7
	ds_bpermute_b32 v7, v4, v5
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v5, v7
	v_lshlrev_b32_e32 v5, 2, v8
	ds_bpermute_b32 v8, v5, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	ds_bpermute_b32 v8, v6, v7
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v7, v1, v7
	v_and_b32_e32 v1, 0x7fffffff, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.430:
	v_bfe_u32 v1, v7, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v1, v7, v1, 0x7fff
                                        ; implicit-def: $vgpr7
	v_lshrrev_b32_e32 v1, 16, v1
; %bb.431:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.432:
	v_lshrrev_b32_e32 v1, 16, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v1, 64, v1
; %bb.433:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	ds_bpermute_b32 v2, v2, v55
	s_wait_dscnt 0x0
	v_dual_mov_b32 v7, 0 :: v_dual_add_f32 v2, v55, v2
	global_load_b32 v7, v7, s[46:47]
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
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v2, v3
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v3, v7, v2
	v_and_b32_e32 v2, 0x7fffffff, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
	s_cbranch_execz .LBB_436
; %bb.434:
	v_bfe_u32 v2, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v2, v3, v2, 0x7fff
                                        ; implicit-def: $vgpr3
	v_lshrrev_b32_e32 v2, 16, v2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
	s_cbranch_execnz .LBB_437
.LBB_435:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_eq_u32_e32 vcc_lo, 0, v33
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execnz .LBB_438
	s_branch .LBB_443
.LBB_436:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
	s_cbranch_execz .LBB_435
.LBB_437:
	v_lshrrev_b32_e32 v2, 16, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, 64, v2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_eq_u32_e32 vcc_lo, 0, v33
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB_443
.LBB_438:
	v_lshlrev_b32_e32 v1, 16, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v3, 0xbfb8aa3b, v1 :: v_dual_lshlrev_b32 v2, 16, v2
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v1
	v_fma_f32 v4, 0xbfb8aa3b, v1, -v3
	v_rndne_f32_e32 v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v4, v1, 0xb2a5705f, v4 :: v_dual_sub_f32 v3, v3, v5
	v_add_f32_e32 v3, v3, v4
	v_cvt_i32_f32_e32 v4, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v3, v3
	v_ldexp_f32 v3, v3, v4
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, 0x7f800000, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v3, 1.0, v3
	v_div_scale_f32 v4, null, v3, v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v5, v4
	v_fma_f32 v6, -v4, v5, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v5, v6, v5
	v_div_scale_f32 v6, vcc_lo, v1, v3, v1
	v_mul_f32_e32 v7, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v8, -v4, v7, v6
	v_fmac_f32_e32 v7, v8, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v4, -v4, v7, v6
	s_wait_alu 0xfffd
	v_div_fmas_f32 v4, v4, v5, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v1, v4, v3, v1
	v_mul_f32_e32 v2, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v1, 0x7fffffff, v2
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.439:
	v_bfe_u32 v1, v2, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v1, v2, v1, 0x7fff
                                        ; implicit-def: $vgpr2
	v_lshrrev_b32_e32 v1, 16, v1
; %bb.440:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.441:
	v_lshrrev_b32_e32 v1, 16, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v1, 64, v1
; %bb.442:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_ashrrev_i32_e32 v35, 31, v34
	s_ashr_i32 s1, s55, 31
	s_mov_b32 s0, s55
	s_wait_alu 0xfffe
	s_mul_u64 s[0:1], s[4:5], s[0:1]
	v_lshlrev_b64_e32 v[2:3], 1, v[34:35]
	s_wait_alu 0xfffe
	s_lshl_b64 s[0:1], s[0:1], 1
	s_wait_alu 0xfffe
	s_add_nc_u64 s[0:1], s[64:65], s[0:1]
	s_wait_alu 0xfffe
	v_add_co_u32 v2, vcc_lo, s0, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s1, v3, vcc_lo
	global_store_b16 v[2:3], v1, off
.LBB_443:
	s_or_b32 exec_lo, exec_lo, s16
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	global_inv scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB_446
; %bb.444:
	s_mov_b32 s1, exec_lo
	s_wait_alu 0xfffe
	v_mbcnt_lo_u32_b32 v1, s1, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_and_b32 s4, exec_lo, vcc_lo
	s_wait_alu 0xfffe
	s_mov_b32 exec_lo, s4
	s_cbranch_execz .LBB_446
; %bb.445:
	s_bcnt1_i32_b32 s1, s1
	s_wait_alu 0xfffe
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, s1
	s_wait_loadcnt 0x0
	global_atomic_add_u32 v1, v2, s[34:35] scope:SCOPE_DEV
.LBB_446:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s1, 0
.LBB_447:
	s_wait_alu 0xfffe
	s_and_b32 vcc_lo, exec_lo, s1
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB_525
; %bb.448:
	s_abs_i32 s0, s3
	s_abs_i32 s5, s10
	s_wait_alu 0xfffe
	s_cvt_f32_u32 s1, s0
	s_sub_co_i32 s4, 0, s0
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s1
	v_readfirstlane_b32 s1, v1
	s_mul_f32 s1, s1, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s1, s1
	s_wait_alu 0xfffe
	s_mul_i32 s4, s4, s1
	s_wait_alu 0xfffe
	s_mul_hi_u32 s4, s1, s4
	s_wait_alu 0xfffe
	s_add_co_i32 s1, s1, s4
	s_xor_b32 s4, s10, s3
	s_wait_alu 0xfffe
	s_mul_hi_u32 s1, s5, s1
	s_ashr_i32 s4, s4, 31
	s_wait_alu 0xfffe
	s_mul_i32 s6, s1, s0
	s_wait_alu 0xfffe
	s_sub_co_i32 s5, s5, s6
	s_add_co_i32 s6, s1, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s7, s5, s0
	s_cmp_ge_u32 s5, s0
	s_cselect_b32 s1, s6, s1
	s_wait_alu 0xfffe
	s_cselect_b32 s5, s7, s5
	s_add_co_i32 s6, s1, 1
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s5, s0
	s_cselect_b32 s0, s6, s1
	s_abs_i32 s1, s56
	s_wait_alu 0xfffe
	s_xor_b32 s0, s0, s4
	s_cvt_f32_u32 s5, s1
	s_sub_co_i32 s6, 0, s1
	s_wait_alu 0xfffe
	s_sub_co_i32 s4, s0, s4
	v_rcp_iflag_f32_e32 v1, s5
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_readfirstlane_b32 s5, v1
	v_mov_b32_e32 v1, 0
	s_mul_f32 s5, s5, 0x4f7ffffe
	v_mov_b32_e32 v2, v1
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s5, s5
	s_wait_alu 0xfffe
	s_mul_i32 s6, s6, s5
	s_wait_alu 0xfffe
	s_mul_hi_u32 s0, s5, s6
	s_abs_i32 s6, s4
	s_wait_alu 0xfffe
	s_add_co_i32 s5, s5, s0
	s_wait_alu 0xfffe
	s_mul_hi_u32 s0, s6, s5
	s_xor_b32 s5, s4, s56
	s_wait_alu 0xfffe
	s_mul_i32 s7, s0, s1
	s_ashr_i32 s5, s5, 31
	s_wait_alu 0xfffe
	s_sub_co_i32 s6, s6, s7
	s_add_co_i32 s7, s0, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s8, s6, s1
	s_cmp_ge_u32 s6, s1
	s_cselect_b32 s0, s7, s0
	s_wait_alu 0xfffe
	s_cselect_b32 s6, s8, s6
	s_add_co_i32 s7, s0, 1
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s6, s1
	s_cselect_b32 s0, s7, s0
	s_ashr_i32 s1, s52, 31
	s_wait_alu 0xfffe
	s_xor_b32 s0, s0, s5
	s_wait_alu 0xfffe
	s_sub_co_i32 s6, s0, s5
	s_mov_b32 s0, s52
	s_wait_alu 0xfffe
	s_ashr_i32 s7, s6, 31
	s_wait_alu 0xfffe
	s_mul_u64 s[0:1], s[6:7], s[0:1]
	s_wait_alu 0xfffe
	s_lshl_b64 s[0:1], s[0:1], 1
	s_wait_kmcnt 0x0
	s_wait_alu 0xfffe
	s_add_nc_u64 s[8:9], s[14:15], s[0:1]
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_i32_e64 s52, v33
	s_cbranch_execz .LBB_450
; %bb.449:
	v_lshlrev_b32_e32 v2, 1, v33
	global_load_u16 v2, v2, s[8:9]
	s_wait_loadcnt 0x0
	v_cmp_gt_i16_e32 vcc_lo, 0, v2
	v_and_b32_e32 v3, 0xffff, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_not_b32_e32 v4, v3
	v_or_b32_e32 v5, 0x8000, v3
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v2, v5, v4, vcc_lo
	v_and_b32_e32 v6, 0x7f80, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v2, 16, v2
	v_and_b32_e32 v3, 0x7f, v3
	v_cmp_ne_u32_e32 vcc_lo, 0x7f80, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_eq_u32_e64 s0, 0, v3
	s_or_b32 vcc_lo, vcc_lo, s0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	v_or_b32_e32 v2, v33, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v2, 0xffff, v2
.LBB_450:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v6, 32, v33
	v_mov_b32_e32 v3, 0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s52, v6
	s_cbranch_execz .LBB_452
; %bb.451:
	v_lshlrev_b32_e32 v3, 1, v33
	global_load_u16 v3, v3, s[8:9] offset:64
	s_wait_loadcnt 0x0
	v_cmp_gt_i16_e32 vcc_lo, 0, v3
	v_and_b32_e32 v4, 0xffff, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_not_b32_e32 v5, v4
	v_or_b32_e32 v7, 0x8000, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v7, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v3, 16, v3
	v_and_b32_e32 v8, 0x7f80, v4
	v_and_b32_e32 v4, 0x7f, v4
	v_cmp_ne_u32_e32 vcc_lo, 0x7f80, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_eq_u32_e64 s0, 0, v4
	s_or_b32 vcc_lo, vcc_lo, s0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	v_or_b32_e32 v3, v33, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v3, 0xffdf, v3
.LBB_452:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v5, 64, v33
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s52, v5
	s_cbranch_execz .LBB_454
; %bb.453:
	v_lshlrev_b32_e32 v1, 1, v33
	global_load_u16 v1, v1, s[8:9] offset:128
	s_wait_loadcnt 0x0
	v_cmp_gt_i16_e32 vcc_lo, 0, v1
	v_and_b32_e32 v4, 0xffff, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_not_b32_e32 v7, v4
	v_or_b32_e32 v8, 0x8000, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v1, v8, v7, vcc_lo
	v_and_b32_e32 v9, 0x7f80, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v1, 16, v1
	v_and_b32_e32 v4, 0x7f, v4
	v_cmp_ne_u32_e32 vcc_lo, 0x7f80, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_eq_u32_e64 s0, 0, v4
	s_or_b32 vcc_lo, vcc_lo, s0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	v_or_b32_e32 v1, v33, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v1, 0xffbf, v1
.LBB_454:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v4, 0x60, v33
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v8, 0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s52, v4
	s_cbranch_execz .LBB_456
; %bb.455:
	v_lshlrev_b32_e32 v8, 1, v33
	global_load_u16 v8, v8, s[8:9] offset:192
	s_wait_loadcnt 0x0
	v_cmp_gt_i16_e32 vcc_lo, 0, v8
	v_and_b32_e32 v9, 0xffff, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_not_b32_e32 v10, v9
	v_or_b32_e32 v11, 0x8000, v9
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v8, v11, v10, vcc_lo
	v_and_b32_e32 v12, 0x7f80, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v8, 16, v8
	v_and_b32_e32 v9, 0x7f, v9
	v_cmp_ne_u32_e32 vcc_lo, 0x7f80, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_eq_u32_e64 s0, 0, v9
	s_or_b32 vcc_lo, vcc_lo, s0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo
	v_or_b32_e32 v8, v33, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v8, 0xff9f, v8
.LBB_456:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v60, 0x80, v33
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s52, v60
	s_cbranch_execz .LBB_458
; %bb.457:
	v_lshlrev_b32_e32 v7, 1, v33
	global_load_u16 v7, v7, s[8:9] offset:256
	s_wait_loadcnt 0x0
	v_and_b32_e32 v9, 0xffff, v7
	v_cmp_gt_i16_e32 vcc_lo, 0, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_not_b32_e32 v10, v9
	v_or_b32_e32 v11, 0x8000, v9
	v_and_b32_e32 v12, 0x7f80, v9
	v_and_b32_e32 v9, 0x7f, v9
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v7, v11, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f80, v12
	v_cmp_eq_u32_e64 s0, 0, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v7, 16, v7
	s_or_b32 vcc_lo, vcc_lo, s0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v7, v33, v7
	v_xor_b32_e32 v7, 0xff7f, v7
.LBB_458:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v10, 0xa0, v33
	v_mov_b32_e32 v9, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s52, v10
	v_mov_b32_e32 v10, 0
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB_460
; %bb.459:
	v_lshlrev_b32_e32 v10, 1, v33
	global_load_u16 v10, v10, s[8:9] offset:320
	s_wait_loadcnt 0x0
	v_cmp_gt_i16_e32 vcc_lo, 0, v10
	v_and_b32_e32 v11, 0xffff, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_not_b32_e32 v12, v11
	v_or_b32_e32 v13, 0x8000, v11
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v10, v13, v12, vcc_lo
	v_and_b32_e32 v14, 0x7f80, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v10, 16, v10
	v_and_b32_e32 v11, 0x7f, v11
	v_cmp_ne_u32_e32 vcc_lo, 0x7f80, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_eq_u32_e64 s0, 0, v11
	s_or_b32 vcc_lo, vcc_lo, s0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v10, 0, v10, vcc_lo
	v_or_b32_e32 v10, v33, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v10, 0xff5f, v10
.LBB_460:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v11, 0xc0, v33
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s52, v11
	s_cbranch_execz .LBB_462
; %bb.461:
	v_lshlrev_b32_e32 v9, 1, v33
	global_load_u16 v9, v9, s[8:9] offset:384
	s_wait_loadcnt 0x0
	v_and_b32_e32 v11, 0xffff, v9
	v_cmp_gt_i16_e32 vcc_lo, 0, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_not_b32_e32 v12, v11
	v_or_b32_e32 v13, 0x8000, v11
	v_and_b32_e32 v14, 0x7f80, v11
	v_and_b32_e32 v11, 0x7f, v11
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v9, v13, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f80, v14
	v_cmp_eq_u32_e64 s0, 0, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v9, 16, v9
	s_or_b32 vcc_lo, vcc_lo, s0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v9, 0, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v9, v33, v9
	v_xor_b32_e32 v9, 0xff3f, v9
.LBB_462:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v11, 0xe0, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s52, v11
	v_mov_b32_e32 v11, 0
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB_464
; %bb.463:
	v_lshlrev_b32_e32 v11, 1, v33
	global_load_u16 v11, v11, s[8:9] offset:448
	s_wait_loadcnt 0x0
	v_cmp_gt_i16_e32 vcc_lo, 0, v11
	v_and_b32_e32 v12, 0xffff, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_not_b32_e32 v13, v12
	v_or_b32_e32 v14, 0x8000, v12
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v11, v14, v13, vcc_lo
	v_and_b32_e32 v15, 0x7f80, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v11, 16, v11
	v_and_b32_e32 v12, 0x7f, v12
	v_cmp_ne_u32_e32 vcc_lo, 0x7f80, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_eq_u32_e64 s0, 0, v12
	s_or_b32 vcc_lo, vcc_lo, s0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v11, 0, v11, vcc_lo
	v_or_b32_e32 v11, v33, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v11, 0xff1f, v11
.LBB_464:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_mul_i32 s0, s6, s56
	v_mbcnt_lo_u32_b32 v59, -1, 0
	s_wait_alu 0xfffe
	s_sub_co_i32 s0, s4, s0
	s_wait_alu 0xfffe
	s_cmp_lt_i32 s0, 0
	s_cbranch_scc1 .LBB_526
; %bb.465:
	v_xor_b32_e32 v13, 16, v59
	v_max_u32_e32 v12, v2, v3
	v_xor_b32_e32 v14, 8, v59
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	v_max3_u32 v12, v12, v1, v8
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v59, v13, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_max3_u32 v12, v12, v7, v10
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_max3_u32 v12, v12, v9, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v14, 2, v14
	v_lshlrev_b32_e32 v13, 2, v13
	ds_bpermute_b32 v13, v13, v12
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 4, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_lshlrev_b32_e32 v14, 2, v14
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 2, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_lshlrev_b32_e32 v14, 2, v14
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 1, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	v_lshlrev_b32_e32 v13, 2, v14
	ds_bpermute_b32 v13, v13, v12
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s1, v12
	s_and_not1_b32 s5, 31, s1
	s_and_not1_b32 s1, 0xffff, s1
	s_wait_alu 0xfffe
	v_cmp_eq_u32_e32 vcc_lo, s5, v33
	s_and_saveexec_b32 s5, vcc_lo
	s_cbranch_execz .LBB_467
; %bb.466:
	s_lshr_b32 s8, s1, 5
	s_cmp_gt_u32 s1, 31
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	s_cmp_lg_u32 s8, 1
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s8, 2
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s8, 3
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s8, 4
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s8, 5
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s8, 6
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v10, 0, v10, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s8, 7
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v9, 0, v9, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v11, 0, v11, vcc_lo
.LBB_467:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s5
	s_cmp_lt_i32 s0, 1
	s_cbranch_scc1 .LBB_527
.LBB_468:
	v_xor_b32_e32 v13, 16, v59
	v_max_u32_e32 v12, v2, v3
	v_xor_b32_e32 v14, 8, v59
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	v_max3_u32 v12, v12, v1, v8
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v59, v13, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_max3_u32 v12, v12, v7, v10
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_max3_u32 v12, v12, v9, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v14, 2, v14
	v_lshlrev_b32_e32 v13, 2, v13
	ds_bpermute_b32 v13, v13, v12
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 4, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_lshlrev_b32_e32 v14, 2, v14
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 2, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_lshlrev_b32_e32 v14, 2, v14
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 1, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	v_lshlrev_b32_e32 v13, 2, v14
	ds_bpermute_b32 v13, v13, v12
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s5, v12
	s_and_not1_b32 s8, 31, s5
	s_and_not1_b32 s5, 0xffff, s5
	s_wait_alu 0xfffe
	v_cmp_eq_u32_e32 vcc_lo, s8, v33
	s_and_saveexec_b32 s8, vcc_lo
	s_cbranch_execz .LBB_470
; %bb.469:
	s_lshr_b32 s9, s5, 5
	s_cmp_gt_u32 s5, 31
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	s_cmp_lg_u32 s9, 1
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s9, 2
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s9, 3
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s9, 4
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s9, 5
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s9, 6
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v10, 0, v10, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s9, 7
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v9, 0, v9, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v11, 0, v11, vcc_lo
.LBB_470:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s8
	s_cmp_lt_i32 s0, 2
	s_mov_b32 s8, -1
	s_cbranch_scc0 .LBB_528
.LBB_471:
	s_cmp_lt_i32 s0, 3
	s_mov_b32 s9, -1
	s_cbranch_scc1 .LBB_531
.LBB_472:
	v_xor_b32_e32 v13, 16, v59
	v_max_u32_e32 v12, v2, v3
	v_xor_b32_e32 v14, 8, v59
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	v_max3_u32 v12, v12, v1, v8
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v59, v13, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_max3_u32 v12, v12, v7, v10
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_max3_u32 v12, v12, v9, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v14, 2, v14
	v_lshlrev_b32_e32 v13, 2, v13
	ds_bpermute_b32 v13, v13, v12
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 4, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_lshlrev_b32_e32 v14, 2, v14
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 2, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_lshlrev_b32_e32 v14, 2, v14
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 1, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	v_lshlrev_b32_e32 v13, 2, v14
	ds_bpermute_b32 v13, v13, v12
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s9, v12
	s_wait_alu 0xfffe
	s_and_not1_b32 s11, 31, s9
	s_and_not1_b32 s9, 0xffff, s9
	s_wait_alu 0xfffe
	v_cmp_eq_u32_e32 vcc_lo, s11, v33
	s_and_saveexec_b32 s11, vcc_lo
	s_cbranch_execz .LBB_474
; %bb.473:
	s_lshr_b32 s14, s9, 5
	s_cmp_gt_u32 s9, 31
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s14, 1
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s14, 2
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s14, 3
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s14, 4
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s14, 5
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s14, 6
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v10, 0, v10, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s14, 7
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v9, 0, v9, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v11, 0, v11, vcc_lo
.LBB_474:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s11
	s_cmp_lt_i32 s0, 4
	s_mov_b32 s11, -1
	s_cbranch_scc0 .LBB_532
.LBB_475:
	s_cmp_lt_i32 s0, 5
	s_mov_b32 s14, -1
	s_cbranch_scc1 .LBB_535
.LBB_476:
	v_xor_b32_e32 v13, 16, v59
	v_max_u32_e32 v12, v2, v3
	v_xor_b32_e32 v14, 8, v59
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	v_max3_u32 v12, v12, v1, v8
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v59, v13, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_max3_u32 v12, v12, v7, v10
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_max3_u32 v12, v12, v9, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v14, 2, v14
	v_lshlrev_b32_e32 v13, 2, v13
	ds_bpermute_b32 v13, v13, v12
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 4, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_lshlrev_b32_e32 v14, 2, v14
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 2, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_lshlrev_b32_e32 v14, 2, v14
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 1, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	v_lshlrev_b32_e32 v13, 2, v14
	ds_bpermute_b32 v13, v13, v12
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s14, v12
	s_wait_alu 0xfffe
	s_and_not1_b32 s15, 31, s14
	s_and_not1_b32 s14, 0xffff, s14
	s_wait_alu 0xfffe
	v_cmp_eq_u32_e32 vcc_lo, s15, v33
	s_and_saveexec_b32 s15, vcc_lo
	s_cbranch_execz .LBB_478
; %bb.477:
	s_lshr_b32 s16, s14, 5
	s_cmp_gt_u32 s14, 31
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s16, 1
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s16, 2
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s16, 3
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s16, 4
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s16, 5
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s16, 6
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v10, 0, v10, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s16, 7
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v9, 0, v9, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v11, 0, v11, vcc_lo
.LBB_478:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s15
	s_cmp_lt_i32 s0, 6
	s_mov_b32 s15, -1
	s_cbranch_scc0 .LBB_536
.LBB_479:
	s_cmp_lt_i32 s0, 7
	s_mov_b32 s16, -1
	s_cbranch_scc1 .LBB_481
.LBB_480:
	v_max_u32_e32 v2, v2, v3
	v_xor_b32_e32 v3, 16, v59
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_u32 v1, v2, v1, v8
	v_cmp_gt_i32_e32 vcc_lo, 32, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_max3_u32 v1, v1, v7, v10
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v2, v59, v3, vcc_lo
	v_xor_b32_e32 v3, 8, v59
	v_max3_u32 v1, v1, v9, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v3
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v3, v59, v3 :: v_dual_lshlrev_b32 v2, 2, v2
	ds_bpermute_b32 v2, v2, v1
	v_lshlrev_b32_e32 v3, 2, v3
	s_wait_dscnt 0x0
	v_max_u32_e32 v1, v1, v2
	ds_bpermute_b32 v2, v3, v1
	v_xor_b32_e32 v3, 4, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v3
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v59, v3, vcc_lo
	v_lshlrev_b32_e32 v3, 2, v3
	s_wait_dscnt 0x0
	v_max_u32_e32 v1, v1, v2
	ds_bpermute_b32 v2, v3, v1
	v_xor_b32_e32 v3, 2, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v3
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v59, v3, vcc_lo
	v_lshlrev_b32_e32 v3, 2, v3
	s_wait_dscnt 0x0
	v_max_u32_e32 v1, v1, v2
	ds_bpermute_b32 v2, v3, v1
	v_xor_b32_e32 v3, 1, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v3
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v59, v3, vcc_lo
	s_wait_dscnt 0x0
	v_max_u32_e32 v1, v1, v2
	v_lshlrev_b32_e32 v2, 2, v3
	ds_bpermute_b32 v2, v2, v1
	s_wait_dscnt 0x0
	v_max_u32_e32 v1, v1, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s16, v1
	s_wait_alu 0xfffe
	s_and_not1_b32 s16, 0xffff, s16
.LBB_481:
	s_cmp_eq_u32 s0, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v58, 0
	s_wait_alu 0xfffe
	s_cselect_b32 s1, s1, -1
	s_cmp_eq_u32 s0, 1
	v_mov_b32_e32 v57, 0
	s_wait_alu 0xfffe
	s_cselect_b32 s1, s5, s1
	s_cmp_eq_u32 s0, 2
	v_mov_b32_e32 v71, 0
	s_wait_alu 0xfffe
	s_cselect_b32 s1, s8, s1
	s_cmp_eq_u32 s0, 3
	v_mov_b32_e32 v70, 0
	s_wait_alu 0xfffe
	s_cselect_b32 s1, s9, s1
	s_cmp_eq_u32 s0, 4
	v_mov_b32_e32 v56, 0
	s_wait_alu 0xfffe
	s_cselect_b32 s1, s11, s1
	s_cmp_eq_u32 s0, 5
	v_mov_b32_e32 v65, 0
	s_wait_alu 0xfffe
	s_cselect_b32 s1, s14, s1
	s_cmp_eq_u32 s0, 6
	s_wait_alu 0xfffe
	s_cselect_b32 s1, s15, s1
	s_cmp_eq_u32 s0, 7
	s_wait_alu 0xfffe
	s_cselect_b32 s5, s16, s1
	s_mul_i32 s1, s4, s3
	s_wait_alu 0xfffe
	s_cmp_gt_i32 s5, -1
	s_cselect_b32 s0, -1, 0
	s_cmp_lt_i32 s5, s52
	s_cselect_b32 s3, -1, 0
	s_sub_co_i32 s1, s10, s1
	s_wait_alu 0xfffe
	s_and_b32 s8, s0, s3
	v_lshl_or_b32 v34, s1, 3, v61
	s_ashr_i32 s1, s5, 31
	s_lshl_b32 s0, s5, 1
	s_wait_alu 0xfffe
	s_and_b32 s3, s8, exec_lo
	s_cselect_b32 s0, s0, 0
	v_cmp_gt_i32_e32 vcc_lo, s54, v34
	s_cselect_b32 s3, s1, 0
	s_ashr_i32 s55, s54, 31
	s_lshr_b32 s9, s53, 31
	s_wait_alu 0xfffe
	s_mul_i32 s3, s3, s54
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v1, 0, v34, vcc_lo
	s_add_co_i32 s9, s53, s9
	s_ashr_i32 s1, s53, 31
	s_wait_alu 0xfffe
	s_ashr_i32 s9, s9, 1
	s_lshr_b32 s10, s1, 28
	v_ashrrev_i32_e32 v2, 31, v1
	s_wait_alu 0xfffe
	s_ashr_i32 s11, s9, 31
	s_add_co_i32 s10, s53, s10
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, s0, s54, v[1:2]
	s_mul_i32 s0, s0, s55
	s_wait_alu 0xfffe
	v_add3_u32 v2, s0, s3, v2
	v_add_co_u32 v7, s0, v1, s54
	v_mul_lo_u32 v3, v1, s11
	v_mad_co_u64_u32 v[41:42], null, v1, s9, 0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v9, null, s55, v2, s0
	v_mul_lo_u32 v8, v2, s9
	s_ashr_i32 s3, s10, 4
	v_mul_lo_u32 v10, v7, s11
	v_mad_co_u64_u32 v[39:40], null, v7, s9, 0
	s_wait_alu 0xfffe
	s_ashr_i32 s0, s3, 31
	v_mul_lo_u32 v11, v9, s9
	s_wait_alu 0xfffe
	v_mul_lo_u32 v12, v1, s0
	v_mad_co_u64_u32 v[35:36], null, v1, s3, 0
	v_mul_lo_u32 v1, v2, s3
	v_mul_lo_u32 v2, v9, s3
	v_add3_u32 v42, v42, v3, v8
	v_mul_lo_u32 v3, v7, s0
	v_mad_co_u64_u32 v[37:38], null, v7, s3, 0
	v_add3_u32 v40, v40, v10, v11
	v_add_co_u32 v73, s0, s18, v41
	v_add3_u32 v36, v36, v12, v1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v74, null, s19, v42, s0
	v_add_co_u32 v67, s0, s18, v39
	v_add3_u32 v38, v38, v3, v2
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v72, null, s19, v40, s0
	v_add_co_u32 v61, s0, s20, v35
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v62, null, s21, v36, s0
	v_add_co_u32 v7, s0, s20, v37
	v_mov_b32_e32 v1, 0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v8, null, s21, v38, s0
	v_mov_b32_e32 v2, 0
	s_mov_b32 s9, exec_lo
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0
	v_cmpx_gt_i32_e64 s3, v33
	s_cbranch_execz .LBB_483
; %bb.482:
	v_lshlrev_b32_e32 v9, 3, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v2, s0, v73, v9
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v3, null, 0, v74, s0
	v_add_co_u32 v9, s0, v67, v9
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v10, null, 0, v72, s0
	v_add_co_u32 v11, s0, v61, v33
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v12, null, 0, v62, s0
	v_add_co_u32 v13, s0, v7, v33
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v14, null, 0, v8, s0
	s_clause 0x1
	global_load_b64 v[57:58], v[2:3], off th:TH_LOAD_NT
	global_load_b64 v[55:56], v[9:10], off th:TH_LOAD_NT
	s_clause 0x1
	global_load_u8 v70, v[11:12], off
	global_load_u8 v71, v[13:14], off
	v_lshlrev_b32_e32 v2, 4, v33
.LBB_483:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s9
	s_mov_b32 s0, s53
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v51, 0 :: v_dual_lshlrev_b32 v2, 1, v2
	s_wait_alu 0xfffe
	s_mul_u64 s[0:1], s[6:7], s[0:1]
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v53, 0
	s_wait_alu 0xfffe
	s_lshl_b64 s[0:1], s[0:1], 1
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v69, 0
	s_wait_alu 0xfffe
	s_add_nc_u64 s[6:7], s[12:13], s[0:1]
	s_clause 0x1
	global_load_b128 v[25:28], v2, s[6:7] offset:16
	global_load_b128 v[29:32], v2, s[6:7]
	v_mov_b32_e32 v54, 0
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_i32_e64 s3, v6
	s_cbranch_execz .LBB_485
; %bb.484:
	v_lshlrev_b32_e32 v3, 3, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v1, s0, v73, v3
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v2, null, 0, v74, s0
	v_add_co_u32 v9, s0, v67, v3
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v10, null, 0, v72, s0
	v_add_co_u32 v11, s0, v61, v33
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v12, null, 0, v62, s0
	v_add_co_u32 v13, s0, v7, v33
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v14, null, 0, v8, s0
	s_clause 0x1
	global_load_b64 v[53:54], v[1:2], off offset:256 th:TH_LOAD_NT
	global_load_b64 v[51:52], v[9:10], off offset:256 th:TH_LOAD_NT
	s_clause 0x1
	global_load_u8 v68, v[11:12], off offset:32
	global_load_u8 v69, v[13:14], off offset:32
	v_lshlrev_b32_e32 v1, 4, v6
.LBB_485:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v48, 0 :: v_dual_lshlrev_b32 v1, 1, v1
	v_mov_b32_e32 v49, 0
	s_mov_b32 s1, exec_lo
	s_clause 0x1
	global_load_b128 v[17:20], v1, s[6:7] offset:16
	global_load_b128 v[21:24], v1, s[6:7]
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v47, 0
	v_mov_b32_e32 v2, 0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v66, 0
	v_cmpx_gt_i32_e64 s3, v5
	s_cbranch_execz .LBB_487
; %bb.486:
	v_lshlrev_b32_e32 v6, 3, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v2, s0, v73, v6
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v3, null, 0, v74, s0
	v_add_co_u32 v9, s0, v67, v6
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v10, null, 0, v72, s0
	v_add_co_u32 v11, s0, v61, v33
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v12, null, 0, v62, s0
	v_add_co_u32 v13, s0, v7, v33
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v14, null, 0, v8, s0
	s_clause 0x1
	global_load_b64 v[49:50], v[2:3], off offset:512 th:TH_LOAD_NT
	global_load_b64 v[47:48], v[9:10], off offset:512 th:TH_LOAD_NT
	s_clause 0x1
	global_load_u8 v65, v[11:12], off offset:64
	global_load_u8 v66, v[13:14], off offset:64
	v_lshlrev_b32_e32 v2, 4, v5
.LBB_487:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v63, 0 :: v_dual_lshlrev_b32 v2, 1, v2
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v46, 0
	s_clause 0x1
	global_load_b128 v[9:12], v2, s[6:7] offset:16
	global_load_b128 v[13:16], v2, s[6:7]
	v_mov_b32_e32 v64, 0
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_i32_e64 s3, v4
	s_cbranch_execz .LBB_489
; %bb.488:
	v_lshlrev_b32_e32 v3, 3, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v1, s0, v73, v3
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v2, null, 0, v74, s0
	v_add_co_u32 v5, s0, v67, v3
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v6, null, 0, v72, s0
	v_add_co_u32 v61, s0, v61, v33
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v62, null, 0, v62, s0
	v_add_co_u32 v7, s0, v7, v33
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v8, null, 0, v8, s0
	s_clause 0x1
	global_load_b64 v[45:46], v[1:2], off offset:768 th:TH_LOAD_NT
	global_load_b64 v[43:44], v[5:6], off offset:768 th:TH_LOAD_NT
	s_clause 0x1
	global_load_u8 v63, v[61:62], off offset:96
	global_load_u8 v64, v[7:8], off offset:96
	v_lshlrev_b32_e32 v1, 4, v4
.LBB_489:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v5, 1, v1
	s_mov_b32 s1, exec_lo
	s_clause 0x1
	global_load_b128 v[1:4], v5, s[6:7] offset:16
	global_load_b128 v[5:8], v5, s[6:7]
	v_cmpx_gt_u32_e32 16, v0
	s_cbranch_execz .LBB_491
; %bb.490:
	v_and_b32_e32 v61, 7, v0
	v_lshlrev_b32_e32 v67, 28, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v62, v61, 22, 0.5
	v_cmp_ne_u32_e64 s0, 1, v61
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v62, 0.5, v62, s0
	v_cmp_ne_u32_e64 s0, 0, v61
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v61, 0, v62, s0
	v_lshlrev_b32_e32 v62, 2, v0
	v_and_or_b32 v61, 0x80000000, v67, v61
	ds_store_b32 v62, v61
.LBB_491:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_and_b32 s0, s8, vcc_lo
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB_522
; %bb.492:
	v_sub_nc_u32_e32 v67, s3, v33
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v61, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 0, v67
	s_cbranch_execz .LBB_494
; %bb.493:
	v_lshlrev_b32_e32 v94, 16, v30
	v_and_b32_e32 v61, 15, v57
	v_lshrrev_b32_e32 v62, 2, v57
	v_lshrrev_b32_e32 v72, 6, v57
	v_lshrrev_b32_e32 v73, 10, v57
	v_lshrrev_b32_e32 v74, 14, v57
	v_lshrrev_b32_e32 v75, 18, v57
	v_lshrrev_b32_e32 v76, 22, v57
	v_lshrrev_b32_e32 v57, 26, v57
	v_and_b32_e32 v30, 0xffff0000, v30
	v_lshlrev_b32_e32 v61, 2, v61
	v_and_b32_e32 v62, 60, v62
	v_lshlrev_b32_e32 v92, 16, v29
	v_and_b32_e32 v29, 0xffff0000, v29
	v_and_b32_e32 v72, 60, v72
	v_and_b32_e32 v73, 60, v73
	v_and_b32_e32 v74, 60, v74
	v_and_b32_e32 v75, 60, v75
	v_and_b32_e32 v76, 60, v76
	v_and_b32_e32 v57, 60, v57
	ds_load_b32 v61, v61
	ds_load_b32 v62, v62
	ds_load_b32 v72, v72
	ds_load_b32 v73, v73
	ds_load_b32 v74, v74
	ds_load_b32 v75, v75
	ds_load_b32 v76, v76
	ds_load_b32 v57, v57
	v_lshrrev_b32_e32 v80, 10, v58
	v_and_b32_e32 v77, 15, v58
	v_lshrrev_b32_e32 v78, 2, v58
	v_lshrrev_b32_e32 v79, 6, v58
	v_lshrrev_b32_e32 v81, 14, v58
	v_lshrrev_b32_e32 v82, 18, v58
	v_lshrrev_b32_e32 v83, 22, v58
	v_lshrrev_b32_e32 v58, 26, v58
	v_lshrrev_b32_e32 v85, 2, v55
	v_lshlrev_b32_e32 v77, 2, v77
	v_and_b32_e32 v78, 60, v78
	v_and_b32_e32 v79, 60, v79
	s_wait_dscnt 0x7
	v_fma_f32 v61, v61, v92, 0
	v_and_b32_e32 v81, 60, v81
	v_and_b32_e32 v82, 60, v82
	v_and_b32_e32 v83, 60, v83
	s_wait_dscnt 0x6
	v_dual_fmac_f32 v61, v62, v29 :: v_dual_and_b32 v58, 60, v58
	v_and_b32_e32 v85, 60, v85
	v_lshrrev_b32_e32 v86, 6, v55
	v_lshrrev_b32_e32 v87, 10, v55
	v_lshrrev_b32_e32 v88, 14, v55
	s_wait_dscnt 0x5
	v_dual_fmac_f32 v61, v72, v94 :: v_dual_and_b32 v80, 60, v80
	v_lshlrev_b32_e32 v72, 16, v31
	v_and_b32_e32 v31, 0xffff0000, v31
	v_lshrrev_b32_e32 v89, 18, v55
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v61, v73, v30
	v_lshrrev_b32_e32 v90, 22, v55
	v_and_b32_e32 v87, 60, v87
	v_and_b32_e32 v91, 15, v56
	v_and_b32_e32 v89, 60, v89
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v61, v74, v72
	v_lshrrev_b32_e32 v93, 2, v56
	v_lshrrev_b32_e32 v62, 6, v56
	v_lshlrev_b32_e32 v91, 2, v91
	v_lshrrev_b32_e32 v74, 18, v56
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v61, v75, v31
	v_and_b32_e32 v84, 15, v55
	v_lshrrev_b32_e32 v55, 26, v55
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v84, 2, v84
	ds_load_b32 v77, v77
	ds_load_b32 v78, v78
	ds_load_b32 v79, v79
	ds_load_b32 v80, v80
	ds_load_b32 v81, v81
	ds_load_b32 v82, v82
	ds_load_b32 v83, v83
	ds_load_b32 v58, v58
	ds_load_b32 v84, v84
	ds_load_b32 v85, v85
	v_and_b32_e32 v86, 60, v86
	v_and_b32_e32 v88, 60, v88
	v_and_b32_e32 v90, 60, v90
	v_and_b32_e32 v55, 60, v55
	ds_load_b32 v86, v86
	ds_load_b32 v87, v87
	ds_load_b32 v88, v88
	ds_load_b32 v89, v89
	ds_load_b32 v90, v90
	ds_load_b32 v55, v55
	s_wait_dscnt 0x7
	v_fma_f32 v84, v84, v92, 0
	v_and_b32_e32 v92, 60, v93
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v84, v85, v29 :: v_dual_and_b32 v29, 60, v62
	ds_load_b32 v73, v91
	ds_load_b32 v85, v92
	ds_load_b32 v29, v29
	s_wait_dscnt 0x8
	v_fmac_f32_e32 v84, v86, v94
	v_lshlrev_b32_e32 v86, 16, v32
	v_lshrrev_b32_e32 v62, 10, v56
	v_and_b32_e32 v32, 0xffff0000, v32
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v84, v87, v30
	v_lshrrev_b32_e32 v30, 14, v56
	v_fmac_f32_e32 v61, v76, v86
	v_and_b32_e32 v62, 60, v62
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v84, v88, v72
	v_and_b32_e32 v30, 60, v30
	v_and_b32_e32 v72, 60, v74
	v_dual_fmac_f32 v61, v57, v32 :: v_dual_lshlrev_b32 v74, 16, v25
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v84, v89, v31
	v_lshrrev_b32_e32 v31, 22, v56
	v_lshrrev_b32_e32 v56, 26, v56
	v_and_b32_e32 v25, 0xffff0000, v25
	v_fmac_f32_e32 v61, v77, v74
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v84, v90, v86 :: v_dual_and_b32 v31, 60, v31
	v_and_b32_e32 v56, 60, v56
	v_lshlrev_b32_e32 v57, 16, v26
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v61, v78, v25 :: v_dual_fmac_f32 v84, v55, v32
	ds_load_b32 v32, v62
	ds_load_b32 v30, v30
	ds_load_b32 v55, v72
	ds_load_b32 v31, v31
	ds_load_b32 v56, v56
	s_wait_dscnt 0x7
	v_dual_fmac_f32 v61, v79, v57 :: v_dual_fmac_f32 v84, v73, v74
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v84, v85, v25 :: v_dual_and_b32 v25, 0xffff0000, v26
	v_lshlrev_b32_e32 v26, 16, v27
	s_wait_dscnt 0x5
	v_dual_fmac_f32 v84, v29, v57 :: v_dual_and_b32 v27, 0xffff0000, v27
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v61, v80, v25
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v84, v32, v25 :: v_dual_and_b32 v29, 7, v71
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v61, v81, v26
	v_cvt_f32_ubyte0_e32 v29, v29
	v_and_b32_e32 v32, 0x7f, v70
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v84, v30, v26 :: v_dual_lshlrev_b32 v25, 16, v28
	v_and_b32_e32 v26, 7, v70
	v_and_b32_e32 v30, 0x7f, v71
	v_fmac_f32_e32 v61, v82, v27
	s_wait_dscnt 0x2
	v_dual_mul_f32 v29, 0x3b000000, v29 :: v_dual_fmac_f32 v84, v55, v27
	v_cvt_f32_ubyte0_e32 v26, v26
	v_lshl_add_u32 v27, v30, 20, 0x3c000000
	v_cmp_lt_u32_e32 vcc_lo, 7, v30
	v_lshl_add_u32 v55, v32, 20, 0x3c000000
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v61, v83, v25 :: v_dual_mul_f32 v26, 0x3b000000, v26
	s_wait_dscnt 0x1
	s_wait_alu 0xfffd
	v_dual_fmac_f32 v84, v31, v25 :: v_dual_cndmask_b32 v27, v29, v27
	v_cmp_lt_u32_e32 vcc_lo, 7, v32
	v_and_b32_e32 v28, 0xffff0000, v28
	v_lshlrev_b32_e32 v29, 24, v71
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v26, v26, v55, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v30
	v_fmac_f32_e32 v61, v58, v28
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_dual_fmac_f32 v84, v56, v28 :: v_dual_cndmask_b32 v25, 0x7fc00000, v27
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v32
	v_lshlrev_b32_e32 v27, 24, v70
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_and_or_b32 v25, 0x80000000, v29, v25
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v26, 0x7fc00000, v26, vcc_lo
	v_and_or_b32 v26, 0x80000000, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v62, v61, v26, 0
	v_fma_f32 v61, v84, v25, 0
.LBB_494:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_lt_i32_e32 32, v67
	s_cbranch_execz .LBB_496
; %bb.495:
	v_lshlrev_b32_e32 v82, 16, v22
	v_and_b32_e32 v25, 15, v53
	v_lshrrev_b32_e32 v26, 2, v53
	v_lshrrev_b32_e32 v27, 6, v53
	v_lshrrev_b32_e32 v28, 10, v53
	v_lshrrev_b32_e32 v29, 14, v53
	v_lshrrev_b32_e32 v30, 18, v53
	v_lshrrev_b32_e32 v31, 22, v53
	v_lshrrev_b32_e32 v32, 26, v53
	v_and_b32_e32 v22, 0xffff0000, v22
	v_lshlrev_b32_e32 v25, 2, v25
	v_and_b32_e32 v26, 60, v26
	v_lshlrev_b32_e32 v80, 16, v21
	v_and_b32_e32 v21, 0xffff0000, v21
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
	v_lshrrev_b32_e32 v56, 6, v54
	v_lshrrev_b32_e32 v58, 14, v54
	v_lshrrev_b32_e32 v55, 2, v54
	v_lshrrev_b32_e32 v57, 10, v54
	v_lshrrev_b32_e32 v70, 18, v54
	v_lshrrev_b32_e32 v71, 22, v54
	v_lshrrev_b32_e32 v73, 2, v51
	v_lshrrev_b32_e32 v74, 6, v51
	v_and_b32_e32 v57, 60, v57
	v_lshrrev_b32_e32 v75, 10, v51
	v_lshrrev_b32_e32 v76, 14, v51
	v_and_b32_e32 v73, 60, v73
	s_wait_dscnt 0x7
	v_fma_f32 v25, v25, v80, 0
	v_lshrrev_b32_e32 v77, 18, v51
	v_lshrrev_b32_e32 v78, 22, v51
	v_and_b32_e32 v75, 60, v75
	v_and_b32_e32 v79, 15, v52
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v25, v26, v21
	v_and_b32_e32 v77, 60, v77
	v_lshrrev_b32_e32 v81, 2, v52
	v_lshrrev_b32_e32 v26, 6, v52
	v_lshlrev_b32_e32 v79, 2, v79
	s_wait_dscnt 0x5
	v_dual_fmac_f32 v25, v27, v82 :: v_dual_and_b32 v56, 60, v56
	v_lshlrev_b32_e32 v27, 16, v23
	v_and_b32_e32 v23, 0xffff0000, v23
	v_and_b32_e32 v53, 15, v54
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v25, v28, v22
	v_lshrrev_b32_e32 v54, 26, v54
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v53, 2, v53
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v25, v29, v27 :: v_dual_and_b32 v58, 60, v58
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_and_b32_e32 v54, 60, v54
	v_lshrrev_b32_e32 v29, 18, v52
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v25, v30, v23
	v_and_b32_e32 v72, 15, v51
	v_lshlrev_b32_e32 v30, 16, v17
	v_and_b32_e32 v55, 60, v55
	v_and_b32_e32 v17, 0xffff0000, v17
	v_and_b32_e32 v70, 60, v70
	v_and_b32_e32 v71, 60, v71
	v_lshlrev_b32_e32 v72, 2, v72
	ds_load_b32 v53, v53
	ds_load_b32 v55, v55
	ds_load_b32 v56, v56
	ds_load_b32 v57, v57
	ds_load_b32 v58, v58
	ds_load_b32 v70, v70
	ds_load_b32 v71, v71
	ds_load_b32 v54, v54
	ds_load_b32 v72, v72
	ds_load_b32 v73, v73
	v_lshrrev_b32_e32 v51, 26, v51
	v_and_b32_e32 v74, 60, v74
	v_and_b32_e32 v76, 60, v76
	v_and_b32_e32 v78, 60, v78
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v51, 60, v51
	ds_load_b32 v74, v74
	ds_load_b32 v75, v75
	ds_load_b32 v76, v76
	ds_load_b32 v77, v77
	ds_load_b32 v78, v78
	ds_load_b32 v51, v51
	s_wait_dscnt 0x7
	v_fma_f32 v72, v72, v80, 0
	v_and_b32_e32 v80, 60, v81
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v72, v73, v21 :: v_dual_and_b32 v21, 60, v26
	ds_load_b32 v28, v79
	ds_load_b32 v73, v80
	ds_load_b32 v21, v21
	s_wait_dscnt 0x8
	v_fmac_f32_e32 v72, v74, v82
	v_lshlrev_b32_e32 v74, 16, v24
	v_lshrrev_b32_e32 v26, 10, v52
	v_and_b32_e32 v24, 0xffff0000, v24
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v72, v75, v22
	v_lshrrev_b32_e32 v22, 14, v52
	v_fmac_f32_e32 v25, v31, v74
	v_and_b32_e32 v26, 60, v26
	s_wait_dscnt 0x6
	v_dual_fmac_f32 v72, v76, v27 :: v_dual_and_b32 v27, 60, v29
	v_lshrrev_b32_e32 v29, 26, v52
	v_dual_fmac_f32 v25, v32, v24 :: v_dual_and_b32 v22, 60, v22
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v72, v77, v23
	v_lshrrev_b32_e32 v23, 22, v52
	v_and_b32_e32 v29, 60, v29
	v_fmac_f32_e32 v25, v53, v30
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v72, v78, v74 :: v_dual_and_b32 v23, 60, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v25, v55, v17
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v72, v51, v24
	ds_load_b32 v24, v26
	ds_load_b32 v22, v22
	ds_load_b32 v26, v27
	ds_load_b32 v23, v23
	ds_load_b32 v27, v29
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v72, v28, v30
	v_lshlrev_b32_e32 v28, 16, v18
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v72, v73, v17 :: v_dual_and_b32 v17, 0xffff0000, v18
	v_dual_fmac_f32 v25, v56, v28 :: v_dual_lshlrev_b32 v18, 16, v19
	s_wait_dscnt 0x5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v72, v21, v28 :: v_dual_and_b32 v19, 0xffff0000, v19
	v_and_b32_e32 v21, 7, v69
	v_fmac_f32_e32 v25, v57, v17
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v72, v24, v17
	v_cvt_f32_ubyte0_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v25, v58, v18 :: v_dual_and_b32 v24, 0x7f, v68
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v72, v22, v18 :: v_dual_lshlrev_b32 v17, 16, v20
	v_and_b32_e32 v18, 7, v68
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v25, v70, v19 :: v_dual_and_b32 v22, 0x7f, v69
	s_wait_dscnt 0x2
	v_dual_mul_f32 v21, 0x3b000000, v21 :: v_dual_fmac_f32 v72, v26, v19
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v18, v18
	v_lshl_add_u32 v19, v22, 20, 0x3c000000
	v_cmp_lt_u32_e32 vcc_lo, 7, v22
	v_lshl_add_u32 v26, v24, 20, 0x3c000000
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v25, v71, v17 :: v_dual_mul_f32 v18, 0x3b000000, v18
	s_wait_dscnt 0x1
	s_wait_alu 0xfffd
	v_dual_fmac_f32 v72, v23, v17 :: v_dual_cndmask_b32 v19, v21, v19
	v_cmp_lt_u32_e32 vcc_lo, 7, v24
	v_and_b32_e32 v20, 0xffff0000, v20
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v18, v18, v26 :: v_dual_lshlrev_b32 v21, 24, v69
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v22
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v25, v54, v20
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_dual_fmac_f32 v72, v27, v20 :: v_dual_cndmask_b32 v17, 0x7fc00000, v19
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v24
	v_lshlrev_b32_e32 v19, 24, v68
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_or_b32 v17, 0x80000000, v21, v17
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v18, 0x7fc00000, v18 :: v_dual_fmac_f32 v61, v72, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v18, 0x80000000, v19, v18
	v_fmac_f32_e32 v62, v25, v18
.LBB_496:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_lt_i32_e32 64, v67
	s_cbranch_execz .LBB_498
; %bb.497:
	v_and_b32_e32 v17, 15, v49
	v_lshrrev_b32_e32 v18, 2, v49
	v_lshrrev_b32_e32 v19, 6, v49
	v_lshrrev_b32_e32 v20, 10, v49
	v_lshrrev_b32_e32 v21, 14, v49
	v_lshrrev_b32_e32 v22, 18, v49
	v_lshrrev_b32_e32 v23, 22, v49
	v_lshrrev_b32_e32 v24, 26, v49
	v_lshlrev_b32_e32 v17, 2, v17
	v_lshlrev_b32_e32 v57, 16, v13
	v_and_b32_e32 v13, 0xffff0000, v13
	v_and_b32_e32 v18, 60, v18
	v_and_b32_e32 v19, 60, v19
	v_and_b32_e32 v20, 60, v20
	v_and_b32_e32 v21, 60, v21
	v_and_b32_e32 v23, 60, v23
	v_lshlrev_b32_e32 v68, 16, v14
	v_and_b32_e32 v22, 60, v22
	v_and_b32_e32 v14, 0xffff0000, v14
	v_and_b32_e32 v24, 60, v24
	ds_load_b32 v17, v17
	ds_load_b32 v18, v18
	ds_load_b32 v19, v19
	ds_load_b32 v20, v20
	ds_load_b32 v21, v21
	ds_load_b32 v22, v22
	ds_load_b32 v23, v23
	ds_load_b32 v24, v24
	v_and_b32_e32 v25, 15, v50
	v_lshrrev_b32_e32 v30, 18, v50
	v_lshrrev_b32_e32 v26, 2, v50
	v_lshrrev_b32_e32 v27, 6, v50
	v_lshrrev_b32_e32 v28, 10, v50
	v_lshrrev_b32_e32 v29, 14, v50
	v_lshrrev_b32_e32 v31, 22, v50
	v_lshrrev_b32_e32 v32, 26, v50
	v_and_b32_e32 v49, 15, v47
	v_lshrrev_b32_e32 v50, 2, v47
	v_lshlrev_b32_e32 v25, 2, v25
	v_and_b32_e32 v26, 60, v26
	s_wait_dscnt 0x7
	v_fma_f32 v17, v17, v57, 0
	v_and_b32_e32 v27, 60, v27
	v_and_b32_e32 v28, 60, v28
	v_and_b32_e32 v29, 60, v29
	v_and_b32_e32 v31, 60, v31
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v17, v18, v13
	v_lshlrev_b32_e32 v49, 2, v49
	v_lshrrev_b32_e32 v51, 6, v47
	v_lshrrev_b32_e32 v52, 10, v47
	v_lshrrev_b32_e32 v53, 14, v47
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v17, v19, v68
	v_lshlrev_b32_e32 v19, 16, v15
	v_and_b32_e32 v15, 0xffff0000, v15
	v_and_b32_e32 v32, 60, v32
	v_lshrrev_b32_e32 v54, 18, v47
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v17, v20, v14
	v_and_b32_e32 v30, 60, v30
	v_lshrrev_b32_e32 v55, 22, v47
	v_lshrrev_b32_e32 v47, 26, v47
	v_and_b32_e32 v51, 60, v51
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v17, v21, v19 :: v_dual_and_b32 v50, 60, v50
	v_and_b32_e32 v52, 60, v52
	v_and_b32_e32 v53, 60, v53
	v_and_b32_e32 v55, 60, v55
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v17, v22, v15
	ds_load_b32 v25, v25
	ds_load_b32 v26, v26
	ds_load_b32 v27, v27
	ds_load_b32 v28, v28
	ds_load_b32 v29, v29
	ds_load_b32 v30, v30
	ds_load_b32 v31, v31
	ds_load_b32 v32, v32
	ds_load_b32 v49, v49
	ds_load_b32 v50, v50
	v_and_b32_e32 v54, 60, v54
	v_and_b32_e32 v47, 60, v47
	ds_load_b32 v51, v51
	ds_load_b32 v52, v52
	ds_load_b32 v53, v53
	ds_load_b32 v54, v54
	ds_load_b32 v55, v55
	ds_load_b32 v47, v47
	v_lshlrev_b32_e32 v22, 16, v9
	v_lshrrev_b32_e32 v58, 2, v48
	v_lshrrev_b32_e32 v18, 6, v48
	v_lshrrev_b32_e32 v21, 18, v48
	v_and_b32_e32 v9, 0xffff0000, v9
	s_wait_dscnt 0x7
	v_fma_f32 v49, v49, v57, 0
	v_and_b32_e32 v57, 60, v58
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v49, v50, v13
	v_and_b32_e32 v13, 60, v18
	v_lshrrev_b32_e32 v18, 10, v48
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v49, v51, v68
	v_lshlrev_b32_e32 v51, 16, v16
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v49, v52, v14 :: v_dual_and_b32 v16, 0xffff0000, v16
	v_fmac_f32_e32 v17, v23, v51
	v_lshrrev_b32_e32 v14, 14, v48
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v49, v53, v19
	v_fmac_f32_e32 v17, v24, v16
	v_and_b32_e32 v19, 60, v21
	v_lshrrev_b32_e32 v21, 26, v48
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v49, v54, v15 :: v_dual_and_b32 v56, 15, v48
	v_fmac_f32_e32 v17, v25, v22
	v_lshrrev_b32_e32 v15, 22, v48
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v49, v55, v51 :: v_dual_and_b32 v18, 60, v18
	v_lshlrev_b32_e32 v56, 2, v56
	ds_load_b32 v20, v56
	ds_load_b32 v50, v57
	ds_load_b32 v13, v13
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v49, v47, v16
	v_and_b32_e32 v15, 60, v15
	v_and_b32_e32 v21, 60, v21
	v_fmac_f32_e32 v17, v26, v9
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v49, v20, v22
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v49, v50, v9 :: v_dual_and_b32 v14, 60, v14
	ds_load_b32 v16, v18
	ds_load_b32 v14, v14
	ds_load_b32 v18, v19
	ds_load_b32 v15, v15
	ds_load_b32 v19, v21
	v_and_b32_e32 v9, 0xffff0000, v10
	v_lshlrev_b32_e32 v20, 16, v10
	v_lshlrev_b32_e32 v10, 16, v11
	v_and_b32_e32 v11, 0xffff0000, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v27, v20
	v_fmac_f32_e32 v17, v28, v9
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v49, v13, v20
	v_and_b32_e32 v13, 7, v66
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v17, v29, v10
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v49, v16, v9
	v_lshlrev_b32_e32 v9, 16, v12
	v_and_b32_e32 v12, 0xffff0000, v12
	v_cvt_f32_ubyte0_e32 v13, v13
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v49, v14, v10 :: v_dual_and_b32 v16, 0x7f, v65
	v_dual_fmac_f32 v17, v30, v11 :: v_dual_and_b32 v10, 7, v65
	v_and_b32_e32 v14, 0x7f, v66
	v_mul_f32_e32 v13, 0x3b000000, v13
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v49, v18, v11
	v_cvt_f32_ubyte0_e32 v10, v10
	v_lshl_add_u32 v18, v16, 20, 0x3c000000
	v_lshl_add_u32 v11, v14, 20, 0x3c000000
	v_cmp_lt_u32_e32 vcc_lo, 7, v14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v17, v31, v9 :: v_dual_mul_f32 v10, 0x3b000000, v10
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v49, v15, v9
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v11, v13, v11, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, 7, v16
	v_lshlrev_b32_e32 v13, 24, v66
	v_fmac_f32_e32 v17, v32, v12
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_dual_fmac_f32 v49, v19, v12 :: v_dual_cndmask_b32 v10, v10, v18
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v9, 0x7fc00000, v11, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v16
	v_lshlrev_b32_e32 v11, 24, v65
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_or_b32 v9, 0x80000000, v13, v9
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v10, 0x7fc00000, v10 :: v_dual_fmac_f32 v61, v49, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v10, 0x80000000, v11, v10
	v_fmac_f32_e32 v62, v17, v10
.LBB_498:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_lt_i32_e32 0x60, v67
	s_cbranch_execz .LBB_500
; %bb.499:
	v_and_b32_e32 v9, 15, v45
	v_lshrrev_b32_e32 v10, 2, v45
	v_lshrrev_b32_e32 v11, 6, v45
	v_lshrrev_b32_e32 v12, 10, v45
	v_lshrrev_b32_e32 v13, 14, v45
	v_lshrrev_b32_e32 v14, 18, v45
	v_lshrrev_b32_e32 v15, 22, v45
	v_lshrrev_b32_e32 v16, 26, v45
	v_lshlrev_b32_e32 v9, 2, v9
	v_lshlrev_b32_e32 v45, 16, v5
	v_and_b32_e32 v5, 0xffff0000, v5
	v_and_b32_e32 v10, 60, v10
	v_and_b32_e32 v11, 60, v11
	v_and_b32_e32 v12, 60, v12
	v_and_b32_e32 v13, 60, v13
	v_and_b32_e32 v15, 60, v15
	v_lshlrev_b32_e32 v47, 16, v6
	v_and_b32_e32 v14, 60, v14
	v_and_b32_e32 v16, 60, v16
	ds_load_b32 v9, v9
	ds_load_b32 v10, v10
	ds_load_b32 v11, v11
	ds_load_b32 v12, v12
	ds_load_b32 v13, v13
	ds_load_b32 v14, v14
	ds_load_b32 v15, v15
	ds_load_b32 v16, v16
	v_lshrrev_b32_e32 v20, 10, v46
	v_and_b32_e32 v6, 0xffff0000, v6
	v_and_b32_e32 v17, 15, v46
	v_lshrrev_b32_e32 v18, 2, v46
	v_lshrrev_b32_e32 v19, 6, v46
	v_lshrrev_b32_e32 v21, 14, v46
	v_lshrrev_b32_e32 v22, 18, v46
	v_lshrrev_b32_e32 v23, 22, v46
	v_lshrrev_b32_e32 v24, 26, v46
	v_and_b32_e32 v25, 15, v43
	v_lshrrev_b32_e32 v26, 2, v43
	v_lshlrev_b32_e32 v17, 2, v17
	s_wait_dscnt 0x7
	v_fma_f32 v9, v9, v45, 0
	v_and_b32_e32 v19, 60, v19
	v_and_b32_e32 v21, 60, v21
	v_and_b32_e32 v23, 60, v23
	s_wait_dscnt 0x6
	v_dual_fmac_f32 v9, v10, v5 :: v_dual_and_b32 v24, 60, v24
	v_lshlrev_b32_e32 v25, 2, v25
	v_lshrrev_b32_e32 v27, 6, v43
	v_lshrrev_b32_e32 v28, 10, v43
	v_lshrrev_b32_e32 v29, 14, v43
	s_wait_dscnt 0x5
	v_dual_fmac_f32 v9, v11, v47 :: v_dual_and_b32 v20, 60, v20
	v_lshlrev_b32_e32 v11, 16, v7
	v_and_b32_e32 v7, 0xffff0000, v7
	v_and_b32_e32 v22, 60, v22
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v9, v12, v6
	v_and_b32_e32 v18, 60, v18
	v_lshrrev_b32_e32 v30, 18, v43
	v_lshrrev_b32_e32 v31, 22, v43
	v_lshrrev_b32_e32 v32, 26, v43
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v9, v13, v11 :: v_dual_and_b32 v26, 60, v26
	v_and_b32_e32 v27, 60, v27
	v_and_b32_e32 v28, 60, v28
	v_and_b32_e32 v29, 60, v29
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v9, v14, v7
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
	v_and_b32_e32 v30, 60, v30
	v_and_b32_e32 v31, 60, v31
	v_and_b32_e32 v32, 60, v32
	ds_load_b32 v27, v27
	ds_load_b32 v28, v28
	ds_load_b32 v29, v29
	ds_load_b32 v30, v30
	ds_load_b32 v31, v31
	ds_load_b32 v32, v32
	v_lshlrev_b32_e32 v14, 16, v1
	v_and_b32_e32 v43, 15, v44
	v_lshrrev_b32_e32 v46, 2, v44
	v_lshrrev_b32_e32 v10, 6, v44
	v_lshrrev_b32_e32 v13, 18, v44
	v_and_b32_e32 v1, 0xffff0000, v1
	v_lshlrev_b32_e32 v43, 2, v43
	s_wait_dscnt 0x7
	v_fma_f32 v25, v25, v45, 0
	v_and_b32_e32 v45, 60, v46
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v25, v26, v5
	v_and_b32_e32 v5, 60, v10
	ds_load_b32 v12, v43
	ds_load_b32 v26, v45
	ds_load_b32 v5, v5
	s_wait_dscnt 0x8
	v_fmac_f32_e32 v25, v27, v47
	v_lshlrev_b32_e32 v27, 16, v8
	v_and_b32_e32 v8, 0xffff0000, v8
	v_lshrrev_b32_e32 v10, 10, v44
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v9, v15, v27
	v_fmac_f32_e32 v9, v16, v8
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v25, v28, v6
	v_lshrrev_b32_e32 v6, 14, v44
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v9, v17, v14
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v25, v29, v11
	v_and_b32_e32 v11, 60, v13
	v_lshrrev_b32_e32 v13, 26, v44
	v_dual_fmac_f32 v9, v18, v1 :: v_dual_and_b32 v6, 60, v6
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v25, v30, v7
	v_lshrrev_b32_e32 v7, 22, v44
	v_and_b32_e32 v13, 60, v13
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v25, v31, v27 :: v_dual_and_b32 v10, 60, v10
	v_and_b32_e32 v7, 60, v7
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v25, v32, v8
	ds_load_b32 v8, v10
	ds_load_b32 v6, v6
	ds_load_b32 v10, v11
	ds_load_b32 v7, v7
	ds_load_b32 v11, v13
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v25, v12, v14
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v25, v26, v1 :: v_dual_lshlrev_b32 v12, 16, v2
	v_and_b32_e32 v1, 0xffff0000, v2
	v_dual_fmac_f32 v9, v19, v12 :: v_dual_lshlrev_b32 v2, 16, v3
	v_and_b32_e32 v3, 0xffff0000, v3
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v25, v5, v12
	v_and_b32_e32 v5, 7, v64
	v_fmac_f32_e32 v9, v20, v1
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v25, v8, v1
	v_lshlrev_b32_e32 v1, 16, v4
	v_dual_fmac_f32 v9, v21, v2 :: v_dual_and_b32 v4, 0xffff0000, v4
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v25, v6, v2 :: v_dual_and_b32 v8, 0x7f, v63
	v_and_b32_e32 v2, 7, v63
	v_and_b32_e32 v6, 0x7f, v64
	v_cvt_f32_ubyte0_e32 v5, v5
	v_fmac_f32_e32 v9, v22, v3
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v25, v10, v3
	v_cvt_f32_ubyte0_e32 v2, v2
	v_lshl_add_u32 v3, v6, 20, 0x3c000000
	v_mul_f32_e32 v5, 0x3b000000, v5
	v_cmp_lt_u32_e32 vcc_lo, 7, v6
	v_lshl_add_u32 v10, v8, 20, 0x3c000000
	v_dual_mul_f32 v2, 0x3b000000, v2 :: v_dual_fmac_f32 v9, v23, v1
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v25, v7, v1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v5, v3, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, 7, v8
	v_lshlrev_b32_e32 v5, 24, v64
	v_fmac_f32_e32 v9, v24, v4
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_dual_fmac_f32 v25, v11, v4 :: v_dual_cndmask_b32 v2, v2, v10
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v6
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v1, 0x7fc00000, v3, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f, v8
	v_lshlrev_b32_e32 v3, 24, v63
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_or_b32 v1, 0x80000000, v5, v1
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v2, 0x7fc00000, v2 :: v_dual_fmac_f32 v61, v25, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v2, 0x80000000, v3, v2
	v_fmac_f32_e32 v62, v9, v2
.LBB_500:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s9, exec_lo
	v_cmpx_gt_i32_e64 s3, v60
	s_cbranch_execz .LBB_504
; %bb.501:
	v_and_b32_e32 v18, 31, v0
	s_mov_b32 s10, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v0, 5, v18
	v_lshlrev_b32_e32 v1, 3, v18
	v_add_co_u32 v0, s0, s6, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v2, null, s7, 0, s0
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v3, vcc_lo, v41, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, 0, v42, vcc_lo
	v_add_co_u32 v8, vcc_lo, 0x101c, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, 0, v2, vcc_lo
	v_add_co_u32 v0, vcc_lo, v39, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v40, vcc_lo
	v_add_co_u32 v2, vcc_lo, s18, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s19, v4, vcc_lo
	v_add_co_u32 v0, vcc_lo, s18, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s19, v1, vcc_lo
	v_add_co_u32 v10, vcc_lo, 0x400, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v11, null, 0, v3, vcc_lo
	v_add_co_u32 v12, vcc_lo, 0x400, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v13, null, 0, v1, vcc_lo
	v_add_co_u32 v0, vcc_lo, s20, v35
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s21, v36, vcc_lo
	v_add_co_u32 v2, vcc_lo, s20, v37
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s21, v38, vcc_lo
	v_add_co_u32 v19, vcc_lo, 0x80, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v20, null, 0, v1, vcc_lo
	v_add_co_u32 v21, vcc_lo, 0x80, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v22, null, 0, v3, vcc_lo
	s_mov_b64 s[6:7], 0x80
.LBB_502:                              ; =>This Inner Loop Header: Depth=1
	global_load_b64 v[16:17], v[10:11], off th:TH_LOAD_NT
	global_load_b64 v[14:15], v[12:13], off th:TH_LOAD_NT
	s_clause 0x1
	global_load_b128 v[4:7], v[8:9], off offset:-28
	global_load_b128 v[0:3], v[8:9], off offset:-12
	v_add_co_u32 v23, vcc_lo, v21, v18
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v24, null, 0, v22, vcc_lo
	v_add_co_u32 v25, vcc_lo, v19, v18
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v26, null, 0, v20, vcc_lo
	global_load_i8 v23, v[23:24], off
	global_load_i8 v24, v[25:26], off
	v_add_co_u32 v8, vcc_lo, 0x400, v8
	s_wait_alu 0xfffe
	s_add_nc_u64 s[6:7], s[6:7], 32
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, 0, v9, vcc_lo
	v_add_co_u32 v10, vcc_lo, 0x100, v10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v11, null, 0, v11, vcc_lo
	v_add_co_u32 v12, vcc_lo, 0x100, v12
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v13, null, 0, v13, vcc_lo
	v_add_co_u32 v19, vcc_lo, v19, 32
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v20, null, 0, v20, vcc_lo
	v_add_co_u32 v21, vcc_lo, v21, 32
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v22, null, 0, v22, vcc_lo
	s_wait_loadcnt 0x5
	v_and_b32_e32 v25, 15, v16
	s_wait_loadcnt 0x4
	v_and_b32_e32 v41, 15, v14
	v_lshrrev_b32_e32 v26, 2, v16
	v_lshrrev_b32_e32 v42, 2, v14
	v_lshrrev_b32_e32 v43, 6, v14
	v_lshlrev_b32_e32 v25, 2, v25
	v_lshlrev_b32_e32 v41, 2, v41
	v_and_b32_e32 v26, 60, v26
	v_and_b32_e32 v42, 60, v42
	v_lshrrev_b32_e32 v27, 6, v16
	ds_load_b32 v25, v25
	ds_load_b32 v41, v41
	ds_load_b32 v26, v26
	ds_load_b32 v42, v42
	v_lshrrev_b32_e32 v28, 10, v16
	v_lshrrev_b32_e32 v29, 14, v16
	v_lshrrev_b32_e32 v30, 18, v16
	v_lshrrev_b32_e32 v31, 22, v16
	v_lshrrev_b32_e32 v16, 26, v16
	v_lshrrev_b32_e32 v44, 10, v14
	v_lshrrev_b32_e32 v45, 14, v14
	v_lshrrev_b32_e32 v46, 18, v14
	v_lshrrev_b32_e32 v47, 22, v14
	v_lshrrev_b32_e32 v14, 26, v14
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v55, 16, v4
	v_and_b32_e32 v43, 60, v43
	v_and_b32_e32 v27, 60, v27
	v_and_b32_e32 v28, 60, v28
	v_and_b32_e32 v29, 60, v29
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v63, 16, v1
	v_and_b32_e32 v1, 0xffff0000, v1
	v_and_b32_e32 v30, 60, v30
	v_and_b32_e32 v31, 60, v31
	v_and_b32_e32 v16, 60, v16
	v_and_b32_e32 v44, 60, v44
	v_and_b32_e32 v45, 60, v45
	v_and_b32_e32 v46, 60, v46
	v_and_b32_e32 v47, 60, v47
	v_and_b32_e32 v14, 60, v14
	v_lshlrev_b32_e32 v57, 16, v6
	ds_load_b32 v43, v43
	ds_load_b32 v27, v27
	ds_load_b32 v28, v28
	ds_load_b32 v29, v29
	ds_load_b32 v30, v30
	ds_load_b32 v31, v31
	ds_load_b32 v16, v16
	ds_load_b32 v44, v44
	ds_load_b32 v45, v45
	ds_load_b32 v46, v46
	ds_load_b32 v47, v47
	ds_load_b32 v14, v14
	v_and_b32_e32 v6, 0xffff0000, v6
	v_lshlrev_b32_e32 v56, 16, v5
	v_and_b32_e32 v5, 0xffff0000, v5
	v_and_b32_e32 v4, 0xffff0000, v4
	s_wait_dscnt 0xf
	v_fma_f32 v25, v25, v55, 0
	v_and_b32_e32 v32, 15, v17
	s_wait_dscnt 0xe
	v_fma_f32 v41, v41, v55, 0
	v_lshrrev_b32_e32 v35, 2, v17
	v_lshrrev_b32_e32 v36, 6, v17
	s_wait_dscnt 0xd
	v_fmac_f32_e32 v25, v26, v4
	v_lshrrev_b32_e32 v37, 10, v17
	v_lshrrev_b32_e32 v38, 14, v17
	v_and_b32_e32 v35, 60, v35
	v_and_b32_e32 v36, 60, v36
	s_wait_dscnt 0xa
	v_fmac_f32_e32 v25, v27, v56
	v_lshrrev_b32_e32 v49, 2, v15
	v_and_b32_e32 v37, 60, v37
	v_lshrrev_b32_e32 v39, 18, v17
	v_lshrrev_b32_e32 v40, 22, v17
	s_wait_dscnt 0x9
	v_dual_fmac_f32 v25, v28, v5 :: v_dual_and_b32 v48, 15, v15
	v_dual_fmac_f32 v41, v42, v4 :: v_dual_lshlrev_b32 v64, 16, v2
	v_and_b32_e32 v49, 60, v49
	s_wait_dscnt 0x8
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v25, v29, v57 :: v_dual_lshlrev_b32 v48, 2, v48
	v_lshrrev_b32_e32 v50, 6, v15
	v_and_b32_e32 v39, 60, v39
	v_lshrrev_b32_e32 v17, 26, v17
	s_wait_dscnt 0x7
	v_dual_fmac_f32 v25, v30, v6 :: v_dual_lshlrev_b32 v58, 16, v7
	v_fmac_f32_e32 v41, v43, v56
	v_lshlrev_b32_e32 v60, 16, v0
	v_and_b32_e32 v7, 0xffff0000, v7
	v_lshrrev_b32_e32 v51, 10, v15
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v25, v31, v58
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v41, v44, v5 :: v_dual_and_b32 v0, 0xffff0000, v0
	v_and_b32_e32 v2, 0xffff0000, v2
	v_lshrrev_b32_e32 v52, 14, v15
	v_fmac_f32_e32 v25, v16, v7
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v41, v45, v57
	v_and_b32_e32 v17, 60, v17
	v_and_b32_e32 v4, 60, v51
	v_lshrrev_b32_e32 v53, 18, v15
	v_lshrrev_b32_e32 v54, 22, v15
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v41, v46, v6 :: v_dual_lshlrev_b32 v32, 2, v32
	v_lshrrev_b32_e32 v15, 26, v15
	v_lshlrev_b32_e32 v65, 16, v3
	v_and_b32_e32 v27, 60, v54
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v41, v47, v58
	ds_load_b32 v32, v32
	ds_load_b32 v5, v35
	ds_load_b32 v28, v36
	ds_load_b32 v35, v48
	v_and_b32_e32 v15, 60, v15
	s_wait_loadcnt 0x0
	v_and_b32_e32 v31, 0x7f, v24
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v41, v14, v7 :: v_dual_and_b32 v38, 60, v38
	v_and_b32_e32 v3, 0xffff0000, v3
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s1, 7, v31
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v25, v32, v60
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v41, v35, v60
	ds_load_b32 v29, v37
	ds_load_b32 v36, v38
	ds_load_b32 v37, v49
	v_fmac_f32_e32 v25, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v25, v28, v63
	v_lshl_add_u32 v28, v31, 20, 0x3c000000
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v25, v29, v1 :: v_dual_and_b32 v50, 60, v50
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v41, v37, v0
	v_and_b32_e32 v40, 60, v40
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v25, v36, v64
	ds_load_b32 v6, v39
	ds_load_b32 v30, v40
	ds_load_b32 v38, v50
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v25, v6, v2 :: v_dual_and_b32 v26, 60, v52
	ds_load_b32 v17, v17
	ds_load_b32 v4, v4
	ds_load_b32 v26, v26
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v41, v38, v63
	v_dual_fmac_f32 v25, v30, v65 :: v_dual_and_b32 v16, 0x7f, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v5, v16, 20, 0x3c000000
	v_cmp_lt_u32_e64 s0, 7, v16
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v25, v17, v3
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v41, v4, v1
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v41, v26, v64 :: v_dual_and_b32 v42, 60, v53
	ds_load_b32 v7, v42
	ds_load_b32 v14, v27
	ds_load_b32 v15, v15
	v_and_b32_e32 v27, 7, v23
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v27, 0x3b000000, v27
	s_wait_alu 0xf1ff
	v_cndmask_b32_e64 v1, v27, v5, s0
	v_cmp_ne_u32_e64 s0, 0x7f, v31
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v41, v7, v2 :: v_dual_and_b32 v0, 7, v24
	s_wait_dscnt 0x1
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v41, v14, v65 :: v_dual_add_nc_u32 v2, s6, v18
	v_cvt_f32_ubyte0_e32 v0, v0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v41, v15, v3 :: v_dual_mul_f32 v0, 0x3b000000, v0
	v_cndmask_b32_e64 v0, v0, v28, s1
	v_cmp_ne_u32_e64 s1, 0x7f, v16
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v0, 0x7fc00000, v0, s0
	v_cndmask_b32_e64 v1, 0x7fc00000, v1, s1
	v_cmp_le_i32_e64 s0, s3, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_or_b32 v0, 0x80000000, v24, v0
	v_and_or_b32 v1, 0x80000000, v23, v1
	s_or_b32 s10, s0, s10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v62, v25, v0
	v_fmac_f32_e32 v61, v41, v1
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s10
	s_cbranch_execnz .LBB_502
; %bb.503:
	s_or_b32 exec_lo, exec_lo, s10
.LBB_504:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s9
	s_mov_b32 s1, 0
	s_mul_i32 s0, s5, 3
	v_xor_b32_e32 v3, 8, v59
	v_mov_b32_e32 v0, 0
	s_wait_alu 0xfffe
	s_lshl_b64 s[0:1], s[0:1], 2
	v_xor_b32_e32 v5, 4, v59
	s_wait_alu 0xfffe
	s_add_nc_u64 s[0:1], s[26:27], s[0:1]
	v_xor_b32_e32 v6, 2, v59
	global_load_b32 v1, v0, s[0:1]
	v_xor_b32_e32 v0, 16, v59
	v_xor_b32_e32 v7, 1, v59
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v0
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v0, v59, v0, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v3
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v59, v3, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v5
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v5, v59, v5 :: v_dual_lshlrev_b32 v0, 2, v0
	ds_bpermute_b32 v2, v0, v62
	v_cmp_gt_i32_e32 vcc_lo, 32, v6
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v6, v59, v6, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v7
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_dual_cndmask_b32 v7, v59, v7 :: v_dual_add_f32 v4, v62, v2
	v_lshlrev_b32_e32 v2, 2, v3
	ds_bpermute_b32 v3, v2, v4
	s_wait_dscnt 0x0
	v_dual_add_f32 v4, v4, v3 :: v_dual_lshlrev_b32 v3, 2, v5
	ds_bpermute_b32 v5, v3, v4
	s_wait_dscnt 0x0
	v_dual_add_f32 v5, v4, v5 :: v_dual_lshlrev_b32 v4, 2, v6
	ds_bpermute_b32 v6, v4, v5
	s_wait_dscnt 0x0
	v_dual_add_f32 v6, v5, v6 :: v_dual_lshlrev_b32 v5, 2, v7
	ds_bpermute_b32 v7, v5, v6
	s_wait_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v6, v1, v6
	v_and_b32_e32 v1, 0x7fffffff, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
; %bb.505:
	v_bfe_u32 v1, v6, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v1, v6, v1, 0x7fff
                                        ; implicit-def: $vgpr6
	v_lshrrev_b32_e32 v1, 16, v1
; %bb.506:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s3, s3
; %bb.507:
	v_lshrrev_b32_e32 v1, 16, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v1, 64, v1
; %bb.508:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	v_mov_b32_e32 v6, 0
	ds_bpermute_b32 v0, v0, v61
	global_load_b32 v6, v6, s[0:1] offset:4
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v61, v0
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
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v2, v6, v0
	v_and_b32_e32 v0, 0x7fffffff, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 0x7f800001, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s0, exec_lo, s0
; %bb.509:
	v_bfe_u32 v0, v2, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v0, v2, v0, 0x7fff
                                        ; implicit-def: $vgpr2
	v_lshrrev_b32_e32 v0, 16, v0
; %bb.510:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.511:
	v_lshrrev_b32_e32 v0, 16, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v0, 64, v0
; %bb.512:
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
	s_cbranch_execz .LBB_515
; %bb.513:
	v_bfe_u32 v1, v2, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v1, v2, v1, 0x7fff
                                        ; implicit-def: $vgpr2
	v_lshrrev_b32_e32 v1, 16, v1
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
	s_cbranch_execnz .LBB_516
.LBB_514:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_eq_u32_e32 vcc_lo, 0, v33
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execnz .LBB_517
	s_branch .LBB_522
.LBB_515:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
	s_cbranch_execz .LBB_514
.LBB_516:
	v_lshrrev_b32_e32 v1, 16, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v1, 64, v1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_eq_u32_e32 vcc_lo, 0, v33
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB_522
.LBB_517:
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
; %bb.518:
	v_bfe_u32 v0, v1, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v0, v1, v0, 0x7fff
                                        ; implicit-def: $vgpr1
	v_lshrrev_b32_e32 v0, 16, v0
; %bb.519:
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s0, s0
; %bb.520:
	v_lshrrev_b32_e32 v0, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v0, 64, v0
; %bb.521:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s0
	v_ashrrev_i32_e32 v35, 31, v34
	s_ashr_i32 s5, s4, 31
	s_wait_alu 0xfffe
	s_mul_u64 s[0:1], s[4:5], s[54:55]
	s_wait_alu 0xfffe
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshlrev_b64_e32 v[1:2], 1, v[34:35]
	s_wait_alu 0xfffe
	s_add_nc_u64 s[0:1], s[62:63], s[0:1]
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v1, vcc_lo, s0, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s1, v2, vcc_lo
	global_store_b16 v[1:2], v0, off
.LBB_522:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	global_inv scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB_525
; %bb.523:
	s_mov_b32 s0, exec_lo
	s_wait_alu 0xfffe
	v_mbcnt_lo_u32_b32 v0, s0, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_and_b32 s1, exec_lo, vcc_lo
	s_wait_alu 0xfffe
	s_mov_b32 exec_lo, s1
	s_cbranch_execz .LBB_525
; %bb.524:
	s_bcnt1_i32_b32 s0, s0
	s_wait_alu 0xfffe
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s0
	s_wait_loadcnt 0x0
	global_atomic_add_u32 v0, v1, s[34:35] scope:SCOPE_DEV
.LBB_525:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.LBB_526:
	s_mov_b32 s1, -1
	s_cmp_lt_i32 s0, 1
	s_cbranch_scc0 .LBB_468
.LBB_527:
	s_mov_b32 s5, -1
	s_cmp_lt_i32 s0, 2
	s_mov_b32 s8, -1
	s_cbranch_scc1 .LBB_471
.LBB_528:
	v_xor_b32_e32 v13, 16, v59
	v_max_u32_e32 v12, v2, v3
	v_xor_b32_e32 v14, 8, v59
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	v_max3_u32 v12, v12, v1, v8
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v59, v13, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_max3_u32 v12, v12, v7, v10
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_max3_u32 v12, v12, v9, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v14, 2, v14
	v_lshlrev_b32_e32 v13, 2, v13
	ds_bpermute_b32 v13, v13, v12
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 4, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_lshlrev_b32_e32 v14, 2, v14
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 2, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_lshlrev_b32_e32 v14, 2, v14
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 1, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	v_lshlrev_b32_e32 v13, 2, v14
	ds_bpermute_b32 v13, v13, v12
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s8, v12
	s_wait_alu 0xfffe
	s_and_not1_b32 s9, 31, s8
	s_and_not1_b32 s8, 0xffff, s8
	s_wait_alu 0xfffe
	v_cmp_eq_u32_e32 vcc_lo, s9, v33
	s_and_saveexec_b32 s9, vcc_lo
	s_cbranch_execz .LBB_530
; %bb.529:
	s_lshr_b32 s11, s8, 5
	s_cmp_gt_u32 s8, 31
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	s_cmp_lg_u32 s11, 1
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s11, 2
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s11, 3
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s11, 4
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s11, 5
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s11, 6
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v10, 0, v10, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s11, 7
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v9, 0, v9, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v11, 0, v11, vcc_lo
.LBB_530:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s9
	s_cmp_lt_i32 s0, 3
	s_mov_b32 s9, -1
	s_cbranch_scc0 .LBB_472
.LBB_531:
	s_cmp_lt_i32 s0, 4
	s_mov_b32 s11, -1
	s_cbranch_scc1 .LBB_475
.LBB_532:
	v_xor_b32_e32 v13, 16, v59
	v_max_u32_e32 v12, v2, v3
	v_xor_b32_e32 v14, 8, v59
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	v_max3_u32 v12, v12, v1, v8
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v59, v13, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_max3_u32 v12, v12, v7, v10
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_max3_u32 v12, v12, v9, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v14, 2, v14
	v_lshlrev_b32_e32 v13, 2, v13
	ds_bpermute_b32 v13, v13, v12
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 4, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_lshlrev_b32_e32 v14, 2, v14
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 2, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_lshlrev_b32_e32 v14, 2, v14
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 1, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	v_lshlrev_b32_e32 v13, 2, v14
	ds_bpermute_b32 v13, v13, v12
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s11, v12
	s_wait_alu 0xfffe
	s_and_not1_b32 s14, 31, s11
	s_and_not1_b32 s11, 0xffff, s11
	v_cmp_eq_u32_e32 vcc_lo, s14, v33
	s_and_saveexec_b32 s14, vcc_lo
	s_cbranch_execz .LBB_534
; %bb.533:
	s_wait_alu 0xfffe
	s_lshr_b32 s15, s11, 5
	s_cmp_gt_u32 s11, 31
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	s_cmp_lg_u32 s15, 1
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s15, 2
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s15, 3
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s15, 4
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s15, 5
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s15, 6
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v10, 0, v10, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s15, 7
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v9, 0, v9, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v11, 0, v11, vcc_lo
.LBB_534:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s14
	s_cmp_lt_i32 s0, 5
	s_mov_b32 s14, -1
	s_cbranch_scc0 .LBB_476
.LBB_535:
	s_cmp_lt_i32 s0, 6
	s_mov_b32 s15, -1
	s_cbranch_scc1 .LBB_479
.LBB_536:
	v_xor_b32_e32 v13, 16, v59
	v_max_u32_e32 v12, v2, v3
	v_xor_b32_e32 v14, 8, v59
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e32 vcc_lo, 32, v13
	v_max3_u32 v12, v12, v1, v8
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v13, v59, v13, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_max3_u32 v12, v12, v7, v10
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_max3_u32 v12, v12, v9, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v14, 2, v14
	v_lshlrev_b32_e32 v13, 2, v13
	ds_bpermute_b32 v13, v13, v12
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 4, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_lshlrev_b32_e32 v14, 2, v14
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 2, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	v_lshlrev_b32_e32 v14, 2, v14
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	ds_bpermute_b32 v13, v14, v12
	v_xor_b32_e32 v14, 1, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v14
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v14, v59, v14, vcc_lo
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	v_lshlrev_b32_e32 v13, 2, v14
	ds_bpermute_b32 v13, v13, v12
	s_wait_dscnt 0x0
	v_max_u32_e32 v12, v12, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s15, v12
	s_wait_alu 0xfffe
	s_and_not1_b32 s16, 31, s15
	s_and_not1_b32 s15, 0xffff, s15
	v_cmp_eq_u32_e32 vcc_lo, s16, v33
	s_and_saveexec_b32 s16, vcc_lo
	s_cbranch_execz .LBB_538
; %bb.537:
	s_wait_alu 0xfffe
	s_lshr_b32 s17, s15, 5
	s_cmp_gt_u32 s15, 31
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	s_cmp_lg_u32 s17, 1
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s17, 2
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s17, 3
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s17, 4
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s17, 5
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s17, 6
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v10, 0, v10, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_lg_u32 s17, 7
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v9, 0, v9, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu 0xfffe
	v_cndmask_b32_e32 v11, 0, v11, vcc_lo
.LBB_538:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s16
	s_cmp_lt_i32 s0, 7
	s_mov_b32 s16, -1
	s_cbranch_scc0 .LBB_480
	s_branch .LBB_481
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN12_GLOBAL__N_116moe_block_kernelILi4ELi8EEEvNS_9BlockArgsE
		.amdhsa_group_segment_fixed_size 68
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 208
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
		.amdhsa_next_free_vgpr 120
		.amdhsa_next_free_sgpr 97
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 255
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN12_GLOBAL__N_116moe_block_kernelILi4ELi8EEEvNS_9BlockArgsE,"axG",@progbits,_ZN12_GLOBAL__N_116moe_block_kernelILi4ELi8EEEvNS_9BlockArgsE,comdat
.Lfunc_end4:
