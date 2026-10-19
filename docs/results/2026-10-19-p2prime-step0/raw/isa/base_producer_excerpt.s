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
