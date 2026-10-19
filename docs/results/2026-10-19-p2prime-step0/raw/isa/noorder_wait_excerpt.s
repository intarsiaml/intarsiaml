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
