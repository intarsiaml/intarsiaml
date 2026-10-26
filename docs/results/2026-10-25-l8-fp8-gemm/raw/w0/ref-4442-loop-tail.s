; Lane l8: end of the 256x128 (4,4,4,2) f8_gemm_kernel main loop on main (hipcc 7.2.4, gfx1201): the last WMMAs of the stage, then the barrier.
; Note s_wait_loadcnt 0x0 before s_barrier_signal and global_inv scope:SCOPE_SE after s_barrier_wait.
.LBB4_21:                               ;   in Loop: Header=BB4_22 Depth=1
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[113:114], v[101:102], v[57:64]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[113:114], v[109:110], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[117:118], v[101:102], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[117:118], v[109:110], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[121:122], v[101:102], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[121:122], v[109:110], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[105:106], v[101:102], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[105:106], v[109:110], v[1:8]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[115:116], v[103:104], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[115:116], v[111:112], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[119:120], v[103:104], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[119:120], v[111:112], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[123:124], v[103:104], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[123:124], v[111:112], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[107:108], v[103:104], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[107:108], v[111:112], v[1:8]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[97:98], v[77:78], v[57:64]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[97:98], v[93:94], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[85:86], v[77:78], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[85:86], v[93:94], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[89:90], v[77:78], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[89:90], v[93:94], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[81:82], v[77:78], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[81:82], v[93:94], v[1:8]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[99:100], v[79:80], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[99:100], v[95:96], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[87:88], v[79:80], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[87:88], v[95:96], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[91:92], v[79:80], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[91:92], v[95:96], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[83:84], v[79:80], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[83:84], v[95:96], v[1:8]
	s_add_nc_u64 s[4:5], s[4:5], 64
	s_add_co_i32 s24, s24, 1
	s_wait_alu 0xfffe
	s_cmp_lg_u32 s9, s4
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc0 .LBB4_41
.LBB4_22:                               ; =>This Inner Loop Header: Depth=1
	s_add_co_i32 s8, s8, 1
	s_cmp_lt_i32 s24, s25
