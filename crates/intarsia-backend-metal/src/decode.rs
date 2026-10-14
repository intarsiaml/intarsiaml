//! The Qwen3-8B decode-shaped GEMV sequence with synthetic weights, direct and indirect-command-buffer encodings
//! (spike parts c and d). Per layer: q 4096->4096, k 4096->1024, v 4096->1024, o 4096->4096, gate 4096->12288,
//! up 4096->12288, down 12288->4096; then lm_head 4096->151936. Dims from ~/models/Qwen3-8B/config.json (also the fixture
//! in intarsia-cli's accuracy/envelope.rs): hidden 4096, 32 heads / 8 kv heads x head_dim 128, intermediate 12288, vocab
//! 151936, 36 layers, untied lm_head. (recipes/qwen3.toml names tensors only, no dims.)

use crate::bench::fill_bf16_random;
use crate::kernels::GemvVariant;
use crate::{encode_gemv, GemvParams, MetalBackend, MtlBuffer};
use intarsia_core::{Error, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor,
    MTLIndirectCommandType, MTLIndirectComputeCommand, MTLResource, MTLResourceOptions, MTLResourceUsage, MTLSize,
};
use std::ptr::NonNull;

pub const HIDDEN: usize = 4096;
pub const KV: usize = 1024;
pub const FFN: usize = 12288;
pub const VOCAB: usize = 151936;
pub const LAYERS: usize = 36;

/// (name, K, N) of the per-layer GEMVs, in execution order.
pub const LAYER_OPS: [(&str, usize, usize); 7] = [
    ("q", HIDDEN, HIDDEN),
    ("k", HIDDEN, KV),
    ("v", HIDDEN, KV),
    ("o", HIDDEN, HIDDEN),
    ("gate", HIDDEN, FFN),
    ("up", HIDDEN, FFN),
    ("down", FFN, HIDDEN),
];

pub struct GemvOp {
    pub name: &'static str,
    pub k: usize,
    pub n: usize,
}

/// One synthetic-weight decode step: `ops[i]` multiplies `w[i]` by the shared input `x_for(k)` into `y[i]`.
pub struct DecodeSeq {
    pub ops: Vec<GemvOp>,
    pub w: Vec<MtlBuffer>,
    pub y: Vec<MtlBuffer>,
    /// Inputs: `x[0]` has 4096 elements, `x[1]` has 12288.
    pub x: [MtlBuffer; 2],
}

impl DecodeSeq {
    pub fn x_for(&self, k: usize) -> &MtlBuffer {
        if k == FFN {
            &self.x[1]
        } else {
            &self.x[0]
        }
    }

    /// Bytes one decode step reads and writes (weights + x + y per GEMV).
    pub fn step_bytes(&self) -> u64 {
        self.ops.iter().map(|o| (o.n * o.k * 2 + o.k * 2 + o.n * 2) as u64).sum()
    }

    pub fn weight_bytes(&self) -> u64 {
        self.ops.iter().map(|o| (o.n * o.k * 2) as u64).sum()
    }

    /// Build the sequence for `layers` layers (+ lm_head when `lm_head`), weights random bf16 from a shared noise pool
    /// (distinct buffers, repeating content), outputs zeroed.
    pub fn qwen3_8b(be: &MetalBackend, layers: usize, lm_head: bool) -> Result<DecodeSeq> {
        let mut ops: Vec<GemvOp> = Vec::new();
        for _ in 0..layers {
            ops.extend(LAYER_OPS.iter().map(|&(name, k, n)| GemvOp { name, k, n }));
        }
        if lm_head {
            ops.push(GemvOp { name: "lm_head", k: HIDDEN, n: VOCAB });
        }
        // 256 MiB of noise reused for every weight buffer (content repeats, memory does not)
        let pool_elems = 128usize << 20;
        let mut pool = vec![0u16; pool_elems];
        // SAFETY: pool is `pool_elems` u16 long.
        unsafe { fill_bf16_random(pool.as_mut_ptr(), pool_elems, 0x1234_5678_9abc_def1) };
        // hazard tracking off by default (the sequence orders itself); IK_METAL_TRACKED=1 measures the default mode
        let untracked = std::env::var_os("IK_METAL_TRACKED").is_none();
        let mut w = Vec::new();
        let mut y = Vec::new();
        for (i, o) in ops.iter().enumerate() {
            let buf = be.new_mtl_buffer(o.n * o.k * 2, untracked)?;
            // SAFETY: freshly allocated, o.n*o.k*2 bytes.
            unsafe {
                let dst = std::slice::from_raw_parts_mut(buf.contents().as_ptr() as *mut u16, o.n * o.k);
                let off = (i * 977) % pool_elems;
                for (j, chunk) in dst.chunks_mut(pool_elems - off).enumerate() {
                    let start = if j == 0 { off } else { 0 };
                    chunk.copy_from_slice(&pool[start..start + chunk.len()]);
                }
            }
            w.push(buf);
            let yb = be.new_mtl_buffer(o.n * 2, untracked)?;
            // SAFETY: n*2 bytes.
            unsafe { std::ptr::write_bytes(yb.contents().as_ptr() as *mut u8, 0, o.n * 2) };
            y.push(yb);
        }
        let mk_x = |k: usize, seed: u64| -> Result<MtlBuffer> {
            let b = be.new_mtl_buffer(k * 2, untracked)?;
            // SAFETY: k*2 bytes.
            unsafe { fill_bf16_random(b.contents().as_ptr() as *mut u16, k, seed) };
            Ok(b)
        };
        Ok(DecodeSeq { ops, w, y, x: [mk_x(HIDDEN, 11)?, mk_x(FFN, 12)?] })
    }

    pub fn clear_outputs(&self) {
        for (o, y) in self.ops.iter().zip(&self.y) {
            // SAFETY: n*2 bytes, GPU idle (callers wait before clearing).
            unsafe { std::ptr::write_bytes(y.contents().as_ptr() as *mut u8, 0, o.n * 2) };
        }
    }

    /// Output bytes of every op, concatenated.
    pub fn read_outputs(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (o, y) in self.ops.iter().zip(&self.y) {
            // SAFETY: n*2 bytes written by the kernel, GPU idle.
            out.extend_from_slice(unsafe { std::slice::from_raw_parts(y.contents().as_ptr() as *const u8, o.n * 2) });
        }
        out
    }

    /// Encode every GEMV directly (kernel arguments via `setBytes`). `pick` chooses the variant per (K, N).
    pub fn encode_direct(
        &self,
        be: &MetalBackend,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        pick: &dyn Fn(usize, usize) -> GemvVariant,
    ) {
        for (i, o) in self.ops.iter().enumerate() {
            let v = pick(o.k, o.n);
            let params = GemvParams { k: o.k as u32, n: o.n as u32, ..Default::default() };
            encode_gemv(
                enc,
                be.pso(v.name).unwrap(),
                (&self.w[i], 0),
                (self.x_for(o.k), 0),
                (&self.y[i], 0),
                &params,
                (o.n as u32).div_ceil(v.rows_per_tg()),
                v.threads_per_tg(),
            );
        }
    }
}

/// The sequence recorded into an `MTLIndirectCommandBuffer`.
pub struct Icb {
    pub icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    /// Kernel arguments, one 256-byte slot per command (ICB commands cannot use `setBytes`).
    params: MtlBuffer,
    /// Every resource the commands touch, for `useResources`.
    resources: Vec<NonNull<ProtocolObject<dyn MTLResource>>>,
    pub count: usize,
}

const PARAM_STRIDE: usize = 256;

impl Icb {
    /// Record `seq` into an ICB. `barrier`: each command waits for the previous one (the dependency pattern of a
    /// real decode step; without it the commands may overlap).
    pub fn build(
        be: &MetalBackend,
        seq: &DecodeSeq,
        pick: &dyn Fn(usize, usize) -> GemvVariant,
        barrier: bool,
    ) -> Result<Icb> {
        let count = seq.ops.len();
        let desc = MTLIndirectCommandBufferDescriptor::new();
        desc.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatch);
        desc.setInheritBuffers(false);
        desc.setInheritPipelineState(false);
        desc.setMaxKernelBufferBindCount(4);
        // SAFETY: descriptor and count are valid.
        let icb = unsafe {
            be.device().newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                &desc,
                count,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| Error::backend("metal: newIndirectCommandBuffer failed"))?;
        let params = be.new_mtl_buffer(count * PARAM_STRIDE, true)?;
        let mut resources: Vec<NonNull<ProtocolObject<dyn MTLResource>>> = Vec::new();
        let mut add = |b: &MtlBuffer| {
            let p = NonNull::from(ProtocolObject::from_ref(&**b));
            if !resources.contains(&p) {
                resources.push(p);
            }
        };
        add(&params);
        for (i, o) in seq.ops.iter().enumerate() {
            let v = pick(o.k, o.n);
            let p = GemvParams { k: o.k as u32, n: o.n as u32, ..Default::default() };
            // SAFETY: slot i lies inside `params`; GemvParams is repr(C), 32 bytes.
            unsafe { *((params.contents().as_ptr() as *mut u8).add(i * PARAM_STRIDE) as *mut GemvParams) = p };
            // SAFETY: i < count.
            let cmd = unsafe { icb.indirectComputeCommandAtIndex(i) };
            cmd.setComputePipelineState(be.pso(v.name).unwrap());
            let xb = seq.x_for(o.k);
            // SAFETY: indices < maxKernelBufferBindCount; offsets inside the buffers.
            unsafe {
                cmd.setKernelBuffer_offset_atIndex(&seq.w[i], 0, 0);
                cmd.setKernelBuffer_offset_atIndex(xb, 0, 1);
                cmd.setKernelBuffer_offset_atIndex(&seq.y[i], 0, 2);
                cmd.setKernelBuffer_offset_atIndex(&params, i * PARAM_STRIDE, 3);
            }
            cmd.concurrentDispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: o.n.div_ceil(v.rows_per_tg() as usize), height: 1, depth: 1 },
                MTLSize { width: v.threads_per_tg() as usize, height: 1, depth: 1 },
            );
            if barrier {
                cmd.setBarrier();
            }
            add(&seq.w[i]);
            add(xb);
            add(&seq.y[i]);
        }
        Ok(Icb { icb, params, resources, count })
    }

    /// Replay: make the resources resident and execute all commands.
    pub fn execute(&self, enc: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        // SAFETY: `resources` points at buffers kept alive by `seq`/`self.params` for as long as the caller holds
        // both; usage Read|Write covers weights (read), x (read) and y (write).
        unsafe {
            enc.useResources_count_usage(
                NonNull::new_unchecked(self.resources.as_ptr() as *mut NonNull<ProtocolObject<dyn MTLResource>>),
                self.resources.len(),
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
            enc.executeCommandsInBuffer_withRange(&self.icb, NSRange { location: 0, length: self.count });
        }
        let _ = &self.params;
    }
}
