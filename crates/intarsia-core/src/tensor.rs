//! Host tensors: dense, row-major, little-endian bytes plus dtype and shape.

use crate::dtype::{decode_to_f32, encode_from_f32, DType};
use crate::error::{Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

impl HostTensor {
    pub fn new(dtype: DType, shape: Vec<usize>, data: Vec<u8>) -> Result<Self> {
        let want = numel(&shape) * dtype.size_bytes();
        if data.len() != want {
            return Err(Error::invalid(format!(
                "HostTensor {dtype} {shape:?}: expected {want} bytes, got {}",
                data.len()
            )));
        }
        Ok(HostTensor { dtype, shape, data })
    }

    pub fn zeros(dtype: DType, shape: &[usize]) -> Self {
        HostTensor { dtype, shape: shape.to_vec(), data: vec![0; numel(shape) * dtype.size_bytes()] }
    }

    /// Build a tensor of `dtype` from f32 values (rounded to nearest even).
    pub fn from_f32_as(dtype: DType, shape: &[usize], vals: &[f32]) -> Self {
        assert_eq!(numel(shape), vals.len(), "shape/values mismatch");
        let mut data = vec![0u8; vals.len() * dtype.size_bytes()];
        encode_from_f32(dtype, vals, &mut data);
        HostTensor { dtype, shape: shape.to_vec(), data }
    }

    pub fn from_f32(shape: &[usize], vals: &[f32]) -> Self {
        Self::from_f32_as(DType::F32, shape, vals)
    }

    pub fn from_i32(shape: &[usize], vals: &[i32]) -> Self {
        assert_eq!(numel(shape), vals.len());
        let mut data = Vec::with_capacity(vals.len() * 4);
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        HostTensor { dtype: DType::I32, shape: shape.to_vec(), data }
    }

    pub fn numel(&self) -> usize {
        numel(&self.shape)
    }

    pub fn nbytes(&self) -> usize {
        self.data.len()
    }

    /// Decode every element to f32.
    pub fn to_f32(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.numel()];
        decode_to_f32(self.dtype, &self.data, &mut out);
        out
    }

    pub fn to_i32(&self) -> Result<Vec<i32>> {
        if self.dtype != DType::I32 {
            return Err(Error::invalid("to_i32 on non-i32 tensor"));
        }
        Ok(self.data.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

/// Row-major strides in elements.
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}
