//! Kernel registry (ARCHITECTURE 3.2): `(op, backend, arch, dtype, shape predicate)`
//! implementations with an exactness class and a tolerance versus the reference.
//!
//! Backends return their registrations from `Backend::kernels()`, an explicit
//! list (no link-section/inventory magic, which breaks with nvcc static libs).

use crate::backend::DeviceInfo;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{Op, OpCall, ShapeInfo};
use std::collections::HashMap;

/// How a kernel relates to the reference numerically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Exactness {
    /// Best speed; run-to-run results may differ (atomics, split-K order).
    Fast,
    /// Run-to-run identical for the same shape/batch, but may differ from serial.
    Deterministic,
    /// Bit-identical to the serial reference order (lossless speculation claims).
    Exact,
}

impl Exactness {
    pub fn as_str(self) -> &'static str {
        match self {
            Exactness::Fast => "fast",
            Exactness::Deterministic => "deterministic",
            Exactness::Exact => "exact",
        }
    }
}

/// Admission tolerance versus the reference implementation. A kernel has one `Tol` for its outputs
/// (`KernelImpl::tol`) and may carry a second, usually tighter, one for f32 outputs
/// (`KernelImpl::tol_f32`, e.g. lm_head logits): the budget is picked per output tensor by dtype
/// (`KernelImpl::tol_for`), because a bf16 output cannot meet an f32 budget (bf16 rounding alone
/// is ~4e-3 relative per element).
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Tol {
    /// Maximum absolute error allowed. Relative-to-scale: see `Tol::scaled`.
    pub max_abs: f32,
    /// Maximum relative L2 error: ||a-b|| / ||b||.
    pub rel_l2: f32,
}

impl Tol {
    pub const fn new(max_abs: f32, rel_l2: f32) -> Tol {
        Tol { max_abs, rel_l2 }
    }
    pub const EXACT: Tol = Tol::new(0.0, 0.0);
}

pub type Predicate = fn(&ShapeInfo) -> bool;

/// Scratch workspace a kernel needs for a shape, in bytes (0 = none). The caller allocates it once
/// at plan time (the maximum over every shape the plan uses) and passes it as `OpCall::scratch`;
/// launchers never allocate (kernels/README.md, "Scratch workspace").
pub type ScratchFn = fn(&ShapeInfo) -> usize;

#[derive(Clone, Debug)]
pub struct KernelImpl {
    pub op: Op,
    /// Unique per backend; passed back to `Backend::launch`.
    pub name: String,
    pub backend: String,
    /// Architectures this impl is built for ("any" matches everything).
    pub archs: Vec<String>,
    /// Accepted activation dtypes (empty = any).
    pub acts: Vec<DType>,
    /// Accepted weight dtypes for matmul-like ops, or KV dtypes for attention (empty = any).
    pub weights: Vec<DType>,
    /// Shape preconditions as code, checked at plan time.
    pub when: Predicate,
    pub exact: Exactness,
    /// Budget for every output, or for non-f32 outputs when `tol_f32` is set.
    pub tol: Tol,
    /// Optional separate budget for f32 outputs; `None` = `tol` applies to all outputs.
    pub tol_f32: Option<Tol>,
    /// The single trusted implementation of an op on the reference backend.
    pub is_reference: bool,
    /// Higher wins when several admitted impls match.
    pub priority: i32,
    /// Scratch bytes needed for a shape; `None` = the kernel uses no scratch.
    pub scratch: Option<ScratchFn>,
    /// Speculative-decoding verify kernel: every row of a multi-row call is bitwise the backend's single-row
    /// decode launch at that row's position. Selected only for calls with `OpCall::exact_rows` set (and then
    /// exclusively, for the ops that have one); ordinary calls never pick it.
    pub rows_exact: bool,
    /// Row-table kernel (`OpCall::rows`): runs the one-row (`T = 1`) decode call of the solo step on every row of a batched
    /// step at once, the rows' state buffers named by a table of device addresses; row `r` is bitwise the solo kernel the
    /// registration is the twin of (G-B, the backend's `tests/row_tables.rs`). Selected only for calls with a row table, and
    /// those only select these.
    pub row_table: bool,
    /// Implements the extended attributes (`OpAttrs::window`, `rope_prop`, `gelu`). Off by default; see `ShapeInfo::ext`.
    pub ext: bool,
}

fn always(_: &ShapeInfo) -> bool {
    true
}

impl KernelImpl {
    pub fn new(op: Op, name: impl Into<String>, backend: impl Into<String>) -> KernelImpl {
        KernelImpl {
            op,
            name: name.into(),
            backend: backend.into(),
            archs: vec!["any".into()],
            acts: vec![],
            weights: vec![],
            when: always,
            exact: Exactness::Fast,
            tol: Tol::new(1e-2, 1e-2),
            tol_f32: None,
            is_reference: false,
            priority: 0,
            scratch: None,
            rows_exact: false,
            row_table: false,
            ext: false,
        }
    }

    /// Mark the kernel as implementing the extended attributes (`OpAttrs::window`, `rope_prop`, `gelu`).
    pub fn ext(mut self) -> Self {
        self.ext = true;
        self
    }

    // Builder methods named after the `register_kernel!` fields.
    pub fn op(mut self, op: Op) -> Self {
        self.op = op;
        self
    }
    pub fn name(mut self, n: &str) -> Self {
        self.name = n.to_string();
        self
    }
    pub fn backend(mut self, b: &str) -> Self {
        self.backend = b.to_string();
        self
    }
    pub fn arch<const N: usize>(mut self, a: [&str; N]) -> Self {
        self.archs = a.iter().map(|s| s.to_string()).collect();
        self
    }
    pub fn acts<const N: usize>(mut self, a: [DType; N]) -> Self {
        self.acts = a.to_vec();
        self
    }
    pub fn weights<const N: usize>(mut self, a: [DType; N]) -> Self {
        self.weights = a.to_vec();
        self
    }
    pub fn when(mut self, p: Predicate) -> Self {
        self.when = p;
        self
    }
    pub fn exact(mut self, e: Exactness) -> Self {
        self.exact = e;
        self
    }
    pub fn tol(mut self, t: Tol) -> Self {
        self.tol = t;
        self
    }
    /// Separate (tighter) budget for f32 outputs; see `tol_for`.
    pub fn tol_f32(mut self, t: Tol) -> Self {
        self.tol_f32 = Some(t);
        self
    }
    pub fn reference(mut self, r: bool) -> Self {
        self.is_reference = r;
        self
    }
    /// Mark as a verify-row kernel (see `KernelImpl::rows_exact`).
    pub fn rows_exact(mut self) -> Self {
        self.rows_exact = true;
        self
    }
    /// Mark as a row-table kernel (see `KernelImpl::row_table`).
    pub fn row_table(mut self) -> Self {
        self.row_table = true;
        self
    }
    pub fn priority(mut self, p: i32) -> Self {
        self.priority = p;
        self
    }
    /// Declare a scratch-workspace requirement as a function of the call's shape.
    pub fn scratch(mut self, f: ScratchFn) -> Self {
        self.scratch = Some(f);
        self
    }

    /// The tolerance that governs an output tensor of dtype `out`: `tol_f32` for f32 outputs when set,
    /// `tol` otherwise.
    pub fn tol_for(&self, out: DType) -> Tol {
        match self.tol_f32 {
            Some(t) if out == DType::F32 => t,
            _ => self.tol,
        }
    }

    /// Scratch bytes this kernel needs for `s` (0 when it declares none).
    pub fn scratch_bytes(&self, s: &ShapeInfo) -> usize {
        self.scratch.map_or(0, |f| f(s))
    }

    pub fn arch_matches(&self, arch: &str) -> bool {
        self.archs.iter().any(|a| a == "any" || a == arch)
    }

    /// Does this impl accept the call (dtypes + shape predicate)?
    pub fn accepts(&self, s: &ShapeInfo) -> bool {
        let act_ok = self.acts.is_empty() || s.act.is_none_or(|d| self.acts.contains(&d));
        let w_ok = self.weights.is_empty() || s.weight.is_none_or(|d| self.weights.contains(&d));
        act_ok && w_ok && (self.ext || !s.ext) && (self.when)(s)
    }
}

/// Declarative registration, mirroring ARCHITECTURE 3.2:
///
/// ```ignore
/// register_kernel! {
///     op: Op::Matmul, name: "bf16_gemv_v1", backend: "cuda",
///     arch: ["sm_121"], weights: [DType::Bf16], acts: [DType::Bf16],
///     when: |s| s.m <= 8 && s.k % 64 == 0,
///     exact: Exactness::Deterministic,
///     tol: Tol::new(2e-2, 5e-3),
/// }
/// ```
/// Expands to a `KernelImpl` value; unlisted fields keep their defaults.
#[macro_export]
macro_rules! register_kernel {
    ( $( $field:ident : $val:expr ),* $(,)? ) => {{
        let k = $crate::registry::KernelImpl::new($crate::ops::Op::Add, "", "");
        $( let k = k.$field($val); )*
        k
    }};
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Admission {
    /// Not tested in this process.
    Untested,
    Admitted,
    /// Failed the admission test; never selected.
    Rejected,
}

/// Registered kernels of one backend plus their admission status.
pub struct Registry {
    backend: String,
    arch: String,
    entries: Vec<(KernelImpl, Admission)>,
    /// When true only `Admitted` impls are selectable (production serve mode).
    pub require_admitted: bool,
    /// When true only reference implementations (`is_reference`) are selectable (`--reference`).
    /// A backend without reference kernels (cuda) then fails with `NoKernel` instead of silently
    /// running something else.
    pub reference_only: bool,
}

impl Registry {
    pub fn new(backend: &str, info: &DeviceInfo, kernels: Vec<KernelImpl>) -> Registry {
        Registry {
            backend: backend.to_string(),
            arch: info.arch.clone(),
            entries: kernels.into_iter().filter(|k| k.backend == backend).map(|k| (k, Admission::Untested)).collect(),
            require_admitted: false,
            reference_only: false,
        }
    }

    pub fn impls(&self) -> impl Iterator<Item = (&KernelImpl, Admission)> {
        self.entries.iter().map(|(k, a)| (k, *a))
    }

    /// Admission status of the implementation called `name`.
    /// Take over another registry's selection state (flags and admission results). Tensor-parallel peer ranks run
    /// the same kernels on identical devices, so they follow rank 0's admission instead of repeating it.
    pub fn adopt_status(&mut self, other: &Registry) {
        self.require_admitted = other.require_admitted;
        self.reference_only = other.reference_only;
        for (k, a) in self.entries.iter_mut() {
            if let Some(x) = other.admission(&k.name) {
                *a = x;
            }
        }
    }

    pub fn admission(&self, name: &str) -> Option<Admission> {
        self.entries.iter().find(|(k, _)| k.name == name).map(|(_, a)| *a)
    }

    pub fn kernel(&self, name: &str) -> Option<&KernelImpl> {
        self.entries.iter().find(|(k, _)| k.name == name).map(|(k, _)| k)
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// A row-table kernel (`OpCall::rows`) takes the admission of the kernel it twins (its name without `_rows`): the sweep has
    /// no case with a table of device addresses, so the row-table kernel is never tested there; its own gate is the backend's
    /// bitwise test against the one-row kernel (`tests/row_tables.rs`). Without this, `serve` (admitted kernels only) would
    /// never select the row-table twins.
    fn effective_admission(&self, k: &KernelImpl, own: Admission) -> Admission {
        if !k.row_table {
            return own;
        }
        k.name.strip_suffix("_rows").and_then(|base| self.admission(base)).unwrap_or(own)
    }

    pub fn set_admission(&mut self, name: &str, a: Admission) {
        for (k, st) in self.entries.iter_mut() {
            if k.name == name {
                *st = a;
            }
        }
    }

    /// Choose the implementation for a call: arch matches, predicate holds, not
    /// rejected; highest priority wins (ties: registration order).
    pub fn select(&self, call: &OpCall) -> Result<&KernelImpl> {
        let s = ShapeInfo::from_call(call);
        // Verify calls (`exact_rows`) of an op that has row-exact kernels may only take one of those; ops without
        // any (row-parallel norms, embedding, argmax, ...) keep the ordinary choice. Ordinary calls skip them.
        let rows_only = call.exact_rows && call.rows.is_none() && self.entries.iter().any(|(k, _)| k.op == call.op && k.rows_exact);
        let mut best: Option<&KernelImpl> = None;
        for (k, adm) in &self.entries {
            if k.op != call.op || !k.arch_matches(&self.arch) || !k.accepts(&s) {
                continue;
            }
            if k.rows_exact != rows_only || k.row_table != call.rows.is_some() {
                continue;
            }
            if self.reference_only && !k.is_reference {
                continue;
            }
            match self.effective_admission(k, *adm) {
                Admission::Rejected => continue,
                Admission::Untested if self.require_admitted => continue,
                _ => {}
            }
            if best.is_none_or(|b| k.priority > b.priority) {
                best = Some(k);
            }
        }
        best.ok_or_else(|| {
            Error::NoKernel(format!(
                "{} on {}/{} (m={} n={} k={} act={:?} w={:?}){}{}{}",
                call.op,
                self.backend,
                self.arch,
                s.m,
                s.n,
                s.k,
                s.act,
                s.weight,
                if self.reference_only { " [reference kernels only]" } else { "" },
                if self.require_admitted { " [admitted kernels only]" } else { "" },
                if rows_only { " [row-exact verify kernels only]" } else { "" },
            ))
        })
    }

    /// Summary used by `doctor` / `plan show`: op -> impl names.
    pub fn by_op(&self) -> HashMap<Op, Vec<&KernelImpl>> {
        let mut m: HashMap<Op, Vec<&KernelImpl>> = HashMap::new();
        for (k, _) in &self.entries {
            m.entry(k.op).or_default().push(k);
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macro_builds_impl() {
        let k = register_kernel! {
            op: Op::Matmul, name: "x", backend: "cpu",
            arch: ["any"], acts: [DType::Bf16], weights: [DType::Bf16],
            when: |s| s.m <= 8,
            exact: Exactness::Deterministic,
            tol: Tol::new(1e-3, 1e-3),
        };
        assert_eq!(k.op, Op::Matmul);
        assert_eq!(k.name, "x");
        let mut s = ShapeInfo::default();
        s.m = 4;
        s.act = Some(DType::Bf16);
        s.weight = Some(DType::Bf16);
        assert!(k.accepts(&s));
        s.m = 16;
        assert!(!k.accepts(&s));
        s.m = 1;
        s.act = Some(DType::F32);
        assert!(!k.accepts(&s));
    }

    #[test]
    fn a_row_table_kernel_takes_the_admission_of_the_kernel_it_twins() {
        let base = KernelImpl::new(Op::GdnConv, "gdn_conv_x", "b");
        let twin = KernelImpl::new(Op::GdnConv, "gdn_conv_x_rows", "b").row_table();
        let other = KernelImpl::new(Op::Matmul, "mm_rows", "b").rows_exact();
        let entries = vec![(base, Admission::Untested), (twin.clone(), Admission::Untested), (other.clone(), Admission::Untested)];
        let mut reg = Registry { backend: "b".into(), arch: "any".into(), entries, require_admitted: true, reference_only: false };
        assert_eq!(reg.effective_admission(&twin, Admission::Untested), Admission::Untested);
        reg.set_admission("gdn_conv_x", Admission::Admitted);
        assert_eq!(reg.effective_admission(&twin, Admission::Untested), Admission::Admitted);
        reg.set_admission("gdn_conv_x", Admission::Rejected);
        assert_eq!(reg.effective_admission(&twin, Admission::Untested), Admission::Rejected);
        // a rows-exact kernel is swept on its own, so its own admission stands
        assert_eq!(reg.effective_admission(&other, Admission::Untested), Admission::Untested);
    }

    #[test]
    fn tol_for_picks_the_f32_budget_only_for_f32_outputs() {
        let k = KernelImpl::new(Op::Logits, "x", "b").tol(Tol::new(1.5e-2, 1e-3));
        // no f32 budget: one tolerance for every output dtype
        assert_eq!(k.tol_for(DType::F32).rel_l2, 1e-3);
        assert_eq!(k.tol_for(DType::Bf16).rel_l2, 1e-3);
        let k = k.tol_f32(Tol::new(1e-3, 2e-4));
        assert_eq!((k.tol_for(DType::F32).max_abs, k.tol_for(DType::F32).rel_l2), (1e-3, 2e-4));
        assert_eq!((k.tol_for(DType::Bf16).max_abs, k.tol_for(DType::Bf16).rel_l2), (1.5e-2, 1e-3));
        assert_eq!(k.tol_for(DType::I32).rel_l2, 1e-3);
    }
}
