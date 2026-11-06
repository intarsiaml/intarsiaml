//! DFlash2 token trees (lane dm, phase 2, docs/results/2026-11-06-metal-dflash2.md "Phase 2 design"): host-side construction of a draft
//! tree from ONE draft block pass. Backend-neutral and pure (no device work), so the selection can be tested on its own.
//!
//! Block row `d` (d >= 1) of the draft proposes the token at depth `d` (the anchor, the token the target already chose, is depth 0).
//! Its candidates do not depend on the parent (block rows only see the anchor and MASK rows); the parent enters through the scorer's
//! edge term (the DFlash2 selector's `pred[parent] . (proj(h_d) * succ[child])`). A child's log-probability is the log-softmax of
//! `(unary + edge_w * edge) / tau` over the row's candidates for that parent; a node's score is the sum along its path. Best-first:
//! the highest-scoring frontier node is taken next until `budget` nodes are taken; the children of a taken node are the `children`
//! best candidates of the next row. Drafts never change the output (every emitted token is a verify row's argmax), so these choices
//! move acceptance only.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A draft tree: node `i` proposes `tokens[i]` at depth `depth[i]` (>= 1) under `parent[i]` (`None` = the anchor). Parents come
/// before their children.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TreeDraft {
    pub tokens: Vec<u32>,
    pub parent: Vec<Option<usize>>,
    pub depth: Vec<usize>,
    /// Path log-probability of each node (the selection score).
    pub score: Vec<f32>,
}

impl TreeDraft {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
    /// Node indices from the root's child down to `i`.
    pub fn path(&self, i: usize) -> Vec<usize> {
        let mut p = vec![i];
        let mut c = i;
        while let Some(q) = self.parent[c] {
            p.push(q);
            c = q;
        }
        p.reverse();
        p
    }
    /// Whether the tree is a single chain (every node's parent is the node before it).
    pub fn is_chain(&self) -> bool {
        self.parent.iter().enumerate().all(|(i, p)| *p == i.checked_sub(1))
    }
    /// Greedy acceptance against the verify rows' tokens: `target[0]` is the target's token after the anchor, `target[1 + i]` the
    /// target's token after node `i`. Returns the accepted node path (possibly empty) and the emitted tokens (accepted drafts + the
    /// bonus token of the last accepted row), as the chain's `spec::accept` does.
    pub fn accept(&self, target: &[u32]) -> (Vec<usize>, Vec<u32>) {
        let mut path = Vec::new();
        let mut want = target[0];
        let mut at: Option<usize> = None;
        loop {
            let next = (0..self.len()).find(|&i| self.parent[i] == at && self.tokens[i] == want);
            match next {
                Some(i) => {
                    path.push(i);
                    at = Some(i);
                    want = target[1 + i];
                }
                None => break,
            }
        }
        let mut emit: Vec<u32> = path.iter().map(|&i| self.tokens[i]).collect();
        emit.push(want);
        (path, emit)
    }
}

/// Knobs of [`build`].
#[derive(Clone, Copy, Debug)]
pub struct TreeParams {
    /// Most nodes (drafts); the verify pass has `budget + 1` rows.
    pub budget: usize,
    /// Children considered per taken node.
    pub children: usize,
    /// Weight of the edge term.
    pub edge_w: f32,
    /// Softmax temperature of the sibling scores.
    pub tau: f32,
}

impl Default for TreeParams {
    fn default() -> Self {
        TreeParams { budget: 7, children: 4, edge_w: 1.0, tau: 1.0 }
    }
}

#[derive(PartialEq)]
struct Front {
    score: f32,
    /// insertion order: ties go to the earlier candidate (deterministic)
    seq: usize,
    parent: Option<usize>,
    depth: usize,
    token: u32,
}
impl Eq for Front {}
impl PartialOrd for Front {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Front {
    fn cmp(&self, o: &Self) -> Ordering {
        self.score.total_cmp(&o.score).then(o.seq.cmp(&self.seq))
    }
}

/// Build a tree from the per-row candidates. `cands[d - 1]` holds depth `d`'s candidates `(unary logit, token)`; `edge(parent_token,
/// d, token)` the edge term of `token` at depth `d` under `parent_token` (the anchor at depth 1).
pub fn build(cands: &[Vec<(f32, u32)>], anchor: u32, p: TreeParams, edge: &dyn Fn(u32, usize, u32) -> f32) -> TreeDraft {
    let mut t = TreeDraft::default();
    let mut heap = BinaryHeap::new();
    let mut seq = 0usize;
    let mut push_children = |heap: &mut BinaryHeap<Front>, parent: Option<usize>, ptok: u32, pscore: f32, depth: usize, seq: &mut usize| {
        let Some(row) = cands.get(depth - 1) else { return };
        if row.is_empty() {
            return;
        }
        let z: Vec<f32> = row.iter().map(|&(u, c)| (u + p.edge_w * edge(ptok, depth, c)) / p.tau).collect();
        let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = m + z.iter().map(|&x| (x - m).exp()).sum::<f32>().ln();
        let mut order: Vec<usize> = (0..row.len()).collect();
        order.sort_by(|&a, &b| z[b].total_cmp(&z[a]).then(a.cmp(&b)));
        for &j in order.iter().take(p.children) {
            heap.push(Front { score: pscore + z[j] - lse, seq: *seq, parent, depth, token: row[j].1 });
            *seq += 1;
        }
    };
    push_children(&mut heap, None, anchor, 0.0, 1, &mut seq);
    while t.len() < p.budget {
        let Some(f) = heap.pop() else { break };
        let i = t.len();
        t.tokens.push(f.token);
        t.parent.push(f.parent);
        t.depth.push(f.depth);
        t.score.push(f.score);
        push_children(&mut heap, Some(i), f.token, f.score, f.depth + 1, &mut seq);
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: usize, k: usize, seed: u64) -> Vec<Vec<(f32, u32)>> {
        let mut s = seed | 1;
        let mut r = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 10_000) as f32 / 1000.0
        };
        (0..n).map(|d| (0..k).map(|j| (r(), (d * 100 + j) as u32)).collect()).collect()
    }

    #[test]
    fn parents_before_children_depths_and_budget() {
        for seed in 1..50u64 {
            let c = rows(7, 8, seed);
            let edge = |a: u32, d: usize, b: u32| (((a * 31 + b * 17 + d as u32) % 13) as f32 - 6.0) / 4.0;
            for budget in [1, 3, 7, 15] {
                let t = build(&c, 9999, TreeParams { budget, children: 4, edge_w: 1.0, tau: 1.0 }, &edge);
                assert_eq!(t.len(), budget.min(7 * 4usize.pow(7)));
                for i in 0..t.len() {
                    match t.parent[i] {
                        None => assert_eq!(t.depth[i], 1),
                        Some(q) => {
                            assert!(q < i, "parent after child");
                            assert_eq!(t.depth[i], t.depth[q] + 1);
                            assert!(t.score[i] <= t.score[q] + 1e-6, "a child never outscores its parent");
                        }
                    }
                    assert!(t.depth[i] <= 7);
                }
                // sibling tokens are distinct (a token never appears twice under one parent)
                for i in 0..t.len() {
                    for j in i + 1..t.len() {
                        assert!(!(t.parent[i] == t.parent[j] && t.tokens[i] == t.tokens[j]));
                    }
                }
            }
        }
    }

    #[test]
    fn one_child_per_node_is_the_greedy_chain() {
        let c = rows(7, 8, 3);
        let edge = |a: u32, _d: usize, b: u32| ((a ^ b) % 7) as f32 / 3.0;
        let t = build(&c, 5, TreeParams { budget: 7, children: 1, edge_w: 1.0, tau: 1.0 }, &edge);
        assert!(t.is_chain());
        // the chain is the argmax of unary + edge per row, walking from the anchor (what `Selector::select` does)
        let mut prev = 5u32;
        for (d, row) in c.iter().enumerate() {
            let best = row.iter().copied().max_by(|x, y| (x.0 + edge(prev, d + 1, x.1)).total_cmp(&(y.0 + edge(prev, d + 1, y.1)))).unwrap();
            assert_eq!(t.tokens[d], best.1);
            prev = best.1;
        }
    }

    #[test]
    fn accept_walks_the_matching_path() {
        // anchor -> {a(0), b(1)}; a -> {c(2)}; b -> {d(3)}
        let t = TreeDraft { tokens: vec![10, 11, 12, 13], parent: vec![None, None, Some(0), Some(1)], depth: vec![1, 1, 2, 2], score: vec![0.0; 4] };
        // target after anchor = 11 (b, node 1), after node 1 = target[2] = 13 (d, node 3), after node 3 = target[4] = 99
        let (path, emit) = t.accept(&[11, 7, 13, 9, 99]);
        assert_eq!(path, vec![1, 3]);
        assert_eq!(emit, vec![11, 13, 99]);
        let (path, emit) = t.accept(&[42, 0, 0, 0, 0]);
        assert!(path.is_empty());
        assert_eq!(emit, vec![42]);
        assert_eq!(t.path(3), vec![1, 3]);
    }
}
