//! Bounded-replay prefill (default on where [`supported`] and on CUDA; `INTARSIA_DSV41_REPLAY=0` turns it off). Design and what is untested:
//! docs/results/2026-11-03-dsv41-replay.md.
//!
//! TensorFold runs the decoder half of the model (layers `n_layers / 2 ..`) only on the last `window` rows of a prompt. Layer
//! `cut` is the model's last compressed-KV source (`kv_source_layers`): every later layer reads *its* compressed KV and index
//! keys, so below the floor `prompt length - window` the decoder half produces nothing another row consumes except its own
//! sliding-window keys, which the replay drops. This module holds the arithmetic both TP ranks derive their passes from. It is
//! pure: a rank's plan is a function of `(prompt length, window, st.len, chunk rows)` only, which both ranks know (the primary
//! sends the same `ids` and the same `truncate` to the other rank, `tp_remote`), so both make the same cut at the same point
//! and run the same collectives. The graphs a plan selects: [`super::graph::build_replay`].

use super::config::Dsv41Config;
use super::graph::GraphTp;
use crate::error::{Error, Result};

/// The environment variable (read once per runtime and by the handshake fingerprint; both ranks must agree).
pub const ENV: &str = "INTARSIA_DSV41_REPLAY";

/// `INTARSIA_DSV41_REPLAY_SKIP` (default on, `0` = off; read by the primary only, which tells the other rank per chunk): a prefill
/// chunk whose decoder half no later row of the prompt reads is run encoder-only ([`decoder_dead`]).
pub const SKIP_ENV: &str = "INTARSIA_DSV41_REPLAY_SKIP";

/// What `INTARSIA_DSV41_REPLAY` asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Setting {
    /// Unset: on when the model and the backend support it, off otherwise.
    Auto,
    /// `1`: on, and the load fails when the model or the backend cannot run it.
    On,
    /// `0` or any other value: off, bit-identical to the path without the option.
    Off,
}

/// [`Setting`] of a value of the variable.
pub fn parse(v: Option<&str>) -> Setting {
    match v {
        None => Setting::Auto,
        Some("1") => Setting::On,
        Some(_) => Setting::Off,
    }
}

/// The variable's [`Setting`].
pub fn setting() -> Setting {
    parse(std::env::var(ENV).ok().as_deref())
}

/// Does this model on this backend run the replay? [`Setting::On`] fails where [`Setting::Auto`] falls back to off. The
/// replay drops window keys by shifting the sparse attention's position (the kernels' window test is `pos0 + row >= window`):
/// read in the CUDA kernel (kernels/cuda/dsv41_attn.cu) and the cpu reference (dsv41_ref.rs) only, and only the CUDA kernel is
/// meant to run it.
pub fn resolve(s: Setting, c: &Dsv41Config, tp: &GraphTp, backend: &str) -> Result<bool> {
    let check = || {
        supported(c, tp)?;
        if backend != "cuda" {
            return Err(Error::Unsupported(format!("deepseek_v41 {ENV}=1: not implemented for the {backend} backend")));
        }
        Ok(())
    };
    match s {
        Setting::Off => Ok(false),
        Setting::On => check().map(|()| true),
        Setting::Auto => Ok(check().is_ok()),
    }
}

/// The decoder half's first layer (TensorFold: `c.layers / 2`).
pub fn cut_layer(n_layers: usize) -> usize {
    n_layers / 2
}

/// First prompt position the decoder half runs on: `total - window` (0 when the prompt has at most `window` tokens: nothing is
/// cut). `total` is the whole prompt's length, `st.len + ids.len()` at the start of the prefill.
pub fn floor(total: usize, window: usize) -> usize {
    total.saturating_sub(window)
}

/// What a prefill chunk of `t` rows at `pos0` runs under a replay `floor` (> 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chunk {
    /// Every row is at or past the floor: all layers on all rows (the decoder layers' windows start at the floor).
    Whole,
    /// No row is at or past the floor: layers `0..cut` and layer `cut`'s kv-source update, no head.
    Encoder,
    /// The floor falls inside the chunk: the encoder part on all `t` rows, then the decoder half on the rows from `first` on.
    Cross { first: usize },
}

/// The chunk's kind. `floor == 0` (no replay: a prompt of at most `window` tokens) is [`Chunk::Whole`] for every chunk.
pub fn classify(floor: usize, pos0: usize, t: usize) -> Chunk {
    if pos0 >= floor {
        Chunk::Whole
    } else if pos0 + t <= floor {
        Chunk::Encoder
    } else {
        Chunk::Cross { first: floor - pos0 }
    }
}

/// Rows of the chunk that go through the decoder half.
pub fn decoder_rows(k: Chunk, t: usize) -> usize {
    match k {
        Chunk::Whole => t,
        Chunk::Encoder => 0,
        Chunk::Cross { first } => t - first,
    }
}

/// Position of the first row of the decoder part of a chunk of kind `k` that starts at `pos0`: the rows it runs are the chunk's
/// from `first` on, which the state (still at `pos0`, the chunk is committed after both parts) does not know.
pub fn decoder_pos0(k: Chunk, pos0: usize) -> usize {
    match k {
        Chunk::Cross { first } => pos0 + first,
        _ => pos0,
    }
}

/// The positions of `INTARSIA_DSV41_DUMP_POS` that fall in the `t` rows of a graph whose first row is at `pos0`.
pub fn dump_hits(pos: &[usize], pos0: usize, t: usize) -> Vec<usize> {
    pos.iter().copied().filter(|&p| p >= pos0 && p < pos0 + t).collect()
}

/// Decoder-half rows of a whole prefill of `total` tokens from `start` (chunks of at most `chunk` rows) under `floor`:
/// `total - max(start, floor)`, which the per-chunk [`decoder_rows`] must sum to.
pub fn prefill_decoder_rows(total: usize, start: usize, chunk: usize, floor: usize) -> usize {
    let (mut pos, mut rows) = (start, 0);
    while pos < total {
        let t = (total - pos).min(chunk);
        rows += decoder_rows(classify(floor, pos, t), t);
        pos += t;
    }
    rows
}

/// The engine prefills a prompt one chunk at a time (`dsv41_engine::Worker::prefill`), and each chunk's [`floor`] is that of the
/// rows the state holds after it: `end - window`. So every chunk of at least `window` rows runs the decoder half on its last
/// `window` rows, as if the prompt ended there. Is that work dead for the call ending at position `end`, with `ahead` more
/// prompt rows to come in chunks of `chunk`?
///
/// A later call `k` (ending at `end_k`) runs decoder rows from `end_k - window` on, and their sliding windows start at that
/// floor (the replay graphs drop the keys below it; a plain one-row pass reads the same `window - 1` keys), so it reads the
/// decoder-layer window keys written at positions `>= end_k - window` and nothing below. The call is dead when no *live* later
/// call reads below its end: the last call is live (it feeds the head); an earlier call is live when its end lies above
/// the lowest floor of a live call after it. Dead calls write window-ring rows (the decoder layers' keys and the DSpark stage rings)
/// at positions that every read leaves alone and that live calls overwrite, and nothing else: the compressed KV and index keys
/// of the decoder half are layer `cut`'s, written by the encoder part of every call. Skipping their decoder half is bit-identical.
/// Only for `chunk >= window` (a shorter chunk would put a later call's floor below this call's end, and [`reuse_ok`] would
/// refuse the later call).
pub fn decoder_dead(end: usize, ahead: usize, chunk: usize, window: usize) -> bool {
    if ahead == 0 || chunk < window || window == 0 {
        return false;
    }
    // ends of the later calls, the last one first
    let last = end + ahead;
    let mut lo = last.saturating_sub(window);
    let mut e = end + ahead.div_ceil(chunk).saturating_sub(1) * chunk;
    while e > end {
        if e > lo {
            lo = lo.min(e.saturating_sub(window));
        }
        e -= chunk;
    }
    end <= lo
}

/// May a prefill that resumes a state at `keep` tokens run under `floor`? The state's decoder-layer window rings hold valid keys
/// from position `from` on (`SequenceState::replay_from`: the floor of the replay prefill that built it, 0 when nothing was
/// skipped). The resumed rows' decoder windows read the ring from `floor` on, so rows `floor..keep` must be valid ones.
/// Nothing is read below `keep` when `keep <= floor`. Otherwise the caller prefills from scratch.
pub fn reuse_ok(keep: usize, floor: usize, from: usize) -> bool {
    keep <= floor || from <= floor
}

/// Is the model one whose decoder half can be replayed on its last `window` rows with the state this engine keeps? Every
/// refusal names the property; none of them holds for the Mia checkpoint's layout (the unit test in
/// `crates/intarsia-models/tests/dsv41_replay.rs` pins that).
pub fn supported(c: &Dsv41Config, tp: &GraphTp) -> Result<()> {
    let no = |m: String| Err(Error::Unsupported(format!("deepseek_v41 {ENV}=1: {m}")));
    if tp.layers.is_some() || tp.first != 0 {
        return no(format!("a truncated or layer-range model (layers {:?}, first {})", tp.layers, tp.first));
    }
    let cut = cut_layer(c.n_layers);
    if cut == 0 || c.window == 0 {
        return no(format!("{} layers, window {}", c.n_layers, c.window));
    }
    if c.ratio(cut) == 0 || c.kv_source(cut) != Some(cut) || c.index_source(cut) != Some(cut) {
        return no(format!("layer {cut} (the first decoder layer) is not both a kv source and an index source"));
    }
    // a source after the cut would leave its slot short of the rows below the floor (nothing would write them)
    if let Some(s) = c.kv_source_layers.iter().find(|&&s| s > cut) {
        return no(format!("layer {s} is a kv source after the cut at {cut}"));
    }
    for l in cut..c.n_layers {
        if c.ratio(l) > 0 && (c.kv_source(l) != Some(cut) || c.index_source(l).is_none_or(|i| i < cut)) {
            return no(format!("decoder layer {l} reads a kv/index source before the cut"));
        }
    }
    if let Some(e) = &c.engram {
        // the Engram rows are hashed and read for every row of a pass; the decoder half would need them for the kept rows only
        if let Some(l) = e.layer_ids.iter().find(|&&l| l >= cut) {
            return no(format!("an Engram layer ({l}) in the decoder half (cut at {cut})"));
        }
    }
    if let Some(m) = &c.mtp {
        // the taps are absorbed into the stage rings for the rows a pass's decoder half runs on; the cut layer's own input is
        // read before its Engram / tap in TensorFold's order, so a tap at the cut is refused too
        if let Some(l) = m.target_layer_ids.iter().find(|&&l| l <= cut) {
            return no(format!("a DSpark tap layer ({l}) in the encoder half or at the cut ({cut})"));
        }
    }
    if let Some(l) = c.candidate_source_layer.filter(|&l| l < cut) {
        return no(format!("the candidate-block source (layer {l}) is before the cut"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_is_the_last_window_rows() {
        assert_eq!(floor(4106, 128), 3978);
        assert_eq!(floor(129, 128), 1);
        assert_eq!(floor(128, 128), 0, "a prompt of exactly the window: nothing is cut");
        assert_eq!(floor(1, 128), 0);
        assert_eq!(floor(0, 128), 0);
    }

    #[test]
    fn classify_covers_encoder_cross_and_whole() {
        // floor 3978 (4106 tokens), chunk 2048: [0, 2048) and [2048, 4096) and [4096, 4106)
        assert_eq!(classify(3978, 0, 2048), Chunk::Encoder);
        assert_eq!(classify(3978, 2048, 2048), Chunk::Cross { first: 1930 });
        assert_eq!(classify(3978, 4096, 10), Chunk::Whole);
        // the chunk that ends exactly on the floor has no decoder row; the one that starts on it has all
        assert_eq!(classify(100, 0, 100), Chunk::Encoder);
        assert_eq!(classify(100, 100, 5), Chunk::Whole);
        // one row past the floor: a one-row decoder part
        assert_eq!(classify(100, 0, 101), Chunk::Cross { first: 100 });
        assert_eq!(decoder_rows(Chunk::Cross { first: 100 }, 101), 1);
        // no replay (floor 0): every chunk is whole, from position 0 on
        assert_eq!(classify(0, 0, 512), Chunk::Whole);
        assert_eq!(classify(0, 512, 1), Chunk::Whole);
    }

    #[test]
    fn a_prefill_sends_exactly_the_last_window_rows_through_the_decoder_half() {
        for chunk in [1, 7, 16, 128, 512, 2048] {
            for total in [1usize, 2, 127, 128, 129, 130, 255, 256, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4096, 4106, 32768] {
                let f = floor(total, 128);
                let want = total.min(128);
                assert_eq!(prefill_decoder_rows(total, 0, chunk, f), want, "total {total} chunk {chunk}");
                // the chunks before the floor are encoder-only, at most one chunk crosses, the rest are whole
                let (mut pos, mut kinds) = (0, Vec::new());
                while pos < total {
                    let t = (total - pos).min(chunk);
                    kinds.push(classify(f, pos, t));
                    pos += t;
                }
                assert!(kinds.iter().filter(|k| matches!(k, Chunk::Cross { .. })).count() <= 1, "total {total} chunk {chunk}");
                let last = kinds.last().unwrap();
                assert!(!matches!(last, Chunk::Encoder), "the chunk with the last row always reaches the head: total {total} chunk {chunk}");
                // order: encoder chunks, then at most one cross, then whole chunks
                let rank = |k: &Chunk| match k {
                    Chunk::Encoder => 0,
                    Chunk::Cross { .. } => 1,
                    Chunk::Whole => 2,
                };
                assert!(kinds.windows(2).all(|w| rank(&w[0]) <= rank(&w[1])), "{kinds:?}");
                if f == 0 {
                    assert!(kinds.iter().all(|k| *k == Chunk::Whole), "a prompt of at most the window is never cut");
                }
            }
        }
    }

    #[test]
    fn position_dumps_of_a_crossing_chunk_use_each_parts_own_first_row() {
        // 4106 tokens, chunk 2048, the middle chunk [2048, 4096) crosses the floor 3978: encoder part on all 2048 rows,
        // decoder part on the last 118 (positions 3978..4096)
        let (floor, pos0, t) = (3978, 2048, 2048);
        let k = classify(floor, pos0, t);
        let Chunk::Cross { first } = k else { panic!("{k:?}") };
        let dec0 = decoder_pos0(k, pos0);
        assert_eq!(dec0, floor);
        let want = [3977, 3978, 4000, 4095, 4096, 100];
        assert_eq!(dump_hits(&want, pos0, t), [3977, 3978, 4000, 4095]);
        assert_eq!(dump_hits(&want, dec0, t - first), [3978, 4000, 4095], "the decoder part's rows are 3978..4096");
        // what the part would have selected at the state's position (the old behaviour): the wrong rows
        assert_ne!(dump_hits(&want, pos0, t - first), dump_hits(&want, dec0, t - first));
        // the row index inside the part's output is `p - dec0`
        assert_eq!(4000 - dec0, 22);
        // every other kind starts where the state is
        assert_eq!(decoder_pos0(Chunk::Whole, 5000), 5000);
        assert_eq!(decoder_pos0(Chunk::Encoder, 0), 0);
    }

    #[test]
    fn a_resumed_prefill_runs_only_the_rows_it_is_given() {
        // 4000 tokens cached, 20 more: floor 3892 is below the resume point, every row is whole
        assert_eq!(prefill_decoder_rows(4020, 4000, 512, floor(4020, 128)), 20);
        // 100 cached, 4000 more: the first chunks are encoder-only
        assert_eq!(prefill_decoder_rows(4100, 100, 512, floor(4100, 128)), 128);
    }

    #[test]
    fn only_the_last_chunks_decoder_half_is_read() {
        let (c, w) = (2048, 128);
        // 4106 tokens: [0, 2048) is dead; [2048, 4096) is read by the 10-row tail (floor 3978 < 4096); the tail is last
        assert!(decoder_dead(2048, 2058, c, w));
        assert!(!decoder_dead(4096, 10, c, w));
        assert!(!decoder_dead(4106, 0, c, w));
        // 4096 tokens: the second chunk is last, the first is dead
        assert!(decoder_dead(2048, 2048, c, w));
        // a tail of at least `window` rows hides the chunk before it
        assert!(decoder_dead(2048, 128, c, w), "[2048, 2176) has floor 2048, nothing below is read");
        assert!(!decoder_dead(2048, 127, c, w), "a 127-row tail's floor 2047 reads position 2047");
        assert!(!decoder_dead(2048, 1, c, w));
        // 32768 tokens in 16 chunks: all but the last are dead
        for e in (2048..32768).step_by(2048) {
            assert!(decoder_dead(e, 32768 - e, c, w), "end {e}");
        }
        // the 4106-token prompt at chunk 512: chunks ending 512 .. 3584 are dead, 4096 is read by the tail
        for e in (512..=3584).step_by(512) {
            assert!(decoder_dead(e, 4106 - e, 512, w), "end {e}");
        }
        assert!(!decoder_dead(4096, 10, 512, w));
        // chunks below the window are never skipped (a later floor would sit below this call's end)
        assert!(!decoder_dead(64, 4000, 64, w));
        assert!(!decoder_dead(2048, 2058, 100, w));
    }

    /// The rule against the definition: simulate every call's reads (a call ending at `e` reads positions `>= e - window`, the last
    /// call and every call that a live call reads below its end), and compare with [`decoder_dead`] for many shapes.
    #[test]
    fn decoder_dead_matches_a_direct_simulation() {
        for window in [1usize, 4, 128] {
            for chunk in [window, window + 1, 2 * window, 3 * window + 5, 512] {
                if chunk < window {
                    continue;
                }
                for total in [1usize, 2, window, window + 1, chunk, chunk + 1, 2 * chunk, 2 * chunk + window - 1, 2 * chunk + window, 5 * chunk + 7, 40 * chunk] {
                    // ends of the calls of a prefill of `total` rows from 0
                    let mut ends = Vec::new();
                    let mut e = 0;
                    while e < total {
                        e = (e + chunk).min(total);
                        ends.push(e);
                    }
                    let n = ends.len();
                    let mut live = vec![false; n];
                    live[n - 1] = true;
                    for j in (0..n - 1).rev() {
                        live[j] = (j + 1..n).any(|k| live[k] && ends[j] > ends[k].saturating_sub(window));
                    }
                    for j in 0..n {
                        assert_eq!(decoder_dead(ends[j], total - ends[j], chunk, window), !live[j], "window {window} chunk {chunk} total {total} call {j}");
                    }
                }
            }
        }
    }

    #[test]
    fn reuse_needs_valid_decoder_window_rows_from_the_floor() {
        // nothing below the resume point is read: always fine
        assert!(reuse_ok(0, 0, 3978));
        assert!(reuse_ok(1000, 3978, 3978));
        assert!(reuse_ok(3978, 3978, 4000));
        // resumed past the floor: the ring rows floor..keep were written (valid from `from`)
        assert!(reuse_ok(4100, 3990, 3978), "the earlier floor 3978 <= this floor 3990");
        assert!(reuse_ok(4100, 3978, 3978));
        assert!(!reuse_ok(4100, 3970, 3978), "positions 3970..3978 were never run through the decoder half");
        // a state built without a cut (from 0) is valid everywhere
        assert!(reuse_ok(125, 0, 0));
        assert!(!reuse_ok(125, 0, 12), "a short prompt resumes rows 0..125 whose decoder keys below 12 were skipped");
    }

    fn mia() -> Dsv41Config {
        let v: serde_json::Value = serde_json::from_str(include_str!("../../../intarsia-models/tests/fixtures/dsv41/mia-config.json")).unwrap();
        Dsv41Config::from_json(&v).unwrap()
    }

    #[test]
    fn the_setting_is_on_where_supported_unless_turned_off() {
        assert_eq!(parse(None), Setting::Auto);
        assert_eq!(parse(Some("1")), Setting::On);
        assert_eq!(parse(Some("0")), Setting::Off);
        assert_eq!(parse(Some("")), Setting::Off);
        let c = mia();
        let tp = GraphTp::default();
        let cut = GraphTp { layers: Some(3), ..tp };
        for (s, t, backend, want) in [
            (Setting::Auto, &tp, "cuda", Some(true)),
            (Setting::Auto, &tp, "hip", Some(false)),
            (Setting::Auto, &tp, "metal", Some(false)),
            (Setting::Auto, &cut, "cuda", Some(false)),
            (Setting::On, &tp, "cuda", Some(true)),
            (Setting::On, &tp, "hip", None),
            (Setting::On, &cut, "cuda", None),
            (Setting::Off, &tp, "cuda", Some(false)),
            (Setting::Off, &cut, "hip", Some(false)),
        ] {
            assert_eq!(resolve(s, &c, t, backend).ok(), want, "{s:?} on {backend}");
        }
    }

    #[test]
    fn the_mia_layout_is_supported_and_variations_are_refused() {
        let c = mia();
        let tp = GraphTp::default();
        assert_eq!(cut_layer(c.n_layers), 20);
        supported(&c, &tp).unwrap();
        // a truncated model or a layer range is not a model
        assert!(supported(&c, &GraphTp { layers: Some(3), ..tp }).is_err());
        assert!(supported(&c, &GraphTp { first: 2, ..tp }).is_err());
        // an Engram layer in the decoder half
        let mut e = c.clone();
        e.engram.as_mut().unwrap().layer_ids = vec![1, 25];
        let m = supported(&e, &tp).unwrap_err().to_string();
        assert!(m.contains("Engram layer (25)"), "{m}");
        // a DSpark tap at or before the cut
        let mut d = c.clone();
        d.mtp.as_mut().unwrap().target_layer_ids = vec![20, 38, 39];
        assert!(supported(&d, &tp).unwrap_err().to_string().contains("DSpark tap layer (20)"));
        // a second kv source after the cut
        let mut s = c.clone();
        s.kv_source_layers = vec![2, 8, 14, 20, 30];
        assert!(supported(&s, &tp).is_err());
        // candidate blocks computed before the cut
        let mut k = c.clone();
        k.candidate_source_layer = Some(14);
        assert!(supported(&k, &tp).unwrap_err().to_string().contains("candidate-block source"));
    }
}
