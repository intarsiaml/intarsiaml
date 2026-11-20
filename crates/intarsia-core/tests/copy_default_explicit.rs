//! Lane copyflipm: an explicit `INTARSIA_SPEC_COPY` value wins over the default and is what the load-time refusal sees, also where the
//! default does not apply (`--max-batch` above 1: the loader keeps refusing it). Own test binary: read once per process.
use intarsia_core::spec::{copy_default_applies, copy_explicit_from_env, copy_from_env, copy_is_default, set_copy_backend, set_copy_default};

#[test]
fn explicit_value_is_explicit_everywhere() {
    std::env::set_var("INTARSIA_SPEC_COPY", "12");
    set_copy_backend("metal");
    set_copy_default(copy_default_applies("metal", 8, 1, true)); // default does not apply
    let c = copy_from_env().expect("explicit value on");
    assert_eq!((c.n, c.len), (12, 15));
    assert_eq!(copy_explicit_from_env().map(|c| (c.n, c.len)), Some((12, 15)), "the refusal check sees it");
    assert!(!copy_is_default());
    set_copy_default(copy_default_applies("metal", 1, 1, true));
    assert_eq!(copy_from_env().map(|c| c.n), Some(12), "n 12, not the default 20");
    // lane copyflipc: CUDA. An explicit value is the backend's length (7), wins over the default where it applies, and is refused-by-the-loader
    // (seen by `copy_explicit_from_env`) where the default does not (`--max-batch` above 1, tp above 1)
    set_copy_backend("cuda");
    set_copy_default(copy_default_applies("cuda", 2, 1, true));
    let c = copy_from_env().expect("explicit value on");
    assert_eq!((c.n, c.len), (12, 7));
    assert_eq!(copy_explicit_from_env().map(|c| (c.n, c.len)), Some((12, 7)));
    assert!(!copy_is_default());
    set_copy_default(copy_default_applies("cuda", 1, 2, true));
    assert_eq!(copy_explicit_from_env().map(|c| c.n), Some(12));
    set_copy_default(copy_default_applies("cuda", 1, 1, true));
    assert_eq!(copy_from_env().map(|c| (c.n, c.len)), Some((12, 7)), "n 12, not the default 20");
}
