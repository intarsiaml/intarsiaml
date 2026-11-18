//! Lane copyflipm: with `INTARSIA_SPEC_COPY` unset the copy drafts are on exactly where the loader says the default applies (Metal, one
//! sequence, one rank) and off everywhere else; `copy_from_env` (the one resolution point) answers the load-time verify-row width and the
//! engine alike. Own test binary: the variable is read once per process.
use intarsia_core::spec::{
    copy_default_applies, copy_explicit_from_env, copy_from_env, copy_is_default, set_copy_backend, set_copy_default, SpecConfig, SpecMode,
};

fn mtp(k: usize) -> SpecConfig {
    SpecConfig { mode: SpecMode::Mtp, k, ..SpecConfig::default() }
}

#[test]
fn unset_is_on_for_metal_one_sequence_and_off_elsewhere() {
    std::env::remove_var("INTARSIA_SPEC_COPY");

    // before the loader says anything: off (a tensor-parallel rank, a test)
    assert!(copy_from_env().is_none() && !copy_is_default());
    assert_eq!(mtp(2).verify_rows(), 3);

    // Metal, max batch 1, tp 1: `INTARSIA_SPEC_COPY=20` at Metal's length 15, in the verify rows and the chain alike
    set_copy_backend("metal");
    set_copy_default(copy_default_applies("metal", 1, 1));
    let c = copy_from_env().expect("on");
    assert_eq!((c.n, c.len, c.adapt), (20, 15, false));
    assert!(copy_is_default());
    assert_eq!(mtp(2).verify_rows(), 16, "the load-time verify rows are those of the chain");
    assert_eq!(mtp(2).save_rows(), 15);
    // the default is not an explicit request: the load-time refusal of unsupported configs does not see it
    assert!(copy_explicit_from_env().is_none());
    // `--spec off` / ngram: the chain never reaches the verify rows
    assert_eq!(SpecConfig { mode: SpecMode::Ngram, k: 2, ..SpecConfig::default() }.verify_rows(), 3);

    // Metal at `--max-batch 8`, tp 2, and other backends: off, unchanged plan
    for (backend, mb, tp) in [("metal", 8, 1), ("metal", 1, 2), ("cuda", 1, 1), ("hip", 1, 1), ("cpu", 1, 1)] {
        set_copy_backend(backend);
        set_copy_default(copy_default_applies(backend, mb, tp));
        assert!(copy_from_env().is_none(), "{backend} mb {mb} tp {tp}");
        assert!(!copy_is_default());
        assert_eq!(mtp(2).verify_rows(), 3, "{backend} mb {mb} tp {tp}: the plan is the one without the variable");
    }
}
