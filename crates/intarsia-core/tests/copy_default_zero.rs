//! Lane copyflipm: an explicit `INTARSIA_SPEC_COPY=0` is off even where the default applies (Metal, one sequence, one rank).
//! Own test binary: the variable is read once per process.
use intarsia_core::spec::{copy_default_applies, copy_explicit_from_env, copy_from_env, copy_is_default, set_copy_backend, set_copy_default, SpecConfig, SpecMode};

#[test]
fn zero_is_off_on_metal_one_sequence() {
    std::env::set_var("INTARSIA_SPEC_COPY", "0");
    set_copy_backend("metal");
    set_copy_default(copy_default_applies("metal", 1, 1));
    assert!(copy_from_env().is_none() && copy_explicit_from_env().is_none() && !copy_is_default());
    assert_eq!(SpecConfig { mode: SpecMode::Mtp, k: 2, ..SpecConfig::default() }.verify_rows(), 3);
}
