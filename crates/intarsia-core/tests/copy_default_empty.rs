//! Lane copyflipm: an empty `INTARSIA_SPEC_COPY` is off even where the default applies. Own test binary: read once per process.
use intarsia_core::spec::{copy_default_applies, copy_from_env, set_copy_backend, set_copy_default};

#[test]
fn empty_is_off_on_metal_one_sequence() {
    std::env::set_var("INTARSIA_SPEC_COPY", "");
    set_copy_backend("metal");
    set_copy_default(copy_default_applies("metal", 1, 1));
    assert!(copy_from_env().is_none());
}
