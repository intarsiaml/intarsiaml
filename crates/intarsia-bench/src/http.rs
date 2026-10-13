//! Small blocking HTTP helpers on top of `ureq` (plain http only).

use anyhow::{bail, Result};
use std::time::Duration;

pub fn agent(timeout: Option<Duration>) -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(timeout)
        .build()
        .into()
}

/// Server origin without a trailing slash or `/v1` suffix: `http://h:8090/v1/` -> `http://h:8090`.
pub fn origin(url: &str) -> String {
    let mut u = url.trim().trim_end_matches('/').to_string();
    if let Some(s) = u.strip_suffix("/v1") {
        u = s.to_string();
    }
    if !u.contains("://") {
        u = format!("http://{u}");
    }
    u
}

/// OpenAI base URL ending in `/v1`.
pub fn api_base(url: &str) -> String {
    format!("{}/v1", origin(url))
}

pub fn check_scheme(url: &str) -> Result<()> {
    if url.trim().starts_with("https://") {
        bail!("https is not supported by this client (built without TLS); use http or a tunnel");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn urls() {
        assert_eq!(origin("http://a:1/v1/"), "http://a:1");
        assert_eq!(origin("a:1"), "http://a:1");
        assert_eq!(api_base("http://a:1"), "http://a:1/v1");
    }
}
