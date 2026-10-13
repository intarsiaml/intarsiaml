//! Built-in web UI: static assets from `web/` compiled into the binary, served as an axum router.
//!
//! `GET /` is the page, `GET /ui/*` its stylesheet and ES modules. No build step, no CDN.

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::get;
use axum::Router;

const ASSETS: &[(&str, &str, &str)] = &[
    ("index.html", "text/html; charset=utf-8", include_str!("../../../web/index.html")),
    ("style.css", "text/css; charset=utf-8", include_str!("../../../web/style.css")),
    ("app.js", "text/javascript; charset=utf-8", include_str!("../../../web/app.js")),
    ("util.js", "text/javascript; charset=utf-8", include_str!("../../../web/util.js")),
    ("chat.js", "text/javascript; charset=utf-8", include_str!("../../../web/chat.js")),
    ("dashboard.js", "text/javascript; charset=utf-8", include_str!("../../../web/dashboard.js")),
    ("arch.js", "text/javascript; charset=utf-8", include_str!("../../../web/arch.js")),
    ("favicon.svg", "image/svg+xml", include_str!("../../../web/favicon.svg")),
];

fn asset(name: &str) -> Response {
    match ASSETS.iter().find(|(n, _, _)| *n == name) {
        Some((_, ct, body)) => {
            ([(header::CONTENT_TYPE, *ct), (header::CACHE_CONTROL, "no-cache")], *body).into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Routes `/`, `/ui` and `/ui/{file}`. Generic over the state so it merges into any router.
pub fn router<S: Clone + Send + Sync + 'static>() -> Router<S> {
    Router::new()
        .route("/", get(|| async { asset("index.html") }))
        .route("/ui", get(|| async { Redirect::permanent("/") }))
        .route("/ui/", get(|| async { Redirect::permanent("/") }))
        .route("/ui/{file}", get(|axum::extract::Path(f): axum::extract::Path<String>| async move { asset(&f) }))
        .route("/favicon.ico", get(|| async { asset("favicon.svg") }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt;

    async fn status(path: &str) -> (StatusCode, String) {
        let r = router::<()>()
            .oneshot(axum::http::Request::get(path).body(axum::body::Body::empty()).unwrap())
            .await
            .unwrap();
        let ct = r.headers().get(header::CONTENT_TYPE).map(|v| v.to_str().unwrap().to_string()).unwrap_or_default();
        (r.status(), ct)
    }

    #[tokio::test]
    async fn serves_every_asset() {
        assert_eq!(status("/").await.0, StatusCode::OK);
        for (n, ct, _) in ASSETS {
            let (s, c) = status(&format!("/ui/{n}")).await;
            assert_eq!(s, StatusCode::OK, "{n}");
            assert_eq!(&c, ct);
        }
        assert_eq!(status("/ui/nope.js").await.0, StatusCode::NOT_FOUND);
    }
}
