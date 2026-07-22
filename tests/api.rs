mod common;

use std::sync::Arc;

use axum::{extract::State, response::IntoResponse};
use serde_json::Value;
use sumi::{renderer::CardRenderer, routes::handle_metrics};
use tempfile::TempDir;

#[tokio::test]
async fn test_metrics_endpoint_initial_state() {
    let dir = TempDir::new().unwrap();
    common::create_dummy_webp(dir.path(), "test_card.webp");

    let renderer = CardRenderer::new(dir.path()).expect("failed to initialize renderer");
    let state = State(Arc::new(renderer));

    // Act
    let response = handle_metrics(state).await.into_response();

    // Assert
    assert_eq!(response.status(), 200, "expected 200 OK");

    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("failed to read response body");

    let body_json: Value =
        serde_json::from_slice(&body_bytes).expect("failed to parse JSON response");

    let stats = body_json.get("stats").expect("missing stats object");
    assert_eq!(
        stats.get("successful_renders").unwrap().as_u64().unwrap(),
        0,
        "new renderer should have 0 successful renders"
    );
    assert_eq!(
        stats.get("failed_renders").unwrap().as_u64().unwrap(),
        0,
        "new renderer should have 0 failed renders"
    );
    assert_eq!(
        stats.get("total_image_data_sent_bytes").unwrap().as_u64().unwrap(),
        0,
        "new renderer should have 0 bytes sent"
    );
}
