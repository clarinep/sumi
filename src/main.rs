mod config;
mod renderer;
mod routes;

use std::{sync::Arc, time::Duration};

use axum::{routing::get, Router};
use tower_http::timeout::TimeoutLayer;

use crate::renderer::CardRenderer;

// we use microsoft mimalloc as it handles memory better
// it will only help when tokio is running multi threads
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("RUST_LOG", "info");
    pretty_env_logger::init();

    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

    log::info!("found {} cores - configuring..", cores);

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        // limit tokio threads so server dont die.
        .max_blocking_threads(cores)
        .build()
        .unwrap()
        .block_on(async {
            let cfg = config::Config::load();
            log::info!("!! starting sumi on port {}", cfg.port);

            let renderer = CardRenderer::new(cfg.cards_dir.to_string_lossy().to_string());
            let state = Arc::new(renderer);

            let app = Router::new()
                .route("/health", get(|| async { "OK" }))
                .route("/metrics", get(routes::handle_metrics))
                .route("/render/drop", get(routes::handle_render_drop))
                .with_state(state)
                .layer(TimeoutLayer::with_status_code(
                    axum::http::StatusCode::REQUEST_TIMEOUT,
                    Duration::from_secs(30),
                ));

            let addr = std::net::SocketAddr::from(([127, 0, 0, 1], cfg.port));
            let listener = tokio::net::TcpListener::bind(addr).await?;

            log::info!("server ready at http://{}", addr);

            axum::serve(listener, app).with_graceful_shutdown(shutdown()).await?;

            Ok(())
        })
}

async fn shutdown() {
    tokio::signal::ctrl_c().await.expect("ctrl+c");
    log::info!("you gave blair way too much caffeine..");
}
