mod config;
mod logger;
mod renderer;
mod routes;

use std::{env, error::Error, net::SocketAddr, sync::Arc};

use axum::{routing::get, serve, Router};
use mimalloc::MiMalloc;
use tokio::{net::TcpListener, signal};
use tracing_subscriber::EnvFilter;

use crate::{
    config::Config,
    renderer::{print::init_font, CardRenderer},
    routes::{handle_metrics, handle_render_drop},
};

// we use microsoft mimalloc as it handles memory better
// it will only help when tokio is running multi threads
#[global_allocator]
static ALLOC: MiMalloc = MiMalloc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "info");
    }

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .event_format(logger::LogFormatter)
        .init();

    let sumi_asc = include_str!("ascii.txt");
    println!("\n\x1b[38;5;62m{sumi_asc}\x1b[0m");

    let cfg = Config::load();
    tracing::info!("!! starting sumi on port {}", cfg.port);
    tracing::info!("baking in lexend deca font..");
    init_font();

    let renderer = CardRenderer::new(cfg.cards_dir.clone()).expect("failed to wake sumi up..");
    let state = Arc::new(renderer);

    // prewarm cache di belakang
    state.card_cache.start_prewarm();

    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/metrics", get(handle_metrics))
        .route("/render/drop", get(handle_render_drop))
        .with_state(state.clone());

    let addr = SocketAddr::from(([127, 0, 0, 1], cfg.port));
    let listener = TcpListener::bind(addr).await?;

    tracing::info!("sumi ready at http://{addr}");

    serve(listener, app).with_graceful_shutdown(shutdown()).await?;

    tracing::info!("sumi is draining tasks, please wait..");
    state.wait_for_tasks_to_finish().await;

    Ok(())
}

async fn shutdown() {
    signal::ctrl_c().await.expect("ctrl+c");
    tracing::info!("you gave sumi way too much caffeine..");
}
