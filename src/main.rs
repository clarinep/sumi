mod config;
mod renderer;
mod routes;

use axum::{routing::get, serve, Router};
use mimalloc::MiMalloc;
use pretty_env_logger::init as init_logger;
use std::{env, error::Error, net::SocketAddr, sync::Arc};
use tokio::{net::TcpListener, runtime, signal};

use crate::{
    config::Config,
    renderer::{canvas::init_font, CardRenderer},
    routes::{handle_metrics, handle_render_drop},
};

// we use microsoft mimalloc as it handles memory better
// it will only help when tokio is running multi threads
#[global_allocator]
static ALLOC: MiMalloc = MiMalloc;

fn main() -> Result<(), Box<dyn Error>> {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "info");
    }
    init_logger();

    runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime")
        .block_on(async {
            let cfg = Config::load();
            log::info!("!! starting sumi on port {}", cfg.port);

            log::info!("baking in lexend deca font..");
            init_font();

            let renderer =
                CardRenderer::new(cfg.cards_dir.clone()).expect("failed to wake sumi up..");
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

            log::info!("server ready at http://{addr}");

            serve(listener, app).with_graceful_shutdown(shutdown()).await?;

            log::info!("sumi is draining tasks, please wait..");
            state.wait_for_tasks_to_finish().await;

            Ok(())
        })
}

async fn shutdown() {
    signal::ctrl_c().await.expect("ctrl+c");
    log::info!("you gave sumi way too much caffeine..");
}
