mod config;
mod logger;
mod renderer;
mod routes;

use std::{error::Error, fmt::Write, net::SocketAddr, panic, sync::Arc};

use axum::{Router, routing::get, serve};
use mimalloc::MiMalloc;
use tokio::{net::TcpListener, signal};
use tracing_subscriber::EnvFilter;

use crate::{
    config::Config,
    renderer::{CardRenderer, print::init_font},
    routes::{handle_metrics, handle_render_drop},
};

// aegis sets up a panic hook so we can format sys errors cleanly
// as unexpected panics will give long unformatted backtraces.
fn aegis() {
    panic::set_hook(Box::new(|info| {
        let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
            s
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.as_str()
        } else {
            "sumi just crashed ..?"
        };

        if let Some(loc) = info.location() {
            tracing::error!(
                "sumi got sleepy..\n      reason: {}\n      location: {}:{}",
                msg,
                loc.file(),
                loc.line()
            );
        } else {
            tracing::error!("sumi got sleepy..\n      reason: {}", msg);
        }
    }));
}

// we use microsoft mimalloc as it handles memory better
// it will only help when tokio is running multi threads
#[global_allocator]
static ALLOC: MiMalloc = MiMalloc;

#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    aegis();

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("sumi=debug,info"));

    tracing_subscriber::fmt().with_env_filter(filter).event_format(logger::LogFormatter).init();

    let welcomer = include_str!("ascii.txt");
    println!();

    let lines: Vec<&str> = welcomer.lines().collect();
    let num_lines = lines.len().max(1) as f32;

    // Sumi color palette: mint green transitioning into bright peach-orange
    let (r1, g1, b1) = (168.0_f32, 230.0_f32, 207.0_f32);
    let (r2, g2, b2) = (255.0_f32, 192.0_f32, 120.0_f32);

    for (y, line) in lines.into_iter().enumerate() {
        let mut styled_line = String::with_capacity(line.len() * 20);
        let num_chars = line.chars().count().max(1) as f32;

        for (x, ch) in line.chars().enumerate() {
            if ch == ' ' {
                styled_line.push(' ');
                continue;
            }
            
            // Smoother horizontal-heavy gradient
            let progress_x = x as f32 / num_chars;
            let progress_y = y as f32 / num_lines;
            let t = (progress_x * 0.8 + progress_y * 0.2).clamp(0.0, 1.0);

            // Simple linear interpolation
            let r = (r2 - r1).mul_add(t, r1) as u8;
            let g = (g2 - g1).mul_add(t, g1) as u8;
            let b = (b2 - b1).mul_add(t, b1) as u8;

            let _ = write!(styled_line, "\x1b[38;2;{r};{g};{b}m{ch}");
        }
        println!("{styled_line}\x1b[0m");
    }

    let cfg = Config::load();

    tracing::info!("baking in lexend deca font..");
    init_font();

    let renderer = match CardRenderer::new(cfg.cards_dir.clone()) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("failed to wake sumi up..\n      reason: {}", e);
            return Ok(());
        }
    };
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

    serve(listener, app).with_graceful_shutdown(nap()).await?;

    tracing::info!("sumi is going to sleep, finishing tasks..");
    state.wait_for_tasks_to_finish().await;

    Ok(())
}

async fn nap() {
    if let Err(e) = signal::ctrl_c().await {
        tracing::error!("sumi failed to sleep..?({})", e);
    }
    tracing::info!("you gave sumi way too much caffeine..");
}
