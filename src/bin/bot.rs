use std::{
    borrow::Cow,
    env, fs,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use mimalloc::MiMalloc;
use poise::serenity_prelude as serenity;
use sumi::{
    metrics::{ImageBytes, RenderDurationMs},
    renderer::{CardRenderer, PrintNumber},
};
use webpx::{AlphaFilter, EncoderConfig, ImageHint, Preset, Unstoppable};

#[global_allocator]
static ALLOC: MiMalloc = MiMalloc;

pub struct Data {
    renderer: Arc<CardRenderer>,
    cards: Arc<[String]>,
}

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Context<'a> = poise::Context<'a, Data, Error>;

fn format_duration(d: Duration) -> String {
    let us = d.as_micros();
    if us < 1_000 {
        format!("{us}µs")
    } else if us < 1_000_000 {
        format!("{:.2}ms", us as f64 / 1_000.0)
    } else {
        format!("{:.2}s", us as f64 / 1_000_000.0)
    }
}

fn format_uptime(d: Duration) -> String {
    let total_secs = d.as_secs();
    let days = total_secs / 86400;
    let hours = (total_secs % 86400) / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    if days > 0 {
        format!("{days}d {hours}h {minutes}m {seconds}s")
    } else if hours > 0 {
        format!("{hours}h {minutes}m {seconds}s")
    } else if minutes > 0 {
        format!("{minutes}m {seconds}s")
    } else {
        format!("{seconds}s")
    }
}

fn scan_card_names(dir: &str) -> Vec<String> {
    let mut names = Vec::new();
    scan_dir(Path::new(dir), Path::new(dir), &mut names);
    names
}

fn scan_dir(base: &Path, current: &Path, acc: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(current) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_dir(base, &path, acc);
        } else if path.extension().is_some_and(|ext| ext == "webp") {
            if let Ok(rel) = path.strip_prefix(base) {
                let name = rel.with_extension("").to_string_lossy().replace('\\', "/");
                acc.push(name);
            }
        }
    }
}

async fn autocomplete_card(
    ctx: Context<'_>,
    partial: &str,
) -> Vec<String> {
    let partial = partial.to_lowercase();
    ctx.data()
        .cards
        .iter()
        .filter(|name| name.to_lowercase().contains(&partial))
        .take(25)
        .cloned()
        .collect()
}

fn encode_custom_quality(bytes: &[u8], quality: u32) -> Option<Vec<u8>> {
    let (pixels, width, height) = webpx::decode_rgba(bytes).ok()?;
    let config = EncoderConfig::new()
        .preset(Preset::Picture)
        .hint(ImageHint::Picture)
        .quality((quality as f32).clamp(1.0, 100.0))
        .method(0)
        .alpha_compression(true)
        .alpha_quality(85)
        .alpha_filter(AlphaFilter::None)
        .low_memory(false)
        .pass(1)
        .exact(false)
        .segments(1);

    config.encode_rgba(&pixels, width, height, Unstoppable).ok()
}

#[poise::command(slash_command, prefix_command, aliases("d"))]
async fn drop(
    ctx: Context<'_>,
    #[autocomplete = "autocomplete_card"]
    #[description = "slot 0 card identifier"]
    left: Option<String>,
    #[autocomplete = "autocomplete_card"]
    #[description = "slot 1 card identifier"]
    right: Option<String>,
    #[description = "number of drops to generate (1-10 max files)"]
    amount: Option<u32>,
    #[description = "webpx encoder qual 1-100 (default 85)"]
    quality: Option<u32>,
) -> Result<(), Error> {
    let cards = &ctx.data().cards;
    if cards.is_empty() {
        ctx.send(
            poise::CreateReply::default()
                .content("```ansi\n\x1b[1;31m✖ no .webp card assets found in assets/ directory\x1b[0m\n```")
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }

    let count = amount.unwrap_or(1).clamp(1, 10);
    let mut attachments = Vec::with_capacity(count as usize);
    let mut total_render_time = Duration::ZERO;
    let mut total_bytes = 0usize;

    for i in 0..count {
        let c1 = if i == 0 {
            left.clone().unwrap_or_else(|| {
                let idx = fastrand::usize(..cards.len());
                cards[idx].clone()
            })
        } else {
            let idx = fastrand::usize(..cards.len());
            cards[idx].clone()
        };

        let c2 = if i == 0 && right.is_some() {
            right.clone().unwrap()
        } else if cards.len() > 1 {
            loop {
                let idx = fastrand::usize(..cards.len());
                if cards[idx] != c1 {
                    break cards[idx].clone();
                }
            }
        } else {
            cards[0].clone()
        };

        let p1 = fastrand::u32(1..=9);
        let p2 = fastrand::u32(1..=9);

        let t_render = Instant::now();
        let bytes = ctx
            .data()
            .renderer
            .render_drop(&c1, &c2, PrintNumber::new(p1), PrintNumber::new(p2))
            .await?;

        let mut final_bytes = bytes.to_vec();
        if let Some(q) = quality {
            if (1..=100).contains(&q) && q != 85 {
                if let Some(reencoded) = encode_custom_quality(&bytes, q) {
                    final_bytes = reencoded;
                }
            }
        }

        let render_time = t_render.elapsed();
        total_render_time += render_time;
        total_bytes += final_bytes.len();

        ctx.data().renderer.stats.record_success(
            ImageBytes(final_bytes.len() as u64),
            RenderDurationMs(render_time.as_millis() as u64),
        );

        let filename = if count == 1 {
            "drop.webp".to_string()
        } else {
            format!("drop_{}.webp", i + 1)
        };

        attachments.push(serenity::CreateAttachment::bytes(
            Cow::Owned(final_bytes),
            filename,
        ));
    }

    let render_fmt = format_duration(total_render_time);
    let size_kb = total_bytes as f64 / 1024.0;

    let msg = if count == 1 {
        format!(
            "```ansi\n\x1b[0;34mRender:\x1b[0m \x1b[1;37m{render_fmt}\x1b[0m  \x1b[0;30m•\x1b[0m  \x1b[0;33mSize:\x1b[0m \x1b[1;37m{size_kb:.1} KB\x1b[0m\n```"
        )
    } else {
        let avg_fmt = format_duration(total_render_time / count);
        format!(
            "```ansi\n\x1b[0;36mDrops:\x1b[0m \x1b[1;37m{count}x\x1b[0m  \x1b[0;30m•\x1b[0m  \x1b[0;34mRender:\x1b[0m \x1b[1;37m{render_fmt}\x1b[0m \x1b[0;30m(avg {avg_fmt})\x1b[0m  \x1b[0;30m•\x1b[0m  \x1b[0;33mSize:\x1b[0m \x1b[1;37m{size_kb:.1} KB\x1b[0m\n```"
        )
    };

    let mut reply_builder = poise::CreateReply::default().content(msg);
    for att in attachments {
        reply_builder = reply_builder.attachment(att);
    }

    ctx.send(reply_builder).await?;
    Ok(())
}

#[poise::command(slash_command, prefix_command, aliases("s", "benchmark", "bench", "metrics"))]
async fn stats(ctx: Context<'_>) -> Result<(), Error> {
    let renderer = &ctx.data().renderer;
    let uptime = renderer.start_time.elapsed();
    let uptime_fmt = format_uptime(uptime);

    let successful = renderer.stats.successful_renders();
    let failed = renderer.stats.failed_renders();
    let total_renders = successful + failed;
    let total_bytes = renderer.stats.total_image_bytes();
    let total_time_ms = renderer.stats.total_render_time_ms();

    let avg_render_ms = if successful > 0 {
        total_time_ms as f64 / successful as f64
    } else {
        0.0
    };

    let uptime_secs = uptime.as_secs_f64().max(0.001);
    let throughput_rps = successful as f64 / uptime_secs;

    let size_mb = total_bytes as f64 / (1024.0 * 1024.0);
    let success_rate = if total_renders > 0 {
        (successful as f64 / total_renders as f64) * 100.0
    } else {
        100.0
    };

    let ping = ctx.ping().await;
    let ping_fmt = format_duration(ping);
    let cards_count = ctx.data().cards.len();

    let text = format!(
        "```ansi
\x1b[0;32m  Uptime        \x1b[0m : \x1b[1;37m{uptime_fmt}\x1b[0m\n\
\x1b[0;32m  Indexed Cards \x1b[0m : \x1b[1;37m{cards_count} cards\x1b[0m \x1b[0;30m(in-memory cache)\x1b[0m\n\
\x1b[0;32m  Gateway Ping  \x1b[0m : \x1b[1;37m{ping_fmt}\x1b[0m\n\
\x1b[0;35m  Total Renders \x1b[0m : \x1b[1;37m{successful}\x1b[0m \x1b[0;30m({success_rate:.1}% success, {failed} failed)\x1b[0m\n\
\x1b[0;35m  Average Speed \x1b[0m : \x1b[1;37m{avg_render_ms:.2} ms\x1b[0m \x1b[0;30mper drop composite\x1b[0m\n\
\x1b[0;35m  Throughput    \x1b[0m : \x1b[1;37m{throughput_rps:.2} drops/sec\x1b[0m\n\
\x1b[0;35m  Total Render  \x1b[0m : \x1b[1;37m{:.2} s\x1b[0m \x1b[0;30mcumulative CPU time\x1b[0m\n\
\x1b[0;35m  Total Output  \x1b[0m : \x1b[1;37m{size_mb:.2} MB\x1b[0m \x1b[0;30mWebP payload\x1b[0m\n\
```",
        total_time_ms as f64 / 1000.0
    );

    ctx.send(poise::CreateReply::default().content(text)).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    println!("[sumi::init] booting bot process...");

    let token = match env::var("DISCORD_TOKEN") {
        Ok(t) if !t.trim().is_empty() => t.trim().to_string(),
        _ => {
            eprintln!("[sumi::fatal] DISCORD_TOKEN secret is empty or not set in GitHub repository settings!");
            std::process::exit(1);
        }
    };

    let cards_dir = env::var("CARDS_DIR").unwrap_or_else(|_| "assets".to_string());
    println!("[sumi::init] scanning cards from directory: {cards_dir}");

    let card_names = scan_card_names(&cards_dir);
    let count = card_names.len();
    println!("[sumi::init] indexed {count} cards");
    let cards: Arc<[String]> = card_names.into();

    let renderer = match CardRenderer::new(&cards_dir) {
        Ok(r) => Arc::new(r),
        Err(e) => {
            eprintln!("[sumi::fatal] failed to initialize CardRenderer: {e}");
            std::process::exit(1);
        }
    };
    renderer.card_cache.start_prewarm();

    let options = poise::FrameworkOptions {
        commands: vec![drop(), stats()],
        prefix_options: poise::PrefixFrameworkOptions {
            prefix: Some("b".into()),
            additional_prefixes: vec![
                poise::Prefix::Literal("b "),
                poise::Prefix::Literal("B"),
                poise::Prefix::Literal("B "),
            ],
            ..Default::default()
        },
        ..Default::default()
    };

    let framework = poise::Framework::builder()
        .options(options)
        .setup(move |ctx, _ready, framework| {
            Box::pin(async move {
                poise::builtins::register_globally(ctx, &framework.options().commands).await?;
                ctx.set_presence(
                    Some(serenity::ActivityData::custom(
                        "soft testing in progress (*ᴗ͈ˬᴗ͈)ꕤ*.ﾟ",
                    )),
                    serenity::OnlineStatus::Online,
                );
                println!("========================================");
                println!("[sumi::ready] tag: {} | cards: {count}", _ready.user.tag());
                println!("========================================");
                Ok(Data { renderer, cards })
            })
        })
        .build();

    let intents =
        serenity::GatewayIntents::non_privileged() | serenity::GatewayIntents::MESSAGE_CONTENT;

    println!("[sumi::init] connecting to discord gateway...");

    let mut client = match serenity::ClientBuilder::new(token, intents)
        .framework(framework)
        .await
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[sumi::fatal] client builder error: {e}");
            std::process::exit(1);
        }
    };

    if let Err(why) = client.start().await {
        eprintln!("[sumi::fatal] gateway connection error: {why:?}");
    }

    Ok(())
}
