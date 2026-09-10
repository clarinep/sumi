use std::{
    borrow::Cow,
    env, fs,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use mimalloc::MiMalloc;
use poise::serenity_prelude as serenity;
use sumi::renderer::{CardRenderer, PrintNumber};

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

#[poise::command(slash_command, prefix_command, aliases("d"))]
async fn drop(
    ctx: Context<'_>,
    #[autocomplete = "autocomplete_card"]
    #[description = "slot 0 card identifier"]
    left: Option<String>,
    #[autocomplete = "autocomplete_card"]
    #[description = "slot 1 card identifier"]
    right: Option<String>,
) -> Result<(), Error> {
    let t0 = Instant::now();
    ctx.defer().await?;

    let cards = &ctx.data().cards;
    if cards.is_empty() {
        ctx.send(
            poise::CreateReply::default()
                .content("*no .webp card assets found in assets/*")
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }

    let c1 = left.unwrap_or_else(|| {
        let idx = fastrand::usize(..cards.len());
        cards[idx].clone()
    });

    let c2 = right.unwrap_or_else(|| {
        if cards.len() > 1 {
            loop {
                let idx = fastrand::usize(..cards.len());
                if cards[idx] != c1 {
                    break cards[idx].clone();
                }
            }
        } else {
            cards[0].clone()
        }
    });

    let p1 = fastrand::u32(1..=9);
    let p2 = fastrand::u32(1..=9);

    let t_render = Instant::now();
    let bytes = ctx
        .data()
        .renderer
        .render_drop(&c1, &c2, PrintNumber::new(p1), PrintNumber::new(p2))
        .await?;
    let render_time = t_render.elapsed();
    let size_kb = bytes.len() as f64 / 1024.0;
    let roundtrip = t0.elapsed();

    let render_fmt = format_duration(render_time);
    let roundtrip_fmt = format_duration(roundtrip);

    let attachment = serenity::CreateAttachment::bytes(Cow::Borrowed(&bytes[..]), "drop.webp");

    let text = format!(
        "`#{p1}`  |  `#{p2}`  •  `{render_fmt}` · `{size_kb:.1}KB` · `{roundtrip_fmt}`"
    );

    ctx.send(
        poise::CreateReply::default()
            .content(text)
            .attachment(attachment),
    )
    .await?;

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
        commands: vec![drop()],
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
