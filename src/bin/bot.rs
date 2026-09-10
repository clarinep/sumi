use std::{borrow::Cow, env, sync::Arc, time::Instant};

use mimalloc::MiMalloc;
use poise::serenity_prelude as serenity;
use sumi::renderer::CardRenderer;

#[global_allocator]
static ALLOC: MiMalloc = MiMalloc;

struct Data {
    renderer: Arc<CardRenderer>,
}

type Error = Box<dyn std::error::Error + Send + Sync>;
type Context<'a> = poise::Context<'a, Data, Error>;

async fn autocomplete_card<'a>(
    ctx: Context<'_>,
    partial: &'a str,
) -> impl Iterator<Item = String> + 'a {
    let names = ctx.data().renderer.card_cache.card_names();
    names
        .into_iter()
        .filter(move |name| name.to_lowercase().contains(&partial.to_lowercase()))
        .take(25)
        .map(|name| name.to_string())
}

#[poise::command(slash_command, prefix_command)]
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

    let t_render = Instant::now();
    let (bytes, c1, c2, p1, p2) = ctx
        .data()
        .renderer
        .render_random_drop(left.as_deref(), right.as_deref())
        .await?;
    let render_us = t_render.elapsed().as_micros();
    let size_kb = bytes.len() as f64 / 1024.0;
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let attachment = serenity::CreateAttachment::bytes(Cow::Borrowed(&bytes[..]), "drop.webp");

    let text = format!(
        "[sumi::drop]\nslot.0: {c1} (print: #{p1})\nslot.1: {c2} (print: #{p2})\nrender: {render_us}µs | size: {size_kb:.1}KB | roundtrip: {total_ms:.1}ms"
    );

    ctx.send(
        poise::CreateReply::default()
            .content(format!("```{text}```"))
            .attachment(attachment),
    )
    .await?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let token = env::var("DISCORD_TOKEN").expect("missing DISCORD_TOKEN env variable");
    let cards_dir = env::var("CARDS_DIR").unwrap_or_else(|_| "assets".to_string());

    let renderer = Arc::new(CardRenderer::new(&cards_dir)?);
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
        .setup(|ctx, _ready, framework| {
            Box::pin(async move {
                poise::builtins::register_globally(ctx, &framework.options().commands).await?;
                ctx.set_activity(Some(serenity::ActivityData::custom("soft testing in progress (*ᴗ͈ˬᴗ͈)ꕤ*.ﾟ")));
                println!("sumi-bot online: {}", _ready.user.tag());
                Ok(Data { renderer })
            })
        })
        .build();

    let intents =
        serenity::GatewayIntents::non_privileged() | serenity::GatewayIntents::MESSAGE_CONTENT;

    let client = serenity::ClientBuilder::new(token, intents)
        .framework(framework)
        .await;

    client?.start().await?;
    Ok(())
}
