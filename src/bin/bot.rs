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
                ctx.set_activity(Some(serenity::ActivityData::custom(
                    "soft testing in progress (*ᴗ͈ˬᴗ͈)ꕤ*.ﾟ",
                )));
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
