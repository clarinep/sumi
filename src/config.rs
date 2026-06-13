use std::{env, path::PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub cards_dir: PathBuf,
}

impl Config {
    pub fn load() -> Self {
        // read from env so its easier to setup to docker etc.
        // you can pass CARDS_DIR env before running
        // for now we will use our default path and default port 8888
        let home = env::var_os("USERPROFILE").or_else(|| env::var_os("HOME"));
        let default_cards_dir = home.map(PathBuf::from).map_or_else(
            || PathBuf::from("assets/cards"),
            |p| p.join("Documents").join("huty").join("cards"),
        );

        let cards_dir = env::var("CARDS_DIR").map_or_else(|_| default_cards_dir, PathBuf::from);

        let port = env::var("PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(8888);

        tracing::info!(
            "config loaded\n      port: [{}]\n      card: [{}]",
            port,
            cards_dir.display()
        );

        Self { port, cards_dir }
    }
}
