use std::{env, path::PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    pub cards_dir: PathBuf,
    pub port: u16,
}

impl Config {
    pub fn from_env() -> Self {
        // read from env so its easier to setup to docker etc.
        // you can pass CARDS_DIR env before running
        // for now we will use our default path and default port 8888
        let home = env::var_os("USERPROFILE").or_else(|| env::var_os("HOME"));
        let default_cards_dir = home.map(PathBuf::from).map_or_else(
            || PathBuf::from("assets/cards"),
            |p| p.join("Documents").join("kizunari").join("cards"),
        );

        let cards_dir = env::var("CARDS_DIR").map_or_else(|_| default_cards_dir, PathBuf::from);

        let port = match env::var("PORT") {
            Ok(s) => s.parse().unwrap_or_else(|_| {
                tracing::error!(
                    "failed to load config..\n      reason: PORT is not a valid u16 port number"
                );
                std::process::exit(1);
            }),
            Err(env::VarError::NotPresent) => 8888,
            Err(e) => {
                tracing::error!(
                    "failed to load config..\n      reason: failed to read PORT env ({})",
                    e
                );
                std::process::exit(1);
            }
        };

        tracing::info!(
            "config loaded\n      port: [{}]\n      card: [{}]",
            port,
            cards_dir.display()
        );

        Self { cards_dir, port }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::from_env()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Custom RAII guard for environment variables to ensure test isolation
    struct EnvGuard {
        key: String,
        original: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &str, value: &str) -> Self {
            let original = std::env::var(key).ok();
            unsafe { std::env::set_var(key, value) };
            EnvGuard { key: key.to_string(), original }
        }

        fn remove(key: &str) -> Self {
            let original = std::env::var(key).ok();
            unsafe { std::env::remove_var(key) };
            EnvGuard { key: key.to_string(), original }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(v) => unsafe { std::env::set_var(&self.key, v) },
                None => unsafe { std::env::remove_var(&self.key) },
            }
        }
    }

    #[test]
    fn test_config_default_port() {
        let _guard = EnvGuard::remove("PORT");
        let config = Config::from_env();
        assert_eq!(config.port, 8888);
    }

    #[test]
    fn test_config_custom_port() {
        let _guard = EnvGuard::set("PORT", "9999");
        let config = Config::from_env();
        assert_eq!(config.port, 9999);
    }

    #[test]
    fn test_config_custom_cards_dir() {
        let _guard = EnvGuard::set("CARDS_DIR", "/tmp/custom_cards");
        let config = Config::from_env();
        assert_eq!(config.cards_dir, PathBuf::from("/tmp/custom_cards"));
    }
}
