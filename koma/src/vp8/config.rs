//! Configuration options and builder tools for WebP encoding.
//!
//! This module provides [`EncoderConfig`] to set compression parameters and
//! [`EncoderConfigBuilder`] for step-by-step configuration.

/// Configuration parameters for the lossy WebP encoder.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncoderConfig {
    /// Target image quality from `0.0` (lowest) to `100.0` (highest).
    pub quality: f32,
    /// Alpha transparency quality from `0` to `100`.
    pub alpha_quality: u8,
    /// Enables performance optimizations for flat-color regions and borders.
    pub fast_anime_shortcuts: bool,
}

impl EncoderConfig {
    /// Creates a new configuration with the specified quality.
    ///
    /// # Arguments
    ///
    /// * `quality` - The target quality factor from `0.0` to `100.0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use koma::EncoderConfig;
    ///
    /// let config = EncoderConfig::new(90.0);
    /// assert_eq!(config.quality, 90.0);
    /// ```
    pub const fn new(quality: f32) -> Self {
        Self { quality, alpha_quality: 85, fast_anime_shortcuts: true }
    }

    /// Returns a builder to configure encoding options step by step.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use koma::EncoderConfig;
    ///
    /// let config = EncoderConfig::builder()
    ///     .quality(92.0)
    ///     .alpha_quality(100)
    ///     .build();
    /// ```
    pub fn builder() -> EncoderConfigBuilder {
        EncoderConfigBuilder::default()
    }
}

impl Default for EncoderConfig {
    /// Returns the default configuration with a quality factor of `85.0`.
    fn default() -> Self {
        Self { quality: 85.0, alpha_quality: 85, fast_anime_shortcuts: true }
    }
}

/// A builder for constructing an [`EncoderConfig`].
#[derive(Debug, Clone, Copy, Default)]
pub struct EncoderConfigBuilder {
    quality: Option<f32>,
    alpha_quality: Option<u8>,
    fast_anime_shortcuts: Option<bool>,
}

impl EncoderConfigBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the target lossy image quality.
    ///
    /// # Arguments
    ///
    /// * `quality` - A value from `0.0` (lowest) to `100.0` (highest).
    pub fn quality(mut self, quality: f32) -> Self {
        self.quality = Some(quality);
        self
    }

    /// Sets the quality for the alpha transparency channel.
    ///
    /// # Arguments
    ///
    /// * `alpha_quality` - A value from `0` to `100`.
    pub fn alpha_quality(mut self, alpha_quality: u8) -> Self {
        self.alpha_quality = Some(alpha_quality);
        self
    }

    /// Toggles fast flat-field shortcuts for uniform graphics.
    ///
    /// # Arguments
    ///
    /// * `enabled` - `true` to enable optimizations, `false` to disable.
    pub fn fast_anime_shortcuts(mut self, enabled: bool) -> Self {
        self.fast_anime_shortcuts = Some(enabled);
        self
    }

    /// Builds and returns the final [`EncoderConfig`].
    pub fn build(self) -> EncoderConfig {
        EncoderConfig {
            quality: self.quality.unwrap_or(85.0),
            alpha_quality: self.alpha_quality.unwrap_or(85),
            fast_anime_shortcuts: self.fast_anime_shortcuts.unwrap_or(true),
        }
    }
}
