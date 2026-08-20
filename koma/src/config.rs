//! Encoding configuration for Koma lossy and lossless WebP pipelines.

/// Image content preset for tuning VP8 rate-distortion and filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Preset {
    /// Default all-around tuning.
    #[default]
    Default,
    /// Tuned for anime, illustrations, and line art (preserves fine lines and gradients).
    Drawing,
    /// Tuned for natural photography.
    Photo,
    /// Tuned for graphics, icons, and UI elements.
    Icon,
    /// Tuned for text, documents, and screenshots.
    Text,
}

/// Alpha filtering strategy for WebP ALPH chunk generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlphaFilter {
    /// Automatically evaluate all spatial predictors (Horizontal, Vertical, Gradient) and pick the best.
    #[default]
    Auto,
    /// No filtering (maximizes alpha encoding throughput, zero CPU overhead).
    None,
    /// Horizontal predictor.
    Horizontal,
    /// Vertical predictor.
    Vertical,
    /// Gradient predictor.
    Gradient,
}

/// Configuration settings for the Koma WebP encoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncoderConfig {
    /// Quality factor from `0.0` (smallest file size) to `100.0` (highest visual fidelity).
    /// Default is `95.0` for high-fidelity anime art rendering.
    pub quality: f32,
    /// Alpha compression quality from `0` to `100`.
    /// Default is `100` (lossless alpha preservation).
    pub alpha_quality: u8,
    /// Whether alpha compression is enabled.
    pub alpha_compression: bool,
    /// Alpha spatial prediction filter selection (None, Auto, Horizontal, Vertical, Gradient).
    pub alpha_filter: AlphaFilter,
    /// Compression method / effort from `0` (fastest) to `6` (strongest compression).
    /// Default is `4` (optimal balance between speed and quality).
    pub method: u8,
    /// Image content preset.
    pub preset: Preset,
    /// Whether to preserve exact RGB values for fully transparent pixels (prevents color bleed).
    pub exact: bool,
    /// Whether to force lossless VP8L encoding.
    pub lossless: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            quality: 95.0,
            alpha_quality: 100,
            alpha_compression: true,
            alpha_filter: AlphaFilter::Auto,
            method: 4,
            preset: Preset::Drawing,
            exact: true,
            lossless: false,
        }
    }
}

impl EncoderConfig {
    /// Creates a new `EncoderConfigBuilder` with defaults tuned for high quality.
    #[inline]
    pub fn builder() -> EncoderConfigBuilder {
        EncoderConfigBuilder::new()
    }
}

/// Builder for constructing an [`EncoderConfig`].
#[derive(Debug, Clone)]
pub struct EncoderConfigBuilder {
    config: EncoderConfig,
}

impl Default for EncoderConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EncoderConfigBuilder {
    /// Creates a new builder with default settings.
    #[inline]
    pub fn new() -> Self {
        Self {
            config: EncoderConfig::default(),
        }
    }

    /// Sets the target lossy quality factor `[0.0, 100.0]`.
    #[inline]
    pub fn quality(mut self, quality: f32) -> Self {
        self.config.quality = quality.clamp(0.0, 100.0);
        self
    }

    /// Sets the alpha channel compression quality `[0, 100]`.
    #[inline]
    pub fn alpha_quality(mut self, alpha_quality: u8) -> Self {
        self.config.alpha_quality = alpha_quality.min(100);
        self
    }

    /// Sets whether alpha compression is enabled.
    #[inline]
    pub fn alpha_compression(mut self, alpha_compression: bool) -> Self {
        self.config.alpha_compression = alpha_compression;
        self
    }

    /// Sets the alpha spatial filter strategy (None, Auto, Horizontal, Vertical, Gradient).
    #[inline]
    pub fn alpha_filter(mut self, alpha_filter: AlphaFilter) -> Self {
        self.config.alpha_filter = alpha_filter;
        self
    }

    /// Sets the compression effort method `[0, 6]`.
    #[inline]
    pub fn method(mut self, method: u8) -> Self {
        self.config.method = method.min(6);
        self
    }

    /// Sets the encoding content preset.
    #[inline]
    pub fn preset(mut self, preset: Preset) -> Self {
        self.config.preset = preset;
        self
    }

    /// Controls whether transparent RGB values are preserved exactly.
    #[inline]
    pub fn exact(mut self, exact: bool) -> Self {
        self.config.exact = exact;
        self
    }

    /// Sets whether to use lossless encoding.
    #[inline]
    pub fn lossless(mut self, lossless: bool) -> Self {
        self.config.lossless = lossless;
        self
    }

    /// Builds and returns the final [`EncoderConfig`].
    #[inline]
    pub fn build(self) -> EncoderConfig {
        self.config
    }
}
