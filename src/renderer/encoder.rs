use bytes::Bytes;
use koma::EncoderConfig;

use crate::renderer::error::{RenderError, Result};

pub const WEBP_QUALITY: f32 = 85.0;
pub const WEBP_ALPHA_QUALITY: u8 = 85;

/// Encodes raw RGBA pixel buffer into compressed WebP bytes via memory-pooled Koma encoder.
/// Decodable by standard libwebp / webpx decoders.
pub(super) fn encode_webp(width: u32, height: u32, pixel_data: &[u8]) -> Result<Bytes> {
    let config = EncoderConfig::builder()
        .quality(WEBP_QUALITY)
        .alpha_quality(WEBP_ALPHA_QUALITY)
        .fast_anime_shortcuts(true)
        .build();

    koma::encode_rgba_webp(width, height, pixel_data, &config)
        .map_err(|err| RenderError::EncodeError(err.to_string()))
}
