use bytes::Bytes;
use koma::EncoderConfig;

use crate::renderer::error::{RenderError, Result};

pub const WEBP_QUALITY: f32 = 85.0;
pub const WEBP_ALPHA_QUALITY: u8 = 85;

/// encodes raw rgba pixel buffer into compressed webp bytes via memory-pooled koma encoder.
/// decodable by standard libwebp / webpx decoders.
pub(super) fn encode_webp(width: u32, height: u32, pixel_data: &[u8]) -> Result<Bytes> {
    let config = EncoderConfig::builder()
        .quality(WEBP_QUALITY)
        .alpha_quality(WEBP_ALPHA_QUALITY)
        .fast_anime_shortcuts(true)
        .build();

    koma::encode_rgba_webp(width, height, pixel_data, &config)
        .map_err(|err| RenderError::EncodeError(err.to_string()))
}
