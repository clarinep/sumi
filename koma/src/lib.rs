//! # Koma WebP Lossy & Alpha Encoder
//!
//! Ground-up, pure-Rust, zero-copy, SIMD-accelerated WebP encoder tuned for
//! anime art, UI assets, and high-throughput rendering.

pub mod alpha;
pub mod color;
pub mod config;
pub mod error;
pub mod pool;
pub mod riff;
pub mod vp8;

pub use config::{AlphaFilter, EncoderConfig, EncoderConfigBuilder, Preset};
pub use error::{KomaError, Result};
pub use pool::{BufferPool, with_thread_scratch};

use bytes::Bytes;

/// Encodes raw RGBA (8-bit per channel) pixel buffer into compressed WebP format.
pub fn encode_rgba_webp(
    width: u32,
    height: u32,
    pixel_data: &[u8],
    config: &EncoderConfig,
) -> Result<Bytes> {
    if width == 0 || height == 0 || width > 16383 || height > 16383 {
        return Err(KomaError::InvalidDimensions { width, height });
    }

    let expected_len = (width as usize) * (height as usize) * 4;
    if pixel_data.len() != expected_len {
        return Err(KomaError::InvalidBufferSize {
            expected: expected_len,
            actual: pixel_data.len(),
        });
    }

    // Step 1: Color space conversion and alpha extraction
    let planar = color::rgba_to_yuv420(pixel_data, width as usize, height as usize);

    // Step 2: Generate optional ALPH chunk if transparency is present
    let alph_chunk = if planar.has_alpha && config.alpha_compression {
        if let Some(ref alpha_plane) = planar.alpha {
            Some(alpha::AlphaEncoder::encode_alpha(
                alpha_plane,
                width as usize,
                height as usize,
                config.alpha_quality,
                config.alpha_filter,
            ))
        } else {
            None
        }
    } else {
        None
    };

    // Step 3: Pure-Rust In-Tree Lossy VP8 Keyframe Encoding
    let quantizer = vp8::quant::Quantizer::from_quality_and_method(config.quality, config.method);
    let vp8_payload = vp8::frame::encode_frame(&planar, &quantizer);

    // Step 4: Package into standard RIFF WebP container (VP8X + ALPH + VP8)
    let webp_bytes = riff::package_webp_riff(
        width,
        height,
        &vp8_payload,
        alph_chunk.as_deref(),
    );

    Ok(webp_bytes)
}

/// Encodes raw RGB (8-bit per channel, 24bpp) pixel buffer into compressed WebP format.
pub fn encode_rgb_webp(
    width: u32,
    height: u32,
    pixel_data: &[u8],
    config: &EncoderConfig,
) -> Result<Bytes> {
    if width == 0 || height == 0 || width > 16383 || height > 16383 {
        return Err(KomaError::InvalidDimensions { width, height });
    }

    let expected_len = (width as usize) * (height as usize) * 3;
    if pixel_data.len() != expected_len {
        return Err(KomaError::InvalidBufferSize {
            expected: expected_len,
            actual: pixel_data.len(),
        });
    }

    // Convert RGB to RGBA for internal pipeline
    let mut rgba = Vec::with_capacity((width as usize) * (height as usize) * 4);
    for chunk in pixel_data.chunks_exact(3) {
        rgba.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
    }

    encode_rgba_webp(width, height, &rgba, config)
}
