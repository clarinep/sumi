//! # koma
//!
//! A high-performance, zero-allocation lossy WebP image encoder written in pure Rust.
//!
//! This library converts raw 32-bit RGBA pixel buffers into compliant WebP image files.
//! It is designed specifically for high-throughput image rendering pipelines.
//!
//! ## Key features
//!
//! - **Memory pooling**: Reuses pre-allocated scratch buffers to eliminate runtime heap allocations.
//! - **Standard compliance**: Produces bitstreams that comply with RFC 6386 and WebP container standards.
//! - **Thread safety**: Supports concurrent encoding across worker threads without lock contention.
//! - **Pure Rust**: Requires no external C or C++ build dependencies.
//!
//! ## Basic usage
//!
//! ```rust,no_run
//! use koma::{encode_rgba_webp, EncoderConfig, Result};
//!
//! fn main() -> Result<()> {
//!     let width = 512;
//!     let height = 768;
//!     let rgba_pixels = vec![255u8; (width * height * 4) as usize];
//!     let config = EncoderConfig::new(85.0);
//!
//!     let webp_bytes = encode_rgba_webp(width, height, &rgba_pixels, &config)?;
//!     assert!(!webp_bytes.is_empty());
//!
//!     Ok(())
//! }
//! ```


pub mod alpha;
pub mod color;
pub mod error;
pub mod pool;
pub mod riff;
pub mod vp8;

#[doc(inline)]
pub use error::{KomaError, Result};
#[doc(inline)]
pub use pool::{EncoderScratch, ENCODER_SCRATCH_POOL};
#[doc(inline)]
pub use riff::assemble_webp;
#[doc(inline)]
pub use vp8::{encode_lossy_frame, EncoderConfig, EncoderConfigBuilder};

/// Encodes an RGBA byte slice into a lossy WebP image.
///
/// This function retrieves a scratch buffer from the global memory pool to avoid
/// memory allocation overhead during image generation.
///
/// # Arguments
///
/// * `width` - The width of the image in pixels (1 to 16,383).
/// * `height` - The height of the image in pixels (1 to 16,383).
/// * `rgba` - A contiguous slice of 8-bit RGBA pixel data. Must contain at least `width * height * 4` bytes.
/// * `config` - Compression parameters for quality and performance.
///
/// # Returns
///
/// Returns an immutable byte buffer containing the encoded WebP image on success.
///
/// # Errors
///
/// This function returns an error in the following situations:
///
/// * [`KomaError::InvalidDimensions`] - If `width` or `height` is 0, or exceeds 16,383 pixels.
/// * [`KomaError::BufferTooSmall`] - If the `rgba` buffer contains fewer than `width * height * 4` bytes.
/// * [`KomaError::EncodeFailure`] - If the underlying bitstream cannot be assembled.
///
/// # Examples
///
/// ```rust,no_run
/// use koma::{encode_rgba_webp, EncoderConfig, Result};
///
/// fn main() -> Result<()> {
///     let width = 100;
///     let height = 150;
///     let rgba = vec![255u8; (width * height * 4) as usize];
///     let config = EncoderConfig::new(85.0);
///
///     let webp_bytes = encode_rgba_webp(width, height, &rgba, &config)?;
///     assert!(!webp_bytes.is_empty());
///     Ok(())
/// }
/// ```
#[inline]
pub fn encode_rgba_webp(
    width: u32,
    height: u32,
    rgba: &[u8],
    config: &EncoderConfig,
) -> Result<bytes::Bytes> {
    let mut scratch = ENCODER_SCRATCH_POOL.get();
    let output = scratch.encode_rgba(width, height, rgba, config)?;
    Ok(bytes::Bytes::copy_from_slice(output))
}
