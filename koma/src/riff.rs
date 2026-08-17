//! RIFF and WebP container chunk assembly.
//!
//! Handles packaging of VP8 lossy bitstreams, alpha payloads (ALPH), and extended
//! container headers (VP8X) into standard RFC WebP containers without dynamic memory allocations.

use crate::error::{KomaError, Result};

pub const RIFF_MAGIC: &[u8; 4] = b"RIFF";
pub const WEBP_MAGIC: &[u8; 4] = b"WEBP";
pub const VP8X_MAGIC: &[u8; 4] = b"VP8X";
pub const ALPH_MAGIC: &[u8; 4] = b"ALPH";
pub const VP8_MAGIC: &[u8; 4] = b"VP8 ";

/// Assembles the outer WebP / RIFF container into the destination buffer.
///
/// Encapsulates raw VP8 frames and optional ALPH chunks into a compliant WebP container
/// formatted according to Google WebP container specifications.
///
/// # Errors
///
/// Returns [`KomaError::EncodeFailure`] if the total payload size exceeds 4GB WebP limits.
#[inline]
pub fn assemble_webp(
    width: u32,
    height: u32,
    vp8_payload: &[u8],
    alpha_payload: Option<&[u8]>,
    target: &mut Vec<u8>,
) -> Result<()> {
    target.clear();

    let vp8_padding = vp8_payload.len() & 1;
    let mut total_body_size = 4 + 8 + vp8_payload.len() + vp8_padding; // "WEBP" + "VP8 " chunk

    if let Some(alpha) = alpha_payload {
        let alpha_padding = alpha.len() & 1;
        total_body_size += 18 + 8 + alpha.len() + alpha_padding; // VP8X chunk (18 bytes) + ALPH chunk
    }

    if total_body_size > u32::MAX as usize - 16 {
        tracing::error!("encode_rgba failed: container size overflow");
        return Err(KomaError::EncodeFailure("container size overflow"));
    }

    target.reserve(total_body_size + 8);

    // RIFF chunk header
    target.extend_from_slice(RIFF_MAGIC);
    target.extend_from_slice(&(total_body_size as u32).to_le_bytes());
    target.extend_from_slice(WEBP_MAGIC);

    // Extended VP8X header if alpha channel present
    if let Some(alpha) = alpha_payload {
        target.extend_from_slice(VP8X_MAGIC);
        target.extend_from_slice(&10u32.to_le_bytes()); // VP8X payload is always 10 bytes

        let flags: u8 = 1 << 4; // Alpha flag (bit 4)
        target.push(flags);
        target.push(0);
        target.push(0);
        target.push(0);

        let canvas_w = width.saturating_sub(1);
        let canvas_h = height.saturating_sub(1);

        target.push((canvas_w & 0xFF) as u8);
        target.push(((canvas_w >> 8) & 0xFF) as u8);
        target.push(((canvas_w >> 16) & 0xFF) as u8);

        target.push((canvas_h & 0xFF) as u8);
        target.push(((canvas_h >> 8) & 0xFF) as u8);
        target.push(((canvas_h >> 16) & 0xFF) as u8);

        // ALPH chunk
        target.extend_from_slice(ALPH_MAGIC);
        target.extend_from_slice(&(alpha.len() as u32).to_le_bytes());
        target.extend_from_slice(alpha);
        if (alpha.len() & 1) != 0 {
            target.push(0);
        }
    }

    // VP8 lossy frame chunk
    target.extend_from_slice(VP8_MAGIC);
    target.extend_from_slice(&(vp8_payload.len() as u32).to_le_bytes());
    target.extend_from_slice(vp8_payload);
    if vp8_padding != 0 {
        target.push(0);
    }

    Ok(())
}
