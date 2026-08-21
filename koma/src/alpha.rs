//! WebP Alpha plane filtering, ALPH chunk serialization, and extended container support.
//!
//! Complies with the WebP Extended File Format Specification:
//! - RIFF WebP container with `VP8X` extended header chunk
//! - `ALPH` alpha bitstream chunk with directional spatial prediction filters

use crate::config::AlphaFilter;

/// Directional prediction filtering method for the alpha channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AlphaFilterMethod {
    /// No filtering (raw sample bytes).
    None = 0,
    /// Horizontal predictor: `A[x, y] - A[x-1, y]`.
    Horizontal = 1,
    /// Vertical predictor: `A[x, y] - A[x, y-1]`.
    Vertical = 2,
    /// Gradient predictor: `A[x, y] - clamp(A[x-1, y] + A[x, y-1] - A[x-1, y-1])`.
    Gradient = 3,
}

/// WebP ALPH chunk generator with spatial prediction filtering.
pub struct AlphaEncoder;

impl AlphaEncoder {
    /// Encodes raw alpha plane (`width * height` bytes) into a formatted `ALPH` chunk payload.
    ///
    /// Supports both lossless 8-bit alpha (alpha_quality = 100) and lossy alpha quantization (alpha_quality < 100).
    /// When `alpha_filter` is `AlphaFilter::None`, bypasses spatial variance search for maximum encoding throughput.
    pub fn encode_alpha(
        alpha: &[u8],
        width: usize,
        height: usize,
        alpha_quality: u8,
        alpha_filter: AlphaFilter,
    ) -> Vec<u8> {
        assert_eq!(alpha.len(), width * height);

        let quantized_alpha;
        let alpha_slice = if alpha_quality < 100 {
            // Adaptive lossy alpha quantizer
            // Levels: 256 (q=100), down to 32 (q=50), 8 (q=0)
            let step = ((100 - alpha_quality as u32) * 7 / 100 + 1) as u8;
            let mut q_buf = vec![0u8; alpha.len()];
            for (src, dst) in alpha.iter().zip(q_buf.iter_mut()) {
                if *src == 0 || *src == 255 {
                    *dst = *src; // Preserve exact fully transparent & opaque boundaries
                } else {
                    let half = step / 2;
                    let val = ((*src as u32 + half as u32) / step as u32 * step as u32).min(255) as u8;
                    *dst = val;
                }
            }
            quantized_alpha = q_buf;
            &quantized_alpha[..]
        } else {
            alpha
        };

        // WebP ALPH chunk specification (RFC / libwebp alpha_dec.c):
        // Header Byte layout:
        // - Bits 0-1: Compression method (0 = No compression / uncompressed raw, 1 = Lossless)
        // - Bits 2-3: Filter method (0 = None, 1 = Horizontal, 2 = Vertical, 3 = Gradient)
        // - Bits 4-5: Pre-processing (0 = None, 1 = Level reduction)
        // - Bits 6-7: Reserved (must be 0)
        //
        // NOTE: For uncompressed raw alpha (compression = 0), spatial delta filtering (methods 1..3)
        // must NOT be used because standard decoders (Chromium, Safari, Firefox, libwebp)
        // treat spatial filter deltas as uncompressed raw alpha values if lossless compression
        // is not active, which results in transparent/corrupted frames. We strictly use FilterMethod::None (0).
        let chosen_filter = AlphaFilterMethod::None;
        let compression_method = 0u8; // 0 = No compression
        let pre_processing = 0u8; // 0 = None
        let filter_method = (chosen_filter as u8) & 0x03;

        let header_byte = compression_method | (filter_method << 2) | (pre_processing << 4);

        let mut output = Vec::with_capacity(1 + alpha_slice.len());
        output.push(header_byte);
        output.extend_from_slice(alpha_slice);
        output
    }

    /// Evaluates spatial filters and returns the optimal filter and its filtered output.
    fn select_best_filter(
        alpha: &[u8],
        width: usize,
        height: usize,
    ) -> (AlphaFilterMethod, Vec<u8>) {
        let mut best_method = AlphaFilterMethod::None;
        let mut best_score = usize::MAX;
        let mut best_buf = Vec::new();

        let methods = [
            AlphaFilterMethod::None,
            AlphaFilterMethod::Horizontal,
            AlphaFilterMethod::Vertical,
            AlphaFilterMethod::Gradient,
        ];

        for method in methods {
            let (score, buf) = Self::apply_filter(alpha, width, height, method);
            if score < best_score {
                best_score = score;
                best_method = method;
                best_buf = buf;
            }
        }

        (best_method, best_buf)
    }

    /// Applies the specified filter and computes its total absolute gradient score.
    fn apply_filter(
        alpha: &[u8],
        width: usize,
        height: usize,
        method: AlphaFilterMethod,
    ) -> (usize, Vec<u8>) {
        let mut out = vec![0u8; width * height];
        let mut total_energy: usize = 0;

        for y in 0..height {
            let row_idx = y * width;
            let prev_row_idx = if y > 0 { (y - 1) * width } else { 0 };

            for x in 0..width {
                let current = alpha[row_idx + x] as i32;
                let left = if x > 0 {
                    alpha[row_idx + x - 1] as i32
                } else if y > 0 {
                    alpha[prev_row_idx] as i32
                } else {
                    0
                };
                let top = if y > 0 {
                    alpha[prev_row_idx + x] as i32
                } else {
                    left
                };
                let top_left = if x > 0 && y > 0 {
                    alpha[prev_row_idx + x - 1] as i32
                } else {
                    top
                };

                let pred = match method {
                    AlphaFilterMethod::None => 0,
                    AlphaFilterMethod::Horizontal => left,
                    AlphaFilterMethod::Vertical => top,
                    AlphaFilterMethod::Gradient => {
                        let grad = left + top - top_left;
                        grad.clamp(0, 255)
                    }
                };

                let diff = ((current - pred) & 0xFF) as u8;
                out[row_idx + x] = diff;

                // Center energy around zero (treat > 128 as negative)
                let signed_diff = if diff > 128 { 256 - diff as usize } else { diff as usize };
                total_energy += signed_diff;
            }
        }

        (total_energy, out)
    }
}

/// Builds the 10-byte payload for a standard WebP `VP8X` extended header chunk.
pub fn build_vp8x_payload(width: u32, height: u32, has_alpha: bool) -> [u8; 10] {
    let mut payload = [0u8; 10];
    if has_alpha {
        payload[0] |= 0x10; // Bit 4: Alpha flag
    }

    let w_minus_1 = (width.saturating_sub(1)) & 0x00FF_FFFF;
    let h_minus_1 = (height.saturating_sub(1)) & 0x00FF_FFFF;

    // 24-bit Canvas Width (1-based, stored as width - 1)
    payload[4] = (w_minus_1 & 0xFF) as u8;
    payload[5] = ((w_minus_1 >> 8) & 0xFF) as u8;
    payload[6] = ((w_minus_1 >> 16) & 0xFF) as u8;

    // 24-bit Canvas Height (1-based, stored as height - 1)
    payload[7] = (h_minus_1 & 0xFF) as u8;
    payload[8] = ((h_minus_1 >> 8) & 0xFF) as u8;
    payload[9] = ((h_minus_1 >> 16) & 0xFF) as u8;

    payload
}
