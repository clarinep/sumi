//! VP8 Quantization, Reciprocal Division & Dequantization matrices.

use super::{
    tables::{AC_LOOKUP, DC_LOOKUP},
    trellis::trellis_quantize_4x4,
};

/// Quantizer setup and scale factors for a macroblock.
#[derive(Debug, Clone, Copy)]
pub struct Quantizer {
    pub y1_dc_quant: i16,
    pub y1_ac_quant: i16,
    pub y2_dc_quant: i16,
    pub y2_ac_quant: i16,
    pub uv_dc_quant: i16,
    pub uv_ac_quant: i16,

    pub qindex: usize,
    pub lambda: u32,
    pub use_trellis: bool,
}

impl Quantizer {
    /// Base quantizer from quality factor [0.0..100.0] and method effort [0..6].
    pub fn from_quality_and_method(quality: f32, method: u8) -> Self {
        // Map quality [0..100] to VP8 qindex [127..0]
        let qindex = if quality >= 100.0 {
            0
        } else if quality <= 0.0 {
            127
        } else {
            let q = (100.0 - quality) * 1.27;
            (q.round() as usize).clamp(0, 127)
        };

        let y1_dc = DC_LOOKUP[qindex];
        let y1_ac = AC_LOOKUP[qindex];
        let y2_dc = DC_LOOKUP[qindex] * 2;
        let y2_ac = (AC_LOOKUP[qindex] * 101 / 100).max(8);
        let uv_dc = DC_LOOKUP[qindex.saturating_sub(4)];
        let uv_ac = AC_LOOKUP[qindex.saturating_sub(4)];

        // Compute RD lambda Lagrange multiplier proportional to q^2
        let q_step = y1_ac as u32;
        let lambda = (q_step * q_step * 68) >> 7; // Tuned for optimal RD curve

        Self {
            y1_dc_quant: y1_dc,
            y1_ac_quant: y1_ac,
            y2_dc_quant: y2_dc,
            y2_ac_quant: y2_ac,
            uv_dc_quant: uv_dc,
            uv_ac_quant: uv_ac,
            qindex,
            lambda,
            use_trellis: method > 0, // method(0) uses ultra-fast deadzone quantization
        }
    }

    /// Creates a quantizer with a quality setting from 0 to 100 (defaults to method 4).
    pub fn from_quality(quality: f32) -> Self {
        Self::from_quality_and_method(quality, 4)
    }

    /// Quantizes a 4x4 block of DCT coefficients using adaptive RD-Trellis or deadzone quantization.
    #[inline(always)]
    pub fn quantize_block(
        &self,
        coeff: &[i16; 16],
        q_dc: i16,
        q_ac: i16,
        plane_type: usize,
        first_coeff: usize,
        quantized: &mut [i16; 16],
        dequant: &mut [i16; 16],
    ) -> bool {
        if self.use_trellis {
            trellis_quantize_4x4(
                coeff,
                q_dc,
                q_ac,
                self.lambda,
                plane_type,
                first_coeff,
                quantized,
                dequant,
            )
        } else {
            Self::quantize_and_dequant(coeff, q_dc, q_ac, quantized, dequant)
        }
    }

    /// Fast deadzone quantization and dequantization.
    #[inline(always)]
    pub fn quantize_and_dequant(
        coeff: &[i16; 16],
        q_dc: i16,
        q_ac: i16,
        quantized: &mut [i16; 16],
        dequant: &mut [i16; 16],
    ) -> bool {
        let mut has_nonzero = false;

        for i in 0..16 {
            let q = if i == 0 { q_dc as i32 } else { q_ac as i32 };
            let c = coeff[i] as i32;

            let sign = c.signum();
            let abs_c = c.abs();
            let q_val = (abs_c + (q >> 1)) / q;
            let q_signed = (sign * q_val) as i16;

            quantized[i] = q_signed;
            dequant[i] = (q_signed as i32 * q) as i16;

            if q_signed != 0 {
                has_nonzero = true;
            }
        }

        has_nonzero
    }
}
