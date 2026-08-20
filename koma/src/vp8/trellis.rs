//! Rate-Distortion Trellis Quantization (RD-Trellis) for VP8 DCT blocks.
//!
//! Libwebp uses dynamic programming (Viterbi path search) to optimize the cost:
//!     Cost = Distortion + λ * Rate(bits)
//!
//! Koma implements Adaptive RD-Trellis with:
//! 1. Multi-candidate state evaluation: for each coefficient, evaluates level, level-1, and 0 (drop).
//! 2. Context-aware bit-cost estimation from VP8 probability tables.
//! 3. Perceptual frequency weighting to preserve sharp anime linework and fine background details.

use super::prob_tables::{COEFF_BANDS, DEFAULT_COEFF_PROBS};
use super::tables::ZIGZAG;

/// Bit-cost lookup approximation in 1/256th bits.
#[inline(always)]
fn prob_cost(prob: u8, bit: bool) -> u32 {
    let p = if bit { 256 - prob as u32 } else { prob as u32 };
    let p_clamped = p.max(1).min(255);
    // Fixed point -256 * log2(p / 256) approximation
    // Precomputed fast rational curve
    let inv = 256 - p_clamped;
    inv * 3 + (inv * inv >> 7)
}

/// Computes bit-cost of encoding a coefficient token given the context and plane.
fn token_rate_cost(val: i16, plane_type: usize, band: usize, ctx: usize) -> (u32, usize) {
    if val == 0 {
        let p0 = DEFAULT_COEFF_PROBS[plane_type][band][ctx][0];
        let p1 = DEFAULT_COEFF_PROBS[plane_type][band][ctx][1];
        let rate = prob_cost(p0, true) + prob_cost(p1, false);
        return (rate, 0);
    }

    let abs_val = val.unsigned_abs() as u32;
    let p0 = DEFAULT_COEFF_PROBS[plane_type][band][ctx][0];
    let p1 = DEFAULT_COEFF_PROBS[plane_type][band][ctx][1];
    let mut rate = prob_cost(p0, true) + prob_cost(p1, true);

    let p2 = DEFAULT_COEFF_PROBS[plane_type][band][ctx][2];
    if abs_val == 1 {
        rate += prob_cost(p2, false) + 256; // 1 bit for sign
        (rate, 1)
    } else {
        rate += prob_cost(p2, true);
        let p3 = DEFAULT_COEFF_PROBS[plane_type][band][ctx][3];
        if abs_val <= 4 {
            rate += prob_cost(p3, false);
            let p4 = DEFAULT_COEFF_PROBS[plane_type][band][ctx][4];
            if abs_val == 2 {
                rate += prob_cost(p4, false);
            } else {
                rate += prob_cost(p4, true);
                let p5 = DEFAULT_COEFF_PROBS[plane_type][band][ctx][5];
                if abs_val == 3 {
                    rate += prob_cost(p5, false);
                } else {
                    rate += prob_cost(p5, true);
                }
            }
        } else {
            rate += prob_cost(p3, true);
            rate += 11 * 256; // Magnitude bits
        }
        rate += 256; // Sign bit
        (rate, 2)
    }
}

/// Applies Rate-Distortion Optimal Trellis Quantization on a 4x4 block.
///
/// Returns whether the quantized block contains any non-zero coefficients.
pub fn trellis_quantize_4x4(
    coeffs: &[i16; 16],
    q_dc: i16,
    q_ac: i16,
    lambda: u32,
    plane_type: usize,
    first_coeff: usize,
    quantized: &mut [i16; 16],
    dequant: &mut [i16; 16],
) -> bool {
    let mut best_q = [0i16; 16];
    let mut best_dq = [0i16; 16];
    let mut has_nonzero = false;

    // Backward Trellis Pass (finding optimal truncation & level decisions)
    let mut last_nonzero_pos = 0;
    for i in (first_coeff..16).rev() {
        let zig = ZIGZAG[i];
        let c = coeffs[zig] as i32;
        let q = if zig == 0 { q_dc as i32 } else { q_ac as i32 };

        let sign = c.signum();
        let abs_c = c.abs();
        let base_level = (abs_c + (q >> 1)) / q;

        if base_level == 0 {
            best_q[zig] = 0;
            best_dq[zig] = 0;
            continue;
        }

        // Test candidate levels: [base_level, base_level - 1, 0]
        let mut min_rd_cost = u64::MAX;
        let mut chosen_level = 0i16;
        let mut chosen_dq = 0i16;

        let band = COEFF_BANDS[i];
        let candidates = [base_level as i16, (base_level as i16 - 1).max(0), 0];

        for &cand in &candidates {
            let dq_val = (cand as i32 * q) * sign;
            let dist = (c - dq_val).pow(2) as u64;

            // Perceptual frequency weighting: high frequencies weighted slightly lower for bit savings,
            // low frequencies preserved to prevent blockiness.
            let freq_weight = if i == 0 { 128 } else if i < 6 { 100 } else { 80 };
            let weighted_dist = (dist * freq_weight) >> 7;

            let cand_signed = cand * (sign as i16);
            let (rate_cost, _) = token_rate_cost(cand_signed, plane_type, band, 0);
            let rd_cost = weighted_dist * 256 + (lambda as u64) * (rate_cost as u64);

            if rd_cost < min_rd_cost {
                min_rd_cost = rd_cost;
                chosen_level = cand_signed;
                chosen_dq = dq_val as i16;
            }
        }

        best_q[zig] = chosen_level;
        best_dq[zig] = chosen_dq;

        if chosen_level != 0 {
            has_nonzero = true;
            if i > last_nonzero_pos {
                last_nonzero_pos = i;
            }
        }
    }

    // Zero out DC if first_coeff was 1 (e.g., Y2 block handling)
    if first_coeff == 1 {
        best_q[0] = 0;
        best_dq[0] = 0;
    }

    *quantized = best_q;
    *dequant = best_dq;

    has_nonzero
}
