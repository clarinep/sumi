//! Macroblock prediction, quantization, and lossy VP8 keyframe encoding.
//!
//! Orchestrates the intra-prediction modes, 4x4 subblock transforms, coefficient
//! quantization, and entropy encoding according to RFC 6386 keyframe specifications.

use crate::vp8::bool_coder::BoolEncoder;
use crate::vp8::config::EncoderConfig;
use crate::vp8::tables::{
    AC_QLOOKUP, DC_QLOOKUP, PCAT1, PCAT2, PCAT3, PROB_EOB, PROB_ONE, PROB_TWO, PROB_ZERO,
    VP8_START_CODE, ZIGZAG,
};
use crate::vp8::transform::{fdct_4x4, forward_wht_4x4, idct_4x4, inverse_wht_4x4};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// High-performance reciprocal quantizer eliminating CPU division pipeline stalls (`IDIV`).
///
/// Converts 15–74 cycle hardware integer divisions into 1-cycle integer multiplications
/// and bit shifts matching Agner Fog's microarchitecture guidelines.
#[derive(Debug, Clone, Copy)]
pub struct FastQuantizer {
    q: i16,
    inv_q: u64,
}

impl FastQuantizer {
    #[inline(always)]
    pub const fn new(q: i16) -> Self {
        let q = if q < 1 { 1 } else { q };
        // Fixed-point reciprocal: (1 << 32) / q
        let inv_q = ((1u64 << 32) + (q as u64 / 2)) / (q as u64);
        Self { q, inv_q }
    }

    /// Quantizes a signed 16-bit coefficient without `IDIV`.
    #[inline(always)]
    pub fn quantize(&self, coeff: i16) -> i16 {
        let abs_c = coeff.unsigned_abs() as u64;
        let q_val = ((abs_c * self.inv_q) >> 32) as i16;
        if coeff < 0 {
            -q_val
        } else {
            q_val
        }
    }

    /// Dequantizes a quantized coefficient.
    #[inline(always)]
    pub fn dequantize(&self, q_coeff: i16) -> i16 {
        q_coeff * self.q
    }
}

/// Maps quality scale `0.0..=100.0` to the VP8 quantizer index `0..=127`.
#[inline(always)]
pub fn quality_to_q_index(quality: f32) -> usize {
    let q = quality.clamp(0.0, 100.0);
    if q >= 100.0 {
        0
    } else if q <= 0.0 {
        127
    } else {
        let factor = (100.0 - q) / 100.0;
        ((factor * 127.0) as usize).clamp(0, 127)
    }
}

/// Checks if 16 contiguous bytes in memory match a target constant byte using SIMD.
#[inline(always)]
fn is_flat_16(slice: &[u8], target: u8) -> bool {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let chunk = _mm_loadu_si128(slice.as_ptr() as *const __m128i);
        let target_vec = _mm_set1_epi8(target as i8);
        let eq = _mm_cmpeq_epi8(chunk, target_vec);
        _mm_movemask_epi8(eq) == 0xFFFF
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        slice[0] == target
            && slice[1] == target
            && slice[2] == target
            && slice[3] == target
            && slice[4] == target
            && slice[5] == target
            && slice[6] == target
            && slice[7] == target
            && slice[8] == target
            && slice[9] == target
            && slice[10] == target
            && slice[11] == target
            && slice[12] == target
            && slice[13] == target
            && slice[14] == target
            && slice[15] == target
    }
}

/// Checks if 8 contiguous bytes in memory match a target constant byte.
#[inline(always)]
fn is_flat_8(slice: &[u8], target: u8) -> bool {
    let target_u64 = u64::from_ne_bytes([target; 8]);
    let chunk_u64 = u64::from_ne_bytes(slice[..8].try_into().unwrap());
    chunk_u64 == target_u64
}

/// Fast sum of 8 unsigned bytes using SIMD SAD or 64-bit word operations.
#[inline(always)]
fn sum_bytes_8(slice: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let chunk_u64 = *(slice.as_ptr() as *const u64);
        let vec = _mm_cvtsi64_si128(chunk_u64 as i64);
        let zero = _mm_setzero_si128();
        let sad = _mm_sad_epu8(vec, zero);
        _mm_cvtsi128_si32(sad) as u32
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        slice[..8].iter().map(|&x| x as u32).sum()
    }
}

/// Fast sum of 16 unsigned bytes using SIMD SAD (Sum of Absolute Differences against zero).
#[inline(always)]
fn sum_bytes_16(slice: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let chunk = _mm_loadu_si128(slice.as_ptr() as *const __m128i);
        let zero = _mm_setzero_si128();
        let sad = _mm_sad_epu8(chunk, zero);
        let lo = _mm_cvtsi128_si32(sad) as u32;
        let hi = _mm_extract_epi16(sad, 4) as u32;
        lo + hi
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        slice.iter().map(|&x| x as u32).sum()
    }
}

/// Encodes quantized AC/DC transform coefficients into the arithmetic boolean bitstream.
#[inline(always)]
pub fn encode_coeffs_block(coeffs: &[i16; 16], start_idx: usize, bool_coder: &mut BoolEncoder) {
    let mut last_nz = -1i32;
    for i in (start_idx..16).rev() {
        if coeffs[ZIGZAG[i]] != 0 {
            last_nz = i as i32;
            break;
        }
    }

    if last_nz < 0 {
        bool_coder.put_bit(false, PROB_EOB);
        return;
    }

    let end_idx = last_nz as usize;
    for i in start_idx..=end_idx {
        let val = coeffs[ZIGZAG[i]];
        if val == 0 {
            bool_coder.put_bit(true, PROB_EOB);
            bool_coder.put_bit(false, PROB_ZERO);
        } else {
            bool_coder.put_bit(true, PROB_EOB);
            bool_coder.put_bit(true, PROB_ZERO);
            let abs_val = val.unsigned_abs() as u32;
            let sign = val < 0;

            match abs_val {
                1 => {
                    bool_coder.put_bit(false, PROB_ONE);
                }
                2 => {
                    bool_coder.put_bit(true, PROB_ONE);
                    bool_coder.put_bit(false, PROB_TWO);
                }
                3..=4 => {
                    bool_coder.put_bit(true, PROB_ONE);
                    bool_coder.put_bit(true, PROB_TWO);
                    bool_coder.put_bit(false, 159);
                    bool_coder.put_bit_equi(abs_val == 4);
                }
                5..=6 => {
                    // Category 1: 5-6 (base 5)
                    bool_coder.put_bit(true, PROB_ONE);
                    bool_coder.put_bit(true, PROB_TWO);
                    bool_coder.put_bit(true, 159);
                    bool_coder.put_bit(abs_val == 6, PCAT1[0]);
                }
                7..=10 => {
                    // Category 2: 7-10 (base 7)
                    bool_coder.put_bit(true, PROB_ONE);
                    bool_coder.put_bit(true, PROB_TWO);
                    bool_coder.put_bit(true, 159);
                    let diff = abs_val - 7;
                    bool_coder.put_bit((diff & 2) != 0, PCAT2[0]);
                    bool_coder.put_bit((diff & 1) != 0, PCAT2[1]);
                }
                11..=18 => {
                    // Category 3: 11-18 (base 11)
                    bool_coder.put_bit(true, PROB_ONE);
                    bool_coder.put_bit(true, PROB_TWO);
                    bool_coder.put_bit(true, 159);
                    let diff = abs_val - 11;
                    bool_coder.put_bit((diff & 4) != 0, PCAT3[0]);
                    bool_coder.put_bit((diff & 2) != 0, PCAT3[1]);
                    bool_coder.put_bit((diff & 1) != 0, PCAT3[2]);
                }
                _ => {
                    // Large AC/DC coefficients
                    bool_coder.put_bit(true, PROB_ONE);
                    bool_coder.put_bit(true, PROB_TWO);
                    bool_coder.put_bit(true, 159);
                    bool_coder.put_literal(abs_val, 11);
                }
            }

            bool_coder.put_bit_equi(sign);

            if i == end_idx {
                bool_coder.put_bit(false, PROB_EOB);
            }
        }
    }
}

/// Encodes a lossy VP8 keyframe bitstream using pre-allocated scratch memory.
#[inline]
pub fn encode_lossy_frame(
    width: u32,
    height: u32,
    pad_width: usize,
    pad_height: usize,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    recon_y: &mut [u8],
    recon_u: &mut [u8],
    recon_v: &mut [u8],
    vp8_entropy_buf: &mut Vec<u8>,
    vp8_output_buf: &mut Vec<u8>,
    config: &EncoderConfig,
) {
    let mb_cols = pad_width / 16;
    let mb_rows = pad_height / 16;

    let q_idx = quality_to_q_index(config.quality);
    let q_y1_dc = FastQuantizer::new(DC_QLOOKUP[q_idx]);
    let q_y1_ac = FastQuantizer::new(AC_QLOOKUP[q_idx]);
    let q_y2_dc = FastQuantizer::new(DC_QLOOKUP[q_idx] * 2);
    let q_y2_ac = FastQuantizer::new(AC_QLOOKUP[q_idx] * 155 / 100);
    let q_uv_dc = FastQuantizer::new(DC_QLOOKUP[q_idx]);
    let q_uv_ac = FastQuantizer::new(AC_QLOOKUP[q_idx]);

    let mut bool_coder = BoolEncoder::new(vp8_entropy_buf);

    // Keyframe RFC 6386 header
    bool_coder.put_bit_equi(false); // YUV
    bool_coder.put_bit_equi(false); // Normal clamping
    bool_coder.put_bit_equi(false); // No segmentation
    bool_coder.put_bit_equi(false); // Filter type: 0
    bool_coder.put_literal(0, 6);   // Filter level
    bool_coder.put_literal(0, 3);   // Sharpness
    bool_coder.put_literal(q_idx as u32, 7); // Base q_index
    bool_coder.put_bit_equi(false); // No delta_q
    bool_coder.put_bit_equi(false);
    bool_coder.put_bit_equi(false);
    bool_coder.put_bit_equi(false);
    bool_coder.put_bit_equi(false);
    bool_coder.put_bit_equi(false); // No refresh probs

    let mut y_dc_coeffs = [0i16; 16];
    let mut y_wht_coeffs = [0i16; 16];
    let mut y_q_wht = [0i16; 16];
    let mut y_deq_wht = [0i16; 16];
    let mut y_rec_dc = [0i16; 16];

    let mut sub_coeffs = [[0i16; 16]; 16];
    let mut sub_q = [[0i16; 16]; 16];

    let y_stride = pad_width;
    let uv_stride = pad_width / 2;

    for mb_y in 0..mb_rows {
        for mb_x in 0..mb_cols {
            let mb_x_px = mb_x * 16;
            let mb_y_px = mb_y * 16;
            let uv_x_px = mb_x * 8;
            let uv_y_px = mb_y * 8;

            // Fast DC predictor calculation for Luma using SIMD SAD
            let mut sum_luma = 0u32;
            let mut count_luma = 0u32;
            if mb_y > 0 {
                let top_off = (mb_y_px - 1) * y_stride + mb_x_px;
                sum_luma += sum_bytes_16(&recon_y[top_off..top_off + 16]);
                count_luma += 16;
            }
            if mb_x > 0 {
                for y in 0..16 {
                    let left_val = recon_y[(mb_y_px + y) * y_stride + (mb_x_px - 1)];
                    sum_luma += left_val as u32;
                }
                count_luma += 16;
            }
            let dc_y = match count_luma {
                32 => ((sum_luma + 16) >> 5) as u8,
                16 => ((sum_luma + 8) >> 4) as u8,
                _ => 128,
            };

            // Intra-prediction mode = DC (0)
            bool_coder.put_bit_equi(false);

            // DC predictor for Chroma
            let mut sum_u = 0u32;
            let mut sum_v = 0u32;
            let mut count_uv = 0u32;
            if mb_y > 0 {
                let top_u_off = (uv_y_px - 1) * uv_stride + uv_x_px;
                sum_u += sum_bytes_8(&recon_u[top_u_off..top_u_off + 8]);
                sum_v += sum_bytes_8(&recon_v[top_u_off..top_u_off + 8]);
                count_uv += 8;
            }
            if mb_x > 0 {
                for y in 0..8 {
                    sum_u += recon_u[(uv_y_px + y) * uv_stride + (uv_x_px - 1)] as u32;
                    sum_v += recon_v[(uv_y_px + y) * uv_stride + (uv_x_px - 1)] as u32;
                }
                count_uv += 8;
            }
            let dc_u = match count_uv {
                16 => ((sum_u + 8) >> 4) as u8,
                8 => ((sum_u + 4) >> 3) as u8,
                _ => 128,
            };
            let dc_v = match count_uv {
                16 => ((sum_v + 8) >> 4) as u8,
                8 => ((sum_v + 4) >> 3) as u8,
                _ => 128,
            };

            // Chroma intra-prediction mode = DC (0)
            bool_coder.put_bit_equi(false);

            // Fast Macroblock Flat-Field Check for Luma (SIMD 16-byte compare shortcut)
            let src_y_off = mb_y_px * y_stride + mb_x_px;
            let mut is_flat_luma = config.fast_anime_shortcuts;
            if is_flat_luma {
                for y in 0..16 {
                    let s_row = &y_plane[src_y_off + y * y_stride..src_y_off + y * y_stride + 16];
                    if !is_flat_16(s_row, dc_y) {
                        is_flat_luma = false;
                        break;
                    }
                }
            }

            if is_flat_luma {
                // Completely skip 16 FDCTs + WHT + 16 IDCTs
                bool_coder.put_bit(false, PROB_EOB); // Y2 EOB
                for _ in 0..16 {
                    bool_coder.put_bit(false, PROB_EOB); // Subblock EOB
                }
                for y in 0..16 {
                    let r_off = (mb_y_px + y) * y_stride + mb_x_px;
                    recon_y[r_off..r_off + 16].fill(dc_y);
                }
            } else {
                // 16 Luma 4x4 subblocks
                let dc_val_i16 = dc_y as i16;
                for blk in 0..16 {
                    let blk_x = (blk & 3) * 4;
                    let blk_y = (blk >> 2) * 4;
                    let mut res = [0i16; 16];
                    for y in 0..4 {
                        let s_idx = src_y_off + (blk_y + y) * y_stride + blk_x;
                        let r_row = y * 4;
                        res[r_row] = (y_plane[s_idx] as i16) - dc_val_i16;
                        res[r_row + 1] = (y_plane[s_idx + 1] as i16) - dc_val_i16;
                        res[r_row + 2] = (y_plane[s_idx + 2] as i16) - dc_val_i16;
                        res[r_row + 3] = (y_plane[s_idx + 3] as i16) - dc_val_i16;
                    }
                    let has_diff = fdct_4x4(&res, &mut sub_coeffs[blk]);
                    y_dc_coeffs[blk] = if has_diff { sub_coeffs[blk][0] } else { 0 };
                }

                forward_wht_4x4(&y_dc_coeffs, &mut y_wht_coeffs);
                for i in 0..16 {
                    let quantizer = if i == 0 { &q_y2_dc } else { &q_y2_ac };
                    y_q_wht[i] = quantizer.quantize(y_wht_coeffs[i]);
                    y_deq_wht[i] = quantizer.dequantize(y_q_wht[i]);
                }
                inverse_wht_4x4(&y_deq_wht, &mut y_rec_dc);

                encode_coeffs_block(&y_q_wht, 0, &mut bool_coder);

                for blk in 0..16 {
                    let blk_x = (blk & 3) * 4;
                    let blk_y = (blk >> 2) * 4;

                    sub_coeffs[blk][0] = 0;
                    for i in 1..16 {
                        sub_q[blk][i] = q_y1_ac.quantize(sub_coeffs[blk][i]);
                    }
                    encode_coeffs_block(&sub_q[blk], 1, &mut bool_coder);

                    let mut dequant = [0i16; 16];
                    for i in 1..16 {
                        dequant[i] = q_y1_ac.dequantize(sub_q[blk][i]);
                    }
                    dequant[0] = y_rec_dc[blk];

                    let mut rec_res = [0i16; 16];
                    idct_4x4(&dequant, &mut rec_res);

                    for y in 0..4 {
                        let r_idx = (mb_y_px + blk_y + y) * y_stride + (mb_x_px + blk_x);
                        let r_row = y * 4;
                        recon_y[r_idx] = (dc_val_i16 + rec_res[r_row]).clamp(0, 255) as u8;
                        recon_y[r_idx + 1] = (dc_val_i16 + rec_res[r_row + 1]).clamp(0, 255) as u8;
                        recon_y[r_idx + 2] = (dc_val_i16 + rec_res[r_row + 2]).clamp(0, 255) as u8;
                        recon_y[r_idx + 3] = (dc_val_i16 + rec_res[r_row + 3]).clamp(0, 255) as u8;
                    }
                }
            }

            // Chroma U & V (4 blocks each)
            let src_u_off = uv_y_px * uv_stride + uv_x_px;
            let src_v_off = uv_y_px * uv_stride + uv_x_px;

            for uv_plane in 0..2 {
                let (src_p, recon_p, s_off, dc_val) = if uv_plane == 0 {
                    (u_plane, &mut *recon_u, src_u_off, dc_u)
                } else {
                    (v_plane, &mut *recon_v, src_v_off, dc_v)
                };

                let mut is_flat_uv = config.fast_anime_shortcuts;
                if is_flat_uv {
                    for y in 0..8 {
                        let s_row = &src_p[s_off + y * uv_stride..s_off + y * uv_stride + 8];
                        if !is_flat_8(s_row, dc_val) {
                            is_flat_uv = false;
                            break;
                        }
                    }
                }

                if is_flat_uv {
                    for _ in 0..4 {
                        bool_coder.put_bit(false, PROB_EOB);
                    }
                    for y in 0..8 {
                        let r_off = (uv_y_px + y) * uv_stride + uv_x_px;
                        recon_p[r_off..r_off + 8].fill(dc_val);
                    }
                } else {
                    let dc_val_i16 = dc_val as i16;
                    for blk in 0..4 {
                        let blk_x = (blk & 1) * 4;
                        let blk_y = (blk >> 1) * 4;

                        let mut res = [0i16; 16];
                        for y in 0..4 {
                            let s_idx = s_off + (blk_y + y) * uv_stride + blk_x;
                            let r_row = y * 4;
                            res[r_row] = (src_p[s_idx] as i16) - dc_val_i16;
                            res[r_row + 1] = (src_p[s_idx + 1] as i16) - dc_val_i16;
                            res[r_row + 2] = (src_p[s_idx + 2] as i16) - dc_val_i16;
                            res[r_row + 3] = (src_p[s_idx + 3] as i16) - dc_val_i16;
                        }

                        let mut coeffs = [0i16; 16];
                        let mut q_coeffs = [0i16; 16];
                        let mut dequant = [0i16; 16];
                        let mut rec_res = [0i16; 16];

                        fdct_4x4(&res, &mut coeffs);
                        for i in 0..16 {
                            let quantizer = if i == 0 { &q_uv_dc } else { &q_uv_ac };
                            q_coeffs[i] = quantizer.quantize(coeffs[i]);
                            dequant[i] = quantizer.dequantize(q_coeffs[i]);
                        }
                        encode_coeffs_block(&q_coeffs, 0, &mut bool_coder);

                        idct_4x4(&dequant, &mut rec_res);

                        for y in 0..4 {
                            let r_idx = (uv_y_px + blk_y + y) * uv_stride + (uv_x_px + blk_x);
                            let r_row = y * 4;
                            recon_p[r_idx] = (dc_val_i16 + rec_res[r_row]).clamp(0, 255) as u8;
                            recon_p[r_idx + 1] = (dc_val_i16 + rec_res[r_row + 1]).clamp(0, 255) as u8;
                            recon_p[r_idx + 2] = (dc_val_i16 + rec_res[r_row + 2]).clamp(0, 255) as u8;
                            recon_p[r_idx + 3] = (dc_val_i16 + rec_res[r_row + 3]).clamp(0, 255) as u8;
                        }
                    }
                }
            }
        }
    }

    bool_coder.finish();

    let entropy_len = vp8_entropy_buf.len();
    vp8_output_buf.clear();
    vp8_output_buf.reserve(10 + entropy_len);

    // Frame tag (3 bytes LE)
    let tag = ((entropy_len as u32) << 5) | 0x10;
    vp8_output_buf.push((tag & 0xFF) as u8);
    vp8_output_buf.push(((tag >> 8) & 0xFF) as u8);
    vp8_output_buf.push(((tag >> 16) & 0xFF) as u8);

    // RFC 6386 magic start code
    vp8_output_buf.extend_from_slice(&VP8_START_CODE);

    // Dimensions
    vp8_output_buf.push((width & 0xFF) as u8);
    vp8_output_buf.push(((width >> 8) & 0x3F) as u8);
    vp8_output_buf.push((height & 0xFF) as u8);
    vp8_output_buf.push(((height >> 8) & 0x3F) as u8);

    vp8_output_buf.extend_from_slice(vp8_entropy_buf);
}

