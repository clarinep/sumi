//! Macroblock prediction, quantization, and lossy VP8 keyframe encoding.
//!
//! Orchestrates the intra-prediction modes, 4x4 subblock transforms, coefficient
//! quantization, and entropy encoding according to RFC 6386 keyframe specifications.

use crate::vp8::bool_coder::BoolEncoder;
use crate::vp8::config::EncoderConfig;
use crate::vp8::header_tables::COEFFS_PROBA0;
use crate::vp8::tables::{
    AC_QLOOKUP, DC_QLOOKUP, KBANDS, PCAT3, PCAT4, PCAT5, PCAT6, VP8_START_CODE, ZIGZAG,
};
use crate::vp8::transform::{fdct_4x4, forward_wht_4x4, idct_4x4, inverse_wht_4x4};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// High-performance reciprocal quantizer eliminating CPU division pipeline stalls (`IDIV`).
///
/// Converts 15–74 cycle hardware integer divisions into 1-cycle integer multiplications
/// and bit shifts matching Agner Fog's microarchitecture guidelines.
#[derive(Debug, Clone, Copy)]
pub struct FastQuantizer {
    q: i16,
    inv_q: u64,
    round_offset: u64,
}

impl FastQuantizer {
    #[inline(always)]
    pub const fn new(q: i16) -> Self {
        let q = if q < 1 { 1 } else { q };
        // Fixed-point reciprocal: (1 << 32) / q
        let inv_q = ((1u64 << 32) + (q as u64 / 2)) / (q as u64);
        let round_offset = (q as u64) / 3;
        Self { q, inv_q, round_offset }
    }

    /// Quantizes a signed 16-bit coefficient without `IDIV`.
    #[inline(always)]
    pub fn quantize(&self, coeff: i16) -> i16 {
        let abs_c = coeff.unsigned_abs() as u64;
        let q_val = (((abs_c + self.round_offset) * self.inv_q) >> 32) as i16;
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
    if slice.len() < 16 { return false; }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let chunk = vld1q_u8(slice.as_ptr());
        let target_vec = vdupq_n_u8(target);
        let eq = vceqq_u8(chunk, target_vec);
        vminvq_u8(eq) == 0xFF
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let chunk = _mm_loadu_si128(slice.as_ptr() as *const __m128i);
        let target_vec = _mm_set1_epi8(target as i8);
        let eq = _mm_cmpeq_epi8(chunk, target_vec);
        _mm_movemask_epi8(eq) == 0xFFFF
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
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
    if slice.len() < 8 { return false; }
    let target_u64 = u64::from_ne_bytes([target; 8]);
    let chunk_u64 = u64::from_ne_bytes(slice[..8].try_into().unwrap_or([0; 8]));
    chunk_u64 == target_u64
}

/// Fast sum of 8 unsigned bytes using SIMD SAD or 64-bit word operations.
#[inline(always)]
fn sum_bytes_8(slice: &[u8]) -> u32 {
    if slice.len() < 8 { return slice.iter().copied().map(u32::from).sum(); }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let chunk = vld1_u8(slice.as_ptr());
        vaddlv_u8(chunk) as u32
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let chunk_u64 = std::ptr::read_unaligned(slice.as_ptr() as *const u64);
        let vec = _mm_cvtsi64_si128(chunk_u64 as i64);
        let zero = _mm_setzero_si128();
        let sad = _mm_sad_epu8(vec, zero);
        _mm_cvtsi128_si32(sad) as u32
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        slice[..8].iter().map(|&x| x as u32).sum()
    }
}

/// Fast sum of 16 unsigned bytes using SIMD SAD (Sum of Absolute Differences against zero).
#[inline(always)]
fn sum_bytes_16(slice: &[u8]) -> u32 {
    if slice.len() < 16 { return slice.iter().copied().map(u32::from).sum(); }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let chunk = vld1q_u8(slice.as_ptr());
        vaddlvq_u8(chunk) as u32
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let chunk = _mm_loadu_si128(slice.as_ptr() as *const __m128i);
        let zero = _mm_setzero_si128();
        let sad = _mm_sad_epu8(chunk, zero);
        let lo = _mm_cvtsi128_si32(sad) as u32;
        let hi = _mm_extract_epi16::<4>(sad) as u32;
        lo + hi
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        slice.iter().map(|&x| x as u32).sum()
    }
}

/// Subtracts DC predictor from 4x4 block pixels into 16 signed 16-bit residuals using SIMD.
#[inline(always)]
fn sub_dc_4x4(src: &[u8], offset: usize, stride: usize, dc: i16, out: &mut [i16; 16]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let dc_vec = vdupq_n_s16(dc);
        let ptr = src.as_ptr().add(offset);
        
        let r0 = (ptr as *const u32).read_unaligned();
        let r1 = (ptr.add(stride) as *const u32).read_unaligned();
        let r2 = (ptr.add(stride * 2) as *const u32).read_unaligned();
        let r3 = (ptr.add(stride * 3) as *const u32).read_unaligned();
        
        let u8_0 = vreinterpret_u8_u32(vdup_n_u32(r0));
        let u8_1 = vreinterpret_u8_u32(vdup_n_u32(r1));
        let u8_2 = vreinterpret_u8_u32(vdup_n_u32(r2));
        let u8_3 = vreinterpret_u8_u32(vdup_n_u32(r3));

        let u8_01 = vcombine_u8(u8_0, u8_1);
        let u8_23 = vcombine_u8(u8_2, u8_3);
        
        let u16_01 = vmovl_u8(vget_low_u8(u8_01));
        let u16_23 = vmovl_u8(vget_low_u8(u8_23));
        
        let s16_01 = vsubq_s16(vreinterpretq_s16_u16(u16_01), dc_vec);
        let s16_23 = vsubq_s16(vreinterpretq_s16_u16(u16_23), dc_vec);
        
        vst1q_s16(out.as_mut_ptr(), s16_01);
        vst1q_s16(out.as_mut_ptr().add(8), s16_23);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let dc_vec = _mm_set1_epi16(dc);
        let ptr = src.as_ptr().add(offset);
        
        let r0 = (ptr as *const u32).read_unaligned();
        let r1 = (ptr.add(stride) as *const u32).read_unaligned();
        let r2 = (ptr.add(stride * 2) as *const u32).read_unaligned();
        let r3 = (ptr.add(stride * 3) as *const u32).read_unaligned();
        
        let v01 = _mm_set_epi32(0, 0, r1 as i32, r0 as i32);
        let v23 = _mm_set_epi32(0, 0, r3 as i32, r2 as i32);
        let zero = _mm_setzero_si128();
        
        let s16_01 = _mm_sub_epi16(_mm_unpacklo_epi8(v01, zero), dc_vec);
        let s16_23 = _mm_sub_epi16(_mm_unpacklo_epi8(v23, zero), dc_vec);
        
        _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, s16_01);
        _mm_storeu_si128(out.as_mut_ptr().add(8) as *mut __m128i, s16_23);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        for y in 0..4 {
            let s_idx = offset + y * stride;
            let r_row = y * 4;
            out[r_row] = (src[s_idx] as i16) - dc;
            out[r_row + 1] = (src[s_idx + 1] as i16) - dc;
            out[r_row + 2] = (src[s_idx + 2] as i16) - dc;
            out[r_row + 3] = (src[s_idx + 3] as i16) - dc;
        }
    }
}

/// Adds DC predictor to 16 residual coefficients, clamps to [0, 255] and stores with SIMD.
#[inline(always)]
fn add_dc_and_clamp_4x4(res: &[i16; 16], dc: i16, dst: &mut [u8], dst_offset: usize, stride: usize) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let dc_vec = vdupq_n_s16(dc);
        let res0 = vld1q_s16(res.as_ptr());
        let res1 = vld1q_s16(res.as_ptr().add(8));
        let sum0 = vaddq_s16(res0, dc_vec);
        let sum1 = vaddq_s16(res1, dc_vec);
        let u8_0 = vqmovun_s16(sum0);
        let u8_1 = vqmovun_s16(sum1);
        
        let ptr = dst.as_mut_ptr().add(dst_offset);
        (ptr as *mut u32).write_unaligned(vget_lane_u32(vreinterpret_u32_u8(u8_0), 0));
        (ptr.add(stride) as *mut u32).write_unaligned(vget_lane_u32(vreinterpret_u32_u8(u8_0), 1));
        (ptr.add(stride * 2) as *mut u32).write_unaligned(vget_lane_u32(vreinterpret_u32_u8(u8_1), 0));
        (ptr.add(stride * 3) as *mut u32).write_unaligned(vget_lane_u32(vreinterpret_u32_u8(u8_1), 1));
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let dc_vec = _mm_set1_epi16(dc);
        let res0 = _mm_loadu_si128(res.as_ptr() as *const __m128i);
        let res1 = _mm_loadu_si128(res.as_ptr().add(8) as *const __m128i);
        let sum0 = _mm_add_epi16(res0, dc_vec);
        let sum1 = _mm_add_epi16(res1, dc_vec);
        let clamped = _mm_packus_epi16(sum0, sum1);
        
        let ptr = dst.as_mut_ptr().add(dst_offset);
        let r0 = _mm_cvtsi128_si32(clamped) as u32;
        let r1 = _mm_cvtsi128_si32(_mm_srli_si128::<4>(clamped)) as u32;
        let r2 = _mm_cvtsi128_si32(_mm_srli_si128::<8>(clamped)) as u32;
        let r3 = _mm_cvtsi128_si32(_mm_srli_si128::<12>(clamped)) as u32;
        
        (ptr as *mut u32).write_unaligned(r0);
        (ptr.add(stride) as *mut u32).write_unaligned(r1);
        (ptr.add(stride * 2) as *mut u32).write_unaligned(r2);
        (ptr.add(stride * 3) as *mut u32).write_unaligned(r3);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        for y in 0..4 {
            let r_idx = dst_offset + y * stride;
            let r_row = y * 4;
            dst[r_idx] = (dc + res[r_row]).clamp(0, 255) as u8;
            dst[r_idx + 1] = (dc + res[r_row + 1]).clamp(0, 255) as u8;
            dst[r_idx + 2] = (dc + res[r_row + 2]).clamp(0, 255) as u8;
            dst[r_idx + 3] = (dc + res[r_row + 3]).clamp(0, 255) as u8;
        }
    }
}

#[inline(always)]
fn sub_pred_4x4(
    src: &[u8],
    src_offset: usize,
    src_stride: usize,
    pred: &[u8],
    pred_offset: usize,
    pred_stride: usize,
    out: &mut [i16; 16],
) {
    for y in 0..4 {
        let s_idx = src_offset + y * src_stride;
        let p_idx = pred_offset + y * pred_stride;
        let r_row = y * 4;
        out[r_row] = (src[s_idx] as i16) - (pred[p_idx] as i16);
        out[r_row + 1] = (src[s_idx + 1] as i16) - (pred[p_idx + 1] as i16);
        out[r_row + 2] = (src[s_idx + 2] as i16) - (pred[p_idx + 2] as i16);
        out[r_row + 3] = (src[s_idx + 3] as i16) - (pred[p_idx + 3] as i16);
    }
}

#[inline(always)]
fn add_pred_and_clamp_4x4(
    res: &[i16; 16],
    pred: &[u8],
    pred_offset: usize,
    pred_stride: usize,
    dst: &mut [u8],
    dst_offset: usize,
    dst_stride: usize,
) {
    for y in 0..4 {
        let p_row = pred_offset + y * pred_stride;
        let r_idx = dst_offset + y * dst_stride;
        let r_row = y * 4;
        dst[r_idx] = ((pred[p_row] as i16) + res[r_row]).clamp(0, 255) as u8;
        dst[r_idx + 1] = ((pred[p_row + 1] as i16) + res[r_row + 1]).clamp(0, 255) as u8;
        dst[r_idx + 2] = ((pred[p_row + 2] as i16) + res[r_row + 2]).clamp(0, 255) as u8;
        dst[r_idx + 3] = ((pred[p_row + 3] as i16) + res[r_row + 3]).clamp(0, 255) as u8;
    }
}

#[inline(always)]
fn build_pred_16x16(
    mb_x: usize,
    mb_y: usize,
    mb_x_px: usize,
    mb_y_px: usize,
    y_stride: usize,
    recon_y: &[u8],
    pred: &mut [u8; 256],
) {
    if mb_y > 0 && mb_x > 0 {
        let top_left = recon_y[(mb_y_px - 1) * y_stride + (mb_x_px - 1)] as i32;
        let top_row = &recon_y[(mb_y_px - 1) * y_stride + mb_x_px..(mb_y_px - 1) * y_stride + mb_x_px + 16];
        for y in 0..16 {
            let left_val = recon_y[(mb_y_px + y) * y_stride + (mb_x_px - 1)] as i32;
            let row_off = y * 16;
            for x in 0..16 {
                let top_val = top_row[x] as i32;
                let val = (top_val + left_val - top_left).clamp(0, 255);
                pred[row_off + x] = val as u8;
            }
        }
    } else if mb_y > 0 {
        let top_row = &recon_y[(mb_y_px - 1) * y_stride + mb_x_px..(mb_y_px - 1) * y_stride + mb_x_px + 16];
        for y in 0..16 {
            pred[y * 16..(y + 1) * 16].copy_from_slice(top_row);
        }
    } else if mb_x > 0 {
        for y in 0..16 {
            let left_val = recon_y[(mb_y_px + y) * y_stride + (mb_x_px - 1)];
            pred[y * 16..(y + 1) * 16].fill(left_val);
        }
    } else {
        pred.fill(128);
    }
}

#[inline(always)]
fn build_pred_8x8(
    mb_x: usize,
    mb_y: usize,
    uv_x_px: usize,
    uv_y_px: usize,
    uv_stride: usize,
    recon: &[u8],
    pred: &mut [u8; 64],
) {
    if mb_y > 0 && mb_x > 0 {
        let top_left = recon[(uv_y_px - 1) * uv_stride + (uv_x_px - 1)] as i32;
        let top_row = &recon[(uv_y_px - 1) * uv_stride + uv_x_px..(uv_y_px - 1) * uv_stride + uv_x_px + 8];
        for y in 0..8 {
            let left_val = recon[(uv_y_px + y) * uv_stride + (uv_x_px - 1)] as i32;
            let row_off = y * 8;
            for x in 0..8 {
                let top_val = top_row[x] as i32;
                let val = (top_val + left_val - top_left).clamp(0, 255);
                pred[row_off + x] = val as u8;
            }
        }
    } else if mb_y > 0 {
        let top_row = &recon[(uv_y_px - 1) * uv_stride + uv_x_px..(uv_y_px - 1) * uv_stride + uv_x_px + 8];
        for y in 0..8 {
            pred[y * 8..(y + 1) * 8].copy_from_slice(top_row);
        }
    } else if mb_x > 0 {
        for y in 0..8 {
            let left_val = recon[(uv_y_px + y) * uv_stride + (uv_x_px - 1)];
            pred[y * 8..(y + 1) * 8].fill(left_val);
        }
    } else {
        pred.fill(128);
    }
}

/// Quantizes and dequantizes 16 coefficients with reciprocal arithmetic.
#[inline(always)]
fn quantize_dequantize_block(
    coeffs: &[i16; 16],
    q_dc: &FastQuantizer,
    q_ac: &FastQuantizer,
    out_q: &mut [i16; 16],
    out_deq: &mut [i16; 16],
) {
    for i in 0..16 {
        let quantizer = if i == 0 { q_dc } else { q_ac };
        out_q[i] = quantizer.quantize(coeffs[i]);
        out_deq[i] = quantizer.dequantize(out_q[i]);
    }
}

/// Quantizes and dequantizes 15 AC coefficients (indices 1..16) for Luma subblock.
#[inline(always)]
fn quantize_dequantize_y1_ac(
    coeffs: &[i16; 16],
    q_ac: &FastQuantizer,
    rec_dc: i16,
    out_q: &mut [i16; 16],
    out_deq: &mut [i16; 16],
) {
    out_q[0] = 0;
    out_deq[0] = rec_dc;
    for i in 1..16 {
        let q_val = q_ac.quantize(coeffs[i]);
        out_q[i] = q_val;
        out_deq[i] = q_ac.dequantize(q_val);
    }
}

/// Encodes a 4x4 block of DCT / WHT transform coefficients into the VP8 token bitstream.
/// - `coeff_type`: 0 for Y1 (16 AC 4x4 blocks), 1 for Y2 (WHT DC of 16x16 macroblock), 2 for UV (Chroma).
/// - `first`: 0 for Y2 and UV blocks, 1 for Y1 blocks (since DC is stored in Y2).
/// - `context`: 0, 1, or 2 based on above and left neighboring block non-zero status.
/// - `coeffs`: 16 quantized coefficients in 4x4 frequency matrix order (coeffs[0] = DC).
/// Returns true if the block contains at least one non-zero coefficient.
#[inline(always)]
pub fn encode_coeffs_block(
    coeffs: &[i16; 16],
    coeff_type: usize,
    first: usize,
    context: usize,
    bool_coder: &mut BoolEncoder,
) -> bool {
    let mut last = -1i32;
    for i in (first..16).rev() {
        if coeffs[ZIGZAG[i]] != 0 {
            last = i as i32;
            break;
        }
    }

    let mut n = first;
    let mut p = &COEFFS_PROBA0[coeff_type][KBANDS[n]][context];

    if last < 0 {
        bool_coder.put_bit(false, p[0]); // EOB
        return false;
    }

    bool_coder.put_bit(true, p[0]); // Has non-zero coefficients

    while n < 16 {
        let c = coeffs[ZIGZAG[n]];
        let v = c.unsigned_abs() as u32;
        let sign = c < 0;
        n += 1;

        if v == 0 {
            bool_coder.put_bit(false, p[1]);
            p = &COEFFS_PROBA0[coeff_type][KBANDS[n]][0];
            continue;
        }

        bool_coder.put_bit(true, p[1]);
        let next_ctx = if v == 1 {
            bool_coder.put_bit(false, p[2]);
            1
        } else {
            bool_coder.put_bit(true, p[2]);
            if v <= 4 {
                bool_coder.put_bit(false, p[3]);
                if v == 2 {
                    bool_coder.put_bit(false, p[4]);
                } else {
                    bool_coder.put_bit(true, p[4]);
                    bool_coder.put_bit(v == 4, p[5]);
                }
            } else if v <= 10 {
                bool_coder.put_bit(true, p[3]);
                bool_coder.put_bit(false, p[6]);
                if v <= 6 {
                    bool_coder.put_bit(false, p[7]);
                    bool_coder.put_bit(v == 6, 159);
                } else {
                    bool_coder.put_bit(true, p[7]);
                    bool_coder.put_bit((v - 7) >= 2, 165);
                    bool_coder.put_bit(((v - 7) & 1) != 0, 145);
                }
            } else {
                bool_coder.put_bit(true, p[3]);
                bool_coder.put_bit(true, p[6]);
                let res = v - 3;
                if res < 16 {
                    // Category 3 (3 bits: base 8..15)
                    bool_coder.put_bit(false, p[8]);
                    bool_coder.put_bit(false, p[9]);
                    let val = res - 8;
                    bool_coder.put_bit((val & 4) != 0, PCAT3[0]);
                    bool_coder.put_bit((val & 2) != 0, PCAT3[1]);
                    bool_coder.put_bit((val & 1) != 0, PCAT3[2]);
                } else if res < 32 {
                    // Category 4 (4 bits: base 16..31)
                    bool_coder.put_bit(false, p[8]);
                    bool_coder.put_bit(true, p[9]);
                    let val = res - 16;
                    bool_coder.put_bit((val & 8) != 0, PCAT4[0]);
                    bool_coder.put_bit((val & 4) != 0, PCAT4[1]);
                    bool_coder.put_bit((val & 2) != 0, PCAT4[2]);
                    bool_coder.put_bit((val & 1) != 0, PCAT4[3]);
                } else if res < 64 {
                    // Category 5 (5 bits: base 32..63)
                    bool_coder.put_bit(true, p[8]);
                    bool_coder.put_bit(false, p[10]);
                    let val = res - 32;
                    bool_coder.put_bit((val & 16) != 0, PCAT5[0]);
                    bool_coder.put_bit((val & 8) != 0, PCAT5[1]);
                    bool_coder.put_bit((val & 4) != 0, PCAT5[2]);
                    bool_coder.put_bit((val & 2) != 0, PCAT5[3]);
                    bool_coder.put_bit((val & 1) != 0, PCAT5[4]);
                } else {
                    // Category 6 (11 bits: base 64..2047)
                    bool_coder.put_bit(true, p[8]);
                    bool_coder.put_bit(true, p[10]);
                    let val = (res - 64).min(2047);
                    for (bit_idx, &prob) in PCAT6.iter().enumerate() {
                        bool_coder.put_bit(((val >> (10 - bit_idx)) & 1) != 0, prob);
                    }
                }
            }
            2
        };

        bool_coder.put_bit_equi(sign);

        if n == 16 {
            return true;
        }

        p = &COEFFS_PROBA0[coeff_type][KBANDS[n]][next_ctx];
        if (n as i32) <= last {
            bool_coder.put_bit(true, p[0]);
        } else {
            bool_coder.put_bit(false, p[0]);
            return true;
        }
    }
    true
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
    vp8_part0_buf: &mut Vec<u8>,
    vp8_tokens_buf: &mut Vec<u8>,
    vp8_output_buf: &mut Vec<u8>,
    config: &EncoderConfig,
) {
    let mb_cols = pad_width / 16;
    let mb_rows = pad_height / 16;

    let q_idx = quality_to_q_index(config.quality);
    let _q_y1_dc = FastQuantizer::new(DC_QLOOKUP[q_idx]);
    let q_y1_ac = FastQuantizer::new(AC_QLOOKUP[q_idx]);
    let q_y2_dc = FastQuantizer::new(DC_QLOOKUP[q_idx] * 2);
    let mut y2ac = AC_QLOOKUP[q_idx] * 155 / 100;
    if y2ac < 8 { y2ac = 8; }
    let q_y2_ac = FastQuantizer::new(y2ac);
    let q_uv_dc = FastQuantizer::new(DC_QLOOKUP[q_idx]);
    let q_uv_ac = FastQuantizer::new(AC_QLOOKUP[q_idx]);

    // ==========================================
    // Partition 0: Frame Headers & Intra Modes
    // ==========================================
    let mut part0_coder = BoolEncoder::new(vp8_part0_buf);

    // 1. Color space & clamping (2 bits)
    part0_coder.put_bit_equi(false); // color_space: 0 (YUV)
    part0_coder.put_bit_equi(false); // clamping_type: 0 (clamping required)

    // 2. Segmentation (1 bit)
    part0_coder.put_bit_equi(false); // segmentation_enabled: 0

    // 3. Loop filter header
    part0_coder.put_bit_equi(false); // filter_type: 0 (normal)
    part0_coder.put_literal(0, 6);   // loop_filter_level: 0
    part0_coder.put_literal(0, 3);   // sharpness_level: 0
    part0_coder.put_bit_equi(false); // loop_filter_adj_enable: 0

    // 4. Token partition count (2 bits) -> 0 means 1 DCT partition (2^0 = 1)
    part0_coder.put_literal(0, 2);   // log2_nbr_of_dct_partitions: 0

    // 5. Dequantization indices
    part0_coder.put_literal(q_idx as u32, 7); // yac_qi: 7 bits
    part0_coder.put_bit_equi(false); // ydc_delta present: 0
    part0_coder.put_bit_equi(false); // y2dc_delta present: 0
    part0_coder.put_bit_equi(false); // y2ac_delta present: 0
    part0_coder.put_bit_equi(false); // uvdc_delta present: 0
    part0_coder.put_bit_equi(false); // uvac_delta present: 0

    // 6. Refresh entropy probs (1 bit)
    part0_coder.put_bit_equi(false); // refresh_entropy_probs: 0

    // 7. Token probability update
    for i in 0..4 {
        for j in 0..8 {
            for k in 0..3 {
                for l in 0..11 {
                    let prob = crate::vp8::header_tables::COEFF_UPDATE_PROBS[i][j][k][l];
                    part0_coder.put_bit(false, prob);
                }
            }
        }
    }

    // 8. mb_no_skip_coeff (1 bit)
    part0_coder.put_bit_equi(false); // mb_no_skip_coeff: 0 (no skipping macroblock flag)

    // 9. Macroblock Prediction Modes for all macroblocks in scanline order
    for mb_y in 0..mb_rows {
        for mb_x in 0..mb_cols {
            if mb_y > 0 && mb_x > 0 {
                // TM_PRED (Mode 4 Luma, Mode 3 Chroma)
                part0_coder.put_bit(true, 145);
                part0_coder.put_bit(true, 156);
                part0_coder.put_bit(true, 163);
                part0_coder.put_bit(true, 128);

                part0_coder.put_bit(true, 142);
                part0_coder.put_bit(true, 114);
                part0_coder.put_bit(true, 183);
            } else if mb_y > 0 {
                // V_PRED (Mode 2 Luma, Mode 1 Chroma)
                part0_coder.put_bit(true, 145);
                part0_coder.put_bit(true, 156);
                part0_coder.put_bit(false, 163);

                part0_coder.put_bit(true, 142);
                part0_coder.put_bit(false, 114);
            } else if mb_x > 0 {
                // H_PRED (Mode 3 Luma, Mode 2 Chroma)
                part0_coder.put_bit(true, 145);
                part0_coder.put_bit(true, 156);
                part0_coder.put_bit(true, 163);
                part0_coder.put_bit(false, 128);

                part0_coder.put_bit(true, 142);
                part0_coder.put_bit(true, 114);
                part0_coder.put_bit(false, 183);
            } else {
                // DC_PRED (Mode 1 Luma, Mode 0 Chroma)
                part0_coder.put_bit(true, 145);
                part0_coder.put_bit(false, 156);

                part0_coder.put_bit(false, 142);
            }
        }
    }

    part0_coder.finish();

    // ==========================================
    // Partition 1: DCT / WHT Residual Tokens
    // ==========================================
    let mut token_coder = BoolEncoder::new(vp8_tokens_buf);

    let mut y_dc_coeffs = [0i16; 16];
    let mut y_wht_coeffs = [0i16; 16];
    let mut y_q_wht = [0i16; 16];
    let mut y_deq_wht = [0i16; 16];
    let mut y_rec_dc = [0i16; 16];

    let mut sub_coeffs = [[0i16; 16]; 16];
    let mut sub_q = [[0i16; 16]; 16];

    let y_stride = pad_width;
    let uv_stride = pad_width / 2;

    let mut above_y2_nz = vec![0u8; mb_cols];
    let mut above_y1_nz = vec![0u8; mb_cols * 4];
    let mut above_u_nz = vec![0u8; mb_cols * 2];
    let mut above_v_nz = vec![0u8; mb_cols * 2];

    for mb_y in 0..mb_rows {
        let mut left_y2_nz = 0u8;
        let mut left_y1_nz = [0u8; 4];
        let mut left_u_nz = [0u8; 2];
        let mut left_v_nz = [0u8; 2];

        for mb_x in 0..mb_cols {
            let mb_x_px = mb_x * 16;
            let mb_y_px = mb_y * 16;
            let uv_x_px = mb_x * 8;
            let uv_y_px = mb_y * 8;

            let mut pred_y = [0u8; 256];
            build_pred_16x16(mb_x, mb_y, mb_x_px, mb_y_px, y_stride, recon_y, &mut pred_y);

            let src_y_off = mb_y_px * y_stride + mb_x_px;

            // 16 Luma 4x4 subblocks
            for blk in 0..16 {
                let blk_x = (blk & 3) * 4;
                let blk_y = (blk >> 2) * 4;
                let p_idx = blk_y * 16 + blk_x;
                let s_idx = src_y_off + blk_y * y_stride + blk_x;
                let mut res = [0i16; 16];
                sub_pred_4x4(y_plane, s_idx, y_stride, &pred_y, p_idx, 16, &mut res);
                let has_diff = fdct_4x4(&res, &mut sub_coeffs[blk]);
                y_dc_coeffs[blk] = if has_diff { sub_coeffs[blk][0] } else { 0 };
            }

            forward_wht_4x4(&y_dc_coeffs, &mut y_wht_coeffs);
            quantize_dequantize_block(&y_wht_coeffs, &q_y2_dc, &q_y2_ac, &mut y_q_wht, &mut y_deq_wht);
            inverse_wht_4x4(&y_deq_wht, &mut y_rec_dc);

            let y2_ctx = (above_y2_nz[mb_x] + left_y2_nz) as usize;
            let y2_nz = encode_coeffs_block(&y_q_wht, 1, 0, y2_ctx, &mut token_coder);
            let y2_nz_u8 = if y2_nz { 1 } else { 0 };
            above_y2_nz[mb_x] = y2_nz_u8;
            left_y2_nz = y2_nz_u8;

            for blk in 0..16 {
                let blk_x = (blk & 3) * 4;
                let blk_y = (blk >> 2) * 4;
                let p_idx = blk_y * 16 + blk_x;
                let sub_x = blk & 3;
                let sub_y = blk >> 2;
                let col = mb_x * 4 + sub_x;
                let ctx = (above_y1_nz[col] + left_y1_nz[sub_y]) as usize;

                let mut dequant = [0i16; 16];
                quantize_dequantize_y1_ac(&sub_coeffs[blk], &q_y1_ac, y_rec_dc[blk], &mut sub_q[blk], &mut dequant);
                let sub_nz = encode_coeffs_block(&sub_q[blk], 0, 1, ctx, &mut token_coder);
                let sub_nz_u8 = if sub_nz { 1 } else { 0 };
                above_y1_nz[col] = sub_nz_u8;
                left_y1_nz[sub_y] = sub_nz_u8;

                let mut rec_res = [0i16; 16];
                idct_4x4(&dequant, &mut rec_res);

                let r_idx = (mb_y_px + blk_y) * y_stride + (mb_x_px + blk_x);
                add_pred_and_clamp_4x4(&rec_res, &pred_y, p_idx, 16, recon_y, r_idx, y_stride);
            }

            // Chroma U
            let mut pred_u = [0u8; 64];
            build_pred_8x8(mb_x, mb_y, uv_x_px, uv_y_px, uv_stride, recon_u, &mut pred_u);
            let src_u_off = uv_y_px * uv_stride + uv_x_px;

            for blk in 0..4 {
                let blk_x = (blk & 1) * 4;
                let blk_y = (blk >> 1) * 4;
                let p_u_idx = blk_y * 8 + blk_x;
                let s_u_idx = src_u_off + blk_y * uv_stride + blk_x;
                let sub_x = blk & 1;
                let sub_y = blk >> 1;
                let col = mb_x * 2 + sub_x;
                let ctx = (above_u_nz[col] + left_u_nz[sub_y]) as usize;

                let mut res = [0i16; 16];
                sub_pred_4x4(u_plane, s_u_idx, uv_stride, &pred_u, p_u_idx, 8, &mut res);

                let mut coeffs = [0i16; 16];
                let mut q_coeffs = [0i16; 16];
                let mut dequant = [0i16; 16];
                let mut rec_res = [0i16; 16];

                fdct_4x4(&res, &mut coeffs);
                quantize_dequantize_block(&coeffs, &q_uv_dc, &q_uv_ac, &mut q_coeffs, &mut dequant);
                let uv_nz = encode_coeffs_block(&q_coeffs, 2, 0, ctx, &mut token_coder);
                let uv_nz_u8 = if uv_nz { 1 } else { 0 };
                above_u_nz[col] = uv_nz_u8;
                left_u_nz[sub_y] = uv_nz_u8;

                idct_4x4(&dequant, &mut rec_res);

                let r_idx = (uv_y_px + blk_y) * uv_stride + (uv_x_px + blk_x);
                add_pred_and_clamp_4x4(&rec_res, &pred_u, p_u_idx, 8, recon_u, r_idx, uv_stride);
            }

            // Chroma V
            let mut pred_v = [0u8; 64];
            build_pred_8x8(mb_x, mb_y, uv_x_px, uv_y_px, uv_stride, recon_v, &mut pred_v);
            let src_v_off = uv_y_px * uv_stride + uv_x_px;

            for blk in 0..4 {
                let blk_x = (blk & 1) * 4;
                let blk_y = (blk >> 1) * 4;
                let p_v_idx = blk_y * 8 + blk_x;
                let s_v_idx = src_v_off + blk_y * uv_stride + blk_x;
                let sub_x = blk & 1;
                let sub_y = blk >> 1;
                let col = mb_x * 2 + sub_x;
                let ctx = (above_v_nz[col] + left_v_nz[sub_y]) as usize;

                let mut res = [0i16; 16];
                sub_pred_4x4(v_plane, s_v_idx, uv_stride, &pred_v, p_v_idx, 8, &mut res);

                let mut coeffs = [0i16; 16];
                let mut q_coeffs = [0i16; 16];
                let mut dequant = [0i16; 16];
                let mut rec_res = [0i16; 16];

                fdct_4x4(&res, &mut coeffs);
                quantize_dequantize_block(&coeffs, &q_uv_dc, &q_uv_ac, &mut q_coeffs, &mut dequant);
                let uv_nz = encode_coeffs_block(&q_coeffs, 2, 0, ctx, &mut token_coder);
                let uv_nz_u8 = if uv_nz { 1 } else { 0 };
                above_v_nz[col] = uv_nz_u8;
                left_v_nz[sub_y] = uv_nz_u8;

                idct_4x4(&dequant, &mut rec_res);

                let r_idx = (uv_y_px + blk_y) * uv_stride + (uv_x_px + blk_x);
                add_pred_and_clamp_4x4(&rec_res, &pred_v, p_v_idx, 8, recon_v, r_idx, uv_stride);
            }
        }
    }

    token_coder.finish();

    // ==========================================
    // Assemble Final VP8 Keyframe Frame Buffer
    // ==========================================
    let part0_len = vp8_part0_buf.len();
    let tokens_len = vp8_tokens_buf.len();
    let total_len = 10 + part0_len + tokens_len;

    vp8_output_buf.clear();
    vp8_output_buf.reserve(total_len);

    // Frame tag (3 bytes LE):
    // bit 0: key_frame = 0
    // bits 1-3: version = 0
    // bit 4: show_frame = 1
    // bits 5-23: first_part_size = part0_len
    let tag = ((part0_len as u32) << 5) | 0x10;
    vp8_output_buf.push((tag & 0xFF) as u8);
    vp8_output_buf.push(((tag >> 8) & 0xFF) as u8);
    vp8_output_buf.push(((tag >> 16) & 0xFF) as u8);

    // RFC 6386 magic start code (0x9D, 0x01, 0x2A)
    vp8_output_buf.extend_from_slice(&VP8_START_CODE);

    // Dimensions (4 bytes LE)
    vp8_output_buf.push((width & 0xFF) as u8);
    vp8_output_buf.push(((width >> 8) & 0x3F) as u8);
    vp8_output_buf.push((height & 0xFF) as u8);
    vp8_output_buf.push(((height >> 8) & 0x3F) as u8);

    // Partition 0 payload
    vp8_output_buf.extend_from_slice(vp8_part0_buf);

    // Partition 1 (Tokens) payload
    vp8_output_buf.extend_from_slice(vp8_tokens_buf);
}

