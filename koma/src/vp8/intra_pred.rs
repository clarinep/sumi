//! High-performance VP8 16x16 and 8x8 Intra Prediction Kernels.
//!
//! Evaluates DC, Vertical (V_PRED), Horizontal (H_PRED), and TrueMotion (TM_PRED)
//! modes for Luma (16x16) and Chroma (8x8) using hardware SIMD (AVX2 / SSE2 / NEON).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// VP8 Intra 16x16 Luma Prediction Modes (RFC 6386 Section 11.1).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intra16x16Mode {
    /// Vertical prediction (replicate row above).
    V = 0,
    /// Horizontal prediction (replicate column to the left).
    H = 1,
    /// DC prediction (average of top row and left column).
    DC = 2,
    /// TrueMotion prediction (gradient: Top + Left - TopLeft).
    TM = 3,
}

/// VP8 Intra 8x8 Chroma Prediction Modes (RFC 6386 Section 11.4).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraUVMode {
    /// DC prediction.
    DC = 0,
    /// Vertical prediction.
    V = 1,
    /// Horizontal prediction.
    H = 2,
    /// TrueMotion prediction.
    TM = 3,
}

/// Computes the Sum of Absolute Differences (SAD) between 16x16 pixels and predictor using SIMD.
#[inline(always)]
pub fn sad_16x16(src: &[u8], src_stride: usize, pred: &[u8; 256]) -> u32 {
    let mut total_sad = 0u32;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        for y in 0..16 {
            let s_ptr = src.as_ptr().add(y * src_stride);
            let p_ptr = pred.as_ptr().add(y * 16);
            let s_vec = vld1q_u8(s_ptr);
            let p_vec = vld1q_u8(p_ptr);
            let diff = vabdq_u8(s_vec, p_vec);
            total_sad += vaddlvq_u8(diff) as u32;
        }
        return total_sad;
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut acc = _mm_setzero_si128();
        for y in 0..16 {
            let s_ptr = src.as_ptr().add(y * src_stride);
            let p_ptr = pred.as_ptr().add(y * 16);
            let s_vec = _mm_loadu_si128(s_ptr as *const __m128i);
            let p_vec = _mm_loadu_si128(p_ptr as *const __m128i);
            let sad = _mm_sad_epu8(s_vec, p_vec);
            acc = _mm_add_epi64(acc, sad);
        }
        let lo = _mm_cvtsi128_si32(acc) as u32;
        let hi = _mm_extract_epi16::<4>(acc) as u32;
        return lo + hi;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        for y in 0..16 {
            let s_row = y * src_stride;
            let p_row = y * 16;
            for x in 0..16 {
                let diff = (src[s_row + x] as i32 - pred[p_row + x] as i32).abs() as u32;
                total_sad += diff;
            }
        }
        total_sad
    }
}

/// Computes the SAD between 8x8 pixels and predictor using SIMD.
#[inline(always)]
pub fn sad_8x8(src: &[u8], src_stride: usize, pred: &[u8; 64]) -> u32 {
    let mut total_sad = 0u32;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        for y in 0..8 {
            let s_ptr = src.as_ptr().add(y * src_stride);
            let p_ptr = pred.as_ptr().add(y * 8);
            let s_vec = vld1_u8(s_ptr);
            let p_vec = vld1_u8(p_ptr);
            let diff = vabd_u8(s_vec, p_vec);
            total_sad += vaddlv_u8(diff) as u32;
        }
        return total_sad;
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut acc = _mm_setzero_si128();
        for y in (0..8).step_by(2) {
            let s_ptr0 = src.as_ptr().add(y * src_stride) as *const u64;
            let s_ptr1 = src.as_ptr().add((y + 1) * src_stride) as *const u64;
            let p_ptr0 = pred.as_ptr().add(y * 8) as *const u64;
            let p_ptr1 = pred.as_ptr().add((y + 1) * 8) as *const u64;

            let s0 = std::ptr::read_unaligned(s_ptr0);
            let s1 = std::ptr::read_unaligned(s_ptr1);
            let p0 = std::ptr::read_unaligned(p_ptr0);
            let p1 = std::ptr::read_unaligned(p_ptr1);

            let s_vec = _mm_set_epi64x(s1 as i64, s0 as i64);
            let p_vec = _mm_set_epi64x(p1 as i64, p0 as i64);

            let sad = _mm_sad_epu8(s_vec, p_vec);
            acc = _mm_add_epi64(acc, sad);
        }
        let lo = _mm_cvtsi128_si32(acc) as u32;
        let hi = _mm_extract_epi16::<4>(acc) as u32;
        return lo + hi;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        for y in 0..8 {
            let s_row = y * src_stride;
            let p_row = y * 8;
            for x in 0..8 {
                let diff = (src[s_row + x] as i32 - pred[p_row + x] as i32).abs() as u32;
                total_sad += diff;
            }
        }
        total_sad
    }
}

/// Fills 16x16 DC prediction buffer.
#[inline(always)]
pub fn predict_16x16_dc(
    recon: &[u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    out: &mut [u8; 256],
) -> u8 {
    let mb_x_px = mb_x * 16;
    let mb_y_px = mb_y * 16;

    let mut sum = 0u32;
    let mut count = 0u32;

    if mb_y > 0 {
        let top_off = (mb_y_px - 1) * stride + mb_x_px;
        for x in 0..16 {
            sum += recon[top_off + x] as u32;
        }
        count += 16;
    }

    if mb_x > 0 {
        for y in 0..16 {
            let left_val = recon[(mb_y_px + y) * stride + (mb_x_px - 1)];
            sum += left_val as u32;
        }
        count += 16;
    }

    let dc = match count {
        32 => ((sum + 16) >> 5) as u8,
        16 => ((sum + 8) >> 4) as u8,
        _ => 128,
    };

    out.fill(dc);
    dc
}

/// Fills 16x16 Vertical prediction buffer.
#[inline(always)]
pub fn predict_16x16_v(
    recon: &[u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    out: &mut [u8; 256],
) {
    if mb_y == 0 {
        out.fill(128);
        return;
    }
    let top_off = (mb_y * 16 - 1) * stride + (mb_x * 16);
    let top_row = &recon[top_off..top_off + 16];
    for y in 0..16 {
        out[y * 16..(y + 1) * 16].copy_from_slice(top_row);
    }
}

/// Fills 16x16 Horizontal prediction buffer.
#[inline(always)]
pub fn predict_16x16_h(
    recon: &[u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    out: &mut [u8; 256],
) {
    if mb_x == 0 {
        out.fill(128);
        return;
    }
    let mb_x_px = mb_x * 16;
    let mb_y_px = mb_y * 16;
    for y in 0..16 {
        let left_val = recon[(mb_y_px + y) * stride + (mb_x_px - 1)];
        out[y * 16..(y + 1) * 16].fill(left_val);
    }
}

/// Fills 16x16 TrueMotion (TM_PRED) buffer: `clip(Left[y] + Top[x] - TopLeft)`.
#[inline(always)]
pub fn predict_16x16_tm(
    recon: &[u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    out: &mut [u8; 256],
) {
    if mb_x == 0 && mb_y == 0 {
        out.fill(128);
        return;
    }
    if mb_y == 0 {
        predict_16x16_h(recon, stride, mb_x, mb_y, out);
        return;
    }
    if mb_x == 0 {
        predict_16x16_v(recon, stride, mb_x, mb_y, out);
        return;
    }

    let mb_x_px = mb_x * 16;
    let mb_y_px = mb_y * 16;

    let top_left = recon[(mb_y_px - 1) * stride + (mb_x_px - 1)] as i32;
    let top_off = (mb_y_px - 1) * stride + mb_x_px;
    let top_row = &recon[top_off..top_off + 16];

    for y in 0..16 {
        let left_val = recon[(mb_y_px + y) * stride + (mb_x_px - 1)] as i32;
        let base = left_val - top_left;
        let out_row = y * 16;
        for x in 0..16 {
            let pred = base + (top_row[x] as i32);
            out[out_row + x] = pred.clamp(0, 255) as u8;
        }
    }
}

/// Fills 8x8 DC prediction buffer for Chroma.
#[inline(always)]
pub fn predict_8x8_dc(
    recon: &[u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    out: &mut [u8; 64],
) -> u8 {
    let uv_x_px = mb_x * 8;
    let uv_y_px = mb_y * 8;

    let mut sum = 0u32;
    let mut count = 0u32;

    if mb_y > 0 {
        let top_off = (uv_y_px - 1) * stride + uv_x_px;
        for x in 0..8 {
            sum += recon[top_off + x] as u32;
        }
        count += 8;
    }

    if mb_x > 0 {
        for y in 0..8 {
            let left_val = recon[(uv_y_px + y) * stride + (uv_x_px - 1)];
            sum += left_val as u32;
        }
        count += 8;
    }

    let dc = match count {
        16 => ((sum + 8) >> 4) as u8,
        8 => ((sum + 4) >> 3) as u8,
        _ => 128,
    };

    out.fill(dc);
    dc
}

/// Fills 8x8 Vertical prediction buffer for Chroma.
#[inline(always)]
pub fn predict_8x8_v(
    recon: &[u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    out: &mut [u8; 64],
) {
    if mb_y == 0 {
        out.fill(128);
        return;
    }
    let top_off = (mb_y * 8 - 1) * stride + (mb_x * 8);
    let top_row = &recon[top_off..top_off + 8];
    for y in 0..8 {
        out[y * 8..(y + 1) * 8].copy_from_slice(top_row);
    }
}

/// Fills 8x8 Horizontal prediction buffer for Chroma.
#[inline(always)]
pub fn predict_8x8_h(
    recon: &[u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    out: &mut [u8; 64],
) {
    if mb_x == 0 {
        out.fill(128);
        return;
    }
    let uv_x_px = mb_x * 8;
    let uv_y_px = mb_y * 8;
    for y in 0..8 {
        let left_val = recon[(uv_y_px + y) * stride + (uv_x_px - 1)];
        out[y * 8..(y + 1) * 8].fill(left_val);
    }
}

/// Fills 8x8 TrueMotion prediction buffer for Chroma.
#[inline(always)]
pub fn predict_8x8_tm(
    recon: &[u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    out: &mut [u8; 64],
) {
    if mb_x == 0 && mb_y == 0 {
        out.fill(128);
        return;
    }
    if mb_y == 0 {
        predict_8x8_h(recon, stride, mb_x, mb_y, out);
        return;
    }
    if mb_x == 0 {
        predict_8x8_v(recon, stride, mb_x, mb_y, out);
        return;
    }

    let uv_x_px = mb_x * 8;
    let uv_y_px = mb_y * 8;

    let top_left = recon[(uv_y_px - 1) * stride + (uv_x_px - 1)] as i32;
    let top_off = (uv_y_px - 1) * stride + uv_x_px;
    let top_row = &recon[top_off..top_off + 8];

    for y in 0..8 {
        let left_val = recon[(uv_y_px + y) * stride + (uv_x_px - 1)] as i32;
        let base = left_val - top_left;
        let out_row = y * 8;
        for x in 0..8 {
            let pred = base + (top_row[x] as i32);
            out[out_row + x] = pred.clamp(0, 255) as u8;
        }
    }
}

/// Evaluates all available 16x16 Luma prediction modes and returns the one with lowest SAD.
#[inline(always)]
pub fn select_best_16x16_luma_mode(
    src: &[u8],
    src_stride: usize,
    recon: &[u8],
    recon_stride: usize,
    mb_x: usize,
    mb_y: usize,
    best_pred_buf: &mut [u8; 256],
) -> Intra16x16Mode {
    let mut cand_buf = [0u8; 256];

    // 1. DC Mode (Baseline)
    predict_16x16_dc(recon, recon_stride, mb_x, mb_y, &mut cand_buf);
    let mut min_sad = sad_16x16(src, src_stride, &cand_buf);
    let mut best_mode = Intra16x16Mode::DC;
    best_pred_buf.copy_from_slice(&cand_buf);

    // 2. TrueMotion Mode (Super effective for anime art, gradients, hair and lighting)
    if mb_x > 0 || mb_y > 0 {
        predict_16x16_tm(recon, recon_stride, mb_x, mb_y, &mut cand_buf);
        let tm_sad = sad_16x16(src, src_stride, &cand_buf);
        if tm_sad < min_sad {
            min_sad = tm_sad;
            best_mode = Intra16x16Mode::TM;
            best_pred_buf.copy_from_slice(&cand_buf);
        }
    }

    // 3. Vertical Mode
    if mb_y > 0 {
        predict_16x16_v(recon, recon_stride, mb_x, mb_y, &mut cand_buf);
        let v_sad = sad_16x16(src, src_stride, &cand_buf);
        if v_sad < min_sad {
            min_sad = v_sad;
            best_mode = Intra16x16Mode::V;
            best_pred_buf.copy_from_slice(&cand_buf);
        }
    }

    // 4. Horizontal Mode
    if mb_x > 0 {
        predict_16x16_h(recon, recon_stride, mb_x, mb_y, &mut cand_buf);
        let h_sad = sad_16x16(src, src_stride, &cand_buf);
        if h_sad < min_sad {
            best_mode = Intra16x16Mode::H;
            best_pred_buf.copy_from_slice(&cand_buf);
        }
    }

    best_mode
}

/// Evaluates all 8x8 Chroma prediction modes across both U and V planes and returns the joint best mode.
#[inline(always)]
pub fn select_best_chroma_mode(
    u_src: &[u8],
    v_src: &[u8],
    uv_stride: usize,
    recon_u: &[u8],
    recon_v: &[u8],
    mb_x: usize,
    mb_y: usize,
    best_u_pred: &mut [u8; 64],
    best_v_pred: &mut [u8; 64],
) -> IntraUVMode {
    let mut u_cand = [0u8; 64];
    let mut v_cand = [0u8; 64];

    // 1. DC Mode
    predict_8x8_dc(recon_u, uv_stride, mb_x, mb_y, &mut u_cand);
    predict_8x8_dc(recon_v, uv_stride, mb_x, mb_y, &mut v_cand);
    let mut min_sad = sad_8x8(u_src, uv_stride, &u_cand) + sad_8x8(v_src, uv_stride, &v_cand);
    let mut best_mode = IntraUVMode::DC;
    best_u_pred.copy_from_slice(&u_cand);
    best_v_pred.copy_from_slice(&v_cand);

    // 2. TrueMotion Mode
    if mb_x > 0 || mb_y > 0 {
        predict_8x8_tm(recon_u, uv_stride, mb_x, mb_y, &mut u_cand);
        predict_8x8_tm(recon_v, uv_stride, mb_x, mb_y, &mut v_cand);
        let tm_sad = sad_8x8(u_src, uv_stride, &u_cand) + sad_8x8(v_src, uv_stride, &v_cand);
        if tm_sad < min_sad {
            min_sad = tm_sad;
            best_mode = IntraUVMode::TM;
            best_u_pred.copy_from_slice(&u_cand);
            best_v_pred.copy_from_slice(&v_cand);
        }
    }

    // 3. Vertical Mode
    if mb_y > 0 {
        predict_8x8_v(recon_u, uv_stride, mb_x, mb_y, &mut u_cand);
        predict_8x8_v(recon_v, uv_stride, mb_x, mb_y, &mut v_cand);
        let v_sad = sad_8x8(u_src, uv_stride, &u_cand) + sad_8x8(v_src, uv_stride, &v_cand);
        if v_sad < min_sad {
            min_sad = v_sad;
            best_mode = IntraUVMode::V;
            best_u_pred.copy_from_slice(&u_cand);
            best_v_pred.copy_from_slice(&v_cand);
        }
    }

    // 4. Horizontal Mode
    if mb_x > 0 {
        predict_8x8_h(recon_u, uv_stride, mb_x, mb_y, &mut u_cand);
        predict_8x8_h(recon_v, uv_stride, mb_x, mb_y, &mut v_cand);
        let h_sad = sad_8x8(u_src, uv_stride, &u_cand) + sad_8x8(v_src, uv_stride, &v_cand);
        if h_sad < min_sad {
            best_mode = IntraUVMode::H;
            best_u_pred.copy_from_slice(&u_cand);
            best_v_pred.copy_from_slice(&v_cand);
        }
    }

    best_mode
}
