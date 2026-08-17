//! Discrete Cosine Transform (FDCT/IDCT) and Walsh-Hadamard Transform (WHT) kernels.
//!
//! Provides hardware-accelerated (SSE2/SSSE3/AVX2) and branchless integer fixed-point 4x4
//! transform arithmetic defined in RFC 6386 Sections 14.1–14.4.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn _mm_mullo_epi32_portable(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let mul02 = _mm_mul_epu32(a, b);
        let a_hi = _mm_srli_si128(a, 4);
        let b_hi = _mm_srli_si128(b, 4);
        let mul13 = _mm_mul_epu32(a_hi, b_hi);
        let unpack_lo = _mm_unpacklo_epi32(mul02, mul13);
        let unpack_hi = _mm_unpackhi_epi32(mul02, mul13);
        _mm_unpacklo_epi64(unpack_lo, unpack_hi)
    }
}

/// 4x4 Forward Discrete Cosine Transform (FDCT) according to RFC 6386 Section 14.3.
///
/// Transforms 16 residual difference values into 16 frequency-domain DCT coefficients.
/// Returns `true` if any non-zero input was detected.
#[inline(always)]
pub fn fdct_4x4(input: &[i16; 16], output: &mut [i16; 16]) -> bool {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let in0_16 = vld1q_s16(input.as_ptr());
        let in1_16 = vld1q_s16(input.as_ptr().add(8));
        let zero_16 = vdupq_n_s16(0);
        let eq0 = vceqq_s16(in0_16, zero_16);
        let eq1 = vceqq_s16(in1_16, zero_16);
        let and = vandq_u16(eq0, eq1);
        if vminvq_u16(and) == 0xFFFF {
            output.fill(0);
            return false;
        }

        // Pass 1: Row transform
        let row0 = vmovl_s16(vget_low_s16(in0_16));
        let row1 = vmovl_s16(vget_high_s16(in0_16));
        let row2 = vmovl_s16(vget_low_s16(in1_16));
        let row3 = vmovl_s16(vget_high_s16(in1_16));

        // Transpose 4x4 int32 to get columns across rows
        let trn0 = vzip1q_s32(row0, row2);
        let trn1 = vzip2q_s32(row0, row2);
        let trn2 = vzip1q_s32(row1, row3);
        let trn3 = vzip2q_s32(row1, row3);

        let v0 = vzip1q_s32(trn0, trn2);
        let v1 = vzip2q_s32(trn0, trn2);
        let v2 = vzip1q_s32(trn1, trn3);
        let v3 = vzip2q_s32(trn1, trn3);

        let a1 = vaddq_s32(v0, v3);
        let b1 = vaddq_s32(v1, v2);
        let c1 = vsubq_s32(v1, v2);
        let d1 = vsubq_s32(v0, v3);

        let t0 = vshlq_n_s32(vaddq_s32(a1, b1), 3);
        let t2 = vshlq_n_s32(vsubq_s32(a1, b1), 3);

        let mut t1 = vdupq_n_s32(14500);
        t1 = vmlaq_n_s32(t1, d1, 5352);
        t1 = vmlaq_n_s32(t1, c1, 2217);
        t1 = vshrq_n_s32(t1, 12);

        let mut t3 = vdupq_n_s32(7500);
        t3 = vmlaq_n_s32(t3, d1, 2217);
        t3 = vmlsq_n_s32(t3, c1, 5352);
        t3 = vshrq_n_s32(t3, 12);

        // Pass 2: Column transform
        let col_trn0 = vzip1q_s32(t0, t2);
        let col_trn1 = vzip2q_s32(t0, t2);
        let col_trn2 = vzip1q_s32(t1, t3);
        let col_trn3 = vzip2q_s32(t1, t3);

        let u0 = vzip1q_s32(col_trn0, col_trn2);
        let u1 = vzip2q_s32(col_trn0, col_trn2);
        let u2 = vzip1q_s32(col_trn1, col_trn3);
        let u3 = vzip2q_s32(col_trn1, col_trn3);

        let col_a1 = vaddq_s32(u0, u3);
        let col_b1 = vaddq_s32(u1, u2);
        let col_c1 = vsubq_s32(u1, u2);
        let col_d1 = vsubq_s32(u0, u3);

        let o0 = vshrq_n_s32(vaddq_s32(vaddq_s32(col_a1, col_b1), vdupq_n_s32(7)), 3);
        let o2 = vshrq_n_s32(vaddq_s32(vsubq_s32(col_a1, col_b1), vdupq_n_s32(7)), 3);

        let mut o1 = vdupq_n_s32(12000);
        o1 = vmlaq_n_s32(o1, col_d1, 5352);
        o1 = vmlaq_n_s32(o1, col_c1, 2217);
        o1 = vshrq_n_s32(o1, 16);

        let mut o3 = vdupq_n_s32(51000);
        o3 = vmlaq_n_s32(o3, col_d1, 2217);
        o3 = vmlsq_n_s32(o3, col_c1, 5352);
        o3 = vshrq_n_s32(o3, 16);

        let out01_16 = vcombine_s16(vmovn_s32(o0), vmovn_s32(o1));
        let out23_16 = vcombine_s16(vmovn_s32(o2), vmovn_s32(o3));

        vst1q_s16(output.as_mut_ptr(), out01_16);
        vst1q_s16(output.as_mut_ptr().add(8), out23_16);
        return true;
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        let in_lo = _mm_loadu_si128(input.as_ptr() as *const __m128i);
        let in_hi = _mm_loadu_si128(input.as_ptr().add(8) as *const __m128i);
        let zero = _mm_setzero_si128();
        let eq_lo = _mm_cmpeq_epi16(in_lo, zero);
        let eq_hi = _mm_cmpeq_epi16(in_hi, zero);
        let mask_lo = _mm_movemask_epi8(eq_lo);
        let mask_hi = _mm_movemask_epi8(eq_hi);
        if mask_lo == 0xFFFF && mask_hi == 0xFFFF {
            output.fill(0);
            return false;
        }

        // Pass 1: Row transform (32-bit unpacked)
        let row0 = _mm_srai_epi32(_mm_unpacklo_epi16(zero, in_lo), 16);
        let row1 = _mm_srai_epi32(_mm_unpackhi_epi16(zero, in_lo), 16);
        let row2 = _mm_srai_epi32(_mm_unpacklo_epi16(zero, in_hi), 16);
        let row3 = _mm_srai_epi32(_mm_unpackhi_epi16(zero, in_hi), 16);

        // Transpose in SSE
        let t_lo0 = _mm_unpacklo_epi32(row0, row1);
        let t_hi0 = _mm_unpackhi_epi32(row0, row1);
        let t_lo1 = _mm_unpacklo_epi32(row2, row3);
        let t_hi1 = _mm_unpackhi_epi32(row2, row3);

        let v0 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(t_lo0), _mm_castsi128_ps(t_lo1)));
        let v1 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(t_lo1), _mm_castsi128_ps(t_lo0)));
        let v2 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(t_hi0), _mm_castsi128_ps(t_hi1)));
        let v3 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(t_hi1), _mm_castsi128_ps(t_hi0)));

        let a1 = _mm_add_epi32(v0, v3);
        let b1 = _mm_add_epi32(v1, v2);
        let c1 = _mm_sub_epi32(v1, v2);
        let d1 = _mm_sub_epi32(v0, v3);

        let t0 = _mm_slli_epi32(_mm_add_epi32(a1, b1), 3);
        let t2 = _mm_slli_epi32(_mm_sub_epi32(a1, b1), 3);

        let k5352 = _mm_set1_epi32(5352);
        let k2217 = _mm_set1_epi32(2217);

        let d1_5352 = _mm_mullo_epi32_portable(d1, k5352);
        let c1_2217 = _mm_mullo_epi32_portable(c1, k2217);
        let d1_2217 = _mm_mullo_epi32_portable(d1, k2217);
        let c1_5352 = _mm_mullo_epi32_portable(c1, k5352);

        let t1 = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(d1_5352, c1_2217), _mm_set1_epi32(14500)), 12);
        let t3 = _mm_srai_epi32(_mm_add_epi32(_mm_sub_epi32(d1_2217, c1_5352), _mm_set1_epi32(7500)), 12);

        // Pass 2: Column transform
        let ct_lo0 = _mm_unpacklo_epi32(t0, t1);
        let ct_hi0 = _mm_unpackhi_epi32(t0, t1);
        let ct_lo1 = _mm_unpacklo_epi32(t2, t3);
        let ct_hi1 = _mm_unpackhi_epi32(t2, t3);

        let u0 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(ct_lo0), _mm_castsi128_ps(ct_lo1)));
        let u1 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(ct_lo1), _mm_castsi128_ps(ct_lo0)));
        let u2 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(ct_hi0), _mm_castsi128_ps(ct_hi1)));
        let u3 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(ct_hi1), _mm_castsi128_ps(ct_hi0)));

        let col_a1 = _mm_add_epi32(u0, u3);
        let col_b1 = _mm_add_epi32(u1, u2);
        let col_c1 = _mm_sub_epi32(u1, u2);
        let col_d1 = _mm_sub_epi32(u0, u3);

        let o0 = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(col_a1, col_b1), _mm_set1_epi32(7)), 3);
        let o2 = _mm_srai_epi32(_mm_add_epi32(_mm_sub_epi32(col_a1, col_b1), _mm_set1_epi32(7)), 3);

        let col_d1_5352 = _mm_mullo_epi32_portable(col_d1, k5352);
        let col_c1_2217 = _mm_mullo_epi32_portable(col_c1, k2217);
        let col_d1_2217 = _mm_mullo_epi32_portable(col_d1, k2217);
        let col_c1_5352 = _mm_mullo_epi32_portable(col_c1, k5352);

        let o1 = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(col_d1_5352, col_c1_2217), _mm_set1_epi32(12000)), 16);
        let o3 = _mm_srai_epi32(_mm_add_epi32(_mm_sub_epi32(col_d1_2217, col_c1_5352), _mm_set1_epi32(51000)), 16);

        let out01_16 = _mm_packs_epi32(o0, o1);
        let out23_16 = _mm_packs_epi32(o2, o3);

        _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, out01_16);
        _mm_storeu_si128(output.as_mut_ptr().add(8) as *mut __m128i, out23_16);
        return true;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let any = (input[0] | input[1] | input[2] | input[3]
            | input[4] | input[5] | input[6] | input[7]
            | input[8] | input[9] | input[10] | input[11]
            | input[12] | input[13] | input[14] | input[15]) != 0;
        if !any {
            output.fill(0);
            return false;
        }

        let mut intermediate = [0i32; 16];

        // Row transform (Pass 1)
        for i in 0..4 {
            let row = i * 4;
            let in0 = input[row] as i32;
            let in1 = input[row + 1] as i32;
            let in2 = input[row + 2] as i32;
            let in3 = input[row + 3] as i32;

            let a1 = in0 + in3;
            let b1 = in1 + in2;
            let c1 = in1 - in2;
            let d1 = in0 - in3;

            intermediate[row] = (a1 + b1) * 8;
            intermediate[row + 1] = (d1 * 5352 + c1 * 2217 + 14500) >> 12;
            intermediate[row + 2] = (a1 - b1) * 8;
            intermediate[row + 3] = (d1 * 2217 - c1 * 5352 + 7500) >> 12;
        }

        // Column transform (Pass 2)
        for i in 0..4 {
            let in0 = intermediate[i];
            let in1 = intermediate[i + 4];
            let in2 = intermediate[i + 8];
            let in3 = intermediate[i + 12];

            let a1 = in0 + in3;
            let b1 = in1 + in2;
            let c1 = in1 - in2;
            let d1 = in0 - in3;

            output[i] = ((a1 + b1 + 7) >> 3) as i16;
            output[i + 4] = ((d1 * 5352 + c1 * 2217 + 12000) >> 16) as i16;
            output[i + 8] = ((a1 - b1 + 7) >> 3) as i16;
            output[i + 12] = ((d1 * 2217 - c1 * 5352 + 51000) >> 16) as i16;
        }

        true
    }
}

/// 4x4 Inverse Discrete Cosine Transform (IDCT) according to RFC 6386 Section 14.4.
#[inline(always)]
pub fn idct_4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let in0_16 = vld1q_s16(input.as_ptr());
        let in1_16 = vld1q_s16(input.as_ptr().add(8));
        let zero_16 = vdupq_n_s16(0);
        let eq1 = vceqq_s16(in1_16, zero_16);
        if vminvq_u16(eq1) == 0xFFFF {
            let eq0 = vceqq_s16(in0_16, zero_16);
            let mask_ac = vsetq_lane_u16(0xFFFF, eq0, 0);
            if vminvq_u16(mask_ac) == 0xFFFF && input[1] == 0 {
                let dc = input[0] as i32;
                if dc == 0 {
                    output.fill(0);
                    return;
                }
                let dc_out = ((dc + 4) >> 3) as i16;
                let vec_dc = vdupq_n_s16(dc_out);
                vst1q_s16(output.as_mut_ptr(), vec_dc);
                vst1q_s16(output.as_mut_ptr().add(8), vec_dc);
                return;
            }
        }

        // Pass 1: Column pass
        let u0 = vmovl_s16(vget_low_s16(in0_16));
        let u1 = vmovl_s16(vget_high_s16(in0_16));
        let u2 = vmovl_s16(vget_low_s16(in1_16));
        let u3 = vmovl_s16(vget_high_s16(in1_16));

        let a1 = vaddq_s32(u0, u2);
        let b1 = vsubq_s32(u0, u2);

        let temp1 = vshrq_n_s32(vmulq_n_s32(u1, 2217), 11);
        let temp2 = vaddq_s32(u3, vshrq_n_s32(vmulq_n_s32(u3, 1258), 11));
        let c1 = vsubq_s32(temp1, temp2);

        let temp3 = vaddq_s32(u1, vshrq_n_s32(vmulq_n_s32(u1, 1258), 11));
        let temp4 = vshrq_n_s32(vmulq_n_s32(u3, 2217), 11);
        let d1 = vaddq_s32(temp3, temp4);

        let t0 = vaddq_s32(a1, d1);
        let t1 = vaddq_s32(b1, c1);
        let t2 = vsubq_s32(b1, c1);
        let t3 = vsubq_s32(a1, d1);

        // Transpose 4x4 matrix
        let trn0 = vzip1q_s32(t0, t2);
        let trn1 = vzip2q_s32(t0, t2);
        let trn2 = vzip1q_s32(t1, t3);
        let trn3 = vzip2q_s32(t1, t3);

        let v0 = vzip1q_s32(trn0, trn2);
        let v1 = vzip2q_s32(trn0, trn2);
        let v2 = vzip1q_s32(trn1, trn3);
        let v3 = vzip2q_s32(trn1, trn3);

        // Pass 2: Row pass
        let row_a1 = vaddq_s32(v0, v2);
        let row_b1 = vsubq_s32(v0, v2);

        let row_temp1 = vshrq_n_s32(vmulq_n_s32(v1, 2217), 11);
        let row_temp2 = vaddq_s32(v3, vshrq_n_s32(vmulq_n_s32(v3, 1258), 11));
        let row_c1 = vsubq_s32(row_temp1, row_temp2);

        let row_temp3 = vaddq_s32(v1, vshrq_n_s32(vmulq_n_s32(v1, 1258), 11));
        let row_temp4 = vshrq_n_s32(vmulq_n_s32(v3, 2217), 11);
        let row_d1 = vaddq_s32(row_temp3, row_temp4);

        let k4 = vdupq_n_s32(4);
        let o0 = vshrq_n_s32(vaddq_s32(vaddq_s32(row_a1, row_d1), k4), 3);
        let o1 = vshrq_n_s32(vaddq_s32(vaddq_s32(row_b1, row_c1), k4), 3);
        let o2 = vshrq_n_s32(vaddq_s32(vsubq_s32(row_b1, row_c1), k4), 3);
        let o3 = vshrq_n_s32(vaddq_s32(vsubq_s32(row_a1, row_d1), k4), 3);

        let out01_16 = vcombine_s16(vmovn_s32(o0), vmovn_s32(o1));
        let out23_16 = vcombine_s16(vmovn_s32(o2), vmovn_s32(o3));

        vst1q_s16(output.as_mut_ptr(), out01_16);
        vst1q_s16(output.as_mut_ptr().add(8), out23_16);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        let in_lo = _mm_loadu_si128(input.as_ptr() as *const __m128i);
        let in_hi = _mm_loadu_si128(input.as_ptr().add(8) as *const __m128i);
        let zero = _mm_setzero_si128();
        let eq_hi = _mm_cmpeq_epi16(in_hi, zero);
        if _mm_movemask_epi8(eq_hi) == 0xFFFF {
            let eq_lo = _mm_cmpeq_epi16(in_lo, zero);
            let mask_lo = _mm_movemask_epi8(eq_lo);
            // If AC coefficients in lower 8 are also zero
            if (mask_lo & 0xFFFC) == 0xFFFC && input[1] == 0 {
                let dc = input[0] as i32;
                if dc == 0 {
                    output.fill(0);
                    return;
                }
                let dc_out = ((dc + 4) >> 3) as i16;
                let vec_dc = _mm_set1_epi16(dc_out);
                _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, vec_dc);
                _mm_storeu_si128(output.as_mut_ptr().add(8) as *mut __m128i, vec_dc);
                return;
            }
        }

        // Pass 1: Column pass (32-bit unpacked)
        let u0 = _mm_srai_epi32(_mm_unpacklo_epi16(zero, in_lo), 16);
        let u1 = _mm_srai_epi32(_mm_unpackhi_epi16(zero, in_lo), 16);
        let u2 = _mm_srai_epi32(_mm_unpacklo_epi16(zero, in_hi), 16);
        let u3 = _mm_srai_epi32(_mm_unpackhi_epi16(zero, in_hi), 16);

        let a1 = _mm_add_epi32(u0, u2);
        let b1 = _mm_sub_epi32(u0, u2);

        let k2217 = _mm_set1_epi32(2217);
        let k1258 = _mm_set1_epi32(1258);

        let temp1 = _mm_srai_epi32(_mm_mullo_epi32_portable(u1, k2217), 11);
        let temp2 = _mm_add_epi32(u3, _mm_srai_epi32(_mm_mullo_epi32_portable(u3, k1258), 11));
        let c1 = _mm_sub_epi32(temp1, temp2);

        let temp3 = _mm_add_epi32(u1, _mm_srai_epi32(_mm_mullo_epi32_portable(u1, k1258), 11));
        let temp4 = _mm_srai_epi32(_mm_mullo_epi32_portable(u3, k2217), 11);
        let d1 = _mm_add_epi32(temp3, temp4);

        let t0 = _mm_add_epi32(a1, d1);
        let t1 = _mm_add_epi32(b1, c1);
        let t2 = _mm_sub_epi32(b1, c1);
        let t3 = _mm_sub_epi32(a1, d1);

        // Transpose 4x4 matrix
        let trn_lo0 = _mm_unpacklo_epi32(t0, t1);
        let trn_hi0 = _mm_unpackhi_epi32(t0, t1);
        let trn_lo1 = _mm_unpacklo_epi32(t2, t3);
        let trn_hi1 = _mm_unpackhi_epi32(t2, t3);

        let v0 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(trn_lo0), _mm_castsi128_ps(trn_lo1)));
        let v1 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(trn_lo1), _mm_castsi128_ps(trn_lo0)));
        let v2 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(trn_hi0), _mm_castsi128_ps(trn_hi1)));
        let v3 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(trn_hi1), _mm_castsi128_ps(trn_hi0)));

        // Pass 2: Row pass
        let row_a1 = _mm_add_epi32(v0, v2);
        let row_b1 = _mm_sub_epi32(v0, v2);

        let row_temp1 = _mm_srai_epi32(_mm_mullo_epi32_portable(v1, k2217), 11);
        let row_temp2 = _mm_add_epi32(v3, _mm_srai_epi32(_mm_mullo_epi32_portable(v3, k1258), 11));
        let row_c1 = _mm_sub_epi32(row_temp1, row_temp2);

        let row_temp3 = _mm_add_epi32(v1, _mm_srai_epi32(_mm_mullo_epi32_portable(v1, k1258), 11));
        let row_temp4 = _mm_srai_epi32(_mm_mullo_epi32_portable(v3, k2217), 11);
        let row_d1 = _mm_add_epi32(row_temp3, row_temp4);

        let k4 = _mm_set1_epi32(4);
        let o0 = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(row_a1, row_d1), k4), 3);
        let o1 = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(row_b1, row_c1), k4), 3);
        let o2 = _mm_srai_epi32(_mm_add_epi32(_mm_sub_epi32(row_b1, row_c1), k4), 3);
        let o3 = _mm_srai_epi32(_mm_add_epi32(_mm_sub_epi32(row_a1, row_d1), k4), 3);

        let out01_16 = _mm_packs_epi32(o0, o1);
        let out23_16 = _mm_packs_epi32(o2, o3);

        _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, out01_16);
        _mm_storeu_si128(output.as_mut_ptr().add(8) as *mut __m128i, out23_16);
        return;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let any_ac = (input[1] | input[2] | input[3] | input[4]
            | input[5] | input[6] | input[7] | input[8]
            | input[9] | input[10] | input[11] | input[12]
            | input[13] | input[14] | input[15]) != 0;
        if !any_ac {
            let dc = input[0] as i32;
            let dc_out = ((dc + 4) >> 3) as i16;
            output.fill(dc_out);
            return;
        }

        let mut intermediate = [0i32; 16];

        // Column pass
        for i in 0..4 {
            let in0 = input[i] as i32;
            let in1 = input[i + 4] as i32;
            let in2 = input[i + 8] as i32;
            let in3 = input[i + 12] as i32;

            let a1 = in0 + in2;
            let b1 = in0 - in2;
            let temp1 = (in1 * 2217) >> 11;
            let temp2 = in3 + ((in3 * 1258) >> 11);
            let c1 = temp1 - temp2;
            let temp3 = in1 + ((in1 * 1258) >> 11);
            let temp4 = (in3 * 2217) >> 11;
            let d1 = temp3 + temp4;

            intermediate[i] = a1 + d1;
            intermediate[i + 4] = b1 + c1;
            intermediate[i + 8] = b1 - c1;
            intermediate[i + 12] = a1 - d1;
        }

        // Row pass
        for i in 0..4 {
            let row = i * 4;
            let in0 = intermediate[row];
            let in1 = intermediate[row + 1];
            let in2 = intermediate[row + 2];
            let in3 = intermediate[row + 3];

            let a1 = in0 + in2;
            let b1 = in0 - in2;
            let temp1 = (in1 * 2217) >> 11;
            let temp2 = in3 + ((in3 * 1258) >> 11);
            let c1 = temp1 - temp2;
            let temp3 = in1 + ((in1 * 1258) >> 11);
            let temp4 = (in3 * 2217) >> 11;
            let d1 = temp3 + temp4;

            output[row] = ((a1 + d1 + 4) >> 3) as i16;
            output[row + 1] = ((b1 + c1 + 4) >> 3) as i16;
            output[row + 2] = ((b1 - c1 + 4) >> 3) as i16;
            output[row + 3] = ((a1 - d1 + 4) >> 3) as i16;
        }
    }
}

/// 4x4 Forward Walsh-Hadamard Transform (WHT) for Luma DC coefficients (RFC 6386 Section 14.1).
#[inline(always)]
pub fn forward_wht_4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let in0_16 = vld1q_s16(input.as_ptr());
        let in1_16 = vld1q_s16(input.as_ptr().add(8));
        let zero_16 = vdupq_n_s16(0);
        let eq0 = vceqq_s16(in0_16, zero_16);
        let eq1 = vceqq_s16(in1_16, zero_16);
        let and = vandq_u16(eq0, eq1);
        if vminvq_u16(and) == 0xFFFF {
            output.fill(0);
            return;
        }

        // Pass 1: Row WHT
        let row0 = vmovl_s16(vget_low_s16(in0_16));
        let row1 = vmovl_s16(vget_high_s16(in0_16));
        let row2 = vmovl_s16(vget_low_s16(in1_16));
        let row3 = vmovl_s16(vget_high_s16(in1_16));

        let trn0 = vzip1q_s32(row0, row2);
        let trn1 = vzip2q_s32(row0, row2);
        let trn2 = vzip1q_s32(row1, row3);
        let trn3 = vzip2q_s32(row1, row3);

        let v0 = vzip1q_s32(trn0, trn2);
        let v1 = vzip2q_s32(trn0, trn2);
        let v2 = vzip1q_s32(trn1, trn3);
        let v3 = vzip2q_s32(trn1, trn3);

        let a1 = vaddq_s32(v0, v3);
        let b1 = vaddq_s32(v1, v2);
        let c1 = vsubq_s32(v1, v2);
        let d1 = vsubq_s32(v0, v3);

        let t0 = vaddq_s32(a1, b1);
        let t1 = vaddq_s32(c1, d1);
        let t2 = vsubq_s32(a1, b1);
        let t3 = vsubq_s32(d1, c1);

        // Pass 2: Column WHT
        let col_trn0 = vzip1q_s32(t0, t2);
        let col_trn1 = vzip2q_s32(t0, t2);
        let col_trn2 = vzip1q_s32(t1, t3);
        let col_trn3 = vzip2q_s32(t1, t3);

        let u0 = vzip1q_s32(col_trn0, col_trn2);
        let u1 = vzip2q_s32(col_trn0, col_trn2);
        let u2 = vzip1q_s32(col_trn1, col_trn3);
        let u3 = vzip2q_s32(col_trn1, col_trn3);

        let col_a1 = vaddq_s32(u0, u3);
        let col_b1 = vaddq_s32(u1, u2);
        let col_c1 = vsubq_s32(u1, u2);
        let col_d1 = vsubq_s32(u0, u3);

        let k1 = vdupq_n_s32(1);
        let o0 = vshrq_n_s32(vaddq_s32(vaddq_s32(col_a1, col_b1), k1), 1);
        let o1 = vshrq_n_s32(vaddq_s32(vaddq_s32(col_c1, col_d1), k1), 1);
        let o2 = vshrq_n_s32(vaddq_s32(vsubq_s32(col_a1, col_b1), k1), 1);
        let o3 = vshrq_n_s32(vaddq_s32(vsubq_s32(col_d1, col_c1), k1), 1);

        let out01_16 = vcombine_s16(vmovn_s32(o0), vmovn_s32(o1));
        let out23_16 = vcombine_s16(vmovn_s32(o2), vmovn_s32(o3));

        vst1q_s16(output.as_mut_ptr(), out01_16);
        vst1q_s16(output.as_mut_ptr().add(8), out23_16);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        let in_lo = _mm_loadu_si128(input.as_ptr() as *const __m128i);
        let in_hi = _mm_loadu_si128(input.as_ptr().add(8) as *const __m128i);
        let zero = _mm_setzero_si128();
        let eq_lo = _mm_cmpeq_epi16(in_lo, zero);
        let eq_hi = _mm_cmpeq_epi16(in_hi, zero);
        if _mm_movemask_epi8(eq_lo) == 0xFFFF && _mm_movemask_epi8(eq_hi) == 0xFFFF {
            output.fill(0);
            return;
        }

        // Pass 1: Row WHT (32-bit unpacked)
        let row0 = _mm_srai_epi32(_mm_unpacklo_epi16(zero, in_lo), 16);
        let row1 = _mm_srai_epi32(_mm_unpackhi_epi16(zero, in_lo), 16);
        let row2 = _mm_srai_epi32(_mm_unpacklo_epi16(zero, in_hi), 16);
        let row3 = _mm_srai_epi32(_mm_unpackhi_epi16(zero, in_hi), 16);

        let t_lo0 = _mm_unpacklo_epi32(row0, row1);
        let t_hi0 = _mm_unpackhi_epi32(row0, row1);
        let t_lo1 = _mm_unpacklo_epi32(row2, row3);
        let t_hi1 = _mm_unpackhi_epi32(row2, row3);

        let v0 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(t_lo0), _mm_castsi128_ps(t_lo1)));
        let v1 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(t_lo1), _mm_castsi128_ps(t_lo0)));
        let v2 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(t_hi0), _mm_castsi128_ps(t_hi1)));
        let v3 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(t_hi1), _mm_castsi128_ps(t_hi0)));

        let a1 = _mm_add_epi32(v0, v3);
        let b1 = _mm_add_epi32(v1, v2);
        let c1 = _mm_sub_epi32(v1, v2);
        let d1 = _mm_sub_epi32(v0, v3);

        let t0 = _mm_add_epi32(a1, b1);
        let t1 = _mm_add_epi32(c1, d1);
        let t2 = _mm_sub_epi32(a1, b1);
        let t3 = _mm_sub_epi32(d1, c1);

        // Pass 2: Column WHT
        let ct_lo0 = _mm_unpacklo_epi32(t0, t1);
        let ct_hi0 = _mm_unpackhi_epi32(t0, t1);
        let ct_lo1 = _mm_unpacklo_epi32(t2, t3);
        let ct_hi1 = _mm_unpackhi_epi32(t2, t3);

        let u0 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(ct_lo0), _mm_castsi128_ps(ct_lo1)));
        let u1 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(ct_lo1), _mm_castsi128_ps(ct_lo0)));
        let u2 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(ct_hi0), _mm_castsi128_ps(ct_hi1)));
        let u3 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(ct_hi1), _mm_castsi128_ps(ct_hi0)));

        let col_a1 = _mm_add_epi32(u0, u3);
        let col_b1 = _mm_add_epi32(u1, u2);
        let col_c1 = _mm_sub_epi32(u1, u2);
        let col_d1 = _mm_sub_epi32(u0, u3);

        let k1 = _mm_set1_epi32(1);
        let o0 = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(col_a1, col_b1), k1), 1);
        let o1 = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(col_c1, col_d1), k1), 1);
        let o2 = _mm_srai_epi32(_mm_add_epi32(_mm_sub_epi32(col_a1, col_b1), k1), 1);
        let o3 = _mm_srai_epi32(_mm_add_epi32(_mm_sub_epi32(col_d1, col_c1), k1), 1);

        let out01_16 = _mm_packs_epi32(o0, o1);
        let out23_16 = _mm_packs_epi32(o2, o3);

        _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, out01_16);
        _mm_storeu_si128(output.as_mut_ptr().add(8) as *mut __m128i, out23_16);
        return;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut intermediate = [0i32; 16];
        for i in 0..4 {
            let row = i * 4;
            let a1 = (input[row] + input[row + 3]) as i32;
            let b1 = (input[row + 1] + input[row + 2]) as i32;
            let c1 = (input[row + 1] - input[row + 2]) as i32;
            let d1 = (input[row] - input[row + 3]) as i32;
            intermediate[row] = a1 + b1;
            intermediate[row + 1] = c1 + d1;
            intermediate[row + 2] = a1 - b1;
            intermediate[row + 3] = d1 - c1;
        }
        for i in 0..4 {
            let a1 = intermediate[i] + intermediate[i + 12];
            let b1 = intermediate[i + 4] + intermediate[i + 8];
            let c1 = intermediate[i + 4] - intermediate[i + 8];
            let d1 = intermediate[i] - intermediate[i + 12];
            output[i] = ((a1 + b1 + 1) >> 1) as i16;
            output[i + 4] = ((c1 + d1 + 1) >> 1) as i16;
            output[i + 8] = ((a1 - b1 + 1) >> 1) as i16;
            output[i + 12] = ((d1 - c1 + 1) >> 1) as i16;
        }
    }
}

/// 4x4 Inverse Walsh-Hadamard Transform (WHT) for Luma DC reconstruction (RFC 6386 Section 14.2).
#[inline(always)]
pub fn inverse_wht_4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let in0_16 = vld1q_s16(input.as_ptr());
        let in1_16 = vld1q_s16(input.as_ptr().add(8));
        let zero_16 = vdupq_n_s16(0);
        let eq0 = vceqq_s16(in0_16, zero_16);
        let eq1 = vceqq_s16(in1_16, zero_16);
        let and = vandq_u16(eq0, eq1);
        if vminvq_u16(and) == 0xFFFF {
            output.fill(0);
            return;
        }
        let mask_ac = vsetq_lane_u16(0xFFFF, eq0, 0);
        if vminvq_u16(eq1) == 0xFFFF && vminvq_u16(mask_ac) == 0xFFFF && input[1] == 0 {
            let dc_out = ((input[0] as i32 + 3) >> 3) as i16;
            let vec_dc = vdupq_n_s16(dc_out);
            vst1q_s16(output.as_mut_ptr(), vec_dc);
            vst1q_s16(output.as_mut_ptr().add(8), vec_dc);
            return;
        }

        // Pass 1: Row IWHT
        let row0 = vmovl_s16(vget_low_s16(in0_16));
        let row1 = vmovl_s16(vget_high_s16(in0_16));
        let row2 = vmovl_s16(vget_low_s16(in1_16));
        let row3 = vmovl_s16(vget_high_s16(in1_16));

        let trn0 = vzip1q_s32(row0, row2);
        let trn1 = vzip2q_s32(row0, row2);
        let trn2 = vzip1q_s32(row1, row3);
        let trn3 = vzip2q_s32(row1, row3);

        let v0 = vzip1q_s32(trn0, trn2);
        let v1 = vzip2q_s32(trn0, trn2);
        let v2 = vzip1q_s32(trn1, trn3);
        let v3 = vzip2q_s32(trn1, trn3);

        let a1 = vaddq_s32(v0, v3);
        let b1 = vaddq_s32(v1, v2);
        let c1 = vsubq_s32(v1, v2);
        let d1 = vsubq_s32(v0, v3);

        let t0 = vaddq_s32(a1, b1);
        let t1 = vaddq_s32(c1, d1);
        let t2 = vsubq_s32(a1, b1);
        let t3 = vsubq_s32(d1, c1);

        // Pass 2: Column IWHT
        let col_trn0 = vzip1q_s32(t0, t2);
        let col_trn1 = vzip2q_s32(t0, t2);
        let col_trn2 = vzip1q_s32(t1, t3);
        let col_trn3 = vzip2q_s32(t1, t3);

        let u0 = vzip1q_s32(col_trn0, col_trn2);
        let u1 = vzip2q_s32(col_trn0, col_trn2);
        let u2 = vzip1q_s32(col_trn1, col_trn3);
        let u3 = vzip2q_s32(col_trn1, col_trn3);

        let col_a1 = vaddq_s32(u0, u3);
        let col_b1 = vaddq_s32(u1, u2);
        let col_c1 = vsubq_s32(u1, u2);
        let col_d1 = vsubq_s32(u0, u3);

        let k3 = vdupq_n_s32(3);
        let o0 = vshrq_n_s32(vaddq_s32(vaddq_s32(col_a1, col_b1), k3), 3);
        let o1 = vshrq_n_s32(vaddq_s32(vaddq_s32(col_c1, col_d1), k3), 3);
        let o2 = vshrq_n_s32(vaddq_s32(vsubq_s32(col_a1, col_b1), k3), 3);
        let o3 = vshrq_n_s32(vaddq_s32(vsubq_s32(col_d1, col_c1), k3), 3);

        let out01_16 = vcombine_s16(vmovn_s32(o0), vmovn_s32(o1));
        let out23_16 = vcombine_s16(vmovn_s32(o2), vmovn_s32(o3));

        vst1q_s16(output.as_mut_ptr(), out01_16);
        vst1q_s16(output.as_mut_ptr().add(8), out23_16);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        let in_lo = _mm_loadu_si128(input.as_ptr() as *const __m128i);
        let in_hi = _mm_loadu_si128(input.as_ptr().add(8) as *const __m128i);
        let zero = _mm_setzero_si128();
        let eq_lo = _mm_cmpeq_epi16(in_lo, zero);
        let eq_hi = _mm_cmpeq_epi16(in_hi, zero);
        if _mm_movemask_epi8(eq_lo) == 0xFFFF && _mm_movemask_epi8(eq_hi) == 0xFFFF {
            output.fill(0);
            return;
        }
        // DC only check
        if _mm_movemask_epi8(eq_hi) == 0xFFFF && (_mm_movemask_epi8(eq_lo) & 0xFFFC) == 0xFFFC && input[1] == 0 {
            let dc_out = ((input[0] as i32 + 3) >> 3) as i16;
            let vec_dc = _mm_set1_epi16(dc_out);
            _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, vec_dc);
            _mm_storeu_si128(output.as_mut_ptr().add(8) as *mut __m128i, vec_dc);
            return;
        }

        // Pass 1: Row IWHT (32-bit unpacked)
        let row0 = _mm_srai_epi32(_mm_unpacklo_epi16(zero, in_lo), 16);
        let row1 = _mm_srai_epi32(_mm_unpackhi_epi16(zero, in_lo), 16);
        let row2 = _mm_srai_epi32(_mm_unpacklo_epi16(zero, in_hi), 16);
        let row3 = _mm_srai_epi32(_mm_unpackhi_epi16(zero, in_hi), 16);

        let t_lo0 = _mm_unpacklo_epi32(row0, row1);
        let t_hi0 = _mm_unpackhi_epi32(row0, row1);
        let t_lo1 = _mm_unpacklo_epi32(row2, row3);
        let t_hi1 = _mm_unpackhi_epi32(row2, row3);

        let v0 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(t_lo0), _mm_castsi128_ps(t_lo1)));
        let v1 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(t_lo1), _mm_castsi128_ps(t_lo0)));
        let v2 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(t_hi0), _mm_castsi128_ps(t_hi1)));
        let v3 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(t_hi1), _mm_castsi128_ps(t_hi0)));

        let a1 = _mm_add_epi32(v0, v3);
        let b1 = _mm_add_epi32(v1, v2);
        let c1 = _mm_sub_epi32(v1, v2);
        let d1 = _mm_sub_epi32(v0, v3);

        let t0 = _mm_add_epi32(a1, b1);
        let t1 = _mm_add_epi32(c1, d1);
        let t2 = _mm_sub_epi32(a1, b1);
        let t3 = _mm_sub_epi32(d1, c1);

        // Pass 2: Column IWHT
        let ct_lo0 = _mm_unpacklo_epi32(t0, t1);
        let ct_hi0 = _mm_unpackhi_epi32(t0, t1);
        let ct_lo1 = _mm_unpacklo_epi32(t2, t3);
        let ct_hi1 = _mm_unpackhi_epi32(t2, t3);

        let u0 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(ct_lo0), _mm_castsi128_ps(ct_lo1)));
        let u1 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(ct_lo1), _mm_castsi128_ps(ct_lo0)));
        let u2 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(ct_hi0), _mm_castsi128_ps(ct_hi1)));
        let u3 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(ct_hi1), _mm_castsi128_ps(ct_hi0)));

        let col_a1 = _mm_add_epi32(u0, u3);
        let col_b1 = _mm_add_epi32(u1, u2);
        let col_c1 = _mm_sub_epi32(u1, u2);
        let col_d1 = _mm_sub_epi32(u0, u3);

        let k3 = _mm_set1_epi32(3);
        let o0 = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(col_a1, col_b1), k3), 3);
        let o1 = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(col_c1, col_d1), k3), 3);
        let o2 = _mm_srai_epi32(_mm_add_epi32(_mm_sub_epi32(col_a1, col_b1), k3), 3);
        let o3 = _mm_srai_epi32(_mm_add_epi32(_mm_sub_epi32(col_d1, col_c1), k3), 3);

        let out01_16 = _mm_packs_epi32(o0, o1);
        let out23_16 = _mm_packs_epi32(o2, o3);

        _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, out01_16);
        _mm_storeu_si128(output.as_mut_ptr().add(8) as *mut __m128i, out23_16);
        return;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut intermediate = [0i32; 16];
        for i in 0..4 {
            let row = i * 4;
            let a1 = (input[row] + input[row + 3]) as i32;
            let b1 = (input[row + 1] + input[row + 2]) as i32;
            let c1 = (input[row + 1] - input[row + 2]) as i32;
            let d1 = (input[row] - input[row + 3]) as i32;
            intermediate[row] = a1 + b1;
            intermediate[row + 1] = c1 + d1;
            intermediate[row + 2] = a1 - b1;
            intermediate[row + 3] = d1 - c1;
        }
        for i in 0..4 {
            let a1 = intermediate[i] + intermediate[i + 12];
            let b1 = intermediate[i + 4] + intermediate[i + 8];
            let c1 = intermediate[i + 4] - intermediate[i + 8];
            let d1 = intermediate[i] - intermediate[i + 12];
            output[i] = ((a1 + b1 + 3) >> 3) as i16;
            output[i + 4] = ((c1 + d1 + 3) >> 3) as i16;
            output[i + 8] = ((a1 - b1 + 3) >> 3) as i16;
            output[i + 12] = ((d1 - c1 + 3) >> 3) as i16;
        }
    }
}

