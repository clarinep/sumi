//! Discrete Cosine Transform (FDCT/IDCT) and Walsh-Hadamard Transform (WHT) kernels.
//!
//! Provides hardware-accelerated (SSE2/SSSE3/AVX2) and branchless integer fixed-point 4x4
//! transform arithmetic defined in RFC 6386 Sections 14.1–14.4.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// 4x4 Forward Discrete Cosine Transform (FDCT) according to RFC 6386 Section 14.3.
///
/// Transforms 16 residual difference values into 16 frequency-domain DCT coefficients.
/// Returns `true` if any non-zero input was detected.
#[inline(always)]
pub fn fdct_4x4(input: &[i16; 16], output: &mut [i16; 16]) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        // Check if all inputs are zero using SSE2 (1 cycle latency PMOVMSKB / PCMPEQW)
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
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let any = (input[0] | input[1] | input[2] | input[3]
            | input[4] | input[5] | input[6] | input[7]
            | input[8] | input[9] | input[10] | input[11]
            | input[12] | input[13] | input[14] | input[15]) != 0;
        if !any {
            output.fill(0);
            return false;
        }
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

/// 4x4 Inverse Discrete Cosine Transform (IDCT) according to RFC 6386 Section 14.4.
#[inline(always)]
pub fn idct_4x4(input: &[i16; 16], output: &mut [i16; 16]) {
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
    }

    #[cfg(not(target_arch = "x86_64"))]
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

/// 4x4 Forward Walsh-Hadamard Transform (WHT) for Luma DC coefficients (RFC 6386 Section 14.1).
#[inline(always)]
pub fn forward_wht_4x4(input: &[i16; 16], output: &mut [i16; 16]) {
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
    }

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

/// 4x4 Inverse Walsh-Hadamard Transform (WHT) for Luma DC reconstruction (RFC 6386 Section 14.2).
#[inline(always)]
pub fn inverse_wht_4x4(input: &[i16; 16], output: &mut [i16; 16]) {
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
    }

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

