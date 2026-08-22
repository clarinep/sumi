//! VP8 4x4 Forward & Inverse Discrete Cosine Transforms (FDCT/IDCT)
//! and 4x4 Walsh-Hadamard Transform (WHT).
//!
//! Standard bit-exact fixed-point integer transforms specified in RFC 6386 Section 14.

/// Standard 4x4 Forward Discrete Cosine Transform (FDCT).
/// Transforms 16 residual spatial differences in `src` (row-major) into 16 frequency coefficients in `dst`.
pub fn fdct_4x4(src: &[i16; 16], dst: &mut [i16; 16]) {
    let mut tmp = [0i32; 16];

    // Horizontal pass
    for i in 0..4 {
        let s0 = src[i * 4 + 0] as i32;
        let s1 = src[i * 4 + 1] as i32;
        let s2 = src[i * 4 + 2] as i32;
        let s3 = src[i * 4 + 3] as i32;

        let a1 = s0 + s3;
        let b1 = s1 + s2;
        let c1 = s1 - s2;
        let d1 = s0 - s3;

        let a2 = a1 + b1;
        let b2 = (c1 * 2217 + d1 * 5352 + 14500) >> 12;
        let c2 = a1 - b1;
        let d2 = (d1 * 2217 - c1 * 5352 + 7500) >> 12;

        tmp[0 * 4 + i] = (a2 * 8) + if i == 0 { 4 } else { 0 };
        tmp[1 * 4 + i] = b2;
        tmp[2 * 4 + i] = c2 * 8;
        tmp[3 * 4 + i] = d2;
    }

    // Vertical pass
    for i in 0..4 {
        let t0 = tmp[i * 4 + 0];
        let t1 = tmp[i * 4 + 1];
        let t2 = tmp[i * 4 + 2];
        let t3 = tmp[i * 4 + 3];

        let a1 = t0 + t3;
        let b1 = t1 + t2;
        let c1 = t1 - t2;
        let d1 = t0 - t3;

        let a2 = a1 + b1;
        let b2 = (c1 * 2217 + d1 * 5352 + 14500) >> 12;
        let c2 = a1 - b1;
        let d2 = (d1 * 2217 - c1 * 5352 + 7500) >> 12;

        dst[i * 4 + 0] = ((a2 + 7) >> 4) as i16;
        dst[i * 4 + 1] = ((b2 + 7) >> 4) as i16;
        dst[i * 4 + 2] = ((c2 + 7) >> 4) as i16;
        dst[i * 4 + 3] = ((d2 + 7) >> 4) as i16;
    }
}

/// Standard 4x4 Inverse Discrete Cosine Transform (IDCT) with reconstruction addition.
/// Adds reconstructed residuals directly into `dst` slice with given `stride`.
pub fn idct_add_4x4(coeffs: &[i16; 16], dst: &mut [u8], stride: usize) {
    let mut tmp = [0i32; 16];

    // Horizontal pass
    for i in 0..4 {
        let c0 = coeffs[i * 4 + 0] as i32;
        let c1 = coeffs[i * 4 + 1] as i32;
        let c2 = coeffs[i * 4 + 2] as i32;
        let c3 = coeffs[i * 4 + 3] as i32;

        let a1 = c0 + c2;
        let b1 = c0 - c2;
        let temp1 = (c1 * 2217) >> 12;
        let temp2 = (c3 * 5352) >> 12;
        let c1_trans = temp1 - temp2;

        let temp3 = (c1 * 5352) >> 12;
        let temp4 = (c3 * 2217) >> 12;
        let d1_trans = temp3 + temp4;

        tmp[i * 4 + 0] = a1 + d1_trans;
        tmp[i * 4 + 3] = a1 - d1_trans;
        tmp[i * 4 + 1] = b1 + c1_trans;
        tmp[i * 4 + 2] = b1 - c1_trans;
    }

    // Vertical pass
    for i in 0..4 {
        let t0 = tmp[0 * 4 + i];
        let t1 = tmp[1 * 4 + i];
        let t2 = tmp[2 * 4 + i];
        let t3 = tmp[3 * 4 + i];

        let a1 = t0 + t2;
        let b1 = t0 - t2;
        let temp1 = (t1 * 2217) >> 12;
        let temp2 = (t3 * 5352) >> 12;
        let c1_trans = temp1 - temp2;

        let temp3 = (t1 * 5352) >> 12;
        let temp4 = (t3 * 2217) >> 12;
        let d1_trans = temp3 + temp4;

        let r0 = (a1 + d1_trans + 4) >> 3;
        let r1 = (b1 + c1_trans + 4) >> 3;
        let r2 = (b1 - c1_trans + 4) >> 3;
        let r3 = (a1 - d1_trans + 4) >> 3;

        let row0 = &mut dst[0 * stride + i];
        *row0 = (*row0 as i32 + r0).clamp(0, 255) as u8;

        let row1 = &mut dst[1 * stride + i];
        *row1 = (*row1 as i32 + r1).clamp(0, 255) as u8;

        let row2 = &mut dst[2 * stride + i];
        *row2 = (*row2 as i32 + r2).clamp(0, 255) as u8;

        let row3 = &mut dst[3 * stride + i];
        *row3 = (*row3 as i32 + r3).clamp(0, 255) as u8;
    }
}

/// 4x4 Walsh-Hadamard Transform (WHT) for 16 Luma DC coefficients in 16x16 macroblocks.
/// Follows RFC 6386 Section 14.3 / libvpx vp8_short_walsh4x4.
pub fn wht_4x4(src: &[i16; 16], dst: &mut [i16; 16]) {
    let mut tmp = [0i32; 16];

    // Horizontal / row pass
    for i in 0..4 {
        let a1 = (src[i * 4 + 0] as i32) + (src[i * 4 + 3] as i32);
        let b1 = (src[i * 4 + 1] as i32) + (src[i * 4 + 2] as i32);
        let c1 = (src[i * 4 + 1] as i32) - (src[i * 4 + 2] as i32);
        let d1 = (src[i * 4 + 0] as i32) - (src[i * 4 + 3] as i32);

        let a2 = a1 + b1;
        let b2 = c1 + d1;
        let c2 = a1 - b1;
        let d2 = d1 - c1;

        tmp[0 * 4 + i] = a2 * 2;
        tmp[1 * 4 + i] = b2 * 2;
        tmp[2 * 4 + i] = c2 * 2;
        tmp[3 * 4 + i] = d2 * 2;
    }

    // Vertical / column pass
    for i in 0..4 {
        let a1 = tmp[i * 4 + 0] + tmp[i * 4 + 3];
        let b1 = tmp[i * 4 + 1] + tmp[i * 4 + 2];
        let c1 = tmp[i * 4 + 1] - tmp[i * 4 + 2];
        let d1 = tmp[i * 4 + 0] - tmp[i * 4 + 3];

        let a2 = a1 + b1;
        let b2 = c1 + d1;
        let c2 = a1 - b1;
        let d2 = d1 - c1;

        dst[i * 4 + 0] = ((a2 + 7) >> 3) as i16;
        dst[i * 4 + 1] = ((b2 + 7) >> 3) as i16;
        dst[i * 4 + 2] = ((c2 + 7) >> 3) as i16;
        dst[i * 4 + 3] = ((d2 + 7) >> 3) as i16;
    }
}

/// 4x4 Inverse Walsh-Hadamard Transform (IWHT).
/// Conforms bit-exactly to RFC 6386 Section 14.3 (vp8_short_inv_walsh4x4_c).
pub fn iwht_4x4(src: &[i16; 16], dst: &mut [i16; 16]) {
    let mut tmp = [0i32; 16];

    // Column pass
    for i in 0..4 {
        let a1 = (src[i + 0] as i32) + (src[i + 12] as i32);
        let b1 = (src[i + 4] as i32) + (src[i + 8] as i32);
        let c1 = (src[i + 4] as i32) - (src[i + 8] as i32);
        let d1 = (src[i + 0] as i32) - (src[i + 12] as i32);

        tmp[i + 0] = a1 + b1;
        tmp[i + 4] = c1 + d1;
        tmp[i + 8] = a1 - b1;
        tmp[i + 12] = d1 - c1;
    }

    // Row pass
    for i in 0..4 {
        let a1 = tmp[i * 4 + 0] + tmp[i * 4 + 3];
        let b1 = tmp[i * 4 + 1] + tmp[i * 4 + 2];
        let c1 = tmp[i * 4 + 1] - tmp[i * 4 + 2];
        let d1 = tmp[i * 4 + 0] - tmp[i * 4 + 3];

        let a2 = a1 + b1;
        let b2 = c1 + d1;
        let c2 = a1 - b1;
        let d2 = d1 - c1;

        dst[i * 4 + 0] = ((a2 + 3) >> 3) as i16;
        dst[i * 4 + 1] = ((b2 + 3) >> 3) as i16;
        dst[i * 4 + 2] = ((c2 + 3) >> 3) as i16;
        dst[i * 4 + 3] = ((d2 + 3) >> 3) as i16;
    }
}
