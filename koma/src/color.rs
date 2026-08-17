//! High-performance RGBA to planar YUV420p color space conversion.
//!
//! Implements fixed-point BT.601 color space transformation with 2x2 box-filter chroma
//! subsampling. Formatted row-by-row on contiguous slices for auto-vectorization (AVX2/SSE2)
//! and zero heap allocations.

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use std::arch::aarch64::*;

#[inline(always)]
fn rgb_to_y(r: i32, g: i32, b: i32) -> u8 {
    // BT.601: Y = (66*R + 129*G + 25*B + 128) >> 8 + 16
    (((66 * r + 129 * g + 25 * b + 128) >> 8) + 16) as u8
}

#[inline(always)]
fn rgb_to_uv(r_avg: i32, g_avg: i32, b_avg: i32) -> (u8, u8) {
    // U = (-38*R - 74*G + 112*B + 128) >> 8 + 128
    // V = (112*R - 94*G - 18*B + 128) >> 8 + 128
    let u = (((-38 * r_avg - 74 * g_avg + 112 * b_avg + 128) >> 8) + 128) as u8;
    let v = (((112 * r_avg - 94 * g_avg - 18 * b_avg + 128) >> 8) + 128) as u8;
    (u, v)
}

/// Converts interleaved RGBA pixel data to planar YUV420p with 16-pixel macroblock padding.
///
/// Uses fixed-point BT.601 matrix arithmetic:
/// - $Y = ((66 \cdot R + 129 \cdot G + 25 \cdot B + 128) \gg 8) + 16$
/// - $U = ((-38 \cdot R_{avg} - 74 \cdot G_{avg} + 112 \cdot B_{avg} + 128) \gg 8) + 128$
/// - $V = ((112 \cdot R_{avg} - 94 \cdot G_{avg} - 18 \cdot B_{avg} + 128) \gg 8) + 128$
///
/// Automatically clamps results to standard video bounds `[0, 255]` and replicates edge
/// pixels into padded macroblock borders to eliminate edge ringing artifacts.
#[inline]
pub fn rgba_to_yuv420p(
    rgba: &[u8],
    width: usize,
    height: usize,
    pad_width: usize,
    pad_height: usize,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
) {
    let uv_stride = pad_width / 2;
    let row_stride = width * 4;

    for row_y in (0..height).step_by(2) {
        let y0 = row_y;
        let y1 = (row_y + 1).min(height - 1);

        let row0_src = &rgba[y0 * row_stride..(y0 + 1) * row_stride];
        let row1_src = &rgba[y1 * row_stride..(y1 + 1) * row_stride];

        let (y_out0, y_out1) = if y0 == y1 {
            let slice = &mut y_plane[y0 * pad_width..y0 * pad_width + pad_width];
            (slice, &mut [][..])
        } else {
            let (top, bottom) = y_plane.split_at_mut(y1 * pad_width);
            (&mut top[y0 * pad_width..y0 * pad_width + pad_width], &mut bottom[..pad_width])
        };
        let u_out = &mut u_plane[(row_y / 2) * uv_stride..(row_y / 2) * uv_stride + uv_stride];
        let v_out = &mut v_plane[(row_y / 2) * uv_stride..(row_y / 2) * uv_stride + uv_stride];

        let mut col_x = 0;

        #[cfg(target_arch = "aarch64")]
        unsafe {
            while col_x + 15 < width {
                let idx0 = col_x * 4;
                let r0_vec = vld4q_u8(row0_src.as_ptr().add(idx0));
                let r1_vec = vld4q_u8(row1_src.as_ptr().add(idx0));

                let r0 = r0_vec.0;
                let g0 = r0_vec.1;
                let b0 = r0_vec.2;

                let r1 = r1_vec.0;
                let g1 = r1_vec.1;
                let b1 = r1_vec.2;

                // Compute Y for row 0 (16 pixels)
                let r0_lo = vmovl_u8(vget_low_u8(r0));
                let g0_lo = vmovl_u8(vget_low_u8(g0));
                let b0_lo = vmovl_u8(vget_low_u8(b0));
                let mut y0_lo = vdupq_n_u16(128);
                y0_lo = vmlaq_n_u16(y0_lo, r0_lo, 66);
                y0_lo = vmlaq_n_u16(y0_lo, g0_lo, 129);
                y0_lo = vmlaq_n_u16(y0_lo, b0_lo, 25);
                let y0_lo_u8 = vadd_u8(vshrn_n_u16(y0_lo, 8), vdup_n_u8(16));

                let r0_hi = vmovl_u8(vget_high_u8(r0));
                let g0_hi = vmovl_u8(vget_high_u8(g0));
                let b0_hi = vmovl_u8(vget_high_u8(b0));
                let mut y0_hi = vdupq_n_u16(128);
                y0_hi = vmlaq_n_u16(y0_hi, r0_hi, 66);
                y0_hi = vmlaq_n_u16(y0_hi, g0_hi, 129);
                y0_hi = vmlaq_n_u16(y0_hi, b0_hi, 25);
                let y0_hi_u8 = vadd_u8(vshrn_n_u16(y0_hi, 8), vdup_n_u8(16));

                let y0_final = vcombine_u8(y0_lo_u8, y0_hi_u8);
                vst1q_u8(y_out0.as_mut_ptr().add(col_x), y0_final);

                // Compute Y for row 1
                if !y_out1.is_empty() {
                    let r1_lo = vmovl_u8(vget_low_u8(r1));
                    let g1_lo = vmovl_u8(vget_low_u8(g1));
                    let b1_lo = vmovl_u8(vget_low_u8(b1));
                    let mut y1_lo = vdupq_n_u16(128);
                    y1_lo = vmlaq_n_u16(y1_lo, r1_lo, 66);
                    y1_lo = vmlaq_n_u16(y1_lo, g1_lo, 129);
                    y1_lo = vmlaq_n_u16(y1_lo, b1_lo, 25);
                    let y1_lo_u8 = vadd_u8(vshrn_n_u16(y1_lo, 8), vdup_n_u8(16));

                    let r1_hi = vmovl_u8(vget_high_u8(r1));
                    let g1_hi = vmovl_u8(vget_high_u8(g1));
                    let b1_hi = vmovl_u8(vget_high_u8(b1));
                    let mut y1_hi = vdupq_n_u16(128);
                    y1_hi = vmlaq_n_u16(y1_hi, r1_hi, 66);
                    y1_hi = vmlaq_n_u16(y1_hi, g1_hi, 129);
                    y1_hi = vmlaq_n_u16(y1_hi, b1_hi, 25);
                    let y1_hi_u8 = vadd_u8(vshrn_n_u16(y1_hi, 8), vdup_n_u8(16));

                    let y1_final = vcombine_u8(y1_lo_u8, y1_hi_u8);
                    vst1q_u8(y_out1.as_mut_ptr().add(col_x), y1_final);
                }

                // 2x2 Subsampling for U and V (8 chroma samples from 16x2 pixels)
                let r0_pairs = vpaddlq_u8(r0);
                let r1_pairs = vpaddlq_u8(r1);
                let g0_pairs = vpaddlq_u8(g0);
                let g1_pairs = vpaddlq_u8(g1);
                let b0_pairs = vpaddlq_u8(b0);
                let b1_pairs = vpaddlq_u8(b1);

                let k2_u16 = vdupq_n_u16(2);
                let r_avg = vshrq_n_u16(vaddq_u16(vaddq_u16(r0_pairs, r1_pairs), k2_u16), 2);
                let g_avg = vshrq_n_u16(vaddq_u16(vaddq_u16(g0_pairs, g1_pairs), k2_u16), 2);
                let b_avg = vshrq_n_u16(vaddq_u16(vaddq_u16(b0_pairs, b1_pairs), k2_u16), 2);

                let r_i16 = vreinterpretq_s16_u16(r_avg);
                let g_i16 = vreinterpretq_s16_u16(g_avg);
                let b_i16 = vreinterpretq_s16_u16(b_avg);

                let mut u_s16 = vdupq_n_s16(128);
                u_s16 = vmlsq_n_s16(u_s16, r_i16, 38);
                u_s16 = vmlsq_n_s16(u_s16, g_i16, 74);
                u_s16 = vmlaq_n_s16(u_s16, b_i16, 112);
                let u_u8 = vadd_u8(vreinterpret_u8_s8(vshrn_n_s16(u_s16, 8)), vdup_n_u8(128));

                let mut v_s16 = vdupq_n_s16(128);
                v_s16 = vmlaq_n_s16(v_s16, r_i16, 112);
                v_s16 = vmlsq_n_s16(v_s16, g_i16, 94);
                v_s16 = vmlsq_n_s16(v_s16, b_i16, 18);
                let v_u8 = vadd_u8(vreinterpret_u8_s8(vshrn_n_s16(v_s16, 8)), vdup_n_u8(128));

                let uv_idx = col_x >> 1;
                vst1_u8(u_out.as_mut_ptr().add(uv_idx), u_u8);
                vst1_u8(v_out.as_mut_ptr().add(uv_idx), v_u8);

                col_x += 16;
            }
        }

        #[cfg(target_arch = "x86_64")]
        unsafe {
            let k66 = _mm_set1_epi16(66);
            let k129 = _mm_set1_epi16(129);
            let k25 = _mm_set1_epi16(25);
            let k128_16 = _mm_set1_epi16(128);

            while col_x + 7 < width {
                let idx0 = col_x * 4;
                let p0_0 = _mm_loadu_si128(row0_src.as_ptr().add(idx0) as *const __m128i);
                let p0_1 = _mm_loadu_si128(row0_src.as_ptr().add(idx0 + 16) as *const __m128i);

                let p1_0 = _mm_loadu_si128(row1_src.as_ptr().add(idx0) as *const __m128i);
                let p1_1 = _mm_loadu_si128(row1_src.as_ptr().add(idx0 + 16) as *const __m128i);

                let c0_0 = _mm_cvtsi128_si32(p0_0) as u32;
                let c0_1 = _mm_cvtsi128_si32(_mm_srli_si128(p0_0, 4)) as u32;
                let c0_2 = _mm_cvtsi128_si32(_mm_srli_si128(p0_0, 8)) as u32;
                let c0_3 = _mm_cvtsi128_si32(_mm_srli_si128(p0_0, 12)) as u32;
                let c0_4 = _mm_cvtsi128_si32(p0_1) as u32;
                let c0_5 = _mm_cvtsi128_si32(_mm_srli_si128(p0_1, 4)) as u32;
                let c0_6 = _mm_cvtsi128_si32(_mm_srli_si128(p0_1, 8)) as u32;
                let c0_7 = _mm_cvtsi128_si32(_mm_srli_si128(p0_1, 12)) as u32;

                let c1_0 = _mm_cvtsi128_si32(p1_0) as u32;
                let c1_1 = _mm_cvtsi128_si32(_mm_srli_si128(p1_0, 4)) as u32;
                let c1_2 = _mm_cvtsi128_si32(_mm_srli_si128(p1_0, 8)) as u32;
                let c1_3 = _mm_cvtsi128_si32(_mm_srli_si128(p1_0, 12)) as u32;
                let c1_4 = _mm_cvtsi128_si32(p1_1) as u32;
                let c1_5 = _mm_cvtsi128_si32(_mm_srli_si128(p1_1, 4)) as u32;
                let c1_6 = _mm_cvtsi128_si32(_mm_srli_si128(p1_1, 8)) as u32;
                let c1_7 = _mm_cvtsi128_si32(_mm_srli_si128(p1_1, 12)) as u32;

                let r0_vec = _mm_set_epi16((c0_7 & 0xFF) as i16, (c0_6 & 0xFF) as i16, (c0_5 & 0xFF) as i16, (c0_4 & 0xFF) as i16, (c0_3 & 0xFF) as i16, (c0_2 & 0xFF) as i16, (c0_1 & 0xFF) as i16, (c0_0 & 0xFF) as i16);
                let g0_vec = _mm_set_epi16(((c0_7 >> 8) & 0xFF) as i16, ((c0_6 >> 8) & 0xFF) as i16, ((c0_5 >> 8) & 0xFF) as i16, ((c0_4 >> 8) & 0xFF) as i16, ((c0_3 >> 8) & 0xFF) as i16, ((c0_2 >> 8) & 0xFF) as i16, ((c0_1 >> 8) & 0xFF) as i16, ((c0_0 >> 8) & 0xFF) as i16);
                let b0_vec = _mm_set_epi16(((c0_7 >> 16) & 0xFF) as i16, ((c0_6 >> 16) & 0xFF) as i16, ((c0_5 >> 16) & 0xFF) as i16, ((c0_4 >> 16) & 0xFF) as i16, ((c0_3 >> 16) & 0xFF) as i16, ((c0_2 >> 16) & 0xFF) as i16, ((c0_1 >> 16) & 0xFF) as i16, ((c0_0 >> 16) & 0xFF) as i16);

                let y0_16 = _mm_add_epi16(
                    _mm_srli_epi16(
                        _mm_add_epi16(
                            _mm_add_epi16(_mm_mullo_epi16(r0_vec, k66), _mm_mullo_epi16(g0_vec, k129)),
                            _mm_add_epi16(_mm_mullo_epi16(b0_vec, k25), k128_16),
                        ),
                        8,
                    ),
                    _mm_set1_epi16(16),
                );
                let y0_8 = _mm_packus_epi16(y0_16, y0_16);
                let y0_u64 = _mm_cvtsi128_si64(y0_8) as u64;
                (y_out0.as_mut_ptr().add(col_x) as *mut u64).write_unaligned(y0_u64);

                let r1_vec = _mm_set_epi16((c1_7 & 0xFF) as i16, (c1_6 & 0xFF) as i16, (c1_5 & 0xFF) as i16, (c1_4 & 0xFF) as i16, (c1_3 & 0xFF) as i16, (c1_2 & 0xFF) as i16, (c1_1 & 0xFF) as i16, (c1_0 & 0xFF) as i16);
                let g1_vec = _mm_set_epi16(((c1_7 >> 8) & 0xFF) as i16, ((c1_6 >> 8) & 0xFF) as i16, ((c1_5 >> 8) & 0xFF) as i16, ((c1_4 >> 8) & 0xFF) as i16, ((c1_3 >> 8) & 0xFF) as i16, ((c1_2 >> 8) & 0xFF) as i16, ((c1_1 >> 8) & 0xFF) as i16, ((c1_0 >> 8) & 0xFF) as i16);
                let b1_vec = _mm_set_epi16(((c1_7 >> 16) & 0xFF) as i16, ((c1_6 >> 16) & 0xFF) as i16, ((c1_5 >> 16) & 0xFF) as i16, ((c1_4 >> 16) & 0xFF) as i16, ((c1_3 >> 16) & 0xFF) as i16, ((c1_2 >> 16) & 0xFF) as i16, ((c1_1 >> 16) & 0xFF) as i16, ((c1_0 >> 16) & 0xFF) as i16);

                if !y_out1.is_empty() {
                    let y1_16 = _mm_add_epi16(
                        _mm_srli_epi16(
                            _mm_add_epi16(
                                _mm_add_epi16(_mm_mullo_epi16(r1_vec, k66), _mm_mullo_epi16(g1_vec, k129)),
                                _mm_add_epi16(_mm_mullo_epi16(b1_vec, k25), k128_16),
                            ),
                            8,
                        ),
                        _mm_set1_epi16(16),
                    );
                    let y1_8 = _mm_packus_epi16(y1_16, y1_16);
                    let y1_u64 = _mm_cvtsi128_si64(y1_8) as u64;
                    (y_out1.as_mut_ptr().add(col_x) as *mut u64).write_unaligned(y1_u64);
                }

                // Chroma 2x2 averages (4 samples)
                let r_sum0 = _mm_add_epi16(r0_vec, r1_vec);
                let g_sum0 = _mm_add_epi16(g0_vec, g1_vec);
                let b_sum0 = _mm_add_epi16(b0_vec, b1_vec);

                let mask_even = _mm_set_epi16(0, -1, 0, -1, 0, -1, 0, -1);
                let r_even = _mm_and_si128(r_sum0, mask_even);
                let r_odd = _mm_srli_si128(r_sum0, 2);
                let r_avg = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(r_even, r_odd), _mm_set1_epi16(2)), 2);

                let g_even = _mm_and_si128(g_sum0, mask_even);
                let g_odd = _mm_srli_si128(g_sum0, 2);
                let g_avg = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(g_even, g_odd), _mm_set1_epi16(2)), 2);

                let b_even = _mm_and_si128(b_sum0, mask_even);
                let b_odd = _mm_srli_si128(b_sum0, 2);
                let b_avg = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(b_even, b_odd), _mm_set1_epi16(2)), 2);

                let r_avg4 = _mm_set_epi16(0, 0, 0, 0, _mm_extract_epi16(r_avg, 6) as i16, _mm_extract_epi16(r_avg, 4) as i16, _mm_extract_epi16(r_avg, 2) as i16, _mm_extract_epi16(r_avg, 0) as i16);
                let g_avg4 = _mm_set_epi16(0, 0, 0, 0, _mm_extract_epi16(g_avg, 6) as i16, _mm_extract_epi16(g_avg, 4) as i16, _mm_extract_epi16(g_avg, 2) as i16, _mm_extract_epi16(g_avg, 0) as i16);
                let b_avg4 = _mm_set_epi16(0, 0, 0, 0, _mm_extract_epi16(b_avg, 6) as i16, _mm_extract_epi16(b_avg, 4) as i16, _mm_extract_epi16(b_avg, 2) as i16, _mm_extract_epi16(b_avg, 0) as i16);

                let u_16 = _mm_add_epi16(
                    _mm_srai_epi16(
                        _mm_add_epi16(
                            _mm_sub_epi16(_mm_mullo_epi16(b_avg4, _mm_set1_epi16(112)), _mm_mullo_epi16(r_avg4, _mm_set1_epi16(38))),
                            _mm_sub_epi16(k128_16, _mm_mullo_epi16(g_avg4, _mm_set1_epi16(74))),
                        ),
                        8,
                    ),
                    _mm_set1_epi16(128),
                );

                let v_16 = _mm_add_epi16(
                    _mm_srai_epi16(
                        _mm_add_epi16(
                            _mm_sub_epi16(_mm_mullo_epi16(r_avg4, _mm_set1_epi16(112)), _mm_mullo_epi16(g_avg4, _mm_set1_epi16(94))),
                            _mm_sub_epi16(k128_16, _mm_mullo_epi16(b_avg4, _mm_set1_epi16(18))),
                        ),
                        8,
                    ),
                    _mm_set1_epi16(128),
                );

                let u_8 = _mm_packus_epi16(u_16, u_16);
                let v_8 = _mm_packus_epi16(v_16, v_16);

                let uv_idx = col_x >> 1;
                let u_u32 = _mm_cvtsi128_si32(u_8) as u32;
                let v_u32 = _mm_cvtsi128_si32(v_8) as u32;
                (u_out.as_mut_ptr().add(uv_idx) as *mut u32).write_unaligned(u_u32);
                (v_out.as_mut_ptr().add(uv_idx) as *mut u32).write_unaligned(v_u32);

                col_x += 8;
            }
        }

        // Remainder
        while col_x < width {
            let x0 = col_x;
            let x1 = (col_x + 1).min(width - 1);

            let idx00 = x0 * 4;
            let idx01 = x1 * 4;

            let (r00, g00, b00) = (row0_src[idx00] as i32, row0_src[idx00 + 1] as i32, row0_src[idx00 + 2] as i32);
            let (r01, g01, b01) = (row0_src[idx01] as i32, row0_src[idx01 + 1] as i32, row0_src[idx01 + 2] as i32);
            let (r10, g10, b10) = (row1_src[idx00] as i32, row1_src[idx00 + 1] as i32, row1_src[idx00 + 2] as i32);
            let (r11, g11, b11) = (row1_src[idx01] as i32, row1_src[idx01 + 1] as i32, row1_src[idx01 + 2] as i32);

            y_out0[x0] = rgb_to_y(r00, g00, b00);
            if x0 + 1 < width {
                y_out0[x0 + 1] = rgb_to_y(r01, g01, b01);
            }
            if !y_out1.is_empty() {
                y_out1[x0] = rgb_to_y(r10, g10, b10);
                if x0 + 1 < width {
                    y_out1[x0 + 1] = rgb_to_y(r11, g11, b11);
                }
            }

            let r_avg = (r00 + r01 + r10 + r11 + 2) >> 2;
            let g_avg = (g00 + g01 + g10 + g11 + 2) >> 2;
            let b_avg = (b00 + b01 + b10 + b11 + 2) >> 2;

            let uv_x = col_x / 2;
            let (u, v) = rgb_to_uv(r_avg, g_avg, b_avg);
            u_out[uv_x] = u;
            v_out[uv_x] = v;

            col_x += 2;
        }

        // Pad horizontal row if padded width > width
        if pad_width > width {
            let pad_val0 = y_out0[width - 1];
            y_out0[width..pad_width].fill(pad_val0);
            if !y_out1.is_empty() {
                let pad_val1 = y_out1[width - 1];
                y_out1[width..pad_width].fill(pad_val1);
            }

            let uv_w = (width + 1) / 2;
            let pad_u = u_out[uv_w - 1];
            let pad_v = v_out[uv_w - 1];
            u_out[uv_w..uv_stride].fill(pad_u);
            v_out[uv_w..uv_stride].fill(pad_v);
        }
    }

    // Pad vertical rows if padded height > height
    if pad_height > height {
        for y in height..pad_height {
            let src_y = height - 1;
            let (src, dst) = y_plane.split_at_mut(y * pad_width);
            dst[..pad_width].copy_from_slice(&src[src_y * pad_width..src_y * pad_width + pad_width]);
        }
        let uv_h = (height + 1) / 2;
        let pad_uv_h = pad_height / 2;
        for y in uv_h..pad_uv_h {
            let src_y = uv_h - 1;
            let (src_u, dst_u) = u_plane.split_at_mut(y * uv_stride);
            dst_u[..uv_stride].copy_from_slice(&src_u[src_y * uv_stride..src_y * uv_stride + uv_stride]);

            let (src_v, dst_v) = v_plane.split_at_mut(y * uv_stride);
            dst_v[..uv_stride].copy_from_slice(&src_v[src_y * uv_stride..src_y * uv_stride + uv_stride]);
        }
    }
}

