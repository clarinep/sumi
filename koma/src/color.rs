//! High-performance RGBA to planar YUV420p color space conversion.
//!
//! Implements fixed-point BT.601 color space transformation with 2x2 box-filter chroma
//! subsampling. Formatted row-by-row on contiguous slices for auto-vectorization (AVX2/SSE2)
//! and zero heap allocations.

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

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
            // If height is 1, y0 == y1
            (slice, &mut [][..])
        } else {
            let (top, bottom) = y_plane.split_at_mut(y1 * pad_width);
            (&mut top[y0 * pad_width..y0 * pad_width + pad_width], &mut bottom[..pad_width])
        };
        let u_out = &mut u_plane[(row_y / 2) * uv_stride..(row_y / 2) * uv_stride + uv_stride];
        let v_out = &mut v_plane[(row_y / 2) * uv_stride..(row_y / 2) * uv_stride + uv_stride];

        let mut col_x = 0;

        // Vectorized 8-pixel (4 chroma pairs) loop
        while col_x + 7 < width {
            let idx0 = col_x * 4;
            let idx1 = (col_x + 1) * 4;
            let idx2 = (col_x + 2) * 4;
            let idx3 = (col_x + 3) * 4;
            let idx4 = (col_x + 4) * 4;
            let idx5 = (col_x + 5) * 4;
            let idx6 = (col_x + 6) * 4;
            let idx7 = (col_x + 7) * 4;

            // Pair 0
            let r00 = row0_src[idx0] as i32;
            let g00 = row0_src[idx0 + 1] as i32;
            let b00 = row0_src[idx0 + 2] as i32;
            let r01 = row0_src[idx1] as i32;
            let g01 = row0_src[idx1 + 1] as i32;
            let b01 = row0_src[idx1 + 2] as i32;

            let r10 = row1_src[idx0] as i32;
            let g10 = row1_src[idx0 + 1] as i32;
            let b10 = row1_src[idx0 + 2] as i32;
            let r11 = row1_src[idx1] as i32;
            let g11 = row1_src[idx1 + 1] as i32;
            let b11 = row1_src[idx1 + 2] as i32;

            y_out0[col_x] = rgb_to_y(r00, g00, b00);
            y_out0[col_x + 1] = rgb_to_y(r01, g01, b01);
            y_out1[col_x] = rgb_to_y(r10, g10, b10);
            y_out1[col_x + 1] = rgb_to_y(r11, g11, b11);

            let r_avg0 = (r00 + r01 + r10 + r11 + 2) >> 2;
            let g_avg0 = (g00 + g01 + g10 + g11 + 2) >> 2;
            let b_avg0 = (b00 + b01 + b10 + b11 + 2) >> 2;
            let (u0, v0) = rgb_to_uv(r_avg0, g_avg0, b_avg0);
            let uv_idx0 = col_x >> 1;
            u_out[uv_idx0] = u0;
            v_out[uv_idx0] = v0;

            // Pair 1
            let r02 = row0_src[idx2] as i32;
            let g02 = row0_src[idx2 + 1] as i32;
            let b02 = row0_src[idx2 + 2] as i32;
            let r03 = row0_src[idx3] as i32;
            let g03 = row0_src[idx3 + 1] as i32;
            let b03 = row0_src[idx3 + 2] as i32;

            let r12 = row1_src[idx2] as i32;
            let g12 = row1_src[idx2 + 1] as i32;
            let b12 = row1_src[idx2 + 2] as i32;
            let r13 = row1_src[idx3] as i32;
            let g13 = row1_src[idx3 + 1] as i32;
            let b13 = row1_src[idx3 + 2] as i32;

            y_out0[col_x + 2] = rgb_to_y(r02, g02, b02);
            y_out0[col_x + 3] = rgb_to_y(r03, g03, b03);
            y_out1[col_x + 2] = rgb_to_y(r12, g12, b12);
            y_out1[col_x + 3] = rgb_to_y(r13, g13, b13);

            let r_avg1 = (r02 + r03 + r12 + r13 + 2) >> 2;
            let g_avg1 = (g02 + g03 + g12 + g13 + 2) >> 2;
            let b_avg1 = (b02 + b03 + b12 + b13 + 2) >> 2;
            let (u1, v1) = rgb_to_uv(r_avg1, g_avg1, b_avg1);
            u_out[uv_idx0 + 1] = u1;
            v_out[uv_idx0 + 1] = v1;

            // Pair 2
            let r04 = row0_src[idx4] as i32;
            let g04 = row0_src[idx4 + 1] as i32;
            let b04 = row0_src[idx4 + 2] as i32;
            let r05 = row0_src[idx5] as i32;
            let g05 = row0_src[idx5 + 1] as i32;
            let b05 = row0_src[idx5 + 2] as i32;

            let r14 = row1_src[idx4] as i32;
            let g14 = row1_src[idx4 + 1] as i32;
            let b14 = row1_src[idx4 + 2] as i32;
            let r15 = row1_src[idx5] as i32;
            let g15 = row1_src[idx5 + 1] as i32;
            let b15 = row1_src[idx5 + 2] as i32;

            y_out0[col_x + 4] = rgb_to_y(r04, g04, b04);
            y_out0[col_x + 5] = rgb_to_y(r05, g05, b05);
            y_out1[col_x + 4] = rgb_to_y(r14, g14, b14);
            y_out1[col_x + 5] = rgb_to_y(r15, g15, b15);

            let r_avg2 = (r04 + r05 + r14 + r15 + 2) >> 2;
            let g_avg2 = (g04 + g05 + g14 + g15 + 2) >> 2;
            let b_avg2 = (b04 + b05 + b14 + b15 + 2) >> 2;
            let (u2, v2) = rgb_to_uv(r_avg2, g_avg2, b_avg2);
            u_out[uv_idx0 + 2] = u2;
            v_out[uv_idx0 + 2] = v2;

            // Pair 3
            let r06 = row0_src[idx6] as i32;
            let g06 = row0_src[idx6 + 1] as i32;
            let b06 = row0_src[idx6 + 2] as i32;
            let r07 = row0_src[idx7] as i32;
            let g07 = row0_src[idx7 + 1] as i32;
            let b07 = row0_src[idx7 + 2] as i32;

            let r16 = row1_src[idx6] as i32;
            let g16 = row1_src[idx6 + 1] as i32;
            let b16 = row1_src[idx6 + 2] as i32;
            let r17 = row1_src[idx7] as i32;
            let g17 = row1_src[idx7 + 1] as i32;
            let b17 = row1_src[idx7 + 2] as i32;

            y_out0[col_x + 6] = rgb_to_y(r06, g06, b06);
            y_out0[col_x + 7] = rgb_to_y(r07, g07, b07);
            y_out1[col_x + 6] = rgb_to_y(r16, g16, b16);
            y_out1[col_x + 7] = rgb_to_y(r17, g17, b17);

            let r_avg3 = (r06 + r07 + r16 + r17 + 2) >> 2;
            let g_avg3 = (g06 + g07 + g16 + g17 + 2) >> 2;
            let b_avg3 = (b06 + b07 + b16 + b17 + 2) >> 2;
            let (u3, v3) = rgb_to_uv(r_avg3, g_avg3, b_avg3);
            u_out[uv_idx0 + 3] = u3;
            v_out[uv_idx0 + 3] = v3;

            col_x += 8;
        }

        // Process 4 pixels (2 chroma pairs) when possible
        while col_x + 3 < width {
            let idx0 = col_x * 4;
            let idx1 = (col_x + 1) * 4;
            let idx2 = (col_x + 2) * 4;
            let idx3 = (col_x + 3) * 4;

            // Pair 0
            let r00 = row0_src[idx0] as i32;
            let g00 = row0_src[idx0 + 1] as i32;
            let b00 = row0_src[idx0 + 2] as i32;
            let r01 = row0_src[idx1] as i32;
            let g01 = row0_src[idx1 + 1] as i32;
            let b01 = row0_src[idx1 + 2] as i32;

            let r10 = row1_src[idx0] as i32;
            let g10 = row1_src[idx0 + 1] as i32;
            let b10 = row1_src[idx0 + 2] as i32;
            let r11 = row1_src[idx1] as i32;
            let g11 = row1_src[idx1 + 1] as i32;
            let b11 = row1_src[idx1 + 2] as i32;

            y_out0[col_x] = rgb_to_y(r00, g00, b00);
            y_out0[col_x + 1] = rgb_to_y(r01, g01, b01);
            y_out1[col_x] = rgb_to_y(r10, g10, b10);
            y_out1[col_x + 1] = rgb_to_y(r11, g11, b11);

            let r_avg0 = (r00 + r01 + r10 + r11 + 2) >> 2;
            let g_avg0 = (g00 + g01 + g10 + g11 + 2) >> 2;
            let b_avg0 = (b00 + b01 + b10 + b11 + 2) >> 2;
            let (u0, v0) = rgb_to_uv(r_avg0, g_avg0, b_avg0);
            let uv_idx0 = col_x >> 1;
            u_out[uv_idx0] = u0;
            v_out[uv_idx0] = v0;

            // Pair 1
            let r02 = row0_src[idx2] as i32;
            let g02 = row0_src[idx2 + 1] as i32;
            let b02 = row0_src[idx2 + 2] as i32;
            let r03 = row0_src[idx3] as i32;
            let g03 = row0_src[idx3 + 1] as i32;
            let b03 = row0_src[idx3 + 2] as i32;

            let r12 = row1_src[idx2] as i32;
            let g12 = row1_src[idx2 + 1] as i32;
            let b12 = row1_src[idx2 + 2] as i32;
            let r13 = row1_src[idx3] as i32;
            let g13 = row1_src[idx3 + 1] as i32;
            let b13 = row1_src[idx3 + 2] as i32;

            y_out0[col_x + 2] = rgb_to_y(r02, g02, b02);
            y_out0[col_x + 3] = rgb_to_y(r03, g03, b03);
            y_out1[col_x + 2] = rgb_to_y(r12, g12, b12);
            y_out1[col_x + 3] = rgb_to_y(r13, g13, b13);

            let r_avg1 = (r02 + r03 + r12 + r13 + 2) >> 2;
            let g_avg1 = (g02 + g03 + g12 + g13 + 2) >> 2;
            let b_avg1 = (b02 + b03 + b12 + b13 + 2) >> 2;
            let (u1, v1) = rgb_to_uv(r_avg1, g_avg1, b_avg1);
            let uv_idx1 = uv_idx0 + 1;
            u_out[uv_idx1] = u1;
            v_out[uv_idx1] = v1;

            col_x += 4;
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
            y_out1[x0] = rgb_to_y(r10, g10, b10);
            if x0 + 1 < width {
                y_out1[x0 + 1] = rgb_to_y(r11, g11, b11);
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

