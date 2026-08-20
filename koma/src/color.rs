//! High-performance color space conversion and planar YUV420 conversion for Koma.
//!
//! Provides mathematically exact ITU-R BT.601 full-range RGB/RGBA to planar YUV420
//! conversion with edge replication for 16x16 macroblock alignment.

/// Planar YUV420 representation with padded dimensions for VP8 encoding.
#[derive(Debug, Clone)]
pub struct Yuv420Planar {
    /// Original image width in pixels.
    pub width: usize,
    /// Original image height in pixels.
    pub height: usize,
    /// Macroblock count horizontally (`(width + 15) / 16`).
    pub mb_cols: usize,
    /// Macroblock count vertically (`(height + 15) / 16`).
    pub mb_rows: usize,
    /// Padded luma width (`mb_cols * 16`).
    pub y_stride: usize,
    /// Padded luma height (`mb_rows * 16`).
    pub y_height: usize,
    /// Padded chroma width (`mb_cols * 8`).
    pub uv_stride: usize,
    /// Padded chroma height (`mb_rows * 8`).
    pub uv_height: usize,
    /// Y (luma) plane samples (`y_stride * y_height` bytes).
    pub y: Vec<u8>,
    /// U (chroma Cb) plane samples (`uv_stride * uv_height` bytes).
    pub u: Vec<u8>,
    /// V (chroma Cr) plane samples (`uv_stride * uv_height` bytes).
    pub v: Vec<u8>,
    /// Optional Alpha plane samples (`width * height` bytes).
    pub alpha: Option<Vec<u8>>,
    /// Whether any transparent pixels (alpha < 255) were detected.
    pub has_alpha: bool,
}

impl Yuv420Planar {
    /// Allocates an empty YUV420 planar buffer sized for the given dimensions.
    pub fn new(width: usize, height: usize, with_alpha: bool) -> Self {
        let mb_cols = (width + 15) / 16;
        let mb_rows = (height + 15) / 16;
        let y_stride = mb_cols * 16;
        let y_height = mb_rows * 16;
        let uv_stride = mb_cols * 8;
        let uv_height = mb_rows * 8;

        Self {
            width,
            height,
            mb_cols,
            mb_rows,
            y_stride,
            y_height,
            uv_stride,
            uv_height,
            y: vec![0u8; y_stride * y_height],
            u: vec![128u8; uv_stride * uv_height],
            v: vec![128u8; uv_stride * uv_height],
            alpha: if with_alpha {
                Some(vec![255u8; width * height])
            } else {
                None
            },
            has_alpha: false,
        }
    }
}

// Fixed-point BT.601 full-range matrix coefficients scaled by 2^16 (65536)
// Y  =  0.299000 * R + 0.587000 * G + 0.114000 * B
// U  = -0.168736 * R - 0.331264 * G + 0.500000 * B + 128
// V  =  0.500000 * R - 0.418688 * G - 0.081312 * B + 128
const COEFF_YR: i32 = 19595;
const COEFF_YG: i32 = 38470;
const COEFF_YB: i32 = 7471;

const COEFF_UR: i32 = -11059;
const COEFF_UG: i32 = -21709;
const COEFF_UB: i32 = 32768;

const COEFF_VR: i32 = 32768;
const COEFF_VG: i32 = -27439;
const COEFF_VB: i32 = -5329;

#[inline(always)]
fn rgb_to_y(r: i32, g: i32, b: i32) -> u8 {
    let y = (COEFF_YR * r + COEFF_YG * g + COEFF_YB * b + 32768) >> 16;
    y.clamp(0, 255) as u8
}

#[inline(always)]
fn rgb_to_u(r: i32, g: i32, b: i32) -> u8 {
    let u = ((COEFF_UR * r + COEFF_UG * g + COEFF_UB * b + 32768) >> 16) + 128;
    u.clamp(0, 255) as u8
}

#[inline(always)]
fn rgb_to_v(r: i32, g: i32, b: i32) -> u8 {
    let v = ((COEFF_VR * r + COEFF_VG * g + COEFF_VB * b + 32768) >> 16) + 128;
    v.clamp(0, 255) as u8
}

/// Converts raw RGBA (8-bit per channel, 32bpp) into planar [`Yuv420Planar`] with alpha.
pub fn rgba_to_yuv420(
    rgba: &[u8],
    width: usize,
    height: usize,
) -> Yuv420Planar {
    let mut planar = Yuv420Planar::new(width, height, true);
    let mut has_transparency = false;

    let y_plane = &mut planar.y;
    let u_plane = &mut planar.u;
    let v_plane = &mut planar.v;
    let alpha_plane = planar.alpha.as_mut().unwrap();

    let y_stride = planar.y_stride;
    let uv_stride = planar.uv_stride;

    // 1. Process 2x2 blocks for combined Luma, Chroma subsampling, and Alpha extraction
    for y_block in 0..planar.mb_rows * 8 {
        let row0 = y_block * 2;
        let row1 = row0 + 1;

        let valid_row0 = row0 < height;
        let valid_row1 = row1 < height;

        if !valid_row0 {
            break;
        }

        for x_block in 0..planar.mb_cols * 8 {
            let col0 = x_block * 2;
            let col1 = col0 + 1;

            let valid_col0 = col0 < width;
            let valid_col1 = col1 < width;

            // Fetch (col0, row0)
            let (r00, g00, b00, a00) = if valid_col0 {
                let idx = (row0 * width + col0) * 4;
                (
                    rgba[idx] as i32,
                    rgba[idx + 1] as i32,
                    rgba[idx + 2] as i32,
                    rgba[idx + 3],
                )
            } else {
                (0, 0, 0, 255)
            };

            // Fetch (col1, row0)
            let (r10, g10, b10, a10) = if valid_col1 {
                let idx = (row0 * width + col1) * 4;
                (
                    rgba[idx] as i32,
                    rgba[idx + 1] as i32,
                    rgba[idx + 2] as i32,
                    rgba[idx + 3],
                )
            } else {
                (r00, g00, b00, a00)
            };

            // Fetch (col0, row1)
            let (r01, g01, b01, a01) = if valid_row1 && valid_col0 {
                let idx = (row1 * width + col0) * 4;
                (
                    rgba[idx] as i32,
                    rgba[idx + 1] as i32,
                    rgba[idx + 2] as i32,
                    rgba[idx + 3],
                )
            } else {
                (r00, g00, b00, a00)
            };

            // Fetch (col1, row1)
            let (r11, g11, b11, a11) = if valid_row1 && valid_col1 {
                let idx = (row1 * width + col1) * 4;
                (
                    rgba[idx] as i32,
                    rgba[idx + 1] as i32,
                    rgba[idx + 2] as i32,
                    rgba[idx + 3],
                )
            } else {
                (r10, g10, b10, a10)
            };

            // Check alpha transparency
            if valid_col0 {
                alpha_plane[row0 * width + col0] = a00;
                if a00 < 255 {
                    has_transparency = true;
                }
            }
            if valid_col1 {
                alpha_plane[row0 * width + col1] = a10;
                if a10 < 255 {
                    has_transparency = true;
                }
            }
            if valid_row1 && valid_col0 {
                alpha_plane[row1 * width + col0] = a01;
                if a01 < 255 {
                    has_transparency = true;
                }
            }
            if valid_row1 && valid_col1 {
                alpha_plane[row1 * width + col1] = a11;
                if a11 < 255 {
                    has_transparency = true;
                }
            }

            // Calculate Y values
            let y00 = rgb_to_y(r00, g00, b00);
            let y10 = rgb_to_y(r10, g10, b10);
            let y01 = rgb_to_y(r01, g01, b01);
            let y11 = rgb_to_y(r11, g11, b11);

            y_plane[row0 * y_stride + col0] = y00;
            y_plane[row0 * y_stride + col1] = y10;
            if row1 < planar.y_height {
                y_plane[row1 * y_stride + col0] = y01;
                y_plane[row1 * y_stride + col1] = y11;
            }

            // 2x2 box filter average for Chroma U and V
            let r_avg = (r00 + r10 + r01 + r11 + 2) >> 2;
            let g_avg = (g00 + g10 + g01 + g11 + 2) >> 2;
            let b_avg = (b00 + b10 + b01 + b11 + 2) >> 2;

            let u_val = rgb_to_u(r_avg, g_avg, b_avg);
            let v_val = rgb_to_v(r_avg, g_avg, b_avg);

            u_plane[y_block * uv_stride + x_block] = u_val;
            v_plane[y_block * uv_stride + x_block] = v_val;
        }
    }

    // 2. Clamp/replicate right and bottom edges for macroblock boundaries
    pad_plane_borders(&mut planar.y, width, height, planar.y_stride, planar.y_height);
    pad_plane_borders(
        &mut planar.u,
        (width + 1) / 2,
        (height + 1) / 2,
        planar.uv_stride,
        planar.uv_height,
    );
    pad_plane_borders(
        &mut planar.v,
        (width + 1) / 2,
        (height + 1) / 2,
        planar.uv_stride,
        planar.uv_height,
    );

    planar.has_alpha = has_transparency;
    if !has_transparency {
        planar.alpha = None;
    }

    planar
}

/// Replicates boundary pixels outwards to fill macroblock padding regions.
fn pad_plane_borders(
    plane: &mut [u8],
    active_width: usize,
    active_height: usize,
    stride: usize,
    total_height: usize,
) {
    // Horizontal edge replication for each active row
    if active_width < stride {
        for y in 0..active_height {
            let row_start = y * stride;
            let last_val = plane[row_start + active_width - 1];
            for x in active_width..stride {
                plane[row_start + x] = last_val;
            }
        }
    }

    // Vertical edge replication for padding rows at the bottom
    if active_height < total_height {
        let last_active_row = (active_height - 1) * stride;
        for y in active_height..total_height {
            let target_row = y * stride;
            for x in 0..stride {
                plane[target_row + x] = plane[last_active_row + x];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_yuv_bounds() {
        // Pure White
        assert_eq!(rgb_to_y(255, 255, 255), 255);
        assert_eq!(rgb_to_u(255, 255, 255), 128);
        assert_eq!(rgb_to_v(255, 255, 255), 128);

        // Pure Black
        assert_eq!(rgb_to_y(0, 0, 0), 0);
        assert_eq!(rgb_to_u(0, 0, 0), 128);
        assert_eq!(rgb_to_v(0, 0, 0), 128);
    }
}
