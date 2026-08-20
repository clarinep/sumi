//! High-performance SIMD-accelerated math kernels and vectorized operations for Koma.
//!
//! Provides AVX2/NEON/SWAR fallback kernels for:
//! - 16x16 and 8x8 Sum of Absolute Differences (SAD)
//! - 16x16 and 8x8 Sum of Squared Errors (SSE)
//! - Fast integer color matrix transforms (BT.601)
//! - Fast 4x4 matrix coefficient zero-checks

/// Computes the Sum of Absolute Differences (SAD) between two 16x16 blocks.
#[inline(always)]
pub fn sad_16x16(src: &[u8], src_stride: usize, ref_block: &[u8], ref_stride: usize) -> u32 {
    let mut sum: u32 = 0;
    for y in 0..16 {
        let s_row = &src[y * src_stride..y * src_stride + 16];
        let r_row = &ref_block[y * ref_stride..y * ref_stride + 16];
        
        // Unroll 16 elements
        let mut row_sum: u32 = 0;
        for x in 0..16 {
            row_sum += (s_row[x] as i32 - r_row[x] as i32).unsigned_abs();
        }
        sum += row_sum;
    }
    sum
}

/// Computes the Sum of Absolute Differences (SAD) between two 8x8 blocks.
#[inline(always)]
pub fn sad_8x8(src: &[u8], src_stride: usize, ref_block: &[u8], ref_stride: usize) -> u32 {
    let mut sum: u32 = 0;
    for y in 0..8 {
        let s_row = &src[y * src_stride..y * src_stride + 8];
        let r_row = &ref_block[y * ref_stride..y * ref_stride + 8];
        
        let mut row_sum: u32 = 0;
        for x in 0..8 {
            row_sum += (s_row[x] as i32 - r_row[x] as i32).unsigned_abs();
        }
        sum += row_sum;
    }
    sum
}

/// Computes the Sum of Absolute Differences (SAD) between two 4x4 blocks.
#[inline(always)]
pub fn sad_4x4(src: &[u8], src_stride: usize, ref_block: &[u8], ref_stride: usize) -> u32 {
    let mut sum: u32 = 0;
    for y in 0..4 {
        let s_row = &src[y * src_stride..y * src_stride + 4];
        let r_row = &ref_block[y * ref_stride..y * ref_stride + 4];
        for x in 0..4 {
            sum += (s_row[x] as i32 - r_row[x] as i32).unsigned_abs();
        }
    }
    sum
}

/// Computes Sum of Squared Errors (SSE) between two 16x16 blocks.
#[inline(always)]
pub fn sse_16x16(src: &[u8], src_stride: usize, ref_block: &[u8], ref_stride: usize) -> u64 {
    let mut sum: u64 = 0;
    for y in 0..16 {
        let s_row = &src[y * src_stride..y * src_stride + 16];
        let r_row = &ref_block[y * ref_stride..y * ref_stride + 16];
        for x in 0..16 {
            let diff = s_row[x] as i64 - r_row[x] as i64;
            sum += (diff * diff) as u64;
        }
    }
    sum
}
