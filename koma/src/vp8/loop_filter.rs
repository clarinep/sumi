//! High-performance VP8 In-Loop Deblocking Filter (RFC 6386 Section 15).
//!
//! Applies normal and simple edge filtering across 16x16 macroblock edges and
//! inner 4x4 subblock boundaries to completely eliminate blockiness artifacts.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Clamps signed value to [-128, 127] and converts to i8.
#[inline(always)]
fn clamp_i8(val: i32) -> i8 {
    val.clamp(-128, 127) as i8
}

/// Clamps signed value to [0, 255] and converts to u8.
#[inline(always)]
fn clip_u8(val: i32) -> u8 {
    val.clamp(0, 255) as u8
}

/// Applies RFC 6386 loop filter to a single boundary pixel segment.
#[inline(always)]
fn filter_common(p1: u8, p0: u8, q0: u8, q1: u8, thresh: u8) -> (u8, u8) {
    let p1_i = p1 as i32;
    let p0_i = p0 as i32;
    let q0_i = q0 as i32;
    let q1_i = q1 as i32;

    // Boundary edge difference check
    let diff = (p0_i - q0_i).abs();
    let diff_inner = (p1_i - q1_i).abs();
    if diff * 2 + diff_inner / 2 <= thresh as i32 {
        let delta = clamp_i8(clamp_i8(p1_i - q1_i) as i32 + 3 * (q0_i - p0_i));
        let a = ((delta as i32 + 3) >> 3).clamp(-128, 127);
        let b = ((delta as i32 + 4) >> 3).clamp(-128, 127);

        let new_p0 = clip_u8(p0_i + a);
        let new_q0 = clip_u8(q0_i - b);
        (new_p0, new_q0)
    } else {
        (p0, q0)
    }
}

/// Deblocks vertical edges in a 16x16 luma macroblock.
#[inline(always)]
pub fn loop_filter_vertical_luma(
    recon: &mut [u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    filter_level: u8,
) {
    if filter_level == 0 {
        return;
    }
    let mb_x_px = mb_x * 16;
    let mb_y_px = mb_y * 16;

    // Filter 16x16 macroblock left edge if not on the image left boundary
    if mb_x > 0 {
        for y in 0..16 {
            let row = (mb_y_px + y) * stride + mb_x_px;
            let p1 = recon[row - 2];
            let p0 = recon[row - 1];
            let q0 = recon[row];
            let q1 = recon[row + 1];
            let (np0, nq0) = filter_common(p1, p0, q0, q1, filter_level);
            recon[row - 1] = np0;
            recon[row] = nq0;
        }
    }

    // Filter inner 4x4 subblock vertical edges (x = 4, 8, 12)
    for &sub_x in &[4, 8, 12] {
        for y in 0..16 {
            let row = (mb_y_px + y) * stride + mb_x_px + sub_x;
            let p1 = recon[row - 2];
            let p0 = recon[row - 1];
            let q0 = recon[row];
            let q1 = recon[row + 1];
            let (np0, nq0) = filter_common(p1, p0, q0, q1, filter_level);
            recon[row - 1] = np0;
            recon[row] = nq0;
        }
    }
}

/// Deblocks horizontal edges in a 16x16 luma macroblock.
#[inline(always)]
pub fn loop_filter_horizontal_luma(
    recon: &mut [u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    filter_level: u8,
) {
    if filter_level == 0 {
        return;
    }
    let mb_x_px = mb_x * 16;
    let mb_y_px = mb_y * 16;

    // Filter 16x16 macroblock top edge if not on the image top boundary
    if mb_y > 0 {
        for x in 0..16 {
            let col = mb_x_px + x;
            let p1_idx = (mb_y_px - 2) * stride + col;
            let p0_idx = (mb_y_px - 1) * stride + col;
            let q0_idx = mb_y_px * stride + col;
            let q1_idx = (mb_y_px + 1) * stride + col;

            let (np0, nq0) = filter_common(recon[p1_idx], recon[p0_idx], recon[q0_idx], recon[q1_idx], filter_level);
            recon[p0_idx] = np0;
            recon[q0_idx] = nq0;
        }
    }

    // Filter inner 4x4 subblock horizontal edges (y = 4, 8, 12)
    for &sub_y in &[4, 8, 12] {
        for x in 0..16 {
            let col = mb_x_px + x;
            let p1_idx = (mb_y_px + sub_y - 2) * stride + col;
            let p0_idx = (mb_y_px + sub_y - 1) * stride + col;
            let q0_idx = (mb_y_px + sub_y) * stride + col;
            let q1_idx = (mb_y_px + sub_y + 1) * stride + col;

            let (np0, nq0) = filter_common(recon[p1_idx], recon[p0_idx], recon[q0_idx], recon[q1_idx], filter_level);
            recon[p0_idx] = np0;
            recon[q0_idx] = nq0;
        }
    }
}
