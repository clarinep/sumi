//! VP8 In-Loop Deblocking Filter (RFC 6386 Section 15).

/// Applies standard in-loop normal filter to a macroblock boundary.
#[inline(always)]
fn filter_edge(p1: u8, p0: u8, q0: u8, q1: u8, limit: i32, thresh: i32) -> (u8, u8) {
    let diff = (q0 as i32) - (p0 as i32);
    if diff.abs() >= limit {
        return (p0, q0);
    }

    let a = (p1 as i32 - p0 as i32).abs();
    let b = (q1 as i32 - q0 as i32).abs();
    if a > thresh || b > thresh {
        return (p0, q0);
    }

    let delta = ((diff * 3 + (p1 as i32 - q1 as i32) + 4) >> 3).clamp(-128, 127);
    let new_p0 = (p0 as i32 + delta).clamp(0, 255) as u8;
    let new_q0 = (q0 as i32 - delta).clamp(0, 255) as u8;

    (new_p0, new_q0)
}

/// In-loop vertical edge filter for 16x16 luma boundary.
pub fn filter_vertical_edge_16(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    limit: i32,
    thresh: i32,
) {
    for row in 0..16 {
        let idx = (y + row) * stride + x;
        let p1 = plane[idx - 2];
        let p0 = plane[idx - 1];
        let q0 = plane[idx];
        let q1 = plane[idx + 1];

        let (new_p0, new_q0) = filter_edge(p1, p0, q0, q1, limit, thresh);
        plane[idx - 1] = new_p0;
        plane[idx] = new_q0;
    }
}

/// In-loop horizontal edge filter for 16x16 luma boundary.
pub fn filter_horizontal_edge_16(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    limit: i32,
    thresh: i32,
) {
    for col in 0..16 {
        let idx = y * stride + (x + col);
        let p1 = plane[idx - 2 * stride];
        let p0 = plane[idx - stride];
        let q0 = plane[idx];
        let q1 = plane[idx + stride];

        let (new_p0, new_q0) = filter_edge(p1, p0, q0, q1, limit, thresh);
        plane[idx - stride] = new_p0;
        plane[idx] = new_q0;
    }
}
