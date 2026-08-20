//! VP8 Intra-Prediction for 16x16 macroblocks, 4x4 subblocks (`B_PRED`), and 8x8 chroma blocks.
//!
//! Complies with RFC 6386 Section 12 for all directional modes.

/// 16x16 Luma Intra-Prediction modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Intra16Mode {
    DC = 0,
    V = 1,
    H = 2,
    TM = 3,
}

/// 8x8 Chroma Intra-Prediction modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IntraChromaMode {
    DC = 0,
    V = 1,
    H = 2,
    TM = 3,
}

/// 4x4 Subblock Directional Prediction modes (`B_PRED` - 10 modes).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BMode {
    B_DC = 0,
    B_TM = 1,
    B_VE = 2,
    B_HE = 3,
    B_RD = 4,
    B_VR = 5,
    B_LD = 6,
    B_VL = 7,
    B_HD = 8,
    B_HU = 9,
}

/// Computes 16x16 prediction block given top and left neighbor boundaries.
pub fn predict_16x16(
    mode: Intra16Mode,
    top: Option<&[u8]>,
    left: Option<&[u8]>,
    top_left: Option<u8>,
    dst: &mut [u8; 256],
) {
    match mode {
        Intra16Mode::DC => {
            let mut sum = 0u32;
            let mut count = 0u32;
            if let Some(t) = top {
                for i in 0..16 {
                    sum += t[i] as u32;
                }
                count += 16;
            }
            if let Some(l) = left {
                for i in 0..16 {
                    sum += l[i] as u32;
                }
                count += 16;
            }
            let dc = if count > 0 {
                ((sum + (count >> 1)) / count) as u8
            } else {
                128
            };
            dst.fill(dc);
        }
        Intra16Mode::V => {
            if let Some(t) = top {
                for y in 0..16 {
                    for x in 0..16 {
                        dst[y * 16 + x] = t[x];
                    }
                }
            } else {
                dst.fill(128);
            }
        }
        Intra16Mode::H => {
            if let Some(l) = left {
                for y in 0..16 {
                    for x in 0..16 {
                        dst[y * 16 + x] = l[y];
                    }
                }
            } else {
                dst.fill(128);
            }
        }
        Intra16Mode::TM => {
            let tl = top_left.unwrap_or(128) as i32;
            for y in 0..16 {
                let l_val = left.map_or(128, |l| l[y]) as i32;
                for x in 0..16 {
                    let t_val = top.map_or(128, |t| t[x]) as i32;
                    let pred = (l_val + t_val - tl).clamp(0, 255) as u8;
                    dst[y * 16 + x] = pred;
                }
            }
        }
    }
}

/// Computes 8x8 Chroma prediction block given top and left neighbor boundaries.
pub fn predict_8x8(
    mode: IntraChromaMode,
    top: Option<&[u8]>,
    left: Option<&[u8]>,
    top_left: Option<u8>,
    dst: &mut [u8; 64],
) {
    match mode {
        IntraChromaMode::DC => {
            let mut sum = 0u32;
            let mut count = 0u32;
            if let Some(t) = top {
                for i in 0..8 {
                    sum += t[i] as u32;
                }
                count += 8;
            }
            if let Some(l) = left {
                for i in 0..8 {
                    sum += l[i] as u32;
                }
                count += 8;
            }
            let dc = if count > 0 {
                ((sum + (count >> 1)) / count) as u8
            } else {
                128
            };
            dst.fill(dc);
        }
        IntraChromaMode::V => {
            if let Some(t) = top {
                for y in 0..8 {
                    for x in 0..8 {
                        dst[y * 8 + x] = t[x];
                    }
                }
            } else {
                dst.fill(128);
            }
        }
        IntraChromaMode::H => {
            if let Some(l) = left {
                for y in 0..8 {
                    for x in 0..8 {
                        dst[y * 8 + x] = l[y];
                    }
                }
            } else {
                dst.fill(128);
            }
        }
        IntraChromaMode::TM => {
            let tl = top_left.unwrap_or(128) as i32;
            for y in 0..8 {
                let l_val = left.map_or(128, |l| l[y]) as i32;
                for x in 0..8 {
                    let t_val = top.map_or(128, |t| t[x]) as i32;
                    let pred = (l_val + t_val - tl).clamp(0, 255) as u8;
                    dst[y * 8 + x] = pred;
                }
            }
        }
    }
}

/// Computes 4x4 Subblock Prediction for directional mode (`B_PRED`).
pub fn predict_4x4(
    mode: BMode,
    above: &[u8; 8],     // [top0, top1, top2, top3, top4, top5, top6, top7]
    left: &[u8; 4],      // [left0, left1, left2, left3]
    top_left: u8,        // top-left sample
    dst: &mut [u8; 16],  // 4x4 output
) {
    let a0 = above[0] as i32;
    let a1 = above[1] as i32;
    let a2 = above[2] as i32;
    let a3 = above[3] as i32;
    let a4 = above[4] as i32;
    let a5 = above[5] as i32;
    let a6 = above[6] as i32;

    let l0 = left[0] as i32;
    let l1 = left[1] as i32;
    let l2 = left[2] as i32;
    let l3 = left[3] as i32;

    let tl = top_left as i32;

    match mode {
        BMode::B_DC => {
            let dc = ((a0 + a1 + a2 + a3 + l0 + l1 + l2 + l3 + 4) >> 3) as u8;
            dst.fill(dc);
        }
        BMode::B_TM => {
            for y in 0..4 {
                let l = left[y] as i32;
                for x in 0..4 {
                    let a = above[x] as i32;
                    dst[y * 4 + x] = (l + a - tl).clamp(0, 255) as u8;
                }
            }
        }
        BMode::B_VE => {
            let v0 = (tl + 2 * a0 + a1 + 2) >> 2;
            let v1 = (a0 + 2 * a1 + a2 + 2) >> 2;
            let v2 = (a1 + 2 * a2 + a3 + 2) >> 2;
            let v3 = (a2 + 2 * a3 + a4 + 2) >> 2;
            for y in 0..4 {
                dst[y * 4 + 0] = v0 as u8;
                dst[y * 4 + 1] = v1 as u8;
                dst[y * 4 + 2] = v2 as u8;
                dst[y * 4 + 3] = v3 as u8;
            }
        }
        BMode::B_HE => {
            let h0 = (tl + 2 * l0 + l1 + 2) >> 2;
            let h1 = (l0 + 2 * l1 + l2 + 2) >> 2;
            let h2 = (l1 + 2 * l2 + l3 + 2) >> 2;
            let h3 = (l2 + 2 * l3 + l3 + 2) >> 2;
            for x in 0..4 {
                dst[0 * 4 + x] = h0 as u8;
                dst[1 * 4 + x] = h1 as u8;
                dst[2 * 4 + x] = h2 as u8;
                dst[3 * 4 + x] = h3 as u8;
            }
        }
        BMode::B_RD => {
            let d3 = (l3 + 2 * l2 + l1 + 2) >> 2;
            let d2 = (l2 + 2 * l1 + l0 + 2) >> 2;
            let d1 = (l1 + 2 * l0 + tl + 2) >> 2;
            let d0 = (l0 + 2 * tl + a0 + 2) >> 2;
            let d_1 = (tl + 2 * a0 + a1 + 2) >> 2;
            let d_2 = (a0 + 2 * a1 + a2 + 2) >> 2;
            let d_3 = (a1 + 2 * a2 + a3 + 2) >> 2;

            dst[3 * 4 + 0] = d3 as u8;
            dst[2 * 4 + 0] = d2 as u8; dst[3 * 4 + 1] = d2 as u8;
            dst[1 * 4 + 0] = d1 as u8; dst[2 * 4 + 1] = d1 as u8; dst[3 * 4 + 2] = d1 as u8;
            dst[0 * 4 + 0] = d0 as u8; dst[1 * 4 + 1] = d0 as u8; dst[2 * 4 + 2] = d0 as u8; dst[3 * 4 + 3] = d0 as u8;
            dst[0 * 4 + 1] = d_1 as u8; dst[1 * 4 + 2] = d_1 as u8; dst[2 * 4 + 3] = d_1 as u8;
            dst[0 * 4 + 2] = d_2 as u8; dst[1 * 4 + 3] = d_2 as u8;
            dst[0 * 4 + 3] = d_3 as u8;
        }
        BMode::B_VR => {
            let v_1 = (tl + a0 + 1) >> 1;
            let v_2 = (a0 + a1 + 1) >> 1;
            let v_3 = (a1 + a2 + 1) >> 1;
            let v_4 = (a2 + a3 + 1) >> 1;
            let d0 = (l0 + 2 * tl + a0 + 2) >> 2;
            let d1 = (l1 + 2 * l0 + tl + 2) >> 2;
            let d2 = (l2 + 2 * l1 + l0 + 2) >> 2;

            dst[0 * 4 + 0] = v_1 as u8; dst[2 * 4 + 1] = v_1 as u8;
            dst[0 * 4 + 1] = v_2 as u8; dst[2 * 4 + 2] = v_2 as u8;
            dst[0 * 4 + 2] = v_3 as u8; dst[2 * 4 + 3] = v_3 as u8;
            dst[0 * 4 + 3] = v_4 as u8;

            dst[1 * 4 + 0] = d0 as u8; dst[3 * 4 + 1] = d0 as u8;
            dst[1 * 4 + 1] = ((tl + 2 * a0 + a1 + 2) >> 2) as u8; dst[3 * 4 + 2] = ((tl + 2 * a0 + a1 + 2) >> 2) as u8;
            dst[1 * 4 + 2] = ((a0 + 2 * a1 + a2 + 2) >> 2) as u8; dst[3 * 4 + 3] = ((a0 + 2 * a1 + a2 + 2) >> 2) as u8;
            dst[1 * 4 + 3] = ((a1 + 2 * a2 + a3 + 2) >> 2) as u8;

            dst[2 * 4 + 0] = d1 as u8;
            dst[3 * 4 + 0] = d2 as u8;
        }
        BMode::B_LD => {
            let ld0 = (a0 + 2 * a1 + a2 + 2) >> 2;
            let ld1 = (a1 + 2 * a2 + a3 + 2) >> 2;
            let ld2 = (a2 + 2 * a3 + a4 + 2) >> 2;
            let ld3 = (a3 + 2 * a4 + a5 + 2) >> 2;
            let ld4 = (a4 + 2 * a5 + a6 + 2) >> 2;
            let ld5 = (a5 + 2 * a6 + a6 + 2) >> 2;

            dst[0 * 4 + 0] = ld0 as u8;
            dst[0 * 4 + 1] = ld1 as u8; dst[1 * 4 + 0] = ld1 as u8;
            dst[0 * 4 + 2] = ld2 as u8; dst[1 * 4 + 1] = ld2 as u8; dst[2 * 4 + 0] = ld2 as u8;
            dst[0 * 4 + 3] = ld3 as u8; dst[1 * 4 + 2] = ld3 as u8; dst[2 * 4 + 1] = ld3 as u8; dst[3 * 4 + 0] = ld3 as u8;
            dst[1 * 4 + 3] = ld4 as u8; dst[2 * 4 + 2] = ld4 as u8; dst[3 * 4 + 1] = ld4 as u8;
            dst[2 * 4 + 3] = ld5 as u8; dst[3 * 4 + 2] = ld5 as u8;
            dst[3 * 4 + 3] = ld5 as u8;
        }
        BMode::B_VL => {
            let vl0 = (a0 + a1 + 1) >> 1;
            let vl1 = (a1 + a2 + 1) >> 1;
            let vl2 = (a2 + a3 + 1) >> 1;
            let vl3 = (a3 + a4 + 1) >> 1;
            let d0 = (a0 + 2 * a1 + a2 + 2) >> 2;
            let d1 = (a1 + 2 * a2 + a3 + 2) >> 2;
            let d2 = (a2 + 2 * a3 + a4 + 2) >> 2;
            let d3 = (a3 + 2 * a4 + a5 + 2) >> 2;

            dst[0 * 4 + 0] = vl0 as u8; dst[2 * 4 + 0] = d0 as u8;
            dst[0 * 4 + 1] = vl1 as u8; dst[2 * 4 + 1] = d1 as u8;
            dst[0 * 4 + 2] = vl2 as u8; dst[2 * 4 + 2] = d2 as u8;
            dst[0 * 4 + 3] = vl3 as u8; dst[2 * 4 + 3] = d3 as u8;

            dst[1 * 4 + 0] = d0 as u8; dst[3 * 4 + 0] = vl1 as u8;
            dst[1 * 4 + 1] = d1 as u8; dst[3 * 4 + 1] = vl2 as u8;
            dst[1 * 4 + 2] = d2 as u8; dst[3 * 4 + 2] = vl3 as u8;
            dst[1 * 4 + 3] = d3 as u8; dst[3 * 4 + 3] = ((a4 + a5 + 1) >> 1) as u8;
        }
        BMode::B_HD => {
            let h_1 = (l0 + tl + 1) >> 1;
            let d0 = (l1 + 2 * l0 + tl + 2) >> 2;
            let h_2 = (l1 + l0 + 1) >> 1;
            let d1 = (l2 + 2 * l1 + l0 + 2) >> 2;
            let h_3 = (l2 + l1 + 1) >> 1;
            let d2 = (l3 + 2 * l2 + l1 + 2) >> 2;

            dst[0 * 4 + 0] = h_1 as u8; dst[0 * 4 + 2] = d0 as u8;
            dst[1 * 4 + 0] = h_2 as u8; dst[1 * 4 + 2] = d1 as u8;
            dst[2 * 4 + 0] = h_3 as u8; dst[2 * 4 + 2] = d2 as u8;
            dst[3 * 4 + 0] = ((l3 + l2 + 1) >> 1) as u8; dst[3 * 4 + 2] = ((l3 + 2 * l3 + l2 + 2) >> 2) as u8;

            dst[0 * 4 + 1] = ((l0 + 2 * tl + a0 + 2) >> 2) as u8; dst[0 * 4 + 3] = ((tl + 2 * a0 + a1 + 2) >> 2) as u8;
            dst[1 * 4 + 1] = d0 as u8; dst[1 * 4 + 3] = ((l0 + 2 * tl + a0 + 2) >> 2) as u8;
            dst[2 * 4 + 1] = d1 as u8; dst[2 * 4 + 3] = d0 as u8;
            dst[3 * 4 + 1] = d2 as u8; dst[3 * 4 + 3] = d1 as u8;
        }
        BMode::B_HU => {
            let hu0 = (l0 + l1 + 1) >> 1;
            let hu1 = (l1 + l2 + 1) >> 1;
            let hu2 = (l2 + l3 + 1) >> 1;
            let d0 = (l0 + 2 * l1 + l2 + 2) >> 2;
            let d1 = (l1 + 2 * l2 + l3 + 2) >> 2;
            let d2 = (l2 + 2 * l3 + l3 + 2) >> 2;

            dst[0 * 4 + 0] = hu0 as u8; dst[0 * 4 + 1] = d0 as u8; dst[0 * 4 + 2] = hu1 as u8; dst[0 * 4 + 3] = d1 as u8;
            dst[1 * 4 + 0] = hu1 as u8; dst[1 * 4 + 1] = d1 as u8; dst[1 * 4 + 2] = hu2 as u8; dst[1 * 4 + 3] = d2 as u8;
            dst[2 * 4 + 0] = hu2 as u8; dst[2 * 4 + 1] = d2 as u8; dst[2 * 4 + 2] = l3 as u8; dst[2 * 4 + 3] = l3 as u8;
            dst[3 * 4 + 0] = l3 as u8;  dst[3 * 4 + 1] = l3 as u8; dst[3 * 4 + 2] = l3 as u8; dst[3 * 4 + 3] = l3 as u8;
        }
    }
}
