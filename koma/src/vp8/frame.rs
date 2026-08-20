//! VP8 Keyframe Macroblock Encoder & Rate-Distortion Mode Decision Engine.
//!
//! Handles 16x16 macroblock traversal, optimal Intra prediction mode selection (SAD/SSE),
//! forward transforms, quantization, and context-adaptive token serialization into Partition 0 & 1.

use super::bool_coder::BoolEncoder;
use super::intra_pred::{predict_16x16, predict_8x8, Intra16Mode, IntraChromaMode};
use super::loop_filter::{filter_horizontal_edge_16, filter_vertical_edge_16};
use super::prob_tables::{COEFF_BANDS, DEFAULT_COEFF_PROBS};
use super::quant::Quantizer;
use super::simd::{sad_16x16, sad_8x8};
use super::tables::{KF_Y_MODE_PROBS, KF_Y_MODE_TREE, UV_MODE_PROBS, UV_MODE_TREE, ZIGZAG};
use super::transform::{fdct_4x4, idct_add_4x4, iwht_4x4, wht_4x4};
use crate::color::Yuv420Planar;

/// Encodes a full YUV420 planar frame into an RFC 6386 VP8 keyframe bitstream.
pub fn encode_frame(planar: &Yuv420Planar, quant: &Quantizer) -> Vec<u8> {
    let mb_cols = planar.mb_cols;
    let mb_rows = planar.mb_rows;

    let mut part0 = BoolEncoder::with_capacity(4096);
    let mut part1 = BoolEncoder::with_capacity(32768);

    // Frame Header Flags in Partition 0:
    // Color space (0 = YUV)
    part0.put_bit(false);
    // Clamping type (0 = standard)
    part0.put_bit(false);
    // Segmentation disabled (0)
    part0.put_bit(false);
    // Simple/Normal loop filter type (0 = Normal)
    part0.put_bit(false);
    // Loop filter level (6 bits)
    let filter_level = (63 - (quant.qindex * 63 / 127)).clamp(0, 63) as u32;
    part0.put_uint(filter_level, 6);
    // Loop filter sharpness (3 bits)
    part0.put_uint(0, 3);
    // Loop filter delta enabled (0)
    part0.put_bit(false);
    // Number of DCT partitions: 0 (1 partition)
    part0.put_uint(0, 2);

    // Base Q index (7 bits)
    part0.put_uint(quant.qindex as u32, 7);
    // DC/AC delta Qs (0 = no deltas)
    part0.put_bit(false); // Y1 DC delta
    part0.put_bit(false); // Y2 DC delta
    part0.put_bit(false); // Y2 AC delta
    part0.put_bit(false); // UV DC delta
    part0.put_bit(false); // UV AC delta

    // Refresh golden & alt-ref frames (1 = refresh all for keyframe)
    part0.put_bit(false); // Update prob tables (0 = use defaults)

    // Macroblock context for adaptive token probabilities
    let mut above_nonzero_y = vec![[false; 4]; mb_cols];
    let mut above_nonzero_u = vec![[false; 2]; mb_cols];
    let mut above_nonzero_v = vec![[false; 2]; mb_cols];

    // Reconstructed working buffers for spatial neighbor prediction
    let mut recon_y = planar.y.clone();
    let mut recon_u = planar.u.clone();
    let mut recon_v = planar.v.clone();

    // Iterate over all macroblocks in raster scan order
    for mb_y in 0..mb_rows {
        let mut left_nonzero_y = [false; 4];
        let mut left_nonzero_u = [false; 2];
        let mut left_nonzero_v = [false; 2];

        for mb_x in 0..mb_cols {
            // Encode macroblock skip flag (0 = do not skip)
            part0.put_bool(false, 128);

            // 1. Evaluate 16x16 Luma Intra-Prediction Mode (Rate-Distortion SAD via SIMD)
            let best_y_mode = select_best_16x16_y_mode(
                &planar.y,
                planar.y_stride,
                &recon_y,
                mb_x,
                mb_y,
            );

            // Encode Y mode into Partition 0
            part0.put_tree(&KF_Y_MODE_TREE, &KF_Y_MODE_PROBS, best_y_mode as usize);

            // 2. Evaluate 8x8 Chroma Intra-Prediction Mode
            let best_uv_mode = select_best_8x8_uv_mode(
                &planar.u,
                &planar.v,
                planar.uv_stride,
                &recon_u,
                &recon_v,
                mb_x,
                mb_y,
            );

            // Encode UV mode into Partition 0
            part0.put_tree(&UV_MODE_TREE, &UV_MODE_PROBS, best_uv_mode as usize);

            // 3. Transform, Quantize, and Encode Luma 16x16 Block
            encode_luma_16x16(
                &planar.y,
                &mut recon_y,
                planar.y_stride,
                mb_x,
                mb_y,
                best_y_mode,
                quant,
                &mut part1,
                &mut above_nonzero_y[mb_x],
                &mut left_nonzero_y,
            );

            // 4. Transform, Quantize, and Encode Chroma U & V Blocks
            encode_chroma_8x8(
                &planar.u,
                &mut recon_u,
                planar.uv_stride,
                mb_x,
                mb_y,
                best_uv_mode,
                quant,
                &mut part1,
                &mut above_nonzero_u[mb_x],
                &mut left_nonzero_u,
            );

            encode_chroma_8x8(
                &planar.v,
                &mut recon_v,
                planar.uv_stride,
                mb_x,
                mb_y,
                best_uv_mode,
                quant,
                &mut part1,
                &mut above_nonzero_v[mb_x],
                &mut left_nonzero_v,
            );

            // In-loop deblocking filter on reconstructed luma boundaries
            if filter_level > 0 {
                let limit = (filter_level as i32) * 2;
                let thresh = filter_level as i32;
                if mb_x > 0 {
                    filter_vertical_edge_16(&mut recon_y, planar.y_stride, mb_x * 16, mb_y * 16, limit, thresh);
                }
                if mb_y > 0 {
                    filter_horizontal_edge_16(&mut recon_y, planar.y_stride, mb_x * 16, mb_y * 16, limit, thresh);
                }
            }
        }
    }

    let p0_bytes = part0.finish();
    let p1_bytes = part1.finish();

    // Assemble 10-byte uncompressed VP8 Keyframe Header:
    // Bit 0: Keyframe flag (0 = keyframe)
    // Bits 1-3: Version (0)
    // Bit 4: Show frame (1 = show)
    // Bits 5-23: Partition 0 size (19 bits)
    let p0_len = p0_bytes.len() as u32;
    let header_tag = (p0_len << 5) | 0x10;

    let mut frame_data = Vec::with_capacity(10 + p0_bytes.len() + p1_bytes.len());
    frame_data.push((header_tag & 0xFF) as u8);
    frame_data.push(((header_tag >> 8) & 0xFF) as u8);
    frame_data.push(((header_tag >> 16) & 0xFF) as u8);

    // Sync code: 0x9D, 0x01, 0x2A
    frame_data.push(0x9D);
    frame_data.push(0x01);
    frame_data.push(0x2A);

    // 16-bit Width & 16-bit Height with 2-bit horizontal/vertical scale (0 = 1:1)
    let w = planar.width as u16;
    let h = planar.height as u16;
    frame_data.push((w & 0xFF) as u8);
    frame_data.push(((w >> 8) & 0x3F) as u8);
    frame_data.push((h & 0xFF) as u8);
    frame_data.push(((h >> 8) & 0x3F) as u8);

    // Append Partitions
    frame_data.extend_from_slice(&p0_bytes);
    frame_data.extend_from_slice(&p1_bytes);

    frame_data
}

/// Selects the best 16x16 Luma prediction mode minimizing Sum of Absolute Differences (SAD).
fn select_best_16x16_y_mode(
    orig: &[u8],
    stride: usize,
    recon: &[u8],
    mb_x: usize,
    mb_y: usize,
) -> Intra16Mode {
    let top = if mb_y > 0 {
        let idx = (mb_y * 16 - 1) * stride + mb_x * 16;
        Some(&recon[idx..idx + 16])
    } else {
        None
    };

    let mut left_buf = [0u8; 16];
    let left = if mb_x > 0 {
        for i in 0..16 {
            left_buf[i] = recon[(mb_y * 16 + i) * stride + mb_x * 16 - 1];
        }
        Some(&left_buf[..])
    } else {
        None
    };

    let top_left = if mb_x > 0 && mb_y > 0 {
        Some(recon[(mb_y * 16 - 1) * stride + mb_x * 16 - 1])
    } else {
        None
    };

    let modes = [
        Intra16Mode::DC,
        Intra16Mode::V,
        Intra16Mode::H,
        Intra16Mode::TM,
    ];

    let mut best_mode = Intra16Mode::DC;
    let mut best_sad = u32::MAX;
    let mut pred_block = [0u8; 256];
    let orig_offset = (mb_y * 16) * stride + mb_x * 16;

    for mode in modes {
        predict_16x16(mode, top, left, top_left, &mut pred_block);
        let sad = sad_16x16(&orig[orig_offset..], stride, &pred_block, 16);

        if sad < best_sad {
            best_sad = sad;
            best_mode = mode;
        }
    }

    best_mode
}

/// Selects the best 8x8 Chroma prediction mode minimizing combined UV SAD.
fn select_best_8x8_uv_mode(
    orig_u: &[u8],
    orig_v: &[u8],
    stride: usize,
    recon_u: &[u8],
    recon_v: &[u8],
    mb_x: usize,
    mb_y: usize,
) -> IntraChromaMode {
    let top_u = if mb_y > 0 {
        let idx = (mb_y * 8 - 1) * stride + mb_x * 8;
        Some(&recon_u[idx..idx + 8])
    } else {
        None
    };
    let top_v = if mb_y > 0 {
        let idx = (mb_y * 8 - 1) * stride + mb_x * 8;
        Some(&recon_v[idx..idx + 8])
    } else {
        None
    };

    let mut left_u_buf = [0u8; 8];
    let mut left_v_buf = [0u8; 8];
    let left_u = if mb_x > 0 {
        for i in 0..8 {
            left_u_buf[i] = recon_u[(mb_y * 8 + i) * stride + mb_x * 8 - 1];
            left_v_buf[i] = recon_v[(mb_y * 8 + i) * stride + mb_x * 8 - 1];
        }
        Some(&left_u_buf[..])
    } else {
        None
    };
    let left_v = if mb_x > 0 { Some(&left_v_buf[..]) } else { None };

    let top_left_u = if mb_x > 0 && mb_y > 0 {
        Some(recon_u[(mb_y * 8 - 1) * stride + mb_x * 8 - 1])
    } else {
        None
    };
    let top_left_v = if mb_x > 0 && mb_y > 0 {
        Some(recon_v[(mb_y * 8 - 1) * stride + mb_x * 8 - 1])
    } else {
        None
    };

    let modes = [
        IntraChromaMode::DC,
        IntraChromaMode::V,
        IntraChromaMode::H,
        IntraChromaMode::TM,
    ];

    let mut best_mode = IntraChromaMode::DC;
    let mut best_sad = u32::MAX;
    let mut pred_u = [0u8; 64];
    let mut pred_v = [0u8; 64];
    let offset = (mb_y * 8) * stride + mb_x * 8;

    for mode in modes {
        predict_8x8(mode, top_u, left_u, top_left_u, &mut pred_u);
        predict_8x8(mode, top_v, left_v, top_left_v, &mut pred_v);

        let sad_u = sad_8x8(&orig_u[offset..], stride, &pred_u, 8);
        let sad_v = sad_8x8(&orig_v[offset..], stride, &pred_v, 8);
        let sad = sad_u + sad_v;

        if sad < best_sad {
            best_sad = sad;
            best_mode = mode;
        }
    }

    best_mode
}

/// Encodes 16 4x4 subblocks of a 16x16 Luma macroblock with Y2 WHT.
fn encode_luma_16x16(
    orig: &[u8],
    recon: &mut [u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    mode: Intra16Mode,
    quant: &Quantizer,
    part1: &mut BoolEncoder,
    above_nonzero: &mut [bool; 4],
    left_nonzero: &mut [bool; 4],
) {
    let top = if mb_y > 0 {
        let idx = (mb_y * 16 - 1) * stride + mb_x * 16;
        Some(&recon[idx..idx + 16])
    } else {
        None
    };

    let mut left_buf = [0u8; 16];
    let left = if mb_x > 0 {
        for i in 0..16 {
            left_buf[i] = recon[(mb_y * 16 + i) * stride + mb_x * 16 - 1];
        }
        Some(&left_buf[..])
    } else {
        None
    };

    let top_left = if mb_x > 0 && mb_y > 0 {
        Some(recon[(mb_y * 16 - 1) * stride + mb_x * 16 - 1])
    } else {
        None
    };

    let mut pred_block = [0u8; 256];
    predict_16x16(mode, top, left, top_left, &mut pred_block);

    // 1. Forward DCT for all 16 subblocks & collect DC coefficients
    let mut raw_dct = [[0i16; 16]; 16];
    let mut dc_coeffs = [0i16; 16];

    for by in 0..4 {
        for bx in 0..4 {
            let b_idx = by * 4 + bx;
            let mut diff = [0i16; 16];

            for y in 0..4 {
                let orig_row = (mb_y * 16 + by * 4 + y) * stride + (mb_x * 16 + bx * 4);
                let pred_row = (by * 4 + y) * 16 + (bx * 4);
                for x in 0..4 {
                    diff[y * 4 + x] = (orig[orig_row + x] as i16) - (pred_block[pred_row + x] as i16);
                }
            }

            fdct_4x4(&diff, &mut raw_dct[b_idx]);
            dc_coeffs[b_idx] = raw_dct[b_idx][0];
        }
    }

    // 2. Walsh-Hadamard Transform on 16 DC coefficients (Y2 block)
    let mut y2_wht = [0i16; 16];
    let mut y2_quant = [0i16; 16];
    let mut y2_dequant = [0i16; 16];
    wht_4x4(&dc_coeffs, &mut y2_wht);

    let _has_y2 = quant.quantize_block(
        &y2_wht,
        quant.y2_dc_quant,
        quant.y2_ac_quant,
        0,
        0,
        &mut y2_quant,
        &mut y2_dequant,
    );

    // Encode Y2 block tokens into Partition 1
    encode_dct_block(part1, &y2_quant, 0, 0, 0);

    // Inverse WHT to reconstruct DC values
    let mut recon_dc = [0i16; 16];
    iwht_4x4(&y2_dequant, &mut recon_dc);

    // 3. Quantize AC coefficients and Reconstruct Luma pixels
    for by in 0..4 {
        for bx in 0..4 {
            let b_idx = by * 4 + bx;
            let mut q_block = [0i16; 16];
            let mut dq_block = [0i16; 16];

            quant.quantize_block(
                &raw_dct[b_idx],
                quant.y1_dc_quant,
                quant.y1_ac_quant,
                1,
                1,
                &mut q_block,
                &mut dq_block,
            );

            // Replace DC with reconstructed Y2 DC
            dq_block[0] = recon_dc[b_idx];

            let ctx = (above_nonzero[bx] as usize) + (left_nonzero[by] as usize);
            let has_ac = encode_dct_block(part1, &q_block, 1, ctx, 1);

            above_nonzero[bx] = has_ac || (q_block[0] != 0);
            left_nonzero[by] = has_ac || (q_block[0] != 0);

            // Reconstruct block into recon buffer
            let dst_offset = (mb_y * 16 + by * 4) * stride + (mb_x * 16 + bx * 4);
            // Copy prediction
            for y in 0..4 {
                let p_idx = (by * 4 + y) * 16 + (bx * 4);
                let r_idx = dst_offset + y * stride;
                recon[r_idx..r_idx + 4].copy_from_slice(&pred_block[p_idx..p_idx + 4]);
            }
            // Add IDCT residuals
            idct_add_4x4(&dq_block, &mut recon[dst_offset..], stride);
        }
    }
}

/// Encodes 4 4x4 subblocks of an 8x8 Chroma macroblock.
fn encode_chroma_8x8(
    orig: &[u8],
    recon: &mut [u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    mode: IntraChromaMode,
    quant: &Quantizer,
    part1: &mut BoolEncoder,
    above_nonzero: &mut [bool; 2],
    left_nonzero: &mut [bool; 2],
) {
    let top = if mb_y > 0 {
        let idx = (mb_y * 8 - 1) * stride + mb_x * 8;
        Some(&recon[idx..idx + 8])
    } else {
        None
    };

    let mut left_buf = [0u8; 8];
    let left = if mb_x > 0 {
        for i in 0..8 {
            left_buf[i] = recon[(mb_y * 8 + i) * stride + mb_x * 8 - 1];
        }
        Some(&left_buf[..])
    } else {
        None
    };

    let top_left = if mb_x > 0 && mb_y > 0 {
        Some(recon[(mb_y * 8 - 1) * stride + mb_x * 8 - 1])
    } else {
        None
    };

    let mut pred_block = [0u8; 64];
    predict_8x8(mode, top, left, top_left, &mut pred_block);

    for by in 0..2 {
        for bx in 0..2 {
            let mut diff = [0i16; 16];
            for y in 0..4 {
                let orig_row = (mb_y * 8 + by * 4 + y) * stride + (mb_x * 8 + bx * 4);
                let pred_row = (by * 4 + y) * 8 + (bx * 4);
                for x in 0..4 {
                    diff[y * 4 + x] = (orig[orig_row + x] as i16) - (pred_block[pred_row + x] as i16);
                }
            }

            let mut raw_dct = [0i16; 16];
            fdct_4x4(&diff, &mut raw_dct);

            let mut q_block = [0i16; 16];
            let mut dq_block = [0i16; 16];
            let has_nonzero = quant.quantize_block(
                &raw_dct,
                quant.uv_dc_quant,
                quant.uv_ac_quant,
                2,
                0,
                &mut q_block,
                &mut dq_block,
            );

            let ctx = (above_nonzero[bx] as usize) + (left_nonzero[by] as usize);
            encode_dct_block(part1, &q_block, 2, ctx, 0);

            above_nonzero[bx] = has_nonzero;
            left_nonzero[by] = has_nonzero;

            // Reconstruct block
            let dst_offset = (mb_y * 8 + by * 4) * stride + (mb_x * 8 + bx * 4);
            for y in 0..4 {
                let p_idx = (by * 4 + y) * 8 + (bx * 4);
                let r_idx = dst_offset + y * stride;
                recon[r_idx..r_idx + 4].copy_from_slice(&pred_block[p_idx..p_idx + 4]);
            }
            idct_add_4x4(&dq_block, &mut recon[dst_offset..], stride);
        }
    }
}

/// Encodes quantized DCT coefficients into VP8 token bitstream.
fn encode_dct_block(
    part1: &mut BoolEncoder,
    coeffs: &[i16; 16],
    plane_type: usize,
    context: usize,
    first_coeff: usize,
) -> bool {
    let mut has_coeffs = false;

    for i in first_coeff..16 {
        let val = coeffs[ZIGZAG[i]];
        if val != 0 {
            has_coeffs = true;
            break;
        }
    }

    if !has_coeffs {
        // Encode EOB (End of Block token)
        let band = COEFF_BANDS[first_coeff];
        let prob = DEFAULT_COEFF_PROBS[plane_type][band][context][0];
        part1.put_bool(false, prob);
        return false;
    }

    let mut current_ctx = context;
    for i in first_coeff..16 {
        let val = coeffs[ZIGZAG[i]];
        let band = COEFF_BANDS[i];

        if val == 0 {
            // Encode zero token
            let p0 = DEFAULT_COEFF_PROBS[plane_type][band][current_ctx][0];
            let p1 = DEFAULT_COEFF_PROBS[plane_type][band][current_ctx][1];
            part1.put_bool(true, p0);
            part1.put_bool(false, p1);
            current_ctx = 0;
        } else {
            // Encode non-zero token
            let p0 = DEFAULT_COEFF_PROBS[plane_type][band][current_ctx][0];
            let p1 = DEFAULT_COEFF_PROBS[plane_type][band][current_ctx][1];
            part1.put_bool(true, p0);
            part1.put_bool(true, p1);

            let abs_val = val.unsigned_abs() as u32;
            let sign = val < 0;

            if abs_val == 1 {
                let p2 = DEFAULT_COEFF_PROBS[plane_type][band][current_ctx][2];
                part1.put_bool(false, p2);
                part1.put_bit(sign);
                current_ctx = 1;
            } else {
                let p2 = DEFAULT_COEFF_PROBS[plane_type][band][current_ctx][2];
                part1.put_bool(true, p2);

                if abs_val <= 4 {
                    let p3 = DEFAULT_COEFF_PROBS[plane_type][band][current_ctx][3];
                    part1.put_bool(false, p3);
                    if abs_val == 2 {
                        let p4 = DEFAULT_COEFF_PROBS[plane_type][band][current_ctx][4];
                        part1.put_bool(false, p4);
                    } else if abs_val == 3 {
                        let p4 = DEFAULT_COEFF_PROBS[plane_type][band][current_ctx][4];
                        let p5 = DEFAULT_COEFF_PROBS[plane_type][band][current_ctx][5];
                        part1.put_bool(true, p4);
                        part1.put_bool(false, p5);
                    } else {
                        let p4 = DEFAULT_COEFF_PROBS[plane_type][band][current_ctx][4];
                        let p5 = DEFAULT_COEFF_PROBS[plane_type][band][current_ctx][5];
                        part1.put_bool(true, p4);
                        part1.put_bool(true, p5);
                    }
                } else {
                    let p3 = DEFAULT_COEFF_PROBS[plane_type][band][current_ctx][3];
                    part1.put_bool(true, p3);
                    // Standard magnitude encoding for VP8 Cat 1-6
                    let extra = (abs_val - 5).min(2047);
                    part1.put_uint(extra, 11);
                }

                part1.put_bit(sign);
                current_ctx = 2;
            }
        }
    }

    true
}
