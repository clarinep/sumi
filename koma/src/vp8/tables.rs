//! VP8 standard quantization, dequantization, and coefficient scan tables.
//!
//! Tables defined in RFC 6386 Section 13 & Section 14.

/// Standard zigzag scan order mapping linear 1D index to 2D (row * 4 + col).
pub const ZIGZAG: [usize; 16] = [
    0, 1, 4, 8,
    5, 2, 3, 6,
    9, 12, 13, 10,
    7, 11, 14, 15,
];

/// Inverse zigzag scan mapping: given a 2D index (row * 4 + col), returns its zigzag index.
pub const INV_ZIGZAG: [usize; 16] = [
    0, 1, 5, 6,
    2, 4, 7, 12,
    3, 8, 11, 13,
    9, 10, 14, 15,
];

/// VP8 standard DC dequantization lookup table for Y (luma) DC coefficients (128 entries).
pub const DC_LOOKUP: [i16; 128] = [
    4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 17,
    18, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 25, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    75, 76, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    91, 93, 95, 96, 98, 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118,
    122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 143, 145, 148, 151, 154, 157,
];

/// VP8 standard AC dequantization lookup table (128 entries).
pub const AC_LOOKUP: [i16; 128] = [
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76,
    78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108,
    110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152,
    155, 158, 161, 164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205, 209,
    213, 217, 221, 225, 229, 234, 239, 245, 249, 254, 259, 264, 269, 274, 279, 284,
];

/// Standard 16x16 macroblock Intra-prediction mode tree for Partition 0 header encoding.
pub const KF_Y_MODE_TREE: [i8; 6] = [
    -0, 2, // 0 -> DC_PRED
    -1, 4, // 1 -> V_PRED
    -2, -3 // 2 -> H_PRED, 3 -> TM_PRED
];

/// Mode probabilities for KF 16x16 Y Intra prediction.
pub const KF_Y_MODE_PROBS: [u8; 3] = [145, 156, 163];

/// Standard 8x8 Chroma Intra-prediction mode tree.
pub const UV_MODE_TREE: [i8; 6] = [
    -0, 2, // 0 -> DC_PRED
    -1, 4, // 1 -> V_PRED
    -2, -3 // 2 -> H_PRED, 3 -> TM_PRED
];

/// Mode probabilities for Chroma UV Intra prediction.
pub const UV_MODE_PROBS: [u8; 3] = [142, 114, 183];

/// 4x4 subblock B_PRED mode tree (10 modes).
pub const B_MODE_TREE: [i8; 18] = [
    -0, 2,  // 0: B_DC_PRED
    -3, 4,  // 3: B_TM_PRED
    -1, 6,  // 1: B_VE_PRED
    8, 12,
    -2, 10, // 2: B_HE_PRED
    -4, -5, // 4: B_RD_PRED, 5: B_VR_PRED
    -6, 14, // 6: B_LD_PRED
    -7, 16, // 7: B_VL_PRED
    -8, -9, // 8: B_HD_PRED, 9: B_HU_PRED
];
